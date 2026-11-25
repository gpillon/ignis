//! An engine-local, single-use grant for an exact continuation of a streamed
//! read. The ordinary tool-call ID is its unpredictable handle. Only a
//! successful engine completion can record the corresponding grant.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ignis_core::{RequestId, scheduler::VerifiedWarmParent};
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::template::{ChatMessage, MessageContent};
use crate::toolcall::ToolCall;

const TTL: Duration = Duration::from_secs(30);
const MAX_GRANTS: usize = 64;
const MAX_HISTORY_BYTES: usize = 128 * 1024;
const ID_PREFIX: &str = "call_read_";
const ID_BYTES: usize = 16;

fn new_call_id() -> Option<String> {
    use std::fmt::Write as _;
    let mut bytes = [0; ID_BYTES];
    getrandom::fill(&mut bytes).ok()?;
    let mut id = String::with_capacity(ID_PREFIX.len() + 2 * ID_BYTES);
    id.push_str(ID_PREFIX);
    for byte in bytes {
        let _ = write!(id, "{byte:02x}");
    }
    Some(id)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct History([u8; 32]);

impl History {
    fn of(messages: &[ChatMessage]) -> Option<Self> {
        if messages.last()?.role != "user"
            || messages.iter().any(|m| match m.content {
                MessageContent::Text(_) => false,
                MessageContent::Null => m.role != "assistant",
                MessageContent::Parts(_) => true,
            })
        {
            return None;
        }
        struct BoundedHash(Sha256, usize);
        impl Write for BoundedHash {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.1 = self
                    .1
                    .checked_sub(bytes.len())
                    .ok_or(io::ErrorKind::InvalidData)?;
                self.0.update(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut digest = BoundedHash(Sha256::new(), MAX_HISTORY_BYTES);
        serde_json::to_writer(&mut digest, messages).ok()?;
        Some(Self(digest.0.finalize().into()))
    }
}

type Args = BTreeMap<String, Value>;

fn read_args(raw: &str) -> Option<Args> {
    // The pilot's read tool uses a flat object, e.g. filePath, offset, limit.
    // Reject duplicates and floats, rather than treating them as equivalent.
    struct FlatArgs;
    impl<'de> Visitor<'de> for FlatArgs {
        type Value = Args;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a flat read argument object")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Args, M::Error> {
            let mut args = Args::new();
            while let Some((key, value)) = map.next_entry::<String, Value>()? {
                if !(value.is_string()
                    || value.is_i64()
                    || value.is_u64()
                    || value.is_boolean()
                    || value.is_null())
                    || args.insert(key, value).is_some()
                {
                    return Err(de::Error::custom("unsupported read arguments"));
                }
            }
            Ok(args)
        }
    }
    if raw.len() > 8192 {
        return None;
    }
    let mut json = serde_json::Deserializer::from_str(raw);
    let args = (&mut json).deserialize_map(FlatArgs).ok()?;
    json.end().ok()?;
    Some(args)
}

#[derive(PartialEq, Eq)]
struct ReadCall {
    id: String,
    args: Args,
}

pub(crate) struct Pending {
    history: History,
    call: Option<ReadCall>,
}

impl Pending {
    pub(crate) fn new(messages: &[ChatMessage]) -> Option<Self> {
        Some(Self {
            history: History::of(messages)?,
            call: None,
        })
    }

    pub(crate) fn observe_call(&mut self, call: &mut ToolCall) -> bool {
        if self.call.is_some()
            || call.cut
            || call.index != 0
            || call.name != "read"
        {
            return false;
        }
        let Some(args) = read_args(&call.arguments) else {
            return false;
        };
        let Some(id) = new_call_id() else { return false };
        call.id.clone_from(&id);
        self.call = Some(ReadCall {
            id,
            args,
        });
        true
    }
}

struct Grant {
    parent: RequestId,
    history: History,
    args: Args,
    expires: Instant,
}

#[derive(Default)]
pub(crate) struct Grants(Mutex<HashMap<String, Grant>>);

impl Grants {
    pub(crate) fn issue_on_success(&self, pending: Pending, parent: RequestId, now: Instant) {
        let Some(call) = pending.call else { return };
        let Ok(mut grants) = self.0.lock() else {
            return;
        };
        grants.retain(|_, g| g.expires > now);
        if grants.len() < MAX_GRANTS {
            grants.entry(call.id).or_insert(Grant {
                parent,
                history: pending.history,
                args: call.args,
                expires: now + TTL,
            });
        }
    }

    pub(crate) fn consume(
        &self,
        messages: &[ChatMessage],
        clock: impl FnOnce() -> Instant,
    ) -> Option<VerifiedWarmParent> {
        let (result, prefix) = messages.split_last()?;
        let (assistant, history) = prefix.split_last()?;
        let empty_assistant = match &assistant.content {
            MessageContent::Null => true,
            MessageContent::Text(text) => text.is_empty(),
            MessageContent::Parts(_) => false,
        };
        if assistant.role != "assistant"
            || assistant.reasoning_content.is_some()
            || assistant.tool_call_id.is_some()
            || !empty_assistant
            || result.role != "tool"
            || result.reasoning_content.is_some()
            || result.tool_calls.is_some()
            || !matches!(&result.content, MessageContent::Text(text) if !text.is_empty())
        {
            return None;
        }
        let [call] = assistant.tool_calls.as_deref()? else {
            return None;
        };
        let id = call.id.as_deref()?;
        if id.len() != ID_PREFIX.len() + 2 * ID_BYTES
            || !id.starts_with(ID_PREFIX)
            || call.function.name != "read"
            || result.tool_call_id.as_deref() != Some(id)
        {
            return None;
        }
        let history = History::of(history)?;
        let args = read_args(&call.function.arguments)?;
        let mut grants = self.0.lock().ok()?;
        let grant = grants.get(id)?;
        if grant.expires <= clock() || grant.history != history || grant.args != args {
            return None;
        }
        let parent = grants.remove(id)?.parent;
        Some(VerifiedWarmParent::from_server_grant(parent))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn messages(id: &str) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
        let parent = vec![ChatMessage::text("user", "Read the file.")];
        let mut child = parent.clone();
        let suffix: Vec<ChatMessage> = serde_json::from_value(json!([
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": id, "type": "function", "function": {"name": "read", "arguments": "{ \"filePath\": \"a.txt\" }"}}
            ]},
            {"role": "tool", "tool_call_id": id, "content": "answer"}
        ])).unwrap();
        child.extend(suffix);
        (parent, child)
    }

    fn read_call(cut: bool) -> ToolCall {
        ToolCall {
            index: 0,
            id: "call_0".into(),
            name: "read".into(),
            arguments: r#"{"filePath":"a.txt"}"#.into(),
            cut,
        }
    }

    fn issue(table: &Grants, now: Instant) -> String {
        let mut pending = Pending::new(&messages("call_0").0).unwrap();
        let mut call = read_call(false);
        assert!(pending.observe_call(&mut call));
        table.issue_on_success(pending, 11, now);
        call.id
    }

    #[test]
    fn cut_call_is_never_proof_even_with_parseable_arguments() {
        let (parent, child) = messages("call_0");
        let mut pending = Pending::new(&parent).unwrap();
        assert!(!pending.observe_call(&mut read_call(true)));
        let grants = Grants::default();
        let now = Instant::now();
        grants.issue_on_success(pending, 11, now);
        assert!(grants.consume(&child, || now).is_none());
    }

    #[test]
    fn identical_calls_do_not_share_a_handle() {
        let table = Grants::default();
        let now = Instant::now();
        let first = issue(&table, now);
        let second = issue(&table, now);
        assert_ne!(first, second);
        assert!(table.consume(&messages(&first).1, || now).is_some());
        assert!(table.consume(&messages(&second).1, || now).is_some());
    }

    #[test]
    fn exact_child_redeems_once_and_bad_attempts_do_not_spend_the_grant() {
        let table = Grants::default();
        let now = Instant::now();
        let id = issue(&table, now);
        let (_, child) = messages(&id);
        let now = || now;
        let mut changed = child.clone();
        changed[0] = ChatMessage::text("user", "Read a different file.");
        assert_eq!(table.consume(&changed, now), None);
        let mut changed = child.clone();
        changed[2].tool_call_id = Some("call_0".into());
        assert_eq!(table.consume(&changed, now), None);
        let mut changed = child.clone();
        changed[1].tool_calls.as_mut().unwrap()[0]
            .function
            .arguments = r#"{"filePath":"b.txt","filePath":"a.txt"}"#.into();
        assert_eq!(table.consume(&changed, now), None);
        assert_eq!(table.consume(&messages("call_1").1, now), None);
        assert_eq!(
            table.consume(&child, now),
            Some(VerifiedWarmParent::from_server_grant(11))
        );
        assert_eq!(table.consume(&child, now), None);
    }

    #[test]
    fn grant_expires_at_ttl_and_cannot_cross_engine_tables() {
        let table = Grants::default();
        let now = Instant::now();
        let child = messages(&issue(&table, now)).1;
        assert_eq!(
            Grants::default().consume(&child, || now),
            None
        );
        assert_eq!(
            table.consume(&child, || now + TTL),
            None
        );
    }

    #[test]
    fn concurrent_consumers_share_one_grant() {
        let table = Grants::default();
        let now = Instant::now();
        let child = messages(&issue(&table, now)).1;
        let barrier = std::sync::Barrier::new(2);
        let run = || {
            barrier.wait();
            usize::from(table.consume(&child, || now).is_some())
        };
        let wins = std::thread::scope(|s| {
            let other = s.spawn(&run);
            run() + other.join().unwrap()
        });
        assert_eq!(wins, 1);
    }
}
