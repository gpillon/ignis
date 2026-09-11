//! Tool-call XML parsing over the content channel (GitHub #121): the chat
//! template's tool-call dialect —
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=PARAM>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! (`.qwen/tmp/chat_template.jinja`; the same format the system prompt
//! instructs the model to reply in when tools are offered) — is scanned out
//! of the `Channel::Content` text [`crate::decoder::OutputDecoder`] already
//! produced, split from plain content, and reassembled into the OpenAI
//! `tool_calls` wire shape.
//!
//! **Boundary robustness.** The scanner buffers everything from the moment
//! it sees the opening `<tool_call>` tag until the matching `</tool_call>`,
//! so a tag split across arbitrary chunk boundaries can never be missed —
//! matching only needs to run against the *whole* buffered span, not track
//! partial-tag state token by token. A call is emitted as a single complete
//! event (id + name + JSON-encoded arguments) only once its closing tag is
//! seen and its body parses; several calls in one response each get the
//! next `index` in the order they closed, so indices are stable and never
//! merge two calls together.
//!
//! **An interrupted or malformed call is dropped whole, never leaked
//! half-written.** A stop/cancel landing before `</tool_call>` closes, or a
//! closed block whose body does not parse as `<function=...>`, is silently
//! discarded at the point it is discovered — never surfaced as partial XML
//! in `content` and never surfaced as a tool call with truncated or
//! garbled arguments (GitHub #121 acceptance criterion 3). The caller
//! learns generation stopped from `finish_reason` (`stop`/`length`); it is
//! never told a tool call happened when the call did not actually
//! complete.
//!
//! **Argument typing.** Each `<parameter>` body is raw text; the value is
//! `serde_json`-parsed so a model that writes `42`, `true`, `["a","b"]` or
//! `{"x":1}` round-trips as that JSON type, and anything that fails to
//! parse as JSON (ordinary prose, a bare path) falls back to a JSON string
//! containing exactly that text. A single leading and single trailing
//! newline immediately inside the `<parameter=...>` / `</parameter>` tags
//! is trimmed (the template's own formatting whitespace); interior
//! newlines — deliberately preserved by the system prompt's "can span
//! multiple lines" contract — are not touched.

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";

/// One fully parsed, complete tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Position among the calls emitted by this scanner (0, 1, 2, ...) —
    /// the OpenAI streaming `tool_calls[].index` a client keys its
    /// reassembly buffer on.
    pub index: usize,
    pub id: String,
    pub name: String,
    /// A JSON-encoded object (`serde_json::to_string` of the parsed
    /// parameters), e.g. `{"path":"a.txt","recursive":true}`.
    pub arguments: String,
}

/// One event out of the scanner: ordinary content text (tool-call markup
/// already stripped out), or a complete tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    Content(String),
    Call(ToolCall),
}

enum State {
    /// Scanning for `<tool_call>`; text that cannot still become that tag
    /// streams straight out as `ToolEvent::Content`.
    Scanning,
    /// Between `<tool_call>` and its matching `</tool_call>` — buffering,
    /// nothing streams out until the block resolves one way or the other.
    Buffering,
}

/// Scans a request's whole content-channel text for tool-call blocks,
/// across as many `feed` calls as the text arrives in.
pub struct ToolCallScanner {
    state: State,
    hold: String,
    next_index: usize,
}

impl Default for ToolCallScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallScanner {
    pub fn new() -> Self {
        Self {
            state: State::Scanning,
            hold: String::new(),
            next_index: 0,
        }
    }

    /// True while a `<tool_call>` has been opened but not yet closed —
    /// checked by the caller *before* [`finish`](Self::finish) to tell a
    /// clean end (last thing emitted was a complete call, or plain
    /// content) from a truncated one (acceptance criterion 3).
    pub fn is_mid_call(&self) -> bool {
        matches!(self.state, State::Buffering)
    }

    /// Whether this scanner has ever emitted a complete call.
    pub fn any_calls(&self) -> bool {
        self.next_index > 0
    }

    /// Feed newly available content text; returns the events it makes
    /// available (zero, one, or more).
    pub fn feed(&mut self, text: &str) -> Vec<ToolEvent> {
        let mut out = Vec::new();
        self.hold.push_str(text);
        loop {
            match self.state {
                State::Scanning => {
                    if let Some(idx) = self.hold.find(OPEN_TAG) {
                        if idx > 0 {
                            out.push(ToolEvent::Content(self.hold[..idx].to_string()));
                        }
                        self.hold.drain(..idx + OPEN_TAG.len());
                        self.state = State::Buffering;
                        continue;
                    }
                    let holdback = tag_prefix_holdback(&self.hold, OPEN_TAG);
                    let safe_len = self.hold.len() - holdback;
                    if safe_len == 0 {
                        break;
                    }
                    let safe: String = self.hold.drain(..safe_len).collect();
                    out.push(ToolEvent::Content(safe));
                    break;
                }
                State::Buffering => {
                    let Some(idx) = self.hold.find(CLOSE_TAG) else {
                        break; // still incomplete; wait for more text
                    };
                    let body: String = self.hold.drain(..idx + CLOSE_TAG.len()).collect();
                    let inner = &body[..body.len() - CLOSE_TAG.len()];
                    if let Some((name, arguments)) = parse_function_block(inner) {
                        let index = self.next_index;
                        self.next_index += 1;
                        out.push(ToolEvent::Call(ToolCall {
                            index,
                            id: format!("call_{index}"),
                            name,
                            arguments,
                        }));
                    }
                    // A closed block that does not parse as a well-formed
                    // `<function=...>` is dropped silently — same
                    // never-leak-garbage rule as an interrupted one.
                    self.state = State::Scanning;
                    continue;
                }
            }
        }
        out
    }

    /// Flush whatever is left at the end of generation.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        match self.state {
            State::Scanning => {
                let text = std::mem::take(&mut self.hold);
                if text.is_empty() {
                    vec![]
                } else {
                    vec![ToolEvent::Content(text)]
                }
            }
            State::Buffering => {
                // Truncated mid-call: drop the whole buffered span, raw
                // XML included — never leaked as content, never surfaced
                // as a call.
                self.hold.clear();
                vec![]
            }
        }
    }
}

/// The length of the trailing suffix of `text` that is a proper (non-empty,
/// non-full) prefix of `tag` — the same holdback shape as
/// `decoder::holdback_len`, generalized to any literal tag.
fn tag_prefix_holdback(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| text.ends_with(&tag[..len]))
        .unwrap_or(0)
}

/// Parses `<function=NAME>...<parameter=P>VALUE</parameter>...</function>`
/// (the text between `<tool_call>` and `</tool_call>`, both already
/// stripped) into `(name, json_arguments)`. Returns `None` if the block
/// does not open with `<function=` or never closes with `</function>`.
fn parse_function_block(inner: &str) -> Option<(String, String)> {
    let after_prefix = inner.trim_start().strip_prefix(FUNCTION_PREFIX)?;
    let name_end = after_prefix.find('>')?;
    let name = after_prefix[..name_end].to_string();
    let body = after_prefix[name_end + 1..].find(FUNCTION_CLOSE).map(|end| &after_prefix[name_end + 1..name_end + 1 + end])?;

    let mut args = serde_json::Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAMETER_PREFIX) {
        let after = &rest[start + PARAMETER_PREFIX.len()..];
        let Some(name_end) = after.find('>') else {
            break; // malformed trailing parameter tag: stop, keep what we have
        };
        let param_name = after[..name_end].to_string();
        let after_name = &after[name_end + 1..];
        let Some(value_end) = after_name.find(PARAMETER_CLOSE) else {
            break; // an unclosed parameter: stop, keep what we have
        };
        let raw_value = &after_name[..value_end];
        let value = trim_one_newline(raw_value);
        args.insert(param_name, parse_argument_value(value));
        rest = &after_name[value_end + PARAMETER_CLOSE.len()..];
    }
    let arguments = serde_json::to_string(&serde_json::Value::Object(args))
        .expect("a Map<String, Value> always serializes");
    Some((name, arguments))
}

/// Trims exactly one leading and one trailing `\n` (the template's own
/// formatting newlines around a parameter value), never more — interior
/// and any *extra* surrounding newlines are the model's own content.
fn trim_one_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

/// A parameter's JSON type: valid JSON (a number, bool, array, object, or
/// an explicitly quoted string) round-trips as that type; anything else —
/// ordinary text — becomes a JSON string of exactly that text.
fn parse_argument_value(value: &str) -> serde_json::Value {
    serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&str]) -> Vec<ToolEvent> {
        let mut scanner = ToolCallScanner::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(scanner.feed(chunk));
        }
        events.extend(scanner.finish());
        events
    }

    fn calls(events: &[ToolEvent]) -> Vec<&ToolCall> {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Call(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    fn content(events: &[ToolEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolEvent::Content(s) => Some(s.as_str()),
                _ => None,
            })
            .collect()
    }

    fn args(call: &ToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).expect("valid JSON arguments")
    }

    #[test]
    fn plain_content_with_no_tool_call_passes_through_unchanged() {
        let events = run(&["hello ", "world"]);
        assert_eq!(content(&events), "hello world");
        assert!(calls(&events).is_empty());
    }

    #[test]
    fn a_single_tool_call_in_one_chunk_is_parsed() {
        let events = run(&[
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(args(calls[0]), serde_json::json!({"path": "a.txt"}));
    }

    #[test]
    fn text_before_the_call_is_preserved_as_content() {
        let events = run(&[
            "Let me check that.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>",
        ]);
        assert_eq!(content(&events), "Let me check that.\n\n");
        assert_eq!(calls(&events).len(), 1);
    }

    #[test]
    fn a_tool_call_split_across_many_chunk_boundaries_still_reassembles() {
        let events = run(&[
            "before <tool_",
            "call>\n<function=read",
            "_file>\n<param",
            "eter=path>\na.txt\n</parameter>\n</functio",
            "n>\n</tool_call> after",
        ]);
        assert_eq!(content(&events), "before  after");
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(args(calls[0]), serde_json::json!({"path": "a.txt"}));
    }

    #[test]
    fn the_open_tag_split_one_byte_at_a_time_is_still_found() {
        let chunks: Vec<String> = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
            .chars()
            .map(|c| c.to_string())
            .collect();
        let refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
        let events = run(&refs);
        let calls = calls(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
    }

    #[test]
    fn several_calls_in_one_response_get_distinct_stable_indices() {
        let events = run(&[
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[1].index, 1);
        assert_eq!(calls[1].name, "b");
        assert_ne!(calls[0].id, calls[1].id);
    }

    #[test]
    fn multiple_parameters_all_land_in_arguments() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(
            args(calls[0]),
            serde_json::json!({"path": "a.txt", "content": "hello"})
        );
    }

    #[test]
    fn a_multiline_parameter_value_keeps_its_interior_newlines() {
        let events = run(&[
            "<tool_call>\n<function=write_file>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
        ]);
        let calls = calls(&events);
        assert_eq!(
            args(calls[0])["content"],
            serde_json::json!("line one\nline two")
        );
    }

    #[test]
    fn numeric_and_boolean_and_array_arguments_round_trip_typed() {
        let events = run(&[
            "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=arr>\n[\"a\",\"b\"]\n</parameter>\n</function>\n</tool_call>",
        ]);
        let a = args(calls(&events)[0]);
        assert_eq!(a["n"], serde_json::json!(42));
        assert_eq!(a["b"], serde_json::json!(true));
        assert_eq!(a["arr"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn a_call_interrupted_before_it_closes_is_dropped_not_leaked() {
        // A stop/cancel landing mid-call (acceptance criterion 3): the
        // open tag and everything after it must never surface as content
        // or as a malformed call.
        let events = run(&["some text <tool_call>\n<function=read_file>\n<parameter=path>\na"]);
        assert_eq!(content(&events), "some text ");
        assert!(calls(&events).is_empty());
        for e in &events {
            if let ToolEvent::Content(text) = e {
                assert!(!text.contains('<'), "{text:?}");
            }
        }
    }

    #[test]
    fn a_malformed_block_that_never_opens_a_function_is_dropped() {
        let events = run(&["<tool_call>garbage, no function here</tool_call>after"]);
        assert!(calls(&events).is_empty());
        assert_eq!(content(&events), "after");
    }

    #[test]
    fn is_mid_call_reports_true_only_while_a_call_is_open() {
        let mut scanner = ToolCallScanner::new();
        assert!(!scanner.is_mid_call());
        scanner.feed("<tool_call>\n<function=f>\n");
        assert!(scanner.is_mid_call());
        scanner.feed("</function>\n</tool_call>");
        assert!(!scanner.is_mid_call());
    }

    #[test]
    fn any_calls_is_false_until_the_first_call_closes() {
        let mut scanner = ToolCallScanner::new();
        assert!(!scanner.any_calls());
        scanner.feed("<tool_call>\n<function=f>\n</function>\n</tool_call>");
        assert!(scanner.any_calls());
    }

    #[test]
    fn streaming_and_whole_string_feeds_agree_by_construction() {
        let whole = "hi <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call> bye";
        let streamed = run(&whole
            .as_bytes()
            .chunks(3)
            .map(|b| std::str::from_utf8(b).unwrap_or_default())
            .collect::<Vec<_>>());
        let batched = run(&[whole]);
        assert_eq!(content(&streamed), content(&batched));
        assert_eq!(calls(&streamed).len(), calls(&batched).len());
        assert_eq!(calls(&streamed)[0].arguments, calls(&batched)[0].arguments);
    }
}
