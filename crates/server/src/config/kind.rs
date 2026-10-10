//! The per-type half of a config field (spec config-v2/01 §The field
//! declaration): how a value of one kind is read from the text an operator
//! typed, and written back out.
//!
//! The field macro (`schema.rs`) names a kind per field and calls it; it
//! never parses anything itself. So a flag, its env var and its config-file
//! key are read by the one [`FieldKind::parse`] — a value from a file is
//! turned back into the same text first ([`FieldKind::file_text`]) — and
//! each kind's grammar is tested here once, independent of which fields use
//! it.
//!
//! A kind's error text continues a sentence the resolver starts with the
//! spelling the value came from: "`--reuse-kv-host-pool-bytes` expects a
//! byte count, got `lots`", or the same with `IGNIS_REUSE_KV_HOST_POOL_BYTES`
//! or `reuse.kv_host_pool_bytes` in front — whichever the operator wrote.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::PathBuf;

use ignis_core::ngram_cache::CacheLocation;
use ignis_core::ngram_table::HotBudget;
use ignis_core::{KvFormat, KvPoolSize, ProposalHead, RopeScaling, SpeculativeBackend};
use serde_json::Value;

use super::{ApiKey, ApiKeySetting, Expose, SpecChoice};
use crate::instruction::{DeveloperMessagePolicy, SystemMessagePolicy};
use crate::thinking::ReasoningEffort;

/// One kind of config value: its Rust type, its grammar, its written form.
pub trait FieldKind {
    /// What the resolved settings hold for a field of this kind.
    type Value: Clone + PartialEq + std::fmt::Debug;
    /// The kind's name in `help --fields`.
    const TAG: &'static str;
    /// A bool: the flag may stand bare, meaning `true`.
    const SWITCH: bool = false;
    /// The flag may be given more than once, each occurrence one more entry
    /// (`;`-joined, the env var's own separator).
    const REPEATABLE: bool = false;

    /// Parse the operator's text (trimmed, never empty: an empty value is an
    /// unset one, and never reaches a kind).
    fn parse(raw: &str) -> Result<Self::Value, String>;

    /// The value as a config file (and `GET /v1/config`) writes it — the
    /// spelling an operator would type, so a written file reads back to the
    /// same value through [`FieldKind::parse`].
    fn render(value: &Self::Value) -> Value;

    /// The value as a number, for [`super::field::Validator::Range`] and
    /// [`super::field::Validator::MultipleOf`]; `None` for a kind no numeric
    /// rule applies to.
    fn number(_value: &Self::Value) -> Option<i64> {
        None
    }

    /// A value read from a config file as the text [`FieldKind::parse`]
    /// takes; `Ok(None)` for `null`, which is unset. Scalars only, unless a
    /// kind takes a list or a map.
    fn file_text(value: &Value) -> Result<Option<String>, String> {
        scalar_text(value)
    }
}

/// The default [`FieldKind::file_text`]: a string, a number or a bool as its
/// text; `null` as unset; a list or a map refused naming the problem.
pub fn scalar_text(value: &Value) -> Result<Option<String>, String> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(b) => Ok(Some(b.to_string())),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::String(s) => Ok(Some(s.clone())),
        Value::Array(_) | Value::Object(_) => Err("expects a single value, not a list or a map".to_owned()),
    }
}

/// `true`/`false`, `on`/`off` or `1`/`0`, any case.
pub struct Bool;

impl FieldKind for Bool {
    type Value = bool;
    const TAG: &'static str = "bool";
    const SWITCH: bool = true;

    fn parse(raw: &str) -> Result<bool, String> {
        match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Ok(true),
            "0" | "false" | "off" => Ok(false),
            _ => Err(format!("must be true or false (on/off and 1/0 are read too), got `{raw}`")),
        }
    }

    fn render(value: &bool) -> Value {
        Value::Bool(*value)
    }
}

/// Free text, kept as typed.
pub struct Text;

impl FieldKind for Text {
    type Value = String;
    const TAG: &'static str = "text";

    fn parse(raw: &str) -> Result<String, String> {
        Ok(raw.to_owned())
    }

    fn render(value: &String) -> Value {
        Value::String(value.clone())
    }
}

/// A filesystem path, kept as typed.
pub struct Path;

impl FieldKind for Path {
    type Value = PathBuf;
    const TAG: &'static str = "path";

    fn parse(raw: &str) -> Result<PathBuf, String> {
        Ok(PathBuf::from(raw))
    }

    fn render(value: &PathBuf) -> Value {
        Value::String(value.display().to_string())
    }
}

/// An unsigned count of something — the kinds differ only in their tag and
/// the unit their error names ("expects a token count").
macro_rules! count_kind {
    ($(#[$doc:meta])* $kind:ident: $int:ty, $tag:literal, $unit:literal) => {
        $(#[$doc])*
        pub struct $kind;

        impl FieldKind for $kind {
            type Value = $int;
            const TAG: &'static str = $tag;

            fn parse(raw: &str) -> Result<$int, String> {
                raw.parse::<$int>().map_err(|_| format!(concat!("expects a ", $unit, ", got `{}`"), raw))
            }

            fn render(value: &$int) -> Value {
                Value::from(*value)
            }

            fn number(value: &$int) -> Option<i64> {
                i64::try_from(*value).ok().or(Some(i64::MAX))
            }
        }
    };
}

count_kind!(
    /// A count of tokens.
    Tokens: u32, "tokens", "token count"
);
count_kind!(
    /// A count of seconds.
    Secs: u32, "seconds", "second count"
);
count_kind!(
    /// A count of retained slots.
    Slots: u32, "slots", "slot count"
);
count_kind!(
    /// A whole percent.
    Percent: u32, "percent", "percent"
);
count_kind!(
    /// A count of decode lanes.
    Lanes: u32, "lanes", "lane count"
);
count_kind!(
    /// A count of mebibytes.
    Mib: u64, "MiB", "MiB count"
);

/// A byte count with the size suffixes an operator actually types: a bare
/// count, or one followed by `K`/`M`/`G` (case-insensitive, binary — `4G`
/// is 4 GiB), optionally spelled `KiB`/`MiB`/`GiB` or `KB`/`MB`/`GB`. A pool
/// budget is naturally a number of gibibytes, and making the operator write
/// 4294967296 invites the typo that silently starts a server with a tenth of
/// the pool it meant.
pub struct Bytes;

/// [`Bytes`]' grammar, for the kinds that take a byte count among other
/// spellings.
pub fn parse_byte_count(raw: &str) -> Option<u64> {
    let digits_end = raw.find(|c: char| !c.is_ascii_digit()).unwrap_or(raw.len());
    let (digits, suffix) = raw.split_at(digits_end);
    if digits.is_empty() {
        return None;
    }
    let multiplier: u64 = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        _ => return None,
    };
    digits.parse::<u64>().ok()?.checked_mul(multiplier)
}

/// A byte count as an operator writes it: the largest of `G`/`M`/`K` that
/// divides it (`1536M`, not 1610612736), or the bare count.
pub fn render_byte_count(bytes: u64) -> Value {
    match [(30, "G"), (20, "M"), (10, "K")].into_iter().find(|&(shift, _)| bytes != 0 && bytes % (1 << shift) == 0) {
        Some((shift, unit)) => Value::String(format!("{}{unit}", bytes >> shift)),
        None => Value::from(bytes),
    }
}

impl FieldKind for Bytes {
    type Value = u64;
    const TAG: &'static str = "bytes";

    fn parse(raw: &str) -> Result<u64, String> {
        parse_byte_count(raw).ok_or_else(|| format!("expects a byte count (a K/M/G suffix is read), got `{raw}`"))
    }

    fn render(value: &u64) -> Value {
        render_byte_count(*value)
    }

    fn number(value: &u64) -> Option<i64> {
        i64::try_from(*value).ok().or(Some(i64::MAX))
    }
}

/// A field that may be left unset: `None` is its default, and `null` in a
/// file. Everything else is the inner kind's.
pub struct Opt<K>(PhantomData<K>);

impl<K: FieldKind> FieldKind for Opt<K> {
    type Value = Option<K::Value>;
    const TAG: &'static str = K::TAG;
    const SWITCH: bool = K::SWITCH;
    const REPEATABLE: bool = K::REPEATABLE;

    fn parse(raw: &str) -> Result<Self::Value, String> {
        K::parse(raw).map(Some)
    }

    fn render(value: &Self::Value) -> Value {
        value.as_ref().map_or(Value::Null, K::render)
    }

    fn number(value: &Self::Value) -> Option<i64> {
        value.as_ref().and_then(K::number)
    }

    fn file_text(value: &Value) -> Result<Option<String>, String> {
        K::file_text(value)
    }
}

/// The KV pool (ADR 0045): a byte count as [`Bytes`] reads one, or a token
/// count — `<n>tok`, `<n>Ktok`, `<n>Mtok`, the multipliers binary
/// (`512Ktok` is 524,288 tokens). Only parsed: what a byte count is worth in
/// tokens depends on the model, so the load's plan judges it.
pub struct KvPool;

impl FieldKind for KvPool {
    type Value = KvPoolSize;
    const TAG: &'static str = "bytes|tokens";

    fn parse(raw: &str) -> Result<KvPoolSize, String> {
        let lower = raw.to_ascii_lowercase();
        let Some(count) = lower.strip_suffix("tok") else {
            return parse_byte_count(raw)
                .map(KvPoolSize::Bytes)
                .ok_or_else(|| format!("expects a byte count or <n>[K|M]tok, got `{raw}`"));
        };
        let (digits, multiplier) = match count.as_bytes().last() {
            Some(b'k') => (&count[..count.len() - 1], 1 << 10),
            Some(b'm') => (&count[..count.len() - 1], 1 << 20),
            _ => (count, 1),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("expects a byte count or <n>[K|M]tok, got `{raw}`"));
        }
        digits
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(multiplier))
            .map(KvPoolSize::Tokens)
            .ok_or_else(|| format!("expects a byte count or <n>[K|M]tok, got `{raw}`"))
    }

    fn render(value: &KvPoolSize) -> Value {
        match *value {
            KvPoolSize::Bytes(bytes) => render_byte_count(bytes),
            KvPoolSize::Tokens(tokens) => Value::String(
                match [(20, "M"), (10, "K")].into_iter().find(|&(shift, _)| tokens != 0 && tokens % (1 << shift) == 0) {
                    Some((shift, unit)) => format!("{}{unit}tok", tokens >> shift),
                    None => format!("{tokens}tok"),
                },
            ),
        }
    }
}

/// The KV storage format (ADR 0022): `bf16` or `hq-e8-2b`.
pub struct KvFormatKind;

impl FieldKind for KvFormatKind {
    type Value = KvFormat;
    const TAG: &'static str = "kv format";

    fn parse(raw: &str) -> Result<KvFormat, String> {
        KvFormat::parse(raw)
    }

    fn render(value: &KvFormat) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// The thinking budget: a whole number of reasoning tokens, at least 1, or
/// `off` for none. `0` is refused rather than read as `off`: the one way to
/// say "none" at startup is the word.
pub struct ThinkingBudget;

impl FieldKind for ThinkingBudget {
    type Value = Option<u32>;
    const TAG: &'static str = "tokens|off";

    fn parse(raw: &str) -> Result<Option<u32>, String> {
        if raw == "off" {
            return Ok(None);
        }
        match raw.parse::<u32>() {
            Ok(n) if n >= 1 => Ok(Some(n)),
            _ => Err(format!(
                "must be a whole number of tokens, at least 1, or `off` for no budget, got `{raw}`"
            )),
        }
    }

    fn render(value: &Option<u32>) -> Value {
        value.map_or_else(|| Value::String("off".to_owned()), Value::from)
    }
}

/// A `reasoning_effort` default, in the protocol's vocabulary.
pub struct Effort;

impl FieldKind for Effort {
    type Value = ReasoningEffort;
    const TAG: &'static str = "reasoning effort";

    fn parse(raw: &str) -> Result<ReasoningEffort, String> {
        ReasoningEffort::parse(&raw.to_ascii_lowercase()).ok_or_else(|| {
            let accepted: Vec<&str> = ReasoningEffort::ALL.iter().map(|e| e.as_str()).collect();
            format!("must be one of {}, got `{raw}`", accepted.join(", "))
        })
    }

    fn render(value: &ReasoningEffort) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// The text rotary table (GitHub #227), in the reference's grammar: `none`
/// or `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]`.
pub struct Rope;

impl FieldKind for Rope {
    type Value = RopeScaling;
    const TAG: &'static str = "rope scaling";

    fn parse(raw: &str) -> Result<RopeScaling, String> {
        RopeScaling::parse(raw).map_err(|e| e.to_string())
    }

    fn render(value: &RopeScaling) -> Value {
        Value::String(value.to_string())
    }
}

/// Where `system` messages go (GitHub #209).
pub struct SystemPolicy;

impl FieldKind for SystemPolicy {
    type Value = SystemMessagePolicy;
    const TAG: &'static str = "policy";

    fn parse(raw: &str) -> Result<SystemMessagePolicy, String> {
        let wanted = raw.to_ascii_lowercase();
        SystemMessagePolicy::ALL
            .into_iter()
            .find(|p| p.as_str() == wanted)
            .ok_or_else(|| format!("is not a system message policy: `{raw}`"))
    }

    fn render(value: &SystemMessagePolicy) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// Where `developer` messages go (GitHub #209).
pub struct DeveloperPolicy;

impl FieldKind for DeveloperPolicy {
    type Value = DeveloperMessagePolicy;
    const TAG: &'static str = "policy";

    fn parse(raw: &str) -> Result<DeveloperMessagePolicy, String> {
        let wanted = raw.to_ascii_lowercase();
        DeveloperMessagePolicy::ALL
            .into_iter()
            .find(|p| p.as_str() == wanted)
            .ok_or_else(|| format!("is not a developer message policy: `{raw}`"))
    }

    fn render(value: &DeveloperMessagePolicy) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// How the server is exposed beyond its bind address (ADR 0028).
pub struct ExposeKind;

impl FieldKind for ExposeKind {
    type Value = Expose;
    const TAG: &'static str = "expose mode";

    fn parse(raw: &str) -> Result<Expose, String> {
        Expose::parse(raw)
    }

    fn render(value: &Expose) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// The API key: the operator's own, or `auto` for one generated at start.
/// Written back as typed — the one place that happens is a file the operator
/// asked to have written; `GET /v1/config` never shows this kind's field.
pub struct ApiKeyKind;

impl FieldKind for ApiKeyKind {
    type Value = ApiKeySetting;
    const TAG: &'static str = "key|auto";

    fn parse(raw: &str) -> Result<ApiKeySetting, String> {
        Ok(match raw {
            "auto" => ApiKeySetting::Generate,
            key => ApiKeySetting::Fixed(ApiKey::new(key)),
        })
    }

    fn render(value: &ApiKeySetting) -> Value {
        Value::String(match value {
            ApiKeySetting::Generate => "auto".to_owned(),
            ApiKeySetting::Fixed(key) => key.as_str().to_owned(),
        })
    }
}

/// A credential the server sends to someone else (`download.token`, spec
/// model-download/02), kept as typed in an [`ApiKey`], whose `Debug` never
/// shows it. Unlike [`ApiKeyKind`] no word is special: `auto` is a token
/// like any other. Written back as typed — the one place that happens is a
/// file the operator asked to have written; `GET /v1/config` and `config
/// print` never show this kind's field.
pub struct Token;

impl FieldKind for Token {
    type Value = ApiKey;
    const TAG: &'static str = "token";

    fn parse(raw: &str) -> Result<ApiKey, String> {
        Ok(ApiKey::new(raw))
    }

    fn render(value: &ApiKey) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// The speculative backend (P5-02, GitHub #150, #307): `dflash2` (the 27B's
/// drafter), `mtp` (Flash-Next's head), or `off`.
pub struct SpecBackend;

impl FieldKind for SpecBackend {
    type Value = SpecChoice;
    const TAG: &'static str = "backend";

    fn parse(raw: &str) -> Result<SpecChoice, String> {
        if raw.eq_ignore_ascii_case("off") {
            return Ok(SpecChoice::Off);
        }
        SpeculativeBackend::parse(&raw.to_ascii_lowercase()).map(SpecChoice::Backend)
    }

    fn render(value: &SpecChoice) -> Value {
        Value::String(match value {
            SpecChoice::Off => "off".to_owned(),
            SpecChoice::Backend(backend) => backend.as_str().to_owned(),
        })
    }
}

/// The drafter's proposal head: `full` or `shortlist`.
pub struct DraftHead;

impl FieldKind for DraftHead {
    type Value = ProposalHead;
    const TAG: &'static str = "head";

    fn parse(raw: &str) -> Result<ProposalHead, String> {
        ProposalHead::parse(&raw.to_ascii_lowercase())
    }

    fn render(value: &ProposalHead) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// Flash-Next's draft row budget (GitHub #307): `0` (the decode route's
/// rows) or `2..=`[`ignis_core::speculation::FLASH_NEXT_VERIFY_ROWS`]. One
/// row is refused, not a range end: a round needs a row for the token it
/// verifies and one for a draft.
pub struct DraftRows;

impl FieldKind for DraftRows {
    type Value = u32;
    const TAG: &'static str = "rows";

    fn parse(raw: &str) -> Result<u32, String> {
        let max = ignis_core::speculation::FLASH_NEXT_VERIFY_ROWS;
        raw.parse::<u32>()
            .ok()
            .filter(|&rows| rows <= max && rows != 1)
            .ok_or_else(|| format!("must be 0 or in 2..={max}, got `{raw}`"))
    }

    fn render(value: &u32) -> Value {
        Value::from(*value)
    }

    fn number(value: &u32) -> Option<i64> {
        Some(i64::from(*value))
    }
}

/// Flash-Next's n-gram hot-row budget (GitHub #306): a byte count, or
/// `auto` for what the host plan leaves.
pub struct HotBytes;

impl FieldKind for HotBytes {
    type Value = HotBudget;
    const TAG: &'static str = "bytes|auto";

    fn parse(raw: &str) -> Result<HotBudget, String> {
        if raw == "auto" {
            return Ok(HotBudget::Auto);
        }
        parse_byte_count(raw)
            .map(HotBudget::Bytes)
            .ok_or_else(|| format!("expects a byte count or auto, got `{raw}`"))
    }

    fn render(value: &HotBudget) -> Value {
        Value::String(value.to_string())
    }
}

/// Where a cache's files go: `model` (beside the artifact), `auto` (the
/// per-user cache directory), or a directory named.
pub struct Location;

impl FieldKind for Location {
    type Value = CacheLocation;
    const TAG: &'static str = "model|auto|dir";

    fn parse(raw: &str) -> Result<CacheLocation, String> {
        Ok(match raw {
            "model" => CacheLocation::Model,
            "auto" => CacheLocation::Auto,
            dir => CacheLocation::Directory(PathBuf::from(dir)),
        })
    }

    fn render(value: &CacheLocation) -> Value {
        Value::String(match value {
            CacheLocation::Model => "model".to_owned(),
            CacheLocation::Auto => "auto".to_owned(),
            CacheLocation::Directory(dir) => dir.display().to_string(),
        })
    }
}

/// The models a request may switch to by naming them (spec model-switch/01
/// §Implicit switch): `<id>=<path>` pairs, `;`-separated — `;` because a
/// Windows path may hold a `,`. A flag names one pair per occurrence; a file
/// may give a map of id to path, a list of pairs, or the joined string.
///
/// Each pair splits at its first `=`. An id may not carry the `@` that
/// starts a lane tag: a request's `model` loses its tag before it is looked
/// up, so no request could name it. An id named twice is refused, not
/// last-wins: two paths for one id leave nothing to say which was meant.
pub struct KnownModels;

impl FieldKind for KnownModels {
    type Value = BTreeMap<String, PathBuf>;
    const TAG: &'static str = "id=path;...";
    const REPEATABLE: bool = true;

    fn parse(raw: &str) -> Result<Self::Value, String> {
        let mut known = BTreeMap::new();
        for entry in raw.split(';').filter(|entry| !entry.trim().is_empty()) {
            let Some((id, path)) = entry.split_once('=') else {
                return Err(format!("takes `<id>=<path>` pairs, got `{entry}`"));
            };
            let (id, path) = (id.trim(), path.trim());
            if id.is_empty() || path.is_empty() {
                let missing = if id.is_empty() { "id" } else { "path" };
                return Err(format!("entry `{entry}` has no {missing}"));
            }
            if id.contains('@') {
                return Err(format!(
                    "id `{id}` holds an `@`, which starts a request's lane tag: no request could name it"
                ));
            }
            if known.insert(id.to_owned(), PathBuf::from(path)).is_some() {
                return Err(format!("names `{id}` twice"));
            }
        }
        Ok(known)
    }

    fn render(value: &Self::Value) -> Value {
        Value::Object(
            value.iter().map(|(id, path)| (id.clone(), Value::String(path.display().to_string()))).collect(),
        )
    }

    fn file_text(value: &Value) -> Result<Option<String>, String> {
        let pair = |id: &str, path: &Value| match path {
            Value::String(path) => Ok(format!("{id}={path}")),
            _ => Err(format!("maps `{id}` to something that is not a path")),
        };
        match value {
            Value::Object(map) => {
                Ok(Some(map.iter().map(|(id, path)| pair(id, path)).collect::<Result<Vec<_>, _>>()?.join(";")))
            }
            Value::Array(items) => Ok(Some(
                items
                    .iter()
                    .map(|item| item.as_str().map(str::to_owned).ok_or_else(|| "lists something that is not an `<id>=<path>` pair".to_owned()))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(";"),
            )),
            other => scalar_text(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A written value reads back to itself: the round trip a generated
    /// config file depends on, for every kind.
    fn round_trips<K: FieldKind>(value: K::Value) {
        let written = K::render(&value);
        let text = K::file_text(&written).expect("a written value is readable").expect("not null");
        assert_eq!(K::parse(&text).expect("parses back"), value, "{written}");
    }

    #[test]
    fn bools_take_every_spelling_and_refuse_the_rest() {
        for (raw, want) in [("true", true), ("ON", true), ("1", true), ("false", false), ("Off", false), ("0", false)] {
            assert_eq!(Bool::parse(raw), Ok(want), "{raw}");
        }
        let err = Bool::parse("yes").unwrap_err();
        assert!(err.contains("`yes`") && err.contains("true or false"), "{err}");
        round_trips::<Bool>(false);
    }

    #[test]
    fn counts_name_their_unit() {
        assert_eq!(Tokens::parse("4096"), Ok(4096));
        assert_eq!(Tokens::parse("4k").unwrap_err(), "expects a token count, got `4k`");
        assert_eq!(Secs::parse("-1").unwrap_err(), "expects a second count, got `-1`");
        assert_eq!(Tokens::number(&7), Some(7));
        round_trips::<Slots>(16);
    }

    #[test]
    fn byte_counts_take_binary_suffixes_and_write_the_largest_that_divides() {
        for (raw, bytes) in [("0", 0), ("4096", 4096), ("4K", 4 << 10), ("512m", 512 << 20), ("2G", 2 << 30), ("6144MiB", 6144 << 20), ("1gb", 1 << 30)] {
            assert_eq!(Bytes::parse(raw), Ok(bytes), "{raw}");
        }
        for raw in ["", "G", "4TB", "-1", "4 GiB please", "1.5G"] {
            assert!(Bytes::parse(raw).is_err(), "{raw}");
        }
        assert_eq!(Bytes::render(&(1536 << 20)), Value::String("1536M".into()));
        assert_eq!(Bytes::render(&(2 << 30)), Value::String("2G".into()));
        assert_eq!(Bytes::render(&1000), Value::from(1000u64));
        assert_eq!(Bytes::render(&0), Value::from(0u64));
        round_trips::<Bytes>(1536 << 20);
        round_trips::<Bytes>(12345);
    }

    #[test]
    fn a_kv_pool_is_bytes_or_tokens() {
        assert_eq!(KvPool::parse("8G"), Ok(KvPoolSize::Bytes(8 << 30)));
        for (raw, tokens) in [("512Ktok", 524_288), ("2Mtok", 2 << 20), ("1000tok", 1000), ("512ktok", 524_288)] {
            assert_eq!(KvPool::parse(raw), Ok(KvPoolSize::Tokens(tokens)), "{raw}");
        }
        for raw in ["tok", "Ktok", "4Gtok", "1.5Ktok", "-1tok", "12 tok", "4TB"] {
            assert!(KvPool::parse(raw).unwrap_err().contains("tok"), "{raw}");
        }
        round_trips::<KvPool>(KvPoolSize::Tokens(524_288));
        round_trips::<KvPool>(KvPoolSize::Tokens(1000));
        round_trips::<KvPool>(KvPoolSize::Bytes(6 << 30));
    }

    #[test]
    fn an_optional_kind_is_null_when_unset_and_the_inner_kind_otherwise() {
        assert_eq!(<Opt<Bytes>>::render(&None), Value::Null);
        assert_eq!(<Opt<Bytes>>::file_text(&Value::Null), Ok(None));
        assert_eq!(<Opt<Bytes>>::parse("1K"), Ok(Some(1024)));
        assert_eq!(<Opt<Bytes>>::number(&Some(5)), Some(5));
        round_trips::<Opt<Bytes>>(Some(4 << 30));
    }

    #[test]
    fn a_file_value_must_be_a_scalar_unless_the_kind_takes_more() {
        assert_eq!(scalar_text(&Value::from(262_144)), Ok(Some("262144".into())));
        assert_eq!(scalar_text(&Value::Bool(true)), Ok(Some("true".into())));
        assert!(scalar_text(&serde_json::json!([1, 2])).unwrap_err().contains("single value"));
        assert!(scalar_text(&serde_json::json!({"a": 1})).is_err());
    }

    #[test]
    fn the_thinking_budget_is_a_count_or_off_and_never_zero() {
        assert_eq!(ThinkingBudget::parse("off"), Ok(None));
        assert_eq!(ThinkingBudget::parse("12288"), Ok(Some(12288)));
        for bad in ["0", "-1", "lots", "8k", "OFF"] {
            assert!(ThinkingBudget::parse(bad).unwrap_err().contains("off"), "{bad}");
        }
        round_trips::<ThinkingBudget>(None);
        round_trips::<ThinkingBudget>(Some(32_768));
    }

    #[test]
    fn the_enumerated_kinds_parse_any_case_and_write_their_canonical_name() {
        assert_eq!(KvFormatKind::parse("hq"), Ok(KvFormat::HqE8_2b));
        assert_eq!(Effort::parse("LOW"), Ok(ReasoningEffort::Low));
        assert!(Effort::parse("nonsense").unwrap_err().contains("xhigh"));
        assert_eq!(SystemPolicy::parse("Strict"), Ok(SystemMessagePolicy::Strict));
        assert_eq!(DeveloperPolicy::parse("One-After-System"), Ok(DeveloperMessagePolicy::OneAfterSystem));
        assert_eq!(SpecBackend::parse("off"), Ok(SpecChoice::Off));
        assert_eq!(SpecBackend::parse("mtp"), Ok(SpecChoice::Backend(SpeculativeBackend::Mtp)));
        assert!(SpecBackend::parse("eagle").unwrap_err().contains("eagle"));
        assert_eq!(DraftHead::parse("shortlist"), Ok(ProposalHead::Shortlist));
        assert_eq!(ExposeKind::parse("cloudflare-quick"), Ok(Expose::CloudflareQuick));
        round_trips::<KvFormatKind>(KvFormat::Bf16);
        round_trips::<SpecBackend>(SpecChoice::Off);
        round_trips::<DeveloperPolicy>(DeveloperMessagePolicy::IntoSystem);
        round_trips::<Rope>(RopeScaling::NONE);
        round_trips::<Rope>(RopeScaling::parse("yarn:4,t=0.25").unwrap());
    }

    #[test]
    fn draft_rows_are_zero_or_two_to_the_verify_rows() {
        assert_eq!(DraftRows::parse("0"), Ok(0));
        assert_eq!(DraftRows::parse("6"), Ok(6));
        for bad in ["1", "9", "rows"] {
            let err = DraftRows::parse(bad).unwrap_err();
            assert!(err.contains("2..=8") && err.contains(bad), "{err}");
        }
    }

    #[test]
    fn hot_bytes_and_locations_take_their_words() {
        assert_eq!(HotBytes::parse("auto"), Ok(HotBudget::Auto));
        assert_eq!(HotBytes::parse("512M"), Ok(HotBudget::Bytes(512 << 20)));
        assert!(HotBytes::parse("auto2").unwrap_err().contains("auto"));
        assert_eq!(Location::parse("model"), Ok(CacheLocation::Model));
        assert_eq!(Location::parse("auto"), Ok(CacheLocation::Auto));
        assert_eq!(Location::parse("D:/kv"), Ok(CacheLocation::Directory(PathBuf::from("D:/kv"))));
        round_trips::<HotBytes>(HotBudget::Bytes(4 << 30));
        round_trips::<Location>(CacheLocation::Directory(PathBuf::from("custom")));
    }

    #[test]
    fn the_api_key_is_a_key_or_auto_and_debug_never_shows_it() {
        assert_eq!(ApiKeyKind::parse("auto"), Ok(ApiKeySetting::Generate));
        let fixed = ApiKeyKind::parse("sk-secret").unwrap();
        assert!(!format!("{fixed:?}").contains("sk-secret"));
        round_trips::<ApiKeyKind>(fixed);
    }

    #[test]
    fn a_token_is_kept_as_typed_and_debug_never_shows_it() {
        let token = Token::parse("hf_secret").unwrap();
        assert_eq!(token.as_str(), "hf_secret");
        assert!(!format!("{token:?}").contains("hf_secret"));
        assert_eq!(Token::parse("auto").unwrap().as_str(), "auto", "no word is special");
        round_trips::<Token>(token);
    }

    #[test]
    fn known_models_split_pairs_and_refuse_bad_ones() {
        let known = KnownModels::parse("qwen3.8-27b=C:/Program Files, x86/27B.ninfer; odd=D:/a=b/m.ninfer;").unwrap();
        assert_eq!(known["qwen3.8-27b"], PathBuf::from("C:/Program Files, x86/27B.ninfer"));
        assert_eq!(known["odd"], PathBuf::from("D:/a=b/m.ninfer"), "split at the first `=` only");
        for (entry, says) in [("qwen3.8-27b", "<id>=<path>"), ("=F:/a", "id"), ("m=", "path"), ("m@agent=F:/a", "@"), ("m=F:/a;m=F:/b", "twice")] {
            assert!(KnownModels::parse(entry).unwrap_err().contains(says), "{entry}");
        }
        // A file may give a map or a list as well as the joined string.
        let map = serde_json::json!({"a": "F:/a.ninfer", "b": "F:/b.ninfer"});
        assert_eq!(KnownModels::file_text(&map), Ok(Some("a=F:/a.ninfer;b=F:/b.ninfer".into())));
        let list = serde_json::json!(["a=F:/a.ninfer"]);
        assert_eq!(KnownModels::file_text(&list), Ok(Some("a=F:/a.ninfer".into())));
        assert!(KnownModels::file_text(&serde_json::json!({"a": 1})).is_err());
        round_trips::<KnownModels>(known);
    }
}
