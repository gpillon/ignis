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

use ignis_core::image::ImageSize;
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::ngram_table::HotBudget;
use ignis_core::{ExceptionWeights, KvFormat, KvPoolSize, PrefillAttention, ProposalHeadSetting, RopeSetting, SpeculativeBackend, VisionResidency};
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
    /// The field may be unset ([`Opt`]): `GET /v1/config/fields` says so, so
    /// a page can show "auto" for a `null` value.
    const NULLABLE: bool = false;

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
count_kind!(
    /// A count of image jobs.
    Jobs: u32, "jobs", "job count"
);
count_kind!(
    /// A count of images.
    Images: u32, "images", "image count"
);
count_kind!(
    /// A length in pixels.
    Pixels: u32, "pixels", "pixel count"
);
count_kind!(
    /// An area in units of 1,048,576 pixels, a 1024x1024 image each.
    Mipx: u32, "Mipx", "Mi-pixel count"
);
count_kind!(
    /// An image's area in latent cells, 16 x 16 pixels each.
    Cells: u32, "cells", "latent-cell count"
);
count_kind!(
    /// A count of characters.
    Chars: u32, "chars", "character count"
);
count_kind!(
    /// Thousandths of one.
    Permille: u32, "permille", "count of thousandths"
);
count_kind!(
    /// A count of retries.
    Retries: u32, "retries", "retry count"
);
count_kind!(
    /// A count of checks in a row.
    Checks: u32, "checks", "check count"
);

count_kind!(
    /// A count of milliseconds.
    Millis: u32, "ms", "millisecond count"
);
count_kind!(
    /// A count of open Realtime sessions.
    Sessions: u32, "sessions", "session count"
);
count_kind!(
    /// A count of live-transcript refreshes.
    Partials: u32, "partials", "partial count"
);
count_kind!(
    /// A count of words.
    Words: u32, "words", "word count"
);
count_kind!(
    /// A whole multiple of something.
    Times: u32, "times", "whole multiple"
);
count_kind!(
    /// A count of waiting requests.
    Requests: u32, "requests", "request count"
);
count_kind!(
    /// A count of the diarizer's 80 ms frames.
    Frames: u32, "frames", "frame count"
);

/// The speech model's compute threads (spec speech/01): a count, or `auto`,
/// the physical cores less one. Rendered `auto` when unset, so a file written
/// from the running config reads back to the same value.
pub struct Threads;

impl FieldKind for Threads {
    type Value = Option<u32>;
    const TAG: &'static str = "count|auto";

    fn parse(raw: &str) -> Result<Self::Value, String> {
        if raw.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        raw.parse::<u32>().ok().filter(|&threads| (1..=1024).contains(&threads)).map(Some).ok_or_else(|| format!("expects `auto` or a thread count in 1..=1024, got `{raw}`"))
    }

    fn render(value: &Self::Value) -> Value {
        value.map_or_else(|| Value::String("auto".to_owned()), Value::from)
    }
}

/// A fraction of one, `0` to `1` (`0.5`), kept as thousandths so a group of
/// them stays `Eq`.
pub struct Fraction;

impl FieldKind for Fraction {
    type Value = u32;
    const TAG: &'static str = "0..1";

    fn parse(raw: &str) -> Result<u32, String> {
        raw.parse::<f64>()
            .ok()
            .filter(|fraction| (0.0..=1.0).contains(fraction))
            .map(|fraction| (fraction * 1000.0).round() as u32)
            .ok_or_else(|| format!("expects a number from 0 to 1, got `{raw}`"))
    }

    fn render(value: &u32) -> Value {
        Value::from(f64::from(*value) / 1000.0)
    }
}

/// A decimal number, at least 0 (`6.5`), kept as thousandths so a group of
/// them stays `Eq`: a third decimal place is the finest it keeps. A
/// [`super::field::Validator::DecimalRange`] bounds it, in the same
/// thousandths.
pub struct Decimal;

impl Decimal {
    /// `raw` in thousandths, or why not.
    fn thousandths(raw: &str) -> Option<u32> {
        raw.parse::<f64>().ok().filter(|value| value.is_finite() && (0.0..=f64::from(u32::MAX) / 1000.0).contains(value)).map(|value| (value * 1000.0).round() as u32)
    }

    fn written(value: u32) -> Value {
        Value::from(f64::from(value) / 1000.0)
    }
}

impl FieldKind for Decimal {
    type Value = u32;
    const TAG: &'static str = "number";

    fn parse(raw: &str) -> Result<u32, String> {
        Decimal::thousandths(raw).ok_or_else(|| format!("expects a number of at least 0, got `{raw}`"))
    }

    fn render(value: &u32) -> Value {
        Decimal::written(*value)
    }

    fn number(value: &u32) -> Option<i64> {
        Some(i64::from(*value))
    }
}

/// A decimal number as [`Decimal`] reads it, or `auto`: the model's own
/// value, which its weights carry. Rendered `auto` when unset.
pub struct AutoDecimal;

impl FieldKind for AutoDecimal {
    type Value = Option<u32>;
    const TAG: &'static str = "number|auto";

    fn parse(raw: &str) -> Result<Self::Value, String> {
        if raw.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        Decimal::thousandths(raw).map(Some).ok_or_else(|| format!("expects `auto` or a number of at least 0, got `{raw}`"))
    }

    fn render(value: &Self::Value) -> Value {
        value.map_or_else(|| Value::String("auto".to_owned()), Decimal::written)
    }

    fn number(value: &Self::Value) -> Option<i64> {
        value.map(i64::from)
    }
}

/// A count, or `auto`: the model's own value, which its weights carry.
/// Rendered `auto` when unset.
pub struct AutoCount;

impl FieldKind for AutoCount {
    type Value = Option<u32>;
    const TAG: &'static str = "count|auto";

    fn parse(raw: &str) -> Result<Self::Value, String> {
        if raw.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        raw.parse::<u32>().map(Some).map_err(|_| format!("expects `auto` or a count, got `{raw}`"))
    }

    fn render(value: &Self::Value) -> Value {
        value.map_or_else(|| Value::String("auto".to_owned()), Value::from)
    }

    fn number(value: &Self::Value) -> Option<i64> {
        value.map(i64::from)
    }
}

/// A list of model names, `;`- or `,`-separated: `speech.compat_names`. An
/// entry holds no whitespace and no `@` (which starts a request's lane tag, as
/// in [`KnownModels`]) and is named once.
pub struct Names;

impl FieldKind for Names {
    type Value = Vec<String>;
    const TAG: &'static str = "name;...";
    const REPEATABLE: bool = true;

    fn parse(raw: &str) -> Result<Self::Value, String> {
        let mut names: Vec<String> = Vec::new();
        for name in raw.split([';', ',']).map(str::trim).filter(|name| !name.is_empty()) {
            if name.contains(char::is_whitespace) || name.contains('@') {
                return Err(format!("takes names with no whitespace and no `@`, got `{name}`"));
            }
            if names.iter().any(|seen| seen == name) {
                return Err(format!("names `{name}` twice"));
            }
            names.push(name.to_owned());
        }
        Ok(names)
    }

    fn render(value: &Self::Value) -> Value {
        Value::Array(value.iter().cloned().map(Value::String).collect())
    }

    fn file_text(value: &Value) -> Result<Option<String>, String> {
        match value {
            Value::Array(items) => Ok(Some(
                items
                    .iter()
                    .map(|item| item.as_str().map(str::to_owned).ok_or_else(|| "lists something that is not a name".to_owned()))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(";"),
            )),
            other => scalar_text(other),
        }
    }
}

/// The loop guard's mode (`crate::loop_guard`): `off`, `log` or `rewind`.
pub struct LoopGuard;

impl FieldKind for LoopGuard {
    type Value = crate::loop_guard::LoopGuardMode;
    const TAG: &'static str = "loop guard mode";

    fn parse(raw: &str) -> Result<crate::loop_guard::LoopGuardMode, String> {
        crate::loop_guard::LoopGuardMode::parse(&raw.to_ascii_lowercase()).ok_or_else(|| {
            format!("must be one of {}, got `{raw}`", crate::loop_guard::LoopGuardMode::NAMES.join(", "))
        })
    }

    fn render(value: &crate::loop_guard::LoopGuardMode) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

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
    const NULLABLE: bool = true;

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

/// The 27B's prompt attention precision (GitHub #335): `fp8` or `fp16`.
pub struct PrefillAttentionKind;

impl FieldKind for PrefillAttentionKind {
    type Value = PrefillAttention;
    const TAG: &'static str = "prefill attention";

    fn parse(raw: &str) -> Result<PrefillAttention, String> {
        PrefillAttention::parse(raw)
    }

    fn render(value: &PrefillAttention) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// How the 27B holds its BF16 exception tensors (GitHub #341): `bf16` or `fp8`.
pub struct ExceptionWeightsKind;

impl FieldKind for ExceptionWeightsKind {
    type Value = ExceptionWeights;
    const TAG: &'static str = "exception weights";

    fn parse(raw: &str) -> Result<ExceptionWeights, String> {
        ExceptionWeights::parse(raw)
    }

    fn render(value: &ExceptionWeights) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// Where a vision load keeps the 27B's vision tower (ADR 0075, GitHub
/// #414): `host` or `device`.
pub struct VisionResidencyKind;

impl FieldKind for VisionResidencyKind {
    type Value = VisionResidency;
    const TAG: &'static str = "vision residency";

    fn parse(raw: &str) -> Result<VisionResidency, String> {
        VisionResidency::parse(raw)
    }

    fn render(value: &VisionResidency) -> Value {
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
/// or `yarn:F[,t=<c>][,bf=<n>][,bs=<n>]`, or `auto`, which the context
/// decides at load.
pub struct Rope;

impl FieldKind for Rope {
    type Value = RopeSetting;
    const TAG: &'static str = "rope scaling";

    fn parse(raw: &str) -> Result<RopeSetting, String> {
        RopeSetting::parse(raw).map_err(|e| e.to_string())
    }

    fn render(value: &RopeSetting) -> Value {
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

/// What a request past its deadline is answered (ADR 0058).
pub struct TimeoutReplyKind;

impl FieldKind for TimeoutReplyKind {
    type Value = crate::deadline::TimeoutReply;
    const TAG: &'static str = "reply";

    fn parse(raw: &str) -> Result<crate::deadline::TimeoutReply, String> {
        let wanted = raw.to_ascii_lowercase();
        crate::deadline::TimeoutReply::ALL.into_iter().find(|r| r.as_str() == wanted).ok_or_else(|| {
            let accepted: Vec<&str> = crate::deadline::TimeoutReply::ALL.iter().map(|r| r.as_str()).collect();
            format!("must be one of {}, got `{raw}`", accepted.join(", "))
        })
    }

    fn render(value: &crate::deadline::TimeoutReply) -> Value {
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
/// like any other. Written back as typed, in clear text, wherever the operator
/// asked for a config to be written: a file, but also `config generate` to
/// stdout and `--dry-run`, as for `server.api_key`; `GET /v1/config` and
/// `config print` never show this kind's field.
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

/// The drafter's proposal head: `full`, `shortlist` or `auto` (ADR 0063),
/// kept as given; the load's speculation resolves `auto`.
pub struct DraftHead;

impl FieldKind for DraftHead {
    type Value = ProposalHeadSetting;
    const TAG: &'static str = "head";

    fn parse(raw: &str) -> Result<ProposalHeadSetting, String> {
        ProposalHeadSetting::parse(&raw.to_ascii_lowercase())
    }

    fn render(value: &ProposalHeadSetting) -> Value {
        Value::String(value.as_str().to_owned())
    }
}

/// Flash-Next's draft row budget (GitHub #307, #329): `0` (the default's
/// [`ignis_core::speculation::FLASH_NEXT_DEFAULT_VERIFY_ROWS`]) or
/// `2..=`[`ignis_core::speculation::FLASH_NEXT_MAX_VERIFY_ROWS`]. One row is
/// refused, not a range end: a round needs a row for the token it verifies
/// and one for a draft.
pub struct DraftRows;

impl FieldKind for DraftRows {
    type Value = u32;
    const TAG: &'static str = "rows";

    fn parse(raw: &str) -> Result<u32, String> {
        let max = ignis_core::speculation::FLASH_NEXT_MAX_VERIFY_ROWS;
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

/// The image sizes a request may ask for (spec image-model/05 §Sizes):
/// `<width>x<height>` entries, `;`- or `,`-separated — a flag names one or
/// more per occurrence, a file may give a list. Each side is a nonzero
/// multiple of 32, the VAE's 16 pixels a latent cell times the DiT's 2 × 2
/// patch, and at most [`crate::config::MAX_IMAGE_SIDE`]; at least one size,
/// and none twice.
pub struct Sizes;

impl FieldKind for Sizes {
    type Value = Vec<ImageSize>;
    const TAG: &'static str = "WxH;...";
    const REPEATABLE: bool = true;

    fn parse(raw: &str) -> Result<Self::Value, String> {
        let mut sizes: Vec<ImageSize> = Vec::new();
        for entry in raw.split([';', ',']).map(str::trim).filter(|entry| !entry.is_empty()) {
            let size = ImageSize::parse(entry)?;
            if size.width % 32 != 0 || size.height % 32 != 0 {
                return Err(format!("takes sizes whose sides are multiples of 32, got `{entry}`"));
            }
            if size.width.max(size.height) > crate::config::MAX_IMAGE_SIDE {
                return Err(format!("takes sides of at most {} pixels, got `{entry}`", crate::config::MAX_IMAGE_SIDE));
            }
            if sizes.contains(&size) {
                return Err(format!("names `{entry}` twice"));
            }
            sizes.push(size);
        }
        if sizes.is_empty() {
            return Err(format!("expects at least one <width>x<height>, got `{raw}`"));
        }
        Ok(sizes)
    }

    fn render(value: &Self::Value) -> Value {
        Value::Array(value.iter().map(|size| Value::String(size.to_string())).collect())
    }

    fn file_text(value: &Value) -> Result<Option<String>, String> {
        match value {
            Value::Array(items) => Ok(Some(
                items
                    .iter()
                    .map(|item| item.as_str().map(str::to_owned).ok_or_else(|| "lists something that is not a `<width>x<height>`".to_owned()))
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
    use ignis_core::RopeScaling;

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
        assert_eq!(PrefillAttentionKind::parse("FP16"), Ok(PrefillAttention::Fp16));
        assert!(PrefillAttentionKind::parse("fp4").unwrap_err().contains("fp8"));
        assert_eq!(ExceptionWeightsKind::parse("FP8"), Ok(ExceptionWeights::Fp8));
        assert!(ExceptionWeightsKind::parse("nvfp4").unwrap_err().contains("bf16"));
        assert_eq!(Effort::parse("LOW"), Ok(ReasoningEffort::Low));
        assert!(Effort::parse("nonsense").unwrap_err().contains("xhigh"));
        assert_eq!(SystemPolicy::parse("Strict"), Ok(SystemMessagePolicy::Strict));
        assert_eq!(DeveloperPolicy::parse("One-After-System"), Ok(DeveloperMessagePolicy::OneAfterSystem));
        assert_eq!(SpecBackend::parse("off"), Ok(SpecChoice::Off));
        assert_eq!(SpecBackend::parse("mtp"), Ok(SpecChoice::Backend(SpeculativeBackend::Mtp)));
        assert!(SpecBackend::parse("eagle").unwrap_err().contains("eagle"));
        assert_eq!(DraftHead::parse("shortlist"), Ok(ProposalHeadSetting::Fixed(ignis_core::ProposalHead::Shortlist)));
        assert_eq!(DraftHead::parse("Auto"), Ok(ProposalHeadSetting::Auto));
        assert_eq!(ExposeKind::parse("cloudflare-quick"), Ok(Expose::CloudflareQuick));
        assert_eq!(TimeoutReplyKind::parse("Partial"), Ok(crate::deadline::TimeoutReply::Partial));
        round_trips::<KvFormatKind>(KvFormat::Bf16);
        round_trips::<PrefillAttentionKind>(PrefillAttention::Fp16);
        round_trips::<ExceptionWeightsKind>(ExceptionWeights::Fp8);
        round_trips::<SpecBackend>(SpecChoice::Off);
        round_trips::<DeveloperPolicy>(DeveloperMessagePolicy::IntoSystem);
        round_trips::<TimeoutReplyKind>(crate::deadline::TimeoutReply::Error);
        round_trips::<Rope>(RopeSetting::Auto);
        round_trips::<DraftHead>(ProposalHeadSetting::Auto);
        round_trips::<DraftHead>(ProposalHeadSetting::Fixed(ignis_core::ProposalHead::Shortlist));
        round_trips::<Rope>(RopeSetting::Fixed(RopeScaling::NONE));
        round_trips::<Rope>(RopeSetting::parse("yarn:4,t=0.25").unwrap());
    }

    #[test]
    fn draft_rows_are_zero_or_two_to_the_verify_rows() {
        assert_eq!(DraftRows::parse("0"), Ok(0));
        assert_eq!(DraftRows::parse("6"), Ok(6));
        assert_eq!(DraftRows::parse("16"), Ok(16));
        assert_eq!(DraftRows::parse("32"), Ok(32));
        for bad in ["1", "33", "rows"] {
            let err = DraftRows::parse(bad).unwrap_err();
            assert!(err.contains("2..=32") && err.contains(bad), "{err}");
        }
    }

    /// Spec speech/08: a decimal keeps three places as thousandths and
    /// reads back; `auto` is the model's own value for the two kinds that
    /// take it.
    #[test]
    fn decimals_keep_thousandths_and_auto_is_the_models_own() {
        assert_eq!(Decimal::parse("6"), Ok(6_000));
        assert_eq!(Decimal::parse("0.25"), Ok(250));
        assert_eq!(Decimal::parse("1.5"), Ok(1_500));
        for bad in ["-1", "fast", "NaN", "inf"] {
            assert!(Decimal::parse(bad).unwrap_err().contains(bad), "{bad}");
        }
        assert_eq!(Decimal::render(&250), serde_json::json!(0.25));
        assert_eq!(Decimal::number(&1_500), Some(1_500));
        assert_eq!(AutoDecimal::parse("auto"), Ok(None));
        assert_eq!(AutoDecimal::parse("0.05"), Ok(Some(50)));
        assert!(AutoDecimal::parse("lots").unwrap_err().contains("auto"));
        assert_eq!(AutoDecimal::render(&None), serde_json::json!("auto"));
        assert_eq!(AutoCount::parse("AUTO"), Ok(None));
        assert_eq!(AutoCount::parse("3"), Ok(Some(3)));
        assert!(AutoCount::parse("1.5").is_err());
        round_trips::<Decimal>(6_500);
        round_trips::<AutoDecimal>(None);
        round_trips::<AutoDecimal>(Some(750));
        round_trips::<AutoCount>(Some(1));
        round_trips::<AutoCount>(None);
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

    #[test]
    fn sizes_are_multiples_of_32_named_once_each() {
        let sizes = Sizes::parse("2048x2048; 2752x1536,1024x1024").unwrap();
        assert_eq!(sizes, [ImageSize::new(2048, 2048), ImageSize::new(2752, 1536), ImageSize::new(1024, 1024)]);
        for (raw, says) in [
            ("1000x1024", "multiples of 32"),
            ("1024x1024;1024x1024", "twice"),
            ("1024", "<width>x<height>"),
            (";", "at least one"),
            ("0x32", "<width>x<height>"),
            ("32x16416", "at most 16384"),
        ] {
            assert!(Sizes::parse(raw).unwrap_err().contains(says), "{raw}");
        }
        // A file may give a list as well as the joined string.
        let list = serde_json::json!(["2048x2048", "1024x1024"]);
        assert_eq!(Sizes::file_text(&list), Ok(Some("2048x2048;1024x1024".into())));
        assert!(Sizes::file_text(&serde_json::json!([2048])).is_err());
        assert_eq!(Sizes::render(&sizes), serde_json::json!(["2048x2048", "2752x1536", "1024x1024"]));
        assert_eq!(Sizes::parse("16384x16384;32x16384"), Ok(vec![ImageSize::new(16384, 16384), ImageSize::new(32, 16384)]), "the bound itself");
        round_trips::<Sizes>(sizes);
    }
}
