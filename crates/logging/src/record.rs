//! The canonical event model: [`LogRecord`], built once per `tracing::Event`
//! from its `Metadata` + fields, and shared verbatim by both layers
//! (`json_layer.rs` renders it as JSONL, `pretty_layer.rs` renders it as a
//! human-readable line) — the two never see different data, only different
//! rendering (issue #78, "the canonical event *is* `tracing::Event` +
//! `Metadata`").
//!
//! `event_name` comes from `Metadata::name()`: call sites that want a
//! stable `ignis.<subsystem>.<event>` name pass it via the `name:` macro
//! argument (`info!(name: "ignis.foo.bar", key = value, "human body")`) —
//! nothing here invents or rewrites it. Sites that don't pass `name:` get
//! `tracing`'s default (`"event <file>:<line>"`); migrating call sites to
//! real names is Phase 2 (#79), out of scope here.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use tracing::field::{Field, Visit};
use tracing::{Event, Level};

/// The service resource attributes (OTel semantic conventions
/// `service.name`/`service.version`, GitHub #79): every event carries these
/// so a log is self-identifying without the reader having to know which
/// binary emitted it from context alone. `ignis` is a single service across
/// all its binaries (`ignis-server`, `vendor-ninfer`, …) — there is no
/// per-crate name here, only one version shared by the whole workspace
/// (`workspace.package.version`), so reading it from this crate's own
/// `CARGO_PKG_VERSION` is equivalent to reading it from the caller's.
pub const SERVICE_NAME: &str = "ignis";
pub const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Whole words (case-insensitive, `_`/`-`/`.`-delimited) that mark an
/// attribute key as sensitive by construction (GitHub #79, ADR 0013:
/// discipline + tests, not a typed attribute registry) — a call site that
/// names a field `authorization`, `user_password`, `session_cookie`, etc.
/// gets it redacted automatically, so a future call site that accidentally
/// attaches a secret under an obviously-named field is caught before it
/// ships, without requiring every call site to remember to redact by hand.
///
/// Whole-word, not substring: `tokens` (a completion token *count* — fine to
/// log, e.g. `ignis.request.done`'s attribute) must not collide with
/// `token`/`bearer_token` (a credential) just because one contains the
/// other's letters.
const SENSITIVE_KEY_WORDS: &[&str] =
    &["password", "secret", "token", "authorization", "bearer", "cookie", "credential"];

/// Adjacent-word pairs (after splitting on `_`/`-`/`.`, joined without the
/// separator) that mark a key as sensitive even though neither word alone
/// is in [`SENSITIVE_KEY_WORDS`] — `api_key`/`private_key` are credentials;
/// `key` alone is too common a word (e.g. a cache or map key) to blanket-flag.
const SENSITIVE_KEY_WORD_PAIRS: &[&str] = &["apikey", "privatekey"];

/// The text a redacted attribute value is replaced with.
const REDACTED: &str = "[REDACTED]";

fn is_sensitive_key(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    if words.iter().any(|word| SENSITIVE_KEY_WORDS.contains(word)) {
        return true;
    }
    words.windows(2).any(|pair| {
        let joined = format!("{}{}", pair[0], pair[1]);
        SENSITIVE_KEY_WORD_PAIRS.contains(&joined.as_str())
    })
}

/// One canonical logging event, OTel LogRecord-shaped (spec §5-8): the same
/// value both layers render, just rendered differently.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LogRecord {
    pub timestamp: String,
    pub severity_text: &'static str,
    pub severity_number: u8,
    pub event_name: String,
    pub body: String,
    pub attributes: BTreeMap<String, Value>,
    #[serde(rename = "service.name")]
    pub service_name: &'static str,
    #[serde(rename = "service.version")]
    pub service_version: &'static str,
}

impl LogRecord {
    /// Build a record from a live `tracing::Event`, timestamped `now`.
    pub fn from_event(event: &Event<'_>, now: SystemTime) -> Self {
        let meta = event.metadata();
        let level = *meta.level();
        let mut visitor = RecordVisitor::default();
        event.record(&mut visitor);
        LogRecord {
            timestamp: format_rfc3339(now),
            severity_text: severity_text(level),
            severity_number: severity_number(level),
            event_name: meta.name().to_owned(),
            body: visitor.body,
            attributes: visitor.attributes,
            service_name: SERVICE_NAME,
            service_version: SERVICE_VERSION,
        }
    }
}

/// OTel severity text: the level's own name, uppercase (spec §6).
pub fn severity_text(level: Level) -> &'static str {
    match level {
        Level::TRACE => "TRACE",
        Level::DEBUG => "DEBUG",
        Level::INFO => "INFO",
        Level::WARN => "WARN",
        Level::ERROR => "ERROR",
    }
}

/// OTel severity number: the base of each level's 1-4 range (TRACE=1,
/// DEBUG=5, INFO=9, WARN=13, ERROR=17 — OTel logs data model §"Severity
/// fields").
pub fn severity_number(level: Level) -> u8 {
    match level {
        Level::TRACE => 1,
        Level::DEBUG => 5,
        Level::INFO => 9,
        Level::WARN => 13,
        Level::ERROR => 17,
    }
}

/// Collects one `tracing::Event`'s fields into `body` (the `message` field,
/// tracing's name for a macro's trailing format-string argument) and
/// `attributes` (everything else), preserving each field's native serde
/// type (`record_i64` → a JSON number, never a string) — spec's "typed
/// structured attributes" requirement.
#[derive(Debug, Default)]
struct RecordVisitor {
    body: String,
    attributes: BTreeMap<String, Value>,
}

impl RecordVisitor {
    fn set(&mut self, field: &Field, value: Value) {
        if field.name() == "message" {
            self.body = match value {
                Value::String(s) => s,
                other => other.to_string(),
            };
        } else if is_sensitive_key(field.name()) {
            self.attributes
                .insert(field.name().to_owned(), Value::String(REDACTED.to_owned()));
        } else {
            self.attributes.insert(field.name().to_owned(), value);
        }
    }
}

impl Visit for RecordVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.set(field, Value::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.set(field, Value::from(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.set(field, Value::from(value));
    }

    fn record_i128(&mut self, field: &Field, value: i128) {
        // No native JSON i128; fall back to its decimal text rather than
        // silently truncating to i64/u64.
        self.set(field, Value::String(value.to_string()));
    }

    fn record_u128(&mut self, field: &Field, value: u128) {
        self.set(field, Value::String(value.to_string()));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        let json = serde_json::Number::from_f64(value)
            .map(Value::Number)
            .unwrap_or(Value::Null);
        self.set(field, json);
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.set(field, Value::String(value.to_owned()));
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // tracing's own message field is recorded as `fmt::Arguments`,
        // whose `Debug` impl is identical to its `Display` impl (plain
        // text, no surrounding quotes) — this is the same path
        // `tracing_subscriber`'s own formatters rely on.
        let text = format!("{value:?}");
        self.set(field, Value::String(text));
    }
}

/// RFC 3339, UTC, millisecond precision (spec §9: "ordering and correlation
/// unambiguous regardless of local timezone") — hand-rolled instead of
/// pulling in a datetime crate: this is the only place in the crate that
/// needs one, and the calendar math (Howard Hinnant's `civil_from_days`) is
/// a few pure lines, easily unit-tested against known epoch values.
pub fn format_rfc3339(time: SystemTime) -> String {
    let since_epoch = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let total_millis = since_epoch.as_millis() as i64;
    let days = total_millis.div_euclid(86_400_000);
    let millis_of_day = total_millis.rem_euclid(86_400_000);

    let (year, month, day) = civil_from_days(days);
    let hour = millis_of_day / 3_600_000;
    let minute = (millis_of_day / 60_000) % 60;
    let second = (millis_of_day / 1_000) % 60;
    let millis = millis_of_day % 1_000;

    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z"
    )
}

/// Days-since-epoch → (year, month, day), UTC civil calendar. Howard
/// Hinnant's `civil_from_days` (public domain algorithm, chrono-compatible
/// for the proleptic Gregorian calendar).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_formats_as_the_unix_epoch_instant() {
        assert_eq!(format_rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn a_known_instant_formats_correctly() {
        // 2021-01-01T00:00:00.000Z == 1609459200 s since epoch.
        let t = UNIX_EPOCH + Duration::from_secs(1_609_459_200);
        assert_eq!(format_rfc3339(t), "2021-01-01T00:00:00.000Z");
    }

    #[test]
    fn sub_second_precision_is_preserved() {
        let t = UNIX_EPOCH + Duration::from_millis(1_609_459_200_500);
        assert_eq!(format_rfc3339(t), "2021-01-01T00:00:00.500Z");
    }

    #[test]
    fn resource_attributes_use_otel_semconv_key_names() {
        let record = LogRecord {
            timestamp: format_rfc3339(UNIX_EPOCH),
            severity_text: "INFO",
            severity_number: 9,
            event_name: "ignis.test.resource".to_owned(),
            body: "x".to_owned(),
            attributes: BTreeMap::new(),
            service_name: SERVICE_NAME,
            service_version: SERVICE_VERSION,
        };
        let json = serde_json::to_value(&record).expect("serializes");
        assert_eq!(json["service.name"], "ignis");
        assert_eq!(json["service.version"], SERVICE_VERSION);
    }

    #[test]
    fn sensitive_key_names_are_flagged_case_insensitively() {
        for key in ["Authorization", "API_KEY", "Bearer-Token", "user_password", "session_cookie"] {
            assert!(is_sensitive_key(key), "{key} should be flagged sensitive");
        }
        for key in ["model", "duration_ms", "artifact_path"] {
            assert!(!is_sensitive_key(key), "{key} should not be flagged sensitive");
        }
    }

    #[test]
    fn severity_mapping_matches_the_otel_range_bases() {
        assert_eq!(severity_number(Level::TRACE), 1);
        assert_eq!(severity_number(Level::DEBUG), 5);
        assert_eq!(severity_number(Level::INFO), 9);
        assert_eq!(severity_number(Level::WARN), 13);
        assert_eq!(severity_number(Level::ERROR), 17);
        assert_eq!(severity_text(Level::INFO), "INFO");
        assert_eq!(severity_text(Level::ERROR), "ERROR");
    }
}
