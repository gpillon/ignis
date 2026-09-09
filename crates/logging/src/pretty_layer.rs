//! The pretty layer: a human-readable line rendered from the same
//! [`LogRecord`] the JSON layer sees (`json_layer.rs`) — free to abbreviate
//! machine values for display (`duration_ms` → seconds, `*_bytes` → GiB)
//! since the underlying event keeps its native units regardless of which
//! layer is active (issue #78, user story 10).

use std::sync::Arc;
use std::time::SystemTime;

use tracing::Subscriber;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

use crate::record::LogRecord;
use crate::sink::LineSink;

pub struct PrettyLayer {
    sink: Arc<dyn LineSink>,
    /// ANSI color codes, gated at construction on the caller's TTY check
    /// (or an explicit override) — never re-probed per event.
    color: bool,
}

impl PrettyLayer {
    pub fn new(sink: Arc<dyn LineSink>, color: bool) -> Self {
        Self { sink, color }
    }
}

impl<S: Subscriber> Layer<S> for PrettyLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let record = LogRecord::from_event(event, SystemTime::now());
        self.sink.write_line(&render(&record, self.color));
    }
}

fn severity_color(severity_text: &str) -> &'static str {
    match severity_text {
        "TRACE" => "\x1b[90m", // bright black
        "DEBUG" => "\x1b[36m", // cyan
        "INFO" => "\x1b[32m",  // green
        "WARN" => "\x1b[33m",  // yellow
        "ERROR" => "\x1b[31m", // red
        _ => "\x1b[0m",
    }
}

const RESET: &str = "\x1b[0m";

/// Render one line: `<timestamp> <SEVERITY> <event_name> - <body> {attrs}`.
/// Attribute values render as their JSON text, except `*_ms` (shown also as
/// seconds) and `*_bytes` (shown also as GiB) — abbreviation only, the
/// underlying value is untouched.
fn render(record: &LogRecord, color: bool) -> String {
    let severity = if color {
        format!("{}{:>5}{RESET}", severity_color(record.severity_text), record.severity_text)
    } else {
        format!("{:>5}", record.severity_text)
    };

    let mut line = format!(
        "{} {} {} - {}",
        record.timestamp, severity, record.event_name, record.body
    );

    if !record.attributes.is_empty() {
        let rendered: Vec<String> = record
            .attributes
            .iter()
            .map(|(key, value)| format!("{key}={}", humanize(key, value)))
            .collect();
        line.push_str(" {");
        line.push_str(&rendered.join(", "));
        line.push('}');
    }

    line
}

fn humanize(key: &str, value: &serde_json::Value) -> String {
    let raw = value.to_string();
    if let Some(ms) = value.as_f64().filter(|_| key.ends_with("_ms")) {
        format!("{raw} ({:.3}s)", ms / 1000.0)
    } else if let Some(bytes) = value.as_f64().filter(|_| key.ends_with("_bytes")) {
        format!("{raw} ({:.3}GiB)", bytes / (1024.0 * 1024.0 * 1024.0))
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tracing_subscriber::layer::SubscriberExt;

    use super::*;
    use crate::sink::MemorySink;

    fn one_line(color: bool, f: impl FnOnce()) -> String {
        let sink = Arc::new(MemorySink::new());
        let layer = PrettyLayer::new(sink.clone(), color);
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, f);
        let mut lines = sink.lines();
        assert_eq!(lines.len(), 1, "expected exactly one emitted line: {lines:?}");
        lines.remove(0)
    }

    #[test]
    fn renders_event_name_and_body() {
        let line = one_line(false, || tracing::info!(name: "ignis.model.loaded", "model ready"));
        assert!(line.contains("ignis.model.loaded"), "{line}");
        assert!(line.contains("model ready"), "{line}");
        assert!(line.contains("INFO"), "{line}");
    }

    #[test]
    fn no_color_means_no_ansi_codes() {
        let line = one_line(false, || tracing::error!(name: "ignis.test.err", "boom"));
        assert!(!line.contains('\x1b'), "{line}");
    }

    #[test]
    fn color_wraps_the_severity_in_ansi_codes() {
        let line = one_line(true, || tracing::error!(name: "ignis.test.err", "boom"));
        assert!(line.contains('\x1b'), "{line}");
    }

    #[test]
    fn duration_ms_is_humanized_as_seconds_alongside_the_raw_value() {
        let line = one_line(false, || {
            tracing::info!(name: "ignis.test.dur", duration_ms = 17_400i64, "done");
        });
        assert!(line.contains("duration_ms=17400"), "{line}");
        assert!(line.contains("17.400s"), "{line}");
    }

    #[test]
    fn bytes_are_humanized_as_gib_alongside_the_raw_value() {
        let gib_in_bytes: i64 = 2 * 1024 * 1024 * 1024;
        let line = one_line(false, || {
            tracing::info!(name: "ignis.test.mem", vram_bytes = gib_in_bytes, "loaded");
        });
        assert!(line.contains(&format!("vram_bytes={gib_in_bytes}")), "{line}");
        assert!(line.contains("2.000GiB"), "{line}");
    }
}
