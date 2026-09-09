//! The JSON (JSONL) layer: one compact JSON object per line, UTF-8, no
//! ANSI, mapping [`LogRecord`] straight onto its `Serialize` impl —
//! `serde_json::to_string` never emits a raw newline mid-object (embedded
//! newlines in `body`/attributes come out `\n`-escaped inside the string),
//! so JSONL framing holds regardless of what a call site logs.

use std::sync::Arc;
use std::time::SystemTime;

use tracing::Subscriber;
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use crate::record::LogRecord;
use crate::sink::LineSink;
use crate::trace_context;

pub struct JsonLayer {
    sink: Arc<dyn LineSink>,
}

impl JsonLayer {
    pub fn new(sink: Arc<dyn LineSink>) -> Self {
        Self { sink }
    }
}

impl<S> Layer<S> for JsonLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        trace_context::on_new_span(attrs, id, ctx);
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        trace_context::on_record(id, values, ctx);
    }

    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        let level = *event.metadata().level();
        let (trace_id, span_id) = trace_context::resolve(event, &ctx);
        let record = LogRecord::from_event(event, SystemTime::now(), trace_id, span_id);
        // `LogRecord` derives `Serialize` from primitive/JSON-native
        // fields only — serialization cannot fail.
        let line = serde_json::to_string(&record).expect("LogRecord always serializes");
        // `write_line_at` (not `write_line`): the queued sink (GitHub #80)
        // routes DEBUG/TRACE and INFO/WARN/ERROR into separate channels with
        // different backpressure policy — the level has to travel with the
        // line for that routing decision.
        self.sink.write_line_at(level, &line);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tracing_subscriber::layer::SubscriberExt;

    use super::*;
    use crate::sink::MemorySink;

    fn one_line(f: impl FnOnce()) -> String {
        let sink = Arc::new(MemorySink::new());
        let layer = JsonLayer::new(sink.clone());
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, f);
        let mut lines = sink.lines();
        assert_eq!(lines.len(), 1, "expected exactly one emitted line: {lines:?}");
        lines.remove(0)
    }

    #[test]
    fn output_is_valid_json() {
        let line = one_line(|| tracing::info!(name: "ignis.test.event", "hello"));
        let _: serde_json::Value = serde_json::from_str(&line).expect("valid json");
    }

    #[test]
    fn one_physical_line_per_event() {
        let sink = Arc::new(MemorySink::new());
        let layer = JsonLayer::new(sink.clone());
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(name: "ignis.test.a", "first");
            tracing::info!(name: "ignis.test.b", "second");
        });
        assert_eq!(sink.lines().len(), 2);
    }

    #[test]
    fn embedded_newlines_do_not_break_jsonl_framing() {
        let line = one_line(|| {
            tracing::info!(name: "ignis.test.event", detail = "line1\nline2", "body\nwith\nnewlines");
        });
        // The sink only ever saw one line (asserted by `one_line`); confirm
        // the raw text itself carries no literal newline byte either.
        assert!(!line.contains('\n'), "line must not contain a raw newline: {line:?}");
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["body"], "body\nwith\nnewlines");
        assert_eq!(value["attributes"]["detail"], "line1\nline2");
    }

    #[test]
    fn event_name_and_body_are_separate_fields() {
        let line = one_line(|| tracing::info!(name: "ignis.model.loaded", "model ready"));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["event_name"], "ignis.model.loaded");
        assert_eq!(value["body"], "model ready");
    }

    #[test]
    fn severity_is_mapped() {
        let line = one_line(|| tracing::warn!(name: "ignis.test.warn", "careful"));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["severity_text"], "WARN");
        assert_eq!(value["severity_number"], 13);
    }

    #[test]
    fn timestamp_is_rfc3339_utc() {
        let line = one_line(|| tracing::info!(name: "ignis.test.ts", "x"));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        let ts = value["timestamp"].as_str().expect("timestamp is a string");
        assert!(ts.ends_with('Z'), "{ts}");
        assert!(ts.contains('T'), "{ts}");
        assert!(ts.contains('.'), "sub-second precision expected: {ts}");
    }

    #[test]
    fn every_event_carries_the_service_resource_attributes() {
        let line = one_line(|| tracing::info!(name: "ignis.test.resource", "x"));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["service.name"], "ignis");
        assert_eq!(value["service.version"], env!("CARGO_PKG_VERSION"));
    }

    /// Testing Decisions (GitHub #79): "a covered logging path never emits a
    /// known-sensitive value" — a call site that (accidentally or not) names
    /// a field like a bearer token must never see the raw value reach a
    /// rendered record.
    #[test]
    fn a_bearer_token_shaped_value_is_redacted_not_logged() {
        let line = one_line(|| {
            tracing::info!(
                name: "ignis.test.secret",
                authorization = "Bearer sk-fake-not-a-real-secret-12345",
                "auth attempt"
            );
        });
        assert!(
            !line.contains("sk-fake-not-a-real-secret-12345"),
            "the secret value must never reach the rendered line: {line}"
        );
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["attributes"]["authorization"], "[REDACTED]");
    }

    #[test]
    fn structured_attributes_retain_their_native_type() {
        let line = one_line(|| {
            tracing::info!(
                name: "ignis.test.types",
                duration_ms = 17_400i64,
                ok = true,
                ratio = 0.5f64,
                label = "gpu-0",
                "typed attrs"
            );
        });
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        let attrs = &value["attributes"];
        assert_eq!(attrs["duration_ms"], serde_json::json!(17_400));
        assert!(attrs["duration_ms"].is_number(), "i64 must stay a JSON number, not a string");
        assert_eq!(attrs["ok"], serde_json::json!(true));
        assert_eq!(attrs["ratio"], serde_json::json!(0.5));
        assert_eq!(attrs["label"], serde_json::json!("gpu-0"));
    }

    /// spec §19 test #10 / GitHub #81: an event inside an active trace
    /// context carries a `trace_id` derived from the span's `request_id`.
    #[test]
    fn an_event_inside_a_request_span_carries_the_trace_id() {
        let line = one_line(|| {
            let span = tracing::info_span!("ignis.admission", request_id = 42u64);
            let _guard = span.enter();
            tracing::info!(name: "ignis.test.traced", "inside a span");
        });
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(value["trace_id"], "0000000000000000000000000000002a");
        assert!(value["span_id"].as_str().is_some());
    }

    /// spec §19 test #11: an event with no active span never fabricates
    /// either id — the fields must be genuinely absent, not `null`.
    #[test]
    fn an_event_outside_any_span_has_no_trace_or_span_id() {
        let line = one_line(|| tracing::info!(name: "ignis.test.untraced", "no span"));
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid json");
        let obj = value.as_object().expect("object");
        assert!(!obj.contains_key("trace_id"), "{value}");
        assert!(!obj.contains_key("span_id"), "{value}");
    }

    /// Root span + child span, within one request's lifecycle: both carry
    /// the same `trace_id`, distinct `span_id`s.
    #[test]
    fn root_and_child_spans_share_the_trace_id_but_not_the_span_id() {
        let sink = Arc::new(MemorySink::new());
        let layer = JsonLayer::new(sink.clone());
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let root = tracing::info_span!("ignis.http.request", request_id = 7u64);
            let _root_guard = root.enter();
            tracing::info!(name: "ignis.test.root_event", "at the root");
            let child = tracing::info_span!("ignis.prefill");
            let _child_guard = child.enter();
            tracing::info!(name: "ignis.test.child_event", "in a child span");
        });
        let lines = sink.lines();
        assert_eq!(lines.len(), 2);
        let root_event: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        let child_event: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(root_event["trace_id"], "00000000000000000000000000000007");
        assert_eq!(child_event["trace_id"], "00000000000000000000000000000007");
        assert_ne!(
            root_event["span_id"], child_event["span_id"],
            "each span keeps its own span_id even though the trace_id is shared"
        );
    }
}
