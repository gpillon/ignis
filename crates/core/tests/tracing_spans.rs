//! GitHub #81 / ADR 0012: the request-lifecycle span tree `ConcreteScheduler`
//! opens (`ignis.admission` / `ignis.prefill` / `ignis.decode.round` /
//! `ignis.completion`, `crates/core/src/concrete.rs`) carries the same
//! `request_id` throughout one request's real lifecycle, and the decode
//! side stays at round granularity — one span per `advance()` call's
//! decode step, never one per generated token.
//!
//! Driven with `MockCompute` behind the `Compute` seam (ADR 0006, same
//! pattern as `n8_lanes.rs`) so the whole test runs on a CPU, on the test's
//! own thread (no separate model thread here — that's `ignis-server`'s
//! concern, not core's).

use std::sync::{Arc, Mutex};

use ignis_core::types::{DecodeParams, RequestClass, RequestInput};
use ignis_core::{ConcreteScheduler, MockCompute, Scheduler};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// One captured span: its stable name and the `request_id` field it
/// declared at creation (every span this crate opens sets it directly, up
/// front — never `Empty` + a later `record`, unlike the HTTP root span
/// `ignis-server` opens).
#[derive(Debug, Clone)]
struct CapturedSpan {
    name: &'static str,
    request_id: Option<u64>,
}

#[derive(Default)]
struct RequestIdVisitor(Option<u64>);

impl Visit for RequestIdVisitor {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "request_id" {
            self.0 = Some(value);
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "request_id"
            && let Ok(v) = u64::try_from(value)
        {
            self.0 = Some(v);
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

/// Records every `ignis.*` span opened while active — enough to count them
/// by name and check their `request_id`, without pulling in `ignis-logging`
/// (this crate has no dependency on it; the span *shape* is what's under
/// test, not how a downstream layer renders it).
struct SpanCounter(Arc<Mutex<Vec<CapturedSpan>>>);

impl<S> Layer<S> for SpanCounter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        if !name.starts_with("ignis.") {
            return;
        }
        let mut visitor = RequestIdVisitor::default();
        attrs.record(&mut visitor);
        self.0.lock().unwrap().push(CapturedSpan { name, request_id: visitor.0 });
    }
}

fn input(tokens: &[u32], max_tokens: u32) -> RequestInput {
    RequestInput {
        model: "qwen3.8-27b".into(),
        tokens: tokens.to_vec(),
        params: DecodeParams {
            max_tokens: Some(max_tokens),
            ..DecodeParams::default()
        },
    }
}

/// Run `f` under a subscriber that records every `ignis.*` span opened
/// during it, returning what `f` produced alongside the capture.
fn capture_spans<T>(f: impl FnOnce() -> T) -> (T, Vec<CapturedSpan>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let layer = SpanCounter(captured.clone());
    let subscriber = tracing_subscriber::registry().with(layer);
    let result = tracing::subscriber::with_default(subscriber, f);
    let spans = captured.lock().unwrap().clone();
    (result, spans)
}

fn count(spans: &[CapturedSpan], name: &str) -> usize {
    spans.iter().filter(|s| s.name == name).count()
}

#[test]
fn a_multi_round_requests_spans_all_share_its_request_id() {
    let (request_id, spans) = capture_spans(|| {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
        // A 1-token prompt (fits one prefill chunk) with a 5-token budget:
        // single-lane, so each `advance()` performs at most one decode
        // round for it — 5 calls, 5 rounds.
        let id = sched.submit(input(&[1], 5), RequestClass::Agent).unwrap();
        for _ in 0..5 {
            sched.advance();
        }
        assert!(sched.is_idle(), "the request must have completed by round 5");
        id
    });

    assert!(!spans.is_empty(), "expected at least one ignis.* span, got none");
    for span in &spans {
        assert_eq!(
            span.request_id,
            Some(request_id),
            "every span in this request's lifecycle must carry its own request_id \
             (the value `trace_id` is derived from, ADR 0012): {spans:?}"
        );
    }
}

#[test]
fn decode_round_span_count_matches_rounds_not_tokens_or_lanes() {
    let (_id, spans) = capture_spans(|| {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
        sched.submit(input(&[1], 5), RequestClass::Agent).unwrap();
        for _ in 0..5 {
            sched.advance();
        }
        assert!(sched.is_idle());
    });

    assert_eq!(
        count(&spans, "ignis.admission"),
        1,
        "dealt a lane exactly once: {spans:?}"
    );
    assert_eq!(
        count(&spans, "ignis.prefill"),
        1,
        "a 1-token prompt fits in a single prefill chunk: {spans:?}"
    );
    assert_eq!(
        count(&spans, "ignis.decode.round"),
        5,
        "5 requested tokens over 5 single-lane advance() calls -> exactly \
         5 rounds (one span per round, never per token): {spans:?}"
    );
    assert_eq!(
        count(&spans, "ignis.completion"),
        1,
        "completes exactly once: {spans:?}"
    );
}

#[test]
fn decode_round_spans_are_shared_across_lanes_in_one_batched_round() {
    // Two single-lane... no: two *concurrent* requests, one round each,
    // decoded together in the same `advance()` call (a real batched decode
    // round) — this asserts the round-per-request-per-call shape holds
    // under batching too, not just the trivial single-request case above.
    let (ids, spans) = capture_spans(|| {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute);
        let a = sched.submit(input(&[1], 1), RequestClass::Agent).unwrap();
        let b = sched.submit(input(&[2], 1), RequestClass::Agent).unwrap();
        sched.advance(); // both prefill, get admitted, and decode their one token in this call
        assert!(sched.is_idle(), "both requests have a 1-token budget");
        (a, b)
    });

    assert_eq!(
        count(&spans, "ignis.decode.round"),
        2,
        "one decode-round span per request in the batched round, not one for the whole call: {spans:?}"
    );
    let round_request_ids: std::collections::BTreeSet<u64> = spans
        .iter()
        .filter(|s| s.name == "ignis.decode.round")
        .filter_map(|s| s.request_id)
        .collect();
    assert_eq!(
        round_request_ids,
        std::collections::BTreeSet::from([ids.0, ids.1]),
        "each request's own round span carries its own id, not the other's: {spans:?}"
    );
}
