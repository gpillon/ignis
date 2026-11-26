//! GitHub #598 — a request the engine ends with [`FinishReason::Error`] after
//! a backend failure leaves one line in the server log that says why: the
//! client is told "see the server log" (`ENGINE_ERROR_MESSAGE`), and before
//! this the failure lived only in `ConcreteScheduler::last_error`.
//!
//! Its own test binary, every test under a capture subscriber: `tracing`
//! caches each callsite's interest, and a test elsewhere that reached these
//! callsites with no subscriber (`compute_errors.rs` does) could leave a
//! scoped capture seeing nothing.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ignis_core::scheduler::{Compute, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome};
use ignis_core::types::{
    ComputeError, DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent,
};
use ignis_core::{ConcreteScheduler, MAX_PREFILL_ATTEMPTS, MockCompute, Scheduler};
use tracing::field::{Field, Visit};
use tracing::{Event, Level};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::Layer;

/// One captured event: its level, its name and every field it recorded,
/// each as text.
#[derive(Debug, Clone)]
struct Line {
    level: Level,
    name: &'static str,
    fields: BTreeMap<String, String>,
}

impl Line {
    fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

#[derive(Default)]
struct Fields(BTreeMap<String, String>);

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().into(), value.into());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().into(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}

struct Capture(Arc<Mutex<Vec<Line>>>);

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        self.0.lock().unwrap().push(Line {
            level: *event.metadata().level(),
            name: event.metadata().name(),
            fields: fields.0,
        });
    }
}

/// Run `f` under a subscriber that records every event, returning what `f`
/// produced and the lines at WARN or above.
fn capture_warnings<T>(f: impl FnOnce() -> T) -> (T, Vec<Line>) {
    let captured = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::registry().with(Capture(captured.clone()));
    let result = tracing::subscriber::with_default(subscriber, f);
    let lines = captured.lock().unwrap().iter().filter(|line| line.level <= Level::WARN).cloned().collect();
    (result, lines)
}

fn input(tokens: Vec<u32>, max: u32) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "qwen3.8-27b".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(max),
            ..DecodeParams::default()
        },
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

fn ended_with_error(events: &[SchedEvent], id: RequestId) -> bool {
    events.iter().any(|e| {
        matches!(e, SchedEvent::Done { request, reason: FinishReason::Error, .. } if *request == id)
    })
}

/// A backend whose every prefill fails: the leaf refusing a sequence it will
/// never grant.
struct AlwaysFailingPrefill;

impl Compute for AlwaysFailingPrefill {
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        Err(ComputeError::Kernel(-1))
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        Ok(Vec::new())
    }
}

#[test]
fn a_prefill_that_keeps_failing_leaves_one_line_naming_its_request_and_the_error() {
    let ((id, events), lines) = capture_warnings(|| {
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", Arc::new(AlwaysFailingPrefill));
        let id = sched.submit(input(vec![1, 2], 4), RequestClass::Agent).unwrap();
        let mut events = Vec::new();
        for _ in 0..MAX_PREFILL_ATTEMPTS {
            events.extend(sched.advance());
        }
        (id, events)
    });
    assert!(ended_with_error(&events, id), "the request ends with an error: {events:?}");
    assert_eq!(lines.len(), 1, "one line for the ended request, not one per attempt: {lines:?}");
    let line = &lines[0];
    assert_eq!(line.name, "ignis.request.engine_error");
    assert_eq!(line.field("request_id"), Some(id.to_string().as_str()));
    assert_eq!(line.field("stage"), Some("prefill"));
    assert_eq!(line.field("error"), Some("kernel error (rc = -1)"));
    assert_eq!(line.field("attempts"), Some(MAX_PREFILL_ATTEMPTS.to_string().as_str()));
    assert_eq!(line.field("claim"), Some("none"));
    assert_eq!(line.field("start_position"), Some("0"));
}

/// A request that claims a sibling's shared prefix and whose every first job
/// fails: its line names the claim, so a failure that only claimants meet can
/// be told from one every request meets.
#[test]
fn a_failed_claimants_line_names_the_shared_prefix_it_claimed() {
    let ((main, sub, events), lines) = capture_warnings(|| {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
        let head: Vec<u32> = (1..=32).collect();
        let main = sched.submit(input(head.clone(), 16), RequestClass::Agent).unwrap();
        let mut events = sched.advance();
        let sub = sched.submit(input([head, (50..58).collect()].concat(), 4), RequestClass::Agent).unwrap();
        for _ in 0..MAX_PREFILL_ATTEMPTS {
            compute.fail_prefill(sub);
            events.extend(sched.advance());
        }
        (main, sub, events)
    });
    assert!(ended_with_error(&events, sub), "the claimant ends with an error: {events:?}");
    assert!(!ended_with_error(&events, main), "its publisher is served on: {events:?}");
    assert_eq!(lines.len(), 1, "{lines:?}");
    let line = &lines[0];
    assert_eq!(line.field("request_id"), Some(sub.to_string().as_str()));
    assert_eq!(line.field("stage"), Some("prefill"));
    assert_eq!(line.field("claim"), Some(format!("shared prefix of request {main}, 32 tokens").as_str()));
    assert_eq!(line.field("start_position"), Some("32"));
    assert_eq!(line.field("batch"), Some(sub.to_string().as_str()));
}

/// A turn resuming from the device checkpoint its conversation's last turn
/// left, whose every first job fails: its line names the checkpoint and where
/// it lives, the claim the burst of GitHub #598 failed on.
#[test]
fn a_failed_claimants_line_names_the_prompt_checkpoint_it_claimed() {
    let ((first, next, events), lines) = capture_warnings(|| {
        let compute = Arc::new(MockCompute::new());
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", compute.clone());
        let first = sched
            .submit(RequestInput { opener_tokens: Some(37), ..input((1..=40).collect(), 4) }, RequestClass::Agent)
            .unwrap();
        while !sched.is_idle() {
            sched.advance();
        }
        let prompt = [(1..=37).collect::<Vec<u32>>(), (500..523).collect()].concat();
        let next = sched
            .submit(RequestInput { opener_tokens: Some(57), ..input(prompt, 4) }, RequestClass::Agent)
            .unwrap();
        let mut events = Vec::new();
        for _ in 0..MAX_PREFILL_ATTEMPTS {
            compute.fail_prefill(next);
            events.extend(sched.advance());
        }
        (first, next, events)
    });
    assert!(ended_with_error(&events, next), "the claimant ends with an error: {events:?}");
    assert_eq!(lines.len(), 1, "{lines:?}");
    let line = &lines[0];
    assert_eq!(line.field("request_id"), Some(next.to_string().as_str()));
    assert_eq!(
        line.field("claim"),
        Some(format!("prompt checkpoint of request {first}, 37 tokens, from device").as_str())
    );
    assert_eq!(line.field("start_position"), Some("37"));
}

/// A first error that differs from the one that ended the request is named
/// beside it: the attempt that ended it is not always the one that went
/// wrong first.
struct FailingDifferently {
    calls: Mutex<i32>,
}

impl Compute for FailingDifferently {
    fn prefill_step(&self, _jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut calls = self.calls.lock().unwrap();
        *calls += 1;
        Err(ComputeError::Kernel(-*calls))
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        Ok(Vec::new())
    }
}

#[test]
fn a_prefill_whose_errors_differ_names_the_first_beside_the_last() {
    let (id, lines) = capture_warnings(|| {
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", Arc::new(FailingDifferently { calls: Mutex::new(0) }));
        let id = sched.submit(input(vec![1, 2], 4), RequestClass::Agent).unwrap();
        for _ in 0..MAX_PREFILL_ATTEMPTS {
            sched.advance();
        }
        id
    });
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0].field("request_id"), Some(id.to_string().as_str()));
    assert_eq!(lines[0].field("error"), Some(format!("kernel error (rc = -{MAX_PREFILL_ATTEMPTS})").as_str()));
    assert_eq!(lines[0].field("first_error"), Some("kernel error (rc = -1)"));
}

/// A backend whose every decode round fails.
struct DecodeFault {
    inner: MockCompute,
}

impl Compute for DecodeFault {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, _jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        Err(ComputeError::Kernel(-9))
    }
}

#[test]
fn a_failed_decode_round_leaves_one_line_per_request_it_ends() {
    let ((ids, events), lines) = capture_warnings(|| {
        let mut sched = ConcreteScheduler::new("qwen3.8-27b", Arc::new(DecodeFault { inner: MockCompute::new() }));
        let ids = [
            sched.submit(input(vec![1, 2], 4), RequestClass::Agent).unwrap(),
            sched.submit(input(vec![3, 4], 4), RequestClass::Agent).unwrap(),
        ];
        (ids, sched.advance())
    });
    for id in ids {
        assert!(ended_with_error(&events, id), "the round ends request {id}: {events:?}");
    }
    assert_eq!(lines.len(), 2, "one line per ended request: {lines:?}");
    let mut named: Vec<String> = lines.iter().filter_map(|line| line.field("request_id").map(String::from)).collect();
    named.sort();
    assert_eq!(named, ids.map(|id| id.to_string()), "{lines:?}");
    for line in &lines {
        assert_eq!(line.name, "ignis.request.engine_error");
        assert_eq!(line.field("stage"), Some("decode"));
        assert_eq!(line.field("error"), Some("kernel error (rc = -9)"));
    }
}
