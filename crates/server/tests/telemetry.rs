//! v1 telemetry (server-02, design §5) — integration tests at the public
//! seam: a real `ConcreteScheduler` (the deterministic `MockCompute`, ADR
//! 0006) driven by the engine's model thread (GitHub #69), asserting the
//! interval + request JSONL lines are emitted and stay consistent under
//! concurrent access.

use std::sync::Arc;
use std::time::Duration;

use ignis_core::{
    mock::MockCompute, ConcreteScheduler, DecodeParams, RequestClass, RequestInput,
    SchedulerConfig,
};
use ignis_server::engine::{collect_tokens, Engine};
use ignis_server::telemetry::{FixedClock, MemorySink};

// The telemetry consumer runs as a separate async task off the model thread
// (GitHub #69) — `nudge()` gives it a few scheduling turns to drain
// whatever is already sitting in the facts channel before a test inspects
// the sink.
#[path = "support/mod.rs"]
mod support;
use support::nudge;

/// A test engine: the concrete scheduler over a deterministic mock, with
/// telemetry written to `sink` (a fixed clock keeps the request lines
/// deterministic — ADR 0006).
fn engine_with_sink(sink: Arc<MemorySink>) -> Engine {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "test-model".into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Engine::with_sinks(Box::new(scheduler), sink, Arc::new(FixedClock::new(0)))
}

fn input(tokens: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        model: "test-model".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(max_tokens),
            ..DecodeParams::default()
        },
    }
}

/// The record kind of a JSONL line (`"interval"` or `"request"`), owned.
fn kind(line: &str) -> String {
    serde_json::from_str::<serde_json::Value>(line)
        .unwrap()
        .get("kind")
        .and_then(|k| k.as_str())
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn a_real_request_emits_interval_and_request_lines() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .await
        .expect("submit");
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let lines = sink.lines();
    // At least one interval line (one per model-thread step), plus the
    // request's lines.
    assert!(
        lines.iter().any(|l| kind(l) == "interval"),
        "at least one interval line: {lines:?}"
    );
    // The request went through the full lifecycle: admitted → ttft → done.
    let events: Vec<String> = lines
        .iter()
        .filter(|l| kind(l) == "request")
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l)
                .unwrap()
                .get("event")
                .and_then(|e| e.as_str())
                .unwrap()
                .to_string()
        })
        .collect();
    assert!(
        events.iter().any(|e| e == "admitted"),
        "an admitted line: {events:?}"
    );
    assert!(
        events.iter().any(|e| e == "ttft"),
        "a ttft line (first token): {events:?}"
    );
    assert!(
        events.iter().any(|e| e == "done"),
        "a done line: {events:?}"
    );
}

#[tokio::test]
async fn the_interval_counters_track_inflight_requests() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .await
        .expect("submit");
    // The model thread races ahead unthrottled (no manual stepping
    // anymore, GitHub #69), so by the time this returns the request may
    // already be fully done — scan every interval line emitted along the
    // way rather than assuming the last one still shows it running.
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let lines = sink.lines();
    let saw_running = lines.iter().filter(|l| kind(l) == "interval").any(|l| {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        v["running"].as_u64().unwrap_or(0) >= 1
    });
    assert!(
        saw_running,
        "at least one interval line must report the request running: {lines:?}"
    );

    // The wait-free ArcSwap snapshot agrees with the last JSONL interval
    // line it was published alongside (GitHub #69 — read without the
    // facts channel).
    let last_interval: serde_json::Value = lines
        .iter()
        .rev()
        .find(|l| kind(l) == "interval")
        .map(|l| serde_json::from_str(l).unwrap())
        .expect("an interval line");
    let snapshot = engine.interval_counters();
    assert_eq!(
        snapshot.running as u64,
        last_interval["running"].as_u64().unwrap()
    );
    assert_eq!(
        snapshot.waiting as u64,
        last_interval["waiting"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn concurrent_submits_do_not_deadlock() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    // Several tasks submit against the shared engine concurrently. The
    // model thread's command channel and the sink's buffer lock must not
    // invert (a deadlock here would hang the join, and the test would time
    // out).
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let engine = engine.clone();
            tokio::spawn(async move {
                for _ in 0..6 {
                    let (_id, mut rx) = engine
                        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
                        .await
                        .expect("submit");
                    let _ = collect_tokens(&mut rx, Duration::from_secs(5)).await;
                }
            })
        })
        .collect();
    for handle in handles {
        handle.await.expect("concurrent submits must not deadlock");
    }
    nudge().await;
    // The shared sink captured interval lines (it is thread-safe).
    assert!(
        sink.lines().iter().any(|l| kind(l) == "interval"),
        "the shared sink captured an interval line"
    );
}

#[tokio::test]
async fn reading_the_sink_while_the_telemetry_consumer_is_still_draining_does_not_block() {
    // Since GitHub #69, sink I/O runs entirely on the async telemetry
    // consumer, off the model thread — a concurrent read of the sink must
    // never block on, or be blocked by, that consumer's own writes (each
    // write only ever holds the sink's own buffer lock for a single push).
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 2), RequestClass::Interactive)
        .await
        .expect("submit");
    for _ in 0..8 {
        let _ = sink.lines();
        tokio::task::yield_now().await;
    }
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;
    let lines = sink.lines();
    assert!(lines.iter().any(|l| kind(l) == "request"));
    assert!(lines.iter().any(|l| kind(l) == "interval"));
}
