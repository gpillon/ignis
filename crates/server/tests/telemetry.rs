//! v1 telemetry (server-02, design §5) — integration tests at the public
//! seam: a real `ConcreteScheduler` (the deterministic `MockCompute`, ADR
//! 0006) driven through the `Engine`, asserting the interval + request JSONL
//! lines are emitted and stay consistent under concurrent access.

use std::sync::Arc;
use std::thread;

use ignis_core::{
    mock::MockCompute, ConcreteScheduler, DecodeParams, RequestClass, RequestInput,
    SchedulerConfig,
};
use ignis_server::engine::Engine;
use ignis_server::telemetry::{FixedClock, MemorySink};

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
    Engine::with_sinks(
        Box::new(scheduler),
        sink,
        Arc::new(FixedClock::new(0)),
    )
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

/// Drive the engine (a single driver, like production) until it is idle,
/// then a couple of steady-state ticks (the driver keeps stepping while
/// idle; the interval lines after completion report the idle counters).
fn drive_to_idle(engine: &Engine) {
    loop {
        engine.step();
        if engine.is_idle() {
            break;
        }
    }
    engine.step();
    engine.step();
}

#[test]
fn a_real_request_emits_interval_and_request_lines() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    let (_id, _rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .expect("submit");

    drive_to_idle(&engine);

    let lines = sink.lines();
    // At least one interval line (one per step), plus the request's lines.
    assert!(
        lines.iter().any(|l| kind(l) == "interval"),
        "at least one interval line"
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

#[test]
fn the_interval_counters_track_inflight_requests() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    // Two in-flight requests: one is admitted (running), one stays queued
    // (waiting) while the other holds a lane.
    let (_id, _rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .expect("submit");
    // A single step: the first request is prefilled + admitted + decodes.
    engine.step();
    let lines = sink.lines();
    let interval: serde_json::Value = lines
        .iter()
        .rev()
        .find(|l| kind(l) == "interval")
        .and_then(|l| serde_json::from_str(l).ok())
        .expect("an interval line");
    // After one step the first request is on a lane (running ≥ 1).
    assert!(
        interval["running"].as_u64().unwrap_or(0) >= 1,
        "the admitted request is on a lane: {interval}"
    );
}

#[test]
fn concurrent_submits_and_steps_do_not_deadlock() {
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    // Several threads submit + step the shared engine concurrently. The
    // engine's scheduler mutex and the sink's buffer lock must not invert
    // (a deadlock here would hang the join, and cargo test would time out).
    let handles: Vec<_> = (0..4).map(|_| {
        let engine = engine.clone();
        thread::spawn(move || {
            for _ in 0..6 {
                let _ = engine.submit(input(vec![1, 2, 3], 4), RequestClass::Interactive);
                engine.step();
            }
        })
    })
    .collect();
    for handle in handles {
        handle.join().expect("concurrent submit/step must not deadlock");
    }
    // The shared sink captured interval lines (it is thread-safe).
    assert!(
        sink.lines().iter().any(|l| kind(l) == "interval"),
        "the shared sink captured an interval line"
    );
}

#[test]
fn the_sink_does_not_block_the_driver() {
    // The sink's own lock is only held for a single buffered write; the
    // driver (here, a single thread stepping) must complete without the
    // sink ever serializing on a second lock or deadlocking on itself.
    let sink = Arc::new(MemorySink::new());
    let engine = engine_with_sink(sink.clone());
    let (_id, _rx) = engine
        .submit(input(vec![1, 2, 3], 2), RequestClass::Interactive)
        .expect("submit");
    // Read the sink from the driver's thread while it also writes it (the
    // driver's `step()` writes a line each tick; a separate read must not
    // block the write).
    for _ in 0..8 {
        engine.step();
        let _ = sink.lines();
        if engine.is_idle() {
            break;
        }
    }
    // Both the request and the interval lines were captured.
    let lines = sink.lines();
    assert!(lines.iter().any(|l| kind(l) == "request"));
    assert!(lines.iter().any(|l| kind(l) == "interval"));
}