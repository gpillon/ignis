//! The "1 main agent + N subagents" load: the trace-replay driver
//! (`replay`) drives a realistic concurrent load into the endpoint —
//! through the real HTTP transport (`HttpEndpoint`) against the in-process
//! mock engine (`common::MockEngine`).
//!
//! The trace fixture (`tests/fixtures/main_plus_10.jsonl`) is **synthetic**:
//! no recorded trace exists in the repo yet (the recorded reference
//! baseline for the 99% gate, ADR 0007, is a separate GPU-driven item). The
//! fixture shapes the load like a real "1 main + ~10 subagents" coding
//! workload: one shared system + tools-style prefix, a long main-agent
//! request, and 10 staggered subagent requests sharing the prefix
//! (sibling-prefix reuse — the load the engine is sized for).

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::MockEngine;
use ignis_bench::client::{replay, Endpoint, HttpEndpoint, ReplayConfig};
use ignis_bench::metrics::Run;
use ignis_bench::trace::{RequestClass, Trace};

/// The synthetic "1 main + 10 subagents" fixture (see the module docs).
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/main_plus_10.jsonl")
}

fn trace() -> Trace {
    Trace::from_path(&fixture()).expect("a valid trace fixture")
}

#[test]
fn a_main_plus_ten_subagents_load_is_replayed_with_per_request_metrics() {
    let engine = MockEngine::start();
    let trace = trace();
    assert_eq!(trace.len(), 11, "1 main + 10 subagents");

    let ep: Arc<dyn Endpoint> = Arc::new(HttpEndpoint::new(engine.url()));
    // `time_scale 0.0` = no artificial waiting between arrivals (a fast
    // test; the staggered-arrival profile is covered by the other test).
    let cfg = ReplayConfig {
        max_concurrency: 10,
        time_scale: 0.0,
    };
    let results = replay(ep, &trace, &cfg);

    // Every request was replayed and produced per-request ttft / tok-s
    // (the ticket's acceptance).
    assert_eq!(results.len(), 11, "one metric per request");
    assert!(results.iter().all(|m| m.ok), "every request must succeed");
    for m in &results {
        assert!(m.ttft_ms > 0.0, "{id}: a measured ttft", id = m.id);
        assert!(m.n_tokens > 0, "{id}: tokens were generated", id = m.id);
        assert!(m.tok_s() > 0.0, "{id}: a measured decode speed (streaming)", id = m.id);
    }

    // The classes are preserved (ADR 0007: the per-class gate tracks main
    // and subagents separately).
    let run = Run::new("mock", results.clone());
    let main = run.stats_for(RequestClass::Main).expect("main stats");
    let sub = run.stats_for(RequestClass::Sub).expect("sub stats");
    assert_eq!(main.n_requests, 1, "exactly one main-agent request");
    assert_eq!(sub.n_requests, 10, "10 subagent requests");

    // The load was actually concurrent (not serial): the mock's peak
    // in-flight exceeds a serial baseline, and the driver respected its
    // concurrency bound.
    let peak = engine.state.peak_in_flight();
    assert!(peak >= 2, "the mock must see overlapping requests (peak {peak})");
    assert!(
        peak <= cfg.max_concurrency,
        "the driver must not exceed max_concurrency (peak {peak})"
    );
    assert_eq!(
        engine.state.n_completed(),
        11,
        "the mock must have completed all 11 requests"
    );
}

#[test]
fn the_staggered_arrivals_produce_a_realistic_concurrency_profile() {
    let engine = MockEngine::start();
    let trace = trace();
    // `time_scale 1.0` = the fixture's arrival offsets at real time (the
    // subagents arrive at 100 ms .. 1000 ms offsets).
    let ep: Arc<dyn Endpoint> = Arc::new(HttpEndpoint::new(engine.url()));
    let cfg = ReplayConfig {
        max_concurrency: 10,
        time_scale: 1.0,
    };
    let start = std::time::Instant::now();
    let results = replay(ep, &trace, &cfg);
    let wall = start.elapsed();
    // The last subagent arrives at t=1000 ms (the fixture's last offset);
    // the replay must not finish before it (arrival offsets are honored,
    // not squashed — the scheduler sees a realistic profile).
    assert!(
        wall >= std::time::Duration::from_millis(900),
        "wall {wall:?} should honor the staggered arrivals"
    );
    assert_eq!(results.len(), 11);
    assert!(results.iter().all(|m| m.ok));
    assert_eq!(
        engine.state.n_completed(),
        11,
        "the mock must have completed all 11 requests"
    );
}