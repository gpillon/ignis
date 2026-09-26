//! v1 telemetry (server-02, design §5) — integration tests at the public
//! seam: a real `ConcreteScheduler` (the deterministic `MockCompute`, ADR
//! 0006) driven by the engine's model thread (GitHub #69), asserting the
//! `ignis.scheduler.interval` event (ADR 0025) and the `ignis.request.*`
//! events (GitHub #79) are emitted and stay consistent under concurrent
//! access.

use std::sync::Arc;
use std::time::Duration;

use ignis_core::{
    mock::MockCompute, ConcreteScheduler, DecodeParams, RequestClass, RequestInput,
    SchedulerConfig,
};
use ignis_logging::MemorySink;
use ignis_server::engine::{collect_tokens, Engine};
use ignis_server::telemetry::FixedClock;

// The telemetry consumer runs as a separate async task off the model thread
// (GitHub #69) — `nudge()` gives it a few scheduling turns to drain
// whatever is already sitting in the facts channel before a test inspects
// the captured events.
#[path = "support/mod.rs"]
mod support;
use support::nudge;

/// A test engine: the concrete scheduler over a deterministic mock, on a
/// fixed clock (keeps the request events deterministic — ADR 0006).
fn engine() -> Engine {
    engine_with_chunk(SchedulerConfig::default().serving_chunk_tokens)
}

/// Same as [`engine`], but with the scheduler's serving prefill chunk width
/// narrowed to `chunk` tokens — small enough that a short test prompt still
/// needs several `SchedEvent::PrefillChunk`s (P3-06), so the engine's real
/// `Command::Submit` → facts-channel → `telemetry_task` wiring for the
/// per-phase fields gets exercised end to end, not just `Telemetry`'s own
/// methods called directly.
fn engine_with_chunk(chunk: u32) -> Engine {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "test-model".into(),
            serving_chunk_tokens: chunk,
            ..SchedulerConfig::default()
        },
        compute,
    );
    Engine::with_clock(Box::new(scheduler), Arc::new(FixedClock::new(0)))
}

fn input(tokens: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: "test-model".into(),
        tokens,
        params: DecodeParams {
            max_tokens: Some(max_tokens),
            ..DecodeParams::default()
        },
        constrained: None,
    }
}

/// Installs `ignis-logging`'s `JsonLayer` (no level filter, so DEBUG is
/// kept) as the thread-local default tracing subscriber for the returned
/// guard's lifetime, capturing every event the engine's telemetry task
/// emits. `#[tokio::test]` defaults to a current-thread runtime, so the
/// engine's spawned telemetry task runs on this same OS thread and observes
/// the same thread-local dispatcher — hold the guard alive for as long as
/// events need to be captured.
fn capture_events() -> (Arc<MemorySink>, tracing::subscriber::DefaultGuard) {
    use tracing_subscriber::layer::SubscriberExt;
    let sink = Arc::new(MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    (sink, tracing::subscriber::set_default(subscriber))
}

fn events(log_sink: &MemorySink) -> Vec<serde_json::Value> {
    log_sink
        .lines()
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn event_names(log_sink: &MemorySink) -> Vec<String> {
    events(log_sink)
        .iter()
        .map(|e| e["event_name"].as_str().unwrap().to_string())
        .collect()
}

/// The `ignis.scheduler.interval` events captured so far, in order.
fn intervals(log_sink: &MemorySink) -> Vec<serde_json::Value> {
    events(log_sink)
        .into_iter()
        .filter(|e| e["event_name"] == "ignis.scheduler.interval")
        .collect()
}

#[tokio::test]
async fn a_real_request_emits_interval_and_request_events() {
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .await
        .expect("submit");
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let events = event_names(&log_sink);
    assert!(
        events.iter().any(|e| e == "ignis.scheduler.interval"),
        "at least one interval event: {events:?}"
    );
    // The request went through the full lifecycle: admitted → ttft → done.
    for name in ["ignis.request.admitted", "ignis.request.ttft", "ignis.request.done"] {
        assert!(events.iter().any(|e| e == name), "a {name} event: {events:?}");
    }
}

/// GitHub #120: the request's admission class rides the canonical
/// `ignis.request.*` events, so a per-class gate cell is attributable
/// without a second run. Submitted directly through `Engine::submit`
/// (bypassing HTTP) — this is the seam `api.rs`'s `class`/"@<lane>"
/// resolution feeds into; that resolution is covered at the HTTP layer by
/// `crates/server/src/api.rs`'s own unit tests.
#[tokio::test]
async fn the_request_class_rides_the_canonical_request_events() {
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Agent)
        .await
        .expect("submit");
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let events = events(&log_sink);
    for name in ["ignis.request.admitted", "ignis.request.ttft", "ignis.request.done"] {
        let event = events
            .iter()
            .find(|e| e["event_name"] == name)
            .unwrap_or_else(|| panic!("a {name} event: {events:?}"));
        assert_eq!(
            event["attributes"]["class"], "agent",
            "{name} should carry the Agent class: {event:?}"
        );
    }
}

/// P5-06 (GitHub #154): a request whose rounds commit runs reports its
/// speculative counters on its one `done` line, through the engine's real
/// event → facts → telemetry wiring.
#[tokio::test]
async fn the_request_log_carries_the_speculative_counters() {
    let (log_sink, _guard) = capture_events();
    // Rounds of 3 against a 7-token cap commit 3, 3, 1: three rounds, six
    // drafts proposed, four committed.
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: "test-model".into(),
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::with_runs(&[3])),
    );
    let engine = Engine::with_clock(Box::new(scheduler), Arc::new(FixedClock::new(0)));
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 7), RequestClass::Interactive)
        .await
        .expect("submit");
    let (tokens, _) = collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    assert_eq!(tokens.len(), 7);
    let events = events(&log_sink);
    let done = events
        .iter()
        .find(|e| e["event_name"] == "ignis.request.done")
        .unwrap_or_else(|| panic!("a done event: {events:?}"));
    assert_eq!(done["attributes"]["tokens"], 7);
    assert_eq!(done["attributes"]["spec.rounds"], 3, "{done}");
    assert_eq!(done["attributes"]["spec.drafted"], 6, "{done}");
    assert_eq!(done["attributes"]["spec.accepted"], 4, "{done}");
}

#[tokio::test]
async fn the_interval_counters_track_inflight_requests() {
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 4), RequestClass::Interactive)
        .await
        .expect("submit");
    // The model thread races ahead unthrottled (no manual stepping
    // anymore, GitHub #69), so by the time this returns the request may
    // already be fully done — scan every interval event emitted along the
    // way rather than assuming the last one still shows it running.
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let intervals = intervals(&log_sink);
    let saw_running = intervals
        .iter()
        .any(|e| e["attributes"]["running"].as_u64().unwrap_or(0) >= 1);
    assert!(
        saw_running,
        "at least one interval event must report the request running: {intervals:?}"
    );

    // The wait-free ArcSwap snapshot (GitHub #69 — read without the facts
    // channel) is published every tick, and an interval event is logged
    // whenever the counters change (ADR 0025) — so the snapshot always
    // agrees with the last event logged.
    let last_interval = intervals.last().expect("an interval event");
    let snapshot = engine.interval_counters();
    assert_eq!(
        snapshot.running as u64,
        last_interval["attributes"]["running"].as_u64().unwrap()
    );
    assert_eq!(
        snapshot.waiting as u64,
        last_interval["attributes"]["waiting"].as_u64().unwrap()
    );
}

/// ADR 0025: the interval event is emitted when the counters change, not
/// once per scheduler step — a steady decode would otherwise flood the
/// DEBUG channel with identical lines.
#[tokio::test]
async fn a_steady_decode_logs_counter_changes_not_every_step() {
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 32), RequestClass::Interactive)
        .await
        .expect("submit");
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let intervals = intervals(&log_sink);
    let counters: Vec<(u64, u64, u64, u64)> = intervals
        .iter()
        .map(|e| {
            let a = &e["attributes"];
            (
                a["tick"].as_u64().unwrap(),
                a["waiting"].as_u64().unwrap(),
                a["running"].as_u64().unwrap(),
                a["kv_evictions"].as_u64().unwrap(),
            )
        })
        .collect();
    for pair in counters.windows(2) {
        assert_ne!(
            (pair[0].1, pair[0].2, pair[0].3),
            (pair[1].1, pair[1].2, pair[1].3),
            "two consecutive interval events carry the same counters: {counters:?}"
        );
    }
    let last_tick = counters.last().expect("an interval event").0;
    assert!(
        last_tick >= 32,
        "32 generated tokens take at least 32 scheduler steps: {counters:?}"
    );
    assert!(
        (counters.len() as u64) < last_tick,
        "{} interval events over {last_tick} steps — one per step: {counters:?}",
        counters.len()
    );
    for event in &intervals {
        assert_eq!(event["severity_text"], "DEBUG", "{event}");
    }
}

#[tokio::test]
async fn concurrent_submits_do_not_deadlock() {
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    // Several tasks submit against the shared engine concurrently. The
    // model thread's command channel and the logging sink's buffer lock
    // must not invert (a deadlock here would hang the join, and the test
    // would time out).
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
    assert!(
        !intervals(&log_sink).is_empty(),
        "the shared sink captured an interval event"
    );
}

#[tokio::test]
async fn reading_the_log_while_the_telemetry_consumer_is_still_draining_does_not_block() {
    // Since GitHub #69, event emission runs entirely on the async telemetry
    // consumer, off the model thread — a concurrent read of the captured
    // log must never block on, or be blocked by, that consumer's own
    // writes (each write only ever holds the sink's own buffer lock for a
    // single push).
    let (log_sink, _guard) = capture_events();
    let engine = engine();
    let (_id, mut rx) = engine
        .submit(input(vec![1, 2, 3], 2), RequestClass::Interactive)
        .await
        .expect("submit");
    for _ in 0..8 {
        let _ = log_sink.lines();
        tokio::task::yield_now().await;
    }
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;
    let names = event_names(&log_sink);
    assert!(names.iter().any(|e| e.starts_with("ignis.request.")), "{names:?}");
    assert!(names.iter().any(|e| e == "ignis.scheduler.interval"), "{names:?}");
}

/// P3-06: exercises the real engine wiring for the request log's per-phase
/// fields — `Command::Submit` capturing `prompt_tokens` before `submit`
/// moves the input, `SchedEvent::PrefillChunk` routed through the facts
/// channel, and `telemetry_task`'s `on_prefill_chunk`/`on_admitted` — not
/// just `Telemetry`'s own methods called directly (see `telemetry.rs`'s
/// unit tests for that). A 10-token prompt at a 4-token chunk width needs 3
/// chunks (matches `crates/core/tests/interleaving.rs`'s
/// `a_long_prompt_is_split_into_chunk_wide_jobs`), so the `admitted` line
/// this produces must show exactly that.
#[tokio::test]
async fn a_chunked_prefill_reports_its_phase_summary_through_the_real_engine() {
    let (log_sink, _guard) = capture_events();
    let engine = engine_with_chunk(4);
    let (_id, mut rx) = engine
        .submit(input((1..=10).collect(), 1), RequestClass::Interactive)
        .await
        .expect("submit");
    collect_tokens(&mut rx, Duration::from_secs(5))
        .await
        .expect("the request completes");
    nudge().await;

    let admitted = events(&log_sink)
        .into_iter()
        .find(|e| e["event_name"] == "ignis.request.admitted")
        .expect("an admitted event");
    assert_eq!(admitted["attributes"]["prompt_tokens"], 10);
    assert_eq!(
        admitted["attributes"]["prefill_chunks_consumed"], 3,
        "a 10-token prompt at a 4-token chunk width takes 3 chunks"
    );
    assert_eq!(admitted["attributes"]["prefilled_tokens"], 10);
}
