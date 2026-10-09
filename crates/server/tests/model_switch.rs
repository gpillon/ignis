//! The model switch's orchestration (spec model-switch/01, GitHub #305),
//! driven directly — not through HTTP — over two or more `MockCompute`
//! models: the gate, the drain, the teardown before the load, and every way
//! a switch can fail without stopping the server. No GPU, and no wall-clock
//! timing proves anything (ADR 0006): a request is held mid-decode with
//! spec server/05's gate, a load with the mock loader's own hold.

use std::sync::Arc;
use std::time::Duration;

use ignis_core::{DecodeParams, FinishReason, RequestClass, RequestInput};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::engine::{collect_completion, Engine, EventStream};
use ignis_server::model_switch::{begin, ArtifactLoader, SwitchOutcome, SwitchRefusal, SwitchStarted, Switcher};
use ignis_server::{ActiveModel, ModelStatus, Server};
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

/// Long enough that no test here ever reaches it unless it means to.
const PATIENT: Duration = Duration::from_secs(30);

fn server_on(loader: &Arc<MockLoader>, model: ActiveModel, drain: Duration) -> Server {
    Server::from_active(model).with_switcher(Switcher::new(Arc::clone(loader) as _, drain))
}

fn input(model: &str, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        constrained: None,
        forced_literal: None,
        warm_up: false,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: model.to_owned(),
        tokens: vec![5, 6, 7],
        params: DecodeParams { max_tokens: Some(max_tokens), ..DecodeParams::default() },
    }
}

async fn ask(engine: &Engine, max_tokens: u32) -> EventStream {
    engine.submit(input(&engine.model_id(), max_tokens), RequestClass::Interactive).await.expect("admitted").1
}

/// Wait (scheduling turns, bounded) until `engine`'s published counters
/// show `in_flight` requests: the drain reads exactly these.
async fn until_in_flight(engine: &Engine, in_flight: u32) {
    for _ in 0..10_000 {
        let counters = engine.interval_counters();
        if counters.waiting + counters.running == in_flight {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the counters never showed {in_flight} in flight");
}

/// A request completes on `server`'s loaded model.
async fn serves(server: &Server) {
    let engine = server.active().engine.clone();
    let mut events = ask(&engine, 3).await;
    let completion = collect_completion(&mut events, Duration::from_secs(5)).await.expect("completes");
    assert_eq!(completion.tokens.len(), 3);
}

#[tokio::test]
async fn a_switch_closes_the_gate_waits_for_the_running_request_then_serves_the_new_model() {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = server_on(&loader, model, PATIENT);
    let old = server.active().engine.clone();
    let mut held = ask(&old, 4).await;
    gate.wait_entered();
    until_in_flight(&old, 1).await;

    let (started, task) = begin(&server, source("mock-b")).expect("begins");
    assert_eq!(started, SwitchStarted { from: "mock-a".into(), to: "mock-b".into() });
    assert_eq!(*server.status(), ModelStatus::Switching { from: "mock-a".into(), to: "mock-b".into() });
    assert_eq!(server.status().refusal(), Some("model_switching"), "the gate refuses new requests");

    // The drain waits on the held request: nothing is torn down or loaded
    // while it runs (it could not finish inside the 30 s window anyway).
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!task.is_finished(), "the switch waits for the running request");
    assert!(loader.loads().is_empty(), "no load before the drain");
    assert_eq!(server.active().engine.model_id(), "mock-a");

    gate.release();
    let completion = collect_completion(&mut held, Duration::from_secs(5)).await.expect("the held request completes");
    assert_eq!(completion.reason, FinishReason::Length, "it finished on its own, not cut");
    assert_eq!(completion.tokens.len(), 4);

    assert_eq!(task.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(server.active().engine.model_id(), "mock-b");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(server.status().refusal(), None);
    assert_eq!(loader.resident_at_load(), [0], "the old model was gone before the new one loaded");
    assert_eq!(loader.resident(), 1);
    serves(&server).await;
}

#[tokio::test]
async fn a_request_past_the_drain_window_is_cut_with_an_error_and_the_switch_completes() {
    let log = Arc::new(MemorySink::new());
    let _logging = tracing::subscriber::set_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())));
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = server_on(&loader, model, Duration::from_millis(50));
    let old = server.active().engine.clone();
    let mut straggler = ask(&old, 64).await;
    gate.wait_entered();
    until_in_flight(&old, 1).await;

    let (_, task) = begin(&server, source("mock-b")).expect("begins");
    // The drain window closes on the held request; the switch says so once
    // it has asked the old model to stop. Only then is the step let go —
    // the model thread then meets the shutdown before its next step.
    let timed_out = |line: &String| {
        serde_json::from_str::<serde_json::Value>(line).unwrap()["event_name"] == "ignis.model.switch_drain_timed_out"
    };
    for _ in 0..10_000 {
        if log.lines().iter().any(timed_out) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let line = log.lines().into_iter().find(timed_out).expect("the drain timed out");
    let line: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(line["attributes"]["waiting"].as_u64().unwrap() + line["attributes"]["running"].as_u64().unwrap(), 1);
    gate.release();

    let cut = collect_completion(&mut straggler, Duration::from_secs(5)).await.expect("the straggler is told");
    assert_eq!(cut.reason, FinishReason::Error, "a request cut by a switch ends with an error, not silently");
    assert!(cut.tokens.len() < 64);
    assert_eq!(task.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(server.active().engine.model_id(), "mock-b");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.resident_at_load(), [0]);
    // The old engine refuses what reaches it after the switch: "not now",
    // never a panic.
    let refused = old.submit(input("mock-a", 3), RequestClass::Interactive).await;
    assert!(matches!(refused, Err(ignis_core::SubmitError::Full)), "{refused:?}");
}

/// AC 9: a target refused before anything is torn down — here an artifact
/// path with no file, through the production loader — leaves the old model
/// serving, untouched.
#[tokio::test]
async fn a_target_with_no_file_is_refused_and_the_old_model_never_stops() {
    let loader = MockLoader::new();
    let options = match ignis_server::config::resolve(&[], |_| None).expect("the default config") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("no flags is a config"),
    };
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::new(ArtifactLoader::new(options)), PATIENT));
    let before = server.active();
    let target = ignis_server::ModelSource {
        artifact: "definitely/not/here/Qwen3.8-Flash-Next.ninfer".into(),
        model: "qwen3.8-flash-next".into(),
    };

    let (_, task) = begin(&server, target).expect("begins");
    let outcome = task.await.expect("the switch task");
    let SwitchOutcome::Refused { reason } = outcome else { panic!("refused, got {outcome:?}") };
    assert!(reason.contains("no such file"), "{reason}");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert!(Arc::ptr_eq(&before, &server.active()), "the loaded model was never replaced");
    assert_eq!(loader.resident(), 1, "nor torn down");
    serves(&server).await;
}

/// A target that fails on the GPU after the old model was torn down: the old
/// model is loaded again from its artifact and serves — a cold restart of
/// what was serving, never a stopped server.
#[tokio::test]
async fn a_load_that_fails_after_the_teardown_reloads_the_previous_model() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let server = server_on(&loader, loader.model("mock-a"), PATIENT);
    let old = server.active().engine.clone();

    let (_, task) = begin(&server, source("mock-broken")).expect("begins");
    let outcome = task.await.expect("the switch task");
    let SwitchOutcome::Restored { reason } = outcome else { panic!("restored, got {outcome:?}") };
    assert!(reason.contains("simulated kernel load error"), "{reason}");
    assert_eq!(server.active().engine.model_id(), "mock-a");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.loads(), ["mock-broken", "mock-a"]);
    assert_eq!(loader.resident_at_load(), [0, 0], "each load found the card empty");
    let stale = old.submit(input("mock-a", 3), RequestClass::Interactive).await;
    assert!(matches!(stale, Err(ignis_core::SubmitError::Full)), "the first engine was torn down: {stale:?}");
    serves(&server).await;
}

/// With nothing to reload, a failed switch leaves `Failed` standing — and a
/// refused switch from there must not reopen the gate over the dead engine —
/// until a switch that loads succeeds.
#[tokio::test]
async fn with_nothing_to_reload_a_failed_switch_stays_failed_until_a_switch_succeeds() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    loader.refuse("mock-refused");
    let mut first = loader.model("mock-a");
    first.source = None;
    let server = server_on(&loader, first, PATIENT);

    let (_, task) = begin(&server, source("mock-broken")).expect("begins");
    let outcome = task.await.expect("the switch task");
    assert!(matches!(&outcome, SwitchOutcome::Failed { reason } if reason.contains("nothing to reload")), "{outcome:?}");
    assert!(matches!(*server.status(), ModelStatus::Failed { .. }));
    assert_eq!(server.status().refusal(), Some("server_not_ready"));

    let (_, task) = begin(&server, source("mock-refused")).expect("a failed server takes a switch");
    assert!(matches!(task.await.expect("the switch task"), SwitchOutcome::Refused { .. }));
    assert!(matches!(*server.status(), ModelStatus::Failed { .. }), "still nothing serving");

    let (started, task) = begin(&server, source("mock-b")).expect("begins");
    assert_eq!(started.from, "mock-a", "the dead model is still the one named");
    assert_eq!(task.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(server.active().engine.model_id(), "mock-b");
    serves(&server).await;
}

#[tokio::test]
async fn a_second_switch_is_refused_naming_the_first_and_switches_are_not_queued() {
    let loader = MockLoader::new();
    let server = server_on(&loader, loader.model("mock-a"), PATIENT);
    let release = loader.hold_next_load();

    let (_, first) = begin(&server, source("mock-b")).expect("begins");
    let refused = begin(&server, source("mock-c")).map(|(started, _)| started);
    assert_eq!(
        refused,
        Err(SwitchRefusal::InProgress(SwitchStarted { from: "mock-a".into(), to: "mock-b".into() }))
    );
    release.send(()).expect("the held load is waiting");
    assert_eq!(first.await.expect("the switch task"), SwitchOutcome::Switched);

    let (started, second) = begin(&server, source("mock-c")).expect("the next switch begins once the first ended");
    assert_eq!(started.from, "mock-b");
    assert_eq!(second.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(server.active().engine.model_id(), "mock-c");
    assert_eq!(loader.loads(), ["mock-b", "mock-c"], "mock-c was loaded once, when asked again");
}

#[tokio::test]
async fn a_server_without_a_loader_or_still_warming_up_does_not_begin_a_switch() {
    let loader = MockLoader::new();
    let bare = Server::from_active(loader.model("mock-a"));
    assert_eq!(begin(&bare, source("mock-b")).map(|(started, _)| started), Err(SwitchRefusal::Unavailable));
    let warming = server_on(&loader, loader.model("mock-a"), PATIENT).with_warm_up();
    assert_eq!(begin(&warming, source("mock-b")).map(|(started, _)| started), Err(SwitchRefusal::WarmingUp));
    assert_eq!(*warming.status(), ModelStatus::WarmingUp, "a refused switch leaves the status alone");
}

/// The fork history names the old model's KV by match key: a switch empties
/// it, and the new model starts with none.
#[tokio::test]
async fn a_switch_starts_the_new_model_with_an_empty_fork_history() {
    let loader = MockLoader::new();
    let server = server_on(&loader, loader.model("mock-a"), PATIENT);
    let ends = ignis_server::reuse::run_ends(&input("mock-a", 3), &[2]);
    server.fork_history.lock().unwrap().record(&ends);
    assert_eq!(server.fork_history.lock().unwrap().longest_seen(&ends), Some(2));

    let (_, task) = begin(&server, source("mock-b")).expect("begins");
    assert_eq!(task.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(server.fork_history.lock().unwrap().longest_seen(&ends), None);
}
