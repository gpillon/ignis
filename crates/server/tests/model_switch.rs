//! The model switch's orchestration (spec model-switch/01, GitHub #305),
//! driven directly — not through HTTP — over two or more `MockCompute`
//! models: the gate, the drain, the teardown before the load, and every way
//! a switch can fail without stopping the server — and the implicit switch a
//! request's own `model` begins (§Implicit switch). No GPU, and no wall-clock
//! timing proves anything (ADR 0006): a request is held mid-decode with
//! spec server/05's gate, a load with the mock loader's own hold.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use ignis_core::{DecodeParams, FinishReason, RequestClass, RequestInput};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::engine::{collect_completion, Engine, EventStream};
use ignis_server::config::file::Format;
use ignis_server::download::catalog::parse_operator;
use ignis_server::model_switch::{
    begin, implicit_switch, known_models, ArtifactLoader, CatalogModels, ImplicitRefusal, SwitchOutcome, SwitchRefusal,
    SwitchStarted, Switcher,
};
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

    // The drain waits on the held request: had it not, the teardown would
    // have met the request still running and cut it (the next test), and
    // the load would have found the old model resident.
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

/// A request reads one model throughout (`Server::pinned`): a switch that
/// lands mid-request does not hand it the next model's engine — the pinned
/// copy keeps the old one, whose engine, torn down, answers "not now".
#[tokio::test]
async fn a_pinned_request_keeps_the_model_it_began_on_across_a_switch() {
    let loader = MockLoader::new();
    let server = server_on(&loader, loader.model("mock-a"), PATIENT);
    let pinned = server.pinned();

    let (_, task) = begin(&server, source("mock-b")).expect("begins");
    assert_eq!(task.await.expect("the switch task"), SwitchOutcome::Switched);
    assert_eq!(server.active().engine.model_id(), "mock-b");
    assert_eq!(pinned.active().engine.model_id(), "mock-a", "the pinned request still sees its own model");
    let refused = pinned.active().engine.submit(input("mock-a", 3), RequestClass::Interactive).await;
    assert!(matches!(refused, Err(ignis_core::SubmitError::Full)), "{refused:?}");
    assert_eq!(*pinned.status(), ModelStatus::Serving, "the status is the server's, shared");
}

// ── the implicit switch: a request's own `model` (spec §Implicit switch) ──

/// The known-models table naming each of `ids` at the artifact
/// [`source`] gives it.
fn known(ids: &[&str]) -> BTreeMap<String, PathBuf> {
    ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect()
}

/// [`server_on`], with `ids` switchable by a request's `model`.
fn implicit_server_on(loader: &Arc<MockLoader>, model: ActiveModel, ids: &[&str]) -> Server {
    Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(ids)))
}

/// Wait (scheduling turns, bounded) until `server` reports a switch.
async fn until_switching(server: &Server) {
    for _ in 0..10_000 {
        if matches!(*server.status(), ModelStatus::Switching { .. }) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("no switch began");
}

/// AC 17, 21: naming a known model switches to it, and the call returns once
/// it serves — the request is then served on it, not handed a `202`.
#[tokio::test]
async fn naming_a_known_model_switches_to_it_and_returns_once_it_serves() {
    let loader = MockLoader::new();
    let server = implicit_server_on(&loader, loader.model("mock-a"), &["mock-a", "mock-b"]);

    assert_eq!(implicit_switch(&server, Some("mock-b")).await, Ok(()));
    assert_eq!(server.active().engine.model_id(), "mock-b");
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(loader.loads(), ["mock-b"]);
    assert_eq!(loader.resident_at_load(), [0], "the same teardown-then-load as an explicit switch");
    serves(&server).await;

    assert_eq!(implicit_switch(&server, Some("mock-a")).await, Ok(()), "and back");
    assert_eq!(server.active().engine.model_id(), "mock-a");
    assert_eq!(loader.loads(), ["mock-b", "mock-a"]);
}

/// AC 19: nothing named, the loaded model, a model nobody listed, or any
/// model with implicit switching off (no table) — nothing switches, and the
/// request's own check answers as it always did.
#[tokio::test]
async fn naming_nothing_the_loaded_model_or_an_unlisted_one_switches_nothing() {
    let loader = MockLoader::new();
    let server = implicit_server_on(&loader, loader.model("mock-a"), &["mock-a", "mock-b"]);
    let before = server.active();
    for requested in [None, Some(""), Some("mock-a"), Some("mock-z")] {
        assert_eq!(implicit_switch(&server, requested).await, Ok(()), "{requested:?}");
    }
    let off = server_on(&loader, loader.model("mock-a"), PATIENT);
    assert_eq!(implicit_switch(&off, Some("mock-b")).await, Ok(()), "no known models: switching is off");
    let bare = Server::from_active(loader.model("mock-a"));
    assert_eq!(implicit_switch(&bare, Some("mock-b")).await, Ok(()), "no switcher at all");

    assert!(Arc::ptr_eq(&before, &server.active()));
    assert_eq!(*server.status(), ModelStatus::Serving);
    assert_eq!(*off.status(), ModelStatus::Serving);
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
}

/// AC 20 and §Implicit switch's ordering: the gate closes the moment the
/// mismatch is seen, the triggering call waits out the drain of what was
/// already running, and a second request naming a model meanwhile — any
/// model, the same target included — is told a switch is under way rather
/// than queueing or retargeting one.
#[tokio::test]
async fn the_gate_closes_at_once_the_drain_runs_first_and_a_second_request_is_told_a_switch_runs() {
    let loader = MockLoader::new();
    let (model, gated, gate) = loader.gated_model("mock-a");
    gated.arm();
    drop(gated);
    let server = implicit_server_on(&loader, model, &["mock-a", "mock-b", "mock-c"]);
    let old = server.active().engine.clone();
    let mut held = ask(&old, 4).await;
    gate.wait_entered();
    until_in_flight(&old, 1).await;

    let trigger = tokio::spawn({
        let server = server.clone();
        async move { implicit_switch(&server, Some("mock-b")).await }
    });
    until_switching(&server).await;
    assert_eq!(*server.status(), ModelStatus::Switching { from: "mock-a".into(), to: "mock-b".into() });
    let running = SwitchStarted { from: "mock-a".into(), to: "mock-b".into() };
    for other in ["mock-c", "mock-b"] {
        assert_eq!(
            implicit_switch(&server, Some(other)).await,
            Err(ImplicitRefusal::Switching(running.clone())),
            "{other}"
        );
    }
    assert!(!trigger.is_finished(), "the trigger waits for the drain, which waits for the held request");
    assert!(loader.loads().is_empty(), "nothing loads before the old model's requests are done");

    gate.release();
    let completion = collect_completion(&mut held, Duration::from_secs(5)).await.expect("the held request completes");
    assert_eq!(completion.reason, FinishReason::Length, "drained, not cut");
    assert_eq!(trigger.await.expect("the trigger task"), Ok(()));
    assert_eq!(server.active().engine.model_id(), "mock-b");
    assert_eq!(loader.loads(), ["mock-b"], "one switch, not one per request that named a model");
}

/// A request is never served on a model it did not name: a switch that did
/// not land its target refuses it with the switch's own reason, whether the
/// target was refused before anything stopped or failed on the GPU and the
/// previous model was reloaded.
#[tokio::test]
async fn a_switch_that_does_not_land_refuses_the_request_with_its_reason() {
    let loader = MockLoader::new();
    loader.refuse("mock-refused");
    loader.break_load("mock-broken");
    let server = implicit_server_on(&loader, loader.model("mock-a"), &["mock-a", "mock-refused", "mock-broken"]);

    let Err(ImplicitRefusal::Failed { to, reason }) = implicit_switch(&server, Some("mock-refused")).await else {
        panic!("refused at prepare");
    };
    assert_eq!(to, "mock-refused");
    assert!(reason.contains("refused at prepare"), "{reason}");
    assert_eq!(server.active().engine.model_id(), "mock-a");

    let Err(ImplicitRefusal::Failed { to, reason }) = implicit_switch(&server, Some("mock-broken")).await else {
        panic!("failed on the load");
    };
    assert_eq!(to, "mock-broken");
    assert!(reason.contains("simulated kernel load error"), "{reason}");
    assert_eq!(server.active().engine.model_id(), "mock-a", "the previous model was reloaded");
    assert_eq!(*server.status(), ModelStatus::Serving);
    serves(&server).await;
}

/// The table a server starts with: the operator's entries and the model it
/// started on, whose own id and artifact win over an entry naming the same
/// id — a switch back reloads what was actually loaded.
// ── the catalog's models (spec model-download/02 §Known models) ─────────────

/// A catalog listing each of `ids` with its artifact `<id>.ninfer`, under a
/// fresh directory of its own.
fn catalog_models(tag: &str, ids: &[&str]) -> CatalogModels {
    let dir = std::env::temp_dir().join(format!("ignis-switch-catalog-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let zeros = "0".repeat(64);
    let entries: String = ids
        .iter()
        .map(|id| {
            format!(
                "  - id: {id}\n    repo: acme/{id}\n    revision: v1\n    artifact: {id}.ninfer\n    files:\n      - {{ name: {id}.ninfer.graft.json, bytes: 1, sha256: \"{zeros}\" }}\n      - {{ name: {id}.ninfer, bytes: 7, sha256: \"{zeros}\" }}\n"
            )
        })
        .collect();
    let catalog = parse_operator(&format!("models:\n{entries}"), Format::Yaml, "test").expect("a valid catalog");
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// Spec model-download/02 AC 14: a catalog entry is switchable when its
/// artifact is on disk **at the time of the request** — a download landing
/// in a running server's directory needs no restart — and refused as an
/// unknown model before, without a download.
#[tokio::test]
async fn a_catalog_entry_whose_artifact_is_on_disk_when_named_is_switchable() {
    let loader = MockLoader::new();
    let models = catalog_models("on-disk", &["mock-cat"]);
    let artifact = models.dir.join("mock-cat.ninfer");
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a"])).with_catalog(models, true));

    assert_eq!(implicit_switch(&server, Some("mock-cat")).await, Ok(()));
    assert!(loader.loads().is_empty(), "not on disk: nothing switches, the request's own check refuses it");
    assert_eq!(server.active().engine.model_id(), "mock-a");

    std::fs::write(&artifact, b"fetched").unwrap();
    assert_eq!(implicit_switch(&server, Some("mock-cat")).await, Ok(()));
    assert_eq!(server.active().engine.model_id(), "mock-cat");
    assert_eq!(loader.prepared_artifacts(), [artifact.clone()], "loaded from the download directory");
    assert!(!server.switcher.as_ref().unwrap().known_models().contains_key("mock-cat"), "derived, not added to the list");
    let _ = std::fs::remove_dir_all(artifact.parent().unwrap());
}

/// AC 14: an explicit known-models entry wins for its id; with implicit
/// switching off the catalog is not consulted, and a live change turning
/// it on consults it again.
#[tokio::test]
async fn an_explicit_known_model_wins_and_switching_off_leaves_the_catalog_out() {
    let loader = MockLoader::new();
    let models = catalog_models("explicit", &["mock-b", "mock-cat"]);
    for id in ["mock-b", "mock-cat"] {
        std::fs::write(models.dir.join(format!("{id}.ninfer")), b"fetched").unwrap();
    }
    let dir = models.dir.clone();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_known_models(known(&["mock-a", "mock-b"])).with_catalog(models, false));

    assert_eq!(implicit_switch(&server, Some("mock-b")).await, Ok(()));
    assert_eq!(loader.prepared_artifacts(), [source("mock-b").artifact], "the operator's path, not the catalog's");

    assert_eq!(implicit_switch(&server, Some("mock-cat")).await, Ok(()));
    assert_eq!(loader.loads(), ["mock-b"], "implicit switching off: the catalog is not consulted");

    let switcher = server.switcher.as_ref().unwrap();
    switcher.set_knobs(PATIENT, switcher.known_models(), true);
    assert_eq!(implicit_switch(&server, Some("mock-cat")).await, Ok(()));
    assert_eq!(loader.loads(), ["mock-b", "mock-cat"]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_model_the_server_starts_on_is_always_known() {
    let named = BTreeMap::from([
        ("mock-b".to_owned(), PathBuf::from("b.ninfer")),
        ("mock-a".to_owned(), PathBuf::from("elsewhere/a.ninfer")),
    ]);
    let table = known_models(&named, Some(&source("mock-a")));
    assert_eq!(
        table,
        BTreeMap::from([
            ("mock-a".to_owned(), source("mock-a").artifact),
            ("mock-b".to_owned(), PathBuf::from("b.ninfer")),
        ])
    );
    assert_eq!(known_models(&BTreeMap::new(), Some(&source("mock-c"))), known(&["mock-c"]), "no flag needed");
    assert_eq!(known_models(&named, None), named, "a start with no artifact adds nothing");
}
