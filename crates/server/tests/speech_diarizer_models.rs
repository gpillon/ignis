//! The diarizer in the catalog and in `GET /v1/models` (spec speech/08, Q116,
//! Q118, GitHub #361): `diarizer.model` resolved against the catalog, the
//! `speech.diarizer` member every status carries, the refusal of a model of
//! another kind at a `PATCH /v1/config`, a switch that never goes to the
//! diarizer, and its load and unload with the speech model.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use ignis_core::compute::ModelKind;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::download::catalog::{Catalog, CatalogEntry, CatalogFile, CatalogLayer};
use ignis_server::speech::cpu::CpuExecutor;
use ignis_server::speech::diarizing::CatalogDiarizer;
use ignis_server::speech::executor::{FakeDiarizer, FakeExecutor, SpeechExecutor};
use ignis_server::speech::{Speech, SpeechModels};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/mod.rs"]
mod support;
use support::realtime::{at_16_khz, live, next_event, send, session, until_true, update};
use support::speech;

/// The diarizer's id in these tests' catalog.
const DIARIZER: &str = "acme-diarizer";

fn catalog_with(dir: &std::path::Path, diarizers: &[&str]) -> SpeechModels {
    let mut models = speech::models(dir);
    let mut entries = models.catalog.entries().to_vec();
    for id in diarizers {
        entries.push(CatalogEntry {
            id: (*id).to_owned(),
            repo: "test/diarizer".to_owned(),
            revision: "0".repeat(40),
            artifact: format!("{id}.gguf"),
            card_class: None,
            family: None,
            aliases: Vec::new(),
            kind: Some(ModelKind::Diarizer),
            files: vec![CatalogFile { name: format!("{id}.gguf"), bytes: 4, sha256: "0".repeat(64) }],
            layer: CatalogLayer::Operator,
        });
    }
    models.catalog = Arc::new(Catalog::from_entries(entries));
    models
}

/// A server whose diarizer is the catalog's (a diarizer entry of our own, no
/// weights on disk), its host the real CPU's. `None` when this CPU cannot
/// run the speech model: the reasons are then the host's.
fn catalog_server(diarizer_model: &str) -> Option<(Server, std::path::PathBuf)> {
    if CpuExecutor::new().availability().is_err() {
        eprintln!("SKIP: this CPU lacks AVX2");
        return None;
    }
    let (server, dir) = speech::server(Arc::new(FakeExecutor::saying(&["hi"])), |_| {});
    let models = catalog_with(&dir, &[DIARIZER]);
    let diarizer = CatalogDiarizer::new(models.clone(), Arc::new(CpuExecutor::new()));
    let speech = Speech::new(Arc::new(FakeExecutor::saying(&["hi"])), Some(models)).with_diarizer(Arc::new(diarizer));
    let server = server
        .with_speech(Arc::new(speech))
        .with_diarizer_group(ignis_server::config::schema::DiarizerGroup { model: diarizer_model.to_owned(), ..Default::default() });
    Some((server, dir))
}

async fn get_models(server: &Server) -> Value {
    let response = server.app().oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap()).await.unwrap();
    speech::json(response).await
}

async fn diarizer_of(server: &Server) -> Value {
    get_models(server).await["speech"]["diarizer"].clone()
}

fn configured(server: Server) -> Server {
    let config = match resolve_with(&[], |_| None, &RealFiles).expect("the options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let server = server.with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    server
}

async fn patch(server: &Server, body: Value) -> (StatusCode, Value) {
    let request = Request::builder().method("PATCH").uri("/v1/config").header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    (response.status(), speech::json(response).await)
}

/// AC 14: a server with no diarizer says so in the `speech` object, beside
/// the speech model's own fields, which stay as they were; the diarizer is no
/// entry of `data`.
#[tokio::test]
async fn a_server_with_no_diarizer_says_so() {
    let (server, _dir) = speech::server(Arc::new(FakeExecutor::saying(&["hi"])), |_| {});
    let list = get_models(&server).await;
    assert_eq!(list["speech"]["available"], true);
    let diarizer = &list["speech"]["diarizer"];
    assert_eq!((diarizer["available"].clone(), diarizer["loaded"].clone()), (json!(false), json!(false)), "{diarizer}");
    assert!(diarizer["reason"].as_str().unwrap().contains("no diarizer is configured"), "{diarizer}");
    assert!(diarizer.get("id").is_none(), "no id: {diarizer}");
    assert_eq!(list["data"].as_array().unwrap().len(), 2, "the text model and the speech model: no request names the diarizer");
}

/// AC 14: the member says, as the speech object does, why the diarizer
/// cannot serve, in the order a person fixes it, and nothing downloads; with
/// its files on disk it is available and not loaded.
#[tokio::test]
async fn the_diarizer_member_is_available_or_says_why_not() {
    let Some((server, dir)) = catalog_server("auto") else { return };
    let before = std::fs::read_dir(&dir).unwrap().count();
    let missing = diarizer_of(&server).await;
    assert_eq!((missing["id"].as_str(), missing["available"].clone(), missing["loaded"].clone()), (Some(DIARIZER), json!(false), json!(false)), "{missing}");
    assert!(missing["reason"].as_str().unwrap().contains(&format!("model download {DIARIZER}")), "{missing}");
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), before, "asking downloaded nothing");

    std::fs::write(dir.join(format!("{DIARIZER}.gguf")), b"gguf").unwrap();
    let there = diarizer_of(&server).await;
    assert_eq!(there, json!({ "id": DIARIZER, "loaded": false, "available": true }));

    for (model, id, says) in [
        ("off", None, "turned off"),
        ("nemotron-x", Some("nemotron-x"), "not in the model catalog"),
        ("qwen3-asr-1.7b", Some("qwen3-asr-1.7b"), "not a diarizer"),
    ] {
        server.update_live(|live| live.diarizer.model = model.to_owned());
        let state = diarizer_of(&server).await;
        assert_eq!((state["id"].as_str(), state["available"].clone()), (id, json!(false)), "{model}: {state}");
        assert!(state["reason"].as_str().unwrap().contains(says), "{model}: {state}");
    }
    // A live change back is taken at once.
    server.update_live(|live| live.diarizer.model = "auto".to_owned());
    assert_eq!(diarizer_of(&server).await["available"], true);
}

/// AC 14: a `PATCH` naming a text, image or speech id as `diarizer.model` is
/// refused and nothing applies; one naming the catalog's diarizer, `off` or
/// `auto` is taken, and the member follows.
#[tokio::test]
async fn a_patch_of_another_kind_as_the_diarizer_is_refused() {
    let Some((server, dir)) = catalog_server("auto") else { return };
    std::fs::write(dir.join(format!("{DIARIZER}.gguf")), b"gguf").unwrap();
    let server = configured(server);
    // A text, an image and a speech id of the built-in catalog (the config's
    // own check).
    for id in ["qwen3.8-27b", "ignis-image-2.1-turbo", "qwen3-asr-1.7b", "qwen3-asr-0.6b"] {
        let (status, refused) = patch(&server, json!({ "diarizer": { "model": id } })).await;
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("config_invalid")), "{id}: {refused}");
        assert!(refused["error"]["message"].as_str().unwrap().contains("not a diarizer"), "{id}: {refused}");
    }
    assert_eq!(server.live().diarizer.model, "auto", "nothing applied");
    for model in [DIARIZER, "off", "auto"] {
        let (status, applied) = patch(&server, json!({ "diarizer": { "model": model } })).await;
        assert_eq!(status, StatusCode::OK, "{model}: {applied}");
        assert_eq!(server.live().diarizer.model, model);
    }
    assert_eq!(diarizer_of(&server).await["id"], DIARIZER);
}

/// The catalog the server runs with is the one a `PATCH` is checked against:
/// an entry of an operator's catalog that is not a diarizer is refused too, as
/// an operator's text entry is for `speech.model`.
#[tokio::test]
async fn a_patch_of_an_operators_speech_entry_as_the_diarizer_is_refused() {
    let (server, dir) = speech::server(Arc::new(FakeExecutor::saying(&["hi"])), |_| {});
    let mut models = speech::models(&dir);
    let mut entries = models.catalog.entries().to_vec();
    let mut operator = entries[0].clone();
    operator.id = "operator-asr".to_owned();
    entries.push(operator);
    models.catalog = Arc::new(Catalog::from_entries(entries));
    let server = configured(server.with_speech(Arc::new(Speech::new(Arc::new(FakeExecutor::saying(&["hi"])), Some(models)))));
    let (status, refused) = patch(&server, json!({ "diarizer": { "model": "operator-asr" } })).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("config_invalid")), "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("operator-asr"), "{refused}");
}

/// A diarizer entry of the catalog is no model a switch goes to or lists.
#[tokio::test]
async fn a_switch_never_goes_to_the_diarizer() {
    use ignis_server::model_switch::{CatalogModels, Switcher, Unswitchable};
    let dir = speech::temp_dir("switch-diarizer");
    std::fs::write(dir.join(format!("{DIARIZER}.gguf")), b"gguf").unwrap();
    let switcher = Switcher::new(Arc::new(NeverLoads), Duration::from_secs(1)).with_catalog(CatalogModels { catalog: catalog_with(&dir, &[DIARIZER]).catalog, dir: dir.clone() }, true);
    assert!(switcher.is_diarizer(DIARIZER) && !switcher.is_diarizer("qwen3-asr-1.7b"));
    assert_eq!(switcher.resolve(DIARIZER), Err(Unswitchable::Diarizer));
    assert!(switcher.known(DIARIZER).is_none());
    assert!(switcher.servable_models("other").iter().all(|servable| servable.id != DIARIZER));
}

/// A loader that loads nothing: the switch test needs a switcher, not a load.
struct NeverLoads;

impl ignis_server::model_switch::ModelLoader for NeverLoads {
    fn prepare(
        &self,
        _: &ignis_server::ModelSource,
        _: Option<&ignis_server::config::Config>,
    ) -> Result<Box<dyn ignis_server::model_switch::PreparedLoad>, String> {
        Err("this test never loads".to_owned())
    }
}

/// AC 15: the diarizer is not loaded until a session asks for speakers
/// (`loaded` in the member), a session that does not ask loads nothing of it,
/// and it is unloaded with the speech model at `speech.idle_unload`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_diarizer_loads_at_the_first_session_asking_and_unloads_with_the_speech_model() {
    let executor = Arc::new(FakeExecutor::saying(&["hi"]));
    let diarizer = Arc::new(FakeDiarizer::hearing(|_| None));
    let (server, _dir) = speech::server_with_diarizer(Arc::clone(&executor), Arc::clone(&diarizer), |group| group.idle_unload = 1, |_| {});
    let live = live(server.clone()).await;
    assert_eq!(diarizer_of(&server).await, json!({ "id": "fake-diarizer", "loaded": false, "available": true }));
    // A session that asks for no speakers loads the speech model only.
    let (mut plain, _) = session(&live).await;
    send(&mut plain, at_16_khz()).await;
    until_true("the speech model loaded", || executor.loaded.load(Ordering::SeqCst)).await;
    assert_eq!(diarizer_of(&server).await["loaded"], false);
    drop(plain);
    until_true("unloaded after the session ended", || !executor.loaded.load(Ordering::SeqCst)).await;
    // One that does loads it, and it stays while the session is open.
    let (mut socket, _) = session(&live).await;
    send(&mut socket, update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 } } }, "ignis": { "diarization": "live" } }))).await;
    assert_eq!(next_event(&mut socket).await["type"], "session.updated");
    until_true("the diarizer loaded", || diarizer.loaded.load(Ordering::SeqCst)).await;
    assert_eq!(diarizer_of(&server).await["loaded"], true);
    tokio::time::sleep(Duration::from_millis(1_800)).await;
    assert!(diarizer.loaded.load(Ordering::SeqCst) && executor.loaded.load(Ordering::SeqCst), "the open session holds both");
    drop(socket);
    until_true("the diarizer unloaded with the speech model", || !diarizer.loaded.load(Ordering::SeqCst)).await;
    assert!(!executor.loaded.load(Ordering::SeqCst));
    assert_eq!(diarizer_of(&server).await["loaded"], false);
}
