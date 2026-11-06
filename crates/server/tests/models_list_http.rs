//! `GET /v1/models` lists every servable model (spec model-switch/03, GitHub
//! #343) over the mock-backed router: the active model first with `active:
//! true`, then every catalog entry whose artifact is under a temporary
//! `download.path` and every known model whose artifact is a file — exactly
//! the ids the implicit switch resolves, read at each `GET`. A switch is held
//! inside its load by the mock loader (`support/switch.rs`), never by a timer
//! (ADR 0006).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::config::file::{Format, RealFiles};
use ignis_server::config::{resolve_with, ApiKey, Config, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::download::catalog::parse_operator;
use ignis_server::engine::Engine;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, ModelSource, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

const PATIENT: Duration = Duration::from_secs(30);

/// The image family's own id (spec image-model/05).
const IMAGE: &str = "ignis-image-2.1-turbo";

/// The built-in catalog's ids, in its order: listed only once on disk.
const BUILT_IN: [&str; 4] = ["qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", IMAGE];

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    send_keyed(app, method, uri, body, None).await
}

async fn send_keyed(app: &axum::Router, method: Method, uri: &str, body: Option<Value>, key: Option<&str>) -> (StatusCode, Value) {
    let mut request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// `GET /v1/models`, which must answer `200`.
async fn get_models(app: &axum::Router) -> Value {
    let (status, body) = send(app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

/// The ids `GET /v1/models` lists, in order.
async fn listed(app: &axum::Router) -> Vec<String> {
    ids(&get_models(app).await)
}

fn ids(list: &Value) -> Vec<String> {
    list["data"].as_array().expect("an array of entries").iter().map(|entry| entry["id"].as_str().unwrap().to_owned()).collect()
}

/// Wait until `server` reports `wanted`, read off the server; panics after
/// `within`.
async fn until(server: &Server, wanted: &str, within: Duration) {
    let deadline = Instant::now() + within;
    while server.status().as_str() != wanted {
        assert!(Instant::now() < deadline, "the status never read {wanted}: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 })
}

/// A directory of this test's own, empty.
fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-models-list-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file_in(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, b"artifact").unwrap();
    path
}

/// How much of an entry is under the directory.
#[derive(Clone, Copy)]
enum Disk {
    Complete,
    Partial,
    Absent,
}

/// An operator catalog with one entry per `(id, state)` — a 1-byte sidecar
/// and a 7-byte `<id>.ninfer` — and the files `state` says under a fresh
/// `download.path`.
fn catalog_in(tag: &str, entries: &[(&str, Disk)]) -> CatalogModels {
    let dir = fresh_dir(tag);
    let zeros = "0".repeat(64);
    let mut yaml = String::from("models:\n");
    for (id, disk) in entries {
        yaml.push_str(&format!(
            "  - id: {id}\n    repo: acme/{id}\n    revision: v1\n    artifact: {id}.ninfer\n    card_class: 32\n    files:\n      - {{ name: {id}.ninfer.graft.json, bytes: 1, sha256: \"{zeros}\" }}\n      - {{ name: {id}.ninfer, bytes: 7, sha256: \"{zeros}\" }}\n"
        ));
        match disk {
            Disk::Complete => {
                std::fs::write(dir.join(format!("{id}.ninfer.graft.json")), b"x").unwrap();
                std::fs::write(dir.join(format!("{id}.ninfer")), b"1234567").unwrap();
            }
            Disk::Partial => std::fs::write(dir.join(format!("{id}.ninfer.graft.json")), b"x").unwrap(),
            Disk::Absent => {}
        }
    }
    let catalog = parse_operator(&yaml, Format::Yaml, "test").expect("a valid catalog");
    CatalogModels { catalog: Arc::new(catalog), dir }
}

/// A server on `mock-a`, loaded from `start` (a file), with the catalog
/// `models` and the known models `known` (the start among them), `implicit`
/// being `switch.allow_implicit` as `main` applies it.
fn server_with(loader: &Arc<MockLoader>, start: &Path, models: CatalogModels, known: &[(&str, &Path)], implicit: bool) -> Server {
    let mut table: BTreeMap<String, PathBuf> = known.iter().map(|(id, path)| (id.to_string(), path.to_path_buf())).collect();
    table.insert("mock-a".to_owned(), start.to_path_buf());
    let implicit_table = if implicit { table.clone() } else { BTreeMap::new() };
    let switcher = Switcher::new(Arc::clone(loader) as _, PATIENT)
        .with_known_models(implicit_table)
        .with_listed_models(table)
        .with_catalog(models, implicit);
    let server = Server::from_active(loader.model("mock-a")).with_switcher(switcher);
    server.update_active(|model| model.source = Some(ModelSource { artifact: start.to_path_buf(), model: "mock-a".to_owned() }));
    server
}

/// The configuration `argv` starts with.
fn started(argv: &[&str]) -> Config {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    match resolve_with(&argv, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// AC 1, 2, 7: the active model, then each catalog entry on disk in catalog
/// order, then each known model whose artifact is a file in id order — each
/// once, the active one not repeated; partial and absent entries, a known
/// model with no file and unknown ids are not listed; `kind` on another
/// entry only for a family's own id, `max_model_len`, `decode_lanes` and
/// `adapted` on the first only, no path anywhere.
#[tokio::test]
async fn the_list_is_the_active_model_then_the_catalog_on_disk_then_the_known_models_with_a_file() {
    let loader = MockLoader::new();
    let models = catalog_in("order", &[("acme-zed", Disk::Complete), ("acme-part", Disk::Partial), ("acme-none", Disk::Absent), ("acme-alpha", Disk::Complete), ("mock-a", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("order-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let fn_file = file_in(&elsewhere, "fn.ninfer");
    let k_file = file_in(&elsewhere, "mock-k.ninfer");
    let gone = elsewhere.join("gone.ninfer");
    let known = [("qwen3.8-flash-next", fn_file.as_path()), ("mock-k", k_file.as_path()), ("mock-gone", gone.as_path()), ("acme-alpha", k_file.as_path())];
    let app = server_with(&loader, &start, models, &known, true).app();

    let list = get_models(&app).await;
    // The built-in catalog comes first: the Flash-Next entry is a catalog id,
    // served from the known models' file, which wins for its id.
    assert_eq!(ids(&list), ["mock-a", "qwen3.8-flash-next", "acme-zed", "acme-alpha", "mock-k"], "{list}");
    let entries = list["data"].as_array().unwrap();
    assert_eq!(entries[0]["active"], true);
    assert_eq!(entries[0]["kind"], "text");
    assert!(entries[0]["max_model_len"].is_u64(), "{list}");
    assert!(entries[0]["decode_lanes"].is_u64() && entries[0]["adapted"].is_string(), "{list}");
    for entry in &entries[1..] {
        assert_eq!(entry["active"], false, "{entry}");
        assert_eq!((entry["object"].as_str(), entry["owned_by"].as_str()), (Some("model"), Some("ignis")), "{entry}");
        for field in ["max_model_len", "decode_lanes", "adapted", "prefill_chunk_solo"] {
            assert!(entry.get(field).is_none(), "{field}: {entry}");
        }
        let kind = if entry["id"] == "qwen3.8-flash-next" { Some("text") } else { None };
        assert_eq!(entry.get("kind").and_then(Value::as_str), kind, "{entry}");
    }
    let text = list.to_string();
    assert!(!text.contains(".ninfer") && !text.contains(&*dir.to_string_lossy()), "no entry carries a path: {list}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// #340 (spec 27b-16gb/05 AC 7): the loaded text model's entry says the
/// lanes it decodes on -- its engine's admission cap, the load's lanes -- and
/// every adaptation its load took, `""` when it took none, so a client reads
/// both without the start log. The image model, which has no engine, has
/// neither.
#[tokio::test]
async fn the_active_entry_says_the_lanes_it_decodes_on_and_what_its_load_adapted() {
    let loader = MockLoader::new();
    // Nothing adapted: the mock's lanes, and an empty `adapted`, present.
    let server = Server::from_active(loader.model("mock-a"));
    let list = get_models(&server.app()).await;
    assert_eq!(list["data"][0]["decode_lanes"], SchedulerConfig::default().max_in_flight as u64, "{list}");
    assert_eq!(list["data"][0]["adapted"], "", "{list}");

    // The context adapted: the steps as the load's plan event spells them.
    let context = "max_context 262144 -> 65536 (adapt_context_step 65536, adapt_spare_tokens 65536)";
    server.update_active(|model| model.adapted = context.to_owned());
    let list = get_models(&server.app()).await;
    assert_eq!(list["data"][0]["adapted"], context, "{list}");

    // The lanes adapted: a load whose scheduler admits one request at a time.
    let lanes = "decode_lanes 2 -> 1, max_context 262144 -> 196608 (adapt_context_step 65536, one lane)";
    let one_lane = ConcreteScheduler::with_config(
        SchedulerConfig { model: "mock-a".into(), max_in_flight: 1, max_prefill_batch: 1, ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let model = ActiveModel::new(Engine::new(Box::new(one_lane)), Arc::new(SimpleTemplateProvider)).with_adapted(lanes.to_owned());
    let list = get_models(&Server::from_active(model).app()).await;
    assert_eq!((&list["data"][0]["decode_lanes"], &list["data"][0]["adapted"]), (&json!(1), &json!(lanes)), "{list}");

    // GitHub #489: and the solo width the load runs, a number -- the prefill
    // chunk itself where no wider one is planned.
    assert_eq!(list["data"][0]["prefill_chunk_solo"], SchedulerConfig::default().serving_chunk_tokens, "{list}");
    let solo = ConcreteScheduler::with_config(
        SchedulerConfig { model: "mock-a".into(), serving_chunk_tokens: 2048, solo_chunk_tokens: Some(8192), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let model = ActiveModel::new(Engine::new(Box::new(solo)), Arc::new(SimpleTemplateProvider));
    let list = get_models(&Server::from_active(model).app()).await;
    assert_eq!(list["data"][0]["prefill_chunk_solo"], 8192, "{list}");

    // The image model: none of them.
    let list = get_models(&Server::from_active(loader.model(IMAGE)).app()).await;
    assert_eq!(list["data"][0]["kind"], "image", "{list}");
    assert!(list["data"][0].get("decode_lanes").is_none() && list["data"][0].get("adapted").is_none(), "{list}");
    assert!(list["data"][0].get("prefill_chunk_solo").is_none(), "{list}");
}

/// AC 2: with `switch.allow_implicit` off only the active model is listed,
/// whatever is on disk.
#[tokio::test]
async fn with_implicit_switching_off_only_the_active_model_is_listed() {
    let loader = MockLoader::new();
    let models = catalog_in("off", &[("acme-full", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("off-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let k_file = file_in(&elsewhere, "mock-k.ninfer");
    let app = server_with(&loader, &start, models, &[("mock-k", &k_file)], false).app();
    let list = get_models(&app).await;
    assert_eq!(ids(&list), ["mock-a"], "{list}");
    assert_eq!(list["data"][0]["active"], true);
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 3: an id is listed exactly when `Switcher::known` resolves it to a
/// file; a chat request naming a listed text model is served, after an
/// implicit switch, and one naming an unlisted id is `404 model_not_found`.
#[tokio::test]
async fn an_id_is_listed_if_and_only_if_a_request_naming_it_is_served() {
    let loader = MockLoader::new();
    let models = catalog_in("iff", &[("acme-full", Disk::Complete), ("acme-part", Disk::Partial), ("acme-none", Disk::Absent)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("iff-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let k_file = file_in(&elsewhere, "mock-k.ninfer");
    let gone = elsewhere.join("gone.ninfer");
    let server = server_with(&loader, &start, models, &[("mock-k", &k_file), ("mock-gone", &gone)], true);
    let app = server.app();
    let switcher = server.switcher.as_ref().unwrap();

    let shown = listed(&app).await;
    let candidates = BUILT_IN.iter().chain(&["acme-full", "acme-part", "acme-none", "mock-k", "mock-gone", "mock-zz", "qwen"]);
    for id in candidates {
        let servable = switcher.known(id).is_some_and(|path| path.is_file());
        assert_eq!(shown.iter().any(|shown| shown == id), servable, "{id}: listed {shown:?}");
    }

    for id in &shown[1..] {
        let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(id))).await;
        assert_eq!(status, StatusCode::OK, "{id}: {completion}");
        assert_eq!(completion["model"], json!(id));
    }
    assert_eq!(loader.loads(), shown[1..], "each listed model was switched to");
    // A known model whose file is gone is refused by name too, before any
    // switch: it is not servable, so naming it never tears the model down.
    let loads = loader.loads();
    for id in ["acme-part", "acme-none", "mock-gone", "mock-zz"] {
        let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(id))).await;
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("model_not_found")), "{id}: {refused}");
    }
    assert_eq!(loader.loads(), loads, "no switch began for an unlisted id");
    // Put back, it is listed and served at the next request.
    std::fs::write(&gone, b"back").unwrap();
    assert!(listed(&app).await.contains(&"mock-gone".to_owned()));
    let (status, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-gone"))).await;
    assert_eq!((status, completion["model"].as_str()), (StatusCode::OK, Some("mock-gone")), "{completion}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 4: an artifact landing under `download.path` on a running server is
/// listed by the next `GET`, and gone once it is removed.
#[tokio::test]
async fn a_download_landing_on_a_running_server_is_listed_at_the_next_get() {
    let loader = MockLoader::new();
    let models = catalog_in("landing", &[("acme-new", Disk::Partial)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("landing-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let app = server_with(&loader, &start, models, &[], true).app();
    assert_eq!(listed(&app).await, ["mock-a"]);
    std::fs::write(dir.join("acme-new.ninfer"), b"1234567").unwrap();
    assert_eq!(listed(&app).await, ["mock-a", "acme-new"]);
    std::fs::remove_file(dir.join("acme-new.ninfer")).unwrap();
    assert_eq!(listed(&app).await, ["mock-a"]);
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 5: a live `PATCH /v1/config` of `switch.allow_implicit` (on, off, on)
/// and of `switch.known_models` changes the next `GET`'s list.
#[tokio::test]
async fn a_live_patch_of_the_switch_knobs_changes_the_next_list() {
    let loader = MockLoader::new();
    let models = catalog_in("patch", &[("acme-full", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("patch-elsewhere");
    let k_file = file_in(&elsewhere, "mock-k.ninfer");
    let config = started(&[]);
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true))
        .with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    let app = server.app();
    assert_eq!(listed(&app).await, ["mock-a", "acme-full"]);

    let patch = |body: Value| {
        let app = app.clone();
        async move {
            let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
            assert_eq!(status, StatusCode::OK, "{applied}");
        }
    };
    patch(json!({ "switch": { "allow_implicit": false } })).await;
    assert_eq!(listed(&app).await, ["mock-a"]);
    patch(json!({ "switch": { "allow_implicit": true } })).await;
    assert_eq!(listed(&app).await, ["mock-a", "acme-full"]);
    patch(json!({ "switch": { "known_models": { "mock-k": k_file.to_string_lossy() } } })).await;
    assert_eq!(listed(&app).await, ["mock-a", "acme-full", "mock-k"]);
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// Every entry but the first has `active: false`, the first `true` and the
/// id the server reports as its active model.
fn assert_first_is_active(server: &Server, list: &Value) {
    let entries = list["data"].as_array().expect("entries");
    assert_eq!(entries[0]["id"], json!(server.active().model_id()), "{list}");
    assert_eq!(entries[0]["active"], true, "{list}");
    assert!(entries[1..].iter().all(|entry| entry["active"] == false), "{list}");
}

/// AC 6: `data[0]` is the active model, `active: true`, while switching —
/// still the old model — and once the switch failed; the rest of the list
/// is listed meanwhile, the switch's target among it.
#[tokio::test]
async fn data_0_is_the_active_model_while_switching_and_once_failed() {
    let loader = MockLoader::new();
    loader.break_load("mock-b");
    let models = catalog_in("statuses", &[("mock-b", Disk::Complete), ("mock-c", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("statuses-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let server = server_with(&loader, &start, models, &[], true);
    let app = server.app();
    let release = loader.hold_next_load();
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-c" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until(&server, "switching", PATIENT).await;
    let list = get_models(&app).await;
    assert_eq!(list["status"], "switching", "{list}");
    assert_eq!(ids(&list), ["mock-a", "mock-b", "mock-c"], "not filtered by switchable: {list}");
    assert_first_is_active(&server, &list);
    release.send(()).expect("the held load is waiting");
    until(&server, "serving", PATIENT).await;
    assert_eq!(listed(&app).await, ["mock-c", "mock-b", "mock-a"], "the catalog, then the start model: a way back");

    // A failed switch with no way back: nothing serves.
    server.update_active(|model| model.source = None);
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until(&server, "failed", PATIENT).await;
    let list = get_models(&app).await;
    assert_eq!(list["status"], "failed", "{list}");
    assert_first_is_active(&server, &list);
    assert!(ids(&list).len() > 1, "the way out is listed: {list}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 6 and 9: while unloaded and loading `data[0]` is the active model with
/// the servable rest after it; polling the list neither resets the idle
/// clock nor loads the unloaded model (spec model-switch/02 AC 5).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn polling_the_list_while_idle_neither_keeps_nor_loads_the_model() {
    let loader = MockLoader::new();
    let models = catalog_in("idle", &[("acme-full", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("idle-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let server = server_with(&loader, &start, models, &[], true).with_idle_unload(Duration::from_secs(1));
    let app = server.app();
    let _clock = server.spawn_idle_clock(Duration::from_millis(50)).expect("a clock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while server.status().as_str() != "unloaded" {
        assert!(Instant::now() < deadline, "polling kept the model: {:?}", server.status());
        assert_eq!(listed(&app).await, ["mock-a", "acme-full"]);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for _ in 0..5 {
        let list = get_models(&app).await;
        assert_eq!((list["status"].as_str(), ids(&list)), (Some("unloaded"), vec!["mock-a".to_owned(), "acme-full".to_owned()]), "{list}");
        assert_first_is_active(&server, &list);
    }
    assert!(loader.loads().is_empty(), "a GET loads nothing: {:?}", loader.loads());

    server.update_live(|live| live.idle_unload = Duration::ZERO);
    let release = loader.hold_next_load();
    let chat_app = app.clone();
    let task = tokio::spawn(async move { send(&chat_app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await });
    until(&server, "loading", PATIENT).await;
    let list = get_models(&app).await;
    assert_eq!(list["status"], "loading", "{list}");
    assert_eq!(ids(&list), ["mock-a", "acme-full"], "{list}");
    assert_first_is_active(&server, &list);
    release.send(()).expect("the held load is waiting");
    let (status, completion) = task.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{completion}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 8: a `GET` reads no artifact — the loader is never asked a kind.
#[tokio::test]
async fn the_list_opens_no_artifact() {
    let loader = MockLoader::new();
    let models = catalog_in("no-open", &[("acme-full", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("no-open-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let image = file_in(&elsewhere, "image.ninfer");
    let app = server_with(&loader, &start, models, &[(IMAGE, &image)], true).app();
    assert_eq!(listed(&app).await, ["mock-a", IMAGE, "acme-full"], "the image model is a built-in catalog entry");
    assert_eq!(loader.kind_of_calls(), 0, "the route asked the loader a kind");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 9: a missing API key is still `401`, the warm-up gate still `503
/// server_not_ready`, with servable models to list.
#[tokio::test]
async fn the_key_and_the_warm_up_gate_still_guard_the_list() {
    let loader = MockLoader::new();
    let models = catalog_in("gates", &[("acme-full", Disk::Complete)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("gates-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let server = server_with(&loader, &start, models, &[], true).with_api_key(ApiKey::new("sk-test"));
    let app = server.app();
    let (status, _) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, list) = send_keyed(&app, Method::GET, "/v1/models", None, Some("sk-test")).await;
    assert_eq!((status, ids(&list)), (StatusCode::OK, vec!["mock-a".to_owned(), "acme-full".to_owned()]));

    let models = catalog_in("gates-warm", &[("acme-full", Disk::Complete)]);
    let warm_dir = models.dir.clone();
    let app = server_with(&loader, &start, models, &[], true).with_warm_up().app();
    let (status, refused) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("server_not_ready")), "{refused}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(warm_dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 10: with a text model loaded and the image model on disk, the image
/// model is listed, with its kind; a chat request naming it is `409
/// model_kind_mismatch` and starts no switch.
#[tokio::test]
async fn the_image_model_on_disk_is_listed_and_a_chat_request_naming_it_is_refused() {
    let loader = MockLoader::new();
    let models = catalog_in("image", &[("acme-none", Disk::Absent)]);
    let dir = models.dir.clone();
    let elsewhere = fresh_dir("image-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer");
    let image = file_in(&elsewhere, "image.ninfer");
    let server = server_with(&loader, &start, models, &[(IMAGE, &image)], true);
    let app = server.app();
    let list = get_models(&app).await;
    assert_eq!(ids(&list), ["mock-a", IMAGE], "{list}");
    assert_eq!(list["data"][1]["kind"], "image", "{list}");
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat(IMAGE))).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::CONFLICT, Some("model_kind_mismatch")), "{refused}");
    assert!(loader.loads().is_empty(), "{:?}", loader.loads());
    assert!(server.switcher.as_ref().unwrap().last_switch().is_none(), "no switch began");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}
