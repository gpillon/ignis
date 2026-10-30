//! What the Playground's Server view needs of the model routes (spec
//! playground/05, GitHub #316), over the mock-backed router: `GET
//! /v1/models/catalog` with a temporary `download.path` holding complete,
//! partial and absent entries; `POST /v1/models/switch` with `model` alone,
//! a known model or a catalog entry on disk, with implicit switching on or
//! off; and `last_switch` on `GET /v1/models` for each way a switch or a
//! reload ends. A switch is held inside its load by the mock loader
//! (`support/switch.rs`), never by a timer (ADR 0006).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::{Format, RealFiles};
use ignis_server::config::{resolve_with, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::download::catalog::parse_operator;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::{ModelSource, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

const PATIENT: Duration = Duration::from_secs(30);

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// `GET /v1/models` until its `status` is `wanted` (scheduling turns, bounded).
async fn until_status(app: &axum::Router, wanted: &str) -> Value {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == wanted {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported {wanted}");
}

/// A directory of this test's own, empty.
fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-catalog-http-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// How much of an entry is under the directory.
#[derive(Clone, Copy)]
enum Disk {
    Complete,
    Partial,
    Absent,
}

/// An operator catalog with one entry per `(id, state)` — each a 1-byte
/// sidecar and a 7-byte `<id>.ninfer` — and the files `state` says under a
/// fresh `download.path`.
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

/// A server on `mock-a`, loaded from `start` (a file), switching with the
/// catalog `models` and the operator-listed `listed`. `implicit` is
/// `switch.allow_implicit`, applied as `main` does: it empties the table an
/// implicit switch reads and the catalog's gate, and leaves the listed models
/// for an explicit switch by id.
fn server_with(loader: &Arc<MockLoader>, start: &Path, models: CatalogModels, listed: &[(&str, &Path)], implicit: bool) -> Server {
    let mut table: BTreeMap<String, PathBuf> = listed.iter().map(|(id, path)| (id.to_string(), path.to_path_buf())).collect();
    table.insert("mock-a".to_owned(), start.to_path_buf());
    let known = if implicit { table.clone() } else { BTreeMap::new() };
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known).with_listed_models(table).with_catalog(models, implicit));
    server.update_active(|model| model.source = Some(ModelSource { artifact: start.to_path_buf(), model: "mock-a".to_owned() }));
    server
}

fn row<'a>(rows: &'a Value, id: &str) -> &'a Value {
    rows["data"].as_array().expect("an array of rows").iter().find(|row| row["id"] == id).unwrap_or_else(|| panic!("no row for {id}: {rows}"))
}

fn file_in(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, contents).unwrap();
    path
}

/// AC 3: every catalog entry (built-in, then operator) and every known model
/// that is not one, each once, with its `source`, `on_disk`, `bytes`,
/// `family`, `loaded` and `switchable`; exactly one row is loaded; no row
/// carries a path.
#[tokio::test]
async fn the_catalog_lists_every_entry_then_every_known_model_with_its_state_on_disk() {
    let loader = MockLoader::new();
    let models = catalog_in("rows", &[("acme-full", Disk::Complete), ("acme-part", Disk::Partial), ("acme-none", Disk::Absent)]);
    let elsewhere = fresh_dir("rows-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let on_disk = file_in(&elsewhere, "mock-k.ninfer", b"known!!");
    let gone = elsewhere.join("gone.ninfer");
    let dir = models.dir.clone();
    let app = server_with(&loader, &start, models, &[("mock-k", &on_disk), ("mock-gone", &gone)], true).app();

    let (status, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK, "{rows}");
    let ids: Vec<&str> = rows["data"].as_array().unwrap().iter().map(|row| row["id"].as_str().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "qwen3.8-27b", "qwen3.8-27b-abliterated", "qwen3.8-flash-next", "ignis-image-2.1-turbo", "qwen3.8-27b-16g", "qwen3.8-27b-abliterated-16g", "qwen3-asr-1.7b", "qwen3-asr-1.7b-q8", "qwen3-asr-0.6b", "nemotron-3-diarization", // the built-in catalog, then the operator's
            "acme-full", "acme-part", "acme-none",
            "mock-a", "mock-gone", "mock-k", // the known models that are not entries, the start model among them
        ]
    );

    let built_in = row(&rows, "qwen3.8-27b");
    assert_eq!((built_in["source"].clone(), built_in["on_disk"].clone()), (json!("built-in"), json!("absent")));
    assert!(built_in["bytes"].as_u64().unwrap() > 1 << 30, "an entry's total, even absent: {built_in}");
    assert_eq!((built_in["loaded"].clone(), built_in["switchable"].clone()), (json!(false), json!(false)), "nothing on disk to switch to");
    assert!(built_in.get("family").is_none(), "no artifact, no header to name a family");

    let full = row(&rows, "acme-full");
    assert_eq!(full, &json!({ "id": "acme-full", "source": "operator", "on_disk": "complete", "bytes": 8, "loaded": false, "switchable": true, "available": true, "download": "none", "bytes_on_disk": 8 }));
    assert_eq!(row(&rows, "acme-part")["on_disk"], "partial");
    assert_eq!(row(&rows, "acme-part")["switchable"], false, "its artifact is not there");
    // GitHub #376: what of an entry is there now, its 1-byte sidecar of 8.
    assert_eq!((row(&rows, "acme-part")["bytes_on_disk"].clone(), row(&rows, "acme-part")["download"].clone()), (json!(1), json!("none")));
    assert_eq!((row(&rows, "acme-none")["on_disk"].clone(), row(&rows, "acme-none")["bytes"].clone()), (json!("absent"), json!(8)));
    assert_eq!(row(&rows, "acme-none")["bytes_on_disk"], 0);

    let known = row(&rows, "mock-k");
    assert_eq!((known["source"].clone(), known["on_disk"].clone(), known["bytes"].clone()), (json!("known"), json!("complete"), json!(7)));
    assert_eq!(known["switchable"], true);
    let missing = row(&rows, "mock-gone");
    assert_eq!((missing["on_disk"].clone(), missing["switchable"].clone()), (json!("absent"), json!(false)));
    assert!(missing.get("bytes").is_none(), "absent: no size: {missing}");
    // A known model is never downloaded: its artifact's size, or 0.
    assert_eq!((known["bytes_on_disk"].clone(), known["download"].clone()), (json!(7), json!("none")));
    assert_eq!((missing["bytes_on_disk"].clone(), missing["download"].clone()), (json!(0), json!("none")));

    let loaded: Vec<&str> = rows["data"].as_array().unwrap().iter().filter(|row| row["loaded"] == true).map(|row| row["id"].as_str().unwrap()).collect();
    assert_eq!(loaded, ["mock-a"], "exactly the model serving");
    assert_eq!(row(&rows, "mock-a")["family"], "qwen38", "the loaded model is always the running family");
    assert_eq!(row(&rows, "mock-a")["switchable"], false, "already loaded");

    let text = rows.to_string();
    for path in [dir.to_string_lossy(), elsewhere.to_string_lossy()] {
        assert!(!text.contains(&*path.replace('\\', "\\\\")) && !text.contains(".ninfer"), "no row carries a path: {text}");
    }
    for object in rows["data"].as_array().unwrap() {
        assert!(object.get("path").is_none() && object.get("artifact").is_none(), "{object}");
    }
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// A known model is listed once and its entry wins: the catalog row carries
/// the catalog's `source`, and the operator's path decides `switchable`.
#[tokio::test]
async fn a_known_model_that_is_also_a_catalog_entry_is_one_row() {
    let loader = MockLoader::new();
    let models = catalog_in("both", &[("mock-b", Disk::Absent)]);
    let elsewhere = fresh_dir("both-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let own = file_in(&elsewhere, "mock-b.ninfer", b"own copy!!");
    let app = server_with(&loader, &start, models.clone(), &[("mock-b", &own)], true).app();
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(rows["data"].as_array().unwrap().iter().filter(|row| row["id"] == "mock-b").count(), 1, "{rows}");
    let b = row(&rows, "mock-b");
    assert_eq!((b["source"].clone(), b["switchable"].clone()), (json!("operator"), json!(true)), "switchable by the known entry's file");
    // 324 item 3: the file the server would load is what `on_disk` and `bytes` read, not `download.path`.
    assert_eq!((b["on_disk"].clone(), b["bytes"].clone()), (json!("complete"), json!(10)), "{b}");
    let _ = (std::fs::remove_dir_all(models.dir), std::fs::remove_dir_all(elsewhere));
}

/// Review-324serving F5: a listed model whose file is gone reads absent and
/// is not switchable, as `resolve` has it, though `download.path` holds the
/// catalog entry of the same id.
#[tokio::test]
async fn a_listed_model_whose_file_is_gone_reads_absent_though_download_path_has_the_entry() {
    let loader = MockLoader::new();
    let models = catalog_in("gone", &[("mock-b", Disk::Complete)]);
    let elsewhere = fresh_dir("gone-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let gone = elsewhere.join("mock-b.ninfer");
    let app = server_with(&loader, &start, models.clone(), &[("mock-b", &gone)], true).app();
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let b = row(&rows, "mock-b");
    assert_eq!((b["on_disk"].clone(), b["switchable"].clone()), (json!("absent"), json!(false)), "{b}");
    let _ = (std::fs::remove_dir_all(models.dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 4: the catalog answers during a switch — `loaded` still the old model,
/// nothing switchable — and `loaded` follows the switch that completes.
#[tokio::test]
async fn the_catalog_answers_during_a_switch_and_loaded_follows_the_completed_one() {
    let loader = MockLoader::new();
    let models = catalog_in("during", &[("mock-b", Disk::Complete)]);
    let elsewhere = fresh_dir("during-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let dir = models.dir.clone();
    let app = server_with(&loader, &start, models, &[], true).app();

    let release = loader.hold_next_load();
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    let (status, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK, "answers while every other route is 503: {rows}");
    assert_eq!(row(&rows, "mock-a")["loaded"], true, "the old model serves until the new one does");
    assert!(rows["data"].as_array().unwrap().iter().all(|row| row["switchable"] == false), "no switch can begin now: {rows}");
    let (status, _) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK);

    release.send(()).expect("the held load is waiting");
    until_status(&app, "serving").await;
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "mock-b")["loaded"].clone(), row(&rows, "mock-a")["loaded"].clone()), (json!(true), json!(false)));
    assert_eq!(row(&rows, "mock-a")["switchable"], true, "the way back");
    assert_eq!(row(&rows, "mock-b")["switchable"], false);
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// A failed switch with nothing serving still answers the catalog, and a
/// switch can begin: it is the way out.
#[tokio::test]
async fn the_catalog_answers_while_failed_and_offers_the_way_out() {
    let loader = MockLoader::new();
    loader.break_load("mock-b");
    let models = catalog_in("failed", &[("mock-b", Disk::Complete), ("mock-c", Disk::Complete)]);
    let elsewhere = fresh_dir("failed-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let dir = models.dir.clone();
    let server = server_with(&loader, &start, models, &[], true);
    server.update_active(|model| model.source = None);
    let app = server.app();
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "failed").await;
    let (status, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK, "{rows}");
    assert_eq!(row(&rows, "mock-c")["switchable"], true, "a switch recovers a failed server");
    assert!(rows["data"].as_array().unwrap().iter().all(|row| row["loaded"] == false), "nothing serves, so no row is loaded: {rows}");
    assert_eq!(row(&rows, "mock-a")["switchable"], true, "the torn-down model is a way out too");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// AC 5: `model` alone switches to a known model and to a catalog entry on
/// disk, with `switch.allow_implicit` on or off; a known model's entry wins
/// over a catalog entry of the same id.
#[tokio::test]
async fn model_alone_switches_to_a_known_model_or_a_catalog_entry_on_disk_with_implicit_on_or_off() {
    for implicit in [true, false] {
        let loader = MockLoader::new();
        let models = catalog_in(&format!("by-id-{implicit}"), &[("mock-cat", Disk::Complete), ("mock-b", Disk::Complete)]);
        let elsewhere = fresh_dir(&format!("by-id-{implicit}-elsewhere"));
        let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
        let known_b = file_in(&elsewhere, "mock-b.ninfer", b"the operator's own");
        let catalog_artifact = models.dir.join("mock-cat.ninfer");
        let dir = models.dir.clone();
        let app = server_with(&loader, &start, models, &[("mock-b", &known_b)], implicit).app();

        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "implicit {implicit}: {accepted}");
        assert_eq!(accepted, json!({ "status": "switching", "from": "mock-a", "to": "mock-b" }));
        assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "mock-b");
        assert_eq!(loader.prepared_artifacts(), [known_b.clone()], "implicit {implicit}: the known entry wins over the catalog's mock-b");

        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-cat" }))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "implicit {implicit}: {accepted}");
        assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "mock-cat");
        assert_eq!(loader.prepared_artifacts(), [known_b, catalog_artifact], "the catalog entry's file under download.path");

        let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-a" }))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "the start model is known without being listed: {accepted}");
        assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "mock-a");
        let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
    }
}

/// AC 6: an id that resolves to nothing, or to an entry not on disk, is `404
/// model_not_found` naming the id — and for a catalog entry the command that
/// fetches it. Nothing is unloaded, nothing downloaded.
#[tokio::test]
async fn model_alone_that_resolves_to_nothing_on_disk_is_a_404_and_touches_nothing() {
    let loader = MockLoader::new();
    let models = catalog_in("refused", &[("mock-gone", Disk::Absent), ("mock-part", Disk::Partial)]);
    let elsewhere = fresh_dir("refused-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let missing = elsewhere.join("missing.ninfer");
    let dir = models.dir.clone();
    let before: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|entry| entry.unwrap().file_name()).collect();
    let app = server_with(&loader, &start, models, &[("mock-missing", &missing)], true).app();

    for (id, says) in [
        ("mock-nope", vec!["mock-nope", "unknown model"]),
        ("mock-gone", vec!["mock-gone", "ignis-server model download mock-gone"]),
        ("mock-part", vec!["mock-part", "ignis-server model download mock-part"]),
        ("mock-missing", vec!["mock-missing", "missing.ninfer"]),
    ] {
        let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": id }))).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{id}: {refused}");
        assert_eq!(refused["error"]["code"], "model_not_found");
        assert_eq!(refused["error"]["param"], "model");
        let message = refused["error"]["message"].as_str().unwrap();
        for part in says {
            assert!(message.contains(part), "{id}: {message}");
        }
    }
    let (_, models_now) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models_now["status"].clone(), models_now["data"][0]["id"].clone()), (json!("serving"), json!("mock-a")));
    assert!(loader.loads().is_empty() && loader.prepared_artifacts().is_empty(), "nothing was unloaded or loaded");
    assert_eq!(loader.resident(), 1);
    let after: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|entry| entry.unwrap().file_name()).collect();
    assert_eq!(before, after, "nothing was downloaded");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// A server with no model loader cannot resolve an id either: `501`, as for
/// a switch with an artifact.
#[tokio::test]
async fn model_alone_on_a_server_with_no_loader_is_501() {
    let loader = MockLoader::new();
    let app = Server::from_active(loader.model("mock-a")).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("switch_unavailable")));
}

/// AC 7: with `artifact` and `model` given the route behaves as before — the
/// artifact is taken as sent, whatever the catalog holds, and an empty one is
/// a `400` naming it.
#[tokio::test]
async fn artifact_and_model_together_behave_as_before() {
    let loader = MockLoader::new();
    let models = catalog_in("explicit", &[("mock-b", Disk::Complete)]);
    let elsewhere = fresh_dir("explicit-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let dir = models.dir.clone();
    let app = server_with(&loader, &start, models, &[], true).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": " ", "model": "mock-b" }))).await;
    assert_eq!((status, refused["error"]["param"].clone()), (StatusCode::BAD_REQUEST, json!("artifact")));
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "elsewhere/b.ninfer", "model": "mock-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "serving").await;
    assert_eq!(loader.prepared_artifacts(), [PathBuf::from("elsewhere/b.ninfer")], "not the catalog's");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// A plain server for the `last_switch` tests: `mock-a`, switching to
/// whatever is named with an artifact.
fn plain(loader: &Arc<MockLoader>) -> Server {
    Server::from_active(loader.model("mock-a")).with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
}

async fn switch_to(app: &axum::Router, model: &str) {
    let (status, accepted) = send(app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": format!("{model}.ninfer"), "model": model }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
}

/// AC 8: no `last_switch` before the first switch; after each switch it
/// carries `from`, `to`, `outcome` and, for every outcome but `switched`, a
/// `reason` — and it is already there when a poller reads the status that
/// follows, whatever that status is.
#[tokio::test]
async fn last_switch_is_absent_before_the_first_switch_and_says_how_each_one_ended() {
    let loader = MockLoader::new();
    loader.refuse("mock-refused");
    loader.break_load("mock-broken");
    let app = plain(&loader).app();
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert!(models.get("last_switch").is_none(), "{models}");

    switch_to(&app, "mock-b").await;
    let models = until_status(&app, "serving").await;
    assert_eq!(models["last_switch"], json!({ "from": "mock-a", "to": "mock-b", "outcome": "switched" }), "no reason for a switch that took");

    switch_to(&app, "mock-refused").await;
    let models = until_status(&app, "serving").await;
    assert_eq!((models["last_switch"]["from"].clone(), models["last_switch"]["to"].clone()), (json!("mock-b"), json!("mock-refused")));
    assert_eq!(models["last_switch"]["outcome"], "refused");
    assert!(models["last_switch"]["reason"].as_str().unwrap().contains("refused at prepare"), "{models}");
    assert_eq!(models["data"][0]["id"], "mock-b", "nothing was torn down");

    switch_to(&app, "mock-broken").await;
    let models = until_status(&app, "serving").await;
    assert_eq!(models["last_switch"]["outcome"], "restored");
    assert!(models["last_switch"]["reason"].as_str().unwrap().contains("simulated kernel load error"), "{models}");
    assert_eq!((models["last_switch"]["from"].clone(), models["last_switch"]["to"].clone()), (json!("mock-b"), json!("mock-broken")));
    assert_eq!(models["data"][0]["id"], "mock-b", "the old model was reloaded");

    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "mock-c.ninfer", "model": "mock-c" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let models = until_status(&app, "serving").await;
    assert_eq!(models["last_switch"]["outcome"], "switched", "the latest replaces the one before");
    assert!(models["last_switch"].get("reason").is_none());
}

/// `failed`: nothing serves, and `last_switch` says why beside the status.
#[tokio::test]
async fn last_switch_says_failed_when_nothing_could_be_reloaded() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let server = plain(&loader);
    server.update_active(|model| model.source = None);
    let app = server.app();
    switch_to(&app, "mock-broken").await;
    let models = until_status(&app, "failed").await;
    assert_eq!(models["last_switch"]["outcome"], "failed", "{models}");
    assert_eq!(models["last_switch"]["reason"], models["reason"]);
    assert_eq!((models["last_switch"]["from"].clone(), models["last_switch"]["to"].clone()), (json!("mock-a"), json!("mock-broken")));
}

/// AC 8: a reload from `PATCH /v1/config` is reported the same way, `from`
/// equal to `to`.
#[tokio::test]
async fn a_reload_from_a_config_patch_reports_from_equal_to() {
    let loader = MockLoader::new();
    let argv: Vec<String> = Vec::new();
    let config = match resolve_with(&argv, |_| None, &RealFiles).unwrap() {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let server = plain(&loader).with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    let app = server.app();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "kv_host_pool_bytes": "3G" } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    let models = until_status(&app, "serving").await;
    assert_eq!(models["last_switch"], json!({ "from": "mock-a", "to": "mock-a", "outcome": "switched" }));
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 61 } }))).await;
    assert_eq!(status, StatusCode::OK);
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["last_switch"]["outcome"], "switched", "a live change is no reload, so it does not replace it");
}

/// AC 9: the preflight of both new routes passes like the other `/v1`
/// routes, and the catalog sits behind the API key and the warm-up.
#[tokio::test]
async fn the_new_routes_answer_the_preflight_and_sit_behind_the_key() {
    let loader = MockLoader::new();
    let keyed = plain(&loader).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    for uri in ["/v1/models/catalog", "/v1/config/fields"] {
        let (status, _) = send(&keyed, Method::OPTIONS, uri, None).await;
        assert_eq!(status, StatusCode::OK, "{uri}");
        let (status, _) = send(&keyed, Method::GET, uri, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
    }
    let warming = plain(&loader).with_warm_up().app();
    let (status, refused) = send(&warming, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("server_not_ready")));
}

/// A server with no switcher at all still lists the model it serves.
#[tokio::test]
async fn a_server_with_no_switcher_lists_the_loaded_model_alone() {
    let loader = MockLoader::new();
    let app = Server::from_active(loader.model("mock-a")).app();
    let (status, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(rows["data"].as_array().unwrap().len(), 1, "{rows}");
    assert_eq!(rows["data"][0]["id"], "mock-a");
    assert_eq!((rows["data"][0]["loaded"].clone(), rows["data"][0]["switchable"].clone()), (json!(true), json!(false)));
}

/// The start model stays switchable by id for the server's life: every
/// switch that lands re-applies the server-level values (`apply_config`), and
/// the table it rebuilds must still hold the model the server started on.
#[tokio::test]
async fn the_start_model_stays_switchable_by_id_after_a_switch_re_applies_the_config() {
    let loader = MockLoader::new();
    let argv: Vec<String> = Vec::new();
    let config = match resolve_with(&argv, |_| None, &RealFiles).unwrap() {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    loader.serve_with(config);
    let models = catalog_in("start-stays", &[("mock-cat", Disk::Absent)]);
    let elsewhere = fresh_dir("start-stays-elsewhere");
    let start = file_in(&elsewhere, "mock-a.ninfer", b"start");
    let dir = models.dir.clone();
    let app = server_with(&loader, &start, models, &[], true).app();

    switch_to(&app, "mock-b").await;
    until_status(&app, "serving").await;
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&rows, "mock-a")["switchable"], true, "the way back to the start model: {rows}");
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-a" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "mock-a");
    assert_eq!(loader.prepared_artifacts().last(), Some(&start));
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(elsewhere));
}

/// A switch task that dies while the old model is being reloaded leaves
/// `failed` and says so in `last_switch`: the status is already `failed`
/// then, so the guard that records a task dying mid-switch must not wait for
/// `switching`.
#[tokio::test]
async fn last_switch_is_recorded_when_the_switch_task_dies_during_the_reload() {
    let loader = MockLoader::new();
    loader.break_load("mock-b");
    let server = plain(&loader);
    let app = server.app();
    let release = loader.hold_next_load();
    let (_, task) = ignis_server::model_switch::begin(&server, switch_support::source("mock-b")).expect("a switch begins");
    let _held_reload = loader.hold_next_load();
    release.send(()).expect("the held load is waiting");
    until_status(&app, "failed").await;
    task.abort();
    assert!(task.await.expect_err("the task was aborted").is_cancelled());
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(models["status"], "failed", "{models}");
    assert_eq!(models["last_switch"]["outcome"], "failed", "{models}");
    assert_eq!(models["last_switch"]["reason"], models["reason"]);
    assert_eq!((models["last_switch"]["from"].clone(), models["last_switch"]["to"].clone()), (json!("mock-a"), json!("mock-b")));
}
