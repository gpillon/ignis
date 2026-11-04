//! A download started through the API (GitHub #376, spec #374, ADR 0071),
//! over the mock-backed router with a temporary `download.path`, against a
//! local endpoint serving small fake files (`support/repo.rs`): no network,
//! no GPU, no real model. `POST /v1/models/download` starts it in the
//! background, `GET /v1/models/catalog` follows it, and what lands under the
//! models folder is the outcome. A download is held mid-transfer by holding
//! the endpoint's response, never by a timer.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use ignis_server::config::Config;
use ignis_server::download_http::Downloads;
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::Server;
use serde_json::json;
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, files_of, Repo};

#[path = "support/model_routes.rs"]
mod routes;
use routes::{config, files_in, model_download, row, send, until_row, until_status, PATIENT};

/// A server on `mock-a`, switching to the catalog `models`, downloading as
/// `config` says.
fn server_with(loader: &Arc<MockLoader>, models: CatalogModels, config: &Config) -> Server {
    Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(models, true))
        .with_downloads(Downloads::of(config, None))
}

/// AC 1, 6, 7: the download answers 202 and runs in the background while the
/// server serves; the row says it runs, with the bytes on disk, and once it
/// ends the entry is on disk, switchable and listed as a servable model —
/// and the server did not switch to it.
#[tokio::test]
async fn a_download_answers_202_runs_in_the_background_and_leaves_a_servable_model() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "lands", &[("acme-new", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-new");
    repo.hold(&files.artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-new" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!((accepted["status"].clone(), accepted["model"].clone(), accepted["bytes"].clone()), (json!("downloading"), json!("acme-new"), json!(files.total())));

    let held = until_row(&app, "acme-new", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;
    assert_eq!((held["download"].clone(), held["on_disk"].clone(), held["switchable"].clone()), (json!("running"), json!("partial"), json!(false)), "{held}");
    assert!(held.get("download_reason").is_none(), "{held}");
    // The loaded model serves meanwhile.
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((status, models["status"].clone(), models["data"][0]["id"].clone()), (StatusCode::OK, json!("serving"), json!("mock-a")));

    repo.release();
    let landed = until_row(&app, "acme-new", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!((landed["download"].clone(), landed["bytes_on_disk"].clone(), landed["switchable"].clone()), (json!("none"), json!(files.total()), json!(true)), "{landed}");
    assert_eq!(files_in(&dir), [files.artifact.0.clone(), files.sidecar.0.clone()], "every file at its name, no .part left");
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().map(|model| model["id"].as_str().unwrap()).collect();
    assert_eq!(ids.first(), Some(&"mock-a"), "no switch: {models}");
    assert!(ids.contains(&"acme-new"), "a servable model now: {models}");
    assert!(loader.prepared_artifacts().is_empty(), "nothing was loaded");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 2: one download at a time — a second one, of another entry or the
/// same, is refused with 409 naming the one that runs; once it ends, the
/// next one begins.
#[tokio::test]
async fn a_second_download_while_one_runs_is_refused_with_409() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "one-at-a-time", &[("acme-a", 32), ("acme-b", 32)]);
    let dir = models.dir.clone();
    repo.hold(&files_of("acme-a").artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-a" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    for id in ["acme-b", "acme-a"] {
        let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": id }))).await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_in_progress")), "{id}: {refused}");
        assert!(refused["error"]["message"].as_str().unwrap().contains("acme-a"), "it names the one that runs: {refused}");
    }
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "acme-a")["download"].clone(), row(&rows, "acme-b")["download"].clone()), (json!("running"), json!("none")), "the refused one was never queued");

    repo.release();
    until_row(&app, "acme-a", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_row(&app, "acme-b", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 3, 4, 9: refused with a reason, before anything is fetched, for an id
/// that is no catalog entry, an entry above the card's class, an entry
/// already on disk, `download.enabled` off and `download.from_ui` off — the
/// last two through their config fields, `--download-enabled false` and
/// `--download-from-ui false`.
#[tokio::test]
async fn a_download_is_refused_with_a_reason_and_fetches_nothing() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let refused = |app: axum::Router, model: &'static str| async move {
        let (status, body) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": model }))).await;
        (status, body["error"]["code"].as_str().unwrap_or_default().to_owned(), body["error"]["message"].as_str().unwrap_or_default().to_owned())
    };

    let models = catalog_served(&repo, "refused", &[("acme-big", 32), ("acme-here", 16)]);
    let dir = models.dir.clone();
    let here = files_of("acme-here");
    std::fs::write(dir.join(&here.sidecar.0), &here.sidecar.1).unwrap();
    std::fs::write(dir.join(&here.artifact.0), &here.artifact.1).unwrap();
    let on_16gb = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(models, true).with_card(Some(card(16, "NVIDIA RTX (Blackwell)", 12, 0))))
        .with_downloads(Downloads::of(&config(&repo, &[]), None))
        .app();

    let (status, code, message) = refused(on_16gb.clone(), "nope").await;
    assert_eq!((status, code.as_str()), (StatusCode::NOT_FOUND, "model_not_found"), "{message}");
    assert!(message.contains("not a catalog entry"), "{message}");
    let (status, code, _) = refused(on_16gb.clone(), "mock-a").await;
    assert_eq!((status, code.as_str()), (StatusCode::NOT_FOUND, "model_not_found"), "the loaded model is no catalog entry here");
    let (status, code, message) = refused(on_16gb.clone(), "acme-big").await;
    assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_above_card_class"), "{message}");
    assert!(message.contains("32 GB") && message.contains("16 GB"), "it names both classes: {message}");
    let (status, code, message) = refused(on_16gb.clone(), "acme-here").await;
    assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_already_downloaded"), "{message}");
    let (status, body) = send(&on_16gb, Method::POST, "/v1/models/download", Some(json!({ "model": " " }))).await;
    assert_eq!((status, body["error"]["param"].clone()), (StatusCode::BAD_REQUEST, json!("model")));
    // Above the class: the row says so, and offers nothing to download.
    let (_, rows) = send(&on_16gb, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "acme-big")["available"].clone(), row(&rows, "acme-big")["download"].clone()), (json!(false), json!("none")));

    for (flag, field) in [("--download-enabled", "download.enabled"), ("--download-from-ui", "download.from_ui")] {
        let models = catalog_served(&repo, "refused-off", &[("acme-off", 32)]);
        let off_dir = models.dir.clone();
        let app = server_with(&loader, models, &config(&repo, &[flag, "false"])).app();
        let (status, code, message) = refused(app.clone(), "acme-off").await;
        assert_eq!((status, code.as_str()), (StatusCode::FORBIDDEN, "download_disabled"), "{flag}: {message}");
        assert!(message.contains(&format!("{field} is false")), "it names {field}: {message}");
        assert!(files_in(&off_dir).is_empty(), "{flag}: nothing landed");
        let _ = std::fs::remove_dir_all(off_dir);
    }

    // Spec #474: on an RTX 3090 a Blackwell-only built-in entry is refused
    // `409 model_above_gpu_generation`, naming the card, before any fetch.
    let built_in = CatalogModels { catalog: Arc::new(ignis_server::download::Catalog::built_in().clone()), dir: repo::fresh_dir("refused-generation") };
    let built_in_dir = built_in.dir.clone();
    let on_3090 = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(built_in, true).with_card(Some(card(24, "NVIDIA GeForce RTX 3090", 8, 6))))
        .with_downloads(Downloads::of(&config(&repo, &[]), None))
        .app();
    for model in ["qwen3.8-flash-next", "swift-1.5-27b-16g"] {
        let (status, code, message) = refused(on_3090.clone(), model).await;
        assert_eq!((status, code.as_str()), (StatusCode::CONFLICT, "model_above_gpu_generation"), "{message}");
        assert!(message.contains("Blackwell generation") && message.contains("NVIDIA GeForce RTX 3090 (compute capability 8.6)"), "{message}");
    }
    assert!(files_in(&built_in_dir).is_empty(), "nothing landed");
    let _ = std::fs::remove_dir_all(built_in_dir);
    assert!(repo.seen().is_empty(), "no refusal reached the endpoint: {:?}", repo.seen());
    let _ = std::fs::remove_dir_all(dir);
}

/// A card of `gib` GiB named `name`, of compute capability `major.minor`.
fn card(gib: u64, name: &str, major: u32, minor: u32) -> ignis_server::config::profile::CardReading {
    ignis_server::config::profile::CardReading::of(gib << 30, name, ignis_server::config::arch::ComputeCapability::new(major, minor))
}

/// GitHub #378: a download the models folder has no room for — its bytes
/// still missing plus `download.free_margin` above the volume's free space —
/// is refused before any byte is written, naming both numbers, and its row
/// keeps the reason. The short disk is the margin itself, set larger than
/// any disk; a partial entry counts only what it still misses.
#[tokio::test]
async fn a_download_the_models_folder_has_no_room_for_is_refused_and_its_row_says_why() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "no-room", &[("acme-huge", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-huge");
    // A partial entry: its sidecar (1 byte) is there, its 65,536-byte artifact is not.
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    let app = server_with(&loader, models, &config(&repo, &["--download-free-margin", "1000000G"])).app();

    let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-huge" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::INSUFFICIENT_STORAGE, json!("download_no_room")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap().to_owned();
    // 1,000,000 GiB of margin plus the 65,536 bytes still missing.
    assert!(message.contains("1073741824065536 bytes needed"), "the bytes still missing plus the margin: {message}");
    assert!(message.contains("bytes free") && message.contains("download.free_margin"), "the free space, and the field to lower: {message}");
    assert_eq!(files_in(&dir), [files.sidecar.0.clone()], "nothing written");
    assert!(repo.seen().is_empty(), "nothing fetched: {:?}", repo.seen());
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    let refused_row = row(&rows, "acme-huge");
    assert_eq!((refused_row["download"].clone(), refused_row["download_reason"].clone()), (json!("failed"), json!(message)), "{refused_row}");
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #378, #376: a refusal or failure kept on a row does not outlive the
/// entry it was about. Refused for want of room, the entry is then fetched
/// another way, by `model download` into the same folder: its row reads
/// complete with no failure, and the failure is forgotten, not hidden — the
/// entry made partial again reads partial with none.
#[tokio::test]
async fn a_kept_failure_goes_once_the_entry_is_complete_another_way() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "stale-failure", &[("acme-later", 32)]);
    let (catalog, dir) = (Arc::clone(&models.catalog), models.dir.clone());
    let files = files_of("acme-later");
    std::fs::write(dir.join(&files.sidecar.0), &files.sidecar.1).unwrap();
    let app = server_with(&loader, models, &config(&repo, &["--download-free-margin", "1000000G"])).app();
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-later" }))).await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE);
    let refused = row(&send(&app, Method::GET, "/v1/models/catalog", None).await.1, "acme-later").clone();
    assert_eq!(refused["download"], "failed", "{refused}");

    let (code, lines) = model_download(&repo, &catalog, &["acme-later"], &dir).await;
    assert_eq!(code, 0, "{lines:?}");
    let landed = row(&send(&app, Method::GET, "/v1/models/catalog", None).await.1, "acme-later").clone();
    assert_eq!((landed["on_disk"].clone(), landed["download"].clone()), (json!("complete"), json!("none")), "{landed}");
    assert!(landed.get("download_reason").is_none(), "a complete row reads no failure: {landed}");

    std::fs::remove_file(dir.join(&files.artifact.0)).unwrap();
    let partial = row(&send(&app, Method::GET, "/v1/models/catalog", None).await.1, "acme-later").clone();
    assert_eq!((partial["on_disk"].clone(), partial["download"].clone()), (json!("partial"), json!("none")), "forgotten, not hidden: {partial}");
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 8: a failed download keeps its reason on the row, in memory, until the
/// next attempt; a 401 or 403 from the endpoint names the token and where
/// the server reads it, and never the token itself.
#[tokio::test]
async fn a_failed_download_keeps_its_reason_on_the_row_until_the_next_attempt() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "failed", &[("acme-gated", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    repo.answer(401);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-gated" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let failed = until_row(&app, "acme-gated", |row| row["download"] == "failed").await;
    let reason = failed["download_reason"].as_str().expect("a failure carries its reason").to_owned();
    assert!(reason.contains("401") && reason.contains("none was sent") && reason.contains("IGNIS_DOWNLOAD_TOKEN"), "{reason}");
    assert!(!reason.contains("HF_TOKEN"), "HF_TOKEN goes to huggingface.co only, not to this endpoint: {reason}");
    let (_, again) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&again, "acme-gated")["download_reason"], json!(reason), "kept, read after read");

    // The next attempt forgets it as it begins.
    repo.answer(0);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-gated" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let landed = until_row(&app, "acme-gated", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!(landed["download"], "none", "{landed}");
    assert!(landed.get("download_reason").is_none(), "{landed}");

    // A token the endpoint refuses: named by where it comes from, never shown.
    let models = catalog_served(&repo, "failed-token", &[("acme-private", 32)]);
    let token_dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &["--download-token", "tok-secret"])).app();
    repo.answer(403);
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-private" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let failed = until_row(&app, "acme-private", |row| row["download"] == "failed").await;
    let reason = failed["download_reason"].as_str().unwrap();
    assert!(reason.contains("403") && reason.contains("refused the token sent") && reason.contains("IGNIS_DOWNLOAD_TOKEN"), "{reason}");
    assert!(!failed.to_string().contains("tok-secret"), "the token never reaches the row: {failed}");
    let sent: Vec<_> = repo.seen().into_iter().filter(|seen| seen.file.starts_with("acme-private")).collect();
    assert!(!sent.is_empty() && sent.iter().all(|seen| seen.authorization.as_deref() == Some("Bearer tok-secret")), "the token went to the endpoint: {sent:?}");
    let _ = (std::fs::remove_dir_all(dir), std::fs::remove_dir_all(token_dir));
}

/// AC 1: the route takes the same authentication as `PATCH /v1/config` —
/// the server's one API key — and answers the browser's preflight.
#[tokio::test]
async fn the_download_route_sits_behind_the_api_key_and_answers_the_preflight() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "keyed", &[("acme-keyed", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let keyed = |key: Option<&'static str>| {
        let mut request = Request::builder().method(Method::POST).uri("/v1/models/download").header(header::CONTENT_TYPE, "application/json");
        if let Some(key) = key {
            request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
        }
        request.body(Body::from(json!({ "model": "acme-keyed" }).to_string())).unwrap()
    };

    let (status, _) = send(&app, Method::OPTIONS, "/v1/models/download", None).await;
    assert_eq!(status, StatusCode::OK);
    for key in [None, Some("sk-wrong")] {
        let response = app.clone().oneshot(keyed(key)).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{key:?}");
        assert_eq!(response.headers().get(header::WWW_AUTHENTICATE).unwrap(), "Bearer");
    }
    assert!(files_in(&dir).is_empty() && repo.seen().is_empty(), "nothing downloaded without the key");
    let response = app.clone().oneshot(keyed(Some("sk-test"))).await.unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let _ = std::fs::remove_dir_all(dir);
}

/// AC 10: a switch still never downloads: a catalog entry not on disk is a
/// `404 model_not_found`, and nothing reaches the endpoint.
#[tokio::test]
async fn a_switch_still_never_downloads() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "no-switch-download", &[("acme-far", 32)]);
    let dir = models.dir.clone();
    let app = server_with(&loader, models, &config(&repo, &[])).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "acme-far" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_FOUND, json!("model_not_found")), "{refused}");
    assert!(repo.seen().is_empty() && files_in(&dir).is_empty(), "nothing fetched");
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&rows, "acme-far")["download"], "none");
    let _ = std::fs::remove_dir_all(dir);
}

/// A server built with no downloads (a test's) answers `501`, as one with
/// no loader answers a switch.
#[tokio::test]
async fn a_server_without_downloads_answers_501() {
    let loader = MockLoader::new();
    let app = Server::from_active(loader.model("mock-a")).app();
    let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "qwen3.8-27b" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("download_unavailable")));
}

/// GitHub #379: the command refuses an entry the server is downloading,
/// saying so, and fetches nothing of it; another entry it fetches at the
/// same time, beside the server's download, which lands as it would have.
#[tokio::test]
async fn model_download_refuses_an_entry_the_server_is_downloading_and_fetches_another_meanwhile() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "lock-server-first", &[("acme-a", 32), ("acme-b", 32)]);
    let (catalog, dir) = (Arc::clone(&models.catalog), models.dir.clone());
    repo.hold(&files_of("acme-a").artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-a" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_row(&app, "acme-a", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;

    let (code, lines) = model_download(&repo, &catalog, &["acme-a"], &dir).await;
    assert_eq!(code, 1, "{lines:?}");
    // Story 17: it says who holds the entry, the server, and its process.
    let holder = format!("by a server (process {}), which holds", std::process::id());
    assert!(lines.iter().any(|line| line.starts_with("acme-a: failed: acme-a is already being downloaded") && line.contains(&holder)), "{lines:?}");
    assert_eq!(repo.seen().iter().filter(|seen| seen.file.starts_with("acme-a")).count(), 2, "the command sent nothing for it: {:?}", repo.seen());

    let (code, lines) = model_download(&repo, &catalog, &["acme-b"], &dir).await;
    assert_eq!(code, 0, "{lines:?}");
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!((row(&rows, "acme-a")["download"].clone(), row(&rows, "acme-b")["on_disk"].clone()), (json!("running"), json!("complete")));

    repo.release();
    let landed = until_row(&app, "acme-a", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!(landed["bytes_on_disk"], json!(files_of("acme-a").total()));
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #379: the page refuses an entry the command is downloading with
/// `409 download_locked`, saying so, and starts another entry meanwhile;
/// once the command has ended, the entry is free again.
#[tokio::test]
async fn the_page_refuses_an_entry_model_download_is_downloading_and_starts_another_meanwhile() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "lock-command-first", &[("acme-a", 32), ("acme-b", 32)]);
    let (catalog, dir) = (Arc::clone(&models.catalog), models.dir.clone());
    repo.hold(&files_of("acme-a").artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let command = model_download(&repo, &catalog, &["acme-a"], &dir);
    let page = async {
        // The command's bytes, as the server's catalog reads them.
        let held = until_row(&app, "acme-a", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;
        assert_eq!(held["download"], "none", "not the server's download: {held}");
        let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-a" }))).await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_locked")), "{refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        let holder = format!("by `ignis-server model download` (process {}), which holds", std::process::id());
        assert!(message.contains("acme-a is already being downloaded") && message.contains(&holder), "it names the holder: {message}");
        let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
        assert!(row(&rows, "acme-a").get("download_reason").is_none(), "a refusal is no failure: {rows}");
        // Short of room too, it is refused for the lock: a refusal for room
        // kept on the row would outlive the command that lands the entry.
        let short = CatalogModels { catalog: Arc::clone(&catalog), dir: dir.clone() };
        let short = server_with(&loader, short, &config(&repo, &["--download-free-margin", "1000000G"])).app();
        let (status, refused) = send(&short, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-a" }))).await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_locked")), "{refused}");
        let (_, rows) = send(&short, Method::GET, "/v1/models/catalog", None).await;
        assert!(row(&rows, "acme-a").get("download_reason").is_none(), "no refusal for room left behind: {rows}");

        let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-b" }))).await;
        assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
        until_row(&app, "acme-b", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
        repo.release();
    };
    let ((code, lines), ()) = tokio::join!(command, page);
    assert_eq!(code, 0, "{lines:?}");
    let (_, rows) = send(&app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(row(&rows, "acme-a")["on_disk"], "complete");
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #377 AC 1, 3: `POST /v1/models/download/cancel` stops the running
/// download and keeps what it fetched — the sidecar verified at its name, the
/// artifact's `.part` with every byte that arrived — and the row reads
/// partial with those bytes; the entry's next download resumes from them
/// with `Range`, fetches nothing twice, and ends verified against the pins.
#[tokio::test]
async fn a_cancelled_download_keeps_its_partial_files_and_resume_continues_from_them() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "cancel-resume", &[("acme-stop", 32)]);
    let dir = models.dir.clone();
    let files = files_of("acme-stop");
    repo.hold(&files.artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();

    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-stop" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_row(&app, "acme-stop", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;

    let (status, cancelled) = send(&app, Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!(status, StatusCode::OK, "{cancelled}");
    assert_eq!(cancelled, json!({ "status": "cancelled", "model": "acme-stop", "bytes": files.total(), "bytes_on_disk": 1 + 4096 }));
    // Stopped, not ended: what arrived stays, nothing is at the artifact's name.
    let part = format!("{}.part", files.artifact.0);
    assert_eq!(files_in(&dir), [files.sidecar.0.clone(), part.clone()]);
    assert_eq!(std::fs::read(dir.join(&part)).unwrap(), files.artifact.1[..4096]);
    let stopped = row(&send(&app, Method::GET, "/v1/models/catalog", None).await.1, "acme-stop").clone();
    assert_eq!((stopped["download"].clone(), stopped["on_disk"].clone(), stopped["bytes_on_disk"].clone()), (json!("none"), json!("partial"), json!(1 + 4096)), "{stopped}");
    assert!(stopped.get("download_reason").is_none(), "a cancel is no failure: {stopped}");
    let (status, refused) = send(&app, Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_not_running")), "nothing left to cancel: {refused}");

    // Resume: the same request, from the bytes on disk.
    repo.release();
    let requests = repo.seen().len();
    let (status, accepted) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-stop" }))).await;
    assert_eq!((status, accepted["bytes_on_disk"].clone()), (StatusCode::ACCEPTED, json!(1 + 4096)), "{accepted}");
    let landed = until_row(&app, "acme-stop", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!((landed["download"].clone(), landed["bytes_on_disk"].clone(), landed["switchable"].clone()), (json!("none"), json!(files.total()), json!(true)), "{landed}");
    let resumed: Vec<(String, Option<String>)> = repo.seen()[requests..].iter().map(|seen| (seen.file.clone(), seen.range.clone())).collect();
    assert_eq!(resumed, [(files.artifact.0.clone(), Some("bytes=4096-".to_owned()))], "only the rest of the artifact");
    assert_eq!(files_in(&dir), [files.artifact.0.clone(), files.sidecar.0.clone()], "no .part left");
    assert_eq!(std::fs::read(dir.join(&files.artifact.0)).unwrap(), files.artifact.1, "the pinned bytes");
    assert_eq!(std::fs::read(dir.join(&files.sidecar.0)).unwrap(), files.sidecar.1);
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #377 AC 1: with no download running there is nothing to cancel
/// (`409`); a server with no downloads answers `501`; the route takes the
/// server's one API key, as `POST /v1/models/download` does, and answers the
/// browser's preflight.
#[tokio::test]
async fn a_cancel_with_nothing_running_is_refused_and_sits_behind_the_api_key() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "cancel-none", &[("acme-idle", 32)]);
    let dir = models.dir.clone();
    let server = server_with(&loader, models, &config(&repo, &[]));
    let (status, refused) = send(&server.clone().app(), Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::CONFLICT, json!("download_not_running")), "{refused}");
    let (status, refused) = send(&Server::from_active(loader.model("mock-a")).app(), Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("download_unavailable")));

    let keyed = server.with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let (status, _) = send(&keyed, Method::OPTIONS, "/v1/models/download/cancel", None).await;
    assert_eq!(status, StatusCode::OK);
    let response = keyed.clone().oneshot(Request::builder().method(Method::POST).uri("/v1/models/download/cancel").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(response.headers().get(header::WWW_AUTHENTICATE).unwrap(), "Bearer");
    let request = Request::builder().method(Method::POST).uri("/v1/models/download/cancel").header(header::AUTHORIZATION, "Bearer sk-test");
    let response = keyed.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::CONFLICT, "past the key: nothing runs");
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #377: a cancel never touches the loaded model, so it stays open
/// while a switch runs, as the routes that report and drive the switch do: a
/// download begun before the switch is cancelled during it, and keeps its
/// partial files, while a new download is still held by the switch.
#[tokio::test]
async fn a_download_is_cancelled_while_a_switch_runs() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "cancel-switching", &[("acme-held", 32), ("acme-next", 32), ("acme-later", 32)]);
    let dir = models.dir.clone();
    let next = files_of("acme-next");
    std::fs::write(dir.join(&next.sidecar.0), &next.sidecar.1).unwrap();
    std::fs::write(dir.join(&next.artifact.0), &next.artifact.1).unwrap();
    repo.hold(&files_of("acme-held").artifact.0, 4096);
    let app = server_with(&loader, models, &config(&repo, &[])).app();
    let (status, _) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-held" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_row(&app, "acme-held", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;

    let release = loader.hold_next_load();
    let (status, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "acme-next" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_status(&app, "switching").await;
    let (status, refused) = send(&app, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-later" }))).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")), "a new download waits for the switch: {refused}");
    let (status, cancelled) = send(&app, Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!((status, cancelled["status"].clone(), cancelled["bytes_on_disk"].clone()), (StatusCode::OK, json!("cancelled"), json!(1 + 4096)), "{cancelled}");
    let stopped = row(&send(&app, Method::GET, "/v1/models/catalog", None).await.1, "acme-held").clone();
    assert_eq!((stopped["download"].clone(), stopped["on_disk"].clone()), (json!("none"), json!("partial")), "{stopped}");

    release.send(()).expect("the held load waits");
    assert_eq!(until_status(&app, "serving").await["data"][0]["id"], "acme-next", "the switch went on as it would have");
    repo.release();
    let _ = std::fs::remove_dir_all(dir);
}

/// GitHub #377 AC 5: after a restart a partial entry stays partial. A server
/// started over the `.part` an earlier one left (a fresh `Downloads`, the
/// same `download.path`) shows the entry partial with its bytes and no
/// download, and fetches nothing until asked.
#[tokio::test]
async fn a_partial_entry_stays_partial_after_a_restart_until_its_resume() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let models = catalog_served(&repo, "restart", &[("acme-later", 32)]);
    let (dir, again) = (models.dir.clone(), models.clone());
    let files = files_of("acme-later");
    repo.hold(&files.artifact.0, 4096);
    let before = server_with(&loader, models, &config(&repo, &[])).app();
    let (status, _) = send(&before, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-later" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_row(&before, "acme-later", |row| row["bytes_on_disk"] == json!(1 + 4096)).await;
    let (status, _) = send(&before, Method::POST, "/v1/models/download/cancel", None).await;
    assert_eq!(status, StatusCode::OK);
    drop(before);
    repo.release();

    let requests = repo.seen().len();
    let after = server_with(&loader, again, &config(&repo, &[])).app();
    for _ in 0..3 {
        let read = row(&send(&after, Method::GET, "/v1/models/catalog", None).await.1, "acme-later").clone();
        assert_eq!((read["on_disk"].clone(), read["download"].clone(), read["bytes_on_disk"].clone()), (json!("partial"), json!("none"), json!(1 + 4096)), "{read}");
        tokio::task::yield_now().await;
    }
    assert_eq!(repo.seen().len(), requests, "nothing resumed by itself");
    let (status, accepted) = send(&after, Method::POST, "/v1/models/download", Some(json!({ "model": "acme-later" }))).await;
    assert_eq!((status, accepted["bytes_on_disk"].clone()), (StatusCode::ACCEPTED, json!(1 + 4096)), "{accepted}");
    until_row(&after, "acme-later", |row| row["on_disk"] == "complete" && row["download"] == "none").await;
    assert_eq!(std::fs::read(dir.join(&files.artifact.0)).unwrap(), files.artifact.1);
    let _ = std::fs::remove_dir_all(dir);
}
