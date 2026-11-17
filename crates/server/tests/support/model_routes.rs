//! What the download and removal route tests share (GitHub #376-#380), and
//! the connector's (spec #656), included via `#[path]` beside `repo.rs`,
//! which the including test declares as `mod repo`: a request to the
//! mock-backed router, the configuration a start with some flags resolves,
//! the catalog's rows, read until one says what a test waits for, the
//! server's status, read until it serves what a test waits for, the files a
//! models folder holds, and `ignis-server model download` run in process.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use serde_json::Value;
use tower::ServiceExt;

use crate::repo::Repo;

/// How long a switch may take before the switcher gives up on it.
pub const PATIENT: Duration = Duration::from_secs(30);

/// One request to `app`: its status and its body, as JSON when it is.
pub async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// The configuration a start with `argv` runs, fetching from `repo`.
pub fn config(repo: &Repo, argv: &[&str]) -> Config {
    let mut args = vec!["--download-endpoint".to_owned(), repo.endpoint()];
    args.extend(argv.iter().map(|arg| arg.to_string()));
    match resolve_with(&args, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// `id`'s row of a `GET /v1/models/catalog` answer.
pub fn row<'a>(rows: &'a Value, id: &str) -> &'a Value {
    rows["data"].as_array().expect("an array of rows").iter().find(|row| row["id"] == id).unwrap_or_else(|| panic!("no row for {id}: {rows}"))
}

/// `id`'s row as `GET /v1/models/catalog` reads it now.
pub async fn catalog_row(app: &axum::Router, id: &str) -> Value {
    let (status, rows) = send(app, Method::GET, "/v1/models/catalog", None).await;
    assert_eq!(status, StatusCode::OK, "{rows}");
    row(&rows, id).clone()
}

/// `GET /v1/models/catalog` until `id`'s row meets `wanted` (scheduling
/// turns, bounded): the bytes the endpoint sent land on disk shortly after,
/// and a download says it ended shortly after its last file landed.
pub async fn until_row(app: &axum::Router, id: &str, wanted: impl Fn(&Value) -> bool) -> Value {
    let mut last = Value::Null;
    for _ in 0..10_000 {
        last = catalog_row(app, id).await;
        if wanted(&last) {
            return last;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("{id}'s row never got there: {last}");
}

/// `GET /v1/models` until its `status` reads `wanted` (bounded turns).
pub async fn until_status(app: &axum::Router, wanted: &str) -> Value {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == wanted {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported {wanted}");
}

/// `GET /v1/models` until its `status` reads `serving` and `server` meets
/// `wanted` in the same turn (bounded turns): a switch or a reload publishes
/// its model, its settings and its status at different moments, so a test
/// waits for every one it then asserts (CODING_STANDARDS, "A test waits for
/// the whole end state").
pub async fn until_serving(app: &axum::Router, server: &ignis_server::Server, wanted: impl Fn(&ignis_server::Server) -> bool) {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == "serving" && wanted(server) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported serving with the state wanted: {:?}", server.status());
}

/// The files under `dir` but the entries' locks (`<artifact>.lock`, GitHub
/// #379), which a fetch or a removal leaves there, empty and released, for
/// the next; sorted.
pub fn files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| !name.ends_with(".lock"))
        .collect();
    names.sort();
    names
}

/// `ignis-server model download` of `ids` from `repo` into `dir`, as `main`
/// runs it: its exit code and the lines it reported.
pub async fn model_download(repo: &Repo, catalog: &ignis_server::download::Catalog, ids: &[&str], dir: &Path) -> (i32, Vec<String>) {
    let command = ignis_server::download::ModelCommand::Download {
        entries: ids.iter().map(|id| catalog.entry(id).expect("a catalog entry").clone()).collect(),
        dir: dir.to_path_buf(),
        endpoint: repo.endpoint(),
        token: None,
        free_margin: 0,
    };
    let lines = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let collected = Arc::clone(&lines);
    let code = ignis_server::download::command::run(command, &mut std::io::sink(), Arc::new(move |line: &str| collected.lock().unwrap().push(line.to_owned()))).await;
    let lines = lines.lock().unwrap().clone();
    (code, lines)
}
