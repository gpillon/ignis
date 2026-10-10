//! Spec model-download/02 AC 7, the log half: what a start logs while it
//! fetches with a token names the URL and never the token.
//!
//! A binary of its own, with one test: `tracing` caches whether a callsite
//! is enabled, and a test fetching with no subscriber, on another thread of
//! the same binary, can settle that cache for the transfer's events before
//! this one's scoped subscriber is consulted — the capture then sees
//! nothing. Alone here, nothing races it.

use std::sync::Arc;

use axum::extract::Path as UrlPath;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use tracing_subscriber::layer::SubscriberExt;

use ignis_server::config::ApiKey;
use ignis_server::download::{CatalogEntry, CatalogFile, CatalogLayer, Downloader};

const SIDECAR_BODY: &[u8] = br#"{"recipe_id":"test"}"#;
const ARTIFACT_BODY: &[u8] = b"the weights, in a few bytes";

fn pinned(name: &str, body: &[u8]) -> CatalogFile {
    use sha2::{Digest, Sha256};
    let sha256 = Sha256::digest(body).iter().map(|b| format!("{b:02x}")).collect();
    CatalogFile { name: name.to_owned(), bytes: body.len() as u64, sha256 }
}

/// Serves the two files, and answers `401` to a request without the token
/// — so the fetch only succeeds if the token really was sent.
async fn resolve(UrlPath((_owner, _name, _revision, file)): UrlPath<(String, String, String, String)>, headers: HeaderMap) -> Response {
    if headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) != Some("Bearer tok-secret") {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    match file.as_str() {
        "m.ninfer.graft.json" => SIDECAR_BODY.into_response(),
        "m.ninfer" => ARTIFACT_BODY.into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[tokio::test]
async fn a_fetch_with_a_token_logs_its_urls_and_never_the_token() {
    let app = Router::new().route("/{owner}/{name}/resolve/{revision}/{file}", get(resolve));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    let entry = CatalogEntry {
        id: "m".to_owned(),
        repo: "acme/m".to_owned(),
        revision: "v1".to_owned(),
        artifact: "m.ninfer".to_owned(),
        files: vec![pinned("m.ninfer.graft.json", SIDECAR_BODY), pinned("m.ninfer", ARTIFACT_BODY)],
        layer: CatalogLayer::Operator,
    };
    let dir = std::env::temp_dir().join(format!("ignis-download-log-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())));

    let downloader = Downloader::new(&format!("http://{addr}"), Some(ApiKey::new("tok-secret"))).expect("downloader");
    downloader.fetch(&entry, &dir).await.expect("fetched with the token");

    let records = sink.lines();
    let started: Vec<&String> = records.iter().filter(|line| line.contains("ignis.model.download_started")).collect();
    assert_eq!(started.len(), 2, "{records:?}");
    assert!(started.iter().all(|line| line.contains(&format!("http://{addr}/acme/m/resolve/v1/"))), "{started:?}");
    assert_eq!(records.iter().filter(|line| line.contains("ignis.model.downloaded")).count(), 2, "{records:?}");
    for line in &records {
        assert!(!line.contains("tok-secret"), "the token leaked: {line}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
