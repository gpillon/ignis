//! What the connector's tests share (spec #656), included via `#[path]` by
//! each `connector_*` test binary: a request to `/v1/mcp` as a coding
//! agent's harness sends it, its JSON-RPC answer, the configuration a start
//! with some flags resolves, and a config file of a test's own.

#![allow(dead_code)]

use std::path::PathBuf;

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use serde_json::{json, Value};
use tower::ServiceExt;

/// The protocol version the tests' harness asks for in `initialize`, and
/// names on every request after it, as MCP's Streamable HTTP has a client do.
pub const PROTOCOL: &str = "2025-06-18";

/// `body`, written as given, as a harness's MCP client POSTs it: both media
/// types, the server's own address as `Host`, here a name a tunnel or a LAN
/// would give it, which rmcp refuses unless its Host check is off, and
/// after `initialize` the protocol version it negotiated.
pub fn mcp_text(body: impl Into<String>, initialize: bool, key: Option<&str>) -> Request<Body> {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri("/v1/mcp")
        .header(header::HOST, "ignis.example:8000")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if !initialize {
        request = request.header("mcp-protocol-version", PROTOCOL);
    }
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    request.body(Body::from(body.into())).unwrap()
}

/// [`mcp_text`] of one JSON-RPC message.
pub fn mcp(body: &Value, key: Option<&str>) -> Request<Body> {
    mcp_text(body.to_string(), body["method"] == "initialize", key)
}

pub async fn exchange(app: &axum::Router, request: Request<Body>) -> (StatusCode, HeaderMap, Bytes) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes)
}

/// One JSON-RPC request with `id` 1, its `params` written as given (text,
/// so a test chooses the order of its keys), answered `200
/// application/json`: its raw body.
pub async fn rpc_text(app: &axum::Router, method: &str, params: &str, key: Option<&str>) -> Bytes {
    let body = format!(r#"{{"jsonrpc":"2.0","id":1,"method":"{method}","params":{params}}}"#);
    let (status, headers, bytes) = exchange(app, mcp_text(body, method == "initialize", key)).await;
    assert_eq!(status, StatusCode::OK, "{method}: {}", String::from_utf8_lossy(&bytes));
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().starts_with("application/json"), "{method}: {headers:?}");
    bytes
}

/// [`rpc_text`] of `params` as JSON.
pub async fn rpc_bytes(app: &axum::Router, method: &str, params: Value, key: Option<&str>) -> Bytes {
    rpc_text(app, method, &params.to_string(), key).await
}

/// [`rpc_bytes`]'s `result`.
pub async fn rpc(app: &axum::Router, method: &str, params: Value, key: Option<&str>) -> Value {
    result_of(method, &rpc_bytes(app, method, params, key).await)
}

/// The `result` of `method`'s answer, `bytes`, whose `id` is 1.
pub fn result_of(method: &str, bytes: &[u8]) -> Value {
    let answer: Value = serde_json::from_slice(bytes).unwrap();
    assert_eq!(answer["id"], 1, "{answer}");
    answer.get("result").cloned().unwrap_or_else(|| panic!("{method}: no result: {answer}"))
}

/// `initialize`'s `params`, as a harness sends them.
pub fn initialize() -> Value {
    json!({ "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": { "name": "a-harness", "version": "1.0" } })
}

/// A config file of its own for one test, under the system temp directory:
/// no test reads the operator's.
pub fn config_file(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-connector-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ignis.config.yaml");
    std::fs::write(&path, "server:\n  request_timeout: 30\n").unwrap();
    path
}

/// The configuration a start with `argv` runs.
pub fn started(argv: &[&str]) -> Config {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    match resolve_with(&argv, |_: &str| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}
