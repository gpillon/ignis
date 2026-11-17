//! The connector's MCP server (spec #656, ticket #657), driven as a harness
//! drives it: MCP's JSON-RPC over Streamable HTTP, `POST /v1/mcp` with the
//! `Accept` a harness sends and a `Host` that is not a loopback name, on the
//! mock-backed router (`support/switch.rs`'s loader for the switch).
//!
//! What a harness sees: `initialize` with ignis's `instructions`, a tool list
//! whose bytes no switch, patch or model changes, `read_catalog` answering
//! the catalog as text, a refused route coming back as a tool error with the
//! server's own JSON and `Retry-After`, the key, and `connector.enabled`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_artifact::vision::ProcessorOptions;
use ignis_core::image::ImageSize;
use ignis_server::config::file::RealFiles;
use ignis_server::config::{resolve_with, ApiKey, Config, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::download::Catalog;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::model_switch::{CatalogModels, Switcher};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/media.rs"]
mod media;

const PATIENT: Duration = Duration::from_secs(30);
const KEY: &str = "sk-connector-test";

/// The protocol version the tests' harness asks for in `initialize`, and
/// names on every request after it, as MCP's Streamable HTTP has a client do.
const PROTOCOL: &str = "2025-06-18";

/// What a harness's MCP client sends on every POST: both media types, the
/// server's own address as `Host`, here a name a tunnel or a LAN would give
/// it, which rmcp refuses unless its Host check is off, and after
/// `initialize` the protocol version it negotiated.
fn mcp(body: &Value, key: Option<&str>) -> Request<Body> {
    let mut request = Request::builder()
        .method(Method::POST)
        .uri("/v1/mcp")
        .header(header::HOST, "ignis.example:8000")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if body["method"] != "initialize" {
        request = request.header("mcp-protocol-version", PROTOCOL);
    }
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    request.body(Body::from(body.to_string())).unwrap()
}

async fn exchange(app: &axum::Router, request: Request<Body>) -> (StatusCode, HeaderMap, Bytes) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes)
}

/// One JSON-RPC request with `id` 1, answered `200 application/json`: its
/// raw body.
async fn rpc_bytes(app: &axum::Router, method: &str, params: Value, key: Option<&str>) -> Bytes {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
    let (status, headers, bytes) = exchange(app, mcp(&body, key)).await;
    assert_eq!(status, StatusCode::OK, "{method}: {}", String::from_utf8_lossy(&bytes));
    assert!(headers[header::CONTENT_TYPE].to_str().unwrap().starts_with("application/json"), "{method}: {headers:?}");
    bytes
}

/// [`rpc_bytes`]'s `result`.
async fn rpc(app: &axum::Router, method: &str, params: Value, key: Option<&str>) -> Value {
    let answer: Value = serde_json::from_slice(&rpc_bytes(app, method, params, key).await).unwrap();
    assert_eq!(answer["id"], 1, "{answer}");
    answer.get("result").cloned().unwrap_or_else(|| panic!("{method}: no result: {answer}"))
}

fn initialize() -> Value {
    json!({ "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": { "name": "a-harness", "version": "1.0" } })
}

fn call(name: &str) -> Value {
    json!({ "name": name, "arguments": {} })
}

async fn get(app: &axum::Router, uri: &str, key: Option<&str>) -> (StatusCode, HeaderMap, Value) {
    let mut request = Request::builder().uri(uri);
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let (status, headers, bytes) = exchange(app, request.body(Body::empty()).unwrap()).await;
    (status, headers, serde_json::from_slice(&bytes).unwrap())
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Value) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let (status, _, bytes) = exchange(app, request.body(Body::from(body.to_string())).unwrap()).await;
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned())))
}

/// `GET /v1/models` until its `status` is `wanted` (scheduling turns,
/// bounded).
async fn until_status(app: &axum::Router, wanted: &str) {
    for _ in 0..10_000 {
        let (status, _, body) = get(app, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == wanted {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported {wanted}");
}

/// A config file of its own for one test, under the system temp directory:
/// no test reads the operator's.
fn config_file(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-connector-http-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ignis.config.yaml");
    std::fs::write(&path, "server:\n  request_timeout: 30\n").unwrap();
    path
}

fn started(argv: &[&str]) -> Config {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    match resolve_with(&argv, |_: &str| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// A mock-backed server on `mock-a`, with a switcher and the configuration
/// a start builds from `config`, its connector as `config` says.
fn server_on(loader: &Arc<MockLoader>, config: Config) -> Server {
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles)));
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
        .with_config(state)
        .with_connector(config.settings().connector.clone());
    server.apply_config(&config);
    server
}

fn server(test: &str) -> (Arc<MockLoader>, Server) {
    let loader = MockLoader::new();
    let path = config_file(test);
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap(), "--profile", "none"]));
    (loader, server)
}

// ── initialize ───────────────────────────────────────────────────────────

/// `initialize` names ignis and carries its `instructions`: under 1 KB, the
/// first line saying what ignis is, the same bytes from a second start with
/// the same configuration. Stateless: no session id, so a restart strands
/// no harness. Any `Host` and any `Origin` are admitted.
#[tokio::test]
async fn initialize_answers_ignis_with_its_instructions() {
    let (_, first) = server("initialize-a");
    let app = first.app();
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": initialize() });
    let mut request = mcp(&body, None);
    request.headers_mut().insert(header::ORIGIN, "https://somewhere.example".parse().unwrap());
    let (status, headers, bytes) = exchange(&app, request).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
    assert!(headers.get("mcp-session-id").is_none(), "stateless: {headers:?}");
    let answer: Value = serde_json::from_slice(&bytes).unwrap();
    let result = &answer["result"];
    assert_eq!(result["protocolVersion"], PROTOCOL, "{result}");
    assert_eq!(result["serverInfo"]["name"], "ignis", "{result}");
    assert_eq!(result["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(result["capabilities"]["tools"].is_object(), "{result}");
    let instructions = result["instructions"].as_str().expect("instructions");
    assert!(instructions.len() < 1024, "{} bytes: {instructions}", instructions.len());
    let first_line = instructions.lines().next().unwrap();
    assert!(first_line.starts_with("ignis") && first_line.contains("inference server"), "{first_line}");
    // An image's cost: the switching, `503` for other clients meanwhile, and
    // vision to see it.
    let image = instructions.lines().find(|line| line.starts_with("generate_image")).expect("the image's line");
    for cost in ["20 s", "503", "vision"] {
        assert!(image.contains(cost), "{cost}: {image}");
    }

    // The harness's acknowledgement is accepted and answered nothing.
    let initialized = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
    let (status, _, _) = exchange(&app, mcp(&initialized, None)).await;
    assert_eq!(status, StatusCode::ACCEPTED);

    let (_, second) = server("initialize-b");
    let again = rpc(&second.app(), "initialize", initialize(), None).await;
    assert_eq!(again["instructions"], result["instructions"]);
}

/// Spec #656: the whole of `instructions` stays under 1 KB with every tool
/// listed, `connector.allow_changes` on.
#[tokio::test]
async fn the_instructions_stay_under_a_kilobyte_with_the_change_tools_listed() {
    let loader = MockLoader::new();
    let path = config_file("instructions-changes");
    let config = started(&["--config", path.to_str().unwrap(), "--profile", "none", "--connector-allow-changes", "true"]);
    assert!(config.settings().connector.allow_changes);
    let app = server_on(&loader, config).app();
    let result = rpc(&app, "initialize", initialize(), None).await;
    let instructions = result["instructions"].as_str().expect("instructions");
    assert!(instructions.len() < 1024, "{} bytes: {instructions}", instructions.len());
}

// ── the tool list ────────────────────────────────────────────────────────

/// `tools/list`'s raw answer, the request's `id` the same every time.
async fn tools_list(app: &axum::Router) -> Bytes {
    rpc_bytes(app, "tools/list", json!({}), None).await
}

/// Spec #656: the tool list is built once at start, so its bytes are the
/// same whatever model is loaded, during and after a switch, after a
/// config patch; and neither connector field can be patched. `/v1/mcp`
/// answers while a switch holds the rest of `/v1`.
#[tokio::test]
async fn the_tool_list_is_the_same_bytes_through_a_switch_a_patch_and_whatever_model_is_loaded() {
    let (loader, server) = server("tools-list");
    let app = server.app();
    let before = tools_list(&app).await;
    let listed: Value = serde_json::from_slice(&before).unwrap();
    let tools = listed["result"]["tools"].as_array().expect("tools");
    let names: Vec<&str> = tools.iter().map(|tool| tool["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["describe_config", "generate_image", "read_catalog", "read_config"]);
    let catalog = &tools[2];
    assert_eq!(catalog["annotations"], json!({ "readOnlyHint": true, "destructiveHint": false }));
    assert_eq!(catalog["inputSchema"], json!({ "type": "object", "properties": {} }));
    assert!(!catalog["description"].as_str().unwrap().is_empty());
    // Spec #656: `generate_image` changes what the server has loaded, so it
    // is not read-only; it takes a prompt and, optionally, a size.
    let image = &tools[1];
    assert_eq!(image["annotations"], json!({ "readOnlyHint": false, "destructiveHint": false }));
    assert_eq!(image["inputSchema"]["required"], json!(["prompt"]));
    let mut inputs: Vec<&String> = image["inputSchema"]["properties"].as_object().expect("properties").keys().collect();
    inputs.sort();
    assert_eq!(inputs, ["prompt", "size"]);
    let description = image["description"].as_str().unwrap();
    for cost in ["503", "vision", "generate-image"] {
        assert!(description.contains(cost), "{cost}: {description}");
    }

    // A switch held in its load: `/v1/mcp` answers, a chat request is held.
    let release = loader.hold_next_load();
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", json!({ "artifact": "mock-b.ninfer", "model": "mock-b" })).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_status(&app, "switching").await;
    let chat = json!({ "model": "mock-a", "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 3 });
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", chat).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")));
    assert_eq!(tools_list(&app).await, before, "during the switch");
    release.send(()).expect("the held load is waiting");
    until_status(&app, "serving").await;
    assert_eq!(server.active().model_id(), "mock-b");
    assert_eq!(tools_list(&app).await, before, "on another model");

    // A live patch, then a patch of either connector field, refused.
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", json!({ "server": { "request_timeout": 91 } })).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(tools_list(&app).await, before, "after a patch");
    for body in [json!({ "connector": { "allow_changes": true } }), json!({ "connector": { "enabled": false } })] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(refused["error"]["code"], "config_field_not_patchable", "{refused}");
    }
    assert_eq!(tools_list(&app).await, before);

    // The image model loaded.
    let image = "ignis-image-2.1-turbo";
    let switch = json!({ "artifact": switch_support::source(image).artifact, "model": image });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", switch).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_status(&app, "serving").await;
    assert_eq!(server.active().model_id(), image);
    assert_eq!(tools_list(&app).await, before, "with the image model loaded");
}

// ── read_catalog ─────────────────────────────────────────────────────────

/// `read_catalog` calls `GET /v1/models/catalog` with the caller's key and
/// answers its body as one text, nothing structured; a tool the list does
/// not hold is a protocol error.
#[tokio::test]
async fn read_catalog_answers_the_catalog_as_text_with_the_callers_key() {
    let (_, server) = server("read-catalog");
    let app = server.with_api_key(ApiKey::new(KEY)).app();
    let result = rpc(&app, "tools/call", call("read_catalog"), Some(KEY)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert!(result.get("structuredContent").is_none(), "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    assert_eq!(content[0]["type"], "text");
    let catalog: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    let (status, _, direct) = get(&app, "/v1/models/catalog", Some(KEY)).await;
    assert_eq!(status, StatusCode::OK, "{direct}");
    assert_eq!(catalog, direct);
    assert!(catalog["data"].as_array().unwrap().iter().any(|row| row["id"] == "mock-a" && row["loaded"] == true), "{catalog}");

    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": call("switch_model") });
    let (status, _, bytes) = exchange(&app, mcp(&body, Some(KEY))).await;
    assert_eq!(status, StatusCode::OK);
    let answer: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(answer["error"]["code"], -32602, "{answer}");
}

// ── errors ───────────────────────────────────────────────────────────────

/// A route that refuses comes back as the tool's error: the server's own
/// JSON, the route's answer to a client, then its `Retry-After`; answered
/// at once, never retried. The MCP route itself answers while the model
/// warms up: only the tool's own route is held.
#[tokio::test]
async fn a_refused_route_is_the_tools_error_with_the_servers_json_and_its_retry_after_unretried() {
    let (_, server) = server("warming");
    let app = server.with_warm_up().app();
    rpc(&app, "initialize", initialize(), None).await;
    rpc(&app, "tools/list", json!({}), None).await;
    let (status, headers, direct) = get(&app, "/v1/models/catalog", None).await;
    assert_eq!((status, direct["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("server_not_ready")));
    assert_eq!(headers[header::RETRY_AFTER], "5");

    let began = Instant::now();
    let result = rpc(&app, "tools/call", call("read_catalog"), None).await;
    // A connector honouring `Retry-After` would have waited 5 s.
    assert!(began.elapsed() < Duration::from_secs(5), "{:?}", began.elapsed());
    assert_eq!(result["isError"], true, "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 2, "{result}");
    let error: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(error, direct);
    assert_eq!(content[1]["text"], "Retry-After: 5");
}

// ── the key and the switch ───────────────────────────────────────────────

/// `/v1/mcp` sits behind the key the rest of `/v1` does.
#[tokio::test]
async fn without_the_key_the_mcp_route_is_refused() {
    let (_, server) = server("key");
    let app = server.with_api_key(ApiKey::new(KEY)).app();
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": initialize() });
    for key in [None, Some("sk-wrong")] {
        let (status, headers, bytes) = exchange(&app, mcp(&body, key)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{key:?}");
        assert_eq!(headers[header::WWW_AUTHENTICATE], "Bearer");
        let refused: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(refused["error"]["code"], "invalid_api_key", "{refused}");
    }
    rpc(&app, "initialize", initialize(), Some(KEY)).await;
}

/// `connector.enabled` off: `/v1/mcp` answers `404`, naming the field, and
/// the rest of `/v1` serves as before.
#[tokio::test]
async fn with_the_connector_off_the_route_answers_404() {
    let loader = MockLoader::new();
    let path = config_file("off");
    let config = started(&["--config", path.to_str().unwrap(), "--profile", "none", "--connector-enabled", "false"]);
    assert_eq!((config.settings().connector.enabled, config.settings().connector.allow_changes), (false, false));
    let app = server_on(&loader, config).app();
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": initialize() });
    let (status, _, bytes) = exchange(&app, mcp(&body, None)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let refused: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(refused["error"]["code"], "connector_disabled", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("connector.enabled"), "{refused}");
    assert_eq!(get(&app, "/v1/models", None).await.0, StatusCode::OK);
}

// ── read_config and describe_config ──────────────────────────────────────

fn call_with(name: &str, arguments: Value) -> Value {
    json!({ "name": name, "arguments": arguments })
}

fn text_of(result: &Value) -> String {
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    content[0]["text"].as_str().unwrap().to_owned()
}

/// `read_config` answers `GET /v1/config?format=yaml` as text, read-only.
#[tokio::test]
async fn read_config_answers_the_running_configuration_as_yaml() {
    let (_, server) = server("read-config");
    let app = server.with_api_key(ApiKey::new(KEY)).app();
    let listed = rpc(&app, "tools/list", json!({}), Some(KEY)).await;
    let tool = listed["tools"].as_array().unwrap().iter().find(|tool| tool["name"] == "read_config").expect("listed");
    assert_eq!(tool["annotations"], json!({ "readOnlyHint": true, "destructiveHint": false }));
    assert_eq!(tool["inputSchema"], json!({ "type": "object", "properties": {} }));

    let result = rpc(&app, "tools/call", call("read_config"), Some(KEY)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let text = text_of(&result);
    let request = Request::builder().uri("/v1/config?format=yaml").header(header::AUTHORIZATION, format!("Bearer {KEY}"));
    let (status, _, direct) = exchange(&app, request.body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(text.as_bytes(), &direct[..]);
    assert!(text.contains("server:") && !text.trim_start().starts_with('{'), "{text}");
}

/// `describe_config` asks by group or by names (a list), requires one of
/// them, and lists the groups its description reads from the schema.
#[tokio::test]
async fn describe_config_asks_by_group_or_names_and_requires_one() {
    let (_, server) = server("describe-config");
    let app = server.app();
    let listed = rpc(&app, "tools/list", json!({}), None).await;
    let tool = listed["tools"].as_array().unwrap().iter().find(|tool| tool["name"] == "describe_config").expect("listed");
    assert_eq!(tool["annotations"], json!({ "readOnlyHint": true, "destructiveHint": false }));
    let description = tool["description"].as_str().unwrap();
    for group in ignis_server::config_http::visible_groups() {
        assert!(description.contains(group), "{group} in {description}");
    }
    assert!(description.contains(&ignis_server::config_http::groups_sentence()));
    assert_eq!(tool["inputSchema"]["properties"]["names"]["type"], "array");

    let result = rpc(&app, "tools/call", call_with("describe_config", json!({ "group": "vram" })), None).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let (_, _, direct) = get(&app, "/v1/config/fields?group=vram", None).await;
    assert_eq!(serde_json::from_str::<Value>(&text_of(&result)).unwrap(), direct);

    let result = rpc(&app, "tools/call", call_with("describe_config", json!({ "names": ["vram.budget_bytes", "server.bind"] })), None).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let fields: Value = serde_json::from_str(&text_of(&result)).unwrap();
    let keys: Vec<&str> = fields["data"].as_array().unwrap().iter().map(|field| field["file_key"].as_str().unwrap()).collect();
    assert_eq!(keys.len(), 2, "{fields}");
    assert!(keys.contains(&"vram.budget_bytes") && keys.contains(&"server.bind"));

    // Neither: refused before the route, a tool error.
    for arguments in [json!({}), json!({ "group": null })] {
        let result = rpc(&app, "tools/call", call_with("describe_config", arguments), None).await;
        assert_eq!(result["isError"], true, "{result}");
        assert!(text_of(&result).contains("`group` or `names`"), "{result}");
    }

    // An unknown group is the route's error, verbatim.
    let result = rpc(&app, "tools/call", call_with("describe_config", json!({ "group": "nope" })), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let error: Value = serde_json::from_str(&text_of(&result)).unwrap();
    assert_eq!(error["error"]["code"], "config_invalid", "{error}");
}

// ── generate_image ───────────────────────────────────────────────────────

/// The image model's served id: its family's own, the built-in catalog's
/// image entry.
const IMAGE: &str = "ignis-image-2.1-turbo";

fn acquirer() -> Arc<MediaAcquirer> {
    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    Arc::new(MediaAcquirer::new(Arc::new(media::processor(limits.clone())), limits, MediaPolicy::new(false, 0)))
}

/// A server on `mock-a` with vision on, its catalog the built-in one over an
/// empty `download.path`: the ids in `on_disk` are on disk at the mock's
/// artifact, so the image entry is `complete` when it is one of them. With
/// `implicit`, a request naming one of them switches to it
/// (`switch.allow_implicit`); without, only a switch by id does. It makes
/// two image sizes, one past 512 px and one within.
fn image_server(test: &str, on_disk: &[&str], implicit: bool) -> (Arc<MockLoader>, Server) {
    let loader = MockLoader::new();
    loader.vision("mock-a", acquirer());
    let table: BTreeMap<String, PathBuf> = on_disk.iter().map(|id| (id.to_string(), switch_support::source(id).artifact)).collect();
    let dir = std::env::temp_dir().join(format!("ignis-connector-http-{test}-models-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = CatalogModels { catalog: Arc::new(Catalog::built_in().clone()), dir };
    let switcher = Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(catalog, implicit);
    let switcher = if implicit { switcher.with_known_models(table) } else { switcher.with_listed_models(table) };
    let server = Server::from_active(loader.model("mock-a")).with_switcher(switcher);
    server.update_live(|live| live.image.sizes = vec![ImageSize::new(1024, 640), ImageSize::new(256, 128)]);
    (loader, server)
}

/// `generate_image`'s `result` for `arguments`.
async fn generate(app: &axum::Router, arguments: Value, key: Option<&str>) -> Value {
    rpc(app, "tools/call", json!({ "name": "generate_image", "arguments": arguments }), key).await
}

fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let mut out = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        let digits: Vec<u8> = chunk.iter().filter(|&&c| c != b'=').map(|&c| value(c)).collect();
        let n = digits.iter().enumerate().fold(0u32, |n, (i, &d)| n | u32::from(d) << (18 - 6 * i));
        out.extend((0..digits.len() - 1).map(|i| (n >> (16 - 8 * i)) as u8));
    }
    out
}

/// The width and height of a result's image block, a PNG.
fn image_size(block: &Value) -> (u32, u32) {
    assert_eq!(block["type"], "image", "{block}");
    assert_eq!(block["mimeType"], "image/png", "{block}");
    let reader = png::Decoder::new(std::io::Cursor::new(unbase64(block["data"].as_str().unwrap()))).read_info().expect("a PNG");
    (reader.info().width, reader.info().height)
}

/// Spec #656: with vision on, `generate_image` answers the image first, a
/// PNG preview whose longest edge is at most 512 px and never enlarged, then
/// one line of text: the size made, that this is a preview, and the skill's
/// `generate-image` script for the full size. The prompt reaches the image
/// model as sent.
#[tokio::test]
async fn generate_image_answers_a_preview_of_at_most_512_px_and_one_line_of_text() {
    let (loader, server) = image_server("preview", &[IMAGE, "mock-a"], true);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a red fox in the snow", "size": "1024x640" }), None).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 2, "{result}");
    assert_eq!(image_size(&content[0]), (512, 320));
    assert_eq!(content[1]["type"], "text", "{result}");
    let line = content[1]["text"].as_str().unwrap();
    assert_eq!(line.lines().count(), 1, "{line}");
    for said in ["1024x640", "512x320", "preview", "generate-image"] {
        assert!(line.contains(said), "{said}: {line}");
    }
    assert_eq!(loader.encoded_prompts(), ["a red fox in the snow"]);

    // An image within 512 px is shown at its own size.
    let result = generate(&app, json!({ "prompt": "a small fox", "size": "256x128" }), None).await;
    let content = result["content"].as_array().expect("content");
    assert_eq!(image_size(&content[0]), (256, 128), "{result}");
    assert!(content[1]["text"].as_str().unwrap().contains("256x128"), "{result}");
}

/// A chat turn naming `model`, sent with `key`, as a harness's next turn.
async fn chat_turn(app: &axum::Router, model: &str, key: &str) -> (StatusCode, Value) {
    let body = json!({ "model": model, "messages": [{ "role": "user", "content": "what did you draw?" }], "max_tokens": 3 });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {key}"));
    let (status, _, bytes) = exchange(app, request.body(Body::from(body.to_string())).unwrap()).await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

/// Spec #656: `generate_image` reads the catalog and names the image model
/// it has on disk, with the caller's key, so the server switches to it (a
/// request naming no model is refused by a text model). No tool switches
/// back: the agent's next chat turn names its own model, and the implicit
/// switch returns to it.
#[tokio::test]
async fn generate_image_names_the_catalogs_image_model_and_the_next_chat_turn_switches_back() {
    let (loader, server) = image_server("switch-back", &[IMAGE, "mock-a"], true);
    let app = server.clone().with_api_key(ApiKey::new(KEY)).app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk", "size": "1024x640" }), Some(KEY)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert_eq!(loader.loads(), [IMAGE]);
    assert_eq!(server.active().model_id(), IMAGE);

    let (status, answer) = chat_turn(&app, "mock-a", KEY).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    assert_eq!(answer["model"], "mock-a");
    assert_eq!(loader.loads(), [IMAGE, "mock-a"]);
    assert_eq!(server.active().model_id(), "mock-a");
}

/// Spec #656: with no image model on disk, `generate_image` is the tool's
/// error, the server's `model_not_found`, naming the command that fetches
/// the catalog's image model (and not `download_model`, which is not
/// listed); nothing loads.
#[tokio::test]
async fn without_the_image_model_on_disk_generate_image_names_its_download() {
    let (loader, server) = image_server("absent", &["mock-a"], true);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    let error: Value = serde_json::from_str(content[0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(error["error"]["code"], "model_not_found", "{error}");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains(&format!("`ignis-server model download {IMAGE}`")), "{message}");
    assert!(!message.contains("download_model"), "{message}");
    assert!(loader.loads().is_empty());
    assert_eq!(server.active().model_id(), "mock-a");
}

/// Spec #656: with `switch.allow_implicit` off, the image route refuses the
/// image model by name, `404`, and `generate_image` passes that answer on
/// as its error, word for word; nothing loads.
#[tokio::test]
async fn with_implicit_switching_off_generate_image_passes_the_servers_404_on() {
    let (loader, server) = image_server("implicit-off", &[IMAGE, "mock-a"], false);
    let app = server.app();
    let result = generate(&app, json!({ "prompt": "a lighthouse at dusk" }), None).await;
    assert_eq!(result["isError"], true, "{result}");
    let error: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    let (status, direct) = send(&app, Method::POST, "/v1/images/generations", json!({ "model": IMAGE, "prompt": "a lighthouse at dusk" })).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{direct}");
    assert_eq!(error, direct);
    assert!(loader.loads().is_empty());
}
