//! The connector's MCP server (spec #656, ticket #657), driven as a harness
//! drives it: MCP's JSON-RPC over Streamable HTTP, `POST /v1/mcp` with the
//! `Accept` a harness sends and a `Host` that is not a loopback name, on the
//! mock-backed router (`support/switch.rs`'s loader for the switch).
//!
//! What a harness sees: `initialize` with ignis's `instructions`, a tool list
//! whose bytes no switch, patch or model changes, `read_catalog` answering
//! the catalog as text, a refused route coming back as a tool error with the
//! server's own JSON and `Retry-After`, the key, and `connector.enabled`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, Method, Request, StatusCode};
use ignis_server::config::file::RealFiles;
use ignis_server::config::{ApiKey, Config};
use ignis_server::config_http::ConfigState;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/harness.rs"]
mod harness;
use harness::{config_file, exchange, initialize, mcp, rpc, rpc_bytes, started, PROTOCOL};

#[path = "support/repo.rs"]
mod repo;

#[path = "support/model_routes.rs"]
mod routes;
use routes::{send, until_serving, until_status, PATIENT};

const KEY: &str = "sk-connector-test";

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
    assert_eq!(names, ["decide", "describe_config", "generate_image", "read_catalog", "read_config"]);
    let catalog = &tools[3];
    assert_eq!(catalog["annotations"], json!({ "readOnlyHint": true, "destructiveHint": false }));
    assert_eq!(catalog["inputSchema"], json!({ "type": "object", "properties": {} }));
    assert!(!catalog["description"].as_str().unwrap().is_empty());
    // Spec #656: `generate_image` changes what the server has loaded, so it
    // is not read-only; it takes a prompt and, optionally, a size.
    let image = &tools[2];
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
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "mock-b.ninfer", "model": "mock-b" }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_status(&app, "switching").await;
    let chat = json!({ "model": "mock-a", "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 3 });
    let (status, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat)).await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::SERVICE_UNAVAILABLE, json!("model_switching")));
    assert_eq!(tools_list(&app).await, before, "during the switch");
    release.send(()).expect("the held load is waiting");
    until_serving(&app, &server, |server| server.active().model_id() == "mock-b").await;
    assert_eq!(tools_list(&app).await, before, "on another model");

    // A live patch, then a patch of either connector field, refused.
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 91 } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(tools_list(&app).await, before, "after a patch");
    for body in [json!({ "connector": { "allow_changes": true } }), json!({ "connector": { "enabled": false } })] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(refused["error"]["code"], "config_field_not_patchable", "{refused}");
    }
    assert_eq!(tools_list(&app).await, before);

    // The image model loaded.
    let image = "ignis-image-2.1-turbo";
    let switch = json!({ "artifact": switch_support::source(image).artifact, "model": image });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_serving(&app, &server, |server| server.active().model_id() == image).await;
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
