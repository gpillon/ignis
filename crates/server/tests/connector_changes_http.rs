//! The connector's change tools (spec #656, ticket #660), driven as a
//! harness drives them: `POST /v1/mcp` over the mock-backed router with a
//! temporary config file and `download.path`, a local endpoint serving small
//! fake files for the downloads (`support/repo.rs`): no network, no GPU.
//!
//! With `connector.allow_changes` on, `change_config`, `download_model` and
//! `cancel_download` are listed, the first and last marked destructive; with
//! it off, the default, they are not in the list and not callable.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use ignis_server::config::file::RealFiles;
use ignis_server::config::Config;
use ignis_server::config_http::ConfigState;
use ignis_server::download_http::Downloads;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/repo.rs"]
mod repo;
use repo::{catalog_served, files_of, Repo};

#[path = "support/model_routes.rs"]
mod routes;
use routes::{config, files_in, row, until_row, until_serving, PATIENT};

#[path = "support/harness.rs"]
mod harness;
use harness::{config_file, initialize, rpc, rpc_bytes};

const CHANGES: [&str; 3] = ["cancel_download", "change_config", "download_model"];

/// A server on `mock-a` with a switcher over a catalog of `acme-new`,
/// downloads from `repo`, the config state a start builds and a connector as
/// the start's `allow_changes` says; with the config file and the models
/// folder it uses.
fn server_on(loader: &Arc<MockLoader>, repo: &Repo, tag: &str, allow_changes: bool) -> (Server, PathBuf, PathBuf) {
    let path = config_file(tag);
    let models = catalog_served(repo, tag, &[("acme-new", 32)]);
    let dir = models.dir.clone();
    let flag = if allow_changes { "true" } else { "false" };
    let config: Config = config(repo, &["--config", path.to_str().unwrap(), "--profile", "none", "--connector-allow-changes", flag]);
    assert_eq!(config.settings().connector.allow_changes, allow_changes);
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles)));
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_catalog(models, true))
        .with_downloads(Downloads::of(&config, None))
        .with_config(state)
        .with_connector(config.settings().connector.clone());
    server.apply_config(&config);
    (server, path, dir)
}

/// A `tools/call` result.
async fn call(app: &axum::Router, name: &str, arguments: Value) -> Value {
    rpc(app, "tools/call", json!({ "name": name, "arguments": arguments }), None).await
}

/// The one text a result carries.
fn text_of(result: &Value) -> String {
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    content[0]["text"].as_str().unwrap().to_owned()
}

/// The server's JSON body a result carries as its text.
fn body_of(result: &Value) -> Value {
    serde_json::from_str(&text_of(result)).unwrap_or_else(|err| panic!("{err}: {result}"))
}

async fn tools_list(app: &axum::Router) -> Bytes {
    rpc_bytes(app, "tools/list", json!({}), None).await
}

fn names_of(listed: &Bytes) -> Vec<String> {
    let listed: Value = serde_json::from_slice(listed).unwrap();
    listed["result"]["tools"].as_array().unwrap().iter().map(|tool| tool["name"].as_str().unwrap().to_owned()).collect()
}

// ── the list ─────────────────────────────────────────────────────────────

/// Both lists, pinned: off (the default), the five tools that read or ask;
/// on, those and the three that change the server, in name order, the change
/// tools with their hints and inputs. Each list is the same bytes through a
/// patch, and `instructions` name the change tools last and only when they
/// are listed.
#[tokio::test]
async fn the_list_has_the_change_tools_only_with_allow_changes_and_is_byte_stable_per_start() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();

    let (off, _, off_dir) = server_on(&loader, &repo, "list-off", false);
    let off = off.app();
    let off_list = tools_list(&off).await;
    assert_eq!(names_of(&off_list), ["decide", "describe_config", "generate_image", "read_catalog", "read_config"]);
    let result = rpc(&off, "initialize", initialize(), None).await;
    let instructions = result["instructions"].as_str().unwrap();
    for name in CHANGES {
        assert!(!instructions.contains(name), "{name}: {instructions}");
    }
    // Not listed, not callable: a protocol error, nothing changed.
    let params = json!({ "name": "change_config", "arguments": { "server": { "request_timeout": 5 } } });
    let answer: Value = serde_json::from_slice(&rpc_bytes(&off, "tools/call", params, None).await).unwrap();
    assert!(answer.get("error").is_some(), "{answer}");

    let loader = MockLoader::new();
    let (on, _, on_dir) = server_on(&loader, &repo, "list-on", true);
    let on = on.app();
    let on_list = tools_list(&on).await;
    assert_eq!(
        names_of(&on_list),
        ["cancel_download", "change_config", "decide", "describe_config", "download_model", "generate_image", "read_catalog", "read_config"]
    );
    let listed: Value = serde_json::from_slice(&on_list).unwrap();
    let tool = |name: &str| listed["result"]["tools"].as_array().unwrap().iter().find(|tool| tool["name"] == name).unwrap().clone();
    assert_eq!(tool("change_config")["annotations"], json!({ "readOnlyHint": false, "destructiveHint": true }));
    assert_eq!(tool("cancel_download")["annotations"], json!({ "readOnlyHint": false, "destructiveHint": true }));
    assert_eq!(tool("download_model")["annotations"], json!({ "readOnlyHint": false, "destructiveHint": false }));
    // A free object: the patch itself, not the configuration's fields.
    let input = &tool("change_config")["inputSchema"];
    assert_eq!((input["type"].clone(), input["additionalProperties"].clone()), (json!("object"), json!(true)), "{input}");
    assert!(input.get("required").is_none() && input.get("properties").is_none(), "{input}");
    assert_eq!(tool("download_model")["inputSchema"]["required"], json!(["model"]));
    assert_eq!(tool("cancel_download")["inputSchema"], json!({ "type": "object", "properties": {} }));
    let description = tool("change_config")["description"].as_str().unwrap().to_owned();
    for warning in ["reload", "prompt state", "null"] {
        assert!(description.contains(warning), "{warning}: {description}");
    }
    let result = rpc(&on, "initialize", initialize(), None).await;
    let instructions = result["instructions"].as_str().unwrap();
    assert!(instructions.len() < 1024, "{} bytes", instructions.len());
    let last = instructions.lines().last().unwrap();
    for name in CHANGES {
        assert!(last.contains(name), "{name}: {instructions}");
    }

    // Through a patch that applies live, each list is the same bytes.
    let patched = call(&on, "change_config", json!({ "server": { "request_timeout": 91 } })).await;
    assert_ne!(patched["isError"], true, "{patched}");
    assert_eq!(tools_list(&on).await, on_list, "after a patch");
    assert_eq!(tools_list(&off).await, off_list);
    let _ = std::fs::remove_dir_all(on_dir);
    let _ = std::fs::remove_dir_all(off_dir);
}

// ── change_config ────────────────────────────────────────────────────────

/// A patch that applies live answers the server's own body as the text: what
/// applied, that nothing reloaded, the configuration now running; `null`
/// removes the key; a refused patch is the tool's error with the server's
/// JSON.
#[tokio::test]
async fn change_config_patches_live_removes_with_null_and_passes_a_refusal_on() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let (server, path, dir) = server_on(&loader, &repo, "change-live", true);
    let app = server.clone().app();

    let result = call(&app, "change_config", json!({ "server": { "request_timeout": 91 } })).await;
    assert_ne!(result["isError"], true, "{result}");
    let applied = body_of(&result);
    assert_eq!((applied["applied"].clone(), applied["persisted"].clone()), (json!("live"), json!(true)), "{applied}");
    assert_eq!(applied["config"]["server"]["request_timeout"], 91);
    assert_eq!(server.live().request_timeout, Duration::from_secs(91));
    assert!(std::fs::read_to_string(&path).unwrap().contains("request_timeout: 91"));

    // null removes the key: the file loses it.
    let result = call(&app, "change_config", json!({ "server": { "request_timeout": null } })).await;
    assert_ne!(result["isError"], true, "{result}");
    assert!(!std::fs::read_to_string(&path).unwrap().contains("request_timeout"));

    // The connector's own fields cannot be patched: the route says so.
    let result = call(&app, "change_config", json!({ "connector": { "allow_changes": false } })).await;
    assert_eq!(result["isError"], true, "{result}");
    assert_eq!(body_of(&result)["error"]["code"], "config_field_not_patchable");
    let _ = std::fs::remove_dir_all(dir);
}

/// A patch with a reload field reloads the loaded model: the tool's result
/// is the server's `202` body, saying it reloaded, the model switching, and
/// the change lands once the model serves.
#[tokio::test]
async fn change_config_with_a_reload_field_answers_that_it_reloaded() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let (server, path, dir) = server_on(&loader, &repo, "change-reload", true);
    let app = server.clone().app();

    let release = loader.hold_next_load();
    let result = call(&app, "change_config", json!({ "server": { "request_timeout": 91 }, "reuse": { "kv_host_pool_bytes": "3G" } })).await;
    assert_ne!(result["isError"], true, "{result}");
    let applied = body_of(&result);
    assert_eq!(applied["applied"], "reload", "{applied}");
    assert_eq!(applied["switching"], json!({ "from": "mock-a", "to": "mock-a" }));
    assert_eq!(server.live().request_timeout, Duration::from_secs(30), "nothing of the patch before the reload serves");
    release.send(()).expect("the held load is waiting");
    until_serving(&app, &server, |server| server.live().request_timeout == Duration::from_secs(91)).await;
    assert!(std::fs::read_to_string(&path).unwrap().contains("kv_host_pool_bytes"));
    let handed = loader.prepared_options();
    let options = handed.last().unwrap().as_ref().expect("the reload hands the patched config to the loader");
    assert_eq!(options.host_pool_bytes, 3 << 30);
    let _ = std::fs::remove_dir_all(dir);
}

// ── download_model and cancel_download ───────────────────────────────────

/// A download started through the tool answers at once, naming the entry,
/// and `read_catalog` shows it running; `cancel_download` stops it keeping
/// its partial files; a second cancel is the tool's error with the server's
/// `download_not_running`; an unknown id is `model_not_found`.
#[tokio::test]
async fn a_download_is_started_with_download_model_followed_in_the_catalog_and_cancelled() {
    let repo = Repo::start().await;
    let loader = MockLoader::new();
    let (server, _, dir) = server_on(&loader, &repo, "download", true);
    let app = server.app();
    let files = files_of("acme-new");
    repo.hold(&files.artifact.0, 4096);

    let result = call(&app, "download_model", json!({ "model": "nope" })).await;
    assert_eq!(result["isError"], true, "{result}");
    assert_eq!(body_of(&result)["error"]["code"], "model_not_found");

    let result = call(&app, "download_model", json!({ "model": "acme-new" })).await;
    assert_ne!(result["isError"], true, "{result}");
    let accepted = body_of(&result);
    assert_eq!((accepted["status"].clone(), accepted["model"].clone()), (json!("downloading"), json!("acme-new")), "{accepted}");
    // Its progress is read through `read_catalog`.
    until_row(&app, "acme-new", |row| row["bytes_on_disk"] == json!(1 + 4096) && row["download"] == "running").await;
    let catalog = body_of(&call(&app, "read_catalog", json!({})).await);
    assert_eq!(row(&catalog, "acme-new")["download"], "running", "{catalog}");

    let result = call(&app, "cancel_download", json!({})).await;
    assert_ne!(result["isError"], true, "{result}");
    let cancelled = body_of(&result);
    assert_eq!((cancelled["status"].clone(), cancelled["model"].clone()), (json!("cancelled"), json!("acme-new")), "{cancelled}");
    let part = format!("{}.part", files.artifact.0);
    assert_eq!(files_in(&dir), [files.sidecar.0.clone(), part], "what arrived stays");

    let result = call(&app, "cancel_download", json!({})).await;
    assert_eq!(result["isError"], true, "{result}");
    assert_eq!(body_of(&result)["error"]["code"], "download_not_running");
    repo.release();
    let _ = std::fs::remove_dir_all(dir);
}
