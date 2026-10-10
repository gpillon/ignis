//! `GET` and `PATCH /v1/config` (spec config-v2/02) over the mock-backed
//! router: the visible-only document, a live change applied at once with no
//! model touched, a change naming one reload field reloading the whole patch
//! through the model switch (`support/switch.rs`'s loader, held inside its
//! load, never by a timer — ADR 0006), a refused patch applying nothing, a
//! failed reload leaving the old model, the old configuration and the config
//! file as they were, and a server with no config file applying anyway and
//! saying so.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::config::file::{Format, RealFiles};
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

const PATIENT: Duration = Duration::from_secs(30);

/// A config file of its own for one test, under the system temp directory.
fn config_file(test: &str, text: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-config-http-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ignis.config.yaml");
    std::fs::write(&path, text).unwrap();
    path
}

fn started(argv: &[&str]) -> Config {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    match resolve_with(&argv, |_| None, &RealFiles).expect("the start options resolve") {
        ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    }
}

/// A mock-backed server running `config`, with a switcher and the config
/// state a start builds, the server's live knobs taken from `config`.
fn server_on(loader: &Arc<MockLoader>, config: Config) -> Server {
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles)));
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
        .with_config(state);
    server.apply_config(&config);
    server
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body)
}

async fn until_serving(app: &axum::Router) {
    for _ in 0..10_000 {
        let (status, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == "serving" {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("`GET /v1/models` never reported serving");
}

fn read_yaml(path: &std::path::Path) -> Value {
    Format::Yaml.read(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// Spec config-v2/02, Testing: the API key the server started with is
/// absent from the document — the key, not just its value — and the bind
/// address is present.
#[tokio::test]
async fn get_shows_the_visible_fields_and_never_the_key() {
    let loader = MockLoader::new();
    let config = started(&["--server-api-key", "sk-secret", "--server-bind", "127.0.0.1:7777"]);
    let app = server_on(&loader, config).app();
    let (status, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(status, StatusCode::OK, "{document}");
    assert!(document["server"].get("api_key").is_none(), "{document}");
    assert!(!document.to_string().contains("sk-secret"));
    assert_eq!(document["server"]["bind"], "127.0.0.1:7777");
    assert_eq!(document["server"]["request_timeout"], 30);
}

#[tokio::test]
async fn a_server_without_a_config_answers_501() {
    let loader = MockLoader::new();
    let app = Server::from_active(loader.model("mock-a")).app();
    let (status, body) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!((status, body["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("config_unavailable")));
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 60 } }))).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
}

/// Spec config-v2/02, Testing: a body of live fields applies at once, no
/// model touched — the loader never asked to prepare anything — and is
/// written into the config file, every other key kept.
#[tokio::test]
async fn a_live_patch_applies_at_once_without_touching_the_model_and_is_written_down() {
    let path = config_file("live", "server:\n  request_timeout: 30\nprofiles:\n  mine:\n    vram:\n      headroom_bytes: 2G\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let body = json!({ "server": { "request_timeout": 90, "developer_message_policy": "reject" }, "switch": { "drain_timeout": 5 } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["applied"], "live");
    assert_eq!(applied["persisted"], "yes", "{applied}");
    assert_eq!(applied["config"]["server"]["request_timeout"], 90);
    assert_eq!(server.live().request_timeout, Duration::from_secs(90));
    assert_eq!(server.live().instruction_policy.developer.as_str(), "reject");
    assert_eq!(server.switcher.as_ref().unwrap().drain_timeout(), Duration::from_secs(5));
    assert!(loader.prepared_options().is_empty(), "no model was prepared, so none was switched");
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["switch"]["drain_timeout"], 5);
    let written = read_yaml(&path);
    assert_eq!(written["server"]["request_timeout"], 90);
    assert_eq!(written["switch"]["drain_timeout"], 5);
    assert_eq!(written["profiles"]["mine"]["vram"]["headroom_bytes"], "2G", "the rest of the file is kept");
}

/// Spec config-v2/02, Testing: one reload field drives the whole patch
/// through the switch — and the live field beside it lands only with it.
#[tokio::test]
async fn a_patch_with_one_reload_field_reloads_with_the_whole_patch() {
    let path = config_file("reload", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let release = loader.hold_next_load();
    let body = json!({ "server": { "request_timeout": 91 }, "reuse": { "kv_host_pool_bytes": "3G" } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    assert_eq!(applied["applied"], "reload");
    assert_eq!(applied["persisted"], "after_reload");
    assert_eq!(applied["switching"], json!({ "from": "mock-a", "to": "mock-a" }));

    // Mid-reload: nothing of the patch is in force, and the config route
    // still answers, showing what runs.
    assert_eq!(server.live().request_timeout, Duration::from_secs(30), "never mixed: the live field waits for the reload");
    let (status, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(status, StatusCode::OK, "GET /v1/config stays open during a switch: {document}");
    assert_eq!(document["server"]["request_timeout"], 30);
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 5 } }))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "a second change waits for the reload");
    assert_eq!(read_yaml(&path)["server"]["request_timeout"], 30, "not written before the reload took");

    release.send(()).expect("the held load is waiting");
    until_serving(&app).await;
    let handed = loader.prepared_options();
    let options = handed.last().unwrap().as_ref().expect("the reload hands the patched config to the loader");
    assert_eq!((options.host_pool_bytes, options.request_timeout_secs), (3 << 30, 91));
    assert_eq!(server.live().request_timeout, Duration::from_secs(91));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["reuse"]["kv_host_pool_bytes"], "3G");
    let written = read_yaml(&path);
    assert_eq!((written["server"]["request_timeout"].clone(), written["reuse"]["kv_host_pool_bytes"].clone()), (json!(91), json!("3G")));
}

/// Spec config-v2/02, Testing: a body naming a field that is not patchable
/// is refused, and nothing in it applies — the live field beside it either.
#[tokio::test]
async fn a_patch_naming_an_unpatchable_field_applies_nothing() {
    let path = config_file("unpatchable", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let before = std::fs::read_to_string(&path).unwrap();
    for body in [
        json!({ "server": { "request_timeout": 92, "api_key": "sk-new" } }),
        json!({ "server": { "request_timeout": 92, "bind": "0.0.0.0:1" } }),
        json!({ "server": { "request_timeout": 92, "ui": false } }),
    ] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert_eq!(refused["error"]["code"], "config_field_not_patchable", "{refused}");
    }
    for (body, says) in [
        (json!({ "server": { "request_timeout": 0 } }), "server.request_timeout"),
        (json!({ "vision": { "max_tokens": 8192 } }), "--vision-enabled"),
        (json!({ "server": { "nope": 1 } }), "server.nope"),
        (json!({ "profiles": {} }), "profiles"),
    ] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(refused["error"]["message"].as_str().unwrap().contains(says), "{body}: {refused}");
    }
    assert_eq!(server.live().request_timeout, Duration::from_secs(30));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before, "the file untouched");
    assert!(loader.prepared_options().is_empty());
}

/// Spec config-v2/02, Testing: a reload that fails leaves the old model
/// serving and the config file as it was.
#[tokio::test]
async fn a_failed_reload_leaves_the_model_the_config_and_the_file_as_they_were() {
    let path = config_file("failed", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let before = std::fs::read_to_string(&path).unwrap();
    loader.refuse("mock-a");
    let body = json!({ "server": { "request_timeout": 93 }, "model": { "max_context": 65536 } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    until_serving(&app).await;
    assert_eq!(server.active().engine.model_id(), "mock-a");
    assert_eq!(server.live().request_timeout, Duration::from_secs(30));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["model"]["max_context"], ignis_server::config::DEFAULT_MAX_CONTEXT);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before, "a failed reload writes nothing");
}

/// Spec config-v2/02 §`PATCH` step 6: with no config file in use the change
/// still applies, and the answer says why it was not written down.
#[tokio::test]
async fn with_no_config_file_a_change_applies_and_says_it_was_not_written() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&[]));
    let app = server.app();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 94 } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["persisted"], "no");
    assert!(applied["reason"].as_str().unwrap().contains("no config file"), "{applied}");
    assert_eq!(server.live().request_timeout, Duration::from_secs(94));
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "prompt": false } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    assert_eq!(applied["persisted"], "no");
    until_serving(&app).await;
}

/// A placeholder start has no artifact to load again: a reload is refused,
/// while a live change still applies.
#[tokio::test]
async fn a_model_with_no_artifact_takes_live_changes_but_no_reload() {
    let loader = MockLoader::new();
    let mut model = loader.model("mock-a");
    model.source = None;
    let config = started(&[]);
    let server = Server::from_active(model)
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT))
        .with_config(Arc::new(ConfigState::new(config, Arc::new(RealFiles))));
    let app = server.app();
    let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "prompt": false } }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{refused}");
    assert_eq!(refused["error"]["code"], "reload_unavailable");
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 95 } }))).await;
    assert_eq!(status, StatusCode::OK);
}
