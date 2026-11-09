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
use ignis_server::config::field::{scope, Applicability, Validator};
use ignis_server::config::file::{Format, RealFiles};
use ignis_server::config::schema::all_fields;
use ignis_core::compute::ModelFamily;
use ignis_server::config::{resolve_with, Config, ConfigOutcome};
use ignis_server::config_http::ConfigState;
use ignis_server::model_switch::Switcher;
use ignis_server::template::{SimpleTemplateProvider, TemplateProvider};
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
    started_with(argv, &[])
}

/// [`started`] with `env` as the whole environment.
fn started_with(argv: &[&str], env: &'static [(&'static str, &'static str)]) -> Config {
    let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
    let env = move |key: &str| env.iter().find(|(name, _)| *name == key).map(|(_, value)| value.to_string());
    match resolve_with(&argv, env, &RealFiles).expect("the start options resolve") {
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
    let (status, _, body) = send_with_headers(app, method, uri, body).await;
    (status, body)
}

async fn send_with_headers(
    app: &axum::Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, headers, body)
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
    assert_eq!(document["server"]["request_timeout"], 2400);
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
    assert_eq!(applied["persisted"], true, "{applied}");
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
    assert_eq!(applied["persisted"], false);
    assert!(applied["reason"].as_str().unwrap().contains("once the reloaded model serves"), "{applied}");
    assert_eq!(applied["switching"], json!({ "from": "mock-a", "to": "mock-a" }));

    // Mid-reload: nothing of the patch is in force, and the config route
    // still answers, showing what runs.
    assert_eq!(server.live().request_timeout, Duration::from_secs(30), "never mixed: the live field waits for the reload");
    let (status, headers, document) = send_with_headers(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(status, StatusCode::OK, "GET /v1/config stays open during a switch: {document}");
    assert_eq!(document["server"]["request_timeout"], 30);
    assert_eq!(headers["ignis-model-status"], "switching", "the reload under way is said");
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 5 } }))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "a second change waits for the reload");
    assert_eq!(read_yaml(&path)["server"]["request_timeout"], 30, "not written before the reload took");

    release.send(()).expect("the held load is waiting");
    until_serving(&app).await;
    let (_, headers, _) = send_with_headers(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(headers["ignis-model-status"], "serving");
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
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap(), "--profile", "none"]));
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
        (json!({ "server": { "request_timeout": 3601 } }), "server.request_timeout"),
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
    assert_eq!(server.active().model_id(), "mock-a");
    assert_eq!(server.live().request_timeout, Duration::from_secs(30));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["model"]["max_context"], ignis_server::config::DEFAULT_MAX_CONTEXT);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before, "a failed reload writes nothing");
}

/// GitHub #329: once the serve stops, a patch that needs a reload is
/// refused `503 server_stopping`, not accepted with a `202` for a reload the
/// process would end under: nothing is prepared, applied or written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn once_the_serve_stops_a_patch_that_needs_a_reload_is_refused() {
    let path = config_file("stopping", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let before = std::fs::read_to_string(&path).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server.clone().serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    stop.send(()).unwrap();
    let served = tokio::time::timeout(PATIENT, serving).await.expect("the serve returned").unwrap().expect("serve");
    assert_eq!(served, ignis_server::Served::Drained);

    let body = json!({ "server": { "request_timeout": 95 }, "reuse": { "kv_host_pool_bytes": "3G" } });
    let (status, headers, refused) = send_with_headers(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("server_stopping")), "{refused}");
    assert_eq!(headers.get("retry-after").map(|value| value.to_str().unwrap()), Some("5"));
    assert!(loader.prepared_options().is_empty(), "no reload began");
    assert_eq!(server.live().request_timeout, Duration::from_secs(30), "nothing of the patch applied");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before, "nor written");
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
    assert_eq!(applied["persisted"], false);
    assert!(applied["reason"].as_str().unwrap().contains("no config file"), "{applied}");
    assert_eq!(server.live().request_timeout, Duration::from_secs(94));
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "prompt": false } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    assert_eq!(applied["persisted"], false);
    assert!(applied["reason"].as_str().unwrap().contains("no config file"), "{applied}");
    until_serving(&app).await;
}

/// Spec config-v2/02 AC 4: the known models a patch names are added to the
/// switcher's, live, the models the operator listed kept.
#[tokio::test]
async fn a_known_models_patch_reaches_the_switcher_and_keeps_the_listed_ones() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--switch-known-models", "mock-b=mock-b.ninfer"]));
    let app = server.app();
    let body = json!({ "switch": { "known_models": { "mock-c": "mock-c.ninfer" } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    let known = server.switcher.as_ref().unwrap().known_models();
    assert_eq!(known.keys().map(String::as_str).collect::<Vec<_>>(), ["mock-a", "mock-b", "mock-c"], "the start model, the listed one, the added one");
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["switch"]["known_models"], json!({ "mock-b": "mock-b.ninfer", "mock-c": "mock-c.ninfer" }));
}

/// Spec model-download/02 AC 14: a catalog entry on disk is switchable, and
/// stays so across a live change — but it is derived, never the operator's
/// list: a `known_models` patch writes the listed models and the patch's to
/// the config file, and no catalog id.
#[tokio::test]
async fn a_known_models_patch_never_writes_a_catalog_model_to_the_config_file() {
    let path = config_file("catalog-known", "switch:\n  known_models:\n    mock-b: mock-b.ninfer\n");
    let dir = path.parent().unwrap().join("models");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("mock-cat.ninfer"), b"on disk").unwrap();
    let zeros = "0".repeat(64);
    let catalog = ignis_server::download::catalog::parse_operator(
        &format!("models:\n  - id: mock-cat\n    repo: acme/cat\n    revision: v1\n    artifact: mock-cat.ninfer\n    card_class: 32\n    files:\n      - {{ name: mock-cat.ninfer.graft.json, bytes: 1, sha256: \"{zeros}\" }}\n      - {{ name: mock-cat.ninfer, bytes: 7, sha256: \"{zeros}\" }}\n"),
        Format::Yaml,
        "test",
    )
    .unwrap();
    let loader = MockLoader::new();
    let config = started(&["--config", path.to_str().unwrap()]);
    let catalog = ignis_server::model_switch::CatalogModels { catalog: Arc::new(catalog), dir };
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT).with_catalog(catalog, true))
        .with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    let app = server.app();

    let body = json!({ "switch": { "known_models": { "mock-c": "mock-c.ninfer" } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(read_yaml(&path)["switch"]["known_models"], json!({ "mock-b": "mock-b.ninfer", "mock-c": "mock-c.ninfer" }));
    assert!(!std::fs::read_to_string(&path).unwrap().contains("mock-cat"));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert!(!document.to_string().contains("mock-cat"), "{document}");
    assert!(!server.switcher.as_ref().unwrap().known_models().contains_key("mock-cat"));

    assert_eq!(ignis_server::model_switch::implicit_switch(&server, Some("mock-cat")).await, Ok(()));
    assert_eq!(loader.loads(), ["mock-cat"], "still switchable after the live change");
}

/// A live change to the thinking defaults is judged by the loaded template
/// as a start's would be (spec server/08): a default it cannot honour, or a
/// budget with no forced close, is refused and nothing applied.
#[tokio::test]
async fn a_thinking_default_the_template_cannot_honour_is_refused_live() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&[]));
    server.update_active(|model| model.template = Arc::new(NoThinkingControl));
    let app = server.app();
    for (body, says) in [
        (json!({ "model": { "enable_thinking": false } }), "cannot disable thinking"),
        (json!({ "model": { "reasoning_effort": "low" } }), "does not support"),
        (json!({ "model": { "thinking_budget": 2048 } }), "no tokenizer"),
    ] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(refused["error"]["message"].as_str().unwrap().contains(says), "{body}: {refused}");
    }
    assert!(server.live().default_enable_thinking);
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "model": { "thinking_budget": "off" } }))).await;
    assert_eq!(status, StatusCode::OK, "no budget needs no close");
    assert_eq!(server.live().default_thinking_budget, None);
}

/// The image model thinks nothing (spec image-model/05): a live change to
/// the thinking defaults is not judged by its template — the same changes
/// the text model's refuses above apply — and the next text model's load
/// judges them.
#[tokio::test]
async fn with_the_image_model_loaded_the_thinking_defaults_are_not_judged() {
    let loader = MockLoader::new();
    let config = started(&[]);
    let server = Server::from_active(loader.model("ignis-image-2.1-turbo"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT))
        .with_config(Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles))));
    server.apply_config(&config);
    server.update_active(|model| model.template = Arc::new(NoThinkingControl));
    let app = server.app();
    for body in [
        json!({ "model": { "enable_thinking": false } }),
        json!({ "model": { "reasoning_effort": "low" } }),
        json!({ "model": { "thinking_budget": 2048 } }),
    ] {
        let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!(status, StatusCode::OK, "{body}: {applied}");
    }
    assert!(!server.live().default_enable_thinking);
    assert_eq!(server.live().default_thinking_budget, Some(2048));
    assert!(loader.loads().is_empty(), "a live change, no reload");
}

/// The placeholder's template, except that it can neither disable thinking
/// nor take an effort: what a real template's capabilities can refuse.
struct NoThinkingControl;

impl TemplateProvider for NoThinkingControl {
    fn apply_chat_template(
        &self,
        messages: &[ignis_server::template::ChatMessage],
        options: &ignis_server::thinking::ThinkingOptions,
        tools: &[Value],
    ) -> Result<ignis_server::template::RenderedPrompt, ignis_server::template::TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[ignis_core::TokenId]) -> String {
        SimpleTemplateProvider.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ignis_server::thinking::ThinkingCapabilities {
        ignis_server::thinking::ThinkingCapabilities { can_disable: false, supported_efforts: Default::default() }
    }

    fn token_decoder(&self) -> Box<dyn ignis_server::decoder::TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }
}

/// A field only one family takes, patched while that family runs, is
/// written under that family's own section of the file, never the general
/// one: a later start of the other model from the same file never sees it,
/// rather than being refused for a value that was never meant for it.
#[tokio::test]
async fn a_family_only_field_is_written_under_its_familys_section() {
    let path = config_file("family-only", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let flash = started(&["--config", path.to_str().unwrap()]).for_family(ModelFamily::FlashNext).unwrap();
    let server = server_on(&loader, flash);
    server.update_active(|model| model.family = ModelFamily::FlashNext);
    let app = server.app();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "spec": { "draft_rows": 6 } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "a reload field: {applied}");
    until_serving(&app).await;

    let written = read_yaml(&path);
    assert_eq!(written["spec"]["qwen38flashnext"]["draft_rows"], 6, "{written:?}");
    assert!(written["spec"].get("draft_rows").is_none(), "never the general section: {written:?}");
    let restarted = started(&["--config", path.to_str().unwrap()]);
    let qwen = restarted.for_family(ModelFamily::Qwen38_27b).expect("a 27B start from the same file is unaffected");
    assert_eq!(qwen.draft_rows, None);
    assert_eq!(restarted.for_family(ModelFamily::FlashNext).unwrap().draft_rows, Some(6), "Flash-Next still gets it");
}

/// A request pins the live knobs with the model: a live change landing
/// mid-request reaches the next request, never half of this one.
#[tokio::test]
async fn a_pinned_request_keeps_the_live_knobs_it_began_with() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&[]));
    let pinned = server.pinned();
    let app = server.app();
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 61 } }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(pinned.live().request_timeout, Duration::from_secs(2400), "the request in flight keeps its own");
    assert_eq!(server.live().request_timeout, Duration::from_secs(61), "the next one gets the change");
}

/// The loop guard's knobs are live: a patch reaches the next request's
/// guard at once, no model touched — the raw prototype back, for one. A
/// configured server starts in `rewind` with no holdback.
#[tokio::test]
async fn a_loop_guard_patch_applies_live() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&[]));
    let app = server.app();
    assert!(server.live().loop_guard.normalize);
    assert_eq!(
        (server.live().loop_guard.mode, server.live().loop_guard.holdback_chars),
        (ignis_server::loop_guard::LoopGuardMode::Rewind, 0),
        "rewind by default"
    );
    let (_, before) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!((&before["model"]["loop_guard"], &before["model"]["loop_guard_holdback"]), (&json!("rewind"), &json!(0)));
    let body = json!({ "model": {
        "loop_guard": "log", "loop_guard_normalize": "off", "loop_guard_window": 3000, "loop_guard_ratio": 120,
        "loop_guard_raw_gate": 0, "loop_guard_clean": 350, "loop_guard_margin": 1000, "loop_guard_holdback": 3000,
        "loop_guard_check_every": 500, "loop_guard_snap": 0, "loop_guard_min_continuation": 1024, "loop_guard_submit_wait": 10,
        "loop_guard_nudge": "Next step.", "loop_guard_nudge_again": "Wrap up.", "loop_guard_persist": 6, "loop_guard_min_kept": 0,
    } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["applied"], "live");
    assert_eq!(applied["config"]["model"]["loop_guard_normalize"], false);
    assert_eq!((&applied["config"]["model"]["loop_guard_raw_gate"], &applied["config"]["model"]["loop_guard_clean"]), (&json!(0), &json!(350)));
    assert_eq!(applied["config"]["model"]["loop_guard_holdback"], 3000);
    assert_eq!(applied["config"]["model"]["loop_guard_nudge"], "Next step.");
    let guard = server.live().loop_guard.clone();
    assert_eq!((guard.nudge.as_str(), guard.nudge_again.as_str()), ("Next step.", "Wrap up."), "the nudges");
    assert_eq!((guard.persist_checks, guard.min_kept_chars), (6, 0), "the persistence and the room to keep");
    assert_eq!(
        (guard.mode, guard.normalize, guard.window_chars, guard.ratio_permille),
        (ignis_server::loop_guard::LoopGuardMode::Log, false, 3000, 120)
    );
    assert_eq!((guard.raw_gate_permille, guard.clean_permille, guard.margin_chars), (0, 350, 1000), "the gate off, the cut's knobs");
    assert_eq!(guard.holdback_chars, 3000, "the holdback");
    assert_eq!(
        (guard.check_every_chars, guard.snap_chars, guard.min_continuation_tokens, guard.submit_wait_secs),
        (500, 0, 1024, 10),
        "the cadence, the snap, the room and the wait"
    );
    assert!(loader.prepared_options().is_empty(), "no model was prepared, so none was switched");
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

// ── GET /v1/config/fields (spec playground/05) ───────────────────────────────

fn declaration<'a>(fields: &'a Value, file_key: &str) -> &'a Value {
    fields["data"].as_array().expect("an array").iter().find(|field| field["file_key"] == file_key).unwrap_or_else(|| panic!("{file_key} is not listed"))
}

/// Spec playground/05 AC 1: every visible field once, in declaration order,
/// with the documented keys; a field not marked `visible` is absent.
#[tokio::test]
async fn the_fields_route_lists_every_visible_field_once_in_declaration_order() {
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&[])).app();
    let (status, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "{fields}");
    let listed: Vec<&str> = fields["data"].as_array().unwrap().iter().map(|field| field["file_key"].as_str().unwrap()).collect();
    let visible: Vec<String> = all_fields().filter(|meta| meta.visible).map(|meta| meta.file_key()).collect();
    assert_eq!(listed, visible, "the visible fields, in the order they are declared");
    assert!(all_fields().any(|meta| !meta.visible), "the whitelist hides something");
    for hidden in ["server.api_key", "download.token"] {
        assert!(!listed.contains(&hidden), "{hidden} is hidden from GET /v1/config, so from here too");
    }
    let mut want = vec![
        "group", "name", "file_key", "flag", "env", "kind", "nullable", "default", "rule", "rule_text", "families", "scoped", "patchable",
        "reload_required", "description",
    ];
    want.sort_unstable();
    for field in fields["data"].as_array().unwrap() {
        let mut have: Vec<&str> = field.as_object().unwrap().keys().map(String::as_str).collect();
        have.sort_unstable();
        assert_eq!(have, want, "{}", field["file_key"]);
    }
    let chunk = declaration(&fields, "model.prefill_chunk");
    assert_eq!((chunk["group"].clone(), chunk["name"].clone()), (json!("model"), json!("prefill_chunk")));
    assert_eq!((chunk["flag"].clone(), chunk["env"].clone()), (json!("--model-prefill-chunk"), json!("IGNIS_MODEL_PREFILL_CHUNK")));
    assert_eq!(chunk["default"], 1024);
    assert!(!chunk["description"].as_str().unwrap().contains('\n'), "one line");
}

/// AC 1: `rule` is structured for each validator, `rule_text` is the sentence
/// `help --fields` prints, `families` and `nullable` match the declaration.
#[tokio::test]
async fn the_fields_rule_families_and_nullable_follow_the_declaration() {
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&[])).app();
    let (_, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(declaration(&fields, "model.prefill_chunk")["rule"], json!({ "multiple_of": 128 }));
    assert_eq!(declaration(&fields, "model.prefill_chunk")["rule_text"], "nonzero multiple of 128");
    // Each value once, in the form the config writes it: the spellings a
    // parser also takes (`hq`, `hq_e8_2b`) are not values.
    assert_eq!(declaration(&fields, "model.kv_format")["rule"], json!({ "one_of": ["bf16", "hq-e8-2b"] }));
    assert_eq!(declaration(&fields, "model.decode_share")["rule"], json!({ "min": 0, "max": 99 }));
    assert_eq!(declaration(&fields, "model.decode_share")["rule_text"], "0..=99");
    assert_eq!(declaration(&fields, "vram.budget_bytes")["rule"], json!({ "min": 1, "max": null }), "an open end is null");
    assert_eq!(declaration(&fields, "vram.budget_bytes")["rule_text"], ">= 1");
    // Every field agrees with its own declaration, whatever the variant.
    for meta in all_fields().filter(|meta| meta.visible) {
        let field = declaration(&fields, &meta.file_key());
        assert_eq!(field["rule"].is_null(), meta.validator == Validator::None, "{}", meta.file_key());
        assert_eq!(field["rule_text"], meta.validator.describe(), "{}", meta.file_key());
        let families = match meta.applies {
            Applicability::AllFamilies => Value::Null,
            Applicability::Only(only) => only.iter().map(|family| json!(scope(*family))).collect::<Vec<_>>().into(),
        };
        assert_eq!(field["families"], families, "{}", meta.file_key());
        assert_eq!(field["nullable"], meta.nullable, "{}", meta.file_key());
        assert_eq!(field["scoped"], meta.scoped);
        assert_eq!((field["patchable"].clone(), field["reload_required"].clone()), (json!(meta.patchable), json!(meta.reload_required)));
    }
    let only: Vec<_> = all_fields().filter(|meta| meta.visible && meta.home_family() == Some(ModelFamily::FlashNext)).collect();
    assert!(!only.is_empty(), "some visible field is Flash-Next's alone");
    assert_eq!(declaration(&fields, &only[0].file_key())["families"], json!(["qwen38flashnext"]));
    assert_eq!(declaration(&fields, "server.request_timeout")["families"], Value::Null, "every family takes it");
    assert_eq!(declaration(&fields, "model.decode_share")["nullable"], true, "Opt<..> may be unset");
    assert_eq!(declaration(&fields, "model.prefill_chunk")["nullable"], false);
}

/// AC 2: static — `200` during a switch, and on a server with no running
/// configuration; `401` without the API key when one is set.
#[tokio::test]
async fn the_fields_route_answers_during_a_switch_without_a_config_and_behind_the_key() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&[]));
    let app = server.app();
    let release = loader.hold_next_load();
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "kv_host_pool_bytes": "3G" } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let (_, headers, _) = send_with_headers(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(headers["ignis-model-status"], "switching");
    let (status, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "static: the model's state does not matter: {fields}");
    assert!(!fields["data"].as_array().unwrap().is_empty());
    release.send(()).expect("the held load is waiting");
    until_serving(&app).await;

    let bare = Server::from_active(loader.model("mock-a")).app();
    let (status, bare_fields) = send(&bare, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "no running configuration is no reason to refuse a declaration");
    assert_eq!(bare_fields, fields);
    assert_eq!(fields["object"], "list");

    let keyed = server_on(&loader, started(&[])).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let (status, _) = send(&keyed, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = send(&keyed, Method::OPTIONS, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "the preflight passes, like the other /v1 routes");
}

/// The preflight of `PATCH /v1/config` from another origin names `PATCH` among
/// the methods it allows: without it a browser on another origin never sends
/// the change.
#[tokio::test]
async fn the_preflight_allows_the_patch_a_config_change_is() {
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&[])).app();
    let (status, headers, _) = send_with_headers(&app, Method::OPTIONS, "/v1/config", None).await;
    assert_eq!(status, StatusCode::OK);
    let allowed = headers["access-control-allow-methods"].to_str().unwrap();
    assert!(allowed.split(',').map(str::trim).any(|method| method == "PATCH"), "{allowed}");
}

/// `help --fields --format json` and the route are one declaration: every
/// key the route serves is the command line's, except `rule` and `families`,
/// which the command line keeps as the sentence and names it always printed
/// (the route's `rule_text` is the former).
#[test]
fn the_command_line_json_and_the_route_share_each_declaration() {
    let cli: Vec<Value> = serde_json::from_str(&ignis_server::config::cli::fields_json()).unwrap();
    let route = ignis_server::config_http::visible_declarations();
    assert_eq!(cli.iter().filter(|field| field["visible"] == true).count(), route.len());
    for served in &route {
        let printed = cli.iter().find(|field| field["file_key"] == served["file_key"]).expect("the CLI lists it too");
        for (key, value) in served.as_object().unwrap() {
            if key == "rule" || key == "families" {
                continue;
            }
            assert_eq!(&printed[key], value, "{} {key}", served["file_key"]);
        }
        assert_eq!(printed["rule"], served["rule_text"], "the CLI's `rule` is the sentence");
    }
}

/// Spec playground/05: the declarations are static, so the warm-up does not
/// hold them either — a page reads them while the first traversal runs.
#[tokio::test]
async fn the_fields_route_answers_during_the_warm_up() {
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&[])).with_warm_up().app();
    let (status, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "the gate is shut: {models}");
    let (status, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    assert_eq!(status, StatusCode::OK, "{fields}");
    assert!(!fields["data"].as_array().unwrap().is_empty());
}

/// Every `one_of` value appears once, in canonical form, whatever spellings
/// the field's parser also takes.
#[tokio::test]
async fn a_one_of_rule_lists_each_value_once_in_canonical_form() {
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&[])).app();
    let (_, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    for meta in all_fields().filter(|meta| meta.visible) {
        let Validator::OneOf(allowed) = meta.validator else { continue };
        let listed = declaration(&fields, &meta.file_key())["rule"]["one_of"].clone();
        let listed: Vec<&str> = listed.as_array().expect("a list").iter().map(|value| value.as_str().unwrap()).collect();
        let mut canonical: Vec<String> = Vec::new();
        for spelling in allowed {
            let written = (meta.canonical)(spelling).ok().and_then(|value| value.as_str().map(str::to_owned)).unwrap_or_else(|| spelling.to_string());
            if !canonical.contains(&written) {
                canonical.push(written);
            }
        }
        assert_eq!(listed, canonical, "{}", meta.file_key());
    }
}

// ── per-model overrides, removals, the rendered file (spec playground/05) ────

/// A 27B serving `argv`'s configuration fitted to its family, on a server
/// whose loader fits every load from the running configuration as the
/// production loader does, `mock-b` being a Flash-Next artifact.
fn qwen_server(loader: &Arc<MockLoader>, argv: &[&str]) -> Server {
    family_server(loader, ModelFamily::Qwen38_27b, started(argv))
}

/// `mock-a`, a model of `family`, serving `start` fitted to that family, on
/// a server whose loader fits every load from the running configuration as
/// the production loader does, `mock-b` being a model of the other family.
fn family_server(loader: &Arc<MockLoader>, family: ModelFamily, start: Config) -> Server {
    let config = start.for_family(family).expect("the family takes the start options");
    let other = if family == ModelFamily::FlashNext { ModelFamily::Qwen38_27b } else { ModelFamily::FlashNext };
    let state = Arc::new(ConfigState::new(config.clone(), Arc::new(RealFiles)));
    loader.fit_from(Arc::clone(&state), &[("mock-a", family), ("mock-b", other)]);
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
        .with_config(state);
    server.update_active(|model| model.family = family);
    server.apply_config(&config);
    server
}

/// Spec playground/05 AC 19: values that change nothing the running model
/// runs with — one scoped to the family not loaded, a group value the
/// running family overrides in its own section — are a live change though
/// they need a reload: applied at once with no model touched, written to the
/// file, and the load of the other family, fitted from the running
/// configuration as the production loader fits it, runs with them.
#[tokio::test]
async fn a_patch_that_changes_nothing_running_is_live_and_the_other_familys_load_runs_with_it() {
    let path = config_file("other-family", "reuse:\n  qwen38:\n    kv_host_pool_bytes: 4G\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let body = json!({ "reuse": { "kv_host_pool_bytes": "3G" }, "model": { "qwen38flashnext": { "max_context": 65536 } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!((applied["applied"].clone(), applied["persisted"].clone()), (json!("live"), json!(true)), "{applied}");
    assert!(loader.prepared_options().is_empty(), "no model was touched");
    let shown = &applied["config"];
    // The file's shape (#316): the group key is the shared value, and the
    // 27B's own value stays in its section; Flash-Next takes the shared one.
    assert_eq!(shown["reuse"]["kv_host_pool_bytes"], "3G", "the shared value: {shown}");
    assert_eq!(shown["reuse"]["qwen38"]["kv_host_pool_bytes"], "4G", "the 27B keeps its own: {shown}");
    assert!(shown["reuse"].get("qwen38flashnext").is_none(), "Flash-Next takes the shared value: {shown}");
    assert_eq!(shown["model"]["qwen38flashnext"]["max_context"], 65536, "{shown}");
    let written = read_yaml(&path);
    assert_eq!(written["reuse"]["kv_host_pool_bytes"], "3G", "{written:?}");
    assert_eq!(written["reuse"]["qwen38"]["kv_host_pool_bytes"], "4G", "{written:?}");
    assert_eq!(written["model"]["qwen38flashnext"]["max_context"], 65536, "{written:?}");

    let switch = json!({ "model": "mock-b", "artifact": "mock-b.ninfer" });
    let (status, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    until_serving(&app).await;
    let fitted = loader.fitted_options();
    let flash = fitted.last().expect("the Flash-Next load was fitted from the running configuration");
    assert_eq!(flash.basis.family(), Some(ModelFamily::FlashNext));
    assert_eq!((flash.host_pool_bytes, flash.max_context), (3 << 30, 65536));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!((document["reuse"]["kv_host_pool_bytes"].clone(), document["model"]["qwen38flashnext"]["max_context"].clone()), (json!("3G"), json!(65536)), "{document}");
}

/// GitHub #316: `GET /v1/config` answers in the file's shape even while the
/// config is fitted to a family: the group key is the shared value, and the
/// running family's own value is in its section — the Playground's "All
/// models" and "Own value" read them off. A patch of the group key leaves
/// the section in force.
#[tokio::test]
async fn get_config_shows_the_shared_value_and_the_running_familys_own_section() {
    let path = config_file("file-shape", "reuse:\n  kv_host_pool_bytes: 2G\n  qwen38:\n    kv_host_pool_bytes: 4G\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["reuse"]["kv_host_pool_bytes"], "2G", "the group key is the shared value: {document}");
    assert_eq!(document["reuse"]["qwen38"]["kv_host_pool_bytes"], "4G", "the running family's own section: {document}");

    let body = json!({ "reuse": { "kv_host_pool_bytes": "3G" } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["applied"], "live", "{applied}");
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["reuse"]["kv_host_pool_bytes"], "3G", "{document}");
    assert_eq!(document["reuse"]["qwen38"]["kv_host_pool_bytes"], "4G", "the section stays in force: {document}");
}

/// Spec playground/05 AC 20: a value waiting for the other family beside a
/// change the running model takes reloads, and both land — in the load, in
/// what `GET /v1/config` shows and in the file.
#[tokio::test]
async fn a_patch_mixing_the_other_familys_value_with_a_running_change_reloads_and_both_land() {
    let path = config_file("mixed", "server:\n  request_timeout: 30\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let body = json!({ "reuse": { "qwen38": { "kv_host_pool_bytes": "3G" }, "qwen38flashnext": { "kv_host_pool_bytes": "1G" } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    assert_eq!(applied["applied"], "reload");
    until_serving(&app).await;
    let fitted = loader.fitted_options();
    let reloaded = fitted.last().expect("the reload's load");
    assert_eq!((reloaded.basis.family(), reloaded.host_pool_bytes), (Some(ModelFamily::Qwen38_27b), 3 << 30));
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["reuse"]["kv_host_pool_bytes"], "2G", "the shared value, the default: {document}");
    assert_eq!(document["reuse"]["qwen38"]["kv_host_pool_bytes"], "3G", "{document}");
    assert_eq!(document["reuse"]["qwen38flashnext"]["kv_host_pool_bytes"], "1G", "{document}");
    let written = read_yaml(&path);
    assert_eq!(written["reuse"]["qwen38"]["kv_host_pool_bytes"], "3G", "{written:?}");
    assert_eq!(written["reuse"]["qwen38flashnext"]["kv_host_pool_bytes"], "1G", "{written:?}");
}

/// Spec playground/05 AC 21: `null` in a family's section removes that
/// family's own value, so the group's applies to it again — live for the
/// family not loaded, through a reload for the running one — and `null` at
/// a group key removes it, so the default applies. The file loses each key,
/// and the section and the group a removal empties go with it.
#[tokio::test]
async fn null_removes_a_familys_own_value_then_the_groups_and_the_file_loses_each() {
    let path = config_file(
        "null-removes",
        "server:\n  request_timeout: 30\nreuse:\n  kv_host_pool_bytes: 3G\n  qwen38:\n    kv_host_pool_bytes: 4G\n  qwen38flashnext:\n    kv_host_pool_bytes: 1G\n",
    );
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let pool = |config: &Value| config["reuse"]["kv_host_pool_bytes"].clone();

    let body = json!({ "reuse": { "qwen38flashnext": { "kv_host_pool_bytes": null } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].clone()), (StatusCode::OK, json!("live")), "the family not loaded: {applied}");
    let shown = &applied["config"]["reuse"];
    assert!(shown.get("qwen38flashnext").is_none(), "the group's applies to it, so it has no section: {applied}");
    assert_eq!((shown["kv_host_pool_bytes"].clone(), shown["qwen38"]["kv_host_pool_bytes"].clone()), (json!("3G"), json!("4G")), "{applied}");
    let written = read_yaml(&path);
    assert!(written["reuse"].get("qwen38flashnext").is_none(), "the emptied section goes: {written:?}");
    assert_eq!(written["reuse"]["qwen38"]["kv_host_pool_bytes"], "4G", "{written:?}");

    let body = json!({ "reuse": { "qwen38": { "kv_host_pool_bytes": null } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].clone()), (StatusCode::ACCEPTED, json!("reload")), "the running family's: {applied}");
    until_serving(&app).await;
    assert_eq!(loader.fitted_options().last().unwrap().host_pool_bytes, 3 << 30, "the reload runs with the group's");
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(pool(&document), "3G", "{document}");
    assert_eq!(read_yaml(&path)["reuse"], json!({ "kv_host_pool_bytes": "3G" }));

    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "kv_host_pool_bytes": null } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    until_serving(&app).await;
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(pool(&document), "2G", "the hardcoded default: {document}");
    let written = read_yaml(&path);
    assert!(written.get("reuse").is_none(), "the emptied group goes: {written:?}");
    assert_eq!(written["server"]["request_timeout"], 30, "the rest of the file is kept");
}

/// Spec playground/05 AC 21: `null` at a group key falls back to the
/// profile in use, as a start from the rewritten file would.
#[tokio::test]
async fn null_at_a_group_key_falls_back_to_the_profile() {
    let path = config_file("null-profile", "vram:\n  headroom_bytes: 2G\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap()]);
    let app = server.app();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "vram": { "headroom_bytes": null } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{applied}");
    until_serving(&app).await;
    let running = server.config.as_ref().unwrap().current();
    let restarted = started(&["--config", path.to_str().unwrap()]).for_family(ModelFamily::Qwen38_27b).unwrap();
    assert_eq!(running.settings().vram.headroom_bytes, 1200 << 20, "the rtx5090 profile's own for the 27B");
    assert_eq!(running.settings(), restarted.settings(), "what a start from the rewritten file resolves");
}

/// Spec playground/05 AC 21: a removal of a key no source sets is accepted
/// and changes nothing — not the configuration, not a byte of the file —
/// and a body of only removals is a patch.
#[tokio::test]
async fn a_removal_no_source_sets_is_a_no_op_200() {
    let text = "# the operator's own notes\nserver:\n  request_timeout: 30\n";
    let path = config_file("null-no-op", text);
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let (_, before) = send(&app, Method::GET, "/v1/config", None).await;
    let body = json!({ "model": { "qwen38": { "max_context": null } }, "reuse": { "kv_host_pool_bytes": null } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].clone(), applied["persisted"].clone()), (StatusCode::OK, json!("live"), json!(true)), "{applied}");
    assert_eq!(applied["config"], before);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "the file untouched, its comment kept");
    assert!(loader.prepared_options().is_empty());
}

/// Spec playground/05 AC 21: a removal leaves the value a start from the
/// rewritten file resolves, so a flag that sets the field is in force again
/// — and named, as it outranks the file at the next start too.
#[tokio::test]
async fn a_removed_field_a_flag_sets_runs_the_flags_value_and_is_named_as_shadowed() {
    let path = config_file("null-flag", "server:\n  request_timeout: 60\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--server-request-timeout", "40"]);
    let app = server.app();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": 90 } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(server.live().request_timeout, Duration::from_secs(90), "a patch outranks the command line while the server runs");
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "server": { "request_timeout": null } }))).await;
    assert_eq!(status, StatusCode::OK, "{applied}");
    assert_eq!(applied["shadowed_at_restart"], json!(["server.request_timeout (set by --server-request-timeout)"]), "{applied}");
    assert_eq!(server.live().request_timeout, Duration::from_secs(40), "the flag's value, as at a start");
    assert!(read_yaml(&path).get("server").is_none(), "the file loses the key");
}

/// Spec playground/05 AC 22: `?format=yaml` answers, as `application/yaml`
/// with the model's state, the YAML `config print` writes for the same
/// configuration minus the hidden fields — the JSON document's own content;
/// `json` or no `format` is the JSON document, and any other is `400
/// config_invalid`.
#[tokio::test]
async fn get_config_as_yaml_is_what_config_print_writes_minus_the_hidden_fields() {
    let path = config_file("yaml", "server:\n  request_timeout: 45\nreuse:\n  qwen38flashnext:\n    kv_host_pool_bytes: 1G\n");
    let path = path.to_str().unwrap();
    let loader = MockLoader::new();
    let app = server_on(&loader, started(&["--config", path, "--server-api-key", "sk-secret"])).app();
    let request = Request::builder().uri("/v1/config?format=yaml").header(header::AUTHORIZATION, "Bearer sk-secret").body(Body::empty()).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/yaml");
    assert_eq!(response.headers()["ignis-model-status"], "serving");
    let text = String::from_utf8(response.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    assert!(!text.contains("sk-secret") && !text.contains("api_key"), "{text}");

    let argv: Vec<String> = ["config", "print", "--file", path, "--server-api-key", "sk-secret"].iter().map(|s| s.to_string()).collect();
    let ConfigOutcome::Print(printed) = resolve_with(&argv, |_| None, &RealFiles).unwrap() else { panic!("config print prints") };
    let mut printed = Format::Yaml.read(&printed).unwrap();
    ignis_server::config_http::remove_invisible(&mut printed, all_fields());
    assert_eq!(text, Format::Yaml.write(&printed), "config print's text, minus the hidden fields");

    let get = |uri: &'static str| {
        let request = Request::builder().uri(uri).header(header::AUTHORIZATION, "Bearer sk-secret").body(Body::empty()).unwrap();
        let app = app.clone();
        async move {
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        }
    };
    let (status, json) = get("/v1/config?format=json").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(Format::Yaml.read(&text).unwrap(), json, "the same document, as YAML");
    assert_eq!(get("/v1/config").await.1, json);
    let (status, refused) = get("/v1/config?format=toml").await;
    assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::BAD_REQUEST, json!("config_invalid")), "{refused}");
    let message = refused["error"]["message"].as_str().unwrap();
    assert!(message.contains("toml") && message.contains("json") && message.contains("yaml"), "{message}");
}

// ── where each value comes from (spec playground/05 follow-up 2) ─────────────

/// The rows `GET /v1/config/origins` gives `key`, as `(scope, source,
/// scoped, spelling)`, in the order served.
fn origins_of(origins: &Value, key: &str) -> Vec<(Value, String, bool, Option<String>)> {
    let rows = origins["data"].as_array().expect("a list of rows");
    let rows = rows.iter().filter(|row| row["key"] == key);
    rows.map(|row| {
        let spelling = row.get("spelling").map(|spelling| spelling.as_str().expect("a spelling is text").to_owned());
        (row["scope"].clone(), row["source"].as_str().unwrap().to_owned(), row["scoped"].as_bool().unwrap(), spelling)
    })
    .collect()
}

/// GitHub #480: under `--profile auto` an RTX 3090 (24 GiB, compute
/// capability 8.6, under a build for it) and a 24 GB Blackwell card (12.0)
/// start on `rtx-24gb`: `GET /v1/config` shows its 27B values, the measured
/// 1,048,576-token context among them, and `GET /v1/config/origins` names
/// the profile as their source. The prompt-attention band and the prefill
/// route are the card's (GitHub #515): the integer band and the dense prefill
/// on the 3090, the card as their source; FP8 and the fused prefill on
/// Blackwell, their defaults.
#[tokio::test]
async fn under_auto_a_24gb_card_runs_rtx_24gb_and_the_config_says_so() {
    use ignis_server::config::arch::{BuiltArchitectures, ComputeCapability};
    use ignis_server::config::file::NoFiles;
    use ignis_server::config::profile::{Card, CardReading};
    use ignis_server::config::resolve_on;

    let rtx3090 = Card::Read(CardReading {
        total: 24 << 30,
        name: "NVIDIA GeForce RTX 3090".to_owned(),
        capability: ComputeCapability::new(8, 6),
        built_for: BuiltArchitectures::parse("86;120a").unwrap(),
    });
    for (card, band, route, route_source) in [(rtx3090, "int8", "int8", "card"), (Card::blackwell(24 << 30), "fp8", "fp16", "default")] {
        let start = match resolve_on(&[], |_| None, &NoFiles, &|| card.clone()).expect("the start resolves") {
            ConfigOutcome::Config(config) => config,
            other => panic!("{other:?}"),
        };
        let loader = MockLoader::new();
        let app = family_server(&loader, ModelFamily::Qwen38_27b, start).app();
        let (status, document) = send(&app, Method::GET, "/v1/config", None).await;
        assert_eq!(status, StatusCode::OK, "{document}");
        let qwen = (&document["model"]["qwen38"], &document["spec"]["qwen38"], &document["vram"]["qwen38"]);
        assert_eq!((&qwen.0["max_context"], &qwen.0["prefill_chunk"]), (&json!(1048576), &json!(2048)), "{band}: {document}");
        // GitHub #489: the solo width as configured, which the load resolves.
        assert_eq!(document["model"]["prefill_chunk_solo"], "auto", "{band}: {document}");
        assert_eq!(qwen.0["prefill_chunk_solo"], 2048, "the profile pins the 27B's to its chunk: {document}");
        assert_eq!((&qwen.1["backend"], &qwen.1["draft_tokens"]), (&json!("dflash2"), &json!(7)), "{band}: {document}");
        assert_eq!(qwen.2["headroom_bytes"], "1200M", "{band}: {document}");
        // GitHub #557: vision is both models' and on for the 27B alone; its
        // envelope and residency are both models'.
        let vision = &document["vision"];
        assert_eq!((&vision["qwen38"]["enabled"], &vision["residency"], &vision["max_tokens"]), (&json!(true), &json!("host"), &json!(32768)), "{band}: {document}");
        assert_eq!(vision["enabled"], false, "Flash-Next's stays off: {document}");
        assert_eq!(qwen.0["prefill_attention"], band, "{document}");
        // Spec #503 (GitHub #505): the prefill route is reported as the load resolves it.
        assert_eq!(qwen.0["prefill_linear"], route, "{document}");
        assert!(document["model"]["qwen38flashnext"].get("prefill_linear").is_none(), "the 27B's alone: {document}");

        let (status, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
        assert_eq!(status, StatusCode::OK, "{origins}");
        let profile = |key: &str| origins_of(&origins, key).into_iter().find(|(scope, ..)| scope == "qwen38").unwrap();
        for key in ["model.max_context", "model.prefill_chunk", "spec.backend", "spec.draft_tokens", "vram.headroom_bytes", "vision.enabled"] {
            assert_eq!(profile(key), (json!("qwen38"), "profile".to_owned(), true, Some("rtx-24gb".to_owned())), "{band}: {key}");
        }
        for key in ["model.prefill_linear", "model.prefill_attention"] {
            let origin = origins_of(&origins, key);
            let expected = if key.ends_with("linear") || route_source == "card" { route_source } else { "default" };
            assert!(!origin.is_empty() && origin.iter().all(|(_, source, ..)| source == expected), "{band}: {key} {origin:?}");
        }
        assert_eq!(origins_of(&origins, "vision.residency")[0], (Value::Null, "profile".to_owned(), false, Some("rtx-24gb".to_owned())), "{band}");
    }
}

/// Spec playground/05 AC 28: every visible field once per scope — `null`,
/// then each family that takes it — with the source of the value that
/// resolves there, whether it was set for that family, and where, spelled
/// as the startup source log spells it: a default (no spelling), a profile's
/// family value and its group value (the profile's name), a file's group
/// and family values (the key and the file), a flag and a family flag, an
/// env var, and a live change (no spelling). An own value equal to the
/// shared one is still the family's own.
#[tokio::test]
async fn the_origins_say_where_each_value_comes_from() {
    let path = config_file(
        "origins",
        "reuse:\n  kv_host_pool_bytes: 3G\n  prompt: false\n  qwen38:\n    kv_host_pool_bytes: 4G\n    prompt: false\n",
    );
    let file = path.display().to_string();
    let argv = ["--config", path.to_str().unwrap(), "--model-prefill-chunk", "2048", "--qwen38flashnext-model-max-context", "65536"];
    let loader = MockLoader::new();
    let start = started_with(&argv, &[("IGNIS_SWITCH_DRAIN_TIMEOUT", "7")]);
    let server = family_server(&loader, ModelFamily::Qwen38_27b, start);
    let app = server.app();
    let (status, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    assert_eq!(status, StatusCode::OK, "{origins}");
    assert_eq!(origins["object"], "list");

    let (shared, qwen, flash) = (Value::Null, json!("qwen38"), json!("qwen38flashnext"));
    let some = |text: &str| Some(text.to_owned());
    let row = |scope: &Value, source: &str, scoped: bool, spelling: Option<String>| (scope.clone(), source.to_owned(), scoped, spelling);
    assert_eq!(
        origins_of(&origins, "server.request_timeout"),
        [row(&shared, "default", false, None), row(&qwen, "default", false, None), row(&flash, "default", false, None)]
    );
    assert_eq!(
        origins_of(&origins, "vram.headroom_bytes"),
        [row(&shared, "profile", false, some("rtx5090")), row(&qwen, "profile", true, some("rtx5090")), row(&flash, "profile", false, some("rtx5090"))],
        "the rtx5090 profile's group value, and its own for the 27B"
    );
    assert_eq!(
        origins_of(&origins, "reuse.kv_host_pool_bytes"),
        [
            row(&shared, "file", false, some(&format!("reuse.kv_host_pool_bytes ({file})"))),
            row(&qwen, "file", true, some(&format!("reuse.qwen38.kv_host_pool_bytes ({file})"))),
            row(&flash, "file", false, some(&format!("reuse.kv_host_pool_bytes ({file})"))),
        ]
    );
    assert_eq!(
        origins_of(&origins, "reuse.prompt")[1],
        row(&qwen, "file", true, some(&format!("reuse.qwen38.prompt ({file})"))),
        "an own value equal to the shared one is the family's own"
    );
    assert_eq!(
        origins_of(&origins, "model.prefill_chunk"),
        [row(&shared, "flag", false, some("--model-prefill-chunk")), row(&qwen, "flag", false, some("--model-prefill-chunk")), row(&flash, "flag", false, some("--model-prefill-chunk"))]
    );
    assert_eq!(
        origins_of(&origins, "model.max_context"),
        [row(&shared, "default", false, None), row(&qwen, "profile", true, some("rtx5090")), row(&flash, "flag", true, some("--qwen38flashnext-model-max-context"))]
    );
    assert_eq!(origins_of(&origins, "switch.drain_timeout")[0], row(&shared, "env", false, some("IGNIS_SWITCH_DRAIN_TIMEOUT")));
    assert_eq!(
        origins_of(&origins, "spec.draft_rows"),
        [row(&shared, "default", false, None), row(&flash, "default", false, None)],
        "a field only Flash-Next takes has no 27B row"
    );
    // GitHub #336: the decode lanes are both models', and the profile's 3 is
    // Flash-Next's own; the 27B's is its default.
    assert_eq!(
        origins_of(&origins, "spec.decode_lanes"),
        [row(&shared, "default", false, None), row(&qwen, "default", false, None), row(&flash, "profile", true, some("rtx5090"))]
    );

    // Every visible field, once per scope, and nothing hidden.
    let keys: Vec<&str> = origins["data"].as_array().unwrap().iter().map(|row| row["key"].as_str().unwrap()).collect();
    for meta in all_fields() {
        let rows = keys.iter().filter(|key| **key == meta.file_key()).count();
        let families = [ModelFamily::Qwen38_27b, ModelFamily::FlashNext].into_iter().filter(|family| meta.applies.to(*family)).count();
        assert_eq!(rows, if meta.visible { 1 + families } else { 0 }, "{}", meta.file_key());
    }
    for row in origins["data"].as_array().unwrap() {
        let mut have: Vec<&str> = row.as_object().unwrap().keys().map(String::as_str).collect();
        have.sort_unstable();
        let want: &[&str] = if row.get("spelling").is_some() { &["key", "scope", "scoped", "source", "spelling"] } else { &["key", "scope", "scoped", "source"] };
        assert_eq!(have, want, "{row}");
    }

    // A live change, the group's and the family not loaded's.
    let body = json!({ "server": { "request_timeout": 60 }, "reuse": { "qwen38flashnext": { "kv_host_pool_bytes": "1G" } } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].clone()), (StatusCode::OK, json!("live")), "{applied}");
    let (_, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    assert_eq!(
        origins_of(&origins, "server.request_timeout"),
        [row(&shared, "live", false, None), row(&qwen, "live", false, None), row(&flash, "live", false, None)]
    );
    let pool = origins_of(&origins, "reuse.kv_host_pool_bytes");
    assert_eq!(pool[1], row(&qwen, "file", true, some(&format!("reuse.qwen38.kv_host_pool_bytes ({file})"))));
    assert_eq!(pool[2], row(&flash, "live", true, None));
}

/// `PATCH /v1/config` with `body`, applied live or through a reload, and
/// returned once the model serves again.
async fn patched(app: &axum::Router, body: Value) -> Value {
    let (status, applied) = send(app, Method::PATCH, "/v1/config", Some(body.clone())).await;
    assert!(status == StatusCode::OK || status == StatusCode::ACCEPTED, "{body}: {status} {applied}");
    until_serving(app).await;
    applied
}

/// Spec playground/05 AC 29, a value then its removal: a field only
/// Flash-Next takes, set at its group key, lives in Flash-Next's section in
/// the running configuration as in the file, so a removal at either key
/// takes it away from both — and what runs is what a start from the
/// rewritten file resolves.
#[tokio::test]
async fn a_family_only_value_set_then_removed_leaves_what_a_start_from_the_file_runs() {
    let path = config_file("family-only-set-remove", "server:\n  request_timeout: 30\n");
    let argv = ["--config", path.to_str().unwrap(), "--profile", "none"];
    let loader = MockLoader::new();
    let server = family_server(&loader, ModelFamily::FlashNext, started(&argv));
    let app = server.app();
    let running = || server.config.as_ref().unwrap().current();
    let restarted = || started(&argv).for_family(ModelFamily::FlashNext).unwrap();
    for removal in [json!({ "spec": { "qwen38flashnext": { "draft_rows": null } } }), json!({ "spec": { "draft_rows": null } })] {
        patched(&app, json!({ "spec": { "draft_rows": 4 } })).await;
        assert_eq!(running().draft_rows, Some(4));
        assert_eq!(read_yaml(&path)["spec"]["qwen38flashnext"]["draft_rows"], 4);
        assert_eq!(running().settings(), restarted().settings(), "set: what the file starts");
        patched(&app, removal.clone()).await;
        assert!(read_yaml(&path).get("spec").is_none(), "{removal}: the file loses the key and the section");
        assert_eq!(running().draft_rows, None, "{removal}: nothing sets it any more");
        assert_eq!(running().settings(), restarted().settings(), "{removal}: what the rewritten file starts");
    }
}

/// Spec playground/05 AC 29, the file's own value then a change at the
/// group key: a value for a field only Flash-Next takes goes to its
/// section, in the running configuration as in the file — so it replaces
/// the file's value rather than yielding to it — and a `null` at the group
/// key removes that section's value.
#[tokio::test]
async fn a_family_only_value_in_the_file_is_replaced_and_removed_at_its_group_key() {
    let path = config_file("family-only-file-remove", "spec:\n  qwen38flashnext:\n    draft_rows: 4\n");
    let argv = ["--config", path.to_str().unwrap(), "--profile", "none"];
    let loader = MockLoader::new();
    let server = family_server(&loader, ModelFamily::FlashNext, started(&argv));
    let app = server.app();
    let running = || server.config.as_ref().unwrap().current();
    let restarted = || started(&argv).for_family(ModelFamily::FlashNext).unwrap();
    assert_eq!(running().draft_rows, Some(4));

    let applied = patched(&app, json!({ "spec": { "draft_rows": 6 } })).await;
    assert_eq!(applied["applied"], "reload", "the running family's own value changes: {applied}");
    assert_eq!(read_yaml(&path)["spec"]["qwen38flashnext"]["draft_rows"], 6);
    assert_eq!(running().draft_rows, Some(6), "the change runs, as the file now says");
    assert_eq!(running().settings(), restarted().settings());

    patched(&app, json!({ "spec": { "draft_rows": null } })).await;
    assert!(read_yaml(&path).get("spec").is_none(), "the file loses the section's key");
    assert_eq!(running().draft_rows, None);
    assert_eq!(running().settings(), restarted().settings(), "what the rewritten file starts");

    // The file's own value, removed at the group key with no change before.
    let path = config_file("family-only-file-null", "spec:\n  qwen38flashnext:\n    draft_rows: 4\n");
    let argv = ["--config", path.to_str().unwrap(), "--profile", "none"];
    let loader = MockLoader::new();
    let server = family_server(&loader, ModelFamily::FlashNext, started(&argv));
    let app = server.app();
    patched(&app, json!({ "spec": { "draft_rows": null } })).await;
    assert!(read_yaml(&path).get("spec").is_none(), "the file loses the section's key");
    let running = server.config.as_ref().unwrap().current();
    assert_eq!(running.draft_rows, None);
    assert_eq!(running.settings(), started(&argv).for_family(ModelFamily::FlashNext).unwrap().settings());
}

/// Spec playground/05 AC 30: a patched group value reaches the 27B while the
/// server runs, over the 27B's own flag; the answer names that flag, since a
/// start from the rewritten file runs the flag's value again.
#[tokio::test]
async fn a_family_flag_that_outranks_a_patched_group_value_at_restart_is_named() {
    let path = config_file("shadowed-family-flag", "server:\n  request_timeout: 30\n");
    let argv = ["--config", path.to_str().unwrap(), "--profile", "none", "--qwen38-vram-headroom-bytes", "1G"];
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &argv);
    let app = server.app();
    let applied = patched(&app, json!({ "vram": { "headroom_bytes": "2G" } })).await;
    assert_eq!(applied["applied"], "reload", "{applied}");
    assert_eq!(applied["shadowed_at_restart"], json!(["vram.headroom_bytes (set by --qwen38-vram-headroom-bytes)"]), "{applied}");
    assert_eq!(server.config.as_ref().unwrap().current().settings().vram.headroom_bytes, 2 << 30, "the patch runs");
    let restarted = started(&argv).for_family(ModelFamily::Qwen38_27b).unwrap();
    assert_eq!(restarted.settings().vram.headroom_bytes, 1 << 30, "the flag's value at the next start");
}

/// The text of `GET /v1/config?format=yaml`.
async fn yaml_of(app: &axum::Router) -> String {
    let request = Request::builder().uri("/v1/config?format=yaml").body(Body::empty()).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    String::from_utf8(response.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap()
}

/// What `ignis-server config print` writes for the config file at `path`
/// and `flags`, minus the hidden fields.
fn printed(path: &str, flags: &[&str]) -> String {
    let argv: Vec<String> = ["config", "print", "--file", path].iter().chain(flags).map(|s| s.to_string()).collect();
    let ConfigOutcome::Print(printed) = resolve_with(&argv, |_| None, &RealFiles).unwrap() else { panic!("config print prints") };
    let mut printed = Format::Yaml.read(&printed).unwrap();
    ignis_server::config_http::remove_invisible(&mut printed, all_fields());
    Format::Yaml.write(&printed)
}

/// Spec playground/05 AC 31: on a running configuration fitted to its
/// family — the built-in `rtx5090` profile with its per-family numbers, a
/// file with a family section, a flag — `GET /v1/config?format=yaml` is the
/// file `config print` writes for the same sources, whichever family runs;
/// and after a live change, the file `config print` writes from the
/// rewritten config file.
#[tokio::test]
async fn get_config_as_yaml_on_a_fitted_configuration_is_what_config_print_writes() {
    for family in [ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
        let path = config_file(
            &format!("yaml-fitted-{}", scope(family)),
            "reuse:\n  kv_host_pool_bytes: 3G\n  qwen38:\n    kv_host_pool_bytes: 4G\nmodel:\n  qwen38flashnext:\n    max_context: 65536\n",
        );
        let path = path.to_str().unwrap();
        let flags = ["--model-prefill-chunk", "2048"];
        let loader = MockLoader::new();
        let server = family_server(&loader, family, started(&[&["--config", path][..], &flags].concat()));
        assert_eq!(server.config.as_ref().unwrap().current().basis.family(), Some(family), "fitted");
        let app = server.app();
        assert_eq!(yaml_of(&app).await, printed(path, &flags), "{family:?}: the start's sources");

        let other = if family == ModelFamily::FlashNext { "qwen38" } else { "qwen38flashnext" };
        let body = json!({ "reuse": { other: { "prompt": false } }, "server": { "request_timeout": 45 } });
        let applied = patched(&app, body).await;
        assert_eq!((applied["applied"].clone(), applied["persisted"].clone()), (json!("live"), json!(true)), "{applied}");
        assert_eq!(yaml_of(&app).await, printed(path, &flags), "{family:?}: the rewritten file's");
    }
}

/// Spec playground/05 AC 31: a removal names its keys — a whole group or a
/// family's whole section set to `null` is `400 config_invalid`, and nothing
/// in the patch applies.
#[tokio::test]
async fn a_whole_group_or_section_set_to_null_is_refused_and_nothing_applies() {
    let path = config_file("null-whole", "server:\n  request_timeout: 30\nreuse:\n  qwen38:\n    prompt: false\n");
    let loader = MockLoader::new();
    let server = qwen_server(&loader, &["--config", path.to_str().unwrap(), "--profile", "none"]);
    let app = server.app();
    let before = std::fs::read_to_string(&path).unwrap();
    for (removal, says) in [(json!(null), "`reuse: null`"), (json!({ "qwen38": null }), "`reuse.qwen38: null`")] {
        let body = json!({ "server": { "request_timeout": 92 }, "reuse": removal });
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body.clone())).await;
        assert_eq!((status, refused["error"]["code"].clone()), (StatusCode::BAD_REQUEST, json!("config_invalid")), "{body}: {refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains(says) && message.contains("one field at a time"), "{message}");
    }
    assert_eq!(server.live().request_timeout, Duration::from_secs(30), "nothing applied");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before, "nothing written");
}

/// Spec playground/05 AC 28: the origins answer as `GET /v1/config` does —
/// during a switch, behind the API key, the preflight passed — and `501` on
/// a server with no running configuration.
#[tokio::test]
async fn the_origins_answer_during_a_switch_behind_the_key_and_501_without_a_config() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--profile", "none"]));
    let app = server.app();
    let release = loader.hold_next_load();
    let (status, _) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "reuse": { "kv_host_pool_bytes": "3G" } }))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let (status, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    assert_eq!(status, StatusCode::OK, "it answers during a switch: {origins}");
    assert_eq!(origins_of(&origins, "reuse.kv_host_pool_bytes")[0].1, "default", "the configuration still running");
    release.send(()).expect("the held load is waiting");
    until_serving(&app).await;
    let (_, origins) = send(&app, Method::GET, "/v1/config/origins", None).await;
    assert_eq!(origins_of(&origins, "reuse.kv_host_pool_bytes")[0].1, "live", "the reload's change, once it serves");

    let bare = Server::from_active(loader.model("mock-a")).app();
    let (status, body) = send(&bare, Method::GET, "/v1/config/origins", None).await;
    assert_eq!((status, body["error"]["code"].clone()), (StatusCode::NOT_IMPLEMENTED, json!("config_unavailable")), "{body}");

    let keyed = server_on(&loader, started(&[])).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let (status, _) = send(&keyed, Method::GET, "/v1/config/origins", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = send(&keyed, Method::OPTIONS, "/v1/config/origins", None).await;
    assert_eq!(status, StatusCode::OK, "the preflight passes, like the other /v1 routes");
}

/// Spec image-model/05 AC 10: every `image.*` field is in `GET
/// /v1/config/fields` — visible, patchable, needing no reload, its bound as
/// a rule — and in `GET /v1/config`; a live `PATCH` changes it at once with
/// no model touched, and one past its bound applies nothing.
#[tokio::test]
async fn the_image_fields_are_listed_and_change_live_within_their_bounds() {
    let path = config_file("image", "image:\n  queue_depth: 3\n");
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--config", path.to_str().unwrap()]));
    let app = server.app();
    let (_, fields) = send(&app, Method::GET, "/v1/config/fields", None).await;
    let names: Vec<&str> = all_fields().filter(|meta| meta.group == "image").map(|meta| meta.name).collect();
    assert_eq!(
        names,
        [
            "queue_depth", "max_n", "max_condition_images", "max_condition_mipx", "max_condition_decodes", "sizes", "max_target_cells",
            "condition_budget_tokens", "min_condition_budget_tokens", "condition_floor_px",
            "prompt_enhancement", "prompt_enhancer", "prompt_enhancer_max_tokens",
        ]
    );
    for name in names {
        let field = declaration(&fields, &format!("image.{name}"));
        assert_eq!((field["patchable"].clone(), field["reload_required"].clone()), (json!(true), json!(false)), "{name}");
        assert_eq!(field["families"], Value::Null, "{name}: every family takes it");
    }
    assert_eq!(declaration(&fields, "image.queue_depth")["rule"], json!({ "min": 0, "max": 64 }));
    assert_eq!(declaration(&fields, "image.max_condition_images")["rule"], json!({ "min": 1, "max": 10 }));
    assert_eq!(declaration(&fields, "image.max_target_cells")["rule"], json!({ "min": 1, "max": 1 << 20 }));
    assert_eq!(declaration(&fields, "image.max_condition_mipx")["rule"], json!({ "min": 1, "max": 64 }));
    assert_eq!(declaration(&fields, "image.sizes")["default"][7], "1024x1024");
    let (_, document) = send(&app, Method::GET, "/v1/config", None).await;
    assert_eq!(document["image"]["queue_depth"], 3, "{document}");
    assert_eq!(server.live().image.queue_depth, 3);

    let body = json!({ "image": { "queue_depth": 2, "sizes": ["1024x1024"], "prompt_enhancement": true } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!((status, applied["applied"].as_str()), (StatusCode::OK, Some("live")), "{applied}");
    let image = &server.live().image;
    assert_eq!((image.queue_depth, image.sizes.len(), image.prompt_enhancement), (2, 1, true));
    assert!(loader.prepared_options().is_empty(), "no model was touched");

    let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "image": { "max_n": 65, "queue_depth": 1 } }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("image.max_n"), "{refused}");
    assert_eq!(server.live().image.queue_depth, 2, "nothing in a refused patch applies");
}

/// Spec image-model/05 §Prompt enhancement (review-322cf M3): a patch naming
/// `image.prompt_enhancer` is refused whole unless it names a text model a
/// rewrite can run on — the loaded one, or one an explicit switch may name,
/// the patch's own `switch.known_models` among them; the image model is
/// none. A patch that does not name it is not judged by it.
#[tokio::test]
async fn a_patch_naming_an_enhancer_no_switch_reaches_is_refused() {
    let loader = MockLoader::new();
    let server = server_on(&loader, started(&["--image-prompt-enhancer", "nobody"]));
    let app = server.app();
    let enhancer = || server.live().image.prompt_enhancer.clone();
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "image": { "queue_depth": 1 } }))).await;
    assert_eq!(status, StatusCode::OK, "an enhancer the start took is not judged again: {applied}");

    for (body, named) in [
        (json!({ "image": { "prompt_enhancer": "nobody-else", "queue_depth": 2 } }), "nobody-else"),
        (
            json!({ "switch": { "known_models": { "ignis-image-2.1-turbo": "image.ninfer" } }, "image": { "prompt_enhancer": "ignis-image-2.1-turbo" } }),
            "ignis-image-2.1-turbo",
        ),
    ] {
        let (status, refused) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
        assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("config_invalid")), "{refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains("image.prompt_enhancer") && message.contains(named), "{message}");
        assert_eq!((server.live().image.queue_depth, enhancer()), (1, "nobody".to_owned()), "nothing applied");
    }

    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(json!({ "image": { "prompt_enhancer": "mock-a" } }))).await;
    assert_eq!(status, StatusCode::OK, "the loaded model: {applied}");
    let body = json!({ "switch": { "known_models": { "mock-b": "mock-b.ninfer" } }, "image": { "prompt_enhancer": "mock-b" } });
    let (status, applied) = send(&app, Method::PATCH, "/v1/config", Some(body)).await;
    assert_eq!(status, StatusCode::OK, "a model the patch itself lists: {applied}");
    assert_eq!(enhancer(), "mock-b");
}
