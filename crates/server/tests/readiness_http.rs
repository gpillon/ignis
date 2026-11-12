//! GitHub #129: a server built `with_warm_up` answers every `/v1` route 503
//! `server_not_ready` until its first traversal has run; then it admits
//! requests as before. One built without it is ready as constructed, and
//! one whose warm-up the engine ends with an error stays not ready.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

async fn send(app: &axum::Router, method: Method, uri: &str) -> axum::response::Response {
    app.clone()
        .oneshot(Request::builder().method(method).uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn a_server_without_warm_up_is_ready_as_built() {
    let server = server();
    assert!(server.is_ready());
    assert_eq!(send(&server.app(), Method::GET, "/v1/models").await.status(), StatusCode::OK);
}

#[tokio::test]
async fn the_api_answers_503_until_the_warm_up_has_run() {
    let server = server().with_warm_up();
    let app = server.app();
    assert!(!server.is_ready());

    let held = send(&app, Method::GET, "/v1/models").await;
    assert_eq!(held.status(), StatusCode::SERVICE_UNAVAILABLE);
    // GitHub #327: before the first warm-up, a few seconds rather than one.
    assert_eq!(held.headers()[header::RETRY_AFTER], "5");
    assert_eq!(held.headers()["access-control-allow-origin"], "*");
    let body: serde_json::Value =
        serde_json::from_slice(&held.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["error"]["code"], "server_not_ready");
    // A preflight and the API reference stay reachable.
    assert_eq!(send(&app, Method::OPTIONS, "/v1/models").await.status(), StatusCode::OK);
    assert_eq!(send(&app, Method::GET, "/v1/openapi.json").await.status(), StatusCode::OK);

    server.warm_up().await.expect("the mock warms up");
    assert!(server.is_ready());
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::OK);
}

/// The engine ended the warm-up's request (`FinishReason::Error`, as a leaf
/// failing it would): a request that is done, but a traversal that never
/// ran — so not a warm-up, and the API stays held back.
#[tokio::test]
async fn a_warm_up_the_engine_ended_with_an_error_leaves_the_server_not_ready() {
    let loader = MockLoader::new();
    loader.fail_requests("mock-a");
    let server = Server::from_active(loader.model("mock-a")).with_warm_up();
    let app = server.app();

    let failed = server.warm_up().await.expect_err("an ended request is not a warm-up");
    assert!(failed.contains("ended with an error"), "{failed}");
    assert!(!server.is_ready());
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn a_keyed_server_reports_not_ready_before_it_asks_for_the_key() {
    let server = server().with_api_key(ignis_server::config::ApiKey::new("sk-test")).with_warm_up();
    let app = server.app();
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::SERVICE_UNAVAILABLE);
    server.warm_up().await.expect("the mock warms up");
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

/// A request of a client's, submitted after the warm-up. The telemetry
/// consumer drains facts in order, so once this one shows in a counter every
/// fact the warm-up produced has been handled (or dropped) already.
async fn a_clients_request(server: &Server) {
    let input = ignis_core::RequestInput {
        decision: None,
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        model: server.active().model_id(),
        tokens: vec![5, 6, 7],
        params: ignis_core::DecodeParams { max_tokens: Some(3), ..ignis_core::DecodeParams::default() },
    };
    let (_, mut events) =
        server.active().text_engine().submit(input, ignis_core::RequestClass::Interactive).await.expect("submit");
    ignis_server::engine::collect_completion(&mut events, std::time::Duration::from_secs(5))
        .await
        .expect("completes");
}

async fn sample(metrics: &axum::Router, name: &str) -> String {
    let text = String::from_utf8(
        send(metrics, Method::GET, "/metrics").await.into_body().collect().await.unwrap().to_bytes().to_vec(),
    )
    .unwrap();
    let line = text.lines().find(|l| l.starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("no {name}"));
    line.rsplit_once(' ').unwrap().1.to_owned()
}

#[tokio::test]
async fn the_warm_up_is_not_a_request_in_the_metrics_or_the_log() {
    use tracing_subscriber::layer::SubscriberExt;
    let log = std::sync::Arc::new(ignis_logging::MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(log.clone())),
    );
    let server = server().with_metrics().with_warm_up();
    let metrics = server.metrics_app().expect("metrics are on");
    server.warm_up().await.expect("the mock warms up");

    a_clients_request(&server).await;
    for _ in 0..200 {
        if sample(&metrics, "ignis_requests_completed_total").await == "1" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    // Only the client's request is in the counters...
    assert_eq!(sample(&metrics, "ignis_requests_accepted_total").await, "1");
    assert_eq!(sample(&metrics, "ignis_requests_completed_total").await, "1");
    assert_eq!(sample(&metrics, "ignis_generated_tokens_total").await, "3");
    assert_eq!(sample(&metrics, "ignis_decoded_tokens_total").await, "3");
    // ...and in the request log: one done line, not two.
    let done = log
        .lines()
        .iter()
        .filter(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["event_name"] == "ignis.request.done")
        .count();
    assert_eq!(done, 1, "the warm-up logs no request line");
}

// ── GET /health and GET /ready (GitHub #327) ─────────────────────────────

async fn probe(app: &axum::Router, path: &str) -> (StatusCode, Option<String>, serde_json::Value) {
    let response = send(app, Method::GET, path).await;
    let status = response.status();
    let retry_after = response.headers().get(header::RETRY_AFTER).map(|v| v.to_str().unwrap().to_owned());
    let body = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    (status, retry_after, body)
}

/// Liveness: the process answers, so it is alive — while the model loads
/// too, or a probe that restarts what fails it would restart a loading
/// server in a loop.
#[tokio::test]
async fn the_liveness_probe_answers_200_from_the_start() {
    let server = server().with_warm_up();
    let app = server.app();
    assert_eq!(probe(&app, "/health").await, (StatusCode::OK, None, serde_json::json!({ "status": "ok" })));
    server.warm_up().await.expect("the mock warms up");
    assert_eq!(probe(&app, "/health").await.0, StatusCode::OK);
}

#[tokio::test]
async fn the_readiness_probe_answers_503_until_the_first_warm_up_then_200() {
    let server = server().with_warm_up();
    let app = server.app();
    let (status, retry_after, body) = probe(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(retry_after.as_deref(), Some("5"));
    assert_eq!(body["error"]["code"], "server_not_ready");

    server.warm_up().await.expect("the mock warms up");
    let (status, retry_after, body) = probe(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(retry_after, None);
    assert_eq!(body, serde_json::json!({ "status": "ok" }));
}

/// Once the process has served, readiness stays green while a model is on
/// its way: a switch, an idle unload, the load after it, and that load
/// failing (the next request tries it again). `/v1` answers each of those on
/// its own (here, a switch is a 503). The one 503 after the warm-up, a
/// failed switch with nothing left to load, is `model_switch_http.rs`'s.
#[tokio::test]
async fn the_readiness_probe_stays_200_while_a_model_is_on_its_way() {
    use ignis_server::ModelStatus;
    let server = server();
    let app = server.app();
    let switching = ModelStatus::Switching { from: "test-model".into(), to: "other".into() };
    let failed = ModelStatus::Failed { reason: "no such file".into(), from_unloaded: true };
    for model in [switching, failed, ModelStatus::Unloaded, ModelStatus::Loading, ModelStatus::Serving] {
        let name = model.as_str();
        server.status.store(Arc::new(model));
        let (status, _, body) = probe(&app, "/ready").await;
        assert_eq!((status, &body), (StatusCode::OK, &serde_json::json!({ "status": "ok" })), "{name}");
    }
    let switching = ModelStatus::Switching { from: "test-model".into(), to: "other".into() };
    server.status.store(Arc::new(switching));
    assert_eq!(send(&app, Method::POST, "/v1/chat/completions").await.status(), StatusCode::SERVICE_UNAVAILABLE);
}

/// Both probes sit outside `/v1`: a keyed server answers them without the
/// key.
#[tokio::test]
async fn the_probes_ask_for_no_key() {
    let server = server().with_api_key(ignis_server::config::ApiKey::new("sk-test"));
    let app = server.app();
    assert_eq!(probe(&app, "/health").await.0, StatusCode::OK);
    assert_eq!(probe(&app, "/ready").await.0, StatusCode::OK);
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

/// A warm-up that never completed is a process that never served: not
/// ready, though alive.
#[tokio::test]
async fn after_a_failed_warm_up_the_server_is_alive_but_not_ready() {
    let loader = MockLoader::new();
    loader.fail_requests("mock-a");
    let server = Server::from_active(loader.model("mock-a")).with_warm_up();
    let app = server.app();
    server.warm_up().await.expect_err("an ended request is not a warm-up");
    assert_eq!(probe(&app, "/ready").await.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(probe(&app, "/health").await.0, StatusCode::OK);
}
