//! GitHub #129: a server built `with_warm_up` answers every `/v1` route 503
//! `server_not_ready` until its first traversal has run; then it admits
//! requests as before. One built without it is ready as constructed.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

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
    assert_eq!(held.headers()[header::RETRY_AFTER], "1");
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

#[tokio::test]
async fn a_keyed_server_reports_not_ready_before_it_asks_for_the_key() {
    let server = server().with_api_key(ignis_server::config::ApiKey::new("sk-test")).with_warm_up();
    let app = server.app();
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::SERVICE_UNAVAILABLE);
    server.warm_up().await.expect("the mock warms up");
    assert_eq!(send(&app, Method::GET, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}
