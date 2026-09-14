//! `--api-key` / `IGNIS_API_KEY`: with a key configured every `/v1` route
//! needs `Authorization: Bearer <key>` (OpenAI's `401 invalid_api_key`
//! otherwise); a CORS preflight and the Playground's pages stay open;
//! without a key nothing changes.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::playground::Assets;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";
const KEY: &str = "sk-test-key";
const PLAYGROUND: Assets = &[("index.html", b"<!doctype html>")];

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

async fn send(app: &axum::Router, method: Method, uri: &str, auth: Option<&str>) -> axum::response::Response {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(auth) = auth {
        request = request.header(header::AUTHORIZATION, auth);
    }
    app.clone().oneshot(request.body(Body::empty()).unwrap()).await.unwrap()
}

async fn body_json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn without_a_key_the_api_is_open() {
    let app = server().app();
    assert_eq!(send(&app, Method::GET, "/v1/models", None).await.status(), StatusCode::OK);
    let any = send(&app, Method::GET, "/v1/models", Some("Bearer whatever")).await;
    assert_eq!(any.status(), StatusCode::OK);
}

#[tokio::test]
async fn with_a_key_a_request_without_it_is_refused_openai_style() {
    let app = server().with_api_key(ApiKey::new(KEY)).app();
    for uri in ["/v1/models"] {
        let res = send(&app, Method::GET, uri, None).await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{uri}");
        assert_eq!(res.headers()[header::WWW_AUTHENTICATE], "Bearer");
        // The 401 still carries CORS headers, so a browser client can read it.
        assert_eq!(res.headers()["access-control-allow-origin"], "*");
        let body = body_json(res).await;
        assert_eq!(body["error"]["code"], "invalid_api_key");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
    for uri in ["/v1/chat/completions", "/v1/responses"] {
        let res = send(&app, Method::POST, uri, None).await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{uri}");
    }
}

#[tokio::test]
async fn with_a_key_a_wrong_or_malformed_one_is_refused() {
    let app = server().with_api_key(ApiKey::new(KEY)).app();
    for auth in ["Bearer sk-wrong", KEY, "Basic sk-test-key", "Bearer "] {
        let res = send(&app, Method::GET, "/v1/models", Some(auth)).await;
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED, "{auth:?}");
    }
}

#[tokio::test]
async fn with_a_key_the_right_one_is_served() {
    let app = server().with_api_key(ApiKey::new(KEY)).app();
    let res = send(&app, Method::GET, "/v1/models", Some(&format!("Bearer {KEY}"))).await;
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(body_json(res).await["data"][0]["id"], MODEL);
}

#[tokio::test]
async fn with_a_key_preflight_and_the_playground_stay_open() {
    let app = server()
        .with_api_key(ApiKey::new(KEY))
        .with_playground(PLAYGROUND)
        .app();
    let preflight = send(&app, Method::OPTIONS, "/v1/chat/completions", None).await;
    assert_eq!(preflight.status(), StatusCode::OK);
    assert_eq!(send(&app, Method::GET, "/ui/", None).await.status(), StatusCode::OK);
    // An unknown path is still a 404, not a 401.
    assert_eq!(send(&app, Method::GET, "/nope", None).await.status(), StatusCode::NOT_FOUND);
}
