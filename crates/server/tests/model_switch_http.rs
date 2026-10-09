//! `POST /v1/models/switch` and what the API says while a switch runs (spec
//! model-switch/01, GitHub #305), over the mock-backed router: `202` with
//! the envelope, `GET /v1/models` reporting `switching` and then `serving`
//! the new id, a request sent mid-switch refused `503 model_switching` with
//! `Retry-After`, a second switch `409`. The switch is held inside its load
//! by the mock loader (`support/switch.rs`), never by a timer (ADR 0006).

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

const PATIENT: Duration = Duration::from_secs(30);

fn server_on(loader: &Arc<MockLoader>) -> Server {
    Server::from_active(loader.model("mock-a")).with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, headers, body)
}

fn switch_to(model: &str) -> Value {
    json!({ "artifact": format!("{model}.ninfer"), "model": model })
}

fn chat(model: &str) -> Value {
    json!({ "model": model, "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 })
}

/// `GET /v1/models` until its `status` is `serving` (scheduling turns,
/// bounded): the switch finishes on its own task.
async fn until_serving(app: &axum::Router) -> Value {
    for _ in 0..10_000 {
        let (status, _, body) = send(app, Method::GET, "/v1/models", None).await;
        if status == StatusCode::OK && body["status"] == "serving" {
            return body;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the switch never finished");
}

#[tokio::test]
async fn a_switch_is_accepted_reported_while_it_runs_and_serves_the_new_model_after() {
    let loader = MockLoader::new();
    let server = server_on(&loader);
    let app = server.app();
    let (status, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "serving");
    assert_eq!(models["data"][0]["id"], "mock-a");
    assert!(models.get("switching").is_none() && models.get("reason").is_none(), "{models}");

    let release = loader.hold_next_load();
    let (status, _, accepted) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    assert_eq!(accepted, json!({ "status": "switching", "from": "mock-a", "to": "mock-b" }));

    // Mid-switch: the models route reports it, everything else waits.
    let (status, _, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["status"], "switching");
    assert_eq!(models["switching"], json!({ "from": "mock-a", "to": "mock-b" }));
    assert_eq!(models["data"][0]["id"], "mock-a", "the old model is named until the new one serves");

    let (status, headers, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "model_switching", "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap().contains("mock-b"), "{refused}");
    assert_eq!(headers[header::RETRY_AFTER], "1");
    assert_eq!(headers["access-control-allow-origin"], "*");

    let (status, _, conflict) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-c"))).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(conflict["error"]["code"], "switch_in_progress");
    assert!(conflict["error"]["message"].as_str().unwrap().contains("mock-a to mock-b"), "{conflict}");

    release.send(()).expect("the held load is waiting");
    let models = until_serving(&app).await;
    assert_eq!(models["data"][0]["id"], "mock-b");
    let (status, _, completion) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b"))).await;
    assert_eq!(status, StatusCode::OK, "{completion}");
    assert_eq!(completion["model"], "mock-b");
    let (status, _, stale) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "the old id is no longer served: {stale}");
}

#[tokio::test]
async fn a_failed_switch_with_nothing_to_reload_says_why_and_takes_the_next_switch() {
    let loader = MockLoader::new();
    loader.break_load("mock-broken");
    let mut first = loader.model("mock-a");
    first.source = None;
    let server = Server::from_active(first).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-broken"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let mut models = Value::Null;
    for _ in 0..10_000 {
        models = send(&app, Method::GET, "/v1/models", None).await.2;
        if models["status"] == "failed" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(models["status"], "failed", "{models}");
    assert!(models["reason"].as_str().unwrap().contains("simulated kernel load error"), "{models}");
    let (status, _, refused) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "server_not_ready", "{refused}");

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "a failed server is recovered by switching");
    assert_eq!(until_serving(&app).await["data"][0]["id"], "mock-b");
}

#[tokio::test]
async fn a_switch_names_both_fields_or_is_refused() {
    let loader = MockLoader::new();
    let app = server_on(&loader).app();
    for (body, param) in [
        (json!({ "artifact": "", "model": "mock-b" }), "artifact"),
        (json!({ "artifact": "mock-b.ninfer", "model": " " }), "model"),
    ] {
        let (status, _, refused) = send(&app, Method::POST, "/v1/models/switch", Some(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(refused["error"]["param"], param, "{refused}");
    }
    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(json!({ "model": "mock-b" }))).await;
    assert!(status.is_client_error(), "a missing artifact is refused: {status}");
    assert!(loader.loads().is_empty());
}

#[tokio::test]
async fn a_server_without_a_loader_answers_501_and_a_warming_one_503() {
    let loader = MockLoader::new();
    let bare = Server::from_active(loader.model("mock-a")).app();
    let (status, _, refused) = send(&bare, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(refused["error"]["code"], "switch_unavailable");

    let warming = server_on(&loader).with_warm_up().app();
    let (status, _, refused) = send(&warming, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused["error"]["code"], "server_not_ready");
    let (status, _, _) = send(&warming, Method::GET, "/v1/models", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "the warm-up still holds every route, as it did");
    let (status, _, _) = send(&warming, Method::OPTIONS, "/v1/models/switch", None).await;
    assert_eq!(status, StatusCode::OK, "a preflight passes");
}

#[tokio::test]
async fn the_switch_route_is_behind_the_api_key() {
    let loader = MockLoader::new();
    let app = server_on(&loader).with_api_key(ignis_server::config::ApiKey::new("sk-test")).app();
    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(loader.loads().is_empty());
}

async fn sample(metrics: &axum::Router, name: &str) -> String {
    let (_, _, text) = send(metrics, Method::GET, "/metrics", None).await;
    let text = text.as_str().expect("the exposition is text").to_owned();
    let line = text.lines().find(|l| l.starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("no {name}"));
    line.rsplit_once(' ').unwrap().1.to_owned()
}

/// `--metrics` follows the switch: the new model's requests are counted,
/// in the same counters the old model's were.
#[tokio::test]
async fn metrics_keep_counting_on_the_model_a_switch_loaded() {
    let loader = MockLoader::new();
    let server = server_on(&loader).with_metrics();
    let app = server.app();
    let metrics = server.metrics_app().expect("metrics are on");
    let (status, _, _) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-a"))).await;
    assert_eq!(status, StatusCode::OK);

    let (status, _, _) = send(&app, Method::POST, "/v1/models/switch", Some(switch_to("mock-b"))).await;
    assert_eq!(status, StatusCode::ACCEPTED);
    until_serving(&app).await;
    let (status, _, _) = send(&app, Method::POST, "/v1/chat/completions", Some(chat("mock-b"))).await;
    assert_eq!(status, StatusCode::OK);
    for _ in 0..10_000 {
        if sample(&metrics, "ignis_requests_completed_total").await == "2" {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(sample(&metrics, "ignis_requests_completed_total").await, "2", "one request on each model");
}
