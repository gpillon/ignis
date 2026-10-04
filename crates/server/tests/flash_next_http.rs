//! A server over a mock engine reporting Flash-Next's identity (spec
//! flash-next/04, GitHub #302): `/v1/models` names the loaded model, text is
//! served, and a request for what Flash-Next does not have -- an image, a
//! `/v1/decide` readout -- is a 400 naming the model, never a silent
//! degradation. The 27B's refusals are unchanged. Over the real axum router,
//! CPU-only (ADR 0006).

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_core::compute::ModelFamily;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

const FLASH_NEXT: &str = "qwen3.8-flash-next";
const PIXEL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn app(model: &str, family: ModelFamily) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: model.into(),
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_family(family)
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
        .app()
}

fn flash_next() -> axum::Router {
    app(FLASH_NEXT, ModelFamily::FlashNext)
}

async fn send(app: &axum::Router, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(body.map_or_else(Body::empty, |body| Body::from(body.to_string())))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let value = serde_json::from_str(&text).unwrap_or_else(|err| panic!("body is not JSON ({err}): {text}"));
    (status, value)
}

fn image_message() -> Value {
    json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": "what is this?" },
            { "type": "image_url", "image_url": { "url": PIXEL } }
        ]
    }])
}

fn assert_names_flash_next(status: u16, body: &Value, code: &str) {
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("Qwen3.8-Flash-Next"), "the refusal names the model: {body}");
}

#[tokio::test]
async fn the_models_list_names_flash_next() {
    let (status, body) = send(&flash_next(), "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"][0]["id"], FLASH_NEXT, "{body}");
}

#[tokio::test]
async fn flash_next_serves_text() {
    let body = json!({ "model": FLASH_NEXT, "messages": [{ "role": "user", "content": "hi" }], "max_tokens": 4 });
    let (status, body) = send(&flash_next(), "POST", "/v1/chat/completions", Some(body)).await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn an_image_is_refused_naming_flash_next_on_every_endpoint() {
    let app = flash_next();
    let chat = json!({ "model": FLASH_NEXT, "messages": image_message(), "max_tokens": 4 });
    let (status, body) = send(&app, "POST", "/v1/chat/completions", Some(chat)).await;
    assert_names_flash_next(status, &body, "vision_disabled");

    let responses = json!({
        "model": FLASH_NEXT,
        "input": [{
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what is this?" },
                { "type": "input_image", "image_url": PIXEL }
            ]
        }],
        "max_output_tokens": 4
    });
    let (status, body) = send(&app, "POST", "/v1/responses", Some(responses)).await;
    assert_names_flash_next(status, &body, "vision_disabled");
}

#[tokio::test]
async fn decide_is_refused_naming_flash_next() {
    let decide = json!({
        "model": FLASH_NEXT,
        "context": "The sky is blue.",
        "questions": [{ "id": "q", "kind": "boolean", "instructions": "Is the sky blue?" }]
    });
    let (status, body) = send(&flash_next(), "POST", "/v1/decide", Some(decide)).await;
    assert_names_flash_next(status, &body, "model_unsupported");
}

/// The 27B's refusal of an image on a load without vision says what it
/// always said, and names no model.
#[tokio::test]
async fn the_27b_image_refusal_is_unchanged() {
    let app = app("test-model", ModelFamily::Qwen38_27b);
    let chat = json!({ "model": "test-model", "messages": image_message(), "max_tokens": 4 });
    let (status, body) = send(&app, "POST", "/v1/chat/completions", Some(chat)).await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"], "vision_disabled", "{body}");
    assert_eq!(
        body["error"]["message"],
        "message 0 content part 1: vision is disabled for this server",
        "{body}"
    );
}
