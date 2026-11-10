//! A server over a mock engine reporting Flash-Next's identity (spec
//! flash-next/04, GitHub #302): with no `--model-id`, it is served under
//! Flash-Next's own id, text is served, and a request for what Flash-Next
//! does not have -- a `/v1/decide` readout -- is a 400 naming the model,
//! never a silent degradation. An image is Flash-Next's to take since GitHub
//! #557 (ADR 0087): with vision on it is served on both APIs, and with
//! vision off it is refused as on the 27B. The 27B's refusals are unchanged.
//! Over the real axum router, CPU-only (ADR 0006).

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_artifact::vision::ProcessorOptions;
use ignis_core::compute::ModelFamily;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::config::{resolve, served_model_for, ConfigOutcome};
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor};

const FLASH_NEXT: &str = "qwen3.8-flash-next";
const PIXEL: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

fn app(model: &str, family: ModelFamily) -> axum::Router {
    server(model, family).app()
}

fn server(model: &str, family: ModelFamily) -> Server {
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
}

/// The id a Flash-Next load started with no `--model-id` is served under:
/// what the start options and the artifact's family decide, as `main`
/// decides it.
fn flash_next_id() -> String {
    let ConfigOutcome::Config(config) = resolve(&[], |_| None).expect("resolve") else {
        panic!("no flags resolve to a config");
    };
    served_model_for(&config, ModelFamily::FlashNext).expect("Flash-Next takes the defaults")
}

fn flash_next() -> axum::Router {
    app(&flash_next_id(), ModelFamily::FlashNext)
}

/// A Flash-Next load with vision on (GitHub #557): its media acquired as the
/// 27B's are, through the processor its load serves.
fn flash_next_with_vision() -> axum::Router {
    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    let acquirer = MediaAcquirer::new(Arc::new(processor(limits.clone())), limits, MediaPolicy::new(false, 0));
    server(&flash_next_id(), ModelFamily::FlashNext).with_media(Arc::new(acquirer)).app()
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
async fn an_unnamed_flash_next_load_is_listed_under_its_own_id() {
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

fn responses_image(url: &str) -> Value {
    json!({
        "model": FLASH_NEXT,
        "input": [{
            "role": "user",
            "content": [
                { "type": "input_text", "text": "what is this?" },
                { "type": "input_image", "image_url": url }
            ]
        }],
        "max_output_tokens": 4
    })
}

/// GitHub #557: Flash-Next takes images, so a load without vision refuses one
/// as the 27B's does, saying vision is off rather than that the model has no
/// tower; `/v1/tokenize` counts no image, as on the 27B.
#[tokio::test]
async fn an_image_on_a_flash_next_load_without_vision_is_refused_as_on_the_27b() {
    let app = flash_next();
    let chat = json!({ "model": FLASH_NEXT, "messages": image_message(), "max_tokens": 4 });
    let (status, body) = send(&app, "POST", "/v1/chat/completions", Some(chat)).await;
    assert_eq!((status, body["error"]["code"].as_str()), (400, Some("vision_disabled")), "{body}");
    assert_eq!(body["error"]["message"], "message 0 content part 1: vision is disabled for this server", "{body}");

    let (status, body) = send(&app, "POST", "/v1/responses", Some(responses_image(PIXEL))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (400, Some("vision_disabled")), "{body}");

    let tokenize = json!({ "model": FLASH_NEXT, "messages": image_message() });
    let (status, body) = send(&app, "POST", "/v1/tokenize", Some(tokenize)).await;
    assert_eq!((status, body["error"]["code"].as_str()), (400, Some("media_not_countable")), "{body}");
}

/// GitHub #557: a Flash-Next load with vision on answers an image through
/// chat completions and Responses alike, its placeholders counted in usage.
#[tokio::test]
async fn a_flash_next_load_with_vision_answers_an_image_on_both_apis() {
    let app = flash_next_with_vision();
    let image = data_uri(&png(64, 64));
    let message = json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": "what is this?" },
            { "type": "image_url", "image_url": { "url": image } }
        ]
    }]);
    let chat = json!({ "model": FLASH_NEXT, "messages": message, "max_tokens": 4 });
    let (status, body) = send(&app, "POST", "/v1/chat/completions", Some(chat)).await;
    assert_eq!(status, 200, "{body}");
    let chat_tokens = body["usage"]["prompt_tokens"].as_u64().expect("usage");

    let (status, body) = send(&app, "POST", "/v1/responses", Some(responses_image(&image))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["input_tokens"].as_u64(), Some(chat_tokens), "the same prompt on both APIs: {body}");
    assert!(chat_tokens > 4, "the image's placeholders are counted: {chat_tokens}");
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
