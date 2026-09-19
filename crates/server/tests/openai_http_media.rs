//! Media on the completion endpoints (GitHub #179): what the HTTP seam owes a
//! `--vision` load before the model sees an image, over the real axum router
//! against a mock-compute engine (CPU-only, ADR 0006).
//!
//! - An image request is submitted with its placeholders expanded and its
//!   multimodal part; `usage` counts the image tokens (chat and responses).
//! - A prompt that no longer fits after expansion is the existing
//!   `context_length_exceeded` 400; without `--vision` an image is
//!   `vision_disabled`; bad media are refused before anything is submitted.
//! - A client disconnect during preprocessing stops the work and submits
//!   nothing.
//!
//! The cache hit on the request log lives in `media_request_log.rs`: a test
//! that captures events needs a binary where every test installs a
//! subscriber.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_artifact::vision::{ProcessorOptions, IMAGE_PAD_ID};
use ignis_core::mock::MockCompute;
use ignis_core::{
    ConcreteScheduler, RequestClass, RequestId, RequestInput, SchedEvent, Scheduler, SchedulerConfig, SubmitError,
};
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy, Preparer};

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor, until, Gated};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

const MODEL: &str = "test-model";

// ── fixtures ────────────────────────────────────────────────────────────────

fn limits() -> ProcessorOptions {
    ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    }
}

/// The concrete scheduler, recording every submission it receives.
struct Recording {
    inner: ConcreteScheduler,
    submitted: Arc<Mutex<Vec<RequestInput>>>,
}

impl Scheduler for Recording {
    fn submit(&mut self, input: RequestInput, class: RequestClass) -> Result<RequestId, SubmitError> {
        self.submitted.lock().unwrap().push(input.clone());
        self.inner.submit(input, class)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.inner.cancel(request)
    }
    fn advance(&mut self) -> Vec<SchedEvent> {
        self.inner.advance()
    }
    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.inner.max_sequence_tokens()
    }
    fn mode(&self) -> ignis_core::EngineMode {
        self.inner.mode()
    }
    fn occupancy(&self) -> ignis_core::Occupancy {
        self.inner.occupancy()
    }
}

struct Harness {
    app: axum::Router,
    submitted: Arc<Mutex<Vec<RequestInput>>>,
}

/// The router over a recording mock engine; with `preparer`, a `--vision`
/// load acquiring media through it.
fn harness(preparer: Option<Arc<dyn Preparer>>, cache_bytes: u64, max_sequence_tokens: Option<u32>) -> Harness {
    let mut config = SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() };
    if let Some(tokens) = max_sequence_tokens {
        config.max_sequence_tokens = tokens;
    }
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let scheduler = Recording {
        inner: ConcreteScheduler::with_config(config, Arc::new(MockCompute::new())),
        submitted: submitted.clone(),
    };
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(10));
    let server = match preparer {
        None => server,
        Some(preparer) => server.with_media(Arc::new(MediaAcquirer::new(
            preparer,
            limits(),
            MediaPolicy::new(false, cache_bytes),
        ))),
    };
    Harness { app: server.app(), submitted }
}

fn vision() -> Harness {
    harness(Some(Arc::new(processor(limits()))), 0, None)
}

fn request(path: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send(app: &axum::Router, path: &str, body: Value) -> (u16, Value) {
    let response = app.clone().oneshot(request(path, body)).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// A user message: three words, then one image.
fn image_message(url: &str) -> Value {
    json!({"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": url}},
    ]})
}

fn chat(messages: Value) -> Value {
    json!({"model": MODEL, "max_tokens": 4, "messages": messages})
}

/// Assert the one submission is the three words plus the 64x64 image's
/// four placeholders, with its multimodal part.
fn assert_expanded(submitted: &Mutex<Vec<RequestInput>>) {
    let submitted = submitted.lock().unwrap();
    assert_eq!(submitted.len(), 1);
    let input = &submitted[0];
    assert_eq!(input.tokens.len(), 7, "{:?}", input.tokens);
    assert_eq!(&input.tokens[3..], [IMAGE_PAD_ID; 4]);
    let multimodal = input.multimodal.as_ref().expect("an image request carries its multimodal part");
    assert_eq!(multimodal.prompt_tokens(), 7);
    assert_eq!(multimodal.media.len(), 1);
    assert_eq!((multimodal.media[0].token_span.begin, multimodal.media[0].token_span.count), (3, 4));
}

// ── the expanded prompt ─────────────────────────────────────────────────────

#[tokio::test]
async fn a_chat_image_request_is_submitted_expanded_and_counted_in_usage() {
    let h = vision();
    let (status, body) = send(&h.app, "/v1/chat/completions", chat(json!([image_message(&data_uri(&png(64, 64)))]))).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 7, "{body}");
    assert_expanded(&h.submitted);
}

#[tokio::test]
async fn a_responses_image_request_is_submitted_expanded_and_counted_in_usage() {
    let h = vision();
    let body = json!({"model": MODEL, "max_output_tokens": 4, "input": [image_message(&data_uri(&png(64, 64)))]});
    let (status, body) = send(&h.app, "/v1/responses", body).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["input_tokens"], 7, "{body}");
    assert_expanded(&h.submitted);
}

#[tokio::test]
async fn a_text_request_on_a_vision_load_carries_no_multimodal_part() {
    let h = vision();
    let (status, body) = send(&h.app, "/v1/chat/completions", chat(json!([{"role": "user", "content": "hello there"}]))).await;
    assert_eq!(status, 200, "{body}");
    assert!(h.submitted.lock().unwrap()[0].multimodal.is_none());
}

#[tokio::test]
async fn a_prompt_that_no_longer_fits_after_expansion_is_context_exceeded() {
    // Three words + 4 generated fit a 10-token context; with the image's four
    // placeholders the request needs 11.
    let h = harness(Some(Arc::new(processor(limits()))), 0, Some(10));
    let text = chat(json!([{"role": "user", "content": "what is this"}]));
    assert_eq!(send(&h.app, "/v1/chat/completions", text).await.0, 200);
    let (status, body) = send(&h.app, "/v1/chat/completions", chat(json!([image_message(&data_uri(&png(64, 64)))]))).await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["error"]["code"], "context_length_exceeded", "{body}");
}

// ── refusals before submission ─────────────────────────────────────────────

#[tokio::test]
async fn without_vision_an_image_is_vision_disabled() {
    let h = harness(None, 0, None);
    let (status, body) = send(&h.app, "/v1/chat/completions", chat(json!([image_message(&data_uri(&png(64, 64)))]))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (400, Some("vision_disabled")), "{body}");
    assert!(h.submitted.lock().unwrap().is_empty());
}

#[tokio::test]
async fn bad_media_are_refused_naming_the_part_before_anything_is_submitted() {
    let h = vision();
    let cases = [
        ("data:image/png;base64,not*base64", "invalid_media"),
        ("data:image/png;base64,iVBORw0KGgpnYXJiYWdl", "invalid_media"),
        ("http://127.0.0.1:1/private.png", "invalid_media"),
    ];
    for (url, code) in cases {
        let (status, body) = send(&h.app, "/v1/chat/completions", chat(json!([image_message(url)]))).await;
        assert_eq!((status, body["error"]["code"].as_str()), (400, Some(code)), "{url}: {body}");
        let message = body["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("message 0 content part 1"), "{url}: {message}");
    }
    assert!(h.submitted.lock().unwrap().is_empty());
}

// ── cancellation ────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_disconnect_during_preprocessing_stops_the_work_and_submits_nothing() {
    let gated = Gated::new(false, processor(limits()));
    let h = harness(Some(gated.clone()), 1 << 20, None);
    let image = data_uri(&png(64, 64));
    let client = tokio::spawn(h.app.clone().oneshot(request("/v1/chat/completions", chat(json!([image_message(&image)])))));
    until("preprocessing to start", || gated.builds.load(Ordering::SeqCst) == 1).await;

    client.abort();
    assert!(client.await.unwrap_err().is_cancelled());
    until("preprocessing to observe the disconnect", || gated.observed_cancel.load(Ordering::SeqCst)).await;

    // The engine is still serving, and the only thing it was ever given is
    // the request sent after the disconnect.
    let text = chat(json!([{"role": "user", "content": "still there"}]));
    assert_eq!(send(&h.app, "/v1/chat/completions", text).await.0, 200);
    let submitted = h.submitted.lock().unwrap();
    assert_eq!(submitted.len(), 1);
    assert!(submitted[0].multimodal.is_none());
}

// ── the request body ────────────────────────────────────────────────────────

/// POST a chat request whose one text part is `text_bytes` long.
async fn post_large(app: &axum::Router, text_bytes: usize) -> u16 {
    let body = chat(json!([{"role": "user", "content": [{"type": "text", "text": "x".repeat(text_bytes)}]}]));
    send(app, "/v1/chat/completions", body).await.0
}

#[tokio::test]
async fn a_vision_load_accepts_a_request_body_past_the_text_limit() {
    // 4 MiB: an inline data URI of a ~3 MiB screenshot, or a prompt around
    // the engine's own context ceiling. Both loads take it now (GitHub
    // #230) -- axum's 2 MiB default sat under one max-context prompt.
    assert_ne!(post_large(&harness(None, 0, None).app, 4 << 20).await, 413, "4 MiB of prompt is not oversized");
    assert_ne!(post_large(&vision().app, 4 << 20).await, 413, "a vision load takes large inline media");
    // Each load still has a cap, and the text one is the lower of the two.
    assert!(ignis_server::api::TEXT_REQUEST_BODY_LIMIT < ignis_server::api::MEDIA_REQUEST_BODY_LIMIT);
    assert_eq!(
        post_large(&harness(None, 0, None).app, ignis_server::api::TEXT_REQUEST_BODY_LIMIT + (1 << 16)).await,
        413,
        "a text-only load refuses a body past its own limit"
    );
}
