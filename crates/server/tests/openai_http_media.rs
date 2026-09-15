//! Media on `POST /v1/chat/completions` (GitHub #179): what the HTTP seam
//! owes a `--vision` load before the model sees an image, over the real
//! axum router against a mock-compute engine (CPU-only, ADR 0006).

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::json;
use tower::ServiceExt;

use ignis_artifact::vision::{ProcessorOptions, VisionProcessor, IMAGE_PAD, IMAGE_PAD_ID, VIDEO_PAD, VIDEO_PAD_ID};
use ignis_artifact::Tokenizer;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

const MODEL: &str = "test-model";

fn processor(limits: ProcessorOptions) -> VisionProcessor {
    let added = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    let tokenizer = json!({
        "version": "1.0",
        "added_tokens": [added(IMAGE_PAD_ID, IMAGE_PAD), added(VIDEO_PAD_ID, VIDEO_PAD)],
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "WordLevel", "vocab": {"x": 0, IMAGE_PAD: IMAGE_PAD_ID, VIDEO_PAD: VIDEO_PAD_ID}, "unk_token": "x"},
    });
    VisionProcessor::new(&Tokenizer::from_bytes(tokenizer.to_string().as_bytes()).unwrap(), limits).unwrap()
}

/// The router over a fresh mock engine, with media acquisition when `vision`.
fn app(vision: bool) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(5));
    if !vision {
        return server.app();
    }
    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    let acquirer = MediaAcquirer::new(Arc::new(processor(limits.clone())), limits, MediaPolicy::new(false, 0));
    server.with_media(Arc::new(acquirer)).app()
}

/// POST a chat request whose one text part is `text_bytes` long.
async fn post_large(app: axum::Router, text_bytes: usize) -> u16 {
    let body = json!({
        "model": MODEL,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x".repeat(text_bytes)}]}],
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let _ = to_bytes(response.into_body(), usize::MAX).await;
    status
}

#[tokio::test]
async fn a_vision_load_accepts_a_request_body_past_the_default_limit() {
    // 4 MiB: an inline data URI of a ~3 MiB screenshot.
    assert_eq!(post_large(app(false), 4 << 20).await, 413, "a text-only load keeps axum's default");
    assert_ne!(post_large(app(true), 4 << 20).await, 413, "a vision load takes large inline media");
}
