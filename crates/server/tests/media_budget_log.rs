//! The vision budget on the request log (GitHub #673), over the real router
//! against a mock-compute engine (CPU-only, ADR 0006):
//!
//! - a request whose images pass the budget together loses its oldest, and
//!   its `ignis.request.admitted` and `ignis.request.done` lines say how many,
//!   and their tokens and bytes, beside the images the model saw;
//! - a request refused for its media, answered before admission, leaves one
//!   `ignis.media.refused` event naming the wire code and, for a budget, the
//!   budget, its limit and the amount requested.
//!
//! Its own binary, every test installing a subscriber: tracing caches a
//! callsite's interest process-wide, so a test running beside it without one
//! can switch these events off before a test captures them.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_artifact::vision::ProcessorOptions;
use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_logging::MemorySink;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor, shade};

const MODEL: &str = "test-model";

fn limits() -> ProcessorOptions {
    ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    }
}

/// The router over the mock, acquiring media under `limits`.
fn app(limits: ProcessorOptions) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let acquirer = MediaAcquirer::new(Arc::new(processor(limits.clone())), limits, MediaPolicy::new(false, 1 << 20));
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app()
}

/// A chat request: a text part, then one image part per URL.
fn images_request(images: &[&str]) -> Request<Body> {
    let mut content = vec![json!({"type": "text", "text": "what are these"})];
    content.extend(images.iter().map(|url| json!({"type": "image_url", "image_url": {"url": url}})));
    let body = json!({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": content}]});
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Every event `sink` captured, once it holds `n` named `name`, for 5 s at
/// most (CODING_STANDARDS.md, "A test waits for the whole end state").
async fn logged(sink: &MemorySink, name: &str, n: usize) -> Vec<Value> {
    let events = || -> Vec<Value> {
        sink.lines().iter().map(|line| serde_json::from_str::<Value>(line).unwrap()).collect()
    };
    let count = || events().iter().filter(|e| e["event_name"] == name).count();
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        while count() < n {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "{} {name} lines, wanted {n}", count());
    events()
}

/// The attributes of the one event named `name` among `events`.
fn attributes(events: &[Value], name: &str) -> Value {
    let named: Vec<&Value> = events.iter().filter(|e| e["event_name"] == name).collect();
    assert_eq!(named.len(), 1, "one {name}: {events:?}");
    named[0]["attributes"].clone()
}

#[tokio::test]
async fn a_request_past_its_vision_budget_logs_the_images_it_dropped() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    // 64 x 64 (4 tokens) then 96 x 64 (6), under a budget of 9: the older goes.
    let (older, newer) = (shade(64, 64, 40), shade(96, 64, 200));
    let app = app(ProcessorOptions { max_vision_tokens: 9, ..limits() });
    let response = app.oneshot(images_request(&[&data_uri(&older), &data_uri(&newer)])).await.unwrap();
    assert_eq!(response.status().as_u16(), 200, "admitted, not refused");
    let _ = to_bytes(response.into_body(), usize::MAX).await;

    let events = logged(&sink, "ignis.request.done", 1).await;
    for name in ["ignis.request.admitted", "ignis.request.done"] {
        let line = attributes(&events, name);
        let field = |key: &str| line[key].clone();
        assert_eq!((field("media.items"), field("media.vision_tokens")), (json!(1), json!(6)), "{name}: {line}");
        assert_eq!(field("media.bytes"), json!(newer.len()), "{name}");
        assert_eq!((field("media.dropped_items"), field("media.dropped_vision_tokens")), (json!(1), json!(4)), "{name}");
        assert_eq!(field("media.dropped_bytes"), json!(older.len()), "{name}");
    }
    assert!(events.iter().all(|e| e["event_name"] != "ignis.media.refused"), "{events:?}");
}

#[tokio::test]
async fn a_request_refused_for_its_media_leaves_one_event_with_the_code_and_the_numbers() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    // One image past the byte budget alone is still refused, before admission.
    let image = png(64, 64);
    let limit = image.len() as u64 - 1;
    let app = app(ProcessorOptions { max_encoded_media_bytes: limit, ..limits() });
    let response = app.clone().oneshot(images_request(&[&data_uri(&image)])).await.unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let events = logged(&sink, "ignis.media.refused", 1).await;
    let refused = attributes(&events, "ignis.media.refused");
    assert_eq!(refused["code"], "media_budget_exceeded", "{refused}");
    assert_eq!(refused["status"], 400);
    assert_eq!(refused["budget"], "encoded_bytes");
    assert_eq!((refused["limit"].clone(), refused["requested"].clone()), (json!(limit), json!(image.len())));
    assert_eq!(refused["part"], "message 0 content part 1", "{refused}");
    // Never the message the client read: a fetch's names its URL, which
    // may carry a signature.
    assert!(refused.get("reason").is_none() && refused.get("message").is_none(), "{refused}");
    let refusal = events.iter().find(|e| e["event_name"] == "ignis.media.refused").unwrap();
    assert_eq!(refusal["severity_text"], "WARN", "{refusal}");

    // A refusal that is no budget names its code alone.
    let response = app.oneshot(images_request(&[&data_uri(&png(201, 1))])).await.unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let events = logged(&sink, "ignis.media.refused", 2).await;
    let refused: Vec<&Value> = events.iter().filter(|e| e["event_name"] == "ignis.media.refused").collect();
    let invalid = &refused[1]["attributes"];
    assert_eq!((invalid["code"].clone(), invalid["part"].clone()), (json!("invalid_media"), json!("message 0 content part 1")), "{invalid}");
    assert!(invalid.get("budget").is_none() && invalid.get("limit").is_none(), "{invalid}");
    assert!(events.iter().all(|e| !e["event_name"].as_str().unwrap().starts_with("ignis.request.")), "never admitted");
}
