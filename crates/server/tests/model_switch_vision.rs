//! A model switch keeps vision (spec #555, GitHub #563), over the mock-backed
//! router (`support/switch.rs`): the request size cap follows the model
//! serving after a switch, whichever way it goes, and a switch to a model
//! whose vision companion is missing is refused `409` naming the file while
//! the active model serves on.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_artifact::vision::ProcessorOptions;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::model_switch::Switcher;
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

#[path = "support/media.rs"]
mod media;

const PATIENT: Duration = Duration::from_secs(30);

fn acquirer() -> Arc<MediaAcquirer> {
    let limits = ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    };
    Arc::new(MediaAcquirer::new(Arc::new(media::processor(limits.clone())), limits, MediaPolicy::new(false, 0)))
}

async fn send(app: &axum::Router, method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().method(method).uri(uri).header(header::CONTENT_TYPE, "application/json");
    let request = request.body(body.map_or_else(Body::empty, |body| Body::from(body.to_string()))).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn switch_and_settle(app: &axum::Router, model: &str) {
    let (status, accepted) = send(app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": format!("{model}.ninfer"), "model": model }))).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{accepted}");
    for _ in 0..10_000 {
        let (_, models) = send(app, Method::GET, "/v1/models", None).await;
        if models["status"] == "serving" && models["data"][0]["id"] == model {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("the switch to {model} never settled");
}

/// A chat whose one text part is `text_bytes` long, with one image part: a
/// body of that size, as a request with an image inline has.
async fn post_sized(app: &axum::Router, model: &str, text_bytes: usize) -> StatusCode {
    let image = media::data_uri(&media::png(64, 64));
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "x".repeat(text_bytes)},
            {"type": "image_url", "image_url": {"url": image}},
        ]}],
        "max_tokens": 1,
    });
    send(app, Method::POST, "/v1/chat/completions", Some(body)).await.0
}

/// Past the text-only cap, inside the media one.
const BETWEEN: usize = ignis_server::api::TEXT_REQUEST_BODY_LIMIT + (1 << 20);

#[tokio::test]
async fn the_request_size_cap_follows_the_model_serving_across_a_switch_both_ways() {
    let loader = MockLoader::new();
    // The 27B takes images and Flash-Next does too: two vision loads.
    loader.vision("mock-27b", acquirer());
    loader.vision("mock-fn", acquirer());
    let server = Server::from_active(loader.model("mock-27b")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    assert_ne!(post_sized(&app, "mock-27b", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "vision on at start");
    switch_and_settle(&app, "mock-fn").await;
    assert_ne!(post_sized(&app, "mock-fn", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "an image within the cap after the switch to Flash-Next");
    switch_and_settle(&app, "mock-27b").await;
    assert_ne!(post_sized(&app, "mock-27b", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "and back");
}

#[tokio::test]
async fn a_text_start_that_switches_to_a_vision_model_takes_its_cap_and_a_switch_back_returns_the_text_one() {
    let loader = MockLoader::new();
    loader.vision("mock-vision", acquirer());
    let server = Server::from_active(loader.model("mock-text")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    assert_eq!(post_sized(&app, "mock-text", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "a text model's own cap");
    switch_and_settle(&app, "mock-vision").await;
    assert_ne!(post_sized(&app, "mock-vision", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE, "the cap the vision model takes");
    switch_and_settle(&app, "mock-text").await;
    assert_eq!(post_sized(&app, "mock-text", BETWEEN).await, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn a_switch_to_a_model_with_its_companion_missing_is_refused_by_name_and_the_old_model_serves() {
    let loader = MockLoader::new();
    loader.vision("mock-27b", acquirer());
    loader.vision("mock-fn", acquirer());
    loader.without_companion("mock-fn");
    let server = Server::from_active(loader.model("mock-27b")).with_switcher(Switcher::new(Arc::clone(&loader) as _, PATIENT));
    let app = server.app();
    let (status, body) =
        send(&app, Method::POST, "/v1/models/switch", Some(json!({ "artifact": "mock-fn.ninfer", "model": "mock-fn" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "vision_companion_missing", "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("mock-fn.ninfer.vision"), "names the file: {message}");
    // Nothing was prepared, loaded or stopped: the 27B serves on, an image included.
    assert!(loader.loads().is_empty() && loader.prepared_artifacts().is_empty(), "{:?}", loader.loads());
    let (_, models) = send(&app, Method::GET, "/v1/models", None).await;
    assert_eq!((models["status"].as_str(), models["data"][0]["id"].as_str()), (Some("serving"), Some("mock-27b")), "{models}");
    assert_eq!(post_sized(&app, "mock-27b", 0).await, StatusCode::OK);
    // A switch to a model that lacks nothing still begins.
    switch_and_settle(&app, "mock-27b").await;
}
