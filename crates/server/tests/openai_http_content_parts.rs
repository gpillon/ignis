//! OpenAI content parts on chat messages (GitHub #175, spec
//! `docs/specs/vision/01-image-input.md` §Wire contract): end-to-end
//! tests over the real axum router against a mock-compute engine (CPU-only,
//! ADR 0006).
//!
//! A message's `content` is a string or an array of parts. `text` parts
//! render exactly as the string they join to (the reference's `"\n"`
//! between adjacent text parts); media parts are
//! recognised but not served yet, each refused with a 400 carrying the
//! reference's error code and naming the offending message/part index,
//! before the request reaches the engine.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

const MODEL: &str = "test-model";

/// The real router over a fresh mock-compute engine. Each call builds a
/// new engine, so its first request is request id 0 — two harnesses given
/// equivalent requests produce identical responses.
fn app() -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        Arc::new(MockCompute::new()),
    );
    Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SimpleTemplateProvider),
    )
    .with_request_timeout(Duration::from_secs(5))
    .app()
}

/// POST `body` to `path`, returning (status, parsed JSON body).
async fn post(app: &axum::Router, path: &str, body: Value) -> (u16, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let value = serde_json::from_str(&text)
        .unwrap_or_else(|err| panic!("body is not JSON ({err}): {text}"));
    (status, value)
}

async fn chat(messages: Value) -> (u16, Value) {
    post(
        &app(),
        "/v1/chat/completions",
        json!({ "model": MODEL, "messages": messages, "max_tokens": 4 }),
    )
    .await
}

/// Assert a 400 with `code` whose message names every fragment in `names`.
fn assert_refused(status: u16, body: &Value, code: &str, names: &[&str]) {
    assert_eq!(status, 400, "expected a 400: {body}");
    assert_eq!(body["error"]["code"], code, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    for name in names {
        assert!(message.contains(name), "message should name {name:?}: {message}");
    }
}

// ── text parts render as the string form ────────────────────────────────

#[tokio::test]
async fn text_parts_produce_the_same_response_as_the_joined_string() {
    let (string_status, string_body) = chat(json!([
        { "role": "user", "content": "hello world\nagain" }
    ]))
    .await;
    let (parts_status, parts_body) = chat(json!([
        { "role": "user", "content": [
            { "type": "text", "text": "hello world" },
            { "type": "text", "text": "again" }
        ] }
    ]))
    .await;
    assert_eq!(string_status, 200, "{string_body}");
    assert_eq!(parts_status, 200, "{parts_body}");
    assert_eq!(parts_body["usage"], string_body["usage"]);
    assert_eq!(parts_body["usage"]["prompt_tokens"], 3, "{parts_body}");
    assert_eq!(
        parts_body["choices"][0]["message"]["content"],
        string_body["choices"][0]["message"]["content"]
    );
}

#[tokio::test]
async fn text_parts_are_accepted_on_user_assistant_and_tool_messages() {
    let text = |t: &str| json!([{ "type": "text", "text": t }]);
    let (status, body) = chat(json!([
        { "role": "system", "content": text("be brief") },
        { "role": "user", "content": text("call it") },
        { "role": "assistant", "content": text("calling"), "tool_calls": [
            { "id": "c1", "type": "function", "function": { "name": "f", "arguments": "{}" } }
        ] },
        { "role": "tool", "tool_call_id": "c1", "content": text("result") },
        { "role": "user", "content": "thanks" }
    ]))
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["prompt_tokens"], 7, "{body}");
}

#[tokio::test]
async fn the_responses_api_accepts_text_parts_too() {
    let (status, body) = post(
        &app(),
        "/v1/responses",
        json!({
            "model": MODEL,
            "max_output_tokens": 2,
            "input": [{ "role": "user", "content": [{ "type": "text", "text": "hi there" }] }]
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["usage"]["input_tokens"], 2, "{body}");
}

// ── media refusals ───────────────────────────────────────────────────────

#[tokio::test]
async fn an_image_url_part_is_refused_as_vision_disabled() {
    let (status, body) = chat(json!([
        { "role": "user", "content": "look" },
        { "role": "user", "content": [
            { "type": "text", "text": "what is this?" },
            { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0KGgo=" } }
        ] }
    ]))
    .await;
    assert_refused(status, &body, "vision_disabled", &["message 1", "part 1"]);
}

#[tokio::test]
async fn the_image_url_detail_field_is_accepted_and_ignored() {
    // Accepted means parsed: the refusal is the vision one, not a shape error.
    let (status, body) = chat(json!([
        { "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png", "detail": "high" } }
        ] }
    ]))
    .await;
    assert_refused(status, &body, "vision_disabled", &["message 0", "part 0"]);
}

#[tokio::test]
async fn image_parts_are_recognised_on_assistant_and_tool_messages() {
    let image = json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.png" } });
    for role in ["assistant", "tool"] {
        let mut message = json!({ "role": role, "content": [image.clone()] });
        if role == "tool" {
            message["tool_call_id"] = json!("c1");
        }
        let (status, body) = chat(json!([{ "role": "user", "content": "hi" }, message])).await;
        assert_refused(status, &body, "vision_disabled", &["message 1", "part 0"]);
    }
}

#[tokio::test]
async fn a_video_url_part_is_refused_as_video_unsupported() {
    let (status, body) = chat(json!([
        { "role": "user", "content": [
            { "type": "text", "text": "watch" },
            { "type": "video_url", "video_url": { "url": "https://example.com/a.mp4" } }
        ] }
    ]))
    .await;
    assert_refused(status, &body, "video_unsupported", &["message 0", "part 1"]);
}

#[tokio::test]
async fn media_in_a_system_message_is_refused() {
    for part in [
        json!({ "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }),
        json!({ "type": "video_url", "video_url": { "url": "https://example.com/a.mp4" } }),
    ] {
        let (status, body) = chat(json!([
            { "role": "system", "content": [{ "type": "text", "text": "rules" }, part] },
            { "role": "user", "content": "hi" }
        ]))
        .await;
        assert_refused(status, &body, "invalid_media", &["message 0", "part 1", "system"]);
    }
}

#[tokio::test]
async fn an_unknown_part_type_is_refused_naming_it() {
    let (status, body) = chat(json!([
        { "role": "user", "content": [{ "type": "input_audio", "input_audio": { "data": "" } }] }
    ]))
    .await;
    assert_refused(
        status,
        &body,
        "modality_not_supported",
        &["message 0", "part 0", "input_audio"],
    );
}

#[tokio::test]
async fn a_media_refusal_wins_over_an_earlier_unknown_part_type() {
    let (status, body) = chat(json!([
        { "role": "user", "content": [{ "type": "input_audio", "input_audio": { "data": "" } }] },
        { "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
        ] }
    ]))
    .await;
    assert_refused(status, &body, "vision_disabled", &["message 1", "part 0"]);
}

#[tokio::test]
async fn malformed_parts_are_refused_naming_the_part() {
    let cases = [
        (json!([{ "text": "no type" }]), "part 0"),
        (json!([{ "type": "text" }]), "part 0"),
        (json!([{ "type": "text", "text": "ok" }, { "type": "text", "text": 7 }]), "part 1"),
        (json!([{ "type": "image_url", "image_url": {} }]), "part 0"),
        (json!([{ "type": "image_url" }]), "part 0"),
        (json!(["bare string"]), "part 0"),
    ];
    for (content, part) in cases {
        let (status, body) = chat(json!([{ "role": "user", "content": content }])).await;
        assert_refused(status, &body, "invalid_request_error", &["message 0", part]);
    }
}

#[tokio::test]
async fn a_malformed_part_anywhere_wins_over_an_earlier_media_refusal() {
    let (status, body) = chat(json!([
        { "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
        ] },
        { "role": "user", "content": [{ "type": "text" }] }
    ]))
    .await;
    assert_refused(status, &body, "invalid_request_error", &["message 1", "part 0"]);
}

#[tokio::test]
async fn an_empty_parts_array_is_refused() {
    let (status, body) = chat(json!([{ "role": "user", "content": [] }])).await;
    assert_refused(status, &body, "invalid_request_error", &["message 0"]);
}

#[tokio::test]
async fn the_responses_api_refuses_media_parts_the_same_way() {
    let (status, body) = post(
        &app(),
        "/v1/responses",
        json!({
            "model": MODEL,
            "input": [{ "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": "https://example.com/a.png" } }
            ] }]
        }),
    )
    .await;
    assert_refused(status, &body, "vision_disabled", &["message 0", "part 0"]);
}
