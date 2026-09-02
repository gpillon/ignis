//! OpenAI-compatible HTTP surface (server-01): end-to-end tests that drive
//! the real axum router against a mock-compute engine (CPU-only, ADR 0006 —
//! the deterministic `MockCompute` stands in for the kernel leaf).
//!
//! Covers the three v1 endpoints + their error paths:
//! - `GET /v1/models` — the loaded model.
//! - `POST /v1/chat/completions` — non-streaming + streaming (SSE).
//! - `POST /v1/responses` — the OpenAI responses API (non-streaming).

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::template::{SimpleTemplateProvider, TemplateProvider};
use ignis_server::Server;

const MODEL: &str = "test-model";

/// A live harness: the real axum router over a mock-compute engine, with
/// the engine's driver loop running (so submitted requests actually
/// advance and route their events).
struct Harness {
    app: axum::Router,
    /// Keeps the driver task alive (it runs for the harness's life; the
    /// test's runtime drop cancels it cleanly).
    #[allow(dead_code)]
    driver: tokio::task::JoinHandle<()>,
}

fn harness() -> Harness {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let server = Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SimpleTemplateProvider),
    )
    .with_request_timeout(Duration::from_secs(5));
    // The engine's driver loop: the single task that advances the engine
    // and routes per-request events into the handlers' streams.
    let driver_engine = server.engine.clone();
    let driver = tokio::spawn(async move { driver_engine.run().await });
    Harness {
        app: server.app(),
        driver,
    }
}

/// The token the seed-0 mock emits for request `id` at decode step `i`
/// (the mock's deterministic contract — used to pin the exact content).
fn mock_tokens(id: u64, n: u32) -> Vec<u32> {
    let mock = MockCompute::new();
    (0..n).map(|i| mock.token_for(id, i)).collect()
}

/// Render a token stream exactly as the built-in template does (decimal
/// id per token, space-joined).
fn rendered(tokens: &[u32]) -> String {
    SimpleTemplateProvider.render_tokens(tokens)
}

/// Make one request against the router, returning (status, body-as-string).
async fn call(
    app: &axum::Router,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (u16, String) {
    let body_bytes = match body {
        Some(v) => v.to_string().into_bytes(),
        None => Vec::new(),
    };
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body_bytes))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

// ── GET /v1/models ────────────────────────────────────────────────────────

#[tokio::test]
async fn list_models_returns_the_loaded_model() {
    let h = harness();
    let (status, body) = call(&h.app, "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "models should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], MODEL, "the loaded model id");
    assert_eq!(v["data"][0]["object"], "model");
}

// ── POST /v1/chat/completions (non-streaming) ────────────────────────────

#[tokio::test]
async fn chat_completions_non_streaming_returns_the_completion() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": "hello world" },
            { "role": "assistant", "content": "hi" }
        ],
        "max_tokens": 4,
        "stream": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "chat should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], MODEL);
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // The exact token stream the mock produced for request 0 (4 steps) —
    // pins that tokens flowed scheduler → engine → template → HTTP.
    let expected = rendered(&mock_tokens(0, 4));
    assert_eq!(v["choices"][0]["message"]["content"], expected);
    assert_eq!(v["usage"]["completion_tokens"], 4);
}

// ── POST /v1/chat/completions (streaming / SSE) ──────────────────────────

#[tokio::test]
async fn chat_completions_streaming_emits_chunks_then_done() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 3,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "streaming chat should be 200: {body}");

    // Parse the SSE stream: the `data:` lines (JSON chunks, then [DONE]).
    let data_lines: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .collect();
    // The terminal marker is the last data line.
    assert_eq!(data_lines.last().map(|s| s.as_str()), Some("[DONE]"));
    let chunks: Vec<serde_json::Value> = data_lines
        .iter()
        .filter(|l| l.as_str() != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    // 3 token chunks + 1 final finish-reason chunk.
    assert_eq!(chunks.len(), 4, "3 tokens + final chunk: {body}");
    // The 3 token chunks carry the mock's exact token ids (request 0).
    let expected_tokens = mock_tokens(0, 3);
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        assert_eq!(chunk["choices"][0]["delta"]["content"], expected_tokens[i].to_string());
    }
    // The final chunk: finish_reason set, empty delta.
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
    // The token sequence, re-rendered, matches the built-in template.
    let streamed_content: String = chunks.iter().take(3)
        .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(streamed_content, rendered(&expected_tokens));
}

// ── POST /v1/responses ────────────────────────────────────────────────────

#[tokio::test]
async fn responses_api_string_input_returns_the_openai_shape() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "input": "tell me a joke",
        "max_output_tokens": 2
    });
    let (status, body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 200, "responses should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["output"][0]["type"], "message");
    assert_eq!(v["output"][0]["role"], "assistant");
    assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
    // The string input is a single user turn; the mock's 2 tokens (request 0).
    assert_eq!(v["usage"]["output_tokens"], 2);
    assert_eq!(v["output"][0]["content"][0]["text"], rendered(&mock_tokens(0, 2)));
}

#[tokio::test]
async fn responses_api_message_input_returns_the_openai_shape() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "input": [
            { "role": "user", "content": "hi" }
        ],
        "max_output_tokens": 1
    });
    let (status, body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 200, "responses (messages) should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(v["usage"]["output_tokens"], 1);
}

// ── error paths ───────────────────────────────────────────────────────────

#[tokio::test]
async fn an_unknown_model_is_a_404() {
    let h = harness();
    let req = serde_json::json!({
        "model": "no-such-model",
        "messages": [{ "role": "user", "content": "hi" }]
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 404, "unknown model should be 404: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "model_not_found");
}

#[tokio::test]
async fn empty_messages_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "messages": []
    });
    let (status, _body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "empty messages should be 400");
}

#[tokio::test]
async fn a_streaming_responses_request_is_a_400() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "input": "hi",
        "stream": true
    });
    let (status, _body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 400, "streaming responses are unsupported in v1: 400");
}