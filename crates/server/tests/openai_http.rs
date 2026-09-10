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

use ignis_core::mock::{GateController, GatedCompute, MockCompute};
use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, PrefillJob, RequestId,
    SchedulerConfig,
};
use ignis_server::engine::Engine;
use ignis_server::template::{SimpleTemplateProvider, TemplateProvider};
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::nudge;

const MODEL: &str = "test-model";

/// A live harness: the real axum router over a mock-compute engine. The
/// engine's model thread (GitHub #69) was already spawned when it was
/// constructed — nothing else to start here.
struct Harness {
    app: axum::Router,
}

/// Build a harness over `compute` (shared by [`harness`] and
/// [`harness_gated`] — they differ only in which `Compute` backs the
/// scheduler).
fn harness_over(compute: Arc<dyn Compute>) -> Harness {
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
    Harness { app: server.app() }
}

fn harness() -> Harness {
    harness_over(Arc::new(MockCompute::new()))
}

/// A harness whose compute is gated (GitHub #69): lets a test hold one
/// decode step open deterministically, to prove an unrelated concurrent
/// request is never held up by it (ADR 0006 — no sleeps).
fn harness_gated() -> (Harness, Arc<GatedCompute>, GateController) {
    let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
    let h = harness_over(gated.clone() as Arc<dyn Compute>);
    (h, gated, controller)
}

/// A compute decorator that exposes the public lifecycle release callback so
/// an HTTP test can wait deterministically for cancellation to free the first
/// request before submitting the next one.
struct ReleaseObservedCompute {
    inner: MockCompute,
    released: std::sync::mpsc::SyncSender<RequestId>,
}

impl Compute for ReleaseObservedCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<(), ComputeError> {
        self.inner.prefill_step(jobs)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
        let _ = self.released.send(request);
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
    // The mock has no real EOS token, so hitting the request's `max_tokens`
    // reports `length` (not `stop`) — GitHub #61 / P1-25's finish_reason.
    assert_eq!(v["choices"][0]["finish_reason"], "length");
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
    // 3 token chunks + 1 final finish-reason chunk. No `stream_options` was
    // sent, so no trailing usage chunk (OpenAI only sends it opt-in).
    assert_eq!(chunks.len(), 4, "3 tokens + final chunk: {body}");
    // The 3 token chunks carry the mock's exact token ids (request 0). The
    // incremental decoder (GitHub #68) reproduces the whole-list render's
    // space-joined shape by construction: the first chunk is bare, every
    // later one carries its leading separator space.
    let expected_tokens = mock_tokens(0, 3);
    for (i, chunk) in chunks.iter().take(3).enumerate() {
        let expected = if i == 0 {
            expected_tokens[i].to_string()
        } else {
            format!(" {}", expected_tokens[i])
        };
        assert_eq!(chunk["choices"][0]["delta"]["content"], expected);
    }
    // The final chunk: finish_reason set, empty delta. The mock has no
    // real EOS, so hitting `max_tokens` reports `length` (GitHub #61).
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "length");
    // The token sequence, re-rendered, matches the built-in template —
    // plain concatenation now that each delta carries its own separator
    // (story 24: streaming and non-streaming agree by construction).
    let streamed_content: String = chunks.iter().take(3)
        .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(streamed_content, rendered(&expected_tokens));
}

#[tokio::test]
async fn chat_completions_streaming_with_include_usage_appends_a_usage_chunk() {
    let h = harness();
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": 3,
        "stream": true,
        "stream_options": { "include_usage": true }
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "streaming chat should be 200: {body}");

    let data_lines: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .collect();
    assert_eq!(data_lines.last().map(|s| s.as_str()), Some("[DONE]"));
    let chunks: Vec<serde_json::Value> = data_lines
        .iter()
        .filter(|l| l.as_str() != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    // 3 token chunks + 1 final finish-reason chunk + 1 trailing usage chunk.
    assert_eq!(chunks.len(), 5, "3 tokens + final chunk + usage chunk: {body}");
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "length");
    // The trailing usage chunk: empty choices, populated usage — sent
    // right before `[DONE]` (OpenAI's summary chunk, opt-in via
    // `stream_options.include_usage`).
    assert_eq!(chunks[4]["choices"], serde_json::json!([]));
    assert_eq!(chunks[4]["usage"]["completion_tokens"], 3);
    assert_eq!(
        chunks[4]["usage"]["total_tokens"],
        chunks[4]["usage"]["prompt_tokens"].as_u64().unwrap() + 3
    );
}

#[tokio::test]
async fn dropping_a_streaming_response_cancels_the_request_and_releases_its_slot() {
    let (released_tx, released_rx) = std::sync::mpsc::sync_channel(1);
    let observed = Arc::new(ReleaseObservedCompute {
        inner: MockCompute::new(),
        released: released_tx,
    });
    let (gated, controller) = GatedCompute::new(observed as Arc<dyn Compute>);
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            max_in_flight: 1,
            ..SchedulerConfig::default()
        },
        gated.clone() as Arc<dyn Compute>,
    );
    let server = Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SimpleTemplateProvider),
    );
    let app = server.app();

    gated.arm();
    let first = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": MODEL,
                "messages": [{ "role": "user", "content": "keep decoding" }],
                "max_tokens": 8192,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();
    let first_response = app.clone().oneshot(first).await.unwrap();
    assert_eq!(first_response.status(), 200);
    let first_body = first_response.into_body();
    controller.wait_entered();
    // If disconnect does not cancel the request, its next decode step will
    // enter this second gate and it cannot reach lifecycle release.
    gated.arm();

    // Dropping the live HTTP body is the transport-level cancellation signal.
    drop(first_body);
    controller.release();
    released_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("disconnecting the SSE client must release its scheduler request");

    // With max_in_flight=1, a successful second submission proves the first
    // request no longer occupies the only in-flight slot.
    let second = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": MODEL,
                "messages": [{ "role": "user", "content": "replacement" }],
                "max_tokens": 1,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();
    let second_response = app.clone().oneshot(second).await.unwrap();
    assert_eq!(second_response.status(), 200);
    drop(second_response.into_body());
    controller.wait_entered();
    controller.release();
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

// ── concurrency (GitHub #69: the isolated model thread) ─────────────────

/// The HTTP-level regression for GitHub #69: a concurrent streaming request
/// must not be serialized behind another request's in-flight generation.
/// One request is held mid-decode via the gated compute; a second,
/// unrelated streaming request is submitted while it is held and must
/// complete without waiting for the held request's generation to finish.
#[tokio::test]
async fn a_concurrent_stream_is_not_serialized_behind_a_held_generation() {
    let (h, gated, controller) = harness_gated();

    // Arm the gate before anything is submitted: the very first decode
    // step (request A's) blocks until released.
    gated.arm();

    let app_a = h.app.clone();
    let task_a = tokio::spawn(async move {
        let req = serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 3,
            "stream": true
        });
        call(&app_a, "POST", "/v1/chat/completions", Some(req)).await
    });
    // Let A's handler run up to `engine.submit(..).await`, which the model
    // thread will drain and then advance into the gated decode step.
    nudge().await;
    controller.wait_entered();

    // B is submitted while A is confirmed stuck mid-decode.
    let app_b = h.app.clone();
    let task_b = tokio::spawn(async move {
        let req = serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 2,
            "stream": true
        });
        call(&app_b, "POST", "/v1/chat/completions", Some(req)).await
    });
    // Let B's submit command actually reach the model thread's queue.
    nudge().await;

    controller.release();

    // B completes without ever needing A's generation to finish first — the
    // old shared-mutex design could not have gotten here without releasing
    // A all the way to completion.
    let (status_b, body_b) = task_b.await.expect("task B must not panic");
    assert_eq!(status_b, 200, "B should be 200: {body_b}");
    let done_b = body_b
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim()))
        .next_back();
    assert_eq!(done_b, Some("[DONE]"), "B's stream must complete: {body_b}");

    // A is not starved either — releasing the gate lets it run to
    // completion normally.
    let (status_a, body_a) = task_a.await.expect("task A must not panic");
    assert_eq!(status_a, 200, "A should be 200: {body_a}");
    let done_a = body_a
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim()))
        .next_back();
    assert_eq!(done_a, Some("[DONE]"), "A's stream must complete: {body_a}");
}
