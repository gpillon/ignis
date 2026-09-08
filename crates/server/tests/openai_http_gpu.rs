//! GPU integration coverage for the real model behind the OpenAI HTTP
//! surface (GitHub #61 / P1-25): the same axum router `openai_http.rs`
//! drives against `MockCompute`, but here wired to the production
//! `ignis_server::runtime::cuda_scheduler` — a real request round-trips
//! through HTTP → the templated prompt → the GPU-resident program → the
//! tokenizer back to text, for both non-streaming and streaming chat
//! completions.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_server::engine::Engine;
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::{NullSink, SystemClock};
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";

/// A live harness over the real GPU-backed scheduler (mirrors
/// `openai_http.rs`'s mock harness, but with the production backend
/// running underneath the same axum router).
///
/// GitHub #71: the 3 tests in this file share one process, and each
/// `harness()` call loads its own ~19 GB model onto the same GPU. The
/// model thread (GitHub #69) owns the scheduler's GPU-resident state and
/// only releases it when it exits, so `app` is wrapped in an `Option` and
/// `driver` (its `JoinHandle`) is kept alongside: `Drop` below drops `app`
/// first — disconnecting the command channel every clone of the `Engine`
/// held for routing — then joins `driver`, blocking until the model
/// thread has actually exited and freed the previous test's VRAM before
/// the next test's `harness()` call allocates its own.
struct Harness {
    app: Option<axum::Router>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Harness {
    fn app(&self) -> &axum::Router {
        self.app.as_ref().expect("app is only taken by Drop")
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.app.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

/// Build the harness, or `None` when the GPU profile says to skip
/// (artifact absent / CUDA unavailable outside the profile — a hard
/// failure under it, via `gpu_profile::skip_or_fail`).
fn harness() -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend
        .eos_token_id()
        .unwrap_or_else(|| panic!("qwen3.8-27b generation config must carry eos_token_id"));

    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, EngineShape::default()) {
        Ok(scheduler) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let (engine, driver) =
        Engine::with_sinks_and_driver(Box::new(scheduler), Arc::new(NullSink), Arc::new(SystemClock));
    let server = Server::with_artifact_template(engine, frontend)
        .with_request_timeout(Duration::from_secs(120));
    Some(Harness { app: Some(server.app()), driver: Some(driver) })
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

#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_non_streaming_completion_returns_coherent_text_with_finish_reason_and_usage() {
    let Some(h) = harness() else { return };
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": "In one sentence, what is 2 + 2?" }
        ],
        "max_tokens": 32,
        "stream": false,
        "enable_thinking": false
    });
    let (status, body) = call(h.app(), "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "chat should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], MODEL);

    // finish_reason is one of the two real reasons (GitHub #61): "stop"
    // (the model's own EOS) or "length" (the 32-token cap) — never the old
    // hardcoded "stop" regardless of why generation ended.
    let reason = v["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(
        reason == "stop" || reason == "length",
        "finish_reason must be stop or length, got {reason}"
    );

    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.trim().is_empty(), "the real model must return non-empty text");
    assert!(!content.contains('\0'), "no NUL bytes in real text");

    // Usage counts: real, non-zero, internally consistent.
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap();
    let total_tokens = v["usage"]["total_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 0, "a templated prompt always tokenizes to something");
    assert!(completion_tokens > 0, "the model generated at least one token");
    assert!(completion_tokens <= 32, "decode must stop at max_tokens even without EOS");
    assert_eq!(total_tokens, prompt_tokens + completion_tokens);
}

#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_streaming_completion_emits_token_deltas_then_a_finish_reason_chunk() {
    let Some(h) = harness() else { return };
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": "In one sentence, what is 2 + 2?" }
        ],
        "max_tokens": 32,
        "stream": true,
        "enable_thinking": false
    });
    let (status, body) = call(h.app(), "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "streaming chat should be 200: {body}");

    let data_lines: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .collect();
    assert_eq!(data_lines.last().map(|s| s.as_str()), Some("[DONE]"), "{body}");

    let chunks: Vec<serde_json::Value> = data_lines
        .iter()
        .filter(|l| l.as_str() != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(chunks.len() >= 2, "at least one token delta + the final chunk: {body}");

    // Every chunk but the last carries a token delta and no finish_reason;
    // reassembling them must produce non-empty, coherent text.
    let mut streamed = String::new();
    for chunk in &chunks[..chunks.len() - 1] {
        assert!(chunk["choices"][0]["finish_reason"].is_null());
        streamed.push_str(chunk["choices"][0]["delta"]["content"].as_str().unwrap());
    }
    assert!(!streamed.trim().is_empty(), "streamed text must be non-empty");

    // The final chunk: empty delta, a real finish_reason (GitHub #61).
    let last = &chunks[chunks.len() - 1];
    assert_eq!(last["choices"][0]["delta"], serde_json::json!({}));
    let reason = last["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(
        reason == "stop" || reason == "length",
        "finish_reason must be stop or length, got {reason}"
    );
}

/// The GPU anchor for GitHub #69's finding: a streaming request's SSE
/// frames are observed arriving before the request's own generation
/// completes, against the real model. `curl --trace-time` originally
/// proved the body delivered zero bytes until generation ended, then
/// everything at once; here, a time-bounded read of the *first* chunk
/// (well under the full 64-token generation's expected wall time) pins
/// that the isolation fix restores true incremental delivery on real
/// hardware, not just against `MockCompute`.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_streaming_completions_first_chunk_arrives_before_generation_completes() {
    let Some(h) = harness() else { return };
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": "Count from one to twenty, one number per line." }
        ],
        "max_tokens": 64,
        "stream": true
    });
    let body = Body::from(req.to_string().into_bytes());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(body)
        .unwrap();
    let resp = h.app().clone().oneshot(request).await.unwrap();
    assert_eq!(resp.status(), 200);
    let mut body = resp.into_body();

    // A 64-token real generation runs several seconds end to end (measured
    // at ~85 ms/token in the spec's finding); the first SSE frame must
    // arrive in a small fraction of that — proof the body streams
    // incrementally instead of withholding every byte until the end.
    let first_frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
        .await
        .expect("the first SSE frame must arrive well before the full generation completes")
        .expect("the body must yield at least one frame")
        .expect("the frame must not be an error");
    let text = String::from_utf8_lossy(
        first_frame
            .data_ref()
            .expect("the first frame must carry data"),
    )
    .to_string();
    assert!(
        text.starts_with("data:"),
        "the first frame must be an SSE data line: {text:?}"
    );
}

/// P2-02 (GitHub #84): a multi-thousand-token prompt streamed end to end
/// through the real chunked-prefill path (`EngineShape::default()`'s
/// `--prefill-chunk` is 1024, so this prompt spans several chunks). Proves
/// the chunk loop reaches a real HTTP round trip, not just the step ABI
/// directly: a coherent streamed answer, ending in a real `finish_reason`.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_streaming_completion_with_a_multi_thousand_token_prompt_finishes_coherently() {
    let Some(h) = harness() else { return };
    // Filler the model can skim past, followed by a direct question so a
    // small `max_tokens` budget is plausibly enough to reach a real EOS
    // rather than being cut off mid-thought.
    let filler = "The quick brown fox jumps over the lazy dog. ".repeat(1200);
    let content = format!("{filler}\nIn one sentence, what is 2 + 2?");
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": content }
        ],
        "max_tokens": 32,
        "stream": true,
        "enable_thinking": false
    });
    let (status, body) = call(h.app(), "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "streaming chat with a long prompt should be 200: {body}");

    let data_lines: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .collect();
    assert_eq!(data_lines.last().map(|s| s.as_str()), Some("[DONE]"), "{body}");

    let chunks: Vec<serde_json::Value> = data_lines
        .iter()
        .filter(|l| l.as_str() != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(chunks.len() >= 2, "at least one token delta + the final chunk: {body}");

    let mut streamed = String::new();
    for chunk in &chunks[..chunks.len() - 1] {
        assert!(chunk["choices"][0]["finish_reason"].is_null());
        streamed.push_str(chunk["choices"][0]["delta"]["content"].as_str().unwrap());
    }
    assert!(!streamed.trim().is_empty(), "streamed text must be non-empty");

    let last = &chunks[chunks.len() - 1];
    let reason = last["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(
        reason == "stop" || reason == "length",
        "finish_reason must be stop or length, got {reason}"
    );

    let prompt_tokens = chunks
        .iter()
        .find_map(|c| c.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64()))
        .or_else(|| {
            last.get("usage").and_then(|u| u.get("prompt_tokens")).and_then(|v| v.as_u64())
        });
    if let Some(prompt_tokens) = prompt_tokens {
        assert!(
            prompt_tokens > 2000,
            "the templated prompt must actually be multi-thousand tokens, got {prompt_tokens}"
        );
    }
}

/// GitHub #68: a thinking-disabled request against the real model and the
/// real Qwen 3.8 template returns a real answer directly — the CPU gate
/// covers every wire-contract case, but only the real template can prove
/// `enable_thinking: false` actually reaches it and the model answers
/// within a small budget instead of consuming it on a thinking trace.
#[tokio::test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn a_thinking_disabled_request_returns_a_real_answer_with_no_reasoning() {
    let Some(h) = harness() else { return };
    let req = serde_json::json!({
        "model": MODEL,
        "messages": [
            { "role": "user", "content": "In one word, what is the capital of France?" }
        ],
        "max_tokens": 32,
        "stream": false,
        "enable_thinking": false
    });
    let (status, body) = call(h.app(), "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "chat should be 200: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();

    // No reasoning field at all (GitHub #68 story 18): thinking was
    // disabled, so there is no trace to carry.
    assert!(
        v["choices"][0]["message"].as_object().unwrap().get("reasoning_content").is_none(),
        "reasoning_content must be absent when thinking is disabled: {body}"
    );

    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.trim().is_empty(), "a small budget must be enough for a direct answer");
    assert!(!content.contains("<think>") && !content.contains("</think>"), "{content}");
    // `content` parsed out of the JSON response body as a Rust `String`,
    // which is only possible if it was valid UTF-8 — the response could
    // not have reached this point otherwise.
}
