//! P5-06 (GitHub #154): a decode round that commits a run of tokens reaches
//! the OpenAI surface one token at a time.
//!
//! The compute is `MockCompute::with_runs` — the CPU stand-in for a
//! speculative leaf (ADR 0006) committing runs of 1..=k+1 tokens per round.
//! Streaming emits one delta per committed token, in order, and `usage`
//! counts committed tokens, exactly as a round of one does.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::mock::MockCompute;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::engine::Engine;
use ignis_server::template::{SimpleTemplateProvider, TemplateProvider};
use ignis_server::Server;

const MODEL: &str = "test-model";

fn app(compute: Arc<MockCompute>) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider)).app()
}

async fn chat(app: &axum::Router, body: serde_json::Value) -> (u16, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string().into_bytes()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// The JSON chunks of an SSE body, without the `[DONE]` marker.
fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:").map(str::trim))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test]
async fn a_streaming_completion_over_runs_emits_one_delta_per_committed_token_in_order() {
    // k = 7: rounds commit 1, 8, 3 and then 1 of a run of 5, cut by the cap.
    let compute = Arc::new(MockCompute::with_runs(&[1, 8, 3, 5]));
    let app = app(compute.clone());
    let (status, body) = chat(
        &app,
        serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 13,
            "stream": true,
            "stream_options": { "include_usage": true }
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    let chunks = sse_chunks(&body);
    // 13 token deltas + the finish-reason chunk + the usage chunk.
    assert_eq!(chunks.len(), 15, "{body}");

    let expected: Vec<u32> = (0..13).map(|step| compute.token_for(0, step)).collect();
    for (i, chunk) in chunks.iter().take(13).enumerate() {
        let text = if i == 0 {
            expected[i].to_string()
        } else {
            format!(" {}", expected[i])
        };
        assert_eq!(chunk["choices"][0]["delta"]["content"], text, "delta {i}: {body}");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }
    assert_eq!(chunks[13]["choices"][0]["finish_reason"], "length");
    assert_eq!(chunks[14]["usage"]["completion_tokens"], 13, "usage counts committed tokens");
    assert_eq!(compute.decode_calls().len(), 4, "four rounds carried thirteen tokens");
}

#[tokio::test]
async fn a_non_streaming_completion_over_runs_returns_every_committed_token() {
    let compute = Arc::new(MockCompute::with_runs(&[4]));
    let app = app(compute.clone());
    let (status, body) = chat(
        &app,
        serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    // Runs of 4 against a 64-token cap: sixteen rounds, every token once.
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let expected: Vec<u32> = (0..64).map(|step| compute.token_for(0, step)).collect();
    assert_eq!(
        v["choices"][0]["message"]["content"],
        SimpleTemplateProvider.render_tokens(&expected)
    );
    assert_eq!(v["usage"]["completion_tokens"], 64);
    assert_eq!(compute.decode_calls().len(), 16);
}

#[tokio::test]
async fn a_run_cut_at_eos_streams_the_tokens_before_it_and_finishes_with_stop() {
    let compute = Arc::new(MockCompute::with_runs(&[4]));
    compute.eos_after(0, 6);
    let app = app(compute.clone());
    let (status, body) = chat(
        &app,
        serde_json::json!({
            "model": MODEL,
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 64,
            "stream": true,
            "stream_options": { "include_usage": true }
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    let chunks = sse_chunks(&body);
    assert_eq!(chunks.len(), 8, "6 token deltas + finish + usage: {body}");
    let streamed: String = chunks
        .iter()
        .take(6)
        .map(|c| c["choices"][0]["delta"]["content"].as_str().unwrap().to_string())
        .collect();
    let expected: Vec<u32> = (0..6).map(|step| compute.token_for(0, step)).collect();
    assert_eq!(streamed, SimpleTemplateProvider.render_tokens(&expected));
    assert_eq!(chunks[6]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[7]["usage"]["completion_tokens"], 6);
}
