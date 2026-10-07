//! ADR 0045, spec vram-budget/03 AC 11 (GitHub #309), at the HTTP surface: a
//! request that sends no cap -- neither `max_tokens` nor
//! `max_completion_tokens` on chat, no `max_output_tokens` on
//! `/v1/responses` -- generates at most the server's default `max_tokens` and
//! ends as a request that reached its own cap does. The handlers do not
//! change: the scheduler resolves the cap (`SchedulerConfig::default_max_tokens`),
//! so these tests only vary that.
//!
//! Driven through the in-process router over `MockCompute`.

#[path = "support/responses.rs"]
mod responses;

use std::collections::HashMap;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{MockCompute, SchedulerConfig};
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use responses::{server, Script, MODEL};

const CHAT: &str = "/v1/chat/completions";
const RESPONSES: &str = "/v1/responses";
/// A whole trained context, so the default, not the context, is what binds.
const CONTEXT: u32 = 262_144;

/// A server whose scheduler runs under `default_max_tokens`, with speculative
/// runs of eight so a long generation is quick.
fn app(default_max_tokens: u32, max_sequence_tokens: u32) -> axum::Router {
    let config = SchedulerConfig {
        kv_page_tokens: 64,
        max_sequence_tokens,
        kv_capacity_pages: 8 * max_sequence_tokens.div_ceil(64),
        default_max_tokens,
        ..SchedulerConfig::default()
    };
    server(&Script::new(HashMap::new()), config, Arc::new(MockCompute::with_runs(&[8]))).app()
}

async fn post(app: &axum::Router, path: &str, body: &JsonValue) -> (u16, JsonValue) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn chat(extra: JsonValue) -> JsonValue {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "enable_thinking": false
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test]
async fn a_chat_request_without_a_cap_stops_at_the_default_with_length() {
    for cap in [38_912u32, 8_192] {
        let (status, answer) = post(&app(cap, CONTEXT), CHAT, &chat(json!({}))).await;
        assert_eq!(status, 200, "{answer}");
        assert_eq!(answer["choices"][0]["finish_reason"], "length", "default {cap}");
        assert_eq!(answer["usage"]["completion_tokens"], cap, "default {cap}");
    }
}

#[tokio::test]
async fn an_explicit_cap_wins_under_either_name() {
    for field in ["max_tokens", "max_completion_tokens"] {
        let (status, answer) = post(&app(8_192, CONTEXT), CHAT, &chat(json!({ field: 9_000 }))).await;
        assert_eq!(status, 200, "{answer}");
        assert_eq!(answer["usage"]["completion_tokens"], 9_000, "{field}, larger than the default");
        let (_, answer) = post(&app(8_192, CONTEXT), CHAT, &chat(json!({ field: 5 }))).await;
        assert_eq!(answer["usage"]["completion_tokens"], 5, "{field}, smaller");
    }
}

#[tokio::test]
async fn a_responses_request_without_a_cap_ends_incomplete_at_the_default() {
    let body = json!({ "model": MODEL, "input": "hi", "enable_thinking": false });
    let (status, answer) = post(&app(8_192, CONTEXT), RESPONSES, &body).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["status"], "incomplete", "{answer}");
    assert_eq!(answer["incomplete_details"], json!({ "reason": "max_output_tokens" }));
    assert_eq!(answer["usage"]["output_tokens"], 8_192);
    assert_eq!(answer["max_output_tokens"], JsonValue::Null, "it echoes what the client sent");
}

#[tokio::test]
async fn zero_runs_to_the_context_as_before() {
    let (status, answer) = post(&app(0, 4_096), CHAT, &chat(json!({}))).await;
    assert_eq!(status, 200, "{answer}");
    let prompt = answer["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(answer["usage"]["completion_tokens"].as_u64().unwrap(), 4_096 - prompt);
    assert_eq!(answer["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn ignore_eos_still_needs_an_explicit_max_tokens() {
    let (status, answer) = post(&app(38_912, CONTEXT), CHAT, &chat(json!({ "ignore_eos": true }))).await;
    assert_eq!(status, 400, "{answer}");
    assert!(answer["error"]["message"].as_str().unwrap().contains("max_tokens"), "{answer}");
}

#[tokio::test]
async fn an_explicit_cap_past_the_context_is_still_refused_naming_its_field() {
    let (status, answer) = post(&app(8_192, 4_096), CHAT, &chat(json!({ "max_tokens": 5_000 }))).await;
    assert_eq!(status, 400, "{answer}");
    let text = answer.to_string();
    assert!(text.contains("context_length_exceeded") && text.contains("max_tokens"), "{answer}");
}
