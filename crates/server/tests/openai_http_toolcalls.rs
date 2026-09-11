//! Tool-call stream hardening (GitHub #121): end-to-end tests over the real
//! axum router, on the CPU gate (ADR 0006 — `MockCompute`, no GPU).
//!
//! Mirrors `openai_http_thinking.rs`'s harness shape (prior art): a
//! recording [`TemplateProvider`] double maps specific mock-emitted token
//! ids to specific literal text, so a test can drive a
//! `<tool_call>...</tool_call>` block — including one split across several
//! scheduler tokens — through the whole router and assert on the resulting
//! response / SSE frames.
//!
//! `MockCompute` has no real EOS token (`crates/core/src/mock.rs`): every
//! request here finishes via `max_tokens`, i.e. `FinishReason::Length`,
//! never `Stop`. The `finish_reason: "tool_calls"` enhancement
//! (`resolve_finish_reason`, only reachable from `Stop`) is therefore
//! pinned by `api.rs`'s own unit tests, not here; what these tests exercise
//! is reassembly correctness and the interrupted-call drop rule, both of
//! which are independent of which finish reason ends the stream.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";

/// See `openai_http_thinking.rs`'s `RecordingTemplateProvider` (same
/// shape): templating delegates to the built-in placeholder, and specific
/// token ids decode to specific literal text.
struct RecordingTemplateProvider {
    inner: ignis_server::template::SimpleTemplateProvider,
    decode: HashMap<TokenId, &'static str>,
}

impl RecordingTemplateProvider {
    fn permissive(decode: HashMap<TokenId, &'static str>) -> Self {
        Self {
            inner: ignis_server::template::SimpleTemplateProvider,
            decode,
        }
    }
}

impl TemplateProvider for RecordingTemplateProvider {
    fn apply_chat_template(&self, messages: &[ChatMessage], options: &ThinkingOptions) -> Vec<TokenId> {
        self.inner.apply_chat_template(messages, options)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|t| self.decode.get(t).copied().unwrap_or("?").to_string())
            .collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(RecordingTokenDecoder {
            decode: self.decode.clone(),
        })
    }
    // `decoder_starts_in_reasoning` deliberately left at the trait default
    // (`thinking.enable_thinking`), same as `openai_http_thinking.rs`.
}

struct RecordingTokenDecoder {
    decode: HashMap<TokenId, &'static str>,
}

impl TokenDecoder for RecordingTokenDecoder {
    fn push(&mut self, token: TokenId) -> String {
        self.decode.get(&token).copied().unwrap_or("?").to_string()
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

struct Harness {
    app: axum::Router,
}

fn harness_with(template: RecordingTemplateProvider) -> Harness {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(template))
        .with_request_timeout(Duration::from_secs(5));
    Harness { app: server.app() }
}

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

fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .filter(|l| l != "[DONE]")
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

// ── non-streaming ────────────────────────────────────────────────────────

/// One mock token carrying a whole, well-formed tool-call block plus
/// leading content text — enough to drive the parser through the real
/// HTTP path in a single decode step.
fn single_call_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(
        mock.token_for(id, 0),
        "before <tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call> after",
    );
    map
}

#[tokio::test]
async fn non_streaming_returns_a_tool_call_and_strips_it_from_content() {
    let h = harness_with(RecordingTemplateProvider::permissive(single_call_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["content"], "before  after");
    let calls = message["tool_calls"].as_array().expect("tool_calls present");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "read_file");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.txt"}));
    assert!(!body.contains("<tool_call"), "raw XML leaked: {body}");
}

/// A tool call left open when `max_tokens` cuts generation off mid-block
/// (acceptance criterion 3): dropped whole, never returned half-written,
/// and the text before the open tag is still delivered as ordinary content.
#[tokio::test]
async fn a_call_truncated_by_max_tokens_is_dropped_not_half_written() {
    let mock = MockCompute::new();
    let mut decode = HashMap::new();
    decode.insert(
        mock.token_for(0, 0),
        "before <tool_call>\n<function=read_file>\n<parameter=path>\na",
    );
    let h = harness_with(RecordingTemplateProvider::permissive(decode));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["content"], "before ");
    assert!(
        message.as_object().unwrap().get("tool_calls").is_none(),
        "an interrupted call must not appear at all: {body}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(!body.contains('<'), "no XML fragment leaked: {body}");
}

/// The all-reasoning-no-content shape (GitHub #70): the token budget is
/// exhausted while still in the reasoning channel, so `content` is
/// genuinely empty — but distinguishably so, since `reasoning_content`
/// carries what was actually generated (never both empty, which is what a
/// silent, unreported empty completion would look like).
#[tokio::test]
async fn an_all_reasoning_generation_reports_populated_reasoning_and_empty_content() {
    let mock = MockCompute::new();
    let mut decode = HashMap::new();
    decode.insert(mock.token_for(0, 0), "still thinking, no marker yet");
    let h = harness_with(RecordingTemplateProvider::permissive(decode));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "still thinking, no marker yet");
    assert_eq!(message["content"], "");
    assert!(message.as_object().unwrap().get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

// ── streaming ────────────────────────────────────────────────────────────

/// The same tool call as `single_call_decode_map`, split across three
/// scheduler tokens so the open tag, the parameter, and the close tag each
/// land in a different SSE-triggering decode step.
fn split_call_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(mock.token_for(id, 0), "before <tool_");
    map.insert(mock.token_for(id, 1), "call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</functio");
    map.insert(mock.token_for(id, 2), "n>\n</tool_call> after");
    map
}

#[tokio::test]
async fn a_tool_call_split_across_scheduler_tokens_reassembles_over_sse() {
    let h = harness_with(RecordingTemplateProvider::permissive(split_call_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 3,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);

    let mut content = String::new();
    let mut tool_call_deltas = Vec::new();
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            tool_call_deltas.extend(calls.clone());
        }
        // No fragment of the tag survives outside a tool_calls delta.
        if delta.get("tool_calls").is_none() {
            let text = delta.to_string();
            assert!(!text.contains("<tool_call"), "leaked into a plain delta: {text}");
        }
    }
    assert_eq!(content, "before  after");
    assert_eq!(tool_call_deltas.len(), 1);
    assert_eq!(tool_call_deltas[0]["index"], 0);
    assert_eq!(tool_call_deltas[0]["function"]["name"], "read_file");
    let args: serde_json::Value = serde_json::from_str(
        tool_call_deltas[0]["function"]["arguments"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.txt"}));
}

/// Two tool calls in one response (acceptance criterion 2): distinct,
/// stable indices — never merged into one call.
fn two_calls_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(
        mock.token_for(id, 0),
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>",
    );
    map
}

#[tokio::test]
async fn several_tool_calls_in_one_response_stream_with_distinct_indices() {
    let h = harness_with(RecordingTemplateProvider::permissive(two_calls_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);
    let mut tool_call_deltas = Vec::new();
    for chunk in &chunks {
        if let Some(calls) = chunk["choices"][0]["delta"].get("tool_calls").and_then(|v| v.as_array()) {
            tool_call_deltas.extend(calls.clone());
        }
    }
    assert_eq!(tool_call_deltas.len(), 2);
    assert_eq!(tool_call_deltas[0]["index"], 0);
    assert_eq!(tool_call_deltas[0]["function"]["name"], "a");
    assert_eq!(tool_call_deltas[1]["index"], 1);
    assert_eq!(tool_call_deltas[1]["function"]["name"], "b");
    assert_ne!(tool_call_deltas[0]["id"], tool_call_deltas[1]["id"]);
}

#[tokio::test]
async fn a_streamed_call_truncated_by_max_tokens_is_dropped_not_half_written() {
    let mock = MockCompute::new();
    let mut decode = HashMap::new();
    decode.insert(
        mock.token_for(0, 0),
        "before <tool_call>\n<function=read_file>\n<parameter=path>\na",
    );
    let h = harness_with(RecordingTemplateProvider::permissive(decode));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);
    let mut content = String::new();
    let mut saw_tool_call = false;
    let mut finish_reason = None;
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        if delta.get("tool_calls").is_some() {
            saw_tool_call = true;
        }
        if let Some(r) = chunk["choices"][0]["finish_reason"].as_str() {
            finish_reason = Some(r.to_string());
        }
    }
    assert_eq!(content, "before ");
    assert!(!saw_tool_call, "an interrupted call must never reach the client: {body}");
    assert_eq!(finish_reason.as_deref(), Some("length"));
    assert!(!body.contains("<tool_call"), "raw XML leaked: {body}");
}
