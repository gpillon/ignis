//! `enable_thinking` / `reasoning_effort` wire contract (GitHub #68):
//! end-to-end tests over the real axum router, on the CPU gate (ADR 0006 —
//! `MockCompute`, no GPU).
//!
//! Mirrors `openai_http.rs`'s harness shape (prior art), but wires a
//! recording [`TemplateProvider`] double in place of the plain
//! `SimpleTemplateProvider`: it captures the [`ThinkingOptions`] each
//! request resolved to (so a test can assert on what actually reached the
//! template seam without reaching inside the server) and lets a test pin
//! exactly which text specific mock-emitted token ids decode to — the
//! `MockCompute` engine's own deterministic token stream is what lets a
//! test drive a `</think>` marker through the whole router and assert on
//! the resulting response / SSE frames.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::{ReasoningEffort, ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";

/// A [`TemplateProvider`] double: templating always delegates to the
/// built-in placeholder (the prompt tokens are not the point of these
/// tests), but every resolved [`ThinkingOptions`] is captured, the
/// advertised capabilities are configurable, and specific token ids can be
/// mapped to specific literal text — enough control to drive a `</think>`
/// marker through the real HTTP → SSE path deterministically.
struct RecordingTemplateProvider {
    inner: ignis_server::template::SimpleTemplateProvider,
    captured: Mutex<Vec<ThinkingOptions>>,
    capabilities: ThinkingCapabilities,
    decode: HashMap<TokenId, &'static str>,
}

impl RecordingTemplateProvider {
    fn new(capabilities: ThinkingCapabilities, decode: HashMap<TokenId, &'static str>) -> Self {
        Self {
            inner: ignis_server::template::SimpleTemplateProvider,
            captured: Mutex::new(Vec::new()),
            capabilities,
            decode,
        }
    }

    fn permissive(decode: HashMap<TokenId, &'static str>) -> Self {
        Self::new(ThinkingCapabilities::permissive(), decode)
    }

    fn captured_options(&self) -> Vec<ThinkingOptions> {
        self.captured.lock().unwrap().clone()
    }
}

impl TemplateProvider for RecordingTemplateProvider {
    fn apply_chat_template(&self, messages: &[ChatMessage], options: &ThinkingOptions) -> Vec<TokenId> {
        self.captured.lock().unwrap().push(*options);
        self.inner.apply_chat_template(messages, options)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|t| self.decode.get(t).copied().unwrap_or("?").to_string())
            .collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        self.capabilities.clone()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(RecordingTokenDecoder {
            decode: self.decode.clone(),
        })
    }
    // `decoder_starts_in_reasoning` deliberately left at the trait default
    // (`thinking.enable_thinking`) — unlike the placeholder, this double
    // stands in for a real thinking-aware template for these tests.
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
    template: Arc<RecordingTemplateProvider>,
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
    let template = Arc::new(template);
    let server = Server::new(Engine::new(Box::new(scheduler)), {
        // `Server::new` takes ownership of a boxed provider; the harness
        // keeps its own `Arc` clone to inspect captured options afterward.
        struct Shared(Arc<RecordingTemplateProvider>);
        impl TemplateProvider for Shared {
            fn apply_chat_template(&self, m: &[ChatMessage], o: &ThinkingOptions) -> Vec<TokenId> {
                self.0.apply_chat_template(m, o)
            }
            fn render_tokens(&self, t: &[TokenId]) -> String {
                self.0.render_tokens(t)
            }
            fn thinking_capabilities(&self) -> ThinkingCapabilities {
                self.0.thinking_capabilities()
            }
            fn token_decoder(&self) -> Box<dyn TokenDecoder> {
                self.0.token_decoder()
            }
            fn decoder_starts_in_reasoning(&self, o: &ThinkingOptions) -> bool {
                self.0.decoder_starts_in_reasoning(o)
            }
        }
        Box::new(Shared(Arc::clone(&template)))
    })
    .with_request_timeout(Duration::from_secs(5));
    Harness {
        app: server.app(),
        template,
    }
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

// ── wire-contract validation ────────────────────────────────────────────

#[tokio::test]
async fn enable_thinking_false_is_accepted_and_reaches_the_template() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let captured = h.template.captured_options();
    assert_eq!(captured.len(), 1);
    assert!(!captured[0].enable_thinking);
}

#[tokio::test]
async fn a_non_boolean_enable_thinking_is_a_400_naming_the_field() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "enable_thinking": "nope"
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("enable_thinking"));
}

#[tokio::test]
async fn matching_top_level_and_chat_template_kwargs_values_are_accepted() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "chat_template_kwargs": { "enable_thinking": false }
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn conflicting_top_level_and_chat_template_kwargs_values_are_a_400() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "enable_thinking": false,
        "chat_template_kwargs": { "enable_thinking": true }
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn an_unsupported_chat_template_kwargs_key_is_a_400() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "chat_template_kwargs": { "top_p": 0.9 }
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].as_str().unwrap().contains("top_p"));
}

#[tokio::test]
async fn an_unknown_reasoning_effort_is_a_400_listing_accepted_values() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "reasoning_effort": "super-duper"
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("xhigh"));
}

#[tokio::test]
async fn an_effort_the_template_cannot_honour_is_a_400_with_a_distinct_code() {
    let caps = ThinkingCapabilities {
        can_disable: true,
        supported_efforts: [ReasoningEffort::Low].into_iter().collect(),
    };
    let h = harness_with(RecordingTemplateProvider::new(caps, HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "reasoning_effort": "high"
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    // A capability error is distinguishable from a plain validation error
    // (the wire contract's machine-readable `code`).
    assert_eq!(v["error"]["code"], "reasoning_effort_unsupported");
}

#[tokio::test]
async fn null_enable_thinking_is_treated_as_unset() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": null
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    // The server's own default (true) applies — not a validation error.
    assert!(h.template.captured_options()[0].enable_thinking);
}

// ── response separation ─────────────────────────────────────────────────

/// Two mock tokens the harness maps to `"reasoning"` then `"</think>answer"`
/// — enough to drive the marker through the whole router in one request.
fn thinking_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(mock.token_for(id, 0), "reasoning");
    map.insert(mock.token_for(id, 1), "</think>answer");
    map
}

#[tokio::test]
async fn non_streaming_splits_reasoning_content_and_content() {
    let h = harness_with(RecordingTemplateProvider::permissive(thinking_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 2,
        "stream": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "reasoning");
    assert_eq!(v["choices"][0]["message"]["content"], "answer");
    // No think markers ever leak into either channel.
    let msg = v["choices"][0]["message"].to_string();
    assert!(!msg.contains("<think"), "{msg}");
}

/// Three mock tokens that split the `</think>` marker itself across two of
/// them (`"</thi"` then `"nk>"`) — unlike `thinking_decode_map`, where one
/// token already carries the whole marker, this drives the marker-straddling
/// case through the real HTTP → scheduler → SSE path, at the seam the spec's
/// Testing Decisions calls primary (not just in `decoder.rs`'s own unit
/// tests, which pin the same case as a bare string table with no HTTP
/// involved).
fn split_marker_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(mock.token_for(id, 0), "reasoning</thi");
    map.insert(mock.token_for(id, 1), "nk>");
    map.insert(mock.token_for(id, 2), "answer");
    map
}

#[tokio::test]
async fn a_marker_split_across_two_scheduler_tokens_is_still_found_over_sse() {
    let h = harness_with(RecordingTemplateProvider::permissive(split_marker_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 3,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .filter(|l| l != "[DONE]")
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect();

    let mut reasoning = String::new();
    let mut content = String::new();
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            reasoning.push_str(r);
        }
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        // The marker must never appear whole (or fragmented) in either
        // channel of any single chunk.
        let text = chunk.to_string();
        assert!(!text.contains("</thi"), "marker fragment leaked: {text}");
    }
    assert_eq!(reasoning, "reasoning");
    assert_eq!(content, "answer");
}

#[tokio::test]
async fn a_thinking_disabled_response_omits_reasoning_content_entirely() {
    let mock = MockCompute::new();
    let mut decode = HashMap::new();
    decode.insert(mock.token_for(0, 0), "the answer");
    let h = harness_with(RecordingTemplateProvider::permissive(decode));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        v["choices"][0]["message"].as_object().unwrap().get("reasoning_content").is_none(),
        "reasoning_content must be omitted, not null: {body}"
    );
    assert_eq!(v["choices"][0]["message"]["content"], "the answer");
}

#[tokio::test]
async fn streaming_carries_reasoning_then_content_deltas_in_order() {
    let h = harness_with(RecordingTemplateProvider::permissive(thinking_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 2,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let data_lines: Vec<String> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .collect();
    assert_eq!(data_lines.last().map(String::as_str), Some("[DONE]"), "{body}");
    let chunks: Vec<serde_json::Value> = data_lines
        .iter()
        .filter(|l| l.as_str() != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    // Reassemble each channel across every chunk, in order — the marker
    // must never appear in either.
    let mut reasoning = String::new();
    let mut content = String::new();
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(r) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
            reasoning.push_str(r);
        }
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
    }
    assert_eq!(reasoning, "reasoning");
    assert_eq!(content, "answer");

    // The reasoning delta(s) all arrive before the content delta(s)
    // (a "thinking…" indicator can swap to the answer the moment content
    // starts, per story 20).
    let last_reasoning_idx = chunks
        .iter()
        .rposition(|c| c["choices"][0]["delta"].get("reasoning_content").is_some());
    let first_content_idx = chunks
        .iter()
        .position(|c| c["choices"][0]["delta"].get("content").is_some());
    if let (Some(last_r), Some(first_c)) = (last_reasoning_idx, first_content_idx) {
        assert!(last_r < first_c, "reasoning must precede content: {body}");
    }
}

#[tokio::test]
async fn responses_api_text_carries_only_the_content_channel() {
    let h = harness_with(RecordingTemplateProvider::permissive(thinking_decode_map(0)));
    let req = serde_json::json!({
        "input": "hi",
        "max_output_tokens": 2
    });
    let (status, body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let text = v["output"][0]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "answer", "text must be the content channel only");
    assert!(!body.contains("reasoning"), "no reasoning field on this endpoint: {body}");
}

// ── multi-turn reasoning (stories 27-29) ────────────────────────────────
//
// The actual drop-vs-preserve *behavior* (whether the prior reasoning text
// reaches the real template's rendered prompt) is pinned with the real
// Qwen-shaped template in `artifact_template.rs`'s
// `apply_chat_template_drops_reasoning_content_unless_preserved` — the
// placeholder provider used by this file's harness has no jinja template to
// observe that in. What belongs at the HTTP seam is that `preserve_thinking`
// parses off the wire and reaches the resolved `ThinkingOptions` the
// provider is handed.

#[tokio::test]
async fn preserve_thinking_reaches_the_resolved_options() {
    let h_dropped = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let h_preserved = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    let messages = serde_json::json!([
        { "role": "user", "content": "hi" },
        { "role": "assistant", "content": "hello", "reasoning_content": "scratch" }
    ]);
    let dropped_req = serde_json::json!({ "messages": messages, "max_tokens": 1 });
    let preserved_req = serde_json::json!({
        "messages": messages,
        "max_tokens": 1,
        "preserve_thinking": true
    });
    let (s1, b1) = call(&h_dropped.app, "POST", "/v1/chat/completions", Some(dropped_req)).await;
    let (s2, b2) = call(&h_preserved.app, "POST", "/v1/chat/completions", Some(preserved_req)).await;
    assert_eq!(s1, 200, "{b1}");
    assert_eq!(s2, 200, "{b2}");
    let captured_dropped = &h_dropped.template.captured_options()[0];
    let captured_preserved = &h_preserved.template.captured_options()[0];
    assert!(!captured_dropped.preserve_thinking);
    assert!(captured_preserved.preserve_thinking);
}
