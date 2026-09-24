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

use ignis_core::thinking_budget::{ThinkingClose, ANSWER_RESERVE};
use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ReasoningEffort, ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::SharedTemplate;

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
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[serde_json::Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        self.captured.lock().unwrap().push(*options);
        self.inner.apply_chat_template(messages, options, tools)
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
    compute: Arc<MockCompute>,
}

fn harness_with(template: RecordingTemplateProvider) -> Harness {
    harness_with_budget(template, None)
}

fn harness_with_budget(template: RecordingTemplateProvider, default_budget: Option<u32>) -> Harness {
    harness_from(template, Setup { default_budget, ..Setup::default() })
}

/// What a harness's server is started with, beyond its template.
#[derive(Default)]
struct Setup {
    /// `--thinking-budget`.
    default_budget: Option<u32>,
    /// `--reasoning-effort`.
    default_effort: Option<ReasoningEffort>,
    /// The close the scheduler forces once a budget is spent; `None` leaves
    /// every budget inert, as a tokenizer that splits `</think>` does.
    close: Option<ThinkingClose>,
}

fn harness_from(template: RecordingTemplateProvider, setup: Setup) -> Harness {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            // Room for a budget above the answer reserve (spec server/08).
            max_sequence_tokens: 65_536,
            thinking_close: setup.close.map(Arc::new),
            ..SchedulerConfig::default()
        },
        compute.clone(),
    );
    let template = Arc::new(template);
    // `Server::new` takes ownership of a boxed provider; the harness keeps
    // its own `Arc` clone (via `SharedTemplate`, GitHub #132) to inspect
    // captured options afterward.
    let server = Server::new(
        Engine::new(Box::new(scheduler)),
        Box::new(SharedTemplate(Arc::clone(&template))),
    )
    .with_request_timeout(Duration::from_secs(5))
    .with_thinking_defaults(true, setup.default_effort)
    .with_thinking_budget(setup.default_budget);
    Harness {
        app: server.app(),
        template,
        compute,
    }
}

/// Qwen3.8's efforts: `max` is not among them.
fn qwen38() -> ThinkingCapabilities {
    ThinkingCapabilities {
        can_disable: true,
        supported_efforts: [ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::Xhigh]
            .into_iter()
            .collect(),
    }
}

/// The thinking budget every decode round of the harness's requests carried.
fn budgets(h: &Harness) -> Vec<Option<u32>> {
    h.compute
        .decode_calls()
        .iter()
        .flatten()
        .map(|job| job.params.thinking_budget)
        .collect()
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
    // A template that takes no effort at all: any other set rounds the
    // effort to one it takes (the next test).
    let caps = ThinkingCapabilities {
        can_disable: true,
        supported_efforts: Default::default(),
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
async fn a_high_effort_reaches_a_qwen38_template_as_xhigh() {
    // Qwen3.8's template raises on anything but low/medium/xhigh, and coding
    // agents send the OpenAI vocabulary's `high`.
    let caps = ThinkingCapabilities {
        can_disable: true,
        supported_efforts: [ReasoningEffort::Low, ReasoningEffort::Medium, ReasoningEffort::Xhigh]
            .into_iter()
            .collect(),
    };
    let h = harness_with(RecordingTemplateProvider::new(caps, HashMap::new()));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "reasoning_effort": "high"
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let options = h.template.captured_options();
    assert!(options[0].enable_thinking);
    assert_eq!(options[0].reasoning_effort, Some(ReasoningEffort::Xhigh));
}

// ── the thinking budget (spec server/08) ─────────────────────────────────

/// A generation with room for any budget these tests name above the answer
/// reserve. The mock stops each request after a few tokens: the budget a
/// round carries is the point, not how far the request runs.
const ROOM: u32 = 20_000;
const MAX_TOKENS: u32 = ANSWER_RESERVE + ROOM;

/// A harness whose requests stop after three tokens.
fn budget_harness(template: RecordingTemplateProvider, setup: Setup) -> Harness {
    let h = harness_from(template, setup);
    for id in 0..64 {
        h.compute.stop_after(id, 3);
    }
    h
}

/// POST `extra` over a one-message request to `path` and return the budget
/// its decode rounds carried — every round of one request carries the same.
async fn budget_of(h: &Harness, path: &str, extra: serde_json::Value) -> Option<u32> {
    let before = budgets(h).len();
    let mut req = match path {
        "/v1/responses" => serde_json::json!({ "input": "hi", "max_output_tokens": MAX_TOKENS }),
        _ => serde_json::json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": MAX_TOKENS
        }),
    };
    req.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    let (status, response) = call(&h.app, "POST", path, Some(req)).await;
    assert_eq!(status, 200, "{extra}: {response}");
    let seen = budgets(h)[before..].to_vec();
    assert!(!seen.is_empty() && seen.iter().all(|&b| b == seen[0]), "{extra}: {seen:?}");
    seen[0]
}

#[tokio::test]
async fn the_server_default_applies_and_a_request_overrides_it_or_opts_out() {
    let h = budget_harness(
        RecordingTemplateProvider::permissive(HashMap::new()),
        Setup { default_budget: Some(8192), ..Setup::default() },
    );
    for (extra, want) in [
        // Absent or null: the server's `--thinking-budget`.
        (serde_json::json!({}), Some(8192)),
        (serde_json::json!({ "thinking_budget": null }), Some(8192)),
        // A number wins, downward or upward: the server does not cap it.
        (serde_json::json!({ "thinking_budget": 64 }), Some(64)),
        (serde_json::json!({ "thinking_budget": 12288 }), Some(12288)),
        // 0: no budget for this request.
        (serde_json::json!({ "thinking_budget": 0 }), None),
        // Thinking off: no block to close, whatever was asked.
        (serde_json::json!({ "thinking_budget": 64, "enable_thinking": false }), None),
        (serde_json::json!({ "reasoning_effort": "none" }), None),
    ] {
        assert_eq!(budget_of(&h, "/v1/chat/completions", extra.clone()).await, want, "{extra}");
    }
    // The responses API resolves the same field the same way.
    assert_eq!(budget_of(&h, "/v1/responses", serde_json::json!({})).await, Some(8192));
    assert_eq!(budget_of(&h, "/v1/responses", serde_json::json!({ "thinking_budget": 0 })).await, None);
}

#[tokio::test]
async fn without_a_server_default_only_a_request_sets_a_budget() {
    let h = budget_harness(RecordingTemplateProvider::permissive(HashMap::new()), Setup::default());
    assert_eq!(budget_of(&h, "/v1/chat/completions", serde_json::json!({})).await, None);
    assert_eq!(
        budget_of(&h, "/v1/chat/completions", serde_json::json!({ "thinking_budget": 64 })).await,
        Some(64)
    );
}

#[tokio::test]
async fn a_malformed_thinking_budget_is_a_400_naming_the_field() {
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::new()));
    for path in ["/v1/chat/completions", "/v1/responses"] {
        for bad in [
            serde_json::json!(-5),
            serde_json::json!(1.5),
            serde_json::json!("64"),
            serde_json::json!(true),
            serde_json::json!({ "tokens": 64 }),
            serde_json::json!(4_294_967_296u64),
        ] {
            let req = match path {
                "/v1/responses" => serde_json::json!({ "input": "hi", "thinking_budget": bad }),
                _ => serde_json::json!({
                    "messages": [{ "role": "user", "content": "hi" }],
                    "thinking_budget": bad
                }),
            };
            let (status, body) = call(&h.app, "POST", path, Some(req)).await;
            assert_eq!(status, 400, "{path} {bad}: {body}");
            let v: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(v["error"]["param"], "thinking_budget", "{path} {bad}: {body}");
            assert!(v["error"]["message"].as_str().unwrap().contains("thinking_budget"), "{body}");
        }
    }
}

#[tokio::test]
async fn max_renders_as_xhigh_and_runs_without_any_budget() {
    let h = budget_harness(
        RecordingTemplateProvider::new(qwen38(), HashMap::new()),
        Setup { default_budget: Some(8192), ..Setup::default() },
    );
    for extra in [
        serde_json::json!({ "reasoning_effort": "max" }),
        // A budget riding the same request — a client default, a stale
        // setting — is ignored rather than refused or honoured.
        serde_json::json!({ "reasoning_effort": "max", "thinking_budget": 64 }),
        serde_json::json!({ "reasoning_effort": "max", "thinking_budget": 0 }),
    ] {
        assert_eq!(budget_of(&h, "/v1/chat/completions", extra.clone()).await, None, "{extra}");
    }
    // Every one of them reached the template as the most Qwen3.8 takes.
    let options = h.template.captured_options();
    assert_eq!(options.len(), 3);
    assert!(options.iter().all(|o| o.enable_thinking && o.reasoning_effort == Some(ReasoningEffort::Xhigh)));

    // Ignored is not unvalidated: a budget that is not a budget is still the
    // client's mistake.
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "reasoning_effort": "max",
        "thinking_budget": "lots"
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn a_server_default_of_max_drops_the_budget_unless_the_request_asks_another_effort() {
    let h = budget_harness(
        RecordingTemplateProvider::new(qwen38(), HashMap::new()),
        Setup {
            default_budget: Some(8192),
            default_effort: Some(ReasoningEffort::Max),
            ..Setup::default()
        },
    );
    for (extra, want) in [
        (serde_json::json!({}), None),
        (serde_json::json!({ "thinking_budget": 64 }), None),
        (serde_json::json!({ "enable_thinking": true }), None),
        // Another effort on the request is the resolved one, and gets the
        // budget rules.
        (serde_json::json!({ "reasoning_effort": "xhigh" }), Some(8192)),
        (serde_json::json!({ "reasoning_effort": "high" }), Some(8192)),
        (serde_json::json!({ "reasoning_effort": "medium", "thinking_budget": 64 }), Some(64)),
    ] {
        assert_eq!(budget_of(&h, "/v1/chat/completions", extra.clone()).await, want, "{extra}");
    }
}

#[tokio::test]
async fn the_budget_leaves_the_answer_room_inside_max_tokens() {
    let h = budget_harness(
        RecordingTemplateProvider::permissive(HashMap::new()),
        Setup { default_budget: Some(8192), ..Setup::default() },
    );
    for (extra, want) in [
        // 1,000 tokens above the reserve: the default is clamped to them.
        (serde_json::json!({ "max_tokens": ANSWER_RESERVE + 1000 }), Some(1000)),
        // A request's own budget is clamped the same way, and one that fits
        // is untouched.
        (serde_json::json!({ "max_tokens": ANSWER_RESERVE + 1000, "thinking_budget": 4096 }), Some(1000)),
        (serde_json::json!({ "max_tokens": ANSWER_RESERVE + 1000, "thinking_budget": 64 }), Some(64)),
        // No room above the reserve: no budget, not a close at token 0.
        (serde_json::json!({ "max_tokens": ANSWER_RESERVE }), None),
        (serde_json::json!({ "max_tokens": 16 }), None),
    ] {
        assert_eq!(budget_of(&h, "/v1/chat/completions", extra.clone()).await, want, "{extra}");
    }
    assert_eq!(
        budget_of(&h, "/v1/responses", serde_json::json!({ "max_output_tokens": ANSWER_RESERVE + 1000 })).await,
        Some(1000)
    );
}

// ── forced-close visibility (spec server/08, the #265/#266 contract) ─────

const THINK_END: TokenId = 999;
/// The close the harness's scheduler forces: its end marker decodes as
/// `</think>`, so the response splits after it like the real one.
const CLOSE: [TokenId; 4] = [900, 901, THINK_END, 902];

fn close_decode_map() -> HashMap<TokenId, &'static str> {
    HashMap::from([
        (900, "\n\nConsidering the limited time"),
        (901, " by the user, now."),
        (THINK_END, "</think>"),
        (902, "\n\n"),
    ])
}

/// A harness whose scheduler forces [`CLOSE`] once a budget is spent, and
/// whose requests stop after twelve tokens: a budget of 3 is spent, the
/// close forced over the next rounds, and a few answer tokens follow.
fn forcing_harness(close: ThinkingClose) -> Harness {
    let h = harness_from(
        RecordingTemplateProvider::permissive(close_decode_map()),
        Setup { close: Some(close), ..Setup::default() },
    );
    for id in 0..16 {
        h.compute.stop_after(id, 12);
    }
    h
}

fn the_close() -> ThinkingClose {
    ThinkingClose::new(CLOSE.to_vec(), THINK_END).unwrap()
}

/// The single-token sets the decode rounds carried, in order.
fn forced_sets(h: &Harness) -> Vec<TokenId> {
    h.compute
        .decode_calls()
        .iter()
        .flatten()
        .filter_map(|job| job.permitted.as_ref().map(|set| set[0]))
        .collect()
}

fn chat_request(extra: serde_json::Value) -> serde_json::Value {
    let mut req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": MAX_TOKENS
    });
    req.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    req
}

#[tokio::test]
async fn a_forced_close_is_reported_on_the_choice_with_the_reasoning_tokens_before_it() {
    let h = forcing_harness(the_close());
    let (status, body) =
        call(&h.app, "POST", "/v1/chat/completions", Some(chat_request(serde_json::json!({ "thinking_budget": 3 })))).await;
    assert_eq!(status, 200, "{body}");
    // The rounds carried the close, one token each.
    assert_eq!(forced_sets(&h), CLOSE.to_vec());
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    // Budget 3: the round that found three emitted let the token it had
    // already drawn through, so the close began after four.
    assert_eq!(choice["thinking_budget_forced_at"], 4, "{body}");
    // The model answered after the close it was handed.
    // Twelve tokens: four free, the four of the close, four more — each an
    // id the harness's decoder does not know, so a `?`.
    assert_eq!(choice["message"]["content"], "????", "{body}");
    assert!(choice["message"]["reasoning_content"].as_str().unwrap().ends_with("by the user, now."), "{body}");
}

#[tokio::test]
async fn an_unforced_response_carries_no_forced_field_at_all() {
    // No budget: nothing to force.
    let h = forcing_harness(the_close());
    let (status, body) =
        call(&h.app, "POST", "/v1/chat/completions", Some(chat_request(serde_json::json!({ "thinking_budget": 0 })))).await;
    assert_eq!(status, 200, "{body}");
    assert!(forced_sets(&h).is_empty());
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0].get("thinking_budget_forced_at").is_none(), "absent, not null: {body}");

    // A model that closes its block itself, before its budget: never forced.
    let natural_end = MockCompute::new().token_for(0, 1);
    let h = forcing_harness(ThinkingClose::new(vec![900, 901, natural_end, 902], natural_end).unwrap());
    let (status, body) =
        call(&h.app, "POST", "/v1/chat/completions", Some(chat_request(serde_json::json!({ "thinking_budget": 3 })))).await;
    assert_eq!(status, 200, "{body}");
    assert!(forced_sets(&h).is_empty());
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0].get("thinking_budget_forced_at").is_none(), "{body}");
}

#[tokio::test]
async fn streaming_reports_a_forced_close_on_the_finish_chunk_only() {
    let h = forcing_harness(the_close());
    let req = chat_request(serde_json::json!({
        "thinking_budget": 3,
        "stream": true,
        "stream_options": { "include_usage": true }
    }));
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .filter(|l| l != "[DONE]")
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect();
    let finish: Vec<&serde_json::Value> =
        chunks.iter().filter(|c| !c["choices"][0]["finish_reason"].is_null()).collect();
    assert_eq!(finish.len(), 1, "{body}");
    assert_eq!(finish[0]["choices"][0]["thinking_budget_forced_at"], 4, "{body}");
    let elsewhere = chunks
        .iter()
        .filter(|c| c["choices"][0]["finish_reason"].is_null())
        .filter(|c| c.to_string().contains("thinking_budget_forced_at"))
        .count();
    assert_eq!(elsewhere, 0, "only the finish chunk carries it: {body}");
}

#[tokio::test]
async fn the_responses_api_reports_a_forced_close_on_the_response_object() {
    let h = forcing_harness(the_close());
    let req = serde_json::json!({ "input": "hi", "max_output_tokens": MAX_TOKENS, "thinking_budget": 3 });
    let (status, body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["thinking_budget_forced_at"], 4, "{body}");

    let req = serde_json::json!({ "input": "hi", "max_output_tokens": MAX_TOKENS, "thinking_budget": 0 });
    let (status, body) = call(&h.app, "POST", "/v1/responses", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v.get("thinking_budget_forced_at").is_none(), "{body}");
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
// The actual keep-vs-strip *behavior* (which prior reasoning reaches the
// rendered prompt) is pinned against the real template in the artifact
// crate's `preserve_thinking_*` tests and against the reference itself in
// `tests/fixtures/vision/expected/chat_history*.json` — the placeholder
// provider used by this file's harness has no jinja template to observe
// that in. What belongs at the HTTP seam is that `preserve_thinking`
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
