//! `POST /v1/tokenize` and `POST /v1/detokenize` (GitHub #285, spec
//! server/10): the prompt counted without being served. End to end through
//! the real axum router over a mock-compute engine (CPU-only, ADR 0006).
//!
//! The template is a byte-level double — one token per byte, so a render's
//! text and its ids are the same thing seen twice — that renders tools and
//! the thinking controls into the prompt, which is what lets these tests
//! tell whether the two routes really share one render path with
//! `/v1/chat/completions`. The real tokenizer's own round trip is the
//! artifact-gated test at the bottom.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_core::{
    ConcreteScheduler, MockCompute, RequestClass, RequestId, RequestInput, SchedEvent, Scheduler, SchedulerConfig,
    SubmitError,
};
use ignis_server::config::ApiKey;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy};
use ignis_server::template::{
    ChatMessage, PromptText, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor, Gated};

const MODEL: &str = "test-model";
const KEY: &str = "sk-tokenize";

// ── the double ──────────────────────────────────────────────────────────────

/// One token per byte, ids 0..=255. The render is a fixed layout of the
/// conversation, the tool block and the thinking controls, so a body's count
/// moves with each of them.
struct ByteTemplate;

impl ByteTemplate {
    fn render(messages: &[ChatMessage], options: &ThinkingOptions, tools: &[Value]) -> String {
        let mut text = String::new();
        if !tools.is_empty() {
            text.push_str(&format!("<tools>{}</tools>\n", Value::Array(tools.to_vec())));
        }
        for message in messages {
            text.push_str(&format!("<|{}|>{}\n", message.role, message.content.text()));
        }
        text.push_str(if options.enable_thinking { "<|assistant|><think>\n" } else { "<|assistant|>" });
        text
    }

    fn prompt(text: String, with_text: bool) -> RenderedPrompt {
        let tokens = text.bytes().map(u32::from).collect::<Vec<_>>();
        let text = with_text.then(|| PromptText {
            offsets: (0..text.len()).map(|at| (at, at + 1)).collect(),
            text,
        });
        RenderedPrompt { text, ..tokens.into() }
    }
}

impl TemplateProvider for ByteTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        Ok(Self::prompt(Self::render(messages, options, tools), false))
    }

    fn apply_chat_template_with_text(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        Ok(Self::prompt(Self::render(messages, options, tools), true))
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<u32>> {
        Some(text.bytes().map(u32::from).collect())
    }

    fn is_token(&self, id: u32) -> bool {
        id < 256
    }

    fn render_tokens(&self, tokens: &[u32]) -> String {
        String::from_utf8_lossy(&tokens.iter().map(|&id| id as u8).collect::<Vec<_>>()).into_owned()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        SimpleTemplateProvider.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }
}

/// The concrete scheduler, recording every submission it receives — what
/// "no lane, no sequence" is asserted against.
struct Recording {
    inner: ConcreteScheduler,
    submitted: Arc<Mutex<Vec<RequestInput>>>,
}

impl Scheduler for Recording {
    fn submit(&mut self, input: RequestInput, class: RequestClass) -> Result<RequestId, SubmitError> {
        self.submitted.lock().unwrap().push(input.clone());
        self.inner.submit(input, class)
    }
    fn cancel(&mut self, request: RequestId) -> bool {
        self.inner.cancel(request)
    }
    fn advance(&mut self) -> Vec<SchedEvent> {
        self.inner.advance()
    }
    fn is_idle(&self) -> bool {
        self.inner.is_idle()
    }
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }
    fn max_sequence_tokens(&self) -> u32 {
        self.inner.max_sequence_tokens()
    }
    fn mode(&self) -> ignis_core::EngineMode {
        self.inner.mode()
    }
    fn occupancy(&self) -> ignis_core::Occupancy {
        self.inner.occupancy()
    }
}

struct Harness {
    app: axum::Router,
    metrics: Option<axum::Router>,
    submitted: Arc<Mutex<Vec<RequestInput>>>,
    max_context: u32,
}

fn build(template: Box<dyn TemplateProvider>, shape: impl FnOnce(Server) -> Server) -> Harness {
    let config = SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() };
    let max_context = config.max_sequence_tokens;
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let scheduler = Recording {
        inner: ConcreteScheduler::with_config(config, Arc::new(MockCompute::new())),
        submitted: submitted.clone(),
    };
    let server = shape(Server::new(Engine::new(Box::new(scheduler)), template).with_request_timeout(Duration::from_secs(10)));
    Harness { app: server.app(), metrics: server.metrics_app(), submitted, max_context }
}

fn harness() -> Harness {
    build(Box::new(ByteTemplate), |server| server)
}

async fn post(app: &axum::Router, path: &str, body: Value) -> (u16, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// `body` with `extra`'s fields laid over it.
fn merged(body: &Value, extra: Value) -> Value {
    let mut out = body.as_object().unwrap().clone();
    out.extend(extra.as_object().unwrap().clone());
    Value::Object(out)
}

fn user(text: &str) -> Value {
    json!({ "role": "user", "content": text })
}

fn tool() -> Value {
    json!({ "type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": { "type": "object", "properties": { "path": { "type": "string" } } } } })
}

/// The bodies the fixture set is made of (acceptance 1): plain, with tools,
/// with thinking off, and with a long system block.
fn fixture_bodies() -> Vec<(&'static str, Value)> {
    vec![
        ("plain", json!({ "model": MODEL, "messages": [user("What is in the file?")] })),
        ("tools", json!({ "model": MODEL, "messages": [user("Read a.txt")], "tools": [tool()] })),
        (
            "thinking off",
            json!({ "model": MODEL, "messages": [user("hi")], "enable_thinking": false }),
        ),
        (
            "long system block",
            json!({ "model": MODEL, "messages": [
                { "role": "system", "content": "You are careful. ".repeat(400) },
                user("hi"),
            ] }),
        ),
    ]
}

// ── the count is the usage count ────────────────────────────────────────────

#[tokio::test]
async fn the_count_is_the_prompt_tokens_the_same_body_reports_when_served() {
    let h = harness();
    let mut counts = Vec::new();
    for (name, body) in fixture_bodies() {
        let (status, counted) = post(&h.app, "/v1/tokenize", body.clone()).await;
        assert_eq!(status, 200, "{name}: {counted}");
        let (status, served) = post(&h.app, "/v1/chat/completions", merged(&body, json!({ "max_tokens": 1, "temperature": 0 }))).await;
        assert_eq!(status, 200, "{name}: {served}");
        assert_eq!(counted["count"], served["usage"]["prompt_tokens"], "{name}");
        counts.push(counted["count"].as_u64().unwrap());
    }
    // The fixtures differ, so the equality above is not vacuous: the tool
    // block and the thinking controls are in the number.
    assert!(counts[1] > counts[0], "tools cost tokens: {counts:?}");
    assert!(counts[2] < counts[0], "thinking off changes the render: {counts:?}");
    assert!(counts[3] > counts[0] + 1000, "a long system block is counted: {counts:?}");
}

#[tokio::test]
async fn sampling_fields_are_inert_and_the_lane_suffix_is_read_like_chat_does() {
    let h = harness();
    let (_, plain) = post(&h.app, "/v1/tokenize", json!({ "messages": [user("hi")] })).await;
    let (status, inert) = post(
        &h.app,
        "/v1/tokenize",
        json!({ "model": "test-model@agent", "messages": [user("hi")], "temperature": 0.7, "max_tokens": 5, "top_p": 0.5 }),
    )
    .await;
    assert_eq!(status, 200, "{inert}");
    assert_eq!(inert["count"], plain["count"]);
}

#[tokio::test]
async fn a_body_chat_would_refuse_is_refused_with_the_same_error() {
    let h = harness();
    for (name, body) in [
        ("bad reasoning_effort", json!({ "messages": [user("hi")], "reasoning_effort": "sideways" })),
        ("bad thinking_budget", json!({ "messages": [user("hi")], "thinking_budget": "many" })),
        ("bad tool_choice", json!({ "messages": [user("hi")], "tools": [tool()], "tool_choice": "required" })),
        ("bad tool", json!({ "messages": [user("hi")], "tools": [{ "type": "function" }] })),
        ("bad role", json!({ "messages": [{ "role": "bogus", "content": "hi" }] })),
    ] {
        let (tokenize_status, tokenized) = post(&h.app, "/v1/tokenize", body.clone()).await;
        let (chat_status, chat) = post(&h.app, "/v1/chat/completions", body).await;
        assert_eq!(tokenize_status, 400, "{name}: {tokenized}");
        assert_eq!((tokenize_status, &tokenized), (chat_status, &chat), "{name}");
    }
}

#[tokio::test]
async fn a_model_the_server_has_not_loaded_is_a_404_as_it_is_on_chat() {
    let h = harness();
    let (status, body) = post(&h.app, "/v1/tokenize", json!({ "model": "other", "messages": [user("hi")] })).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
}

// ── the ceiling ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn every_answer_carries_the_servers_context_ceiling() {
    let h = harness();
    let (_, models) = {
        let response = h
            .app
            .clone()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (200, serde_json::from_slice::<Value>(&bytes).unwrap())
    };
    let ceiling = models["data"][0]["max_model_len"].as_u64().unwrap();
    assert_eq!(ceiling, u64::from(h.max_context));
    for body in [json!({ "messages": [user("hi")] }), json!({ "prompt": "hi" })] {
        let (status, answer) = post(&h.app, "/v1/tokenize", body).await;
        assert_eq!(status, 200, "{answer}");
        assert_eq!(answer["max_model_len"], ceiling);
    }
}

// ── the optional fields ─────────────────────────────────────────────────────

#[tokio::test]
async fn ids_and_text_are_only_there_when_asked_for() {
    let h = harness();
    let body = json!({ "messages": [user("hi")] });
    let (_, bare) = post(&h.app, "/v1/tokenize", body.clone()).await;
    assert!(bare.get("token_ids").is_none() && bare.get("text").is_none(), "{bare}");

    let (_, both) = post(&h.app, "/v1/tokenize", merged(&body, json!({ "return_token_ids": true, "return_text": true }))).await;
    let text = both["text"].as_str().unwrap();
    assert_eq!(text, "<|user|>hi\n<|assistant|><think>\n");
    let ids = both["token_ids"].as_array().unwrap();
    assert_eq!(ids.len() as u64, both["count"].as_u64().unwrap());
    assert_eq!(both["count"], bare["count"], "asking for the text does not change the count");
}

// ── round trip ──────────────────────────────────────────────────────────────

/// `detokenize(tokenize(x).token_ids)` is the rendered text, byte for byte.
async fn round_trips(h: &Harness, body: Value) {
    let (status, tokenized) = post(&h.app, "/v1/tokenize", merged(&body, json!({ "return_token_ids": true, "return_text": true }))).await;
    assert_eq!(status, 200, "{tokenized}");
    let (status, back) = post(&h.app, "/v1/detokenize", json!({ "token_ids": tokenized["token_ids"] })).await;
    assert_eq!(status, 200, "{back}");
    assert_eq!(back["text"], tokenized["text"]);
}

#[tokio::test]
async fn tokenize_then_detokenize_returns_the_rendered_text() {
    let h = harness();
    // ASCII, multi-byte UTF-8, and a prompt ending mid-grapheme: an emoji
    // with a zero-width joiner after it, and a base letter with its
    // combining accent still to come.
    for text in ["plain ascii", "città — 日本語 — 🦀", "family 👨\u{200d}", "e\u{0301}", "cafe\u{0301}"] {
        round_trips(&h, json!({ "messages": [user(text)] })).await;
        round_trips(&h, json!({ "prompt": text })).await;
    }
}

// ── the raw form ────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_raw_form_applies_no_template() {
    let h = harness();
    let (status, answer) = post(&h.app, "/v1/tokenize", json!({ "prompt": "ab", "return_token_ids": true, "return_text": true })).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["token_ids"], json!([97, 98]), "the tokenizer's own ids, no chat markers");
    assert_eq!(answer["count"], 2);
    assert_eq!(answer["text"], "ab");
}

#[tokio::test]
async fn a_provider_with_no_tokenizer_refuses_the_raw_form_rather_than_inventing_ids() {
    let h = build(Box::new(SimpleTemplateProvider), |server| server);
    let (status, answer) = post(&h.app, "/v1/tokenize", json!({ "prompt": "ab" })).await;
    assert_eq!(status, 501, "{answer}");
    assert_eq!(answer["error"]["code"], "tokenizer_unavailable");
}

// ── refusals ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn each_refusal_has_its_code() {
    let h = harness();
    for (name, path, body, code, param) in [
        ("both forms", "/v1/tokenize", json!({ "messages": [user("hi")], "prompt": "hi" }), "invalid_request_error", None),
        ("neither form", "/v1/tokenize", json!({ "model": MODEL }), "invalid_request_error", None),
        ("empty messages", "/v1/tokenize", json!({ "messages": [] }), "invalid_request_error", None),
        ("stream", "/v1/tokenize", json!({ "messages": [user("hi")], "stream": true }), "invalid_request_error", Some("stream")),
        (
            "an image",
            "/v1/tokenize",
            json!({ "messages": [{ "role": "user", "content": [
                { "type": "text", "text": "what is this" },
                { "type": "image_url", "image_url": { "url": "https://example.invalid/a.png" } },
            ] }] }),
            "media_not_countable",
            Some("messages"),
        ),
        ("an id out of range", "/v1/detokenize", json!({ "token_ids": [104, 105, 256] }), "invalid_request_error", Some("token_ids")),
    ] {
        let (status, answer) = post(&h.app, path, body).await;
        assert_eq!(status, 400, "{name}: {answer}");
        assert_eq!(answer["error"]["code"], code, "{name}: {answer}");
        assert_eq!(answer["error"]["type"], "invalid_request_error", "{name}");
        if let Some(param) = param {
            assert_eq!(answer["error"]["param"], param, "{name}");
        }
    }
}

#[tokio::test]
async fn an_out_of_vocabulary_id_is_named_by_its_index() {
    let h = harness();
    let (status, answer) = post(&h.app, "/v1/detokenize", json!({ "token_ids": [104, 300, 500] })).await;
    assert_eq!(status, 400);
    let message = answer["error"]["message"].as_str().unwrap();
    assert!(message.contains("token_ids[1]") && message.contains("300"), "{message}");
    assert!(!message.contains("500"), "only the first offender is named: {message}");
}

#[tokio::test]
async fn an_empty_id_array_answers_an_empty_string() {
    let h = harness();
    let (status, answer) = post(&h.app, "/v1/detokenize", json!({ "token_ids": [] })).await;
    assert_eq!(status, 200, "{answer}");
    assert_eq!(answer["text"], "");
}

// ── media ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn an_image_is_refused_on_a_vision_load_and_nothing_is_fetched_or_prepared() {
    let builds = Gated::new(true, processor(media_limits()));
    let h = build(Box::new(ByteTemplate), |server| {
        server.with_media(Arc::new(MediaAcquirer::new(builds.clone(), media_limits(), MediaPolicy::new(false, 0))))
    });
    let url = data_uri(&png(64, 64));
    let (status, answer) = post(
        &h.app,
        "/v1/tokenize",
        json!({ "messages": [{ "role": "user", "content": [
            { "type": "text", "text": "what is this" },
            { "type": "image_url", "image_url": { "url": url } },
        ] }] }),
    )
    .await;
    assert_eq!(status, 400, "{answer}");
    assert_eq!(answer["error"]["code"], "media_not_countable");
    assert!(
        answer["error"]["message"].as_str().unwrap().contains("send the request"),
        "the message says what to do instead: {answer}"
    );
    assert_eq!(builds.builds.load(Ordering::SeqCst), 0, "no image was decoded");
    assert!(h.submitted.lock().unwrap().is_empty());
}

fn media_limits() -> ignis_artifact::vision::ProcessorOptions {
    ignis_artifact::vision::ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 256 << 20,
        max_decoded_pixels: 1 << 24,
        max_raw_patches: 1 << 17,
        max_vision_tokens: 1 << 15,
    }
}

// ── free ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_tokenize_call_never_reaches_the_scheduler_and_is_counted_on_its_own_series() {
    let h = build(Box::new(ByteTemplate), Server::with_metrics);
    let metrics = h.metrics.clone().expect("metrics are on");
    let scrape = || async {
        let response = metrics.clone().oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap()).await.unwrap();
        String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap()
    };
    assert!(!scrape().await.contains("ignis_tokenize_requests_total"), "absent until used");

    for _ in 0..2 {
        post(&h.app, "/v1/tokenize", json!({ "messages": [user("hi")] })).await;
    }
    post(&h.app, "/v1/detokenize", json!({ "token_ids": [104, 105] })).await;

    assert!(h.submitted.lock().unwrap().is_empty(), "no RequestInput was submitted");
    let text = scrape().await;
    assert!(text.contains("ignis_tokenize_requests_total{route=\"tokenize\"} 2"), "{text}");
    assert!(text.contains("ignis_tokenize_requests_total{route=\"detokenize\"} 1"), "{text}");
    // Not a request in the lifecycle: no acceptance, no rejection.
    assert!(text.contains("ignis_requests_accepted_total 0"), "{text}");
    assert!(text.contains("ignis_requests_rejected_total{reason=\"full\"} 0"), "{text}");
}

// ── the surface ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn both_routes_sit_behind_the_key_and_answer_the_cors_preflight() {
    let h = build(Box::new(ByteTemplate), |server| server.with_api_key(ApiKey::new(KEY)));
    for path in ["/v1/tokenize", "/v1/detokenize"] {
        let (status, answer) = post(&h.app, path, json!({ "prompt": "hi", "token_ids": [] })).await;
        assert_eq!(status, 401, "{path}: {answer}");

        let preflight = Request::builder().method("OPTIONS").uri(path).body(Body::empty()).unwrap();
        let response = h.app.clone().oneshot(preflight).await.unwrap();
        assert_eq!(response.status(), 200, "{path}");
        assert_eq!(response.headers()["access-control-allow-origin"], "*");

        let keyed = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {KEY}"))
            .body(Body::from(json!({ "prompt": "hi", "token_ids": [] }).to_string()))
            .unwrap();
        assert_eq!(h.app.clone().oneshot(keyed).await.unwrap().status(), 200, "{path}");
    }
}

// ── the real tokenizer ──────────────────────────────────────────────────────

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// The same round trip on the artifact's own byte-level BPE, where a token
/// can end inside a UTF-8 character. Skips when the artifact is not at its
/// machine-local path (the convention `responses_real_template.rs` follows).
#[tokio::test]
async fn the_round_trip_holds_on_the_real_template_and_tokenizer() {
    if !Path::new(ARTIFACT).exists() {
        eprintln!("skip: {ARTIFACT} does not exist");
        return;
    }
    let reader = ignis_artifact::Reader::open(Path::new(ARTIFACT)).expect("open artifact");
    let frontend = ignis_artifact::FrontendSet::from_reader(&reader).expect("frontend set");
    let h = build(Box::new(ignis_server::artifact_template::ArtifactTemplateProvider::new(frontend)), |server| server);
    for text in ["plain ascii", "città — 日本語 — 🦀", "family 👨\u{200d}", "caf\u{e9}"] {
        round_trips(&h, json!({ "messages": [user(text)] })).await;
        round_trips(&h, json!({ "prompt": text })).await;
    }
    // The tokenizer normalizes to NFC before it splits, so the ids of a
    // decomposed "e" + accent spell the composed character: byte identity
    // holds for text the normalizer leaves alone, and the ids of anything
    // else are its normalized form's -- which is what `count` counts.
    let (_, decomposed) = post(&h.app, "/v1/tokenize", json!({ "prompt": "cafe\u{301}", "return_token_ids": true })).await;
    let (_, back) = post(&h.app, "/v1/detokenize", json!({ "token_ids": decomposed["token_ids"] })).await;
    assert_eq!(back["text"], "caf\u{e9}");
    let (status, answer) = post(&h.app, "/v1/detokenize", json!({ "token_ids": [1, u32::MAX] })).await;
    assert_eq!((status, answer["error"]["message"].as_str().is_some_and(|m| m.contains("token_ids[1]"))), (400, true), "{answer}");
    // Served, the same body reports the number the count promised.
    let body = json!({ "messages": [user("What is in the file?")], "tools": [tool()] });
    let (_, counted) = post(&h.app, "/v1/tokenize", body.clone()).await;
    let (_, served) = post(&h.app, "/v1/chat/completions", merged(&body, json!({ "max_tokens": 1 }))).await;
    assert_eq!(counted["count"], served["usage"]["prompt_tokens"]);
}
