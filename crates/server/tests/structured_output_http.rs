//! Structured output on chat completions (GitHub #587, #588, spec #287, ADR
//! 0089): `response_format` with `json_object` and with `json_schema`, and
//! beside tools (#592), end to end over the real router on `MockCompute`
//! (ADR 0006).
//!
//! The "model" follows a script ([`MockCompute::script`]) that misses on
//! purpose: a prose preamble, a code fence, a remark after the object, an
//! early stop, nothing but whitespace; against a schema, key names of its
//! own, an enum in the wrong case, a number out of range, a missing key, an
//! extra key, a string its pattern refuses. Each draw of the answer is checked
//! against the grammar before the next round consumes it, and a miss is
//! repaired. The mock's repair ranks what the script was going to write next
//! above the rest, then the vocabulary in order; the test vocabulary puts
//! the tokens that close a JSON value first, so a repair with nothing better
//! to go on moves the answer toward its end, as a model that has to stop
//! would.
//!
//! A `json_schema` request's schema is described to the model after the
//! conversation (GitHub #589): the template double records what it was asked
//! to render, and can report the conversation's end as the checkpoint
//! boundary, as the real template does.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use ignis_core::compute::ModelFamily;
use ignis_core::mock::GatedCompute;
use ignis_core::{Compute, ConcreteScheduler, MockCompute, SchedulerConfig, TokenId, PendingToken};
use ignis_grammar::{GrammarCache, Vocab, VocabEntry, DEFAULT_CACHE_BYTES, DEFAULT_MAX_WS_RUN};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    described, ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions, THINKING_CLOSE_TEXT};
use ignis_server::Server;

const MODEL: &str = "test-model";
const CHAT: &str = "/v1/chat/completions";
const RESPONSES: &str = "/v1/responses";

#[path = "support/responses.rs"]
mod responses;

// ── the test vocabulary ─────────────────────────────────────────────────
//
// The lowest ids close a JSON value, in the order a stuck answer needs
// them: a string first (inside one every character is allowed), then a
// key's colon, an object, an array, and an object's opening at the root.
// Every printable ASCII character is also its own code point, and a line
// break is 10.

const QUOTE: TokenId = 0;
const COLON: TokenId = 1;
const CLOSE_OBJECT: TokenId = 2;
const CLOSE_ARRAY: TokenId = 3;
const OPEN_OBJECT: TokenId = 4;
/// The model's EOS: what ends a complete answer.
const STOP: TokenId = 5;
/// The reasoning block's close, one token as in the real tokenizer.
const THINK_END: TokenId = 6;
/// A code fence, one token as a real tokenizer has it.
const FENCE: TokenId = 7;
/// A tool call's opener, `<tool_call>`: one control token, as in the real
/// tokenizer, which no grammar accepts (#592).
const TOOL_CALL: TokenId = 8;
const NEWLINE: TokenId = 10;
const VOCAB: usize = 128;

/// What `token` reads as: its text, or `None` for a control token.
fn text_of(token: TokenId) -> Option<String> {
    match token {
        QUOTE => Some("\"".into()),
        COLON => Some(":".into()),
        CLOSE_OBJECT => Some("}".into()),
        CLOSE_ARRAY => Some("]".into()),
        OPEN_OBJECT => Some("{".into()),
        FENCE => Some("```".into()),
        NEWLINE => Some("\n".into()),
        32..=126 => char::from_u32(token).map(String::from),
        _ => None,
    }
}

/// `text` in the test vocabulary: `</think>`, `<tool_call>` and a fence one
/// token each, every other character its own code point.
fn encode(text: &str) -> Vec<TokenId> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix("</think>") {
            ids.push(THINK_END);
            rest = after;
        } else if let Some(after) = rest.strip_prefix("<tool_call>") {
            ids.push(TOOL_CALL);
            rest = after;
        } else if let Some(after) = rest.strip_prefix("```") {
            ids.push(FENCE);
            rest = after;
        } else {
            ids.push(if c == '\n' { NEWLINE } else { c as TokenId });
            rest = &rest[c.len_utf8()..];
        }
    }
    ids
}

/// The template over the test vocabulary: the placeholder's prompt, the
/// vocabulary's text back out, and its grammars.
#[derive(Clone)]
struct Vocabulary {
    grammars: Arc<GrammarCache>,
    /// Every render, in order: the schema description it was asked to put
    /// after the conversation, and the prompt it made.
    renders: Arc<Mutex<Vec<Render>>>,
    /// Report the conversation's end as the real provider does (GitHub
    /// #589): after the conversation's words come the description's, then a
    /// generation opener, and the checkpoint boundary is where the
    /// conversation ends.
    boundaries: bool,
    /// The tokenizer writes `<tool_call>` a character a token, as one with
    /// no such special token would.
    split_tool_call: bool,
}

/// One render the template made.
#[derive(Clone, Debug)]
struct Render {
    description: Option<String>,
    tokens: Vec<TokenId>,
}

/// The generation opener [`Vocabulary::with_boundaries`] ends a prompt with:
/// no text of the vocabulary.
const GENERATION_OPENER: TokenId = 11;

impl Vocabulary {
    fn new() -> Self {
        let entries = (0..VOCAB as TokenId)
            .map(|id| match (id, text_of(id)) {
                (STOP | THINK_END | TOOL_CALL, _) => VocabEntry::Control,
                (_, Some(text)) => VocabEntry::Text(text.into_bytes()),
                (_, None) => VocabEntry::Absent,
            })
            .collect();
        Self {
            grammars: Arc::new(GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES)),
            renders: Arc::default(),
            boundaries: false,
            split_tool_call: false,
        }
    }

    /// [`Vocabulary::new`], reporting where the conversation ends.
    fn with_boundaries() -> Self {
        Self { boundaries: true, ..Self::new() }
    }

    fn renders(&self) -> Vec<Render> {
        self.renders.lock().unwrap().clone()
    }

    /// The prompt of `messages`, with `description` after them when there is
    /// one, as the placeholder renders them.
    fn render(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        description: Option<&str>,
    ) -> Result<RenderedPrompt, TemplateRejection> {
        let conversation = SimpleTemplateProvider.apply_chat_template(messages, options, tools)?;
        let mut rendered = match description {
            Some(description) => SimpleTemplateProvider.apply_chat_template(&described(messages, description), options, tools)?,
            None => conversation.clone(),
        };
        if self.boundaries {
            rendered.opener_tokens = Some(conversation.tokens.len() as u32);
            rendered.tokens.push(GENERATION_OPENER);
        }
        self.renders
            .lock()
            .unwrap()
            .push(Render { description: description.map(str::to_owned), tokens: rendered.tokens.clone() });
        Ok(rendered)
    }
}

impl TemplateProvider for Vocabulary {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        self.render(messages, options, tools, None)
    }

    fn apply_chat_template_described(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
        description: &str,
    ) -> Result<RenderedPrompt, TemplateRejection> {
        self.render(messages, options, tools, Some(description))
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        let opener: Vec<TokenId> = "<tool_call>".chars().map(|c| c as TokenId).collect();
        match self.split_tool_call {
            true => Some(text.split("<tool_call>").map(encode).collect::<Vec<_>>().join(&opener[..])),
            false => Some(encode(text)),
        }
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens.iter().map(|&t| Decoder.push(t)).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder)
    }

    fn grammars(&self) -> Option<Arc<GrammarCache>> {
        Some(self.grammars.clone())
    }
}

struct Decoder;

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        match token {
            THINK_END => "</think>".into(),
            TOOL_CALL => "<tool_call>".into(),
            STOP => "<stop>".into(),
            token => text_of(token).unwrap_or_else(|| "?".into()),
        }
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server whose first request writes `script`, then stops.
fn app(script: &str) -> (axum::Router, Arc<MockCompute>) {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(script));
    (server(mock.clone(), SchedulerConfig::default()).app(), mock)
}

fn server(compute: Arc<dyn Compute>, config: SchedulerConfig) -> Server {
    server_over(compute, config, Vocabulary::new())
}

/// A server whose template is `vocabulary`, so a test can watch its grammars.
fn server_over(compute: Arc<dyn Compute>, config: SchedulerConfig, vocabulary: Vocabulary) -> Server {
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..config }, compute);
    Server::new(Engine::new(Box::new(scheduler)), Box::new(vocabulary))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
}

/// A chat request for a JSON object, thinking off, greedy.
fn chat(stream: bool, max_tokens: u32) -> JsonValue {
    json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "describe a cat" }],
        "response_format": { "type": "json_object" },
        "enable_thinking": false,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    })
}

async fn post(app: &axum::Router, body: &JsonValue) -> (u16, Option<String>, String) {
    post_text(app, &body.to_string()).await
}

/// `body` as written: a schema's keys keep their order.
async fn post_text(app: &axum::Router, body: &str) -> (u16, Option<String>, String) {
    post_to(app, CHAT, body).await
}

/// [`post_text`] to `path`.
async fn post_to(app: &axum::Router, path: &str, body: &str) -> (u16, Option<String>, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let stopped_by = response.headers().get("ignis-stopped-by").map(|v| v.to_str().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, stopped_by, String::from_utf8(bytes.to_vec()).unwrap())
}

/// One answer: its content, its reasoning, its finish reason.
#[derive(Debug, PartialEq)]
struct Answer {
    content: String,
    reasoning: Option<String>,
    finish: String,
}

/// The answer to `body`, whole or streamed as `body["stream"]` says.
async fn answer(app: &axum::Router, body: &JsonValue) -> Answer {
    answer_text(app, &body.to_string(), body["stream"] == true).await
}

/// The answer to `body` sent as written, streamed when `stream`.
async fn answer_text(app: &axum::Router, body: &str, stream: bool) -> Answer {
    let (status, _, text) = post_text(app, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    if stream {
        let chunks: Vec<JsonValue> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let delta = |field: &str| -> String {
            chunks.iter().filter_map(|c| c["choices"][0]["delta"][field].as_str()).collect()
        };
        let finish = chunks.iter().find_map(|c| c["choices"][0]["finish_reason"].as_str()).expect("a finish chunk");
        let reasoning = delta("reasoning_content");
        return Answer {
            content: delta("content"),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            finish: finish.to_owned(),
        };
    }
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    let message = &v["choices"][0]["message"];
    assert!(message.as_object().unwrap().contains_key("refusal") && message["refusal"].is_null(), "refusal is null: {text}");
    Answer {
        content: message["content"].as_str().unwrap().to_owned(),
        reasoning: message["reasoning_content"].as_str().map(str::to_owned),
        finish: v["choices"][0]["finish_reason"].as_str().unwrap().to_owned(),
    }
}

/// The content as a JSON object, or a panic naming it.
fn object(content: &str) -> serde_json::Map<String, JsonValue> {
    match serde_json::from_str::<JsonValue>(content) {
        Ok(JsonValue::Object(object)) => object,
        other => panic!("not a JSON object: {content:?} ({other:?})"),
    }
}

/// The longest run of JSON whitespace in `text`.
fn longest_whitespace(text: &str) -> usize {
    text.split(|c: char| !matches!(c, ' ' | '\n' | '\t' | '\r')).map(str::len).max().unwrap_or(0)
}

// ── every answer is a valid object (acceptance 1, 4) ────────────────────

#[tokio::test]
async fn a_preamble_a_fence_a_remark_and_an_early_stop_each_come_back_a_valid_object() {
    let cases: [(&str, &str, JsonValue); 4] = [
        ("a prose preamble", "Sure! Here it is:\n{\"a\": 1}", json!({ "a": 1 })),
        ("a code fence", "```json\n{\"a\": 1}\n```", json!({ "a": 1 })),
        ("a remark after the object", "{\"a\": 1}\nThat is all.", json!({ "a": 1 })),
        ("an early stop", "{\"a\": [1, 2", json!({ "a": [1, 2] })),
    ];
    for (what, script, expected) in cases {
        let (app, mock) = app(script);
        let whole = answer(&app, &chat(false, 64)).await;
        assert_eq!(whole.finish, "stop", "{what}: {whole:?}");
        assert_eq!(JsonValue::Object(object(&whole.content)), expected, "{what}: {whole:?}");
        assert!(!mock.repair_calls().is_empty(), "{what}: the scripted misses were repaired");
        let (app, _) = self::app(script);
        let streamed = answer(&app, &chat(true, 64)).await;
        assert_eq!(streamed, whole, "{what}: streamed and whole carry the same answer");
    }
}

/// Speculation stays on for a held answer (GitHub #590): over a mock that
/// commits runs as verify rounds do, each run is checked before it is
/// committed and cut at its first miss, and every answer, its repairs and
/// its finish are the ones the same model gives one token a round.
#[tokio::test]
async fn with_speculation_each_answer_is_the_one_without() {
    let scripts = [
        "Sure! Here it is:
{\"a\": 1}",
        "```json
{\"a\": 1}
```",
        "{\"a\": 1}
That is all.",
        "{\"a\": [1, 2",
        "{\"name\": \"Tom\", \"legs\": 4x, \"toys\": [1, 2]}",
    ];
    for script in scripts {
        let served = |mock: MockCompute| {
            let mock = Arc::new(mock);
            mock.script(0, &encode(script));
            (server(mock.clone(), SchedulerConfig::default()).app(), mock)
        };
        let (app, plain) = served(MockCompute::new());
        let without = answer(&app, &chat(false, 64)).await;
        let (app, runs) = served(MockCompute::with_runs(&[4, 2, 6]));
        let with = answer(&app, &chat(false, 64)).await;
        assert_eq!(with, without, "{script:?}: the same answer with speculation");
        assert_eq!(runs.repair_calls().len(), plain.repair_calls().len(), "{script:?}: the same repairs");
        assert!(!runs.cut_calls().is_empty(), "{script:?}: the held lane's runs went to the host's cut");
    }
}

/// A model that writes a valid object is never repaired: what it wrote is
/// what comes back, byte for byte.
#[tokio::test]
async fn an_answer_that_never_misses_comes_back_as_written() {
    let (app, mock) = app("{\"name\": \"Tom\", \"legs\": 4}");
    let whole = answer(&app, &chat(false, 64)).await;
    assert_eq!(whole.content, "{\"name\": \"Tom\", \"legs\": 4}");
    assert_eq!(whole.finish, "stop");
    assert!(mock.repair_calls().is_empty(), "nothing to repair");
}

/// The prompt here never says "JSON", and the request is served all the same.
#[tokio::test]
async fn the_prompt_need_not_mention_json() {
    let (app, _) = app("{}");
    let whole = answer(&app, &chat(false, 8)).await;
    assert_eq!(whole.content, "{}");
}

// ── whitespace stays bounded (acceptance 5) ─────────────────────────────

#[tokio::test]
async fn a_model_that_writes_only_whitespace_never_streams_an_endless_run() {
    let script = " ".repeat(400);
    let (app, mock) = app(&script);
    let whole = answer(&app, &chat(false, 100)).await;
    assert!(
        whole.finish == "length" || serde_json::from_str::<JsonValue>(&whole.content).is_ok_and(|v| v.is_object()),
        "a valid object or a cut: {whole:?}"
    );
    assert!(
        longest_whitespace(&whole.content) <= DEFAULT_MAX_WS_RUN as usize,
        "no run of whitespace past the bound: {:?}",
        whole.content
    );
    assert!(whole.content.trim_start().starts_with('{'), "the object opened once the run was spent: {whole:?}");
    assert!(!mock.repair_calls().is_empty());
}

// ── cuts (acceptance 3) ──────────────────────────────────────────────────

#[tokio::test]
async fn a_cut_by_max_tokens_returns_the_content_as_generated_with_length() {
    for stream in [false, true] {
        let (app, _) = app("{\"a\": \"hello world\"}");
        let cut = answer(&app, &chat(stream, 6)).await;
        assert_eq!(cut.finish, "length", "{cut:?}");
        assert_eq!(cut.content, "{\"a\": ", "the six tokens generated, unrepaired: {cut:?}");
    }
}

#[tokio::test]
async fn a_passed_deadline_answers_with_the_object_so_far_as_a_length_cut() {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode("{\"a\": 1, \"b\": 2}"));
    let (gated, gate) = GatedCompute::new(mock as Arc<dyn Compute>);
    // The fourth round waits at the gate past the deadline.
    gated.arm_at(0, 3);
    let app = server(gated, SchedulerConfig { max_in_flight: 1, ..SchedulerConfig::default() })
        .with_request_timeout(Duration::from_secs(1))
        .app();
    let (status, stopped_by, text) = post(&app, &chat(false, 64)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(stopped_by.as_deref(), Some("request_timeout"), "ADR 0058's partial answer");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{text}");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!content.is_empty() && "{\"a\": 1, \"b\": 2}".starts_with(content), "the answer so far: {content:?}");
    drop(gate);
}

// ── thinking (acceptance 2) ──────────────────────────────────────────────

/// The reasoning is the model's own, unchecked; the answer after its
/// `</think>` is held.
#[tokio::test]
async fn with_thinking_on_the_reasoning_is_free_and_the_answer_after_it_is_held() {
    let script = "no JSON here!</think>\n\nAnswer: {\"a\": 1}";
    let mut whole = None;
    for stream in [false, true] {
        let (app, mock) = app(script);
        let mut body = chat(stream, 64);
        body["enable_thinking"] = json!(true);
        let got = answer(&app, &body).await;
        assert_eq!(got.reasoning.as_deref(), Some("no JSON here!"), "the reasoning, untouched: {got:?}");
        assert_eq!(JsonValue::Object(object(&got.content)), json!({ "a": 1 }), "{got:?}");
        assert_eq!(got.finish, "stop");
        let repaired = mock.repair_calls().len();
        assert_eq!(repaired, "Answer:".len(), "only the answer's misses are repaired");
        match &whole {
            None => whole = Some(got),
            Some(whole) => assert_eq!(&got, whole, "streamed and whole carry the same answer"),
        }
    }
}

/// A thinking budget's forced close is followed by a held answer: the close
/// is the server's text and passes untouched, the model's answer after it
/// is checked (spec #287 story 23).
#[tokio::test]
async fn after_a_thinking_budgets_forced_close_the_answer_is_held() {
    const BUDGET: usize = 4;
    let close = encode(THINKING_CLOSE_TEXT);
    // The model reasons past its budget; whatever it would write where the
    // close is forced is overruled, and from about where the close ends it
    // writes spaces, a word the grammar refuses, and its object.
    let mut script = "x".repeat(BUDGET + close.len() - 4);
    script.push_str(&" ".repeat(12));
    script.push_str("ok {\"a\": 1}");
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(&script));
    let config = SchedulerConfig {
        thinking_close: Some(Arc::new(ignis_core::thinking_budget::ThinkingClose::new(close, THINK_END).unwrap())),
        ..SchedulerConfig::default()
    };
    let app = server(mock.clone(), config).app();
    let mut body = chat(false, 2200);
    body["enable_thinking"] = json!(true);
    body["thinking_budget"] = json!(BUDGET);
    let (status, _, text) = post(&app, &body).await;
    assert_eq!(status, 200, "{text}");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert!(v["choices"][0]["thinking_budget_forced_at"].is_u64(), "the close was forced: {text}");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(JsonValue::Object(object(content)), json!({ "a": 1 }), "{text}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{text}");
    assert_eq!(mock.repair_calls().len(), 2, "the word after the close was held to the grammar");
}

// ── refusals (acceptance 6, the field contract) ──────────────────────────

#[tokio::test]
async fn flash_next_refuses_structured_output_naming_the_model() {
    let mock = Arc::new(MockCompute::new());
    let app = server(mock, SchedulerConfig::default()).with_family(ModelFamily::FlashNext).app();
    let (status, _, text) = post(&app, &chat(false, 8)).await;
    assert_eq!(status, 400, "{text}");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["code"], "model_unsupported", "{text}");
    assert_eq!(v["error"]["param"], "response_format", "{text}");
    assert!(v["error"]["message"].as_str().unwrap().contains("Qwen3.8-Flash-Next"), "{text}");
}

/// #594: a client that falls back when the model does not serve structured
/// output (the Playground's run_js check) decides by the refusal's code, which
/// is the same on chat completions, on `/v1/responses` over HTTP and over its
/// WebSocket mode, where the message's words differ (`response_format`,
/// `text.format`).
#[tokio::test]
async fn flash_next_refuses_text_format_with_the_same_code_over_http_and_the_socket() {
    let flash_next = || server(Arc::new(MockCompute::new()), SchedulerConfig::default()).with_family(ModelFamily::FlashNext);
    let body = responses_body(r#"{"type":"json_object"}"#, false, 8);
    let (status, _, text) = post_to(&flash_next().app(), RESPONSES, &body).await;
    assert_eq!(status, 400, "{text}");
    let refused: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!((&refused["error"]["code"], &refused["error"]["param"]), (&json!("model_unsupported"), &json!("text.format")), "{text}");

    let live = responses::live(flash_next()).await;
    let mut socket = responses::socket(&live).await;
    responses::send_text(&mut socket, create_frame(&body)).await;
    let events = responses::events_until(&mut socket, responses::terminal).await;
    let error = events.last().unwrap();
    assert_eq!(error["type"], "error", "{error}");
    assert_eq!(error["status"], 400, "{error}");
    assert_eq!(error["error"]["code"], "model_unsupported", "{error}");
    assert!(error["error"]["message"].as_str().unwrap().starts_with("text.format:"), "{error}");
}

#[tokio::test]
async fn any_other_format_is_refused_and_text_is_a_free_answer() {
    let (app, _) = app("{}");
    for format in [json!({ "type": "grammar" }), json!("json_object"), json!({})] {
        let mut body = chat(false, 8);
        body["response_format"] = format.clone();
        let (status, _, text) = post(&app, &body).await;
        assert_eq!(status, 400, "{format}: {text}");
        assert!(text.contains("\"param\":\"response_format\""), "{format}: {text}");
    }
    // A free answer is whatever the model writes.
    let (app, mock) = self::app("Sure!");
    let mut body = chat(false, 8);
    body["response_format"] = json!({ "type": "text" });
    assert_eq!(answer(&app, &body).await.content, "Sure!");
    assert!(mock.repair_calls().is_empty());
}

/// A template whose vocabulary cannot be read as bytes holds nothing to a
/// grammar, and says so rather than serving a free answer as an object.
#[tokio::test]
async fn a_template_with_no_vocabulary_refuses_json_object() {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(5))
        .app();
    let (status, _, text) = post(&app, &chat(false, 8)).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("\"param\":\"response_format\""), "{text}");
}

// ── json_schema (#588) ───────────────────────────────────────────────────

/// A chat request whose answer is held to `schema`, the schema sent as
/// written: a `serde_json::Value` sorts an object's keys, and the answer
/// writes them in the schema's order.
fn schema_chat(schema: &str, strict: bool, stream: bool, max_tokens: u32) -> String {
    let mut body = chat(stream, max_tokens);
    body["response_format"] =
        json!({ "type": "json_schema", "json_schema": { "name": "answer", "strict": strict, "schema": "SCHEMA" } });
    body.to_string().replace("\"SCHEMA\"", schema)
}

/// The repository's root.
fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The Python that runs `tools/grammar-check/validate.py` with its pinned
/// `jsonschema`: `IGNIS_GRAMMAR_CHECK_PYTHON` (what CI sets), else the venv
/// `make grammar-venv` makes under `.scratch/grammar-venv`.
fn validator_python() -> Option<PathBuf> {
    if let Some(python) = std::env::var_os("IGNIS_GRAMMAR_CHECK_PYTHON") {
        return Some(PathBuf::from(python));
    }
    let venv = repo().join(".scratch/grammar-venv");
    [venv.join("Scripts/python.exe"), venv.join("bin/python")].into_iter().find(|python| python.exists())
}

/// Every answer, `(case, schema, text)`, valid under its schema by an
/// independent validator: `jsonschema` (Draft 2020-12, formats checked,
/// numbers as exact decimals) through `validate.py`. On CI (`CI` set, as
/// every CI service sets it) a missing validator fails the test, the owner's
/// call of 2026-11-13; on a machine without one, a skip that says so.
fn assert_independently_valid(answers: &[(&str, &str, String)]) {
    let Some(python) = validator_python() else {
        let missing = "no Python with tools/grammar-check/requirements.txt (IGNIS_GRAMMAR_CHECK_PYTHON, or `make grammar-venv`)";
        assert!(std::env::var_os("CI").is_none(), "{missing}: on CI the independent validator must run");
        eprintln!("skip: {missing}");
        return;
    };
    let rows: String = answers
        .iter()
        .map(|(case, schema, text)| {
            json!({ "case": case, "source": "served", "ours": "valid", "schema": schema, "text": text }).to_string() + "\n"
        })
        .collect();
    let file = std::env::temp_dir().join(format!("ignis-588-answers-{}-{}.jsonl", std::process::id(), answers[0].0.replace(' ', "-")));
    std::fs::write(&file, rows).unwrap();
    let output = std::process::Command::new(&python)
        .arg("-I")
        .arg(repo().join("tools/grammar-check/validate.py"))
        .arg(&file)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", python.display()));
    let _ = std::fs::remove_file(&file);
    let report = String::from_utf8_lossy(&output.stdout);
    eprintln!("{report}");
    assert!(output.status.success(), "the independent validator refuses an answer:\n{report}\n{}", String::from_utf8_lossy(&output.stderr));
    assert!(report.contains(&format!("ours valid, theirs valid: {}", answers.len())), "every answer was validated: {report}");
}

/// The keys of an answer's object (sorted: their order is checked on the
/// text).
fn keys(value: &JsonValue) -> Vec<&str> {
    value.as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default()
}

/// A model that misses a schema on purpose: what it writes, and what a
/// reader checks of the answer by hand.
struct Miss {
    what: &'static str,
    schema: &'static str,
    script: &'static str,
    holds: fn(&JsonValue) -> bool,
}

/// The ticket's misses (#588), each against a strict schema.
fn misses() -> Vec<Miss> {
    vec![
        Miss {
            what: "its own key names",
            schema: r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}},"required":["name","age"],"additionalProperties":false}"#,
            script: r#"{"nombre": "Tom", "edad": 4}"#,
            holds: |v| keys(v) == ["age", "name"] && v["name"].is_string() && v["age"].is_u64(),
        },
        Miss {
            what: "an enum in the wrong case",
            schema: r#"{"type":"object","properties":{"mood":{"type":"string","enum":["happy","sad"]}},"required":["mood"],"additionalProperties":false}"#,
            script: r#"{"mood": "Happy"}"#,
            holds: |v| keys(v) == ["mood"] && ["happy", "sad"].contains(&v["mood"].as_str().unwrap_or("")),
        },
        Miss {
            what: "numbers out of range",
            schema: r#"{"type":"object","properties":{"legs":{"type":"integer","minimum":0,"maximum":8},"weight":{"type":"number","exclusiveMinimum":0,"maximum":9.5,"multipleOf":0.25}},"required":["legs","weight"],"additionalProperties":false}"#,
            script: r#"{"legs": 12, "weight": 20.1}"#,
            holds: |v| {
                let weight = v["weight"].as_f64().unwrap_or(-1.0);
                v["legs"].as_u64().is_some_and(|legs| legs <= 8) && weight > 0.0 && weight <= 9.5 && (weight * 4.0).fract() == 0.0
            },
        },
        Miss {
            what: "a missing key",
            schema: r#"{"type":"object","properties":{"title":{"type":"string"},"done":{"type":"boolean"}},"required":["title","done"],"additionalProperties":false}"#,
            script: r#"{"title": "milk"}"#,
            holds: |v| keys(v) == ["done", "title"] && v["done"].is_boolean(),
        },
        Miss {
            what: "an extra key",
            schema: r#"{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"],"additionalProperties":false}"#,
            script: r#"{"city": "Oslo", "population": 709000, "country": "Norway"}"#,
            holds: |v| keys(v) == ["city", "country"] && v["city"] == "Oslo" && v["country"].is_string(),
        },
        Miss {
            what: "a string its pattern refuses",
            schema: r#"{"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}-[0-9]{4}$"}},"required":["code"],"additionalProperties":false}"#,
            script: r#"{"code": "abc-12"}"#,
            holds: |v| {
                let code = v["code"].as_str().unwrap_or("").as_bytes();
                code.len() == 8
                    && code[..3].iter().all(u8::is_ascii_uppercase)
                    && code[3] == b'-'
                    && code[4..].iter().all(u8::is_ascii_digit)
            },
        },
        Miss {
            what: "a date in words",
            schema: r#"{"type":"object","properties":{"when":{"type":"string","format":"date"}},"required":["when"],"additionalProperties":false}"#,
            script: r#"{"when": "next Friday"}"#,
            holds: |v| {
                let when = v["when"].as_str().unwrap_or("").as_bytes();
                when.len() == 10
                    && when.iter().enumerate().all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() })
            },
        },
    ]
}

/// Acceptance 1: whatever the model misses, the answer is valid under the
/// schema, streamed and whole alike, and the independent validator agrees.
#[tokio::test]
async fn a_model_that_misses_the_schema_is_repaired_into_an_answer_valid_under_it() {
    let mut answers = Vec::new();
    for miss in misses() {
        let (app, mock) = app(miss.script);
        let whole = answer_text(&app, &schema_chat(miss.schema, true, false, 256), false).await;
        assert_eq!(whole.finish, "stop", "{}: {whole:?}", miss.what);
        let value: JsonValue =
            serde_json::from_str(&whole.content).unwrap_or_else(|e| panic!("{}: not JSON ({e}): {:?}", miss.what, whole.content));
        eprintln!("{}: {:?}", miss.what, whole.content);
        assert!((miss.holds)(&value), "{}: {}", miss.what, whole.content);
        assert!(!mock.repair_calls().is_empty(), "{}: the scripted misses were repaired", miss.what);
        let (app, _) = self::app(miss.script);
        let streamed = answer_text(&app, &schema_chat(miss.schema, true, true, 256), true).await;
        assert_eq!(streamed, whole, "{}: streamed and whole carry the same answer", miss.what);
        answers.push((miss.what, miss.schema, whole.content));
    }
    assert_independently_valid(&answers);
}

/// Keys come in the schema's order whatever order the model writes them in,
/// and a model that writes them so is never repaired: the schema reaches the
/// grammar as sent.
#[tokio::test]
async fn keys_come_in_the_schemas_order() {
    const SCHEMA: &str = r#"{"type":"object","properties":{"zeta":{"type":"integer"},"alpha":{"type":"integer"}},"required":["zeta","alpha"],"additionalProperties":false}"#;
    let (app, _) = app(r#"{"alpha": 1, "zeta": 2}"#);
    let got = answer_text(&app, &schema_chat(SCHEMA, true, false, 128), false).await;
    let (zeta, alpha) = (got.content.find("\"zeta\""), got.content.find("\"alpha\""));
    assert!(got.content.starts_with("{\"zeta\"") && alpha > zeta, "{got:?}");
    let written = r#"{"zeta": 1, "alpha": 2}"#;
    let (app, mock) = self::app(written);
    let got = answer_text(&app, &schema_chat(SCHEMA, true, false, 128), false).await;
    assert_eq!(got.content, written);
    assert_eq!(got.finish, "stop");
    assert!(mock.repair_calls().is_empty(), "nothing to repair");
    assert_independently_valid(&[("schema order", SCHEMA, got.content)]);
}

/// Acceptance 3: without `strict`, a keyword outside the subset is served
/// and not held, and the guide's generated reference lists it as inert;
/// under `strict` it is refused by name.
#[tokio::test]
async fn without_strict_a_keyword_outside_the_subset_is_served_inert_and_the_guide_lists_it() {
    const SCHEMA: &str = r#"{"type":"object","properties":{"name":{"type":"string","minLength":3}},"required":["name"],"additionalProperties":false}"#;
    let (app, mock) = app(r#"{"name": "Al"}"#);
    let got = answer_text(&app, &schema_chat(SCHEMA, false, false, 64), false).await;
    assert_eq!(got.content, r#"{"name": "Al"}"#, "minLength is not held: {got:?}");
    assert!(mock.repair_calls().is_empty());
    let (status, _, text) = post_text(&app, &schema_chat(SCHEMA, true, false, 64)).await;
    assert_eq!(status, 400, "{text}");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["code"], "unsupported_keyword", "{text}");
    assert!(v["error"]["message"].as_str().unwrap().contains("`minLength`"), "{text}");
    let page = ignis_server::user_docs::json_schema();
    let inert = page.split("## Inert without `strict`").nth(1).and_then(|rest| rest.split("\n## ").next()).expect("the inert section");
    assert!(inert.contains("`minLength`"), "the guide lists minLength as inert: {inert}");
}

/// The owner's call of 2026-11-13 on spec #287: without `strict`, a
/// `pattern` that is not a regular language and a `$ref` that is not local
/// are served inert, as a keyword outside the subset is, and only `strict`
/// refuses them (by name, in the test below): the answer is held to the rest
/// of the schema.
#[tokio::test]
async fn without_strict_a_non_regular_pattern_and_a_non_local_reference_are_served_inert() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"string","pattern":"(a)\\1"},"b":{"$ref":"https://example.com/s.json"},"c":{"type":"integer"}},"required":["a","b","c"],"additionalProperties":false}"#;
    let script = r#"{"a": "xy", "b": 7, "c": x5}"#;
    let (app, _) = app(script);
    let got = answer_text(&app, &schema_chat(schema, false, false, 64), false).await;
    let value: JsonValue = serde_json::from_str(&got.content).unwrap_or_else(|e| panic!("not JSON ({e}): {got:?}"));
    assert_eq!((&value["a"], &value["b"]), (&json!("xy"), &json!(7)), "neither is held: {got:?}");
    assert!(value["c"].is_i64(), "the rest of the schema is held: {got:?}");
    let (status, _, text) = post_text(&app, &schema_chat(schema, true, false, 64)).await;
    assert_eq!(status, 400, "under strict: {text}");
}

/// Acceptance 2: every refusal the spec names is a 400 before anything is
/// generated, with the refusal's own code, `param` naming
/// `response_format`, and a message naming the keyword or the limit.
#[tokio::test]
async fn every_refusal_answers_400_naming_its_keyword_or_limit_before_generation() {
    let object = |inner: &str| {
        format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#)
    };
    let properties: Vec<String> = (0..=5000).map(|i| format!(r#""p{i}":{{"type":"null"}}"#)).collect();
    let nested = (0..=10).fold(r#"{"type":"string"}"#.to_owned(), |inner, _| object(&inner));
    let values: Vec<String> = (0..=1000).map(|i| format!(r#""v{i}""#)).collect();
    let cases: Vec<(&str, String, bool, &str, &str)> = vec![
        ("a keyword outside the subset", object(r#"{"type":"string","allOf":[]}"#), true, "unsupported_keyword", "allOf"),
        ("oneOf", object(r#"{"oneOf":[{"type":"string"}]}"#), true, "unsupported_keyword", "oneOf"),
        ("a root that is not an object", r#"{"type":"string"}"#.into(), true, "root_not_object", "type"),
        ("a root anyOf", r#"{"anyOf":[{"type":"object"}]}"#.into(), true, "root_anyof", "anyOf"),
        ("an open object", r#"{"type":"object","properties":{}}"#.into(), true, "additional_properties_not_false", "additionalProperties"),
        (
            "a property not required",
            r#"{"type":"object","properties":{"a":{"type":"string"}},"required":[],"additionalProperties":false}"#.into(),
            true,
            "not_all_required",
            "required",
        ),
        ("a non-regular pattern", object(r#"{"type":"string","pattern":"(a)\\1"}"#), true, "pattern_not_regular", "pattern"),
        ("a non-local reference", object(r#"{"$ref":"https://example.com/s.json"}"#), true, "unsupported_ref", "$ref"),
        (
            "no finite value",
            r##"{"type":"object","properties":{"child":{"$ref":"#"}},"required":["child"],"additionalProperties":false}"##.into(),
            true,
            "no_finite_value",
            "$ref",
        ),
        // OpenAI's four limits, with `strict` and without it.
        ("too many properties", format!(r#"{{"type":"object","properties":{{{}}}}}"#, properties.join(",")), false, "too_many_properties", "properties"),
        ("nested too deep", nested, true, "too_deeply_nested", "nesting"),
        ("too many enum values", object(&format!(r#"{{"type":"string","enum":[{}]}}"#, values.join(","))), false, "too_many_enum_values", "enum values"),
        ("too many characters", object(&format!(r#"{{"type":"string","enum":["{}"]}}"#, "x".repeat(120_001))), true, "too_many_characters", "characters"),
    ];
    for (what, schema, strict, code, keyword) in cases {
        let mock = Arc::new(MockCompute::new());
        let app = server(mock.clone(), SchedulerConfig::default()).app();
        let (status, _, text) = post_text(&app, &schema_chat(&schema, strict, false, 64)).await;
        assert_eq!(status, 400, "{what}: {}", &text[..text.len().min(400)]);
        let v: JsonValue = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["code"], code, "{what}: {text}");
        assert_eq!(v["error"]["param"], "response_format", "{what}: {text}");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.contains(&format!("`{keyword}`")), "{what}: the message names `{keyword}`: {message}");
        assert!(mock.prefill_calls().is_empty() && mock.decode_calls().is_empty(), "{what}: nothing was generated");
    }
}

/// A `json_schema` format whose members are not what OpenAI's are is a 400
/// naming `response_format`, before anything is generated.
#[tokio::test]
async fn a_json_schema_format_without_a_schema_is_refused_naming_response_format() {
    let cases = [
        ("no json_schema object", json!({ "type": "json_schema" }), "json_schema object"),
        ("json_schema not an object", json!({ "type": "json_schema", "json_schema": "x" }), "must be an object"),
        ("no schema", json!({ "type": "json_schema", "json_schema": { "name": "x" } }), "`schema` is required"),
        (
            "strict not a boolean",
            json!({ "type": "json_schema", "json_schema": { "name": "x", "schema": { "type": "object" }, "strict": "yes" } }),
            "`strict` must be a boolean",
        ),
        ("name not a string", json!({ "type": "json_schema", "json_schema": { "name": 3, "schema": { "type": "object" } } }), "`name` must be a string"),
    ];
    for (what, format, says) in cases {
        let mock = Arc::new(MockCompute::new());
        let app = server(mock.clone(), SchedulerConfig::default()).app();
        let mut body = chat(false, 8);
        body["response_format"] = format;
        let (status, _, text) = post(&app, &body).await;
        assert_eq!(status, 400, "{what}: {text}");
        let v: JsonValue = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["param"], "response_format", "{what}: {text}");
        assert!(v["error"]["message"].as_str().unwrap().contains(says), "{what}: {text}");
        assert!(mock.prefill_calls().is_empty(), "{what}: nothing was generated");
    }
}

/// Acceptance 4: a schema sent again, its whitespace changed or not, is
/// compiled once; the same schema with another `strict` is another grammar.
#[tokio::test]
async fn a_schema_sent_again_is_not_compiled_again() {
    const SCHEMA: &str = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    const SPACED: &str = r#"{ "type": "object", "properties": { "ok": { "type": "boolean" } }, "required": [ "ok" ], "additionalProperties": false }"#;
    let vocabulary = Vocabulary::new();
    let grammars = vocabulary.grammars.clone();
    let app = server_over(Arc::new(MockCompute::new()), SchedulerConfig::default(), vocabulary).app();
    let before = grammars.compiles();
    for schema in [SCHEMA, SPACED, SCHEMA] {
        let (status, _, text) = post_text(&app, &schema_chat(schema, true, false, 16)).await;
        assert_eq!(status, 200, "{text}");
    }
    assert_eq!(grammars.compiles() - before, 1, "three requests, one compile");
    let (status, _, text) = post_text(&app, &schema_chat(SCHEMA, false, false, 16)).await;
    assert_eq!(status, 200, "{text}");
    assert_eq!(grammars.compiles() - before, 2, "without strict, a grammar of its own");
}

// ── the schema description (#589) ────────────────────────────────────────

/// A strict schema, sent with spaces and its keys out of alphabetical order.
const CAT: &str = r#"{ "type": "object", "properties": { "name": { "type": "string" }, "age": { "type": "integer" } }, "required": ["name", "age"], "additionalProperties": false }"#;

/// What the model is told of [`CAT`] under a format whose `description` is
/// "A cat.": that description, then the spike's measured sentence and the
/// schema, compact, its keys in the order sent.
const CAT_DESCRIBED: &str = "A cat.\n\nRespond with a JSON object that matches this JSON schema:\n\
{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":[\"name\",\"age\"],\"additionalProperties\":false}";

/// A chat request for [`CAT`], its format described as "A cat.", with
/// `extra` fields beside it.
fn cat_chat(extra: JsonValue) -> String {
    let mut body = chat(false, 16);
    body["response_format"] = json!({
        "type": "json_schema",
        "json_schema": { "name": "cat", "description": "A cat.", "strict": true, "schema": "SCHEMA" }
    });
    for (field, value) in extra.as_object().unwrap() {
        body[field] = value.clone();
    }
    body.to_string().replace("\"SCHEMA\"", CAT)
}

/// The one render a server over a fresh vocabulary made for `body`.
async fn render_of(body: &str, server: impl FnOnce(Vocabulary) -> Server) -> Render {
    let vocabulary = Vocabulary::new();
    let app = server(vocabulary.clone()).app();
    let (status, _, text) = post_text(&app, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    let renders = vocabulary.renders();
    assert_eq!(renders.len(), 1, "{renders:?}");
    renders[0].clone()
}

fn over_mock(vocabulary: Vocabulary) -> Server {
    server_over(Arc::new(MockCompute::new()), SchedulerConfig::default(), vocabulary)
}

/// The prompt the same conversation renders with no `response_format`.
async fn free_prompt() -> Vec<TokenId> {
    let mut body = chat(false, 16);
    body.as_object_mut().unwrap().remove("response_format");
    render_of(&body.to_string(), over_mock).await.tokens
}

#[tokio::test]
async fn a_json_schema_request_describes_its_schema_after_the_conversation() {
    let render = render_of(&cat_chat(json!({})), over_mock).await;
    assert_eq!(render.description.as_deref(), Some(CAT_DESCRIBED));
    // The conversation is the client's, as sent: the description comes after
    // its last message, never inside it.
    let free = free_prompt().await;
    assert!(render.tokens.starts_with(&free) && render.tokens.len() > free.len(), "{render:?}");
}

#[tokio::test]
async fn with_the_description_off_the_prompt_is_the_one_rendered_before() {
    let render = render_of(&cat_chat(json!({ "ignis_describe_schema": false })), over_mock).await;
    assert_eq!(render.description, None);
    assert_eq!(render.tokens, free_prompt().await, "byte for byte the free request's prompt");
}

#[tokio::test]
async fn json_object_appends_nothing() {
    for describe in [json!(null), json!(true)] {
        let mut body = chat(false, 16);
        body["ignis_describe_schema"] = describe.clone();
        let render = render_of(&body.to_string(), over_mock).await;
        assert_eq!(render.description, None, "{describe}");
        assert_eq!(render.tokens, free_prompt().await, "{describe}");
    }
}

/// `--model-describe-schema false` turns the description off for a request
/// that says nothing, and a request's own field wins over it.
#[tokio::test]
async fn the_server_default_is_overridden_by_the_request() {
    let off = |vocabulary: Vocabulary| over_mock(vocabulary).with_describe_schema(false);
    assert_eq!(render_of(&cat_chat(json!({})), off).await.description, None);
    let asked = render_of(&cat_chat(json!({ "ignis_describe_schema": true })), off).await;
    assert_eq!(asked.description.as_deref(), Some(CAT_DESCRIBED));
    let on = render_of(&cat_chat(json!({ "ignis_describe_schema": null })), over_mock).await;
    assert_eq!(on.description.as_deref(), Some(CAT_DESCRIBED), "null is the server's default");
}

#[tokio::test]
async fn a_switch_that_is_not_a_boolean_is_refused_naming_it() {
    let mock = Arc::new(MockCompute::new());
    let app = server(mock.clone(), SchedulerConfig::default()).app();
    for value in [json!("false"), json!(0), json!({})] {
        let (status, _, text) = post_text(&app, &cat_chat(json!({ "ignis_describe_schema": value }))).await;
        assert_eq!(status, 400, "{value}: {text}");
        let v: JsonValue = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["param"], "ignis_describe_schema", "{text}");
        assert!(v["error"]["message"].as_str().unwrap().contains("boolean"), "{text}");
    }
    assert!(mock.prefill_calls().is_empty(), "nothing was generated");
}

/// Acceptance 2: the description sits after the conversation, so the next
/// turn resumes from the first turn's checkpoint, taken where its
/// conversation ended, as its reported cached tokens show.
#[tokio::test]
async fn the_next_turn_reuses_the_described_turns_conversation() {
    const SCHEMA: &str = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
    let mock = Arc::new(MockCompute::new());
    for request in 0..2 {
        mock.script(request, &encode("{\"ok\": true}"));
    }
    let vocabulary = Vocabulary::with_boundaries();
    let app = server_over(mock, SchedulerConfig::default(), vocabulary.clone()).app();
    // Twenty words: past one whole 16-token page, where a checkpoint is taken.
    let words = (0..20).map(|n| format!("w{n}")).collect::<Vec<_>>().join(" ");
    let turn = |messages: JsonValue| {
        json!({
            "model": MODEL, "messages": messages, "max_tokens": 16, "enable_thinking": false, "temperature": 0,
            "response_format": { "type": "json_schema", "json_schema": { "name": "ok", "strict": true, "schema": "SCHEMA" } }
        })
        .to_string()
        .replace("\"SCHEMA\"", SCHEMA)
    };
    let usage = |text: &str| serde_json::from_str::<JsonValue>(text).unwrap()["usage"].clone();
    let (status, _, first) = post_text(&app, &turn(json!([{ "role": "user", "content": words }]))).await;
    assert_eq!(status, 200, "{first}");
    let first = usage(&first);
    assert_eq!(first["prompt_tokens_details"]["cached_tokens"], 0);
    let (status, _, second) = post_text(
        &app,
        &turn(json!([
            { "role": "user", "content": words },
            { "role": "assistant", "content": "{\"ok\": true}" },
            { "role": "user", "content": "again" }
        ])),
    )
    .await;
    assert_eq!(status, 200, "{second}");
    let renders = vocabulary.renders();
    assert!(renders.iter().all(|render| render.description.is_some()), "both turns described: {renders:?}");
    assert!(first["prompt_tokens"].as_u64().unwrap() > 21, "the first prompt carries its description: {first}");
    assert_eq!(usage(&second)["prompt_tokens_details"]["cached_tokens"], 20, "the first turn's conversation, and only it");
}

// ── tools beside response_format (#592) ─────────────────────────────────

/// What the agent's last answer is held to.
const WEATHER: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"sky":{"type":"string","enum":["sunny","rainy"]}},"required":["city","sky"],"additionalProperties":false}"#;

/// A call in the template's dialect, as the model writes it: text no JSON
/// grammar would let through.
const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";

/// The user's question, the agent's first turn.
fn question() -> JsonValue {
    json!([{ "role": "user", "content": "What is the weather in Oslo?" }])
}

/// The conversation after the first turn's call and the tool's result.
fn after_the_call() -> JsonValue {
    json!([
        { "role": "user", "content": "What is the weather in Oslo?" },
        { "role": "assistant", "content": "", "tool_calls": [{ "id": "call_0", "type": "function",
            "function": { "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}" } }] },
        { "role": "tool", "tool_call_id": "call_0", "content": "sunny, 21 C" },
    ])
}

/// An agent's turn, as an SDK sends it: `messages`, a tool, `tool_choice`,
/// and `response_format` holding the answer to [`WEATHER`]; greedy.
fn agent_turn(messages: JsonValue, tool_choice: JsonValue, thinking: bool, stream: bool) -> String {
    let body = json!({
        "model": MODEL,
        "messages": messages,
        "tools": [{ "type": "function", "function": { "name": "get_weather", "parameters": {
            "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] } } }],
        "tool_choice": tool_choice,
        "response_format": { "type": "json_schema", "json_schema": { "name": "weather", "strict": true, "schema": "SCHEMA" } },
        "enable_thinking": thinking,
        "temperature": 0,
        "max_tokens": 256,
        "stream": stream,
    });
    body.to_string().replace("\"SCHEMA\"", WEATHER)
}

/// What a turn came back with: its content, its calls as (name,
/// arguments), its finish reason.
#[derive(Debug, PartialEq)]
struct Turn {
    content: String,
    calls: Vec<(String, JsonValue)>,
    finish: String,
}

/// The turn `body` asks for, whole or streamed.
async fn turn(app: &axum::Router, body: &str, stream: bool) -> Turn {
    let (status, _, text) = post_text(app, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    let call = |call: &JsonValue| {
        let function = &call["function"];
        let arguments = serde_json::from_str(function["arguments"].as_str().unwrap()).unwrap();
        (function["name"].as_str().unwrap().to_owned(), arguments)
    };
    if stream {
        let chunks: Vec<JsonValue> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let deltas = || chunks.iter().map(|c| &c["choices"][0]["delta"]);
        return Turn {
            content: deltas().filter_map(|d| d["content"].as_str()).collect(),
            calls: deltas().filter_map(|d| d["tool_calls"].as_array()).flatten().map(call).collect(),
            finish: chunks.iter().find_map(|c| c["choices"][0]["finish_reason"].as_str()).expect("a finish chunk").to_owned(),
        };
    }
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    let message = &v["choices"][0]["message"];
    Turn {
        content: message["content"].as_str().unwrap_or_default().to_owned(),
        calls: message["tool_calls"].as_array().map(|calls| calls.iter().map(call).collect()).unwrap_or_default(),
        finish: v["choices"][0]["finish_reason"].as_str().unwrap().to_owned(),
    }
}

/// `text` as the model writes it: with thinking on, after a reasoning block.
fn reasoned(thinking: bool, text: &str) -> String {
    match thinking {
        true => format!("Let me see.</think>\n\n{text}"),
        false => text.to_owned(),
    }
}

/// Acceptance 1: a turn that opens a tool call is released from the schema
/// and its call comes back as the model wrote it, with no miss and no
/// repair; the next turn, after the tool's result, is held again and answers
/// under the schema. With thinking on the call opens after the reasoning,
/// where the answer would start; with speculation, inside a verified run.
#[tokio::test]
async fn a_turn_that_opens_a_tool_call_is_released_and_the_next_turn_answers_under_the_schema() {
    for thinking in [false, true] {
        for runs in [false, true] {
            for stream in [false, true] {
                let case = format!("thinking {thinking}, runs {runs}, stream {stream}");
                let mock = Arc::new(if runs { MockCompute::with_runs(&[4, 2, 6]) } else { MockCompute::new() });
                mock.script(0, &encode(&reasoned(thinking, CALL)));
                mock.script(1, &encode(&reasoned(thinking, "```json\n{\"city\": \"Oslo\", \"sky\": \"sunny\"}\n```")));
                let app = server(mock.clone(), SchedulerConfig::default()).app();

                let first = turn(&app, &agent_turn(question(), json!("auto"), thinking, stream), stream).await;
                assert_eq!(first.calls, [("get_weather".to_owned(), json!({ "city": "Oslo" }))], "{case}: {first:?}");
                assert_eq!(first.finish, "tool_calls", "{case}: {first:?}");
                assert_eq!(first.content, "", "{case}: {first:?}");
                assert!(mock.repair_calls().is_empty(), "{case}: the call's text was never held to the schema");
                let rounds = mock.decode_calls().concat();
                let last = rounds.iter().filter(|job| job.request == 0).last().expect("the first turn decoded");
                assert!(!last.checked, "{case}: the released turn's rounds are no longer checked");
                // A verified run that opens the call loses no draft to it:
                // the opener is no miss, and nothing after it is checked.
                let cuts: Vec<_> = mock.cut_calls().into_iter().filter(|(request, ..)| *request == 0).collect();
                assert!(cuts.iter().all(|(_, shown, kept)| *kept == shown.len()), "{case}: {cuts:?}");
                if runs && thinking {
                    let opened = cuts.iter().any(|(_, shown, _)| shown[1..].contains(&PendingToken::Token(TOOL_CALL)));
                    assert!(opened, "{case}: a verified run opened the call: {cuts:?}");
                }

                let second = turn(&app, &agent_turn(after_the_call(), json!("auto"), thinking, stream), stream).await;
                assert!(second.calls.is_empty(), "{case}: {second:?}");
                assert_eq!(second.finish, "stop", "{case}: {second:?}");
                let answer: JsonValue = serde_json::from_str(&second.content).unwrap_or_else(|e| panic!("{case}: {e}: {second:?}"));
                assert_eq!(answer, json!({ "city": "Oslo", "sky": "sunny" }), "{case}");
                assert!(mock.repair_calls().iter().all(|repair| repair.request == 1), "{case}");
                assert!(!mock.repair_calls().is_empty(), "{case}: the second turn's fence was repaired");
                if runs {
                    assert!(mock.cut_calls().iter().any(|(request, ..)| *request == 1), "{case}: the held turn kept its runs");
                }
            }
        }
    }
}

/// A released turn's call counts no miss and no repair on the structured
/// series (#593); it is still a request held to a grammar.
#[tokio::test]
async fn a_released_turn_counts_no_miss() {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(CALL));
    let server = server(mock.clone(), SchedulerConfig::default()).with_metrics();
    let metrics = server.metrics_app().expect("metrics are on");
    let first = turn(&server.app(), &agent_turn(question(), json!("auto"), false, false), false).await;
    assert_eq!(first.finish, "tool_calls", "{first:?}");
    let series = |text: &str, name: &str| {
        let wanted = format!("{name}{{type=\"json_schema\"}} ");
        text.lines().find_map(|l| l.strip_prefix(&wanted)?.parse::<f64>().ok())
    };
    let mut scrape = String::new();
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
            let response = metrics.clone().oneshot(request).await.unwrap();
            scrape = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
            if series(&scrape, "ignis_structured_requests_total") == Some(1.0) && series(&scrape, "ignis_structured_misses_total").is_some() {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "the request never reached the series:\n{scrape}");
    assert_eq!(series(&scrape, "ignis_structured_misses_total"), Some(0.0), "{scrape}");
    assert_eq!(series(&scrape, "ignis_structured_repairs_total"), Some(0.0), "{scrape}");
}

/// Acceptance 2: a turn that answers instead of calling is held to the
/// schema from its first token. Once the answer has started a tool call can
/// no longer open: an opener there is a miss like any other, repaired into
/// the answer.
#[tokio::test]
async fn a_turn_that_answers_directly_returns_an_answer_valid_under_the_schema() {
    let scripts = [
        ("an enum in the wrong case", "{\"city\": \"Oslo\", \"sky\": \"Sunny\"}"),
        ("an opener inside the answer", "{\"city\": \"Oslo\", \"sky\": <tool_call>\"sunny\"}"),
    ];
    let mut answers = Vec::new();
    for (what, script) in scripts {
        for stream in [false, true] {
            let (app, mock) = app(script);
            let got = turn(&app, &agent_turn(question(), json!("auto"), false, stream), stream).await;
            assert!(got.calls.is_empty(), "{what}: {got:?}");
            assert_eq!(got.finish, "stop", "{what}: {got:?}");
            let answer: JsonValue = serde_json::from_str(&got.content).unwrap_or_else(|e| panic!("{what}: {e}: {got:?}"));
            assert_eq!(answer["city"], "Oslo", "{what}: {got:?}");
            assert!(["sunny", "rainy"].contains(&answer["sky"].as_str().unwrap_or("")), "{what}: {got:?}");
            assert!(!mock.repair_calls().is_empty(), "{what}: the miss was repaired");
            answers.push((what, WEATHER, got.content));
        }
    }
    assert_independently_valid(&answers);
}

/// Until the answer starts, a miss's repair may open the call: the opener is
/// in the permitted set beside the answer's first tokens, as masking every
/// step would have it. Once the answer has started it is not. The scripted
/// model writes a word before its object, each letter a miss before the
/// answer starts, and an enum in the wrong case inside it.
#[tokio::test]
async fn until_the_answer_starts_a_repair_may_open_the_call_and_after_it_may_not() {
    let (app, mock) = app("Here: {\"city\": \"Oslo\", \"sky\": \"Sunny\"}");
    let got = turn(&app, &agent_turn(question(), json!("auto"), false, false), false).await;
    assert!(got.calls.is_empty(), "{got:?}");
    assert!(serde_json::from_str::<JsonValue>(&got.content).is_ok(), "{got:?}");
    let opens: Vec<bool> = mock.repair_calls().iter().map(|repair| repair.permitted.permits(TOOL_CALL)).collect();
    assert_eq!(opens.first(), Some(&true), "a repair before the answer could open the call: {opens:?}");
    assert_eq!(opens.last(), Some(&false), "a repair inside the answer could not: {opens:?}");
    assert!(opens.windows(2).all(|pair| pair[0] >= pair[1]), "never again once the answer started: {opens:?}");
}

/// Acceptance 3: `tool_choice: "required"` or a named function forces the
/// call beside `response_format`, as without it (#286): the opening is the
/// server's, the model writes the rest, and the turn is a call, not held to
/// the schema. The model here was about to answer.
#[tokio::test]
async fn a_forced_tool_choice_beside_response_format_forces_the_call() {
    let cases = [
        (json!("required"), "<tool_call>\n<function", "=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>"),
        (
            json!({ "type": "function", "function": { "name": "get_weather" } }),
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
        ),
    ];
    for (tool_choice, opening, rest) in cases {
        for stream in [false, true] {
            let forced = encode(opening).len();
            let answering: String = "{\"city\": \"Oslo\", \"sky\": \"sunny\"}".chars().chain(std::iter::repeat(' ')).take(forced).collect();
            let (app, mock) = app(&format!("{answering}{rest}"));
            let got = turn(&app, &agent_turn(question(), tool_choice.clone(), false, stream), stream).await;
            assert_eq!(got.calls, [("get_weather".to_owned(), json!({ "city": "Oslo" }))], "{tool_choice}: {got:?}");
            assert_eq!(got.finish, "tool_calls", "{tool_choice}: {got:?}");
            assert!(mock.repair_calls().is_empty(), "{tool_choice}: the forced turn was not held to the schema");
        }
    }
}

/// A tokenizer whose `<tool_call>` is not one token cannot tell a call's
/// opener from the answer's first token, so tools a turn may call beside
/// structured output are refused there, naming `response_format`, rather
/// than offered and never callable. Without tools to call the request is
/// served.
#[tokio::test]
async fn tools_a_turn_may_call_on_a_tokenizer_with_no_single_token_opener_are_refused() {
    let vocabulary = Vocabulary { split_tool_call: true, ..Vocabulary::new() };
    let app = server_over(Arc::new(MockCompute::new()), SchedulerConfig::default(), vocabulary).app();
    let (status, _, text) = post_text(&app, &agent_turn(question(), json!("auto"), false, false)).await;
    assert_eq!(status, 400, "{text}");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["param"], "response_format", "{text}");
    assert!(v["error"]["message"].as_str().unwrap().contains("<tool_call>"), "{text}");
    let (status, _, text) = post_text(&app, &agent_turn(question(), json!("none"), false, false)).await;
    assert_eq!(status, 200, "{text}");
}

// ── ignis's own callers (#594) ──────────────────────────────────────────

/// An edit's rewrite on the loaded model: one condition image.
async fn edit_rewrite(script: &str) -> (Result<ignis_server::images::enhance::Rewritten, String>, Arc<MockCompute>) {
    use ignis_core::image::{ImageSize, RgbaImage};
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(script));
    let server = server(mock.clone(), SchedulerConfig::default());
    let image = RgbaImage { size: ImageSize::new(1, 1), pixels: vec![255, 0, 0, 255] };
    (ignis_server::images::enhance::rewrite(&server, "make it blue", &[image], 1).await, mock)
}

/// The edit enhancer's request carries a strict `json_schema`: a fenced
/// answer's first token is a miss, caught and repaired like any held answer.
/// And the answer, held, is the object itself: no fence is stripped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_edit_enhancer_holds_its_answer_to_a_schema_and_reads_it_as_is() {
    let (rewritten, mock) = edit_rewrite("{\"Rewritten\": \"Make the car blue.\"}").await;
    assert_eq!(rewritten.unwrap().prompt, "Make the car blue.");
    assert!(mock.repair_calls().is_empty(), "a conforming answer needs no repair");

    let (_, mock) = edit_rewrite("```json\n{\"Rewritten\": \"Make the car blue.\"}\n```").await;
    assert!(!mock.repair_calls().is_empty(), "the fence was refused by the grammar, so the request carried one");
}

/// #589 beside #592: a turn that may answer is told the schema; a forced
/// call's turn, which writes no answer, is not.
#[tokio::test]
async fn a_forced_calls_turn_is_not_told_the_schema() {
    for (tool_choice, described) in [(json!("auto"), true), (json!("required"), false)] {
        let render = render_of(&agent_turn(question(), tool_choice.clone(), false, false), over_mock).await;
        assert_eq!(render.description.is_some(), described, "{tool_choice}: {render:?}");
    }
}

// ── text.format on the Responses API (#591) ──────────────────────────────

/// A Responses request holding its answer to `format` (the `text.format`
/// object as JSON text, sent as written), thinking off, greedy.
fn responses_body(format: &str, stream: bool, max_output_tokens: u32) -> String {
    json!({
        "model": MODEL,
        "input": "describe a cat",
        "text": { "format": "FORMAT" },
        "enable_thinking": false,
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "stream": stream,
    })
    .to_string()
    .replace("\"FORMAT\"", format)
}

/// `body` as a `response.create` frame: the same text, a `type` first, so a
/// schema's keys keep the order they were sent in.
fn create_frame(body: &str) -> String {
    body.replacen('{', "{\"type\":\"response.create\",", 1)
}

/// `text.format` for a schema, its members flat beside the type.
fn schema_format_text(schema: &str, strict: bool) -> String {
    format!(r#"{{"type":"json_schema","name":"answer","strict":{strict},"schema":{schema}}}"#)
}

/// What a Responses answer says, whole or streamed, over HTTP or the socket.
#[derive(Debug, PartialEq)]
struct Reply {
    text: String,
    status: String,
    incomplete: Option<String>,
}

fn output_text(response: &JsonValue) -> String {
    response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message")
        .flat_map(|item| item["content"].as_array().unwrap())
        .filter_map(|part| part["text"].as_str())
        .collect()
}

/// The reply in a terminal event's `response`; with `deltas`, the text the
/// stream's deltas added up to, which must be the response's own.
fn reply_of(response: &JsonValue, deltas: Option<String>) -> Reply {
    let text = output_text(response);
    if let Some(deltas) = deltas {
        assert_eq!(deltas, text, "the streamed deltas add up to the response's text");
    }
    Reply {
        text,
        status: response["status"].as_str().unwrap().to_owned(),
        incomplete: response["incomplete_details"]["reason"].as_str().map(str::to_owned),
    }
}

/// The reply to `body` over HTTP, whole or streamed as `stream` says.
async fn responses_reply(app: &axum::Router, body: &str, stream: bool) -> Reply {
    let (status, _, text) = post_to(app, RESPONSES, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    if !stream {
        return reply_of(&serde_json::from_str::<JsonValue>(&text).unwrap(), None);
    }
    let events = responses::sse_events(&text);
    let deltas = events
        .iter()
        .filter(|(name, _)| name == "response.output_text.delta")
        .map(|(_, event)| event["delta"].as_str().unwrap())
        .collect();
    let (_, last) = events.last().expect("a terminal event");
    reply_of(&last["response"], Some(deltas))
}

/// The reply to `body` as a `response.create` on the WebSocket mode.
async fn socket_reply(live: &responses::Live, body: &str) -> Reply {
    let mut socket = responses::socket(live).await;
    responses::send_text(&mut socket, create_frame(body)).await;
    let events = responses::events_until(&mut socket, responses::terminal).await;
    let deltas = events
        .iter()
        .filter(|event| event["type"] == "response.output_text.delta")
        .map(|event| event["delta"].as_str().unwrap())
        .collect();
    let last = events.last().unwrap();
    assert!(last["type"] == "response.completed" || last["type"] == "response.incomplete", "{last}");
    reply_of(&last["response"], Some(deltas))
}

/// A server over the test vocabulary whose request 0 writes `script`,
/// serving on a socket.
async fn live_over(script: &str) -> (responses::Live, Arc<MockCompute>) {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(script));
    (responses::live(server(mock.clone(), SchedulerConfig::default())).await, mock)
}

/// Acceptance 1 and 4: whatever the model misses, the answer under
/// `text.format` is valid under the schema over HTTP, streamed or whole, and
/// over the WebSocket mode, and all three carry the same answer.
#[tokio::test]
async fn responses_text_format_json_schema_answers_validate_under_scripted_misses() {
    for miss in misses() {
        let format = schema_format_text(miss.schema, true);
        let (app, mock) = app(miss.script);
        let whole = responses_reply(&app, &responses_body(&format, false, 256), false).await;
        assert_eq!(whole.status, "completed", "{}: {whole:?}", miss.what);
        let value: JsonValue =
            serde_json::from_str(&whole.text).unwrap_or_else(|e| panic!("{}: not JSON ({e}): {whole:?}", miss.what));
        assert!((miss.holds)(&value), "{}: {}", miss.what, whole.text);
        assert!(!mock.repair_calls().is_empty(), "{}: the scripted misses were repaired", miss.what);
        let (app, _) = self::app(miss.script);
        let streamed = responses_reply(&app, &responses_body(&format, true, 256), true).await;
        assert_eq!(streamed, whole, "{}: streamed and whole", miss.what);
        let (live, mock) = live_over(miss.script).await;
        let socket = socket_reply(&live, &responses_body(&format, false, 256)).await;
        assert_eq!(socket, whole, "{}: the socket and HTTP", miss.what);
        assert!(!mock.repair_calls().is_empty(), "{}: repaired on the socket", miss.what);
    }
}

/// `text.format` keeps the schema's key order over HTTP and over the socket,
/// whose frame is parsed into a value that sorts them.
#[tokio::test]
async fn responses_text_format_writes_keys_in_the_schemas_order() {
    const SCHEMA: &str = r#"{"type":"object","properties":{"zeta":{"type":"integer"},"alpha":{"type":"integer"}},"required":["zeta","alpha"],"additionalProperties":false}"#;
    let format = schema_format_text(SCHEMA, true);
    let (app, _) = app(r#"{"alpha": 1, "zeta": 2}"#);
    let http = responses_reply(&app, &responses_body(&format, false, 128), false).await;
    let (live, _) = live_over(r#"{"alpha": 1, "zeta": 2}"#).await;
    let socket = socket_reply(&live, &responses_body(&format, false, 128)).await;
    for got in [&http, &socket] {
        assert!(got.text.starts_with("{\"zeta\"") && got.text.find("\"alpha\"") > got.text.find("\"zeta\""), "{got:?}");
    }
}

/// `json_object` on Responses: the misses the chat test scripts.
#[tokio::test]
async fn responses_text_format_json_object_answers_a_valid_object_under_scripted_misses() {
    let cases: [(&str, &str, JsonValue); 4] = [
        ("a prose preamble", "Sure! Here it is:\n{\"a\": 1}", json!({ "a": 1 })),
        ("a code fence", "```json\n{\"a\": 1}\n```", json!({ "a": 1 })),
        ("a remark after the object", "{\"a\": 1}\nThat is all.", json!({ "a": 1 })),
        ("an early stop", "{\"a\": [1, 2", json!({ "a": [1, 2] })),
    ];
    let format = r#"{"type":"json_object"}"#;
    for (what, script, expected) in cases {
        let (app, mock) = app(script);
        let whole = responses_reply(&app, &responses_body(format, false, 64), false).await;
        assert_eq!(whole.status, "completed", "{what}: {whole:?}");
        assert_eq!(JsonValue::Object(object(&whole.text)), expected, "{what}: {whole:?}");
        assert!(!mock.repair_calls().is_empty(), "{what}: repaired");
        let (app, _) = self::app(script);
        assert_eq!(responses_reply(&app, &responses_body(format, true, 64), true).await, whole, "{what}: streamed");
        let (live, _) = live_over(script).await;
        assert_eq!(socket_reply(&live, &responses_body(format, false, 64)).await, whole, "{what}: socket");
    }
}

/// Acceptance 3: a cut by `max_output_tokens` is `incomplete` with
/// `incomplete_details.reason: "max_output_tokens"` and the text as
/// generated, as a free answer's cut is, over HTTP and the socket.
#[tokio::test]
async fn a_cut_by_max_output_tokens_is_incomplete_as_a_free_answers_is() {
    let script = "{\"a\": \"hello world\"}";
    let expected = Reply { text: "{\"a\": ".into(), status: "incomplete".into(), incomplete: Some("max_output_tokens".into()) };
    for stream in [false, true] {
        let (app, _) = app(script);
        let held = responses_body(r#"{"type":"json_object"}"#, stream, 6);
        assert_eq!(responses_reply(&app, &held, stream).await, expected, "stream {stream}");
    }
    let (live, _) = live_over(script).await;
    assert_eq!(socket_reply(&live, &responses_body(r#"{"type":"json_object"}"#, false, 6)).await, expected, "socket");
    // A free answer is cut the same way.
    let (app, _) = app(script);
    let free = responses_body(r#"{"type":"text"}"#, false, 6);
    assert_eq!(responses_reply(&app, &free, false).await, expected, "a free answer");
}

// ── tools beside text.format (#592 on the Responses API) ─────────────────

/// An agent's turn on `/v1/responses`, as an SDK sends it: `input`, a flat
/// function tool, `tool_choice`, and `text.format` holding the answer to
/// [`WEATHER`]; greedy, thinking off.
fn responses_agent_turn(input: JsonValue, tool_choice: JsonValue, stream: bool) -> String {
    json!({
        "model": MODEL,
        "input": input,
        "tools": [{ "type": "function", "name": "get_weather", "parameters": {
            "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] } }],
        "tool_choice": tool_choice,
        "text": { "format": "FORMAT" },
        "enable_thinking": false,
        "temperature": 0,
        "max_output_tokens": 256,
        "stream": stream,
    })
    .to_string()
    .replace("\"FORMAT\"", &schema_format_text(WEATHER, true))
}

/// The Responses input after the first turn's call and the tool's result.
fn responses_after_the_call() -> JsonValue {
    json!([
        { "role": "user", "content": "What is the weather in Oslo?" },
        { "type": "function_call", "call_id": "call_0", "name": "get_weather", "arguments": "{\"city\":\"Oslo\"}" },
        { "type": "function_call_output", "call_id": "call_0", "output": "sunny, 21 C" },
    ])
}

/// A Responses turn as a [`Turn`]: its message text, its calls as (name,
/// arguments), and its status in place of a finish reason.
fn responses_turn_of(response: &JsonValue) -> Turn {
    let calls = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| (item["name"].as_str().unwrap().to_owned(), serde_json::from_str(item["arguments"].as_str().unwrap()).unwrap()))
        .collect();
    Turn { content: output_text(response), calls, finish: response["status"].as_str().unwrap().to_owned() }
}

/// The wires a Responses turn goes over.
#[derive(Debug, Clone, Copy)]
enum Wire {
    Whole,
    Streamed,
    Socket,
}

/// Turns sent over `wire`, in order, on one server (one socket for the
/// WebSocket mode) over `mock`.
async fn responses_turns(mock: &Arc<MockCompute>, wire: Wire, turns: &[(JsonValue, JsonValue)]) -> Vec<Turn> {
    let mut out = Vec::new();
    match wire {
        Wire::Whole | Wire::Streamed => {
            let stream = matches!(wire, Wire::Streamed);
            let app = server(mock.clone(), SchedulerConfig::default()).app();
            for (input, tool_choice) in turns {
                let body = responses_agent_turn(input.clone(), tool_choice.clone(), stream);
                let (status, _, text) = post_to(&app, RESPONSES, &body).await;
                assert_eq!(status, 200, "{body}: {text}");
                let response = match stream {
                    false => serde_json::from_str::<JsonValue>(&text).unwrap(),
                    true => responses::sse_events(&text).last().expect("a terminal event").1["response"].clone(),
                };
                out.push(responses_turn_of(&response));
            }
        }
        Wire::Socket => {
            let live = responses::live(server(mock.clone(), SchedulerConfig::default())).await;
            let mut socket = responses::socket(&live).await;
            for (input, tool_choice) in turns {
                responses::send_text(&mut socket, create_frame(&responses_agent_turn(input.clone(), tool_choice.clone(), false))).await;
                let events = responses::events_until(&mut socket, responses::terminal).await;
                let last = events.last().unwrap();
                assert_eq!(last["type"], "response.completed", "{last}");
                out.push(responses_turn_of(&last["response"]));
            }
        }
    }
    out
}

/// #592 acceptance 1 on the Responses API, over HTTP whole and streamed and
/// over its WebSocket mode: a turn that opens a tool call is released from
/// `text.format`'s schema and its call comes back as the model wrote it, with
/// no miss and no repair; the next turn, after the tool's result, is held
/// again and answers under the schema.
#[tokio::test]
async fn on_responses_a_turn_that_calls_a_tool_is_released_and_the_next_answers_under_the_schema() {
    for wire in [Wire::Whole, Wire::Streamed, Wire::Socket] {
        let mock = Arc::new(MockCompute::new());
        mock.script(0, &encode(CALL));
        mock.script(1, &encode("```json\n{\"city\": \"Oslo\", \"sky\": \"sunny\"}\n```"));
        let turns = [(json!("What is the weather in Oslo?"), json!("auto")), (responses_after_the_call(), json!("auto"))];
        let [first, second] = <[Turn; 2]>::try_from(responses_turns(&mock, wire, &turns).await).unwrap();
        assert_eq!(first.calls, [("get_weather".to_owned(), json!({ "city": "Oslo" }))], "{wire:?}: {first:?}");
        assert_eq!((first.content.as_str(), first.finish.as_str()), ("", "completed"), "{wire:?}: {first:?}");
        assert!(mock.repair_calls().iter().all(|repair| repair.request == 1), "{wire:?}: the call's text was never held");
        assert!(second.calls.is_empty(), "{wire:?}: {second:?}");
        let answer: JsonValue = serde_json::from_str(&second.content).unwrap_or_else(|e| panic!("{wire:?}: {e}: {second:?}"));
        assert_eq!(answer, json!({ "city": "Oslo", "sky": "sunny" }), "{wire:?}");
        assert!(!mock.repair_calls().is_empty(), "{wire:?}: the second turn's fence was repaired");
    }
}

/// #592 acceptance 3 on the Responses API: `tool_choice: "required"` beside
/// `text.format` forces the call, as without it; the model here was about to
/// answer, and the forced turn is not held to the schema.
#[tokio::test]
async fn on_responses_a_forced_tool_choice_beside_text_format_forces_the_call() {
    let opening = "<tool_call>\n<function";
    let rest = "=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
    let forced = encode(opening).len();
    let answering: String = "{\"city\": \"Oslo\", \"sky\": \"sunny\"}".chars().chain(std::iter::repeat(' ')).take(forced).collect();
    for wire in [Wire::Whole, Wire::Streamed, Wire::Socket] {
        let mock = Arc::new(MockCompute::new());
        mock.script(0, &encode(&format!("{answering}{rest}")));
        let turns = [(json!("What is the weather in Oslo?"), json!("required"))];
        let got = responses_turns(&mock, wire, &turns).await.remove(0);
        assert_eq!(got.calls, [("get_weather".to_owned(), json!({ "city": "Oslo" }))], "{wire:?}: {got:?}");
        assert!(mock.repair_calls().is_empty(), "{wire:?}: the forced turn was not held to the schema");
    }
}

/// Acceptance 2: every refusal is a 400 naming `text.format` and the keyword
/// or limit, before anything is generated, over HTTP and over the socket.
#[tokio::test]
async fn text_format_refusals_name_the_field_and_the_keyword_or_limit() {
    let object = |inner: &str| {
        format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#)
    };
    let properties: Vec<String> = (0..=5000).map(|i| format!(r#""p{i}":{{"type":"null"}}"#)).collect();
    let cases: Vec<(&str, String, &str, &str)> = vec![
        ("a keyword outside the subset", schema_format_text(&object(r#"{"type":"string","allOf":[]}"#), true), "unsupported_keyword", "`allOf`"),
        ("a root that is not an object", schema_format_text(r#"{"type":"string"}"#, true), "root_not_object", "`type`"),
        (
            "too many properties",
            schema_format_text(&format!(r#"{{"type":"object","properties":{{{}}}}}"#, properties.join(",")), false),
            "too_many_properties",
            "`properties`",
        ),
        ("no schema", r#"{"type":"json_schema","name":"x"}"#.into(), "invalid_request_error", "`schema` is required"),
        ("a strict that is not a boolean", r#"{"type":"json_schema","name":"x","strict":"yes","schema":{}}"#.into(), "invalid_request_error", "`strict`"),
        ("another type", r#"{"type":"xml"}"#.into(), "unsupported_value", "xml"),
    ];
    for (what, format, code, mention) in cases {
        let body = responses_body(&format, false, 64);
        let mock = Arc::new(MockCompute::new());
        let app = server(mock.clone(), SchedulerConfig::default()).app();
        let (status, _, text) = post_to(&app, RESPONSES, &body).await;
        assert_eq!(status, 400, "{what}: {}", &text[..text.len().min(300)]);
        let v: JsonValue = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["code"], code, "{what}: {text}");
        assert_eq!(v["error"]["param"], "text.format", "{what}: {text}");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.starts_with("text.format"), "{what}: {message}");
        assert!(message.contains(mention), "{what}: the message names {mention}: {message}");
        assert!(mock.prefill_calls().is_empty() && mock.decode_calls().is_empty(), "{what}: nothing was generated");

        let (live, _) = live_over("").await;
        let mut socket = responses::socket(&live).await;
        responses::send_text(&mut socket, create_frame(&body)).await;
        let event = responses::next_event(&mut socket).await;
        assert_eq!(event["type"], "error", "{what}: {event}");
        assert_eq!(event["error"]["param"], "text.format", "{what}: {event}");
        assert_eq!(event["error"]["code"], code, "{what}: {event}");
    }
}

/// The response echoes `text.format` as sent, and a free request's `text`.
#[tokio::test]
async fn the_response_echoes_its_text_format() {
    let (app, _) = app("{}");
    let (_, _, text) = post_to(&app, RESPONSES, &responses_body(r#"{"type":"json_object"}"#, false, 8)).await;
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["text"]["format"], json!({ "type": "json_object" }), "{text}");
    let (app, _) = self::app("hi");
    let free = json!({ "model": MODEL, "input": "hi", "max_output_tokens": 4 }).to_string();
    let (_, _, text) = post_to(&app, RESPONSES, &free).await;
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["text"]["format"], json!({ "type": "text" }), "{text}");
}

/// The renders a Responses request for [`CAT`] (described as "A cat.") makes,
/// with `extra` fields beside it.
async fn responses_renders(extra: JsonValue) -> Vec<Render> {
    let format = format!(r#"{{"type":"json_schema","name":"cat","description":"A cat.","strict":true,"schema":{CAT}}}"#);
    let mut body: JsonValue = serde_json::from_str(&responses_body(r#"{"type":"text"}"#, false, 16)).unwrap();
    for (field, value) in extra.as_object().unwrap() {
        body[field] = value.clone();
    }
    let body = body.to_string().replace(r#"{"type":"text"}"#, &format);
    let vocabulary = Vocabulary::new();
    let app = over_mock(vocabulary.clone()).app();
    let (status, _, text) = post_to(&app, RESPONSES, &body).await;
    assert_eq!(status, 200, "{text}");
    vocabulary.renders()
}

/// The schema is described to the model after the conversation on the
/// Responses API too, with the format's `description`, and
/// `ignis_describe_schema: false` leaves the prompt as sent.
#[tokio::test]
async fn a_responses_json_schema_is_described_unless_the_request_says_not() {
    let on = responses_renders(json!({})).await;
    assert_eq!(on.len(), 1, "{on:?}");
    assert_eq!(on[0].description.as_deref(), Some(CAT_DESCRIBED));
    let off = responses_renders(json!({ "ignis_describe_schema": false })).await;
    assert_eq!(off[0].description, None, "{off:?}");
    assert!(on[0].tokens.len() > off[0].tokens.len());
}

// ── /v1/tokenize counts the description (#589, #591) ─────────────────────

async fn tokenize(app: &axum::Router, body: JsonValue) -> JsonValue {
    let (status, _, text) = post_to(app, "/v1/tokenize", &body.to_string().replace("\"SCHEMA\"", CAT)).await;
    assert_eq!(status, 200, "{text}");
    serde_json::from_str(&text).unwrap()
}

/// `usage.prompt_tokens` of a served chat completion of `messages` held to
/// `format`.
async fn served_prompt_tokens(app: &axum::Router, messages: &JsonValue, format: &JsonValue) -> u64 {
    let body = json!({
        "model": MODEL, "messages": messages, "response_format": format,
        "enable_thinking": false, "max_tokens": 4, "temperature": 0,
    })
    .to_string()
    .replace("\"SCHEMA\"", CAT);
    let (status, _, text) = post_text(app, &body).await;
    assert_eq!(status, 200, "{text}");
    serde_json::from_str::<JsonValue>(&text).unwrap()["usage"]["prompt_tokens"].as_u64().unwrap()
}

/// The count of a `json_schema` body includes the description, and is the
/// `usage.prompt_tokens` the same body reports when served.
#[tokio::test]
async fn tokenize_counts_the_schema_description_chat_serves_the_body_with() {
    let app = over_mock(Vocabulary::new()).app();
    let messages = json!([{ "role": "user", "content": "describe a cat" }]);
    let format = json!({ "type": "json_schema", "json_schema": { "name": "cat", "description": "A cat.", "strict": true, "schema": "SCHEMA" } });
    let count = |v: &JsonValue| v["count"].as_u64().unwrap();
    let described =
        tokenize(&app, json!({ "messages": messages, "response_format": format, "enable_thinking": false, "return_text": true })).await;
    let off = tokenize(
        &app,
        json!({ "messages": messages, "response_format": format, "ignis_describe_schema": false, "enable_thinking": false }),
    )
    .await;
    let free = tokenize(&app, json!({ "messages": messages, "enable_thinking": false })).await;
    let object =
        tokenize(&app, json!({ "messages": messages, "response_format": { "type": "json_object" }, "enable_thinking": false })).await;
    assert_eq!(count(&off), count(&free), "off counts the conversation");
    assert_eq!(count(&object), count(&free), "json_object adds nothing");
    // The placeholder template is one token per word: the sentence alone is eleven.
    assert!(count(&described) >= count(&free) + 11, "{described} vs {free}");
    assert_eq!(described["text"].as_str().unwrap().chars().count() as u64, count(&described), "the text spells the ids: {described}");
    assert_eq!(served_prompt_tokens(&app, &messages, &format).await, count(&described), "the count is the served prompt");
    // A schema chat refuses is refused here, naming the field.
    let refused = json!({ "messages": messages, "response_format": { "type": "json_schema", "json_schema": { "name": "x", "schema": { "type": "string" }, "strict": true } } });
    let (status, _, text) = post_to(&app, "/v1/tokenize", &refused.to_string()).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("\"param\":\"response_format\""), "{text}");
}
