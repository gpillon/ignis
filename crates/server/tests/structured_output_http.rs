//! Structured output's first tracer bullet (GitHub #587, spec #287, ADR
//! 0089): `response_format: {"type": "json_object"}` on chat completions,
//! end to end over the real router on `MockCompute` (ADR 0006).
//!
//! The "model" follows a script ([`MockCompute::script`]) that misses on
//! purpose: a prose preamble, a code fence, a remark after the object, an
//! early stop, nothing but whitespace. Each draw of the answer is checked
//! against the grammar before the next round consumes it, and a miss is
//! repaired. The mock's repair ranks what the script was going to write next
//! above the rest, then the vocabulary in order; the test vocabulary puts
//! the tokens that close a JSON value first, so a repair with nothing better
//! to go on moves the answer toward its end, as a model that has to stop
//! would.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use ignis_core::compute::ModelFamily;
use ignis_core::mock::GatedCompute;
use ignis_core::{Compute, ConcreteScheduler, MockCompute, SchedulerConfig, TokenId};
use ignis_grammar::{GrammarCache, Vocab, VocabEntry, DEFAULT_CACHE_BYTES, DEFAULT_MAX_WS_RUN};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions, THINKING_CLOSE_TEXT};
use ignis_server::Server;

const MODEL: &str = "test-model";
const CHAT: &str = "/v1/chat/completions";

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

/// `text` in the test vocabulary: `</think>` and a fence one token each,
/// every other character its own code point.
fn encode(text: &str) -> Vec<TokenId> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix("</think>") {
            ids.push(THINK_END);
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
}

impl Vocabulary {
    fn new() -> Self {
        let entries = (0..VOCAB as TokenId)
            .map(|id| match (id, text_of(id)) {
                (STOP | THINK_END, _) => VocabEntry::Control,
                (_, Some(text)) => VocabEntry::Text(text.into_bytes()),
                (_, None) => VocabEntry::Absent,
            })
            .collect();
        Self { grammars: Arc::new(GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES)) }
    }
}

impl TemplateProvider for Vocabulary {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        Some(encode(text))
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
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..config }, compute);
    Server::new(Engine::new(Box::new(scheduler)), Box::new(Vocabulary::new()))
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
    let request = Request::builder()
        .method("POST")
        .uri(CHAT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
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
    let (status, _, text) = post(app, body).await;
    assert_eq!(status, 200, "{body}: {text}");
    if body["stream"] == true {
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

#[tokio::test]
async fn json_schema_stays_refused_naming_the_work_in_progress() {
    let (app, _) = app("{}");
    let mut body = chat(false, 8);
    body["response_format"] = json!({ "type": "json_schema", "json_schema": { "name": "x", "schema": { "type": "object" } } });
    let (status, _, text) = post(&app, &body).await;
    assert_eq!(status, 400, "{text}");
    let v: JsonValue = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["code"], "unsupported_value", "{text}");
    assert_eq!(v["error"]["param"], "response_format", "{text}");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(message.contains("json_schema") && message.contains("#287"), "{message}");
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

/// A forced tool call and a held answer would each own the first token:
/// refused until a turn may open a call under the grammar (#592).
#[tokio::test]
async fn a_forced_tool_call_beside_json_object_is_refused() {
    let (app, _) = app("{}");
    let mut body = chat(false, 64);
    body["tools"] = json!([{ "type": "function", "function": { "name": "f", "parameters": { "type": "object" } } }]);
    body["tool_choice"] = json!("required");
    let (status, _, text) = post(&app, &body).await;
    assert_eq!(status, 400, "{text}");
    assert!(text.contains("\"param\":\"response_format\"") && text.contains("#592"), "{text}");
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
