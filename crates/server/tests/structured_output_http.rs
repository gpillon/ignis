//! Structured output on chat completions (GitHub #587, #588, spec #287, ADR
//! 0089): `response_format` with `json_object` and with `json_schema`, end
//! to end over the real router on `MockCompute` (ADR 0006).
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

use std::path::{Path, PathBuf};
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
    let request = Request::builder()
        .method("POST")
        .uri(CHAT)
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
/// `jsonschema`: `IGNIS_GRAMMAR_CHECK_PYTHON`, else the venv
/// `docs/agents/testing/structured-output.md` makes under
/// `.scratch/grammar-venv`.
fn validator_python() -> Option<PathBuf> {
    if let Some(python) = std::env::var_os("IGNIS_GRAMMAR_CHECK_PYTHON") {
        return Some(PathBuf::from(python));
    }
    let venv = repo().join(".scratch/grammar-venv");
    [venv.join("Scripts/python.exe"), venv.join("bin/python")].into_iter().find(|python| python.exists())
}

/// Every answer, `(case, schema, text)`, valid under its schema by an
/// independent validator: `jsonschema` (Draft 2020-12, formats checked,
/// numbers as exact decimals) through `validate.py`. Machine-local: where no
/// Python has it, a skip that says so.
fn assert_independently_valid(answers: &[(&str, &str, String)]) {
    let Some(python) = validator_python() else {
        eprintln!("skip: no Python with tools/grammar-check/requirements.txt (IGNIS_GRAMMAR_CHECK_PYTHON, .scratch/grammar-venv)");
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
