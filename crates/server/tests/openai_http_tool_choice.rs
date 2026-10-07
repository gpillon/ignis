//! `tool_choice: "required"` and a named function (GitHub #286, spec
//! server/11): the server forces the opening of the tool-call dialect and the
//! model writes the rest. End to end over the real router on `MockCompute`
//! (ADR 0006).
//!
//! The template here encodes a literal one token per character, except
//! `<tool_call>` and `</think>`, which are one token each as in the real
//! tokenizer. Its decoder reads those ids back, and reads the mock's own
//! free tokens from a script: what the "model" writes after the forced
//! opening.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";
/// `<tool_call>`'s one token, past every character's id.
const OPEN: TokenId = 0x0011_0001;
const OPENER: &str = "<tool_call>\n<function=read_file>\n";
const REQUIRED_OPENER: &str = "<tool_call>\n<function";
const ARGUMENTS: &str = "<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>";

/// One token per character, `<tool_call>` one token, and `</think>` one
/// token when `think_end` names it.
fn encode(text: &str, think_end: Option<TokenId>) -> Vec<TokenId> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix("<tool_call>") {
            ids.push(OPEN);
            rest = after;
        } else if let (Some(after), Some(think_end)) = (rest.strip_prefix("</think>"), think_end) {
            ids.push(think_end);
            rest = after;
        } else {
            ids.push(c as TokenId);
            rest = &rest[c.len_utf8()..];
        }
    }
    ids
}

/// What the test template's tokenizer can encode.
#[derive(Clone, Copy, PartialEq)]
enum Tokenizer {
    /// `<tool_call>` and `</think>` are one token each, as in the real one.
    Whole,
    /// `</think>` is split into characters.
    SplitThinkEnd,
    /// No tokenizer at all: nothing encodes.
    Absent,
}

#[derive(Clone)]
struct Forcing {
    script: HashMap<TokenId, &'static str>,
    think_end: TokenId,
    tokenizer: Tokenizer,
}

impl Forcing {
    fn text(&self, token: TokenId) -> String {
        match token {
            OPEN => "<tool_call>".into(),
            t if t == self.think_end => "</think>".into(),
            t => match self.script.get(&t) {
                Some(text) => (*text).into(),
                None => char::from_u32(t).map(String::from).unwrap_or_else(|| "?".into()),
            },
        }
    }
}

impl TemplateProvider for Forcing {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        match self.tokenizer {
            Tokenizer::Whole => Some(encode(text, Some(self.think_end))),
            Tokenizer::SplitThinkEnd => Some(encode(text, None)),
            Tokenizer::Absent => None,
        }
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens.iter().map(|&t| self.text(t)).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(ForcingDecoder(self.clone()))
    }
}

struct ForcingDecoder(Forcing);

impl TokenDecoder for ForcingDecoder {
    fn push(&mut self, token: TokenId) -> String {
        self.0.text(token)
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server whose request 0 writes `script` at the given generation steps
/// and stops on its own at `eos` — and whose `</think>` is the mock's own
/// token at step 1, so a thinking request closes its block there.
fn app(script: &[(u32, &'static str)], eos: Option<u32>) -> (axum::Router, Arc<MockCompute>) {
    app_with(script, eos, Tokenizer::Whole)
}

fn app_with(script: &[(u32, &'static str)], eos: Option<u32>, tokenizer: Tokenizer) -> (axum::Router, Arc<MockCompute>) {
    app_closing_at(1, script, eos, tokenizer)
}

/// [`app_with`], with the block closing at step `close` instead of 1.
fn app_closing_at(
    close: u32,
    script: &[(u32, &'static str)],
    eos: Option<u32>,
    tokenizer: Tokenizer,
) -> (axum::Router, Arc<MockCompute>) {
    let mock = Arc::new(MockCompute::new());
    if let Some(step) = eos {
        mock.eos_after(0, step);
    }
    let template = Forcing {
        script: script.iter().map(|&(step, text)| (mock.token_for(0, step), text)).collect(),
        think_end: mock.token_for(0, close),
        tokenizer,
    };
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        mock.clone(),
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(template))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0);
    (server.app(), mock)
}

/// The forced opening's length in the test template's tokens.
fn tokens(text: &str) -> u32 {
    encode(text, None).len() as u32
}

fn read_file() -> JsonValue {
    json!({ "type": "function", "function": { "name": "read_file", "parameters": {
        "type": "object", "properties": { "path": { "type": "string" } } } } })
}

fn list_dir() -> JsonValue {
    json!({ "type": "function", "function": { "name": "list_dir", "parameters": {
        "type": "object", "properties": {} } } })
}

fn chat(tool_choice: JsonValue, thinking: bool, stream: bool) -> JsonValue {
    json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "what is in a.txt?" }],
        "tools": [read_file(), list_dir()],
        "tool_choice": tool_choice,
        "enable_thinking": thinking,
        "max_tokens": 64,
        "stream": stream,
    })
}

async fn post(app: &axum::Router, path: &str, body: JsonValue) -> (u16, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_chunks(body: &str) -> Vec<JsonValue> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:").map(str::trim))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// The one call a non-streaming body carries, as (name, arguments).
fn the_call(body: &str) -> (String, JsonValue) {
    let v: JsonValue = serde_json::from_str(body).unwrap();
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap_or_else(|| panic!("no tool_calls: {body}"));
    assert_eq!(calls.len(), 1, "{body}");
    let function = &calls[0]["function"];
    let arguments = serde_json::from_str(function["arguments"].as_str().unwrap()).unwrap();
    (function["name"].as_str().unwrap().to_owned(), arguments)
}

fn assert_no_opener_leaked(body: &str) {
    assert!(!body.contains("<tool_call") && !body.contains("<function"), "the opening leaked: {body}");
}

// ── thinking off: forced from the first token ───────────────────────────

#[tokio::test]
async fn a_named_function_is_forced_from_the_first_token_and_the_model_writes_its_arguments() {
    let forced = tokens(OPENER);
    let (app, mock) = app(&[(forced, ARGUMENTS)], Some(forced + 1));
    let (status, body) =
        post(&app, "/v1/chat/completions", chat(json!({ "type": "function", "function": { "name": "read_file" } }), false, false)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(the_call(&body), ("read_file".to_owned(), json!({ "path": "a.txt" })));
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(v["choices"][0]["message"]["content"], "");
    assert_eq!(v["usage"]["completion_tokens"], forced + 1, "the forced tokens are generated tokens");
    assert_no_opener_leaked(&body);
    // The prefill drew the first token; every round after drew one forced
    // token until the opening was spent.
    let prefill = mock.prefill_calls().concat();
    assert_eq!(prefill.last().and_then(|job| job.permitted.as_deref()), Some(&[OPEN][..]));
    let sets = mock.decode_calls().concat().iter().filter(|job| job.permitted.is_some()).count();
    assert_eq!(sets as u32, forced - 1);
}

#[tokio::test]
async fn required_forces_the_opening_and_leaves_the_name_to_the_model() {
    let forced = tokens(REQUIRED_OPENER);
    let (app, _) = app(&[(forced, "=list_dir>\n</function>\n</tool_call>")], Some(forced + 1));
    let (status, body) = post(&app, "/v1/chat/completions", chat(json!("required"), false, false)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(the_call(&body), ("list_dir".to_owned(), json!({})));
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_no_opener_leaked(&body);
}

#[tokio::test]
async fn a_forced_call_streams_the_way_an_unforced_one_does() {
    let forced = tokens(OPENER);
    let (app, _) = app(&[(forced, ARGUMENTS)], Some(forced + 1));
    let (status, body) =
        post(&app, "/v1/chat/completions", chat(json!({ "type": "function", "function": { "name": "read_file" } }), false, true)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);
    let calls: Vec<&JsonValue> =
        chunks.iter().filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array()).flatten().collect();
    assert_eq!(calls.len(), 1, "one whole call delta: {body}");
    assert_eq!(calls[0]["function"]["name"], "read_file");
    assert_eq!(calls[0]["function"]["arguments"], "{\"path\":\"a.txt\"}");
    let content: String =
        chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "", "{body}");
    assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");
    assert_no_opener_leaked(&body);
}

/// A forced call the cap cuts before it closes is dropped whole, as an
/// unforced one is: no call, the ordinary `finish_reason`, and nothing of
/// the forced opening in the content.
#[tokio::test]
async fn a_forced_call_cut_before_it_closes_is_dropped_whole() {
    let forced = tokens(OPENER);
    for stream in [false, true] {
        let (app, _) = app(&[(forced, "<parameter=path>\na")], None);
        let mut request = chat(json!({ "type": "function", "function": { "name": "read_file" } }), false, stream);
        request["max_tokens"] = json!(forced + 1);
        let (status, body) = post(&app, "/v1/chat/completions", request).await;
        assert_eq!(status, 200, "{body}");
        assert_no_opener_leaked(&body);
        assert!(!body.contains("tool_calls\":["), "no call: {body}");
        let finish = match stream {
            false => serde_json::from_str::<JsonValue>(&body).unwrap()["choices"][0]["finish_reason"].clone(),
            true => sse_chunks(&body).last().unwrap()["choices"][0]["finish_reason"].clone(),
        };
        assert_eq!(finish, "length", "{body}");
    }
}

// ── thinking on: forced after the block closes ─────────────────────────

/// The model reasons at step 0 and closes its block at step 1; step 2 is
/// the token drawn in the round that emitted the close, unseen by the
/// scheduler. The joiner (`\n`) follows it at step 3, then the whole
/// opening, then the model again.
#[tokio::test]
async fn with_thinking_on_the_call_is_forced_after_the_models_own_close() {
    let forced = tokens(OPENER);
    let free = 4 + forced;
    for stream in [false, true] {
        let (app, _) = app(&[(0, "Reading the file."), (2, "\n\n"), (free, ARGUMENTS)], Some(free + 1));
        let (status, body) =
            post(&app, "/v1/chat/completions", chat(json!({ "type": "function", "function": { "name": "read_file" } }), true, stream)).await;
        assert_eq!(status, 200, "{body}");
        assert_no_opener_leaked(&body);
        if stream {
            let chunks = sse_chunks(&body);
            let reasoning: String =
                chunks.iter().filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str()).collect();
            assert_eq!(reasoning, "Reading the file.");
            let content: String =
                chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
            assert_eq!(content, "", "the unseen `\\n\\n` and the joiner are trimmed: {body}");
            assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "tool_calls", "{body}");
        } else {
            assert_eq!(the_call(&body), ("read_file".to_owned(), json!({ "path": "a.txt" })));
            let v: JsonValue = serde_json::from_str(&body).unwrap();
            assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Reading the file.");
            assert_eq!(v["choices"][0]["message"]["content"], "");
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        }
    }
}

/// The token after the model's own `</think>` is drawn before the close is
/// seen, so it is the model's. When it ends the turn, the request ends
/// there with no call (spec 11, known gaps): the forcing never got a draw.
#[tokio::test]
async fn an_end_of_turn_drawn_unseen_after_the_close_ends_the_request_with_no_call() {
    let (app, _) = app(&[(0, "Nothing to do.")], Some(2));
    let (status, body) = post(&app, "/v1/chat/completions", chat(json!("required"), true, false)).await;
    assert_eq!(status, 200, "{body}");
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["message"].get("tool_calls").is_none(), "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_no_opener_leaked(&body);
}

/// With thinking on the reasoning spends the cap the call needs too, by an
/// amount nobody knows when the request arrives: a cap that fits the
/// opening is accepted, and a block that never closes inside it ends
/// `length` with no call (spec 11, departures).
#[tokio::test]
async fn with_thinking_on_a_block_that_spends_the_cap_ends_length_with_no_call() {
    let (app, mock) = app_closing_at(1_000, &[], None, Tokenizer::Whole);
    let mut request = chat(json!("required"), true, false);
    request["max_tokens"] = json!(tokens(REQUIRED_OPENER) + 2);
    let (status, body) = post(&app, "/v1/chat/completions", request).await;
    assert_eq!(status, 200, "{body}");
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["message"].get("tool_calls").is_none(), "{body}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(mock.decode_calls().concat().iter().all(|job| job.permitted.is_none()), "never forced");
}

// ── unchanged ────────────────────────────────────────────────────────────

/// Forcing changes what the model may draw, never what it reads: the
/// prompt `"required"` or a named function prefills is the one `"auto"`
/// prefills, with thinking off and on.
#[tokio::test]
async fn forcing_leaves_the_prompt_as_auto_renders_it() {
    for thinking in [false, true] {
        let mut prompts = Vec::new();
        for choice in [json!("auto"), json!("required"), json!({ "type": "function", "function": { "name": "read_file" } })] {
            let (app, mock) = app(&[], None);
            let (status, body) = post(&app, "/v1/chat/completions", chat(choice.clone(), thinking, false)).await;
            assert_eq!(status, 200, "{choice}: {body}");
            prompts.push(mock.prefill_calls().concat().into_iter().flat_map(|job| job.tokens).collect::<Vec<_>>());
        }
        assert!(!prompts[0].is_empty());
        assert!(prompts.iter().all(|prompt| *prompt == prompts[0]), "thinking {thinking}: {prompts:?}");
    }
}

#[tokio::test]
async fn auto_and_none_force_nothing() {
    for choice in [json!("auto"), json!("none"), JsonValue::Null] {
        let (app, mock) = app(&[], None);
        let mut request = chat(choice.clone(), false, false);
        if choice.is_null() {
            request.as_object_mut().unwrap().remove("tool_choice");
        }
        let (status, body) = post(&app, "/v1/chat/completions", request).await;
        assert_eq!(status, 200, "{choice}: {body}");
        assert!(mock.prefill_calls().concat().iter().all(|job| job.permitted.is_none()), "{choice}");
        assert!(mock.decode_calls().concat().iter().all(|job| job.permitted.is_none()), "{choice}");
    }
}

// ── refusals ─────────────────────────────────────────────────────────────

async fn refused(app: &axum::Router, path: &str, request: JsonValue) -> JsonValue {
    let (status, body) = post(app, path, request).await;
    assert_eq!(status, 400, "{body}");
    serde_json::from_str::<JsonValue>(&body).unwrap()["error"].clone()
}

#[tokio::test]
async fn a_function_that_is_not_among_the_tools_is_refused() {
    let (app, _) = app(&[], None);
    let error = refused(&app, "/v1/chat/completions", chat(json!({ "type": "function", "function": { "name": "write_file" } }), false, false)).await;
    assert_eq!(error["param"], "tool_choice");
    assert!(error["message"].as_str().unwrap().contains("write_file"), "{error}");
}

#[tokio::test]
async fn required_with_no_tools_is_refused() {
    let (app, _) = app(&[], None);
    let mut request = chat(json!("required"), false, false);
    request.as_object_mut().unwrap().remove("tools");
    let error = refused(&app, "/v1/chat/completions", request).await;
    assert_eq!(error["param"], "tool_choice");
}

#[tokio::test]
async fn a_cap_shorter_than_the_opening_is_refused_naming_the_cap() {
    let (app, _) = app(&[], None);
    for (field, value) in [("max_tokens", 5), ("max_completion_tokens", 5)] {
        let mut request = chat(json!("required"), false, false);
        let body = request.as_object_mut().unwrap();
        body.remove("max_tokens");
        body.insert(field.into(), json!(value));
        let error = refused(&app, "/v1/chat/completions", request).await;
        assert_eq!(error["param"], field, "{error}");
    }
}

#[tokio::test]
async fn a_template_that_cannot_encode_the_opening_is_refused_not_ignored() {
    let (app, _) = app_with(&[], None, Tokenizer::Absent);
    let error = refused(&app, "/v1/chat/completions", chat(json!("required"), false, false)).await;
    assert_eq!(error["param"], "tool_choice");
    assert_eq!(error["code"], "tool_choice_unsupported");
}

/// With thinking on the scheduler sees the block close by its one
/// `</think>` token; a tokenizer that splits it cannot force after the
/// block, and says so. Thinking off needs no close and is served.
#[tokio::test]
async fn with_thinking_on_a_split_close_is_refused_naming_the_way_out() {
    let (app, _) = app_with(&[], None, Tokenizer::SplitThinkEnd);
    let error = refused(&app, "/v1/chat/completions", chat(json!("required"), true, false)).await;
    assert_eq!(error["param"], "tool_choice");
    assert_eq!(error["code"], "tool_choice_unsupported");
    assert!(error["message"].as_str().unwrap().contains("enable_thinking: false"), "{error}");
    let (status, body) = post(&app, "/v1/chat/completions", chat(json!("required"), false, false)).await;
    assert_eq!(status, 200, "{body}");
}

#[tokio::test]
async fn an_unknown_tool_choice_shape_is_refused() {
    let (app, _) = app(&[], None);
    for choice in [json!("always"), json!({ "type": "allowed_tools", "tools": [] }), json!(3)] {
        let error = refused(&app, "/v1/chat/completions", chat(choice.clone(), false, false)).await;
        assert_eq!(error["param"], "tool_choice", "{choice}: {error}");
    }
}

// ── the Responses API ────────────────────────────────────────────────────

fn response_request(tool_choice: JsonValue) -> JsonValue {
    json!({
        "model": MODEL,
        "input": "what is in a.txt?",
        "tools": [
            { "type": "function", "name": "read_file", "parameters": {
                "type": "object", "properties": { "path": { "type": "string" } } } },
            { "type": "function", "name": "list_dir", "parameters": { "type": "object", "properties": {} } },
        ],
        "tool_choice": tool_choice,
        "enable_thinking": false,
        "max_output_tokens": 64,
    })
}

#[tokio::test]
async fn the_responses_api_forces_a_named_function_in_its_own_shape() {
    let forced = tokens(OPENER);
    let (app, _) = app(&[(forced, ARGUMENTS)], Some(forced + 1));
    let (status, body) = post(&app, "/v1/responses", response_request(json!({ "type": "function", "name": "read_file" }))).await;
    assert_eq!(status, 200, "{body}");
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    let calls: Vec<&JsonValue> = v["output"].as_array().unwrap().iter().filter(|i| i["type"] == "function_call").collect();
    assert_eq!(calls.len(), 1, "{body}");
    assert_eq!(calls[0]["name"], "read_file");
    assert_eq!(calls[0]["arguments"], "{\"path\":\"a.txt\"}");
    assert_eq!(v["tool_choice"], json!({ "type": "function", "name": "read_file" }), "echoed as sent");
    assert_no_opener_leaked(&body);
}

#[tokio::test]
async fn the_responses_api_forces_required_and_refuses_what_chat_refuses() {
    let forced = tokens(REQUIRED_OPENER);
    let (app, _) = app(&[(forced, "=list_dir>\n</function>\n</tool_call>")], Some(forced + 1));
    let (status, body) = post(&app, "/v1/responses", response_request(json!("required"))).await;
    assert_eq!(status, 200, "{body}");
    let v: JsonValue = serde_json::from_str(&body).unwrap();
    let call = v["output"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").expect(&body);
    assert_eq!(call["name"], "list_dir");
    let mut short = response_request(json!("required"));
    short["max_output_tokens"] = json!(5);
    assert_eq!(refused(&app, "/v1/responses", short).await["param"], "max_output_tokens");
    let unknown = response_request(json!({ "type": "function", "name": "write_file" }));
    assert_eq!(refused(&app, "/v1/responses", unknown).await["param"], "tool_choice");
}
