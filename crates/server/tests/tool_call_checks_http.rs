//! The call check (GitHub #608, spec #582, ADR 0094) over the real router on
//! the mock engine (ADR 0006): every tool call block the parser closes, on
//! chat completions and on the Responses API (HTTP and WebSocket), is checked
//! against its tool's `parameters` and counted by outcome and by kind of
//! violation, and a strict tool whose schema lies outside OpenAI's strict
//! subset is counted as it arrives. Nothing a client receives changes: each
//! response is held to the same request's response on a server with metrics
//! off, which checks nothing.
//!
//! The mock writes a script a character a token (`MockCompute::script`), and
//! the template reads each such token back as its character.

#[path = "support/responses.rs"]
mod responses;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig, TokenId};
use ignis_grammar::{GrammarCache, Vocab, VocabEntry};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::metrics::Metrics;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;

const MODEL: &str = "test-model";

/// Where the character tokens start: `CHARS + c` reads as the character `c`.
const CHARS: TokenId = 0xFFE0_0000;

/// A token that reads as a whole call: what a first token would have to be
/// for the first-token probe of #518 to see a call close.
const WHOLE_CALL: TokenId = CHARS - 1;
const WHOLE_CALL_TEXT: &str = "<tool_call>
<function=edit>
<parameter=path>
a.txt
</parameter>
</function>
</tool_call>";

fn encode(text: &str) -> Vec<TokenId> {
    text.chars().map(|c| CHARS + u32::from(c)).collect()
}

/// The placeholder template's prompt, a character a token out, and, with
/// `grammars`, a vocabulary structured output could hold an answer to (a
/// byte a token): what makes the strict subset readable.
struct Chars {
    grammars: Option<Arc<GrammarCache>>,
}

impl TemplateProvider for Chars {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[JsonValue],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens.iter().map(|&t| piece(t)).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder)
    }

    fn grammars(&self) -> Option<Arc<GrammarCache>> {
        self.grammars.clone()
    }
}

fn piece(token: TokenId) -> String {
    if token == WHOLE_CALL {
        return WHOLE_CALL_TEXT.to_owned();
    }
    token.checked_sub(CHARS).and_then(char::from_u32).map(String::from).unwrap_or_default()
}

struct Decoder;

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        piece(token)
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A vocabulary of the 256 bytes.
fn byte_grammars() -> Arc<GrammarCache> {
    let entries = (0..=255u8).map(|b| VocabEntry::Text(vec![b])).collect();
    Arc::new(GrammarCache::new(Vocab::new(entries, &[]), 1 << 20))
}

/// A server whose model writes `scripts[i]` as its request `i`, with
/// metrics on or off.
fn server(scripts: &[&str], grammars: bool, metrics: bool) -> Server {
    let mock = Arc::new(MockCompute::new());
    for (request, script) in scripts.iter().enumerate() {
        mock.script(request as u64, &encode(script));
    }
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, mock);
    let template = Chars { grammars: grammars.then(byte_grammars) };
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(template))
        .with_request_timeout(std::time::Duration::from_secs(5))
        .with_wall_clock(Arc::new(ignis_server::telemetry::FixedClock::new(responses::CREATED_AT_MS)))
        .with_counted_response_ids()
        .with_seedless_seed(0);
    if metrics { server.with_metrics() } else { server }
}

/// The tools every request here declares: `edit`, closed, with a required
/// path, an integer line and an enum mode; `note`, with no parameters.
fn tools() -> JsonValue {
    json!([
        {"type": "function", "function": {"name": "edit", "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "pattern": "^[^ ]+$"},
                "line": {"type": "integer"},
                "mode": {"type": "string", "enum": ["replace", "append"]}
            },
            "required": ["path"],
            "additionalProperties": false
        }}},
        {"type": "function", "function": {"name": "note"}}
    ])
}

fn call(function: &str, parameters: &[(&str, &str)]) -> String {
    let parameters: String =
        parameters.iter().map(|(name, value)| format!("<parameter={name}>\n{value}\n</parameter>\n")).collect();
    format!("<tool_call>\n<function={function}>\n{parameters}</function>\n</tool_call>")
}

/// A model that writes a call of every kind the check counts: two valid,
/// one each missing a required parameter, with a wrong type, outside an
/// enum, with an undeclared parameter, breaking a pattern, and calling an
/// undeclared tool, then a block that does not parse.
fn every_kind() -> String {
    [
        "Let me look.\n".to_owned(),
        call("edit", &[("path", "a.txt"), ("line", "3"), ("mode", "append")]),
        call("note", &[("text", "anything goes")]),
        call("edit", &[("line", "3")]),
        call("edit", &[("path", "a.txt"), ("line", "three")]),
        call("edit", &[("path", "a.txt"), ("mode", "Replace")]),
        call("edit", &[("path", "a.txt"), ("force", "true")]),
        call("edit", &[("path", "a b.txt")]),
        call("search", &[("query", "x")]),
        "<tool_call>\nno function here\n</tool_call>".to_owned(),
    ]
    .concat()
}

/// What [`every_kind`] counts.
const EVERY_KIND: &[&str] = &[
    "ignis_tool_calls_total{outcome=\"valid\"} 2",
    "ignis_tool_calls_total{outcome=\"invalid\"} 6",
    "ignis_tool_calls_total{outcome=\"dropped\"} 1",
    "ignis_tool_call_violations_total{kind=\"missing\"} 1",
    "ignis_tool_call_violations_total{kind=\"type\"} 1",
    "ignis_tool_call_violations_total{kind=\"enum\"} 1",
    "ignis_tool_call_violations_total{kind=\"unknown_parameter\"} 1",
    "ignis_tool_call_violations_total{kind=\"constraint\"} 1",
    "ignis_tool_call_violations_total{kind=\"undeclared_tool\"} 1",
    "ignis_tool_strict_outside_subset_total 0",
    // No value holds a tag, so the cuts drop no text of one (#610).
    "ignis_tool_call_values_split_total 0",
];

/// `metrics`' exposition holds every line of `lines`, each times `times`.
fn assert_counts(metrics: &Metrics, lines: &[&str], times: u64) {
    let text = metrics.render();
    for line in lines {
        let (series, value) = line.rsplit_once(' ').unwrap();
        let expected = format!("\n{series} {}\n", value.parse::<u64>().unwrap() * times);
        assert!(text.contains(&expected), "no `{}` in:\n{text}", expected.trim());
    }
}

async fn post(server: &Server, path: &str, body: &JsonValue) -> (u16, String) {
    let request = Request::post(path).header("content-type", "application/json").body(Body::from(body.to_string())).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// What a stream assembles to: a chat stream's content joined, its calls
/// and its finish reason; a Responses stream's completed response. Where a
/// live stream cuts its text into chunks follows when the tokens reached
/// it, with the check or without (the recorded streams of the Responses
/// suites hold the framing).
fn assembled(body: &str) -> JsonValue {
    let events: Vec<JsonValue> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    if let Some(completed) = events.iter().find(|e| e["type"] == "response.completed") {
        return completed["response"].clone();
    }
    let (mut content, mut calls, mut finish) = (String::new(), Vec::new(), JsonValue::Null);
    for chunk in &events {
        let choice = &chunk["choices"][0];
        content.push_str(choice["delta"]["content"].as_str().unwrap_or_default());
        calls.extend(choice["delta"]["tool_calls"].as_array().cloned().unwrap_or_default());
        if !choice["finish_reason"].is_null() {
            finish = choice["finish_reason"].clone();
        }
    }
    json!({ "content": content, "tool_calls": calls, "finish_reason": finish })
}

/// A chat body's text with its wall-clock `created` second blanked: the
/// one field two runs of a request may differ in.
fn without_created(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while let Some(at) = rest.find("\"created\":") {
        let after = &rest[at + "\"created\":".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        out.push_str(&rest[..at]);
        out.push_str("\"created\":0");
        rest = &after[digits..];
    }
    out.push_str(rest);
    out
}

fn chat(stream: bool, stop: bool) -> JsonValue {
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "go"}],
        "tools": tools(),
        "max_tokens": 4000,
        "enable_thinking": false,
        "stream": stream,
    });
    if stop {
        body["stop"] = json!(["never written"]);
    }
    body
}

/// Each of chat completions' ways to deliver an answer (whole, whole with a
/// stop sequence, streamed, streamed with one) counts every call block once,
/// and sends what a server that checks nothing sends: the same bytes whole,
/// the same answer streamed.
#[tokio::test]
async fn chat_completions_count_every_kind_of_call_and_send_the_same_answer() {
    let script = every_kind();
    for (stream, stop) in [(false, false), (false, true), (true, false), (true, true)] {
        let checked = server(&[&script], false, true);
        let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
        let (status, body) = post(&checked, "/v1/chat/completions", &chat(stream, stop)).await;
        assert_eq!(status, 200, "{body}");
        let (_, unchecked) = post(&server(&[&script], false, false), "/v1/chat/completions", &chat(stream, stop)).await;
        match stream {
            false => assert_eq!(without_created(&body), without_created(&unchecked), "stop {stop}"),
            true => assert_eq!(assembled(&body), assembled(&unchecked), "stop {stop}"),
        }
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert_counts(&metrics, EVERY_KIND, 1);
        if !stream {
            let reply: JsonValue = serde_json::from_str(&body).unwrap();
            let calls = reply["choices"][0]["message"]["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 8, "every call but the dropped block is delivered, invalid or not");
            assert_eq!(calls[0]["function"]["arguments"], "{\"line\":3,\"mode\":\"append\",\"path\":\"a.txt\"}");
        }
    }
}

/// One call written by the first token, through the pipeline a stop
/// sequence runs and the first-token probe that reads the same token beside
/// it (#518): counted once.
#[tokio::test]
async fn a_call_is_counted_once_beside_the_first_token_probe() {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &[WHOLE_CALL]);
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, mock);
    let checked = Server::new(Engine::new(Box::new(scheduler)), Box::new(Chars { grammars: None })).with_metrics();
    let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
    let (status, body) = post(&checked, "/v1/chat/completions", &chat(false, true)).await;
    assert_eq!(status, 200);
    let reply: JsonValue = serde_json::from_str(&body).unwrap();
    assert_eq!(reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"path\":\"a.txt\"}");
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
    assert_counts(&metrics, &["ignis_tool_calls_total{outcome=\"valid\"} 1", "ignis_tool_calls_total{outcome=\"invalid\"} 0"], 1);
}

/// A request with no tool calls exports nothing of the family.
#[tokio::test]
async fn a_server_that_read_no_call_exports_no_tool_call_series() {
    let checked = server(&["just words"], false, true);
    let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
    let (status, _) = post(&checked, "/v1/chat/completions", &chat(false, false)).await;
    assert_eq!(status, 200);
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
    assert!(!metrics.render().contains("ignis_tool_"), "{}", metrics.render());
}

fn response_body(stream: bool) -> JsonValue {
    let tools: Vec<JsonValue> = tools()
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            let mut flat = tool["function"].clone();
            flat["type"] = json!("function");
            flat
        })
        .collect();
    json!({ "model": MODEL, "input": "go", "tools": tools, "max_output_tokens": 4000, "enable_thinking": false, "stream": stream })
}

/// The Responses API counts every call block as chat completions do, on
/// HTTP whole and streamed and over the WebSocket, and sends the same events.
#[tokio::test]
async fn the_responses_api_counts_every_kind_of_call_on_http_and_websocket() {
    let script = every_kind();
    for stream in [false, true] {
        let checked = server(&[&script], false, true);
        let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
        let (status, body) = post(&checked, "/v1/responses", &response_body(stream)).await;
        assert_eq!(status, 200, "{body}");
        let (_, unchecked) = post(&server(&[&script], false, false), "/v1/responses", &response_body(stream)).await;
        match stream {
            false => assert_eq!(body, unchecked),
            true => assert_eq!(assembled(&body), assembled(&unchecked)),
        }
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert_counts(&metrics, EVERY_KIND, 1);
    }

    let checked = server(&[&script, &script], false, true);
    let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
    let live = responses::live(checked).await;
    let mut socket = responses::socket(&live).await;
    for _ in 0..2 {
        let mut event = response_body(false);
        event["type"] = json!("response.create");
        event.as_object_mut().unwrap().remove("stream");
        responses::send(&mut socket, event).await;
        let events = responses::events_until(&mut socket, responses::terminal).await;
        assert_eq!(events.last().unwrap()["type"], "response.completed", "{events:?}");
        let calls = events.iter().filter(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "function_call").count();
        assert_eq!(calls, 8);
    }
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 2").await;
    assert_counts(&metrics, EVERY_KIND, 2);
}

/// Under `parallel_tool_calls: false` (#609) the turn ends where its first
/// block closes, and that block is the only one counted: a first block that
/// does not parse is counted dropped, and still ends the turn.
#[tokio::test]
async fn under_one_call_only_the_block_that_ends_the_turn_is_counted() {
    let dropped_first = ["<tool_call>\nno function here\n</tool_call>".to_owned(), call("edit", &[("path", "a.txt")])].concat();
    let cases: [(String, &[&str]); 2] = [
        (
            every_kind(),
            &["ignis_tool_calls_total{outcome=\"valid\"} 1", "ignis_tool_calls_total{outcome=\"invalid\"} 0", "ignis_tool_calls_total{outcome=\"dropped\"} 0"],
        ),
        (
            dropped_first,
            &["ignis_tool_calls_total{outcome=\"valid\"} 0", "ignis_tool_calls_total{outcome=\"invalid\"} 0", "ignis_tool_calls_total{outcome=\"dropped\"} 1"],
        ),
    ];
    for (script, counts) in &cases {
        for stream in [false, true] {
            let checked = server(&[script], false, true);
            let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
            let mut body = chat(stream, false);
            body["parallel_tool_calls"] = json!(false);
            let (status, served) = post(&checked, "/v1/chat/completions", &body).await;
            assert_eq!(status, 200, "{served}");
            assert_counts(&metrics, counts, 1);
        }
        let checked = server(&[script], false, true);
        let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
        let mut body = response_body(false);
        body["parallel_tool_calls"] = json!(false);
        let (status, served) = post(&checked, "/v1/responses", &body).await;
        assert_eq!(status, 200, "{served}");
        assert_counts(&metrics, counts, 1);
    }
}

/// Strict tools whose parameters lie outside the strict subset are counted,
/// one per request that sends one, as the request arrives: a strict tool
/// inside the subset and a free one are not counted. Chat completions then
/// refuse the request (GitHub #611, `held_tools_http`); the Responses API
/// serves it as before until #613.
fn strict_tools() -> JsonValue {
    json!([
        // Outside: an object not closed, a property not required.
        {"type": "function", "function": {"name": "open", "strict": true, "parameters": {
            "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}, "required": ["a"]
        }}},
        // Outside: a keyword the subset does not have.
        {"type": "function", "function": {"name": "short", "strict": true, "parameters": {
            "type": "object", "properties": {"a": {"type": "string", "minLength": 2}}, "required": ["a"],
            "additionalProperties": false
        }}},
        {"type": "function", "function": {"name": "inside", "strict": true, "parameters": {
            "type": "object", "properties": {"a": {"type": ["string", "null"]}}, "required": ["a"],
            "additionalProperties": false
        }}},
        {"type": "function", "function": {"name": "free", "strict": false, "parameters": {
            "type": "object", "properties": {"a": {"type": "string"}}
        }}}
    ])
}

#[tokio::test]
async fn strict_tools_outside_the_subset_are_counted_as_they_arrive() {
    let script = call("open", &[("a", "x")]);
    let mut body = chat(false, false);
    body["tools"] = strict_tools();
    let checked = server(&[&script], true, true);
    let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
    let (status, refused) = post(&checked, "/v1/chat/completions", &body).await;
    assert_eq!(status, 400, "{refused}");
    assert!(refused.contains("tools[0].function.parameters"), "{refused}");
    let (_, unchecked) = post(&server(&[&script], true, false), "/v1/chat/completions", &body).await;
    assert_eq!(refused, unchecked);
    assert_counts(&metrics, &["ignis_tool_strict_outside_subset_total 2"], 1);

    // The Responses API's flat tools, the same count again, and served.
    let mut flat = response_body(false);
    flat["tools"] = strict_tools()
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| {
            let mut flat = tool["function"].clone();
            flat["type"] = json!("function");
            flat
        })
        .collect();
    let (status, served) = post(&checked, "/v1/responses", &flat).await;
    assert_eq!(status, 200, "{served}");
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
    assert_counts(&metrics, &["ignis_tool_strict_outside_subset_total 4", "ignis_tool_calls_total{outcome=\"valid\"} 1"], 1);

    // A model without the grammar's vocabulary (Flash-Next has none) has no
    // subset to read them by.
    let without = server(&[""], false, true);
    let metrics = Arc::clone(without.metrics.as_ref().unwrap());
    let (status, _) = post(&without, "/v1/chat/completions", &body).await;
    assert_eq!(status, 200);
    responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
    assert!(!metrics.render().contains("ignis_tool_"));
}

/// Where a value ends (#610): a model that writes a value holding a
/// `</parameter>` line, which the parser cuts there, dropping the text after
/// it that a held tool's confirmed close would keep, and three it does not
/// count: a plain value, one holding its call's own close (the residue,
/// which both rules cut), and one whose call closes with its required path
/// missing (which the confirmed close would not close).
fn split_values() -> String {
    [
        call("edit", &[("path", "a.txt\n</parameter>\nnot a step"), ("mode", "append")]),
        call("edit", &[("path", "a.txt")]),
        call("edit", &[("path", "x\n</function>\n</tool_call>\ny")]),
        call("edit", &[("mode", "a\n</parameter>\nb")]),
    ]
    .concat()
}

/// The value whose text the parser dropped is counted split, on
/// chat completions and on the Responses API, whole and streamed, and each
/// response is what a server that checks nothing sends.
#[tokio::test]
async fn a_value_the_confirmed_close_would_read_on_is_counted_split_and_served_as_before() {
    let script = split_values();
    let counts = &["ignis_tool_call_values_split_total 1", "ignis_tool_calls_total{outcome=\"valid\"} 3", "ignis_tool_calls_total{outcome=\"invalid\"} 1"];
    for stream in [false, true] {
        let checked = server(&[&script], false, true);
        let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
        let (status, body) = post(&checked, "/v1/chat/completions", &chat(stream, false)).await;
        assert_eq!(status, 200, "{body}");
        let (_, unchecked) = post(&server(&[&script], false, false), "/v1/chat/completions", &chat(stream, false)).await;
        match stream {
            false => {
                assert_eq!(without_created(&body), without_created(&unchecked));
                let reply: JsonValue = serde_json::from_str(&body).unwrap();
                let calls = reply["choices"][0]["message"]["tool_calls"].as_array().unwrap();
                assert_eq!(calls[0]["function"]["arguments"], "{\"mode\":\"append\",\"path\":\"a.txt\"}", "cut, as ever");
            }
            true => assert_eq!(assembled(&body), assembled(&unchecked)),
        }
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert_counts(&metrics, counts, 1);

        let checked = server(&[&script], false, true);
        let metrics = Arc::clone(checked.metrics.as_ref().unwrap());
        let (status, body) = post(&checked, "/v1/responses", &response_body(stream)).await;
        assert_eq!(status, 200, "{body}");
        let (_, unchecked) = post(&server(&[&script], false, false), "/v1/responses", &response_body(stream)).await;
        match stream {
            false => assert_eq!(body, unchecked),
            true => assert_eq!(assembled(&body), assembled(&unchecked)),
        }
        responses::metric_reaches(&metrics, "ignis_requests_completed_total 1").await;
        assert_counts(&metrics, counts, 1);
    }
}
