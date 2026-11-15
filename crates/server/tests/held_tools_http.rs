//! Held tools on chat completions (GitHub #611, spec #582, ADR 0094): a
//! **strict tool**'s calls are held to its `parameters` as the model writes
//! them, end to end over the real router on `MockCompute` (ADR 0006).
//!
//! The "model" follows a script ([`MockCompute::script`]) that misses on
//! purpose: a parameter name of its own, an invented parameter, an
//! undeclared function, a value outside an enum, a required parameter
//! missing at an early close; and a file whose content holds the dialect's
//! own tags, which is no miss. Each draw of the content is checked against
//! the tool calls' grammar before the next round consumes it, and a miss is
//! repaired. The mock's repair ranks what the script was going to write next
//! above the rest, then the vocabulary in order, so a script that writes the
//! right number of tokens after a miss comes back in line with it.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::Value as JsonValue;
use tower::ServiceExt;

use ignis_core::compute::ModelFamily;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig, TokenId};
use ignis_grammar::json::Json;
use ignis_grammar::{GrammarCache, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

#[path = "support/responses.rs"]
mod responses;

const MODEL: &str = "test-model";
const CHAT: &str = "/v1/chat/completions";

// ── the test vocabulary ─────────────────────────────────────────────────
//
// Printable ASCII and the line break, each its own code point; the model's
// EOS, the reasoning's close and the call's two tags are added tokens, read
// by the tool calls' grammar as the text the decoder gives them.

const STOP: TokenId = 5;
const THINK_END: TokenId = 6;
const OPEN: TokenId = 7;
const CLOSE: TokenId = 8;
const VOCAB: TokenId = 128;

const TAGS: [(TokenId, &str); 3] = [(THINK_END, "</think>"), (OPEN, "<tool_call>"), (CLOSE, "</tool_call>")];

fn encode(text: &str) -> Vec<TokenId> {
    let mut ids = Vec::new();
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some((id, tag)) = TAGS.iter().find(|(_, tag)| rest.starts_with(tag)) {
            ids.push(*id);
            rest = &rest[tag.len()..];
        } else {
            ids.push(c as TokenId);
            rest = &rest[c.len_utf8()..];
        }
    }
    ids
}

/// The placeholder's prompt, the vocabulary's text back out, its grammars.
struct Vocabulary {
    grammars: Arc<GrammarCache>,
}

impl Vocabulary {
    fn new() -> Self {
        let entries = (0..VOCAB)
            .map(|id| match id {
                STOP | THINK_END | OPEN | CLOSE => VocabEntry::Control,
                10 | 32..=126 => VocabEntry::Text(vec![id as u8]),
                _ => VocabEntry::Absent,
            })
            .collect();
        let added = TAGS.iter().map(|(id, tag)| (*id, tag.as_bytes().to_vec())).collect();
        Self { grammars: Arc::new(GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES).with_added_texts(added)) }
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
        match TAGS.iter().find(|(id, _)| *id == token) {
            Some((_, tag)) => tag.to_string(),
            None => char::from_u32(token).map(String::from).unwrap_or_default(),
        }
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server whose first request writes `script`, then stops, with metrics.
fn server(script: &str, family: ModelFamily) -> (Server, Arc<MockCompute>) {
    let mock = Arc::new(MockCompute::new());
    mock.script(0, &encode(script));
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, mock.clone());
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(Vocabulary::new()))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
        .with_family(family)
        .with_metrics();
    (server, mock)
}

/// `write_file`, strict: `path` and `content` free strings, `mode` one of
/// two, in this order (not the alphabet's).
const WRITE_FILE: &str = r#"{"type":"function","function":{"name":"write_file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"},"mode":{"type":"string","enum":["w","a"]}},"required":["path","content","mode"],"additionalProperties":false}}}"#;
/// `search`, not strict.
const SEARCH: &str = r#"{"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}"#;

/// A chat request with `tools` (their JSON as sent), thinking off, greedy.
fn chat(tools: &[&str], stream: bool, max_tokens: u32) -> String {
    format!(
        r#"{{"model":"{MODEL}","messages":[{{"role":"user","content":"write it"}}],"tools":[{}],"enable_thinking":false,"temperature":0,"max_tokens":{max_tokens},"stream":{stream}}}"#,
        tools.join(",")
    )
}

/// A call in the template's form.
fn call(name: &str, args: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in args {
        out.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    out.push_str("</function>\n</tool_call>");
    out
}

async fn post(server: &Server, body: &str) -> (u16, JsonValue, String) {
    let request = Request::builder().method("POST").uri(CHAT).header("content-type", "application/json").body(Body::from(body.to_owned())).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    let json = serde_json::from_str(&text).unwrap_or(JsonValue::Null);
    (status, json, text)
}

/// What a whole response delivered: its content, its calls (name and
/// arguments as sent), its finish reason.
async fn served(server: &Server, body: &str) -> (String, Vec<(String, String)>, String) {
    let (status, json, text) = post(server, body).await;
    assert_eq!(status, 200, "{text}");
    let message = &json["choices"][0]["message"];
    let calls = message["tool_calls"]
        .as_array()
        .map(|calls| calls.iter().map(|c| (c["function"]["name"].as_str().unwrap().to_owned(), c["function"]["arguments"].as_str().unwrap().to_owned())).collect())
        .unwrap_or_default();
    (message["content"].as_str().unwrap_or_default().to_owned(), calls, json["choices"][0]["finish_reason"].as_str().unwrap().to_owned())
}

/// Whether `arguments` validate against `tool`'s `parameters`, as the
/// independent reading of #608's validator has it.
fn valid(tool: &str, arguments: &str) -> bool {
    let tool = Json::parse(tool).unwrap();
    let parameters = tool.get("function").unwrap().get("parameters").unwrap();
    ignis_grammar::validate::violations(parameters, &Json::parse(arguments).unwrap()).is_empty()
}

/// The held tool calls' structured-output counts once the request is done.
async fn tool_counts(server: &Server, requests: u64, misses: u64) {
    counted(server.metrics.as_ref().unwrap(), requests, misses).await;
}

/// [`tool_counts`] in `metrics`.
async fn counted(metrics: &ignis_server::metrics::Metrics, requests: u64, misses: u64) {
    counted_as(metrics, "tool", requests, misses).await;
}

/// The structured-output counts of `kind` once the request is done, each
/// miss repaired.
async fn structured_counts(server: &Server, kind: &str, requests: u64, misses: u64) {
    counted_as(server.metrics.as_ref().unwrap(), kind, requests, misses).await;
}

/// [`structured_counts`] in `metrics`.
async fn counted_as(metrics: &ignis_server::metrics::Metrics, kind: &str, requests: u64, misses: u64) {
    responses::metric_reaches(metrics, &format!("ignis_structured_requests_total{{type=\"{kind}\"}} {requests}")).await;
    let exposition = metrics.render();
    for line in [format!("ignis_structured_misses_total{{type=\"{kind}\"}} {misses}"), format!("ignis_structured_repairs_total{{type=\"{kind}\"}} {misses}")] {
        assert!(exposition.contains(&format!("\n{line}\n")), "no `{line}` in:\n{exposition}");
    }
}

#[tokio::test]
async fn a_strict_tools_call_is_held_and_delivered_in_the_schemas_order() {
    let script = format!("On it.\n{}", call("write_file", &[("mode", "w"), ("content", "hi"), ("path", "a.txt")]));
    let (server, mock) = server(&script, ModelFamily::Qwen38_27b);
    let (content, calls, finish) = served(&server, &chat(&[WRITE_FILE], false, 256)).await;
    assert_eq!((content.as_str(), finish.as_str()), ("On it.\n", "tool_calls"));
    assert_eq!(calls, [("write_file".to_owned(), r#"{"path":"a.txt","content":"hi","mode":"w"}"#.to_owned())]);
    tool_counts(&server, 1, 0).await;
    assert!(mock.decode_calls().iter().flatten().all(|job| job.checked), "every round of the content is checked");
}

/// The scripts that miss on purpose, each one call to `write_file`, and
/// what each comes out as: always valid, its misses repaired.
#[tokio::test]
async fn scripts_that_miss_on_purpose_all_come_out_valid_with_their_repairs_counted() {
    for (what, script, arguments, misses) in scripts_that_miss() {
        let (server, _) = server(&script, ModelFamily::Qwen38_27b);
        let (_, calls, finish) = served(&server, &chat(&[WRITE_FILE], false, 512)).await;
        assert_eq!(finish, "tool_calls", "{what}");
        assert_eq!(calls, [("write_file".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid(WRITE_FILE, arguments), "{what}: {arguments}");
        tool_counts(&server, 1, misses).await;
    }
}

/// What each script that misses on purpose is, writes, comes out as (the
/// call's arguments) and misses, the same on every surface.
fn scripts_that_miss() -> Vec<(&'static str, String, &'static str, u64)> {
    let close = "</parameter>\n</function>\n</tool_call>";
    vec![
        // The model's own name for `path`: four letters, `file`, repaired
        // one at a time to the declared one, its value kept under it.
        (
            "a parameter name of the model's own",
            "<tool_call>\n<function=write_file>\n<parameter=file>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nw\n</parameter>\n</function>\n</tool_call>".into(),
            r#"{"path":"a.txt","content":"hi","mode":"w"}"#,
            4,
        ),
        // An invented `flag`: repaired to `path`, whose value is then the
        // invented one's and, as text, the opener of the path it wrote
        // again (a written parameter's opener is a value's own text).
        (
            "an invented parameter",
            format!("<tool_call>\n<function=write_file>\n<parameter=flag>\ntrue\n</parameter>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nw\n{close}"),
            r#"{"path":"true\n</parameter>\n<parameter=path>\na.txt","content":"hi","mode":"w"}"#,
            4,
        ),
        // A function the request does not declare, of the declared one's
        // length: repaired letter by letter to `write_file`.
        (
            "an undeclared function",
            format!("<tool_call>\n<function=save_files>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nw\n{close}"),
            r#"{"path":"a.txt","content":"hi","mode":"w"}"#,
            10,
        ),
        // `x` is neither `w` nor `a`: repaired to the script's next letter
        // the enum allows, the `a` of `</parameter>`.
        (
            "a value outside an enum",
            format!("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nx\n{close}"),
            r#"{"path":"a.txt","content":"hi","mode":"a"}"#,
            1,
        ),
        // After `mode`, a held string, the close comes with `content`
        // missing: its `/` is a miss, and from there the grammar admits one
        // token a step, `parameter=content>`, over eighteen of the script's
        // (the close it tried and a word after it, none of them the token
        // admitted); the script then writes `content`'s value and the call's
        // close.
        (
            "a missing required parameter with an early close",
            format!("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=mode>\nw\n</parameter>\n</function>\n</tool_call>\nOkay.\nhi\n{close}"),
            r#"{"path":"a.txt","content":"hi","mode":"w"}"#,
            18,
        ),
        // A file about tool calls: its lines that look like the call's own
        // tags are its text, the call's every parameter written before them.
        (
            "a file's content with tag-like lines",
            format!("<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=mode>\nw\n</parameter>\n<parameter=content>\n<parameter=path>\nb.txt\n</parameter>\n</function>\nnot the end\n</tool_call> either\n{close}"),
            r#"{"path":"a.txt","content":"<parameter=path>\nb.txt\n</parameter>\n</function>\nnot the end\n</tool_call> either","mode":"w"}"#,
            0,
        ),
    ]
}

#[tokio::test]
async fn a_strict_tool_outside_what_can_be_held_is_refused_before_generation_by_name() {
    for (tool, code, param, named) in refused_tools() {
        for choice in ["\"auto\"", "\"required\""] {
            let (server, mock) = server("", ModelFamily::Qwen38_27b);
            let body = chat(&[SEARCH, &tool], false, 64).replacen("\"tools\"", &format!("\"tool_choice\":{choice},\"tools\""), 1);
            let (status, json, text) = post(&server, &body).await;
            assert_eq!(status, 400, "{tool}: {text}");
            assert_eq!((json["error"]["code"].as_str(), json["error"]["param"].as_str()), (Some(code), Some(param)), "{text}");
            let message = json["error"]["message"].as_str().unwrap();
            assert!(message.starts_with(&format!("{param}: strict tool refused, {named}")), "{message}");
            assert!(mock.prefill_calls().is_empty(), "nothing is generated");
        }
        // Flash-Next serves them, `strict` inert there until #581.
        let (server, _) = server("", ModelFamily::FlashNext);
        let (status, _, text) = post(&server, &chat(&[SEARCH, &tool], false, 64)).await;
        assert_eq!(status, 200, "{text}");
    }
}

/// Strict tools the second of a request's tools, each refused: the tool as
/// sent to chat completions, the refusal's code, its `param` there, and the
/// keyword its message names.
fn refused_tools() -> Vec<(String, &'static str, &'static str, &'static str)> {
    let strict = |parameters: &str| format!(r#"{{"type":"function","function":{{"name":"g","strict":true,"parameters":{parameters}}}}}"#);
    let object = |inner: &str| strict(&format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#));
    vec![
        (object(r#"{"type":"string","minLength":2}"#), "unsupported_keyword", "tools[1].function.parameters", "`minLength`"),
        (strict(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#), "additional_properties_not_false", "tools[1].function.parameters", "`additionalProperties`"),
        (strict(r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"],"additionalProperties":false}"#), "not_all_required", "tools[1].function.parameters", "`required`"),
        (object(r#"{"type":"string","enum":["two\nlines"]}"#), "no_writable_value", "tools[1].function.parameters", "`enum`"),
        (
            r#"{"type":"function","function":{"name":"a>b","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}"#.into(),
            "tool_name_unwritable",
            "tools[1].function.name",
            "`name`",
        ),
        (
            r#"{"type":"function","function":{"name":"search","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}"#.into(),
            "duplicate_tool_name",
            "tools[1].function.name",
            "`name`",
        ),
    ]
}

#[tokio::test]
async fn beside_a_strict_tool_a_free_tools_arguments_stay_free() {
    // `search`'s call is read by today's rule: its value ends at its first
    // `</parameter>`, and a parameter it does not declare is its own.
    let free = "<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let script = format!("{free}\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "w")]));
    let (server, _) = server(&script, ModelFamily::Qwen38_27b);
    let (_, calls, finish) = served(&server, &chat(&[SEARCH, WRITE_FILE], false, 512)).await;
    assert_eq!(finish, "tool_calls");
    assert_eq!(
        calls,
        [
            ("search".to_owned(), r#"{"limit":"many","query":"cats"}"#.to_owned()),
            ("write_file".to_owned(), r#"{"path":"a.txt","content":"hi","mode":"w"}"#.to_owned())
        ]
    );
    tool_counts(&server, 1, 0).await;
}

/// A request with no strict tool is the request it was: no grammar, so no
/// round of it is checked, and the call as the parser reads it today, its
/// keys sorted.
#[tokio::test]
async fn a_request_with_no_strict_tool_is_held_to_nothing() {
    let not_strict = WRITE_FILE.replace("\"strict\":true", "\"strict\":false");
    let script = call("write_file", &[("mode", "x"), ("path", "a.txt")]);
    let (server, mock) = server(&script, ModelFamily::Qwen38_27b);
    let (_, calls, _) = served(&server, &chat(&[&not_strict], false, 256)).await;
    assert_eq!(calls, [("write_file".to_owned(), r#"{"mode":"x","path":"a.txt"}"#.to_owned())]);
    assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked));
    assert!(mock.repair_calls().is_empty());
    assert!(!server.metrics.as_ref().unwrap().render().contains("ignis_structured_"));
}

/// Streamed, a held call arrives once complete, in one delta, its misses
/// repaired before it: what the whole response delivers.
#[tokio::test]
async fn a_streamed_held_call_arrives_only_once_complete() {
    let script = format!("Writing.\n{}\nDone.", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "x")]));
    let (whole, _) = server(&script, ModelFamily::Qwen38_27b);
    let (content, calls, finish) = served(&whole, &chat(&[WRITE_FILE], false, 256)).await;
    let (streamed, _) = server(&script, ModelFamily::Qwen38_27b);
    let (status, _, text) = post(&streamed, &chat(&[WRITE_FILE], true, 256)).await;
    assert_eq!(status, 200);
    let chunks: Vec<JsonValue> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let deltas: Vec<&JsonValue> = chunks.iter().map(|c| &c["choices"][0]["delta"]).collect();
    let streamed_content: String = deltas.iter().filter_map(|d| d["content"].as_str()).collect();
    let streamed_calls: Vec<&JsonValue> = deltas.iter().filter_map(|d| d["tool_calls"].as_array()).flatten().collect();
    assert_eq!(streamed_calls.len(), 1, "one delta carries the whole call: {text}");
    let call = &streamed_calls[0]["function"];
    assert_eq!(calls, [(call["name"].as_str().unwrap().to_owned(), call["arguments"].as_str().unwrap().to_owned())]);
    assert_eq!(streamed_content, content);
    assert!(!streamed_content.contains("<tool_call>") && !streamed_content.contains("<function"));
    let streamed_finish = chunks.iter().filter_map(|c| c["choices"][0]["finish_reason"].as_str()).last();
    assert_eq!((streamed_finish, finish.as_str()), (Some("tool_calls"), "tool_calls"), "{text}");
    assert_eq!(calls[0].1, r#"{"path":"a.txt","content":"hi","mode":"a"}"#);
}

/// A call `max_tokens` cuts is the call as generated: open, so not
/// delivered, with `finish_reason: "length"` (#583 changes how it is sent).
#[tokio::test]
async fn a_held_call_the_cap_cuts_returns_as_generated_with_length() {
    let script = format!("Writing.\n{}", call("write_file", &[("path", "a.txt"), ("content", "a long text"), ("mode", "w")]));
    let (server, _) = server(&script, ModelFamily::Qwen38_27b);
    let (content, calls, finish) = served(&server, &chat(&[WRITE_FILE], false, 40)).await;
    assert_eq!((content.as_str(), calls.len(), finish.as_str()), ("Writing.\n", 0, "length"));
}

/// `tools` is read in its key order as sent by the deserializer that read
/// it before, so a tool nested past serde_json's bound is refused with the
/// body, as before, and the same nesting in a field read as a plain value is
/// refused the same way.
#[tokio::test]
async fn a_tool_nested_past_the_parsers_bound_is_refused_with_the_body_as_before() {
    let deep = format!("{}1{}", "[".repeat(130), "]".repeat(130));
    let (server, _) = server("", ModelFamily::Qwen38_27b);
    let tool = format!(r#"{{"type":"function","function":{{"name":"g","parameters":{deep}}}}}"#);
    let (status, _, text) = post(&server, &chat(&[&tool], false, 8)).await;
    let elsewhere = chat(&[SEARCH], false, 8).replacen("\"tools\"", &format!("\"tool_choice\":{{\"x\":{{\"y\":{deep}}}}},\"tools\""), 1);
    let (other_status, _, other) = post(&server, &elsewhere).await;
    assert_eq!((status, other_status), (400, 400), "{text} / {other}");
    assert!(text.contains("tools[0].function.parameters[0]") && text.contains("recursion limit exceeded"), "{text}");
    assert!(other.contains("recursion limit exceeded"), "{other}");
}

/// With thinking on the reasoning is free, a call written there included;
/// the content is held from the token after `</think>`.
#[tokio::test]
async fn the_reasoning_is_free_and_the_content_after_it_held() {
    let script = format!("I could <tool_call>\n<function=nope> here.</think>\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "x")]));
    let (server, _) = server(&script, ModelFamily::Qwen38_27b);
    let body = chat(&[WRITE_FILE], false, 256).replace("\"enable_thinking\":false", "\"enable_thinking\":true");
    let (status, json, text) = post(&server, &body).await;
    assert_eq!(status, 200, "{text}");
    let message = &json["choices"][0]["message"];
    assert!(message["reasoning_content"].as_str().unwrap().contains("<function=nope>"), "{text}");
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], r#"{"path":"a.txt","content":"hi","mode":"a"}"#);
    tool_counts(&server, 1, 1).await;
}

// ── GitHub #615: beside `tool_choice`, `response_format` and
// `parallel_tool_calls` ──────────────────────────────────────────────────

/// `chat`'s request with `field` (its JSON) added.
fn chat_with(tools: &[&str], stream: bool, max_tokens: u32, field: &str) -> String {
    chat(tools, stream, max_tokens).replacen("\"tools\"", &format!("{field},\"tools\""), 1)
}

/// `"required"`'s forced opening, `<tool_call>` to `function`.
const REQUIRED_OPENING: &str = "<tool_call>\n<function";

/// `response_format` asking for `{"ok": <boolean>}`.
const FORMAT: &str = r#""response_format":{"type":"json_schema","json_schema":{"name":"done","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}"#;

/// Text of as many tokens as `forced` encodes to: what the model would have
/// written where the server forces it.
fn in_place_of(forced: &str) -> String {
    "Sure, I will ".chars().cycle().take(encode(forced).len()).collect()
}

/// A call forced by `tool_choice`, `"required"` or named, is held after its
/// opening: the model's own text after it misses on purpose (an undeclared
/// function, its own name for a parameter, a value outside the enum) and
/// comes out valid, its misses counted and the forced tokens none of them,
/// though the model would have written other text where they were forced.
#[tokio::test]
async fn a_forced_calls_misses_after_its_opening_are_repaired_and_its_forced_tokens_are_none() {
    let named_opening = "<tool_call>\n<function=write_file>\n";
    let cases: Vec<(&str, String, &str, u64)> = vec![
        // `save_files` repaired letter by letter, ten, and `x` to `a`.
        (
            r#""tool_choice":"required""#,
            format!(
                "{}=save_files>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nx\n</parameter>\n</function>\n</tool_call>",
                in_place_of(REQUIRED_OPENING)
            ),
            r#"{"path":"a.txt","content":"hi","mode":"a"}"#,
            11,
        ),
        // The model's own `file` for `path`, four letters.
        (
            r#""tool_choice":{"type":"function","function":{"name":"write_file"}}"#,
            format!(
                "{}<parameter=file>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nw\n</parameter>\n</function>\n</tool_call>",
                in_place_of(named_opening)
            ),
            r#"{"path":"a.txt","content":"hi","mode":"w"}"#,
            4,
        ),
    ];
    for (choice, script, arguments, misses) in cases {
        let (server, _) = server(&script, ModelFamily::Qwen38_27b);
        let (content, calls, finish) = served(&server, &chat_with(&[WRITE_FILE], false, 512, choice)).await;
        assert_eq!((content.as_str(), finish.as_str()), ("", "tool_calls"), "{choice}");
        assert_eq!(calls, [("write_file".to_owned(), arguments.to_owned())], "{choice}");
        assert!(valid(WRITE_FILE, arguments), "{choice}: {arguments}");
        tool_counts(&server, 1, misses).await;
    }
}

/// `response_format` beside a strict tool: a turn either opens a held call
/// before it answers or answers under the schema. The call the opener starts
/// is held, its miss repaired; the answer is held to the schema, its miss
/// repaired. Both count under the answer's `type`, the call's miss included.
#[tokio::test]
async fn beside_a_structured_answer_a_turn_makes_a_held_call_or_an_answer_valid_under_the_schema() {
    let calling = call("write_file", &[("mode", "x"), ("content", "hi"), ("path", "a.txt")]);
    let (calls_to, _) = server(&calling, ModelFamily::Qwen38_27b);
    let (content, calls, finish) = served(&calls_to, &chat_with(&[WRITE_FILE], false, 256, FORMAT)).await;
    assert_eq!((content.as_str(), finish.as_str()), ("", "tool_calls"));
    assert_eq!(calls, [("write_file".to_owned(), r#"{"path":"a.txt","content":"hi","mode":"a"}"#.to_owned())]);
    structured_counts(&calls_to, "json_schema", 1, 1).await;
    // One request, one type: the answer's.
    structured_counts(&calls_to, "tool", 0, 0).await;

    // The answer's `,` after `true` is a miss, repaired to the close, and
    // the script's next `"` one more, repaired to the stop.
    let (answers, _) = server(r#"{"ok":true,"why":1}"#, ModelFamily::Qwen38_27b);
    let (content, calls, finish) = served(&answers, &chat_with(&[WRITE_FILE], false, 256, FORMAT)).await;
    assert_eq!((content.as_str(), calls.len(), finish.as_str()), (r#"{"ok":true}"#, 0, "stop"));
    structured_counts(&answers, "json_schema", 1, 2).await;
}

/// With thinking on the reasoning is free and never read as a call, however
/// it looks; the hold starts at the call after it, on every turn that holds
/// a strict tool's call: `"required"`, whose literal is forced after the
/// reasoning closes, and `response_format`, whose opener hands the call to
/// the tools' grammar.
#[tokio::test]
async fn with_thinking_on_a_call_written_in_the_reasoning_is_neither_held_nor_parsed() {
    let reasoning = "I could <tool_call>\n<function=nope>\n<parameter=x>\ny\n</parameter>\n</function>\n</tool_call> here.</think>\n";
    let body = "<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nx\n</parameter>\n</function>\n</tool_call>";
    // After the model's own close and the line break drawn with it, the
    // joiner and the opening are forced in place of what it would write.
    let required = format!("{reasoning}{}=wrote_file>\n{body}", in_place_of(&format!("\n{REQUIRED_OPENING}")));
    let cases = [
        (r#""tool_choice":"required""#, required, "tool", 2),
        (FORMAT, format!("{reasoning}<tool_call>\n<function=write_file>\n{body}"), "json_schema", 1),
    ];
    for (field, script, kind, misses) in cases {
        let (server, _) = server(&script, ModelFamily::Qwen38_27b);
        let request = chat_with(&[WRITE_FILE], false, 512, field).replace("\"enable_thinking\":false", "\"enable_thinking\":true");
        let (status, json, text) = post(&server, &request).await;
        assert_eq!(status, 200, "{text}");
        let message = &json["choices"][0]["message"];
        assert!(message["reasoning_content"].as_str().unwrap().contains("<function=nope>\n<parameter=x>"), "{field}: {text}");
        let calls = message["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1, "{field}: {text}");
        assert_eq!(calls[0]["function"]["name"], "write_file", "{field}");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"path":"a.txt","content":"hi","mode":"a"}"#, "{field}");
        structured_counts(&server, kind, 1, misses).await;
    }
}

/// `parallel_tool_calls: false` beside a strict tool: the turn ends at the
/// call's confirmed close, not at the call closer its `content` holds on a
/// line of its own, where the first-occurrence rule would end it; the value
/// is delivered whole, the second call never, and nothing past the close is
/// generated for the client, streamed or not.
#[tokio::test]
async fn with_parallel_tool_calls_false_a_held_call_ends_the_turn_at_its_confirmed_close() {
    let first = call("write_file", &[("path", "a.txt"), ("mode", "w"), ("content", "first\n</tool_call>\nlast")]);
    let script = format!("{first}\n{}", call("write_file", &[("path", "b.txt"), ("mode", "w"), ("content", "no")]));
    for stream in [false, true] {
        let (server, mock) = server(&script, ModelFamily::Qwen38_27b);
        let (status, json, text) = post(&server, &chat_with(&[WRITE_FILE], stream, 512, r#""parallel_tool_calls":false"#)).await;
        assert_eq!(status, 200, "{text}");
        let chunks: Vec<JsonValue> = match stream {
            false => vec![json],
            true => text.lines().filter_map(|l| l.strip_prefix("data: ")).filter(|d| *d != "[DONE]").map(|d| serde_json::from_str(d).unwrap()).collect(),
        };
        let part = if stream { "delta" } else { "message" };
        let calls: Vec<&JsonValue> = chunks.iter().filter_map(|c| c["choices"][0][part]["tool_calls"].as_array()).flatten().collect();
        assert_eq!(calls.len(), 1, "stream {stream}: {text}");
        let whole = r#"{"path":"a.txt","content":"first\n</tool_call>\nlast","mode":"w"}"#;
        assert_eq!(calls[0]["function"]["arguments"], whole, "stream {stream}");
        let finish = chunks.iter().filter_map(|c| c["choices"][0]["finish_reason"].as_str()).last();
        assert_eq!(finish, Some("tool_calls"), "stream {stream}: {text}");
        if !stream {
            let tokens = chunks[0]["usage"]["completion_tokens"].as_u64();
            assert_eq!(tokens, Some(encode(&first).len() as u64), "the tokens up to the confirmed close");
        }
        assert!(!text.contains("b.txt"), "stream {stream}: the second call reached the client: {text}");
        let released = tokio::time::timeout(Duration::from_secs(10), async {
            while !mock.released_requests().contains(&0) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await;
        assert!(released.is_ok(), "stream {stream}: the request ran on past the close");
        tool_counts(&server, 1, 0).await;
    }
}

// ── the Responses API (GitHub #613) ──────────────────────────────────────
//
// The same hold on `/v1/responses`, over HTTP and over the WebSocket mode:
// a function tool sent with `strict: true` flat beside its name.

const RESPONSES: &str = "/v1/responses";

/// A chat tool in the Responses API's flat shape: its function's members
/// beside its `type`, in the order sent.
fn flat(tool: &str) -> String {
    let members = tool.strip_prefix(r#"{"type":"function","function":{"#).and_then(|rest| rest.strip_suffix("}}")).expect("a chat tool");
    format!(r#"{{"type":"function",{members}}}"#)
}

/// A Responses request with `tools` (their JSON as sent), thinking off,
/// greedy.
fn responses_body(tools: &[&str], stream: bool, max_output_tokens: u32) -> String {
    format!(
        r#"{{"model":"{MODEL}","input":"write it","tools":[{}],"enable_thinking":false,"temperature":0,"max_output_tokens":{max_output_tokens},"stream":{stream}}}"#,
        tools.join(",")
    )
}

/// `body` as a `response.create` frame, its text as written: a `type`
/// first, every key in the order sent.
fn create_frame(body: &str) -> String {
    body.replacen('{', "{\"type\":\"response.create\",", 1)
}

/// What a response delivered: its reasoning, its text, its calls (name and
/// arguments as sent), its status.
#[derive(Debug, PartialEq)]
struct Delivered {
    reasoning: String,
    text: String,
    calls: Vec<(String, String)>,
    status: String,
}

fn delivered(response: &JsonValue) -> Delivered {
    let items = response["output"].as_array().unwrap();
    let texts = |kind: &str| -> String {
        items
            .iter()
            .filter(|item| item["type"] == kind)
            .flat_map(|item| item["content"].as_array().unwrap())
            .filter_map(|part| part["text"].as_str())
            .collect()
    };
    let calls = items
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| (item["name"].as_str().unwrap().to_owned(), item["arguments"].as_str().unwrap().to_owned()))
        .collect();
    Delivered { reasoning: texts("reasoning"), text: texts("message"), calls, status: response["status"].as_str().unwrap().to_owned() }
}

/// What a stream of events delivered, read from its terminal event's
/// response, once held to what a client sees on the way: every call in one
/// arguments delta, its whole arguments, and text deltas that add up to the
/// response's text and never carry a call's tags.
fn delivered_by(events: &[JsonValue]) -> Delivered {
    let of = |kind: &'static str| events.iter().filter(move |event| event["type"] == kind);
    let terminal = events.last().expect("a terminal event");
    let whole = delivered(&terminal["response"]);
    let deltas: Vec<&JsonValue> = of("response.function_call_arguments.delta").collect();
    let done: Vec<&JsonValue> = of("response.function_call_arguments.done").collect();
    assert_eq!(deltas.len(), whole.calls.len(), "one arguments delta per call: {events:?}");
    for ((delta, done), (_, arguments)) in deltas.iter().zip(&done).zip(&whole.calls) {
        assert_eq!((delta["delta"].as_str(), done["arguments"].as_str()), (Some(arguments.as_str()), Some(arguments.as_str())));
    }
    let text: String = of("response.output_text.delta").map(|event| event["delta"].as_str().unwrap()).collect();
    assert_eq!(text, whole.text, "the text deltas add up to the response's text");
    assert!(!text.contains("<tool_call>") && !text.contains("<function"), "{text:?}");
    whole
}

/// The response to `body` over HTTP, whole or streamed, and the server's
/// metrics.
async fn over_http(script: &str, body: &str, stream: bool) -> (Delivered, Arc<ignis_server::metrics::Metrics>, Arc<MockCompute>) {
    let (server, mock) = server(script, ModelFamily::Qwen38_27b);
    let request = Request::builder().method("POST").uri(RESPONSES).header("content-type", "application/json").body(Body::from(body.to_owned())).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    assert_eq!(status, 200, "{text}");
    let out = match stream {
        false => delivered(&serde_json::from_str(&text).unwrap()),
        true => delivered_by(&responses::sse_events(&text).into_iter().map(|(_, event)| event).collect::<Vec<_>>()),
    };
    (out, server.metrics.clone().unwrap(), mock)
}

/// The events answering `body` as a `response.create` on the WebSocket mode
/// of a server whose first request writes `script`, and its metrics.
async fn socket_events(script: &str, family: ModelFamily, body: &str) -> (Vec<JsonValue>, Arc<ignis_server::metrics::Metrics>, Arc<MockCompute>) {
    let (server, mock) = server(script, family);
    let metrics = server.metrics.clone().unwrap();
    let live = responses::live(server).await;
    let mut socket = responses::socket(&live).await;
    responses::send_text(&mut socket, create_frame(body)).await;
    (responses::events_until(&mut socket, responses::terminal).await, metrics, mock)
}

/// The response to `body` on the WebSocket mode, and the server's metrics.
async fn over_socket(script: &str, body: &str) -> (Delivered, Arc<ignis_server::metrics::Metrics>) {
    let (events, metrics, _) = socket_events(script, ModelFamily::Qwen38_27b, body).await;
    (delivered_by(&events), metrics)
}

/// #611's scripts that miss on purpose, and one that writes the parameters
/// in an order of its own after some text, each come out on the Responses
/// API valid and in the schema's order, their misses repaired and counted,
/// the same whole, streamed and on the WebSocket mode, whose frame is
/// parsed into a value that sorts the keys of the `parameters` sent.
#[tokio::test]
async fn on_the_responses_api_the_scripts_come_out_valid_over_http_and_the_socket() {
    let own_order = ("a call in the model's own order", format!("On it.\n{}", call("write_file", &[("mode", "w"), ("content", "hi"), ("path", "a.txt")])), r#"{"path":"a.txt","content":"hi","mode":"w"}"#, 0);
    let tool = flat(WRITE_FILE);
    for (what, script, arguments, misses) in std::iter::once(own_order).chain(scripts_that_miss()) {
        let (whole, metrics, mock) = over_http(&script, &responses_body(&[&tool], false, 512), false).await;
        assert_eq!(whole.status, "completed", "{what}");
        assert_eq!(whole.calls, [("write_file".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid(WRITE_FILE, arguments), "{what}: {arguments}");
        counted(&metrics, 1, misses).await;
        assert!(mock.decode_calls().iter().flatten().all(|job| job.checked), "{what}: every round of the content is checked");
        let (streamed, metrics, _) = over_http(&script, &responses_body(&[&tool], true, 512), true).await;
        assert_eq!(streamed, whole, "{what}: streamed and whole");
        counted(&metrics, 1, misses).await;
        let (socket, metrics) = over_socket(&script, &responses_body(&[&tool], false, 512)).await;
        assert_eq!(socket, whole, "{what}: the socket and HTTP");
        counted(&metrics, 1, misses).await;
    }
}

/// A strict tool the hold refuses is refused on the Responses API as on chat
/// completions, before anything is generated, its `param` naming the tool's
/// field where the request sent it: flat beside the tool's `type`, or in its
/// `function` for a tool sent in the chat shape, which the Responses API
/// takes as it is. The WebSocket mode answers the same refusal as an `error`
/// event; Flash-Next serves the tool, `strict` inert there.
#[tokio::test]
async fn on_the_responses_api_a_refused_strict_tool_is_named_where_it_was_sent() {
    let as_chat = refused_tools().into_iter().map(|(tool, code, param, named)| (tool, code, param.to_owned(), named));
    let as_flat = refused_tools().into_iter().map(|(tool, code, param, named)| (flat(&tool), code, param.replacen(".function.", ".", 1), named));
    for (tool, code, param, named) in as_flat.chain(as_chat.take(1)) {
        let body = responses_body(&[&flat(SEARCH), &tool], false, 64);
        let (server, mock) = server("", ModelFamily::Qwen38_27b);
        let request = Request::builder().method("POST").uri(RESPONSES).header("content-type", "application/json").body(Body::from(body.clone())).unwrap();
        let response = server.app().oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let json: JsonValue = serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
        assert_eq!(status, 400, "{tool}: {json}");
        let error = &json["error"];
        assert_eq!((error["code"].as_str(), error["param"].as_str()), (Some(code), Some(param.as_str())), "{json}");
        assert!(error["message"].as_str().unwrap().starts_with(&format!("{param}: strict tool refused, {named}")), "{json}");
        assert!(mock.prefill_calls().is_empty(), "nothing is generated");
        let (events, _, mock) = socket_events("", ModelFamily::Qwen38_27b, &body).await;
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!((events[0]["type"].as_str(), events[0]["status"].as_u64()), (Some("error"), Some(400)), "{events:?}");
        assert_eq!(&events[0]["error"], error, "the socket's refusal is HTTP's");
        assert!(mock.prefill_calls().is_empty(), "nothing is generated on the socket");
        // A warm-up, which generates nothing, refuses it too.
        let (events, _, mock) = socket_events("", ModelFamily::Qwen38_27b, &body.replacen('{', "{\"generate\":false,", 1)).await;
        assert_eq!((events.len(), &events[0]["error"]), (1, error), "a warm-up's refusal: {events:?}");
        assert!(mock.prefill_calls().is_empty(), "nothing is prefilled for the warm-up");
        let (events, _, _) = socket_events("", ModelFamily::FlashNext, &body).await;
        assert_eq!(events.last().unwrap()["type"], "response.completed", "{events:?}");
    }
}

/// A Responses request whose tools are not strict, `strict: false` or no
/// `strict` at all (normalized only from #614), is held to nothing: no
/// round of it is checked, and its call is read as today, its keys sorted.
#[tokio::test]
async fn on_the_responses_api_a_request_with_no_strict_tool_is_held_to_nothing() {
    let script = call("write_file", &[("mode", "x"), ("path", "a.txt")]);
    for tool in [flat(WRITE_FILE).replace("\"strict\":true", "\"strict\":false"), flat(WRITE_FILE).replace("\"strict\":true,", "")] {
        let body = responses_body(&[&tool], false, 256);
        let (whole, metrics, mock) = over_http(&script, &body, false).await;
        assert_eq!(whole.calls, [("write_file".to_owned(), r#"{"mode":"x","path":"a.txt"}"#.to_owned())], "{tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked));
        assert!(mock.repair_calls().is_empty());
        assert!(!metrics.render().contains("ignis_structured_"));
        let (socket, _) = over_socket(&script, &body).await;
        assert_eq!(socket, whole, "{tool}: the socket and HTTP");
    }
}

/// #611's other scripts on the Responses API, over HTTP and the WebSocket
/// mode: beside a strict tool a free tool's arguments stay free, and with
/// thinking on the reasoning is free, a call written there included, and the
/// content after its `</think>` held.
#[tokio::test]
async fn on_the_responses_api_a_free_tool_and_the_reasoning_stay_free() {
    let free = "<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let beside = format!("{free}\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "w")]));
    let reasoned = format!("I could <tool_call>\n<function=nope> here.</think>\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "x")]));
    let thinking = responses_body(&[&flat(WRITE_FILE)], false, 256).replace("\"enable_thinking\":false", "\"enable_thinking\":true");
    // Each script, its request, the calls it delivers, what its reasoning
    // holds, and its misses.
    let cases = [
        (beside, responses_body(&[&flat(SEARCH), &flat(WRITE_FILE)], false, 512), r#"[["search","{\"limit\":\"many\",\"query\":\"cats\"}"],["write_file","{\"path\":\"a.txt\",\"content\":\"hi\",\"mode\":\"w\"}"]]"#, "", 0),
        (reasoned, thinking, r#"[["write_file","{\"path\":\"a.txt\",\"content\":\"hi\",\"mode\":\"a\"}"]]"#, "<function=nope>", 1),
    ];
    for (script, body, calls, reasoning, misses) in cases {
        let calls: Vec<(String, String)> = serde_json::from_str(calls).unwrap();
        let (whole, metrics, _) = over_http(&script, &body, false).await;
        assert_eq!(whole.calls, calls, "{script}");
        assert!(whole.reasoning.contains(reasoning) && whole.reasoning.is_empty() == reasoning.is_empty(), "{whole:?}");
        counted(&metrics, 1, misses).await;
        let (socket, metrics) = over_socket(&script, &body).await;
        assert_eq!(socket, whole, "{script}: the socket and HTTP");
        counted(&metrics, 1, misses).await;
    }
}

/// #615 on the Responses API, over HTTP, whole and streamed, and over the
/// WebSocket mode, through the code chat completions take: beside
/// `text.format` a turn makes a held call or an answer valid under the
/// schema, both counted under `type="json_schema"`; a `"required"` call
/// missing after its forced opening comes out valid, its misses under
/// `type="tool"` and none for the forced tokens.
#[tokio::test]
async fn on_the_responses_api_text_format_and_a_forced_call_beside_a_strict_tool_are_held_as_on_chat() {
    let format = r#""text":{"format":{"type":"json_schema","name":"done","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}"#;
    let held = ("write_file".to_owned(), r#"{"path":"a.txt","content":"hi","mode":"a"}"#.to_owned());
    let delivered = |text: &str, calls: Vec<(String, String)>| Delivered { reasoning: String::new(), text: text.to_owned(), calls, status: "completed".to_owned() };
    let forced = format!(
        "{}=save_files>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nx\n</parameter>\n</function>\n</tool_call>",
        in_place_of(REQUIRED_OPENING)
    );
    // Each case: the field beside the tool, the script, what it delivers,
    // and the type its misses count under.
    let cases = [
        (format, call("write_file", &[("mode", "x"), ("content", "hi"), ("path", "a.txt")]), delivered("", vec![held.clone()]), "json_schema", 1),
        // The `,` after `true` repaired to the close, then the `"` to the stop.
        (format, r#"{"ok":true,"why":1}"#.to_owned(), delivered(r#"{"ok":true}"#, vec![]), "json_schema", 2),
        (r#""tool_choice":"required""#, forced, delivered("", vec![held.clone()]), "tool", 11),
    ];
    for (field, script, expected, kind, misses) in cases {
        let body = |stream: bool| responses_body(&[&flat(WRITE_FILE)], stream, 512).replacen("\"tools\"", &format!("{field},\"tools\""), 1);
        let (whole, metrics, _) = over_http(&script, &body(false), false).await;
        assert_eq!(whole, expected, "{field}: {script}");
        counted_as(&metrics, kind, 1, misses).await;
        let (streamed, _, _) = over_http(&script, &body(true), true).await;
        assert_eq!(streamed, whole, "{field}: streamed and whole");
        let (socket, metrics) = over_socket(&script, &body(false)).await;
        assert_eq!(socket, whole, "{field}: the socket and HTTP");
        counted_as(&metrics, kind, 1, misses).await;
    }
}
