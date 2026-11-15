//! Held tools on chat completions (GitHub #611, spec #582, ADR 0094): a
//! **strict tool**'s calls are held to its `parameters` as the model writes
//! them, end to end over the real router on `MockCompute` (ADR 0006).
//!
//! The "model" follows a script ([`MockCompute::script`]) that misses on
//! purpose: a parameter name of its own, an invented parameter, an
//! undeclared function, a value outside an enum, a required parameter
//! missing at an early close; a word for a number, a string for a boolean,
//! an object missing a key, a value outside its bounds (#612); and a file
//! whose content holds the dialect's own tags, which is no miss. Each draw
//! of the content is checked against
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
use ignis_server::api::ToolArguments;
use ignis_server::Server;

#[path = "support/responses.rs"]
mod responses;

#[path = "support/held_tools_corpus.rs"]
mod corpus;

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

/// `edit`, strict and typed (#612): a path, an integer `line` from 1 to
/// 1000, a boolean, an object holding a list of edits, and a string that may
/// be null.
const EDIT: &str = r#"{"type":"function","function":{"name":"edit","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer","minimum":1,"maximum":1000},"dry_run":{"type":"boolean"},"change":{"type":"object","properties":{"edits":{"type":"array","items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"],"additionalProperties":false}}},"required":["edits"],"additionalProperties":false},"note":{"type":["string","null"]}},"required":["path","line","dry_run","change","note"],"additionalProperties":false}}}"#;

/// Scripts that write a typed value wrong, each one call to `edit`, and what
/// each comes out as: always valid, its misses repaired, the nullable `note`
/// that none of them writes delivered `null`.
#[tokio::test]
async fn scripts_that_write_a_typed_value_wrong_come_out_valid_with_their_repairs_counted() {
    for (what, script, arguments, misses) in typed_scripts() {
        let (server, _) = server(&script, ModelFamily::Qwen38_27b);
        let (_, calls, finish) = served(&server, &chat(&[EDIT], false, 512)).await;
        assert_eq!(finish, "tool_calls", "{what}");
        assert_eq!(calls, [("edit".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid(EDIT, arguments), "{what}: {arguments}");
        tool_counts(&server, 1, misses).await;
    }
}

/// What each script that writes a typed value wrong is, writes, comes out
/// as (the call's arguments) and misses, the same on every surface.
fn typed_scripts() -> Vec<(&'static str, String, &'static str, u64)> {
    let edit = |values: &[(&str, &str)]| call("edit", values);
    let change = ("change", r#"{"edits": [{"old": "x", "new": "y"}]}"#);
    vec![
        // `t` cannot start an integer from 1 to 1000, and no later token of
        // the script can: the lowest the mask permits, `1`. Then `welve`,
        // five misses more, each repaired to the script's next line break,
        // the whitespace a number may be followed by.
        (
            "a word for a number",
            edit(&[("path", "a.txt"), ("line", "twelve"), ("dry_run", "false"), change]),
            r#"{"path":"a.txt","line":1,"dry_run":false,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}"#,
            6,
        ),
        // `"no"` for a boolean: each of its four tokens a miss, repaired to
        // the script's next letter that spells `true` (the `t` and `r` of
        // `parameter`, the `u` of `function`, an `e`).
        (
            "a string for a boolean",
            edit(&[("path", "a.txt"), ("line", "12"), ("dry_run", "\"no\""), change]),
            r#"{"path":"a.txt","line":12,"dry_run":true,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}"#,
            4,
        ),
        // The first edit closes without its `new`: that `}` is a miss,
        // repaired to the second edit's `,`, and the grammar then writes the
        // missing key and its value from the second edit's own tokens, the
        // rest of which is repaired into the list's close.
        (
            "an object missing a key",
            edit(&[("path", "a.txt"), ("line", "12"), ("dry_run", "false"), ("change", r#"{"edits": [{"old": "x"}, {"old": "y", "new": "z"}]}"#)]),
            r#"{"path":"a.txt","line":12,"dry_run":false,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}"#,
            22,
        ),
        // `2000` is past the maximum at its fourth digit: repaired to the
        // line break after it, so `200`.
        (
            "a value outside its bounds",
            edit(&[("path", "a.txt"), ("line", "2000"), ("dry_run", "false"), change]),
            r#"{"path":"a.txt","line":200,"dry_run":false,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}"#,
            1,
        ),
        // No miss: the parameters in the model's own order, `note` left out.
        (
            "every value right, the nullable one left out",
            edit(&[("line", "12"), ("path", "a.txt"), change, ("dry_run", "true")]),
            r#"{"path":"a.txt","line":12,"dry_run":true,"change":{"edits":[{"old":"x","new":"y"}]},"note":null}"#,
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
/// keys sorted. On chat completions a tool sent without `strict` is such a
/// tool, as OpenAI's default there (GitHub #614 normalizes it on the
/// Responses API only).
#[tokio::test]
async fn a_request_with_no_strict_tool_is_held_to_nothing() {
    let script = call("write_file", &[("mode", "x"), ("path", "a.txt")]);
    for tool in [WRITE_FILE.replace("\"strict\":true", "\"strict\":false"), WRITE_FILE.replace("\"strict\":true,", "")] {
        let (server, mock) = server(&script, ModelFamily::Qwen38_27b);
        let (_, calls, _) = served(&server, &chat(&[&tool], false, 256)).await;
        assert_eq!(calls, [("write_file".to_owned(), r#"{"mode":"x","path":"a.txt"}"#.to_owned())], "{tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked));
        assert!(mock.repair_calls().is_empty());
        assert!(!server.metrics.as_ref().unwrap().render().contains("ignis_structured_"));
    }
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

/// A named `tool_choice` beside a strict tool, thinking on, where its forced
/// opening yields: the model's run commits `</think>`, its own
/// `<tool_call>` and a line break, and the line break after them is not
/// the opening's `<`. The call it then writes to `wrote_file`, another of
/// the request's tools, is held to the named `write_file`: the name's `o`
/// and the mode `x` repaired.
#[tokio::test]
async fn a_named_tool_choice_holds_the_call_to_its_name_where_its_opening_yields() {
    let wrote = SEARCH.replace("\"search\"", "\"wrote_file\"");
    let script = format!("ok</think><tool_call>\n\n{}", &call("wrote_file", &[("mode", "x"), ("path", "a.txt"), ("content", "hi")])["<tool_call>\n".len()..]);
    let mock = Arc::new(MockCompute::with_runs(&[5]));
    mock.script(0, &encode(&script));
    let scheduler = ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, mock.clone());
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(Vocabulary::new()))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
        .with_family(ModelFamily::Qwen38_27b)
        .with_metrics();
    let named = r#""tool_choice":{"type":"function","function":{"name":"write_file"}}"#;
    let body = chat_with(&[WRITE_FILE, &wrote], false, 512, named).replace("\"enable_thinking\":false", "\"enable_thinking\":true");
    let (status, json, text) = post(&server, &body).await;
    assert_eq!(status, 200, "{text}");
    let calls = json["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "{text}");
    assert_eq!(calls[0]["function"]["name"], "write_file", "{text}");
    assert_eq!(calls[0]["function"]["arguments"], r#"{"path":"a.txt","content":"hi","mode":"a"}"#);
    tool_counts(&server, 1, 2).await;
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
    (delivered_over_http(&server, body, stream).await, server.metrics.clone().unwrap(), mock)
}

/// What `server` delivers to `body` over HTTP, whole or streamed.
async fn delivered_over_http(server: &Server, body: &str, stream: bool) -> Delivered {
    let text = responses_text(server, body).await;
    match stream {
        false => delivered(&serde_json::from_str(&text).unwrap()),
        true => delivered_by(&responses::sse_events(&text).into_iter().map(|(_, event)| event).collect::<Vec<_>>()),
    }
}

/// `server`'s answer to `body` on `/v1/responses`, a 200, as text.
async fn responses_text(server: &Server, body: &str) -> String {
    let request = Request::builder().method("POST").uri(RESPONSES).header("content-type", "application/json").body(Body::from(body.to_owned())).unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    assert_eq!(status, 200, "{text}");
    text
}

/// The events answering `body` as a `response.create` on the WebSocket mode
/// of a server whose first request writes `script`, and its metrics.
async fn socket_events(script: &str, family: ModelFamily, body: &str) -> (Vec<JsonValue>, Arc<ignis_server::metrics::Metrics>, Arc<MockCompute>) {
    let (server, mock) = server(script, family);
    let metrics = server.metrics.clone().unwrap();
    (socket_events_of(server, body).await, metrics, mock)
}

/// The events `server` answers `body` with as a `response.create` on the
/// WebSocket mode.
async fn socket_events_of(server: Server, body: &str) -> Vec<JsonValue> {
    let live = responses::live(server).await;
    let mut socket = responses::socket(&live).await;
    responses::send_text(&mut socket, create_frame(body)).await;
    responses::events_until(&mut socket, responses::terminal).await
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

/// The scripts that write a typed value wrong (#612) come out on the
/// Responses API as on chat completions: valid, the nullable one `null`,
/// their misses repaired and counted, the same whole, streamed and on the
/// WebSocket mode.
#[tokio::test]
async fn on_the_responses_api_the_typed_scripts_come_out_valid_over_http_and_the_socket() {
    let tool = flat(EDIT);
    for (what, script, arguments, misses) in typed_scripts() {
        let (whole, metrics, _) = over_http(&script, &responses_body(&[&tool], false, 512), false).await;
        assert_eq!(whole.status, "completed", "{what}");
        assert_eq!(whole.calls, [("edit".to_owned(), arguments.to_owned())], "{what}");
        counted(&metrics, 1, misses).await;
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

/// A Responses request whose tools are sent `strict: false` is held to
/// nothing, as before #614 normalized the tools sent without `strict`: no
/// round of it is checked, its call is read as today, its keys sorted, and
/// the response echoes its tools as sent, in the flat shape and the chat
/// one.
#[tokio::test]
async fn on_the_responses_api_a_request_with_no_strict_tool_is_held_to_nothing() {
    let script = call("write_file", &[("mode", "x"), ("path", "a.txt")]);
    let not_strict = WRITE_FILE.replace("\"strict\":true", "\"strict\":false");
    for tool in [flat(&not_strict), not_strict.clone()] {
        let body = responses_body(&[&tool], false, 256);
        let (response, metrics, mock) = responses_post(&script, ModelFamily::Qwen38_27b, &body).await;
        let whole = delivered(&response);
        assert_eq!(whole.calls, [("write_file".to_owned(), r#"{"mode":"x","path":"a.txt"}"#.to_owned())], "{tool}");
        assert_eq!(response["tools"], serde_json::from_str::<JsonValue>(&format!("[{tool}]")).unwrap(), "echoed as sent");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked));
        assert!(mock.repair_calls().is_empty());
        assert!(!metrics.render().contains("ignis_structured_"));
        let (socket, _) = over_socket(&script, &body).await;
        assert_eq!(socket, whole, "{tool}: the socket and HTTP");
    }
}

/// #611's other scripts on the Responses API, over HTTP and the WebSocket
/// mode: beside a strict tool a free tool's arguments stay free (a tool sent
/// `strict: false`: one sent without `strict` is normalized there, #614), and
/// with thinking on the reasoning is free, a call written there included, and
/// the content after its `</think>` held.
#[tokio::test]
async fn on_the_responses_api_a_free_tool_and_the_reasoning_stay_free() {
    let free = "<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let beside = format!("{free}\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "w")]));
    let reasoned = format!("I could <tool_call>\n<function=nope> here.</think>\n{}", call("write_file", &[("path", "a.txt"), ("content", "hi"), ("mode", "x")]));
    let thinking = responses_body(&[&flat(WRITE_FILE)], false, 256).replace("\"enable_thinking\":false", "\"enable_thinking\":true");
    // Each script, its request, the calls it delivers, what its reasoning
    // holds, and its misses.
    let cases = [
        (beside, responses_body(&[&flat(&strict_false(SEARCH)), &flat(WRITE_FILE)], false, 512), r#"[["search","{\"limit\":\"many\",\"query\":\"cats\"}"],["write_file","{\"path\":\"a.txt\",\"content\":\"hi\",\"mode\":\"w\"}"]]"#, "", 0),
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
/// `type="tool"` and none for the forced tokens; and under
/// `parallel_tool_calls: false` a value holding `</tool_call>` on its own
/// line is delivered whole, the response ending at the confirmed close.
#[tokio::test]
async fn on_the_responses_api_text_format_and_a_forced_call_beside_a_strict_tool_are_held_as_on_chat() {
    let format = r#""text":{"format":{"type":"json_schema","name":"done","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}}}"#;
    let held = ("write_file".to_owned(), r#"{"path":"a.txt","content":"hi","mode":"a"}"#.to_owned());
    let delivered = |text: &str, calls: Vec<(String, String)>| Delivered { reasoning: String::new(), text: text.to_owned(), calls, status: "completed".to_owned() };
    let forced = format!(
        "{}=save_files>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n<parameter=mode>\nx\n</parameter>\n</function>\n</tool_call>",
        in_place_of(REQUIRED_OPENING)
    );
    let first = call("write_file", &[("path", "a.txt"), ("mode", "w"), ("content", "first\n</tool_call>\nlast")]);
    let two_calls = format!("{first}\n{}", call("write_file", &[("path", "b.txt"), ("mode", "w"), ("content", "no")]));
    let whole_value = ("write_file".to_owned(), r#"{"path":"a.txt","content":"first\n</tool_call>\nlast","mode":"w"}"#.to_owned());
    // Each case: the field beside the tool, the script, what it delivers,
    // and the type its misses count under.
    let cases = [
        (format, call("write_file", &[("mode", "x"), ("content", "hi"), ("path", "a.txt")]), delivered("", vec![held.clone()]), "json_schema", 1),
        // The `,` after `true` repaired to the close, then the `"` to the stop.
        (format, r#"{"ok":true,"why":1}"#.to_owned(), delivered(r#"{"ok":true}"#, vec![]), "json_schema", 2),
        (r#""tool_choice":"required""#, forced, delivered("", vec![held.clone()]), "tool", 11),
        (r#""parallel_tool_calls":false"#, two_calls, delivered("", vec![whole_value]), "tool", 0),
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

// ── the Responses API's default (GitHub #614) ────────────────────────────
//
// A function tool sent without `strict` is normalized into strict form, as
// OpenAI's Responses API does: held where its schema allows it, served free
// where it does not, and echoed with the `strict` it was served with.

/// `find`, flat and without `strict`, as a client writes it: its objects
/// open, `path` required, `recursive`, `size`, `tag` (which may be `null`)
/// and `where` (an object of one property) optional.
const FIND: &str = r#"{"type":"function","name":"find","parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"},"size":{"type":"integer"},"tag":{"type":["string","null"]},"where":{"type":"object","properties":{"hidden":{"type":"boolean"}}}},"required":["path"]}}"#;

/// A chat tool sent `strict: false`, beside its `name`.
fn strict_false(tool: &str) -> String {
    tool.replacen("\"name\":", "\"strict\":false,\"name\":", 1)
}

/// Whether `arguments` validate against a flat tool's own `parameters`, as
/// the independent reading of #608's validator has it.
fn valid_flat(tool: &str, arguments: &str) -> bool {
    let tool = Json::parse(tool).unwrap();
    ignis_grammar::validate::violations(tool.get("parameters").unwrap(), &Json::parse(arguments).unwrap()).is_empty()
}

/// The whole response to `body` over HTTP on a model of `family` whose
/// first request writes `script`, and the server's metrics.
async fn responses_post(script: &str, family: ModelFamily, body: &str) -> (JsonValue, Arc<ignis_server::metrics::Metrics>, Arc<MockCompute>) {
    let (server, mock) = server(script, family);
    (serde_json::from_str(&responses_text(&server, body).await).unwrap(), server.metrics.clone().unwrap(), mock)
}

/// A tool sent without `strict` whose schema normalizes has its calls held:
/// an invented parameter is repaired away, the optional parameters a call
/// leaves out stay out of its arguments (`tag`, which may be `null`,
/// included), one it writes is held to its type, and a key the nested
/// object does not declare is repaired away. The arguments are valid under
/// the schema as sent, its objects open; the same whole, streamed and on the
/// WebSocket mode, the misses counted under `type="tool"`; and the response
/// echoes `strict: true`.
#[tokio::test]
async fn on_the_responses_api_a_tool_sent_without_strict_is_normalized_held_and_echoed_strict_true() {
    let cases = [
        // `flag` repaired letter by letter to `path`: the first declared
        // name's letter the script writes later is the `p` of its
        // `</parameter>`, and its value holds none of the others' first.
        ("an invented parameter", "<tool_call>\n<function=find>\n<parameter=flag>\na.md\n</parameter>\n</function>\n</tool_call>".to_owned(), r#"{"path":"a.md"}"#, 4),
        // After `true` the object may only close: each byte of `, "x": 1`
        // but its spaces repaired to the space the script writes later, its
        // `1` to the `}`, and that `}` to the line break after it.
        (
            "an optional parameter written, an object with a key of its own",
            call("find", &[("where", r#"{"hidden": true, "x": 1}"#), ("size", "3"), ("path", "src")]),
            r#"{"path":"src","size":3,"where":{"hidden":true}}"#,
            7,
        ),
    ];
    for (what, script, arguments, misses) in cases {
        let body = responses_body(&[FIND], false, 512);
        let (response, metrics, mock) = responses_post(&script, ModelFamily::Qwen38_27b, &body).await;
        let whole = delivered(&response);
        assert_eq!(whole.status, "completed", "{what}");
        assert_eq!(whole.calls, [("find".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid_flat(FIND, arguments), "{what}: {arguments}");
        assert_eq!(response["tools"][0]["strict"], true, "{what}");
        counted(&metrics, 1, misses).await;
        assert!(mock.decode_calls().iter().flatten().all(|job| job.checked), "{what}: every round of the content is checked");
        let (streamed, metrics, _) = over_http(&script, &responses_body(&[FIND], true, 512), true).await;
        assert_eq!(streamed, whole, "{what}: streamed and whole");
        counted(&metrics, 1, misses).await;
        let (events, metrics, _) = socket_events(&script, ModelFamily::Qwen38_27b, &body).await;
        assert_eq!(delivered_by(&events), whole, "{what}: the socket and HTTP");
        assert_eq!(events.last().unwrap()["response"]["tools"][0]["strict"], true, "{what}");
        counted(&metrics, 1, misses).await;
    }
}

/// A tool sent without `strict` whose schema does not normalize is served
/// free, never refused, and echoed `strict: false`: a keyword outside the
/// strict subset, an object that admits keys it does not name. On Flash-Next
/// every such tool is: its schema would normalize, and the call is still
/// read as written. No round is checked and nothing is counted, over HTTP
/// and the WebSocket mode alike.
#[tokio::test]
async fn on_the_responses_api_a_tool_whose_schema_does_not_normalize_is_served_free_and_echoed_strict_false() {
    let length = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string","minLength":1}}}}"#;
    let map = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string"},"labels":{"type":"object","additionalProperties":{"type":"string"}}}}}"#;
    let look = "<tool_call>\n<function=look>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let find = "<tool_call>\n<function=find>\n<parameter=flag>\nsrc\n</parameter>\n</function>\n</tool_call>";
    let cases = [
        (ModelFamily::Qwen38_27b, length, look, r#"{"limit":"many","query":"cats"}"#),
        (ModelFamily::Qwen38_27b, map, look, r#"{"limit":"many","query":"cats"}"#),
        (ModelFamily::FlashNext, FIND, find, r#"{"flag":"src"}"#),
    ];
    for (family, tool, script, arguments) in cases {
        let body = responses_body(&[tool], false, 256);
        let (response, metrics, mock) = responses_post(script, family, &body).await;
        let whole = delivered(&response);
        let name = Json::parse(tool).unwrap().get("name").unwrap().as_str().unwrap().to_owned();
        assert_eq!(whole.calls, [(name, arguments.to_owned())], "{tool}");
        assert_eq!(response["tools"][0]["strict"], false, "{tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked), "{tool}");
        assert!(mock.repair_calls().is_empty(), "{tool}");
        assert!(!metrics.render().contains("ignis_structured_"), "{tool}");
        let (events, _, _) = socket_events(script, family, &body).await;
        assert_eq!(delivered_by(&events), whole, "{tool}: the socket and HTTP");
        assert_eq!(events.last().unwrap()["response"]["tools"][0]["strict"], false, "{tool}");
    }
}

/// The response's `tools` are the request's, each as sent, and a tool sent
/// without `strict`, or with `null`, carries the one it was served with
/// beside its `name`, in its `function` for a tool in the chat shape: `true`
/// where it is held, `false` where its schema does not normalize and under
/// `tool_choice: "none"`, which holds nothing. A `strict` sent `true` or
/// `false` is echoed as sent, but on Flash-Next, which holds no tool's
/// calls, every tool echoes `false`, a `strict: true` inert there.
#[tokio::test]
async fn on_the_responses_api_each_tool_echoes_the_strict_it_was_served_with() {
    let length = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string","minLength":1}}}}"#;
    let chat_shaped = r#"{"type":"function","function":{"name":"list","parameters":{"type":"object","properties":{"dir":{"type":"string"}}}}}"#;
    let null = FIND.replacen("\"name\":\"find\"", "\"name\":\"seek\",\"strict\":null", 1);
    let (search, write_file) = (flat(&strict_false(SEARCH)), flat(WRITE_FILE));
    let tools = [FIND, length, chat_shaped, null.as_str(), search.as_str(), write_file.as_str()];
    // The tools as sent, each sent without `strict` given the one it is
    // echoed with.
    let echoed = |strict: [Option<bool>; 6]| -> JsonValue {
        let echoed = tools.iter().zip(strict).map(|(tool, strict)| {
            let mut tool: JsonValue = serde_json::from_str(tool).unwrap();
            if let Some(strict) = strict {
                let function = if tool.get("function").is_some() { &mut tool["function"] } else { &mut tool };
                function["strict"] = JsonValue::Bool(strict);
            }
            tool
        });
        JsonValue::Array(echoed.collect())
    };
    let held = echoed([Some(true), Some(false), Some(true), Some(true), None, None]);
    let none_held = echoed([Some(false), Some(false), Some(false), Some(false), None, None]);
    let flash_next = echoed([Some(false), Some(false), Some(false), Some(false), None, Some(false)]);
    let body = responses_body(&tools, false, 64);
    let (response, _, _) = responses_post("", ModelFamily::Qwen38_27b, &body).await;
    assert_eq!(response["tools"], held);
    let (events, _, _) = socket_events("", ModelFamily::Qwen38_27b, &body).await;
    assert_eq!(events.last().unwrap()["response"]["tools"], held, "the socket's echo");
    let (response, _, _) = responses_post("", ModelFamily::FlashNext, &body).await;
    assert_eq!(response["tools"], flash_next, "on Flash-Next");
    let (response, _, _) = responses_post("", ModelFamily::Qwen38_27b, &body.replacen("\"tools\"", "\"tool_choice\":\"none\",\"tools\"", 1)).await;
    assert_eq!(response["tools"], none_held, "under tool_choice none");
}

// ── the served close-out's corpus (GitHub #617) ──────────────────────────

/// The corpus `held_tools_gpu.rs` serves on the 27B, every request and its
/// free twin, sent to the mock 27B: none is refused, each request is held
/// (counted once under the type its grammar counts under) and each twin
/// holds no tool, and on the Responses API each tool sent without `strict`
/// is echoed `strict: true`, held. A request the card would refuse is found
/// here.
#[tokio::test]
async fn the_gpu_corpus_is_served_its_requests_held_and_their_free_twins_free() {
    for case in corpus::corpus() {
        for held in [true, false] {
            let what = format!("{} ({:?}, held {held})", case.label, case.api);
            let (server, _) = server("", ModelFamily::Qwen38_27b);
            let request = Request::builder()
                .method("POST")
                .uri(case.api.path())
                .header("content-type", "application/json")
                .body(Body::from(case.body(MODEL, held, 64)))
                .unwrap();
            let response = server.app().oneshot(request).await.unwrap();
            let status = response.status().as_u16();
            let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
            assert_eq!(status, 200, "{what}: {text}");
            let metrics = server.metrics.as_ref().unwrap();
            responses::metric_reaches(metrics, "ignis_request_duration_seconds_count 1").await;
            let exposition = metrics.render();
            let counted = |kind: &str| exposition.contains(&format!("\nignis_structured_requests_total{{type=\"{kind}\"}} 1\n"));
            match held {
                true => assert!(counted(case.counted_as()), "{what}: not held:\n{exposition}"),
                false => assert!(!counted("tool"), "{what}: the twin holds a tool:\n{exposition}"),
            }
            if held && case.api == corpus::Api::Responses {
                let echoed = corpus::Delivered::read(case.api, &serde_json::from_str(&text).unwrap()).echoed_strict;
                for (tool, strict) in corpus::declared(case.api, &case.tools_sent(true)).iter().zip(&echoed) {
                    if tool.hold == corpus::Hold::Normalized {
                        assert_eq!(strict, &JsonValue::Bool(true), "{what}: {} sent without strict is echoed held", tool.name);
                    }
                }
            }
        }
    }
}

/// The conversations the GPU legs hold to one call per turn, on the mock 27B
/// and Flash-Next, chat completions and the Responses API: the control's
/// first turn makes both calls its script writes; under
/// `parallel_tool_calls: false` the first turn ends at the first, the next
/// turn, answered, makes the other, and the one after answers.
#[tokio::test]
async fn the_gpu_legs_conversations_take_their_calls_one_turn_at_a_time() {
    let two = format!("{}\n{}", call("read_file", &[("path", "README.md")]), call("read_file", &[("path", "Cargo.toml")]));
    let one = call("read_file", &[("path", "Cargo.toml")]);
    for family in [ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
        for api in [corpus::Api::Chat, corpus::Api::Responses] {
            let tools: Vec<String> = corpus::PARALLEL_TOOLS.iter().map(|tool| if api == corpus::Api::Chat { (*tool).to_owned() } else { corpus::flat(tool) }).collect();
            let mut turns = Vec::new();
            for (scripts, parallel) in [(vec![two.clone(), "done".to_owned()], None), (vec![two.clone(), one.clone(), "done".to_owned()], Some(false))] {
                let (server, mock) = server(&scripts[0], family);
                for (i, script) in scripts.iter().enumerate().skip(1) {
                    mock.script(i as ignis_core::RequestId, &encode(script));
                }
                let send = |path: &'static str, body: String| {
                    let app = server.app();
                    async move {
                        let request = Request::builder().method("POST").uri(path).header("content-type", "application/json").body(Body::from(body)).unwrap();
                        let response = app.oneshot(request).await.unwrap();
                        let status = response.status().as_u16();
                        (status, String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap())
                    }
                };
                turns.push(corpus::conversation(send, MODEL, api, &tools, corpus::PARALLEL_PROMPT, parallel).await);
            }
            let what = format!("{family:?}, {api:?}");
            let failures = corpus::one_call_per_turn(&what, api, &turns[0], &turns[1]);
            assert!(failures.is_empty(), "{failures:?}");
            let calls = |turns: &Result<Vec<corpus::Turn>, String>| -> Vec<Vec<String>> {
                turns.as_ref().unwrap().iter().map(|turn| turn.calls.iter().map(|call| call.arguments.clone()).collect()).collect()
            };
            let (readme, cargo) = (r#"{"path":"README.md"}"#.to_owned(), r#"{"path":"Cargo.toml"}"#.to_owned());
            assert_eq!(calls(&turns[0]), [vec![readme.clone(), cargo.clone()], vec![]], "{what}: the control");
            assert_eq!(calls(&turns[1]), [vec![readme], vec![cargo], vec![]], "{what}: under false");
        }
    }
}

/// What the corpus test prints of a free twin's call: each fault where it
/// is (a missing parameter, a bound, an undeclared key in a closed object
/// and in an open one), the value there, and where two deliveries part.
#[test]
fn the_gpu_corpus_reads_a_calls_faults_where_they_are() {
    let parameters = |tool: &str| corpus::declared(corpus::Api::Chat, &[tool.to_owned()]).remove(0).parameters;
    let kinds = |tool: &str, arguments: &str| -> Vec<(String, String)> {
        corpus::faults(&parameters(tool), arguments).unwrap().into_iter().map(|fault| (fault.kind, fault.at)).collect()
    };
    let deploy = r#"{"service":"api","version":"v1.4.0","region":"eu-west","replicas":25,"extra":1}"#;
    let pair = |kind: &str, at: &str| (kind.to_owned(), at.to_owned());
    let mut found = kinds(corpus::DEPLOY, deploy);
    found.sort();
    assert_eq!(found, [pair("constraint", "/replicas"), pair("missing", "/approved_by"), pair("missing", "/notes"), pair("unknown", "/extra")]);
    let open = r#"{"path":"a","line":1,"change":{"edits":[{"old":"x","new":"y","why":"z"}]}}"#;
    assert_eq!(kinds(corpus::EDIT_OPEN, open), [pair("undeclared", "/change/edits/0/why")]);
    assert!(corpus::faults(&parameters(corpus::DEPLOY), "not json").is_none());
    assert_eq!((corpus::shown_at(deploy, "/replicas"), corpus::shown_at(deploy, "/notes")), ("25".to_owned(), "(absent)".to_owned()));
    assert_eq!(corpus::shown_at(open, "/change/edits/0"), r#"{"old":"x","new":"y","why":"z"}"#);
    assert_eq!([corpus::parting("abc", "abd"), corpus::parting("ab", "abc"), corpus::parting("a", "a")], [Some(2), Some(2), None]);
}

// ── the operator's switch (GitHub #616) ──────────────────────────────────

/// A server whose first request writes `script`, the operator's switch
/// (`model.tool_arguments`) at `held`.
fn held_server(script: &str, family: ModelFamily) -> (Server, Arc<MockCompute>) {
    let (server, mock) = server(script, family);
    (server.with_tool_arguments(ToolArguments::Held), mock)
}

/// A flat Responses tool in chat completions' shape: its members in its
/// `function`, in the order sent.
fn chat_shaped(tool: &str) -> String {
    let members = tool.strip_prefix(r#"{"type":"function","#).expect("a flat tool");
    format!(r#"{{"type":"function","function":{{{members}}}"#)
}

/// `find`'s two scripts that miss, the arguments each delivers held and its
/// misses: an invented parameter repaired away, the optional parameters it
/// leaves out left out; an optional parameter written, held to its type,
/// and a nested object's own key repaired away (as on the Responses API by
/// default, [`on_the_responses_api_a_tool_sent_without_strict_is_normalized_held_and_echoed_strict_true`]).
fn find_scripts() -> [(&'static str, String, &'static str, u64); 2] {
    [
        ("an invented parameter", "<tool_call>\n<function=find>\n<parameter=flag>\na.md\n</parameter>\n</function>\n</tool_call>".to_owned(), r#"{"path":"a.md"}"#, 4),
        (
            "an optional parameter written, an object with a key of its own",
            call("find", &[("where", r#"{"hidden": true, "x": 1}"#), ("size", "3"), ("path", "src")]),
            r#"{"path":"src","size":3,"where":{"hidden":true}}"#,
            7,
        ),
    ]
}

/// With the switch at `held`, chat completions hold a tool sent without
/// `strict`, and one sent `strict: false`, normalized as the Responses API
/// holds a tool sent without `strict`: the arguments valid under the schema
/// as sent, every round of the content checked, the misses counted under
/// `type="tool"`. At the default, `as-sent`, the same call is read as the
/// model wrote it.
#[tokio::test]
async fn with_the_switch_held_a_chat_tool_not_sent_strict_has_its_calls_held() {
    let find = chat_shaped(FIND);
    for tool in [find.clone(), strict_false(&find)] {
        for (what, script, arguments, misses) in find_scripts() {
            let (server, mock) = held_server(&script, ModelFamily::Qwen38_27b);
            let (_, calls, finish) = served(&server, &chat(&[&tool], false, 512)).await;
            assert_eq!(finish, "tool_calls", "{what}: {tool}");
            assert_eq!(calls, [("find".to_owned(), arguments.to_owned())], "{what}: {tool}");
            assert!(valid(&tool, arguments), "{what}: {arguments}");
            assert!(mock.decode_calls().iter().flatten().all(|job| job.checked), "{what}: every round of the content is checked");
            tool_counts(&server, 1, misses).await;
        }
        let (server, mock) = server(&find_scripts()[0].1, ModelFamily::Qwen38_27b);
        let (_, calls, _) = served(&server, &chat(&[&tool], false, 512)).await;
        assert_eq!(calls, [("find".to_owned(), r#"{"flag":"a.md"}"#.to_owned())], "at the default: {tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked), "at the default: {tool}");
    }
}

/// With the switch at `held`, a tool whose schema does not normalize is
/// served free, never refused, sent without `strict` or with `strict:
/// false`: a keyword outside the strict subset, an object that admits keys
/// it does not name. On chat completions and the Responses API alike its
/// call is read as written, no round checked, nothing counted, and the
/// Responses API echoes it `strict: false`.
#[tokio::test]
async fn with_the_switch_held_a_tool_whose_schema_does_not_normalize_stays_free() {
    let length = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string","minLength":1}}}}"#;
    let map = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string"},"labels":{"type":"object","additionalProperties":{"type":"string"}}}}}"#;
    let look = "<tool_call>\n<function=look>\n<parameter=query>\ncats\n</parameter>\n<parameter=limit>\nmany\n</parameter>\n</function>\n</tool_call>";
    let free = [("look".to_owned(), r#"{"limit":"many","query":"cats"}"#.to_owned())];
    for tool in [length.to_owned(), strict_false(length), map.to_owned(), strict_false(map)] {
        let (server, mock) = held_server(look, ModelFamily::Qwen38_27b);
        let (_, calls, _) = served(&server, &chat(&[&chat_shaped(&tool)], false, 256)).await;
        assert_eq!(calls, free, "chat: {tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked), "chat: {tool}");
        assert!(mock.repair_calls().is_empty(), "chat: {tool}");
        assert!(!server.metrics.as_ref().unwrap().render().contains("ignis_structured_"), "chat: {tool}");
        let (server, mock) = held_server(look, ModelFamily::Qwen38_27b);
        let response: JsonValue = serde_json::from_str(&responses_text(&server, &responses_body(&[&tool], false, 256)).await).unwrap();
        assert_eq!(delivered(&response).calls, free, "responses: {tool}");
        assert_eq!(response["tools"][0]["strict"], false, "responses: {tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked), "responses: {tool}");
        assert!(!server.metrics.as_ref().unwrap().render().contains("ignis_structured_"), "responses: {tool}");
    }
}

/// With the switch at `held`, a strict tool is the strict tool it was: one
/// outside what can be held is refused before generation with the same 400,
/// naming the same field, beside a tool the switch normalizes.
#[tokio::test]
async fn with_the_switch_held_a_strict_tool_outside_what_can_be_held_is_refused_as_before() {
    for (tool, code, param, named) in refused_tools() {
        let (server, mock) = held_server("", ModelFamily::Qwen38_27b);
        let (status, json, text) = post(&server, &chat(&[SEARCH, &tool], false, 64)).await;
        assert_eq!(status, 400, "{tool}: {text}");
        assert_eq!((json["error"]["code"].as_str(), json["error"]["param"].as_str()), (Some(code), Some(param)), "{text}");
        let message = json["error"]["message"].as_str().unwrap();
        assert!(message.starts_with(&format!("{param}: strict tool refused, {named}")), "{message}");
        assert!(mock.prefill_calls().is_empty(), "nothing is generated");
    }
}

/// With the switch at `held`, the Responses API holds a tool sent `strict:
/// false` as it holds one sent without `strict` by default: the same
/// arguments and misses whole, streamed and on the WebSocket mode, and the
/// response echoes it `strict: true`, the `strict` it was served with.
#[tokio::test]
async fn with_the_switch_held_the_responses_api_holds_a_tool_sent_strict_false_and_echoes_strict_true() {
    let tool = strict_false(FIND);
    for (what, script, arguments, misses) in find_scripts() {
        let body = responses_body(&[&tool], false, 512);
        let (server, mock) = held_server(&script, ModelFamily::Qwen38_27b);
        let response: JsonValue = serde_json::from_str(&responses_text(&server, &body).await).unwrap();
        let whole = delivered(&response);
        assert_eq!(whole.calls, [("find".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid_flat(&tool, arguments), "{what}: {arguments}");
        assert_eq!(response["tools"][0]["strict"], true, "{what}");
        assert!(mock.decode_calls().iter().flatten().all(|job| job.checked), "{what}: every round of the content is checked");
        tool_counts(&server, 1, misses).await;
        let (server, _) = held_server(&script, ModelFamily::Qwen38_27b);
        assert_eq!(delivered_over_http(&server, &responses_body(&[&tool], true, 512), true).await, whole, "{what}: streamed and whole");
        tool_counts(&server, 1, misses).await;
        let (server, _) = held_server(&script, ModelFamily::Qwen38_27b);
        let metrics = server.metrics.clone().unwrap();
        let events = socket_events_of(server, &body).await;
        assert_eq!(delivered_by(&events), whole, "{what}: the socket and HTTP");
        assert_eq!(events.last().unwrap()["response"]["tools"][0]["strict"], true, "{what}");
        counted(&metrics, 1, misses).await;
    }
}

/// With the switch at `held`, the response's `tools` echo the `strict` each
/// tool was served with, one sent `strict: false` as one sent without: `true`
/// where held, `false` where its schema does not normalize and under
/// `tool_choice: "none"`; `strict: true` as sent; and on Flash-Next, which
/// the switch does not reach, every tool `false`, as at the default.
#[tokio::test]
async fn with_the_switch_held_each_tool_echoes_the_strict_it_was_served_with() {
    let length = r#"{"type":"function","name":"look","parameters":{"type":"object","properties":{"query":{"type":"string","minLength":1}}}}"#;
    let chat_shaped_false = strict_false(r#"{"type":"function","function":{"name":"list","parameters":{"type":"object","properties":{"dir":{"type":"string"}}}}}"#);
    let (search, write_file) = (flat(&strict_false(SEARCH)), flat(WRITE_FILE));
    let length_false = strict_false(length);
    let tools = [FIND, length_false.as_str(), chat_shaped_false.as_str(), search.as_str(), write_file.as_str()];
    let echoed = |strict: [Option<bool>; 5]| -> JsonValue {
        let echoed = tools.iter().zip(strict).map(|(tool, strict)| {
            let mut tool: JsonValue = serde_json::from_str(tool).unwrap();
            if let Some(strict) = strict {
                let function = if tool.get("function").is_some() { &mut tool["function"] } else { &mut tool };
                function["strict"] = JsonValue::Bool(strict);
            }
            tool
        });
        JsonValue::Array(echoed.collect())
    };
    let body = responses_body(&tools, false, 64);
    let (server, _) = held_server("", ModelFamily::Qwen38_27b);
    let response: JsonValue = serde_json::from_str(&responses_text(&server, &body).await).unwrap();
    assert_eq!(response["tools"], echoed([Some(true), Some(false), Some(true), Some(true), None]));
    let (server, _) = held_server("", ModelFamily::Qwen38_27b);
    let none = body.replacen("\"tools\"", "\"tool_choice\":\"none\",\"tools\"", 1);
    let response: JsonValue = serde_json::from_str(&responses_text(&server, &none).await).unwrap();
    assert_eq!(response["tools"], echoed([Some(false), Some(false), Some(false), Some(false), None]), "under tool_choice none");
    let (server, _) = held_server("", ModelFamily::FlashNext);
    let response: JsonValue = serde_json::from_str(&responses_text(&server, &body).await).unwrap();
    assert_eq!(response["tools"], echoed([Some(false); 5]), "on Flash-Next");
    let (response, _, _) = responses_post("", ModelFamily::Qwen38_27b, &body).await;
    assert_eq!(response["tools"], echoed([Some(true), None, None, None, None]), "at the default: `strict: false` as sent");
}

/// On Flash-Next the switch changes nothing until it holds tool calls
/// (#581): at `held` a tool sent without `strict`, or with `strict: false`,
/// is served free on chat completions and the Responses API, its call read
/// as written, no round checked and nothing counted.
#[tokio::test]
async fn on_flash_next_the_switch_changes_nothing() {
    let script = "<tool_call>\n<function=find>\n<parameter=flag>\nsrc\n</parameter>\n</function>\n</tool_call>";
    let free = [("find".to_owned(), r#"{"flag":"src"}"#.to_owned())];
    for tool in [FIND.to_owned(), strict_false(FIND)] {
        let (server, mock) = held_server(script, ModelFamily::FlashNext);
        let (_, calls, _) = served(&server, &chat(&[&chat_shaped(&tool)], false, 256)).await;
        assert_eq!(calls, free, "chat: {tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked) && mock.repair_calls().is_empty(), "chat: {tool}");
        assert!(!server.metrics.as_ref().unwrap().render().contains("ignis_structured_"), "chat: {tool}");
        let (server, mock) = held_server(script, ModelFamily::FlashNext);
        let body = responses_body(&[&tool], false, 256);
        assert_eq!(delivered_over_http(&server, &body, false).await.calls, free, "responses: {tool}");
        assert!(mock.decode_calls().iter().flatten().all(|job| !job.checked), "responses: {tool}");
        let (server, _) = held_server(script, ModelFamily::FlashNext);
        assert_eq!(delivered_by(&socket_events_of(server, &body).await).calls, free, "the socket: {tool}");
    }
}
