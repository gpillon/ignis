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
    let metrics = server.metrics.as_ref().unwrap();
    responses::metric_reaches(metrics, &format!("ignis_structured_requests_total{{type=\"tool\"}} {requests}")).await;
    let exposition = metrics.render();
    for line in [format!("ignis_structured_misses_total{{type=\"tool\"}} {misses}"), format!("ignis_structured_repairs_total{{type=\"tool\"}} {misses}")] {
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
    let close = "</parameter>\n</function>\n</tool_call>";
    let cases: Vec<(&str, String, &str, u64)> = vec![
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
    ];
    for (what, script, arguments, misses) in cases {
        let (server, _) = server(&script, ModelFamily::Qwen38_27b);
        let (_, calls, finish) = served(&server, &chat(&[WRITE_FILE], false, 512)).await;
        assert_eq!(finish, "tool_calls", "{what}");
        assert_eq!(calls, [("write_file".to_owned(), arguments.to_owned())], "{what}");
        assert!(valid(WRITE_FILE, arguments), "{what}: {arguments}");
        tool_counts(&server, 1, misses).await;
    }
}

#[tokio::test]
async fn a_strict_tool_outside_what_can_be_held_is_refused_before_generation_by_name() {
    let strict = |parameters: &str| format!(r#"{{"type":"function","function":{{"name":"g","strict":true,"parameters":{parameters}}}}}"#);
    let object = |inner: &str| strict(&format!(r#"{{"type":"object","properties":{{"a":{inner}}},"required":["a"],"additionalProperties":false}}"#));
    let cases: Vec<(String, &str, &str, &str)> = vec![
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
    ];
    for (tool, code, param, named) in cases {
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
