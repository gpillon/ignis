//! Tool-call stream hardening (GitHub #121): end-to-end tests over the real
//! axum router, on the CPU gate (ADR 0006 — `MockCompute`, no GPU).
//!
//! Mirrors `openai_http_thinking.rs`'s harness shape (prior art): a
//! recording [`TemplateProvider`] double maps specific mock-emitted token
//! ids to specific literal text, so a test can drive a
//! `<tool_call>...</tool_call>` block — including one split across several
//! scheduler tokens — through the whole router and assert on the resulting
//! response / SSE frames.
//!
//! `MockCompute` has no real EOS token (`crates/core/src/mock.rs`): every
//! request here finishes via `max_tokens`, i.e. `FinishReason::Length`,
//! never `Stop`. The `finish_reason: "tool_calls"` enhancement
//! (`resolve_finish_reason`, only reachable from `Stop`) is therefore
//! pinned by `api.rs`'s own unit tests, not here; what these tests exercise
//! is reassembly correctness and the cut-call rule (GitHub #583: a call the
//! cap cuts reaches the client with its arguments so far), and one test
//! scripts the mock's EOS (`eos_after`) to pin the drop of a call its own
//! end leaves open.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use ignis_core::{mock::MockCompute, ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";

/// See `openai_http_thinking.rs`'s `RecordingTemplateProvider` (same
/// shape): templating delegates to the built-in placeholder, and specific
/// token ids decode to specific literal text.
struct RecordingTemplateProvider {
    inner: ignis_server::template::SimpleTemplateProvider,
    decode: HashMap<TokenId, &'static str>,
}

impl RecordingTemplateProvider {
    fn permissive(decode: HashMap<TokenId, &'static str>) -> Self {
        Self {
            inner: ignis_server::template::SimpleTemplateProvider,
            decode,
        }
    }
}

impl TemplateProvider for RecordingTemplateProvider {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[serde_json::Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        self.inner.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .map(|t| self.decode.get(t).copied().unwrap_or("?").to_string())
            .collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(RecordingTokenDecoder {
            decode: self.decode.clone(),
        })
    }
    // `decoder_starts_in_reasoning` deliberately left at the trait default
    // (`thinking.enable_thinking`), same as `openai_http_thinking.rs`.
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
}

fn harness_with(template: RecordingTemplateProvider) -> Harness {
    harness_on(Arc::new(MockCompute::new()), template)
}

/// [`harness_with`] on `compute`, which a test may script first (an EOS).
fn harness_on(compute: Arc<MockCompute>, template: RecordingTemplateProvider) -> Harness {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            ..SchedulerConfig::default()
        },
        compute,
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(template))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0);
    Harness { app: server.app() }
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

fn sse_chunks(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data:").map(|s| s.trim().to_string()))
        .filter(|l| l != "[DONE]")
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

// ── non-streaming ────────────────────────────────────────────────────────

/// One mock token carrying a whole, well-formed tool-call block plus
/// leading content text — enough to drive the parser through the real
/// HTTP path in a single decode step.
fn single_call_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(
        mock.token_for(id, 0),
        "before <tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call> after",
    );
    map
}

#[tokio::test]
async fn non_streaming_returns_a_tool_call_and_strips_it_from_content() {
    let h = harness_with(RecordingTemplateProvider::permissive(single_call_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["content"], "before  after");
    let calls = message["tool_calls"].as_array().expect("tool_calls present");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "read_file");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.txt"}));
    assert!(!body.contains("<tool_call"), "raw XML leaked: {body}");
}

// ── GitHub #583: a cut tool call ─────────────────────────────────────────

/// The issue's example, after content and a closed call, over three
/// tokens: `max_tokens: 3` cuts it after `printl`.
fn cut_call_decode_map() -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(mock.token_for(0, 0), "Writing it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>");
    map.insert(mock.token_for(0, 1), "<tool_call>\n<function=write_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n");
    map.insert(mock.token_for(0, 2), PRINTL);
    map
}

const PRINTL: &str = "<parameter=content>\nfn main() {\n    printl";

/// The cut call's arguments: exactly the issue's.
const CUT_ARGUMENTS: &str = r#"{"path":"src/main.rs","content":"fn main() {\n    printl"#;

fn cut_request(stream: bool) -> serde_json::Value {
    serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": write_file_tools(),
        "max_tokens": 3,
        "enable_thinking": false,
        "stream": stream
    })
}

/// The calls of a whole answer, `(index, name, arguments)`, its content and
/// its finish reason.
fn whole_calls(body: &str) -> (Vec<(u64, String, String)>, String, String) {
    let v: serde_json::Value = serde_json::from_str(body).unwrap();
    let message = &v["choices"][0]["message"];
    let calls = message["tool_calls"].as_array().map_or_else(Vec::new, |calls| {
        calls
            .iter()
            .enumerate()
            .map(|(index, c)| (index as u64, c["function"]["name"].as_str().unwrap().to_owned(), c["function"]["arguments"].as_str().unwrap().to_owned()))
            .collect()
    });
    (calls, message["content"].as_str().unwrap().to_owned(), v["choices"][0]["finish_reason"].as_str().unwrap().to_owned())
}

/// A call `max_tokens` cuts before it closes reaches the client as OpenAI's
/// API returns one: its name and the arguments written so far, last, with
/// `finish_reason: "length"`; the content and the closed call before it
/// unchanged.
#[tokio::test]
async fn a_call_cut_by_max_tokens_carries_its_name_and_the_arguments_written_so_far() {
    let h = harness_with(RecordingTemplateProvider::permissive(cut_call_decode_map()));
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(cut_request(false))).await;
    assert_eq!(status, 200, "{body}");
    let (calls, content, finish) = whole_calls(&body);
    assert_eq!((content.as_str(), finish.as_str()), ("Writing it.\n", "length"), "{body}");
    assert_eq!(
        calls,
        [(0, "read_file".to_owned(), r#"{"path":"src/main.rs"}"#.to_owned()), (1, "write_file".to_owned(), CUT_ARGUMENTS.to_owned())],
        "{body}"
    );
    assert!(!body.contains("<tool_call") && !body.contains("<parameter"), "no XML fragment leaked: {body}");
}

/// The same answer with a stop sequence, collected token by token: the
/// sequence written inside the call's arguments matches nothing, and the
/// cut call is the same.
#[tokio::test]
async fn a_call_cut_by_max_tokens_is_the_same_with_a_stop_sequence_it_holds() {
    let h = harness_with(RecordingTemplateProvider::permissive(cut_call_decode_map()));
    let mut request = cut_request(false);
    request["stop"] = serde_json::json!(["main"]);
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(request)).await;
    assert_eq!(status, 200, "{body}");
    let (calls, content, finish) = whole_calls(&body);
    assert_eq!((content.as_str(), finish.as_str(), calls.len()), ("Writing it.\n", "length", 2), "{body}");
    assert_eq!(calls[1], (1, "write_file".to_owned(), CUT_ARGUMENTS.to_owned()));
}

/// Cut before its function's name closes, the call names nothing: no call,
/// `length`, the content before it. Cut right after the name, its arguments
/// are `{`.
#[tokio::test]
async fn a_call_cut_before_its_name_closes_is_dropped_and_right_after_it_is_an_open_brace() {
    let mock = MockCompute::new();
    for (text, calls) in [("Reading.\n<tool_call>\n<function=read_fi", 0), ("Reading.\n<tool_call>\n<function=read_file>\n", 1)] {
        let decode = HashMap::from([(mock.token_for(0, 0), text)]);
        let h = harness_with(RecordingTemplateProvider::permissive(decode));
        let mut request = cut_request(false);
        request["max_tokens"] = serde_json::json!(1);
        let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(request)).await;
        assert_eq!(status, 200, "{body}");
        let (got, content, finish) = whole_calls(&body);
        assert_eq!((content.as_str(), finish.as_str(), got.len()), ("Reading.\n", "length", calls), "{body}");
        if calls == 1 {
            assert_eq!(got[0].1, "read_file");
            assert_eq!(got[0].2, "{");
        }
    }
}

/// A call open at the model's own end is dropped whole, as before: `stop`
/// would call it complete.
#[tokio::test]
async fn a_call_open_at_the_models_own_end_is_dropped_whole() {
    for stream in [false, true] {
        let compute = Arc::new(MockCompute::new());
        compute.eos_after(0, 3);
        let mut decode = cut_call_decode_map();
        decode.insert(compute.token_for(0, 0), "Writing it.\n");
        let h = harness_on(compute, RecordingTemplateProvider::permissive(decode));
        let mut request = cut_request(stream);
        request["max_tokens"] = serde_json::json!(8);
        let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(request)).await;
        assert_eq!(status, 200, "{body}");
        let (names, finish): (Vec<String>, String) = match stream {
            false => {
                let (calls, _, finish) = whole_calls(&body);
                (calls.into_iter().map(|c| c.1).collect(), finish)
            }
            true => {
                let chunks = sse_chunks(&body);
                let names = chunks
                    .iter()
                    .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
                    .flatten()
                    .map(|c| c["function"]["name"].as_str().unwrap().to_owned())
                    .collect();
                let finish = chunks.iter().filter_map(|c| c["choices"][0]["finish_reason"].as_str()).last().unwrap().to_owned();
                (names, finish)
            }
        };
        assert_eq!((names, finish.as_str()), (Vec::<String>::new(), "stop"), "stream {stream}: {body}");
    }
}

/// A client that sends a cut call back in its history is served: its
/// arguments are not a JSON object, and the template renders them `{}`.
#[tokio::test]
async fn a_history_holding_a_cut_call_is_still_served() {
    let mock = MockCompute::new();
    let h = harness_with(RecordingTemplateProvider::permissive(HashMap::from([(mock.token_for(0, 0), "Done.")])));
    let request = serde_json::json!({
        "messages": [
            { "role": "user", "content": "write it" },
            { "role": "assistant", "content": "Writing it.\n", "tool_calls": [{
                "id": "call_1", "type": "function", "function": { "name": "write_file", "arguments": CUT_ARGUMENTS }
            }] },
            { "role": "tool", "tool_call_id": "call_1", "content": "Not run: the reply was cut by the token limit." },
            { "role": "user", "content": "go on" }
        ],
        "tools": write_file_tools(),
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(request)).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(whole_calls(&body).1, "Done.");
}

/// The all-reasoning-no-content shape (GitHub #70): the token budget is
/// exhausted while still in the reasoning channel, so `content` is
/// genuinely empty — but distinguishably so, since `reasoning_content`
/// carries what was actually generated (never both empty, which is what a
/// silent, unreported empty completion would look like).
#[tokio::test]
async fn an_all_reasoning_generation_reports_populated_reasoning_and_empty_content() {
    let mock = MockCompute::new();
    let mut decode = HashMap::new();
    decode.insert(mock.token_for(0, 0), "still thinking, no marker yet");
    let h = harness_with(RecordingTemplateProvider::permissive(decode));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "still thinking, no marker yet");
    assert_eq!(message["content"], "");
    assert!(message.as_object().unwrap().get("tool_calls").is_none());
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

// ── streaming ────────────────────────────────────────────────────────────

/// The same tool call as `single_call_decode_map`, split across three
/// scheduler tokens so the open tag, the parameter, and the close tag each
/// land in a different SSE-triggering decode step.
fn split_call_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(mock.token_for(id, 0), "before <tool_");
    map.insert(mock.token_for(id, 1), "call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</functio");
    map.insert(mock.token_for(id, 2), "n>\n</tool_call> after");
    map
}

#[tokio::test]
async fn a_tool_call_split_across_scheduler_tokens_reassembles_over_sse() {
    let h = harness_with(RecordingTemplateProvider::permissive(split_call_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 3,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);

    let mut content = String::new();
    let mut tool_call_deltas = Vec::new();
    for chunk in &chunks {
        let delta = &chunk["choices"][0]["delta"];
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            content.push_str(c);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            tool_call_deltas.extend(calls.clone());
        }
        // No fragment of the tag survives outside a tool_calls delta.
        if delta.get("tool_calls").is_none() {
            let text = delta.to_string();
            assert!(!text.contains("<tool_call"), "leaked into a plain delta: {text}");
        }
    }
    assert_eq!(content, "before  after");
    assert_eq!(tool_call_deltas.len(), 1);
    assert_eq!(tool_call_deltas[0]["index"], 0);
    assert_eq!(tool_call_deltas[0]["function"]["name"], "read_file");
    let args: serde_json::Value = serde_json::from_str(
        tool_call_deltas[0]["function"]["arguments"].as_str().unwrap(),
    )
    .unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.txt"}));
}

/// Two tool calls in one response (acceptance criterion 2): distinct,
/// stable indices — never merged into one call.
fn two_calls_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(
        mock.token_for(id, 0),
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>",
    );
    map
}

#[tokio::test]
async fn several_tool_calls_in_one_response_stream_with_distinct_indices() {
    let h = harness_with(RecordingTemplateProvider::permissive(two_calls_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 1,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);
    let mut tool_call_deltas = Vec::new();
    for chunk in &chunks {
        if let Some(calls) = chunk["choices"][0]["delta"].get("tool_calls").and_then(|v| v.as_array()) {
            tool_call_deltas.extend(calls.clone());
        }
    }
    assert_eq!(tool_call_deltas.len(), 2);
    assert_eq!(tool_call_deltas[0]["index"], 0);
    assert_eq!(tool_call_deltas[0]["function"]["name"], "a");
    assert_eq!(tool_call_deltas[1]["index"], 1);
    assert_eq!(tool_call_deltas[1]["function"]["name"], "b");
    assert_ne!(tool_call_deltas[0]["id"], tool_call_deltas[1]["id"]);
}

/// Streamed, the cut call is sent once, as one complete `tool_calls` delta
/// (its id, its name, the arguments so far) with the next `index`, the last
/// delta before the `length` finish chunk.
#[tokio::test]
async fn a_streamed_call_cut_by_max_tokens_is_one_whole_delta_before_the_length_finish() {
    let h = harness_with(RecordingTemplateProvider::permissive(cut_call_decode_map()));
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(cut_request(true))).await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_chunks(&body);
    let content: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(content, "Writing it.\n");
    let with_calls: Vec<usize> = (0..chunks.len()).filter(|&i| chunks[i]["choices"][0]["delta"].get("tool_calls").is_some()).collect();
    let calls: Vec<&serde_json::Value> = with_calls.iter().flat_map(|&i| chunks[i]["choices"][0]["delta"]["tool_calls"].as_array().unwrap()).collect();
    assert_eq!(calls.len(), 2, "{body}");
    assert_eq!((&calls[0]["index"], &calls[0]["function"]["name"]), (&serde_json::json!(0), &serde_json::json!("read_file")));
    let cut = calls[1];
    assert_eq!((&cut["index"], &cut["function"]["name"], &cut["function"]["arguments"]), (&serde_json::json!(1), &serde_json::json!("write_file"), &serde_json::json!(CUT_ARGUMENTS)));
    assert!(cut["id"].as_str().is_some_and(|id| !id.is_empty()), "{body}");
    // The cut call's delta, then the finish chunk, and nothing between.
    let finish = chunks.iter().position(|c| c["choices"][0]["finish_reason"].is_string()).unwrap();
    assert_eq!((with_calls[1] + 1, chunks[finish]["choices"][0]["finish_reason"].as_str()), (finish, Some("length")), "{body}");
    assert!(!body.contains("<tool_call") && !body.contains("<parameter"), "raw XML leaked: {body}");
}

// ── schema-typed arguments ───────────────────────────────────────────────

/// A `write_file` call whose content reads as JSON: with the request's
/// `tools` declaring `content` a string, the argument is that text on both
/// paths — not the number the schema-free rule would make of it.
fn json_looking_content_decode_map(id: u64) -> HashMap<TokenId, &'static str> {
    let mock = MockCompute::new();
    let mut map = HashMap::new();
    map.insert(
        mock.token_for(id, 0),
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.json\n</parameter>\n<parameter=content>\n123\n</parameter>\n<parameter=line>\n7\n</function>\n</tool_call>",
    );
    map
}

fn write_file_tools() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "line": {"type": "integer"}
                }
            }
        }
    }])
}

#[tokio::test]
async fn non_streaming_types_arguments_by_the_request_tools() {
    let h = harness_with(RecordingTemplateProvider::permissive(json_looking_content_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": write_file_tools(),
        "max_tokens": 1,
        "enable_thinking": false
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().expect("tool_calls present");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.json", "content": "123", "line": 7}));
}

#[tokio::test]
async fn streaming_types_arguments_by_the_request_tools() {
    let h = harness_with(RecordingTemplateProvider::permissive(json_looking_content_decode_map(0)));
    let req = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "tools": write_file_tools(),
        "max_tokens": 1,
        "enable_thinking": false,
        "stream": true
    });
    let (status, body) = call(&h.app, "POST", "/v1/chat/completions", Some(req)).await;
    assert_eq!(status, 200, "{body}");
    let deltas: Vec<serde_json::Value> = sse_chunks(&body)
        .iter()
        .filter_map(|c| c["choices"][0]["delta"].get("tool_calls").and_then(|v| v.as_array()).cloned())
        .flatten()
        .collect();
    assert_eq!(deltas.len(), 1);
    let args: serde_json::Value =
        serde_json::from_str(deltas[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, serde_json::json!({"path": "a.json", "content": "123", "line": 7}));
}
