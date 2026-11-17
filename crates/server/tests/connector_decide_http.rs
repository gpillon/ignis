//! The connector's `decide` (spec #656, ticket #659), driven as a harness
//! drives it: `tools/call` over MCP's Streamable HTTP, `POST /v1/mcp`, its
//! body written as raw text, on a mock engine whose template can name
//! answers (CPU only, ADR 0006).
//!
//! What a harness sees: the tool listed with `state` and `questions` as its
//! only input, a decision that is the route's own, read over the keys in
//! the order the harness wrote them, a `model` or a thinking field refused
//! before anything runs, and on Flash-Next the server's own
//! `model_unsupported` as the tool's error. The bodies here are text, never
//! a `serde_json::Value`, whose object sorts its keys in this build.

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelFamily;
use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::types::TokenId;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_server::config::ApiKey;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "test-model";
const KEY: &str = "sk-connector-decide-test";

/// The protocol version the harness negotiated, named on every request
/// after `initialize`.
const PROTOCOL: &str = "2025-06-18";

/// Just enough tokenizer to name two options: `/v1/decide` refuses a load
/// whose tokenizer can label none.
struct TwoLetters;

impl LabelTokenizer for TwoLetters {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        match text {
            "A" => Some(vec![1]),
            "B" => Some(vec![2]),
            _ => None,
        }
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        match ids {
            [1] => Some("A".to_owned()),
            [2] => Some("B".to_owned()),
            _ => None,
        }
    }
}

/// [`SimpleTemplateProvider`], one token per word, with a two-label answer
/// alphabet: a word written in another order is another token.
struct DecidingTemplate;

impl TemplateProvider for DecidingTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn answer_alphabet(&self) -> AnswerAlphabet {
        AnswerAlphabet::from_tokenizer(&TwoLetters)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        SimpleTemplateProvider.render_tokens(tokens)
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        SimpleTemplateProvider.thinking_capabilities()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        SimpleTemplateProvider.token_decoder()
    }
}

/// A fresh server of `family` on the deterministic mock, its connector on
/// by default, and the mock, whose prefills say what the model read.
fn server(family: ModelFamily) -> (axum::Router, Arc<MockCompute>) {
    server_keyed(family, None)
}

/// [`server`], started with `key` when there is one.
fn server_keyed(family: ModelFamily, key: Option<&str>) -> (axum::Router, Arc<MockCompute>) {
    let compute = Arc::new(MockCompute::new());
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::clone(&compute) as Arc<dyn ignis_core::Compute>,
    );
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate)).with_family(family);
    let server = match key {
        Some(key) => server.with_api_key(ApiKey::new(key)),
        None => server,
    };
    (server.app(), compute)
}

/// Every prompt the mock prefilled, in order.
fn prefilled(compute: &MockCompute) -> Vec<Vec<TokenId>> {
    compute.prefill_calls().into_iter().flatten().map(|job| job.tokens).collect()
}

async fn exchange(app: &axum::Router, request: Request<Body>) -> (StatusCode, Bytes) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    (status, response.into_body().collect().await.unwrap().to_bytes())
}

/// One JSON-RPC request to `/v1/mcp`, `params` written as given: its
/// `result`.
async fn rpc(app: &axum::Router, method: &str, params: &str) -> Value {
    rpc_keyed(app, method, params, None).await
}

/// [`rpc`], sent with `key` when there is one.
async fn rpc_keyed(app: &axum::Router, method: &str, params: &str, key: Option<&str>) -> Value {
    let body = format!(r#"{{"jsonrpc":"2.0","id":1,"method":"{method}","params":{params}}}"#);
    let mut request = Request::builder()
        .method(Method::POST)
        .uri("/v1/mcp")
        .header(header::HOST, "ignis.example:8000")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if method != "initialize" {
        request = request.header("mcp-protocol-version", PROTOCOL);
    }
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let (status, bytes) = exchange(app, request.body(Body::from(body)).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{method}: {}", String::from_utf8_lossy(&bytes));
    let answer: Value = serde_json::from_slice(&bytes).unwrap();
    answer.get("result").cloned().unwrap_or_else(|| panic!("{method}: no result: {answer}"))
}

/// `decide` called with `arguments`, written as given.
async fn decide(app: &axum::Router, arguments: &str) -> Value {
    decide_keyed(app, arguments, None).await
}

/// [`decide`], sent with `key` when there is one.
async fn decide_keyed(app: &axum::Router, arguments: &str, key: Option<&str>) -> Value {
    rpc_keyed(app, "tools/call", &format!(r#"{{"name":"decide","arguments":{arguments}}}"#), key).await
}

/// `POST /v1/decide` with `body`, written as given: the route's status and
/// body, as a client reads them.
async fn direct(app: &axum::Router, body: &str) -> (StatusCode, Bytes) {
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/decide")
        .header(header::CONTENT_TYPE, "application/json");
    exchange(app, request.body(Body::from(body.to_owned())).unwrap()).await
}

/// The one text a result's `content` holds.
fn text_of(result: &Value) -> String {
    let content = result["content"].as_array().expect("content");
    assert_eq!(content.len(), 1, "{result}");
    assert_eq!(content[0]["type"], "text", "{result}");
    content[0]["text"].as_str().unwrap().to_owned()
}

/// A decision whose meaning is its keys' order: a `state` record whose
/// keys are not in alphabetical order, and a `choice` whose options are
/// not, each option answered by the label at its position.
const WRITTEN: &str = r#"{"state": {"zeta": "the order record", "alpha": "a note on it"},
 "questions": {"which": {"type": "choice", "instructions": "Which field matters?",
   "criteria": {"zulu": "the record itself", "alpha": "the note"}}}}"#;

/// [`WRITTEN`]'s `state` record and its `choice`'s options, each as
/// written and with its keys sorted, as MCP's map would hand them over.
const STATE: [&str; 2] = [r#"{"zeta": "the order record", "alpha": "a note on it"}"#, r#"{"alpha": "a note on it", "zeta": "the order record"}"#];
const OPTIONS: [&str; 2] = [r#"{"zulu": "the record itself", "alpha": "the note"}"#, r#"{"alpha": "the note", "zulu": "the record itself"}"#];

// ── the tool list ────────────────────────────────────────────────────────

/// Spec #656: `decide` is listed with `state` and `questions` as its whole
/// input, both required and nothing else admitted: no `model`, so a call
/// cannot switch the server, and no thinking field. Its description says
/// in one line when to use it.
#[tokio::test]
async fn decide_is_listed_with_state_and_questions_as_its_only_input() {
    let (app, _) = server(ModelFamily::Qwen38_27b);
    let listed = rpc(&app, "tools/list", "{}").await;
    let tool = listed["tools"].as_array().unwrap().iter().find(|tool| tool["name"] == "decide").expect("decide is listed");
    assert_eq!(tool["annotations"], json!({ "readOnlyHint": false, "destructiveHint": false }));
    let input = &tool["inputSchema"];
    assert_eq!(input["type"], "object", "{input}");
    let mut properties: Vec<&String> = input["properties"].as_object().expect("properties").keys().collect();
    properties.sort();
    assert_eq!(properties, ["questions", "state"]);
    assert_eq!(input["required"], json!(["state", "questions"]));
    assert_eq!(input["additionalProperties"], false);
    let description = tool["description"].as_str().unwrap();
    assert_eq!(description.lines().count(), 1, "{description}");
    assert!(description.contains("27B"), "{description}");
}

// ── a decision ───────────────────────────────────────────────────────────

/// Spec #656: `decide` calls `POST /v1/decide`, and its answer is the
/// route's, word for word. The route reads a `state` object and a
/// `choice`'s options in the order they were written, and so does the
/// tool, though MCP hands a tool its arguments as a map sorted by key: the
/// model reads the same prompt through the tool as from the same text sent
/// to the route, and another prompt from the keys sorted.
#[tokio::test]
async fn a_decision_through_the_tool_is_the_routes_own_over_the_keys_in_the_order_written() {
    let (app, compute) = server(ModelFamily::Qwen38_27b);
    let result = decide(&app, WRITTEN).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    assert!(result.get("structuredContent").is_none(), "{result}");
    let answer = text_of(&result);
    let through_the_tool = prefilled(&compute);
    assert!(!through_the_tool.is_empty());

    // The same text, sent to the route by a client, on a fresh server.
    let (app, compute) = server(ModelFamily::Qwen38_27b);
    let (status, body) = direct(&app, WRITTEN).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(answer, String::from_utf8_lossy(&body));
    assert_eq!(prefilled(&compute), through_the_tool);

    // Each sorted alone, as MCP's map hands them over: the `state` record,
    // another prompt; the `choice`'s options, another prompt and another
    // option at each label.
    let written: Value = serde_json::from_str(&answer).unwrap();
    for (what, [as_written, sorted]) in [("the state", STATE), ("the options", OPTIONS)] {
        assert_eq!(WRITTEN.matches(as_written).count(), 1, "{what}");
        let (app, compute) = server(ModelFamily::Qwen38_27b);
        let (status, body) = direct(&app, &WRITTEN.replace(as_written, sorted)).await;
        assert_eq!(status, StatusCode::OK, "{what}: {}", String::from_utf8_lossy(&body));
        assert_ne!(prefilled(&compute), through_the_tool, "{what}: the fixture tells the orders apart");
        if what == "the options" {
            let sorted: Value = serde_json::from_slice(&body).unwrap();
            assert_ne!(sorted["answers"]["which"]["probabilities"], written["answers"]["which"]["probabilities"], "{sorted} {written}");
        }
    }
}

/// Spec #656: the call goes to the route with the harness's own key, as
/// its request would: on a server with a key, the decision is answered.
#[tokio::test]
async fn a_decision_is_asked_with_the_harnesss_key() {
    let (app, compute) = server_keyed(ModelFamily::Qwen38_27b, Some(KEY));
    let result = decide_keyed(&app, WRITTEN, Some(KEY)).await;
    assert_ne!(result["isError"], json!(true), "{result}");
    let answer: Value = serde_json::from_str(&text_of(&result)).unwrap();
    assert!(answer["answers"]["which"]["choice"].is_string(), "{answer}");
    assert!(!prefilled(&compute).is_empty());
}

/// Spec #656: the tool takes no `model`, so it cannot switch the server,
/// and no thinking field: a call carrying one is the tool's error, naming
/// it, before anything is prefilled.
#[tokio::test]
async fn a_model_or_a_thinking_field_is_refused_before_anything_runs() {
    let (app, compute) = server(ModelFamily::Qwen38_27b);
    let state_and_questions = &WRITTEN[1..WRITTEN.len() - 1];
    for (field, value) in [
        ("model", r#""other-model""#),
        ("enable_thinking", "true"),
        ("reasoning_effort", r#""high""#),
        ("preserve_thinking", "true"),
        ("chat_template_kwargs", r#"{"enable_thinking": true}"#),
    ] {
        let result = decide(&app, &format!(r#"{{{state_and_questions}, "{field}": {value}}}"#)).await;
        assert_eq!(result["isError"], true, "{field}: {result}");
        let error: Value = serde_json::from_str(&text_of(&result)).unwrap();
        assert_eq!(error["error"]["code"], "invalid_request_error", "{error}");
        assert_eq!(error["error"]["param"], field, "{error}");
    }
    assert!(prefilled(&compute).is_empty());
}

/// `/v1/mcp` keeps a request's body as written for `decide`, held to the
/// cap rmcp holds a body to, 4 MiB: past it the request is refused `413`,
/// as rmcp refused it, before anything runs.
#[tokio::test]
async fn a_body_past_the_mcp_cap_is_refused_413() {
    let (app, compute) = server(ModelFamily::Qwen38_27b);
    let state = "x".repeat(4 << 20);
    let body = format!(r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"decide","arguments":{{"state":"{state}","questions":{{}}}}}}}}"#);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/mcp")
        .header(header::HOST, "ignis.example:8000")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream")
        .header("mcp-protocol-version", PROTOCOL);
    let (status, _) = exchange(&app, request.body(Body::from(body)).unwrap()).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(prefilled(&compute).is_empty());
}

// ── Flash-Next ───────────────────────────────────────────────────────────

/// Spec #656: on Flash-Next, which has no readouts, the route's own
/// `400 model_unsupported` is the tool's error, the server's JSON as a
/// client reads it, and nothing runs.
#[tokio::test]
async fn on_flash_next_the_tool_answers_the_servers_model_unsupported() {
    let (app, compute) = server(ModelFamily::FlashNext);
    let result = decide(&app, WRITTEN).await;
    assert_eq!(result["isError"], true, "{result}");
    let error = text_of(&result);
    let (status, body) = direct(&app, WRITTEN).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(error, String::from_utf8_lossy(&body));
    let error: Value = serde_json::from_str(&error).unwrap();
    assert_eq!(error["error"]["code"], "model_unsupported", "{error}");
    assert!(prefilled(&compute).is_empty());
}

// ── instructions ─────────────────────────────────────────────────────────

/// Spec #656: `instructions` say in one line when to use `decide`, and that
/// it works on the 27B only.
#[tokio::test]
async fn the_instructions_say_when_to_use_decide_and_that_it_is_the_27bs() {
    let (app, _) = server(ModelFamily::Qwen38_27b);
    let params = format!(r#"{{"protocolVersion":"{PROTOCOL}","capabilities":{{}},"clientInfo":{{"name":"a-harness","version":"1.0"}}}}"#);
    let result = rpc(&app, "initialize", &params).await;
    let instructions = result["instructions"].as_str().expect("instructions");
    let line = instructions.lines().find(|line| line.starts_with("decide")).expect("decide's line");
    assert!(line.contains("27B"), "{line}");
    assert!(line.len() <= 120, "{} bytes: {line}", line.len());
}
