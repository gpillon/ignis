//! The reasoning redirect over HTTP (GitHub #315, ADR 0048): a thinking turn
//! whose EOS falls inside its reasoning block answers after a `</think>` the
//! leaf drew in its place, on chat completions (streamed and not) and
//! `/v1/responses` -- over the real router against a mock-compute engine
//! (CPU-only, ADR 0006). The request log says where it happened.
//!
//! Its own binary, like `thinking_budget_request_log.rs`: tracing caches a
//! callsite's interest process-wide, so a test running beside it without a
//! subscriber can switch the events off before this one captures them.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_core::mock::MockCompute;
use ignis_core::thinking_budget::ThinkingClose;
use ignis_core::{ConcreteScheduler, SchedulerConfig, TokenId};
use ignis_logging::MemorySink;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";
const THINK_END: TokenId = 999;
/// The mock's EOS steps: the first inside the reasoning, the second in the
/// answer after the close it became.
const EOS_IN_REASONING: u32 = 3;
const EOS_IN_ANSWER: u32 = 8;

/// The placeholder's prompts, a thinking-aware start (the trait default:
/// a generation with thinking on starts inside the block), and a decoder
/// that writes the close as `</think>` and every other id as `?`.
struct ThinkingTemplate;

impl TemplateProvider for ThinkingTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        ignis_server::template::SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn render_tokens(&self, tokens: &[TokenId]) -> String {
        tokens.iter().map(|&t| decode(t)).collect()
    }

    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }

    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder)
    }
}

fn decode(token: TokenId) -> &'static str {
    match token {
        THINK_END => "</think>",
        _ => "?",
    }
}

struct Decoder;

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        decode(token).to_string()
    }
    fn finish(&mut self) -> String {
        String::new()
    }
}

/// A server whose requests draw their EOS at steps 3 and 8, with a close
/// configured -- or none, the server before the redirect could close a block
/// it was not told how to.
fn app(close: bool) -> axum::Router {
    let compute = Arc::new(MockCompute::new());
    for id in 0..16 {
        compute.eos_after(id, EOS_IN_REASONING);
        compute.eos_after(id, EOS_IN_ANSWER);
    }
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            thinking_close: close
                .then(|| Arc::new(ThinkingClose::new(vec![900, THINK_END, 902], THINK_END).expect("close"))),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(ThinkingTemplate))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
        .with_thinking_defaults(true, None)
        .app()
}

async fn post(app: &axum::Router, path: &str, body: Value) -> String {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap()
}

fn chat(extra: Value) -> Value {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 32
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    body
}

fn capture() -> (Arc<MemorySink>, tracing::subscriber::DefaultGuard) {
    let sink = Arc::new(MemorySink::new());
    let guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    (sink, guard)
}

fn records(sink: &MemorySink) -> Vec<Value> {
    sink.lines().iter().map(|line| serde_json::from_str::<Value>(line).unwrap()).collect()
}

/// The `attributes` of the `ignis.request.done` events, once `n` have landed
/// (the telemetry consumer runs off the model thread).
async fn done_lines(sink: &MemorySink, n: usize) -> Vec<Value> {
    let lines = || -> Vec<Value> {
        records(sink)
            .into_iter()
            .filter(|e| e["event_name"] == "ignis.request.done")
            .map(|e| e["attributes"].clone())
            .collect()
    };
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        while lines().len() < n {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "{} done lines, wanted {n}", lines().len());
    lines()
}

/// The all-reasoning WARN (spec server/13), however many times it fired.
fn all_reasoning_warnings(sink: &MemorySink) -> usize {
    records(sink)
        .iter()
        .filter(|e| e["severity_text"] == "WARN")
        .filter(|e| e["body"].as_str().unwrap_or_default().contains("reasoning but no content"))
        .count()
}

/// Chat, not streamed: the reasoning, then the answer after the close the
/// EOS became, ending on the answer's own EOS.
#[tokio::test]
async fn a_chat_turn_whose_eos_falls_in_its_reasoning_answers_after_the_close() {
    let (sink, _guard) = capture();
    let body = post(&app(true), "/v1/chat/completions", chat(json!({}))).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["reasoning_content"], "???", "{body}");
    assert_eq!(choice["message"]["content"], "????", "{body}");
    assert_eq!(choice["finish_reason"], "stop", "{body}");
    let line = &done_lines(&sink, 1).await[0];
    assert_eq!(line["reasoning_redirected_at"], EOS_IN_REASONING, "{line}");
    assert_eq!(all_reasoning_warnings(&sink), 0);
}

/// Chat, streamed: the same channels, delta by delta.
#[tokio::test]
async fn a_streamed_chat_turn_answers_after_the_close() {
    let (sink, _guard) = capture();
    let body = post(&app(true), "/v1/chat/completions", chat(json!({ "stream": true }))).await;
    let chunks: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:").map(str::trim))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let joined = |field: &str| -> String {
        chunks.iter().filter_map(|c| c["choices"][0]["delta"][field].as_str()).collect()
    };
    assert_eq!(joined("reasoning_content"), "???", "{body}");
    assert_eq!(joined("content"), "????", "{body}");
    let finish: Vec<&Value> = chunks.iter().map(|c| &c["choices"][0]["finish_reason"]).filter(|f| !f.is_null()).collect();
    assert_eq!(finish, [&json!("stop")], "{body}");
    assert_eq!(done_lines(&sink, 1).await[0]["reasoning_redirected_at"], EOS_IN_REASONING);
    assert_eq!(all_reasoning_warnings(&sink), 0);
}

/// `/v1/responses`: a reasoning item, then a message, and the response
/// completes.
#[tokio::test]
async fn a_responses_turn_answers_after_the_close() {
    let (sink, _guard) = capture();
    let body = post(&app(true), "/v1/responses", json!({ "model": MODEL, "input": "hi", "max_output_tokens": 32 })).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "completed", "{body}");
    let output = v["output"].as_array().unwrap();
    let types: Vec<&str> = output.iter().map(|item| item["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["reasoning", "message"], "{body}");
    assert_eq!(output[0]["content"][0]["text"], "???", "{body}");
    assert_eq!(output[1]["content"][0]["text"], "????", "{body}");
    assert_eq!(done_lines(&sink, 1).await[0]["reasoning_redirected_at"], EOS_IN_REASONING);
    assert_eq!(all_reasoning_warnings(&sink), 0);

    // Streamed, the same items arrive and the response completes.
    let body = post(
        &app(true),
        "/v1/responses",
        json!({ "model": MODEL, "input": "hi", "max_output_tokens": 32, "stream": true }),
    )
    .await;
    let completed = body
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|data| serde_json::from_str::<Value>(data.trim()).unwrap())
        .find(|event| event["type"] == "response.completed")
        .expect("the response completes");
    let completed = &completed["response"];
    assert_eq!(completed["status"], "completed", "{body}");
    assert_eq!(completed["output"][1]["content"][0]["text"], "????", "{body}");
}

/// The same EOS with thinking off ends the turn, as it always has, and the
/// request log says nothing about a redirect.
#[tokio::test]
async fn a_thinking_off_turn_ends_on_the_same_eos() {
    let (sink, _guard) = capture();
    let body = post(&app(true), "/v1/chat/completions", chat(json!({ "enable_thinking": false }))).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["content"], "???", "{body}");
    assert!(choice["message"]["reasoning_content"].is_null(), "{body}");
    assert_eq!(choice["finish_reason"], "stop", "{body}");
    let line = &done_lines(&sink, 1).await[0];
    assert!(line.get("reasoning_redirected_at").is_none(), "{line}");
}

/// The control: a server with no close configured has nothing to redirect
/// to, so the same turn ends inside its reasoning -- all reasoning, the
/// WARN, and no field on the request log.
#[tokio::test]
async fn without_a_close_the_turn_ends_in_its_reasoning() {
    let (sink, _guard) = capture();
    let body = post(&app(false), "/v1/chat/completions", chat(json!({}))).await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["reasoning_content"], "???", "{body}");
    assert_eq!(choice["finish_reason"], "stop", "{body}");
    let line = &done_lines(&sink, 1).await[0];
    assert!(line.get("reasoning_redirected_at").is_none(), "{line}");
    assert_eq!(all_reasoning_warnings(&sink), 1);
}
