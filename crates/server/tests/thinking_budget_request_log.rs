//! The thinking budget on the request log (spec server/08): the
//! `ignis.request.done` line says which budget a request ran under, whether
//! its close was forced and after how many reasoning tokens, and when
//! `reasoning_effort: "max"` dropped a budget — over the real router against
//! a mock-compute engine (CPU-only, ADR 0006).
//!
//! Its own binary, like `media_request_log.rs`: tracing caches a callsite's
//! interest process-wide, so a test running beside it without a subscriber
//! can switch the request events off before this one captures them.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_core::mock::MockCompute;
use ignis_core::thinking_budget::{ThinkingClose, ANSWER_RESERVE};
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_logging::MemorySink;
use ignis_server::engine::Engine;
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::ReasoningTemplate;

const MODEL: &str = "test-model";
const THINK_END: u32 = 999;

/// A server whose scheduler forces a four-token close once a budget is
/// spent, whose template starts every thinking generation inside the block,
/// and whose requests stop after twelve tokens.
fn app(default_budget: Option<u32>) -> axum::Router {
    let compute = Arc::new(MockCompute::new());
    for id in 0..16 {
        compute.stop_after(id, 12);
    }
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            max_sequence_tokens: 65_536,
            thinking_close: Some(Arc::new(ThinkingClose::new(vec![900, 901, THINK_END, 902], THINK_END).unwrap())),
            ..SchedulerConfig::default()
        },
        compute,
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(ReasoningTemplate))
        .with_request_timeout(Duration::from_secs(5))
        .with_thinking_budget(default_budget)
        .app()
}

/// POST one chat completion with `extra` over room for any budget here.
async fn chat(app: &axum::Router, extra: Value) {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "hello" }],
        "max_tokens": ANSWER_RESERVE + 20_000
    });
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let _ = to_bytes(response.into_body(), usize::MAX).await;
}

/// The `attributes` of the `ignis.request.done` events, once `n` have
/// landed: the telemetry consumer runs off the model thread, so a line lands
/// after the response that caused it, never before.
async fn done_lines(sink: &MemorySink, n: usize) -> Vec<Value> {
    let lines = || -> Vec<Value> {
        sink.lines()
            .iter()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
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

fn capture() -> (Arc<MemorySink>, tracing::subscriber::DefaultGuard) {
    let sink = Arc::new(MemorySink::new());
    let guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    (sink, guard)
}

fn thinking_keys(line: &Value) -> Vec<String> {
    line.as_object()
        .expect("attributes are an object")
        .keys()
        .filter(|k| k.starts_with("thinking"))
        .cloned()
        .collect()
}

#[tokio::test]
async fn a_forced_close_is_on_the_done_line_with_its_budget_and_where_it_began() {
    let (sink, _guard) = capture();
    let app = app(None);
    chat(&app, json!({ "thinking_budget": 3 })).await;
    let line = &done_lines(&sink, 1).await[0];
    assert_eq!(line["thinking_budget"], 3, "{line}");
    assert_eq!(line["thinking_forced"], true, "{line}");
    assert_eq!(line["thinking_forced_at"], 4, "{line}");
    assert!(line.get("thinking_budget_dropped").is_none(), "{line}");
}

#[tokio::test]
async fn a_budget_that_never_bit_is_on_the_done_line_as_not_forced() {
    let (sink, _guard) = capture();
    let app = app(Some(8192));
    chat(&app, json!({})).await;
    let line = &done_lines(&sink, 1).await[0];
    assert_eq!(line["thinking_budget"], 8192, "{line}");
    assert_eq!(line["thinking_forced"], false, "{line}");
    assert_eq!(thinking_keys(line), ["thinking_budget", "thinking_forced"], "{line}");
}

#[tokio::test]
async fn a_request_without_a_budget_says_nothing_about_one() {
    let (sink, _guard) = capture();
    let app = app(Some(8192));
    chat(&app, json!({ "thinking_budget": 0 })).await;
    let line = &done_lines(&sink, 1).await[0];
    assert!(thinking_keys(line).is_empty(), "{line}");
}

#[tokio::test]
async fn max_dropping_a_budget_is_on_the_done_line() {
    let (sink, _guard) = capture();
    let app = app(Some(8192));
    // The request's own budget, and the server default.
    chat(&app, json!({ "reasoning_effort": "max", "thinking_budget": 64 })).await;
    chat(&app, json!({ "reasoning_effort": "max" })).await;
    let lines = done_lines(&sink, 2).await;
    for line in &lines {
        assert_eq!(line["thinking_budget_dropped"], "max", "{line}");
        assert_eq!(thinking_keys(line), ["thinking_budget_dropped"], "{line}");
    }
}

#[tokio::test]
async fn max_with_nothing_to_drop_says_nothing() {
    let (sink, _guard) = capture();
    let app = app(None);
    chat(&app, json!({ "reasoning_effort": "max" })).await;
    let line = &done_lines(&sink, 1).await[0];
    assert!(thinking_keys(line).is_empty(), "{line}");
}
