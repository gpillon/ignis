//! A decision on the request log (GitHub #241, spec 05): one
//! `ignis.decision.done` line per served `POST /v1/decide`, carrying the
//! question count and the primitive types.
//!
//! Why the endpoint emits one at all. A decision's N questions are N
//! internal requests, and each earns its own `ignis.request.admitted` /
//! `done` pair from the telemetry consumer — but twenty of those is not a
//! reading of "one decision of twenty questions", and the fan-out is what
//! the caller asked for. Nothing else in the log ties them back together.
//!
//! Its own binary, like `media_request_log.rs`: tracing caches a callsite's
//! interest process-wide, so a test running beside this one without a
//! subscriber can switch the events off before it captures them.

use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::http::Request;
use serde_json::Value;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::types::TokenId;
use ignis_core::{ConcreteScheduler, SchedulerConfig};
use ignis_logging::MemorySink;
use ignis_server::Server;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{
    ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};

const MODEL: &str = "test-model";

/// Just enough tokenizer to name three options: `/v1/decide` refuses a load
/// whose tokenizer can label none, and nothing here asks for a fourth.
struct Letters;

impl LabelTokenizer for Letters {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        match text {
            "A" => Some(vec![1]),
            "B" => Some(vec![2]),
            "C" => Some(vec![3]),
            _ => None,
        }
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        match ids {
            [1] => Some("A".to_owned()),
            [2] => Some("B".to_owned()),
            [3] => Some("C".to_owned()),
            _ => None,
        }
    }
}

/// [`SimpleTemplateProvider`] with an answer alphabet, which the real
/// provider gets from the loaded artifact's tokenizer.
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
        AnswerAlphabet::from_tokenizer(&Letters)
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

fn app() -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate)).app()
}

async fn decide(app: &axum::Router, body: &str) -> (u16, Value) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned().into_bytes()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

/// The `attributes` of every `ignis.decision.done` event captured, in order.
fn decisions(sink: &MemorySink) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == "ignis.decision.done")
        .map(|event| event["attributes"].clone())
        .collect()
}

const THREE_KINDS: &str = r#"{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "test-model",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Urgent?" },
    "department": { "type": "choice", "instructions": "Which team?", "criteria": { "billing": null, "technical": null } },
    "frustration": { "type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Angry"] }
  }
}"#;

#[tokio::test]
async fn a_served_decision_is_one_line_with_its_question_count_and_types() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );

    let (status, response) = decide(&app(), THREE_KINDS).await;
    assert_eq!(status, 200, "{response}");

    let logged = decisions(&sink);
    assert_eq!(logged.len(), 1, "one line per decision, not per question: {logged:?}");
    let line = &logged[0];
    assert_eq!(line["questions"], 3);
    assert_eq!(line["answered"], 3);
    assert_eq!(line["errors"], 0);
    // Declared order, deduplicated — one of seven strings, never unbounded.
    assert_eq!(line["types"], "noul,choice,score");
    // The same spelling the `ignis.request.*` events use, and a decision
    // defaults to `Agent` where every other route defaults to `Interactive`.
    assert_eq!(line["class"], "agent");
    // A readout generates nothing, and the line says so rather than leaving
    // a reader to infer it from a missing field.
    assert_eq!(line["output_tokens"], 0);
    assert!(
        line["input_tokens"].as_u64().is_some_and(|tokens| tokens > 0),
        "the prompts it paid for: {line}"
    );
    assert!(line["duration_ms"].as_u64().is_some(), "{line}");
}

#[tokio::test]
async fn the_same_primitive_twice_is_named_once() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let body = r#"{
      "state": "s",
      "questions": {
        "a": { "type": "noul", "instructions": "Urgent?" },
        "b": { "type": "noul", "instructions": "Angry?" },
        "c": { "type": "noul", "instructions": "Paying?" }
      }
    }"#;

    let (status, response) = decide(&app(), body).await;
    assert_eq!(status, 200, "{response}");

    let logged = decisions(&sink);
    assert_eq!(logged.len(), 1);
    assert_eq!(logged[0]["questions"], 3, "three questions");
    assert_eq!(logged[0]["types"], "noul", "of one primitive");
}

#[tokio::test]
async fn the_callers_lane_tag_reaches_the_line() {
    // A decision defaults to `Agent`, and a caller who says otherwise is
    // believed (GitHub #120). The log has to say which, or a reader cannot
    // tell a fan-out of subagent work from a person waiting on it.
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let body = r#"{"state":"s","model":"test-model@interactive",
                   "questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;

    let (status, response) = decide(&app(), body).await;
    assert_eq!(status, 200, "{response}");
    let logged = decisions(&sink);
    assert_eq!(logged.len(), 1);
    assert_eq!(logged[0]["class"], "interactive");
}

#[tokio::test]
async fn a_refused_decision_is_not_logged_as_a_served_one() {
    // The line means "a decision was evaluated". A request refused before
    // the first submit cost no prefill and produced no answer, and logging
    // it here would put a zero-question, zero-token decision in the same
    // series as the real ones. Its 422 is the record of it.
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let body = r#"{"state":"s","enable_thinking":true,"questions":{"q":{"type":"noul","instructions":"Urgent?"}}}"#;

    let (status, response) = decide(&app(), body).await;
    assert_eq!(status, 422, "{response}");
    assert_eq!(response["error"]["code"], "thinking_unsupported");
    assert!(decisions(&sink).is_empty(), "nothing was evaluated");
}
