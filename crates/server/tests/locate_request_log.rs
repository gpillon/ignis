//! A `locate` on the request log (GitHub #278, spec 22 § Observability): one
//! `ignis.decide.located` line per answered `locate`, naming the kind,
//! method and compression that produced it, what it read — windows, a
//! fold's templates, its chosen template's rows, each `choice`'s candidates
//! — and `found` with the "none" option's and the yes/no's probabilities.
//!
//! Its own binary, like `decide_request_log.rs`: tracing caches a callsite's
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

/// Every letter and digit: enough labels for a last `choice` of sixteen
/// candidates and "none".
struct Letters;

impl Letters {
    fn entries() -> Vec<String> {
        ('A'..='Z').chain('a'..='z').chain('0'..='9').map(|c| c.to_string()).collect()
    }
}

impl LabelTokenizer for Letters {
    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        Self::entries().iter().position(|entry| entry == text).map(|index| vec![index as TokenId])
    }

    fn decode(&self, ids: &[TokenId]) -> Option<String> {
        let entries = Self::entries();
        ids.iter().map(|&id| entries.get(id as usize).cloned()).collect::<Option<Vec<_>>>().map(|p| p.concat())
    }
}

/// The placeholder template with its text and word offsets — what a
/// `locate` maps its segments by — an alphabet, and a literal encoder for
/// the copy scaffold.
struct LocatingTemplate;

impl TemplateProvider for LocatingTemplate {
    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template(messages, options, tools)
    }

    fn apply_chat_template_with_text(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
    ) -> Result<RenderedPrompt, TemplateRejection> {
        SimpleTemplateProvider.apply_chat_template_with_text(messages, options, tools)
    }

    fn answer_alphabet(&self) -> AnswerAlphabet {
        AnswerAlphabet::from_tokenizer(&Letters)
    }

    fn encode_literal(&self, text: &str) -> Option<Vec<TokenId>> {
        Some(text.chars().map(|c| c as TokenId).collect())
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
    let served = ignis_core::locate::calibrated_artifacts().next().expect("a calibrated artifact");
    let compute = Arc::new(MockCompute::with_blob_identity(ignis_core::BlobIdentity {
        artifact: served,
        ..ignis_core::BlobIdentity::UNSET
    }));
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
        compute,
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(LocatingTemplate)).app()
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
    (status, serde_json::from_str(&String::from_utf8(bytes.to_vec()).unwrap()).unwrap_or(Value::Null))
}

/// The `attributes` of every `ignis.decide.located` event captured.
fn located(sink: &MemorySink) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event_name"] == "ignis.decide.located")
        .map(|event| event["attributes"].clone())
        .collect()
}

/// Seven kinds of line, twenty of each: past five templates and sixteen
/// rows, so the fold's both levels are read.
fn log() -> String {
    let verbs = ["started", "stopped", "ready", "failed", "retried", "evicted", "scaled"];
    let mut lines = Vec::new();
    for r in 0..20 {
        for (k, verb) in verbs.iter().enumerate() {
            lines.push(format!("2026-09-28T10:{:02}:{k:02}Z pod-{r}{k} {verb} node{k}", r % 60));
        }
    }
    serde_json::to_string(&lines.join("\n")).unwrap()
}

#[tokio::test]
async fn a_folded_locate_logs_what_it_read_and_what_it_found() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let body = format!(r#"{{"state":{},"questions":{{"q":{{"type":"locate","instructions":"which pod failed on node3"}}}}}}"#, log());
    let (status, response) = decide(&app(), &body).await;
    assert_eq!(status, 200, "{response}");
    let lines = located(&sink);
    assert_eq!(lines.len(), 1, "one line per locate: {lines:?}");
    let line = &lines[0];
    assert_eq!(line["question"], "q");
    assert_eq!((line["kind"].as_str(), line["method"].as_str(), line["compression"].as_str()),
        (Some("log"), Some("shortlist"), Some("template_fold")), "{line}");
    assert_eq!(line["templates"], 7, "{line}");
    assert_eq!(line["rows"], 20, "{line}");
    assert!(line["template"].as_u64().is_some_and(|t| t < 7), "{line}");
    assert_eq!(line["windows"], 2, "one window at each level: {line}");
    // Three `choice`s' candidates: five templates, sixteen rows' lines.
    let candidates: Vec<&str> = line["candidates"].as_str().expect("candidates").split(';').collect();
    assert_eq!(candidates.len(), 2, "{line}");
    assert_eq!(candidates[0].split(',').count(), 5, "{line}");
    assert_eq!(candidates[1].split(',').count(), 16, "{line}");
    let (p_none, p_yes, found) = (line["p_none"].as_f64(), line["p_yes"].as_f64(), line["found"].as_f64());
    let (p_none, p_yes, found) = (p_none.expect("p_none"), p_yes.expect("p_yes"), found.expect("found"));
    assert!((found - (1.0 - p_none + p_yes) / 2.0).abs() < 1e-9, "{line}");
    assert_eq!(response["answers"]["q"]["found"].as_f64(), Some(found));
    assert!(line["input_tokens"].as_u64().is_some_and(|tokens| tokens > 0), "{line}");
}
