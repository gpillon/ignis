//! Structured output on the metrics and the request log (GitHub #593, spec
//! #287, ADR 0089, amending ADR 0017): a request held to a grammar
//! moves the `ignis_structured_*` series by its misses and repairs, records
//! each repair's time, and says its type and counts on its `done` line,
//! whether it ended on its own or at its caller's stop sequence -- over the
//! real router on `MockCompute` (ADR 0006), whose scripted model misses on
//! purpose.
//!
//! Its own binary, like `thinking_budget_request_log.rs`: tracing caches a
//! callsite's interest process-wide, so a test running beside it without a
//! subscriber can switch the request events off before this one captures
//! them.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value as JsonValue};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig, TokenId};
use ignis_grammar::{GrammarCache, Vocab, VocabEntry, DEFAULT_CACHE_BYTES};
use ignis_logging::MemorySink;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::template::{ChatMessage, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

const MODEL: &str = "test-model";
const CHAT: &str = "/v1/chat/completions";
/// The lowest ids close a JSON value, so a repair with nothing better to go
/// on moves the answer toward its end; `STOP` is the model's EOS.
const STOP: TokenId = 5;
const VOCAB: usize = 128;

fn text_of(token: TokenId) -> Option<String> {
    match token {
        0 => Some("\"".into()),
        1 => Some(":".into()),
        2 => Some("}".into()),
        3 => Some("]".into()),
        4 => Some("{".into()),
        10 => Some("\n".into()),
        32..=126 => char::from_u32(token).map(String::from),
        _ => None,
    }
}

fn encode(text: &str) -> Vec<TokenId> {
    text.chars().map(|c| if c == '\n' { 10 } else { c as TokenId }).collect()
}

#[derive(Clone)]
struct Vocabulary(Arc<GrammarCache>);

impl Vocabulary {
    fn new() -> Self {
        let entries = (0..VOCAB as TokenId)
            .map(|id| match (id, text_of(id)) {
                (STOP, _) => VocabEntry::Control,
                (_, Some(text)) => VocabEntry::Text(text.into_bytes()),
                (_, None) => VocabEntry::Absent,
            })
            .collect();
        Self(Arc::new(GrammarCache::new(Vocab::new(entries, &[STOP]), DEFAULT_CACHE_BYTES)))
    }
}

struct Decoder;

impl TokenDecoder for Decoder {
    fn push(&mut self, token: TokenId) -> String {
        text_of(token).unwrap_or_default()
    }
    fn finish(&mut self) -> String {
        String::new()
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
        tokens.iter().map(|&t| text_of(t).unwrap_or_default()).collect()
    }
    fn thinking_capabilities(&self) -> ThinkingCapabilities {
        ThinkingCapabilities::permissive()
    }
    fn token_decoder(&self) -> Box<dyn TokenDecoder> {
        Box::new(Decoder)
    }
    fn grammars(&self) -> Option<Arc<GrammarCache>> {
        Some(self.0.clone())
    }
}

/// A server with metrics on whose second, third and fourth requests write
/// their scripts (the first is unscripted and unconstrained).
fn apps(scripts: [&str; 3]) -> (axum::Router, axum::Router, Arc<MockCompute>) {
    let mock = Arc::new(MockCompute::new());
    for (request, script) in (1..).zip(scripts) {
        mock.script(request, &encode(script));
    }
    let scheduler =
        ConcreteScheduler::with_config(SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() }, mock.clone());
    let server = Server::new(Engine::new(Box::new(scheduler)), Box::new(Vocabulary::new()))
        .with_request_timeout(Duration::from_secs(5))
        .with_seedless_seed(0)
        .with_metrics();
    let metrics = server.metrics_app().expect("metrics are on");
    (server.app(), metrics, mock)
}

async fn chat(app: &axum::Router, structured: bool, stop: Option<&str>) {
    let mut body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": "describe a cat" }],
        "enable_thinking": false,
        "temperature": 0,
        "max_tokens": 64,
    });
    if structured {
        body["response_format"] = json!({ "type": "json_object" });
    }
    if let Some(stop) = stop {
        body["stop"] = json!([stop]);
    }
    let request = Request::builder()
        .method("POST")
        .uri(CHAT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let _ = to_bytes(response.into_body(), usize::MAX).await;
}

async fn scrape(metrics: &axum::Router) -> String {
    let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap()
}

/// The value of `name` with exactly the label text `labels` (empty for none).
fn value(text: &str, name: &str, labels: &str) -> Option<f64> {
    let wanted = if labels.is_empty() { name.to_owned() } else { format!("{name}{{{labels}}}") };
    text.lines().find_map(|l| l.strip_prefix(&wanted)?.strip_prefix(' ')?.parse().ok())
}

/// Scrape until `done`: the telemetry consumer runs off the model thread.
async fn scrape_until(metrics: &axum::Router, done: impl Fn(&str) -> bool) -> String {
    let mut last = String::new();
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            last = scrape(metrics).await;
            if done(&last) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "the projection never settled; last scrape:\n{last}");
    last
}

async fn done_lines(sink: &MemorySink, n: usize) -> Vec<JsonValue> {
    let lines = || -> Vec<JsonValue> {
        sink.lines()
            .iter()
            .map(|line| serde_json::from_str::<JsonValue>(line).unwrap())
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

#[tokio::test]
async fn scripted_misses_move_the_structured_series_and_the_done_line() {
    let sink = Arc::new(MemorySink::new());
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())));
    // The fourth writes an `x` where the object needs a comma or its close,
    // then the comma: two misses, before its caller's stop sequence `"b"`.
    let (api, metrics, mock) = apps(["Sure! Here it is:\n{\"a\": 1}", "{\"a\": 1}", "{\"a\": 1x, \"b\": 2}"]);
    let misses_of = |request: u64| mock.repair_calls().iter().filter(|job| job.request == request).count();
    let object = r#"type="json_object""#;
    // `ignis_structured_requests_total` is the series a request's end writes
    // last, so a scrape that shows it moved shows every other series moved.
    let structured_requests = |t: &str| value(t, "ignis_structured_requests_total", object);

    // A request with a free answer adds nothing: the family is absent.
    chat(&api, false, None).await;
    let text = scrape_until(&metrics, |t| value(t, "ignis_requests_completed_total", "") == Some(1.0)).await;
    assert!(!text.contains("ignis_structured_"), "{text}");

    // A request whose script misses: every miss repaired, each repair timed.
    chat(&api, true, None).await;
    let text = scrape_until(&metrics, |t| structured_requests(t) == Some(1.0)).await;
    let misses = misses_of(1);
    assert!(misses >= 2, "the preamble misses more than once: {misses}");
    assert_eq!(value(&text, "ignis_requests_completed_total", ""), Some(2.0), "{text}");
    assert_eq!(value(&text, "ignis_structured_requests_total", r#"type="json_schema""#), Some(0.0), "{text}");
    assert_eq!(value(&text, "ignis_structured_misses_total", object), Some(misses as f64), "{text}");
    assert_eq!(value(&text, "ignis_structured_repairs_total", object), Some(misses as f64), "{text}");
    assert_eq!(value(&text, "ignis_structured_repair_seconds_count", ""), Some(misses as f64), "{text}");
    assert_eq!(value(&text, "ignis_structured_repair_seconds_bucket", r#"le="+Inf""#), Some(misses as f64), "{text}");
    assert!(text.contains("# TYPE ignis_structured_repair_seconds histogram"), "{text}");

    // A request that never misses is counted and moves neither count.
    chat(&api, true, None).await;
    let text = scrape_until(&metrics, |t| structured_requests(t) == Some(2.0)).await;
    assert_eq!(value(&text, "ignis_structured_misses_total", object), Some(misses as f64), "{text}");
    assert_eq!(value(&text, "ignis_structured_repair_seconds_count", ""), Some(misses as f64), "{text}");

    // A request its caller's stop sequence ends completed, with its misses:
    // the engine releases it with no finish of its own (spec server/09).
    chat(&api, true, Some("\"b\"")).await;
    let text = scrape_until(&metrics, |t| structured_requests(t) == Some(3.0)).await;
    let stopped = misses_of(3);
    assert_eq!(stopped, 2, "the x and the comma after the repaired one");
    assert_eq!(value(&text, "ignis_structured_misses_total", object), Some((misses + stopped) as f64), "{text}");
    assert_eq!(value(&text, "ignis_structured_repairs_total", object), Some((misses + stopped) as f64), "{text}");

    // The log: the type and counts on the structured requests' lines only.
    let lines = done_lines(&sink, 4).await;
    assert!(lines[0].get("structured.type").is_none() && lines[0].get("structured.misses").is_none(), "{}", lines[0]);
    assert_eq!(lines[1]["structured.type"], "json_object", "{}", lines[1]);
    assert_eq!(lines[1]["structured.misses"], misses, "{}", lines[1]);
    assert_eq!(lines[1]["structured.repairs"], misses, "{}", lines[1]);
    assert_eq!(lines[2]["structured.type"], "json_object", "{}", lines[2]);
    assert_eq!(lines[2]["structured.misses"], 0, "{}", lines[2]);
    assert_eq!(lines[2]["structured.repairs"], 0, "{}", lines[2]);
    assert_eq!(lines[3]["finish_reason"], "stop", "{}", lines[3]);
    assert_eq!(lines[3]["structured.type"], "json_object", "{}", lines[3]);
    assert_eq!(lines[3]["structured.misses"], stopped, "{}", lines[3]);
    assert_eq!(lines[3]["structured.repairs"], stopped, "{}", lines[3]);
}
