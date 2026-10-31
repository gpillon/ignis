//! The media cache on the request log (GitHub #179): the same image sent
//! twice is a cache hit the second time, visible on each request's
//! `ignis.request.admitted` event — over the real router against a
//! mock-compute engine (CPU-only, ADR 0006).
//!
//! And a readout-only decision over an image (GitHub #245): it never takes
//! a decode lane, so it has no `admitted` event, and its media reach the
//! `ignis.request.done` event every request ends with instead.
//!
//! Its own binary, like `telemetry.rs`: tracing caches a callsite's interest
//! process-wide, so a test running beside it without a subscriber can switch
//! the request events off before this one captures them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use ignis_artifact::vision::{PreparedMedia, ProcessorError, ProcessorOptions, VisionProcessor, IMAGE_PAD_ID};
use ignis_core::decision::{AnswerAlphabet, LabelTokenizer};
use ignis_core::mock::MockCompute;
use ignis_core::vision::Multimodal;
use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome,
    RequestId, SchedulerConfig, TokenId,
};
use ignis_logging::MemorySink;
use ignis_server::decoder::TokenDecoder;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy, Preparer};
use ignis_server::template::{
    ChatMessage, ContentRejection, RenderedPrompt, SimpleTemplateProvider, TemplateProvider, TemplateRejection,
};
use ignis_server::thinking::{ThinkingCapabilities, ThinkingOptions};
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor};

const MODEL: &str = "test-model";

fn limits() -> ProcessorOptions {
    ProcessorOptions {
        min_pixels: 32 * 32,
        max_pixels: 1 << 20,
        max_encoded_media_bytes: 1 << 20,
        max_decoded_pixels: 1 << 20,
        max_raw_patches: 1 << 16,
        max_vision_tokens: 1 << 14,
    }
}

/// The real processor, counting its builds.
struct Counting {
    inner: VisionProcessor,
    builds: AtomicUsize,
}

impl Preparer for Counting {
    fn prepare_media(&self, item: usize, bytes: &[u8], _: &dyn Fn() -> bool) -> Result<PreparedMedia, ProcessorError> {
        self.builds.fetch_add(1, Ordering::SeqCst);
        self.inner.prepare_media(item, bytes)
    }
}

/// The deterministic mock, reporting `micros` of encode time on every chunk
/// of a multimodal prompt that covers an image placeholder (GitHub #192) —
/// what the CUDA leaf's real encode reports on the GPU, without one; a chunk
/// past the image encodes nothing there either. The rest of each outcome is
/// the mock's own, so a decision's readout still reaches its caller (GitHub
/// #245).
struct EncodingCompute {
    inner: MockCompute,
    micros: u64,
}

impl Compute for EncodingCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        let mut outcomes = self.inner.prefill_step(jobs)?;
        for (outcome, job) in outcomes.iter_mut().zip(jobs) {
            let covers_image = job.multimodal.is_some() && job.tokens.contains(&IMAGE_PAD_ID);
            outcome.encode_micros = if covers_image { self.micros } else { 0 };
        }
        Ok(outcomes)
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }

    // Everything else is the mock's, which overrides these too: a partial
    // delegation would answer from the trait's defaults instead.
    fn release(&self, request: RequestId) {
        self.inner.release(request);
    }

    fn release_prefix(&self, publisher: RequestId, tokens: u32) {
        self.inner.release_prefix(publisher, tokens);
    }

    fn snapshot_size(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.snapshot_size(request)
    }

    fn evict(&self, request: RequestId) -> Result<u64, ComputeError> {
        self.inner.evict(request)
    }

    fn restore(&self, request: RequestId, context_tokens: u32) -> Result<(), ComputeError> {
        self.inner.restore(request, context_tokens)
    }

    fn discard_snapshot(&self, request: RequestId) {
        self.inner.discard_snapshot(request);
    }
}

fn counting() -> Arc<Counting> {
    Arc::new(Counting { inner: processor(limits()), builds: AtomicUsize::new(0) })
}

/// The router over `compute`, acquiring media through `preparer`.
fn app(compute: Arc<dyn Compute>, preparer: Arc<dyn Preparer>) -> axum::Router {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        compute,
    );
    let acquirer = MediaAcquirer::new(preparer, limits(), MediaPolicy::new(false, 1 << 20));
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app()
}

/// The chat request an image test sends.
fn image_request(image: &str) -> Request<Body> {
    let body = json!({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": image}},
    ]}]});
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Every event `sink` captured, once it holds `n` named `name`. The request
/// lines are written by the telemetry consumer after the response the test
/// read, so a test polls for every line it asserts on, for 5 s at most
/// (CODING_STANDARDS.md, "A test waits for the whole end state").
async fn logged(sink: &MemorySink, name: &str, n: usize) -> Vec<Value> {
    let events = || -> Vec<Value> {
        sink.lines().iter().map(|line| serde_json::from_str::<Value>(line).unwrap()).collect()
    };
    let count = || events().iter().filter(|e| e["event_name"] == name).count();
    let settled = tokio::time::timeout(Duration::from_secs(5), async {
        while count() < n {
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(settled.is_ok(), "{} {name} lines, wanted {n}", count());
    events()
}

/// The `attributes` of every `ignis.request.admitted` event among `events`,
/// in order.
fn admitted_attributes(events: &[Value]) -> Vec<Value> {
    events
        .iter()
        .filter(|e| e["event_name"] == "ignis.request.admitted")
        .map(|e| e["attributes"].clone())
        .collect()
}

#[tokio::test]
async fn the_encode_time_of_a_multimodal_prefill_reaches_the_admitted_event() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let compute = Arc::new(EncodingCompute { inner: MockCompute::new(), micros: 250_000 });
    let app = app(compute, counting());

    let response = app.clone().oneshot(image_request(&data_uri(&png(64, 64)))).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let _ = to_bytes(response.into_body(), usize::MAX).await;
    // A text request on the same load reports no media at all.
    let text = json!({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": "hello"}]});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(text.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let _ = to_bytes(response.into_body(), usize::MAX).await;

    let admitted = admitted_attributes(&logged(&sink, "ignis.request.admitted", 2).await);
    assert_eq!(admitted.len(), 2, "{admitted:?}");
    // The prompt's 4 placeholders span one chunk, so one encode is reported.
    assert_eq!(admitted[0]["media.encode_seconds"], 0.25, "{:?}", admitted[0]);
    let text = admitted[1].as_object().expect("attributes are an object");
    assert!(text.keys().all(|k| !k.starts_with("media.")), "{text:?}");
}

#[tokio::test]
async fn an_image_over_the_vision_budget_is_admitted_shrunk_and_the_log_shows_both_sizes() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    // GitHub #248: 256 x 256 is 64 merged tokens, over a budget of 16.
    let budgeted = ProcessorOptions { max_vision_tokens: 16, ..limits() };
    let preparer = Arc::new(Counting { inner: processor(budgeted.clone()), builds: AtomicUsize::new(0) });
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let acquirer = MediaAcquirer::new(preparer, budgeted, MediaPolicy::new(false, 1 << 20));
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app();

    let response = app.oneshot(image_request(&data_uri(&png(256, 256)))).await.unwrap();
    assert_eq!(response.status().as_u16(), 200, "an image over the budget is shrunk, not refused");
    let _ = to_bytes(response.into_body(), usize::MAX).await;

    let admitted = admitted_attributes(&logged(&sink, "ignis.request.admitted", 1).await);
    assert_eq!(admitted.len(), 1, "{admitted:?}");
    assert_eq!(admitted[0]["media.vision_tokens"], 16, "{:?}", admitted[0]);
    assert_eq!(admitted[0]["media.natural_vision_tokens"], 64, "{:?}", admitted[0]);
}

#[tokio::test]
async fn the_same_image_sent_twice_is_a_cache_hit_on_the_admitted_event() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let preparer = counting();
    let app = app(Arc::new(MockCompute::new()), preparer.clone());
    let image = data_uri(&png(64, 64));
    for _ in 0..2 {
        let response = app.clone().oneshot(image_request(&image)).await.unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let _ = to_bytes(response.into_body(), usize::MAX).await;
    }

    let admitted = admitted_attributes(&logged(&sink, "ignis.request.admitted", 2).await);
    assert_eq!(admitted.len(), 2, "{admitted:?}");
    let cache = |a: &Value| (a["media.cache_hits"].clone(), a["media.cache_misses"].clone());
    assert_eq!(cache(&admitted[0]), (json!(0), json!(1)), "{:?}", admitted[0]);
    assert_eq!(cache(&admitted[1]), (json!(1), json!(0)), "{:?}", admitted[1]);
    assert_eq!(admitted[1]["media.items"], 1);
    assert_eq!(admitted[1]["media.vision_tokens"], 4);
    assert_eq!(admitted[1]["media.bytes"], png(64, 64).len());
    assert_eq!(preparer.builds.load(Ordering::SeqCst), 1);
}

// ── a decision over an image (GitHub #245) ──────────────────────────────

/// Just enough tokenizer to name a `noul`'s answers: `/v1/decide` refuses a
/// load whose tokenizer can label none.
struct ThreeLetters;

impl LabelTokenizer for ThreeLetters {
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

/// [`SimpleTemplateProvider`] that can decide over an image: an answer
/// alphabet for the readout, the literals a `point`'s constrained decode
/// forces (one id per character, as `decide_http.rs`'s fixture does), and
/// the placeholder's multimodal render.
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
        AnswerAlphabet::from_tokenizer(&ThreeLetters)
    }

    fn prepare_multimodal(
        &self,
        messages: &[ChatMessage],
        options: &ThinkingOptions,
        tools: &[Value],
        media: Vec<PreparedMedia>,
    ) -> Result<(RenderedPrompt, Multimodal), ContentRejection> {
        SimpleTemplateProvider.prepare_multimodal(messages, options, tools, media)
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

/// A decide-capable router of its own, encoding every multimodal chunk in
/// 250 ms: one per test request, so no request claims another's prefix.
fn deciding_app() -> axum::Router {
    let compute = Arc::new(EncodingCompute { inner: MockCompute::new(), micros: 250_000 });
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        compute,
    );
    let acquirer = MediaAcquirer::new(counting(), limits(), MediaPolicy::new(false, 1 << 20));
    Server::new(Engine::new(Box::new(scheduler)), Box::new(DecidingTemplate))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app()
}

/// One question of `kind` over `image` (or over a text `state` when `None`),
/// answered 200: every event the request log captured, once it holds the
/// question's `done` line. Its `admitted` line, when it has one, comes
/// before: the consumer writes a request's lines in the order they happen.
async fn decide_one(kind: &str, image: Option<&str>) -> Vec<Value> {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let state = match image {
        Some(uri) => json!([{"type": "image_url", "image_url": {"url": uri}}]),
        None => json!("The queue is backing up."),
    };
    let body = json!({"state": state, "questions": {"q": {"type": kind, "instructions": "the button"}}});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/decide")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = deciding_app().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&bytes));
    let answer: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(answer["answers"]["q"]["type"], kind, "{answer}");
    logged(&sink, "ignis.request.done", 1).await
}

/// The `media.*` attributes of the one `name` event among `events`.
fn media_on(events: &[Value], name: &str) -> serde_json::Map<String, Value> {
    let found: Vec<_> = events.iter().filter(|e| e["event_name"] == name).collect();
    assert_eq!(found.len(), 1, "one {name}: {events:?}");
    found[0]["attributes"]
        .as_object()
        .expect("attributes are an object")
        .iter()
        .filter(|(key, _)| key.starts_with("media."))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

#[tokio::test]
async fn a_noul_over_an_image_reports_its_media_on_done() {
    let image = data_uri(&png(64, 64));
    let events = decide_one("noul", Some(&image)).await;

    assert!(
        events.iter().all(|e| e["event_name"] != "ignis.request.admitted"),
        "a readout takes no decode lane (GitHub #238): {events:?}"
    );
    let media = media_on(&events, "ignis.request.done");
    assert_eq!(media.get("media.encode_seconds"), Some(&json!(0.25)), "{events:?}");
    assert_eq!(media.get("media.vision_tokens"), Some(&json!(4)), "{media:?}");
}

#[tokio::test]
async fn a_noul_reports_the_same_media_a_point_over_the_same_image_does() {
    let image = data_uri(&png(64, 64));
    let noul = media_on(&decide_one("noul", Some(&image)).await, "ignis.request.done");
    let point = decide_one("point", Some(&image)).await;

    // A `point` on a load with no calibrated head is a constrained decode,
    // which takes a lane and so has always reported on `admitted`.
    let admitted = media_on(&point, "ignis.request.admitted");
    assert_eq!(admitted.len(), 8, "every media field: {admitted:?}");
    // `preprocess_seconds` is each request's own wall clock, so it is
    // compared for being there; everything else is the same number.
    let measured = |media: &serde_json::Map<String, Value>| {
        assert!(media["media.preprocess_seconds"].as_f64().is_some(), "{media:?}");
        let mut media = media.clone();
        media.remove("media.preprocess_seconds");
        media
    };
    assert_eq!(measured(&noul), measured(&admitted), "the same image, the same cost, whatever the primitive");
    assert_eq!(media_on(&point, "ignis.request.done"), admitted);
}

#[tokio::test]
async fn a_text_decision_reports_no_media_on_done() {
    let events = decide_one("noul", None).await;
    let media = media_on(&events, "ignis.request.done");
    assert!(media.is_empty(), "{media:?}");
}
