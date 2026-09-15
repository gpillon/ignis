//! The media cache on the request log (GitHub #179): the same image sent
//! twice is a cache hit the second time, visible on each request's
//! `ignis.request.admitted` event — over the real router against a
//! mock-compute engine (CPU-only, ADR 0006).
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

use ignis_artifact::vision::{PreparedMedia, ProcessorError, ProcessorOptions, VisionProcessor};
use ignis_core::mock::MockCompute;
use ignis_core::{
    Compute, ComputeError, ConcreteScheduler, DecodeJob, DecodeOutcome, PrefillJob, PrefillOutcome,
    RequestId, SchedulerConfig,
};
use ignis_logging::MemorySink;
use ignis_server::engine::Engine;
use ignis_server::media::{MediaAcquirer, MediaPolicy, Preparer};
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::Server;

#[path = "support/mod.rs"]
mod support;
use support::media::{data_uri, png, processor};
use support::nudge;

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
/// of a multimodal prompt (GitHub #192) — what the CUDA leaf's real encode
/// reports on the GPU, without one.
struct EncodingCompute {
    inner: MockCompute,
    micros: u64,
}

impl Compute for EncodingCompute {
    fn prefill_step(&self, jobs: &[PrefillJob]) -> Result<Vec<PrefillOutcome>, ComputeError> {
        self.inner.prefill_step(jobs)?;
        Ok(jobs
            .iter()
            .map(|job| PrefillOutcome {
                encode_micros: if job.multimodal.is_some() { self.micros } else { 0 },
            })
            .collect())
    }

    fn decode_step(&self, jobs: &[DecodeJob]) -> Result<Vec<DecodeOutcome>, ComputeError> {
        self.inner.decode_step(jobs)
    }

    fn release(&self, request: RequestId) {
        self.inner.release(request);
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

/// The `attributes` of every `ignis.request.admitted` event captured, in order.
fn admitted_attributes(sink: &MemorySink) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
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
    nudge().await;

    let admitted = admitted_attributes(&sink);
    assert_eq!(admitted.len(), 2, "{admitted:?}");
    // The prompt's 4 placeholders span one chunk, so one encode is reported.
    assert_eq!(admitted[0]["media.encode_seconds"], 0.25, "{:?}", admitted[0]);
    let text = admitted[1].as_object().expect("attributes are an object");
    assert!(text.keys().all(|k| !k.starts_with("media.")), "{text:?}");
}

#[tokio::test]
async fn the_same_image_sent_twice_is_a_cache_hit_on_the_admitted_event() {
    let sink = Arc::new(MemorySink::new());
    let _guard = tracing::subscriber::set_default(
        tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone())),
    );
    let preparer = counting();
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    let acquirer = MediaAcquirer::new(preparer.clone(), limits(), MediaPolicy::new(false, 1 << 20));
    let app = Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
        .with_request_timeout(Duration::from_secs(10))
        .with_media(Arc::new(acquirer))
        .app();
    let image = data_uri(&png(64, 64));
    let body = json!({"model": MODEL, "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": image}},
    ]}]});
    for _ in 0..2 {
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
    nudge().await;

    let admitted: Vec<Value> = sink
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|e| e["event_name"] == "ignis.request.admitted")
        .map(|e| e["attributes"].clone())
        .collect();
    assert_eq!(admitted.len(), 2, "{admitted:?}");
    let cache = |a: &Value| (a["media.cache_hits"].clone(), a["media.cache_misses"].clone());
    assert_eq!(cache(&admitted[0]), (json!(0), json!(1)), "{:?}", admitted[0]);
    assert_eq!(cache(&admitted[1]), (json!(1), json!(0)), "{:?}", admitted[1]);
    assert_eq!(admitted[1]["media.items"], 1);
    assert_eq!(admitted[1]["media.vision_tokens"], 4);
    assert_eq!(admitted[1]["media.bytes"], png(64, 64).len());
    assert_eq!(preparer.builds.load(Ordering::SeqCst), 1);
}
