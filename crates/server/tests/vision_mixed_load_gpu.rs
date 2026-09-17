//! GitHub #181 — the whole vision path under an agentic load, on one load of
//! the real model behind the production HTTP surface: `cuda_scheduler` with
//! `--vision`, `--spec dflash2`, prompt reuse on, and every image fetched by
//! URL from a loopback image server (`--media-allow-private-network`).
//!
//! Each piece has its own GPU test already — the canary, DFlash2 on a
//! multimodal prefill, media-aware prefix identity, a multimodal sequence
//! through KV-RAM. Here they run together, in the serving shape, and every
//! lane must stay sane:
//!
//! 1. **Interleaved.** Three text lanes are decoding when an image request
//!    arrives; its prefill, in 128-token chunks, runs between their rounds.
//!    Each text lane still answers its own question, the image request
//!    answers about its image, and its first token arrives while a text
//!    lane is still streaming.
//! 2. **Siblings.** Over one system prompt, a red swatch, then a blue swatch
//!    and the red one again while the first is still warm. The swatches are
//!    the same size, so every prompt has the same token ids: only the
//!    image's identity keeps the blue sibling off the red one's pages. Each
//!    names its own colour, and the reused-token counters move.
//! 3. **KV-RAM.** The KV pool holds exactly one max-context sequence. An
//!    Agent image request is decoding when an Interactive request whose
//!    reservation is the whole pool arrives: the image request — the only
//!    Agent, so the eviction order's first victim — is snapshotted to
//!    KV-RAM, restored once the intruder is done, and still describes its
//!    image.
//!
//! What HTTP cannot show is read off the request log (`ignis.request.*`),
//! recorded by a tracing layer: that the evicted and the restored request
//! are that Agent request, not a re-prefill, and that the drafter drafted.
//!
//! Sane, not token-identical: a decode round's width is part of its
//! numerics (`cuda_leaf_vision_gpu.rs`'s module doc). hq-e8-2b, the serving
//! default, since nothing here carries a derived tolerance.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request};
use axum::routing::get;
use axum::Router;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use ignis_artifact::{FrontendSet, Reader};
use ignis_core::gpu_profile;
use ignis_core::{KvFormat, KvGeometry, Speculation, SpeculativeBackend, Vision};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::Engine;
use ignis_server::media::{load_processor, MediaAcquirer, MediaPolicy};
use ignis_server::runtime::{cuda_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::Server;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const MAX_CONTEXT: u32 = 4096;
/// Narrower than the canary image's 196-token placeholder run.
const PREFILL_CHUNK: u32 = 128;
/// How long any one phase may take before the test calls it hung.
const PHASE_DEADLINE: Duration = Duration::from_secs(300);

/// Longer than one 64-token KV page, so the block is a prefix of its own
/// (the same prompt `vision_prefix_reuse_gpu.rs` measures reuse on).
const SYSTEM: &str = "You are a meticulous visual assistant working inside an automated \
    quality-control pipeline. Every request shows you exactly one image. Look at the \
    whole image before answering, never guess about details you cannot see, and keep \
    every answer as short as the question allows. When a question asks for a colour, \
    name the single dominant colour of the image in plain English, using one common \
    word such as red, green, blue, yellow, black or white.";
const SWATCH_QUESTION: &str = "This image comes from a batch of flat colour swatches that a \
    printer produced during calibration. The operator needs to log the colour of each \
    swatch before the batch can be approved, and the log accepts one lowercase word \
    per swatch. Please look at the swatch shown above and tell me its colour. \
    Answer with one word only.";

// ── the load ────────────────────────────────────────────────────────────────

/// The router and its metrics over one GPU load. `Drop` releases the router
/// first and then joins the model thread, so the VRAM is free on return
/// (`openai_http_gpu.rs`'s harness, GitHub #71).
struct Harness {
    app: Option<Router>,
    metrics: Router,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        drop(self.app.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

impl Harness {
    fn app(&self) -> &Router {
        self.app.as_ref().expect("app is only taken by Drop")
    }
}

/// The KV pool's byte budget: exactly one `MAX_CONTEXT` sequence of hq pages,
/// so a request reserving the whole context can only run alone.
fn one_context_pool_bytes() -> u64 {
    KvFormat::HqE8_2b.bytes_per_token(KvGeometry::qwen38_27b()) * u64::from(MAX_CONTEXT)
}

fn harness() -> Option<Harness> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let vision = Vision::default();
    let shape = EngineShape {
        prefill_chunk: PREFILL_CHUNK,
        max_context: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: Some(one_context_pool_bytes()),
        prompt_reuse: true,
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7")),
        vision: Some(vision),
        ..EngineShape::default()
    };
    let scheduler = match cuda_scheduler(path, MODEL.into(), eos, shape) {
        Ok(scheduler) => scheduler,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("cuda_scheduler: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    // `main.rs`'s `--vision` wiring, with the loopback image server allowed.
    let processor = load_processor(&frontend, vision, MAX_CONTEXT).expect("vision processor");
    let acquirer = MediaAcquirer::new(
        Arc::new(processor.clone()),
        processor.options().clone(),
        MediaPolicy::new(true, 64 << 20),
    );
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor);
    let server = Server::new(engine, Box::new(provider))
        .with_media(Arc::new(acquirer))
        .with_metrics()
        .with_request_timeout(PHASE_DEADLINE);
    let metrics = server.metrics_app().expect("metrics are on");
    Some(Harness { app: Some(server.app()), metrics, driver: Some(driver) })
}

// ── the image server ────────────────────────────────────────────────────────

/// A flat 256x256 PNG of one colour.
fn swatch(rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, 256, 256);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let pixels: Vec<u8> = (0..256 * 256).flat_map(|_| rgb).collect();
        encoder.write_header().unwrap().write_image_data(&pixels).unwrap();
    }
    out
}

/// Serve `/number.png` (the canary's "47"), `/red.png` and `/blue.png` on a
/// loopback port; the base URL.
async fn image_server() -> String {
    let number = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision_canary/number.png"),
    )
    .expect("the canary's number image");
    let png = |bytes: Vec<u8>| move || {
        let bytes = bytes.clone();
        async move { ([(header::CONTENT_TYPE, "image/png")], bytes) }
    };
    let app = Router::new()
        .route("/number.png", get(png(number)))
        .route("/red.png", get(png(swatch([220, 20, 20]))))
        .route("/blue.png", get(png(swatch([20, 20, 220]))));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind the image server");
    let address = listener.local_addr().expect("its address");
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve images") });
    format!("http://{address}")
}

// ── the request log ─────────────────────────────────────────────────────────

/// One `ignis.request.*` event, with the fields this test reads.
#[derive(Debug, Clone, Default)]
struct Logged {
    name: &'static str,
    request_id: Option<u64>,
    class: Option<String>,
    drafted: Option<u64>,
}

impl Visit for Logged {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "request_id" => self.request_id = Some(value),
            "spec.drafted" => self.drafted = Some(value),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "class" {
            self.class = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {}
}

/// Records every `ignis.request.*` event the server emits.
#[derive(Clone, Default)]
struct RequestLog(Arc<Mutex<Vec<Logged>>>);

impl<S: tracing::Subscriber> Layer<S> for RequestLog {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        let name = event.metadata().name();
        if name.starts_with("ignis.request.") {
            let mut logged = Logged { name, ..Logged::default() };
            event.record(&mut logged);
            self.0.lock().unwrap().push(logged);
        }
    }
}

impl RequestLog {
    fn named(&self, name: &str) -> Vec<Logged> {
        self.0.lock().unwrap().iter().filter(|e| e.name == name).cloned().collect()
    }
}

// ── requests ────────────────────────────────────────────────────────────────

/// A greedy, thinking-off streaming chat body.
fn chat(messages: Value, max_tokens: u32, class: &str) -> Value {
    json!({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
        "enable_thinking": false,
        "class": class,
    })
}

fn text(question: &str, max_tokens: u32, class: &str) -> Value {
    chat(json!([{ "role": "user", "content": question }]), max_tokens, class)
}

fn image(url: &str, question: &str, max_tokens: u32, class: &str) -> Value {
    chat(
        json!([{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": url } },
            { "type": "text", "text": question },
        ]}]),
        max_tokens,
        class,
    )
}

fn swatch_request(url: &str) -> Value {
    chat(
        json!([
            { "role": "system", "content": SYSTEM },
            { "role": "user", "content": [
                { "type": "image_url", "image_url": { "url": url } },
                { "type": "text", "text": SWATCH_QUESTION },
            ]},
        ]),
        16,
        "interactive",
    )
}

/// What a streamed completion has delivered so far.
#[derive(Debug, Default)]
struct Streamed {
    text: String,
    tokens: usize,
    first_token: Option<Instant>,
    finish_reason: Option<String>,
    done: Option<Instant>,
}

/// A completion streaming in the background.
struct Stream {
    name: &'static str,
    state: Arc<Mutex<Streamed>>,
    task: tokio::task::JoinHandle<()>,
}

impl Stream {
    fn start(app: &Router, name: &'static str, body: Value) -> Self {
        let state = Arc::new(Mutex::new(Streamed::default()));
        let (app, shared) = (app.clone(), Arc::clone(&state));
        let task = tokio::spawn(async move {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = app.oneshot(request).await.unwrap();
            let status = response.status();
            if !status.is_success() {
                let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
                panic!("{name}: HTTP {status}: {}", String::from_utf8_lossy(&bytes));
            }
            let mut body = response.into_body();
            let mut pending = String::new();
            while let Some(frame) = body.frame().await {
                let frame = frame.unwrap_or_else(|e| panic!("{name}: SSE frame: {e}"));
                let Some(data) = frame.data_ref() else { continue };
                pending.push_str(&String::from_utf8_lossy(data));
                while let Some(end) = pending.find("\n\n") {
                    let event: String = pending.drain(..end + 2).collect();
                    for line in event.lines() {
                        let Some(payload) = line.strip_prefix("data: ") else { continue };
                        if payload == "[DONE]" {
                            continue;
                        }
                        let chunk: Value = serde_json::from_str(payload)
                            .unwrap_or_else(|e| panic!("{name}: chunk {payload:?}: {e}"));
                        if let Some(error) = chunk.get("error") {
                            panic!("{name}: stream error {error}");
                        }
                        let mut state = shared.lock().unwrap();
                        if let Some(content) = chunk.pointer("/choices/0/delta/content").and_then(Value::as_str) {
                            if !content.is_empty() {
                                state.text.push_str(content);
                                state.tokens += 1;
                                state.first_token.get_or_insert_with(Instant::now);
                            }
                        }
                        if let Some(reason) = chunk.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
                            state.finish_reason = Some(reason.to_owned());
                        }
                    }
                }
            }
            shared.lock().unwrap().done = Some(Instant::now());
        });
        Self { name, state, task }
    }

    fn tokens(&self) -> usize {
        self.state.lock().unwrap().tokens
    }

    /// Wait until it has streamed `n` tokens; fails if it ends first.
    async fn until_tokens(&self, n: usize) {
        let deadline = Instant::now() + PHASE_DEADLINE;
        while self.tokens() < n {
            assert!(
                self.state.lock().unwrap().done.is_none() && !self.task.is_finished(),
                "{} ended after {} tokens, before {n}: {:?}",
                self.name,
                self.tokens(),
                self.state.lock().unwrap()
            );
            assert!(Instant::now() < deadline, "{} streamed no {n} tokens in time", self.name);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Wait for it to finish; everything it streamed.
    async fn finish(self) -> Streamed {
        tokio::time::timeout(PHASE_DEADLINE, self.task)
            .await
            .unwrap_or_else(|_| panic!("{} did not finish in time", self.name))
            .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()));
        let state = std::mem::take(&mut *self.state.lock().unwrap());
        eprintln!(
            "{}: {} tokens, finish {:?}: {:?}",
            self.name, state.tokens, state.finish_reason, state.text
        );
        assert!(
            matches!(state.finish_reason.as_deref(), Some("stop" | "length")),
            "{} ended without a real finish reason: {state:?}",
            self.name
        );
        state
    }
}

/// The sum of a Prometheus metric over all its label sets.
async fn metric(metrics: &Router, name: &str) -> f64 {
    let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    assert!(response.status().is_success(), "GET /metrics: {}", response.status());
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    text.lines()
        .filter(|line| {
            line.strip_prefix(name).is_some_and(|rest| rest.starts_with(' ') || rest.starts_with('{'))
        })
        .map(|line| line.rsplit(' ').next().unwrap().parse::<f64>().unwrap())
        .sum()
}

// ── the test ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
async fn an_agentic_load_with_images_by_url_keeps_every_lane_sane() {
    // The request log, printed for the profile's output and recorded for the
    // assertions below.
    let log = RequestLog::default();
    tracing_subscriber::registry()
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .with(tracing_subscriber::fmt::layer())
        .with(log.clone())
        .try_init()
        .expect("this test binary installs the only subscriber");
    let Some(h) = harness() else { return };
    let images = image_server().await;
    let number = format!("{images}/number.png");

    // 1. An image prefill interleaved with three decoding text lanes.
    let lanes = [
        ("primes", "List the first forty prime numbers, separated by commas.", "173"),
        ("months", "Name the twelve months of the year in order, separated by commas.", "September"),
        ("words", "Write the English words for the numbers one through thirty, separated by commas.", "twenty"),
    ];
    let streams: Vec<Stream> =
        lanes.iter().map(|(name, question, _)| Stream::start(h.app(), name, text(question, 200, "interactive"))).collect();
    for stream in &streams {
        stream.until_tokens(2).await;
    }
    let picture = Stream::start(
        h.app(),
        "image",
        image(&number, "What number is shown in the image? Answer in one short sentence.", 32, "interactive"),
    );
    let picture = picture.finish().await;
    let mut still_streaming = false;
    for (stream, (name, _, expected)) in streams.into_iter().zip(lanes) {
        let lane = stream.finish().await;
        assert!(lane.text.contains(expected), "text lane {name} must still answer its question: {:?}", lane.text);
        still_streaming |= lane.done.unwrap() > picture.first_token.unwrap();
    }
    assert!(picture.text.contains("47"), "the image request answers about its image: {:?}", picture.text);
    assert!(still_streaming, "the image's first token arrived while a text lane was still decoding");

    // 2. Siblings over one system prompt: same-size swatches, identical ids.
    let reused = |metrics: &Router| {
        let metrics = metrics.clone();
        async move {
            metric(&metrics, "ignis_prefix_reused_tokens_total").await
                + metric(&metrics, "ignis_retained_reused_tokens_total").await
        }
    };
    let red = Stream::start(h.app(), "red", swatch_request(&format!("{images}/red.png")));
    red.until_tokens(1).await;
    let reused_before = reused(&h.metrics).await;
    let blue = Stream::start(h.app(), "blue", swatch_request(&format!("{images}/blue.png")));
    let red_again = Stream::start(h.app(), "red again", swatch_request(&format!("{images}/red.png")));
    let (red, blue, red_again) = (red.finish().await, blue.finish().await, red_again.finish().await);
    let reused_tokens = reused(&h.metrics).await - reused_before;
    eprintln!("siblings reused {reused_tokens} prompt tokens");
    assert!(red.text.to_lowercase().contains("red"), "{:?}", red.text);
    assert!(blue.text.to_lowercase().contains("blue"), "the blue sibling is not handed the red image: {:?}", blue.text);
    assert!(red_again.text.to_lowercase().contains("red"), "{:?}", red_again.text);
    assert!(reused_tokens > 0.0, "the siblings reused a warm prefix");

    // 3. An Agent image request through KV-RAM.
    let evictions_before = metric(&h.metrics, "ignis_kv_cache_evictions_total").await;
    let agent = Stream::start(
        h.app(),
        "agent image",
        image(&number, "Describe this image in detail: the digits, their colour, the background.", 160, "agent"),
    );
    agent.until_tokens(4).await;
    // Its prompt is a few dozen tokens, so its reservation is the whole pool.
    let intruder =
        Stream::start(h.app(), "intruder", text("What is 2 + 2? Answer with a number only.", MAX_CONTEXT - 64, "interactive"));
    let intruder = intruder.finish().await;
    let agent = agent.finish().await;
    let evictions = metric(&h.metrics, "ignis_kv_cache_evictions_total").await - evictions_before;
    eprintln!("KV-RAM: {evictions} evictions for the intruder");
    assert!(evictions >= 1.0, "the Agent image request went to KV-RAM for the intruder: {evictions} evictions");
    assert!(intruder.text.contains('4'), "{:?}", intruder.text);
    assert!(agent.tokens > 4, "the restored image request kept decoding");
    assert!(agent.text.contains("47"), "restored, it still describes its image: {:?}", agent.text);

    // The request log: the one Agent request is the one evicted and restored,
    // and speculation drafted — for it, and across the load.
    let done = log.named("ignis.request.done");
    let agents: Vec<u64> =
        done.iter().filter(|e| e.class.as_deref() == Some("agent")).filter_map(|e| e.request_id).collect();
    assert_eq!(agents.len(), 1, "one Agent request: {done:?}");
    let ids = |name| log.named(name).iter().filter_map(|e| e.request_id).collect::<Vec<_>>();
    assert_eq!(ids("ignis.request.evicted"), agents, "the Agent image request is the one evicted");
    assert_eq!(ids("ignis.request.restored"), agents, "and restored from its blob, not re-prefilled");
    let agent_drafted = done.iter().find(|e| e.request_id == Some(agents[0])).and_then(|e| e.drafted);
    assert!(agent_drafted.is_some_and(|n| n > 0), "DFlash2 drafted for the restored request: {agent_drafted:?}");
    assert!(done.iter().all(|e| e.drafted.is_some()), "every request ran speculative rounds: {done:?}");
}
