//! Flash-Next's MTP speculation stays on with images (GitHub #564, spec #555,
//! ADR 0087), on the real artifact behind the server: the Flash-Next
//! scheduler the server builds (`runtime::flash_next_scheduler`) with
//! `--vision-enabled` and `--spec-backend mtp`, the production router over it,
//! and the four vision canary images (`tests/fixtures/vision_canary/`).
//!
//! The head takes the trunk's three-axis positions and, where its next token
//! is an image placeholder, the vision row the trunk took for it -- a rule of
//! ours, since the reference has no MTP head. Its drafts change only how many
//! tokens a round commits; the trunk verifies every one, so:
//!
//! - **Greedy MTP on equals MTP off.** Each canary image with two questions --
//!   the fixture's own, whose answer is a sentence, and a request for a long
//!   description that runs to the token budget, so the acceptance has a
//!   sample -- generated on a load with the head and on one without it, token
//!   for token. A verify round runs `k + 1` columns through other kernel
//!   shapes than a plain round, so a split passes only as a near tie: the
//!   prefill route of the load without the head, after the agreed text (the
//!   same tokens through both runs up to the split, teacher-forced), puts the
//!   head's pick within `PICK_GAP_ULPS` (16) BF16 ulps under the headless
//!   run's, or above it -- ADR 0056's near-tie bound
//!   (`crates/core/tests/support/logit_drift.rs`). Counted at the larger
//!   pick's magnitude, at most the row's largest, so never fewer ulps than ADR
//!   0056 counts; on the prefill route, the one row a readout gives, where
//!   ADR 0056 reads the plain round's. The owner's call (2026-11-10, #564,
//!   #565): after an image every route pair drifts about twice as far as on
//!   text (`crates/core/tests/flash_next_vision_speculative_gpu.rs`), with no
//!   state a verify round fails to keep; text with only the vision start and
//!   end tokens around it drifts as far, with no image, no three-axis
//!   positions and no pool, so the two special tokens carry it: a property of
//!   the model's inputs, not a defect. The one-ulp rule this test held first,
//!   the text path's (`crates/core/tests/flash_next_speculative_gpu.rs`,
//!   `assert_near_tie`), is narrower than that drift; making the routes agree
//!   bit for bit is GitHub #567.
//!   That rule's verdict is printed beside each split. Read through a
//!   decision readout of the two picks' logits at that position.
//! - **Nothing refuses or silently disables the head with vision on.** Every
//!   image run on the head's load reports verify rounds, and the long ones
//!   drafts; through `/v1/chat/completions`, each image's long description is
//!   the engine's own run of the same prompt, and the load's
//!   `ignis_speculative_*` counters move over those requests.
//! - **The acceptance on image prompts is printed beside text prompts'**:
//!   accepted of drafted, and tokens a round. Reported, not gated.
//!
//! The runs go through the server's engine -- its model thread, scheduler,
//! media cache, encode and leaf -- with prompts the artifact's own frontend
//! prepares, as the routes prepare them, because a split is judged on token
//! ids, which the routes do not return. A 128-token prefill chunk, narrower
//! than an image, so a span ends inside each image and the head's entry there
//! takes the image's next column. BF16 KV, the correctness oracle (ADR 0022);
//! prompt reuse off, so each run prefills its whole prompt.
//!
//! ```text
//! cargo test -p ignis-server --features cuda --test flash_next_vision_mtp_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`) with its MTP and vision companions. Explicit GPU
//! profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing artifact,
//! companion or GPU is a skip, under it a failure. Two loads one after the
//! other, each pinning the ~38 GB expert pool.

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::vision::VisionProcessor;
use ignis_artifact::{ChatMessage, ChatRenderOptions, ContentPart, FrontendSet, MessageContent, Reader, Role};
use ignis_core::types::{DecisionRead, DecodeParams, RequestClass, RequestInput, SchedEvent, SpecCounters};
use ignis_core::vision::Multimodal;
use ignis_core::{gpu_profile, KvFormat, Speculation, SpeculativeBackend, Vision, VisionResidency};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::engine::{collect_readout, Engine};
use ignis_server::media::{load_processor, MediaAcquirer, MediaPolicy};
use ignis_server::runtime::{flash_next_scheduler, EngineShape};
use ignis_server::telemetry::SystemClock;
use ignis_server::Server;

use support::media::data_uri;
use support::vision_canary::load_canaries;

// ADR 0056's near-tie bound and one BF16 ulp, as the core tests' logit drift
// reads them: one copy of the rule.
#[path = "../../core/tests/support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{bf16_ulp, PICK_GAP_ULPS};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
const MAX_CONTEXT: u32 = 8192;
/// Narrower than a canary image's placeholder run.
const PREFILL_CHUNK: u32 = 128;
/// The head's draft tokens, the `rtxpro6000` profile's.
const DRAFT_TOKENS: u32 = 3;
/// Tokens each prompt may generate.
const TOTAL: u32 = 64;
/// How long one request may take before the test calls it hung.
const DEADLINE: Duration = Duration::from_secs(300);
/// The second question per canary image, whose answer runs to the token
/// budget rather than stopping after one sentence.
const DETAIL_QUESTION: &str = "Describe everything you can see in this image, in detail.";
/// Text prompts of the same budget, whose acceptance the images' is printed beside.
const TEXT_QUESTIONS: [&str; 4] = [
    "Describe a busy harbour at sunrise, in detail.",
    "Explain how a binary search works, step by step.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system, with one fact about each.",
];

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

/// The vision both loads take: the context's envelope, the tower in host
/// memory (the RTX 5090 profile's residency).
fn vision() -> Vision {
    Vision::new(MAX_CONTEXT).expect("an envelope").with_residency(VisionResidency::Host)
}

// ── the loads ───────────────────────────────────────────────────────────────

/// One load behind the router: the app, its metrics, the engine handle the
/// runs submit to, and the model thread. `Drop` releases the router and the
/// handle first, then joins the model thread, so the VRAM is free on return
/// (`openai_http_gpu.rs`'s harness, GitHub #71).
struct Load {
    app: Option<Router>,
    metrics: Option<Router>,
    engine: Option<Engine>,
    driver: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Load {
    fn drop(&mut self) {
        drop(self.app.take());
        drop(self.metrics.take());
        drop(self.engine.take());
        if let Some(driver) = self.driver.take() {
            let _ = driver.join();
        }
    }
}

impl Load {
    fn app(&self) -> &Router {
        self.app.as_ref().expect("taken only by Drop")
    }

    fn engine(&self) -> &Engine {
        self.engine.as_ref().expect("taken only by Drop")
    }
}

/// A Flash-Next load with vision, and with the MTP head when `mtp`, or `None`
/// through `skip_or_fail`.
fn load(dir: &Path, mtp: bool) -> Option<Load> {
    let path = dir.join(ARTIFACT_FILE_NAME);
    let reader = Reader::open(&path).unwrap_or_else(|e| panic!("open {}: {e:?}", path.display()));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let shape = EngineShape {
        max_context: MAX_CONTEXT,
        prefill_chunk: PREFILL_CHUNK,
        kv_format: KvFormat::Bf16,
        decode_lanes: 1,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        host_pool_bytes: 0,
        speculation: mtp.then(|| Speculation::new(SpeculativeBackend::Mtp, DRAFT_TOKENS).expect("mtp-3")),
        vision: Some(vision()),
        ..EngineShape::default()
    };
    let scheduler = match flash_next_scheduler(&path, MODEL.into(), eos, shape, None) {
        Ok((scheduler, _reservations)) => scheduler,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load Flash-Next with vision (MTP {mtp}): {e}"));
            return None;
        }
    };
    // `main.rs`'s `--vision-enabled` wiring.
    let processor = load_processor(&frontend, vision(), MAX_CONTEXT).expect("the vision processor");
    let acquirer =
        MediaAcquirer::new(Arc::new(processor.clone()), processor.options().clone(), MediaPolicy::new(false, 64 << 20));
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor);
    let server = Server::new(engine.clone(), Box::new(provider))
        .with_media(Arc::new(acquirer))
        .with_metrics()
        .with_request_timeout(DEADLINE);
    let metrics = server.metrics_app().expect("metrics are on");
    Some(Load { app: Some(server.app()), metrics: Some(metrics), engine: Some(engine), driver: Some(driver) })
}

/// A load's pinned expert pool frees asynchronously: wait, bounded, until the
/// host has back what it had before the load.
fn wait_for_host_memory(before: u64) {
    use ignis_core::residency::available_physical_bytes;
    for _ in 0..60 {
        if available_physical_bytes().unwrap_or(0) + (1 << 30) >= before {
            return;
        }
        std::thread::sleep(Duration::from_secs(2));
    }
}

// ── the prompts ─────────────────────────────────────────────────────────────

/// A prompt as the server's frontend prepares it: its tokens, and an image's
/// positions, rope delta and media item.
#[derive(Clone)]
struct Prompt {
    name: String,
    image: Option<Vec<u8>>,
    question: String,
    tokens: Vec<u32>,
    multimodal: Option<Multimodal>,
}

/// Thinking off, as the vision canary renders (`vision_dflash2_gpu.rs`'s
/// `prompt_for`), with the image first when there is one.
fn prepare(frontend: &FrontendSet, processor: &VisionProcessor, name: String, image: Option<Vec<u8>>, question: &str) -> Prompt {
    let mut parts = Vec::new();
    if image.is_some() {
        parts.push(ContentPart::Image { url: None });
    }
    parts.push(ContentPart::Text(question.to_owned()));
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(parts),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let options = ChatRenderOptions { enable_thinking: false, ..Default::default() };
    let (tokens, multimodal) = match &image {
        Some(bytes) => {
            let prepared = frontend
                .prepare_prompt(processor, &messages, &[bytes.as_slice()], options, None)
                .unwrap_or_else(|e| panic!("{name}: prepare the prompt: {e}"));
            let (tokens, multimodal) = Multimodal::from_prepared(prepared);
            assert_eq!(multimodal.media.len(), 1, "{name}: one image");
            assert_ne!(multimodal.rope_delta, 0, "{name}: an image's rope delta");
            (tokens, Some(multimodal))
        }
        None => {
            let rendered = frontend
                .chat_template()
                .render_with_thinking_and_tools(&messages, options, None)
                .unwrap_or_else(|e| panic!("{name}: render: {e}"));
            (frontend.tokenizer().encode(&rendered).unwrap_or_else(|e| panic!("{name}: tokenize: {e}")), None)
        }
    };
    Prompt { name, image, question: question.to_owned(), tokens, multimodal }
}

/// The four canaries with their own question and the long one, then the text
/// prompts.
fn prompts(dir: &Path) -> (Vec<Prompt>, Vec<Prompt>) {
    let reader = Reader::open(&dir.join(ARTIFACT_FILE_NAME)).expect("open the artifact");
    let frontend = FrontendSet::from_reader(&reader).expect("frontend");
    let processor = load_processor(&frontend, vision(), MAX_CONTEXT).expect("the vision processor");
    let mut images = Vec::new();
    for canary in load_canaries(&frontend) {
        for (kind, question) in [("own", canary.question.as_str()), ("detail", DETAIL_QUESTION)] {
            let name = format!("{} ({kind})", canary.id);
            images.push(prepare(&frontend, &processor, name, Some(canary.image.clone()), question));
        }
    }
    let texts = TEXT_QUESTIONS
        .iter()
        .enumerate()
        .map(|(i, q)| prepare(&frontend, &processor, format!("text {i}"), None, q))
        .collect();
    (images, texts)
}

/// A greedy request of `tokens` (and their image), or a decision readout.
fn input(tokens: Vec<u32>, multimodal: Option<Multimodal>, decision: Option<DecisionRead>) -> RequestInput {
    RequestInput {
        decision,
        model: MODEL.into(),
        tokens,
        params: DecodeParams { max_tokens: Some(TOTAL), ..DecodeParams::default() },
        multimodal: multimodal.map(Arc::new),
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
    }
}

// ── the runs ────────────────────────────────────────────────────────────────

/// A greedy run: its tokens, and the speculative rounds it reported (none on
/// a load without the head).
struct Run {
    tokens: Vec<u32>,
    spec: Option<SpecCounters>,
}

async fn run(load: &Load, prompt: &Prompt) -> Run {
    let input = input(prompt.tokens.clone(), prompt.multimodal.clone(), None);
    let (_, mut events) = load
        .engine()
        .submit(input, RequestClass::Interactive)
        .await
        .unwrap_or_else(|e| panic!("{}: submit: {e:?}", prompt.name));
    let mut tokens = Vec::new();
    loop {
        let event = tokio::time::timeout(DEADLINE, events.recv())
            .await
            .unwrap_or_else(|_| panic!("{}: no event in {DEADLINE:?}", prompt.name))
            .unwrap_or_else(|| panic!("{}: the engine gave up", prompt.name));
        match event {
            SchedEvent::Token { token, .. } => tokens.push(token),
            SchedEvent::Done { reason, spec, .. } => {
                assert!(!tokens.is_empty(), "{}: nothing generated ({reason:?})", prompt.name);
                return Run { tokens, spec };
            }
            _ => {}
        }
    }
}

/// The logits of `picks` that the prefill route of `load` gives after the
/// prompt and `agreed`: a decision readout at that position.
async fn prefill_logits(load: &Load, prompt: &Prompt, agreed: &[u32], picks: [u32; 2]) -> [f32; 2] {
    let mut tokens = prompt.tokens.clone();
    tokens.extend_from_slice(agreed);
    let multimodal = prompt.multimodal.clone().map(|mut m| {
        assert!(m.append_text(agreed.len()), "{}: the agreed text continues the prompt's positions", prompt.name);
        m
    });
    let decision = DecisionRead::Answers(Arc::from(picks.to_vec()));
    let input = input(tokens, multimodal, Some(decision));
    let (_, mut events) = load
        .engine()
        .submit(input, RequestClass::Interactive)
        .await
        .unwrap_or_else(|e| panic!("{}: submit the readout: {e:?}", prompt.name));
    let readout = collect_readout(&mut events, DEADLINE)
        .await
        .unwrap_or_else(|e| panic!("{}: the readout: {e:?}", prompt.name));
    [readout.logits[0], readout.logits[1]]
}

/// `on` against `off`: identical, or identical up to a first split where the
/// prefill route on `off_load`, after the agreed text, rates the two picks
/// within ADR 0056's near tie: `on`'s pick at most `PICK_GAP_ULPS` BF16 ulps
/// under `off`'s at the larger pick's magnitude, or above it. A run that
/// ended where the other goes on picked `eos` there. A line per split, the
/// one-ulp rule's verdict beside; panics on a split past the near tie.
async fn judge(off_load: &Load, prompt: &Prompt, eos: u32, off: &[u32], on: &[u32]) -> Option<String> {
    let at = match off.iter().zip(on).position(|(a, b)| a != b) {
        Some(at) => at,
        None if off.len() == on.len() => return None,
        None => off.len().min(on.len()),
    };
    let pick = |run: &[u32]| run.get(at).copied().unwrap_or(eos);
    let (off_pick, on_pick) = (pick(off), pick(on));
    let [off_logit, on_logit] = prefill_logits(off_load, prompt, &off[..at], [off_pick, on_pick]).await;
    let gap = off_logit - on_logit;
    let ulp = bf16_ulp(off_logit.abs().max(on_logit.abs()));
    let ulps = gap / ulp;
    let line = format!(
        "{}: MTP on splits from MTP off at token {at}: off {off_pick} on {on_pick}; the prefill route's gap off - on = {gap} \
         ({ulps} ulps of {ulp}): ADR 0056's near tie ({PICK_GAP_ULPS} ulps) {}, the one-ulp rule {}",
        prompt.name,
        if ulps <= PICK_GAP_ULPS { "holds" } else { "FAILS" },
        if gap <= ulp { "holds" } else { "would refuse it" }
    );
    println!("{line}");
    assert!(ulps <= PICK_GAP_ULPS, "{line}: the prefill route sides with MTP off past ADR 0056's near tie");
    Some(line)
}

// ── the routes ──────────────────────────────────────────────────────────────

/// A greedy, thinking-off chat completion with `prompt`'s image as a data
/// URI: its content.
async fn chat(app: &Router, prompt: &Prompt) -> String {
    let image = prompt.image.as_ref().expect("an image prompt");
    let body = json!({
        "model": MODEL,
        "messages": [{ "role": "user", "content": [
            { "type": "image_url", "image_url": { "url": data_uri(image) } },
            { "type": "text", "text": prompt.question },
        ]}],
        "max_tokens": TOTAL,
        "temperature": 0,
        "stream": false,
        "enable_thinking": false,
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(status.is_success(), "{}: HTTP {status}: {}", prompt.name, String::from_utf8_lossy(&bytes));
    let value: Value = serde_json::from_slice(&bytes).expect("a JSON completion");
    value.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// A Prometheus counter's value, 0 while the listener does not show it yet.
async fn counter(metrics: &Router, name: &str) -> u64 {
    let request = Request::builder().uri("/metrics").body(Body::empty()).unwrap();
    let response = metrics.clone().oneshot(request).await.unwrap();
    assert!(response.status().is_success(), "GET /metrics: {}", response.status());
    let text = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
    text.lines()
        .find_map(|line| line.strip_prefix(name).and_then(|rest| rest.strip_prefix(' ')))
        .map_or(0, |value| value.trim().parse::<f64>().expect("a counter") as u64)
}

/// The load's verify rounds and drafts as `/metrics` counts them, once `done`
/// holds of them or, bounded, as they stand: the telemetry task sums a
/// request's counters after it ends, so they may trail its response.
async fn speculative_counters_once(metrics: &Router, done: impl Fn((u64, u64)) -> bool) -> (u64, u64) {
    let mut now = (0, 0);
    for _ in 0..100 {
        now = (
            counter(metrics, "ignis_speculative_rounds_total").await,
            counter(metrics, "ignis_speculative_drafted_tokens_total").await,
        );
        if done(now) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    now
}

// ── the report ──────────────────────────────────────────────────────────────

fn add(total: &mut SpecCounters, spec: &SpecCounters) {
    total.rounds += spec.rounds;
    total.drafted += spec.drafted;
    total.accepted += spec.accepted;
}

fn acceptance_line(what: &str, spec: &SpecCounters) -> String {
    let rate = if spec.drafted == 0 { 0.0 } else { f64::from(spec.accepted) / f64::from(spec.drafted) };
    let per_round =
        if spec.rounds == 0 { 0.0 } else { f64::from(spec.rounds + spec.accepted) / f64::from(spec.rounds) };
    format!(
        "{what:28} {:4} rounds  {:4} drafted  {:4} accepted  acceptance {:5.1}%  {per_round:.2} tokens a round",
        spec.rounds,
        spec.drafted,
        spec.accepted,
        100.0 * rate
    )
}

// ── the test ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "GPU profile only: Flash-Next with its MTP and vision companions on the RTX 5090"]
async fn mtp_on_equals_mtp_off_on_the_canary_images_and_reports_the_acceptance() {
    let dir = model_dir();
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let mtp_companion = ignis_core::flash_next_mtp::companion_path(&dir);
    if !mtp_companion.exists() && gpu_profile::skip_or_fail(&format!("no MTP companion at {}", mtp_companion.display())) {
        return;
    }
    let (images, texts) = prompts(&dir);
    let frontend = {
        let reader = Reader::open(&path).expect("open the artifact");
        FrontendSet::from_reader(&reader).expect("frontend")
    };
    let eos = frontend.eos_token_id().expect("eos");
    let before = ignis_core::residency::available_physical_bytes().unwrap_or(0);

    // MTP on: every prompt through the engine, each image's long description
    // through the route too.
    let (on_images, on_texts) = {
        let Some(on) = load(&dir, true) else { return };
        let mut on_images = Vec::new();
        for prompt in &images {
            on_images.push(run(&on, prompt).await);
        }
        let mut on_texts = Vec::new();
        for prompt in &texts {
            on_texts.push(run(&on, prompt).await);
        }
        // The counters are the request log's, summed by the telemetry task
        // after each request ends: first at least the engine's runs'.
        let metrics = on.metrics.as_ref().expect("metrics");
        let summed = |runs: &[&Run], field: fn(&SpecCounters) -> u32| -> u64 {
            runs.iter().filter_map(|run| run.spec.as_ref()).map(|spec| u64::from(field(spec))).sum()
        };
        let runs: Vec<&Run> = on_images.iter().chain(&on_texts).collect();
        let engine = (summed(&runs, |s| s.rounds), summed(&runs, |s| s.drafted));
        let counted = speculative_counters_once(metrics, |now| now.0 >= engine.0 && now.1 >= engine.1).await;
        assert!(counted.0 >= engine.0 && counted.1 >= engine.1, "the counters {counted:?} hold the engine's runs' {engine:?}");
        for (prompt, engine_run) in images.iter().zip(&on_images).filter(|(p, _)| p.question == DETAIL_QUESTION) {
            let content = chat(on.app(), prompt).await;
            let mut ids = engine_run.tokens.clone();
            if ids.last() == Some(&eos) {
                ids.pop();
            }
            let engine_text = frontend.tokenizer().decode(&ids).expect("decode");
            println!("{} through /v1/chat/completions: {content:?}", prompt.name);
            assert_eq!(content.trim(), engine_text.trim(), "{}: the route's answer is not the engine's run", prompt.name);
        }
        // Then the route's requests on top of them.
        let after = speculative_counters_once(metrics, |now| now.0 > counted.0 && now.1 > counted.1).await;
        let moved = (after.0 - counted.0, after.1 - counted.1);
        println!("through the route: {} verify rounds, {} drafts over the four long descriptions", moved.0, moved.1);
        assert!(moved.0 > 0 && moved.1 > 0, "the route's image requests ran no drafted verify round: {moved:?}");
        (on_images, on_texts)
    };
    wait_for_host_memory(before);

    // MTP off: the image prompts again, and each split judged on this load.
    let Some(off) = load(&dir, false) else { return };
    let mut splits = Vec::new();
    for (prompt, on_run) in images.iter().zip(&on_images) {
        let off_run = run(&off, prompt).await;
        assert!(off_run.spec.is_none(), "{}: a load without the head ran verify rounds", prompt.name);
        if let Some(line) = judge(&off, prompt, eos, &off_run.tokens, &on_run.tokens).await {
            splits.push(line);
        }
    }
    drop(off);

    // The acceptance, image prompts beside text prompts.
    let mut image_total = SpecCounters::default();
    let mut text_total = SpecCounters::default();
    println!("MTP-{DRAFT_TOKENS} acceptance, one lane, BF16 KV, {TOTAL} tokens a prompt:");
    for (prompt, run) in images.iter().zip(&on_images) {
        let spec = run.spec.unwrap_or_else(|| panic!("{}: the head's load ran no verify round", prompt.name));
        println!("  {}", acceptance_line(&prompt.name, &spec));
        add(&mut image_total, &spec);
        if prompt.question == DETAIL_QUESTION {
            assert!(spec.drafted > 0, "{}: the head drafted nothing on a long answer about an image", prompt.name);
        }
    }
    for (prompt, run) in texts.iter().zip(&on_texts) {
        let spec = run.spec.unwrap_or_else(|| panic!("{}: the head's load ran no verify round", prompt.name));
        println!("  {}", acceptance_line(&prompt.name, &spec));
        add(&mut text_total, &spec);
    }
    println!("  {}", acceptance_line("image prompts", &image_total));
    println!("  {}", acceptance_line("text prompts", &text_total));
    println!("{} of {} image runs split from MTP off, each a near-tie", splits.len(), images.len());
}
