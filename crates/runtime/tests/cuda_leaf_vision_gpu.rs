//! GPU coverage for a multimodal request through the production `CudaLeaf`
//! and the scheduler (GitHub #178): an image prompt whose placeholder run
//! spans several prefill chunks, interleaved with text requests already
//! decoding, must leave every text lane sane -- each still answers its own
//! question, as it does when it runs alone -- and answer about the image
//! itself; and 100 image requests in a row leave no media embedding live and
//! the leaf's footprint where it was.
//!
//! GitHub #194: an image request evicted to KV-RAM mid-decode and restored
//! continues with the same greedy tokens as one never evicted — the blob
//! carries its `rope_delta`, and nothing vision-related is live after its
//! image is prefilled.
//!
//! "Sane", not "token-identical": a decode round is one batch-wide traversal,
//! so the round's width is part of its numerics, and no test in this engine
//! claims a lane's tokens are independent of who shares its round. What must
//! not change is the answer.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice,
    FrontendSet, MessageContent, ModelScope, Reader, Role,
};
use ignis_core::gpu_profile;
use ignis_core::vision::Multimodal;
use ignis_core::{
    ConcreteScheduler, DecodeParams, KvFormat, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, TokenId, Vision,
};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const MAX_CONTEXT: u32 = 2048;
const MAX_TOKENS: u32 = 24;
/// The serving chunk: narrower than the image's 196-token placeholder run.
const SERVING_CHUNK: u32 = 64;

fn image() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("server")
        .join("tests")
        .join("fixtures")
        .join("vision_canary")
        .join("number.png");
    std::fs::read(path).expect("the canary's number image")
}

fn params() -> DecodeParams {
    DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() }
}

fn text_input(frontend: &FrontendSet, question: &str) -> RequestInput {
    let messages = [ChatMessage::text(Role::User, question)];
    let rendered = frontend.chat_template().render_with_thinking_and_tools(&messages, ChatRenderOptions { enable_thinking: false, ..Default::default() }, None).expect("render");
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens: frontend.tokenizer().encode(&rendered).expect("tokenize"),
        params: params(),
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        constrained: None,
    }
}

fn image_input(frontend: &FrontendSet, image: &[u8]) -> RequestInput {
    image_question(frontend, image, "What number is shown in the image?")
}

fn image_question(frontend: &FrontendSet, image: &[u8], question: &str) -> RequestInput {
    let processor = frontend.vision_processor().expect("vision processor");
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![
            ContentPart::Image { url: None },
            ContentPart::Text(question.into()),
        ]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend.prepare_prompt(&processor, &messages, &[image], ChatRenderOptions { enable_thinking: false, ..Default::default() }, None).expect("prepare");
    let (tokens, multimodal) = Multimodal::from_prepared(prepared);
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens,
        params: params(),
        multimodal: Some(Arc::new(multimodal)),
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        constrained: None,
    }
}

/// Request `id`'s generated tokens among `events`.
fn generated(events: &[SchedEvent], id: RequestId) -> Vec<TokenId> {
    events
        .iter()
        .filter_map(|event| match event {
            SchedEvent::Token { request, token } if *request == id => Some(*token),
            _ => None,
        })
        .collect()
}

/// The scheduler configuration both tests start from: the serving chunk and
/// the leaf's own page geometry, as `cuda_scheduler` wires it.
fn config(pages: u32) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_sequence_tokens: MAX_CONTEXT,
        serving_chunk_tokens: SERVING_CHUNK,
        kv_page_tokens: ignis_runtime::KV_PAGE_TOKENS,
        kv_capacity_pages: pages,
        ..SchedulerConfig::default()
    }
}

/// Advance `sched` until it is idle, failing on a compute error.
fn run_to_idle(sched: &mut ConcreteScheduler, events: &mut Vec<SchedEvent>) {
    while !sched.is_idle() {
        events.extend(sched.advance());
        assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
    }
}

/// Run `inputs` (submitted in order, the first ones given `lead` advances to
/// start decoding before the rest arrive) to idle; the requests' ids and events.
fn run(
    compute: &Arc<RuntimeCompute<CudaLeaf>>,
    pages: u32,
    first: Vec<RequestInput>,
    lead: usize,
    then: Vec<RequestInput>,
) -> (Vec<RequestId>, Vec<SchedEvent>) {
    let mut sched = ConcreteScheduler::with_config(config(pages), compute.clone());
    let mut ids: Vec<RequestId> =
        first.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")).collect();
    let mut events = Vec::new();
    for _ in 0..lead {
        events.extend(sched.advance());
    }
    ids.extend(then.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")));
    run_to_idle(&mut sched, &mut events);
    (ids, events)
}

/// The vision load both tests run on.
struct Loaded {
    model: Arc<Model<CudaLeaf>>,
    compute: Arc<RuntimeCompute<CudaLeaf>>,
    frontend: FrontendSet,
    pages: u32,
}

/// Load the model with vision, or `None` where the profile allows a skip.
fn load() -> Option<Loaded> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return None;
            }
            unreachable!();
        }
    };
    let config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::Bf16,
        kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(KvFormat::Bf16, MAX_CONTEXT),
        vision: Some(Vision::default()),
        ..CudaLeafConfig::default()
    };
    // GitHub #213: the eviction leg below pushes a sequence to KV-RAM, which
    // is placed in this arena. The same figure its scheduler budgets.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(4 << 30)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = Arc::new(RuntimeCompute::new(model.clone(), eos));
    // The leaf's errors are `tracing::error!` events; without a subscriber a
    // failing round only reports `Kernel(-1)`.
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    let pages = model.stats().expect("stats").kv_page_count;
    Some(Loaded { model, compute, frontend, pages })
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_image_prefill_interleaved_with_text_lanes_leaves_them_as_they_run_alone() {
    let Some(Loaded { model, compute, frontend, pages }) = load() else {
        return;
    };

    // Short, unambiguous questions: the lane's answer is checkable text.
    let questions = [
        ("What is 2 + 2? Answer with a number only.", "4"),
        ("What is the capital of Italy? Answer with one word.", "Rome"),
        ("What colour is a ripe banana? Answer with one word.", "ellow"),
    ];
    let texts = || questions.iter().map(|(q, _)| text_input(&frontend, q)).collect::<Vec<_>>();

    let (alone_ids, alone) = run(&compute, pages, texts(), 0, Vec::new());
    let image = image();
    // The text lanes start decoding first; the image prompt then prefills in
    // 64-token chunks between their decode rounds.
    let (mixed_ids, mixed) = run(&compute, pages, texts(), 6, vec![image_input(&frontend, &image)]);

    let decode = |tokens: Vec<TokenId>| frontend.tokenizer().decode(&tokens).expect("decode");
    for (n, (question, answer)) in questions.iter().enumerate() {
        let solo = decode(generated(&alone, alone_ids[n]));
        let interleaved = decode(generated(&mixed, mixed_ids[n]));
        eprintln!("text lane {n} ({question}): alone {solo:?}, interleaved {interleaved:?}");
        assert!(solo.contains(answer), "text lane {n} alone: {solo:?}");
        assert!(
            interleaved.contains(answer),
            "text lane {n} must still answer its own question while an image prefill interleaves: {interleaved:?}"
        );
    }
    let answer = generated(&mixed, mixed_ids[questions.len()]);
    assert!(!answer.is_empty(), "the image request answers");
    let text = decode(answer);
    eprintln!("image answer: {text:?}");
    assert!(text.contains("47"), "a sane answer about the image: {text:?}");
    assert_eq!(compute.live_media(), 0);

    // 100 image requests in a row: every embedding is released, and the
    // leaf's footprint does not grow.
    let before = model.stats().expect("stats").vram_bytes;
    for round in 0..100 {
        let input = RequestInput {
            params: DecodeParams { max_tokens: Some(1), ..DecodeParams::default() },
            ..image_input(&frontend, &image)
        };
        let (ids, events) = run(&compute, pages, vec![input], 0, Vec::new());
        assert_eq!(generated(&events, ids[0]).len(), 1, "round {round}");
        assert_eq!(compute.live_media(), 0, "round {round}");
        assert_eq!(compute.live_sequences(), 0, "round {round}");
    }
    assert_eq!(model.stats().expect("stats").vram_bytes, before);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_fan_out_over_one_image_encodes_it_once_and_answers_the_same() {
    // GitHub #243, acceptances 1 and 2. Four questions about one picture:
    // one embedding, four prefills. The answers must be what each question
    // gets on its own — an embedding reused is the same bytes at the same
    // columns, so nothing about the answer may move.
    let Some(Loaded { compute, frontend, pages, .. }) = load() else {
        return;
    };
    let image = image();
    let questions = [
        "What number is shown in the image?",
        "What colour are the digits in the image?",
        "What colour is the background of the image?",
        "How many digits are shown in the image?",
    ];
    let ask = |qs: &[&str]| -> (Vec<Vec<TokenId>>, std::time::Duration) {
        let inputs: Vec<_> =
            qs.iter().map(|q| image_question(&frontend, &image, q)).collect();
        let started = std::time::Instant::now();
        let (ids, events) = run(&compute, pages, inputs, 0, Vec::new());
        let elapsed = started.elapsed();
        (ids.iter().map(|&id| generated(&events, id)).collect(), elapsed)
    };
    let decode = |tokens: &[TokenId]| frontend.tokenizer().decode(tokens).expect("decode");

    // One question first, so the picture is encoded exactly once here and the
    // entry it leaves is what the fan-out must find.
    let (first, solo_time) = ask(&questions[..1]);
    assert_eq!(compute.live_media(), 0, "nothing holds it once the request is done");
    assert_eq!(compute.cached_media(), 1, "and it is still encoded");
    let solo = decode(&first[0]);
    assert!(solo.contains("47"), "the control answers about the image: {solo:?}");

    let (together, fan_out) = ask(&questions);
    assert_eq!(compute.live_media(), 0, "nothing holds the embedding afterwards");
    assert_eq!(compute.cached_media(), 1, "four questions, one embedding");
    for (n, question) in questions.iter().enumerate() {
        let answer = decode(&together[n]);
        eprintln!("fan-out lane {n} ({question}): {answer:?}");
        assert!(!together[n].is_empty(), "lane {n} answered");
    }
    assert!(
        decode(&together[0]).contains("47"),
        "the number question answers the same off a shared embedding: {:?}",
        decode(&together[0])
    );

    // Informational: this canary is a small image, so its encode is
    // milliseconds and no wall-clock claim here would mean anything.
    // Acceptance 2 is measured end to end through the server on the
    // 4096x4096 screenshot, against the 6.84 / 13.50 / 27.92 s baseline in
    // `docs/findings/2026-09-20-number-width-and-decide-e2e.md`. What this
    // test owns is the structural half: one entry, four answers.
    eprintln!(
        "one question {:.2}s; four over the same picture {:.2}s",
        solo_time.as_secs_f64(),
        fan_out.as_secs_f64()
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_image_request_evicted_to_kv_ram_continues_as_if_never_evicted() {
    let Some(Loaded { compute, frontend, pages, .. }) = load() else {
        return;
    };
    let image = image();

    // GitHub #194: one resident sequence, so an interactive text request
    // arriving while the image request decodes can only run by snapshotting
    // it to KV-RAM. Both runs decode the image request alone, before and
    // after, so its greedy tokens must not move.
    let evictable = |with_intruder: bool| -> (Vec<TokenId>, Vec<SchedEvent>, RequestId) {
        let mut sched = ConcreteScheduler::with_config(
            SchedulerConfig { resident_slot_capacity: 1, host_capacity_bytes: 4 << 30, ..config(pages) },
            compute.clone(),
        );
        // A long answer, so the intruder arrives while it is still decoding.
        let question = "Describe this image in detail: the digits, their colour, the background.";
        let input = RequestInput {
            params: DecodeParams { max_tokens: Some(48), ..DecodeParams::default() },
            ..image_question(&frontend, &image, question)
        };
        let victim = sched.submit(input, RequestClass::Agent).expect("submit");
        let mut events = Vec::new();
        while generated(&events, victim).len() < 2 {
            events.extend(sched.advance());
            assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
            assert!(!sched.is_idle(), "the image request finished before it could be evicted");
        }
        if with_intruder {
            let intruder = text_input(&frontend, "What is 2 + 2? Answer with a number only.");
            sched.submit(intruder, RequestClass::Interactive).expect("submit");
        }
        run_to_idle(&mut sched, &mut events);
        (generated(&events, victim), events, victim)
    };
    let (expected, _, _) = evictable(false);
    assert!(expected.len() > 8, "the control decoded well past the point the intruder arrives");
    let (actual, events, victim) = evictable(true);
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == victim)),
        "the image request was snapshotted: {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == victim)),
        "and restored from its blob"
    );
    assert!(!events.iter().any(|e| matches!(e, SchedEvent::Requeued { .. })), "never re-prefilled");
    eprintln!("evicted image answer: {:?}", frontend.tokenizer().decode(&actual).expect("decode"));
    assert_eq!(actual, expected, "an image request restored from KV-RAM must continue exactly");
    assert_eq!(compute.live_media(), 0);
}
