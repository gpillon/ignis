//! GitHub #212 (ADR 0030, spec `.scratch/vram-budget/specs/01-vram-budget.md`
//! §Slice 6): the vision encoder runs out of the prefill scratch instead of
//! an arena of its own, on a DFlash2 load through the production `CudaLeaf`
//! and the scheduler.
//!
//! The legs run on two loads, one after the other, one for each side of the
//! arena's `max`: at a 128-token prefill chunk the encoder's workspace is the
//! larger, at 1024 the prefill scratch is. Each load runs:
//!
//! 1. **An image request alone.**
//! 2. **A mixed load.** Three text lanes are decoding long answers when an
//!    image request arrives; its 196-token placeholder run prefills in
//!    64-token chunks, and the test counts the verify rounds that commit text
//!    tokens between those chunks.
//! 3. **Encodes between chunks.** One prompt with three images. A chunk never
//!    crosses a media boundary, so every encode after the first lands between
//!    two prefill chunks of the same request, and the first image's run spans
//!    several chunks.
//!
//! Every request's greedy tokens must equal the ones recorded on the build
//! before the change (`fixtures/vision_shared_workspace_tokens.json`, recorded
//! on efdcf58's sources with `IGNIS_RECORD_VISION_WORKSPACE_TOKENS=1`). The runs are
//! single-threaded `ConcreteScheduler::advance` loops, so the same build
//! gives the same tokens run after run; a shared arena that let an encode
//! and a prefill chunk overlap would move them.
//!
//! Over all three legs, the leaf's allocation counter shows no allocation of
//! any kind: media encode takes nothing from the device while serving.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice,
    DraftModule, FrontendSet, MessageContent, ModelScope, Reader, Role,
};
use ignis_core::gpu_profile;
use ignis_core::seq::{alloc_count, AllocCount, AllocKind};
use ignis_core::vision::Multimodal;
use ignis_core::{
    ConcreteScheduler, DecodeParams, KvFormat, RequestClass, RequestId, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, Speculation, SpeculativeBackend, TokenId, Vision,
};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const MAX_CONTEXT: u32 = 2048;
const MAX_TOKENS: u32 = 32;
/// The serving chunk: narrower than number.png's 196-token placeholder run.
const SERVING_CHUNK: u32 = 64;
const RECORD_ENV: &str = "IGNIS_RECORD_VISION_WORKSPACE_TOKENS";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("vision_shared_workspace_tokens.json")
}

fn canary_image(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("server")
        .join("tests")
        .join("fixtures")
        .join("vision_canary")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn params() -> DecodeParams {
    DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() }
}

fn options() -> ChatRenderOptions {
    ChatRenderOptions { enable_thinking: false, ..Default::default() }
}

fn text_input(frontend: &FrontendSet, question: &str, max_tokens: u32) -> RequestInput {
    let messages = [ChatMessage::text(Role::User, question)];
    let rendered = frontend.chat_template().render_with_thinking_and_tools(&messages, options(), None).expect("render");
    RequestInput {
        model: MODEL.into(),
        tokens: frontend.tokenizer().encode(&rendered).expect("tokenize"),
        params: DecodeParams { max_tokens: Some(max_tokens), ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
    }
}

/// One user turn of `parts`, an image part per entry of `images` in order.
fn multimodal_input(frontend: &FrontendSet, parts: Vec<ContentPart>, images: &[&[u8]]) -> RequestInput {
    let processor = frontend.vision_processor().expect("vision processor");
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(parts),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend.prepare_prompt(&processor, &messages, images, options(), None).expect("prepare");
    let (tokens, multimodal) = Multimodal::from_prepared(prepared);
    RequestInput {
        model: MODEL.into(),
        tokens,
        params: params(),
        multimodal: Some(Arc::new(multimodal)),
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
    }
}

fn image_question(frontend: &FrontendSet, image: &[u8], question: &str) -> RequestInput {
    multimodal_input(frontend, vec![ContentPart::Image { url: None }, ContentPart::Text(question.into())], &[image])
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

/// What one [`run`] generated.
struct Ran {
    /// Every request's generated tokens, in submission order.
    tokens: Vec<Vec<TokenId>>,
    /// Ticks after `then` arrived, before any of it emitted a token, in which
    /// a request of `first` committed one: decode rounds that ran between
    /// `then`'s prefill chunks.
    interleaved_ticks: usize,
}

/// Run `first` (given `lead` advances to start decoding) and then `then` to
/// idle.
fn run(
    compute: &Arc<RuntimeCompute<CudaLeaf>>,
    pages: u32,
    first: Vec<RequestInput>,
    lead: usize,
    then: Vec<RequestInput>,
) -> Ran {
    let config = SchedulerConfig {
        model: MODEL.into(),
        max_sequence_tokens: MAX_CONTEXT,
        serving_chunk_tokens: SERVING_CHUNK,
        kv_page_tokens: ignis_runtime::KV_PAGE_TOKENS,
        kv_capacity_pages: pages,
        ..SchedulerConfig::default()
    };
    let mut sched = ConcreteScheduler::with_config(config, compute.clone());
    let mut ids: Vec<RequestId> =
        first.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")).collect();
    let mut events = Vec::new();
    for _ in 0..lead {
        events.extend(sched.advance());
    }
    let leading = ids.len();
    ids.extend(then.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")));
    let committed = |tick: &[SchedEvent], among: &[RequestId]| {
        tick.iter().any(|event| matches!(event, SchedEvent::Token { request, .. } if among.contains(request)))
    };
    let mut later_started = ids.len() == leading;
    let mut interleaved_ticks = 0;
    while !sched.is_idle() {
        let tick = sched.advance();
        assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
        if !later_started {
            if committed(&tick, &ids[leading..]) {
                later_started = true;
            } else if committed(&tick, &ids[..leading]) {
                interleaved_ticks += 1;
            }
        }
        events.extend(tick);
    }
    assert_eq!(compute.live_media(), 0, "every media embedding released");
    Ran { tokens: ids.into_iter().map(|id| generated(&events, id)).collect(), interleaved_ticks }
}

/// The widest media item's placeholder run in `input`.
fn widest_item(input: &RequestInput) -> usize {
    let multimodal = input.multimodal.as_ref().expect("multimodal");
    multimodal.media.iter().map(|item| item.token_span.count).max().expect("a media item")
}

struct Loaded {
    compute: Arc<RuntimeCompute<CudaLeaf>>,
    frontend: FrontendSet,
    pages: u32,
}

/// The vision + DFlash2 load at `prefill_chunk_tokens`, or `None` where the
/// profile allows a skip.
fn load(prefill_chunk_tokens: u32) -> Option<Loaded> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let eos = frontend.eos_token_id().expect("eos");
    let scope = ModelScope { draft: Some(DraftModule::Dflash2), vision: true };
    let (plan, handles) = bind_model_scope_27b_with(&reader, scope).unwrap_or_else(|e| panic!("bind: {e}"));
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
        prefill_chunk_tokens,
        speculation: Some(Speculation::new(SpeculativeBackend::Dflash2, 7).expect("dflash2-7")),
        vision: Some(Vision::default()),
        ..CudaLeafConfig::default()
    };
    // GitHub #213: KV-RAM is one arena pinned at the load, so a leaf without
    // one refuses every spill. A gibibyte costs this load nothing it notices
    // and keeps a leg that starts spilling from failing for the wrong reason.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config)
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = Arc::new(RuntimeCompute::new(model.clone(), eos));
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    let stats = model.stats().expect("stats");
    eprintln!("reserved beside the weights: {:#?}", stats.reserved);
    Some(Loaded { compute, frontend, pages: stats.kv_page_count })
}

/// Every leg on one load, keyed `{shape}/{leg}`, each request's tokens in
/// submission order.
fn legs(loaded: &Loaded, shape: &str) -> BTreeMap<String, Vec<Vec<TokenId>>> {
    let Loaded { compute, frontend, pages } = loaded;
    let (compute, frontend, pages) = (compute, frontend, *pages);
    let number = canary_image("number.png");
    let colour = canary_image("colour.png");
    let circles = canary_image("circles.png");
    let before = AllocKind::ALL.map(alloc_count);

    let mut runs: BTreeMap<String, Vec<Vec<TokenId>>> = BTreeMap::new();
    let alone = run(compute, pages, vec![image_question(&frontend, &number, "What number is shown in the image?")], 0, Vec::new());
    runs.insert(format!("{shape}/image_alone"), alone.tokens);

    let texts = [
        "Write a paragraph about the history of Rome.",
        "Explain how a bicycle gear works.",
        "Describe the water cycle step by step.",
    ]
    .map(|q| text_input(&frontend, q, 128));
    // The text lanes are prefilled and decoding after one tick; the image
    // prompt then prefills in 64-token chunks between their verify rounds.
    let picture = image_question(&frontend, &number, "Describe this image in detail.");
    assert!(widest_item(&picture) > SERVING_CHUNK as usize, "the image's run spans several chunks");
    let mixed = run(compute, pages, texts.to_vec(), 1, vec![picture]);
    eprintln!("mixed: {} verify rounds committed text between the image's prefill chunks", mixed.interleaved_ticks);
    assert!(mixed.interleaved_ticks >= 2, "the image prefill interleaved with decoding text lanes");
    for (n, tokens) in mixed.tokens[..texts.len()].iter().enumerate() {
        assert!(tokens.len() > 32, "text lane {n} was still decoding when the image arrived: {} tokens", tokens.len());
    }
    runs.insert(format!("{shape}/mixed"), mixed.tokens);

    let parts = vec![
        ContentPart::Text("First image:".into()),
        ContentPart::Image { url: None },
        ContentPart::Text("Second image:".into()),
        ContentPart::Image { url: None },
        ContentPart::Text("Third image:".into()),
        ContentPart::Image { url: None },
        ContentPart::Text("Describe each of the three images in one short sentence.".into()),
    ];
    let three = multimodal_input(&frontend, parts, &[&number, &colour, &circles]);
    let items = three.multimodal.as_ref().expect("multimodal").media.len();
    assert_eq!(items, 3, "three media items");
    assert!(widest_item(&three) > SERVING_CHUNK as usize, "an image's run spans several chunks");
    runs.insert(format!("{shape}/encodes_between_chunks"), run(compute, pages, vec![three], 0, Vec::new()).tokens);

    let counts: Vec<(AllocKind, AllocCount)> =
        AllocKind::ALL.iter().zip(&before).map(|(&kind, earlier)| (kind, alloc_count(kind).since(earlier))).collect();
    eprintln!("{shape}: allocation counts over the legs: {counts:#?}");
    for (kind, count) in &counts {
        assert_eq!(*count, AllocCount::default(), "{shape}: {kind:?} allocated while serving images");
    }

    let decode = |tokens: &[TokenId]| frontend.tokenizer().decode(tokens).expect("decode");
    for (leg, requests) in &runs {
        for (n, tokens) in requests.iter().enumerate() {
            assert!(!tokens.is_empty(), "{leg} request {n} generated nothing");
            eprintln!("{leg} request {n}: {:?}", decode(tokens));
        }
    }
    runs
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn media_encode_out_of_the_prefill_scratch_generates_the_recorded_tokens() {
    // Both sides of the `max`, one load at a time: at a 128-token chunk the
    // encoder's workspace is the larger, so prefill runs in an arena sized
    // for the encoder; at 1024 the prefill scratch is, and the encode runs in
    // bytes a prefill chunk sized.
    let mut runs = BTreeMap::new();
    for (prefill_chunk_tokens, shape) in [(128, "encoder_sized"), (1024, "prefill_sized")] {
        let Some(loaded) = load(prefill_chunk_tokens) else {
            return;
        };
        runs.extend(legs(&loaded, shape));
    }

    let path = fixture_path();
    if std::env::var_os(RECORD_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("fixture dir")).expect("create fixture dir");
        let json = serde_json::to_string_pretty(&runs).expect("serialize");
        std::fs::write(&path, json + "\n").unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        eprintln!("recorded {}", path.display());
        return;
    }
    let recorded = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {} (record it with {RECORD_ENV}=1): {e}", path.display()));
    let recorded: BTreeMap<String, Vec<Vec<TokenId>>> = serde_json::from_str(&recorded).expect("fixture json");
    assert_eq!(runs.keys().collect::<Vec<_>>(), recorded.keys().collect::<Vec<_>>(), "the fixture's legs");
    for (leg, requests) in &runs {
        assert_eq!(requests, &recorded[leg], "{leg}: the tokens recorded before the shared workspace");
    }
}
