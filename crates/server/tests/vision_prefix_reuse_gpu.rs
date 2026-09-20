//! GitHub #193 — multimodal requests share prefixes under a media-aware
//! identity, on the real model: the production `CudaLeaf`, the scheduler, and
//! the artifact template provider's own multimodal render (which reports the
//! system block and the generation opener over the expanded tokens).
//!
//! Three sibling requests over one shared system prompt, each with a
//! same-size flat image: red, then blue and red again while the first is
//! still running. The token ids of the three prompts are identical — every
//! placeholder of every image is the same id — so only the images' identity
//! keeps them apart.
//!
//! - the blue sibling shares the system block and nothing past it, and still
//!   answers "blue": it was not handed the red image's pages;
//! - the second red sibling reuses past the image, and still answers "red";
//! - the reused-token counter moves.
//!
//! Twice: with prompt reuse off, where the only door is the sibling prefix
//! cache — the path that matched raw token ids before #193 — and the sibling
//! counter has to move by exactly the second red's claim, since the blue one
//! has nothing it may claim at all; then with it on, where the second red
//! resumes from the first's retained checkpoint at its generation opener. Off
//! first: a scheduler leaves its retained state behind in the leaf, and a
//! later scheduler's request ids start over.
//!
//! Each answer is also taken alone first, reusing nothing, to show the model
//! tells the two swatches apart at all — otherwise "blue" after reuse would
//! prove nothing.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, ModelScope, Reader};
use ignis_core::gpu_profile;
use ignis_core::{
    ConcreteScheduler, DecodeParams, KvFormat, RequestClass, RequestId,
    RequestInput, SchedEvent, Scheduler, SchedulerConfig, TokenId, Vision,
};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const MAX_CONTEXT: u32 = 2048;
const MAX_TOKENS: u32 = 16;
const SERVING_CHUNK: u32 = 256;

/// Longer than one 64-token KV page, so the block is a prefix of its own.
const SYSTEM: &str = "You are a meticulous visual assistant working inside an automated \
    quality-control pipeline. Every request shows you exactly one image. Look at the \
    whole image before answering, never guess about details you cannot see, and keep \
    every answer as short as the question allows. When a question asks for a colour, \
    name the single dominant colour of the image in plain English, using one common \
    word such as red, green, blue, yellow, black or white.";

/// Long enough after the image that the generation opener's page lies past
/// the image's last placeholder: the head a same-image sibling shares then
/// covers the picture itself.
const QUESTION: &str = "This image comes from a batch of flat colour swatches that a \
    printer produced during calibration. The operator needs to log the colour of each \
    swatch before the batch can be approved, and the log accepts one lowercase word \
    per swatch. Please look at the swatch shown above and tell me its colour. \
    Answer with one word only.";

/// A flat `width` x `height` PNG of one colour.
fn swatch(rgb: [u8; 3], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..width * height).flat_map(|_| rgb).collect();
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

fn request(provider: &ArtifactTemplateProvider, processor: &ignis_artifact::vision::VisionProcessor, image: &[u8]) -> RequestInput {
    let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
            {"type": "text", "text": QUESTION}
        ]}
    ]))
    .expect("messages");
    let media = processor.prepare_media(0, image).expect("prepare media");
    let options = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let (rendered, multimodal) =
        provider.prepare_multimodal(&messages, &options, &[], vec![media]).expect("prepare prompt");
    RequestInput {
        decision: None,
        program: None,
        model: MODEL.into(),
        tokens: rendered.tokens,
        params: DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() },
        multimodal: Some(Arc::new(multimodal)),
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
    }
}

struct Run {
    ids: Vec<RequestId>,
    tokens: HashMap<RequestId, Vec<TokenId>>,
    /// Leading prompt tokens each request skipped through a shared prefix.
    prefix_reused: HashMap<RequestId, u32>,
    /// Leading prompt tokens each request skipped through a checkpoint.
    state_reused: HashMap<RequestId, u32>,
    sibling_reused_tok: u64,
}

/// Submit `first`, give it `lead` advances, submit `then`, run to idle.
fn run(
    compute: &Arc<RuntimeCompute<CudaLeaf>>,
    config: SchedulerConfig,
    first: Vec<RequestInput>,
    lead: usize,
    then: Vec<RequestInput>,
) -> Run {
    let mut sched = ConcreteScheduler::with_config(config, compute.clone());
    let mut ids: Vec<RequestId> =
        first.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")).collect();
    let mut tokens: HashMap<RequestId, Vec<TokenId>> = HashMap::new();
    let mut prefix_reused: HashMap<RequestId, u32> = HashMap::new();
    let mut state_reused: HashMap<RequestId, u32> = HashMap::new();
    let mut collect = |events: Vec<SchedEvent>| {
        for event in events {
            match event {
                SchedEvent::Token { request, token } => tokens.entry(request).or_default().push(token),
                SchedEvent::PrefixReused { request, tokens: skipped, .. } => {
                    prefix_reused.insert(request, skipped);
                }
                SchedEvent::StateReused { request, tokens: skipped, .. } => {
                    state_reused.insert(request, skipped);
                }
                _ => {}
            }
        }
    };
    for _ in 0..lead {
        collect(sched.advance());
        assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
    }
    ids.extend(then.into_iter().map(|input| sched.submit(input, RequestClass::Agent).expect("submit")));
    while !sched.is_idle() {
        let events = sched.advance();
        assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
        collect(events);
    }
    Run { ids, tokens, prefix_reused, state_reused, sibling_reused_tok: sched.sibling_prefix_reused_tok() }
}

impl Run {
    /// The most `request` skipped, through either kind of reuse.
    fn reused(&self, request: RequestId) -> u32 {
        let prefix = self.prefix_reused.get(&request).copied().unwrap_or(0);
        prefix.max(self.state_reused.get(&request).copied().unwrap_or(0))
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn siblings_sending_same_size_images_share_only_what_their_images_agree_on() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().expect("vision processor");
    let eos = frontend.eos_token_id().expect("eos");
    let tokenizer_frontend = FrontendSet::from_reader(&reader).expect("frontend");
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor.clone());
    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let leaf_config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::Bf16,
        kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(KvFormat::Bf16, MAX_CONTEXT),
        vision: Some(Vision::default()),
        ..CudaLeafConfig::default()
    };
    // GitHub #213: KV-RAM is one arena pinned at the load, so a leaf without
    // one refuses every spill. A gibibyte costs this load nothing it notices
    // and keeps a leg that starts spilling from failing for the wrong reason.
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config)
        .with_kv_ram_arena(1024 * 1024 * 1024)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute = Arc::new(RuntimeCompute::new(model.clone(), eos));
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    let stats = model.stats().expect("stats");
    let config = |prompt_reuse: bool| SchedulerConfig {
        model: MODEL.into(),
        max_sequence_tokens: MAX_CONTEXT,
        serving_chunk_tokens: SERVING_CHUNK,
        kv_page_tokens: ignis_runtime::KV_PAGE_TOKENS,
        kv_capacity_pages: stats.kv_page_count,
        prompt_reuse,
        ..SchedulerConfig::default()
    };

    let (red, blue) = (swatch([220, 20, 20], 256, 256), swatch([20, 20, 220], 256, 256));
    let red_input = request(&provider, &processor, &red);
    let blue_input = request(&provider, &processor, &blue);
    assert_eq!(red_input.tokens, blue_input.tokens, "same-size images: identical token ids");
    let item = red_input.multimodal.as_ref().unwrap().media[0].token_span;
    let block = red_input.system_block_tokens.expect("the render reports its system block");
    let opener = red_input.opener_tokens.expect("and its generation opener");
    let page = ignis_runtime::KV_PAGE_TOKENS;
    eprintln!(
        "prompt {} tokens, block {block}, image {}..{}, opener {opener}",
        red_input.tokens.len(),
        item.begin,
        item.begin + item.count
    );
    assert!(block >= page, "the block is at least a page");
    assert!(
        (opener / page * page) as usize >= item.begin + item.count,
        "the opener's page is past the image, so a same-image sibling can share it"
    );

    let text = |tokens: &[TokenId]| tokenizer_frontend.tokenizer().decode(tokens).expect("decode");
    // Each answer alone, reusing nothing.
    let alone = run(&compute, config(false), vec![red_input.clone(), blue_input.clone()], 0, Vec::new());
    let (red_alone, blue_alone) = (&alone.tokens[&alone.ids[0]], &alone.tokens[&alone.ids[1]]);
    eprintln!("alone: red {:?}, blue {:?}", text(red_alone), text(blue_alone));
    assert!(
        alone.prefix_reused.is_empty() && alone.state_reused.is_empty(),
        "two requests in one batch have nothing warm to claim"
    );
    assert!(text(red_alone).to_lowercase().contains("red"), "{:?}", text(red_alone));
    assert!(text(blue_alone).to_lowercase().contains("blue"), "{:?}", text(blue_alone));

    for prompt_reuse in [false, true] {
        // The siblings: red publishes, then blue and red again arrive. With
        // prompt reuse off nothing outlives the first red, and a one-word
        // answer is a single decode run, so they arrive the advance after its
        // first chunk published the whole-page head. With it on, the head is
        // cut at the block, then the opener's page, then the checkpoint is
        // captured at the opener: three chunks, and whatever is retained
        // stays for them.
        let lead = if prompt_reuse { 3 } else { 1 };
        let siblings = run(
            &compute,
            config(prompt_reuse),
            vec![red_input.clone()],
            lead,
            vec![blue_input.clone(), red_input.clone()],
        );
        let (first, other, same) = (siblings.ids[0], siblings.ids[1], siblings.ids[2]);
        let answer = |id: RequestId| text(&siblings.tokens[&id]);
        eprintln!(
            "siblings (prompt reuse {prompt_reuse}): red {:?}, blue {:?} (reused {}), red again {:?} \
             (prefix {:?}, checkpoint {:?}); sibling counter {}",
            answer(first),
            answer(other),
            siblings.reused(other),
            answer(same),
            siblings.prefix_reused.get(&same),
            siblings.state_reused.get(&same),
            siblings.sibling_reused_tok
        );
        let (blue_reused, same_reused) = (siblings.reused(other), siblings.reused(same));
        assert!(
            blue_reused as usize <= item.begin,
            "the blue sibling shares nothing of the red image: reused {blue_reused}, image begins at {}",
            item.begin
        );
        assert!(
            same_reused as usize >= item.begin + item.count,
            "the red sibling reuses past its image: reused {same_reused}"
        );
        if prompt_reuse {
            assert!(blue_reused > 0, "the blue sibling shares the system block");
        } else {
            // No block, no checkpoint: the whole-prompt head covers the image,
            // so the blue sibling claims nothing, and the counter is the red
            // sibling's claim alone.
            assert_eq!(blue_reused, 0);
            assert_eq!(siblings.prefix_reused.get(&same).copied(), Some(same_reused));
            assert_eq!(siblings.sibling_reused_tok, u64::from(same_reused), "the counter moves by it");
        }
        assert!(siblings.sibling_reused_tok > 0, "the reused-token counter moves");
        assert!(answer(other).to_lowercase().contains("blue"), "{:?}", answer(other));
        assert!(answer(same).to_lowercase().contains("red"), "{:?}", answer(same));
        assert!(answer(first).to_lowercase().contains("red"), "{:?}", answer(first));
        assert_eq!(compute.live_media(), 0);
    }
}
