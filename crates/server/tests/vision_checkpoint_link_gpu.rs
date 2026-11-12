//! GitHub #249 (ADR 0029, coordinator decision 2026-10-31) — a prompt
//! checkpoint captured over a **pages-only link that cuts an image**, on the
//! real model: the production `CudaLeaf`, the scheduler, and the artifact
//! template provider's own multimodal render.
//!
//! An image request whose picture spans its generation opener's page floor has
//! no prefix to publish there (#193: none ends inside an image), so its capture
//! lends the whole pages below the opener to a pages-only link -- one that ends
//! inside the picture -- and the checkpoint stands on it. A claimant then
//! shares those pages and copies the tail page, image placeholders included.
//!
//! The claim is ADR 0029's: bit-exact against a cold prefill **split at the
//! same boundaries**. The serving chunk is set to the first turn's opener, so
//! a cold prefill with prompt reuse off is cut where the claimant's history
//! was, and the next turn ends at its own opener, so the claimant cuts
//! nothing past it either. Three claimants, each against its control:
//!
//! - the identical repeat of the first turn, against the first turn itself;
//! - the next turn, standing on the link on the device;
//! - the next turn again after the checkpoint was spilled to KV-RAM and comes
//!   back from there.
//!
//! An unsplit cold prefill of the next turn is printed beside them: where it
//! parts company is the chunking effect, recorded and not asserted.
//!
//! The second test is the common next turn: an answer long enough that the
//! next turn's opener's page floor lies past the picture. That turn takes no
//! link of its own. Standing on the first turn's checkpoint, it publishes the
//! pages it warmed as a chained prefix whose parent is the image-cutting link,
//! and captures at its opener over that; the turn after it stands on both.
//! Its serving chunk is half a KV page and every opener the conversation is
//! cut at is a multiple of it, so a cold prefill with prompt reuse off, cut
//! every chunk, is cut where each claimant's whole history was (asserted from
//! the chunks themselves), and each claimant is held to it token for token.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::vision::VisionProcessor;
use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader, VisionResidency};
use ignis_core::checkpoint::{RetainedKind, RetainedStateOperation, ReuseSource};
use ignis_core::gpu_profile;
use ignis_core::{
    model_load, ConcreteScheduler, DecodeParams, ExceptionWeights, KvFormat, RequestClass, RequestInput, SchedEvent,
    Scheduler, SchedulerConfig, TokenId, Vision,
};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute};
use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::template::{ChatMessage, TemplateProvider};
use ignis_server::thinking::ThinkingOptions;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const MAX_CONTEXT: u32 = 2048;
/// The load's prefill width: the serving chunk below must fit in it.
const LOAD_CHUNK: u32 = 2048;
const MAX_TOKENS: u32 = 24;
/// KV-RAM, in the leaf and in the scheduler's ledger: room for the one
/// checkpoint this spills, many times over.
const KV_RAM_BYTES: u64 = 1024 * 1024 * 1024;

const QUESTION: &str = "Name the two colours.";
const ANSWER: &str = "Red and blue.";
const NEXT_QUESTION: &str = "Which colour is on the left?";

/// The second test's serving chunk: half a KV page, a multiple of which every
/// opener its conversation is cut at is.
const HALF_PAGE: u32 = 32;
/// Its retained slots: the first turn's checkpoint, the next turn's chained
/// prefix and its checkpoint at once, with one to spare.
const CHAIN_SLOTS: u32 = 4;
/// Its first question, whose opener lands at 1,184 tokens: a multiple of
/// [`HALF_PAGE`], 32 past the page floor the picture spans.
const CHAIN_QUESTION: &str =
    "Name the two colours in this picture, the one on the left first and then the one on the right.";
/// The next turn's long answer, about a hundred tokens, putting the next
/// opener at 1,312: a multiple of [`HALF_PAGE`], its page floor past the picture.
const LONG_ANSWER: &str = "The picture is split down the middle into two flat halves with a sharp vertical edge \
    between them. The left half is a strong, warm red, close to a pure red with only a little blue in it, and the \
    right half is a deep, saturated blue with a touch of green. There is no gradient, no texture and no other shape \
    anywhere in the image: just the two solid colours side by side, each filling exactly half of the frame from the \
    top edge to the bottom edge. Nothing else is drawn on it, not even a border.";
const THIRD_ANSWER: &str = "Red.";
const THIRD_QUESTION: &str = "And on the right?";

/// A `width` x `height` PNG, red on its left half and blue on its right:
/// 1,152 vision tokens at 1152 x 1024, enough that a KV-RAM restore of the
/// state past it clears the restore floor.
fn halves(width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let pixels: Vec<u8> = (0..height)
            .flat_map(|_| (0..width).flat_map(|x| if x < width / 2 { [220, 30, 30] } else { [30, 60, 220] }))
            .collect();
        writer.write_image_data(&pixels).unwrap();
    }
    out
}

fn request(
    provider: &ArtifactTemplateProvider,
    processor: &VisionProcessor,
    image: &[u8],
    messages: serde_json::Value,
) -> RequestInput {
    let messages: Vec<ChatMessage> = serde_json::from_value(messages).expect("messages");
    let media = processor.prepare_media(0, image).expect("prepare media");
    let options = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let (rendered, multimodal) =
        provider.prepare_multimodal(&messages, &options, &[], vec![media]).expect("prepare prompt");
    RequestInput {
        decision: None,
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
        model: MODEL.into(),
        tokens: rendered.tokens,
        params: DecodeParams { max_tokens: Some(MAX_TOKENS), ..DecodeParams::default() },
        multimodal: Some(Arc::new(multimodal)),
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        reuse_boundaries: Vec::new(),
    }
}

/// A user message carrying the picture and `question`.
fn with_the_picture(question: &str) -> serde_json::Value {
    serde_json::json!({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        {"type": "text", "text": question}
    ]})
}

/// A text-only request, rendered the same way.
fn text_request(provider: &ArtifactTemplateProvider, content: &str) -> RequestInput {
    let messages: Vec<ChatMessage> =
        serde_json::from_value(serde_json::json!([{"role": "user", "content": content}])).expect("messages");
    let options = ThinkingOptions { enable_thinking: false, ..ThinkingOptions::default() };
    let rendered = provider.apply_chat_template(&messages, &options, &[]).expect("render");
    RequestInput {
        decision: None,
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
        model: MODEL.into(),
        tokens: rendered.tokens,
        params: DecodeParams { max_tokens: Some(4), ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: rendered.opener_tokens,
        user_turn_tokens: rendered.user_turn_tokens,
        system_block_tokens: rendered.system_block_tokens,
        reuse_boundaries: Vec::new(),
    }
}

/// `input` cut at its generation opener: the prompt hands over to the model
/// there, so a claimant prefills one span and captures nothing past it.
fn ending_at_its_opener(mut input: RequestInput) -> RequestInput {
    let opener = input.opener_tokens.expect("an opener") as usize;
    input.tokens.truncate(opener);
    let multimodal = Arc::make_mut(input.multimodal.as_mut().expect("multimodal"));
    let axis = multimodal.positions.len() / 3;
    let positions: Vec<i32> = (0..3)
        .flat_map(|a| multimodal.positions[a * axis..a * axis + opener].iter().copied())
        .collect();
    multimodal.positions = positions;
    input
}

/// What one request generated, what it resumed from, and where each of its
/// prefill chunks ended.
struct Served {
    tokens: Vec<TokenId>,
    reused: Vec<(ReuseSource, u32)>,
    retained: Vec<(RetainedStateOperation, ReuseSource, RetainedKind)>,
    ends: Vec<u32>,
}

impl Served {
    /// Where its chunks ended, up to `at`.
    fn ends_to(&self, at: u32) -> impl Iterator<Item = u32> + '_ {
        self.ends.iter().copied().filter(move |&end| end <= at)
    }
}

/// Submit `input` on `sched` and run it to completion alone.
fn serve(sched: &mut ConcreteScheduler, input: RequestInput) -> Served {
    let id = sched.submit(input, RequestClass::Interactive).expect("submit");
    let mut served = Served { tokens: Vec::new(), reused: Vec::new(), retained: Vec::new(), ends: Vec::new() };
    while !sched.is_idle() {
        let events = sched.advance();
        assert!(sched.last_error().is_none(), "compute error: {:?}", sched.last_error());
        for event in events {
            match event {
                SchedEvent::Token { request, token } if request == id => served.tokens.push(token),
                SchedEvent::StateReused { request, source, tokens, .. } if request == id => {
                    served.reused.push((source, tokens))
                }
                SchedEvent::RetainedState { operation, source, kind } => served.retained.push((operation, source, kind)),
                SchedEvent::PrefillChunk { request, prefilled_tokens, .. } if request == id => {
                    served.ends.push(prefilled_tokens)
                }
                _ => {}
            }
        }
    }
    served
}

/// The artifact, opened: its multimodal template provider, its vision
/// processor, its EOS and a tokenizer to print answers with. `None` is a skip.
struct Artifact {
    reader: Reader,
    provider: ArtifactTemplateProvider,
    processor: VisionProcessor,
    eos: TokenId,
    tokenizer: FrontendSet,
}

fn open() -> Option<Artifact> {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let processor = frontend.vision_processor().expect("vision processor");
    let eos = frontend.eos_token_id().expect("eos");
    let tokenizer = FrontendSet::from_reader(&reader).expect("frontend");
    let provider = ArtifactTemplateProvider::new(frontend).with_vision(processor.clone());
    Some(Artifact { reader, provider, processor, eos, tokenizer })
}

/// The 27B on the device, its vision tower resident, behind the scheduler's
/// compute seam: hq-e8-2b, the serving format, whose residual window rides
/// every checkpoint image and KV-RAM blob (GitHub #257), with `slots`
/// retained slots in host memory and a KV-RAM arena. Answers the compute and
/// the KV pool's pages; `None` is a skip.
fn load(reader: Reader, eos: TokenId, slots: u32) -> Option<(Arc<RuntimeCompute<CudaLeaf>>, u32)> {
    let vision = Vision::default().with_residency(VisionResidency::Device);
    let (plan, handles) = bind_model_scope_27b_with(&reader, model_load::model_scope(None, Some(vision), ExceptionWeights::Bf16))
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
    let leaf_config = CudaLeafConfig {
        max_context_tokens: MAX_CONTEXT,
        kv_format: KvFormat::HqE8_2b,
        kv_pool_bytes: ignis_runtime::auto_kv_pool_bytes(KvFormat::HqE8_2b, MAX_CONTEXT),
        prefill_chunk_tokens: LOAD_CHUNK,
        retained_device_slots: 0,
        retained_host_slots: slots,
        vision: Some(vision),
        ..CudaLeafConfig::default()
    };
    let leaf = CudaLeaf::new(device, reader, artifact, handles, leaf_config)
        .with_kv_ram_arena(KV_RAM_BYTES)
        .unwrap_or_else(|e| panic!("pin KV-RAM: {e}"));
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let kv_pages = model.stats().expect("stats").kv_page_count;
    Some((Arc::new(RuntimeCompute::new(model, eos)), kv_pages))
}

/// A scheduler's config over that load: its pool, KV-RAM and `slots`
/// retained slots, prompt reuse on or off, and a serving chunk of `chunk`.
fn scheduler_config(kv_pages: u32, slots: u32, prompt_reuse: bool, chunk: u32) -> SchedulerConfig {
    SchedulerConfig {
        model: MODEL.into(),
        max_sequence_tokens: MAX_CONTEXT,
        serving_chunk_tokens: chunk,
        kv_page_tokens: ignis_runtime::KV_PAGE_TOKENS,
        kv_capacity_pages: kv_pages,
        host_capacity_bytes: KV_RAM_BYTES,
        prompt_reuse,
        retained_slots: slots,
        ..SchedulerConfig::default()
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_claimant_standing_on_a_link_that_cuts_an_image_matches_a_split_cold_prefill() {
    let Some(artifact) = open() else { return };
    let image = halves(1152, 1024);
    let turn = request(&artifact.provider, &artifact.processor, &image, serde_json::json!([with_the_picture(QUESTION)]));
    let next = ending_at_its_opener(request(
        &artifact.provider,
        &artifact.processor,
        &image,
        serde_json::json!([
            with_the_picture(QUESTION),
            {"role": "assistant", "content": ANSWER},
            {"role": "user", "content": NEXT_QUESTION}
        ]),
    ));

    // The geometry this rests on, said rather than hoped for.
    let page = ignis_runtime::KV_PAGE_TOKENS;
    let item = turn.multimodal.as_ref().unwrap().media[0].token_span;
    let (opener, next_opener) = (turn.opener_tokens.unwrap(), next.opener_tokens.unwrap());
    let floor = opener / page * page;
    eprintln!(
        "turn {} tokens, image {}..{}, opener {opener} (floor {floor}); next turn {} tokens, opener {next_opener}",
        turn.tokens.len(),
        item.begin,
        item.begin + item.count,
        next.tokens.len()
    );
    assert!(
        (item.begin as u32) < floor && floor < (item.begin + item.count) as u32,
        "the image spans the opener's page floor"
    );
    assert!(
        next_opener / page * page < (item.begin + item.count) as u32,
        "and the next turn's: it publishes nothing a cold prefill would not cut at"
    );
    assert_eq!(next.tokens[..opener as usize], turn.tokens[..opener as usize], "the next turn extends the first to its opener");
    assert!(opener >= ignis_core::checkpoint::KV_RAM_RESTORE_FLOOR_TOKENS, "a KV-RAM restore of it clears the floor");
    assert!(opener <= LOAD_CHUNK);

    // One retained slot, so the next capture gives the checkpoint up to KV-RAM.
    let Some((compute, kv_pages)) = load(artifact.reader, artifact.eos, 1) else { return };
    let config = |prompt_reuse: bool, chunk: u32| scheduler_config(kv_pages, 1, prompt_reuse, chunk);
    let text = |tokens: &[TokenId]| artifact.tokenizer.tokenizer().decode(tokens).expect("decode");

    // The controls, with prompt reuse off: nothing captured, nothing left in
    // the leaf for the run after them. Cut at the first turn's opener, as the
    // claimant's history was, and then unsplit.
    let split = serve(&mut ConcreteScheduler::with_config(config(false, opener), compute.clone()), next.clone());
    let unsplit = serve(&mut ConcreteScheduler::with_config(config(false, LOAD_CHUNK), compute.clone()), next.clone());
    eprintln!("next turn, cold, split at {opener}: {:?}", text(&split.tokens));
    eprintln!(
        "next turn, cold, unsplit: {:?} (first parts company at {:?}; the chunking effect, not asserted)",
        text(&unsplit.tokens),
        split.tokens.iter().zip(&unsplit.tokens).position(|(a, b)| a != b)
    );
    assert!(split.reused.is_empty() && unsplit.reused.is_empty());

    let mut sched = ConcreteScheduler::with_config(config(true, opener), compute.clone());
    let first = serve(&mut sched, turn.clone());
    eprintln!("first turn: {:?}", text(&first.tokens));
    assert!(first.reused.is_empty(), "the first turn is cold");
    assert!(text(&first.tokens).to_lowercase().contains("red"), "{:?}", text(&first.tokens));

    // The identical repeat is split where the first turn was: at its opener.
    let repeat = serve(&mut sched, turn.clone());
    assert_eq!(repeat.reused, [(ReuseSource::Device, opener)], "the repeat claims the checkpoint");
    assert_eq!(repeat.tokens, first.tokens, "and generates what the first turn did");

    let on_device = serve(&mut sched, next.clone());
    eprintln!("next turn, from the checkpoint on the device: {:?}", text(&on_device.tokens));
    assert_eq!(on_device.reused, [(ReuseSource::Device, opener)], "the next turn claims the checkpoint");
    assert_eq!(on_device.tokens, split.tokens, "and generates what the split cold prefill does");

    // A text prompt's publish takes the one slot: the checkpoint is given up
    // to KV-RAM, and the next turn comes back from there.
    let filler = text_request(&artifact.provider, &"Write one sentence about the sea. ".repeat(24));
    let spilled = serve(&mut sched, filler);
    assert!(
        spilled.retained.contains(&(RetainedStateOperation::Spill, ReuseSource::KvRam, RetainedKind::Checkpoint)),
        "the checkpoint spilled to KV-RAM: {:?}",
        spilled.retained
    );
    let from_kv_ram = serve(&mut sched, next.clone());
    eprintln!("next turn, from the checkpoint in KV-RAM: {:?}", text(&from_kv_ram.tokens));
    assert_eq!(from_kv_ram.reused, [(ReuseSource::KvRam, opener)], "the next turn restores the checkpoint");
    assert_eq!(from_kv_ram.tokens, split.tokens, "and still generates what the split cold prefill does");
    assert_eq!(compute.live_media(), 0);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn claimants_standing_on_a_prefix_chained_over_the_link_match_a_split_cold_prefill() {
    let Some(artifact) = open() else { return };
    let image = halves(1152, 1024);
    let render = |messages: serde_json::Value| request(&artifact.provider, &artifact.processor, &image, messages);
    let turn = render(serde_json::json!([with_the_picture(CHAIN_QUESTION)]));
    // The next turn as the server renders it, four tokens past its opener with
    // thinking off, so it captures there.
    let next = render(serde_json::json!([
        with_the_picture(CHAIN_QUESTION),
        {"role": "assistant", "content": LONG_ANSWER},
        {"role": "user", "content": NEXT_QUESTION}
    ]));
    let third = ending_at_its_opener(render(serde_json::json!([
        with_the_picture(CHAIN_QUESTION),
        {"role": "assistant", "content": LONG_ANSWER},
        {"role": "user", "content": NEXT_QUESTION},
        {"role": "assistant", "content": THIRD_ANSWER},
        {"role": "user", "content": THIRD_QUESTION}
    ])));

    // The geometry this rests on, said rather than hoped for.
    let page = ignis_runtime::KV_PAGE_TOKENS;
    let item = turn.multimodal.as_ref().unwrap().media[0].token_span;
    let image_end = (item.begin + item.count) as u32;
    let (opener, next_opener) = (turn.opener_tokens.unwrap(), next.opener_tokens.unwrap());
    let next_floor = next_opener / page * page;
    eprintln!(
        "turn {} tokens, image {}..{image_end}, opener {opener}; next turn {} tokens, opener {next_opener} (floor \
         {next_floor}); third turn {} tokens",
        turn.tokens.len(),
        item.begin,
        next.tokens.len(),
        third.tokens.len()
    );
    assert!(
        (item.begin as u32) < opener / page * page && opener / page * page < image_end,
        "the image spans the first opener's page floor: its capture lends to a link that cuts the picture"
    );
    assert!(
        next_floor >= image_end && next_opener % page != 0,
        "the next turn's floor lies past it: a chained prefix there, over the link, and a capture at its opener, \
         a cut of its own"
    );
    assert_eq!(
        (opener % HALF_PAGE, next_opener % HALF_PAGE),
        (0, 0),
        "every capture is a multiple of the serving chunk, so a cold prefill cut every chunk is cut where each \
         claimant's history was"
    );
    assert_eq!(next.tokens[..opener as usize], turn.tokens[..opener as usize], "the next turn extends the first to its opener");
    assert_eq!(third.tokens[..next_opener as usize], next.tokens[..next_opener as usize], "and the third the next");

    let Some((compute, kv_pages)) = load(artifact.reader, artifact.eos, CHAIN_SLOTS) else { return };
    let config = |prompt_reuse: bool| scheduler_config(kv_pages, CHAIN_SLOTS, prompt_reuse, HALF_PAGE);
    let text = |tokens: &[TokenId]| artifact.tokenizer.tokenizer().decode(tokens).expect("decode");

    // The controls, with prompt reuse off: cold, cut every half page, and
    // nothing left in the leaf for the run after them.
    let next_control = serve(&mut ConcreteScheduler::with_config(config(false), compute.clone()), next.clone());
    let third_control = serve(&mut ConcreteScheduler::with_config(config(false), compute.clone()), third.clone());
    assert!(next_control.reused.is_empty() && third_control.reused.is_empty());
    eprintln!("next turn, cold: {:?}", text(&next_control.tokens));
    eprintln!("third turn, cold: {:?}", text(&third_control.tokens));

    let mut sched = ConcreteScheduler::with_config(config(true), compute.clone());
    let first = serve(&mut sched, turn.clone());
    eprintln!("first turn: {:?}", text(&first.tokens));
    assert!(first.reused.is_empty(), "the first turn is cold");
    assert_eq!(sched.prefix_pinned_pages(), opener / page, "its capture lent the pages below its opener to a link");

    let on_link = serve(&mut sched, next.clone());
    eprintln!("next turn, from the checkpoint on the link: {:?}", text(&on_link.tokens));
    assert_eq!(on_link.reused, [(ReuseSource::Device, opener)], "the next turn claims the first turn's checkpoint");
    assert_eq!(
        sched.prefix_pinned_pages(),
        next_floor / page,
        "and publishes the pages it warmed below its floor as a prefix chained over the link"
    );
    assert_eq!(
        first.ends_to(opener).chain(on_link.ends.iter().copied()).collect::<Vec<_>>(),
        next_control.ends,
        "its whole history is cut where the cold prefill is"
    );
    assert_eq!(on_link.tokens, next_control.tokens, "and it generates what the split cold prefill does");

    let over_chain = serve(&mut sched, third.clone());
    eprintln!("third turn, from the checkpoint over the chain: {:?}", text(&over_chain.tokens));
    assert_eq!(
        over_chain.reused,
        [(ReuseSource::Device, next_opener)],
        "the third turn claims the next turn's checkpoint, standing on the chained prefix and the link"
    );
    assert_eq!(
        first
            .ends_to(opener)
            .chain(on_link.ends_to(next_opener))
            .chain(over_chain.ends.iter().copied())
            .collect::<Vec<_>>(),
        third_control.ends,
        "its whole history is cut where the cold prefill is"
    );
    assert_eq!(over_chain.tokens, third_control.tokens, "and it generates what the split cold prefill does");
    assert_eq!(compute.live_media(), 0);
}
