//! The vision residency's identity on the real model (GitHub #414, ADR 0075):
//! a load under `host`, whose tower is staged into the prefill scratch for
//! each media encode, answers exactly as a load under `device`, whose tower
//! stays in VRAM.
//!
//! One load of each, one after the other (the card holds one), each encoding
//! the vision canary's four images, a 1,024-token item and a 16,384-token item
//! (the processor's item bound, a 4096 x 4096 image), and scoring each canary
//! prompt teacher-forced over its first 32 answer positions, prefilled whole
//! and in 48-token spans. Compared between the two:
//! - every media embedding, byte for byte;
//! - every canary prompt's predictions, and the hash of every scored
//!   position's logits.
//!
//! The leaf's accounting of the two residencies is
//! `crates/core/tests/vision_residency_gpu.rs`.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

#[path = "support/mod.rs"]
mod support;

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b_with, materialize, ChatMessage, ChatRenderOptions, ContentPart, CudaDevice, FrontendSet,
    MessageContent, Reader, Role,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_options, model_scope};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step;
use ignis_core::vision::{vision_item_control, Multimodal};
use ignis_core::{ExceptionWeights, KvFormat, RopeScaling, Vision, VisionResidency};

use support::vision_canary::{argmax_lowest_id, load_canaries, prefill_prompt};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// Room for the 16,384-token item; only the canaries are prefilled.
const MAX_CONTEXT: u32 = 20_480;
const PREFILL_CHUNK: u32 = 1024;
/// The processor's item bound for this artifact, and the envelope: the
/// largest item a load encodes.
const ITEM_BOUND: u32 = 16_384;
const FIRST_N: usize = 32;
/// Narrower than every canary image's placeholder run (64..196 tokens).
const SPANNING_CHUNK: u32 = 48;

/// What one load produced: each item's embedding bits, and each canary
/// prompt's predictions and logits hashes per way of prefilling it.
struct Capture {
    embeddings: Vec<(String, Vec<u16>)>,
    scored: Vec<(String, Vec<u32>, Vec<u64>)>,
}

fn fnv64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

fn logits_hash(logits: &[f32]) -> u64 {
    fnv64(logits.iter().flat_map(|v| v.to_bits().to_le_bytes()))
}

/// A deterministic `side` x `side` RGB PNG with detail at every scale -- a
/// gradient under xorshift noise -- so no two patches encode alike.
fn patterned_png(side: u32) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut pixels = Vec::with_capacity((side * side * 3) as usize);
    for y in 0..side {
        for x in 0..side {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 56) as u32;
            pixels.push(((x * 255 / side + noise / 4) % 256) as u8);
            pixels.push(((y * 255 / side + noise / 3) % 256) as u8);
            pixels.push((((x ^ y) & 0xff) + noise / 2) as u8);
        }
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, side, side);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");
    }
    out
}

fn image_prompt(frontend: &FrontendSet, image: &[u8], question: &str) -> (Vec<i32>, Multimodal) {
    let processor = frontend.vision_processor().unwrap_or_else(|e| panic!("vision processor: {e}"));
    let messages = [ChatMessage {
        role: Role::User,
        content: MessageContent::Parts(vec![ContentPart::Image { url: None }, ContentPart::Text(question.into())]),
        tool_calls: Vec::new(),
        reasoning_content: None,
    }];
    let prepared = frontend
        .prepare_prompt(&processor, &messages, &[image], ChatRenderOptions { enable_thinking: false, ..Default::default() }, None)
        .unwrap_or_else(|e| panic!("prepare prompt: {e}"));
    let (token_ids, prompt) = Multimodal::from_prepared(prepared);
    (token_ids.iter().map(|&t| t as i32).collect(), prompt)
}

/// One load at `residency` on `device`, its encodes and its canary scores;
/// `None` when outside the profile the artifact cannot be materialized. Both
/// loads share the test's one device, so the residency is all that differs.
fn capture(
    reader: &Reader, frontend: &FrontendSet, device: &mut CudaDevice, residency: VisionResidency,
) -> Option<Capture> {
    let vision = Vision::new(ITEM_BOUND).unwrap().with_item_max_tokens(ITEM_BOUND).with_residency(residency);
    let (plan, handles) = bind_model_scope_27b_with(reader, model_scope(None, Some(vision), ExceptionWeights::Bf16))
        .unwrap_or_else(|e| panic!("bind ({residency}): {e}"));
    let mut artifact = match materialize(reader, &plan, device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize ({residency}): {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b_with_options(
        reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        None,
        Some(vision),
        RopeScaling::NONE,
    )
    .unwrap_or_else(|e| panic!("ignis_model_load ({residency}): {e}"));
    let config = ModelConfig::qwen38_27b();
    let pool = SeqPool::create(
        &config,
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT / 64 + 8,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let vocab = config.vocab as usize;
    let mut out = Capture { embeddings: Vec::new(), scored: Vec::new() };

    for canary in load_canaries(frontend) {
        let (tokens, prompt) = image_prompt(frontend, &canary.image, &canary.question);
        let item = &prompt.media[0];
        let embedding = step::encode_media(&model, item.grid, &item.patches, &vision_item_control(item.grid))
            .unwrap_or_else(|(rc, e)| panic!("{} ({residency}): encode {rc}: {e}", canary.id));
        let bits = embedding.read_bits(config.hidden).unwrap_or_else(|e| panic!("{}: read: {e}", canary.id));
        out.embeddings.push((format!("canary {}", canary.id), bits));
        for (mode, chunk) in [("whole", PREFILL_CHUNK), ("spanning", SPANNING_CHUNK)] {
            let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let mut logits = vec![0f32; vocab];
            prefill_prompt(&model, &pool, &mut sequence, &tokens, &prompt, &embedding, chunk, &mut logits)
                .unwrap_or_else(|e| panic!("{} {mode} ({residency}): prefill: {e}", canary.id));
            let mut position = tokens.len() as u64;
            let (mut predictions, mut hashes) = (Vec::new(), Vec::new());
            for i in 0..FIRST_N.min(canary.expected.len()) {
                predictions.push(argmax_lowest_id(&logits));
                hashes.push(logits_hash(&logits));
                let forced = [canary.expected[i] as i32];
                step::prefill_program(&model, &pool, &mut sequence, &forced, position, Some(&mut logits))
                    .unwrap_or_else(|e| panic!("{} {mode} ({residency}): forced {i}: {e}", canary.id));
                position += 1;
            }
            let agree = predictions.iter().zip(&canary.expected).filter(|(a, b)| a == b).count();
            eprintln!("{residency}: canary {} {mode}: {agree}/{} agree with the reference", canary.id, predictions.len());
            out.scored.push((format!("canary {} {mode}", canary.id), predictions, hashes));
        }
    }

    for (side, merged) in [(1024u32, 1024usize), (4096, ITEM_BOUND as usize)] {
        let image = patterned_png(side);
        let (_, prompt) = image_prompt(frontend, &image, "Describe this image in one sentence.");
        let item = &prompt.media[0];
        assert_eq!(item.token_span.count, merged, "a {side} x {side} image is a {merged}-token item");
        let started = std::time::Instant::now();
        let embedding = step::encode_media(&model, item.grid, &item.patches, &vision_item_control(item.grid))
            .unwrap_or_else(|(rc, e)| panic!("{merged}-token item ({residency}): encode {rc}: {e}"));
        eprintln!("{residency}: a {merged}-token item encoded in {:.1} ms (one cold encode)", started.elapsed().as_secs_f64() * 1e3);
        let bits = embedding.read_bits(config.hidden).unwrap_or_else(|e| panic!("{merged}-token item: read: {e}"));
        out.embeddings.push((format!("{merged}-token item"), bits));
    }

    drop(pool);
    drop(model);
    let _ = artifact.release_arena(device);
    Some(out)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_host_load_encodes_byte_identical_embeddings_and_scores_identical_logits() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let mut cuda = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let Some(device) = capture(&reader, &frontend, &mut cuda, VisionResidency::Device) else {
        return;
    };
    let Some(host) = capture(&reader, &frontend, &mut cuda, VisionResidency::Host) else {
        return;
    };

    assert_eq!(host.embeddings.len(), device.embeddings.len());
    let mut differing = Vec::new();
    for ((name, d), (_, h)) in device.embeddings.iter().zip(&host.embeddings) {
        assert_eq!(d.len(), h.len(), "{name}");
        let values = d.iter().zip(h).filter(|(a, b)| a != b).count();
        eprintln!("{name}: {} BF16 values, {values} differ", d.len());
        if values > 0 {
            differing.push(format!("{name}: {values} of {}", d.len()));
        }
    }
    assert!(differing.is_empty(), "embeddings differ under host: {differing:?}");

    assert_eq!(host.scored.len(), device.scored.len());
    for ((name, d_predictions, d_hashes), (_, h_predictions, h_hashes)) in device.scored.iter().zip(&host.scored) {
        assert_eq!(h_predictions, d_predictions, "{name}: predictions");
        assert_eq!(h_hashes, d_hashes, "{name}: every scored position's logits");
    }
}
