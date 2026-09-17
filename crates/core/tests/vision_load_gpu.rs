//! GPU test for vision as a load option (GitHub #177): a load with a vision
//! envelope binds every `vision/*` object and reports, beside its weights,
//! what it reserved for vision — the output transient, and whatever the
//! encoder workspace grew the shared prefill scratch by (GitHub #212) — and a
//! load without it binds and reserves nothing of vision, reporting today's
//! figure. Its device footprint is that reservation plus the decode round's
//! rope staging (`DECODE_ROPE_STAGING_BYTES`, GitHub #178) and nothing else.
//!
//! One upload of the vision-bearing plan serves every load: the text scope is
//! its first handles (`bind_model_scope_27b_with` places the vision tensors
//! last), so the load without the option sees exactly the handles it always
//! did. The reservation at the default envelope is printed for the ticket.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b_with, materialize, text_scope_27b, CudaDevice, ModelScope, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_options;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::program_stats;
use ignis_core::{KvFormat, Vision, VISION_OBJECTS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

/// GitHub #178: beside the reservation, a vision load stages its decode
/// rounds' rotation positions (`position + rope_delta`) in a model-owned
/// buffer at a stable address, so the decode graphs read them at replay.
/// `ignis_program_stats` counts that buffer; `vision_reserved_bytes` does
/// not, and should not — that number is the per-item output transient plus
/// what the encoder workspace grows the prefill scratch by (GitHub #212: the
/// two share one arena), which is what the startup capacity line reports.
/// One I32 per lane over the leaf's `IGNIS_DECODE_MAX_BATCH` (8,
/// `ignis_step.h`).
///
/// #177 wrote this test before #178 added the buffer, and no GPU profile ran
/// on a branch carrying both until #195 — which is why the delta below is 32
/// bytes wider than the weights and the reservation alone.
const DECODE_ROPE_STAGING_BYTES: u64 = 4 * 8;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_vision_load_binds_the_tower_and_reserves_its_workspace_and_a_plain_load_reports_todays() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let (plan, handles) = bind_model_scope_27b_with(&reader, ModelScope { draft: None, vision: true })
        .unwrap_or_else(|e| panic!("bind with vision: {e}"));
    let text_len = text_scope_27b().len();
    assert_eq!(handles.len(), text_len + VISION_OBJECTS);
    let vision_weight_bytes: u64 = plan.device_objects[text_len..].iter().map(|p| p.bytes).sum();

    let mut device = match CudaDevice::create(0) {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + vision: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // (program VRAM, bound tensors, vision reserved) for one load at
    // `max_context`, each dropped before the next.
    let load = |handles: &[ignis_artifact::ObjectHandle], max_context: u32, vision: Option<Vision>| {
        let model = load_qwen38_27b_with_options(
            &reader,
            &artifact,
            handles,
            128,
            max_context,
            KvFormat::Bf16,
            None,
            vision,
        )
        .unwrap_or_else(|e| panic!("ignis_model_load ({vision:?}): {e}"));
        let pool = SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::Bf16,
                kv_page_group_count: 8,
                max_context_tokens: max_context,
                slot_count: 1,
                retained_slot_count: 0,
            },
        )
        .unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("program stats: {e}"));
        let model_stats = model.stats();
        (stats.vram_bytes, model_stats.bound_tensor_count, model_stats.vision_reserved_bytes)
    };

    // A small context caps the envelope, so this pair stays cheap.
    const SMALL_CONTEXT: u32 = 1024;
    let vision = Vision::default();
    let (plain_vram, plain_bound, plain_reserved) = load(&handles[..text_len], SMALL_CONTEXT, None);
    let (vision_vram, vision_bound, vision_reserved) = load(&handles, SMALL_CONTEXT, Some(vision));
    let (plain_again, _, _) = load(&handles[..text_len], SMALL_CONTEXT, None);

    assert_eq!(plain_reserved, 0, "a load without vision reserves nothing for it");
    assert_eq!(plain_again, plain_vram, "a load without the option reports today's figure");
    assert_eq!(vision_bound, plain_bound + VISION_OBJECTS as u64, "every vision object crosses the ABI");
    assert!(
        vision_reserved > vision.output_transient_bytes(SMALL_CONTEXT),
        "the reservation is the output transient plus the encoder's growth of the shared scratch"
    );
    assert_eq!(
        vision_vram - plain_vram,
        vision_weight_bytes + vision_reserved + DECODE_ROPE_STAGING_BYTES,
        "vision adds its weights, its reservation and the decode round's rope staging, nothing else"
    );

    // The default envelope, uncapped (the serving context): the number the
    // ticket records.
    const SERVING_CONTEXT: u32 = 262_144;
    let (_, _, default_reserved) = load(&handles, SERVING_CONTEXT, Some(vision));
    println!(
        "vision reservation at the default envelope ({} merged tokens): {default_reserved} bytes \
         ({:.1} MiB) + weights {vision_weight_bytes} bytes ({:.1} MiB)",
        vision.max_tokens(),
        default_reserved as f64 / (1024.0 * 1024.0),
        vision_weight_bytes as f64 / (1024.0 * 1024.0)
    );
    assert!(default_reserved > vision_reserved, "the envelope sizes the reservation");

    let _ = artifact.release_arena(&mut device);
}
