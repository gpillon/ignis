//! GPU test for speculation as a load option (P5-02, GitHub #150): a load
//! with `--spec dflash2` reports the drafter's VRAM — its weights plus the
//! per-lane window pool — in `ignis_program_stats`, and a load without it
//! reports today's figure.
//!
//! One upload of the drafter-bearing plan serves both loads: the text scope
//! is its first 906 handles (`bind_model_scope_27b` places the text tensors
//! first, at the offsets a text-only plan does), so the load without the
//! option sees exactly the handles it always did.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, materialize, text_scope_27b, CudaDevice, DraftModule, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_speculation;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::program_stats;
use ignis_core::{KvFormat, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 1024;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_dflash2_load_reports_the_drafters_vram_and_a_plain_load_reports_todays() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let text_len = text_scope_27b().len();
    assert_eq!(handles.len(), text_len + 66);
    let drafter_weight_bytes: u64 = plan.device_objects[text_len..].iter().map(|p| p.bytes).sum();

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
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
    // (program VRAM, bound tensors) for one load, each dropped before the next.
    let load = |handles: &[ignis_artifact::ObjectHandle], speculation: Option<Speculation>| {
        let model = load_qwen38_27b_with_speculation(
            &reader,
            &artifact,
            handles,
            128,
            MAX_CONTEXT,
            KvFormat::Bf16,
            speculation,
        )
        .unwrap_or_else(|e| panic!("ignis_model_load ({speculation:?}): {e}"));
        let pool = SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::Bf16,
                kv_page_group_count: 8,
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
            },
        )
        .unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("program stats: {e}"));
        (stats.vram_bytes, model.stats().bound_tensor_count)
    };

    // P5-04 (GitHub #153): any load with a draft window also carries the
    // verify substrate (its staging, ReplaySSM records, accept scratch and a
    // decode scratch sized for k+1 columns per lane). A verify-only load at
    // the same window carries exactly that and no drafter, so the drafter's
    // own cost is the difference between the two -- still to the byte.
    let verify_only = Speculation::new(SpeculativeBackend::VerifyOnly, 7).unwrap();
    let (plain_vram, plain_bound) = load(&handles[..text_len], None);
    let (spec_vram, spec_bound) = load(&handles, Some(spec));
    let (verify_vram, verify_bound) = load(&handles[..text_len], Some(verify_only));
    // Asked again after the speculative loads, so nothing they allocated
    // leaks into the plain load's figure.
    let (plain_again, _) = load(&handles[..text_len], None);

    assert_eq!(plain_again, plain_vram, "a load without the option reports today's figure");
    assert_eq!(spec_bound, plain_bound + 66, "every dflash2 object crosses the ABI");
    assert_eq!(verify_bound, plain_bound, "a verify-only load binds no drafter object");
    assert!(verify_vram > plain_vram, "a draft window reserves the verify substrate");
    assert_eq!(spec.window_pool_bytes(), 8 * 80 * 1024 * 1024);
    assert_eq!(
        spec_vram - verify_vram,
        drafter_weight_bytes + spec.window_pool_bytes(),
        "the drafter adds its weights and 8 lanes x 80 MiB of window pool, nothing else"
    );

    let _ = artifact.release_arena(&mut device);
}
