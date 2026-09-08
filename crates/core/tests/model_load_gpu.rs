//! GPU integration test for the model-load step ABI call (P1-17, GitHub
//! #53, ADR 0009): the real Qwen 3.8-27B artifact loads onto the device and
//! every expected text-scope object binds with the expected qtype / layout
//! / shape.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure** (`ignis_core::gpu_profile::skip_or_fail`).
//! Run via `scripts/gpu-profile.ps1` (stops the reference `ninfer-serve`
//! first — the RTX 5090 is exclusive, ADR 0006).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::program_stats;

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads) -- mirrors `crates/artifact/tests/real_artifact.rs`.
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn real_nvfp4full_model_load_binds_every_text_scope_object() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));

    // Bind + place every text-scope tensor (P1-17): a missing or
    // mis-shaped object is a load failure here already (ADR 0002), before
    // the leaf ever sees a descriptor.
    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
    assert_eq!(plan.device_objects.len(), 906, "the full text-scope inventory");
    assert_eq!(plan.host_objects.len(), 0, "every text-scope tensor is a device tensor");

    let device = CudaDevice::create(0);
    let mut device = match device {
        Ok(d) => d,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // The H2D upload is where a busy/contended GPU actually surfaces (a
    // cudaMalloc/cudaMemcpy failure) -- route it through the profile like
    // device creation above, not a hard panic.
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    assert!(artifact.stats().device_capacity_bytes > 0, "VRAM used is reported");

    // `ignis_model_load`'s tensor binding is pure host-side name/shape
    // matching against already-uploaded pointers, so a failure here at a
    // small, affordable chunk width (128, well under any real device's free
    // memory) is always a real descriptor-building or artifact-contract bug
    // -- never device contention -- so a hard failure is correct under and
    // outside the profile alike. (P2-01, GitHub #83, gave the load its own
    // small scratch cudaMalloc too; that failure mode is deterministic on
    // the chunk width, not on a busy GPU, and is exercised separately by
    // `an_unaffordable_prefill_chunk_fails_the_load_naming_the_shortfall`
    // below.)
    let model = load_qwen38_27b(&reader, &artifact, &handles, 128, 128)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let stats = model.stats();
    // The 247 `*_input_scale_divisor` scalars (one per NVFP4 projection,
    // `inventory::text_scope_27b_is_complete`) are materialized on the
    // device but do not cross the model-load ABI yet (G2, see
    // `crates/core/src/model_load.rs`); each is a 4-byte FP32 scalar.
    const DIVISOR_COUNT: u64 = 247;
    assert_eq!(
        stats.bound_tensor_count,
        906 - DIVISOR_COUNT,
        "every non-divisor text-scope object is bound by the leaf"
    );
    assert_eq!(
        stats.vram_bytes,
        artifact.stats().h2d_bytes - DIVISOR_COUNT * 4,
        "the leaf's VRAM accounting matches the materializer's upload total minus the divisor scalars"
    );

    drop(model);
    // This file's other GPU tests also materialize the artifact in the same
    // process (--test-threads=1 runs them serially, but in one process): a
    // `MaterializedArtifact` has no `Drop` of its own (its arena needs the
    // `Device` that produced it to release, mirrors `CudaLeaf::drop`,
    // crates/runtime/src/cuda_leaf.rs), so leaving this unreleased would
    // overcommit the device by one more ~19 GiB artifact per test.
    let _ = artifact.release_arena(&mut device);
}

/// P2-01 (GitHub #83): the program scratch reservation grows with the
/// configured prefill chunk width. Loads the same materialized artifact
/// twice (no repeat upload) with two different chunk widths and the same
/// pool geometry, so any VRAM delta is attributable to the chunk alone.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn larger_prefill_chunk_reserves_more_program_vram() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // Wide enough to bound both chunk widths compared below (P2-01, GitHub
    // #83: prefill_chunk_tokens must not exceed max_context_tokens).
    const MAX_CONTEXT: u32 = 1024;
    let vram_for = |chunk: u32| -> u64 {
        let model = load_qwen38_27b(&reader, &artifact, &handles, chunk, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("ignis_model_load (chunk={chunk}): {e}"));
        let pool = SeqPool::create(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_page_group_count: 8,
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
            },
        )
        .unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("program stats: {e}"));
        stats.vram_bytes
    };

    let small = vram_for(128);
    let large = vram_for(1024);
    assert!(
        large > small,
        "a wider prefill chunk must reserve more program scratch: {large} <= {small}"
    );
    // See the release_arena note in the first test above.
    let _ = artifact.release_arena(&mut device);
}

/// P2-01 (GitHub #83): a prefill chunk wide enough that its scratch
/// reservation cannot fit beside the already-resident weights fails the
/// *load*, never the first long prompt.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_unaffordable_prefill_chunk_fails_the_load_naming_the_shortfall() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));
    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // A multiple of 128 wide enough that a single layer's plain activation
    // buffers alone dwarf any real device's free memory beside the
    // ~19 GiB-resident 27B weights. max_context_tokens is set to match (P2-01,
    // GitHub #83: prefill_chunk_tokens must not exceed max_context_tokens) so
    // the load fails on the memory budget, not on that unrelated precondition.
    const HUGE_CHUNK: u32 = 8 * 1024 * 1024;
    let err = match load_qwen38_27b(&reader, &artifact, &handles, HUGE_CHUNK, HUGE_CHUNK) {
        Ok(_) => panic!("an unaffordable chunk width must fail the load"),
        Err(e) => e,
    };
    assert!(
        err.contains("bytes are free") || err.contains("scratch"),
        "the error should name the shortfall: {err}"
    );
    // See the release_arena note on the first test in this file.
    let _ = artifact.release_arena(&mut device);
}
