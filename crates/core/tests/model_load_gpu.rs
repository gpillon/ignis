//! GPU integration test for the model-load step ABI call (P1-17, GitHub
//! #53, ADR 0009): the real Qwen 3.8-27B artifact loads onto the device and
//! every expected text-scope object binds with the expected qtype / layout
//! / shape, plus P2-01's two program-scratch properties (GitHub #83).
//!
//! One `#[test]`, one `CudaDevice`, one `materialize()`, several
//! `load_qwen38_27b` calls over it. Until GitHub #135 this file held three
//! `#[test]` fns, each creating its own device and paying its own full
//! ~19 GiB upload -- correct (each released its arena explicitly, since a
//! `MaterializedArtifact` has no `Drop` of its own) but three uploads where
//! one will do. The weights are immutable once uploaded, so sharing them
//! cannot carry state from one property to the next; what genuinely differs
//! per property is the model handle's own scratch reservation, and those are
//! still built and dropped one at a time.
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

/// The 247 `*_input_scale_divisor` scalars (one per NVFP4 projection,
/// `inventory::text_scope_27b_is_complete`) are materialized on the device
/// but do not cross the model-load ABI yet (G2, see
/// `crates/core/src/model_load.rs`); each is a 4-byte FP32 scalar.
const DIVISOR_COUNT: u64 = 247;

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_real_artifact_loads_and_its_program_scratch_tracks_the_prefill_chunk() {
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

    let mut device = match CudaDevice::create(0) {
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

    // GitHub #135: what one materialize() costs. Every GPU-gated test binary
    // in the profile pays this once, so it is the figure that says whether
    // sharing the upload across the whole stage would be worth its risk.
    // `upload_seconds` has been recorded by the materializer all along and
    // read by nothing.
    println!(
        "ignis.profile.materialize: upload_seconds={:.3} h2d_bytes={} device_capacity_bytes={}",
        artifact.stats().upload_seconds,
        artifact.stats().h2d_bytes,
        artifact.stats().device_capacity_bytes,
    );

    // --- P1-17: every text-scope object binds -------------------------------
    //
    // `ignis_model_load`'s tensor binding is pure host-side name/shape
    // matching against already-uploaded pointers, so a failure here at a
    // small, affordable chunk width (128, well under any real device's free
    // memory) is always a real descriptor-building or artifact-contract bug
    // -- never device contention -- so a hard failure is correct under and
    // outside the profile alike. (P2-01, GitHub #83, gave the load its own
    // small scratch cudaMalloc too; that failure mode is deterministic on
    // the chunk width, not on a busy GPU, and is the last section below.)
    {
        let model =
            load_qwen38_27b(&reader, &artifact, &handles, 128, 128, ignis_core::KvFormat::Bf16)
                .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
        let stats = model.stats();
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
    }

    // --- P2-01: the scratch reservation grows with the prefill chunk --------
    //
    // The same materialized artifact, loaded twice (no repeat upload) at two
    // chunk widths with the same pool geometry, so any VRAM delta is
    // attributable to the chunk alone.
    {
        // Wide enough to bound both chunk widths compared below (P2-01,
        // GitHub #83: prefill_chunk_tokens must not exceed max_context_tokens).
        const MAX_CONTEXT: u32 = 1024;
        let vram_for = |chunk: u32| -> u64 {
            let model = load_qwen38_27b(&reader, &artifact, &handles, chunk, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
                .unwrap_or_else(|e| panic!("ignis_model_load (chunk={chunk}): {e}"));
            let pool = SeqPool::create(
                &ModelConfig::qwen38_27b(),
                &SeqPoolBudget {
                    kv_format: ignis_core::KvFormat::Bf16,
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
    }

    // --- P2-01: an unaffordable chunk fails the load, not the first prompt --
    //
    // Last, so a rejected load's own allocation attempt cannot disturb the
    // measurements above.
    {
        // A multiple of 128 wide enough that a single layer's plain activation
        // buffers alone dwarf any real device's free memory beside the
        // ~19 GiB-resident 27B weights. max_context_tokens is set to match (P2-01,
        // GitHub #83: prefill_chunk_tokens must not exceed max_context_tokens) so
        // the load fails on the memory budget, not on that unrelated precondition.
        const HUGE_CHUNK: u32 = 8 * 1024 * 1024;
        let err = match load_qwen38_27b(&reader, &artifact, &handles, HUGE_CHUNK, HUGE_CHUNK, ignis_core::KvFormat::Bf16) {
            Ok(_) => panic!("an unaffordable chunk width must fail the load"),
            Err(e) => e,
        };
        assert!(
            err.contains("bytes are free") || err.contains("scratch"),
            "the error should name the shortfall: {err}"
        );
    }

    // A `MaterializedArtifact` has no `Drop` of its own (its arena needs the
    // `Device` that produced it to release, mirrors `CudaLeaf::drop`,
    // crates/runtime/src/cuda_leaf.rs), so release it explicitly rather than
    // leaving ~19 GiB resident for whatever runs next in this process.
    let _ = artifact.release_arena(&mut device);
}
