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
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads) -- mirrors `crates/artifact/tests/real_artifact.rs`.
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";

#[test]
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
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    assert!(artifact.stats().device_capacity_bytes > 0, "VRAM used is reported");

    // `ignis_model_load` does no CUDA work (kernel/src/model.cu is pure
    // host-side name/shape matching against already-uploaded pointers), so
    // its error is always a real descriptor-building or artifact-contract
    // bug, never GPU contention -- a hard failure here is correct under and
    // outside the profile alike.
    let model = load_qwen38_27b(&reader, &artifact, &handles)
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
}
