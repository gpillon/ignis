//! GPU integration test for one GDN layer in the program (P1-22, GitHub #58,
//! ADR 0009): the layer's full attention + MLP tail, composed in the kernel
//! leaf on top of the sequence's GDN state (the conv taps + fp32 recurrent
//! slot, GitHub #55), checked against the P1-20 f64 layer reference
//! (crates/artifact f64_reference.rs, evaluate_layer on the GDN layer).
//!
//! The acceptance criteria (GitHub #58):
//! - the layer output is within bf16 tolerance of the f64 reference for T=1
//!   and after 4 sequential tokens (the GDN slot + conv taps carry state
//!   across the 4 tokens);
//! - releasing and re-allocating the sequence resets the state (a fresh slot
//!   reads zero, so a re-allocated sequence matches the f64 reference's zero
//!   start);
//! - the layer-4 BF16 output-projection arm is exercised (the GDN layer used
//!   here is the 27B's BF16 output exception).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure** (`ignis_core::gpu_profile::skip_or_fail`).
//! Run via `scripts/gpu-profile.ps1` (stops the reference `ninfer-serve`
//! first -- the RTX 5090 is exclusive, ADR 0006).

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_text_scope_27b, f64_reference, materialize, CudaDevice, Device, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gdn_layer::run_gdn_layer;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads) -- mirrors `crates/core/tests/step_degenerate_gpu.rs`.
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const HIDDEN: usize = 5120;
/// The GDN layer the f64 reference records (GDN's BF16 output-projection
/// exception -- the layer-4 BF16 output arm the acceptance criteria requires).
const GDN_LAYER: u32 = 4;
/// The pool's max context (the largest single-sequence KV reservation); the
/// GDN layer test needs only a small KV reservation (the GDN state pool, not
/// the KV pages, carries the layer's state).
const MAX_CONTEXT_TOKENS: u32 = 128;
/// A few BF16 unit roundoffs: the layer's ~8 bf16 storage roundings, each one
/// ulp, aggregate over the whole residual (the end-to-end reduction
/// criterion, the reference op tests' A16 linear tolerance scaled by the
/// layer's op count).
const BF16_LAYER_TOLERANCE: f64 = 8.0 / 256.0;

/// bf16 storage -> f64 (bit-exact promotion: bf16 is fp32's top 16 bits,
/// zero-extended).
fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

/// A deterministic, finite, non-degenerate residual (token-major [T, HIDDEN]),
/// rounded to BF16 so the device and the f64 reference see the same input.
fn deterministic_residual_bf16(tokens: usize) -> Vec<u16> {
    (0..tokens * HIDDEN)
        .map(|i| {
            let t = i / HIDDEN;
            let h = i % HIDDEN;
            // A varied, finite pattern (a sine seed + a per-feature scale),
            // rounded to BF16 (round-to-nearest-even).
            let seed = (t * 13 + h) as f32;
            let v = seed.sin() * 0.01 * (1.0 + (h % 997) as f32 * 1e-3);
            f32_to_bf16(v)
        })
        .collect()
}

/// f32 -> BF16 (round-to-nearest-even, the IEEE 754 rounding of the low 16
/// bits into the high 16).
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// The BF16 residual promoted to f64 (the f64 reference's input, the same
/// values the device's BF16 input sees).
fn residual_f64(residual_bf16: &[u16]) -> Vec<f64> {
    residual_bf16.iter().map(|&b| bf16_to_f64(b)).collect()
}

/// The layer's residual matches the f64 reference within a few BF16 units
/// (the relative L2 over the whole residual + a gross absolute bound).
fn assert_matches_bf16_tolerance(actual_bf16: &[u16], reference: &[f64], label: &str) {
    assert_eq!(actual_bf16.len(), reference.len());
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    let mut max_abs_error = 0f64;
    let mut max_abs_reference = 0f64;
    for (&a_bits, &r) in actual_bf16.iter().zip(reference.iter()) {
        let a = bf16_to_f64(a_bits);
        let error = a - r;
        squared_error += error * error;
        squared_reference += r * r;
        max_abs_error = max_abs_error.max(error.abs());
        max_abs_reference = max_abs_reference.max(r.abs());
    }
    let relative_l2 = squared_error.sqrt() / squared_reference.sqrt().max(1e-30);
    assert!(
        relative_l2 <= BF16_LAYER_TOLERANCE,
        "{label}: relative L2 error {relative_l2} exceeds {BF16_LAYER_TOLERANCE}"
    );
    let gross_limit = BF16_LAYER_TOLERANCE * (1.0 + max_abs_reference);
    assert!(
        max_abs_error <= gross_limit,
        "{label}: max absolute error {max_abs_error} exceeds the gross limit {gross_limit}"
    );
}

/// Runs the GDN layer for `tokens` sequential tokens of one fresh sequence
/// (zero state), and checks the residual against the f64 reference (the
/// reference starts from zero state, matching a fresh sequence's zero slot).
///
/// Returns `true` if the caller should skip (a busy GPU or a kernel error
/// outside the GPU profile, `gpu_profile::skip_or_fail`); under the profile
/// that condition panics instead, so `true` is never actually returned.
fn run_layer_and_check(
    reader: &Reader,
    model: &Model,
    pool: &SeqPool,
    device: &mut CudaDevice,
    tokens: usize,
    label: &str,
) -> bool {
    let sequence = pool
        .alloc(MAX_CONTEXT_TOKENS)
        .unwrap_or_else(|e| panic!("{label}: sequence alloc: {e}"));

    let residual_bf16 = deterministic_residual_bf16(tokens);
    let input_bytes: Vec<u8> = residual_bf16.iter().flat_map(|&b| b.to_le_bytes()).collect();
    let in_buf = device
        .allocate(input_bytes.len() as u64)
        .unwrap_or_else(|e| panic!("{label}: allocate input: {e}"));
    let out_buf = device
        .allocate(input_bytes.len() as u64)
        .unwrap_or_else(|e| panic!("{label}: allocate output: {e}"));
    device
        .copy_h2d(&in_buf, 0, &input_bytes)
        .unwrap_or_else(|e| panic!("{label}: H2D input: {e}"));
    device
        .synchronize()
        .unwrap_or_else(|e| panic!("{label}: synchronize before the layer: {e}"));

    if let Err(e) = run_gdn_layer(model, pool, &sequence, GDN_LAYER, &in_buf, &out_buf, tokens as u64) {
        if gpu_profile::skip_or_fail(&format!("{label}: ignis_gdn_layer_step: {e}")) {
            return true;
        }
        unreachable!("skip_or_fail panics under the profile");
    }

    let mut out_bytes = vec![0u8; input_bytes.len()];
    device
        .copy_d2h(&out_buf, 0, &mut out_bytes)
        .unwrap_or_else(|e| panic!("{label}: D2H output: {e}"));
    let out_bf16: Vec<u16> = out_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // The f64 reference (P1-20): the same layer, the same (BF16-promoted)
    // input, starting from zero state. The GDN layer has no RoPE, so the
    // positions are the token indices (unused by the recurrence).
    let positions: Vec<i32> = (0..tokens as i32).collect();
    let reference = f64_reference::evaluate_layer(
        reader,
        GDN_LAYER as usize,
        &f64_reference::LayerInput::new(residual_f64(&residual_bf16), positions)
            .unwrap_or_else(|e| panic!("{label}: layer reference input: {e}")),
    )
    .unwrap_or_else(|e| panic!("{label}: f64 layer reference: {e}"));

    assert_matches_bf16_tolerance(&out_bf16, &reference.residual, label);
    drop(sequence);
    false
}

/// The GDN layer's residual matches the f64 reference for T=1 and T=4 (the
/// sequence's GDN slot and conv taps carry state across the 4 tokens; a
/// released and re-allocated sequence resets them), exercising the layer-4
/// BF16 output-projection arm.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn gdn_layer_matches_f64_reference() {
    let path = Path::new(ARTIFACT);
    if !path.exists()
        && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}"))
    {
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

    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(a) => a,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on the device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // `ignis_model_load` does no CUDA work (kernel/src/model.cu is pure
    // host-side name/shape matching against already-uploaded pointers), so
    // its error is a real contract bug, never GPU contention -- a hard
    // failure under and outside the profile alike (mirrors model_load_gpu.rs).
    let model = load_qwen38_27b(&reader, &artifact, &handles)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));

    // The sequence pool (the GDN state: conv taps + fp32 recurrent slot),
    // sized by the model's GDN geometry. A small KV budget (the GDN layer
    // needs only the GDN state pool; the KV pages are reserved but unused
    // here).
    let cfg = ModelConfig::qwen38_27b();
    let budget = SeqPoolBudget {
        kv_page_group_count: 4,
        max_context_tokens: MAX_CONTEXT_TOKENS,
        slot_count: 2,
    };
    let pool = SeqPool::create(&cfg, &budget)
        .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    // T=1: a fresh sequence's GDN slot + conv taps carry state across one
    // token (the zero-state start matches the f64 reference).
    if run_layer_and_check(&reader, &model, &pool, &mut device, 1, "T=1") {
        return;
    }

    // T=4: the sequence's GDN slot + conv taps carry state across 4
    // sequential tokens (the recurrence and the rolling conv taps evolve over
    // the 4 tokens, matching the f64 reference's causal recurrence).
    if run_layer_and_check(&reader, &model, &pool, &mut device, 4, "T=4 (state carries)") {
        return;
    }

    // Re-allocating a released sequence resets its GDN slot + conv taps (a
    // fresh slot reads zero, ignis_seq.h): a re-allocated sequence must match
    // the f64 reference's zero-state start again, not the previous occupant's
    // accumulated state.
    run_layer_and_check(&reader, &model, &pool, &mut device, 1, "re-alloc reset");
}