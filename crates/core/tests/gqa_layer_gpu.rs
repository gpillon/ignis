//! GPU integration test for one GQA layer in the program (P1-21, GitHub #57,
//! ADR 0009). It exercises device-resident input norm, fused attention input
//! projection, q/k norm plus RoPE, KV append, causal attention with the output
//! gate, output residual, and the MLP tail against P1-20's f64 reference.
//!
//! The two representative layers cover the artifact's BF16 exception arm
//! (layer 3) and its ordinary NVFP4 arm (layer 27). For each, T=1 and four
//! sequential T=1 calls must agree with the f64 oracle: the latter proves that
//! K/V are retained in the sequence's per-GQA-layer paged cache and that the
//! leaf advances RoPE positions once per token rather than once per layer.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_text_scope_27b, f64_reference, materialize, CudaDevice, Device, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::gqa_layer::run_gqa_layer;
use ignis_core::model_load::{load_qwen38_27b, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const HIDDEN: usize = 5120;
const BF16_GQA_LAYER: u32 = 3;
const NVFP4_GQA_LAYER: u32 = 27;
const MAX_CONTEXT_TOKENS: u32 = 128;
const BF16_LAYER_TOLERANCE: f64 = 8.0 / 256.0;

fn bf16_to_f64(bits: u16) -> f64 {
    f32::from_bits(u32::from(bits) << 16) as f64
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

fn deterministic_residual_bf16(tokens: usize) -> Vec<u16> {
    (0..tokens * HIDDEN)
        .map(|i| {
            let token = i / HIDDEN;
            let feature = i % HIDDEN;
            let seed = (token * 13 + feature) as f32;
            f32_to_bf16(seed.sin() * 0.01 * (1.0 + (feature % 997) as f32 * 1e-3))
        })
        .collect()
}

fn residual_f64(residual_bf16: &[u16]) -> Vec<f64> {
    residual_bf16
        .iter()
        .map(|&bits| bf16_to_f64(bits))
        .collect()
}

fn assert_matches_bf16_tolerance(actual_bf16: &[u16], reference: &[f64], label: &str) {
    assert_eq!(actual_bf16.len(), reference.len());
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    let mut max_abs_error = 0f64;
    let mut max_abs_reference = 0f64;
    for (&actual_bits, &expected) in actual_bf16.iter().zip(reference) {
        let error = bf16_to_f64(actual_bits) - expected;
        squared_error += error * error;
        squared_reference += expected * expected;
        max_abs_error = max_abs_error.max(error.abs());
        max_abs_reference = max_abs_reference.max(expected.abs());
    }
    let relative_l2 = squared_error.sqrt() / squared_reference.sqrt().max(1e-30);
    assert!(
        relative_l2 <= BF16_LAYER_TOLERANCE,
        "{label}: relative L2 error {relative_l2} exceeds {BF16_LAYER_TOLERANCE}"
    );
    let gross_limit = BF16_LAYER_TOLERANCE * (1.0 + max_abs_reference);
    assert!(
        max_abs_error <= gross_limit,
        "{label}: max absolute error {max_abs_error} exceeds {gross_limit}"
    );
}

fn run_layer_and_check(
    reader: &Reader,
    model: &Model,
    pool: &SeqPool,
    device: &mut CudaDevice,
    layer: u32,
    tokens: usize,
    label: &str,
) -> bool {
    let sequence = pool
        .alloc(MAX_CONTEXT_TOKENS)
        .unwrap_or_else(|e| panic!("{label}: sequence alloc: {e}"));
    let residual_bf16 = deterministic_residual_bf16(tokens);
    let input_bytes: Vec<u8> = residual_bf16
        .iter()
        .flat_map(|&bits| bits.to_le_bytes())
        .collect();
    let token_bytes = HIDDEN * std::mem::size_of::<u16>();
    let in_buf = device
        .allocate(token_bytes as u64)
        .unwrap_or_else(|e| panic!("{label}: allocate input: {e}"));
    let out_buf = device
        .allocate(token_bytes as u64)
        .unwrap_or_else(|e| panic!("{label}: allocate output: {e}"));
    let mut out_bf16 = Vec::with_capacity(tokens * HIDDEN);
    for token in 0..tokens {
        device
            .copy_h2d(
                &in_buf,
                0,
                &input_bytes[token * token_bytes..(token + 1) * token_bytes],
            )
            .unwrap_or_else(|e| panic!("{label}: H2D token {token}: {e}"));
        if let Err(e) = run_gqa_layer(model, pool, &sequence, layer, &in_buf, &out_buf, 1) {
            if gpu_profile::skip_or_fail(&format!(
                "{label}: token {token}: ignis_gqa_layer_step: {e}"
            )) {
                return true;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
        let mut out_bytes = vec![0u8; token_bytes];
        device
            .copy_d2h(&out_buf, 0, &mut out_bytes)
            .unwrap_or_else(|e| panic!("{label}: D2H token {token}: {e}"));
        out_bf16.extend(
            out_bytes
                .chunks_exact(2)
                .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]])),
        );
    }
    let positions: Vec<i32> = (0..tokens as i32).collect();
    let reference = f64_reference::evaluate_layer(
        reader,
        layer as usize,
        &f64_reference::LayerInput::new(residual_f64(&residual_bf16), positions)
            .unwrap_or_else(|e| panic!("{label}: layer reference input: {e}")),
    )
    .unwrap_or_else(|e| panic!("{label}: f64 layer reference: {e}"));
    for token in 0..tokens {
        assert_matches_bf16_tolerance(
            &out_bf16[token * HIDDEN..(token + 1) * HIDDEN],
            &reference.residual[token * HIDDEN..(token + 1) * HIDDEN],
            &format!("{label}: token {token}"),
        );
    }
    false
}

/// Both the BF16 output exception and the normal NVFP4 GQA layer match the
/// f64 reference for a single token and four sequential T=1 causal calls.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn gqa_layers_match_f64_reference() {
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
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA device unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the text scope on device: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
    let cfg = ModelConfig::qwen38_27b();
    let pool = SeqPool::create(
        &cfg,
        &SeqPoolBudget {
            kv_page_group_count: 4,
            max_context_tokens: MAX_CONTEXT_TOKENS,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("ignis_seq_pool_create: {e}"));

    for (layer, format) in [(BF16_GQA_LAYER, "BF16"), (NVFP4_GQA_LAYER, "NVFP4")] {
        if run_layer_and_check(
            &reader,
            &model,
            &pool,
            &mut device,
            layer,
            1,
            &format!("{format} layer {layer}, T=1"),
        ) {
            return;
        }
        if run_layer_and_check(
            &reader,
            &model,
            &pool,
            &mut device,
            layer,
            4,
            &format!("{format} layer {layer}, 4 x T=1 (KV/position carries)"),
        ) {
            return;
        }
    }
}
