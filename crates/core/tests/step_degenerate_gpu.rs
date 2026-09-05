//! GPU integration test for the degenerate step ABI (P1-18, GitHub #54,
//! ADR 0009): embedding -> final RMSNorm -> W8G32 output head -> argmax,
//! every decoder layer skipped, checked against an independent Rust f64
//! reference decoded straight from the artifact's container bytes (the
//! acceptance criteria's "Rust f64 reference ... from the artifact's host
//! decoders").
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the
//! profile the same condition is a **hard failure**
//! (`ignis_core::gpu_profile::skip_or_fail`). Run via
//! `scripts/gpu-profile.ps1` (stops the reference `ninfer-serve` first --
//! the RTX 5090 is exclusive, ADR 0006).
//!
//! The f64 reference recomputes the full 248320-entry logits vector per
//! token id from the container's raw W8G32 bytes -- independent of
//! `ignis_artifact::normalize`'s (private) dequant path, per spec 04's
//! "independent oracle" convention. This is deliberately slow (a full
//! [vocab, hidden] GEMV in scalar f64 per token id, ADR 0005/0007: G1 is a
//! correctness floor, not a performance gate) -- expect tens of seconds per
//! token id.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_text_scope_27b, materialize, row_split_geometry, CudaDevice, NumericFormat, Reader,
    RowSplitGeometry,
};
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::step::decode_degenerate_batch;

/// The fork-local model cache (the artifact the running `ninfer-serve`
/// loads) -- mirrors `crates/core/tests/model_load_gpu.rs`.
const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const HIDDEN: usize = 5120;
const VOCAB: usize = 248320;
/// The Qwen 3.8-27B text config's RMSNorm epsilon -- must match the
/// topology descriptor `crates/core/src/model_load.rs` sends the leaf.
const RMS_NORM_EPS: f64 = 1.0e-6;
/// One BF16 unit roundoff -- the reference op tests' A16 linear tolerance
/// (`kernel/vendor/tests/ops/linear/linear_test_common.cpp`), reused here
/// as an end-to-end reduction criterion over the full logits vector.
const BF16_UNIT_ROUNDOFF: f64 = 1.0 / 256.0;

/// IEEE-754 half (F16) -> f32 (exact; decodes the W8G32 group scales).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 >> 15) & 1;
    let exp = (bits as u32 >> 10) & 0x1F;
    let mant = bits as u32 & 0x3FF;
    let bits32 = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half: value = mant * 2^-24. `p` is the 10-bit
            // mantissa's leading-zero count within the field (0..9).
            let p = mant.leading_zeros() - 22;
            let e = 112 - p;
            let m = (mant - (1u32 << (9 - p))) << (14 + p);
            (sign << 31) | (e << 23) | m
        }
    } else if exp == 31 {
        (sign << 31) | (0x7F << 23) | (mant << 13) // Inf / NaN
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits32)
}

/// bf16 storage -> f32 (bit-exact promotion: bf16 is fp32's top 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Decode one W8G32 row-split row to f64 (the per-group int8 code times the
/// group's F16 scale -- the reference's rowsplit decode atom, independently
/// re-derived here rather than calling `ignis_artifact::normalize`'s
/// private decoder).
fn w8_decode_row(payload: &[u8], geom: &RowSplitGeometry, row: u64) -> Vec<f64> {
    let groups_per_row = geom.groups_per_row as usize;
    let cols = geom.columns as usize;
    let low = &payload[..geom.low_plane_bytes as usize];
    let scale_off = geom.scale_plane_offset as usize;
    let scales = &payload[scale_off..scale_off + geom.scale_plane_bytes as usize];

    let mut out = vec![0f64; cols];
    let base_group = row as usize * groups_per_row;
    let mut c = 0usize;
    for g in 0..groups_per_row {
        if c >= cols {
            break;
        }
        let gi = base_group + g;
        let scale = f16_to_f32(u16::from_le_bytes([scales[gi * 2], scales[gi * 2 + 1]])) as f64;
        let group_base = gi * 32;
        let width = (cols - c).min(32);
        for ci in 0..width {
            let code = low[group_base + ci] as i8 as f64;
            out[c + ci] = code * scale;
        }
        c += width;
    }
    out
}

/// The dot product of a W8G32 row-split row with `vector` (the output
/// head's per-vocab-entry projection), computed the same way as
/// [`w8_decode_row`] but without materializing the decoded row.
fn w8_row_dot(payload: &[u8], geom: &RowSplitGeometry, row: u64, vector: &[f64]) -> f64 {
    let groups_per_row = geom.groups_per_row as usize;
    let cols = geom.columns as usize;
    let low = &payload[..geom.low_plane_bytes as usize];
    let scale_off = geom.scale_plane_offset as usize;
    let scales = &payload[scale_off..scale_off + geom.scale_plane_bytes as usize];

    let mut acc = 0f64;
    let base_group = row as usize * groups_per_row;
    let mut c = 0usize;
    for g in 0..groups_per_row {
        if c >= cols {
            break;
        }
        let gi = base_group + g;
        let scale = f16_to_f32(u16::from_le_bytes([scales[gi * 2], scales[gi * 2 + 1]])) as f64;
        let group_base = gi * 32;
        let width = (cols - c).min(32);
        for ci in 0..width {
            let code = low[group_base + ci] as i8 as f64;
            acc += code * scale * vector[c + ci];
        }
        c += width;
    }
    acc
}

/// The degenerate program's f64 reference for one token id: embed -> final
/// RMSNorm -> output head, over the full vocabulary.
fn f64_reference_logits(reader: &Reader, token_id: u32) -> Vec<f64> {
    let shape = [VOCAB as u64, HIDDEN as u64];
    let geom =
        row_split_geometry(NumericFormat::W8G32F16S, &shape).expect("token_embedding/output_head geometry");

    let embed_payload = reader.payload("text/token_embedding").expect("token_embedding payload").data;
    let embed_row = w8_decode_row(embed_payload, &geom, u64::from(token_id));

    let norm_payload = reader.payload("text/final_norm").expect("final_norm payload").data;
    let norm_weight: Vec<f64> = (0..HIDDEN)
        .map(|d| bf16_to_f32(u16::from_le_bytes([norm_payload[d * 2], norm_payload[d * 2 + 1]])) as f64)
        .collect();

    let mean_sq: f64 = embed_row.iter().map(|x| x * x).sum::<f64>() / HIDDEN as f64;
    let inv = 1.0 / (mean_sq + RMS_NORM_EPS).sqrt();
    let normed: Vec<f64> =
        embed_row.iter().zip(norm_weight.iter()).map(|(x, w)| x * inv * w).collect();

    let head_payload = reader.payload("text/output_head").expect("output_head payload").data;
    (0..VOCAB as u64).map(|v| w8_row_dot(head_payload, &geom, v, &normed)).collect()
}

/// The lowest-index argmax (`ninfer::ops::argmax`'s tie-break convention).
fn argmax_lowest_index(logits: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > logits[best] {
            best = i;
        }
    }
    best
}

/// The device logits match the f64 reference within one BF16 unit
/// roundoff, evaluated the same way as the reference op tests' A16 linear
/// criterion (relative L2 over the whole vector + a gross absolute bound).
fn assert_matches_bf16_tolerance(actual: &[f32], reference: &[f64], token_id: u32) {
    assert_eq!(actual.len(), reference.len());
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    let mut max_abs_reference = 0f64;
    let mut max_abs_error = 0f64;
    for (&a, &r) in actual.iter().zip(reference.iter()) {
        let error = f64::from(a) - r;
        squared_error += error * error;
        squared_reference += r * r;
        max_abs_reference = max_abs_reference.max(r.abs());
        max_abs_error = max_abs_error.max(error.abs());
    }
    let relative_l2 = squared_error.sqrt() / squared_reference.sqrt().max(1.0e-30);
    let gross_limit = BF16_UNIT_ROUNDOFF + 2.0 * BF16_UNIT_ROUNDOFF * max_abs_reference;
    assert!(
        relative_l2 <= BF16_UNIT_ROUNDOFF,
        "token {token_id}: relative L2 error {relative_l2} exceeds {BF16_UNIT_ROUNDOFF}"
    );
    assert!(
        max_abs_error <= gross_limit,
        "token {token_id}: max absolute error {max_abs_error} exceeds the gross limit {gross_limit}"
    );
}

#[test]
fn degenerate_program_matches_f64_reference_for_four_tokens() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("the real artifact is absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {ARTIFACT}: {e}"));

    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind text scope: {e}"));

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
    // its error is always a real descriptor-building or artifact-contract
    // bug, never GPU contention -- a hard failure here is correct under and
    // outside the profile alike (mirrors model_load_gpu.rs).
    let model = load_qwen38_27b(&reader, &artifact, &handles)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));

    // A few token ids spanning the vocabulary (the acceptance criteria's
    // "4 token ids"): the first, the second, one mid-range, and the last.
    let token_ids: [i32; 4] = [0, 1, 12_345, (VOCAB - 1) as i32];
    let mut device_logits = vec![0f32; token_ids.len() * VOCAB];
    let device_token_ids = match decode_degenerate_batch(&model, &token_ids, Some(&mut device_logits))
    {
        Ok(ids) => ids,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("ignis_decode: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    for (i, &token_id) in token_ids.iter().enumerate() {
        let reference = f64_reference_logits(&reader, token_id as u32);
        let actual = &device_logits[i * VOCAB..(i + 1) * VOCAB];
        assert_matches_bf16_tolerance(actual, &reference, token_id as u32);

        let reference_argmax = argmax_lowest_index(&reference);
        assert_eq!(
            device_token_ids[i] as usize, reference_argmax,
            "token {token_id}: device argmax disagrees with the f64 reference"
        );
    }

    drop(model);
}
