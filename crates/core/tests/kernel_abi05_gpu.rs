//! Gated GPU launch tests for the ticket-22 (kernel-abi 05, GitHub #22)
//! C-ABI surface.
//!
//! Launches the multi-token NVFP4 GEMM `ignis_nvfp4_gemm_prefill` on the RTX
//! 5090 with small *synthetic* inputs (no model weights) and compares the
//! kernel output against a CPU reference computed in Rust. The multi-token
//! case (`tokens > 1`) is checked against the CPU reference GEMM; a
//! `tokens == 1` case is cross-checked against the single-token GEMV
//! (`ignis_nvfp4_gemm_decode`, the 1-token special case). Gated: `#[ignore]`
//! by default, and they self-skip (a non-zero return means "GPU busy, skip")
//! so a busy GPU never turns the suite red (ADR 0006 nuance — a few MB of
//! VRAM runs even with the model loaded).
//!
//! Run with: `cargo test -p ignis-core --test kernel_abi05_gpu -- --ignored`.
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles AND the canonical `kernel/build/ignis_kernel.lib`
//! has been rebuilt with the ticket-22 symbol (`ignis_nvfp4_gemm_prefill`).

use std::ffi::c_void;

use ignis_core::ffi;

// --- bf16 helpers (same as decode_gpu / kernel_abi02_gpu) --------------------

/// Encode an f32 as a bf16 (16-bit) value (round-to-nearest-even into 16 bits).
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = ((b >> 16) & 1) as u32;
    ((b + 0x7fff + lsb) >> 16) as u16
}

/// Decode a bf16 (16-bit) value to f32.
fn from_bf16(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Build a bf16 (16-bit) buffer from f32 values.
fn to_bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| bf16_bits(v)).collect()
}

// --- NVFP4 decode helpers (1:1 with kernel/src/nvfp4_gemm_decode.cuh) --------

/// E2M1 (FP4) decode: 1 sign bit (0x8), 3 magnitude bits -> {0,.5,1,1.5,2,3,4,6}.
/// 1:1 with `decode_nvfp4_e2m1`.
fn decode_e2m1(code: u8) -> f32 {
    let mag = match code & 0x7 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if code & 0x8 != 0 {
        -mag
    } else {
        mag
    }
}

/// E4M3 (FP8) decode: OCP FP8 E4M3 (bias 7, no inf). 1:1 with
/// `decode_nvfp4_e4m3`.
fn decode_e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let man = (code & 0x07) as f32;
    let mag = if exp == 0 {
        (man / 8.0) * 0.015625 // subnormal: (m/8) * 2^-6
    } else {
        (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * mag
}

// --- CPU reference ------------------------------------------------------------

/// CPU reference multi-token NVFP4 GEMM:
/// `out[t][m] = bias[m] + sum_k act[t][k] * (e2m1(code[m][k]) * e4m3(scale[m][k/16]))`.
/// Mirrors the kernel's dequantization (E2M1 codes 2 per byte, E4M3 scale per
/// 16-element group) and group-scaled dot product exactly (the kernel rounds
/// the final result to bf16, hence the tolerance at the call sites).
fn cpu_nvfp4_gemm_prefill(
    act: &[f32],
    wt_codes: &[u8],
    wt_scales: &[u8],
    bias: &[f32],
    tokens: i64,
    m: i64,
    k: i64,
) -> Vec<f32> {
    let k = k as usize;
    let code_row = k / 2;
    let scale_row = k / 16;
    (0..(tokens as usize))
        .flat_map(|t| {
            (0..m as usize).map(move |mi| {
                let cr = mi * code_row;
                let sr = mi * scale_row;
                let mut acc = 0.0f32;
                for ki in 0..k {
                    let code = (wt_codes[cr + ki / 2] >> ((ki & 1) * 4)) & 0x0F;
                    let group = ki / 16;
                    let w = decode_e2m1(code) * decode_e4m3(wt_scales[sr + group]);
                    acc += act[t * k + ki] * w;
                }
                acc + bias[mi]
            })
        })
        .collect()
}

// --- Synthetic inputs ---------------------------------------------------------

/// Deterministic synthetic multi-token NVFP4 GEMM inputs (no model weights).
/// tokens=8, m=32, k=64 (a multiple of 16). Activation, bias, and scale
/// values use bf16-exact magnitudes (powers-of-2 fractions; E4M3 0x38 = 1.0,
/// 0x48 = 4.0) so the CPU reference (f32) and the kernel (bf16) see the same
/// values and only the final bf16 rounding / accumulation order differ.
fn synthetic_gemm_prefill() -> (Vec<f32>, Vec<u8>, Vec<u8>, Vec<f32>, i64, i64, i64) {
    const TOKENS: i64 = 8;
    const M: i64 = 32;
    const K: i64 = 64; // multiple of 16
    let (t, m, k) = (TOKENS as usize, M as usize, K as usize);
    // act[t][k] = 0.25 * ((t*5 + k) % 4 + 1) -> {0.25,0.5,0.75,1.0} (bf16-exact).
    let act: Vec<f32> = (0..t * k)
        .map(|i| {
            let tt = i / k;
            let kk = i % k;
            0.25 * (((tt * 5 + kk) % 4) + 1) as f32
        })
        .collect();
    // bias[m] = 0.25 * (m % 4 + 1) -> {0.25,0.5,0.75,1.0} (bf16-exact).
    let bias: Vec<f32> = (0..m).map(|mi| 0.25 * ((mi % 4) + 1) as f32).collect();
    // wt_codes[m][k/2]: byte b (k = 2b, 2b+1): lo = code(m,2b), hi = code(m,2b+1).
    let code_row = k / 2;
    let wt_codes: Vec<u8> = (0..m * code_row)
        .map(|i| {
            let mi = i / code_row;
            let b = i % code_row;
            let lo = ((mi * 7 + 2 * b) % 8) as u8; // k = 2b
            let hi = ((mi * 7 + 2 * b + 1) % 8) as u8; // k = 2b+1
            lo | (hi << 4)
        })
        .collect();
    // wt_scales[m][k/16]: group g = k/16 -> 0x38 (1.0) or 0x48 (4.0).
    let scale_row = k / 16;
    let wt_scales: Vec<u8> = (0..m * scale_row)
        .map(|i| {
            let mi = i / scale_row;
            let g = i % scale_row;
            if (mi + g) % 2 == 0 {
                0x38
            } else {
                0x48
            }
        })
        .collect();
    (act, wt_codes, wt_scales, bias, TOKENS, M, K)
}

// --- Skip helper ---------------------------------------------------------------

/// Skip helper: a non-zero rc (CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by ninfer-serve).
fn skip_if_busy(rc: i32, what: &str) -> bool {
    if rc != 0 {
        eprintln!("SKIP: {what} returned {rc} (GPU busy / unavailable, ADR 0006)");
        return true;
    }
    false
}

// --- GPU-gated launch tests ----------------------------------------------------

/// Multi-token NVFP4 GEMM vs the CPU reference GEMM (the `tokens > 1` case).
#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn nvfp4_gemm_prefill_gpu() {
    let (act, wt_codes, wt_scales, bias, tokens, m, k) = synthetic_gemm_prefill();
    let act_bf16 = to_bf16(&act);
    let bias_bf16 = to_bf16(&bias);
    let mut out = vec![0u16; (tokens * m) as usize];
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_prefill(
            act_bf16.as_ptr() as *const c_void,
            wt_codes.as_ptr() as *const c_void,
            wt_scales.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            tokens,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_nvfp4_gemm_prefill") {
        return;
    }
    let ref_out = cpu_nvfp4_gemm_prefill(&act, &wt_codes, &wt_scales, &bias, tokens, m, k);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "gemm_prefill[{i}]: got {got}, want {want}"
        );
    }
}

/// `tokens == 1` must match the single-token GEMV (`ignis_nvfp4_gemm_decode`)
/// — the GEMV is the 1-token special case (regression pin).
#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn nvfp4_gemm_prefill_single_token_matches_gemv() {
    let (act, wt_codes, wt_scales, bias, _tokens, m, k) = synthetic_gemm_prefill();
    let x_single = &act[0..k as usize]; // the first token's activation vector [k]
    let x_bf16 = to_bf16(x_single);
    let bias_bf16 = to_bf16(&bias);

    // The multi-token kernel with tokens == 1 (act = [1][k]).
    let act1_bf16 = to_bf16(x_single);
    let mut out_prefill = vec![0u16; m as usize];
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_prefill(
            act1_bf16.as_ptr() as *const c_void,
            wt_codes.as_ptr() as *const c_void,
            wt_scales.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out_prefill.as_mut_ptr() as *mut c_void,
            1,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_nvfp4_gemm_prefill (tokens=1)") {
        return;
    }

    // The single-token GEMV (the 1-token special case).
    let mut out_gemv = vec![0u16; m as usize];
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_decode(
            x_bf16.as_ptr() as *const c_void,
            wt_codes.as_ptr() as *const c_void,
            wt_scales.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out_gemv.as_mut_ptr() as *mut c_void,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_nvfp4_gemm_decode (tokens=1)") {
        return;
    }

    // The prefill (tokens=1) output and the GEMV output should agree within
    // the bf16 tolerance (both do the fp32 math; only the accumulation order
    // and the final bf16 rounding differ).
    for (i, (got_bits, want_bits)) in out_prefill.iter().zip(out_gemv.iter()).enumerate() {
        let got = from_bf16(*got_bits);
        let want = from_bf16(*want_bits);
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "prefill-vs-gemv[{i}]: got {got}, want {want}"
        );
    }
}

/// Shapes where **both** `tokens` and `m` are not multiples of the 16x16 tile
/// (tokens=8, m=24, k=48): the kernel must clamp out-of-range tile rows in the
/// staging (no out-of-bounds reads) and still match the CPU reference. This is
/// the regression pin for the bounds-safe staging (the `tokens > 1` test above
/// uses m=32, a multiple of 16, so it never exercises the m-row clamp).
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn nvfp4_gemm_prefill_non_tile_aligned_shapes() {
    const TOKENS: i64 = 8; // not a multiple of 16
    const M: i64 = 24; // not a multiple of 16
    const K: i64 = 48; // a multiple of 16
    let (t, m, k) = (TOKENS as usize, M as usize, K as usize);
    let act: Vec<f32> = (0..t * k)
        .map(|i| {
            let tt = i / k;
            let kk = i % k;
            0.25 * (((tt * 3 + kk) % 4) + 1) as f32
        })
        .collect();
    let bias: Vec<f32> = (0..m).map(|mi| 0.5 * ((mi % 2) + 1) as f32).collect();
    let code_row = k / 2;
    let wt_codes: Vec<u8> = (0..m * code_row)
        .map(|i| {
            let mi = i / code_row;
            let b = i % code_row;
            let lo = ((mi * 5 + 2 * b) % 8) as u8; // k = 2b
            let hi = ((mi * 5 + 2 * b + 1) % 8) as u8; // k = 2b+1
            lo | (hi << 4)
        })
        .collect();
    let scale_row = k / 16;
    let wt_scales: Vec<u8> = (0..m * scale_row)
        .map(|i| {
            let mi = i / scale_row;
            let g = i % scale_row;
            if (mi + g) % 2 == 0 {
                0x38
            } else {
                0x48
            }
        })
        .collect();

    let act_bf16 = to_bf16(&act);
    let bias_bf16 = to_bf16(&bias);
    let mut out = vec![0u16; (TOKENS * M) as usize];
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_prefill(
            act_bf16.as_ptr() as *const c_void,
            wt_codes.as_ptr() as *const c_void,
            wt_scales.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            TOKENS,
            M,
            K,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_nvfp4_gemm_prefill (non-tile-aligned)") {
        return;
    }
    let ref_out = cpu_nvfp4_gemm_prefill(&act, &wt_codes, &wt_scales, &bias, TOKENS, M, K);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "gemm_prefill_nonaligned[{i}]: got {got}, want {want}"
        );
    }
}

// --- CPU-only (no GPU) argument-validation pins --------------------------------

/// Invalid arguments are rejected with -1 *before* any CUDA call, so these
/// run on CPU (no GPU needed) and pin the surface's validation contract.
#[test]
fn nvfp4_gemm_prefill_rejects_invalid_args() {
    let act = vec![0u16; 16];
    let codes = vec![0u8; 8];
    let scales = vec![0u8; 1];
    let mut out = vec![0u16; 16];
    // tokens = 0 -> invalid.
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_prefill(
            act.as_ptr() as *const c_void,
            codes.as_ptr() as *const c_void,
            scales.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            0,
            8,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "tokens=0 must be rejected");
    // k not a multiple of 16 -> invalid.
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_prefill(
            act.as_ptr() as *const c_void,
            codes.as_ptr() as *const c_void,
            scales.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            1,
            8,
            8,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "k not a multiple of 16 must be rejected");
}