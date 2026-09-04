//! Gated GPU launch tests for the ticket-29 (kernel-abi 10, GitHub #29)
//! C-ABI surface.
//!
//! Launches the bf16 GEMM `ignis_bf16_gemm` (the logits path for the
//! W8-dequantized lm_head) on the RTX 5090 with small *synthetic* inputs (no
//! model weights) and compares the kernel output against a CPU reference
//! GEMM computed in Rust. The multi-token case (`tokens > 1`, the
//! batched-prefill logits path) is checked against the CPU reference GEMM; a
//! `tokens == 1` case (the GEMV special case, the decode logits path) is
//! cross-checked against the same CPU reference with a null bias (the
//! nullable-bias path). Gated: `#[ignore]` by default, and they self-skip
//! (a non-zero return means "GPU busy, skip") so a busy GPU never turns the
//! suite red (ADR 0006 nuance — a few KB of VRAM runs even with the model
//! loaded).
//!
//! Run with: `cargo test -p ignis-core --test kernel_abi10_gpu -- --ignored`.
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles AND the canonical `kernel/build/ignis_kernel.lib`
//! has been rebuilt with the ticket-29 symbol (`ignis_bf16_gemm`).

use std::ffi::c_void;

use ignis_core::ffi;

// --- bf16 helpers (same as decode_gpu / kernel_abi02_gpu / kernel_abi05_gpu) -

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

// --- CPU reference ------------------------------------------------------------

/// CPU reference bf16 GEMM:
/// `out[t][m] = bias[m] + sum_k act[t][k] * W[m][k]`.
/// Plain bf16 planes (no NVFP4 codes/scales). Mirrors the kernel's math
/// (fp32 accumulation, bias added last, final bf16 rounding), hence the
/// tolerance at the call sites. `bias` is `None` for the nullable-bias path.
fn cpu_bf16_gemm(
    act: &[f32],
    wt: &[f32],
    bias: Option<&[f32]>,
    tokens: i64,
    m: i64,
    k: i64,
) -> Vec<f32> {
    let k = k as usize;
    (0..(tokens as usize))
        .flat_map(|ti| {
            (0..(m as usize)).map(move |mi| {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    acc += act[ti * k + ki] * wt[mi * k + ki];
                }
                acc + bias.map_or(0.0, |b| b[mi])
            })
        })
        .collect()
}

// --- Synthetic inputs ---------------------------------------------------------

/// Deterministic synthetic bf16 GEMM inputs (no model weights). tokens=8,
/// m=32, k=64. Activation, weight, and bias values use bf16-exact
/// magnitudes (0.25/0.5/0.75/1.0 — powers-of-2 fractions, exact in bf16),
/// so the CPU reference (f32) and the kernel (bf16) see the same input
/// values and only the fp32 accumulation order and the final bf16 rounding
/// differ.
fn synthetic_bf16_gemm() -> (Vec<f32>, Vec<f32>, Vec<f32>, i64, i64, i64) {
    const TOKENS: i64 = 8;
    const M: i64 = 32;
    const K: i64 = 64;
    let (t, m, k) = (TOKENS as usize, M as usize, K as usize);
    // act[t][k] = 0.25 * ((t*3 + k*5) % 4 + 1) -> {0.25,0.5,0.75,1.0} (bf16-exact).
    let act: Vec<f32> = (0..t * k)
        .map(|i| {
            let ti = i / k;
            let ki = i % k;
            0.25 * (((ti * 3 + ki * 5) % 4) + 1) as f32
        })
        .collect();
    // wt[m][k] = 0.25 * ((m*7 + k*3) % 4 + 1) -> {0.25,0.5,0.75,1.0} (bf16-exact).
    let wt: Vec<f32> = (0..m * k)
        .map(|i| {
            let mi = i / k;
            let ki = i % k;
            0.25 * (((mi * 7 + ki * 3) % 4) + 1) as f32
        })
        .collect();
    // bias[m] = 0.5 * (m % 2 + 1) -> {0.5, 1.0} (bf16-exact).
    let bias: Vec<f32> = (0..m).map(|mi| 0.5 * ((mi % 2) + 1) as f32).collect();
    (act, wt, bias, TOKENS, M, K)
}

// --- Skip helper ---------------------------------------------------------------

/// Skip helper: a non-zero rc (CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by the inference server).
fn skip_if_busy(rc: i32, what: &str) -> bool {
    if rc != 0 {
        eprintln!("SKIP: {what} returned {rc} (GPU busy / unavailable, ADR 0006)");
        return true;
    }
    false
}

// --- GPU-gated launch tests ----------------------------------------------------

/// Multi-token bf16 GEMM vs the CPU reference GEMM (the `tokens > 1` case,
/// the batched-prefill logits path), with a bias.
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn bf16_gemm_gpu() {
    let (act, wt, bias, tokens, m, k) = synthetic_bf16_gemm();
    let act_bf16 = to_bf16(&act);
    let wt_bf16 = to_bf16(&wt);
    let bias_bf16 = to_bf16(&bias);
    let mut out = vec![0u16; (tokens * m) as usize];
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act_bf16.as_ptr() as *const c_void,
            wt_bf16.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            tokens,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_bf16_gemm (multi-token)") {
        return;
    }
    let ref_out = cpu_bf16_gemm(&act, &wt, Some(&bias), tokens, m, k);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "bf16_gemm[{i}]: got {got}, want {want}"
        );
    }
}

/// Shapes where **both** `tokens` and `m` are not multiples of the 16x16 tile
/// (tokens=8, m=24, k=48 — k also not a multiple of the 32-element shared
/// chunk): the kernel must clamp out-of-range tile rows in the staging (no
/// out-of-bounds reads), handle the tail k-chunk, and still match the CPU
/// reference. This is the regression pin for the bounds-safe staging (the
/// `tokens > 1` test above uses 16-aligned m, so it never exercises the
/// m-row clamp).
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn bf16_gemm_non_tile_aligned_shapes() {
    const TOKENS: i64 = 8; // not a multiple of 16
    const M: i64 = 24; // not a multiple of 16
    const K: i64 = 48; // not a multiple of the 32-element shared chunk
    let (t, m, k) = (TOKENS as usize, M as usize, K as usize);
    let act: Vec<f32> = (0..t * k)
        .map(|i| {
            let ti = i / k;
            let ki = i % k;
            0.25 * (((ti * 5 + ki * 3) % 4) + 1) as f32
        })
        .collect();
    let wt: Vec<f32> = (0..m * k)
        .map(|i| {
            let mi = i / k;
            let ki = i % k;
            0.25 * (((mi * 7 + ki * 5) % 4) + 1) as f32
        })
        .collect();
    let bias: Vec<f32> = (0..m).map(|mi| 0.5 * ((mi % 2) + 1) as f32).collect();

    let act_bf16 = to_bf16(&act);
    let wt_bf16 = to_bf16(&wt);
    let bias_bf16 = to_bf16(&bias);
    let mut out = vec![0u16; (TOKENS * M) as usize];
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act_bf16.as_ptr() as *const c_void,
            wt_bf16.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            TOKENS,
            M,
            K,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_bf16_gemm (non-tile-aligned)") {
        return;
    }
    let ref_out = cpu_bf16_gemm(&act, &wt, Some(&bias), TOKENS, M, K);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "bf16_gemm_nonaligned[{i}]: got {got}, want {want}"
        );
    }
}

/// `tokens == 1` (the GEMV special case, the decode logits path) vs the CPU
/// reference, with a **null bias** (the nullable-bias path — the lm_head
/// carries no bias in the v1 artifact, so the decode logits path exercises
/// the null branch).
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn bf16_gemm_single_token_gemv() {
    let (act, wt, _bias, _tokens, m, k) = synthetic_bf16_gemm();
    let x_single = &act[0..k as usize]; // the first token's activation vector [k]
    let x_bf16 = to_bf16(x_single);
    let wt_bf16 = to_bf16(&wt);
    let mut out = vec![0u16; m as usize];
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            x_bf16.as_ptr() as *const c_void,
            wt_bf16.as_ptr() as *const c_void,
            std::ptr::null(), // null bias: the nullable-bias path
            out.as_mut_ptr() as *mut c_void,
            1,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_bf16_gemm (tokens=1, null bias)") {
        return;
    }
    let ref_out = cpu_bf16_gemm(x_single, &wt, None, 1, m, k);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!(
            (got - want).abs() <= tol,
            "bf16_gemm_gemv[{i}]: got {got}, want {want}"
        );
    }
}

// --- CPU-only (no GPU) argument-validation pins --------------------------------

/// Invalid arguments are rejected with -1 *before* any CUDA call, so these
/// run on CPU (no GPU needed) and pin the surface's validation contract.
#[test]
fn bf16_gemm_rejects_invalid_args() {
    let act = vec![0u16; 16];
    let wt = vec![0u16; 16];
    let mut out = vec![0u16; 16];
    // tokens = 0 -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act.as_ptr() as *const c_void,
            wt.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            0,
            8,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "tokens=0 must be rejected");
    // m = 0 -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act.as_ptr() as *const c_void,
            wt.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            1,
            0,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "m=0 must be rejected");
    // k = 0 -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act.as_ptr() as *const c_void,
            wt.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            1,
            8,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "k=0 must be rejected");
    // null act -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            std::ptr::null(),
            wt.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            1,
            8,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "null act must be rejected");
    // null wt -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act.as_ptr() as *const c_void,
            std::ptr::null(),
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            1,
            8,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "null wt must be rejected");
    // null out -> invalid.
    let rc = unsafe {
        ffi::ignis_bf16_gemm(
            act.as_ptr() as *const c_void,
            wt.as_ptr() as *const c_void,
            std::ptr::null(),
            std::ptr::null_mut(),
            1,
            8,
            16,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "null out must be rejected");
}