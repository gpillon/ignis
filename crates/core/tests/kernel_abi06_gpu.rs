//! Gated GPU launch tests for the ticket-06 (kernel-abi 06, GitHub #28)
//! C-ABI surface.
//!
//! Launches the two new kernel-abi 06 ops on the RTX 5090 with small
//! *synthetic* inputs (no model weights) and compares the kernel output
//! against a CPU reference computed in Rust:
//! - `ignis_gdn_causal_conv` (the GDN 4-tap depthwise causal conv + SiLU,
//!   the `gdn/convolution` tensor) vs the CPU reference conv (the same f32
//!   fma chain, SiLU, and 3-tap state shift — the final bf16 rounding is
//!   the only difference, hence the tolerance at the call sites);
//! - `ignis_rope_qk` (the GQA split-half NeoX RoPE, θ = 1e7, rotary_dim 64
//!   of head_dim 256 — 32 pairs) vs the CPU reference rotation (the fp64
//!   sincos oracle, the reference's bit-stable unscaled route — v1 is
//!   unscaled, factor 1.0).
//!
//! The multi-token conv case checks the output + the updated 3-tap state;
//! the `tokens == 1` case (the GEMV special case) checks the single-token
//! conv + the state update; the RoPE `pos` sweep checks the rotation (a
//! `pos == 0` identity pin + the un-rotated dims staying bit-exact).
//!
//! Gated: `#[ignore]`d by default, and they self-skip (a non-zero return
//! means "GPU busy, skip") so a busy GPU never turns the suite red (the
//! ADR 0006 nuance — a few KB of VRAM runs even with the model loaded).
//!
//! Run with: `cargo test -p ignis-core --test kernel_abi06_gpu -- --ignored`.
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles AND the canonical `kernel/build/ignis_kernel.lib`
//! has been rebuilt with the ticket-06 symbols (`ignis_gdn_causal_conv`,
//! `ignis_rope_qk`).

use std::ffi::c_void;

use ignis_core::{ffi, rope_inv_frequencies};

// --- bf16 helpers (same as decode_gpu / kernel_abi05_gpu) --------------------

/// Encode an f32 as a bf16 (16-bit) value (round-to-nearest-even into 16 bits).
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = (b >> 16) & 1;
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

// --- CPU reference conv -------------------------------------------------------

/// SiLU (the reference's `ops/common/math.cuh`: `x / (1 + exp(-x))`), the
/// fp32 form the kernel's epilogue uses.
fn silu_f32(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

/// CPU reference for the 4-tap depthwise causal conv + SiLU: per channel,
/// the rolling 3-tap state `(s0, s1, s2)` + the current tap `p`:
/// `out = silu(w0*s0 + w1*s1 + w2*s2 + w3*p)`, then the state shift
/// `(s0, s1, s2) = (s1, s2, p)`. Mirrors the kernel's f32 fma accumulation
/// order 1:1 (the final bf16 rounding is the only difference — the caller
/// applies the bf16 tolerance).
///
/// Returns `(out, state_out)`: the conv'd + SiLU'd output and the updated
/// rolling state after the chunk (the last 3 consumed taps).
fn cpu_gdn_causal_conv(
    projected: &[f32],   // [tokens][channels]
    conv_weight: &[f32], // [4][channels] (tap-major: w0, w1, w2, w3)
    state_in: &[f32],    // [channels][3] (s0, s1, s2 per channel)
    tokens: i64,
    channels: i64,
) -> (Vec<f32>, Vec<f32>) {
    let (t, c) = (tokens as usize, channels as usize);
    let mut out = vec![0.0f32; t * c];
    let mut state = state_in.to_vec(); // [channels][3]

    for tk in 0..t {
        for ch in 0..c {
            let col = tk * c + ch;
            let s0 = state[ch * 3];
            let s1 = state[ch * 3 + 1];
            let s2 = state[ch * 3 + 2];
            let w0 = conv_weight[ch];
            let w1 = conv_weight[c + ch];
            let w2 = conv_weight[2 * c + ch];
            let w3 = conv_weight[3 * c + ch];
            let p = projected[col];

            // The kernel's fma chain: w0*s0, + w1*s1, + w2*s2, + w3*p
            // (the same accumulation order — a single rounding per fma).
            let acc = w0.mul_add(s0, w1.mul_add(s1, w2.mul_add(s2, w3.mul_add(p, 0.0f32))));
            out[col] = silu_f32(acc);

            // The 3-tap state shift (the reference's `s0 = s1; s1 = s2; s2 = p;`).
            state[ch * 3] = s1;
            state[ch * 3 + 1] = s2;
            state[ch * 3 + 2] = p;
        }
    }
    (out, state)
}

// --- CPU reference RoPE -------------------------------------------------------

/// The RoPE inverse-frequency table (the reference's `rope_linear_frequencies`
/// table, `inv_freq[p] = θ^(-2p/rotary_dim)`), computed in f64 for the fp64
/// sincos oracle (the kernel consumes the f32 table, `rope_inv_frequencies`).
fn cpu_rope_inv_freqs(theta: f64, rotary_dim: i64) -> Vec<f64> {
    (0..(rotary_dim / 2))
        .map(|p| theta.powf(-2.0 * (p as f64) / (rotary_dim as f64)))
        .collect()
}

/// CPU reference for the GQA split-half NeoX RoPE (the fp64 sincos oracle):
/// per (batch, seq, head) row and each rotary pair `p in [0, R/2)`, with
/// `a = x[p]`, `b = x[p + R/2]`, `phi = pos * inv_freq[p]`:
/// `x[p] = a*cos(phi) - b*sin(phi)`, `x[p + R/2] = b*cos(phi) + a*sin(phi)`.
/// The un-rotated dims `[R, head_dim)` are left unchanged. Returns the
/// rotated q/k in fresh buffers (the kernel is in-place).
#[allow(clippy::too_many_arguments)] // mirrors the wide flat C ABI (ADR 0001)
fn cpu_rope_qk(
    q: &[f32],
    k: &[f32],
    inv_freq: &[f64],
    batch: i64,
    seq: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    pos: i32,
) -> (Vec<f32>, Vec<f32>) {
    let (b, s, nq, nk) = (
        batch as usize,
        seq as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
    );
    let (hd, half) = (head_dim as usize, (rotary_dim / 2) as usize);
    let mut q = q.to_vec();
    let mut k = k.to_vec();
    let phi_of = |p: usize| pos as f64 * inv_freq[p];
    let rotate_row = |row: &mut [f32], base: usize| {
        for p in 0..half {
            let phi = phi_of(p);
            let (c, sn) = (phi.cos(), phi.sin());
            let a = row[base + p] as f64;
            let bb = row[base + p + half] as f64;
            row[base + p] = (a * c - bb * sn) as f32;
            row[base + p + half] = (bb * c + a * sn) as f32;
        }
    };

    for bi in 0..b {
        for si in 0..s {
            for h in 0..nq {
                let base = ((bi * s + si) * nq + h) * hd;
                rotate_row(&mut q, base);
            }
            for h in 0..nk {
                let base = ((bi * s + si) * nk + h) * hd;
                rotate_row(&mut k, base);
            }
        }
    }
    (q, k)
}

// --- Synthetic inputs ---------------------------------------------------------

/// Deterministic synthetic GDN causal-conv inputs (no model weights).
/// `tokens=8`, `channels=64`. Projected / weight / state values use
/// bf16-exact magnitudes (multiples of 0.25 within a small range) so the
/// CPU reference (f32) and the kernel (bf16) see the same values and only
/// the final bf16 rounding / SiLU libm rounding differ.
fn synthetic_gdn_conv() -> (Vec<f32>, Vec<f32>, Vec<f32>, i64, i64) {
    const TOKENS: i64 = 8;
    const CHANNELS: i64 = 64;
    let (t, c) = (TOKENS as usize, CHANNELS as usize);
    // projected[t][c] = 0.25 * ((t*5 + c*3) % 7) - 0.5  ->  bf16-exact in [-0.5, 1.0).
    let projected: Vec<f32> = (0..t * c)
        .map(|i| {
            let (tk, ch) = (i / c, i % c);
            0.25 * (((tk * 5 + ch * 3) % 7) as f32) - 0.5
        })
        .collect();
    // conv_weight[tap][c] = 0.25 * ((tap*7 + c*5) % 5) - 0.5  ->  bf16-exact in [-0.5, 0.5).
    let conv_weight: Vec<f32> = (0..4 * c)
        .map(|i| {
            let (tap, ch) = (i / c, i % c);
            0.25 * (((tap * 7 + ch * 5) % 5) as f32) - 0.5
        })
        .collect();
    // state_in[c][j] = 0.25 * ((c*3 + j*11) % 7) - 0.5  ->  bf16-exact in [-0.5, 1.0).
    let state_in: Vec<f32> = (0..c * 3)
        .map(|i| {
            let (ch, j) = (i / 3, i % 3);
            0.25 * (((ch * 3 + j * 11) % 7) as f32) - 0.5
        })
        .collect();
    (projected, conv_weight, state_in, TOKENS, CHANNELS)
}

/// Deterministic synthetic RoPE q/k (no model weights). `batch=1`,
/// `seq=4`, `num_q_heads=8`, `num_kv_heads=2` (GQA 4:1), `head_dim=256`,
/// `rotary_dim=64` (32 pairs). Values use bf16-exact magnitudes (0.25 steps
/// within a small range) so the CPU reference (f64 oracle) and the kernel
/// (bf16 in, f32 math) see the same values.
fn synthetic_rope_qk() -> (Vec<f32>, Vec<f32>, i64, i64, i64, i64, i64, i64) {
    const BATCH: i64 = 1;
    const SEQ: i64 = 4;
    const NUM_Q: i64 = 8;
    const NUM_KV: i64 = 2;
    const HEAD_DIM: i64 = 256;
    const ROTARY: i64 = 64;
    let (b, s, nq, nk, hd) = (
        BATCH as usize,
        SEQ as usize,
        NUM_Q as usize,
        NUM_KV as usize,
        HEAD_DIM as usize,
    );
    // q[b][s][h][d] = 0.25 * ((flat*3 % 5) - 1)  ->  bf16-exact in [-0.25, 0.75].
    let q: Vec<f32> = (0..b * s * nq * hd)
        .map(|i| 0.25 * (((i * 3) % 5) as f32) - 0.25)
        .collect();
    // k[b][s][h][d] = 0.25 * ((flat*5 % 7) - 3)  ->  bf16-exact in [-0.75, 0.75].
    let k: Vec<f32> = (0..b * s * nk * hd)
        .map(|i| 0.25 * (((i * 5) % 7) as f32) - 0.75)
        .collect();
    (q, k, BATCH, SEQ, NUM_Q, NUM_KV, HEAD_DIM, ROTARY)
}

// --- Skip helper --------------------------------------------------------------

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

/// Multi-token GDN causal conv vs the CPU reference conv (the `tokens > 1`
/// case): the output and the updated 3-tap state are both checked (the
/// state is the last 3 consumed taps — a bit-exact pin).
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): run with -- --ignored"]
fn gdn_causal_conv_multi_token_gpu() {
    let (projected, conv_weight, state_in, tokens, channels) = synthetic_gdn_conv();
    let projected_bf16 = to_bf16(&projected);
    let conv_weight_bf16 = to_bf16(&conv_weight);
    let state_in_bf16 = to_bf16(&state_in);
    let mut out = vec![0u16; (tokens * channels) as usize];
    let mut state_out = vec![0u16; (channels * 3) as usize];

    let rc = unsafe {
        ffi::ignis_gdn_causal_conv(
            projected_bf16.as_ptr() as *const c_void,
            conv_weight_bf16.as_ptr() as *const c_void,
            state_in_bf16.as_ptr() as *const c_void,
            state_out.as_mut_ptr() as *mut c_void,
            out.as_mut_ptr() as *mut c_void,
            tokens,
            channels,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_gdn_causal_conv (multi-token)") {
        return;
    }

    let (ref_out, ref_state) =
        cpu_gdn_causal_conv(&projected, &conv_weight, &state_in, tokens, channels);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.01;
        assert!(
            (got - want).abs() <= tol,
            "gdn_conv_out[{i}]: got {got}, want {want}"
        );
    }
    // The 3-tap state update (s0, s1, s2) = the last 3 consumed taps (no
    // arithmetic — a bit-exact pin on the bf16 values).
    for (i, &got_bits) in state_out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_state[i];
        assert_eq!(got, want, "gdn_conv_state[{i}]: got {got}, want {want}");
    }
}

/// `tokens == 1` GDN causal conv (the GEMV special case): the single-token
/// conv + the 3-tap state update (the state becomes `(s1_in, s2_in, p0)`).
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): run with -- --ignored"]
fn gdn_causal_conv_single_token_gpu() {
    let (projected, conv_weight, state_in, _tokens, channels) = synthetic_gdn_conv();
    let p_single = &projected[0..channels as usize]; // the first (only) token's row
    let projected_bf16 = to_bf16(p_single);
    let conv_weight_bf16 = to_bf16(&conv_weight);
    let state_in_bf16 = to_bf16(&state_in);
    let mut out = vec![0u16; channels as usize];
    let mut state_out = vec![0u16; (channels * 3) as usize];

    let rc = unsafe {
        ffi::ignis_gdn_causal_conv(
            projected_bf16.as_ptr() as *const c_void,
            conv_weight_bf16.as_ptr() as *const c_void,
            state_in_bf16.as_ptr() as *const c_void,
            state_out.as_mut_ptr() as *mut c_void,
            out.as_mut_ptr() as *mut c_void,
            1,
            channels,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_gdn_causal_conv (tokens=1)") {
        return;
    }

    let (ref_out, ref_state) = cpu_gdn_causal_conv(p_single, &conv_weight, &state_in, 1, channels);
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.01;
        assert!(
            (got - want).abs() <= tol,
            "gdn_conv_single_out[{i}]: got {got}, want {want}"
        );
    }
    // The single-token state update (s0, s1, s2) = (s1_in, s2_in, p0).
    for (i, &got_bits) in state_out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want = ref_state[i];
        assert_eq!(
            got, want,
            "gdn_conv_single_state[{i}]: got {got}, want {want}"
        );
    }
}

/// GQA RoPE (θ = 1e7, rotary_dim 64 of head_dim 256 — 32 pairs) vs the
/// CPU reference rotation (the fp64 sincos oracle). A `pos` sweep is
/// checked; `pos == 0` is the identity rotation (a bit-exact pin), and the
/// un-rotated dims `[rotary_dim, head_dim)` remain bit-exact for every pos.
#[test]
#[ignore = "GPU launch test — a few KB of VRAM, runs even with the model loaded (ADR 0006 nuance): run with -- --ignored"]
fn rope_qk_pos_sweep_gpu() {
    let (q, k, batch, seq, num_q, num_kv, head_dim, rotary) = synthetic_rope_qk();
    const THETA: f64 = 1e7;
    let inv_freq_f32 = rope_inv_frequencies(THETA, rotary); // the kernel's f32 table
    let q_bf16_in = to_bf16(&q);
    let k_bf16_in = to_bf16(&k);
    let ref_inv_freq = cpu_rope_inv_freqs(THETA, rotary); // the fp64 oracle table

    for &pos in &[0i32, 1, 5, 17, 255, 4095] {
        let mut q_bf16 = q_bf16_in.clone();
        let mut k_bf16 = k_bf16_in.clone();
        let rc = unsafe {
            ffi::ignis_rope_qk(
                q_bf16.as_mut_ptr() as *mut c_void,
                k_bf16.as_mut_ptr() as *mut c_void,
                inv_freq_f32.as_ptr() as *const c_void,
                batch,
                seq,
                num_q,
                num_kv,
                head_dim,
                rotary,
                pos,
                std::ptr::null_mut(),
            )
        };
        if skip_if_busy(rc, &format!("ignis_rope_qk (pos={pos})")) {
            return;
        }

        let (ref_q, ref_k) = cpu_rope_qk(
            &q,
            &k,
            &ref_inv_freq,
            batch,
            seq,
            num_q,
            num_kv,
            head_dim,
            rotary,
            pos,
        );

        if pos == 0 {
            // The identity rotation (cos = 1, sin = 0): q/k are bit-exact
            // (the un-rotated dims are never written).
            for (i, (got, want)) in q_bf16.iter().zip(q_bf16_in.iter()).enumerate() {
                assert_eq!(
                    *got, *want,
                    "rope_qk_identity_q[{i}]: got {got:#06x}, want {want:#06x}"
                );
            }
            for (i, (got, want)) in k_bf16.iter().zip(k_bf16_in.iter()).enumerate() {
                assert_eq!(
                    *got, *want,
                    "rope_qk_identity_k[{i}]: got {got:#06x}, want {want:#06x}"
                );
            }
        } else {
            for (i, &got_bits) in q_bf16.iter().enumerate() {
                let got = from_bf16(got_bits);
                let want = ref_q[i];
                let tol = want.abs().max(1.0) * 0.01;
                assert!(
                    (got - want).abs() <= tol,
                    "rope_qk_q[{i}] (pos={pos}): got {got}, want {want}"
                );
            }
            for (i, &got_bits) in k_bf16.iter().enumerate() {
                let got = from_bf16(got_bits);
                let want = ref_k[i];
                let tol = want.abs().max(1.0) * 0.01;
                assert!(
                    (got - want).abs() <= tol,
                    "rope_qk_k[{i}] (pos={pos}): got {got}, want {want}"
                );
            }
        }
    }
}

// --- CPU-only (no GPU) argument-validation pins --------------------------------

/// Invalid args are rejected with -1 *before* any CUDA call, so these run
/// on CPU (no GPU needed) and pin the surface's validation contract.
#[test]
fn gdn_causal_conv_rejects_invalid_args() {
    let projected = vec![0u16; 512];
    let conv_weight = vec![0u16; 4 * 64];
    let state_in = vec![0u16; 3 * 64];
    let mut state_out = vec![0u16; 3 * 64];
    let mut out = vec![0u16; 512];
    let proj = projected.as_ptr() as *const c_void;
    let cw = conv_weight.as_ptr() as *const c_void;
    let si = state_in.as_ptr() as *const c_void;
    let so = state_out.as_mut_ptr() as *mut c_void;
    let o = out.as_mut_ptr() as *mut c_void;

    // tokens = 0 -> invalid.
    let rc =
        unsafe { ffi::ignis_gdn_causal_conv(proj, cw, si, so, o, 0, 64, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "tokens=0 must be rejected");
    // channels = 0 -> invalid.
    let rc = unsafe { ffi::ignis_gdn_causal_conv(proj, cw, si, so, o, 8, 0, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "channels=0 must be rejected");
    // null projected -> invalid.
    let rc = unsafe {
        ffi::ignis_gdn_causal_conv(std::ptr::null(), cw, si, so, o, 8, 64, std::ptr::null_mut())
    };
    assert_eq!(rc, -1, "null projected must be rejected");
    // null state_out -> invalid.
    let rc = unsafe {
        ffi::ignis_gdn_causal_conv(
            proj,
            cw,
            si,
            std::ptr::null_mut(),
            o,
            8,
            64,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "null state_out must be rejected");
}

/// RoPE invalid args are rejected with -1 *before* any CUDA call, so these
/// run on CPU (no GPU needed) and pin the surface's validation contract.
#[test]
fn rope_qk_rejects_invalid_args() {
    let mut q = vec![0u16; 256];
    let mut k = vec![0u16; 256];
    let inv = rope_inv_frequencies(1e7, 64);
    let qp = q.as_mut_ptr() as *mut c_void;
    let kp = k.as_mut_ptr() as *mut c_void;
    let ip = inv.as_ptr() as *const c_void;

    // odd rotary_dim -> invalid.
    let rc =
        unsafe { ffi::ignis_rope_qk(qp, kp, ip, 1, 1, 1, 1, 256, 63, 0, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "odd rotary_dim must be rejected");
    // rotary_dim > head_dim -> invalid.
    let rc =
        unsafe { ffi::ignis_rope_qk(qp, kp, ip, 1, 1, 1, 1, 256, 320, 0, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "rotary_dim > head_dim must be rejected");
    // rotary_dim = 0 -> invalid.
    let rc = unsafe { ffi::ignis_rope_qk(qp, kp, ip, 1, 1, 1, 1, 256, 0, 0, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "rotary_dim=0 must be rejected");
    // null q -> invalid.
    let rc = unsafe {
        ffi::ignis_rope_qk(
            std::ptr::null_mut(),
            kp,
            ip,
            1,
            1,
            1,
            1,
            256,
            64,
            0,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, -1, "null q must be rejected");
    // batch = 0 -> invalid.
    let rc =
        unsafe { ffi::ignis_rope_qk(qp, kp, ip, 0, 1, 1, 1, 256, 64, 0, std::ptr::null_mut()) };
    assert_eq!(rc, -1, "batch=0 must be rejected");
}
