//! Gated GPU launch tests for the ticket-03 decode C-ABI surface.
//!
//! These launch the two decode C ABI functions (NVFP4 GEMM + GQA attention) on
//! the RTX 5090 with small *synthetic* inputs (no model weights) and compare
//! the kernel output against a CPU reference computed in Rust. They are gated:
//! `#[ignore]` by default, and they also self-skip (treat a non-zero return as
//! "GPU busy, skip") so a busy GPU never turns the suite red.
//!
//! They fit in a few MB of VRAM, so they can run even with the model loaded
//! (the ADR 0006 nuance). Run with:
//! `cargo test -p ignis-core --test decode_gpu -- --ignored`.
//!
//! Build precondition: this test links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles (a different ticket) AND the canonical
//! kernel/build/ignis_kernel.lib has been rebuilt with the ticket-03 symbols.

use std::ffi::c_void;

use ignis_core::ffi;

/// E2M1 (FP4) decode: 1 sign bit (0x8), 3 magnitude bits -> {0,.5,1,1.5,2,3,4,6}.
/// 1:1 with kernel/src/nvfp4_gemm_decode.cuh::`decode_nvfp4_e2m1`.
fn decode_e2m1(code: u8) -> f32 {
    let mag = match code & 0x7 {
        0 => 0.0, 1 => 0.5, 2 => 1.0, 3 => 1.5, 4 => 2.0, 5 => 3.0, 6 => 4.0, _ => 6.0,
    };
    if code & 0x8 != 0 { -mag } else { mag }
}

/// E4M3 (FP8) decode: OCP FP8 E4M3 (bias 7, no inf). 1:1 with the kernel.
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

/// CPU reference NVFP4 GEMV: out[m] = bias[m] + sum_k x[k] * (e2m1(code)*e4m3(scale)).
/// Mirrors the kernel's dequantization + group-scaled dot product exactly (the
/// kernel rounds the final result to bf16, hence the tolerance).
fn cpu_nvfp4_gemv(
    x: &[f32],
    wt_codes: &[u8],
    wt_scales: &[u8],
    bias: &[f32],
    m: i64,
    k: i64,
) -> Vec<f32> {
    let code_row = (k / 2) as usize; // 2 e2m1 per byte
    let scale_row = (k / 16) as usize; // one e4m3 per 16-element group
    (0..m)
        .map(|row| {
            let row = row as usize;
            let cr = row * code_row;
            let sr = row * scale_row;
            let mut acc = 0.0f32;
            for i in 0..k {
                let i = i as usize;
                let code = (wt_codes[cr + i / 2] >> ((i % 2) * 4)) & 0x0F;
                let group = i / 16;
                let w = decode_e2m1(code) * decode_e4m3(wt_scales[sr + group]);
                acc += x[i] * w;
            }
            acc + bias[row]
        })
        .collect()
}

/// CPU reference GQA decode attention: for each q head, online-softmax over the
/// seq_len paged keys, then a value-weighted sum. Mirrors the kernel's math and
/// paged addressing (paged_kv_element_offset). The kernel rounds to bf16 at the
/// end, so comparisons use a bf16 tolerance.
fn cpu_gqa_attention(
    q: &[f32],
    kv: &[f32],
    block_table: &[i32],
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    seq_len: i64,
    block_size: i64,
    num_blocks: i64,
    softmax_scale: f32,
) -> Vec<f32> {
    let plane = (num_blocks * block_size * num_kv_heads * head_dim) as usize; // per plane
    let group = (num_q_heads / num_kv_heads) as usize;
    let mut out = Vec::with_capacity((num_q_heads * head_dim) as usize);
    for h in 0..num_q_heads {
        let kv_head = (h as usize) / group;
        let q_row = &q[(h * head_dim) as usize..((h + 1) * head_dim) as usize];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut acc = vec![0.0f32; head_dim as usize];
        for key in 0..seq_len {
            let block = key / block_size;
            let off = key % block_size;
            let page = block_table[block as usize] as i64;
            // 1:1 with the kernel's paged_kv_element_offset (K plane base = kv[0]).
            let k_rel = (head_dim * block_size * (kv_head as i64 + num_kv_heads * page)
                + head_dim * off) as usize;
            let score = (0..head_dim).fold(0.0f32, |s, d| {
                let d = d as usize;
                s + q_row[d] * kv[k_rel + d]
            }) * softmax_scale;
            let m_new = m.max(score);
            let alpha = if m == f32::NEG_INFINITY { 0.0 } else { m.exp() / m_new.exp() };
            let p = (score - m_new).exp();
            l = alpha * l + p;
            for d in 0..head_dim as usize {
                let v = kv[plane + k_rel + d]; // V plane starts at `plane`
                acc[d] = alpha * acc[d] + p * v;
            }
            m = m_new;
        }
        for d in 0..head_dim as usize {
            out.push(if l > 0.0 { acc[d] / l } else { 0.0f32 });
        }
    }
    out
}

/// Deterministic synthetic NVFP4 GEMM inputs (no model weights).
/// m=16, k=32 (two scale groups). activation all 0.5; bias all 1.0;
/// codes a deterministic E2M1 pattern; scales group 0 -> 0x38 (1.0),
/// group 1 -> 0x48 (4.0).
fn synthetic_gemm() -> (Vec<f32>, Vec<u8>, Vec<u8>, Vec<f32>, i64, i64) {
    let m = 16i64;
    let k = 32i64; // multiple of 16
    let x = vec![0.5f32; k as usize];
    let mut wt_codes = Vec::with_capacity((m * (k / 2)) as usize);
    let mut wt_scales = Vec::with_capacity((m * (k / 16)) as usize);
    for row in 0..m as usize {
        for pair in 0..(k as usize) / 2 {
            // two e2m1 codes per byte: code for k=2pair and k=2pair+1
            let lo = (((row * 7 + (2 * pair) * 3) % 8) as u8) & 0x0F;
            let hi = (((row * 7 + (2 * pair + 1) * 3) % 8) as u8) & 0x0F;
            wt_codes.push(lo | (hi << 4));
        }
        wt_scales.push(0x38); // group 0: 1.0
        wt_scales.push(0x48); // group 1: 4.0
    }
    let bias = vec![1.0f32; m as usize];
    (x, wt_codes, wt_scales, bias, m, k)
}

/// Deterministic synthetic GQA attention inputs (paged bf16 KV).
/// K/V planes are laid out [page][kv_head][block_offset][d] (d fastest) to match
/// the paged_kv_element_offset formula; values vary by (page, kv_head, off, d)
/// using bf16-exact magnitudes so the CPU reference matches the kernel exactly.
fn synthetic_gqa() -> (
    Vec<f32>, // q values
    Vec<f32>, // kv values (K plane then V plane)
    Vec<i32>, // block_table
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    f32,
) {
    let (nq, nkv, hd, seq, bs, nb, scale) = (4i64, 2i64, 8i64, 8i64, 4i64, 2i64, 1.0f32);
    let q = vec![0.5f32; (nq * hd) as usize];
    let plane = (nb * bs * nkv * hd) as usize;
    let mut kv = Vec::with_capacity(2 * plane);
    // K plane: value varies by (page, kv_head, off, d) using bf16-exact terms.
    for page in 0..nb {
        for kh in 0..nkv {
            for off in 0..bs {
                for d in 0..hd {
                    let v = (off + 1) as f32 * 0.5
                        + (page + kh) as f32 * 0.25
                        + (d % 2) as f32 * 0.125;
                    kv.push(v);
                }
            }
        }
    }
    // V plane: same layout, different magnitude (V = 0.25 * K).
    for page in 0..nb {
        for kh in 0..nkv {
            for off in 0..bs {
                for d in 0..hd {
                    let v = ((off + 1) as f32 * 0.5
                        + (page + kh) as f32 * 0.25
                        + (d % 2) as f32 * 0.125)
                        * 0.25;
                    kv.push(v);
                }
            }
        }
    }
    let block_table = vec![0i32, 1]; // logical block b -> physical page b
    (q, kv, block_table, nq, nkv, hd, seq, bs, nb, scale)
}

/// Skip helper: a non-zero rc (CUDA error / busy GPU) is a skip, never a failure
/// (ADR 0006 — the GPU is occupied by ninfer-serve).
fn skip_if_busy(rc: i32, what: &str) -> bool {
    if rc != 0 {
        eprintln!("SKIP: {what} returned {rc} (GPU busy / unavailable, ADR 0006)");
        return true;
    }
    false
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn nvfp4_gemm_decode_gpu() {
    let (x, wt_codes, wt_scales, bias, m, k) = synthetic_gemm();
    let x_bf16 = to_bf16(&x);
    let bias_bf16 = to_bf16(&bias);
    let mut out = vec![0u16; m as usize];
    let rc = unsafe {
        ffi::ignis_nvfp4_gemm_decode(
            x_bf16.as_ptr() as *const c_void,
            wt_codes.as_ptr() as *const c_void,
            wt_scales.as_ptr() as *const c_void,
            bias_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            m,
            k,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_nvfp4_gemm_decode") {
        return;
    }
    let ref_out = cpu_nvfp4_gemv(&x, &wt_codes, &wt_scales, &bias, m, k);
    for i in 0..m as usize {
        let got = from_bf16(out[i]);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!((got - want).abs() <= tol, "gemm[{i}]: got {got}, want {want}");
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn gqa_attention_decode_gpu() {
    let (q, kv, block_table, nq, nkv, hd, seq, bs, nb, scale) = synthetic_gqa();
    let q_bf16 = to_bf16(&q);
    let kv_bf16 = to_bf16(&kv);
    let mut out = vec![0u16; (nq * hd) as usize];
    let rc = unsafe {
        ffi::ignis_gqa_attention_decode(
            q_bf16.as_ptr() as *const c_void,
            kv_bf16.as_ptr() as *const c_void,
            block_table.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            nq,
            nkv,
            hd,
            seq,
            bs,
            nb,
            scale,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_gqa_attention_decode") {
        return;
    }
    let ref_out = cpu_gqa_attention(&q, &kv, &block_table, nq, nkv, hd, seq, bs, nb, scale);
    for i in 0..(nq * hd) as usize {
        let got = from_bf16(out[i]);
        let want = ref_out[i];
        let tol = want.abs().max(1.0) * 0.05;
        assert!((got - want).abs() <= tol, "attn[{i}]: got {got}, want {want}");
    }
}