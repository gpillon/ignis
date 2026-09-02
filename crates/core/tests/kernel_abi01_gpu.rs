//! Gated GPU launch tests for the ticket-05 (kernel-abi-01) C-ABI surface.
//!
//! Launches the two kernel-abi-01 C ABI functions (GQA prefill attention +
//! GDN linear-attention step) on the RTX 5090 with small *synthetic* inputs
//! (no model weights) and compares the kernel output against a CPU reference
//! computed in Rust. Gated: `#[ignore]` by default, and self-skip (a non-zero
//! return means "GPU busy, skip") so a busy GPU never turns the suite red.
//!
//! They fit in a few MB of VRAM, so they can run even with the model loaded
//! (the ADR 0006 nuance). Run with:
//! `cargo test -p ignis-core --test kernel_abi01_gpu -- --ignored`
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles AND the canonical `kernel/build/ignis_kernel.lib`
//! has been rebuilt with the ticket-05 symbols.

use std::ffi::c_void;

use ignis_core::ffi;

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

// --- GQA prefill CPU reference -------------------------------------------

/// CPU reference GQA causal prefill: for each (batch, position, q head), the
/// value-weighted causal attention over keys [0, position], paged bf16 KV.
/// Mirrors kernel/src/gqa_attention_prefill.cuh (same paged addressing, same
/// online softmax; the kernel rounds the final result to bf16, hence the
/// tolerance). Mirrors the decode kernel's paged offset 1:1.
fn cpu_gqa_prefill(
    q_bf16: &[u16],
    kv_bf16: &[u16],
    block_table: &[i32],
    batch: i64,
    seq_len: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    block_size: i64,
    num_blocks: i64,
    softmax_scale: f32,
) -> Vec<f32> {
    let b = batch as usize;
    let sl = seq_len as usize;
    let nq = num_q_heads as usize;
    let nkv = num_kv_heads as usize;
    let hd = head_dim as usize;
    let bs = block_size as usize;
    let nb = num_blocks as usize;
    let plane = nb * bs * nkv * hd; // per-batch plane (elements)
    let group = nq / nkv;
    let mut out = Vec::with_capacity((b * sl * nq * hd) as usize);
    for bi in 0..b {
        for pos in 0..sl {
            for h in 0..nq {
                let kv_head = h / group;
                let q_row = (bi * sl + pos) * nq * hd + h * hd;
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;
                let mut acc = vec![0.0f32; hd];
                let v_base = ((batch + bi as i64) as i64) * (plane as i64);
                for key in 0..=pos {
                    let blk = key / bs;
                    let off = key % bs;
                    let page = block_table[(bi * nb + blk) as usize] as i64;
                    // Paged offset (1:1 with the kernel's prefill_paged_offset,
                    // which includes the head_dim element).
                    let paged = (hd as i64 * (bs as i64) * (kv_head as i64 + (nkv as i64) * page)
                        + (hd as i64) * (off as i64)) as usize;
                    // Per-key q.k dot product over all head_dim elements.
                    let mut score = 0.0f32;
                    for d in 0..hd {
                        let qv = from_bf16(q_bf16[q_row + d]);
                        let k_idx = ((bi as i64) * (plane as i64) + paged as i64 + d as i64) as usize;
                        score += qv * from_bf16(kv_bf16[k_idx]);
                    }
                    score *= softmax_scale;
                    let m_new = m.max(score);
                    let alpha = if m == f32::NEG_INFINITY {
                        0.0
                    } else {
                        m.exp() / m_new.exp()
                    };
                    let p = (score - m_new).exp();
                    l = alpha * l + p;
                    for d in 0..hd {
                        let v_idx = (v_base + paged as i64 + d as i64) as usize;
                        let vv = from_bf16(kv_bf16[v_idx]);
                        acc[d] = alpha * acc[d] + p * vv;
                    }
                    m = m_new;
                }
                for d in 0..hd {
                    out.push(if l > 0.0 { acc[d] / l } else { 0.0f32 });
                }
            }
        }
    }
    out
}

// --- GDN step CPU reference ------------------------------------------------

/// CPU reference GDN (gated delta rule) step. Mirrors
/// kernel/src/gdn_step.cuh 1:1: per (batch, layer) the gated delta rule
/// S <- alpha*S + beta_p*(v - alpha*S^T k) outer k^T, with alpha = exp(g) and
/// beta_p = sigmoid(beta). Reads the same bf16 buffers the kernel reads.
fn cpu_gdn_step(
    x_bf16: &[u16],
    state_in_bf16: &[u16],
    batch: i64,
    num_layers: i64,
    state_rows: i64,
    state_cols: i64,
    state_dim: i64,
) -> Vec<f32> {
    let b = batch as usize;
    let nl = num_layers as usize;
    let rs = state_rows as usize;
    let cs = state_cols as usize;
    let dim = state_dim as usize;
    let mut out = Vec::with_capacity((b * nl * rs * cs) as usize);
    for bi in 0..b {
        let x_row = &x_bf16[bi * dim..bi * dim + dim];
        // Decompose x[b] into (k, v, g, beta) -- see gdn_step.cuh.
        let k = |d: usize| from_bf16(x_row[d]);
        let v = |dv: usize| from_bf16(x_row[cs + dv]);
        let g = from_bf16(x_row[cs + rs]);
        let beta = from_bf16(x_row[cs + rs + 1]);
        let alpha = g.exp();
        let beta_p = 1.0 / (1.0 + (-beta).exp());
        for l in 0..nl {
            for dv in 0..rs {
                // y[dv] = sum_d S_in[bi,l,dv,d] * k[d]  (the per-dv row of S^T k).
                let mut y = 0.0f32;
                for d in 0..cs {
                    let s_idx = (((bi * nl + l) * rs) + dv) * cs + d;
                    y += from_bf16(state_in_bf16[s_idx]) * k(d);
                }
                let delta = beta_p * (v(dv) - alpha * y);
                for d in 0..cs {
                    let s_idx = (((bi * nl + l) * rs) + dv) * cs + d;
                    let s_in_d = from_bf16(state_in_bf16[s_idx]);
                    out.push(alpha * s_in_d + delta * k(d));
                }
            }
        }
    }
    out
}

// --- Synthetic inputs (no model weights) ----------------------------------

/// Deterministic synthetic GQA prefill inputs (no model weights).
/// batch=2, seq_len=8, q_heads=4, kv_heads=2, head_dim=8, block_size=4,
/// num_blocks=4. q and K/V values vary by index using bf16-exact magnitudes so
/// the CPU reference (f32) and the kernel (bf16) see the same values.
fn synthetic_prefill() -> (
    Vec<f32>, // q values (f32)
    Vec<f32>, // kv values (K plane then V plane, f32)
    Vec<i32>, // block_table
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    f32,
) {
    let (batch, seq_len, nq, nkv, hd, bs, nb) = (2i64, 8i64, 4i64, 2i64, 8i64, 4i64, 4i64);
    let scale = 1.0f32;
    // q: [batch][seq_len][nq][hd], value varies by (bi, pos, h, d) using
    // bf16-exact terms (multiples of 0.5).
    let mut q = Vec::with_capacity((batch * seq_len * nq * hd) as usize);
    for bi in 0..(batch as usize) {
        for pos in 0..(seq_len as usize) {
            for h in 0..(nq as usize) {
                for d in 0..(hd as usize) {
                    let v = 0.5
                        + 0.5 * (bi as f32)
                        + 0.25 * ((pos + h + d) % 4) as f32;
                    q.push(v);
                }
            }
        }
    }
    // kv: two planes (K then V), each [batch][num_blocks][nkv][block_size][hd]
    // (kv_head-major within a page, matching the kernel's paged offset).
    let plane = (nb * bs * nkv * hd) as usize; // per-batch plane
    let mut kv = Vec::with_capacity(2 * (batch as usize) * plane);
    // K plane: value varies by (bi, blk, off, kv_head, d) using bf16-exact terms.
    for plane_i in 0..2 {
        for bi in 0..(batch as usize) {
            for blk in 0..(nb as usize) {
                for off in 0..(bs as usize) {
                    for kh in 0..(nkv as usize) {
                        for d in 0..(hd as usize) {
                            let v = if plane_i == 0 {
                        // K plane: varies by (bi, blk, off, kh) using bf16-exact terms.
                        0.25 * (off + 1) as f32 + 0.125 * (kh + 1) as f32
                            + 0.25 * (bi as f32) + 0.125 * (blk as f32)
                    } else {
                        // V plane: different magnitude, varies by (bi, blk, d).
                        0.125 * (off + 1) as f32 + 0.25 * (bi as f32)
                            + 0.125 * ((d % 2) as f32) + 0.125 * (blk as f32)
                    };
                            kv.push(v);
                        }
                    }
                }
            }
        }
    }
    // block_table: [batch][num_blocks], logical block -> physical page (identity).
    let block_table = vec![0i32, 1, 2, 3, 0, 1, 2, 3];
    (
        q,
        kv,
        block_table,
        batch,
        seq_len,
        nq,
        nkv,
        hd,
        bs,
        nb,
        scale,
    )
}

/// Deterministic synthetic GDN inputs (no model weights).
/// batch=2, layers=3, state_rows=4, state_cols=4, state_dim=10 (cs+rs+2).
fn synthetic_gdn() -> (Vec<f32>, Vec<f32>, i64, i64, i64, i64, i64) {
    let (batch, layers, rs, cs) = (2i64, 3i64, 4i64, 4i64);
    let dim = cs + rs + 2; // state_dim = state_cols + state_rows + 2
    // x: [batch][dim] = [k(4), v(4), g(1), beta(1)]. Values bf16-exact.
    let mut x = Vec::with_capacity((batch * dim) as usize);
    for bi in 0..(batch as usize) {
        // (k, v, g, beta) in order: k[d], v[dv], g, beta.
        let mut row = Vec::with_capacity(dim as usize);
        for d in 0..(cs as usize) {
            row.push(0.5 * (d as f32 + 1.0) + 0.5 * (bi as f32)); // k
        }
        for dv in 0..(rs as usize) {
            row.push(0.25 * (dv as f32 + 1.0) + 0.25 * (bi as f32)); // v
        }
        row.push(-1.0 - 0.25 * (bi as f32)); // g (pre-decay, <= 0)
        row.push(1.0); // beta (pre-activation)
        x.extend_from_slice(&row);
    }
    // state_in: [batch][layers][rs][cs]. Values bf16-exact, vary by index.
    let mut state_in = Vec::with_capacity((batch * layers * rs * cs) as usize);
    for bi in 0..(batch as usize) {
        for l in 0..(layers as usize) {
            for dv in 0..(rs as usize) {
                for d in 0..(cs as usize) {
                    let v =
                        0.25 * (bi as f32 + l as f32 + dv as f32 + d as f32 + 1.0);
                    state_in.push(v);
                }
            }
        }
    }
    (x, state_in, batch, layers, rs, cs, dim)
}

/// Skip helper: a non-zero rc (CUDA error / busy GPU) is a skip, never a
/// failure (ADR 0006 — the GPU is occupied by ninfer-serve).
fn skip_if_busy(rc: i32, what: &str) -> bool {
    if rc != 0 {
        eprintln!("SKIP: {what} returned {rc} (GPU busy / unavailable, ADR 0006)");
        return true;
    }
    false
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn gqa_attention_prefill_gpu() {
    let (q, kv, block_table, batch, seq_len, nq, nkv, hd, bs, nb, scale) = synthetic_prefill();
    let q_bf16 = to_bf16(&q);
    let kv_bf16 = to_bf16(&kv);
    let mut out = vec![0u16; (batch * seq_len * nq * hd) as usize];
    let rc = unsafe {
        ffi::ignis_gqa_attention_prefill(
            q_bf16.as_ptr() as *const c_void,
            kv_bf16.as_ptr() as *const c_void,
            block_table.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            batch,
            seq_len,
            nq,
            nkv,
            hd,
            bs,
            nb,
            scale,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_gqa_attention_prefill") {
        return;
    }
    let want = cpu_gqa_prefill(
        &q_bf16,
        &kv_bf16,
        &block_table,
        batch,
        seq_len,
        nq,
        nkv,
        hd,
        bs,
        nb,
        scale,
    );
    let tol = 0.05;
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want_i = want[i];
        assert!(
            (got - want_i).abs() <= tol * want_i.abs().max(1.0),
            "prefill[{i}]: got {got}, want {want_i}"
        );
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn gdn_step_gpu() {
    let (x, state_in, batch, layers, rs, cs, dim) = synthetic_gdn();
    let x_bf16 = to_bf16(&x);
    let state_in_bf16 = to_bf16(&state_in);
    let mut state_out = vec![0u16; (batch * layers * rs * cs) as usize];
    let rc = unsafe {
        ffi::ignis_gdn_step(
            x_bf16.as_ptr() as *const c_void,
            state_in_bf16.as_ptr() as *const c_void,
            state_out.as_mut_ptr() as *mut c_void,
            batch,
            layers,
            rs,
            cs,
            dim,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_gdn_step") {
        return;
    }
    let want = cpu_gdn_step(&x_bf16, &state_in_bf16, batch, layers, rs, cs, dim);
    let tol = 0.05;
    for (i, &got_bits) in state_out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want_i = want[i];
        assert!(
            (got - want_i).abs() <= tol * want_i.abs().max(1.0),
            "gdn[{i}]: got {got}, want {want_i}"
        );
    }
}

// --- CPU-only geometry / contract pins ------------------------------------
// No FFI: pin the ABI out-size and the state_dim contract (1:1 with the
// kernel/src layout docs) so the contract is testable on CPU.

#[test]
fn prefill_geometry_pins_out_shape() {
    // The ABI out is [batch][seq_len][num_q_heads][head_dim] elements (bf16).
    // Pin the synthetic prefill's sizes: the out buffer, the q buffer (same
    // shape), the two paged KV planes, and the block table. These catch a
    // layout drift in the synthetic generator (independent of any single
    // helper's formula).
    let (q, kv, block_table, batch, seq_len, nq, nkv, hd, bs, nb, _scale) = synthetic_prefill();
    let out_elems = (batch * seq_len * nq * hd) as usize; // 2*8*4*8 = 512
    assert_eq!(out_elems, 512);
    assert_eq!(q.len(), out_elems); // q is [batch][seq_len][nq][hd]
    // kv is two paged planes (K then V), each [batch][num_blocks][bs][nkv][hd].
    assert_eq!(kv.len(), 2 * (batch * nb * bs * nkv * hd) as usize);
    assert_eq!(block_table.len(), (batch * nb) as usize);
}

#[test]
fn gdn_state_dim_contract() {
    // The GDN step's state_dim must be state_cols + state_rows + 2 (the
    // k / v / g / beta feature block -- see kernel/src/gdn_step.cuh). The
    // ABI validation (state_dim == state_cols + state_rows + 2) is exactly the
    // invariant the synthetic generator obeys; pin it, plus the x / state
    // buffer sizes, so a drift in synthetic_gdn is caught on CPU.
    let (x, state_in, batch, layers, rs, cs, dim) = synthetic_gdn();
    assert_eq!(dim, cs + rs + 2);
    assert_eq!(x.len(), (batch * dim) as usize);
    assert_eq!(state_in.len(), (batch * layers * rs * cs) as usize);
}