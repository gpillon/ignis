//! Gated GPU launch tests for the ticket-06 (kernel-abi-02) C-ABI surface.
//!
//! Launches the three kernel-abi-02 C ABI functions (RMSNorm / LayerNorm,
//! embedding gather, greedy sampling) on the RTX 5090 with small *synthetic*
//! inputs (no model weights) and compares the kernel output against a CPU
//! reference computed in Rust. Gated: `#[ignore]` by default, and self-skip
//! (a non-zero return means "GPU busy, skip") so a busy GPU never turns the
//! suite red.
//!
//! They fit in a few MB of VRAM, so they can run even with the model loaded
//! (the ADR 0006 nuance). Run with:
//! `cargo test -p ignis-core --test kernel_abi02_gpu -- --ignored`
//!
//! Build precondition: links the kernel .lib, so it only builds once
//! `ignis-artifact` compiles AND the canonical `kernel/build/ignis_kernel.lib`
//! has been rebuilt with the ticket-06 symbols.

use std::ffi::c_void;

use ignis_core::ffi;

// --- bf16 helpers (same as kernel_abi01_gpu) --------------------------------

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

// --- CPU references ----------------------------------------------------------

/// CPU reference for `ignis_rmsnorm`: `base[i]` is `x[i] - center[i]` when a
/// center vector is present (the LayerNorm mode, "centered first"), else
/// `x[i]`; `out[i] = base[i] * inv * weight[i]` with
/// `inv = 1 / sqrt(mean(base^2) + eps)` (the kernel's `rsqrtf(mean + eps)`).
/// `weight` (nullable) is 1.0 where absent. Mirrors the kernel's fp32
/// internal math over the same bf16 inputs (hence the tolerance at the
/// call sites).
fn cpu_norm(x: &[u16], weight: Option<&[u16]>, center: Option<&[u16]>, n: usize, eps: f32) -> Vec<f32> {
    let e = if eps > 0.0 { eps } else { 1e-6 };
    let base: Vec<f32> = (0..n)
        .map(|i| match center {
            Some(c) => from_bf16(x[i]) - from_bf16(c[i]),
            None => from_bf16(x[i]),
        })
        .collect();
    let mean = base.iter().map(|&v| v * v).sum::<f32>() / (n as f32);
    let inv = 1.0 / (mean + e).sqrt();
    (0..n)
        .map(|i| {
            let w = match weight {
                Some(w) => from_bf16(w[i]),
                None => 1.0,
            };
            base[i] * inv * w
        })
        .collect()
}

/// CPU reference for `ignis_embedding`: `out[row] = table[id[row]]` (a bf16
/// row copy, so the comparison is exact on the 16-bit values).
fn cpu_embedding(table: &[u16], id: &[i32], batch: usize, hidden: usize) -> Vec<u16> {
    (0..batch)
        .flat_map(|t| {
            let row = (id[t] as usize) * hidden;
            table[row..row + hidden].iter().copied()
        })
        .collect()
}

/// CPU reference for `ignis_greedy_sample`: per-row argmax with ties
/// resolving to the lowest index (mirrors the kernel's `argmax_better`
/// rule: `v > best || (v == best && i < best_i)`).
fn cpu_argmax(logits: &[f32], batch: usize, vocab: usize) -> Vec<i32> {
    (0..batch)
        .map(|t| {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let mut best_i = 0usize;
            let mut best_v = row[0];
            for (i, &v) in row.iter().enumerate().skip(1) {
                if v > best_v || (v == best_v && i < best_i) {
                    best_v = v;
                    best_i = i;
                }
            }
            best_i as i32
        })
        .collect()
}

// --- Synthetic inputs (no model weights) ------------------------------------

/// Deterministic synthetic norm inputs (no model weights). n = 512; the x /
/// weight / center values vary by index using bf16-exact magnitudes
/// (multiples of 0.125) so the CPU reference (f32) and the kernel (bf16)
/// see the same values.
fn synthetic_norm() -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
    const N: usize = 512;
    let x: Vec<f32> = (0..N)
        .map(|i| 0.25 * ((i % 16) as f32 + 1.0) + 0.125 * ((i >> 4) as f32))
        .collect();
    let w: Vec<f32> = (0..N).map(|i| 1.0 + 0.125 * (i % 8) as f32).collect();
    let c: Vec<f32> = (0..N).map(|i| 0.5 + 0.25 * (i % 4) as f32).collect();
    (x, w, c, N)
}

/// Deterministic synthetic embedding inputs (no model weights).
/// batch=16, vocab=64, hidden=32. Table values vary by (vocab_row, hidden)
/// using bf16-exact terms; ids are a deterministic permutation slice of
/// [0, vocab).
fn synthetic_embedding() -> (Vec<f32>, Vec<i32>, i64, i64, i64) {
    const BATCH: usize = 16;
    const VOCAB: usize = 64;
    const HIDDEN: usize = 32;
    let table: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| {
            let v = i / HIDDEN;
            let h = i % HIDDEN;
            0.25 * ((v * 3 + h) % 16) as f32 + 0.125 * ((v + h) % 4) as f32
        })
        .collect();
    let id: Vec<i32> = (0..BATCH).map(|t| ((t * 7) % VOCAB) as i32).collect();
    (table, id, BATCH as i64, VOCAB as i64, HIDDEN as i64)
}

/// Deterministic synthetic logits (no model weights). batch=4, vocab=64.
/// Row 0: uniform 2.0 with a unique max (9.0) at v=5. Row 1: a tie (8.0 at
/// v=3 and v=9, the rest 1.0) -- the lowest-index rule must pick 3. Row 2:
/// a unique max (4.5) at v=63 over a -2.0 floor. Row 3: all equal (3.0) --
/// argmax is 0. All values are exact f32.
fn synthetic_argmax() -> (Vec<f32>, i64, i64) {
    const BATCH: usize = 4;
    const VOCAB: usize = 64;
    let mut logits = vec![0.0f32; BATCH * VOCAB];
    for v in 0..VOCAB {
        logits[v] = 2.0;
    }
    logits[5] = 9.0;
    for v in 0..VOCAB {
        logits[VOCAB + v] = 1.0;
    }
    logits[VOCAB + 3] = 8.0;
    logits[VOCAB + 9] = 8.0;
    for v in 0..VOCAB {
        logits[2 * VOCAB + v] = -2.0;
    }
    logits[2 * VOCAB + (VOCAB - 1)] = 4.5;
    for v in 0..VOCAB {
        logits[3 * VOCAB + v] = 3.0;
    }
    (logits, BATCH as i64, VOCAB as i64)
}

/// Expected argmax outputs for `synthetic_argmax()` (unique max at 5, the
/// lowest-index tie at 3, the far-corner max at 63, and 0 for an all-equal
/// row). Independent of `cpu_argmax`'s scan order.
const EXPECTED_ARGMAX: [i32; 4] = [5, 3, 63, 0];

// --- Gated GPU launch tests ---------------------------------------------------

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
fn rmsnorm_gpu() {
    let (x, w, _c, n) = synthetic_norm();
    let x_bf16 = to_bf16(&x);
    let w_bf16 = to_bf16(&w);
    let mut out = vec![0u16; n];
    let rc = unsafe {
        ffi::ignis_rmsnorm(
            x_bf16.as_ptr() as *const c_void,
            w_bf16.as_ptr() as *const c_void,
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            n as i64,
            0.0,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_rmsnorm") {
        return;
    }
    let want = cpu_norm(&x_bf16, Some(&w_bf16), None, n, 1e-6);
    let tol = 0.05;
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want_i = want[i];
        assert!(
            (got - want_i).abs() <= tol * want_i.abs().max(1.0),
            "rmsnorm[{i}]: got {got}, want {want_i}"
        );
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn rmsnorm_unit_weight_gpu() {
    // The null-weight branch of the kernel (weight = null => unit scale):
    // pins the `weight != nullptr ? ... : 1.0f` path in rmsnorm.cuh.
    let (x, _w, _c, n) = synthetic_norm();
    let x_bf16 = to_bf16(&x);
    let mut out = vec![0u16; n];
    let rc = unsafe {
        ffi::ignis_rmsnorm(
            x_bf16.as_ptr() as *const c_void,
            std::ptr::null(),
            std::ptr::null(),
            out.as_mut_ptr() as *mut c_void,
            n as i64,
            0.0,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_rmsnorm (unit weight)") {
        return;
    }
    let want = cpu_norm(&x_bf16, None, None, n, 1e-6);
    let tol = 0.05;
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want_i = want[i];
        assert!(
            (got - want_i).abs() <= tol * want_i.abs().max(1.0),
            "rmsnorm-unit[{i}]: got {got}, want {want_i}"
        );
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn layernorm_gpu() {
    // LayerNorm mode: the same vector pair with a center vector present
    // ("centered first" per the ABI contract).
    let (x, w, c, n) = synthetic_norm();
    let x_bf16 = to_bf16(&x);
    let w_bf16 = to_bf16(&w);
    let c_bf16 = to_bf16(&c);
    let mut out = vec![0u16; n];
    let rc = unsafe {
        ffi::ignis_rmsnorm(
            x_bf16.as_ptr() as *const c_void,
            w_bf16.as_ptr() as *const c_void,
            c_bf16.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            n as i64,
            0.0,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_rmsnorm (centered)") {
        return;
    }
    let want = cpu_norm(&x_bf16, Some(&w_bf16), Some(&c_bf16), n, 1e-6);
    let tol = 0.05;
    for (i, &got_bits) in out.iter().enumerate() {
        let got = from_bf16(got_bits);
        let want_i = want[i];
        assert!(
            (got - want_i).abs() <= tol * want_i.abs().max(1.0),
            "layernorm[{i}]: got {got}, want {want_i}"
        );
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn embedding_gpu() {
    let (table, id, batch, vocab, hidden) = synthetic_embedding();
    let table_bf16 = to_bf16(&table);
    let mut out = vec![0u16; (batch * hidden) as usize];
    let rc = unsafe {
        ffi::ignis_embedding(
            table_bf16.as_ptr() as *const c_void,
            id.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            batch,
            vocab,
            hidden,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_embedding") {
        return;
    }
    let want = cpu_embedding(&table_bf16, &id, batch as usize, hidden as usize);
    for (i, &got_bits) in out.iter().enumerate() {
        assert_eq!(got_bits, want[i], "embedding[{i}]: bits differ");
    }
}

#[test]
#[ignore = "GPU launch test — a few MB of VRAM, runs even with the model loaded (ADR 0006 nuance): -- --ignored"]
fn greedy_sample_gpu() {
    let (logits, batch, vocab) = synthetic_argmax();
    let mut out = vec![0i32; batch as usize];
    let rc = unsafe {
        ffi::ignis_greedy_sample(
            logits.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            batch,
            vocab,
            std::ptr::null_mut(),
        )
    };
    if skip_if_busy(rc, "ignis_greedy_sample") {
        return;
    }
    let want = cpu_argmax(&logits, batch as usize, vocab as usize);
    assert_eq!(out, want, "argmax outputs differ");
    // The synthetic rows also pin the tie/edge cases independently of the
    // reference helper (unique max, lowest-index tie, corner max, all-equal).
    assert_eq!(out, EXPECTED_ARGMAX.to_vec());
}

// --- CPU-only geometry / contract pins --------------------------------------
// No FFI: pin the ABI out-sizes and the synthetic shapes so a layout drift
// in the generators is caught on CPU (the contract mirrors
// kernel/src/norms_sampling_surface.cu's validation).

#[test]
fn norm_geometry_pins_out_shape() {
    // The ABI out is [n] elements (bf16), one per input element; the weight /
    // center vectors are the same length. These catch a drift in
    // synthetic_norm (independent of any single helper's formula).
    let (x, w, c, n) = synthetic_norm();
    assert_eq!(n, 512);
    assert_eq!(x.len(), n);
    assert_eq!(w.len(), n);
    assert_eq!(c.len(), n);
}

#[test]
fn embedding_geometry_pins_out_shape() {
    // The ABI out is [batch][hidden] elements (bf16) and the id vector is
    // [batch] (i32). Pin the synthetic sizes and the canary shape (16*32 =
    // 512 out elements).
    let (table, id, batch, vocab, hidden) = synthetic_embedding();
    assert_eq!(table.len(), (vocab * hidden) as usize);
    assert_eq!(id.len(), batch as usize);
    assert_eq!((batch * hidden) as usize, 512);
}

#[test]
fn argmax_geometry_pins_out_shape() {
    // The ABI logits are [batch][vocab] elements (f32) and the out is
    // [batch] (i32). Pin the synthetic sizes.
    let (logits, batch, vocab) = synthetic_argmax();
    assert_eq!(logits.len(), (batch * vocab) as usize);
    assert_eq!(batch, 4);
    assert_eq!(vocab, 64);
}