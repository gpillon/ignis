//! The graphs' other ops, ours (spec speech/01 §2, ADR 0068):
//! LayerNorm, RMSNorm, GELU, SiLU, softmax, RoPE / M-RoPE, attention and the
//! 3×3 stride-2 convolution.
//!
//! Each op has one body for the scalar reference and the AVX2 path: the
//! elementwise and row ops compile the same Rust under `#[target_feature]`
//! (same bits, vectorised), and attention's inner products are written with
//! intrinsics. Each is tested against fp64 at the model's real geometry, with
//! a negative control that must fail (ADR 0043).

use std::cell::RefCell;

use crate::cpu::Isa;
use crate::math;
use crate::matmul::{matmul, Weight};
use crate::pool::Pool;
use crate::quant::{f16_to_f32, lanes_sum};

/// Defines `pub fn $name(isa, …)` over one `#[inline(always)]` body, compiled
/// plain for the scalar reference and under AVX2/FMA/F16C for the AVX2 path.
macro_rules! dual {
    ($(#[$m:meta])* fn $name:ident($($a:ident: $t:ty),*) $body:block) => {
        $(#[$m])*
        pub fn $name(isa: Isa, $($a: $t),*) {
            #[inline(always)]
            fn body($($a: $t),*) $body
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma,f16c")]
            fn avx2($($a: $t),*) {
                body($($a),*)
            }
            #[cfg(target_arch = "x86_64")]
            if isa.avx2() {
                // SAFETY: an AVX2 Isa exists only after detection.
                unsafe { avx2($($a),*) };
                return;
            }
            let _ = isa;
            body($($a),*)
        }
    };
}

/// Σ x, eight lanes then the kernels' horizontal order.
#[inline(always)]
fn sum(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut c = x.chunks_exact(8);
    for ch in &mut c {
        for l in 0..8 {
            acc[l] += ch[l];
        }
    }
    let mut s = lanes_sum(acc);
    for v in c.remainder() {
        s += v;
    }
    s
}

#[inline(always)]
fn sum_sq_dev(x: &[f32], mean: f32) -> f32 {
    let mut acc = [0f32; 8];
    let mut c = x.chunks_exact(8);
    for ch in &mut c {
        for l in 0..8 {
            let d = ch[l] - mean;
            acc[l] += d * d;
        }
    }
    let mut s = lanes_sum(acc);
    for v in c.remainder() {
        s += (v - mean) * (v - mean);
    }
    s
}

dual! {
    /// LayerNorm with bias over rows of `w.len()`, in place (the encoder's, eps 1e-5).
    fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
        let d = w.len();
        for row in x.chunks_exact_mut(d) {
            let mean = sum(row) / d as f32;
            let var = sum_sq_dev(row, mean) / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..d {
                row[i] = (row[i] - mean) * inv * w[i] + b[i];
            }
        }
    }
}

dual! {
    /// RMSNorm over rows of `w.len()`, in place (the decoder's and its QK-norm, eps 1e-6).
    fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
        let d = w.len();
        for row in x.chunks_exact_mut(d) {
            let ms = sum_sq_dev(row, 0.0) / d as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for i in 0..d {
                row[i] = row[i] * inv * w[i];
            }
        }
    }
}

dual! {
    /// GELU with the exact erf, in place.
    fn gelu(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = math::gelu(*v);
        }
    }
}

dual! {
    /// SwiGLU's gate: `g = silu(g) · u`.
    fn silu_mul(g: &mut [f32], u: &[f32]) {
        for (a, &b) in g.iter_mut().zip(u) {
            *a = math::silu(*a) * b;
        }
    }
}

dual! {
    /// `x += y`.
    fn add(x: &mut [f32], y: &[f32]) {
        for (a, &b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }
}

dual! {
    /// `x[r][i] += b[i]` for every row of `b.len()`.
    fn add_bias(x: &mut [f32], b: &[f32]) {
        for row in x.chunks_exact_mut(b.len()) {
            for (a, &c) in row.iter_mut().zip(b) {
                *a += c;
            }
        }
    }
}

dual! {
    /// `x[r][i] = gelu(x[r][i] + b[i])` for every row of `b.len()`.
    fn add_bias_gelu(x: &mut [f32], b: &[f32]) {
        for row in x.chunks_exact_mut(b.len()) {
            for (a, &c) in row.iter_mut().zip(b) {
                *a = math::gelu(*a + c);
            }
        }
    }
}

dual! {
    /// Softmax of one row, in place.
    fn softmax(x: &mut [f32]) {
        softmax_body(x)
    }
}

#[inline(always)]
fn softmax_body(x: &mut [f32]) {
    let mut acc = [f32::NEG_INFINITY; 8];
    let mut c = x.chunks_exact(8);
    for ch in &mut c {
        for l in 0..8 {
            acc[l] = acc[l].max(ch[l]);
        }
    }
    let mut max = acc.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    for &v in c.remainder() {
        max = max.max(v);
    }
    for v in x.iter_mut() {
        *v = math::exp(*v - max);
    }
    let inv = 1.0 / sum(x);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// f32 → f16, round to nearest even: the KV cache's store. The AVX2 path's `vcvtps2ph` and the
/// scalar conversion give the same bits.
pub fn to_f16(isa: Isa, src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if isa.avx2() {
        // SAFETY: an AVX2 Isa exists only after detection; both slices hold src.len() elements.
        unsafe { to_f16_avx2(src, dst) };
        return;
    }
    let _ = isa;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = crate::quant::f32_to_f16(s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn to_f16_avx2(src: &[f32], dst: &mut [u16]) {
    use std::arch::x86_64::*;
    let n8 = src.len() / 8 * 8;
    for i in (0..n8).step_by(8) {
        // SAFETY: i + 8 <= len for both slices.
        unsafe {
            let h = _mm256_cvtps_ph::<0>(_mm256_loadu_ps(src.as_ptr().add(i)));
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        }
    }
    for i in n8..src.len() {
        dst[i] = crate::quant::f32_to_f16(src[i]);
    }
}

// ---------------------------------------------------------------- RoPE / M-RoPE

/// The rotation angles of a forward pass's rows: M-RoPE as the decoder GGUF
/// declares it (`rope.dimension_sections`, interleaved, θ), over three
/// position streams (time, height, width). Text and audio rows carry the same
/// position in all three, which makes it plain RoPE.
pub struct Rope {
    half: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// The f32 inverse frequencies θ^(−2i/d), as the checkpoint's code computes them.
    pub fn inv_freq(theta: f32, head_dim: usize) -> Vec<f32> {
        (0..head_dim / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32)).collect()
    }

    /// Which stream rotates frequency `i`: height on i ≡ 1 (mod 3) under 3·sections[1], width on
    /// i ≡ 2 (mod 3) under 3·sections[2], time otherwise (the interleaved layout).
    pub fn stream(i: usize, sections: [usize; 3]) -> usize {
        if i % 3 == 1 && i < 3 * sections[1] {
            1
        } else if i % 3 == 2 && i < 3 * sections[2] {
            2
        } else {
            0
        }
    }

    pub fn new(positions: &[[u32; 3]], sections: [usize; 3], theta: f32, head_dim: usize) -> Rope {
        let half = head_dim / 2;
        let inv = Rope::inv_freq(theta, head_dim);
        let (mut cos, mut sin) = (Vec::with_capacity(positions.len() * half), Vec::with_capacity(positions.len() * half));
        for p in positions {
            for (i, f) in inv.iter().enumerate() {
                // The angle is rounded to f32 as the checkpoint's code rounds it.
                let a = p[Rope::stream(i, sections)] as f32 * f;
                cos.push((a as f64).cos() as f32);
                sin.push((a as f64).sin() as f32);
            }
        }
        Rope { half, cos, sin }
    }

    /// Text positions `start..start + n`, all three streams equal.
    pub fn text(start: usize, n: usize, sections: [usize; 3], theta: f32, head_dim: usize) -> Rope {
        let pos: Vec<[u32; 3]> = (start..start + n).map(|p| [p as u32; 3]).collect();
        Rope::new(&pos, sections, theta, head_dim)
    }

    /// Rotates every head of every row of `x` (rows of `heads · 2·half`) in the NeoX layout:
    /// element i pairs with i + half.
    pub fn apply(&self, isa: Isa, x: &mut [f32], heads: usize) {
        rope_rows(isa, x, heads, self.half, &self.cos, &self.sin);
    }
}

dual! {
    fn rope_rows(x: &mut [f32], heads: usize, half: usize, cos: &[f32], sin: &[f32]) {
        let hd = 2 * half;
        for (r, row) in x.chunks_exact_mut(heads * hd).enumerate() {
            let (c, s) = (&cos[r * half..(r + 1) * half], &sin[r * half..(r + 1) * half]);
            for h in row.chunks_exact_mut(hd) {
                let (lo, hi) = h.split_at_mut(half);
                for i in 0..half {
                    let (x0, x1) = (lo[i], hi[i]);
                    lo[i] = x0 * c[i] - x1 * s[i];
                    hi[i] = x1 * c[i] + x0 * s[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------- attention

/// Keys and values: `t` positions of `hkv` heads of `hd`, f16 (the decoder's cache) or f32.
#[derive(Clone, Copy)]
pub enum Kv<'a> {
    F16 { k: &'a [u16], v: &'a [u16] },
    F32 { k: &'a [f32], v: &'a [f32] },
}

#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
}

/// Which keys query row i sees.
#[derive(Clone, Copy, Debug)]
pub enum Mask {
    /// Keys 0..=past + i (the decoder; the new rows' keys are already in the cache).
    Causal { past: usize },
    /// All `t` keys (the encoder, inside one window).
    Full,
}

thread_local! {
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Query rows of 8 per task.
const QROWS: usize = 8;

/// Scaled dot-product attention with GQA: `out[i][h] = softmax(q[i][h] · K[kvh]ᵀ · scale) · V[kvh]`,
/// kvh = h / (hq / hkv). `q` and `out` hold `m` rows of `hq · hd`.
#[allow(clippy::too_many_arguments)]
pub fn attention(pool: &Pool, isa: Isa, shape: AttnShape, q: &[f32], m: usize, kv: Kv, t: usize, mask: Mask, out: &mut [f32]) {
    let AttnShape { hq, hkv, hd } = shape;
    assert!(hq % hkv == 0 && hd % 8 == 0);
    assert!(q.len() >= m * hq * hd && out.len() >= m * hq * hd);
    let kvs = hkv * hd;
    match kv {
        Kv::F16 { k, v } => assert!(k.len() >= t * kvs && v.len() >= t * kvs),
        Kv::F32 { k, v } => assert!(k.len() >= t * kvs && v.len() >= t * kvs),
    }
    if let Mask::Causal { past } = mask {
        assert!(past + m <= t, "causal rows past the cache");
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let blocks = m.div_ceil(QROWS);
    let op = OutPtr(out.as_mut_ptr());
    pool.run(hq * blocks, &|task| {
        let (h, b) = (task / blocks, task % blocks);
        let kvh = h / (hq / hkv);
        SCORES.with(|cell| {
            let mut s = cell.borrow_mut();
            if s.len() < t {
                s.resize(t, 0.0);
            }
            for i in b * QROWS..((b + 1) * QROWS).min(m) {
                let len = match mask {
                    Mask::Causal { past } => past + i + 1,
                    Mask::Full => t,
                };
                let qr = &q[(i * hq + h) * hd..(i * hq + h + 1) * hd];
                // SAFETY: row i, head h of the output is this task's alone.
                let o = unsafe { std::slice::from_raw_parts_mut(op.get().add((i * hq + h) * hd), hd) };
                attend_row(isa, qr, kv, kvs, kvh * hd, len, scale, &mut s[..len], o);
            }
        });
    });
}

#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: each attention task writes disjoint output rows.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// One query row of one head against `len` keys at stride `kvs`, offset `off`.
#[allow(clippy::too_many_arguments)]
fn attend_row(isa: Isa, q: &[f32], kv: Kv, kvs: usize, off: usize, len: usize, scale: f32, s: &mut [f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if isa.is_avx2() {
        // SAFETY: an AVX2 Isa exists only after detection; the slices hold len rows at stride kvs.
        unsafe { attend_row_avx2(q, kv, kvs, off, len, scale, s, out) };
        return;
    }
    let _ = isa;
    let hd = q.len();
    let key = |j: usize, d: usize| match kv {
        Kv::F16 { k, .. } => f16_to_f32(k[j * kvs + off + d]),
        Kv::F32 { k, .. } => k[j * kvs + off + d],
    };
    let val = |j: usize, d: usize| match kv {
        Kv::F16 { v, .. } => f16_to_f32(v[j * kvs + off + d]),
        Kv::F32 { v, .. } => v[j * kvs + off + d],
    };
    for (j, sj) in s.iter_mut().enumerate() {
        let mut acc = [0f32; 8];
        for d0 in (0..hd).step_by(8) {
            for l in 0..8 {
                acc[l] += q[d0 + l] * key(j, d0 + l);
            }
        }
        *sj = lanes_sum(acc) * scale;
    }
    softmax_body(s);
    out.fill(0.0);
    for (j, &p) in s.iter().enumerate() {
        for (d, o) in out.iter_mut().enumerate() {
            *o += p * val(j, d);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
unsafe fn attend_row_avx2(q: &[f32], kv: Kv, kvs: usize, off: usize, len: usize, scale: f32, s: &mut [f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let hd = q.len();
    unsafe {
        let load = |base: *const u8, f16: bool, j: usize, d: usize| -> __m256 {
            if f16 {
                _mm256_cvtph_ps(_mm_loadu_si128((base as *const u16).add(j * kvs + off + d) as *const __m128i))
            } else {
                _mm256_loadu_ps((base as *const f32).add(j * kvs + off + d))
            }
        };
        let (kb, vb, f16) = match kv {
            Kv::F16 { k, v } => (k.as_ptr() as *const u8, v.as_ptr() as *const u8, true),
            Kv::F32 { k, v } => (k.as_ptr() as *const u8, v.as_ptr() as *const u8, false),
        };
        for j in 0..len {
            let mut acc = _mm256_setzero_ps();
            for d in (0..hd).step_by(8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(d)), load(kb, f16, j, d), acc);
            }
            s[j] = crate::avx2::hsum256(acc) * scale;
        }
        softmax_body(s);
        // Σ_j p_j · v_j, 64 dimensions (8 accumulators) at a time.
        for d0 in (0..hd).step_by(64) {
            let w = (hd - d0).min(64) / 8;
            let mut acc = [_mm256_setzero_ps(); 8];
            for (j, &p) in s.iter().enumerate() {
                let pv = _mm256_set1_ps(p);
                for (c, a) in acc.iter_mut().enumerate().take(w) {
                    *a = _mm256_fmadd_ps(pv, load(vb, f16, j, d0 + 8 * c), *a);
                }
            }
            for (c, a) in acc.iter().enumerate().take(w) {
                _mm256_storeu_ps(out.as_mut_ptr().add(d0 + 8 * c), *a);
            }
        }
    }
}

// ---------------------------------------------------------------- convolution

thread_local! {
    static COLS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// A 3×3 convolution of stride 2 and zero padding 1, then bias and GELU: the
/// encoder's stem. `x` is `[c_in][h][w]`; the result is `[c_out][⌈h/2⌉][⌈w/2⌉]`.
/// The weight holds `c_out` rows of `c_in·9` in `[c_in][kh][kw]` order (the GGUF's).
#[allow(clippy::too_many_arguments)]
pub fn conv3x3s2_gelu(pool: &Pool, isa: Isa, w: &Weight, bias: &[f32], x: &[f32], c_in: usize, h: usize, wd: usize, out: &mut Vec<f32>) {
    let (oh, ow) = (h.div_ceil(2), wd.div_ceil(2));
    let (c_out, k) = (w.n, c_in * 9);
    assert_eq!(w.k, k, "{}: k {} for {c_in} channels", w.name, w.k);
    assert!(x.len() >= c_in * h * wd);
    let np = oh * ow;
    COLS.with(|cell| {
        let mut cols = cell.borrow_mut();
        cols.resize(np * k, 0.0);
        let cp = OutPtr(cols.as_mut_ptr());
        // im2col: row (y, x) of the output holds the 3×3 patch of every input channel.
        pool.run(oh, &|y| {
            for xo in 0..ow {
                // SAFETY: output row y's patches are this task's.
                let row = unsafe { std::slice::from_raw_parts_mut(cp.get().add((y * ow + xo) * k), k) };
                for c in 0..c_in {
                    for kh in 0..3 {
                        let iy = (2 * y + kh) as isize - 1;
                        for kw in 0..3 {
                            let ix = (2 * xo + kw) as isize - 1;
                            row[c * 9 + kh * 3 + kw] = if iy < 0 || iy >= h as isize || ix < 0 || ix >= wd as isize {
                                0.0
                            } else {
                                x[(c * h + iy as usize) * wd + ix as usize]
                            };
                        }
                    }
                }
            }
        });
        let mut y = vec![0f32; np * c_out];
        matmul(pool, isa, w, &cols, np, &mut y);
        add_bias_gelu(isa, &mut y, bias);
        // [position][c_out] → [c_out][position]
        out.resize(c_out * np, 0.0);
        for p in 0..np {
            for c in 0..c_out {
                out[c * np + p] = y[p * c_out + c];
            }
        }
    });
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
