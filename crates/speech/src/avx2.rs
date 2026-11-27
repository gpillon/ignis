//! The AVX2 + FMA + F16C kernels (spec speech/01 §1, ADR 0068).
//!
//! Compiled only on x86-64. Nothing sets a global `target-cpu`: every
//! function carries its own `#[target_feature]`, and it is reached only
//! through an [`Isa`](crate::cpu::Isa) that [`cpu::select`](crate::cpu::select)
//! built after `is_x86_feature_detected!` reported all three features.
//!
//! # Safety
//!
//! Every `unsafe fn` here needs AVX2, FMA and F16C on the running CPU, and
//! pointers valid for the rows, row strides and block counts it is given.

use std::arch::x86_64::*;

use crate::quant::*;

/// One register tile: `out[r * ldo + c] = dot(weight row c, activation row r)`
/// for MR activation rows and NR weight rows (or NR groups of 8 rows for the
/// interleaved Q4_K). `n` is the block count along k (elements for F16/F32).
pub type TileFn = unsafe fn(w: *const u8, wrb: usize, a: *const u8, arb: usize, n: usize, out: *mut f32, ldo: usize);

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn f16s(h: u16) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)))
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub(crate) fn hsum256(v: __m256) -> f32 {
    let x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_add_ss(x, _mm_movehdup_ps(x));
    _mm_cvtss_f32(x)
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn hsum128(x: __m128) -> f32 {
    let x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_add_ss(x, _mm_movehdup_ps(x));
    _mm_cvtss_f32(x)
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
pub(crate) fn hmax256(v: __m256) -> f32 {
    let x = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let x = _mm_max_ps(x, _mm_movehl_ps(x, x));
    let x = _mm_max_ss(x, _mm_movehdup_ps(x));
    _mm_cvtss_f32(x)
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn hsum_epi32(v: __m256i) -> i32 {
    let x = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0b0100_1110>(x));
    let x = _mm_add_epi32(x, _mm_shuffle_epi32::<0b1011_0001>(x));
    _mm_cvtsi128_si32(x)
}

/// Packs 4×8 i32 (elements 0..31 in order) into 32 i8 in order.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn pack32(i0: __m256i, i1: __m256i, i2: __m256i, i3: __m256i) -> __m256i {
    let p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
    _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
}

// ---------------------------------------------------------------- activation quantizers

/// f32 → Q8_K, bit-identical to [`crate::quant::quantize_q8k`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn quantize_q8k(x: *const f32, y: *mut BlockQ8K, nb: usize) {
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        let lim = _mm256_set1_epi32(127);
        for i in 0..nb {
            let xb = x.add(i * QK_K);
            let yb = &mut *y.add(i);
            let mut vmax = _mm256_setzero_ps();
            for j in 0..32 {
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb.add(8 * j))));
            }
            let amax = hmax256(vmax);
            // The signed value of the first element of largest magnitude, as the scalar rule takes it.
            let vam = _mm256_set1_ps(amax);
            let mut max = amax;
            for j in 0..32 {
                let v = _mm256_loadu_ps(xb.add(8 * j));
                let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(_mm256_andnot_ps(sign, v), vam));
                if mask != 0 {
                    max = *xb.add(8 * j + mask.trailing_zeros() as usize);
                    break;
                }
            }
            let iscale = -127.0f32 / max;
            if amax == 0.0 || !iscale.is_finite() {
                yb.d = 0.0;
                yb.qs = [0; 256];
                yb.bsums = [0; 16];
                continue;
            }
            let vis = _mm256_set1_ps(iscale);
            for g in 0..8 {
                let mut iv = [_mm256_setzero_si256(); 4];
                for (q, v) in iv.iter_mut().enumerate() {
                    let s = _mm256_mul_ps(_mm256_loadu_ps(xb.add(32 * g + 8 * q)), vis);
                    let r = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(s);
                    *v = _mm256_min_epi32(_mm256_cvtps_epi32(r), lim);
                }
                yb.bsums[2 * g] = hsum_epi32(_mm256_add_epi32(iv[0], iv[1])) as i16;
                yb.bsums[2 * g + 1] = hsum_epi32(_mm256_add_epi32(iv[2], iv[3])) as i16;
                _mm256_storeu_si256(yb.qs.as_mut_ptr().add(32 * g) as *mut __m256i, pack32(iv[0], iv[1], iv[2], iv[3]));
            }
            yb.d = 1.0 / iscale;
        }
    }
}

/// f32 → Q8_0, bit-identical to [`crate::quant::quantize_q80`].
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn quantize_q80(x: *const f32, y: *mut BlockQ80, nb: usize) {
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        for i in 0..nb {
            let xb = x.add(i * QK8_0);
            let v = [_mm256_loadu_ps(xb), _mm256_loadu_ps(xb.add(8)), _mm256_loadu_ps(xb.add(16)), _mm256_loadu_ps(xb.add(24))];
            let mut m = _mm256_andnot_ps(sign, v[0]);
            for q in &v[1..] {
                m = _mm256_max_ps(m, _mm256_andnot_ps(sign, *q));
            }
            let amax = hmax256(m);
            let d = amax / 127.0;
            let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
            let id = if id.is_finite() { id } else { 0.0 };
            let yb = &mut *y.add(i);
            yb.d = _mm_extract_epi16::<0>(_mm_cvtps_ph::<0>(_mm_set_ss(d))) as u16;
            let vid = _mm256_set1_ps(id);
            let mut iv = [_mm256_setzero_si256(); 4];
            for (q, out) in iv.iter_mut().enumerate() {
                let r = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(v[q], vid));
                *out = _mm256_cvtps_epi32(r);
            }
            _mm256_storeu_si256(yb.qs.as_mut_ptr() as *mut __m256i, pack32(iv[0], iv[1], iv[2], iv[3]));
        }
    }
}

// ---------------------------------------------------------------- Q4_K × Q8_K, the file's layout

/// The 12-byte Q4_K scale field as (8 scales, 8 mins), i16 each: the scalar decode, widened.
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
fn q4k_scales_vec(s: &[u8; 12]) -> (__m128i, __m128i) {
    let (sc, m) = q4k_scales(s);
    let widen = |b: [u8; 8]| _mm_cvtepu8_epi16(_mm_set_epi64x(0, i64::from_le_bytes(b)));
    (widen(sc), widen(m))
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_q4k<const MR: usize, const NR: usize>(
    w: *const u8,
    wrb: usize,
    a: *const u8,
    arb: usize,
    nb: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = [[_mm256_setzero_ps(); NR]; MR];
        let mut accm = [[_mm_setzero_ps(); NR]; MR];
        for i in 0..nb {
            let mut sc = [_mm256_setzero_si256(); NR];
            let mut mins = [_mm_setzero_si128(); NR];
            let mut dw = [0f32; NR];
            let mut dmw = [0f32; NR];
            let mut qp = [std::ptr::null::<u8>(); NR];
            for c in 0..NR {
                let b = &*(w.add(c * wrb) as *const BlockQ4K).add(i);
                let (s, m) = q4k_scales_vec(&b.scales);
                sc[c] = _mm256_broadcastsi128_si256(s);
                mins[c] = m;
                dw[c] = f16s(b.d);
                dmw[c] = f16s(b.dmin);
                qp[c] = b.qs.as_ptr();
            }
            let mut sumi = [[_mm256_setzero_si256(); NR]; MR];
            for j in 0..4 {
                let shl = _mm256_set1_epi16((((4 * j + 1) << 8) | (4 * j)) as i16);
                let shh = _mm256_set1_epi16((((4 * j + 3) << 8) | (4 * j + 2)) as i16);
                let mut ql = [_mm256_setzero_si256(); NR];
                let mut qh = [_mm256_setzero_si256(); NR];
                let mut sl = [_mm256_setzero_si256(); NR];
                let mut sh = [_mm256_setzero_si256(); NR];
                for c in 0..NR {
                    let bits = _mm256_loadu_si256(qp[c].add(32 * j) as *const __m256i);
                    ql[c] = _mm256_and_si256(bits, m4);
                    qh[c] = _mm256_and_si256(_mm256_srli_epi16::<4>(bits), m4);
                    sl[c] = _mm256_shuffle_epi8(sc[c], shl);
                    sh[c] = _mm256_shuffle_epi8(sc[c], shh);
                }
                for r in 0..MR {
                    let q8 = (*(a.add(r * arb) as *const BlockQ8K).add(i)).qs.as_ptr().add(64 * j);
                    let yl = _mm256_loadu_si256(q8 as *const __m256i);
                    let yh = _mm256_loadu_si256(q8.add(32) as *const __m256i);
                    for c in 0..NR {
                        let pl = _mm256_madd_epi16(sl[c], _mm256_maddubs_epi16(ql[c], yl));
                        let ph = _mm256_madd_epi16(sh[c], _mm256_maddubs_epi16(qh[c], yh));
                        sumi[r][c] = _mm256_add_epi32(sumi[r][c], _mm256_add_epi32(pl, ph));
                    }
                }
            }
            for r in 0..MR {
                let ab = &*(a.add(r * arb) as *const BlockQ8K).add(i);
                let bs = _mm256_loadu_si256(ab.bsums.as_ptr() as *const __m256i);
                let q8s = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256::<1>(bs));
                for c in 0..NR {
                    acc[r][c] = _mm256_fmadd_ps(_mm256_set1_ps(dw[c] * ab.d), _mm256_cvtepi32_ps(sumi[r][c]), acc[r][c]);
                    let pm = _mm_madd_epi16(mins[c], q8s);
                    accm[r][c] = _mm_fmadd_ps(_mm_set1_ps(-dmw[c] * ab.d), _mm_cvtepi32_ps(pm), accm[r][c]);
                }
            }
        }
        for r in 0..MR {
            for c in 0..NR {
                *out.add(r * ldo + c) = hsum256(acc[r][c]) + hsum128(accm[r][c]);
            }
        }
    }
}

// ---------------------------------------------------------------- Q6_K × Q8_K

/// The 6-bit values are used unsigned (0..63), so `maddubs` cannot saturate;
/// the −32 offset is applied once per block through the activation's sums.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_q6k<const MR: usize, const NR: usize>(
    w: *const u8,
    wrb: usize,
    a: *const u8,
    arb: usize,
    nb: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let mut acc = [[_mm256_setzero_ps(); NR]; MR];
        let mut accm = [[_mm256_setzero_ps(); NR]; MR];
        for i in 0..nb {
            let mut d = [0f32; NR];
            let mut sc128 = [_mm_setzero_si128(); NR];
            let mut sc16 = [_mm256_setzero_si256(); NR];
            let mut wb = [std::ptr::null::<BlockQ6K>(); NR];
            for c in 0..NR {
                let b = (w.add(c * wrb) as *const BlockQ6K).add(i);
                d[c] = f16s((*b).d);
                sc128[c] = _mm_loadu_si128((*b).scales.as_ptr() as *const __m128i);
                sc16[c] = _mm256_cvtepi8_epi16(sc128[c]);
                wb[c] = b;
            }
            let mut sumi = [[_mm256_setzero_si256(); NR]; MR];
            for half in 0..2 {
                let mut q = [[_mm256_setzero_si256(); 4]; NR];
                let mut s = [[_mm256_setzero_si256(); 4]; NR];
                for c in 0..NR {
                    let ql = (*wb[c]).ql.as_ptr().add(64 * half);
                    let qh = (*wb[c]).qh.as_ptr().add(32 * half);
                    let b1 = _mm256_loadu_si256(ql as *const __m256i);
                    let b2 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                    let bh = _mm256_loadu_si256(qh as *const __m256i);
                    let h0 = _mm256_slli_epi16::<4>(_mm256_and_si256(bh, m2));
                    let h1 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(bh), m2));
                    let h2 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(bh), m2));
                    let h3 = _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(bh), m2));
                    q[c][0] = _mm256_or_si256(_mm256_and_si256(b1, m4), h0);
                    q[c][1] = _mm256_or_si256(_mm256_and_si256(b2, m4), h1);
                    q[c][2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(b1), m4), h2);
                    q[c][3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(b2), m4), h3);
                    for k in 0..4 {
                        let idx = (4 * half + k) as i64;
                        let shuf = _mm_set_epi64x((2 * idx + 1) * 0x0101_0101_0101_0101, (2 * idx) * 0x0101_0101_0101_0101);
                        s[c][k] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(sc128[c], shuf));
                    }
                }
                for r in 0..MR {
                    let q8 = (*(a.add(r * arb) as *const BlockQ8K).add(i)).qs.as_ptr().add(128 * half);
                    for k in 0..4 {
                        let y = _mm256_loadu_si256(q8.add(32 * k) as *const __m256i);
                        for c in 0..NR {
                            let p = _mm256_madd_epi16(s[c][k], _mm256_maddubs_epi16(q[c][k], y));
                            sumi[r][c] = _mm256_add_epi32(sumi[r][c], p);
                        }
                    }
                }
            }
            for r in 0..MR {
                let ab = &*(a.add(r * arb) as *const BlockQ8K).add(i);
                let bs = _mm256_loadu_si256(ab.bsums.as_ptr() as *const __m256i);
                for c in 0..NR {
                    acc[r][c] = _mm256_fmadd_ps(_mm256_set1_ps(d[c] * ab.d), _mm256_cvtepi32_ps(sumi[r][c]), acc[r][c]);
                    let corr = _mm256_madd_epi16(sc16[c], bs);
                    accm[r][c] = _mm256_fmadd_ps(_mm256_set1_ps(-32.0 * d[c] * ab.d), _mm256_cvtepi32_ps(corr), accm[r][c]);
                }
            }
        }
        for r in 0..MR {
            for c in 0..NR {
                *out.add(r * ldo + c) = hsum256(_mm256_add_ps(acc[r][c], accm[r][c]));
            }
        }
    }
}

// ---------------------------------------------------------------- Q8_0 × Q8_0

/// Signed × signed through `maddubs(|w|, sign(x, w))`; 2·127·127 fits i16.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_q80<const MR: usize, const NR: usize>(
    w: *const u8,
    wrb: usize,
    a: *const u8,
    arb: usize,
    nb: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_ps(); NR]; MR];
        for i in 0..nb {
            let mut wq = [_mm256_setzero_si256(); NR];
            let mut aw = [_mm256_setzero_si256(); NR];
            let mut dw = [0f32; NR];
            for c in 0..NR {
                let b = &*(w.add(c * wrb) as *const BlockQ80).add(i);
                wq[c] = _mm256_loadu_si256(b.qs.as_ptr() as *const __m256i);
                aw[c] = _mm256_sign_epi8(wq[c], wq[c]);
                dw[c] = f16s(b.d);
            }
            for r in 0..MR {
                let ab = &*(a.add(r * arb) as *const BlockQ80).add(i);
                let y = _mm256_loadu_si256(ab.qs.as_ptr() as *const __m256i);
                let da = f16s(ab.d);
                for c in 0..NR {
                    let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw[c], _mm256_sign_epi8(y, wq[c])), ones);
                    acc[r][c] = _mm256_fmadd_ps(_mm256_set1_ps(dw[c] * da), _mm256_cvtepi32_ps(p), acc[r][c]);
                }
            }
        }
        for r in 0..MR {
            for c in 0..NR {
                *out.add(r * ldo + c) = hsum256(acc[r][c]);
            }
        }
    }
}

// ---------------------------------------------------------------- F16 and F32 weights × f32

/// `k` is the element count; a tail under 8 is summed after the lanes, as the scalar reference does.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_f16<const MR: usize, const NR: usize>(
    w: *const u8,
    wrb: usize,
    a: *const u8,
    arb: usize,
    k: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let k8 = k / 8 * 8;
        let mut acc = [[_mm256_setzero_ps(); NR]; MR];
        let mut i = 0;
        while i < k8 {
            let mut wv = [_mm256_setzero_ps(); NR];
            for c in 0..NR {
                wv[c] = _mm256_cvtph_ps(_mm_loadu_si128(w.add(c * wrb + 2 * i) as *const __m128i));
            }
            for r in 0..MR {
                let av = _mm256_loadu_ps(a.add(r * arb + 4 * i) as *const f32);
                for c in 0..NR {
                    acc[r][c] = _mm256_fmadd_ps(av, wv[c], acc[r][c]);
                }
            }
            i += 8;
        }
        for r in 0..MR {
            for c in 0..NR {
                let mut s = hsum256(acc[r][c]);
                for j in k8..k {
                    s += f16s(*(w.add(c * wrb + 2 * j) as *const u16)) * *(a.add(r * arb + 4 * j) as *const f32);
                }
                *out.add(r * ldo + c) = s;
            }
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_f32<const MR: usize, const NR: usize>(
    w: *const u8,
    wrb: usize,
    a: *const u8,
    arb: usize,
    k: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let k8 = k / 8 * 8;
        let mut acc = [[_mm256_setzero_ps(); NR]; MR];
        let mut i = 0;
        while i < k8 {
            let mut wv = [_mm256_setzero_ps(); NR];
            for c in 0..NR {
                wv[c] = _mm256_loadu_ps(w.add(c * wrb + 4 * i) as *const f32);
            }
            for r in 0..MR {
                let av = _mm256_loadu_ps(a.add(r * arb + 4 * i) as *const f32);
                for c in 0..NR {
                    acc[r][c] = _mm256_fmadd_ps(av, wv[c], acc[r][c]);
                }
            }
            i += 8;
        }
        for r in 0..MR {
            for c in 0..NR {
                let mut s = hsum256(acc[r][c]);
                for j in k8..k {
                    s += *(w.add(c * wrb + 4 * j) as *const f32) * *(a.add(r * arb + 4 * j) as *const f32);
                }
                *out.add(r * ldo + c) = s;
            }
        }
    }
}

/// The register tiles every non-interleaved format offers, (activation rows, weight rows).
pub const TILES: [(usize, usize); 16] =
    [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3), (6, 1), (6, 2), (8, 1)];

pub fn tile_fn(ty: WType, mr: usize, nr: usize) -> Option<TileFn> {
    macro_rules! pick {
        ($f:ident) => {
            match (mr, nr) {
                (1, 1) => Some($f::<1, 1> as TileFn),
                (1, 2) => Some($f::<1, 2> as TileFn),
                (1, 3) => Some($f::<1, 3> as TileFn),
                (1, 4) => Some($f::<1, 4> as TileFn),
                (2, 1) => Some($f::<2, 1> as TileFn),
                (2, 2) => Some($f::<2, 2> as TileFn),
                (2, 3) => Some($f::<2, 3> as TileFn),
                (2, 4) => Some($f::<2, 4> as TileFn),
                (3, 1) => Some($f::<3, 1> as TileFn),
                (3, 2) => Some($f::<3, 2> as TileFn),
                (4, 1) => Some($f::<4, 1> as TileFn),
                (4, 2) => Some($f::<4, 2> as TileFn),
                (4, 3) => Some($f::<4, 3> as TileFn),
                (6, 1) => Some($f::<6, 1> as TileFn),
                (6, 2) => Some($f::<6, 2> as TileFn),
                (8, 1) => Some($f::<8, 1> as TileFn),
                _ => None,
            }
        };
    }
    match ty {
        WType::Q4K => pick!(tile_q4k),
        WType::Q6K => pick!(tile_q6k),
        WType::Q80 => pick!(tile_q80),
        WType::F16 => pick!(tile_f16),
        WType::F32 => pick!(tile_f32),
        WType::Q4Kx8 => None,
    }
}

// ---------------------------------------------------------------- Q4_K × Q8_K, our interleaved layout

/// MR activation rows × 8·G weight rows over [`BlockQ4Kx8`] groups (stride
/// `wgs` bytes). One 32-byte weight load carries 4 consecutive k for each of
/// 8 rows, and the matching 4 activation bytes are broadcast. The `maddubs`
/// results stay in i16 for a whole 32-element sub-block (8 × 3810 < 32767),
/// so the scale `madd` runs once per sub-block, and the block scaling is one
/// FMA for 8 rows.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn tile_q4kx8<const MR: usize, const G: usize>(
    w: *const u8,
    wgs: usize,
    a: *const u8,
    arb: usize,
    nb: usize,
    out: *mut f32,
    ldo: usize,
) {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = [[_mm256_setzero_ps(); G]; MR];
        for i in 0..nb {
            let mut bp = [std::ptr::null::<BlockQ4Kx8>(); G];
            for (g, p) in bp.iter_mut().enumerate() {
                *p = (w.add(g * wgs) as *const BlockQ4Kx8).add(i);
            }
            let mut ap = [std::ptr::null::<BlockQ8K>(); MR];
            for (r, p) in ap.iter_mut().enumerate() {
                *p = (a.add(r * arb) as *const BlockQ8K).add(i);
            }
            let mut sumi = [[_mm256_setzero_si256(); G]; MR];
            for j in 0..8 {
                let (j64, hi) = (j / 2, j % 2);
                let mut sc = [_mm256_setzero_si256(); G];
                for g in 0..G {
                    let s8 = _mm_loadl_epi64((*bp[g]).scales[j].as_ptr() as *const __m128i);
                    sc[g] = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(s8, s8));
                }
                let mut p16 = [[_mm256_setzero_si256(); G]; MR];
                for q in 0..8 {
                    let mut wv = [_mm256_setzero_si256(); G];
                    for g in 0..G {
                        let bytes = _mm256_loadu_si256((*bp[g]).qs.as_ptr().add(32 * (8 * j64 + q)) as *const __m256i);
                        wv[g] = if hi == 1 { _mm256_and_si256(_mm256_srli_epi16::<4>(bytes), m4) } else { _mm256_and_si256(bytes, m4) };
                    }
                    for r in 0..MR {
                        let x = std::ptr::read_unaligned((*ap[r]).qs.as_ptr().add(64 * j64 + 32 * hi + 4 * q) as *const i32);
                        let av = _mm256_set1_epi32(x);
                        for g in 0..G {
                            p16[r][g] = _mm256_add_epi16(p16[r][g], _mm256_maddubs_epi16(wv[g], av));
                        }
                    }
                }
                for r in 0..MR {
                    for g in 0..G {
                        sumi[r][g] = _mm256_add_epi32(sumi[r][g], _mm256_madd_epi16(p16[r][g], sc[g]));
                    }
                }
            }
            // d · Σ_j sc_j · dot_j − dmin · Σ_j m_j · bsum32_j
            let mut dv = [_mm256_setzero_ps(); G];
            let mut dmv = [_mm256_setzero_ps(); G];
            let mut mp = [[_mm256_setzero_si256(); 4]; G];
            for g in 0..G {
                let b = &*bp[g];
                dv[g] = _mm256_cvtph_ps(_mm_loadu_si128(b.d.as_ptr() as *const __m128i));
                dmv[g] = _mm256_cvtph_ps(_mm_loadu_si128(b.dmin.as_ptr() as *const __m128i));
                for p in 0..4 {
                    let m0 = _mm_cvtepu8_epi16(_mm_loadl_epi64(b.mins[2 * p].as_ptr() as *const __m128i));
                    let m1 = _mm_cvtepu8_epi16(_mm_loadl_epi64(b.mins[2 * p + 1].as_ptr() as *const __m128i));
                    mp[g][p] = _mm256_set_m128i(_mm_unpackhi_epi16(m0, m1), _mm_unpacklo_epi16(m0, m1));
                }
            }
            for r in 0..MR {
                let ab = &*ap[r];
                let bs = _mm256_loadu_si256(ab.bsums.as_ptr() as *const __m256i);
                let b32 = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256::<1>(bs));
                let bb = [
                    _mm256_broadcastd_epi32(b32),
                    _mm256_broadcastd_epi32(_mm_shuffle_epi32::<0x55>(b32)),
                    _mm256_broadcastd_epi32(_mm_shuffle_epi32::<0xAA>(b32)),
                    _mm256_broadcastd_epi32(_mm_shuffle_epi32::<0xFF>(b32)),
                ];
                let vda = _mm256_set1_ps(ab.d);
                for g in 0..G {
                    let mut summ = _mm256_madd_epi16(mp[g][0], bb[0]);
                    for p in 1..4 {
                        summ = _mm256_add_epi32(summ, _mm256_madd_epi16(mp[g][p], bb[p]));
                    }
                    acc[r][g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sumi[r][g]), _mm256_mul_ps(dv[g], vda), acc[r][g]);
                    acc[r][g] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(summ), _mm256_mul_ps(dmv[g], vda), acc[r][g]);
                }
            }
        }
        for r in 0..MR {
            for g in 0..G {
                _mm256_storeu_ps(out.add(r * ldo + 8 * g), acc[r][g]);
            }
        }
    }
}

/// The interleaved tiles, (activation rows, groups of 8 weight rows).
pub const TILES_X8: [(usize, usize); 11] = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (6, 1), (8, 1)];

pub fn tile_x8_fn(mr: usize, g: usize) -> Option<TileFn> {
    match (mr, g) {
        (1, 1) => Some(tile_q4kx8::<1, 1> as TileFn),
        (1, 2) => Some(tile_q4kx8::<1, 2> as TileFn),
        (1, 4) => Some(tile_q4kx8::<1, 4> as TileFn),
        (2, 1) => Some(tile_q4kx8::<2, 1> as TileFn),
        (2, 2) => Some(tile_q4kx8::<2, 2> as TileFn),
        (3, 1) => Some(tile_q4kx8::<3, 1> as TileFn),
        (3, 2) => Some(tile_q4kx8::<3, 2> as TileFn),
        (4, 1) => Some(tile_q4kx8::<4, 1> as TileFn),
        (4, 2) => Some(tile_q4kx8::<4, 2> as TileFn),
        (6, 1) => Some(tile_q4kx8::<6, 1> as TileFn),
        (8, 1) => Some(tile_q4kx8::<8, 1> as TileFn),
        _ => None,
    }
}
