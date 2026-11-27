//! The block formats the kernels read, the activation quantizers, the scalar
//! dot products and the fp64 dequantization the tests compare against.
//!
//! The block layouts and the Q8_K / Q8_0 quantization rules are the GGUF file
//! format (ADR 0068): data the kernels read, not copied code. The
//! scalar kernels are the reference every AVX2 kernel is tested against; on
//! x86-64 they are never served (Q28, Q35, Q64).

pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ80 {
    pub d: u16,
    pub qs: [i8; 32],
}

/// Our interleaved Q4_K layout for 8 consecutive weight rows (spec speech/01
/// §1): the 6-bit scales and mins decoded once and transposed to
/// [sub-block][row], and the nibbles regrouped so that one 32-byte load holds
/// 4 consecutive k of each of the 8 rows (low nibble k, high nibble k + 32).
/// 1,184 B per 8×256 weights against the file's 1,152.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4Kx8 {
    pub d: [u16; 8],
    pub dmin: [u16; 8],
    pub scales: [[u8; 8]; 8],
    pub mins: [[u8; 8]; 8],
    pub qs: [u8; 1024],
}

const _: () = assert!(std::mem::size_of::<BlockQ4Kx8>() == 1184);
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);
const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);
const _: () = assert!(std::mem::size_of::<BlockQ80>() == 34);

/// A matmul weight format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WType {
    /// Q4_K in the file's layout.
    Q4K,
    /// Q4_K repacked into [`BlockQ4Kx8`] groups of 8 rows.
    Q4Kx8,
    Q6K,
    Q80,
    F16,
    F32,
}

impl WType {
    /// Bytes of one row of `k` weights (for Q4Kx8, one eighth of a group).
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            WType::Q4K => k / QK_K * 144,
            WType::Q4Kx8 => k / QK_K * 1184 / 8,
            WType::Q6K => k / QK_K * 210,
            WType::Q80 => k / QK8_0 * 34,
            WType::F16 => k * 2,
            WType::F32 => k * 4,
        }
    }

    /// The row length must be a multiple of this.
    pub fn k_multiple(self) -> usize {
        match self {
            WType::Q4K | WType::Q4Kx8 | WType::Q6K => QK_K,
            WType::Q80 => QK8_0,
            WType::F16 | WType::F32 => 1,
        }
    }

    /// The activation format this weight multiplies (each matmul quantizes its own input).
    pub fn act(self) -> AType {
        match self {
            WType::Q4K | WType::Q4Kx8 | WType::Q6K => AType::Q8K,
            WType::Q80 => AType::Q80,
            WType::F16 | WType::F32 => AType::F32,
        }
    }
}

/// An activation format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AType {
    Q8K,
    Q80,
    F32,
}

impl AType {
    pub fn row_bytes(self, k: usize) -> usize {
        match self {
            AType::Q8K => k / QK_K * 292,
            AType::Q80 => k / QK8_0 * 34,
            AType::F32 => k * 4,
        }
    }
}

// ---------------------------------------------------------------- f16

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalise the mantissa.
            let mut e: i32 = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Round to nearest even: the result F16C's `vcvtps2ph` gives with imm 0.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (r & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1;
    }
    sign | r as u16
}

// ---------------------------------------------------------------- activation quantizers

/// f32 → Q8_K: per 256-element block, scale −127 / (the signed value of
/// largest magnitude), so that value maps to −127; the 16 sums of 16 are kept
/// for the Q4_K mins and the Q6_K offset.
pub fn quantize_q8k(x: &[f32], y: &mut [BlockQ8K]) {
    for (i, b) in y.iter_mut().enumerate() {
        let xs = &x[i * QK_K..(i + 1) * QK_K];
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in xs {
            let a = v.abs();
            if a > amax {
                amax = a;
                max = v;
            }
        }
        let iscale = -127.0f32 / max;
        // An all-zero block, or one so small its scale overflows, is a zero block.
        if amax == 0.0 || !iscale.is_finite() {
            b.d = 0.0;
            b.qs = [0; 256];
            b.bsums = [0; 16];
            continue;
        }
        for j in 0..QK_K {
            // Round half to even, as the AVX2 path's vroundps does.
            b.qs[j] = ((iscale * xs[j]).round_ties_even() as i32).min(127) as i8;
        }
        for j in 0..16 {
            b.bsums[j] = b.qs[j * 16..j * 16 + 16].iter().map(|&q| q as i32).sum::<i32>() as i16;
        }
        b.d = 1.0 / iscale;
    }
}

/// f32 → Q8_0: per 32-element block, d = amax / 127 stored as f16 and the
/// values scaled by 127 / amax, rounded half to even.
pub fn quantize_q80(x: &[f32], y: &mut [BlockQ80]) {
    for (i, b) in y.iter_mut().enumerate() {
        let xs = &x[i * QK8_0..(i + 1) * QK8_0];
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        // A block so small its inverse scale overflows quantizes to zeros.
        let id = if id.is_finite() { id } else { 0.0 };
        b.d = f32_to_f16(d);
        for j in 0..QK8_0 {
            b.qs[j] = (xs[j] * id).round_ties_even() as i8;
        }
    }
}

// ---------------------------------------------------------------- scalar dot products

/// The 12-byte Q4_K scale field as 8 scales and 8 mins of 6 bits.
#[inline]
pub fn q4k_scales(s: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        if j < 4 {
            sc[j] = s[j] & 63;
            m[j] = s[j + 4] & 63;
        } else {
            sc[j] = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
            m[j] = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        }
    }
    (sc, m)
}

pub fn dot_q4k_q8k(w: &[BlockQ4K], a: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (wb, ab) in w.iter().zip(a) {
        let (sc, m) = q4k_scales(&wb.scales);
        let mut sumi = 0i32;
        for j in 0..4 {
            let q = &wb.qs[32 * j..32 * j + 32];
            let alo = &ab.qs[64 * j..64 * j + 32];
            let ahi = &ab.qs[64 * j + 32..64 * j + 64];
            let mut slo = 0i32;
            let mut shi = 0i32;
            for l in 0..32 {
                slo += (q[l] & 0xF) as i32 * alo[l] as i32;
                shi += (q[l] >> 4) as i32 * ahi[l] as i32;
            }
            sumi += sc[2 * j] as i32 * slo + sc[2 * j + 1] as i32 * shi;
        }
        let mut summ = 0i32;
        for j in 0..8 {
            summ += m[j] as i32 * (ab.bsums[2 * j] as i32 + ab.bsums[2 * j + 1] as i32);
        }
        let d = f16_to_f32(wb.d) * ab.d;
        let dmin = f16_to_f32(wb.dmin) * ab.d;
        sumf += d * sumi as f32 - dmin * summ as f32;
    }
    sumf
}

/// Row `c` (0..8) of an interleaved group, against one activation row.
pub fn dot_q4kx8_q8k(w: &[BlockQ4Kx8], c: usize, a: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (wb, ab) in w.iter().zip(a) {
        let mut sumi = 0i32;
        for j in 0..8 {
            let (j64, hi) = (j / 2, j % 2);
            let mut s = 0i32;
            for q in 0..8 {
                for b in 0..4 {
                    let byte = wb.qs[32 * (8 * j64 + q) + 4 * c + b];
                    let w4 = if hi == 1 { byte >> 4 } else { byte & 0xF };
                    s += w4 as i32 * ab.qs[64 * j64 + 32 * hi + 4 * q + b] as i32;
                }
            }
            sumi += wb.scales[j][c] as i32 * s;
        }
        let mut summ = 0i32;
        for j in 0..8 {
            summ += wb.mins[j][c] as i32 * (ab.bsums[2 * j] as i32 + ab.bsums[2 * j + 1] as i32);
        }
        sumf += f16_to_f32(wb.d[c]) * ab.d * sumi as f32 - f16_to_f32(wb.dmin[c]) * ab.d * summ as f32;
    }
    sumf
}

pub fn dot_q6k_q8k(w: &[BlockQ6K], a: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (wb, ab) in w.iter().zip(a) {
        let mut sumi = 0i32;
        for half in 0..2 {
            let ql = &wb.ql[64 * half..];
            let qh = &wb.qh[32 * half..];
            let sc = &wb.scales[8 * half..];
            let q8 = &ab.qs[128 * half..];
            for l in 0..32 {
                let is = l / 16;
                let (q1, q2, q3, q4) = q6k_quad(ql, qh, l);
                sumi += sc[is] as i32 * q1 * q8[l] as i32
                    + sc[is + 2] as i32 * q2 * q8[l + 32] as i32
                    + sc[is + 4] as i32 * q3 * q8[l + 64] as i32
                    + sc[is + 6] as i32 * q4 * q8[l + 96] as i32;
            }
        }
        sumf += f16_to_f32(wb.d) * ab.d * sumi as f32;
    }
    sumf
}

/// The four signed 6-bit values at `l`, `l + 32`, `l + 64` and `l + 96` of a Q6_K half block.
#[inline]
fn q6k_quad(ql: &[u8], qh: &[u8], l: usize) -> (i32, i32, i32, i32) {
    (
        ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32,
        ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32,
        ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32,
        ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32,
    )
}

pub fn dot_q80_q80(w: &[BlockQ80], a: &[BlockQ80]) -> f32 {
    let mut sumf = 0f32;
    for (wb, ab) in w.iter().zip(a) {
        let sumi: i32 = wb.qs.iter().zip(&ab.qs).map(|(&x, &y)| x as i32 * y as i32).sum();
        sumf += f16_to_f32(wb.d) * f16_to_f32(ab.d) * sumi as f32;
    }
    sumf
}

/// F16 weights against f32 activations, summed in eight lanes as the AVX2 kernel does.
pub fn dot_f16_f32(w: &[u16], a: &[f32]) -> f32 {
    let k8 = w.len() / 8 * 8;
    let mut acc = [0f32; 8];
    for i in (0..k8).step_by(8) {
        for l in 0..8 {
            acc[l] += f16_to_f32(w[i + l]) * a[i + l];
        }
    }
    let mut s = lanes_sum(acc);
    for i in k8..w.len() {
        s += f16_to_f32(w[i]) * a[i];
    }
    s
}

pub fn dot_f32_f32(w: &[f32], a: &[f32]) -> f32 {
    let k8 = w.len() / 8 * 8;
    let mut acc = [0f32; 8];
    for i in (0..k8).step_by(8) {
        for l in 0..8 {
            acc[l] += w[i + l] * a[i + l];
        }
    }
    let mut s = lanes_sum(acc);
    for i in k8..w.len() {
        s += w[i] * a[i];
    }
    s
}

/// The eight lanes summed in the AVX2 kernels' horizontal order.
#[inline]
pub fn lanes_sum(a: [f32; 8]) -> f32 {
    let x = [a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]];
    let y = [x[0] + x[2], x[1] + x[3]];
    y[0] + y[1]
}

// ---------------------------------------------------------------- the Q4_K repack

/// Repacks 8 consecutive Q4_K rows of `nb` blocks each into `nb` interleaved blocks.
pub fn repack_q4k_x8(rows: [&[BlockQ4K]; 8], dst: &mut [BlockQ4Kx8]) {
    for (i, d) in dst.iter_mut().enumerate() {
        for (c, row) in rows.iter().enumerate() {
            let s = &row[i];
            d.d[c] = s.d;
            d.dmin[c] = s.dmin;
            let (sc, m) = q4k_scales(&s.scales);
            for j in 0..8 {
                d.scales[j][c] = sc[j];
                d.mins[j][c] = m[j];
            }
            for j64 in 0..4 {
                for q in 0..8 {
                    let o = 32 * (8 * j64 + q) + 4 * c;
                    d.qs[o..o + 4].copy_from_slice(&s.qs[32 * j64 + 4 * q..32 * j64 + 4 * q + 4]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- fp64 dequantization (the reference)

/// Row `n` of a weight of type `ty` and row length `k`, dequantized in fp64.
/// For Q4Kx8, `data` is the whole repacked weight.
pub fn dequant_row_f64(ty: WType, data: &[u8], n: usize, k: usize, out: &mut [f64]) {
    let rb = ty.row_bytes(k);
    match ty {
        WType::Q4K => {
            for (i, b) in cast_slice::<BlockQ4K>(&data[n * rb..], k / QK_K).iter().enumerate() {
                let (d, dmin) = (f16_to_f32(b.d) as f64, f16_to_f32(b.dmin) as f64);
                let (sc, m) = q4k_scales(&b.scales);
                for j in 0..4 {
                    for l in 0..32 {
                        let q = b.qs[32 * j + l];
                        out[i * QK_K + 64 * j + l] = d * sc[2 * j] as f64 * (q & 0xF) as f64 - dmin * m[2 * j] as f64;
                        out[i * QK_K + 64 * j + 32 + l] =
                            d * sc[2 * j + 1] as f64 * (q >> 4) as f64 - dmin * m[2 * j + 1] as f64;
                    }
                }
            }
        }
        WType::Q4Kx8 => {
            let nb = k / QK_K;
            let (g, c) = (n / 8, n % 8);
            for (i, b) in cast_slice::<BlockQ4Kx8>(&data[g * nb * 1184..], nb).iter().enumerate() {
                let (d, dmin) = (f16_to_f32(b.d[c]) as f64, f16_to_f32(b.dmin[c]) as f64);
                for j64 in 0..4 {
                    for q in 0..8 {
                        for bb in 0..4 {
                            let byte = b.qs[32 * (8 * j64 + q) + 4 * c + bb];
                            for hi in 0..2 {
                                let j = 2 * j64 + hi;
                                let w4 = if hi == 1 { byte >> 4 } else { byte & 0xF };
                                out[i * QK_K + 64 * j64 + 32 * hi + 4 * q + bb] =
                                    d * b.scales[j][c] as f64 * w4 as f64 - dmin * b.mins[j][c] as f64;
                            }
                        }
                    }
                }
            }
        }
        WType::Q6K => {
            for (i, b) in cast_slice::<BlockQ6K>(&data[n * rb..], k / QK_K).iter().enumerate() {
                let d = f16_to_f32(b.d) as f64;
                for half in 0..2 {
                    let (ql, qh, sc) = (&b.ql[64 * half..], &b.qh[32 * half..], &b.scales[8 * half..]);
                    let y = &mut out[i * QK_K + 128 * half..];
                    for l in 0..32 {
                        let is = l / 16;
                        let (q1, q2, q3, q4) = q6k_quad(ql, qh, l);
                        y[l] = d * sc[is] as f64 * q1 as f64;
                        y[l + 32] = d * sc[is + 2] as f64 * q2 as f64;
                        y[l + 64] = d * sc[is + 4] as f64 * q3 as f64;
                        y[l + 96] = d * sc[is + 6] as f64 * q4 as f64;
                    }
                }
            }
        }
        WType::Q80 => {
            for (i, b) in cast_slice::<BlockQ80>(&data[n * rb..], k / QK8_0).iter().enumerate() {
                let d = f16_to_f32(b.d) as f64;
                for j in 0..QK8_0 {
                    out[i * QK8_0 + j] = d * b.qs[j] as f64;
                }
            }
        }
        WType::F16 => {
            for (j, h) in cast_slice::<u16>(&data[n * rb..], k).iter().enumerate() {
                out[j] = f16_to_f32(*h) as f64;
            }
        }
        WType::F32 => {
            for (j, v) in cast_slice::<f32>(&data[n * rb..], k).iter().enumerate() {
                out[j] = *v as f64;
            }
        }
    }
}

/// Row `m` of a quantized activation, dequantized in fp64.
pub fn dequant_act_f64(ty: AType, data: &[u8], m: usize, k: usize, out: &mut [f64]) {
    let row = &data[m * ty.row_bytes(k)..];
    match ty {
        AType::Q8K => {
            for (i, b) in cast_slice::<BlockQ8K>(row, k / QK_K).iter().enumerate() {
                for j in 0..QK_K {
                    out[i * QK_K + j] = b.d as f64 * b.qs[j] as f64;
                }
            }
        }
        AType::Q80 => {
            for (i, b) in cast_slice::<BlockQ80>(row, k / QK8_0).iter().enumerate() {
                let d = f16_to_f32(b.d) as f64;
                for j in 0..QK8_0 {
                    out[i * QK8_0 + j] = d * b.qs[j] as f64;
                }
            }
        }
        AType::F32 => {
            for (j, v) in cast_slice::<f32>(row, k).iter().enumerate() {
                out[j] = *v as f64;
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A type any bytes are a valid value of, with no padding and no pointer: the block formats,
/// `u16` (f16 bits) and `f32`. Sealed, so the casts below take nothing else.
pub trait Pod: Copy + sealed::Sealed {}

macro_rules! pod {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl Pod for $t {}
    )*};
}

// Each block is `repr(C)` of byte, i8, i16, u16 and f32 fields with no padding (the size
// assertions above), so every bit pattern is a value.
pod!(u16, f32, BlockQ4K, BlockQ6K, BlockQ8K, BlockQ80, BlockQ4Kx8);

/// Reinterprets the first `n` elements' worth of `bytes` as `T` (checked size and alignment).
///
/// A type some bytes are not a value of is refused at compile time:
///
/// ```compile_fail
/// let _ = ignis_speech::quant::cast_slice::<bool>(&[2u8; 4], 1);
/// ```
pub fn cast_slice<T: Pod>(bytes: &[u8], n: usize) -> &[T] {
    assert!(bytes.len() >= n * std::mem::size_of::<T>(), "slice too short");
    assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<T>(), 0, "slice misaligned");
    // SAFETY: length and alignment checked above; any bytes are a `T` (`Pod`).
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) }
}

/// [`cast_slice`], writable: a `T` written is bytes with no padding to read back.
pub fn cast_slice_mut<T: Pod>(bytes: &mut [u8], n: usize) -> &mut [T] {
    assert!(bytes.len() >= n * std::mem::size_of::<T>(), "slice too short");
    assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<T>(), 0, "slice misaligned");
    // SAFETY: as in `cast_slice`, and the borrow is exclusive.
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, n) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_is_exact_for_every_half() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f32_to_f16(f) & 0x7c00 == 0x7c00 && f32_to_f16(f) & 0x3ff != 0);
            } else {
                assert_eq!(f32_to_f16(f), h, "half {h:#06x}");
            }
        }
        // Ties go to even.
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn q8k_maps_the_largest_magnitude_to_minus_127() {
        let mut x = vec![0f32; QK_K];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 - 100.0) * 0.01;
        }
        let mut b = [BlockQ8K { d: 0.0, qs: [0; 256], bsums: [0; 16] }];
        quantize_q8k(&x, &mut b);
        // The largest magnitude is +1.55 at i = 255: it maps to -127 with a negative scale.
        assert_eq!(b[0].qs[255], -127);
        assert!((b[0].d * -127.0 - 1.55).abs() < 1e-6);
        assert_eq!(b[0].bsums[0], b[0].qs[..16].iter().map(|&q| q as i16).sum::<i16>());
    }
}
