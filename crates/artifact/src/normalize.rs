//! The A1 normalization layer (#27, spec 04): map each artifact tensor (by
//! the container's `format` + `layout`, ADR 0002: the container is the
//! authority) to the buffer its **target kernel** expects.
//!
//! **The rule (mixed-quant materialization / normalization — NOT a blanket
//! dequant):**
//!
//! - **NVFP4** (`blockscale-k16-m128x4-v1`): **preserved** — the packed E2M1
//!   code plane, the E4M3FN scale plane, and the trailing FP32
//!   `weight_divisor` (the dequant scale; the reference applies it to the
//!   group scales as `coeff = e4m3_scale * 1/divisor`). The `ignis_nvfp4_*`
//!   GEMM kernels consume the codes + scales directly.
//! - **BF16 / FP32 / I32** (`contiguous-le-v1`): **preserved as-is** (the
//!   norms, `gdn/convolution`, `gdn/a_b_projection`, the GDN recurrence
//!   params `a_log` / `dt_bias`, the `*_input_scale_divisor` scalars — the
//!   kernels consume the format directly).
//! - **W8 / Q4 / Q5 / Q6** (`row-split-k128-v1`): the *exceptional* formats
//!   — **dequantized to bf16** (the `text/token_embedding` +
//!   `text/output_head` W8G32 endpoints, the mtp weights, the vision
//!   backbone).
//!
//! The dequant math is the reference's rowsplit decode atoms
//! (`linear/{w8,q4,q5,q6}/*_rowsplit_storage.cuh`): per-group F16 scale
//! (2 bytes LE) × the signed code, converted to bf16 (round-to-nearest,
//! matching the kernel's `__float2bfloat16_rn` convention).
//!
//! **Out of scope (v1, spec 04 non-goals):** the FP8 profile
//! (`FP8_E4M3FN_ROW_BF16S` — not this artifact), the *fused-kernel* dequant
//! (the dequant is host-side for now; the fused-kernel dequant is the later
//! performance material, ADR 0005), and the vision / mtp / dflash2 /
//! draft_head tensors (not in the v1 *text* scope — noted, not normalized
//! for v1 text; see the [`crate::inventory`] scope rule).

use crate::{
    block_scale_geometry, fail, row_split_geometry, tensor_encoded_size, NumericFormat, Object,
    Result, StorageLayout, Reader,
};

// ---------------------------------------------------------------------------
// The normalized form
// ---------------------------------------------------------------------------

/// The kernel-expected buffer for one normalized artifact tensor (the A1
/// normalization result, spec 04).
#[derive(Debug, Clone)]
pub enum NormalizedTensor {
    /// NVFP4 (preserved): the packed E2M1 codes (`[m][k/2]` bytes, 2 codes
    /// per byte), the E4M3FN group scales (`[m][k/16]` bytes, one per
    /// 16-element group), and the per-tensor FP32 `weight_divisor` (the
    /// dequant scale; the kernel applies it to the group scales as
    /// `coeff = e4m3_scale * 1/divisor`). `m` = output rows, `k` = input
    /// dim (the payload's rank-two shape).
    Nvfp4 {
        m: u64,
        k: u64,
        codes: Vec<u8>,
        scales: Vec<u8>,
        divisor: f32,
    },
    /// A contiguous format preserved as-is (BF16 / FP32 / I32): the raw
    /// little-endian word bytes, verbatim (the payloads the kernels consume
    /// directly).
    Contiguous {
        format: NumericFormat,
        shape: Vec<u64>,
        bytes: Vec<u8>,
    },
    /// An exceptional row-split format dequantized to bf16 (W8 / Q4 / Q5 /
    /// Q6): row-major `[rows][cols]` bf16 words (the payload's padded-K
    /// columns are not part of the output).
    DequantBf16 {
        rows: u64,
        cols: u64,
        data: Vec<u16>,
    },
}

impl NormalizedTensor {
    /// The element count this tensor normalizes to (NVFP4: `m * k`; the
    /// contiguous families: the shape's product; the dequant family:
    /// `rows * cols`).
    pub fn element_count(&self) -> u64 {
        match self {
            NormalizedTensor::Nvfp4 { m, k, .. } => m * k,
            NormalizedTensor::Contiguous { shape, .. } => shape.iter().product(),
            NormalizedTensor::DequantBf16 { rows, cols, .. } => rows * cols,
        }
    }
}

// ---------------------------------------------------------------------------
// The normalization rule
// ---------------------------------------------------------------------------

/// Map one artifact tensor (its `format` + `layout` + `shape`, and its raw
/// payload bytes) to the kernel-expected buffer (the A1 rule: preserve
/// NVFP4 / BF16 / FP32 / I32, dequant the exceptional row-split W8 / Q4 /
/// Q5 / Q6 formats to bf16).
///
/// The payload must be the tensor's exact stored bytes (the layout's
/// encoded size — the container enforces this at read time; the check is
/// re-verified here so the function is safe standalone).
pub fn normalize_tensor(
    shape: &[u64],
    format: NumericFormat,
    layout: StorageLayout,
    payload: &[u8],
) -> Result<NormalizedTensor> {
    // The layout's exact encoded size (the container guarantees it; the
    // re-check makes the standalone function total — a truncated or padded
    // payload is a fault, not a mis-normalization).
    let encoded = tensor_encoded_size(layout, format, shape)?;
    if payload.len() as u64 != encoded {
        return Err(fail(format!(
            "payload is {} bytes, but the {:?} payload for shape {:?} encodes to {} bytes",
            payload.len(),
            layout,
            shape,
            encoded
        )));
    }

    match (layout, format) {
        // ── NVFP4: preserved (codes + scales + weight divisor) ───────────
        (StorageLayout::BlockScaleK16M128x4V1, NumericFormat::Nvfp4) => {
            let g = block_scale_geometry(format, shape)?;
            let code_end = g.code_plane_bytes as usize;
            let scale_start = g.scale_plane_offset as usize;
            let scale_end = scale_start + g.scale_plane_bytes as usize;
            let divisor_start = g.weight_divisor_offset as usize;
            Ok(NormalizedTensor::Nvfp4 {
                m: shape[0],
                k: shape[1],
                codes: payload[..code_end].to_vec(),
                scales: payload[scale_start..scale_end].to_vec(),
                divisor: f32::from_le_bytes(
                    payload[divisor_start..divisor_start + 4]
                        .try_into()
                        .map_err(|_| fail("NVFP4 payload is missing the trailing weight divisor"))?,
                ),
            })
        }

        // ── BF16 / FP32 / I32: preserved as-is ───────────────────────────
        (
            StorageLayout::ContiguousLeV1,
            NumericFormat::Bf16 | NumericFormat::Fp32 | NumericFormat::I32,
        ) => Ok(NormalizedTensor::Contiguous {
            format,
            shape: shape.to_vec(),
            bytes: payload.to_vec(),
        }),

        // ── W8 / Q4 / Q5 / Q6: dequantized to bf16 (the exceptional
        //     formats) ────────────────────────────────────────────────────
        (
            StorageLayout::RowSplitK128V1,
            NumericFormat::W8G32F16S
            | NumericFormat::Q4G64F16S
            | NumericFormat::Q5G64F16S
            | NumericFormat::Q6G64F16S,
        ) => dequant_row_split_bf16(format, shape, payload),

        // ── out of scope (the v1 rule is closed) ─────────────────────────
        (_, NumericFormat::Fp8E4M3FnRowBf16S) => Err(fail(
            "FP8_E4M3FN_ROW_BF16S is not in the mixed-quant normalization v1 scope \
             (the FP8 profile is not this artifact, spec 04)",
        )),
        _ => Err(fail(format!(
            "no normalization rule for the ({:?}, {:?}) combination \
             (the v1 rule covers NVFP4, BF16 / FP32 / I32, and W8 / Q4 / Q5 / Q6)",
            format, layout
        ))),
    }
}

/// Dequantize a `row-split-k128-v1` payload (W8 / Q4 / Q5 / Q6) to a
/// row-major `[rows][cols]` bf16 buffer (the reference's rowsplit decode
/// atoms: the per-element code × the group's F16 scale; the payload's plane
/// layout is the reader-computed geometry — low plane, then the high plane
/// (Q5 / Q6 only), then the F16 scale plane, ADR 0002).
///
/// **Signed-code convention (Q4 / Q5 / Q6, NOT pinned for #27):** the decode
/// below uses the two's-complement form for the signed codes (the W8 arm is a
/// plain signed int8 — no zero-point; the Q4 / Q5 / Q6 arms are the
/// `(code ^ zero_point) - zero_point` form). Q4 / Q5 / Q6 are the vision /
/// mtp backbones (out of the v1 *text* scope — the 27B text artifact's
/// `text/*` objects carry only the two W8 endpoints), so this convention is
/// **not load-bearing for #27** and is left as-is here. It must be pinned
/// against the reference's `*_rowsplit_storage.cuh` (the reference's Q4 / Q5
/// / Q6 signed-code bit-packing) before the vision scope is normalized.
fn dequant_row_split_bf16(
    format: NumericFormat,
    shape: &[u64],
    payload: &[u8],
) -> Result<NormalizedTensor> {
    let g = row_split_geometry(format, shape)?;
    let rows = g.rows as usize;
    let cols = g.columns as usize;
    let group_size = g.group_size as usize;
    let groups_per_row = g.groups_per_row as usize;
    let low = &payload[..g.low_plane_bytes as usize];
    let high_off = g.high_plane_offset as usize;
    let high = &payload[high_off..high_off + g.high_plane_bytes as usize];
    let scale_off = g.scale_plane_offset as usize;
    let scales = &payload[scale_off..scale_off + g.scale_plane_bytes as usize];

    // The per-element signed code (the reference's per-format bit layout;
    // the code index `ci` is the element's position within its group of
    // `group_size`).
    let decode_code = |ci: usize, gi: usize| -> i32 {
        match format {
            NumericFormat::W8G32F16S => {
                // 32 int8 codes per group (one signed byte per element).
                i32::from(low[gi * 32 + ci] as i8)
            }
            NumericFormat::Q4G64F16S => {
                // 2 four-bit codes per byte (even element = the low
                // nibble): q = (nibble ^ 8) - 8 (unsigned 4-bit -> signed
                // -8..7).
                let b = low[gi * 32 + ci / 2];
                let nib = ((b >> (ci % 2) * 4) & 0x0F) as i32;
                (nib ^ 8) - 8
            }
            NumericFormat::Q5G64F16S => {
                // 4 low bits + 1 high bit per element (8 high bits per
                // high byte): q = (low4 | (high_bit << 4)) ^ 16 - 16
                // (-16..15).
                let b = low[gi * 32 + ci / 2];
                let hb = high[gi * 8 + ci / 8];
                let low4 = ((b >> (ci % 2) * 4) & 0x0F) as i32;
                let high_bit = ((hb >> (ci % 8)) & 1) as i32;
                ((low4 | (high_bit << 4)) ^ 16) - 16
            }
            NumericFormat::Q6G64F16S => {
                // 4 low bits + 2 high bits per element (4 high bits per
                // high byte): q = (low4 | (high2 << 4)) ^ 32 - 32
                // (-32..31).
                let b = low[gi * 32 + ci / 2];
                let hb = high[gi * 16 + ci / 4];
                let low4 = ((b >> (ci % 2) * 4) & 0x0F) as i32;
                let high2 = ((hb >> (ci % 4) * 2) & 3) as i32;
                ((low4 | (high2 << 4)) ^ 32) - 32
            }
            // Unreachable: `normalize_tensor` only routes a row-split
            // quantized format here (the match arm's closed set).
            _ => unreachable!("dequant_row_split_bf16 only decodes W8 / Q4 / Q5 / Q6"),
        }
    };

    let mut data = vec![0u16; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            // The element's position in the payload (the row-major element
            // index over the *padded* columns; the group index is
            // per-row-major: row r's groups are [r*gpr, (r+1)*gpr)).
            let e = r * groups_per_row * group_size + c;
            let gi = e / group_size;
            let ci = e % group_size;
            let scale = f16_to_f32(u16::from_le_bytes(
                scales[gi * 2..gi * 2 + 2]
                    .try_into()
                    .map_err(|_| fail("row-split scale plane is truncated"))?,
            ));
            data[r * cols + c] = f32_to_bf16(decode_code(ci, gi) as f32 * scale);
        }
    }

    Ok(NormalizedTensor::DequantBf16 {
        rows: g.rows,
        cols: g.columns,
        data,
    })
}

// ---------------------------------------------------------------------------
// The W8 endpoint dequant (the A1 exceptional formats, host-side, ADR 0005)
// ---------------------------------------------------------------------------

/// The two W8 text-scope endpoints dequantized to host-side bf16 (the A1
/// exceptional formats, ADR 0005): the `text/token_embedding` (the embedding
/// table) + the `text/output_head` (the lm_head). Only these two tensors are
/// materialized to the host — the NVFP4 GEMM planes stay device-resident (not
/// host-copied; the #26 lesson: no host-weight explosion on the load path).
#[derive(Debug, Clone)]
pub struct W8Endpoints {
    /// The `text/token_embedding` (W8G32 -> bf16, the embedding table
    /// `[vocab][hidden]`).
    pub embedding: Vec<u16>,
    /// The embedding's stored shape (the `text/token_embedding`'s
    /// `[rows][cols]` = `[vocab][hidden]`).
    pub embedding_shape: (u64, u64),
    /// The `text/output_head` (W8G32 -> bf16, the lm_head `[vocab][hidden]`).
    pub lm_head: Vec<u16>,
    /// The lm_head's stored shape (the `text/output_head`'s `[rows][cols]`
    /// = `[vocab][hidden]`).
    pub lm_head_shape: (u64, u64),
}

/// Dequantize the two W8 text-scope endpoints (the `text/token_embedding` +
/// the `text/output_head`, the A1 exceptional formats) to host-side bf16
/// (ADR 0005). Only these two tensors are read + copied to the host (the
/// NVFP4 GEMM planes stay device-resident — the #26 lesson: no host weight
/// explosion on the load path; the whole-text-scope copy is A3's, not A1's).
pub fn dequant_w8_endpoints(reader: &Reader) -> Result<W8Endpoints> {
    let (embedding, embedding_shape) = dequant_w8_endpoint(reader, "text/token_embedding")?;
    let (lm_head, lm_head_shape) = dequant_w8_endpoint(reader, "text/output_head")?;
    Ok(W8Endpoints {
        embedding,
        embedding_shape,
        lm_head,
        lm_head_shape,
    })
}

/// One W8 endpoint's dequant (the reader's `name` -> the dequantized bf16
/// buffer + the stored shape, the `normalize_tensor` result's `DequantBf16`
/// variant).
fn dequant_w8_endpoint(reader: &Reader, name: &str) -> Result<(Vec<u16>, (u64, u64))> {
    let object = reader
        .find(name)
        .ok_or_else(|| fail(format!("the W8 endpoint `{name}` is absent from the artifact")))?;
    let Object::Tensor(desc) = object else {
        return Err(fail(format!("`{name}` is not a tensor object")));
    };
    let span = reader.payload_at(object)?;
    match normalize_tensor(&desc.shape, desc.format, desc.layout, span.data)? {
        NormalizedTensor::DequantBf16 { rows, cols, data } => Ok((data, (rows, cols))),
        other => Err(fail(format!(
            "`{name}` must normalize to the dequantized bf16 (the W8 exceptional format), \
             got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Numeric primitives (the reference's `__half2float` /
// `__float2bfloat16_rn` conventions, host-side)
// ---------------------------------------------------------------------------

/// Decode an IEEE 754 half-precision (F16) word to f32 (exact; the
/// reference's `__half2float` on the rowsplit group scales).
fn f16_to_f32(v: u16) -> f32 {
    let sign = ((v as u32) >> 15) & 1;
    let exp = ((v as u32) >> 10) & 0x1F;
    let mant = (v as u32) & 0x3FF;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // ±0
        } else {
            // Subnormal half: value = mant * 2^-24, normalized to an f32.
            // `p` = the 10-bit mantissa's leading-zero count (0..9, so
            // `mant` is in [2^(9-p), 2^(10-p))): E = 112 - p,
            // M23 = (mant - 2^(9-p)) << (14 + p). (`p` is the
            // leading-zero count in the 10-bit field — the u32
            // `leading_zeros` counts the 22 leading zeros of the 10-bit
            // value's zero-extension, so `p = leading_zeros - 22`.)
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
    f32::from_bits(bits)
}

/// Encode an f32 as a bf16 word (round-to-nearest-even, mirroring the
/// kernel's `__float2bfloat16_rn` convention).
fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = ((b >> 16) & 1) as u32;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

// ---------------------------------------------------------------------------
// Tests (CPU-only, synthetic fixtures — ADR 0006: no GPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{fixture, Reader};

    /// The hand-picked F16 test constants (exactly-representable values;
    /// the expected bf16 words below are hand-derived from them).
    const F16_ONE: u16 = 0x3C00; // 1.0
    const F16_ONE_POINT_FIVE: u16 = 0x3E00; // 1.5
    const F16_TWO: u16 = 0x4000; // 2.0

    /// One fixture holding a tensor per normalization family (NVFP4, BF16,
    /// FP32, I32, W8, Q4, Q5, Q6 — the closed v1 rule's input space).
    ///
    /// The payloads are deterministic (each object's region is filled with
    /// a known pattern) so the expected normalized buffers are hand-
    /// computable (the tests' independent reference). The offsets are
    /// 256-aligned ascending (the reader's invariants).
    fn family_fixture(tag: &str) -> (fixture::TempArtifact, Reader, Vec<u8>) {
        // (name, shape, format, layout, encoded bytes) — the geometry's
        // exact sizes (the fixture's directory entries).
        let spec: [(&str, Vec<u64>, &str, &str, u64); 8] = [
            ("w/nvfp4", vec![128, 64], "NVFP4", "blockscale-k16-m128x4-v1", 4612),
            ("w/bf16", vec![2, 4], "BF16", "contiguous-le-v1", 16),
            ("w/fp32-scalar", vec![], "FP32", "contiguous-le-v1", 4),
            ("w/i32", vec![3], "I32", "contiguous-le-v1", 12),
            ("w/w8", vec![2, 64], "W8G32_F16S", "row-split-k128-v1", 272),
            ("w/q4", vec![2, 128], "Q4G64_F16S", "row-split-k128-v1", 264),
            ("w/q5", vec![2, 128], "Q5G64_F16S", "row-split-k128-v1", 520),
            ("w/q6", vec![2, 128], "Q6G64_F16S", "row-split-k128-v1", 520),
        ];

        // 256-aligned ascending offsets (the reader's invariant).
        let mut objects = Vec::with_capacity(spec.len());
        let mut offsets: Vec<(u64, u64)> = Vec::with_capacity(spec.len()); // (start, end)
        let mut offset = 0u64;
        for (name, shape, format, layout, bytes) in &spec {
            if !offsets.is_empty() {
                offset = (offset + 255) / 256 * 256;
            }
            offsets.push((offset, offset + bytes));
            objects.push(fixture::FixtureObject::Tensor {
                name,
                shape: shape.clone(),
                format,
                layout,
                offset,
                bytes: *bytes,
            });
            offset += bytes;
        }
        let total = (offset + 255) / 256 * 256;
        let mut payload = vec![0u8; total as usize];

        // The region helpers (each object's [start, end)).
        let region = |i: usize| offsets[i].0;
        let _ = region; // the fills below use the offsets directly.

        // ── NVFP4 (4612 bytes): codes 0x42 + scales 0x77 + divisor 1000.0
        let nv = offsets[0].0 as usize;
        payload[nv..nv + 4096].fill(0x42);
        payload[nv + 4096..nv + 4608].fill(0x77);
        payload[nv + 4608..nv + 4612].copy_from_slice(&1000.0f32.to_le_bytes());

        // ── BF16 [2, 4]: the hand-picked bf16 words 1.0, -2.5, 0.0, 3.5
        // (the exact u16 bit patterns: 0x3F80, 0xC048, 0x0000, 0x40C0).
        let b = offsets[1].0 as usize;
        for (i, bits) in [0x3F80u16, 0xC048, 0x0000, 0x40C0].iter().enumerate() {
            payload[b + i * 2..b + i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }

        // ── FP32 rank-0 scalar (the `*_input_scale_divisor` family): 2.5.
        let f = offsets[2].0 as usize;
        payload[f..f + 4].copy_from_slice(&2.5f32.to_le_bytes());

        // ── I32 [3]: 1, -2, 3 (little-endian).
        let i3 = offsets[3].0 as usize;
        for (i, v) in [1i32, -2, 3].iter().enumerate() {
            payload[i3 + i * 4..i3 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // ── W8 [2, 64] (272 bytes): 8 groups of 32 int8 codes (the
        // per-element pattern `(ci * 7) % 17 - 8` in [-8, 8]) + 8 F16
        // scales (even groups 2.0, odd groups 1.5).
        let w8 = offsets[4].0 as usize;
        for gi in 0..8usize {
            for ci in 0..32usize {
                payload[w8 + gi * 32 + ci] =
                    (((ci * 7) % 17) as i32 - 8) as i8 as u8;
            }
            let scale = if gi % 2 == 0 { F16_TWO } else { F16_ONE_POINT_FIVE };
            payload[w8 + 256 + gi * 2..w8 + 258 + gi * 2].copy_from_slice(&scale.to_le_bytes());
        }

        // ── Q4 [2, 128] (264 bytes): low plane 0x1B (nibbles 11, 1),
        // high plane absent (Q4 has 0 high bytes), scale plane F16 1.0.
        let q4 = offsets[5].0 as usize;
        payload[q4..q4 + 128].fill(0x1B);
        for gi in 0..4usize {
            payload[q4 + 256 + gi * 2..q4 + 258 + gi * 2]
                .copy_from_slice(&F16_ONE.to_le_bytes());
        }

        // ── Q5 [2, 128] (520 bytes): low plane 0x1B, high plane 0x00
        // (8 bytes/group at offset 256), scale plane F16 1.0 (at 512).
        let q5 = offsets[6].0 as usize;
        payload[q5..q5 + 128].fill(0x1B);
        for gi in 0..4usize {
            payload[q5 + 512 + gi * 2..q5 + 514 + gi * 2]
                .copy_from_slice(&F16_ONE.to_le_bytes());
        }

        // ── Q6 [2, 128] (520 bytes): low plane 0x1B, high plane 0xFF (16
        // bytes/group at offset 256 — every high2 = 3, the negative
        // 6-bit range), scale plane F16 1.0 (at 512).
        let q6 = offsets[7].0 as usize;
        payload[q6..q6 + 128].fill(0x1B);
        payload[q6 + 256..q6 + 320].fill(0xFF);
        for gi in 0..4usize {
            payload[q6 + 512 + gi * 2..q6 + 514 + gi * 2]
                .copy_from_slice(&F16_ONE.to_le_bytes());
        }

        let artifact = fixture::write_fixture(&objects, &payload, tag).expect("fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        (artifact, reader, payload)
    }

    /// The payload span of `name` in the fixture (the reader's mapping).
    fn span<'a>(reader: &'a Reader, name: &str) -> &'a [u8] {
        let object = reader.find(name).expect("object present");
        reader.payload_at(object).expect("span").data
    }

    /// NVFP4: the payload's code plane, scale plane, and trailing FP32
    /// weight divisor are preserved verbatim (no dequant — the GEMM kernel
    /// consumes the planes directly, spec 04).
    #[test]
    fn nvfp4_normalization_preserves_planes_and_divisor() {
        let (_art, reader, payload) = family_fixture("norm-nvfp4");
        let n = normalize_tensor(
            &[128, 64],
            NumericFormat::Nvfp4,
            StorageLayout::BlockScaleK16M128x4V1,
            span(&reader, "w/nvfp4"),
        )
        .expect("nvfp4 normalizes");
        match &n {
            NormalizedTensor::Nvfp4 { m, k, codes, scales, divisor } => {
                assert_eq!((*m, *k), (128, 64));
                // The planes are copied verbatim (the payload's exact bytes).
                assert_eq!(codes, &payload[0..4096], "the code plane is verbatim");
                assert!(codes.iter().all(|&b| b == 0x42));
                assert_eq!(scales, &payload[4096..4608], "the scale plane is verbatim");
                assert!(scales.iter().all(|&b| b == 0x77));
                // The trailing FP32 divisor (1000.0, the test's known value).
                assert_eq!(*divisor, 1000.0f32);
            }
            other => panic!("NVFP4 must normalize to its preserved planes, got {other:?}"),
        }
        assert_eq!(n.element_count(), 128 * 64);
    }

    /// BF16 / FP32 / I32 (contiguous-le-v1) are preserved as-is: the raw
    /// little-endian bytes, verbatim (the kernels consume the format
    /// directly). The bf16 words are the test's hand-picked values
    /// (1.0, -2.5, 0.0, 3.5 — the exact bit patterns asserted below).
    #[test]
    fn contiguous_formats_are_preserved_as_is() {
        let (_art, reader, _payload) = family_fixture("norm-contig");

        // BF16 [2, 4]
        let n = normalize_tensor(
            &[2, 4],
            NumericFormat::Bf16,
            StorageLayout::ContiguousLeV1,
            span(&reader, "w/bf16"),
        )
        .expect("bf16 normalizes");
        match &n {
            NormalizedTensor::Contiguous { format, shape, bytes } => {
                assert_eq!(*format, NumericFormat::Bf16);
                assert_eq!(*shape, vec![2, 4]);
                let words: [u16; 4] = [
                    u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
                    u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
                    u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
                    u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
                ];
                assert_eq!(words, [0x3F80, 0xC048, 0x0000, 0x40C0], "bf16 1.0, -2.5, 0.0, 3.5");
            }
            other => panic!("BF16 must normalize as-is, got {other:?}"),
        }

        // FP32 rank-0 scalar (the `*_input_scale_divisor` family).
        let n = normalize_tensor(
            &[],
            NumericFormat::Fp32,
            StorageLayout::ContiguousLeV1,
            span(&reader, "w/fp32-scalar"),
        )
        .expect("fp32 scalar normalizes");
        match &n {
            NormalizedTensor::Contiguous { format, shape, bytes } => {
                assert_eq!(*format, NumericFormat::Fp32);
                assert_eq!(*shape, Vec::<u64>::new());
                assert_eq!(*bytes, 2.5f32.to_le_bytes(), "the fp32 scalar is verbatim");
            }
            other => panic!("FP32 must normalize as-is, got {other:?}"),
        }
        assert_eq!(n.element_count(), 1);

        // I32 [3].
        let n = normalize_tensor(
            &[3],
            NumericFormat::I32,
            StorageLayout::ContiguousLeV1,
            span(&reader, "w/i32"),
        )
        .expect("i32 normalizes");
        match &n {
            NormalizedTensor::Contiguous { format, bytes, .. } => {
                assert_eq!(*format, NumericFormat::I32);
                let words: Vec<i32> = bytes
                    .chunks_exact(4)
                    .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
                    .collect();
                assert_eq!(words, vec![1, -2, 3], "the i32 words are verbatim");
            }
            other => panic!("I32 must normalize as-is, got {other:?}"),
        }
    }

    /// W8 (the exceptional format, the `text/token_embedding` +
    /// `text/output_head` endpoints): the dequantized bf16 buffer matches
    /// the CPU reference (code × group scale, per element). The spot-checks
    /// are hand-derived literals (independent of the implementation's
    /// decode path); the full-buffer loop recomputes the reference from the
    /// known code/scale patterns.
    #[test]
    fn w8_dequant_matches_cpu_reference() {
        let (_art, reader, _payload) = family_fixture("norm-w8");
        let n = normalize_tensor(
            &[2, 64],
            NumericFormat::W8G32F16S,
            StorageLayout::RowSplitK128V1,
            span(&reader, "w/w8"),
        )
        .expect("w8 dequants");
        let NormalizedTensor::DequantBf16 { rows, cols, data } = n else {
            panic!("W8 must normalize to dequantized bf16");
        };
        assert_eq!((rows, cols), (2, 64));
        assert_eq!(data.len(), 128);

        // The reference: the payload's element e (row-major over the 64
        // columns; 4 groups of 32 per row, over the padded 128) has
        // code `(ci * 7) % 17 - 8` (ci = e % 32 within its group of 32)
        // × the group's F16 scale (even groups 2.0, odd 1.5).
        for e in 0..128usize {
            let r = e / 64;
            let c = e % 64;
            let gi = r * 4 + c / 32; // per-row groups (128 padded / 32)
            let ci = c % 32;
            let code = ((ci * 7) % 17) as i32 - 8;
            let scale = if gi % 2 == 0 { 2.0f32 } else { 1.5f32 };
            let want = f32_to_bf16(code as f32 * scale);
            assert_eq!(
                data[e],
                want,
                "element ({r}, {c}): code {code} x scale {scale}"
            );
        }
        // Spot-check the hand-derived words (independent literals):
        // (0, 0): ci 0 -> code -8, group 0 scale 2.0 -> -16.0 -> 0xC180.
        assert_eq!(data[0], 0xC180, "(-8 x 2.0) -> bf16 -16.0");
        // (0, 1): ci 1 -> code -1, group 0 scale 2.0 -> -2.0 -> 0xC000.
        assert_eq!(data[1], 0xC000, "(-1 x 2.0) -> bf16 -2.0");
        // (0, 32): group 1 (odd) scale 1.5, ci 0 -> code -8 -> -12.0
        // -> 0xC140 (the exact f32 bits of -12.0 are 0xC1400000).
        assert_eq!(data[32], 0xC140, "(-8 x 1.5) -> bf16 -12.0");
        // (1, 63): group 5 (odd) scale 1.5, ci 31 -> (217 % 17) - 8 = 5
        // -> 7.5 -> 0x40F0.
        assert_eq!(data[127], 0x40F0, "(5 x 1.5) -> bf16 7.5");
    }

    /// Q4 / Q5 / Q6: the dequantized bf16 buffers match the hand-derived
    /// per-format bit math (the reference's decode atoms; the expected words
    /// are independent literals).
    #[test]
    fn q4_q5_q6_dequants_match_reference() {
        let (_art, reader, _payload) = family_fixture("norm-q456");
        // Q4 [2, 128]: low byte 0x1B -> low nibble 11 -> (11^8)-8 = -5,
        // high nibble 1 -> (1^8)-8 = -7... (nibble ^ 8) - 8: 11 -> -5,
        // 1 -> -7? No: (1 ^ 8) - 8 = 9 - 8 = +1. So even elements -5.0
        // (bf16 0xC0A0), odd +1.0 (bf16 0x3F80); scale 1.0.
        let n = normalize_tensor(
            &[2, 128],
            NumericFormat::Q4G64F16S,
            StorageLayout::RowSplitK128V1,
            span(&reader, "w/q4"),
        )
        .expect("q4 dequants");
        let data = n_data(&n);
        for (idx, &w) in data.iter().enumerate() {
            let want = if idx % 2 == 0 { 0xC0A0u16 } else { 0x3F80 };
            assert_eq!(w, want, "q4 element {idx}");
        }
        // Q5 [2, 128]: low byte 0x1B (low nibble 11 / high nibble 1), high
        // plane 0x00; element 2j: 11 | 0 = 11 -> 11^16 = 27 -> 27-16 = 11;
        // element 2j+1: 1 -> 1^16 = 17 -> 17-16 = 1; scale 1.0. Even
        // 11.0 (bf16 0x4130), odd 1.0 (bf16 0x3F80).
        let n = normalize_tensor(
            &[2, 128],
            NumericFormat::Q5G64F16S,
            StorageLayout::RowSplitK128V1,
            span(&reader, "w/q5"),
        )
        .expect("q5 dequants");
        let data = n_data(&n);
        for (idx, &w) in data.iter().enumerate() {
            let want = if idx % 2 == 0 { 0x4130u16 } else { 0x3F80 };
            assert_eq!(w, want, "q5 element {idx}");
        }
        // Q6 [2, 128]: low byte 0x1B, high plane 0xFF (every high2 = 3);
        // element 2j: X = 11 | (3 << 4) = 59 -> (59 ^ 32) - 32 = -5;
        // element 2j+1: X = 1 | (3 << 4) = 49 -> (49 ^ 32) - 32 = -15;
        // scale 1.0. Even -5.0 (bf16 0xC0A0), odd -15.0 (bf16 0xC170).
        let n = normalize_tensor(
            &[2, 128],
            NumericFormat::Q6G64F16S,
            StorageLayout::RowSplitK128V1,
            span(&reader, "w/q6"),
        )
        .expect("q6 dequants");
        let data = n_data(&n);
        for (idx, &w) in data.iter().enumerate() {
            let want = if idx % 2 == 0 { 0xC0A0u16 } else { 0xC170 };
            assert_eq!(w, want, "q6 element {idx}");
        }
    }

    /// The v1 rule is closed: the FP8 profile, an out-of-rule combination,
    /// and a truncated payload are all rejected with a clear error (not a
    /// silent mis-normalization).
    #[test]
    fn out_of_scope_inputs_are_rejected() {
        let (_art, reader, payload) = family_fixture("norm-reject");
        // FP8 (the FP8 profile — not this artifact, spec 04 non-goal).
        let err = normalize_tensor(
            &[128, 64],
            NumericFormat::Fp8E4M3FnRowBf16S,
            StorageLayout::RowScaleV1,
            &vec![0u8; 8448],
        )
        .expect_err("fp8 is out of scope");
        assert!(err.to_string().contains("FP8"), "{err}");
        // A payload shorter than the layout's encoded size.
        let err = normalize_tensor(
            &[2, 4],
            NumericFormat::Bf16,
            StorageLayout::ContiguousLeV1,
            &payload[4864..4866],
        )
        .expect_err("a truncated payload must fail");
        assert!(err.to_string().contains("bytes"), "{err}");
        // An out-of-rule (format, layout) combination (W8 in contiguous).
        let err = normalize_tensor(
            &[2, 64],
            NumericFormat::W8G32F16S,
            StorageLayout::ContiguousLeV1,
            &payload[5632..5640],
        )
        .expect_err("an out-of-rule combination must fail");
        assert!(
            err.to_string().contains("contiguous-le-v1"),
            "the geometry's closed-registry check rejects it: {err}"
        );
        // And the reader's span (the object's exact stored bytes) is what
        // the normalize step consumes in production (a sanity pin: the span
        // is the object's full payload, not a prefix).
        assert_eq!(span(&reader, "w/w8").len(), 272);
    }

    /// The f16 -> f32 / f32 -> bf16 primitives against hand-picked values
    /// (the exact-F16 constants the group scales use; the bf16 words are
    /// hand-derived).
    #[test]
    fn numeric_primitives_match_hand_values() {
        // f16 -> f32 (exact): the hand-picked constants decode to their
        // known f32 values.
        assert_eq!(f16_to_f32(F16_ONE), 1.0f32);
        assert_eq!(f16_to_f32(F16_TWO), 2.0f32);
        assert_eq!(f16_to_f32(0x3E00), 1.5f32);
        assert_eq!(f16_to_f32(0x3800), 0.5f32);
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert_eq!(f16_to_f32(0x8000), 0.0f32); // -0.0 (the sign bit)
        // f16 subnormals (exp = 0, `mant` in 1..1023): the rowsplit scale
        // planes carry small magnitudes, so subnormal F16 words are a
        // real case (the W8 dequant's group scales). Value =
        // `mant * 2^-24` (the subnormal convention), hand-derived:
        //   0x0001 = 1 * 2^-24        -> 5.960464477539063e-8
        //   0x0200 = 512 * 2^-24      -> 2^-15 = 3.0517578125e-5
        //   0x03FF = 1023 * 2^-24     -> 6.10351562499e-5 (the max subnormal)
        assert_eq!(f16_to_f32(0x0001), 1.0f32 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0200), 2.0f32.powi(-15));
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        // The subnormal boundary (the smallest normal, 0x0400 = 2^-14,
        // must match the largest subnormal's successor).
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        // f32 -> bf16 (RNE): the hand-derived words (the exact f32 bit
        // patterns: 11.0 = 0x41300000, -11.0 = 0xC1300000, -16.0 =
        // 0xC1800000 — the top 16 bits, no rounding needed).
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(-2.0), 0xC000);
        assert_eq!(f32_to_bf16(6.0), 0x40C0);
        assert_eq!(f32_to_bf16(11.0), 0x4130);
        assert_eq!(f32_to_bf16(-11.0), 0xC130);
        assert_eq!(f32_to_bf16(-16.0), 0xC180);
        assert_eq!(f32_to_bf16(7.5), 0x40F0);
        assert_eq!(f32_to_bf16(0.0), 0x0000);
    }

    /// A dequantized tensor's bf16 words (the test's accessor over the
    /// `DequantBf16` variant).
    fn n_data(n: &NormalizedTensor) -> &[u16] {
        match n {
            NormalizedTensor::DequantBf16 { data, .. } => data,
            other => panic!("expected a dequantized bf16 tensor, got {other:?}"),
        }
    }

    /// A fixture holding the two W8 endpoints (`text/token_embedding` +
    /// `text/output_head`, the `dequant_w8_endpoints` input). Each endpoint's
    /// W8 payload is filled with a known code (int8 3) + scale (F16 2.0), so
    /// the dequant is hand-computable (3 × 2.0 = 6.0 -> bf16 0x40C0).
    fn w8_endpoint_fixture(tag: &str) -> (fixture::TempArtifact, Reader) {
        // (name, shape) — the two W8 endpoints (small, CPU-friendly).
        let spec: [(&str, Vec<u64>); 2] = [
            ("text/token_embedding", vec![2, 32]),
            ("text/output_head", vec![4, 64]),
        ];
        // The W8 (W8G32) geometry per shape (the encoded size + the plane
        // offsets, the reader's invariants).
        let geoms: Vec<_> = spec
            .iter()
            .map(|(_, shape)| crate::row_split_geometry(NumericFormat::W8G32F16S, shape).unwrap())
            .collect();
        // 256-aligned ascending offsets (the reader's invariant).
        let mut objects = Vec::with_capacity(2);
        let mut offsets: Vec<u64> = Vec::with_capacity(2);
        let mut offset = 0u64;
        for (i, (name, shape)) in spec.iter().enumerate() {
            if !offsets.is_empty() {
                offset = (offset + 255) / 256 * 256;
            }
            offsets.push(offset);
            objects.push(fixture::FixtureObject::Tensor {
                name: *name,
                shape: shape.clone(),
                format: "W8G32_F16S",
                layout: "row-split-k128-v1",
                offset,
                bytes: geoms[i].encoded_bytes,
            });
            offset += geoms[i].encoded_bytes;
        }
        let total = (offset + 255) / 256 * 256;
        let mut payload = vec![0u8; total as usize];
        // Each endpoint's W8 payload: the code plane (int8 3) + the scale
        // plane (F16 2.0 = 0x4000) -> the dequant is 3 * 2.0 = 6.0
        // (bf16 0x40C0).
        for (i, _) in spec.iter().enumerate() {
            let g = &geoms[i];
            let base = offsets[i] as usize;
            payload[base..base + g.low_plane_bytes as usize].fill(3i8 as u8);
            let scale_off = base + g.scale_plane_offset as usize;
            let scale_count = (g.scale_plane_bytes / 2) as usize;
            for gi in 0..scale_count {
                payload[scale_off + gi * 2..scale_off + gi * 2 + 2]
                    .copy_from_slice(&0x4000u16.to_le_bytes());
            }
        }
        let artifact = fixture::write_fixture(&objects, &payload, tag).expect("fixture");
        let reader = Reader::open(&artifact.path).expect("open fixture");
        (artifact, reader)
    }

    /// The two W8 endpoints (the A1 exceptional formats, ADR 0005 host-side
    /// dequant) dequantize to the hand-computed bf16 (code 3 × scale 2.0 =
    /// 6.0 -> 0x40C0), and `dequant_w8_endpoints` returns only the two W8
    /// buffers (the NVFP4 / other tensors are not copied — the #26 lesson:
    /// the NVFP4 planes stay device-resident, not host-copied).
    #[test]
    fn dequant_w8_endpoints_returns_only_the_two_w8_endpoints() {
        let (_art, reader) = w8_endpoint_fixture("w8-endpoints");
        let endpoints = dequant_w8_endpoints(&reader).expect("dequant the W8 endpoints");
        // The endpoint shapes (the stored `[rows][cols]`).
        assert_eq!(endpoints.embedding_shape, (2, 32));
        assert_eq!(endpoints.lm_head_shape, (4, 64));
        // The dequant content (the hand-derived value: 3 * 2.0 = 6.0 ->
        // bf16 0x40C0), carried to the host buffer (not empty / not zero).
        assert_eq!(endpoints.embedding.len(), 2 * 32);
        assert!(
            endpoints.embedding.iter().all(|&v| v == 0x40C0),
            "the W8 embedding dequants to bf16 6.0"
        );
        assert_eq!(endpoints.lm_head.len(), 4 * 64);
        assert!(
            endpoints.lm_head.iter().all(|&v| v == 0x40C0),
            "the W8 lm_head dequants to bf16 6.0"
        );
    }
}