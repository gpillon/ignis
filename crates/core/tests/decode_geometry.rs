//! CPU-verifiable geometry/quant tests for the ticket-03 decode C-ABI surface.
//!
//! Re-implements, 1:1, the index and quant-decode math that the leaf kernels
//! use, and verifies it against hand-computed reference values. This is pure
//! Rust — no GPU, no FFI, no kernel .lib symbols — so it runs green in
//! isolation (ADR 0006: the GPU is reserved for the gated launch tests).
//!
//! The formulas mirror the reference (ninfer) index math:
//!   - `paged_kv_element_offset` : ops/kernel/paged_kv_address.cuh
//!   - `nvfp4_scale_offset`      : ops/linear/nvfp4/nvfp4_codec.cuh
//!   - E2M1 / E4M3 decode        : ops/linear/nvfp4/nvfp4_codec.cuh
//!   - GQA head grouping          : ops/kernel/gqa_attention_geometry.cuh
//!
//! The leaf kernels (kernel/src/nvfp4_gemm_decode.cuh,
//! kernel/src/gqa_attention_decode.cuh) use these same formulas, so these tests
//! pin the CPU-verifiable core of the decode geometry.

/// 1:1 with kernel/src/gqa_attention_decode.cuh::`paged_kv_element_offset` and
/// the reference `ops/kernel/paged_kv_address.cuh::paged_kv_element_offset`.
/// Layout: [physical_page][kv_head][block_offset][d] with `d` fastest.
fn paged_kv_element_offset(
    head_dim: i64,
    num_kv_heads: i64,
    block_size: i64,
    physical_page: i32,
    kv_head: i32,
    block_offset: i32,
    d: i32,
) -> i64 {
    head_dim * block_size * (kv_head as i64 + num_kv_heads * physical_page as i64)
        + head_dim * (block_offset as i64)
        + d as i64
}

/// 1:1 with the reference `ops/linear/nvfp4/nvfp4_codec.cuh::nvfp4_scale_offset`.
/// Blockscale-K16 scale-plane addressing shared by every reader of an NVFP4
/// weight: group `g` of parent row `p` maps to a position in a 512-element
/// scale tile (the E4M3 group-scale plane), swizzled by 32-row quartiles.
fn nvfp4_scale_offset(parent_row: i32, group: i32, scale_tiles_per_row: i32) -> i64 {
    let m_tile = parent_row / 128;
    let row_inner = parent_row - m_tile * 128;
    let scale_tile = group / 4;
    let scale_lane = group & 3;
    let row_mod32 = row_inner & 31;
    let row_quartile = row_inner >> 5;
    (m_tile as i64 * scale_tiles_per_row as i64 + scale_tile as i64) * 512
        + row_mod32 as i64 * 16
        + row_quartile as i64 * 4
        + scale_lane as i64
}

/// 1:1 with kernel/src/nvfp4_gemm_decode.cuh::`decode_nvfp4_e2m1`.
/// E2M1 (FP4): 1 sign bit (0x8), 3 magnitude bits -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
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
    if code & 0x8 != 0 { -mag } else { mag }
}

/// 1:1 with kernel/src/nvfp4_gemm_decode.cuh::`decode_nvfp4_e4m3`.
/// OCP FP8 E4M3 (bias 7, no inf): 1 sign bit, 4 exponent bits, 3 mantissa bits.
/// Subnormals (exp == 0) use (m/8) * 2^-6; normals use (1 + m/8) * 2^(exp-7).
fn decode_e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((code >> 3) & 0x0F) as i32;
    let man = (code & 0x07) as f32;
    let mag = if exp == 0 {
        (man / 8.0) * 0.015625 // (m/8) * 2^-6 (bias 7)
    } else {
        (1.0 + man / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * mag
}

#[test]
fn paged_kv_offset_matches_reference_layout() {
    // Qwen 27B GQA decode geometry: head_dim 256, 4 kv heads, 64-key pages.
    // page 3, kv_head 2, block_offset 10, d 16:
    //   offset = 256*64*(2 + 4*3) + 256*10 + 16 = 16384*14 + 2560 + 16
    assert_eq!(
        paged_kv_element_offset(256, 4, 64, 3, 2, 10, 16),
        256 * 64 * 14 + 256 * 10 + 16
    );
    // Origin (page 0, kv_head 0, offset 0, d 0) is element 0.
    assert_eq!(paged_kv_element_offset(256, 4, 64, 0, 0, 0, 0), 0);
    // `d` is the fastest dim: consecutive d are 1 apart.
    assert_eq!(
        paged_kv_element_offset(128, 8, 64, 5, 1, 7, 3) + 1,
        paged_kv_element_offset(128, 8, 64, 5, 1, 7, 4)
    );
    // A full page (page 0 -> page 1, same kv_head/offset/d) holds
    // num_kv_heads * block_size * head_dim elements.
    assert_eq!(
        paged_kv_element_offset(64, 2, 32, 1, 0, 0, 0),
        paged_kv_element_offset(64, 2, 32, 0, 0, 0, 0) + 64 * 32 * 2
    );
}

#[test]
fn nvfp4_scale_offset_matches_reference_layout() {
    // A 5120-input problem has scale_tiles_per_row = 5120/64 = 80.
    // (parent_row 0, group 0) is the origin of the scale plane.
    assert_eq!(nvfp4_scale_offset(0, 0, 80), 0);
    // group 4 -> scale_tile 1 (lane 0) -> one 512-element tile.
    assert_eq!(nvfp4_scale_offset(0, 4, 80), 512);
    // parent_row 1 -> row_mod32 1 -> +16 within the tile.
    assert_eq!(nvfp4_scale_offset(1, 0, 80), 16);
    // parent_row 32 -> row_quartile 1, row_mod32 0 -> +4.
    assert_eq!(nvfp4_scale_offset(32, 0, 80), 4);
    // parent_row 128 -> m_tile 1 -> one full row (80 tiles * 512).
    assert_eq!(nvfp4_scale_offset(128, 0, 80), 80 * 512);
}

#[test]
fn e2m1_decode_matches_fp4_table() {
    // E2M1 unsigned magnitude codes -> {0, .5, 1, 1.5, 2, 3, 4, 6}.
    let table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for (code, want) in (0..8u8).zip(table.iter()) {
        assert_eq!(decode_e2m1(code), *want, "code {code}");
    }
    // The sign bit (0x8) negates the magnitude.
    assert_eq!(decode_e2m1(0x8 | 0x4), -2.0);
    assert_eq!(decode_e2m1(0x8 | 0x7), -6.0);
}

#[test]
fn e4m3_decode_matches_fp8_spec() {
    // OCP FP8 E4M3: hand-computed normal + subnormal values (unambiguous range).
    // bit7 = sign, bits6..3 = exponent (bias 7), bits2..0 = mantissa.
    assert_eq!(decode_e4m3(0x00), 0.0); // subnormal: 0
    assert_eq!(decode_e4m3(0x01), 0.001_953_125); // subnormal: (1/8)*2^-6
    assert_eq!(decode_e4m3(0x08), 0.015_625); // exp 1, man 0: 2^-6
    assert_eq!(decode_e4m3(0x38), 1.0); // exp 7, man 0: 2^0
    assert_eq!(decode_e4m3(0x3A), 1.25); // exp 7, man 2
    assert_eq!(decode_e4m3(0x48), 4.0); // exp 9, man 0: 2^2
    assert_eq!(decode_e4m3(0x50), 8.0); // exp 10, man 0: 2^3
    // Negative: sign bit (0x80) with exp 7, man 0 -> -1.0.
    assert_eq!(decode_e4m3(0xB8), -1.0);
}

#[test]
fn gqa_kv_head_mapping() {
    // Qwen 27B: 24 q heads, 4 kv heads, group = 24/4 = 6.
    let (num_q_heads, num_kv_heads) = (24, 4);
    assert_eq!(num_q_heads % num_kv_heads, 0);
    let group = num_q_heads / num_kv_heads;
    assert_eq!(group, 6);
    // q head h maps to kv head h / group: 0..5->0, 6..11->1, 12..17->2, 18..23->3.
    assert_eq!(0 / group, 0);
    assert_eq!(5 / group, 0);
    assert_eq!(6 / group, 1);
    assert_eq!(17 / group, 2);
    assert_eq!(23 / group, 3);
}