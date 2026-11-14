// The hq-e8-2b 8-lane group row decode, lean form -- OURS (ADR 0051 tier 1; spec #570, GitHub
// #571), the decoder of the warp-specialized verify attention's decode warps (hq_verify_ws.cuh).
// Its source is ignis's own #268 decoder, not upstream code: hq_decode_row_group_fast, which
// kernel/vendor/src/ops/kernel/hq_codec.cuh carries as a patch kernel/vendor/manifest.json records
// (kernel/vendor/patches/src/ops/kernel/hq_codec.cuh.diff; ADR 0031, GitHub #268). Bit-exact
// against hq_decode_row_group and hq_decode_row_group_fast (kernel/tests/test_hq_codec_kv_rows.cu,
// the fifth way, called as that route calls it). It builds for every architecture the leaf does.
#pragma once

#include "ops/kernel/hq_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ignis_hq_verify {

using namespace ninfer::ops;

__device__ __forceinline__ void st_shared_u16(std::uint32_t addr, std::uint32_t value) {
    asm volatile("st.shared.u16 [%0], %1;" ::"r"(addr), "h"(static_cast<std::uint16_t>(value))
                 : "memory");
}

// hq_decode_row_group_fast's values (hq_codec.cuh), the same contract (warp-converged call,
// `active` per 8-lane group, symbols staged in `out` itself), from fewer instructions in the
// walk: a chain carries the shared-memory byte address of its next symbol, and the swizzled
// slot is that address XOR the chunk flip, which needs `out` to be a 512-byte aligned row of
// shared memory (a row is 512 bytes, so the flip of bits 4..6 stays inside it). Every value is
// computed by the same expression as there; only the store addresses are formed differently.
__device__ __forceinline__ void hq_decode_row_group_lean(const std::uint8_t* codes,
                                                         const std::uint8_t* meta,
                                                         __nv_bfloat16* out, int lane,
                                                         int xor_chunk,
                                                         std::uint64_t dither_seed,
                                                         bool active) {
    const int warp_lane = static_cast<int>(threadIdx.x & 31u);
    const int gbase     = warp_lane & ~7;
    constexpr unsigned kWarp = 0xFFFFFFFFu;

    std::uint32_t mw0 = 0u, mw1 = 0u;
    if (active) {
        mw0 = *reinterpret_cast<const std::uint32_t*>(meta);
        mw1 = *reinterpret_cast<const std::uint32_t*>(meta + 4);
    }
    const unsigned used_bits = ((mw0 >> 24) & 0xFFu) | ((mw1 & 0x3u) << 8);
    const std::uint32_t k    = (mw0 >> 16) & 0xFu;
    const bool fast          = active && used_bits != 0 && k == 0;
    std::uint16_t* symbols   = reinterpret_cast<std::uint16_t*>(out);
    if (active && (used_bits == 0 || k > kHqMaxRiceK)) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<uint4*>(out + hq_swz_element((lane + 8 * i) * 8, xor_chunk)) =
                make_uint4(0u, 0u, 0u, 0u);
        }
    } else if (active && k != 0) {
        hq_decode_row_group<true>(codes, meta, out, lane, xor_chunk, dither_seed);
    }
    uint2 xw = make_uint2(0u, 0u);
    if (fast) { xw = *reinterpret_cast<const uint2*>(codes + 8 * lane); }
    const std::uint32_t x0 = xw.x;
    const std::uint32_t x1 = xw.y;

    const int c0 = __popc(x0);
    const int c  = c0 + __popc(x1);
    int inc      = c;
    int run = x1 != 0u ? __ffs(x1) - 1 : (x0 != 0u ? 31 + __ffs(x0) : 64);
    int az  = (x0 | x1) == 0u ? 1 : 0;
#pragma unroll
    for (int delta = 1; delta < 8; delta <<= 1) {
        const int left = __shfl_up_sync(kWarp, inc, delta, 8);
        const int lrun = __shfl_up_sync(kWarp, run, delta, 8);
        const int laz  = __shfl_up_sync(kWarp, az, delta, 8);
        if (lane >= delta) { inc += left; }
        if (lane >= delta && az != 0) {
            run += lrun;
            az &= laz;
        }
    }

    const int idx_total = __shfl_sync(kWarp, inc, gbase + 7);
    int carry           = __shfl_up_sync(kWarp, run, 1, 8);
    if (lane == 0) { carry = 0; }
    const int base = inc - c;
    const int e0   = min(base + c0, kHqHeadDim);
    const int e1   = min(base + c, kHqHeadDim);
    // Byte addresses of symbol i: row + 2 i, stored at (row + 2 i) ^ flip.
    const std::uint32_t row   = static_cast<std::uint32_t>(__cvta_generic_to_shared(out));
    const std::uint32_t flip  = static_cast<std::uint32_t>(xor_chunk) << 4;
    std::uint32_t a0          = row + 2u * static_cast<std::uint32_t>(base);
    std::uint32_t a1          = row + 2u * static_cast<std::uint32_t>(base + c0);
    const std::uint32_t end0  = row + 2u * static_cast<std::uint32_t>(e0);
    const std::uint32_t end1  = row + 2u * static_cast<std::uint32_t>(e1);
    std::uint32_t v0 = x0;
    std::uint32_t v1 = x1;
    const int lead1  = x0 != 0u ? __ffs(x0) - 1 : carry + 32;
    if (a0 < end0) {
        const int p = hq_clz_nonzero(v0);
        int z       = carry + p;
        if (z >= static_cast<int>(kHqUnaryGuard)) { z = 0; }
        st_shared_u16(a0 ^ flip, static_cast<std::uint32_t>(z));
        v0 = __funnelshift_lc(0u, v0, p + 1);
        a0 += 2u;
    }
    if (a1 < end1) {
        const int p = hq_clz_nonzero(v1);
        int z       = lead1 + p;
        if (z >= static_cast<int>(kHqUnaryGuard)) { z = 0; }
        st_shared_u16(a1 ^ flip, static_cast<std::uint32_t>(z));
        v1 = __funnelshift_lc(0u, v1, p + 1);
        a1 += 2u;
    }
    const int steps = max(static_cast<int>(end0 - a0), static_cast<int>(end1 - a1)) >> 1;
    int step = 0;
#pragma unroll 1
    for (; step + 2 <= steps; step += 2) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int p0 = hq_clz_nonzero(v0);
            const int p1 = hq_clz_nonzero(v1);
            if (a0 < end0) { st_shared_u16(a0 ^ flip, static_cast<std::uint32_t>(p0)); }
            if (a1 < end1) { st_shared_u16(a1 ^ flip, static_cast<std::uint32_t>(p1)); }
            v0 = __funnelshift_lc(0u, v0, p0 + 1);
            v1 = __funnelshift_lc(0u, v1, p1 + 1);
            a0 += 2u;
            a1 += 2u;
        }
    }
    if (step < steps) {
        const int p0 = hq_clz_nonzero(v0);
        const int p1 = hq_clz_nonzero(v1);
        if (a0 < end0) { st_shared_u16(a0 ^ flip, static_cast<std::uint32_t>(p0)); }
        if (a1 < end1) { st_shared_u16(a1 ^ flip, static_cast<std::uint32_t>(p1)); }
    }

    if (fast && idx_total < kHqHeadDim && lane == 0) {
#pragma unroll 1
        for (int t = idx_total; t < kHqHeadDim; ++t) { symbols[hq_swz_element(t, xor_chunk)] = 0; }
    }
    __syncwarp(kWarp);
    if (!fast) { return; }

    const std::uint32_t escalation = (mw0 >> 20) & 0x3u;
    // The norm's FP16 bits as a half by __ushort_as_half, not through a pointer cast (the
    // vendored encoder's store of these bits went wrong that way under ptxas 13.0, #600).
    const float norm = __half2float(__ushort_as_half(static_cast<unsigned short>(mw0 & 0xFFFFu)));
    const float inv_scale =
        static_cast<float>(1u << escalation) / (kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int word     = lane + 8 * i;
        const int wb       = hq_swz_element(word * 8, xor_chunk);
        const uint4 staged = *reinterpret_cast<const uint4*>(symbols + wb);
        const std::uint32_t pairs[4] = {staged.x, staged.y, staged.z, staged.w};
        const std::uint32_t coset  = (pairs[3] >> 16) & 1u;
        const std::uint32_t coset2 = coset * 0x00010001u;
        std::uint32_t y2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y2[j] = pairs[j] ^ ((pairs[j] & 0x00010001u) * 0xFFFFu) ^ coset2;
        }
        const std::uint32_t x      = y2[0] ^ y2[1] ^ y2[2] ^ (y2[3] & 0xFFFFu);
        const std::uint32_t parity = ((x >> 1) ^ (x >> 17)) & 1u;
        const std::uint32_t u      = pairs[3] >> 17;
        const std::uint32_t t2     = u ^ (0u - (u & 1u));
        int y[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[2 * j]     = static_cast<std::int16_t>(y2[j] & 0xFFFFu);
            y[2 * j + 1] = static_cast<std::int16_t>(y2[j] >> 16);
        }
        y[7] = static_cast<int>(((t2 ^ parity) << 1) ^ coset);
        const std::uint64_t wseed = hq_dither_word_seed(dither_seed, word);
        const std::uint32_t wlo   = static_cast<std::uint32_t>(wseed);
        const std::uint32_t whi   = static_cast<std::uint32_t>(wseed >> 32);
        constexpr float kInv255   = 1.0f / 255.0f;
        constexpr float kFold     = 8388608.0f * kInv255 + 0.5f;
        uint4 packed;
        auto* values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float magic = __uint_as_float(
                __byte_perm(j < 4 ? wlo : whi, 0x4B000000u, 0x7540u | static_cast<unsigned>(j & 3)));
            const float dither = __fmaf_rn(magic, kInv255, -kFold);
            values[j] = __float2bfloat16((static_cast<float>(y[j]) + dither) * norm * inv_scale);
        }
        *reinterpret_cast<uint4*>(out + wb) = packed;
    }
}

} // namespace ignis_hq_verify
