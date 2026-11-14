// The hq-e8-2b verify attention at the 27B's 48 query rows, warp-specialized -- an ADAPTED op
// (ADR 0051 tier 2; ADR 0031 option (b) and ADR 0093: the vendored kernel's routes stay as the
// oracle; the occupancy step of docs/findings/2026-09-24-hq-verify-decode-cost.md; spec #570,
// GitHub #571).
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1
// (the commit kernel/vendor/manifest.json pins), Apache-2.0 (kernel/vendor/LICENSE):
//   src/ops/kernel/gqa_attention_decode_bf16.cuh
//       gqa_attention_small_t_tc_partial_bf16_kernel's hq route: the masked columns' count and
//       the per-lane offsets of q, the positions, the new rows, the table row and the partials;
//       write_neutral (the neutral partials); the bounds and their early returns; the split
//       arithmetic (the active splits, tile or key units, the split's key range, its tiles and
//       pages); the ring-valid row; the fused append's per-unit body (the encode, the residual
//       row, the ring mark); and the epilogue (each output row un-rotated in shared memory, then
//       the partial stores).
// From the vendored reference, called and not copied: the geometry and partial-index helpers
// (gqa_small_t_tc_row_to_qt, gqa_valid_q_head, gqa_partial_stat_index, gqa_partial_acc_index,
// gqa_small_t_active_splits, gqa_kv_new_index), the codec's (hq_encode_row_warp,
// hq_store_rotated_row_warp, hq_ring_mark_valid, hq_ifwht256_sign, hq_engine_signs_fill, the row
// addresses and the dither seed), div_up, load_vec, store_vec, pdl::sync and pdl::publish.
//
// Changes from upstream (ours): the block's structure. The same partials as the hq routes, bit
// for bit, from one block per SM in which the decode and the MMAs no longer take turns:
//   - Producers (twelve warps) decode the K and V tiles of the block's split into a ring of
//     three 32-key role tiles in shared memory, one warp call = four rows, each warp staging
//     its own code rows by cp.async one call ahead, with hq_decode_row_group_lean (the
//     throughput decoder's values, a cheaper walk). They never wait on a block barrier, only
//     on a ring slot the consumers have released.
//   - Consumers (eight warps) run the MMAs on each tile. QK runs on six of them, one (row
//     group, key half) each: the throughput route's MMAs in its k order per score, only
//     another warp computes half of them. Each row's max over the 32 keys of a tile is the max
//     of its two halves' maxima (exact), and its sum is rebuilt in the reference's own order,
//     ((s0 + s1) + s2) + s3 per lane and then warp_sum<4>, from the halves' per-lane terms
//     exchanged through shared memory. PV runs on all eight, the 48 x 256 output accumulators
//     spread over them: warp w owns output dimensions [32 w, 32 w + 32) of all three 16-row
//     groups. The ldmatrix addresses of the k loops are four swizzled bases per operand and an
//     immediate, and step k + 1's fragments load before step k's MMAs.
//   - Each ring slot has a full and an empty mbarrier; a slot's k-th use waits on parity k & 1
//     (in-order producers and consumers keep every waiter within one phase of its barrier).
//   - The fused append runs on sixteen warps, its scratch aliasing the ring, and the epilogue
//     (un-rotation, partial stores) on every warp; the q staging and rotation on the consumers
//     alone, while the producers decode the first tiles. The decoder is ours too
//     (hq_decode_lean.cuh).
// The partial layout, the masks, the neutral partials, the fused append, the residual window
// and the rotated frame are the vendored kernel's. Launch it as that kernel's hq route, with
// its arguments, on its grid (KV heads, split capacity, lanes), kHqVerifyWsThreads threads and
// no dynamic shared memory (87,408 bytes of static shared memory, one block per SM). Bit-exact
// against both vendored routes: kernel/tests/test_hq_verify_exact.cu.
//
// Blackwell only (spec #570). The block's static shared memory is more than an sm_8x block
// takes (ptxas refuses it there, kernel/CMakeLists.txt) and the ring waits on
// mbarrier.try_wait, so a build for an older card (IGNIS_CUDA_ARCHITECTURES, GitHub #456)
// compiles the kernel there as a trap, as hq_prompt_attention.cu does its 8-bit band: nothing
// launches it below compute capability 12.0.
#pragma once

#include "core/pdl.cuh"
#include "hq_decode_lean.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#define IGNIS_HQ_VERIFY_WS 0
#else
#define IGNIS_HQ_VERIFY_WS 1
#endif

namespace ignis_hq_verify {

using namespace ninfer;
using namespace ninfer::ops;

inline constexpr int kHqVerifyWsProducers = 12;
inline constexpr int kHqVerifyWsThreads   = (8 + kHqVerifyWsProducers) * 32;

#if IGNIS_HQ_VERIFY_WS

// ldmatrix at a shared-memory byte address plus a compile-time offset: the swizzled tile
// addresses of a k loop are four per-lane bases, the rest of k an immediate.
template <int Off>
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], std::uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr), "n"(Off));
}

template <int Off>
__device__ __forceinline__ void ldsm_x4_t(unsigned (&r)[4], std::uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr), "n"(Off));
}

// Byte address of element (row, 16 k + coloff) of a 256-wide gqa_small_t_tc_swz tile at
// `tile` is base[k & 3] + 128 (k >> 2) with base[j] = tile + 512 row + 16 (((2 j + coloff / 8))
// ^ (row & 7)): the XOR flips only the low three bits of the 8-element chunk index.
__device__ __forceinline__ void swz256_bases(std::uint32_t (&base)[4], std::uint32_t tile, int row,
                                             int coloff) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        base[j] = tile + 512u * static_cast<std::uint32_t>(row) +
                  16u * static_cast<std::uint32_t>((2 * j + (coloff >> 3)) ^ (row & 7));
    }
}

template <int K>
__device__ __forceinline__ void ldsm_swz256_x4(unsigned (&r)[4], const std::uint32_t (&base)[4]) {
    ldsm_x4<128 * (K >> 2)>(r, base[K & 3]);
}

__device__ __forceinline__ void mbar_init(std::uint32_t addr, std::uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(std::uint32_t addr) {
    asm volatile("{\n\t.reg .b64 state;\n\tmbarrier.arrive.shared.b64 state, [%0];\n\t}" ::"r"(addr)
                 : "memory");
}

// The waiting warp is suspended by the hardware until the phase completes (or the hint, in
// nanoseconds, runs out), so a waiter takes no issue slots from the warps it waits on.
__device__ __forceinline__ void mbar_wait(std::uint32_t addr, std::uint32_t parity) {
    std::uint32_t done = 0;
    do {
        asm volatile(
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared.b64 p, [%1], %2, %3;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t}"
            : "=r"(done)
            : "r"(addr), "r"(parity), "r"(1000000u)
            : "memory");
    } while (done == 0);
}

// The consumer warps' own barrier (named barrier 1; __syncthreads is barrier 0).
__device__ __forceinline__ void consumer_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }

#endif // IGNIS_HQ_VERIFY_WS

template <typename Geometry, typename CacheInput>
__launch_bounds__(kHqVerifyWsThreads, 1) __global__ void hq_verify_ws_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, GqaTcKVHq kv,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
#if !IGNIS_HQ_VERIFY_WS
    __trap();
#else
    pdl::sync();
    constexpr int TokenTile = 8;
    constexpr int Producers = kHqVerifyWsProducers;
    constexpr int Consumers = 8;
    constexpr int Warps     = Consumers + Producers;
    constexpr int Threads   = Warps * 32;
    constexpr int Groups    = 3;
    constexpr int Br        = Groups * 16;
    constexpr int Bc        = 32;
    constexpr int D         = kGqaHeadDim;
    constexpr int QKKs      = D / 16;
    constexpr int OutNt     = D / 8 / Consumers;
    constexpr int QkWarps   = Groups * 2;
    constexpr int kQRows    = Geometry::GroupSize * TokenTile;
    constexpr int PageIds   = 128;
    constexpr int Ring      = 3;
    constexpr int kCallRows = 4;                           // rows per producer warp call
    constexpr int kCallsPerTile = Bc / kCallRows;          // 8
    constexpr int kStageBytes = kCallRows * (kHqRowBudgetBytes + kHqMetaBytes);  // 288
    constexpr float Log2E   = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(kQRows == Br, "the 27B geometry: 48 query rows in three row groups");
    static_assert(OutNt == 4 && Bc == 32 && D == 256);
    static_assert(Producers >= 1 && Warps <= 32);

    __shared__ __align__(1024) __nv_bfloat16 q_s[Br * D];
    __shared__ __align__(1024) __nv_bfloat16 ring_s[Ring][Bc * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ __align__(16) std::uint8_t stage_s[Producers][2][kStageBytes];
    __shared__ __align__(8) std::uint64_t full_bar[Ring];
    __shared__ __align__(8) std::uint64_t empty_bar[Ring];
    __shared__ std::int32_t physical_pages_s[PageIds];
    __shared__ std::int8_t signs_s[kGqaHeadDim];
    __shared__ float max_s[2][Br];
    __shared__ float alpha_s[Br];
    __shared__ float sum_lo_s[Groups][2][32];
    __shared__ float sum_hi_s[Groups][4][32];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = static_cast<int>(blockIdx.z);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    int valid_tokens      = tokens;
    if (valid_columns != nullptr) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    const std::int64_t column_base =
        column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    partial_acc +=
        static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens * split_count;
    partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) {
        write_neutral();
        return;
    }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    hq_engine_signs_fill(signs_s);
    if (tid == 0) {
#pragma unroll
        for (int b = 0; b < Ring; ++b) {
            mbar_init(static_cast<std::uint32_t>(__cvta_generic_to_shared(&full_bar[b])),
                      kCallsPerTile * 32);
            mbar_init(static_cast<std::uint32_t>(__cvta_generic_to_shared(&empty_bar[b])),
                      Consumers * 32);
        }
    }
    __syncthreads();

    constexpr int kRingWords = static_cast<int>(kGqaHqRecentKeys) / 32;
    const std::uint32_t* ring_valid_row = nullptr;
    if (kv.ring_valid != nullptr) {
        ring_valid_row = kv.ring_valid + static_cast<std::int64_t>(table_row) * kRingWords;
    }

    if constexpr (CacheInput::writes_cache) {
        // The vendored kernel's fused append: each (token, role) unit is one warp's encode, its
        // scratch aliasing the ring (unused until the producers start).
        constexpr int kAppendWarps = 16;
        static_assert(kAppendWarps * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow) * sizeof(float) <=
                          sizeof(ring_s),
                      "per-warp append scratch must fit the ring it aliases");
        float* append_scratch = reinterpret_cast<float*>(&ring_s[0][0]);
        if (warp < kAppendWarps) {
            for (int unit = warp; unit < valid_tokens * 2; unit += kAppendWarps) {
                const int t          = unit >> 1;
                const bool role_v    = (unit & 1) != 0;
                const std::int32_t p = pos[t];
                if (p < split_start || p >= split_end) { continue; }
                float* u = append_scratch + warp * (kHqSmemFloatsPerRow + kHqSmemSymbolsPerRow);
                std::uint32_t* syms = reinterpret_cast<std::uint32_t*>(u + kHqSmemFloatsPerRow);
                const std::int64_t base =
                    static_cast<std::int64_t>(t) * Geometry::KVHeads * kGqaHeadDim;
                const __nv_bfloat16* src =
                    (role_v ? input.v : input.k) + base + gqa_kv_new_index<Geometry>(kv_head, 0, 0);
                hq_encode_row_warp(src, signs_s, 0, u, syms,
                                   hq_row_codes_mut<Geometry>(role_v ? kv.codes_v : kv.codes_k,
                                                              block_table, kv_head, p),
                                   hq_row_meta_mut<Geometry>(role_v ? kv.meta_v : kv.meta_k,
                                                             block_table, kv_head, p),
                                   hq_dither_row_seed(kv_head, p, role_v));
                if (kv.residual_k != nullptr) {
                    hq_store_rotated_row_warp(
                        src, signs_s,
                        hq_residual_row<Geometry>(role_v ? kv.residual_v : kv.residual_k,
                                                  table_row, kv_head, p));
                    hq_ring_mark_valid(const_cast<std::uint32_t*>(ring_valid_row), p);
                }
            }
        }
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;

    // Consumer state lives across the loop; producers never touch it.
    float acc[Groups][OutNt][4];
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int n = 0; n < OutNt; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[g][n][i] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const bool qk_warp = warp < QkWarps;
    const int qk_group = warp >> 1;
    const int qk_half  = warp & 1;
    const int qk_row0  = qk_group * 16 + gid;
    const int qk_row1  = qk_row0 + 8;

    if (warp >= Consumers) {
        // ---- producers ----
        const int pw          = warp - Consumers;
        const int total_calls = key_blocks * 2 * kCallsPerTile;
        auto stage_call = [&](int g, int buf) {
            if (g < total_calls) {
                const int r   = g / kCallsPerTile;
                const int j   = g - r * kCallsPerTile;
                const bool rv = (r & 1) != 0;
                const int k0  = first_tile + (r >> 1) * Bc + j * kCallRows;
                const int page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
                const std::uint8_t* codes =
                    (rv ? kv.codes_v : kv.codes_k) +
                    paged_kv_element_offset<kHqCodePlaneExtent, Geometry::KVHeads>(
                        page, kv_head, k0 & kPagedKVPageMask, 0);
                const std::uint8_t* meta =
                    (rv ? kv.meta_v : kv.meta_k) +
                    paged_kv_element_offset<kHqMetaPlaneExtent, Geometry::KVHeads>(
                        page, kv_head, k0 & kPagedKVPageMask, 0);
                constexpr int kCodeChunks = kCallRows * kHqRowBudgetBytes / 16;  // 16
                constexpr int kMetaChunks = kCallRows * kHqMetaBytes / 16;       // 2
                if (lane < kCodeChunks) {
                    cp_async<16, Cache::cg>(&stage_s[pw][buf][lane * 16], codes + lane * 16);
                } else if (lane < kCodeChunks + kMetaChunks) {
                    cp_async<16, Cache::cg>(&stage_s[pw][buf][(lane - kCodeChunks) * 16 +
                                                              kCallRows * kHqRowBudgetBytes],
                                            meta + (lane - kCodeChunks) * 16);
                }
            }
            cp_commit();
        };
        int buf = 0;
        stage_call(pw, 0);
        for (int g = pw; g < total_calls; g += Producers, buf ^= 1) {
            stage_call(g + Producers, buf ^ 1);
            const int r     = g / kCallsPerTile;
            const int j     = g - r * kCallsPerTile;
            const bool role_v = (r & 1) != 0;
            const int slot  = r % Ring;
            const int use   = r / Ring;
            if (use > 0) {
                mbar_wait(static_cast<std::uint32_t>(__cvta_generic_to_shared(&empty_bar[slot])),
                          static_cast<std::uint32_t>((use - 1) & 1));
            }
            cp_wait<1>();
            __syncwarp();
            const int k0    = first_tile + (r >> 1) * Bc;
            const int row_g = lane >> 3;
            const int lane8 = lane & 7;
            const int key_l = j * kCallRows + row_g;
            const int key   = k0 + key_l;
            __nv_bfloat16* row_dst = &ring_s[slot][key_l * D];
            bool codec_row = false;
            if (key >= split_start && key < split_end) {
                const bool side_row =
                    kv.residual_k != nullptr &&
                    (key < static_cast<int>(kGqaHqSinkKeys) ||
                     (key >= window - static_cast<int>(kGqaHqRecentKeys) &&
                      hq_ring_slot_valid(ring_valid_row, key)));
                if (side_row) {
                    const __nv_bfloat16* side = hq_residual_row<Geometry>(
                        role_v ? kv.residual_v : kv.residual_k, table_row, kv_head, key);
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const int chunk = ((lane8 * 4 + jj) ^ (key_l & 7)) << 3;
                        store_vec(row_dst + chunk, load_vec<int4>(side + lane8 * 32 + jj * 8));
                    }
                } else {
                    codec_row = true;
                }
            } else {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    const int chunk = ((lane8 * 4 + jj) ^ (key_l & 7)) << 3;
                    store_vec(row_dst + chunk, make_int4(0, 0, 0, 0));
                }
            }
            hq_decode_row_group_lean(&stage_s[pw][buf][row_g * kHqRowBudgetBytes],
                                     &stage_s[pw][buf][kCallRows * kHqRowBudgetBytes +
                                                       row_g * kHqMetaBytes],
                                     row_dst, lane8, key_l & 7,
                                     hq_dither_row_seed(kv_head, key, role_v), codec_row);
            mbar_arrive(static_cast<std::uint32_t>(__cvta_generic_to_shared(&full_bar[slot])));
        }
        cp_wait<0>();
    } else {
        // ---- consumers ----
        // q rows, rotated into the codec frame, by the consumers alone while the
        // producers decode the first tiles.
        for (int chunk = tid; chunk < kQRows * (D / 8); chunk += Consumers * 32) {
            const int row = chunk / (D / 8);
            const int d   = (chunk - row * (D / 8)) * 8;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            int4 value = make_int4(0, 0, 0, 0);
            if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                value = load_vec<int4>(&q[gqa_q_index<Geometry>(q_head, d, token)]);
            }
            store_vec(&q_s[row * D + gqa_small_t_tc_swz(row, d)], value);
        }
        consumer_sync();
        for (int row = warp; row < row_count; row += Consumers) {
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                reg[s] = __bfloat162float(q_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)]);
            }
            hq_fwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                q_s[row * D + gqa_small_t_tc_swz(row, s * 32 + lane)] = __float2bfloat16(reg[s]);
            }
        }
        consumer_sync();

        const int a_mat    = lane >> 3;
        const int a_rin    = lane & 7;
        const int a_rowoff = a_rin + ((a_mat & 1) << 3);
        const int a_coloff = (a_mat >> 1) << 3;
        const int b_rin    = lane & 7;
        const int b_koff   = ((lane >> 3) & 1) << 3;
        int qabs0 = -1, qabs1 = -1;
        {
            int q_head = 0, token0 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(qk_row0, tokens, kv_head, q_head, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(qk_row1, tokens, kv_head, q_head, token1);
            if (qk_warp) {
                qabs0 = (qk_row0 < row_count) ? pos[token0] : -1;
                qabs1 = (qk_row1 < row_count) ? pos[token1] : -1;
            }
        }
        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0    = first_tile + kb * Bc;
            const int rk    = 2 * kb;
            const int slotk = rk % Ring;
            mbar_wait(static_cast<std::uint32_t>(__cvta_generic_to_shared(&full_bar[slotk])),
                      static_cast<std::uint32_t>((rk / Ring) & 1));
            const __nv_bfloat16* k_s = ring_s[slotk];

            float score[2][4];
            if (qk_warp) {
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
                }
                {
                    std::uint32_t abase[4], bbase[4];
                    swz256_bases(abase, smem_addr(q_s), qk_group * 16 + a_rowoff, a_coloff);
                    swz256_bases(bbase, smem_addr(k_s), (qk_half * 2 + (lane >> 4)) * 8 + b_rin,
                                 b_koff);
                    unsigned af[2][4], bf[2][4];
                    ldsm_swz256_x4<0>(af[0], abase);
                    ldsm_swz256_x4<0>(bf[0], bbase);
                    auto qk_step = [&](auto kc) {
                        constexpr int k = decltype(kc)::value;
                        if constexpr (k + 1 < QKKs) {
                            ldsm_swz256_x4<k + 1>(af[(k + 1) & 1], abase);
                            ldsm_swz256_x4<k + 1>(bf[(k + 1) & 1], bbase);
                        }
                        const unsigned(&a)[4] = af[k & 1];
                        const unsigned(&b)[4] = bf[k & 1];
                        mma_bf16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1],
                                 a[2], a[3], b[0], b[1]);
                        mma_bf16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1],
                                 a[2], a[3], b[2], b[3]);
                    };
                    [&]<int... K>(std::integer_sequence<int, K...>) {
                        (qk_step(std::integral_constant<int, K>{}), ...);
                    }(std::make_integer_sequence<int, QKKs>{});
                }
                float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int col0 = (qk_half * 2 + j) * 8 + 2 * lid;
                    const int key0 = k0 + col0;
                    const int key1 = col0 + 1 + k0;
                    score[j][0] = (qk_row0 < row_count && key0 >= split_start && key0 < split_end &&
                                   key0 <= qabs0)
                                      ? score[j][0] * scale
                                      : -CUDART_INF_F;
                    score[j][1] = (qk_row0 < row_count && key1 >= split_start && key1 < split_end &&
                                   key1 <= qabs0)
                                      ? score[j][1] * scale
                                      : -CUDART_INF_F;
                    score[j][2] = (qk_row1 < row_count && key0 >= split_start && key0 < split_end &&
                                   key0 <= qabs1)
                                      ? score[j][2] * scale
                                      : -CUDART_INF_F;
                    score[j][3] = (qk_row1 < row_count && key1 >= split_start && key1 < split_end &&
                                   key1 <= qabs1)
                                      ? score[j][3] * scale
                                      : -CUDART_INF_F;
                    bm0 = fmaxf(bm0, fmaxf(score[j][0], score[j][1]));
                    bm1 = fmaxf(bm1, fmaxf(score[j][2], score[j][3]));
                }
                bm0 = warp_max<4>(bm0, FullMask);
                bm1 = warp_max<4>(bm1, FullMask);
                if (lid == 0) {
                    max_s[qk_half][qk_row0] = bm0;
                    max_s[qk_half][qk_row1] = bm1;
                }
            }
            consumer_sync();  // the K slot read; both halves' maxima published
            mbar_arrive(static_cast<std::uint32_t>(__cvta_generic_to_shared(&empty_bar[slotk])));

            if (qk_warp) {
                const float bm0 = fmaxf(max_s[0][qk_row0], max_s[1][qk_row0]);
                const float bm1 = fmaxf(max_s[0][qk_row1], max_s[1][qk_row1]);
                const float nm0 = fmaxf(m0, bm0);
                const float nm1 = fmaxf(m1, bm1);
                const float alpha0 =
                    (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
                const float alpha1 =
                    (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
                __nv_bfloat16* p_g = p_s + qk_group * 16 * Bc;
                float part0[2], part1[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int col0  = (qk_half * 2 + j) * 8 + 2 * lid;
                    const int col1  = col0 + 1;
                    const float p00 = (nm0 > -CUDART_INF_F && score[j][0] > -CUDART_INF_F)
                                          ? exp2_approx((score[j][0] - nm0) * Log2E)
                                          : 0.0f;
                    const float p01 = (nm0 > -CUDART_INF_F && score[j][1] > -CUDART_INF_F)
                                          ? exp2_approx((score[j][1] - nm0) * Log2E)
                                          : 0.0f;
                    const float p10 = (nm1 > -CUDART_INF_F && score[j][2] > -CUDART_INF_F)
                                          ? exp2_approx((score[j][2] - nm1) * Log2E)
                                          : 0.0f;
                    const float p11 = (nm1 > -CUDART_INF_F && score[j][3] > -CUDART_INF_F)
                                          ? exp2_approx((score[j][3] - nm1) * Log2E)
                                          : 0.0f;
                    part0[j] = p00 + p01;
                    part1[j] = p10 + p11;
                    p_g[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
                    p_g[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
                    p_g[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
                    p_g[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
                }
                if (qk_half == 0) {
                    float b0 = 0.0f, b1 = 0.0f;
                    b0 += part0[0];
                    b1 += part1[0];
                    b0 += part0[1];
                    b1 += part1[1];
                    sum_lo_s[qk_group][0][lane] = b0;
                    sum_lo_s[qk_group][1][lane] = b1;
                    if (lid == 0) {
                        alpha_s[qk_row0] = alpha0;
                        alpha_s[qk_row1] = alpha1;
                    }
                } else {
                    sum_hi_s[qk_group][0][lane] = part0[0];
                    sum_hi_s[qk_group][1][lane] = part0[1];
                    sum_hi_s[qk_group][2][lane] = part1[0];
                    sum_hi_s[qk_group][3][lane] = part1[1];
                }
                m0 = nm0;
                m1 = nm1;
            }
            consumer_sync();  // P, the rescales and the sums published

            if (qk_warp) {
                float b0 = sum_lo_s[qk_group][0][lane];
                float b1 = sum_lo_s[qk_group][1][lane];
                b0 += sum_hi_s[qk_group][0][lane];
                b1 += sum_hi_s[qk_group][2][lane];
                b0 += sum_hi_s[qk_group][1][lane];
                b1 += sum_hi_s[qk_group][3][lane];
                b0 = warp_sum<4>(b0, FullMask);
                b1 = warp_sum<4>(b1, FullMask);
                const float alpha0 = alpha_s[qk_row0];
                const float alpha1 = alpha_s[qk_row1];
                l0 = l0 * alpha0 + b0;
                l1 = l1 * alpha1 + b1;
            }

            const int rv    = rk + 1;
            const int slotv = rv % Ring;
            mbar_wait(static_cast<std::uint32_t>(__cvta_generic_to_shared(&full_bar[slotv])),
                      static_cast<std::uint32_t>((rv / Ring) & 1));
            const __nv_bfloat16* v_s = ring_s[slotv];
            {
                const int vrow         = (lane >> 4) * 16 + b_koff + b_rin;
                const std::uint32_t vb = smem_addr(v_s) + 512u * static_cast<std::uint32_t>(vrow);
                unsigned vf[OutNt][4];
                ldsm_x4_t<0>(vf[0], vb + 16u * static_cast<std::uint32_t>((warp * OutNt + 0) ^ b_rin));
                ldsm_x4_t<0>(vf[1], vb + 16u * static_cast<std::uint32_t>((warp * OutNt + 1) ^ b_rin));
                ldsm_x4_t<0>(vf[2], vb + 16u * static_cast<std::uint32_t>((warp * OutNt + 2) ^ b_rin));
                ldsm_x4_t<0>(vf[3], vb + 16u * static_cast<std::uint32_t>((warp * OutNt + 3) ^ b_rin));
                mbar_arrive(static_cast<std::uint32_t>(__cvta_generic_to_shared(&empty_bar[slotv])));
                const std::uint32_t pb = smem_addr(p_s) + 64u * static_cast<std::uint32_t>(a_rowoff);
                const std::uint32_t p0 =
                    pb + 16u * static_cast<std::uint32_t>((0 + (a_coloff >> 3)) ^ (a_rowoff & 3));
                const std::uint32_t p1 =
                    pb + 16u * static_cast<std::uint32_t>((2 + (a_coloff >> 3)) ^ (a_rowoff & 3));
                unsigned pfa[Groups][2][4];
                ldsm_x4<0>(pfa[0][0], p0);
                ldsm_x4<0>(pfa[0][1], p1);
                ldsm_x4<1024>(pfa[1][0], p0);
                ldsm_x4<1024>(pfa[1][1], p1);
                ldsm_x4<2048>(pfa[2][0], p0);
                ldsm_x4<2048>(pfa[2][1], p1);
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    const float alpha0 = alpha_s[g * 16 + gid];
                    const float alpha1 = alpha_s[g * 16 + gid + 8];
                    const unsigned(&pf)[2][4] = pfa[g];
#pragma unroll
                    for (int n = 0; n < OutNt; ++n) {
                        acc[g][n][0] *= alpha0;
                        acc[g][n][1] *= alpha0;
                        acc[g][n][2] *= alpha1;
                        acc[g][n][3] *= alpha1;
                        mma_bf16(acc[g][n][0], acc[g][n][1], acc[g][n][2], acc[g][n][3], pf[0][0],
                                 pf[0][1], pf[0][2], pf[0][3], vf[n][0], vf[n][1]);
                        mma_bf16(acc[g][n][0], acc[g][n][1], acc[g][n][2], acc[g][n][3], pf[1][0],
                                 pf[1][1], pf[1][2], pf[1][3], vf[n][2], vf[n][3]);
                    }
                }
            }
        }
    }
    __syncthreads();  // every slot released; q no longer read

    if (qk_warp && qk_half == 0 && lid == 0) {
        if (qk_row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(qk_row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (qk_row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(qk_row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
        }
    }
    if (warp < Consumers) {
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
#pragma unroll
            for (int n = 0; n < OutNt; ++n) {
                const int d0   = (warp * OutNt + n) * 8 + 2 * lid;
                const int d1   = d0 + 1;
                const int row0 = g * 16 + gid;
                const int row1 = row0 + 8;
                if (row0 < row_count) {
                    q_s[row0 * D + d0] = __float2bfloat16(acc[g][n][0]);
                    q_s[row0 * D + d1] = __float2bfloat16(acc[g][n][1]);
                }
                if (row1 < row_count) {
                    q_s[row1 * D + d0] = __float2bfloat16(acc[g][n][2]);
                    q_s[row1 * D + d1] = __float2bfloat16(acc[g][n][3]);
                }
            }
        }
    }
    __syncthreads();

    for (int row = warp; row < row_count; row += Warps) {
        float reg[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) { reg[s] = __bfloat162float(q_s[row * D + s * 32 + lane]); }
        hq_ifwht256_sign(reg, signs_s, 0, lane);
#pragma unroll
        for (int s = 0; s < 8; ++s) { q_s[row * D + s * 32 + lane] = __float2bfloat16(reg[s]); }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst = gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&q_s[row * D + d]));
        }
    }
    pdl::publish();
#endif // IGNIS_HQ_VERIFY_WS
}

} // namespace ignis_hq_verify
