// The 27B prefill's MLP gate and up product with SwiGLU in its epilogue, rounding as the split
// route does -- an ADAPTED op (ADR 0051; spec #646, GitHub #651; kernel/src/prefill_swiglu.h).
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (the
// commit kernel/vendor/manifest.json pins, ADR 0010), Apache-2.0 (kernel/vendor/LICENSE):
//   src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh
//       nvfp4_linear_swiglu_w4a4_tma_kernel: the TMA and mbarrier pipeline (one producer thread
//       loading the activation codes and scales and, for each CTA's 64 gate rows [p, p + 64), the
//       64 up rows [17408 + p, ...) with their weight scales), the consumer warps' ldmatrix and
//       block-scaled FP4 MMA loop, and the epilogue's staging of each token's 64 outputs in
//       shared memory before 16-byte stores.
// Called and not copied: its shared storage (Nvfp4LinearSwiGluTmaSharedStorage), the schedule
// (Nvfp4W4a4TmaSchedule), the TMA, mbarrier and raster helpers (ops/linear/nvfp4/
// nvfp4_w4a4_tma.cuh), ldmatrix, mma_nvfp4_e4m3, load_vec, store_vec and silu.
//
// Changed here (ours):
//   - the epilogue rounds gate and up to BF16 before the product, as the split route's product
//     stores BF16(acc * alpha) and its SiLU-mul reads those BF16 values:
//     out = BF16(silu(BF16(g * alpha)) * BF16(u * alpha)). Upstream multiplies the fp32 values,
//     one rounding fewer, so its bytes differ from the split route's. The fp32 accumulators are
//     the split product's: each output is the same sequence of MMAs over the same K order, only
//     its N slot in the tile differs.
//   - two pipeline stages, not three: what the served measurement ran (the two were within the
//     bench's noise).
#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::prefill_swiglu {

// The 256-token tile, the 128 weight rows (64 gate, 64 up) and the two stages.
using GateUpSchedule = ninfer::ops::detail::Nvfp4W4a4TmaSchedule<256, 2, 1>;

// NINFER_NVFP4_TMA_DESCRIPTOR_PARAM names the descriptor block unqualified.
using ninfer::ops::detail::Nvfp4W4a4TmaDescriptors;

template <class Geometry, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void swiglu_tma_kernel(
    NINFER_NVFP4_TMA_DESCRIPTOR_PARAM descriptors, float alpha, __nv_bfloat16* __restrict__ output) {
    using namespace ninfer::ops;
    using namespace ninfer::ops::detail;
    static_assert(Geometry::kOutputRows == 34816);
    static_assert(Geometry::kInputRows == 5120);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert(Schedule::kBlockN == 128);
    static_assert(Schedule::kWarpsN == 2);
    static_assert(Schedule::kMmaN == 8);

    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    constexpr int kPairN        = Schedule::kBlockN / 2;
    static_assert((kIntermediate % kPairN) == 0);
    static_assert((kIntermediate % 128) == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<Nvfp4LinearSwiGluTmaSharedStorage<Schedule>*>(shared_bytes);
    static_assert(nvfp4_tma_scale_slots_are_safe<Schedule>,
                  "the activation-scale buffer needs two slots and at most three stages");
    int block_x = 0;
    int block_y = 0;
    nvfp4_tma_raster_blocks(block_x, block_y);
    const int token_begin = block_y * Schedule::kBlockM;
    const int pair_begin  = block_x * kPairN;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            nvfp4_mbarrier_init(&shared.full[stage], 1);
            nvfp4_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
    }
    __syncthreads();

    constexpr int kKTiles = Geometry::kInputRows / Schedule::kBlockK;

    if (threadIdx.x < Schedule::kProducerThreads) {
        if constexpr (Schedule::kProducerThreads == 128) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" : : : "memory");
        }
        if (threadIdx.x == 0) {
#ifdef _WIN32
            const Nvfp4W4a4TmaDescriptors* descriptor_block = descriptors;
#else
            const Nvfp4W4a4TmaDescriptors* descriptor_block = &descriptors;
#endif
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
                const int stage                 = k_tile % Schedule::kStages;
                const std::uint32_t empty_phase = 1U ^ ((k_tile / Schedule::kStages) & 1U);
                nvfp4_mbarrier_wait(&shared.empty[stage], empty_phase);
                constexpr std::uint32_t kScaleBytes =
                    Schedule::kBlockM * Schedule::kScaleWordsPerRow * 4;
                constexpr std::uint32_t kTransactionBytes =
                    Schedule::kBlockM * Schedule::kCodeRowBytes +
                    Schedule::kBlockN * Schedule::kCodeRowBytes + kScaleBytes +
                    2 * Schedule::kBlockN * Schedule::kK64PerStage * 4;
                // TMA's innermost box cannot be narrower than 16 bytes and 16 bytes of
                // activation scales cover two K tiles, so the box is fetched on the even tile
                // only and the odd tile expects that many bytes fewer.
                const bool load_scales = (k_tile & 1) == 0;
                nvfp4_mbarrier_arrive_expect_tx(&shared.full[stage],
                                                load_scales ? kTransactionBytes
                                                            : kTransactionBytes - kScaleBytes);

                auto& tensors = shared.scratch.tensors;
                nvfp4_tma_load_2d(tensors.a_codes[stage], &descriptor_block->a_codes,
                                  k_tile * Schedule::kCodeRowBytes, token_begin,
                                  &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_codes[stage], &descriptor_block->b_codes,
                                  k_tile * Schedule::kCodeRowBytes, pair_begin,
                                  &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_codes[stage] + kPairN * Schedule::kCodeRowBytes,
                                  &descriptor_block->b_codes, k_tile * Schedule::kCodeRowBytes,
                                  pair_begin + kIntermediate, &shared.full[stage]);
                if (load_scales) {
                    nvfp4_tma_load_2d(tensors.a_scale4[(k_tile / 2) & 1],
                                      &descriptor_block->a_scales, (k_tile / 2) * 16, token_begin,
                                      &shared.full[stage]);
                }

                const int gate_scale_row = ((pair_begin / 128) * Geometry::kScaleTilesPerRow +
                                            k_tile * Schedule::kK64PerStage) *
                                           32;
                const int up_scale_row =
                    (((pair_begin + kIntermediate) / 128) * Geometry::kScaleTilesPerRow +
                     k_tile * Schedule::kK64PerStage) *
                    32;
                nvfp4_tma_load_2d(tensors.b_scales[stage][0], &descriptor_block->b_scales, 0,
                                  gate_scale_row, &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_scales[stage][1], &descriptor_block->b_scales, 0,
                                  up_scale_row, &shared.full[stage]);
            }
        }
        return;
    }

    if constexpr (Schedule::kProducerThreads == 128) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" : : : "memory");
    }
    auto& tensors             = shared.scratch.tensors;
    const int consumer_thread = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int lane            = consumer_thread & 31;
    const int warp            = consumer_thread >> 5;
    const int warp_m          = warp / Schedule::kWarpsN;
    const int warp_n          = warp - warp_m * Schedule::kWarpsN;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    const int pair_mod128   = pair_begin & 127;

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        nvfp4_mbarrier_wait(&shared.full[stage], full_phase);

#pragma unroll
        for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
            unsigned a_scales[Schedule::kMmaM];
            unsigned b_scales[Schedule::kMmaN];

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int row          = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                const int logical_byte = local_k64 * 32 + a_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                a_scales[mma_m] =
                    tensors.a_scale4[(k_tile / 2) & 1][scale_row * Schedule::kScaleWordsPerRow +
                                                       (k_tile & 1) * Schedule::kK64PerStage +
                                                       local_k64];
            }

#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                constexpr int kFragmentsPerBranch = Schedule::kMmaN / 2;
                const int branch                  = mma_n / kFragmentsPerBranch;
                const int pair_fragment           = mma_n - branch * kFragmentsPerBranch;
                const int pair_row =
                    warp_n * (kPairN / Schedule::kWarpsN) + pair_fragment * 8 + b_row_offset;
                const int storage_row  = branch * kPairN + pair_row;
                const int logical_byte = local_k64 * 32 + b_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((storage_row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.b_codes[stage] + storage_row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));

                const int scale_pair_row = pair_mod128 + warp_n * (kPairN / Schedule::kWarpsN) +
                                           pair_fragment * 8 + sfb_row;
                const int row_mod32    = scale_pair_row & 31;
                const int row_quartile = scale_pair_row >> 5;
                b_scales[mma_n] =
                    load_vec<unsigned>(tensors.b_scales[stage][branch] +
                                       (local_k64 * 32 + row_mod32) * 16 + row_quartile * 4);
            }

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                    mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                   accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                   a_fragments[mma_m][0], a_fragments[mma_m][1],
                                   a_fragments[mma_m][2], a_fragments[mma_m][3],
                                   b_fragments[mma_n][0], b_fragments[mma_n][1], a_scales[mma_m],
                                   b_scales[mma_n]);
                }
            }
        }
        if (lane == 0) { nvfp4_mbarrier_arrive(&shared.empty[stage]); }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr int kOutputStride     = kPairN + 8;
    constexpr int kGateMmaFragments = Schedule::kMmaN / 2;
    auto* shared_output             = shared.scratch.output;
    const int accumulator_row       = lane >> 2;
    const int accumulator_col       = 2 * (lane & 3);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < kGateMmaFragments; ++mma_n) {
            const int pair_row =
                warp_n * (kPairN / Schedule::kWarpsN) + mma_n * 8 + accumulator_col;
            auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token0 * kOutputStride + pair_row);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token1 * kOutputStride + pair_row);
            const auto& gate = accumulators[mma_m][mma_n];
            const auto& up   = accumulators[mma_m][mma_n + kGateMmaFragments];
            // ignis: the split route's product stores BF16(acc * alpha) and its SiLU-mul reads
            // those BF16 values, so gate and up round here before the product.
            float g[4];
            float u[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                g[i] = __bfloat162float(__float2bfloat16_rn(gate[i] * alpha));
                u[i] = __bfloat162float(__float2bfloat16_rn(up[i] * alpha));
            }
            *destination0 = __floats2bfloat162_rn(silu(g[0]) * u[0], silu(g[1]) * u[1]);
            *destination1 = __floats2bfloat162_rn(silu(g[2]) * u[2], silu(g[3]) * u[3]);
        }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");
    constexpr int kVectorsPerRow = kPairN / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int token_local = task / kVectorsPerRow;
        const int row_vector  = task - token_local * kVectorsPerRow;
        const int token       = token_begin + token_local;
        const uint4 values =
            load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
        store_vec(output + static_cast<std::int64_t>(token) * kIntermediate + pair_begin +
                      row_vector * 8,
                  values);
    }
}

}  // namespace ignis::prefill_swiglu
