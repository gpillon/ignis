// ignis kernel leaf: A16 NVFP4 linear on BF16 tensor-core MMA, ours (see
// kernel/include/ignis_nvfp4_a16_mma.h for why it exists and what numerics
// it keeps).
//
//   out[N, T] = W[N, K] * x[K, T]
//
// One CTA computes a kBlockRows x kBlockCols tile of `out` (weight rows x
// tokens), walking K in kBlockK steps through a kStages-deep cp.async ring.
// Each ring slot holds the raw operands exactly as they sit in memory:
//
//   codes    kBlockRows rows x kBlockK/2 bytes of E2M1 pairs (the code
//            plane is row-major, so a tile row is one contiguous run)
//   scales   the whole 512-byte blockscale-k16-m128x4 tile for this K step
//            (128 weight rows x 4 groups, swizzled; a 64-row CTA reads half)
//   tokens   kBlockCols x kBlockK BF16 activations, XOR-swizzled
//
// Per K step the CTA decodes its weight tile once into a BF16 shared tile
// (one thread per row x 16-element group: eight code bytes, one scale), then
// runs m16n8k16 BF16 MMAs from it with the ldmatrix fragment walk of a plain
// BF16 GEMM. The decoded tile is shared by every token column of the CTA, so
// the decode costs one pass over the weight per 128 tokens.

#include "ignis_nvfp4_a16_mma.h"

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

using ninfer::ops::Cache;
using ninfer::ops::cp_async;
using ninfer::ops::cp_async_zfill;
using ninfer::ops::cp_commit;
using ninfer::ops::cp_wait;
using ninfer::ops::ldmatrix_x2;
using ninfer::ops::ldmatrix_x4;
using ninfer::ops::mma_bf16;
using ninfer::ops::smem_addr;
using ninfer::ops::detail::decode_nvfp4_e2m1x2;
using ninfer::ops::detail::decode_nvfp4_e4m3;

constexpr int kBlockRows = 64;
constexpr int kBlockCols = 128;
constexpr int kBlockK = 64;
constexpr int kWarpRows = 32;
constexpr int kWarpCols = 32;
constexpr int kStages = 2;

// The narrowest call this route takes. A lone CTA walks the whole K axis at
// ~1.1 us per K step, so the route has a latency floor (~90 us at K=5120)
// that the GEMV slices undercut until T is wide enough. Measured 2026-09-29
// (ignis_nvfp4_a16_mma_bench): at T=33 the route ran 0.76-0.92x the GEMVs on
// every shape; at T=64 it wins 1.38-1.50x on problems of 4,096+ rows, but
// loses on the narrow ones ([1280, 5120] 0.70x, [256, 5120] 0.59x), which
// win from T=128 (1.37x, 1.05x).
constexpr std::int32_t kMinTokens = 64;
constexpr std::int32_t kMinTokensNarrow = 128;
constexpr std::int32_t kNarrowRowsBelow = 4096;

constexpr int kWarpsM = kBlockRows / kWarpRows;
constexpr int kWarpsN = kBlockCols / kWarpCols;
constexpr int kThreads = kWarpsM * kWarpsN * 32;
constexpr int kMmaRows = kWarpRows / 16;
constexpr int kMmaCols = kWarpCols / 8;
constexpr int kMmaK = kBlockK / 16;

// The blockscale layout's own tile: 128 weight rows x 64 K values (4 groups).
constexpr int kScaleTileRows = 128;
constexpr int kScaleTileBytes = 512;

constexpr int kCodeBytesPerRow = kBlockK / 2;
constexpr int kCodeStageBytes = kBlockRows * kCodeBytesPerRow;
constexpr int kTokenStageBytes = kBlockCols * kBlockK * 2;
constexpr int kStageBytes = kCodeStageBytes + kScaleTileBytes + kTokenStageBytes;
constexpr int kWeightTileBytes = kBlockRows * kBlockK * 2;
constexpr int kSharedBytes = kStages * kStageBytes + kWeightTileBytes;

static_assert(kBlockK == 64, "one blockscale tile per K step");
static_assert(kScaleTileRows % kBlockRows == 0, "a CTA's rows sit inside one scale tile");
static_assert(kBlockRows * (kBlockK / 16) == kThreads, "one decode job per thread");
static_assert(kSharedBytes <= 48 * 1024, "fits the default dynamic shared memory");

// The XOR swizzle the BF16 GEMM uses: 8-element chunks of a 64-wide row are
// permuted by the row's low three bits, so ldmatrix's eight row reads land in
// eight different bank groups.
__device__ __forceinline__ int swizzled_col(int row, int col) {
  return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t *>(&packed);
}

template <int M, int K, bool FullTokens>
__global__ __launch_bounds__(kThreads) void nvfp4_a16_mma_kernel(
    const __nv_bfloat16 *__restrict__ x, const std::uint8_t *__restrict__ codes,
    const std::uint8_t *__restrict__ scales, float inverse_weight_divisor,
    __nv_bfloat16 *__restrict__ out, std::int32_t tokens) {
  static_assert(M % kBlockRows == 0);
  static_assert(K % kBlockK == 0);
  constexpr int kTiles = K / kBlockK;
  static_assert(kTiles >= kStages);
  constexpr int kScaleTilesPerRow = K / 64;

  extern __shared__ __align__(16) unsigned char shared_raw[];
  unsigned char *ring = shared_raw;
  auto *weight_tile = reinterpret_cast<__nv_bfloat16 *>(shared_raw + kStages * kStageBytes);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / kWarpsN;
  const int wn = warp - wm * kWarpsN;
  const int gid = lane >> 2;
  const int lid = lane & 3;

  const int tiles_n = tokens / kBlockCols + static_cast<int>(tokens % kBlockCols != 0);
  const int tile_m = static_cast<int>(blockIdx.x) / tiles_n;
  const int tile_n = static_cast<int>(blockIdx.x) - tile_m * tiles_n;
  const int m0 = tile_m * kBlockRows;
  const int n0 = tile_n * kBlockCols;
  const int scale_tile_m = m0 / kScaleTileRows;
  const int row_in_scale_tile0 = m0 - scale_tile_m * kScaleTileRows;

  auto stage_codes = [&](int stage) { return ring + stage * kStageBytes; };
  auto stage_scales = [&](int stage) { return ring + stage * kStageBytes + kCodeStageBytes; };
  auto stage_tokens = [&](int stage) {
    return reinterpret_cast<__nv_bfloat16 *>(ring + stage * kStageBytes + kCodeStageBytes +
                                             kScaleTileBytes);
  };

  auto load_stage = [&](int stage, int k_tile) {
    const int k0 = k_tile * kBlockK;
    unsigned char *code_dst = stage_codes(stage);
    for (int item = tid; item < kBlockRows * 2; item += kThreads) {
      const int row = item >> 1;
      const int half = item & 1;
      cp_async<16, Cache::cg>(code_dst + row * kCodeBytesPerRow + half * 16,
                              codes + static_cast<std::int64_t>(m0 + row) * (K / 2) + k0 / 2 +
                                  half * 16);
    }
    unsigned char *scale_dst = stage_scales(stage);
    const std::uint8_t *scale_src =
        scales + (static_cast<std::int64_t>(scale_tile_m) * kScaleTilesPerRow + k_tile) *
                     kScaleTileBytes;
    for (int item = tid; item < kScaleTileBytes / 16; item += kThreads) {
      cp_async<16, Cache::cg>(scale_dst + item * 16, scale_src + item * 16);
    }
    __nv_bfloat16 *token_dst = stage_tokens(stage);
    for (int item = tid; item < kBlockCols * (kBlockK / 8); item += kThreads) {
      const int col = item / (kBlockK / 8);
      const int kk = (item - col * (kBlockK / 8)) * 8;
      __nv_bfloat16 *dst = &token_dst[col * kBlockK + swizzled_col(col, kk)];
      const int token = n0 + col;
      if constexpr (FullTokens) {
        cp_async<16, Cache::cg>(dst, &x[static_cast<std::int64_t>(token) * K + k0 + kk]);
      } else {
        const bool valid = token < tokens;
        cp_async_zfill<16, Cache::cg>(
            dst, &x[static_cast<std::int64_t>(valid ? token : 0) * K + k0 + kk], valid ? 16 : 0);
      }
    }
  };

  // One thread decodes one (row, 16-element group) of the K step: eight code
  // bytes and the group's scale, into two 16-byte chunks of the BF16 tile.
  const int decode_row = tid >> 2;
  const int decode_group = tid & 3;
  const int decode_scale_row = row_in_scale_tile0 + decode_row;
  const int decode_scale_index =
      (decode_scale_row & 31) * 16 + (decode_scale_row >> 5) * 4 + decode_group;

  auto decode_stage = [&](int stage) {
    const uint2 packed = *reinterpret_cast<const uint2 *>(
        stage_codes(stage) + decode_row * kCodeBytesPerRow + decode_group * 8);
    const float scale = decode_nvfp4_e4m3(stage_scales(stage)[decode_scale_index]);
    const std::uint32_t words[2] = {packed.x, packed.y};
    std::uint32_t decoded[8];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const float2 pair = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(words[w] >> (8 * b)));
        decoded[w * 4 + b] = pack_bf16x2(pair.x * scale, pair.y * scale);
      }
    }
    const int col = decode_group * 16;
    *reinterpret_cast<uint4 *>(
        &weight_tile[decode_row * kBlockK + swizzled_col(decode_row, col)]) =
        make_uint4(decoded[0], decoded[1], decoded[2], decoded[3]);
    *reinterpret_cast<uint4 *>(
        &weight_tile[decode_row * kBlockK + swizzled_col(decode_row, col + 8)]) =
        make_uint4(decoded[4], decoded[5], decoded[6], decoded[7]);
  };

  float accum[kMmaRows][kMmaCols][4] = {};

  const int a_matrix = lane >> 3;
  const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
  const int a_col_offset = (a_matrix >> 1) << 3;
  const int b_inner_row = lane & 7;
  const int b_k_offset = ((lane >> 3) & 1) << 3;

#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    load_stage(stage, stage);
    cp_commit();
  }

#pragma unroll 1
  for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
    const int stage = k_tile % kStages;
    if (k_tile + kStages <= kTiles) {
      cp_wait<kStages - 1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    decode_stage(stage);
    __syncthreads();

    const __nv_bfloat16 *token_tile = stage_tokens(stage);
#pragma unroll
    for (int k_step = 0; k_step < kMmaK; ++k_step) {
      unsigned a_frag[kMmaRows][4];
      unsigned b_frag[kMmaCols][2];
#pragma unroll
      for (int mi = 0; mi < kMmaRows; ++mi) {
        const int row = wm * kWarpRows + mi * 16 + a_row_offset;
        const int col = k_step * 16 + a_col_offset;
        ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    smem_addr(&weight_tile[row * kBlockK + swizzled_col(row, col)]));
      }
#pragma unroll
      for (int ni = 0; ni < kMmaCols; ++ni) {
        const int row = wn * kWarpCols + ni * 8 + b_inner_row;
        const int col = k_step * 16 + b_k_offset;
        ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                    smem_addr(&token_tile[row * kBlockK + swizzled_col(row, col)]));
      }
#pragma unroll
      for (int mi = 0; mi < kMmaRows; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kMmaCols; ++ni) {
          mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                   a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], b_frag[ni][0],
                   b_frag[ni][1]);
        }
      }
    }

    __syncthreads();
    const int next = k_tile + kStages;
    if (next < kTiles) {
      load_stage(stage, next);
      cp_commit();
    }
  }

  auto store = [&](int row, int token, float value) {
    out[static_cast<std::int64_t>(token) * M + row] =
        __float2bfloat16_rn(value * inverse_weight_divisor);
  };
#pragma unroll
  for (int mi = 0; mi < kMmaRows; ++mi) {
    const int row0 = m0 + wm * kWarpRows + mi * 16 + gid;
    const int row1 = row0 + 8;
#pragma unroll
    for (int ni = 0; ni < kMmaCols; ++ni) {
      const int token0 = n0 + wn * kWarpCols + ni * 8 + 2 * lid;
      const int token1 = token0 + 1;
      const float *value = accum[mi][ni];
      if (FullTokens || token0 < tokens) {
        store(row0, token0, value[0]);
        store(row1, token0, value[2]);
      }
      if (FullTokens || token1 < tokens) {
        store(row0, token1, value[1]);
        store(row1, token1, value[3]);
      }
    }
  }
}

template <int M, int K, bool FullTokens>
void launch_variant(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out,
                    cudaStream_t stream) {
  const std::int32_t tokens = x.ne[1];
  const int tiles_n = (tokens + kBlockCols - 1) / kBlockCols;
  const int blocks = (M / kBlockRows) * tiles_n;
  nvfp4_a16_mma_kernel<M, K, FullTokens><<<blocks, kThreads, kSharedBytes, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data), static_cast<const std::uint8_t *>(w.qdata),
      static_cast<const std::uint8_t *>(w.scales), 1.0F / w.weight_scale_divisor,
      static_cast<__nv_bfloat16 *>(out.data), tokens);
  CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_geometry(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out,
                     cudaStream_t stream) {
  constexpr int M = Geometry::kOutputRows;
  constexpr int K = Geometry::kInputRows;
  if (x.ne[1] % kBlockCols == 0) {
    launch_variant<M, K, true>(x, w, out, stream);
  } else {
    launch_variant<M, K, false>(x, w, out, stream);
  }
}

} // namespace

bool ignis_nvfp4_a16_mma_applies(std::int32_t output_rows, std::int32_t input_rows,
                                 std::int32_t tokens) {
  return tokens >= (output_rows >= kNarrowRowsBelow ? kMinTokens : kMinTokensNarrow) &&
         ninfer::ops::detail::is_nvfp4_linear_problem(output_rows, input_rows);
}

void ignis_nvfp4_a16_mma(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out,
                         cudaStream_t stream) {
  using namespace ninfer::ops::detail;
  if (!ignis_nvfp4_a16_mma_applies(w.n, w.k, x.ne[1])) {
    throw std::invalid_argument("ignis_nvfp4_a16_mma: unsupported shape");
  }
  switch (resolve_nvfp4_problem(w.n, w.k)) {
  case Nvfp4Problem::AttnInput:
    launch_geometry<Nvfp4AttnInputGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::GdnInput:
    launch_geometry<Nvfp4GdnInputGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::MlpGateUp:
    launch_geometry<Nvfp4MlpGateUpGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::Residual6144:
    launch_geometry<Nvfp4Residual6144Geometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::Residual17408:
    launch_geometry<Nvfp4Residual17408Geometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::DFlash2Feature:
    launch_geometry<Nvfp4DFlash2FeatureGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::DFlash2Qkv:
    launch_geometry<Nvfp4DFlash2QkvGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::DFlash2AttnOut:
    launch_geometry<Nvfp4DFlash2AttnOutGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::DFlash2ConvProj:
    launch_geometry<Nvfp4DFlash2ConvProjGeometry>(x, w, out, stream);
    return;
  case Nvfp4Problem::DFlash2Selector:
    launch_geometry<Nvfp4DFlash2SelectorGeometry>(x, w, out, stream);
    return;
  }
  throw std::logic_error("ignis_nvfp4_a16_mma: unreachable NVFP4 problem");
}
