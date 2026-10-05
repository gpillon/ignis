// ignis kernel leaf -- the Flash-Next program's linear (spec flash-next/04,
// GitHub #302; OURS, ADR 0043).
//
// fn_linear routes a Linear by its stored format (layout.md section 6.1):
// - FP8 row-scale: kern's ignis_fp8_linear (GEMV up to 8 tokens, tensor cores
//   beyond).
// - BF16: the route below. The vendored BF16 linear only admits its own
//   registered 27B problems, and any Flash-Next linear may be re-converted to
//   BF16 (the coordinator's decision of 2026-10-05: a part whose FP8 cost the
//   conversion flags), so every Flash-Next shape needs one.
//
// The BF16 route, y[t][o] = sum_c W[o][c] * x[t][c], fp32 accumulation:
// - up to kGemvTokens tokens: one warp per output row over every token, the
//   weight row read once in 16-byte vectors;
// - wider: BF16 tensor cores (WMMA 16x16x16), a 64-row by 64-token block tile
//   over 32-wide K steps, operands staged in shared memory.
// Both need `cols` to be a multiple of 8 (16-byte rows); every Flash-Next
// linear's is a multiple of 64.

#include "flash_next_internal.h"

#include "ignis_fp8_linear.h"

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <string>

namespace ignis::flash_next {

namespace {

constexpr int32_t kGemvTokens = 8;
constexpr int32_t kGemvWarps = 8;

__device__ __forceinline__ void store_out(void *y, std::int64_t index, float value, bool y_f32) {
  if (y_f32) {
    static_cast<float *>(y)[index] = value;
  } else {
    static_cast<__nv_bfloat16 *>(y)[index] = __float2bfloat16(value);
  }
}

// One warp per output row, every token at once.
__global__ void bf16_gemv(const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ x,
                          void *__restrict__ y, int32_t rows, int32_t cols, int32_t tokens,
                          bool y_f32) {
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t row = static_cast<int32_t>(blockIdx.x) * kGemvWarps + warp;
  if (row >= rows) {
    return;
  }
  float acc[kGemvTokens] = {};
  const auto *w_row = reinterpret_cast<const uint4 *>(w + static_cast<std::int64_t>(row) * cols);
  const int32_t vectors = cols / 8;
  for (int32_t v = lane; v < vectors; v += 32) {
    const uint4 wv = w_row[v];
    const auto *wh = reinterpret_cast<const __nv_bfloat162 *>(&wv);
    for (int32_t t = 0; t < tokens; ++t) {
      const uint4 xv =
          reinterpret_cast<const uint4 *>(x + static_cast<std::int64_t>(t) * cols)[v];
      const auto *xh = reinterpret_cast<const __nv_bfloat162 *>(&xv);
      float sum = 0.0F;
      for (int32_t k = 0; k < 4; ++k) {
        const float2 a = __bfloat1622float2(wh[k]);
        const float2 b = __bfloat1622float2(xh[k]);
        sum = fmaf(a.x, b.x, sum);
        sum = fmaf(a.y, b.y, sum);
      }
      acc[t] += sum;
    }
  }
  for (int32_t t = 0; t < tokens; ++t) {
    float v = acc[t];
    for (int32_t offset = 16; offset > 0; offset /= 2) {
      v += __shfl_xor_sync(0xffffffffU, v, offset);
    }
    if (lane == 0) {
      store_out(y, static_cast<std::int64_t>(t) * rows + row, v, y_f32);
    }
  }
}

constexpr int32_t kTileRows = 64;
constexpr int32_t kTileTokens = 64;
constexpr int32_t kTileK = 32;
constexpr int32_t kGemmThreads = 128;  // 4 warps, each 32 rows x 32 tokens
constexpr int32_t kSmemPad = 8;        // bf16 elements, keeps rows 16-byte aligned off-bank

// A 64-row x 64-token tile of y on the tensor cores.
__global__ void bf16_gemm(const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ x,
                          void *__restrict__ y, int32_t rows, int32_t cols, int32_t tokens,
                          bool y_f32) {
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 w_tile[kTileRows][kTileK + kSmemPad];
  __shared__ __align__(32) __nv_bfloat16 x_tile[kTileTokens][kTileK + kSmemPad];
  __shared__ __align__(32) float out_tile[kTileRows][kTileTokens + 4];

  const int32_t row0 = static_cast<int32_t>(blockIdx.x) * kTileRows;
  const int32_t token0 = static_cast<int32_t>(blockIdx.y) * kTileTokens;
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t warp_row = (warp / 2) * 32;
  const int32_t warp_token = (warp % 2) * 32;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (auto &rowf : acc) {
    for (auto &f : rowf) {
      wmma::fill_fragment(f, 0.0F);
    }
  }

  for (int32_t k0 = 0; k0 < cols; k0 += kTileK) {
    // Stage both operands: 64 x 32 each, 8 bf16 (16 bytes) per thread step.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kTileRows * kTileK / 8; i += kGemmThreads) {
      const int32_t r = i / (kTileK / 8);
      const int32_t c = (i % (kTileK / 8)) * 8;
      const int32_t row = row0 + r;
      const int32_t token = token0 + r;
      const bool k_in = k0 + c < cols;
      uint4 wv = make_uint4(0, 0, 0, 0);
      uint4 xv = make_uint4(0, 0, 0, 0);
      if (row < rows && k_in) {
        wv = *reinterpret_cast<const uint4 *>(w + static_cast<std::int64_t>(row) * cols + k0 + c);
      }
      if (token < tokens && k_in) {
        xv = *reinterpret_cast<const uint4 *>(x + static_cast<std::int64_t>(token) * cols + k0 + c);
      }
      *reinterpret_cast<uint4 *>(&w_tile[r][c]) = wv;
      *reinterpret_cast<uint4 *>(&x_tile[r][c]) = xv;
    }
    __syncthreads();
    for (int32_t kk = 0; kk < kTileK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
      for (int32_t i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(a[i], &w_tile[warp_row + 16 * i][kk], kTileK + kSmemPad);
        // B (K x tokens) column-major is x_tile's token rows.
        wmma::load_matrix_sync(b[i], &x_tile[warp_token + 16 * i][kk], kTileK + kSmemPad);
      }
      for (int32_t i = 0; i < 2; ++i) {
        for (int32_t j = 0; j < 2; ++j) {
          wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  for (int32_t i = 0; i < 2; ++i) {
    for (int32_t j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(&out_tile[warp_row + 16 * i][warp_token + 16 * j], acc[i][j],
                              kTileTokens + 4, wmma::mem_row_major);
    }
  }
  __syncthreads();
  // y is token-major: y[token][row].
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < kTileRows * kTileTokens; i += kGemmThreads) {
    const int32_t t = i / kTileRows;
    const int32_t r = i % kTileRows;
    const int32_t row = row0 + r;
    const int32_t token = token0 + t;
    if (row < rows && token < tokens) {
      store_out(y, static_cast<std::int64_t>(token) * rows + row, out_tile[r][t], y_f32);
    }
  }
}

bool aligned16(const void *p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
}

}  // namespace

int32_t fn_linear(const Linear &w, const void *x, int32_t rows, void *y, bool y_f32,
                  ninfer::DeviceArena &scratch, cudaStream_t stream) {
  (void)scratch;  // neither route needs a workspace
  if (w.data == nullptr || x == nullptr || y == nullptr || rows <= 0 || w.rows <= 0 || w.cols <= 0) {
    fn_set_error("fn_linear: null operand or empty shape");
    return -1;
  }
  switch (w.format) {
    case WeightFormat::Fp8RowScale:
      if (ignis_fp8_linear(w.data, static_cast<uint32_t>(w.rows), static_cast<uint32_t>(w.cols), x,
                           static_cast<uint32_t>(rows), y, y_f32 ? 1U : 0U, stream) != 0) {
        fn_set_error(std::string("fn_linear (FP8 [") + std::to_string(w.rows) + "," +
                     std::to_string(w.cols) + "]): " + ignis_fp8_linear_last_error());
        return -1;
      }
      return 0;
    case WeightFormat::Bf16: {
      if (w.cols % 8 != 0 || !aligned16(w.data) || !aligned16(x)) {
        fn_set_error("fn_linear (BF16 [" + std::to_string(w.rows) + "," + std::to_string(w.cols) +
                     "]): cols must be a multiple of 8 and the weight and x 16-byte aligned");
        return -1;
      }
      const auto *wb = static_cast<const __nv_bfloat16 *>(w.data);
      const auto *xb = static_cast<const __nv_bfloat16 *>(x);
      if (rows <= kGemvTokens) {
        const dim3 grid((w.rows + kGemvWarps - 1) / kGemvWarps);
        bf16_gemv<<<grid, kGemvWarps * 32, 0, stream>>>(wb, xb, y, w.rows, w.cols, rows, y_f32);
      } else {
        const dim3 grid((w.rows + kTileRows - 1) / kTileRows, (rows + kTileTokens - 1) / kTileTokens);
        bf16_gemm<<<grid, kGemmThreads, 0, stream>>>(wb, xb, y, w.rows, w.cols, rows, y_f32);
      }
      if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        fn_set_error(std::string("fn_linear (BF16): launch failed: ") + cudaGetErrorString(err));
        return -1;
      }
      return 0;
    }
  }
  fn_set_error("fn_linear: unknown weight format " + std::to_string(static_cast<int32_t>(w.format)));
  return -1;
}

}  // namespace ignis::flash_next
