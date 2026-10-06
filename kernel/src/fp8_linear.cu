// ignis kernel leaf: the FP8 row-scale linear (spec flash-next/04's, written for spec 02's shared
// expert) -- OURS (kernel/include/ignis_moe.h, ADR 0043; no port claim).
//
// The weight is the container's FP8_E4M3FN_ROW_BF16S / row-scale-v1 payload (layout.md §6.1):
// E4M3FN codes [rows][cols], then BF16 scales [rows] at the next multiple of 256 bytes. Every
// product of an E4M3 code (4 significant bits) and a BF16 activation (8) is exact in fp32, so the
// only rounding is fp32 accumulation and the output's own; the row scale is applied once to the
// sum.
//
//   GEMV route, 1..8 tokens: one warp per output row, each lane streaming 16 codes at a time
//            and accumulating every token in registers in a fixed order, then a fixed butterfly.
//   MMA route, more tokens: 128 x 128 tiles on BF16 tensor cores (m16n8k16, fp32 accumulate),
//            the codes widened to BF16 exactly in registers; a three-stage cp.async pipeline.
//
// The SwiGLU form runs the gate and up weights in the same CTA and writes silu(g) * u in BF16.
// Both routes are deterministic: no atomics, fixed summation order per output.

#include "ignis_fp8_linear.h"
#include "moe_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>

namespace ignis_moe {
namespace {

constexpr int kGemvMaxTokens = 8;

__host__ __device__ inline size_t scale_offset(uint32_t rows, uint32_t cols) {
  return (static_cast<size_t>(rows) * cols + 255) / 256 * 256;
}

__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2 *>(&h));
}

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t v) {
  const float2 f = e4m3x2_to_float2(v);
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);  // exact: E4M3 fits BF16
  return *reinterpret_cast<const uint32_t *>(&b);
}

struct Fp8Weight {
  const uint8_t *codes;
  const __nv_bfloat16 *scales;
};

__host__ inline Fp8Weight fp8_weight(const void *payload, uint32_t rows, uint32_t cols) {
  const uint8_t *p = static_cast<const uint8_t *>(payload);
  return Fp8Weight{p, reinterpret_cast<const __nv_bfloat16 *>(p + scale_offset(rows, cols))};
}

// ---- GEMV route -----------------------------------------------------------------------------

// Sum over one row of codes against every token: acc[t] = sum_c e4m3(code[c]) * x[t][c].
__device__ __forceinline__ void gemv_row(const uint8_t *row, const __nv_bfloat16 *x, int tokens, int cols,
                                         float (&acc)[kGemvMaxTokens]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int t = 0; t < kGemvMaxTokens; ++t) acc[t] = 0.0f;
  for (int c = lane; c < cols / 16; c += 32) {
    const uint4 codes = __ldcs(reinterpret_cast<const uint4 *>(row) + c);
    const uint16_t *cp = reinterpret_cast<const uint16_t *>(&codes);
    float2 w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = e4m3x2_to_float2(cp[i]);
#pragma unroll
    for (int t = 0; t < kGemvMaxTokens; ++t) {
      if (t < tokens) {
        const uint4 *xr = reinterpret_cast<const uint4 *>(x + static_cast<size_t>(t) * cols) + 2 * c;
        const uint4 x0 = __ldg(xr), x1 = __ldg(xr + 1);
        const __nv_bfloat162 *xb0 = reinterpret_cast<const __nv_bfloat162 *>(&x0);
        const __nv_bfloat162 *xb1 = reinterpret_cast<const __nv_bfloat162 *>(&x1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 a = __bfloat1622float2(xb0[i]);
          acc[t] = fmaf(w[i].x, a.x, acc[t]);
          acc[t] = fmaf(w[i].y, a.y, acc[t]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 a = __bfloat1622float2(xb1[i]);
          acc[t] = fmaf(w[4 + i].x, a.x, acc[t]);
          acc[t] = fmaf(w[4 + i].y, a.y, acc[t]);
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < kGemvMaxTokens; ++t) acc[t] = warp_sum(acc[t]);
}

// MODE 0: y[t][r] = scale[r] * acc (fp32 or BF16). MODE 1: h[t][r] = silu(gate) * up in BF16.
template <int MODE>
__global__ void fp8_gemv_kernel(const __nv_bfloat16 *__restrict__ x, int tokens, int rows, int cols,
                                Fp8Weight w0, Fp8Weight w1, void *y, int y_f32) {
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  float a0[kGemvMaxTokens];
  gemv_row(w0.codes + static_cast<size_t>(r) * cols, x, tokens, cols, a0);
  const float s0 = __bfloat162float(w0.scales[r]);
  if (MODE == 0) {
    if (lane < tokens) {
      float v = 0.0f;
#pragma unroll
      for (int t = 0; t < kGemvMaxTokens; ++t) v = t == lane ? a0[t] * s0 : v;
      const size_t at = static_cast<size_t>(lane) * rows + r;
      if (y_f32) {
        static_cast<float *>(y)[at] = v;
      } else {
        static_cast<__nv_bfloat16 *>(y)[at] = __float2bfloat16_rn(v);
      }
    }
  } else {
    float a1[kGemvMaxTokens];
    gemv_row(w1.codes + static_cast<size_t>(r) * cols, x, tokens, cols, a1);
    const float s1 = __bfloat162float(w1.scales[r]);
    if (lane < tokens) {
      float g = 0.0f, u = 0.0f;
#pragma unroll
      for (int t = 0; t < kGemvMaxTokens; ++t) {
        g = t == lane ? a0[t] * s0 : g;
        u = t == lane ? a1[t] * s1 : u;
      }
      static_cast<__nv_bfloat16 *>(y)[static_cast<size_t>(lane) * rows + r] = __float2bfloat16_rn(silu(g) * u);
    }
  }
}

// ---- MMA route ------------------------------------------------------------------------------

constexpr int kBM = 128;     // tokens per CTA
constexpr int kBN = 128;     // output columns per CTA (MODE 1: 64 gate + 64 up)
constexpr int kBK = 32;      // inputs per stage
constexpr int kStages = 3;
constexpr int kAPad = kBK + 8;   // BF16 per A row in shared memory
constexpr int kBPad = kBK + 16;  // code bytes per B row in shared memory

struct MmaSmem {
  __nv_bfloat16 a[kStages][kBM][kAPad];
  uint8_t b[kStages][kBN][kBPad];
};

__device__ __forceinline__ void cp_async16_zfill(void *smem, const void *gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

template <int MODE>
__global__ void __launch_bounds__(256) fp8_mma_kernel(const __nv_bfloat16 *__restrict__ x, int tokens, int rows,
                                                      int cols, Fp8Weight w0, Fp8Weight w1, void *y, int y_f32) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  MmaSmem &sm = *reinterpret_cast<MmaSmem *>(smem_raw);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp >> 2;  // 0..1: 64 tokens each
  const int warp_n = warp & 3;   // 0..3: 32 columns each
  const int m0 = blockIdx.y * kBM;
  // Output column block: MODE 0 covers weight rows [n0, n0 + 128); MODE 1 covers gate rows
  // [n0, n0 + 64) as tile columns 0..63 and up rows [n0, n0 + 64) as 64..127.
  const int n0 = blockIdx.x * (MODE == 0 ? kBN : kBN / 2);

  auto b_row_ptr = [&](int j) -> const uint8_t * {
    int row;
    const uint8_t *codes;
    if (MODE == 0) {
      row = n0 + j;
      codes = w0.codes;
    } else {
      row = n0 + (j & 63);
      codes = j < 64 ? w0.codes : w1.codes;
    }
    return row < rows ? codes + static_cast<size_t>(row) * cols : nullptr;
  };

  auto load_stage = [&](int stage, int k0) {
    // A: 128 rows x 64 bytes = 512 chunks of 16 bytes, two per thread.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int chunk = tid + i * 256;
      const int row = chunk >> 2;
      const int part = chunk & 3;
      const bool valid = m0 + row < tokens;
      const __nv_bfloat16 *src = x + static_cast<size_t>(valid ? m0 + row : 0) * cols + k0 + part * 8;
      cp_async16_zfill(&sm.a[stage][row][part * 8], src, valid);
    }
    // B: 128 rows x 32 code bytes = 256 chunks of 16 bytes, one per thread.
    {
      const int row = tid >> 1;
      const int part = tid & 1;
      const uint8_t *r = b_row_ptr(row);
      cp_async16_zfill(&sm.b[stage][row][part * 16], r != nullptr ? r + k0 + part * 16 : w0.codes, r != nullptr);
    }
  };

  float acc[4][4][4] = {};
  const int k_steps = cols / kBK;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < k_steps) load_stage(s, s * kBK);
    cp_async_commit();
  }
  for (int ks = 0; ks < k_steps; ++ks) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const int next = ks + kStages - 1;
    if (next < k_steps) load_stage(next % kStages, next * kBK);
    cp_async_commit();
    const int st = ks % kStages;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      uint32_t af[4][4];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r = warp_m * 64 + mi * 16 + (lane & 15);
        const int c = kk + (lane >> 4) * 8;
        ldmatrix_x4(af[mi], &sm.a[st][r][c]);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int n = warp_n * 32 + ni * 8 + (lane >> 2);
        const int k = kk + 2 * (lane & 3);
        const uint32_t b0 = e4m3x2_to_bf16x2(*reinterpret_cast<const uint16_t *>(&sm.b[st][n][k]));
        const uint32_t b1 = e4m3x2_to_bf16x2(*reinterpret_cast<const uint16_t *>(&sm.b[st][n][k + 8]));
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) mma_bf16(acc[mi][ni], af[mi], b0, b1);
      }
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  const int g = lane >> 2;
  const int c = lane & 3;
  if (MODE == 0) {
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int col = n0 + warp_n * 32 + ni * 8 + 2 * c;
      if (col >= rows) continue;
      const float s0 = __bfloat162float(w0.scales[col]);
      const float s1 = __bfloat162float(w0.scales[col + 1]);
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int t = m0 + warp_m * 64 + mi * 16 + g + hh * 8;
          if (t >= tokens) continue;
          const float v0 = acc[mi][ni][2 * hh] * s0;
          const float v1 = acc[mi][ni][2 * hh + 1] * s1;
          const size_t at = static_cast<size_t>(t) * rows + col;
          if (y_f32) {
            *reinterpret_cast<float2 *>(static_cast<float *>(y) + at) = make_float2(v0, v1);
          } else {
            *reinterpret_cast<__nv_bfloat162 *>(static_cast<__nv_bfloat16 *>(y) + at) = __floats2bfloat162_rn(v0, v1);
          }
        }
      }
    }
  } else {
    // Gate and up of one output sit in different warps: meet in shared memory.
    float *tile = reinterpret_cast<float *>(smem_raw);  // [128][129]
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int j = warp_n * 32 + ni * 8 + 2 * c;
      const int row = n0 + (j & 63);
      const Fp8Weight &w = j < 64 ? w0 : w1;
      const float s0 = row < rows ? __bfloat162float(w.scales[row]) : 0.0f;
      const float s1 = row + 1 < rows ? __bfloat162float(w.scales[row + 1]) : 0.0f;
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int t = warp_m * 64 + mi * 16 + g + hh * 8;
          tile[t * 129 + j] = acc[mi][ni][2 * hh] * s0;
          tile[t * 129 + j + 1] = acc[mi][ni][2 * hh + 1] * s1;
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < kBM * 64; i += 256) {
      const int t = i >> 6;
      const int j = i & 63;
      if (m0 + t >= tokens || n0 + j >= rows) continue;
      const float h = silu(tile[t * 129 + j]) * tile[t * 129 + 64 + j];
      static_cast<__nv_bfloat16 *>(y)[static_cast<size_t>(m0 + t) * rows + n0 + j] = __float2bfloat16_rn(h);
    }
  }
}

constexpr size_t kMmaSmem = sizeof(MmaSmem) > sizeof(float) * kBM * 129 ? sizeof(MmaSmem) : sizeof(float) * kBM * 129;

constexpr int kMaxDevices = 64;
std::atomic<bool> g_prepared[kMaxDevices];

int32_t require_fp8_prepared(const char *op) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices || !g_prepared[device].load()) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_fp8_linear_prepare (or ignis_moe_prepare) at load");
  }
  return 0;
}

template <int MODE>
int32_t launch(const char *op, Fp8Weight w0, Fp8Weight w1, uint32_t rows, uint32_t cols, const void *x,
               uint32_t tokens, void *y, uint32_t y_f32, cudaStream_t s) {
  if (require_fp8_prepared(op) != 0) return -1;
  const __nv_bfloat16 *xb = static_cast<const __nv_bfloat16 *>(x);
  if (tokens <= static_cast<uint32_t>(kGemvMaxTokens)) {
    const int warps = 8;
    fp8_gemv_kernel<MODE><<<(rows + warps - 1) / warps, warps * 32, 0, s>>>(xb, static_cast<int>(tokens), static_cast<int>(rows),
                                                                          static_cast<int>(cols), w0, w1, y, static_cast<int>(y_f32));
  } else {
    const int cols_per_cta = MODE == 0 ? kBN : kBN / 2;
    const dim3 grid((rows + cols_per_cta - 1) / cols_per_cta, (tokens + kBM - 1) / kBM);
    fp8_mma_kernel<MODE><<<grid, 256, kMmaSmem, s>>>(xb, static_cast<int>(tokens), static_cast<int>(rows), static_cast<int>(cols),
                                                     w0, w1, y, static_cast<int>(y_f32));
  }
  return check_launch(op);
}

int32_t validate(const char *op, uint32_t rows, uint32_t cols, uint32_t tokens, const void *y) {
  // The MMA route stores two outputs at a time (float2 / bf16x2).
  if ((reinterpret_cast<uintptr_t>(y) & 7) != 0) return fail(std::string(op) + ": the output must be 8-byte aligned");
  if (rows == 0 || rows % 16 != 0) return fail(std::string(op) + ": rows must be a positive multiple of 16");
  if (cols == 0 || cols % 64 != 0) return fail(std::string(op) + ": cols must be a positive multiple of 64");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be at least 1");
  return 0;
}

}  // namespace
}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" int32_t ignis_fp8_linear_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(fp8_mma_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kMmaSmem));
  }
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(fp8_mma_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kMmaSmem));
  }
  if (err != cudaSuccess) return fail(std::string("ignis_fp8_linear_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_fp8_linear_prepare: device id out of range");
  g_prepared[device].store(true);
  return 0;
}

extern "C" const char *ignis_fp8_linear_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_fp8_linear(const void *weight, uint32_t rows, uint32_t cols, const void *x,
                                    uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  if (weight == nullptr || x == nullptr || y == nullptr) return fail("ignis_fp8_linear: NULL pointer");
  if (validate("ignis_fp8_linear", rows, cols, tokens, y) != 0) return -1;
  if ((reinterpret_cast<uintptr_t>(weight) | reinterpret_cast<uintptr_t>(x)) & 15) {
    return fail("ignis_fp8_linear: weight and x must be 16-byte aligned");
  }
  const Fp8Weight w = fp8_weight(weight, rows, cols);
  return launch<0>("ignis_fp8_linear", w, w, rows, cols, x, tokens, y, y_f32, static_cast<cudaStream_t>(stream));
}

extern "C" int32_t ignis_fp8_linear_swiglu(const void *gate, const void *up, uint32_t rows, uint32_t cols,
                                           const void *x, uint32_t tokens, void *h, void *stream) {
  if (gate == nullptr || up == nullptr || x == nullptr || h == nullptr) return fail("ignis_fp8_linear_swiglu: NULL pointer");
  if (validate("ignis_fp8_linear_swiglu", rows, cols, tokens, h) != 0) return -1;
  if ((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(x)) & 15) {
    return fail("ignis_fp8_linear_swiglu: weights and x must be 16-byte aligned");
  }
  return launch<1>("ignis_fp8_linear_swiglu", fp8_weight(gate, rows, cols), fp8_weight(up, rows, cols), rows, cols, x,
                   tokens, h, 0, static_cast<cudaStream_t>(stream));
}
