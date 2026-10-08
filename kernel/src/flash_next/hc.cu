// ignis kernel leaf -- Flash-Next's hyper-connection mix and inject (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). See hc.h for the math.
//
// A mix runs 97 times per decode round, so at decode widths it is latency, not
// bytes (~6.6 MB of weights; GitHub #306). Every route opens with the same
// norm: one CTA per (stream, row), which also takes the 4-output block-inject
// matvec as per-stream partials (no single-CTA GEMV), summed in stream order
// by whichever kernel finishes the mix -- at up to kMaxFusedRows decode rows
// the same code runs inside the down launch instead (hc_norm_down). Then:
// - the decode route (rows <= kDecodeRows, 4 streams): mix_down split by stream
//   over the card, then mix_up and the stream reduce in one kernel that
//   rebuilds the activation from the down partials in every CTA. Three
//   launches, all wide -- two at up to kMaxFusedRows rows since GitHub #306's
//   fusion study, where every down CTA runs the norm of its own stream itself
//   (hc_norm_down);
// - the prefill route: mix_down and mix_up through fn_linear (tensor cores),
//   the activation and the reduce on their own.
// Both routes round to BF16 at the module's points (hc.h); they differ only in
// fp32 summation order. Partials are combined in a fixed order: no atomics,
// every replay agrees.
//
// The previous sublayer's inject (and, after the MoE, its combine) can ride the
// fused decode route too (fn_hc_mix_after, GitHub #306 step 2): every down CTA
// rebuilds its stream as the inject would have left it -- recomputing the
// combine's gate and row where there is one -- and hc_up_reduce, which runs
// after every down CTA has read the residual, stores it.

#include "hc.h"

#include "fusion.h"

#include "../moe_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace ignis::flash_next {

namespace {

// Rows a mix processes at once: its scratch is a few of these rows' streams,
// so a prefill chunk of any width needs at most this many rows of it.
constexpr int32_t kWaveRows = 1024;
constexpr int32_t kThreads = 256;
constexpr int32_t kWarps = kThreads / 32;
// The norm's per-stream inject partials hold one accumulator per stream, and
// a norm thread holds up to kNormSpan elements of its stream (registers, and
// the row in shared memory): streams of up to kNormSpan * kThreads.
constexpr int32_t kMaxStreams = 8;
constexpr int32_t kNormSpan = 16;
// The decode route: its row ceiling, the stream count its lane map is built
// for, and the rank its activation (fp32, every row) fits in shared memory.
constexpr int32_t kDecodeRows = 8;
constexpr int32_t kDecodeStreams = 4;
constexpr int32_t kMaxDecodeRank = 512;

__device__ __forceinline__ float bf(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float block_sum(float v, float *shared) {
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    v += __shfl_xor_sync(0xffffffffU, v, offset);
  }
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0F;
  for (int32_t w = 0; w < kWarps; ++w) {
    total += shared[w];
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    v += __shfl_xor_sync(0xffffffffU, v, offset);
  }
  return v;
}

__device__ __forceinline__ float sigmoid(float v) {
  return 1.0F / (1.0F + __expf(-v));
}

// Two E4M3FN codes, exactly, as fp32.
__device__ __forceinline__ float2 e4m3x2(uint16_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2 *>(&h));
}

// acc += the dot of 8 BF16 weights with 8 BF16 inputs (one 16-byte vector each).
__device__ __forceinline__ float dot8(const uint4 &wv, const uint4 &xv, float acc) {
  const auto *wh = reinterpret_cast<const __nv_bfloat162 *>(&wv);
  const auto *xh = reinterpret_cast<const __nv_bfloat162 *>(&xv);
  for (int32_t k = 0; k < 4; ++k) {
    const float2 a = __bfloat1622float2(wh[k]);
    const float2 b = __bfloat1622float2(xh[k]);
    acc = fmaf(a.x, b.x, acc);
    acc = fmaf(a.y, b.y, acc);
  }
  return acc;
}

// The FP8 row-scale payload's scales (layout.md 6.1): BF16 [rows] at the next
// multiple of 256 bytes after the codes.
const __nv_bfloat16 *fp8_scales(const Linear &w) {
  const std::size_t codes = static_cast<std::size_t>(w.rows) * w.cols;
  return reinterpret_cast<const __nv_bfloat16 *>(static_cast<const uint8_t *>(w.data) + (codes + 255) / 256 * 256);
}

// One (row, stream) of the grouped RMSNorm, (1 + w), by the whole CTA: the
// stream's `width` elements are load(i) (its BF16 values, as fp32) and `w`'s,
// the normed row to `out` (shared memory) and, unless null, to `global`.
// Thread t holds elements t, t + kThreads, ... and sums their squares in that
// order (the strided loop's order). Its loads are all issued before the
// reduce: one memory round trip.
template <class Load>
__device__ __forceinline__ void norm_stream_of(Load load, const __nv_bfloat16 *__restrict__ w, __nv_bfloat16 *out,
                                               __nv_bfloat16 *__restrict__ global, int32_t width, float eps,
                                               float *partial) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  float v[kNormSpan];
  float scale[kNormSpan];
#pragma unroll
  for (int32_t j = 0; j < kNormSpan; ++j) {
    const int32_t i = tid + j * kThreads;
    v[j] = i < width ? load(i) : 0.0F;
    scale[j] = i < width ? 1.0F + __bfloat162float(w[i]) : 0.0F;
  }
  float squares = 0.0F;
#pragma unroll
  for (int32_t j = 0; j < kNormSpan; ++j) {
    if (tid + j * kThreads < width) {
      squares = fmaf(v[j], v[j], squares);
    }
  }
  const float inv = rsqrtf(block_sum(squares, partial) / static_cast<float>(width) + eps);
#pragma unroll
  for (int32_t j = 0; j < kNormSpan; ++j) {
    const int32_t i = tid + j * kThreads;
    if (i < width) {
      const __nv_bfloat16 n = __float2bfloat16(v[j] * inv * scale[j]);
      if (global != nullptr) {
        global[i] = n;
      }
      out[i] = n;
    }
  }
}

// norm_stream_of the stream `in` itself.
__device__ __forceinline__ void norm_stream(const __nv_bfloat16 *__restrict__ in, const __nv_bfloat16 *__restrict__ w,
                                            __nv_bfloat16 *out, __nv_bfloat16 *__restrict__ global, int32_t width,
                                            float eps, float *partial) {
  norm_stream_of([&](int32_t i) { return __bfloat162float(in[i]); }, w, out, global, width, eps, partial);
}

// The block-inject matvec over one normed (row, stream) `row_normed` (shared
// memory, read back in 16-byte vectors): part[j] = output j over stream s's
// columns only (inject_weight sums the streams).
__device__ __forceinline__ void inject_partials(const __nv_bfloat16 *row_normed, const __nv_bfloat16 *__restrict__ inject,
                                                int32_t s, int32_t streams, int32_t width, float *__restrict__ part,
                                                float (*partials)[kWarps]) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  __syncthreads();
  float dots[kMaxStreams] = {};
  const std::int64_t inject_stride = static_cast<std::int64_t>(streams) * width;
#pragma unroll 2
  for (int32_t c = tid; c < width / 8; c += kThreads) {
    const uint4 nv = reinterpret_cast<const uint4 *>(row_normed)[c];
#pragma unroll
    for (int32_t j = 0; j < kMaxStreams; ++j) {
      if (j < streams) {
        const auto *wr = reinterpret_cast<const uint4 *>(inject + j * inject_stride + s * width);
        dots[j] = dot8(__ldg(wr + c), nv, dots[j]);
      }
    }
  }
  const int32_t warp = tid / 32;
  const int32_t lane = tid % 32;
#pragma unroll
  for (int32_t j = 0; j < kMaxStreams; ++j) {
    if (j < streams) {
      const float v = warp_sum(dots[j]);
      if (lane == 0) {
        partials[j][warp] = v;
      }
    }
  }
  __syncthreads();
  if (tid < streams) {
    float total = 0.0F;
    for (int32_t k = 0; k < kWarps; ++k) {
      total += partials[tid][k];
    }
    part[tid] = total;
  }
}

// normed[r] = grouped RMSNorm of hidden[r], (1 + w), one CTA per (stream,
// row); with block-inject weights, part[r][s][j] = the inject matvec's output
// j over stream s's columns only (inject_weight sums the streams). Two memory
// round trips per CTA, not one per element.
__global__ void hc_norm(const __nv_bfloat16 *__restrict__ hidden, const __nv_bfloat16 *__restrict__ w,
                        const __nv_bfloat16 *__restrict__ inject, __nv_bfloat16 *__restrict__ normed,
                        float *__restrict__ part, int32_t streams, int32_t width, float eps) {
  __shared__ float partial[kWarps];
  __shared__ float partials[kMaxStreams][kWarps];
  __shared__ __align__(16) __nv_bfloat16 row_normed[kNormSpan * kThreads];
  const int32_t s = static_cast<int32_t>(blockIdx.x);
  const std::int64_t row = blockIdx.y;
  const std::int64_t base = (row * streams + s) * width;
  norm_stream(hidden + base, w + s * width, row_normed, normed + base, width, eps, partial);
  if (inject != nullptr) {
    inject_partials(row_normed, inject, s, streams, width, part + (row * streams + s) * streams, partials);
  }
}

// inj[s] = bf16(2 * bf16(sigmoid(bf16(raw / streams)))), raw = the inject
// matvec's output s rounded to BF16 as a linear's output is: its per-stream
// partials (`part`, one row's [streams][streams]) summed in stream order.
__device__ __forceinline__ float inject_weight(const float *part, int32_t s, int32_t streams) {
  float raw = 0.0F;
  for (int32_t q = 0; q < streams; ++q) {
    raw += part[q * streams + s];
  }
  const float scaled = bf(bf(raw) / static_cast<float>(streams));
  return bf(2.0F * bf(sigmoid(scaled)));
}

// act = silu(bf16(down / streams)), each step rounded as the BF16 module does.
__global__ void hc_down_activation(const __nv_bfloat16 *__restrict__ down,
                                   __nv_bfloat16 *__restrict__ act, std::int64_t count,
                                   float streams) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float d = bf(__bfloat162float(down[i]) / streams);
    act[i] = __float2bfloat16(d / (1.0F + __expf(-d)));
  }
}

// x[r][h] = mean_s(bf16(sigmoid(up) * normed)), and, with the inject partials,
// inj[r][s] (inject_weight).
__global__ void hc_reduce(const __nv_bfloat16 *__restrict__ up, const __nv_bfloat16 *__restrict__ normed,
                          const float *__restrict__ inject_part, __nv_bfloat16 *__restrict__ x,
                          float *__restrict__ inj, int32_t streams, int32_t width) {
  const std::int64_t row = blockIdx.x;
  for (int32_t h = static_cast<int32_t>(threadIdx.x); h < width; h += kThreads) {
    float sum = 0.0F;
    for (int32_t s = 0; s < streams; ++s) {
      const std::int64_t i = (row * streams + s) * width + h;
      const float m = bf(sigmoid(__bfloat162float(up[i])));
      sum += bf(m * __bfloat162float(normed[i]));
    }
    x[row * width + h] = __float2bfloat16(sum / static_cast<float>(streams));
  }
  if (inject_part != nullptr && static_cast<int32_t>(threadIdx.x) < streams) {
    const int32_t s = static_cast<int32_t>(threadIdx.x);
    inj[row * streams + s] = inject_weight(inject_part + row * streams * streams, s, streams);
  }
}

// A 16-byte load of the normed rows: from global memory through the read-only
// path, or from shared memory.
template <bool kShared>
__device__ __forceinline__ uint4 load16(const uint4 *p) {
  if constexpr (kShared) {
    return *p;
  } else {
    return __ldg(p);
  }
}

// The decode route's mix_down, split by stream: part[s][t][k] = the sum over
// stream s's columns c of W[k][c] * normed[t][c], in fp32 (an FP8 row scale is
// applied once the streams are summed). One warp per (output k, stream s);
// `normed` is stream s of row 0, its rows `row_stride` apart.
template <bool kFp8, bool kShared>
__device__ __forceinline__ void down_split(const void *__restrict__ w, const __nv_bfloat16 *normed,
                                           std::int64_t row_stride, float *__restrict__ part, int32_t rank,
                                           int32_t streams, int32_t width, int32_t tokens, int32_t s) {
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t k = static_cast<int32_t>(blockIdx.x) * kWarps + warp;
  if (k >= rank) {
    return;
  }
  const std::int64_t cols = static_cast<std::int64_t>(streams) * width;
  const std::int64_t first = k * cols + static_cast<std::int64_t>(s) * width;
  float acc[kDecodeRows] = {};
  if (kFp8) {
    const auto *codes = reinterpret_cast<const uint4 *>(static_cast<const uint8_t *>(w) + first);
#pragma unroll 4
    for (int32_t v = lane; v < width / 16; v += 32) {
      const uint4 c = __ldcs(codes + v);
      const auto *cp = reinterpret_cast<const uint16_t *>(&c);
      float2 wf[8];
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        wf[i] = e4m3x2(cp[i]);
      }
#pragma unroll
      for (int32_t t = 0; t < kDecodeRows; ++t) {
        if (t < tokens) {
          const auto *xr = reinterpret_cast<const uint4 *>(normed + t * row_stride) + 2 * v;
          const uint4 x0 = load16<kShared>(xr);
          const uint4 x1 = load16<kShared>(xr + 1);
          const auto *xb0 = reinterpret_cast<const __nv_bfloat162 *>(&x0);
          const auto *xb1 = reinterpret_cast<const __nv_bfloat162 *>(&x1);
#pragma unroll
          for (int32_t i = 0; i < 4; ++i) {
            const float2 a = __bfloat1622float2(xb0[i]);
            acc[t] = fmaf(wf[i].x, a.x, acc[t]);
            acc[t] = fmaf(wf[i].y, a.y, acc[t]);
          }
#pragma unroll
          for (int32_t i = 0; i < 4; ++i) {
            const float2 a = __bfloat1622float2(xb1[i]);
            acc[t] = fmaf(wf[4 + i].x, a.x, acc[t]);
            acc[t] = fmaf(wf[4 + i].y, a.y, acc[t]);
          }
        }
      }
    }
  } else {
    const auto *row = reinterpret_cast<const uint4 *>(static_cast<const __nv_bfloat16 *>(w) + first);
#pragma unroll 4
    for (int32_t v = lane; v < width / 8; v += 32) {
      const uint4 wv = __ldcs(row + v);
#pragma unroll
      for (int32_t t = 0; t < kDecodeRows; ++t) {
        if (t < tokens) {
          acc[t] = dot8(wv, load16<kShared>(reinterpret_cast<const uint4 *>(normed + t * row_stride) + v), acc[t]);
        }
      }
    }
  }
#pragma unroll
  for (int32_t t = 0; t < kDecodeRows; ++t) {
    if (t < tokens) {
      const float v = warp_sum(acc[t]);
      if (lane == 0) {
        part[(static_cast<std::int64_t>(s) * tokens + t) * rank + k] = v;
      }
    }
  }
}

template <bool kFp8>
__global__ void hc_down_split(const void *__restrict__ w, const __nv_bfloat16 *__restrict__ normed,
                              float *__restrict__ part, int32_t rank, int32_t streams, int32_t width,
                              int32_t tokens) {
  const int32_t s = static_cast<int32_t>(blockIdx.y);
  down_split<kFp8, false>(w, normed + static_cast<std::int64_t>(s) * width, static_cast<std::int64_t>(streams) * width,
                          part, rank, streams, width, tokens, s);
}

// What the fused down launch adds to the residual as it reads it (GitHub #306,
// step 2): the previous sublayer's inject, hidden_s + bf16(y * inj_s), y the
// sublayer's BF16 output -- or, with `acc`, the MoE combine of acc, shared, x
// and w_gate (ignis_moe_combine's operands, its code), which stream 0's writer
// CTA also stores to `y` for hc_up_reduce. Null `inj`: nothing pending.
struct PendingArgs {
  __nv_bfloat16 *y = nullptr;
  const float *inj = nullptr;
  const long long *acc = nullptr;
  const float *shared = nullptr;
  const __nv_bfloat16 *x = nullptr;
  const __nv_bfloat16 *w_gate = nullptr;
};

// The MoE combine's gate of one row, by the whole CTA: sigmoid(x . w_gate) as
// ignis_moe_combine's CTA computes it -- each thread's strided fmaf chain, the
// warp butterfly, the warps' partials summed in order by thread 0 -- broadcast
// through `slot`. (Both run kThreads == 256 threads, so the chains are the same.)
__device__ __forceinline__ float combine_gate(const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ w,
                                              int32_t width, float *partial, float *slot) {
  float dot = 0.0F;
  for (int32_t k = static_cast<int32_t>(threadIdx.x); k < width; k += kThreads) {
    dot = fmaf(__bfloat162float(x[k]), __bfloat162float(w[k]), dot);
  }
  dot = warp_sum(dot);
  if ((threadIdx.x & 31) == 0) {
    partial[threadIdx.x >> 5] = dot;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (int32_t w8 = 0; w8 < kWarps; ++w8) {
      sum += partial[w8];
    }
    *slot = 1.0F / (1.0F + expf(-sum));
  }
  __syncthreads();
  return *slot;
}

// The decode route's norm and mix_down in one launch (GitHub #306, the fusion
// study). CTA (b, s) of the down split first normalizes stream s of every row
// into shared memory with hc_norm's own code, then takes hc_down_split's
// outputs from there: its partials are hc_down_split's, bit for bit. The last
// CTA of each stream (b == gridDim.x - 1, past the down split's blocks) writes
// the normed rows hc_up_reduce reads and the inject partials, as hc_norm does.
// With a pending inject every CTA rebuilds its stream as hc_inject_kernel (and
// combine_kernel) would have left it, value for value; hc_up_reduce stores it.
// The pending kind is a template parameter so the rebuild's loads stay free of
// stores and branches: the norm issues them all before its reduce, one memory
// round trip (a runtime branch with the combined y's store inside serialized
// them: the folded mix ran 3.8 us slower than the inject and the mix apart).
// Dynamic shared memory: the rows' normed stream, tokens * width BF16.
enum PendingKind : int { kNothing = 0, kInject = 1, kCombine = 2 };

template <bool kFp8, int kPending>
__global__ void hc_norm_down(const __nv_bfloat16 *__restrict__ hidden, const __nv_bfloat16 *__restrict__ w_norm,
                             const __nv_bfloat16 *__restrict__ inject, const void *__restrict__ w_down,
                             __nv_bfloat16 *__restrict__ normed, float *__restrict__ inject_part,
                             float *__restrict__ down_part, int32_t rank, int32_t streams, int32_t width,
                             int32_t tokens, float eps, PendingArgs p) {
  extern __shared__ __align__(16) unsigned char fused_smem[];
  __shared__ float partial[kWarps];
  __shared__ float partials[kMaxStreams][kWarps];
  __shared__ float gate_slot;
  auto *rows_normed = reinterpret_cast<__nv_bfloat16 *>(fused_smem);
  const int32_t s = static_cast<int32_t>(blockIdx.y);
  const bool writer = blockIdx.x == gridDim.x - 1;
  for (int32_t t = 0; t < tokens; ++t) {
    const std::int64_t base = (static_cast<std::int64_t>(t) * streams + s) * width;
    if constexpr (kPending == kNothing) {
      norm_stream(hidden + base, w_norm + s * width, rows_normed + t * width, writer ? normed + base : nullptr, width,
                  eps, partial);
    } else {
      const std::int64_t row = static_cast<std::int64_t>(t) * width;
      float gate = 0.0F;
      if constexpr (kPending == kCombine) {
        gate = combine_gate(p.x + row, p.w_gate, width, partial, &gate_slot);
      }
      const float injection = __ldg(p.inj + t * streams + s);
      // The combine's BF16 row, as combine_kernel stores it.
      const auto combined = [&](int32_t i) {
        return __float2bfloat16_rn(fmaf(gate, __ldg(p.shared + row + i), ignis_moe::from_fixed(__ldg(p.acc + row + i))));
      };
      const auto load = [&](int32_t i) {
        float y;
        if constexpr (kPending == kCombine) {
          y = __bfloat162float(combined(i));
        } else {
          y = __bfloat162float(__ldg(p.y + row + i));
        }
        return bf(__bfloat162float(hidden[base + i]) + bf(y * injection));
      };
      norm_stream_of(load, w_norm + s * width, rows_normed + t * width, writer ? normed + base : nullptr, width, eps,
                     partial);
      if constexpr (kPending == kCombine) {
        // Stream 0's writer stores the combined row for hc_up_reduce, after the batch of loads.
        if (writer && s == 0) {
          for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += kThreads) {
            p.y[row + i] = combined(i);
          }
        }
      }
    }
    if (writer && inject != nullptr) {
      inject_partials(rows_normed + t * width, inject, s, streams, width,
                      inject_part + (static_cast<std::int64_t>(t) * streams + s) * streams, partials);
    }
  }
  if (writer) {
    return;
  }
  __syncthreads();
  down_split<kFp8, true>(w_down, rows_normed, width, down_part, rank, streams, width, tokens, s);
}

// The decode route folds the norm into its down launch unless switched off (fusion.h,
// IGNIS_FN_HC_FUSED).
bool decode_fused() { return fused(Fusion::HcNorm); }

// The rows the fused launch takes: every down CTA normalizes each row in turn,
// so past three rows the serial norms cost what the saved launch did
// (bench_flash_next_hc, 2026-10-08: 1 row 14.2 -> 10.6 us, 3 rows 18.1 ->
// 16.5, 4 rows 19.9 -> 20.1, 8 rows 27.7 -> 35.3); past it the norm keeps its
// own launch.
constexpr int32_t kMaxFusedRows = 3;
// Its dynamic shared memory, the rows' normed stream, inside the default 48 KiB
// with the kernel's static arrays.
static_assert(kMaxFusedRows * kNormSpan * kThreads * sizeof(__nv_bfloat16) +
                      (kWarps + kMaxStreams * kWarps + 1) * sizeof(float) <=
                  48 * 1024,
              "the fused norm's rows outgrow the default dynamic shared memory");

// The decode route's mix_up and reduce in one pass, 16 h per CTA, two per
// warp. Every CTA first rebuilds the activation from the down partials
// (act = silu(bf16(bf16(down) / streams)), down = the streams' partials summed
// in order, FP8-scaled, rounded as a linear's output), then four lanes share
// each (stream, h) row of mix_up: lane = 16 * (h of the pair) + 4 * stream +
// quarter. x[t][h] = bf16(sum over streams in order of bf16(bf16(sigmoid(up))
// * normed) / streams), and block 0 writes inj from the inject partials.
template <bool kFp8>
__global__ void hc_up_reduce(const void *__restrict__ w, const __nv_bfloat16 *__restrict__ up_scales,
                             const float *__restrict__ down_part, const __nv_bfloat16 *__restrict__ down_scales,
                             const __nv_bfloat16 *__restrict__ normed, const float *__restrict__ inject_part,
                             __nv_bfloat16 *__restrict__ x, float *__restrict__ inj, int32_t rank, int32_t width,
                             int32_t tokens, PendingArgs pending, __nv_bfloat16 *__restrict__ residual,
                             long long *__restrict__ acc_zero) {
  constexpr int32_t streams = kDecodeStreams;
  __shared__ __align__(16) float act[kDecodeRows * kMaxDecodeRank];
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < tokens * rank; i += kThreads) {
    const int32_t t = i / rank;
    const int32_t k = i % rank;
    float sum = 0.0F;
    for (int32_t s = 0; s < streams; ++s) {
      sum += down_part[(static_cast<std::int64_t>(s) * tokens + t) * rank + k];
    }
    const float down = bf(down_scales != nullptr ? sum * __bfloat162float(down_scales[k]) : sum);
    const float d = bf(down / static_cast<float>(streams));
    act[i] = bf(d / (1.0F + __expf(-d)));
  }
  if (inject_part != nullptr && blockIdx.x == 0 && static_cast<int32_t>(threadIdx.x) < tokens * streams) {
    const int32_t t = static_cast<int32_t>(threadIdx.x) / streams;
    const int32_t s = static_cast<int32_t>(threadIdx.x) % streams;
    inj[t * streams + s] = inject_weight(inject_part + t * streams * streams, s, streams);
  }
  __syncthreads();

  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  const int32_t pair = lane / 16;
  const int32_t s = (lane / 4) % 4;
  const int32_t quarter = lane % 4;
  const int32_t h = (static_cast<int32_t>(blockIdx.x) * kWarps + warp) * 2 + pair;
  const std::int64_t o = static_cast<std::int64_t>(s) * width + h;  // the mix_up row
  float acc[kDecodeRows] = {};
  if (kFp8) {
    const auto *codes = reinterpret_cast<const uint4 *>(static_cast<const uint8_t *>(w) + o * rank);
#pragma unroll 2
    for (int32_t v = quarter; v < rank / 16; v += 4) {
      const uint4 c = __ldcs(codes + v);
      const auto *cp = reinterpret_cast<const uint16_t *>(&c);
      float2 wf[8];
#pragma unroll
      for (int32_t i = 0; i < 8; ++i) {
        wf[i] = e4m3x2(cp[i]);
      }
#pragma unroll
      for (int32_t t = 0; t < kDecodeRows; ++t) {
        if (t < tokens) {
          const float4 *a = reinterpret_cast<const float4 *>(&act[t * rank + v * 16]);
#pragma unroll
          for (int32_t i = 0; i < 4; ++i) {
            const float4 av = a[i];
            acc[t] = fmaf(wf[2 * i].x, av.x, acc[t]);
            acc[t] = fmaf(wf[2 * i].y, av.y, acc[t]);
            acc[t] = fmaf(wf[2 * i + 1].x, av.z, acc[t]);
            acc[t] = fmaf(wf[2 * i + 1].y, av.w, acc[t]);
          }
        }
      }
    }
  } else {
    const auto *row = reinterpret_cast<const uint4 *>(static_cast<const __nv_bfloat16 *>(w) + o * rank);
#pragma unroll 2
    for (int32_t v = quarter; v < rank / 8; v += 4) {
      const uint4 wv = __ldcs(row + v);
      const auto *wh = reinterpret_cast<const __nv_bfloat162 *>(&wv);
#pragma unroll
      for (int32_t t = 0; t < kDecodeRows; ++t) {
        if (t < tokens) {
          const float4 *a = reinterpret_cast<const float4 *>(&act[t * rank + v * 8]);
#pragma unroll
          for (int32_t i = 0; i < 2; ++i) {
            const float4 av = a[i];
            const float2 w0 = __bfloat1622float2(wh[2 * i]);
            const float2 w1 = __bfloat1622float2(wh[2 * i + 1]);
            acc[t] = fmaf(w0.x, av.x, acc[t]);
            acc[t] = fmaf(w0.y, av.y, acc[t]);
            acc[t] = fmaf(w1.x, av.z, acc[t]);
            acc[t] = fmaf(w1.y, av.w, acc[t]);
          }
        }
      }
    }
  }
  const float scale = kFp8 ? __bfloat162float(up_scales[o]) : 1.0F;
  const int32_t leader = lane & ~15;  // stream 0, quarter 0 of this lane's h
#pragma unroll
  for (int32_t t = 0; t < kDecodeRows; ++t) {
    if (t < tokens) {
      float dot = acc[t];
      dot += __shfl_xor_sync(0xffffffffU, dot, 1);
      dot += __shfl_xor_sync(0xffffffffU, dot, 2);
      const float up = bf(kFp8 ? dot * scale : dot);
      const float m = bf(sigmoid(up));
      const float term = bf(m * __bfloat162float(normed[t * streams * static_cast<std::int64_t>(width) + o]));
      float sum = 0.0F;
#pragma unroll
      for (int32_t q = 0; q < streams; ++q) {
        sum += __shfl_sync(0xffffffffU, term, leader + 4 * q);
      }
      if (lane == leader) {
        x[static_cast<std::int64_t>(t) * width + h] = __float2bfloat16(sum / static_cast<float>(streams));
      }
    }
  }
  // A pending inject (GitHub #306, step 2): the residual the down launch rebuilt, stored, each
  // (stream, h) by its quarter-0 lane as hc_inject_kernel would store it; with a combine, the
  // accumulator it read zeroed for the next MoE op (ignis_moe_combine's contract), by stream 0's.
  if (pending.inj != nullptr && quarter == 0) {
    for (int32_t t = 0; t < tokens; ++t) {
      const std::int64_t i = t * streams * static_cast<std::int64_t>(width) + o;
      const float injection =
          bf(__bfloat162float(pending.y[static_cast<std::int64_t>(t) * width + h]) * pending.inj[t * streams + s]);
      residual[i] = __float2bfloat16(__bfloat162float(residual[i]) + injection);
      if (acc_zero != nullptr && s == 0) {
        acc_zero[static_cast<std::int64_t>(t) * width + h] = 0;
      }
    }
  }
}

// hidden[r][s][h] = bf16(hidden + bf16(y[r][h] * inj[r][s])).
__global__ void hc_inject_kernel(const __nv_bfloat16 *__restrict__ y, const float *__restrict__ inj,
                                 __nv_bfloat16 *__restrict__ hidden, int32_t streams, int32_t width,
                                 std::int64_t count) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::int64_t per_row = static_cast<std::int64_t>(streams) * width;
  const std::int64_t row = i / per_row;
  const int32_t s = static_cast<int32_t>((i % per_row) / width);
  const int32_t h = static_cast<int32_t>(i % width);
  const float injection = bf(__bfloat162float(y[row * width + h]) * inj[row * streams + s]);
  hidden[i] = __float2bfloat16(__bfloat162float(hidden[i]) + injection);
}

std::size_t aligned(std::size_t bytes) {
  return (bytes + 255) / 256 * 256;
}

bool aligned16(const void *p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
}

bool launched(const char *what) {
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    fn_set_error(std::string(what) + ": launch failed: " + cudaGetErrorString(err));
    return false;
  }
  return true;
}

// The decode route takes `rows` rows of this geometry and these weights.
bool decode_route(const Geometry &g, const HcWeights &w, int32_t rows) {
  const auto vectors = [](const Linear &l) {
    return l.format == WeightFormat::Fp8RowScale ? 16 : 8;  // elements per 16-byte load
  };
  return rows <= kDecodeRows && g.streams == kDecodeStreams && g.hidden % 16 == 0 &&
         g.hc_rank <= kMaxDecodeRank && g.hidden % vectors(w.mix_down) == 0 &&
         g.hc_rank % vectors(w.mix_up) == 0 && g.hc_rank % 4 == 0 && aligned16(w.mix_down.data) &&
         aligned16(w.mix_up.data);
}

}  // namespace

void fn_hc_set_decode_fused(bool on) { set_fused(Fusion::HcNorm, on); }

std::size_t fn_hc_mix_scratch_bytes(const Geometry &g, int32_t rows) {
  const auto wave = static_cast<std::size_t>(std::min(rows, kWaveRows));
  const auto width = static_cast<std::size_t>(g.residual_width());
  const auto rank = static_cast<std::size_t>(g.hc_rank);
  const auto streams = static_cast<std::size_t>(g.streams);
  const auto decode = static_cast<std::size_t>(std::min(rows, kDecodeRows));
  // normed, up: BF16 [wave][streams * hidden]; down, act: BF16 [wave][rank];
  // the inject partials: fp32 [wave][streams][streams]; the decode route's
  // down partials: fp32 [streams][rows][rank].
  return 2 * aligned(wave * width * 2) + 2 * aligned(wave * rank * 2) + aligned(wave * streams * streams * 4) +
         aligned(streams * decode * rank * 4);
}

namespace {

// The rows the folded inject (GitHub #306, step 2) takes: one. Every down CTA rebuilds each row
// in turn (and recomputes each row's combine gate), so the fold wins at one row and loses from
// two (bench_flash_next_hc --fold 1|0, 2026-10-08, us a mix, inject / inject and combine: 1 row
// 11.48 / 13.19 folded against 11.81 / 14.56 apart; 2 rows 14.33 / 19.36 against 13.71 / 16.51;
// 3 rows 18.66 / 25.74 against 17.81 / 20.59); past it the inject runs on its own.
constexpr int32_t kMaxFoldRows = 1;

bool folds(const Geometry &g, const HcWeights &w, int32_t rows) {
  return decode_route(g, w, rows) && rows <= kMaxFoldRows && rows <= kMaxFusedRows && decode_fused() &&
         fused(Fusion::Inject);
}

// fn_hc_mix, and, with `pending` (only where folds() holds), the inject folded in: the fused down
// launch rebuilds the residual, hc_up_reduce stores it to `residual` (the mix's own input) and
// zeroes `acc_zero`.
int32_t mix(const Geometry &g, const HcWeights &w, const void *hidden, int32_t rows, void *x, float *inj,
            ninfer::DeviceArena &scratch, cudaStream_t stream, const PendingArgs &pending,
            __nv_bfloat16 *residual, long long *acc_zero);

}  // namespace

int32_t fn_hc_mix(const Geometry &g, const HcWeights &w, const void *hidden, int32_t rows, void *x,
                  float *inj, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  return mix(g, w, hidden, rows, x, inj, scratch, stream, PendingArgs{}, nullptr, nullptr);
}

int32_t fn_hc_flush(const Geometry &g, const PendingInject &pending, void *hidden, int32_t rows,
                    cudaStream_t stream) {
  if (!pending.pending()) {
    return 0;
  }
  if (pending.acc != nullptr &&
      ignis_moe_combine(pending.acc, pending.shared, pending.x, pending.w_gate, static_cast<uint32_t>(rows), pending.y,
                        stream) != 0) {
    fn_set_error(std::string("the MoE combine: ") + ignis_moe_last_error());
    return -1;
  }
  return fn_hc_inject(g, pending.y, pending.inj, rows, hidden, stream);
}

int32_t fn_hc_mix_after(const Geometry &g, const HcWeights &w, const PendingInject &pending, void *hidden,
                        int32_t rows, void *x, float *inj, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  if (!pending.pending() || !folds(g, w, rows)) {
    if (fn_hc_flush(g, pending, hidden, rows, stream) != 0) {
      return -1;
    }
    return fn_hc_mix(g, w, hidden, rows, x, inj, scratch, stream);
  }
  if (pending.y == nullptr || pending.inj == inj || pending.y == x || pending.y == hidden ||
      (pending.acc != nullptr && (pending.shared == nullptr || pending.x == nullptr || pending.w_gate == nullptr))) {
    fn_set_error("fn_hc_mix_after: a pending inject needs y and the combine's operands, and its injection weights "
                 "and y apart from the mix's outputs");
    return -1;
  }
  PendingArgs p;
  p.y = static_cast<__nv_bfloat16 *>(pending.y);
  p.inj = pending.inj;
  p.acc = reinterpret_cast<const long long *>(pending.acc);
  p.shared = pending.shared;
  p.x = static_cast<const __nv_bfloat16 *>(pending.x);
  p.w_gate = static_cast<const __nv_bfloat16 *>(pending.w_gate);
  return mix(g, w, hidden, rows, x, inj, scratch, stream, p, static_cast<__nv_bfloat16 *>(hidden),
             reinterpret_cast<long long *>(pending.acc));
}

namespace {

int32_t mix(const Geometry &g, const HcWeights &w, const void *hidden, int32_t rows, void *x, float *inj,
            ninfer::DeviceArena &scratch, cudaStream_t stream, const PendingArgs &pending,
            __nv_bfloat16 *residual, long long *acc_zero) {
  if (hidden == nullptr || x == nullptr || w.hc_norm == nullptr || rows <= 0) {
    fn_set_error("fn_hc_mix: null operand or no rows");
    return -1;
  }
  if (g.streams <= 0 || g.streams > kMaxStreams || g.hidden <= 0 || g.hidden % 8 != 0 ||
      g.hidden > kNormSpan * kThreads) {
    fn_set_error("fn_hc_mix: " + std::to_string(g.streams) + " streams of " + std::to_string(g.hidden) +
                 " (supported: 1.." + std::to_string(kMaxStreams) + " streams, a multiple of 8 up to " +
                 std::to_string(kNormSpan * kThreads) + " wide)");
    return -1;
  }
  // The decode route indexes both projections by the geometry, the prefill
  // route by the Linear's own shape: they must be one shape.
  if (w.mix_down.rows != g.hc_rank || w.mix_down.cols != g.residual_width() || w.mix_up.rows != g.residual_width() ||
      w.mix_up.cols != g.hc_rank) {
    fn_set_error("fn_hc_mix: mix_down [" + std::to_string(w.mix_down.rows) + "," + std::to_string(w.mix_down.cols) +
                 "] / mix_up [" + std::to_string(w.mix_up.rows) + "," + std::to_string(w.mix_up.cols) +
                 "] are not [rank, streams * hidden] / [streams * hidden, rank]");
    return -1;
  }
  const bool with_inject = w.block_inject != nullptr && inj != nullptr;
  if (with_inject && !aligned16(w.block_inject)) {
    fn_set_error("fn_hc_mix: the block-inject weight must be 16-byte aligned");
    return -1;
  }
  const int32_t width = g.residual_width();
  auto scope = scratch.scope();
  const int32_t wave = std::min(rows, kWaveRows);
  auto *normed = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * width * 2).data);
  auto *up = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * width * 2).data);
  auto *down = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.hc_rank * 2).data);
  auto *act = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.hc_rank * 2).data);
  auto *inject_part = static_cast<float *>(
      scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.streams * g.streams * 4).data);
  auto *down_part = static_cast<float *>(
      scratch.alloc_bytes(static_cast<std::size_t>(g.streams) * std::min(rows, kDecodeRows) * g.hc_rank * 4).data);
  const auto *block_inject = with_inject ? static_cast<const __nv_bfloat16 *>(w.block_inject) : nullptr;

  for (int32_t first = 0; first < rows; first += wave) {
    const int32_t n = std::min(wave, rows - first);
    const auto *in = static_cast<const __nv_bfloat16 *>(hidden) + static_cast<std::int64_t>(first) * width;
    auto *out = static_cast<__nv_bfloat16 *>(x) + static_cast<std::int64_t>(first) * g.hidden;
    float *out_inj = with_inject ? inj + static_cast<std::int64_t>(first) * g.streams : nullptr;
    const bool decode = decode_route(g, w, n);
    const bool fused = decode && n <= kMaxFusedRows && decode_fused();
    if (pending.inj != nullptr && (!fused || n != rows)) {
      fn_set_error("fn_hc_mix: a folded inject outside the fused decode route");
      return -1;
    }
    if (!fused) {
      hc_norm<<<dim3(static_cast<uint32_t>(g.streams), static_cast<uint32_t>(n)), kThreads, 0, stream>>>(
          in, static_cast<const __nv_bfloat16 *>(w.hc_norm), block_inject, normed, inject_part, g.streams, g.hidden,
          g.rms_norm_eps);
      if (!launched("fn_hc_mix: norm")) {
        return -1;
      }
    }
    if (decode) {
      const bool down_fp8 = w.mix_down.format == WeightFormat::Fp8RowScale;
      const bool up_fp8 = w.mix_up.format == WeightFormat::Fp8RowScale;
      const auto down_blocks = static_cast<uint32_t>((g.hc_rank + kWarps - 1) / kWarps);
      const dim3 down_grid(down_blocks, static_cast<uint32_t>(g.streams));
      if (fused) {
        // One more CTA per stream: the normed rows' and the inject partials' writer.
        const dim3 fused_grid(down_blocks + 1, static_cast<uint32_t>(g.streams));
        const std::size_t fused_bytes = static_cast<std::size_t>(n) * g.hidden * sizeof(__nv_bfloat16);
        const auto *norm_w = static_cast<const __nv_bfloat16 *>(w.hc_norm);
        const int kind = pending.inj == nullptr ? kNothing : pending.acc == nullptr ? kInject : kCombine;
        const auto launch = [&](auto kernel) {
          kernel<<<fused_grid, kThreads, fused_bytes, stream>>>(in, norm_w, block_inject, w.mix_down.data, normed,
                                                                inject_part, down_part, g.hc_rank, g.streams, g.hidden,
                                                                n, g.rms_norm_eps, pending);
        };
        if (down_fp8) {
          launch(kind == kNothing ? hc_norm_down<true, kNothing>
                                  : kind == kInject ? hc_norm_down<true, kInject> : hc_norm_down<true, kCombine>);
        } else {
          launch(kind == kNothing ? hc_norm_down<false, kNothing>
                                  : kind == kInject ? hc_norm_down<false, kInject> : hc_norm_down<false, kCombine>);
        }
      } else if (down_fp8) {
        hc_down_split<true><<<down_grid, kThreads, 0, stream>>>(w.mix_down.data, normed, down_part, g.hc_rank,
                                                                 g.streams, g.hidden, n);
      } else {
        hc_down_split<false><<<down_grid, kThreads, 0, stream>>>(w.mix_down.data, normed, down_part, g.hc_rank,
                                                                  g.streams, g.hidden, n);
      }
      if (!launched(fused ? "fn_hc_mix: norm and down" : "fn_hc_mix: down")) {
        return -1;
      }
      const auto grid = static_cast<uint32_t>(g.hidden / (2 * kWarps));
      const __nv_bfloat16 *down_scales = down_fp8 ? fp8_scales(w.mix_down) : nullptr;
      const float *parts = with_inject ? inject_part : nullptr;
      if (up_fp8) {
        hc_up_reduce<true><<<grid, kThreads, 0, stream>>>(w.mix_up.data, fp8_scales(w.mix_up), down_part, down_scales,
                                                           normed, parts, out, out_inj, g.hc_rank, g.hidden, n,
                                                           pending, residual, acc_zero);
      } else {
        hc_up_reduce<false><<<grid, kThreads, 0, stream>>>(w.mix_up.data, nullptr, down_part, down_scales, normed,
                                                            parts, out, out_inj, g.hc_rank, g.hidden, n, pending,
                                                            residual, acc_zero);
      }
      if (!launched("fn_hc_mix: up and reduce")) {
        return -1;
      }
      continue;
    }
    if (fn_linear(w.mix_down, normed, n, down, false, scratch, stream) != 0) {
      return -1;
    }
    const std::int64_t down_count = static_cast<std::int64_t>(n) * g.hc_rank;
    hc_down_activation<<<static_cast<uint32_t>((down_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        down, act, down_count, static_cast<float>(g.streams));
    if (!launched("fn_hc_mix: activation")) {
      return -1;
    }
    if (fn_linear(w.mix_up, act, n, up, false, scratch, stream) != 0) {
      return -1;
    }
    hc_reduce<<<n, kThreads, 0, stream>>>(up, normed, with_inject ? inject_part : nullptr, out, out_inj, g.streams,
                                           g.hidden);
    if (!launched("fn_hc_mix: reduce")) {
      return -1;
    }
  }
  return 0;
}

}  // namespace

int32_t fn_hc_inject(const Geometry &g, const void *y, const float *inj, int32_t rows, void *hidden,
                     cudaStream_t stream) {
  if (y == nullptr || inj == nullptr || hidden == nullptr || rows <= 0) {
    fn_set_error("fn_hc_inject: null operand or no rows");
    return -1;
  }
  const std::int64_t count = static_cast<std::int64_t>(rows) * g.residual_width();
  hc_inject_kernel<<<static_cast<uint32_t>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(y), inj, static_cast<__nv_bfloat16 *>(hidden), g.streams,
      g.hidden, count);
  return launched("fn_hc_inject") ? 0 : -1;
}

}  // namespace ignis::flash_next
