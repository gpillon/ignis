// ignis kernel leaf: the dense prefill of the 27B's trellis body -- OURS (GitHub #509, spec #503;
// kernel/include/ignis_dense_prefill.h). Ported from the prototype's bench on the branch
// prototype/sm8x-dense-prefill (kernel/tests/bench_sm8x_dense.cu, phase 2a's kernels with phase
// 2c's levers A, one quantize launch per shared input, and D, the fragments double-buffered
// across the stage boundary over 3 or 4 stages), written from the PTX ISA and our own kernels.
//
// Four kernels, each its own launch on the caller's stream:
//   dense_prefill_decode_kernel<K2>     a warp turns four trellis tiles (16 outputs x 64 inputs)
//                                       into W8 codes through 1 KiB of shared memory, so a dense
//                                       block of 128 x 64 is written as one contiguous 8 KiB run;
//   dense_prefill_a8_kernel<MAXB>       one CTA per token: the row in registers (in <= 8 x MAXB x
//                                       128), rotated and quantized once per matrix of the group;
//   dense_prefill_a8_wide_kernel        one CTA per token, two passes over the row (in > 6144);
//   dense_prefill_gemm_kernel<BM, NST>  BM tokens x 128 outputs a CTA, 64 inputs a stage, NST
//                                       cp.async stages, ldmatrix from XOR-swizzled 64-byte rows,
//                                       8 warps of 64 x 64 (BM 256) or 64 x 32 (BM 128), int32
//                                       accumulators over the whole K, the epilogue in fp32.
// GitHub #511 fuses A8 into the RMSnorm that produces its input, with one more:
//   dense_prefill_rmsnorm_a8_kernel     one CTA per token: the RMSnorm before a block's first
//                                       matrices (the vendored rmsnorm's arithmetic, bit for bit),
//                                       then each matrix's A8 of the normalized row.
// Down's A8 stays dense_prefill_a8_wide_kernel's standalone pass: fused into gate+up's epilogue
// (each block's max kept there, then a light second pass) it measured slower than that pass on
// the RTX 5090 and on an sm_86 card.
// None of them has a tensor-core form but the s8 x s8 -> s32 MMA (the full-rate rule of spec
// #503); the SASS check of #512 holds them to it.

#include "dense_prefill_internal.cuh"

#include "moe_common.cuh"
#include "trellis_decode.cuh"
#include "trellis_linear_internal.cuh"

// The vendored rmsnorm's device helpers (block_reduce_sum and the offset epilogue), which the fused
// RMSnorm restates; none of its kernels is instantiated here.
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>

namespace ignis_dense {
namespace {

using ignis_tl::check_launch;
using ignis_tl::fail;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlockN = 128;  // outputs per dense block: one output-Hadamard block
constexpr int kBlockK = 64;   // inputs per dense block, and per GEMM stage
constexpr int kBlockBytes = kBlockN * kBlockK;
constexpr int kDecodeWarps = 4;
constexpr int kERow = 132;     // fp32 per row of the epilogue's staging
constexpr int kRowBlocks = 6;  // the register-resident A8's 128-input blocks a warp: in <= 6144
constexpr uint32_t kRowMaxIn = kWarps * kRowBlocks * 128;
// The trellis affine map at s' = 4 q + 510: sw = 4 c1 and off = 1534 c1 + c0 (c1 = fp16 0x1EEE,
// c0 = fp16 0xC931), exact in fp32.
constexpr float kSw = 887.0f / 32768.0f;
constexpr float kOff = -119.0f / 65536.0f;

// Input k of token t in the quantized activations: 64-input slabs, [in / 64][tokens][64].
__host__ __device__ __forceinline__ size_t a_index(int tokens, int t, int k) {
  return (static_cast<size_t>(k >> 6) * tokens + t) * 64 + (k & 63);
}

// ---- decode ------------------------------------------------------------------------------------

// One warp per strip of 16 outputs x 64 inputs (four trellis tiles down K): lane t decodes its
// eight steps of each tile -- inputs 16 i + {r0, r0 + 1, r0 + 8, r0 + 9}, outputs {c0, c0 + 8},
// r0 = 2 (t % 4), c0 = t / 4, the format's tensor-core order -- into the strip in shared memory,
// then the warp stores the strip's 1 KiB. Strips run outputs-fastest, so a CTA's warps read
// neighbouring trellis tiles and write neighbouring KiB.
template <int K2>
__global__ void __launch_bounds__(kDecodeWarps * 32) dense_prefill_decode_kernel(const uint32_t *trellis, int stride,
                                                                                int in, int out, int8_t *w) {
  __shared__ __align__(16) uint8_t strip[kDecodeWarps][16 * kBlockK];
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int kbs = in / kBlockK;
  const long long s = static_cast<long long>(blockIdx.x) * kDecodeWarps + warp;
  if (s >= static_cast<long long>(out / 16) * kbs) return;
  const int sub = static_cast<int>(s % 8);
  const int kb = static_cast<int>((s / 8) % kbs);
  const int nb = static_cast<int>(s / (8LL * kbs));
  uint8_t *sp = strip[warp];
  const int r0 = (lane & 3) * 2;
  const int c0 = lane >> 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t *tile = trellis + (static_cast<size_t>(kb * 4 + i) * stride + nb * 8 + sub) * words;
    uint32_t hi, lo;
    if constexpr (K2 >= 10) {
      const ignis_trellis::LanePlan3 plan = ignis_trellis::lane_plan3(K2, lane);
      ignis_trellis::lane_window3(ignis_tl::load_stream(tile + plan.w0), ignis_tl::load_stream(tile + plan.w1),
                                  ignis_tl::load_stream(tile + plan.w2), plan, hi, lo);
    } else {
      const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
      ignis_trellis::lane_window(ignis_tl::load_stream(tile + plan.w0), ignis_tl::load_stream(tile + plan.w1), plan, hi,
                                 lo);
    }
    uint32_t bits[8];
    ignis_trellis::window_codes<K2>(hi, lo, bits);
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = ignis_trellis::w8_code(bits[j]);
    auto pair = [](int a, int b) { return static_cast<uint16_t>((a & 0xFF) | ((b & 0xFF) << 8)); };
    const int k = 16 * i + r0;
    *reinterpret_cast<uint16_t *>(sp + c0 * kBlockK + k) = pair(q[0], q[1]);
    *reinterpret_cast<uint16_t *>(sp + c0 * kBlockK + k + 8) = pair(q[2], q[3]);
    *reinterpret_cast<uint16_t *>(sp + (c0 + 8) * kBlockK + k) = pair(q[4], q[5]);
    *reinterpret_cast<uint16_t *>(sp + (c0 + 8) * kBlockK + k + 8) = pair(q[6], q[7]);
  }
  __syncwarp();
  uint4 *dst = reinterpret_cast<uint4 *>(w + static_cast<size_t>(nb * kbs + kb) * kBlockBytes + sub * 16 * kBlockK);
  const uint4 *src = reinterpret_cast<const uint4 *>(sp);
  dst[lane] = src[lane];
  dst[lane + 32] = src[lane + 32];
}

// ---- A8 ----------------------------------------------------------------------------------------

struct A8Target {
  const __half *suh;
  int8_t *q;
  float *scale;
  int32_t *sum;
};

struct A8Group {
  A8Target m[IGNIS_DENSE_PREFILL_MAX_GROUP];
  int count;
};

// One 128-block's four entries of a lane, x o suh then the Hadamard. Each product is rounded on
// its own (__fmul_rn: never contracted into the butterfly's first add), so the rotation is a
// fixed sequence of fp32 operations a host reference reproduces bit for bit.
__device__ __forceinline__ void rotate4(const float (&x)[4], const __half *suh, float (&v)[4]) {
  float s[4];
  ignis_tl::load_half4(suh, s);
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = __fmul_rn(x[q], s[q]);
  ignis_moe::warp_hadamard128(v);
}

__device__ __forceinline__ float max4(const float (&v)[4]) {
  return fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
}

// q = clamp(rn(v to_q), +-127) of four entries, their sum added to `sum`, packed little-endian.
__device__ __forceinline__ uint32_t quantize4(const float (&v)[4], float to_q, int &sum) {
  int q[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q[i] = max(-127, min(127, __float2int_rn(v[i] * to_q)));
    sum += q[i];
  }
  return (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (static_cast<uint32_t>(q[3] & 0xFF) << 24);
}

// The token's amax from the CTA's `Warps` warps' maxima (s_max[warp]), and the scale it gives:
// 127 / amax and amax / 127, or 0 and 0 for a row of zeros or a non-finite one. The max does not
// depend on which warp held which block, so every A8 kernel below gives the same scale.
template <int Warps>
__device__ __forceinline__ void token_scale(const float *s_max, float &to_q, float &scale) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < Warps; ++i) amax = fmaxf(amax, s_max[i]);
  const bool finite = amax > 0.0f && amax <= 3.0e38f;
  to_q = finite ? 127.0f / amax : 0.0f;
  scale = finite ? amax * (1.0f / 127.0f) : 0.0f;
}

// The token's sum of q from each lane's `sum` over the CTA's `Warps` warps, written with the scale
// by thread 0 (an integer sum: the same whichever warp held which block).
template <int Warps>
__device__ __forceinline__ void token_sum(int sum, int *s_sum, const A8Target &m, int t, float scale) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  if (lane == 0) s_sum[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int i = 0; i < Warps; ++i) total += s_sum[i];
    m.scale[t] = scale;
    m.sum[t] = total;
  }
}

// One CTA per token: each warp holds blocks warp, warp + 8, ... of the token's row in registers
// (x read once), then per matrix of the group rotates them with that matrix's suh, takes the amax
// over the whole K and writes its INT8 row, scale and sum. One CTA per SM is all it asks for: left
// free, ptxas held the six-block row to 64 registers on sm_120a and spilled.
template <int MAXB>
__global__ void __launch_bounds__(kThreads, 1) dense_prefill_a8_kernel(const __nv_bfloat16 *x, int tokens, int in,
                                                                       const __grid_constant__ A8Group g) {
  __shared__ float s_max[IGNIS_DENSE_PREFILL_MAX_GROUP][kWarps];
  __shared__ int s_sum[IGNIS_DENSE_PREFILL_MAX_GROUP][kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = in / 128;
  const int t = blockIdx.x;
  float xv[MAXB][4];
#pragma unroll
  for (int j = 0; j < MAXB; ++j) {
    const int blk = warp + kWarps * j;
    if (blk < nb) ignis_tl::load_bf16x4(x + static_cast<size_t>(t) * in + blk * 128 + 4 * lane, xv[j]);
  }
#pragma unroll
  for (int mi = 0; mi < IGNIS_DENSE_PREFILL_MAX_GROUP; ++mi) {
    if (mi >= g.count) break;
    const A8Target &m = g.m[mi];
    float v[MAXB][4];
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int blk = warp + kWarps * j;
      if (blk < nb) {
        rotate4(xv[j], m.suh + blk * 128 + 4 * lane, v[j]);
        mx = fmaxf(mx, max4(v[j]));
      }
    }
    mx = ignis_moe::warp_max(mx);
    if (lane == 0) s_max[mi][warp] = mx;
    __syncthreads();
    float to_q, scale;
    token_scale<kWarps>(s_max[mi], to_q, scale);
    int sum = 0;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int blk = warp + kWarps * j;
      if (blk < nb) {
        *reinterpret_cast<uint32_t *>(m.q + a_index(tokens, t, blk * 128 + 4 * lane)) = quantize4(v[j], to_q, sum);
      }
    }
    token_sum<kWarps>(sum, s_sum[mi], m, t, scale);
  }
}

// One CTA per token of a row too wide for registers: the first pass takes the amax, the second
// rotates the row again (x from the L2) and writes it.
__global__ void __launch_bounds__(kThreads) dense_prefill_a8_wide_kernel(const __nv_bfloat16 *x, int tokens, int in,
                                                                         const __grid_constant__ A8Target m) {
  __shared__ float s_max[kWarps];
  __shared__ int s_sum[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = in / 128;
  const int t = blockIdx.x;
  auto rotated = [&](int blk, float (&v)[4]) {
    float xv[4];
    ignis_tl::load_bf16x4(x + static_cast<size_t>(t) * in + blk * 128 + 4 * lane, xv);
    rotate4(xv, m.suh + blk * 128 + 4 * lane, v);
  };
  float mx = 0.0f;
  for (int blk = warp; blk < nb; blk += kWarps) {
    float v[4];
    rotated(blk, v);
    mx = fmaxf(mx, max4(v));
  }
  mx = ignis_moe::warp_max(mx);
  if (lane == 0) s_max[warp] = mx;
  __syncthreads();
  float to_q, scale;
  token_scale<kWarps>(s_max, to_q, scale);
  int sum = 0;
  for (int blk = warp; blk < nb; blk += kWarps) {
    float v[4];
    rotated(blk, v);
    *reinterpret_cast<uint32_t *>(m.q + a_index(tokens, t, blk * 128 + 4 * lane)) = quantize4(v, to_q, sum);
  }
  token_sum<kWarps>(sum, s_sum, m, t, scale);
}

// ---- A8 fused into its producers (GitHub #511) -------------------------------------------------

// The RMSnorm before a block's first matrices, with their A8: one CTA of 512 threads per token.
// Its first half is the vendored rmsnorm's wide-row form restated operation for operation
// (kernel/vendor/src/ops/kernel/rmsnorm.cuh, rmsnorm_cta_bf16x2_kernel<Offset, 512, 8, true>, what
// ninfer's launcher runs for a unit-offset row of 3072 < in <= 8192, in % 1024 == 0): the same
// pairs per thread in the same order, the same sum of squares, block_reduce_sum, rsqrtf and offset
// epilogue, so its BF16 row is that kernel's bit for bit. The row stays in shared memory (and goes
// to h when asked); the second half is dense_prefill_a8_kernel's per matrix of the group, from the
// row: each 128-block through rotate4, the amax over the row, quantize4 and the sum. A block's
// values do not depend on which warp holds it, nor the max and the integer sum on the order, so
// the activations are the standalone A8's of h bit for bit.
constexpr int kNormThreads = 512;
constexpr int kNormWarps = kNormThreads / 32;
constexpr int kNormMaxPairs = 8;     // the vendored form's pairs per thread
constexpr int kNormRowBlocks = 4;    // the 128-blocks a warp holds: 16 x 4 x 128 = 8192 inputs
constexpr uint32_t kNormMaxIn = 2 * kNormThreads * kNormMaxPairs;
static_assert(kNormWarps * kNormRowBlocks * 128 == static_cast<int>(kNormMaxIn), "every block of the widest row has a warp");

// The widths the fused RMSnorm takes: the vendored launcher's wide-row form's.
bool norm_width(uint32_t in) { return in > 3072 && in <= kNormMaxIn && in % 1024 == 0; }

__global__ void __launch_bounds__(kNormThreads, 2)
    dense_prefill_rmsnorm_a8_kernel(const __nv_bfloat162 *x, const __nv_bfloat162 *norm, int d, float eps,
                                    __nv_bfloat162 *h, int tokens, const __grid_constant__ A8Group g) {
  __shared__ __align__(16) __nv_bfloat162 row[kNormMaxIn / 2];
  __shared__ float warp_sums[kNormWarps];
  __shared__ float inv_shared;
  __shared__ float s_max[IGNIS_DENSE_PREFILL_MAX_GROUP][kNormWarps];
  __shared__ int s_sum[IGNIS_DENSE_PREFILL_MAX_GROUP][kNormWarps];
  const int t = blockIdx.x;
  // The vendored form, its x and weight loads, sum of squares and epilogue, with the row kept.
  const int pairs = d / 2;
  const int pairs_per_thread = pairs / kNormThreads;
  const std::int64_t row_base = static_cast<std::int64_t>(t) * pairs;
  __nv_bfloat162 values[kNormMaxPairs];
  __nv_bfloat162 weights[kNormMaxPairs];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < kNormMaxPairs; ++k) {
    if (k < pairs_per_thread) {
      const int pair = static_cast<int>(threadIdx.x) + k * kNormThreads;
      values[k] = x[row_base + pair];
      weights[k] = norm[pair];
      const float2 xf = __bfloat1622float2(values[k]);
      sum += xf.x * xf.x + xf.y * xf.y;
    }
  }
  const float block_sum = ninfer::ops::block_reduce_sum<kNormThreads>(sum, warp_sums);
  if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
  __syncthreads();
  const float inv = inv_shared;
#pragma unroll
  for (int k = 0; k < kNormMaxPairs; ++k) {
    if (k < pairs_per_thread) {
      const int pair = static_cast<int>(threadIdx.x) + k * kNormThreads;
      const float2 xf = __bfloat1622float2(values[k]);
      const float2 wf = __bfloat1622float2(weights[k]);
      const __nv_bfloat162 r =
          __floats2bfloat162_rn(ninfer::ops::rmsnorm_epilogue<ninfer::ops::RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0f),
                                ninfer::ops::rmsnorm_epilogue<ninfer::ops::RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0f));
      row[pair] = r;
      if (h != nullptr) h[row_base + pair] = r;
    }
  }
  __syncthreads();
  // dense_prefill_a8_kernel's A8 per matrix, warp w holding blocks w, w + 16, ...
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = d / 128;
  const __nv_bfloat16 *rb = reinterpret_cast<const __nv_bfloat16 *>(row);
#pragma unroll
  for (int mi = 0; mi < IGNIS_DENSE_PREFILL_MAX_GROUP; ++mi) {
    if (mi >= g.count) break;
    const A8Target &m = g.m[mi];
    float v[kNormRowBlocks][4];
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < kNormRowBlocks; ++j) {
      const int blk = warp + kNormWarps * j;
      if (blk < nb) {
        float xv[4];
        ignis_tl::load_bf16x4(rb + blk * 128 + 4 * lane, xv);
        rotate4(xv, m.suh + blk * 128 + 4 * lane, v[j]);
        mx = fmaxf(mx, max4(v[j]));
      }
    }
    mx = ignis_moe::warp_max(mx);
    if (lane == 0) s_max[mi][warp] = mx;
    __syncthreads();
    float to_q, scale;
    token_scale<kNormWarps>(s_max[mi], to_q, scale);
    int qsum = 0;
#pragma unroll
    for (int j = 0; j < kNormRowBlocks; ++j) {
      const int blk = warp + kNormWarps * j;
      if (blk < nb) {
        *reinterpret_cast<uint32_t *>(m.q + a_index(tokens, t, blk * 128 + 4 * lane)) = quantize4(v[j], to_q, qsum);
      }
    }
    token_sum<kNormWarps>(qsum, s_sum[mi], m, t, scale);
  }
}

// ---- GEMM --------------------------------------------------------------------------------------

struct GemmParams {
  const int8_t *a;      // [in / 64][tokens][64]
  const int8_t *w;      // the dense copy, blocked
  const float *scale;   // [tokens]
  const int32_t *sum;   // [tokens]
  const __half *svh;    // [out]
  int tokens;
  int in;
  int out;
  int mt;       // token tiles
  int nt;       // output tiles
  int raster;   // kTokensFastest or kOutputsFastest
  int epilogue; // ignis_tl::kStore, kAdd or kSwiGluUp
  __nv_bfloat16 *y;
  __nv_bfloat16 *y2;  // the split's second half, else unused
  int ldy;
  int split;          // 0, or the split's half width
  int32_t *acc;       // non-null: the int32 accumulators [tokens][ldacc] instead of y
  int ldacc;
};

template <int BM> struct Tile {
  static constexpr int kWarpsM = BM == 256 ? 4 : 2;
  static constexpr int kWarpsN = kWarps / kWarpsM;
  static constexpr int kWM = BM / kWarpsM;     // 64
  static constexpr int kWN = kBlockN / kWarpsN;  // 64 or 32
  static constexpr int kMI = kWM / 16;
  static constexpr int kNI = kWN / 8;
  static constexpr int kAStage = BM * kBlockK;
  static constexpr int kBStage = kBlockN * kBlockK;
  static constexpr int kStage = kAStage + kBStage;
};

template <int BM, int NST> __host__ __device__ constexpr int gemm_smem() { return Tile<BM>::kStage * NST; }
// sm_86's opt-in limit per block is 99 KiB; the epilogue's staging overlays the stages.
static_assert(gemm_smem<256, 4>() <= 99 * 1024, "the deepest GEMM must fit 99 KiB of shared memory per block");
static_assert(Tile<256>::kWarpsM * 16 * kERow * 4 <= gemm_smem<128, 3>(), "the epilogue's staging overlays the stages");

// 16-byte chunk c (0..3) of row r of a 64-byte-row stage: XOR-swizzled with (r / 2) % 4, so the
// eight rows an ldmatrix phase reads at one chunk land on eight different 4-bank groups.
__device__ __forceinline__ uint32_t swz(int r, int c) { return static_cast<uint32_t>(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

__device__ __forceinline__ void cp_async16(uint32_t smem, const void *gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}

__device__ __forceinline__ void cp_async16_zfill(uint32_t smem, const void *gmem, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}

// D = A B + D: A 16 x 32 s8 (row), B 32 x 8 s8 (col), int32, exact.
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Stage kt (inputs 64 kt ..) of the tile: BM token rows of the activations (rows past the tokens
// zero-filled) and the matrix's 128 x 64 block.
template <int BM>
__device__ __forceinline__ void load_stage(const GemmParams &p, uint32_t st, int kt, int row0, const int8_t *wcol) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < BM * 4 / kThreads; ++i) {
    const int idx = tid + i * kThreads;
    const int r = idx >> 2;
    const int c = idx & 3;
    const int t = row0 + r;
    const bool valid = t < p.tokens;
    cp_async16_zfill(st + swz(r, c), p.a + a_index(p.tokens, valid ? t : 0, kt * kBlockK + 16 * c), valid);
  }
  const int8_t *blk = wcol + static_cast<size_t>(kt) * kBlockBytes;
#pragma unroll
  for (int i = 0; i < kBlockN * 4 / kThreads; ++i) {
    const int idx = tid + i * kThreads;
    const int r = idx >> 2;
    const int c = idx & 3;
    cp_async16(st + Tile<BM>::kAStage + swz(r, c), blk + r * 64 + 16 * c);
  }
}

// The fragments of k32 step kk (0, 1) of a stage: A for the warp's kMI m16 blocks, B for its kNI
// n8 blocks two at a time (b[j][0..1] block 2j, b[j][2..3] block 2j + 1).
template <int BM>
__device__ __forceinline__ void load_frags(uint32_t st, int kk, int wm, int wn, uint32_t (&a)[Tile<BM>::kMI][4],
                                           uint32_t (&b)[Tile<BM>::kNI / 2][4]) {
  using C = Tile<BM>;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < C::kMI; ++i) ldsm_x4(a[i], st + swz(wm * C::kWM + 16 * i + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
  for (int j = 0; j < C::kNI / 2; ++j) {
    ldsm_x4(b[j], st + C::kAStage + swz(wn * C::kWN + 16 * j + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1)));
  }
}

template <int BM>
__device__ __forceinline__ void mma_frags(const uint32_t (&a)[Tile<BM>::kMI][4], const uint32_t (&b)[Tile<BM>::kNI / 2][4],
                                          int32_t (&acc)[Tile<BM>::kMI][Tile<BM>::kNI][4]) {
  using C = Tile<BM>;
#pragma unroll
  for (int i = 0; i < C::kMI; ++i)
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) mma_s8(acc[i][n], a[i], b[n >> 1][2 * (n & 1)], b[n >> 1][2 * (n & 1) + 1]);
}

// The main loop over the tile's in / 64 stages, its matrix block column at wcol. Step i loads its
// second k32 step's fragments, issues stage i + NST - 1 into the buffer step i - 1 read (which
// step i - 1's barrier freed), runs the first k32 step's MMAs, then waits for stage i + 1 and
// syncs, loads its first fragments, and runs the second k32 step's MMAs on fragments that landed
// long before: one barrier a stage, with the barrier and the ldmatrix latency under MMAs.
template <int BM, int NST>
__device__ __forceinline__ void mainloop(const GemmParams &p, uint32_t base, const int8_t *wcol, int row0, int wm, int wn,
                                         int32_t (&acc)[Tile<BM>::kMI][Tile<BM>::kNI][4]) {
  using C = Tile<BM>;
  const int n = p.in / kBlockK;
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < n) load_stage<BM>(p, base + s * C::kStage, s, row0, wcol);
    ignis_moe::cp_async_commit();
  }
  uint32_t a0[C::kMI][4], b0[C::kNI / 2][4], a1[C::kMI][4], b1[C::kNI / 2][4];
  ignis_moe::cp_async_wait<NST - 2>();
  __syncthreads();
  load_frags<BM>(base, 0, wm, wn, a0, b0);
  // The stage read and the stage written rotate through the ring by offset, no modulo; the loop
  // stays rolled.
  const uint32_t last = base + (NST - 1) * C::kStage;
  uint32_t rd = base, wr = last;
#pragma unroll 1
  for (int kt = 0; kt < n; ++kt) {
    load_frags<BM>(rd, 1, wm, wn, a1, b1);
    const int pf = kt + NST - 1;
    if (pf < n) load_stage<BM>(p, wr, pf, row0, wcol);
    ignis_moe::cp_async_commit();
    wr = wr == last ? base : wr + C::kStage;
    mma_frags<BM>(a0, b0, acc);
    rd = rd == last ? base : rd + C::kStage;
    if (kt + 1 < n) {
      ignis_moe::cp_async_wait<NST - 2>();
      __syncthreads();
      load_frags<BM>(rd, 0, wm, wn, a0, b0);
    }
    mma_frags<BM>(a1, b1, acc);
  }
  ignis_moe::cp_async_wait<0>();
}

// The tile's epilogue from its full-K accumulators: with p.acc the int32 accumulators, else one
// m16 block of every warp row at a time through shared memory -- y = acc scale sw + off scale sum
// per token in fp32 -- then each 128-column row back through the Hadamard, svh, and BF16 out as
// the fused prefill's epilogue writes it.
template <int BM>
__device__ __forceinline__ void epilogue(const GemmParams &p, uint8_t *smem, int row0, int tn, int wm, int wn,
                                         const int32_t (&acc)[Tile<BM>::kMI][Tile<BM>::kNI][4]) {
  using C = Tile<BM>;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c4 = lane & 3;
  if (p.acc != nullptr) {
#pragma unroll
    for (int i = 0; i < C::kMI; ++i) {
      const int t = row0 + wm * C::kWM + 16 * i + g;
#pragma unroll
      for (int n = 0; n < C::kNI; ++n) {
        const int col = tn * kBlockN + wn * C::kWN + 8 * n + 2 * c4;
        if (t < p.tokens) *reinterpret_cast<int2 *>(p.acc + static_cast<size_t>(t) * p.ldacc + col) = make_int2(acc[i][n][0], acc[i][n][1]);
        if (t + 8 < p.tokens) *reinterpret_cast<int2 *>(p.acc + static_cast<size_t>(t + 8) * p.ldacc + col) = make_int2(acc[i][n][2], acc[i][n][3]);
      }
    }
    return;
  }
  // Where the tile's 128 columns land: y at column tn 128, or under the split the head's half.
  __nv_bfloat16 *dst = p.y;
  int dcol = tn * kBlockN;
  if (p.split != 0) {
    const int head = dcol / (2 * p.split);
    const int within = dcol % (2 * p.split);
    dst = within < p.split ? p.y : p.y2;
    dcol = head * p.split + (within < p.split ? within : within - p.split);
  }
  float *e = reinterpret_cast<float *>(smem);
#pragma unroll
  for (int i = 0; i < C::kMI; ++i) {
    __syncthreads();
    const int t0 = row0 + wm * C::kWM + 16 * i + g;
    float f1[2] = {0.0f, 0.0f}, f0[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = t0 + 8 * h;
      if (t < p.tokens) {
        const float s = p.scale[t];
        f1[h] = s * kSw;
        f0[h] = s * kOff * static_cast<float>(p.sum[t]);
      }
    }
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) {
      const int col = wn * C::kWN + 8 * n + 2 * c4;
      const int er = wm * 16 + g;
      *reinterpret_cast<float2 *>(e + er * kERow + col) =
          make_float2(fmaf(static_cast<float>(acc[i][n][0]), f1[0], f0[0]), fmaf(static_cast<float>(acc[i][n][1]), f1[0], f0[0]));
      *reinterpret_cast<float2 *>(e + (er + 8) * kERow + col) =
          make_float2(fmaf(static_cast<float>(acc[i][n][2]), f1[1], f0[1]), fmaf(static_cast<float>(acc[i][n][3]), f1[1], f0[1]));
    }
    __syncthreads();
    for (int rr = warp; rr < C::kWarpsM * 16; rr += kWarps) {
      const int t = row0 + (rr >> 4) * C::kWM + 16 * i + (rr & 15);
      if (t >= p.tokens) continue;
      const float4 q = *reinterpret_cast<const float4 *>(e + rr * kERow + 4 * lane);
      float v[4] = {q.x, q.y, q.z, q.w};
      ignis_moe::warp_hadamard128(v);
      float s[4];
      ignis_tl::load_half4(p.svh + tn * kBlockN + 4 * lane, s);
      __nv_bfloat16 *y = dst + static_cast<size_t>(t) * p.ldy + dcol + 4 * lane;
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] *= s[j];
      if (p.epilogue == ignis_tl::kSwiGluUp) {
        float gate[4];
        ignis_tl::load_bf16x4(y, gate);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = ignis_moe::silu(gate[j]) * v[j];
      }
      ignis_tl::store_bf16x4(y, v, p.epilogue == ignis_tl::kAdd);
    }
  }
}

// One CTA per output tile, in the call's raster order.
template <int BM, int NST>
__global__ void __launch_bounds__(kThreads, 1) dense_prefill_gemm_kernel(const __grid_constant__ GemmParams p) {
  using C = Tile<BM>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int tile = static_cast<int>(blockIdx.x);
  const int tm = p.raster == kTokensFastest ? tile % p.mt : tile / p.nt;
  const int tn = p.raster == kTokensFastest ? tile / p.mt : tile % p.nt;
  const int row0 = tm * BM;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / C::kWarpsN;
  const int wn = warp % C::kWarpsN;
  int32_t acc[C::kMI][C::kNI][4];
#pragma unroll
  for (int i = 0; i < C::kMI; ++i)
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0;
  const int8_t *wcol = p.w + static_cast<size_t>(tn) * (p.in / kBlockK) * kBlockBytes;
  mainloop<BM, NST>(p, base, wcol, row0, wm, wn, acc);
  epilogue<BM>(p, smem, row0, tn, wm, wn, acc);
}

// ---- host --------------------------------------------------------------------------------------

constexpr int kMaxDevices = 64;
std::mutex g_mutex;
State g_state[kMaxDevices];

bool aligned(const void *p, uintptr_t a) { return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

bool valid_k2(uint32_t k2) { return k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12; }

bool valid_shape(uint32_t in, uint32_t out) { return in != 0 && out != 0 && in % 128 == 0 && out % 128 == 0; }

int32_t check_weight(const char *op, const ignis_trellis_weight *w) {
  const std::string who = std::string(op) + ": weight";
  if (w == nullptr) return fail(who + " is NULL");
  if (!valid_k2(w->k2)) return fail(who + ": k2 must be 4, 5, 6, 8, 10 or 12");
  if (!valid_shape(w->in, w->out)) return fail(who + ": in and out must be positive multiples of 128");
  const uint32_t stride = w->tile_stride == 0 ? w->out / 16 : w->tile_stride;
  if (stride < w->out / 16) return fail(who + ": tile_stride is narrower than out / 16");
  if (!aligned(w->trellis, 16) || !aligned(w->suh, 8) || !aligned(w->svh, 8)) {
    return fail(who + ": trellis must be 16-byte aligned, suh and svh 8-byte aligned");
  }
  return 0;
}

int32_t check_activations(const char *op, const ignis_dense_prefill_activations *a) {
  if (a == nullptr || !aligned(a->q, 16) || !aligned(a->scale, 4) || !aligned(a->sum, 4)) {
    return fail(std::string(op) + ": activations: q must be 16-byte aligned, scale and sum 4-byte aligned");
  }
  return 0;
}

int32_t check_workspace(const char *op, const ignis_trellis_workspace *workspace, uint64_t need) {
  if (workspace == nullptr || !aligned(workspace->base, 256) || need == 0 || workspace->bytes < need) {
    return fail(std::string(op) + ": the workspace must be 256-byte aligned and hold " + std::to_string(need) +
                " bytes, got " + std::to_string(workspace == nullptr ? 0 : workspace->bytes));
  }
  return 0;
}

template <int BM, int NST> int32_t prepare_one(State &st) {
  cudaError_t err = cudaFuncSetAttribute(dense_prefill_gemm_kernel<BM, NST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         gemm_smem<BM, NST>());
  if (err != cudaSuccess) return fail(std::string("ignis_dense_prefill_prepare: GEMM attributes: ") + cudaGetErrorString(err));
  int n = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, dense_prefill_gemm_kernel<BM, NST>, kThreads, gemm_smem<BM, NST>());
  if (err != cudaSuccess || n < 1) {
    return fail("ignis_dense_prefill_prepare: the " + std::to_string(BM) + "-token GEMM at " + std::to_string(NST) +
                " stages does not fit an SM");
  }
  st.resident[BM == 128 ? 1 : 0][NST == 4 ? 1 : 0] = n;
  return 0;
}

template <int BM, int NST> cudaError_t launch_gemm(const GemmParams &p, cudaStream_t s) {
  dense_prefill_gemm_kernel<BM, NST><<<p.mt * p.nt, kThreads, gemm_smem<BM, NST>(), s>>>(p);
  return cudaGetLastError();
}

template <int K2> void launch_decode_k2(const ignis_trellis_weight &w, int8_t *dense, cudaStream_t s) {
  const int strips = static_cast<int>(w.out / 16) * static_cast<int>(w.in / kBlockK);
  const int stride = static_cast<int>(w.tile_stride == 0 ? w.out / 16 : w.tile_stride);
  dense_prefill_decode_kernel<K2><<<(strips + kDecodeWarps - 1) / kDecodeWarps, kDecodeWarps * 32, 0, s>>>(
      static_cast<const uint32_t *>(w.trellis), stride, static_cast<int>(w.in), static_cast<int>(w.out), dense);
}

// The quantized activations of matrix i of a dense_linear call in the workspace: past the fused
// op's counters (ignis_tl::layout's arrivals, which stay zero), matrix after matrix.
struct Carve {
  uint64_t q, scale, sum, bytes;
};

Carve carve(uint32_t in, uint32_t tokens) {
  Carve c{};
  c.q = 0;
  c.scale = ignis_tl::align256(static_cast<uint64_t>(tokens) * in);
  c.sum = c.scale + ignis_tl::align256(4ull * tokens);
  c.bytes = c.sum + ignis_tl::align256(4ull * tokens);
  return c;
}

uint64_t activations_offset() { return ignis_tl::align256(256 + 4ull * ignis_tl::kArrivalBlocks); }

// The activations of a group of `count` matrices of `in` inputs in the workspace, matrix i at its
// carve.
void group_activations(const ignis_trellis_workspace &ws, uint32_t in, uint32_t count, uint32_t tokens,
                       ignis_dense_prefill_activations *a) {
  const Carve c = carve(in, tokens);
  char *base = static_cast<char *>(ws.base) + activations_offset();
  for (uint32_t i = 0; i < count; ++i) {
    char *m = base + i * c.bytes;
    a[i] = ignis_dense_prefill_activations{m + c.q, reinterpret_cast<float *>(m + c.scale),
                                           reinterpret_cast<int32_t *>(m + c.sum)};
  }
}

// Each matrix of a group in turn, on activations already quantized: its decode into `dense`, then
// its GEMM into out[i].
int32_t decode_and_gemm(const ignis_trellis_weight *const *w, uint32_t count, const ignis_dense_prefill_activations *a,
                        uint32_t tokens, const ignis_dense_prefill_output *out, void *dense, uint64_t dense_bytes,
                        void *stream) {
  for (uint32_t i = 0; i < count; ++i) {
    if (ignis_dense_prefill_decode(w[i], dense, dense_bytes, stream) != 0) return -1;
    if (ignis_dense_prefill_gemm(w[i], dense, &a[i], tokens, &out[i], stream) != 0) return -1;
  }
  return 0;
}

// The test-only switch of GitHub #511 (ignis_dense_prefill_set_fused_producers), and the fused
// RMSnorm launches enqueued, which a test of the switch reads.
std::atomic<uint32_t> g_fused_producers{1};
std::atomic<uint64_t> g_fused_norm_launches{0};

}  // namespace

int32_t state(const char *op, State *out) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  std::lock_guard<std::mutex> lock(g_mutex);
  if (device < 0 || device >= kMaxDevices || g_state[device].sms == 0) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_dense_prefill_prepare at load");
  }
  *out = g_state[device];
  return 0;
}

Plan plan(const State &st, uint32_t tokens, uint32_t in, uint32_t out) {
  const long long nt = out / kBlockN;
  // The GEMM's time at a tile height, in units of a token row's work: its waves of resident CTAs
  // times the tile's rows.
  const auto cost = [&](int bm, int stages) {
    const long long tiles = (static_cast<long long>(tokens) + bm - 1) / bm * nt;
    const long long slots = static_cast<long long>(st.resident[bm == 128 ? 1 : 0][stages == 4 ? 1 : 0]) * st.sms;
    return (tiles + slots - 1) / slots * bm;
  };
  Plan p;
  const bool long_k = in >= 8192;
  p.bm = cost(128, 4) < cost(256, long_k ? 4 : 3) ? 128 : 256;
  p.stages = p.bm == 128 || long_k ? 4 : 3;
  p.raster = tokens > out ? kOutputsFastest : kTokensFastest;
  return p;
}

int32_t gemm_at(const char *op, const Plan &pl, const ignis_trellis_weight *w, const void *dense,
                const ignis_dense_prefill_activations *a, uint32_t tokens, const ignis_dense_prefill_output *out,
                int32_t *acc, uint32_t ldacc, void *stream) {
  if (check_weight(op, w) != 0 || check_activations(op, a) != 0) return -1;
  if (!aligned(dense, 16)) return fail(std::string(op) + ": the dense copy must be 16-byte aligned");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be positive");
  if ((pl.bm != 256 && pl.bm != 128) || (pl.stages != 3 && pl.stages != 4) ||
      (pl.raster != kTokensFastest && pl.raster != kOutputsFastest)) {
    return fail(std::string(op) + ": no GEMM at that tile plan");
  }
  GemmParams p{};
  p.a = static_cast<const int8_t *>(a->q);
  p.w = static_cast<const int8_t *>(dense);
  p.scale = a->scale;
  p.sum = a->sum;
  p.svh = static_cast<const __half *>(w->svh);
  p.tokens = static_cast<int>(tokens);
  p.in = static_cast<int>(w->in);
  p.out = static_cast<int>(w->out);
  p.mt = static_cast<int>((tokens + pl.bm - 1) / pl.bm);
  p.nt = static_cast<int>(w->out / kBlockN);
  p.raster = pl.raster;
  if (acc != nullptr) {
    if (!aligned(acc, 8) || ldacc < w->out || ldacc % 2 != 0) {
      return fail(std::string(op) + ": acc must be 8-byte aligned and ldacc even, at least out");
    }
    p.acc = acc;
    p.ldacc = static_cast<int>(ldacc);
  } else {
    if (out == nullptr) return fail(std::string(op) + ": no output");
    if (out->epilogue != IGNIS_TRELLIS_STORE && out->epilogue != IGNIS_TRELLIS_ADD && out->epilogue != IGNIS_DENSE_PREFILL_SWIGLU_UP) {
      return fail(std::string(op) + ": unknown epilogue");
    }
    const uint32_t width = out->split == 0 ? w->out : w->out / 2;
    if (!aligned(out->y, 8) || out->ldy % 4 != 0 || out->ldy < width) {
      return fail(std::string(op) + ": y must be 8-byte aligned and ldy a multiple of 4, at least the columns it holds");
    }
    if (out->split != 0 && (out->split % 128 != 0 || w->out % (2 * out->split) != 0 || !aligned(out->y2, 8))) {
      return fail(std::string(op) + ": a split is a multiple of 128 dividing out / 2, with y2 8-byte aligned");
    }
    p.epilogue = out->epilogue == IGNIS_TRELLIS_ADD ? ignis_tl::kAdd
                 : out->epilogue == IGNIS_DENSE_PREFILL_SWIGLU_UP ? ignis_tl::kSwiGluUp
                                                            : ignis_tl::kStore;
    p.y = static_cast<__nv_bfloat16 *>(out->y);
    p.y2 = static_cast<__nv_bfloat16 *>(out->y2);
    p.ldy = static_cast<int>(out->ldy);
    p.split = static_cast<int>(out->split);
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err;
  if (pl.bm == 256) {
    err = pl.stages == 4 ? launch_gemm<256, 4>(p, s) : launch_gemm<256, 3>(p, s);
  } else {
    err = pl.stages == 4 ? launch_gemm<128, 4>(p, s) : launch_gemm<128, 3>(p, s);
  }
  if (err != cudaSuccess) return fail(std::string(op) + ": launch failed: " + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_dense

using namespace ignis_dense;

extern "C" int32_t ignis_dense_prefill_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string("ignis_dense_prefill_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_dense_prefill_prepare: device index out of range");
  State st;
  err = cudaDeviceGetAttribute(&st.sms, cudaDevAttrMultiProcessorCount, device);
  if (err != cudaSuccess || st.sms <= 0) return fail("ignis_dense_prefill_prepare: no multiprocessor count");
  if (prepare_one<256, 3>(st) != 0 || prepare_one<256, 4>(st) != 0 || prepare_one<128, 3>(st) != 0 ||
      prepare_one<128, 4>(st) != 0) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[device] = st;
  return 0;
}

extern "C" uint64_t ignis_dense_prefill_weight_bytes(uint32_t in, uint32_t out) {
  return valid_shape(in, out) ? static_cast<uint64_t>(in) * out : 0;
}

extern "C" uint64_t ignis_dense_prefill_workspace_bytes(uint32_t in, uint32_t matrices, uint32_t tokens) {
  if (in == 0 || in % 128 != 0 || matrices < 1 || matrices > IGNIS_DENSE_PREFILL_MAX_GROUP || tokens == 0) return 0;
  return activations_offset() + matrices * carve(in, tokens).bytes;
}

extern "C" int32_t ignis_dense_prefill_decode(const ignis_trellis_weight *w, void *dense, uint64_t dense_bytes,
                                              void *stream) {
  const char *op = "ignis_dense_prefill_decode";
  if (check_weight(op, w) != 0) return -1;
  if (!aligned(dense, 16) || dense_bytes < static_cast<uint64_t>(w->in) * w->out) {
    return fail(std::string(op) + ": the dense copy must be 16-byte aligned and hold in * out = " +
                std::to_string(static_cast<uint64_t>(w->in) * w->out) + " bytes, got " + std::to_string(dense_bytes));
  }
  State st;
  if (state(op, &st) != 0) return -1;
  int8_t *d = static_cast<int8_t *>(dense);
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (w->k2) {
  case 4: launch_decode_k2<4>(*w, d, s); break;
  case 5: launch_decode_k2<5>(*w, d, s); break;
  case 6: launch_decode_k2<6>(*w, d, s); break;
  case 8: launch_decode_k2<8>(*w, d, s); break;
  case 10: launch_decode_k2<10>(*w, d, s); break;
  default: launch_decode_k2<12>(*w, d, s); break;
  }
  return check_launch(op);
}

extern "C" int32_t ignis_dense_prefill_quantize(const void *x, uint32_t tokens, const ignis_trellis_weight *const *w,
                                                uint32_t count, const ignis_dense_prefill_activations *out,
                                                void *stream) {
  const char *op = "ignis_dense_prefill_quantize";
  if (w == nullptr || out == nullptr || count < 1 || count > IGNIS_DENSE_PREFILL_MAX_GROUP) {
    return fail(std::string(op) + ": 1 to " + std::to_string(IGNIS_DENSE_PREFILL_MAX_GROUP) + " matrices");
  }
  if (x == nullptr || !aligned(x, 16)) return fail(std::string(op) + ": x must be 16-byte aligned");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be positive");
  for (uint32_t i = 0; i < count; ++i) {
    if (check_weight(op, w[i]) != 0 || check_activations(op, &out[i]) != 0) return -1;
    if (w[i]->in != w[0]->in) return fail(std::string(op) + ": the matrices read inputs of different widths");
  }
  State st;
  if (state(op, &st) != 0) return -1;
  const int in = static_cast<int>(w[0]->in);
  const __nv_bfloat16 *xb = static_cast<const __nv_bfloat16 *>(x);
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  auto target = [&](uint32_t i) {
    return A8Target{static_cast<const __half *>(w[i]->suh), static_cast<int8_t *>(out[i].q), out[i].scale, out[i].sum};
  };
  if (w[0]->in <= kRowMaxIn) {
    A8Group g{};
    g.count = static_cast<int>(count);
    for (uint32_t i = 0; i < count; ++i) g.m[i] = target(i);
    if (w[0]->in <= kWarps * 5 * 128) {
      dense_prefill_a8_kernel<5><<<tokens, kThreads, 0, s>>>(xb, static_cast<int>(tokens), in, g);
    } else {
      dense_prefill_a8_kernel<kRowBlocks><<<tokens, kThreads, 0, s>>>(xb, static_cast<int>(tokens), in, g);
    }
    return check_launch(op);
  }
  for (uint32_t i = 0; i < count; ++i) {
    dense_prefill_a8_wide_kernel<<<tokens, kThreads, 0, s>>>(xb, static_cast<int>(tokens), in, target(i));
    if (check_launch(op) != 0) return -1;
  }
  return 0;
}

extern "C" int32_t ignis_dense_prefill_gemm(const ignis_trellis_weight *w, const void *dense,
                                            const ignis_dense_prefill_activations *a, uint32_t tokens,
                                            const ignis_dense_prefill_output *out, void *stream) {
  const char *op = "ignis_dense_prefill_gemm";
  State st;
  if (check_weight(op, w) != 0 || state(op, &st) != 0) return -1;
  return gemm_at(op, plan(st, tokens, w->in, w->out), w, dense, a, tokens, out, nullptr, 0, stream);
}

extern "C" int32_t ignis_dense_prefill_accumulate(const ignis_trellis_weight *w, const void *dense,
                                                  const ignis_dense_prefill_activations *a, uint32_t tokens,
                                                  int32_t *acc, uint32_t ldacc, void *stream) {
  const char *op = "ignis_dense_prefill_accumulate";
  State st;
  if (check_weight(op, w) != 0 || state(op, &st) != 0) return -1;
  if (acc == nullptr) return fail(std::string(op) + ": NULL accumulators");
  return gemm_at(op, plan(st, tokens, w->in, w->out), w, dense, a, tokens, nullptr, acc, ldacc, stream);
}

extern "C" int32_t ignis_dense_prefill_linear(const ignis_trellis_weight *const *w, uint32_t count, const void *x,
                                              uint32_t tokens, const ignis_dense_prefill_output *out,
                                              const ignis_trellis_workspace *workspace, void *dense,
                                              uint64_t dense_bytes, void *stream) {
  const char *op = "ignis_dense_prefill_linear";
  if (w == nullptr || out == nullptr || count < 1 || count > IGNIS_DENSE_PREFILL_MAX_GROUP) {
    return fail(std::string(op) + ": 1 to " + std::to_string(IGNIS_DENSE_PREFILL_MAX_GROUP) + " matrices");
  }
  if (check_weight(op, w[0]) != 0) return -1;
  if (check_workspace(op, workspace, ignis_dense_prefill_workspace_bytes(w[0]->in, count, tokens)) != 0) return -1;
  ignis_dense_prefill_activations a[IGNIS_DENSE_PREFILL_MAX_GROUP];
  group_activations(*workspace, w[0]->in, count, tokens, a);
  if (ignis_dense_prefill_quantize(x, tokens, w, count, a, stream) != 0) return -1;
  return decode_and_gemm(w, count, a, tokens, out, dense, dense_bytes, stream);
}

extern "C" int32_t ignis_dense_prefill_rmsnorm_quantize(const void *x, const void *norm, float eps, uint32_t tokens,
                                                        void *h, const ignis_trellis_weight *const *w, uint32_t count,
                                                        const ignis_dense_prefill_activations *out, void *stream) {
  const char *op = "ignis_dense_prefill_rmsnorm_quantize";
  if (w == nullptr || out == nullptr || count < 1 || count > IGNIS_DENSE_PREFILL_MAX_GROUP) {
    return fail(std::string(op) + ": 1 to " + std::to_string(IGNIS_DENSE_PREFILL_MAX_GROUP) + " matrices");
  }
  if (!aligned(x, 4) || !aligned(norm, 4) || (h != nullptr && !aligned(h, 4))) {
    return fail(std::string(op) + ": x and the norm's weight must be 4-byte aligned, and h when given");
  }
  if (!(eps > 0.0f) || !std::isfinite(eps)) return fail(std::string(op) + ": eps must be positive and finite");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be positive");
  for (uint32_t i = 0; i < count; ++i) {
    if (check_weight(op, w[i]) != 0 || check_activations(op, &out[i]) != 0) return -1;
    if (w[i]->in != w[0]->in) return fail(std::string(op) + ": the matrices read inputs of different widths");
  }
  if (!norm_width(w[0]->in)) {
    return fail(std::string(op) + ": in " + std::to_string(w[0]->in) +
                ": the fused RMSnorm takes a multiple of 1024 in (3072, 8192], the vendored rmsnorm's wide-row form");
  }
  State st;
  if (state(op, &st) != 0) return -1;
  A8Group g{};
  g.count = static_cast<int>(count);
  for (uint32_t i = 0; i < count; ++i) {
    g.m[i] = A8Target{static_cast<const __half *>(w[i]->suh), static_cast<int8_t *>(out[i].q), out[i].scale, out[i].sum};
  }
  dense_prefill_rmsnorm_a8_kernel<<<tokens, kNormThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat162 *>(x), static_cast<const __nv_bfloat162 *>(norm), static_cast<int>(w[0]->in), eps,
      static_cast<__nv_bfloat162 *>(h), static_cast<int>(tokens), g);
  if (check_launch(op) != 0) return -1;
  g_fused_norm_launches.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

extern "C" int32_t ignis_dense_prefill_norm_linear(const ignis_trellis_weight *const *w, uint32_t count, const void *x,
                                                   const void *norm, float eps, uint32_t tokens, void *h,
                                                   const ignis_dense_prefill_output *out,
                                                   const ignis_trellis_workspace *workspace, void *dense,
                                                   uint64_t dense_bytes, void *stream) {
  const char *op = "ignis_dense_prefill_norm_linear";
  if (w == nullptr || out == nullptr || count < 1 || count > IGNIS_DENSE_PREFILL_MAX_GROUP) {
    return fail(std::string(op) + ": 1 to " + std::to_string(IGNIS_DENSE_PREFILL_MAX_GROUP) + " matrices");
  }
  if (check_weight(op, w[0]) != 0) return -1;
  if (check_workspace(op, workspace, ignis_dense_prefill_workspace_bytes(w[0]->in, count, tokens)) != 0) return -1;
  ignis_dense_prefill_activations a[IGNIS_DENSE_PREFILL_MAX_GROUP];
  group_activations(*workspace, w[0]->in, count, tokens, a);
  if (ignis_dense_prefill_rmsnorm_quantize(x, norm, eps, tokens, h, w, count, a, stream) != 0) return -1;
  return decode_and_gemm(w, count, a, tokens, out, dense, dense_bytes, stream);
}

extern "C" void ignis_dense_prefill_set_fused_producers(uint32_t fused) {
  g_fused_producers.store(fused != 0 ? 1u : 0u, std::memory_order_relaxed);
}

extern "C" uint32_t ignis_dense_prefill_fused_producers(void) { return g_fused_producers.load(std::memory_order_relaxed); }

extern "C" uint64_t ignis_dense_prefill_fused_norm_launches(void) {
  return g_fused_norm_launches.load(std::memory_order_relaxed);
}
