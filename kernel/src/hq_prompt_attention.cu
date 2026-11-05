// ignis kernel leaf: the hq-e8-2b prompt attention without prompt planes
// (kernel/include/ignis_hq_prompt_attention.h; ADR 0061, spec 27b-16gb/03, GitHub #335) -- an
// ADAPTED op (ADR 0051 tier 2).
//
// Upstream: mkyt/qwen3-27b-5090, commit 0594e77ca099f397ccefed0c68c981e32869989f, MIT
// ("Copyright (c) 2026 Masahiro Kiyota"):
//   engine/src/attention.cu  attention_fa_kernel, its BF16 path and its FP8 path (kF8 = 2): the
//                            CTA (a 64-token query tile of two query heads of one KV group, 8
//                            warps of 16 tokens), Q in registers as A fragments (E4M3 with one
//                            scale per row on the FP8 path), 32-key tiles in two cp.async stages
//                            under the tc_swz / sw8 swizzles, the online softmax with each row on
//                            a quad, P from the S accumulators as PV's A fragment (in E4M3 as
//                            256 p in V^T's permuted key order, through an f16x2 exp2), the mask
//                            only on tiles the diagonal crosses, query tiles aligned to absolute
//                            64-token blocks and launched longest first;
//   engine/src/mma.cuh       the block-scaled mxf8f6f4 m16n8k32 form, exp2_e4m3x2, sum_e4m3x4,
//                            ldmatrix (plain and transposed); attention.cu's vt_off.
// From the vendored reference (kernel/vendor, ADR 0010), called and not copied: the hq decode
// of a key (hq_decode_row_group_fast, hq_codec.cuh), the rotated store of a chunk row
// (hq_store_rotated_row_warp), and where each key comes from -- the chunk's own rows, the
// residual side planes, the codec -- as gqa_attention_prefill_hq_scratch_kernel
// (gqa_attention_prefill_hq.cuh) decides it.
//
// Changes from upstream (ours to change, no parity claim):
//   - The keys are the hq-e8-2b codec's, read in two passes that share one online-softmax state:
//       the residual window (sinks, the ring before the chunk, the chunk) on the BF16 path, its
//       rows exact as the prompt planes held them;
//       every other key on the 8-bit path, decoded ONCE per chunk and layer into a fixed band of
//       IgnisHqPromptOptions::band_keys keys -- K row-major in INT8 with one scale per 32-key tile
//       and KV head, V already transposed into V^T's permuted order in E4M3 with one power-of-two
//       scale -- and walked band after band. (Owner, 2026-10-21: a decode per CTA tile costs
//       ~1.7 ns per key and token, more than the whole attention.)
//   - Q K^T on that path runs in INT8 (Q with a scale per row), exact in int32, as SageAttention
//     does and kernel/src/image/attn8.cu after it; P V stays on the block-scaled E4M3 MMA.
//     Upstream's E4M3 Q costs ~2.5% of each query's value, the same for every key it scores: a
//     bias, not noise, measured at 32K keys as the median row x1.06 of A1's distance from BF16.
//   - The attention runs in the codec's rotated frame: Q is rotated once per call (and rounded
//     to BF16 for the window, quantized to INT8 with a per-row scale for the band), the output
//     is un-rotated once per row by the merge.
//   - Both passes run in base 2 with P = 2^(s - m + 8), so O and l carry one scale throughout.
//   - Key splits: each CTA attends one of `splits` equal parts of its keys in every pass and
//     keeps its own state across passes (O over its l in fp16, m and l in fp32); a merge kernel
//     combines the splits, normalizes, un-rotates and stores bf16.
//   - The V tile's scale rides the block-scaled MMA's B scale (the same byte in every lane);
//     the K tile's folds into the score scale with Q's and 1/16.
//   - The fallback (IgnisHqPromptPrecision::kFp16, spec 27b-16gb/03 Solution 6): the band in
//     FP16 at half the keys, Q K^T and P V on the FP16 MMAs with FP16 accumulate inside a tile
//     (the rate GeForce runs at twice FP32 accumulate), P = p in FP16, each tile's P V folded into
//     the FP32 O; Q pre-scaled by 1/16 as upstream's BF16 path does.
//   - One geometry: 24 query heads over 4 KV heads of 256, one sequence.

#include "ignis_hq_prompt_attention.h"

#include "hq_prompt_keys.cuh"

#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_prefill_hq.cuh"
#include "ops/kernel/hq_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

using Geometry = ninfer::ops::Gqa27Geometry;

constexpr int kHeadDim = 256;
constexpr int kQHeads = Geometry::QHeads;    // 24
constexpr int kKvHeads = Geometry::KVHeads;  // 4
constexpr int kGroup = Geometry::GroupSize;  // 6
constexpr int kTile = static_cast<int>(kIgnisHqPromptTileKeys);  // keys per tile
constexpr int kSinks = static_cast<int>(ninfer::ops::kGqaHqSinkKeys);
constexpr int kRing = static_cast<int>(ninfer::ops::kGqaHqRecentKeys);
constexpr int kQ = 64;  // query tokens per CTA
constexpr int kHeadsPerCta = 2;
constexpr int kHeadPairs = kGroup / kHeadsPerCta;        // 3
constexpr int kFaThreads = kHeadsPerCta * kQ / 16 * 32;  // 256
constexpr int kVtTileBytes = kHeadDim * kTile;           // 8 KiB of E4M3 V^T
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kE4m3Max = 448.0f;
constexpr float kInt8Max = 127.0f;
// The UE8M0 encoding of 2^0 in every byte: a unit block scale.
constexpr std::uint32_t kUnitScale = 0x7F7F7F7Fu;
constexpr int kRowThreads = 256;  // prep / merge: one warp per (token, head) row

static_assert(kGroup % kHeadsPerCta == 0, "a CTA's heads share one KV group");
static_assert(kFaThreads == 256, "the FA CTA is 8 warps");

// A tile's K and V bytes in one pass: 8-bit rows of 256 bytes, or 16-bit rows of 512.
template <bool kFp8>
constexpr int kKTileBytes = kTile * kHeadDim * (kFp8 ? 1 : 2);
template <bool kFp8>
constexpr int kStageBytes = 2 * kKTileBytes<kFp8>;
template <bool kFp8>
constexpr int kFaSmemBytes = 2 * kStageBytes<kFp8>;  // two stages: 32 KiB (FP8), 64 KiB (BF16)

// PROTOTYPE (sm8x dense prefill, phase 1, step 2): IGNIS_PROTO_BAND, in place of the FP16 band
// (IGNIS_MODEL_PREFILL_ATTENTION=fp16), the band an sm_8x card could run on its integer MMAs:
//   i8:   Q K^T as the 8-bit band (INT8 Q per row, INT8 K per 32-key tile, exact in int32); P V on
//         the u8 x s8 MMA, P = rn(255 p) (p in [0, 1], scale 1/255), V^T in s8 with amax / 127 per
//         32-key tile and KV head, exact in int32 within a tile, folded into the fp32 O;
//   i8qk: Q K^T as i8, P V as the FP16 band (P = p and V in FP16, FP16 accumulate per tile).
constexpr int kProtoOff = 0;
constexpr int kProtoI8 = 1;
constexpr int kProtoI8Qk = 2;
// The V tile's bytes: FP16 rows under i8qk, else as K's.
template <bool kFp8, int kProto>
constexpr int kVTileBytesP = kProto == kProtoI8Qk ? kTile * kHeadDim * 2 : kKTileBytes<kFp8>;
template <bool kFp8, int kProto>
constexpr int kStageBytesP = kKTileBytes<kFp8> + kVTileBytesP<kFp8, kProto>;
template <bool kFp8, int kProto>
constexpr int kFaSmemBytesP = 2 * kStageBytesP<kFp8, kProto>;
static_assert(kStageBytesP<true, kProtoOff> == kStageBytes<true> && kStageBytesP<false, kProtoOff> == kStageBytes<false>,
              "the band as it was keeps its stages");

// ---- PTX helpers (upstream engine/src/mma.cuh) --------------------------------------------------

// GitHub #456: the 8-bit band's own instructions -- the block-scaled E4M3 MMA and the e4m3x2
// conversions -- are sm_120a's. A build for an older card compiles this file for it too, for
// the BF16 window and the FP16 band, which are sm_80 throughout: there the helpers that issue
// those instructions trap, and so does the 8-bit band at its entry. The host never runs that
// band below compute capability 12.0 (--model-prefill-attention resolves to fp16 there).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#define IGNIS_HQ_PROMPT_FP8_BAND 0
#else
#define IGNIS_HQ_PROMPT_FP8_BAND 1
#endif

__device__ __forceinline__ std::uint32_t smem_u32(const void *p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(std::uint32_t smem, const void *g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(std::uint32_t *r, std::uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}
__device__ __forceinline__ void ldsm_x4_trans(std::uint32_t *r, std::uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}
__device__ __forceinline__ void mma_bf16(float *d, const std::uint32_t *a, const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D += A B with FP16 operands and an FP16 accumulator (two registers of half2, in the fp32
// form's fragment positions): the fallback band's MMA.
__device__ __forceinline__ void mma_f16acc(std::uint32_t *d, const std::uint32_t *a,
                                           const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float2 half2_bits(std::uint32_t x) {
  return __half22float2(*reinterpret_cast<const __half2 *>(&x));
}
// D += A B over INT8, int32 accumulate (exact): the band's Q K^T.
__device__ __forceinline__ void mma_s8(int *d, const std::uint32_t *a, const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// PROTOTYPE: D += A B with A u8 and B s8, int32 accumulate (exact): the i8 band's P V.
__device__ __forceinline__ void mma_u8s8(int *d, const std::uint32_t *a, const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D += (A * 2^(sfa-127)) (B * 2^(sfb-127)): E4M3 m16n8k32 with fp32 accumulate on the
// block-scaled form, which the 5090 runs at twice the plain form's rate. Every byte of a scale
// register carries the same scale here, so which lane supplies it does not matter.
__device__ __forceinline__ void mma_e4m3(float *d, const std::uint32_t *a, const std::uint32_t *b,
                                         std::uint32_t sfa, std::uint32_t sfb) {
#if IGNIS_HQ_PROMPT_FP8_BAND
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3."
      "f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "r"(sfb));
#else
  __trap();
#endif
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// E4M3 of 2^a, 2^b (a in byte 0) through an f16x2 exp2: enough for the 3-bit mantissa.
__device__ __forceinline__ std::uint32_t exp2_e4m3x2(float a, float b) {
#if IGNIS_HQ_PROMPT_FP8_BAND
  std::uint32_t h;
  std::uint16_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(b), "f"(a));
  asm("ex2.approx.f16x2 %0, %0;" : "+r"(h));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"(h));
  return r;
#else
  __trap();
  return 0;
#endif
}
// The sum of the four E4M3 values of x.
__device__ __forceinline__ float sum_e4m3x4(std::uint32_t x) {
#if IGNIS_HQ_PROMPT_FP8_BAND
  std::uint32_t lo, hi;
  asm("{\n.reg .b16 a, b;\n"
      "mov.b32 {a, b}, %2;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n}"
      : "=r"(lo), "=r"(hi)
      : "r"(x));
  const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&lo));
  const float2 g = __half22float2(*reinterpret_cast<const __half2 *>(&hi));
  return (f.x + f.y) + (g.x + g.y);
#else
  __trap();
  return 0.0f;
#endif
}
__device__ __forceinline__ std::uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t *>(&h);
}
__device__ __forceinline__ std::uint32_t pack_half(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t *>(&h);
}
// One INT8 code of x (|x| <= 127 by construction), as a byte.
__device__ __forceinline__ std::uint8_t i8(float x) {
  return static_cast<std::uint8_t>(static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(x)))));
}
__device__ __forceinline__ std::uint8_t e4m3(float x) {
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
// 16-byte unit `unit` of K (or V) row `row` in a staged tile, XOR-swizzled by row % 8 so that
// ldmatrix is conflict-free: rows of 256 bytes on the FP8 path (upstream sw8), 512 on the BF16
// path (upstream tc_swz).
template <bool kFp8>
__device__ __forceinline__ std::uint32_t row_swz(int row, int unit) {
  return static_cast<std::uint32_t>(row * kHeadDim * (kFp8 ? 1 : 2) + ((unit ^ (row & 7)) << 4));
}
// 16-byte segment (dim d, keys 16 hh .. 16 hh + 15) of a staged V^T tile.
__device__ __forceinline__ std::uint32_t vt_off(int d, int hh) {
  return static_cast<std::uint32_t>(((2 * d + hh) ^ ((d >> 2) & 7)) << 4);
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}

// ---- which keys a pass reads, and where each one comes from ----------------------------------

// A pass's keys in column order: column i < split is key first0 + i, a later one key
// first1 + i - split; columns from `count` on hold no key. The window pass is the sinks then the
// ring and the chunk; a band is one run of keys.
struct Columns {
  int first0;
  int split;
  int first1;
  int count;

  __device__ __forceinline__ int key(int i) const { return i < split ? first0 + i : first1 + i - split; }
  // How many columns hold keys below `limit`: keys rise with the column.
  __device__ __forceinline__ int before(int limit) const {
    const int head = min(split, count);
    const int n0 = max(0, min(head, limit - first0));
    if (n0 < head || count <= split) return n0;
    return n0 + max(0, min(count - split, limit - first1));
  }
};

// ---- Q: rotate into the codec's frame, once per call ------------------------------------------

// q8 / q_scale (or null): INT8 with one scale per row, for the 8-bit band; qh (or null): FP16 over
// 16, for the fallback band; q16 (or null): BF16, for the window pass, with the single rounding
// the prompt planes' consumer gave it.
__global__ void __launch_bounds__(kRowThreads) prep_q_kernel(const __nv_bfloat16 *q, int rows,
                                                             std::uint8_t *q8, float *q_scale,
                                                             __half *qh, __nv_bfloat16 *q16) {
  __shared__ std::int8_t signs[kHeadDim];
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * (kRowThreads / 32) + static_cast<int>(threadIdx.x) / 32;
  if (row >= rows) return;
  const __nv_bfloat16 *src = q + static_cast<std::int64_t>(row) * kHeadDim;
  float reg[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) reg[s] = __bfloat162float(src[s * 32 + lane]);
  ninfer::ops::hq_fwht256_sign(reg, signs, 0, lane);
  if (q16 != nullptr) {
    __nv_bfloat16 *dst = q16 + static_cast<std::int64_t>(row) * kHeadDim;
#pragma unroll
    for (int s = 0; s < 8; ++s) dst[s * 32 + lane] = __float2bfloat16(reg[s]);
  }
  if (qh != nullptr) {
    __half *dst = qh + static_cast<std::int64_t>(row) * kHeadDim;
#pragma unroll
    for (int s = 0; s < 8; ++s) dst[s * 32 + lane] = __float2half_rn(reg[s] * (1.0f / 16.0f));
  }
  if (q8 == nullptr) return;
  float mx = 0.0f;
#pragma unroll
  for (int s = 0; s < 8; ++s) mx = fmaxf(mx, fabsf(reg[s]));
  mx = warp_max(mx);
  const float inv = mx > 0.0f ? kInt8Max / mx : 0.0f;
  std::uint8_t *dst = q8 + static_cast<std::int64_t>(row) * kHeadDim;
#pragma unroll
  for (int s = 0; s < 8; ++s) dst[s * 32 + lane] = i8(reg[s] * inv);
  if (lane == 0) q_scale[row] = mx * (1.0f / kInt8Max);
}

// ---- a pass's keys: one 32-key tile of one KV head and role per CTA ---------------------------

// Where a pass's tiles live. 8-bit: k = INT8 [kv_head][capacity][256], v = E4M3 V^T
// [kv_head][capacity / 32][256 dims x 32 keys], scale = [role][kv_head][capacity / 32] (K's
// amax / 127, V's a power of two). BF16 / FP16: k, v = [kv_head][capacity][256], no scale.
struct PassBuffers {
  std::uint8_t *k;
  std::uint8_t *v;
  float *scale;
  std::int32_t capacity;  // columns per KV head
};

template <int kProto = kProtoOff>
__device__ __forceinline__ void store_fp8_tile(const __nv_bfloat16 *tile, bool role_v, int head,
                                               int tile_index, PassBuffers buf, float *warp_amax,
                                               float &tile_scale);

// `role_first`: the role of blockIdx.z == 0 (K, or V when 1): a pass launches both roles, a
// fetch one.
// `kHalf` (with !kFp8): the fallback band, the rows stored in FP16 instead of BF16.
// PROTOTYPE `kProto` (with kFp8): i8 stores V^T in s8; i8qk stores K in INT8 and V rows in FP16.
template <bool kFp8, bool kHalf = false, int kProto = kProtoOff>
__global__ void __launch_bounds__(256) tile_kernel(IgnisHqPromptKeys src, Columns cols,
                                                   PassBuffers buf, int role_first) {
  __shared__ __align__(16) __nv_bfloat16 tile[kTile * kHeadDim];
  __shared__ std::int8_t signs[kHeadDim];
  __shared__ float warp_amax[8];
  __shared__ float tile_scale;
  const int tile_index = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const bool role_v = role_first + static_cast<int>(blockIdx.z) != 0;
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int column0 = tile_index * kTile;
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
  // Four rows a warp, one per 8-lane group; a column past the pass's keys is a zero row.
  {
    const int key_l = warp * 4 + (lane >> 3);
    const int column = column0 + key_l;
    ignis_hq_prompt::fetch_rows_warp(src, head, role_v, column < cols.count ? cols.key(column) : -1,
                                     tile + key_l * kHeadDim, signs);
  }
  __syncthreads();

  if constexpr (!kFp8 || kProto == kProtoI8Qk) {
    if (kFp8 && !role_v) {  // PROTOTYPE i8qk: K in INT8, as the 8-bit band
      store_fp8_tile<kProtoOff>(tile, role_v, head, tile_index, buf, warp_amax, tile_scale);
      return;
    }
    // BF16 rows as the prompt planes held them, four 16-byte units per thread; rows past the
    // buffer's capacity (a fetch of a count that is no multiple of 32) are not stored.
    std::uint8_t *base = role_v ? buf.v : buf.k;
    auto *dst = reinterpret_cast<uint4 *>(
        base + (static_cast<std::int64_t>(head) * buf.capacity + column0) * kHeadDim * 2);
    const auto *from = reinterpret_cast<const uint4 *>(tile);
    for (int u = tid; u < kTile * kHeadDim / 8; u += 256) {
      if (column0 + u / (kHeadDim / 8) >= buf.capacity) continue;
      uint4 w = from[u];
      if constexpr (kHalf || kProto == kProtoI8Qk) {
        std::uint32_t *x = &w.x;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&x[j]));
          x[j] = pack_half(f.x, f.y);
        }
      }
      dst[u] = w;
    }
  } else {
    store_fp8_tile<kProto>(tile, role_v, head, tile_index, buf, warp_amax, tile_scale);
  }
}

// The 8-bit half of tile_kernel: the tile's scale, then K rows in INT8 or V^T segments in E4M3.
// PROTOTYPE i8: V^T in s8 under amax / 127, in the same segment order.
template <int kProto>
__device__ __forceinline__ void store_fp8_tile(const __nv_bfloat16 *tile, bool role_v, int head,
                                               int tile_index, PassBuffers buf, float *warp_amax,
                                               float &tile_scale) {
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int column0 = tile_index * kTile;
  // One scale for the tile. V: a power of two, E4M3 keeping its relative precision over 15
  // binades below its largest value. K: amax / 127 for INT8 -- its rows share a norm (the
  // checkpoint's per-head k norm), so a scale per tile costs no row its resolution.
  float mx = 0.0f;
  {
    const uint4 *p = reinterpret_cast<const uint4 *>(tile) + tid * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = p[i];
      const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
        mx = fmaxf(mx, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  }
  mx = warp_max(mx);
  if (lane == 0) warp_amax[warp] = mx;
  __syncthreads();
  if (tid == 0) {
    float amax = 0.0f;
    for (int w = 0; w < 8; ++w) amax = fmaxf(amax, warp_amax[w]);
    float scale = 1.0f;
    if (amax > 0.0f && role_v && kProto == kProtoI8) {
      scale = amax / kInt8Max;
    } else if (amax > 0.0f && role_v) {
      int e = 0;
      (void)frexpf(amax / kE4m3Max, &e);  // amax / 448 < 2^e
      scale = ldexpf(1.0f, e);
    } else if (amax > 0.0f) {
      scale = amax / kInt8Max;
    }
    tile_scale = scale;
    const int tiles = buf.capacity / kTile;
    buf.scale[((role_v ? kKvHeads : 0) + head) * tiles + tile_index] = scale;
  }
  __syncthreads();
  const float inv = 1.0f / tile_scale;  // V: exact, a power of two

  if (!role_v) {
    // K rows in INT8: two 16-byte units per thread.
    std::uint8_t *dst =
        buf.k + (static_cast<std::int64_t>(head) * buf.capacity + column0) * kHeadDim;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int u = tid + i * 256, r = u >> 4, col = u & 15;
      const __nv_bfloat16 *v = tile + r * kHeadDim + col * 16;
      std::uint32_t w[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) {
        w[x] = static_cast<std::uint32_t>(i8(__bfloat162float(v[4 * x + 0]) * inv)) |
               static_cast<std::uint32_t>(i8(__bfloat162float(v[4 * x + 1]) * inv)) << 8 |
               static_cast<std::uint32_t>(i8(__bfloat162float(v[4 * x + 2]) * inv)) << 16 |
               static_cast<std::uint32_t>(i8(__bfloat162float(v[4 * x + 3]) * inv)) << 24;
      }
      *reinterpret_cast<uint4 *>(dst + r * kHeadDim + col * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  } else {
    // V^T: segment (d, hh) holds dim d of keys 16 hh .. 16 hh + 15, byte 4 j + i being key
    // 16 hh + 2 j + (i & 1) + 8 (i >> 1) -- the order in which a thread holds P in the S
    // accumulators, so P packs into PV's A fragment without shuffles.
    std::uint8_t *dst =
        buf.v + (static_cast<std::int64_t>(head) * (buf.capacity / kTile) + tile_index) *
                    kVtTileBytes;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int seg = tid + i * 256, d = seg >> 1, hh = seg & 1;
      std::uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        std::uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int key_l = 16 * hh + 2 * j + (b & 1) + 8 * (b >> 1);
          const float x = __bfloat162float(tile[key_l * kHeadDim + d]) * inv;
          word |= static_cast<std::uint32_t>(kProto == kProtoI8 ? i8(x) : e4m3(x)) << (8 * b);
        }
        w[j] = word;
      }
      *reinterpret_cast<uint4 *>(dst + seg * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

// ---- the attention over one pass ----------------------------------------------------------------

struct Carry {
  // [split][token][head][256], in the rotated frame, normalized by the split's own l: FP16 keeps
  // 11 bits of it where BF16 would keep 8, and the value stays inside V's own range.
  __half *o;
  float *m;          // [split][token][head], log2 units of the scaled score
  float *l;          // [split][token][head], in units of 256 p
};

struct Queries {
  const std::uint8_t *q8;   // INT8 [token][head][256]
  const float *q_scale;     // [token][head]
  const __half *qh;         // FP16 [token][head][256], over 16
  const __nv_bfloat16 *q16; // BF16 [token][head][256]
};

// kFp8: the 8-bit band (INT8 Q K^T, E4M3 P V); kHalf: the FP16-accumulate fallback band; neither:
// the BF16 window.
// PROTOTYPE kProto (with kFp8): the i8 / i8qk bands, Q K^T as the 8-bit band's.
template <bool kFp8, bool kHalf = false, int kProto = kProtoOff>
__global__ void __launch_bounds__(kFaThreads, 1)
    fa_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
              int splits, int resume, Carry carry) {
  static_assert(!(kFp8 && kHalf), "one band precision");
  static_assert(kProto == kProtoOff || kFp8, "the prototype bands run Q K^T in INT8");
  constexpr int kStage = kStageBytesP<kFp8, kProto>;
  // PROTOTYPE (phase 2d): kFp8 is the 8-bit storage (INT8 Q and K, 8-bit V^T); only the 8-bit
  // band as it was (kProtoOff) multiplies P V on the FP8 MMA. The i8 and i8qk instantiations
  // store 8 bits but run integer and FP16 MMAs, which every sm_80 card has: they are no FP8 band.
#if !IGNIS_HQ_PROMPT_FP8_BAND
  constexpr bool kFp8Mma = kFp8 && kProto == kProtoOff;
  if constexpr (kFp8Mma) {
    __trap();  // the 8-bit band below sm_120: never a silent stand-in (GitHub #456)
  }
#endif
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  constexpr int kKSteps = kFp8 ? kHeadDim / 32 : kHeadDim / 16;  // QK contraction steps
  const std::uint32_t sbase = smem_u32(fa_smem);
  const int lead = start % kQ;  // rows of the first absolute block before this call
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;  // longest first
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;  // this tile's first row in the call; negative: before start
  const int rend = min(start + q0 + kQ, start + width);  // keys some row of the tile sees
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = (tid / 32) % (kQ / 16), lane = tid % 32;
  const int g = static_cast<int>(blockIdx.y) / kHeadPairs;  // KV head
  const int h = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta +
                tid / (kQ / 16 * 32);
  // This CTA's columns, then its split of them, in whole tiles.
  const int n = (cols.before(rend) + kTile - 1) / kTile;
  const int cbeg = split * n / splits, cend = (split + 1) * n / splits;
  // A split with nothing here keeps the state it carried in; on the first pass it still writes
  // the empty state, which is what the merge reads for it.
  if (cend <= cbeg && resume != 0) return;

  auto issue = [&](int c) {
    const std::uint32_t st = sbase + static_cast<std::uint32_t>((c & 1) * kStage);
    if constexpr (kProto == kProtoI8Qk) {
      // K in INT8 rows as the 8-bit band, V in FP16 rows as the fallback band.
      const std::uint8_t *ksrc =
          buf.k + (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim;
      const std::int64_t at = (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim * 2;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int u = tid + j * kFaThreads;
        cp_async16(st + row_swz<true>(u >> 4, u & 15), ksrc + u * 16);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int u = tid + j * kFaThreads;
        cp_async16(st + kKTileBytes<true> + row_swz<false>(u >> 5, u & 31), buf.v + at + u * 16);
      }
    } else if constexpr (kFp8) {
      const std::uint8_t *ksrc =
          buf.k + (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim;
      const std::uint8_t *vsrc =
          buf.v + (static_cast<std::int64_t>(g) * (buf.capacity / kTile) + c) * kVtTileBytes;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int u = tid + j * kFaThreads;
        cp_async16(st + row_swz<true>(u >> 4, u & 15), ksrc + u * 16);
        cp_async16(st + kKTileBytes<true> + vt_off(u >> 1, u & 1), vsrc + u * 16);
      }
    } else {
      const std::int64_t at = (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim * 2;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int u = tid + j * kFaThreads;
        cp_async16(st + row_swz<false>(u >> 5, u & 31), buf.k + at + u * 16);
        cp_async16(st + kKTileBytes<false> + row_swz<false>(u >> 5, u & 31), buf.v + at + u * 16);
      }
    }
  };
  if (cend > cbeg) issue(cbeg);
  cp_commit();
  if (cend - cbeg > 1) issue(cbeg + 1);
  cp_commit();

  // Q of this warp's 16 tokens as A fragments: a[0] / a[1] = rows ta / tb, a[2] / a[3] the same
  // 16 dims on (FP8: 4 (lane % 4) .. + 3 of each 32-dim step; BF16: 2 (lane % 4), + 1 of each
  // 16-dim step, + 8).
  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kKSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = kFp8 && in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h]
                     : kHalf  ? 16.0f
                              : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
#pragma unroll
  for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = q0 + ((i & 1) ? tb : ta);
      const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
      if (t < 0 || t >= width) {
        qa[kk][i] = 0u;
      } else if constexpr (kFp8) {
        qa[kk][i] = *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                             16 * (i / 2) + 4 * (lane % 4));
      } else if constexpr (kHalf) {
        qa[kk][i] = *reinterpret_cast<const std::uint32_t *>(qs.qh + row * kHeadDim + kk * 16 +
                                                             8 * (i / 2) + c2);
      } else {
        qa[kk][i] = *reinterpret_cast<const std::uint32_t *>(qs.q16 + row * kHeadDim + kk * 16 +
                                                             8 * (i / 2) + c2);
      }
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;  // positions of the two rows
  const int warp_last = start + q0 + warp * 16 + 15;     // the warp's last row position

  float o[kHeadDim / 8][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row];
      const __half *po = carry.o + row * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  const int lm = lane / 8, lr = lane % 8;

  for (int c = cbeg; c < cend; ++c) {
    cp_wait<1>();
    __syncthreads();
    const std::uint32_t ks = sbase + static_cast<std::uint32_t>((c & 1) * kStage);
    const std::uint32_t vs = ks + kKTileBytes<kFp8>;
    const int key0 = cols.key(c * kTile);  // the tile's keys are key0 .. key0 + 31
    if (key0 <= warp_last) {
      // S: 16 tokens x 32 keys = 4 n8 tiles.
      float sc[4][4] = {};
      float sk = 1.0f;
      std::uint32_t sfv = kUnitScale;
      float sv = 1.0f;  // PROTOTYPE i8: the V tile's amax / 127
      if constexpr (kFp8) {
        const int tiles = buf.capacity / kTile;
        sk = buf.scale[g * tiles + c];
        if constexpr (kProto == kProtoI8) sv = buf.scale[(kKvHeads + g) * tiles + c];
        // The V tile's power-of-two scale as a UE8M0 byte: its float exponent field.
        sfv = ((__float_as_uint(buf.scale[(kKvHeads + g) * tiles + c]) >> 23) & 0xFFu) * 0x01010101u;
        // Q K^T in INT8, exact in int32: |S| <= 256 * 127^2 < 2^22, so 1.5 * 2^23 + S holds S
        // in the float's mantissa and one subtraction lifts it exactly (kernel/src/image/attn8.cu).
        int si[4][4] = {};
#pragma unroll
        for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            std::uint32_t b[4];
            ldsm_x4(b, ks + row_swz<true>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma_s8(si[2 * p], qa[kk], b);
            mma_s8(si[2 * p + 1], qa[kk], b + 2);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) sc[nt][e] = __fsub_rn(__int_as_float(0x4B400000 + si[nt][e]), 12582912.0f);
        }
      } else if constexpr (kHalf) {
        // FP16 accumulate across the 256 dims, then into the fp32 softmax.
        std::uint32_t sh[4][2] = {};
#pragma unroll
        for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            std::uint32_t b[4];
            ldsm_x4(b, ks + row_swz<false>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma_f16acc(sh[2 * p], qa[kk], b);
            mma_f16acc(sh[2 * p + 1], qa[kk], b + 2);
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float2 lo = half2_bits(sh[nt][0]), hi = half2_bits(sh[nt][1]);
          sc[nt][0] = lo.x;
          sc[nt][1] = lo.y;
          sc[nt][2] = hi.x;
          sc[nt][3] = hi.y;
        }
      } else {
#pragma unroll
        for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            std::uint32_t b[4];
            ldsm_x4(b, ks + row_swz<false>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma_bf16(sc[2 * p], qa[kk], b);
            mma_bf16(sc[2 * p + 1], qa[kk], b + 2);
          }
        }
      }
      // Base 2: S scaled by its Q, K and 1/16 scales times log2 e, so m is in log2 units.
      const float fs[2] = {f[0] * sk, f[1] * sk};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nt][e] *= fs[e / 2];
      }
      // The mask only on a tile past the warp's first row or past the pass's keys; the online
      // softmax with each row spread over the 4 lanes of a quad.
      const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = nt * 8 + c2 + (e & 1);
          if (edge && (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb))) {
            sc[nt][e] = -CUDART_INF_F;
          }
          mx[e / 2] = fmaxf(mx[e / 2], sc[nt][e]);
        }
      }
      float corr[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -CUDART_INF_F ? 1.0f : ex2_approx(m[r] - mn);
        m[r] = mn;
      }
      // P x 256 = 2^(s - m + 8).
      const float bias[2] = {8.0f - (m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                             8.0f - (m[1] == -CUDART_INF_F ? 0.0f : m[1])};
      float ps[2] = {0.0f, 0.0f};
      if constexpr (kProto == kProtoI8) {
        // PROTOTYPE i8: P = rn(255 p) as one u8 A fragment over the 32 keys, in V^T's order (the
        // bytes the E4M3 path packs); P V on the u8 x s8 MMA, exact in int32 for the tile
        // (|sum| <= 32 * 255 * 127 < 2^24), then folded into the fp32 O at 256 p = P 256 / 255.
        std::uint32_t pf[4];
        int pi[2] = {0, 0};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          std::uint32_t hx[4];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int a = __float2int_rn(255.0f * ex2_approx(sc[nt][2 * r] + bias[r] - 8.0f));
            const int b = __float2int_rn(255.0f * ex2_approx(sc[nt][2 * r + 1] + bias[r] - 8.0f));
            hx[nt] = static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b) << 8;
            pi[r] += a + b;
          }
          pf[r] = hx[0] | hx[1] << 16;
          pf[2 + r] = hx[2] | hx[3] << 16;
        }
        constexpr float kU8 = 256.0f / 255.0f;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 1);
          pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 2);
          l[r] = l[r] * corr[r] + static_cast<float>(pi[r]) * kU8;
        }
        const float fv = sv * kU8;
#pragma unroll
        for (int dp = 0; dp < kHeadDim / 16; ++dp) {
          int t[2][4] = {};
          std::uint32_t b[4];
          ldsm_x4(b, vs + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
          mma_u8s8(t[0], pf, b);
          mma_u8s8(t[1], pf, b + 2);
#pragma unroll
          for (int x = 0; x < 2; ++x) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              o[2 * dp + x][e] = o[2 * dp + x][e] * corr[e / 2] + static_cast<float>(t[x][e]) * fv;
            }
          }
        }
      } else if constexpr (kFp8 && kProto == kProtoOff) {
        // One E4M3 A fragment over the 32 keys, in V^T's order.
        std::uint32_t pf[4];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          std::uint32_t hx[4];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            hx[nt] = exp2_e4m3x2(sc[nt][2 * r] + bias[r], sc[nt][2 * r + 1] + bias[r]);
          }
          pf[r] = hx[0] | hx[1] << 16;
          pf[2 + r] = hx[2] | hx[3] << 16;
          ps[r] = sum_e4m3x4(pf[r]) + sum_e4m3x4(pf[2 + r]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
          l[r] = l[r] * corr[r] + ps[r];
        }
        // O = O corr + P V: n = dims (two n8 tiles per ldmatrix.x4 of V^T), k = the 32 keys.
#pragma unroll
        for (int dp = 0; dp < kHeadDim / 16; ++dp) {
          float t[2][4] = {};
          std::uint32_t b[4];
          ldsm_x4(b, vs + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
          mma_e4m3(t[0], pf, b, kUnitScale, sfv);
          mma_e4m3(t[1], pf, b + 2, kUnitScale, sfv);
#pragma unroll
          for (int x = 0; x < 2; ++x) {
#pragma unroll
            for (int e = 0; e < 4; ++e) o[2 * dp + x][e] = o[2 * dp + x][e] * corr[e / 2] + t[x][e];
          }
        }
      } else if constexpr (kHalf || kProto == kProtoI8Qk) {
        // P = p (not 256 p: the FP16 accumulator holds 32 keys of P V) as FP16 A fragments, each
        // tile's P V folded into the fp32 O at 256 p, the scale O and l carry.
        std::uint32_t pf[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const std::uint32_t p2 = pack_half(ex2_approx(sc[nt][2 * r] + bias[r] - 8.0f),
                                               ex2_approx(sc[nt][2 * r + 1] + bias[r] - 8.0f));
            const float2 pr = half2_bits(p2);
            ps[r] += pr.x + pr.y;
            pf[nt / 2][(nt % 2) * 2 + r] = p2;
          }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
          l[r] = l[r] * corr[r] + 256.0f * ps[r];
        }
#pragma unroll
        for (int dp = 0; dp < kHeadDim / 16; ++dp) {
          std::uint32_t t[2][2] = {};
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            std::uint32_t b[4];
            ldsm_x4_trans(b, vs + row_swz<false>(16 * j + lr + 8 * (lm % 2), 2 * dp + lm / 2));
            mma_f16acc(t[0], pf[j], b);
            mma_f16acc(t[1], pf[j], b + 2);
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const float2 lo = half2_bits(t[x][0]), hi = half2_bits(t[x][1]);
            o[2 * dp + x][0] = o[2 * dp + x][0] * corr[0] + 256.0f * lo.x;
            o[2 * dp + x][1] = o[2 * dp + x][1] * corr[0] + 256.0f * lo.y;
            o[2 * dp + x][2] = o[2 * dp + x][2] * corr[1] + 256.0f * hi.x;
            o[2 * dp + x][3] = o[2 * dp + x][3] * corr[1] + 256.0f * hi.y;
          }
        }
      } else {
        // BF16 A fragments for keys 0-15 and 16-31, straight from the S accumulators.
        std::uint32_t pf[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            const std::uint32_t p2 = pack_bf16(ex2_approx(sc[nt][2 * r] + bias[r]),
                                               ex2_approx(sc[nt][2 * r + 1] + bias[r]));
            const float2 pr = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&p2));
            ps[r] += pr.x + pr.y;
            pf[nt / 2][(nt % 2) * 2 + r] = p2;
          }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
          ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
          l[r] = l[r] * corr[r] + ps[r];
        }
#pragma unroll
        for (int dt = 0; dt < kHeadDim / 8; ++dt) {
          o[dt][0] *= corr[0];
          o[dt][1] *= corr[0];
          o[dt][2] *= corr[1];
          o[dt][3] *= corr[1];
        }
        // O += P V: k = keys (two k16 steps), n = dims (two n8 tiles per ldmatrix.x4.trans).
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
          for (int dp = 0; dp < kHeadDim / 16; ++dp) {
            std::uint32_t b[4];
            ldsm_x4_trans(b, vs + row_swz<false>(16 * j + lr + 8 * (lm % 2), 2 * dp + lm / 2));
            mma_bf16(o[2 * dp], pf[j], b);
            mma_bf16(o[2 * dp + 1], pf[j], b + 2);
          }
        }
      }
    }
    __syncthreads();  // every warp is done with this stage
    if (c + 2 < cend) issue(c + 2);
    cp_commit();
  }

  // The split's state for the next pass or the merge, O normalized by its own l.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0) {
      carry.m[row] = m[r];
      carry.l[row] = l[r];
    }
  }
}

// ---- PROTOTYPE (sm8x dense prefill, phase 2d): the i8 band's own kernel ------------------------
//
// The i8 band's math exactly as fa_kernel<true, false, kProtoI8> (5ea54bc6) runs it, tile by tile
// and in the same fp32 order, every rounding written out (__fmul_rn, __fadd_rn, __fmaf_rn) so the
// compiler cannot fuse it another way; the bench holds the output to that kernel's, byte for
// byte. What changes is the schedule, one template switch per lever, so the bench can measure
// each:
//   - always, the conversions off the slow pipes: the int32 accumulators start at 1.5 2^23, so one
//     FADD lifts S and each tile's P V exactly (no I2F: |S| < 2^22, |P V| <= 32 255 127 < 2^20);
//     P = rn(255 p) is the low byte of 255 p + 1.5 2^23 (no F2I; ties to even as cvt.rni),
//     packed with PRMT, its row sum with IDP4A; the mask only on a tile that needs it;
//   - kStages cp.async stages with one barrier per tile (kOneBarrier), or 5ea54bc6's two;
//   - kPipeQk: tile c + 1's Q K^T is issued before tile c's softmax, so its MMAs run under it;
//   - kQSmem: Q's fragments by ldmatrix from shared memory instead of 32 registers;
//   - kPingPong: the two warps of each scheduler run half a tile apart -- the early one runs tile
//     c's Q K^T, softmax and P V, the late one tile c - 1's P V, then tile c's Q K^T and softmax --
//     so one warp's MMAs fill the other's softmax; a stage is refilled one tile later;
//   - kFuseQk: tile c + 1's Q K^T inside tile c's P V loop, one k step every two dims, so the
//     SASS itself alternates the two MMAs with the fold (a separate next-tile block, kPipeQk, is
//     front-loaded by the compiler and leaves the softmax without MMAs);
//   - kOneHead: a CTA of 4 warps for one head's 64 rows, two CTAs an SM, so each scheduler holds
//     warps of two CTAs whose barriers do not line them up (each K and V tile is staged per 64
//     rows instead of 128);
//   - kPad: the staged tiles padded instead of XOR-swizzled (K rows 272 bytes apart, V^T 16 bytes
//     after every 4 dims), still conflict-free, so every ldmatrix of a tile is one base register
//     and an immediate offset.
// kDiag (the bench's --diag; the op never launches it) disables stages: kDiagNoPv drops P V and
// its fold, kDiagNoSoftmax takes P from S's bits, kDiagNoQk takes S from Q's bits (or the lane).
constexpr int kDiagNoPv = 1;
constexpr int kDiagNoSoftmax = 2;
constexpr int kDiagNoQk = 4;
// Finer cuts inside the softmax (fa_i8_kernel only): no exp2 (its input taken as is), no max and
// sum reductions (each row's max taken from one element, no quad shuffles), no P -> u8 conversion
// and packing (P's bytes from the bits of 255 p).
constexpr int kDiagNoExp = 8;
constexpr int kDiagNoMaxSum = 16;
constexpr int kDiagNoPack = 32;
// The fold with one FP32 op an element instead of three (o += the accumulator's bits as a float):
// what the exact conversion and the scale cost.
constexpr int kDiagCheapFold = 64;
constexpr int kKRowPad = kHeadDim + 16;  // a padded K row: 272 bytes
template <bool kPad>
constexpr int kI8KBytes = kPad ? kTile * kKRowPad : kKTileBytes<true>;
template <bool kPad>
constexpr int kI8VBytes = kPad ? kVtTileBytes + kHeadDim / 4 * 16 : kVtTileBytes;
template <bool kPad>
constexpr int kI8TileBytes = kI8KBytes<kPad> + kI8VBytes<kPad>;  // 16 KiB, padded 17.5 KiB
template <bool kPad>
constexpr int kQSmemBytes = kHeadsPerCta * kQ * (kPad ? kKRowPad : kHeadDim);  // the CTA's Q
constexpr std::int32_t kMagic = 0x4B400000;  // 1.5 2^23 as float bits
constexpr float kMagicF = 12582912.0f;
template <int kStages, bool kQSmem, bool kPad>
constexpr int kI8SmemBytes = kStages * kI8TileBytes<kPad> + (kQSmem ? kQSmemBytes<kPad> : 0);

// 16-byte unit `unit` of K row `row`, and V^T segment (dim d, keys 16 hh .. 16 hh + 15), in a
// staged tile: XOR-swizzled as fa_kernel's (row_swz<true>, vt_off), or padded.
template <bool kPad>
__device__ __forceinline__ std::uint32_t k_at(int row, int unit) {
  return kPad ? static_cast<std::uint32_t>(row * kKRowPad + unit * 16) : row_swz<true>(row, unit);
}
template <bool kPad>
__device__ __forceinline__ std::uint32_t vt_at(int d, int hh) {
  return kPad ? static_cast<std::uint32_t>(d * 32 + (d >> 2) * 16 + hh * 16) : vt_off(d, hh);
}

template <int kStages, bool kOneBarrier, bool kPipeQk, bool kQSmem, bool kPingPong, bool kPad,
          bool kFuseQk, bool kOneHead, int kDiag = 0>
__global__ void __launch_bounds__(kOneHead ? kFaThreads / 2 : kFaThreads, kOneHead ? 2 : 1)
    fa_i8_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                 int splits, int resume, Carry carry) {
  static_assert(kStages >= 2 && (!kPipeQk || (kOneBarrier && kStages >= 3)),
                "the pipelined Q K^T needs one barrier per tile and three stages");
  static_assert(!kPingPong || (!kPipeQk && kStages >= 3),
                "the ping-pong holds a stage one tile longer: three stages");
  static_assert(!kFuseQk || (kOneBarrier && !kPipeQk && !kPingPong && kStages >= 3),
                "the fused Q K^T reads the next tile during this one's P V: one barrier, three stages");
  static_assert(!kOneHead || (!kPingPong && !kQSmem), "one head a CTA: no partner warp, no shared Q");
  constexpr int kThreads = kOneHead ? kFaThreads / 2 : kFaThreads;
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  constexpr int kKSteps = kHeadDim / 32;
  constexpr int kTileB = kI8TileBytes<kPad>;
  const std::uint32_t sbase = smem_u32(fa_smem);
  const std::uint32_t qbase = sbase + static_cast<std::uint32_t>(kStages * kTileB);
  const int lead = start % kQ;
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;  // longest first
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;
  const int rend = min(start + q0 + kQ, start + width);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = (tid / 32) % (kQ / 16), lane = tid % 32;
  // kOneHead: blockIdx.y is the head (24), else the KV head and head pair (12).
  const int g = static_cast<int>(blockIdx.y) / (kOneHead ? kGroup : kHeadPairs);
  const int hl = kOneHead ? 0 : tid / (kQ / 16 * 32);  // the CTA's head, 0 or 1
  const int h = kOneHead ? static_cast<int>(blockIdx.y)
                         : g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + hl;
  const int n = (cols.before(rend) + kTile - 1) / kTile;
  const int cbeg = split * n / splits, cend = (split + 1) * n / splits;
  if (cend <= cbeg && resume != 0) return;

  auto stage_of = [](int c) { return static_cast<std::uint32_t>((c % kStages) * kTileB); };
  auto issue = [&](int c) {
    const std::uint32_t st = sbase + stage_of(c);
    const std::uint8_t *ksrc =
        buf.k + (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim;
    const std::uint8_t *vsrc =
        buf.v + (static_cast<std::int64_t>(g) * (buf.capacity / kTile) + c) * kVtTileBytes;
#pragma unroll
    for (int j = 0; j < 2 * kFaThreads / kThreads; ++j) {
      const int u = tid + j * kThreads;
      cp_async16(st + k_at<kPad>(u >> 4, u & 15), ksrc + u * 16);
      cp_async16(st + kI8KBytes<kPad> + vt_at<kPad>(u >> 1, u & 1), vsrc + u * 16);
    }
  };
  // kQSmem: the CTA's two heads of 64 rows, laid out as K, in the first tile's group; a row
  // outside the call is zeros, as the register path's.
  auto issue_q = [&]() {
    if constexpr (kQSmem) {
#pragma unroll
      for (int j = 0; j < kHeadsPerCta * kQ * 16 / kFaThreads; ++j) {
        const int u = tid + j * kFaThreads;
        const int head = u / (kQ * 16), row = (u / 16) % kQ, unit = u % 16;
        const std::uint32_t dst =
            qbase + static_cast<std::uint32_t>(head * (kQSmemBytes<kPad> / kHeadsPerCta)) + k_at<kPad>(row, unit);
        const int t = q0 + row;
        if (t >= 0 && t < width) {
          const int hh = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + head;
          cp_async16(dst, qs.q8 + (static_cast<std::int64_t>(t) * kQHeads + hh) * kHeadDim + unit * 16);
        } else {
          asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u));
        }
      }
    }
  };

  // Q K^T's A fragments as fa_kernel holds them (kQSmem: read per k step instead).
  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kQSmem ? 1 : kKSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h] : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
  if constexpr (!kQSmem) {
#pragma unroll
    for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = q0 + ((i & 1) ? tb : ta);
        const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
        qa[kk][i] = t < 0 || t >= width
                        ? 0u
                        : *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                                   16 * (i / 2) + 4 * (lane % 4));
      }
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;
  const int warp_last = start + q0 + warp * 16 + 15;

  float o[kHeadDim / 8][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row];
      const __half *po = carry.o + row * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  const int lm = lane / 8, lr = lane % 8;
  const std::uint32_t qrow = qbase + static_cast<std::uint32_t>(hl * (kQSmemBytes<kPad> / kHeadsPerCta));
  const int tiles_per_head = buf.capacity / kTile;

  // S of tile c's keys into si, from 1.5 2^23 (exact in int32).
  auto qk_tile = [&](int c, int (&si)[4][4]) {
    const std::uint32_t ks = sbase + stage_of(c);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = kMagic;
    if constexpr ((kDiag & kDiagNoQk) != 0) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          std::uint32_t bits = static_cast<std::uint32_t>(lane * 977 + nt * 31 + e);
          if constexpr (!kQSmem) bits ^= qa[nt][e];
          si[nt][e] = kMagic + ((static_cast<int>(bits) ^ (c * 131)) & 0x3FFF) - 0x2000;
        }
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < kKSteps; ++kk) {
        std::uint32_t a[4];
        if constexpr (kQSmem) {
          ldsm_x4(a, qrow + k_at<kPad>(warp * 16 + lr + 8 * (lm & 1), 2 * kk + (lm >> 1)));
        } else {
          a[0] = qa[kk][0];
          a[1] = qa[kk][1];
          a[2] = qa[kk][2];
          a[3] = qa[kk][3];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          std::uint32_t b[4];
          ldsm_x4(b, ks + k_at<kPad>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
          mma_s8(si[2 * p], a, b);
          mma_s8(si[2 * p + 1], a, b + 2);
        }
      }
    }
  };

  // Tile c's softmax from its S: m and l move on; P's u8 A fragment, each row's correction of O,
  // and the factor of the tile's P V.
  constexpr float kU8 = 256.0f / 255.0f;
  auto softmax = [&](int c, int key0, const int (&si)[4][4], std::uint32_t (&pf)[4], float (&corr)[2],
                     float &fv) {
    const float sk = buf.scale[g * tiles_per_head + c];
    const float sv = buf.scale[(kKvHeads + g) * tiles_per_head + c];
    fv = __fmul_rn(sv, kU8);
    int pi[2];
    if constexpr ((kDiag & kDiagNoSoftmax) != 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        pf[r] = static_cast<std::uint32_t>(si[0][2 * r] ^ si[1][2 * r + 1]) & 0x7F7F7F7Fu;
        pf[2 + r] = static_cast<std::uint32_t>(si[2][2 * r] ^ si[3][2 * r + 1]) & 0x7F7F7F7Fu;
        pi[r] = static_cast<int>(pf[r] & 0xFFFFu);
        corr[r] = 1.0f;
        l[r] = l[r] + static_cast<float>(pi[r]);
      }
    } else {
      float sc[4][4];
      const float fs[2] = {__fmul_rn(f[0], sk), __fmul_rn(f[1], sk)};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nt][e] = __fmul_rn(__fsub_rn(__int_as_float(si[nt][e]), kMagicF), fs[e / 2]);
      }
      const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
      if (edge) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int col = nt * 8 + c2 + (e & 1);
            if (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb)) sc[nt][e] = -CUDART_INF_F;
          }
        }
      }
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
      if constexpr ((kDiag & kDiagNoMaxSum) != 0) {
        mx[0] = sc[0][0];
        mx[1] = sc[0][2];
      } else {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) mx[e / 2] = fmaxf(mx[e / 2], sc[nt][e]);
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if constexpr ((kDiag & kDiagNoMaxSum) == 0) {
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
          mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        }
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -CUDART_INF_F ? 1.0f
                  : (kDiag & kDiagNoExp) != 0 ? __fsub_rn(m[r], mn)
                                               : ex2_approx(__fsub_rn(m[r], mn));
        m[r] = mn;
      }
      const float bias[2] = {__fsub_rn(8.0f, m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                             __fsub_rn(8.0f, m[1] == -CUDART_INF_F ? 0.0f : m[1])};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        std::uint32_t w[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          auto ex = [](float x) { return (kDiag & kDiagNoExp) != 0 ? x : ex2_approx(x); };
          const float a = __fmul_rn(255.0f, ex(__fsub_rn(__fadd_rn(sc[nt][2 * r], bias[r]), 8.0f)));
          const float b = __fmul_rn(255.0f, ex(__fsub_rn(__fadd_rn(sc[nt][2 * r + 1], bias[r]), 8.0f)));
          if constexpr ((kDiag & kDiagNoPack) != 0) {
            w[nt] = __float_as_uint(a) ^ __float_as_uint(b);
          } else {
            // rn(x) for 0 <= x < 2^22 is the low bits of x + 1.5 2^23: bytes [a, b] of the pair.
            w[nt] = __byte_perm(__float_as_uint(__fadd_rn(a, kMagicF)), __float_as_uint(__fadd_rn(b, kMagicF)), 0x0040);
          }
        }
        if constexpr ((kDiag & kDiagNoPack) != 0) {
          pf[r] = w[0] ^ w[1];
          pf[2 + r] = w[2] ^ w[3];
        } else {
          pf[r] = __byte_perm(w[0], w[1], 0x5410);
          pf[2 + r] = __byte_perm(w[2], w[3], 0x5410);
        }
        pi[r] = static_cast<int>(__dp4a(pf[2 + r], 0x01010101u, __dp4a(pf[r], 0x01010101u, 0u)));
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if constexpr ((kDiag & kDiagNoMaxSum) == 0) {
          pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 1);
          pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 2);
        }
        l[r] = __fmaf_rn(l[r], corr[r], __fmul_rn(static_cast<float>(pi[r]), kU8));
      }
    }
  };

  // O = O corr + P V for the tile staged at `stage`.
  auto pv = [&](std::uint32_t stage, const std::uint32_t (&pf)[4], const float (&corr)[2], float fv) {
    if constexpr ((kDiag & kDiagNoPv) == 0) {
      const std::uint32_t vs = sbase + stage + kI8KBytes<kPad>;
#pragma unroll
      for (int dp = 0; dp < kHeadDim / 16; ++dp) {
        int t[2][4];
#pragma unroll
        for (int x = 0; x < 2; ++x) t[x][0] = t[x][1] = t[x][2] = t[x][3] = kMagic;
        std::uint32_t b[4];
        ldsm_x4(b, vs + vt_at<kPad>(16 * dp + 8 * (lm / 2) + lr, lm % 2));
        mma_u8s8(t[0], pf, b);
        mma_u8s8(t[1], pf, b + 2);
#pragma unroll
        for (int x = 0; x < 2; ++x) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            if constexpr ((kDiag & kDiagCheapFold) != 0) {
              o[2 * dp + x][e] = __fadd_rn(o[2 * dp + x][e], __int_as_float(t[x][e]));
            } else {
              o[2 * dp + x][e] = __fmaf_rn(o[2 * dp + x][e], corr[e / 2],
                                           __fmul_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fv));
            }
          }
        }
      }
    }
  };
  // pv, with tile cn's Q K^T into sn interleaved: k step dp / 2, keys 16 (dp % 2) .. + 15.
  auto pv_qk = [&](std::uint32_t stage, const std::uint32_t (&pf)[4], const float (&corr)[2], float fv,
                   int cn, int (&sn)[4][4]) {
    const std::uint32_t vs = sbase + stage + kI8KBytes<kPad>;
    const std::uint32_t ks = sbase + stage_of(cn);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) sn[nt][0] = sn[nt][1] = sn[nt][2] = sn[nt][3] = kMagic;
    std::uint32_t a[4];
#pragma unroll
    for (int dp = 0; dp < kHeadDim / 16; ++dp) {
      int t[2][4];
      if constexpr ((kDiag & kDiagNoPv) == 0) {
#pragma unroll
        for (int x = 0; x < 2; ++x) t[x][0] = t[x][1] = t[x][2] = t[x][3] = kMagic;
        std::uint32_t b[4];
        ldsm_x4(b, vs + vt_at<kPad>(16 * dp + 8 * (lm / 2) + lr, lm % 2));
        mma_u8s8(t[0], pf, b);
        mma_u8s8(t[1], pf, b + 2);
      }
      if constexpr ((kDiag & kDiagNoQk) == 0) {
        const int kk = dp >> 1, p = dp & 1;
        if (p == 0) {
          if constexpr (kQSmem) {
            ldsm_x4(a, qrow + k_at<kPad>(warp * 16 + lr + 8 * (lm & 1), 2 * kk + (lm >> 1)));
          } else {
            a[0] = qa[kk][0];
            a[1] = qa[kk][1];
            a[2] = qa[kk][2];
            a[3] = qa[kk][3];
          }
        }
        std::uint32_t bk[4];
        ldsm_x4(bk, ks + k_at<kPad>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
        mma_s8(sn[2 * p], a, bk);
        mma_s8(sn[2 * p + 1], a, bk + 2);
      }
      if constexpr ((kDiag & kDiagNoPv) == 0) {
#pragma unroll
        for (int x = 0; x < 2; ++x) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            o[2 * dp + x][e] = __fmaf_rn(o[2 * dp + x][e], corr[e / 2],
                                         __fmul_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fv));
          }
        }
      }
    }
    if constexpr ((kDiag & kDiagNoQk) != 0) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          std::uint32_t bits = static_cast<std::uint32_t>(lane * 977 + nt * 31 + e);
          if constexpr (!kQSmem) bits ^= qa[nt][e];
          sn[nt][e] = kMagic + ((static_cast<int>(bits) ^ (cn * 131)) & 0x3FFF) - 0x2000;
        }
      }
    }
  };
  auto soft_pv = [&](int c, int key0, const int (&si)[4][4]) {
    std::uint32_t pf[4];
    float corr[2], fv;
    softmax(c, key0, si, pf, corr, fv);
    pv(stage_of(c), pf, corr, fv);
  };

  if constexpr (kFuseQk) {
    // Tiles cbeg .. cbeg + kStages - 2 ahead; tile cbeg's Q K^T alone; then iteration c waits for
    // tile c + 1, refills tile c - 1's stage, runs tile c's softmax and its P V with tile c + 1's
    // Q K^T inside. Keys rise with the column, so a warp past its last tile is past every later one.
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
      if (i == 0) issue_q();
      if (cbeg + i < cend) issue(cbeg + i);
      cp_commit();
    }
    int sn[4][4];
    if (cend > cbeg) {
      cp_wait<kStages - 2>();
      __syncthreads();
      if (cols.key(cbeg * kTile) <= warp_last) qk_tile(cbeg, sn);
    }
    for (int c = cbeg; c < cend; ++c) {
      cp_wait<kStages - 3>();
      __syncthreads();  // tile c + 1 is everyone's; every warp is done with tile c - 1's stage
      if (c + kStages - 1 < cend) issue(c + kStages - 1);
      cp_commit();
      const int key0 = cols.key(c * kTile);
      if (key0 <= warp_last) {
        std::uint32_t pf[4];
        float corr[2], fv;
        softmax(c, key0, sn, pf, corr, fv);
        if (c + 1 < cend && cols.key((c + 1) * kTile) <= warp_last) {
          pv_qk(stage_of(c), pf, corr, fv, c + 1, sn);
        } else {
          pv(stage_of(c), pf, corr, fv);
        }
      }
    }
  } else if constexpr (kPingPong) {
    // One barrier: iteration c refills tile c - 2's stage at its top (the late warp reads tile
    // c - 1's V^T during it). Two barriers: it refills tile c - 1's stage after its second.
    constexpr int kPre = kOneBarrier ? kStages - 2 : kStages - 1;
#pragma unroll
    for (int i = 0; i < kPre; ++i) {
      if (i == 0) issue_q();
      if (cbeg + i < cend) issue(cbeg + i);
      cp_commit();
    }
    std::uint32_t ppf[4] = {0u, 0u, 0u, 0u};
    float pcorr[2] = {1.0f, 1.0f}, pfv = 0.0f;
    std::uint32_t pstage = 0;
    bool held = false;  // the late warp: a tile whose P V is still to fold
    // Which warp of a scheduler runs late: a warp and its partner on the same scheduler differ
    // in it whether the schedulers take warps w and w + 4 or w and w + 1.
    const bool late = (hl ^ (warp & 1)) != 0;
    for (int c = cbeg; c < cend; ++c) {
      if constexpr (kOneBarrier) {
        cp_wait<kStages - 3>();
        __syncthreads();  // tile c is everyone's; every warp is done with tile c - 2's stage
        if (c + kStages - 2 < cend) issue(c + kStages - 2);
        cp_commit();
      } else {
        cp_wait<kStages - 2>();
        __syncthreads();
      }
      if (late && held) pv(pstage, ppf, pcorr, pfv);
      held = false;
      const int key0 = cols.key(c * kTile);
      if (key0 <= warp_last) {
        int si[4][4];
        qk_tile(c, si);
        std::uint32_t pf[4];
        float corr[2], fv;
        softmax(c, key0, si, pf, corr, fv);
        if (!late) {
          pv(stage_of(c), pf, corr, fv);
        } else {
#pragma unroll
          for (int i = 0; i < 4; ++i) ppf[i] = pf[i];
          pcorr[0] = corr[0];
          pcorr[1] = corr[1];
          pfv = fv;
          pstage = stage_of(c);
          held = true;
        }
      }
      if constexpr (!kOneBarrier) {
        __syncthreads();  // every warp is done with tile c - 1's stage
        if (c + kStages - 1 < cend) issue(c + kStages - 1);
        cp_commit();
      }
    }
    if (late && held) pv(pstage, ppf, pcorr, pfv);
  } else if constexpr (!kPipeQk) {
    constexpr int kPre = kOneBarrier ? kStages - 1 : kStages;
#pragma unroll
    for (int i = 0; i < kPre; ++i) {
      if (i == 0) issue_q();
      if (cbeg + i < cend) issue(cbeg + i);
      cp_commit();
    }
    for (int c = cbeg; c < cend; ++c) {
      if constexpr (kOneBarrier) {
        cp_wait<kStages - 2>();
        __syncthreads();  // tile c is everyone's; every warp is done with tile c - 1's stage
        if (c + kStages - 1 < cend) issue(c + kStages - 1);
        cp_commit();
      } else {
        cp_wait<kStages - 1>();
        __syncthreads();
      }
      const int key0 = cols.key(c * kTile);
      if (key0 <= warp_last) {
        int si[4][4];
        qk_tile(c, si);
        soft_pv(c, key0, si);
      }
      if constexpr (!kOneBarrier) {
        __syncthreads();  // every warp is done with this stage
        if (c + kStages < cend) issue(c + kStages);
        cp_commit();
      }
    }
  } else {
    // Tiles cbeg .. cbeg + kStages - 2 ahead; iteration c waits for tile c + 1 (whose Q K^T it
    // issues), refills tile c - 1's stage with tile c + kStages - 1, then runs tile c's softmax
    // and P V on the S the previous iteration issued.
#pragma unroll
    for (int i = 0; i < kStages - 1; ++i) {
      if (i == 0) issue_q();
      if (cbeg + i < cend) issue(cbeg + i);
      cp_commit();
    }
    int sn[4][4];
    if (cend > cbeg) {
      cp_wait<kStages - 2>();
      __syncthreads();
      if (cols.key(cbeg * kTile) <= warp_last) qk_tile(cbeg, sn);
    }
    for (int c = cbeg; c < cend; ++c) {
      int si[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) si[nt][e] = sn[nt][e];
      }
      cp_wait<kStages - 3>();
      __syncthreads();  // tile c + 1 is everyone's; every warp is done with tile c - 1's stage
      if (c + kStages - 1 < cend) issue(c + kStages - 1);
      cp_commit();
      if (c + 1 < cend && cols.key((c + 1) * kTile) <= warp_last) qk_tile(c + 1, sn);
      const int key0 = cols.key(c * kTile);
      if (key0 <= warp_last) soft_pv(c, key0, si);
    }
  }

  // The split's state, as fa_kernel leaves it.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0) {
      carry.m[row] = m[r];
      carry.l[row] = l[r];
    }
  }
}

// ---- PROTOTYPE (phase 2d): the i8 band with the head dim split over a warp pair -----------------
//
// fa_i8_kernel's math, row for row and tile for tile (the bench holds it to i8ref byte for byte),
// on twice the warps: a CTA of 16 warps for the same 64 queries of two heads, warps w and w ^ 8 a
// pair on the same 16 rows. Each runs Q K^T over its half of the 256 dims (4 of the 8 k steps)
// into an exact int32 partial S, the pair sums the two partials through shared memory (the integer
// sum is the S of the whole row, in any order), both run the same softmax on it, and each folds
// P V for its half of O. So a thread holds 64 fp32 of O and 16 registers of Q instead of 128 and
// 32, the kernel fits 128 registers, and a scheduler holds 4 warps instead of 2. The cost: the
// softmax twice, and 2 KiB of partial S a warp and tile through shared memory.
constexpr int kPairThreads = 2 * kFaThreads;                    // 16 warps
constexpr int kPairXBytes = kPairThreads / 32 * 32 * 16 * 4;    // partial S: 2 KiB a warp
template <int kStages, bool kQSmem>
constexpr int kPairSmemBytes = kStages * kI8TileBytes<false> + kPairXBytes + (kQSmem ? kQSmemBytes<false> : 0);

__device__ __forceinline__ void bar_named(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

template <int kStages, bool kOneBarrier, bool kQSmem, int kDiag = 0>
__global__ void __launch_bounds__(kPairThreads, 1)
    fa_i8_pair_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                      int splits, int resume, Carry carry) {
  static_assert(kStages >= 2 && (!kOneBarrier || kStages >= 3), "one barrier needs three stages");
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  constexpr int kTileB = kI8TileBytes<false>;
  constexpr int kHalfSteps = kHeadDim / 32 / 2;  // k steps of Q K^T a warp: 4
  constexpr int kHalfDt = kHeadDim / 8 / 2;      // n8 tiles of O a warp: 16
  const std::uint32_t sbase = smem_u32(fa_smem);
  const std::uint32_t xbase = sbase + static_cast<std::uint32_t>(kStages * kTileB);
  const std::uint32_t qbase = xbase + static_cast<std::uint32_t>(kPairXBytes);
  const int lead = start % kQ;
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;  // longest first
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;
  const int rend = min(start + q0 + kQ, start + width);
  const int tid = static_cast<int>(threadIdx.x);
  const int wid = tid / 32, lane = tid % 32;
  const int half = wid / 8;        // dims 128 half .. 128 half + 127
  const int rg = wid % 8;          // the pair's 16 rows: head rg / 4, rows 16 (rg % 4) ..
  const int warp = rg % 4;
  const int hl = rg / 4;
  const int g = static_cast<int>(blockIdx.y) / kHeadPairs;
  const int h = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + hl;
  const int n = (cols.before(rend) + kTile - 1) / kTile;
  const int cbeg = split * n / splits, cend = (split + 1) * n / splits;
  if (cend <= cbeg && resume != 0) return;

  auto stage_of = [](int c) { return static_cast<std::uint32_t>((c % kStages) * kTileB); };
  auto issue = [&](int c) {
    const std::uint32_t st = sbase + stage_of(c);
    const std::uint8_t *ksrc =
        buf.k + (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim;
    const std::uint8_t *vsrc =
        buf.v + (static_cast<std::int64_t>(g) * (buf.capacity / kTile) + c) * kVtTileBytes;
    cp_async16(st + row_swz<true>(tid >> 4, tid & 15), ksrc + tid * 16);
    cp_async16(st + kKTileBytes<true> + vt_off(tid >> 1, tid & 1), vsrc + tid * 16);
  };
  auto issue_q = [&]() {
    if constexpr (kQSmem) {
#pragma unroll
      for (int j = 0; j < kHeadsPerCta * kQ * 16 / kPairThreads; ++j) {
        const int u = tid + j * kPairThreads;
        const int head = u / (kQ * 16), row = (u / 16) % kQ, unit = u % 16;
        const std::uint32_t dst = qbase + static_cast<std::uint32_t>(head * kQ * kHeadDim) + row_swz<true>(row, unit);
        const int t = q0 + row;
        if (t >= 0 && t < width) {
          const int hh = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + head;
          cp_async16(dst, qs.q8 + (static_cast<std::int64_t>(t) * kQHeads + hh) * kHeadDim + unit * 16);
        } else {
          asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u));
        }
      }
    }
  };

  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kQSmem ? 1 : kHalfSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h] : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
  if constexpr (!kQSmem) {
#pragma unroll
    for (int k = 0; k < kHalfSteps; ++k) {
      const int kk = kHalfSteps * half + k;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = q0 + ((i & 1) ? tb : ta);
        const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
        qa[k][i] = t < 0 || t >= width
                       ? 0u
                       : *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                                  16 * (i / 2) + 4 * (lane % 4));
      }
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;
  const int warp_last = start + q0 + warp * 16 + 15;

  // This warp's half of O: n8 tiles kHalfDt half .. kHalfDt half + 15.
  float o[kHalfDt][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kHalfDt; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row];
      const __half *po = carry.o + row * kHeadDim + kHalfDt * 8 * half;
#pragma unroll
      for (int dt = 0; dt < kHalfDt; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  const int lm = lane / 8, lr = lane % 8;
  const std::uint32_t qrow = qbase + static_cast<std::uint32_t>(hl * kQ * kHeadDim);
  const std::uint32_t xmine = xbase + static_cast<std::uint32_t>(wid * 2048 + lane * 16);
  const std::uint32_t xpart = xbase + static_cast<std::uint32_t>((wid ^ 8) * 2048 + lane * 16);
  const int tiles_per_head = buf.capacity / kTile;
  constexpr float kU8 = 256.0f / 255.0f;

  auto tile_work = [&](int c, int key0) {
    const std::uint32_t ks = sbase + stage_of(c);
    const std::uint32_t vs = ks + kKTileBytes<true>;
    const float sk = buf.scale[g * tiles_per_head + c];
    const float sv = buf.scale[(kKvHeads + g) * tiles_per_head + c];
    // Q K^T over this warp's half of the dims; half 0 carries 1.5 2^23, so the pair's sum is S
    // from 1.5 2^23 as fa_i8_kernel's.
    int si[4][4];
    const int s0 = half == 0 ? kMagic : 0;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = s0;
    if constexpr ((kDiag & kDiagNoQk) != 0) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) si[nt][e] += ((lane * 977 + nt * 31 + e) ^ (c * 131)) & 0x1FFF;
      }
    } else {
#pragma unroll
      for (int k = 0; k < kHalfSteps; ++k) {
        const int kk = kHalfSteps * half + k;
        std::uint32_t a[4];
        if constexpr (kQSmem) {
          ldsm_x4(a, qrow + row_swz<true>(warp * 16 + lr + 8 * (lm & 1), 2 * kk + (lm >> 1)));
        } else {
          a[0] = qa[k][0];
          a[1] = qa[k][1];
          a[2] = qa[k][2];
          a[3] = qa[k][3];
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          std::uint32_t b[4];
          ldsm_x4(b, ks + row_swz<true>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
          mma_s8(si[2 * p], a, b);
          mma_s8(si[2 * p + 1], a, b + 2);
        }
      }
    }
    // The pair's partial S summed: each writes its own, then adds its partner's.
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      asm volatile("st.shared.v4.s32 [%0], {%1, %2, %3, %4};" ::"r"(xmine + nt * 512), "r"(si[nt][0]),
                   "r"(si[nt][1]), "r"(si[nt][2]), "r"(si[nt][3])
                   : "memory");
    }
    bar_named(1 + rg, 64);
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      int x0, x1, x2, x3;
      asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                   : "r"(xpart + nt * 512)
                   : "memory");
      si[nt][0] += x0;
      si[nt][1] += x1;
      si[nt][2] += x2;
      si[nt][3] += x3;
    }
    // The softmax, as fa_i8_kernel's (both warps of the pair run it on the same S).
    std::uint32_t pf[4];
    float corr[2];
    int pi[2];
    if constexpr ((kDiag & kDiagNoSoftmax) != 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        pf[r] = static_cast<std::uint32_t>(si[0][2 * r] ^ si[1][2 * r + 1]) & 0x7F7F7F7Fu;
        pf[2 + r] = static_cast<std::uint32_t>(si[2][2 * r] ^ si[3][2 * r + 1]) & 0x7F7F7F7Fu;
        pi[r] = static_cast<int>(pf[r] & 0xFFFFu);
        corr[r] = 1.0f;
        l[r] = l[r] + static_cast<float>(pi[r]);
      }
    } else {
      float sc[4][4];
      const float fs[2] = {__fmul_rn(f[0], sk), __fmul_rn(f[1], sk)};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nt][e] = __fmul_rn(__fsub_rn(__int_as_float(si[nt][e]), kMagicF), fs[e / 2]);
      }
      const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
      if (edge) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int col = nt * 8 + c2 + (e & 1);
            if (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb)) sc[nt][e] = -CUDART_INF_F;
          }
        }
      }
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) mx[e / 2] = fmaxf(mx[e / 2], sc[nt][e]);
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -CUDART_INF_F ? 1.0f : ex2_approx(__fsub_rn(m[r], mn));
        m[r] = mn;
      }
      const float bias[2] = {__fsub_rn(8.0f, m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                             __fsub_rn(8.0f, m[1] == -CUDART_INF_F ? 0.0f : m[1])};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        std::uint32_t w[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float a = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[nt][2 * r], bias[r]), 8.0f)));
          const float b = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[nt][2 * r + 1], bias[r]), 8.0f)));
          w[nt] = __byte_perm(__float_as_uint(__fadd_rn(a, kMagicF)), __float_as_uint(__fadd_rn(b, kMagicF)), 0x0040);
        }
        pf[r] = __byte_perm(w[0], w[1], 0x5410);
        pf[2 + r] = __byte_perm(w[2], w[3], 0x5410);
        pi[r] = static_cast<int>(__dp4a(pf[2 + r], 0x01010101u, __dp4a(pf[r], 0x01010101u, 0u)));
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 1);
        pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 2);
        l[r] = __fmaf_rn(l[r], corr[r], __fmul_rn(static_cast<float>(pi[r]), kU8));
      }
    }
    // P V for this warp's half of the dims.
    if constexpr ((kDiag & kDiagNoPv) == 0) {
      const float fv = __fmul_rn(sv, kU8);
#pragma unroll
      for (int j = 0; j < kHalfDt / 2; ++j) {
        const int dp = kHalfDt / 2 * half + j;
        int t[2][4];
#pragma unroll
        for (int x = 0; x < 2; ++x) t[x][0] = t[x][1] = t[x][2] = t[x][3] = kMagic;
        std::uint32_t b[4];
        ldsm_x4(b, vs + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
        mma_u8s8(t[0], pf, b);
        mma_u8s8(t[1], pf, b + 2);
#pragma unroll
        for (int x = 0; x < 2; ++x) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            o[2 * j + x][e] = __fmaf_rn(o[2 * j + x][e], corr[e / 2],
                                        __fmul_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fv));
          }
        }
      }
    } else {
      (void)sv;
    }
  };

  constexpr int kPre = kOneBarrier ? kStages - 1 : kStages;
#pragma unroll
  for (int i = 0; i < kPre; ++i) {
    if (i == 0) issue_q();
    if (cbeg + i < cend) issue(cbeg + i);
    cp_commit();
  }
  for (int c = cbeg; c < cend; ++c) {
    if constexpr (kOneBarrier) {
      cp_wait<kStages - 2>();
      __syncthreads();  // tile c is everyone's; every warp is done with tile c - 1's stage and S
      if (c + kStages - 1 < cend) issue(c + kStages - 1);
      cp_commit();
    } else {
      cp_wait<kStages - 1>();
      __syncthreads();
    }
    // Both warps of a pair skip a tile together (the same rows).
    const int key0 = cols.key(c * kTile);
    if (key0 <= warp_last) tile_work(c, key0);
    if constexpr (!kOneBarrier) {
      __syncthreads();  // every warp is done with this stage and its partner's S
      if (c + kStages < cend) issue(c + kStages);
      cp_commit();
    }
  }

  // The split's state: each warp its half of O; m and l (the same in both) from half 0.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim + kHalfDt * 8 * half;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHalfDt; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0 && half == 0) {
      carry.m[row] = m[r];
      carry.l[row] = l[r];
    }
  }
}

void check_launch(const char *what);

template <int kStages, bool kOneBarrier, bool kQSmem, int kDiag = 0>
void launch_fa_i8_pair(dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
                       PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = kPairSmemBytes<kStages, kQSmem>;
  static const cudaError_t attribute = cudaFuncSetAttribute(
      fa_i8_pair_kernel<kStages, kOneBarrier, kQSmem, kDiag>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  fa_i8_pair_kernel<kStages, kOneBarrier, kQSmem, kDiag><<<grid, kPairThreads, kSmem, stream>>>(
      qs, start, width, scale, cols, buf, splits, resume, carry);
  check_launch("band attention");
}

// ---- PROTOTYPE (sm8x dense prefill, phase 2e): the i8 band without 2d's bit identity ------------
//
// Phase 2e drops the rule that the i8 band match 5ea54bc6 byte for byte; the KLD gate scores each
// change instead. fa_i8f_kernel is fa_i8_kernel v0 (its conversions, two stages, two barriers) with
// three levers:
//   - kF, the fold group (32, 64 or 128 keys): V^T has one amax / 127 scale per group
//     (tile_vgroup_kernel), the softmax takes each row's max over the whole group before P is
//     quantized, so the group's kF / 32 u8 x s8 MMAs of a dim tile accumulate in one int32
//     accumulator and O is folded once a group instead of once a 32-key tile. The sum stays exact
//     from 1.5 2^23: |sum| <= kF 255 127 = 4,145,280 < 2^22 at kF = 128. K keeps its scale per
//     32-key tile (it folds into the score scale for free). kF = 32 is v0 byte for byte (the bench
//     checks it).
//   - kLazy: O and l carry a reference m_ref that moves only when a row's running max m passes it
//     by more than kLazyLog2 = 8 (log2 units: a factor 256). P is still rn(255 2^(s - m)) against
//     the running max, so its bytes are the eager path's; the group's P V and P sum enter O and l
//     times 2^(m - m_ref), in [1, 256], a per-row factor folded into the V scale. The fold is one
//     FADD (the magic) and one FFMA an element instead of three ops, and O is rescaled only on a
//     group whose max passes the threshold (warp-uniform). Exactness: every term of O and of l
//     carries the same factor 2^(-m_ref), so O / l -- what the carry stores and the merge reads --
//     does not depend on the reference, and the lazy path differs from the eager one by fp32
//     rounding only; 2^8 of headroom cannot overflow fp32 (l <= 2^16 keys x 256 x 2^8 = 2^32). The
//     carried l is brought back to the running max before it is stored, so the state means what
//     it meant.
//   - kSplitKV: one buffer for the group's K rows and one for its V^T, refilled at opposite points
//     of the group (V^T during Q K^T and the softmax, the next K during P V), two barriers a group:
//     half the shared memory of two stages, which kF = 128 needs (64 KiB a group, against about
//     100 KiB a CTA).
//   - kQs: the CTA's Q (two heads of 64 rows, 32 KiB) in shared memory, one ldmatrix a k step
//     shared by the group's kF / 32 tiles, instead of 32 registers a thread: the register file is
//     what every lever here runs into (O holds 128 of them).
// kDiag (the bench's --diag) as fa_i8_kernel's: kDiagNoPv, kDiagNoSoftmax, kDiagNoQk.
constexpr float kLazyLog2 = 8.0f;
template <int kF>
constexpr int kI8fGroupBytes = kF * kHeadDim * 2;  // a group's K rows and V^T, 8 bits each
constexpr int kI8fQBytes = kHeadsPerCta * kQ * kHeadDim;  // kQs: the CTA's Q, 32 KiB
template <int kF, bool kSplitKV, bool kQs = false>
constexpr int kI8fSmemBytes = (kSplitKV ? kI8fGroupBytes<kF> : 2 * kI8fGroupBytes<kF>) + (kQs ? kI8fQBytes : 0);

template <int kF, bool kLazy, bool kSplitKV, bool kQs, int kDiag = 0>
__global__ void __launch_bounds__(kFaThreads, 1)
    fa_i8f_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                  int splits, int resume, Carry carry) {
  static_assert(kF == 32 || kF == 64 || kF == 128, "a fold group is 1, 2 or 4 tiles");
  constexpr int kSub = kF / kTile;
  constexpr int kKSteps = kHeadDim / 32;
  constexpr int kKBytes = kF * kHeadDim;
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  const std::uint32_t sbase = smem_u32(fa_smem);
  const std::uint32_t qbase = sbase + static_cast<std::uint32_t>(kI8fSmemBytes<kF, kSplitKV, false>);
  const int lead = start % kQ;
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;  // longest first
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;
  const int rend = min(start + q0 + kQ, start + width);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = (tid / 32) % (kQ / 16), lane = tid % 32;
  const int g = static_cast<int>(blockIdx.y) / kHeadPairs;
  const int hl = tid / (kQ / 16 * 32);  // the CTA's head, 0 or 1
  const int h = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + hl;
  // This CTA's groups, then its split of them.
  const int n = (cols.before(rend) + kF - 1) / kF;
  const int gbeg = split * n / splits, gend = (split + 1) * n / splits;
  if (gend <= gbeg && resume != 0) return;

  auto stage_of = [](int gi) {
    return kSplitKV ? 0u : static_cast<std::uint32_t>((gi & 1) * kI8fGroupBytes<kF>);
  };
  auto issue_k = [&](int gi) {
    const std::uint32_t st = sbase + stage_of(gi);
    const std::uint8_t *ksrc =
        buf.k + (static_cast<std::int64_t>(g) * buf.capacity + gi * kF) * kHeadDim;
#pragma unroll
    for (int j = 0; j < kF * 16 / kFaThreads; ++j) {
      const int u = tid + j * kFaThreads;
      cp_async16(st + row_swz<true>(u >> 4, u & 15), ksrc + u * 16);
    }
  };
  auto issue_v = [&](int gi) {
    const std::uint32_t st = sbase + stage_of(gi) + static_cast<std::uint32_t>(kKBytes);
    const std::uint8_t *vsrc =
        buf.v + (static_cast<std::int64_t>(g) * (buf.capacity / kTile) + gi * kSub) * kVtTileBytes;
#pragma unroll
    for (int j = 0; j < kSub * 2; ++j) {
      const int u = tid + j * kFaThreads;  // 512 16-byte segments a tile
      cp_async16(st + static_cast<std::uint32_t>((u >> 9) * kVtTileBytes) + vt_off((u & 511) >> 1, u & 1),
                 vsrc + u * 16);
    }
  };
  // kQs: the CTA's two heads of 64 rows, laid out as K rows; a row outside the call is zeros.
  auto issue_q = [&]() {
    if constexpr (kQs) {
#pragma unroll
      for (int j = 0; j < kI8fQBytes / 16 / kFaThreads; ++j) {
        const int u = tid + j * kFaThreads;
        const int head = u / (kQ * 16), row = (u / 16) % kQ, unit = u % 16;
        const std::uint32_t dst = qbase + static_cast<std::uint32_t>(head * kQ * kHeadDim) + row_swz<true>(row, unit);
        const int t = q0 + row;
        if (t >= 0 && t < width) {
          const int hh = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + head;
          cp_async16(dst, qs.q8 + (static_cast<std::int64_t>(t) * kQHeads + hh) * kHeadDim + unit * 16);
        } else {
          asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u));
        }
      }
    }
  };

  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kQs ? 1 : kKSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h] : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
  if constexpr (!kQs) {
#pragma unroll
    for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = q0 + ((i & 1) ? tb : ta);
        const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
        qa[kk][i] = t < 0 || t >= width
                        ? 0u
                        : *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                                   16 * (i / 2) + 4 * (lane % 4));
      }
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;
  const int warp_last = start + q0 + warp * 16 + 15;
  const std::uint32_t qrow = qbase + static_cast<std::uint32_t>(hl * kQ * kHeadDim);

  float o[kHeadDim / 8][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row];
      const __half *po = carry.o + row * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  float mref[2] = {m[0], m[1]};  // kLazy: the reference O and l are held at
  const int lm = lane / 8, lr = lane % 8;
  const int tiles_per_head = buf.capacity / kTile;
  constexpr float kU8 = 256.0f / 255.0f;

  // S of group gi's tiles, from 1.5 2^23 (exact in int32); a tile wholly past the warp's last row
  // is left empty (the mask covers it). A k step's Q fragment serves every tile of the group.
  auto qk = [&](int gi, int (&si)[kSub][4][4]) {
    const std::uint32_t ks = sbase + stage_of(gi);
    bool act[kSub];
#pragma unroll
    for (int j = 0; j < kSub; ++j) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) si[j][nt][0] = si[j][nt][1] = si[j][nt][2] = si[j][nt][3] = kMagic;
      act[j] = j == 0 || cols.key((gi * kSub + j) * kTile) <= warp_last;
    }
    if constexpr ((kDiag & kDiagNoQk) != 0) {
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            std::uint32_t bits = static_cast<std::uint32_t>(lane * 977 + nt * 31 + e);
            if constexpr (!kQs) bits ^= qa[nt][e];
            si[j][nt][e] = kMagic + ((static_cast<int>(bits) ^ ((gi * kSub + j) * 131)) & 0x3FFF) - 0x2000;
          }
        }
      }
    } else {
#pragma unroll
      for (int kk = 0; kk < kKSteps; ++kk) {
        std::uint32_t a[4];
        if constexpr (kQs) {
          ldsm_x4(a, qrow + row_swz<true>(warp * 16 + lr + 8 * (lm & 1), 2 * kk + (lm >> 1)));
        } else {
          a[0] = qa[kk][0];
          a[1] = qa[kk][1];
          a[2] = qa[kk][2];
          a[3] = qa[kk][3];
        }
#pragma unroll
        for (int j = 0; j < kSub; ++j) {
          if (act[j]) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              std::uint32_t b[4];
              ldsm_x4(b, ks + static_cast<std::uint32_t>(j * kTile * kHeadDim) +
                             row_swz<true>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
              mma_s8(si[j][2 * p], a, b);
              mma_s8(si[j][2 * p + 1], a, b + 2);
            }
          }
        }
      }
    }
  };

  // Group gi's softmax from its S: m (and m_ref) and l move on; P's u8 A fragments, one per tile;
  // the fold's per-row factors (eager: O corr + (P V) fv; lazy: O + (P V) fv 2^(m - m_ref)).
  auto softmax = [&](int gi, const int (&si)[kSub][4][4], std::uint32_t (&pf)[kSub][4], float (&fa)[2],
                     float (&fb)[2]) {
    const float sv = buf.scale[(kKvHeads + g) * tiles_per_head + gi * kSub];
    const float fv = __fmul_rn(sv, kU8);
    unsigned pi[2] = {0u, 0u};
    if constexpr ((kDiag & kDiagNoSoftmax) != 0) {
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          pf[j][r] = static_cast<std::uint32_t>(si[j][0][2 * r] ^ si[j][1][2 * r + 1]) & 0x7F7F7F7Fu;
          pf[j][2 + r] = static_cast<std::uint32_t>(si[j][2][2 * r] ^ si[j][3][2 * r + 1]) & 0x7F7F7F7Fu;
          pi[r] += pf[j][r] & 0xFFFFu;
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        fa[r] = 1.0f;
        fb[r] = fv;
        l[r] = l[r] + static_cast<float>(pi[r]);
      }
    } else {
      float sc[kSub][4][4];
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
        const int c = gi * kSub + j;
        const float sk = buf.scale[g * tiles_per_head + c];
        const float fs[2] = {__fmul_rn(f[0], sk), __fmul_rn(f[1], sk)};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) sc[j][nt][e] = __fmul_rn(__fsub_rn(__int_as_float(si[j][nt][e]), kMagicF), fs[e / 2]);
        }
        const int key0 = cols.key(c * kTile);
        const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
        if (edge) {
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int col = nt * 8 + c2 + (e & 1);
              if (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb)) sc[j][nt][e] = -CUDART_INF_F;
            }
          }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) mx[e / 2] = fmaxf(mx[e / 2], sc[j][nt][e]);
        }
      }
      float corr[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -CUDART_INF_F ? 1.0f : ex2_approx(__fsub_rn(m[r], mn));
        m[r] = mn;
      }
      const float bias[2] = {__fsub_rn(8.0f, m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                             __fsub_rn(8.0f, m[1] == -CUDART_INF_F ? 0.0f : m[1])};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int j = 0; j < kSub; ++j) {
          std::uint32_t w[4];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const float a = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[j][nt][2 * r], bias[r]), 8.0f)));
            const float b = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[j][nt][2 * r + 1], bias[r]), 8.0f)));
            // rn(x) for 0 <= x < 2^22 is the low bits of x + 1.5 2^23: bytes [a, b] of the pair.
            w[nt] = __byte_perm(__float_as_uint(__fadd_rn(a, kMagicF)), __float_as_uint(__fadd_rn(b, kMagicF)), 0x0040);
          }
          pf[j][r] = __byte_perm(w[0], w[1], 0x5410);
          pf[j][2 + r] = __byte_perm(w[2], w[3], 0x5410);
          pi[r] = __dp4a(pf[j][2 + r], 0x01010101u, __dp4a(pf[j][r], 0x01010101u, pi[r]));
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 1);
        pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 2);
      }
      if constexpr (!kLazy) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          l[r] = __fmaf_rn(l[r], corr[r], __fmul_rn(static_cast<float>(static_cast<int>(pi[r])), kU8));
          fa[r] = corr[r];
          fb[r] = fv;
        }
      } else {
        (void)corr;
        // -inf - -inf is NaN, which passes no threshold: a row with no key yet keeps its reference.
        const bool need = __fsub_rn(m[0], mref[0]) > kLazyLog2 || __fsub_rn(m[1], mref[1]) > kLazyLog2;
        if (__any_sync(0xffffffffu, need)) {
          float cr[2];
#pragma unroll
          for (int r = 0; r < 2; ++r) {
            cr[r] = mref[r] == m[r] || m[r] == -CUDART_INF_F ? 1.0f : ex2_approx(__fsub_rn(mref[r], m[r]));
            l[r] = __fmul_rn(l[r], cr[r]);
            mref[r] = m[r];
          }
#pragma unroll
          for (int dt = 0; dt < kHeadDim / 8; ++dt) {
            o[dt][0] = __fmul_rn(o[dt][0], cr[0]);
            o[dt][1] = __fmul_rn(o[dt][1], cr[0]);
            o[dt][2] = __fmul_rn(o[dt][2], cr[1]);
            o[dt][3] = __fmul_rn(o[dt][3], cr[1]);
          }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const float fac = mref[r] == m[r] ? 1.0f : ex2_approx(__fsub_rn(m[r], mref[r]));
          l[r] = __fmaf_rn(__fmul_rn(static_cast<float>(static_cast<int>(pi[r])), kU8), fac, l[r]);
          fa[r] = 1.0f;
          fb[r] = __fmul_rn(fv, fac);
        }
      }
    }
  };

  // O = O fa + (P V) fb for group gi's V^T: one int32 accumulator over the group's tiles.
  auto pv = [&](int gi, const std::uint32_t (&pf)[kSub][4], const float (&fa)[2], const float (&fb)[2]) {
    if constexpr ((kDiag & kDiagNoPv) == 0) {
      const std::uint32_t vs = sbase + stage_of(gi) + static_cast<std::uint32_t>(kKBytes);
#pragma unroll
      for (int dp = 0; dp < kHeadDim / 16; ++dp) {
        int t[2][4];
#pragma unroll
        for (int x = 0; x < 2; ++x) t[x][0] = t[x][1] = t[x][2] = t[x][3] = kMagic;
#pragma unroll
        for (int j = 0; j < kSub; ++j) {
          std::uint32_t b[4];
          ldsm_x4(b, vs + static_cast<std::uint32_t>(j * kVtTileBytes) + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
          mma_u8s8(t[0], pf[j], b);
          mma_u8s8(t[1], pf[j], b + 2);
        }
#pragma unroll
        for (int x = 0; x < 2; ++x) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            if constexpr (kLazy) {
              o[2 * dp + x][e] = __fmaf_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fb[e / 2], o[2 * dp + x][e]);
            } else {
              o[2 * dp + x][e] = __fmaf_rn(o[2 * dp + x][e], fa[e / 2],
                                           __fmul_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fb[e / 2]));
            }
          }
        }
      }
    } else {
      (void)fa;
      (void)fb;
    }
  };

  if constexpr (!kSplitKV) {
    // v0's schedule a group at a time: two stages, the next but one refilled after the second barrier.
    issue_q();
    if (gend > gbeg) {
      issue_k(gbeg);
      issue_v(gbeg);
    }
    cp_commit();
    if (gend - gbeg > 1) {
      issue_k(gbeg + 1);
      issue_v(gbeg + 1);
    }
    cp_commit();
    for (int gi = gbeg; gi < gend; ++gi) {
      cp_wait<1>();
      __syncthreads();
      if (cols.key(gi * kF) <= warp_last) {
        int si[kSub][4][4];
        qk(gi, si);
        std::uint32_t pf[kSub][4];
        float fa[2], fb[2];
        softmax(gi, si, pf, fa, fb);
        pv(gi, pf, fa, fb);
      }
      __syncthreads();  // every warp is done with this stage
      if (gi + 2 < gend) {
        issue_k(gi + 2);
        issue_v(gi + 2);
      }
      cp_commit();
    }
  } else {
    // One K buffer, one V^T buffer: the top barrier makes K(gi) everyone's and frees V^T (every
    // warp is past P V of gi - 1); the second makes V^T(gi) everyone's and frees K.
    issue_q();
    if (gend > gbeg) issue_k(gbeg);
    cp_commit();
    for (int gi = gbeg; gi < gend; ++gi) {
      cp_wait<0>();
      __syncthreads();
      issue_v(gi);
      cp_commit();
      const bool work = cols.key(gi * kF) <= warp_last;
      std::uint32_t pf[kSub][4];
      float fa[2] = {1.0f, 1.0f}, fb[2] = {0.0f, 0.0f};
      if (work) {
        int si[kSub][4][4];
        qk(gi, si);
        softmax(gi, si, pf, fa, fb);
      }
      cp_wait<0>();
      __syncthreads();
      if (gi + 1 < gend) issue_k(gi + 1);
      cp_commit();
      if (work) pv(gi, pf, fa, fb);
    }
  }

  // The split's state, as fa_kernel leaves it (kLazy: l brought back to the running max).
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0) {
      carry.m[row] = m[r];
      carry.l[row] = kLazy && mref[r] != m[r] ? __fmul_rn(l[r], ex2_approx(__fsub_rn(mref[r], m[r]))) : l[r];
    }
  }
}

// PROTOTYPE (phase 2e): V^T of a pass in s8 with one amax / 127 scale per kF-key group (stored for
// each 32-key tile of it, so the attention reads it at the group's first tile), the segment order
// store_fp8_tile's; K stays per tile (tile_kernel<true, false, kProtoI8> with role K alone). A
// column past the pass's keys is a zero row.
template <int kF>
__global__ void __launch_bounds__(256) tile_vgroup_kernel(IgnisHqPromptKeys src, Columns cols, PassBuffers buf) {
  constexpr int kSub = kF / kTile;
  extern __shared__ __align__(16) std::uint8_t vg_smem[];
  auto *rows = reinterpret_cast<__nv_bfloat16 *>(vg_smem);  // kF rows of 256
  __shared__ std::int8_t signs[kHeadDim];
  __shared__ float warp_amax[8];
  __shared__ float group_scale;
  const int group = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const int key_l = i * kTile + warp * 4 + (lane >> 3);
    const int column = group * kF + key_l;
    ignis_hq_prompt::fetch_rows_warp(src, head, true, column < cols.count ? cols.key(column) : -1,
                                     rows + key_l * kHeadDim, signs);
  }
  __syncthreads();
  float mx = 0.0f;
  for (int u = tid; u < kF * kHeadDim / 8; u += 256) {
    const uint4 w = reinterpret_cast<const uint4 *>(rows)[u];
    const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
      mx = fmaxf(mx, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
  }
  mx = warp_max(mx);
  if (lane == 0) warp_amax[warp] = mx;
  __syncthreads();
  const int tiles = buf.capacity / kTile;
  if (tid == 0) {
    float amax = 0.0f;
    for (int w = 0; w < 8; ++w) amax = fmaxf(amax, warp_amax[w]);
    const float s = amax > 0.0f ? amax / kInt8Max : 1.0f;
    group_scale = s;
    for (int i = 0; i < kSub; ++i) buf.scale[(kKvHeads + head) * tiles + group * kSub + i] = s;
  }
  __syncthreads();
  const float inv = 1.0f / group_scale;
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const __nv_bfloat16 *tile = rows + i * kTile * kHeadDim;
    std::uint8_t *dst = buf.v + (static_cast<std::int64_t>(head) * tiles + group * kSub + i) * kVtTileBytes;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int seg = tid + k * 256, d = seg >> 1, hh = seg & 1;
      std::uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        std::uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int key_l = 16 * hh + 2 * j + (b & 1) + 8 * (b >> 1);
          word |= static_cast<std::uint32_t>(i8(__bfloat162float(tile[key_l * kHeadDim + d]) * inv)) << (8 * b);
        }
        w[j] = word;
      }
      *reinterpret_cast<uint4 *>(dst + seg * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

// ---- PROTOTYPE (phase 2e): the control, i8qk brought up to date ------------------------------------
//
// fa_kernel<true, false, kProtoI8Qk> (5ea54bc6: INT8 Q K^T, P V on the FP16 MMA with FP16
// accumulate per tile, V in FP16 rows) on fa_i8_kernel v0's schedule and conversions: S lifted
// from 1.5 2^23 by one FADD, the mask only on a tile that needs it, P = p packed two at a time. O
// and l are held in units of p (the carry's are 256 p), so each element's fold is one FFMA after
// its half -> float conversion: o = o corr + (P V).
constexpr int kI8qkStageBytes = kKTileBytes<true> + kTile * kHeadDim * 2;  // INT8 K, FP16 V: 24 KiB

template <int kDiag = 0>
__global__ void __launch_bounds__(kFaThreads, 1)
    fa_i8qk2_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                    int splits, int resume, Carry carry) {
  constexpr int kKSteps = kHeadDim / 32;
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  const std::uint32_t sbase = smem_u32(fa_smem);
  const int lead = start % kQ;
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;
  const int rend = min(start + q0 + kQ, start + width);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = (tid / 32) % (kQ / 16), lane = tid % 32;
  const int g = static_cast<int>(blockIdx.y) / kHeadPairs;
  const int h = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + tid / (kQ / 16 * 32);
  const int n = (cols.before(rend) + kTile - 1) / kTile;
  const int cbeg = split * n / splits, cend = (split + 1) * n / splits;
  if (cend <= cbeg && resume != 0) return;

  auto issue = [&](int c) {
    const std::uint32_t st = sbase + static_cast<std::uint32_t>((c & 1) * kI8qkStageBytes);
    const std::uint8_t *ksrc = buf.k + (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim;
    const std::int64_t at = (static_cast<std::int64_t>(g) * buf.capacity + c * kTile) * kHeadDim * 2;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int u = tid + j * kFaThreads;
      cp_async16(st + row_swz<true>(u >> 4, u & 15), ksrc + u * 16);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int u = tid + j * kFaThreads;
      cp_async16(st + kKTileBytes<true> + row_swz<false>(u >> 5, u & 31), buf.v + at + u * 16);
    }
  };

  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kKSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h] : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
#pragma unroll
  for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = q0 + ((i & 1) ? tb : ta);
      const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
      qa[kk][i] = t < 0 || t >= width
                      ? 0u
                      : *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                                 16 * (i / 2) + 4 * (lane % 4));
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;
  const int warp_last = start + q0 + warp * 16 + 15;

  float o[kHeadDim / 8][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};  // O and l in units of p
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row] * (1.0f / 256.0f);
      const __half *po = carry.o + row * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  const int lm = lane / 8, lr = lane % 8;
  const int tiles_per_head = buf.capacity / kTile;

  if (cend > cbeg) issue(cbeg);
  cp_commit();
  if (cend - cbeg > 1) issue(cbeg + 1);
  cp_commit();
  for (int c = cbeg; c < cend; ++c) {
    cp_wait<1>();
    __syncthreads();
    const std::uint32_t ks = sbase + static_cast<std::uint32_t>((c & 1) * kI8qkStageBytes);
    const std::uint32_t vs = ks + kKTileBytes<true>;
    const int key0 = cols.key(c * kTile);
    if (key0 <= warp_last) {
      int si[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = kMagic;
      if constexpr ((kDiag & kDiagNoQk) == 0) {
#pragma unroll
        for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            std::uint32_t b[4];
            ldsm_x4(b, ks + row_swz<true>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma_s8(si[2 * p], qa[kk], b);
            mma_s8(si[2 * p + 1], qa[kk], b + 2);
          }
        }
      }
      const float sk = buf.scale[g * tiles_per_head + c];
      const float fs[2] = {__fmul_rn(f[0], sk), __fmul_rn(f[1], sk)};
      float sc[4][4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nt][e] = __fmul_rn(__fsub_rn(__int_as_float(si[nt][e]), kMagicF), fs[e / 2]);
      }
      const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
      if (edge) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int col = nt * 8 + c2 + (e & 1);
            if (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb)) sc[nt][e] = -CUDART_INF_F;
          }
        }
      }
      float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) mx[e / 2] = fmaxf(mx[e / 2], sc[nt][e]);
      }
      float corr[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        const float mn = fmaxf(m[r], mx[r]);
        corr[r] = mn == -CUDART_INF_F ? 1.0f : ex2_approx(__fsub_rn(m[r], mn));
        m[r] = mn;
      }
      const float bias[2] = {__fsub_rn(8.0f, m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                             __fsub_rn(8.0f, m[1] == -CUDART_INF_F ? 0.0f : m[1])};
      // P = p as FP16 A fragments for keys 0-15 and 16-31, straight from the S accumulators.
      std::uint32_t pf[2][4];
      float ps[2] = {0.0f, 0.0f};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const std::uint32_t p2 = pack_half(ex2_approx(__fsub_rn(__fadd_rn(sc[nt][2 * r], bias[r]), 8.0f)),
                                             ex2_approx(__fsub_rn(__fadd_rn(sc[nt][2 * r + 1], bias[r]), 8.0f)));
          const float2 pr = half2_bits(p2);
          ps[r] += pr.x + pr.y;
          pf[nt / 2][(nt % 2) * 2 + r] = p2;
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 1);
        ps[r] += __shfl_xor_sync(0xffffffffu, ps[r], 2);
        l[r] = __fmaf_rn(l[r], corr[r], ps[r]);
      }
      if constexpr ((kDiag & kDiagNoPv) == 0) {
#pragma unroll
        for (int dp = 0; dp < kHeadDim / 16; ++dp) {
          std::uint32_t t[2][2] = {};
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            std::uint32_t b[4];
            ldsm_x4_trans(b, vs + row_swz<false>(16 * j + lr + 8 * (lm % 2), 2 * dp + lm / 2));
            mma_f16acc(t[0], pf[j], b);
            mma_f16acc(t[1], pf[j], b + 2);
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const float2 lo = half2_bits(t[x][0]), hi = half2_bits(t[x][1]);
            o[2 * dp + x][0] = __fmaf_rn(o[2 * dp + x][0], corr[0], lo.x);
            o[2 * dp + x][1] = __fmaf_rn(o[2 * dp + x][1], corr[0], lo.y);
            o[2 * dp + x][2] = __fmaf_rn(o[2 * dp + x][2], corr[1], hi.x);
            o[2 * dp + x][3] = __fmaf_rn(o[2 * dp + x][3], corr[1], hi.y);
          }
        }
      }
    }
    __syncthreads();
    if (c + 2 < cend) issue(c + 2);
    cp_commit();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0) {
      carry.m[row] = m[r];
      carry.l[row] = l[r] * 256.0f;
    }
  }
}

template <int kF, bool kLazy, bool kSplitKV, bool kQs, int kDiag = 0>
void launch_fa_i8f(dim3 grid, Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                   int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = kI8fSmemBytes<kF, kSplitKV, kQs>;
  static const cudaError_t attribute = cudaFuncSetAttribute(
      fa_i8f_kernel<kF, kLazy, kSplitKV, kQs, kDiag>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  fa_i8f_kernel<kF, kLazy, kSplitKV, kQs, kDiag><<<grid, kFaThreads, kSmem, stream>>>(qs, start, width, scale, cols,
                                                                                      buf, splits, resume, carry);
  check_launch("i8f attention");
}

template <int kDiag = 0>
void launch_fa_i8qk2(dim3 grid, Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                     int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = 2 * kI8qkStageBytes;
  static const cudaError_t attribute =
      cudaFuncSetAttribute(fa_i8qk2_kernel<kDiag>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  fa_i8qk2_kernel<kDiag><<<grid, kFaThreads, kSmem, stream>>>(qs, start, width, scale, cols, buf, splits, resume,
                                                              carry);
  check_launch("i8qk2 attention");
}

template <int kF>
void launch_tile_vgroup(IgnisHqPromptKeys src, Columns cols, PassBuffers buf, cudaStream_t stream) {
  constexpr int kSmem = kF * kHeadDim * 2;
  static const cudaError_t attribute =
      cudaFuncSetAttribute(tile_vgroup_kernel<kF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  tile_vgroup_kernel<kF><<<dim3(static_cast<unsigned>((cols.count + kF - 1) / kF), kKvHeads, 1), 256, kSmem, stream>>>(
      src, cols, buf);
  check_launch("V^T groups");
}

// ---- the merge: the splits combined, normalized and un-rotated, one warp per row ---------------

__global__ void __launch_bounds__(kRowThreads) merge_kernel(Carry carry, int width, int splits,
                                                            __nv_bfloat16 *out) {
  __shared__ std::int8_t signs[kHeadDim];
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int row = static_cast<int>(blockIdx.x) * (kRowThreads / 32) + static_cast<int>(threadIdx.x) / 32;
  if (row >= width * kQHeads) return;
  const std::int64_t split_stride = static_cast<std::int64_t>(width) * kQHeads;
  float mx = -CUDART_INF_F;
  for (int s = 0; s < splits; ++s) mx = fmaxf(mx, carry.m[s * split_stride + row]);
  __nv_bfloat16 *dst = out + static_cast<std::int64_t>(row) * kHeadDim;
  if (mx == -CUDART_INF_F) {
#pragma unroll
    for (int s = 0; s < 8; ++s) dst[s * 32 + lane] = __float2bfloat16(0.0f);
    return;
  }
  float reg[8] = {};
  float l = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const std::int64_t at = s * split_stride + row;
    const float ms = carry.m[at];
    if (ms == -CUDART_INF_F) continue;
    const float w = ex2_approx(ms - mx);
    const float share = carry.l[at] * w;
    l += share;
    const __half *po = carry.o + at * kHeadDim;
#pragma unroll
    for (int k = 0; k < 8; ++k) reg[k] += __half2float(po[k * 32 + lane]) * share;
  }
  const float inv = l > 0.0f ? 1.0f / l : 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) reg[k] *= inv;
  ninfer::ops::hq_ifwht256_sign(reg, signs, 0, lane);
#pragma unroll
  for (int k = 0; k < 8; ++k) dst[k * 32 + lane] = __float2bfloat16(reg[k]);
}

// ---- host ---------------------------------------------------------------------------------------

constexpr std::size_t kAlign = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

// PROTOTYPE (phase 2e): a pass's planes hold whole fold groups of up to 128 keys (fa_i8f_kernel),
// so a capacity rounds up to 128 keys; the passes still launch by their key count.
constexpr std::uint32_t kCapacityKeys = 128;
std::uint32_t round_up_tile(std::uint32_t keys) {
  return (keys + kCapacityKeys - 1) / kCapacityKeys * kCapacityKeys;
}

// The band pass's keys per band -- the FP16 fallback's band holds half as many in the same bytes
// -- and the most columns the window pass can hold.
std::uint32_t band_capacity(std::uint32_t visible_keys, std::uint32_t band_keys,
                            IgnisHqPromptPrecision precision) {
  const std::uint32_t keys =
      precision == IgnisHqPromptPrecision::kFp16 ? band_keys / 2 : band_keys;
  return std::min(round_up_tile(keys), round_up_tile(visible_keys));
}
std::uint32_t window_capacity(std::int32_t width, std::uint32_t visible_keys) {
  return round_up_tile(std::min<std::uint32_t>(
      visible_keys, static_cast<std::uint32_t>(kSinks + kRing + width)));
}

// The band area holds the FP8 band during the band pass and, before it, the window pass's BF16
// Q and keys: the window pass runs first, so the two never live together.
struct BandArea {
  std::size_t band_k, band_v, q16, window_k, window_v, bytes;
};
BandArea band_area(std::int32_t width, std::uint32_t visible_keys, std::uint32_t band_keys,
                   std::uint32_t window) {
  BandArea a{};
  // Either band's plane: 8-bit at band_keys, FP16 at half of them.
  const std::size_t fp8_plane = aligned(std::max(
      static_cast<std::size_t>(kKvHeads) *
          band_capacity(visible_keys, band_keys, IgnisHqPromptPrecision::kFp8) * kHeadDim,
      static_cast<std::size_t>(kKvHeads) *
          band_capacity(visible_keys, band_keys, IgnisHqPromptPrecision::kFp16) * kHeadDim * 2));
  a.band_k = 0;
  a.band_v = fp8_plane;
  a.q16 = 0;
  a.window_k = aligned(static_cast<std::size_t>(width) * kQHeads * kHeadDim * 2);
  const std::size_t bf16_plane = aligned(static_cast<std::size_t>(kKvHeads) * window * kHeadDim * 2);
  a.window_v = a.window_k + bf16_plane;
  a.bytes = std::max(2 * fp8_plane, a.window_v + bf16_plane);
  return a;
}

struct Workspace {
  // q_band: the band's Q, INT8 (the 8-bit band) or FP16 (the fallback), sized for FP16.
  ninfer::Tensor q_band, q_scale, area, band_scale, carry_o, carry_m, carry_l;
};

// One allocation order for the op and for its size query, so the two cannot drift.
// The same bytes for both band precisions: Q's buffer holds FP16, the band area either band.
template <class Allocator>
Workspace allocate(Allocator &arena, std::int32_t width, std::uint32_t visible_keys,
                   std::uint32_t band_keys) {
  const std::uint32_t band = band_capacity(visible_keys, band_keys, IgnisHqPromptPrecision::kFp8);
  const BandArea a = band_area(width, visible_keys, band_keys, window_capacity(width, visible_keys));
  Workspace w;
  w.q_band = arena.alloc(ninfer::DType::U8, {2 * kHeadDim, kQHeads, width, 1});
  w.q_scale = arena.alloc(ninfer::DType::FP32, {kQHeads, width, 1, 1});
  w.area = arena.alloc(ninfer::DType::U8, {static_cast<std::int32_t>(a.bytes), 1, 1, 1});
  w.band_scale = arena.alloc(ninfer::DType::FP32,
                             {static_cast<std::int32_t>(band / kIgnisHqPromptTileKeys), kKvHeads, 2, 1});
  w.carry_o = arena.alloc(ninfer::DType::FP16, {kHeadDim, kQHeads, width, kIgnisHqPromptMaxSplits});
  w.carry_m = arena.alloc(ninfer::DType::FP32, {kQHeads, width, kIgnisHqPromptMaxSplits, 1});
  w.carry_l = arena.alloc(ninfer::DType::FP32, {kQHeads, width, kIgnisHqPromptMaxSplits, 1});
  return w;
}

int sm_count() {
  static const int count = [] {
    int device = 0, n = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || n <= 0) {
      return 1;
    }
    return n;
  }();
  return count;
}

// Key splits: as few as fill the card's waves best, and never so many that a split attends
// fewer than 8 tiles of the history.
std::int32_t pick_splits(std::int32_t width, std::uint32_t visible_keys) {
  const std::int64_t ctas = static_cast<std::int64_t>((width + kQ - 1) / kQ) * kKvHeads * kHeadPairs;
  const std::int64_t tiles = (static_cast<std::int64_t>(visible_keys) + kTile - 1) / kTile;
  const auto most = static_cast<std::int32_t>(
      std::clamp<std::int64_t>(tiles / 8, 1, kIgnisHqPromptMaxSplits));
  const int sms = sm_count();
  std::int32_t best = 1;
  double best_cost = static_cast<double>((ctas + sms - 1) / sms);
  for (std::int32_t s = 2; s <= most; ++s) {
    const double cost = static_cast<double>((ctas * s + sms - 1) / sms) / s;
    if (cost < best_cost - 1e-9) {
      best = s;
      best_cost = cost;
    }
  }
  return best;
}

void require(bool ok, const char *what) {
  if (!ok) throw std::invalid_argument(std::string("ignis_hq_prompt_attention: ") + what);
}

void check_launch(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

template <bool kFp8, bool kHalf = false, int kProto = kProtoOff>
void launch_fa(dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
               PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = kFaSmemBytesP<kFp8, kProto>;
  static const cudaError_t attribute = cudaFuncSetAttribute(
      fa_kernel<kFp8, kHalf, kProto>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  fa_kernel<kFp8, kHalf, kProto><<<grid, kFaThreads, kSmem, stream>>>(
      qs, start, width, scale, cols, buf, splits, resume, carry);
  check_launch(kFp8 || kHalf ? "band attention" : "window attention");
}

// PROTOTYPE (phase 2d): the i8 kernel's levers, one instantiation each, so the bench walks them
// in order; the op runs kI8Default. Each row is the one before it plus one lever.
struct I8Variant {
  const char *name;
  int stages;
  bool one_barrier, pipe_qk, q_smem, ping_pong, pad, fuse_qk, one_head;
  bool pair;  // fa_i8_pair_kernel (stages, one_barrier and q_smem apply)
};
constexpr I8Variant kI8Variants[] = {
    {"conversions (2 stages, 2 barriers)", 2, false, false, false, false, false, false, false, false},
    {"+ one barrier, 2 stages", 2, true, false, false, false, false, false, false, false},
    {"+ one barrier, 3 stages", 3, true, false, false, false, false, false, false, false},
    {"+ one barrier, 4 stages", 4, true, false, false, false, false, false, false, false},
    {"QK of tile c+1 under the softmax, 3 stages", 3, true, true, false, false, false, false, false, false},
    {"QK under the softmax, 4 stages", 4, true, true, false, false, false, false, false, false},
    {"QK under the softmax, Q from shared memory, 4 stages", 4, true, true, true, false, false, false, false, false},
    {"ping-pong of each scheduler's two warps, 3 stages", 3, true, false, false, true, false, false, false, false},
    {"ping-pong, 4 stages", 4, true, false, false, true, false, false, false, false},
    {"conversions, padded tiles (2 stages, 2 barriers)", 2, false, false, false, false, true, false, false, false},
    {"ping-pong, 3 stages, padded tiles", 3, true, false, false, true, true, false, false, false},
    {"ping-pong, 4 stages, padded tiles", 4, true, false, false, true, true, false, false, false},
    {"QK of tile c+1 inside the P V loop, 3 stages", 3, true, false, false, false, false, true, false, false},
    {"QK inside the P V loop, 4 stages", 4, true, false, false, false, false, true, false, false},
    {"QK inside the P V loop, 4 stages, padded tiles", 4, true, false, false, false, true, true, false, false},
    {"QK inside the P V loop, 4 stages, Q from shared memory", 4, true, false, true, false, false, true, false, false},
    {"conversions, 3 stages, 2 barriers", 3, false, false, false, false, false, false, false, false},
    {"conversions, 4 stages, 2 barriers", 4, false, false, false, false, false, false, false, false},
    {"ping-pong, 3 stages, 2 barriers", 3, false, false, false, true, false, false, false, false},
    {"ping-pong, 4 stages, 2 barriers", 4, false, false, false, true, false, false, false, false},
    {"one head a CTA, 2 CTAs an SM (2 stages, 2 barriers)", 2, false, false, false, false, false, false, true, false},
    {"one head a CTA, 3 stages, 2 barriers", 3, false, false, false, false, false, false, true, false},
    {"one head a CTA, 3 stages, one barrier", 3, true, false, false, false, false, false, true, false},
    {"one head a CTA, 2 stages, 2 barriers, padded tiles", 2, false, false, false, false, true, false, true, false},
    {"head dim split over a warp pair, 16 warps (2 stages, 2 barriers)", 2, false, false, false, false, false, false, false, true},
    {"warp pair, 3 stages, one barrier", 3, true, false, false, false, false, false, false, true},
    {"warp pair, 3 stages, 2 barriers", 3, false, false, false, false, false, false, false, true},
    {"warp pair, Q from shared memory (2 stages, 2 barriers)", 2, false, false, true, false, false, false, false, true},
};
constexpr int kI8VariantCount = static_cast<int>(sizeof(kI8Variants) / sizeof(kI8Variants[0]));
constexpr int kI8Default = 0;

template <int kStages, bool kOneBarrier, bool kPipeQk, bool kQSmem, bool kPingPong, bool kPad, bool kFuseQk,
          bool kOneHead, int kDiag = 0>
void launch_fa_i8(dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
                  PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = kI8SmemBytes<kStages, kQSmem, kPad>;
  static const cudaError_t attribute =
      cudaFuncSetAttribute(fa_i8_kernel<kStages, kOneBarrier, kPipeQk, kQSmem, kPingPong, kPad, kFuseQk, kOneHead, kDiag>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  // kOneHead: twice the CTAs, one head each (blockIdx.y the head), of half the threads.
  const dim3 g = kOneHead ? dim3(grid.x, kKvHeads * kGroup) : grid;
  fa_i8_kernel<kStages, kOneBarrier, kPipeQk, kQSmem, kPingPong, kPad, kFuseQk, kOneHead, kDiag>
      <<<g, kOneHead ? kFaThreads / 2 : kFaThreads, kSmem, stream>>>(qs, start, width, scale, cols, buf, splits,
                                                                     resume, carry);
  check_launch("band attention");
}

template <int kVariant, int kDiag = 0>
void launch_fa_i8_variant(dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
                          PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr I8Variant v = kI8Variants[kVariant];
  if constexpr (v.pair) {
    launch_fa_i8_pair<v.stages, v.one_barrier, v.q_smem, kDiag>(grid, qs, start, width, scale, cols, buf, splits,
                                                                 resume, carry, stream);
  } else {
    launch_fa_i8<v.stages, v.one_barrier, v.pipe_qk, v.q_smem, v.ping_pong, v.pad, v.fuse_qk, v.one_head, kDiag>(
        grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
  }
}

void launch_i8(int variant, int diag, dim3 grid, Queries qs, int start, int width, float scale,
               Columns cols, PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
#ifdef IGNIS_HQ_PROMPT_DIAG
  // The bench's stage-disabled copies of the default variant.
  if (diag != 0) {
    switch (diag) {
      case kDiagNoPv: return launch_fa_i8_variant<kI8Default, kDiagNoPv>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoPv | kDiagNoSoftmax: return launch_fa_i8_variant<kI8Default, kDiagNoPv | kDiagNoSoftmax>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoQk: return launch_fa_i8_variant<kI8Default, kDiagNoQk>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoPv | kDiagNoSoftmax | kDiagNoQk: return launch_fa_i8_variant<kI8Default, kDiagNoPv | kDiagNoSoftmax | kDiagNoQk>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoExp: return launch_fa_i8_variant<kI8Default, kDiagNoExp>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoMaxSum: return launch_fa_i8_variant<kI8Default, kDiagNoMaxSum>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoPack: return launch_fa_i8_variant<kI8Default, kDiagNoPack>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagCheapFold: return launch_fa_i8_variant<kI8Default, kDiagCheapFold>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      case kDiagNoExp | kDiagNoMaxSum | kDiagNoPack: return launch_fa_i8_variant<kI8Default, kDiagNoExp | kDiagNoMaxSum | kDiagNoPack>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
      default: throw std::invalid_argument("ignis_hq_prompt_attention: no such i8 diagnostic");
    }
  }
#else
  if (diag != 0) throw std::invalid_argument("ignis_hq_prompt_attention: the i8 diagnostics are the bench's");
#endif
  switch (variant) {
#define IGNIS_I8_CASE(i) \
  case i: return launch_fa_i8_variant<i>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream);
    IGNIS_I8_CASE(0)
    IGNIS_I8_CASE(1)
    IGNIS_I8_CASE(2)
    IGNIS_I8_CASE(3)
    IGNIS_I8_CASE(4)
    IGNIS_I8_CASE(5)
    IGNIS_I8_CASE(6)
    IGNIS_I8_CASE(7)
    IGNIS_I8_CASE(8)
    IGNIS_I8_CASE(9)
    IGNIS_I8_CASE(10)
    IGNIS_I8_CASE(11)
    IGNIS_I8_CASE(12)
    IGNIS_I8_CASE(13)
    IGNIS_I8_CASE(14)
    IGNIS_I8_CASE(15)
    IGNIS_I8_CASE(16)
    IGNIS_I8_CASE(17)
    IGNIS_I8_CASE(18)
    IGNIS_I8_CASE(19)
    IGNIS_I8_CASE(20)
    IGNIS_I8_CASE(21)
    IGNIS_I8_CASE(22)
    IGNIS_I8_CASE(23)
    IGNIS_I8_CASE(24)
    IGNIS_I8_CASE(25)
    IGNIS_I8_CASE(26)
    IGNIS_I8_CASE(27)
#undef IGNIS_I8_CASE
    default: throw std::invalid_argument("ignis_hq_prompt_attention: no such i8 variant");
  }
}
static_assert(kI8VariantCount == 28, "launch_i8 names every variant");

// PROTOTYPE: which band a call runs. The op takes it from the precision and IGNIS_PROTO_BAND, the
// bench names it: kI8 is the i8 band on fa_i8_kernel, kI8Ref the same band on 5ea54bc6's
// fa_kernel<true, false, kProtoI8>, kept as the reference the bench holds kI8 to; phase 2e adds
// kI8F (fa_i8f_kernel: the fold group, the lazy rescale) and kI8Qk2 (fa_i8qk2_kernel, the control).
enum class Band { kFp8, kFp16, kI8, kI8Ref, kI8Qk, kI8F, kI8Qk2 };

// PROTOTYPE (phase 2e): the window pass's form. kBf16: as served (BF16, f32 accumulate). kW8: the
// band's own kernel and tiles (8-bit Q K^T; INT8 K per 32-key tile; P V as the band runs it: u8 x
// s8 for the i8 bands, FP16 for i8qk and i8qk2), so every key goes through one 8-bit pass. kW16:
// the FP16 band's kernel (FP16 rows, FP16 accumulate inside a tile, folded into fp32 per tile).
enum class Window { kBf16, kW8, kW16 };

// PROTOTYPE (phase 2e): a call's levers besides its band.
struct Proto {
  Window window = Window::kBf16;
  int fold = 32;                // kI8F: keys a fold group (32, 64, 128)
  bool lazy = false;            // kI8F: the lazy rescale
  bool split_kv = false;        // kI8F: one K and one V^T buffer instead of two stages
  bool q_smem = false;          // kI8F: Q in shared memory
  int i8_variant = kI8Default;  // kI8: phase 2d's variant
  int diag = 0;                 // kI8 or kI8F: the bench's stage cuts
};

// PROTOTYPE (phase 2e): the key splits 2d measured best on the RTX 5090, 2 up to 16K visible keys
// and 3 above (IGNIS_PROTO_SPLITS=rule, the bench's splits -1); every band takes them.
std::int32_t rule_splits(std::uint32_t visible_keys) { return visible_keys <= 16384 ? 2 : 3; }

// PROTOTYPE: the switches, read once; a value they do not know throws.
//   IGNIS_PROTO_BAND    off | i8 | i8ref | i8qk | i8f | i8qk2; off: the precision decides, else the
//                       band replaces the FP16 band.
//   IGNIS_PROTO_FOLD    32 | 64 | 128 (i8f's fold group; default 32)
//   IGNIS_PROTO_LAZY    0 | 1 (i8f's lazy rescale; default 0)
//   IGNIS_PROTO_SPLITKV 0 | 1 (i8f's one K and one V^T buffer; default 0; fold 128 needs 1)
//   IGNIS_PROTO_QSMEM   0 | 1 (i8f's Q in shared memory; default 0)
//   IGNIS_PROTO_WINDOW  off | w8 | w16 (the window pass; w8 needs an 8-bit band)
//   IGNIS_PROTO_SPLITS  auto | rule | 1..5 (auto: the op's pick_splits)
struct ProtoEnv {
  int band = -1;
  Proto proto;
  std::int32_t splits = 0;  // 0: pick_splits; -1: rule_splits; else the count
  bool any = false;
};
const ProtoEnv &proto_env() {
  static const ProtoEnv env = [] {
    auto get = [](const char *name) {
      const char *v = std::getenv(name);
      return std::string(v == nullptr ? "" : v);
    };
    ProtoEnv e;
    const std::string band = get("IGNIS_PROTO_BAND");
    if (band == "i8") e.band = static_cast<int>(Band::kI8);
    else if (band == "i8ref") e.band = static_cast<int>(Band::kI8Ref);
    else if (band == "i8qk") e.band = static_cast<int>(Band::kI8Qk);
    else if (band == "i8f") e.band = static_cast<int>(Band::kI8F);
    else if (band == "i8qk2") e.band = static_cast<int>(Band::kI8Qk2);
    else if (!band.empty() && band != "off")
      throw std::invalid_argument("IGNIS_PROTO_BAND must be off, i8, i8ref, i8qk, i8f or i8qk2, got '" + band + "'");
    const std::string fold = get("IGNIS_PROTO_FOLD");
    if (fold == "64") e.proto.fold = 64;
    else if (fold == "128") e.proto.fold = 128;
    else if (!fold.empty() && fold != "32")
      throw std::invalid_argument("IGNIS_PROTO_FOLD must be 32, 64 or 128, got '" + fold + "'");
    const std::string lazy = get("IGNIS_PROTO_LAZY");
    if (lazy == "1") e.proto.lazy = true;
    else if (!lazy.empty() && lazy != "0") throw std::invalid_argument("IGNIS_PROTO_LAZY must be 0 or 1, got '" + lazy + "'");
    const std::string skv = get("IGNIS_PROTO_SPLITKV");
    if (skv == "1") e.proto.split_kv = true;
    else if (!skv.empty() && skv != "0") throw std::invalid_argument("IGNIS_PROTO_SPLITKV must be 0 or 1, got '" + skv + "'");
    const std::string qsm = get("IGNIS_PROTO_QSMEM");
    if (qsm == "1") e.proto.q_smem = true;
    else if (!qsm.empty() && qsm != "0") throw std::invalid_argument("IGNIS_PROTO_QSMEM must be 0 or 1, got '" + qsm + "'");
    const std::string win = get("IGNIS_PROTO_WINDOW");
    if (win == "w8") e.proto.window = Window::kW8;
    else if (win == "w16") e.proto.window = Window::kW16;
    else if (!win.empty() && win != "off")
      throw std::invalid_argument("IGNIS_PROTO_WINDOW must be off, w8 or w16, got '" + win + "'");
    const std::string sp = get("IGNIS_PROTO_SPLITS");
    if (sp == "rule") e.splits = -1;
    else if (sp.size() == 1 && sp[0] >= '1' && sp[0] <= '5') e.splits = sp[0] - '0';
    else if (!sp.empty() && sp != "auto")
      throw std::invalid_argument("IGNIS_PROTO_SPLITS must be auto, rule or 1..5, got '" + sp + "'");
    if (e.band != static_cast<int>(Band::kI8F) && (e.proto.fold != 32 || e.proto.lazy || e.proto.split_kv || e.proto.q_smem))
      throw std::invalid_argument("IGNIS_PROTO_FOLD, _LAZY, _SPLITKV and _QSMEM are i8f's");
    if (e.proto.fold == 128 && !e.proto.split_kv)
      throw std::invalid_argument("IGNIS_PROTO_FOLD=128 needs IGNIS_PROTO_SPLITKV=1 (two 64 KiB stages do not fit)");
    if (e.proto.window == Window::kW8 && e.band < 0)
      throw std::invalid_argument("IGNIS_PROTO_WINDOW=w8 runs the window on an 8-bit band's kernel: set IGNIS_PROTO_BAND");
    e.any = e.band >= 0 || e.proto.window != Window::kBf16 || e.splits != 0;
    std::fprintf(stderr,
                 "ignis proto: IGNIS_PROTO_BAND=%s FOLD=%d LAZY=%d SPLITKV=%d QSMEM=%d WINDOW=%s SPLITS=%s\n",
                 band.empty() ? "off" : band.c_str(), e.proto.fold, e.proto.lazy ? 1 : 0, e.proto.split_kv ? 1 : 0,
                 e.proto.q_smem ? 1 : 0,
                 e.proto.window == Window::kW8 ? "w8" : e.proto.window == Window::kW16 ? "w16" : "off",
                 sp.empty() ? "auto" : sp.c_str());
    return e;
  }();
  return env;
}

// PROTOTYPE (phase 2e): fa_i8f_kernel by its levers. kF = 128 runs only with one K and one V^T
// buffer (two 64 KiB stages exceed a CTA's shared memory); the diagnostics cut stages of kI8fDiag.
struct I8fDiagOf {
  static constexpr int kF = 128;
  static constexpr bool kLazy = true;
  static constexpr bool kSplitKV = true;
  static constexpr bool kQs = true;
};
void launch_i8f(const Proto &p, dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
                PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
#define IGNIS_I8F(F, L, S, Q, D) \
  return launch_fa_i8f<F, L, S, Q, D>(grid, qs, start, width, scale, cols, buf, splits, resume, carry, stream)
  if (p.diag != 0) {
#ifdef IGNIS_HQ_PROMPT_DIAG
    constexpr int F = I8fDiagOf::kF;
    constexpr bool L = I8fDiagOf::kLazy, S = I8fDiagOf::kSplitKV, Q = I8fDiagOf::kQs;
    switch (p.diag) {
      case kDiagNoPv: IGNIS_I8F(F, L, S, Q, kDiagNoPv);
      case kDiagNoPv | kDiagNoSoftmax: IGNIS_I8F(F, L, S, Q, kDiagNoPv | kDiagNoSoftmax);
      case kDiagNoQk: IGNIS_I8F(F, L, S, Q, kDiagNoQk);
      case kDiagNoPv | kDiagNoSoftmax | kDiagNoQk: IGNIS_I8F(F, L, S, Q, kDiagNoPv | kDiagNoSoftmax | kDiagNoQk);
      default: throw std::invalid_argument("ignis_hq_prompt_attention: no such i8f diagnostic");
    }
#else
    throw std::invalid_argument("ignis_hq_prompt_attention: the i8f diagnostics are the bench's");
#endif
  }
  const int key = p.fold * 8 + (p.lazy ? 4 : 0) + (p.split_kv ? 2 : 0) + (p.q_smem ? 1 : 0);
  switch (key) {
    case 32 * 8 + 0: IGNIS_I8F(32, false, false, false, 0);
    case 32 * 8 + 4: IGNIS_I8F(32, true, false, false, 0);
    case 32 * 8 + 5: IGNIS_I8F(32, true, false, true, 0);
    case 64 * 8 + 4: IGNIS_I8F(64, true, false, false, 0);
    case 64 * 8 + 5: IGNIS_I8F(64, true, false, true, 0);
    case 64 * 8 + 2: IGNIS_I8F(64, false, true, false, 0);
    case 64 * 8 + 6: IGNIS_I8F(64, true, true, false, 0);
    case 64 * 8 + 3: IGNIS_I8F(64, false, true, true, 0);
    case 64 * 8 + 7: IGNIS_I8F(64, true, true, true, 0);
    case 128 * 8 + 6: IGNIS_I8F(128, true, true, false, 0);
    case 128 * 8 + 3: IGNIS_I8F(128, false, true, true, 0);
    case 128 * 8 + 7: IGNIS_I8F(128, true, true, true, 0);
    default: throw std::invalid_argument("ignis_hq_prompt_attention: no such i8f kernel instantiated");
  }
#undef IGNIS_I8F
}

// The op's workspace as raw pointers, laid out by allocate (the bench sizes its own).
struct WorkspacePtrs {
  std::uint8_t *q_band;
  float *q_scale;
  std::uint8_t *area;
  float *band_scale;
  __half *carry_o;
  float *carry_m;
  float *carry_l;
};

// The bench's per-pass timing: called before the first launch (kMarkBegin) and after each launch
// with what it ended. The op passes none.
enum PassMark : int { kMarkBegin, kMarkPrep, kMarkWindowTiles, kMarkWindowFa, kMarkBandTiles, kMarkBandFa, kMarkMerge };
struct PassHook {
  void (*mark)(void *ctx, int what);
  void *ctx;
};

// PROTOTYPE (phase 2d): the chunk's passes -- Q's preparation, the window pass, the band passes
// and the merge -- as the op ran them inline before, on its workspace; the op and the bench both
// run them through here. `window_pass`: the residual window is on (the chunk attends first).
void attend(const __nv_bfloat16 *q, const IgnisHqPromptKeys &src, bool window_pass,
            std::int32_t start_position, std::int32_t width, float scale, Band band_kind,
            std::uint32_t band_keys, std::int32_t splits_option, const WorkspacePtrs &ws,
            __nv_bfloat16 *out, cudaStream_t stream, const PassHook *hook = nullptr,
            const Proto &proto = Proto{}) {
  auto mark = [&](int what) {
    if (hook != nullptr) hook->mark(hook->ctx, what);
  };
  const auto visible = static_cast<std::uint32_t>(start_position + width);
  const bool half_band = band_kind == Band::kFp16;
  const bool i8qk_band = band_kind == Band::kI8Qk || band_kind == Band::kI8Qk2;
  const Window wmode = window_pass ? proto.window : Window::kBf16;
  if (wmode == Window::kW8 && (band_kind == Band::kFp16 || band_kind == Band::kFp8)) {
    throw std::invalid_argument("ignis_hq_prompt_attention: the W8 window runs on an i8 band's kernel");
  }
  // The band's keys: the FP16 rows (FP16 band, i8qk's V) fit half of band_keys.
  const std::uint32_t band = band_capacity(
      visible, band_keys,
      half_band || i8qk_band ? IgnisHqPromptPrecision::kFp16 : IgnisHqPromptPrecision::kFp8);
  const std::uint32_t window = window_capacity(width, visible);
  const BandArea area = band_area(width, visible, band_keys, window);
  std::uint8_t *area_base = ws.area;

  // Q: INT8 for an 8-bit band (and the W8 window), FP16 over 16 for the FP16 band or the W16
  // window (in the window's Q slot when the band is 8-bit), BF16 for the window as served.
  Queries qs{};
  qs.q8 = half_band ? nullptr : ws.q_band;
  qs.q_scale = ws.q_scale;
  qs.qh = half_band ? reinterpret_cast<const __half *>(ws.q_band)
          : wmode == Window::kW16 ? reinterpret_cast<const __half *>(area_base + area.q16)
                                  : nullptr;
  qs.q16 = window_pass && wmode == Window::kBf16 ? reinterpret_cast<const __nv_bfloat16 *>(area_base + area.q16)
                                                 : nullptr;
  const int rows = width * kQHeads;
  const int row_blocks = (rows + kRowThreads / 32 - 1) / (kRowThreads / 32);
  mark(kMarkBegin);
  prep_q_kernel<<<row_blocks, kRowThreads, 0, stream>>>(
      q, rows, const_cast<std::uint8_t *>(qs.q8), ws.q_scale, const_cast<__half *>(qs.qh),
      const_cast<__nv_bfloat16 *>(qs.q16));
  check_launch("prep_q");
  mark(kMarkPrep);

  const std::int32_t splits = splits_option > 0    ? splits_option
                              : splits_option < 0 ? rule_splits(visible)
                                                  : pick_splits(width, visible);
  const Carry carry{ws.carry_o, ws.carry_m, ws.carry_l};
  // Up to one more query tile than width / 64: the first one may start mid-block.
  const dim3 fa_grid(static_cast<unsigned>(((width + kQ - 1) / kQ + 1) * splits),
                     static_cast<unsigned>(kKvHeads * kHeadPairs));
  int resume = 0;

  // An 8-bit pass (a band, or the window under W8) on band_kind's tiles and kernel.
  auto pass8 = [&](const Columns &cols, const PassBuffers &buf, int mark_tiles, int mark_fa) {
    const unsigned tiles = static_cast<unsigned>((cols.count + kTile - 1) / kTile);
    if (i8qk_band) {
      tile_kernel<true, false, kProtoI8Qk><<<dim3(tiles, kKvHeads, 2), 256, 0, stream>>>(src, cols, buf, 0);
    } else if (band_kind == Band::kI8F && proto.fold != 32) {
      tile_kernel<true, false, kProtoI8><<<dim3(tiles, kKvHeads, 1), 256, 0, stream>>>(src, cols, buf, 0);
      check_launch("K tiles");
      if (proto.fold == 64) {
        launch_tile_vgroup<64>(src, cols, buf, stream);
      } else {
        launch_tile_vgroup<128>(src, cols, buf, stream);
      }
    } else {
      tile_kernel<true, false, kProtoI8><<<dim3(tiles, kKvHeads, 2), 256, 0, stream>>>(src, cols, buf, 0);
    }
    check_launch("8-bit tiles");
    mark(mark_tiles);
    switch (band_kind) {
      case Band::kI8:
        launch_i8(proto.i8_variant, proto.diag, fa_grid, qs, start_position, width, scale, cols, buf, splits,
                  resume, carry, stream);
        break;
      case Band::kI8Ref:
        launch_fa<true, false, kProtoI8>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume,
                                         carry, stream);
        break;
      case Band::kI8Qk:
        launch_fa<true, false, kProtoI8Qk>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume,
                                           carry, stream);
        break;
      case Band::kI8F:
        launch_i8f(proto, fa_grid, qs, start_position, width, scale, cols, buf, splits, resume, carry, stream);
        break;
      case Band::kI8Qk2:
        launch_fa_i8qk2(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume, carry, stream);
        break;
      default:
        throw std::invalid_argument("ignis_hq_prompt_attention: not an 8-bit prototype band");
    }
    mark(mark_fa);
  };

  // The window pass: the sinks, then the ring before the chunk and the chunk, in BF16 (or W8, W16).
  const auto sinks = static_cast<std::int32_t>(std::min<std::uint32_t>(kSinks, visible));
  const std::int32_t ring_from = std::max(kSinks, start_position - kRing);
  if (window_pass) {
    const Columns cols{0, sinks, ring_from,
                       sinks + std::max(0, static_cast<std::int32_t>(visible) - ring_from)};
    if (wmode == Window::kW8) {
      // 8-bit K (and s8 V^T, or i8qk's FP16 V rows) in the BF16 planes' room; the scales after K.
      const PassBuffers buf{area_base + area.window_k, area_base + area.window_v,
                            reinterpret_cast<float *>(area_base + area.window_k +
                                                      static_cast<std::size_t>(kKvHeads) * window * kHeadDim),
                            static_cast<std::int32_t>(window)};
      pass8(cols, buf, kMarkWindowTiles, kMarkWindowFa);
    } else {
      const PassBuffers buf{area_base + area.window_k, area_base + area.window_v, nullptr,
                            static_cast<std::int32_t>(window)};
      const dim3 tiles(static_cast<unsigned>((cols.count + kTile - 1) / kTile), kKvHeads, 2);
      if (wmode == Window::kW16) {
        tile_kernel<false, true><<<tiles, 256, 0, stream>>>(src, cols, buf, 0);
      } else {
        tile_kernel<false><<<tiles, 256, 0, stream>>>(src, cols, buf, 0);
      }
      check_launch("window tiles");
      mark(kMarkWindowTiles);
      if (wmode == Window::kW16) {
        launch_fa<false, true>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume, carry,
                               stream);
      } else {
        launch_fa<false>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume, carry, stream);
      }
      mark(kMarkWindowFa);
    }
    resume = 1;
  }

  // The band pass: every key only the codec holds, a band at a time, in 8 bits (or FP16).
  const std::int32_t codec_begin = window_pass ? sinks : 0;
  const std::int32_t codec_end =
      window_pass ? std::min(ring_from, static_cast<std::int32_t>(visible))
                  : static_cast<std::int32_t>(visible);
  for (std::int32_t begin = codec_begin; begin < codec_end; begin += static_cast<std::int32_t>(band)) {
    const Columns cols{begin, INT_MAX, 0, std::min(static_cast<std::int32_t>(band), codec_end - begin)};
    const PassBuffers buf{area_base + area.band_k, area_base + area.band_v, ws.band_scale,
                          static_cast<std::int32_t>(band)};
    const dim3 tiles(static_cast<unsigned>((cols.count + kTile - 1) / kTile), kKvHeads, 2);
    if (band_kind == Band::kFp16) {
      tile_kernel<false, true><<<tiles, 256, 0, stream>>>(src, cols, buf, 0);
      check_launch("band tiles");
      mark(kMarkBandTiles);
      launch_fa<false, true>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume,
                             carry, stream);
      mark(kMarkBandFa);
    } else if (band_kind == Band::kFp8) {
      tile_kernel<true><<<tiles, 256, 0, stream>>>(src, cols, buf, 0);
      check_launch("band tiles");
      mark(kMarkBandTiles);
      launch_fa<true>(fa_grid, qs, start_position, width, scale, cols, buf, splits, resume, carry,
                      stream);
      mark(kMarkBandFa);
    } else {
      pass8(cols, buf, kMarkBandTiles, kMarkBandFa);
    }
    resume = 1;
  }

  merge_kernel<<<row_blocks, kRowThreads, 0, stream>>>(carry, width, splits, out);
  check_launch("merge");
  mark(kMarkMerge);
}

} // namespace

// PROTOTYPE (phase 2d): the bench (kernel/tests/bench_sm8x_attn.cu) includes this file for its
// kernels and attend(), without the op's entry points, which link the vendored library.
#ifndef IGNIS_HQ_PROMPT_KERNELS_ONLY

IgnisHqPromptKeys ignis_hq_prompt_keys(const ninfer::PagedKVLayerView &cache,
                                       const ninfer::Tensor &k, const ninfer::Tensor &v,
                                       std::int32_t start_position) {
  IgnisHqPromptKeys keys;
  keys.codes_k = cache.k_pages.data;
  keys.codes_v = cache.v_pages.data;
  keys.meta_k = cache.k_scale_pages.data;
  keys.meta_v = cache.v_scale_pages.data;
  keys.block_table = static_cast<const std::int32_t *>(cache.block_table.data);
  keys.residual_k = cache.residual_k.data;
  keys.residual_v = cache.residual_v.data;
  keys.ring_valid = static_cast<const std::uint32_t *>(cache.ring_valid.data);
  const bool attend_first = cache.residual_k.data != nullptr;
  keys.new_k = attend_first ? k.data : nullptr;
  keys.new_v = attend_first ? v.data : nullptr;
  keys.start = start_position;
  keys.width = k.ne[2];
  return keys;
}

void ignis_hq_prompt_fetch_rows(const IgnisHqPromptKeys &keys, bool values,
                                std::int32_t key_begin, std::int32_t count, void *dst,
                                cudaStream_t stream) {
  if (count <= 0) return;
  require(key_begin >= 0 && dst != nullptr, "fetch: a negative key or no destination");
  const Columns cols{key_begin, INT_MAX, 0, count};
  const PassBuffers buf{values ? nullptr : static_cast<std::uint8_t *>(dst),
                        values ? static_cast<std::uint8_t *>(dst) : nullptr, nullptr, count};
  tile_kernel<false><<<dim3(static_cast<unsigned>((count + kTile - 1) / kTile), kKvHeads, 1), 256, 0,
                       stream>>>(keys, cols, buf, values ? 1 : 0);
  check_launch("fetch rows");
}

std::size_t ignis_hq_prompt_workspace_bytes(std::int32_t width, std::uint32_t visible_keys,
                                            std::uint32_t band_keys) {
  ninfer::WorkspaceLayoutBuilder layout;
  (void)allocate(layout, width, visible_keys, band_keys);
  return layout.peak_bytes(1);
}

void ignis_hq_prompt_attention(const ninfer::Tensor &q, const ninfer::Tensor &k,
                               const ninfer::Tensor &v, const ninfer::Tensor &positions,
                               std::int32_t start_position, const ninfer::Tensor &gate,
                               float scale, const ninfer::PagedKVLayerView &cache,
                               ninfer::WorkspaceArena &workspace, ninfer::Tensor &out,
                               cudaStream_t stream, const IgnisHqPromptOptions &options) {
  const std::int32_t width = q.ne[2];
  require(cache.dtype == ninfer::DType::U8 && cache.quant_group == 32 &&
              cache.num_kv_heads == kKvHeads && cache.head_dim == kHeadDim,
          "the cache is not an hq-e8-2b cache of 4 KV heads of 256");
  require(q.dtype == ninfer::DType::BF16 && q.ne[0] == kHeadDim && q.ne[1] == kQHeads &&
              width >= 1 && q.ne[3] == 1 && q.is_contiguous(),
          "q must be BF16 [256, 24, T]");
  require(k.dtype == ninfer::DType::BF16 && v.dtype == ninfer::DType::BF16 &&
              k.ne[0] == kHeadDim && k.ne[1] == kKvHeads && k.ne[2] == width &&
              v.ne[0] == kHeadDim && v.ne[1] == kKvHeads && v.ne[2] == width &&
              k.is_contiguous() && v.is_contiguous(),
          "k and v must be BF16 [256, 4, T]");
  require(out.dtype == ninfer::DType::BF16 && out.numel() == q.numel() && out.is_contiguous(),
          "out must be BF16 [256, 24, T]");
  require(positions.dtype == ninfer::DType::I32 && positions.ne[0] == width,
          "positions must be I32 [T]");
  require(start_position >= 0 &&
              static_cast<std::int64_t>(start_position) + width <=
                  static_cast<std::int64_t>(ninfer::ops::kGqaAttentionMaximumVisibleKeys),
          "the chunk must end inside the attention's key envelope");
  require(options.band_keys >= kIgnisHqPromptTileKeys &&
              options.band_keys % kIgnisHqPromptTileKeys == 0,
          "band_keys must be a positive multiple of 32");
  require(options.splits >= 0 && options.splits <= kIgnisHqPromptMaxSplits, "splits out of range");
  require(cache.block_table.data != nullptr && cache.k_pages.data != nullptr &&
              cache.k_scale_pages.data != nullptr,
          "the cache view has no planes");

  const auto visible = static_cast<std::uint32_t>(start_position + width);
  auto scope = workspace.scope();
  const Workspace ws = allocate(workspace, width, visible, options.band_keys);
  // PROTOTYPE: IGNIS_PROTO_BAND takes the FP16 band's place, and only its place; the phase 2e
  // switches (window, fold, lazy rescale, splits) likewise run only there.
  const ProtoEnv &penv = proto_env();
  require(!penv.any || options.precision == IgnisHqPromptPrecision::kFp16,
          "IGNIS_PROTO_* replace the FP16 band: set IGNIS_MODEL_PREFILL_ATTENTION=fp16");
  const Band band = penv.band >= 0 ? static_cast<Band>(penv.band)
                    : options.precision == IgnisHqPromptPrecision::kFp16 ? Band::kFp16
                                                                          : Band::kFp8;
  const std::int32_t splits_option = penv.splits != 0 ? penv.splits : options.splits;

  // ADR 0037 (GitHub #258): with the residual window the chunk reads the ring as it stood
  // before the chunk, so it attends first and is appended after; without it, it is appended
  // first and read back through the codec.
  const bool window_pass = cache.residual_k.data != nullptr;
  if (!window_pass) ninfer::ops::gqa_kv_append(k, v, positions, cache, stream);

  const IgnisHqPromptKeys src = ignis_hq_prompt_keys(cache, k, v, start_position);
  const WorkspacePtrs ptrs{static_cast<std::uint8_t *>(ws.q_band.data),
                           static_cast<float *>(ws.q_scale.data),
                           static_cast<std::uint8_t *>(ws.area.data),
                           static_cast<float *>(ws.band_scale.data),
                           static_cast<__half *>(ws.carry_o.data),
                           static_cast<float *>(ws.carry_m.data),
                           static_cast<float *>(ws.carry_l.data)};
  attend(static_cast<const __nv_bfloat16 *>(q.data), src, window_pass, start_position, width, scale,
         band, options.band_keys, splits_option, ptrs, static_cast<__nv_bfloat16 *>(out.data), stream,
         nullptr, penv.proto);

  if (options.after_attention) options.after_attention(src);
  if (window_pass) ninfer::ops::gqa_kv_append(k, v, positions, cache, stream);
  ninfer::ops::sigmoid_mul(gate, out, stream);
}

#endif  // IGNIS_HQ_PROMPT_KERNELS_ONLY
