// The geometry, the PTX helpers and the pass structures the hq-e8-2b prompt attention's two
// translation units share (GitHub #510): kernel/src/hq_prompt_attention.cu, the FP8 and FP16 bands
// and the BF16 window, and kernel/src/hq_prompt_int8.cu, the integer band. Moved here from the
// former text for text. The integer band lives in a translation unit of its own because the
// existing kernels' sm_120a code moves with whatever else a translation unit instantiates, and
// those kernels stay as they were. In an anonymous namespace, as they were, so each unit's kernels
// keep their names; include it from those two files only.
#pragma once

#include "ignis_hq_prompt_attention.h"

#include "ops/kernel/gqa_attention_geometry.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

using Geometry = ninfer::ops::Gqa27Geometry;

constexpr int kHeadDim = 256;
constexpr int kQHeads = Geometry::QHeads;    // 24
constexpr int kKvHeads = Geometry::KVHeads;  // 4
constexpr int kGroup = Geometry::GroupSize;  // 6
constexpr int kTile = static_cast<int>(kIgnisHqPromptTileKeys);  // keys per tile
constexpr int kQ = 64;  // query tokens per CTA
constexpr int kHeadsPerCta = 2;
constexpr int kHeadPairs = kGroup / kHeadsPerCta;        // 3
constexpr int kFaThreads = kHeadsPerCta * kQ / 16 * 32;  // 256
constexpr int kVtTileBytes = kHeadDim * kTile;           // 8 KiB of E4M3 V^T
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kInt8Max = 127.0f;

static_assert(kGroup % kHeadsPerCta == 0, "a CTA's heads share one KV group");
static_assert(kFaThreads == 256, "the FA CTA is 8 warps");

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
// D += A B over INT8, int32 accumulate (exact): the band's Q K^T.
__device__ __forceinline__ void mma_s8(int *d, const std::uint32_t *a, const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// One INT8 code of x (|x| <= 127 by construction), as a byte.
__device__ __forceinline__ std::uint8_t i8(float x) {
  return static_cast<std::uint8_t>(static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(x)))));
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

// Where a pass's tiles live. 8-bit: k = INT8 [kv_head][capacity][256], v = E4M3 V^T
// [kv_head][capacity / 32][256 dims x 32 keys], scale = [role][kv_head][capacity / 32] (K's
// amax / 127, V's a power of two). BF16 / FP16: k, v = [kv_head][capacity][256], no scale.
struct PassBuffers {
  std::uint8_t *k;
  std::uint8_t *v;
  float *scale;
  std::int32_t capacity;  // columns per KV head
};

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

} // namespace
