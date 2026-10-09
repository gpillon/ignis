// ignis kernel leaf: device and host helpers shared by the Flash-Next MoE ops -- OURS (see
// kernel/include/ignis_moe.h). Leaf-internal: geometry, the error channel, the slot check, the
// fixed-point accumulator, the tensor-core MMA and the 128-wide Hadamard on a warp.
#ifndef IGNIS_MOE_COMMON_CUH
#define IGNIS_MOE_COMMON_CUH

#include "ignis_moe.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace ignis_moe {

constexpr int kHidden = IGNIS_MOE_HIDDEN;
constexpr int kExperts = IGNIS_MOE_EXPERTS;
constexpr int kTopK = IGNIS_MOE_TOP_K;
constexpr int kInter = IGNIS_MOE_INTERMEDIATE;
constexpr int kGateUpOut = 2 * kInter;  // 1280

// ---- error channel (host) --------------------------------------------------------------------

void set_error(const std::string &message);
const char *last_error_cstr();
int32_t fail(const std::string &message);
// 0, or -1 with the launch error recorded, for the kernel(s) just enqueued.
int32_t check_launch(const char *op);

// ---- per-device preparation (host) -----------------------------------------------------------

// What ignis_moe_prepare records for a device: the register ticket kernel's grid (0: not
// prepared), the staged kernel's grid, and the cluster decode route's CTAs per cluster (0: the
// device runs none). require_prepared fails the op, naming it, on a device nobody prepared;
// nothing configures itself lazily, so an op's first call inside a stream capture is like any
// other.
struct DecodeLaunch {
  int grid = 0;
  int staged_grid = 0;
  int cluster_size = 0;
};
int32_t require_prepared(const char *op, DecodeLaunch *decode);
// Each translation unit's share of ignis_moe_prepare: its kernels' attributes.
int32_t prepare_router();
int32_t prepare_prefill();
int32_t prepare_decode(int *grid);
int32_t prepare_decode_staged(int *grid);
// The cluster route (moe_decode_cluster.cu): 16 CTAs per expert where the device co-schedules
// a token's ten such clusters, else 8, else 0 (the route is then refused); never fails.
void prepare_decode_clusters(int *cluster_size);
int32_t decode_clusters(int cluster_size, const __nv_bfloat16 *x, int tokens, const int32_t *ids,
                        const float *weights, const ignis_moe_slot *slots, long long *acc,
                        cudaStream_t stream);
// The staged kernel (moe_decode_staged.cu): the tickets route's 1..kStagedMaxTokens tokens.
struct DecodeCounters;
int32_t decode_staged(int grid, const __nv_bfloat16 *x, int tokens, const int32_t *ids, const float *weights,
                      const ignis_moe_slot *slots, DecodeCounters *counters, int cap, long long *gate_up,
                      long long *acc, unsigned long long *trace, cudaStream_t stream);

// ---- slots -----------------------------------------------------------------------------------

__host__ __device__ __forceinline__ bool valid_k2(uint32_t k2) {
  return k2 == 4u || k2 == 5u || k2 == 6u || k2 == 8u;
}

// An expert id the router could not have produced traps: it would index outside the slot table
// and the grouping buffers.
__device__ __forceinline__ void check_expert(int expert) {
  if (static_cast<unsigned>(expert) >= static_cast<unsigned>(kExperts)) { __trap(); }
}

// The slot of (expert, projection), trapped if it is not resident: a selected projection without
// a record is a residency bug, and reading it as weights would only hide it.
__device__ __forceinline__ ignis_moe_slot load_slot(const ignis_moe_slot *slots, int expert,
                                                    int projection) {
  check_expert(expert);
  const ignis_moe_slot s = slots[expert * 2 + projection];
  if (s.record == nullptr || !valid_k2(s.k2)) { __trap(); }
  return s;
}

// Runs f(std::integral_constant<int, K2>{}) for the slot's K class: the one place the four K
// instantiations are chosen.
template <typename F> __device__ __forceinline__ void dispatch_k2(uint32_t k2, F &&f) {
  switch (k2) {
  case 4: f(std::integral_constant<int, 4>{}); break;
  case 5: f(std::integral_constant<int, 5>{}); break;
  case 6: f(std::integral_constant<int, 6>{}); break;
  default: f(std::integral_constant<int, 8>{}); break;
  }
}

// A record's planes (layout.md §3): the trellis at offset 0, then suh [in] and svh [out] fp16.
struct RecordPlanes {
  const uint32_t *trellis;
  const __half *suh;
  const __half *svh;
};

__device__ __forceinline__ RecordPlanes record_planes(const ignis_moe_slot &slot, uint32_t in,
                                                      uint32_t out, uint32_t k2) {
  const char *base = static_cast<const char *>(slot.record);
  const __half *suh = reinterpret_cast<const __half *>(base + in * out / 16u * k2);
  return RecordPlanes{reinterpret_cast<const uint32_t *>(base), suh, suh + in};
}

// Bytes of a record's trellis tensor: in * out * K / 8.
__host__ __device__ __forceinline__ uint32_t trellis_bytes(uint32_t in, uint32_t out, uint32_t k2) {
  return in * out / 16u * k2;
}

// ---- the routed accumulator: int64 with 32 fractional bits -----------------------------------

// v * 2^32 as an integer. The float product is exact (power of two); the conversion rounds only
// the bits below 2^-32, so integer sums of these are exact and order-independent. A value the
// format cannot hold -- not finite, or |v| >= 2^31 -- traps: saturating or zeroing it would turn
// a broken activation into a finite wrong answer.
__device__ __forceinline__ long long to_fixed(float v) {
  if (!(fabsf(v) < 2147483648.0f)) { __trap(); }
  return __float2ll_rn(v * 4294967296.0f);
}

__device__ __forceinline__ float from_fixed(long long v) {
  return static_cast<float>(v) * 2.3283064365386963e-10f;  // 2^-32, one rounding to fp32
}

__device__ __forceinline__ void add_fixed(long long *acc, float v) {
  atomicAdd(reinterpret_cast<unsigned long long *>(acc), static_cast<unsigned long long>(to_fixed(v)));
}

// ---- tensor cores: mma.m16n8k16, fp16 inputs, fp32 accumulate --------------------------------

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&h);
}

// ---- cp.async -------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- the 128-wide Sylvester Hadamard on one warp ---------------------------------------------

// v holds elements 4 * lane .. 4 * lane + 3 of a 128-vector; on return, the vector times
// H128 / sqrt(128) (natural order: entry (i, j) = (-1)^popcount(i & j)).
__device__ __forceinline__ void warp_hadamard128(float (&v)[4]) {
  float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c;
  v[1] = b + d;
  v[2] = a - c;
  v[3] = b - d;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool upper = (lane & m) != 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float o = __shfl_xor_sync(0xFFFFFFFFu, v[i], m);
      v[i] = upper ? o - v[i] : v[i] + o;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= 0.08838834764831845f;  // 1 / sqrt(128)
}

// The power of two that brings a value bounded by `bound` into [2^13, 2^14): the fp16 operand
// scale. Exact to apply and to undo; 1 for a zero bound.
__device__ __forceinline__ float fp16_operand_scale(float bound) {
  if (!(bound > 0.0f)) return 1.0f;
  int e;
  frexpf(bound, &e);  // bound in [2^(e-1), 2^e)
  return ldexpf(1.0f, 14 - e);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, m);
  return v;
}

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

}  // namespace ignis_moe

#endif  // IGNIS_MOE_COMMON_CUH
