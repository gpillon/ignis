// ignis kernel leaf: what the dense trellis linear's translation units share -- OURS
// (kernel/include/ignis_trellis_linear.h): the error channel, the per-device state
// ignis_trellis_prepare records, the workspace layout, the split policy and the launch
// parameters.
#ifndef IGNIS_TRELLIS_LINEAR_INTERNAL_CUH
#define IGNIS_TRELLIS_LINEAR_INTERNAL_CUH

#include "ignis_trellis_linear.h"
#include "pdl_launch.cuh"
#include "trellis_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace ignis_tl {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kDecodeMaxTokens = IGNIS_TRELLIS_DECODE_MAX_TOKENS;
// The activation scale's target: the largest rotated entry of a token's 128-input block lands
// in [2^(E - 1), 2^E). Both routes accumulate in fp16 and fold into fp32 with the block's inverse
// scale every 32 inputs, so the 32 products a fold sums, |a| < 2^9 times |w| <= 3.4531 (the mul1
// codebook's extremes), stay below 65504 whatever their signs: 32 * 512 * 3.4531 = 56,576.
constexpr int kScaleExp = 9;
// Tokens per prefill output tile.
constexpr int kPrefillRows = 128;
// SM ids the cap's admission bitmap covers.
constexpr int kMaxSmIds = 512;

// ---- error channel (host) --------------------------------------------------------------------

int32_t fail(const std::string &message);
int32_t check_launch(const char *op);

// ---- per-device state (host) -----------------------------------------------------------------

struct DeviceState {
  int sms = 0;               // 0: not prepared
  int decode_ctas[2][4] = {};  // resident CTAs per SM of the decode kernel, [K <= 4, any K][decode_class]
  int prefill_ctas = 0;      // resident CTAs per SM of the prefill kernel
  uint32_t cap = 0;          // the debug SM cap, 0 for none
  uint32_t admit[kMaxSmIds / 32] = {};  // SM ids whose CTAs run when capped
  int fq = 0;                // PROTOTYPE: IGNIS_PROTO_FAKEQUANT, a FakeQuant
};

// PROTOTYPE (sm8x dense prefill, phase 1): the prefill route's fake quant, IGNIS_PROTO_FAKEQUANT.
// Off leaves every route as it was; the others touch the prefill route (more than 64 tokens) only.
enum FakeQuant : int { kFqOff = 0, kFqW8 = 1, kFqW8A8Token = 2, kFqW8A8Block = 3 };
int32_t require_prepared(const char *op, DeviceState *state);
// Each kernel translation unit's share of ignis_trellis_prepare: its attributes and occupancy.
int32_t prepare_decode(DeviceState *state);
int32_t prepare_prefill(DeviceState *state);

// ---- workspace -------------------------------------------------------------------------------

// Byte offsets of a call's regions, 256-aligned: the counters (the ticket, the finished-CTA count,
// then one arrival count per output block), the rotated activations xr and their inverse scales,
// and (decode) the splits' partial sums.
//   xr       fp16. Decode: [weights][in / 128][tokens][128], each 128-block in fragment order
//            (rotate_kernel's frag_position); prefill: [tokens][in] row-major, one weight at a time.
//   inv      fp32 [weights][in / 128][tokens], an exact power of two.
//   partials fp32 [splits][tokens][weights * out]; every entry is written before it is read.
struct Layout {
  uint64_t arrivals = 0;
  uint64_t xr = 0;
  uint64_t inv = 0;
  uint64_t partials = 0;
  uint64_t bytes = 0;
  int splits = 1;
};

__host__ __device__ inline uint64_t align256(uint64_t v) { return (v + 255) / 256 * 256; }

// The arrival counts' room, in output blocks: one size for every call, so that every call's
// counts sit where every other call's do. The rotated activations follow the room; sized by the
// call's own blocks, it put a narrow call's activations over a wider call's counts, which that
// call then found nonzero and reduced its splits early -- a load's calls of several shapes share
// one workspace (GitHub #339). Bounds a call's out at IGNIS_TRELLIS_MAX_OUT.
constexpr uint32_t kArrivalBlocks = IGNIS_TRELLIS_MAX_OUT / 128;

// The decode route's split of the inputs: as many work units (output blocks x splits) as the
// kernel has resident CTAs on the card (or on the SMs the debug cap admits), so each CTA runs one
// unit and streams its blocks pipelined instead of paying a unit's latency several times; never
// below one 128-input block per split. The bits of a call depend on it, so they are the same run
// to run on one card and may differ between cards (as the split of any GEMM library).
inline int decode_splits(uint32_t in, uint32_t blocks_out, int resident) {
  const int nb = static_cast<int>(in / 128);
  int s = resident / static_cast<int>(blocks_out);
  if (s > nb) s = nb;
  return s < 1 ? 1 : s;
}

// The decode kernel's token class: up to 8, 16, 32 or 64 tokens (one, two, four or eight groups
// of eight, the MMA's N).
inline int decode_class(uint32_t tokens) { return tokens <= 8 ? 0 : tokens <= 16 ? 1 : tokens <= 32 ? 2 : 3; }

// Resident CTAs of the decode kernel that serves `tokens` (K <= 4 only, or `wide`).
inline int decode_resident(const DeviceState &st, uint32_t tokens, bool wide) {
  return st.decode_ctas[wide ? 1 : 0][decode_class(tokens)] * static_cast<int>(st.cap != 0 ? st.cap : st.sms);
}

inline Layout layout(uint32_t in, uint32_t out, uint32_t weights, uint32_t tokens, int resident) {
  Layout l;
  const uint64_t nb = in / 128;
  const bool decode = tokens <= static_cast<uint32_t>(kDecodeMaxTokens);
  const uint64_t xr_weights = decode ? weights : 1;
  l.arrivals = 256;
  l.xr = align256(l.arrivals + 4ull * kArrivalBlocks);
  l.inv = align256(l.xr + 2ull * xr_weights * tokens * in);
  l.partials = align256(l.inv + 4ull * xr_weights * nb * tokens);
  l.bytes = l.partials;
  if (decode) {
    l.splits = decode_splits(in, weights * (out / 128), resident);
    l.bytes = align256(l.partials + 4ull * l.splits * tokens * weights * out);
  }
  return l;
}

// ---- launch parameters -----------------------------------------------------------------------

struct Weight {
  const uint32_t *trellis;
  const __half *suh;
  const __half *svh;
  int k2;
  int stride;  // tiles per stored row
};

enum Epilogue : int { kStore = 0, kAdd = 1, kSwiGlu = 2, kSwiGluUp = 3 };

struct RotateParams {
  const __nv_bfloat16 *x;
  const __half *suh[2];
  int weights;
  int tokens;
  int in;
  int scale_exp;
  int frag_order;  // decode's fragment-order blocks, else row-major
  __half *xr;
  float *inv;
  int fq;          // PROTOTYPE: a FakeQuant; launch_rotate quantizes the prefill activations under A8
};

struct DecodeParams {
  Weight w[2];
  int weights;
  int tokens;
  int in;
  int out;
  int nb;         // in / 128
  int splits;
  int units;      // weights * (out / 128) * splits
  int epilogue;   // kStore, kAdd or kSwiGlu
  const __half *xr;
  const float *inv;
  float *partials;
  uint32_t *counters;  // [0] ticket, [1] finished CTAs
  uint32_t *arrivals;
  __nv_bfloat16 *y;
  int ldy;
  int capped;
  uint32_t admit[kMaxSmIds / 32];
};

struct PrefillParams {
  Weight w;
  int tokens;
  int in;
  int out;
  int nb;
  int tiles;      // ceil(tokens / kPrefillRows) * (out / 128)
  int epilogue;   // kStore, kAdd or kSwiGluUp
  const __half *xr;
  const float *inv;
  uint32_t *counters;
  __nv_bfloat16 *y;
  int ldy;
  int capped;
  uint32_t admit[kMaxSmIds / 32];
  int w8;         // PROTOTYPE: the host picks prefill_kernel<K2, true> (the W8 decode)
};

int32_t launch_rotate(const RotateParams &p, cudaStream_t stream);
int32_t launch_decode(const DeviceState &st, const DecodeParams &p, cudaStream_t stream);
int32_t launch_prefill(const DeviceState &st, const PrefillParams &p, cudaStream_t stream);

// ---- device helpers --------------------------------------------------------------------------

// Whether this CTA's SM is one the cap admits (always, uncapped).
__device__ __forceinline__ bool sm_admitted(int capped, const uint32_t (&admit)[kMaxSmIds / 32]) {
  if (!capped) return true;
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid < static_cast<uint32_t>(kMaxSmIds) && ((admit[smid >> 5] >> (smid & 31)) & 1u) != 0;
}

// Programmatic dependent launch: the rotation lets the kernel after it start, which issues its
// weight loads and then waits here for the rotated activations (none below sm_90, pdl_launch.cuh).
__device__ __forceinline__ void grid_dependency_wait() { ignis::pdl::wait(); }
__device__ __forceinline__ void grid_dependents_launch() { ignis::pdl::launch_dependents(); }

// Runs f(std::integral_constant<int, K2>{}) for a weight's K class: the one place the six K
// instantiations are chosen. A k2 the host did not admit traps.
template <typename F> __device__ __forceinline__ void dispatch_k2(int k2, F &&f) {
  switch (k2) {
  case 4: f(std::integral_constant<int, 4>{}); break;
  case 5: f(std::integral_constant<int, 5>{}); break;
  case 6: f(std::integral_constant<int, 6>{}); break;
  case 8: f(std::integral_constant<int, 8>{}); break;
  case 10: f(std::integral_constant<int, 10>{}); break;
  case 12: f(std::integral_constant<int, 12>{}); break;
  default: __trap();
  }
}

// A 4-byte cp.async (the activations' inverse scales, whose runs are not 16-byte aligned).
__device__ __forceinline__ void cp_async4(void *smem, const void *gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(gmem));
}

// m16n8k16, fp16 inputs and fp16 accumulation: both routes' MMA (folded into fp32 every 32 inputs).
__device__ __forceinline__ void mma_f16acc(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The weights' streaming read: used once per call, so it should not displace the activations,
// partial sums and counters from the L2.
__device__ __forceinline__ uint32_t load_stream(const uint32_t *p) { return __ldcs(p); }

// Four BF16 values at y (8-byte aligned), with the residual add when `add`.
__device__ __forceinline__ void store_bf16x4(__nv_bfloat16 *y, const float (&v)[4], bool add) {
  float o[4] = {v[0], v[1], v[2], v[3]};
  if (add) {
    const uint2 r = *reinterpret_cast<const uint2 *>(y);
    const __nv_bfloat162 r0 = *reinterpret_cast<const __nv_bfloat162 *>(&r.x);
    const __nv_bfloat162 r1 = *reinterpret_cast<const __nv_bfloat162 *>(&r.y);
    o[0] += __low2float(r0);
    o[1] += __high2float(r0);
    o[2] += __low2float(r1);
    o[3] += __high2float(r1);
  }
  const __nv_bfloat162 a = __floats2bfloat162_rn(o[0], o[1]);
  const __nv_bfloat162 b = __floats2bfloat162_rn(o[2], o[3]);
  uint2 packed;
  packed.x = *reinterpret_cast<const uint32_t *>(&a);
  packed.y = *reinterpret_cast<const uint32_t *>(&b);
  *reinterpret_cast<uint2 *>(y) = packed;
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16 *y, float (&v)[4]) {
  const uint2 r = *reinterpret_cast<const uint2 *>(y);
  const __nv_bfloat162 r0 = *reinterpret_cast<const __nv_bfloat162 *>(&r.x);
  const __nv_bfloat162 r1 = *reinterpret_cast<const __nv_bfloat162 *>(&r.y);
  v[0] = __low2float(r0);
  v[1] = __high2float(r0);
  v[2] = __low2float(r1);
  v[3] = __high2float(r1);
}

__device__ __forceinline__ void load_half4(const __half *p, float (&v)[4]) {
  const uint2 r = *reinterpret_cast<const uint2 *>(p);
  const __half2 h0 = *reinterpret_cast<const __half2 *>(&r.x);
  const __half2 h1 = *reinterpret_cast<const __half2 *>(&r.y);
  v[0] = __low2float(h0);
  v[1] = __high2float(h0);
  v[2] = __low2float(h1);
  v[3] = __high2float(h1);
}

}  // namespace ignis_tl

#endif  // IGNIS_TRELLIS_LINEAR_INTERNAL_CUH
