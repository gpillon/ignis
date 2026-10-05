// ignis kernel leaf: Flash-Next's routed experts for 1..8 decode tokens, in ONE launch for all
// experts and all four K -- OURS (kernel/include/ignis_moe.h).
//
// The launch is a persistent grid (two CTAs per SM) that takes work units by ticket from a
// counter in the workspace, each CTA one ticket ahead of the unit it is running. Every unit is
// the same shape: 640 inputs against one block of 128 output columns, one 16-column tile per
// warp, the tokens as MMA rows. Tickets [0, nA) are gate/up units, [nA, nA + nB) down units:
//
//   gate/up unit (expert u, column block cb in 0..9, k-split s in 0..3)
//       columns [128 cb, 128 cb + 128) of the fused plane (cb < 5: gate block cb; cb >= 5: up
//       block cb - 5) over inputs 640s .. 640s + 639. The CTA rotates the tokens' inputs (x o suh,
//       then the 128-wide Hadamard), scales each row by a power of two into fp16 range, and runs
//       m16n8k16 MMAs on weights decoded from the trellis straight into B fragments. Its
//       pre-rotation sums are added into an int64 fixed-point accumulator; the eighth arrival on
//       output block b (gate and up, four splits each) reads gate and up back, rotates them,
//       applies svh and SwiGLU, writes 128 entries of the expert's h, zeroes what it read and
//       bumps h_ready[u].
//   down unit (expert u, column block c in 0..19)
//       issues its weight loads, waits for h_ready[u] == 5, rotates h o suh_down the same way,
//       multiplies, applies the output Hadamard and svh, and adds weight x value for every token
//       that selected u into the fixed-point output accumulator.
//
// Memory-level parallelism is the point of the shape: a warp's tile is 64-128 contiguous bytes,
// so each lane loads ONE u32 word of each of its 40 tiles -- every load of the unit is in flight
// at once, in 40 registers -- and takes the two words its decode window needs from its
// neighbours with shuffles.
//
// A down ticket is only handed out after every gate/up ticket has been taken by a running CTA,
// and gate/up units wait on nothing, so the wait cannot deadlock whatever the residency.
// Deterministic by construction: every floating-point sum inside a unit has a fixed order, the
// cross-unit sums are integer, and nothing reads the slot address except to load from it. The
// last CTA out resets the counters, so the launch replays from a CUDA graph.

#include "moe_decode_common.cuh"
#include "trellis_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kSplitK = kHidden / kDecodeSplits;  // 640: the k range of every unit, both kinds
constexpr int kKTiles = kSplitK / 16;              // 40
constexpr int kAStride = kSplitK + 8;              // fp16 per A row, padded off the bank stride

struct Shared {
  int n_unique;
  int ticket;
  int is_last;
  int unique_id[kDecodeMaxUnique];
  uint32_t unique_sel[kDecodeMaxUnique];
  float unique_w[kDecodeMaxUnique][kDecodeMaxTokens];
  int slot_of[kDecodeMaxTokens * kTopK];
  int first[kDecodeMaxTokens * kTopK];
  float scale[kDecodeMaxTokens];
  float blockmax[kDecodeMaxTokens][kSplitK / 128];
  float xh[kDecodeMaxTokens][kSplitK];  // rotated rows; the epilogues' scratch afterwards
  __half a[kDecodeMaxTokens][kAStride];
};

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Rotate `tokens` rows of 640 inputs (input `in(t, k)` times suh[k]) into s.a as fp16, each
// row scaled by its own power of two (s.scale) so its largest entry sits in [2^13, 2^14).
template <typename In>
__device__ void prepare_a(Shared &s, int tokens, const __half *suh, In in) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int p = warp; p < tokens * (kSplitK / 128); p += kWarps) {
    const int t = p / (kSplitK / 128);
    const int blk = p % (kSplitK / 128);
    float v[4];
    float m = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = blk * 128 + 4 * lane + q;
      v[q] = in(t, k) * __half2float(suh[k]);
    }
    warp_hadamard128(v);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      s.xh[t][blk * 128 + 4 * lane + q] = v[q];
      m = fmaxf(m, fabsf(v[q]));
    }
    m = warp_max(m);
    if (lane == 0) s.blockmax[t][blk] = m;
  }
  __syncthreads();
  if (threadIdx.x < tokens) {
    float m = 0.0f;
    for (int blk = 0; blk < kSplitK / 128; ++blk) m = fmaxf(m, s.blockmax[threadIdx.x][blk]);
    s.scale[threadIdx.x] = fp16_operand_scale(m);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < tokens * kSplitK; i += blockDim.x) {
    const int t = i / kSplitK;
    const int k = i % kSplitK;
    s.a[t][k] = __float2half_rn(s.xh[t][k] * s.scale[t]);
  }
  __syncthreads();
}

// The lane's word of each of the warp's 40 tiles, tile kt at base + kt * stride u32 words.
template <int K2>
__device__ __forceinline__ void load_tiles(const uint32_t *base, int stride, uint32_t (&words)[kKTiles]) {
  constexpr int tile_words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kt = 0; kt < kKTiles; ++kt) words[kt] = lane < tile_words ? __ldg(base + kt * stride + lane) : 0u;
}

// acc += A(tokens x 640) . W(640 x 16) for the warp's tile, from the words load_tiles fetched.
template <int K2>
__device__ __forceinline__ void mma_tiles(const Shared &s, int tokens, const uint32_t (&words)[kKTiles],
                                          float (&acc)[2][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
  const bool row = g < tokens;
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
#pragma unroll
  for (int kt = 0; kt < kKTiles; ++kt) {
    const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, words[kt], plan.w0);
    const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, words[kt], plan.w1);
    uint32_t frag[4];
    ignis_trellis::decode_fragment<K2>(w0, w1, plan, frag);
    const int k = kt * 16 + 2 * c;
    uint32_t a[4] = {0u, 0u, 0u, 0u};
    if (row) {
      a[0] = *reinterpret_cast<const uint32_t *>(&s.a[g][k]);
      a[2] = *reinterpret_cast<const uint32_t *>(&s.a[g][k + 8]);
    }
    mma_f16(acc[0], a, frag[0], frag[1]);
    mma_f16(acc[1], a, frag[2], frag[3]);
  }
}

struct Params {
  const __nv_bfloat16 *x;
  int tokens;
  const int32_t *ids;
  const float *weights;
  const ignis_moe_slot *slots;
  DecodeCounters *counters;
  int cap;             // the workspace's decode tokens: the row stride of gate_up and h
  long long *gate_up;  // int64 [unique][cap][1280], fixed point, zero between calls
  float *h;            // f32 [unique][cap][640]
  long long *acc;
};

template <int K2>
__device__ void gate_up_unit(Shared &s, const Params &p, int u, int cb, int split, const ignis_moe_slot &slot) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  const RecordPlanes planes = record_planes(slot, kHidden, kGateUpOut, K2);
  const uint32_t *trellis = planes.trellis;
  const __half *suh = planes.suh;
  const __half *svh = planes.svh;
  const int k0 = split * kSplitK;
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int tiles_n = kGateUpOut / 16;

  // Every weight load of the unit first; the rotation overlaps them.
  uint32_t w[kKTiles];
  load_tiles<K2>(trellis + (static_cast<size_t>(k0 / 16) * tiles_n + 8 * cb + warp) * words, tiles_n * words, w);
  prepare_a(s, tokens, suh + k0, [&](int t, int k) {
    return __bfloat162float(p.x[static_cast<size_t>(t) * kHidden + k0 + k]);
  });
  float acc[2][4] = {};
  mma_tiles<K2>(s, tokens, w, acc);

  const int g = lane >> 2;
  const int c = lane & 3;
  if (g < tokens) {
    const float inv = 1.0f / s.scale[g];
    long long *dst = p.gate_up + (static_cast<size_t>(u) * p.cap + g) * kGateUpOut + 128 * cb + warp * 16 + 2 * c;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      add_fixed(dst + hf * 8, acc[hf][0] * inv);
      add_fixed(dst + hf * 8 + 1, acc[hf][1] * inv);
    }
  }
  __threadfence();
  __syncthreads();
  const int b = cb % kGateUpBlocks;
  if (threadIdx.x == 0) {
    const uint32_t before = atomicAdd(&p.counters->gate_up_arrivals[u * kGateUpBlocks + b], 1u);
    s.is_last = before == 2 * kDecodeSplits - 1;
  }
  __syncthreads();
  if (!s.is_last) return;

  // Eighth arrival for (u, b): read gate and up back, rotate, scale, SwiGLU; zero what was read.
  __threadfence();
  float(*y)[256] = reinterpret_cast<float(*)[256]>(&s.xh[0][0]);
  long long *row = p.gate_up + static_cast<size_t>(u) * p.cap * kGateUpOut;
  for (int i = threadIdx.x; i < tokens * 256; i += blockDim.x) {
    const int t = i / 256;
    const int col = i % 256;
    long long *at = row + static_cast<size_t>(t) * kGateUpOut + (col < 128 ? 0 : kInter) + 128 * b + (col & 127);
    y[t][col] = from_fixed(__ldcg(at));
    *at = 0;
  }
  __syncthreads();
  for (int q2 = warp; q2 < 2 * tokens; q2 += kWarps) {
    const int t = q2 >> 1;
    const int half = q2 & 1;
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = y[t][half * 128 + 4 * lane + q];
    warp_hadamard128(v);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int col = (half ? kInter : 0) + 128 * b + 4 * lane + q;
      y[t][half * 128 + 4 * lane + q] = v[q] * __half2float(svh[col]);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < tokens * 128; i += blockDim.x) {
    const int t = i / 128;
    const int j = i % 128;
    p.h[(static_cast<size_t>(u) * p.cap + t) * kInter + 128 * b + j] = silu(y[t][j]) * y[t][128 + j];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    p.counters->gate_up_arrivals[u * kGateUpBlocks + b] = 0;
    __threadfence();
    atomicAdd(&p.counters->h_ready[u], 1u);
  }
}

template <int K2>
__device__ void down_unit(Shared &s, const Params &p, int u, int cb, const ignis_moe_slot &slot) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  const RecordPlanes planes = record_planes(slot, kInter, kHidden, K2);
  const uint32_t *trellis = planes.trellis;
  const __half *suh = planes.suh;
  const __half *svh = planes.svh;
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int tiles_n = kHidden / 16;

  // The weights do not depend on h: their loads are in flight while the unit waits for it.
  uint32_t w[kKTiles];
  load_tiles<K2>(trellis + static_cast<size_t>(8 * cb + warp) * words, tiles_n * words, w);
  if (threadIdx.x == 0) {
    while (ld_acquire(&p.counters->h_ready[u]) < static_cast<uint32_t>(kGateUpBlocks)) __nanosleep(64);
  }
  __syncthreads();
  const float *h = p.h + static_cast<size_t>(u) * p.cap * kInter;
  prepare_a(s, tokens, suh, [&](int t, int k) { return __ldcg(h + static_cast<size_t>(t) * kInter + k); });
  float acc[2][4] = {};
  mma_tiles<K2>(s, tokens, w, acc);

  float(*y)[128] = reinterpret_cast<float(*)[128]>(&s.xh[0][0]);
  const int g = lane >> 2;
  const int c = lane & 3;
  __syncthreads();  // every warp is done reading s.xh's rotated rows before y reuses them
  if (g < tokens) {
    const float inv = 1.0f / s.scale[g];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int col = warp * 16 + hf * 8 + 2 * c;
      y[g][col] = acc[hf][0] * inv;
      y[g][col + 1] = acc[hf][1] * inv;
    }
  }
  __syncthreads();
  if (warp < tokens && (s.unique_sel[u] >> warp & 1u)) {
    const int t = warp;
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = y[t][4 * lane + q];
    warp_hadamard128(v);
    const float wt = s.unique_w[u][t];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int col = 128 * cb + 4 * lane + q;
      add_fixed(p.acc + static_cast<size_t>(t) * kHidden + col, wt * (v[q] * __half2float(svh[col])));
    }
  }
  __syncthreads();
}

__global__ void __launch_bounds__(kThreads, 2) experts_decode_kernel(Params p) {
  __shared__ Shared s;
  build_unique(s, p.ids, p.weights, p.tokens);
  const int n_unique = s.n_unique;
  const int n_gate_up = n_unique * 2 * kGateUpBlocks * kDecodeSplits;
  const int n_total = n_gate_up + n_unique * kDownBlocks;
  if (threadIdx.x == 0) s.ticket = static_cast<int>(atomicAdd(&p.counters->ticket, 1u));
  __syncthreads();
  int ticket = s.ticket;
  __syncthreads();
  while (ticket < n_total) {
    // The next ticket is taken now and read after this unit, so its round trip overlaps the
    // unit's work. Still deadlock-free: a CTA's prefetched ticket is later than its current one,
    // so every gate/up ticket is held by a CTA whose current unit is gate/up, which waits on
    // nothing.
    uint32_t next = 0;
    if (threadIdx.x == 0) next = atomicAdd(&p.counters->ticket, 1u);
    if (ticket < n_gate_up) {
      const int u = ticket / (2 * kGateUpBlocks * kDecodeSplits);
      const int cb = ticket / kDecodeSplits % (2 * kGateUpBlocks);
      const int split = ticket % kDecodeSplits;
      const ignis_moe_slot slot = load_slot(p.slots, s.unique_id[u], IGNIS_MOE_PROJ_GATE_UP);
      dispatch_k2(slot.k2, [&](auto k2) { gate_up_unit<decltype(k2)::value>(s, p, u, cb, split, slot); });
    } else {
      const int d = ticket - n_gate_up;
      const int u = d / kDownBlocks;
      const int cb = d % kDownBlocks;
      const ignis_moe_slot slot = load_slot(p.slots, s.unique_id[u], IGNIS_MOE_PROJ_DOWN);
      dispatch_k2(slot.k2, [&](auto k2) { down_unit<decltype(k2)::value>(s, p, u, cb, slot); });
    }
    __syncthreads();
    if (threadIdx.x == 0) s.ticket = static_cast<int>(next);
    __syncthreads();
    ticket = s.ticket;
  }
  // The last CTA out leaves the counters as it found them, so the launch replays.
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&p.counters->done, 1u) == gridDim.x - 1) {
      for (int u = 0; u < n_unique; ++u) p.counters->h_ready[u] = 0;
      p.counters->ticket = 0;
      p.counters->done = 0;
      __threadfence();
    }
  }
}

}  // namespace
}  // namespace ignis_moe

namespace ignis_moe {

int32_t prepare_decode(int *grid) {
  int device = 0, sms = 0, per_sm = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (err == cudaSuccess) err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, experts_decode_kernel, kThreads, 0);
  if (err != cudaSuccess) return fail(std::string("ignis_moe_prepare (decode): ") + cudaGetErrorString(err));
  if (per_sm <= 0) return fail("ignis_moe_prepare (decode): the decode kernel fits no CTA on an SM");
  // Every CTA resident at once when the launch runs alone; the ticket scheme does not need it,
  // but a grid larger than the card would only queue CTAs behind spinning ones.
  *grid = sms * per_sm;
  return 0;
}

}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" uint64_t ignis_moe_workspace_bytes(uint32_t decode_tokens, uint32_t prefill_tokens) {
  return workspace_layout(decode_tokens, prefill_tokens).total;
}

extern "C" int32_t ignis_moe_plan_bytes(uint32_t decode_tokens, uint32_t prefill_tokens, struct ignis_moe_plan *plan) {
  if (plan == nullptr) return fail("ignis_moe_plan_bytes: plan is NULL");
  if (decode_tokens == 0 || decode_tokens > static_cast<uint32_t>(kDecodeMaxTokens) || prefill_tokens == 0) {
    return fail("ignis_moe_plan_bytes: decode_tokens must be 1.." + std::to_string(kDecodeMaxTokens) +
                " and prefill_tokens at least 1");
  }
  const uint64_t t = decode_tokens > prefill_tokens ? decode_tokens : prefill_tokens;
  plan->workspace = align256(workspace_layout(decode_tokens, prefill_tokens).total);
  plan->acc = align256(t * kHidden * 8);
  plan->router = align256(t * kTopK * 4) + align256(t * kTopK * 4) + align256(t * kExperts * 4);
  plan->shared = align256(t * kInter * 2) + align256(t * kHidden * 4);
  plan->total = plan->workspace + plan->acc + plan->router + plan->shared;
  return 0;
}

extern "C" int32_t ignis_moe_workspace_init(const struct ignis_moe_workspace *workspace, int64_t *acc, void *stream) {
  if (check_workspace("ignis_moe_workspace_init", workspace) != 0) return -1;
  if (acc == nullptr) return fail("ignis_moe_workspace_init: acc is NULL");
  if (ignis_moe_prepare() != 0) return -1;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const uint64_t rows = workspace->decode_tokens > workspace->prefill_tokens ? workspace->decode_tokens : workspace->prefill_tokens;
  cudaError_t err =
      cudaMemsetAsync(workspace->base, 0, workspace_layout(workspace->decode_tokens, workspace->prefill_tokens).total, s);
  if (err == cudaSuccess) err = cudaMemsetAsync(acc, 0, rows * kHidden * sizeof(int64_t), s);
  if (err != cudaSuccess) return fail(std::string("ignis_moe_workspace_init: ") + cudaGetErrorString(err));
  return 0;
}

extern "C" int32_t ignis_moe_experts_decode(const void *x, uint32_t tokens, const int32_t *ids,
                                            const float *weights, const struct ignis_moe_slot *slots,
                                            const struct ignis_moe_workspace *workspace, int64_t *acc, void *stream) {
  if (x == nullptr || ids == nullptr || weights == nullptr || slots == nullptr || acc == nullptr) {
    return fail("ignis_moe_experts_decode: NULL pointer");
  }
  if (check_workspace("ignis_moe_experts_decode", workspace) != 0) return -1;
  if (tokens == 0 || tokens > workspace->decode_tokens) {
    return fail("ignis_moe_experts_decode: tokens must be 1.." + std::to_string(workspace->decode_tokens) +
                " (the workspace's decode_tokens)");
  }
  DecodeLaunch launch;
  if (require_prepared("ignis_moe_experts_decode", &launch) != 0) return -1;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (decode_route() == IGNIS_MOE_DECODE_CLUSTERS) {
    if (launch.cluster_size == 0) return fail("ignis_moe_experts_decode: this device runs no decode cluster");
    return decode_clusters(launch.cluster_size, static_cast<const __nv_bfloat16 *>(x), static_cast<int>(tokens), ids,
                           weights, slots, reinterpret_cast<long long *>(acc), s);
  }
  const WorkspaceLayout l = workspace_layout(workspace->decode_tokens, workspace->prefill_tokens);
  char *ws = static_cast<char *>(workspace->base);
  Params p;
  p.x = static_cast<const __nv_bfloat16 *>(x);
  p.tokens = static_cast<int>(tokens);
  p.ids = ids;
  p.weights = weights;
  p.slots = slots;
  p.counters = reinterpret_cast<DecodeCounters *>(ws + l.decode_counters);
  p.cap = static_cast<int>(workspace->decode_tokens);
  p.gate_up = reinterpret_cast<long long *>(ws + l.decode_gate_up);
  p.h = reinterpret_cast<float *>(ws + l.decode_h);
  p.acc = reinterpret_cast<long long *>(acc);
  experts_decode_kernel<<<launch.grid, kThreads, 0, s>>>(p);
  return check_launch("ignis_moe_experts_decode");
}
