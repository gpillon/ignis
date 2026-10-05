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

#include "moe_common.cuh"
#include "moe_workspace.cuh"
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

// The distinct experts of the call in order of first appearance (token-major, rank order),
// with each token's routing weight for them and the mask of tokens that selected them.
__device__ void build_unique(Shared &s, const int32_t *ids, const float *weights, int tokens) {
  const int n = tokens * kTopK;
  const int i = threadIdx.x;
  for (int j = threadIdx.x; j < kDecodeMaxUnique * kDecodeMaxTokens; j += blockDim.x) {
    (&s.unique_w[0][0])[j] = 0.0f;
  }
  if (i < kDecodeMaxUnique) s.unique_sel[i] = 0u;
  int e = -1;
  if (i < n) {
    e = ids[i];
    int f = i;
    for (int j = 0; j < i; ++j) {
      if (ids[j] == e) {
        f = j;
        break;
      }
    }
    s.first[i] = f;
  }
  __syncthreads();
  if (i < n && s.first[i] == i) {
    int slot = 0;
    for (int j = 0; j < i; ++j) slot += s.first[j] == j;
    s.slot_of[i] = slot;
    s.unique_id[slot] = e;
  }
  if (i == 0) {
    int count = 0;
    for (int j = 0; j < n; ++j) count += s.first[j] == j;
    s.n_unique = count;
  }
  __syncthreads();
  if (i < n) {
    const int u = s.slot_of[s.first[i]];
    s.unique_w[u][i / kTopK] = weights[i];
    atomicOr(&s.unique_sel[u], 1u << (i / kTopK));
  }
  __syncthreads();
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
  long long *gate_up;  // int64 [unique][8 tokens][1280], fixed point, zero between calls
  float *h;
  long long *acc;
};

template <int K2>
__device__ void gate_up_unit(Shared &s, const Params &p, int u, int cb, int split, const ignis_moe_slot &slot) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  const uint32_t *trellis = static_cast<const uint32_t *>(slot.record);
  const __half *suh = reinterpret_cast<const __half *>(static_cast<const char *>(slot.record) +
                                                       trellis_bytes(kHidden, kGateUpOut, K2));
  const __half *svh = suh + kHidden;
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
    long long *dst = p.gate_up + (static_cast<size_t>(u) * kDecodeMaxTokens + g) * kGateUpOut + 128 * cb + warp * 16 + 2 * c;
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
  long long *row = p.gate_up + static_cast<size_t>(u) * kDecodeMaxTokens * kGateUpOut;
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
    p.h[(static_cast<size_t>(u) * kDecodeMaxTokens + t) * kInter + 128 * b + j] = silu(y[t][j]) * y[t][128 + j];
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
  const uint32_t *trellis = static_cast<const uint32_t *>(slot.record);
  const __half *suh = reinterpret_cast<const __half *>(static_cast<const char *>(slot.record) +
                                                       trellis_bytes(kInter, kHidden, K2));
  const __half *svh = suh + kInter;
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int tiles_n = kHidden / 16;

  // The weights do not depend on h: their loads are in flight while the unit waits for it.
  uint32_t w[kKTiles];
  load_tiles<K2>(trellis + static_cast<size_t>(8 * cb + warp) * words, tiles_n * words, w);
  if (threadIdx.x == 0) {
    while (ld_acquire(&p.counters->h_ready[u]) < static_cast<uint32_t>(kGateUpBlocks)) __nanosleep(64);
  }
  __syncthreads();
  const float *h = p.h + static_cast<size_t>(u) * kDecodeMaxTokens * kInter;
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
      switch (slot.k2) {
      case 4: gate_up_unit<4>(s, p, u, cb, split, slot); break;
      case 5: gate_up_unit<5>(s, p, u, cb, split, slot); break;
      case 6: gate_up_unit<6>(s, p, u, cb, split, slot); break;
      default: gate_up_unit<8>(s, p, u, cb, split, slot); break;
      }
    } else {
      const int d = ticket - n_gate_up;
      const int u = d / kDownBlocks;
      const int cb = d % kDownBlocks;
      const ignis_moe_slot slot = load_slot(p.slots, s.unique_id[u], IGNIS_MOE_PROJ_DOWN);
      switch (slot.k2) {
      case 4: down_unit<4>(s, p, u, cb, slot); break;
      case 5: down_unit<5>(s, p, u, cb, slot); break;
      case 6: down_unit<6>(s, p, u, cb, slot); break;
      default: down_unit<8>(s, p, u, cb, slot); break;
      }
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

using namespace ignis_moe;

extern "C" uint64_t ignis_moe_workspace_bytes(uint32_t max_tokens) {
  return workspace_layout(max_tokens).total;
}

extern "C" int32_t ignis_moe_plan_bytes(uint32_t max_tokens, struct ignis_moe_plan *plan) {
  if (plan == nullptr) return fail("ignis_moe_plan_bytes: plan is NULL");
  if (max_tokens == 0) return fail("ignis_moe_plan_bytes: max_tokens must be at least 1");
  const uint64_t t = max_tokens;
  plan->workspace = align256(workspace_layout(max_tokens).total);
  plan->acc = align256(t * kHidden * 8);
  plan->router = align256(t * kTopK * 4) + align256(t * kTopK * 4) + align256(t * kExperts * 4);
  plan->shared = align256(t * kInter * 2) + align256(t * kHidden * 4);
  plan->total = plan->workspace + plan->acc + plan->router + plan->shared;
  return 0;
}

extern "C" int32_t ignis_moe_workspace_init(void *workspace, uint32_t max_tokens, int64_t *acc, void *stream) {
  if (workspace == nullptr || acc == nullptr) return fail("ignis_moe_workspace_init: NULL pointer");
  if (max_tokens == 0) return fail("ignis_moe_workspace_init: max_tokens must be at least 1");
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t err = cudaMemsetAsync(workspace, 0, workspace_layout(max_tokens).total, s);
  if (err == cudaSuccess) err = cudaMemsetAsync(acc, 0, static_cast<size_t>(max_tokens) * kHidden * sizeof(int64_t), s);
  if (err != cudaSuccess) return fail(std::string("ignis_moe_workspace_init: ") + cudaGetErrorString(err));
  return 0;
}

extern "C" int32_t ignis_moe_experts_decode(const void *x, uint32_t tokens, const int32_t *ids,
                                            const float *weights, const struct ignis_moe_slot *slots,
                                            void *workspace, int64_t *acc, void *stream) {
  if (x == nullptr || ids == nullptr || weights == nullptr || slots == nullptr || workspace == nullptr || acc == nullptr) {
    return fail("ignis_moe_experts_decode: NULL pointer");
  }
  if (tokens == 0 || tokens > static_cast<uint32_t>(kDecodeMaxTokens)) {
    return fail("ignis_moe_experts_decode: tokens must be 1.." + std::to_string(kDecodeMaxTokens));
  }
  static int grid = 0;
  if (grid == 0) {
    int device = 0, sms = 0, per_sm = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, experts_decode_kernel, kThreads, 0);
    grid = sms * (per_sm > 0 ? per_sm : 1);
  }
  const WorkspaceLayout l = workspace_layout(0);
  char *ws = static_cast<char *>(workspace);
  Params p;
  p.x = static_cast<const __nv_bfloat16 *>(x);
  p.tokens = static_cast<int>(tokens);
  p.ids = ids;
  p.weights = weights;
  p.slots = slots;
  p.counters = reinterpret_cast<DecodeCounters *>(ws + l.decode_counters);
  p.gate_up = reinterpret_cast<long long *>(ws + l.decode_gate_up);
  p.h = reinterpret_cast<float *>(ws + l.decode_h);
  p.acc = reinterpret_cast<long long *>(acc);
  experts_decode_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(p);
  return check_launch("ignis_moe_experts_decode");
}
