// ignis kernel leaf: Flash-Next's routed experts for 1..8 decode tokens, one thread-block
// cluster per selected expert -- OURS (kernel/include/ignis_moe.h, IGNIS_MOE_DECODE_CLUSTERS).
//
// The ticket kernel (moe_decode.cu) is bound by its structure, not by DRAM (finding
// docs/findings/2026-10-05-moe-decode-is-structure-bound.md): units in a second wave, a
// reduction stage between gate/up and down, round trips per unit. Here every distinct expert of
// the call is one cluster of C CTAs (C = 16 where the card schedules it, else 8) that runs the
// whole expert with no global synchronization:
//
//   gate/up   CTA r owns the inputs [2560 r / C, 2560 (r + 1) / C) of every one of the 1280
//             columns: it rotates its slice of the tokens' inputs (x o suh, 128-wide Hadamard,
//             a power-of-two fp16 scale over the slice), and its eight warps decode their ten
//             column tiles from the trellis straight into m16n8k16 B fragments. Its pre-rotation
//             partial sums, unscaled, stay in its shared memory.
//   SwiGLU    cluster barrier; CTA j < 5 sums gate block j and up block j over the C CTAs'
//             shared memory (distributed shared memory, fixed order), rotates both, applies svh
//             and SwiGLU: h's block j.
//   down      cluster barrier; every CTA gathers the five blocks of h, rotates them (h o suh,
//             fp16 scale over the 640) and runs the columns [2560 r / C, 2560 (r + 1) / C) of
//             down in four k-runs, summed in its shared memory in fixed order; its first weight
//             loads were issued before the first barrier.
//   output    cluster barrier; CTA r rotates the down output blocks r, r + C, ... (their
//             pre-rotation sums read across the cluster), applies svh and each selecting
//             token's routing weight and adds them into the fixed-point accumulator; a last
//             barrier keeps every CTA's shared memory alive until the cluster is done reading it.
//
// A warp streams work items of ten 16 x 16 tiles of one column tile, two in flight: the ten
// words a lane needs of item i + 1 are loaded while item i is decoded. Deterministic: every
// floating-point sum has a fixed order, and the experts meet only in the integer accumulator.
// Nothing outside the accumulator is written to global memory, so the launch needs no
// workspace and replays from a CUDA graph as is.

#include "moe_decode_common.cuh"
#include "trellis_decode.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace ignis_moe {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kT = kDecodeMaxTokens;
constexpr int kRun = 10;                        // k-tiles in one work item
constexpr int kColTilesGu = kGateUpOut / 16;    // 80
constexpr int kKTilesGu = kHidden / 16;         // 160
constexpr int kColTilesDn = kHidden / 16;       // 160
constexpr int kKTilesDn = kInter / 16;          // 40
constexpr int kRunsDn = kKTilesDn / kRun;       // 4

template <int C> struct Geo {
  static constexpr int kKtGu = kKTilesGu / C;           // gate/up k-tiles of one CTA: 10 or 20
  static constexpr int kInGu = kKtGu * 16;              // its inputs: 160 or 320
  static constexpr int kRunsGu = kKtGu / kRun;          // items per column tile: 1 or 2
  static constexpr int kItemsGu = kColTilesGu * kRunsGu / kWarps;  // per warp: 10 or 20
  static constexpr int kColTilesDnCta = kColTilesDn / C;            // 10 or 20
  static constexpr int kColsDn = kColTilesDnCta * 16;               // down columns of one CTA
  static constexpr int kItemsDn = kColTilesDnCta * kRunsDn / kWarps;  // per warp: 5 or 10
  static_assert(kKTilesGu % C == 0 && kKtGu % kRun == 0 && kColTilesGu % kWarps == 0, "gate/up split");
  static_assert((kColTilesDnCta * kRunsDn) % kWarps == 0, "down split");
  static_assert(C >= kGateUpBlocks, "the SwiGLU blocks need one CTA each");
};

template <int C> struct Smem {
  int n_unique;
  int unique_id[kDecodeMaxUnique];
  uint32_t unique_sel[kDecodeMaxUnique];
  float unique_w[kDecodeMaxUnique][kT];
  int slot_of[kT * kTopK];
  int first[kT * kTopK];
  float scale[kT];
  float blockmax[kT][kGateUpBlocks];
  alignas(16) __half a_gu[kT][Geo<C>::kInGu + 8];  // padded off the bank stride
  alignas(16) __half a_dn[kT][kInter + 8];
  alignas(16) float h[kT][256];                    // CTA j < 5: gate | up block j, then h block j
  alignas(16) float dn_sums[kT][Geo<C>::kColsDn];
  // Gate/up partial sums [T][1280]; then the rotated h [T][640]; then the down partials
  // [kRunsDn][T][kColsDn].
  alignas(16) float part[kT * kGateUpOut];
};

struct Params {
  const __nv_bfloat16 *x;
  int tokens;
  const int32_t *ids;
  const float *weights;
  const ignis_moe_slot *slots;
  long long *acc;
};

// Item i of a warp: the ten words a lane needs of each of its k-tiles, tile kt at
// tile0 + kt * stride (u32 words).
template <int K2>
__device__ __forceinline__ void load_run(const uint32_t *tile0, int stride, uint32_t (&w)[kRun]) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kt = 0; kt < kRun; ++kt) w[kt] = lane < words ? __ldg(tile0 + kt * stride + lane) : 0u;
}

// acc += A(tokens x 160) . W(160 x 16) for one item; a_row is the lane's A row at the item's
// first input (rows past the tokens contribute zero).
template <int K2>
__device__ __forceinline__ void mma_run(const __half *a_row, bool row, const uint32_t (&w)[kRun], float (&acc)[2][4]) {
  const int lane = threadIdx.x & 31;
  const int c = lane & 3;
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
#pragma unroll
  for (int kt = 0; kt < kRun; ++kt) {
    const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, w[kt], plan.w0);
    const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, w[kt], plan.w1);
    uint32_t frag[4];
    ignis_trellis::decode_fragment<K2>(w0, w1, plan, frag);
    uint32_t a[4] = {0u, 0u, 0u, 0u};
    if (row) {
      a[0] = *reinterpret_cast<const uint32_t *>(a_row + kt * 16 + 2 * c);
      a[2] = *reinterpret_cast<const uint32_t *>(a_row + kt * 16 + 8 + 2 * c);
    }
    mma_f16(acc[0], a, frag[0], frag[1]);
    mma_f16(acc[1], a, frag[2], frag[3]);
  }
}

// Runs a warp's kCount items, two in flight: `w0` / `w1` already hold items 0 and 1's loads (the
// caller issued them), tile(i) is item i's first tile, use(i, words) consumes item i.
template <int K2, int kCount, typename Tile, typename Use>
__device__ __forceinline__ void run_items(uint32_t (&w0)[kRun], uint32_t (&w1)[kRun], int stride, Tile tile, Use use) {
  for (int i = 0; i < kCount; i += 2) {
    use(i, w0);
    if (i + 2 < kCount) load_run<K2>(tile(i + 2), stride, w0);
    if (i + 1 < kCount) {
      use(i + 1, w1);
      if (i + 3 < kCount) load_run<K2>(tile(i + 3), stride, w1);
    }
  }
}

// One token row's 128-block `blk` of `in(k)` times suh, rotated, on the calling warp: lane holds
// elements 4 lane .. 4 lane + 3.
template <typename In>
__device__ __forceinline__ void rotated_block(const __half *suh, int blk, In in, float (&v)[4]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int k = blk * 128 + 4 * lane + q;
    v[q] = in(k) * __half2float(suh[k]);
  }
  warp_hadamard128(v);
}

template <int K2, int C>
__device__ void gate_up_phase(Smem<C> &s, const Params &p, int rank, const ignis_moe_slot &slot) {
  using G = Geo<C>;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  const RecordPlanes planes = record_planes(slot, kHidden, kGateUpOut, K2);
  constexpr int words = ignis_trellis::tile_words(K2);
  const int k0 = rank * G::kInGu;  // the CTA's first input

  // The warp's items: column tile warp + 8 m, k-run r of the CTA's slice; item i = (m, r).
  auto tile = [&](int i) {
    const int j = warp + kWarps * (i / G::kRunsGu);
    const int kt = rank * G::kKtGu + (i % G::kRunsGu) * kRun;
    return planes.trellis + (static_cast<size_t>(kt) * kColTilesGu + j) * words;
  };
  const int stride = kColTilesGu * words;
  uint32_t w0[kRun], w1[kRun];
  load_run<K2>(tile(0), stride, w0);
  load_run<K2>(tile(1), stride, w1);

  // The slice's rotation overlaps the loads: the 128-blocks it touches, scaled over the slice.
  const int b_first = k0 / 128;
  const int b_count = (k0 + G::kInGu - 1) / 128 - b_first + 1;
  auto x_at = [&](int t) {
    return [&, t](int k) { return __bfloat162float(p.x[static_cast<size_t>(t) * kHidden + k]); };
  };
  for (int q2 = warp; q2 < tokens * b_count; q2 += kWarps) {
    const int t = q2 / b_count;
    const int blk = b_first + q2 % b_count;
    float v[4];
    rotated_block(planes.suh, blk, x_at(t), v);
    float m = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = blk * 128 + 4 * lane + q;
      if (k >= k0 && k < k0 + G::kInGu) m = fmaxf(m, fabsf(v[q]));
    }
    m = warp_max(m);
    if (lane == 0) s.blockmax[t][q2 % b_count] = m;
  }
  __syncthreads();
  if (threadIdx.x < tokens) {
    float m = 0.0f;
    for (int b = 0; b < b_count; ++b) m = fmaxf(m, s.blockmax[threadIdx.x][b]);
    s.scale[threadIdx.x] = fp16_operand_scale(m);
  }
  __syncthreads();
  for (int q2 = warp; q2 < tokens * b_count; q2 += kWarps) {
    const int t = q2 / b_count;
    const int blk = b_first + q2 % b_count;
    float v[4];
    rotated_block(planes.suh, blk, x_at(t), v);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = blk * 128 + 4 * lane + q;
      if (k >= k0 && k < k0 + G::kInGu) s.a_gu[t][k - k0] = __float2half_rn(v[q] * s.scale[t]);
    }
  }
  __syncthreads();

  const int g = lane >> 2;
  const int c = lane & 3;
  const bool row = g < tokens;
  const float inv = row ? 1.0f / s.scale[g] : 0.0f;
  float acc[2][4] = {};
  run_items<K2, G::kItemsGu>(w0, w1, stride, tile, [&](int i, const uint32_t (&w)[kRun]) {
    mma_run<K2>(&s.a_gu[g][(i % G::kRunsGu) * kRun * 16], row, w, acc);
    if (i % G::kRunsGu == G::kRunsGu - 1) {
      if (row) {
        const int col = 16 * (warp + kWarps * (i / G::kRunsGu)) + 2 * c;
        float *dst = s.part + g * kGateUpOut + col;
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          dst[8 * hf] = acc[hf][0] * inv;
          dst[8 * hf + 1] = acc[hf][1] * inv;
        }
      }
#pragma unroll
      for (auto &a : acc) a[0] = a[1] = a[2] = a[3] = 0.0f;
    }
  });
}

template <int K2, int C>
__device__ void down_phase(Smem<C> &s, const Params &p, int u, int rank, const ignis_moe_slot &gu_slot,
                           const ignis_moe_slot &slot) {
  using G = Geo<C>;
  cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  const RecordPlanes planes = record_planes(slot, kInter, kHidden, K2);
  constexpr int words = ignis_trellis::tile_words(K2);

  // The warp's items: CTA item warp + 8 m = (local column tile, k-run). The first two are
  // loaded now, so they fly while the cluster finishes gate/up and the SwiGLU.
  auto item = [&](int i) { return warp + kWarps * i; };
  auto tile = [&](int i) {
    const int j = rank * G::kColTilesDnCta + item(i) / kRunsDn;
    const int kt = (item(i) % kRunsDn) * kRun;
    return planes.trellis + (static_cast<size_t>(kt) * kColTilesDn + j) * words;
  };
  const int stride = kColTilesDn * words;
  uint32_t w0[kRun], w1[kRun];
  load_run<K2>(tile(0), stride, w0);
  load_run<K2>(tile(1), stride, w1);

  cluster.sync();  // every CTA's gate/up partials are in its shared memory

  if (rank < kGateUpBlocks) {
    // Gate block `rank` and up block `rank`, summed over the cluster in rank order, rotated,
    // scaled by svh, SwiGLU.
    const RecordPlanes gu = record_planes(gu_slot, kHidden, kGateUpOut, gu_slot.k2);
    for (int i = threadIdx.x; i < tokens * 256; i += kThreads) {
      const int t = i / 256;
      const int col = i % 256;
      const int at = t * kGateUpOut + (col < 128 ? 0 : kInter) + 128 * rank + (col & 127);
      float sum = 0.0f;
      for (int q = 0; q < C; ++q) sum += cluster.map_shared_rank(s.part, q)[at];
      s.h[t][col] = sum;
    }
    __syncthreads();
    for (int q2 = warp; q2 < 2 * tokens; q2 += kWarps) {
      const int t = q2 >> 1;
      const int half = q2 & 1;
      float v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = s.h[t][half * 128 + 4 * lane + q];
      warp_hadamard128(v);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int col = (half ? kInter : 0) + 128 * rank + 4 * lane + q;
        s.h[t][half * 128 + 4 * lane + q] = v[q] * __half2float(gu.svh[col]);
      }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tokens * 128; i += kThreads) {
      const int t = i / 128;
      const int j = i % 128;
      s.h[t][j] = silu(s.h[t][j]) * s.h[t][128 + j];
    }
  }

  cluster.sync();  // h's five blocks are in CTAs 0..4

  // h o suh rotated per 128-block into part (free now: nobody reads the gate/up partials), the
  // fp16 operand scale over the 640, then a_dn.
  float(*hr)[kInter] = reinterpret_cast<float(*)[kInter]>(s.part);
  for (int q2 = warp; q2 < tokens * kGateUpBlocks; q2 += kWarps) {
    const int t = q2 / kGateUpBlocks;
    const int blk = q2 % kGateUpBlocks;
    const float *hb = cluster.map_shared_rank(&s.h[0][0], blk) + t * 256;
    float v[4];
    rotated_block(planes.suh, blk, [&](int k) { return hb[k - 128 * blk]; }, v);
    float m = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      hr[t][blk * 128 + 4 * lane + q] = v[q];
      m = fmaxf(m, fabsf(v[q]));
    }
    m = warp_max(m);
    if (lane == 0) s.blockmax[t][blk] = m;
  }
  __syncthreads();
  if (threadIdx.x < tokens) {
    float m = 0.0f;
    for (int b = 0; b < kGateUpBlocks; ++b) m = fmaxf(m, s.blockmax[threadIdx.x][b]);
    s.scale[threadIdx.x] = fp16_operand_scale(m);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < tokens * kInter; i += kThreads) {
    const int t = i / kInter;
    const int k = i % kInter;
    s.a_dn[t][k] = __float2half_rn(hr[t][k] * s.scale[t]);
  }
  __syncthreads();

  const int g = lane >> 2;
  const int c = lane & 3;
  const bool row = g < tokens;
  const float inv = row ? 1.0f / s.scale[g] : 0.0f;
  run_items<K2, G::kItemsDn>(w0, w1, stride, tile, [&](int i, const uint32_t (&w)[kRun]) {
    float acc[2][4] = {};
    const int run = item(i) % kRunsDn;
    mma_run<K2>(&s.a_dn[g][run * kRun * 16], row, w, acc);
    if (row) {
      const int col = 16 * (item(i) / kRunsDn) + 2 * c;
      float *dst = s.part + (run * kT + g) * G::kColsDn + col;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        dst[8 * hf] = acc[hf][0] * inv;
        dst[8 * hf + 1] = acc[hf][1] * inv;
      }
    }
  });
  __syncthreads();
  for (int i = threadIdx.x; i < tokens * G::kColsDn; i += kThreads) {
    const int t = i / G::kColsDn;
    const int col = i % G::kColsDn;
    float sum = 0.0f;
#pragma unroll
    for (int run = 0; run < kRunsDn; ++run) sum += s.part[(run * kT + t) * G::kColsDn + col];
    s.dn_sums[t][col] = sum;
  }

  cluster.sync();  // every CTA's pre-rotation down sums are in its shared memory

  // Output blocks rank, rank + C, ...: rotated, svh, the token's routing weight, accumulated.
  constexpr int kMyBlocks = (kDownBlocks + C - 1) / C;
  const uint32_t sel = s.unique_sel[u];
  for (int q2 = warp; q2 < kMyBlocks * tokens; q2 += kWarps) {
    const int b = rank + C * (q2 / tokens);
    const int t = q2 % tokens;
    if (b >= kDownBlocks || !(sel >> t & 1u)) continue;
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int col = 128 * b + 4 * lane + q;
      v[q] = cluster.map_shared_rank(&s.dn_sums[0][0], col / G::kColsDn)[t * G::kColsDn + col % G::kColsDn];
    }
    warp_hadamard128(v);
    const float wt = s.unique_w[u][t];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int col = 128 * b + 4 * lane + q;
      add_fixed(p.acc + static_cast<size_t>(t) * kHidden + col, wt * (v[q] * __half2float(planes.svh[col])));
    }
  }

  cluster.sync();  // nobody reads this CTA's shared memory any more
}

template <int C>
__global__ void __launch_bounds__(kThreads, 2) cluster_decode_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<C> &s = *reinterpret_cast<Smem<C> *>(smem_raw);
  build_unique(s, p.ids, p.weights, p.tokens);
  const int u = static_cast<int>(blockIdx.x) / C;
  // The whole cluster leaves together: u is the same for its CTAs and none has synchronized.
  if (u >= s.n_unique) return;
  const int rank = static_cast<int>(cg::this_cluster().block_rank());
  const ignis_moe_slot gu = load_slot(p.slots, s.unique_id[u], IGNIS_MOE_PROJ_GATE_UP);
  const ignis_moe_slot dn = load_slot(p.slots, s.unique_id[u], IGNIS_MOE_PROJ_DOWN);
  dispatch_k2(gu.k2, [&](auto k2) { gate_up_phase<decltype(k2)::value, C>(s, p, rank, gu); });
  dispatch_k2(dn.k2, [&](auto k2) { down_phase<decltype(k2)::value, C>(s, p, u, rank, gu, dn); });
}

template <int C>
cudaLaunchConfig_t launch_config(int tokens, cudaStream_t stream, cudaLaunchAttribute *attr) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(tokens * kTopK * C));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = sizeof(Smem<C>);
  cfg.stream = stream;
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = C;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cfg;
}

// How many C-CTA clusters of the kernel the device runs at once (0 if it cannot run one).
template <int C>
void active_clusters(int *count) {
  *count = 0;
  cudaError_t err = cudaFuncSetAttribute(cluster_decode_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(sizeof(Smem<C>)));
  if (err == cudaSuccess && C > 8) {
    err = cudaFuncSetAttribute(cluster_decode_kernel<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  cudaLaunchAttribute attr;
  const cudaLaunchConfig_t cfg = launch_config<C>(1, nullptr, &attr);
  if (err == cudaSuccess) err = cudaOccupancyMaxActiveClusters(count, cluster_decode_kernel<C>, &cfg);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();  // an unsupported size is an answer here, not an error to keep
    *count = 0;
  }
}

}  // namespace

void prepare_decode_clusters(int *cluster_size) {
  // Sixteen CTAs per expert where the card co-schedules a whole token's ten experts that way,
  // else eight (the portable size), else none.
  int sixteen = 0, eight = 0;
  active_clusters<16>(&sixteen);
  active_clusters<8>(&eight);
  *cluster_size = sixteen >= kTopK ? 16 : eight >= 1 ? 8 : 0;
}

int32_t decode_clusters(int cluster_size, const __nv_bfloat16 *x, int tokens, const int32_t *ids, const float *weights,
                        const ignis_moe_slot *slots, long long *acc, cudaStream_t stream) {
  const Params p{x, tokens, ids, weights, slots, acc};
  cudaLaunchAttribute attr;
  cudaError_t err;
  if (cluster_size == 16) {
    const cudaLaunchConfig_t cfg = launch_config<16>(tokens, stream, &attr);
    err = cudaLaunchKernelEx(&cfg, cluster_decode_kernel<16>, p);
  } else {
    const cudaLaunchConfig_t cfg = launch_config<8>(tokens, stream, &attr);
    err = cudaLaunchKernelEx(&cfg, cluster_decode_kernel<8>, p);
  }
  if (err != cudaSuccess) return fail(std::string("ignis_moe_experts_decode (clusters): ") + cudaGetErrorString(err));
  return check_launch("ignis_moe_experts_decode (clusters)");
}

}  // namespace ignis_moe
