// ignis kernel leaf: Flash-Next's routed experts for a prefill chunk -- OURS
// (kernel/include/ignis_moe.h).
//
// Grouping, on the device, three small launches:
//   count    per 64-token chunk of the call, assignments per expert (shared-memory integer
//            counts, so their order does not matter);
//   scan     per expert, its first sorted row and each chunk's base inside it; the work items
//            (expert, first row) of 64-row tiles, one partial tile per expert with rows;
//   scatter  every assignment (t * 10 + rank) to its sorted row: experts in id order, and inside
//            an expert, assignments in call order (a stable rank within each chunk).
// Then one launch per projection family over every expert and K:
//   gate/up  item x output block b (0..4): 64 rows x 256 columns (gate 128b.., up 640 + 128b..)
//            over all 2560 inputs, k-block by k-block: the rows' inputs rotated (x o suh, the
//            128-wide Hadamard) and scaled by a per-row power of two from a norm bound into fp16
//            for the tensor cores, the weight tiles staged by cp.async and decoded from shared
//            memory straight into MMA B fragments. The epilogue rotates back, applies svh and
//            SwiGLU, and stores the row's h.
//   down     item x output block c (0..19): the same over h's 640 inputs; the epilogue adds
//            weight x value into the fixed-point accumulator at the row's token.
// An item of at most 16 rows takes a narrow layout (one m16 block, all eight warps along the
// columns), so a one-token expert costs its weight read and little else; a wide item's m16
// blocks beyond its rows skip their MMAs. Everything is deterministic: a row's result does not depend on the
// other rows of its tile, and the cross-expert sum is integer.

#include "moe_common.cuh"
#include "moe_workspace.cuh"
#include "trellis_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

constexpr int kThreads = 256;
constexpr int kAStride = 128 + 8;  // fp16 per A row of one k-block, padded off the bank stride

// ---- grouping -------------------------------------------------------------------------------

__global__ void group_count_kernel(const int32_t *__restrict__ ids, int tokens, uint32_t *__restrict__ hist) {
  __shared__ uint32_t h[kExperts];
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const int a0 = blockIdx.x * kGroupChunk * kTopK;
  const int a1 = min(tokens * kTopK, a0 + kGroupChunk * kTopK);
  for (int a = a0 + threadIdx.x; a < a1; a += blockDim.x) {
    const int e = ids[a];
    check_expert(e);
    atomicAdd(&h[e], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) hist[static_cast<size_t>(blockIdx.x) * kExperts + i] = h[i];
}

// One CTA of 512 threads, thread e = expert e.
__global__ void group_scan_kernel(int chunks, uint32_t *__restrict__ hist, uint32_t *__restrict__ offset,
                                  uint32_t *__restrict__ item_count, int2 *__restrict__ items) {
  __shared__ uint32_t scan[kExperts];
  const int e = threadIdx.x;
  uint32_t total = 0;
  for (int c = 0; c < chunks; ++c) {
    const uint32_t v = hist[static_cast<size_t>(c) * kExperts + e];
    hist[static_cast<size_t>(c) * kExperts + e] = total;
    total += v;
  }
  auto inclusive_scan = [&](uint32_t v) {
    scan[e] = v;
    __syncthreads();
    for (int d = 1; d < kExperts; d <<= 1) {
      const uint32_t add = e >= d ? scan[e - d] : 0u;
      __syncthreads();
      scan[e] += add;
      __syncthreads();
    }
    const uint32_t r = scan[e];
    __syncthreads();
    return r;
  };
  const uint32_t first = inclusive_scan(total) - total;
  offset[e] = first;
  if (e == kExperts - 1) offset[kExperts] = first + total;
  for (int c = 0; c < chunks; ++c) hist[static_cast<size_t>(c) * kExperts + e] += first;
  const uint32_t tiles = (total + kTileRows - 1) / kTileRows;
  const uint32_t tiles_end = inclusive_scan(tiles);
  for (uint32_t r = 0; r < tiles; ++r) items[tiles_end - tiles + r] = make_int2(e, static_cast<int>(r) * kTileRows);
  if (e == kExperts - 1) *item_count = tiles_end;
}

// One CTA of kGroupChunk * kTopK threads per chunk.
__global__ void group_scatter_kernel(const int32_t *__restrict__ ids, int tokens, const uint32_t *__restrict__ base,
                                     int32_t *__restrict__ sorted) {
  __shared__ int32_t e_s[kGroupChunk * kTopK];
  const int a0 = blockIdx.x * kGroupChunk * kTopK;
  const int n = min(kGroupChunk * kTopK, tokens * kTopK - a0);
  const int i = threadIdx.x;
  if (i < n) e_s[i] = ids[a0 + i];
  __syncthreads();
  if (i < n) {
    const int e = e_s[i];
    check_expert(e);
    uint32_t rank = 0;
    for (int j = 0; j < i; ++j) rank += e_s[j] == e;
    sorted[base[static_cast<size_t>(blockIdx.x) * kExperts + e] + rank] = a0 + i;
  }
}

// ---- the grouped GEMMs ----------------------------------------------------------------------

struct Params {
  const __nv_bfloat16 *x;
  const float *weights;
  const ignis_moe_slot *slots;
  const uint32_t *offset;
  const uint32_t *item_count;
  const int2 *items;
  const int32_t *sorted;
  float *h;
  long long *acc;
};

// Dynamic shared memory, carved by hand: the A tile and two weight stages, then the epilogue's
// fp32 tile over the same bytes, then the rows' metadata after both.
constexpr int kStageWordsMax = 8 * 16 * ignis_trellis::tile_words(8);       // gate/up stage at K = 4
constexpr size_t kABytes = sizeof(__half) * kTileRows * kAStride;            // 17,408
constexpr size_t kBBytes = sizeof(uint32_t) * 2 * kStageWordsMax;            // 32,768
constexpr int kYStride = 256 + 4;
constexpr size_t kYBytes = sizeof(float) * kTileRows * kYStride;             // 66,560
constexpr size_t kMetaOffset = (kABytes + kBBytes > kYBytes ? kABytes + kBBytes : kYBytes);
constexpr size_t kSmemBytes = kMetaOffset + 3 * sizeof(float) * kTileRows;

struct Tile {
  __half (*a)[kAStride];
  uint32_t *b;  // [2][stage words]
  float (*y)[kYStride];
  float *scale;
  int *token;
  float *weight;
};

__device__ __forceinline__ Tile carve(unsigned char *smem) {
  Tile t;
  t.a = reinterpret_cast<__half(*)[kAStride]>(smem);
  t.b = reinterpret_cast<uint32_t *>(smem + kABytes);
  t.y = reinterpret_cast<float(*)[kYStride]>(smem);
  t.scale = reinterpret_cast<float *>(smem + kMetaOffset);
  t.token = reinterpret_cast<int *>(smem + kMetaOffset + sizeof(float) * kTileRows);
  t.weight = reinterpret_cast<float *>(smem + kMetaOffset + 2 * sizeof(float) * kTileRows);
  return t;
}

// Per-row fp16 scale from a bound: a rotated entry is at most the 2-norm of its 128-block of
// in(row, k) * suh[k], so the largest block norm bounds the whole row.
template <int kRows, typename In>
__device__ void row_scales(const Tile &t, int rows, int in_width, const __half *suh, In in) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int r = warp; r < kRows; r += kThreads / 32) {
    float bound = 0.0f;
    if (r < rows) {
      for (int blk = 0; blk < in_width / 128; ++blk) {
        float ss = 0.0f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int k = blk * 128 + 4 * lane + q;
          const float v = in(r, k) * __half2float(suh[k]);
          ss = fmaf(v, v, ss);
        }
        bound = fmaxf(bound, warp_sum(ss));
      }
      bound = sqrtf(bound) * 1.0001f;  // margin for the rounding of the norm itself
    }
    if (lane == 0) t.scale[r] = fp16_operand_scale(bound);
  }
}

// Rotate k-block `kb` of the first kRows rows into t.a as fp16 (rows past `rows` zero).
template <int kRows, typename In>
__device__ void prepare_a(const Tile &t, int rows, int kb, const __half *suh, In in) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int r = warp; r < kRows; r += kThreads / 32) {
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r < rows) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int k = kb * 128 + 4 * lane + q;
        v[q] = in(r, k) * __half2float(suh[k]);
      }
      warp_hadamard128(v);
    }
    const float s = t.scale[r];
    *reinterpret_cast<uint2 *>(&t.a[r][4 * lane]) =
        make_uint2(pack_half2(v[0] * s, v[1] * s), pack_half2(v[2] * s, v[3] * s));
  }
}

// Stage the weight tiles of k-block `kb`: for each of its 8 k-tiles, `runs` runs of 8
// consecutive tiles starting at tile column run_start[r], into stage tile slots kt * 8 * runs + 8 r.
template <int K2>
__device__ __forceinline__ void stage_weights(uint32_t *stage, const uint32_t *trellis, int tiles_n, int kb, int runs,
                                              const int *run_start) {
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int chunks_per_run = 8 * words / 4;  // 16-byte chunks
  const int total = 8 * runs * chunks_per_run;
  for (int c = threadIdx.x; c < total; c += kThreads) {
    const int run = c / chunks_per_run;
    const int within = c % chunks_per_run;
    const int kt = run / runs;
    const int r = run % runs;
    const uint32_t *src = trellis + (static_cast<size_t>(kb * 8 + kt) * tiles_n + run_start[r]) * words + within * 4;
    uint32_t *dst = stage + (kt * 8 * runs + 8 * r) * words + within * 4;
    cp_async16(dst, src);
  }
}

// One k-block of MMAs for the warp: MT m16 blocks (rows warp_m * 16 * MT ..), NT tiles at stage
// slots slot0 + n of each k-tile (8 * runs tiles per k-tile).
template <int K2, int MT, int NT>
__device__ __forceinline__ void mma_block(const Tile &t, const uint32_t *stage, int runs, int slot0, int warp_m, int rows,
                                          const ignis_trellis::LanePlan &plan, float (&acc)[MT][NT][2][4]) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
  const int row0 = warp_m * 16 * MT;
  if (row0 >= rows) return;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    uint32_t af[MT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
      ldmatrix_x4(af[mi], &t.a[row0 + mi * 16 + (lane & 15)][kt * 16 + (lane >> 4) * 8]);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const uint32_t *tile = stage + (kt * 8 * runs + slot0 + n) * words;
      uint32_t frag[4];
      ignis_trellis::decode_fragment<K2>(tile[plan.w0], tile[plan.w1], plan, frag);
#pragma unroll
      for (int mi = 0; mi < MT; ++mi) {
        if (row0 + mi * 16 < rows) {
          mma_f16(acc[mi][n][0], af[mi], frag[0], frag[1]);
          mma_f16(acc[mi][n][1], af[mi], frag[2], frag[3]);
        }
      }
    }
  }
}

__device__ __forceinline__ int item_rows(const Params &p, int e, int r0) {
  return min(kTileRows, static_cast<int>(p.offset[e + 1] - p.offset[e]) - r0);
}

// The warp layout of a tile. Wide items (more than 16 rows) put the 8 warps on a 2 x 4 grid of
// 32-row x (NT tiles) blocks; narrow items -- the small groups real routing is full of -- have
// one m16 block, so all 8 warps go along the columns and none sits idle.
template <bool kNarrow, int kTilesPerCta> struct WarpGrid {
  static constexpr int kMT = kNarrow ? 1 : 2;                                  // m16 blocks per warp
  static constexpr int kNT = kNarrow ? kTilesPerCta / 8 : kTilesPerCta / 4;    // n tiles per warp
  static constexpr int kRows = kNarrow ? 16 : kTileRows;                       // rows prepared
  __device__ static int warp_m(int warp) { return kNarrow ? 0 : warp >> 2; }
  __device__ static int slot0(int warp) { return kNarrow ? kNT * warp : kNT * (warp & 3); }
};

// acc / scale into t.y: row r, column (slot0 + n) * 16 + ...
template <class G>
__device__ __forceinline__ void store_tile(const Tile &t, int warp, const float (&acc)[G::kMT][G::kNT][2][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
#pragma unroll
  for (int mi = 0; mi < G::kMT; ++mi) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int r = G::warp_m(warp) * 16 * G::kMT + mi * 16 + g + hh * 8;
      const float inv = 1.0f / t.scale[r];
#pragma unroll
      for (int n = 0; n < G::kNT; ++n) {
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int col = (G::slot0(warp) + n) * 16 + hf * 8 + 2 * c;
          t.y[r][col] = acc[mi][n][hf][2 * hh] * inv;
          t.y[r][col + 1] = acc[mi][n][hf][2 * hh + 1] * inv;
        }
      }
    }
  }
}

template <int K2, bool kNarrow>
__device__ void gate_up_tile(const Tile &t, const Params &p, int e, int r0, int b, const ignis_moe_slot &slot) {
  using G = WarpGrid<kNarrow, 16>;
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int stage_words = 8 * 16 * words;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rows = item_rows(p, e, r0);
  const uint32_t first_row = p.offset[e] + r0;
  const RecordPlanes planes = record_planes(slot, kHidden, kGateUpOut, K2);
  const __half *suh = planes.suh;
  const __half *svh = planes.svh;

  if (threadIdx.x < kTileRows) {
    t.token[threadIdx.x] = threadIdx.x < rows ? p.sorted[first_row + threadIdx.x] / kTopK : 0;
  }
  __syncthreads();
  auto x_in = [&](int r, int k) { return __bfloat162float(p.x[static_cast<size_t>(t.token[r]) * kHidden + k]); };
  row_scales<G::kRows>(t, rows, kHidden, suh, x_in);

  const int run_start[2] = {8 * b, kInter / 16 + 8 * b};
  stage_weights<K2>(t.b, planes.trellis, kGateUpOut / 16, 0, 2, run_start);
  cp_async_commit();
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
  float acc[G::kMT][G::kNT][2][4] = {};
  constexpr int kBlocks = kHidden / 128;
  for (int kb = 0; kb < kBlocks; ++kb) {
    // t.scale is visible, and every warp is done with t.a and with the stage refilled below.
    __syncthreads();
    if (kb + 1 < kBlocks) stage_weights<K2>(t.b + ((kb + 1) & 1) * stage_words, planes.trellis, kGateUpOut / 16, kb + 1, 2, run_start);
    cp_async_commit();
    prepare_a<G::kRows>(t, rows, kb, suh, x_in);
    cp_async_wait<1>();
    __syncthreads();
    mma_block<K2, G::kMT, G::kNT>(t, t.b + (kb & 1) * stage_words, 2, G::slot0(warp), G::warp_m(warp), rows, plan, acc);
  }
  cp_async_wait<0>();
  __syncthreads();

  // Epilogue: rows x 256 to shared memory (undoing the row scale), rotate back, svh, SwiGLU.
  store_tile<G>(t, warp, acc);
  __syncthreads();
  for (int r = warp; r < rows; r += kThreads / 32) {
    float gv[4], uv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      gv[q] = t.y[r][4 * lane + q];
      uv[q] = t.y[r][128 + 4 * lane + q];
    }
    warp_hadamard128(gv);
    warp_hadamard128(uv);
    float out[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int j = 128 * b + 4 * lane + q;
      out[q] = silu(gv[q] * __half2float(svh[j])) * (uv[q] * __half2float(svh[kInter + j]));
    }
    *reinterpret_cast<float4 *>(p.h + static_cast<size_t>(first_row + r) * kInter + 128 * b + 4 * lane) =
        make_float4(out[0], out[1], out[2], out[3]);
  }
}

template <int K2, bool kNarrow>
__device__ void down_tile(const Tile &t, const Params &p, int e, int r0, int cb, const ignis_moe_slot &slot) {
  using G = WarpGrid<kNarrow, 8>;
  constexpr int words = ignis_trellis::tile_words(K2);
  constexpr int stage_words = 8 * 8 * words;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rows = item_rows(p, e, r0);
  const uint32_t first_row = p.offset[e] + r0;
  const RecordPlanes planes = record_planes(slot, kInter, kHidden, K2);
  const __half *suh = planes.suh;
  const __half *svh = planes.svh;

  if (threadIdx.x < kTileRows) {
    const int a = threadIdx.x < rows ? p.sorted[first_row + threadIdx.x] : 0;
    t.token[threadIdx.x] = a / kTopK;
    t.weight[threadIdx.x] = threadIdx.x < rows ? p.weights[a] : 0.0f;
  }
  __syncthreads();
  auto h_in = [&](int r, int k) { return p.h[static_cast<size_t>(first_row + r) * kInter + k]; };
  row_scales<G::kRows>(t, rows, kInter, suh, h_in);

  const int run_start[1] = {8 * cb};
  stage_weights<K2>(t.b, planes.trellis, kHidden / 16, 0, 1, run_start);
  cp_async_commit();
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
  float acc[G::kMT][G::kNT][2][4] = {};
  constexpr int kBlocks = kInter / 128;
  for (int kb = 0; kb < kBlocks; ++kb) {
    __syncthreads();
    if (kb + 1 < kBlocks) stage_weights<K2>(t.b + ((kb + 1) & 1) * stage_words, planes.trellis, kHidden / 16, kb + 1, 1, run_start);
    cp_async_commit();
    prepare_a<G::kRows>(t, rows, kb, suh, h_in);
    cp_async_wait<1>();
    __syncthreads();
    mma_block<K2, G::kMT, G::kNT>(t, t.b + (kb & 1) * stage_words, 1, G::slot0(warp), G::warp_m(warp), rows, plan, acc);
  }
  cp_async_wait<0>();
  __syncthreads();

  store_tile<G>(t, warp, acc);
  __syncthreads();
  for (int r = warp; r < rows; r += kThreads / 32) {
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = t.y[r][4 * lane + q];
    warp_hadamard128(v);
    const float w = t.weight[r];
    long long *dst = p.acc + static_cast<size_t>(t.token[r]) * kHidden + 128 * cb + 4 * lane;
#pragma unroll
    for (int q = 0; q < 4; ++q) add_fixed(dst + q, w * (v[q] * __half2float(svh[128 * cb + 4 * lane + q])));
  }
}

__global__ void __launch_bounds__(kThreads) prefill_gate_up_kernel(Params p) {
  if (blockIdx.x >= *p.item_count) return;
  extern __shared__ __align__(16) unsigned char smem[];
  const Tile t = carve(smem);
  const int2 item = p.items[blockIdx.x];
  const ignis_moe_slot slot = load_slot(p.slots, item.x, IGNIS_MOE_PROJ_GATE_UP);
  const bool narrow = item_rows(p, item.x, item.y) <= 16;
  dispatch_k2(slot.k2, [&](auto k2) {
    if (narrow) {
      gate_up_tile<decltype(k2)::value, true>(t, p, item.x, item.y, blockIdx.y, slot);
    } else {
      gate_up_tile<decltype(k2)::value, false>(t, p, item.x, item.y, blockIdx.y, slot);
    }
  });
}

__global__ void __launch_bounds__(kThreads) prefill_down_kernel(Params p) {
  if (blockIdx.x >= *p.item_count) return;
  extern __shared__ __align__(16) unsigned char smem[];
  const Tile t = carve(smem);
  const int2 item = p.items[blockIdx.x];
  const ignis_moe_slot slot = load_slot(p.slots, item.x, IGNIS_MOE_PROJ_DOWN);
  const bool narrow = item_rows(p, item.x, item.y) <= 16;
  dispatch_k2(slot.k2, [&](auto k2) {
    if (narrow) {
      down_tile<decltype(k2)::value, true>(t, p, item.x, item.y, blockIdx.y, slot);
    } else {
      down_tile<decltype(k2)::value, false>(t, p, item.x, item.y, blockIdx.y, slot);
    }
  });
}

}  // namespace

int32_t prepare_prefill() {
  const cudaError_t a1 =
      cudaFuncSetAttribute(prefill_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
  const cudaError_t a2 =
      cudaFuncSetAttribute(prefill_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
  if (a1 != cudaSuccess || a2 != cudaSuccess) {
    return fail(std::string("ignis_moe_prepare (prefill): ") + cudaGetErrorString(a1 != cudaSuccess ? a1 : a2));
  }
  return 0;
}

}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" int32_t ignis_moe_experts_prefill(const void *x, uint32_t tokens, const int32_t *ids,
                                             const float *weights, const struct ignis_moe_slot *slots,
                                             const struct ignis_moe_workspace *workspace, int64_t *acc, void *stream) {
  if (x == nullptr || ids == nullptr || weights == nullptr || slots == nullptr || acc == nullptr) {
    return fail("ignis_moe_experts_prefill: NULL pointer");
  }
  if (check_workspace("ignis_moe_experts_prefill", workspace) != 0) return -1;
  if (tokens == 0 || tokens > workspace->prefill_tokens) {
    return fail("ignis_moe_experts_prefill: tokens must be 1.." + std::to_string(workspace->prefill_tokens) +
                " (the workspace's prefill_tokens)");
  }
  if (require_prepared("ignis_moe_experts_prefill", nullptr) != 0) return -1;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const WorkspaceLayout l = workspace_layout(workspace->decode_tokens, workspace->prefill_tokens);
  char *ws = static_cast<char *>(workspace->base);
  uint32_t *hist = reinterpret_cast<uint32_t *>(ws + l.chunk_hist);
  Params p;
  p.x = static_cast<const __nv_bfloat16 *>(x);
  p.weights = weights;
  p.slots = slots;
  p.offset = reinterpret_cast<const uint32_t *>(ws + l.expert_offset);
  p.item_count = reinterpret_cast<const uint32_t *>(ws + l.item_count);
  p.items = reinterpret_cast<const int2 *>(ws + l.items);
  p.sorted = reinterpret_cast<const int32_t *>(ws + l.sorted);
  p.h = reinterpret_cast<float *>(ws + l.prefill_h);
  p.acc = reinterpret_cast<long long *>(acc);

  const int chunks = static_cast<int>(group_chunks(tokens));
  group_count_kernel<<<chunks, 256, 0, s>>>(ids, static_cast<int>(tokens), hist);
  group_scan_kernel<<<1, kExperts, 0, s>>>(chunks, hist, const_cast<uint32_t *>(p.offset), const_cast<uint32_t *>(p.item_count),
                                           const_cast<int2 *>(p.items));
  group_scatter_kernel<<<chunks, kGroupChunk * kTopK, 0, s>>>(ids, static_cast<int>(tokens), hist, const_cast<int32_t *>(p.sorted));
  if (check_launch("ignis_moe_experts_prefill (grouping)") != 0) return -1;
  const uint32_t items = max_items(tokens);
  prefill_gate_up_kernel<<<dim3(items, kGateUpBlocks), kThreads, kSmemBytes, s>>>(p);
  if (check_launch("ignis_moe_experts_prefill (gate/up)") != 0) return -1;
  prefill_down_kernel<<<dim3(items, kDownBlocks), kThreads, kSmemBytes, s>>>(p);
  return check_launch("ignis_moe_experts_prefill (down)");
}
