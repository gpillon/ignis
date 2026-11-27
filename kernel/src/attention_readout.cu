// The attention readout (GitHub #260, ADR 0038; the head set GitHub #263;
// its rows over a text span GitHub #275, ADR 0041). See
// attention_readout.h.
//
// One warp per key: eight coordinates per lane against a query held in shared
// memory, reduced across the warp, scaled once. The query is rotated into the
// codec's frame first under hq-e8-2b -- the same orthonormal rotation the
// test-only tap's host mirror applies (`hq_rotate`, crates/core/src/attn_tap.rs)
// and that the hq keys are stored in -- so the score is the frame-free
// `q . k / sqrt(head_dim)` either way. Under hq-e8-2b a warp fetches its keys
// four at a time, one per 8-lane group, exactly as the prompt attention fetched
// them (GitHub #335, hq_prompt_keys.cuh), into shared memory first.
#include "attention_readout.h"

#include "hq_prompt_keys.cuh"

#include "ops/kernel/hq_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

namespace {

constexpr int kHeadDim = 256;
constexpr int kKvHeads = 4;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
static_assert(kThreads == kHeadDim, "one thread per query coordinate while it is rotated");
// GitHub #335: the keys a warp fetches at once under hq-e8-2b, one per 8-lane group.
constexpr int kFetchRows = 4;

// The hq rows a block's warps fetch into, and the signs the fetch rotates the
// chunk's own rows with.
struct HqStage {
  __nv_bfloat16 rows[kWarpsPerBlock][kFetchRows][kHeadDim];
  std::int8_t signs[kHeadDim];
};

// Keys [k0, k0 + kFetchRows) of the span into `stage` for this warp, as the
// prompt attention read them; a key past the span is a zero row nobody reads.
__device__ __forceinline__ void fetch_span_rows(const IgnisHqPromptKeys &keys, int kv_head,
                                                std::int64_t key_begin, std::int64_t key_count,
                                                std::int64_t k0, HqStage &stage, int warp) {
  const int group = (static_cast<int>(threadIdx.x) & 31) >> 3;
  const std::int64_t k = k0 + group;
  ignis_hq_prompt::fetch_rows_warp(keys, kv_head, false,
                                   k < key_count ? static_cast<int>(key_begin + k) : -1,
                                   stage.rows[warp][group], stage.signs);
}

// `kHq`: keys as ignis's prompt attention read them (`keys`); otherwise from the
// BF16 pages.
template <bool kHq>
__global__ void attention_readout_kernel(const __nv_bfloat16 *query, const IgnisHqPromptKeys keys,
                                         const __nv_bfloat16 *k_pages,
                                         const std::int32_t *block_table, std::int32_t kv_head,
                                         std::int64_t key_begin, std::int64_t key_count,
                                         float scale, float *scores) {
  __shared__ float q[kHeadDim];
  __shared__ __align__(16) HqStage stage;
  const int t = static_cast<int>(threadIdx.x);
  if (kHq) {
    ninfer::ops::hq_engine_signs_fill(stage.signs);
  }
  float value = __bfloat162float(query[t]);
  if (kHq) {
    // R = H * diag(signs) / 16: signs, then the natural-order Walsh-Hadamard
    // butterfly -- (a, b) -> (a + b, a - b) over pairs `len` apart -- then
    // 1/sqrt(256). Orthonormal, so R q . R k == q . k.
    q[t] = value * ninfer::ops::hq_engine_sign(t);
    __syncthreads();
    for (int len = 1; len < kHeadDim; len <<= 1) {
      const float a = q[t];
      const float b = q[t ^ len];
      __syncthreads();
      q[t] = (t & len) ? (b - a) : (a + b);
      __syncthreads();
    }
    value = q[t] * (1.0F / 16.0F);
    __syncthreads();
  }
  q[t] = value;
  __syncthreads();

  const int warp = t / 32;
  const int lane = t % 32;
  const int per_warp = kHq ? kFetchRows : 1;
  for (std::int64_t k0 = (static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * per_warp;
       k0 < key_count; k0 += static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock * per_warp) {
    if (kHq) {
      fetch_span_rows(keys, kv_head, key_begin, key_count, k0, stage, warp);
    }
    for (int r = 0; r < per_warp && k0 + r < key_count; ++r) {
      const std::int64_t k = k0 + r;
      const __nv_bfloat16 *row =
          kHq ? stage.rows[warp][r]
              : k_pages + ninfer::ops::paged_kv_element_offset<kHeadDim, kKvHeads>(
                              block_table, kv_head, static_cast<std::int32_t>(key_begin + k), 0);
      float acc = 0.0F;
#pragma unroll
      for (int j = 0; j < kHeadDim / 32; ++j) {
        const int d = lane + 32 * j;
        acc += q[d] * __bfloat162float(row[d]);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
      }
      if (lane == 0) {
        scores[k] = acc * scale;
      }
    }
    __syncwarp();  // the warp's rows are read before the next fetch rewrites them
  }
}

// GitHub #263 (ADR 0038): the most query heads one KV head serves -- the
// armed heads a block of the fused kernel holds at once.
constexpr int kGroup = 6;

// Order-preserving float -> uint in the high half, the key index in the low:
// `atomicMax` over these is an argmax, the larger index winning a tie. Every
// finite score packs above zero, so a zero slot means no key was scored.
__device__ __forceinline__ unsigned long long pack_score(float score, std::uint32_t key) {
  std::uint32_t u = __float_as_uint(score);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (static_cast<unsigned long long>(u) << 32) | key;
}

// GitHub #264 (ADR 0038): one gather launch's arguments. The fused kernel
// above has already reduced this layer's heads to their argmax; this reads the
// four keys around each of them so the host can place the peak inside its
// image cell.
struct NeighbourGatherArgs {
  const __nv_bfloat16 *query;
  IgnisHqPromptKeys keys;  // hq: where the prompt attention read its keys
  const __nv_bfloat16 *k_pages;
  const std::int32_t *block_table;
  std::int64_t key_begin;
  std::int64_t key_count;
  std::int32_t grid_cols;
  float scale;
  std::int32_t group;
  std::int32_t heads; // of the set, on this layer
  std::int32_t query_head[kReadoutLayerHeads];
  std::int32_t slot[kReadoutLayerHeads];
  const unsigned long long *best; // the fused launch's packed (score, key)
  float *neighbours;              // [4 * whole set], NaN off the grid
};

// One block per (head of this layer, neighbour): the block rotates its query
// head into shared memory exactly as the fused kernel does, scores the one key
// its neighbour sits on, and writes it. At most 24 heads x 4 -- 384 dot
// products against keys fetched as the prompt attention fetched them.
template <bool kHq>
__global__ void neighbour_gather_kernel(const NeighbourGatherArgs args) {
  __shared__ float q[kHeadDim];
  __shared__ __align__(16) HqStage stage;
  const int head = static_cast<int>(blockIdx.x);
  const int which = static_cast<int>(blockIdx.y);
  const int t = static_cast<int>(threadIdx.x);
  const int slot = args.slot[head];
  const std::uint32_t peak =
      static_cast<std::uint32_t>(args.best[slot] & 0xffffffffu);
  // The span is the image's grid, row-major: a neighbour off it has no score.
  const std::int32_t cols = args.grid_cols;
  const std::int64_t col = static_cast<std::int64_t>(peak) % cols;
  const std::int64_t row = static_cast<std::int64_t>(peak) / cols;
  const std::int64_t rows = (args.key_count + cols - 1) / cols;
  std::int64_t key = -1;
  switch (which) {
    case 0: key = col > 0 ? static_cast<std::int64_t>(peak) - 1 : -1; break;
    case 1: key = col + 1 < cols ? static_cast<std::int64_t>(peak) + 1 : -1; break;
    case 2: key = row > 0 ? static_cast<std::int64_t>(peak) - cols : -1; break;
    default: key = row + 1 < rows ? static_cast<std::int64_t>(peak) + cols : -1; break;
  }
  // The last grid row may be short: a key past the span is off the grid too.
  if (key >= args.key_count) {
    key = -1;
  }
  if (key < 0) {
    if (t == 0) {
      args.neighbours[4 * slot + which] = __builtin_nanf("");
    }
    return;
  }
  const int query_head = args.query_head[head];
  const int kv_head = query_head / args.group;
  float value = __bfloat162float(args.query[static_cast<std::int64_t>(query_head) * kHeadDim + t]);
  if (kHq) {
    // The codec's rotation, exactly as the fused kernel applies it.
    q[t] = value * ninfer::ops::hq_engine_sign(t);
    __syncthreads();
    for (int len = 1; len < kHeadDim; len <<= 1) {
      const float a = q[t];
      const float b = q[t ^ len];
      __syncthreads();
      q[t] = (t & len) ? (b - a) : (a + b);
      __syncthreads();
    }
    value = q[t] * (1.0F / 16.0F);
    __syncthreads();
  }
  q[t] = value;
  __syncthreads();
  const std::int64_t position = args.key_begin + key;
  if (kHq) {
    // The one row, by warp 0's first group (the others fetch none).
    ninfer::ops::hq_engine_signs_fill(stage.signs);
    __syncthreads();
    if (t < 32) {
      ignis_hq_prompt::fetch_rows_warp(args.keys, kv_head, false,
                                       t < 8 ? static_cast<int>(position) : -1,
                                       stage.rows[0][t >> 3], stage.signs);
    }
    __syncthreads();
  }
  const __nv_bfloat16 *krow =
      kHq ? stage.rows[0][0]
          : args.k_pages + ninfer::ops::paged_kv_element_offset<kHeadDim, kKvHeads>(
                               args.block_table, kv_head, static_cast<std::int32_t>(position), 0);
  float acc = q[t] * __bfloat162float(krow[t]);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  __shared__ float warp_sum[kWarpsPerBlock];
  const int warp = t / 32;
  if (t % 32 == 0) {
    warp_sum[warp] = acc;
  }
  __syncthreads();
  if (t == 0) {
    float total = 0.0F;
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      total += warp_sum[w];
    }
    args.neighbours[4 * slot + which] = total * args.scale;
  }
}

// One fused launch's arguments, by value: the heads armed on this layer and
// the keys their argmax skips live in the launch, not in device memory.
struct SetReadoutArgs {
  const __nv_bfloat16 *query; // the chunk's last column, [q_heads * 256]
  IgnisHqPromptKeys keys;     // hq: where the prompt attention read its keys
  const __nv_bfloat16 *k_pages;
  const std::int32_t *block_table;
  std::int64_t key_begin;
  std::int64_t key_count;
  float scale;
  std::int32_t group; // query heads per KV head
  std::int32_t heads; // armed query heads, each at most once
  std::int32_t query_head[kReadoutLayerHeads + 1];
  std::int32_t slot[kReadoutLayerHeads + 1]; // into `best`; -1 for a pointing head outside the set
  std::int32_t row_head;                     // whose scores go to `scores`; -1 none
  float *scores;
  float *rows; // GitHub #275: [whole set][key_count] by slot, or null
  unsigned long long *best;
  std::int32_t excluded_count;
  std::int32_t excluded[kReadoutMaxExcluded];
};

// Grid (key blocks) x (KV heads). Each block rotates the armed query heads of
// its KV head into shared memory once; each warp reads one key row into
// registers (under hq four at a time, through shared memory) and scores it
// against every one of them; each warp keeps its best
// (score, key) per head in registers, the block reduces them, and one
// `atomicMax` per block per head publishes it. The first prototype published
// one per key per head, on as many addresses as heads, and was 3-5x slower
// (spec 14, tools/pointing-scenes/bench_readout.cu).
template <bool kHq>
__global__ void attention_set_readout_kernel(const SetReadoutArgs args) {
  __shared__ float q[kGroup][kHeadDim];
  __shared__ std::int32_t mine_head[kGroup];
  __shared__ std::int32_t mine_slot[kGroup];
  __shared__ std::int32_t mine_count;
  __shared__ std::int32_t mine_row;
  __shared__ unsigned long long warp_best[kWarpsPerBlock][kGroup];
  __shared__ __align__(16) HqStage stage;
  const int kv_head = static_cast<int>(blockIdx.y);
  const int t = static_cast<int>(threadIdx.x);
  if (kHq) {
    ninfer::ops::hq_engine_signs_fill(stage.signs);
  }
  if (t == 0) {
    int n = 0;
    int row = -1;
    for (int i = 0; i < args.heads && n < kGroup; ++i) {
      if (args.query_head[i] / args.group == kv_head) {
        if (args.query_head[i] == args.row_head) {
          row = n;
        }
        mine_head[n] = args.query_head[i];
        mine_slot[n] = args.slot[i];
        ++n;
      }
    }
    mine_count = n;
    mine_row = row;
  }
  __syncthreads();
  const int armed = mine_count;
  if (armed == 0) {
    return; // block-uniform: this KV head holds no armed head
  }
  for (int m = 0; m < armed; ++m) {
    float value =
        __bfloat162float(args.query[static_cast<std::int64_t>(mine_head[m]) * kHeadDim + t]);
    if (kHq) {
      // The codec's rotation, exactly as the single-head kernel applies it.
      q[m][t] = value * ninfer::ops::hq_engine_sign(t);
      __syncthreads();
      for (int len = 1; len < kHeadDim; len <<= 1) {
        const float a = q[m][t];
        const float b = q[m][t ^ len];
        __syncthreads();
        q[m][t] = (t & len) ? (b - a) : (a + b);
        __syncthreads();
      }
      value = q[m][t] * (1.0F / 16.0F);
      __syncthreads();
    }
    q[m][t] = value;
  }
  __syncthreads();

  const int warp = t / 32;
  const int lane = t % 32;
  unsigned long long best[kGroup] = {};
  const int per_warp = kHq ? kFetchRows : 1;
  for (std::int64_t k0 = (static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * per_warp;
       k0 < args.key_count;
       k0 += static_cast<std::int64_t>(gridDim.x) * kWarpsPerBlock * per_warp) {
    if (kHq) {
      fetch_span_rows(args.keys, kv_head, args.key_begin, args.key_count, k0, stage, warp);
    }
    for (int r = 0; r < per_warp && k0 + r < args.key_count; ++r) {
      const std::int64_t k = k0 + r;
      const std::int64_t position = args.key_begin + k;
      const __nv_bfloat16 *row =
          kHq ? stage.rows[warp][r]
              : args.k_pages +
                    ninfer::ops::paged_kv_element_offset<kHeadDim, kKvHeads>(
                        args.block_table, kv_head, static_cast<std::int32_t>(position), 0);
      float key[kHeadDim / 32];
#pragma unroll
      for (int j = 0; j < kHeadDim / 32; ++j) {
        key[j] = __bfloat162float(row[lane + 32 * j]);
      }
      bool excluded = false;
      for (int e = 0; e < args.excluded_count; ++e) {
        excluded = excluded || args.excluded[e] == k;
      }
#pragma unroll
      for (int m = 0; m < kGroup; ++m) {
        if (m < armed) { // warp-uniform
          float acc = 0.0F;
#pragma unroll
          for (int j = 0; j < kHeadDim / 32; ++j) {
            acc += q[m][lane + 32 * j] * key[j];
          }
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
          }
          if (lane == 0) {
            const float score = acc * args.scale;
            if (m == mine_row) {
              args.scores[k] = score;
            }
            // GitHub #275: a set read in rows keeps every score it forms, at
            // its head's slot of the whole set.
            if (args.rows != nullptr && mine_slot[m] >= 0) {
              args.rows[static_cast<std::int64_t>(mine_slot[m]) * args.key_count + k] = score;
            }
            if (!excluded && mine_slot[m] >= 0) {
              const unsigned long long packed = pack_score(score, static_cast<std::uint32_t>(k));
              best[m] = packed > best[m] ? packed : best[m];
            }
          }
        }
      }
    }
    __syncwarp();  // the warp's rows are read before the next fetch rewrites them
  }
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < kGroup; ++m) {
      warp_best[warp][m] = best[m];
    }
  }
  __syncthreads();
  if (t < armed && mine_slot[t] >= 0) {
    unsigned long long block_best = 0;
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      block_best = warp_best[w][t] > block_best ? warp_best[w][t] : block_best;
    }
    if (block_best != 0) {
      atomicMax(&args.best[mine_slot[t]], block_best);
    }
  }
}

} // namespace

int32_t ignis_attention_readout_run(const AttentionReadoutTarget &target,
                                    const void *rotated_query, int32_t q_heads, int32_t kv_heads,
                                    int32_t tokens, int64_t visible_keys,
                                    const ninfer::PagedKVBatchLayerView &cache,
                                    const IgnisHqPromptKeys *hq_keys, float scale,
                                    cudaStream_t stream, const char **error) {
  const bool writes_row = target.device_scores != nullptr;
  bool geometry = kv_heads == kKvHeads && q_heads % kv_heads == 0 && tokens > 0 &&
                  (!writes_row || (target.query_head >= 0 && target.query_head < q_heads)) &&
                  target.set_heads >= 0 && target.set_heads <= kReadoutLayerHeads &&
                  target.excluded_count >= 0 && target.excluded_count <= kReadoutMaxExcluded &&
                  (writes_row || target.set_heads > 0) &&
                  (target.set_heads == 0 ||
                   (q_heads / kv_heads <= kGroup && target.device_set_best != nullptr &&
                    ((target.device_set_neighbours != nullptr && target.grid_cols > 0) ||
                     target.device_set_rows != nullptr)));
  for (int32_t i = 0; geometry && i < target.set_heads; ++i) {
    geometry = target.set_query_head[i] >= 0 && target.set_query_head[i] < q_heads &&
               target.set_slot[i] >= 0;
    for (int32_t j = 0; geometry && j < i; ++j) {
      geometry = target.set_query_head[j] != target.set_query_head[i];
    }
  }
  if (!geometry) {
    *error = "attention readout: the layer's head geometry is not the one it was built for";
    return -1;
  }
  const std::int64_t key_end = target.key_begin + target.key_count;
  // Keys past the history are not keys attention read.
  if (target.key_begin < 0 || target.key_count <= 0 || key_end > visible_keys) {
    return 0;
  }
  const bool hq = cache.dtype == ninfer::DType::U8;
  // Under hq-e8-2b only ignis's prompt attention says where its keys are: a
  // small-T route decodes in registers and names none.
  if (hq && hq_keys == nullptr) {
    return 0;
  }
  const IgnisHqPromptKeys keys = hq ? *hq_keys : IgnisHqPromptKeys{};
  const auto *last_column = static_cast<const __nv_bfloat16 *>(rotated_query) +
                            (static_cast<std::int64_t>(tokens) - 1) * q_heads * kHeadDim;
  const std::int64_t wanted_blocks = (target.key_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const unsigned blocks = static_cast<unsigned>(wanted_blocks < 1024 ? wanted_blocks : 1024);
  if (target.set_heads == 0) {
    // GitHub #260: the pointing head alone, one KV head, as before the set.
    const std::int32_t kv_head = target.query_head / (q_heads / kv_heads);
    const auto *query = last_column + static_cast<std::int64_t>(target.query_head) * kHeadDim;
    if (hq) {
      attention_readout_kernel<true><<<blocks, kThreads, 0, stream>>>(
          query, keys, nullptr, nullptr, kv_head, target.key_begin, target.key_count, scale,
          target.device_scores);
    } else {
      attention_readout_kernel<false><<<blocks, kThreads, 0, stream>>>(
          query, keys, static_cast<const __nv_bfloat16 *>(cache.k_pages.data),
          static_cast<const std::int32_t *>(cache.block_tables.data), kv_head, target.key_begin,
          target.key_count, scale, target.device_scores);
    }
  } else {
    // GitHub #263: every head of the set this layer holds -- and the pointing
    // head, when it is this layer's -- in one launch.
    SetReadoutArgs args{};
    args.query = last_column;
    args.keys = keys;
    args.k_pages = hq ? nullptr : static_cast<const __nv_bfloat16 *>(cache.k_pages.data);
    args.block_table = hq ? nullptr : static_cast<const std::int32_t *>(cache.block_tables.data);
    args.key_begin = target.key_begin;
    args.key_count = target.key_count;
    args.scale = scale;
    args.group = q_heads / kv_heads;
    args.heads = target.set_heads;
    bool row_in_set = false;
    for (int32_t i = 0; i < target.set_heads; ++i) {
      args.query_head[i] = target.set_query_head[i];
      args.slot[i] = target.set_slot[i];
      row_in_set = row_in_set || (writes_row && target.set_query_head[i] == target.query_head);
    }
    if (writes_row && !row_in_set) {
      args.query_head[args.heads] = target.query_head;
      args.slot[args.heads] = -1;
      ++args.heads;
    }
    args.row_head = writes_row ? target.query_head : -1;
    args.scores = target.device_scores;
    args.rows = target.device_set_rows;
    args.best = target.device_set_best;
    args.excluded_count = target.excluded_count;
    for (int32_t e = 0; e < target.excluded_count; ++e) {
      args.excluded[e] = target.excluded[e];
    }
    const dim3 grid(blocks, static_cast<unsigned>(kv_heads));
    if (hq) {
      attention_set_readout_kernel<true><<<grid, kThreads, 0, stream>>>(args);
    } else {
      attention_set_readout_kernel<false><<<grid, kThreads, 0, stream>>>(args);
    }
    // GitHub #264: the four keys around each of this layer's argmaxes, in a
    // second launch on the same stream -- the fused one above has to have
    // finished reducing them before they can be read. A set read in rows
    // over a text span (#275) has no grid, and names nowhere to put them.
    if (target.device_set_neighbours == nullptr) {
      const cudaError_t launched = cudaGetLastError();
      if (launched != cudaSuccess) {
        *error = cudaGetErrorString(launched);
        return -1;
      }
      *target.read = true;
      return 0;
    }
    NeighbourGatherArgs gather{};
    gather.query = args.query;
    gather.keys = args.keys;
    gather.k_pages = args.k_pages;
    gather.block_table = args.block_table;
    gather.key_begin = args.key_begin;
    gather.key_count = args.key_count;
    gather.grid_cols = target.grid_cols;
    gather.scale = args.scale;
    gather.group = args.group;
    gather.heads = target.set_heads;
    for (int32_t i = 0; i < target.set_heads; ++i) {
      gather.query_head[i] = target.set_query_head[i];
      gather.slot[i] = target.set_slot[i];
    }
    gather.best = target.device_set_best;
    gather.neighbours = target.device_set_neighbours;
    const dim3 gather_grid(static_cast<unsigned>(target.set_heads), 4);
    if (hq) {
      neighbour_gather_kernel<true><<<gather_grid, kThreads, 0, stream>>>(gather);
    } else {
      neighbour_gather_kernel<false><<<gather_grid, kThreads, 0, stream>>>(gather);
    }
  }
  const cudaError_t launched = cudaGetLastError();
  if (launched != cudaSuccess) {
    *error = cudaGetErrorString(launched);
    return -1;
  }
  *target.read = true;
  return 0;
}
