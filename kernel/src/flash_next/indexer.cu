// ignis kernel leaf -- the Flash-Next QSA indexer (spec flash-next/04, GitHub #302, slice S3):
// OURS (ADR 0043), no port claim. The math, its numerics and the tie rule are stated in
// indexer.h; this file is the stages' kernels (fn_indexer_select, the program's entry point, is
// indexer_entry.cu).
//
//   append_blocks  one warp per block the call completes: pool its four raw keys (the earlier
//                  ones from the lane's tail), k_layernorm, rope at the block's first position,
//                  write the key to its page.
//   append_tail    one CTA per lane: the raw keys of the lane's incomplete last block.
//   queries        one warp per (row, head): q_layernorm and rope at the row's position.
//   score          a CTA per (row tile, 64 blocks): the tile's block keys and queries in shared
//                  memory, sum_h relu(q_h . k) in fp32 per (row, block).
//   select         one CTA per row: an MSB-first radix select of the k-th largest score, then
//                  the selected blocks in ascending order -- every score above the k-th, and
//                  the lowest-index blocks among those equal to it -- and the row's tail tokens.

#include "indexer.h"

#include <cmath>

namespace ignis::flash_next::indexer {
namespace {

constexpr int32_t kBlocksPerScoreCta = 64;
constexpr int32_t kSelectThreads = 512;
constexpr int32_t kKeysWarps = 4;

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  return v;
}

// The checkpoint's (1 + w) RMSNorm of one 128-wide row held as 4 values per lane (dims lane +
// 32 i), rounded once to BF16: x * (1 / sqrt(mean(x^2) + eps)) * (1 + w), each step in fp32.
__device__ __forceinline__ void rms_norm_row(float (&x)[4], const __nv_bfloat16 *w, float eps, int lane) {
  float ss = 0.0F;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss = fmaf(x[i], x[i], ss);
  const float mean = warp_sum(ss) / static_cast<float>(kHeadDim);
  const float r = 1.0F / sqrtf(mean + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = round_bf16((x[i] * r) * (1.0F + __bfloat162float(w[lane + 32 * i])));
}

// Rope on dims [0, 64) of a 128-wide row (pair i = (i, i + 32), held by lane i as x[0], x[1]) in
// torch's BF16 arithmetic: cos and sin rounded to BF16, each product rounded, then the sum.
__device__ __forceinline__ void rope_row(float (&x)[4], const Rope &rope, int32_t position, int lane) {
  const float phi = static_cast<float>(position) * rope.inv_freq[lane];
  const float c = round_bf16(cosf(phi));
  const float s = round_bf16(sinf(phi));
  const float x1 = x[0], x2 = x[1];
  x[0] = round_bf16(round_bf16(x1 * c) + round_bf16(-x2 * s));
  x[1] = round_bf16(round_bf16(x2 * c) + round_bf16(x1 * s));
}

struct KeysArgs {
  Paged paged;
  Rope rope;
  const __nv_bfloat16 *k_norm;
  const int32_t *slots;
  const int32_t *positions;
  const __nv_bfloat16 *qk;
  int32_t qk_stride;
  int32_t key_col;
  int32_t tokens;
  float eps;
};

__global__ void append_blocks_kernel(KeysArgs a) {
  const int seq = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int j = blockIdx.x * kKeysWarps + static_cast<int>(threadIdx.x >> 5);
  const int32_t p0 = a.positions[seq];
  const int32_t end = p0 + a.tokens;
  const int32_t b = p0 / kCompress + j;  // blocks completed by this call: [p0 / 4, end / 4)
  if (b >= end / kCompress) return;
  const int32_t slot = a.slots[seq];

  float x[4];
  for (int t = 0; t < kCompress; ++t) {
    const int32_t pos = b * kCompress + t;
    const __nv_bfloat16 *src =
        pos < p0 ? a.paged.tail_keys + (static_cast<size_t>(slot) * (kCompress - 1) + t) * kHeadDim
                 : a.qk + static_cast<size_t>(seq * a.tokens + (pos - p0)) * a.qk_stride + a.key_col;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float v = __bfloat162float(src[lane + 32 * i]);
      x[i] = t == 0 ? v : x[i] + v;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = round_bf16(x[i] / static_cast<float>(kCompress));
  rms_norm_row(x, a.k_norm, a.eps, lane);
  rope_row(x, a.rope, b * kCompress, lane);

  const int32_t first = b * kCompress;
  const int32_t page = a.paged.block_tables[static_cast<size_t>(slot) * a.paged.logical_pages + first / kPageTokens];
  __nv_bfloat16 *dst =
      a.paged.block_keys + (static_cast<size_t>(page) * kBlocksPerPage + (first % kPageTokens) / kCompress) * kHeadDim;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = __float2bfloat16_rn(x[i]);
}

__global__ void append_tail_kernel(KeysArgs a) {
  const int seq = blockIdx.x;
  const int32_t p0 = a.positions[seq];
  const int32_t end = p0 + a.tokens;
  const int32_t first = end / kCompress * kCompress;  // the incomplete last block's first position
  const int32_t slot = a.slots[seq];
  for (int32_t pos = first > p0 ? first : p0; pos < end; ++pos) {
    const __nv_bfloat16 *src = a.qk + static_cast<size_t>(seq * a.tokens + (pos - p0)) * a.qk_stride + a.key_col;
    a.paged.tail_keys[(static_cast<size_t>(slot) * (kCompress - 1) + (pos - first)) * kHeadDim + threadIdx.x] =
        src[threadIdx.x];
  }
}

struct RowArgs {
  const int32_t *slots;
  const int32_t *positions;
  int32_t tokens;
  int32_t row_begin;
  int32_t rows;
};

__device__ __forceinline__ int32_t row_seq(const RowArgs &r, int32_t local) { return (r.row_begin + local) / r.tokens; }

__device__ __forceinline__ int32_t row_position(const RowArgs &r, int32_t local) {
  const int32_t row = r.row_begin + local;
  return r.positions[row / r.tokens] + row % r.tokens;
}

__global__ void queries_kernel(RowArgs r, Rope rope, const __nv_bfloat16 *q_norm, const __nv_bfloat16 *qk,
                               int32_t qk_stride, int32_t heads, float eps, __nv_bfloat16 *q) {
  const int32_t local = blockIdx.x;
  const int h = static_cast<int>(threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16 *src = qk + static_cast<size_t>(r.row_begin + local) * qk_stride + h * kHeadDim;
  float x[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] = __bfloat162float(src[lane + 32 * i]);
  rms_norm_row(x, q_norm, eps, lane);
  rope_row(x, rope, row_position(r, local), lane);
  __nv_bfloat16 *dst = q + (static_cast<size_t>(local) * heads + h) * kHeadDim;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = __float2bfloat16_rn(x[i]);
}

// Shared-memory row pitch in 32-bit words: one pad word makes consecutive rows start in
// consecutive banks.
constexpr int32_t kKeyPitchWords = kHeadDim / 2 + 1;

template <int kRows, int kBlocksPerThread>
__global__ void __launch_bounds__(kRows * kBlocksPerScoreCta / kBlocksPerThread)
    score_kernel(RowArgs r, Paged paged, const __nv_bfloat16 *q, int32_t heads, float scale, float *scores,
                 int32_t score_stride) {
  constexpr int kThreads = kRows * kBlocksPerScoreCta / kBlocksPerThread;
  constexpr int kBlockGroups = kBlocksPerScoreCta / kBlocksPerThread;
  extern __shared__ uint32_t smem[];
  uint32_t *keys = smem;                                    // [64][kKeyPitchWords]
  uint32_t *qs = smem + kBlocksPerScoreCta * kKeyPitchWords;  // [kRows][heads * 64 + 1]
  const int32_t q_pitch = heads * (kHeadDim / 2) + 1;

  const int32_t row0 = blockIdx.y * kRows;
  const int32_t tile_rows = min(kRows, r.rows - row0);
  // Rows of a tile are one lane's consecutive tokens, so the last has the most blocks.
  const int32_t tile_blocks = (row_position(r, row0 + tile_rows - 1) + 1) / kCompress;
  const int32_t b0 = blockIdx.x * kBlocksPerScoreCta;
  if (b0 >= tile_blocks) return;
  const int32_t slot = r.slots[row_seq(r, row0)];

  for (int i = threadIdx.x; i < kBlocksPerScoreCta * (kHeadDim / 2); i += kThreads) {
    const int jj = i / (kHeadDim / 2), w = i % (kHeadDim / 2);
    const int32_t b = b0 + jj;
    uint32_t v = 0;
    if (b < tile_blocks) {
      const int32_t first = b * kCompress;
      const int32_t page = paged.block_tables[static_cast<size_t>(slot) * paged.logical_pages + first / kPageTokens];
      const uint32_t *src = reinterpret_cast<const uint32_t *>(
          paged.block_keys + (static_cast<size_t>(page) * kBlocksPerPage + (first % kPageTokens) / kCompress) * kHeadDim);
      v = src[w];
    }
    keys[jj * kKeyPitchWords + w] = v;
  }
  for (int i = threadIdx.x; i < kRows * heads * (kHeadDim / 2); i += kThreads) {
    const int rr = i / (heads * (kHeadDim / 2)), w = i % (heads * (kHeadDim / 2));
    qs[rr * q_pitch + w] =
        rr < tile_rows ? reinterpret_cast<const uint32_t *>(q + static_cast<size_t>(row0 + rr) * heads * kHeadDim)[w] : 0U;
  }
  __syncthreads();

  const int rr = threadIdx.x % kRows;
  const int group = threadIdx.x / kRows;
  if (rr >= tile_rows) return;
  float acc[kBlocksPerThread][4];
#pragma unroll
  for (int k = 0; k < kBlocksPerThread; ++k)
#pragma unroll
    for (int h = 0; h < 4; ++h) acc[k][h] = 0.0F;
  const uint32_t *qrow = qs + rr * q_pitch;
#pragma unroll 4
  for (int w = 0; w < kHeadDim / 2; ++w) {
    float2 qv[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const uint32_t bits = qrow[h * (kHeadDim / 2) + w];
      qv[h] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits));
    }
#pragma unroll
    for (int k = 0; k < kBlocksPerThread; ++k) {
      const uint32_t bits = keys[(group + k * kBlockGroups) * kKeyPitchWords + w];
      const float2 kv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits));
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        acc[k][h] = fmaf(qv[h].x, kv.x, acc[k][h]);  // BF16 products are exact in fp32
        acc[k][h] = fmaf(qv[h].y, kv.y, acc[k][h]);
      }
    }
  }
  const int32_t blocks = (row_position(r, row0 + rr) + 1) / kCompress;
  float *out = scores + static_cast<size_t>(row0 + rr) * score_stride;
#pragma unroll
  for (int k = 0; k < kBlocksPerThread; ++k) {
    const int32_t b = b0 + group + k * kBlockGroups;
    if (b >= blocks) continue;
    float s = 0.0F;
#pragma unroll
    for (int h = 0; h < 4; ++h) s += fmaxf(acc[k][h], 0.0F);
    out[b] = s * scale;
  }
}

// A score's radix key: scores are >= 0, so their bits order like their values; -0 and NaN count
// as 0.
__device__ __forceinline__ uint32_t score_key(float s) { return s > 0.0F ? __float_as_uint(s) : 0U; }

// Exclusive prefix count of `flag` over the CTA (thread order), and the CTA's total.
__device__ __forceinline__ int cta_exclusive_count(bool flag, int *warp_totals, int *total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const unsigned ballot = __ballot_sync(0xFFFFFFFFu, flag);
  const int in_warp = __popc(ballot & ((1U << lane) - 1U));
  if (lane == 0) warp_totals[warp] = __popc(ballot);
  __syncthreads();
  int before = 0, all = 0;
  for (int w = 0; w < kSelectThreads / 32; ++w) {
    const int t = warp_totals[w];
    before += w < warp ? t : 0;
    all += t;
  }
  __syncthreads();
  *total = all;
  return before + in_warp;
}

struct SelectArgs {
  RowArgs r;
  const float *scores;
  int32_t score_stride;
  int32_t block_topk;
  int32_t width;
  int32_t *tokens;  // at the batch's global rows
  int32_t *counts;
};

__global__ void __launch_bounds__(kSelectThreads) select_kernel(SelectArgs a) {
  __shared__ int hist[256];
  __shared__ int warp_totals[kSelectThreads / 32];
  __shared__ uint32_t s_prefix;
  __shared__ int s_remaining;

  const int32_t local = blockIdx.x;
  const int32_t row = a.r.row_begin + local;
  const int32_t visible = row_position(a.r, local) + 1;
  const int32_t blocks = visible / kCompress;
  int32_t *out = a.tokens + static_cast<size_t>(row) * a.width;

  if (blocks <= a.block_topk) {  // every visible token
    for (int t = threadIdx.x; t < a.width; t += kSelectThreads) out[t] = t < visible ? t : -1;
    if (threadIdx.x == 0) a.counts[row] = visible;
    return;
  }

  const float *s = a.scores + static_cast<size_t>(local) * a.score_stride;
  uint32_t prefix = 0, mask = 0;
  int remaining = a.block_topk;  // the rank, among keys matching the prefix, of the k-th largest
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += kSelectThreads) hist[i] = 0;
    __syncthreads();
    for (int32_t b = threadIdx.x; b < blocks; b += kSelectThreads) {
      const uint32_t key = score_key(s[b]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255U], 1);
    }
    __syncthreads();
    if (threadIdx.x < 32) {  // the digit where the count from the top reaches `remaining`
      const int lane = threadIdx.x;
      int own = 0;  // bins 255 - 8 lane - 7 .. 255 - 8 lane, highest first
#pragma unroll
      for (int k = 0; k < 8; ++k) own += hist[255 - 8 * lane - k];
      int incl = own;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_up_sync(0xFFFFFFFFu, incl, off);
        if (lane >= off) incl += v;
      }
      const int excl = incl - own;
      if (excl < remaining && remaining <= incl) {
        int cum = excl;
        for (int k = 0; k < 8; ++k) {
          const int bin = 255 - 8 * lane - k;
          if (cum + hist[bin] >= remaining) {
            s_prefix = prefix | (static_cast<uint32_t>(bin) << shift);
            s_remaining = remaining - cum;
            break;
          }
          cum += hist[bin];
        }
      }
    }
    __syncthreads();
    prefix = s_prefix;
    remaining = s_remaining;
    mask |= 255U << shift;
    __syncthreads();
  }
  // prefix is the k-th largest key; `remaining` of the keys equal to it are taken, lowest first.
  const uint32_t kth = prefix;
  const int take_equal = remaining;
  int taken = 0, equal_seen = 0;
  for (int32_t base = 0; base < blocks; base += kSelectThreads) {
    const int32_t b = base + threadIdx.x;
    const uint32_t key = b < blocks ? score_key(s[b]) : 0U;
    const bool equal = b < blocks && key == kth;
    int equal_total = 0;
    const int equal_before = equal_seen + cta_exclusive_count(equal, warp_totals, &equal_total);
    const bool take = b < blocks && (key > kth || (equal && equal_before < take_equal));
    int take_total = 0;
    const int at = taken + cta_exclusive_count(take, warp_totals, &take_total);
    if (take) {
#pragma unroll
      for (int t = 0; t < kCompress; ++t) out[at * kCompress + t] = b * kCompress + t;
    }
    taken += take_total;
    equal_seen += equal_total;
  }
  const int32_t tail = visible - blocks * kCompress;
  const int32_t count = a.block_topk * kCompress + tail;
  for (int t = threadIdx.x; t < a.width; t += kSelectThreads) {
    if (t >= a.block_topk * kCompress) out[t] = t < count ? blocks * kCompress + (t - a.block_topk * kCompress) : -1;
  }
  if (threadIdx.x == 0) a.counts[row] = count;
}

Status launched(const char *what) { return cudaPeekAtLastError() == cudaSuccess ? nullptr : what; }

RowArgs row_args(const Batch &batch, int32_t row_begin, int32_t rows) {
  return RowArgs{batch.slots, batch.positions, batch.tokens, row_begin, rows};
}

size_t align256(size_t v) { return (v + 255) / 256 * 256; }

}  // namespace

Rope rope_from(const ninfer::ops::RopeFrequencies &frequencies) {
  Rope rope;
  for (int i = 0; i < kRotaryDim / 2; ++i) {
    rope.inv_freq[i] = 1.0F / static_cast<float>(1.0 / frequencies.inv_frequency[i]);
  }
  return rope;
}

Status check_geometry(const Geometry &g) {
  if (g.indexer_head_dim != kHeadDim) return "indexer: head dim is not 128";
  if (g.indexer_heads != 4) return "indexer: the score kernel holds 4 query heads";
  if (g.indexer_kv_heads != 1) return "indexer: one key head expected";
  if (g.compress_ratio != kCompress) return "indexer: compress ratio is not 4";
  if (g.rotary_dim != kRotaryDim) return "indexer: rotary dim is not 64";
  if (g.indexer_budget % kCompress != 0 || g.indexer_budget <= 0) return "indexer: budget is not a positive multiple of 4";
  return nullptr;
}

Status append_keys(const Geometry &g, const Paged &paged, const Rope &rope, const void *k_norm,
                   const Batch &batch, const void *qk, cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  if (batch.lanes <= 0 || batch.tokens <= 0) return "indexer: empty batch";
  KeysArgs a{paged,
             rope,
             static_cast<const __nv_bfloat16 *>(k_norm),
             batch.slots,
             batch.positions,
             static_cast<const __nv_bfloat16 *>(qk),
             (g.indexer_heads + g.indexer_kv_heads) * kHeadDim,
             g.indexer_heads * kHeadDim,
             batch.tokens,
             g.rms_norm_eps};
  const int32_t blocks_per_lane = batch.tokens / kCompress + 1;
  const dim3 grid((blocks_per_lane + kKeysWarps - 1) / kKeysWarps, batch.lanes);
  append_blocks_kernel<<<grid, kKeysWarps * 32, 0, stream>>>(a);
  if (const Status st = launched("indexer: append_blocks launch failed")) return st;
  append_tail_kernel<<<batch.lanes, kHeadDim, 0, stream>>>(a);
  return launched("indexer: append_tail launch failed");
}

Status prepare_queries(const Geometry &g, const Rope &rope, const void *q_norm, const Batch &batch,
                       int32_t row_begin, int32_t rows, const void *qk, __nv_bfloat16 *q,
                       cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  if (rows <= 0) return nullptr;
  queries_kernel<<<rows, g.indexer_heads * 32, 0, stream>>>(
      row_args(batch, row_begin, rows), rope, static_cast<const __nv_bfloat16 *>(q_norm),
      static_cast<const __nv_bfloat16 *>(qk), (g.indexer_heads + g.indexer_kv_heads) * kHeadDim, g.indexer_heads,
      g.rms_norm_eps, q);
  return launched("indexer: queries launch failed");
}

Status score(const Geometry &g, const Paged &paged, const Batch &batch, int32_t row_begin, int32_t rows,
             const __nv_bfloat16 *q, int32_t max_blocks, float *scores, int32_t score_stride,
             cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  if (rows <= 0 || max_blocks <= 0) return nullptr;
  if (score_stride < max_blocks) return "indexer: score stride below the block bound";
  const float scale = 1.0F / sqrtf(static_cast<float>(kHeadDim));
  const RowArgs r = row_args(batch, row_begin, rows);
  const int heads = g.indexer_heads;
  const size_t q_bytes_per_row = (static_cast<size_t>(heads) * (kHeadDim / 2) + 1) * 4;
  const size_t key_bytes = static_cast<size_t>(kBlocksPerScoreCta) * kKeyPitchWords * 4;
  const unsigned bx = static_cast<unsigned>((max_blocks + kBlocksPerScoreCta - 1) / kBlocksPerScoreCta);
  if (batch.tokens == 1) {
    score_kernel<1, 1><<<dim3(bx, rows), 64, key_bytes + q_bytes_per_row, stream>>>(r, paged, q, heads, scale,
                                                                                     scores, score_stride);
  } else {
    constexpr int kRows = 16;
    score_kernel<kRows, 4><<<dim3(bx, (rows + kRows - 1) / kRows), kRows * 16, key_bytes + kRows * q_bytes_per_row,
                             stream>>>(r, paged, q, heads, scale, scores, score_stride);
  }
  return launched("indexer: score launch failed");
}

Status select_blocks(const Geometry &g, const Batch &batch, int32_t row_begin, int32_t rows,
                     const float *scores, int32_t score_stride, int32_t *tokens, int32_t *counts,
                     cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  if (rows <= 0) return nullptr;
  SelectArgs a{row_args(batch, row_begin, rows), scores, score_stride, g.indexer_budget / kCompress,
               g.selection_width(), tokens, counts};
  select_kernel<<<rows, kSelectThreads, 0, stream>>>(a);
  return launched("indexer: select launch failed");
}

std::size_t select_scratch_bytes(const Geometry &g, int32_t rows, int32_t max_visible) {
  const size_t wave = static_cast<size_t>(rows < kWaveRows ? rows : kWaveRows);
  const size_t blocks = static_cast<size_t>(max_visible / kCompress > 0 ? max_visible / kCompress : 1);
  return align256(wave * g.indexer_heads * kHeadDim * sizeof(__nv_bfloat16)) + align256(wave * blocks * sizeof(float));
}

Status select(const Geometry &g, const Paged &paged, const Rope &rope, const void *q_norm,
              const Batch &batch, const void *qk, Selection &out, ninfer::DeviceArena &scratch,
              cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  out.dense = batch.max_visible <= g.dense_threshold();
  if (out.dense) return nullptr;
  const int32_t rows = batch.rows();
  const int32_t max_blocks = batch.max_visible / kCompress;
  auto scope = scratch.scope();
  const int32_t wave = rows < kWaveRows ? rows : kWaveRows;
  auto *q = static_cast<__nv_bfloat16 *>(
      scratch.alloc_bytes(static_cast<size_t>(wave) * g.indexer_heads * kHeadDim * sizeof(__nv_bfloat16)).data);
  auto *scores = static_cast<float *>(scratch.alloc_bytes(static_cast<size_t>(wave) * max_blocks * sizeof(float)).data);
  // Waves never straddle lanes while a lane has several tokens (a score tile shares one lane's keys).
  const int32_t span = batch.tokens == 1 ? rows : batch.tokens;
  for (int32_t lane_first = 0; lane_first < rows; lane_first += span) {
    for (int32_t t0 = 0; t0 < span; t0 += wave) {
      const int32_t begin = lane_first + t0;
      const int32_t n = span - t0 < wave ? span - t0 : wave;
      if (const Status st = prepare_queries(g, rope, q_norm, batch, begin, n, qk, q, stream)) return st;
      if (const Status st = score(g, paged, batch, begin, n, q, max_blocks, scores, max_blocks, stream)) return st;
      if (const Status st = select_blocks(g, batch, begin, n, scores, max_blocks, out.tokens, out.counts, stream)) {
        return st;
      }
    }
  }
  return nullptr;
}

}  // namespace ignis::flash_next::indexer
