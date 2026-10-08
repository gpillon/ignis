// ignis kernel leaf -- the Flash-Next QSA sparse attention (spec flash-next/04, GitHub #302,
// slice S3): OURS (ADR 0043). What it computes and why it is shaped this way: qsa_sparse.h.
//
//   attend   one CTA of 4 warps per (row, KV head, split). The group's 12 query rows (padded to
//            16) are held as m16n8k16 A fragments for the whole list. Per 32-token tile: the
//            tile's positions, then its K and V rows gathered into shared memory (row pitch 264
//            BF16, so ldmatrix's eight rows hit eight different bank quads); S = Q K^T with each
//            warp on 8 tokens; the online softmax over the tile's 16 x 32 scores in fp32; O += P V
//            with each warp on 64 of the 256 output columns. One split writes O / l in BF16, several
//            write (O, m, l) partials.
//   combine  one CTA per (row, query head): the partials merged by their maxima.
//   hq rows  (hq-e8-2b only) eight lanes per (token, KV head, role) row: the fresh, residual or
//            codec source of qsa_sparse.h, a codec row decoded by the vendored hq_decode_row_group
//            into shared memory, then each warp un-rotates its four staged rows (the vendored
//            inverse FWHT with the engine signs) into the plain-frame BF16 scratch.

#include "qsa_sparse.h"

#include "ops/kernel/hq_codec.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cmath>

namespace ignis::flash_next::sparse {
namespace {

using ninfer::kPagedKVPageSize;
using ninfer::ops::kPagedKVPageMask;
using ninfer::ops::kPagedKVPageShift;

constexpr int kThreads = 128;
constexpr int kPitch = kHeadDim + 8;     // BF16 elements per K/V/Q row in shared memory
constexpr int kPPitch = kTileTokens + 8;  // BF16 elements per P row
constexpr int kRowsM = 16;                // the MMA's M: kGroup query heads, zero-padded
constexpr float kLog2e = 1.4426950408889634F;

// -inf without MSVC's INFINITY macro (an overflowing constant nvcc warns about).
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xFF800000); }

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

struct AttendArgs {
  KvSource kv;
  const int32_t *slots;
  int32_t tokens;  // tokens per lane: row r belongs to lane r / tokens
  int32_t q_heads;
  const __nv_bfloat16 *q;
  const int32_t *list;    // [rows][width]
  const int32_t *counts;  // [rows]
  int32_t width;
  int32_t splits;
  int32_t chunk;  // listed tokens per split, a multiple of kTileTokens
  __nv_bfloat16 *out;
  float *partial_o;   // [rows][kv_heads][splits][kGroup][kHeadDim]
  float2 *partial_ml;  // [rows][kv_heads][splits][kGroup]: (max, sum), max in log2 units
};

struct Smem {
  __nv_bfloat16 k[kTileTokens * kPitch];  // also stages Q before the loop
  __nv_bfloat16 v[kTileTokens * kPitch];
  float s[kRowsM][kTileTokens];
  __nv_bfloat16 p[kRowsM * kPPitch];
  float m[kRowsM], l[kRowsM], alpha[kRowsM];
  int32_t pos[kTileTokens];
};

__global__ void __launch_bounds__(kThreads) attend_kernel(AttendArgs a) {
  __shared__ __align__(16) Smem sm;
  const int split = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int32_t count = a.counts[row];
  const int32_t begin = split * a.chunk;
  const int32_t end = min(count, begin + a.chunk);
  const int32_t kv_heads = a.kv.kv_heads;

  // Q: the group's 12 query rows into the K buffer (rows 12..15 zero), then A fragments.
  const __nv_bfloat16 *qsrc = a.q + (static_cast<size_t>(row) * a.q_heads + kvh * kGroup) * kHeadDim;
  for (int c = tid; c < kRowsM * (kHeadDim / 8); c += kThreads) {
    const int r = c / (kHeadDim / 8), part = c % (kHeadDim / 8);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < kGroup) v = *reinterpret_cast<const uint4 *>(qsrc + r * kHeadDim + part * 8);
    *reinterpret_cast<uint4 *>(&sm.k[r * kPitch + part * 8]) = v;
  }
  if (tid < kRowsM) {
    sm.m[tid] = neg_inf();
    sm.l[tid] = 0.0F;
  }
  __syncthreads();
  uint32_t qf[kHeadDim / 16][4];
#pragma unroll
  for (int ks = 0; ks < kHeadDim / 16; ++ks) ldsm_x4(qf[ks], &sm.k[(lane & 15) * kPitch + ks * 16 + (lane >> 4) * 8]);
  __syncthreads();

  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0F;

  const int32_t slot = a.slots[row / a.tokens];
  const int32_t *list = a.list + static_cast<size_t>(row) * a.width;
  const float scale = kLog2e / sqrtf(static_cast<float>(kHeadDim));
  for (int32_t t0 = begin; t0 < end; t0 += kTileTokens) {
    if (tid < kTileTokens) sm.pos[tid] = t0 + tid < end ? list[t0 + tid] : -1;
    __syncthreads();
    for (int c = tid; c < 2 * kTileTokens * (kHeadDim / 8); c += kThreads) {
      const int role = c / (kTileTokens * (kHeadDim / 8));  // 0 K, 1 V
      const int cc = c % (kTileTokens * (kHeadDim / 8));
      const int t = cc / (kHeadDim / 8), part = cc % (kHeadDim / 8);
      const int32_t pos = sm.pos[t];
      uint4 v = make_uint4(0, 0, 0, 0);
      if (pos >= 0) {
        size_t at;
        if (a.kv.mode == KvSource::Mode::ByIndex) {
          at = ((static_cast<size_t>(row) * a.width + (t0 + t)) * kv_heads + kvh) * kHeadDim;
        } else if (a.kv.mode == KvSource::Mode::ByPosition) {
          at = (static_cast<size_t>(pos) * kv_heads + kvh) * kHeadDim;
        } else {
          const int32_t page =
              a.kv.block_tables[static_cast<size_t>(slot) * a.kv.logical_pages + (pos >> kPagedKVPageShift)];
          at = ((static_cast<size_t>(page) * kv_heads + kvh) * kPagedKVPageSize + (pos & kPagedKVPageMask)) * kHeadDim;
        }
        v = *reinterpret_cast<const uint4 *>((role == 0 ? a.kv.k : a.kv.v) + at + part * 8);
      }
      *reinterpret_cast<uint4 *>(&(role == 0 ? sm.k : sm.v)[t * kPitch + part * 8]) = v;
    }
    __syncthreads();

    // S = Q K^T: warp `warp` on tokens 8 warp .. 8 warp + 7.
    float s[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ks += 2) {
      uint32_t b[4];
      ldsm_x4(b, &sm.k[(8 * warp + (lane & 7)) * kPitch + ks * 16 + (lane >> 3) * 8]);
      mma_bf16(s, qf[ks], b[0], b[1]);
      mma_bf16(s, qf[ks + 1], b[2], b[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = g + (j >> 1) * 8, t = 8 * warp + 2 * tig + (j & 1);
      sm.s[r][t] = t0 + t < end ? s[j] * scale : neg_inf();
    }
    __syncthreads();

    // Online softmax: 8 threads per query row, 4 tokens each.
    {
      const int r = tid >> 3, part = tid & 7;
      float x[4], mt = neg_inf();
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        x[j] = sm.s[r][part * 4 + j];
        mt = fmaxf(mt, x[j]);
      }
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xFFFFFFFFu, mt, off));
      const float m_old = sm.m[r];
      const float m_new = fmaxf(m_old, mt);
      float lt = 0.0F;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const __nv_bfloat16 p = __float2bfloat16_rn(m_new == neg_inf() ? 0.0F : exp2f(x[j] - m_new));
        sm.p[r * kPPitch + part * 4 + j] = p;
        lt += __bfloat162float(p);  // the weights P V uses
      }
#pragma unroll
      for (int off = 1; off < 8; off <<= 1) lt += __shfl_xor_sync(0xFFFFFFFFu, lt, off);
      __syncwarp();
      if (part == 0) {
        const float alpha = m_new == neg_inf() ? 1.0F : exp2f(m_old - m_new);
        sm.alpha[r] = alpha;
        sm.l[r] = sm.l[r] * alpha + lt;
        sm.m[r] = m_new;
      }
    }
    __syncthreads();

    // O = alpha O + P V: warp `warp` on columns 64 warp .. 64 warp + 63.
    const float a_lo = sm.alpha[g], a_hi = sm.alpha[g + 8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      o[nt][0] *= a_lo;
      o[nt][1] *= a_lo;
      o[nt][2] *= a_hi;
      o[nt][3] *= a_hi;
    }
#pragma unroll
    for (int ks = 0; ks < kTileTokens / 16; ++ks) {
      uint32_t pa[4];
      ldsm_x4(pa, &sm.p[(lane & 15) * kPPitch + ks * 16 + (lane >> 4) * 8]);
#pragma unroll
      for (int nt = 0; nt < 8; nt += 2) {
        uint32_t b[4];
        ldsm_x4_trans(b, &sm.v[(ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kPitch + 64 * warp + nt * 8 +
                              (lane >> 4) * 8]);
        mma_bf16(o[nt], pa, b[0], b[1]);
        mma_bf16(o[nt + 1], pa, b[2], b[3]);
      }
    }
    __syncthreads();
  }

  // Rows g and g + 8 of this thread's fragments; only the first kGroup are heads.
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int r = g + half * 8;
    if (r >= kGroup) continue;
    if (a.splits == 1) {
      const float inv = sm.l[r] > 0.0F ? 1.0F / sm.l[r] : 0.0F;
      __nv_bfloat16 *dst = a.out + (static_cast<size_t>(row) * a.q_heads + kvh * kGroup + r) * kHeadDim;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = 64 * warp + nt * 8 + 2 * tig;
        *reinterpret_cast<__nv_bfloat162 *>(dst + col) =
            __floats2bfloat162_rn(o[nt][2 * half] * inv, o[nt][2 * half + 1] * inv);
      }
    } else {
      const size_t unit = (static_cast<size_t>(row) * kv_heads + kvh) * a.splits + split;
      float *dst = a.partial_o + (unit * kGroup + r) * kHeadDim;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = 64 * warp + nt * 8 + 2 * tig;
        *reinterpret_cast<float2 *>(dst + col) = make_float2(o[nt][2 * half], o[nt][2 * half + 1]);
      }
      if (warp == 0 && tig == 0) a.partial_ml[unit * kGroup + r] = make_float2(sm.m[r], sm.l[r]);
    }
  }
}

// With `gate`, the output gate of qsa.cu's gate_kernel on the combined value: the same two
// roundings, so the same bits as combining, then gating in a launch of its own.
__global__ void combine_kernel(const float *partial_o, const float2 *partial_ml, int32_t kv_heads, int32_t splits,
                               int32_t q_heads, __nv_bfloat16 *out, const __nv_bfloat16 *gate) {
  const int row = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int kvh = h / kGroup, r = h % kGroup;
  const size_t first = (static_cast<size_t>(row) * kv_heads + kvh) * splits;
  float m = neg_inf();
  for (int s = 0; s < splits; ++s) m = fmaxf(m, partial_ml[(first + s) * kGroup + r].x);
  float num = 0.0F, den = 0.0F;
  for (int s = 0; s < splits; ++s) {
    const float2 ml = partial_ml[(first + s) * kGroup + r];
    if (ml.y == 0.0F) continue;  // an empty split (its list ended before it)
    const float w = exp2f(ml.x - m);
    num += w * partial_o[((first + s) * kGroup + r) * kHeadDim + d];
    den += w * ml.y;
  }
  const size_t row_head = static_cast<size_t>(row) * q_heads + h;
  __nv_bfloat16 o = __float2bfloat16_rn(den > 0.0F ? num / den : 0.0F);
  if (gate != nullptr) {
    const float sig = __bfloat162float(
        __float2bfloat16_rn(1.0F / (1.0F + expf(-__bfloat162float(gate[row_head * 2 * kHeadDim + kHeadDim + d])))));
    o = __float2bfloat16_rn(__bfloat162float(o) * sig);
  }
  out[row_head * kHeadDim + d] = o;
}

constexpr int kHqThreads = 128;
constexpr int kHqUnits = kHqThreads / 8;  // rows per CTA, eight lanes each

struct HqArgs {
  HqSource hq;
  const int32_t *slots;
  const int32_t *positions;
  int32_t tokens;
  const int32_t *list;    // listed: [rows][width]
  const int32_t *counts;  // listed: [rows]
  int32_t width;
  int32_t visible;  // visible: positions [0, visible)
  __nv_bfloat16 *k;
  __nv_bfloat16 *v;
};

template <bool kListed>
__global__ void __launch_bounds__(kHqThreads) hq_rows_kernel(HqArgs a) {
  using namespace ninfer::ops;
  __shared__ __align__(16) __nv_bfloat16 staged[kHqUnits][kHqHeadDim];
  __shared__ __align__(16) std::uint16_t symbols[kHqUnits][kHqHeadDim];
  __shared__ std::int8_t signs[kHqHeadDim];
  __shared__ int rotated[kHqUnits];
  __shared__ __nv_bfloat16 *dsts[kHqUnits];
  hq_engine_signs_fill(signs);
  __syncthreads();

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int u = tid >> 3, lane8 = tid & 7;
  const int32_t kvh2 = a.hq.kv_heads * 2;
  const int32_t row = kListed ? static_cast<int32_t>(blockIdx.y) : 0;
  const int32_t unit = static_cast<int32_t>(blockIdx.x) * kHqUnits + u;
  const int32_t i = unit / kvh2;
  const int32_t head = (unit % kvh2) >> 1;
  const bool role_v = (unit & 1) != 0;
  const int32_t seq = kListed ? row / a.tokens : 0;
  const int32_t first = a.positions[seq];
  const int32_t slot = a.slots[seq];
  int32_t pos = -1;
  if (kListed) {
    if (i < a.width && i < a.counts[row]) pos = a.list[static_cast<size_t>(row) * a.width + i];
  } else if (i < a.visible) {
    pos = i;
  }
  bool rotated_row = false;  // a rotated row waits in staged[u] for the warp's un-rotation
  __nv_bfloat16 *dst = nullptr;
  if (pos >= 0) {
    dst = (role_v ? a.v : a.k) +
          ((kListed ? static_cast<size_t>(row) * a.width + i : static_cast<size_t>(pos)) * a.hq.kv_heads + head) *
              kHqHeadDim;
    const int32_t end = first + a.tokens;
    const bool fresh = a.hq.fresh_k != nullptr && pos >= first && pos < end;
    const int32_t window_end = a.hq.fresh_k != nullptr ? first : end;
    const bool side =
        !fresh && a.hq.residual_k != nullptr &&
        (pos < static_cast<int32_t>(kGqaHqSinkKeys) ||
         (pos >= window_end - static_cast<int32_t>(kGqaHqRecentKeys) && pos < window_end &&
          hq_ring_slot_valid(a.hq.ring_valid == nullptr
                                 ? nullptr
                                 : a.hq.ring_valid + static_cast<size_t>(slot) * (kGqaHqRecentKeys / 32),
                             pos)));
    if (fresh) {
      const __nv_bfloat16 *src = (role_v ? a.hq.fresh_v : a.hq.fresh_k) +
                                 ((static_cast<size_t>(seq) * a.tokens + (pos - first)) * a.hq.kv_heads + head) *
                                     kHqHeadDim;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        reinterpret_cast<uint4 *>(dst)[lane8 * 4 + j] = reinterpret_cast<const uint4 *>(src)[lane8 * 4 + j];
      }
    } else if (side) {
      const int32_t side_row = pos < static_cast<int32_t>(kGqaHqSinkKeys)
                                   ? pos
                                   : static_cast<int32_t>(kGqaHqSinkKeys) +
                                         (pos & (static_cast<int32_t>(kGqaHqRecentKeys) - 1));
      const __nv_bfloat16 *src =
          (role_v ? a.hq.residual_v : a.hq.residual_k) +
          ((static_cast<size_t>(slot) * (kGqaHqSinkKeys + kGqaHqRecentKeys) + side_row) * a.hq.kv_heads + head) *
              kHqHeadDim;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        reinterpret_cast<uint4 *>(staged[u])[lane8 * 4 + j] = reinterpret_cast<const uint4 *>(src)[lane8 * 4 + j];
      }
      rotated_row = true;
    } else {
      const int32_t page =
          a.hq.block_tables[static_cast<size_t>(slot) * a.hq.logical_pages + (pos >> kPagedKVPageShift)];
      const size_t row_at =
          (static_cast<size_t>(page) * a.hq.kv_heads + head) * kPagedKVPageSize + (pos & kPagedKVPageMask);
      hq_decode_row_group((role_v ? a.hq.v_codes : a.hq.k_codes) + row_at * kHqRowBudgetBytes,
                          (role_v ? a.hq.v_meta : a.hq.k_meta) + row_at * kHqMetaBytes, staged[u], lane8, 0,
                          hq_dither_row_seed(head, pos, role_v), symbols[u]);
      rotated_row = true;
    }
  }
  if (lane8 == 0) {
    rotated[u] = rotated_row;
    dsts[u] = dst;
  }
  __syncwarp();
  // Each warp un-rotates its own four units' staged rows.
  for (int q = 0; q < 4; ++q) {
    const int w = warp * 4 + q;
    if (!rotated[w]) continue;
    float reg[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) reg[s] = __bfloat162float(staged[w][s * 32 + lane]);
    hq_ifwht256_sign(reg, signs, 0, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) dsts[w][s * 32 + lane] = __float2bfloat16_rn(reg[s]);
  }
}


Status launched(const char *what) { return cudaPeekAtLastError() == cudaSuccess ? nullptr : what; }

// Splits so that a narrow call still has ~512 CTAs; a list of selection_width tokens has at most
// ceil(width / 32) tiles.
int32_t splits_of(int32_t rows, int32_t kv_heads, int32_t width) {
  const int32_t units = rows * kv_heads;
  const int32_t tiles = (width + kTileTokens - 1) / kTileTokens;
  if (units >= 256) return 1;
  const int32_t want = (512 + units - 1) / units;
  return want < tiles ? want : tiles;
}

}  // namespace

Status check_geometry(const Geometry &g) {
  if (g.head_dim != kHeadDim) return "sparse attention: head dim is not 256";
  if (g.kv_heads <= 0 || g.q_heads != g.kv_heads * kGroup) return "sparse attention: GQA group is not 12";
  if (g.selection_width() <= 0) return "sparse attention: no selection width";
  return nullptr;
}

int32_t splits_for(const Geometry &g, int32_t rows) { return splits_of(rows, g.kv_heads, g.selection_width()); }

std::size_t partial_bytes(const Geometry &g, int32_t rows) {
  const int32_t splits = splits_for(g, rows);
  if (splits <= 1) return 0;
  const size_t units = static_cast<size_t>(rows) * g.kv_heads * splits * kGroup;
  return (units * kHeadDim * sizeof(float) + 255) / 256 * 256 + units * sizeof(float2);
}

Status attend(const Geometry &g, const KvSource &kv, const Batch &batch, const __nv_bfloat16 *q,
              const Selection &selection, __nv_bfloat16 *out, void *partials, cudaStream_t stream,
              const __nv_bfloat16 *gate) {
  if (const Status st = check_geometry(g)) return st;
  if (selection.dense) return "sparse attention: the selection is dense";
  if (kv.kv_heads != g.kv_heads) return "sparse attention: KV source head count is not the topology's";
  if (kv.mode == KvSource::Mode::ByPosition && batch.lanes != 1) {
    return "sparse attention: a by-position source serves one lane";
  }
  const int32_t rows = batch.rows();
  if (rows <= 0) return nullptr;
  const int32_t splits = splits_for(g, rows);
  if (splits > 1 && partials == nullptr) return "sparse attention: split call without partials";
  if (splits <= 1 && gate != nullptr) return "sparse attention: the output gate rides a split call's combine";
  const int32_t width = g.selection_width();
  const int32_t tiles_per_split = ((width + kTileTokens - 1) / kTileTokens + splits - 1) / splits;
  AttendArgs a{};
  a.kv = kv;
  a.slots = batch.slots;
  a.tokens = batch.tokens;
  a.q_heads = g.q_heads;
  a.q = q;
  a.list = selection.tokens;
  a.counts = selection.counts;
  a.width = width;
  a.splits = splits;
  a.chunk = tiles_per_split * kTileTokens;
  a.out = out;
  if (splits > 1) {
    const size_t units = static_cast<size_t>(rows) * g.kv_heads * splits * kGroup;
    a.partial_o = static_cast<float *>(partials);
    a.partial_ml = reinterpret_cast<float2 *>(static_cast<char *>(partials) +
                                              (units * kHeadDim * sizeof(float) + 255) / 256 * 256);
  }
  attend_kernel<<<dim3(splits, g.kv_heads, rows), kThreads, 0, stream>>>(a);
  if (const Status st = launched("sparse attention: attend launch failed")) return st;
  if (splits > 1) {
    combine_kernel<<<dim3(rows, g.q_heads), kHeadDim, 0, stream>>>(a.partial_o, a.partial_ml, g.kv_heads, splits,
                                                                   g.q_heads, out, gate);
    return launched("sparse attention: combine launch failed");
  }
  return nullptr;
}


std::size_t listed_hq_bytes(const Geometry &g, int32_t rows) {
  return static_cast<size_t>(rows) * g.selection_width() * g.kv_heads * kHeadDim * sizeof(__nv_bfloat16);
}

std::size_t visible_hq_bytes(const Geometry &g, int32_t max_visible) {
  return static_cast<size_t>(max_visible) * g.kv_heads * kHeadDim * sizeof(__nv_bfloat16);
}

namespace {

Status check_hq(const Geometry &g, const HqSource &hq) {
  if (const Status st = check_geometry(g)) return st;
  if (hq.k_codes == nullptr || hq.k_meta == nullptr || hq.v_codes == nullptr || hq.v_meta == nullptr ||
      hq.block_tables == nullptr) {
    return "sparse attention: hq source without its planes";
  }
  if (hq.kv_heads != g.kv_heads) return "sparse attention: hq source head count is not the topology's";
  if ((hq.fresh_k == nullptr) != (hq.fresh_v == nullptr) || (hq.residual_k == nullptr) != (hq.residual_v == nullptr)) {
    return "sparse attention: hq fresh or residual rows given for one role only";
  }
  return nullptr;
}

HqArgs hq_args(const HqSource &hq, const Batch &batch, __nv_bfloat16 *k, __nv_bfloat16 *v) {
  HqArgs a{};
  a.hq = hq;
  a.slots = batch.slots;
  a.positions = batch.positions;
  a.tokens = batch.tokens;
  a.k = k;
  a.v = v;
  return a;
}

}  // namespace

std::size_t scratch_bytes(const Geometry &g, int32_t kv_format, int32_t rows, int32_t tokens, int32_t max_visible) {
  const auto align = [](std::size_t v) { return (v + 255) / 256 * 256; };
  std::size_t bytes = align(partial_bytes(g, rows));
  if (kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    bytes += 2 * align(tokens == 1 ? listed_hq_bytes(g, rows) : visible_hq_bytes(g, max_visible));
  }
  return bytes;
}

Status decode_listed_hq(const Geometry &g, const HqSource &hq, const Batch &batch, const Selection &selection,
                        __nv_bfloat16 *k, __nv_bfloat16 *v, KvSource *out, cudaStream_t stream) {
  if (const Status st = check_hq(g, hq)) return st;
  if (selection.dense) return "sparse attention: the selection is dense";
  const int32_t rows = batch.rows();
  if (rows <= 0) return nullptr;
  HqArgs a = hq_args(hq, batch, k, v);
  a.list = selection.tokens;
  a.counts = selection.counts;
  a.width = g.selection_width();
  const int32_t units = a.width * g.kv_heads * 2;
  hq_rows_kernel<true><<<dim3((units + kHqUnits - 1) / kHqUnits, rows), kHqThreads, 0, stream>>>(a);
  if (const Status st = launched("sparse attention: hq listed rows launch failed")) return st;
  *out = KvSource{k, v, nullptr, 0, g.kv_heads, KvSource::Mode::ByIndex};
  return nullptr;
}

Status decode_visible_hq(const Geometry &g, const HqSource &hq, const Batch &batch, __nv_bfloat16 *k,
                         __nv_bfloat16 *v, KvSource *out, cudaStream_t stream) {
  if (const Status st = check_hq(g, hq)) return st;
  if (batch.lanes != 1) return "sparse attention: the visible-row decode serves one prefill lane";
  if (batch.max_visible <= 0) return nullptr;
  HqArgs a = hq_args(hq, batch, k, v);
  a.visible = batch.max_visible;
  const int64_t units = static_cast<int64_t>(batch.max_visible) * g.kv_heads * 2;
  hq_rows_kernel<false><<<static_cast<unsigned>((units + kHqUnits - 1) / kHqUnits), kHqThreads, 0, stream>>>(a);
  if (const Status st = launched("sparse attention: hq visible rows launch failed")) return st;
  *out = KvSource{k, v, nullptr, 0, g.kv_heads, KvSource::Mode::ByPosition};
  return nullptr;
}

}  // namespace ignis::flash_next::sparse
