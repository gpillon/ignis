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

#include "qsa_sparse.h"

#include <cmath>

namespace ignis::flash_next::sparse {
namespace {

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
        if (a.kv.by_index) {
          at = ((static_cast<size_t>(row) * a.width + (t0 + t)) * kv_heads + kvh) * kHeadDim;
        } else {
          const int32_t page = a.kv.block_tables[static_cast<size_t>(slot) * a.kv.logical_pages + (pos >> 6)];
          at = ((static_cast<size_t>(page) * kv_heads + kvh) * 64 + (pos & 63)) * kHeadDim;
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

__global__ void combine_kernel(const float *partial_o, const float2 *partial_ml, int32_t kv_heads, int32_t splits,
                               int32_t q_heads, __nv_bfloat16 *out) {
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
  out[(static_cast<size_t>(row) * q_heads + h) * kHeadDim + d] = __float2bfloat16_rn(den > 0.0F ? num / den : 0.0F);
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
              const Selection &selection, __nv_bfloat16 *out, void *partials, cudaStream_t stream) {
  if (const Status st = check_geometry(g)) return st;
  if (selection.dense) return "sparse attention: the selection is dense";
  if (kv.kv_heads != g.kv_heads) return "sparse attention: KV source head count is not the topology's";
  const int32_t rows = batch.rows();
  if (rows <= 0) return nullptr;
  const int32_t splits = splits_for(g, rows);
  if (splits > 1 && partials == nullptr) return "sparse attention: split call without partials";
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
                                                                   g.q_heads, out);
    return launched("sparse attention: combine launch failed");
  }
  return nullptr;
}

}  // namespace ignis::flash_next::sparse
