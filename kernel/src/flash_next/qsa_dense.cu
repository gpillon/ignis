// ignis kernel leaf -- the Flash-Next QSA dense attention (spec flash-next/04, GitHub #302, slice
// S2): OURS (ADR 0043), no port claim. Causal GQA attention for a prefill call of one lane while
// the lane's visible tokens are at most dense_threshold(): query head h of row t (position
// P + t, P = positions[0]) over the keys [0, P + t] of KV head h / 12,
//   out[t][h] = sum_j softmax_j(q[t][h] . k[j] / 16) v[j],
// softmax in fp32, the probabilities rounded to BF16 for the value product (as FlashAttention
// does), the output rounded once to BF16.
//
// FlashAttention-2's shape on BF16 m16n8k16 tensor cores: one CTA per (64 query rows, query
// head), four warps of 16 rows; the keys walked in 32-row tiles, double-buffered through shared
// memory with cp.async (rows past the visible keys zero-filled); the scores' accumulator fragments
// reused as the probability operand; online softmax with exp2 (the 1/16 scale and log2(e) folded
// into one factor). Shared memory rows are 512 bytes, so each 16-byte chunk is XOR-swizzled by the
// row's low three bits: the eight rows one ldmatrix reads hit eight distinct bank groups.
// K and V come from S3's KvSource: the BF16 pages through the lane's block-table row, or a
// decoded plain-frame scratch by position.

#include "qsa.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ignis::flash_next::qsa {

namespace {

constexpr int kRows = 64;          // query rows per CTA
constexpr int kKeys = 32;          // keys per tile
constexpr int kWarps = kRows / 16;
constexpr int kThreads = kWarps * 32;
constexpr int kChunks = kHeadDim / 8;  // 16-byte chunks per row
constexpr int kQBytes = kRows * kHeadDim * 2;
constexpr int kTileBytes = kKeys * kHeadDim * 2;
constexpr int kSmemBytes = kQBytes + 4 * kTileBytes;  // Q, then K and V double-buffered

struct DenseArgs {
  sparse::KvSource source;
  const int32_t *slots;
  int32_t slot_count;
  const int32_t *positions;
  const __nv_bfloat16 *q;
  __nv_bfloat16 *out;
  int32_t tokens;
  int32_t max_visible;
  float scale_log2;
};

// Byte offset of chunk `chunk` of row `row` in a swizzled tile.
__device__ __forceinline__ uint32_t swizzled(int row, int chunk) {
  return static_cast<uint32_t>((row * kChunks + (chunk ^ (row & 7))) * 16);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t *>(&v);
}

// Key row `key` of KV head `head` in `plane` (K or V of the source).
__device__ __forceinline__ const __nv_bfloat16 *key_row(const sparse::KvSource &s, const __nv_bfloat16 *plane,
                                                        int32_t slot, int head, int32_t key) {
  if (s.mode == sparse::KvSource::Mode::Paged) {
    const int32_t page = s.block_tables[static_cast<int64_t>(slot) * s.logical_pages + (key >> 6)];
    return plane + ((static_cast<int64_t>(page) * s.kv_heads + head) * 64 + (key & 63)) * kHeadDim;
  }
  return plane + (static_cast<int64_t>(key) * s.kv_heads + head) * kHeadDim;
}

// One 32-key tile of K and V into a buffer pair; keys at or past `keys` are zero-filled.
__device__ __forceinline__ void load_tile(const DenseArgs &a, int32_t slot, int head, int32_t key0, int32_t keys,
                                          uint32_t k_smem, uint32_t v_smem) {
  for (int i = static_cast<int>(threadIdx.x); i < kKeys * kChunks; i += kThreads) {
    const int row = i / kChunks;
    const int chunk = i % kChunks;
    const int32_t key = key0 + row;
    const bool valid = key < keys;
    const int32_t at = valid ? key : 0;
    cp_async16(k_smem + swizzled(row, chunk), key_row(a.source, a.source.k, slot, head, at) + chunk * 8, valid);
    cp_async16(v_smem + swizzled(row, chunk), key_row(a.source, a.source.v, slot, head, at) + chunk * 8, valid);
  }
}

__global__ void __launch_bounds__(kThreads) dense_kernel(DenseArgs a) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t q_smem = base;
  const uint32_t k_smem[2] = {base + kQBytes, base + kQBytes + kTileBytes};
  const uint32_t v_smem[2] = {base + kQBytes + 2 * kTileBytes, base + kQBytes + 3 * kTileBytes};

  const int warp = static_cast<int>(threadIdx.x >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int head = static_cast<int>(blockIdx.y);
  const int kv_head = head / kGroup;
  const int32_t row0 = static_cast<int32_t>(blockIdx.x) * kRows;
  const int32_t rows_here = min(kRows, a.tokens - row0);
  const int32_t first = a.positions[0];
  const int32_t slot = a.slots[0];
  if (first < 0 || first + a.tokens > a.max_visible) __trap();
  if (a.source.mode == sparse::KvSource::Mode::Paged && (slot < 0 || slot >= a.slot_count)) __trap();
  // The keys this CTA reads: up to its last row's position.
  const int32_t keys = first + row0 + rows_here;
  const int32_t tiles = (keys + kKeys - 1) / kKeys;

  for (int i = static_cast<int>(threadIdx.x); i < kRows * kChunks; i += kThreads) {
    const int row = i / kChunks;
    const int chunk = i % kChunks;
    const bool valid = row < rows_here;
    const __nv_bfloat16 *src =
        a.q + (static_cast<int64_t>(valid ? row0 + row : row0) * kQHeads + head) * kHeadDim + chunk * 8;
    cp_async16(q_smem + swizzled(row, chunk), src, valid);
  }
  load_tile(a, slot, kv_head, 0, keys, k_smem[0], v_smem[0]);
  cp_async_commit();

  // This thread's two rows of the warp's 16, and their positions (rows past the call clamp to
  // its last position: finite, never stored).
  const int r_lo = warp * 16 + (lane >> 2);
  int32_t pos[2];
  for (int h = 0; h < 2; ++h) pos[h] = first + min(row0 + r_lo + 8 * h, a.tokens - 1);

  float o[kHeadDim / 8][4];
#pragma unroll
  for (auto &t : o) t[0] = t[1] = t[2] = t[3] = 0.0F;
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float l[2] = {0.0F, 0.0F};

  for (int32_t tile = 0; tile < tiles; ++tile) {
    const int buf = tile & 1;
    if (tile + 1 < tiles) {
      load_tile(a, slot, kv_head, (tile + 1) * kKeys, keys, k_smem[buf ^ 1], v_smem[buf ^ 1]);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // S = Q K^T for the warp's 16 rows and the tile's 32 keys.
    float s[4][4];
#pragma unroll
    for (auto &t : s) t[0] = t[1] = t[2] = t[3] = 0.0F;
#pragma unroll
    for (int ks = 0; ks < kHeadDim / 16; ++ks) {
      uint32_t qa[4];
      ldmatrix_x4(qa, q_smem + swizzled(warp * 16 + (lane & 15), 2 * ks + (lane >> 4)));
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        uint32_t kb[4];
        ldmatrix_x4(kb, k_smem[buf] + swizzled(np * 16 + (lane & 7) + ((lane >> 4) << 3), 2 * ks + ((lane >> 3) & 1)));
        mma_bf16(s[2 * np], qa, kb[0], kb[1]);
        mma_bf16(s[2 * np + 1], qa, kb[2], kb[3]);
      }
    }

    // Causal mask, scale, online softmax per row (a row's 32 scores sit in one lane quad).
    const int32_t key0 = tile * kKeys;
    float tile_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        const int32_t key = key0 + nt * 8 + 2 * (lane & 3) + (e & 1);
        const float v = key <= pos[h] ? s[nt][e] * a.scale_log2 : -CUDART_INF_F;
        s[nt][e] = v;
        tile_max[h] = fmaxf(tile_max[h], v);
      }
    }
    float corr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffU, tile_max[h], 1));
      tile_max[h] = fmaxf(tile_max[h], __shfl_xor_sync(0xffffffffU, tile_max[h], 2));
      const float m_new = fmaxf(m[h], tile_max[h]);
      corr[h] = exp2f(m[h] - m_new);
      m[h] = m_new;
      l[h] *= corr[h];
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        const float p = exp2f(s[nt][e] - m[h]);
        s[nt][e] = p;
        l[h] += p;
      }
    }
#pragma unroll
    for (auto &t : o) {
      t[0] *= corr[0];
      t[1] *= corr[0];
      t[2] *= corr[1];
      t[3] *= corr[1];
    }

    // O += P V: the score fragments of key blocks (2kk, 2kk + 1) are the A operand of k-step kk.
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]),
                              pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int dp = 0; dp < kHeadDim / 16; ++dp) {
        uint32_t vb[4];
        ldmatrix_x4_trans(vb, v_smem[buf] + swizzled(kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), 2 * dp + (lane >> 4)));
        mma_bf16(o[2 * dp], pa, vb[0], vb[1]);
        mma_bf16(o[2 * dp + 1], pa, vb[2], vb[3]);
      }
    }
    __syncthreads();
  }

  float inv[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float sum = l[h];
    sum += __shfl_xor_sync(0xffffffffU, sum, 1);
    sum += __shfl_xor_sync(0xffffffffU, sum, 2);
    inv[h] = 1.0F / sum;
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int32_t row = row0 + r_lo + 8 * h;
    if (row >= a.tokens) continue;
    __nv_bfloat16 *dst = a.out + (static_cast<int64_t>(row) * kQHeads + head) * kHeadDim + 2 * (lane & 3);
#pragma unroll
    for (int nt = 0; nt < kHeadDim / 8; ++nt) {
      *reinterpret_cast<__nv_bfloat162 *>(dst + nt * 8) =
          __floats2bfloat162_rn(o[nt][2 * h] * inv[h], o[nt][2 * h + 1] * inv[h]);
    }
  }
}

}  // namespace

Status attend_dense(const Geometry &g, const sparse::KvSource &source, int32_t slots, const Batch &batch,
                    const __nv_bfloat16 *q, __nv_bfloat16 *out, cudaStream_t stream) {
  if (batch.lanes != 1 || batch.tokens <= 0 || batch.slots == nullptr || batch.positions == nullptr) {
    return "qsa dense attention: a call is one lane of at least one token";
  }
  if (batch.max_visible > g.dense_threshold()) return "qsa dense attention: more visible tokens than dense_threshold()";
  if (source.kv_heads != kKvHeads || source.k == nullptr || source.v == nullptr ||
      source.mode == sparse::KvSource::Mode::ByIndex ||
      (source.mode == sparse::KvSource::Mode::Paged && (source.block_tables == nullptr || slots <= 0))) {
    return "qsa dense attention: K/V must be the BF16 pages or a by-position scratch of 2 KV heads";
  }
  static_assert(kSmemBytes <= 227 * 1024, "dense attention's shared memory");
  if (cudaFuncSetAttribute(dense_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes) != cudaSuccess) {
    return "qsa dense attention: cannot set the shared memory size";
  }
  DenseArgs a{source, batch.slots, slots, batch.positions, q, out, batch.tokens, batch.max_visible,
              0.0625F * 1.4426950408889634F};
  const dim3 grid(static_cast<unsigned>((batch.tokens + kRows - 1) / kRows), kQHeads);
  dense_kernel<<<grid, kThreads, kSmemBytes, stream>>>(a);
  return cudaPeekAtLastError() == cudaSuccess ? nullptr : "qsa dense attention: launch failed";
}

}  // namespace ignis::flash_next::qsa
