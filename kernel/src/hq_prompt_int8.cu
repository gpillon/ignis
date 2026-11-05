// ignis kernel leaf: the hq-e8-2b prompt attention's integer band (GitHub #510, spec #503) --
// OURS, kernel/src/hq_prompt_int8.h. ignis_hq_prompt_attention (hq_prompt_attention.cu) runs its
// passes under IgnisHqPromptPrecision::kInt8: it stores each pass's K tiles (tile_kernel<true>,
// the 8-bit band's) and its Q, and calls `attend` here for V^T and the attention.

#include "hq_prompt_int8.h"

#include "hq_prompt_common.cuh"
#include "hq_prompt_keys.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace {

// D += A B with A u8 and B s8, int32 accumulate (exact): the integer band's P V.
__device__ __forceinline__ void mma_u8s8(int *d, const std::uint32_t *a, const std::uint32_t *b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

void check_launch(const char *what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

// ---- the integer band (GitHub #510, spec #503) ---------------------------------------------------
//
// IgnisHqPromptPrecision::kInt8: the 8-bit band a card without the FP8 MMA runs, on the integer
// MMAs every sm_80 card has at their full rate (s8 and u8 into s32), apart from fa_kernel's FP8
// axis, so the 86 cubin carries it live. Ported from the prototype prototype/sm8x-dense-prefill
// (phases 2d and 2e, its fa_i8f_kernel at the measured winner). One kernel serves the band and,
// in the prefill pass, the residual window, so every key goes through it:
//   - Q K^T in INT8, Q with one scale per row and K with one per 32-key tile (tile_kernel<true>'s
//     K, the 8-bit band's), exact in int32 from 1.5 2^23 (|S| <= 256 127^2 < 2^22): one FADD lifts
//     it;
//   - P V on the u8 x s8 MMA, P = rn(255 p) against the row's running max, V^T in s8 with one
//     amax / 127 scale per fold group of kF keys and KV head (tile_vt_group_kernel). The softmax
//     takes each row's max over the whole group before P is quantized, so the group's kF / 32 MMAs
//     of a dim tile accumulate in one int32 accumulator, exact from 1.5 2^23 (|sum| <= kF 255 127
//     = 4,145,280 < 2^22 at kF = 128), and O is folded into fp32 once a group;
//   - a lazy rescale: O and l carry a reference max m_ref that moves only when a row's running
//     max passes it by more than kLazyLog2 = 8 (log2 units: 2^8); the group's P V and P sum enter
//     O and l times 2^(m - m_ref), in [1, 256], a per-row factor folded into V's scale, so the
//     fold is one FADD and one FFMA an element. O / l, what the carry stores and the merge reads,
//     does not depend on m_ref (every term of both carries the same 2^-m_ref); 2^8 of headroom
//     cannot overflow fp32 (l <= 2^16 keys x 256 x 2^8 = 2^32); the carried l is brought back to
//     the running max before it is stored, so the state means what fa_kernel's means;
//   - one K buffer and one V^T buffer: the group's first barrier makes K everyone's and frees V^T
//     (every warp is past the last group's P V), V^T then loads under Q K^T and the softmax; the
//     second makes V^T everyone's and frees K, the next K then loads under P V;
//   - kQs: the CTA's Q (two heads of 64 rows, 32 KiB) in shared memory, one ldmatrix a k step
//     shared by the group's tiles, instead of 32 registers a thread (O holds 128).
// Every fp32 rounding is written out (__fmul_rn, __fadd_rn, __fmaf_rn), so a compiler cannot
// contract it differently from the prototype the KLD gate measured. A plane's capacity need not
// be a multiple of kF: a ragged last group loads only the rows and tiles inside the plane, and the
// rest of the buffers, whatever they hold, sit under masked columns (P = 0 there, exactly).
// The cache does not change: the window's rows stay BF16 in the residual planes, which decode
// reads; only this pass quantizes them.
constexpr std::int32_t kMagic = 0x4B400000;  // 1.5 2^23 as float bits
constexpr float kMagicF = 12582912.0f;
constexpr float kLazyLog2 = 8.0f;
template <int kF>
constexpr int kInt8GroupBytes = kF * kHeadDim * 2;         // a group's K rows and V^T, 8 bits each
constexpr int kInt8QBytes = kHeadsPerCta * kQ * kHeadDim;  // kQs: the CTA's Q, 32 KiB
template <int kF, bool kQs>
constexpr int kInt8SmemBytes = kInt8GroupBytes<kF> + (kQs ? kInt8QBytes : 0);  // 96 KiB at 128 with Q

template <int kF, bool kQs>
__global__ void __launch_bounds__(kFaThreads, 1)
    fa_int8_kernel(Queries qs, int start, int width, float scale, Columns cols, PassBuffers buf,
                   int splits, int resume, Carry carry) {
  static_assert(kF == 64 || kF == 128, "a fold group is 2 or 4 tiles");
  constexpr int kSub = kF / kTile;
  constexpr int kKSteps = kHeadDim / 32;
  constexpr int kKBytes = kF * kHeadDim;
  extern __shared__ __align__(128) std::uint8_t fa_smem[];
  const std::uint32_t sbase = smem_u32(fa_smem);
  const std::uint32_t qbase = sbase + static_cast<std::uint32_t>(kInt8GroupBytes<kF>);
  const int lead = start % kQ;
  const int ntiles = (lead + width + kQ - 1) / kQ;
  const int tile = ntiles - 1 - static_cast<int>(blockIdx.x) / splits;  // longest first
  const int split = static_cast<int>(blockIdx.x) % splits;
  if (tile < 0) return;
  const int q0 = tile * kQ - lead;
  const int rend = min(start + q0 + kQ, start + width);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = (tid / 32) % (kQ / 16), lane = tid % 32;
  const int g = static_cast<int>(blockIdx.y) / kHeadPairs;
  const int hl = tid / (kQ / 16 * 32);  // the CTA's head, 0 or 1
  const int h = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + hl;
  // This CTA's groups, then its split of them.
  const int n = (cols.before(rend) + kF - 1) / kF;
  const int gbeg = split * n / splits, gend = (split + 1) * n / splits;
  if (gend <= gbeg && resume != 0) return;
  const int tiles_per_head = buf.capacity / kTile;

  auto issue_k = [&](int gi) {
    const std::uint8_t *ksrc =
        buf.k + (static_cast<std::int64_t>(g) * buf.capacity + gi * kF) * kHeadDim;
    const int rows = min(kF, buf.capacity - gi * kF);
#pragma unroll
    for (int j = 0; j < kF * 16 / kFaThreads; ++j) {
      const int u = tid + j * kFaThreads;
      if ((u >> 4) < rows) cp_async16(sbase + row_swz<true>(u >> 4, u & 15), ksrc + u * 16);
    }
  };
  auto issue_v = [&](int gi) {
    const std::uint32_t st = sbase + static_cast<std::uint32_t>(kKBytes);
    const std::uint8_t *vsrc =
        buf.v + (static_cast<std::int64_t>(g) * tiles_per_head + gi * kSub) * kVtTileBytes;
    const int tiles = min(kSub, tiles_per_head - gi * kSub);
#pragma unroll
    for (int j = 0; j < kSub * 2; ++j) {
      const int u = tid + j * kFaThreads;  // 512 16-byte segments a tile
      if ((u >> 9) < tiles) {
        cp_async16(st + static_cast<std::uint32_t>((u >> 9) * kVtTileBytes) + vt_off((u & 511) >> 1, u & 1),
                   vsrc + u * 16);
      }
    }
  };
  // kQs: the CTA's two heads of 64 rows, laid out as K rows; a row outside the call is zeros.
  auto issue_q = [&]() {
    if constexpr (kQs) {
#pragma unroll
      for (int j = 0; j < kInt8QBytes / 16 / kFaThreads; ++j) {
        const int u = tid + j * kFaThreads;
        const int head = u / (kQ * 16), row = (u / 16) % kQ, unit = u % 16;
        const std::uint32_t dst =
            qbase + static_cast<std::uint32_t>(head * kQ * kHeadDim) + row_swz<true>(row, unit);
        const int t = q0 + row;
        if (t >= 0 && t < width) {
          const int hh = g * kGroup + (static_cast<int>(blockIdx.y) % kHeadPairs) * kHeadsPerCta + head;
          cp_async16(dst, qs.q8 + (static_cast<std::int64_t>(t) * kQHeads + hh) * kHeadDim + unit * 16);
        } else {
          asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u));
        }
      }
    }
  };

  const int r0 = lane / 4, c2 = 2 * (lane % 4);
  const int ta = warp * 16 + r0, tb = ta + 8;
  std::uint32_t qa[kQs ? 1 : kKSteps][4];
  float f[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    const bool in = t >= 0 && t < width;
    const float sq = in ? qs.q_scale[static_cast<std::int64_t>(t) * kQHeads + h] : 1.0f;
    f[r] = in ? sq * scale * kLog2e : 0.0f;
  }
  if constexpr (!kQs) {
#pragma unroll
    for (int kk = 0; kk < kKSteps; ++kk) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = q0 + ((i & 1) ? tb : ta);
        const std::int64_t row = static_cast<std::int64_t>(t) * kQHeads + h;
        qa[kk][i] = t < 0 || t >= width
                        ? 0u
                        : *reinterpret_cast<const std::uint32_t *>(qs.q8 + row * kHeadDim + kk * 32 +
                                                                   16 * (i / 2) + 4 * (lane % 4));
      }
    }
  }
  const int pa = start + q0 + ta, pb = start + q0 + tb;
  const int warp_last = start + q0 + warp * 16 + 15;
  const std::uint32_t qrow = qbase + static_cast<std::uint32_t>(hl * kQ * kHeadDim);

  float o[kHeadDim / 8][4];
  float m[2] = {-CUDART_INF_F, -CUDART_INF_F}, l[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < kHeadDim / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  if (resume != 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int t = q0 + (r ? tb : ta);
      if (t < 0 || t >= width) continue;
      const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
      m[r] = carry.m[row];
      l[r] = carry.l[row];
      const __half *po = carry.o + row * kHeadDim;
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(po + dt * 8 + c2));
        o[dt][2 * r] = v.x * l[r];
        o[dt][2 * r + 1] = v.y * l[r];
      }
    }
  }
  float mref[2] = {m[0], m[1]};  // the reference max O and l are held at
  const int lm = lane / 8, lr = lane % 8;
  constexpr float kU8 = 256.0f / 255.0f;  // P sums in units of 256 p, the carry's

  // S of group gi's tiles, from 1.5 2^23; a tile wholly past the warp's last row is left empty
  // (the mask covers it). A k step's Q fragment serves every tile of the group.
  auto qk = [&](int gi, int (&si)[kSub][4][4]) {
    bool act[kSub];
#pragma unroll
    for (int j = 0; j < kSub; ++j) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) si[j][nt][0] = si[j][nt][1] = si[j][nt][2] = si[j][nt][3] = kMagic;
      act[j] = j == 0 || cols.key((gi * kSub + j) * kTile) <= warp_last;
    }
#pragma unroll
    for (int kk = 0; kk < kKSteps; ++kk) {
      std::uint32_t a[4];
      if constexpr (kQs) {
        ldsm_x4(a, qrow + row_swz<true>(warp * 16 + lr + 8 * (lm & 1), 2 * kk + (lm >> 1)));
      } else {
        a[0] = qa[kk][0];
        a[1] = qa[kk][1];
        a[2] = qa[kk][2];
        a[3] = qa[kk][3];
      }
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
        if (act[j]) {
#pragma unroll
          for (int p = 0; p < 2; ++p) {
            std::uint32_t b[4];
            ldsm_x4(b, sbase + static_cast<std::uint32_t>(j * kTile * kHeadDim) +
                           row_swz<true>(16 * p + lr + 8 * (lm / 2), 2 * kk + lm % 2));
            mma_s8(si[j][2 * p], a, b);
            mma_s8(si[j][2 * p + 1], a, b + 2);
          }
        }
      }
    }
  };

  // Group gi's softmax from its S: m, m_ref and l move on; P's u8 A fragments, one per tile; fb,
  // the fold's per-row factor (O + (P V) fb).
  auto softmax = [&](int gi, const int (&si)[kSub][4][4], std::uint32_t (&pf)[kSub][4], float (&fb)[2]) {
    const float sv = buf.scale[(kKvHeads + g) * tiles_per_head + gi * kSub];
    const float fv = __fmul_rn(sv, kU8);
    unsigned pi[2] = {0u, 0u};
    float sc[kSub][4][4];
    float mx[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
    for (int j = 0; j < kSub; ++j) {
      const int c = gi * kSub + j;
      const float sk = buf.scale[g * tiles_per_head + min(c, tiles_per_head - 1)];
      const float fs[2] = {__fmul_rn(f[0], sk), __fmul_rn(f[1], sk)};
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[j][nt][e] = __fmul_rn(__fsub_rn(__int_as_float(si[j][nt][e]), kMagicF), fs[e / 2]);
      }
      const int key0 = cols.key(c * kTile);
      const bool edge = key0 + kTile - 1 > start + q0 + warp * 16 || (c + 1) * kTile > cols.count;
      if (edge) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int col = nt * 8 + c2 + (e & 1);
            if (c * kTile + col >= cols.count || key0 + col > (e < 2 ? pa : pb)) sc[j][nt][e] = -CUDART_INF_F;
          }
        }
      }
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) mx[e / 2] = fmaxf(mx[e / 2], sc[j][nt][e]);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
      mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
      m[r] = fmaxf(m[r], mx[r]);
    }
    const float bias[2] = {__fsub_rn(8.0f, m[0] == -CUDART_INF_F ? 0.0f : m[0]),
                           __fsub_rn(8.0f, m[1] == -CUDART_INF_F ? 0.0f : m[1])};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
        std::uint32_t w[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const float a = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[j][nt][2 * r], bias[r]), 8.0f)));
          const float b = __fmul_rn(255.0f, ex2_approx(__fsub_rn(__fadd_rn(sc[j][nt][2 * r + 1], bias[r]), 8.0f)));
          // rn(x) for 0 <= x < 2^22 is the low bits of x + 1.5 2^23: bytes [a, b] of the pair.
          w[nt] = __byte_perm(__float_as_uint(__fadd_rn(a, kMagicF)), __float_as_uint(__fadd_rn(b, kMagicF)), 0x0040);
        }
        pf[j][r] = __byte_perm(w[0], w[1], 0x5410);
        pf[j][2 + r] = __byte_perm(w[2], w[3], 0x5410);
        pi[r] = __dp4a(pf[j][2 + r], 0x01010101u, __dp4a(pf[j][r], 0x01010101u, pi[r]));
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 1);
      pi[r] += __shfl_xor_sync(0xffffffffu, pi[r], 2);
    }
    // -inf - -inf is NaN, which passes no threshold: a row with no key yet keeps its reference.
    const bool need = __fsub_rn(m[0], mref[0]) > kLazyLog2 || __fsub_rn(m[1], mref[1]) > kLazyLog2;
    if (__any_sync(0xffffffffu, need)) {
      float cr[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        cr[r] = mref[r] == m[r] || m[r] == -CUDART_INF_F ? 1.0f : ex2_approx(__fsub_rn(mref[r], m[r]));
        l[r] = __fmul_rn(l[r], cr[r]);
        mref[r] = m[r];
      }
#pragma unroll
      for (int dt = 0; dt < kHeadDim / 8; ++dt) {
        o[dt][0] = __fmul_rn(o[dt][0], cr[0]);
        o[dt][1] = __fmul_rn(o[dt][1], cr[0]);
        o[dt][2] = __fmul_rn(o[dt][2], cr[1]);
        o[dt][3] = __fmul_rn(o[dt][3], cr[1]);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float fac = mref[r] == m[r] ? 1.0f : ex2_approx(__fsub_rn(m[r], mref[r]));
      l[r] = __fmaf_rn(__fmul_rn(static_cast<float>(static_cast<int>(pi[r])), kU8), fac, l[r]);
      fb[r] = __fmul_rn(fv, fac);
    }
  };

  // O += (P V) fb for group gi's V^T: one int32 accumulator over the group's tiles.
  auto pv = [&](const std::uint32_t (&pf)[kSub][4], const float (&fb)[2]) {
    const std::uint32_t vs = sbase + static_cast<std::uint32_t>(kKBytes);
#pragma unroll
    for (int dp = 0; dp < kHeadDim / 16; ++dp) {
      int t[2][4];
#pragma unroll
      for (int x = 0; x < 2; ++x) t[x][0] = t[x][1] = t[x][2] = t[x][3] = kMagic;
#pragma unroll
      for (int j = 0; j < kSub; ++j) {
        std::uint32_t b[4];
        ldsm_x4(b, vs + static_cast<std::uint32_t>(j * kVtTileBytes) + vt_off(16 * dp + 8 * (lm / 2) + lr, lm % 2));
        mma_u8s8(t[0], pf[j], b);
        mma_u8s8(t[1], pf[j], b + 2);
      }
#pragma unroll
      for (int x = 0; x < 2; ++x) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          o[2 * dp + x][e] = __fmaf_rn(__fsub_rn(__int_as_float(t[x][e]), kMagicF), fb[e / 2], o[2 * dp + x][e]);
        }
      }
    }
  };

  issue_q();
  if (gend > gbeg) issue_k(gbeg);
  cp_commit();
  for (int gi = gbeg; gi < gend; ++gi) {
    cp_wait<0>();
    __syncthreads();
    issue_v(gi);
    cp_commit();
    const bool work = cols.key(gi * kF) <= warp_last;
    std::uint32_t pf[kSub][4];
    float fb[2] = {0.0f, 0.0f};
    if (work) {
      int si[kSub][4][4];
      qk(gi, si);
      softmax(gi, si, pf, fb);
    }
    cp_wait<0>();
    __syncthreads();
    if (gi + 1 < gend) issue_k(gi + 1);
    cp_commit();
    if (work) pv(pf, fb);
  }

  // The split's state, as fa_kernel leaves it: O over l, and l brought back to the running max.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int t = q0 + (r ? tb : ta);
    if (t < 0 || t >= width) continue;
    const std::int64_t row = (static_cast<std::int64_t>(split) * width + t) * kQHeads + h;
    __half *po = carry.o + row * kHeadDim;
    const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
#pragma unroll
    for (int dt = 0; dt < kHeadDim / 8; ++dt) {
      *reinterpret_cast<__half2 *>(po + dt * 8 + c2) =
          __floats2half2_rn(o[dt][2 * r] * inv, o[dt][2 * r + 1] * inv);
    }
    if (c2 == 0) {
      carry.m[row] = m[r];
      carry.l[row] = mref[r] != m[r] ? __fmul_rn(l[r], ex2_approx(__fsub_rn(mref[r], m[r]))) : l[r];
    }
  }
}

// V^T of an integer pass in s8, with one amax / 127 scale per kF-key group and KV head, stored at
// each 32-key tile of the group (the attention reads it at the group's first), in store_fp8_tile's
// segment order; K is tile_kernel<true>'s, a scale per tile. A column past the pass's keys is a
// zero row; a ragged last group writes only the tiles inside the plane.
template <int kF>
__global__ void __launch_bounds__(256) tile_vt_group_kernel(IgnisHqPromptKeys src, Columns cols, PassBuffers buf) {
  constexpr int kSub = kF / kTile;
  extern __shared__ __align__(16) std::uint8_t vg_smem[];
  auto *rows = reinterpret_cast<__nv_bfloat16 *>(vg_smem);  // kF rows of 256
  __shared__ std::int8_t signs[kHeadDim];
  __shared__ float warp_amax[8];
  __shared__ float group_scale;
  const int group = static_cast<int>(blockIdx.x);
  const int head = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid >> 5, lane = tid & 31;
  const int tiles = buf.capacity / kTile;
  const int own = min(kSub, tiles - group * kSub);  // the group's tiles inside the plane
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    const int key_l = i * kTile + warp * 4 + (lane >> 3);
    const int column = group * kF + key_l;
    ignis_hq_prompt::fetch_rows_warp(src, head, true, column < cols.count ? cols.key(column) : -1,
                                     rows + key_l * kHeadDim, signs);
  }
  __syncthreads();
  float mx = 0.0f;
  for (int u = tid; u < kF * kHeadDim / 8; u += 256) {
    const uint4 w = reinterpret_cast<const uint4 *>(rows)[u];
    const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&words[j]));
      mx = fmaxf(mx, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
  }
  mx = warp_max(mx);
  if (lane == 0) warp_amax[warp] = mx;
  __syncthreads();
  if (tid == 0) {
    float amax = 0.0f;
    for (int w = 0; w < 8; ++w) amax = fmaxf(amax, warp_amax[w]);
    const float s = amax > 0.0f ? amax / kInt8Max : 1.0f;
    group_scale = s;
    for (int i = 0; i < own; ++i) buf.scale[(kKvHeads + head) * tiles + group * kSub + i] = s;
  }
  __syncthreads();
  const float inv = 1.0f / group_scale;
#pragma unroll
  for (int i = 0; i < kSub; ++i) {
    if (i >= own) break;
    const __nv_bfloat16 *tile = rows + i * kTile * kHeadDim;
    std::uint8_t *dst = buf.v + (static_cast<std::int64_t>(head) * tiles + group * kSub + i) * kVtTileBytes;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int seg = tid + k * 256, d = seg >> 1, hh = seg & 1;
      std::uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        std::uint32_t word = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int key_l = 16 * hh + 2 * j + (b & 1) + 8 * (b >> 1);
          word |= static_cast<std::uint32_t>(i8(__bfloat162float(tile[key_l * kHeadDim + d]) * inv)) << (8 * b);
        }
        w[j] = word;
      }
      *reinterpret_cast<uint4 *>(dst + seg * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
  }
}

// ---- host ---------------------------------------------------------------------------------------

// The integer band's kernel variant: the keys of a fold group and whether Q sits in shared
// memory. The prototype measured the winners per card, and they differ between the RTX 5090 and
// an sm_86 card; THIS is the one place a GPU generation picks its variant. Every generation starts
// at the prototype's winner (fold 128, Q in shared memory: 96 KiB, inside an sm_86 CTA's 99), and
// a variant is instantiated only when a generation picks it.
struct Int8Variant {
  int fold_keys;
  bool q_in_shared;
};
Int8Variant int8_variant(int compute_major) {
  switch (compute_major) {
    case 8:   // Ampere (8.6) and Ada (8.9)
    case 12:  // Blackwell (12.0)
    default:
      return {128, true};
  }
}

int compute_major() {
  static const int major = [] {
    int device = 0, value = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
      return 0;
    }
    return value;
  }();
  return major;
}

template <int kF, bool kQs>
void launch_fa_int8(dim3 grid, Queries qs, int start, int width, float scale, Columns cols,
                    PassBuffers buf, int splits, int resume, Carry carry, cudaStream_t stream) {
  constexpr int kSmem = kInt8SmemBytes<kF, kQs>;
  static const cudaError_t attribute =
      cudaFuncSetAttribute(fa_int8_kernel<kF, kQs>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  fa_int8_kernel<kF, kQs><<<grid, kFaThreads, kSmem, stream>>>(qs, start, width, scale, cols, buf,
                                                                splits, resume, carry);
  check_launch("integer band attention");
}

template <int kF>
void launch_vt_groups(IgnisHqPromptKeys src, Columns cols, PassBuffers buf, cudaStream_t stream) {
  constexpr int kSmem = kF * kHeadDim * 2;  // the group's BF16 rows
  static const cudaError_t attribute =
      cudaFuncSetAttribute(tile_vt_group_kernel<kF>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  if (attribute != cudaSuccess) {
    throw std::runtime_error(std::string("ignis_hq_prompt_attention: shared memory attribute: ") +
                             cudaGetErrorString(attribute));
  }
  tile_vt_group_kernel<kF><<<dim3(static_cast<unsigned>((cols.count + kF - 1) / kF), kKvHeads, 1), 256, kSmem,
                             stream>>>(src, cols, buf);
  check_launch("V^T groups");
}

} // namespace

namespace ignis_hq_prompt_int8 {

// The counts the prototype measured best for the band on the RTX 5090. P is quantized against
// each split's own running max, so the count moves the band's output; a count chosen from the
// card's SM count would make it vary from card to card.
std::int32_t splits(std::uint32_t visible_keys) { return visible_keys <= 16384 ? 2 : 3; }

void attend(const IgnisHqPromptKeys &src, const Pass &pass, cudaStream_t stream) {
  Queries qs{};
  qs.q8 = pass.q8;
  qs.q_scale = pass.q_scale;
  const Columns cols{pass.first0, pass.split, pass.first1, pass.count};
  const PassBuffers buf{pass.k, pass.v, pass.scales, pass.capacity};
  const Carry carry{pass.carry_o, pass.carry_m, pass.carry_l};
  // Up to one more query tile than width / 64: the first one may start mid-block.
  const dim3 grid(static_cast<unsigned>(((pass.width + kQ - 1) / kQ + 1) * pass.splits),
                  static_cast<unsigned>(kKvHeads * kHeadPairs));
  const Int8Variant variant = int8_variant(compute_major());
  if (variant.fold_keys == 128 && variant.q_in_shared) {
    launch_vt_groups<128>(src, cols, buf, stream);
    launch_fa_int8<128, true>(grid, qs, pass.start, pass.width, pass.scale, cols, buf, pass.splits, pass.resume,
                              carry, stream);
    return;
  }
  throw std::logic_error("ignis_hq_prompt_attention: no integer band kernel is built for this variant");
}

}  // namespace ignis_hq_prompt_int8
