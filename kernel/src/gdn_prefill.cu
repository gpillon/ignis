// ignis kernel leaf: the GDN prefill's chunked delta rule on our own kernels, in the MMA forms a
// GeForce card runs at full rate (kernel/src/gdn_prefill.h; GitHub #514, spec #503) -- OURS.
//
// The algebra, per value head and 32-token chunk, from the state S0 (value x key) at the chunk's
// start. G is the inclusive cumulative log decay inside the chunk (G_C its last), A the strictly
// lower beta_r (k_r . k_c) exp(G_r - G_c), T = (I + A)^-1, P the lower (incl. diagonal)
// (q_t . k_s) exp(G_t - G_s). The token recurrence S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1}
// k_t) k_t^T unrolls into
//
//   U = T diag(beta) (V - diag(exp G) K S0^T)          (the chunk's new values)
//   O = diag(exp G) Q S0^T + P U                        (the readout, times `scale`)
//   S = exp(G_C) S0 + U^T diag(exp(G_C - G)) K           (the state after the chunk)
//
// The prepare kernel builds T diag(beta) and P on the CUDA cores in fp32 (the solve by forward
// substitution, one column per lane), and the recurrence runs the rest with S in registers.
//
// The full-rate rule (spec #503): every MMA here is m16n8k16 with f16 operands and an f16
// accumulator. A product that needs fp32 accumulation accumulates 32 products in f16 (two MMAs
// chained) and folds that span into fp32 on the CUDA cores. An f16 operand row that is not bounded
// by construction (S, R = V - .., U) is multiplied by a power of two that brings its largest
// magnitude into [2^8, 2^9), and T diag(beta) by one per chunk and head that brings its largest
// into [1/2, 1); q and k are unit vectors and |P| <= 1. So no span of 32 products exceeds 2^14 in
// magnitude (32 x 2^9 x 1), far from f16's 65504, every rounding is f16's 2^-11 of what it
// rounds, and the scales come back off exactly in fp32.
//
// Layouts are the vendored op's (kernel/vendor/.../gated_delta_net.cpp): q, k [T][qk][128], v and
// out [T][value][128], g and beta [T][value], the state [value head][value][key].

#include "gdn_prefill.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ignis::gdn_prefill {
namespace {

constexpr int kD = 128;   // the head (state) dimension
constexpr int kC = 32;    // the chunk
constexpr int kMaxGroup = 8;
constexpr int kPrepThreads = 256;
// Per (chunk, value head), FP32: exp(G_t) [32], exp(G_C - G_t) [32], then the bits of T diag(beta)'s
// scale exponent and exp(G_C), padded to a multiple of 16 bytes.
constexpr int kRecordFloats = 80;
constexpr int kRecGamma = 0, kRecDelta = 32, kRecTexp = 64, kRecGammaC = 65;
constexpr float kLog2e = 1.4426950408889634f;
// A scaled operand row's largest magnitude lies in [2^(kTarget-1), 2^kTarget).
constexpr int kTarget = 9;

std::size_t align256(std::size_t b) { return (b + 255) / 256 * 256; }

struct Layout {
  std::size_t khat = 0, qhat = 0, tmat = 0, pmat = 0, record = 0, total = 0;
  std::int32_t chunks = 0;
};

Layout layout_of(std::int32_t qk_heads, std::int32_t value_heads, std::int32_t tokens) {
  Layout l;
  if (tokens <= 0) return l;
  l.chunks = (tokens + kC - 1) / kC;
  const std::size_t rows = static_cast<std::size_t>(l.chunks) * kC;
  const std::size_t blocks = static_cast<std::size_t>(l.chunks) * value_heads;
  std::size_t off = 0;
  const auto take = [&](std::size_t bytes) {
    const std::size_t at = off;
    off += align256(bytes);
    return at;
  };
  l.khat = take(rows * qk_heads * kD * 2);
  l.qhat = take(rows * qk_heads * kD * 2);
  l.tmat = take(blocks * kC * kC * 2);
  l.pmat = take(blocks * kC * kC * 2);
  l.record = take(blocks * kRecordFloats * 4);
  l.total = off;
  return l;
}

// --- device helpers ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint32_t smem_u32(const void *p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}
// 16 bytes global -> shared, zero-filled when !valid (src is then not read).
__device__ __forceinline__ void cp16(std::uint32_t dst, const void *src, bool valid) {
  const int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(std::uint32_t (&r)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(std::uint32_t (&r)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// D += A B, f16 operands, f16 accumulator: the full-rate form.
__device__ __forceinline__ void mma_h(std::uint32_t (&d)[2], const std::uint32_t (&a)[4], std::uint32_t b0,
                                      std::uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t *>(&h);
}
__device__ __forceinline__ float2 unpack_h2(std::uint32_t x) {
  return __half22float2(*reinterpret_cast<const __half2 *>(&x));
}
__device__ __forceinline__ float2 unpack_bf2(std::uint32_t x) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&x));
}
// 2^e as a float, exact, for e in [-126, 127].
__device__ __forceinline__ float pow2(int e) { return __int_as_float((127 + e) << 23); }
// The exponent e that brings a largest magnitude m into [2^(kTarget-1), 2^kTarget): m 2^e.
// 0 for zero or a non-finite m (which then propagates); clamped so 2^e and 2^-e are normal.
__device__ __forceinline__ int scale_exp(float m) {
  if (!(m > 0.0f) || !(m <= 3.0e38f)) return 0;
  const int ex = ((__float_as_int(m) >> 23) & 0xFF) - 127;
  return min(max(kTarget - 1 - ex, -120), 120);
}
// The largest magnitude over the four lanes that hold one accumulator row.
__device__ __forceinline__ float row_max(float m) {
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  return fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
}

// Shared-memory swizzles: byte offsets of (row, 16-byte column chunk), conflict-free for the
// ldmatrix reads below. 256-byte rows (q, k), 64-byte rows (T, P), and the value stripe's.
__device__ __forceinline__ int swz256(int row, int ch) { return row * 256 + ((ch ^ (row & 7)) << 4); }
__device__ __forceinline__ int swz64(int row, int ch) { return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4); }
template <int NW> __device__ __forceinline__ int swz_v(int row, int ch) {
  if constexpr (NW >= 4) {
    return row * (32 * NW) + ((ch ^ (row & 7)) << 4);
  } else {
    return swz64(row, ch);
  }
}

// --- the prepare: norms, K K^T and Q K^T, decays, the solve -------------------------------------

struct PrepSmem {
  float kf[kC][kD + 4];  // the normalized keys as their f16 values (the recurrence's)
  float qf[kC][kD + 4];
  float kk[kC][kC + 1];
  float qk[kC][kC + 1];
};
static_assert(sizeof(PrepSmem) <= 48 * 1024, "the prepare's static shared memory");
// After K K^T and Q K^T, kf and qf hold each value head's A [32][32] and cumulative G [32].
constexpr int kHeadScratch = kC * kC + kC;
static_assert(4 * kHeadScratch <= kC * (kD + 4), "four heads' scratch per normalized block");

__global__ void __launch_bounds__(kPrepThreads)
    gdn_prefill_prepare_kernel(const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k,
                               const float *__restrict__ g, const float *__restrict__ beta, int T, int qk_heads,
                               int value_heads, __half *__restrict__ khat, __half *__restrict__ qhat,
                               __half *__restrict__ tmat, __half *__restrict__ pmat, float *__restrict__ record) {
  __shared__ PrepSmem sm;
  const int chunk = static_cast<int>(blockIdx.x);
  const int hq = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31, warp = tid >> 5;
  const int t0 = chunk * kC;
  const int nvalid = min(kC, T - t0);

  // 1. q and k normalized as the vendored l2norm does (x / sqrt(sum x^2 + 1e-6)), rounded to f16;
  //    rows past T are zero. Warp w normalizes rows 8w .. 8w + 7 of the 64 (keys, then queries).
#pragma unroll 1
  for (int r = warp * 8; r < warp * 8 + 8; ++r) {
    const bool is_q = r >= kC;
    const int t = r & (kC - 1);
    const std::size_t at = (static_cast<std::size_t>(t0 + t) * qk_heads + hq) * kD + lane * 4;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (t < nvalid) {
      const uint2 raw = *reinterpret_cast<const uint2 *>((is_q ? q : k) + at);
      const float2 lo = unpack_bf2(raw.x), hi = unpack_bf2(raw.y);
      x[0] = lo.x, x[1] = lo.y, x[2] = hi.x, x[3] = hi.y;
    }
    float sum = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float inv = rsqrtf(sum + 1e-6f);
    const std::uint32_t h01 = pack_h2(x[0] * inv, x[1] * inv), h23 = pack_h2(x[2] * inv, x[3] * inv);
    *reinterpret_cast<uint2 *>((is_q ? qhat : khat) + at) = make_uint2(h01, h23);
    const float2 f01 = unpack_h2(h01), f23 = unpack_h2(h23);
    *reinterpret_cast<float4 *>(&(is_q ? sm.qf : sm.kf)[t][lane * 4]) = make_float4(f01.x, f01.y, f23.x, f23.y);
  }
  __syncthreads();

  // 2. K K^T and Q K^T in fp32: thread (r, cb) takes columns cb, cb + 8, cb + 16, cb + 24 of row r.
  {
    const int r = tid >> 3, cb = tid & 7;
    float akk[4] = {}, aqk[4] = {};
#pragma unroll 4
    for (int i = 0; i < kD; i += 4) {
      const float4 kr = *reinterpret_cast<const float4 *>(&sm.kf[r][i]);
      const float4 qr = *reinterpret_cast<const float4 *>(&sm.qf[r][i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float4 kc = *reinterpret_cast<const float4 *>(&sm.kf[cb + 8 * j][i]);
        akk[j] = fmaf(kr.x, kc.x, fmaf(kr.y, kc.y, fmaf(kr.z, kc.z, fmaf(kr.w, kc.w, akk[j]))));
        aqk[j] = fmaf(qr.x, kc.x, fmaf(qr.y, kc.y, fmaf(qr.z, kc.z, fmaf(qr.w, kc.w, aqk[j]))));
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      sm.kk[r][cb + 8 * j] = akk[j];
      sm.qk[r][cb + 8 * j] = aqk[j];
    }
  }
  __syncthreads();

  // 3. Per value head of the group (one warp each): G, A, the solve, T diag(beta), P, the record.
  const int group = value_heads / qk_heads;
  if (warp < group) {
    const int hv = hq * group + warp;
    float *scratch = (warp < 4 ? &sm.kf[0][0] : &sm.qf[0][0]) + (warp & 3) * kHeadScratch;
    float(*a)[kC] = reinterpret_cast<float(*)[kC]>(scratch);
    float *gs = scratch + kC * kC;
    const std::size_t tv = static_cast<std::size_t>(t0 + lane) * value_heads + hv;
    const float gt = lane < nvalid ? g[tv] : 0.0f;
    const float bt = lane < nvalid ? beta[tv] : 0.0f;
    float G = gt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float up = __shfl_up_sync(0xffffffffu, G, o);
      if (lane >= o) G += up;
    }
    const float Gc = __shfl_sync(0xffffffffu, G, kC - 1);
    gs[lane] = G;
    __syncwarp();
    // Column `lane` of A: the select keeps the upper triangle's exp (which may overflow) out.
#pragma unroll 4
    for (int r = 0; r < kC; ++r) {
      const float br = __shfl_sync(0xffffffffu, bt, r);
      a[r][lane] = lane < r ? br * sm.kk[r][lane] * exp2f((gs[r] - G) * kLog2e) : 0.0f;
    }
    __syncwarp();
    // T = (I + A)^-1, column `lane`: T[r][c] = -sum_{j<r} A[r][j] T[j][c] below the diagonal.
    float tc[kC];
#pragma unroll
    for (int r = 0; r < kC; ++r) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < r; ++j) acc = fmaf(a[r][j], tc[j], acc);
      tc[r] = r < lane ? 0.0f : (r == lane ? 1.0f : -acc);
    }
    // T diag(beta): column `lane` times beta_lane, over one power-of-two scale per chunk and head.
    float m = 0.0f;
#pragma unroll
    for (int r = 0; r < kC; ++r) {
      tc[r] *= bt;
      m = fmaxf(m, fabsf(tc[r]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const int texp = (m > 0.0f && m <= 3.0e38f) ? min(max(((__float_as_int(m) >> 23) & 0xFF) - 126, -100), 100) : 0;
    const float tscale = pow2(-texp);
    const std::size_t block = static_cast<std::size_t>(chunk) * value_heads + hv;
    __half *tb = tmat + block * kC * kC;
    __half *pb = pmat + block * kC * kC;
#pragma unroll
    for (int r = 0; r < kC; ++r) tb[r * kC + lane] = __float2half_rn(tc[r] * tscale);
    // P column `lane` (key s = lane): (q_t . k_s) exp(G_t - G_s) for s <= t.
#pragma unroll 4
    for (int t = 0; t < kC; ++t) {
      const float p = lane <= t ? sm.qk[t][lane] * exp2f((gs[t] - G) * kLog2e) : 0.0f;
      pb[t * kC + lane] = __float2half_rn(p);
    }
    float *rec = record + block * kRecordFloats;
    rec[kRecGamma + lane] = exp2f(G * kLog2e);
    rec[kRecDelta + lane] = exp2f((Gc - G) * kLog2e);
    if (lane < kRecordFloats - 64) {
      rec[64 + lane] = lane == 0 ? __int_as_float(texp) : (lane == 1 ? exp2f(Gc * kLog2e) : 0.0f);
    }
  }
}

// --- the recurrence: S in registers, one warp per 16 value rows of a head ----------------------

template <int NW> struct Stage {
  static constexpr int kKhat = 0;      // [32][128] f16, swz256
  static constexpr int kQhat = 8192;   // [32][128] f16, swz256
  static constexpr int kT = 16384;     // [32][32] f16, swz64
  static constexpr int kP = 18432;     // [32][32] f16, swz64
  static constexpr int kV = 20480;     // [32][16 NW] bf16, swz_v
  static constexpr int kVBytes = 32 * 16 * NW * 2;
  static constexpr int kRec = kV + kVBytes;  // the record, kRecordFloats FP32
  static constexpr int kBytes = kRec + kRecordFloats * 4;
  static constexpr int kSmem = 2 * kBytes + NW * 1024;  // two stages, then a 1 KiB output stage per warp
};

template <int NW>
__global__ void __launch_bounds__(32 * NW, 1)
    gdn_prefill_recurrence_kernel(const __half *__restrict__ khat, const __half *__restrict__ qhat,
                                  const __half *__restrict__ tmat, const __half *__restrict__ pmat,
                                  const float *__restrict__ record, const __nv_bfloat16 *__restrict__ v,
                                  const float *state_in, float *state_out, __nv_bfloat16 *__restrict__ out,
                                  int T, int qk_heads, int value_heads, float scale) {
  using St = Stage<NW>;
  constexpr int kThreads = 32 * NW;
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31, warp = tid >> 5;
  const int g8 = lane >> 2, tq = lane & 3;
  constexpr int kBlocksPerHead = kD / (16 * NW);
  const int h = static_cast<int>(blockIdx.x) / kBlocksPerHead;
  const int dv0 = (static_cast<int>(blockIdx.x) % kBlocksPerHead) * 16 * NW;
  const int hq = h / (value_heads / qk_heads);
  const int chunks = (T + kC - 1) / kC;
  const std::uint32_t base = smem_u32(smem);
  __nv_bfloat16 *stage_out = reinterpret_cast<__nv_bfloat16 *>(smem + 2 * St::kBytes + warp * 1024);

  const auto load_stage = [&](int cc, int buf) {
    const std::uint32_t sb = base + buf * St::kBytes;
    for (int i = tid; i < 512; i += kThreads) {
      const int r = i >> 4, ch = i & 15;
      const std::size_t at = (static_cast<std::size_t>(cc * kC + r) * qk_heads + hq) * kD + ch * 8;
      cp16(sb + St::kKhat + swz256(r, ch), khat + at, true);
      cp16(sb + St::kQhat + swz256(r, ch), qhat + at, true);
    }
    const std::size_t block = static_cast<std::size_t>(cc) * value_heads + h;
    for (int i = tid; i < 128; i += kThreads) {
      const int r = i >> 2, ch = i & 3;
      const std::size_t at = (block * kC + r) * kC + ch * 8;
      cp16(sb + St::kT + swz64(r, ch), tmat + at, true);
      cp16(sb + St::kP + swz64(r, ch), pmat + at, true);
    }
    for (int i = tid; i < 32 * 2 * NW; i += kThreads) {
      const int r = i / (2 * NW), ch = i % (2 * NW);
      const int t = cc * kC + r;
      const bool valid = t < T;
      const __nv_bfloat16 *src = valid ? v + (static_cast<std::size_t>(t) * value_heads + h) * kD + dv0 + ch * 8 : v;
      cp16(sb + St::kV + swz_v<NW>(r, ch), src, valid);
    }
    for (int i = tid; i < kRecordFloats / 4; i += kThreads) {
      cp16(sb + St::kRec + i * 16, record + block * kRecordFloats + i * 4, true);
    }
    cp_commit();
  };

  // This warp's slice of S: rows (values) d0 = dv0 + 16 warp + g8 and d0 + 8, the accumulator
  // layout over 16 tiles of 8 keys: s[j][0..1] = S[d0][8j + 2tq ..], s[j][2..3] = S[d0 + 8][..].
  const int d0 = dv0 + 16 * warp + g8;
  float s[16][4];
  {
    const float *row0 = state_in + (static_cast<std::size_t>(h) * kD + d0) * kD;
    const float *row1 = row0 + 8 * kD;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float2 a = *reinterpret_cast<const float2 *>(row0 + 8 * j + 2 * tq);
      const float2 b = *reinterpret_cast<const float2 *>(row1 + 8 * j + 2 * tq);
      s[j][0] = a.x, s[j][1] = a.y, s[j][2] = b.x, s[j][3] = b.y;
    }
  }

  // ldmatrix lane addressing: non-transposed B fragments of a [token][..] tile (rows 16p + ..,
  // chunks 2ks + ..), and transposed ones (rows 16ks + .., chunks 2jj + ..).
  const int nrow = ((lane >> 4) << 3) + (lane & 7), nch = (lane >> 3) & 1;
  const int trow = (((lane >> 3) & 1) << 3) + (lane & 7), tch = lane >> 4;

  load_stage(0, 0);
#pragma unroll 1
  for (int cc = 0; cc < chunks; ++cc) {
    const int buf = cc & 1;
    if (cc + 1 < chunks) {
      load_stage(cc + 1, buf ^ 1);
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    const std::uint32_t sb = base + buf * St::kBytes;
    const float *rec = reinterpret_cast<const float *>(smem + buf * St::kBytes + St::kRec);

    // 1. S's row scales.
    float m0 = 0.0f, m1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      m0 = fmaxf(m0, fmaxf(fabsf(s[j][0]), fabsf(s[j][1])));
      m1 = fmaxf(m1, fmaxf(fabsf(s[j][2]), fabsf(s[j][3])));
    }
    const int es0 = scale_exp(row_max(m0)), es1 = scale_exp(row_max(m1));
    const float sc0 = pow2(es0), sc1 = pow2(es1);

    // 2. S K^T and S Q^T [16 values x 32 tokens]: spans of 32 keys in f16, folded into fp32.
    float sk[4][4] = {}, sq[4][4] = {};
#pragma unroll
    for (int span = 0; span < 4; ++span) {
      std::uint32_t hk[4][2] = {}, hqv[4][2] = {};
#pragma unroll
      for (int kk = 2 * span; kk < 2 * span + 2; ++kk) {
        const std::uint32_t a[4] = {pack_h2(s[2 * kk][0] * sc0, s[2 * kk][1] * sc0),
                                    pack_h2(s[2 * kk][2] * sc1, s[2 * kk][3] * sc1),
                                    pack_h2(s[2 * kk + 1][0] * sc0, s[2 * kk + 1][1] * sc0),
                                    pack_h2(s[2 * kk + 1][2] * sc1, s[2 * kk + 1][3] * sc1)};
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int off = swz256(16 * p + nrow, 2 * kk + nch);
          std::uint32_t b[4];
          ldsm_x4(b, sb + St::kKhat + off);
          mma_h(hk[2 * p], a, b[0], b[1]);
          mma_h(hk[2 * p + 1], a, b[2], b[3]);
          ldsm_x4(b, sb + St::kQhat + off);
          mma_h(hqv[2 * p], a, b[0], b[1]);
          mma_h(hqv[2 * p + 1], a, b[2], b[3]);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 klo = unpack_h2(hk[j][0]), khi = unpack_h2(hk[j][1]);
        const float2 qlo = unpack_h2(hqv[j][0]), qhi = unpack_h2(hqv[j][1]);
        sk[j][0] += klo.x, sk[j][1] += klo.y, sk[j][2] += khi.x, sk[j][3] += khi.y;
        sq[j][0] += qlo.x, sq[j][1] += qlo.y, sq[j][2] += qhi.x, sq[j][3] += qhi.y;
      }
    }
    const float isc0 = pow2(-es0), isc1 = pow2(-es1);

    // 3. R = V - exp(G) K S0^T, transposed [values x tokens]; V's fragments come transposed.
    float r[4][4];
    {
      std::uint32_t vf[2][4];
#pragma unroll
      for (int qq = 0; qq < 2; ++qq) ldsm_x4_t(vf[qq], sb + St::kV + swz_v<NW>(16 * qq + nrow, 2 * warp + nch));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 vlo = unpack_bf2(vf[j >> 1][(j & 1) * 2]), vhi = unpack_bf2(vf[j >> 1][(j & 1) * 2 + 1]);
        const float2 gm = *reinterpret_cast<const float2 *>(rec + kRecGamma + 8 * j + 2 * tq);
        r[j][0] = vlo.x - gm.x * (sk[j][0] * isc0);
        r[j][1] = vlo.y - gm.y * (sk[j][1] * isc0);
        r[j][2] = vhi.x - gm.x * (sk[j][2] * isc1);
        r[j][3] = vhi.y - gm.y * (sk[j][3] * isc1);
      }
    }
    // A fragments over the 32 tokens of a [values x tokens] accumulator, row-scaled.
    const auto fragments = [&](const float (&x)[4][4], float f0, float f1, std::uint32_t (&af)[2][4]) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        af[ks][0] = pack_h2(x[2 * ks][0] * f0, x[2 * ks][1] * f0);
        af[ks][1] = pack_h2(x[2 * ks][2] * f1, x[2 * ks][3] * f1);
        af[ks][2] = pack_h2(x[2 * ks + 1][0] * f0, x[2 * ks + 1][1] * f0);
        af[ks][3] = pack_h2(x[2 * ks + 1][2] * f1, x[2 * ks + 1][3] * f1);
      }
    };
    const auto max_rows = [&](const float (&x)[4][4], int &e0, int &e1) {
      float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        a0 = fmaxf(a0, fmaxf(fabsf(x[j][0]), fabsf(x[j][1])));
        a1 = fmaxf(a1, fmaxf(fabsf(x[j][2]), fabsf(x[j][3])));
      }
      e0 = scale_exp(row_max(a0));
      e1 = scale_exp(row_max(a1));
    };
    // x [values x tokens] times the lower-triangular B = M^T of a [token][token] tile M (T or P):
    // the key blocks past each token tile are zero and skipped; one span of up to 32 keys.
    const auto lower = [&](const std::uint32_t (&af)[2][4], int tile, std::uint32_t (&acc)[4][2]) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int ks = 0; ks <= p; ++ks) {
          std::uint32_t b[4];
          ldsm_x4(b, sb + tile + swz64(16 * p + nrow, 2 * ks + nch));
          mma_h(acc[2 * p], af[ks], b[0], b[1]);
          mma_h(acc[2 * p + 1], af[ks], b[2], b[3]);
        }
      }
    };

    // 4. U = T diag(beta) R: [values x tokens] = R (T diag(beta))^T.
    float u[4][4];
    {
      int er0, er1;
      max_rows(r, er0, er1);
      std::uint32_t rf[2][4];
      fragments(r, pow2(er0), pow2(er1), rf);
      std::uint32_t hu[4][2] = {};
      lower(rf, St::kT, hu);
      const float te = pow2(__float_as_int(rec[kRecTexp]));
      const float f0 = pow2(-er0), f1 = pow2(-er1);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 lo = unpack_h2(hu[j][0]), hi = unpack_h2(hu[j][1]);
        u[j][0] = (lo.x * f0) * te, u[j][1] = (lo.y * f0) * te;
        u[j][2] = (hi.x * f1) * te, u[j][3] = (hi.y * f1) * te;
      }
    }
    int eu0, eu1;
    max_rows(u, eu0, eu1);
    const float iu0 = pow2(-eu0), iu1 = pow2(-eu1);

    // 5. O = exp(G) Q S0^T + P U, times the readout scale, out through the warp's stage.
    {
      std::uint32_t uf[2][4];
      fragments(u, pow2(eu0), pow2(eu1), uf);
      std::uint32_t ho[4][2] = {};
      lower(uf, St::kP, ho);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 lo = unpack_h2(ho[j][0]), hi = unpack_h2(ho[j][1]);
        const float2 gm = *reinterpret_cast<const float2 *>(rec + kRecGamma + 8 * j + 2 * tq);
        const float o[4] = {gm.x * (sq[j][0] * isc0) + lo.x * iu0, gm.y * (sq[j][1] * isc0) + lo.y * iu0,
                            gm.x * (sq[j][2] * isc1) + hi.x * iu1, gm.y * (sq[j][3] * isc1) + hi.y * iu1};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = 8 * j + 2 * tq + (e & 1), d = g8 + 8 * (e >> 1);
          stage_out[t * 16 + d] = __float2bfloat16_rn(scale * o[e]);
        }
      }
      __syncwarp();
      const int nvalid = min(kC, T - cc * kC);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int id = lane + 32 * i, row = id >> 1, half = id & 1;
        if (row < nvalid) {
          const uint4 val = *reinterpret_cast<const uint4 *>(stage_out + row * 16 + 8 * half);
          *reinterpret_cast<uint4 *>(out + (static_cast<std::size_t>(cc * kC + row) * value_heads + h) * kD + dv0 +
                                     16 * warp + 8 * half) = val;
        }
      }
      __syncwarp();
    }

    // 6. S <- exp(G_C) S0 + U^T diag(exp(G_C - G)) K: spans of the chunk's 32 tokens in f16.
    {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 dl = *reinterpret_cast<const float2 *>(rec + kRecDelta + 8 * j + 2 * tq);
        u[j][0] *= dl.x, u[j][1] *= dl.y, u[j][2] *= dl.x, u[j][3] *= dl.y;
      }
      std::uint32_t uf[2][4];
      // exp(G_C - G) <= 1: U's own scales keep these rows in range.
      fragments(u, pow2(eu0), pow2(eu1), uf);
      const float gc = rec[kRecGammaC];
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        std::uint32_t hs[2][2] = {};
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          std::uint32_t b[4];
          ldsm_x4_t(b, sb + St::kKhat + swz256(16 * ks + trow, 2 * jj + tch));
          mma_h(hs[0], uf[ks], b[0], b[1]);
          mma_h(hs[1], uf[ks], b[2], b[3]);
        }
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const float2 lo = unpack_h2(hs[x][0]), hi = unpack_h2(hs[x][1]);
          float *sj = s[2 * jj + x];
          sj[0] = fmaf(gc, sj[0], lo.x * iu0);
          sj[1] = fmaf(gc, sj[1], lo.y * iu0);
          sj[2] = fmaf(gc, sj[2], hi.x * iu1);
          sj[3] = fmaf(gc, sj[3], hi.y * iu1);
        }
      }
    }
    __syncthreads();  // every warp is done with this stage before the next load overwrites it
  }

  {
    float *row0 = state_out + (static_cast<std::size_t>(h) * kD + d0) * kD;
    float *row1 = row0 + 8 * kD;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      *reinterpret_cast<float2 *>(row0 + 8 * j + 2 * tq) = make_float2(s[j][0], s[j][1]);
      *reinterpret_cast<float2 *>(row1 + 8 * j + 2 * tq) = make_float2(s[j][2], s[j][3]);
    }
  }
}

// --- the recurrence's shape per device ----------------------------------------------------------

struct DevicePlan {
  bool ready = false;
  int sms = 0;
  int occupancy[3] = {};  // CTAs per SM for NW = 8, 4, 2 (0: cannot launch)
};
constexpr int kMaxDevices = 16;
constexpr int kWarpChoices[3] = {8, 4, 2};
std::mutex g_plan_mutex;
std::array<DevicePlan, kMaxDevices> g_plans{};

template <int NW> int occupancy_of() {
  cudaError_t err = cudaFuncSetAttribute(gdn_prefill_recurrence_kernel<NW>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         Stage<NW>::kSmem);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  int blocks = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gdn_prefill_recurrence_kernel<NW>, 32 * NW,
                                                      Stage<NW>::kSmem);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return blocks;
}

DevicePlan plan_for_current_device() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) {
    throw std::runtime_error("gdn_prefill: no current CUDA device");
  }
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  DevicePlan &p = g_plans[device];
  if (!p.ready) {
    if (cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || p.sms <= 0) {
      throw std::runtime_error("gdn_prefill: no multiprocessor count");
    }
    // Set for this device's context, once: the attribute and the occupancy each launch relies on.
    p.occupancy[0] = occupancy_of<8>();
    p.occupancy[1] = occupancy_of<4>();
    p.occupancy[2] = occupancy_of<2>();
    p.ready = true;
  }
  return p;
}

// The warps per CTA that finish first: each CTA walks every chunk of its 16 NW value rows, so the
// time is the waves of CTAs times the warps one SM scheduler (four per SM) interleaves; at a tie
// the wider CTA, which reads each chunk's keys and queries once for more rows.
int choose_warps(const DevicePlan &p, int value_heads) {
  int best = 0;
  long best_cost = 0;
  for (int i = 0; i < 3; ++i) {
    const int nw = kWarpChoices[i], occ = p.occupancy[i];
    if (occ <= 0) continue;
    const long ctas = static_cast<long>(value_heads) * (kD / (16 * nw));
    const long resident = static_cast<long>(p.sms) * occ;
    const long waves = (ctas + resident - 1) / resident;
    const long per_sm = std::min<long>(occ, (ctas + p.sms - 1) / p.sms);
    const long per_scheduler = (nw * per_sm + 3) / 4;
    const long cost = waves * per_scheduler;
    if (best == 0 || cost < best_cost) {
      best = nw;
      best_cost = cost;
    }
  }
  if (best == 0) throw std::runtime_error("gdn_prefill: the recurrence fits no SM of this device");
  return best;
}

void require(bool ok, const std::string &what) {
  if (!ok) throw std::invalid_argument("gdn_prefill: " + what);
}

// `name` in the operating system's environment block, empty when unset.
std::string environment(const char *name) {
#ifdef _WIN32
  std::string value(64, '\0');
  DWORD n = GetEnvironmentVariableA(name, value.data(), static_cast<DWORD>(value.size()));
  if (n >= value.size()) {
    value.assign(n, '\0');
    n = GetEnvironmentVariableA(name, value.data(), static_cast<DWORD>(value.size()));
  }
  value.resize(n < value.size() ? n : 0);
  return value;
#else
  const char *v = std::getenv(name);
  return v == nullptr ? std::string() : std::string(v);
#endif
}

} // namespace

bool route_for_load(int compute_capability_major) {
  const std::string named = environment("IGNIS_GDN_PREFILL");
  if (named.empty()) return runs_by_default(compute_capability_major);
  if (named == "ours") return true;
  if (named == "vendored") return false;
  throw std::invalid_argument("IGNIS_GDN_PREFILL is `ours` or `vendored`, not `" + named + "`");
}

std::size_t workspace_bytes(std::int32_t qk_heads, std::int32_t value_heads, std::int32_t tokens) {
  return layout_of(qk_heads, value_heads, tokens).total;
}

std::int32_t recurrence_warps() { return choose_warps(plan_for_current_device(), 48); }

void chunked(const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v, const ninfer::Tensor &g,
             const ninfer::Tensor &beta, float scale, const ninfer::Tensor &state_in, ninfer::Tensor &state_out,
             ninfer::Tensor &out, void *workspace, std::size_t bytes, cudaStream_t stream) {
  const std::int32_t qk_heads = q.ne[1], value_heads = v.ne[1], T = q.ne[2];
  require(T > 0, "T must be positive");
  require(qk_heads > 0 && value_heads % qk_heads == 0 && value_heads / qk_heads <= kMaxGroup,
          "value heads must be 1 to 8 per q/k head");
  const auto shaped = [](const ninfer::Tensor &t, ninfer::DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2) {
    return t.dtype == dtype && t.ne[0] == n0 && t.ne[1] == n1 && t.ne[2] == n2 && t.ne[3] == 1 && t.is_contiguous() &&
           t.data != nullptr;
  };
  require(shaped(q, ninfer::DType::BF16, kD, qk_heads, T) && shaped(k, ninfer::DType::BF16, kD, qk_heads, T),
          "q and k must be BF16 [128, qk_heads, T]");
  require(shaped(v, ninfer::DType::BF16, kD, value_heads, T) && shaped(out, ninfer::DType::BF16, kD, value_heads, T),
          "v and out must be BF16 [128, value_heads, T]");
  require(shaped(g, ninfer::DType::FP32, value_heads, T, 1) && shaped(beta, ninfer::DType::FP32, value_heads, T, 1),
          "g and beta must be FP32 [value_heads, T]");
  require(shaped(state_in, ninfer::DType::FP32, kD, kD, value_heads) &&
              shaped(state_out, ninfer::DType::FP32, kD, kD, value_heads),
          "the states must be FP32 [128, 128, value_heads]");
  require(std::isfinite(scale) && scale > 0.0f, "scale must be positive");
  const Layout l = layout_of(qk_heads, value_heads, T);
  require(workspace != nullptr && bytes >= l.total && reinterpret_cast<std::uintptr_t>(workspace) % 256 == 0,
          "the workspace must hold workspace_bytes(..), 256-byte aligned");

  const DevicePlan plan = plan_for_current_device();
  const int nw = choose_warps(plan, value_heads);
  auto *ws = static_cast<unsigned char *>(workspace);
  auto *khat = reinterpret_cast<__half *>(ws + l.khat);
  auto *qhat = reinterpret_cast<__half *>(ws + l.qhat);
  auto *tmat = reinterpret_cast<__half *>(ws + l.tmat);
  auto *pmat = reinterpret_cast<__half *>(ws + l.pmat);
  auto *rec = reinterpret_cast<float *>(ws + l.record);

  gdn_prefill_prepare_kernel<<<dim3(static_cast<unsigned>(l.chunks), static_cast<unsigned>(qk_heads)), kPrepThreads, 0,
                               stream>>>(static_cast<const __nv_bfloat16 *>(q.data),
                                         static_cast<const __nv_bfloat16 *>(k.data), static_cast<const float *>(g.data),
                                         static_cast<const float *>(beta.data), T, qk_heads, value_heads, khat, qhat,
                                         tmat, pmat, rec);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(std::string("gdn_prefill: prepare launch: ") + cudaGetErrorString(err));

  const auto launch = [&](auto kernel, int warps, int smem_bytes) {
    const unsigned grid = static_cast<unsigned>(value_heads * (kD / (16 * warps)));
    kernel<<<grid, 32 * warps, smem_bytes, stream>>>(khat, qhat, tmat, pmat, rec,
                                                     static_cast<const __nv_bfloat16 *>(v.data),
                                                     static_cast<const float *>(state_in.data),
                                                     static_cast<float *>(state_out.data),
                                                     static_cast<__nv_bfloat16 *>(out.data), T, qk_heads, value_heads,
                                                     scale);
  };
  if (nw == 8) {
    launch(gdn_prefill_recurrence_kernel<8>, 8, Stage<8>::kSmem);
  } else if (nw == 4) {
    launch(gdn_prefill_recurrence_kernel<4>, 4, Stage<4>::kSmem);
  } else {
    launch(gdn_prefill_recurrence_kernel<2>, 2, Stage<2>::kSmem);
  }
  err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(std::string("gdn_prefill: recurrence launch: ") + cudaGetErrorString(err));
}

} // namespace ignis::gdn_prefill
