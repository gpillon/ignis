// PROTOTYPE (sm8x dense prefill, phases 2a and 2c; GitHub #474) -- OURS, throwaway, never to merge.
//
// The dense prefill route an sm_8x card would run, as a standalone bench: every trellis matrix is
// decoded once per prefill chunk into a dense INT8 copy, the rotated activations become INT8
// with one scale per token, and our own integer GEMM multiplies them, mma.sync m16n8k32 s8.s8 ->
// s32 over the full K, with an fp32 epilogue on the CUDA cores. Written from the PTX ISA and our
// own kernels (trellis_linear_*, moe_common.cuh, the 8-bit band's s8 path), no library GEMM.
//
// The new kernels, the ones the SASS check holds to integer MMA (names start sm8x_dense_):
//   sm8x_dense_decode  trellis tile -> the W8 code q = clamp(rn((s - 510) / 4), +-127) of phase 1's
//                      V2 (trellis_decode.cuh's window_codes and w8_code, the same bits), stored
//                      as the dense W: blocks of 128 outputs x 64 inputs, 8 KiB each, [n][k]
//                      inside, block (nb, kb) at (nb * in / 64 + kb) * 8 KiB.
//   sm8x_dense_a8      x o suh, the 128-wide Hadamard, then per token amax / 127 over the whole K,
//                      ties to even (rotate_token_kernel's arithmetic): INT8 in 64-input slabs,
//                      [in / 64][tokens][64], so a GEMM stage's activations are one contiguous run;
//                      the scale and the sum of the token's INT8 values (the offset's rank-1 term).
//   sm8x_dense_gemm    BM tokens x 128 outputs a CTA, 64 inputs a stage, a 4-stage cp.async
//                      pipeline, ldmatrix from XOR-swizzled shared memory, 8 warps of 64 x 64 (BM
//                      256) or 64 x 32 (BM 128), int32 accumulators over the whole K. Epilogue:
//                      y = acc sa[t] sw + off sa[t] sum_q[t], with sw = 4 c1 and off = 1534 c1 +
//                      c0 the trellis affine map's scale and zero offset (c1 = fp16 0x1EEE, c0 =
//                      fp16 0xC931: sw = 887 / 2^15, off = -119 / 2^16, both exact in fp32), then
//                      the output Hadamard over the 128 columns and svh, BF16 out: the layout and
//                      epilogues (store, add, SwiGLU's up pass) of today's fused prefill kernel.
// Phase 2c's three levers, each behind a switch (the defaults are phase 2a's route):
//   A  --a8 shared            sm8x_dense_a8_shared: one launch per shared input (GDN qkv and z, GQA
//                             q, k and v, MLP gate and up) reads x once and writes each matrix's
//                             INT8, scale and sum with that matrix's own suh, bit for bit the
//                             per-matrix A8 (the artifact's siblings do not share suh).
//   B  --sched persistent     sm8x_dense_gemm_persistent: one CTA per SM (the SM count read at run
//                             time) loops over the tiles; the last partial wave (or it and one
//                             full wave) is split along K into one segment per CTA, the int32
//                             partials summed by the tile's last arrival, which runs the epilogue
//                             once: exact in any order, and no CTA waits for another.
//   C  --decode slice         each matrix in output slices of about half the L2
//                             (cudaDevAttrL2CacheSize, or --slice-mb), every slice decoded into one
//                             slice buffer and multiplied at once, while it is still in the L2, by
//                             the GEMM of B or of phase 2a (--l2-persist: a persisting access-policy
//                             window on the slice buffer where the card has one).
//   --ablation runs phase 2a's route, each lever alone and the combinations: the check under every
//   one of them and the per-chunk table of each.
//
// What the bench reports (plain text):
//   device     compute capability, SMs, shared memory, the L2 and its persisting maximum, the
//              runtime and driver versions, lever C's slice sizes
//   peaks      register-loop rates: s8 m16n8k32, f16 m16n8k16 with f32 and with f16 accumulate;
//              the L2-resident read bandwidth (a working set of at most half the L2)
//   check      with --artifact, real matrices of the 16 GB artifact and a synthetic activation
//              stand-in: the W8 codes against a host decode, the A8 values against phase 1's V2
//              (IGNIS_PROTO_FAKEQUANT=w8a8tok) bit for bit, the GEMM's int32 accumulators against
//              an exact integer reference (dp4a, no tensor core), the outputs of both routes
//              against an fp64 reference of the quantized product, and dense against V2
//   diag       the GEMM without its global loads and ldmatrix; the decode into the whole W against
//              one slice buffer; lever C taken apart; lever A's fold, estimated
//   speed      per body shape and T: decode, A8, GEMM (TOPS, % of the s8 peak), their total, and
//              today's fused trellis prefill kernel (ignis_trellis_linear) at the same T
//   chunk      the body's per-chunk totals at the public quant's K mix, dense against fused
//   lever B    with --ablation, per shape: the grid launch against the persistent one
//   overlap    one GDN + MLP layer and one GQA + MLP layer, the next matrix's (or slice's) decode on
//              a second stream through a double buffer, against the same sequence on one stream
//   project    the body's prefill-GEMM time per 8K and per 64K tokens, dense against fused
//
//   ignis_sm8x_dense_bench [--artifact PATH] [--tokens 2048,4096] [--check-tokens N] [--reps N]
//                          [--no-check] [--no-speed] [--no-peaks] [--diag] [--no-fused]
//                          [--a8 per-matrix|shared] [--sched grid|persistent] [--decode whole|slice]
//                          [--slice-mb MB] [--l2-persist] [--ablation]
//
// If the fused trellis linear does not prepare on a card (or with --no-fused), the bench says so and
// runs the dense route alone: no fused column, no V2 comparisons.

#include "moe_common.cuh"
#include "trellis_linear_internal.cuh"
#include "trellis_test_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define BENCH_CUDA(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s (line %d)\n", #expr, cudaGetErrorString(err_), __LINE__); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace sm8x {

using ignis_moe::cp_async_commit;
using ignis_moe::cp_async_wait;
using ignis_moe::ldmatrix_x4;
using ignis_moe::warp_hadamard128;
using trellis_test::DeviceBytes;

// ---- the affine map ----------------------------------------------------------------------------

// value(q) = sw q + off: the trellis codebook's (1024 + s') c1 + c0 at s' = 4 q + 510, exact.
inline double map_c1() { return static_cast<double>(moe_test::f16_to_f32(0x1EEE)); }
inline double map_c0() { return static_cast<double>(moe_test::f16_to_f32(0xC931)); }
inline double map_sw() { return 4.0 * map_c1(); }
inline double map_off() { return 1534.0 * map_c1() + map_c0(); }

enum Epilogue : int { kStore = 0, kAdd = 1, kSwiGluUp = 3 };

// ---- sm8x_dense_decode ---------------------------------------------------------------------------

constexpr int kDecWarps = 4;
constexpr int kBlockN = 128;  // outputs per dense-W block
constexpr int kBlockK = 64;   // inputs per dense-W block
constexpr int kBlockBytes = kBlockN * kBlockK;

__device__ __forceinline__ uint32_t load_stream(const uint32_t *p) { return __ldcs(p); }

// Input k of token t in the INT8 activations: 64-input slabs, [in / 64][tokens][64].
__host__ __device__ __forceinline__ size_t a_index(int tokens, int t, int k) {
  return (static_cast<size_t>(k >> 6) * tokens + t) * 64 + (k & 63);
}

// One warp per strip of 16 outputs x 64 inputs (four trellis tiles down K): lane t decodes its
// eight steps of each tile -- inputs 16 i + {r0, r0 + 1, r0 + 8, r0 + 9}, outputs {c0, c0 + 8}, r0 =
// 2 (t % 4), c0 = t / 4 (the format's tensor-core order) -- into the strip in shared memory, then
// the warp stores the strip's 1 KiB. Strips run outputs-fastest, so a CTA's warps read
// neighbouring trellis tiles and write neighbouring KiB.
// WARPS 4 is the standalone launch; WARPS 2 (2 KiB of shared memory) fits beside a GEMM CTA in an
// SM's 100 KiB, for the overlap.
template <int K2, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) sm8x_dense_decode(const uint32_t *trellis, int stride, int in,
                                                                 int out, int8_t *w) {
  __shared__ __align__(16) uint8_t strip[WARPS][16 * kBlockK];
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int kbs = in / kBlockK;
  const long long s = static_cast<long long>(blockIdx.x) * WARPS + warp;
  if (s >= static_cast<long long>(out / 16) * kbs) return;
  const int sub = static_cast<int>(s % 8);
  const int kb = static_cast<int>((s / 8) % kbs);
  const int nb = static_cast<int>(s / (8LL * kbs));
  uint8_t *sp = strip[warp];
  const int r0 = (lane & 3) * 2;
  const int c0 = lane >> 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t *tile = trellis + (static_cast<size_t>(kb * 4 + i) * stride + nb * 8 + sub) * words;
    uint32_t hi, lo;
    if constexpr (K2 >= 10) {
      const ignis_trellis::LanePlan3 plan = ignis_trellis::lane_plan3(K2, lane);
      ignis_trellis::fragment_window3(load_stream(tile + plan.w0), load_stream(tile + plan.w1),
                                      load_stream(tile + plan.w2), plan, hi, lo);
    } else {
      const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
      ignis_trellis::fragment_window(load_stream(tile + plan.w0), load_stream(tile + plan.w1), plan, hi, lo);
    }
    uint32_t bits[8];
    ignis_trellis::window_codes<K2>(hi, lo, bits);
    int q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = ignis_trellis::w8_code(bits[j]);
    auto pair = [](int a, int b) { return static_cast<uint16_t>((a & 0xFF) | ((b & 0xFF) << 8)); };
    const int k = 16 * i + r0;
    *reinterpret_cast<uint16_t *>(sp + c0 * kBlockK + k) = pair(q[0], q[1]);
    *reinterpret_cast<uint16_t *>(sp + c0 * kBlockK + k + 8) = pair(q[2], q[3]);
    *reinterpret_cast<uint16_t *>(sp + (c0 + 8) * kBlockK + k) = pair(q[4], q[5]);
    *reinterpret_cast<uint16_t *>(sp + (c0 + 8) * kBlockK + k + 8) = pair(q[6], q[7]);
  }
  __syncwarp();
  uint4 *dst = reinterpret_cast<uint4 *>(w + static_cast<size_t>(nb * kbs + kb) * kBlockBytes + sub * 16 * kBlockK);
  const uint4 *src = reinterpret_cast<const uint4 *>(sp);
  dst[lane] = src[lane];
  dst[lane + 32] = src[lane + 32];
}

// ---- sm8x_dense_a8 -------------------------------------------------------------------------------

// One CTA per token, rotate_token_kernel's arithmetic (trellis_linear.cu): x o suh, the Hadamard in
// fp32, amax over the whole K, q = clamp(rn(v 127 / amax), +-127). The row is rotated twice (the
// second pass reads x from the L2).
__global__ void __launch_bounds__(256) sm8x_dense_a8(const __nv_bfloat16 *x, const __half *suh, int tokens, int in,
                                                      int8_t *a, float *sa, int32_t *sq) {
  __shared__ float s_max[8];
  __shared__ int s_sum[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = in / 128;
  const int t = blockIdx.x;
  if (t >= tokens) return;
  auto rotated = [&](int blk, float (&v)[4]) {
    const int k0 = blk * 128 + 4 * lane;
    float xv[4], s[4];
    ignis_tl::load_bf16x4(x + static_cast<size_t>(t) * in + k0, xv);
    ignis_tl::load_half4(suh + k0, s);
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = xv[q] * s[q];
    warp_hadamard128(v);
  };
  float m = 0.0f;
  for (int blk = warp; blk < nb; blk += 8) {
    float v[4];
    rotated(blk, v);
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
  }
  m = ignis_moe::warp_max(m);
  if (lane == 0) s_max[warp] = m;
  __syncthreads();
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, s_max[i]);
  const bool finite = amax > 0.0f && amax <= 3.0e38f;
  const float to_q = finite ? 127.0f / amax : 0.0f;
  const float inv = finite ? amax * (1.0f / 127.0f) : 0.0f;
  int sum = 0;
  for (int blk = warp; blk < nb; blk += 8) {
    float v[4];
    rotated(blk, v);
    int q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = max(-127, min(127, __float2int_rn(v[i] * to_q)));
      sum += q[i];
    }
    const uint32_t packed = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (static_cast<uint32_t>(q[3] & 0xFF) << 24);
    *reinterpret_cast<uint32_t *>(a + a_index(tokens, t, blk * 128 + 4 * lane)) = packed;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  if (lane == 0) s_sum[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) total += s_sum[i];
    sa[t] = inv;
    sq[t] = total;
  }
}

// --diag only, a meter of lever A's fold estimate (not the route): sm8x_dense_a8's second pass
// alone, each token's amax given, which is what down's A8 would cost if the gate+up epilogue
// handed it every token's amax (its first pass, the amax over the row, gone).
__global__ void __launch_bounds__(256) sm8x_dense_a8_onepass(const __nv_bfloat16 *x, const __half *suh, int tokens, int in,
                                                              const float *amax_in, int8_t *a, float *sa, int32_t *sq) {
  __shared__ int s_sum[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = in / 128;
  const int t = blockIdx.x;
  if (t >= tokens) return;
  const float amax = amax_in[t];
  const bool finite = amax > 0.0f && amax <= 3.0e38f;
  const float to_q = finite ? 127.0f / amax : 0.0f;
  const float inv = finite ? amax * (1.0f / 127.0f) : 0.0f;
  int sum = 0;
  for (int blk = warp; blk < nb; blk += 8) {
    const int k0 = blk * 128 + 4 * lane;
    float v[4], xv[4], s[4];
    ignis_tl::load_bf16x4(x + static_cast<size_t>(t) * in + k0, xv);
    ignis_tl::load_half4(suh + k0, s);
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = xv[q] * s[q];
    warp_hadamard128(v);
    int q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = max(-127, min(127, __float2int_rn(v[i] * to_q)));
      sum += q[i];
    }
    const uint32_t packed = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (static_cast<uint32_t>(q[3] & 0xFF) << 24);
    *reinterpret_cast<uint32_t *>(a + a_index(tokens, t, k0)) = packed;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  if (lane == 0) s_sum[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) total += s_sum[i];
    sa[t] = inv;
    sq[t] = total;
  }
}

// ---- sm8x_dense_a8_shared (phase 2c, lever A) ----------------------------------------------------

// The matrices that read one input (GDN qkv and z, GQA q, k and v, MLP gate and up) do not share
// their input rotation: the artifact gives each its own suh (the signs agree, the magnitudes do
// not), so one INT8 copy cannot serve them all. This launch reads x once instead: each warp holds
// its blocks of the token's row in registers, then runs sm8x_dense_a8's arithmetic once per matrix
// with that matrix's suh, in the same order (x o suh in fp32, warp_hadamard128, amax over the whole
// K, q = clamp(rn(v 127 / amax), +-127)), and writes each matrix's INT8, scale and sum of q.
constexpr int kA8MaxShared = 3;

struct A8Target {
  const __half *suh;
  int8_t *a;
  float *sa;
  int32_t *sq;
};

struct A8Shared {
  A8Target m[kA8MaxShared];
  int count;
};

// MAXB: the 128-input blocks a warp holds, in <= 8 MAXB 128.
template <int MAXB>
__global__ void __launch_bounds__(256) sm8x_dense_a8_shared(const __nv_bfloat16 *x, int tokens, int in,
                                                             const __grid_constant__ A8Shared g) {
  __shared__ float s_max[kA8MaxShared][8];
  __shared__ int s_sum[kA8MaxShared][8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = in / 128;
  const int t = blockIdx.x;
  if (t >= tokens) return;
  float xv[MAXB][4];
#pragma unroll
  for (int j = 0; j < MAXB; ++j) {
    const int blk = warp + 8 * j;
    if (blk < nb) ignis_tl::load_bf16x4(x + static_cast<size_t>(t) * in + blk * 128 + 4 * lane, xv[j]);
  }
#pragma unroll
  for (int mi = 0; mi < kA8MaxShared; ++mi) {
    if (mi >= g.count) break;
    const A8Target &m = g.m[mi];
    float v[MAXB][4];
    float mx = 0.0f;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int blk = warp + 8 * j;
      if (blk < nb) {
        float s[4];
        ignis_tl::load_half4(m.suh + blk * 128 + 4 * lane, s);
#pragma unroll
        for (int q = 0; q < 4; ++q) v[j][q] = xv[j][q] * s[q];
        warp_hadamard128(v[j]);
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[j][0]), fabsf(v[j][1])), fmaxf(fabsf(v[j][2]), fabsf(v[j][3]))));
      }
    }
    mx = ignis_moe::warp_max(mx);
    if (lane == 0) s_max[mi][warp] = mx;
    __syncthreads();
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, s_max[mi][i]);
    const bool finite = amax > 0.0f && amax <= 3.0e38f;
    const float to_q = finite ? 127.0f / amax : 0.0f;
    const float inv = finite ? amax * (1.0f / 127.0f) : 0.0f;
    int sum = 0;
#pragma unroll
    for (int j = 0; j < MAXB; ++j) {
      const int blk = warp + 8 * j;
      if (blk < nb) {
        int q[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          q[i] = max(-127, min(127, __float2int_rn(v[j][i] * to_q)));
          sum += q[i];
        }
        const uint32_t packed = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) | ((q[2] & 0xFF) << 16) | (static_cast<uint32_t>(q[3] & 0xFF) << 24);
        *reinterpret_cast<uint32_t *>(m.a + a_index(tokens, t, blk * 128 + 4 * lane)) = packed;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    if (lane == 0) s_sum[mi][warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      int total = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) total += s_sum[mi][i];
      m.sa[t] = inv;
      m.sq[t] = total;
    }
  }
}

// ---- sm8x_dense_gemm -----------------------------------------------------------------------------

constexpr int kGemmThreads = 256;
constexpr int kStages = 4;
constexpr int kBK = 64;   // inputs (bytes) per stage
constexpr int kBN = 128;  // outputs per CTA: one Hadamard block
constexpr int kERow = 132;  // fp32 per epilogue staging row

struct GemmParams {
  const int8_t *a;    // [in / 64][tokens][64] (a_index)
  const int8_t *w;    // dense W, blocked (sm8x_dense_decode)
  const float *sa;    // [tokens]
  const int32_t *sq;  // [tokens]
  const __half *svh;  // [out]
  float sw;
  float off;
  int tokens;
  int in;
  int out;
  int mt;        // token tiles
  int nt;        // output tiles
  int raster;    // 0: token tiles fastest, 1: output tiles fastest
  int epilogue;  // kStore, kAdd, kSwiGluUp
  __nv_bfloat16 *y;
  int ldy;
  int32_t *raw;  // non-null: the int32 accumulators [tokens][ldr] instead of y (the check)
  int ldr;
};

template <int BM> struct Cfg {
  static constexpr int kWarpsM = BM == 256 ? 4 : 2;
  static constexpr int kWarpsN = 8 / kWarpsM;
  static constexpr int kWM = BM / kWarpsM;  // 64
  static constexpr int kWN = kBN / kWarpsN;  // 64 or 32
  static constexpr int kMI = kWM / 16;
  static constexpr int kNI = kWN / 8;
  static constexpr int kAStage = BM * kBK;
  static constexpr int kBStage = kBN * kBK;
  static constexpr int kStage = kAStage + kBStage;
  static constexpr int kSmem = kStage * kStages;
  static_assert(kWarpsM * 16 * kERow * 4 <= kSmem, "the epilogue staging overlays the stages");
};

// 16-byte chunk c (0..3) of row r of a 64-byte-row stage: XOR-swizzled with (r / 2) % 4, so the
// eight rows an ldmatrix phase reads at one chunk land on eight different 4-bank groups.
__device__ __forceinline__ uint32_t swz(int r, int c) { return static_cast<uint32_t>(r * 64 + ((c ^ ((r >> 1) & 3)) << 4)); }

__device__ __forceinline__ void cp_async16(uint32_t smem, const void *gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}

__device__ __forceinline__ void cp_async16_zfill(uint32_t smem, const void *gmem, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t smem) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem));
}

// The integer MMA: D = A B + D, A 16 x 32 s8 (row), B 32 x 8 s8 (col), int32, exact.
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BM>
__device__ __forceinline__ void load_stage(const GemmParams &p, uint32_t st, int kt, int row0, const int8_t *wcol) {
  using C = Cfg<BM>;
  const int tid = threadIdx.x;
#pragma unroll
  for (int i = 0; i < BM * 4 / kGemmThreads; ++i) {
    const int idx = tid + i * kGemmThreads;
    const int r = idx >> 2;
    const int c = idx & 3;
    const int t = row0 + r;
    const bool valid = t < p.tokens;
    cp_async16_zfill(st + swz(r, c), p.a + a_index(p.tokens, valid ? t : 0, kt * kBK + 16 * c), valid);
  }
  const int8_t *blk = wcol + static_cast<size_t>(kt) * kBlockBytes;
#pragma unroll
  for (int i = 0; i < kBN * 4 / kGemmThreads; ++i) {
    const int idx = tid + i * kGemmThreads;
    const int r = idx >> 2;
    const int c = idx & 3;
    cp_async16(st + C::kAStage + swz(r, c), blk + r * 64 + 16 * c);
  }
}

// The fragments of k32 step kk (0, 1) of a stage: A for the warp's kMI m16 blocks, B for its kNI
// n8 blocks two at a time (b[j][0..1] block 2j, b[j][2..3] block 2j + 1).
template <int BM>
__device__ __forceinline__ void load_frags(uint32_t st, int kk, int wm, int wn, uint32_t (&a)[Cfg<BM>::kMI][4],
                                           uint32_t (&b)[Cfg<BM>::kNI / 2][4]) {
  using C = Cfg<BM>;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < C::kMI; ++i) ldsm_x4(a[i], st + swz(wm * C::kWM + 16 * i + (lane & 15), 2 * kk + (lane >> 4)));
#pragma unroll
  for (int j = 0; j < C::kNI / 2; ++j) {
    ldsm_x4(b[j], st + C::kAStage + swz(wn * C::kWN + 16 * j + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1)));
  }
}

template <int BM>
__device__ __forceinline__ void mma_frags(const uint32_t (&a)[Cfg<BM>::kMI][4], const uint32_t (&b)[Cfg<BM>::kNI / 2][4],
                                          int32_t (&acc)[Cfg<BM>::kMI][Cfg<BM>::kNI][4]) {
  using C = Cfg<BM>;
#pragma unroll
  for (int i = 0; i < C::kMI; ++i)
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) mma_s8(acc[i][n], a[i], b[n >> 1][2 * (n & 1)], b[n >> 1][2 * (n & 1) + 1]);
}

// The main loop over k-steps [k0, k1) (64 inputs each) of one tile, its W block column at wcol.
// The pipeline: at the top of step i, stage i is waited for and visible; stage i + 3 goes into the
// buffer step i - 1 read, which the barrier has just freed. Within a stage the second k32 step's
// fragments load while the first one's MMAs run.
// kDiag (a diagnostic of the bench, not the route): 1 skips the main loop's global loads (the
// stages keep the prologue's data), 2 also its ldmatrix (one set of fragments for every step).
template <int BM, int kDiag = 0>
__device__ __forceinline__ void gemm_mainloop(const GemmParams &p, uint32_t base, const int8_t *wcol, int row0, int k0, int k1,
                                              int wm, int wn, int32_t (&acc)[Cfg<BM>::kMI][Cfg<BM>::kNI][4]) {
  using C = Cfg<BM>;
  const int n = k1 - k0;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < n) load_stage<BM>(p, base + s * C::kStage, k0 + s, row0, wcol);
    cp_async_commit();
  }
  uint32_t a0[C::kMI][4], b0[C::kNI / 2][4], a1[C::kMI][4], b1[C::kNI / 2][4];
  if constexpr (kDiag == 2) {
    cp_async_wait<0>();
    __syncthreads();
    load_frags<BM>(base, 0, wm, wn, a0, b0);
    load_frags<BM>(base, 1, wm, wn, a1, b1);
  }
  for (int kt = 0; kt < n; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const int pf = kt + kStages - 1;
    if (kDiag == 0 && pf < n) load_stage<BM>(p, base + (pf % kStages) * C::kStage, k0 + pf, row0, wcol);
    cp_async_commit();
    const uint32_t st = base + (kt % kStages) * C::kStage;
    if constexpr (kDiag < 2) {
      load_frags<BM>(st, 0, wm, wn, a0, b0);
      load_frags<BM>(st, 1, wm, wn, a1, b1);
    }
    mma_frags<BM>(a0, b0, acc);
    mma_frags<BM>(a1, b1, acc);
  }
  cp_async_wait<0>();
}

// The epilogue of one tile from its full-K accumulators: with p.raw the int32 accumulators
// [tokens][ldr], else, one m16 block of every warp row at a time through shared memory, the
// scales and the offset's rank-1 term per token in fp32, then each 128-column row back through
// the Hadamard, svh, and BF16 out as the fused kernel's epilogue.
template <int BM>
__device__ __forceinline__ void gemm_epilogue(const GemmParams &p, uint8_t *smem, int row0, int tn, int wm, int wn,
                                              const int32_t (&acc)[Cfg<BM>::kMI][Cfg<BM>::kNI][4]) {
  using C = Cfg<BM>;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c4 = lane & 3;
  if (p.raw != nullptr) {
#pragma unroll
    for (int i = 0; i < C::kMI; ++i) {
      const int t = row0 + wm * C::kWM + 16 * i + g;
#pragma unroll
      for (int n = 0; n < C::kNI; ++n) {
        const int col = tn * kBN + wn * C::kWN + 8 * n + 2 * c4;
        if (t < p.tokens) *reinterpret_cast<int2 *>(p.raw + static_cast<size_t>(t) * p.ldr + col) = make_int2(acc[i][n][0], acc[i][n][1]);
        if (t + 8 < p.tokens) *reinterpret_cast<int2 *>(p.raw + static_cast<size_t>(t + 8) * p.ldr + col) = make_int2(acc[i][n][2], acc[i][n][3]);
      }
    }
    return;
  }
  float *e = reinterpret_cast<float *>(smem);
#pragma unroll
  for (int i = 0; i < C::kMI; ++i) {
    __syncthreads();
    const int t0 = row0 + wm * C::kWM + 16 * i + g;
    float f1[2] = {0.0f, 0.0f}, f0[2] = {0.0f, 0.0f};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int t = t0 + 8 * h;
      if (t < p.tokens) {
        const float s = p.sa[t];
        f1[h] = s * p.sw;
        f0[h] = s * p.off * static_cast<float>(p.sq[t]);
      }
    }
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) {
      const int col = wn * C::kWN + 8 * n + 2 * c4;
      const int er = wm * 16 + g;
      *reinterpret_cast<float2 *>(e + er * kERow + col) =
          make_float2(fmaf(static_cast<float>(acc[i][n][0]), f1[0], f0[0]), fmaf(static_cast<float>(acc[i][n][1]), f1[0], f0[0]));
      *reinterpret_cast<float2 *>(e + (er + 8) * kERow + col) =
          make_float2(fmaf(static_cast<float>(acc[i][n][2]), f1[1], f0[1]), fmaf(static_cast<float>(acc[i][n][3]), f1[1], f0[1]));
    }
    __syncthreads();
    for (int rr = warp; rr < C::kWarpsM * 16; rr += 8) {
      const int t = row0 + (rr >> 4) * C::kWM + 16 * i + (rr & 15);
      if (t >= p.tokens) continue;
      const float4 q = *reinterpret_cast<const float4 *>(e + rr * kERow + 4 * lane);
      float v[4] = {q.x, q.y, q.z, q.w};
      warp_hadamard128(v);
      float s[4];
      ignis_tl::load_half4(p.svh + tn * kBN + 4 * lane, s);
      __nv_bfloat16 *y = p.y + static_cast<size_t>(t) * p.ldy + tn * kBN + 4 * lane;
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] *= s[j];
      if (p.epilogue == kSwiGluUp) {
        float gate[4];
        ignis_tl::load_bf16x4(y, gate);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = ignis_moe::silu(gate[j]) * v[j];
      }
      ignis_tl::store_bf16x4(y, v, p.epilogue == kAdd);
    }
  }
}

template <int BM>
__device__ __forceinline__ void zero_acc(int32_t (&acc)[Cfg<BM>::kMI][Cfg<BM>::kNI][4]) {
#pragma unroll
  for (int i = 0; i < Cfg<BM>::kMI; ++i)
#pragma unroll
    for (int n = 0; n < Cfg<BM>::kNI; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0;
}

__device__ __forceinline__ void tile_coords(const GemmParams &p, int tile, int &tm, int &tn) {
  tm = p.raster == 0 ? tile % p.mt : tile / p.nt;
  tn = p.raster == 0 ? tile / p.mt : tile % p.nt;
}

__device__ __forceinline__ const int8_t *w_column(const GemmParams &p, int tn) {
  return p.w + static_cast<size_t>(tn) * (p.in / kBK) * kBlockBytes;
}

// Phase 2a's launch: one CTA per tile.
template <int BM, int kDiag = 0>
__global__ void __launch_bounds__(kGemmThreads, 1) sm8x_dense_gemm(const __grid_constant__ GemmParams p) {
  using C = Cfg<BM>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  int tm, tn;
  tile_coords(p, blockIdx.x, tm, tn);
  const int row0 = tm * BM;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / C::kWarpsN;
  const int wn = warp % C::kWarpsN;
  int32_t acc[C::kMI][C::kNI][4];
  zero_acc<BM>(acc);
  gemm_mainloop<BM, kDiag>(p, base, w_column(p, tn), row0, 0, p.in / kBK, wm, wn, acc);
  gemm_epilogue<BM>(p, smem, row0, tn, wm, wn, acc);
}

// ---- sm8x_dense_gemm_persistent (phase 2c, lever B) ------------------------------------------------

// One CTA per SM loops over the tiles. The tiles in raster order split in two:
//   - the data-parallel tiles [0, dp_tiles), dp_tiles a multiple of the grid: CTA c takes tiles c,
//     c + grid, ..., each over its full K, epilogue at once;
//   - the stream-K tiles, the last sk_tiles: their k-steps in one line (tile-major), cut into segs
//     segments as evenly as possible, segment j for CTA j. A segment's piece of a tile that does
//     not cover its whole K writes its int32 partial sums to a slot and counts in at the tile's
//     arrival counter; the last to arrive adds every other piece's partials to its own (integer
//     sums: exact, in any order) and runs the epilogue once, after the full K, then resets the
//     counter for the next launch. No CTA waits for another.
// A segment touches at most two tiles partly (its first and its last), so each has two slots.
// The persistent kernel's register caps: uncapped (255) it takes 254 registers, the whole 64K file,
// so no decode CTA fits beside it; capped at 224 (no spills on 86 or 120a, cuobjdump -res-usage) it
// leaves room for the 2-warp decode CTAs of the co-resident overlap, at a small cost in speed. The
// route runs the first, the co-resident overlap the second.
constexpr int kRegsRoute = 255;
constexpr int kRegsCoresident = 224;

struct PSched {
  int dp_tiles;
  int sk_tiles;
  int segs;
  int sk_iters;  // sk_tiles * k-steps
  int *cnt;      // [sk_tiles] arrivals, zero between launches
  int4 *part;    // [2 segs][frags * 256] partial accumulators, in fragment order
};

__device__ __forceinline__ int seg_begin(const PSched &s, int j) {
  return static_cast<int>(static_cast<long long>(j) * s.sk_iters / s.segs);
}

// The segment that holds k-step i of the stream-K line.
__device__ __forceinline__ int seg_of(const PSched &s, int i) {
  int j = static_cast<int>(static_cast<long long>(i) * s.segs / s.sk_iters);
  while (j + 1 < s.segs && seg_begin(s, j + 1) <= i) ++j;
  while (j > 0 && seg_begin(s, j) > i) --j;
  return j;
}

template <int BM, int REGS>
__global__ void __maxnreg__(REGS) sm8x_dense_gemm_persistent(const __grid_constant__ GemmParams p, const __grid_constant__ PSched s) {
  using C = Cfg<BM>;
  constexpr int kFrags = C::kMI * C::kNI;
  // No static shared memory, so that a 2-warp decode CTA still fits beside this one in an SM's
  // 100 KiB (the co-resident overlap): every thread walks the same static schedule in registers,
  // and the one broadcast (the last arrival) goes through the stage buffers' last word, free
  // between two main loops.
  extern __shared__ __align__(128) uint8_t smem[];
  int *const s_last = reinterpret_cast<int *>(smem + C::kSmem) - 1;
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int wm = warp / C::kWarpsN;
  const int wn = warp % C::kWarpsN;
  const int kts = p.in / kBK;
  const int j = blockIdx.x;
  const int seg_a = j < s.segs ? seg_begin(s, j) : 0, seg_b = j < s.segs ? seg_begin(s, j + 1) : 0;
  int32_t acc[C::kMI][C::kNI][4];
  // next: a data-parallel tile below dp_tiles, then dp_tiles + a stream-K k-step, -1 when done.
  int next = j < s.dp_tiles ? j : (j < s.segs ? s.dp_tiles + seg_a : -1);
  for (;;) {
    int tile, k0, k1, slot = -1;
    if (next >= 0 && next < s.dp_tiles) {
      tile = next;
      k0 = 0;
      k1 = kts;
      next += gridDim.x;
      if (next >= s.dp_tiles) next = j < s.segs ? s.dp_tiles + seg_a : -1;
    } else if (next >= 0 && next - s.dp_tiles < seg_b) {
      const int it = next - s.dp_tiles;
      const int lt = it / kts;
      tile = s.dp_tiles + lt;
      k0 = it - lt * kts;
      k1 = min(kts, k0 + (seg_b - it));
      if (k0 != 0 || k1 != kts) slot = 2 * j + (it == seg_a ? 0 : 1);
      next += k1 - k0;
    } else {
      break;
    }
    __syncthreads();  // the previous unit is done with the shared memory
    {
      int tm, tn;
      tile_coords(p, tile, tm, tn);
      zero_acc<BM>(acc);
      gemm_mainloop<BM>(p, base, w_column(p, tn), tm * BM, k0, k1, wm, wn, acc);
    }
    if (slot >= 0) {
      const int lt = tile - s.dp_tiles;
      const int first = seg_of(s, lt * kts), last = seg_of(s, lt * kts + kts - 1);
      int4 *mine = s.part + static_cast<size_t>(slot) * kFrags * kGemmThreads;
#pragma unroll
      for (int i = 0; i < C::kMI; ++i)
#pragma unroll
        for (int n = 0; n < C::kNI; ++n) __stcg(mine + (i * C::kNI + n) * kGemmThreads + tid, make_int4(acc[i][n][0], acc[i][n][1], acc[i][n][2], acc[i][n][3]));
      __threadfence();
      __syncthreads();
      if (tid == 0) *s_last = atomicAdd(s.cnt + lt, 1) == last - first;
      __syncthreads();
      if (!*s_last) continue;
      __threadfence();
      for (int jj = first; jj <= last; ++jj) {
        if (jj == j) continue;
        const int other_slot = 2 * jj + (seg_begin(s, jj) / kts == lt ? 0 : 1);
        const int4 *other = s.part + static_cast<size_t>(other_slot) * kFrags * kGemmThreads;
#pragma unroll
        for (int i = 0; i < C::kMI; ++i)
#pragma unroll
          for (int n = 0; n < C::kNI; ++n) {
            const int4 v = __ldcg(other + (i * C::kNI + n) * kGemmThreads + tid);
            acc[i][n][0] += v.x;
            acc[i][n][1] += v.y;
            acc[i][n][2] += v.z;
            acc[i][n][3] += v.w;
          }
      }
      if (tid == 0) s.cnt[lt] = 0;
    }
    int tm, tn;
    tile_coords(p, tile, tm, tn);
    gemm_epilogue<BM>(p, smem, tm * BM, tn, wm, wn, acc);
  }
}

// ---- host side of the route ----------------------------------------------------------------------

struct Dense {
  int in = 0, out = 0;
  int8_t *w = nullptr;  // the dense W, in * out bytes
};

template <int WARPS> void launch_decode_w(const ignis_trellis_weight &tw, int8_t *w, cudaStream_t s) {
  const int grid = static_cast<int>((static_cast<long long>(tw.out / 16) * (tw.in / kBlockK) + WARPS - 1) / WARPS);
  const uint32_t *t = static_cast<const uint32_t *>(tw.trellis);
  const int stride = static_cast<int>(tw.tile_stride == 0 ? tw.out / 16 : tw.tile_stride);
  const int in = static_cast<int>(tw.in), out = static_cast<int>(tw.out);
  switch (tw.k2) {
  case 4: sm8x_dense_decode<4, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  case 5: sm8x_dense_decode<5, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  case 6: sm8x_dense_decode<6, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  case 8: sm8x_dense_decode<8, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  case 10: sm8x_dense_decode<10, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  default: sm8x_dense_decode<12, WARPS><<<grid, WARPS * 32, 0, s>>>(t, stride, in, out, w); break;
  }
  BENCH_CUDA(cudaGetLastError());
}

void launch_decode(const ignis_trellis_weight &tw, int8_t *w, cudaStream_t s, int warps = kDecWarps) {
  if (warps == 2) {
    launch_decode_w<2>(tw, w, s);
  } else {
    launch_decode_w<kDecWarps>(tw, w, s);
  }
}

void launch_a8(const void *x, const void *suh, int tokens, int in, int8_t *a, float *sa, int32_t *sq, cudaStream_t s) {
  sm8x_dense_a8<<<tokens, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), static_cast<const __half *>(suh), tokens, in, a, sa, sq);
  BENCH_CUDA(cudaGetLastError());
}

constexpr int kA8SharedBlocks = 5;  // a warp's 128-input blocks: in <= 5120, every shared input of the body

void launch_a8_shared(const void *x, int tokens, int in, const A8Shared &g, cudaStream_t s) {
  if (in % 128 != 0 || in > 8 * 128 * kA8SharedBlocks || g.count < 1 || g.count > kA8MaxShared) {
    std::fprintf(stderr, "FATAL: the shared A8 takes in <= %d and 1 to %d matrices\n", 8 * 128 * kA8SharedBlocks, kA8MaxShared);
    std::exit(EXIT_FAILURE);
  }
  sm8x_dense_a8_shared<kA8SharedBlocks><<<tokens, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(x), tokens, in, g);
  BENCH_CUDA(cudaGetLastError());
}

void prepare_gemm() {
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm_persistent<256, kRegsRoute>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm_persistent<128, kRegsRoute>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm_persistent<256, kRegsCoresident>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm_persistent<128, kRegsCoresident>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128>::kSmem));
}

struct GemmCall {
  const int8_t *a;
  const int8_t *w;
  const float *sa;
  const int32_t *sq;
  const void *svh;
  int tokens, in, out;
  int epilogue;
  void *y;
  int ldy;
  int32_t *raw = nullptr;
  int bm = 256;
  int raster = 0;
  int diag = 0;
  int ldr = 0;  // the raw output's row stride (0: out)
};

GemmParams gemm_params(const GemmCall &c) {
  GemmParams p{};
  p.a = c.a;
  p.w = c.w;
  p.sa = c.sa;
  p.sq = c.sq;
  p.svh = static_cast<const __half *>(c.svh);
  p.sw = static_cast<float>(map_sw());
  p.off = static_cast<float>(map_off());
  p.tokens = c.tokens;
  p.in = c.in;
  p.out = c.out;
  p.mt = (c.tokens + c.bm - 1) / c.bm;
  p.nt = c.out / kBN;
  p.raster = c.raster;
  p.epilogue = c.epilogue;
  p.y = static_cast<__nv_bfloat16 *>(c.y);
  p.ldy = c.ldy;
  p.raw = c.raw;
  p.ldr = c.ldr == 0 ? c.out : c.ldr;
  return p;
}

void launch_gemm(const GemmCall &c, cudaStream_t s) {
  const GemmParams p = gemm_params(c);
  const int grid = p.mt * p.nt;
  if (c.diag != 0) {
    if (c.diag == 1) sm8x_dense_gemm<256, 1><<<grid, kGemmThreads, Cfg<256>::kSmem, s>>>(p);
    else sm8x_dense_gemm<256, 2><<<grid, kGemmThreads, Cfg<256>::kSmem, s>>>(p);
  } else if (c.bm == 256) {
    sm8x_dense_gemm<256><<<grid, kGemmThreads, Cfg<256>::kSmem, s>>>(p);
  } else {
    sm8x_dense_gemm<128><<<grid, kGemmThreads, Cfg<128>::kSmem, s>>>(p);
  }
  BENCH_CUDA(cudaGetLastError());
}

// ---- lever B's host side: the persistent launch and its stream-K split ----------------------------

int g_grid_sms = 0;  // the persistent grid: the SM count, read at run time

// The stream-K workspace: the arrival counters (zeroed once; each launch leaves them zero) and two
// partial slots per segment, for the largest grid and tile.
struct PWork {
  int *cnt = nullptr;
  int4 *part = nullptr;
  int max_tiles = 0;
};
PWork g_pwork;
constexpr int kMinSegSteps = 4;  // a stream-K segment's least k-steps

void ensure_pwork() {
  if (g_pwork.cnt != nullptr) return;
  g_pwork.max_tiles = 4 * g_grid_sms + 64;
  BENCH_CUDA(cudaMalloc(&g_pwork.cnt, sizeof(int) * g_pwork.max_tiles));
  BENCH_CUDA(cudaMemset(g_pwork.cnt, 0, sizeof(int) * g_pwork.max_tiles));
  BENCH_CUDA(cudaMalloc(&g_pwork.part, static_cast<size_t>(2) * g_grid_sms * Cfg<256>::kMI * Cfg<256>::kNI * kGemmThreads * sizeof(int4)));
}

// sk 0: stream-K over the last partial wave alone; sk 1: over it and the full wave before it. A
// whole number of waves is all data-parallel; fewer tiles than SMs are all stream-K.
PSched make_psched(int tiles, int kts, int sk) {
  ensure_pwork();
  const int g = g_grid_sms;
  PSched s{};
  const int waves = tiles / g, rem = tiles % g;
  if (rem == 0) {
    s.sk_tiles = 0;
  } else if (waves == 0) {
    s.sk_tiles = tiles;
  } else {
    s.sk_tiles = rem + (sk == 1 ? g : 0);
  }
  s.dp_tiles = tiles - s.sk_tiles;
  s.sk_iters = s.sk_tiles * kts;
  s.segs = s.sk_tiles == 0 ? 0 : std::max(1, std::min(g, s.sk_iters / kMinSegSteps));
  if (s.sk_tiles > g_pwork.max_tiles) {
    std::fprintf(stderr, "FATAL: %d stream-K tiles past the workspace's %d\n", s.sk_tiles, g_pwork.max_tiles);
    std::exit(EXIT_FAILURE);
  }
  s.cnt = g_pwork.cnt;
  s.part = g_pwork.part;
  return s;
}

// cap: the co-resident overlap's register-capped instantiation.
void launch_gemm_persistent(const GemmCall &c, int sk, cudaStream_t st, bool cap = false) {
  const GemmParams p = gemm_params(c);
  const PSched s = make_psched(p.mt * p.nt, p.in / kBK, sk);
  const int grid = s.dp_tiles > 0 ? g_grid_sms : s.segs;
  if (c.bm == 256) {
    if (cap) sm8x_dense_gemm_persistent<256, kRegsCoresident><<<grid, kGemmThreads, Cfg<256>::kSmem, st>>>(p, s);
    else sm8x_dense_gemm_persistent<256, kRegsRoute><<<grid, kGemmThreads, Cfg<256>::kSmem, st>>>(p, s);
  } else {
    if (cap) sm8x_dense_gemm_persistent<128, kRegsCoresident><<<grid, kGemmThreads, Cfg<128>::kSmem, st>>>(p, s);
    else sm8x_dense_gemm_persistent<128, kRegsRoute><<<grid, kGemmThreads, Cfg<128>::kSmem, st>>>(p, s);
  }
  BENCH_CUDA(cudaGetLastError());
}

// ---- lever C's host side: output slices sized from the L2 -----------------------------------------

size_t g_slice_bytes = 0;  // the slice's target size: half the L2 unless --slice-mb

// The 128-output blocks per slice of a matrix with `in` inputs: as many as fit the target, at least one.
int slice_blocks(int in) { return std::max(1, static_cast<int>(g_slice_bytes / (static_cast<size_t>(kBlockN) * in))); }

// The blocks per slice of a matrix of nb output blocks: its slices made even (80 blocks at 76 a
// slice are two slices of 40, not 76 and 4).
int slice_even(int in, int nb) {
  const int n = (nb + slice_blocks(in) - 1) / slice_blocks(in);
  return (nb + n - 1) / n;
}

// Output blocks [nb0, nb0 + nbs) of a trellis matrix as a matrix of its own: the same tiles (the
// trellis's 16 x 16 tiles are output-block-major within a 16-input row of tiles), a narrower out.
ignis_trellis_weight trellis_slice(const ignis_trellis_weight &tw, int nb0, int nbs) {
  ignis_trellis_weight s = tw;
  const uint32_t words = static_cast<uint32_t>(ignis_trellis::tile_words(static_cast<int>(tw.k2)));
  s.trellis = static_cast<const uint32_t *>(tw.trellis) + static_cast<size_t>(nb0) * 8 * words;
  s.tile_stride = tw.tile_stride == 0 ? tw.out / 16 : tw.tile_stride;
  s.svh = static_cast<const __half *>(tw.svh) + static_cast<size_t>(nb0) * kBlockN;
  s.out = static_cast<uint32_t>(nbs) * kBlockN;
  return s;
}

}  // namespace sm8x

// ---- the meters: an exact integer reference and the register-loop peaks (not the route) ---------

namespace meter {

using trellis_test::DeviceBytes;

// acc[t][n] = sum_k a[t][k] w[n][k], dp4a on the CUDA cores: 64 x 64 outputs a CTA, 4 x 4 a thread.
__global__ void __launch_bounds__(256) ref_int_gemm(const int8_t *a, const int8_t *w, int tokens, int in, int out, int32_t *acc) {
  __shared__ int32_t as[64][17];
  __shared__ int32_t ws[64][17];
  const int tx = threadIdx.x & 15;
  const int ty = threadIdx.x >> 4;
  const int n0 = blockIdx.x * 64;
  const int t0 = blockIdx.y * 64;
  int32_t c[4][4] = {};
  for (int k0 = 0; k0 < in; k0 += 64) {
    for (int i = threadIdx.x; i < 64 * 16; i += 256) {
      const int r = i >> 4;
      const int wd = i & 15;
      const int t = t0 + r;
      as[r][wd] = t < tokens ? *reinterpret_cast<const int32_t *>(a + sm8x::a_index(tokens, t, k0 + 4 * wd)) : 0;
      const int n = n0 + r;
      ws[r][wd] = *reinterpret_cast<const int32_t *>(w + (static_cast<size_t>(n >> 7) * (in >> 6) + (k0 >> 6)) * 8192 + (n & 127) * 64 + 4 * wd);
    }
    __syncthreads();
#pragma unroll
    for (int wd = 0; wd < 16; ++wd) {
      int32_t av[4], bv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        av[i] = as[ty + 16 * i][wd];
        bv[i] = ws[tx + 16 * i][wd];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[i][j] = __dp4a(av[i], bv[j], c[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int t = t0 + ty + 16 * i;
      if (t < tokens) acc[static_cast<size_t>(t) * out + n0 + tx + 16 * j] = c[i][j];
    }
}

__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

enum PeakForm : int { kS8 = 0, kF16F32 = 1, kF16F16 = 2 };
const char *const kPeakName[3] = {"s8 m16n8k32 -> s32", "f16 m16n8k16 -> f32", "f16 m16n8k16 -> f16"};
constexpr double kPeakOps[3] = {2.0 * 16 * 8 * 32, 2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 16};
constexpr int kChains = 8;

// Register-only loops, eight independent chains a warp, +A B then -A B so the accumulators stay
// bounded; block 0's clock64 / %globaltimer give the SM clock the loop ran at.
template <int FORM>
__global__ void __launch_bounds__(256) peak_kernel(int iters, uint32_t seed, float *sink, unsigned long long *clocks) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4], b[2], bn[2];
  for (int i = 0; i < 4; ++i) {
    const uint32_t h = mix(id * 8u + i + seed);
    a[i] = FORM == kS8 ? (h & 0x3F3F3F3Fu) : (0x3C003C00u ^ (h & 0x03FF03FFu));
  }
  for (int i = 0; i < 2; ++i) {
    const uint32_t h = mix(id * 8u + 4 + i + seed);
    if (FORM == kS8) {
      b[i] = h & 0x3F3F3F3Fu;
      uint32_t r = 0;
      for (int j = 0; j < 4; ++j) r |= static_cast<uint32_t>(static_cast<uint8_t>(-static_cast<int8_t>((b[i] >> (8 * j)) & 0xFF))) << (8 * j);
      bn[i] = r;
    } else {
      b[i] = 0x3C003C00u ^ (h & 0x03FF03FFu);
      bn[i] = b[i] ^ 0x80008000u;
    }
  }
  int32_t iacc[kChains][4] = {};
  float facc[kChains][4] = {};
  uint32_t hacc[kChains][2] = {};
  unsigned long long c0 = 0, g0 = 0;
  if (id == 0) {
    c0 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const uint32_t *bb = pass == 0 ? b : bn;
#pragma unroll
      for (int c = 0; c < kChains; ++c) {
        if (FORM == kS8) {
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+r"(iacc[c][0]), "+r"(iacc[c][1]), "+r"(iacc[c][2]), "+r"(iacc[c][3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
        } else if (FORM == kF16F32) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(facc[c][0]), "+f"(facc[c][1]), "+f"(facc[c][2]), "+f"(facc[c][3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
        } else {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                       : "+r"(hacc[c][0]), "+r"(hacc[c][1])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
        }
      }
    }
  }
  if (id == 0) {
    unsigned long long g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    clocks[0] = clock64() - c0;
    clocks[1] = g1 - g0;
  }
  float s = 0.0f;
  for (int c = 0; c < kChains; ++c) {
    s += static_cast<float>(iacc[c][0] + iacc[c][1] + iacc[c][2] + iacc[c][3]);
    s += facc[c][0] + facc[c][1] + facc[c][2] + facc[c][3];
    s += __uint_as_float(hacc[c][0]) + __uint_as_float(hacc[c][1]);
  }
  sink[id] = s;
}

// L2-resident read bandwidth: a buffer well inside the L2 read with 16-byte
// ld.global.cg, four independent loads in flight per thread per step, power-of-two indexing.
__global__ void __launch_bounds__(256) l2_read(const uint4 *p, uint32_t mask, int iters, unsigned *sink) {
  uint32_t acc = 0;
  const uint32_t stride = gridDim.x * blockDim.x;
  uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) & mask;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint4 v = __ldcg(p + ((i + u * stride) & mask));
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    i = (i + 4 * stride) & mask;
  }
  if (acc == 0x12345678u) *sink = acc;
}

// The working set: the largest power of two within half the L2 (32 MiB at most), so it stays
// resident on a card with a small L2 too (phase 2a's fixed 32 MiB read DRAM there).
double l2_read_tbs(int sms, size_t l2_bytes, size_t *set_bytes) {
  size_t bytes = 1ull << 20;
  while (bytes * 2 <= l2_bytes / 2 && bytes * 2 <= (32ull << 20)) bytes *= 2;
  *set_bytes = bytes;
  DeviceBytes buf(bytes), sink(4);
  BENCH_CUDA(cudaMemset(buf.p, 1, bytes));
  const int blocks = 8 * sms, iters = 256;
  const uint32_t mask = static_cast<uint32_t>(bytes / 16 - 1);
  l2_read<<<blocks, 256>>>(buf.as<uint4>(), mask, iters, sink.as<unsigned>());
  cudaEvent_t a, b;
  BENCH_CUDA(cudaEventCreate(&a));
  BENCH_CUDA(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 5; ++r) {
    BENCH_CUDA(cudaEventRecord(a));
    l2_read<<<blocks, 256>>>(buf.as<uint4>(), mask, iters, sink.as<unsigned>());
    BENCH_CUDA(cudaEventRecord(b));
    BENCH_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  const double moved = static_cast<double>(blocks) * 256 * iters * 4 * 16;
  return moved / (best * 1e-3) / 1e12;
}

struct Rate {
  double tops = 0.0, mhz = 0.0;
};

template <int FORM> Rate peak(int sms, int iters, int reps, int warmup_ms) {
  const int blocks = 4 * sms;
  DeviceBytes sink(static_cast<size_t>(blocks) * 256 * 4), clocks(16);
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  float spent = 0.0f;
  while (spent < static_cast<float>(warmup_ms)) {
    BENCH_CUDA(cudaEventRecord(e0));
    peak_kernel<FORM><<<blocks, 256>>>(iters, 7u, sink.as<float>(), clocks.as<unsigned long long>());
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    spent += ms;
  }
  double best = 1e30;
  unsigned long long hc[2] = {0, 0};
  for (int r = 0; r < reps; ++r) {
    BENCH_CUDA(cudaEventRecord(e0));
    peak_kernel<FORM><<<blocks, 256>>>(iters, 11u + r, sink.as<float>(), clocks.as<unsigned long long>());
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < best) {
      best = ms;
      BENCH_CUDA(cudaMemcpy(hc, clocks.p, sizeof(hc), cudaMemcpyDeviceToHost));
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  Rate rate;
  const double ops = kPeakOps[FORM] * 2.0 * kChains * static_cast<double>(iters) * blocks * 8;
  rate.tops = ops / (best * 1e-3) / 1e12;
  rate.mhz = hc[1] > 0 ? static_cast<double>(hc[0]) / static_cast<double>(hc[1]) * 1e3 : 0.0;
  return rate;
}

}  // namespace meter

// ---- the bench ---------------------------------------------------------------------------------

namespace {

using namespace sm8x;
using trellis_test::DeviceBytes;

int g_reps = 5;
int g_sms = 0;
double g_s8_peak = 0.0;

void set_env(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

// ignis_trellis_prepare under a fake-quant mode (phase 1's IGNIS_PROTO_FAKEQUANT): "off" is today's
// fused prefill kernel, "w8a8tok" phase 1's V2. Not fatal: if the leaf does not prepare on a card,
// the bench says why and runs the dense route alone (no fused timings, no V2 comparisons).
bool g_fused = true;

bool prepare_fused(const char *fakequant) {
  set_env("IGNIS_TRELLIS_SM_CAP", "0");
  set_env("IGNIS_PROTO_FAKEQUANT", fakequant);
  if (ignis_trellis_prepare() != 0) {
    std::printf("fused trellis linear unavailable on this card (ignis_trellis_prepare: %s): the dense route runs alone\n",
                ignis_trellis_last_error());
    g_fused = false;
  }
  return g_fused;
}

// Device microseconds per launch: `inner` back-to-back launches between two events, best of g_reps.
template <typename F> double time_us(F &&launch, int inner, cudaStream_t s) {
  for (int i = 0; i < 2; ++i) launch(i);
  BENCH_CUDA(cudaStreamSynchronize(s));
  cudaEvent_t a, b;
  BENCH_CUDA(cudaEventCreate(&a));
  BENCH_CUDA(cudaEventCreate(&b));
  double best = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    BENCH_CUDA(cudaEventRecord(a, s));
    for (int i = 0; i < inner; ++i) launch(i);
    BENCH_CUDA(cudaEventRecord(b, s));
    BENCH_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, 1000.0 * ms / inner);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return best;
}

// ---- the artifact ------------------------------------------------------------------------------

// A `.ninfer` v2 artifact's directory, read the way crates/artifact's Reader does: the 8-byte magic,
// the JSON length, the JSON, the payload at the next 4 KiB.
struct Artifact {
  FILE *f = nullptr;
  std::string json;
  uint64_t payload = 0;

  bool open(const std::string &path) {
    f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    unsigned char prefix[16];
    if (std::fread(prefix, 1, 16, f) != 16 || std::memcmp(prefix, "NINFER\0\x02", 8) != 0) return false;
    uint64_t n = 0;
    std::memcpy(&n, prefix + 8, 8);
    json.resize(n);
    if (std::fread(json.data(), 1, n, f) != n) return false;
    payload = (16 + n + 4095) / 4096 * 4096;
    return true;
  }

  struct Entry {
    std::string format;
    uint64_t offset = 0, bytes = 0;
    std::vector<uint64_t> shape;
  };

  bool find(const std::string &name, Entry &e) const {
    const size_t at = json.find("\"name\":\"" + name + "\"");
    if (at == std::string::npos) return false;
    const size_t lo = json.rfind('{', at), hi = json.find('}', at);
    const std::string obj = json.substr(lo, hi - lo + 1);
    auto number = [&](const char *key) {
      const size_t p = obj.find(key);
      return p == std::string::npos ? 0ull : std::strtoull(obj.c_str() + p + std::strlen(key), nullptr, 10);
    };
    e.offset = number("\"offset\":");
    e.bytes = number("\"bytes\":");
    const size_t fp = obj.find("\"format\":\"");
    if (fp == std::string::npos) return false;
    e.format = obj.substr(fp + 10, obj.find('"', fp + 10) - fp - 10);
    const size_t sp = obj.find("\"shape\":[");
    if (sp == std::string::npos) return false;
    const char *c = obj.c_str() + sp + 9;
    e.shape.clear();
    while (*c != ']' && *c != '\0') {
      char *end = nullptr;
      e.shape.push_back(std::strtoull(c, &end, 10));
      c = *end == ',' ? end + 1 : end;
    }
    return true;
  }

  void read(uint64_t offset, uint64_t bytes, void *dst) const {
#ifdef _WIN32
    _fseeki64(f, static_cast<long long>(payload + offset), SEEK_SET);
#else
    fseeko(f, static_cast<off_t>(payload + offset), SEEK_SET);
#endif
    if (std::fread(dst, 1, bytes, f) != bytes) {
      std::fprintf(stderr, "FATAL: short read from the artifact\n");
      std::exit(EXIT_FAILURE);
    }
  }
};

// A trellis matrix of the artifact: [out, in], its bytes the trellis, suh (in fp16), svh (out fp16).
struct HostTrellis {
  std::string name;
  uint32_t in = 0, out = 0, k2 = 0;
  std::vector<uint16_t> words, suh, svh;
};

bool load_trellis(const Artifact &art, const std::string &name, HostTrellis &m) {
  Artifact::Entry e;
  if (!art.find(name, e) || e.shape.size() != 2 || e.format.rfind("TRELLIS_MUL1_K", 0) != 0) return false;
  m.name = name;
  m.out = static_cast<uint32_t>(e.shape[0]);
  m.in = static_cast<uint32_t>(e.shape[1]);
  m.k2 = 2 * static_cast<uint32_t>(std::atoi(e.format.c_str() + 14));
  const uint64_t tb = static_cast<uint64_t>(m.in) * m.out / 16 * m.k2;
  std::vector<uint8_t> bytes(tb + 2ull * m.in + 2ull * m.out);
  art.read(e.offset, bytes.size(), bytes.data());
  m.words.resize(tb / 2);
  std::memcpy(m.words.data(), bytes.data(), tb);
  m.suh.resize(m.in);
  std::memcpy(m.suh.data(), bytes.data() + tb, 2ull * m.in);
  m.svh.resize(m.out);
  std::memcpy(m.svh.data(), bytes.data() + tb + 2ull * m.in, 2ull * m.out);
  return true;
}

// The host decode's code integers s of tile (tk, tc): the format restated (moe_fixture.h's
// host_trellis_decode), s = the byte sum of state * 0x83DCD12D, at element (r, c) of the tile.
void host_tile_codes(const uint16_t *words, uint32_t k2, int tn, int tk, int tc, int (&s)[16][16]) {
  const uint32_t w16 = moe_test::trellis_tile_words16(k2);
  const int bits = static_cast<int>(w16) * 16;
  int perm[256];
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2;
    const int c0 = t / 4;
    const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
    for (int j = 0; j < 8; ++j) perm[t * 8 + j] = rows[j % 4] * 16 + c0 + (j >= 4 ? 8 : 0);
  }
  const uint16_t *tile = words + (static_cast<size_t>(tk) * tn + tc) * w16;
  auto bit = [&](int b) {
    b = ((b % bits) + bits) % bits;
    const uint32_t u32 = tile[2 * (b / 32)] | (static_cast<uint32_t>(tile[2 * (b / 32) + 1]) << 16);
    return (u32 >> (31 - (b % 32))) & 1u;
  };
  for (int p = 0, e = 0; p < 256; ++p) {
    e += moe_test::trellis_step_bits(k2, p);
    uint32_t state = 0;
    for (int j = 0; j < 16; ++j) state = (state << 1) | bit(e - 16 + j);
    const uint32_t prod = state * 0x83DCD12Du;
    s[perm[p] / 16][perm[p] % 16] = static_cast<int>((prod & 255u) + ((prod >> 8) & 255u) + ((prod >> 16) & 255u) + (prod >> 24));
  }
}

int host_w8(int s) {
  const float v = static_cast<float>(s - 510) * 0.25f;
  return std::max(-127, std::min(127, static_cast<int>(std::nearbyint(v))));  // ties to even (FE_TONEAREST)
}

size_t dense_index(int in, int n, int k) {
  return (static_cast<size_t>(n / kBlockN) * (in / kBlockK) + k / kBlockK) * kBlockBytes + (n % kBlockN) * kBlockK + (k % kBlockK);
}

// The activation stand-in: hidden-state-like BF16 rows, roughly Gaussian per entry, a few loud
// channels (one in 97 at 24x), a per-token gain in [0.5, 1.5), and token 0 at 8x.
std::vector<uint16_t> standin_x(int tokens, int in, uint32_t seed) {
  std::vector<uint16_t> x(static_cast<size_t>(tokens) * in);
  for (int t = 0; t < tokens; ++t) {
    const float gain = (0.5f + 0.5f * (moe_test::hash_uniform(seed + 1, t, 1.0f) + 1.0f)) * (t == 0 ? 8.0f : 1.0f);
    for (int k = 0; k < in; ++k) {
      const size_t i = static_cast<size_t>(t) * in + k;
      const float u = moe_test::hash_uniform(seed, 3 * i, 1.0f) + moe_test::hash_uniform(seed, 3 * i + 1, 1.0f) +
                      moe_test::hash_uniform(seed, 3 * i + 2, 1.0f);
      const float ch = moe_test::hash_u32(seed + 2, k) % 97 == 0 ? 24.0f : 1.0f;
      x[i] = moe_test::f32_to_bf16(u * ch * gain);
    }
  }
  return x;
}

// ---- the levers (phase 2c) -----------------------------------------------------------------------

// A configuration of the route: phase 2a's is all three off.
struct Lever {
  std::string name;
  bool a8_shared = false;   // A: one A8 launch per shared input (--a8 shared)
  bool persistent = false;  // B: the persistent tile scheduler with a stream-K tail (--sched persistent)
  bool slice = false;       // C: decode into L2-sized output slices, each consumed at once (--decode slice)
};

std::string lever_name(const Lever &l) {
  std::string n;
  if (l.a8_shared) n += "A";
  if (l.persistent) n += n.empty() ? "B" : "+B";
  if (l.slice) n += n.empty() ? "C" : "+C";
  return n.empty() ? std::string("phase 2a") : n;
}

// One scheduler variant of the GEMM: the speed section picks the fastest per shape, the check runs
// every one.
struct RouteVariant {
  int raster = 0;
  int sk = 0;        // persistent: 0 stream-K over the last partial wave, 1 over it and one full wave more
  bool cap = false;  // persistent: the co-resident overlap's register-capped instantiation
};

std::vector<RouteVariant> route_variants(const Lever &cfg) {
  if (!cfg.persistent) return {RouteVariant{0, 0, false}};
  std::vector<RouteVariant> v;
  for (const bool cap : {false, true})
    for (const int raster : {0, 1})
      for (const int sk : {0, 1}) v.push_back(RouteVariant{raster, sk, cap});
  return v;
}

std::string route_label(const Lever &cfg) {
  std::string s;
  if (cfg.persistent) s += ", persistent: 2 rasters x 2 stream-K splits x 2 register caps";
  if (cfg.slice) s += ", slice decode";
  return s;
}

void gemm_sched(const Lever &cfg, const RouteVariant &rv, const GemmCall &c, cudaStream_t s) {
  if (cfg.persistent) {
    launch_gemm_persistent(c, rv.sk, s, rv.cap);
  } else {
    launch_gemm(c, s);
  }
}

// The GEMM call of output blocks [nb0, nb0 + nbs) of c, its W the slice in wbuf.
GemmCall slice_call(const GemmCall &c, int nb0, int nbs, int8_t *wbuf) {
  GemmCall cs = c;
  cs.w = wbuf;
  cs.out = nbs * kBlockN;
  cs.svh = static_cast<const __half *>(c.svh) + static_cast<size_t>(nb0) * kBlockN;
  if (c.y != nullptr) cs.y = static_cast<__nv_bfloat16 *>(c.y) + static_cast<size_t>(nb0) * kBlockN;
  if (c.raw != nullptr) {
    cs.raw = c.raw + static_cast<size_t>(nb0) * kBlockN;
    cs.ldr = c.ldr == 0 ? c.out : c.ldr;
  }
  return cs;
}

// The route's GEMM under a configuration: with --decode slice, each output slice is decoded into
// wbuf (one slice's room) and multiplied at once, while it is still in the L2. part (the
// decomposition of --diag): 0 both, 1 the slices' decodes alone, 2 their GEMMs alone.
void run_route_gemm(const Lever &cfg, const RouteVariant &rv, const GemmCall &c, const ignis_trellis_weight &tw, int8_t *wbuf,
                    cudaStream_t s, int part = 0) {
  if (!cfg.slice) {
    gemm_sched(cfg, rv, c, s);
    return;
  }
  const int nb = c.out / kBlockN, per = slice_even(c.in, nb);
  for (int nb0 = 0; nb0 < nb; nb0 += per) {
    const int nbs = std::min(per, nb - nb0);
    if (part != 2) launch_decode(trellis_slice(tw, nb0, nbs), wbuf, s);
    if (part == 1) continue;
    gemm_sched(cfg, rv, slice_call(c, nb0, nbs, wbuf), s);
  }
}

// The slice decode's codes for the W8 check: every slice decoded into its place of a whole W.
std::vector<int8_t> slice_decode_full(const Lever &, const ignis_trellis_weight &tw, int8_t *wfull, cudaStream_t s) {
  const int in = static_cast<int>(tw.in), nb = static_cast<int>(tw.out) / kBlockN, per = slice_even(in, nb);
  BENCH_CUDA(cudaMemsetAsync(wfull, 0x5A, static_cast<size_t>(in) * tw.out, s));
  for (int nb0 = 0; nb0 < nb; nb0 += per) {
    const int nbs = std::min(per, nb - nb0);
    launch_decode(trellis_slice(tw, nb0, nbs), wfull + static_cast<size_t>(nb0) * (in / kBlockK) * kBlockBytes, s);
  }
  BENCH_CUDA(cudaStreamSynchronize(s));
  return trellis_test::download<int8_t>(wfull, static_cast<size_t>(in) * tw.out);
}


// ---- the check -----------------------------------------------------------------------------------

int g_failed = 0;

void verdict(bool ok, const std::string &what) {
  std::printf("    %s: %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok) ++g_failed;
}

struct Agreement {
  double worst = 0.0;
  double rel_rms = 0.0;
  size_t over = 0;
};

Agreement agree(const std::vector<double> &got, const std::vector<double> &want, const std::vector<double> &bound) {
  Agreement a;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(got[i] - want[i]);
    const double r = bound[i] > 0.0 ? err / bound[i] : (err > 0.0 ? 1e30 : 0.0);
    if (!(r <= 1.0)) ++a.over;
    a.worst = std::max(a.worst, std::isnan(r) ? 1e30 : r);
    num += err * err;
    den += want[i] * want[i];
  }
  a.rel_rms = den > 0.0 ? std::sqrt(num / den) : 0.0;
  return a;
}

// The matrices that read the same input as `name` (lever A's groups), in the layer's order, and
// the index of `name` among them. A matrix outside the groups is a group of one.
std::vector<std::string> input_group(const std::string &name, int &self) {
  static const std::vector<std::vector<std::string>> groups = {
      {"mlp/gate_proj", "mlp/up_proj"}, {"gdn/in_proj_qkv", "gdn/in_proj_z"}, {"attention/q_proj", "attention/k_proj", "attention/v_proj"}};
  for (const auto &g : groups)
    for (size_t i = 0; i < g.size(); ++i) {
      const std::string &leaf = g[i];
      if (name.size() > leaf.size() && name.compare(name.size() - leaf.size(), leaf.size(), leaf) == 0) {
        const std::string prefix = name.substr(0, name.size() - leaf.size());
        std::vector<std::string> out;
        for (const auto &l : g) out.push_back(prefix + l);
        self = static_cast<int>(i);
        return out;
      }
    }
  self = 0;
  return {name};
}

// A trellis matrix's suh alone (2 in bytes after its trellis).
bool load_suh(const Artifact &art, const std::string &name, uint32_t in, std::vector<uint16_t> &suh) {
  Artifact::Entry e;
  if (!art.find(name, e) || e.shape.size() != 2 || e.format.rfind("TRELLIS_MUL1_K", 0) != 0 || e.shape[1] != in) return false;
  const uint64_t k2 = 2 * static_cast<uint64_t>(std::atoi(e.format.c_str() + 14));
  const uint64_t tb = static_cast<uint64_t>(e.shape[1]) * e.shape[0] / 16 * k2;
  suh.resize(in);
  art.read(e.offset + tb, 2ull * in, suh.data());
  return true;
}

std::vector<int> g_cfg_failed;  // per configuration of the check

// One matrix: the references once (the host decode's codes, V2's rotation and output, the integer
// reference, the fp64 reference and its bounds), then the seven checks for every configuration.
void check_matrix(const Artifact &art, const std::string &name, int tokens, cudaStream_t stream, const std::vector<Lever> &cfgs) {
  HostTrellis m;
  if (!load_trellis(art, name, m)) {
    std::printf("  %s: not a trellis matrix of the artifact\n", name.c_str());
    ++g_failed;
    for (int &f : g_cfg_failed) ++f;
    return;
  }
  const int in = static_cast<int>(m.in), out = static_cast<int>(m.out);
  std::printf("  %s: in %d out %d K %.1f, %d tokens\n", name.c_str(), in, out, m.k2 / 2.0, tokens);
  DeviceBytes d_words(m.words.size() * 2 + 256), d_suh(2ull * in), d_svh(2ull * out);
  trellis_test::upload(d_words, m.words);
  trellis_test::upload(d_suh, m.suh);
  trellis_test::upload(d_svh, m.svh);
  const ignis_trellis_weight tw{d_words.p, d_suh.p, d_svh.p, m.k2, m.in, m.out, 0};

  // The host decode's codes, every tile of every 8th tile row.
  std::vector<int8_t> want_w8;
  {
    int s[16][16];
    for (int tk = 3; tk < in / 16; tk += 8)
      for (int tc = 0; tc < out / 16; ++tc) {
        host_tile_codes(m.words.data(), m.k2, out / 16, tk, tc, s);
        for (int r = 0; r < 16; ++r)
          for (int c = 0; c < 16; ++c) want_w8.push_back(static_cast<int8_t>(host_w8(s[r][c])));
      }
  }
  auto w8_wrong = [&](const std::vector<int8_t> &w) {
    size_t wrong = 0, i = 0;
    for (int tk = 3; tk < in / 16; tk += 8)
      for (int tc = 0; tc < out / 16; ++tc)
        for (int r = 0; r < 16; ++r)
          for (int c = 0; c < 16; ++c) wrong += w[dense_index(in, tc * 16 + c, tk * 16 + r)] != want_w8[i++];
    return wrong;
  };

  // The whole dense W and the per-matrix A8: the integer reference's operands.
  DeviceBytes d_w(static_cast<size_t>(in) * out);
  launch_decode(tw, d_w.as<int8_t>(), stream);
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int8_t> w = trellis_test::download<int8_t>(d_w.p, static_cast<size_t>(in) * out);
  const std::vector<uint16_t> xh = standin_x(tokens, in, 4242);
  DeviceBytes d_x(xh.size() * 2), d_a(static_cast<size_t>(tokens) * in), d_sa(4ull * tokens), d_sq(4ull * tokens);
  trellis_test::upload(d_x, xh);
  launch_a8(d_x.p, d_suh.p, tokens, in, d_a.as<int8_t>(), d_sa.as<float>(), d_sq.as<int32_t>(), stream);
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int8_t> a = trellis_test::download<int8_t>(d_a.p, static_cast<size_t>(tokens) * in);
  const std::vector<float> sa = trellis_test::download<float>(d_sa.p, tokens);
  const std::vector<int32_t> sq = trellis_test::download<int32_t>(d_sq.p, tokens);

  // Lever A's group: the partners' suh, and the per-matrix A8 of every member.
  int self = 0;
  const std::vector<std::string> group = input_group(name, self);
  const int gn = static_cast<int>(group.size());
  std::vector<DeviceBytes> g_suh(gn);
  std::vector<std::vector<int8_t>> g_a(gn);
  std::vector<std::vector<float>> g_sa(gn);
  std::vector<std::vector<int32_t>> g_sq(gn);
  for (int i = 0; i < gn; ++i) {
    if (i == self) {
      g_a[i] = a;
      g_sa[i] = sa;
      g_sq[i] = sq;
      continue;
    }
    std::vector<uint16_t> suh;
    if (!load_suh(art, group[i], m.in, suh)) {
      std::printf("    %s: not a trellis matrix of the artifact with in %d\n", group[i].c_str(), in);
      ++g_failed;
      for (int &f : g_cfg_failed) ++f;
      return;
    }
    g_suh[i] = DeviceBytes(2ull * in);
    trellis_test::upload(g_suh[i], suh);
    DeviceBytes pa(static_cast<size_t>(tokens) * in), psa(4ull * tokens), psq(4ull * tokens);
    launch_a8(d_x.p, g_suh[i].p, tokens, in, pa.as<int8_t>(), psa.as<float>(), psq.as<int32_t>(), stream);
    BENCH_CUDA(cudaStreamSynchronize(stream));
    g_a[i] = trellis_test::download<int8_t>(pa.p, static_cast<size_t>(tokens) * in);
    g_sa[i] = trellis_test::download<float>(psa.p, tokens);
    g_sq[i] = trellis_test::download<int32_t>(psq.p, tokens);
  }

  // V2's rotation and output.
  DeviceBytes d_yv(2ull * tokens * out);
  const bool v2 = g_fused && prepare_fused("w8a8tok");
  if (!v2) std::printf("    V2 comparisons skipped: the fused trellis linear is unavailable\n");
  std::vector<uint16_t> xr;
  std::vector<float> inv;
  if (v2) {
    const uint64_t ws_bytes = ignis_trellis_workspace_bytes(m.in, m.out, 1, tokens);
    DeviceBytes ws_mem(ws_bytes);
    ignis_trellis_workspace ws{ws_mem.p, ws_bytes};
    TL_RC(ignis_trellis_workspace_init(&ws, stream));
    TL_RC(ignis_trellis_linear(&tw, d_x.p, tokens, d_yv.p, m.out, IGNIS_TRELLIS_STORE, &ws, stream));
    BENCH_CUDA(cudaStreamSynchronize(stream));
    const ignis_tl::Layout l = ignis_tl::layout(m.in, m.out, 1, tokens, 1);
    xr = trellis_test::download<uint16_t>(static_cast<char *>(ws_mem.p) + l.xr, static_cast<size_t>(tokens) * in);
    inv = trellis_test::download<float>(static_cast<char *>(ws_mem.p) + l.inv, static_cast<size_t>(in / 128) * tokens);
    prepare_fused("off");
  }
  struct A8Diff {
    size_t values = 0, scales = 0, sums = 0;
  };
  auto a8_vs_v2 = [&](const std::vector<int8_t> &qa, const std::vector<float> &qs, const std::vector<int32_t> &qq) {
    A8Diff d;
    for (int t = 0; t < tokens; ++t) {
      long long sum = 0;
      for (int k = 0; k < in; ++k) {
        const int8_t q = qa[a_index(tokens, t, k)];
        d.values += moe_test::f16_to_f32(xr[static_cast<size_t>(t) * in + k]) != static_cast<float>(q);
        sum += q;
      }
      for (int b = 0; b < in / 128; ++b) d.scales += inv[static_cast<size_t>(b) * tokens + t] != qs[t];
      d.sums += sum != qq[t];
    }
    return d;
  };

  // The integer reference, from the whole W and the per-matrix A8.
  DeviceBytes d_raw(4ull * tokens * out), d_ref(4ull * tokens * out);
  meter::ref_int_gemm<<<dim3(out / 64, (tokens + 63) / 64), 256, 0, stream>>>(d_a.as<int8_t>(), d_w.as<int8_t>(), tokens, in, out, d_ref.as<int32_t>());
  BENCH_CUDA(cudaGetLastError());
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int32_t> ref = trellis_test::download<int32_t>(d_ref.p, static_cast<size_t>(tokens) * out);
  long long ref_amax = 0;
  for (const int32_t v : ref) ref_amax = std::max(ref_amax, std::llabs(static_cast<long long>(v)));

  // The fp64 reference of the quantized product and the bounds.
  const std::vector<double> yv = trellis_test::bf16_to_f64(trellis_test::download<uint16_t>(d_yv.p, static_cast<size_t>(tokens) * out));
  const double sw = map_sw(), off = map_off();
  // The tolerance's scale s (trellis_test_common.h): |svh| times the RMS over the output's
  // 128-column block of sqrt(sum_k (a_k w_kj)^2) = sqrt(sum_k a_k^2 mean_j w_kj^2).
  std::vector<double> wsq(static_cast<size_t>(in) * (out / 128), 0.0);
  for (int n = 0; n < out; ++n)
    for (int k = 0; k < in; ++k) {
      const double wv = sw * w[dense_index(in, n, k)] + off;
      wsq[static_cast<size_t>(k) * (out / 128) + n / 128] += wv * wv / 128.0;
    }
  std::vector<double> yref(static_cast<size_t>(tokens) * out), sc(yref.size()), b16(yref.size()), b8(yref.size());
  for (int t = 0; t < tokens; ++t) {
    double *y = yref.data() + static_cast<size_t>(t) * out;
    for (int n = 0; n < out; ++n) y[n] = static_cast<double>(sa[t]) * (sw * ref[static_cast<size_t>(t) * out + n] + off * sq[t]);
    moe_test::hadamard128_inplace(y, out);
    std::vector<double> s2(out / 128, 0.0);
    for (int k = 0; k < in; ++k) {
      const double av = static_cast<double>(sa[t]) * a[a_index(tokens, t, k)];
      for (int b = 0; b < out / 128; ++b) s2[b] += av * av * wsq[static_cast<size_t>(k) * (out / 128) + b];
    }
    for (int n = 0; n < out; ++n) {
      const size_t i = static_cast<size_t>(t) * out + n;
      const double sv = moe_test::f16_to_f32(m.svh[n]);
      y[n] *= sv;
      sc[i] = std::fabs(sv) * std::sqrt(s2[n / 128]);
      b16[i] = std::ldexp(std::fabs(y[n]), -8) + std::ldexp(sc[i], -8);  // fp16 accumulation (V2)
      b8[i] = std::ldexp(std::fabs(y[n]), -8) + std::ldexp(sc[i], -9);   // BF16 rounding of an exact sum
    }
  }
  // Both outputs are BF16: each sits within its own bound of the exact value, so their difference
  // sits within the sum of the two, 2^-7 |Y| + 3 2^-9 s.
  std::vector<double> bsum(yref.size());
  for (size_t i = 0; i < bsum.size(); ++i) bsum[i] = b8[i] + b16[i];
  const Agreement vr = v2 ? agree(yv, yref, b16) : Agreement{};

  // The seven checks under every configuration, each on that configuration's own kernels.
  DeviceBytes d_wc(static_cast<size_t>(in) * out), d_ac(static_cast<size_t>(tokens) * in * kA8MaxShared),
      d_sac(4ull * tokens * kA8MaxShared), d_sqc(4ull * tokens * kA8MaxShared), d_yd(2ull * tokens * out);
  for (size_t ci = 0; ci < cfgs.size(); ++ci) {
    const Lever &cfg = cfgs[ci];
    const int failed_before = g_failed;
    if (cfgs.size() > 1) std::printf("   [%s]\n", cfg.name.c_str());

    // W8: the configuration's decode against the host decode.
    {
      const size_t wrong = cfg.slice ? w8_wrong(slice_decode_full(cfg, tw, d_wc.as<int8_t>(), stream)) : w8_wrong(w);
      std::printf("    W8 codes against the host decode%s: %zu of %zu differ\n", cfg.slice ? " (the slice decode)" : "", wrong, want_w8.size());
      verdict(wrong == 0, "W8 codes bit-identical to phase 1's quantizer");
    }

    // A8: the configuration's quantization against V2's rotation; the shared launch also against
    // the per-matrix A8 of every member of the group.
    const int8_t *qa = d_a.as<int8_t>();
    const float *qs = d_sa.as<float>();
    const int32_t *qq = d_sq.as<int32_t>();
    if (cfg.a8_shared && gn > 1) {
      A8Shared g{};
      g.count = gn;
      for (int i = 0; i < gn; ++i) {
        g.m[i] = A8Target{static_cast<const __half *>(i == self ? d_suh.p : g_suh[i].p), d_ac.as<int8_t>() + static_cast<size_t>(i) * tokens * in,
                          d_sac.as<float>() + static_cast<size_t>(i) * tokens, d_sqc.as<int32_t>() + static_cast<size_t>(i) * tokens};
      }
      BENCH_CUDA(cudaMemsetAsync(d_ac.p, 0x5A, d_ac.bytes, stream));
      launch_a8_shared(d_x.p, tokens, in, g, stream);
      BENCH_CUDA(cudaStreamSynchronize(stream));
      size_t differ = 0;
      for (int i = 0; i < gn; ++i) {
        const std::vector<int8_t> ga = trellis_test::download<int8_t>(g.m[i].a, static_cast<size_t>(tokens) * in);
        const std::vector<float> gs = trellis_test::download<float>(g.m[i].sa, tokens);
        const std::vector<int32_t> gq = trellis_test::download<int32_t>(g.m[i].sq, tokens);
        size_t d = 0;
        for (size_t k = 0; k < ga.size(); ++k) d += ga[k] != g_a[i][k];
        for (int t = 0; t < tokens; ++t) d += std::memcmp(&gs[t], &g_sa[i][t], 4) != 0 || gq[t] != g_sq[i][t];
        differ += d;
      }
      std::printf("    shared A8 (one launch for %d matrices) against the per-matrix A8 of each: %zu values, scales or sums differ\n", gn, differ);
      qa = g.m[self].a;
      qs = g.m[self].sa;
      qq = g.m[self].sq;
      if (v2) {
        const A8Diff d = a8_vs_v2(trellis_test::download<int8_t>(qa, static_cast<size_t>(tokens) * in), trellis_test::download<float>(qs, tokens),
                                  trellis_test::download<int32_t>(qq, tokens));
        std::printf("    A8 (shared) against V2's rotation: %zu of %zu values, %zu scales and %zu sums differ\n", d.values,
                    static_cast<size_t>(tokens) * in, d.scales, d.sums);
        verdict(differ == 0 && d.values == 0 && d.scales == 0 && d.sums == 0, "A8 bit-identical to V2's and to the per-matrix A8");
      } else {
        verdict(differ == 0, "A8 bit-identical to the per-matrix A8");
      }
    } else if (v2) {
      const A8Diff d = a8_vs_v2(a, sa, sq);
      std::printf("    A8 against V2's rotation: %zu of %zu values, %zu scales and %zu sums differ\n", d.values,
                  static_cast<size_t>(tokens) * in, d.scales, d.sums);
      verdict(d.values == 0 && d.scales == 0 && d.sums == 0, "A8 bit-identical to V2's");
    }

    // The GEMM's int32 accumulators against the integer reference, both tile heights, every
    // scheduler variant the configuration times.
    for (const int bm : {256, 128}) {
      size_t wrong = 0, total = 0;
      for (const RouteVariant &rv : route_variants(cfg)) {
        BENCH_CUDA(cudaMemsetAsync(d_raw.p, 0x5A, d_raw.bytes, stream));
        GemmCall c{qa, d_w.as<int8_t>(), qs, qq, d_svh.p, tokens, in, out, kStore, nullptr, out};
        c.raw = d_raw.as<int32_t>();
        c.bm = bm;
        c.raster = rv.raster;
        run_route_gemm(cfg, rv, c, tw, d_wc.as<int8_t>(), stream);
        BENCH_CUDA(cudaStreamSynchronize(stream));
        const std::vector<int32_t> raw = trellis_test::download<int32_t>(d_raw.p, static_cast<size_t>(tokens) * out);
        for (size_t i = 0; i < raw.size(); ++i) wrong += raw[i] != ref[i];
        total += raw.size();
      }
      std::printf("    int32 accumulators (BM %d%s) against the integer reference: %zu of %zu differ (max |acc| %lld)\n", bm,
                  route_label(cfg).c_str(), wrong, total, ref_amax);
      verdict(wrong == 0, "int32 accumulators exactly equal (BM " + std::to_string(bm) + ")");
    }

    // The outputs: dense and V2 against the fp64 reference of the quantized product, and each other.
    {
      GemmCall c{qa, d_w.as<int8_t>(), qs, qq, d_svh.p, tokens, in, out, kStore, d_yd.p, out};
      run_route_gemm(cfg, route_variants(cfg).front(), c, tw, d_wc.as<int8_t>(), stream);
      BENCH_CUDA(cudaStreamSynchronize(stream));
    }
    const std::vector<double> yd = trellis_test::bf16_to_f64(trellis_test::download<uint16_t>(d_yd.p, static_cast<size_t>(tokens) * out));
    const Agreement dr = agree(yd, yref, b8);
    std::printf("    dense - ref: worst %.3f of 2^-8|Y| + 2^-9 s (BF16 rounding of the exact sum), rel RMS %.3g\n", dr.worst, dr.rel_rms);
    verdict(dr.over == 0, "dense output = the exact product within BF16 rounding");
    if (v2) {
      const Agreement dv = agree(yd, yv, bsum);
      size_t same = 0, one_ulp = 0;
      double max_ulps = 0.0;
      for (size_t i = 0; i < yd.size(); ++i) {
        const double mm = std::max(std::fabs(yd[i]), std::fabs(yv[i]));
        int e = 0;
        std::frexp(mm, &e);
        const double ulp = mm > 0.0 ? std::ldexp(1.0, e - 8) : 1.0;  // BF16: 8 significant bits
        const double u = std::fabs(yd[i] - yv[i]) / ulp;
        same += u == 0.0;
        one_ulp += u <= 1.0;
        max_ulps = std::max(max_ulps, u);
      }
      std::printf("    V2    - ref: worst %.3f of 2^-8|Y| + 2^-8 s (fp16 accumulation), rel RMS %.3g\n", vr.worst, vr.rel_rms);
      std::printf("    dense - V2:  worst %.3f of the two bounds' sum, rel RMS %.3g; BF16 bits equal at %.2f%%, within 1 ulp at %.3f%%, max %.1f ulp\n",
                  dv.worst, dv.rel_rms, 100.0 * same / yd.size(), 100.0 * one_ulp / yd.size(), max_ulps);
      verdict(vr.over == 0, "V2 output = the exact product within fp16 accumulation's bound");
      verdict(dv.over == 0, "dense output = V2 within the two bounds' sum");
    }
    g_cfg_failed[ci] += g_failed - failed_before;
  }
}

// ---- speed -------------------------------------------------------------------------------------

struct Shape {
  const char *name;
  uint32_t in, out, k2;
  int swiglu;
};

// bench_trellis_linear.cu's tables: the body's matrices at the quant's dominant K, and one decode
// token's (matrix, K) mix over the 64 layers.
const Shape kBody[] = {
    {"gdn qkv", 5120, 10240, 6, 0}, {"gdn z", 5120, 6144, 6, 0}, {"gdn out", 6144, 5120, 6, 0},
    {"gqa q", 5120, 12288, 6, 0},   {"gqa k", 5120, 1024, 8, 0},  {"gqa v", 5120, 1024, 8, 0},
    {"gqa o", 6144, 5120, 6, 0},    {"mlp gate+up", 5120, 17408, 6, 1}, {"mlp down", 17408, 5120, 6, 0},
};
struct MixEntry {
  Shape shape;
  int calls;
};
const MixEntry kMix[] = {
    {{"gdn qkv", 5120, 10240, 4, 0}, 4},  {{"gdn qkv", 5120, 10240, 6, 0}, 42}, {{"gdn qkv", 5120, 10240, 8, 0}, 2},
    {{"gdn z", 5120, 6144, 4, 0}, 3},     {{"gdn z", 5120, 6144, 6, 0}, 40},    {{"gdn z", 5120, 6144, 8, 0}, 5},
    {{"gdn out", 6144, 5120, 6, 0}, 40},  {{"gdn out", 6144, 5120, 8, 0}, 8},
    {{"gqa q", 5120, 12288, 4, 0}, 6},    {{"gqa q", 5120, 12288, 6, 0}, 10},
    {{"gqa k, v", 5120, 1024, 8, 0}, 28}, {{"gqa k, v", 5120, 1024, 10, 0}, 4},
    {{"gqa o", 6144, 5120, 6, 0}, 14},    {{"gqa o", 6144, 5120, 8, 0}, 2},
    {{"mlp gate+up", 5120, 17408, 4, 1}, 9}, {{"mlp gate+up", 5120, 17408, 6, 1}, 49}, {{"mlp gate+up", 5120, 17408, 8, 1}, 6},
    {{"mlp down", 17408, 5120, 4, 0}, 3}, {{"mlp down", 17408, 5120, 6, 0}, 53}, {{"mlp down", 17408, 5120, 8, 0}, 8},
};

constexpr uint64_t kPoolBytes = 512ull << 20;  // well past the L2

// Copies of a matrix's trellis (hash words) and scales, enough that cycling them misses the L2.
struct Pool {
  std::vector<DeviceBytes> mem;
  std::vector<ignis_trellis_weight> w;
  int copies = 0;
};

uint64_t trellis_bytes(const Shape &s) { return static_cast<uint64_t>(s.in) * s.out / 16 * s.k2; }

void make_pool(Pool &pool, const Shape &s, int per) {
  pool.copies = static_cast<int>(std::max<uint64_t>(2, std::min<uint64_t>(32, kPoolBytes / (trellis_bytes(s) * per) + 1)));
  const uint64_t n_words = trellis_bytes(s) / 2;
  for (int c = 0; c < pool.copies * per; ++c) {
    pool.mem.emplace_back(n_words * 2 + 2ull * s.in + 2ull * s.out + 256);
    char *base = static_cast<char *>(pool.mem.back().p);
    trellis_test::device_hash_words(base, n_words, 90000 + c);
    std::vector<uint16_t> scales(s.in + s.out, moe_test::f32_to_f16(0.03f));
    BENCH_CUDA(cudaMemcpy(base + n_words * 2, scales.data(), scales.size() * 2, cudaMemcpyHostToDevice));
    pool.w.push_back(ignis_trellis_weight{base, base + n_words * 2, base + n_words * 2 + 2ull * s.in, s.k2, s.in, s.out, 0});
  }
}

// Today's fused prefill kernel, a call of `tokens` tokens cycling the pool, graph-replayed as
// bench_trellis_linear.cu times it (rotation included).
double time_fused(const Pool &pool, const Shape &s, int tokens, cudaStream_t stream) {
  const int per = s.swiglu ? 2 : 1;
  const uint64_t ws_bytes = ignis_trellis_workspace_bytes(s.in, s.out, per, tokens);
  DeviceBytes ws_mem(ws_bytes), x(static_cast<size_t>(tokens) * s.in * 2), y(static_cast<size_t>(tokens) * s.out * 2);
  ignis_trellis_workspace ws{ws_mem.p, ws_bytes};
  TL_RC(ignis_trellis_workspace_init(&ws, stream));
  const std::vector<uint16_t> xh = standin_x(tokens, s.in, 91000);
  BENCH_CUDA(cudaMemcpy(x.p, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  const int calls = 8;
  auto issue = [&](int i) {
    const int c = i % pool.copies;
    if (s.swiglu) {
      TL_RC(ignis_trellis_linear_swiglu(&pool.w[2 * c], &pool.w[2 * c + 1], x.p, tokens, y.p, s.out, &ws, stream));
    } else {
      TL_RC(ignis_trellis_linear(&pool.w[c], x.p, tokens, y.p, s.out, IGNIS_TRELLIS_STORE, &ws, stream));
    }
  };
  for (int i = 0; i < 2; ++i) issue(i);
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  BENCH_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < calls; ++i) issue(i);
  BENCH_CUDA(cudaStreamEndCapture(stream, &graph));
  BENCH_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b;
  BENCH_CUDA(cudaEventCreate(&a));
  BENCH_CUDA(cudaEventCreate(&b));
  BENCH_CUDA(cudaGraphLaunch(exec, stream));
  double best = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    BENCH_CUDA(cudaEventRecord(a, stream));
    BENCH_CUDA(cudaGraphLaunch(exec, stream));
    BENCH_CUDA(cudaEventRecord(b, stream));
    BENCH_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, 1000.0 * ms / calls);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  BENCH_CUDA(cudaGraphExecDestroy(exec));
  BENCH_CUDA(cudaGraphDestroy(graph));
  return best;
}

// The route's buffers at the largest shape: two dense W (the double buffer; the first is also
// lever C's slice buffer), the INT8 activations (room for lever A's three planes) and their
// scales, x and y.
struct Route {
  DeviceBytes w[2], a, sa, sq, x, y, svh;
  size_t w_bytes = 0;
  int max_tokens = 0;
  Route(int max_tokens_) : max_tokens(max_tokens_) {
    w_bytes = 17408ull * 5120;
    w[0] = DeviceBytes(w_bytes);
    w[1] = DeviceBytes(w_bytes);
    a = DeviceBytes(static_cast<size_t>(max_tokens) * 17408);
    sa = DeviceBytes(4ull * max_tokens * kA8MaxShared);
    sq = DeviceBytes(4ull * max_tokens * kA8MaxShared);
    x = DeviceBytes(2ull * max_tokens * 17408);
    y = DeviceBytes(2ull * max_tokens * 17408);
    svh = DeviceBytes(2ull * 17408);
    std::vector<uint16_t> s(17408, moe_test::f32_to_f16(0.03f));
    trellis_test::upload(svh, s);
    const std::vector<uint16_t> xh = standin_x(max_tokens, 17408, 5150);
    trellis_test::upload(x, xh);
    BENCH_CUDA(cudaMemset(w[0].p, 0, w_bytes));
    BENCH_CUDA(cudaMemset(w[1].p, 0, w_bytes));
  }
  // Lever A's targets: plane i of the activations, scales and sums.
  A8Shared shared(int tokens, int in, int count) {
    A8Shared g{};
    g.count = count;
    for (int i = 0; i < count; ++i)
      g.m[i] = A8Target{svh.as<__half>(), a.as<int8_t>() + static_cast<size_t>(i) * tokens * in, sa.as<float>() + static_cast<size_t>(i) * tokens,
                        sq.as<int32_t>() + static_cast<size_t>(i) * tokens};
    return g;
  }
};

double time_decode(const Pool &pool, int8_t *const (&w)[2], cudaStream_t s) {
  return time_us([&](int i) { launch_decode(pool.w[i % pool.w.size()], w[i & 1], s); }, 8, s);
}

double time_a8(Route &r, const Shape &sh, int tokens, cudaStream_t s) {
  return time_us([&](int) { launch_a8(r.x.p, r.svh.p, tokens, sh.in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), s); }, 8, s);
}

double time_a8_shared(Route &r, int in, int count, int tokens, cudaStream_t s) {
  const A8Shared g = r.shared(tokens, in, count);
  return time_us([&](int) { launch_a8_shared(r.x.p, tokens, in, g, s); }, 8, s);
}

// Lever A's group of a body shape: the matrices one shared A8 serves (1 outside the groups).
int shared_group(const Shape &sh) {
  const std::string n = sh.name;
  if (n == "gdn qkv" || n == "gdn z" || n == "mlp gate+up") return 2;
  if (n == "gqa q" || n == "gqa k" || n == "gqa v" || n == "gqa k, v") return 3;
  return 1;
}

std::map<std::string, double> g_a8_us;  // per (shape, tokens, A8 mode)

// The A8 a call of the shape costs: per matrix, or the shape's share of its group's one launch
// (gate+up is the whole group's launch: both matrices).
double a8_cost(Route &r, const Shape &sh, int tokens, const Lever &cfg, cudaStream_t s) {
  const int per = sh.swiglu ? 2 : 1;
  const int group = shared_group(sh);
  const bool shared = cfg.a8_shared && group > 1;
  const std::string k = std::string(sh.name) + "/" + std::to_string(tokens) + (shared ? "/shared" : "/per-matrix");
  auto it = g_a8_us.find(k);
  if (it != g_a8_us.end()) return it->second;
  double us;
  if (!shared) {
    us = time_a8(r, sh, tokens, s) * per;
  } else {
    const double t = time_a8_shared(r, sh.in, group, tokens, s);
    us = sh.swiglu ? t : t / group;
  }
  g_a8_us[k] = us;
  return us;
}

GemmCall gemm_call(Route &r, const Shape &sh, int tokens, int wbuf, int bm, int raster) {
  GemmCall c{r.a.as<int8_t>(), r.w[wbuf].as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), r.svh.p, tokens,
             static_cast<int>(sh.in), static_cast<int>(sh.out), kStore, r.y.p, static_cast<int>(sh.out)};
  c.bm = bm;
  c.raster = raster;
  return c;
}

double time_gemm(Route &r, const Shape &sh, int tokens, int bm, int raster, cudaStream_t s) {
  return time_us([&](int i) { launch_gemm(gemm_call(r, sh, tokens, i & 1, bm, raster), s); }, 6, s);
}

double time_gemm_persistent(Route &r, const Shape &sh, int tokens, int bm, int raster, int sk, cudaStream_t s) {
  return time_us([&](int i) { launch_gemm_persistent(gemm_call(r, sh, tokens, i & 1, bm, raster), sk, s); }, 6, s);
}

struct GemmChoice {
  double us = 1e30;
  int bm = 256, raster = 0, sk = 0;
};

// The GEMM's tile height and raster order, the faster of the four on this card for this shape.
GemmChoice best_gemm(Route &r, const Shape &sh, int tokens, cudaStream_t s) {
  GemmChoice best;
  for (const int bm : {256, 128})
    for (const int raster : {0, 1}) {
      const double us = time_gemm(r, sh, tokens, bm, raster, s);
      if (us < best.us) best = GemmChoice{us, bm, raster, 0};
    }
  return best;
}

// Lever B: the tile height, raster order and stream-K split, the fastest of the eight.
GemmChoice best_gemm_persistent(Route &r, const Shape &sh, int tokens, cudaStream_t s) {
  GemmChoice best;
  for (const int bm : {256, 128})
    for (const int raster : {0, 1})
      for (const int sk : {0, 1}) {
        const double us = time_gemm_persistent(r, sh, tokens, bm, raster, sk, s);
        if (us < best.us) best = GemmChoice{us, bm, raster, sk};
      }
  return best;
}

std::map<std::string, GemmChoice> g_choice;   // per (shape, tokens): the grid launch
std::map<std::string, GemmChoice> g_pchoice;  // per (shape, tokens): the persistent launch

std::string key(const Shape &s, int tokens) { return std::string(s.name) + "/" + std::to_string(tokens); }

GemmChoice gemm_choice(Route &r, const Shape &sh, int tokens, bool persistent, cudaStream_t s) {
  std::map<std::string, GemmChoice> &m = persistent ? g_pchoice : g_choice;
  auto it = m.find(key(sh, tokens));
  if (it != m.end()) return it->second;
  const GemmChoice c = persistent ? best_gemm_persistent(r, sh, tokens, s) : best_gemm(r, sh, tokens, s);
  m[key(sh, tokens)] = c;
  return c;
}

std::string choice_label(const GemmChoice &c, bool persistent) {
  std::string s = "BM " + std::to_string(c.bm) + (c.raster == 0 ? ", tokens fastest" : ", outputs fastest");
  if (persistent) s += c.sk == 0 ? ", stream-K tail" : ", stream-K tail + 1 wave";
  return s;
}

// Graph-replayed device microseconds per call of `seq` (one call = all its launches): `inner`
// calls captured once, replayed best of g_reps. Lever C's many small launches are timed this way,
// so the host's launch rate does not count.
template <typename F> double time_graph_us(F &&seq, int inner, cudaStream_t s) {
  for (int i = 0; i < 2; ++i) seq(i);
  BENCH_CUDA(cudaStreamSynchronize(s));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  BENCH_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < inner; ++i) seq(i);
  BENCH_CUDA(cudaStreamEndCapture(s, &graph));
  BENCH_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t a, b;
  BENCH_CUDA(cudaEventCreate(&a));
  BENCH_CUDA(cudaEventCreate(&b));
  BENCH_CUDA(cudaGraphLaunch(exec, s));
  double best = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    BENCH_CUDA(cudaEventRecord(a, s));
    BENCH_CUDA(cudaGraphLaunch(exec, s));
    BENCH_CUDA(cudaEventRecord(b, s));
    BENCH_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, 1000.0 * ms / inner);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  BENCH_CUDA(cudaGraphExecDestroy(exec));
  BENCH_CUDA(cudaGraphDestroy(graph));
  return best;
}

// Lever C's L2 access-policy window: the slice buffer marked persisting, where the card has a
// persisting L2 (cudaDevAttrMaxPersistingL2CacheSize > 0).
int g_persist_max = 0;
bool g_persist = false;

void set_window(cudaStream_t s, void *p, size_t bytes, bool on) {
  if (g_persist_max <= 0) return;
  cudaStreamAttrValue v{};
  if (on) {
    const size_t win = std::min(bytes, static_cast<size_t>(g_persist_max));
    v.accessPolicyWindow.base_ptr = p;
    v.accessPolicyWindow.num_bytes = win;
    v.accessPolicyWindow.hitRatio = 1.0f;
    v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  } else {
    v.accessPolicyWindow.num_bytes = 0;
  }
  BENCH_CUDA(cudaStreamSetAttribute(s, cudaStreamAttributeAccessPolicyWindow, &v));
  if (!on) BENCH_CUDA(cudaCtxResetPersistingL2Cache());
}

// Lever C against the whole decode, one call of the shape (both matrices for gate+up) each, both
// graph-replayed and timed alternately three times (best of each), cycling the pool's copies so the
// trellis misses the L2: whole = the decode of the whole W, then the GEMM; slice = every slice
// decoded and multiplied in turn. Their difference is lever C's effect on decode + GEMM: a
// difference of two totals measured side by side, not of two measured minutes apart.
struct SliceTimes {
  double whole = 1e30, slice = 1e30;
};

SliceTimes time_slice(Route &r, const Pool &pool, const Shape &sh, int tokens, const Lever &cfg, const GemmChoice &gc, cudaStream_t s) {
  const int per = sh.swiglu ? 2 : 1;
  const RouteVariant rv{gc.raster, gc.sk};
  const size_t slice = static_cast<size_t>(slice_even(static_cast<int>(sh.in), static_cast<int>(sh.out) / kBlockN)) * kBlockN * sh.in;
  SliceTimes t;
  for (int round = 0; round < 3; ++round) {
    t.whole = std::min(t.whole, time_graph_us([&](int i) {
      for (int m = 0; m < per; ++m) {
        launch_decode(pool.w[(i * per + m) % pool.w.size()], r.w[0].as<int8_t>(), s);
        gemm_sched(cfg, rv, gemm_call(r, sh, tokens, 0, gc.bm, gc.raster), s);
      }
    }, 4, s));
    set_window(s, r.w[0].p, slice, g_persist);
    t.slice = std::min(t.slice, time_graph_us([&](int i) {
      for (int m = 0; m < per; ++m) {
        const ignis_trellis_weight &tw = pool.w[(i * per + m) % pool.w.size()];
        run_route_gemm(cfg, rv, gemm_call(r, sh, tokens, 1, gc.bm, gc.raster), tw, r.w[0].as<int8_t>(), s);
      }
    }, 4, s));
    set_window(s, nullptr, 0, false);
  }
  return t;
}

struct ChunkTotals {
  double decode = 0, a8 = 0, gemm = 0, fused = 0;
  double total() const { return decode + a8 + gemm; }
};

std::map<std::string, double> g_fused_us;  // per (shape, K, tokens)
std::map<std::string, double> g_dec_us;    // per (shape, K): the whole decode
std::map<std::string, SliceTimes> g_slice_us;  // per (shape, K, tokens, scheduler): lever C against the whole decode

std::string kkey(const Shape &s, int tokens) { return key(s, tokens) + "/K" + std::to_string(s.k2); }

// One shape under a configuration: decode, A8, GEMM per call (us), and today's fused kernel. With
// --decode slice, the decode is the whole decode's plus what slicing changes in decode + GEMM.
struct ShapeTimes {
  double dec = 0, a8 = 0, gemm = 0, fused = 0;
  GemmChoice gc;
};

ShapeTimes shape_times(Route &r, const Shape &sh, int tokens, const Lever &cfg, cudaStream_t s) {
  const int per = sh.swiglu ? 2 : 1;
  int8_t *const wb[2] = {r.w[0].as<int8_t>(), r.w[1].as<int8_t>()};
  ShapeTimes st;
  st.gc = gemm_choice(r, sh, tokens, cfg.persistent, s);
  st.gemm = st.gc.us * per;
  st.a8 = a8_cost(r, sh, tokens, cfg, s);
  const std::string fk = kkey(sh, tokens), dk = kkey(sh, 0);
  const bool need_fused = g_fused && g_fused_us.find(fk) == g_fused_us.end();
  const bool need_dec = g_dec_us.find(dk) == g_dec_us.end();
  const std::string sk = fk + (cfg.persistent ? "/persistent" : "/grid");
  const bool need_slice = cfg.slice && g_slice_us.find(sk) == g_slice_us.end();
  Pool pool;
  if (need_fused || need_dec || need_slice) make_pool(pool, sh, per);
  if (need_dec) g_dec_us[dk] = time_decode(pool, wb, s) * per;
  if (need_fused) g_fused_us[fk] = time_fused(pool, sh, tokens, s);
  if (need_slice) g_slice_us[sk] = time_slice(r, pool, sh, tokens, cfg, st.gc, s);
  st.fused = g_fused ? g_fused_us[fk] : std::nan("");
  st.dec = cfg.slice ? g_dec_us[dk] + g_slice_us[sk].slice - g_slice_us[sk].whole : g_dec_us[dk];
  return st;
}

void speed(int tokens, Route &r, cudaStream_t s, const Lever &cfg, ChunkTotals &chunk) {
  // The clocks up first: about half a second of the largest GEMM.
  for (int i = 0; i < 400 * 2048 / tokens; ++i) launch_gemm(gemm_call(r, kBody[7], tokens, i & 1, 256, 0), s);
  BENCH_CUDA(cudaStreamSynchronize(s));
  const bool base = !cfg.a8_shared && !cfg.persistent && !cfg.slice;
  if (base) {
    std::printf("\n== T = %d: per call (us), the dense route against today's fused trellis prefill kernel ==\n", tokens);
  } else {
    std::printf("\n== T = %d, levers %s: per call (us), the dense route against today's fused trellis prefill kernel ==\n", tokens,
                cfg.name.c_str());
  }
  std::printf("%-12s %6s %6s %3s | %8s %7s %8s %7s %6s %9s | %9s %7s | %6s\n", "matrix", "in", "out", "K", "decode", "A8",
              "GEMM", "TOPS", "%s8", "dense", "fused", "TFLOPS", "f/d");
  for (const Shape &sh : kBody) {
    const int per = sh.swiglu ? 2 : 1;
    const ShapeTimes st = shape_times(r, sh, tokens, cfg, s);
    const double ops = 2.0 * tokens * sh.in * sh.out * per;
    const double tops = ops / (st.gemm * 1e-6) / 1e12;
    const double dense = st.dec + st.a8 + st.gemm;
    std::printf("%-12s %6u %6u %3.1f | %8.1f %7.1f %8.1f %7.1f %5.1f%% %9.1f | %9.1f %7.1f | %6.2f   (%s)\n", sh.name, sh.in, sh.out,
                sh.k2 / 2.0, st.dec, st.a8, st.gemm, tops, 100.0 * tops / g_s8_peak, dense, st.fused, ops / (st.fused * 1e-6) / 1e12,
                st.fused / dense, choice_label(st.gc, cfg.persistent).c_str());
  }
  // The chunk: the body at the quant's K mix (decode and the fused kernel depend on K; A8 and the
  // GEMM do not).
  chunk = ChunkTotals{};
  double ops_total = 0.0;
  for (const MixEntry &e : kMix) {
    const Shape &sh = e.shape;
    const int per = sh.swiglu ? 2 : 1;
    const ShapeTimes st = shape_times(r, sh, tokens, cfg, s);
    chunk.decode += e.calls * st.dec;
    chunk.a8 += e.calls * st.a8;
    chunk.gemm += e.calls * st.gemm;
    chunk.fused += e.calls * st.fused;
    ops_total += e.calls * 2.0 * tokens * sh.in * sh.out * per;
  }
  std::printf("chunk of %d tokens, the 64-layer body at the quant's K mix (ms): decode %.2f + A8 %.2f + GEMM %.2f = dense %.2f;"
              " fused %.2f; fused / dense %.2f; GEMM %.1f TOPS (%.1f%% of s8), fused %.1f TFLOPS%s\n",
              tokens, chunk.decode / 1e3, chunk.a8 / 1e3, chunk.gemm / 1e3, chunk.total() / 1e3, chunk.fused / 1e3,
              chunk.fused / chunk.total(), ops_total / (chunk.gemm * 1e-6) / 1e12,
              100.0 * ops_total / (chunk.gemm * 1e-6) / 1e12 / g_s8_peak, ops_total / (chunk.fused * 1e-6) / 1e12,
              base ? "" : (" [" + cfg.name + "]").c_str());
}

// The ablation's GEMM and A8 choices measured up front, each shape's grid and persistent searches
// (and per-matrix and shared A8) back to back, so that a slow drift of the clocks over the run does
// not land on one configuration.
void prepare_choices(int tokens, Route &r, cudaStream_t s) {
  for (int i = 0; i < 400 * 2048 / tokens; ++i) launch_gemm(gemm_call(r, kBody[7], tokens, i & 1, 256, 0), s);
  BENCH_CUDA(cudaStreamSynchronize(s));
  const Lever per_matrix = Lever{}, shared = Lever{"A", true, false, false};
  auto one = [&](const Shape &sh) {
    if (g_choice.count(key(sh, tokens)) != 0) return;
    GemmChoice g = gemm_choice(r, sh, tokens, false, s), p = gemm_choice(r, sh, tokens, true, s);
    for (int rep = 0; rep < 3; ++rep) {
      g.us = std::min(g.us, time_gemm(r, sh, tokens, g.bm, g.raster, s));
      p.us = std::min(p.us, time_gemm_persistent(r, sh, tokens, p.bm, p.raster, p.sk, s));
    }
    g_choice[key(sh, tokens)] = g;
    g_pchoice[key(sh, tokens)] = p;
    a8_cost(r, sh, tokens, per_matrix, s);
    a8_cost(r, sh, tokens, shared, s);
  };
  for (const Shape &sh : kBody) one(sh);
  for (const MixEntry &e : kMix) one(e.shape);
}

// Lever B's per-shape table: the grid launch against the persistent one, each at its best choice,
// timed alternately three times (best of each).
void lever_b_table(int tokens, Route &r, cudaStream_t s) {
  std::printf("\n== lever B at T = %d: the GEMM per call (us), one CTA per tile against the persistent stream-K launch (%d SMs) ==\n",
              tokens, g_sms);
  std::printf("%-12s %6s %6s %6s | %8s %6s %-28s | %8s %6s %-44s | %6s\n", "matrix", "in", "out", "tiles", "grid", "%s8", "",
              "persist", "%s8", "", "g/p");
  for (const Shape &sh : kBody) {
    GemmChoice g = gemm_choice(r, sh, tokens, false, s), p = gemm_choice(r, sh, tokens, true, s);
    for (int rep = 0; rep < 3; ++rep) {
      g.us = std::min(g.us, time_gemm(r, sh, tokens, g.bm, g.raster, s));
      p.us = std::min(p.us, time_gemm_persistent(r, sh, tokens, p.bm, p.raster, p.sk, s));
    }
    const double ops = 2.0 * tokens * sh.in * sh.out;
    const int tiles = ((tokens + p.bm - 1) / p.bm) * static_cast<int>(sh.out / kBN);
    std::printf("%-12s %6u %6u %6d | %8.1f %5.1f%% %-28s | %8.1f %5.1f%% %-44s | %6.3f\n", sh.name, sh.in, sh.out, tiles, g.us,
                100.0 * ops / (g.us * 1e-6) / 1e12 / g_s8_peak, choice_label(g, false).c_str(), p.us,
                100.0 * ops / (p.us * 1e-6) / 1e12 / g_s8_peak, choice_label(p, true).c_str(), g.us / p.us);
  }
}

// One layer's matrices in order, the next one's decode on a second stream through the double
// buffer, against the same sequence on one stream.
struct OverlapTimes {
  double serial = 0, naive = 0, coresident = 0, sliced = 0;
};

// serial: decode, A8, GEMM per matrix on one stream. naive: the next matrix's decode (4 warps a CTA)
// on a second stream of the same priority. coresident: that decode at 2 warps a CTA, small enough to
// sit beside a GEMM CTA, on the lowest-priority stream, the A8s and GEMMs on the highest. Under
// lever A the first matrix of a shared input runs the group's one A8 and the others none; under
// lever B the GEMMs are the persistent launch.
void overlap(int tokens, Route &r, cudaStream_t s1, cudaStream_t s2, cudaStream_t hi, cudaStream_t lo, const Lever &cfg,
             OverlapTimes *gdn_t, OverlapTimes *gqa_t) {
  const Shape gdn[] = {{"gdn qkv", 5120, 10240, 6, 0}, {"gdn z", 5120, 6144, 6, 0}, {"gdn out", 6144, 5120, 6, 0},
                       {"mlp gate", 5120, 17408, 6, 0}, {"mlp up", 5120, 17408, 6, 0}, {"mlp down", 17408, 5120, 6, 0}};
  const int gdn_a8[] = {2, 0, 1, 2, 0, 1};
  const Shape gqa[] = {{"gqa q", 5120, 12288, 6, 0}, {"gqa k", 5120, 1024, 8, 0}, {"gqa v", 5120, 1024, 8, 0},
                       {"gqa o", 6144, 5120, 6, 0}, {"mlp gate", 5120, 17408, 6, 0}, {"mlp up", 5120, 17408, 6, 0},
                       {"mlp down", 17408, 5120, 6, 0}};
  const int gqa_a8[] = {3, 0, 0, 1, 2, 0, 1};
  auto run = [&](const Shape *seq, const int *a8n, int n, OverlapTimes *out) {
    std::vector<Pool> pools(n);
    for (int i = 0; i < n; ++i) make_pool(pools[i], seq[i], 1);
    std::vector<GemmCall> calls;
    std::vector<GemmChoice> choices;
    for (int i = 0; i < n; ++i) {
      const Shape body = std::string(seq[i].name).rfind("mlp gate", 0) == 0 || std::string(seq[i].name).rfind("mlp up", 0) == 0
                             ? Shape{"mlp gate+up", 5120, 17408, 6, 1}
                             : seq[i];
      const GemmChoice gc = gemm_choice(r, body, tokens, cfg.persistent, s1);
      choices.push_back(gc);
      calls.push_back(gemm_call(r, seq[i], tokens, i & 1, gc.bm, gc.raster));
    }
    auto a8 = [&](int i, cudaStream_t s) {
      if (!cfg.a8_shared || a8n[i] == 1) {
        launch_a8(r.x.p, r.svh.p, tokens, seq[i].in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), s);
      } else if (a8n[i] > 1) {
        launch_a8_shared(r.x.p, tokens, seq[i].in, r.shared(tokens, seq[i].in, a8n[i]), s);
      }
    };
    auto gemm = [&](int i, cudaStream_t s, bool cap) { gemm_sched(cfg, RouteVariant{choices[i].raster, choices[i].sk, cap}, calls[i], s); };
    std::vector<cudaEvent_t> dec(n), gem(n);
    for (int i = 0; i < n; ++i) {
      BENCH_CUDA(cudaEventCreateWithFlags(&dec[i], cudaEventDisableTiming));
      BENCH_CUDA(cudaEventCreateWithFlags(&gem[i], cudaEventDisableTiming));
    }
    cudaEvent_t a, b, fork;
    BENCH_CUDA(cudaEventCreate(&a));
    BENCH_CUDA(cudaEventCreate(&b));
    BENCH_CUDA(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    auto serial = [&](int rep) {
      for (int i = 0; i < n; ++i) {
        launch_decode(pools[i].w[rep % pools[i].w.size()], r.w[i & 1].as<int8_t>(), s1);
        a8(i, s1);
        gemm(i, s1, false);
      }
    };
    auto overlapped = [&](int rep, cudaStream_t sc, cudaStream_t sd, int warps, bool cap) {
      BENCH_CUDA(cudaEventRecord(fork, sc));
      BENCH_CUDA(cudaStreamWaitEvent(sd, fork, 0));
      launch_decode(pools[0].w[rep % pools[0].w.size()], r.w[0].as<int8_t>(), sd, warps);
      BENCH_CUDA(cudaEventRecord(dec[0], sd));
      for (int i = 0; i < n; ++i) {
        a8(i, sc);
        BENCH_CUDA(cudaStreamWaitEvent(sc, dec[i], 0));
        gemm(i, sc, cap);
        BENCH_CUDA(cudaEventRecord(gem[i], sc));
        if (i + 1 < n) {
          if (i >= 1) BENCH_CUDA(cudaStreamWaitEvent(sd, gem[i - 1], 0));  // buffer (i + 1) % 2 is free
          launch_decode(pools[i + 1].w[rep % pools[i + 1].w.size()], r.w[(i + 1) & 1].as<int8_t>(), sd, warps);
          BENCH_CUDA(cudaEventRecord(dec[i + 1], sd));
        }
      }
    };
    auto measure = [&](cudaStream_t sc, auto &&f) {
      f(0);
      BENCH_CUDA(cudaDeviceSynchronize());
      double best = 1e30;
      for (int rep = 0; rep < g_reps; ++rep) {
        BENCH_CUDA(cudaEventRecord(a, sc));
        f(rep + 1);
        BENCH_CUDA(cudaEventRecord(b, sc));
        BENCH_CUDA(cudaEventSynchronize(b));
        float ms = 0.0f;
        BENCH_CUDA(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, 1000.0 * ms);
      }
      return best;
    };
    out->serial = measure(s1, serial);
    out->naive = measure(s1, [&](int rep) { overlapped(rep, s1, s2, kDecWarps, false); });
    out->coresident = measure(hi, [&](int rep) { overlapped(rep, hi, lo, 2, true); });
    // Lever C under the co-resident overlap: every matrix's slices in one line, at half lever C's
    // target so that two slice buffers fit where one did; slice k + 1's decode (2 warps a CTA, the
    // lowest priority) into buffer (k + 1) % 2 while slice k's GEMM runs on the highest.
    {
      struct SliceItem {
        int m, nb0, nbs;
      };
      const size_t keep = g_slice_bytes;
      g_slice_bytes = keep / 2;
      std::vector<SliceItem> items;
      for (int i = 0; i < n; ++i) {
        const int nb = static_cast<int>(seq[i].out) / kBlockN, per = slice_even(static_cast<int>(seq[i].in), nb);
        for (int nb0 = 0; nb0 < nb; nb0 += per) items.push_back(SliceItem{i, nb0, std::min(per, nb - nb0)});
      }
      g_slice_bytes = keep;
      std::vector<cudaEvent_t> sdec(items.size()), sgem(items.size());
      for (size_t k = 0; k < items.size(); ++k) {
        BENCH_CUDA(cudaEventCreateWithFlags(&sdec[k], cudaEventDisableTiming));
        BENCH_CUDA(cudaEventCreateWithFlags(&sgem[k], cudaEventDisableTiming));
      }
      auto sliced = [&](int rep) {
        auto decode_item = [&](size_t k) {
          const SliceItem &it = items[k];
          launch_decode(trellis_slice(pools[it.m].w[rep % pools[it.m].w.size()], it.nb0, it.nbs), r.w[k & 1].as<int8_t>(), lo, 2);
          BENCH_CUDA(cudaEventRecord(sdec[k], lo));
        };
        BENCH_CUDA(cudaEventRecord(fork, hi));
        BENCH_CUDA(cudaStreamWaitEvent(lo, fork, 0));
        decode_item(0);
        for (size_t k = 0; k < items.size(); ++k) {
          const SliceItem &it = items[k];
          if (it.nb0 == 0) a8(it.m, hi);
          BENCH_CUDA(cudaStreamWaitEvent(hi, sdec[k], 0));
          gemm_sched(cfg, RouteVariant{choices[it.m].raster, choices[it.m].sk, true}, slice_call(calls[it.m], it.nb0, it.nbs, r.w[k & 1].as<int8_t>()),
                     hi);
          BENCH_CUDA(cudaEventRecord(sgem[k], hi));
          if (k + 1 < items.size()) {
            if (k >= 1) BENCH_CUDA(cudaStreamWaitEvent(lo, sgem[k - 1], 0));  // buffer (k + 1) % 2 is free
            decode_item(k + 1);
          }
        }
      };
      out->sliced = measure(hi, sliced);
      for (size_t k = 0; k < items.size(); ++k) {
        cudaEventDestroy(sdec[k]);
        cudaEventDestroy(sgem[k]);
      }
    }
    for (int i = 0; i < n; ++i) {
      cudaEventDestroy(dec[i]);
      cudaEventDestroy(gem[i]);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaEventDestroy(fork);
  };
  run(gdn, gdn_a8, 6, gdn_t);
  run(gqa, gqa_a8, 7, gqa_t);
}

void kernel_report(const char *name, const void *f, int threads, int smem) {
  cudaFuncAttributes at{};
  BENCH_CUDA(cudaFuncGetAttributes(&at, f));
  int ctas = 0;
  BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, f, threads, smem));
  std::printf("  %-34s %3d regs, %5zu B local (spills), %6d B dynamic + %5zu B static smem, %d threads, %d CTA/SM\n", name,
              at.numRegs, at.localSizeBytes, smem, at.sharedSizeBytes, threads, ctas);
}

std::vector<int> parse_list(const std::string &s) {
  std::vector<int> v;
  size_t pos = 0;
  while (pos <= s.size()) {
    const size_t comma = s.find(',', pos);
    v.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return v;
}

Lever make_lever(bool a8_shared, bool persistent, bool slice) {
  Lever l;
  l.a8_shared = a8_shared;
  l.persistent = persistent;
  l.slice = slice;
  l.name = lever_name(l);
  return l;
}

// The ablation's rows: phase 2a's route, each lever alone (C on both schedulers: its per-slice
// launches are the persistent launch's case), and the combinations.
std::vector<Lever> ablation_levers() {
  return {make_lever(false, false, false), make_lever(true, false, false), make_lever(false, true, false), make_lever(false, false, true),
          make_lever(false, true, true),   make_lever(true, true, false),  make_lever(true, true, true)};
}

}  // namespace

int main(int argc, char **argv) {
  std::string artifact;
  std::vector<int> tokens = {2048, 4096};
  int check_tokens = 2048;
  bool do_check = true, do_speed = true, do_peaks = true, diag = false, ablation = false;
  Lever cfg;
  double slice_mb = 0.0;
  auto usage = [&]() {
    std::fprintf(stderr,
                 "usage: %s [--artifact PATH] [--tokens 2048,4096] [--check-tokens N] [--reps N] [--no-check] [--no-speed] [--no-peaks]"
                 " [--diag] [--no-fused] [--a8 per-matrix|shared] [--sched grid|persistent] [--decode whole|slice] [--slice-mb MB]"
                 " [--l2-persist] [--ablation]\n",
                 argv[0]);
    return 2;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--artifact" && more) {
      artifact = argv[++i];
    } else if (a == "--tokens" && more) {
      tokens = parse_list(argv[++i]);
    } else if (a == "--check-tokens" && more) {
      check_tokens = std::atoi(argv[++i]);
    } else if (a == "--reps" && more) {
      g_reps = std::atoi(argv[++i]);
    } else if (a == "--no-check") {
      do_check = false;
    } else if (a == "--no-speed") {
      do_speed = false;
    } else if (a == "--no-peaks") {
      do_peaks = false;
    } else if (a == "--diag") {
      diag = true;
    } else if (a == "--no-fused") {
      g_fused = false;
    } else if (a == "--a8" && more) {
      const std::string v = argv[++i];
      if (v != "per-matrix" && v != "shared") return usage();
      cfg.a8_shared = v == "shared";
    } else if (a == "--sched" && more) {
      const std::string v = argv[++i];
      if (v != "grid" && v != "persistent") return usage();
      cfg.persistent = v == "persistent";
    } else if (a == "--decode" && more) {
      const std::string v = argv[++i];
      if (v != "whole" && v != "slice") return usage();
      cfg.slice = v == "slice";
    } else if (a == "--slice-mb" && more) {
      slice_mb = std::atof(argv[++i]);
    } else if (a == "--l2-persist") {
      g_persist = true;
    } else if (a == "--ablation") {
      ablation = true;
    } else {
      return usage();
    }
  }
  cfg.name = lever_name(cfg);
  std::fesetround(FE_TONEAREST);
  int runtime = 0, driver = 0;
  BENCH_CUDA(cudaRuntimeGetVersion(&runtime));
  BENCH_CUDA(cudaDriverGetVersion(&driver));
  std::printf("sm8x dense prefill bench (prototype, phases 2a and 2c): CUDA runtime %d.%d, driver supports %d.%d\n", runtime / 1000,
              (runtime % 1000) / 10, driver / 1000, (driver % 1000) / 10);
  if (driver < runtime) {
    std::fprintf(stderr, "FATAL: the driver supports CUDA %d.%d, the bench needs %d.%d or newer\n", driver / 1000,
                 (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
    return 1;
  }
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  g_grid_sms = g_sms;
  BENCH_CUDA(cudaDeviceGetAttribute(&g_persist_max, cudaDevAttrMaxPersistingL2CacheSize, 0));
  g_slice_bytes = slice_mb > 0.0 ? static_cast<size_t>(slice_mb * 1048576.0) : static_cast<size_t>(prop.l2CacheSize) / 2;
  std::printf("device: sm_%d%d, %d SMs, %zu B shared memory per block (opt-in), %zu B per SM, %d MB L2\n", prop.major, prop.minor,
              prop.multiProcessorCount, prop.sharedMemPerBlockOptin, prop.sharedMemPerMultiprocessor, prop.l2CacheSize >> 20);
  std::printf("levers: %s%s; lever C's slices: %.1f MB target (%s), %d blocks of 128 outputs at in 5120, %d at 6144, %d at 17408;"
              " persisting L2: %d B max%s\n",
              ablation ? "the ablation, " : "", cfg.name.c_str(), g_slice_bytes / 1048576.0, slice_mb > 0.0 ? "--slice-mb" : "half the L2",
              slice_blocks(5120), slice_blocks(6144), slice_blocks(17408), g_persist_max,
              g_persist ? (g_persist_max > 0 ? ", the window on (--l2-persist)" : ", --l2-persist has no effect") : "");
  if (g_persist && g_persist_max > 0) BENCH_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(g_persist_max)));
  prepare_gemm();
  ensure_pwork();
  if (g_fused) {
    prepare_fused("off");  // a failure leaves g_fused false and is printed
  } else {
    std::printf("fused trellis linear: not run (--no-fused): the dense route runs alone\n");
  }
  std::printf("kernels:\n");
  kernel_report("sm8x_dense_gemm<256>", reinterpret_cast<const void *>(sm8x_dense_gemm<256>), kGemmThreads, Cfg<256>::kSmem);
  kernel_report("sm8x_dense_gemm<128>", reinterpret_cast<const void *>(sm8x_dense_gemm<128>), kGemmThreads, Cfg<128>::kSmem);
  kernel_report("sm8x_dense_gemm_persistent<256>", reinterpret_cast<const void *>(sm8x_dense_gemm_persistent<256, kRegsRoute>), kGemmThreads,
                Cfg<256>::kSmem);
  kernel_report("sm8x_dense_gemm_persistent<128>", reinterpret_cast<const void *>(sm8x_dense_gemm_persistent<128, kRegsRoute>), kGemmThreads,
                Cfg<128>::kSmem);
  kernel_report("sm8x_dense_gemm_persistent<256, 224>", reinterpret_cast<const void *>(sm8x_dense_gemm_persistent<256, kRegsCoresident>),
                kGemmThreads, Cfg<256>::kSmem);
  kernel_report("sm8x_dense_gemm_persistent<128, 224>", reinterpret_cast<const void *>(sm8x_dense_gemm_persistent<128, kRegsCoresident>),
                kGemmThreads, Cfg<128>::kSmem);
  kernel_report("sm8x_dense_decode<6, 4>", reinterpret_cast<const void *>(sm8x_dense_decode<6, 4>), 128, 0);
  kernel_report("sm8x_dense_decode<10, 4>", reinterpret_cast<const void *>(sm8x_dense_decode<10, 4>), 128, 0);
  kernel_report("sm8x_dense_decode<6, 2>", reinterpret_cast<const void *>(sm8x_dense_decode<6, 2>), 64, 0);
  kernel_report("sm8x_dense_a8", reinterpret_cast<const void *>(sm8x_dense_a8), 256, 0);
  kernel_report("sm8x_dense_a8_shared<5>", reinterpret_cast<const void *>(sm8x_dense_a8_shared<kA8SharedBlocks>), 256, 0);

  if (do_peaks) {
    const meter::Rate s8 = meter::peak<meter::kS8>(g_sms, 4096, 10, 300);
    const meter::Rate f32 = meter::peak<meter::kF16F32>(g_sms, 4096, 10, 300);
    const meter::Rate f16 = meter::peak<meter::kF16F16>(g_sms, 4096, 10, 300);
    g_s8_peak = s8.tops;
    size_t l2_set = 0;
    const double l2 = meter::l2_read_tbs(g_sms, static_cast<size_t>(prop.l2CacheSize), &l2_set);
    std::printf("L2-resident read bandwidth (%zu KiB, half the L2 or less, ld.global.cg, all SMs): %.2f TB/s\n", l2_set >> 10, l2);
    std::printf("register-loop peaks (%d blocks x 256 threads, 8 chains):\n", 4 * g_sms);
    std::printf("  %-22s %7.1f TOPS   at %5.0f MHz\n", meter::kPeakName[0], s8.tops, s8.mhz);
    std::printf("  %-22s %7.1f TFLOPS at %5.0f MHz\n", meter::kPeakName[1], f32.tops, f32.mhz);
    std::printf("  %-22s %7.1f TFLOPS at %5.0f MHz\n", meter::kPeakName[2], f16.tops, f16.mhz);
    std::printf("  ratios (clock-adjusted): s8 / f16-f32acc %.2f, s8 / f16-f16acc %.2f, f16-f16acc / f16-f32acc %.2f\n",
                (s8.tops / s8.mhz) / (f32.tops / f32.mhz), (s8.tops / s8.mhz) / (f16.tops / f16.mhz),
                (f16.tops / f16.mhz) / (f32.tops / f32.mhz));
  }
  if (g_s8_peak <= 0.0) g_s8_peak = 1.0;

  cudaStream_t s1, s2, hi, lo;
  BENCH_CUDA(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  BENCH_CUDA(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  int least = 0, greatest = 0;
  BENCH_CUDA(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  BENCH_CUDA(cudaStreamCreateWithPriority(&hi, cudaStreamNonBlocking, greatest));
  BENCH_CUDA(cudaStreamCreateWithPriority(&lo, cudaStreamNonBlocking, least));
  const std::vector<Lever> cfgs = ablation ? ablation_levers() : std::vector<Lever>{cfg};
  if (do_check) {
    if (artifact.empty()) {
      std::printf("check: skipped (no --artifact)\n");
    } else {
      Artifact art;
      if (!art.open(artifact)) {
        std::fprintf(stderr, "FATAL: %s is not a readable .ninfer v2 artifact\n", artifact.c_str());
        return 1;
      }
      const size_t slash = artifact.find_last_of("/\\");
      std::printf("check on %s, %d tokens of the activation stand-in%s:\n",
                  (slash == std::string::npos ? artifact : artifact.substr(slash + 1)).c_str(), check_tokens,
                  cfgs.size() > 1 ? ", every configuration of the ablation" : "");
      g_cfg_failed.assign(cfgs.size(), 0);
      const int before = g_failed;
      const char *names[] = {"text/layers/0/mlp/gate_proj", "text/layers/1/gdn/in_proj_qkv", "text/layers/3/attention/k_proj",
                             "text/layers/15/attention/v_proj", "text/layers/3/mlp/down"};
      for (const char *n : names) check_matrix(art, n, check_tokens, s1, cfgs);
      if (cfgs.size() > 1 || cfg.name != "phase 2a") {
        for (size_t i = 0; i < cfgs.size(); ++i)
          std::printf("check [%s]: %s (%d failed)\n", cfgs[i].name.c_str(), g_cfg_failed[i] == 0 ? "PASS" : "FAIL", g_cfg_failed[i]);
      }
      (void)before;
    }
  }
  if (diag) {
    Route r(*std::max_element(tokens.begin(), tokens.end()));
    std::printf("\n== GEMM diagnostic (BM 256, tokens fastest): full, no global loads, no loads and no ldmatrix ==\n");
    for (const int t : tokens)
      for (const Shape &sh : kBody) {
        double us[3];
        for (int d = 0; d < 3; ++d) {
          us[d] = time_us([&](int i) {
            GemmCall c = gemm_call(r, sh, t, i & 1, 256, 0);
            c.diag = d;
            launch_gemm(c, s1);
          }, 6, s1);
        }
        const double ops = 2.0 * t * sh.in * sh.out;
        std::printf("T %d %-12s %8.1f us %6.1f%% | %8.1f us %6.1f%% | %8.1f us %6.1f%% of s8\n", t, sh.name, us[0],
                    100.0 * ops / (us[0] * 1e-6) / 1e12 / g_s8_peak, us[1], 100.0 * ops / (us[1] * 1e-6) / 1e12 / g_s8_peak, us[2],
                    100.0 * ops / (us[2] * 1e-6) / 1e12 / g_s8_peak);
      }
    // Lever C's premise: the whole decode writes its W through to DRAM; one L2-sized slice rewritten
    // in place should not. The same trellis (gate_proj's shape, K 3, the pool missing the L2) decoded
    // into the whole W, and slice by slice into one slice buffer.
    std::printf("\n== decode diagnostic (mlp gate, K 3): the whole W, against each slice into one slice buffer ==\n");
    {
      const Shape sh{"mlp gate", 5120, 17408, 6, 0};
      Pool pool;
      make_pool(pool, sh, 1);
      const int nb = static_cast<int>(sh.out) / kBlockN;
      int8_t *const wb[2] = {r.w[0].as<int8_t>(), r.w[1].as<int8_t>()};
      const double whole = time_decode(pool, wb, s1);
      for (const size_t mb : {size_t(0), size_t(6), size_t(24), g_slice_bytes >> 20}) {
        const size_t keep = g_slice_bytes;
        if (mb != 0) g_slice_bytes = mb << 20;
        const int per = mb == 0 ? nb : slice_even(static_cast<int>(sh.in), nb);
        const double us = time_us([&](int i) {
          const ignis_trellis_weight &tw = pool.w[i % pool.w.size()];
          for (int nb0 = 0; nb0 < nb; nb0 += per) launch_decode(trellis_slice(tw, nb0, std::min(per, nb - nb0)), r.w[0].as<int8_t>(), s1);
        }, 4, s1);
        g_slice_bytes = keep;
        const double gb = (static_cast<double>(sh.in) * sh.out * (1.0 + sh.k2 / 16.0)) / 1e9;
        std::printf("  %-34s %8.1f us, %5.2f TB/s of W + trellis%s\n",
                    mb == 0 ? "whole W (one launch, one buffer)" : ("slices of " + std::to_string(per) + " blocks (" + std::to_string(mb) + " MB)").c_str(),
                    us, gb / (us * 1e-6) / 1e3, mb == 0 ? "" : " (one slice buffer)");
      }
      std::printf("  whole W alternating two buffers (the speed tables' decode): %.1f us\n", whole);
    }
    // Lever C taken apart: the whole route and the sliced one (the persistent launch, BM 256, tokens
    // fastest, stream-K tail + 1 wave), each in full, its decodes alone and its GEMMs alone, graphs
    // timed alternately three times (best of each). sliced - (decodes + GEMMs) is what the two lose
    // to each other when interleaved; GEMMs sliced - GEMM whole is the cost of cutting the GEMM.
    std::printf("\n== lever C taken apart (us per call; persistent GEMM): full, decodes alone, GEMMs alone ==\n");
    for (const int t : tokens) {
      const Shape shapes[] = {{"gdn qkv", 5120, 10240, 6, 0}, {"mlp gate", 5120, 17408, 6, 0}, {"mlp down", 17408, 5120, 6, 0}};
      for (const Shape &sh : shapes) {
        Pool pool;
        make_pool(pool, sh, 1);
        const Lever whole_cfg = make_lever(false, true, false), slice_cfg = make_lever(false, true, true);
        const RouteVariant rv{0, 1};
        double us[2][3] = {{1e30, 1e30, 1e30}, {1e30, 1e30, 1e30}};
        for (int round = 0; round < 3; ++round)
          for (int sl = 0; sl < 2; ++sl)
            for (int part = 0; part < 3; ++part) {
              us[sl][part] = std::min(us[sl][part], time_graph_us([&](int i) {
                const ignis_trellis_weight &tw = pool.w[i % pool.w.size()];
                if (sl == 1) {
                  run_route_gemm(slice_cfg, rv, gemm_call(r, sh, t, 1, 256, 0), tw, r.w[0].as<int8_t>(), s1, part);
                } else {
                  if (part != 2) launch_decode(tw, r.w[0].as<int8_t>(), s1);
                  if (part != 1) gemm_sched(whole_cfg, rv, gemm_call(r, sh, t, 0, 256, 0), s1);
                }
              }, 4, s1));
            }
        std::printf("T %d %-9s whole %8.1f = decode %6.1f + GEMM %8.1f %+7.1f | sliced (%d of %d blocks) %8.1f = decodes %6.1f + GEMMs %8.1f %+7.1f"
                    " | sliced - whole %+7.1f, of which decode %+7.1f, GEMM %+7.1f, interleaving %+7.1f\n",
                    t, sh.name, us[0][0], us[0][1], us[0][2], us[0][0] - us[0][1] - us[0][2], slice_even(static_cast<int>(sh.in), static_cast<int>(sh.out) / kBlockN),
                    static_cast<int>(sh.out) / kBlockN, us[1][0], us[1][1], us[1][2], us[1][0] - us[1][1] - us[1][2], us[1][0] - us[0][0],
                    us[1][1] - us[0][1], us[1][2] - us[0][2], (us[1][0] - us[1][1] - us[1][2]) - (us[0][0] - us[0][1] - us[0][2]));
      }
    }
    // Lever A's fold, estimated (not built). Into the RMSnorm before the first matrices of a block:
    // the norm already holds x, so the fold adds what each extra matrix costs inside the shared
    // launch (its rotation and INT8 write: (n3 - n1) / 2) and saves the rest of the shared launch
    // (the x read, the launch). Into the gate+up epilogue before down: an epilogue holds 128
    // columns, down's amax spans 17408, so the most it can do is hand each token's amax to a
    // one-pass A8; the saving is the per-matrix A8 less that one pass.
    std::printf("\n== lever A's fold, estimated: A8 per call (us) ==\n");
    for (const int t : tokens) {
      std::vector<float> ones(t, 1.0f);
      DeviceBytes amax(4ull * t);
      trellis_test::upload(amax, ones);
      auto per_matrix = [&](int in) {
        return time_us([&](int) { launch_a8(r.x.p, r.svh.p, t, in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), s1); }, 8, s1);
      };
      auto onepass = [&](int in) {
        return time_us([&](int) {
          sm8x_dense_a8_onepass<<<t, 256, 0, s1>>>(r.x.as<__nv_bfloat16>(), r.svh.as<__half>(), t, in, amax.as<float>(), r.a.as<int8_t>(),
                                                    r.sa.as<float>(), r.sq.as<int32_t>());
        }, 8, s1);
      };
      const double p5 = per_matrix(5120), p6 = per_matrix(6144), p17 = per_matrix(17408), o5 = onepass(5120), o17 = onepass(17408);
      double sh[4] = {0, 0, 0, 0};
      for (int n = 1; n <= 3; ++n) sh[n] = time_a8_shared(r, 5120, n, t, s1);
      const double marginal = (sh[3] - sh[1]) / 2.0;
      auto norm_fold = [&](int n) { return sh[n] - n * marginal; };
      const double down_fold = p17 - o17;
      const double gb5 = 3.0 * t * 5120 / 1e9;  // the x read and one INT8 write
      std::printf("T %d: per-matrix in 5120 %.1f, 6144 %.1f, 17408 %.1f; one pass (amax given) in 5120 %.1f, 17408 %.1f;"
                  " shared in 5120 for 1, 2, 3 matrices %.1f, %.1f, %.1f (%.2f TB/s of x read + INT8 at 1)\n",
                  t, p5, p6, p17, o5, o17, sh[1], sh[2], sh[3], gb5 / (sh[1] * 1e-6) / 1e3);
      std::printf("        each extra matrix in the shared launch %.1f; the RMSnorm fold saves %.1f per 2-matrix group, %.1f per 3-matrix group;"
                  " the gate+up fold saves %.1f of down's %.1f\n",
                  marginal, norm_fold(2), norm_fold(3), down_fold, p17);
      std::printf("        per chunk (48 GDN + 16 GQA + 64 MLP blocks): the folds save %.2f ms of lever A's %.2f ms of A8, %.2f ms from the RMSnorm folds and %.2f ms from down's\n",
                  (48 * norm_fold(2) + 16 * norm_fold(3) + 64 * norm_fold(2) + 64 * down_fold) / 1e3,
                  (48 * (sh[2] + p6) + 16 * (sh[3] + p6) + 64 * (sh[2] + p17)) / 1e3, (48 * norm_fold(2) + 16 * norm_fold(3) + 64 * norm_fold(2)) / 1e3,
                  64 * down_fold / 1e3);
    }
  }
  if (do_speed) {
    Route r(*std::max_element(tokens.begin(), tokens.end()));
    std::vector<std::vector<ChunkTotals>> chunks(tokens.size(), std::vector<ChunkTotals>(cfgs.size()));
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (ablation) prepare_choices(tokens[i], r, s1);
      for (size_t c = 0; c < cfgs.size(); ++c) speed(tokens[i], r, s1, cfgs[c], chunks[i][c]);
    }
    if (ablation) {
      for (const int t : tokens) lever_b_table(t, r, s1);
      for (size_t i = 0; i < tokens.size(); ++i) {
        std::printf("\n== ablation at T = %d: per chunk of the 64-layer body (ms) ==\n", tokens[i]);
        std::printf("%-12s | %8s %8s %9s %9s | %9s %7s | %8s\n", "levers", "decode", "A8", "GEMM", "dense", "fused", "f/d", "vs 2a");
        for (size_t c = 0; c < cfgs.size(); ++c) {
          const ChunkTotals &k = chunks[i][c];
          std::printf("%-12s | %8.2f %8.2f %9.2f %9.2f | %9.2f %7.2f | %8.3f\n", cfgs[c].name.c_str(), k.decode / 1e3, k.a8 / 1e3,
                      k.gemm / 1e3, k.total() / 1e3, k.fused / 1e3, k.fused / k.total(), chunks[i][0].total() / k.total());
        }
      }
    }
    std::vector<Lever> ov = {cfgs[0]};
    if (ablation) ov.push_back(make_lever(true, true, false));
    std::printf("\n== overlap: the next matrix's decode on a second stream (us per layer) ==\n");
    for (const Lever &l : ov)
      for (size_t i = 0; i < tokens.size(); ++i) {
        OverlapTimes g, q;
        overlap(tokens[i], r, s1, s2, hi, lo, l, &g, &q);
        const std::string tag = ov.size() > 1 ? " [" + l.name + "]" : std::string();
        std::printf("T = %d%s: GDN + MLP layer %.1f serial, %.1f naive (%.3fx), %.1f co-resident (%.3fx), %.1f sliced co-resident (%.3fx)\n",
                    tokens[i], tag.c_str(), g.serial, g.naive, g.serial / g.naive, g.coresident, g.serial / g.coresident, g.sliced,
                    g.serial / g.sliced);
        std::printf("        GQA + MLP layer %.1f serial, %.1f naive (%.3fx), %.1f co-resident (%.3fx), %.1f sliced co-resident (%.3fx)\n", q.serial,
                    q.naive, q.serial / q.naive, q.coresident, q.serial / q.coresident, q.sliced, q.serial / q.sliced);
        const double ser = 48 * g.serial + 16 * q.serial;
        const double naive = 48 * g.naive + 16 * q.naive, cores = 48 * g.coresident + 16 * q.coresident, sl = 48 * g.sliced + 16 * q.sliced;
        std::printf("        48 GDN + 16 GQA layers at K 3 (k, v at K 4): %.2f ms serial, %.2f naive (%.3fx), %.2f co-resident (%.3fx),"
                    " %.2f sliced co-resident (%.3fx)\n",
                    ser / 1e3, naive / 1e3, ser / naive, cores / 1e3, ser / cores, sl / 1e3, ser / sl);
      }
    std::printf("\n== projection: the body's prefill GEMMs per prompt, decode + A8 + GEMM on one stream (ms) ==\n");
    for (size_t c = 0; c < cfgs.size(); ++c)
      for (size_t i = 0; i < tokens.size(); ++i) {
        for (const int prompt : {8192, 65536}) {
          const double n = static_cast<double>(prompt) / tokens[i];
          const ChunkTotals &k = chunks[i][c];
          const std::string tag = cfgs.size() > 1 ? "[" + cfgs[c].name + "] " : std::string();
          std::printf("%sT = %d, %6d tokens: %4.0f chunks; dense %8.1f (decode %7.1f, A8 %7.1f, GEMM %8.1f), fused %8.1f; fused / dense %.2f\n",
                      tag.c_str(), tokens[i], prompt, n, n * k.total() / 1e3, n * k.decode / 1e3, n * k.a8 / 1e3, n * k.gemm / 1e3,
                      n * k.fused / 1e3, k.fused / k.total());
        }
      }
  }
  BENCH_CUDA(cudaStreamDestroy(s1));
  BENCH_CUDA(cudaStreamDestroy(s2));
  BENCH_CUDA(cudaStreamDestroy(hi));
  BENCH_CUDA(cudaStreamDestroy(lo));
  if (g_failed != 0) {
    std::printf("\n%d check(s) FAILED\n", g_failed);
    return 1;
  }
  std::printf("\ndone\n");
  return 0;
}
