// PROTOTYPE (sm8x dense prefill, phase 2a; GitHub #474) -- OURS, throwaway, never to merge.
//
// The dense prefill route an sm_8x card would run, as a standalone bench: every trellis matrix is
// decoded once per prefill chunk into a dense INT8 copy, the rotated activations become INT8
// with one scale per token, and our own integer GEMM multiplies them, mma.sync m16n8k32 s8.s8 ->
// s32 over the full K, with an fp32 epilogue on the CUDA cores. Written from the PTX ISA and our
// own kernels (trellis_linear_*, moe_common.cuh, the 8-bit band's s8 path), no library GEMM.
//
// The three new kernels, the ones the SASS check holds to integer MMA (names start sm8x_dense_):
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
//
// What the bench reports (plain text):
//   device     compute capability, SMs, shared memory, the runtime and driver versions
//   peaks      register-loop rates: s8 m16n8k32, f16 m16n8k16 with f32 and with f16 accumulate
//   check      with --artifact, real matrices of the 16 GB artifact and a synthetic activation
//              stand-in: the W8 codes against a host decode, the A8 values against phase 1's V2
//              (IGNIS_PROTO_FAKEQUANT=w8a8tok) bit for bit, the GEMM's int32 accumulators against
//              an exact integer reference (dp4a, no tensor core), the outputs of both routes
//              against an fp64 reference of the quantized product, and dense against V2
//   speed      per body shape and T: decode, A8, GEMM (TOPS, % of the s8 peak), their total, and
//              today's fused trellis prefill kernel (ignis_trellis_linear) at the same T
//   chunk      the body's per-chunk totals at the public quant's K mix, dense against fused
//   overlap    one GDN + MLP layer and one GQA + MLP layer, the next matrix's decode on a second
//              stream through the double buffer, against the same sequence on one stream
//   project    the body's prefill-GEMM time per 8K and per 64K tokens, dense against fused
//
//   ignis_sm8x_dense_bench [--artifact PATH] [--tokens 2048,4096] [--check-tokens N] [--reps N]
//                          [--no-check] [--no-speed] [--no-peaks]

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
  int32_t *raw;  // non-null: the int32 accumulators [tokens][out] instead of y (the check)
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
__device__ __forceinline__ void load_stage(const GemmParams &p, uint32_t st, int kt, int row0, int tn) {
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
  const int8_t *blk = p.w + (static_cast<size_t>(tn) * (p.in / kBK) + kt) * kBlockBytes;
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

// kDiag (a diagnostic of the bench, not the route): 1 skips the main loop's global loads (the
// stages keep the prologue's data), 2 also its ldmatrix (one set of fragments for every step).
template <int BM, int kDiag = 0>
__global__ void __launch_bounds__(kGemmThreads, 1) sm8x_dense_gemm(const __grid_constant__ GemmParams p) {
  using C = Cfg<BM>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int tile = blockIdx.x;
  const int tm = p.raster == 0 ? tile % p.mt : tile / p.nt;
  const int tn = p.raster == 0 ? tile / p.mt : tile % p.nt;
  const int row0 = tm * BM;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int wm = warp / C::kWarpsN;
  const int wn = warp % C::kWarpsN;
  const int kts = p.in / kBK;

  int32_t acc[C::kMI][C::kNI][4];
#pragma unroll
  for (int i = 0; i < C::kMI; ++i)
#pragma unroll
    for (int n = 0; n < C::kNI; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0;

  // The pipeline: at the top of step kt, stage kt is waited for and visible; stage kt + 3 goes
  // into the buffer step kt - 1 read, which the barrier has just freed. Within a stage the second
  // k32 step's fragments load while the first one's MMAs run.
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < kts) load_stage<BM>(p, base + s * C::kStage, s, row0, tn);
    cp_async_commit();
  }
  uint32_t a0[C::kMI][4], b0[C::kNI / 2][4], a1[C::kMI][4], b1[C::kNI / 2][4];
  if constexpr (kDiag == 2) {
    cp_async_wait<0>();
    __syncthreads();
    load_frags<BM>(base, 0, wm, wn, a0, b0);
    load_frags<BM>(base, 1, wm, wn, a1, b1);
  }
  for (int kt = 0; kt < kts; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const int pf = kt + kStages - 1;
    if (kDiag == 0 && pf < kts) load_stage<BM>(p, base + (pf % kStages) * C::kStage, pf, row0, tn);
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

  const int g = lane >> 2;
  const int c4 = lane & 3;
  if (p.raw != nullptr) {
#pragma unroll
    for (int i = 0; i < C::kMI; ++i) {
      const int t = row0 + wm * C::kWM + 16 * i + g;
#pragma unroll
      for (int n = 0; n < C::kNI; ++n) {
        const int col = tn * kBN + wn * C::kWN + 8 * n + 2 * c4;
        if (t < p.tokens) *reinterpret_cast<int2 *>(p.raw + static_cast<size_t>(t) * p.out + col) = make_int2(acc[i][n][0], acc[i][n][1]);
        if (t + 8 < p.tokens) *reinterpret_cast<int2 *>(p.raw + static_cast<size_t>(t + 8) * p.out + col) = make_int2(acc[i][n][2], acc[i][n][3]);
      }
    }
    return;
  }

  // Epilogue, one m16 block of every warp row at a time through shared memory: the scales and the
  // offset's rank-1 term per token in fp32, then each 128-column row back through the Hadamard,
  // svh, and BF16 out as the fused kernel's epilogue.
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

void prepare_gemm() {
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<128>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
  BENCH_CUDA(cudaFuncSetAttribute(sm8x_dense_gemm<256, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg<256>::kSmem));
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
};

void launch_gemm(const GemmCall &c, cudaStream_t s) {
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

// L2-resident read bandwidth: a 32 MiB buffer (well inside the L2) read with 16-byte
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

double l2_read_tbs(int sms) {
  const size_t bytes = 32ull << 20;
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
// fused prefill kernel, "w8a8tok" phase 1's V2.
void prepare_fused(const char *fakequant) {
  set_env("IGNIS_TRELLIS_SM_CAP", "0");
  set_env("IGNIS_PROTO_FAKEQUANT", fakequant);
  TL_RC(ignis_trellis_prepare());
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

void check_matrix(const Artifact &art, const std::string &name, int tokens, cudaStream_t stream) {
  HostTrellis m;
  if (!load_trellis(art, name, m)) {
    std::printf("  %s: not a trellis matrix of the artifact\n", name.c_str());
    ++g_failed;
    return;
  }
  const int in = static_cast<int>(m.in), out = static_cast<int>(m.out);
  std::printf("  %s: in %d out %d K %.1f, %d tokens\n", name.c_str(), in, out, m.k2 / 2.0, tokens);
  DeviceBytes d_words(m.words.size() * 2 + 256), d_suh(2ull * in), d_svh(2ull * out);
  trellis_test::upload(d_words, m.words);
  trellis_test::upload(d_suh, m.suh);
  trellis_test::upload(d_svh, m.svh);
  const ignis_trellis_weight tw{d_words.p, d_suh.p, d_svh.p, m.k2, m.in, m.out, 0};

  // W8: the dense W against the host decode, every tile of every 8th tile row.
  DeviceBytes d_w(static_cast<size_t>(in) * out);
  launch_decode(tw, d_w.as<int8_t>(), stream);
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int8_t> w = trellis_test::download<int8_t>(d_w.p, static_cast<size_t>(in) * out);
  {
    size_t checked = 0, wrong = 0;
    int s[16][16];
    for (int tk = 3; tk < in / 16; tk += 8) {
      for (int tc = 0; tc < out / 16; ++tc) {
        host_tile_codes(m.words.data(), m.k2, out / 16, tk, tc, s);
        for (int r = 0; r < 16; ++r)
          for (int c = 0; c < 16; ++c) {
            ++checked;
            wrong += w[dense_index(in, tc * 16 + c, tk * 16 + r)] != host_w8(s[r][c]);
          }
      }
    }
    std::printf("    W8 codes against the host decode: %zu of %zu differ\n", wrong, checked);
    verdict(wrong == 0, "W8 codes bit-identical to phase 1's quantizer");
  }

  // A8 against V2's own rotation (rotate_token_kernel), read back from its workspace.
  const std::vector<uint16_t> xh = standin_x(tokens, in, 4242);
  DeviceBytes d_x(xh.size() * 2), d_a(static_cast<size_t>(tokens) * in), d_sa(4ull * tokens), d_sq(4ull * tokens);
  trellis_test::upload(d_x, xh);
  launch_a8(d_x.p, d_suh.p, tokens, in, d_a.as<int8_t>(), d_sa.as<float>(), d_sq.as<int32_t>(), stream);
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int8_t> a = trellis_test::download<int8_t>(d_a.p, static_cast<size_t>(tokens) * in);
  const std::vector<float> sa = trellis_test::download<float>(d_sa.p, tokens);
  const std::vector<int32_t> sq = trellis_test::download<int32_t>(d_sq.p, tokens);

  prepare_fused("w8a8tok");
  const uint64_t ws_bytes = ignis_trellis_workspace_bytes(m.in, m.out, 1, tokens);
  DeviceBytes ws_mem(ws_bytes), d_yv(2ull * tokens * out);
  ignis_trellis_workspace ws{ws_mem.p, ws_bytes};
  TL_RC(ignis_trellis_workspace_init(&ws, stream));
  TL_RC(ignis_trellis_linear(&tw, d_x.p, tokens, d_yv.p, m.out, IGNIS_TRELLIS_STORE, &ws, stream));
  BENCH_CUDA(cudaStreamSynchronize(stream));
  {
    const ignis_tl::Layout l = ignis_tl::layout(m.in, m.out, 1, tokens, 1);
    const std::vector<uint16_t> xr = trellis_test::download<uint16_t>(static_cast<char *>(ws_mem.p) + l.xr, static_cast<size_t>(tokens) * in);
    const std::vector<float> inv = trellis_test::download<float>(static_cast<char *>(ws_mem.p) + l.inv, static_cast<size_t>(in / 128) * tokens);
    size_t wrong = 0, wrong_scale = 0, wrong_sum = 0;
    for (int t = 0; t < tokens; ++t) {
      long long sum = 0;
      for (int k = 0; k < in; ++k) {
        const int8_t q = a[a_index(tokens, t, k)];
        wrong += moe_test::f16_to_f32(xr[static_cast<size_t>(t) * in + k]) != static_cast<float>(q);
        sum += q;
      }
      for (int b = 0; b < in / 128; ++b) wrong_scale += inv[static_cast<size_t>(b) * tokens + t] != sa[t];
      wrong_sum += sum != sq[t];
    }
    std::printf("    A8 against V2's rotation: %zu of %zu values, %zu scales and %zu sums differ\n", wrong,
                static_cast<size_t>(tokens) * in, wrong_scale, wrong_sum);
    verdict(wrong == 0 && wrong_scale == 0 && wrong_sum == 0, "A8 bit-identical to V2's");
  }
  prepare_fused("off");

  // The GEMM's int32 accumulators against the integer reference, both tile heights.
  DeviceBytes d_raw(4ull * tokens * out), d_ref(4ull * tokens * out);
  meter::ref_int_gemm<<<dim3(out / 64, (tokens + 63) / 64), 256, 0, stream>>>(d_a.as<int8_t>(), d_w.as<int8_t>(), tokens, in, out, d_ref.as<int32_t>());
  BENCH_CUDA(cudaGetLastError());
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<int32_t> ref = trellis_test::download<int32_t>(d_ref.p, static_cast<size_t>(tokens) * out);
  for (const int bm : {256, 128}) {
    BENCH_CUDA(cudaMemsetAsync(d_raw.p, 0x5A, d_raw.bytes, stream));
    GemmCall c{d_a.as<int8_t>(), d_w.as<int8_t>(), d_sa.as<float>(), d_sq.as<int32_t>(), d_svh.p, tokens, in, out, kStore, nullptr, out};
    c.raw = d_raw.as<int32_t>();
    c.bm = bm;
    launch_gemm(c, stream);
    BENCH_CUDA(cudaStreamSynchronize(stream));
    const std::vector<int32_t> raw = trellis_test::download<int32_t>(d_raw.p, static_cast<size_t>(tokens) * out);
    size_t wrong = 0;
    long long amax = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
      wrong += raw[i] != ref[i];
      amax = std::max(amax, std::llabs(static_cast<long long>(ref[i])));
    }
    std::printf("    int32 accumulators (BM %d) against the integer reference: %zu of %zu differ (max |acc| %lld)\n", bm,
                wrong, raw.size(), amax);
    verdict(wrong == 0, "int32 accumulators exactly equal (BM " + std::to_string(bm) + ")");
  }

  // The outputs: dense and V2 against the fp64 reference of the quantized product, and each other.
  DeviceBytes d_yd(2ull * tokens * out);
  GemmCall c{d_a.as<int8_t>(), d_w.as<int8_t>(), d_sa.as<float>(), d_sq.as<int32_t>(), d_svh.p, tokens, in, out, kStore, d_yd.p, out};
  launch_gemm(c, stream);
  BENCH_CUDA(cudaStreamSynchronize(stream));
  const std::vector<double> yd = trellis_test::bf16_to_f64(trellis_test::download<uint16_t>(d_yd.p, static_cast<size_t>(tokens) * out));
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
  const Agreement dr = agree(yd, yref, b8), vr = agree(yv, yref, b16), dv = agree(yd, yv, bsum);
  size_t same = 0, one_ulp = 0;
  double max_ulps = 0.0;
  for (size_t i = 0; i < yd.size(); ++i) {
    const double m = std::max(std::fabs(yd[i]), std::fabs(yv[i]));
    int e = 0;
    std::frexp(m, &e);
    const double ulp = m > 0.0 ? std::ldexp(1.0, e - 8) : 1.0;  // BF16: 8 significant bits
    const double u = std::fabs(yd[i] - yv[i]) / ulp;
    same += u == 0.0;
    one_ulp += u <= 1.0;
    max_ulps = std::max(max_ulps, u);
  }
  std::printf("    dense - ref: worst %.3f of 2^-8|Y| + 2^-9 s (BF16 rounding of the exact sum), rel RMS %.3g\n", dr.worst, dr.rel_rms);
  std::printf("    V2    - ref: worst %.3f of 2^-8|Y| + 2^-8 s (fp16 accumulation), rel RMS %.3g\n", vr.worst, vr.rel_rms);
  std::printf("    dense - V2:  worst %.3f of the two bounds' sum, rel RMS %.3g; BF16 bits equal at %.2f%%, within 1 ulp at %.3f%%, max %.1f ulp\n",
              dv.worst, dv.rel_rms, 100.0 * same / yd.size(), 100.0 * one_ulp / yd.size(), max_ulps);
  verdict(dr.over == 0, "dense output = the exact product within BF16 rounding");
  verdict(vr.over == 0, "V2 output = the exact product within fp16 accumulation's bound");
  verdict(dv.over == 0, "dense output = V2 within the two bounds' sum");
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

// The route's buffers at the largest shape: two dense W (the double buffer), the INT8 activations
// and their scales, x and y.
struct Route {
  DeviceBytes w[2], a, sa, sq, x, y, svh;
  size_t w_bytes = 0;
  Route(int max_tokens) {
    w_bytes = 17408ull * 5120;
    w[0] = DeviceBytes(w_bytes);
    w[1] = DeviceBytes(w_bytes);
    a = DeviceBytes(static_cast<size_t>(max_tokens) * 17408);
    sa = DeviceBytes(4ull * max_tokens);
    sq = DeviceBytes(4ull * max_tokens);
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
};

double time_decode(const Pool &pool, int8_t *const (&w)[2], cudaStream_t s) {
  return time_us([&](int i) { launch_decode(pool.w[i % pool.w.size()], w[i & 1], s); }, 8, s);
}

double time_a8(Route &r, const Shape &sh, int tokens, cudaStream_t s) {
  return time_us([&](int) { launch_a8(r.x.p, r.svh.p, tokens, sh.in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), s); }, 8, s);
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

struct GemmChoice {
  double us = 1e30;
  int bm = 256, raster = 0;
};

// The GEMM's tile height and raster order, the faster of the four on this card for this shape.
GemmChoice best_gemm(Route &r, const Shape &sh, int tokens, cudaStream_t s) {
  GemmChoice best;
  for (const int bm : {256, 128})
    for (const int raster : {0, 1}) {
      const double us = time_gemm(r, sh, tokens, bm, raster, s);
      if (us < best.us) best = GemmChoice{us, bm, raster};
    }
  return best;
}

std::map<std::string, GemmChoice> g_choice;  // per (shape, tokens)

std::string key(const Shape &s, int tokens) { return std::string(s.name) + "/" + std::to_string(tokens); }

struct ChunkTotals {
  double decode = 0, a8 = 0, gemm = 0, fused = 0;
  double total() const { return decode + a8 + gemm; }
};

void speed(int tokens, Route &r, cudaStream_t s, ChunkTotals &chunk) {
  int8_t *const wb[2] = {r.w[0].as<int8_t>(), r.w[1].as<int8_t>()};
  std::printf("\n== T = %d: per call (us), the dense route against today's fused trellis prefill kernel ==\n", tokens);
  std::printf("%-12s %6s %6s %3s | %8s %7s %8s %7s %6s %9s | %9s %7s | %6s\n", "matrix", "in", "out", "K", "decode", "A8",
              "GEMM", "TOPS", "%s8", "dense", "fused", "TFLOPS", "f/d");
  for (const Shape &sh : kBody) {
    const int per = sh.swiglu ? 2 : 1;
    Pool pool;
    make_pool(pool, sh, per);
    const double dec = time_decode(pool, wb, s) * per;
    const double a8 = time_a8(r, sh, tokens, s) * per;
    const GemmChoice gc = best_gemm(r, sh, tokens, s);
    g_choice[key(sh, tokens)] = gc;
    const double gemm = gc.us * per;
    const double ops = 2.0 * tokens * sh.in * sh.out * per;
    const double tops = ops / (gemm * 1e-6) / 1e12;
    const double fused = time_fused(pool, sh, tokens, s);
    const double dense = dec + a8 + gemm;
    std::printf("%-12s %6u %6u %3.1f | %8.1f %7.1f %8.1f %7.1f %5.1f%% %9.1f | %9.1f %7.1f | %6.2f   (BM %d, %s)\n", sh.name,
                sh.in, sh.out, sh.k2 / 2.0, dec, a8, gemm, tops, 100.0 * tops / g_s8_peak, dense, fused,
                ops / (fused * 1e-6) / 1e12, fused / dense, gc.bm, gc.raster == 0 ? "tokens fastest" : "outputs fastest");
  }
  // The chunk: the body at the quant's K mix (decode and the fused kernel depend on K; A8 and the
  // GEMM do not).
  std::map<std::string, std::pair<double, double>> a8_gemm;  // per shape name
  chunk = ChunkTotals{};
  double ops_total = 0.0;
  for (const MixEntry &e : kMix) {
    const Shape &sh = e.shape;
    const int per = sh.swiglu ? 2 : 1;
    Pool pool;
    make_pool(pool, sh, per);
    const double dec = time_decode(pool, wb, s) * per;
    auto it = a8_gemm.find(sh.name);
    if (it == a8_gemm.end()) {
      const auto ch = g_choice.find(key(sh, tokens));
      const GemmChoice gc = ch != g_choice.end() ? ch->second : best_gemm(r, sh, tokens, s);
      const double a8 = time_a8(r, sh, tokens, s) * per;
      const double gemm = time_gemm(r, sh, tokens, gc.bm, gc.raster, s) * per;
      it = a8_gemm.emplace(sh.name, std::make_pair(a8, gemm)).first;
    }
    const double fused = time_fused(pool, sh, tokens, s);
    chunk.decode += e.calls * dec;
    chunk.a8 += e.calls * it->second.first;
    chunk.gemm += e.calls * it->second.second;
    chunk.fused += e.calls * fused;
    ops_total += e.calls * 2.0 * tokens * sh.in * sh.out * per;
  }
  std::printf("chunk of %d tokens, the 64-layer body at the quant's K mix (ms): decode %.2f + A8 %.2f + GEMM %.2f = dense %.2f;"
              " fused %.2f; fused / dense %.2f; GEMM %.1f TOPS (%.1f%% of s8), fused %.1f TFLOPS\n",
              tokens, chunk.decode / 1e3, chunk.a8 / 1e3, chunk.gemm / 1e3, chunk.total() / 1e3, chunk.fused / 1e3,
              chunk.fused / chunk.total(), ops_total / (chunk.gemm * 1e-6) / 1e12,
              100.0 * ops_total / (chunk.gemm * 1e-6) / 1e12 / g_s8_peak, ops_total / (chunk.fused * 1e-6) / 1e12);
}

// One layer's matrices in order, the next one's decode on a second stream through the double
// buffer, against the same sequence on one stream.
struct OverlapTimes {
  double serial = 0, naive = 0, coresident = 0;
};

// serial: decode, A8, GEMM per matrix on one stream. naive: the next matrix's decode (4 warps a CTA)
// on a second stream of the same priority. coresident: that decode at 2 warps a CTA, small enough to
// sit beside a GEMM CTA, on the lowest-priority stream, the A8s and GEMMs on the highest.
void overlap(int tokens, Route &r, cudaStream_t s1, cudaStream_t s2, cudaStream_t hi, cudaStream_t lo, OverlapTimes *gdn_t,
             OverlapTimes *gqa_t) {
  const Shape gdn[] = {{"gdn qkv", 5120, 10240, 6, 0}, {"gdn z", 5120, 6144, 6, 0}, {"gdn out", 6144, 5120, 6, 0},
                       {"mlp gate", 5120, 17408, 6, 0}, {"mlp up", 5120, 17408, 6, 0}, {"mlp down", 17408, 5120, 6, 0}};
  const Shape gqa[] = {{"gqa q", 5120, 12288, 6, 0}, {"gqa k", 5120, 1024, 8, 0}, {"gqa v", 5120, 1024, 8, 0},
                       {"gqa o", 6144, 5120, 6, 0}, {"mlp gate", 5120, 17408, 6, 0}, {"mlp up", 5120, 17408, 6, 0},
                       {"mlp down", 17408, 5120, 6, 0}};
  auto run = [&](const Shape *seq, int n, OverlapTimes *out) {
    std::vector<Pool> pools(n);
    for (int i = 0; i < n; ++i) make_pool(pools[i], seq[i], 1);
    std::vector<GemmCall> calls;
    for (int i = 0; i < n; ++i) {
      const Shape body = std::string(seq[i].name).rfind("mlp gate", 0) == 0 || std::string(seq[i].name).rfind("mlp up", 0) == 0
                             ? Shape{"mlp gate+up", 5120, 17408, 6, 1}
                             : seq[i];
      const auto ch = g_choice.find(key(body, tokens));
      const GemmChoice gc = ch != g_choice.end() ? ch->second : GemmChoice{};
      calls.push_back(gemm_call(r, seq[i], tokens, i & 1, gc.bm, gc.raster));
    }
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
        launch_a8(r.x.p, r.svh.p, tokens, seq[i].in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), s1);
        launch_gemm(calls[i], s1);
      }
    };
    auto overlapped = [&](int rep, cudaStream_t sc, cudaStream_t sd, int warps) {
      BENCH_CUDA(cudaEventRecord(fork, sc));
      BENCH_CUDA(cudaStreamWaitEvent(sd, fork, 0));
      launch_decode(pools[0].w[rep % pools[0].w.size()], r.w[0].as<int8_t>(), sd, warps);
      BENCH_CUDA(cudaEventRecord(dec[0], sd));
      for (int i = 0; i < n; ++i) {
        launch_a8(r.x.p, r.svh.p, tokens, seq[i].in, r.a.as<int8_t>(), r.sa.as<float>(), r.sq.as<int32_t>(), sc);
        BENCH_CUDA(cudaStreamWaitEvent(sc, dec[i], 0));
        launch_gemm(calls[i], sc);
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
    out->naive = measure(s1, [&](int rep) { overlapped(rep, s1, s2, kDecWarps); });
    out->coresident = measure(hi, [&](int rep) { overlapped(rep, hi, lo, 2); });
    for (int i = 0; i < n; ++i) {
      cudaEventDestroy(dec[i]);
      cudaEventDestroy(gem[i]);
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaEventDestroy(fork);
  };
  run(gdn, 6, gdn_t);
  run(gqa, 7, gqa_t);
}

void kernel_report(const char *name, const void *f, int threads, int smem) {
  cudaFuncAttributes at{};
  BENCH_CUDA(cudaFuncGetAttributes(&at, f));
  int ctas = 0;
  BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, f, threads, smem));
  std::printf("  %-26s %3d regs, %5zu B local (spills), %6d B dynamic + %5zu B static smem, %d threads, %d CTA/SM\n", name,
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

}  // namespace

int main(int argc, char **argv) {
  std::string artifact;
  std::vector<int> tokens = {2048, 4096};
  int check_tokens = 2048;
  bool do_check = true, do_speed = true, do_peaks = true, diag = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--artifact" && i + 1 < argc) {
      artifact = argv[++i];
    } else if (a == "--tokens" && i + 1 < argc) {
      tokens = parse_list(argv[++i]);
    } else if (a == "--check-tokens" && i + 1 < argc) {
      check_tokens = std::atoi(argv[++i]);
    } else if (a == "--reps" && i + 1 < argc) {
      g_reps = std::atoi(argv[++i]);
    } else if (a == "--no-check") {
      do_check = false;
    } else if (a == "--no-speed") {
      do_speed = false;
    } else if (a == "--no-peaks") {
      do_peaks = false;
    } else if (a == "--diag") {
      diag = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--artifact PATH] [--tokens 2048,4096] [--check-tokens N] [--reps N] [--no-check] [--no-speed] [--no-peaks]\n",
                   argv[0]);
      return 2;
    }
  }
  std::fesetround(FE_TONEAREST);
  int runtime = 0, driver = 0;
  BENCH_CUDA(cudaRuntimeGetVersion(&runtime));
  BENCH_CUDA(cudaDriverGetVersion(&driver));
  std::printf("sm8x dense prefill bench (prototype, phase 2a): CUDA runtime %d.%d, driver supports %d.%d\n", runtime / 1000,
              (runtime % 1000) / 10, driver / 1000, (driver % 1000) / 10);
  if (driver < runtime) {
    std::fprintf(stderr, "FATAL: the driver supports CUDA %d.%d, the bench needs %d.%d or newer\n", driver / 1000,
                 (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
    return 1;
  }
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  std::printf("device: sm_%d%d, %d SMs, %zu B shared memory per block (opt-in), %zu B per SM, %d MB L2\n", prop.major, prop.minor,
              prop.multiProcessorCount, prop.sharedMemPerBlockOptin, prop.sharedMemPerMultiprocessor, prop.l2CacheSize >> 20);
  prepare_gemm();
  prepare_fused("off");
  std::printf("kernels:\n");
  kernel_report("sm8x_dense_gemm<256>", reinterpret_cast<const void *>(sm8x_dense_gemm<256>), kGemmThreads, Cfg<256>::kSmem);
  kernel_report("sm8x_dense_gemm<128>", reinterpret_cast<const void *>(sm8x_dense_gemm<128>), kGemmThreads, Cfg<128>::kSmem);
  kernel_report("sm8x_dense_decode<6, 4>", reinterpret_cast<const void *>(sm8x_dense_decode<6, 4>), 128, 0);
  kernel_report("sm8x_dense_decode<10, 4>", reinterpret_cast<const void *>(sm8x_dense_decode<10, 4>), 128, 0);
  kernel_report("sm8x_dense_decode<6, 2>", reinterpret_cast<const void *>(sm8x_dense_decode<6, 2>), 64, 0);
  kernel_report("sm8x_dense_a8", reinterpret_cast<const void *>(sm8x_dense_a8), 256, 0);

  if (do_peaks) {
    const meter::Rate s8 = meter::peak<meter::kS8>(g_sms, 4096, 10, 300);
    const meter::Rate f32 = meter::peak<meter::kF16F32>(g_sms, 4096, 10, 300);
    const meter::Rate f16 = meter::peak<meter::kF16F16>(g_sms, 4096, 10, 300);
    g_s8_peak = s8.tops;
    std::printf("L2-resident read bandwidth (32 MiB, ld.global.cg, all SMs): %.2f TB/s\n", meter::l2_read_tbs(g_sms));
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
  if (do_check) {
    if (artifact.empty()) {
      std::printf("check: skipped (no --artifact)\n");
    } else {
      Artifact art;
      if (!art.open(artifact)) {
        std::fprintf(stderr, "FATAL: %s is not a readable .ninfer v2 artifact\n", artifact.c_str());
        return 1;
      }
      std::printf("check on %s, %d tokens of the activation stand-in:\n", artifact.c_str(), check_tokens);
      const char *names[] = {"text/layers/0/mlp/gate_proj", "text/layers/1/gdn/in_proj_qkv", "text/layers/3/attention/k_proj",
                             "text/layers/15/attention/v_proj", "text/layers/3/mlp/down"};
      for (const char *n : names) check_matrix(art, n, check_tokens, s1);
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
  }
  if (do_speed) {
    Route r(*std::max_element(tokens.begin(), tokens.end()));
    std::vector<ChunkTotals> chunks(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) speed(tokens[i], r, s1, chunks[i]);
    std::printf("\n== overlap: the next matrix's decode on a second stream (us per layer) ==\n");
    std::vector<double> ovl_chunk(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
      OverlapTimes g, q;
      overlap(tokens[i], r, s1, s2, hi, lo, &g, &q);
      std::printf("T = %d: GDN + MLP layer %.1f serial, %.1f naive (%.3fx), %.1f co-resident (%.3fx)\n", tokens[i], g.serial,
                  g.naive, g.serial / g.naive, g.coresident, g.serial / g.coresident);
      std::printf("        GQA + MLP layer %.1f serial, %.1f naive (%.3fx), %.1f co-resident (%.3fx)\n", q.serial, q.naive,
                  q.serial / q.naive, q.coresident, q.serial / q.coresident);
      const double ser = 48 * g.serial + 16 * q.serial;
      const double best = std::min(48 * g.naive + 16 * q.naive, 48 * g.coresident + 16 * q.coresident);
      std::printf("        48 GDN + 16 GQA layers at K 3 (k, v at K 4): %.2f ms serial, %.2f ms naive, %.2f ms co-resident\n",
                  ser / 1e3, (48 * g.naive + 16 * q.naive) / 1e3, (48 * g.coresident + 16 * q.coresident) / 1e3);
      ovl_chunk[i] = std::min(1.0, best / ser);
    }
    std::printf("\n== projection: the body's prefill GEMMs per prompt (ms) ==\n");
    for (size_t i = 0; i < tokens.size(); ++i) {
      for (const int prompt : {8192, 65536}) {
        const double n = static_cast<double>(prompt) / tokens[i];
        std::printf("T = %d, %6d tokens: %4.0f chunks; dense %8.1f (best overlap %8.1f), fused %8.1f, fused / dense %.2f\n", tokens[i],
                    prompt, n, n * chunks[i].total() / 1e3, n * chunks[i].total() * ovl_chunk[i] / 1e3, n * chunks[i].fused / 1e3,
                    chunks[i].fused / chunks[i].total());
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
