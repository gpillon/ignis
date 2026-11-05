// PROTOTYPE (sm8x dense prefill, phase 2d; GitHub #474), throwaway: the 27B's prompt attention
// over a whole causal prompt, chunk by chunk, for each band a card below compute capability 12.0
// could run, against today's 8-bit band where the card has it. NOT a CTest test (a timing is a
// finding, not a pass/fail; docs/agents/testing.md), though its check exits 1 when it fails.
//
// Standalone, so one binary carries it to another card: it includes the attention's source
// (kernel/src/hq_prompt_attention.cu) for its kernels and its pass driver, attend(), without the
// op's entry points, and builds the hq-e8-2b cache itself with the vendored codec's encoder. It
// never prints the device's name.
//
//   ignis_sm8x_attn_bench [--prompts 8192,65536] [--chunk 4096] [--rounds 5] [--diag]
//                         [--no-check] [--variants] [--peaks S8,F16F16,F16F32]
//
// The prompt: every chunk of --chunk tokens attends its whole prefix and itself causally, the
// residual window on as the engine serves it (the sinks, the ring of the 512 keys before the
// chunk, the chunk, in BF16), every other key from the codec through the band. One GQA layer is
// timed per chunk; a prompt's total is the sum over its chunks times the 27B's 16 GQA layers.
// The append to the cache and the output gate are not attention and are not timed.
//
// The bands:
//   fp16   the FP16 band, what an sm_86 card runs today: Q K^T and P V on the FP16 MMA with FP16
//          accumulate, 16384 keys a band;
//   fp8    today's 8-bit band (compute capability 12.0 only): INT8 Q K^T, P V on the block-scaled
//          E4M3 MMA with FP32 accumulate, 32768 keys a band;
//   i8ref  the i8 band as phase 1 measured it (5ea54bc6's fa_kernel): INT8 Q K^T, P V on the
//          u8 x s8 MMA, exact in int32 per 32-key tile;
//   i8     the same band on fa_i8_kernel (phase 2d's schedule); its output must equal i8ref's
//          byte for byte.
//
// Sections: the setting, the kernels' resources, the MMA peaks, the check (i8 against i8ref, every
// chunk of every prompt, every i8 variant, byte for byte in the output and in the carried state;
// and each band's distance from i8ref, so a broken setup cannot pass), the per-chunk and per-
// prompt times per band and pass with their efficiencies, and with --diag the i8 kernel's stages
// and the key splits.

#define IGNIS_HQ_PROMPT_KERNELS_ONLY 1
#define IGNIS_HQ_PROMPT_DIAG 1
#include "hq_prompt_attention.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace bench {

#define BENCH_CUDA(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s (%s:%d)\n", #expr, cudaGetErrorString(_err),   \
                   __FILE__, __LINE__);                                                         \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

constexpr int kLayers = 16;  // the 27B's GQA layers
constexpr int kPageTokens = static_cast<int>(ninfer::kPagedKVPageSize);
constexpr int kSideRows = kSinks + kRing;

struct DeviceBytes {
  void *p = nullptr;
  std::size_t bytes = 0;
  DeviceBytes() = default;
  explicit DeviceBytes(std::size_t n) { alloc(n); }
  void alloc(std::size_t n) {
    if (p != nullptr) BENCH_CUDA(cudaFree(p));
    bytes = n;
    BENCH_CUDA(cudaMalloc(&p, std::max<std::size_t>(n, 256)));
  }
  ~DeviceBytes() {
    if (p != nullptr) cudaFree(p);
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
  template <class T> T *as() const { return static_cast<T *>(p); }
};

// ---- the rows: one deterministic Gaussian per (stream, index), as the codec is calibrated on ---

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ float gauss(std::uint64_t stream, std::uint64_t index) {
  const std::uint64_t h = mix64(index * 0x2545F4914F6CDD1Dull ^ (stream << 56) ^ stream);
  const float u1 = (static_cast<float>(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
  const float u2 = static_cast<float>((h >> 16) & 0xFFFFFFu) * (1.0f / 16777216.0f);
  return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
constexpr std::uint64_t kStreamK = 1, kStreamV = 2, kStreamQ = 3;
// Key `key`'s row of KV head `head`, unrotated, as the projection would give it.
__device__ __forceinline__ __nv_bfloat16 kv_value(bool role_v, int head, int key, int d) {
  return __float2bfloat16(gauss(role_v ? kStreamV : kStreamK,
                                (static_cast<std::uint64_t>(key) * kKvHeads + head) * kHeadDim + d));
}

struct Cache {
  DeviceBytes codes_k, codes_v, meta_k, meta_v, table, side_k, side_v;
  int pages = 0;
};

// One warp per (key, KV head, role): the row through the codec's encoder into its page, as the
// append would store it.
__global__ void __launch_bounds__(256) encode_rows(std::uint8_t *codes_k, std::uint8_t *codes_v,
                                                   std::uint8_t *meta_k, std::uint8_t *meta_v,
                                                   const std::int32_t *table, int key_begin, int count) {
  __shared__ std::int8_t signs[kHeadDim];
  __shared__ __align__(16) __nv_bfloat16 row[8][kHeadDim];
  __shared__ float u_scaled[8][kHeadDim];
  __shared__ std::uint32_t syms[8][kHeadDim];
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int w = static_cast<int>(blockIdx.x) * 8 + warp;
  if (w >= count * kKvHeads * 2) return;
  const bool role_v = (w & 1) != 0;
  const int head = (w >> 1) % kKvHeads;
  const int key = key_begin + (w >> 1) / kKvHeads;
  for (int d = lane; d < kHeadDim; d += 32) row[warp][d] = kv_value(role_v, head, key, d);
  __syncwarp();
  std::uint8_t *codes = ninfer::ops::hq_row_codes_mut<Geometry>(role_v ? codes_v : codes_k, table, head, key);
  std::uint8_t *meta = ninfer::ops::hq_row_meta_mut<Geometry>(role_v ? meta_v : meta_k, table, head, key);
  ninfer::ops::hq_encode_row_warp(row[warp], signs, 0, u_scaled[warp], syms[warp], codes, meta,
                                  ninfer::ops::hq_dither_row_seed(head, key, role_v));
}

// One warp per (key, KV head, role): the exact row, rotated, into its side-plane row (a sink, or
// the ring slot key & 511), as the append keeps it.
__global__ void __launch_bounds__(256) side_rows(__nv_bfloat16 *side_k, __nv_bfloat16 *side_v, int key_begin,
                                                 int count) {
  __shared__ std::int8_t signs[kHeadDim];
  __shared__ __align__(16) __nv_bfloat16 row[8][kHeadDim];
  ninfer::ops::hq_engine_signs_fill(signs);
  __syncthreads();
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int w = static_cast<int>(blockIdx.x) * 8 + warp;
  if (w >= count * kKvHeads * 2) return;
  const bool role_v = (w & 1) != 0;
  const int head = (w >> 1) % kKvHeads;
  const int key = key_begin + (w >> 1) / kKvHeads;
  for (int d = lane; d < kHeadDim; d += 32) row[warp][d] = kv_value(role_v, head, key, d);
  __syncwarp();
  ninfer::ops::hq_store_rotated_row_warp(row[warp], signs,
                                         ninfer::ops::hq_residual_row<Geometry>(role_v ? side_v : side_k, 0, head, key));
}

// The chunk's own inputs: Q [width][24][256], K and V [width][4][256], BF16, by position.
__global__ void chunk_inputs(__nv_bfloat16 *q, __nv_bfloat16 *k, __nv_bfloat16 *v, int start, int width) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t nq = static_cast<std::int64_t>(width) * kQHeads * kHeadDim;
  if (i < nq) {
    const std::int64_t pos = start + i / (kQHeads * kHeadDim);
    q[i] = __float2bfloat16(gauss(kStreamQ, static_cast<std::uint64_t>(pos) * kQHeads * kHeadDim + i % (kQHeads * kHeadDim)));
  }
  if (i < static_cast<std::int64_t>(width) * kKvHeads * kHeadDim) {
    const int t = static_cast<int>(i / (kKvHeads * kHeadDim));
    const int head = static_cast<int>((i / kHeadDim) % kKvHeads);
    const int d = static_cast<int>(i % kHeadDim);
    k[i] = kv_value(false, head, start + t, d);
    v[i] = kv_value(true, head, start + t, d);
  }
}

// ---- comparisons on the device -----------------------------------------------------------------

__global__ void count_diff(const std::uint32_t *a, const std::uint32_t *b, std::size_t words,
                           unsigned long long *count) {
  unsigned long long n = 0;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    n += a[i] != b[i];
  }
  for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
  if ((threadIdx.x & 31) == 0 && n != 0) atomicAdd(count, n);
}
// [0] max |a - b|, [1] sum (a - b)^2, [2] sum b^2, [3] non-finite values in a, as float bits /
// doubles through atomics (max of non-negative floats by their bits).
__global__ void bf16_distance(const __nv_bfloat16 *a, const __nv_bfloat16 *b, std::size_t n, unsigned *maxbits,
                              double *sums, unsigned long long *bad) {
  float mx = 0.0f;
  double se = 0.0, sb = 0.0;
  unsigned long long nb = 0;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const float x = __bfloat162float(a[i]), y = __bfloat162float(b[i]);
    if (!isfinite(x)) {
      ++nb;
      continue;
    }
    const float d = fabsf(x - y);
    mx = fmaxf(mx, d);
    se += static_cast<double>(d) * d;
    sb += static_cast<double>(y) * y;
  }
  atomicMax(maxbits, __float_as_uint(mx));
  atomicAdd(&sums[0], se);
  atomicAdd(&sums[1], sb);
  if (nb != 0) atomicAdd(bad, nb);
}

// ---- MMA peaks: register-only loops, eight independent chains a warp ---------------------------

enum PeakForm : int { kPkS8 = 0, kPkF16F16 = 1, kPkF16F32 = 2, kPkBf16F32 = 3, kPkE4m3 = 4, kPkForms = 5 };
const char *const kPeakName[kPkForms] = {"s8 m16n8k32 -> s32", "f16 m16n8k16 -> f16", "f16 m16n8k16 -> f32",
                                         "bf16 m16n8k16 -> f32", "e4m3 m16n8k32 block-scaled -> f32"};
constexpr double kPeakOps[kPkForms] = {2.0 * 16 * 8 * 32, 2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 16,
                                       2.0 * 16 * 8 * 32};
constexpr int kChains = 8;

template <int FORM>
__global__ void __launch_bounds__(256) peak_loop(int iters, std::uint32_t seed, float *sink, unsigned long long *clocks) {
  const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t a[4], b[2], bn[2];
  for (int i = 0; i < 4; ++i) a[i] = (FORM == kPkS8 || FORM == kPkE4m3) ? ((id * 2654435761u + i + seed) & 0x1F1F1F1Fu) : 0x3C003C00u;
  for (int i = 0; i < 2; ++i) {
    b[i] = (FORM == kPkS8 || FORM == kPkE4m3) ? ((id * 40503u + i + seed) & 0x1F1F1F1Fu) : 0x3C003C00u;
    bn[i] = (FORM == kPkS8) ? (b[i] ^ 0x80808080u) : (FORM == kPkE4m3 ? (b[i] | 0x80808080u) : (b[i] ^ 0x80008000u));
  }
  int iacc[kChains][4] = {};
  float facc[kChains][4] = {};
  std::uint32_t hacc[kChains][2] = {};
  unsigned long long c0 = 0, g0 = 0;
  if (id == 0) {
    c0 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const std::uint32_t *bb = pass == 0 ? b : bn;
#pragma unroll
      for (int c = 0; c < kChains; ++c) {
        if constexpr (FORM == kPkS8) {
          mma_s8(iacc[c], a, bb);
        } else if constexpr (FORM == kPkF16F16) {
          mma_f16acc(hacc[c], a, bb);
        } else if constexpr (FORM == kPkF16F32) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(facc[c][0]), "+f"(facc[c][1]), "+f"(facc[c][2]), "+f"(facc[c][3])
                       : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
        } else if constexpr (FORM == kPkBf16F32) {
          mma_bf16(facc[c], a, bb);
        } else {
          mma_e4m3(facc[c], a, bb, kUnitScale, kUnitScale);
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

struct Rate {
  double tops = 0.0, mhz = 0.0;
};

template <int FORM> Rate peak(int sms) {
  const int blocks = 4 * sms, iters = 2048;
  DeviceBytes sink(static_cast<std::size_t>(blocks) * 256 * 4), clocks(16);
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  float spent = 0.0f, best = 1e30f;
  unsigned long long hc[2] = {0, 0};
  for (int r = 0; spent < 300.0f || r < 8; ++r) {
    BENCH_CUDA(cudaEventRecord(e0));
    peak_loop<FORM><<<blocks, 256>>>(iters, 11u + r, sink.as<float>(), clocks.as<unsigned long long>());
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    spent += ms;
    if (spent > 150.0f && ms < best) {
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

// ---- one chunk's attention through attend(), its passes timed by events ------------------------

enum Cat : int { kCatPrep, kCatWindowTiles, kCatWindowFa, kCatBandTiles, kCatBandFa, kCatMerge, kCats };
const char *const kCatName[kCats] = {"prep", "win tiles", "win FA", "band tiles", "band FA", "merge"};

struct Marks {
  static constexpr int kMax = 64;
  cudaEvent_t ev[kMax];
  int what[kMax];
  int n = 0;
  cudaStream_t stream = nullptr;
  Marks() {
    for (auto &e : ev) BENCH_CUDA(cudaEventCreate(&e));
  }
  ~Marks() {
    for (auto &e : ev) cudaEventDestroy(e);
  }
  static void mark(void *ctx, int what) {
    auto *m = static_cast<Marks *>(ctx);
    if (m->n >= kMax) {
      std::fprintf(stderr, "FATAL: too many passes in one chunk\n");
      std::exit(EXIT_FAILURE);
    }
    BENCH_CUDA(cudaEventRecord(m->ev[m->n], m->stream));
    m->what[m->n++] = what;
  }
  // Milliseconds per category since kMarkBegin.
  void collect(double (&ms)[kCats]) const {
    for (double &x : ms) x = 0.0;
    for (int i = 1; i < n; ++i) {
      float t = 0.0f;
      BENCH_CUDA(cudaEventElapsedTime(&t, ev[i - 1], ev[i]));
      int cat = kCatPrep;
      switch (what[i]) {
        case kMarkPrep: cat = kCatPrep; break;
        case kMarkWindowTiles: cat = kCatWindowTiles; break;
        case kMarkWindowFa: cat = kCatWindowFa; break;
        case kMarkBandTiles: cat = kCatBandTiles; break;
        case kMarkBandFa: cat = kCatBandFa; break;
        default: cat = kCatMerge; break;
      }
      ms[cat] += t;
    }
  }
};

struct Run {
  Band band;
  int variant = kI8Default;  // the i8 band's
  int diag = 0;
  int splits = 0;  // 0: the op's pick
};

struct Bench {
  int width = 4096;
  int max_ctx = 65536;
  Cache cache;
  DeviceBytes q, k, v, out, out_ref, q_band, q_scale, area, band_scale, carry_o, carry_m, carry_l;
  DeviceBytes carry_o_ref, carry_m_ref, carry_l_ref, counter, dist_max, dist_sums;
  Marks marks;
  cudaStream_t stream = nullptr;

  void init(int ctx, int chunk) {
    width = chunk;
    max_ctx = ctx;
    BENCH_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    marks.stream = stream;
    cache.pages = (ctx + kPageTokens - 1) / kPageTokens;
    const std::size_t rows = static_cast<std::size_t>(cache.pages) * kPageTokens * kKvHeads;
    cache.codes_k.alloc(rows * ninfer::ops::kHqCodePlaneExtent);
    cache.codes_v.alloc(rows * ninfer::ops::kHqCodePlaneExtent);
    cache.meta_k.alloc(rows * ninfer::ops::kHqMetaPlaneExtent);
    cache.meta_v.alloc(rows * ninfer::ops::kHqMetaPlaneExtent);
    cache.table.alloc(static_cast<std::size_t>(cache.pages) * 4);
    cache.side_k.alloc(static_cast<std::size_t>(kSideRows) * kKvHeads * kHeadDim * 2);
    cache.side_v.alloc(static_cast<std::size_t>(kSideRows) * kKvHeads * kHeadDim * 2);
    std::vector<std::int32_t> table(static_cast<std::size_t>(cache.pages));
    for (int i = 0; i < cache.pages; ++i) table[static_cast<std::size_t>(i)] = i;
    BENCH_CUDA(cudaMemcpy(cache.table.p, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
    // The whole prompt's keys through the codec, once; the sinks' side rows once.
    for (int at = 0; at < ctx; at += 8192) {
      const int count = std::min(8192, ctx - at);
      encode_rows<<<(count * kKvHeads * 2 + 7) / 8, 256, 0, stream>>>(
          cache.codes_k.as<std::uint8_t>(), cache.codes_v.as<std::uint8_t>(), cache.meta_k.as<std::uint8_t>(),
          cache.meta_v.as<std::uint8_t>(), cache.table.as<std::int32_t>(), at, count);
      BENCH_CUDA(cudaGetLastError());
    }
    side_rows<<<(kSinks * kKvHeads * 2 + 7) / 8, 256, 0, stream>>>(cache.side_k.as<__nv_bfloat16>(),
                                                                  cache.side_v.as<__nv_bfloat16>(), 0, kSinks);
    BENCH_CUDA(cudaGetLastError());
    const std::size_t qe = static_cast<std::size_t>(width) * kQHeads * kHeadDim;
    const std::size_t kve = static_cast<std::size_t>(width) * kKvHeads * kHeadDim;
    q.alloc(qe * 2);
    k.alloc(kve * 2);
    v.alloc(kve * 2);
    out.alloc(qe * 2);
    out_ref.alloc(qe * 2);
    // The workspace, as allocate lays it out, sized for the longest prompt's last chunk.
    const auto visible = static_cast<std::uint32_t>(ctx);
    const std::uint32_t band = band_capacity(visible, kIgnisHqPromptBandKeys, IgnisHqPromptPrecision::kFp8);
    const BandArea a = band_area(width, visible, kIgnisHqPromptBandKeys, window_capacity(width, visible));
    q_band.alloc(static_cast<std::size_t>(2) * kHeadDim * kQHeads * width);
    q_scale.alloc(static_cast<std::size_t>(kQHeads) * width * 4);
    area.alloc(a.bytes);
    band_scale.alloc(static_cast<std::size_t>(band / kTile) * kKvHeads * 2 * 4);
    const std::size_t carry_rows = static_cast<std::size_t>(kQHeads) * width * kIgnisHqPromptMaxSplits;
    carry_o.alloc(carry_rows * kHeadDim * 2);
    carry_m.alloc(carry_rows * 4);
    carry_l.alloc(carry_rows * 4);
    carry_o_ref.alloc(carry_rows * kHeadDim * 2);
    carry_m_ref.alloc(carry_rows * 4);
    carry_l_ref.alloc(carry_rows * 4);
    counter.alloc(8);
    dist_max.alloc(8);
    dist_sums.alloc(32);
    BENCH_CUDA(cudaStreamSynchronize(stream));
  }

  WorkspacePtrs ws() const {
    return WorkspacePtrs{q_band.as<std::uint8_t>(), q_scale.as<float>(), area.as<std::uint8_t>(),
                         band_scale.as<float>(), carry_o.as<__half>(), carry_m.as<float>(), carry_l.as<float>()};
  }

  // The chunk at `start`: the ring as it stood before it, and its own Q, K and V.
  void stage_chunk(int start) {
    const int ring_from = std::max(kSinks, start - kRing);
    if (start > ring_from) {
      const int count = start - ring_from;
      side_rows<<<(count * kKvHeads * 2 + 7) / 8, 256, 0, stream>>>(cache.side_k.as<__nv_bfloat16>(),
                                                                    cache.side_v.as<__nv_bfloat16>(), ring_from, count);
      BENCH_CUDA(cudaGetLastError());
    }
    const std::int64_t n = static_cast<std::int64_t>(width) * kQHeads * kHeadDim;
    chunk_inputs<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        q.as<__nv_bfloat16>(), k.as<__nv_bfloat16>(), v.as<__nv_bfloat16>(), start, width);
    BENCH_CUDA(cudaGetLastError());
  }

  IgnisHqPromptKeys keys(int start) const {
    IgnisHqPromptKeys s;
    s.codes_k = cache.codes_k.p;
    s.codes_v = cache.codes_v.p;
    s.meta_k = cache.meta_k.p;
    s.meta_v = cache.meta_v.p;
    s.block_table = cache.table.as<std::int32_t>();
    s.residual_k = cache.side_k.p;
    s.residual_v = cache.side_v.p;
    s.ring_valid = nullptr;  // every ring slot holds the key the chunk names
    s.new_k = k.p;
    s.new_v = v.p;
    s.start = start;
    s.width = width;
    return s;
  }

  // One chunk (already staged), timed per pass; returns the ms per category.
  void attend_chunk(int start, const Run &run, double (&ms)[kCats], bool timed) {
    marks.n = 0;
    const PassHook hook{&Marks::mark, &marks};
    attend(q.as<__nv_bfloat16>(), keys(start), true, start, width, 1.0f / 16.0f, run.band, kIgnisHqPromptBandKeys,
           run.splits, ws(), out.as<__nv_bfloat16>(), stream, timed ? &hook : nullptr, run.variant, run.diag);
    BENCH_CUDA(cudaStreamSynchronize(stream));
    if (timed) {
      marks.collect(ms);
    } else {
      for (double &x : ms) x = 0.0;
    }
  }

  int splits_of(int start, int splits_option) const {
    return splits_option > 0 ? splits_option : pick_splits(width, static_cast<std::uint32_t>(start + width));
  }

  unsigned long long differ(const DeviceBytes &a, const DeviceBytes &b, std::size_t bytes) {
    BENCH_CUDA(cudaMemsetAsync(counter.p, 0, 8, stream));
    count_diff<<<1024, 256, 0, stream>>>(a.as<std::uint32_t>(), b.as<std::uint32_t>(), bytes / 4,
                                         counter.as<unsigned long long>());
    BENCH_CUDA(cudaGetLastError());
    unsigned long long n = 0;
    BENCH_CUDA(cudaMemcpyAsync(&n, counter.p, 8, cudaMemcpyDeviceToHost, stream));
    BENCH_CUDA(cudaStreamSynchronize(stream));
    return n;
  }

  // max |out - out_ref| and the RMS of the difference over the RMS of out_ref.
  void distance(double *maxd, double *rel, unsigned long long *bad) {
    BENCH_CUDA(cudaMemsetAsync(dist_max.p, 0, 8, stream));
    BENCH_CUDA(cudaMemsetAsync(dist_sums.p, 0, 32, stream));
    BENCH_CUDA(cudaMemsetAsync(counter.p, 0, 8, stream));
    const std::size_t n = static_cast<std::size_t>(width) * kQHeads * kHeadDim;
    bf16_distance<<<1024, 256, 0, stream>>>(out.as<__nv_bfloat16>(), out_ref.as<__nv_bfloat16>(), n,
                                            dist_max.as<unsigned>(), dist_sums.as<double>(),
                                            counter.as<unsigned long long>());
    BENCH_CUDA(cudaGetLastError());
    unsigned mb = 0;
    double sums[2] = {0, 0};
    BENCH_CUDA(cudaMemcpyAsync(&mb, dist_max.p, 4, cudaMemcpyDeviceToHost, stream));
    BENCH_CUDA(cudaMemcpyAsync(sums, dist_sums.p, 16, cudaMemcpyDeviceToHost, stream));
    BENCH_CUDA(cudaMemcpyAsync(bad, counter.p, 8, cudaMemcpyDeviceToHost, stream));
    BENCH_CUDA(cudaStreamSynchronize(stream));
    float f;
    std::memcpy(&f, &mb, 4);
    *maxd = f;
    *rel = sums[1] > 0 ? std::sqrt(sums[0] / sums[1]) : 0.0;
  }

  void keep_reference(int splits) {
    const std::size_t qe = static_cast<std::size_t>(width) * kQHeads * kHeadDim;
    const std::size_t rows = static_cast<std::size_t>(kQHeads) * width * splits;
    BENCH_CUDA(cudaMemcpyAsync(out_ref.p, out.p, qe * 2, cudaMemcpyDeviceToDevice, stream));
    BENCH_CUDA(cudaMemcpyAsync(carry_o_ref.p, carry_o.p, rows * kHeadDim * 2, cudaMemcpyDeviceToDevice, stream));
    BENCH_CUDA(cudaMemcpyAsync(carry_m_ref.p, carry_m.p, rows * 4, cudaMemcpyDeviceToDevice, stream));
    BENCH_CUDA(cudaMemcpyAsync(carry_l_ref.p, carry_l.p, rows * 4, cudaMemcpyDeviceToDevice, stream));
    BENCH_CUDA(cudaStreamSynchronize(stream));
  }
  void clear_state() {
    const std::size_t qe = static_cast<std::size_t>(width) * kQHeads * kHeadDim;
    const std::size_t rows = static_cast<std::size_t>(kQHeads) * width * kIgnisHqPromptMaxSplits;
    BENCH_CUDA(cudaMemsetAsync(out.p, 0xA5, qe * 2, stream));
    BENCH_CUDA(cudaMemsetAsync(carry_o.p, 0xA5, rows * kHeadDim * 2, stream));
    BENCH_CUDA(cudaMemsetAsync(carry_m.p, 0xA5, rows * 4, stream));
    BENCH_CUDA(cudaMemsetAsync(carry_l.p, 0xA5, rows * 4, stream));
  }
};

// The (query, key) pairs a chunk attends, per pass, as attend() splits them.
struct Pairs {
  double window = 0, band = 0;
  double total() const { return window + band; }
};
Pairs chunk_pairs(int start, int width) {
  Pairs p;
  const int visible = start + width;
  const int sinks = std::min(kSinks, visible);
  const int ring_from = std::max(kSinks, start - kRing);
  for (int t = 0; t < width; ++t) {
    const int pos = start + t;
    p.window += std::min(sinks, pos + 1) + std::max(0, pos + 1 - ring_from);
  }
  const int codec = std::max(0, std::min(ring_from, visible) - sinks);
  p.band = static_cast<double>(width) * codec;
  return p;
}
// Multiply-adds of Q K^T, and as many of P V, for `pairs` pairs over the 24 heads of 256.
double macs(double pairs) { return pairs * kQHeads * kHeadDim; }

const char *band_name(Band b) {
  switch (b) {
    case Band::kFp8: return "fp8";
    case Band::kFp16: return "fp16";
    case Band::kI8: return "i8";
    case Band::kI8Ref: return "i8ref";
    default: return "i8qk";
  }
}

std::vector<int> parse_list(const std::string &s) {
  std::vector<int> v;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t comma = s.find(',', pos);
    v.push_back(std::atoi(s.substr(pos, comma - pos).c_str()));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return v;
}
std::vector<double> parse_doubles(const std::string &s) {
  std::vector<double> v;
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t comma = s.find(',', pos);
    v.push_back(std::atof(s.substr(pos, comma - pos).c_str()));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return v;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

void kernel_report(const char *name, const void *f, int smem, int threads = 256) {
  cudaFuncAttributes at{};
  BENCH_CUDA(cudaFuncGetAttributes(&at, f));
  if (smem > 48 * 1024) BENCH_CUDA(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  int ctas = 0;
  BENCH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, f, threads, smem));
  std::printf("  %-52s %3d regs, %4zu B local (spills), %6d B dynamic + %5zu B static smem, %d CTA/SM\n", name,
              at.numRegs, at.localSizeBytes, smem, at.sharedSizeBytes, ctas);
}

template <int I> void report_variant() {
  constexpr I8Variant v = kI8Variants[I];
  char name[96];
  std::snprintf(name, sizeof(name), "fa_i8_kernel v%d <%d,%d,%d,%d,%d,%d,%d,%d>", I, v.stages, v.one_barrier,
                v.pipe_qk, v.q_smem, v.ping_pong, v.pad, v.fuse_qk, v.one_head);
  kernel_report(name,
                reinterpret_cast<const void *>(fa_i8_kernel<v.stages, v.one_barrier, v.pipe_qk, v.q_smem, v.ping_pong, v.pad, v.fuse_qk, v.one_head>),
                kI8SmemBytes<v.stages, v.q_smem, v.pad>, v.one_head ? 128 : 256);
}
template <int kDiag> void report_diag(const char *what) {
  constexpr I8Variant v = kI8Variants[kI8Default];
  char name[96];
  std::snprintf(name, sizeof(name), "fa_i8_kernel default, %s", what);
  kernel_report(
      name, reinterpret_cast<const void *>(fa_i8_kernel<v.stages, v.one_barrier, v.pipe_qk, v.q_smem, v.ping_pong, v.pad, v.fuse_qk, v.one_head, kDiag>),
      kI8SmemBytes<v.stages, v.q_smem, v.pad>, v.one_head ? 128 : 256);
}

}  // namespace bench

int main(int argc, char **argv) {
  using namespace bench;
  std::vector<int> prompts = {8192, 65536};
  int chunk = 4096, rounds = 5;
  bool diag = false, check = true, variants = false;
  int profile = -100, profile_start = -1;  // --profile V START: one chunk of i8 variant V (-1: i8ref, -2: fp16, -3: fp8)
  std::vector<double> nominal;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--prompts") prompts = parse_list(next());
    else if (a == "--chunk") chunk = std::atoi(next().c_str());
    else if (a == "--rounds") rounds = std::atoi(next().c_str());
    else if (a == "--diag") diag = true;
    else if (a == "--no-check") check = false;
    else if (a == "--variants") variants = true;
    else if (a == "--peaks") nominal = parse_doubles(next());
    else if (a == "--profile") {
      profile = std::atoi(next().c_str());
      profile_start = std::atoi(next().c_str());
    }
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  int max_ctx = 0;
  for (int p : prompts) {
    if (p < chunk || p % chunk != 0) {
      std::fprintf(stderr, "every prompt must be a positive multiple of --chunk (%d)\n", chunk);
      return 2;
    }
    max_ctx = std::max(max_ctx, p);
  }
  if (chunk < 64 || chunk % 64 != 0 || rounds < 1) {
    std::fprintf(stderr, "need --chunk a multiple of 64 and --rounds >= 1\n");
    return 2;
  }

  int device = 0, runtime = 0, driver = 0;
  BENCH_CUDA(cudaGetDevice(&device));
  BENCH_CUDA(cudaRuntimeGetVersion(&runtime));
  BENCH_CUDA(cudaDriverGetVersion(&driver));
  if (driver < runtime) {
    std::fprintf(stderr, "the driver supports CUDA %d.%d; this bench needs %d.%d or newer\n", driver / 1000,
                 (driver % 1000) / 10, runtime / 1000, (runtime % 1000) / 10);
    return 2;
  }
  int major = 0, minor = 0, sms = 0, l2 = 0, smem_optin = 0;
  BENCH_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  BENCH_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  BENCH_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  BENCH_CUDA(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, device));
  BENCH_CUDA(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  std::size_t free_b = 0, total_b = 0;
  BENCH_CUDA(cudaMemGetInfo(&free_b, &total_b));
  const bool has_fp8 = major >= 12;
  std::printf("sm8x prompt attention, phase 2d (prototype, GitHub #474)\n");
  std::printf("device: compute capability %d.%d, %d SMs, L2 %.1f MiB, %d B shared memory per block (opt-in), "
              "%.1f of %.1f GiB free, runtime %d.%d, driver %d.%d\n",
              major, minor, sms, l2 / 1048576.0, smem_optin, free_b / 1073741824.0, total_b / 1073741824.0,
              runtime / 1000, (runtime % 1000) / 10, driver / 1000, (driver % 1000) / 10);
  std::printf("geometry: %d query heads over %d KV heads of %d, %d GQA layers; chunk %d; band %u keys (fp16: %u); "
              "sinks %d, ring %d\n",
              kQHeads, kKvHeads, kHeadDim, kLayers, chunk, kIgnisHqPromptBandKeys, kIgnisHqPromptBandKeys / 2, kSinks,
              kRing);
  std::printf("bands: fp16%s, i8ref, i8 (variant %d: %s)\n\n", has_fp8 ? ", fp8" : " (fp8 needs compute capability 12.0)",
              kI8Default, kI8Variants[kI8Default].name);

  // ---- kernels ----
  std::printf("== kernels\n");
  kernel_report("fa_kernel<bf16> (the window)", reinterpret_cast<const void *>(fa_kernel<false>), kFaSmemBytesP<false, kProtoOff>);
  kernel_report("fa_kernel<fp16> (the FP16 band)", reinterpret_cast<const void *>(fa_kernel<false, true>),
                kFaSmemBytesP<false, kProtoOff>);
  kernel_report("fa_kernel<fp8> (the 8-bit band)", reinterpret_cast<const void *>(fa_kernel<true>), kFaSmemBytesP<true, kProtoOff>);
  kernel_report("fa_kernel<i8> (i8ref, 5ea54bc6)", reinterpret_cast<const void *>(fa_kernel<true, false, kProtoI8>),
                kFaSmemBytesP<true, kProtoI8>);
  report_variant<0>();
  report_variant<1>();
  report_variant<2>();
  report_variant<3>();
  report_variant<4>();
  report_variant<5>();
  report_variant<6>();
  report_variant<7>();
  report_variant<8>();
  report_variant<9>();
  report_variant<10>();
  report_variant<11>();
  report_variant<12>();
  report_variant<13>();
  report_variant<14>();
  report_variant<15>();
  report_variant<16>();
  report_variant<17>();
  report_variant<18>();
  report_variant<19>();
  report_variant<20>();
  report_variant<21>();
  report_variant<22>();
  report_variant<23>();
  static_assert(kI8VariantCount == 24, "the kernel report names every variant");
  report_diag<kDiagNoPv>("no P V");
  report_diag<kDiagNoPv | kDiagNoSoftmax>("Q K^T only");
  report_diag<kDiagNoQk>("no Q K^T");
  report_diag<kDiagNoPv | kDiagNoSoftmax | kDiagNoQk>("loads only");
  kernel_report("tile_kernel<i8> (band tiles, i8 and i8ref)", reinterpret_cast<const void *>(tile_kernel<true, false, kProtoI8>), 0);
  kernel_report("tile_kernel<fp16>", reinterpret_cast<const void *>(tile_kernel<false, true>), 0);
  kernel_report("tile_kernel<bf16> (window tiles)", reinterpret_cast<const void *>(tile_kernel<false>), 0);
  std::printf("\n");

  // ---- peaks ----
  std::printf("== peaks (register-only MMA loops, %d CTAs of 8 warps)\n", 4 * sms);
  Rate rates[kPkForms];
  rates[kPkS8] = peak<kPkS8>(sms);
  rates[kPkF16F16] = peak<kPkF16F16>(sms);
  rates[kPkF16F32] = peak<kPkF16F32>(sms);
  rates[kPkBf16F32] = peak<kPkBf16F32>(sms);
  if (has_fp8) rates[kPkE4m3] = peak<kPkE4m3>(sms);
  for (int f = 0; f < kPkForms; ++f) {
    if (f == kPkE4m3 && !has_fp8) continue;
    std::printf("  %-36s %8.1f TOPS at %.0f MHz\n", kPeakName[f], rates[f].tops, rates[f].mhz);
  }
  // The rates the efficiencies use: measured, or the --peaks the caller names (s8, f16 with f16
  // accumulate, f16 / bf16 with f32 accumulate; the block-scaled E4M3 stays measured).
  double pk_s8 = rates[kPkS8].tops, pk_f16 = rates[kPkF16F16].tops, pk_f32 = rates[kPkBf16F32].tops;
  const double pk_e4m3 = has_fp8 ? rates[kPkE4m3].tops : 0.0;
  if (nominal.size() == 3) {
    pk_s8 = nominal[0];
    pk_f16 = nominal[1];
    pk_f32 = nominal[2];
    std::printf("  efficiencies below against the given peaks: s8 %.0f, f16-f16 %.0f, f16-f32 %.0f TOPS "
                "(e4m3 block-scaled: measured)\n", pk_s8, pk_f16, pk_f32);
  } else {
    std::printf("  efficiencies below against these measured peaks\n");
  }
  std::printf("\n");

  Bench b;
  if (profile > -100) {
    // For a profiler: one chunk at START, twice (the first warms), nothing else.
    b.init(std::max(max_ctx, profile_start + chunk), chunk);
    const Band bd = profile >= 0 ? Band::kI8 : profile == -1 ? Band::kI8Ref : profile == -2 ? Band::kFp16 : Band::kFp8;
    double ms[kCats];
    for (int i = 0; i < 2; ++i) {
      b.stage_chunk(profile_start);
      b.attend_chunk(profile_start, Run{bd, std::max(profile, 0), 0, 0}, ms, true);
    }
    std::printf("profile: %s v%d at start %d: band FA %.3f ms, window FA %.3f ms\n", band_name(bd), profile, profile_start,
                ms[kCatBandFa], ms[kCatWindowFa]);
    return 0;
  }
  b.init(max_ctx, chunk);

  std::vector<Band> bands = {Band::kFp16};
  if (has_fp8) bands.push_back(Band::kFp8);
  bands.push_back(Band::kI8Ref);
  bands.push_back(Band::kI8);
  int failures = 0;
  double dummy[kCats];

  // ---- check ----
  if (check) {
    std::printf("== check: i8 against i8ref, byte for byte (the output and the state carried between passes)\n");
    struct Case {
      int prompt, splits;
    };
    std::vector<Case> cases;
    for (int p : prompts) cases.push_back({p, 0});
    cases.push_back({prompts.front(), 1});
    cases.push_back({prompts.front(), 3});
    for (const Case &cs : cases) {
      const int chunks = cs.prompt / chunk;
      std::vector<unsigned long long> diff_out(kI8VariantCount, 0), diff_carry(kI8VariantCount, 0);
      double worst_rel[2] = {0, 0}, worst_max[2] = {0, 0};
      unsigned long long nonfinite = 0;
      for (int c = 0; c < chunks; ++c) {
        const int start = c * chunk;
        const int splits = b.splits_of(start, cs.splits);
        b.stage_chunk(start);
        b.clear_state();
        b.attend_chunk(start, Run{Band::kI8Ref, kI8Default, 0, cs.splits}, dummy, false);
        b.keep_reference(splits);
        const std::size_t qe = static_cast<std::size_t>(chunk) * kQHeads * kHeadDim;
        const std::size_t rows = static_cast<std::size_t>(kQHeads) * chunk * splits;
        for (int var = 0; var < kI8VariantCount; ++var) {
          b.clear_state();
          b.attend_chunk(start, Run{Band::kI8, var, 0, cs.splits}, dummy, false);
          diff_out[var] += b.differ(b.out, b.out_ref, qe * 2);
          diff_carry[var] += b.differ(b.carry_o, b.carry_o_ref, rows * kHeadDim * 2) +
                             b.differ(b.carry_m, b.carry_m_ref, rows * 4) + b.differ(b.carry_l, b.carry_l_ref, rows * 4);
        }
        // The other bands' distance from i8ref (a broken setup would show here), and i8ref's own
        // finiteness.
        if (cs.splits == 0) {
          int idx = 0;
          for (Band other : {Band::kFp16, Band::kFp8}) {
            if (other == Band::kFp8 && !has_fp8) continue;
            b.clear_state();
            b.attend_chunk(start, Run{other, kI8Default, 0, 0}, dummy, false);
            double mx = 0, rel = 0;
            unsigned long long bad = 0;
            b.distance(&mx, &rel, &bad);
            worst_max[idx] = std::max(worst_max[idx], mx);
            worst_rel[idx] = std::max(worst_rel[idx], rel);
            nonfinite += bad;
            ++idx;
          }
          BENCH_CUDA(cudaMemcpyAsync(b.out.p, b.out_ref.p, qe * 2, cudaMemcpyDeviceToDevice, b.stream));
          double mx = 0, rel = 0;
          unsigned long long bad = 0;
          b.distance(&mx, &rel, &bad);
          nonfinite += bad;
        }
      }
      for (int var = 0; var < kI8VariantCount; ++var) {
        const bool ok = diff_out[var] == 0 && diff_carry[var] == 0;
        failures += ok ? 0 : 1;
        std::printf("check [%s] prompt %6d, splits %s, %2d chunks, i8 v%d: %llu output words and %llu state words differ "
                    "from i8ref\n",
                    ok ? "PASS" : "FAIL", cs.prompt, cs.splits == 0 ? "auto" : std::to_string(cs.splits).c_str(), chunks,
                    var, diff_out[var], diff_carry[var]);
      }
      if (cs.splits == 0) {
        std::printf("  sanity, prompt %d: fp16 against i8ref: max |d| %.4f, RMS(d)/RMS %.4f", cs.prompt, worst_max[0],
                    worst_rel[0]);
        if (has_fp8) std::printf("; fp8 against i8ref: max |d| %.4f, RMS(d)/RMS %.4f", worst_max[1], worst_rel[1]);
        std::printf("; non-finite outputs %llu\n", nonfinite);
        if (nonfinite != 0 || worst_rel[0] > 0.2 || worst_rel[0] == 0.0) {
          std::printf("check [FAIL] the bands disagree too much, or not at all: the setup is broken\n");
          ++failures;
        }
      }
    }
    std::printf("check: %s\n\n", failures == 0 ? "all PASS" : "FAILED");
  }

  // ---- speed: alternated rounds, the median per chunk and pass ----
  struct Series {
    std::string name;
    Run run;
  };
  std::vector<Series> series;
  for (Band bd : bands) series.push_back({band_name(bd), Run{bd, kI8Default, 0, 0}});
  if (variants) {
    for (int var = 0; var < kI8VariantCount; ++var) {
      if (var == kI8Default) continue;
      series.push_back({"i8 v" + std::to_string(var), Run{Band::kI8, var, 0, 0}});
    }
  }
  if (diag) {
    series.push_back({"i8 no P V", Run{Band::kI8, kI8Default, kDiagNoPv, 0}});
    series.push_back({"i8 Q K^T only", Run{Band::kI8, kI8Default, kDiagNoPv | kDiagNoSoftmax, 0}});
    series.push_back({"i8 no Q K^T", Run{Band::kI8, kI8Default, kDiagNoQk, 0}});
    series.push_back({"i8 loads only", Run{Band::kI8, kI8Default, kDiagNoPv | kDiagNoSoftmax | kDiagNoQk, 0}});
    for (int s = 1; s <= kIgnisHqPromptMaxSplits; ++s) {
      series.push_back({"i8 splits " + std::to_string(s), Run{Band::kI8, kI8Default, 0, s}});
      series.push_back({"fp16 splits " + std::to_string(s), Run{Band::kFp16, kI8Default, 0, s}});
    }
  }

  for (int p : prompts) {
    const int chunks = p / chunk;
    // ms[series][chunk][cat], per round
    std::vector<std::vector<std::vector<std::vector<double>>>> t(
        series.size(), std::vector<std::vector<std::vector<double>>>(chunks, std::vector<std::vector<double>>(kCats)));
    // One untimed prompt per series first (the attributes, the caches, the clocks).
    for (std::size_t si = 0; si < series.size(); ++si) {
      for (int c = 0; c < chunks; ++c) {
        b.stage_chunk(c * chunk);
        b.attend_chunk(c * chunk, series[si].run, dummy, false);
      }
    }
    for (int r = 0; r < rounds; ++r) {
      for (std::size_t k = 0; k < series.size(); ++k) {
        // Alternate the order round by round, so a drift of the clocks lands on no one series.
        const std::size_t si = (r % 2 == 0) ? k : series.size() - 1 - k;
        for (int c = 0; c < chunks; ++c) {
          double ms[kCats];
          b.stage_chunk(c * chunk);
          b.attend_chunk(c * chunk, series[si].run, ms, true);
          for (int x = 0; x < kCats; ++x) t[si][c][x].push_back(ms[x]);
        }
      }
    }
    // Medians.
    std::vector<std::vector<std::vector<double>>> med(series.size(), std::vector<std::vector<double>>(chunks, std::vector<double>(kCats)));
    for (std::size_t si = 0; si < series.size(); ++si)
      for (int c = 0; c < chunks; ++c)
        for (int x = 0; x < kCats; ++x) med[si][c][x] = median(t[si][c][x]);

    Pairs total_pairs;
    for (int c = 0; c < chunks; ++c) {
      const Pairs pc = chunk_pairs(c * chunk, chunk);
      total_pairs.window += pc.window;
      total_pairs.band += pc.band;
    }
    const double ops_all = 4.0 * macs(total_pairs.total());  // Q K^T and P V, 2 ops a multiply-add
    const double ops_win = 4.0 * macs(total_pairs.window), ops_band = 4.0 * macs(total_pairs.band);
    auto floor_s = [&](Band bd) {  // the whole prompt's MMA time at the band's own rate, one layer
      switch (bd) {
        case Band::kFp16: return ops_all / (pk_f16 * 1e12);
        case Band::kFp8: return 0.5 * ops_all / (pk_s8 * 1e12) + 0.5 * ops_all / (pk_e4m3 * 1e12);
        default: return ops_all / (pk_s8 * 1e12);
      }
    };
    auto band_floor_s = [&](Band bd) {  // the band pass alone at the band's rate
      switch (bd) {
        case Band::kFp16: return ops_band / (pk_f16 * 1e12);
        case Band::kFp8: return 0.5 * ops_band / (pk_s8 * 1e12) + 0.5 * ops_band / (pk_e4m3 * 1e12);
        default: return ops_band / (pk_s8 * 1e12);
      }
    };

    std::printf("== prompt %d: %d chunks of %d, %.3g pairs a head (window %.3g, band %.3g); per layer %.3g TOPs, "
                "times %d layers\n",
                p, chunks, chunk, total_pairs.total(), total_pairs.window, total_pairs.band, ops_all / 1e12, kLayers);
    // Per chunk, the bands (not the diag series).
    std::printf("per chunk, one layer, ms (median of %d rounds): prep + window tiles + window FA + band tiles + band FA + "
                "merge = total\n", rounds);
    for (std::size_t si = 0; si < bands.size(); ++si) {
      for (int c = 0; c < chunks; ++c) {
        const auto &m = med[si][c];
        double tot = 0;
        for (double x : m) tot += x;
        std::printf("  %-6s chunk %2d (start %6d): %7.3f %7.3f %8.3f %7.3f %8.3f %7.3f = %8.3f\n", series[si].name.c_str(), c,
                    c * chunk, m[kCatPrep], m[kCatWindowTiles], m[kCatWindowFa], m[kCatBandTiles], m[kCatBandFa],
                    m[kCatMerge], tot);
      }
    }
    std::printf("per prompt, %d layers, s:\n", kLayers);
    std::printf("  %-18s %8s %8s %8s %8s %8s %8s | %9s | %8s %6s | %8s %6s | %6s\n", "series", "prep", "w.tiles", "w.FA",
                "b.tiles", "b.FA", "merge", "total", "floor", "eff", "b.floor", "b.eff", "w.eff");
    const double win_floor = ops_win / (pk_f32 * 1e12);
    for (std::size_t si = 0; si < series.size(); ++si) {
      double cat[kCats] = {};
      for (int c = 0; c < chunks; ++c)
        for (int x = 0; x < kCats; ++x) cat[x] += med[si][c][x] * 1e-3 * kLayers;
      double tot = 0;
      for (double x : cat) tot += x;
      const Band bd = series[si].run.band;
      const double fl = floor_s(bd) * kLayers, bfl = band_floor_s(bd) * kLayers;
      std::printf("  %-18s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f | %9.4f | %8.4f %5.1f%% | %8.4f %5.1f%% | %5.1f%%\n",
                  series[si].name.c_str(), cat[kCatPrep], cat[kCatWindowTiles], cat[kCatWindowFa], cat[kCatBandTiles],
                  cat[kCatBandFa], cat[kCatMerge], tot, fl, 100.0 * fl / tot, bfl, 100.0 * bfl / cat[kCatBandFa],
                  100.0 * win_floor * kLayers / cat[kCatWindowFa]);
    }
    std::printf("  floor: the whole prompt's Q K^T and P V at the band's MMA rate (fp16: f16-f16; fp8: s8 and e4m3; "
                "i8: s8); eff = floor / total.\n  b.floor, b.eff: the band pass's own pairs against its band FA time; "
                "w.eff: the window's pairs at f16-f32 against its window FA time.\n\n");
  }
  std::printf("bench: %s\n", failures == 0 ? "done" : "check(s) FAILED");
  return failures == 0 ? 0 : 1;
}
