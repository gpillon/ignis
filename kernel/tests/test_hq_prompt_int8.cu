// GitHub #510 (spec #503): the integer band against an fp64 attention -- OURS, not vendored.
//
// The integer band (IgnisHqPromptPrecision::kInt8, kernel/include/ignis_hq_prompt_attention.h)
// rounds every key it reads to 8 bits, the residual window's included, so no other band can be its
// reference: each brings roundings of its own. This test holds it to the attention itself, computed
// in fp64 on the CUDA cores (no tensor core) over the rows the band reads before it rounds them
// (`ignis_hq_prompt_fetch_rows`, from `after_attention`), within a bound built from each rounding
// the band makes (below), and holds its carried state, read through `after_merge`, to the same
// reference.
//
// The prompt: 65,536 tokens of one GQA layer prefilled at chunk 4,096 through the op itself, on a
// production hq-e8-2b pool (the op's own append fills the cache), with Gaussian rows from fixed
// seeds: K with a mean per KV head, and Q at a standard deviation of 1, 2, 3 and 4 in the four KV
// groups, from near-flat attention over 64K keys to rows a few keys dominate, whose running max
// grows past the lazy rescale's 2^8. It is checked at chunk 0 (the window alone), chunk 1 (the 8K
// prefix: the window and one band) and chunk 15 (the 64K prefix: the window and two bands), on
// three runs:
//   split 1, the default band      the output and the carried state;
//   the band's own split rule      the output (2 splits at 8K, 3 at 64K);
//   split 1, a band of 4,128 keys  the output and the carried state across 16 passes at 64K, the
//                                  band no multiple of the 128-key group (ragged last groups).
// Every output and every carried value of every chunk is also held finite. The window's plane
// (4,640 columns from chunk 1 on) is no multiple of 128 either, in every run.
//
// ---- the bound ---------------------------------------------------------------------------------
// Per query row and head, s_j = c q . k_j with c = log2(e) / 16 (the op's base 2), p_j =
// 2^(s_j - max s), L = sum_j p_j, w_j = p_j / L, O = sum_j w_j v_j the fp64 attention, R the codec's
// rotation (the frame the band quantizes in; orthonormal). The band rounds:
//   - Q per row to INT8, step dq = amax(R q) / 127, and K per 32-key tile, step dk_j = amax / 127
//     over the tile: the score moves by delta_j, |delta_j| <= c (|R k_j|_1 dq + |R q|_1 dk_j +
//     256 dq dk_j / 2) / 2 (Holder; D the largest), with variance at most
//     sigma_j^2 = c^2 (|k_j|^2 dq^2 + |q|^2 dk_j^2 + 256 dq^2 dk_j^2 / 12) / 12;
//   - P to u8, rn(255 p) against the running max, which is at most the final one: in units of the
//     final max, a key rounded to 0 loses all of its p, and any other is off by at most 1/510;
//   - V^T per 128-key group to s8, step dv_j = amax / 127 over the group: variance dv_j^2 / 12 per
//     element, in any frame;
//   - the split states to FP16 (2^-11 of each rotated element) and the output to BF16 (2^-8).
// The window (sinks, ring, chunk) takes the same roundings, on the same kernel. To first order the
// output moves by sum_j w_j e_v,j + sum_j (ln2 w_j delta_j + eps_j / L) (v_j - O). With the scores
// off by at most 6 sigma (sigma the row's largest sigma_j), a weight moves by at most kappa =
// 2^(12 sigma), and the keys split in two:
//   - the small keys, whose u8 could be under 4 (255 p_j / kappa < 4): rounding toward 0 below one
//     half, their eps_j has a sign, at most e_j = min(kappa p_j, 1/510) (Peaked rows: the long tail
//     rounds away, and O moves toward the head's own mean.) Their sum is bounded apart, about
//     their e-weighted mean v~ (independent of the v's): |v~_d - O_d| S / L, S = sum e_j, plus a
//     zero-mean rest of variance sum e_j^2 (v_jd - v~_d)^2 / L^2;
//   - the others, rounded to nearest at 4 units or more: zero-mean, variance 1 / (3 510^2) each.
// The rest is a sum of independent zero-mean terms (a delta common to every key, as Q's error
// against K's mean, moves no weight), of standard deviation per element at most
//   sigma_d^2 = sum_j w_j^2 dv_j^2 / 12 + sum_j ln2^2 w_j^2 sigma_j^2 (v_jd - O_d)^2
//               + sum_others (v_jd - O_d)^2 / (3 510^2 L^2) + sum_small e_j^2 (v_jd - v~_d)^2 / L^2
//               + (2^-11 |o|_2 / (16 sqrt 3))^2,
// the last term the FP16 state's rounding spread over 256 elements by the un-rotation (|o|_2: the
// stored state's, weighted over the splits). Every element of the output is held to
//   |o_d - O_d| <= 2^-8 |O_d| + (1 + 2^-8) (|v~_d - O_d| S / L + 6 sigma_d + 2^-18 max |v|),
// in the trellis tests' style: the small keys' signed rounding, six standard deviations of a sum
// of many small roundings, the BF16 output's rounding, and room for the fp32 accumulation. The
// carried state at one split: O over l, un-rotated, to the same bound without the BF16 term; the
// max m to D (no statistics: Holder); the log normalizer m + log2(l / 256) to D - log2(1 - x), x =
// kappa (S + 6 sqrt(others / 3) / 510) / L, where x < 1. Where tiles and groups begin is the op's
// business: dk_j and dv_j take the largest amax over every key that could share a tile (32) or a
// group (128) with key j, a step at least the op's, and the window's first group joins the sinks
// to the ring's first 96 keys.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing or busy GPU or a kernel error
// fails this test. It runs on compute capability 8.6 as on 12.0 (no sm120a-only label).

#include "ignis_hq_prompt_attention.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CUDA_FATAL(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));          \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

constexpr int kHeadDim = 256;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kGroup = kQHeads / kKvHeads;  // 6
constexpr int kSinks = 32;
constexpr int kRing = 512;
constexpr int kChunk = 4096;
constexpr int kPrompt = 65536;
constexpr int kChunks = kPrompt / kChunk;
constexpr float kScale = 1.0F / 16.0F;
constexpr double kC = 1.4426950408889634 / 16.0;  // the score's scale in log2 units
constexpr double kSigmas = 6.0;
constexpr double kInf = std::numeric_limits<double>::infinity();
const std::vector<int> kChecked = {0, 1, kChunks - 1};

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) { CUDA_FATAL(cudaMalloc(&p, std::max<std::size_t>(bytes, 16))); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

template <class T>
std::vector<T> download(const void *p, std::size_t n) {
  std::vector<T> out(n);
  CUDA_FATAL(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return out;
}

double bf16(std::uint16_t bits) {
  std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &w, sizeof(f));
  return static_cast<double>(f);
}

double f16(std::uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(bits));
  return static_cast<double>(__half2float(h));
}

// ---- the inputs: Gaussian rows from a counter hash, the same in every run ----------------------

__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// dst [tokens][heads][256] BF16: N(0, sigma_h^2) + mean_h[d], sigma_h = sigma[h / per_sigma].
__global__ void gaussian_rows_kernel(__nv_bfloat16 *dst, std::int64_t n, int heads, std::uint64_t seed,
                                     float sigma0, float sigma_step, int per_sigma, float mean_amp) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int d = static_cast<int>(i % kHeadDim);
  const int h = static_cast<int>((i / kHeadDim) % heads);
  const std::uint64_t r = mix64(seed * 0x100000001B3ull ^ static_cast<std::uint64_t>(i));
  const double u1 = (static_cast<double>(r >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  const double u2 = static_cast<double>(mix64(r) >> 11) * (1.0 / 9007199254740992.0);
  const double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
  const float sigma = sigma0 + sigma_step * static_cast<float>(h / per_sigma);
  // A fixed mean per head and dimension, from its own hash.
  const std::uint64_t mh = mix64(0xC0FFEEull + static_cast<std::uint64_t>(h * kHeadDim + d));
  const float mean = mean_amp * (static_cast<float>(mh >> 40) * (1.0F / 16777216.0F) * 2.0F - 1.0F);
  dst[i] = __float2bfloat16(static_cast<float>(z) * sigma + mean);
}

void gaussian_rows(void *dst, int tokens, int heads, std::uint64_t seed, float sigma0, float sigma_step,
                   int per_sigma, float mean_amp) {
  const std::int64_t n = static_cast<std::int64_t>(tokens) * heads * kHeadDim;
  gaussian_rows_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
      static_cast<__nv_bfloat16 *>(dst), n, heads, seed, sigma0, sigma_step, per_sigma, mean_amp);
  CUDA_FATAL(cudaGetLastError());
}

// ---- the codec's rotation, restated in double ------------------------------------------------

// The codec's per-coordinate sign (kernel/vendor/src/ops/kernel/hq_codec.cuh `hq_engine_sign`),
// as test_attention_readout.cu restates it.
double codec_sign(int d) {
  std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  return (x & 1u) ? 1.0 : -1.0;
}

void hadamard(double *x) {
  for (int len = 1; len < kHeadDim; len <<= 1) {
    for (int base = 0; base < kHeadDim; base += 2 * len) {
      for (int i = base; i < base + len; ++i) {
        const double a = x[i], b = x[i + len];
        x[i] = a + b;
        x[i + len] = a - b;
      }
    }
  }
}
// R v = H diag(signs) v / 16, and its inverse diag(signs) H y / 16.
void rotate(double *x) {
  for (int d = 0; d < kHeadDim; ++d) x[d] *= codec_sign(d);
  hadamard(x);
  for (int d = 0; d < kHeadDim; ++d) x[d] /= 16.0;
}
void unrotate(double *x) {
  hadamard(x);
  for (int d = 0; d < kHeadDim; ++d) x[d] *= codec_sign(d) / 16.0;
}

template <class F>
void parallel_for(std::size_t n, F &&f) {
  const unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      for (std::size_t i = t; i < n; i += threads) f(i);
    });
  }
  for (auto &th : pool) th.join();
}

// ---- the fp64 reference ----------------------------------------------------------------------

constexpr int kRefRows = 4;                    // sampled query rows a CTA
constexpr int kRefHeads = kRefRows * kGroup;   // query heads a CTA, one KV group: 24
constexpr int kRefKeys = 8;                    // keys a block
constexpr int kRowPad = kHeadDim + 1;
// A key whose u8 could be under this many units is a small key: its rounding has a sign (below
// 0.5 it is the whole of 255 p, rounded down), so the bound takes it apart from the rest.
constexpr double kSmallBytes = 4.0;

struct RefIn {
  const float *q;        // [rows][24][256], the original frame
  const double *q_stat;  // [rows][24][3]: dq, |R q|_1, |q|_2^2
  const int *pos;        // [rows]
  const float *k;        // [4][visible][256], the original frame
  const float *v;
  const float *k_stat;   // [4][visible][4]: |R k|_1, |k|_2^2, dk, dv
  int visible;
};
struct RefOut {
  double *m, *L, *D, *A, *S, *n_out, *n, *sd;  // [rows][24]
  double *O, *B, *Vbar, *C;               // [rows][24][256]
};

// One KV head and kRefRows sampled rows a CTA, one thread a dimension. Pass 1: the online softmax
// in fp64 (m, L, O), D and the largest score deviation; passes 2 and 3, at the final m and L, the
// bound's sums: A, B, the small keys' S and their mean Vbar (pass 2), and their C (pass 3).
__global__ void __launch_bounds__(256) reference_kernel(RefIn in, RefOut out) {
  __shared__ float qs[kRefHeads][kRowPad];
  __shared__ float kb[kRefKeys][kRowPad];
  __shared__ float vb[kRefKeys][kHeadDim];
  __shared__ float kst[kRefKeys][4];
  __shared__ double sb[kRefHeads][kRefKeys];
  __shared__ double xb[kRefHeads][kRefKeys];
  __shared__ double eb[kRefHeads][kRefKeys];
  __shared__ double qst[kRefHeads][3];
  __shared__ double m_s[kRefHeads], l_s[kRefHeads], c_s[kRefHeads], d_s[kRefHeads], sd_s[kRefHeads],
      s_s[kRefHeads];
  __shared__ int pos_s[kRefRows];
  const int g = static_cast<int>(blockIdx.y);
  const int row0 = static_cast<int>(blockIdx.x) * kRefRows;
  const int tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < kRefHeads * kHeadDim; i += 256) {
    const int qh = i / kHeadDim, d = i % kHeadDim;
    const int row = row0 + qh / kGroup, h = g * kGroup + qh % kGroup;
    qs[qh][d] = in.q[(static_cast<std::int64_t>(row) * kQHeads + h) * kHeadDim + d];
  }
  if (tid < kRefHeads) {
    const int row = row0 + tid / kGroup, h = g * kGroup + tid % kGroup;
    for (int s = 0; s < 3; ++s) qst[tid][s] = in.q_stat[(static_cast<std::int64_t>(row) * kQHeads + h) * 3 + s];
    m_s[tid] = -CUDART_INF;
    l_s[tid] = 0.0;
    d_s[tid] = 0.0;
    sd_s[tid] = 0.0;
    s_s[tid] = 0.0;
  }
  if (tid < kRefRows) pos_s[tid] = in.pos[row0 + tid];
  __syncthreads();
  int last = 0;
  for (int r = 0; r < kRefRows; ++r) last = max(last, pos_s[r]);
  const int keys = min(last + 1, in.visible);
  const std::int64_t head_rows = static_cast<std::int64_t>(g) * in.visible;
  const double ln2 = 0.6931471805599453;

  auto load = [&](int b) {
    for (int i = tid; i < kRefKeys * kHeadDim; i += 256) {
      const int j = i / kHeadDim, d = i % kHeadDim, key = b + j;
      const bool ok = key < keys;
      kb[j][d] = ok ? in.k[(head_rows + key) * kHeadDim + d] : 0.0F;
      vb[j][d] = ok ? in.v[(head_rows + key) * kHeadDim + d] : 0.0F;
    }
    if (tid < kRefKeys * 4) {
      const int j = tid / 4, key = b + j;
      kst[j][tid % 4] = key < keys ? in.k_stat[(head_rows + key) * 4 + tid % 4] : 0.0F;
    }
  };
  // The variance of key j's score deviation for query head qh, log2 units.
  auto var_delta = [&](int qh, int j) {
    const double dq = qst[qh][0], dk = kst[j][2];
    return kC * kC * (static_cast<double>(kst[j][1]) * dq * dq + qst[qh][2] * dk * dk + 256.0 * dq * dq * dk * dk / 12.0) / 12.0;
  };
  // The score of pair tid (query head tid / kRefKeys, key b + tid % kRefKeys), -inf past the row,
  // and its deviation's Holder bound.
  auto score = [&](int b, double &worst) {
    const int qh = tid / kRefKeys, j = tid % kRefKeys, key = b + j;
    worst = 0.0;
    if (key >= keys || key > pos_s[qh / kGroup]) return -CUDART_INF;
    double acc = 0.0;
    for (int d = 0; d < kHeadDim; ++d) acc += static_cast<double>(qs[qh][d]) * static_cast<double>(kb[j][d]);
    const double dq = qst[qh][0], dk = kst[j][2];
    worst = kC * (static_cast<double>(kst[j][0]) * dq / 2.0 + qst[qh][1] * dk / 2.0 + 256.0 * dq * dk / 4.0) +
            ldexp(kC * sqrt(qst[qh][2] * static_cast<double>(kst[j][1])) + 1.0, -20);
    return kC * acc;
  };

  // Pass 1.
  double o[kRefHeads];
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) o[qh] = 0.0;
  for (int b = 0; b < keys; b += kRefKeys) {
    load(b);
    __syncthreads();
    if (tid < kRefHeads * kRefKeys) {
      double worst;
      sb[tid / kRefKeys][tid % kRefKeys] = score(b, worst);
      xb[tid / kRefKeys][tid % kRefKeys] = worst;
    }
    __syncthreads();
    if (tid < kRefHeads) {
      double mx = m_s[tid], dmax = d_s[tid], sdmax = sd_s[tid];
      for (int j = 0; j < kRefKeys; ++j) {
        mx = fmax(mx, sb[tid][j]);
        if (sb[tid][j] != -CUDART_INF) {
          dmax = fmax(dmax, xb[tid][j]);
          sdmax = fmax(sdmax, sqrt(var_delta(tid, j)));
        }
      }
      const double corr = m_s[tid] == -CUDART_INF ? 0.0 : exp2(m_s[tid] - mx);
      double l = l_s[tid] * corr;
      for (int j = 0; j < kRefKeys; ++j) {
        const double p = sb[tid][j] == -CUDART_INF ? 0.0 : exp2(sb[tid][j] - mx);
        sb[tid][j] = p;
        l += p;
      }
      m_s[tid] = mx;
      l_s[tid] = l;
      c_s[tid] = corr;
      d_s[tid] = dmax;
      sd_s[tid] = sdmax;
    }
    __syncthreads();
#pragma unroll
    for (int qh = 0; qh < kRefHeads; ++qh) {
      double acc = o[qh] * c_s[qh];
      for (int j = 0; j < kRefKeys; ++j) acc += sb[qh][j] * static_cast<double>(vb[j][tid]);
      o[qh] = acc;
    }
    __syncthreads();
  }
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) o[qh] /= l_s[qh];

  // Pass 2. kappa: how far a weight can move with the scores, 2^(12 sigma), sigma the largest
  // score deviation's standard deviation; a key whose u8 could be below kSmallBytes is small.
  double bsum[kRefHeads], vsum[kRefHeads];
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) bsum[qh] = vsum[qh] = 0.0;
  double a_acc = 0.0, n_out = 0.0, n_acc = 0.0;  // thread tid < kRefHeads: its query head's
  for (int b = 0; b < keys; b += kRefKeys) {
    load(b);
    __syncthreads();
    if (tid < kRefHeads * kRefKeys) {
      double worst;
      sb[tid / kRefKeys][tid % kRefKeys] = score(b, worst);
    }
    __syncthreads();
    if (tid < kRefHeads) {
      const double m = m_s[tid], L = l_s[tid], kappa = exp2(12.0 * sd_s[tid]);
      double small = 0.0;
      for (int j = 0; j < kRefKeys; ++j) {
        const double s = sb[tid][j];
        xb[tid][j] = 0.0;
        eb[tid][j] = 0.0;
        if (s == -CUDART_INF) continue;
        const double p = exp2(s - m), w = p / L, dv = kst[j][3];
        a_acc += w * w * dv * dv / 12.0;
        n_acc += 1.0;
        double a = ln2 * ln2 * w * w * var_delta(tid, j);
        if (255.0 * p / kappa < kSmallBytes) {
          eb[tid][j] = fmin(kappa * p, 1.0 / 510.0);
          small += eb[tid][j];
        } else {
          a += 1.0 / (3.0 * 510.0 * 510.0) / (L * L);
          n_out += 1.0;
        }
        xb[tid][j] = a;
      }
      s_s[tid] += small;
    }
    __syncthreads();
#pragma unroll
    for (int qh = 0; qh < kRefHeads; ++qh) {
      double acc = bsum[qh], vacc = vsum[qh];
      for (int j = 0; j < kRefKeys; ++j) {
        const double v = static_cast<double>(vb[j][tid]);
        acc += xb[qh][j] * (v - o[qh]) * (v - o[qh]);
        vacc += eb[qh][j] * v;
      }
      bsum[qh] = acc;
      vsum[qh] = vacc;
    }
    __syncthreads();
  }
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) {
    const int row = row0 + qh / kGroup, h = g * kGroup + qh % kGroup;
    const std::int64_t at = (static_cast<std::int64_t>(row) * kQHeads + h) * kHeadDim + tid;
    out.O[at] = o[qh];
    out.B[at] = bsum[qh];
    vsum[qh] = s_s[qh] > 0.0 ? vsum[qh] / s_s[qh] : 0.0;  // the small keys' mean
    out.Vbar[at] = vsum[qh];
  }

  // Pass 3: the small keys about their mean.
  double csum[kRefHeads];
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) csum[qh] = 0.0;
  for (int b = 0; b < keys; b += kRefKeys) {
    load(b);
    __syncthreads();
    if (tid < kRefHeads * kRefKeys) {
      double worst;
      sb[tid / kRefKeys][tid % kRefKeys] = score(b, worst);
    }
    __syncthreads();
    if (tid < kRefHeads) {
      const double m = m_s[tid], kappa = exp2(12.0 * sd_s[tid]);
      for (int j = 0; j < kRefKeys; ++j) {
        const double s = sb[tid][j];
        const double p = s == -CUDART_INF ? 0.0 : exp2(s - m);
        eb[tid][j] = s != -CUDART_INF && 255.0 * p / kappa < kSmallBytes ? fmin(kappa * p, 1.0 / 510.0) : 0.0;
      }
    }
    __syncthreads();
#pragma unroll
    for (int qh = 0; qh < kRefHeads; ++qh) {
      double acc = csum[qh];
      for (int j = 0; j < kRefKeys; ++j) {
        const double dv = static_cast<double>(vb[j][tid]) - vsum[qh];
        acc += eb[qh][j] * eb[qh][j] * dv * dv;
      }
      csum[qh] = acc;
    }
    __syncthreads();
  }
#pragma unroll
  for (int qh = 0; qh < kRefHeads; ++qh) {
    const int row = row0 + qh / kGroup, h = g * kGroup + qh % kGroup;
    out.C[(static_cast<std::int64_t>(row) * kQHeads + h) * kHeadDim + tid] = csum[qh];
  }
  if (tid < kRefHeads) {
    const int row = row0 + tid / kGroup, h = g * kGroup + tid % kGroup;
    const std::int64_t at = static_cast<std::int64_t>(row) * kQHeads + h;
    out.m[at] = m_s[tid];
    out.L[at] = l_s[tid];
    out.D[at] = d_s[tid];
    out.A[at] = a_acc;
    out.S[at] = s_s[tid];
    out.n_out[at] = n_out;
    out.n[at] = n_acc;
    out.sd[at] = sd_s[tid];
  }
}

// ---- one checked chunk -----------------------------------------------------------------------

// The query rows held to the reference: the chunk's first and last 64-token tiles (every warp
// and both heads of their CTAs) and 64 rows across the middle.
std::vector<int> sampled_rows() {
  std::vector<int> rows;
  for (int r = 0; r < 64; ++r) rows.push_back(r);
  for (int k = 0; k < 64; ++k) rows.push_back(64 + 61 * k);
  for (int r = kChunk - 64; r < kChunk; ++r) rows.push_back(r);
  std::sort(rows.begin(), rows.end());
  return rows;
}

struct Reference {
  std::vector<double> m, L, D, A, S, n_out, n, sd, O, B, Vbar, C;
  double vmax = 0.0;  // the largest |v| element of the visible keys
};

// The reference for `rows` of the chunk at `start` over the rows the band read (`fetched_k`,
// `fetched_v`: BF16 [4][visible][256], the rotated frame) and the chunk's queries (`q`, BF16
// [width][24][256]).
Reference reference(const std::vector<int> &rows, int start, int visible, const void *q_dev,
                    const void *fetched_k, const void *fetched_v) {
  const std::size_t plane = static_cast<std::size_t>(kKvHeads) * visible * kHeadDim;
  const std::vector<std::uint16_t> hk = download<std::uint16_t>(fetched_k, plane);
  const std::vector<std::uint16_t> hv = download<std::uint16_t>(fetched_v, plane);
  std::vector<float> fk(plane), fv(plane);
  std::vector<double> amax_k(static_cast<std::size_t>(kKvHeads) * visible),
      amax_v(amax_k.size()), l1_k(amax_k.size()), l2_k(amax_k.size());
  std::vector<double> vmax_part(amax_k.size());
  parallel_for(amax_k.size(), [&](std::size_t r) {
    double x[kHeadDim], y[kHeadDim];
    double ak = 0.0, av = 0.0, l1 = 0.0, l2 = 0.0, vm = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
      x[d] = bf16(hk[r * kHeadDim + d]);
      y[d] = bf16(hv[r * kHeadDim + d]);
      ak = std::max(ak, std::fabs(x[d]));
      av = std::max(av, std::fabs(y[d]));
      l1 += std::fabs(x[d]);
      l2 += x[d] * x[d];
    }
    unrotate(x);
    unrotate(y);
    for (int d = 0; d < kHeadDim; ++d) {
      fk[r * kHeadDim + d] = static_cast<float>(x[d]);
      fv[r * kHeadDim + d] = static_cast<float>(y[d]);
      vm = std::max(vm, std::fabs(y[d]));
    }
    amax_k[r] = ak;
    amax_v[r] = av;
    l1_k[r] = l1;
    l2_k[r] = l2;
    vmax_part[r] = vm;
  });
  Reference ref;
  for (double vm : vmax_part) ref.vmax = std::max(ref.vmax, vm);

  // The steps: the largest amax over every key that could share key j's tile or group.
  const int ring_from = std::max(kSinks, start - kRing);
  std::vector<float> kstat(static_cast<std::size_t>(kKvHeads) * visible * 4);
  for (int g = 0; g < kKvHeads; ++g) {
    const double *ak = amax_k.data() + static_cast<std::size_t>(g) * visible;
    const double *av = amax_v.data() + static_cast<std::size_t>(g) * visible;
    double joint = 0.0;  // the window's first group: the sinks and the ring's first 96 keys
    for (int i = 0; i < std::min(kSinks, visible); ++i) joint = std::max(joint, av[i]);
    for (int i = ring_from; i < std::min(ring_from + 96, visible); ++i) joint = std::max(joint, av[i]);
    for (int j = 0; j < visible; ++j) {
      double mk = 0.0, mv = 0.0;
      for (int i = std::max(0, j - 31); i <= std::min(visible - 1, j + 31); ++i) mk = std::max(mk, ak[i]);
      for (int i = std::max(0, j - 127); i <= std::min(visible - 1, j + 127); ++i) mv = std::max(mv, av[i]);
      if (j < kSinks || (j >= ring_from && j < ring_from + 96)) mv = std::max(mv, joint);
      float *st = kstat.data() + (static_cast<std::size_t>(g) * visible + j) * 4;
      st[0] = static_cast<float>(l1_k[static_cast<std::size_t>(g) * visible + j]);
      st[1] = static_cast<float>(l2_k[static_cast<std::size_t>(g) * visible + j]);
      st[2] = static_cast<float>(mk / 127.0);
      st[3] = static_cast<float>(mv / 127.0);
    }
  }

  // The sampled queries, the original frame, and their rotated-frame statistics.
  const int nrows = static_cast<int>(rows.size());
  std::vector<float> fq(static_cast<std::size_t>(nrows) * kQHeads * kHeadDim);
  std::vector<double> qstat(static_cast<std::size_t>(nrows) * kQHeads * 3);
  std::vector<int> pos(rows.size());
  for (int r = 0; r < nrows; ++r) {
    pos[static_cast<std::size_t>(r)] = start + rows[static_cast<std::size_t>(r)];
    const std::vector<std::uint16_t> hq = download<std::uint16_t>(
        static_cast<const std::uint16_t *>(q_dev) + static_cast<std::size_t>(rows[static_cast<std::size_t>(r)]) * kQHeads * kHeadDim,
        static_cast<std::size_t>(kQHeads) * kHeadDim);
    for (int h = 0; h < kQHeads; ++h) {
      double x[kHeadDim], l2 = 0.0;
      for (int d = 0; d < kHeadDim; ++d) {
        x[d] = bf16(hq[static_cast<std::size_t>(h) * kHeadDim + d]);
        fq[(static_cast<std::size_t>(r) * kQHeads + h) * kHeadDim + d] = static_cast<float>(x[d]);
        l2 += x[d] * x[d];
      }
      rotate(x);
      double amax = 0.0, l1 = 0.0;
      for (int d = 0; d < kHeadDim; ++d) {
        amax = std::max(amax, std::fabs(x[d]));
        l1 += std::fabs(x[d]);
      }
      double *st = qstat.data() + (static_cast<std::size_t>(r) * kQHeads + h) * 3;
      st[0] = amax / 127.0;
      st[1] = l1;
      st[2] = l2;
    }
  }

  DeviceBytes dq(fq.size() * 4), dqs(qstat.size() * 8), dpos(pos.size() * 4), dk(plane * 4), dv(plane * 4),
      dks(kstat.size() * 4);
  CUDA_FATAL(cudaMemcpy(dq.p, fq.data(), fq.size() * 4, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(dqs.p, qstat.data(), qstat.size() * 8, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(dpos.p, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(dk.p, fk.data(), plane * 4, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(dv.p, fv.data(), plane * 4, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(dks.p, kstat.data(), kstat.size() * 4, cudaMemcpyHostToDevice));
  const std::size_t heads = static_cast<std::size_t>(nrows) * kQHeads;
  DeviceBytes om(heads * 8), oL(heads * 8), oD(heads * 8), oA(heads * 8), oS(heads * 8), onout(heads * 8),
      on(heads * 8), osd(heads * 8), oO(heads * kHeadDim * 8), oB(heads * kHeadDim * 8), oV(heads * kHeadDim * 8),
      oC(heads * kHeadDim * 8);
  const RefIn in{static_cast<const float *>(dq.p), static_cast<const double *>(dqs.p), static_cast<const int *>(dpos.p),
                 static_cast<const float *>(dk.p), static_cast<const float *>(dv.p), static_cast<const float *>(dks.p),
                 visible};
  const RefOut out{static_cast<double *>(om.p), static_cast<double *>(oL.p),    static_cast<double *>(oD.p),
                   static_cast<double *>(oA.p), static_cast<double *>(oS.p),    static_cast<double *>(onout.p),
                   static_cast<double *>(on.p), static_cast<double *>(osd.p),   static_cast<double *>(oO.p),
                   static_cast<double *>(oB.p), static_cast<double *>(oV.p),    static_cast<double *>(oC.p)};
  reference_kernel<<<dim3(static_cast<unsigned>(nrows / kRefRows), kKvHeads), 256>>>(in, out);
  CUDA_FATAL(cudaGetLastError());
  CUDA_FATAL(cudaDeviceSynchronize());
  ref.m = download<double>(om.p, heads);
  ref.L = download<double>(oL.p, heads);
  ref.D = download<double>(oD.p, heads);
  ref.A = download<double>(oA.p, heads);
  ref.S = download<double>(oS.p, heads);
  ref.n_out = download<double>(onout.p, heads);
  ref.n = download<double>(on.p, heads);
  ref.sd = download<double>(osd.p, heads);
  ref.O = download<double>(oO.p, heads * kHeadDim);
  ref.B = download<double>(oB.p, heads * kHeadDim);
  ref.Vbar = download<double>(oV.p, heads * kHeadDim);
  ref.C = download<double>(oC.p, heads * kHeadDim);
  return ref;
}

// How much of a bound a set of values used: the largest |got - want| / bound (below 1 passes),
// the relative RMS error, and how many are past it.
struct Use {
  double worst = 0.0, num = 0.0, den = 0.0;
  std::size_t bad = 0, count = 0;
  std::string where;  // the worst value's place and terms
  template <class Where>
  void add(double got, double want, double bound, Where &&describe) {
    const double err = std::fabs(got - want);
    const double ratio = bound > 0.0 ? err / bound : (err > 0.0 ? kInf : 0.0);
    if (!(ratio <= 1.0)) ++bad;
    if (!(ratio <= worst)) {
      worst = std::isnan(ratio) ? kInf : ratio;
      char line[160];
      std::snprintf(line, sizeof(line), "got %.6g want %.6g bound %.4g: ", got, want, bound);
      where = line + describe();
    }
    num += err * err;
    den += want * want;
    ++count;
  }
  double rel_rms() const { return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num); }
};

void report(const std::string &label, const Use &u) {
  std::printf("  %-52s %8zu values  worst %.3f of the bound  rel RMS %.5f  past %zu\n", label.c_str(), u.count,
              u.worst, u.rel_rms(), u.bad);
  if (u.worst > 0.5) std::printf("    worst at %s\n", u.where.c_str());
  check(u.count > 0 && u.bad == 0, label + ": every value inside the bound");
}

struct Run {
  const char *name;
  std::int32_t splits;
  std::uint32_t band_keys;
  bool carry;
};

void run(const Run &spec) {
  std::printf("[%s] splits %s, band %u keys\n", spec.name,
              spec.splits == 0 ? "by the rule" : std::to_string(spec.splits).c_str(), spec.band_keys);
  constexpr auto page = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  ignis_seq_pool_spec pool_spec{};
  pool_spec.num_kv_heads = kKvHeads;
  pool_spec.head_dim = kHeadDim;
  pool_spec.kv_format = IGNIS_KV_FORMAT_HQ_E8_2B;
  pool_spec.kv_page_group_count = (static_cast<std::uint32_t>(kPrompt) + page - 1) / page;
  pool_spec.max_context_tokens = static_cast<std::uint32_t>(kPrompt);
  pool_spec.slot_count = 1;
  pool_spec.gdn_num_layers = 2;
  pool_spec.kv_num_layers = kIgnisGqaLayerCount;
  pool_spec.gdn_conv_channels = 6;
  pool_spec.gdn_value_heads = 2;
  pool_spec.gdn_head_dim = 4;
  pool_spec.vocab = 32;
  ignis_seq_pool *pool = nullptr;
  ignis_seq *seq = nullptr;
  if (ignis_seq_pool_create(&pool_spec, &pool) != 0 || ignis_seq_alloc(pool, pool_spec.max_context_tokens, &seq) != 0) {
    std::fprintf(stderr, "FATAL: pool or sequence allocation failed\n");
    std::exit(EXIT_FAILURE);
  }
  const ninfer::PagedKVLayerView layer = ignis_kv_layer_view(pool, seq, 0);
  check(layer.residual_k.data != nullptr, "the pool carries the residual window");

  const auto q_elems = static_cast<std::size_t>(kHeadDim) * kQHeads * kChunk;
  const auto kv_elems = static_cast<std::size_t>(kHeadDim) * kKvHeads * kChunk;
  const std::size_t fetched = static_cast<std::size_t>(kKvHeads) * kPrompt * kHeadDim * 2;
  DeviceBytes qd(q_elems * 2), kd(kv_elems * 2), vd(kv_elems * 2), gd(q_elems * 2), od(q_elems * 2),
      pd(static_cast<std::size_t>(kChunk) * 4), fk(fetched), fv(fetched);
  CUDA_FATAL(cudaMemset(gd.p, 0, q_elems * 2));  // sigmoid(0) = 1/2: the output is half the attention
  ninfer::DeviceArena workspace(ignis_hq_prompt_workspace_bytes(kChunk, kPrompt, spec.band_keys));
  const std::vector<int> rows = sampled_rows();

  for (int c = 0; c < kChunks; ++c) {
    const int start = c * kChunk, visible = start + kChunk;
    gaussian_rows(qd.p, kChunk, kQHeads, 3 * c + 1, 1.0F, 1.0F, kGroup, 0.0F);  // sigma 1..4 by KV group
    gaussian_rows(kd.p, kChunk, kKvHeads, 3 * c + 2, 1.0F, 0.0F, 1, 0.75F);     // a mean per KV head
    gaussian_rows(vd.p, kChunk, kKvHeads, 3 * c + 3, 1.0F, 0.0F, 1, 0.0F);
    std::vector<std::int32_t> positions(kChunk);
    for (int t = 0; t < kChunk; ++t) positions[static_cast<std::size_t>(t)] = start + t;
    CUDA_FATAL(cudaMemcpy(pd.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
    const ninfer::Tensor q(qd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, kChunk, 1});
    const ninfer::Tensor k(kd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kChunk, 1});
    const ninfer::Tensor v(vd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kChunk, 1});
    const ninfer::Tensor gate(gd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, kChunk, 1});
    ninfer::Tensor out(od.p, ninfer::DType::BF16, {kHeadDim, kQHeads, kChunk, 1});
    const ninfer::Tensor pos(pd.p, ninfer::DType::I32, {kChunk, 1, 1, 1});

    const bool checked = std::find(kChecked.begin(), kChecked.end(), c) != kChecked.end();
    IgnisHqPromptOptions options;
    options.precision = IgnisHqPromptPrecision::kInt8;
    options.splits = spec.splits;
    options.band_keys = spec.band_keys;
    IgnisHqPromptCarry carry;
    options.after_merge = [&](const IgnisHqPromptCarry &state) { carry = state; };
    if (checked) {
      options.after_attention = [&](const IgnisHqPromptKeys &keys) {
        ignis_hq_prompt_fetch_rows(keys, false, 0, visible, fk.p, nullptr);
        ignis_hq_prompt_fetch_rows(keys, true, 0, visible, fv.p, nullptr);
      };
    }
    ignis_hq_prompt_attention(q, k, v, pos, start, gate, kScale, layer, workspace, out, nullptr, options);
    CUDA_FATAL(cudaDeviceSynchronize());

    // Every output and every carried value finite (a split with no key keeps m = -inf).
    const std::vector<std::uint16_t> ho = download<std::uint16_t>(od.p, q_elems);
    std::size_t nonfinite = 0;
    for (const std::uint16_t b : ho) nonfinite += std::isfinite(bf16(b)) ? 0 : 1;
    const std::size_t states = static_cast<std::size_t>(carry.splits) * kChunk * kQHeads;
    const std::vector<std::uint16_t> co = download<std::uint16_t>(carry.o, states * kHeadDim);
    const std::vector<float> cm = download<float>(carry.m, states), cl = download<float>(carry.l, states);
    std::size_t bad_state = 0;
    for (const std::uint16_t b : co) bad_state += std::isfinite(f16(b)) ? 0 : 1;
    for (std::size_t i = 0; i < states; ++i) {
      bad_state += std::isfinite(cl[i]) && cl[i] >= 0.0F && !std::isnan(cm[i]) && !(std::isinf(cm[i]) && cm[i] > 0.0F) ? 0 : 1;
      if (spec.splits == 1) bad_state += std::isfinite(cm[i]) && cl[i] > 0.0F ? 0 : 1;
    }
    check(carry.splits == (spec.splits != 0 ? spec.splits : (visible <= 16384 ? 2 : 3)) && carry.width == kChunk,
          "chunk " + std::to_string(c) + ": the split count the run asks for (the rule: 2 up to 16K keys, 3 above)");
    check(nonfinite == 0, "chunk " + std::to_string(c) + ": no non-finite output (" + std::to_string(nonfinite) + ")");
    check(bad_state == 0, "chunk " + std::to_string(c) + ": every carried value finite (" + std::to_string(bad_state) + ")");
    if (!checked) continue;

    const Reference ref = reference(rows, start, visible, qd.p, fk.p, fv.p);
    // The FP16 state's norm per sampled row and head, over its splits at their shares.
    std::vector<double> state_norm(rows.size() * kQHeads, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r) {
      for (int h = 0; h < kQHeads; ++h) {
        double mx = -kInf, num = 0.0, den = 0.0;
        for (int s = 0; s < carry.splits; ++s) mx = std::max(mx, static_cast<double>(cm[(static_cast<std::size_t>(s) * kChunk + rows[r]) * kQHeads + h]));
        for (int s = 0; s < carry.splits; ++s) {
          const std::size_t at = (static_cast<std::size_t>(s) * kChunk + rows[r]) * kQHeads + h;
          if (std::isinf(cm[at])) continue;
          const double share = cl[at] * std::exp2(cm[at] - mx);
          double n2 = 0.0;
          for (int d = 0; d < kHeadDim; ++d) n2 += f16(co[at * kHeadDim + d]) * f16(co[at * kHeadDim + d]);
          num += share * std::sqrt(n2);
          den += share;
        }
        state_norm[r * kQHeads + h] = den > 0.0 ? num / den : 0.0;
      }
    }
    const double slack = std::ldexp(ref.vmax, -18);
    auto sigma = [&](std::size_t rh, int d) {
      const double fp16 = std::ldexp(state_norm[rh], -11) / (16.0 * std::sqrt(3.0));
      const double L = ref.L[rh];
      return std::sqrt(ref.A[rh] + ref.B[rh * kHeadDim + d] + ref.C[rh * kHeadDim + d] / (L * L) + fp16 * fp16);
    };
    // The small keys' rounding, which has a sign: their weight times how far their mean is from O.
    auto small = [&](std::size_t rh, int d) {
      return std::fabs(ref.Vbar[rh * kHeadDim + d] - ref.O[rh * kHeadDim + d]) * ref.S[rh] / ref.L[rh];
    };
    auto terms = [&](std::size_t r, int h, int d) {
      const std::size_t rh = r * kQHeads + h;
      char line[256];
      std::snprintf(line, sizeof(line),
                    "row %d head %d dim %d: small %.3g sqrt A %.3g sqrt B %.3g sqrt C/L^2 %.3g, m %.4g L %.4g S %.3g D %.3g",
                    rows[r], h, d, small(rh, d), std::sqrt(ref.A[rh]), std::sqrt(ref.B[rh * kHeadDim + d]),
                    std::sqrt(ref.C[rh * kHeadDim + d]) / ref.L[rh], ref.m[rh], ref.L[rh], ref.S[rh], ref.D[rh]);
      return std::string(line);
    };
    const std::string at = "chunk " + std::to_string(c) + " (" + std::to_string(visible) + " keys)";
    Use output;
    std::size_t keys_seen = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      for (int h = 0; h < kQHeads; ++h) {
        const std::size_t rh = r * kQHeads + h;
        keys_seen += static_cast<std::size_t>(ref.n[rh]) == static_cast<std::size_t>(start + rows[r] + 1) ? 0 : 1;
        for (int d = 0; d < kHeadDim; ++d) {
          const double want = ref.O[rh * kHeadDim + d];
          const double got = 2.0 * bf16(ho[(static_cast<std::size_t>(rows[r]) * kQHeads + h) * kHeadDim + d]);
          output.add(got, want,
                     std::ldexp(std::fabs(want), -8) +
                         (1.0 + std::ldexp(1.0, -8)) * (small(rh, d) + kSigmas * sigma(rh, d) + slack),
                     [&] { return terms(r, h, d); });
        }
      }
    }
    check(keys_seen == 0, at + ": the reference saw every key up to each row");
    report(at + ", output", output);
    if (!spec.carry) continue;

    // The carried state at one split: O over l un-rotated, the max, the log normalizer.
    Use state_o, state_m, state_l;
    std::size_t unbounded = 0;
    for (std::size_t r = 0; r < rows.size(); ++r) {
      for (int h = 0; h < kQHeads; ++h) {
        const std::size_t rh = r * kQHeads + h;
        const std::size_t at_state = static_cast<std::size_t>(rows[r]) * kQHeads + h;
        double x[kHeadDim];
        for (int d = 0; d < kHeadDim; ++d) x[d] = f16(co[at_state * kHeadDim + d]);
        unrotate(x);
        for (int d = 0; d < kHeadDim; ++d) {
          state_o.add(x[d], ref.O[rh * kHeadDim + d], small(rh, d) + kSigmas * sigma(rh, d) + slack,
                      [&] { return terms(r, h, d); });
        }
        const double D = ref.D[rh];
        state_m.add(cm[at_state], ref.m[rh], D + std::ldexp(std::fabs(ref.m[rh]) + 1.0, -18), [&] { return terms(r, h, 0); });
        // The P sum's rounding: the small keys' at most S, the others' six standard deviations,
        // against a sum the scores can have lowered by kappa.
        const double kappa = std::exp2(12.0 * ref.sd[rh]);
        const double xr = kappa * (ref.S[rh] + kSigmas * std::sqrt(ref.n_out[rh] / 3.0) / 510.0) / ref.L[rh];
        if (xr < 1.0) {
          state_l.add(cm[at_state] + std::log2(cl[at_state] / 256.0), ref.m[rh] + std::log2(ref.L[rh]),
                      D - std::log2(1.0 - xr) + std::ldexp(1.0, -16), [&] { return terms(r, h, 0); });
        } else {
          ++unbounded;
        }
      }
    }
    report(at + ", carried O / l", state_o);
    report(at + ", carried m", state_m);
    report(at + ", carried log normalizer", state_l);
    std::printf("  %s: the log normalizer unbounded (x >= 1) on %zu of %zu rows\n", at.c_str(), unbounded,
                rows.size() * kQHeads);
  }
  ignis_seq_release(pool, seq);
  ignis_seq_pool_free(pool);
}

} // namespace

int main() {
  int device = 0, major = 0, minor = 0;
  CUDA_FATAL(cudaGetDevice(&device));
  CUDA_FATAL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_FATAL(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  std::printf("integer band against fp64: compute capability %d.%d, a %d-token prompt at chunk %d\n", major, minor,
              kPrompt, kChunk);
  run({"split 1", 1, kIgnisHqPromptBandKeys, true});
  run({"split rule", 0, kIgnisHqPromptBandKeys, false});
  run({"split 1, narrow band", 1, 4096 + 32, true});
  if (g_failed != 0) {
    std::fprintf(stderr, "hq prompt int8: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("hq prompt int8: all checks passed\n");
  return EXIT_SUCCESS;
}
