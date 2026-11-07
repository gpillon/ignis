// The dense prefill's leaf C ABI against exact references -- OURS (GitHub #509, spec #503;
// kernel/include/ignis_dense_prefill.h). Ported from the checks of the prototype's bench
// (prototype/sm8x-dense-prefill, kernel/tests/bench_sm8x_dense.cu), with host references where
// the bench compared against the fused route's emulation.
//
//   W8       the decode's codes against a host decode of the format (moe_fixture.h's restatement
//            of the codebook), every code of a synthetic matrix, every tile of every 8th tile row
//            of an artifact matrix.
//   A8       the quantize launch's INT8 values, scales and sums against a host reference of the
//            header's arithmetic, bit for bit; a group's launch also against each member alone.
//   int32    the GEMM's accumulators against an exact integer reference on the CUDA cores (dp4a,
//            no tensor core): at the plan's own choice, and for each group's first matrix at
//            every tile height, depth and raster.
//   outputs  the composed call the engine runs (ignis_dense_prefill_linear) against the fp64
//            epilogue of those accumulators, within |y - Y| <= 2^-8 |Y| + 2^-9 s: the store, the
//            residual add, SwiGLU's two passes and the query/gate split.
//   tile plan (GitHub #530)  the plan's picks on the host, for the body's shapes at 2048 to 8192
//            tokens on 82, 128 and 170 SMs; and every tile height, depth and raster against the
//            op's own plan, every BF16 byte, for every body shape at 1, 300, 2048 and 2049 tokens
//            through each epilogue.
//   shared workspace  the fused op's decode route on the workspace a dense call used gives its
//            fresh workspace's bits: the dense call leaves the fused op's counters alone.
//   fused producers (GitHub #511)  A8 fused into the RMSnorm that produces its input, against
//            the standalone A8 on the same inputs, bit for bit: the RMSnorm's row against the
//            vendored rmsnorm's and its activations against the standalone A8 of that row, per
//            matrix of the group; and the composed calls the engine runs (the norm with a group's
//            projections, the MLP: the norm with gate and up, then down on its standalone A8)
//            against the unfused composition, every output bit.
//   rejects  the arguments the ops refuse.
//
// Two arms. synthetic: counter-hash trellis words at every K class (2, 2.5, 3, 4, 5, 6) and every
// A8 kernel (in up to 5120, up to 6144, wider), always run. artifact: real matrices of the base
// 16 GB artifact, one of each K class its body holds (2, 3, 4, 5) and each input width it has
// (5120, 6144, 17408), with the matrices that share their input: $IGNIS_DENSE_PREFILL_ARTIFACT,
// else the path below; absent, a SKIP line outside the GPU profile and a failure under it
// (IGNIS_GPU_PROFILE=1, docs/agents/testing.md).
//
// Not labelled sm120a-only: every kernel fits sm_86, and the test passes there.

#include "dense_prefill_internal.cuh"
#include "trellis_test_common.h"

#include "ninfer/ops/rmsnorm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace trellis_test;

namespace {

constexpr const char *kDefaultArtifact =
    "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";

// The affine map of a W8 code (the header): sw q + off, exact.
double map_sw() { return 4.0 * moe_test::f16_to_f32(0x1EEE); }
double map_off() { return 1534.0 * moe_test::f16_to_f32(0x1EEE) + moe_test::f16_to_f32(0xC931); }

// ---- host references -------------------------------------------------------------------------

size_t a_index(int tokens, int t, int k) { return (static_cast<size_t>(k >> 6) * tokens + t) * 64 + (k & 63); }

size_t dense_index(int in, int n, int k) {
  return (static_cast<size_t>(n / 128) * (in / 64) + k / 64) * 8192 + static_cast<size_t>(n % 128) * 64 + (k % 64);
}

int host_w8(int s) {
  const float v = static_cast<float>(s - 510) * 0.25f;
  return std::max(-127, std::min(127, static_cast<int>(std::nearbyint(v))));  // ties to even
}

// The codebook integers s of tile (tk, tc), at element (r, c) of the tile: the format restated
// (moe_fixture.h's host_trellis_decode), s = the byte sum of state * 0x83DCD12D.
void host_tile_codes(const uint16_t *words, uint32_t k2, int tiles_n, int tk, int tc, int (&s)[16][16]) {
  const uint32_t w16 = moe_test::trellis_tile_words16(k2);
  const int bits = static_cast<int>(w16) * 16;
  int perm[256];
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2;
    const int c0 = t / 4;
    const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
    for (int j = 0; j < 8; ++j) perm[t * 8 + j] = rows[j % 4] * 16 + c0 + (j >= 4 ? 8 : 0);
  }
  const uint16_t *tile = words + (static_cast<size_t>(tk) * tiles_n + tc) * w16;
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

// The kernel's 128-wide Hadamard (moe_common.cuh's warp_hadamard128), its fp32 operations in its
// order: lane l holds entries 4 l .. 4 l + 3.
void host_hadamard128(float *v) {
  float l[32][4];
  for (int lane = 0; lane < 32; ++lane) {
    const float *p = v + 4 * lane;
    const float a = p[0] + p[1], b = p[0] - p[1], c = p[2] + p[3], d = p[2] - p[3];
    l[lane][0] = a + c;
    l[lane][1] = b + d;
    l[lane][2] = a - c;
    l[lane][3] = b - d;
  }
  for (int m = 1; m < 32; m <<= 1) {
    float n[32][4];
    for (int lane = 0; lane < 32; ++lane) {
      for (int i = 0; i < 4; ++i) {
        const float o = l[lane ^ m][i];
        n[lane][i] = (lane & m) != 0 ? o - l[lane][i] : l[lane][i] + o;
      }
    }
    std::memcpy(l, n, sizeof(l));
  }
  for (int lane = 0; lane < 32; ++lane)
    for (int i = 0; i < 4; ++i) v[4 * lane + i] = l[lane][i] * 0.08838834764831845f;
}

// A8 as the header states it, in fp32: q in the kernel's slab layout.
struct A8 {
  std::vector<int8_t> q;
  std::vector<float> scale;
  std::vector<int32_t> sum;
};

A8 host_a8(const std::vector<uint16_t> &x, const std::vector<uint16_t> &suh, int tokens, int in) {
  A8 a;
  a.q.assign(static_cast<size_t>(tokens) * in, 0);
  a.scale.assign(tokens, 0.0f);
  a.sum.assign(tokens, 0);
  std::vector<float> v(in);
  for (int t = 0; t < tokens; ++t) {
    for (int k = 0; k < in; ++k) {
      const float xv = moe_test::bf16_to_f32(x[static_cast<size_t>(t) * in + k]);
      v[k] = xv * moe_test::f16_to_f32(suh[k]);
    }
    float amax = 0.0f;
    for (int b = 0; b < in; b += 128) {
      host_hadamard128(v.data() + b);
      for (int i = 0; i < 128; ++i) amax = std::fmax(amax, std::fabs(v[b + i]));
    }
    const bool finite = amax > 0.0f && amax <= 3.0e38f;
    const float to_q = finite ? 127.0f / amax : 0.0f;
    a.scale[t] = finite ? amax * (1.0f / 127.0f) : 0.0f;
    int sum = 0;
    for (int k = 0; k < in; ++k) {
      const float r = v[k] * to_q;
      const int q = std::isnan(r) ? 0 : std::max(-127, std::min(127, static_cast<int>(std::nearbyint(r))));
      a.q[a_index(tokens, t, k)] = static_cast<int8_t>(q);
      sum += q;
    }
    a.sum[t] = sum;
  }
  return a;
}

// The epilogue's fp64 value Y from the exact accumulators, and the bound's scale s.
struct Expected {
  std::vector<double> y, s;
};

Expected expected(const std::vector<int32_t> &acc, const A8 &a, const std::vector<int8_t> &w, const std::vector<uint16_t> &svh,
                  int tokens, int in, int out) {
  const double sw = map_sw(), off = map_off();
  const int nb = out / 128;
  std::vector<double> wsq(static_cast<size_t>(in) * nb, 0.0);
  for (int n = 0; n < out; ++n)
    for (int k = 0; k < in; ++k) {
      const double wv = sw * w[dense_index(in, n, k)] + off;
      wsq[static_cast<size_t>(k) * nb + n / 128] += wv * wv / 128.0;
    }
  Expected e;
  e.y.resize(static_cast<size_t>(tokens) * out);
  e.s.resize(e.y.size());
  std::vector<double> s2(nb);
  for (int t = 0; t < tokens; ++t) {
    double *y = e.y.data() + static_cast<size_t>(t) * out;
    const double sa = a.scale[t];
    for (int n = 0; n < out; ++n) y[n] = sa * (sw * acc[static_cast<size_t>(t) * out + n] + off * a.sum[t]);
    moe_test::hadamard128_inplace(y, out);
    std::fill(s2.begin(), s2.end(), 0.0);
    for (int k = 0; k < in; ++k) {
      const double av = sa * a.q[a_index(tokens, t, k)];
      const double av2 = av * av;
      const double *row = wsq.data() + static_cast<size_t>(k) * nb;
      for (int b = 0; b < nb; ++b) s2[b] += av2 * row[b];
    }
    for (int n = 0; n < out; ++n) {
      const double sv = moe_test::f16_to_f32(svh[n]);
      y[n] *= sv;
      e.s[static_cast<size_t>(t) * out + n] = std::fabs(sv) * std::sqrt(s2[n / 128]);
    }
  }
  return e;
}

// acc[t][n] = sum_k a[t][k] w[n][k], dp4a on the CUDA cores (no tensor core): 64 x 64 outputs a
// CTA, 4 x 4 a thread.
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
      as[r][wd] = t < tokens ? *reinterpret_cast<const int32_t *>(a + (static_cast<size_t>(k0 >> 6) * tokens + t) * 64 + 4 * wd) : 0;
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

// ---- inputs ----------------------------------------------------------------------------------

// A trellis matrix, host and device.
struct Matrix {
  std::string name;
  uint32_t k2 = 0, in = 0, out = 0;
  std::vector<uint16_t> words, suh, svh;
  DeviceBytes d_words, d_suh, d_svh;
  ignis_trellis_weight w{};

  void upload_all() {
    d_words = DeviceBytes(words.size() * 2 + 256);
    d_suh = DeviceBytes(suh.size() * 2);
    d_svh = DeviceBytes(svh.size() * 2);
    upload(d_words, words);
    upload(d_suh, suh);
    upload(d_svh, svh);
    w = ignis_trellis_weight{d_words.p, d_suh.p, d_svh.p, k2, in, out, 0};
  }
};

// Scale vectors like a quant's: random sign, magnitude in [0.5, 2) times `amp`.
std::vector<uint16_t> hash_scales(uint32_t stream, int n, float amp) {
  std::vector<uint16_t> v(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t h = moe_test::hash_u32(stream, static_cast<uint64_t>(i));
    const float mag = amp * (0.5f + 1.5f * static_cast<float>(h >> 9) * 0x1p-23f);
    v[i] = moe_test::f32_to_f16((h & 1u) ? -mag : mag);
  }
  return v;
}

Matrix synthetic(uint32_t k2, uint32_t in, uint32_t out, uint32_t stream) {
  Matrix m;
  m.name = "synthetic K" + std::to_string(k2 / 2) + (k2 % 2 != 0 ? ".5" : "") + " [" + std::to_string(out) + " x " +
           std::to_string(in) + "]";
  m.k2 = k2;
  m.in = in;
  m.out = out;
  m.words = moe_test::hash_trellis_words(stream, static_cast<int>(in), static_cast<int>(out), k2);
  m.suh = hash_scales(stream + 1, static_cast<int>(in), 0.05f);
  m.svh = hash_scales(stream + 2, static_cast<int>(out), 0.03f);
  m.upload_all();
  return m;
}

// Hidden-state-like BF16 rows: a sum of three uniforms per entry, one channel in 97 at 24x, a
// per-token gain in [0.5, 1.5), token 0 at 8x, and token 1 all zeros (its scale is 0).
std::vector<uint16_t> standin_x(int tokens, int in, uint32_t seed) {
  std::vector<uint16_t> x(static_cast<size_t>(tokens) * in);
  for (int t = 0; t < tokens; ++t) {
    const float gain = (0.5f + 0.5f * (moe_test::hash_uniform(seed + 1, t, 1.0f) + 1.0f)) * (t == 0 ? 8.0f : 1.0f);
    for (int k = 0; k < in; ++k) {
      const size_t i = static_cast<size_t>(t) * in + k;
      const float u = moe_test::hash_uniform(seed, 3 * i, 1.0f) + moe_test::hash_uniform(seed, 3 * i + 1, 1.0f) +
                      moe_test::hash_uniform(seed, 3 * i + 2, 1.0f);
      const float ch = moe_test::hash_u32(seed + 2, k) % 97 == 0 ? 24.0f : 1.0f;
      x[i] = t == 1 ? 0 : moe_test::f32_to_bf16(u * ch * gain);
    }
  }
  return x;
}

// ---- the artifact (a .ninfer v2 directory, read the way crates/artifact's Reader does) --------

struct Artifact {
  FILE *f = nullptr;
  std::string json;
  uint64_t payload = 0;

  ~Artifact() {
    if (f != nullptr) std::fclose(f);
  }

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

  // The record of `name`: its format, payload offset and [out, in] shape.
  bool find(const std::string &name, std::string &format, uint64_t &offset, uint64_t &out, uint64_t &in) const {
    const size_t at = json.find("\"name\":\"" + name + "\"");
    if (at == std::string::npos) return false;
    const size_t lo = json.rfind('{', at), hi = json.find('}', at);
    const std::string obj = json.substr(lo, hi - lo + 1);
    const size_t op = obj.find("\"offset\":"), fp = obj.find("\"format\":\""), sp = obj.find("\"shape\":[");
    if (op == std::string::npos || fp == std::string::npos || sp == std::string::npos) return false;
    offset = std::strtoull(obj.c_str() + op + 9, nullptr, 10);
    format = obj.substr(fp + 10, obj.find('"', fp + 10) - fp - 10);
    char *end = nullptr;
    out = std::strtoull(obj.c_str() + sp + 9, &end, 10);
    in = *end == ',' ? std::strtoull(end + 1, nullptr, 10) : 0;
    return true;
  }

  bool read(uint64_t offset, uint64_t bytes, void *dst) const {
#ifdef _WIN32
    if (_fseeki64(f, static_cast<long long>(payload + offset), SEEK_SET) != 0) return false;
#else
    if (fseeko(f, static_cast<off_t>(payload + offset), SEEK_SET) != 0) return false;
#endif
    return std::fread(dst, 1, bytes, f) == bytes;
  }

  // A trellis matrix: its tiles, then suh [in] and svh [out] in fp16.
  bool matrix(const std::string &name, Matrix &m) const {
    std::string format;
    uint64_t offset = 0, out = 0, in = 0;
    if (!find(name, format, offset, out, in) || format.rfind("TRELLIS_MUL1_K", 0) != 0 || in == 0) return false;
    const std::string k = format.substr(14);
    m.k2 = k == "2P5" ? 5 : 2 * static_cast<uint32_t>(std::atoi(k.c_str()));
    m.name = name;
    m.in = static_cast<uint32_t>(in);
    m.out = static_cast<uint32_t>(out);
    const uint64_t tb = in * out / 16 * m.k2;
    std::vector<uint8_t> bytes(tb + 2 * in + 2 * out);
    if (!read(offset, bytes.size(), bytes.data())) return false;
    m.words.resize(tb / 2);
    std::memcpy(m.words.data(), bytes.data(), tb);
    m.suh.resize(in);
    std::memcpy(m.suh.data(), bytes.data() + tb, 2 * in);
    m.svh.resize(out);
    std::memcpy(m.svh.data(), bytes.data() + tb + 2 * in, 2 * out);
    m.upload_all();
    return true;
  }

  // A 1-D BF16 tensor (a norm's weight).
  bool vector(const std::string &name, std::vector<uint16_t> &v) const {
    std::string format;
    uint64_t offset = 0, n = 0, rest = 0;
    if (!find(name, format, offset, n, rest) || format != "BF16" || n == 0 || rest != 0) return false;
    v.resize(n);
    return read(offset, 2 * n, v.data());
  }
};

// ---- the checks ------------------------------------------------------------------------------

double g_worst = 0.0;  // the worst share of the output bound, over every output checked

struct DeviceA8 {
  DeviceBytes q, scale, sum;
  ignis_dense_prefill_activations a{};
  DeviceA8(int tokens, int in) : q(static_cast<size_t>(tokens) * in), scale(4ull * tokens), sum(4ull * tokens) {
    a = ignis_dense_prefill_activations{q.p, scale.as<float>(), sum.as<int32_t>()};
  }
  A8 fetch(int tokens, int in) const {
    MOE_CUDA(cudaDeviceSynchronize());
    return A8{download<int8_t>(q.p, static_cast<size_t>(tokens) * in), download<float>(scale.p, tokens),
              download<int32_t>(sum.p, tokens)};
  }
};

bool same_a8(const A8 &x, const A8 &y) {
  return x.q == y.q && x.sum == y.sum && std::memcmp(x.scale.data(), y.scale.data(), 4 * x.scale.size()) == 0;
}

// What one output of a group is checked as.
struct Out {
  uint32_t epilogue = IGNIS_TRELLIS_STORE;
  uint32_t split = 0;
};

// The bound and the agreement of `got` against Y (plus `base` when the epilogue adds), with
// |silu(gate)| scaling the s term of SwiGLU's up pass.
Agreement agree_out(const std::vector<double> &got, const Expected &e, const std::vector<double> &base,
                    const std::vector<double> &gain) {
  std::vector<double> want(got.size()), bound(got.size());
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = gain.empty() ? 1.0 : gain[i];
    want[i] = (base.empty() ? 0.0 : base[i]) + g * e.y[i];
    bound[i] = std::ldexp(std::fabs(want[i]), -8) + std::ldexp(std::fabs(g) * e.s[i], -9);
  }
  return agree(got, want, bound);
}

// The columns of `y` [tokens][out] the split sends to its first (half 0) or second half.
std::vector<double> split_half(const std::vector<double> &y, int tokens, int out, int split, int half) {
  std::vector<double> o(static_cast<size_t>(tokens) * (out / 2));
  for (int t = 0; t < tokens; ++t)
    for (int c = 0; c < out; ++c) {
      const int head = c / (2 * split), within = c % (2 * split);
      if ((within < split ? 0 : 1) != half) continue;
      o[static_cast<size_t>(t) * (out / 2) + head * split + within - half * split] = y[static_cast<size_t>(t) * out + c];
    }
  return o;
}

// One group of matrices that read the same input, through every check. `w8_step`: every that
// many-th tile row's codes are checked (1: all).
void check_group(const std::string &label, const std::vector<Matrix *> &g, const std::vector<Out> &outs, int tokens,
                 uint32_t seed, int w8_step) {
  const int n = static_cast<int>(g.size());
  const int in = static_cast<int>(g[0]->in);
  std::printf("  %s: %d matrix(es) of in %d, %d tokens\n", label.c_str(), n, in, tokens);
  const std::vector<uint16_t> xh = standin_x(tokens, in, seed);
  DeviceBytes d_x(xh.size() * 2);
  upload(d_x, xh);

  // A8: the group's launch against the host, and against each member's launch alone.
  std::vector<A8> host(n);
  std::vector<DeviceA8> dev;
  const ignis_trellis_weight *wl[IGNIS_DENSE_PREFILL_MAX_GROUP];
  ignis_dense_prefill_activations al[IGNIS_DENSE_PREFILL_MAX_GROUP];
  dev.reserve(n);
  for (int i = 0; i < n; ++i) {
    host[i] = host_a8(xh, g[i]->suh, tokens, in);
    dev.emplace_back(tokens, in);
    wl[i] = &g[i]->w;
    al[i] = dev[i].a;
  }
  TL_RC(ignis_dense_prefill_quantize(d_x.p, static_cast<uint32_t>(tokens), wl, static_cast<uint32_t>(n), al, nullptr));
  std::vector<A8> got(n);
  for (int i = 0; i < n; ++i) {
    got[i] = dev[i].fetch(tokens, in);
    size_t values = 0, scales = 0, sums = 0;
    for (size_t k = 0; k < got[i].q.size(); ++k) values += got[i].q[k] != host[i].q[k];
    for (int t = 0; t < tokens; ++t) {
      scales += std::memcmp(&got[i].scale[t], &host[i].scale[t], 4) != 0;
      sums += got[i].sum[t] != host[i].sum[t];
    }
    std::printf("    %s: A8 against the host: %zu of %zu values, %zu scales, %zu sums differ\n", g[i]->name.c_str(), values,
                got[i].q.size(), scales, sums);
    check(values == 0 && scales == 0 && sums == 0, label + ": " + g[i]->name + ": A8 bit-identical to the host reference");
    if (n > 1) {
      DeviceA8 alone(tokens, in);
      TL_RC(ignis_dense_prefill_quantize(d_x.p, static_cast<uint32_t>(tokens), &wl[i], 1, &alone.a, nullptr));
      check(same_a8(alone.fetch(tokens, in), got[i]), label + ": " + g[i]->name + ": the group's A8 is the member's alone");
    }
  }

  // W8 and the accumulators, per matrix.
  uint64_t dense_bytes = 0;
  for (const Matrix *m : g) dense_bytes = std::max<uint64_t>(dense_bytes, ignis_dense_prefill_weight_bytes(m->in, m->out));
  DeviceBytes d_dense(dense_bytes);
  std::vector<std::vector<int32_t>> ref(n);
  std::vector<std::vector<int8_t>> wq(n);
  ignis_dense::State st;
  check(ignis_dense::state("test", &st) == 0, "the device is prepared");
  for (int i = 0; i < n; ++i) {
    const Matrix &m = *g[i];
    const int out = static_cast<int>(m.out);
    MOE_CUDA(cudaMemset(d_dense.p, 0x5A, dense_bytes));
    TL_RC(ignis_dense_prefill_decode(&m.w, d_dense.p, dense_bytes, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    wq[i] = download<int8_t>(d_dense.p, static_cast<size_t>(in) * out);
    size_t wrong = 0, total = 0;
    int s[16][16];
    for (int tk = 0; tk < in / 16; tk += w8_step)
      for (int tc = 0; tc < out / 16; ++tc) {
        host_tile_codes(m.words.data(), m.k2, out / 16, tk, tc, s);
        for (int r = 0; r < 16; ++r)
          for (int c = 0; c < 16; ++c) {
            wrong += wq[i][dense_index(in, tc * 16 + c, tk * 16 + r)] != static_cast<int8_t>(host_w8(s[r][c]));
            ++total;
          }
      }
    std::printf("    %s (K %.1f): W8 against the host decode: %zu of %zu codes differ\n", m.name.c_str(), m.k2 / 2.0, wrong, total);
    check(wrong == 0, label + ": " + m.name + ": W8 codes equal the host decode");

    DeviceBytes d_ref(4ull * tokens * out), d_acc(4ull * tokens * out);
    ref_int_gemm<<<dim3(out / 64, (tokens + 63) / 64), 256>>>(static_cast<const int8_t *>(dev[i].q.p), d_dense.as<int8_t>(),
                                                             tokens, in, out, d_ref.as<int32_t>());
    MOE_CUDA(cudaGetLastError());
    MOE_CUDA(cudaDeviceSynchronize());
    ref[i] = download<int32_t>(d_ref.p, static_cast<size_t>(tokens) * out);
    long long amax = 0;
    for (const int32_t v : ref[i]) amax = std::max(amax, std::llabs(static_cast<long long>(v)));
    auto differ = [&]() {
      MOE_CUDA(cudaDeviceSynchronize());
      const std::vector<int32_t> acc = download<int32_t>(d_acc.p, static_cast<size_t>(tokens) * out);
      size_t d = 0;
      for (size_t k = 0; k < acc.size(); ++k) d += acc[k] != ref[i][k];
      return d;
    };
    MOE_CUDA(cudaMemset(d_acc.p, 0x5A, d_acc.bytes));
    TL_RC(ignis_dense_prefill_accumulate(&m.w, d_dense.p, &dev[i].a, static_cast<uint32_t>(tokens), d_acc.as<int32_t>(),
                                         m.out, nullptr));
    const ignis_dense::Plan own = ignis_dense::plan(st, static_cast<uint32_t>(tokens), m.in, m.out);
    const size_t d_own = differ();
    std::printf("    %s: int32 accumulators (the plan's BM %d, %d stages, %s fastest) against the integer reference: "
                "%zu of %zu differ (max |acc| %lld)\n",
                m.name.c_str(), own.bm, own.stages, own.raster == ignis_dense::kTokensFastest ? "tokens" : "outputs", d_own,
                ref[i].size(), amax);
    check(d_own == 0, label + ": " + m.name + ": int32 accumulators exactly equal (the plan's own)");
    if (i == 0) {
      for (const int bm : ignis_dense::kTileHeights)
        for (const int stages : ignis_dense::kDepths)
          for (const int raster : {ignis_dense::kTokensFastest, ignis_dense::kOutputsFastest}) {
            ignis_dense::Plan p;
            p.bm = bm;
            p.stages = stages;
            p.raster = raster;
            MOE_CUDA(cudaMemset(d_acc.p, 0x5A, d_acc.bytes));
            TL_RC(ignis_dense::gemm_at("test", p, &m.w, d_dense.p, &dev[i].a, static_cast<uint32_t>(tokens), nullptr,
                                       d_acc.as<int32_t>(), m.out, nullptr));
            const size_t d = differ();
            check(d == 0, label + ": " + m.name + ": int32 accumulators exactly equal at BM " + std::to_string(bm) + ", " +
                              std::to_string(stages) + " stages, raster " + std::to_string(raster) + " (" +
                              std::to_string(d) + " differ)");
          }
      std::printf("    %s: int32 accumulators at BM 256 and 128, 3 and 4 stages, both rasters: checked\n", m.name.c_str());
    }
  }

  // The outputs, through the composed call.
  std::vector<DeviceBytes> d_y(n), d_y2(n);
  std::vector<std::vector<uint16_t>> residual(n);
  std::vector<ignis_dense_prefill_output> ol(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t width = outs[i].split == 0 ? g[i]->out : g[i]->out / 2;
    const size_t bytes = 2ull * tokens * width;
    if (outs[i].epilogue == IGNIS_DENSE_PREFILL_SWIGLU_UP) {
      // The up pass reads and writes the gate pass's buffer.
      ol[i] = ol[i - 1];
      ol[i].epilogue = IGNIS_DENSE_PREFILL_SWIGLU_UP;
      continue;
    }
    d_y[i] = DeviceBytes(bytes);
    if (outs[i].epilogue == IGNIS_TRELLIS_ADD) {
      residual[i].resize(static_cast<size_t>(tokens) * width);
      for (size_t k = 0; k < residual[i].size(); ++k) residual[i][k] = moe_test::f32_to_bf16(moe_test::hash_uniform(seed + 7, k, 0.8f));
      upload(d_y[i], residual[i]);
    } else {
      MOE_CUDA(cudaMemset(d_y[i].p, 0xFF, bytes));  // NaN: a column the call misses fails
    }
    if (outs[i].split != 0) {
      d_y2[i] = DeviceBytes(bytes);
      MOE_CUDA(cudaMemset(d_y2[i].p, 0xFF, bytes));
    }
    ol[i] = ignis_dense_prefill_output{d_y[i].p, width, outs[i].epilogue, outs[i].split, 0, d_y2[i].p};
  }
  const uint64_t ws_bytes = ignis_dense_prefill_workspace_bytes(static_cast<uint32_t>(in), static_cast<uint32_t>(n),
                                                                static_cast<uint32_t>(tokens));
  DeviceBytes d_ws(ws_bytes);
  MOE_CUDA(cudaMemset(d_ws.p, 0, ws_bytes));
  const ignis_trellis_workspace ws{d_ws.p, ws_bytes};
  TL_RC(ignis_dense_prefill_linear(wl, static_cast<uint32_t>(n), d_x.p, static_cast<uint32_t>(tokens), ol.data(), &ws,
                                   d_dense.p, dense_bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < n; ++i) {
    const Matrix &m = *g[i];
    const int out = static_cast<int>(m.out);
    if (i + 1 < n && outs[i + 1].epilogue == IGNIS_DENSE_PREFILL_SWIGLU_UP) continue;  // the pair is checked at its up pass
    const Expected e = expected(ref[i], host[i], wq[i], m.svh, tokens, in, out);
    Agreement a;
    std::string what;
    if (outs[i].epilogue == IGNIS_DENSE_PREFILL_SWIGLU_UP) {
      // The gate pass alone (the same kernel and plan, so the bits the call stored), then the
      // pair's output against silu(gate) times the up's Y.
      const Matrix &gm = *g[i - 1];
      DeviceBytes d_gate(2ull * tokens * gm.out);
      TL_RC(ignis_dense_prefill_decode(&gm.w, d_dense.p, dense_bytes, nullptr));
      const ignis_dense_prefill_output go{d_gate.p, gm.out, IGNIS_TRELLIS_STORE, 0, 0, nullptr};
      TL_RC(ignis_dense_prefill_gemm(&gm.w, d_dense.p, &dev[i - 1].a, static_cast<uint32_t>(tokens), &go, nullptr));
      MOE_CUDA(cudaDeviceSynchronize());
      const std::vector<double> gate = bf16_to_f64(download<uint16_t>(d_gate.p, static_cast<size_t>(tokens) * gm.out));
      const Expected eg = expected(ref[i - 1], host[i - 1], wq[i - 1], gm.svh, tokens, in, static_cast<int>(gm.out));
      const Agreement ag = agree_out(gate, eg, {}, {});
      std::printf("    %s: gate pass - Y: worst %.3f of the bound, rel RMS %.3g\n", gm.name.c_str(), ag.worst, ag.rel_rms);
      check(ag.bad == 0, label + ": " + gm.name + ": the gate pass within 2^-8 |Y| + 2^-9 s");
      g_worst = std::max(g_worst, ag.worst);
      // The kernel's silu is fp32's (moe_common.cuh), which reads a gate below about -88 as 0:
      // expf overflows where the exact value lies under fp32's range (and BF16's) anyway.
      std::vector<double> gain(gate.size());
      for (size_t k = 0; k < gate.size(); ++k) {
        const float gv = static_cast<float>(gate[k]);
        gain[k] = gv / (1.0f + std::exp(-gv));
      }
      const std::vector<double> h = bf16_to_f64(download<uint16_t>(d_y[i - 1].p, static_cast<size_t>(tokens) * out));
      a = agree_out(h, e, {}, gain);
      what = "SwiGLU (silu(gate) x up)";
    } else if (outs[i].split != 0) {
      const int split = static_cast<int>(outs[i].split);
      const int half = out / 2;
      Expected h0{split_half(e.y, tokens, out, split, 0), split_half(e.s, tokens, out, split, 0)};
      Expected h1{split_half(e.y, tokens, out, split, 1), split_half(e.s, tokens, out, split, 1)};
      const Agreement a0 = agree_out(bf16_to_f64(download<uint16_t>(d_y[i].p, static_cast<size_t>(tokens) * half)), h0, {}, {});
      const Agreement a1 = agree_out(bf16_to_f64(download<uint16_t>(d_y2[i].p, static_cast<size_t>(tokens) * half)), h1, {}, {});
      a = a0.worst >= a1.worst ? a0 : a1;
      a.bad = a0.bad + a1.bad;
      what = "the split (" + std::to_string(split) + "-column halves)";
    } else {
      std::vector<double> base;
      if (outs[i].epilogue == IGNIS_TRELLIS_ADD) base = bf16_to_f64(residual[i]);
      a = agree_out(bf16_to_f64(download<uint16_t>(d_y[i].p, static_cast<size_t>(tokens) * out)), e, base, {});
      what = outs[i].epilogue == IGNIS_TRELLIS_ADD ? "the residual add" : "the store";
    }
    std::printf("    %s: %s - Y: worst %.3f of 2^-8 |Y| + 2^-9 s, rel RMS %.3g\n", m.name.c_str(), what.c_str(), a.worst,
                a.rel_rms);
    check(a.bad == 0, label + ": " + m.name + ": " + what + " within 2^-8 |Y| + 2^-9 s (" + std::to_string(a.bad) + " over)");
    g_worst = std::max(g_worst, a.worst);
  }
}

// ---- A8 fused into its producers (GitHub #511) -----------------------------------------------

constexpr float kEps = 1e-6f;  // the 27B's rms_norm_eps

// A norm's weight like a trained one's: the gain is 1 + w, w around 0.
std::vector<uint16_t> standin_norm(int in, uint32_t seed) {
  std::vector<uint16_t> v(in);
  for (int k = 0; k < in; ++k) v[k] = moe_test::f32_to_bf16(moe_test::hash_uniform(seed, k, 0.6f));
  return v;
}

// The vendored rmsnorm with a unit offset (ninfer::ops::rmsnorm) of x [tokens][in] into h.
void vendored_rmsnorm(const void *x, const void *norm, int tokens, int in, void *h) {
  const ninfer::Tensor xt(const_cast<void *>(x), ninfer::DType::BF16, {in, tokens, 1, 1});
  const ninfer::Tensor nt(const_cast<void *>(norm), ninfer::DType::BF16, {in, 1, 1, 1});
  ninfer::Tensor ht(h, ninfer::DType::BF16, {in, tokens, 1, 1});
  ninfer::ops::rmsnorm(xt, nt, kEps, /*unit_offset=*/true, ht, nullptr);
  MOE_CUDA(cudaDeviceSynchronize());
}

size_t differing(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b) {
  size_t d = 0;
  for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
  return d;
}

// The RMSnorm before a group of matrices fused with their A8: its row against the vendored
// rmsnorm's, its activations against the standalone A8 of that row (and the host's), with h
// written and without; then the composed call against the vendored rmsnorm followed by
// ignis_dense_prefill_linear, every output bit.
void check_norm_group(const std::string &label, const std::vector<Matrix *> &g, const std::vector<Out> &outs,
                      const std::vector<uint16_t> &norm, int tokens, uint32_t seed) {
  const int n = static_cast<int>(g.size());
  const int in = static_cast<int>(g[0]->in);
  std::printf("  %s: the RMSnorm with %d matrix(es)' A8, in %d, %d tokens\n", label.c_str(), n, in, tokens);
  const std::vector<uint16_t> xh = standin_x(tokens, in, seed);
  const size_t row_elems = static_cast<size_t>(tokens) * in;
  DeviceBytes d_x(2 * row_elems), d_norm(2ull * in), d_ref(2 * row_elems), d_h(2 * row_elems);
  upload(d_x, xh);
  upload(d_norm, norm);
  vendored_rmsnorm(d_x.p, d_norm.p, tokens, in, d_ref.p);
  const std::vector<uint16_t> href = download<uint16_t>(d_ref.p, row_elems);

  const ignis_trellis_weight *wl[IGNIS_DENSE_PREFILL_MAX_GROUP];
  std::vector<DeviceA8> ref, fused, bare;
  ref.reserve(n);
  fused.reserve(n);
  bare.reserve(n);
  ignis_dense_prefill_activations ar[IGNIS_DENSE_PREFILL_MAX_GROUP], af[IGNIS_DENSE_PREFILL_MAX_GROUP],
      ab[IGNIS_DENSE_PREFILL_MAX_GROUP];
  for (int i = 0; i < n; ++i) {
    wl[i] = &g[i]->w;
    ref.emplace_back(tokens, in);
    fused.emplace_back(tokens, in);
    bare.emplace_back(tokens, in);
    ar[i] = ref[i].a;
    af[i] = fused[i].a;
    ab[i] = bare[i].a;
  }
  TL_RC(ignis_dense_prefill_quantize(d_ref.p, static_cast<uint32_t>(tokens), wl, static_cast<uint32_t>(n), ar, nullptr));
  MOE_CUDA(cudaMemset(d_h.p, 0xFF, d_h.bytes));
  TL_RC(ignis_dense_prefill_rmsnorm_quantize(d_x.p, d_norm.p, kEps, static_cast<uint32_t>(tokens), d_h.p, wl,
                                             static_cast<uint32_t>(n), af, nullptr));
  TL_RC(ignis_dense_prefill_rmsnorm_quantize(d_x.p, d_norm.p, kEps, static_cast<uint32_t>(tokens), nullptr, wl,
                                             static_cast<uint32_t>(n), ab, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const size_t rows = differing(download<uint16_t>(d_h.p, row_elems), href);
  std::printf("    the fused RMSnorm's row against the vendored rmsnorm's: %zu of %zu values differ\n", rows, row_elems);
  check(rows == 0, label + ": the fused RMSnorm's row is the vendored rmsnorm's bit for bit");
  for (int i = 0; i < n; ++i) {
    const A8 want = ref[i].fetch(tokens, in);
    const A8 got = fused[i].fetch(tokens, in);
    size_t values = 0, scales = 0, sums = 0;
    for (size_t k = 0; k < got.q.size(); ++k) values += got.q[k] != want.q[k];
    for (int t = 0; t < tokens; ++t) {
      scales += std::memcmp(&got.scale[t], &want.scale[t], 4) != 0;
      sums += got.sum[t] != want.sum[t];
    }
    std::printf("    %s: the fused A8 against the standalone A8 of the row: %zu of %zu values, %zu scales, %zu sums differ\n",
                g[i]->name.c_str(), values, got.q.size(), scales, sums);
    check(values == 0 && scales == 0 && sums == 0, label + ": " + g[i]->name + ": the fused A8 is the standalone A8's bit for bit");
    check(same_a8(want, host_a8(href, g[i]->suh, tokens, in)), label + ": " + g[i]->name + ": the standalone A8 of the row is the host's");
    check(same_a8(bare[i].fetch(tokens, in), want), label + ": " + g[i]->name + ": the fused A8 without h is the same");
  }

  // The composed call against the unfused composition.
  uint64_t dense_bytes = 0;
  for (const Matrix *m : g) dense_bytes = std::max<uint64_t>(dense_bytes, ignis_dense_prefill_weight_bytes(m->in, m->out));
  DeviceBytes d_dense(dense_bytes);
  const uint64_t ws_bytes = ignis_dense_prefill_workspace_bytes(static_cast<uint32_t>(in), static_cast<uint32_t>(n),
                                                                static_cast<uint32_t>(tokens));
  DeviceBytes d_ws(ws_bytes);
  const ignis_trellis_workspace ws{d_ws.p, ws_bytes};
  std::vector<DeviceBytes> y_ref, y2_ref, y_got, y2_got;
  std::vector<ignis_dense_prefill_output> o_ref(n), o_got(n);
  for (int i = 0; i < n; ++i) {
    const uint32_t width = outs[i].split == 0 ? g[i]->out : g[i]->out / 2;
    const size_t elems = static_cast<size_t>(tokens) * width;
    // The same start in both: a residual under an add, else NaN (a column the call misses fails).
    std::vector<uint16_t> start(elems, 0xFFFF);
    if (outs[i].epilogue == IGNIS_TRELLIS_ADD) {
      for (size_t k = 0; k < elems; ++k) start[k] = moe_test::f32_to_bf16(moe_test::hash_uniform(seed + 5, k, 0.8f));
    }
    y_ref.emplace_back(2 * elems);
    y_got.emplace_back(2 * elems);
    upload(y_ref.back(), start);
    upload(y_got.back(), start);
    y2_ref.emplace_back(outs[i].split != 0 ? 2 * elems : 0);
    y2_got.emplace_back(outs[i].split != 0 ? 2 * elems : 0);
    if (outs[i].split != 0) {
      upload(y2_ref.back(), start);
      upload(y2_got.back(), start);
    }
    o_ref[i] = ignis_dense_prefill_output{y_ref[i].p, width, outs[i].epilogue, outs[i].split, 0, y2_ref[i].p};
    o_got[i] = ignis_dense_prefill_output{y_got[i].p, width, outs[i].epilogue, outs[i].split, 0, y2_got[i].p};
  }
  TL_RC(ignis_dense_prefill_linear(wl, static_cast<uint32_t>(n), d_ref.p, static_cast<uint32_t>(tokens), o_ref.data(), &ws,
                                   d_dense.p, dense_bytes, nullptr));
  MOE_CUDA(cudaMemset(d_h.p, 0xFF, d_h.bytes));
  TL_RC(ignis_dense_prefill_norm_linear(wl, static_cast<uint32_t>(n), d_x.p, d_norm.p, kEps, static_cast<uint32_t>(tokens),
                                        d_h.p, o_got.data(), &ws, d_dense.p, dense_bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(differing(download<uint16_t>(d_h.p, row_elems), href) == 0, label + ": the composed call's row is the vendored rmsnorm's");
  for (int i = 0; i < n; ++i) {
    const size_t elems = y_ref[i].bytes / 2;
    size_t d = differing(download<uint16_t>(y_got[i].p, elems), download<uint16_t>(y_ref[i].p, elems));
    if (outs[i].split != 0) d += differing(download<uint16_t>(y2_got[i].p, elems), download<uint16_t>(y2_ref[i].p, elems));
    std::printf("    %s: the composed call's output against the vendored rmsnorm then the dense linear: %zu differ\n",
                g[i]->name.c_str(), d);
    check(d == 0, label + ": " + g[i]->name + ": the norm's composed call gives the unfused output bit for bit");
  }
}

// The MLP of gate, up and down as the engine runs it on the dense prefill: the RMSnorm with gate
// and up's A8 (ignis_dense_prefill_norm_linear, the normalized row not written), gate's store and
// up's SwiGLU pass into h, then down's standalone A8 of h and its add into y
// (ignis_dense_prefill_linear), against the vendored rmsnorm then ignis_dense_prefill_linear over
// gate and up and over down, every bit of h and y.
void check_mlp(const std::string &label, Matrix &gate, Matrix &up, Matrix &down, const std::vector<uint16_t> &norm, int tokens,
               uint32_t seed) {
  const int in = static_cast<int>(gate.in), ffn = static_cast<int>(gate.out);
  std::printf("  %s: the MLP with its norm's A8 fused, in %d, ffn %d, %d tokens\n", label.c_str(), in, ffn, tokens);
  DeviceBytes d_norm(2ull * in), d_post(2ull * tokens * in);
  upload(d_norm, norm);
  const ignis_trellis_weight *gu[2] = {&gate.w, &up.w};
  const ignis_trellis_weight *dl[1] = {&down.w};
  DeviceBytes d_dense(std::max(ignis_dense_prefill_weight_bytes(gate.in, gate.out), ignis_dense_prefill_weight_bytes(down.in, down.out)));
  const size_t h_elems = static_cast<size_t>(tokens) * ffn;
  DeviceBytes d_href(2 * h_elems), d_h(2 * h_elems);

  std::vector<uint16_t> residual(static_cast<size_t>(tokens) * in);
  for (size_t k = 0; k < residual.size(); ++k) residual[k] = moe_test::f32_to_bf16(moe_test::hash_uniform(seed + 9, k, 0.8f));
  DeviceBytes y_ref(2ull * tokens * in), y_got(2ull * tokens * in);
  upload(y_ref, residual);
  upload(y_got, residual);
  const uint64_t ws_bytes = std::max(ignis_dense_prefill_workspace_bytes(gate.in, 2, static_cast<uint32_t>(tokens)),
                                     ignis_dense_prefill_workspace_bytes(down.in, 1, static_cast<uint32_t>(tokens)));
  DeviceBytes d_ws(ws_bytes);
  const ignis_trellis_workspace ws{d_ws.p, ws_bytes};
  // Unfused: the norm of the residual, gate and up into h, down added into y.
  vendored_rmsnorm(y_ref.p, d_norm.p, tokens, in, d_post.p);
  const ignis_dense_prefill_output o_ref[2] = {ignis_dense_prefill_output{d_href.p, gate.out, IGNIS_TRELLIS_STORE, 0, 0, nullptr},
                                               ignis_dense_prefill_output{d_href.p, gate.out, IGNIS_DENSE_PREFILL_SWIGLU_UP, 0, 0, nullptr}};
  TL_RC(ignis_dense_prefill_linear(gu, 2, d_post.p, static_cast<uint32_t>(tokens), o_ref, &ws, d_dense.p, d_dense.bytes, nullptr));
  const ignis_dense_prefill_output od_ref{y_ref.p, gate.in, IGNIS_TRELLIS_ADD, 0, 0, nullptr};
  TL_RC(ignis_dense_prefill_linear(dl, 1, d_href.p, static_cast<uint32_t>(tokens), &od_ref, &ws, d_dense.p, d_dense.bytes, nullptr));
  // The engine's: the norm with gate and up's A8, then down's standalone A8 of h, y the residual itself.
  MOE_CUDA(cudaMemset(d_h.p, 0xFF, d_h.bytes));
  const ignis_dense_prefill_output o_got[2] = {ignis_dense_prefill_output{d_h.p, gate.out, IGNIS_TRELLIS_STORE, 0, 0, nullptr},
                                               ignis_dense_prefill_output{d_h.p, gate.out, IGNIS_DENSE_PREFILL_SWIGLU_UP, 0, 0, nullptr}};
  TL_RC(ignis_dense_prefill_norm_linear(gu, 2, y_got.p, d_norm.p, kEps, static_cast<uint32_t>(tokens), nullptr, o_got, &ws,
                                        d_dense.p, d_dense.bytes, nullptr));
  const ignis_dense_prefill_output od_got{y_got.p, gate.in, IGNIS_TRELLIS_ADD, 0, 0, nullptr};
  TL_RC(ignis_dense_prefill_linear(dl, 1, d_h.p, static_cast<uint32_t>(tokens), &od_got, &ws, d_dense.p, d_dense.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint16_t> href = download<uint16_t>(d_href.p, h_elems);
  const size_t dh = differing(download<uint16_t>(d_h.p, h_elems), href);
  const size_t dy = differing(download<uint16_t>(y_got.p, residual.size()), download<uint16_t>(y_ref.p, residual.size()));
  check(download<uint16_t>(y_ref.p, residual.size()) != residual, label + ": down's add moved the residual");
  std::printf("    the composed MLP against the unfused one: %zu of %zu SwiGLU values, %zu of %zu residual values differ\n", dh,
              h_elems, dy, residual.size());
  check(dh == 0 && dy == 0, label + ": the composed MLP gives the unfused SwiGLU row and residual bit for bit");
}

// ---- the arms --------------------------------------------------------------------------------

void fused_producers_arm() {
  std::printf("fused producers: A8 in the RMSnorm, against the standalone A8\n");
  // The norm's widths: 4096 and 8192 bracket the 27B's 5120, a group of every size.
  {
    Matrix a = synthetic(6, 4096, 384, 7000);
    check_norm_group("synthetic in 4096", {&a}, {Out{}}, standin_norm(4096, 81), 300, 82);
  }
  {
    Matrix q = synthetic(4, 5120, 512, 7010), k = synthetic(8, 5120, 256, 7020), v = synthetic(12, 5120, 128, 7030);
    check_norm_group("synthetic q, k, v at in 5120", {&q, &k, &v},
                     {Out{IGNIS_TRELLIS_STORE, 128}, Out{IGNIS_TRELLIS_STORE, 0}, Out{IGNIS_TRELLIS_STORE, 0}},
                     standin_norm(5120, 83), 270, 84);
  }
  {
    Matrix a = synthetic(5, 8192, 256, 7040), b = synthetic(10, 8192, 128, 7050);
    check_norm_group("synthetic pair at in 8192", {&a, &b}, {Out{}, Out{IGNIS_TRELLIS_ADD, 0}}, standin_norm(8192, 85), 130, 86);
  }
  // The MLP at down's two A8 widths: up to 6144 (the register-resident A8) and past it (two passes).
  for (const uint32_t ffn : {768u, 6400u}) {
    Matrix gate = synthetic(6, 4096, ffn, 7100 + ffn), up = synthetic(8, 4096, ffn, 7110 + ffn),
           down = synthetic(4, ffn, 4096, 7120 + ffn);
    check_mlp("synthetic MLP, ffn " + std::to_string(ffn), gate, up, down, standin_norm(4096, 87), 300, 88);
  }
  check(ignis_dense_prefill_fused_norm_launches() >= 11, "the fused RMSnorm launches are counted (three per norm group, one per MLP)");
  check(ignis_dense_prefill_fused_producers() == 1, "the fused producers are the default");
  ignis_dense_prefill_set_fused_producers(0);
  check(ignis_dense_prefill_fused_producers() == 0, "the switch turns the fused producers off");
  ignis_dense_prefill_set_fused_producers(7);
  check(ignis_dense_prefill_fused_producers() == 1, "the switch turns them back on");
}

// The body's matrix shapes [out x in] (bench_dense_prefill.cu's mix): the GDN layers' qkv and z,
// the output projection of both layer kinds (one shape), the attention's q and its k and v, the
// MLP's gate and up (one shape) and its down; each at a K class the quant gives it.
struct BodyShape {
  const char *name;
  uint32_t in, out, k2;
};
const BodyShape kBody[] = {
    {"gdn qkv", 5120, 10240, 6}, {"gdn z", 5120, 6144, 6},         {"gdn out, gqa o", 6144, 5120, 6},
    {"gqa q", 5120, 12288, 4},   {"gqa k, v", 5120, 1024, 8},      {"mlp gate, up", 5120, 17408, 6},
    {"mlp down", 17408, 5120, 6},
};
// Where the shapes of the sweep's pinned cases sit in kBody.
constexpr size_t kGqaKv = 4, kMlpGateUp = 5, kMlpDown = 6;

// GitHub #530: the tile plan's picks, on the host. Each body shape at 2048, 4096, 6656 and 8192
// tokens on 82, 128 and 170 SMs, one resident CTA per SM at every tile height and depth (what the
// RTX 5090 reports at each). The tile heights below are worked out by hand from the plan's rule:
// a 128-row tile weighs 1.10 times a 256-row tile per row, so BM 128 only where its waves x rows,
// times 1.10, stay under 256's. mlp gate, up at 2048 tokens on 170 SMs: 16 x 136 = 2,176 tiles of
// 128 rows in 13 waves, 1,664 rows, x 1.10 = 1,830, against 8 x 136 = 1,088 tiles of 256 in 7
// waves, 1,792 rows: BM 256, where the waves x rows alone took 128.
void plan_arm() {
  std::printf("the tile plan: its picks on the host\n");
  constexpr uint32_t widths[] = {2048, 4096, 6656, 8192};
  constexpr int sms[] = {82, 128, 170};
  // [SM count][body shape][width]: the tile height.
  const int want[std::size(sms)][std::size(kBody)][std::size(widths)] = {
      {
          // 82 SMs
          {256, 256, 256, 256},  // gdn qkv
          {256, 256, 256, 256},  // gdn z
          {256, 256, 256, 256},  // gdn out, gqa o
          {256, 256, 256, 256},  // gqa q
          {256, 256, 256, 128},  // gqa k, v
          {256, 256, 256, 256},  // mlp gate, up
          {256, 256, 256, 256},  // mlp down
      },
      {
          // 128 SMs
          {256, 256, 256, 256},  // gdn qkv
          {256, 256, 256, 256},  // gdn z
          {128, 256, 256, 256},  // gdn out, gqa o
          {256, 256, 256, 256},  // gqa q
          {128, 256, 256, 256},  // gqa k, v
          {256, 256, 256, 256},  // mlp gate, up
          {128, 256, 256, 256},  // mlp down
      },
      {
          // 170 SMs
          {256, 256, 256, 256},  // gdn qkv
          {128, 256, 256, 256},  // gdn z
          {256, 256, 256, 256},  // gdn out, gqa o
          {256, 256, 256, 256},  // gqa q
          {128, 256, 128, 256},  // gqa k, v
          {256, 256, 256, 256},  // mlp gate, up
          {256, 256, 256, 256},  // mlp down
      },
  };
  auto plan_at = [](int sm_count, const BodyShape &s, uint32_t tokens) {
    ignis_dense::State st;
    st.sms = sm_count;
    st.resident[0][0] = st.resident[0][1] = st.resident[1][0] = st.resident[1][1] = 1;
    return ignis_dense::plan(st, tokens, s.in, s.out);
  };
  int wrong = 0;
  for (size_t c = 0; c < std::size(sms); ++c)
    for (size_t i = 0; i < std::size(kBody); ++i)
      for (size_t j = 0; j < std::size(widths); ++j) {
        const BodyShape &s = kBody[i];
        const ignis_dense::Plan p = plan_at(sms[c], s, widths[j]);
        // The depth follows the tile height: 4 stages for BM 128 or a K of 8192 or more.
        const int stages = want[c][i][j] == 128 || s.in >= 8192 ? 4 : 3;
        const bool ok = p.bm == want[c][i][j] && p.stages == stages;
        wrong += ok ? 0 : 1;
        check(ok, "the tile plan: " + std::string(s.name) + " at " + std::to_string(widths[j]) + " tokens on " +
                      std::to_string(sms[c]) + " SMs takes BM " + std::to_string(want[c][i][j]) + ", " + std::to_string(stages) +
                      " stages (got BM " + std::to_string(p.bm) + ", " + std::to_string(p.stages) + ")");
      }
  std::printf("  %zu picks (%zu shapes, %zu widths, 82, 128 and 170 SMs): %d differ from the rule's\n",
              std::size(sms) * std::size(kBody) * std::size(widths), std::size(kBody), std::size(widths), wrong);
  // The cases the RTX 5090's sweep showed (#530), on its 170 SMs, at the figures of
  // docs/findings/2026-11-07-dense-prefill-tile-plan-weighs-a-128-row-tile.md.
  check(plan_at(170, kBody[kMlpGateUp], 2048).bm == 256, "the tile plan: mlp gate, up at 2048 tokens on 170 SMs takes BM 256 "
                                                         "(BM 128 took 7.8-8.9% longer there)");
  check(plan_at(170, kBody[kGqaKv], 2048).bm == 128, "the tile plan: gqa k, v at 2048 tokens on 170 SMs keeps BM 128 "
                                                     "(BM 256 took 82-83% longer there)");
  check(plan_at(170, kBody[kMlpDown], 6656).bm == 256, "the tile plan: mlp down at 6656 tokens on 170 SMs takes BM 256 "
                                                       "(BM 128 took 9.2-9.3% longer there)");
}

bool bf16_nan(uint16_t v) { return (v & 0x7F80u) == 0x7F80u && (v & 0x007Fu) != 0; }

// GitHub #530: every tile plan writes the bytes of the plan the op picks. The integer GEMM is
// exact and the epilogue works per element on the same int32s and the same scales, so the eight
// plans (two tile heights x two depths x two rasters) must agree to the byte. Each body shape, a
// synthetic matrix decoded once, at 1, 300, 2048 (a multiple of 256) and 2049 tokens, through
// every epilogue: the store, the residual add, SwiGLU's two passes (the gate pass then the up
// pass, both of this matrix at the plan under test) and the query/gate split into 256-column
// halves. y starts as NaN, but for the add, where it holds the same residual at every plan, and
// the op's own output holds no NaN, so a column a plan misses differs. Each BF16 output is held
// to the op's own (ignis_dense_prefill_gemm) with memcmp.
void plans_arm() {
  std::printf("tile plans: the 8 plans' BF16 outputs against the op's own plan, every body shape and epilogue\n");
  ignis_dense::State st;
  check(ignis_dense::state("test", &st) == 0, "the device is prepared");
  struct Epilogue {
    const char *name;
    uint32_t epilogue, split;
  };
  const Epilogue epilogues[4] = {{"store", IGNIS_TRELLIS_STORE, 0},
                                 {"add", IGNIS_TRELLIS_ADD, 0},
                                 {"SwiGLU", IGNIS_DENSE_PREFILL_SWIGLU_UP, 0},
                                 {"split", IGNIS_TRELLIS_STORE, 256}};
  uint32_t stream = 8000;
  for (const BodyShape &s : kBody) {
    stream += 10;
    Matrix m = synthetic(s.k2, s.in, s.out, stream);
    const uint64_t dense_bytes = ignis_dense_prefill_weight_bytes(s.in, s.out);
    DeviceBytes d_dense(dense_bytes);
    TL_RC(ignis_dense_prefill_decode(&m.w, d_dense.p, dense_bytes, nullptr));
    for (const int tokens : {1, 300, 2048, 2049}) {
      const uint32_t t = static_cast<uint32_t>(tokens);
      const std::vector<uint16_t> xh = standin_x(tokens, static_cast<int>(s.in), stream + 1);
      DeviceBytes d_x(xh.size() * 2);
      upload(d_x, xh);
      DeviceA8 a(tokens, static_cast<int>(s.in));
      const ignis_trellis_weight *wl[1] = {&m.w};
      TL_RC(ignis_dense_prefill_quantize(d_x.p, t, wl, 1, &a.a, nullptr));
      // y [tokens][out], or under the split its two halves [tokens][out / 2] one after the other.
      const size_t elems = static_cast<size_t>(tokens) * s.out;
      DeviceBytes d_y(2 * elems), d_residual(2 * elems);
      std::vector<uint16_t> residual(elems);
      for (size_t k = 0; k < elems; ++k) residual[k] = moe_test::f32_to_bf16(moe_test::hash_uniform(stream + 2, k, 0.8f));
      upload(d_residual, residual);
      // One epilogue's output: y reset, then the GEMM at `p` (the op's own plan when null).
      auto run = [&](const Epilogue &e, const ignis_dense::Plan *p) {
        if (e.epilogue == IGNIS_TRELLIS_ADD) {
          MOE_CUDA(cudaMemcpy(d_y.p, d_residual.p, d_y.bytes, cudaMemcpyDeviceToDevice));
        } else {
          MOE_CUDA(cudaMemset(d_y.p, 0xFF, d_y.bytes));
        }
        auto gemm = [&](const ignis_dense_prefill_output &o) {
          if (p == nullptr) {
            TL_RC(ignis_dense_prefill_gemm(&m.w, d_dense.p, &a.a, t, &o, nullptr));
          } else {
            TL_RC(ignis_dense::gemm_at("test", *p, &m.w, d_dense.p, &a.a, t, &o, nullptr, 0, nullptr));
          }
        };
        if (e.split != 0) {
          gemm(ignis_dense_prefill_output{d_y.p, s.out / 2, e.epilogue, e.split, 0, static_cast<uint16_t *>(d_y.p) + elems / 2});
        } else {
          // SwiGLU's up pass reads the gate pass's output from y.
          if (e.epilogue == IGNIS_DENSE_PREFILL_SWIGLU_UP) {
            gemm(ignis_dense_prefill_output{d_y.p, s.out, IGNIS_TRELLIS_STORE, 0, 0, nullptr});
          }
          gemm(ignis_dense_prefill_output{d_y.p, s.out, e.epilogue, 0, 0, nullptr});
        }
        MOE_CUDA(cudaDeviceSynchronize());
        return download<uint16_t>(d_y.p, elems);
      };
      const ignis_dense::Plan own = ignis_dense::plan(st, t, s.in, s.out);
      size_t worst = 0;
      for (const Epilogue &e : epilogues) {
        const std::vector<uint16_t> want = run(e, nullptr);
        const size_t nans = static_cast<size_t>(std::count_if(want.begin(), want.end(), bf16_nan));
        const std::string what = std::string(s.name) + " [" + std::to_string(s.out) + " x " + std::to_string(s.in) + "] at T " +
                                 std::to_string(tokens) + ", " + e.name;
        check(nans == 0, "tile plans: " + what + ": the op's own output holds no NaN (" + std::to_string(nans) + ")");
        for (const int bm : ignis_dense::kTileHeights)
          for (const int stages : ignis_dense::kDepths)
            for (const int raster : {ignis_dense::kTokensFastest, ignis_dense::kOutputsFastest}) {
              ignis_dense::Plan p;
              p.bm = bm;
              p.stages = stages;
              p.raster = raster;
              const std::vector<uint16_t> got = run(e, &p);
              // memcmp holds every byte; on a mismatch, differing() counts the BF16 values behind it.
              const size_t diffs = std::memcmp(got.data(), want.data(), 2 * elems) == 0 ? 0 : differing(got, want);
              worst = std::max(worst, diffs);
              check(diffs == 0, "tile plans: " + what + ": BM " + std::to_string(bm) + ", " + std::to_string(stages) +
                                    " stages, raster " + std::to_string(raster) + " writes the op's bytes (" +
                                    std::to_string(diffs) + " BF16 values differ)");
            }
      }
      std::printf("  %s [%u x %u] at T %d (the op's plan: BM %d, %d stages, %s fastest): store, add, SwiGLU and split at the "
                  "8 plans: at most %zu BF16 values differ\n",
                  s.name, s.out, s.in, tokens, own.bm, own.stages, own.raster == ignis_dense::kTokensFastest ? "tokens" : "outputs",
                  worst);
    }
  }
}

void synthetic_arm() {
  std::printf("synthetic: counter-hash words at every K class\n");
  // Every K class, the 256-token tile's tail (300 = 256 + 44) and both A8 row kernels' widths.
  const uint32_t k2s[6] = {4, 5, 6, 8, 10, 12};
  const uint32_t ins[6] = {384, 1152, 640, 5248, 512, 896};
  for (int i = 0; i < 6; ++i) {
    Matrix m = synthetic(k2s[i], ins[i], 384, 1000 + 10 * i);
    check_group("synthetic K class " + std::to_string(k2s[i]), {&m}, {Out{}}, 300, 20 + i, 1);
  }
  // The two-pass A8 (in past 6144) with the residual add.
  {
    Matrix m = synthetic(6, 6272, 256, 2000);
    check_group("synthetic wide input", {&m}, {Out{IGNIS_TRELLIS_ADD, 0}}, 130, 31, 1);
  }
  // A group of three that shares its input: the split, a store and an add.
  {
    Matrix q = synthetic(4, 640, 512, 3000), k = synthetic(8, 640, 256, 3010), v = synthetic(12, 640, 128, 3020);
    check_group("synthetic q, k, v", {&q, &k, &v},
                {Out{IGNIS_TRELLIS_STORE, 128}, Out{IGNIS_TRELLIS_STORE, 0}, Out{IGNIS_TRELLIS_ADD, 0}}, 200, 41, 1);
  }
  // SwiGLU's pair, gate and up of different K.
  {
    Matrix gate = synthetic(5, 768, 384, 4000), up = synthetic(10, 768, 384, 4010);
    check_group("synthetic gate and up", {&gate, &up}, {Out{IGNIS_TRELLIS_STORE, 0}, Out{IGNIS_DENSE_PREFILL_SWIGLU_UP, 0}}, 270,
                51, 1);
  }
}

void artifact_arm() {
  const char *env = std::getenv("IGNIS_DENSE_PREFILL_ARTIFACT");
  const std::string path = env != nullptr && *env != '\0' ? env : kDefaultArtifact;
  Artifact art;
  if (!art.open(path)) {
    // A skip is never green in the GPU profile (docs/agents/testing.md).
    const char *profile = std::getenv("IGNIS_GPU_PROFILE");
    const std::string why = "artifact: no base 16 GB artifact at " + path + " (IGNIS_DENSE_PREFILL_ARTIFACT)";
    if (profile != nullptr && std::string(profile) == "1") {
      check(false, why);
    } else {
      std::printf("SKIP: %s\n", why.c_str());
    }
    return;
  }
  std::printf("artifact: %s\n", path.c_str());
  auto load = [&](const std::string &name, Matrix &m) {
    const bool ok = art.matrix(name, m);
    check(ok, "artifact: " + name + " is a trellis matrix of the artifact");
    return ok;
  };
  const int tokens = 520;  // two 256-token tiles and a tail, four 128-token tiles and a tail
  const int step = 8;
  {
    // K 2: layer 0's MLP, SwiGLU's pair.
    Matrix gate, up;
    if (load("text/layers/0/mlp/gate_proj", gate) && load("text/layers/0/mlp/up_proj", up)) {
      check_group("layer 0 gate and up", {&gate, &up}, {Out{IGNIS_TRELLIS_STORE, 0}, Out{IGNIS_DENSE_PREFILL_SWIGLU_UP, 0}}, tokens,
                  61, step);
    }
  }
  {
    // K 3: layer 1's GDN projections.
    Matrix qkv, z;
    if (load("text/layers/1/gdn/in_proj_qkv", qkv) && load("text/layers/1/gdn/in_proj_z", z)) {
      check_group("layer 1 qkv and z", {&qkv, &z}, {Out{}, Out{}}, tokens, 62, step);
    }
  }
  for (const char *layer : {"3", "15"}) {
    // K 4 (layer 3's k_proj) and K 5 (layer 15's v_proj): the attention's projections, the query
    // and gate split per head of 256.
    Matrix q, k, v;
    const std::string p = std::string("text/layers/") + layer + "/attention/";
    if (load(p + "q_proj", q) && load(p + "k_proj", k) && load(p + "v_proj", v)) {
      check_group(std::string("layer ") + layer + " q, k and v", {&k, &q, &v},
                  {Out{}, Out{IGNIS_TRELLIS_STORE, 256}, Out{}}, tokens, 63, step);
    }
  }
  {
    // The two other input widths: the MLP's down (17408, two A8 passes) and the attention's
    // output (6144), each adding into the residual.
    Matrix down, o;
    if (load("text/layers/3/mlp/down", down)) check_group("layer 3 down", {&down}, {Out{IGNIS_TRELLIS_ADD, 0}}, tokens, 64, step);
    if (load("text/layers/3/attention/output", o)) check_group("layer 3 output", {&o}, {Out{IGNIS_TRELLIS_ADD, 0}}, tokens, 65, step);
  }
  // GitHub #511: the fused producers on real matrices after their real norms.
  auto norm = [&](const std::string &name, std::vector<uint16_t> &v) {
    const bool ok = art.vector(name, v);
    check(ok, "artifact: " + name + " is a BF16 vector of the artifact");
    return ok;
  };
  {
    // Layer 1's GDN projections after its input norm.
    Matrix qkv, z;
    std::vector<uint16_t> n;
    if (load("text/layers/1/gdn/in_proj_qkv", qkv) && load("text/layers/1/gdn/in_proj_z", z) &&
        norm("text/layers/1/input_norm", n)) {
      check_norm_group("layer 1 input norm, qkv and z", {&qkv, &z}, {Out{}, Out{}}, n, tokens, 66);
    }
  }
  {
    // Layer 3's attention projections after its input norm, the query and gate split.
    Matrix q, k, v;
    std::vector<uint16_t> n;
    if (load("text/layers/3/attention/q_proj", q) && load("text/layers/3/attention/k_proj", k) &&
        load("text/layers/3/attention/v_proj", v) && norm("text/layers/3/input_norm", n)) {
      check_norm_group("layer 3 input norm, q, k and v", {&q, &k, &v}, {Out{IGNIS_TRELLIS_STORE, 256}, Out{}, Out{}}, n, tokens, 67);
    }
  }
  {
    // Layer 0's MLP after its post-attention norm: down's 17408 inputs.
    Matrix gate, up, down;
    std::vector<uint16_t> n;
    if (load("text/layers/0/mlp/gate_proj", gate) && load("text/layers/0/mlp/up_proj", up) &&
        load("text/layers/0/mlp/down", down) && norm("text/layers/0/post_attention_norm", n)) {
      check_mlp("layer 0 MLP", gate, up, down, n, tokens, 68);
    }
  }
}

void shared_workspace_arm() {
  std::printf("shared workspace: the fused op's decode route after a dense call\n");
  Matrix m = synthetic(6, 512, 256, 5000);
  const int tokens = 8;
  const std::vector<uint16_t> xh = standin_x(tokens, 512, 71);
  DeviceBytes d_x(xh.size() * 2), y_fresh(2ull * tokens * 256), y_after(2ull * tokens * 256);
  upload(d_x, xh);
  const uint64_t fused = ignis_trellis_workspace_bytes(512, 256, 1, tokens);
  const uint64_t dense = ignis_dense_prefill_workspace_bytes(512, 1, 300);
  DeviceBytes fresh(fused), shared(std::max(fused, dense));
  const ignis_trellis_workspace wf{fresh.p, fused}, wsh{shared.p, shared.bytes};
  TL_RC(ignis_trellis_workspace_init(&wf, nullptr));
  TL_RC(ignis_trellis_workspace_init(&wsh, nullptr));
  TL_RC(ignis_trellis_linear(&m.w, d_x.p, tokens, y_fresh.p, 256, IGNIS_TRELLIS_STORE, &wf, nullptr));
  // A dense call of 300 tokens over the shared workspace first.
  const std::vector<uint16_t> xd = standin_x(300, 512, 72);
  DeviceBytes d_xd(xd.size() * 2), d_yd(2ull * 300 * 256), d_dense(512 * 256);
  upload(d_xd, xd);
  const ignis_trellis_weight *wl[1] = {&m.w};
  const ignis_dense_prefill_output o{d_yd.p, 256, IGNIS_TRELLIS_STORE, 0, 0, nullptr};
  TL_RC(ignis_dense_prefill_linear(wl, 1, d_xd.p, 300, &o, &wsh, d_dense.p, d_dense.bytes, nullptr));
  TL_RC(ignis_trellis_linear(&m.w, d_x.p, tokens, y_after.p, 256, IGNIS_TRELLIS_STORE, &wsh, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<uint16_t>(y_fresh.p, tokens * 256) == download<uint16_t>(y_after.p, tokens * 256),
        "the fused decode route gives its fresh workspace's bits on a workspace a dense call used");
}

void rejects_arm() {
  std::printf("rejects\n");
  Matrix m = synthetic(6, 256, 256, 6000);
  const int tokens = 16;
  DeviceBytes x(2ull * tokens * 256), y(2ull * tokens * 256), y2(2ull * tokens * 256), dense(256 * 256);
  DeviceA8 a(tokens, 256);
  const ignis_trellis_weight *wl[4] = {&m.w, &m.w, &m.w, &m.w};
  ignis_dense_prefill_activations al[4] = {a.a, a.a, a.a, a.a};
  check(ignis_dense_prefill_quantize(x.p, tokens, wl, 0, al, nullptr) != 0, "a quantize of no matrix is refused");
  check(ignis_dense_prefill_quantize(x.p, tokens, wl, 4, al, nullptr) != 0, "a quantize of four matrices is refused");
  check(ignis_dense_prefill_quantize(x.p, 0, wl, 1, al, nullptr) != 0, "a quantize of zero tokens is refused");
  check(ignis_dense_prefill_quantize(static_cast<char *>(x.p) + 2, tokens, wl, 1, al, nullptr) != 0,
        "unaligned activations are refused");
  Matrix wide = synthetic(6, 384, 256, 6010);
  const ignis_trellis_weight *mixed[2] = {&m.w, &wide.w};
  check(ignis_dense_prefill_quantize(x.p, tokens, mixed, 2, al, nullptr) != 0, "a group of different input widths is refused");
  ignis_dense_prefill_activations bad = a.a;
  bad.q = static_cast<char *>(a.q.p) + 4;
  check(ignis_dense_prefill_quantize(x.p, tokens, wl, 1, &bad, nullptr) != 0, "unaligned INT8 activations are refused");
  check(ignis_dense_prefill_decode(&m.w, dense.p, 256 * 256 - 1, nullptr) != 0, "a dense copy one byte short is refused");
  ignis_trellis_weight odd = m.w;
  odd.in = 100;
  check(ignis_dense_prefill_decode(&odd, dense.p, dense.bytes, nullptr) != 0, "in = 100 is refused");
  odd = m.w;
  odd.k2 = 7;
  check(ignis_dense_prefill_decode(&odd, dense.p, dense.bytes, nullptr) != 0, "k2 = 7 is refused");
  ignis_dense_prefill_output o{y.p, 256, 2, 0, 0, nullptr};
  check(ignis_dense_prefill_gemm(&m.w, dense.p, &a.a, tokens, &o, nullptr) != 0, "an unknown epilogue is refused");
  o = ignis_dense_prefill_output{y.p, 254, IGNIS_TRELLIS_STORE, 0, 0, nullptr};
  check(ignis_dense_prefill_gemm(&m.w, dense.p, &a.a, tokens, &o, nullptr) != 0, "ldy below out is refused");
  o = ignis_dense_prefill_output{y.p, 128, IGNIS_TRELLIS_STORE, 64, 0, y2.p};
  check(ignis_dense_prefill_gemm(&m.w, dense.p, &a.a, tokens, &o, nullptr) != 0, "a split that is no multiple of 128 is refused");
  o = ignis_dense_prefill_output{y.p, 128, IGNIS_TRELLIS_STORE, 128, 0, nullptr};
  check(ignis_dense_prefill_gemm(&m.w, dense.p, &a.a, tokens, &o, nullptr) != 0, "a split without its second half is refused");
  check(ignis_dense_prefill_gemm(&m.w, dense.p, &a.a, tokens, nullptr, nullptr) != 0, "a GEMM without an output is refused");
  check(ignis_dense_prefill_accumulate(&m.w, dense.p, &a.a, tokens, static_cast<int32_t *>(y.p), 255, nullptr) != 0,
        "an accumulator row narrower than out is refused");
  o = ignis_dense_prefill_output{y.p, 256, IGNIS_TRELLIS_STORE, 0, 0, nullptr};
  const uint64_t need = ignis_dense_prefill_workspace_bytes(256, 1, tokens);
  DeviceBytes ws_mem(need);
  const ignis_trellis_workspace short_ws{ws_mem.p, need - 1};
  check(ignis_dense_prefill_linear(wl, 1, x.p, tokens, &o, &short_ws, dense.p, dense.bytes, nullptr) != 0,
        "a workspace one byte short is refused");
  check(ignis_dense_prefill_workspace_bytes(100, 1, 8) == 0, "a workspace query at in = 100 answers 0");
  check(ignis_dense_prefill_workspace_bytes(256, 4, 8) == 0, "a workspace query for four matrices answers 0");
  check(ignis_dense_prefill_weight_bytes(256, 100) == 0, "a dense size at out = 100 answers 0");
  check(ignis_dense_prefill_weight_bytes(5120, 17408) == 5120ull * 17408, "the dense copy of [17408 x 5120] is 89,128,960 bytes");

  // GitHub #511's ops.
  Matrix n2048 = synthetic(6, 2048, 128, 6020), n5248 = synthetic(6, 5248, 128, 6030), n4096 = synthetic(6, 4096, 128, 6040);
  DeviceBytes nx(2ull * tokens * 5248), nw(2ull * 5248);
  DeviceA8 na(tokens, 5248);
  const ignis_trellis_weight *w2048[1] = {&n2048.w}, *w5248[1] = {&n5248.w}, *w4096[1] = {&n4096.w};
  check(ignis_dense_prefill_rmsnorm_quantize(nx.p, nw.p, kEps, tokens, nullptr, w2048, 1, &na.a, nullptr) != 0,
        "the fused RMSnorm refuses in = 2048, below the vendored wide-row form");
  check(ignis_dense_prefill_rmsnorm_quantize(nx.p, nw.p, kEps, tokens, nullptr, w5248, 1, &na.a, nullptr) != 0,
        "the fused RMSnorm refuses in = 5248, no multiple of 1024");
  check(ignis_dense_prefill_rmsnorm_quantize(nx.p, nw.p, 0.0f, tokens, nullptr, w4096, 1, &na.a, nullptr) != 0,
        "the fused RMSnorm refuses eps = 0");
  check(ignis_dense_prefill_rmsnorm_quantize(nx.p, nw.p, NAN, tokens, nullptr, w4096, 1, &na.a, nullptr) != 0,
        "the fused RMSnorm refuses eps = NaN");
  check(ignis_dense_prefill_rmsnorm_quantize(nx.p, nw.p, kEps, 0, nullptr, w4096, 1, &na.a, nullptr) != 0,
        "the fused RMSnorm refuses zero tokens");
  check(ignis_dense_prefill_rmsnorm_quantize(static_cast<char *>(nx.p) + 2, nw.p, kEps, tokens, nullptr, w4096, 1, &na.a,
                                             nullptr) != 0,
        "the fused RMSnorm refuses an unaligned x");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  int major = 0, minor = 0;
  MOE_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  MOE_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
  std::printf("dense prefill vs exact references (compute capability %d.%d)\n", major, minor);
  TL_RC(ignis_trellis_prepare());
  TL_RC(ignis_dense_prefill_prepare());
  plan_arm();
  synthetic_arm();
  plans_arm();
  fused_producers_arm();
  shared_workspace_arm();
  rejects_arm();
  artifact_arm();
  std::printf("worst share of the output bound: %.3f\n", g_worst);
  if (moe_test::g_failed != 0) {
    std::fprintf(stderr, "test_dense_prefill: %d failure(s)\n", moe_test::g_failed);
    return 1;
  }
  std::printf("test_dense_prefill: OK\n");
  return 0;
}
