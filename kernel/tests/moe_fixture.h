// Shared host-side helpers for the Flash-Next MoE op tests -- OURS (spec flash-next/02, GitHub
// #300): the IGNFX001 fixture reader, the counter hash the recorder uses to regenerate inputs,
// BF16/FP16/E4M3 conversions, a host restatement of the trellis format, and fp64 references.
//
// The trellis decoder here is the format's definition restated on the host (the recorder,
// kernel/tests/fixtures/flash_next/record.py, holds the same definition and asserts it against
// exllamav3's `reconstruct`). The tests use it as the fp64 reference's weight source at real
// geometry, and first prove it against the recorded `reconstruct` output, so a wrong restatement
// fails a test instead of agreeing with a wrong kernel.
#ifndef IGNIS_MOE_FIXTURE_H
#define IGNIS_MOE_FIXTURE_H

#include "ignis_moe.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace moe_test {

inline int g_failed = 0;

inline void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

#define MOE_CUDA(expr)                                                                             \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#define MOE_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_moe_last_error());     \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// The decode route a test runs: "clusters" among its arguments selects IGNIS_MOE_DECODE_CLUSTERS
// (the device prepared first, so its cluster size is known); a device that runs no decode cluster
// fails the test rather than skipping it. Returns the route's name for the log.
inline std::string select_decode_route(int argc, char **argv) {
  bool clusters = false;
  for (int i = 1; i < argc; ++i) clusters = clusters || std::string(argv[i]) == "clusters";
  if (!clusters) return "tickets";
  MOE_RC(ignis_moe_prepare());
  const int32_t size = ignis_moe_decode_cluster_size();
  if (size == 0) {
    std::fprintf(stderr, "FATAL: this device runs no decode cluster (ignis_moe_decode_cluster_size() == 0)\n");
    std::exit(EXIT_FAILURE);
  }
  MOE_RC(ignis_moe_set_decode_route(IGNIS_MOE_DECODE_CLUSTERS));
  return "clusters of " + std::to_string(size) + " CTAs";
}

struct DeviceBytes {
  void *p = nullptr;
  std::size_t bytes = 0;
  DeviceBytes() = default;
  explicit DeviceBytes(std::size_t n) : bytes(n) {
    if (n > 0) { MOE_CUDA(cudaMalloc(&p, n)); }
  }
  ~DeviceBytes() {
    if (p != nullptr) { cudaFree(p); }
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
  template <typename T> T *as() const { return static_cast<T *>(p); }
};

template <typename T> void upload(DeviceBytes &d, const std::vector<T> &v) {
  MOE_CUDA(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> std::vector<T> download(const void *p, std::size_t n) {
  std::vector<T> v(n);
  MOE_CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

// ---- the counter hash (record.py: lowbias32 of i * 0x9E3779B9 + stream * 0x85EBCA6B) --------

inline uint32_t lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

inline uint32_t hash_u32(uint32_t stream, uint64_t i) {
  return lowbias32(static_cast<uint32_t>(i) * 0x9E3779B9u + stream * 0x85EBCA6Bu);
}

// float32 in [-amplitude, amplitude), computed exactly as record.py's hash_uniform.
inline float hash_uniform(uint32_t stream, uint64_t i, float amplitude) {
  const float u = static_cast<float>(hash_u32(stream, i) >> 8) * 0x1p-23f - 1.0f;
  return u * amplitude;
}

// ---- 16-bit floats -------------------------------------------------------------------------

inline uint16_t f32_to_bf16(float v) {  // round to nearest even (finite inputs)
  uint32_t b;
  std::memcpy(&b, &v, 4);
  b += 0x7FFFu + ((b >> 16) & 1u);
  return static_cast<uint16_t>(b >> 16);
}

inline float bf16_to_f32(uint16_t h) {
  const uint32_t b = static_cast<uint32_t>(h) << 16;
  float v;
  std::memcpy(&v, &b, 4);
  return v;
}

inline uint16_t f32_to_f16(float v) {
  const __half h = __float2half_rn(v);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

inline float f16_to_f32(uint16_t b) {
  __half h;
  std::memcpy(&h, &b, 2);
  return __half2float(h);
}

// E4M3FN (no infinities, 0x7F/0xFF NaN) to float.
inline float e4m3_to_f32(uint8_t c) {
  const int sign = (c >> 7) & 1;
  const int exp = (c >> 3) & 15;
  const int man = c & 7;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -9);
  } else if (exp == 15 && man == 7) {
    v = NAN;
  } else {
    v = std::ldexp(1.0f + man / 8.0f, exp - 7);
  }
  return sign ? -v : v;
}

// ---- the IGNFX001 container (record.py's Writer) ---------------------------------------------

struct FixtureTensor {
  uint32_t dtype = 0;
  std::vector<uint64_t> dims;
  std::vector<uint8_t> bytes;
  uint64_t count() const {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
  }
  template <typename T> std::vector<T> as() const {
    std::vector<T> v(bytes.size() / sizeof(T));
    std::memcpy(v.data(), bytes.data(), v.size() * sizeof(T));
    return v;
  }
};

inline std::map<std::string, FixtureTensor> read_fixture(const std::string &path) {
  std::map<std::string, FixtureTensor> out;
  FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "FATAL: fixture %s is missing\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "IGNFX001", 8) != 0) {
    std::fprintf(stderr, "FATAL: %s is not an IGNFX001 fixture\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  for (;;) {
    uint32_t name_len = 0;
    if (std::fread(&name_len, 4, 1, f) != 1) break;
    std::string name(name_len, '\0');
    uint32_t dtype = 0, ndim = 0;
    bool ok = std::fread(name.data(), 1, name_len, f) == name_len && std::fread(&dtype, 4, 1, f) == 1 &&
              std::fread(&ndim, 4, 1, f) == 1;
    FixtureTensor t;
    t.dtype = dtype;
    t.dims.resize(ndim);
    uint64_t nbytes = 0;
    ok = ok && (ndim == 0 || std::fread(t.dims.data(), 8, ndim, f) == ndim) && std::fread(&nbytes, 8, 1, f) == 1;
    t.bytes.resize(nbytes);
    ok = ok && (nbytes == 0 || std::fread(t.bytes.data(), 1, nbytes, f) == nbytes);
    if (!ok) {
      std::fprintf(stderr, "FATAL: %s is truncated at record '%s'\n", path.c_str(), name.c_str());
      std::exit(EXIT_FAILURE);
    }
    out.emplace(name, std::move(t));
  }
  std::fclose(f);
  return out;
}

inline const FixtureTensor &need(const std::map<std::string, FixtureTensor> &fx, const std::string &name) {
  const auto it = fx.find(name);
  if (it == fx.end()) {
    std::fprintf(stderr, "FATAL: fixture record '%s' is missing\n", name.c_str());
    std::exit(EXIT_FAILURE);
  }
  return it->second;
}

// ---- the trellis format, restated on the host (record.py's decode_inner) -------------------

inline int trellis_step_bits(uint32_t k2, int p) {
  return k2 == 5 ? (p % 2 == 1 ? 3 : 2) : static_cast<int>(k2 / 2);
}

inline uint32_t trellis_tile_words16(uint32_t k2) { return 8 * k2; }  // 16 * K int16 words

inline uint16_t mul1_f16_bits(uint32_t state) {
  const uint32_t prod = state * 0x83DCD12Du;
  const uint32_t bsum = (prod & 255u) + ((prod >> 8) & 255u) + ((prod >> 16) & 255u) + (prod >> 24);
  const float k_inv = f16_to_f32(0x1EEE);
  const float k_bias = f16_to_f32(0xC931);
  return f32_to_f16((1024.0f + static_cast<float>(bsum)) * k_inv + k_bias);  // exact, one rounding
}

// The inner (rotated-basis) fp16 matrix [in][out] of an exllamav3 trellis tensor
// [in/16][out/16][16 K] (int16 words, here as raw u16).
inline std::vector<uint16_t> host_trellis_decode(const uint16_t *words, uint32_t k2, int in, int out) {
  const int tn = out / 16;
  const uint32_t w16 = trellis_tile_words16(k2);
  const int bits = static_cast<int>(w16) * 16;
  int perm[256];
  for (int t = 0; t < 32; ++t) {
    const int r0 = (t % 4) * 2;
    const int c0 = t / 4;
    const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
    for (int j = 0; j < 8; ++j) perm[t * 8 + j] = rows[j % 4] * 16 + c0 + (j >= 4 ? 8 : 0);
  }
  int end[256];
  for (int p = 0, e = 0; p < 256; ++p) {
    e += trellis_step_bits(k2, p);
    end[p] = e;
  }
  std::vector<uint16_t> w(static_cast<std::size_t>(in) * out);
  for (int tk = 0; tk < in / 16; ++tk) {
    for (int tc = 0; tc < tn; ++tc) {
      const uint16_t *tile = words + (static_cast<std::size_t>(tk) * tn + tc) * w16;
      auto bit = [&](int b) {
        b = ((b % bits) + bits) % bits;
        const uint32_t u32 = tile[2 * (b / 32)] | (static_cast<uint32_t>(tile[2 * (b / 32) + 1]) << 16);
        return (u32 >> (31 - (b % 32))) & 1u;
      };
      for (int p = 0; p < 256; ++p) {
        uint32_t state = 0;
        for (int j = 0; j < 16; ++j) state = (state << 1) | bit(end[p] - 16 + j);
        const int r = perm[p] / 16;
        const int c = perm[p] % 16;
        w[static_cast<std::size_t>(tk * 16 + r) * out + tc * 16 + c] = mul1_f16_bits(state);
      }
    }
  }
  return w;
}

// record.py's checksum: sum_i u16[i] * (lowbias32(i) | 1) mod 2^64.
inline uint64_t checksum_u16(const std::vector<uint16_t> &v) {
  uint64_t s = 0;
  for (std::size_t i = 0; i < v.size(); ++i) {
    s += static_cast<uint64_t>(v[i]) * static_cast<uint64_t>(lowbias32(static_cast<uint32_t>(i)) | 1u);
  }
  return s;
}

// record.py's trellis_words: counter-hash words for a real-geometry projection.
inline std::vector<uint16_t> hash_trellis_words(uint32_t stream, int in, int out, uint32_t k2) {
  std::vector<uint16_t> w(static_cast<std::size_t>(in / 16) * (out / 16) * trellis_tile_words16(k2));
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<uint16_t>(hash_u32(stream, i) & 0xFFFFu);
  return w;
}

// ---- fp64 references -------------------------------------------------------------------------

// In-place Sylvester (natural order) Walsh-Hadamard over consecutive 128-blocks, scaled 1/sqrt(128):
// the block-diagonal H128/sqrt(128) of the trellis format.
inline void hadamard128_inplace(double *v, std::size_t n) {
  for (std::size_t b = 0; b < n; b += 128) {
    for (int h = 1; h < 128; h <<= 1) {
      for (int i = 0; i < 128; i += 2 * h) {
        for (int j = i; j < i + h; ++j) {
          const double a = v[b + j];
          const double c = v[b + j + h];
          v[b + j] = a + c;
          v[b + j + h] = a - c;
        }
      }
    }
    for (int i = 0; i < 128; ++i) v[b + i] /= std::sqrt(128.0);
  }
}

// y = had(had(x o suh) . inner) o svh for one token: the projection the record stands for.
inline std::vector<double> project_f64(const std::vector<uint16_t> &inner, const uint16_t *suh,
                                       const uint16_t *svh, int in, int out, const double *x) {
  std::vector<double> xs(in);
  for (int k = 0; k < in; ++k) xs[k] = x[k] * f16_to_f32(suh[k]);
  hadamard128_inplace(xs.data(), in);
  std::vector<double> y(out, 0.0);
  for (int k = 0; k < in; ++k) {
    const uint16_t *row = inner.data() + static_cast<std::size_t>(k) * out;
    for (int n = 0; n < out; ++n) y[n] += xs[k] * f16_to_f32(row[n]);
  }
  hadamard128_inplace(y.data(), out);
  for (int n = 0; n < out; ++n) y[n] *= f16_to_f32(svh[n]);
  return y;
}

inline double silu(double v) { return v / (1.0 + std::exp(-v)); }

}  // namespace moe_test

#endif  // IGNIS_MOE_FIXTURE_H
