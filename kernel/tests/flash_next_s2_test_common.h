// Shared pieces of slice S2's kernel-leaf tests (GitHub #302, spec flash-next/04) -- OURS: FP8
// row-scale projections at the real geometry kept as their device payload plus a code LUT (tens
// of millions of weights, so no fp64 copy), their fp64 product on host threads, and BF16 helpers.
#ifndef IGNIS_FLASH_NEXT_S2_TEST_COMMON_H
#define IGNIS_FLASH_NEXT_S2_TEST_COMMON_H

#include "fp8_test_common.h"

#include "flash_next/flash_next_internal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#define FN_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_,                              \
                   ignis::flash_next::fn_last_error());                                            \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace s2_test {

using namespace moe_test;

inline double bf16r(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

inline double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// E4M3FN value of every code.
inline const double *e4m3_lut() {
  static const std::vector<double> lut = [] {
    std::vector<double> l(256);
    for (int c = 0; c < 256; ++c) l[c] = e4m3_to_f32(static_cast<uint8_t>(c));
    return l;
  }();
  return lut.data();
}

struct Proj {
  int rows = 0, cols = 0;
  std::vector<uint8_t> payload;  // codes, padding to 256, BF16 scales
  std::vector<double> scale;
  std::unique_ptr<DeviceBytes> dev;
};

// Random FP8 weights scaled so that inputs of rms `input_rms` give outputs of about `output_rms`.
inline void make_proj(Proj &p, uint32_t stream, int rows, int cols, double input_rms, double output_rms) {
  const double *lut = e4m3_lut();
  double s = 0.0;
  int n = 0;
  for (int c = 0; c < 256; ++c) {
    if ((c & 0x7F) == 0x7F) continue;
    s += lut[c] * lut[c];
    ++n;
  }
  const double mag = output_rms / (std::sqrt(static_cast<double>(cols)) * std::sqrt(s / n) * input_rms);
  Fp8Matrix m = make_fp8(stream, rows, cols, static_cast<float>(mag));
  p.rows = rows;
  p.cols = cols;
  p.payload = std::move(m.payload);
  p.scale = std::move(m.scale);
  p.dev = std::make_unique<DeviceBytes>(p.payload.size());
  upload(*p.dev, p.payload);
}

inline ignis::flash_next::Linear linear(const Proj &p) {
  ignis::flash_next::Linear l;
  l.data = p.dev->p;
  l.rows = p.rows;
  l.cols = p.cols;
  l.format = ignis::flash_next::WeightFormat::Fp8RowScale;
  return l;
}

// W[r][c] in fp64.
inline double weight(const Proj &p, int r, int c) {
  return e4m3_lut()[p.payload[static_cast<std::size_t>(r) * p.cols + c]] * p.scale[r];
}

// y[r] = scale[r] * sum_c code[r][c] * x[c] in fp64, rows split over host threads; with `bound`,
// also each row's fp32 accumulation bound as the FP8 linear's own test states it
// (fp8_sum_bound: (cols / 8 + 64) u sum |terms|).
inline std::vector<double> project(const Proj &p, const std::vector<double> &x, std::vector<double> *bound = nullptr) {
  const double *lut = e4m3_lut();
  std::vector<double> y(p.rows);
  if (bound != nullptr) bound->assign(p.rows, 0.0);
  constexpr int kThreads = 8;
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; ++t) {
    pool.emplace_back([&, t]() {
      for (int r = t; r < p.rows; r += kThreads) {
        const uint8_t *row = &p.payload[static_cast<std::size_t>(r) * p.cols];
        double s = 0.0, a = 0.0;
        for (int c = 0; c < p.cols; ++c) {
          s += lut[row[c]] * x[c];
          a += std::fabs(lut[row[c]] * x[c]);
        }
        y[r] = s * p.scale[r];
        if (bound != nullptr) (*bound)[r] = fp8_sum_bound(p.cols, a * std::fabs(p.scale[r]));
      }
    });
  }
  for (auto &t : pool) t.join();
  return y;
}

// n BF16 values uniform in [lo, hi), from the counter hash.
inline std::vector<uint16_t> bf16_vector(uint32_t stream, std::size_t n, double lo, double hi, uint64_t first = 0) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double u = 0.5 * (hash_uniform(stream, first + i, 1.0F) + 1.0);
    v[i] = f32_to_bf16(static_cast<float>(lo + (hi - lo) * u));
  }
  return v;
}

inline std::vector<double> as_double(const uint16_t *v, std::size_t n) {
  std::vector<double> d(n);
  for (std::size_t i = 0; i < n; ++i) d[i] = bf16_to_f32(v[i]);
  return d;
}

inline std::unique_ptr<DeviceBytes> device_copy(const std::vector<uint16_t> &host) {
  auto d = std::make_unique<DeviceBytes>(host.size() * 2);
  upload(*d, host);
  return d;
}

}  // namespace s2_test

#endif  // IGNIS_FLASH_NEXT_S2_TEST_COMMON_H
