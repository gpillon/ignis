// GitHub #302 (spec flash-next/04): Flash-Next's hyper-connection mix and
// inject (kernel/src/flash_next/hc.cu) -- OURS, not vendored.
//
// At Flash-Next's geometry (4 streams of 2560, rank 320), BF16 weights,
// against an fp64 reference of transformers' Qwen4ExpTextGatedResidual that
// rounds to BF16 exactly where the BF16 module does (every module output) and
// computes in fp64 inside each module:
// - the mix's output and injection weights within a few BF16 ulps (the only
//   difference is fp32 against fp64 accumulation, which can move a rounding
//   by one ulp and carry it through sigmoid and the stream mean);
// - the final mixer (no inject weights) the same;
// - the inject, given the same weights, bit for bit.
// Rows 1, 3 (decode lanes) and 1100 (past one 1024-row wave).
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "flash_next/hc.h"

#include "ignis_fp8_linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

void cuda_ok(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

uint32_t lcg(uint32_t &state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

uint16_t bf16_bits(double v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t rounding = 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

double bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

double bf(double v) {
  return bf16_value(bf16_bits(v));
}

std::vector<uint16_t> random_bf16(std::size_t n, double scale, uint32_t seed) {
  std::vector<uint16_t> out(n);
  for (auto &v : out) {
    const double u = static_cast<double>(lcg(seed) >> 8) / 16777216.0 * 2.0 - 1.0;
    v = bf16_bits(u * scale);
  }
  return out;
}

template <typename T>
T *upload(const std::vector<T> &host) {
  T *dev = nullptr;
  cuda_ok(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
  cuda_ok(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return dev;
}

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kWidth = kStreams * kHidden;
constexpr int kRank = 320;
constexpr double kEps = 1e-6;

struct Weights {
  std::vector<uint16_t> norm, down, up, inject;
};

// The BF16 module, in fp64 between its roundings.
void reference_mix(const Weights &w, const std::vector<uint16_t> &hidden, int row, bool with_inject,
                   std::vector<double> &x, std::vector<double> &inj) {
  std::vector<double> normed(kWidth);
  for (int s = 0; s < kStreams; ++s) {
    double squares = 0.0;
    for (int h = 0; h < kHidden; ++h) {
      const double v = bf16_value(hidden[static_cast<std::size_t>(row) * kWidth + s * kHidden + h]);
      squares += v * v;
    }
    const double inv = 1.0 / std::sqrt(squares / kHidden + kEps);
    for (int h = 0; h < kHidden; ++h) {
      const int i = s * kHidden + h;
      normed[i] = bf(bf16_value(hidden[static_cast<std::size_t>(row) * kWidth + i]) * inv *
                     (1.0 + bf16_value(w.norm[i])));
    }
  }
  std::vector<double> act(kRank);
  for (int k = 0; k < kRank; ++k) {
    double d = 0.0;
    for (int i = 0; i < kWidth; ++i) {
      d += bf16_value(w.down[static_cast<std::size_t>(k) * kWidth + i]) * normed[i];
    }
    const double scaled = bf(bf(d) / kStreams);
    act[k] = bf(scaled / (1.0 + std::exp(-scaled)));
  }
  x.assign(kHidden, 0.0);
  for (int h = 0; h < kHidden; ++h) {
    double sum = 0.0;
    for (int s = 0; s < kStreams; ++s) {
      const int i = s * kHidden + h;
      double u = 0.0;
      for (int k = 0; k < kRank; ++k) {
        u += bf16_value(w.up[static_cast<std::size_t>(i) * kRank + k]) * act[k];
      }
      const double m = bf(1.0 / (1.0 + std::exp(-bf(u))));
      sum += bf(m * normed[i]);
    }
    x[h] = bf(sum / kStreams);
  }
  inj.assign(kStreams, 0.0);
  if (with_inject) {
    for (int s = 0; s < kStreams; ++s) {
      double raw = 0.0;
      for (int i = 0; i < kWidth; ++i) {
        raw += bf16_value(w.inject[static_cast<std::size_t>(s) * kWidth + i]) * normed[i];
      }
      const double gate = bf(1.0 / (1.0 + std::exp(-bf(bf(raw) / kStreams))));
      inj[s] = bf(2.0 * gate);
    }
  }
}

void mix_case(const Weights &w, const ignis::flash_next::HcWeights &dw, int rows, bool with_inject,
              ninfer::DeviceArena &scratch) {
  const std::string label = std::string(with_inject ? "mix" : "final mixer") + ", " +
                            std::to_string(rows) + " rows";
  const auto hidden = random_bf16(static_cast<std::size_t>(rows) * kWidth, 2.0, 0x4C0U + rows);
  auto *d_hidden = upload(hidden);
  void *d_x = nullptr;
  float *d_inj = nullptr;
  cuda_ok(cudaMalloc(&d_x, static_cast<std::size_t>(rows) * kHidden * 2), "cudaMalloc x");
  cuda_ok(cudaMalloc(&d_inj, static_cast<std::size_t>(rows) * kStreams * 4), "cudaMalloc inj");
  ignis::flash_next::Geometry g;
  g.streams = kStreams;
  g.hidden = kHidden;
  g.hc_rank = kRank;
  g.rms_norm_eps = static_cast<float>(kEps);
  check(scratch.capacity() >= ignis::flash_next::fn_hc_mix_scratch_bytes(g, rows),
        label + ": the scratch the test reserved holds the mix's own figure");
  ignis::flash_next::HcWeights weights = dw;
  if (!with_inject) {
    weights.block_inject = nullptr;
  }
  check(ignis::flash_next::fn_hc_mix(g, weights, d_hidden, rows, d_x, with_inject ? d_inj : nullptr,
                                     scratch, nullptr) == 0,
        label + ": runs: " + ignis::flash_next::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> x(static_cast<std::size_t>(rows) * kHidden);
  std::vector<float> inj(static_cast<std::size_t>(rows) * kStreams);
  cuda_ok(cudaMemcpy(x.data(), d_x, x.size() * 2, cudaMemcpyDeviceToHost), "download x");
  cuda_ok(cudaMemcpy(inj.data(), d_inj, inj.size() * 4, cudaMemcpyDeviceToHost), "download inj");

  // Check a spread of rows (all of them when few): the reference is slow.
  double worst = 0.0;
  int bad = 0;
  for (int row = 0; row < rows; row += rows > 8 ? 137 : 1) {
    std::vector<double> rx, rinj;
    reference_mix(w, hidden, row, with_inject, rx, rinj);
    double rms = 0.0;
    for (double v : rx) {
      rms += v * v;
    }
    rms = std::sqrt(rms / kHidden);
    for (int h = 0; h < kHidden; ++h) {
      const double got = bf16_value(x[static_cast<std::size_t>(row) * kHidden + h]);
      const double err = std::fabs(got - rx[h]) / (std::fabs(rx[h]) + 1e-3 * rms);
      worst = std::max(worst, err);
      bad += err > std::ldexp(1.0, -6) ? 1 : 0;
    }
    for (int s = 0; with_inject && s < kStreams; ++s) {
      bad += std::fabs(inj[static_cast<std::size_t>(row) * kStreams + s] - rinj[s]) >
                     std::ldexp(std::fabs(rinj[s]), -6)
                 ? 1
                 : 0;
    }
  }
  check(bad == 0, label + ": " + std::to_string(bad) + " values past 2^-6 of the reference");
  std::printf("  %s: worst relative error %.2e\n", label.c_str(), worst);

  // The inject, given these weights: bit for bit.
  if (with_inject) {
    const auto y = random_bf16(static_cast<std::size_t>(rows) * kHidden, 1.0, 0x1E7U + rows);
    auto *d_y = upload(y);
    check(ignis::flash_next::fn_hc_inject(g, d_y, d_inj, rows, d_hidden, nullptr) == 0,
          label + ": inject runs: " + ignis::flash_next::fn_last_error());
    cuda_ok(cudaDeviceSynchronize(), "sync");
    std::vector<uint16_t> out(hidden.size());
    cuda_ok(cudaMemcpy(out.data(), d_hidden, out.size() * 2, cudaMemcpyDeviceToHost), "download hidden");
    int mismatched = 0;
    for (int row = 0; row < rows; ++row) {
      for (int s = 0; s < kStreams; ++s) {
        for (int h = 0; h < kHidden; ++h) {
          const std::size_t i = static_cast<std::size_t>(row) * kWidth + s * kHidden + h;
          const double injection = bf(bf16_value(y[static_cast<std::size_t>(row) * kHidden + h]) *
                                      inj[static_cast<std::size_t>(row) * kStreams + s]);
          mismatched += bf16_bits(bf16_value(hidden[i]) + injection) != out[i] ? 1 : 0;
        }
      }
    }
    check(mismatched == 0, label + ": inject differs from BF16 add-of-product in " +
                               std::to_string(mismatched) + " elements");
    cudaFree(d_y);
  }
  cudaFree(d_hidden);
  cudaFree(d_x);
  cudaFree(d_inj);
}

}  // namespace

int main() {
  int devices = 0;
  cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return 1;
  }
  Weights w;
  w.norm = random_bf16(kWidth, 0.2, 1);
  w.down = random_bf16(static_cast<std::size_t>(kRank) * kWidth, 0.02, 2);
  w.up = random_bf16(static_cast<std::size_t>(kWidth) * kRank, 0.1, 3);
  w.inject = random_bf16(static_cast<std::size_t>(kStreams) * kWidth, 0.02, 4);
  ignis::flash_next::HcWeights dw;
  dw.hc_norm = upload(w.norm);
  dw.mix_down = {upload(w.down), kRank, kWidth, ignis::flash_next::WeightFormat::Bf16};
  dw.mix_up = {upload(w.up), kWidth, kRank, ignis::flash_next::WeightFormat::Bf16};
  dw.block_inject = upload(w.inject);

  ninfer::DeviceArena scratch(64u << 20);
  for (int rows : {1, 3, 1100}) {
    mix_case(w, dw, rows, true, scratch);
  }
  mix_case(w, dw, 3, false, scratch);

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next hyper-connection test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next hyper-connection test: ok\n");
  return 0;
}
