// GitHub #302 (spec flash-next/04): the Flash-Next program's linear,
// fn_linear (kernel/src/flash_next/linear.cu) -- OURS, not vendored.
//
// Both stored formats at the Flash-Next shapes the program runs them at, on
// both routes (the GEMV up to 8 tokens, the tensor cores beyond), against an
// fp64 reference of the same stored values:
// - BF16 [rows][cols]: the route every linear has, because a part whose FP8
//   cost the conversion flags is re-converted to BF16;
// - FP8 row-scale (layout.md 6.1): routed to kern's ignis_fp8_linear, checked
//   here only for the routing and its refusals, its numerics being kern's own
//   test's.
// The tolerance is the rigorous one for fp32 accumulation of `cols` exact
// products (BF16 x BF16 and E4M3 x BF16 are exact in fp32): cols * 2^-24 *
// sum|w x|, plus one BF16 rounding (2^-8 |ref|) for a BF16 output.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "flash_next/flash_next_internal.h"

#include "ignis_fp8_linear.h"

#include <cuda_bf16.h>
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

uint16_t bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  const uint32_t rounding = 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

double bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

// E4M3FN (bias 7, no infinities, 0x7F/0xFF are NaN).
double e4m3_value(uint8_t code) {
  const int sign = (code & 0x80) ? -1 : 1;
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  if (exponent == 0) {
    return sign * std::ldexp(mantissa / 8.0, -6);
  }
  return sign * std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
}

std::vector<uint16_t> random_bf16(std::size_t n, float scale, uint32_t seed) {
  std::vector<uint16_t> out(n);
  for (auto &v : out) {
    const float u = static_cast<float>(lcg(seed) >> 8) / 16777216.0F * 2.0F - 1.0F;
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

// Runs one BF16 shape at one token count and compares both outputs (BF16, fp32).
void bf16_case(int32_t rows, int32_t cols, int32_t tokens, ninfer::DeviceArena &scratch) {
  const std::string label = "BF16 [" + std::to_string(rows) + "," + std::to_string(cols) + "] x " +
                            std::to_string(tokens) + " tokens";
  const auto w = random_bf16(static_cast<std::size_t>(rows) * cols, 0.05F, 0x302U + rows);
  const auto x = random_bf16(static_cast<std::size_t>(tokens) * cols, 1.0F, 0x7U + tokens);
  auto *dw = upload(w);
  auto *dx = upload(x);
  void *dy16 = nullptr;
  void *dy32 = nullptr;
  cuda_ok(cudaMalloc(&dy16, static_cast<std::size_t>(tokens) * rows * 2), "cudaMalloc y16");
  cuda_ok(cudaMalloc(&dy32, static_cast<std::size_t>(tokens) * rows * 4), "cudaMalloc y32");
  ignis::flash_next::Linear linear{dw, rows, cols, ignis::flash_next::WeightFormat::Bf16};
  check(ignis::flash_next::fn_linear(linear, dx, tokens, dy16, false, scratch, nullptr) == 0,
        label + ": BF16 out runs: " + ignis::flash_next::fn_last_error());
  check(ignis::flash_next::fn_linear(linear, dx, tokens, dy32, true, scratch, nullptr) == 0,
        label + ": fp32 out runs: " + ignis::flash_next::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> y16(static_cast<std::size_t>(tokens) * rows);
  std::vector<float> y32(static_cast<std::size_t>(tokens) * rows);
  cuda_ok(cudaMemcpy(y16.data(), dy16, y16.size() * 2, cudaMemcpyDeviceToHost), "download");
  cuda_ok(cudaMemcpy(y32.data(), dy32, y32.size() * 4, cudaMemcpyDeviceToHost), "download");

  double worst = 0.0;
  int bad = 0;
  for (int32_t t = 0; t < tokens; ++t) {
    for (int32_t r = 0; r < rows; ++r) {
      double ref = 0.0;
      double mag = 0.0;
      for (int32_t c = 0; c < cols; ++c) {
        const double p = bf16_value(w[static_cast<std::size_t>(r) * cols + c]) *
                         bf16_value(x[static_cast<std::size_t>(t) * cols + c]);
        ref += p;
        mag += std::fabs(p);
      }
      const std::size_t i = static_cast<std::size_t>(t) * rows + r;
      const double accumulation = static_cast<double>(cols) * std::ldexp(mag, -24);
      const double e32 = std::fabs(y32[i] - ref);
      const double e16 = std::fabs(bf16_value(y16[i]) - ref);
      worst = std::max(worst, e32 / (accumulation + 1e-30));
      if (e32 > accumulation || e16 > std::ldexp(std::fabs(ref), -8) + accumulation) {
        ++bad;
      }
    }
  }
  check(bad == 0, label + ": " + std::to_string(bad) + " outputs outside the fp32/BF16 bound");
  std::printf("  %s: worst fp32 error %.3f of its bound\n", label.c_str(), worst);
  cudaFree(dw);
  cudaFree(dx);
  cudaFree(dy16);
  cudaFree(dy32);
}

// An FP8 row-scale payload of `rows` x `cols`: random finite codes, then the
// BF16 scales at the next multiple of 256 bytes.
std::vector<uint8_t> fp8_payload(int32_t rows, int32_t cols, uint32_t seed,
                                 std::vector<double> &decoded) {
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scales_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> payload(scales_at + static_cast<std::size_t>(rows) * 2, 0);
  decoded.assign(codes, 0.0);
  std::vector<double> scale(rows);
  for (int32_t r = 0; r < rows; ++r) {
    const uint16_t bits = bf16_bits(0.01F + static_cast<float>(lcg(seed) % 100) * 1e-4F);
    std::memcpy(&payload[scales_at + static_cast<std::size_t>(r) * 2], &bits, 2);
    scale[r] = bf16_value(bits);
  }
  for (std::size_t i = 0; i < codes; ++i) {
    uint8_t code = static_cast<uint8_t>(lcg(seed) >> 24);
    if ((code & 0x7F) == 0x7F) {
      code = static_cast<uint8_t>(code - 1);  // never NaN
    }
    payload[i] = code;
    decoded[i] = e4m3_value(code) * scale[i / cols];
  }
  return payload;
}

void fp8_case(int32_t rows, int32_t cols, int32_t tokens, ninfer::DeviceArena &scratch) {
  const std::string label = "FP8 [" + std::to_string(rows) + "," + std::to_string(cols) + "] x " +
                            std::to_string(tokens) + " tokens";
  std::vector<double> w;
  const auto payload = fp8_payload(rows, cols, 0xF8U + rows, w);
  const auto x = random_bf16(static_cast<std::size_t>(tokens) * cols, 1.0F, 0x11U + tokens);
  auto *dw = upload(payload);
  auto *dx = upload(x);
  void *dy = nullptr;
  cuda_ok(cudaMalloc(&dy, static_cast<std::size_t>(tokens) * rows * 4), "cudaMalloc y");
  ignis::flash_next::Linear linear{dw, rows, cols, ignis::flash_next::WeightFormat::Fp8RowScale};
  check(ignis::flash_next::fn_linear(linear, dx, tokens, dy, true, scratch, nullptr) == 0,
        label + ": runs: " + ignis::flash_next::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<float> y(static_cast<std::size_t>(tokens) * rows);
  cuda_ok(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "download");
  int bad = 0;
  for (int32_t t = 0; t < tokens; ++t) {
    for (int32_t r = 0; r < rows; ++r) {
      double ref = 0.0;
      double mag = 0.0;
      for (int32_t c = 0; c < cols; ++c) {
        const double p = w[static_cast<std::size_t>(r) * cols + c] * bf16_value(x[static_cast<std::size_t>(t) * cols + c]);
        ref += p;
        mag += std::fabs(p);
      }
      const double bound = static_cast<double>(cols) * std::ldexp(mag, -24) + std::ldexp(std::fabs(ref), -23);
      if (std::fabs(y[static_cast<std::size_t>(t) * rows + r] - ref) > bound) {
        ++bad;
      }
    }
  }
  check(bad == 0, label + ": " + std::to_string(bad) + " outputs off the fp64 reference");
  cudaFree(dw);
  cudaFree(dx);
  cudaFree(dy);
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
  ninfer::DeviceArena scratch(1 << 20);

  // The shapes the program runs: the HC mix's down and up, the GDN a/b pair,
  // the PLE value projection; 1/3/8 tokens on the GEMV, 9/70/300 on the tensor
  // cores (300: a ragged last tile).
  const int32_t shapes[][2] = {{320, 10240}, {10240, 320}, {96, 2560}, {2560, 2560}};
  for (const auto &shape : shapes) {
    for (int32_t tokens : {1, 3, 8, 9, 70, 300}) {
      bf16_case(shape[0], shape[1], tokens, scratch);
    }
  }
  for (int32_t tokens : {1, 3, 70}) {
    fp8_case(320, 10240, tokens, scratch);
    fp8_case(2560, 2560, tokens, scratch);
  }

  // Refusals, named.
  ignis::flash_next::Linear odd{reinterpret_cast<void *>(0x1000), 16, 60,
                                ignis::flash_next::WeightFormat::Bf16};
  check(ignis::flash_next::fn_linear(odd, reinterpret_cast<void *>(0x1000), 1,
                                     reinterpret_cast<void *>(0x1000), false, scratch, nullptr) != 0 &&
            std::string(ignis::flash_next::fn_last_error()).find("multiple of 8") != std::string::npos,
        "a BF16 weight whose rows are not 16-byte multiples is refused by name");
  ignis::flash_next::Linear empty{};
  check(ignis::flash_next::fn_linear(empty, nullptr, 1, nullptr, false, scratch, nullptr) != 0,
        "a null weight is refused");

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next linear test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next linear test: ok\n");
  return 0;
}
