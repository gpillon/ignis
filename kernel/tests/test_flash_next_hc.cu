// GitHub #302 (spec flash-next/04): Flash-Next's hyper-connection mix and
// inject (kernel/src/flash_next/hc.cu) -- OURS, not vendored.
//
// At Flash-Next's geometry (4 streams of 2560, rank 320), BF16 weights,
// against an fp64 reference of transformers' Qwen4ExpTextGatedResidual that
// rounds to BF16 exactly where the BF16 module does (every module output) and
// computes in fp64 inside each module:
// - the mix's output and injection weights within a few BF16 ulps (the only
//   difference is fp32 against fp64 accumulation, which can move a rounding
//   by one ulp and carry it through sigmoid and the stream mean). An output
//   is a mean of four streams' terms that can cancel, so its error is
//   measured against the terms' mean magnitude, not against the mean itself:
//   one term's one-ulp move is up to 2^-8 of that term, many times more of
//   a result the terms cancel down to (window 1, 2026-10-05: 4.5e-2 and
//   7.7e-2 against the result, reproduced on the CPU by fp32 against fp64
//   accumulation, which measured 6.4e-3 against the terms);
// - the final mixer (no inject weights) the same;
// - the inject, given the same weights, bit for bit;
// - a mix replayed gives the same bits (partials are summed in a fixed order).
// mix_down / mix_up in both stored formats: BF16, and FP8 row-scale as the
// artifact stores them (the reference's weights are then scale * e4m3(code)).
// Rows 1, 3, 8 (decode lanes, up to the decode route's ceiling), 9 (the first
// row count past it) and 1100 (past one 1024-row wave).
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "flash_next/hc.h"

#include "ignis_fp8_linear.h"

#include <cuda_fp8.h>
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

double e4m3_value(uint8_t code) {
  const int sign = (code & 0x80) ? -1 : 1;
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  if (exponent == 0) {
    return sign * std::ldexp(mantissa / 8.0, -6);
  }
  return sign * std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
}

// An FP8 row-scale payload (layout.md 6.1) of uniform values in +-`scale`:
// codes [rows][cols], then BF16 scales at the next multiple of 256 bytes.
// `value` gets each weight as the reference reads it, scale * e4m3(code).
std::vector<uint8_t> fp8_payload(int rows, int cols, double scale, uint32_t seed, std::vector<double> &value) {
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scales_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> payload(scales_at + static_cast<std::size_t>(rows) * 2, 0);
  value.assign(codes, 0.0);
  for (int r = 0; r < rows; ++r) {
    const uint16_t bits = bf16_bits(scale / 448.0 * (0.75 + static_cast<double>(lcg(seed) % 64) / 128.0));
    std::memcpy(&payload[scales_at + static_cast<std::size_t>(r) * 2], &bits, 2);
    for (int c = 0; c < cols; ++c) {
      const double u = static_cast<double>(lcg(seed) >> 8) / 16777216.0 * 2.0 - 1.0;
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      payload[i] = __nv_cvt_float_to_fp8(static_cast<float>(u * 448.0), __NV_SATFINITE, __NV_E4M3);
      value[i] = e4m3_value(payload[i]) * bf16_value(bits);
    }
  }
  return payload;
}

std::vector<double> bf16_values(const std::vector<uint16_t> &bits) {
  std::vector<double> out(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) {
    out[i] = bf16_value(bits[i]);
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

// The reference's weights: hc_norm and block_inject BF16, mix_down / mix_up as
// the device's format decodes them.
struct Weights {
  std::vector<uint16_t> norm, inject;
  std::vector<double> down, up;
};

// The BF16 module, in fp64 between its roundings.
// `terms[h]` is the mean magnitude of the four stream terms x[h] averages.
void reference_mix(const Weights &w, const std::vector<uint16_t> &hidden, int row, bool with_inject,
                   std::vector<double> &x, std::vector<double> &terms, std::vector<double> &inj) {
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
      d += w.down[static_cast<std::size_t>(k) * kWidth + i] * normed[i];
    }
    const double scaled = bf(bf(d) / kStreams);
    act[k] = bf(scaled / (1.0 + std::exp(-scaled)));
  }
  x.assign(kHidden, 0.0);
  terms.assign(kHidden, 0.0);
  for (int h = 0; h < kHidden; ++h) {
    double sum = 0.0;
    double magnitude = 0.0;
    for (int s = 0; s < kStreams; ++s) {
      const int i = s * kHidden + h;
      double u = 0.0;
      for (int k = 0; k < kRank; ++k) {
        u += w.up[static_cast<std::size_t>(i) * kRank + k] * act[k];
      }
      const double m = bf(1.0 / (1.0 + std::exp(-bf(u))));
      sum += bf(m * normed[i]);
      magnitude += std::fabs(bf(m * normed[i]));
    }
    x[h] = bf(sum / kStreams);
    terms[h] = magnitude / kStreams;
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

void mix_case(const Weights &w, const ignis::flash_next::HcWeights &dw, const char *format, int rows,
              bool with_inject, ninfer::DeviceArena &scratch) {
  const std::string label = std::string(with_inject ? "mix" : "final mixer") + " (" + format + "), " +
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

  // Replayed, the same bits.
  check(ignis::flash_next::fn_hc_mix(g, weights, d_hidden, rows, d_x, with_inject ? d_inj : nullptr,
                                     scratch, nullptr) == 0,
        label + ": runs again: " + ignis::flash_next::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> x2(x.size());
  std::vector<float> inj2(inj.size());
  cuda_ok(cudaMemcpy(x2.data(), d_x, x2.size() * 2, cudaMemcpyDeviceToHost), "download x");
  cuda_ok(cudaMemcpy(inj2.data(), d_inj, inj2.size() * 4, cudaMemcpyDeviceToHost), "download inj");
  check(x2 == x && (!with_inject || std::memcmp(inj2.data(), inj.data(), inj.size() * 4) == 0),
        label + ": a second run gives the same bits");

  // Check a spread of rows (all of them when few) and the last three (a
  // narrow last wave takes the decode route): the reference is slow.
  std::vector<int> checked;
  for (int row = 0; row < rows; row += rows > 16 ? 137 : 1) {
    checked.push_back(row);
  }
  for (int row = std::max(checked.back() + 1, rows - 3); row < rows; ++row) {
    checked.push_back(row);
  }
  double worst = 0.0;
  int bad = 0;
  for (int row : checked) {
    std::vector<double> rx, rterms, rinj;
    reference_mix(w, hidden, row, with_inject, rx, rterms, rinj);
    double rms = 0.0;
    for (double v : rx) {
      rms += v * v;
    }
    rms = std::sqrt(rms / kHidden);
    for (int h = 0; h < kHidden; ++h) {
      const double got = bf16_value(x[static_cast<std::size_t>(row) * kHidden + h]);
      const double err = std::fabs(got - rx[h]) / (rterms[h] + 1e-3 * rms);
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
  std::printf("  %s: worst error against the terms %.2e\n", label.c_str(), worst);

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

// The kernel nodes one fn_hc_mix call captures into a graph.
int kernel_nodes(const ignis::flash_next::Geometry &g, const ignis::flash_next::HcWeights &w, const void *hidden,
                 int rows, void *x, float *inj, ninfer::DeviceArena &scratch) {
  cudaStream_t stream = nullptr;
  cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
  cudaGraph_t graph = nullptr;
  cuda_ok(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
  const int32_t rc = ignis::flash_next::fn_hc_mix(g, w, hidden, rows, x, inj, scratch, stream);
  cuda_ok(cudaStreamEndCapture(stream, &graph), "end capture");
  std::size_t count = 0;
  cuda_ok(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
  std::vector<cudaGraphNode_t> nodes(count);
  cuda_ok(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
  int kernels = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    cuda_ok(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    kernels += type == cudaGraphNodeTypeKernel ? 1 : 0;
  }
  cuda_ok(cudaGraphDestroy(graph), "cudaGraphDestroy");
  cuda_ok(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return rc == 0 ? kernels : -1;
}

// GitHub #306 (fusion study): the decode route with its norm folded into the down projection
// (fn_hc_set_decode_fused) gives the three-launch route's bits, in two launches at up to three
// rows; past three rows the switch changes nothing.
void fused_case(const ignis::flash_next::HcWeights &dw, const char *format, int rows, bool with_inject,
                ninfer::DeviceArena &scratch) {
  const std::string label = std::string("fused norm, ") + (with_inject ? "mix" : "final mixer") + " (" + format +
                            "), " + std::to_string(rows) + " rows";
  const auto hidden = random_bf16(static_cast<std::size_t>(rows) * kWidth, 2.0, 0xF05U + rows);
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
  ignis::flash_next::HcWeights weights = dw;
  if (!with_inject) {
    weights.block_inject = nullptr;
  }
  float *inj = with_inject ? d_inj : nullptr;
  std::vector<uint16_t> x[2];
  std::vector<float> injections[2];
  int nodes[2] = {};
  for (int fused = 0; fused < 2; ++fused) {
    ignis::flash_next::fn_hc_set_decode_fused(fused == 1);
    cuda_ok(cudaMemset(d_x, 0xFF, static_cast<std::size_t>(rows) * kHidden * 2), "poison x");
    cuda_ok(cudaMemset(d_inj, 0xFF, static_cast<std::size_t>(rows) * kStreams * 4), "poison inj");
    check(ignis::flash_next::fn_hc_mix(g, weights, d_hidden, rows, d_x, inj, scratch, nullptr) == 0,
          label + ": runs: " + ignis::flash_next::fn_last_error());
    cuda_ok(cudaDeviceSynchronize(), "sync");
    x[fused].resize(static_cast<std::size_t>(rows) * kHidden);
    injections[fused].resize(static_cast<std::size_t>(rows) * kStreams);
    cuda_ok(cudaMemcpy(x[fused].data(), d_x, x[fused].size() * 2, cudaMemcpyDeviceToHost), "download x");
    cuda_ok(cudaMemcpy(injections[fused].data(), d_inj, injections[fused].size() * 4, cudaMemcpyDeviceToHost),
            "download inj");
    nodes[fused] = kernel_nodes(g, weights, d_hidden, rows, d_x, inj, scratch);
  }
  ignis::flash_next::fn_hc_set_decode_fused(true);
  check(x[1] == x[0], label + ": x is the three-launch route's, bit for bit");
  check(!with_inject || std::memcmp(injections[1].data(), injections[0].data(), injections[0].size() * 4) == 0,
        label + ": the injection weights are the three-launch route's, bit for bit");
  if (rows <= 3) {
    check(nodes[0] == 3 && nodes[1] == 2, label + ": 3 launches unfused, 2 fused (got " + std::to_string(nodes[0]) +
                                              ", " + std::to_string(nodes[1]) + ")");
  } else {
    check(nodes[0] == nodes[1] && nodes[0] >= 3,
          label + ": past three rows the switch changes no launch (got " + std::to_string(nodes[0]) + ", " +
              std::to_string(nodes[1]) + ")");
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
  const auto down = random_bf16(static_cast<std::size_t>(kRank) * kWidth, 0.02, 2);
  const auto up = random_bf16(static_cast<std::size_t>(kWidth) * kRank, 0.1, 3);
  w.down = bf16_values(down);
  w.up = bf16_values(up);
  w.inject = random_bf16(static_cast<std::size_t>(kStreams) * kWidth, 0.02, 4);
  ignis::flash_next::HcWeights dw;
  dw.hc_norm = upload(w.norm);
  dw.mix_down = {upload(down), kRank, kWidth, ignis::flash_next::WeightFormat::Bf16};
  dw.mix_up = {upload(up), kWidth, kRank, ignis::flash_next::WeightFormat::Bf16};
  dw.block_inject = upload(w.inject);

  // The artifact's formats: FP8 row-scale mix_down / mix_up.
  Weights w8 = w;
  ignis::flash_next::HcWeights dw8 = dw;
  dw8.mix_down = {upload(fp8_payload(kRank, kWidth, 0.03, 5, w8.down)), kRank, kWidth,
                  ignis::flash_next::WeightFormat::Fp8RowScale};
  dw8.mix_up = {upload(fp8_payload(kWidth, kRank, 0.15, 6, w8.up)), kWidth, kRank,
                ignis::flash_next::WeightFormat::Fp8RowScale};

  ninfer::DeviceArena scratch(64u << 20);
  for (int rows : {1, 3, 8, 9, 1027, 1100}) {
    mix_case(w, dw, "BF16", rows, true, scratch);
    mix_case(w8, dw8, "FP8", rows, true, scratch);
  }
  for (int rows : {1, 3, 9}) {
    mix_case(w, dw, "BF16", rows, false, scratch);
    mix_case(w8, dw8, "FP8", rows, false, scratch);
  }
  // A part re-converted to BF16 (linear.cu): one projection of each format.
  Weights w8_down = w8;
  w8_down.up = w.up;
  ignis::flash_next::HcWeights dw8_down = dw8;
  dw8_down.mix_up = dw.mix_up;
  Weights w8_up = w;
  w8_up.up = w8.up;
  ignis::flash_next::HcWeights dw8_up = dw;
  dw8_up.mix_up = dw8.mix_up;
  for (int rows : {1, 3, 9}) {
    mix_case(w8_down, dw8_down, "FP8 down, BF16 up", rows, true, scratch);
    mix_case(w8_up, dw8_up, "BF16 down, FP8 up", rows, true, scratch);
  }
  for (int rows : {1, 2, 3, 4, 8, 9}) {
    for (bool with_inject : {true, false}) {
      fused_case(dw, "BF16", rows, with_inject, scratch);
      fused_case(dw8, "FP8", rows, with_inject, scratch);
    }
    fused_case(dw8_down, "FP8 down, BF16 up", rows, true, scratch);
    fused_case(dw8_up, "BF16 down, FP8 up", rows, true, scratch);
  }

  // What the norm cannot hold, and projections whose shape is not the
  // geometry's, are refused by name, not computed wrong.
  {
    ignis::flash_next::Geometry g;
    g.streams = kStreams;
    g.hc_rank = kRank;
    g.rms_norm_eps = static_cast<float>(kEps);
    void *d_out = nullptr;
    float *d_inj = nullptr;
    cuda_ok(cudaMalloc(&d_out, static_cast<std::size_t>(kWidth) * 2), "cudaMalloc out");
    cuda_ok(cudaMalloc(&d_inj, kStreams * 4), "cudaMalloc inj");
    for (int hidden : {2564, 4104}) {
      g.hidden = hidden;
      check(ignis::flash_next::fn_hc_mix(g, dw, d_out, 1, d_out, d_inj, scratch, nullptr) != 0,
            "a stream width of " + std::to_string(hidden) + " is refused");
    }
    g.hidden = kHidden;
    ignis::flash_next::HcWeights misaligned = dw;
    misaligned.block_inject = static_cast<const char *>(dw.block_inject) + 2;
    check(ignis::flash_next::fn_hc_mix(g, misaligned, d_out, 1, d_out, d_inj, scratch, nullptr) != 0,
          "a block-inject weight off 16 bytes is refused");
    ignis::flash_next::HcWeights reshaped = dw;
    reshaped.mix_up.cols = kRank + 16;
    check(ignis::flash_next::fn_hc_mix(g, reshaped, d_out, 1, d_out, d_inj, scratch, nullptr) != 0,
          "a mix_up of another rank than the geometry's is refused");
    cuda_ok(cudaDeviceSynchronize(), "sync");
    cudaFree(d_out);
    cudaFree(d_inj);
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next hyper-connection test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next hyper-connection test: ok\n");
  return 0;
}
