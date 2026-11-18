// The FP8 arms of the 27B's BF16 exception matrices (GitHub #341, kernel/src/fp8_exception.h) --
// OURS. The dispatchers' arms are plumbing over kernels test_fp8_linear and test_image_gemm hold to
// their bounds; this holds the plumbing:
//
//   qkgv   ops::attn_input_proj on a [14336][5120] weight converted with ignis_fp8_quantize_rows
//          into its four q | k | gate | v payloads. A16Only at 1, 8, 9 and 200 tokens: every
//          output's projection against the fp64 product of its own dequantized rows (so a segment
//          landing in the wrong output fails), within the FP8 linear's accumulation bound and
//          BF16's half ulp. AllowA8: at 8 tokens bit for bit the A16 route; past 8 bit for bit one
//          ignis_image_quantize_tokens and one ignis_image_gemm_w8a8 per segment, and within a
//          loose distance of the A16 route (the activation quantization's).
//   out    ops::linear_add on a [5120][6144] weight: A16Only bit for bit ignis_fp8_linear_add;
//          AllowA8 past 8 tokens bit for bit the quantization and ignis_image_gemm_w8a8_add.
//   space  each AllowA8 call runs in an arena of exactly the capacity the workspace query
//          reports; A16Only reports none.
//   refusals  AllowA4, a short payload, a wrong shape.
//   paired (GitHub #678) the form qkgv's GEMV calls take, as the measurement switch
//          IGNIS_FP8_GEMV_PAIRED names it at each call: unset or `1` the grouped GEMV's paired form,
//          `0` its one-row form, any other value refused by name; out never takes the paired form.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_exception.h"
#include "fp8_test_common.h"
#include "ignis_fp8_linear.h"
#include "ignis_image_gemm.h"
#include "moe_fixture.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_add.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace moe_test;
namespace fe = ignis::fp8_exception;
using ninfer::ops::LinearPolicy;

namespace {

constexpr int kHidden = 5120, kOutCols = 6144;

// A BF16 weight on the device, its FP8 conversion at `offsets` (one payload per row block), and
// every dequantized value on the host.
struct Converted {
  DeviceBytes bf16, fp8;
  std::vector<double> value;  // dequantized, row-major over all rows
  Converted(int rows, int cols, std::size_t fp8_bytes) : bf16(static_cast<std::size_t>(rows) * cols * 2), fp8(fp8_bytes) {}
};

std::unique_ptr<Converted> convert(uint32_t seed, const std::vector<uint32_t> &blocks, int cols,
                                   const std::vector<uint64_t> &offsets, uint64_t bytes) {
  int rows = 0;
  for (uint32_t b : blocks) rows += static_cast<int>(b);
  auto c = std::make_unique<Converted>(rows, cols, bytes);
  std::vector<uint16_t> w(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    const float mag = 0.004f * (1.0f + static_cast<float>(r % 7));
    for (int k = 0; k < cols; ++k) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + k;
      w[i] = f32_to_bf16(hash_uniform(seed, i, mag));
    }
  }
  upload(c->bf16, w);
  MOE_CUDA(cudaMemset(c->fp8.p, 0, bytes));
  int row0 = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    MOE_RC(ignis_fp8_quantize_rows(c->bf16.as<uint8_t>() + static_cast<std::size_t>(row0) * cols * 2, blocks[i], cols,
                                   c->fp8.as<uint8_t>() + offsets[i], nullptr));
    row0 += static_cast<int>(blocks[i]);
  }
  MOE_CUDA(cudaDeviceSynchronize());
  const auto payload = download<uint8_t>(c->fp8.p, bytes);
  c->value.resize(w.size());
  row0 = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const uint8_t *p = payload.data() + offsets[i];
    const std::size_t scale_at = (static_cast<std::size_t>(blocks[i]) * cols + 255) / 256 * 256;
    for (uint32_t r = 0; r < blocks[i]; ++r) {
      uint16_t s;
      std::memcpy(&s, p + scale_at + 2 * static_cast<std::size_t>(r), 2);
      const double scale = bf16_to_f32(s);
      for (int k = 0; k < cols; ++k) {
        c->value[static_cast<std::size_t>(row0 + r) * cols + k] = e4m3_to_f32(p[static_cast<std::size_t>(r) * cols + k]) * scale;
      }
    }
    row0 += static_cast<int>(blocks[i]);
  }
  return c;
}

ninfer::Weight fp8_weight(const void *data, int n, int k, uint64_t bytes) {
  ninfer::Weight w{};
  w.payload = data;
  w.payload_bytes = bytes;
  w.qtype = ninfer::QType::FP8_E4M3FN_ROW_BF16S;
  w.layout = ninfer::QuantLayout::RowScale;
  w.shape[0] = w.padded_shape[0] = n;
  w.shape[1] = w.padded_shape[1] = k;
  w.ndim = 2;
  w.qdata = data;
  w.n = n;
  w.k = k;
  return w;
}

std::vector<uint16_t> activations(uint32_t seed, int tokens, int cols) {
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(seed, i, 2.0f));
  return x;
}

std::vector<int> some(int count, int want) {
  std::vector<int> s;
  const int step = count > want ? count / want : 1;
  for (int i = 0; i < count; i += step) s.push_back(i);
  if (s.back() != count - 1) s.push_back(count - 1);
  return s;
}

struct Qkgv {
  DeviceBytes q, k, g, v;
  explicit Qkgv(int tokens)
      : q(static_cast<std::size_t>(tokens) * 6144 * 2), k(static_cast<std::size_t>(tokens) * 1024 * 2),
        g(static_cast<std::size_t>(tokens) * 6144 * 2), v(static_cast<std::size_t>(tokens) * 1024 * 2) {}
};

// Runs attn_input_proj under `policy` in an arena of exactly the reported capacity.
void run_qkgv(const ninfer::Weight &w, const DeviceBytes &dx, int tokens, Qkgv &out, LinearPolicy policy) {
  const std::size_t capacity = ninfer::ops::attn_input_proj_workspace_capacity_bytes(
      ninfer::QType::FP8_E4M3FN_ROW_BF16S, fe::kQkgvRows, kHidden, policy, 1, tokens);
  ninfer::DeviceArena arena(capacity > 0 ? capacity : 256);
  const ninfer::Tensor x(dx.p, ninfer::DType::BF16, {kHidden, tokens});
  ninfer::Tensor q(out.q.p, ninfer::DType::BF16, {6144, tokens}), g(out.g.p, ninfer::DType::BF16, {6144, tokens}),
      k(out.k.p, ninfer::DType::BF16, {1024, tokens}), v(out.v.p, ninfer::DType::BF16, {1024, tokens});
  ninfer::ops::attn_input_proj(x, w, q, g, k, v, policy, arena, nullptr);
  MOE_CUDA(cudaDeviceSynchronize());
}

void qkgv_arms() {
  const fe::QkgvLayout layout = fe::qkgv_layout();
  const std::vector<uint32_t> blocks(fe::kQkgvSegmentRows, fe::kQkgvSegmentRows + fe::kQkgvSegments);
  const std::vector<uint64_t> offsets(layout.at, layout.at + fe::kQkgvSegments);
  const auto c = convert(4400, blocks, kHidden, offsets, layout.at[fe::kQkgvSegments]);
  const ninfer::Weight w = fp8_weight(c->fp8.p, fe::kQkgvRows, kHidden, layout.at[fe::kQkgvSegments]);
  // Each output's rows within the parent: q 0, k 6144, gate 7168, v 13312.
  const int first_row[4] = {0, 6144, 7168, 13312};
  const int out_rows[4] = {6144, 1024, 6144, 1024};

  for (int tokens : {1, 8, 9, 200}) {
    const std::vector<uint16_t> x = activations(4500 + tokens, tokens, kHidden);
    DeviceBytes dx(x.size() * 2);
    upload(dx, x);
    Qkgv a16(tokens);
    run_qkgv(w, dx, tokens, a16, LinearPolicy::A16Only);
    const DeviceBytes *outs[4] = {&a16.q, &a16.k, &a16.g, &a16.v};
    int bad = 0;
    double worst = 0.0;
    for (int o = 0; o < 4; ++o) {
      const auto y = download<uint16_t>(outs[o]->p, static_cast<std::size_t>(tokens) * out_rows[o]);
      for (int t : some(tokens, 6)) {
        for (int r : some(out_rows[o], 64)) {
          const double *wr = &c->value[static_cast<std::size_t>(first_row[o] + r) * kHidden];
          double ref = 0.0, abs_sum = 0.0;
          for (int k = 0; k < kHidden; ++k) {
            const double p = wr[k] * bf16_to_f32(x[static_cast<std::size_t>(t) * kHidden + k]);
            ref += p;
            abs_sum += std::fabs(p);
          }
          const double tol = fp8_sum_bound(kHidden, abs_sum) + std::fabs(ref) * 0x1p-8 + 1e-30;
          const double err = std::fabs(bf16_to_f32(y[static_cast<std::size_t>(t) * out_rows[o] + r]) - ref);
          worst = std::max(worst, err / tol);
          if (!(err <= tol)) ++bad;
        }
      }
    }
    std::printf("  qkgv A16 x %4d tokens: q | k | gate | v against fp64 of their own rows, worst %.2f of bound, %d over\n",
                tokens, worst, bad);
    check(bad == 0, "qkgv A16 x " + std::to_string(tokens) + ": each output is its own rows' product");
  }

  for (int tokens : {8, 9, 300, 2048}) {
    const std::vector<uint16_t> x = activations(4600 + tokens, tokens, kHidden);
    DeviceBytes dx(x.size() * 2);
    upload(dx, x);
    Qkgv a8(tokens), a16(tokens), direct(tokens);
    run_qkgv(w, dx, tokens, a8, LinearPolicy::AllowA8);
    run_qkgv(w, dx, tokens, a16, LinearPolicy::A16Only);
    const DeviceBytes *got[4] = {&a8.q, &a8.k, &a8.g, &a8.v};
    const DeviceBytes *exact[4] = {&a16.q, &a16.k, &a16.g, &a16.v};
    bool same = true;
    if (tokens <= fe::kGemvTokens) {
      for (int o = 0; o < 4; ++o) {
        const std::size_t n = static_cast<std::size_t>(tokens) * out_rows[o];
        same = same && download<uint16_t>(got[o]->p, n) == download<uint16_t>(exact[o]->p, n);
      }
      std::printf("  qkgv A8 x %4d tokens: %s the A16 route\n", tokens, same ? "bit-identical to" : "DIFFERS from");
      check(same, "qkgv A8 x " + std::to_string(tokens) + " stays on the A16 route");
      continue;
    }
    DeviceBytes codes(static_cast<std::size_t>(tokens) * kHidden), scales(static_cast<std::size_t>(tokens) * 4);
    MOE_RC(ignis_image_quantize_tokens(dx.p, tokens, kHidden, codes.p, scales.as<float>(), nullptr));
    const DeviceBytes *want[4] = {&direct.q, &direct.k, &direct.g, &direct.v};
    for (int o = 0; o < 4; ++o) {
      MOE_RC(ignis_image_gemm_w8a8(c->fp8.as<uint8_t>() + layout.at[o], out_rows[o], kHidden, codes.p, scales.as<float>(),
                                   tokens, want[o]->p, 0, nullptr));
    }
    MOE_CUDA(cudaDeviceSynchronize());
    double num = 0.0, den = 0.0;
    for (int o = 0; o < 4; ++o) {
      const std::size_t n = static_cast<std::size_t>(tokens) * out_rows[o];
      const auto y = download<uint16_t>(got[o]->p, n), d = download<uint16_t>(want[o]->p, n),
                 e = download<uint16_t>(exact[o]->p, n);
      same = same && y == d;
      for (std::size_t i = 0; i < n; i += 7) {
        const double a = bf16_to_f32(y[i]), b = bf16_to_f32(e[i]);
        num += (a - b) * (a - b);
        den += b * b;
      }
    }
    const double rel = std::sqrt(num / den);
    std::printf("  qkgv A8 x %4d tokens: %s the direct W8A8 calls; relative L2 to the A16 route %.4f\n", tokens,
                same ? "bit-identical to" : "DIFFERS from", rel);
    check(same, "qkgv A8 x " + std::to_string(tokens) + " equals the quantization and one GEMM per segment");
    check(rel < 0.05, "qkgv A8 x " + std::to_string(tokens) + " is within the activation quantization of A16");
  }

  // Refusals.
  const std::vector<uint16_t> x = activations(4700, 9, kHidden);
  DeviceBytes dx(x.size() * 2);
  upload(dx, x);
  Qkgv out(9);
  bool refused = false;
  try {
    run_qkgv(w, dx, 9, out, LinearPolicy::AllowA4);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "qkgv: AllowA4 refused");
  refused = false;
  try {
    run_qkgv(fp8_weight(c->fp8.p, fe::kQkgvRows, kHidden, layout.at[fe::kQkgvSegments] - 256), dx, 9, out,
             LinearPolicy::A16Only);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "qkgv: a payload short of the four segments refused");
}

void out_arms() {
  const uint64_t bytes = ignis_fp8_payload_bytes(kHidden, kOutCols);
  const auto c = convert(4800, {static_cast<uint32_t>(kHidden)}, kOutCols, {0}, bytes);
  const ninfer::Weight w = fp8_weight(c->fp8.p, kHidden, kOutCols, bytes);
  for (int tokens : {1, 9, 300}) {
    for (LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
      const std::vector<uint16_t> x = activations(4900 + tokens, tokens, kOutCols);
      std::vector<uint16_t> res(static_cast<std::size_t>(tokens) * kHidden);
      for (std::size_t i = 0; i < res.size(); ++i) res[i] = f32_to_bf16(hash_uniform(4950, i, 3.0f));
      DeviceBytes dx(x.size() * 2), dr(res.size() * 2), want(res.size() * 2);
      upload(dx, x);
      upload(dr, res);
      upload(want, res);
      const std::size_t capacity = ninfer::ops::linear_add_workspace_capacity_bytes(
          ninfer::QType::FP8_E4M3FN_ROW_BF16S, kHidden, kOutCols, policy, 1, tokens);
      ninfer::DeviceArena arena(capacity > 0 ? capacity : 256);
      const ninfer::Tensor tx(dx.p, ninfer::DType::BF16, {kOutCols, tokens});
      ninfer::Tensor tr(dr.p, ninfer::DType::BF16, {kHidden, tokens});
      ninfer::ops::linear_add(tx, w, tr, policy, arena, nullptr);
      const bool a8 = fe::quantizes(policy, tokens);
      if (a8) {
        DeviceBytes codes(static_cast<std::size_t>(tokens) * kOutCols), scales(static_cast<std::size_t>(tokens) * 4);
        MOE_RC(ignis_image_quantize_tokens(dx.p, tokens, kOutCols, codes.p, scales.as<float>(), nullptr));
        MOE_RC(ignis_image_gemm_w8a8_add(c->fp8.p, kHidden, kOutCols, codes.p, scales.as<float>(), tokens, want.p, nullptr));
      } else {
        MOE_RC(ignis_fp8_linear_add(c->fp8.p, kHidden, kOutCols, dx.p, tokens, want.p, nullptr));
      }
      MOE_CUDA(cudaDeviceSynchronize());
      const bool same = download<uint16_t>(dr.p, res.size()) == download<uint16_t>(want.p, res.size());
      const char *name = policy == LinearPolicy::A16Only ? "A16" : "A8";
      std::printf("  out %s x %4d tokens (%s route, workspace %zu B): %s\n", name, tokens, a8 ? "W8A8" : "W8A16",
                  capacity, same ? "bit-identical to the direct call" : "DIFFERS from the direct call");
      check(same, std::string("out ") + name + " x " + std::to_string(tokens) + " equals the direct call");
      check((capacity == 0) == !a8, std::string("out ") + name + " x " + std::to_string(tokens) +
                                        ": workspace only on the quantizing route");
    }
  }
  bool refused = false;
  try {
    DeviceBytes dx(9 * kOutCols * 2), dr(9 * kHidden * 2);
    ninfer::DeviceArena arena(1u << 20);
    const ninfer::Tensor tx(dx.p, ninfer::DType::BF16, {kOutCols, 9});
    ninfer::Tensor tr(dr.p, ninfer::DType::BF16, {kHidden, 9});
    ninfer::ops::linear_add(tx, fp8_weight(c->fp8.p, kHidden, kOutCols, bytes - 2), tr, LinearPolicy::A16Only, arena,
                            nullptr);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "out: a short payload refused");
  refused = false;
  try {
    // The output projection's arm takes [5120][6144] only: the same bytes as [6144][5120] are refused.
    DeviceBytes dx(9 * kHidden * 2), dr(9 * kOutCols * 2);
    ninfer::DeviceArena arena(1u << 20);
    const ninfer::Tensor tx(dx.p, ninfer::DType::BF16, {kHidden, 9});
    ninfer::Tensor tr(dr.p, ninfer::DType::BF16, {kOutCols, 9});
    ninfer::ops::linear_add(tx, fp8_weight(c->fp8.p, kOutCols, kHidden, bytes), tr, LinearPolicy::A16Only, arena,
                            nullptr);
  } catch (const std::exception &) {
    refused = true;
  }
  check(refused, "out: a wrong shape refused");
}

// Sets (or, with null, removes) a variable in the operating system's environment, as a host
// process's own runtime does (Rust's std::env::set_var).
void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

// GitHub #678: which form qkgv's GEMV calls run, by the paired form's launch counter, on zeroed
// weights (test_fp8_linear holds the two forms' bytes to each other): one paired launch a call at 1
// and 8 tokens under either policy unless IGNIS_FP8_GEMV_PAIRED=0, none past 8 tokens, none for out.
void paired_arms() {
  constexpr const char *kSwitch = "IGNIS_FP8_GEMV_PAIRED";
  const fe::QkgvLayout layout = fe::qkgv_layout();
  DeviceBytes qkgv(layout.at[fe::kQkgvSegments]), out(ignis_fp8_payload_bytes(kHidden, kOutCols));
  MOE_CUDA(cudaMemset(qkgv.p, 0, qkgv.bytes));
  MOE_CUDA(cudaMemset(out.p, 0, out.bytes));
  const ninfer::Weight wq = fp8_weight(qkgv.p, fe::kQkgvRows, kHidden, qkgv.bytes);
  const ninfer::Weight wo = fp8_weight(out.p, kHidden, kOutCols, out.bytes);
  const std::vector<uint16_t> x = activations(4990, 9, kOutCols);
  DeviceBytes dx(x.size() * 2), dr(9 * kHidden * 2);
  upload(dx, x);
  const char *const values[3] = {nullptr, "1", "0"};
  for (const char *value : values) {
    set_environment(kSwitch, value);
    const bool paired = value == nullptr || std::strcmp(value, "1") == 0;
    const std::string named = std::string(kSwitch) + (value == nullptr ? " unset" : std::string("=") + value);
    for (int tokens : {1, 8, 9}) {
      for (LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
        Qkgv o(tokens);
        const uint64_t before = ignis_fp8_linear_paired_launches();
        run_qkgv(wq, dx, tokens, o, policy);
        const uint64_t launches = ignis_fp8_linear_paired_launches() - before;
        const uint64_t want = paired && tokens <= fe::kGemvTokens ? 1 : 0;
        std::printf("  paired %s: qkgv %s x %d tokens, %llu paired launch(es)\n", named.c_str(),
                    policy == LinearPolicy::A16Only ? "A16" : "A8", tokens, static_cast<unsigned long long>(launches));
        check(launches == want, named + ": qkgv x " + std::to_string(tokens) + " runs the " +
                                    (want == 1 ? "paired" : "one-row or tensor-core") + " form");
      }
    }
    const uint64_t before = ignis_fp8_linear_paired_launches();
    const std::size_t capacity = ninfer::ops::linear_add_workspace_capacity_bytes(
        ninfer::QType::FP8_E4M3FN_ROW_BF16S, kHidden, kOutCols, LinearPolicy::A16Only, 1, 1);
    ninfer::DeviceArena arena(capacity > 0 ? capacity : 256);
    const ninfer::Tensor tx(dx.p, ninfer::DType::BF16, {kOutCols, 1});
    ninfer::Tensor tr(dr.p, ninfer::DType::BF16, {kHidden, 1});
    ninfer::ops::linear_add(tx, wo, tr, LinearPolicy::A16Only, arena, nullptr);
    MOE_CUDA(cudaDeviceSynchronize());
    check(ignis_fp8_linear_paired_launches() == before, named + ": out x 1 keeps the one-row form");
  }
  set_environment(kSwitch, "2");
  bool refused = false;
  try {
    Qkgv o(1);
    run_qkgv(wq, dx, 1, o, LinearPolicy::A16Only);
  } catch (const std::invalid_argument &e) {
    refused = std::string(e.what()).find(kSwitch) != std::string::npos;
  }
  check(refused, std::string(kSwitch) + "=2 refused by name");
  set_environment(kSwitch, nullptr);
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  MOE_RC(ignis_fp8_linear_prepare());
  if (ignis_image_gemm_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_image_gemm_prepare: %s\n", ignis_image_gemm_last_error());
    return 1;
  }
  std::printf("FP8 arms of the 27B's BF16 exception matrices (GitHub #341)\n");
  qkgv_arms();
  out_arms();
  paired_arms();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_fp8_exception_arms: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_fp8_exception_arms: OK\n");
  return 0;
}
