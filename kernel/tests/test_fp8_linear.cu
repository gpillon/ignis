// The FP8 row-scale linear (spec flash-next/04's op, written for spec 02's shared expert) at
// Flash-Next's shapes -- OURS (GitHub #300).
//
//   linear   [640][2560] and [2560][640] (the shared expert's projections) and [48][2560] (the
//            GDN gating rows, narrower than a tile) at 1, 3, 8 tokens (GEMV route) and 9..2048
//            (tensor-core route), fp32 and BF16 outputs, against fp64.
//   swiglu   the shared expert's gate/up pair, silu(g) * u in BF16, against fp64.
//
// Tolerance: E4M3 x BF16 products are exact in fp32, so the error is fp32 accumulation, bounded
// by (cols / 8 + 64) u sum|terms| (fp8_test_common.h), plus the output's own rounding: one fp32
// rounding of the scaled sum, or BF16's half ulp (at most 2^-8 relative). Every output is checked; for
// the widest calls a fixed sample of tokens. Each call also runs twice and must agree bit for
// bit.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_test_common.h"
#include "ignis_moe.h"
#include "moe_fixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

std::vector<int> sample_tokens(int tokens) {
  std::vector<int> s;
  const int step = tokens > 256 ? tokens / 97 : 1;
  for (int t = 0; t < tokens; t += step) s.push_back(t);
  if (s.back() != tokens - 1) s.push_back(tokens - 1);
  return s;
}

std::vector<double> token_row(const std::vector<uint16_t> &x, int t, int cols) {
  std::vector<double> v(cols);
  for (int c = 0; c < cols; ++c) v[c] = bf16_to_f32(x[static_cast<std::size_t>(t) * cols + c]);
  return v;
}

void linear_arm(int rows, int cols, int tokens, bool f32_out) {
  const Fp8Matrix m = make_fp8(static_cast<uint32_t>(1000 + rows + cols), rows, cols, 0.004f);
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(77, i, 2.0f));
  DeviceBytes dw(m.payload.size()), dx(x.size() * 2), dy(static_cast<std::size_t>(tokens) * rows * 4);
  upload(dw, m.payload);
  upload(dx, x);
  MOE_RC(ignis_fp8_linear(dw.p, rows, cols, dx.p, tokens, dy.p, f32_out ? 1 : 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::size_t n = static_cast<std::size_t>(tokens) * rows;
  std::vector<float> y(n);
  if (f32_out) {
    y = download<float>(dy.p, n);
  } else {
    const auto b = download<uint16_t>(dy.p, n);
    for (std::size_t i = 0; i < n; ++i) y[i] = bf16_to_f32(b[i]);
  }
  const auto first = download<uint8_t>(dy.p, n * (f32_out ? 4 : 2));
  MOE_RC(ignis_fp8_linear(dw.p, rows, cols, dx.p, tokens, dy.p, f32_out ? 1 : 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<uint8_t>(dy.p, n * (f32_out ? 4 : 2)) == first, "fp8 linear: a second run agrees bit for bit");

  int bad = 0;
  double worst = 0.0;
  for (int t : sample_tokens(tokens)) {
    const std::vector<double> xr = token_row(x, t, cols);
    for (int r = 0; r < rows; ++r) {
      double ref = 0.0, abs_sum = 0.0;
      fp8_row_f64(m, r, xr.data(), &ref, &abs_sum);
      const double got = y[static_cast<std::size_t>(t) * rows + r];
      const double tol = fp8_sum_bound(cols, abs_sum) + std::fabs(ref) * (f32_out ? 0x1p-24 : 0x1p-8) + 1e-30;
      const double err = std::fabs(got - ref);
      worst = std::max(worst, err / tol);
      if (err > tol) ++bad;
    }
  }
  std::printf("  linear [%5d][%5d] x %4d tokens, %s out: worst error %.2f of bound, %d over\n", rows, cols, tokens,
              f32_out ? "fp32" : "bf16", worst, bad);
  check(bad == 0, "fp8 linear [" + std::to_string(rows) + "][" + std::to_string(cols) + "] x " + std::to_string(tokens) +
                      " within the fp32 accumulation bound of fp64");
}

void swiglu_arm(int tokens) {
  const int rows = IGNIS_MOE_INTERMEDIATE, cols = IGNIS_MOE_HIDDEN;
  const Fp8Matrix g = make_fp8(3001, rows, cols, 0.003f), u = make_fp8(3002, rows, cols, 0.003f);
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(78, i, 2.0f));
  DeviceBytes dg(g.payload.size()), du(u.payload.size()), dx(x.size() * 2), dh(static_cast<std::size_t>(tokens) * rows * 2);
  upload(dg, g.payload);
  upload(du, u.payload);
  upload(dx, x);
  MOE_RC(ignis_fp8_linear_swiglu(dg.p, du.p, rows, cols, dx.p, tokens, dh.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto h = download<uint16_t>(dh.p, static_cast<std::size_t>(tokens) * rows);
  MOE_RC(ignis_fp8_linear_swiglu(dg.p, du.p, rows, cols, dx.p, tokens, dh.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<uint16_t>(dh.p, h.size()) == h, "fp8 swiglu: a second run agrees bit for bit");
  int bad = 0;
  double worst = 0.0;
  for (int t : sample_tokens(tokens)) {
    const std::vector<double> xr = token_row(x, t, cols);
    for (int r = 0; r < rows; ++r) {
      double gv, ga, uv, ua;
      fp8_row_f64(g, r, xr.data(), &gv, &ga);
      fp8_row_f64(u, r, xr.data(), &uv, &ua);
      const double ref = silu(gv) * uv;
      // Propagated accumulation error (silu' <= 1.1), the fp32 epilogue, and BF16's half ulp.
      const double tol = 1.1 * std::fabs(uv) * fp8_sum_bound(cols, ga) + std::fabs(silu(gv)) * fp8_sum_bound(cols, ua) +
                         std::fabs(ref) * (0x1p-8 + 0x1p-20) + 1e-30;
      const double err = std::fabs(bf16_to_f32(h[static_cast<std::size_t>(t) * rows + r]) - ref);
      worst = std::max(worst, err / tol);
      if (err > tol) ++bad;
    }
  }
  std::printf("  swiglu [%d][%d] x %4d tokens: worst error %.2f of bound, %d over\n", rows, cols, tokens, worst, bad);
  check(bad == 0, "fp8 swiglu x " + std::to_string(tokens) + " within bound of fp64");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::printf("FP8 row-scale linear (E4M3 + BF16 row scale), BF16 activations, fp32 accumulation\n");
  for (int tokens : {1, 3, 8, 9, 200, 2048}) linear_arm(640, 2560, tokens, tokens % 2 == 1);
  for (int tokens : {1, 8, 129, 1000}) linear_arm(2560, 640, tokens, true);
  for (int tokens : {2, 300}) linear_arm(48, 2560, tokens, false);
  for (int tokens : {1, 3, 8, 64, 1024}) swiglu_arm(tokens);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_fp8_linear: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_fp8_linear: OK\n");
  return 0;
}
