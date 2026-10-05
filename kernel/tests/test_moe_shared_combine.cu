// Flash-Next's shared expert and MoE combine -- OURS (spec flash-next/02 Acceptance 4,
// GitHub #300).
//
//   shared    the FP8 SwiGLU expert at its real shapes (gate/up [640][2560], down [2560][640]) at
//             1, 3 and 300 tokens: its BF16 intermediate h against fp64 within the swiglu bound
//             (test_fp8_linear), its fp32 output against the fp64 down projection of that same
//             h within the fp32 accumulation bound, and end to end against fp64 with an
//             unrounded h within BF16's rounding of h (relative L2 <= 4e-3, ~2^-9 / sqrt(3)
//             expected).
//   combine   acc * 2^-32 + sigmoid(x . w_gate) * shared, rounded once to BF16, against fp64:
//             within BF16's half ulp (at most 2^-8 relative) plus the fp32 terms (the gate's dot product bound times
//             |shared|, one rounding each of the conversion and the fma). The accumulator reads
//             back as zero.
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

constexpr int H = IGNIS_MOE_HIDDEN;
constexpr int I = IGNIS_MOE_INTERMEDIATE;

void shared_arm(int tokens) {
  const Fp8Matrix g = make_fp8(5001, I, H, 0.003f), u = make_fp8(5002, I, H, 0.003f), d = make_fp8(5003, H, I, 0.004f);
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * H);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(79, i, 2.0f));
  DeviceBytes dg(g.payload.size()), du(u.payload.size()), dd(d.payload.size()), dx(x.size() * 2),
      dh(static_cast<std::size_t>(tokens) * I * 2), ds(static_cast<std::size_t>(tokens) * H * 4);
  upload(dg, g.payload);
  upload(du, u.payload);
  upload(dd, d.payload);
  upload(dx, x);
  MOE_RC(ignis_moe_shared_expert(dg.p, du.p, dd.p, dx.p, tokens, dh.p, ds.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto h = download<uint16_t>(dh.p, static_cast<std::size_t>(tokens) * I);
  const auto s = download<float>(ds.p, static_cast<std::size_t>(tokens) * H);
  int h_bad = 0, s_bad = 0;
  double worst_e2e = 0.0;
  const int step = tokens > 64 ? 23 : 1;
  for (int t = 0; t < tokens; t += step) {
    std::vector<double> xr(H), hk(I), h64(I);
    for (int k = 0; k < H; ++k) xr[k] = bf16_to_f32(x[static_cast<std::size_t>(t) * H + k]);
    for (int r = 0; r < I; ++r) {
      double gv, ga, uv, ua;
      fp8_row_f64(g, r, xr.data(), &gv, &ga);
      fp8_row_f64(u, r, xr.data(), &uv, &ua);
      h64[r] = silu(gv) * uv;
      hk[r] = bf16_to_f32(h[static_cast<std::size_t>(t) * I + r]);
      const double tol = 1.1 * std::fabs(uv) * fp8_sum_bound(H, ga) + std::fabs(silu(gv)) * fp8_sum_bound(H, ua) +
                         std::fabs(h64[r]) * (0x1p-8 + 0x1p-20) + 1e-30;
      if (std::fabs(hk[r] - h64[r]) > tol) ++h_bad;
    }
    double num = 0.0, den = 0.0;
    for (int r = 0; r < H; ++r) {
      double ref, abs_sum, ref64, abs64;
      fp8_row_f64(d, r, hk.data(), &ref, &abs_sum);
      fp8_row_f64(d, r, h64.data(), &ref64, &abs64);
      const double got = s[static_cast<std::size_t>(t) * H + r];
      if (std::fabs(got - ref) > fp8_sum_bound(I, abs_sum) + std::fabs(ref) * 0x1p-24 + 1e-30) ++s_bad;
      num += (got - ref64) * (got - ref64);
      den += ref64 * ref64;
    }
    worst_e2e = std::max(worst_e2e, std::sqrt(num / den));
  }
  std::printf("  shared expert x %3d tokens: h %d over bound, output %d over bound, end-to-end relative L2 %.2e\n",
              tokens, h_bad, s_bad, worst_e2e);
  check(h_bad == 0, "shared expert h within bound of fp64 (" + std::to_string(tokens) + " tokens)");
  check(s_bad == 0, "shared expert output within the fp32 bound of fp64 over its own h (" + std::to_string(tokens) + " tokens)");
  check(worst_e2e <= 4e-3, "shared expert end to end within BF16's rounding of h (" + std::to_string(tokens) + " tokens)");
}

void combine_arm(int tokens) {
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * H), wg(H);
  std::vector<float> shared(static_cast<std::size_t>(tokens) * H);
  std::vector<int64_t> acc(static_cast<std::size_t>(tokens) * H);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(80, i, 2.0f));
  for (int k = 0; k < H; ++k) wg[k] = f32_to_bf16(hash_uniform(81, k, 0.02f));
  for (std::size_t i = 0; i < shared.size(); ++i) shared[i] = hash_uniform(82, i, 0.5f);
  for (std::size_t i = 0; i < acc.size(); ++i) {
    acc[i] = static_cast<int64_t>(static_cast<int32_t>(hash_u32(83, i))) * 7;  // |value| < 7 * 2^31 * 2^-32 = 3.5
  }
  DeviceBytes dx(x.size() * 2), dwg(wg.size() * 2), ds(shared.size() * 4), da(acc.size() * 8), dout(x.size() * 2);
  upload(dx, x);
  upload(dwg, wg);
  upload(ds, shared);
  upload(da, acc);
  MOE_RC(ignis_moe_combine(da.as<int64_t>(), ds.as<float>(), dx.p, dwg.p, tokens, dout.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto out = download<uint16_t>(dout.p, x.size());
  const auto after = download<int64_t>(da.p, acc.size());
  bool zero = true;
  for (int64_t v : after) zero = zero && v == 0;
  check(zero, "combine zeroes the accumulator it read");
  int bad = 0;
  for (int t = 0; t < tokens; ++t) {
    double dot = 0.0, abs_dot = 0.0;
    for (int k = 0; k < H; ++k) {
      const double p = static_cast<double>(bf16_to_f32(x[static_cast<std::size_t>(t) * H + k])) * bf16_to_f32(wg[k]);
      dot += p;
      abs_dot += std::fabs(p);
    }
    const double gate = 1.0 / (1.0 + std::exp(-dot));
    const double gate_tol = 0.25 * std::ldexp(H / 256.0 + 16.0, -24) * abs_dot + 1e-7;  // sigmoid' <= 1/4, + expf
    for (int k = 0; k < H; ++k) {
      const std::size_t i = static_cast<std::size_t>(t) * H + k;
      const double routed = std::ldexp(static_cast<double>(acc[i]), -32);
      const double ref = routed + gate * shared[i];
      const double tol = std::fabs(ref) * 0x1p-8 + gate_tol * std::fabs(shared[i]) + (std::fabs(routed) + std::fabs(ref)) * 0x1p-23 + 1e-30;
      if (std::fabs(bf16_to_f32(out[i]) - ref) > tol) ++bad;
    }
  }
  std::printf("  combine x %4d tokens: %d outputs over bound\n", tokens, bad);
  check(bad == 0, "combine within bound of fp64 (" + std::to_string(tokens) + " tokens)");
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  MOE_RC(ignis_moe_prepare());
  std::printf("shared expert (FP8 SwiGLU, sigmoid gate) and combine\n");
  for (int tokens : {1, 3, 300}) shared_arm(tokens);
  for (int tokens : {1, 3, 4096}) combine_arm(tokens);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_shared_combine: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_shared_combine: OK\n");
  return 0;
}
