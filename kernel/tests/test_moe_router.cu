// Flash-Next's MoE router at its real geometry (512 experts x 2560) -- OURS (spec flash-next/02
// Acceptance 1, GitHub #300).
//
//   fp64        the fp32 logits against an fp64 dot product, within the bound recursive fp32
//               summation admits for the kernel's chain (stated below); the top-10 set of the
//               BF16-rounded logits against the set of the fp64 logits rounded the same way,
//               exact except at near-ties: a differing expert whose fp64 logit lies within that
//               bound of a BF16 rounding midpoint, where fp32 accumulation can legitimately
//               round either way. Near-ties are counted and printed.
//   selection   exact, no tolerance: from the kernel's own fp32 logits, the host rounds to BF16
//               and sorts by (value descending, expert ascending); ids must match in order.
//               Weights against an fp64 softmax over the same ten values, within BF16's
//               half-ulp plus fp32 slack.
//   checkpoint  against transformers' Qwen4ExpTextTopKRouter recorded on the same inputs
//               (fixtures/flash_next/router_ref.bin): BF16 logits differ by at most one ulp
//               where cuBLAS's fp32 accumulation rounds differently, sets differ only where
//               such a logit is involved, and the BF16 weights agree.
//   ties        every logit equal (ids 0..9, weights 0.1 in BF16), and twelve experts tied at
//               the top (the ten lowest ids win, ascending).
//   determinism two runs agree bit for bit, and a token's outputs are the same whether it is
//               routed alone, in three, or in the 192-token call.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU or fixture fails.

#include "ignis_moe.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr int H = IGNIS_MOE_HIDDEN;
constexpr int E = IGNIS_MOE_EXPERTS;
constexpr int K = IGNIS_MOE_TOP_K;
// Recursive fp32 summation over the kernel's chain (80 products per lane, then 5 butterfly
// levels) errs by at most ~85 u sum|x w|, u = 2^-24: 5.1e-6. Stated with margin.
constexpr double kLogitRel = 6e-6;

struct Out {
  std::vector<int32_t> ids;
  std::vector<float> weights;
  std::vector<float> logits;
};

Out run(const void *d_x, int tokens, const void *d_w) {
  DeviceBytes ids(static_cast<std::size_t>(tokens) * K * 4), w(static_cast<std::size_t>(tokens) * K * 4),
      lg(static_cast<std::size_t>(tokens) * E * 4);
  MOE_RC(ignis_moe_router(d_x, tokens, d_w, ids.as<int32_t>(), w.as<float>(), lg.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  return Out{download<int32_t>(ids.p, static_cast<std::size_t>(tokens) * K),
             download<float>(w.p, static_cast<std::size_t>(tokens) * K),
             download<float>(lg.p, static_cast<std::size_t>(tokens) * E)};
}

// Host selection on BF16-rounded fp32 logits: (value descending, id ascending).
std::vector<int> select_top(const float *logits) {
  std::vector<int> order(E);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return bf16_to_f32(f32_to_bf16(logits[a])) > bf16_to_f32(f32_to_bf16(logits[b]));
  });
  order.resize(K);
  return order;
}

// Whether a BF16 rounding midpoint lies within `tol` of `v`: there fp32 accumulation may round
// either way.
bool ambiguous_bf16(double v, double tol) {
  const uint16_t lo = f32_to_bf16(static_cast<float>(v - tol));
  const uint16_t hi = f32_to_bf16(static_cast<float>(v + tol));
  return lo != hi;
}

void fp64_and_selection_arms(const std::vector<uint16_t> &x, const std::vector<uint16_t> &w, int T,
                             const Out &o, const std::vector<uint16_t> &torch_logits,
                             const std::vector<int32_t> &torch_ids, const std::vector<uint16_t> &torch_w) {
  int logit_bad = 0, near_ties = 0, set_bad = 0, order_bad = 0, weight_bad = 0;
  int torch_logit_diff = 0, torch_logit_far = 0, torch_set_diff = 0, torch_set_unexplained = 0;
  int weights_equal_torch = 0, weights_compared = 0;
  double worst_logit = 0.0;
  for (int t = 0; t < T; ++t) {
    std::vector<double> l64(E), tol(E);
    for (int e = 0; e < E; ++e) {
      double s = 0.0, a = 0.0;
      for (int k = 0; k < H; ++k) {
        const double p = static_cast<double>(bf16_to_f32(x[static_cast<std::size_t>(t) * H + k])) *
                         bf16_to_f32(w[static_cast<std::size_t>(e) * H + k]);
        s += p;
        a += std::fabs(p);
      }
      l64[e] = s;
      tol[e] = kLogitRel * a;
      const double err = std::fabs(o.logits[static_cast<std::size_t>(t) * E + e] - s);
      worst_logit = std::max(worst_logit, a > 0 ? err / a : 0.0);
      if (err > tol[e]) ++logit_bad;
    }
    // fp64 reference set.
    std::vector<float> l64f(E);
    for (int e = 0; e < E; ++e) l64f[e] = static_cast<float>(l64[e]);
    const std::vector<int> ref = select_top(l64f.data());
    const int32_t *got = &o.ids[static_cast<std::size_t>(t) * K];
    std::set<int> a(ref.begin(), ref.end()), b(got, got + K), diff;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(diff, diff.begin()));
    if (!diff.empty()) {
      bool explained = false;
      for (int e : diff) explained = explained || ambiguous_bf16(l64[e], tol[e]);
      if (explained) {
        ++near_ties;
      } else {
        ++set_bad;
      }
    }
    // Exact selection from the kernel's own logits.
    const std::vector<int> own = select_top(&o.logits[static_cast<std::size_t>(t) * E]);
    if (!std::equal(own.begin(), own.end(), got)) ++order_bad;
    // Weights against fp64 softmax over the ten BF16 values the kernel chose.
    double m = -INFINITY, sum = 0.0;
    double v[K];
    for (int r = 0; r < K; ++r) {
      v[r] = bf16_to_f32(f32_to_bf16(o.logits[static_cast<std::size_t>(t) * E + got[r]]));
      m = std::max(m, v[r]);
    }
    for (int r = 0; r < K; ++r) sum += std::exp(v[r] - m);
    for (int r = 0; r < K; ++r) {
      const double w64 = std::exp(v[r] - m) / sum;
      const double wk = o.weights[static_cast<std::size_t>(t) * K + r];
      if (std::fabs(wk - w64) > 0x1p-8 * w64 + 1e-7) ++weight_bad;
    }
    // The checkpoint's router on the same inputs.
    if (!torch_logits.empty()) {
      std::set<int> logit_differs;
      for (int e = 0; e < E; ++e) {
        const uint16_t ours = f32_to_bf16(o.logits[static_cast<std::size_t>(t) * E + e]);
        const uint16_t theirs = torch_logits[static_cast<std::size_t>(t) * E + e];
        if (ours != theirs) {
          ++torch_logit_diff;
          logit_differs.insert(e);
          // The checkpoint's value must be a BF16 rounding of something within the fp32
          // accumulation bound of the exact logit, as ours is; near zero that is many ulps.
          const double mag = std::fabs(l64[e]) + tol[e];
          const double ulp = mag > 0 ? std::ldexp(1.0, std::ilogb(mag) - 7) : 0.0;
          if (std::fabs(bf16_to_f32(theirs) - l64[e]) > tol[e] + ulp) ++torch_logit_far;
        }
      }
      std::set<int> c(&torch_ids[static_cast<std::size_t>(t) * K], &torch_ids[static_cast<std::size_t>(t) * K] + K), d;
      std::set_symmetric_difference(c.begin(), c.end(), b.begin(), b.end(), std::inserter(d, d.begin()));
      if (!d.empty()) {
        ++torch_set_diff;
        bool explained = false;
        for (int e : d) explained = explained || logit_differs.count(e) != 0;
        if (!explained) ++torch_set_unexplained;
      } else {
        // Same set: compare weights expert by expert.
        for (int r = 0; r < K; ++r) {
          for (int q = 0; q < K; ++q) {
            if (torch_ids[static_cast<std::size_t>(t) * K + q] == got[r]) {
              ++weights_compared;
              if (f32_to_bf16(o.weights[static_cast<std::size_t>(t) * K + r]) == torch_w[static_cast<std::size_t>(t) * K + q]) {
                ++weights_equal_torch;
              }
            }
          }
        }
      }
    }
  }
  std::printf("  fp64: worst |l32 - l64| / sum|xw| = %.3e (bound %.1e), %d logits over it\n", worst_logit, kLogitRel, logit_bad);
  std::printf("  fp64: top-10 sets: %d near-ties (counted), %d unexplained differences\n", near_ties, set_bad);
  std::printf("  selection: %d tokens out of order, %d weights outside fp64 + bf16 bound\n", order_bad, weight_bad);
  check(logit_bad == 0, "fp32 logits within the stated bound of fp64");
  check(set_bad == 0, "top-10 set equals fp64's except at near-ties");
  check(order_bad == 0, "ids are the (bf16 value desc, id asc) top-10 of the kernel's own logits");
  check(weight_bad == 0, "weights within bf16 half-ulp + fp32 of an fp64 softmax over the ten");
  if (!torch_logits.empty()) {
    std::printf("  checkpoint: %d of %d bf16 logits differ (%d outside the fp32 bound + 1 ulp of fp64); %d token sets differ "
                "(%d unexplained by a differing logit); weights equal %d / %d\n",
                torch_logit_diff, T * E, torch_logit_far, torch_set_diff, torch_set_unexplained,
                weights_equal_torch, weights_compared);
    check(torch_logit_far == 0, "the checkpoint router's bf16 logits are roundings within the fp32 bound of fp64");
    check(torch_logit_diff * 1000 <= T * E, "bf16 logits equal the checkpoint router's for >= 99.9%");
    check(torch_set_unexplained == 0, "every set difference with the checkpoint router involves a differing logit");
    check(weights_equal_torch * 100 >= weights_compared * 99, "bf16 weights equal the checkpoint router's for >= 99%");
  }
}

void ties_arm() {
  const int T = 2;
  std::vector<uint16_t> x(static_cast<std::size_t>(T) * H), w(static_cast<std::size_t>(E) * H);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(11, i, 1.0f));
  DeviceBytes dx(x.size() * 2), dw(w.size() * 2);
  upload(dx, x);
  // Every row equal: all 512 logits tie.
  for (int e = 0; e < E; ++e) {
    for (int k = 0; k < H; ++k) w[static_cast<std::size_t>(e) * H + k] = f32_to_bf16(hash_uniform(12, k, 0.03f));
  }
  upload(dw, w);
  Out o = run(dx.p, T, dw.p);
  for (int t = 0; t < T; ++t) {
    for (int r = 0; r < K; ++r) {
      check(o.ids[t * K + r] == r, "all-equal logits: ids are 0..9 ascending");
      check(f32_to_bf16(o.weights[t * K + r]) == f32_to_bf16(0.1f), "all-equal logits: weights are bf16(0.1)");
    }
  }
  // Twelve experts (500..511) tied far above the rest: the ten lowest of them win, ascending.
  for (int e = 0; e < E; ++e) {
    for (int k = 0; k < H; ++k) {
      const float base = e >= 500 ? (bf16_to_f32(x[k]) > 0 ? 0.05f : -0.05f) : hash_uniform(13 + e, k, 0.01f);
      w[static_cast<std::size_t>(e) * H + k] = f32_to_bf16(base);
    }
  }
  upload(dw, w);
  o = run(dx.p, 1, dw.p);
  for (int r = 0; r < K; ++r) check(o.ids[r] == 500 + r, "twelve tied at the top: ids 500..509 ascending");
  std::printf("  ties: all-equal -> %d..%d, twelve tied -> %d..%d\n", o.ids[0] - 500, o.ids[K - 1] - 500, o.ids[0], o.ids[K - 1]);
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  const auto fx = read_fixture(std::string(IGNIS_FLASH_NEXT_FIXTURE_DIR) + "/router_ref.bin");
  const auto geo = need(fx, "geometry").as<int32_t>();
  const auto amp = need(fx, "amplitudes").as<float>();
  const int T = geo[0];
  check(geo[1] == H && geo[2] == E && geo[3] == K, "fixture geometry is 512 x 2560, top-10");
  std::vector<uint16_t> x(static_cast<std::size_t>(T) * H), w(static_cast<std::size_t>(E) * H);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(static_cast<uint32_t>(geo[4]), i, amp[0]));
  for (std::size_t i = 0; i < w.size(); ++i) w[i] = f32_to_bf16(hash_uniform(static_cast<uint32_t>(geo[5]), i, amp[1]));
  DeviceBytes dx(x.size() * 2), dw(w.size() * 2);
  upload(dx, x);
  upload(dw, w);

  std::printf("router at 512 x 2560, %d tokens\n", T);
  const Out o = run(dx.p, T, dw.p);
  fp64_and_selection_arms(x, w, T, o, need(fx, "logits").as<uint16_t>(), need(fx, "ids").as<int32_t>(),
                          need(fx, "weights").as<uint16_t>());

  // Determinism: a second run, and tokens routed alone and in three.
  const Out again = run(dx.p, T, dw.p);
  check(again.ids == o.ids && again.weights == o.weights && again.logits == o.logits, "two runs agree bit for bit");
  for (int count : {1, 3}) {
    const int first = 5;
    const Out part = run(static_cast<const uint16_t *>(dx.p) + static_cast<std::size_t>(first) * H, count, dw.p);
    bool same = true;
    for (int t = 0; t < count; ++t) {
      for (int r = 0; r < K; ++r) {
        same = same && part.ids[t * K + r] == o.ids[(first + t) * K + r] && part.weights[t * K + r] == o.weights[(first + t) * K + r];
      }
      for (int e = 0; e < E; ++e) same = same && part.logits[t * E + e] == o.logits[(first + t) * E + e];
    }
    check(same, "tokens routed " + std::to_string(count) + " at a time match the 192-token call bit for bit");
  }
  ties_arm();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_moe_router: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_moe_router: OK\n");
  return 0;
}
