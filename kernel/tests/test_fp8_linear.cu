// The FP8 row-scale linear (spec flash-next/04's op, written for spec 02's shared expert) at
// Flash-Next's shapes -- OURS (GitHub #300).
//
//   linear   [640][2560] and [2560][640] (the shared expert's projections) and [48][2560] (the
//            GDN gating rows, narrower than a tile) at 1, 3, 8 tokens (GEMV route) and 9..2048
//            (tensor-core route), fp32 and BF16 outputs, against fp64.
//   swiglu   the shared expert's gate/up pair, silu(g) * u in BF16, against fp64.
//   grouped  (GitHub #306) GDN's four projections in one grouped GEMV launch against one linear
//            each, bit for bit, 1..8 tokens; and its refusals.
//   paired   (GitHub #678) the grouped GEMV's paired form against its one-row form, bit for bit,
//            1..8 tokens, at the 27B's q | k | gate | v and at four other column widths; and its
//            refusal past 6144 columns.
//   add      (GitHub #341) the residual add on both routes against BF16(residual + the fp32
//            linear), bit for bit.
//   quantize (GitHub #341) a BF16 weight to the row-scale-v1 payload, every code and scale
//            against the contract; and its refusals.
//
// Tolerance: E4M3 x BF16 products are exact in fp32, so the error is fp32 accumulation, bounded
// by (cols / 8 + 64) u sum|terms| (fp8_test_common.h), plus the output's own rounding: one fp32
// rounding of the scaled sum, or BF16's half ulp (at most 2^-8 relative). Every output is checked; for
// the widest calls a fixed sample of tokens. Each call also runs twice and must agree bit for
// bit.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "fp8_test_common.h"
#include "ignis_fp8_linear.h"
#include "ignis_moe.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

// GitHub #306: the grouped GEMV (ignis_fp8_linear_grouped) against one ignis_fp8_linear call per
// segment, bit for bit -- GDN's four projections of one input at Flash-Next's shapes, every
// output poisoned first -- and what it refuses.
void grouped_arm(int tokens, bool f32_out) {
  const int cols = IGNIS_MOE_HIDDEN;
  const int shapes[4] = {10240, 6144, 48, 48};
  const std::size_t out_bytes = f32_out ? 4 : 2;
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(79, i, 2.0f));
  DeviceBytes dx(x.size() * 2);
  upload(dx, x);
  std::vector<std::unique_ptr<DeviceBytes>> w, y_one, y_grouped;
  ignis_fp8_segment segments[4];
  for (int i = 0; i < 4; ++i) {
    const Fp8Matrix m = make_fp8(static_cast<uint32_t>(4000 + i), shapes[i], cols, 0.004f);
    w.push_back(std::make_unique<DeviceBytes>(m.payload.size()));
    upload(*w.back(), m.payload);
    for (auto *ys : {&y_one, &y_grouped}) {
      ys->push_back(std::make_unique<DeviceBytes>(static_cast<std::size_t>(tokens) * shapes[i] * out_bytes));
      MOE_CUDA(cudaMemset(ys->back()->p, 0xFF, ys->back()->bytes));
    }
    MOE_RC(ignis_fp8_linear(w[i]->p, shapes[i], cols, dx.p, tokens, y_one[i]->p, f32_out ? 1 : 0, nullptr));
    segments[i] = ignis_fp8_segment{w[i]->p, static_cast<uint32_t>(shapes[i]), y_grouped[i]->p};
  }
  MOE_RC(ignis_fp8_linear_grouped(segments, 4, cols, dx.p, tokens, f32_out ? 1 : 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  bool same = true;
  for (int i = 0; i < 4; ++i) {
    same = same && download<uint8_t>(y_grouped[i]->p, y_grouped[i]->bytes) ==
                       download<uint8_t>(y_one[i]->p, y_one[i]->bytes);
  }
  check(same, "fp8 grouped GEMV x " + std::to_string(tokens) + (f32_out ? " (fp32)" : " (bf16)") +
                  ": every segment is its own linear's, bit for bit");
  if (tokens == 1) {
    check(ignis_fp8_linear_grouped(segments, 4, cols, dx.p, 9, 0, nullptr) != 0, "fp8 grouped GEMV: 9 tokens refused");
    check(ignis_fp8_linear_grouped(segments, 5, cols, dx.p, 1, 0, nullptr) != 0, "fp8 grouped GEMV: 5 segments refused");
    ignis_fp8_segment odd = segments[2];
    odd.rows = 40;
    check(ignis_fp8_linear_grouped(&odd, 1, cols, dx.p, 1, 0, nullptr) != 0,
          "fp8 grouped GEMV: rows off a multiple of 16 refused");
    odd = segments[2];
    odd.weight = static_cast<const char *>(segments[2].weight) + 8;
    check(ignis_fp8_linear_grouped(&odd, 1, cols, dx.p, 1, 0, nullptr) != 0,
          "fp8 grouped GEMV: a weight off 16 bytes refused");
  }
}

// GitHub #678: the grouped GEMV's paired form (ignis_fp8_linear_grouped_paired) against its one-row
// form (ignis_fp8_linear_grouped), bit for bit at 1..8 tokens, every output poisoned first, for
// segments of `rows` over `cols` columns; fp32 outputs at even token counts, BF16 at odd ones.
void paired_arm(const std::vector<int> &rows, int cols, const std::string &what) {
  const int count = static_cast<int>(rows.size());
  std::vector<std::unique_ptr<DeviceBytes>> w;
  for (int i = 0; i < count; ++i) {
    const Fp8Matrix m = make_fp8(static_cast<uint32_t>(5200 + 16 * i + cols), rows[i], cols, 0.004f);
    w.push_back(std::make_unique<DeviceBytes>(m.payload.size()));
    upload(*w.back(), m.payload);
  }
  for (int tokens = 1; tokens <= IGNIS_FP8_GEMV_MAX_TOKENS; ++tokens) {
    const bool f32_out = tokens % 2 == 0;
    std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(84, i, 2.0f));
    DeviceBytes dx(x.size() * 2);
    upload(dx, x);
    std::vector<std::unique_ptr<DeviceBytes>> y_one, y_paired;
    std::vector<ignis_fp8_segment> one(count), paired(count);
    for (int i = 0; i < count; ++i) {
      for (auto *ys : {&y_one, &y_paired}) {
        ys->push_back(std::make_unique<DeviceBytes>(static_cast<std::size_t>(tokens) * rows[i] * (f32_out ? 4 : 2)));
        MOE_CUDA(cudaMemset(ys->back()->p, 0xFF, ys->back()->bytes));
      }
      one[i] = ignis_fp8_segment{w[i]->p, static_cast<uint32_t>(rows[i]), y_one[i]->p};
      paired[i] = ignis_fp8_segment{w[i]->p, static_cast<uint32_t>(rows[i]), y_paired[i]->p};
    }
    MOE_RC(ignis_fp8_linear_grouped(one.data(), count, cols, dx.p, tokens, f32_out ? 1 : 0, nullptr));
    MOE_RC(ignis_fp8_linear_grouped_paired(paired.data(), count, cols, dx.p, tokens, f32_out ? 1 : 0, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    bool same = true;
    for (int i = 0; i < count; ++i) {
      same = same && download<uint8_t>(y_paired[i]->p, y_paired[i]->bytes) ==
                         download<uint8_t>(y_one[i]->p, y_one[i]->bytes);
    }
    check(same, "fp8 grouped GEMV paired " + what + " x " + std::to_string(tokens) + (f32_out ? " (fp32)" : " (bf16)") +
                    ": the one-row form's bytes");
  }
  std::printf("  grouped paired %s x 1..8 tokens: compared with the one-row form byte for byte\n", what.c_str());
}

// GitHub #678: what the paired form refuses beyond the grouped call's own limits: past 6144 columns.
void paired_refusal() {
  const Fp8Matrix m = make_fp8(5300, 64, 7168, 0.004f);
  DeviceBytes dw(m.payload.size()), dx(7168 * 2), dy(64 * 2);
  upload(dw, m.payload);
  const ignis_fp8_segment one{dw.p, 64, dy.p};
  check(ignis_fp8_linear_grouped_paired(&one, 1, 7168, dx.p, 1, 0, nullptr) != 0,
        "fp8 grouped GEMV paired: 7168 columns refused");
}

// GitHub #341: the residual add against the fp32 linear plus the add on the host, bit for bit, on
// both routes.
void linear_add_arm(int tokens) {
  const int rows = 5120, cols = 6144;
  const Fp8Matrix m = make_fp8(4100, rows, cols, 0.004f);
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols), res(static_cast<std::size_t>(tokens) * rows);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(79, i, 2.0f));
  for (std::size_t i = 0; i < res.size(); ++i) res[i] = f32_to_bf16(hash_uniform(80, i, 3.0f));
  DeviceBytes dw(m.payload.size()), dx(x.size() * 2), dyf(res.size() * 4), dr(res.size() * 2);
  upload(dw, m.payload);
  upload(dx, x);
  upload(dr, res);
  MOE_RC(ignis_fp8_linear(dw.p, rows, cols, dx.p, tokens, dyf.p, 1, nullptr));
  MOE_RC(ignis_fp8_linear_add(dw.p, rows, cols, dx.p, tokens, dr.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto yf = download<float>(dyf.p, res.size());
  const auto got = download<uint16_t>(dr.p, res.size());
  std::size_t off = 0;
  for (std::size_t i = 0; i < res.size(); ++i) off += got[i] != f32_to_bf16(bf16_to_f32(res[i]) + yf[i]);
  std::printf("  linear add [%d][%d] x %d tokens: %zu of %zu outputs off BF16(residual + y)\n", rows, cols, tokens, off,
              res.size());
  check(off == 0, "fp8 linear add x " + std::to_string(tokens) + " is BF16(residual + y) bit for bit");
}

// The E4M3FN code nearest to v (ties to the even code), saturated to +-448, the sign kept on a
// zero: what cvt.rn.satfinite gives for a finite fp32. By search over the 127 magnitudes, so it shares no arithmetic with the
// kernel.
uint8_t e4m3_nearest(float v) {
  const float a = std::fabs(v);
  uint8_t best = 0;
  float best_err = a;
  for (uint8_t c = 1; c < 0x7F; ++c) {
    const float err = std::fabs(e4m3_to_f32(c) - a);
    if (err < best_err || (err == best_err && (c & 1) == 0)) {
      best = c;
      best_err = err;
    }
  }
  return static_cast<uint8_t>(best | (std::signbit(v) ? 0x80 : 0));
}

// The least BF16 >= q (q finite and positive).
uint16_t bf16_at_least(float q) {
  uint16_t b = f32_to_bf16(q);
  if (bf16_to_f32(b) < q) ++b;
  return b;
}

// GitHub #341: a BF16 weight to the row-scale-v1 payload, every code and scale against the host's
// own reading of the contract -- scale = the least BF16 >= fl32(amax / 448) (1 for an all-zero
// row), code = E4M3(fl32(w / scale)), nearest even, saturated. Rows of very different magnitude, an
// all-zero row and a row of one outlier; a second run agrees bit for bit; and its refusals.
void quantize_arm(int rows, int cols) {
  std::vector<uint16_t> w(static_cast<std::size_t>(rows) * cols);
  for (int r = 0; r < rows; ++r) {
    const float mag = std::ldexp(1.0f, (r % 13) - 9);
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      w[i] = f32_to_bf16(r == 1 ? 0.0f : r == 2 ? (c == 77 ? 3.5f : hash_uniform(91, i, 0.01f)) : hash_uniform(90, i, mag));
    }
  }
  const uint64_t bytes = ignis_fp8_payload_bytes(rows, cols);
  const std::size_t scale_at = (static_cast<std::size_t>(rows) * cols + 255) / 256 * 256;
  check(bytes == scale_at + static_cast<uint64_t>(rows) * 2, "fp8 payload bytes: codes, padding to 256, BF16 scales");
  DeviceBytes dw(w.size() * 2), dp(bytes);
  upload(dw, w);
  MOE_CUDA(cudaMemset(dp.p, 0xA5, bytes));
  MOE_RC(ignis_fp8_quantize_rows(dw.p, rows, cols, dp.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto got = download<uint8_t>(dp.p, bytes);
  MOE_CUDA(cudaMemset(dp.p, 0x5A, bytes));
  MOE_RC(ignis_fp8_quantize_rows(dw.p, rows, cols, dp.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  check(download<uint8_t>(dp.p, bytes) == got, "fp8 quantize: a second run agrees bit for bit");

  int bad_codes = 0, bad_scales = 0;
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(bf16_to_f32(w[static_cast<std::size_t>(r) * cols + c])));
    const uint16_t scale = amax == 0.0f ? f32_to_bf16(1.0f) : bf16_at_least(amax / 448.0f);
    uint16_t have;
    std::memcpy(&have, &got[scale_at + 2 * static_cast<std::size_t>(r)], 2);
    if (have != scale) ++bad_scales;
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      if (got[i] != e4m3_nearest(bf16_to_f32(w[i]) / bf16_to_f32(scale))) ++bad_codes;
    }
  }
  std::printf("  quantize [%5d][%5d]: %d scale(s) and %d code(s) off the contract\n", rows, cols, bad_scales, bad_codes);
  check(bad_scales == 0 && bad_codes == 0, "fp8 quantize [" + std::to_string(rows) + "][" + std::to_string(cols) + "] matches the contract");
  check(std::all_of(got.begin() + static_cast<std::ptrdiff_t>(rows) * cols, got.begin() + static_cast<std::ptrdiff_t>(scale_at),
                    [](uint8_t b) { return b == 0; }),
        "fp8 quantize: the padding before the scales is zero");
  if (rows == 64) {
    check(ignis_fp8_quantize_rows(dw.p, rows, 100, dp.p, nullptr) != 0, "fp8 quantize: cols off 16 refused");
    check(ignis_fp8_quantize_rows(dw.p, 0, cols, dp.p, nullptr) != 0, "fp8 quantize: zero rows refused");
    check(ignis_fp8_quantize_rows(static_cast<const char *>(dw.p) + 2, rows, cols, dp.p, nullptr) != 0,
          "fp8 quantize: a weight off 16 bytes refused");
  }
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  MOE_RC(ignis_fp8_linear_prepare());
  std::printf("FP8 row-scale linear (E4M3 + BF16 row scale), BF16 activations, fp32 accumulation\n");
  for (int tokens : {1, 3, 8, 9, 200, 2048}) linear_arm(640, 2560, tokens, tokens % 2 == 1);
  for (int tokens : {1, 8, 129, 1000}) linear_arm(2560, 640, tokens, true);
  for (int tokens : {2, 300}) linear_arm(48, 2560, tokens, false);
  // GitHub #341: the 27B's exception shapes, on both routes.
  for (int tokens : {8, 9, 128, 200}) linear_arm(5120, 6144, tokens, tokens % 2 == 0);
  for (int tokens : {1, 9, 128}) linear_arm(6144, 5120, tokens, false);
  for (int tokens : {1, 3, 8, 64, 1024}) swiglu_arm(tokens);
  for (int tokens : {1, 2, 3, 5, 8}) grouped_arm(tokens, tokens == 3);
  for (int tokens : {1, 3, 8, 9, 200}) linear_add_arm(tokens);
  // GitHub #678: the paired form at the 27B's q | k | gate | v [14336][5120] and at the widths a
  // lane's chunks take past or up to a row's end: Flash-Next GDN's four projections at 2560, and one
  // segment at 640, 4096 and 6144 columns.
  paired_arm({6144, 1024, 6144, 1024}, 5120, "q|k|gate|v [14336][5120]");
  paired_arm({10240, 6144, 48, 48}, IGNIS_MOE_HIDDEN, "GDN [16480][2560]");
  paired_arm({2560}, 640, "[2560][640]");
  paired_arm({512}, 4096, "[512][4096]");
  paired_arm({5120}, 6144, "[5120][6144]");
  paired_refusal();
  quantize_arm(64, 5120);
  quantize_arm(16, 6144);
  quantize_arm(3, 2560);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_fp8_linear: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_fp8_linear: OK\n");
  return 0;
}
