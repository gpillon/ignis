// GitHub #627 (spec #618): the 27B MTP head's W8G32 matrices on an RTX 30 or RTX 40 card
// (kernel/src/w8_imma.h) -- OURS.
//
// - The route a load takes: ours on every card below compute capability 12 whatever the
//   measurement switch IGNIS_MTP_HEAD_LINEAR says, the vendored `ops::linear` on Blackwell unless
//   it names ours, and any value but `ours` and `vendored` refused naming the variable.
// - What the route takes: the head's five W8G32 geometries; a matrix of another format, rows not a
//   multiple of 16 or inputs not of 256, and a wide call without room for its planes are refused.
// - The activation's split is its definition (w8_imma.h `detail::split`) bit for bit: the planes
//   and scales a host restatement of the same fp32 arithmetic writes, a group of zeros included.
// - `linear` at each of the head's geometries (the stem's projection, the attention's fused
//   projection and output, the MLP's gate and up and its down) and a K-padded matrix, at the
//   widths a call has: 1 to 8 tokens (the vendored SIMT route), 9 to 128 (the integer MMA),
//   tokens that fill no tile and tiles of every width. Each sampled output against the W8 matrix
//   times the BF16 activation in f64 (Y):
//     |y - Y| <= 2^-8 |Y| + E,  E = Q + F
//   where 2^-8 |Y| is the BF16 output's rounding, F = (k / 32 + 16) 2^-23 S the fp32 steps'
//   (S = sum_k |w_k x_k|, every product and sum the routes take, each rounded at most once a
//   step), and Q, the integer MMA's alone, the fixed point's: sum over the groups of |ws| (amax /
//   65024) sum |code|, each activation within half a step of amax / kSplitRange. The integer
//   MMA is also held to the same product over the planes it read (Y'), where Q is zero: what
//   remains is the BF16 rounding and F, so a fragment, a token or a group out of place, which
//   moves an output by its own size, cannot hide in the fixed point's slack.
// - The integer MMA is deterministic and the same at every width: a column at 1, 9 or 64 tokens
//   is that column of a 128-token call, bit for bit.
//
// The head on the real artifact, on this route under the switch, is held to its weights in
// double precision by crates/core/tests/mtp_head_gpu.rs.
//
// Not sm120a-only: every kernel here builds for each architecture of the list. ADR 0006 /
// docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this test.

#include "w8_imma.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

namespace {

using ignis::RouteSwitch;
namespace imma = ignis::w8_imma;

int g_failed = 0;

void expect(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
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

template <class T>
T *device_copy(const std::vector<T> &host) {
  T *device = nullptr;
  cuda(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
  cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> host_copy(const void *device, std::size_t count) {
  std::vector<T> host(count);
  cuda(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return host;
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t word;
  std::memcpy(&word, &value, sizeof(word));
  word += 0x7FFFu + ((word >> 16) & 1u);
  return static_cast<std::uint16_t>(word >> 16);
}

// ---- the route rule ---------------------------------------------------------------------------

void the_route_is_ours_below_blackwell_and_the_switch_names_it_on_blackwell() {
  for (const int major : {8, 9, 10, 11}) {
    for (const RouteSwitch named : {RouteSwitch::Unset, RouteSwitch::Ours, RouteSwitch::Vendored}) {
      expect(imma::takes_ours(named, major),
             "a card of major " + std::to_string(major) + " runs ours whatever the switch says");
    }
  }
  expect(!imma::takes_ours(RouteSwitch::Unset, 12), "Blackwell keeps the vendored route by default");
  expect(!imma::takes_ours(RouteSwitch::Vendored, 12), "Blackwell keeps the vendored route when it is named");
  expect(imma::takes_ours(RouteSwitch::Ours, 12), "Blackwell runs ours when the switch names it");

  constexpr const char *kSwitch = "IGNIS_MTP_HEAD_LINEAR";
  set_environment(kSwitch, nullptr);
  expect(!imma::route_for_load(12) && imma::route_for_load(8), "unset: the card decides");
  set_environment(kSwitch, "ours");
  expect(imma::route_for_load(12), "ours on Blackwell");
  set_environment(kSwitch, "vendored");
  expect(!imma::route_for_load(12) && imma::route_for_load(8), "vendored on Blackwell, ours on 8.x");
  set_environment(kSwitch, "int8");
  bool refused = false;
  try {
    (void)imma::route_for_load(8);
  } catch (const std::invalid_argument &e) {
    refused = std::string(e.what()).find(kSwitch) != std::string::npos;
  }
  expect(refused, "any other value is refused, naming the variable");
  set_environment(kSwitch, nullptr);
}

// ---- matrices ----------------------------------------------------------------------------------

struct Matrix {
  const char *name = "";
  std::int32_t n = 0;
  std::int32_t k = 0;
  std::int32_t padded_k = 0;
  std::vector<std::int8_t> codes;   // [n][padded_k]
  std::vector<std::uint16_t> scales;  // fp16 [n][padded_k / 32]
  std::int8_t *d_codes = nullptr;
  std::uint16_t *d_scales = nullptr;

  ninfer::Weight weight() const {
    ninfer::Weight w{};
    w.qtype = ninfer::QType::W8G32_F16S;
    w.layout = ninfer::QuantLayout::RowSplit;
    w.qdata = d_codes;
    w.scales = d_scales;
    w.payload = d_codes;
    w.payload_bytes = static_cast<std::uint64_t>(n) * padded_k + static_cast<std::uint64_t>(n) * (padded_k / 32) * 2;
    w.ndim = 2;
    w.shape[0] = n;
    w.shape[1] = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = padded_k;
    w.n = n;
    w.k = k;
    w.group_size = 32;
    w.group = 32;
    w.scale_dtype = ninfer::DType::FP16;
    return w;
  }
  float scale(std::int32_t row, std::int32_t group) const {
    return __half2float(__ushort_as_half(scales[static_cast<std::size_t>(row) * (padded_k / 32) + group]));
  }
  std::int32_t code(std::int32_t row, std::int32_t input) const {
    return codes[static_cast<std::size_t>(row) * padded_k + input];
  }
};

// Codes over the whole INT8 range (the padding included, which no route may read into a sum: it
// holds codes that would show) and positive scales over a 2^4 span.
Matrix make_matrix(const char *name, std::int32_t n, std::int32_t k, std::int32_t padded_k, std::mt19937 &rng) {
  Matrix m;
  m.name = name;
  m.n = n;
  m.k = k;
  m.padded_k = padded_k;
  m.codes.resize(static_cast<std::size_t>(n) * padded_k);
  std::uniform_int_distribution<int> code(-128, 127);
  for (auto &c : m.codes) c = static_cast<std::int8_t>(code(rng));
  m.scales.resize(static_cast<std::size_t>(n) * (padded_k / 32));
  std::uniform_real_distribution<float> exponent(-10.0f, -6.0f);
  for (auto &s : m.scales) s = __half_as_ushort(__float2half_rn(std::exp2(exponent(rng))));
  m.d_codes = device_copy(m.codes);
  m.d_scales = device_copy(m.scales);
  return m;
}

void free_matrix(Matrix &m) {
  cuda(cudaFree(m.d_codes), "cudaFree");
  cuda(cudaFree(m.d_scales), "cudaFree");
}

// BF16 [k, tokens], token-major: normal values, one input in 97 forty times larger (the outlier
// channels a hidden state has), and in token 2 one group of zeros.
std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens, std::mt19937 &rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * tokens);
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t i = 0; i < k; ++i) {
      float v = normal(rng);
      if (i % 97 == 13) v *= 40.0f;
      if (t == 2 && i / 32 == 3) v = 0.0f;
      x[static_cast<std::size_t>(t) * k + i] = float_to_bf16(v);
    }
  }
  return x;
}

// ---- the split, restated ------------------------------------------------------------------------

struct HostPlanes {
  std::vector<std::int8_t> hi, lo;
  std::vector<float> scale;  // [k / 32][tokens]
};

HostPlanes host_split(const std::vector<std::uint16_t> &x, std::int32_t k, std::int32_t tokens) {
  HostPlanes p;
  p.hi.resize(static_cast<std::size_t>(k) * tokens);
  p.lo.resize(p.hi.size());
  p.scale.resize(static_cast<std::size_t>(k / 32) * tokens);
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t g = 0; g < k / 32; ++g) {
      float amax = 0.0f;
      for (std::int32_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_to_float(x[static_cast<std::size_t>(t) * k + 32 * g + i])));
      const bool live = amax > 0.0f && std::isfinite(amax);
      const float to_q = live ? static_cast<float>(imma::kSplitRange) / amax : 0.0f;
      for (std::int32_t i = 0; i < 32; ++i) {
        const std::size_t at = static_cast<std::size_t>(t) * k + 32 * g + i;
        const float v = bf16_to_float(x[at]) * to_q;
        long q = std::lrintf(v);
        q = std::min<long>(std::max<long>(q, -imma::kSplitRange), imma::kSplitRange);
        const long hi = (q + 128) >> 8;
        p.hi[at] = static_cast<std::int8_t>(hi);
        p.lo[at] = static_cast<std::int8_t>(q - 256 * hi);
      }
      p.scale[static_cast<std::size_t>(g) * tokens + t] = live ? amax / static_cast<float>(imma::kSplitRange) : 0.0f;
    }
  }
  return p;
}

void the_split_is_its_definition_bit_for_bit(std::mt19937 &rng) {
  constexpr std::int32_t k = 5120, tokens = 5;
  std::vector<std::uint16_t> x = make_activation(k, tokens, rng);
  // A group whose largest value is negative, and one whose values all tie in magnitude.
  for (std::int32_t i = 0; i < 32; ++i) {
    x[static_cast<std::size_t>(1) * k + 64 + i] = float_to_bf16(i == 7 ? -3.0f : 0.25f * static_cast<float>(i % 5));
    x[static_cast<std::size_t>(4) * k + 96 + i] = float_to_bf16(i % 2 ? 1.5f : -1.5f);
  }
  std::uint16_t *d_x = device_copy(x);
  const std::size_t bytes = imma::split_bytes(k, tokens);
  void *room = nullptr;
  cuda(cudaMalloc(&room, bytes), "cudaMalloc room");
  const ninfer::Tensor xt(d_x, ninfer::DType::BF16, {k, tokens});
  imma::detail::split(xt, room, bytes, nullptr);
  cuda(cudaDeviceSynchronize(), "split");
  const imma::detail::Planes planes = imma::detail::planes(room, k, tokens);
  const HostPlanes want = host_split(x, k, tokens);
  expect(host_copy<std::int8_t>(planes.hi, want.hi.size()) == want.hi, "the split's high plane");
  expect(host_copy<std::int8_t>(planes.lo, want.lo.size()) == want.lo, "the split's low plane");
  expect(host_copy<float>(planes.scale, want.scale.size()) == want.scale, "the split's scales");
  expect(want.scale[3 * tokens + 2] == 0.0f, "a group of zeros splits to scale 0");
  cuda(cudaFree(room), "cudaFree");
  cuda(cudaFree(d_x), "cudaFree");
}

// ---- what the route takes ------------------------------------------------------------------------

void what_the_route_takes(std::mt19937 &rng) {
  Matrix ok = make_matrix("16 x 256", 16, 256, 256, rng);
  expect(imma::serves(ok.weight()), "a W8G32 matrix of 16 rows and 256 inputs");
  ninfer::Weight bf16 = ok.weight();
  bf16.qtype = ninfer::QType::BF16_CTRL;
  expect(!imma::serves(bf16), "a BF16 matrix is not one");
  ninfer::Weight rows = ok.weight();
  rows.n = 24;
  expect(!imma::serves(rows), "rows not a multiple of 16");
  ninfer::Weight inputs = ok.weight();
  inputs.k = 128;
  expect(!imma::serves(inputs), "inputs not a multiple of 256");

  constexpr std::int32_t tokens = 9;
  std::uint16_t *d_x = device_copy(make_activation(256, tokens, rng));
  std::uint16_t *d_out = device_copy(std::vector<std::uint16_t>(16 * tokens));
  const ninfer::Tensor x(d_x, ninfer::DType::BF16, {256, tokens});
  ninfer::Tensor out(d_out, ninfer::DType::BF16, {16, tokens});
  void *room = nullptr;
  const std::size_t bytes = imma::split_bytes(256, tokens);
  cuda(cudaMalloc(&room, bytes), "cudaMalloc room");
  bool refused = false;
  try {
    imma::linear(x, ok.weight(), out, room, bytes - 1, nullptr);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  expect(refused, "a wide call without room for its planes is refused");
  refused = false;
  try {
    imma::linear(x, bf16, out, room, bytes, nullptr);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  expect(refused, "a matrix the route does not take is refused");
  // A narrow call runs the SIMT route and reads no room.
  const ninfer::Tensor narrow_x = x.slice(1, 0, 8);
  ninfer::Tensor narrow_out = out.slice(1, 0, 8);
  imma::linear(narrow_x, ok.weight(), narrow_out, nullptr, 0, nullptr);
  cuda(cudaDeviceSynchronize(), "narrow call");
  cuda(cudaFree(room), "cudaFree");
  cuda(cudaFree(d_x), "cudaFree");
  cuda(cudaFree(d_out), "cudaFree");
  free_matrix(ok);
}

// ---- the product -------------------------------------------------------------------------------

constexpr double kBf16Unit = 1.0 / 256.0;

struct Bound {
  double value = 0.0;  // the f64 product
  double slack = 0.0;  // E
};

// Row `row`, token `t` of the product over the BF16 activation itself (`planes` null), or over
// the planes the integer MMA read.
Bound product(const Matrix &m, const std::vector<std::uint16_t> &x, std::int32_t tokens, std::int32_t row,
              std::int32_t t, const HostPlanes *planes, bool fixed_point) {
  double y = 0.0, s = 0.0, q = 0.0;
  for (std::int32_t g = 0; g < m.k / 32; ++g) {
    const double ws = m.scale(row, g);
    double amax = 0.0, codes = 0.0;
    for (std::int32_t i = 0; i < 32; ++i) {
      const std::int32_t input = 32 * g + i;
      const double w = ws * m.code(row, input);
      const std::size_t at = static_cast<std::size_t>(t) * m.k + input;
      const double a = planes == nullptr
                           ? static_cast<double>(bf16_to_float(x[at]))
                           : planes->scale[static_cast<std::size_t>(g) * tokens + t] *
                                 (256.0 * planes->hi[at] + planes->lo[at]);
      y += w * a;
      s += std::fabs(w * a);
      amax = std::max(amax, std::fabs(static_cast<double>(bf16_to_float(x[at]))));
      codes += std::fabs(static_cast<double>(m.code(row, input)));
    }
    // Half a step, and the fp32 product x (kSplitRange / amax) the rounding reads, within 2^-24 of
    // kSplitRange steps.
    if (fixed_point) q += ws * (amax / (2.0 * imma::kSplitRange)) * 1.01 * codes;
  }
  const double f = (m.k / 32 + 16) * std::ldexp(1.0, -23) * s;
  return Bound{y, q + f};
}

// The rows a check reads: the first three CTAs' rows, the last CTA's and 48 between.
std::vector<std::int32_t> sampled_rows(std::int32_t n, std::mt19937 &rng) {
  std::vector<std::int32_t> rows;
  for (std::int32_t r = 0; r < std::min(n, 48); ++r) rows.push_back(r);
  for (std::int32_t r = std::max(48, n - 16); r < n; ++r) rows.push_back(r);
  std::uniform_int_distribution<std::int32_t> pick(0, n - 1);
  for (int i = 0; i < 48 && n > 64; ++i) rows.push_back(pick(rng));
  return rows;
}

void the_route_computes_each_geometry_at_each_width(std::mt19937 &rng) {
  constexpr std::int32_t kMaxTokens = 128;
  struct Geometry {
    const char *name;
    std::int32_t n, k, padded_k;
  };
  const Geometry geometries[] = {
      {"the stem's projection [5120, 10240]", 5120, 10240, 10240},
      {"the attention's fused projection [14336, 5120]", 14336, 5120, 5120},
      {"the attention's output [5120, 6144]", 5120, 6144, 6144},
      {"the MLP's gate and up [34816, 5120]", 34816, 5120, 5120},
      {"the MLP's down [5120, 17408]", 5120, 17408, 17408},
      {"a K-padded matrix [32, 256 stored 384]", 32, 256, 384},
  };
  const std::int32_t widths[] = {1, 4, 5, 8, 9, 16, 17, 33, 64, 65, kMaxTokens};
  void *room = nullptr;
  const std::size_t room_bytes = imma::split_bytes(17408, kMaxTokens);
  cuda(cudaMalloc(&room, room_bytes), "cudaMalloc room");
  for (const Geometry &g : geometries) {
    Matrix m = make_matrix(g.name, g.n, g.k, g.padded_k, rng);
    const std::vector<std::uint16_t> x = make_activation(g.k, kMaxTokens, rng);
    std::uint16_t *d_x = device_copy(x);
    std::uint16_t *d_out = nullptr;
    cuda(cudaMalloc(&d_out, static_cast<std::size_t>(g.n) * kMaxTokens * 2), "cudaMalloc out");
    const std::vector<std::int32_t> rows = sampled_rows(g.n, rng);
    double worst = 0.0;
    for (const std::int32_t tokens : widths) {
      const ninfer::Tensor xt(d_x, ninfer::DType::BF16, {g.k, tokens});
      ninfer::Tensor out(d_out, ninfer::DType::BF16, {g.n, tokens});
      cuda(cudaMemset(d_out, 0xFF, static_cast<std::size_t>(g.n) * tokens * 2), "cudaMemset out");
      imma::linear(xt, m.weight(), out, room, room_bytes, nullptr);
      cuda(cudaDeviceSynchronize(), "linear");
      const auto y = host_copy<std::uint16_t>(d_out, static_cast<std::size_t>(g.n) * tokens);
      const bool wide = tokens > imma::kSimtMaxTokens;
      HostPlanes read;
      if (wide) {
        const imma::detail::Planes p = imma::detail::planes(room, g.k, tokens);
        const std::size_t count = static_cast<std::size_t>(g.k) * tokens;
        read.hi = host_copy<std::int8_t>(p.hi, count);
        read.lo = host_copy<std::int8_t>(p.lo, count);
        read.scale = host_copy<float>(p.scale, static_cast<std::size_t>(g.k / 32) * tokens);
      }
      int bad = 0;
      for (const std::int32_t r : rows) {
        for (std::int32_t t = 0; t < tokens; ++t) {
          const double got = bf16_to_float(y[static_cast<std::size_t>(t) * g.n + r]);
          const Bound want = product(m, x, tokens, r, t, nullptr, wide);
          const double limit = kBf16Unit * std::fabs(want.value) + (1.0 + kBf16Unit) * want.slack;
          worst = std::max(worst, std::fabs(got - want.value) / limit);
          if (!(std::fabs(got - want.value) <= limit) && bad++ < 3) {
            std::fprintf(stderr, "  %s, %d tokens, row %d token %d: %.9g against %.9g (limit %.3g)\n", g.name, tokens, r, t,
                         got, want.value, limit);
          }
          if (wide) {
            const Bound exact = product(m, x, tokens, r, t, &read, false);
            const double tight = kBf16Unit * std::fabs(exact.value) + (1.0 + kBf16Unit) * exact.slack;
            if (!(std::fabs(got - exact.value) <= tight) && bad++ < 3) {
              std::fprintf(stderr, "  %s, %d tokens, row %d token %d: %.9g against the planes' %.9g (limit %.3g)\n",
                           g.name, tokens, r, t, got, exact.value, tight);
            }
          }
        }
      }
      expect(bad == 0, std::string(g.name) + " at " + std::to_string(tokens) + " tokens");
    }
    std::printf("%s: worst error %.3f of its bound\n", g.name, worst);
    cuda(cudaFree(d_out), "cudaFree");
    cuda(cudaFree(d_x), "cudaFree");
    free_matrix(m);
  }
  cuda(cudaFree(room), "cudaFree");
}

void a_column_of_the_integer_mma_does_not_depend_on_the_columns_beside_it(std::mt19937 &rng) {
  constexpr std::int32_t n = 5120, k = 17408, kMaxTokens = 128;
  Matrix m = make_matrix("down", n, k, k, rng);
  std::uint16_t *d_x = device_copy(make_activation(k, kMaxTokens, rng));
  std::uint16_t *d_wide = nullptr, *d_narrow = nullptr;
  cuda(cudaMalloc(&d_wide, static_cast<std::size_t>(n) * kMaxTokens * 2), "cudaMalloc");
  cuda(cudaMalloc(&d_narrow, static_cast<std::size_t>(n) * kMaxTokens * 2), "cudaMalloc");
  void *room = nullptr;
  const std::size_t room_bytes = imma::split_bytes(k, kMaxTokens);
  cuda(cudaMalloc(&room, room_bytes), "cudaMalloc room");
  ninfer::Tensor wide(d_wide, ninfer::DType::BF16, {n, kMaxTokens});
  imma::detail::imma(ninfer::Tensor(d_x, ninfer::DType::BF16, {k, kMaxTokens}), m.weight(), wide, room, room_bytes, nullptr);
  cuda(cudaDeviceSynchronize(), "the wide call");
  const auto all = host_copy<std::uint16_t>(d_wide, static_cast<std::size_t>(n) * kMaxTokens);
  for (const std::int32_t tokens : {1, 9, 64, kMaxTokens}) {
    ninfer::Tensor narrow(d_narrow, ninfer::DType::BF16, {n, tokens});
    imma::detail::imma(ninfer::Tensor(d_x, ninfer::DType::BF16, {k, tokens}), m.weight(), narrow, room, room_bytes, nullptr);
    cuda(cudaDeviceSynchronize(), "the narrow call");
    const auto part = host_copy<std::uint16_t>(d_narrow, static_cast<std::size_t>(n) * tokens);
    expect(std::equal(part.begin(), part.end(), all.begin()),
           "the first " + std::to_string(tokens) + " columns at " + std::to_string(tokens) + " tokens are the 128-token call's");
  }
  cuda(cudaFree(room), "cudaFree");
  cuda(cudaFree(d_narrow), "cudaFree");
  cuda(cudaFree(d_wide), "cudaFree");
  cuda(cudaFree(d_x), "cudaFree");
  free_matrix(m);
}

}  // namespace

int main() {
  int devices = 0;
  cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  std::mt19937 rng(627);
  the_route_is_ours_below_blackwell_and_the_switch_names_it_on_blackwell();
  what_the_route_takes(rng);
  the_split_is_its_definition_bit_for_bit(rng);
  the_route_computes_each_geometry_at_each_width(rng);
  a_column_of_the_integer_mma_does_not_depend_on_the_columns_beside_it(rng);
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
