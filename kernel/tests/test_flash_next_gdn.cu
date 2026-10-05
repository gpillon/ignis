// GitHub #302 (spec flash-next/04, slice S2): OURS -- the Flash-Next GDN layer core
// (kernel/src/flash_next/gdn.cu) at the real geometry (hidden 2560, 16 / 48 heads of 128, conv 4,
// FP8 row-scale projections) against an fp64 restatement of the checkpoint's
// Qwen4ExpTextGatedDeltaNet that rounds to BF16 where the checkpoint stores BF16.
//
// Three lanes in a five-slot state pool, as the program drives them: one-lane prefill calls (one
// crossing the recurrence's 64-token chunk), a one-token prefill, a three-lane decode round run
// eagerly, two more captured once as a CUDA graph and replayed with new inputs, then a second
// prefill chunk on a lane whose state is no longer zero. Every call's output, every lane's final
// state, the untouched slots and every call's scratch peak (against fn_gdn_layer_scratch_bytes)
// are checked.
//
// Tolerance: the layer's output inherits the vendored recurrence's own criterion (relative L2
// 4.1e-3, gross 5.5e-3 of the largest reference; vendor/tests/ops/test_gated_delta_net.cpp),
// plus the BF16 output rounding (2^-9 relative) and the rare BF16 rounding flips of the
// intermediate stores. Checked: relative L2 per call <= 2^-7 and every element within 2^-6 of the
// call's largest reference; the fp32 state within relative L2 2^-7 of the fp64 one; the conv taps
// (BF16 projections) within one BF16 ulp, almost all of them exact.

#include "flash_next/gdn.h"

#include "fp8_test_common.h"
#include "ignis_fp8_linear.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace moe_test;
namespace fn = ignis::flash_next;

#define FN_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, fn::fn_last_error());       \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

constexpr int H = 2560;
constexpr int C = fn::gdn::kConvChannels;
constexpr int KW = fn::gdn::kKeyWidth;
constexpr int VW = fn::gdn::kValueWidth;
constexpr int VH = fn::gdn::kValueHeads;
constexpr int D = fn::gdn::kHeadDim;
constexpr int kSlots = 5;
constexpr float kEps = 1e-6F;

double bf16r(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

// An FP8 row-scale matrix kept as its device payload plus a code LUT (no fp64 copy of the codes:
// the projections here are tens of millions of weights).
struct Proj {
  int rows = 0, cols = 0;
  std::vector<uint8_t> payload;
  std::vector<double> scale;
  std::unique_ptr<DeviceBytes> dev;
};

double g_lut[256];

double code_rms() {
  double s = 0.0;
  int n = 0;
  for (int c = 0; c < 256; ++c) {
    if ((c & 0x7F) == 0x7F) continue;
    s += g_lut[c] * g_lut[c];
    ++n;
  }
  return std::sqrt(s / n);
}

void make_proj(Proj &p, uint32_t stream, int rows, int cols, double input_rms, double output_rms) {
  const double mag = output_rms / (std::sqrt(static_cast<double>(cols)) * code_rms() * input_rms);
  Fp8Matrix m = make_fp8(stream, rows, cols, static_cast<float>(mag));
  p.rows = rows;
  p.cols = cols;
  p.payload = std::move(m.payload);
  p.scale = std::move(m.scale);
  p.dev = std::make_unique<DeviceBytes>(p.payload.size());
  upload(*p.dev, p.payload);
}

fn::Linear linear(const Proj &p) {
  fn::Linear l;
  l.data = p.dev->p;
  l.rows = p.rows;
  l.cols = p.cols;
  l.format = fn::WeightFormat::Fp8RowScale;
  return l;
}

// y[r] = scale[r] * sum_c code[r][c] * x[c] in fp64, rows split over threads.
std::vector<double> project(const Proj &p, const std::vector<double> &x) {
  std::vector<double> y(p.rows);
  const unsigned threads = 8;
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t) {
    pool.emplace_back([&, t]() {
      for (int r = static_cast<int>(t); r < p.rows; r += static_cast<int>(threads)) {
        const uint8_t *row = &p.payload[static_cast<std::size_t>(r) * p.cols];
        double s = 0.0;
        for (int c = 0; c < p.cols; ++c) s += g_lut[row[c]] * x[c];
        y[r] = s * p.scale[r];
      }
    });
  }
  for (auto &t : pool) t.join();
  return y;
}

struct Weights {
  Proj qkv, z, a, b, out;
  std::vector<uint16_t> conv, a_log, dt_bias, norm;  // [C][4], [VH], [VH], [D]
  std::unique_ptr<DeviceBytes> d_conv, d_a_log, d_dt_bias, d_norm;
  fn::GdnWeights view() const {
    fn::GdnWeights w;
    w.in_proj_qkv = linear(qkv);
    w.in_proj_z = linear(z);
    w.in_proj_a = linear(a);
    w.in_proj_b = linear(b);
    w.out_proj = linear(out);
    w.conv = d_conv->p;
    w.a_log = d_a_log->p;
    w.dt_bias = d_dt_bias->p;
    w.norm = d_norm->p;
    return w;
  }
};

std::vector<uint16_t> bf16_vector(uint32_t stream, std::size_t n, double lo, double hi) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double u = 0.5 * (hash_uniform(stream, i, 1.0F) + 1.0);
    v[i] = f32_to_bf16(static_cast<float>(lo + (hi - lo) * u));
  }
  return v;
}

void build(Weights &w) {
  const double x_rms = 1.0 / std::sqrt(3.0);  // inputs uniform in [-1, 1)
  make_proj(w.qkv, 101, C, H, x_rms, 1.0);
  make_proj(w.z, 103, VW, H, x_rms, 1.0);
  make_proj(w.a, 105, VH, H, x_rms, 0.5);
  make_proj(w.b, 107, VH, H, x_rms, 1.0);
  make_proj(w.out, 109, H, VW, 0.5, 1.0);
  w.conv = bf16_vector(111, static_cast<std::size_t>(C) * 4, -0.5, 0.5);
  // A_log = log(U(1, 16)) (the checkpoint's init); dt_bias low, so softplus(a + dt_bias) is small
  // and the state keeps a long memory -- the carry between calls matters.
  w.a_log.resize(VH);
  for (int h = 0; h < VH; ++h) {
    w.a_log[h] = f32_to_bf16(std::log(1.0F + 7.5F * (hash_uniform(113, h, 1.0F) + 1.0F)));
  }
  w.dt_bias = bf16_vector(115, VH, -5.0, -2.0);
  w.norm = bf16_vector(117, D, 0.5, 1.5);
  for (auto [dev, host] : {std::pair{&w.d_conv, &w.conv}, std::pair{&w.d_a_log, &w.a_log},
                           std::pair{&w.d_dt_bias, &w.dt_bias}, std::pair{&w.d_norm, &w.norm}}) {
    *dev = std::make_unique<DeviceBytes>(host->size() * 2);
    upload(**dev, *host);
  }
}

// One lane of the fp64 restatement: its conv taps (the last three BF16 projected qkv rows) and
// its recurrent state, [VH][D (value)][D (key)].
struct Lane {
  std::vector<double> taps = std::vector<double>(static_cast<std::size_t>(3) * C, 0.0);
  std::vector<double> state = std::vector<double>(static_cast<std::size_t>(VH) * D * D, 0.0);
};

double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

// One token through the layer; returns y (fp64, unrounded).
std::vector<double> reference_token(const Weights &w, Lane &lane, const std::vector<double> &x) {
  std::vector<double> qkv = project(w.qkv, x), z = project(w.z, x), a = project(w.a, x), b = project(w.b, x);
  for (auto *v : {&qkv, &z, &a, &b}) {
    for (double &e : *v) e = bf16r(e);
  }
  // Depthwise causal conv over [taps..., qkv] then SiLU, rounded once (the vendored op's storage).
  std::vector<double> conv(C);
  for (int c = 0; c < C; ++c) {
    double s = 0.0;
    for (int j = 0; j < 3; ++j) s += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * 4 + j]) * lane.taps[static_cast<std::size_t>(j) * C + c];
    s += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * 4 + 3]) * qkv[c];
    conv[c] = bf16r(s * sigmoid(s));
  }
  for (int j = 0; j < 2; ++j) {
    std::copy_n(&lane.taps[static_cast<std::size_t>(j + 1) * C], C, &lane.taps[static_cast<std::size_t>(j) * C]);
  }
  std::copy_n(qkv.data(), C, &lane.taps[static_cast<std::size_t>(2) * C]);

  auto l2 = [](const double *v, double *out) {
    double s = 0.0;
    for (int d = 0; d < D; ++d) s += v[d] * v[d];
    const double inv = 1.0 / std::sqrt(s + 1e-6);
    for (int d = 0; d < D; ++d) out[d] = v[d] * inv;
  };
  std::vector<double> gated(VW);
  const double scale = 1.0 / std::sqrt(static_cast<double>(D));
  for (int h = 0; h < VH; ++h) {
    const int qh = h / 3;  // repeat_interleave(3) of the 16 q/k heads
    double q[D], k[D];
    l2(&conv[static_cast<std::size_t>(qh) * D], q);
    l2(&conv[KW + static_cast<std::size_t>(qh) * D], k);
    const double *v = &conv[2 * KW + static_cast<std::size_t>(h) * D];
    const double x_dt = a[h] + bf16_to_f32(w.dt_bias[h]);
    const double softplus = x_dt > 20.0 ? x_dt : std::log1p(std::exp(x_dt));
    const double alpha = std::exp(-std::exp(static_cast<double>(bf16_to_f32(w.a_log[h]))) * softplus);
    const double beta = bf16r(sigmoid(b[h]));
    double *s = &lane.state[static_cast<std::size_t>(h) * D * D];
    double delta[D];
    for (int r = 0; r < D; ++r) {
      double sk = 0.0;
      for (int c = 0; c < D; ++c) sk += s[r * D + c] * k[c];
      delta[r] = beta * (v[r] - alpha * sk);
    }
    for (int r = 0; r < D; ++r) {
      for (int c = 0; c < D; ++c) s[r * D + c] = alpha * s[r * D + c] + delta[r] * k[c];
    }
    double o[D], sum = 0.0;
    for (int r = 0; r < D; ++r) {
      double sq = 0.0;
      for (int c = 0; c < D; ++c) sq += s[r * D + c] * q[c];
      o[r] = bf16r(scale * sq);
      sum += o[r] * o[r];
    }
    const double inv = 1.0 / std::sqrt(sum / D + kEps);
    for (int r = 0; r < D; ++r) {
      const double n = bf16r(o[r] * inv);
      const double m = bf16r(bf16_to_f32(w.norm[r]) * n);
      gated[static_cast<std::size_t>(h) * D + r] = bf16r(m * sigmoid(z[static_cast<std::size_t>(h) * D + r]));
    }
  }
  return project(w.out, gated);
}

// Inputs of one token of one lane, uniform in [-1, 1).
std::vector<uint16_t> token_input(uint32_t lane, int position) {
  std::vector<uint16_t> v(H);
  for (int i = 0; i < H; ++i) {
    v[i] = f32_to_bf16(hash_uniform(2000 + lane, static_cast<uint64_t>(position) * H + i, 1.0F));
  }
  return v;
}

std::vector<double> as_double(const std::vector<uint16_t> &v) {
  std::vector<double> d(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) d[i] = bf16_to_f32(v[i]);
  return d;
}

// Output check of one call: rows of y (BF16) against the fp64 rows.
void check_output(const std::string &name, const std::vector<uint16_t> &y, const std::vector<std::vector<double>> &ref) {
  double num = 0.0, den = 0.0, worst = 0.0, peak = 0.0;
  for (std::size_t r = 0; r < ref.size(); ++r) {
    for (int i = 0; i < H; ++i) {
      const double e = bf16_to_f32(y[r * H + i]) - ref[r][i];
      num += e * e;
      den += ref[r][i] * ref[r][i];
      worst = std::max(worst, std::fabs(e));
      peak = std::max(peak, std::fabs(ref[r][i]));
    }
  }
  const double rel = std::sqrt(num / den);
  std::printf("  %s: %zu rows, relative L2 %.3e, worst %.3e of max %.3e\n", name.c_str(), ref.size(), rel, worst, peak);
  check(rel <= std::ldexp(1.0, -7), name + ": relative L2 within 2^-7");
  check(worst <= std::ldexp(peak, -6), name + ": every element within 2^-6 of the largest reference");
}

struct Device {
  DeviceBytes conv{static_cast<std::size_t>(kSlots) * 3 * C * 2};
  DeviceBytes recurrent{static_cast<std::size_t>(kSlots) * VH * D * D * 4};
  fn::gdn::State state() const {
    fn::gdn::State s;
    s.conv = conv.p;
    s.recurrent = recurrent.as<float>();
    s.slots = kSlots;
    return s;
  }
};

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = H;
  g.gdn_qk_heads = fn::gdn::kQkHeads;
  g.gdn_value_heads = VH;
  g.gdn_head_dim = D;
  g.gdn_conv_kernel = 4;
  g.rms_norm_eps = kEps;
  return g;
}

}  // namespace

int main() {
  for (int c = 0; c < 256; ++c) g_lut[c] = e4m3_to_f32(static_cast<uint8_t>(c));
  MOE_CUDA(cudaSetDevice(0));
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return EXIT_FAILURE;
  }
  const fn::Geometry g = geometry();
  check(fn::gdn::check_geometry(g) == nullptr, "the real geometry is accepted");
  {
    fn::Geometry other = g;
    other.gdn_value_heads = 32;
    check(fn::gdn::check_geometry(other) != nullptr, "another value-head count is refused");
  }

  Weights w;
  build(w);
  const fn::GdnWeights wv = w.view();
  Device dev;
  // Slots 1 and 2 are other sequences' state: a pattern no call may touch.
  MOE_CUDA(cudaMemset(dev.conv.p, 0, dev.conv.bytes));
  MOE_CUDA(cudaMemset(dev.recurrent.p, 0, dev.recurrent.bytes));
  for (int s : {1, 2}) {
    MOE_CUDA(cudaMemset(static_cast<char *>(dev.conv.p) + static_cast<std::size_t>(s) * 3 * C * 2, 0x3C, 3 * C * 2));
    MOE_CUDA(cudaMemset(static_cast<char *>(dev.recurrent.p) + static_cast<std::size_t>(s) * VH * D * D * 4, 0x3C,
                        static_cast<std::size_t>(VH) * D * D * 4));
  }

  constexpr int kMaxRows = 70;
  ninfer::DeviceArena arena(fn::fn_gdn_layer_scratch_bytes(g, kMaxRows));
  DeviceBytes d_x(static_cast<std::size_t>(kMaxRows) * H * 2), d_y(static_cast<std::size_t>(kMaxRows) * H * 2);
  DeviceBytes d_slots(16 * 4);
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const int slot_of[3] = {3, 0, 4};
  Lane lanes[3];
  int position[3] = {0, 0, 0};

  // One call: `lanes_in_call` lanes (all three in decode) of `tokens` tokens each.
  auto call = [&](const std::string &name, const std::vector<int> &lane_ids, int tokens, cudaGraphExec_t *graph) {
    const int rows = static_cast<int>(lane_ids.size()) * tokens;
    std::vector<uint16_t> x(static_cast<std::size_t>(rows) * H);
    std::vector<std::vector<double>> ref;
    std::vector<int32_t> slots;
    for (std::size_t l = 0; l < lane_ids.size(); ++l) {
      const int id = lane_ids[l];
      slots.push_back(slot_of[id]);
      for (int t = 0; t < tokens; ++t) {
        const std::vector<uint16_t> xt = token_input(static_cast<uint32_t>(id), position[id] + t);
        std::copy(xt.begin(), xt.end(), x.begin() + (l * tokens + t) * static_cast<std::size_t>(H));
        ref.push_back(reference_token(w, lanes[id], as_double(xt)));
      }
      position[id] += tokens;
    }
    upload(d_x, x);
    upload(d_slots, slots);
    fn::Batch batch;
    batch.lanes = static_cast<int32_t>(lane_ids.size());
    batch.tokens = tokens;
    batch.slots = d_slots.as<int32_t>();
    if (graph != nullptr) {
      MOE_CUDA(cudaGraphLaunch(*graph, stream));
    } else {
      arena.reset_peak();
      FN_RC(fn::gdn::run(g, dev.state(), wv, batch, d_x.p, d_y.p, arena, stream));
    }
    MOE_CUDA(cudaStreamSynchronize(stream));
    if (graph == nullptr) {
      check(arena.peak_used() <= fn::fn_gdn_layer_scratch_bytes(g, rows),
            name + ": scratch peak " + std::to_string(arena.peak_used()) + " within fn_gdn_layer_scratch_bytes");
    }
    check_output(name, download<uint16_t>(d_y.p, static_cast<std::size_t>(rows) * H), ref);
  };

  call("prefill lane A, 70 tokens (one 64-token chunk + tail)", {0}, 70, nullptr);
  call("prefill lane B, 5 tokens", {1}, 5, nullptr);
  call("prefill lane C, 1 token", {2}, 1, nullptr);
  call("decode round 1, 3 lanes, eager", {0, 1, 2}, 1, nullptr);

  // Rounds 2 and 3 replay one captured graph; only x changes between them (the slots too could:
  // they are read on the device).
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  {
    fn::Batch batch;
    batch.lanes = 3;
    batch.tokens = 1;
    batch.slots = d_slots.as<int32_t>();
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int32_t rc = fn::gdn::run(g, dev.state(), wv, batch, d_x.p, d_y.p, arena, stream);
    MOE_CUDA(cudaStreamEndCapture(stream, &graph));
    FN_RC(rc);
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  }
  call("decode round 2, graph replay", {0, 1, 2}, 1, &exec);
  call("decode round 3, graph replay", {0, 1, 2}, 1, &exec);
  MOE_CUDA(cudaGraphExecDestroy(exec));
  MOE_CUDA(cudaGraphDestroy(graph));
  call("prefill lane A again, 20 tokens on a carried state", {0}, 20, nullptr);

  // Final states: the lanes' against fp64, the other slots untouched.
  const std::vector<uint16_t> conv = download<uint16_t>(dev.conv.p, static_cast<std::size_t>(kSlots) * 3 * C);
  const std::vector<float> rec = download<float>(dev.recurrent.p, static_cast<std::size_t>(kSlots) * VH * D * D);
  for (int id = 0; id < 3; ++id) {
    const std::size_t cb = static_cast<std::size_t>(slot_of[id]) * 3 * C;
    int exact = 0, within = 0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(3) * C; ++i) {
      const double got = bf16_to_f32(conv[cb + i]), want = lanes[id].taps[i];
      exact += got == want;
      within += std::fabs(got - want) <= std::ldexp(std::fabs(want), -7);
    }
    check(within == 3 * C, "lane " + std::to_string(id) + ": conv taps within one BF16 ulp");
    check(exact >= 3 * C * 99 / 100, "lane " + std::to_string(id) + ": conv taps 99% exact");
    const std::size_t rb = static_cast<std::size_t>(slot_of[id]) * VH * D * D;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(VH) * D * D; ++i) {
      const double e = rec[rb + i] - lanes[id].state[i];
      num += e * e;
      den += lanes[id].state[i] * lanes[id].state[i];
    }
    const double rel = std::sqrt(num / den);
    std::printf("  lane %d (slot %d): taps %d/%d exact, state relative L2 %.3e\n", id, slot_of[id], exact, 3 * C, rel);
    check(rel <= std::ldexp(1.0, -7), "lane " + std::to_string(id) + ": recurrent state within relative L2 2^-7");
  }
  for (int s : {1, 2}) {
    bool same = true;
    const auto *cb = reinterpret_cast<const uint8_t *>(&conv[static_cast<std::size_t>(s) * 3 * C]);
    for (std::size_t i = 0; i < static_cast<std::size_t>(3) * C * 2; ++i) same = same && cb[i] == 0x3C;
    const auto *rb = reinterpret_cast<const uint8_t *>(&rec[static_cast<std::size_t>(s) * VH * D * D]);
    for (std::size_t i = 0; i < static_cast<std::size_t>(VH) * D * D * 4; ++i) same = same && rb[i] == 0x3C;
    check(same, "slot " + std::to_string(s) + " (no lane's) is untouched");
  }

  // Refusals by name.
  {
    fn::Batch bad;
    bad.lanes = 2;
    bad.tokens = 4;
    bad.slots = d_slots.as<int32_t>();
    check(fn::gdn::run(g, dev.state(), wv, bad, d_x.p, d_y.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "one lane of several") != nullptr,
          "two lanes of several tokens are refused by name");
    fn::GdnWeights short_w = wv;
    short_w.in_proj_z.rows = 4096;
    bad.lanes = 1;
    check(fn::gdn::run(g, dev.state(), short_w, bad, d_x.p, d_y.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "wrong shape") != nullptr,
          "a wrongly shaped weight is refused by name");
  }
  MOE_CUDA(cudaStreamDestroy(stream));

  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next gdn test: %d check(s) FAILED\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash_next gdn test: ok\n");
  return EXIT_SUCCESS;
}
