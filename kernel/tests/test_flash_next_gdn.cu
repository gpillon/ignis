// GitHub #302 (spec flash-next/04, slice S2): OURS -- the Flash-Next GDN layer core
// (kernel/src/flash_next/gdn.cu) at the real geometry (hidden 2560, 16 / 48 heads of 128, conv 4,
// FP8 row-scale projections) against an fp64 restatement of the checkpoint's
// Qwen4ExpTextGatedDeltaNet that rounds to BF16 where the checkpoint stores BF16.
//
// Three lanes in a five-slot state pool, as the program drives them: one-lane prefill calls (one
// crossing the recurrence's 64-token chunk and the convolution's 64-token chunk), a one-token
// prefill, a three-lane decode round run eagerly, two more captured once as a CUDA graph and
// replayed with new inputs and the lanes in another order (the slots are read on the device),
// then a second prefill chunk on a lane whose state is no longer zero. Every call's output, every
// lane's final state, the untouched slots and every call's scratch peak (against
// fn_gdn_layer_scratch_bytes) are checked. The convolution weight is the artifact's
// [channels][4] (conv1d.weight), random per tap, so a tap-major read cannot pass. Then
// fn_gdn_layer itself on a seq pool: its layer and slot, and its refusals. And (GitHub #306) the
// fused route -- one grouped projection launch, the gating inside the convolution's -- against
// the five launches, bit for bit, at decode and prefill shapes.
//
// Tolerance: the layer's output inherits the vendored recurrence's own criterion (relative L2
// 4.1e-3, gross 5.5e-3 of the largest reference; vendor/tests/ops/test_gated_delta_net.cpp),
// plus the BF16 output rounding (2^-9 relative) and the rare BF16 rounding flips of the
// intermediate stores. Checked: relative L2 per call <= 2^-7 and every element within 2^-6 of the
// call's largest reference; the fp32 state within relative L2 2^-7 of the fp64 one; the conv taps
// (the BF16 projections) within the rounding of the FP8 linear's fp32 accumulation bound, almost
// all of them exact.

#include "flash_next/fusion.h"
#include "flash_next/gdn.h"

#include "flash_next_s2_test_common.h"
#include "ignis_fp8_linear.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace s2_test;
namespace fn = ignis::flash_next;

namespace {

constexpr int H = 2560;
constexpr int C = fn::gdn::kConvChannels;
constexpr int KW = fn::gdn::kKeyWidth;
constexpr int VW = fn::gdn::kValueWidth;
constexpr int VH = fn::gdn::kValueHeads;
constexpr int D = fn::gdn::kHeadDim;
constexpr int kSlots = 5;
constexpr float kEps = 1e-6F;

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
  w.d_conv = device_copy(w.conv);
  w.d_a_log = device_copy(w.a_log);
  w.d_dt_bias = device_copy(w.dt_bias);
  w.d_norm = device_copy(w.norm);
}

// One lane of the fp64 restatement: its conv taps (the last three BF16 projected qkv rows) and
// its recurrent state, [VH][D (value)][D (key)].
struct Lane {
  std::vector<double> taps = std::vector<double>(static_cast<std::size_t>(3) * C, 0.0);
  std::vector<double> tap_bound = std::vector<double>(static_cast<std::size_t>(3) * C, 0.0);  // fp32 sum bound
  std::vector<double> state = std::vector<double>(static_cast<std::size_t>(VH) * D * D, 0.0);
};

// One token through the layer; returns y (fp64, unrounded).
std::vector<double> reference_token(const Weights &w, Lane &lane, const std::vector<double> &x) {
  std::vector<double> qkv_bound;
  std::vector<double> qkv = project(w.qkv, x, &qkv_bound), z = project(w.z, x), a = project(w.a, x), b = project(w.b, x);
  for (auto *v : {&qkv, &z, &a, &b}) {
    for (double &e : *v) e = bf16r(e);
  }
  // Depthwise causal conv over [taps..., qkv] with the weight [c][j], rounded to BF16 (F.conv1d in
  // BF16), then SiLU rounded again.
  std::vector<double> conv(C);
  for (int c = 0; c < C; ++c) {
    double s = 0.0;
    for (int j = 0; j < 3; ++j) s += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * 4 + j]) * lane.taps[static_cast<std::size_t>(j) * C + c];
    s += bf16_to_f32(w.conv[static_cast<std::size_t>(c) * 4 + 3]) * qkv[c];
    const double sb = bf16r(s);
    conv[c] = bf16r(sb * sigmoid(sb));
  }
  for (int j = 0; j < 2; ++j) {
    std::copy_n(&lane.taps[static_cast<std::size_t>(j + 1) * C], C, &lane.taps[static_cast<std::size_t>(j) * C]);
    std::copy_n(&lane.tap_bound[static_cast<std::size_t>(j + 1) * C], C, &lane.tap_bound[static_cast<std::size_t>(j) * C]);
  }
  std::copy_n(qkv.data(), C, &lane.taps[static_cast<std::size_t>(2) * C]);
  std::copy_n(qkv_bound.data(), C, &lane.tap_bound[static_cast<std::size_t>(2) * C]);

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

// The kernel nodes `launch()` captures on `stream`, or -1 when it fails.
template <class Launch>
int captured_kernels(cudaStream_t stream, Launch launch) {
  cudaGraph_t graph = nullptr;
  MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const int32_t rc = launch();
  MOE_CUDA(cudaStreamEndCapture(stream, &graph));
  std::size_t count = 0;
  MOE_CUDA(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  MOE_CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
  int kernels = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    MOE_CUDA(cudaGraphNodeGetType(node, &type));
    kernels += type == cudaGraphNodeTypeKernel ? 1 : 0;
  }
  MOE_CUDA(cudaGraphDestroy(graph));
  return rc == 0 ? kernels : -1;
}

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
        ref.push_back(reference_token(w, lanes[id], as_double(xt.data(), xt.size())));
      }
      position[id] += tokens;
    }
    upload(d_x, x);
    upload(d_slots, slots);
    MOE_CUDA(cudaDeviceSynchronize());  // pageable uploads may still be in flight for another stream
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

  // Rounds 2 and 3 replay one captured graph; round 3 takes the lanes in another order, so its
  // inputs and slots both change under the same launches.
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
  call("decode round 3, graph replay, lanes C A B", {2, 0, 1}, 1, &exec);
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
      // Both sides round to BF16 values within the FP8 linear's fp32 accumulation bound of the
      // same sum: they differ by at most that bound plus the two roundings (a near-zero tap may
      // round apart by more than one of its own ulps).
      within += std::fabs(got - want) <= lanes[id].tap_bound[i] + std::ldexp(std::fabs(got) + std::fabs(want), -8);
    }
    check(within == 3 * C, "lane " + std::to_string(id) + ": conv taps within the projection's rounding bound");
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

  // fn_gdn_layer on a seq pool of three GDN layers, one sequence: bit-identical to gdn::run on
  // buffers of its own, its state in layer 1 of the sequence's slot and nowhere else.
  {
    ignis_seq_pool_spec spec{};
    spec.num_kv_heads = 2;
    spec.head_dim = 256;
    spec.kv_format = IGNIS_KV_FORMAT_BF16;
    spec.kv_page_group_count = 4;
    spec.max_context_tokens = 256;
    spec.slot_count = 2;
    spec.gdn_num_layers = 3;
    spec.gdn_conv_channels = C;
    spec.gdn_value_heads = VH;
    spec.gdn_head_dim = D;
    spec.vocab = 1024;
    spec.kv_num_layers = 1;
    ignis_seq_pool *pool = nullptr;
    ignis_seq *seq = nullptr;
    if (ignis_seq_pool_create(&spec, &pool) != 0 || ignis_seq_alloc(pool, 64, &seq) != 0) {
      std::fprintf(stderr, "FATAL: seq pool: %s\n", ignis_seq_last_error());
      return EXIT_FAILURE;
    }
    fn::Context ctx;
    ctx.g = g;
    ctx.pool = pool;
    Device own;
    MOE_CUDA(cudaMemset(own.conv.p, 0, own.conv.bytes));
    MOE_CUDA(cudaMemset(own.recurrent.p, 0, own.recurrent.bytes));
    DeviceBytes d_y_own(d_y.bytes);
    bool same = true;
    for (int tokens : {6, 1}) {
      std::vector<uint16_t> x;
      for (int t = 0; t < tokens; ++t) {
        const std::vector<uint16_t> xt = token_input(7, 100 * tokens + t);
        x.insert(x.end(), xt.begin(), xt.end());
      }
      upload(d_x, x);
      upload(d_slots, std::vector<int32_t>{seq->slot});
      MOE_CUDA(cudaDeviceSynchronize());
      fn::Batch batch;
      batch.lanes = 1;
      batch.tokens = tokens;
      batch.slots = d_slots.as<int32_t>();
      FN_RC(fn::fn_gdn_layer(ctx, 1, wv, batch, d_x.p, d_y.p, arena, stream));
      FN_RC(fn::gdn::run(g, own.state(), wv, batch, d_x.p, d_y_own.p, arena, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      same = same && download<uint16_t>(d_y.p, static_cast<std::size_t>(tokens) * H) ==
                         download<uint16_t>(d_y_own.p, static_cast<std::size_t>(tokens) * H);
    }
    check(same, "fn_gdn_layer: prefill and decode outputs bit-identical to gdn::run");
    const std::size_t conv_bytes = static_cast<std::size_t>(3) * C * 2;
    const std::size_t rec_bytes = static_cast<std::size_t>(VH) * D * D * 4;
    auto bytes_at = [](const void *p, std::size_t n) { return download<uint8_t>(p, n); };
    check(bytes_at(pool->gdn_pool.conv_slot(1, seq->slot).data, conv_bytes) ==
                  bytes_at(static_cast<const char *>(own.conv.p) + seq->slot * conv_bytes, conv_bytes) &&
              bytes_at(pool->gdn_pool.recurrent_slot(1, seq->slot).data, rec_bytes) ==
                  bytes_at(static_cast<const char *>(own.recurrent.p) + seq->slot * rec_bytes, rec_bytes),
          "fn_gdn_layer: the state lands in GDN layer 1 at the sequence's slot");
    bool others_zero = true;
    for (uint32_t layer : {0U, 2U}) {
      for (uint8_t b : bytes_at(pool->gdn_pool.conv_slot(layer, seq->slot).data, conv_bytes)) others_zero = others_zero && b == 0;
      for (uint8_t b : bytes_at(pool->gdn_pool.recurrent_slot(layer, seq->slot).data, rec_bytes)) others_zero = others_zero && b == 0;
    }
    check(others_zero, "fn_gdn_layer: GDN layers 0 and 2 are untouched");
    fn::Batch one;
    one.lanes = 1;
    one.tokens = 1;
    one.slots = d_slots.as<int32_t>();
    check(fn::fn_gdn_layer(ctx, 3, wv, one, d_x.p, d_y.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "GDN layer 3") != nullptr,
          "fn_gdn_layer: a GDN ordinal past the pool's layers is refused by name");
    fn::Context no_pool;
    no_pool.g = g;
    check(fn::fn_gdn_layer(no_pool, 0, wv, one, d_x.p, d_y.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "no seq pool") != nullptr,
          "fn_gdn_layer: a context without a seq pool is refused by name");
    ignis_seq_release(pool, seq);
    ignis_seq_pool_free(pool);
  }

  // GitHub #306, step 4 (fusion.h's Gdn): the four projections in one grouped GEMV launch and the
  // gating inside the convolution's launch, against the five launches, bit for bit -- the output
  // and every slot's conv taps and recurrent state, each run from the same state over a poisoned
  // arena -- for 1, 3 and 5 decode lanes and one-lane prefills of 6 tokens (grouped) and 70 (past
  // the GEMV's 8 rows: only the gating moves). Fused, a call captures 4 fewer kernels (1 past 8 rows).
  {
    const std::size_t conv_bytes = dev.conv.bytes, rec_bytes = dev.recurrent.bytes;
    const std::vector<uint8_t> conv0 = download<uint8_t>(dev.conv.p, conv_bytes);
    const std::vector<uint8_t> rec0 = download<uint8_t>(dev.recurrent.p, rec_bytes);
    struct Shape {
      std::vector<int32_t> slots;
      int tokens;
    };
    const Shape shapes[] = {{{3}, 1}, {{3, 0, 4}, 1}, {{4, 3, 2, 1, 0}, 1}, {{0}, 6}, {{4}, 70}};
    for (const Shape &shape : shapes) {
      const int lanes_n = static_cast<int>(shape.slots.size());
      const int rows = lanes_n * shape.tokens;
      const std::string name = "fused GDN, " + std::to_string(lanes_n) + " lane(s) of " + std::to_string(shape.tokens);
      std::vector<uint16_t> x;
      for (int r = 0; r < rows; ++r) {
        const std::vector<uint16_t> xt = token_input(9, 1000 + r);
        x.insert(x.end(), xt.begin(), xt.end());
      }
      upload(d_x, x);
      upload(d_slots, shape.slots);
      MOE_CUDA(cudaDeviceSynchronize());
      fn::Batch batch;
      batch.lanes = lanes_n;
      batch.tokens = shape.tokens;
      batch.slots = d_slots.as<int32_t>();
      std::vector<uint16_t> y[2];
      std::vector<uint8_t> conv_after[2], rec_after[2];
      int kernels[2] = {};
      for (int fused = 0; fused < 2; ++fused) {
        fn::set_fused(fn::Fusion::Gdn, fused == 1);
        upload(dev.conv, conv0);
        upload(dev.recurrent, rec0);
        MOE_CUDA(cudaMemset(arena.base(), 0xFF, arena.capacity()));
        MOE_CUDA(cudaMemset(d_y.p, 0xFF, d_y.bytes));
        MOE_CUDA(cudaDeviceSynchronize());
        FN_RC(fn::gdn::run(g, dev.state(), wv, batch, d_x.p, d_y.p, arena, stream));
        MOE_CUDA(cudaStreamSynchronize(stream));
        y[fused] = download<uint16_t>(d_y.p, static_cast<std::size_t>(rows) * H);
        conv_after[fused] = download<uint8_t>(dev.conv.p, conv_bytes);
        rec_after[fused] = download<uint8_t>(dev.recurrent.p, rec_bytes);
        kernels[fused] = captured_kernels(stream, [&] { return fn::gdn::run(g, dev.state(), wv, batch, d_x.p, d_y.p, arena, stream); });
      }
      fn::set_fused(fn::Fusion::Gdn, true);
      const bool finite = std::all_of(y[0].begin(), y[0].end(), [](uint16_t v) { return (v & 0x7F80U) != 0x7F80U; });
      check(finite, name + ": the five-launch output is finite (the comparison has power)");
      check(y[1] == y[0], name + ": the output is the five-launch route's, bit for bit");
      check(conv_after[1] == conv_after[0] && conv_after[0] != conv0,
            name + ": the conv taps are the five-launch route's, bit for bit");
      check(rec_after[1] == rec_after[0] && rec_after[0] != rec0,
            name + ": the recurrent state is the five-launch route's, bit for bit");
      const int fewer = rows <= 8 ? 4 : 1;
      check(kernels[0] - kernels[1] == fewer, name + ": " + std::to_string(fewer) + " fewer kernels fused (got " +
                                                  std::to_string(kernels[0]) + " and " + std::to_string(kernels[1]) +
                                                  ")");
    }
    upload(dev.conv, conv0);
    upload(dev.recurrent, rec0);
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
