// GitHub #307 (spec flash-next/07 phase C): OURS -- the verify round's GDN half at the real geometry
// (hidden 2560, 16 / 48 heads of 128, conv 4, FP8 row-scale projections, random weights): a verify
// call of k + 1 columns per lane (gdn::run with Batch::verify, the vendored replay record) and our
// fold of the first c records, against c one-token rounds over the same columns.
//
// Three lanes in a four-slot pool, each given history by a one-lane call first; then the verify
// call over four columns per lane, committed at c = 1, 3 and 4 (and, separately, every lane at
// c = 2 replayed from a captured graph with the counts read on the device). Checked, bit for bit:
// - the verify call leaves every slot's recurrent state and conv taps as they were (it records);
// - after the fold each lane's recurrent state and conv taps equal those of c one-token rounds
//   over the same columns -- the state the next step reads -- and slot 1, no lane's, is untouched;
// - the verify call's outputs of the committed columns equal the one-token rounds' outputs.
// The one-token rounds take the same columns' x through the same projections at another row
// count, so an FP8 linear whose result depended on its row count would show here, before the
// whole-model test (crates/core/tests/flash_next_speculative_gpu.rs) ever ran.

#include "flash_next/gdn.h"
#include "flash_next/verify.h"

#include "flash_next_s2_test_common.h"
#include "ignis_fp8_linear.h"

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
constexpr int VW = fn::gdn::kValueWidth;
constexpr int VH = fn::gdn::kValueHeads;
constexpr int D = fn::gdn::kHeadDim;
constexpr int kSlots = 4;
// Two lanes of kColumns are 8 rows: the widest round the default row budget (8) allows at a full
// window, and so the rows the verify state's records are sized for.
constexpr int kLanes = 2;
constexpr int kColumns = 4;  // a window of 3 drafts
constexpr std::size_t kConvSlotBytes = static_cast<std::size_t>(3) * C * 2;
constexpr std::size_t kStateSlotBytes = static_cast<std::size_t>(VH) * D * D * 4;

struct Weights {
  Proj qkv, z, a, b, out;
  std::unique_ptr<DeviceBytes> conv, a_log, dt_bias, norm;
  fn::GdnWeights view() const {
    fn::GdnWeights w;
    w.in_proj_qkv = linear(qkv);
    w.in_proj_z = linear(z);
    w.in_proj_a = linear(a);
    w.in_proj_b = linear(b);
    w.out_proj = linear(out);
    w.conv = conv->p;
    w.a_log = a_log->p;
    w.dt_bias = dt_bias->p;
    w.norm = norm->p;
    return w;
  }
};

void build(Weights &w) {
  const double x_rms = 1.0 / std::sqrt(3.0);
  make_proj(w.qkv, 201, C, H, x_rms, 1.0);
  make_proj(w.z, 203, VW, H, x_rms, 1.0);
  make_proj(w.a, 205, VH, H, x_rms, 0.5);
  make_proj(w.b, 207, VH, H, x_rms, 1.0);
  make_proj(w.out, 209, H, VW, 0.5, 1.0);
  w.conv = device_copy(bf16_vector(211, static_cast<std::size_t>(C) * 4, -0.5, 0.5));
  std::vector<uint16_t> a_log(VH);
  for (int h = 0; h < VH; ++h) a_log[h] = f32_to_bf16(std::log(1.0F + 7.5F * (hash_uniform(213, h, 1.0F) + 1.0F)));
  w.a_log = device_copy(a_log);
  w.dt_bias = device_copy(bf16_vector(215, VH, -5.0, -2.0));
  w.norm = device_copy(bf16_vector(217, D, 0.5, 1.5));
}

std::vector<uint16_t> token_input(uint32_t lane, int position) {
  std::vector<uint16_t> v(H);
  for (int i = 0; i < H; ++i) v[i] = f32_to_bf16(hash_uniform(3000 + lane, static_cast<uint64_t>(position) * H + i, 1.0F));
  return v;
}

// The verify round's geometry: the GDN layer's, plus what verify::create checks of the rest.
fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = H;
  g.gdn_qk_heads = fn::gdn::kQkHeads;
  g.gdn_value_heads = VH;
  g.gdn_head_dim = D;
  g.gdn_conv_kernel = 4;
  g.rms_norm_eps = 1e-6F;
  g.vocab = 248320;
  g.streams = 4;
  g.indexer_kv_heads = 1;
  g.indexer_head_dim = 128;
  g.compress_ratio = 4;
  g.ngram_size = 3;
  g.ngram_conv_kernel = 4;
  return g;
}

struct Pool {
  DeviceBytes conv{static_cast<std::size_t>(kSlots) * kConvSlotBytes};
  DeviceBytes recurrent{static_cast<std::size_t>(kSlots) * kStateSlotBytes};
  fn::gdn::State state() const {
    fn::gdn::State s;
    s.conv = conv.p;
    s.recurrent = recurrent.as<float>();
    s.slots = kSlots;
    return s;
  }
  std::vector<uint8_t> conv_bytes() const { return download<uint8_t>(conv.p, conv.bytes); }
  std::vector<uint8_t> state_bytes() const { return download<uint8_t>(recurrent.p, recurrent.bytes); }
  void restore(const std::vector<uint8_t> &c, const std::vector<uint8_t> &r) {
    MOE_CUDA(cudaMemcpy(conv.p, c.data(), c.size(), cudaMemcpyHostToDevice));
    MOE_CUDA(cudaMemcpy(recurrent.p, r.data(), r.size(), cudaMemcpyHostToDevice));
  }
};

void upload_to(ninfer::DeviceBuffer &buffer, const std::vector<int32_t> &v) {
  MOE_CUDA(cudaMemcpy(buffer.p, v.data(), v.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
}

bool same_slot(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, int slot, std::size_t slot_bytes) {
  return std::memcmp(a.data() + slot * slot_bytes, b.data() + slot * slot_bytes, slot_bytes) == 0;
}

}  // namespace

int main() {
  MOE_CUDA(cudaSetDevice(0));
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return EXIT_FAILURE;
  }
  const fn::Geometry g = geometry();
  Weights w;
  build(w);
  const fn::GdnWeights wv = w.view();
  Pool pool;
  MOE_CUDA(cudaMemset(pool.conv.p, 0, pool.conv.bytes));
  MOE_CUDA(cudaMemset(pool.recurrent.p, 0, pool.recurrent.bytes));
  // Slot 1 is no lane's: a pattern nothing may touch.
  MOE_CUDA(cudaMemset(static_cast<char *>(pool.conv.p) + kConvSlotBytes, 0x3C, kConvSlotBytes));
  MOE_CUDA(cudaMemset(static_cast<char *>(pool.recurrent.p) + kStateSlotBytes, 0x3C, kStateSlotBytes));

  std::string error;
  auto state = fn::verify::create(g, IGNIS_KV_FORMAT_BF16, kLanes, kColumns - 1, 0, 1, 1, false, &error);
  check(state != nullptr, "verify::create: " + error);
  if (state == nullptr) return EXIT_FAILURE;
  check(state->window(kLanes) == kColumns - 1 && state->window(1) == kColumns - 1,
        "the default row budget holds the full window of 3 drafts at one and two lanes");
  fn::verify::Sections sections;
  sections.gdn_layers = 1;
  sections.recurrent[0] = pool.recurrent.as<float>();
  sections.conv[0] = pool.conv.p;

  constexpr int kMaxRows = kLanes * kColumns;
  ninfer::DeviceArena arena(fn::fn_gdn_layer_scratch_bytes(g, 16));
  DeviceBytes d_x(static_cast<std::size_t>(16) * H * 2), d_y(static_cast<std::size_t>(kMaxRows) * H * 2);
  DeviceBytes d_slots(16 * 4);
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const int32_t slot_of[kLanes] = {2, 0};

  // History: five tokens per lane through a one-lane call each.
  for (int lane = 0; lane < kLanes; ++lane) {
    std::vector<uint16_t> x;
    for (int t = 0; t < 5; ++t) {
      const auto xt = token_input(static_cast<uint32_t>(lane), t);
      x.insert(x.end(), xt.begin(), xt.end());
    }
    upload(d_x, x);
    upload(d_slots, std::vector<int32_t>{slot_of[lane]});
    fn::Batch batch;
    batch.lanes = 1;
    batch.tokens = 5;
    batch.slots = d_slots.as<int32_t>();
    FN_RC(fn::gdn::run(g, pool.state(), wv, batch, d_x.p, d_y.p, arena, stream));
  }
  MOE_CUDA(cudaStreamSynchronize(stream));
  const std::vector<uint8_t> conv0 = pool.conv_bytes(), state0 = pool.state_bytes();

  // The verify call's columns: lane-major, column j of lane l the lane's token 5 + j.
  std::vector<uint16_t> columns(static_cast<std::size_t>(kMaxRows) * H);
  for (int lane = 0; lane < kLanes; ++lane) {
    for (int j = 0; j < kColumns; ++j) {
      const auto xt = token_input(static_cast<uint32_t>(lane), 5 + j);
      std::copy(xt.begin(), xt.end(), columns.begin() + (static_cast<std::size_t>(lane) * kColumns + j) * H);
    }
  }
  const auto column_of = [&](int lane, int j) {
    return std::vector<uint16_t>(columns.begin() + (static_cast<std::size_t>(lane) * kColumns + j) * H,
                                 columns.begin() + (static_cast<std::size_t>(lane) * kColumns + j + 1) * H);
  };

  // One verify call and its fold at `commit` per lane; returns its outputs [lanes * columns][H].
  auto verify = [&](const std::vector<int32_t> &commit, cudaGraphExec_t *graph) {
    upload(d_x, columns);
    upload(d_slots, std::vector<int32_t>(slot_of, slot_of + kLanes));
    upload_to(*state->valid_columns, std::vector<int32_t>(kLanes, kColumns));
    upload_to(*state->commit, commit);
    fn::Batch batch;
    batch.lanes = kLanes;
    batch.tokens = kColumns;
    batch.slots = d_slots.as<int32_t>();
    batch.verify = &state->records;
    if (graph != nullptr) {
      MOE_CUDA(cudaGraphLaunch(*graph, stream));
    } else {
      // The pass alone first: it must leave every slot as it was.
      FN_RC(fn::gdn::run(g, pool.state(), wv, batch, d_x.p, d_y.p, arena, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      check(pool.conv_bytes() == conv0 && pool.state_bytes() == state0,
            "the verify call leaves every slot's taps and recurrent state as they were");
      check(fn::verify::fold(*state, sections, kLanes, kColumns - 1, d_slots.as<int32_t>(), stream, &error) == 0,
            "the fold launches: " + error);
    }
    MOE_CUDA(cudaStreamSynchronize(stream));
    return download<uint16_t>(d_y.p, static_cast<std::size_t>(kMaxRows) * H);
  };

  // The same columns through one-token rounds: round j runs the lanes with commit > j.
  auto rounds = [&](const std::vector<int32_t> &commit) {
    std::vector<std::vector<uint16_t>> outputs(static_cast<std::size_t>(kMaxRows));
    for (int j = 0; j < kColumns; ++j) {
      std::vector<int32_t> slots;
      std::vector<uint16_t> x;
      std::vector<int> lanes;
      for (int lane = 0; lane < kLanes; ++lane) {
        if (commit[lane] <= j) continue;
        slots.push_back(slot_of[lane]);
        const auto xt = column_of(lane, j);
        x.insert(x.end(), xt.begin(), xt.end());
        lanes.push_back(lane);
      }
      if (lanes.empty()) break;
      upload(d_x, x);
      upload(d_slots, slots);
      fn::Batch batch;
      batch.lanes = static_cast<int32_t>(lanes.size());
      batch.tokens = 1;
      batch.slots = d_slots.as<int32_t>();
      FN_RC(fn::gdn::run(g, pool.state(), wv, batch, d_x.p, d_y.p, arena, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      const auto y = download<uint16_t>(d_y.p, lanes.size() * H);
      for (std::size_t i = 0; i < lanes.size(); ++i) {
        outputs[static_cast<std::size_t>(lanes[i]) * kColumns + j].assign(y.begin() + i * H, y.begin() + (i + 1) * H);
      }
    }
    return outputs;
  };

  auto compare = [&](const std::string &name, const std::vector<int32_t> &commit, const std::vector<uint16_t> &y_verify,
                     const std::vector<uint8_t> &conv_v, const std::vector<uint8_t> &state_v) {
    pool.restore(conv0, state0);
    const auto y_rounds = rounds(commit);
    const auto conv_r = pool.conv_bytes(), state_r = pool.state_bytes();
    for (int lane = 0; lane < kLanes; ++lane) {
      const std::string at = name + " lane " + std::to_string(lane) + " (c = " + std::to_string(commit[lane]) + ")";
      check(same_slot(state_v, state_r, slot_of[lane], kStateSlotBytes),
            at + ": the folded recurrent state is c one-token rounds' bit for bit");
      check(same_slot(conv_v, conv_r, slot_of[lane], kConvSlotBytes), at + ": the folded conv taps are theirs");
      for (int j = 0; j < commit[lane]; ++j) {
        const std::size_t row = static_cast<std::size_t>(lane) * kColumns + j;
        check(std::equal(y_rounds[row].begin(), y_rounds[row].end(), y_verify.begin() + row * H),
              at + ": column " + std::to_string(j) + "'s output is the one-token round's");
      }
    }
    check(same_slot(state_v, state0, 1, kStateSlotBytes) && same_slot(conv_v, conv0, 1, kConvSlotBytes),
          name + ": slot 1, no lane's, is untouched");
  };

  {
    const std::vector<int32_t> commit = {1, 3, 4};
    pool.restore(conv0, state0);
    const auto y = verify(commit, nullptr);
    compare("eager", commit, y, pool.conv_bytes(), pool.state_bytes());
  }

  // The pass and the fold captured once and replayed: the counts are read on the device.
  {
    pool.restore(conv0, state0);
    upload(d_slots, std::vector<int32_t>(slot_of, slot_of + kLanes));
    upload_to(*state->valid_columns, std::vector<int32_t>(kLanes, kColumns));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    fn::Batch batch;
    batch.lanes = kLanes;
    batch.tokens = kColumns;
    batch.slots = d_slots.as<int32_t>();
    batch.verify = &state->records;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    FN_RC(fn::gdn::run(g, pool.state(), wv, batch, d_x.p, d_y.p, arena, stream));
    FN_RC(fn::verify::fold(*state, sections, kLanes, kColumns - 1, d_slots.as<int32_t>(), stream, &error));
    MOE_CUDA(cudaStreamEndCapture(stream, &graph));
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    const std::vector<int32_t> commit = {2, 2, 2};
    pool.restore(conv0, state0);
    const auto y = verify(commit, &exec);
    compare("graph", commit, y, pool.conv_bytes(), pool.state_bytes());
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next verify test: %d check(s) FAILED\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash_next verify test: PASS\n");
  return EXIT_SUCCESS;
}
