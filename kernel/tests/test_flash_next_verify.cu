// GitHub #307 (spec flash-next/07 phase C): OURS -- the verify round's device side at the real
// geometry: the GDN half, the MTP head's restore and the refusals of a call its state cannot hold.
//
// fold (hidden 2560, 16 / 48 heads of 128, conv 4, FP8 row-scale projections, random weights): a
// verify call of k + 1 columns per lane (gdn::run with Batch::verify, the vendored replay record)
// and our fold of the first c records, against c one-token rounds over the same columns. Two lanes
// in a four-slot pool, each given history by a one-lane call first; then the verify call over four
// columns per lane, committed at c = 1 and 3 (and, separately, both lanes at c = 2 replayed from a
// captured graph with the counts read on the device). Checked, bit for bit:
// - the verify call leaves every slot's recurrent state and conv taps as they were (it records);
// - after the fold each lane's recurrent state and conv taps equal those of c one-token rounds
//   over the same columns -- the state the next step reads -- and slot 1, no lane's, is untouched;
// - the verify call's outputs of the committed columns equal the one-token rounds' outputs.
// The one-token rounds take the same columns' x through the same projections at another row
// count, so an FP8 linear whose result depended on its row count would show here, before the
// whole-model test (crates/core/tests/flash_next_speculative_gpu.rs) ever ran.
//
// restore_head: an MTP load's two attention sections (the trunk's, then the head's) in a
// three-slot pool, two lanes at windows k = 2 and 3, each lane committing c = 1 or k (both ways
// round, then every column, k + 1, beside 1), BF16 and hq-e8-2b. The pass's save, then every section overwritten (what the pass and the
// head's alignment write), then restore_head. Checked, bit for bit, against the state the save saw
// and the recorded indexer keys: the head's indexer tail of each lane's new frontier p + c, its
// ring words (the saved ones plus the committed positions' bits) and the side rows of the rejected
// columns p + c .. p + k; every other byte -- the head's committed columns, the trunk's section,
// the n-gram conv, slot 1 -- as the overwrite left it. The lanes' frontiers differ mod 4 (the tail
// takes saved and recorded keys) and one lane's columns wrap the ring.
//
// refusals: a call wider than its state's records (3 lanes x 4 columns against 8 rows, on a
// three-lane load and on a two-lane one), more lanes than the load's, or a window past its widest is
// refused by name before anything is written -- by every verify step and by the GDN layer's record
// call.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next/gdn.h"
#include "flash_next/verify.h"

#include "flash_next_s2_test_common.h"
#include "ignis_fp8_linear.h"

#include "core/arena.h"
#include "ninfer/ops/gqa_attention.h"

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

// ---- restore_head -----------------------------------------------------------------------------

constexpr int kAttention = 2;  // attention section 0 the trunk's, 1 the MTP head's
constexpr int kHead = kAttention - 1;
constexpr int kHeadSlots = 3;  // slots 2 and 0 the lanes', 1 no lane's
constexpr int kKeyDim = 128;   // the indexer's key dim
constexpr int kCompress = 4;
constexpr int kTailElements = (kCompress - 1) * kKeyDim;
constexpr int kSink = static_cast<int>(ninfer::ops::kGqaHqSinkKeys);
constexpr int kRecent = static_cast<int>(ninfer::ops::kGqaHqRecentKeys);
constexpr int kRingWords = kRecent / 32;
constexpr int kKvHeads = 2, kRowDim = 256;
constexpr std::size_t kSideSlotElements = static_cast<std::size_t>(kSink + kRecent) * kKvHeads * kRowDim;
// Frontiers 1 and 2 mod 4, the second's columns across the ring's end (4094 & 511 = 510).
constexpr int32_t kHeadPositions[kLanes] = {1001, 4094};

// The side row `position` names in a slot's plane: its sink row, or its ring slot (the vendored
// hq_residual_row's addressing).
std::size_t side_row(int slot, int position, int head) {
  const int row = position < kSink ? position : kSink + (position & (kRecent - 1));
  return slot * kSideSlotElements + (static_cast<std::size_t>(row) * kKvHeads + head) * kRowDim;
}

// One state of the sections restore_head and the save read, host side.
struct HeadSections {
  std::vector<uint16_t> tails[kAttention];            // [slot][compress - 1][key dim]
  std::vector<uint16_t> side_k[kAttention], side_v[kAttention];  // [slot][sink + recent][kv head][256]
  std::vector<uint32_t> ring;                         // [slot][ring words]
  std::vector<uint16_t> ngram;                        // [slot][columns][channels]
};

HeadSections head_sections(uint32_t stream, const fn::Geometry &g) {
  HeadSections h;
  for (int a = 0; a < kAttention; ++a) {
    h.tails[a] = bf16_vector(stream + 10 * a, static_cast<std::size_t>(kHeadSlots) * kTailElements, -2.0, 2.0);
    h.side_k[a] = bf16_vector(stream + 10 * a + 1, kHeadSlots * kSideSlotElements, -2.0, 2.0);
    h.side_v[a] = bf16_vector(stream + 10 * a + 2, kHeadSlots * kSideSlotElements, -2.0, 2.0);
  }
  h.ring.resize(static_cast<std::size_t>(kHeadSlots) * kRingWords);
  for (std::size_t i = 0; i < h.ring.size(); ++i) h.ring[i] = hash_u32(stream + 7, i);
  h.ngram = bf16_vector(stream + 8,
                        static_cast<std::size_t>(kHeadSlots) * g.ngram_conv_state_columns() * g.residual_width(),
                        -1.0, 1.0);
  return h;
}

struct HeadPool {
  std::unique_ptr<DeviceBytes> tails[kAttention], side_k[kAttention], side_v[kAttention], ring, ngram;
  fn::verify::Sections sections;

  HeadPool(const fn::Geometry &g, bool hq) {
    const HeadSections shape = head_sections(0, g);
    sections.attention_layers = kAttention;
    sections.tail_slot_elements = kTailElements;
    for (int a = 0; a < kAttention; ++a) {
      tails[a] = device_copy(shape.tails[a]);
      sections.tails[a] = tails[a]->p;
      if (hq) {
        side_k[a] = device_copy(shape.side_k[a]);
        side_v[a] = device_copy(shape.side_v[a]);
        sections.residual_k[a] = side_k[a]->p;
        sections.residual_v[a] = side_v[a]->p;
      }
    }
    if (hq) {
      ring = std::make_unique<DeviceBytes>(shape.ring.size() * 4);
      sections.ring = ring->as<uint32_t>();
    }
    ngram = device_copy(shape.ngram);
    sections.ngram_conv = ngram->p;
    sections.ngram_columns = g.ngram_conv_state_columns();
    sections.ngram_channels = g.residual_width();
  }

  void write(const HeadSections &h) {
    for (int a = 0; a < kAttention; ++a) {
      upload(*tails[a], h.tails[a]);
      if (ring) {
        upload(*side_k[a], h.side_k[a]);
        upload(*side_v[a], h.side_v[a]);
      }
    }
    if (ring) upload(*ring, h.ring);
    upload(*ngram, h.ngram);
  }

  HeadSections read() const {
    HeadSections h;
    for (int a = 0; a < kAttention; ++a) {
      h.tails[a] = download<uint16_t>(tails[a]->p, tails[a]->bytes / 2);
      if (ring) {
        h.side_k[a] = download<uint16_t>(side_k[a]->p, side_k[a]->bytes / 2);
        h.side_v[a] = download<uint16_t>(side_v[a]->p, side_v[a]->bytes / 2);
      }
    }
    if (ring) h.ring = download<uint32_t>(ring->p, ring->bytes / 4);
    h.ngram = download<uint16_t>(ngram->p, ngram->bytes / 2);
    return h;
  }
};

// One round at window k: the save over `pre`, every section overwritten with `post`, then
// restore_head with each lane's commit count; against the state it must leave.
void restore_head_arm(const fn::Geometry &g, bool hq, uint32_t k, const std::vector<int32_t> &commit,
                      cudaStream_t stream) {
  const std::string name = std::string("restore_head ") + (hq ? "hq-e8-2b" : "BF16") + " k = " + std::to_string(k) +
                           " c = {" + std::to_string(commit[0]) + ", " + std::to_string(commit[1]) + "}";
  std::string error;
  auto state = fn::verify::create(g, hq ? IGNIS_KV_FORMAT_HQ_E8_2B : IGNIS_KV_FORMAT_BF16, kLanes, k, 0, kAttention,
                                  1, true, &error);
  check(state != nullptr, name + ": verify::create: " + error);
  if (state == nullptr) return;
  check(state->window(kLanes) == k, name + ": two lanes run at the load's window");
  HeadPool pool(g, hq);
  const HeadSections pre = head_sections(1000 + 100 * k, g), post = head_sections(2000 + 100 * k, g);
  const int32_t slot_of[kLanes] = {2, 0};
  DeviceBytes d_slots(kLanes * 4), d_positions(kLanes * 4);
  upload(d_slots, std::vector<int32_t>(slot_of, slot_of + kLanes));
  upload(d_positions, std::vector<int32_t>(kHeadPositions, kHeadPositions + kLanes));

  pool.write(pre);
  check(fn::verify::save(*state, pool.sections, g, kLanes, k, d_slots.as<int32_t>(), d_positions.as<int32_t>(), stream,
                         &error) == 0,
        name + ": the save launches: " + error);
  MOE_CUDA(cudaStreamSynchronize(stream));
  pool.write(post);
  // The round's recorded indexer keys, every section's: [lanes][k + 1][key dim] each.
  const std::size_t key_elements = static_cast<std::size_t>(kLanes) * (k + 1) * kKeyDim;
  std::vector<uint16_t> keys[kAttention];
  for (int a = 0; a < kAttention; ++a) {
    keys[a] = bf16_vector(3000 + 100 * k + a, key_elements, -2.0, 2.0);
    MOE_CUDA(cudaMemcpy(static_cast<char *>(state->records.indexer_keys) + a * state->records.indexer_layer_bytes,
                        keys[a].data(), key_elements * 2, cudaMemcpyHostToDevice));
  }
  upload_to(*state->commit, commit);
  check(fn::verify::restore_head(*state, pool.sections, g, kLanes, k, d_slots.as<int32_t>(), d_positions.as<int32_t>(),
                                 stream, &error) == 0,
        name + ": restore_head launches: " + error);
  MOE_CUDA(cudaStreamSynchronize(stream));
  const HeadSections got = pool.read();

  HeadSections want = post;
  for (int lane = 0; lane < kLanes; ++lane) {
    const int slot = slot_of[lane];
    const int32_t p = kHeadPositions[lane], c = commit[lane], f = p + c;
    const int32_t first_new = f / kCompress * kCompress, first_old = p / kCompress * kCompress;
    for (int32_t q = first_new; q < f; ++q) {
      for (int d = 0; d < kKeyDim; ++d) {
        want.tails[kHead][static_cast<std::size_t>(slot) * kTailElements + (q - first_new) * kKeyDim + d] =
            q < p ? pre.tails[kHead][static_cast<std::size_t>(slot) * kTailElements + (q - first_old) * kKeyDim + d]
                  : keys[kHead][(static_cast<std::size_t>(lane) * (k + 1) + (q - p)) * kKeyDim + d];
      }
    }
    if (!hq) continue;
    for (int w = 0; w < kRingWords; ++w) want.ring[slot * kRingWords + w] = pre.ring[slot * kRingWords + w];
    for (int32_t q = p; q < f; ++q) {
      if (q < kSink) continue;
      const int r = q & (kRecent - 1);
      want.ring[slot * kRingWords + r / 32] |= 1U << (r % 32);
    }
    for (int32_t j = c; j <= static_cast<int32_t>(k); ++j) {
      for (int head = 0; head < kKvHeads; ++head) {
        const std::size_t at = side_row(slot, p + j, head);
        std::copy_n(&pre.side_k[kHead][at], kRowDim, &want.side_k[kHead][at]);
        std::copy_n(&pre.side_v[kHead][at], kRowDim, &want.side_v[kHead][at]);
      }
    }
  }
  check(got.tails[kHead] == want.tails[kHead],
        name + ": the head's indexer tail is the new frontier's -- saved keys below p, recorded ones from p");
  check(got.tails[0] == want.tails[0], name + ": the trunk's indexer tail is left alone");
  check(got.ngram == want.ngram, name + ": the n-gram conv is left alone");
  if (hq) {
    check(got.ring == want.ring, name + ": the ring words are the saved ones plus the committed positions' bits");
    check(got.side_k[kHead] == want.side_k[kHead] && got.side_v[kHead] == want.side_v[kHead],
          name + ": the head's side rows of the rejected columns are back, the committed ones left");
    check(got.side_k[0] == want.side_k[0] && got.side_v[0] == want.side_v[0],
          name + ": the trunk's side rows are left alone");
  }
  // A call past the load's lanes is refused before it writes anything.
  check(fn::verify::restore_head(*state, pool.sections, g, kLanes + 1, k, d_slots.as<int32_t>(),
                                 d_positions.as<int32_t>(), stream, &error) != 0 &&
            error.find("columns overrun this load's round") != std::string::npos,
        name + ": restore_head refuses three lanes on a two-lane load by name");
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
    const std::vector<int32_t> commit = {1, 3};
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
    const std::vector<int32_t> commit = {2, 2};
    pool.restore(conv0, state0);
    const auto y = verify(commit, &exec);
    compare("graph", commit, y, pool.conv_bytes(), pool.state_bytes());
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
  }

  // Refusals: a call the state was not sized for, by name, before it writes anything. Three lanes
  // of four columns are 12 rows against the records' 8; one lane at a window of 7 fits the rows but
  // not the widest window's saved ring rows and drafts (3); a window of 0 drafts nothing.
  {
    pool.restore(conv0, state0);
    upload(d_slots, std::vector<int32_t>{2, 0, 3});
    fn::Batch wide;
    wide.lanes = 3;
    wide.tokens = kColumns;
    wide.slots = d_slots.as<int32_t>();
    wide.verify = &state->records;
    check(fn::gdn::run(g, pool.state(), wv, wide, d_x.p, d_y.p, arena, stream) != 0 &&
              std::string(fn::fn_last_error()).find("8 rows (got 3 x 4)") != std::string::npos,
          "gdn: a verify call of 3 lanes x 4 columns on 8 record rows is refused by name");
    const auto refused = [&](const std::string &what, int32_t rc) {
      check(rc != 0 && error.find("columns overrun this load's round") != std::string::npos,
            what + " is refused by name (got \"" + error + "\")");
      error.clear();
    };
    const int32_t *slots = d_slots.as<int32_t>();
    for (const auto &[lanes, window] : {std::pair<uint32_t, uint32_t>{3, 3}, {1, 7}, {kLanes, 0}}) {
      const std::string call = std::to_string(lanes) + " lanes at a window of " + std::to_string(window);
      refused("save of " + call, fn::verify::save(*state, sections, g, lanes, window, slots, slots, stream, &error));
      refused("accept of " + call, fn::verify::accept(*state, g, lanes, window, d_y.p, d_y.p, stream, &error));
      refused("fold of " + call, fn::verify::fold(*state, sections, lanes, window, slots, stream, &error));
      refused("restore of " + call, fn::verify::restore(*state, sections, g, lanes, window, slots, slots, stream, &error));
    }
    // On a three-lane load three lanes are the load's and a window of 3 its widest (one lane's):
    // 3 x 4 columns are refused on the rows alone.
    auto three = fn::verify::create(g, IGNIS_KV_FORMAT_BF16, 3, kColumns - 1, 0, 1, 1, false, &error);
    check(three != nullptr && three->window(1) == kColumns - 1 && three->window(3) == 1,
          "a three-lane load drafts 3 at one lane and 1 at three: " + error);
    if (three != nullptr) {
      refused("fold of 3 lanes at a window of 3 on a three-lane load",
              fn::verify::fold(*three, sections, 3, kColumns - 1, slots, stream, &error));
      refused("save of 3 lanes at a window of 3 on a three-lane load",
              fn::verify::save(*three, sections, g, 3, kColumns - 1, slots, slots, stream, &error));
    }
    MOE_CUDA(cudaStreamSynchronize(stream));
    check(pool.conv_bytes() == conv0 && pool.state_bytes() == state0, "a refused call writes nothing");
  }

  for (const bool hq : {false, true}) {
    for (const uint32_t k : {2U, 3U}) {
      const auto c = static_cast<int32_t>(k);
      restore_head_arm(g, hq, k, {1, c}, stream);
      restore_head_arm(g, hq, k, {c, 1}, stream);
      restore_head_arm(g, hq, k, {c + 1, 1}, stream);
    }
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next verify test: %d check(s) FAILED\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash_next verify test: PASS\n");
  return EXIT_SUCCESS;
}
