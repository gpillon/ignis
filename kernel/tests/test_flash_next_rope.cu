// ADR 0059 (spec rope-scaling/02, ACs 7-8): Flash-Next's two rope sites under a YaRN table --
// OURS. At factor 4 (theta 1e7, rotary 64 of each head, 262,144 original positions):
//
//   prepare   qsa::prepare (prepare_kernel: the norms and rope of the 24 query heads and 2 key
//             heads) at positions 0, 262,143, 262,144, 589,823 and 1,048,575;
//   queries   indexer::prepare_queries (rope_row on the 4 query heads) at the same positions;
//   keys      indexer::append_keys (rope_row on a pooled block key, at its block's first
//             position) for the blocks starting at 0, 262,140, 262,144, 589,820 and 1,048,572;
//   select    indexer::select over a 2,600-token prefill: every sparse row's 512 blocks;
//   sources   (GitHub #557) qsa::prepare at a span's three axes (pair i at axis i % 3, the
//             checkpoint's interleaved M-RoPE), at a lane's position plus its rope delta (decode
//             and verify after an image), and -- bit for bit the plain call -- at three equal
//             axes and at a delta of 0.
//
// Each against a host restatement of the checkpoint's arithmetic: the (1 + w) RMSNorm in fp32
// rounded once to BF16 -- exact here, since the inputs sit on a half-integer grid whose squares
// sum exactly in any order -- then phi = fp32(position) * inv_freq, c = round_bf16(cos(phi) * m)
// and the same for s with m the table's attention factor, and the BF16 rotation. CUDA's cosf and
// sinf are within 2 ulps of the correctly rounded value, so where a BF16 rounding boundary falls
// within those ulps of cos(phi) * m the restatement takes either side (counted and printed).
// Every value must match it; and the restatement under the linear table (attention factor 1) must
// differ from it on most rotary values, so a kernel without YaRN would fail there.
//
// select: every sparse row's blocks are the documented rule -- the 512 largest scores, the
// lowest blocks among equal ones -- over the queries and block keys restated above, scored in
// the kernel's own fp32 order (exact BF16 products summed dim by dim, relu summed head by head,
// times fp32 1/sqrt(128)); and at least one row selects other blocks than the linear table's
// restatement would.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next/indexer.h"
#include "flash_next/qsa.h"

#include "flash_next_s2_test_common.h"

#include "core/arena.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace s2_test;
namespace fn = ignis::flash_next;
namespace ix = ignis::flash_next::indexer;
namespace qsa = ignis::flash_next::qsa;

namespace {

constexpr float kEps = 1e-6F;
constexpr int kIxHd = ix::kHeadDim;
constexpr int kIxHeads = 4;
constexpr int kQk = (kIxHeads + 1) * kIxHd;
constexpr int kPositions = 5;
constexpr int32_t kRowPositions[kPositions] = {0, 262143, 262144, 589823, 1048575};
constexpr int32_t kBlockFirsts[kPositions] = {0, 262140, 262144, 589820, 1048572};
constexpr int kSelectTokens = 2600;
constexpr int kBlockTopk = 512;

#define IX_OK(expr)                                                                                \
  do {                                                                                             \
    const char *st_ = (expr);                                                                      \
    if (st_ != nullptr) {                                                                          \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, st_);                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = 2560;
  g.q_heads = qsa::kQHeads;
  g.kv_heads = qsa::kKvHeads;
  g.head_dim = qsa::kHeadDim;
  g.rotary_dim = 64;
  g.indexer_heads = kIxHeads;
  g.indexer_head_dim = kIxHd;
  g.indexer_kv_heads = 1;
  g.compress_ratio = 4;
  g.indexer_budget = 2048;
  g.rms_norm_eps = kEps;
  return g;
}

float round_bf16(float v) { return bf16_to_f32(f32_to_bf16(v)); }

// Values in {+-0.5, +-1, +-1.5, +-2}: their squares, and the squares of their 4-token means, sum
// exactly in fp32 whatever the order, so the kernels' norm is restated bit for bit.
std::vector<uint16_t> grid(uint32_t stream, std::size_t n) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const uint32_t h = hash_u32(stream, i);
    const float x = 0.5F * static_cast<float>(1U + (h & 3U));
    v[i] = f32_to_bf16(((h >> 2) & 1U) != 0 ? -x : x);
  }
  return v;
}

// (1 + w) norm weights on a 1/128 grid in [-0.25, 0.25].
std::vector<uint16_t> norm_weights(uint32_t stream, int n) {
  std::vector<uint16_t> w(n);
  for (int d = 0; d < n; ++d) {
    w[d] = f32_to_bf16(static_cast<float>(static_cast<int>(hash_u32(stream, d) % 65) - 32) / 128.0F);
  }
  return w;
}

// The kernels' (1 + w) RMSNorm of n values, each step in fp32, rounded once to BF16.
std::vector<float> norm(const std::vector<float> &x, const std::vector<uint16_t> &w) {
  const int n = static_cast<int>(x.size());
  float ss = 0.0F;
  for (float v : x) ss = std::fmaf(v, v, ss);
  const float r = 1.0F / std::sqrt(ss / static_cast<float>(n) + kEps);
  std::vector<float> out(n);
  for (int d = 0; d < n; ++d) out[d] = round_bf16((x[d] * r) * (1.0F + bf16_to_f32(w[d])));
  return out;
}

std::vector<float> widen(const uint16_t *x, int n) {
  std::vector<float> out(n);
  for (int d = 0; d < n; ++d) out[d] = bf16_to_f32(x[d]);
  return out;
}

// The BF16 values cos(phi) * m (or sin) may round to on the device: every fp32 within CUDA's
// 2-ulp bound of the correctly rounded value, times m.
std::vector<float> trig_candidates(double exact, float m) {
  float v = static_cast<float>(exact);
  for (int i = 0; i < 2; ++i) v = std::nextafter(v, -INFINITY);
  std::vector<float> out;
  for (int i = 0; i < 5; ++i, v = std::nextafter(v, INFINITY)) {
    const float b = round_bf16(v * m);
    if (std::find(out.begin(), out.end(), b) == out.end()) out.push_back(b);
  }
  return out;
}

// One normed row roped at `position`: per dim, the BF16 values the device may write.
struct Restated {
  std::vector<std::vector<uint16_t>> allowed;
  int ambiguous_pairs = 0;
};

// GitHub #557: a row roped at three axes' positions, pair i (dims i, i + 32) at axis i % 3 -- the
// checkpoint's interleaved M-RoPE (Qwen4ExpTextRotaryEmbedding's recomposition of mrope_section
// [11, 11, 10], checked against transformers 5.17: one frequency table, and YaRN's attention factor
// on every axis alike). A text row is the same position on all three.
Restated restate_axes(const std::vector<float> &normed, const ix::Rope &rope, const std::array<int32_t, 3> &axes) {
  Restated out;
  out.allowed.resize(normed.size());
  for (std::size_t d = 64; d < normed.size(); ++d) out.allowed[d] = {f32_to_bf16(normed[d])};
  for (int i = 0; i < 32; ++i) {
    const float phi = static_cast<float>(axes[i % 3]) * rope.inv_freq[i];
    const std::vector<float> cs = trig_candidates(std::cos(static_cast<double>(phi)), rope.scale);
    const std::vector<float> sn = trig_candidates(std::sin(static_cast<double>(phi)), rope.scale);
    out.ambiguous_pairs += cs.size() > 1 || sn.size() > 1;
    const float x1 = normed[i], x2 = normed[i + 32];
    auto add = [](std::vector<uint16_t> &set, float v) {
      const uint16_t b = f32_to_bf16(v);
      if (std::find(set.begin(), set.end(), b) == set.end()) set.push_back(b);
    };
    for (float c : cs) {
      for (float s : sn) {
        add(out.allowed[i], round_bf16(round_bf16(x1 * c) + round_bf16(-x2 * s)));
        add(out.allowed[i + 32], round_bf16(round_bf16(x2 * c) + round_bf16(x1 * s)));
      }
    }
  }
  return out;
}

Restated restate(const std::vector<float> &normed, const ix::Rope &rope, int32_t position) {
  return restate_axes(normed, rope, {position, position, position});
}

// What the device rows say against their restatements, over a whole arm.
struct Tally {
  long values = 0, mismatches = 0, rotary = 0, rotary_as_linear = 0, rotary_overlap = 0, ambiguous_pairs = 0;
  void add(const uint16_t *got, const Restated &want, const Restated &linear) {
    ambiguous_pairs += want.ambiguous_pairs;
    for (std::size_t d = 0; d < want.allowed.size(); ++d) {
      const auto &set = want.allowed[d];
      ++values;
      mismatches += std::find(set.begin(), set.end(), got[d]) == set.end();
      if (d < 64) {
        ++rotary;
        const auto &lin = linear.allowed[d];
        rotary_as_linear += std::find(lin.begin(), lin.end(), got[d]) != lin.end();
        rotary_overlap += std::any_of(set.begin(), set.end(),
                                      [&](uint16_t v) { return std::find(lin.begin(), lin.end(), v) != lin.end(); });
      }
    }
  }
  // `linear`: what the alternative restatement add() was handed is (GitHub #557's arms hand it
  // another position, not another table).
  void report(const std::string &arm, const std::string &linear = "the linear table's") const {
    std::printf("  %s: %ld values, %ld outside the restatement, %ld/%ld rotary values %s (%ld where the two "
                "restatements share a value), %ld ambiguous pairs\n",
                arm.c_str(), values, mismatches, rotary_as_linear, rotary, linear.c_str(), rotary_overlap,
                ambiguous_pairs);
    check(values > 0 && mismatches == 0, arm + ": every value is the restatement's");
    // With every value the restatement's, a value can be the alternative's only where the two
    // restatements share one: near angle zero (a low-frequency pair, or the small image coordinates
    // of GitHub #557's three-axis arm) bf16 cannot tell two angles apart. What an arm must show is
    // that its inputs tell the two apart on most rotary values, so that a kernel rotating at the
    // alternative fails the check above there.
    check(rotary_as_linear <= rotary_overlap, arm + ": the rotary values are " + linear + " only where both agree");
    check(rotary_overlap * 2 < rotary, arm + ": the inputs tell the restatement from " + linear);
  }
};

// The documented rule on a score row: the k largest, ties by lowest block index; ascending.
std::vector<int32_t> rule(const std::vector<float> &s, int k) {
  std::vector<int32_t> order(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) order[i] = static_cast<int32_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return s[a] > s[b]; });
  order.resize(k);
  std::sort(order.begin(), order.end());
  return order;
}

// The score kernel's arithmetic for one (row, block): per head the exact BF16 products summed in
// fp32 dim by dim, relu summed head by head, times fp32 1/sqrt(128).
float score(const uint16_t *q, const uint16_t *key) {
  float s = 0.0F;
  for (int h = 0; h < kIxHeads; ++h) {
    float acc = 0.0F;
    for (int d = 0; d < kIxHd; ++d) acc = std::fmaf(bf16_to_f32(q[h * kIxHd + d]), bf16_to_f32(key[d]), acc);
    s += std::max(acc, 0.0F);
  }
  return s * (1.0F / std::sqrt(static_cast<float>(kIxHd)));
}

// The value a restatement settles on: the device's own where it lies in the allowed set (checked
// by the Tally), else the set's first.
std::vector<uint16_t> settled(const uint16_t *got, const Restated &r) {
  std::vector<uint16_t> out(r.allowed.size());
  for (std::size_t d = 0; d < out.size(); ++d) {
    const auto &set = r.allowed[d];
    out[d] = std::find(set.begin(), set.end(), got[d]) != set.end() ? got[d] : set[0];
  }
  return out;
}

}  // namespace

int main() {
  MOE_CUDA(cudaSetDevice(0));
  const fn::Geometry g = geometry();
  ignis_topology topology{};
  topology.rotary_dim = 64;
  topology.rope_theta = 1e7;
  ignis::RopeScaling four;
  four.factor = 4.0F;
  const ix::Rope rope = ix::rope_from(fn::text_rope(topology, four));
  const ix::Rope linear = ix::rope_from(fn::text_rope(topology, {}));
  check(rope.scale > 1.1F && linear.scale == 1.0F, "the factor-4 table carries its attention factor");
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  DeviceBytes d_slots(64 * 4), d_positions(64 * 4);
  upload(d_slots, std::vector<int32_t>{0, 1, 2, 3, 4});

  // ---- prepare: the QSA's 24 query heads and 2 key heads -----------------------------------
  {
    constexpr int D = qsa::kHeadDim;
    const std::vector<uint16_t> qg = grid(11, static_cast<std::size_t>(kPositions) * qsa::kQProjWidth);
    const std::vector<uint16_t> k = grid(12, static_cast<std::size_t>(kPositions) * qsa::kKvWidth);
    const std::vector<uint16_t> q_norm = norm_weights(13, D), k_norm = norm_weights(14, D);
    auto d_qg = device_copy(qg);
    auto d_k = device_copy(k);
    auto d_q_norm = device_copy(q_norm);
    auto d_k_norm = device_copy(k_norm);
    DeviceBytes d_q(static_cast<std::size_t>(kPositions) * qsa::kOutWidth * 2);
    upload(d_positions, std::vector<int32_t>(kRowPositions, kRowPositions + kPositions));
    fn::Batch batch;
    batch.lanes = kPositions;
    batch.tokens = 1;
    batch.positions = d_positions.as<int32_t>();
    IX_OK(qsa::prepare(g, rope, d_q_norm->p, d_k_norm->p, batch, d_qg->as<__nv_bfloat16>(), d_q.as<__nv_bfloat16>(),
                       d_k->as<__nv_bfloat16>(), stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    const auto q_out = download<uint16_t>(d_q.p, static_cast<std::size_t>(kPositions) * qsa::kOutWidth);
    const auto k_out = download<uint16_t>(d_k->p, static_cast<std::size_t>(kPositions) * qsa::kKvWidth);
    Tally tally;
    for (int r = 0; r < kPositions; ++r) {
      for (int h = 0; h < qsa::kQHeads; ++h) {
        const std::vector<float> n = norm(widen(&qg[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * 2 * D], D), q_norm);
        tally.add(&q_out[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * D], restate(n, rope, kRowPositions[r]),
                  restate(n, linear, kRowPositions[r]));
      }
      for (int h = 0; h < qsa::kKvHeads; ++h) {
        const std::vector<float> n = norm(widen(&k[(static_cast<std::size_t>(r) * qsa::kKvHeads + h) * D], D), k_norm);
        tally.add(&k_out[(static_cast<std::size_t>(r) * qsa::kKvHeads + h) * D], restate(n, rope, kRowPositions[r]),
                  restate(n, linear, kRowPositions[r]));
      }
    }
    tally.report("prepare (q and k)");
  }

  // ---- prepare's rotation sources (GitHub #557): three axes, a rope delta, neither -----------
  // A span with an image rotates each token at its three axes (batch.rope_positions, axis-major
  // [3][rows]); a lane after an image at its position plus its rope delta (batch.rope_deltas
  // [lanes]: decode and verify); every other call at its positions, as before. The positions stay
  // the KV index and rotate nothing once a source is named. The axes and deltas are far apart, so
  // they move the rotary pairs they reach (the YaRN table throughout).
  {
    constexpr int D = qsa::kHeadDim;
    constexpr int kRows = 8;
    const std::vector<uint16_t> qg = grid(31, static_cast<std::size_t>(kRows) * qsa::kQProjWidth);
    const std::vector<uint16_t> k = grid(32, static_cast<std::size_t>(kRows) * qsa::kKvWidth);
    const std::vector<uint16_t> q_norm = norm_weights(33, D), k_norm = norm_weights(34, D);
    auto d_q_norm = device_copy(q_norm);
    auto d_k_norm = device_copy(k_norm);
    DeviceBytes d_axes(3 * kRows * 4), d_deltas(kRows * 4);
    struct Out {
      std::vector<uint16_t> q, k;
      bool operator==(const Out &o) const { return q == o.q && k == o.k; }
    };
    // prepare over the kRows rows at `batch` (lane-major); k is normed in place, so every call
    // starts from the same inputs.
    const auto run = [&](const fn::Batch &batch) {
      auto d_qg = device_copy(qg);
      auto d_k = device_copy(k);
      DeviceBytes d_q(static_cast<std::size_t>(kRows) * qsa::kOutWidth * 2);
      IX_OK(qsa::prepare(g, rope, d_q_norm->p, d_k_norm->p, batch, d_qg->as<__nv_bfloat16>(),
                         d_q.as<__nv_bfloat16>(), d_k->as<__nv_bfloat16>(), stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      return Out{download<uint16_t>(d_q.p, static_cast<std::size_t>(kRows) * qsa::kOutWidth),
                 download<uint16_t>(d_k->p, static_cast<std::size_t>(kRows) * qsa::kKvWidth)};
    };
    // Every head of row r against its restatement at `want` and, for the tally's bar, at `other`.
    const auto tally_row = [&](Tally &tally, const Out &out, int r, const std::array<int32_t, 3> &want,
                               const std::array<int32_t, 3> &other) {
      for (int h = 0; h < qsa::kQHeads; ++h) {
        const std::vector<float> n = norm(widen(&qg[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * 2 * D], D), q_norm);
        tally.add(&out.q[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * D], restate_axes(n, rope, want),
                  restate_axes(n, rope, other));
      }
      for (int h = 0; h < qsa::kKvHeads; ++h) {
        const std::vector<float> n = norm(widen(&k[(static_cast<std::size_t>(r) * qsa::kKvHeads + h) * D], D), k_norm);
        tally.add(&out.k[(static_cast<std::size_t>(r) * qsa::kKvHeads + h) * D], restate_axes(n, rope, want),
                  restate_axes(n, rope, other));
      }
    };
    const auto flat = [](int32_t p) { return std::array<int32_t, 3>{p, p, p}; };

    // Three axes: one lane's span at KV positions 1000.., every token's (t, h, w) distinct.
    const std::array<int32_t, 3> axes[kRows] = {{0, 17, 589823},       {262143, 5, 70000},  {300000, 1048575, 3},
                                                {99999, 524288, 42},   {123456, 654321, 7}, {42, 262145, 1000000},
                                                {2, 524287, 262144},   {777777, 1, 31337}};
    std::vector<int32_t> axis_major(3 * kRows);
    for (int r = 0; r < kRows; ++r) {
      for (int a = 0; a < 3; ++a) axis_major[static_cast<std::size_t>(a) * kRows + r] = axes[r][a];
    }
    upload(d_axes, axis_major);
    upload(d_positions, std::vector<int32_t>{1000});
    fn::Batch span;
    span.lanes = 1;
    span.tokens = kRows;
    span.positions = d_positions.as<int32_t>();
    span.rope_positions = d_axes.as<int32_t>();
    const Out three = run(span);
    Tally tally;
    long off_axis = 0, off_axis_as_t = 0;
    for (int r = 0; r < kRows; ++r) {
      tally_row(tally, three, r, axes[r], flat(1000 + r));
      // The pairs on axes 1 and 2 (i % 3 != 0) are not where axis 0's position puts them.
      for (int h = 0; h < qsa::kQHeads; ++h) {
        const std::vector<float> n = norm(widen(&qg[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * 2 * D], D), q_norm);
        const Restated t_only = restate_axes(n, rope, flat(axes[r][0]));
        const uint16_t *got = &three.q[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * D];
        for (int i = 0; i < 32; ++i) {
          if (i % 3 == 0) continue;
          for (int d : {i, i + 32}) {
            ++off_axis;
            const auto &set = t_only.allowed[d];
            off_axis_as_t += std::find(set.begin(), set.end(), got[d]) != set.end();
          }
        }
      }
    }
    tally.report("prepare at three axes", "the KV position's");
    std::printf("  prepare at three axes: %ld/%ld values of the pairs on axes 1 and 2 axis 0's\n", off_axis_as_t,
                off_axis);
    check(off_axis_as_t * 10 < off_axis, "prepare at three axes: pair i rotates at axis i % 3, not at axis 0");

    // Neither, three equal axes, and deltas of zero: the same bytes -- the text path, bit for bit.
    const int32_t first = 262140;
    upload(d_positions, std::vector<int32_t>{first});
    fn::Batch plain;
    plain.lanes = 1;
    plain.tokens = kRows;
    plain.positions = d_positions.as<int32_t>();
    const Out text = run(plain);
    for (int r = 0; r < kRows; ++r) {
      for (int a = 0; a < 3; ++a) axis_major[static_cast<std::size_t>(a) * kRows + r] = first + r;
    }
    upload(d_axes, axis_major);
    fn::Batch equal = plain;
    equal.rope_positions = d_axes.as<int32_t>();
    check(run(equal) == text, "three equal axes rotate bit for bit as the positions do");
    upload(d_deltas, std::vector<int32_t>{0});
    fn::Batch zero = plain;
    zero.rope_deltas = d_deltas.as<int32_t>();
    check(run(zero) == text, "a rope delta of 0 rotates bit for bit as the positions do");
    Tally text_tally;
    for (int r = 0; r < kRows; ++r) tally_row(text_tally, text, r, flat(first + r), flat(first + r + 600000));
    text_tally.report("prepare at the positions", "another position's");

    // A rope delta: 8 decode lanes of one token, and 4 verify lanes of 2 columns.
    const int32_t decode_positions[kRows] = {0, 9, 64, 262143, 600000, 500000, 1048000, 77};
    const int32_t decode_deltas[kRows] = {400000, 600000, 900000, 500000, -450000, -480000, -1000000, 1500000};
    upload(d_positions, std::vector<int32_t>(decode_positions, decode_positions + kRows));
    upload(d_deltas, std::vector<int32_t>(decode_deltas, decode_deltas + kRows));
    fn::Batch decode;
    decode.lanes = kRows;
    decode.tokens = 1;
    decode.positions = d_positions.as<int32_t>();
    decode.rope_deltas = d_deltas.as<int32_t>();
    const Out decoded = run(decode);
    Tally decode_tally;
    for (int r = 0; r < kRows; ++r) {
      tally_row(decode_tally, decoded, r, flat(decode_positions[r] + decode_deltas[r]), flat(decode_positions[r]));
    }
    decode_tally.report("prepare at a rope delta (decode)", "the position's alone");
    const int32_t verify_positions[4] = {1000, 300, 262143, 7000};
    const int32_t verify_deltas[4] = {499000, 700000, -262000, 1200000};
    upload(d_positions, std::vector<int32_t>(verify_positions, verify_positions + 4));
    upload(d_deltas, std::vector<int32_t>(verify_deltas, verify_deltas + 4));
    fn::Batch verify;
    verify.lanes = 4;
    verify.tokens = 2;
    verify.positions = d_positions.as<int32_t>();
    verify.rope_deltas = d_deltas.as<int32_t>();
    const Out verified = run(verify);
    Tally verify_tally;
    for (int r = 0; r < kRows; ++r) {
      const int32_t position = verify_positions[r / 2] + r % 2;
      tally_row(verify_tally, verified, r, flat(position + verify_deltas[r / 2]), flat(position));
    }
    verify_tally.report("prepare at a rope delta (verify)", "the position's alone");

    // One source at a time: a batch naming both is refused.
    fn::Batch both = span;
    both.rope_deltas = d_deltas.as<int32_t>();
    DeviceBytes d_q(static_cast<std::size_t>(kRows) * qsa::kOutWidth * 2);
    auto d_qg = device_copy(qg);
    auto d_k = device_copy(k);
    check(qsa::prepare(g, rope, d_q_norm->p, d_k_norm->p, both, d_qg->as<__nv_bfloat16>(), d_q.as<__nv_bfloat16>(),
                       d_k->as<__nv_bfloat16>(), stream) != nullptr,
          "a batch naming three axes and a rope delta is refused");
  }

  const std::vector<uint16_t> ix_q_norm = norm_weights(21, kIxHd), ix_k_norm = norm_weights(22, kIxHd);
  auto d_ix_q_norm = device_copy(ix_q_norm);
  auto d_ix_k_norm = device_copy(ix_k_norm);

  // ---- queries: the indexer's 4 query heads --------------------------------------------------
  {
    const std::vector<uint16_t> qk = grid(23, static_cast<std::size_t>(kPositions) * kQk);
    auto d_qk = device_copy(qk);
    DeviceBytes d_q(static_cast<std::size_t>(kPositions) * kIxHeads * kIxHd * 2);
    upload(d_positions, std::vector<int32_t>(kRowPositions, kRowPositions + kPositions));
    fn::Batch batch;
    batch.lanes = kPositions;
    batch.tokens = 1;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    IX_OK(ix::prepare_queries(g, rope, d_ix_q_norm->p, batch, 0, kPositions, d_qk->p, d_q.as<__nv_bfloat16>(), stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    const auto q_out = download<uint16_t>(d_q.p, static_cast<std::size_t>(kPositions) * kIxHeads * kIxHd);
    Tally tally;
    for (int r = 0; r < kPositions; ++r) {
      for (int h = 0; h < kIxHeads; ++h) {
        const std::vector<float> n = norm(widen(&qk[static_cast<std::size_t>(r) * kQk + h * kIxHd], kIxHd), ix_q_norm);
        tally.add(&q_out[(static_cast<std::size_t>(r) * kIxHeads + h) * kIxHd], restate(n, rope, kRowPositions[r]),
                  restate(n, linear, kRowPositions[r]));
      }
    }
    tally.report("indexer queries");
  }

  // ---- keys: a pooled block key per lane, roped at its block's first position ---------------
  {
    constexpr int32_t kLogical = 1048576 / ix::kPageTokens;
    constexpr int kPages = kPositions + 1;  // a lane's page each, and one every other logical page names
    std::vector<int32_t> tables(static_cast<std::size_t>(kPositions) * kLogical, kPositions);
    for (int l = 0; l < kPositions; ++l) tables[static_cast<std::size_t>(l) * kLogical + kBlockFirsts[l] / ix::kPageTokens] = l;
    DeviceBytes d_tables(tables.size() * 4), d_keys(static_cast<std::size_t>(kPages) * ix::kBlocksPerPage * kIxHd * 2),
        d_tail(static_cast<std::size_t>(kPositions) * (ix::kCompress - 1) * kIxHd * 2);
    upload(d_tables, tables);
    MOE_CUDA(cudaMemset(d_keys.p, 0, d_keys.bytes));
    MOE_CUDA(cudaMemset(d_tail.p, 0, d_tail.bytes));
    ix::Paged paged;
    paged.block_tables = d_tables.as<int32_t>();
    paged.logical_pages = kLogical;
    paged.block_keys = d_keys.as<__nv_bfloat16>();
    paged.tail_keys = d_tail.as<__nv_bfloat16>();
    const std::vector<uint16_t> qk = grid(24, static_cast<std::size_t>(kPositions) * ix::kCompress * kQk);
    auto d_qk = device_copy(qk);
    upload(d_positions, std::vector<int32_t>(kBlockFirsts, kBlockFirsts + kPositions));
    fn::Batch batch;
    batch.lanes = kPositions;
    batch.tokens = ix::kCompress;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    IX_OK(ix::append_keys(g, paged, rope, d_ix_k_norm->p, batch, d_qk->p, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    const auto keys = download<uint16_t>(d_keys.p, static_cast<std::size_t>(kPages) * ix::kBlocksPerPage * kIxHd);
    Tally tally;
    for (int l = 0; l < kPositions; ++l) {
      std::vector<float> pooled(kIxHd);
      for (int d = 0; d < kIxHd; ++d) {
        float sum = 0.0F;
        for (int t = 0; t < ix::kCompress; ++t) {
          sum += bf16_to_f32(qk[(static_cast<std::size_t>(l) * ix::kCompress + t) * kQk + kIxHeads * kIxHd + d]);
        }
        pooled[d] = round_bf16(sum / static_cast<float>(ix::kCompress));
      }
      const std::vector<float> n = norm(pooled, ix_k_norm);
      const std::size_t at = (static_cast<std::size_t>(l) * ix::kBlocksPerPage +
                              (kBlockFirsts[l] % ix::kPageTokens) / ix::kCompress) * kIxHd;
      tally.add(&keys[at], restate(n, rope, kBlockFirsts[l]), restate(n, linear, kBlockFirsts[l]));
    }
    tally.report("indexer block keys");
    bool sink_clean = true;
    for (std::size_t i = static_cast<std::size_t>(kPositions) * ix::kBlocksPerPage * kIxHd; i < keys.size(); ++i) {
      sink_clean = sink_clean && keys[i] == 0;
    }
    check(sink_clean, "indexer block keys: no key lands on a page its block does not map");
  }

  // ---- select: a 2,600-token prefill's sparse rows --------------------------------------------
  {
    const int32_t logical = (kSelectTokens + ix::kPageTokens - 1) / ix::kPageTokens;
    std::vector<int32_t> tables(logical);
    for (int32_t p = 0; p < logical; ++p) tables[p] = p;
    DeviceBytes d_tables(tables.size() * 4), d_keys(static_cast<std::size_t>(logical) * ix::kBlocksPerPage * kIxHd * 2),
        d_tail(static_cast<std::size_t>(ix::kCompress - 1) * kIxHd * 2);
    upload(d_tables, tables);
    ix::Paged paged;
    paged.block_tables = d_tables.as<int32_t>();
    paged.logical_pages = logical;
    paged.block_keys = d_keys.as<__nv_bfloat16>();
    paged.tail_keys = d_tail.as<__nv_bfloat16>();
    const std::vector<uint16_t> qk = grid(25, static_cast<std::size_t>(kSelectTokens) * kQk);
    auto d_qk = device_copy(qk);
    upload(d_positions, std::vector<int32_t>{0});
    fn::Batch batch;
    batch.lanes = 1;
    batch.tokens = kSelectTokens;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    batch.max_visible = kSelectTokens;
    const int32_t first_sparse = kBlockTopk * ix::kCompress + ix::kCompress - 1;  // 2,051: 513 blocks
    const int32_t sparse_rows = kSelectTokens - first_sparse;
    const int32_t width = g.selection_width();
    DeviceBytes d_tokens(static_cast<std::size_t>(kSelectTokens) * width * 4), d_counts(kSelectTokens * 4);
    DeviceBytes d_q(static_cast<std::size_t>(sparse_rows) * kIxHeads * kIxHd * 2);
    ninfer::DeviceArena arena(ix::select_scratch_bytes(g, kSelectTokens, kSelectTokens) + (1 << 20));
    fn::Selection out{d_tokens.as<int32_t>(), d_counts.as<int32_t>()};
    IX_OK(ix::append_keys(g, paged, rope, d_ix_k_norm->p, batch, d_qk->p, stream));
    IX_OK(ix::select(g, paged, rope, d_ix_q_norm->p, batch, d_qk->p, out, arena, stream));
    check(!out.dense, "select: a 2,600-token prefill selects");
    IX_OK(ix::prepare_queries(g, rope, d_ix_q_norm->p, batch, first_sparse, sparse_rows, d_qk->p,
                              d_q.as<__nv_bfloat16>(), stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    const auto keys = download<uint16_t>(d_keys.p, static_cast<std::size_t>(logical) * ix::kBlocksPerPage * kIxHd);
    const auto q_out = download<uint16_t>(d_q.p, static_cast<std::size_t>(sparse_rows) * kIxHeads * kIxHd);
    const auto tokens = download<int32_t>(d_tokens.p, static_cast<std::size_t>(kSelectTokens) * width);
    const int32_t blocks = kSelectTokens / ix::kCompress;
    Tally tally;
    // Block b's key sits at page b / 16, its slot b % 16 (identity tables).
    std::vector<std::vector<uint16_t>> key(blocks), key_linear(blocks);
    for (int32_t b = 0; b < blocks; ++b) {
      std::vector<float> pooled(kIxHd);
      for (int d = 0; d < kIxHd; ++d) {
        float sum = 0.0F;
        for (int t = 0; t < ix::kCompress; ++t) {
          sum += bf16_to_f32(qk[(static_cast<std::size_t>(b) * ix::kCompress + t) * kQk + kIxHeads * kIxHd + d]);
        }
        pooled[d] = round_bf16(sum / static_cast<float>(ix::kCompress));
      }
      const std::vector<float> n = norm(pooled, ix_k_norm);
      const Restated want = restate(n, rope, b * ix::kCompress), lin = restate(n, linear, b * ix::kCompress);
      const uint16_t *got = &keys[static_cast<std::size_t>(b) * kIxHd];
      tally.add(got, want, lin);
      key[b] = settled(got, want);
      key_linear[b] = settled(got, lin);
    }
    long rows_differing_from_linear = 0;
    bool selections = true;
    for (int32_t r = 0; r < sparse_rows; ++r) {
      const int32_t position = first_sparse + r;
      std::vector<uint16_t> q(kIxHeads * kIxHd), q_linear(kIxHeads * kIxHd);
      for (int h = 0; h < kIxHeads; ++h) {
        const std::vector<float> n =
            norm(widen(&qk[static_cast<std::size_t>(position) * kQk + h * kIxHd], kIxHd), ix_q_norm);
        const Restated want = restate(n, rope, position), lin = restate(n, linear, position);
        const uint16_t *got = &q_out[(static_cast<std::size_t>(r) * kIxHeads + h) * kIxHd];
        tally.add(got, want, lin);
        const std::vector<uint16_t> s = settled(got, want), sl = settled(got, lin);
        std::copy(s.begin(), s.end(), q.begin() + h * kIxHd);
        std::copy(sl.begin(), sl.end(), q_linear.begin() + h * kIxHd);
      }
      const int32_t row_blocks = (position + 1) / ix::kCompress;
      std::vector<float> scores(row_blocks), scores_linear(row_blocks);
      for (int32_t b = 0; b < row_blocks; ++b) {
        scores[b] = score(q.data(), key[b].data());
        scores_linear[b] = score(q_linear.data(), key_linear[b].data());
      }
      const std::vector<int32_t> want = rule(scores, kBlockTopk);
      std::vector<int32_t> got(kBlockTopk);
      for (int i = 0; i < kBlockTopk; ++i) got[i] = tokens[static_cast<std::size_t>(position) * width + 4 * i] / 4;
      selections = selections && got == want;
      rows_differing_from_linear += rule(scores_linear, kBlockTopk) != want;
    }
    tally.report("select (keys and queries)");
    std::printf("  select: %d sparse rows, %ld select other blocks under the linear table\n", sparse_rows,
                rows_differing_from_linear);
    check(selections, "select: every sparse row's blocks are the host top-k over the restated queries and keys");
    check(rows_differing_from_linear > 0, "select: the YaRN table moves the selection");
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next rope: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash-next rope: all checks passed\n");
  return EXIT_SUCCESS;
}
