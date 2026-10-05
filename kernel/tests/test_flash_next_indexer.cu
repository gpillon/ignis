// The Flash-Next QSA indexer (spec flash-next/04, GitHub #302, slice S3) at real geometry --
// OURS (ADR 0043): hidden 2560, 4 query heads of 128 and one key head, compress 4, budget 2048
// (512 blocks), rotary 64 at theta 1e7, against the checkpoint's own Qwen4ExpTextQSAIndexer
// (fixtures/flash_next_indexer/record_indexer.py, which asserted the module's selected-token mask
// equal to its op-for-op re-derivation before writing).
//
// Two sequences of 4100 tokens through the real stages -- kern's FP8 linear for the projection,
// then append_keys, prepare_queries, score, select_blocks -- in the shapes the program runs them:
//   lane 0 (slot 0): prefill chunks [0,1023) [1023,2050) [2050,2051) [2051,4096), so blocks
//                    complete across chunk boundaries from the tail, then decode 4096..4099;
//   lane 1 (slot 1): the TIE sequence (hundreds of blocks score exactly 0 for its recorded rows),
//                    prefill [0,4096) in one call, then decode 4096..4099 beside lane 0;
//   lane 0 again (slot 2): [0,4099) in one call -- its block keys and tail must equal slot 0's
//                    after decode 4098 bit for bit, and its rows 4096..4098 select what decode did.
//
// Checked:
// - the projection is the exact sum (the fixture's inputs make every fp32 partial sum exact), so
//   the indexer's own arithmetic is all that is compared;
// - block keys (all 1025 of lane 0) and queries of the recorded rows against the module's, to a
//   BF16 ulp (the RMSNorm's fp32 sum of squares runs in another order);
// - scores of the recorded rows against the module's, within kScoreTol;
// - selection: every recorded row's blocks equal the module's except at DOCUMENTED TIES, i.e. a
//   block in one set and not the other must score within kScoreTol of the module's k-th score;
//   rows with at most 512 blocks list every visible token; the tail tokens follow the blocks;
// - the tie rule exactly: every sparse row's selection equals "the k largest of OUR scores,
//   lowest block index first among equal ones" recomputed on the host from the downloaded scores;
// - the wave driver (select) writes what the staged calls wrote; a dense call (max_visible <=
//   2051) sets Selection::dense and writes nothing;
// - a decode call captured in a CUDA graph and replayed equals its eager run.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "../src/flash_next/indexer.h"

#include "ignis_moe.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace moe_test;
namespace fn = ignis::flash_next;
namespace ix = ignis::flash_next::indexer;

namespace {

constexpr int kHidden = 2560;
constexpr int kHeads = 4;
constexpr int kHd = 128;
constexpr int kQk = (kHeads + 1) * kHd;
constexpr int kT = 4100;
constexpr int kBlockTopk = 512;
constexpr int kWidth = 2051;
constexpr int kSlots = 3;
constexpr int kLogicalPages = 72;
constexpr int kPhysicalPages = kSlots * kLogicalPages;
constexpr float kScoreTol = 2e-3F;

constexpr uint32_t kCodes = 0x1D00, kScale = 0x1D01, kQNorm = 0x1D02, kKNorm = 0x1D03;
constexpr uint32_t kX0 = 0x1D10, kX1 = 0x1D11, kXQ = 0x1D12;

#define IX_OK(expr)                                                                                \
  do {                                                                                             \
    const ix::Status st_ = (expr);                                                                 \
    if (st_ != nullptr) {                                                                          \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, st_);                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = kHidden;
  g.head_dim = 256;
  g.rotary_dim = 64;
  g.indexer_heads = kHeads;
  g.indexer_head_dim = kHd;
  g.indexer_kv_heads = 1;
  g.compress_ratio = 4;
  g.indexer_budget = 2048;
  g.rms_norm_eps = 1e-6F;
  return g;
}

// record_indexer.py's synthetic weights: FP8 codes of magnitude 0.125..1.875, power-of-two row
// scales, (1 + w) norm weights on a 1/128 grid.
struct Weights {
  std::vector<uint8_t> payload;
  std::vector<double> code_value;
  std::vector<double> scale;
  std::vector<uint16_t> q_norm, k_norm;
};

Weights make_weights() {
  Weights w;
  const std::size_t codes = static_cast<std::size_t>(kQk) * kHidden;
  const std::size_t scale_at = (codes + 255) / 256 * 256;
  w.payload.assign(scale_at + kQk * 2, 0);
  w.code_value.resize(codes);
  for (std::size_t i = 0; i < codes; ++i) {
    const uint32_t h = hash_u32(kCodes, i);
    const uint8_t c = static_cast<uint8_t>((((h >> 5) & 1U) << 7) | ((4U + (h & 3U)) << 3) | ((h >> 2) & 7U));
    w.payload[i] = c;
    w.code_value[i] = e4m3_to_f32(c);
  }
  w.scale.resize(kQk);
  for (int r = 0; r < kQk; ++r) {
    const float s = std::ldexp(1.0F, -static_cast<int>(2 + hash_u32(kScale, r) % 4));
    const uint16_t b = f32_to_bf16(s);
    std::memcpy(&w.payload[scale_at + 2 * static_cast<std::size_t>(r)], &b, 2);
    w.scale[r] = s;
  }
  for (int d = 0; d < kHd; ++d) {
    w.q_norm.push_back(f32_to_bf16(static_cast<float>(static_cast<int>(hash_u32(kQNorm, d) % 65) - 32) / 128.0F));
    w.k_norm.push_back(f32_to_bf16(static_cast<float>(static_cast<int>(hash_u32(kKNorm, d) % 65) - 32) / 128.0F));
  }
  return w;
}

void grid_token(uint32_t stream, int t, uint16_t *out) {
  for (int c = 0; c < kHidden; ++c) {
    const uint32_t h = hash_u32(stream, static_cast<uint64_t>(t) * kHidden + c);
    out[c] = f32_to_bf16(static_cast<float>(static_cast<int>(h % 17) - 8) / 16.0F);
  }
}

// The fixture's per-row record.
struct Row {
  int lane = 0, position = 0;
  std::vector<uint16_t> q;   // [4][128] BF16
  std::vector<float> scores;  // [blocks]
  std::vector<int32_t> select;  // ascending blocks (the documented rule)
};

double bf16_ulps(uint16_t a, uint16_t b) {
  const float fa = bf16_to_f32(a), fb = bf16_to_f32(b);
  if (fa == fb) return 0.0;
  const float mag = std::max(std::fabs(fa), std::fabs(fb));
  return std::fabs(fa - fb) / std::ldexp(1.0F, std::ilogb(mag) - 7);
}

// The documented rule on a score row: the k largest, ties by lowest block index; ascending.
std::vector<int32_t> rule(const std::vector<float> &s, int k) {
  std::vector<int32_t> order(s.size());
  for (std::size_t i = 0; i < s.size(); ++i) order[i] = static_cast<int32_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    const float sa = s[a] > 0.0F ? s[a] : 0.0F, sb = s[b] > 0.0F ? s[b] : 0.0F;
    return sa > sb;
  });
  order.resize(k);
  std::sort(order.begin(), order.end());
  return order;
}

constexpr std::size_t kPayloadBytes = (static_cast<std::size_t>(kQk) * kHidden + 255) / 256 * 256 + kQk * 2;
constexpr int32_t kScoreStride = 2048;  // the decode calls' bound: max_visible 8192 / 4

struct Device {
  DeviceBytes w{kPayloadBytes};
  DeviceBytes q_norm{kHd * 2}, k_norm{kHd * 2};
  DeviceBytes xin{static_cast<std::size_t>(kT) * kHidden * 2};
  DeviceBytes qk{static_cast<std::size_t>(kT) * kQk * 2};
  DeviceBytes q{static_cast<std::size_t>(kT) * kHeads * kHd * 2};
  DeviceBytes scores{static_cast<std::size_t>(kT) * kScoreStride * 4};
  DeviceBytes tok_a{static_cast<std::size_t>(kT) * kWidth * 4}, tok_b{static_cast<std::size_t>(kT) * kWidth * 4};
  DeviceBytes cnt_a{kT * 4}, cnt_b{kT * 4};
  DeviceBytes tables{kPhysicalPages * 4};
  DeviceBytes keys{static_cast<std::size_t>(kPhysicalPages) * 16 * kHd * 2};
  DeviceBytes tail{static_cast<std::size_t>(kSlots) * 3 * kHd * 2};
  DeviceBytes slots{16}, positions{16};
  int32_t score_stride = kScoreStride;
  cudaStream_t stream = nullptr;
};

ix::Paged paged_of(const Device &d) {
  ix::Paged p;
  p.block_tables = d.tables.as<int32_t>();
  p.logical_pages = kLogicalPages;
  p.block_keys = d.keys.as<__nv_bfloat16>();
  p.tail_keys = d.tail.as<__nv_bfloat16>();
  return p;
}

// One call of the program's shape: the projection, the key append, and -- unless dense -- the
// staged selection into tok_a/cnt_a, the wave driver into tok_b/cnt_b, and the two compared.
struct CallResult {
  bool dense = false;
  int rows = 0, max_blocks = 0;
};

CallResult run_call(Device &d, const fn::Geometry &g, const ix::Rope &rope, ninfer::DeviceArena &arena,
                    const std::vector<int32_t> &slots, const std::vector<int32_t> &positions, int tokens,
                    int max_visible, const std::string &what) {
  fn::Batch b;
  b.lanes = static_cast<int32_t>(slots.size());
  b.tokens = tokens;
  b.slots = d.slots.as<int32_t>();
  b.positions = d.positions.as<int32_t>();
  b.max_visible = max_visible;
  MOE_CUDA(cudaMemcpyAsync(d.slots.p, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice, d.stream));
  MOE_CUDA(cudaMemcpyAsync(d.positions.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice, d.stream));
  const int rows = b.rows();
  if (ignis_fp8_linear(d.w.p, kQk, kHidden, d.xin.p, rows, d.qk.p, 0, d.stream) != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear: %s\n", ignis_fp8_linear_last_error());
    std::exit(EXIT_FAILURE);
  }
  const ix::Paged paged = paged_of(d);
  IX_OK(ix::append_keys(g, paged, rope, d.k_norm.p, b, d.qk.p, d.stream));
  CallResult res;
  res.rows = rows;
  res.max_blocks = max_visible / 4;
  fn::Selection sel_b{d.tok_b.as<int32_t>(), d.cnt_b.as<int32_t>()};
  IX_OK(ix::select(g, paged, rope, d.q_norm.p, b, d.qk.p, sel_b, arena, d.stream));
  res.dense = sel_b.dense;
  if (res.dense) return res;
  IX_OK(ix::prepare_queries(g, rope, d.q_norm.p, b, 0, rows, d.qk.p, d.q.as<__nv_bfloat16>(), d.stream));
  IX_OK(ix::score(g, paged, b, 0, rows, d.q.as<__nv_bfloat16>(), res.max_blocks, d.scores.as<float>(),
                  d.score_stride, d.stream));
  IX_OK(ix::select_blocks(g, b, 0, rows, d.scores.as<float>(), d.score_stride, d.tok_a.as<int32_t>(),
                          d.cnt_a.as<int32_t>(), d.stream));
  MOE_CUDA(cudaStreamSynchronize(d.stream));
  const auto ta = download<int32_t>(d.tok_a.p, static_cast<std::size_t>(rows) * kWidth);
  const auto tb = download<int32_t>(d.tok_b.p, static_cast<std::size_t>(rows) * kWidth);
  const auto ca = download<int32_t>(d.cnt_a.p, rows);
  const auto cb = download<int32_t>(d.cnt_b.p, rows);
  check(ta == tb && ca == cb, what + ": the wave driver's selection differs from the staged one");
  // Rows with at most 512 complete blocks list every visible token, sparse call or not.
  bool dense_rows = true;
  for (int row = 0; row < rows && dense_rows; ++row) {
    const int p = positions[row / tokens] + row % tokens;
    if ((p + 1) / 4 > kBlockTopk) continue;
    dense_rows = ca[row] == p + 1;
    for (int i = 0; i < kWidth && dense_rows; ++i) dense_rows = ta[static_cast<std::size_t>(row) * kWidth + i] == (i <= p ? i : -1);
  }
  check(dense_rows, what + ": a row with at most 512 blocks does not list every visible token");
  return res;
}

// Checks the recorded rows that fall in a sparse call: batch lane l is the fixture's lane
// fixture_lanes[l], its first token at positions[l].
void check_rows(const Device &d, const std::vector<Row> &recorded, const std::vector<int> &fixture_lanes,
                const std::vector<int32_t> &positions, int tokens, const std::string &what, int *rows_checked,
                int *documented_ties, double *max_score_err, double *max_q_ulps) {
  for (const Row &rec : recorded) {
    for (std::size_t l = 0; l < positions.size(); ++l) {
      const int t = rec.position - positions[l];
      if (fixture_lanes[l] != rec.lane || t < 0 || t >= tokens) continue;
      const int row = static_cast<int>(l) * tokens + t;
      const std::string tag = what + " lane " + std::to_string(rec.lane) + " row " + std::to_string(rec.position);
      ++*rows_checked;
      const int blocks = (rec.position + 1) / 4;
      const auto tokens_row = download<int32_t>(d.tok_a.as<int32_t>() + static_cast<std::size_t>(row) * kWidth, kWidth);
      const int32_t count = download<int32_t>(d.cnt_a.as<int32_t>() + row, 1)[0];
      if (blocks <= kBlockTopk) {
        bool ok = count == rec.position + 1;
        for (int i = 0; i < kWidth; ++i) ok = ok && tokens_row[i] == (i <= rec.position ? i : -1);
        check(ok, tag + ": a dense row must list every visible token");
        continue;
      }
      // q
      const auto q = download<uint16_t>(d.q.as<__nv_bfloat16>() + static_cast<std::size_t>(row) * kHeads * kHd, kHeads * kHd);
      for (int i = 0; i < kHeads * kHd; ++i) *max_q_ulps = std::max(*max_q_ulps, bf16_ulps(q[i], rec.q[i]));
      // scores
      const auto s = download<float>(d.scores.as<float>() + static_cast<std::size_t>(row) * d.score_stride, blocks);
      double err = 0.0;
      for (int b = 0; b < blocks; ++b) err = std::max(err, std::fabs(static_cast<double>(s[b]) - rec.scores[b]));
      *max_score_err = std::max(*max_score_err, err);
      check(err <= kScoreTol, tag + ": score error " + std::to_string(err) + " above the tolerance");
      // the list's shape
      std::vector<int32_t> ours;
      bool shape = count == kBlockTopk * 4 + (rec.position + 1 - blocks * 4);
      for (int i = 0; i < kBlockTopk; ++i) {
        const int32_t b0 = tokens_row[4 * i] / 4;
        for (int j = 0; j < 4; ++j) shape = shape && tokens_row[4 * i + j] == 4 * b0 + j;
        shape = shape && (i == 0 || b0 > ours.back()) && b0 < blocks;
        ours.push_back(b0);
      }
      for (int i = kBlockTopk * 4; i < kWidth; ++i) {
        shape = shape && tokens_row[i] == (i < count ? blocks * 4 + (i - kBlockTopk * 4) : -1);
      }
      check(shape, tag + ": the list is not 512 ascending blocks then the tail");
      // the tie rule, exactly, on our own scores
      check(ours == rule(s, kBlockTopk), tag + ": selection differs from the tie rule on its own scores");
      // against the module: a difference only at documented ties
      std::vector<float> sorted(rec.scores);
      std::sort(sorted.begin(), sorted.end(), std::greater<float>());
      const float kth = sorted[kBlockTopk - 1];
      std::set<int32_t> a(ours.begin(), ours.end()), m(rec.select.begin(), rec.select.end());
      int differ = 0;
      bool all_ties = true;
      for (int32_t b : a) {
        if (m.count(b) == 0) { ++differ; all_ties = all_ties && std::fabs(rec.scores[b] - kth) <= kScoreTol; }
      }
      for (int32_t b : m) {
        if (a.count(b) == 0) { ++differ; all_ties = all_ties && std::fabs(rec.scores[b] - kth) <= kScoreTol; }
      }
      *documented_ties += differ;
      check(all_ties, tag + ": a selected block differs from the module's away from a score tie");
    }
  }
}

std::vector<uint16_t> slot_block_keys(const Device &d, int slot, int blocks) {
  const auto tables = download<int32_t>(d.tables.p, static_cast<std::size_t>(kPhysicalPages));
  std::vector<uint16_t> out(static_cast<std::size_t>(blocks) * kHd);
  for (int b = 0; b < blocks; ++b) {
    const int page = tables[slot * kLogicalPages + (4 * b) / 64];
    const std::size_t at = (static_cast<std::size_t>(page) * 16 + ((4 * b) % 64) / 4) * kHd;
    MOE_CUDA(cudaMemcpy(&out[static_cast<std::size_t>(b) * kHd], d.keys.as<uint16_t>() + at, kHd * 2,
                        cudaMemcpyDeviceToHost));
  }
  return out;
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return 1;
  }
  const auto fx = read_fixture(IGNIS_FLASH_NEXT_INDEXER_FIXTURE);
  const fn::Geometry g = geometry();
  std::printf("Flash-Next QSA indexer: 4 x 128 queries, one key, compress 4, top 512 blocks, rotary 64\n");

  // Rope: the checkpoint's fp32 table from the load's double table.
  const ix::Rope rope = ix::rope_from(ninfer::ops::rope_linear_frequencies(1e7F, 64));
  const auto inv = need(fx, "inv_freq").as<float>();
  bool rope_ok = inv.size() == 32;
  for (int i = 0; i < 32 && rope_ok; ++i) rope_ok = rope.inv_freq[i] == inv[i];
  check(rope_ok, "rope_from does not give the checkpoint's fp32 inv_freq table");

  // Inputs.
  const Weights w = make_weights();
  std::vector<uint16_t> x0(static_cast<std::size_t>(kT) * kHidden), x1(x0.size()), xq(kHidden);
  for (int t = 0; t < kT; ++t) {
    grid_token(kX0, t, &x0[static_cast<std::size_t>(t) * kHidden]);
    grid_token(kX1, t, &x1[static_cast<std::size_t>(t) * kHidden]);
  }
  grid_token(kXQ, 0, xq.data());
  const auto x_rep = need(fx, "x_rep").as<uint16_t>();
  const auto lane0_rows = need(fx, "lane0.rows").as<int32_t>();
  const auto lane1_rows = need(fx, "lane1.rows").as<int32_t>();
  for (int t = 0; t < kT; ++t) {
    if ((t / 4) % 4 != 0) std::memcpy(&x1[static_cast<std::size_t>(t) * kHidden], x_rep.data(), kHidden * 2);
  }
  for (int32_t r : lane1_rows) std::memcpy(&x1[static_cast<std::size_t>(r) * kHidden], xq.data(), kHidden * 2);

  std::vector<Row> recorded;
  for (int lane = 0; lane < 2; ++lane) {
    for (int32_t r : (lane == 0 ? lane0_rows : lane1_rows)) {
      const std::string p = "lane" + std::to_string(lane) + ".row" + std::to_string(r) + ".";
      Row row;
      row.lane = lane;
      row.position = r;
      row.q = need(fx, p + "q").as<uint16_t>();
      row.scores = need(fx, p + "scores").as<float>();
      row.select = need(fx, p + "select").as<int32_t>();
      recorded.push_back(std::move(row));
    }
  }

  Device d;
  MOE_CUDA(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
  upload(d.w, w.payload);
  upload(d.q_norm, w.q_norm);
  upload(d.k_norm, w.k_norm);
  std::vector<int32_t> tables(kPhysicalPages);
  for (int i = 0; i < kPhysicalPages; ++i) tables[i] = (i * 173 + 29) % kPhysicalPages;  // a permutation
  upload(d.tables, tables);
  MOE_CUDA(cudaMemset(d.keys.p, 0, d.keys.bytes));
  MOE_CUDA(cudaMemset(d.tail.p, 0, d.tail.bytes));
  ninfer::DeviceArena arena(ix::select_scratch_bytes(g, kT, 8192) + (1 << 20));

  // The exactness premise: the device projection equals the exact sum, bit for bit.
  {
    MOE_CUDA(cudaMemcpy(d.xin.p, x0.data(), 3 * kHidden * 2, cudaMemcpyHostToDevice));
    if (ignis_fp8_linear(d.w.p, kQk, kHidden, d.xin.p, 3, d.qk.p, 0, d.stream) != 0) return 1;
    MOE_CUDA(cudaStreamSynchronize(d.stream));
    const auto qk = download<uint16_t>(d.qk.p, 3 * kQk);
    bool exact = true;
    for (int t = 0; t < 3; ++t) {
      for (int r = 0; r < kQk; ++r) {
        double s = 0.0;
        for (int c = 0; c < kHidden; ++c) {
          s += w.code_value[static_cast<std::size_t>(r) * kHidden + c] * bf16_to_f32(x0[static_cast<std::size_t>(t) * kHidden + c]);
        }
        exact = exact && qk[t * kQk + r] == f32_to_bf16(static_cast<float>(s * w.scale[r]));
      }
    }
    check(exact, "the FP8 projection of grid inputs is not the exact sum");
  }

  int rows_checked = 0, ties = 0;
  double max_score_err = 0.0, max_q_ulps = 0.0;
  auto load_rows = [&](const std::vector<uint16_t> &x, int first, int n) {
    MOE_CUDA(cudaMemcpy(d.xin.p, &x[static_cast<std::size_t>(first) * kHidden], static_cast<std::size_t>(n) * kHidden * 2,
                        cudaMemcpyHostToDevice));
  };

  // Lane 0, slot 0: prefill chunks whose boundaries split blocks.
  const int chunks[][2] = {{0, 1023}, {1023, 2050}, {2050, 2051}, {2051, 4096}};
  for (const auto &c : chunks) {
    const int n = c[1] - c[0];
    load_rows(x0, c[0], n);
    const std::string what = "prefill [" + std::to_string(c[0]) + "," + std::to_string(c[1]) + ")";
    const CallResult res = run_call(d, g, rope, arena, {0}, {c[0]}, n, c[1], what);
    check(res.dense == (c[1] <= 2051), what + ": Selection::dense is wrong");
    if (!res.dense) check_rows(d, recorded, {0}, {c[0]}, n, what, &rows_checked, &ties, &max_score_err, &max_q_ulps);
  }
  // Lane 1, slot 1: one prefill call.
  {
    load_rows(x1, 0, 4096);
    const CallResult res = run_call(d, g, rope, arena, {1}, {0}, 4096, 4096, "lane 1 prefill");
    check(!res.dense, "lane 1 prefill: a 4096-token call is not dense");
    check_rows(d, recorded, {1}, {0}, 4096, "lane 1 prefill", &rows_checked, &ties, &max_score_err, &max_q_ulps);
  }
  // Decode 4096..4099, both lanes in one call; the graph bound 8192 stands for max_visible.
  std::vector<int32_t> decode_lists[3];
  std::vector<uint16_t> keys_after_4098, tail_after_4098;
  for (int p = 4096; p < 4100; ++p) {
    MOE_CUDA(cudaMemcpy(d.xin.p, &x0[static_cast<std::size_t>(p) * kHidden], kHidden * 2, cudaMemcpyHostToDevice));
    MOE_CUDA(cudaMemcpy(d.xin.as<uint16_t>() + kHidden, &x1[static_cast<std::size_t>(p) * kHidden], kHidden * 2,
                        cudaMemcpyHostToDevice));
    const std::string what = "decode " + std::to_string(p);
    run_call(d, g, rope, arena, {0, 1}, {p, p}, 1, 8192, what);
    check_rows(d, recorded, {0, 1}, {p, p}, 1, what, &rows_checked, &ties, &max_score_err, &max_q_ulps);
    if (p < 4099) decode_lists[p - 4096] = download<int32_t>(d.tok_a.p, kWidth);
    if (p == 4098) {
      keys_after_4098 = slot_block_keys(d, 0, 1024);
      tail_after_4098 = download<uint16_t>(d.tail.p, 3 * kHd);
    }
  }

  // A decode call captured in a graph and replayed equals its eager run (re-appending block 1024
  // writes the same key).
  {
    const auto eager = download<int32_t>(d.tok_a.p, 2 * kWidth);
    fn::Batch b;
    b.lanes = 2;
    b.tokens = 1;
    b.slots = d.slots.as<int32_t>();
    b.positions = d.positions.as<int32_t>();
    b.max_visible = 8192;
    fn::Selection sel{d.tok_b.as<int32_t>(), d.cnt_b.as<int32_t>()};
    MOE_CUDA(cudaMemset(d.tok_b.p, 0x7F, 2 * kWidth * 4));
    cudaGraph_t graph = nullptr;
    MOE_CUDA(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
    if (ignis_fp8_linear(d.w.p, kQk, kHidden, d.xin.p, 2, d.qk.p, 0, d.stream) != 0) return 1;
    IX_OK(ix::append_keys(g, paged_of(d), rope, d.k_norm.p, b, d.qk.p, d.stream));
    IX_OK(ix::select(g, paged_of(d), rope, d.q_norm.p, b, d.qk.p, sel, arena, d.stream));
    MOE_CUDA(cudaStreamEndCapture(d.stream, &graph));
    cudaGraphExec_t exec = nullptr;
    MOE_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    MOE_CUDA(cudaGraphLaunch(exec, d.stream));
    MOE_CUDA(cudaStreamSynchronize(d.stream));
    check(download<int32_t>(d.tok_b.p, 2 * kWidth) == eager, "decode 4099: the graph replay differs from eager");
    MOE_CUDA(cudaGraphExecDestroy(exec));
    MOE_CUDA(cudaGraphDestroy(graph));
  }

  // Lane 0's block keys against the module's, all 1025 blocks.
  {
    const auto want = need(fx, "lane0.block_keys").as<uint16_t>();
    const auto got = slot_block_keys(d, 0, 1025);
    double worst = 0.0;
    std::size_t exact = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const double u = bf16_ulps(got[i], want[i]);
      worst = std::max(worst, u);
      exact += u == 0.0;
    }
    std::printf("  block keys: %zu / %zu bit-exact with the module, worst %.0f ulp\n", exact, want.size(), worst);
    check(worst <= 1.0, "block keys differ from the module's by more than one BF16 ulp");
  }

  // Lane 0 again in one call, slot 2: the same state and the same selections as chunks + decode.
  {
    load_rows(x0, 0, 4099);
    run_call(d, g, rope, arena, {2}, {0}, 4099, 4099, "one-shot [0,4099)");
    check(slot_block_keys(d, 2, 1024) == keys_after_4098, "one-shot block keys differ from chunked + decode");
    check(download<uint16_t>(d.tail.as<uint16_t>() + 2 * 3 * kHd, 3 * kHd) == tail_after_4098,
          "one-shot tail differs from chunked + decode");
    for (int p = 4096; p < 4099; ++p) {
      check(download<int32_t>(d.tok_a.as<int32_t>() + static_cast<std::size_t>(p) * kWidth, kWidth) == decode_lists[p - 4096],
            "one-shot row " + std::to_string(p) + " selects differently from decode");
    }
  }

  std::printf("  %d recorded rows checked; scores within %.2e of the module; queries within %.0f ulp; "
              "%d block(s) differ from the module, all at documented ties\n",
              rows_checked, max_score_err, max_q_ulps, ties);
  check(rows_checked == static_cast<int>(recorded.size()) - 2,  // rows 2049 and 2050 sit in dense calls
        "not every recorded row was checked: " + std::to_string(rows_checked));
  check(max_q_ulps <= 1.0, "queries differ from the module's by more than one BF16 ulp");
  MOE_CUDA(cudaStreamDestroy(d.stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_indexer: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_flash_next_indexer: OK\n");
  return 0;
}
