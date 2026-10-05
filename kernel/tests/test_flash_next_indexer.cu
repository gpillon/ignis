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
// - block keys (all 1025 of both lanes) and queries of the recorded rows against the module's,
//   element by element within the bound ElementCheck derives (one BF16 ulp, plus the rotary
//   pair's carry for rope dims);
// - scores of the recorded rows against the module's, each within the bound score_bound derives
//   from the actual query and key differences and fp32 accumulation;
// - selection: every recorded row's blocks equal the MODULE's own selection except at DOCUMENTED
//   TIES: a block in one set and not the other must score within the two blocks' bounds of the
//   module's k-th block (an exact tie, the zero scores relu makes, is a zero gap); rows with at
//   most 512 blocks list every visible token; the tail tokens follow the blocks;
// - the tie rule exactly: every sparse row's selection equals "the k largest of OUR scores,
//   lowest block index first among equal ones" recomputed on the host from the downloaded scores;
// - the wave driver (select) writes what the staged calls wrote; a dense call (max_visible <=
//   2051) sets Selection::dense and writes nothing;
// - a decode call captured in a CUDA graph and replayed equals its eager run;
// - the checks bite: a block key roped one block late (positions shifted by 4 into a spare slot)
//   and a query roped one position late (a shifted batch) each fail the element check;
// - fn_indexer_select, the program's entry point, on a real seq pool with indexer sections
//   (attention layer 1 of 2, its arena exactly fn_indexer_select_scratch_bytes): a dense chunk
//   [0, 2000) sets Selection::dense, then the sparse chunk [2000, 4096) selects what the stages'
//   one-shot run selected, and the pool's layer-1 section holds the stages' block keys.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "../src/flash_next/indexer.h"

#include "ignis_moe.h"
#include "ignis_seq_internal.h"
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
constexpr int kSlots = 4;  // lane 0 chunked, lane 1, lane 0 one-shot, the mutation arm
constexpr int kLogicalPages = 72;
constexpr int kPhysicalPages = kSlots * kLogicalPages;
constexpr int kRotary = 64;

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
  std::vector<uint16_t> q;              // [4][128] BF16, the module's
  std::vector<float> scores;            // [blocks], the module's
  std::vector<int32_t> module_select;  // ascending blocks the module's own mask keeps
};

// One BF16 ulp at |v| (the smallest normal's below it).
double ulp_bf16(double v) {
  const double a = std::fabs(v);
  return a < 0x1p-126 ? 0x1p-133 : std::ldexp(1.0, std::ilogb(a) - 7);
}

// A 128-wide normed + roped row (a query head or a block key) against the module's, element by
// element. The one step whose arithmetic differs from torch's is the RMSNorm's fp32 sum of
// squares (another order), so a normed value x may land one BF16 ulp away; the pooled mean of
// four BF16 values in fp32 is exact either way. A dimension past the rotary 64 is that x: one
// ulp. A rotary dimension is bf16(bf16(x1 c) + bf16(-+x2 s)) for its pair (x1, x2): a product
// moves by at most ulp(x) from x, ulp(x) more if c or s (fp32 cos/sin rounded to BF16) sits on the
// other side of a BF16 rounding boundary, and ulp(x) from its own rounding; the sum by those two
// terms plus both sides' final rounding (ulp(y), and ulp(r) for a cancelling sum that grew), so
// |diff| <= 6 ulp(r) + ulp(y), r = |(y1, y2)| = |(x1, x2)| (the rotation keeps the pair's norm).
// That bound is what lets a near-zero rotary output -- torch's cancelling sum, hundreds of the
// output's own ulps away after a one-ulp input difference -- pass. A rope angle off by more than
// ~0.05 rad (one position on the first seven pairs, one block on the first ten) fails it.
struct ElementCheck {
  std::size_t exact = 0, total = 0, violations = 0;
  void add(const uint16_t *got, const uint16_t *want, std::size_t n) {
    for (std::size_t r = 0; r < n; r += kHd) {
      for (int i = 0; i < kHd; ++i) {
        const double w = bf16_to_f32(want[r + i]);
        double bound = ulp_bf16(w);
        if (i < kRotary) {
          const int pair = i % (kRotary / 2);
          const double y1 = bf16_to_f32(want[r + pair]), y2 = bf16_to_f32(want[r + pair + kRotary / 2]);
          bound += 6.0 * ulp_bf16(std::sqrt(y1 * y1 + y2 * y2));
        }
        exact += got[r + i] == want[r + i];
        violations += !(std::fabs(bf16_to_f32(got[r + i]) - w) <= bound);
        ++total;
      }
    }
  }
};

// The bound on |our score - the module's| for one block: the first-order effect of the actual
// differences of the queries (dq) and the key (dk), |dq| |k| + |q| |dk| + |dq| |dk| over the four
// heads (relu moves no score further than its argument), plus both sides' fp32 accumulation --
// 128 exact BF16 products summed in any order, at most 127 u sum|q k| each, u = 2^-24 -- and the
// relu sum and scale (5 u of the score), all divided by sqrt(128) like the score.
double score_bound(const uint16_t *q_ours, const uint16_t *q_mod, const uint16_t *k_ours, const uint16_t *k_mod,
                   double score) {
  double first = 0.0, terms = 0.0;
  for (int h = 0; h < kHeads; ++h) {
    for (int dd = 0; dd < kHd; ++dd) {
      const double qm = bf16_to_f32(q_mod[h * kHd + dd]), km = bf16_to_f32(k_mod[dd]);
      const double dq = std::fabs(bf16_to_f32(q_ours[h * kHd + dd]) - qm);
      const double dk = std::fabs(bf16_to_f32(k_ours[dd]) - km);
      first += dq * std::fabs(km) + std::fabs(qm) * dk + dq * dk;
      terms += std::fabs(qm * km);
    }
  }
  return (first + 2.0 * 127.0 * 0x1p-24 * terms) / std::sqrt(static_cast<double>(kHd)) + 5.0 * 0x1p-24 * std::fabs(score);
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

// What the recorded rows add up to.
struct Stats {
  int rows_checked = 0;
  int documented_ties = 0;       // blocks in one selection and not the other
  double worst_score = 0.0;      // the largest |our score - the module's| / its bound
  ElementCheck queries;
};

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
  if (res.dense) {
    MOE_CUDA(cudaStreamSynchronize(d.stream));
    return res;
  }
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

// Checks the recorded rows that fall in a sparse call: batch lane l is fixture lane
// fixture_lanes[l] in slot slots[l], its first token at positions[l]; module_keys[lane] are the
// module's block keys of fixture lane `lane`.
void check_rows(const Device &d, const std::vector<Row> &recorded, const std::vector<int> &fixture_lanes,
                const std::vector<int32_t> &slots, const std::vector<int32_t> &positions, int tokens,
                const std::vector<uint16_t> (&module_keys)[2], const std::string &what, Stats *st) {
  for (std::size_t l = 0; l < positions.size(); ++l) {
    std::vector<uint16_t> keys;  // this lane's block keys, read once
    for (const Row &rec : recorded) {
      const int t = rec.position - positions[l];
      if (fixture_lanes[l] != rec.lane || t < 0 || t >= tokens) continue;
      const int row = static_cast<int>(l) * tokens + t;
      const std::string tag = what + " lane " + std::to_string(rec.lane) + " row " + std::to_string(rec.position);
      ++st->rows_checked;
      const int blocks = (rec.position + 1) / 4;
      const auto tokens_row = download<int32_t>(d.tok_a.as<int32_t>() + static_cast<std::size_t>(row) * kWidth, kWidth);
      const int32_t count = download<int32_t>(d.cnt_a.as<int32_t>() + row, 1)[0];
      if (blocks <= kBlockTopk) {
        bool ok = count == rec.position + 1;
        for (int i = 0; i < kWidth; ++i) ok = ok && tokens_row[i] == (i <= rec.position ? i : -1);
        check(ok, tag + ": a dense row must list every visible token");
        continue;
      }
      if (keys.empty()) keys = slot_block_keys(d, slots[l], (kT + 3) / 4);
      const uint16_t *mk = module_keys[rec.lane].data();
      // queries
      const auto q = download<uint16_t>(d.q.as<__nv_bfloat16>() + static_cast<std::size_t>(row) * kHeads * kHd, kHeads * kHd);
      st->queries.add(q.data(), rec.q.data(), q.size());
      // scores, each within its derived bound
      const auto s = download<float>(d.scores.as<float>() + static_cast<std::size_t>(row) * d.score_stride, blocks);
      std::vector<double> bound(blocks);
      bool scores_ok = true;
      for (int b = 0; b < blocks; ++b) {
        bound[b] = score_bound(q.data(), rec.q.data(), &keys[static_cast<std::size_t>(b) * kHd],
                               &mk[static_cast<std::size_t>(b) * kHd], rec.scores[b]);
        const double dev = std::fabs(static_cast<double>(s[b]) - rec.scores[b]);
        st->worst_score = std::max(st->worst_score, dev / bound[b]);
        scores_ok = scores_ok && dev <= bound[b];
      }
      check(scores_ok, tag + ": a score differs from the module's past its derived bound");
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
      // against the module's own selection: a block in only one of the two must tie with the
      // module's k-th block within the two blocks' score bounds (an exact tie is a zero gap)
      std::vector<int32_t> by_score(blocks);
      for (int b = 0; b < blocks; ++b) by_score[b] = b;
      std::stable_sort(by_score.begin(), by_score.end(),
                       [&](int32_t a, int32_t b) { return rec.scores[a] > rec.scores[b]; });
      const int32_t kth = by_score[kBlockTopk - 1];
      std::set<int32_t> a(ours.begin(), ours.end()), m(rec.module_select.begin(), rec.module_select.end());
      bool at_ties = true;
      for (const auto &[one, other] : {std::make_pair(&a, &m), std::make_pair(&m, &a)}) {
        for (int32_t b : *one) {
          if (other->count(b) != 0) continue;
          ++st->documented_ties;
          at_ties = at_ties && std::fabs(static_cast<double>(rec.scores[b]) - rec.scores[kth]) <= bound[b] + bound[kth];
        }
      }
      check(at_ties, tag + ": a selected block differs from the module's away from a score tie");
    }
  }
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

  const std::vector<uint16_t> module_keys[2] = {need(fx, "lane0.block_keys").as<uint16_t>(),
                                                need(fx, "lane1.block_keys").as<uint16_t>()};
  std::vector<Row> recorded;
  for (int lane = 0; lane < 2; ++lane) {
    for (int32_t r : (lane == 0 ? lane0_rows : lane1_rows)) {
      const std::string p = "lane" + std::to_string(lane) + ".row" + std::to_string(r) + ".";
      Row row;
      row.lane = lane;
      row.position = r;
      row.q = need(fx, p + "q").as<uint16_t>();
      row.scores = need(fx, p + "scores").as<float>();
      row.module_select = need(fx, p + "module_select").as<int32_t>();
      recorded.push_back(std::move(row));
    }
  }

  Device d;
  MOE_CUDA(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
  upload(d.w, w.payload);
  upload(d.q_norm, w.q_norm);
  upload(d.k_norm, w.k_norm);
  std::vector<int32_t> tables(kPhysicalPages);
  for (int i = 0; i < kPhysicalPages; ++i) tables[i] = (i * 173 + 29) % kPhysicalPages;  // a permutation of 288
  upload(d.tables, tables);
  MOE_CUDA(cudaMemsetAsync(d.keys.p, 0, d.keys.bytes, d.stream));
  MOE_CUDA(cudaMemsetAsync(d.tail.p, 0, d.tail.bytes, d.stream));
  ninfer::DeviceArena arena(ix::select_scratch_bytes(g, kT, 8192) + (1 << 20));

  // The exactness premise: the device projection equals the exact sum, bit for bit.
  {
    MOE_CUDA(cudaMemcpyAsync(d.xin.p, x0.data(), 3 * kHidden * 2, cudaMemcpyHostToDevice, d.stream));
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

  Stats st;
  auto load_rows = [&](const std::vector<uint16_t> &x, int first, int n) {
    MOE_CUDA(cudaMemcpyAsync(d.xin.p, &x[static_cast<std::size_t>(first) * kHidden],
                             static_cast<std::size_t>(n) * kHidden * 2, cudaMemcpyHostToDevice, d.stream));
  };

  // Lane 0, slot 0: prefill chunks whose boundaries split blocks.
  const int chunks[][2] = {{0, 1023}, {1023, 2050}, {2050, 2051}, {2051, 4096}};
  for (const auto &c : chunks) {
    const int n = c[1] - c[0];
    load_rows(x0, c[0], n);
    const std::string what = "prefill [" + std::to_string(c[0]) + "," + std::to_string(c[1]) + ")";
    const CallResult res = run_call(d, g, rope, arena, {0}, {c[0]}, n, c[1], what);
    check(res.dense == (c[1] <= 2051), what + ": Selection::dense is wrong");
    if (!res.dense) check_rows(d, recorded, {0}, {0}, {c[0]}, n, module_keys, what, &st);
  }
  // Lane 1, slot 1: one prefill call.
  {
    load_rows(x1, 0, 4096);
    const CallResult res = run_call(d, g, rope, arena, {1}, {0}, 4096, 4096, "lane 1 prefill");
    check(!res.dense, "lane 1 prefill: a 4096-token call is not dense");
    check_rows(d, recorded, {1}, {1}, {0}, 4096, module_keys, "lane 1 prefill", &st);
  }
  // Decode 4096..4099, both lanes in one call; the graph bound 8192 stands for max_visible.
  std::vector<int32_t> decode_lists[3];
  std::vector<uint16_t> keys_after_4098, tail_after_4098;
  for (int p = 4096; p < 4100; ++p) {
    MOE_CUDA(cudaMemcpyAsync(d.xin.p, &x0[static_cast<std::size_t>(p) * kHidden], kHidden * 2, cudaMemcpyHostToDevice,
                             d.stream));
    MOE_CUDA(cudaMemcpyAsync(d.xin.as<uint16_t>() + kHidden, &x1[static_cast<std::size_t>(p) * kHidden], kHidden * 2,
                             cudaMemcpyHostToDevice, d.stream));
    const std::string what = "decode " + std::to_string(p);
    run_call(d, g, rope, arena, {0, 1}, {p, p}, 1, 8192, what);
    check_rows(d, recorded, {0, 1}, {0, 1}, {p, p}, 1, module_keys, what, &st);
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
    MOE_CUDA(cudaMemsetAsync(d.tok_b.p, 0x7F, 2 * kWidth * 4, d.stream));
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

  // Both lanes' block keys against the module's, all 1025 blocks.
  for (int lane = 0; lane < 2; ++lane) {
    ElementCheck keys;
    keys.add(slot_block_keys(d, lane, 1025).data(), module_keys[lane].data(), module_keys[lane].size());
    std::printf("  lane %d block keys: %zu / %zu bit-exact with the module, %zu past the element bound\n", lane,
                keys.exact, keys.total, keys.violations);
    check(keys.violations == 0, "lane " + std::to_string(lane) + " block keys differ from the module's past the bound");
  }

  // Lane 0 again in one call, slot 2: the same state and the same selections as chunks + decode.
  std::vector<int32_t> one_shot_lists;
  {
    load_rows(x0, 0, 4099);
    run_call(d, g, rope, arena, {2}, {0}, 4099, 4099, "one-shot [0,4099)");
    one_shot_lists = download<int32_t>(d.tok_a.p, static_cast<std::size_t>(4096) * kWidth);
    check(slot_block_keys(d, 2, 1024) == keys_after_4098, "one-shot block keys differ from chunked + decode");
    check(download<uint16_t>(d.tail.as<uint16_t>() + 2 * 3 * kHd, 3 * kHd) == tail_after_4098,
          "one-shot tail differs from chunked + decode");
    for (int p = 4096; p < 4099; ++p) {
      check(download<int32_t>(d.tok_a.as<int32_t>() + static_cast<std::size_t>(p) * kWidth, kWidth) == decode_lists[p - 4096],
            "one-shot row " + std::to_string(p) + " selects differently from decode");
    }
  }

  // fn_indexer_select on a seq pool: attention layer 1 of 2, lane 0's tokens.
  {
    ignis_seq_pool_spec spec{};
    spec.num_kv_heads = 2;
    spec.head_dim = 256;
    spec.kv_format = IGNIS_KV_FORMAT_BF16;
    spec.kv_page_group_count = 80;
    spec.max_context_tokens = 4160;
    spec.slot_count = 1;
    spec.gdn_num_layers = 1;
    spec.gdn_conv_channels = 10240;
    spec.gdn_value_heads = 48;
    spec.gdn_head_dim = 128;
    spec.vocab = 1024;
    spec.kv_num_layers = 2;
    spec.indexer_key_dim = kHd;
    spec.indexer_compress_tokens = 4;
    ignis_seq_pool *pool = nullptr;
    ignis_seq *seq = nullptr;
    if (ignis_seq_pool_create(&spec, &pool) != 0 || ignis_seq_alloc(pool, 4096, &seq) != 0) {
      std::fprintf(stderr, "FATAL: seq pool: %s\n", ignis_seq_last_error());
      return 1;
    }
    fn::IndexerLayerState layers[2];
    for (int l = 0; l < 2; ++l) {
      layers[l].block_keys = pool->indexer_block_keys(l);
      layers[l].blocks_per_page = ix::kBlocksPerPage;
      layers[l].tail_keys = pool->indexer_tail_keys(l);
    }
    fn::Context ctx;
    ctx.g = g;
    ctx.kv_format = IGNIS_KV_FORMAT_BF16;
    ctx.rope = ninfer::ops::rope_linear_frequencies(1e7F, 64);
    ctx.pool = pool;
    ctx.indexer = layers;
    fn::IndexerWeights wts;
    wts.qk_proj = fn::Linear{d.w.p, kQk, kHidden, fn::WeightFormat::Fp8RowScale};
    wts.q_norm = d.q_norm.p;
    wts.k_norm = d.k_norm.p;
    const std::vector<int32_t> slot = {seq->slot};
    bool ok = true;
    for (const auto &c : {std::pair<int, int>{0, 2000}, std::pair<int, int>{2000, 4096}}) {
      const int n = c.second - c.first;
      load_rows(x0, c.first, n);
      const std::vector<int32_t> first = {c.first};
      MOE_CUDA(cudaMemcpyAsync(d.slots.p, slot.data(), 4, cudaMemcpyHostToDevice, d.stream));
      MOE_CUDA(cudaMemcpyAsync(d.positions.p, first.data(), 4, cudaMemcpyHostToDevice, d.stream));
      fn::Batch b;
      b.lanes = 1;
      b.tokens = n;
      b.slots = d.slots.as<int32_t>();
      b.positions = d.positions.as<int32_t>();
      b.max_visible = c.second;
      ninfer::DeviceArena exact_arena(fn::fn_indexer_select_scratch_bytes(g, n, c.second));
      fn::Selection out{d.tok_b.as<int32_t>(), d.cnt_b.as<int32_t>()};
      if (fn::fn_indexer_select(ctx, 1, wts, b, d.xin.p, out, exact_arena, d.stream) != 0) {
        std::fprintf(stderr, "FATAL: fn_indexer_select: %s\n", fn::fn_last_error());
        return 1;
      }
      MOE_CUDA(cudaStreamSynchronize(d.stream));
      if (c.first == 0) {
        check(out.dense, "fn_indexer_select: a call up to 2000 visible tokens must set Selection::dense");
        continue;
      }
      check(!out.dense, "fn_indexer_select: a call up to 4096 visible tokens is not dense");
      const auto got = download<int32_t>(d.tok_b.p, static_cast<std::size_t>(n) * kWidth);
      ok = ok && std::equal(got.begin(), got.end(), one_shot_lists.begin() + static_cast<std::size_t>(c.first) * kWidth);
    }
    check(ok, "fn_indexer_select: the sparse chunk selects differently from the stages' one-shot run");
    // The pool's layer-1 section against the stages' keys (slot 0, chunked + decode).
    const auto row = download<int32_t>(pool->kv_pool.block_table_row(seq->slot).data, 64);
    bool keys_equal = true;
    for (int bk = 0; bk < 1024 && keys_equal; ++bk) {
      const std::size_t at = (static_cast<std::size_t>(row[bk / 16]) * ix::kBlocksPerPage + bk % 16) * kHd;
      const auto key = download<uint16_t>(static_cast<uint16_t *>(layers[1].block_keys) + at, kHd);
      keys_equal = std::equal(key.begin(), key.end(), keys_after_4098.begin() + static_cast<std::size_t>(bk) * kHd);
    }
    check(keys_equal, "fn_indexer_select: the pool's layer-1 block keys differ from the stages'");
    ignis_seq_release(pool, seq);
    ignis_seq_pool_free(pool);
  }

  // The checks bite. A block key roped one block late: the first 64 tokens appended as if they
  // started at position 4, into the spare slot, so block b + 1 holds block b's pooled, normed key
  // roped at 4 (b + 1). Its non-rotary half must still match; the element check must flag it.
  {
    load_rows(x0, 0, 64);
    run_call(d, g, rope, arena, {3}, {4}, 64, 68, "mutation: keys roped one block late");
    const auto late = slot_block_keys(d, 3, 17);
    ElementCheck mutated;
    mutated.add(&late[kHd], module_keys[0].data(), 16 * kHd);
    bool nope_equal = true;
    for (int b = 0; b < 16; ++b) {
      for (int i = kRotary; i < kHd; ++i) {
        nope_equal = nope_equal && late[static_cast<std::size_t>(b + 1) * kHd + i] == module_keys[0][static_cast<std::size_t>(b) * kHd + i];
      }
    }
    std::printf("  mutation, keys roped one block late: %zu of %zu values past the element bound\n", mutated.violations,
                mutated.total);
    check(nope_equal, "mutation arm: the late-roped keys' non-rotary half should equal the module's");
    check(mutated.violations > 0, "mutation arm: the element check misses keys roped one block late");
  }
  // A query roped one position late: lane 0's row 4099 prepared as if at 4100.
  {
    load_rows(x0, 4099, 1);
    if (ignis_fp8_linear(d.w.p, kQk, kHidden, d.xin.p, 1, d.qk.p, 0, d.stream) != 0) return 1;
    const std::vector<int32_t> late_slot = {0}, late_pos = {4100};
    MOE_CUDA(cudaMemcpyAsync(d.slots.p, late_slot.data(), 4, cudaMemcpyHostToDevice, d.stream));
    MOE_CUDA(cudaMemcpyAsync(d.positions.p, late_pos.data(), 4, cudaMemcpyHostToDevice, d.stream));
    fn::Batch b;
    b.lanes = 1;
    b.tokens = 1;
    b.slots = d.slots.as<int32_t>();
    b.positions = d.positions.as<int32_t>();
    b.max_visible = 4101;
    IX_OK(ix::prepare_queries(g, rope, d.q_norm.p, b, 0, 1, d.qk.p, d.q.as<__nv_bfloat16>(), d.stream));
    MOE_CUDA(cudaStreamSynchronize(d.stream));
    const auto late = download<uint16_t>(d.q.p, kHeads * kHd);
    const auto it = std::find_if(recorded.begin(), recorded.end(),
                                 [](const Row &r) { return r.lane == 0 && r.position == 4099; });
    ElementCheck mutated;
    mutated.add(late.data(), it->q.data(), late.size());
    std::printf("  mutation, query roped one position late: %zu of %zu values past the element bound\n",
                mutated.violations, mutated.total);
    check(mutated.violations > 0, "mutation arm: the element check misses a query roped one position late");
  }

  std::printf("  %d recorded rows checked; scores within %.2f of their derived bounds; queries %zu / %zu bit-exact, "
              "%zu past the element bound; %d block(s) differ from the module, all at documented ties\n",
              st.rows_checked, st.worst_score, st.queries.exact, st.queries.total, st.queries.violations,
              st.documented_ties);
  check(st.rows_checked == static_cast<int>(recorded.size()) - 2,  // rows 2049 and 2050 sit in dense calls
        "not every recorded row was checked: " + std::to_string(st.rows_checked));
  check(st.queries.violations == 0, "queries differ from the module's past the element bound");
  MOE_CUDA(cudaStreamDestroy(d.stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_indexer: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_flash_next_indexer: OK\n");
  return 0;
}
