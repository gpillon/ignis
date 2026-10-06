// Shared by the Flash-Next sparse attention tests (spec flash-next/04, GitHub #302, slice S3) --
// OURS: the checkpoint's selection shape, synthetic paged K/V at real geometry, the fp64
// reference over a row's listed tokens, and one call's upload / run / check.
#ifndef IGNIS_FLASH_NEXT_SPARSE_TEST_COMMON_H
#define IGNIS_FLASH_NEXT_SPARSE_TEST_COMMON_H

#include "../src/flash_next/qsa_sparse.h"

#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace sparse_test {

using namespace moe_test;
namespace fn = ignis::flash_next;
namespace sp = ignis::flash_next::sparse;

constexpr int kQHeads = 24, kKvHeads = 2, kHd = 256, kWidth = 2051;
constexpr int kSeqTokens = 6000;
constexpr int kSlots = 3;
constexpr int kLogicalPages = (kSeqTokens + 63) / 64;
constexpr int kPhysicalPages = kSlots * kLogicalPages;
constexpr int kMaxRows = 300;
constexpr uint32_t kK = 0x5A00, kV = 0x5A01, kQ = 0x5A02, kPick = 0x5A03;

#define SP_OK(expr)                                                                                \
  do {                                                                                             \
    const ::sparse_test::sp::Status st_ = (expr);                                                  \
    if (st_ != nullptr) {                                                                          \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, st_);                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

inline fn::Geometry geometry() {
  fn::Geometry g;
  g.q_heads = kQHeads;
  g.kv_heads = kKvHeads;
  g.head_dim = kHd;
  g.rotary_dim = 64;
  g.indexer_budget = 2048;
  g.compress_ratio = 4;
  return g;
}

// The checkpoint's selection shape for a row at `p`: every visible token while it has at most
// 512 blocks, else 512 distinct blocks (a hash-driven partial shuffle), ascending, then the tail.
inline std::vector<int32_t> make_list(int p, uint32_t salt) {
  std::vector<int32_t> out;
  const int blocks = (p + 1) / 4;
  if (blocks <= 512) {
    for (int t = 0; t <= p; ++t) out.push_back(t);
    return out;
  }
  std::vector<int32_t> ids(blocks);
  for (int b = 0; b < blocks; ++b) ids[b] = b;
  for (int i = 0; i < 512; ++i) {
    const int j = i + static_cast<int>(hash_u32(kPick + salt, static_cast<uint64_t>(p) * 4096 + i) % (blocks - i));
    std::swap(ids[i], ids[j]);
  }
  ids.resize(512);
  std::sort(ids.begin(), ids.end());
  for (int32_t b : ids)
    for (int t = 0; t < 4; ++t) out.push_back(4 * b + t);
  for (int t = blocks * 4; t <= p; ++t) out.push_back(t);
  return out;
}

// Synthetic BF16 K/V pages, counter-hash values in [-1, 1), pages permuted.
struct Pages {
  std::vector<uint16_t> k, v;   // [physical page][kv head][64][256]
  std::vector<int32_t> tables;  // [slot][logical page]
  size_t row_at(int slot, int pos, int head) const {
    const int page = tables[slot * kLogicalPages + pos / 64];
    return ((static_cast<size_t>(page) * kKvHeads + head) * 64 + pos % 64) * kHd;
  }
};

inline Pages make_pages() {
  Pages pg;
  const size_t n = static_cast<size_t>(kPhysicalPages) * kKvHeads * 64 * kHd;
  pg.k.resize(n);
  pg.v.resize(n);
  for (size_t i = 0; i < n; ++i) {
    pg.k[i] = f32_to_bf16(hash_uniform(kK, i, 1.0F));
    pg.v[i] = f32_to_bf16(hash_uniform(kV, i, 1.0F));
  }
  pg.tables.resize(kPhysicalPages);
  for (int i = 0; i < kPhysicalPages; ++i) pg.tables[i] = (i * 101 + 7) % kPhysicalPages;  // a permutation
  return pg;
}

// The K (role 0) or V (role 1) row a route should read for (slot, position, head), in fp64.
using RowFn = std::function<void(int role, int slot, int pos, int head, double *out)>;

inline RowFn exact_rows(const Pages &pg) {
  return [&pg](int role, int slot, int pos, int head, double *out) {
    const uint16_t *src = &(role == 0 ? pg.k : pg.v)[pg.row_at(slot, pos, head)];
    for (int d = 0; d < kHd; ++d) out[d] = bf16_to_f32(src[d]);
  };
}

// fp64 attention of one row (all query heads) over its list, and each output's error bound.
// - The kernel's weights are p = bf16(exp2(s - m)) for its running maximum m: the exact weight
//   times (1 + d), |d| <= 2^-9 (BF16) + 2^-12 (exp2f and the fp32 rescales by exp2 of maxima
//   differences, at most 65 of them); normalized by the sum of the same weights, the output moves
//   by at most (2^-9 + 2^-12) sum_j w_j |v_j - out| (w the exact softmax).
// - A score moves by at most e_j = 255 u sum_d |q_d k_jd| / 16 (fp32 accumulation of 256 exact
//   products, u = 2^-24), so every weight by a factor within e^(+-2 max e_j):
//   (e^(2 max e_j) - 1) sum_j w_j |v_j - out|.
// - The fp32 accumulation of up to n products: n u sum_j w_j |v_j|; the BF16 rounding of the
//   result: 2^-9 |out|.
// The rows are the ones the kernel reads (for hq KV: its decoded scratch, downloaded).
struct Reference {
  std::vector<double> out, bound;  // [q_heads][head_dim]
};

inline Reference reference(const RowFn &rows, const std::vector<uint16_t> &q, int row, int slot,
                           const std::vector<int32_t> &list) {
  const size_t n = list.size();
  std::vector<double> k(n * kKvHeads * kHd), v(n * kKvHeads * kHd);
  for (size_t j = 0; j < n; ++j) {
    for (int h = 0; h < kKvHeads; ++h) {
      rows(0, slot, list[j], h, &k[(j * kKvHeads + h) * kHd]);
      rows(1, slot, list[j], h, &v[(j * kKvHeads + h) * kHd]);
    }
  }
  Reference ref;
  ref.out.assign(static_cast<size_t>(kQHeads) * kHd, 0.0);
  ref.bound.assign(ref.out.size(), 0.0);
  std::vector<double> w(n);
  for (int h = 0; h < kQHeads; ++h) {
    const int kvh = h / sp::kGroup;
    const uint16_t *qh = &q[(static_cast<size_t>(row) * kQHeads + h) * kHd];
    double m = -1e300, terms = 0.0;
    for (size_t j = 0; j < n; ++j) {
      double dot = 0.0, abs_sum = 0.0;
      const double *kr = &k[(j * kKvHeads + kvh) * kHd];
      for (int d = 0; d < kHd; ++d) {
        dot += static_cast<double>(bf16_to_f32(qh[d])) * kr[d];
        abs_sum += std::fabs(static_cast<double>(bf16_to_f32(qh[d])) * kr[d]);
      }
      w[j] = dot / 16.0;
      m = std::max(m, w[j]);
      terms = std::max(terms, abs_sum);
    }
    const double reweight = std::exp(2.0 * 255.0 * 0x1p-24 * terms / 16.0) - 1.0;
    double l = 0.0;
    for (size_t j = 0; j < n; ++j) {
      w[j] = std::exp(w[j] - m);
      l += w[j];
    }
    for (size_t j = 0; j < n; ++j) w[j] /= l;
    double *out = &ref.out[static_cast<size_t>(h) * kHd];
    double *bound = &ref.bound[static_cast<size_t>(h) * kHd];
    for (size_t j = 0; j < n; ++j) {
      const double *vr = &v[(j * kKvHeads + kvh) * kHd];
      for (int d = 0; d < kHd; ++d) out[d] += w[j] * vr[d];
    }
    for (int d = 0; d < kHd; ++d) {
      double dev = 0.0, mag = 0.0;
      for (size_t j = 0; j < n; ++j) {
        const double vj = v[(j * kKvHeads + kvh) * kHd + d];
        dev += w[j] * std::fabs(vj - out[d]);
        mag += w[j] * std::fabs(vj);
      }
      bound[d] = (0x1p-9 + 0x1p-12 + reweight) * dev + static_cast<double>(n) * 0x1p-24 * mag +
                 0x1p-9 * std::fabs(out[d]) + 0x1p-40;
    }
  }
  return ref;
}

struct Device {
  DeviceBytes q{static_cast<size_t>(kMaxRows) * kQHeads * kHd * 2};
  DeviceBytes out{static_cast<size_t>(kMaxRows) * kQHeads * kHd * 2};
  DeviceBytes tokens{static_cast<size_t>(kMaxRows) * kWidth * 4};
  DeviceBytes counts{kMaxRows * 4};
  DeviceBytes slots{16}, positions{16};
  DeviceBytes partials;
  explicit Device(size_t partial_bytes) : partials(partial_bytes) {}
};

struct Call {
  std::vector<int32_t> slots, positions;
  int tokens = 0;
  int max_visible = kSeqTokens;
  std::vector<std::vector<int32_t>> lists;  // per row
};

// What a route enqueues for one call: everything between the uploads and the output.
using Enqueue = std::function<void(const fn::Batch &, const fn::Selection &, cudaStream_t)>;

// Uploads a call's lists, runs `enqueue` (eagerly or captured in a CUDA graph and replayed),
// and returns the BF16 output.
inline std::vector<uint16_t> run(Device &d, const Call &c, const Enqueue &enqueue, cudaStream_t stream,
                                 bool graph = false) {
  const int rows = static_cast<int>(c.lists.size());
  std::vector<int32_t> tokens(static_cast<size_t>(rows) * kWidth, -1), counts(rows);
  for (int r = 0; r < rows; ++r) {
    counts[r] = static_cast<int32_t>(c.lists[r].size());
    std::copy(c.lists[r].begin(), c.lists[r].end(), tokens.begin() + static_cast<size_t>(r) * kWidth);
  }
  // Everything on `stream`: a non-blocking stream is not ordered after the legacy stream.
  MOE_CUDA(cudaMemcpyAsync(d.tokens.p, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice, stream));
  MOE_CUDA(cudaMemcpyAsync(d.counts.p, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice, stream));
  MOE_CUDA(cudaMemcpyAsync(d.slots.p, c.slots.data(), c.slots.size() * 4, cudaMemcpyHostToDevice, stream));
  MOE_CUDA(cudaMemcpyAsync(d.positions.p, c.positions.data(), c.positions.size() * 4, cudaMemcpyHostToDevice,
                           stream));
  fn::Batch b;
  b.lanes = static_cast<int32_t>(c.slots.size());
  b.tokens = c.tokens;
  b.slots = d.slots.as<int32_t>();
  b.positions = d.positions.as<int32_t>();
  b.max_visible = c.max_visible;
  fn::Selection sel{d.tokens.as<int32_t>(), d.counts.as<int32_t>()};
  MOE_CUDA(cudaMemsetAsync(d.out.p, 0xFF, static_cast<size_t>(rows) * kQHeads * kHd * 2, stream));
  if (graph) {
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ex = nullptr;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    enqueue(b, sel, stream);
    MOE_CUDA(cudaStreamEndCapture(stream, &gr));
    MOE_CUDA(cudaGraphInstantiate(&ex, gr, 0));
    MOE_CUDA(cudaGraphLaunch(ex, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    MOE_CUDA(cudaGraphExecDestroy(ex));
    MOE_CUDA(cudaGraphDestroy(gr));
  } else {
    enqueue(b, sel, stream);
    MOE_CUDA(cudaStreamSynchronize(stream));
  }
  return download<uint16_t>(d.out.p, static_cast<size_t>(rows) * kQHeads * kHd);
}

// Every `sample_step`-th row of a call against the fp64 reference over the rows `rows` names,
// each output within its derived bound (reference()). Returns whether every checked one was.
inline bool check_call(const std::string &what, const RowFn &rows, const std::vector<uint16_t> &q, const Call &c,
                       const std::vector<uint16_t> &got, int sample_step) {
  const int n = static_cast<int>(c.lists.size());
  double worst = 0.0;
  int checked = 0;
  for (int r = 0; r < n; r += sample_step) {
    const int slot = c.slots[r / c.tokens];
    const Reference ref = reference(rows, q, r, slot, c.lists[r]);
    for (size_t i = 0; i < ref.out.size(); ++i) {
      const double err = std::fabs(bf16_to_f32(got[static_cast<size_t>(r) * kQHeads * kHd + i]) - ref.out[i]);
      worst = std::max(worst, err / ref.bound[i]);
      if (!(err <= ref.bound[i])) {
        check(false, what + ": row " + std::to_string(r) + " element " + std::to_string(i) + " error " +
                         std::to_string(err) + " above " + std::to_string(ref.bound[i]));
        return false;
      }
    }
    ++checked;
  }
  std::printf("  %-40s %3d rows checked, worst error %.3f of the bound\n", what.c_str(), checked, worst);
  return true;
}

// Position-coded pages for the exactness arms: every K row zero, except the `needles` (slot,
// position) rows whose K is 4 in every dimension; V row (pos, head) one-hot at dimension
// code(pos, head). With q = 1 everywhere, a row without a needle in its list has all scores 0,
// so every weight is exactly 1 and its output is exactly (listed positions coding to d) / count:
// one wrong, missing or repeated row moves an output by 1/count, ~16 BF16 ulps of the value at
// 2051 tokens. A needle scores 64 against 0 for the rest, so the output is its V row to
// exp2(-92).
inline int code_of(int pos, int head) { return (pos * 7 + head * 3) % kHd; }

inline Pages make_coded_pages(const std::vector<std::pair<int, int>> &needles) {
  Pages pg;
  const size_t n = static_cast<size_t>(kPhysicalPages) * kKvHeads * 64 * kHd;
  pg.k.assign(n, 0);
  pg.v.assign(n, 0);
  pg.tables.resize(kPhysicalPages);
  for (int i = 0; i < kPhysicalPages; ++i) pg.tables[i] = (i * 101 + 7) % kPhysicalPages;
  for (int slot = 0; slot < kSlots; ++slot) {
    for (int pos = 0; pos < kSeqTokens; ++pos) {
      for (int h = 0; h < kKvHeads; ++h) pg.v[pg.row_at(slot, pos, h) + code_of(pos, h)] = f32_to_bf16(1.0F);
    }
  }
  for (const auto &[slot, pos] : needles) {
    for (int h = 0; h < kKvHeads; ++h)
      for (int d = 0; d < kHd; ++d) pg.k[pg.row_at(slot, pos, h) + d] = f32_to_bf16(4.0F);
  }
  return pg;
}

// The exact expectation over coded pages for a row without a needle, compared to a BF16 ulp
// (the kernel divides by the count through an fp32 reciprocal, then rounds once to BF16).
// Returns the number of outputs past that.
inline int coded_mismatches(const Call &c, const std::vector<uint16_t> &got, int r) {
  int bad = 0;
  const auto &list = c.lists[r];
  for (int h = 0; h < kQHeads; ++h) {
    const int kvh = h / sp::kGroup;
    std::vector<double> want(kHd, 0.0);
    for (int32_t pos : list) want[code_of(pos, kvh)] += 1.0;
    for (int d = 0; d < kHd; ++d) {
      const double w = want[d] / static_cast<double>(list.size());
      const double o = bf16_to_f32(got[(static_cast<size_t>(r) * kQHeads + h) * kHd + d]);
      const double ulp = w == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(w) - 7);
      bad += !(std::fabs(o - w) <= ulp);
    }
  }
  return bad;
}

}  // namespace sparse_test

#endif  // IGNIS_FLASH_NEXT_SPARSE_TEST_COMMON_H
