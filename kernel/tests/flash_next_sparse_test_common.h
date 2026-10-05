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

// fp64 attention of one row (all query heads) over its list.
inline std::vector<double> reference(const RowFn &rows, const std::vector<uint16_t> &q, int row, int slot,
                                     const std::vector<int32_t> &list) {
  const size_t n = list.size();
  std::vector<double> k(n * kKvHeads * kHd), v(n * kKvHeads * kHd);
  for (size_t j = 0; j < n; ++j) {
    for (int h = 0; h < kKvHeads; ++h) {
      rows(0, slot, list[j], h, &k[(j * kKvHeads + h) * kHd]);
      rows(1, slot, list[j], h, &v[(j * kKvHeads + h) * kHd]);
    }
  }
  std::vector<double> out(static_cast<size_t>(kQHeads) * kHd, 0.0), s(n);
  for (int h = 0; h < kQHeads; ++h) {
    const int kvh = h / sp::kGroup;
    const uint16_t *qh = &q[(static_cast<size_t>(row) * kQHeads + h) * kHd];
    double m = -1e300;
    for (size_t j = 0; j < n; ++j) {
      double dot = 0.0;
      const double *kr = &k[(j * kKvHeads + kvh) * kHd];
      for (int d = 0; d < kHd; ++d) dot += static_cast<double>(bf16_to_f32(qh[d])) * kr[d];
      s[j] = dot / 16.0;
      m = std::max(m, s[j]);
    }
    double l = 0.0;
    for (size_t j = 0; j < n; ++j) {
      s[j] = std::exp(s[j] - m);
      l += s[j];
    }
    for (size_t j = 0; j < n; ++j) {
      const double *vr = &v[(j * kKvHeads + kvh) * kHd];
      for (int d = 0; d < kHd; ++d) out[static_cast<size_t>(h) * kHd + d] += s[j] / l * vr[d];
    }
  }
  return out;
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

// Every `sample_step`-th row of a call against the fp64 reference over the rows `rows` names.
// S is BF16 products summed in fp32; P is rounded to BF16 (2^-9 relative per weight) and the
// output too (2^-9 of |out|): |out - ref| <= 2^-9 max|v| + 2^-9 |ref| + fp32 noise, asserted as
// 2^-8 (1 + |ref|) since max|v| <= 1.
inline void check_call(const std::string &what, const RowFn &rows, const std::vector<uint16_t> &q, const Call &c,
                       const std::vector<uint16_t> &got, int sample_step) {
  const int n = static_cast<int>(c.lists.size());
  double worst = 0.0;
  int checked = 0;
  for (int r = 0; r < n; r += sample_step) {
    const int slot = c.slots[r / c.tokens];
    const auto ref = reference(rows, q, r, slot, c.lists[r]);
    for (size_t i = 0; i < ref.size(); ++i) {
      const double o = bf16_to_f32(got[static_cast<size_t>(r) * kQHeads * kHd + i]);
      const double err = std::fabs(o - ref[i]);
      const double bound = std::ldexp(1.0, -8) * (1.0 + std::fabs(ref[i]));
      worst = std::max(worst, err / bound);
      if (!(err <= bound)) {
        check(false, what + ": row " + std::to_string(r) + " element " + std::to_string(i) + " error " +
                         std::to_string(err) + " above " + std::to_string(bound));
        return;
      }
    }
    ++checked;
  }
  std::printf("  %-40s %3d rows checked, worst error %.3f of the bound\n", what.c_str(), checked, worst);
}

}  // namespace sparse_test

#endif  // IGNIS_FLASH_NEXT_SPARSE_TEST_COMMON_H
