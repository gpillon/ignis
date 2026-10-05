// The Flash-Next QSA sparse attention (spec flash-next/04, GitHub #302, slice S3) at real
// geometry -- OURS (ADR 0043): 24 query heads over 2 KV heads of 256, every row attending to the
// tokens its selection lists (up to 2051), against fp64 attention masked to those tokens.
//
//   prefill  one lane, 300 rows at positions 5000..5299, each with its own 512 blocks + tail
//            (one split per row), and a call straddling the dense threshold (rows <= 2050 list
//            every visible token);
//   decode   three lanes of one token (a long sparse row, a short sparse row, a dense row): the
//            lists spread over 65 splits and merged; also captured in a CUDA graph and replayed;
//   index    the same decode rows read from a [rows][width][kv_heads][256] scratch by list index
//            (the hq-e8-2b route's source) must equal the paged run bit for bit.
//
// K, V and q are counter-hash BF16 values in [-1, 1) (q in [-3, 3) for peaked softmaxes); pages
// are permuted so the gather goes through the block table.
// Tolerance: S is exact BF16 products summed in fp32; P is rounded to BF16 (2^-9 relative per
// weight) and the output to BF16 (2^-9 of |out|), so |out - ref| <= 2^-9 max|v| + 2^-9 |ref| +
// fp32 noise; asserted as 2^-8 (max|v| <= 1) + 2^-8 |ref|.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "../src/flash_next/qsa_sparse.h"

#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace moe_test;
namespace fn = ignis::flash_next;
namespace sp = ignis::flash_next::sparse;

namespace {

constexpr int kQHeads = 24, kKvHeads = 2, kHd = 256, kWidth = 2051;
constexpr int kSeqTokens = 6000;
constexpr int kSlots = 3;
constexpr int kLogicalPages = (kSeqTokens + 63) / 64;
constexpr int kPhysicalPages = kSlots * kLogicalPages;
constexpr uint32_t kK = 0x5A00, kV = 0x5A01, kQ = 0x5A02, kPick = 0x5A03;

#define SP_OK(expr)                                                                                \
  do {                                                                                             \
    const sp::Status st_ = (expr);                                                                 \
    if (st_ != nullptr) {                                                                          \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, st_);                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

fn::Geometry geometry() {
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
std::vector<int32_t> make_list(int p, uint32_t salt) {
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
  for (int32_t b : ids) for (int t = 0; t < 4; ++t) out.push_back(4 * b + t);
  for (int t = blocks * 4; t <= p; ++t) out.push_back(t);
  return out;
}

struct Pages {
  std::vector<uint16_t> k, v;  // [physical page][kv head][64][256]
  std::vector<int32_t> tables;  // [slot][logical page]
  float kv(const std::vector<uint16_t> &plane, int slot, int pos, int head, int d) const {
    const int page = tables[slot * kLogicalPages + pos / 64];
    return bf16_to_f32(plane[((static_cast<size_t>(page) * kKvHeads + head) * 64 + pos % 64) * kHd + d]);
  }
};

Pages make_pages() {
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

// fp64 attention of one row (all query heads) over its list.
std::vector<double> reference(const Pages &pg, const std::vector<uint16_t> &q, int row, int slot,
                              const std::vector<int32_t> &list) {
  std::vector<double> out(static_cast<size_t>(kQHeads) * kHd, 0.0);
  std::vector<double> s(list.size());
  for (int h = 0; h < kQHeads; ++h) {
    const int kvh = h / 12;
    const uint16_t *qh = &q[(static_cast<size_t>(row) * kQHeads + h) * kHd];
    double m = -1e300;
    for (size_t j = 0; j < list.size(); ++j) {
      double dot = 0.0;
      for (int d = 0; d < kHd; ++d) dot += static_cast<double>(bf16_to_f32(qh[d])) * pg.kv(pg.k, slot, list[j], kvh, d);
      s[j] = dot / 16.0;
      m = std::max(m, s[j]);
    }
    double l = 0.0;
    for (size_t j = 0; j < list.size(); ++j) {
      s[j] = std::exp(s[j] - m);
      l += s[j];
    }
    for (size_t j = 0; j < list.size(); ++j) {
      for (int d = 0; d < kHd; ++d) out[static_cast<size_t>(h) * kHd + d] += s[j] / l * pg.kv(pg.v, slot, list[j], kvh, d);
    }
  }
  return out;
}

constexpr int kMaxRows = 300;

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
  std::vector<std::vector<int32_t>> lists;  // per row
};

// Uploads a call's lists and runs it; returns the BF16 output.
std::vector<uint16_t> run(Device &d, const fn::Geometry &g, const Call &c, const sp::KvSource &kv,
                          cudaStream_t stream, bool graph = false) {
  const int rows = static_cast<int>(c.lists.size());
  std::vector<int32_t> tokens(static_cast<size_t>(rows) * kWidth, -1), counts(rows);
  for (int r = 0; r < rows; ++r) {
    counts[r] = static_cast<int32_t>(c.lists[r].size());
    std::copy(c.lists[r].begin(), c.lists[r].end(), tokens.begin() + static_cast<size_t>(r) * kWidth);
  }
  MOE_CUDA(cudaMemcpy(d.tokens.p, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice));
  MOE_CUDA(cudaMemcpy(d.counts.p, counts.data(), counts.size() * 4, cudaMemcpyHostToDevice));
  MOE_CUDA(cudaMemcpy(d.slots.p, c.slots.data(), c.slots.size() * 4, cudaMemcpyHostToDevice));
  MOE_CUDA(cudaMemcpy(d.positions.p, c.positions.data(), c.positions.size() * 4, cudaMemcpyHostToDevice));
  fn::Batch b;
  b.lanes = static_cast<int32_t>(c.slots.size());
  b.tokens = c.tokens;
  b.slots = d.slots.as<int32_t>();
  b.positions = d.positions.as<int32_t>();
  b.max_visible = kSeqTokens;
  fn::Selection sel{d.tokens.as<int32_t>(), d.counts.as<int32_t>()};
  MOE_CUDA(cudaMemset(d.out.p, 0xFF, static_cast<size_t>(rows) * kQHeads * kHd * 2));
  if (graph) {
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ex = nullptr;
    MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, stream));
    MOE_CUDA(cudaStreamEndCapture(stream, &gr));
    MOE_CUDA(cudaGraphInstantiate(&ex, gr, 0));
    MOE_CUDA(cudaGraphLaunch(ex, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    MOE_CUDA(cudaGraphExecDestroy(ex));
    MOE_CUDA(cudaGraphDestroy(gr));
  } else {
    SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
  }
  return download<uint16_t>(d.out.p, static_cast<size_t>(rows) * kQHeads * kHd);
}

void check_call(const std::string &what, const Pages &pg, const std::vector<uint16_t> &q, const Call &c,
                const std::vector<uint16_t> &got, int sample_step) {
  const int rows = static_cast<int>(c.lists.size());
  double worst = 0.0;
  int checked = 0;
  for (int r = 0; r < rows; r += sample_step) {
    const int slot = c.slots[r / c.tokens];
    const auto ref = reference(pg, q, r, slot, c.lists[r]);
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
  std::printf("  %-34s %3d rows checked, worst error %.3f of the bound\n", what.c_str(), checked, worst);
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  const fn::Geometry g = geometry();
  std::printf("Flash-Next QSA sparse attention: 24 q / 2 KV heads of 256, lists of up to 2051 tokens\n");
  const Pages pg = make_pages();
  std::vector<uint16_t> q(static_cast<size_t>(kMaxRows) * kQHeads * kHd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16(hash_uniform(kQ, i, 3.0F));

  DeviceBytes dk(pg.k.size() * 2), dv(pg.v.size() * 2), dtables(pg.tables.size() * 4);
  upload(dk, pg.k);
  upload(dv, pg.v);
  upload(dtables, pg.tables);
  Device d(std::max<size_t>(sp::partial_bytes(g, 3), 256));
  upload(d.q, q);
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  sp::KvSource paged;
  paged.k = dk.as<__nv_bfloat16>();
  paged.v = dv.as<__nv_bfloat16>();
  paged.block_tables = dtables.as<int32_t>();
  paged.logical_pages = kLogicalPages;
  paged.kv_heads = kKvHeads;

  check(sp::splits_for(g, 300) == 1, "a 300-row call must not split");
  check(sp::splits_for(g, 3) == 65, "a 3-row call must spread each list over 65 splits");

  // Prefill-shaped: one lane, its own list per row.
  for (int first : {5000, 1900}) {
    Call c;
    c.slots = {0};
    c.positions = {first};
    c.tokens = kMaxRows;
    for (int r = 0; r < kMaxRows; ++r) c.lists.push_back(make_list(first + r, 0));
    const auto got = run(d, g, c, paged, stream);
    check_call("prefill rows " + std::to_string(first) + "..", pg, q, c, got, 7);
  }

  // Decode-shaped: three lanes, one token each, split over 65 CTAs per (row, KV head).
  Call dec;
  dec.slots = {0, 1, 2};
  dec.positions = {5999, 2100, 1000};
  dec.tokens = 1;
  for (int l = 0; l < 3; ++l) dec.lists.push_back(make_list(dec.positions[l], static_cast<uint32_t>(l + 1)));
  const auto eager = run(d, g, dec, paged, stream);
  check_call("decode 3 lanes, 65 splits", pg, q, dec, eager, 1);
  check(run(d, g, dec, paged, stream, true) == eager, "decode: the graph replay differs from eager");

  // The by-index source: the same rows' listed K/V copied into [rows][width][kv_heads][256].
  {
    const size_t n = static_cast<size_t>(3) * kWidth * kKvHeads * kHd;
    std::vector<uint16_t> sk(n, 0), sv(n, 0);
    for (int r = 0; r < 3; ++r) {
      for (size_t i = 0; i < dec.lists[r].size(); ++i) {
        for (int h = 0; h < kKvHeads; ++h) {
          const int page = pg.tables[dec.slots[r] * kLogicalPages + dec.lists[r][i] / 64];
          const size_t src = ((static_cast<size_t>(page) * kKvHeads + h) * 64 + dec.lists[r][i] % 64) * kHd;
          const size_t dst = ((static_cast<size_t>(r) * kWidth + i) * kKvHeads + h) * kHd;
          std::copy(&pg.k[src], &pg.k[src] + kHd, &sk[dst]);
          std::copy(&pg.v[src], &pg.v[src] + kHd, &sv[dst]);
        }
      }
    }
    DeviceBytes scratch_k(n * 2), scratch_v(n * 2);
    upload(scratch_k, sk);
    upload(scratch_v, sv);
    sp::KvSource by_index;
    by_index.k = scratch_k.as<__nv_bfloat16>();
    by_index.v = scratch_v.as<__nv_bfloat16>();
    by_index.kv_heads = kKvHeads;
    by_index.by_index = true;
    check(run(d, g, dec, by_index, stream) == eager, "by-index source differs from the paged run");
  }

  // Refusals.
  {
    fn::Selection dense;
    dense.dense = true;
    fn::Batch b;
    b.lanes = 1;
    b.tokens = 1;
    check(sp::attend(g, paged, b, d.q.as<__nv_bfloat16>(), dense, d.out.as<__nv_bfloat16>(), nullptr, stream) != nullptr,
          "a dense selection must be refused");
    fn::Geometry wrong = g;
    wrong.q_heads = 16;
    check(sp::check_geometry(wrong) != nullptr, "a GQA group other than 12 must be refused");
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_sparse_attention: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_flash_next_sparse_attention: OK\n");
  return 0;
}
