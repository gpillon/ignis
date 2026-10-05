// GitHub #302 (spec flash-next/04, slice S2): OURS -- the Flash-Next QSA attention sublayer
// (kernel/src/flash_next/qsa.cu, qsa_dense.cu) at the real geometry (hidden 2560, 24 query heads,
// 2 KV heads of 256, rotary 64, FP8 projections), on paged K/V of both formats.
//
// Arms:
//   prepare     the norms and rope against a restatement of the checkpoint's BF16 arithmetic.
//   dense       the dense kernel alone, on the pages and on a by-position scratch (bit-identical),
//               against fp64 attention with the bound of its own roundings.
//   layer       qsa::run, three lanes in a paged store per format: one-shot and chunked prefill
//               (a chunk shorter than the hq ring on >= 1024 history, one longer than the ring
//               up to dense_threshold()), decode rounds on S3's sparse route -- the first eager,
//               two replaying captured graphs, the last with the lanes in another order -- and a
//               sparse prefill chunk past dense_threshold(). Every call's K/V rows in the pages
//               against K/V recomputed from x; every BF16 call against fp64 attention over those
//               recomputed K/V on sampled rows; every hq-e8-2b call bit-identical to BF16 while
//               every key it reads is the call's own, else against fp64 attention over the very
//               rows it reads (decoded beforehand from the same store state and the call's own
//               rows), and against the BF16 route within the difference the measured codec error
//               makes between the two fp64 references; the residual window's rows exact after a
//               chunk wider than the ring.
//   entry       fn_qsa_attention itself on seq pools of both formats: bit-identical to qsa::run,
//               its K/V in its own attention layer's planes and window only, its refusals.
//
// Bounds. Attention on the same q and K/V: the probabilities are rounded to BF16 for the value
// product while the normalizer is their fp32 sum, and the output is rounded once, so
// |o - o_ref| <= 3 * 2^-9 * A with A = sum_j p_j |v_j| (stated as 2^-7 A; 2^-6 A for the whole
// sublayer, which adds rare BF16 rounding flips of q against the restatement). The gate and
// o_proj propagate it elementwise: y's bound is sum_c |W_ic| B_c + 2^-8 |y_i|, B_c the gated
// element's. The hq route is held to the same bound against the fp64 attention over the rows it
// reads, and to the BF16 route within |y_ref(read rows) - y_ref(exact rows)| plus both bounds:
// the tolerance is the codec error measured on those rows, carried through the same fp64 path.
//
// K rows against the restatement: a projection value one BF16 flip apart (the FP8 linear's fp32
// sum against fp64) moves a rope output by one rounding of its larger product, which can be many
// ulps of a small output after cancellation: the check allows 2^-6 of the head row's largest
// value, and requires 99% of the values bit-exact.

#include "flash_next/qsa.h"

#include "flash_next_s2_test_common.h"
#include "ignis_fp8_linear.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"
#include "ninfer/ops/rope.h"

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
namespace qsa = ignis::flash_next::qsa;
namespace sparse = ignis::flash_next::sparse;

namespace {

constexpr int H = 2560;
constexpr int QH = qsa::kQHeads;
constexpr int KVH = qsa::kKvHeads;
constexpr int D = qsa::kHeadDim;
constexpr int kSlots = 3;
constexpr int kLogicalPages = 40;  // 2560 tokens a lane
constexpr int kPages = 128;
constexpr int kWidth = 2051;       // dense_threshold() = selection_width()
constexpr int kSparseRows = 40;    // the sparse prefill chunk
constexpr double kScale = 1.0 / 16.0;
constexpr float kEps = 1e-6F;

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = H;
  g.q_heads = QH;
  g.kv_heads = KVH;
  g.head_dim = D;
  g.rotary_dim = qsa::kRotaryDim;
  g.compress_ratio = 4;
  g.indexer_budget = 2048;
  g.rms_norm_eps = kEps;
  return g;
}

struct Weights {
  Proj q, k, v, o;
  std::vector<uint16_t> q_norm, k_norm;
  std::unique_ptr<DeviceBytes> d_q_norm, d_k_norm;
  fn::QsaWeights view() const {
    fn::QsaWeights w;
    w.q_proj = linear(q);
    w.k_proj = linear(k);
    w.v_proj = linear(v);
    w.o_proj = linear(o);
    w.q_norm = d_q_norm->p;
    w.k_norm = d_k_norm->p;
    return w;
  }
};

void build(Weights &w) {
  const double x_rms = 1.0 / std::sqrt(3.0);
  make_proj(w.q, 201, qsa::kQProjWidth, H, x_rms, 1.0);
  make_proj(w.k, 203, qsa::kKvWidth, H, x_rms, 1.0);
  make_proj(w.v, 205, qsa::kKvWidth, H, x_rms, 1.0);
  make_proj(w.o, 207, H, qsa::kOutWidth, 0.3, 1.0);
  w.q_norm = bf16_vector(209, D, -0.25, 0.25);
  w.k_norm = bf16_vector(211, D, -0.25, 0.25);
  w.d_q_norm = device_copy(w.q_norm);
  w.d_k_norm = device_copy(w.k_norm);
}

// ---- the checkpoint's norm + rope, restated -------------------------------------------------

struct Rope {
  float inv_freq[32];
};

// (1 + w) RMSNorm rounded once, then rope on pairs (i, i + 32) of the first 64 in BF16 arithmetic.
std::vector<double> norm_rope(const double *x, const std::vector<uint16_t> &w, const Rope &rope, int32_t position) {
  double ss = 0.0;
  for (int d = 0; d < D; ++d) ss += x[d] * x[d];
  const double r = 1.0 / std::sqrt(ss / D + kEps);
  std::vector<double> n(D);
  for (int d = 0; d < D; ++d) n[d] = bf16r(x[d] * r * (1.0 + bf16_to_f32(w[d])));
  for (int i = 0; i < 32; ++i) {
    const float phi = static_cast<float>(position) * rope.inv_freq[i];
    const double c = bf16r(std::cos(static_cast<double>(phi)));
    const double s = bf16r(std::sin(static_cast<double>(phi)));
    const double x1 = n[i], x2 = n[i + 32];
    n[i] = bf16r(bf16r(x1 * c) + bf16r(-x2 * s));
    n[i + 32] = bf16r(bf16r(x2 * c) + bf16r(x1 * s));
  }
  return n;
}

// ---- inputs, the restated K/V and the paged stores ------------------------------------------

std::vector<uint16_t> token_input(int lane, int32_t position) {
  return bf16_vector(3000 + lane, H, -1.0, 1.0, static_cast<uint64_t>(position) * H);
}

// A lane's K and V rows recomputed from its inputs: [position][KVH][D] each, fp64 of BF16 values.
struct RefKv {
  std::vector<double> k, v;
  void append(const Weights &w, const Rope &rope, int lane, int32_t position) {
    const std::vector<uint16_t> xb = token_input(lane, position);
    const std::vector<double> x = as_double(xb.data(), xb.size());
    std::vector<double> kp = project(w.k, x), vp = project(w.v, x);
    for (double &e : kp) e = bf16r(e);
    for (int h = 0; h < KVH; ++h) {
      const std::vector<double> kr = norm_rope(&kp[static_cast<std::size_t>(h) * D], w.k_norm, rope, position);
      k.insert(k.end(), kr.begin(), kr.end());
    }
    for (double e : vp) v.push_back(bf16r(e));
  }
};

struct Store {
  int32_t format = IGNIS_KV_FORMAT_BF16;
  std::unique_ptr<DeviceBytes> k, v, k_meta, v_meta, res_k, res_v, ring;
  explicit Store(int32_t f) : format(f) {
    const std::size_t rows = static_cast<std::size_t>(kPages) * KVH * 64;
    const std::size_t row_bytes = f == IGNIS_KV_FORMAT_BF16 ? D * 2 : 64;
    for (auto *p : {&k, &v}) *p = std::make_unique<DeviceBytes>(rows * row_bytes);
    if (f == IGNIS_KV_FORMAT_HQ_E8_2B) {
      for (auto *p : {&k_meta, &v_meta}) *p = std::make_unique<DeviceBytes>(rows * 8);
      for (auto *p : {&res_k, &res_v}) *p = std::make_unique<DeviceBytes>(static_cast<std::size_t>(kSlots) * 544 * KVH * D * 2);
      ring = std::make_unique<DeviceBytes>(static_cast<std::size_t>(kSlots) * 16 * 4);
    }
    for (auto *p : {&k, &v, &k_meta, &v_meta, &res_k, &res_v, &ring}) {
      if (*p) MOE_CUDA(cudaMemset((*p)->p, 0, (*p)->bytes));
    }
  }
  qsa::Kv kv(const int32_t *tables) const {
    qsa::Kv out;
    out.kv_format = format;
    out.block_tables = tables;
    out.logical_pages = kLogicalPages;
    out.slots = kSlots;
    out.k = k->p;
    out.v = v->p;
    if (format == IGNIS_KV_FORMAT_HQ_E8_2B) {
      out.k_meta = k_meta->p;
      out.v_meta = v_meta->p;
      out.residual_k = res_k->as<__nv_bfloat16>();
      out.residual_v = res_v->as<__nv_bfloat16>();
      out.ring = ring->as<uint32_t>();
    }
    return out;
  }
  sparse::HqSource hq(const int32_t *tables) const {
    sparse::HqSource s;
    s.k_codes = k->as<uint8_t>();
    s.k_meta = k_meta->as<uint8_t>();
    s.v_codes = v->as<uint8_t>();
    s.v_meta = v_meta->as<uint8_t>();
    s.block_tables = tables;
    s.logical_pages = kLogicalPages;
    s.kv_heads = KVH;
    s.residual_k = res_k->as<__nv_bfloat16>();
    s.residual_v = res_v->as<__nv_bfloat16>();
    s.ring_valid = ring->as<uint32_t>();
    return s;
  }
};

// Rows [0, n) of the lane at block-table row `slot` of a BF16 plane, [n][KVH][D] in fp64.
std::vector<double> plane_rows(const void *plane_dev, std::size_t plane_pages, const std::vector<int32_t> &tables,
                               int logical_pages, int slot, int n) {
  const std::vector<uint16_t> plane = download<uint16_t>(plane_dev, plane_pages * KVH * 64 * D);
  std::vector<double> rows(static_cast<std::size_t>(n) * KVH * D);
  for (int j = 0; j < n; ++j) {
    const int page = tables[static_cast<std::size_t>(slot) * logical_pages + j / 64];
    for (int h = 0; h < KVH; ++h) {
      const std::size_t at = ((static_cast<std::size_t>(page) * KVH + h) * 64 + j % 64) * D;
      for (int d = 0; d < D; ++d) rows[(static_cast<std::size_t>(j) * KVH + h) * D + d] = bf16_to_f32(plane[at + d]);
    }
  }
  return rows;
}

std::vector<double> lane_rows(const Store &s, const std::vector<int32_t> &tables, int slot, int n, bool role_v) {
  return plane_rows((role_v ? s.v : s.k)->p, kPages, tables, kLogicalPages, slot, n);
}

// Rows [first, first + count) against the restated ones: every value within 2^-6 of its head
// row's largest (a flip moved through rope), 99% bit-exact.
void check_rows(const std::string &name, const std::vector<double> &got, const std::vector<double> &want,
                int32_t first, int32_t count) {
  long exact = 0, total = 0;
  bool within = true;
  for (std::size_t row = static_cast<std::size_t>(first) * KVH; row < static_cast<std::size_t>(first + count) * KVH; ++row) {
    double peak = 0.0;
    for (int d = 0; d < D; ++d) peak = std::max(peak, std::fabs(want[row * D + d]));
    for (int d = 0; d < D; ++d) {
      const double e = std::fabs(got[row * D + d] - want[row * D + d]);
      exact += e == 0.0;
      ++total;
      within = within && e <= std::ldexp(peak, -6);
    }
  }
  check(within, name + ": every value within 2^-6 of its head row's largest of the K/V recomputed from x");
  check(exact * 100 >= total * 99, name + ": 99% of the K/V values bit-exact");
}

// ---- the reference of one row and its bounds ------------------------------------------------

struct RowRef {
  std::vector<double> y;      // fp64, unrounded
  std::vector<double> bound;  // a route reading the same K/V rows, against y
};

// sum_c |W[i][c]| b[c] for every output i.
std::vector<double> project_abs(const Proj &p, const std::vector<double> &b) {
  const double *lut = e4m3_lut();
  std::vector<double> y(p.rows);
  for (int r = 0; r < p.rows; ++r) {
    const uint8_t *row = &p.payload[static_cast<std::size_t>(r) * p.cols];
    double s = 0.0;
    for (int c = 0; c < p.cols; ++c) s += std::fabs(lut[row[c]]) * b[c];
    y[r] = s * std::fabs(p.scale[r]);
  }
  return y;
}

// One query row at `position` attending `keys` (ascending) of the K/V rows k, v ([n][KVH][D],
// indexed by position): y in fp64 and the bound of a route that reads these rows against it.
RowRef reference_row(const Weights &w, const Rope &rope, const std::vector<uint16_t> &x_bf16, int32_t position,
                     const std::vector<int32_t> &keys, const std::vector<double> &k, const std::vector<double> &v) {
  const std::vector<double> x = as_double(x_bf16.data(), x_bf16.size());
  std::vector<double> qg = project(w.q, x);
  for (double &e : qg) e = bf16r(e);
  std::vector<double> gated(qsa::kOutWidth), b(qsa::kOutWidth);
  for (int h = 0; h < QH; ++h) {
    const std::vector<double> q = norm_rope(&qg[static_cast<std::size_t>(h) * 2 * D], w.q_norm, rope, position);
    const int kvh = h / qsa::kGroup;
    std::vector<double> p(keys.size());
    double m = -1e300;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const std::size_t row = (static_cast<std::size_t>(keys[i]) * KVH + kvh) * D;
      double dot = 0.0;
      for (int d = 0; d < D; ++d) dot += q[d] * k[row + d];
      p[i] = dot * kScale;
      m = std::max(m, p[i]);
    }
    double l = 0.0;
    for (double &e : p) l += (e = std::exp(e - m));
    for (double &e : p) e /= l;
    for (int d = 0; d < D; ++d) {
      double o = 0.0, a = 0.0;
      for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::size_t at = (static_cast<std::size_t>(keys[i]) * KVH + kvh) * D + d;
        o += p[i] * v[at];
        a += p[i] * std::fabs(v[at]);
      }
      const std::size_t c = static_cast<std::size_t>(h) * D + d;
      const double sg = bf16r(sigmoid(qg[static_cast<std::size_t>(h) * 2 * D + D + d]));
      gated[c] = bf16r(bf16r(o) * sg);
      b[c] = sg * std::ldexp(a, -6) + std::ldexp(std::fabs(gated[c]), -8);
    }
  }
  RowRef r;
  r.y = project(w.o, gated);
  r.bound = project_abs(w.o, b);
  for (int i = 0; i < H; ++i) r.bound[i] += std::ldexp(std::fabs(r.y[i]), -8) + 1e-6;
  return r;
}

// Elementwise |got - want| <= bound over rows; reports the worst ratio.
void check_bounded(const std::string &name, const std::vector<std::vector<double>> &got,
                   const std::vector<std::vector<double>> &want, const std::vector<std::vector<double>> &bound) {
  double worst = 0.0, num = 0.0, den = 0.0;
  bool ok = true;
  for (std::size_t r = 0; r < got.size(); ++r) {
    for (std::size_t i = 0; i < got[r].size(); ++i) {
      const double e = std::fabs(got[r][i] - want[r][i]);
      worst = std::max(worst, e / bound[r][i]);
      ok = ok && e <= bound[r][i];
      num += e * e;
      den += want[r][i] * want[r][i];
    }
  }
  std::printf("  %s: %zu rows, relative L2 %.3e, worst %.3f of the bound\n", name.c_str(), got.size(),
              std::sqrt(num / den), worst);
  check(ok, name + ": every element within its bound");
}

std::vector<double> row_of(const std::vector<uint16_t> &y, int row) {
  return as_double(&y[static_cast<std::size_t>(row) * H], H);
}

// The visible tokens a hand-made selection keeps at `position`: everything while at most kWidth
// are visible, else the first 64, every third and the last 48 (always the row's own token).
std::vector<int32_t> selection_at(int32_t position) {
  std::vector<int32_t> list;
  for (int32_t j = 0; j <= position; ++j) {
    if (position < kWidth || j < 64 || j % 3 == 0 || j > position - 48) list.push_back(j);
  }
  return list;
}

}  // namespace

int main() {
  MOE_CUDA(cudaSetDevice(0));
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return EXIT_FAILURE;
  }
  const fn::Geometry g = geometry();
  check(qsa::check_geometry(g) == nullptr, "the real geometry is accepted");
  {
    fn::Geometry other = g;
    other.kv_heads = 4;
    check(qsa::check_geometry(other) != nullptr, "another KV head count is refused");
  }
  const ninfer::ops::RopeFrequencies frequencies = ninfer::ops::rope_linear_frequencies(1e7F, 64);
  const fn::indexer::Rope ix_rope = fn::indexer::rope_from(frequencies);
  Rope rope;
  std::memcpy(rope.inv_freq, ix_rope.inv_freq, sizeof(rope.inv_freq));
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  Weights w;
  build(w);
  const fn::QsaWeights wv = w.view();

  // Block tables: every lane's logical pages on distinct, scattered physical pages.
  std::vector<int32_t> tables(static_cast<std::size_t>(kSlots) * kLogicalPages);
  {
    std::vector<int32_t> perm(kPages);
    for (int i = 0; i < kPages; ++i) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [](int32_t a, int32_t b) { return hash_u32(77, a) < hash_u32(77, b); });
    for (std::size_t i = 0; i < tables.size(); ++i) tables[i] = perm[i];
  }
  DeviceBytes d_tables(tables.size() * 4);
  upload(d_tables, tables);
  const int32_t *dt = d_tables.as<int32_t>();
  DeviceBytes d_slots(16 * 4), d_positions(16 * 4);

  // ---- prepare: five rows at scattered positions --------------------------------------------
  {
    const int rows = 5;
    const int32_t positions[rows] = {0, 1, 4097, 65535, 131071};
    std::vector<uint16_t> qg = bf16_vector(301, static_cast<std::size_t>(rows) * qsa::kQProjWidth, -2.0, 2.0);
    std::vector<uint16_t> k = bf16_vector(303, static_cast<std::size_t>(rows) * qsa::kKvWidth, -2.0, 2.0);
    auto d_qg = device_copy(qg);
    auto d_k = device_copy(k);
    DeviceBytes d_q(static_cast<std::size_t>(rows) * qsa::kOutWidth * 2);
    upload(d_positions, std::vector<int32_t>(positions, positions + rows));
    MOE_CUDA(cudaDeviceSynchronize());
    fn::Batch batch;
    batch.lanes = rows;
    batch.tokens = 1;
    batch.positions = d_positions.as<int32_t>();
    check(qsa::prepare(g, ix_rope, wv.q_norm, wv.k_norm, batch, d_qg->as<__nv_bfloat16>(), d_q.as<__nv_bfloat16>(),
                       d_k->as<__nv_bfloat16>(), stream) == nullptr,
          "prepare launches");
    MOE_CUDA(cudaStreamSynchronize(stream));
    const std::vector<uint16_t> q_out = download<uint16_t>(d_q.p, static_cast<std::size_t>(rows) * qsa::kOutWidth);
    const std::vector<uint16_t> k_out = download<uint16_t>(d_k->p, static_cast<std::size_t>(rows) * qsa::kKvWidth);
    long exact = 0, total = 0;
    bool within = true;
    auto compare = [&](const uint16_t *got, const std::vector<double> &want) {
      for (int d = 0; d < D; ++d) {
        const double e = std::fabs(bf16_to_f32(got[d]) - want[d]);
        exact += e == 0.0;
        ++total;
        within = within && e <= std::ldexp(std::fabs(want[d]), -6) + 1e-3;
      }
    };
    for (int r = 0; r < rows; ++r) {
      for (int h = 0; h < QH; ++h) {
        const std::vector<double> x = as_double(&qg[(static_cast<std::size_t>(r) * QH + h) * 2 * D], D);
        compare(&q_out[(static_cast<std::size_t>(r) * QH + h) * D], norm_rope(x.data(), w.q_norm, rope, positions[r]));
      }
      for (int h = 0; h < KVH; ++h) {
        const std::vector<double> x = as_double(&k[(static_cast<std::size_t>(r) * KVH + h) * D], D);
        compare(&k_out[(static_cast<std::size_t>(r) * KVH + h) * D], norm_rope(x.data(), w.k_norm, rope, positions[r]));
      }
    }
    std::printf("  prepare: %ld/%ld values bit-exact\n", exact, total);
    check(within, "prepare: every value within a BF16 rounding flip of the restatement");
    check(exact * 1000 >= total * 995, "prepare: 99.5% of the values bit-exact");
  }

  // ---- dense: the kernel alone, pages and by-position, against fp64 ------------------------
  {
    Store pages(IGNIS_KV_FORMAT_BF16);
    const int32_t n = kWidth;
    std::vector<uint16_t> k = bf16_vector(401, static_cast<std::size_t>(n) * qsa::kKvWidth, -1.0, 1.0);
    std::vector<uint16_t> v = bf16_vector(403, static_cast<std::size_t>(n) * qsa::kKvWidth, -1.0, 1.0);
    auto d_k = device_copy(k);
    auto d_v = device_copy(v);
    // All n rows into lane 1's pages, one append.
    upload(d_slots, std::vector<int32_t>{1});
    upload(d_positions, std::vector<int32_t>{0});
    MOE_CUDA(cudaDeviceSynchronize());
    fn::Batch fill;
    fill.lanes = 1;
    fill.tokens = n;
    fill.slots = d_slots.as<int32_t>();
    fill.positions = d_positions.as<int32_t>();
    check(qsa::append(g, pages.kv(dt), fill, d_k->as<__nv_bfloat16>(), d_v->as<__nv_bfloat16>(), stream) == nullptr,
          "dense: append launches");
    MOE_CUDA(cudaStreamSynchronize(stream));
    const std::vector<double> kh = as_double(k.data(), k.size()), vh = as_double(v.data(), v.size());
    check(lane_rows(pages, tables, 1, n, false) == kh && lane_rows(pages, tables, 1, n, true) == vh,
          "dense: the BF16 append stores every row bit for bit");
    sparse::KvSource paged{pages.k->as<__nv_bfloat16>(), pages.v->as<__nv_bfloat16>(), dt, kLogicalPages, KVH,
                           sparse::KvSource::Mode::Paged};
    sparse::KvSource by_position{d_k->as<__nv_bfloat16>(), d_v->as<__nv_bfloat16>(), nullptr, 0, KVH,
                                 sparse::KvSource::Mode::ByPosition};
    for (auto [first, tokens] : {std::pair{0, 1}, std::pair{0, 77}, std::pair{1000, 130}, std::pair{1900, 151}}) {
      std::vector<uint16_t> q = bf16_vector(405 + first, static_cast<std::size_t>(tokens) * qsa::kOutWidth, -3.0, 3.0);
      auto d_q = device_copy(q);
      DeviceBytes d_o1(q.size() * 2), d_o2(q.size() * 2);
      upload(d_positions, std::vector<int32_t>{first});
      MOE_CUDA(cudaDeviceSynchronize());
      fn::Batch batch;
      batch.lanes = 1;
      batch.tokens = tokens;
      batch.slots = d_slots.as<int32_t>();
      batch.positions = d_positions.as<int32_t>();
      batch.max_visible = first + tokens;
      check(qsa::attend_dense(g, paged, kSlots, batch, d_q->as<__nv_bfloat16>(), d_o1.as<__nv_bfloat16>(), stream) ==
                    nullptr &&
                qsa::attend_dense(g, by_position, kSlots, batch, d_q->as<__nv_bfloat16>(), d_o2.as<__nv_bfloat16>(),
                                  stream) == nullptr,
            "dense: launches");
      MOE_CUDA(cudaStreamSynchronize(stream));
      const std::vector<uint16_t> o1 = download<uint16_t>(d_o1.p, q.size());
      const std::vector<uint16_t> o2 = download<uint16_t>(d_o2.p, q.size());
      const std::string name = "dense P=" + std::to_string(first) + " T=" + std::to_string(tokens);
      check(o1 == o2, name + ": pages and by-position scratch bit-identical");
      bool ok = true;
      double worst = 0.0;
      for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < QH; ++h) {
          const int kvh = h / qsa::kGroup;
          const uint16_t *qr = &q[(static_cast<std::size_t>(t) * QH + h) * D];
          const int keys = first + t + 1;
          std::vector<double> p(keys);
          double m = -1e300, l = 0.0;
          for (int j = 0; j < keys; ++j) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) dot += bf16_to_f32(qr[d]) * kh[(static_cast<std::size_t>(j) * KVH + kvh) * D + d];
            p[j] = dot * kScale;
            m = std::max(m, p[j]);
          }
          for (int j = 0; j < keys; ++j) l += (p[j] = std::exp(p[j] - m));
          for (int d = 0; d < D; ++d) {
            double o = 0.0, a = 0.0;
            for (int j = 0; j < keys; ++j) {
              const double vv = vh[(static_cast<std::size_t>(j) * KVH + kvh) * D + d];
              o += p[j] * vv;
              a += p[j] * std::fabs(vv);
            }
            const double e = std::fabs(bf16_to_f32(o1[(static_cast<std::size_t>(t) * QH + h) * D + d]) - o / l);
            const double bound = std::ldexp(a / l, -7) + 1e-30;
            worst = std::max(worst, e / bound);
            ok = ok && e <= bound;
          }
        }
      }
      std::printf("  %s: worst %.3f of the bound 2^-7 sum p|v|\n", name.c_str(), worst);
      check(ok, name + ": every output within 2^-7 sum_j p_j |v_j| of fp64");
    }
  }

  // ---- layer: three lanes, both formats -----------------------------------------------------
  Store bf16_store(IGNIS_KV_FORMAT_BF16), hq_store(IGNIS_KV_FORMAT_HQ_E8_2B);
  const qsa::Kv kv_bf16 = bf16_store.kv(dt), kv_hq = hq_store.kv(dt);
  constexpr int kMaxRows = 1200;
  ninfer::DeviceArena arena(fn::fn_qsa_attention_scratch_bytes(g, kMaxRows));
  DeviceBytes d_x(static_cast<std::size_t>(kMaxRows) * H * 2);
  DeviceBytes d_y_bf16(static_cast<std::size_t>(kMaxRows) * H * 2), d_y_hq(static_cast<std::size_t>(kMaxRows) * H * 2);
  DeviceBytes d_sel_tokens(static_cast<std::size_t>(kSparseRows) * kWidth * 4), d_sel_counts(kSparseRows * 4);
  const std::size_t decoded_bytes = std::max(sparse::listed_hq_bytes(g, kSparseRows), sparse::visible_hq_bytes(g, kWidth));
  DeviceBytes d_dec_k(decoded_bytes), d_dec_v(decoded_bytes);
  DeviceBytes d_fresh_k(static_cast<std::size_t>(kMaxRows) * qsa::kKvWidth * 2);
  DeviceBytes d_fresh_v(static_cast<std::size_t>(kMaxRows) * qsa::kKvWidth * 2);
  const int slot_of[kSlots] = {2, 0, 1};
  int32_t frontier[kSlots] = {0, 0, 0};
  RefKv ref[kSlots];

  // One call of both formats. `lists` holds each row's selection, else the call is dense.
  // `check_at`: the call's rows checked against fp64; `same`: the formats must agree bit for bit.
  auto call = [&](const std::string &name, const std::vector<int> &lanes, int tokens,
                  const std::vector<std::vector<int32_t>> *lists, const std::vector<int> &check_at, bool same,
                  cudaGraphExec_t *graphs) {
    const int rows = static_cast<int>(lanes.size()) * tokens;
    std::vector<uint16_t> x(static_cast<std::size_t>(rows) * H);
    std::vector<int32_t> slots, firsts;
    for (std::size_t l = 0; l < lanes.size(); ++l) {
      slots.push_back(slot_of[lanes[l]]);
      firsts.push_back(frontier[lanes[l]]);
      for (int t = 0; t < tokens; ++t) {
        const std::vector<uint16_t> xt = token_input(lanes[l], frontier[lanes[l]] + t);
        std::copy(xt.begin(), xt.end(), x.begin() + (l * tokens + t) * static_cast<std::size_t>(H));
        ref[lanes[l]].append(w, rope, lanes[l], frontier[lanes[l]] + t);
      }
    }
    upload(d_x, x);
    upload(d_slots, slots);
    upload(d_positions, firsts);
    fn::Batch batch;
    batch.lanes = static_cast<int32_t>(lanes.size());
    batch.tokens = tokens;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    fn::Selection selection;
    selection.dense = lists == nullptr;
    batch.max_visible = 0;
    for (std::size_t l = 0; l < lanes.size(); ++l) batch.max_visible = std::max(batch.max_visible, firsts[l] + tokens);
    if (lists != nullptr) {
      std::vector<int32_t> tok(static_cast<std::size_t>(rows) * kWidth, -1), counts(rows);
      for (int r = 0; r < rows; ++r) {
        std::copy((*lists)[r].begin(), (*lists)[r].end(), tok.begin() + static_cast<std::size_t>(r) * kWidth);
        counts[r] = static_cast<int32_t>((*lists)[r].size());
      }
      upload(d_sel_tokens, tok);
      upload(d_sel_counts, counts);
      selection.tokens = d_sel_tokens.as<int32_t>();
      selection.counts = d_sel_counts.as<int32_t>();
    }
    MOE_CUDA(cudaDeviceSynchronize());  // pageable uploads may still be in flight for another stream
    // The hq call's rows, decoded before it runs from the same store state with its own K/V as
    // the fresh rows -- the BF16 call's, computed by the same kernels from the same inputs: exactly
    // what the hq call reads.
    std::vector<double> kd, vd;
    auto decode_before_hq = [&]() {
      std::vector<uint16_t> fk, fv;
      for (std::size_t l = 0; l < lanes.size(); ++l) {
        const int n = firsts[l] + tokens;
        for (auto [rows_of, out] : {std::pair{lane_rows(bf16_store, tables, slot_of[lanes[l]], n, false), &fk},
                                    std::pair{lane_rows(bf16_store, tables, slot_of[lanes[l]], n, true), &fv}}) {
          for (std::size_t i = static_cast<std::size_t>(firsts[l]) * qsa::kKvWidth; i < static_cast<std::size_t>(n) * qsa::kKvWidth; ++i) {
            out->push_back(f32_to_bf16(static_cast<float>(rows_of[i])));
          }
        }
      }
      upload(d_fresh_k, fk);
      upload(d_fresh_v, fv);
      MOE_CUDA(cudaDeviceSynchronize());
      sparse::HqSource src = hq_store.hq(dt);
      src.fresh_k = d_fresh_k.as<__nv_bfloat16>();
      src.fresh_v = d_fresh_v.as<__nv_bfloat16>();
      sparse::KvSource out;
      const bool listed = lists != nullptr && tokens == 1;
      const sparse::Status st =
          listed ? sparse::decode_listed_hq(g, src, batch, selection, d_dec_k.as<__nv_bfloat16>(), d_dec_v.as<__nv_bfloat16>(),
                                            &out, stream)
                 : sparse::decode_visible_hq(g, src, batch, d_dec_k.as<__nv_bfloat16>(), d_dec_v.as<__nv_bfloat16>(), &out,
                                             stream);
      check(st == nullptr, name + ": decoding the hq call's rows launches");
      MOE_CUDA(cudaStreamSynchronize(stream));
      const std::size_t n = listed ? static_cast<std::size_t>(rows) * kWidth : static_cast<std::size_t>(batch.max_visible);
      kd = as_double(download<uint16_t>(d_dec_k.p, n * KVH * D).data(), n * KVH * D);
      vd = as_double(download<uint16_t>(d_dec_v.p, n * KVH * D).data(), n * KVH * D);
    };
    for (int f = 0; f < 2; ++f) {
      void *y = f == 0 ? d_y_bf16.p : d_y_hq.p;
      if (f == 1 && !same) decode_before_hq();
      if (graphs != nullptr) {
        MOE_CUDA(cudaGraphLaunch(graphs[f], stream));
        MOE_CUDA(cudaStreamSynchronize(stream));
      } else {
        arena.reset_peak();
        FN_RC(qsa::run(g, f == 0 ? kv_bf16 : kv_hq, ix_rope, wv, batch, d_x.p, selection, y, arena, stream));
        MOE_CUDA(cudaStreamSynchronize(stream));
        // A sparse prefill call under hq-e8-2b also decodes every visible row: S3's plan line.
        const bool s3_line = f == 1 && lists != nullptr && tokens > 1;
        const std::size_t allowed = fn::fn_qsa_attention_scratch_bytes(g, rows) +
                                    (s3_line ? 2 * ((sparse::visible_hq_bytes(g, batch.max_visible) + 255) / 256 * 256) : 0);
        check(arena.peak_used() <= allowed, name + ": scratch peak within its plan lines");
      }
    }
    const std::vector<uint16_t> y_bf16 = download<uint16_t>(d_y_bf16.p, static_cast<std::size_t>(rows) * H);
    const std::vector<uint16_t> y_hq = download<uint16_t>(d_y_hq.p, static_cast<std::size_t>(rows) * H);
    if (same) check(y_bf16 == y_hq, name + ": hq-e8-2b bit-identical to BF16 (every key read is the call's own)");

    std::vector<std::vector<double>> got, got_hq, want, bound, want_hq, bound_hq, codec_bound;
    for (std::size_t l = 0; l < lanes.size(); ++l) {
      const int lane = lanes[l];
      const int n = firsts[l] + tokens;
      const std::vector<double> k = lane_rows(bf16_store, tables, slot_of[lane], n, false);
      const std::vector<double> v = lane_rows(bf16_store, tables, slot_of[lane], n, true);
      check_rows(name + " lane " + std::to_string(lane) + " K", k, ref[lane].k, firsts[l], tokens);
      check_rows(name + " lane " + std::to_string(lane) + " V", v, ref[lane].v, firsts[l], tokens);
      for (int r : check_at) {
        if (r / tokens != static_cast<int>(l)) continue;
        const int32_t position = firsts[l] + r % tokens;
        std::vector<int32_t> keys;
        if (lists != nullptr) {
          keys = (*lists)[r];
        } else {
          for (int32_t j = 0; j <= position; ++j) keys.push_back(j);
        }
        const std::vector<uint16_t> xr = token_input(lane, position);
        const RowRef fp64 = reference_row(w, rope, xr, position, keys, ref[lane].k, ref[lane].v);
        got.push_back(row_of(y_bf16, r));
        got_hq.push_back(row_of(y_hq, r));
        want.push_back(fp64.y);
        bound.push_back(fp64.bound);
        if (!same) {
          std::vector<double> kdr = kd, vdr = vd;
          if (lists != nullptr && tokens == 1) {
            // The listed scratch holds row r's i-th listed key at r * width + i: re-index by position.
            kdr.assign(k.size(), 0.0);
            vdr.assign(v.size(), 0.0);
            for (std::size_t i = 0; i < keys.size(); ++i) {
              const std::size_t from = (static_cast<std::size_t>(r) * kWidth + i) * KVH * D;
              const std::size_t to = static_cast<std::size_t>(keys[i]) * KVH * D;
              std::copy_n(&kd[from], KVH * D, &kdr[to]);
              std::copy_n(&vd[from], KVH * D, &vdr[to]);
            }
          }
          const RowRef read = reference_row(w, rope, xr, position, keys, kdr, vdr);
          want_hq.push_back(read.y);
          bound_hq.push_back(read.bound);
          // hq against BF16: the two fp64 references' difference -- the measured codec error of
          // the rows read, carried through attention, gate and o_proj -- plus both routes' bounds.
          std::vector<double> tol(H);
          for (int i = 0; i < H; ++i) tol[i] = std::fabs(read.y[i] - fp64.y[i]) + read.bound[i] + fp64.bound[i];
          codec_bound.push_back(tol);
        }
      }
    }
    check_bounded(name + " (BF16 vs fp64)", got, want, bound);
    if (!same) {
      check_bounded(name + " (hq-e8-2b vs fp64 over the rows it read)", got_hq, want_hq, bound_hq);
      check_bounded(name + " (hq-e8-2b vs BF16, codec tolerance)", got_hq, got, codec_bound);
      double num = 0.0, den = 0.0;
      for (std::size_t r = 0; r < want.size(); ++r) {
        for (int i = 0; i < H; ++i) {
          num += (want_hq[r][i] - want[r][i]) * (want_hq[r][i] - want[r][i]);
          den += want[r][i] * want[r][i];
        }
      }
      std::printf("  %s: the codec's effect on the fp64 output, relative L2 %.3e\n", name.c_str(), std::sqrt(num / den));
    }
    for (std::size_t l = 0; l < lanes.size(); ++l) frontier[lanes[l]] += tokens;
  };

  call("lane A prefill P=0 T=1200 (one shot)", {0}, 1200, nullptr, {0, 1, 63, 64, 65, 600, 1137, 1199}, true, nullptr);
  call("lane A prefill P=1200 T=300 (shorter than the ring, on 1200 keys)", {0}, 300, nullptr, {0, 1, 150, 299}, false,
       nullptr);
  call("lane A prefill P=1500 T=551 (wider than the ring, to dense_threshold)", {0}, 551, nullptr, {0, 38, 39, 300, 550},
       false, nullptr);

  // The residual window after a chunk wider than the ring: sinks and the last 512 keys exact
  // (up to the rotation's round trip), the rest decoded by the codec.
  {
    fn::Batch m;
    m.lanes = 1;
    m.tokens = 1;
    upload(d_slots, std::vector<int32_t>{slot_of[0]});
    upload(d_positions, std::vector<int32_t>{frontier[0] - 1});
    MOE_CUDA(cudaDeviceSynchronize());
    m.slots = d_slots.as<int32_t>();
    m.positions = d_positions.as<int32_t>();
    m.max_visible = frontier[0];
    sparse::KvSource out;
    check(sparse::decode_visible_hq(g, hq_store.hq(dt), m, d_dec_k.as<__nv_bfloat16>(), d_dec_v.as<__nv_bfloat16>(),
                                    &out, stream) == nullptr,
          "window: decode launches");
    MOE_CUDA(cudaStreamSynchronize(stream));
    const int n = frontier[0];
    const std::vector<double> kd = as_double(download<uint16_t>(d_dec_k.p, static_cast<std::size_t>(n) * KVH * D).data(),
                                             static_cast<std::size_t>(n) * KVH * D);
    const std::vector<double> k = lane_rows(bf16_store, tables, slot_of[0], n, false);
    double window_worst = 0.0, codec_least = 1e300, codec_sum = 0.0;
    int codec_rows = 0;
    for (int j = 0; j < n; ++j) {
      for (int h = 0; h < KVH; ++h) {
        double num = 0.0, den = 0.0;
        for (int d = 0; d < D; ++d) {
          const std::size_t at = (static_cast<std::size_t>(j) * KVH + h) * D + d;
          num += (kd[at] - k[at]) * (kd[at] - k[at]);
          den += k[at] * k[at];
        }
        const double rel = std::sqrt(num / den);
        if (j < 32 || j >= n - 512) {
          window_worst = std::max(window_worst, rel);
        } else {
          codec_least = std::min(codec_least, rel);
          codec_sum += rel;
          ++codec_rows;
        }
      }
    }
    std::printf("  window: sink + recent rows worst relative L2 %.3e; codec rows mean %.3e, least %.3e\n",
                window_worst, codec_sum / codec_rows, codec_least);
    check(window_worst <= std::ldexp(1.0, -7), "window: every sink and recent row exact up to the rotation round trip");
    check(codec_least > std::ldexp(1.0, -6), "window: the rows outside it are codec rows");
  }

  call("lane B prefill P=0 T=700", {1}, 700, nullptr, {0, 699}, true, nullptr);
  call("lane C prefill P=0 T=40", {2}, 40, nullptr, {39}, true, nullptr);

  // Decode: S3's sparse route on hand-made selections, rows in the call's lane order.
  auto lists_for = [&](const std::vector<int> &lanes) {
    std::vector<std::vector<int32_t>> lists;
    for (int lane : lanes) lists.push_back(selection_at(frontier[lane]));
    return lists;
  };
  {
    const auto lists = lists_for({0, 1, 2});
    call("decode round 1, 3 lanes, eager", {0, 1, 2}, 1, &lists, {0, 1, 2}, false, nullptr);
  }
  cudaGraphExec_t graphs[2];
  cudaGraph_t captured[2];
  {
    fn::Batch batch;
    batch.lanes = 3;
    batch.tokens = 1;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    batch.max_visible = 4096;
    fn::Selection selection;
    selection.tokens = d_sel_tokens.as<int32_t>();
    selection.counts = d_sel_counts.as<int32_t>();
    for (int f = 0; f < 2; ++f) {
      MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      const int32_t rc = qsa::run(g, f == 0 ? kv_bf16 : kv_hq, ix_rope, wv, batch, d_x.p, selection,
                                  f == 0 ? d_y_bf16.p : d_y_hq.p, arena, stream);
      MOE_CUDA(cudaStreamEndCapture(stream, &captured[f]));
      FN_RC(rc);
      MOE_CUDA(cudaGraphInstantiate(&graphs[f], captured[f], 0));
    }
  }
  {
    const auto lists = lists_for({0, 1, 2});
    call("decode round 2, graph replay", {0, 1, 2}, 1, &lists, {0, 1, 2}, false, graphs);
  }
  {
    const auto lists = lists_for({2, 0, 1});
    call("decode round 3, graph replay, lanes C A B", {2, 0, 1}, 1, &lists, {0, 1, 2}, false, graphs);
  }
  for (int f = 0; f < 2; ++f) {
    MOE_CUDA(cudaGraphExecDestroy(graphs[f]));
    MOE_CUDA(cudaGraphDestroy(captured[f]));
  }

  // A prefill chunk past dense_threshold: every row its own selection, on S3's sparse route.
  {
    std::vector<std::vector<int32_t>> lists;
    for (int t = 0; t < kSparseRows; ++t) lists.push_back(selection_at(frontier[0] + t));
    call("lane A sparse prefill T=40 past dense_threshold", {0}, kSparseRows, &lists, {0, 17, 39}, false, nullptr);
  }

  // ---- entry: fn_qsa_attention on seq pools -------------------------------------------------
  // Attention layer 1 of a three-layer pool, one sequence, dense prefill then a decode token:
  // bit-identical to qsa::run on stores of its own, its rows in layer 1's planes (and, under hq,
  // window) only.
  for (int32_t format : {IGNIS_KV_FORMAT_BF16, IGNIS_KV_FORMAT_HQ_E8_2B}) {
    const std::string fname = format == IGNIS_KV_FORMAT_BF16 ? "entry (bf16)" : "entry (hq-e8-2b)";
    ignis_seq_pool_spec spec{};
    spec.num_kv_heads = KVH;
    spec.head_dim = D;
    spec.kv_format = format;
    spec.kv_page_group_count = 8;
    spec.max_context_tokens = 256;
    spec.slot_count = 2;
    spec.gdn_num_layers = 1;
    spec.gdn_conv_channels = 10240;
    spec.gdn_value_heads = 48;
    spec.gdn_head_dim = 128;
    spec.vocab = 1024;
    spec.kv_num_layers = 3;
    ignis_seq_pool *pool = nullptr;
    ignis_seq *seq = nullptr;
    if (ignis_seq_pool_create(&spec, &pool) != 0 || ignis_seq_alloc(pool, 128, &seq) != 0) {
      std::fprintf(stderr, "FATAL: seq pool: %s\n", ignis_seq_last_error());
      return EXIT_FAILURE;
    }
    Store own(format);
    const qsa::Kv own_kv = own.kv(dt);
    fn::Context ctx;
    ctx.g = g;
    ctx.kv_format = format;
    ctx.rope = frequencies;
    ctx.pool = pool;
    bool same = true;
    int32_t first = 0;
    for (int tokens : {50, 1}) {
      std::vector<uint16_t> x;
      for (int t = 0; t < tokens; ++t) {
        const std::vector<uint16_t> xt = token_input(7, first + t);
        x.insert(x.end(), xt.begin(), xt.end());
      }
      upload(d_x, x);
      upload(d_slots, std::vector<int32_t>{seq->slot});
      upload(d_positions, std::vector<int32_t>{first});
      std::vector<int32_t> list(kWidth, -1);
      for (int32_t j = 0; j <= first; ++j) list[j] = j;
      upload(d_sel_tokens, list);
      upload(d_sel_counts, std::vector<int32_t>{first + 1});
      MOE_CUDA(cudaDeviceSynchronize());
      fn::Batch batch;
      batch.lanes = 1;
      batch.tokens = tokens;
      batch.slots = d_slots.as<int32_t>();
      batch.positions = d_positions.as<int32_t>();
      batch.max_visible = first + tokens;
      fn::Selection selection;
      selection.dense = tokens > 1;
      selection.tokens = d_sel_tokens.as<int32_t>();
      selection.counts = d_sel_counts.as<int32_t>();
      // The own store's slot of the same index: its block-table row differs, its values do not.
      FN_RC(fn::fn_qsa_attention(ctx, 1, wv, batch, d_x.p, selection, d_y_bf16.p, arena, stream));
      FN_RC(qsa::run(g, own_kv, ix_rope, wv, batch, d_x.p, selection, d_y_hq.p, arena, stream));
      MOE_CUDA(cudaStreamSynchronize(stream));
      same = same && download<uint16_t>(d_y_bf16.p, static_cast<std::size_t>(tokens) * H) ==
                         download<uint16_t>(d_y_hq.p, static_cast<std::size_t>(tokens) * H);
      first += tokens;
    }
    check(same, fname + ": prefill and decode outputs bit-identical to qsa::run");
    // The sequence's rows in each attention layer's K plane (codes under hq): layer 1 holds the
    // own store's rows, layers 0 and 2 nothing.
    const std::vector<int32_t> row = download<int32_t>(pool->kv_pool.block_table_row(seq->slot).data, 4);
    const std::size_t row_bytes = format == IGNIS_KV_FORMAT_BF16 ? D * 2 : 64;
    auto rows_of = [&](const void *plane, const std::vector<int32_t> &pages_of) {
      std::vector<uint8_t> out;
      for (int32_t j = 0; j < first; ++j) {
        for (int h = 0; h < KVH; ++h) {
          const std::size_t at = ((static_cast<std::size_t>(pages_of[j / 64]) * KVH + h) * 64 + j % 64) * row_bytes;
          const std::vector<uint8_t> r = download<uint8_t>(static_cast<const uint8_t *>(plane) + at, row_bytes);
          out.insert(out.end(), r.begin(), r.end());
        }
      }
      return out;
    };
    const std::vector<int32_t> own_pages(tables.begin() + static_cast<std::ptrdiff_t>(seq->slot) * kLogicalPages,
                                         tables.begin() + static_cast<std::ptrdiff_t>(seq->slot + 1) * kLogicalPages);
    const auto plane = [&](int32_t layer) {
      return pool->kv_pool.plane(ignis_kv_plane_index(format, layer, IGNIS_KV_PLANE_K)).data;
    };
    check(rows_of(plane(1), row) == rows_of(own.k->p, own_pages), fname + ": layer 1's K plane holds the call's rows");
    bool empty = true;
    for (int32_t layer : {0, 2}) {
      for (uint8_t b : rows_of(plane(layer), row)) empty = empty && b == 0;
    }
    check(empty, fname + ": attention layers 0 and 2 are untouched");
    if (format == IGNIS_KV_FORMAT_HQ_E8_2B) {
      const std::size_t window = static_cast<std::size_t>(544) * KVH * D * 2;
      const std::vector<uint8_t> w1 = download<uint8_t>(pool->hq_residual_plane(false, 1, seq->slot), window);
      const std::vector<uint8_t> own_w =
          download<uint8_t>(static_cast<const uint8_t *>(own.res_k->p) + seq->slot * window, window);
      check(w1 == own_w, fname + ": layer 1's residual window holds the call's rows");
      bool other_windows = true;
      for (int32_t layer : {0, 2}) {
        for (uint8_t b : download<uint8_t>(pool->hq_residual_plane(false, layer, seq->slot), window)) {
          other_windows = other_windows && b == 0;
        }
      }
      check(other_windows, fname + ": layers 0 and 2's residual windows are untouched");
    }
    // Refusals by name.
    fn::Batch one;
    one.lanes = 1;
    one.tokens = 1;
    one.slots = d_slots.as<int32_t>();
    one.positions = d_positions.as<int32_t>();
    one.max_visible = first + 1;
    fn::Selection dense;
    dense.dense = true;
    auto refused = [&](const fn::Context &c, int32_t ordinal, const char *needle) {
      return fn::fn_qsa_attention(c, ordinal, wv, one, d_x.p, dense, d_y_bf16.p, arena, stream) != 0 &&
             std::strstr(fn::fn_last_error(), needle) != nullptr;
    };
    check(refused(ctx, 3, "attention layer 3"), fname + ": an attention ordinal past the pool's layers is refused");
    fn::Context other = ctx;
    other.kv_format = format == IGNIS_KV_FORMAT_BF16 ? IGNIS_KV_FORMAT_HQ_E8_2B : IGNIS_KV_FORMAT_BF16;
    check(refused(other, 1, "format"), fname + ": a KV format other than the pool's is refused");
    other = ctx;
    other.rope.attention_factor = 2.0F;
    check(refused(other, 1, "attention factor"), fname + ": a rope attention factor is refused");
    other = ctx;
    other.pool = nullptr;
    check(refused(other, 1, "no seq pool"), fname + ": a context without a seq pool is refused");
    ignis_seq_release(pool, seq);
    ignis_seq_pool_free(pool);
  }

  // Refusals by name.
  {
    fn::Batch bad;
    bad.lanes = 2;
    bad.tokens = 4;
    bad.slots = d_slots.as<int32_t>();
    bad.positions = d_positions.as<int32_t>();
    bad.max_visible = 8;
    fn::Selection dense;
    dense.dense = true;
    check(qsa::run(g, kv_bf16, ix_rope, wv, bad, d_x.p, dense, d_y_bf16.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "one lane") != nullptr,
          "a dense call of two lanes is refused by name");
    fn::QsaWeights short_w = wv;
    short_w.k_proj.rows = 256;
    bad.lanes = 1;
    check(qsa::run(g, kv_bf16, ix_rope, short_w, bad, d_x.p, dense, d_y_bf16.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "wrong shape") != nullptr,
          "a wrongly shaped weight is refused by name");
    fn::Batch many;
    many.lanes = qsa::kMaxDecodeLanes + 1;
    many.tokens = 1;
    many.slots = d_slots.as<int32_t>();
    many.positions = d_positions.as<int32_t>();
    many.max_visible = 8;
    fn::Selection listed;
    listed.tokens = d_sel_tokens.as<int32_t>();
    listed.counts = d_sel_counts.as<int32_t>();
    check(qsa::run(g, kv_bf16, ix_rope, wv, many, d_x.p, listed, d_y_bf16.p, arena, stream) != 0 &&
              std::strstr(fn::fn_last_error(), "decode call is at most") != nullptr,
          "a decode call of more lanes than kMaxDecodeLanes is refused by name");
  }
  MOE_CUDA(cudaStreamDestroy(stream));

  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next qsa test: %d check(s) FAILED\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash_next qsa test: ok\n");
  return EXIT_SUCCESS;
}
