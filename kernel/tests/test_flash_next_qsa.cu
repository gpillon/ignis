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
//               up to dense_threshold()), then decode rounds on S3's sparse route, the first eager
//               and two replaying captured graphs. Every BF16 call against fp64 on sampled rows;
//               every hq-e8-2b call bit-identical to BF16 while every key it reads is the call's
//               own, else within a bound DERIVED FROM THE MEASURED CODEC ERROR of the very rows it
//               read (the rows decoded back after the call), propagated through the softmax, the
//               gate and o_proj; the residual window's rows exact after a chunk wider than the ring.
//
// Bounds. Attention on the same q and K/V: the probabilities are rounded to BF16 for the value
// product while the normalizer is their fp32 sum, and the output is rounded once, so
// |o - o_ref| <= 3 * 2^-9 * A with A = sum_j p_j |v_j| (stated as 2^-7 A). The gate and o_proj
// propagate it elementwise: y's bound is sum_c |W_ic| B_c + 2^-8 |y_i|, B_c the gated element's.
// hq against BF16 adds, per query head, Delta = max_j |q . (k'_j - k_j)| / 16 over the keys read
// (k' decoded, k exact): every probability moves by a factor within e^(+-2 Delta), so
// |o' - o| <= (e^(2 Delta) - 1) sum_j p_j |v'_j| + sum_j p_j |v'_j - v_j|.

#include "flash_next/qsa.h"

#include "flash_next_s2_test_common.h"
#include "ignis_fp8_linear.h"

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

double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }

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

// ---- inputs and the paged stores ------------------------------------------------------------

std::vector<uint16_t> token_input(int lane, int32_t position) {
  return bf16_vector(3000 + lane, H, -1.0, 1.0, static_cast<uint64_t>(position) * H);
}

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

// Rows [0, n) of a lane in a BF16 store, [n][KVH][D] in fp64, per role.
std::vector<double> lane_rows(const Store &s, const std::vector<int32_t> &tables, int slot, int n, bool role_v) {
  const std::vector<uint16_t> plane =
      download<uint16_t>((role_v ? s.v : s.k)->p, static_cast<std::size_t>(kPages) * KVH * 64 * D);
  std::vector<double> rows(static_cast<std::size_t>(n) * KVH * D);
  for (int j = 0; j < n; ++j) {
    const int page = tables[static_cast<std::size_t>(slot) * kLogicalPages + j / 64];
    for (int h = 0; h < KVH; ++h) {
      const std::size_t at = ((static_cast<std::size_t>(page) * KVH + h) * 64 + j % 64) * D;
      for (int d = 0; d < D; ++d) rows[(static_cast<std::size_t>(j) * KVH + h) * D + d] = bf16_to_f32(plane[at + d]);
    }
  }
  return rows;
}

// ---- the reference of one row and its bounds ------------------------------------------------

struct RowRef {
  std::vector<double> y;          // fp64, unrounded
  std::vector<double> bound;      // BF16 route against y
  std::vector<double> hq_bound;   // hq-e8-2b route against the BF16 route (when decoded rows given)
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

// One query row at `position` attending `keys` (ascending) of the lane's exact rows
// [n][KVH][D] (k, v). With decoded rows (k', v', same layout, indexed like the exact ones)
// also the hq-against-BF16 bound.
RowRef reference_row(const Weights &w, const Rope &rope, const std::vector<uint16_t> &x_bf16, int32_t position,
                     const std::vector<int32_t> &keys, const std::vector<double> &k, const std::vector<double> &v,
                     const std::vector<double> *kd, const std::vector<double> *vd) {
  const std::vector<double> x = as_double(x_bf16.data(), x_bf16.size());
  std::vector<double> qg = project(w.q, x);
  for (double &e : qg) e = bf16r(e);
  std::vector<double> gated(qsa::kOutWidth), b_bf16(qsa::kOutWidth), b_hq(qsa::kOutWidth);
  for (int h = 0; h < QH; ++h) {
    const std::vector<double> q = norm_rope(&qg[static_cast<std::size_t>(h) * 2 * D], w.q_norm, rope, position);
    const int kvh = h / qsa::kGroup;
    std::vector<double> s(keys.size()), p(keys.size());
    double m = -1e300;
    double delta = 0.0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const double *kr = &k[(static_cast<std::size_t>(keys[i]) * KVH + kvh) * D];
      double dot = 0.0, ddot = 0.0;
      for (int d = 0; d < D; ++d) dot += q[d] * kr[d];
      if (kd != nullptr) {
        const double *kdr = &(*kd)[(static_cast<std::size_t>(keys[i]) * KVH + kvh) * D];
        for (int d = 0; d < D; ++d) ddot += q[d] * (kdr[d] - kr[d]);
        delta = std::max(delta, std::fabs(ddot) * kScale);
      }
      s[i] = dot * kScale;
      m = std::max(m, s[i]);
    }
    double l = 0.0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      p[i] = std::exp(s[i] - m);
      l += p[i];
    }
    const double grow = std::exp(2.0 * delta);
    for (int d = 0; d < D; ++d) {
      double o = 0.0, a = 0.0, ad = 0.0, dv = 0.0;
      for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::size_t at = (static_cast<std::size_t>(keys[i]) * KVH + kvh) * D + d;
        o += p[i] * v[at];
        a += p[i] * std::fabs(v[at]);
        if (vd != nullptr) {
          ad += p[i] * std::fabs((*vd)[at]);
          dv += p[i] * std::fabs((*vd)[at] - v[at]);
        }
      }
      o /= l;
      a /= l;
      ad /= l;
      dv /= l;
      const std::size_t c = static_cast<std::size_t>(h) * D + d;
      const double sg = bf16r(sigmoid(qg[static_cast<std::size_t>(h) * 2 * D + D + d]));
      gated[c] = bf16r(bf16r(o) * sg);
      b_bf16[c] = sg * std::ldexp(a, -6) + std::ldexp(std::fabs(gated[c]), -8);
      b_hq[c] = b_bf16[c] + sg * ((grow - 1.0) * ad + dv + std::ldexp(grow * ad, -6)) +
                std::ldexp(std::fabs(gated[c]), -8);
    }
  }
  RowRef r;
  r.y = project(w.o, gated);
  r.bound = project_abs(w.o, b_bf16);
  for (int i = 0; i < H; ++i) r.bound[i] += std::ldexp(std::fabs(r.y[i]), -8) + 1e-6;
  if (kd != nullptr) {
    r.hq_bound = project_abs(w.o, b_hq);
    for (int i = 0; i < H; ++i) r.hq_bound[i] += std::ldexp(std::fabs(r.y[i]), -7) + 1e-6;
  }
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
  const fn::indexer::Rope ix_rope = fn::indexer::rope_from(ninfer::ops::rope_linear_frequencies(1e7F, 64));
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
    const std::vector<uint16_t> k_in = k;
    auto d_qg = device_copy(qg);
    auto d_k = device_copy(k);
    DeviceBytes d_q(static_cast<std::size_t>(rows) * qsa::kOutWidth * 2);
    upload(d_positions, std::vector<int32_t>(positions, positions + rows));
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
        const std::vector<double> x = as_double(&k_in[(static_cast<std::size_t>(r) * KVH + h) * D], D);
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
    fn::Batch fill;
    fill.lanes = 1;
    fill.tokens = n;
    fill.slots = d_slots.as<int32_t>();
    fill.positions = d_positions.as<int32_t>();
    check(qsa::append(g, pages.kv(dt), fill, d_k->as<__nv_bfloat16>(), d_v->as<__nv_bfloat16>(), stream) == nullptr,
          "dense: append launches");
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
      fn::Batch batch;
      batch.lanes = 1;
      batch.tokens = tokens;
      batch.slots = d_slots.as<int32_t>();
      batch.positions = d_positions.as<int32_t>();
      batch.max_visible = first + tokens;
      check(qsa::attend_dense(g, paged, batch, d_q->as<__nv_bfloat16>(), d_o1.as<__nv_bfloat16>(), stream) == nullptr &&
                qsa::attend_dense(g, by_position, batch, d_q->as<__nv_bfloat16>(), d_o2.as<__nv_bfloat16>(), stream) ==
                    nullptr,
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
  DeviceBytes d_sel_tokens(static_cast<std::size_t>(kSlots) * kWidth * 4), d_sel_counts(kSlots * 4);
  DeviceBytes d_dec_k(sparse::listed_hq_bytes(g, kSlots) + sparse::visible_hq_bytes(g, kWidth));
  DeviceBytes d_dec_v(sparse::listed_hq_bytes(g, kSlots) + sparse::visible_hq_bytes(g, kWidth));
  const int slot_of[kSlots] = {2, 0, 1};
  int32_t frontier[kSlots] = {0, 0, 0};

  // One call of both formats. `lists` (decode) holds each lane's selection, else the call is dense.
  // `check_rows`: the call's rows checked against fp64; `same`: the formats must agree bit for bit.
  auto call = [&](const std::string &name, const std::vector<int> &lanes, int tokens,
                  const std::vector<std::vector<int32_t>> *lists, const std::vector<int> &check_rows, bool same,
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
    for (int f = 0; f < 2; ++f) {
      void *y = f == 0 ? d_y_bf16.p : d_y_hq.p;
      if (graphs != nullptr) {
        MOE_CUDA(cudaGraphLaunch(graphs[f], stream));
      } else {
        arena.reset_peak();
        FN_RC(qsa::run(g, f == 0 ? kv_bf16 : kv_hq, ix_rope, wv, batch, d_x.p, selection, y, arena, stream));
        MOE_CUDA(cudaStreamSynchronize(stream));
        check(arena.peak_used() <= fn::fn_qsa_attention_scratch_bytes(g, rows),
              name + ": scratch peak within fn_qsa_attention_scratch_bytes");
      }
    }
    MOE_CUDA(cudaStreamSynchronize(stream));
    const std::vector<uint16_t> y_bf16 = download<uint16_t>(d_y_bf16.p, static_cast<std::size_t>(rows) * H);
    const std::vector<uint16_t> y_hq = download<uint16_t>(d_y_hq.p, static_cast<std::size_t>(rows) * H);
    if (same) check(y_bf16 == y_hq, name + ": hq-e8-2b bit-identical to BF16 (every key read is the call's own)");

    // The rows the hq call read, decoded back after it with the window of its frontier: a key
    // exact here was exact in the call too (the call's window covers the 512 keys before it and
    // its own), so the measured error is an upper bound of the call's.
    std::vector<double> kd, vd;
    if (!same) {
      fn::Batch m = batch;
      fn::Selection ms = selection;
      sparse::HqSource src = hq_store.hq(dt);
      sparse::KvSource out;
      const bool listed = lists != nullptr;
      const sparse::Status st =
          listed ? sparse::decode_listed_hq(g, src, m, ms, d_dec_k.as<__nv_bfloat16>(), d_dec_v.as<__nv_bfloat16>(), &out, stream)
                 : sparse::decode_visible_hq(g, src, m, d_dec_k.as<__nv_bfloat16>(), d_dec_v.as<__nv_bfloat16>(), &out, stream);
      check(st == nullptr, name + ": decoding the rows back launches");
      MOE_CUDA(cudaStreamSynchronize(stream));
      const std::size_t n = listed ? static_cast<std::size_t>(rows) * kWidth : static_cast<std::size_t>(batch.max_visible);
      kd = as_double(download<uint16_t>(d_dec_k.p, n * KVH * D).data(), n * KVH * D);
      vd = as_double(download<uint16_t>(d_dec_v.p, n * KVH * D).data(), n * KVH * D);
    }

    std::vector<std::vector<double>> got, got_hq, want, bound, hq_bound;
    for (int r : check_rows) {
      const int l = r / tokens;
      const int lane = lanes[l];
      const int32_t position = firsts[l] + r % tokens;
      const int n = firsts[l] + tokens;
      const std::vector<double> k = lane_rows(bf16_store, tables, slot_of[lane], n, false);
      const std::vector<double> v = lane_rows(bf16_store, tables, slot_of[lane], n, true);
      std::vector<int32_t> keys;
      std::vector<double> kdr, vdr;
      if (lists != nullptr) {
        keys = (*lists)[r];
      } else {
        for (int32_t j = 0; j <= position; ++j) keys.push_back(j);
      }
      if (!same && lists != nullptr) {
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
      const std::vector<double> *kp = same ? nullptr : (lists != nullptr ? &kdr : &kd);
      const std::vector<double> *vp = same ? nullptr : (lists != nullptr ? &vdr : &vd);
      RowRef ref = reference_row(w, rope, token_input(lane, position), position, keys, k, v, kp, vp);
      got.push_back(row_of(y_bf16, r));
      got_hq.push_back(row_of(y_hq, r));
      want.push_back(ref.y);
      bound.push_back(ref.bound);
      if (!same) hq_bound.push_back(ref.hq_bound);
    }
    check_bounded(name + " (BF16 vs fp64)", got, want, bound);
    if (!same) check_bounded(name + " (hq-e8-2b vs BF16)", got_hq, got, hq_bound);
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

  // Decode: S3's sparse route on hand-made selections (a strided subset past dense_threshold, all
  // visible tokens below it), always with the new token.
  auto lists_now = [&]() {
    std::vector<std::vector<int32_t>> lists(kSlots);
    for (int lane = 0; lane < kSlots; ++lane) {
      const int32_t p = frontier[lane];
      for (int32_t j = 0; j <= p; ++j) {
        if (p < kWidth || j < 64 || j % 3 == 0 || j > p - 48) lists[lane].push_back(j);
      }
    }
    return lists;
  };
  {
    const auto lists = lists_now();
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
  for (int round = 2; round <= 3; ++round) {
    const auto lists = lists_now();
    call("decode round " + std::to_string(round) + ", graph replay", {0, 1, 2}, 1, &lists, {0, 1, 2}, false, graphs);
  }
  for (int f = 0; f < 2; ++f) {
    MOE_CUDA(cudaGraphExecDestroy(graphs[f]));
    MOE_CUDA(cudaGraphDestroy(captured[f]));
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
  }
  MOE_CUDA(cudaStreamDestroy(stream));

  if (g_failed != 0) {
    std::fprintf(stderr, "flash_next qsa test: %d check(s) FAILED\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash_next qsa test: ok\n");
  return EXIT_SUCCESS;
}
