// Real-geometry expert sets and their fp64 reference, shared by the Flash-Next routed-expert
// tests -- OURS (spec flash-next/02, GitHub #300).
//
// A set holds a handful of physical expert-projection records (layout.md §3: trellis, suh, svh,
// 4 KiB padding) at every K class, built from counter-hash trellis words (any bit pattern is a
// valid encoding) and fp16 channel scales, placed in one device buffer in a chosen order. The
// 512 expert ids map onto them many-to-one through the slot table, so routing can use every id
// while the host keeps only the physical weights. Their inner weights come from the kernel
// decoder, which test_trellis_decode holds bit-exact to exllamav3's `reconstruct`; the set
// re-checks one record per K class against the host restatement before trusting them.
#ifndef IGNIS_MOE_EXPERTS_COMMON_H
#define IGNIS_MOE_EXPERTS_COMMON_H

#include "ignis_moe.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace moe_test {

constexpr int kH = IGNIS_MOE_HIDDEN;
constexpr int kI = IGNIS_MOE_INTERMEDIATE;
constexpr int kE = IGNIS_MOE_EXPERTS;
constexpr int kTop = IGNIS_MOE_TOP_K;

inline const std::vector<double> &f16_lut() {
  static std::vector<double> lut = [] {
    std::vector<double> t(65536);
    for (int i = 0; i < 65536; ++i) t[i] = f16_to_f32(static_cast<uint16_t>(i));
    return t;
  }();
  return lut;
}

struct Record {
  uint32_t k2 = 0;
  int in = 0, out = 0;
  std::vector<uint16_t> words, suh, svh;  // as stored
  std::vector<uint16_t> inner;            // decoded [in][out] fp16 bits
  std::vector<uint8_t> bytes() const {
    uint64_t total = 0;
    ignis_moe_record_bytes(in == kH ? IGNIS_MOE_PROJ_GATE_UP : IGNIS_MOE_PROJ_DOWN, k2, &total);
    std::vector<uint8_t> b(total, 0);
    std::memcpy(b.data(), words.data(), words.size() * 2);
    std::memcpy(b.data() + words.size() * 2, suh.data(), suh.size() * 2);
    std::memcpy(b.data() + words.size() * 2 + suh.size() * 2, svh.data(), svh.size() * 2);
    return b;
  }
};

// The study's K mix (run 8, layer 1: gate/up 43% K=2, 22% 2.5, 32% 3, 2-3% 4) in miniature,
// with every class present.
inline const uint32_t kRecordK2[] = {4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 8, 8, 4};
constexpr int kRecords = 16;

inline Record make_record(uint32_t stream, int in, int out, uint32_t k2, float suh_mag, float svh_mag) {
  Record r;
  r.k2 = k2;
  r.in = in;
  r.out = out;
  r.words = hash_trellis_words(stream, in, out, k2);
  r.suh.resize(in);
  r.svh.resize(out);
  for (int i = 0; i < in; ++i) {
    const float u = hash_uniform(stream + 1, i, 1.0f);
    r.suh[i] = f32_to_f16((u < 0 ? -1.0f : 1.0f) * suh_mag * (0.5f + std::fabs(u)));
  }
  for (int i = 0; i < out; ++i) {
    const float u = hash_uniform(stream + 2, i, 1.0f);
    r.svh[i] = f32_to_f16((u < 0 ? -1.0f : 1.0f) * svh_mag * (0.5f + std::fabs(u)));
  }
  return r;
}

struct ExpertSet {
  std::vector<Record> gu, dn;  // kRecords physical records each
  std::unique_ptr<DeviceBytes> buffer;
  std::vector<ignis_moe_slot> slots;  // 1024 entries
  DeviceBytes d_slots{static_cast<std::size_t>(kE) * 2 * sizeof(ignis_moe_slot)};

  static int gu_record(int expert) { return expert % kRecords; }
  static int dn_record(int expert) { return (expert * 7 + 3) % kRecords; }

  ExpertSet() {
    for (int r = 0; r < kRecords; ++r) {
      gu.push_back(make_record(40000 + 16 * r, kH, 2 * kI, kRecordK2[r], 0.03f, 0.3f));
      dn.push_back(make_record(50000 + 16 * r, kI, kH, kRecordK2[(r + 5) % kRecords], 0.05f, 0.1f));
    }
    // Inner weights from the kernel decoder, one record per K class re-checked on the host.
    std::vector<uint32_t> checked;
    for (auto *set : {&gu, &dn}) {
      for (Record &r : *set) {
        DeviceBytes w(r.words.size() * 2), out(static_cast<std::size_t>(r.in) * r.out * 2);
        upload(w, r.words);
        MOE_RC(ignis_moe_trellis_reconstruct(w.p, r.k2, r.in, r.out, out.p, nullptr));
        MOE_CUDA(cudaDeviceSynchronize());
        r.inner = download<uint16_t>(out.p, static_cast<std::size_t>(r.in) * r.out);
        if (set == &dn && std::find(checked.begin(), checked.end(), r.k2) == checked.end()) {
          checked.push_back(r.k2);
          check(r.inner == host_trellis_decode(r.words.data(), r.k2, r.in, r.out),
                "kernel decode equals the host restatement (down, k2 " + std::to_string(r.k2) + ")");
        }
      }
    }
  }

  // Place every record in one device buffer, in `order` (a permutation of 0..2*kRecords-1 over
  // gate/up records then down records), and point all 512 experts' slots at them.
  void place(const std::vector<int> &order) {
    std::vector<std::vector<uint8_t>> blobs;
    for (const Record &r : gu) blobs.push_back(r.bytes());
    for (const Record &r : dn) blobs.push_back(r.bytes());
    std::size_t total = 0;
    for (const auto &b : blobs) total += b.size();
    buffer = std::make_unique<DeviceBytes>(total);
    std::vector<std::size_t> offset(blobs.size());
    std::size_t at = 0;
    for (int idx : order) {
      offset[idx] = at;
      MOE_CUDA(cudaMemcpy(static_cast<char *>(buffer->p) + at, blobs[idx].data(), blobs[idx].size(), cudaMemcpyHostToDevice));
      at += blobs[idx].size();
    }
    slots.assign(static_cast<std::size_t>(kE) * 2, ignis_moe_slot{nullptr, 0, 0});
    for (int e = 0; e < kE; ++e) {
      const int g = gu_record(e), d = dn_record(e);
      slots[e * 2 + IGNIS_MOE_PROJ_GATE_UP] = {static_cast<char *>(buffer->p) + offset[g], gu[g].k2, 0};
      slots[e * 2 + IGNIS_MOE_PROJ_DOWN] = {static_cast<char *>(buffer->p) + offset[kRecords + d], dn[d].k2, 0};
    }
    upload(d_slots, slots);
  }
};

// One projection in fp64: had(had(x o suh) . inner) o svh.
inline std::vector<double> project(const Record &r, const std::vector<double> &x) {
  const auto &lut = f16_lut();
  std::vector<double> xs(r.in);
  for (int k = 0; k < r.in; ++k) xs[k] = x[k] * lut[r.suh[k]];
  hadamard128_inplace(xs.data(), r.in);
  std::vector<double> y(r.out, 0.0);
  for (int k = 0; k < r.in; ++k) {
    const uint16_t *row = r.inner.data() + static_cast<std::size_t>(k) * r.out;
    const double a = xs[k];
    for (int n = 0; n < r.out; ++n) y[n] += a * lut[row[n]];
  }
  hadamard128_inplace(y.data(), r.out);
  for (int n = 0; n < r.out; ++n) y[n] *= lut[r.svh[n]];
  return y;
}

// One routed expert on one token: down(silu(gate) * up).
inline std::vector<double> expert_f64(const ExpertSet &set, int expert, const std::vector<double> &x) {
  const std::vector<double> gu = project(set.gu[ExpertSet::gu_record(expert)], x);
  std::vector<double> h(kI);
  for (int j = 0; j < kI; ++j) h[j] = silu(gu[j]) * gu[kI + j];
  return project(set.dn[ExpertSet::dn_record(expert)], h);
}

// Token t's routed output: the weighted sum of its experts.
inline std::vector<double> routed_f64(const ExpertSet &set, const int32_t *ids, const float *weights,
                                      const std::vector<double> &x) {
  std::vector<double> y(kH, 0.0);
  for (int r = 0; r < kTop; ++r) {
    const std::vector<double> o = expert_f64(set, ids[r], x);
    for (int j = 0; j < kH; ++j) y[j] += static_cast<double>(weights[r]) * o[j];
  }
  return y;
}

// Distinct ids for `tokens` tokens: token t draws its ten experts from the hash, the first
// `shared` of them copied from token t - 1 so distinct-expert de-duplication is exercised.
inline void make_routing(int tokens, uint32_t stream, int shared, std::vector<int32_t> &ids,
                         std::vector<float> &weights) {
  ids.assign(static_cast<std::size_t>(tokens) * kTop, 0);
  weights.assign(static_cast<std::size_t>(tokens) * kTop, 0.0f);
  uint64_t draw = 0;
  for (int t = 0; t < tokens; ++t) {
    int32_t *row = &ids[static_cast<std::size_t>(t) * kTop];
    int n = 0;
    if (t > 0) {
      for (; n < shared; ++n) row[n] = ids[static_cast<std::size_t>(t - 1) * kTop + n];
    }
    while (n < kTop) {
      const int32_t e = static_cast<int32_t>(hash_u32(stream, draw++) % kE);
      if (std::find(row, row + n, e) == row + n) row[n++] = e;
    }
    float sum = 0.0f;
    float raw[kTop];
    for (int r = 0; r < kTop; ++r) {
      raw[r] = 0.2f + std::fabs(hash_uniform(stream + 1, static_cast<uint64_t>(t) * kTop + r, 1.0f));
      sum += raw[r];
    }
    for (int r = 0; r < kTop; ++r) weights[static_cast<std::size_t>(t) * kTop + r] = bf16_to_f32(f32_to_bf16(raw[r] / sum));
  }
}

inline std::vector<uint16_t> make_tokens(int tokens, uint32_t stream, float amplitude) {
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * kH);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(hash_uniform(stream, i, amplitude));
  return x;
}

inline std::vector<double> token_f64(const std::vector<uint16_t> &x, int t) {
  std::vector<double> v(kH);
  for (int k = 0; k < kH; ++k) v[k] = bf16_to_f32(x[static_cast<std::size_t>(t) * kH + k]);
  return v;
}

inline std::vector<double> fixed_to_f64(const std::vector<int64_t> &acc, int t) {
  std::vector<double> v(kH);
  for (int k = 0; k < kH; ++k) v[k] = std::ldexp(static_cast<double>(acc[static_cast<std::size_t>(t) * kH + k]), -32);
  return v;
}

// ||a - b|| / ||b|| and max |a - b| / max |b|.
inline void rel_errors(const std::vector<double> &a, const std::vector<double> &b, double *l2, double *linf) {
  double num = 0.0, den = 0.0, mx = 0.0, bm = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
    mx = std::max(mx, std::fabs(a[i] - b[i]));
    bm = std::max(bm, std::fabs(b[i]));
  }
  *l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  *linf = bm > 0 ? mx / bm : mx;
}

}  // namespace moe_test

#endif  // IGNIS_MOE_EXPERTS_COMMON_H
