// The Flash-Next QSA sparse attention over hq-e8-2b KV (spec flash-next/04, GitHub #302, slice
// S3) -- OURS (ADR 0043; the codec is vendored and only called, ADR 0022): the selected rows
// decoded into a plain-frame BF16 scratch, then the same gathered attention as BF16 KV.
//
// The synthetic K/V pages of flash_next_sparse_test_common.h are encoded by the vendored
// hq_encode_row_warp (dither seeded by each row's absolute position). The expected rows come from
// an independent path: the vendored single-thread hq_decode_row_thread (bit-identical to the
// group decoder the route uses, by the codec's own contract) and an fp64 inverse rotation on the
// host (signs, then the Sylvester Hadamard / 16).
//
//   rows      the decode call's listed rows in the scratch equal the expected rows to a BF16 ulp
//             (fp32 FWHT against fp64);
//   decode    attention over the decoded rows against fp64 attention over the expected rows, also
//             replayed from a CUDA graph;
//   fresh     the call's own token rows read exactly from its BF16 K/V instead of the codec;
//   residual  sink and recent-window rows read from the side planes (rotated frame) instead of
//             the codec, by the 27B's decode rule (no fresh rows: the window is the call's last
//             512 keys, its own among them, as the ring stands after the append);
//   prefill   one lane's visible rows decoded once (the chunk's own rows fresh), read by position;
//   prefill + residual  a 200-token chunk at 3000 with its own rows fresh and the ring as it stands
//             BEFORE the call's append: sinks and [2488, 3000) from the side planes;
//   report    the hq route against the BF16 route on the same keys and values, beside the codec's
//             measured row error on these rows (ADR 0022's in-house acceptance needs real
//             Flash-Next rows for its tolerance; this prints the synthetic figure).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next_sparse_test_common.h"

#include "ops/kernel/hq_codec.cuh"

#include <functional>

using namespace sparse_test;

namespace {

constexpr int kSink = static_cast<int>(ninfer::ops::kGqaHqSinkKeys);
constexpr int kRecent = static_cast<int>(ninfer::ops::kGqaHqRecentKeys);
constexpr int kRowsPerRole = kPhysicalPages * kKvHeads * 64;

// One warp per (physical row, role): the vendored encoder.
__global__ void encode_kernel(const __nv_bfloat16 *k, const __nv_bfloat16 *v, const int32_t *position_of,
                              uint8_t *k_codes, uint8_t *k_meta, uint8_t *v_codes, uint8_t *v_meta) {
  using namespace ninfer::ops;
  __shared__ float u_scaled[4][kHqHeadDim];
  __shared__ uint32_t syms[4][kHqHeadDim];
  __shared__ int8_t signs[kHqHeadDim];
  hq_engine_signs_fill(signs);
  __syncthreads();
  const int warp = threadIdx.x >> 5;
  const int unit = blockIdx.x * 4 + warp;
  if (unit >= kRowsPerRole * 2) return;
  const int row = unit >> 1;
  const bool role_v = (unit & 1) != 0;
  const int head = (row / 64) % kKvHeads;
  hq_encode_row_warp((role_v ? v : k) + static_cast<size_t>(row) * kHqHeadDim, signs, 0, u_scaled[warp], syms[warp],
                     (role_v ? v_codes : k_codes) + static_cast<size_t>(row) * kHqRowBudgetBytes,
                     (role_v ? v_meta : k_meta) + static_cast<size_t>(row) * kHqMetaBytes,
                     hq_dither_row_seed(head, position_of[row], role_v));
}

// One thread per (physical row, role): the vendored serial decoder, rotated frame.
__global__ void reference_decode_kernel(const uint8_t *k_codes, const uint8_t *k_meta, const uint8_t *v_codes,
                                        const uint8_t *v_meta, const int32_t *position_of, __nv_bfloat16 *out) {
  using namespace ninfer::ops;
  const int unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= kRowsPerRole * 2) return;
  const int row = unit >> 1;
  const bool role_v = (unit & 1) != 0;
  const int head = (row / 64) % kKvHeads;
  hq_decode_row_thread((role_v ? v_codes : k_codes) + static_cast<size_t>(row) * kHqRowBudgetBytes,
                       (role_v ? v_meta : k_meta) + static_cast<size_t>(row) * kHqMetaBytes,
                       out + static_cast<size_t>(unit) * kHqHeadDim, hq_dither_row_seed(head, position_of[row], role_v));
}

// The engine sign diagonal and the Sylvester Hadamard, on the host (hq_codec.cuh's definitions).
double engine_sign(int d) {
  uint32_t x = 0x5EED01U ^ (static_cast<uint32_t>(d) * 0x9E3779B9U);
  x ^= x >> 16;
  x *= 0x85EBCA6BU;
  x ^= x >> 13;
  return (x & 1U) ? 1.0 : -1.0;
}

// forward: u = H (signs . x) / 16; else its inverse x = signs . (H u) / 16 (H the Sylvester
// Hadamard of order 256).
void hadamard256(const double *in, double *out, bool forward) {
  double t[kHd];
  for (int d = 0; d < kHd; ++d) t[d] = forward ? in[d] * engine_sign(d) : in[d];
  for (int len = 1; len < kHd; len <<= 1) {
    for (int i = 0; i < kHd; ++i) {
      if ((i & len) == 0) {
        const double a = t[i], b = t[i + len];
        t[i] = a + b;
        t[i + len] = a - b;
      }
    }
  }
  for (int d = 0; d < kHd; ++d) out[d] = t[d] / 16.0 * (forward ? 1.0 : engine_sign(d));
}

double ulps(uint16_t a, double want) {
  const float fa = bf16_to_f32(a);
  if (fa == static_cast<float>(want)) return 0.0;
  const double mag = std::max(std::fabs(static_cast<double>(fa)), std::fabs(want));
  if (mag < 1e-30) return 0.0;
  return std::fabs(fa - want) / std::ldexp(1.0, std::ilogb(mag) - 7);
}

// The residual window's side planes as the vendored fill leaves them: every (slot, position)
// `is_side` names (up to last[slot]) holds the BF16 rounding of its rotated row (the dual write),
// every ring bit set. `plain` is what the route must return for such a row: the fp64 un-rotation
// of the stored BF16 values.
struct SidePlanes {
  std::vector<uint16_t> k, v;
  std::vector<uint32_t> ring;
  std::vector<double> plain;  // [role][slot][sink + recent][kv_heads][256]
  static size_t index(int slot, int pos, int head) {
    const int row = pos < kSink ? pos : kSink + (pos & (kRecent - 1));
    return (static_cast<size_t>(slot) * (kSink + kRecent) + row) * kKvHeads * kHd + static_cast<size_t>(head) * kHd;
  }
};

SidePlanes make_side(const Pages &pg, const std::function<bool(int, int)> &is_side, const std::vector<int> &last) {
  SidePlanes sp_;
  const size_t n = static_cast<size_t>(kSlots) * (kSink + kRecent) * kKvHeads * kHd;
  sp_.k.assign(n, 0);
  sp_.v.assign(n, 0);
  sp_.ring.assign(static_cast<size_t>(kSlots) * (kRecent / 32), 0xFFFFFFFFU);
  sp_.plain.assign(2 * n, 0.0);
  for (int slot = 0; slot < static_cast<int>(last.size()); ++slot) {
    for (int pos = 0; pos <= last[slot]; ++pos) {
      if (!is_side(slot, pos)) continue;
      for (int h = 0; h < kKvHeads; ++h) {
        for (int role = 0; role < 2; ++role) {
          double x[kHd], u[kHd], back[kHd];
          const size_t at = pg.row_at(slot, pos, h);
          for (int dd = 0; dd < kHd; ++dd) x[dd] = bf16_to_f32((role == 0 ? pg.k : pg.v)[at + dd]);
          hadamard256(x, u, true);
          uint16_t *dst = &(role == 0 ? sp_.k : sp_.v)[SidePlanes::index(slot, pos, h)];
          for (int dd = 0; dd < kHd; ++dd) {
            dst[dd] = f32_to_bf16(static_cast<float>(u[dd]));
            u[dd] = bf16_to_f32(dst[dd]);
          }
          hadamard256(u, back, false);
          std::copy(back, back + kHd, &sp_.plain[static_cast<size_t>(role) * n + SidePlanes::index(slot, pos, h)]);
        }
      }
    }
  }
  return sp_;
}

}  // namespace

int main() {
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  const fn::Geometry g = geometry();
  std::printf("Flash-Next QSA sparse attention (hq-e8-2b KV): rows decoded to BF16, then gathered\n");
  const Pages pg = make_pages();
  std::vector<uint16_t> q(static_cast<size_t>(kMaxRows) * kQHeads * kHd);
  for (size_t i = 0; i < q.size(); ++i) q[i] = f32_to_bf16(hash_uniform(kQ, i, 3.0F));

  // Each physical row's absolute position (its slot's logical page * 64 + offset).
  std::vector<int32_t> position_of(kRowsPerRole);
  for (int slot = 0; slot < kSlots; ++slot) {
    for (int lp = 0; lp < kLogicalPages; ++lp) {
      const int page = pg.tables[slot * kLogicalPages + lp];
      for (int h = 0; h < kKvHeads; ++h)
        for (int o = 0; o < 64; ++o) position_of[(page * kKvHeads + h) * 64 + o] = lp * 64 + o;
    }
  }

  DeviceBytes dk(pg.k.size() * 2), dv(pg.v.size() * 2), dtables(pg.tables.size() * 4), dpos(position_of.size() * 4);
  upload(dk, pg.k);
  upload(dv, pg.v);
  upload(dtables, pg.tables);
  upload(dpos, position_of);
  DeviceBytes kc(static_cast<size_t>(kRowsPerRole) * 64), km(static_cast<size_t>(kRowsPerRole) * 8);
  DeviceBytes vc(static_cast<size_t>(kRowsPerRole) * 64), vm(static_cast<size_t>(kRowsPerRole) * 8);
  MOE_CUDA(cudaMemset(kc.p, 0, kc.bytes));
  MOE_CUDA(cudaMemset(vc.p, 0, vc.bytes));
  encode_kernel<<<(kRowsPerRole * 2 + 3) / 4, 128>>>(dk.as<__nv_bfloat16>(), dv.as<__nv_bfloat16>(), dpos.as<int32_t>(),
                                                     kc.as<uint8_t>(), km.as<uint8_t>(), vc.as<uint8_t>(), vm.as<uint8_t>());
  DeviceBytes rotated(static_cast<size_t>(kRowsPerRole) * 2 * kHd * 2);
  reference_decode_kernel<<<(kRowsPerRole * 2 + 127) / 128, 128>>>(kc.as<uint8_t>(), km.as<uint8_t>(), vc.as<uint8_t>(),
                                                                   vm.as<uint8_t>(), dpos.as<int32_t>(),
                                                                   rotated.as<__nv_bfloat16>());
  MOE_CUDA(cudaDeviceSynchronize());
  // The expected decoded rows, plain frame, fp64 un-rotation: [role][physical row][256].
  std::vector<double> decoded(static_cast<size_t>(2) * kRowsPerRole * kHd);
  {
    const auto rot = download<uint16_t>(rotated.p, static_cast<size_t>(kRowsPerRole) * 2 * kHd);
    double u[kHd];
    for (int unit = 0; unit < kRowsPerRole * 2; ++unit) {
      for (int dd = 0; dd < kHd; ++dd) u[dd] = bf16_to_f32(rot[static_cast<size_t>(unit) * kHd + dd]);
      hadamard256(u, &decoded[(static_cast<size_t>(unit & 1) * kRowsPerRole + (unit >> 1)) * kHd], false);
    }
  }
  auto decoded_row = [&](int role, int slot, int pos, int head) {
    return &decoded[(static_cast<size_t>(role) * kRowsPerRole + pg.row_at(slot, pos, head) / kHd) * kHd];
  };
  const RowFn exact = exact_rows(pg);
  const RowFn codec = [&](int role, int slot, int pos, int head, double *out) {
    std::copy(decoded_row(role, slot, pos, head), decoded_row(role, slot, pos, head) + kHd, out);
  };

  Device d(std::max<size_t>(sp::partial_bytes(g, 3), 256));
  upload(d.q, q);
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  sp::HqSource hq;
  hq.k_codes = kc.as<uint8_t>();
  hq.k_meta = km.as<uint8_t>();
  hq.v_codes = vc.as<uint8_t>();
  hq.v_meta = vm.as<uint8_t>();
  hq.block_tables = dtables.as<int32_t>();
  hq.logical_pages = kLogicalPages;
  hq.kv_heads = kKvHeads;
  DeviceBytes sk(sp::listed_hq_bytes(g, 3)), sv(sp::listed_hq_bytes(g, 3));
  auto listed = [&](const sp::HqSource &src) -> Enqueue {
    return [&, src](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      sp::KvSource kv;
      SP_OK(sp::decode_listed_hq(g, src, b, sel, sk.as<__nv_bfloat16>(), sv.as<__nv_bfloat16>(), &kv, s));
      SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    };
  };

  Call dec;
  dec.slots = {0, 1, 2};
  dec.positions = {5999, 2100, 1000};
  dec.tokens = 1;
  for (int l = 0; l < 3; ++l) dec.lists.push_back(make_list(dec.positions[l], static_cast<uint32_t>(l + 1)));

  // rows + decode
  const auto eager = run(d, dec, listed(hq), stream);
  {
    const auto got_k = download<uint16_t>(sk.p, sp::listed_hq_bytes(g, 3) / 2);
    const auto got_v = download<uint16_t>(sv.p, sp::listed_hq_bytes(g, 3) / 2);
    double worst = 0.0;
    size_t exact_n = 0, total = 0;
    for (int r = 0; r < 3; ++r) {
      for (size_t i = 0; i < dec.lists[r].size(); ++i) {
        for (int h = 0; h < kKvHeads; ++h) {
          for (int role = 0; role < 2; ++role) {
            const double *want = decoded_row(role, dec.slots[r], dec.lists[r][i], h);
            const uint16_t *got = &(role == 0 ? got_k : got_v)[((static_cast<size_t>(r) * kWidth + i) * kKvHeads + h) * kHd];
            for (int dd = 0; dd < kHd; ++dd) {
              const double u = ulps(got[dd], want[dd]);
              worst = std::max(worst, u);
              exact_n += u == 0.0;
              ++total;
            }
          }
        }
      }
    }
    std::printf("  decoded rows: %zu / %zu values bit-exact with the fp64 un-rotation, worst %.0f ulp\n", exact_n,
                total, worst);
    check(worst <= 1.0, "decoded rows differ from the vendored decode + fp64 un-rotation by more than an ulp");
  }
  check_call("decode over decoded rows", codec, q, dec, eager, 1);
  check(run(d, dec, listed(hq), stream, true) == eager, "hq decode: the graph replay differs from eager");

  // report: the hq route against the BF16 route on the same keys and values.
  {
    DeviceBytes ek(sp::listed_hq_bytes(g, 3)), ev(sp::listed_hq_bytes(g, 3));
    std::vector<uint16_t> hk(sp::listed_hq_bytes(g, 3) / 2, 0), hv(hk.size(), 0);
    double err2 = 0.0, ref2 = 0.0;
    for (int r = 0; r < 3; ++r) {
      for (size_t i = 0; i < dec.lists[r].size(); ++i) {
        for (int h = 0; h < kKvHeads; ++h) {
          const size_t at = pg.row_at(dec.slots[r], dec.lists[r][i], h);
          const size_t dst = ((static_cast<size_t>(r) * kWidth + i) * kKvHeads + h) * kHd;
          std::copy(&pg.k[at], &pg.k[at] + kHd, &hk[dst]);
          std::copy(&pg.v[at], &pg.v[at] + kHd, &hv[dst]);
          for (int role = 0; role < 2; ++role) {
            const double *dec_row = decoded_row(role, dec.slots[r], dec.lists[r][i], h);
            for (int dd = 0; dd < kHd; ++dd) {
              const double x = bf16_to_f32((role == 0 ? pg.k : pg.v)[at + dd]);
              err2 += (dec_row[dd] - x) * (dec_row[dd] - x);
              ref2 += x * x;
            }
          }
        }
      }
    }
    upload(ek, hk);
    upload(ev, hv);
    sp::KvSource bf16;
    bf16.k = ek.as<__nv_bfloat16>();
    bf16.v = ev.as<__nv_bfloat16>();
    bf16.kv_heads = kKvHeads;
    bf16.mode = sp::KvSource::Mode::ByIndex;
    const auto base = run(d, dec, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      SP_OK(sp::attend(g, bf16, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    }, stream);
    double diff2 = 0.0, base2 = 0.0, worst = 0.0;
    for (size_t i = 0; i < base.size(); ++i) {
      const double a = bf16_to_f32(eager[i]), b = bf16_to_f32(base[i]);
      diff2 += (a - b) * (a - b);
      base2 += b * b;
      worst = std::max(worst, std::fabs(a - b));
    }
    std::printf("  hq vs BF16 route, same K/V: output rel. RMS %.4f (max |diff| %.4f); codec row rel. RMS %.4f\n",
                std::sqrt(diff2 / base2), worst, std::sqrt(err2 / ref2));
  }

  // fresh: each lane's own token read exactly from the call's BF16 K/V.
  {
    std::vector<uint16_t> fk(static_cast<size_t>(3) * kKvHeads * kHd), fv(fk.size());
    for (int r = 0; r < 3; ++r) {
      for (int h = 0; h < kKvHeads; ++h) {
        const size_t at = pg.row_at(dec.slots[r], dec.positions[r], h);
        std::copy(&pg.k[at], &pg.k[at] + kHd, &fk[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
        std::copy(&pg.v[at], &pg.v[at] + kHd, &fv[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
      }
    }
    DeviceBytes dfk(fk.size() * 2), dfv(fv.size() * 2);
    upload(dfk, fk);
    upload(dfv, fv);
    sp::HqSource with_fresh = hq;
    with_fresh.fresh_k = dfk.as<__nv_bfloat16>();
    with_fresh.fresh_v = dfv.as<__nv_bfloat16>();
    const RowFn rows = [&](int role, int slot, int pos, int head, double *out) {
      (pos == dec.positions[slot] ? exact : codec)(role, slot, pos, head, out);
    };
    check_call("decode, own token fresh", rows, q, dec, run(d, dec, listed(with_fresh), stream), 1);
  }

  // The side planes as the route will see them: uploaded, and the expected rows.
  struct Uploaded {
    DeviceBytes k, v, ring;
    explicit Uploaded(const SidePlanes &sides) : k(sides.k.size() * 2), v(sides.v.size() * 2), ring(sides.ring.size() * 4) {
      upload(k, sides.k);
      upload(v, sides.v);
      upload(ring, sides.ring);
    }
  };
  auto residual_rows = [&](const SidePlanes &sides, const std::function<bool(int, int)> &is_side,
                           const std::function<bool(int, int)> &is_fresh) -> RowFn {
    return [&, is_side, is_fresh](int role, int slot, int pos, int head, double *out) {
      if (is_fresh(slot, pos)) {
        exact(role, slot, pos, head, out);
      } else if (is_side(slot, pos)) {
        const double *src = &sides.plain[static_cast<size_t>(role) * sides.k.size() + SidePlanes::index(slot, pos, head)];
        std::copy(src, src + kHd, out);
      } else {
        codec(role, slot, pos, head, out);
      }
    };
  };

  // residual: sinks and each lane's recent window [p + 1 - 512, p + 1) from the side planes.
  {
    const auto is_side = [&](int slot, int pos) {
      const int p = dec.positions[slot];
      return pos < kSink || (pos >= p + 1 - kRecent && pos <= p);
    };
    const SidePlanes sides = make_side(pg, is_side, {dec.positions[0], dec.positions[1], dec.positions[2]});
    Uploaded up(sides);
    sp::HqSource with_residual = hq;
    with_residual.residual_k = up.k.as<__nv_bfloat16>();
    with_residual.residual_v = up.v.as<__nv_bfloat16>();
    with_residual.ring_valid = up.ring.as<uint32_t>();
    check_call("decode, residual window", residual_rows(sides, is_side, [](int, int) { return false; }), q, dec,
               run(d, dec, listed(with_residual), stream), 1);
  }

  // prefill + residual: a 200-token chunk at P = 3000 on a 3000-token history, its own rows
  // fresh, the ring as it stands BEFORE the call's append (keys [P - 512, P) and the sinks).
  {
    constexpr int kP = 3000, kChunk = 200;
    Call c;
    c.slots = {0};
    c.positions = {kP};
    c.tokens = kChunk;
    c.max_visible = kP + kChunk;
    for (int r = 0; r < kChunk; ++r) c.lists.push_back(make_list(kP + r, 7));
    std::vector<uint16_t> fk(static_cast<size_t>(kChunk) * kKvHeads * kHd), fv(fk.size());
    for (int r = 0; r < kChunk; ++r) {
      for (int h = 0; h < kKvHeads; ++h) {
        const size_t at = pg.row_at(0, kP + r, h);
        std::copy(&pg.k[at], &pg.k[at] + kHd, &fk[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
        std::copy(&pg.v[at], &pg.v[at] + kHd, &fv[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
      }
    }
    DeviceBytes dfk(fk.size() * 2), dfv(fv.size() * 2);
    upload(dfk, fk);
    upload(dfv, fv);
    const auto is_side = [](int slot, int pos) { return slot == 0 && (pos < kSink || (pos >= kP - kRecent && pos < kP)); };
    const auto is_fresh = [](int slot, int pos) { return slot == 0 && pos >= kP; };
    const SidePlanes sides = make_side(pg, is_side, {kP - 1});
    Uploaded up(sides);
    sp::HqSource src = hq;
    src.fresh_k = dfk.as<__nv_bfloat16>();
    src.fresh_v = dfv.as<__nv_bfloat16>();
    src.residual_k = up.k.as<__nv_bfloat16>();
    src.residual_v = up.v.as<__nv_bfloat16>();
    src.ring_valid = up.ring.as<uint32_t>();
    DeviceBytes pk(sp::visible_hq_bytes(g, c.max_visible)), pv(sp::visible_hq_bytes(g, c.max_visible));
    const auto got = run(d, c, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      sp::KvSource kv;
      SP_OK(sp::decode_visible_hq(g, src, b, pk.as<__nv_bfloat16>(), pv.as<__nv_bfloat16>(), &kv, s));
      SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    }, stream);
    check_call("prefill, fresh + residual (pre-append)", residual_rows(sides, is_side, is_fresh), q, c, got, 9);
  }

  // prefill: one lane's visible rows decoded once per call, the chunk's own rows fresh.
  {
    Call c;
    c.slots = {0};
    c.positions = {5000};
    c.tokens = kMaxRows;
    c.max_visible = 5000 + kMaxRows;
    for (int r = 0; r < kMaxRows; ++r) c.lists.push_back(make_list(5000 + r, 0));
    std::vector<uint16_t> fk(static_cast<size_t>(kMaxRows) * kKvHeads * kHd), fv(fk.size());
    for (int r = 0; r < kMaxRows; ++r) {
      for (int h = 0; h < kKvHeads; ++h) {
        const size_t at = pg.row_at(0, 5000 + r, h);
        std::copy(&pg.k[at], &pg.k[at] + kHd, &fk[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
        std::copy(&pg.v[at], &pg.v[at] + kHd, &fv[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
      }
    }
    DeviceBytes dfk(fk.size() * 2), dfv(fv.size() * 2);
    upload(dfk, fk);
    upload(dfv, fv);
    sp::HqSource src = hq;
    src.fresh_k = dfk.as<__nv_bfloat16>();
    src.fresh_v = dfv.as<__nv_bfloat16>();
    DeviceBytes pk(sp::visible_hq_bytes(g, c.max_visible)), pv(sp::visible_hq_bytes(g, c.max_visible));
    const auto got = run(d, c, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      sp::KvSource kv;
      SP_OK(sp::decode_visible_hq(g, src, b, pk.as<__nv_bfloat16>(), pv.as<__nv_bfloat16>(), &kv, s));
      SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    }, stream);
    const RowFn rows = [&](int role, int slot, int pos, int head, double *out) {
      (pos >= 5000 ? exact : codec)(role, slot, pos, head, out);
    };
    check_call("prefill, visible rows by position", rows, q, c, got, 7);
  }

  // Refusals.
  {
    sp::HqSource half = hq;
    half.fresh_k = d.q.as<__nv_bfloat16>();  // one role only
    fn::Batch b;
    b.lanes = 1;
    b.tokens = 1;
    sp::KvSource kv;
    fn::Selection sel{d.tokens.as<int32_t>(), d.counts.as<int32_t>()};
    check(sp::decode_listed_hq(g, half, b, sel, sk.as<__nv_bfloat16>(), sv.as<__nv_bfloat16>(), &kv, stream) != nullptr,
          "fresh rows for one role must be refused");
    b.lanes = 2;
    check(sp::decode_visible_hq(g, hq, b, sk.as<__nv_bfloat16>(), sv.as<__nv_bfloat16>(), &kv, stream) != nullptr,
          "a two-lane visible decode must be refused");
  }

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_sparse_attention_hq: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_flash_next_sparse_attention_hq: OK\n");
  return 0;
}
