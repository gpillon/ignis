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
// Every arm checks two things apart: the rows the route READ (its scratch, downloaded) against
// their independent expectation to a BF16 ulp (the fp32 un-rotation's one rounding), and the
// attention against fp64 attention over exactly those rows read, within the same derived bound as
// BF16 KV (flash_next_sparse_test_common.h).
//
//   decode    the codec rows, attention over them, a CUDA graph replay;
//   fresh     the call's own token rows read exactly from its BF16 K/V instead of the codec;
//   residual  sink and recent-window rows read from the side planes (rotated frame) instead of
//             the codec, by the 27B's decode rule (no fresh rows: the window is the call's last
//             512 keys, its own among them, as the ring stands after the append);
//   prefill   one lane's visible rows decoded once (the chunk's own rows fresh), read by position;
//   prefill + residual  a 200-token chunk at 3000 with its own rows fresh and the ring as it stands
//             BEFORE the call's append: sinks and [2488, 3000) from the side planes;
//   hq vs BF16 the hq route's distance from the BF16 route on the same keys and values, and the
//             codec's row error, printed on these synthetic rows; then asserted on real Flash-Next KV
//             rows (kernel/tests/fixtures/hq_kv_rows_flash_next.bin, spec 04 AC7) within the codec's
//             measured effect on them, on all three routes: decode, S2's dense prefill
//             (qsa::attend_dense over the visible rows decoded by position) and sparse prefill.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next_sparse_test_common.h"

#include "../src/flash_next/qsa.h"

#include "ops/kernel/hq_codec.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <set>

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

// The real rows' positions per (role, head) in the Flash-Next fixture.
constexpr int kRealPositions = 256;

// The fixture's first layer block (KV layer ordinal 0, model layer 3) as [role][head][position][256]
// BF16 bits: the 27B fixture's format (crates/core/tests/flash_next_kv_fixture_capture_gpu.rs writes it,
// hq_kv_fixture_integrity.rs holds it to its SHA-256). A missing or mis-shaped file is a failure.
std::vector<uint16_t> real_layer_rows(const char *path) {
  std::vector<uint8_t> data;
  if (FILE *f = std::fopen(path, "rb")) {
    std::fseek(f, 0, SEEK_END);
    data.resize(static_cast<size_t>(std::ftell(f)));
    std::fseek(f, 0, SEEK_SET);
    data.resize(std::fread(data.data(), 1, data.size(), f));
    std::fclose(f);
  }
  auto word = [&](size_t i) {
    uint32_t w = 0;
    if (8 + 4 * (i + 1) <= data.size()) std::memcpy(&w, data.data() + 8 + 4 * i, 4);
    return w;
  };
  // magic, then version, head_dim, kv_heads, roles, layers, ordinals[layers], first, rows, total, checksum.
  const uint32_t layers = word(4);
  const size_t header = 8 + 4 * (5 + layers + 3) + 8;
  const bool shaped = data.size() >= 8 && std::memcmp(data.data(), "IGNHQKV1", 8) == 0 && word(0) == 1 &&
                      word(1) == kHd && word(2) == kKvHeads && word(3) == 2 && layers > 0 && word(5) == 0 &&
                      word(5 + layers) == 0 && word(6 + layers) == kRealPositions &&
                      data.size() == header + static_cast<size_t>(word(7 + layers)) * kHd * 2;
  if (!shaped) {
    std::fprintf(stderr, "FATAL: %s is missing or not the Flash-Next KV fixture (re-run "
                         "flash_next_kv_fixture_capture_gpu.rs)\n", path);
    std::exit(1);
  }
  std::vector<uint16_t> rows(static_cast<size_t>(2) * kKvHeads * kRealPositions * kHd);
  std::memcpy(rows.data(), data.data() + header, rows.size() * 2);
  return rows;
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

  // What the route read, as a RowFn over its downloaded scratch: by list index for a decode call
  // (one row per slot), by position for a prefill call. The attention arms are checked over THESE
  // rows (tight, like BF16 KV); the rows themselves are checked against their independent
  // expectation (fresh: the BF16 rows; side: the fp64 un-rotation of the stored side rows; codec:
  // the vendored serial decoder + fp64 un-rotation) to a BF16 ulp, the one rounding the route adds.
  struct Read {
    std::vector<uint16_t> k, v;
    std::vector<std::map<int32_t, int32_t>> index;  // listed: per row, position -> list index
  };
  auto read_listed = [&](const Call &c) {
    Read rd;
    rd.k = download<uint16_t>(sk.p, sp::listed_hq_bytes(g, 3) / 2);
    rd.v = download<uint16_t>(sv.p, sp::listed_hq_bytes(g, 3) / 2);
    rd.index.resize(c.lists.size());
    for (size_t r = 0; r < c.lists.size(); ++r)
      for (size_t i = 0; i < c.lists[r].size(); ++i) rd.index[r][c.lists[r][i]] = static_cast<int32_t>(i);
    return rd;
  };
  auto listed_rows = [&](const Call &c, const Read &rd) -> RowFn {
    return [&c, &rd](int role, int slot, int pos, int head, double *out) {
      const size_t r = static_cast<size_t>(std::find(c.slots.begin(), c.slots.end(), slot) - c.slots.begin());
      const size_t at = ((r * kWidth + rd.index[r].at(pos)) * kKvHeads + head) * kHd;
      for (int dd = 0; dd < kHd; ++dd) out[dd] = bf16_to_f32((role == 0 ? rd.k : rd.v)[at + dd]);
    };
  };
  auto positional_rows = [](const Read &rd) -> RowFn {
    return [&rd](int role, int, int pos, int head, double *out) {
      const size_t at = (static_cast<size_t>(pos) * kKvHeads + head) * kHd;
      for (int dd = 0; dd < kHd; ++dd) out[dd] = bf16_to_f32((role == 0 ? rd.k : rd.v)[at + dd]);
    };
  };
  // Every row a call's lists name, read against its expectation, to a BF16 ulp.
  auto rows_within_ulp = [&](const std::string &what, const Call &c, const RowFn &read, const RowFn &want) {
    std::set<std::pair<int, int>> seen;
    for (size_t r = 0; r < c.lists.size(); ++r)
      for (int32_t pos : c.lists[r]) seen.emplace(c.slots[r / c.tokens], pos);
    size_t exact_n = 0, total = 0, past = 0;
    double a[kHd], b[kHd];
    for (const auto &[slot, pos] : seen) {
      for (int h = 0; h < kKvHeads; ++h) {
        for (int role = 0; role < 2; ++role) {
          read(role, slot, pos, h, a);
          want(role, slot, pos, h, b);
          for (int dd = 0; dd < kHd; ++dd) {
            const double ulp = b[dd] == 0.0 ? 0x1p-133 : std::ldexp(1.0, std::ilogb(b[dd]) - 7);
            exact_n += a[dd] == static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(b[dd]))));
            past += !(std::fabs(a[dd] - b[dd]) <= ulp);
            ++total;
          }
        }
      }
    }
    std::printf("  %-40s rows read: %zu / %zu values the BF16 rounding of the expectation, %zu past an ulp\n",
                what.c_str(), exact_n, total, past);
    check(past == 0, what + ": a row the route read differs from its expectation by more than a BF16 ulp");
  };

  Call dec;
  dec.slots = {0, 1, 2};
  dec.positions = {5999, 2100, 1000};
  dec.tokens = 1;
  for (int l = 0; l < 3; ++l) dec.lists.push_back(make_list(dec.positions[l], static_cast<uint32_t>(l + 1)));

  // decode: the codec rows, then attention over what was read; a graph replay.
  const auto eager = run(d, dec, listed(hq), stream);
  {
    const Read rd = read_listed(dec);
    rows_within_ulp("decode", dec, listed_rows(dec, rd), codec);
    check_call("decode over the rows read", listed_rows(dec, rd), q, dec, eager, 1);
  }
  check(run(d, dec, listed(hq), stream, true) == eager, "hq decode: the graph replay differs from eager");

  // hq vs BF16: the hq route against the BF16 route on the same keys and values, as figures. Spec 04
  // AC7's tolerance comes from the codec error on real Flash-Next KV rows (needs the artifact); a
  // worst-case bound from these synthetic rows would be too loose to fail, so none is asserted here.
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
    std::printf("  hq vs BF16 route, same K/V (synthetic rows): output rel. RMS %.4f (max |diff| %.4f); "
                "codec row rel. RMS %.4f\n",
                std::sqrt(diff2 / base2), worst, std::sqrt(err2 / ref2));
  }

  // hq vs BF16 on real Flash-Next rows (spec 04 AC7): every page refilled with the K/V rows of the first
  // QSA layer (model layer 3) from kernel/tests/fixtures/hq_kv_rows_flash_next.bin, its 256 positions per
  // head tiled over the pages, and encoded by the same vendored encoder at each row's own position. The
  // three routes, each run hq and BF16 on the same rows, codec only (no fresh rows, no residual window:
  // every row hq reads is a codec row, so every difference is the codec's):
  //   decode  the three decode rows above: their listed rows on the sparse kernel;
  //   dense   one lane's prefill chunk of 251 rows at 1800, up to dense_threshold(): its visible rows
  //           decoded by position, on the dense kernel (qsa_dense.cu) -- every 25th row checked;
  //   sparse prefill  one lane's 128 rows at 5000, past dense_threshold(): its visible rows decoded
  //           by position, each row's own selection on the sparse kernel -- every 16th row checked.
  // Held, per route:
  // - each output within reference()'s per-element bound of fp64 attention over the rows the route
  //   read (hq: its scratch, downloaded; BF16: the pages), the dense kernel's form of it on the dense
  //   route;
  // - per element, hq within the BF16 output plus the codec's MEASURED effect on it -- fp64 attention
  //   over the rows hq read minus fp64 attention over the BF16 rows -- plus both bounds: AC7's
  //   tolerance, derived from the codec error measured on these rows;
  // - in aggregate, what is left of the route distance after that effect is rounding, at most 2^-7
  //   relative RMS. Not a worst-case codec bound (0312e0b: that one grows as e^(2 max |q.dK|) and
  //   cannot fail): a route whose hq error is anything but the codec's own fails it.
  // The per-element bound held here is reference()'s with BF16 rounding at 2^-8 (#306 item 8): at
  // 2^-9 the BF16 route itself passed it on 75 of these 18432 decode outputs, by up to 1.56x.
  // Measured 2026-10-07: decode -- codec effect rel. RMS 0.1480, route distance 0.1480, 2.4e-3
  // left; worst of the bound hq 0.754, BF16 0.798; dense -- 0.1404, 0.1404, 2.4e-3 left; hq 0.508,
  // BF16 0.517; sparse prefill -- 0.1288, 0.1288, 2.4e-3 left; hq 0.741, BF16 0.798. Against the
  // codec tolerance every route reaches 0.99 by construction: where the effect dominates, the
  // distance is the effect and the bounds are the margin.
  {
    const std::vector<uint16_t> real = real_layer_rows(IGNIS_HQ_KV_FLASH_NEXT_FIXTURE_PATH);
    Pages rp = pg;
    for (int row = 0; row < kRowsPerRole; ++row) {
      const int head = (row / 64) % kKvHeads, src = position_of[row] % kRealPositions;
      for (int role = 0; role < 2; ++role) {
        const uint16_t *from = &real[((static_cast<size_t>(role) * kKvHeads + head) * kRealPositions + src) * kHd];
        std::copy(from, from + kHd, &(role == 0 ? rp.k : rp.v)[static_cast<size_t>(row) * kHd]);
      }
    }
    DeviceBytes rk(rp.k.size() * 2), rv(rp.v.size() * 2);
    upload(rk, rp.k);
    upload(rv, rp.v);
    DeviceBytes rkc(kc.bytes), rkm(km.bytes), rvc(vc.bytes), rvm(vm.bytes);
    MOE_CUDA(cudaMemset(rkc.p, 0, rkc.bytes));
    MOE_CUDA(cudaMemset(rvc.p, 0, rvc.bytes));
    encode_kernel<<<(kRowsPerRole * 2 + 3) / 4, 128>>>(rk.as<__nv_bfloat16>(), rv.as<__nv_bfloat16>(), dpos.as<int32_t>(),
                                                       rkc.as<uint8_t>(), rkm.as<uint8_t>(), rvc.as<uint8_t>(), rvm.as<uint8_t>());
    reference_decode_kernel<<<(kRowsPerRole * 2 + 127) / 128, 128>>>(rkc.as<uint8_t>(), rkm.as<uint8_t>(), rvc.as<uint8_t>(),
                                                                     rvm.as<uint8_t>(), dpos.as<int32_t>(),
                                                                     rotated.as<__nv_bfloat16>());
    MOE_CUDA(cudaDeviceSynchronize());
    std::vector<double> rdecoded(decoded.size());
    {
      const auto rot = download<uint16_t>(rotated.p, static_cast<size_t>(kRowsPerRole) * 2 * kHd);
      double u[kHd];
      for (int unit = 0; unit < kRowsPerRole * 2; ++unit) {
        for (int dd = 0; dd < kHd; ++dd) u[dd] = bf16_to_f32(rot[static_cast<size_t>(unit) * kHd + dd]);
        hadamard256(u, &rdecoded[(static_cast<size_t>(unit & 1) * kRowsPerRole + (unit >> 1)) * kHd], false);
      }
    }
    const RowFn rexact = exact_rows(rp);
    const RowFn rcodec = [&](int role, int slot, int pos, int head, double *out) {
      const double *from = &rdecoded[(static_cast<size_t>(role) * kRowsPerRole + rp.row_at(slot, pos, head) / kHd) * kHd];
      std::copy(from, from + kHd, out);
    };
    sp::HqSource rh = hq;
    rh.k_codes = rkc.as<uint8_t>();
    rh.k_meta = rkm.as<uint8_t>();
    rh.v_codes = rvc.as<uint8_t>();
    rh.v_meta = rvm.as<uint8_t>();

    // One route's hq output (over `hq_rows`, what it read) against its BF16 output (over the pages'
    // rows), every `step`-th row of call `c`.
    auto routes = [&](const std::string &what, const Call &c, const std::vector<uint16_t> &hq_out, const RowFn &hq_rows,
                      const std::vector<uint16_t> &bf_out, int step, bool dense) {
      double effect2 = 0.0, out2 = 0.0, diff2 = 0.0, left2 = 0.0, worst_hq = 0.0, worst_bf = 0.0, worst_tol = 0.0;
      size_t past_hq = 0, past_bf = 0, past_tol = 0, total = 0;
      for (int r = 0; r < static_cast<int>(c.lists.size()); r += step) {
        const int slot = c.slots[r / c.tokens];
        const Reference hq_ref = reference(hq_rows, q, r, slot, c.lists[r], dense);
        const Reference bf_ref = reference(rexact, q, r, slot, c.lists[r], dense);
        for (size_t i = 0; i < hq_ref.out.size(); ++i) {
          const size_t at = static_cast<size_t>(r) * kQHeads * kHd + i;
          const double h = bf16_to_f32(hq_out[at]), b = bf16_to_f32(bf_out[at]);
          const double effect = hq_ref.out[i] - bf_ref.out[i], diff = h - b;
          const double tol = std::fabs(effect) + hq_ref.bound[i] + bf_ref.bound[i];
          effect2 += effect * effect;
          out2 += bf_ref.out[i] * bf_ref.out[i];
          diff2 += diff * diff;
          left2 += (diff - effect) * (diff - effect);
          past_hq += !(std::fabs(h - hq_ref.out[i]) <= hq_ref.bound[i]);
          past_bf += !(std::fabs(b - bf_ref.out[i]) <= bf_ref.bound[i]);
          past_tol += !(std::fabs(diff) <= tol);
          worst_hq = std::max(worst_hq, std::fabs(h - hq_ref.out[i]) / hq_ref.bound[i]);
          worst_bf = std::max(worst_bf, std::fabs(b - bf_ref.out[i]) / bf_ref.bound[i]);
          worst_tol = std::max(worst_tol, std::fabs(diff) / tol);
          ++total;
        }
      }
      const double left = std::sqrt(left2 / out2);
      std::printf("  %-28s codec effect on the fp64 output rel. RMS %.4f; route distance %.4f; left after the "
                  "effect %.2e (limit 2^-7)\n",
                  what.c_str(), std::sqrt(effect2 / out2), std::sqrt(diff2 / out2), left);
      std::printf("  %-28s of the per-element bound: hq worst %.3f, BF16 worst %.3f; hq vs BF16 worst %.3f of the "
                  "codec tolerance; %zu elements\n",
                  what.c_str(), worst_hq, worst_bf, worst_tol, total);
      check(past_hq == 0, what + ": " + std::to_string(past_hq) + " hq outputs past the bound of fp64 over the rows read");
      check(past_bf == 0, what + ": " + std::to_string(past_bf) + " BF16 outputs past the bound of fp64 over the rows");
      check(past_tol == 0, what + ": " + std::to_string(past_tol) +
                               " hq outputs past the BF16 one plus the codec's measured effect and both bounds");
      check(left <= 0x1p-7, what + ": the route distance is not the codec's measured effect");
    };

    // decode: the listed rows, hq through its scratch and BF16 staged by list index.
    {
      const auto got = run(d, dec, listed(rh), stream);
      const Read rd = read_listed(dec);
      rows_within_ulp("decode, real rows", dec, listed_rows(dec, rd), rcodec);
      std::vector<uint16_t> hk(sp::listed_hq_bytes(g, 3) / 2, 0), hv(hk.size(), 0);
      double row_err2 = 0.0, row_ref2 = 0.0;
      for (int r = 0; r < 3; ++r) {
        for (size_t i = 0; i < dec.lists[r].size(); ++i) {
          for (int h = 0; h < kKvHeads; ++h) {
            const size_t at = rp.row_at(dec.slots[r], dec.lists[r][i], h);
            const size_t dst = ((static_cast<size_t>(r) * kWidth + i) * kKvHeads + h) * kHd;
            std::copy(&rp.k[at], &rp.k[at] + kHd, &hk[dst]);
            std::copy(&rp.v[at], &rp.v[at] + kHd, &hv[dst]);
            double a[kHd], b[kHd];
            for (int role = 0; role < 2; ++role) {
              rcodec(role, dec.slots[r], dec.lists[r][i], h, a);
              rexact(role, dec.slots[r], dec.lists[r][i], h, b);
              for (int dd = 0; dd < kHd; ++dd) {
                row_err2 += (a[dd] - b[dd]) * (a[dd] - b[dd]);
                row_ref2 += b[dd] * b[dd];
              }
            }
          }
        }
      }
      DeviceBytes ek(hk.size() * 2), ev(hv.size() * 2);
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
      std::printf("  real layer-3 rows: codec row rel. L2 %.4f\n", std::sqrt(row_err2 / row_ref2));
      routes("decode, real rows", dec, got, listed_rows(dec, rd), base, 1, false);
    }

    // dense: one lane's visible rows, hq decoded by position, BF16 from the pages.
    {
      constexpr int kFirst = 1800, kTokens = 251;
      Call dc;
      dc.slots = {0};
      dc.positions = {kFirst};
      dc.tokens = kTokens;
      dc.max_visible = kFirst + kTokens;
      for (int r = 0; r < kTokens; ++r) dc.lists.push_back(make_list(kFirst + r, 0));
      check(dc.max_visible <= g.dense_threshold(), "dense, real rows: the chunk is within dense_threshold()");
      DeviceBytes pk(sp::visible_hq_bytes(g, dc.max_visible)), pv(sp::visible_hq_bytes(g, dc.max_visible));
      const auto got = run(d, dc, [&](const fn::Batch &b, const fn::Selection &, cudaStream_t s) {
        sp::KvSource kv;
        SP_OK(sp::decode_visible_hq(g, rh, b, pk.as<__nv_bfloat16>(), pv.as<__nv_bfloat16>(), &kv, s));
        SP_OK(fn::qsa::attend_dense(g, kv, kSlots, b, d.q.as<__nv_bfloat16>(), d.out.as<__nv_bfloat16>(), s));
      }, stream);
      Read rd;
      rd.k = download<uint16_t>(pk.p, sp::visible_hq_bytes(g, dc.max_visible) / 2);
      rd.v = download<uint16_t>(pv.p, sp::visible_hq_bytes(g, dc.max_visible) / 2);
      rows_within_ulp("dense, real rows", dc, positional_rows(rd), rcodec);
      const sp::KvSource pages{rk.as<__nv_bfloat16>(), rv.as<__nv_bfloat16>(), dtables.as<int32_t>(), kLogicalPages,
                               kKvHeads, sp::KvSource::Mode::Paged};
      const auto base = run(d, dc, [&](const fn::Batch &b, const fn::Selection &, cudaStream_t s) {
        SP_OK(fn::qsa::attend_dense(g, pages, kSlots, b, d.q.as<__nv_bfloat16>(), d.out.as<__nv_bfloat16>(), s));
      }, stream);
      routes("dense, real rows", dc, got, positional_rows(rd), base, 25, true);
    }

    // sparse prefill: one lane's chunk past dense_threshold(), each row its own selection over its
    // visible rows -- hq decoded by position, BF16 from the pages -- on the sparse kernel.
    {
      constexpr int kFirst = 5000, kTokens = 128;  // 256 units: one split, no partials
      Call sc;
      sc.slots = {0};
      sc.positions = {kFirst};
      sc.tokens = kTokens;
      sc.max_visible = kFirst + kTokens;
      for (int r = 0; r < kTokens; ++r) sc.lists.push_back(make_list(kFirst + r, 11));
      DeviceBytes pk(sp::visible_hq_bytes(g, sc.max_visible)), pv(sp::visible_hq_bytes(g, sc.max_visible));
      const auto got = run(d, sc, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
        sp::KvSource kv;
        SP_OK(sp::decode_visible_hq(g, rh, b, pk.as<__nv_bfloat16>(), pv.as<__nv_bfloat16>(), &kv, s));
        SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
      }, stream);
      Read rd;
      rd.k = download<uint16_t>(pk.p, sp::visible_hq_bytes(g, sc.max_visible) / 2);
      rd.v = download<uint16_t>(pv.p, sp::visible_hq_bytes(g, sc.max_visible) / 2);
      rows_within_ulp("sparse prefill, real rows", sc, positional_rows(rd), rcodec);
      const sp::KvSource pages{rk.as<__nv_bfloat16>(), rv.as<__nv_bfloat16>(), dtables.as<int32_t>(), kLogicalPages,
                               kKvHeads, sp::KvSource::Mode::Paged};
      const auto base = run(d, sc, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
        SP_OK(sp::attend(g, pages, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
      }, stream);
      routes("sparse prefill, real rows", sc, got, positional_rows(rd), base, 16, false);
    }
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
    const RowFn want = [&](int role, int slot, int pos, int head, double *out) {
      (pos == dec.positions[slot] ? exact : codec)(role, slot, pos, head, out);
    };
    const auto got = run(d, dec, listed(with_fresh), stream);
    const Read rd = read_listed(dec);
    rows_within_ulp("decode, own token fresh", dec, listed_rows(dec, rd), want);
    check_call("decode, own token fresh", listed_rows(dec, rd), q, dec, got, 1);
  }

  // The side planes as the route will see them, uploaded; and the expectation of a call's rows.
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
    const auto got = run(d, dec, listed(with_residual), stream);
    const Read rd = read_listed(dec);
    rows_within_ulp("decode, residual window", dec, listed_rows(dec, rd),
                    residual_rows(sides, is_side, [](int, int) { return false; }));
    check_call("decode, residual window", listed_rows(dec, rd), q, dec, got, 1);
  }

  // A prefill call through the visible-row decode, read back by position.
  auto prefill = [&](const std::string &what, const sp::HqSource &src, const Call &c, const RowFn &want, int step) {
    DeviceBytes pk(sp::visible_hq_bytes(g, c.max_visible)), pv(sp::visible_hq_bytes(g, c.max_visible));
    const auto got = run(d, c, [&](const fn::Batch &b, const fn::Selection &sel, cudaStream_t s) {
      sp::KvSource kv;
      SP_OK(sp::decode_visible_hq(g, src, b, pk.as<__nv_bfloat16>(), pv.as<__nv_bfloat16>(), &kv, s));
      SP_OK(sp::attend(g, kv, b, d.q.as<__nv_bfloat16>(), sel, d.out.as<__nv_bfloat16>(), d.partials.p, s));
    }, stream);
    Read rd;
    rd.k = download<uint16_t>(pk.p, sp::visible_hq_bytes(g, c.max_visible) / 2);
    rd.v = download<uint16_t>(pv.p, sp::visible_hq_bytes(g, c.max_visible) / 2);
    rows_within_ulp(what, c, positional_rows(rd), want);
    check_call(what, positional_rows(rd), q, c, got, step);
  };
  // The call's own BF16 K/V rows [tokens][kv_heads][256], as fn_qsa_attention hands them over.
  auto own_rows = [&](int first, int tokens, std::vector<uint16_t> &fk, std::vector<uint16_t> &fv) {
    fk.assign(static_cast<size_t>(tokens) * kKvHeads * kHd, 0);
    fv.assign(fk.size(), 0);
    for (int r = 0; r < tokens; ++r) {
      for (int h = 0; h < kKvHeads; ++h) {
        const size_t at = pg.row_at(0, first + r, h);
        std::copy(&pg.k[at], &pg.k[at] + kHd, &fk[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
        std::copy(&pg.v[at], &pg.v[at] + kHd, &fv[(static_cast<size_t>(r) * kKvHeads + h) * kHd]);
      }
    }
  };

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
    std::vector<uint16_t> fk, fv;
    own_rows(kP, kChunk, fk, fv);
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
    prefill("prefill, fresh + residual (pre-append)", src, c, residual_rows(sides, is_side, is_fresh), 9);
  }

  // prefill: one lane's visible rows decoded once per call, the chunk's own rows fresh.
  {
    Call c;
    c.slots = {0};
    c.positions = {5000};
    c.tokens = kMaxRows;
    c.max_visible = 5000 + kMaxRows;
    for (int r = 0; r < kMaxRows; ++r) c.lists.push_back(make_list(5000 + r, 0));
    std::vector<uint16_t> fk, fv;
    own_rows(5000, kMaxRows, fk, fv);
    DeviceBytes dfk(fk.size() * 2), dfv(fv.size() * 2);
    upload(dfk, fk);
    upload(dfv, fv);
    sp::HqSource src = hq;
    src.fresh_k = dfk.as<__nv_bfloat16>();
    src.fresh_v = dfv.as<__nv_bfloat16>();
    const RowFn want = [&](int role, int slot, int pos, int head, double *out) {
      (pos >= 5000 ? exact : codec)(role, slot, pos, head, out);
    };
    prefill("prefill, visible rows by position", src, c, want, 7);
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
