// GitHub #688 (spec #684): an extended round's attention over hq-e8-2b, 9..16
// columns a lane at 1..4 lanes, on ignis's warp-specialized route in chunks of
// the small-T tile against the vendored chunked small-T route, byte for byte --
// OURS. Both run the one chunk loop of the batch body
// (`ignis::extended_attention::attend`, kernel/src/extended_attention.h), on
// its two routes.
//
// ADR 0093 holds the route to the vendored kernel's bytes up to 8 columns a
// lane (kernel/tests/test_hq_verify_exact.cu, at the kernel's own launch, on
// its partials). Past 8 columns the vendored route is ChunkedSmallT: the lane's
// columns in chunks of the 8-column tile, each chunk appending its own K/V and
// attending from its first column of the lane's width, each with its own
// reducer launch. This test is that contract at the launcher's seam, where the
// round calls it (kernel/src/gqa_layer.cu `run_gqa_layer_batch`): the same q,
// k, v, gate, positions, valid columns, table rows, cache view, envelope and
// workspace go through the oracle and through the loop on ignis's route, from
// the same cache bytes, and every byte of the output and every byte of the
// cache after the call must be equal -- the code and metadata planes the fused
// append writes, the residual window's rows and the ring's valid marks.
//
// The oracle, by lane count, is what the vendored wrapper runs:
//   2..4 lanes  `ninfer::ops::gqa_attention`, whose resolver gives
//               ChunkedSmallT to every call of two lanes or more past the tile;
//   1 lane      the loop on the vendored chunks (the exported
//               `detail::gqa_attention_small_t_launch` per chunk), since the
//               resolver sends one lane of the 27B's 24 heads to the prompt
//               route instead. The loop on the vendored chunks is itself held
//               to `gqa_attention` at 2..4 lanes on every case, so the one-lane
//               oracle is the vendored route and not a second opinion, and an
//               extended round of 2..4 lanes on the vendored chunks runs what
//               the wrapper would.
//
// Cases: every width 9..16 at 1..4 lanes, dense (no valid-column tensor) and
// masked. A verify round is always masked (valid = a lane's extent + 1, never
// 0); the masks rotate six prefixes over the lanes so that every lane count
// and every lane position sees a full lane, a lane that ends at the chunk
// boundary (8), one column past it (9), inside chunk 1 (3), at the anchor
// alone (1) and inside chunk 2 short of the width, with no two lanes of a
// case masked alike. The
// lanes' histories differ (two past the 512-key ring, one inside it, one
// barely past the sinks), and the even lanes' ring slots under the call's own
// columns are cleared first, as a rollback leaves them, so the marks the
// append sets are bytes a route writes; the odd lanes keep the marks a prefill
// left, a sequence's first round. The envelope is the pool's context, as the
// decode graph declares it, which gives each chunk several splits.
//
// A case that compared nothing would pass, so the oracle's own result is
// checked on every case: every valid column's output left its sentinel and is
// not zero, every masked column's output is exact zero, the append changed
// code bytes and ring marks. And ignis's route runs every dense case a second
// time over a workspace of other bytes: a partial read before it was written
// would move the output.
//
// The route is Blackwell's (its image is in the sm_120a set only), so this
// test carries the sm120a-only label and fails on a card without the image.
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU, a
// kernel error or a missing fixture fails this test.

#include "extended_attention.h"
#include "hq_decode_attention.h"
#include "ignis_hq_ring.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#ifndef IGNIS_HQ_KV_FIXTURE_PATH
#error "IGNIS_HQ_KV_FIXTURE_PATH must be defined by kernel/tests/CMakeLists.txt"
#endif

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

#define CUDA_FATAL(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));          \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

// ---- geometry ----------------------------------------------------------------

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQHeads     = 24;
constexpr std::int32_t kKvHeads    = 4;
constexpr std::int32_t kGqaOrdinal = 0;
const float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

// The extended round's shapes: past the 8-column tile, up to two of them, at up
// to four lanes (64 columns a round).
constexpr std::int32_t kMinWidth = 9;
constexpr std::int32_t kMaxWidth = 16;
constexpr std::int32_t kMaxLanes = 4;
// Each lane's context, the envelope every call declares, and the keys each
// lane holds before the call: past the ring twice, inside it, barely past the
// 32 sinks.
constexpr std::uint32_t kContext = 4096;
constexpr std::int32_t kHistory[kMaxLanes] = {2500, 700, 131, 40};

// ---- the committed real-row fixture ---------------------------------------------
//
// kernel/tests/fixtures/hq_kv_rows_27b.bin, rows in the order
// layer -> role -> kv_head -> position -> head_dim. Its full validation
// (version, checksum) is test_hq_codec_kv_rows.cu's; here a wrong file still
// fails.
struct Fixture {
  std::uint32_t kv_heads       = 0;
  std::uint32_t role_count     = 0;
  std::uint32_t layer_count    = 0;
  std::uint32_t rows_per_block = 0;
  std::vector<std::uint16_t> rows;

  const std::uint16_t *row(std::uint32_t layer, std::uint32_t role, std::uint32_t kv_head,
                           std::uint32_t position) const {
    const std::size_t block =
        ((static_cast<std::size_t>(layer) * role_count + role) * kv_heads + kv_head) * rows_per_block;
    return rows.data() + (block + position) * kHeadDim;
  }
};

Fixture load_fixture() {
  std::FILE *f = std::fopen(IGNIS_HQ_KV_FIXTURE_PATH, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "FATAL: cannot open fixture %s\n", IGNIS_HQ_KV_FIXTURE_PATH);
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::uint8_t> data;
  std::fseek(f, 0, SEEK_END);
  data.resize(static_cast<std::size_t>(std::ftell(f)));
  std::fseek(f, 0, SEEK_SET);
  const bool read_ok = std::fread(data.data(), 1, data.size(), f) == data.size();
  std::fclose(f);
  auto u32 = [&](std::size_t off) {
    std::uint32_t v;
    std::memcpy(&v, data.data() + off, 4);
    return v;
  };
  // magic, format_version, head_dim, kv_heads, role_count, layer_count,
  // layer ordinals, first_position, rows_per_block, total_rows, checksum.
  if (!read_ok || data.size() < 28 || std::memcmp(data.data(), "IGNHQKV1", 8) != 0 ||
      u32(12) != static_cast<std::uint32_t>(kHeadDim) || u32(16) != static_cast<std::uint32_t>(kKvHeads)) {
    std::fprintf(stderr, "FATAL: %s is not the 27B IGNHQKV1 fixture\n", IGNIS_HQ_KV_FIXTURE_PATH);
    std::exit(EXIT_FAILURE);
  }
  Fixture fx;
  fx.kv_heads    = u32(16);
  fx.role_count  = u32(20);
  fx.layer_count = u32(24);
  std::size_t off = 28 + 4 * static_cast<std::size_t>(fx.layer_count);
  if (data.size() < off + 20) {
    std::fprintf(stderr, "FATAL: %s has a truncated header\n", IGNIS_HQ_KV_FIXTURE_PATH);
    std::exit(EXIT_FAILURE);
  }
  fx.rows_per_block = u32(off + 4);
  const std::uint32_t total_rows = u32(off + 8);
  off += 20;
  const std::uint64_t expected =
      static_cast<std::uint64_t>(fx.layer_count) * fx.role_count * fx.kv_heads * fx.rows_per_block;
  if (fx.role_count != 2 || expected != total_rows ||
      data.size() != off + static_cast<std::size_t>(total_rows) * kHeadDim * 2) {
    std::fprintf(stderr, "FATAL: %s has an inconsistent or truncated payload\n", IGNIS_HQ_KV_FIXTURE_PATH);
    std::exit(EXIT_FAILURE);
  }
  fx.rows.resize(static_cast<std::size_t>(total_rows) * kHeadDim);
  std::memcpy(fx.rows.data(), data.data() + off, fx.rows.size() * 2);
  return fx;
}

// K and V for `tokens` rows, [head_dim, kv_heads, tokens], token `t` the
// fixture's row at `first + t`, walking its layers past a block.
void fill_kv(const Fixture &fx, std::int32_t tokens, std::int32_t first, std::vector<std::uint16_t> &k,
             std::vector<std::uint16_t> &v) {
  k.assign(static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens, 0);
  v.assign(k.size(), 0);
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t h = 0; h < kKvHeads; ++h) {
      const std::size_t dst = (static_cast<std::size_t>(t) * kKvHeads + static_cast<std::size_t>(h)) * kHeadDim;
      const auto absolute = static_cast<std::uint32_t>(first + t);
      const auto layer    = (absolute / fx.rows_per_block) % fx.layer_count;
      const auto position = absolute % fx.rows_per_block;
      std::memcpy(k.data() + dst, fx.row(layer, 0, static_cast<std::uint32_t>(h), position),
                  static_cast<std::size_t>(kHeadDim) * 2);
      std::memcpy(v.data() + dst, fx.row(layer, 1, static_cast<std::uint32_t>(h), position),
                  static_cast<std::size_t>(kHeadDim) * 2);
    }
  }
}

// Queries [head_dim, q_heads, width, lanes] from the fixture's K rows (post-RoPE
// q and k leave the same fused norm), a different source row per head, column
// and lane, so no two lanes and no two columns ask the same question.
void fill_q(const Fixture &fx, std::int32_t width, std::int32_t lanes, std::vector<std::uint16_t> &q) {
  q.assign(static_cast<std::size_t>(kHeadDim) * kQHeads * width * lanes, 0);
  for (std::int32_t b = 0; b < lanes; ++b) {
    for (std::int32_t w = 0; w < width; ++w) {
      for (std::int32_t h = 0; h < kQHeads; ++h) {
        const std::size_t dst = ((static_cast<std::size_t>(b) * width + w) * kQHeads + h) * kHeadDim;
        const auto layer = static_cast<std::uint32_t>((h / kKvHeads) % fx.layer_count);
        const auto head  = static_cast<std::uint32_t>(h % kKvHeads);
        const auto pos   = static_cast<std::uint32_t>((w * 5 + b * 31 + 7) % fx.rows_per_block);
        std::memcpy(q.data() + dst, fx.row(layer, 0, head, pos), static_cast<std::size_t>(kHeadDim) * 2);
      }
    }
  }
}

// The sigmoid gate, a smooth spread: a constant would hide a dropped gate.
void fill_gate(std::size_t elements, std::vector<std::uint16_t> &gate) {
  gate.resize(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    const __nv_bfloat16 value = __float2bfloat16(std::sin(static_cast<float>(i) * 0.37F) * 1.5F);
    std::memcpy(&gate[i], &value, sizeof(value));
  }
}

struct DeviceBytes {
  void *p = nullptr;
  std::size_t bytes = 0;
  explicit DeviceBytes(std::size_t n) : bytes(std::max<std::size_t>(n, 1)) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &)            = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
  template <typename T>
  void upload(const std::vector<T> &host) {
    CUDA_FATAL(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
};

std::vector<std::uint8_t> to_host(const void *device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  if (bytes != 0) { CUDA_FATAL(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost)); }
  return host;
}

std::size_t differing(const std::vector<std::uint8_t> &a, const std::vector<std::uint8_t> &b) {
  if (a.size() != b.size()) { return std::max(a.size(), b.size()); }
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { n += a[i] != b[i] ? 1 : 0; }
  return n;
}

// ---- the cache a call reads and writes ----------------------------------------
//
// One GQA layer's planes of the pool, through the production view
// (`ignis_kv_batch_layer_view`): the code and metadata planes, the residual
// window's rows and the ring's valid words.
constexpr int kPlaneCount = 7;
const char *const kPlaneNames[kPlaneCount] = {"K codes",    "V codes",    "K metadata", "V metadata",
                                              "residual K", "residual V", "ring marks"};
constexpr int kRingPlane = 6;

struct Planes {
  std::vector<std::uint8_t> bytes[kPlaneCount];
};

const ninfer::Tensor &plane(const ninfer::PagedKVBatchLayerView &cache, int index) {
  switch (index) {
  case 0: return cache.k_pages;
  case 1: return cache.v_pages;
  case 2: return cache.k_scale_pages;
  case 3: return cache.v_scale_pages;
  case 4: return cache.residual_k;
  case 5: return cache.residual_v;
  default: return cache.ring_valid;
  }
}

Planes snapshot(const ninfer::PagedKVBatchLayerView &cache) {
  Planes planes;
  for (int i = 0; i < kPlaneCount; ++i) { planes.bytes[i] = to_host(plane(cache, i).data, plane(cache, i).bytes()); }
  return planes;
}

void restore(const ninfer::PagedKVBatchLayerView &cache, const Planes &planes) {
  for (int i = 0; i < kPlaneCount; ++i) {
    CUDA_FATAL(cudaMemcpy(plane(cache, i).data, planes.bytes[i].data(), planes.bytes[i].size(),
                          cudaMemcpyHostToDevice));
  }
}

// ---- the routes ------------------------------------------------------------------

enum class Route {
  // `ninfer::ops::gqa_attention`: ChunkedSmallT at two lanes or more.
  Vendored,
  // The batch body's chunk loop on the vendored chunks: the one-lane oracle,
  // held to `Vendored` wherever both run.
  VendoredChunks,
  // The same loop on ignis's route, the route under test.
  Ours,
};

const char *name_of(Route route) {
  switch (route) {
  case Route::Vendored: return "gqa_attention";
  case Route::VendoredChunks: return "the chunk loop on the vendored chunks";
  case Route::Ours: return "the chunk loop on ignis's route";
  }
  return "?";
}

// One call's shape: lanes, columns a lane, and each lane's valid prefix
// (empty: the dense form, no valid-column tensor).
struct Case {
  std::int32_t lanes = 0;
  std::int32_t width = 0;
  std::vector<std::int32_t> valid;

  std::int32_t valid_of(std::int32_t lane) const { return valid.empty() ? width : valid[static_cast<std::size_t>(lane)]; }

  std::string name() const {
    std::string s = std::to_string(lanes) + (lanes == 1 ? " lane, " : " lanes, ") + std::to_string(width) + " columns, ";
    if (valid.empty()) { return s + "dense"; }
    s += "valid";
    for (std::int32_t v : valid) { s += " " + std::to_string(v); }
    return s;
  }
};

struct Result {
  std::vector<std::uint8_t> out;
  Planes cache;
};

// Everything the cases share: the pool, its lanes with their histories, the
// cache view and the bytes every call starts from.
struct Bench {
  ignis_seq_pool *pool = nullptr;
  std::vector<ignis_seq *> lanes;
  ninfer::PagedKVBatchLayerView cache;
  ninfer::ops::GqaExecutionEnvelope envelope{1, kContext};
  Planes baseline;
  cudaStream_t stream = nullptr;
};

// The workspace the round reserves for the call (kernel/src/gqa_layer.cu): the
// vendored query at the call's lanes and width, and for one lane past the
// tile, whose columns run in chunks, one chunk's.
std::size_t workspace_bytes(const Bench &bench, const Case &c) {
  if (c.lanes == 1) {
    return ninfer::ops::gqa_attention_workspace_capacity_bytes(
        kQHeads, bench.cache.dtype, bench.envelope, 1, 1, ninfer::ops::detail::gqa_small_t_chunk_tokens(bench.cache.dtype));
  }
  return ninfer::ops::gqa_attention_workspace_capacity_bytes(kQHeads, bench.cache.dtype, bench.envelope, c.lanes,
                                                             c.width, c.width);
}

Result run(const Bench &bench, const Fixture &fx, const Case &c, Route route, int workspace_fill) {
  const std::string label = c.name() + ", " + name_of(route);
  const std::int32_t columns = c.width * c.lanes;

  // The call's inputs: each lane's new rows follow its history in the fixture,
  // its positions the valid prefix in sequence and the tail repeating the
  // last valid one (the vendored A1 contract's masked form).
  std::vector<std::uint16_t> k, v, q, gate;
  k.resize(static_cast<std::size_t>(kHeadDim) * kKvHeads * columns);
  v.resize(k.size());
  for (std::int32_t b = 0; b < c.lanes; ++b) {
    std::vector<std::uint16_t> lane_k, lane_v;
    fill_kv(fx, c.width, kHistory[b] + b * 97, lane_k, lane_v);
    const std::size_t at = static_cast<std::size_t>(b) * c.width * kKvHeads * kHeadDim;
    std::copy(lane_k.begin(), lane_k.end(), k.begin() + static_cast<std::ptrdiff_t>(at));
    std::copy(lane_v.begin(), lane_v.end(), v.begin() + static_cast<std::ptrdiff_t>(at));
  }
  fill_q(fx, c.width, c.lanes, q);
  fill_gate(q.size(), gate);
  std::vector<std::int32_t> positions(static_cast<std::size_t>(columns));
  std::vector<std::int32_t> table_rows(static_cast<std::size_t>(c.lanes));
  for (std::int32_t b = 0; b < c.lanes; ++b) {
    const std::int32_t valid = c.valid_of(b);
    for (std::int32_t w = 0; w < c.width; ++w) {
      positions[static_cast<std::size_t>(b) * c.width + w] = kHistory[b] + std::min(w, valid - 1);
    }
    table_rows[static_cast<std::size_t>(b)] = bench.lanes[static_cast<std::size_t>(b)]->kv.bound_row();
  }

  const std::size_t out_bytes = q.size() * 2;
  DeviceBytes k_device(k.size() * 2), v_device(v.size() * 2), q_device(q.size() * 2), gate_device(gate.size() * 2);
  DeviceBytes out_device(out_bytes), pos_device(positions.size() * 4), rows_device(table_rows.size() * 4);
  DeviceBytes valid_device(c.valid.size() * 4);
  k_device.upload(k);
  v_device.upload(v);
  q_device.upload(q);
  gate_device.upload(gate);
  pos_device.upload(positions);
  rows_device.upload(table_rows);
  if (!c.valid.empty()) { valid_device.upload(c.valid); }
  // A sentinel no route writes: BF16 0xABAB is neither zero nor an attention
  // output's likely bits, so an untouched column shows.
  CUDA_FATAL(cudaMemset(out_device.p, 0xAB, out_bytes));

  const ninfer::Tensor q_t(q_device.p, ninfer::DType::BF16, {kHeadDim, kQHeads, c.width, c.lanes});
  const ninfer::Tensor k_t(k_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, c.width, c.lanes});
  const ninfer::Tensor v_t(v_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, c.width, c.lanes});
  const ninfer::Tensor gate_t(gate_device.p, ninfer::DType::BF16, {kHeadDim, kQHeads, c.width, c.lanes});
  const ninfer::Tensor pos_t(pos_device.p, ninfer::DType::I32, {c.width, c.lanes, 1, 1});
  const ninfer::Tensor rows_t(rows_device.p, ninfer::DType::I32, {c.lanes, 1, 1, 1});
  const ninfer::Tensor valid_t =
      c.valid.empty() ? ninfer::Tensor{} : ninfer::Tensor(valid_device.p, ninfer::DType::I32, {c.lanes, 1, 1, 1});
  ninfer::Tensor out_t(out_device.p, ninfer::DType::BF16, {kHeadDim, kQHeads, c.width, c.lanes});

  restore(bench.cache, bench.baseline);
  ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes(bench, c), 256));
  CUDA_FATAL(cudaMemset(workspace.base(), workspace_fill, workspace.capacity()));

  Result result;
  try {
    switch (route) {
    case Route::Vendored:
      ninfer::ops::gqa_attention(q_t, k_t, v_t, pos_t, valid_t, rows_t, gate_t, kScale, bench.cache, bench.envelope,
                                 workspace, out_t, bench.stream);
      break;
    case Route::VendoredChunks:
    case Route::Ours:
      ignis::extended_attention::attend(route == Route::Ours ? ignis::extended_attention::Route::kOurs
                                                             : ignis::extended_attention::Route::kVendored,
                                        q_t, k_t, v_t, pos_t, valid_t, rows_t, gate_t, kScale, bench.cache,
                                        bench.envelope, workspace, out_t, bench.stream);
      break;
    }
    CUDA_FATAL(cudaStreamSynchronize(bench.stream));
    CUDA_FATAL(cudaGetLastError());
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FATAL: %s threw: %s\n", label.c_str(), e.what());
    std::exit(EXIT_FAILURE);
  }
  result.out   = to_host(out_device.p, out_bytes);
  result.cache = snapshot(bench.cache);
  return result;
}

// Whether every byte pair of `out`'s column `w` of lane `b` is `value`.
bool column_is(const std::vector<std::uint8_t> &out, const Case &c, std::int32_t b, std::int32_t w,
               std::uint8_t value) {
  const std::size_t column = static_cast<std::size_t>(kHeadDim) * kQHeads * 2;
  const auto begin = out.begin() + static_cast<std::ptrdiff_t>((static_cast<std::size_t>(b) * c.width + w) * column);
  return std::all_of(begin, begin + static_cast<std::ptrdiff_t>(column), [value](std::uint8_t x) { return x == value; });
}

// The oracle's own result is a real call's: nothing vacuous is compared.
void check_oracle(const Bench &bench, const Case &c, const Result &oracle) {
  const std::string name = c.name();
  for (std::int32_t b = 0; b < c.lanes; ++b) {
    for (std::int32_t w = 0; w < c.width; ++w) {
      const std::string at = name + ": lane " + std::to_string(b) + " column " + std::to_string(w);
      if (w < c.valid_of(b)) {
        check(!column_is(oracle.out, c, b, w, 0xAB), at + " kept its sentinel: the oracle wrote no output there");
        check(!column_is(oracle.out, c, b, w, 0x00), at + " is all zero where the lane's prefix covers it");
      } else {
        check(column_is(oracle.out, c, b, w, 0x00),
              at + " is past the lane's valid prefix and is not exact zero (the vendored A1 contract)");
      }
    }
  }
  const std::size_t appended = differing(bench.baseline.bytes[0], oracle.cache.bytes[0]) +
                               differing(bench.baseline.bytes[1], oracle.cache.bytes[1]);
  check(appended > 0, name + ": the fused append changed no code byte");
  // Lane 0 is in every case, its ring slots under the call cleared first.
  check(differing(bench.baseline.bytes[kRingPlane], oracle.cache.bytes[kRingPlane]) > 0,
        name + ": the fused append set no ring mark");
  if (c.lanes > 1) {
    const std::size_t column = static_cast<std::size_t>(kHeadDim) * kQHeads * 2;
    check(!std::equal(oracle.out.begin(), oracle.out.begin() + static_cast<std::ptrdiff_t>(column),
                      oracle.out.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(c.width) * column)),
          name + ": lanes 0 and 1 produced the same first column: the lane axis never reached the kernel");
  }
}

// Every byte of `got` against `want`: the output, then each cache plane. The
// count of differing bytes, each differing piece named.
std::size_t hold(const Case &c, const Result &want, Route wanted, const Result &got, Route route) {
  const std::string who = c.name() + ": " + name_of(route) + " against " + name_of(wanted);
  std::size_t total = differing(want.out, got.out);
  check(total == 0, who + ": " + std::to_string(total) + " of " + std::to_string(want.out.size()) +
                        " output bytes differ");
  for (int i = 0; i < kPlaneCount; ++i) {
    const std::size_t n = differing(want.cache.bytes[i], got.cache.bytes[i]);
    check(n == 0, who + ": " + std::to_string(n) + " bytes of the cache's " + kPlaneNames[i] + " differ after the call");
    total += n;
  }
  return total;
}

// The six valid prefixes the masked cases rotate over the lanes, at `width`:
// full, the chunk boundary, one column past it, inside chunk 1, the anchor
// alone, and inside chunk 2 short of the width. All six differ at every width,
// so up to four lanes taking consecutive ones never mask alike: where chunk 2
// has no room for a prefix of its own (one column at width 9, two at 10), the
// entry falls back to another prefix inside chunk 1.
std::vector<std::int32_t> prefixes(std::int32_t width) {
  const std::int32_t past_boundary = width > 9 ? 9 : 5;
  const std::int32_t inside_two    = width >= 12 ? width - 2 : (width == 11 ? 10 : 7);
  return {width, 8, past_boundary, 3, 1, inside_two};
}

} // namespace

int main() {
  cudaDeviceProp prop{};
  CUDA_FATAL(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  if (!ignis::hq_decode_attention::runs_on(prop.major) || !ignis::hq_decode_attention::image_here()) {
    std::fprintf(stderr,
                 "FATAL: this binary holds no image of ignis's hq attention route for this card (compute capability "
                 "%d.%d): the test is sm120a-only, run `ctest -LE sm120a-only` here\n",
                 prop.major, prop.minor);
    return EXIT_FAILURE;
  }
  const Fixture fx = load_fixture();

  // The pool: four lanes of `kContext` tokens under hq-e8-2b, the GDN and
  // vocabulary geometry small (no layer is stepped here).
  constexpr auto page = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = kKvHeads;
  spec.head_dim            = kHeadDim;
  spec.kv_format           = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = (kContext + page - 1) / page * kMaxLanes;
  spec.max_context_tokens  = kContext;
  spec.slot_count          = kMaxLanes;
  spec.gdn_num_layers      = 2;
  spec.kv_num_layers       = kIgnisGqaLayerCount;
  spec.gdn_conv_channels   = 6;
  spec.gdn_value_heads     = 2;
  spec.gdn_head_dim        = 4;
  spec.vocab               = 32;
  Bench bench;
  if (ignis_seq_pool_create(&spec, &bench.pool) != 0 || bench.pool == nullptr) {
    std::fprintf(stderr, "FATAL: the sequence pool was refused\n");
    return EXIT_FAILURE;
  }
  CUDA_FATAL(cudaStreamCreate(&bench.stream));
  for (std::int32_t b = 0; b < kMaxLanes; ++b) {
    ignis_seq *seq = nullptr;
    if (ignis_seq_alloc(bench.pool, kContext, &seq) != 0 || seq == nullptr) {
      std::fprintf(stderr, "FATAL: lane %d's sequence was refused\n", b);
      return EXIT_FAILURE;
    }
    bench.lanes.push_back(seq);
    // The lane's history, through the vendored A2 on the production view, as
    // a prefill writes it: every code row, and its last 512 keys in the ring.
    std::vector<std::uint16_t> hk, hv;
    fill_kv(fx, kHistory[b], b * 97, hk, hv);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(kHistory[b]));
    for (std::int32_t t = 0; t < kHistory[b]; ++t) { positions[static_cast<std::size_t>(t)] = t; }
    DeviceBytes hk_device(hk.size() * 2), hv_device(hv.size() * 2), pos_device(positions.size() * 4);
    hk_device.upload(hk);
    hv_device.upload(hv);
    pos_device.upload(positions);
    const ninfer::Tensor hk_t(hk_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kHistory[b], 1});
    const ninfer::Tensor hv_t(hv_device.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, kHistory[b], 1});
    const ninfer::Tensor pos_t(pos_device.p, ninfer::DType::I32, {kHistory[b], 1, 1, 1});
    try {
      ninfer::ops::gqa_kv_append(hk_t, hv_t, pos_t, ignis_kv_layer_view(bench.pool, seq, kGqaOrdinal), bench.stream);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FATAL: lane %d's history append threw: %s\n", b, e.what());
      return EXIT_FAILURE;
    }
    CUDA_FATAL(cudaStreamSynchronize(bench.stream));
  }
  bench.cache    = ignis_kv_batch_layer_view(bench.pool, kGqaOrdinal);
  bench.baseline = snapshot(bench.cache);
  check(bench.cache.dtype == ninfer::DType::U8 && bench.cache.residual_k.data != nullptr &&
            bench.cache.ring_valid.data != nullptr,
        "the pool's view is hq-e8-2b with its residual window");
  // The even lanes' ring slots under the widest call's columns, cleared as a
  // rollback leaves them (kernel/include/ignis_hq_ring.h); the odd lanes keep
  // the marks their history left.
  bool cleared_a_mark = false;
  for (std::int32_t b = 0; b < kMaxLanes; b += 2) {
    const ninfer::KvRingWords keep = ignis_hq_ring_invalidate_mask(
        static_cast<std::uint64_t>(kHistory[b]), static_cast<std::uint64_t>(kHistory[b] + kMaxWidth));
    const std::size_t row = static_cast<std::size_t>(bench.lanes[static_cast<std::size_t>(b)]->slot) * kIgnisHqRingWords * 4;
    for (std::int32_t i = 0; i < kIgnisHqRingWords; ++i) {
      std::uint32_t word;
      std::memcpy(&word, bench.baseline.bytes[kRingPlane].data() + row + static_cast<std::size_t>(i) * 4, 4);
      cleared_a_mark = cleared_a_mark || (word & ~keep.w[i]) != 0;
      word &= keep.w[i];
      std::memcpy(bench.baseline.bytes[kRingPlane].data() + row + static_cast<std::size_t>(i) * 4, &word, 4);
    }
  }
  check(cleared_a_mark, "lane 0's history is past the ring, so clearing its slots under the call cleared marks");

  // Each chunk's split capacity at the envelope: several, or the reducer's
  // merge over splits is never exercised.
  const std::int32_t tile_splits = ninfer::ops::detail::gqa_attention_split_capacity(
      kQHeads, ninfer::ops::detail::gqa_small_t_chunk_tokens(bench.cache.dtype), bench.cache.dtype, bench.envelope);
  std::printf("envelope %u keys: %d splits a full chunk; histories %d, %d, %d, %d keys\n", kContext, tile_splits,
              kHistory[0], kHistory[1], kHistory[2], kHistory[3]);
  check(tile_splits > 1, "a full chunk runs on more than one split at this envelope");

  int cases = 0;
  for (std::int32_t lanes = 1; lanes <= kMaxLanes; ++lanes) {
    for (std::int32_t width = kMinWidth; width <= kMaxWidth; ++width) {
      check(ignis::hq_decode_attention::serves_chunked(kQHeads, width, bench.cache.dtype),
            "serves_chunked admits " + std::to_string(width) + " columns a lane over hq-e8-2b");
      const std::vector<std::int32_t> rotate = prefixes(width);
      std::vector<Case> shapes;
      shapes.push_back(Case{lanes, width, {}});
      for (std::size_t turn = 0; turn < rotate.size(); ++turn) {
        Case masked{lanes, width, {}};
        for (std::int32_t b = 0; b < lanes; ++b) {
          masked.valid.push_back(rotate[(static_cast<std::size_t>(b) + turn) % rotate.size()]);
        }
        shapes.push_back(masked);
      }
      std::size_t differing_bytes = 0;
      for (const Case &c : shapes) {
        // The oracle: `gqa_attention` where its resolver gives ChunkedSmallT,
        // the restated loop at one lane, and the two held to one another
        // wherever both run.
        const bool wrapper_chunks =
            ninfer::ops::detail::gqa_attention_resolve_route(kQHeads, c.width, c.lanes, bench.cache.dtype,
                                                             bench.envelope) ==
            ninfer::ops::detail::GqaAttentionRoute::ChunkedSmallT;
        check(wrapper_chunks == (c.lanes > 1),
              c.name() + ": the vendored resolver gives ChunkedSmallT at two lanes or more and not at one");
        const Result loop = run(bench, fx, c, Route::VendoredChunks, 0x7F);
        Route oracle_route = Route::VendoredChunks;
        const Result *oracle = &loop;
        Result wrapper;
        if (wrapper_chunks) {
          wrapper = run(bench, fx, c, Route::Vendored, 0x7F);
          differing_bytes += hold(c, wrapper, Route::Vendored, loop, Route::VendoredChunks);
          oracle_route = Route::Vendored;
          oracle       = &wrapper;
        }
        check_oracle(bench, c, *oracle);
        const Result ours = run(bench, fx, c, Route::Ours, 0x7F);
        differing_bytes += hold(c, *oracle, oracle_route, ours, Route::Ours);
        if (c.valid.empty()) {
          const Result again = run(bench, fx, c, Route::Ours, 0x00);
          check(differing(ours.out, again.out) == 0,
                c.name() + ": ignis's route's output follows the workspace's bytes before the call");
        }
        ++cases;
      }
      std::printf("%d lane%s, %2d columns: %zu shapes (dense and %zu masks), bytes differing from the vendored "
                  "chunked route: %zu\n",
                  lanes, lanes == 1 ? " " : "s", width, shapes.size(), shapes.size() - 1, differing_bytes);
    }
  }

  for (ignis_seq *seq : bench.lanes) { ignis_seq_release(bench.pool, seq); }
  CUDA_FATAL(cudaStreamDestroy(bench.stream));
  ignis_seq_pool_free(bench.pool);

  if (g_failed != 0) {
    std::fprintf(stderr, "test_hq_verify_chunked_exact: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("test_hq_verify_chunked_exact: ALL PASSED (%d shapes: widths %d..%d, lanes 1..%d, dense and masked; "
              "ignis's route in chunks held to the vendored chunked small-T route, output and cache)\n",
              cases, kMinWidth, kMaxWidth, kMaxLanes);
  return 0;
}
