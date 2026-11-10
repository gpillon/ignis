// GitHub #558 (spec #555, ADR 0087): Flash-Next's rotations over prompts with images, against the
// reference implementation -- OURS. The fixture (fixtures/flash_next_mrope/record_mrope.py) ran the
// checkpoint's own transformers code at Flash-Next's geometry on synthetic inputs: get_rope_index's
// three-axis positions (Qwen3.5's multimodal rope index) and compute_3d_position_ids' decode
// positions, then one QSA layer (Qwen4ExpTextAttention with its Qwen4ExpTextQSAIndexer, the
// rotary table and the cache the text model builds) over each prompt in one call and each decode
// step after it. It recorded what the module's rotations returned: the attention's q (heads 0 and
// 13) and k (both heads), the indexer's queries, and every block key -- pooled, normed and rotated
// at its FIRST token's three axes. The test regenerates the inputs from the same counter-hash
// streams (their checksums are the fixture's) and runs the stages the program runs:
// qsa::prepare, indexer::append_keys and indexer::prepare_queries, one call per prefill chunk the
// fixture names, then one per decode step, so that a block straddles a call boundary.
//
// Cases: a text-only prompt; one image; two images (block [12, 16) starts on the second image's
// (1, 0) token and straddles a call); an image across block boundaries (blocks [8, 12) and
// [12, 16) start on image tokens and each straddles a call); decode after an image (the prompt ends
// on the image, and the first decode step completes block [8, 12), begun on its (1, 0) token).
//
// Checked:
// - vision: each call at the batch's source -- a chunk's three axes (rope_positions), a decode
//   step's position plus the prompt's rope delta (rope_deltas) -- on an indexer section with axis
//   records: every recorded value within the bound ElementCheck derives (the indexer test's: one
//   BF16 ulp, plus six of the rotary pair's magnitude on a rope dim -- a rope angle off by one
//   position on the first pairs fails it). The section's key slots start as garbage, as a reused
//   page's do;
// - a block one call begins and leaves incomplete holds its first token's AxisRecord, those axes
//   exactly, until the call that completes it replaces it with the key;
// - the checks bite: without axis records the straddling blocks rotate at their KV position and
//   fail the bound; a query rotated at its KV position fails it too; a record whose magic is gone
//   (a section written before records) rotates its block at the KV position;
// - the text path: the text-only prompt with no rotation source and no records gives the vision
//   arm's bytes bit for bit (three equal axes are one position), and both are the reference's.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next/indexer.h"
#include "flash_next/qsa.h"

#include "flash_next_s2_test_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace s2_test;
namespace fn = ignis::flash_next;
namespace ix = ignis::flash_next::indexer;
namespace qsa = ignis::flash_next::qsa;

namespace {

constexpr float kEps = 1e-6F;
constexpr int kRotary = 64;
constexpr int kIxHd = ix::kHeadDim;
constexpr int kIxHeads = 4;
constexpr int kQk = (kIxHeads + 1) * kIxHd;
constexpr int kHd = qsa::kHeadDim;
constexpr int kCases = 5;
constexpr const char *kCaseNames[kCases] = {"text", "one_image", "two_images", "image_across_blocks",
                                            "decode_after_image"};
// The record_mrope.py streams: per case c the q_proj, k_proj and index_qk_proj outputs.
constexpr uint32_t kStreamBase = 0x5580;

#define IX_OK(expr)                                                                                \
  do {                                                                                             \
    const char *st_ = (expr);                                                                      \
    if (st_ != nullptr) {                                                                          \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, st_);                                         \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

fn::Geometry geometry() {
  fn::Geometry g;
  g.hidden = 2560;
  g.q_heads = qsa::kQHeads;
  g.kv_heads = qsa::kKvHeads;
  g.head_dim = qsa::kHeadDim;
  g.rotary_dim = kRotary;
  g.indexer_heads = kIxHeads;
  g.indexer_head_dim = kIxHd;
  g.indexer_kv_heads = 1;
  g.compress_ratio = 4;
  g.indexer_budget = 2048;
  g.rms_norm_eps = kEps;
  return g;
}

// record_mrope.py's grid(): element i = r * width + j is 0.5 (1 + h % 4), negative when bit 2 of h
// is set -- squares and 4-token means that sum exactly in fp32 in any order.
std::vector<uint16_t> grid(uint32_t stream, std::size_t n) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const uint32_t h = hash_u32(stream, i);
    const float x = 0.5F * static_cast<float>(1U + (h & 3U));
    v[i] = f32_to_bf16(((h >> 2) & 1U) != 0 ? -x : x);
  }
  return v;
}

// record_mrope.py's checksum(): sum of bits(v[i]) * (i + 1).
int64_t checksum(const std::vector<uint16_t> &v) {
  uint64_t s = 0;
  for (std::size_t i = 0; i < v.size(); ++i) s += static_cast<uint64_t>(v[i]) * (i + 1);
  return static_cast<int64_t>(s);
}

double ulp_bf16(double v) {
  const double a = std::fabs(v);
  return a < 0x1p-126 ? 0x1p-133 : std::ldexp(1.0, std::ilogb(a) - 7);
}

// test_flash_next_indexer.cu's element bound, for rows `width` wide whose first 64 dims rotate as
// pairs (i, i + 32): one BF16 ulp of the reference's value, plus six of the pair's magnitude on a
// rope dim (a one-ulp norm difference carried through the rotation's three roundings).
struct ElementCheck {
  long exact = 0, total = 0, violations = 0;
  void add(const uint16_t *got, const uint16_t *want, std::size_t n, int width) {
    for (std::size_t r = 0; r < n; r += static_cast<std::size_t>(width)) {
      for (int i = 0; i < width; ++i) {
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
  void report(const std::string &what) const {
    std::printf("  %s: %ld values, %ld bit-exact, %ld outside the bound\n", what.c_str(), total, exact, violations);
    check(total > 0 && violations == 0, what + ": every value within BF16 rounding of the reference's");
  }
};

struct Case {
  std::string name;
  int32_t length = 0, decode = 0, delta = 0;
  std::vector<int32_t> calls;      // [n][2]: the prefill chunks
  std::vector<int32_t> positions;  // [3][length + decode]
  std::vector<int64_t> sums;
  std::vector<uint16_t> attn_q, attn_k, ix_q, block_keys;
  int32_t total() const { return length + decode; }
};

Case load_case(const std::map<std::string, moe_test::FixtureTensor> &fx, int c) {
  const std::string at = "c" + std::to_string(c) + ".";
  Case k;
  k.name = kCaseNames[c];
  k.length = need(fx, at + "length").as<int32_t>()[0];
  k.decode = need(fx, at + "decode").as<int32_t>()[0];
  k.delta = need(fx, at + "delta").as<int32_t>()[0];
  k.calls = need(fx, at + "calls").as<int32_t>();
  k.positions = need(fx, at + "positions").as<int32_t>();
  k.sums = need(fx, at + "sums").as<int64_t>();
  k.attn_q = need(fx, at + "attn_q").as<uint16_t>();
  k.attn_k = need(fx, at + "attn_k").as<uint16_t>();
  k.ix_q = need(fx, at + "ix_q").as<uint16_t>();
  k.block_keys = need(fx, at + "block_keys").as<uint16_t>();
  check(static_cast<int32_t>(k.positions.size()) == 3 * k.total(), k.name + ": three axes per token");
  return k;
}

// One call of a case: rows [begin, end), a prefill chunk at its three axes or a decode step at its
// position plus the rope delta.
struct Call {
  int32_t begin = 0, end = 0;
  bool decode = false;
};

std::vector<Call> calls_of(const Case &k) {
  std::vector<Call> out;
  for (std::size_t i = 0; i + 1 < k.calls.size(); i += 2) out.push_back({k.calls[i], k.calls[i + 1], false});
  for (int32_t s = 0; s < k.decode; ++s) out.push_back({k.length + s, k.length + s + 1, true});
  return out;
}

// How a run rotates and keeps its block records.
enum class Arm {
  Vision,      // the batch's source: three axes per chunk, a rope delta per decode step; records on
  NoRecords,   // the same sources, records off: a straddling block rotates at its KV position
  Text,        // no source and no records: today's text path
  KvQueries,   // records on, but the chunks' queries and keys at their KV positions (no source)
};

struct Outputs {
  std::vector<uint16_t> q, k, ix_q, block_keys;
  // After each call: the key slot words of the block it left incomplete (or empty).
  std::vector<std::vector<int32_t>> open_records;
};

Outputs run_case(const fn::Geometry &g, const ix::Rope &rope, const Case &k, int case_index, Arm arm,
                 const std::vector<int> &q_heads, const DeviceBytes &d_q_norm, const DeviceBytes &d_k_norm,
                 const DeviceBytes &d_ix_q_norm, const DeviceBytes &d_ix_k_norm, cudaStream_t stream,
                 int32_t corrupt_record_at_block = -1) {
  const int32_t n_rows = k.total();
  const uint32_t s_qg = kStreamBase + 16U * static_cast<uint32_t>(case_index), s_k = s_qg + 1, s_qk = s_qg + 2;
  const std::vector<uint16_t> qg = grid(s_qg, static_cast<std::size_t>(n_rows) * qsa::kQProjWidth);
  const std::vector<uint16_t> kk = grid(s_k, static_cast<std::size_t>(n_rows) * qsa::kKvWidth);
  const std::vector<uint16_t> qk = grid(s_qk, static_cast<std::size_t>(n_rows) * kQk);
  check(checksum(qg) == k.sums[0] && checksum(kk) == k.sums[1] && checksum(qk) == k.sums[2],
        k.name + ": the regenerated inputs are the recorder's");
  auto d_qg = device_copy(qg);
  auto d_k = device_copy(kk);
  auto d_qk = device_copy(qk);
  DeviceBytes d_q(static_cast<std::size_t>(n_rows) * qsa::kOutWidth * 2);
  DeviceBytes d_ix_q(static_cast<std::size_t>(n_rows) * kIxHeads * kIxHd * 2);

  // One slot, one page (every case is under 64 tokens), the page's key slots garbage to start with.
  check(n_rows <= ix::kPageTokens, k.name + ": the case fits one page");
  DeviceBytes d_tables(4), d_slots(4), d_positions(4), d_axes(static_cast<std::size_t>(3) * n_rows * 4),
      d_deltas(4);
  upload(d_tables, std::vector<int32_t>{0});
  upload(d_slots, std::vector<int32_t>{0});
  upload(d_deltas, std::vector<int32_t>{k.delta});
  const std::vector<uint16_t> garbage = bf16_vector(0x558F, static_cast<std::size_t>(ix::kBlocksPerPage) * kIxHd, -3, 3);
  auto d_keys = device_copy(garbage);
  DeviceBytes d_tail(static_cast<std::size_t>(ix::kCompress - 1) * kIxHd * 2);
  MOE_CUDA(cudaMemset(d_tail.p, 0, d_tail.bytes));
  ix::Paged paged;
  paged.block_tables = d_tables.as<int32_t>();
  paged.logical_pages = 1;
  paged.block_keys = d_keys->as<__nv_bfloat16>();
  paged.tail_keys = d_tail.as<__nv_bfloat16>();
  paged.axis_records = arm == Arm::Vision || arm == Arm::KvQueries;

  Outputs out;
  for (const Call &call : calls_of(k)) {
    const int32_t n = call.end - call.begin;
    upload(d_positions, std::vector<int32_t>{call.begin});
    fn::Batch batch;
    batch.lanes = 1;
    batch.tokens = n;
    batch.slots = d_slots.as<int32_t>();
    batch.positions = d_positions.as<int32_t>();
    batch.max_visible = call.end;
    if (arm == Arm::Vision || arm == Arm::NoRecords) {
      if (call.decode) {
        batch.rope_deltas = d_deltas.as<int32_t>();
      } else {
        std::vector<int32_t> axes(static_cast<std::size_t>(3) * n);
        for (int a = 0; a < 3; ++a) {
          for (int32_t t = 0; t < n; ++t) axes[a * n + t] = k.positions[a * n_rows + call.begin + t];
        }
        upload(d_axes, axes);
        batch.rope_positions = d_axes.as<int32_t>();
      }
    } else if (arm == Arm::KvQueries && call.decode) {
      batch.rope_deltas = d_deltas.as<int32_t>();
    }
    const auto *qg_rows = d_qg->as<__nv_bfloat16>() + static_cast<std::size_t>(call.begin) * qsa::kQProjWidth;
    auto *k_rows = d_k->as<__nv_bfloat16>() + static_cast<std::size_t>(call.begin) * qsa::kKvWidth;
    auto *q_rows = d_q.as<__nv_bfloat16>() + static_cast<std::size_t>(call.begin) * qsa::kOutWidth;
    IX_OK(qsa::prepare(g, rope, d_q_norm.p, d_k_norm.p, batch, qg_rows, q_rows, k_rows, stream));
    const void *qk_rows = d_qk->as<__nv_bfloat16>() + static_cast<std::size_t>(call.begin) * kQk;
    IX_OK(ix::append_keys(g, paged, rope, d_ix_k_norm.p, batch, qk_rows, stream));
    IX_OK(ix::prepare_queries(g, rope, d_ix_q_norm.p, batch, 0, n, qk_rows,
                              d_ix_q.as<__nv_bfloat16>() + static_cast<std::size_t>(call.begin) * kIxHeads * kIxHd,
                              stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    const int32_t open = call.end % ix::kCompress != 0 ? call.end / ix::kCompress : -1;
    if (open >= 0) {
      out.open_records.push_back(download<int32_t>(
          static_cast<const unsigned char *>(d_keys->p) + static_cast<std::size_t>(open) * kIxHd * 2, 5));
      if (open == corrupt_record_at_block) {
        // A section written before records: whatever the slot held, no record of this block.
        MOE_CUDA(cudaMemset(static_cast<unsigned char *>(d_keys->p) + static_cast<std::size_t>(open) * kIxHd * 2, 0,
                            sizeof(int32_t)));
      }
    } else {
      out.open_records.emplace_back();
    }
  }
  const auto q_all = download<uint16_t>(d_q.p, static_cast<std::size_t>(n_rows) * qsa::kOutWidth);
  for (int32_t r = 0; r < n_rows; ++r) {
    for (int h : q_heads) {
      const uint16_t *at = &q_all[(static_cast<std::size_t>(r) * qsa::kQHeads + h) * kHd];
      out.q.insert(out.q.end(), at, at + kHd);
    }
  }
  out.k = download<uint16_t>(d_k->p, static_cast<std::size_t>(n_rows) * qsa::kKvWidth);
  out.ix_q = download<uint16_t>(d_ix_q.p, static_cast<std::size_t>(n_rows) * kIxHeads * kIxHd);
  out.block_keys = download<uint16_t>(d_keys->p, static_cast<std::size_t>(n_rows / ix::kCompress) * kIxHd);
  return out;
}

// The rows of the blocks whose first token is in an earlier call than their last.
std::vector<int32_t> straddling_blocks(const Case &k) {
  std::vector<int32_t> out;
  for (const Call &call : calls_of(k)) {
    if (call.begin % ix::kCompress != 0 && call.end >= (call.begin / ix::kCompress + 1) * ix::kCompress) {
      out.push_back(call.begin / ix::kCompress);
    }
  }
  return out;
}

bool three_axes_differ(const Case &k, int32_t token) {
  const int32_t n = k.total();
  return !(k.positions[token] == k.positions[n + token] && k.positions[token] == k.positions[2 * n + token] &&
           k.positions[token] == token);
}

}  // namespace

int main() {
  MOE_CUDA(cudaSetDevice(0));
  const auto fx = read_fixture(IGNIS_FLASH_NEXT_MROPE_FIXTURE);
  const fn::Geometry g = geometry();
  ignis_topology topology{};
  topology.rotary_dim = kRotary;
  topology.rope_theta = 1e7;
  const ix::Rope rope = ix::rope_from(fn::text_rope(topology, {}));
  const std::vector<float> inv_freq = need(fx, "inv_freq").as<float>();
  check(inv_freq.size() == kRotary / 2 && std::memcmp(inv_freq.data(), rope.inv_freq, sizeof(rope.inv_freq)) == 0 &&
            rope.scale == 1.0F,
        "the load's table is the reference's, bit for bit, with no attention factor");
  check(need(fx, "cases").as<int32_t>()[0] == kCases, "the fixture holds the five cases");
  const std::vector<int32_t> q_heads32 = need(fx, "q_heads").as<int32_t>();
  const std::vector<int> q_heads(q_heads32.begin(), q_heads32.end());
  auto d_q_norm = device_copy(need(fx, "norm.q").as<uint16_t>());
  auto d_k_norm = device_copy(need(fx, "norm.k").as<uint16_t>());
  auto d_ix_q_norm = device_copy(need(fx, "norm.ix_q").as<uint16_t>());
  auto d_ix_k_norm = device_copy(need(fx, "norm.ix_k").as<uint16_t>());
  cudaStream_t stream;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto run = [&](const Case &k, int c, Arm arm, int32_t corrupt = -1) {
    return run_case(g, rope, k, c, arm, q_heads, *d_q_norm, *d_k_norm, *d_ix_q_norm, *d_ix_k_norm, stream, corrupt);
  };

  long straddling_seen = 0, straddling_moved = 0;
  for (int c = 0; c < kCases; ++c) {
    const Case k = load_case(fx, c);
    std::printf("%s: %d tokens + %d decode steps, rope delta %d\n", k.name.c_str(), k.length, k.decode, k.delta);
    const Outputs got = run(k, c, Arm::Vision);
    ElementCheck q, kc, ixq, keys;
    q.add(got.q.data(), k.attn_q.data(), got.q.size(), kHd);
    kc.add(got.k.data(), k.attn_k.data(), got.k.size(), kHd);
    ixq.add(got.ix_q.data(), k.ix_q.data(), got.ix_q.size(), kIxHd);
    keys.add(got.block_keys.data(), k.block_keys.data(), got.block_keys.size(), kIxHd);
    q.report(k.name + ": the attention's queries");
    kc.report(k.name + ": the attention's keys");
    ixq.report(k.name + ": the indexer's queries");
    keys.report(k.name + ": the indexer's block keys");

    // Each block a call leaves incomplete holds its first token's three axes.
    const std::vector<Call> calls = calls_of(k);
    bool records = true;
    for (std::size_t i = 0; i < calls.size(); ++i) {
      if (calls[i].end % ix::kCompress == 0) continue;
      const int32_t first = calls[i].end / ix::kCompress * ix::kCompress;
      const std::vector<int32_t> &w = got.open_records[i];
      const int32_t n = k.total();
      // Begun by this call, or by an earlier one whose record still stands.
      records = records && w.size() == 5 && w[0] == ix::kAxisRecordMagic && w[1] == first &&
                w[2] == k.positions[first] && w[3] == k.positions[n + first] && w[4] == k.positions[2 * n + first];
    }
    check(records, k.name + ": a block left incomplete holds its first token's AxisRecord, those three axes");

    // The checks bite where a block straddles a call and starts on a token whose rotation is not its
    // KV position (an image token, or text past an image): without records it rotates at the KV
    // position.
    const std::vector<int32_t> straddling = straddling_blocks(k);
    std::vector<int32_t> moved;
    for (int32_t b : straddling) {
      ++straddling_seen;
      if (three_axes_differ(k, b * ix::kCompress)) moved.push_back(b);
    }
    straddling_moved += static_cast<long>(moved.size());
    if (!moved.empty()) {
      const Outputs bare = run(k, c, Arm::NoRecords);
      ElementCheck at_kv;
      for (int32_t b : moved) {
        at_kv.add(&bare.block_keys[static_cast<std::size_t>(b) * kIxHd], &k.block_keys[static_cast<std::size_t>(b) * kIxHd],
                  kIxHd, kIxHd);
      }
      std::printf("  without records, %zu straddling block(s) begun off their KV position: %ld of %ld values outside "
                  "the bound\n",
                  moved.size(), at_kv.violations, at_kv.total);
      check(at_kv.violations > 0, k.name + ": a straddling block without its record fails the bound");
      // A record that is not this block's: the key rotates at the KV position, as without records.
      const Outputs lost = run(k, c, Arm::Vision, moved[0]);
      check(std::equal(lost.block_keys.begin() + moved[0] * kIxHd, lost.block_keys.begin() + (moved[0] + 1) * kIxHd,
                       bare.block_keys.begin() + moved[0] * kIxHd),
            k.name + ": a slot without this block's record rotates it at its KV position");
    }
    if (k.name != "text") {
      // Queries rotated at their KV positions (no source) leave the reference.
      const Outputs kv = run(k, c, Arm::KvQueries);
      ElementCheck at_kv;
      at_kv.add(kv.ix_q.data(), k.ix_q.data(), kv.ix_q.size(), kIxHd);
      check(at_kv.violations > 0, k.name + ": queries rotated at their KV positions fail the bound");
    } else {
      // The text path: no source and no records is the vision arm, bit for bit.
      const Outputs text = run(k, c, Arm::Text);
      check(text.q == got.q && text.k == got.k && text.ix_q == got.ix_q && text.block_keys == got.block_keys,
            "text: the text path (no rotation source, no records) gives the vision arm's bytes bit for bit");
    }
  }
  std::printf("%ld straddling blocks, %ld of them begun off their KV position\n", straddling_seen, straddling_moved);
  check(straddling_moved >= 4, "the cases straddle calls with blocks begun off their KV position");

  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next mrope: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("flash-next mrope: all checks passed\n");
  return EXIT_SUCCESS;
}
