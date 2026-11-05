// GitHub #339 (spec 27b-16gb/05a): the 27B program binds the 16 GB artifact's
// trellis body -- OURS, not vendored.
//
// Through `ignis_model_plan_reservations`, on a four-layer topology of the
// 27B's own widths (three GDN layers and the GQA layer 3) whose descriptors
// carry shapes and, where an arm says so, addresses that are never read:
//
// - the body is the head's: a trellis `text/output_head` binds the trellis
//   schema (spec 01's names), and a W8 one the served schema;
// - one body, never a mix: an NVFP4 tensor in the trellis schema, or a
//   trellis tensor where the schema binds a stored weight, fails naming it;
// - each trellis code is its K (a record too short for its K is refused), the
//   shapes are as strict as the served schema's, and a descriptor that carries
//   addresses has its `suh` and `svh` where the record puts them;
// - the trellis workspace is a plan line: the widest of the body's calls at
//   1..64 tokens and the chunk and of the head's at a round's columns or a
//   span-logits block, asked of the op on this device -- and 0 on the served
//   body, whose plan is otherwise what it was;
// - the two lines the trellis body changes besides its own, `workspace` and
//   `decode_graph`, are their closed forms (spec 05a's As built, "The plan's
//   trellis lines"): the program's outer buffers and the wider of the two
//   layers -- each its activations, the vendored queries the trellis body
//   keeps (the GQA attention's, the GDN gating's and recurrence's) and, on the
//   GQA layer, the q | gate buffer; no projection workspace -- at the 16 GiB
//   load's shape;
// - GitHub #509: under the dense prefill the plan holds its one buffer, the
//   body's largest matrix in INT8 (`trellis_dense_bytes`), and the trellis
//   workspace asks the dense op at the chunk in place of the fused prefill;
//   the served body and an unknown route are refused;
// - the review of spec #503: the prefill chunk's arena is sized by the routes
//   the load runs -- no GDN conv plane since #507, q_proj's interleaved buffer
//   only for the fused op's calls under the dense prefill, our GDN prefill's
//   workspace in place of the vendored op's (#514), named here by its
//   measurement switch so the closed forms hold on any card -- and a round's
//   arena by none of them.
//
// The op's workspace query needs a prepared device (the decode route splits
// for its SM count), so this runs on the GPU like every CTest here. ADR 0006 /
// docs/agents/testing.md: no SKIP_RETURN_CODE, so nothing here reads as a
// skip; a host without a device fails.

#include "ignis_dense_prefill.h"
#include "ignis_gqa_workspace.h"
#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_trellis_linear.h"

#include "gdn_prefill.h"

#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

// The 27B's widths (crates/core/src/compute.rs `ModelConfig::qwen38_27b`).
constexpr int32_t kHidden = 5120;
constexpr int32_t kVocab = 248320;
constexpr int32_t kQWidth = 24 * 256;
constexpr int32_t kKvWidth = 4 * 256;
constexpr int32_t kFfn = 17408;
constexpr int32_t kValueHeads = 48;
constexpr int32_t kKeyWidth = 16 * 128;
constexpr int32_t kStateRows = kValueHeads * 128;
constexpr int32_t kConvChannels = 2 * kKeyWidth + kStateRows;
constexpr uint32_t kChunk = 1024;
constexpr uint32_t kMaxContext = 4096;

struct Topology {
  std::vector<int32_t> kinds;
  ignis_topology raw{};
};

// `layers` layers, every fourth GQA (none for 0).
Topology topology_of(uint32_t layers) {
  Topology t;
  for (uint32_t i = 0; i < layers; ++i) {
    t.kinds.push_back((i + 1) % 4 == 0 ? IGNIS_LAYER_GQA : IGNIS_LAYER_GDN);
  }
  t.raw.num_layers = layers;
  t.raw.layer_kinds = t.kinds.empty() ? nullptr : t.kinds.data();
  t.raw.hidden = kHidden;
  t.raw.vocab = kVocab;
  t.raw.num_q_heads = 24;
  t.raw.num_kv_heads = 4;
  t.raw.head_dim = 256;
  t.raw.rotary_dim = 64;
  t.raw.rope_theta = 1e7;
  t.raw.gdn_state_rows = kStateRows;
  t.raw.gdn_state_cols = kKeyWidth;
  t.raw.gdn_num_layers = static_cast<uint64_t>(std::count(t.kinds.begin(), t.kinds.end(), IGNIS_LAYER_GDN));
  t.raw.gdn_value_heads = kValueHeads;
  t.raw.gdn_head_dim = 128;
  t.raw.gdn_conv_kernel = 4;
  t.raw.gdn_q_width = kKeyWidth;
  t.raw.gdn_z_width = kStateRows;
  t.raw.gdn_ab_width = 2 * kValueHeads;
  t.raw.ffn_intermediate = kFfn;
  t.raw.rms_norm_eps = 1e-6f;
  return t;
}

// 2K of a trellis code.
uint32_t k2_of(int32_t qtype) {
  switch (qtype) {
  case IGNIS_QTYPE_TRELLIS_MUL1_K2: return 4;
  case IGNIS_QTYPE_TRELLIS_MUL1_K2P5: return 5;
  case IGNIS_QTYPE_TRELLIS_MUL1_K3: return 6;
  case IGNIS_QTYPE_TRELLIS_MUL1_K4: return 8;
  case IGNIS_QTYPE_TRELLIS_MUL1_K5: return 10;
  default: return 12;
  }
}

uint64_t tile_bytes(int32_t rows, int32_t cols, int32_t qtype) {
  return static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) / 16 * k2_of(qtype);
}

// A record's bytes as the artifact stores it: tiles, suh, svh, padded to 4096.
uint64_t record_bytes(int32_t rows, int32_t cols, int32_t qtype) {
  const uint64_t data = tile_bytes(rows, cols, qtype) + 2ull * (static_cast<uint64_t>(rows) + cols);
  return (data + 4095) / 4096 * 4096;
}

struct Entry {
  std::string name;
  std::vector<int32_t> shape;
  int32_t qtype = IGNIS_QTYPE_BF16_CTRL;
  int32_t layout = IGNIS_LAYOUT_CONTIGUOUS;
  uint64_t bytes = 0;
  const void *qdata = nullptr;
  const void *qhigh = nullptr;
  const void *scales = nullptr;
};

Entry stored(const std::string &name, std::vector<int32_t> shape, int32_t qtype = IGNIS_QTYPE_BF16_CTRL,
             int32_t layout = IGNIS_LAYOUT_CONTIGUOUS) {
  Entry e{name, std::move(shape), qtype, layout};
  uint64_t n = 1;
  for (int32_t d : e.shape) n *= static_cast<uint64_t>(d);
  e.bytes = n * 2;
  return e;
}

// The public quant's K varies per matrix; the arms below use K3 and K4.
Entry trellis(const std::string &name, int32_t rows, int32_t cols, int32_t qtype = IGNIS_QTYPE_TRELLIS_MUL1_K3) {
  Entry e{name, {rows, cols}, qtype, IGNIS_LAYOUT_TRELLIS_TILE16};
  e.bytes = record_bytes(rows, cols, qtype);
  return e;
}

// The served body's endpoints over no layer: the W8 head.
std::vector<Entry> served_endpoints() {
  return {stored("text/token_embedding", {kVocab, kHidden}, IGNIS_QTYPE_W8G32_F16S, IGNIS_LAYOUT_ROW_SPLIT),
          stored("text/final_norm", {kHidden}),
          stored("text/output_head", {kVocab, kHidden}, IGNIS_QTYPE_W8G32_F16S, IGNIS_LAYOUT_ROW_SPLIT)};
}

// The trellis body: spec 01's names and shapes, the head at K4.
std::vector<Entry> trellis_body(uint32_t layers) {
  std::vector<Entry> out{
      stored("text/token_embedding", {kVocab, kHidden}, IGNIS_QTYPE_W8G32_F16S, IGNIS_LAYOUT_ROW_SPLIT),
      stored("text/final_norm", {kHidden}),
      trellis("text/output_head", kVocab, kHidden, IGNIS_QTYPE_TRELLIS_MUL1_K4)};
  for (uint32_t i = 0; i < layers; ++i) {
    const std::string p = "text/layers/" + std::to_string(i) + "/";
    out.push_back(stored(p + "input_norm", {kHidden}));
    if ((i + 1) % 4 == 0) {
      out.push_back(trellis(p + "attention/q_proj", 2 * kQWidth, kHidden));
      out.push_back(trellis(p + "attention/k_proj", kKvWidth, kHidden));
      out.push_back(trellis(p + "attention/v_proj", kKvWidth, kHidden));
      out.push_back(stored(p + "attention/query_norm", {256}));
      out.push_back(stored(p + "attention/key_norm", {256}));
      out.push_back(trellis(p + "attention/output", kHidden, kQWidth));
    } else {
      out.push_back(stored(p + "gdn/a_log", {kValueHeads}, IGNIS_QTYPE_FP32_CTRL));
      out.push_back(stored(p + "gdn/dt_bias", {kValueHeads}, IGNIS_QTYPE_FP32_CTRL));
      out.push_back(stored(p + "gdn/convolution", {4, kConvChannels}));
      out.push_back(stored(p + "gdn/a_b_projection", {2 * kValueHeads, kHidden}));
      out.push_back(trellis(p + "gdn/in_proj_qkv", kConvChannels, kHidden));
      out.push_back(trellis(p + "gdn/in_proj_z", kStateRows, kHidden));
      out.push_back(stored(p + "gdn/norm", {128}));
      out.push_back(trellis(p + "gdn/output", kHidden, kStateRows));
    }
    out.push_back(stored(p + "post_attention_norm", {kHidden}));
    out.push_back(trellis(p + "mlp/gate_proj", kFfn, kHidden));
    out.push_back(trellis(p + "mlp/up_proj", kFfn, kHidden));
    out.push_back(trellis(p + "mlp/down", kHidden, kFfn, IGNIS_QTYPE_TRELLIS_MUL1_K4));
  }
  return out;
}

Entry &find(std::vector<Entry> &entries, const std::string &name) {
  for (Entry &e : entries) {
    if (e.name == name) return e;
  }
  std::fprintf(stderr, "no entry %s\n", name.c_str());
  std::exit(1);
}

struct Plan {
  int32_t rc = -1;
  std::string error;
  ignis_model_reservations reserved{};
};

Plan plan(const std::vector<Entry> &entries, const Topology &topology, uint32_t lanes, int32_t backend = 0,
          uint32_t draft_tokens = 0, uint32_t max_context = kMaxContext,
          uint32_t prefill_linear = IGNIS_PREFILL_LINEAR_FUSED, uint32_t chunk = kChunk) {
  std::vector<ignis_bound_tensor> tensors;
  for (const Entry &e : entries) {
    ignis_bound_tensor t{};
    t.name = e.name.c_str();
    t.qtype = e.qtype;
    t.layout = e.layout;
    t.qdata = e.qdata;
    t.qhigh = e.qhigh;
    t.scales = e.scales;
    t.bytes = e.bytes;
    t.ndim = static_cast<uint32_t>(e.shape.size());
    for (size_t i = 0; i < 4; ++i) {
      t.shape[i] = i < e.shape.size() ? e.shape[i] : 1;
      t.padded_shape[i] = t.shape[i];
    }
    tensors.push_back(t);
  }
  ignis_model_load_options options{};
  options.size = sizeof(options);
  options.decode_lanes = lanes;
  options.speculative_backend = backend;
  options.draft_tokens = draft_tokens;
  options.prefill_linear = prefill_linear;
  Plan p;
  p.rc = ignis_model_plan_reservations(tensors.data(), tensors.size(), &topology.raw, chunk, max_context,
                                       IGNIS_KV_FORMAT_HQ_E8_2B, &options, &p.reserved);
  if (p.rc != 0) p.error = ignis_model_last_error();
  return p;
}

// The line, recomputed here from the op's own query: the body's seven shapes
// at 1..64 tokens and the chunk, the head's at 1..head_tokens.
uint64_t expected_line(uint32_t head_tokens) {
  struct Call {
    uint32_t in, out, weights;
  };
  const Call body[] = {{kHidden, kConvChannels, 1}, {kHidden, kStateRows, 1}, {kStateRows, kHidden, 1},
                       {kHidden, 2 * kQWidth, 1},   {kHidden, kKvWidth, 1},   {kQWidth, kHidden, 1},
                       {kHidden, kFfn, 2},          {kFfn, kHidden, 1}};
  uint64_t line = 0;
  for (const Call &c : body) {
    for (uint32_t t = 1; t <= IGNIS_TRELLIS_DECODE_MAX_TOKENS; ++t) {
      line = std::max(line, ignis_trellis_workspace_bytes(c.in, c.out, c.weights, t));
    }
    line = std::max(line, ignis_trellis_workspace_bytes(c.in, c.out, c.weights, kChunk));
  }
  for (uint32_t t = 1; t <= head_tokens; ++t) {
    line = std::max(line, ignis_trellis_workspace_bytes(kHidden, kVocab, 1, t));
  }
  return line;
}

// GitHub #509: the line under the dense prefill -- the same decode-route and
// head asks, and at the chunk each shared input's group of matrices asked of
// the dense op.
uint64_t expected_dense_line(uint32_t head_tokens) {
  struct Call {
    uint32_t in, out, weights;
  };
  const Call body[] = {{kHidden, kConvChannels, 1}, {kHidden, kStateRows, 1}, {kStateRows, kHidden, 1},
                       {kHidden, 2 * kQWidth, 1},   {kHidden, kKvWidth, 1},   {kQWidth, kHidden, 1},
                       {kHidden, kFfn, 2},          {kFfn, kHidden, 1}};
  uint64_t line = 0;
  for (const Call &c : body) {
    for (uint32_t t = 1; t <= IGNIS_TRELLIS_DECODE_MAX_TOKENS; ++t) {
      line = std::max(line, ignis_trellis_workspace_bytes(c.in, c.out, c.weights, t));
    }
  }
  struct Group {
    uint32_t in, matrices;
  };
  // GDN qkv and z, GQA q, k and v, gate and up, the two outputs and down.
  const Group groups[] = {{kHidden, 2}, {kHidden, 3}, {kStateRows, 1}, {kQWidth, 1}, {kFfn, 1}};
  for (const Group &g : groups) line = std::max(line, ignis_dense_prefill_workspace_bytes(g.in, g.matrices, kChunk));
  // GitHub #511: the MLP with its A8 fused, down's block maxima past the activations.
  line = std::max(line, ignis_dense_prefill_mlp_workspace_bytes(kHidden, kFfn, kChunk));
  for (uint32_t t = 1; t <= head_tokens; ++t) {
    line = std::max(line, ignis_trellis_workspace_bytes(kHidden, kVocab, 1, t));
  }
  return line;
}

// The closed forms of the trellis body's `workspace` and `decode_graph`
// (kernel/src/model.cu's `compute_program_scratch_bytes` over its trellis
// arms), every buffer rounded to the arena's 256 bytes as the arena does.
uint64_t aligned(uint64_t bytes) { return (bytes + 255) / 256 * 256; }
uint64_t bf16(uint64_t elements) { return aligned(2 * elements); }
uint64_t word(uint64_t elements) { return aligned(4 * elements); }

// The program's outer scope at `T` columns a row, `B` rows: token ids, the
// residual pair, the final norm (one column a row, or every column on a
// verify round) and one row's logits a lane.
uint64_t outer_bytes(uint64_t T, uint64_t B, bool head_every_column) {
  const uint64_t c = T * B;
  return word(c) + 2 * bf16(kHidden * c) + bf16(kHidden * (head_every_column ? c : B)) + bf16(kVocab * B) +
         word(1);
}

// The GQA layer: normalized, post; query, gate, rotated query, attention;
// key, value, rotated key; positions and KV rows; fused; the attention's
// query; and `q_proj`'s interleaved q | gate before its split -- under the
// dense prefill (`dense`, GitHub #509) only for a call the fused op takes, 64
// tokens at most, since the dense GEMM writes the query and the gate split.
uint64_t gqa_bytes(uint64_t T, uint64_t B, uint32_t max_context, uint64_t *attention, bool dense = false) {
  const uint64_t c = T * B;
  *attention = aligned(ignis_gqa_attention_capacity_bytes(
      ninfer::DType::U8, {1, max_context}, static_cast<int32_t>(B), 1, static_cast<int32_t>(T)));
  const uint64_t fused_columns = dense ? std::min<uint64_t>(c, IGNIS_TRELLIS_DECODE_MAX_TOKENS) : c;
  return 2 * bf16(kHidden * c) + 4 * bf16(kQWidth * c) + 3 * bf16(kKvWidth * c) + word(c) + word(B) +
         bf16(kFfn * c) + *attention + bf16(2ull * kQWidth * fused_columns);
}

// Which arena a GDN layer's scratch is planned for (the review of spec #503):
// a decode round's, whose conv is the vendored one into a [conv_channels, T]
// plane; the prefill chunk's, whose conv writes q, k and v split (GitHub #507);
// and the prefill chunk's on our GDN prefill (GitHub #514), whose recurrence
// takes its own workspace from ignis::gdn_prefill::kMinTokens tokens.
enum class GdnArena { kRound, kPrefill, kPrefillOurs };

// The GDN layer: h, post; qkv and, on a round, its convolution; query, key;
// value, z, the recurrence's readout, gated; g, beta; fused; the gating's and
// the recurrence's queries.
uint64_t gdn_bytes(uint64_t T, uint64_t B, GdnArena arena, uint64_t *gating, uint64_t *recurrence) {
  const uint64_t c = T * B;
  const int32_t qk_heads = kKeyWidth / 128;
  *gating = aligned(ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(kValueHeads, kHidden, 1,
                                                                          static_cast<int32_t>(c)));
  const auto vendored = [&](uint64_t width) {
    return aligned(ninfer::ops::gated_delta_net_workspace_capacity_bytes(qk_heads, kValueHeads, /*normalize_qk=*/true,
                                                                         1, static_cast<int32_t>(width)));
  };
  *recurrence = vendored(c);
  if (arena == GdnArena::kPrefillOurs && c >= static_cast<uint64_t>(ignis::gdn_prefill::kMinTokens)) {
    *recurrence = std::max<uint64_t>(
        aligned(ignis::gdn_prefill::workspace_bytes(qk_heads, kValueHeads, static_cast<int32_t>(c))),
        vendored(ignis::gdn_prefill::kMinTokens - 1));
  }
  const uint64_t conv_planes = arena == GdnArena::kRound ? 2 : 1;
  return 2 * bf16(kHidden * c) + conv_planes * bf16(kConvChannels * c) + 2 * bf16(kKeyWidth * c) +
         4 * bf16(kStateRows * c) + 2 * word(kValueHeads * c) + bf16(kFfn * c) + *gating + *recurrence;
}

// Sets (or, with null, removes) a variable in the operating system's
// environment, where the leaf reads the GDN prefill's measurement switch
// (kernel/src/gdn_prefill.h `route_for_load`).
void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

constexpr const char *kGdnSwitch = "IGNIS_GDN_PREFILL";

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device (the trellis workspace query needs one)\n");
    return 1;
  }
  const Topology four = topology_of(4);
  const Topology none = topology_of(0);
  // The closed forms below are the vendored GDN prefill's unless a case names
  // ours: the measurement switch picks the route on any card.
  set_environment(kGdnSwitch, "vendored");

  // The trellis body binds, and its workspace is the widest call.
  {
    const Plan p = plan(trellis_body(4), four, 2);
    check(p.rc == 0, "the trellis body binds: " + p.error);
    if (ignis_trellis_prepare() != 0) {
      std::fprintf(stderr, "FAIL: ignis_trellis_prepare: %s\n", ignis_trellis_last_error());
      return 1;
    }
    // Two lanes, no draft window: the head's widest call is a span-logits block.
    const uint64_t line = expected_line(32);
    check(p.reserved.trellis_workspace_bytes == line,
          "two lanes: the line is the widest call (" + std::to_string(p.reserved.trellis_workspace_bytes) +
              " vs " + std::to_string(line) + ")");
    // Eight lanes of a seven-draft verify round: 64 head columns.
    const Plan wide = plan(trellis_body(4), four, 8, IGNIS_SPECULATIVE_VERIFY_ONLY, 7);
    check(wide.rc == 0, "a windowed trellis load plans: " + wide.error);
    check(wide.reserved.trellis_workspace_bytes == expected_line(64),
          "8 lanes x 8 columns: the head at 64 tokens");
    check(wide.reserved.trellis_workspace_bytes > p.reserved.trellis_workspace_bytes,
          "the head's round columns grow the line");
    std::printf("trellis workspace at chunk %u: %llu bytes at 2 lanes, %llu at 8 lanes x 8 columns\n", kChunk,
                static_cast<unsigned long long>(p.reserved.trellis_workspace_bytes),
                static_cast<unsigned long long>(wide.reserved.trellis_workspace_bytes));
  }

  // `workspace` and `decode_graph` at the 16 GiB load's shape (chunk 1,024,
  // 262,144 keys of hq-e8-2b, two lanes, window 7): their closed forms. The
  // drafter's prefill append, which `rtx50-16gb`'s DFlash2 adds to
  // `workspace`, is the served body's own term; the verify-only backend has
  // the same round arena without it. The prefill chunk's GDN layer holds no
  // conv plane (GitHub #507), a round's does.
  constexpr uint32_t kLanes = 2, kWindow = 7, kContext = 262144;
  {
    const Plan p = plan(trellis_body(4), four, kLanes, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kContext);
    check(p.rc == 0, "the 16 GiB shape plans: " + p.error);
    uint64_t attention = 0, gating = 0, recurrence = 0;
    const uint64_t chunk_outer = outer_bytes(kChunk, 1, false);
    const uint64_t chunk_gqa = gqa_bytes(kChunk, 1, kContext, &attention);
    const uint64_t chunk_gdn = gdn_bytes(kChunk, 1, GdnArena::kPrefill, &gating, &recurrence);
    const uint64_t workspace = chunk_outer + std::max(chunk_gqa, chunk_gdn);
    std::printf("workspace at chunk %u: outer %llu + max(GQA %llu [attention %llu], GDN %llu [gating %llu, "
                "recurrence %llu]) = %llu\n",
                kChunk, static_cast<unsigned long long>(chunk_outer), static_cast<unsigned long long>(chunk_gqa),
                static_cast<unsigned long long>(attention), static_cast<unsigned long long>(chunk_gdn),
                static_cast<unsigned long long>(gating), static_cast<unsigned long long>(recurrence),
                static_cast<unsigned long long>(workspace));
    check(p.reserved.workspace_bytes == workspace,
          "workspace is its closed form (" + std::to_string(p.reserved.workspace_bytes) + " vs " +
              std::to_string(workspace) + ")");
    const uint64_t round_outer = outer_bytes(kWindow + 1, kLanes, true);
    const uint64_t round_gqa = gqa_bytes(kWindow + 1, kLanes, kContext, &attention);
    const uint64_t round_gdn = gdn_bytes(kWindow + 1, kLanes, GdnArena::kRound, &gating, &recurrence);
    const uint64_t decode_graph = round_outer + std::max(round_gqa, round_gdn) + 2 * 4 * kLanes;
    std::printf("decode_graph at %u lanes x %u columns: outer %llu + max(GQA %llu [attention %llu], GDN %llu "
                "[gating %llu, recurrence %llu]) + %u = %llu\n",
                kLanes, kWindow + 1, static_cast<unsigned long long>(round_outer),
                static_cast<unsigned long long>(round_gqa), static_cast<unsigned long long>(attention),
                static_cast<unsigned long long>(round_gdn), static_cast<unsigned long long>(gating),
                static_cast<unsigned long long>(recurrence), 2 * 4 * kLanes,
                static_cast<unsigned long long>(decode_graph));
    check(p.reserved.decode_graph_bytes == decode_graph,
          "decode_graph is its closed form (" + std::to_string(p.reserved.decode_graph_bytes) + " vs " +
              std::to_string(decode_graph) + ")");
  }

  // The review of spec #503: the same shape on the route an 8.x card runs,
  // the dense prefill and our GDN prefill (GitHub #509, #514), whatever card
  // this is. The prefill chunk's GQA layer holds q_proj's interleaved buffer
  // only for a call the fused op takes, and its GDN layer our recurrence's
  // workspace in place of the vendored op's; the round arena does not move.
  {
    const Plan vendored = plan(trellis_body(4), four, kLanes, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kContext,
                               IGNIS_PREFILL_LINEAR_DENSE);
    set_environment(kGdnSwitch, "ours");
    const Plan p = plan(trellis_body(4), four, kLanes, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kContext,
                        IGNIS_PREFILL_LINEAR_DENSE);
    set_environment(kGdnSwitch, "vendored");
    check(p.rc == 0 && vendored.rc == 0, "the 16 GiB shape plans the 8.x routes: " + p.error + vendored.error);
    uint64_t attention = 0, gating = 0, recurrence = 0;
    const uint64_t chunk_outer = outer_bytes(kChunk, 1, false);
    const uint64_t chunk_gqa = gqa_bytes(kChunk, 1, kContext, &attention, /*dense=*/true);
    const uint64_t chunk_gdn = gdn_bytes(kChunk, 1, GdnArena::kPrefillOurs, &gating, &recurrence);
    const uint64_t workspace = chunk_outer + std::max(chunk_gqa, chunk_gdn);
    std::printf("8.x routes at chunk %u: workspace outer %llu + max(GQA %llu, GDN %llu [recurrence %llu]) = %llu "
                "(vendored GDN prefill %llu)\n",
                kChunk, static_cast<unsigned long long>(chunk_outer), static_cast<unsigned long long>(chunk_gqa),
                static_cast<unsigned long long>(chunk_gdn), static_cast<unsigned long long>(recurrence),
                static_cast<unsigned long long>(workspace),
                static_cast<unsigned long long>(vendored.reserved.workspace_bytes));
    check(p.reserved.workspace_bytes == workspace,
          "the 8.x routes' workspace is its closed form (" + std::to_string(p.reserved.workspace_bytes) + " vs " +
              std::to_string(workspace) + ")");
    check(p.reserved.workspace_bytes <= vendored.reserved.workspace_bytes, "our GDN prefill never grows the plan");
    check(p.reserved.decode_graph_bytes == vendored.reserved.decode_graph_bytes &&
              p.reserved.trellis_workspace_bytes == vendored.reserved.trellis_workspace_bytes &&
              p.reserved.trellis_dense_bytes == vendored.reserved.trellis_dense_bytes,
          "our GDN prefill moves no other line");
  }

  // What the routes' sizing holds back at the default chunk and the solo
  // chunks (#489), at rtx-24gb's context: a printout for the report, the
  // closed forms above being the check.
  for (const uint32_t chunk : {2048u, 4096u, 8192u}) {
    constexpr uint32_t kMillion = 1048576;
    const Plan fused = plan(trellis_body(4), four, 8, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kMillion,
                            IGNIS_PREFILL_LINEAR_FUSED, chunk);
    const Plan dense = plan(trellis_body(4), four, 8, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kMillion,
                            IGNIS_PREFILL_LINEAR_DENSE, chunk);
    set_environment(kGdnSwitch, "ours");
    const Plan ours = plan(trellis_body(4), four, 8, IGNIS_SPECULATIVE_VERIFY_ONLY, kWindow, kMillion,
                           IGNIS_PREFILL_LINEAR_DENSE, chunk);
    set_environment(kGdnSwitch, "vendored");
    check(fused.rc == 0 && dense.rc == 0 && ours.rc == 0,
          "chunk " + std::to_string(chunk) + " plans: " + fused.error + dense.error + ours.error);
    std::printf("chunk %u at %u keys: workspace fused %llu, dense %llu, dense + our GDN prefill %llu\n", chunk,
                kMillion, static_cast<unsigned long long>(fused.reserved.workspace_bytes),
                static_cast<unsigned long long>(dense.reserved.workspace_bytes),
                static_cast<unsigned long long>(ours.reserved.workspace_bytes));
  }

  // The served body: no trellis line; the trellis endpoints alone: one.
  {
    const Plan served = plan(served_endpoints(), none, 2);
    check(served.rc == 0, "the served endpoints bind: " + served.error);
    check(served.reserved.trellis_workspace_bytes == 0, "the served body has no trellis workspace");
    const Plan head = plan(trellis_body(0), none, 2);
    check(head.rc == 0, "the trellis endpoints bind: " + head.error);
    check(head.reserved.trellis_workspace_bytes > 0, "a trellis head has a workspace");
    check(head.reserved.workspace_bytes == served.reserved.workspace_bytes &&
              head.reserved.sampling_bytes == served.reserved.sampling_bytes &&
              head.reserved.decode_graph_bytes == served.reserved.decode_graph_bytes,
          "with no layer, the bodies' other lines are the same");
  }

  // GitHub #509: the dense prefill's two lines.
  {
    const Plan fused = plan(trellis_body(4), four, 2);
    const Plan dense = plan(trellis_body(4), four, 2, 0, 0, kMaxContext, IGNIS_PREFILL_LINEAR_DENSE);
    check(dense.rc == 0, "the trellis body plans the dense prefill: " + dense.error);
    check(fused.reserved.trellis_dense_bytes == 0, "the fused prefill reserves no dense buffer");
    check(dense.reserved.trellis_dense_bytes == static_cast<uint64_t>(kHidden) * kFfn,
          "the dense buffer is the largest matrix in INT8, gate, up and down's 5120 x 17408 (" +
              std::to_string(dense.reserved.trellis_dense_bytes) + ")");
    check(dense.reserved.trellis_workspace_bytes == expected_dense_line(32),
          "the dense prefill's trellis workspace is its widest ask (" +
              std::to_string(dense.reserved.trellis_workspace_bytes) + " vs " + std::to_string(expected_dense_line(32)) +
              ")");
    // The prefill chunk's workspace loses q_proj's interleaved buffer past
    // the fused op's 64 tokens; no other line moves.
    uint64_t attention = 0, gating = 0, recurrence = 0;
    const uint64_t dense_workspace =
        outer_bytes(kChunk, 1, false) + std::max(gqa_bytes(kChunk, 1, kMaxContext, &attention, /*dense=*/true),
                                                 gdn_bytes(kChunk, 1, GdnArena::kPrefill, &gating, &recurrence));
    check(dense.reserved.workspace_bytes == dense_workspace,
          "the dense prefill's workspace is its closed form (" + std::to_string(dense.reserved.workspace_bytes) +
              " vs " + std::to_string(dense_workspace) + ")");
    check(dense.reserved.workspace_bytes <= fused.reserved.workspace_bytes, "the dense prefill never grows it");
    check(dense.reserved.decode_graph_bytes == fused.reserved.decode_graph_bytes,
          "the dense prefill moves no round line");
    std::printf("dense prefill at chunk %u: trellis_dense %llu bytes, trellis_workspace %llu (fused prefill %llu)\n",
                kChunk, static_cast<unsigned long long>(dense.reserved.trellis_dense_bytes),
                static_cast<unsigned long long>(dense.reserved.trellis_workspace_bytes),
                static_cast<unsigned long long>(fused.reserved.trellis_workspace_bytes));
    const Plan served = plan(served_endpoints(), none, 2, 0, 0, kMaxContext, IGNIS_PREFILL_LINEAR_DENSE);
    check(served.rc != 0 && contains(served.error, "the dense prefill") && contains(served.error, "NVFP4"),
          "the served (NVFP4) body refuses the dense prefill: " + served.error);
    const Plan unknown = plan(trellis_body(4), four, 2, 0, 0, kMaxContext, 2);
    check(unknown.rc != 0 && contains(unknown.error, "prefill_linear 2"),
          "an unknown prefill route is refused: " + unknown.error);
  }

  // One body, never a mix.
  {
    std::vector<Entry> mixed = trellis_body(4);
    Entry &k = find(mixed, "text/layers/3/attention/k_proj");
    k.qtype = IGNIS_QTYPE_NVFP4;
    k.layout = IGNIS_LAYOUT_BLOCKSCALE_K16_M128X4;
    const Plan p = plan(mixed, four, 2);
    check(p.rc != 0 && contains(p.error, "text/layers/3/attention/k_proj is qtype 7") &&
              contains(p.error, "a load binds one body, never a mix"),
          "an NVFP4 matrix in the trellis body is refused by name: " + p.error);

    std::vector<Entry> served = served_endpoints();
    find(served, "text/final_norm") = trellis("text/final_norm", 128, 128);
    find(served, "text/final_norm").shape = {kHidden};
    const Plan q = plan(served, none, 2);
    check(q.rc != 0 && contains(q.error, "text/final_norm is a trellis tensor where the schema binds a stored"),
          "a trellis tensor where the schema binds a stored weight is refused by name: " + q.error);

    // The served artifact's fused plane in a trellis body: the split parts
    // are what the schema asks for.
    std::vector<Entry> fused = trellis_body(4);
    for (Entry &e : fused) {
      if (e.name == "text/layers/0/gdn/in_proj_qkv") e = trellis("text/layers/0/gdn/query_key_value_z", 16384, kHidden);
    }
    const Plan r = plan(fused, four, 2);
    check(r.rc != 0 && contains(r.error, "missing bound tensor: text/layers/0/gdn/in_proj_qkv"),
          "the trellis schema asks for the split parts: " + r.error);
  }

  // Shapes, K and planes.
  {
    std::vector<Entry> shape = trellis_body(4);
    find(shape, "text/layers/3/attention/q_proj").shape = {kQWidth, kHidden};
    const Plan p = plan(shape, four, 2);
    check(p.rc != 0 && contains(p.error, "text/layers/3/attention/q_proj has an unexpected shape (want [12288,5120])"),
          "a q_proj of the query alone is refused: " + p.error);

    std::vector<Entry> short_k = trellis_body(4);
    Entry &down = find(short_k, "text/layers/1/mlp/down");
    down.bytes = record_bytes(kHidden, kFfn, IGNIS_QTYPE_TRELLIS_MUL1_K3);  // K3's record, K4's code
    const Plan q = plan(short_k, four, 2);
    check(q.rc != 0 && contains(q.error, "text/layers/1/mlp/down's") && contains(q.error, "cannot hold"),
          "a record too short for its K is refused: " + q.error);
    for (int32_t code = IGNIS_QTYPE_TRELLIS_MUL1_K2; code <= IGNIS_QTYPE_TRELLIS_MUL1_K6; ++code) {
      std::vector<Entry> exact = trellis_body(0);
      Entry &head = find(exact, "text/output_head");
      head.qtype = code;
      head.bytes = tile_bytes(kVocab, kHidden, code) + 2ull * (kVocab + kHidden);
      check(plan(exact, none, 2).rc == 0, "code " + std::to_string(code) + ": its K's exact record binds");
      head.bytes -= 1;
      check(plan(exact, none, 2).rc != 0, "code " + std::to_string(code) + ": a byte short is refused");
    }

    // Addresses are never read, only checked against the record's layout.
    std::vector<Entry> placed = trellis_body(0);
    Entry &head = find(placed, "text/output_head");
    const auto *base = reinterpret_cast<const uint8_t *>(uintptr_t{1} << 40);
    const uint64_t tiles = tile_bytes(kVocab, kHidden, head.qtype);
    head.qdata = base;
    head.qhigh = base + tiles;
    head.scales = base + tiles + 2ull * kHidden;
    check(plan(placed, none, 2).rc == 0, "suh and svh where the record puts them bind");
    head.scales = base + tiles + 2ull * kVocab;  // suh's length taken from out, not in
    const Plan r = plan(placed, none, 2);
    check(r.rc != 0 && contains(r.error, "text/output_head's suh and svh are not where its record puts them"),
          "a misplaced svh is refused: " + r.error);
  }

  set_environment(kGdnSwitch, nullptr);
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
