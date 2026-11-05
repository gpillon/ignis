// GitHub #507: the GDN prefill copies no layout between its kernels -- OURS, not vendored.
//
// The GDN layer's prefill body (kernel/src/gdn_layer.cu) used to split its conv's output into the
// query, key and value planes with three strided device-to-device copies, and to copy its input
// residual into the output buffer before the output projection added into it. Now the conv writes
// the planes split (kernel/src/gdn_conv_split.h) and the body consumes its residual in place when
// the caller hands it one buffer, as the chunk loop (kernel/src/step.cu) does. The arithmetic is
// unchanged, so the bound is bit equality against the body as it ran before, which the leaf keeps
// behind a test-only switch (`ignis_gdn_layer_set_layout_copies`, kernel/src/layer_internal.h)
// so this run produces its own golden.
//
// The model is synthetic and small but at the 27B's widths: a trellis body (the 16 GB entries'
// and the one 8.x cards load) of five layers in the 27B's pattern, GDN, GDN, GDN, GQA, GDN, with
// counter-hash weights.
//
//   layer     a GDN layer (the first, and the one past the GQA layer) through
//             `ignis_gdn_layer_step` at the prefill chunks 1024, 2048 and
//             4096, two calls in a row so the second reads the taps and the recurrent state the
//             first carried: the pre-#507 body on two buffers against today's on one buffer
//             (in place) and on two (the step ABI's form for a caller that keeps its input).
//             Outputs, conv taps and recurrent state, bit for bit; the input of the two-buffer
//             form untouched; and the copy counter: four per call before, none in place, the
//             residual's one on two buffers.
//   program   `ignis_program_prefill` on the chunked route over a span of three chunks: the
//             pre-#507 loop (two buffers, a swap after every layer) against today's (a GDN layer
//             in place, a swap after a GQA one), across GDN-to-GDN, GDN-to-GQA and GQA-to-GDN
//             handoffs. The span's last logits, the pending token and every GDN layer's carried
//             state, bit for bit; no layout copy in today's prefill.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "ignis_gdn_layer.h"
#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_step.h"

#include "layer_internal.h"
#include "moe_fixture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

// The 27B's widths (crates/core/src/compute.rs `ModelConfig::qwen38_27b`), a small vocabulary.
constexpr int32_t kHidden = 5120;
constexpr int32_t kVocab = 1024;
constexpr int32_t kQWidth = 24 * 256;
constexpr int32_t kKvWidth = 4 * 256;
constexpr int32_t kFfn = 17408;
constexpr int32_t kValueHeads = 48;
constexpr int32_t kKeyWidth = 16 * 128;
constexpr int32_t kStateRows = kValueHeads * 128;
constexpr int32_t kConvChannels = 2 * kKeyWidth + kStateRows;
constexpr uint32_t kChunk = 4096;
constexpr uint32_t kMaxContext = 9216;
// The 27B's pattern (a GQA layer every fourth, kernel/src/layer_internal.h), so the pool's GDN-
// relative layers are the leaf's own: 0, 1, 2 and, past the GQA layer, 3.
const std::vector<int32_t> kKinds = {IGNIS_LAYER_GDN, IGNIS_LAYER_GDN, IGNIS_LAYER_GDN, IGNIS_LAYER_GQA,
                                     IGNIS_LAYER_GDN};
constexpr uint32_t kGdnLayers = 4;

#define LOAD_RC(expr, last_error)                                                                  \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, last_error());              \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// One bound tensor and the device bytes behind it.
struct Bound {
  std::string name;
  std::vector<int32_t> shape;
  int32_t qtype = IGNIS_QTYPE_BF16_CTRL;
  int32_t layout = IGNIS_LAYOUT_CONTIGUOUS;
  DeviceBytes data;
  uint64_t tiles = 0;  // a trellis record's tile bytes (suh and svh follow them)
};

uint32_t g_seed = 1;

Bound bf16(const std::string &name, std::vector<int32_t> shape, float amplitude) {
  std::size_t n = 1;
  for (int32_t d : shape) n *= static_cast<std::size_t>(d);
  std::vector<uint16_t> v(n);
  const uint32_t seed = g_seed++;
  for (std::size_t i = 0; i < n; ++i) v[i] = f32_to_bf16(hash_uniform(seed, i, amplitude));
  Bound b{name, std::move(shape)};
  b.data = DeviceBytes(n * 2);
  upload(b.data, v);
  return b;
}

Bound fp32(const std::string &name, int32_t n, float amplitude) {
  std::vector<float> v(static_cast<std::size_t>(n));
  const uint32_t seed = g_seed++;
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = hash_uniform(seed, i, amplitude);
  Bound b{name, {n}, IGNIS_QTYPE_FP32_CTRL};
  b.data = DeviceBytes(v.size() * 4);
  upload(b.data, v);
  return b;
}

// A trellis record of counter-hash words at K3 (2K = 6), `[tiles | suh fp16[in] | svh fp16[out]]`
// as the artifact stores it, padded to 4096 bytes. suh is +-1 and svh +-1/64, so each layer's
// adds stay a few units wide and every value finite.
Bound trellis(const std::string &name, int32_t rows, int32_t cols) {
  const uint64_t tiles = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) / 16 * 6;
  const uint64_t bytes = (tiles + 2ull * (static_cast<uint64_t>(rows) + cols) + 4095) / 4096 * 4096;
  std::vector<uint16_t> v(bytes / 2, 0);
  const uint32_t seed = g_seed++;
  for (uint64_t i = 0; i < tiles / 2; ++i) v[i] = static_cast<uint16_t>(hash_u32(seed, i));
  for (int32_t i = 0; i < cols; ++i) v[tiles / 2 + i] = (hash_u32(seed + 1000, i) & 1u) ? 0x3C00u : 0xBC00u;
  for (int32_t i = 0; i < rows; ++i) {
    v[tiles / 2 + cols + i] = (hash_u32(seed + 2000, i) & 1u) ? 0x2400u : 0xA400u;
  }
  Bound b{name, {rows, cols}, IGNIS_QTYPE_TRELLIS_MUL1_K3, IGNIS_LAYOUT_TRELLIS_TILE16};
  b.data = DeviceBytes(bytes);
  b.tiles = tiles;
  upload(b.data, v);
  return b;
}

struct Fixture {
  std::vector<Bound> tensors;
  ignis_model *model = nullptr;
  ignis_seq_pool *pool = nullptr;

  Fixture() {
    tensors.push_back(bf16("text/token_embedding", {kVocab, kHidden}, 0.5f));
    tensors.push_back(bf16("text/final_norm", {kHidden}, 0.1f));
    tensors.push_back(trellis("text/output_head", kVocab, kHidden));
    for (std::size_t i = 0; i < kKinds.size(); ++i) {
      const std::string p = "text/layers/" + std::to_string(i) + "/";
      tensors.push_back(bf16(p + "input_norm", {kHidden}, 0.1f));
      if (kKinds[i] == IGNIS_LAYER_GQA) {
        tensors.push_back(trellis(p + "attention/q_proj", 2 * kQWidth, kHidden));
        tensors.push_back(trellis(p + "attention/k_proj", kKvWidth, kHidden));
        tensors.push_back(trellis(p + "attention/v_proj", kKvWidth, kHidden));
        tensors.push_back(bf16(p + "attention/query_norm", {256}, 0.1f));
        tensors.push_back(bf16(p + "attention/key_norm", {256}, 0.1f));
        tensors.push_back(trellis(p + "attention/output", kHidden, kQWidth));
      } else {
        tensors.push_back(fp32(p + "gdn/a_log", kValueHeads, 1.0f));
        tensors.push_back(fp32(p + "gdn/dt_bias", kValueHeads, 1.0f));
        tensors.push_back(bf16(p + "gdn/convolution", {4, kConvChannels}, 0.5f));
        tensors.push_back(bf16(p + "gdn/a_b_projection", {2 * kValueHeads, kHidden}, 0.02f));
        tensors.push_back(trellis(p + "gdn/in_proj_qkv", kConvChannels, kHidden));
        tensors.push_back(trellis(p + "gdn/in_proj_z", kStateRows, kHidden));
        tensors.push_back(bf16(p + "gdn/norm", {128}, 0.1f));
        tensors.push_back(trellis(p + "gdn/output", kHidden, kStateRows));
      }
      tensors.push_back(bf16(p + "post_attention_norm", {kHidden}, 0.1f));
      tensors.push_back(trellis(p + "mlp/gate_proj", kFfn, kHidden));
      tensors.push_back(trellis(p + "mlp/up_proj", kFfn, kHidden));
      tensors.push_back(trellis(p + "mlp/down", kHidden, kFfn));
    }

    std::vector<ignis_bound_tensor> bound;
    for (const Bound &b : tensors) {
      ignis_bound_tensor t{};
      t.name = b.name.c_str();
      t.qtype = b.qtype;
      t.layout = b.layout;
      t.qdata = b.data.p;
      t.bytes = b.data.bytes;
      if (b.layout == IGNIS_LAYOUT_TRELLIS_TILE16) {
        t.qhigh = static_cast<const uint8_t *>(b.data.p) + b.tiles;
        t.scales = static_cast<const uint8_t *>(b.data.p) + b.tiles + 2ull * b.shape[1];
      }
      t.ndim = static_cast<uint32_t>(b.shape.size());
      for (std::size_t d = 0; d < 4; ++d) {
        t.shape[d] = d < b.shape.size() ? b.shape[d] : 1;
        t.padded_shape[d] = t.shape[d];
      }
      bound.push_back(t);
    }
    ignis_topology topology{};
    topology.num_layers = static_cast<uint32_t>(kKinds.size());
    topology.layer_kinds = kKinds.data();
    topology.hidden = kHidden;
    topology.vocab = kVocab;
    topology.num_q_heads = 24;
    topology.num_kv_heads = 4;
    topology.head_dim = 256;
    topology.rotary_dim = 64;
    topology.rope_theta = 1e7;
    topology.gdn_state_rows = kStateRows;
    topology.gdn_state_cols = kKeyWidth;
    topology.gdn_num_layers = kGdnLayers;
    topology.gdn_value_heads = kValueHeads;
    topology.gdn_head_dim = 128;
    topology.gdn_conv_kernel = 4;
    topology.gdn_q_width = kKeyWidth;
    topology.gdn_z_width = kStateRows;
    topology.gdn_ab_width = 2 * kValueHeads;
    topology.ffn_intermediate = kFfn;
    topology.rms_norm_eps = 1e-6f;
    ignis_model_load_options options{};
    options.size = sizeof(options);
    options.decode_lanes = 1;
    LOAD_RC(ignis_model_load(bound.data(), bound.size(), &topology, kChunk, kMaxContext, IGNIS_KV_FORMAT_BF16,
                             &options, &model),
            ignis_model_last_error);

    ignis_seq_pool_spec spec{};
    spec.num_kv_heads = 4;
    spec.head_dim = 256;
    spec.kv_format = IGNIS_KV_FORMAT_BF16;
    spec.kv_page_group_count = kMaxContext / 64 + 8;
    spec.max_context_tokens = kMaxContext;
    spec.slot_count = 1;
    spec.gdn_num_layers = kGdnLayers;
    spec.gdn_conv_channels = kConvChannels;
    spec.gdn_value_heads = kValueHeads;
    spec.gdn_head_dim = 128;
    spec.vocab = kVocab;
    spec.kv_num_layers = 1;
    LOAD_RC(ignis_seq_pool_create(&spec, &pool), ignis_seq_last_error);
  }

  ~Fixture() {
    ignis_seq_pool_free(pool);
    ignis_model_free(model);
  }

  ignis_seq *alloc() {
    ignis_seq *seq = nullptr;
    LOAD_RC(ignis_seq_alloc(pool, kMaxContext, &seq), ignis_seq_last_error);
    return seq;
  }

  // A GDN layer's carried state in `seq`'s slot: its conv taps, then its fp32 recurrent state.
  std::vector<uint8_t> state(const ignis_seq *seq, uint32_t gdn_layer) const {
    const ninfer::Tensor conv = pool->gdn_pool.conv_slot(gdn_layer, seq->slot);
    const ninfer::Tensor recurrent = pool->gdn_pool.recurrent_slot(gdn_layer, seq->slot);
    MOE_CUDA(cudaDeviceSynchronize());
    std::vector<uint8_t> bytes = download<uint8_t>(conv.data, pool->gdn_pool.conv_slot_bytes());
    const std::vector<uint8_t> r = download<uint8_t>(recurrent.data, pool->gdn_pool.recurrent_slot_bytes());
    bytes.insert(bytes.end(), r.begin(), r.end());
    return bytes;
  }
};

// A residual of a hidden state's size, varied per token and feature.
std::vector<uint16_t> residual(uint32_t seed, int32_t tokens) {
  std::vector<uint16_t> v(static_cast<std::size_t>(tokens) * kHidden);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = f32_to_bf16(hash_uniform(seed, i, 1.0f));
  return v;
}

bool all_finite_bf16(const std::vector<uint16_t> &v) {
  for (uint16_t b : v) {
    if ((b & 0x7F80u) == 0x7F80u) return false;
  }
  return true;
}

void same(bool equal, const std::string &label) { check(equal, label + " differs"); }

// --- layer ---------------------------------------------------------------------------------

enum class Form { BeforeTwoBuffers, InPlace, TwoBuffers };

const char *form_name(Form f) {
  return f == Form::BeforeTwoBuffers ? "the pre-#507 body" : f == Form::InPlace ? "in place" : "two buffers";
}

struct LayerRun {
  std::vector<uint16_t> out[2];
  std::vector<uint8_t> state;
  uint64_t copies = 0;
  bool input_kept = true;
  bool residual_moved = true;  // each call's output differs from its input
};

LayerRun run_layer(Fixture &f, uint32_t layer, int32_t T, Form form) {
  ignis_gdn_layer_set_layout_copies(form == Form::BeforeTwoBuffers);
  ignis_seq *seq = f.alloc();
  LayerRun run;
  const uint64_t before = ignis_gdn_layer_layout_copies_enqueued();
  for (int call = 0; call < 2; ++call) {
    const std::vector<uint16_t> x = residual(50 + 7 * static_cast<uint32_t>(call) + static_cast<uint32_t>(T), T);
    DeviceBytes in(x.size() * 2), out(x.size() * 2);
    upload(in, x);
    void *dst = form == Form::InPlace ? in.p : out.p;
    LOAD_RC(ignis_gdn_layer_step(f.model, f.pool, seq, layer, in.p, dst, static_cast<uint64_t>(T)),
            ignis_gdn_layer_last_error);
    run.out[call] = download<uint16_t>(dst, x.size());
    run.residual_moved = run.residual_moved && run.out[call] != x;
    if (form != Form::InPlace) run.input_kept = run.input_kept && download<uint16_t>(in.p, x.size()) == x;
  }
  run.copies = ignis_gdn_layer_layout_copies_enqueued() - before;
  run.state = f.state(seq, ignis_gdn_relative_layer(layer));
  ignis_seq_release(f.pool, seq);
  ignis_gdn_layer_set_layout_copies(false);
  return run;
}

void the_layer_is_the_body_before_it_bit_for_bit(Fixture &f) {
  for (const uint32_t layer : {0u, 4u}) {
    for (const int32_t T : {1024, 2048, 4096}) {
      const std::string at = "layer " + std::to_string(layer) + ", chunk " + std::to_string(T);
      const LayerRun before = run_layer(f, layer, T, Form::BeforeTwoBuffers);
      check(all_finite_bf16(before.out[0]) && all_finite_bf16(before.out[1]), at + ": the outputs are finite");
      check(before.residual_moved, at + ": each call moves its residual");
      check(std::any_of(before.state.begin(), before.state.end(), [](uint8_t b) { return b != 0; }),
            at + ": the carried state is not zero");
      check(before.copies == 8, at + ": the pre-#507 body enqueued " + std::to_string(before.copies) +
                                    " layout copies over two calls, not 8");
      for (const Form form : {Form::InPlace, Form::TwoBuffers}) {
        const LayerRun now = run_layer(f, layer, T, form);
        const std::string label = at + ", " + form_name(form);
        same(now.out[0] == before.out[0], label + ": the first call's output");
        same(now.out[1] == before.out[1], label + ": the second call's output (carried state)");
        same(now.state == before.state, label + ": the conv taps and recurrent state after both");
        const uint64_t want = form == Form::InPlace ? 0 : 2;
        check(now.copies == want, label + ": " + std::to_string(now.copies) + " layout copies enqueued, not " +
                                      std::to_string(want));
        check(now.input_kept, label + ": the input residual is untouched");
      }
    }
  }
}

// --- program -------------------------------------------------------------------------------

struct ProgramRun {
  std::vector<float> logits;
  int32_t pending = -1;
  std::vector<uint8_t> state[kGdnLayers];
  uint64_t copies = 0;
};

ProgramRun run_program(Fixture &f, bool before_507) {
  ignis_gdn_layer_set_layout_copies(before_507);
  ignis_seq *seq = f.alloc();
  const uint64_t span = 2 * kChunk + 777;
  std::vector<int32_t> ids(span);
  for (uint64_t i = 0; i < span; ++i) ids[i] = static_cast<int32_t>(hash_u32(77, i) % kVocab);
  ignis_sampling_params sampling{};
  sampling.size = sizeof(sampling);
  sampling.greedy = 1;
  sampling.reasoning_close_id = -1;
  ProgramRun run;
  run.logits.assign(kVocab, 0.0f);
  const uint64_t before = ignis_gdn_layer_layout_copies_enqueued();
  LOAD_RC(ignis_program_prefill(f.model, f.pool, seq, ids.data(), span, 0, &sampling, nullptr, run.logits.data()),
          ignis_step_last_error);
  run.copies = ignis_gdn_layer_layout_copies_enqueued() - before;
  run.pending = ignis_seq_pending_token(seq);
  for (uint32_t g = 0; g < kGdnLayers; ++g) run.state[g] = f.state(seq, g);
  ignis_seq_release(f.pool, seq);
  ignis_gdn_layer_set_layout_copies(false);
  return run;
}

void the_chunked_prefill_is_the_loop_before_it_bit_for_bit(Fixture &f) {
  const ProgramRun before = run_program(f, true);
  const ProgramRun now = run_program(f, false);
  bool finite = true;
  for (float v : before.logits) finite = finite && std::isfinite(v);
  check(finite, "program: the span's last logits are finite");
  check(*std::min_element(before.logits.begin(), before.logits.end()) <
            *std::max_element(before.logits.begin(), before.logits.end()),
        "program: the span's last logits are not all one value");
  check(before.copies == 48, "program: the pre-#507 loop enqueued " + std::to_string(before.copies) +
                                 " layout copies over four GDN layers and three chunks, not 48");
  check(now.copies == 0, "program: today's prefill enqueued " + std::to_string(now.copies) + " layout copies");
  same(std::memcmp(now.logits.data(), before.logits.data(), kVocab * sizeof(float)) == 0,
       "program: the span's last logits");
  check(now.pending == before.pending, "program: the pending token");
  for (uint32_t g = 0; g < kGdnLayers; ++g) {
    same(now.state[g] == before.state[g], "program: GDN layer " + std::to_string(g) + "'s carried state");
  }
}

} // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device\n");
    return 1;
  }
  {
    Fixture f;
    the_layer_is_the_body_before_it_bit_for_bit(f);
    the_chunked_prefill_is_the_loop_before_it_bit_for_bit(f);
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
