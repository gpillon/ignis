// GitHub #302 (spec flash-next/04): the Flash-Next binder
// (kernel/src/flash_next/bind.cu) -- OURS, not vendored.
//
// Descriptors named and shaped as docs/formats/flash-next-layout.md section 6
// and crates/artifact/src/flash_next.rs's inventory have them, at Flash-Next's
// real geometry (48 layers), carrying no planes: the binder only reads names,
// formats and shapes. A complete set binds and every weight lands where the
// program reads it; a linear re-converted to BF16 binds as BF16; a missing,
// extra, mis-shaped or mis-formatted tensor is refused by name.
//
// Host-only. ADR 0006: no SKIP_RETURN_CODE.

#include "flash_next/bind.h"
#include "flash_next/verify.h"

#include "ignis_model.h"
#include "ignis_seq.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

struct Named {
  std::string name;
  std::vector<int32_t> shape;
  bool linear;  // FP8 row-scale by default, else BF16
};

constexpr int32_t H = 2560, HC = 10240, RANK = 320, V = 248320, E = 512, I = 640;

std::vector<Named> flash_next_tensors() {
  std::vector<Named> out{
      {"embed_tokens.weight", {V, H}, true},
      {"lm_head.weight", {V, H}, true},
      {"hyper_connection_mixer.hc_norm.weight", {HC}, false},
      {"hyper_connection_mixer.input_mix_weight_down.weight", {RANK, HC}, true},
      {"hyper_connection_mixer.input_mix_weight_up.weight", {HC, RANK}, true},
  };
  for (int l = 0; l < 48; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    for (const char *block : {"attn_hyper_connection", "mlp_hyper_connection"}) {
      const std::string b = p + block;
      out.push_back({b + ".block_inject_weight.weight", {4, HC}, false});
      out.push_back({b + ".hc_norm.weight", {HC}, false});
      out.push_back({b + ".input_mix_weight_down.weight", {RANK, HC}, true});
      out.push_back({b + ".input_mix_weight_up.weight", {HC, RANK}, true});
    }
    if ((l + 1) % 4 == 0) {
      out.push_back({p + "self_attn.q_proj.weight", {12288, H}, true});
      out.push_back({p + "self_attn.k_proj.weight", {512, H}, true});
      out.push_back({p + "self_attn.v_proj.weight", {512, H}, true});
      out.push_back({p + "self_attn.o_proj.weight", {H, 6144}, true});
      out.push_back({p + "self_attn.q_norm.weight", {256}, false});
      out.push_back({p + "self_attn.k_norm.weight", {256}, false});
      out.push_back({p + "self_attn.indexer.index_qk_proj.weight", {640, H}, true});
      out.push_back({p + "self_attn.indexer.q_layernorm.weight", {128}, false});
      out.push_back({p + "self_attn.indexer.k_layernorm.weight", {128}, false});
    } else {
      out.push_back({p + "linear_attn.in_proj_qkv.weight", {10240, H}, true});
      out.push_back({p + "linear_attn.in_proj_z.weight", {6144, H}, true});
      out.push_back({p + "linear_attn.in_proj_a.weight", {48, H}, true});
      out.push_back({p + "linear_attn.in_proj_b.weight", {48, H}, true});
      out.push_back({p + "linear_attn.out_proj.weight", {H, 6144}, true});
      out.push_back({p + "linear_attn.conv1d.weight", {10240, 1, 4}, false});
      out.push_back({p + "linear_attn.A_log", {48}, false});
      out.push_back({p + "linear_attn.dt_bias", {48}, false});
      out.push_back({p + "linear_attn.norm.weight", {128}, false});
    }
    out.push_back({p + "mlp.gate.weight", {E, H}, false});
    out.push_back({p + "mlp.shared_expert.gate_proj.weight", {I, H}, true});
    out.push_back({p + "mlp.shared_expert.up_proj.weight", {I, H}, true});
    out.push_back({p + "mlp.shared_expert.down_proj.weight", {H, I}, true});
    out.push_back({p + "mlp.shared_expert_gate.weight", {1, H}, false});
    if (l == 1) {
      out.push_back({p + "ple.conv1d.weight", {HC, 1, 4}, false});
      out.push_back({p + "ple.key_proj.weight", {HC, H}, true});
      out.push_back({p + "ple.norm_conv.weight", {HC}, false});
      out.push_back({p + "ple.norm_key.weight", {HC}, false});
      out.push_back({p + "ple.norm_query.weight", {HC}, false});
      out.push_back({p + "ple.value_proj.weight", {H, H}, true});
    }
  }
  return out;
}

ignis_topology flash_next_topology(std::vector<int32_t> &kinds) {
  kinds.clear();
  for (int i = 0; i < 48; ++i) {
    kinds.push_back((i + 1) % 4 == 0 ? IGNIS_LAYER_GQA : IGNIS_LAYER_GDN);
  }
  ignis_topology t{};
  t.num_layers = 48;
  t.layer_kinds = kinds.data();
  t.hidden = H;
  t.vocab = V;
  t.num_q_heads = 24;
  t.num_kv_heads = 2;
  t.head_dim = 256;
  t.rotary_dim = 64;
  t.rope_theta = 1e7;
  t.gdn_state_rows = 6144;
  t.gdn_state_cols = 2048;
  t.gdn_num_layers = 36;
  t.gdn_q_width = 2048;
  t.gdn_z_width = 6144;
  t.gdn_ab_width = 96;
  t.rms_norm_eps = 1e-6f;
  t.family = IGNIS_MODEL_FAMILY_FLASH_NEXT;
  t.gdn_value_heads = 48;
  t.gdn_head_dim = 128;
  t.gdn_conv_kernel = 4;
  t.moe = {E, 10, I, I};
  t.hyper = {4, RANK};
  t.indexer = {4, 128, 1, 4, 2048};
  t.ngram = {3, 8, H, 4, 1};
  return t;
}

// Descriptors for `named`, each with a distinct fake plane address (index+1)
// so the test can see which weight landed where.
std::vector<ignis_bound_tensor> descriptors(const std::vector<Named> &named) {
  std::vector<ignis_bound_tensor> out;
  for (std::size_t i = 0; i < named.size(); ++i) {
    ignis_bound_tensor t{};
    t.name = named[i].name.c_str();
    t.qtype = named[i].linear ? IGNIS_QTYPE_FP8_E4M3FN_ROW_BF16S : IGNIS_QTYPE_BF16_CTRL;
    t.layout = named[i].linear ? IGNIS_LAYOUT_ROW_SCALE : IGNIS_LAYOUT_CONTIGUOUS;
    t.qdata = reinterpret_cast<const void *>(static_cast<uintptr_t>((i + 1) * 256));
    t.ndim = static_cast<uint32_t>(named[i].shape.size());
    for (std::size_t d = 0; d < 4; ++d) {
      t.shape[d] = d < named[i].shape.size() ? named[i].shape[d] : 1;
      t.padded_shape[d] = t.shape[d];
    }
    out.push_back(t);
  }
  return out;
}

const void *plane_of(const std::vector<Named> &named, const std::string &name) {
  for (std::size_t i = 0; i < named.size(); ++i) {
    if (named[i].name == name) {
      return reinterpret_cast<const void *>(static_cast<uintptr_t>((i + 1) * 256));
    }
  }
  return nullptr;
}

// Binds `named` (after `edit`) and returns the binder's error, empty on success.
std::string bind_error(std::vector<Named> named, const std::function<void(std::vector<ignis_bound_tensor> &)> &edit = {}) {
  std::vector<int32_t> kinds;
  const ignis_topology topology = flash_next_topology(kinds);
  auto tensors = descriptors(named);
  if (edit) {
    edit(tensors);
  }
  std::string error;
  const auto weights = ignis::flash_next::bind_flash_next(tensors.data(), tensors.size(), topology, &error);
  return weights == nullptr ? error : std::string();
}

std::size_t index_of(const std::vector<Named> &named, const std::string &name) {
  for (std::size_t i = 0; i < named.size(); ++i) {
    if (named[i].name == name) return i;
  }
  return named.size();
}

}  // namespace

int main() {
  const std::vector<Named> named = flash_next_tensors();

  // A complete set binds, and each weight is where the program reads it.
  {
    std::vector<int32_t> kinds;
    const ignis_topology topology = flash_next_topology(kinds);
    const auto tensors = descriptors(named);
    std::string error;
    const auto w = ignis::flash_next::bind_flash_next(tensors.data(), tensors.size(), topology, &error);
    check(w != nullptr, "the complete Flash-Next set binds: " + error);
    if (w != nullptr) {
      check(w->layers.size() == 48, "48 layers");
      check(w->layers[3].attention && !w->layers[2].attention, "layer 3 is QSA, layer 2 GDN");
      check(w->layers[3].qsa.q_proj.data == plane_of(named, "layers.3.self_attn.q_proj.weight") &&
                w->layers[3].qsa.q_proj.rows == 12288 && w->layers[3].qsa.q_proj.cols == 2560,
            "QSA q_proj: [24 x 2 x 256, 2560] from its own tensor");
      check(w->layers[3].qsa.indexer.qk_proj.rows == 640, "indexer qk_proj: (4 + 1) x 128 rows");
      check(w->layers[2].gdn.in_proj_a.data == plane_of(named, "layers.2.linear_attn.in_proj_a.weight") &&
                w->layers[2].gdn.in_proj_b.data == plane_of(named, "layers.2.linear_attn.in_proj_b.weight"),
            "GDN in_proj_a and in_proj_b stay apart");
      check(w->layers[2].gdn.norm == plane_of(named, "layers.2.linear_attn.norm.weight"), "GDN norm");
      check(w->layers[0].attn_hc.block_inject == plane_of(named, "layers.0.attn_hyper_connection.block_inject_weight.weight"),
            "attention HC block-inject weight");
      check(w->final_mixer.block_inject == nullptr, "the final mixer has no inject weights");
      check(w->ngram.key_proj.data == plane_of(named, "layers.1.ple.key_proj.weight") &&
                w->ngram.key_proj.rows == 10240,
            "the n-gram projections come from layer 1's PLE");
      check(w->layers[47].moe.router == plane_of(named, "layers.47.mlp.gate.weight") &&
                w->layers[47].moe.shared_down == plane_of(named, "layers.47.mlp.shared_expert.down_proj.weight"),
            "MoE router and shared expert");
      check(w->embed.format == ignis::flash_next::WeightFormat::Fp8RowScale, "embed_tokens FP8");
    }
  }

  // A linear the conversion moved to BF16 binds as BF16.
  {
    std::vector<int32_t> kinds;
    const ignis_topology topology = flash_next_topology(kinds);
    auto tensors = descriptors(named);
    auto &head = tensors[index_of(named, "lm_head.weight")];
    head.qtype = IGNIS_QTYPE_BF16_CTRL;
    head.layout = IGNIS_LAYOUT_CONTIGUOUS;
    std::string error;
    const auto w = ignis::flash_next::bind_flash_next(tensors.data(), tensors.size(), topology, &error);
    check(w != nullptr && w->head.format == ignis::flash_next::WeightFormat::Bf16,
          "a BF16 lm_head binds as BF16: " + error);
  }

  // Refusals, named.
  {
    auto missing = named;
    missing.erase(missing.begin() + static_cast<std::ptrdiff_t>(index_of(named, "layers.5.linear_attn.A_log")));
    const std::string error = bind_error(missing);
    check(error.find("missing bound tensor: layers.5.linear_attn.A_log") != std::string::npos,
          "a missing tensor is named: got \"" + error + "\"");
  }
  {
    auto extra = named;
    extra.push_back({"layers.0.mlp.experts.0.gate_up_proj", {1280, H}, false});
    const std::string error = bind_error(extra);
    check(error.find("extra bound tensor: layers.0.mlp.experts.0.gate_up_proj") != std::string::npos,
          "an expert handed over as a bound tensor is an extra: got \"" + error + "\"");
  }
  {
    auto wrong = named;
    wrong[index_of(named, "layers.7.self_attn.k_proj.weight")].shape = {1024, H};
    const std::string error = bind_error(wrong);
    check(error.find("layers.7.self_attn.k_proj.weight has an unexpected shape (want [512,2560])") != std::string::npos,
          "a mis-shaped tensor is named with the shape it should have: got \"" + error + "\"");
  }
  {
    const std::string error = bind_error(named, [&](std::vector<ignis_bound_tensor> &t) {
      auto &norm = t[index_of(named, "layers.0.attn_hyper_connection.hc_norm.weight")];
      norm.qtype = IGNIS_QTYPE_FP8_E4M3FN_ROW_BF16S;
      norm.layout = IGNIS_LAYOUT_ROW_SCALE;
    });
    check(error.find("layers.0.attn_hyper_connection.hc_norm.weight is not BF16") != std::string::npos,
          "a norm in FP8 is refused: got \"" + error + "\"");
  }
  {
    const std::string error = bind_error(named, [&](std::vector<ignis_bound_tensor> &t) {
      auto &gate = t[index_of(named, "layers.4.mlp.shared_expert.gate_proj.weight")];
      gate.qtype = IGNIS_QTYPE_BF16_CTRL;
      gate.layout = IGNIS_LAYOUT_CONTIGUOUS;
    });
    check(error.find("shared expert runs FP8") != std::string::npos,
          "a BF16 shared expert is refused (its op runs FP8 only): got \"" + error + "\"");
  }
  {
    std::vector<int32_t> kinds;
    ignis_topology topology = flash_next_topology(kinds);
    topology.hyper = {0, 0};
    const auto tensors = descriptors(named);
    std::string error;
    check(ignis::flash_next::bind_flash_next(tensors.data(), tensors.size(), topology, &error) == nullptr &&
              error.find("not a Flash-Next topology") != std::string::npos,
          "a topology without hyper-connections is not Flash-Next's: got \"" + error + "\"");
  }

  // Through the load ABI: the Flash-Next family reaches this binder and its
  // program (kernel/src/flash_next/program.cu), whose plan lines a set that
  // binds gets back; a binder fault comes back by name, the 27B's options
  // and a lane count past the round's widest are refused, and a load
  // without its expert residency is refused before it reserves anything.
  {
    std::vector<int32_t> kinds;
    const ignis_topology topology = flash_next_topology(kinds);
    auto tensors = descriptors(named);
    ignis_model_reservations out{};
    const auto plan = [&](const ignis_model_load_options *options) {
      const int32_t rc = ignis_model_plan_reservations(tensors.data(), tensors.size(), &topology, 8192,
                                                       131072, IGNIS_KV_FORMAT_HQ_E8_2B, options, &out);
      return rc == 0 ? std::string("(planned)") : std::string(ignis_model_last_error());
    };
    const std::string bound = plan(nullptr);
    check(bound == "(planned)", "the load ABI plans a complete Flash-Next set: got \"" + bound + "\"");
    check(out.workspace_bytes > 0 && out.decode_graph_bytes > 0 && out.sampling_bytes > 0 &&
              out.activation_bytes > 0 && out.verify_round_bytes == 0 && out.drafter_round_bytes == 0 &&
              out.media_embedding_bytes == 0,
          "its plan names a workspace, a decode round, sampling and activations, and no 27B line");
    ignis_model_load_options lanes{};
    lanes.size = sizeof(lanes);
    lanes.decode_lanes = 9;
    const std::string wide = plan(&lanes);
    check(wide.find("decode_lanes 9") != std::string::npos,
          "nine decode lanes are refused by name: got \"" + wide + "\"");
    ignis_model *model = nullptr;
    const int32_t loaded = ignis_model_load(tensors.data(), tensors.size(), &topology, 8192, 131072,
                                            IGNIS_KV_FORMAT_HQ_E8_2B, nullptr, &model);
    const std::string unresident = ignis_model_last_error();
    check(loaded != 0 && model == nullptr && unresident.find("needs its expert residency") != std::string::npos,
          "a load without its residency is refused by name: got \"" + unresident + "\"");
    // GitHub #557: vision on a Flash-Next load plans its embedding pool, as the 27B's: floored at
    // the envelope's own output (8,192 columns of 2,560 BF16, 64 pages of 128 columns) and rounded
    // up to whole pages, in VRAM under the device residency and none of it under the host's; and
    // its activations grow by the forward's vision inputs alone (a chunk's three axes and media
    // rows, a round's rope deltas: 98,304 + 32,768 + 256 bytes). The attention readouts stay the
    // 27B's.
    check(plan(nullptr) == "(planned)", "the text load plans again");
    const uint64_t text_activations = out.activation_bytes;
    ignis_model_load_options vision{};
    vision.size = sizeof(vision);
    vision.vision_max_tokens = 8192;
    const std::string pooled = plan(&vision);
    check(pooled == "(planned)" && out.media_embedding_bytes == 41943040 &&
              out.activation_bytes == text_activations + 131328,
          "vision on a Flash-Next load plans the envelope's pool and the vision inputs: got \"" + pooled + "\", " +
              std::to_string(out.media_embedding_bytes) + " pool bytes, " +
              std::to_string(out.activation_bytes - text_activations) + " more activation bytes");
    vision.vision_embedding_pool_bytes = 50000000;
    const std::string asked = plan(&vision);
    check(asked == "(planned)" && out.media_embedding_bytes == 50462720,
          "a pool asked past the floor is rounded up to whole pages: got \"" + asked + "\", " +
              std::to_string(out.media_embedding_bytes));
    vision.vision_residency = IGNIS_VISION_RESIDENCY_HOST;
    const std::string hosted = plan(&vision);
    check(hosted == "(planned)" && out.media_embedding_bytes == 0 && out.activation_bytes == text_activations + 131328,
          "a host-resident pool leaves the VRAM plan: got \"" + hosted + "\", " +
              std::to_string(out.media_embedding_bytes));
    ignis_model_load_options readouts{};
    readouts.size = sizeof(readouts);
    readouts.attention_text_max_keys = 4096;
    const std::string refused = plan(&readouts);
    check(refused.find("Qwen3.8-Flash-Next has no attention readouts") != std::string::npos,
          "the attention readouts on a Flash-Next load are refused by name: got \"" + refused + "\"");
    // GitHub #335: the prompt attention's precision is the 27B's; Flash-Next's QSA has no band.
    ignis_model_load_options fp16{};
    fp16.size = sizeof(fp16);
    fp16.prefill_attention = IGNIS_PREFILL_ATTENTION_FP16;
    const std::string qsa = plan(&fp16);
    check(qsa.find("prefill_attention is the 27B's") != std::string::npos,
          "an FP16 prefill attention on a Flash-Next load is refused by name: got \"" + qsa + "\"");
    // GitHub #307 (spec flash-next/07): Flash-Next drafts with its MTP head; the 27B's drafter is
    // refused by name, and so is a row budget no verify round fits.
    ignis_model_load_options dflash2{};
    dflash2.size = sizeof(dflash2);
    dflash2.speculative_backend = IGNIS_SPECULATIVE_DFLASH2;
    dflash2.draft_tokens = 3;
    const std::string not_ours = plan(&dflash2);
    check(not_ours.find("DFlash2 is the 27B's drafter") != std::string::npos,
          "DFlash2 on a Flash-Next load is refused by name: got \"" + not_ours + "\"");
    ignis_model_load_options verify_only{};
    verify_only.size = sizeof(verify_only);
    verify_only.speculative_backend = IGNIS_SPECULATIVE_VERIFY_ONLY;
    verify_only.draft_tokens = 3;
    const std::string verified = plan(&verify_only);
    check(verified == "(planned)" && out.verify_round_bytes > 0,
          "a verify-only Flash-Next load plans its verify round's line: got \"" + verified + "\"");
    verify_only.draft_row_budget = 1;
    const std::string narrow = plan(&verify_only);
    check(narrow.find("row budget of 1") != std::string::npos,
          "a row budget that leaves one lane no draft is refused by name: got \"" + narrow + "\"");
    // GitHub #329 item 5: a budget past the default's 8 rows, up to verify::kMaxRows. Its windows
    // are crates/core's flash_next_window; 0 and 8 plan alike; a wider budget grows the verify
    // round's, the sampling and the decode round's lines, and only as far as the load's rounds
    // reach.
    {
      namespace verify = ignis::flash_next::verify;
      const auto widths = [](uint32_t draft, uint32_t budget) {
        std::vector<uint32_t> w;
        for (uint32_t lanes = 1; lanes <= 8; ++lanes) w.push_back(verify::window_for(draft, budget, lanes));
        return w;
      };
      check(widths(3, 16) == std::vector<uint32_t>{3, 3, 3, 3, 2, 1, 1, 1}, "3 drafts at 16 rows: every width drafts");
      check(widths(3, 24) == std::vector<uint32_t>{3, 3, 3, 3, 3, 3, 2, 2}, "3 drafts at 24 rows");
      check(widths(7, 32) == std::vector<uint32_t>{7, 7, 7, 7, 5, 4, 3, 3}, "7 drafts at 32 rows");
      check(widths(3, 0) == widths(3, 8) && verify::kMaxRows == 32, "0 is the default's 8 rows; the maximum is 32");
      const auto planned = [&](uint32_t lanes, uint32_t budget, ignis_model_reservations *lines) {
        ignis_model_load_options o = verify_only;
        o.decode_lanes = lanes;
        o.draft_row_budget = budget;
        const std::string why = plan(&o);
        *lines = out;
        return why;
      };
      const auto same = [](const ignis_model_reservations &a, const ignis_model_reservations &b) {
        return a.workspace_bytes == b.workspace_bytes && a.sampling_bytes == b.sampling_bytes &&
               a.decode_graph_bytes == b.decode_graph_bytes && a.activation_bytes == b.activation_bytes &&
               a.verify_round_bytes == b.verify_round_bytes;
      };
      ignis_model_reservations zero{}, eight{}, sixteen{}, twelve{}, maximum{};
      check(planned(4, 0, &zero) == "(planned)" && planned(4, 8, &eight) == "(planned)" && same(zero, eight),
            "a budget of 8 plans exactly what the default does");
      const std::string wide = planned(4, 16, &sixteen);
      check(wide == "(planned)", "a budget of 16 rows is planned: got \"" + wide + "\"");
      check(sixteen.verify_round_bytes > eight.verify_round_bytes && sixteen.sampling_bytes > eight.sampling_bytes &&
                sixteen.decode_graph_bytes > eight.decode_graph_bytes && sixteen.workspace_bytes == eight.workspace_bytes &&
                sixteen.activation_bytes == eight.activation_bytes,
            "16 rows grow the verify round's, the sampling and the decode round's lines, not the prefill's");
      std::printf("flash-next bind: 4 lanes x 3 drafts, budget 8 -> 16: verify %llu -> %llu, sampling %llu -> %llu, "
                  "decode graph %llu -> %llu bytes\n",
                  static_cast<unsigned long long>(eight.verify_round_bytes),
                  static_cast<unsigned long long>(sixteen.verify_round_bytes),
                  static_cast<unsigned long long>(eight.sampling_bytes),
                  static_cast<unsigned long long>(sixteen.sampling_bytes),
                  static_cast<unsigned long long>(eight.decode_graph_bytes),
                  static_cast<unsigned long long>(sixteen.decode_graph_bytes));
      // Three lanes of 3 drafts reach 12 rows: the maximum's budget reserves what 12 does.
      check(planned(3, 12, &twelve) == "(planned)" && planned(3, 32, &maximum) == "(planned)" && same(twelve, maximum),
            "a budget no round reaches reserves nothing past the rows the load's rounds take");
      ignis_model_reservations ignored{};
      const std::string past = planned(4, 33, &ignored);
      check(past.find("row budget of 33") != std::string::npos && past.find("2..32") != std::string::npos,
            "a budget past the maximum is refused by name: got \"" + past + "\"");

      // The default is main's (0a810c0b), not only 0 == 8 here: its windows, and its plan lines
      // as main's own build planned them (the same descriptors, 8192-token chunks, 131072
      // tokens), with no speculation, verify-only at budgets 0, 2, 6 and 8, both KV formats.
      check(widths(3, 8) == std::vector<uint32_t>{3, 3, 1, 1, 0, 0, 0, 0} &&
                widths(2, 0) == std::vector<uint32_t>{2, 2, 1, 1, 0, 0, 0, 0} &&
                widths(7, 0) == std::vector<uint32_t>{7, 3, 1, 1, 0, 0, 0, 0},
            "the default's windows are main's");
      struct MainPlan {
        int32_t format;
        uint32_t backend, drafts, lanes, budget;
        uint64_t decode_graph, verify_round, activation;
      };
      constexpr int32_t kHq = IGNIS_KV_FORMAT_HQ_E8_2B, kBf16 = IGNIS_KV_FORMAT_BF16;
      constexpr uint32_t kNone = IGNIS_SPECULATIVE_NONE, kVerify = IGNIS_SPECULATIVE_VERIFY_ONLY;
      const MainPlan mains[] = {
          {kHq, kNone, 0, 0, 0, 17772032, 0, 772235264},
          {kHq, kNone, 0, 8, 0, 40844288, 0, 779275264},
          {kHq, kVerify, 3, 1, 0, 23594752, 5996032, 773131264},
          {kHq, kVerify, 3, 2, 8, 40844288, 11988992, 779275264},
          {kHq, kVerify, 3, 4, 0, 40844288, 13222912, 779275264},
          {kHq, kVerify, 3, 4, 6, 32269312, 10534912, 775691264},
          {kHq, kVerify, 3, 5, 8, 40844288, 13840128, 779275264},
          {kHq, kVerify, 3, 8, 0, 40844288, 15691008, 779275264},
          {kHq, kVerify, 3, 8, 2, 40844288, 7233792, 779275264},
          {kHq, kVerify, 7, 1, 0, 40844288, 11793920, 779275264},
          {kHq, kVerify, 2, 0, 0, 32269312, 9600000, 775691264},
          {kBf16, kNone, 0, 0, 0, 17772032, 0, 772235264},
          {kBf16, kVerify, 3, 1, 8, 23594752, 5897472, 773131264},
          {kBf16, kVerify, 3, 4, 8, 40844288, 12829440, 779275264},
          {kBf16, kVerify, 3, 8, 0, 40844288, 14904064, 779275264},
      };
      for (const MainPlan &m : mains) {
        ignis_model_load_options o{};
        o.size = sizeof(o);
        o.speculative_backend = m.backend;
        o.draft_tokens = m.drafts;
        o.decode_lanes = m.lanes;
        o.draft_row_budget = m.budget;
        ignis_model_reservations r{};
        const int32_t rc = ignis_model_plan_reservations(tensors.data(), tensors.size(), &topology, 8192, 131072,
                                                         m.format, &o, &r);
        check(rc == 0 && r.workspace_bytes == 1016074496 && r.sampling_bytes == 4623344 &&
                  r.decode_graph_bytes == m.decode_graph && r.verify_round_bytes == m.verify_round &&
                  r.activation_bytes == m.activation && r.drafter_round_bytes == 0 && r.media_embedding_bytes == 0,
              "format " + std::to_string(m.format) + " backend " + std::to_string(m.backend) + ", " +
                  std::to_string(m.drafts) + " drafts, " + std::to_string(m.lanes) + " lanes, budget " +
                  std::to_string(m.budget) + " plans main's lines: got decode graph " +
                  std::to_string(r.decode_graph_bytes) + ", verify round " + std::to_string(r.verify_round_bytes) +
                  ", activations " + std::to_string(r.activation_bytes) + ", sampling " +
                  std::to_string(r.sampling_bytes) + ", workspace " + std::to_string(r.workspace_bytes));
      }

      // Eight lanes of 3 drafts at budgets 8, 16, 24 and 32 (8 lanes x 4 columns = 32 rows): the
      // three lines a wide round grows, plan figures (the finding's VRAM table).
      ignis_model_reservations previous{};
      for (const uint32_t budget : {8u, 16u, 24u, 32u}) {
        ignis_model_reservations lines{};
        check(planned(8, budget, &lines) == "(planned)", "8 lanes at a budget of " + std::to_string(budget));
        check(budget == 8 || (lines.verify_round_bytes > previous.verify_round_bytes &&
                              lines.sampling_bytes > previous.sampling_bytes &&
                              lines.decode_graph_bytes > previous.decode_graph_bytes &&
                              lines.workspace_bytes == previous.workspace_bytes &&
                              lines.activation_bytes == previous.activation_bytes),
              "8 lanes: budget " + std::to_string(budget) + " grows the verify, sampling and decode lines only");
        std::printf("flash-next bind: 8 lanes x 3 drafts, budget %u: verify %llu, sampling %llu, decode graph %llu "
                    "bytes\n",
                    budget, static_cast<unsigned long long>(lines.verify_round_bytes),
                    static_cast<unsigned long long>(lines.sampling_bytes),
                    static_cast<unsigned long long>(lines.decode_graph_bytes));
        previous = lines;
      }
    }
    tensors.pop_back();
    const std::string missing = plan(nullptr);
    check(missing.find("bind_flash_next: missing bound tensor") != std::string::npos,
          "a binder fault comes back through the load ABI: got \"" + missing + "\"");
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next bind test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next bind test: ok\n");
  return 0;
}
