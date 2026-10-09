// ignis kernel leaf -- the Flash-Next program (spec flash-next/04, GitHub #302, slice S1; OURS,
// ADR 0043). See program.h for what a load holds and flash_next_internal.h for the layer.
//
// One forward, for a prefill chunk and a decode round alike: embed the call's ids into every
// hyper-connection stream, then per decoder layer
//   the n-gram embedding (its layer only), the attention mix, GDN or (the indexer's selection,
//   then QSA attention), the inject; the MoE mix, this layer's router, the residency step's
//   demand half, the routed experts (the decode route for a round, the prefill route for a
//   chunk), the combine, the inject -- and beside them, the shared expert on a branch of its
//   own (forked after the mix, joined at the combine) and, on the branch residency forks, the
//   next layer's router on the same input (the lookahead residency ranks) and the step's
//   prefetch half;
// then the head (the final mixer and lm_head) on the rows that are drawn from. Since GitHub #306's
// decode fusion (fusion.h) each inject -- the MoE's with its combine -- is handed to the next mix,
// which on a one-lane decode round applies it inside its own first launch, and a round's selection
// is made inside residency's demand launch.
//
// A chunk is one lane of up to prefill_chunk_tokens tokens, run eagerly out of the handle's
// scratch arena, its frontiers advanced once the chunk's work is confirmed complete. A round is
// 1..decode_lanes lanes of one token, out of the decode arena, with everything that changes
// between rounds (ids, slots, positions, n-gram rows, sampling configs) staged at stable
// addresses first -- so the same op sequence replays from a graph captured per width.

#include "program.h"

#include "bind.h"
#include "embed_head.h"
#include "fusion.h"
#include "gdn.h"
#include "hc.h"
#include "indexer.h"
#include "qsa.h"
#include "mtp.h"
#include "qsa_sparse.h"
#include "verify.h"

#include "ignis_fn_residual_tap.h"
#include "ignis_fp8_linear.h"
#include "ignis_seq_internal.h"
#include "../moe_common.cuh"
#include "../permitted_tokens.h"
#include "../step_internal.h"

#include "ninfer/ops/rope.h"
#include "ninfer/ops/sampling.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace ignis::flash_next {

namespace {

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The rows of a span-logits block (the measurement readout): the head over this many rows at a
// time, their logits in the chunk's scratch, then copied out.
constexpr int32_t kSpanLogitRows = 32;

}  // namespace

// The KV positions a decode round's indexer and attention are sized for: the context, past the
// dense threshold so a round is never dense (flash_next_internal.h: a decode graph's selection is
// always written).
int32_t decode_max_visible(const FlashNextModel &fn) {
  return std::max<int32_t>(static_cast<int32_t>(fn.max_context_tokens), fn.g.dense_threshold() + 1);
}

namespace {

// The ops' peak in the arena over one call of `rows` rows, `tokens` per lane, at most
// `max_visible` visible positions. The ops run one after another, each in its own scope; only
// the selection a QSA layer's indexer writes and its attention reads is held across two.
std::size_t call_scratch_bytes(const Geometry &g, int32_t kv_format, int32_t rows, int32_t tokens,
                               int32_t max_visible, bool span_head) {
  const std::size_t selection =
      aligned(static_cast<std::size_t>(rows) * g.selection_width() * sizeof(int32_t)) +
      aligned(static_cast<std::size_t>(rows) * sizeof(int32_t));
  std::size_t attention = fn_qsa_attention_scratch_bytes(g, rows);
  if (kv_format == IGNIS_KV_FORMAT_HQ_E8_2B && tokens > 1) {
    // A sparse prefill call decodes every visible row of the lane (S3's plan line).
    attention += 2 * aligned(sparse::visible_hq_bytes(g, max_visible));
  }
  const std::size_t qsa = selection + std::max(fn_indexer_select_scratch_bytes(g, rows, max_visible), attention);
  const int32_t head_rows = span_head ? kSpanLogitRows : 1;
  const std::size_t head =
      aligned(static_cast<std::size_t>(head_rows) * g.vocab * sizeof(__nv_bfloat16)) + fn_head_scratch_bytes(g, head_rows);
  const std::size_t peak = std::max({fn_hc_mix_scratch_bytes(g, rows), fn_gdn_layer_scratch_bytes(g, rows), qsa,
                                     fn_ngram_add_scratch_bytes(g, rows), head});
  // Each scope's alignment.
  return peak + 4096;
}

int32_t attention_layers(const FlashNextModel &fn) {
  int32_t n = 0;
  for (const LayerWeights &layer : fn.weights->layers) n += layer.attention ? 1 : 0;
  return n;
}

int32_t gdn_layers(const FlashNextModel &fn) {
  return static_cast<int32_t>(fn.weights->layers.size()) - attention_layers(fn);
}

// The pool's attention sections: the trunk's, and an MTP load's head one past them.
int32_t attention_sections(const FlashNextModel &fn) { return attention_layers(fn) + (fn.mtp != nullptr ? 1 : 0); }

Sizes plan_sizes(const FlashNextModel &fn) {
  const Geometry &g = fn.g;
  Sizes s;
  const auto chunk = static_cast<int32_t>(fn.prefill_chunk_tokens);
  const auto lanes = static_cast<int32_t>(fn.decode_lanes);
  // A decode call's rows: a round's lanes, or a verify round's columns (whose hq rows are the
  // listed ones, as a round's: sized as one-token rows).
  const auto decode_rows = static_cast<int32_t>(fn.decode_rows);
  s.prefill_scratch = call_scratch_bytes(g, fn.kv_format, chunk, chunk, static_cast<int32_t>(fn.max_context_tokens),
                                         /*span_head=*/true);
  s.decode_scratch = call_scratch_bytes(g, fn.kv_format, decode_rows, 1, decode_max_visible(fn), /*span_head=*/false) +
                     fn_head_scratch_bytes(g, decode_rows);
  const auto rows = static_cast<std::size_t>(fn.rows());
  s.activations = aligned(rows * g.residual_width() * 2) + 2 * aligned(rows * g.hidden * 2) +
                  aligned(2 * rows * g.streams * sizeof(float)) + aligned(rows * sizeof(int32_t)) +
                  2 * aligned(static_cast<std::size_t>(std::max(lanes, 1)) * sizeof(int32_t)) +
                  aligned(rows * g.ngram_heads * g.ngram_row_bytes());
  const std::size_t router = aligned(rows * g.experts_per_token * sizeof(int32_t)) +
                             aligned(rows * g.experts_per_token * sizeof(float)) +
                             aligned(rows * g.experts * sizeof(float));
  s.moe = aligned(ignis_moe_workspace_bytes(fn.decode_rows, fn.prefill_chunk_tokens)) +
          aligned(rows * g.hidden * sizeof(int64_t)) + 2 * router +
          aligned(rows * g.shared_intermediate * sizeof(__nv_bfloat16)) + aligned(rows * g.hidden * sizeof(float));
  s.sampling_logits = static_cast<std::size_t>(g.vocab) * IGNIS_DECODE_MAX_BATCH * sizeof(std::uint16_t);
  s.sampling_workspace =
      std::max<std::size_t>(ninfer::ops::sampling_workspace_capacity_bytes(g.vocab, 1, IGNIS_DECODE_MAX_BATCH), 256);
  if (fn.speculative_backend != IGNIS_SPECULATIVE_NONE) {
    s.verify = verify::plan(g, fn.kv_format, fn.decode_lanes, fn.draft_tokens, fn.draft_row_budget,
                            attention_sections(fn), gdn_layers(fn), fn.mtp != nullptr)
                   .total();
  }
  if (fn.mtp != nullptr) {
    // The head's entries' tokens and its expert slot table; its combine beside the layer's ops.
    s.activations += aligned(rows * sizeof(int32_t)) +
                     aligned(static_cast<std::size_t>(g.experts) * 2 * sizeof(ignis_moe_slot));
    s.prefill_scratch = std::max(s.prefill_scratch, mtp::combine_scratch_bytes(g, chunk) + 4096);
    s.decode_scratch = std::max(s.decode_scratch, mtp::combine_scratch_bytes(g, decode_rows) + 4096);
  }
  return s;
}

// The step ABI's sampling staging, as model.cu sizes it for the 27B (the line `reservations`
// reports below).
std::size_t sampling_staging_bytes(const Sizes &s) {
  const std::size_t lanes = IGNIS_DECODE_MAX_BATCH;
  return sizeof(ninfer::ops::SamplingConfig) + 2 * sizeof(int32_t) + sizeof(ninfer::ops::SamplingConfig) * lanes +
         2 * sizeof(int32_t) * lanes + sizeof(int32_t) * lanes * IGNIS_MAX_PERMITTED_TOKENS +
         sizeof(int32_t) * lanes + sizeof(float) * lanes + s.sampling_logits + s.sampling_workspace;
}

// A decode round's inputs in FlashNextModel::round_inputs (GitHub #306, step 7): byte offsets of
// its sampling configs, positions, ids, slots, n-gram rows, permitted sets and their counts, each
// 64-byte aligned, at the load's lane count.
struct RoundInputs {
  std::size_t configs = 0, positions = 0, ids = 0, slots = 0, ngram = 0, permitted = 0, counts = 0, total = 0;
};

RoundInputs round_inputs_layout(const FlashNextModel &fn) {
  const auto lanes = static_cast<std::size_t>(std::max<uint32_t>(fn.decode_lanes, 1));
  RoundInputs at;
  std::size_t end = 0;
  const auto take = [&](std::size_t bytes) {
    const std::size_t here = end;
    end += (bytes + 63) / 64 * 64;
    return here;
  };
  at.configs = take(lanes * sizeof(ninfer::ops::SamplingConfig));
  at.positions = take(lanes * sizeof(int32_t));
  at.ids = take(lanes * sizeof(int32_t));
  at.slots = take(lanes * sizeof(int32_t));
  at.ngram = take(lanes * static_cast<std::size_t>(fn.g.ngram_heads) * fn.g.ngram_row_bytes());
  at.permitted = take(lanes * IGNIS_MAX_PERMITTED_TOKENS * sizeof(int32_t));
  at.counts = take(lanes * sizeof(int32_t));
  at.total = end;
  return at;
}

// A refusal of a geometry this program does not run, or nullptr.
std::string geometry_refusal(const Geometry &g) {
  if (const char *why = gdn::check_geometry(g)) return why;
  if (const char *why = qsa::check_geometry(g)) return why;
  if (const char *why = sparse::check_geometry(g)) return why;
  if (const char *why = indexer::check_geometry(g)) return why;
  if (g.hidden != IGNIS_MOE_HIDDEN || g.experts != IGNIS_MOE_EXPERTS || g.experts_per_token != IGNIS_MOE_TOP_K ||
      g.expert_intermediate != IGNIS_MOE_INTERMEDIATE) {
    return "the MoE ops are written for 512 experts, top-10, of 2560 -> 640 -> 2560";
  }
  if (g.streams <= 0 || g.hc_rank <= 0) return "a hyper-connection geometry of no streams";
  if (g.ngram_layer < 0 || g.ngram_layer >= g.layers) return "the n-gram embedding's layer is not a layer";
  return {};
}

std::string pool_refusal(const FlashNextModel &fn, const ignis_seq_pool &pool) {
  const Geometry &g = fn.g;
  const int32_t attention = attention_sections(fn);
  if (pool.kv_format != fn.kv_format) return "the pool's KV format is not this load's";
  if (pool.kv_num_layers != attention || pool.kv_num_kv_heads != g.kv_heads || pool.kv_head_dim != g.head_dim) {
    return "the pool's KV planes are not this model's attention layers";
  }
  if (!pool.has_indexer() || pool.indexer_key_dim != g.indexer_kv_heads * g.indexer_head_dim ||
      pool.indexer_compress_tokens != g.compress_ratio) {
    return "the pool has no indexer section of this model's keys";
  }
  if (!pool.has_ngram_conv() ||
      pool.ngram_conv_slot_bytes != static_cast<std::uint64_t>(g.ngram_conv_state_columns()) * g.residual_width() * 2) {
    return "the pool has no n-gram conv state of this model's";
  }
  if (pool.vocab != g.vocab) return "the pool's penalty rows are not this model's vocab";
  return {};
}

}  // namespace

Views views_of(const FlashNextModel &fn, ignis_seq_pool *pool) {
  Views v;
  for (int32_t layer = 0; layer < pool->kv_num_layers; ++layer) {
    IndexerLayerState state;
    state.block_keys = pool->indexer_block_keys(layer);
    state.blocks_per_page = static_cast<int32_t>(ninfer::kPagedKVPageSize) / pool->indexer_compress_tokens;
    state.tail_keys = pool->indexer_tail_keys(layer);
    v.indexer.push_back(state);
  }
  v.ctx.g = fn.g;
  v.ctx.kv_format = fn.kv_format;
  v.ctx.rope = fn.rope;
  v.ctx.pool = pool;
  v.ctx.indexer = v.indexer.data();
  v.ctx.ngram.conv_columns = pool->ngram_conv->p;
  return v;
}

namespace {

bool check_cuda(cudaError_t err, const std::string &what, std::string *error) {
  if (err == cudaSuccess) return true;
  *error = what + " failed: " + cudaGetErrorString(err);
  return false;
}

}  // namespace

// The call's whole forward to the final residual: embed, every layer. `phase` is residency's.
int32_t forward(FlashNextModel &fn, const Context &ctx, const Batch &batch, uint32_t phase,
                ninfer::DeviceArena &scratch, cudaStream_t stream, std::string *error) {
  const Geometry &g = fn.g;
  const Weights &w = *fn.weights;
  const int32_t rows = batch.rows();
  auto *residual = fn.residual->p;
  auto *x = fn.x->p;
  auto *y = fn.y->p;
  auto *acc = static_cast<int64_t *>(fn.moe_acc->p);
  const auto fail = [&](const std::string &where, const char *detail) {
    *error = where + ": " + detail;
    return -1;
  };
  // GitHub #306, step 2: a sublayer's inject -- the MoE's with its combine -- is left pending and
  // applied by the next mix (fn_hc_mix_after: inside its down launch on the fused decode route,
  // else on its own first, as it always ran); the n-gram add and the end of the forward, which
  // read the residual, flush it first. A mix's injection weights alternate between two buffers, so
  // no mix overwrites the weights its pending inject reads. A pending combine leaves the routed
  // accumulator to be zeroed by it: a failure before it ran zeroes it (ignis_moe.h's contract).
  PendingInject pending;
  float *const injections[2] = {static_cast<float *>(fn.injections->p),
                                static_cast<float *>(fn.injections->p) + static_cast<std::size_t>(fn.rows()) * g.streams};
  int32_t next_injections = 0;
  const auto fail_pending = [&](const std::string &where, const char *detail) {
    if (pending.acc != nullptr) {
      (void)cudaMemsetAsync(acc, 0, static_cast<std::size_t>(rows) * g.hidden * sizeof(int64_t), stream);
    }
    return fail(where, detail);
  };

  if (fn_embed(g, w.embed, static_cast<const int32_t *>(fn.token_ids->p), rows, residual, stream) != 0) {
    return fail("the embedding", fn_last_error());
  }
  int32_t attention_ordinal = 0;
  int32_t gdn_ordinal = 0;
  for (int32_t l = 0; l < g.layers; ++l) {
    const LayerWeights &layer = w.layers[static_cast<std::size_t>(l)];
    const std::string where = "layer " + std::to_string(l);
    if (l == g.ngram_layer) {
      if (fn_hc_flush(g, pending, residual, rows, stream) != 0) {
        return fail_pending(where + " pending inject", fn_last_error());
      }
      pending = PendingInject{};
      if (fn_ngram_add(ctx, w.ngram, batch, fn.ngram_rows->p, residual, scratch, stream) != 0) {
        return fail(where + " n-gram embedding", fn_last_error());
      }
    }
    // The attention sublayer.
    float *inj = injections[next_injections];
    next_injections ^= 1;
    if (fn_hc_mix_after(g, layer.attn_hc, pending, residual, rows, x, inj, scratch, stream) != 0) {
      return fail_pending(where + " attention mix", fn_last_error());
    }
    pending = PendingInject{};
    if (layer.attention) {
      auto scope = scratch.scope();
      Selection selection;
      selection.tokens = static_cast<int32_t *>(
          scratch.alloc_bytes(static_cast<std::size_t>(rows) * g.selection_width() * sizeof(int32_t)).data);
      selection.counts =
          static_cast<int32_t *>(scratch.alloc_bytes(static_cast<std::size_t>(rows) * sizeof(int32_t)).data);
      if (fn_indexer_select(ctx, attention_ordinal, layer.qsa.indexer, batch, x, selection, scratch, stream) != 0) {
        return fail(where + " indexer", fn_last_error());
      }
      if (fn_qsa_attention(ctx, attention_ordinal, layer.qsa, batch, x, selection, y, scratch, stream) != 0) {
        return fail(where + " attention", fn_last_error());
      }
      ++attention_ordinal;
    } else {
      if (fn_gdn_layer(ctx, gdn_ordinal, layer.gdn, batch, x, y, scratch, stream) != 0) {
        return fail(where + " GDN", fn_last_error());
      }
      ++gdn_ordinal;
    }
    pending.y = y;
    pending.inj = inj;

    // The MoE sublayer.
    float *mlp_inj = injections[next_injections];
    next_injections ^= 1;
    if (fn_hc_mix_after(g, layer.mlp_hc, pending, residual, rows, x, mlp_inj, scratch, stream) != 0) {
      return fail(where + " MoE mix", fn_last_error());
    }
    pending = PendingInject{};
    // The shared expert reads only `x`: it runs on a branch of its own (side_branch.h) beside the
    // router, residency and the routed experts, joined before the combine. Every failure after
    // the fork joins that branch and residency's lookahead branch first, so no return leaves
    // either open (a capture would then fail to end) or `x` still being read.
    auto *shared = static_cast<float *>(fn.shared_out->p);
    std::string branch_error;
    const int32_t shared_rc = run_beside(
        fn.shared_branch, stream,
        [&](cudaStream_t side) {
          return ignis_moe_shared_expert(layer.moe.shared_gate, layer.moe.shared_up, layer.moe.shared_down, x,
                                         static_cast<uint32_t>(rows), fn.shared_h->p, shared, side);
        },
        &branch_error);
    const auto fail_joined = [&](const std::string &what, const char *detail) {
      (void)join_side(fn.shared_branch, stream);
      (void)ignis_residency_join(fn.residency, stream);
      return fail(what, detail);
    };
    if (shared_rc != 0) {
      return fail_joined(where + " shared expert", branch_error.empty() ? ignis_moe_last_error() : branch_error.c_str());
    }
    auto *ids = static_cast<int32_t *>(fn.router_ids->p);
    auto *weights = static_cast<float *>(fn.router_weights->p);
    auto *logits = static_cast<float *>(fn.router_logits->p);
    // GitHub #306, step 5: a decode round's selection is made inside residency's demand launch
    // (the same ids and weights), and the lookahead runs the router's logits only (its selection
    // was never read: the prefetch ranks the logits).
    const bool routed = fused(Fusion::Route);
    const bool select_in_step = routed && phase == IGNIS_RESIDENCY_DECODE;
    if (select_in_step ? ignis_moe_router_logits(x, static_cast<uint32_t>(rows), layer.moe.router, logits, stream) != 0
                       : ignis_moe_router(x, static_cast<uint32_t>(rows), layer.moe.router, ids, weights, logits,
                                          stream) != 0) {
      return fail_joined(where + " router", ignis_moe_last_error());
    }
    // The step's demand half on the layer's stream; its lookahead -- the next layer's router on
    // this input, which only the prefetch reads -- on the branch it forks, beside the experts.
    void *branch = nullptr;
    void **lookahead = l + 1 < g.layers ? &branch : nullptr;
    const auto layer_index = static_cast<uint32_t>(l);
    if ((select_in_step ? ignis_residency_step_demand_routed(fn.residency, layer_index, phase, logits, ids, weights,
                                                             static_cast<uint32_t>(rows), stream, lookahead)
                        : ignis_residency_step_demand(fn.residency, layer_index, phase, ids,
                                                      static_cast<uint32_t>(rows), stream, lookahead)) != 0) {
      return fail_joined(where + " expert residency", ignis_residency_last_error());
    }
    if (branch != nullptr) {
      const void *next_router = w.layers[static_cast<std::size_t>(l + 1)].moe.router;
      auto *lookahead_logits = static_cast<float *>(fn.lookahead_logits->p);
      if ((routed ? ignis_moe_router_logits(x, static_cast<uint32_t>(rows), next_router, lookahead_logits, branch)
                  : ignis_moe_router(x, static_cast<uint32_t>(rows), next_router,
                                     static_cast<int32_t *>(fn.lookahead_ids->p),
                                     static_cast<float *>(fn.lookahead_weights->p), lookahead_logits, branch)) != 0) {
        return fail_joined(where + " lookahead router", ignis_moe_last_error());
      }
      if (cudaEventRecord(fn.lookahead_read, static_cast<cudaStream_t>(branch)) != cudaSuccess) {
        return fail_joined(where + " lookahead event", cudaGetErrorString(cudaGetLastError()));
      }
      if (ignis_residency_step_prefetch(fn.residency, static_cast<const float *>(fn.lookahead_logits->p),
                                        static_cast<uint32_t>(rows)) != 0) {
        return fail_joined(where + " expert prefetch", ignis_residency_last_error());
      }
    }
    const ignis_moe_slot *slots = ignis_residency_slot_table(fn.residency, static_cast<uint32_t>(l));
    ignis_moe_workspace workspace{fn.moe_workspace->p, fn.decode_rows, fn.prefill_chunk_tokens};
    const int32_t experts =
        phase == IGNIS_RESIDENCY_DECODE
            ? ignis_moe_experts_decode(x, static_cast<uint32_t>(rows), ids, weights, slots, &workspace, acc, stream)
            : ignis_moe_experts_prefill(x, static_cast<uint32_t>(rows), ids, weights, slots, &workspace, acc, stream);
    // The routed accumulator must be zero on every op's entry (ignis_moe.h); a failure before
    // the combine re-zeroes what the experts added, or the next call would add into it.
    const auto fail_moe = [&](const std::string &what, const std::string &detail) {
      (void)cudaMemsetAsync(acc, 0, static_cast<std::size_t>(rows) * g.hidden * sizeof(int64_t), stream);
      return fail_joined(where + what, detail.c_str());
    };
    if (experts != 0) {
      return fail_moe(" experts", ignis_moe_last_error());
    }
    if (!join_side(fn.shared_branch, stream)) {
      return fail_moe(" shared expert join", cudaGetErrorString(cudaGetLastError()));
    }
    // The combine and the inject, pending for the next mix.
    pending.y = y;
    pending.inj = mlp_inj;
    pending.acc = acc;
    pending.shared = shared;
    pending.x = x;
    pending.w_gate = layer.moe.shared_expert_gate;
    // The next sublayer's mix rewrites `x`: the lookahead router must have read it.
    if (branch != nullptr && cudaStreamWaitEvent(stream, fn.lookahead_read, 0) != cudaSuccess) {
      return fail_pending(where + " lookahead join", cudaGetErrorString(cudaGetLastError()));
    }
  }
  if (fn_hc_flush(g, pending, residual, rows, stream) != 0) {
    return fail_pending("the last layer's MoE inject", fn_last_error());
  }
  return 0;
}

// Every per-layer frontier of the sequence moves with its program frontier once a call's device
// work is confirmed complete. All of them, the arrays' entries past Flash-Next's 12 attention and
// 36 GDN layers included: a layer the model does not have is trivially caught up, and the chunk
// boundary a snapshot or a clone checks (ignis_seq_at_chunk_boundary) reads every entry.
void advance_frontiers(ignis_seq *seq, uint32_t tokens) {
  seq->position += tokens;
  for (auto &frontier : seq->gqa_positions) frontier += tokens;
  for (auto &frontier : seq->gdn_positions) frontier += tokens;
}

int32_t mtp_entries(FlashNextModel &fn, const Context &ctx, const Batch &batch, const int32_t *tokens,
                    ninfer::DeviceArena &scratch, cudaStream_t stream, std::string *error) {
  mtp::Buffers b;
  b.residual = fn.residual->p;
  b.x = fn.x->p;
  b.y = fn.y->p;
  b.inj = static_cast<float *>(fn.injections->p);
  b.workspace = ignis_moe_workspace{fn.moe_workspace->p, fn.decode_rows, fn.prefill_chunk_tokens};
  b.acc = static_cast<int64_t *>(fn.moe_acc->p);
  b.router_ids = static_cast<int32_t *>(fn.router_ids->p);
  b.router_weights = static_cast<float *>(fn.router_weights->p);
  b.router_logits = static_cast<float *>(fn.router_logits->p);
  b.shared_h = fn.shared_h->p;
  b.shared_out = static_cast<float *>(fn.shared_out->p);
  if (mtp::combine(fn.g, *fn.mtp, fn.weights->embed, tokens, batch, b, scratch, stream) != 0 ||
      mtp::layer(ctx, attention_layers(fn), *fn.mtp, static_cast<const ignis_moe_slot *>(fn.mtp_slots->p), batch, b,
                 scratch, stream) != 0) {
    *error = std::string("the MTP head: ") + fn_last_error();
    return -1;
  }
  return 0;
}

namespace {

float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

// What a Flash-Next prefill's options may ask for: the chunked route at the engine's policy, the
// span's n-gram rows, its permitted draw's probability and the span-logits readout -- nothing
// multimodal, no attention readout.
std::string prefill_options_refusal(const ignis_prefill_options *options, bool &span_logits_ok) {
  if (options == nullptr) return "a Flash-Next prefill needs its options: the span's n-gram rows";
  if (options->size != sizeof(ignis_prefill_options)) {
    return "unrecognized ignis_prefill_options size " + std::to_string(options->size);
  }
  if (options->route != IGNIS_PREFILL_ROUTE_CHUNKED) return "Flash-Next runs the chunked route only";
  if (options->compute_policy != IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT) {
    return "Flash-Next has no NVFP4 projection for a compute policy to choose";
  }
  if (options->mrope_positions != nullptr || options->media != nullptr || options->media_column_count != 0 ||
      options->rope_delta != 0) {
    return "Qwen3.8-Flash-Next serves no images";
  }
  if ((options->attention_gqa_ordinal >= 0 && options->out_attention_scores != nullptr) ||
      options->attention_set_count != 0 || options->out_attention_set_rows != nullptr) {
    return "Qwen3.8-Flash-Next serves no attention readouts";
  }
  if (options->ngram_rows == nullptr) return "a Flash-Next prefill needs the span's n-gram rows";
  span_logits_ok = true;
  return {};
}

}  // namespace

uint32_t max_decode_lanes() {
  return static_cast<uint32_t>(std::min({IGNIS_DECODE_MAX_BATCH, IGNIS_MOE_DECODE_MAX_TOKENS, gdn::kMaxLanes,
                                         qsa::kMaxDecodeLanes}));
}

int32_t FlashNextModel::rows() const {
  return static_cast<int32_t>(std::max(prefill_chunk_tokens, decode_rows));
}

FlashNextModel::FlashNextModel() = default;
FlashNextModel::~FlashNextModel() {
  if (lookahead_read != nullptr) cudaEventDestroy(lookahead_read);
  shared_branch.destroy();
}

std::unique_ptr<FlashNextModel> bind_model(const ignis_bound_tensor *tensors, uint64_t count,
                                           const ignis_topology &topology, uint32_t prefill_chunk_tokens,
                                           uint32_t max_context_tokens, int32_t kv_format,
                                           const ignis_model_load_options *options, std::string *error) {
  auto fn = std::make_unique<FlashNextModel>();
  fn->g = Geometry::from(topology);
  fn->kv_format = kv_format;
  fn->prefill_chunk_tokens = prefill_chunk_tokens;
  fn->max_context_tokens = max_context_tokens;
  fn->decode_lanes = options != nullptr && options->decode_lanes != 0 ? options->decode_lanes : kDefaultDecodeLanes;
  fn->residency = options != nullptr ? options->residency : nullptr;
  if (options != nullptr) {
    fn->speculative_backend = options->speculative_backend;
    fn->draft_tokens = options->draft_tokens;
    fn->draft_row_budget = options->draft_row_budget;
  }
  fn->decode_rows = fn->decode_lanes;
  if (fn->decode_lanes > max_decode_lanes()) {
    *error = "decode_lanes " + std::to_string(fn->decode_lanes) + " is more than the " +
             std::to_string(max_decode_lanes()) + " a Flash-Next round serves";
    return nullptr;
  }
  if (options != nullptr && options->rope_scaling_factor != 0.0F && options->rope_scaling_factor != 1.0F) {
    *error = "Qwen3.8-Flash-Next rotates at its own linear table, not a scaled one";
    return nullptr;
  }
  // The binder first: a set of descriptors is right or wrong whatever the
  // geometry, and a caller learns which before learning whether this
  // program runs the geometry. An MTP load's head tensors (`mtp.*`) bind on
  // their own; on any other load they are the trunk binder's extras.
  const bool mtp = fn->speculative_backend == IGNIS_SPECULATIVE_MTP;
  std::vector<ignis_bound_tensor> trunk, head;
  for (uint64_t i = 0; i < count; ++i) {
    const bool is_head = mtp && tensors[i].name != nullptr && std::strncmp(tensors[i].name, "mtp.", 4) == 0;
    (is_head ? head : trunk).push_back(tensors[i]);
  }
  fn->weights = bind_flash_next(trunk.data(), trunk.size(), topology, error);
  if (fn->weights == nullptr) {
    return nullptr;
  }
  if (mtp) {
    fn->mtp = bind_mtp(head.data(), head.size(), topology, error);
    if (fn->mtp == nullptr) {
      *error = "the MTP head: " + *error;
      return nullptr;
    }
    if (options->mtp_expert_slots != nullptr) {
      const auto entries = static_cast<std::size_t>(Geometry::from(topology).experts) * 2;
      fn->mtp_slot_table.assign(options->mtp_expert_slots, options->mtp_expert_slots + entries);
    }
  } else if (options != nullptr && options->mtp_expert_slots != nullptr) {
    *error = "mtp_expert_slots without the MTP backend";
    return nullptr;
  }
  if (const std::string why = geometry_refusal(fn->g); !why.empty()) {
    *error = "Qwen3.8-Flash-Next's program does not run this geometry (its weights bind): " + why;
    return nullptr;
  }
  // Spec flash-next/07: the verify round, with a test's drafts (VERIFY_ONLY) or the MTP head's.
  if (fn->speculative_backend != IGNIS_SPECULATIVE_NONE) {
    if (const std::string why = verify::refusal(fn->g, fn->decode_lanes, fn->draft_tokens, fn->draft_row_budget,
                                                attention_sections(*fn), gdn_layers(*fn));
        !why.empty()) {
      *error = "Qwen3.8-Flash-Next's verify round: " + why;
      return nullptr;
    }
    fn->decode_rows = verify::decode_rows(fn->decode_lanes, fn->draft_tokens, fn->draft_row_budget);
  }
  try {
    fn->rope = ninfer::ops::rope_linear_frequencies(static_cast<float>(topology.rope_theta),
                                                    static_cast<int>(topology.rotary_dim));
    fn->sizes = plan_sizes(*fn);
  } catch (const std::exception &e) {
    *error = std::string("Qwen3.8-Flash-Next's plan: ") + e.what();
    return nullptr;
  }
  return fn;
}

ignis_model_reservations reservations(const FlashNextModel &fn) {
  ignis_model_reservations out{};
  out.workspace_bytes = fn.sizes.prefill_scratch;
  out.sampling_bytes = sampling_staging_bytes(fn.sizes);
  out.decode_graph_bytes = fn.sizes.decode_scratch;
  out.activation_bytes = fn.sizes.activations + fn.sizes.moe;
  out.verify_round_bytes = fn.sizes.verify;
  return out;
}

ignis_model_reservations reserved(const ignis_model &model) {
  const auto bytes = [](const std::unique_ptr<ninfer::DeviceBuffer> &buffer) -> uint64_t {
    return buffer != nullptr ? buffer->bytes : 0;
  };
  const auto capacity = [](const std::unique_ptr<ninfer::DeviceArena> &arena) -> uint64_t {
    return arena != nullptr ? arena->capacity() : 0;
  };
  const FlashNextModel &fn = *model.flash_next;
  ignis_model_reservations out{};
  out.workspace_bytes = capacity(model.scratch);
  out.sampling_bytes = bytes(model.sampling_single_configs) + bytes(model.sampling_single_positions) +
                       bytes(model.sampling_single_out) + bytes(model.sampling_decode_configs) +
                       bytes(model.sampling_decode_positions) + bytes(model.sampling_decode_out) +
                       bytes(model.sampling_decode_logits) + bytes(model.sampling_decode_permitted) +
                       bytes(model.sampling_decode_permitted_counts) + bytes(model.sampling_decode_permitted_probs) +
                       capacity(model.sampling_workspace);
  out.decode_graph_bytes =
      capacity(model.decode_graph_scratch) + bytes(model.decode_graph_token_ids) + bytes(model.decode_graph_slots);
  for (const auto *buffer : {&fn.residual, &fn.x, &fn.y, &fn.injections, &fn.token_ids, &fn.slots, &fn.positions,
                             &fn.ngram_rows, &fn.moe_workspace, &fn.moe_acc, &fn.router_ids, &fn.router_weights,
                             &fn.router_logits, &fn.lookahead_ids, &fn.lookahead_weights, &fn.lookahead_logits,
                             &fn.shared_h, &fn.shared_out, &fn.mtp_tokens, &fn.mtp_slots}) {
    out.activation_bytes += bytes(*buffer);
  }
  out.verify_round_bytes = fn.verify != nullptr ? fn.verify->device_bytes() : 0;
  return out;
}

int32_t finish_load(ignis_model &model, std::string *error) {
  FlashNextModel &fn = *model.flash_next;
  const Geometry &g = fn.g;
  if (fn.residency == nullptr) {
    *error = "a Flash-Next load needs its expert residency (ignis_model_load_options.residency)";
    return -1;
  }
  model.hidden = static_cast<uint64_t>(g.hidden);
  model.vocab = static_cast<uint64_t>(g.vocab);
  model.rms_norm_eps = g.rms_norm_eps;
  model.prefill_chunk_tokens = fn.prefill_chunk_tokens;
  model.max_context_tokens = fn.max_context_tokens;
  model.kv_format = fn.kv_format;

  // The MoE ops and the FP8 linear configure the device once, here, outside any capture.
  if (ignis_moe_prepare() != 0) {
    *error = std::string("ignis_moe_prepare: ") + ignis_moe_last_error();
    return -1;
  }
  if (!check_cuda(cudaStreamCreate(&model.stream), "cudaStreamCreate", error)) {
    model.stream = nullptr;
    return -1;
  }
  if (!check_cuda(cudaEventCreateWithFlags(&fn.lookahead_read, cudaEventDisableTiming), "the lookahead event",
                  error)) {
    fn.lookahead_read = nullptr;
    return -1;
  }
  if (!fn.shared_branch.create(error)) {
    return -1;
  }
  try {
    const Sizes &s = fn.sizes;
    const auto rows = static_cast<std::size_t>(fn.rows());
    const auto lanes = static_cast<std::size_t>(std::max<uint32_t>(fn.decode_lanes, 1));
    const auto buffer = [](std::size_t bytes) { return std::make_unique<ninfer::DeviceBuffer>(aligned(bytes)); };

    model.scratch = std::make_unique<ninfer::DeviceArena>(s.prefill_scratch);
    model.prefill_scratch_bytes = s.prefill_scratch;
    model.decode_graph_scratch = std::make_unique<ninfer::DeviceArena>(s.decode_scratch);

    // The sampling staging, as model.cu reserves it for the 27B.
    model.sampling_single_configs = std::make_unique<ninfer::DeviceBuffer>(sizeof(ninfer::ops::SamplingConfig));
    model.sampling_single_positions = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t));
    model.sampling_single_out = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t));
    model.sampling_decode_configs =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(ninfer::ops::SamplingConfig) * IGNIS_DECODE_MAX_BATCH);
    model.sampling_decode_positions = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model.sampling_decode_out = std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model.sampling_decode_logits = std::make_unique<ninfer::DeviceBuffer>(s.sampling_logits);
    model.sampling_decode_permitted = std::make_unique<ninfer::DeviceBuffer>(
        sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH * IGNIS_MAX_PERMITTED_TOKENS);
    model.sampling_decode_permitted_counts =
        std::make_unique<ninfer::DeviceBuffer>(sizeof(int32_t) * IGNIS_DECODE_MAX_BATCH);
    model.sampling_decode_permitted_probs = std::make_unique<ninfer::DeviceBuffer>(sizeof(float) * IGNIS_DECODE_MAX_BATCH);
    model.sampling_workspace = std::make_unique<ninfer::DeviceArena>(s.sampling_workspace);

    fn.residual = buffer(rows * g.residual_width() * 2);
    fn.x = buffer(rows * g.hidden * 2);
    fn.y = buffer(rows * g.hidden * 2);
    fn.injections = buffer(2 * rows * g.streams * sizeof(float));  // two mixes' (forward)
    fn.token_ids = buffer(rows * sizeof(int32_t));
    fn.slots = buffer(lanes * sizeof(int32_t));
    fn.positions = buffer(lanes * sizeof(int32_t));
    fn.ngram_rows = buffer(rows * g.ngram_heads * g.ngram_row_bytes());
    fn.moe_workspace = buffer(ignis_moe_workspace_bytes(fn.decode_rows, fn.prefill_chunk_tokens));
    fn.moe_acc = buffer(rows * g.hidden * sizeof(int64_t));
    fn.router_ids = buffer(rows * g.experts_per_token * sizeof(int32_t));
    fn.router_weights = buffer(rows * g.experts_per_token * sizeof(float));
    fn.router_logits = buffer(rows * g.experts * sizeof(float));
    fn.lookahead_ids = buffer(rows * g.experts_per_token * sizeof(int32_t));
    fn.lookahead_weights = buffer(rows * g.experts_per_token * sizeof(float));
    fn.lookahead_logits = buffer(rows * g.experts * sizeof(float));
    fn.shared_h = buffer(rows * g.shared_intermediate * sizeof(__nv_bfloat16));
    fn.shared_out = buffer(rows * g.hidden * sizeof(float));
    fn.round_inputs = std::make_unique<ninfer::PinnedHostBuffer>(round_inputs_layout(fn).total);
  } catch (const std::exception &e) {
    *error = std::string("Flash-Next's reservations: ") + e.what();
    return -1;
  }
  if (fn.speculative_backend != IGNIS_SPECULATIVE_NONE) {
    fn.verify = verify::create(g, fn.kv_format, fn.decode_lanes, fn.draft_tokens, fn.draft_row_budget,
                               attention_sections(fn), gdn_layers(fn), fn.mtp != nullptr, error);
    if (fn.verify == nullptr) {
      *error = "Flash-Next's verify round: " + *error;
      return -1;
    }
  }
  if (fn.mtp != nullptr) {
    // The head's experts are all resident: every slot names a record and a K class.
    if (fn.mtp_slot_table.size() != static_cast<std::size_t>(g.experts) * 2) {
      *error = "an MTP load needs its expert slot table (ignis_model_load_options.mtp_expert_slots)";
      return -1;
    }
    for (std::size_t i = 0; i < fn.mtp_slot_table.size(); ++i) {
      const ignis_moe_slot &slot = fn.mtp_slot_table[i];
      if (slot.record == nullptr || !ignis_moe::valid_k2(slot.k2)) {
        *error = "the MTP head's expert slot " + std::to_string(i) + " names no record or no K class";
        return -1;
      }
    }
    try {
      const auto rows = static_cast<std::size_t>(fn.rows());
      fn.mtp_tokens = std::make_unique<ninfer::DeviceBuffer>(aligned(rows * sizeof(int32_t)));
      fn.mtp_slots =
          std::make_unique<ninfer::DeviceBuffer>(aligned(fn.mtp_slot_table.size() * sizeof(ignis_moe_slot)));
    } catch (const std::exception &e) {
      *error = std::string("the MTP head's reservations: ") + e.what();
      return -1;
    }
    if (!check_cuda(cudaMemcpy(fn.mtp_slots->p, fn.mtp_slot_table.data(),
                               fn.mtp_slot_table.size() * sizeof(ignis_moe_slot), cudaMemcpyHostToDevice),
                    "uploading the MTP head's slot table", error)) {
      return -1;
    }
  }
  ignis_moe_workspace workspace{fn.moe_workspace->p, fn.decode_rows, fn.prefill_chunk_tokens};
  if (ignis_moe_workspace_init(&workspace, static_cast<int64_t *>(fn.moe_acc->p), model.stream) != 0) {
    *error = std::string("ignis_moe_workspace_init: ") + ignis_moe_last_error();
    return -1;
  }
  return check_cuda(cudaStreamSynchronize(model.stream), "the load's synchronize", error) ? 0 : -1;
}

int32_t program_prefill(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq, const int32_t *token_ids,
                        uint64_t num_tokens, uint64_t start_position, const ignis_sampling_params *sampling,
                        const ignis_prefill_options *options, float *out_logits) {
  const auto refuse = [](const std::string &why) {
    step::set_error("ignis_program_prefill: " + why);
    return -1;
  };
  if (pool == nullptr || seq == nullptr || token_ids == nullptr || sampling == nullptr || num_tokens == 0) {
    return refuse("null argument or empty span");
  }
  if (!step::sampling_size_ok(*sampling)) return refuse("unrecognized ignis_sampling_params size");
  FlashNextModel &fn = *model->flash_next;
  const Geometry &g = fn.g;
  bool span_ok = false;
  if (const std::string why = prefill_options_refusal(options, span_ok); !why.empty()) return refuse(why);
  if (const std::string why = pool_refusal(fn, *pool); !why.empty()) return refuse(why);
  if (!ignis_seq_belongs_to(*pool, *seq)) return refuse("the sequence was not drawn from this pool");
  if (const char *why = ignis_seq_restore_refusal(*seq)) return refuse(why);
  if (seq->position != start_position) return refuse("start_position does not match the sequence frontier");
  if (num_tokens > ignis_seq_token_capacity(*seq) - seq->position) {
    return refuse("span exceeds the sequence KV capacity");
  }
  if (seq->position + num_tokens > fn.max_context_tokens) {
    return refuse("the span runs past the load's max_context_tokens, which its scratch is sized for");
  }
  for (uint64_t i = 0; i < num_tokens; ++i) {
    if (token_ids[i] < 0 || token_ids[i] >= g.vocab) {
      return refuse("token id " + std::to_string(token_ids[i]) + " at " + std::to_string(i) +
                    " is outside the vocabulary");
    }
  }
  const auto began = std::chrono::steady_clock::now();
  cudaStream_t stream = model->stream;
  Views views = views_of(fn, pool);
  const std::size_t row_bytes = static_cast<std::size_t>(g.ngram_heads) * g.ngram_row_bytes();
  std::uint16_t *span_logits = options->out_span_logits;
  const auto vocab = static_cast<std::size_t>(g.vocab);

  uint64_t offset = 0;
  while (offset < num_tokens) {
    const auto chunk = static_cast<int32_t>(std::min<uint64_t>(fn.prefill_chunk_tokens, num_tokens - offset));
    const bool last = offset + static_cast<uint64_t>(chunk) == num_tokens;
    const auto position = static_cast<int32_t>(seq->position);
    std::string error;
    int32_t successor = -1;
    std::vector<std::uint16_t> host_logits;
    const bool ok = [&] {
      auto scope = model->scratch->scope();
      if (!check_cuda(cudaMemcpyAsync(fn.token_ids->p, token_ids + offset, chunk * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream),
                      "staging the chunk's ids", &error) ||
          !check_cuda(cudaMemcpyAsync(fn.ngram_rows->p, options->ngram_rows + offset * row_bytes, chunk * row_bytes,
                                      cudaMemcpyHostToDevice, stream),
                      "staging the chunk's n-gram rows", &error) ||
          !check_cuda(cudaMemcpyAsync(fn.slots->p, &seq->slot, sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                      "staging the slot", &error) ||
          !check_cuda(cudaMemcpyAsync(fn.positions->p, &position, sizeof(int32_t), cudaMemcpyHostToDevice, stream),
                      "staging the position", &error)) {
        return false;
      }
      Batch batch;
      batch.lanes = 1;
      batch.tokens = chunk;
      batch.slots = static_cast<const int32_t *>(fn.slots->p);
      batch.positions = static_cast<const int32_t *>(fn.positions->p);
      batch.max_visible = position + chunk;
      if (forward(fn, views.ctx, batch, IGNIS_RESIDENCY_PREFILL, *model->scratch, stream, &error) != 0) {
        return false;
      }
      // Test-only residual-stack tap (kernel/include/ignis_fn_residual_tap.h): one flag load when
      // disarmed.
      if (ignis_fn_residual_tap_record(position, chunk, g.residual_width(), fn.residual->p, stream) != 0) {
        error = ignis_fn_residual_tap_last_error();
        return false;
      }
      const auto residual_row = [&](int32_t row) {
        return static_cast<const unsigned char *>(fn.residual->p) +
               static_cast<std::size_t>(row) * g.residual_width() * sizeof(__nv_bfloat16);
      };
      // The measurement readout: the head over every row, a block at a time.
      if (span_ok && span_logits != nullptr) {
        auto block_scope = model->scratch->scope();
        void *block = model->scratch->alloc_bytes(kSpanLogitRows * vocab * sizeof(std::uint16_t)).data;
        for (int32_t row = 0; row < chunk; row += kSpanLogitRows) {
          const int32_t n = std::min(kSpanLogitRows, chunk - row);
          if (fn_head(g, fn.weights->final_mixer, fn.weights->head, residual_row(row), n, block, *model->scratch,
                      stream) != 0) {
            error = std::string("the span head: ") + fn_last_error();
            return false;
          }
          if (!check_cuda(cudaMemcpyAsync(span_logits + (offset + static_cast<uint64_t>(row)) * vocab, block,
                                          static_cast<std::size_t>(n) * vocab * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost, stream),
                          "copying the span's logits", &error)) {
            return false;
          }
        }
      }
      if (last) {
        // The draw's row is computed in the span readout's own block (the 32-row block of the
        // chunk that holds its last row), so a span's row and the draw's are one computation, bit
        // for bit, whatever the linears' routes do with a row count; the head reads the weights
        // once either way.
        const int32_t block_first = (chunk - 1) / kSpanLogitRows * kSpanLogitRows;
        const int32_t block_rows = chunk - block_first;
        // Its own scope: the MTP head's entries below reuse the arena (the copies out of the block
        // are ordered before them on the stream).
        auto draw_scope = model->scratch->scope();
        auto *block = static_cast<unsigned char *>(
            model->scratch->alloc_bytes(static_cast<std::size_t>(kSpanLogitRows) * vocab * sizeof(std::uint16_t)).data);
        if (fn_head(g, fn.weights->final_mixer, fn.weights->head, residual_row(block_first), block_rows, block,
                    *model->scratch, stream) != 0) {
          error = std::string("the head: ") + fn_last_error();
          return false;
        }
        void *logits = block + static_cast<std::size_t>(block_rows - 1) * vocab * sizeof(std::uint16_t);
        const ninfer::Tensor logits_tensor(logits, ninfer::DType::BF16, {g.vocab, 1, 1, 1});
        if (step::sample_single(model, pool, seq, logits_tensor, *sampling, ninfer::ops::kSamplePurposePrefill,
                                position + chunk - 1, &successor, options->out_permitted_prob) != 0) {
          error = "the draw failed";  // sample_single named it on the step channel
          return false;
        }
        if (out_logits != nullptr) {
          host_logits.resize(vocab);
          if (!check_cuda(cudaMemcpyAsync(host_logits.data(), logits, vocab * sizeof(std::uint16_t),
                                          cudaMemcpyDeviceToHost, stream),
                          "copying the logits", &error)) {
            return false;
          }
        }
      }
      // Spec flash-next/07: the MTP head's entries for the chunk's positions, entry p from the
      // trunk's stack at p and token p + 1 -- the span's next id, or at the span's last position
      // the token just drawn -- so the head's frontier is the sequence's.
      if (fn.mtp != nullptr) {
        auto *next = static_cast<int32_t *>(fn.mtp_tokens->p);
        const bool staged =
            (chunk == 1 || check_cuda(cudaMemcpyAsync(next, token_ids + offset + 1, (chunk - 1) * sizeof(int32_t),
                                                      cudaMemcpyHostToDevice, stream),
                                      "staging the head's tokens", &error)) &&
            (last ? check_cuda(cudaMemcpyAsync(next + chunk - 1, model->sampling_single_out->p, sizeof(int32_t),
                                               cudaMemcpyDeviceToDevice, stream),
                               "staging the drawn token", &error)
                  : check_cuda(cudaMemcpyAsync(next + chunk - 1, token_ids + offset + chunk, sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream),
                               "staging the next chunk's token", &error));
        if (!staged) return false;
        Batch batch;
        batch.lanes = 1;
        batch.tokens = chunk;
        batch.slots = static_cast<const int32_t *>(fn.slots->p);
        batch.positions = static_cast<const int32_t *>(fn.positions->p);
        batch.max_visible = position + chunk;
        if (mtp_entries(fn, views.ctx, batch, next, *model->scratch, stream, &error) != 0) return false;
      }
      return check_cuda(cudaStreamSynchronize(stream), "the chunk's synchronize", &error);
    }();
    if (!ok) {
      // A chunk that stops inside a layer may leave residency's prefetch forked.
      (void)ignis_residency_join(fn.residency, stream);
      (void)cudaStreamSynchronize(stream);
      if (error != "the draw failed") {
        step::set_error("ignis_program_prefill: chunk at span offset " + std::to_string(offset) + " (" +
                        std::to_string(chunk) + " tokens) for sequence slot " + std::to_string(seq->slot) +
                        " failed: " + error + "; the sequence is not usable past it, release it");
      }
      return -1;
    }
    advance_frontiers(seq, static_cast<uint32_t>(chunk));
    if (last) {
      seq->pending_token = successor;
      if (out_logits != nullptr) {
        for (std::size_t v = 0; v < vocab; ++v) out_logits[v] = bf16_to_f32(host_logits[v]);
      }
    }
    offset += static_cast<uint64_t>(chunk);
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count());
  model->last_step_kernel_count = static_cast<uint64_t>(g.layers);
  model->last_step_graph_launches = 0;
  return 0;
}

namespace {

// One round's device work after its staging: the forward of `width` lanes of one token, the head
// on every lane into the shared sampling logits, and the batched draw. What a captured graph
// holds, and what the eager path runs (with the permitted mask between the head and the draw).
int32_t run_round(ignis_model *model, FlashNextModel &fn, const Context &ctx, uint32_t width, bool constrained,
                  std::string *error) {
  const Geometry &g = fn.g;
  cudaStream_t stream = model->stream;
  ninfer::DeviceArena &scratch = *model->decode_graph_scratch;
  auto scope = scratch.scope();
  Batch batch;
  batch.lanes = static_cast<int32_t>(width);
  batch.tokens = 1;
  batch.slots = static_cast<const int32_t *>(fn.slots->p);
  batch.positions = static_cast<const int32_t *>(fn.positions->p);
  batch.max_visible = decode_max_visible(fn);
  if (forward(fn, ctx, batch, IGNIS_RESIDENCY_DECODE, scratch, stream, error) != 0) {
    return -1;
  }
  if (fn_head(g, fn.weights->final_mixer, fn.weights->head, fn.residual->p, batch.lanes,
              model->sampling_decode_logits->p, scratch, stream) != 0) {
    *error = std::string("the head: ") + fn_last_error();
    return -1;
  }
  if (constrained &&
      ignis_permit_mask(model->sampling_decode_logits->p, g.vocab, width,
                        static_cast<const int32_t *>(model->sampling_decode_permitted->p),
                        static_cast<const int32_t *>(model->sampling_decode_permitted_counts->p),
                        IGNIS_MAX_PERMITTED_TOKENS, stream) != 0) {
    *error = "the permitted-set mask launch failed";
    return -1;
  }
  const auto lanes = static_cast<int32_t>(width);
  const ninfer::Tensor logits_tensor(model->sampling_decode_logits->p, ninfer::DType::BF16, {g.vocab, lanes, 1, 1});
  ninfer::Tensor out_tensor(model->sampling_decode_out->p, ninfer::DType::I32, {lanes, 1, 1, 1});
  const ninfer::Tensor positions_tensor(model->sampling_decode_positions->p, ninfer::DType::I32, {lanes, 1, 1, 1});
  try {
    ninfer::DeviceArena::Scope workspace_scope = model->sampling_workspace->scope();
    ninfer::ops::sample(logits_tensor, out_tensor, g.vocab,
                        static_cast<const ninfer::ops::SamplingConfig *>(model->sampling_decode_configs->p),
                        positions_tensor, ninfer::ops::kSamplePurposeDecode, *model->sampling_workspace, stream);
  } catch (const std::exception &e) {
    *error = std::string("the draw: ") + e.what();
    return -1;
  }
  // Spec flash-next/07: on an MTP load every committed position gets its head entry, this round's
  // from its stack and the token it drew; it drafts nothing (the next verify round runs at
  // extent 0 and drafts).
  if (fn.mtp != nullptr &&
      mtp_entries(fn, ctx, batch, static_cast<const int32_t *>(model->sampling_decode_out->p), scratch, stream, error) !=
          0) {
    return -1;
  }
  return 0;
}

}  // namespace

int32_t program_decode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *const *sequences, uint64_t batch_size,
                       const ignis_sampling_params *sampling, int32_t *out_token_ids,
                       const ignis_decode_options *options) {
  const auto refuse = [](const std::string &why) {
    step::set_error("ignis_program_decode: " + why);
    return -1;
  };
  if (pool == nullptr || sequences == nullptr || sampling == nullptr || out_token_ids == nullptr || batch_size == 0) {
    return refuse("null argument or empty batch");
  }
  FlashNextModel &fn = *model->flash_next;
  const Geometry &g = fn.g;
  if (batch_size > fn.decode_lanes) {
    return refuse("a round of " + std::to_string(batch_size) + " lanes on a load of " +
                  std::to_string(fn.decode_lanes));
  }
  if (options == nullptr || options->size != sizeof(ignis_decode_options)) {
    return refuse("a Flash-Next round needs its options: the lanes' n-gram rows");
  }
  if (options->ngram_rows == nullptr) return refuse("a Flash-Next round needs the lanes' n-gram rows");
  if (const std::string why = pool_refusal(fn, *pool); !why.empty()) return refuse(why);
  if (fn.captured_pool != nullptr && fn.captured_pool != pool) {
    return refuse("the round's pool is not the one its graphs were captured against");
  }
  // Spec flash-next/07: a speculative window is the verify round (speculative.cu).
  if (options->speculative_window != 0 || options->drafts != nullptr) {
    if (fn.verify == nullptr) return refuse("this Flash-Next load has no speculative decoding");
    return program_verify(model, pool, sequences, batch_size, sampling, out_token_ids, options);
  }

  const auto width = static_cast<uint32_t>(batch_size);
  const auto vocab = static_cast<int32_t>(g.vocab);
  std::vector<ninfer::ops::SamplingConfig> configs(batch_size);
  std::vector<int32_t> ids(batch_size), slots(batch_size), positions(batch_size);
  std::vector<int32_t> permitted(batch_size * IGNIS_MAX_PERMITTED_TOKENS, -1), permitted_counts(batch_size, 0);
  bool constrained = false;
  for (uint64_t i = 0; i < batch_size; ++i) {
    ignis_seq *seq = sequences[i];
    const std::string at = " at index " + std::to_string(i);
    if (seq == nullptr || seq->pending_token < 0) return refuse("sequence is null or was not prefilled" + at);
    if (!ignis_seq_belongs_to(*pool, *seq)) return refuse("the sequence was not drawn from this pool" + at);
    if (const char *why = ignis_seq_restore_refusal(*seq)) return refuse(why + at);
    if (seq->position >= ignis_seq_token_capacity(*seq) || seq->position >= fn.max_context_tokens) {
      return refuse("sequence reached its KV capacity" + at);
    }
    if (!step::sampling_size_ok(sampling[i])) return refuse("unrecognized ignis_sampling_params size" + at);
    const ignis_sampling_params &lane = sampling[i];
    if (!step::unconstrained(lane)) {
      if (lane.permitted_count > IGNIS_MAX_PERMITTED_TOKENS || lane.permitted_ids == nullptr) {
        return refuse("a permitted set of " + std::to_string(lane.permitted_count) + " ids" + at);
      }
      for (uint32_t k = 0; k < lane.permitted_count; ++k) {
        if (lane.permitted_ids[k] < 0 || lane.permitted_ids[k] >= vocab) {
          return refuse("permitted id " + std::to_string(lane.permitted_ids[k]) + at + " is outside the vocabulary");
        }
        permitted[i * IGNIS_MAX_PERMITTED_TOKENS + k] = lane.permitted_ids[k];
      }
      permitted_counts[i] = static_cast<int32_t>(lane.permitted_count);
      constrained = true;
    }
    configs[i] = step::to_sampling_config(lane, pool->token_counts_for(seq->slot));
    ids[i] = seq->pending_token;
    slots[i] = seq->slot;
    positions[i] = static_cast<int32_t>(seq->position);
  }

  const auto began = std::chrono::steady_clock::now();
  cudaStream_t stream = model->stream;
  const std::size_t row_bytes = static_cast<std::size_t>(g.ngram_heads) * g.ngram_row_bytes();
  const std::size_t config_bytes = batch_size * sizeof(ninfer::ops::SamplingConfig);
  const std::size_t lane_bytes = batch_size * sizeof(int32_t);
  const std::size_t permitted_bytes = permitted.size() * sizeof(int32_t);
  const void *src_configs = configs.data();
  const void *src_positions = positions.data();
  const void *src_ids = ids.data();
  const void *src_slots = slots.data();
  const void *src_ngram = options->ngram_rows;
  const void *src_permitted = permitted.data();
  const void *src_counts = permitted_counts.data();
  // GitHub #306, step 7: the copies' sources gathered into page-locked memory first, so each
  // copy is queued without the driver's pageable staging pass (fusion.h's Staging). The buffer
  // is rewritten only after the previous round's synchronize.
  if (fn.round_inputs != nullptr && fused(Fusion::Staging)) {
    const RoundInputs at = round_inputs_layout(fn);
    auto *base = static_cast<unsigned char *>(fn.round_inputs->data());
    const auto put = [&](std::size_t offset, const void *src, std::size_t bytes) -> const void * {
      std::memcpy(base + offset, src, bytes);
      return base + offset;
    };
    src_configs = put(at.configs, configs.data(), config_bytes);
    src_positions = put(at.positions, positions.data(), lane_bytes);
    src_ids = put(at.ids, ids.data(), lane_bytes);
    src_slots = put(at.slots, slots.data(), lane_bytes);
    src_ngram = put(at.ngram, options->ngram_rows, batch_size * row_bytes);
    if (constrained) {
      src_permitted = put(at.permitted, permitted.data(), permitted_bytes);
      src_counts = put(at.counts, permitted_counts.data(), lane_bytes);
    }
  }
  std::string error;
  const bool staged =
      check_cuda(cudaMemcpyAsync(model->sampling_decode_configs->p, src_configs, config_bytes, cudaMemcpyHostToDevice,
                                 stream),
                 "staging the sampling configs", &error) &&
      check_cuda(cudaMemcpyAsync(model->sampling_decode_positions->p, src_positions, lane_bytes,
                                 cudaMemcpyHostToDevice, stream),
                 "staging the sampling positions", &error) &&
      check_cuda(cudaMemcpyAsync(fn.token_ids->p, src_ids, lane_bytes, cudaMemcpyHostToDevice, stream),
                 "staging the ids", &error) &&
      check_cuda(cudaMemcpyAsync(fn.slots->p, src_slots, lane_bytes, cudaMemcpyHostToDevice, stream),
                 "staging the slots", &error) &&
      check_cuda(cudaMemcpyAsync(fn.positions->p, src_positions, lane_bytes, cudaMemcpyHostToDevice, stream),
                 "staging the positions", &error) &&
      check_cuda(cudaMemcpyAsync(fn.ngram_rows->p, src_ngram, batch_size * row_bytes, cudaMemcpyHostToDevice, stream),
                 "staging the n-gram rows", &error) &&
      (!constrained ||
       (check_cuda(cudaMemcpyAsync(model->sampling_decode_permitted->p, src_permitted, permitted_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "staging the permitted sets", &error) &&
        check_cuda(cudaMemcpyAsync(model->sampling_decode_permitted_counts->p, src_counts, lane_bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "staging the permitted counts", &error)));
  if (!staged) {
    // Copies queued before the failure may still read round_inputs, which the next round rewrites.
    (void)cudaStreamSynchronize(stream);
    return refuse(error);
  }

  // A constrained round runs eagerly: its mask sits between the head and the draw, as the 27B's.
  const bool use_graph = model->decode_graph_ready[width - 1] && !constrained;
  bool ok = true;
  if (use_graph) {
    // The prefetch copies of eager work before this replay join first (ignis_residency.h).
    ok = ignis_residency_join(fn.residency, stream) == 0 &&
         check_cuda(cudaGraphLaunch(model->decode_graph_exec[width - 1], stream), "cudaGraphLaunch", &error);
    if (!ok && error.empty()) error = std::string("ignis_residency_join: ") + ignis_residency_last_error();
  } else {
    Views views = views_of(fn, pool);
    ok = run_round(model, fn, views.ctx, width, constrained, &error) == 0;
  }
  if (ok && constrained && options->out_permitted_probs != nullptr) {
    ok = ignis_permit_probability(model->sampling_decode_logits->p, vocab, width,
                                  static_cast<const int32_t *>(model->sampling_decode_permitted->p),
                                  static_cast<const int32_t *>(model->sampling_decode_permitted_counts->p),
                                  IGNIS_MAX_PERMITTED_TOKENS, static_cast<const int32_t *>(model->sampling_decode_out->p),
                                  static_cast<float *>(model->sampling_decode_permitted_probs->p), stream) == 0 &&
         check_cuda(cudaMemcpyAsync(options->out_permitted_probs, model->sampling_decode_permitted_probs->p,
                                    batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
                    "copying the permitted probabilities", &error);
  }
  std::vector<int32_t> successors(batch_size, -1);
  ok = ok &&
       check_cuda(cudaMemcpyAsync(successors.data(), model->sampling_decode_out->p, batch_size * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream),
                  "copying the drawn ids", &error) &&
       check_cuda(cudaStreamSynchronize(stream), "the round's synchronize", &error);
  if (!ok) {
    (void)ignis_residency_join(fn.residency, stream);
    (void)cudaStreamSynchronize(stream);
    return refuse("the round failed: " + error + "; its lanes are not usable past it, release them");
  }
  if (!constrained && options->out_permitted_probs != nullptr) {
    std::fill(options->out_permitted_probs, options->out_permitted_probs + batch_size, 0.0F);
  }
  if (options->out_committed_counts != nullptr) {
    std::fill(options->out_committed_counts, options->out_committed_counts + batch_size, 1);
  }
  if (options->out_extents != nullptr) {
    std::fill(options->out_extents, options->out_extents + batch_size, 0U);
  }
  // The round is atomic: every lane commits its pending token and draws its next one together.
  for (uint64_t i = 0; i < batch_size; ++i) {
    out_token_ids[i] = sequences[i]->pending_token;
    sequences[i]->pending_token = successors[i];
    advance_frontiers(sequences[i], 1);
  }
  model->last_step_micros = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - began).count());
  model->last_step_kernel_count = static_cast<uint64_t>(g.layers);
  model->last_step_graph_launches = use_graph ? 1 : 0;
  return 0;
}

int32_t capture_decode_graphs(ignis_model *model, ignis_seq_pool *pool, uint32_t *out_ready_mask,
                              std::string *error) {
  FlashNextModel &fn = *model->flash_next;
  *out_ready_mask = 0;
  if (const std::string why = pool_refusal(fn, *pool); !why.empty()) {
    *error = "ignis_decode_graph_capture: " + why;
    return -1;
  }
  Views views = views_of(fn, pool);
  cudaStream_t stream = model->stream;
  // Eager work before a capture leaves residency's prefetch to join first.
  if (ignis_residency_join(fn.residency, stream) != 0 || cudaStreamSynchronize(stream) != cudaSuccess) {
    *error = std::string("ignis_decode_graph_capture: ignis_residency_join: ") + ignis_residency_last_error();
    return -1;
  }
  for (uint32_t width = 1; width <= fn.decode_lanes; ++width) {
    std::string failure;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture",
                         &failure);
    if (ok) {
      ok = run_round(model, fn, views.ctx, width, /*constrained=*/false, &failure) == 0;
      if (!ok) {
        // A round that stopped after a step with a lookahead left residency's prefetch forked:
        // it joins before the capture ends (ignis_residency.h rule (b)), or the end fails and
        // the fork outlives the capture.
        (void)ignis_residency_join(fn.residency, stream);
      }
      const cudaError_t end = cudaStreamEndCapture(stream, &graph);
      if (end != cudaSuccess && ok) {
        ok = check_cuda(end, "cudaStreamEndCapture", &failure);
      }
    }
    if (ok) {
      ok = check_cuda(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate", &failure);
    }
    if (graph != nullptr) cudaGraphDestroy(graph);
    if (model->decode_graph_exec[width - 1] != nullptr) {
      cudaGraphExecDestroy(model->decode_graph_exec[width - 1]);
      model->decode_graph_exec[width - 1] = nullptr;
    }
    model->decode_graph_ready[width - 1] = ok;
    if (ok) {
      model->decode_graph_exec[width - 1] = exec;
      *out_ready_mask |= 1U << (width - 1);
    } else {
      // A failed capture degrades this width to the eager round; it never refuses service.
      // A forward that stopped inside the capture after a lookahead step left residency's
      // prefetch fork on the dead capture: one join, outside any capture, clears it (dc4af5b)
      // before the next width begins its own.
      (void)cudaGetLastError();
      (void)ignis_residency_join(fn.residency, stream);
      (void)cudaGetLastError();
      *error = "ignis_decode_graph_capture: width " + std::to_string(width) + ": " + failure;
    }
  }
  // Spec flash-next/07: the verify round's pass and commit graphs, after the rounds'.
  if (fn.verify != nullptr) {
    std::string failure;
    (void)capture_verify_graphs(model, pool, &failure);
    if (!failure.empty()) *error = "ignis_decode_graph_capture: " + failure;
  }
  fn.captured_pool = pool;
  return 0;
}

uint32_t verify_ready_mask(const ignis_model &model) {
  const FlashNextModel &fn = *model.flash_next;
  return fn.verify != nullptr ? fn.verify->ready_mask() : 0;
}

}  // namespace ignis::flash_next

void ignis::flash_next::FlashNextModelDeleter::operator()(FlashNextModel *model) const {
  delete model;
}
