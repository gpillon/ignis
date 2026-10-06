// ignis kernel leaf -- the Flash-Next program's internal contract (spec
// flash-next/04, GitHub #302; ADR 0043: OURS, not vendored).
//
// The seam between the program driver (slice S1: the layer loop, the
// hyper-connection streams, the binder, the state sections and the plan) and
// the op families the other slices build:
//   S2  fn_gdn_layer, fn_qsa_attention    GDN at hidden 2560; GQA 24/2 dense,
//                                         BF16 and hq-e8-2b KV
//   S3  fn_indexer_select                 the QSA indexer and the selected
//                                         tokens sparse attention reads
//   S4  fn_ngram_add                      the n-gram embedding's device side
//   spec 02 (kern)  ignis_moe.h           router, experts, shared expert, combine
//   spec 03 (resid) ignis_residency.h     the expert slot tables (declared there,
//                                         not here: see "Expert residency" below)
// Everything below is declared here and defined by the slice named on it.
//
// One layer of the program, as transformers' Qwen4ExpTextDecoderLayer runs it
// (the oracle, ADR 0043):
//
//   if layer == ngram.layer:  fn_ngram_add(hidden)                      [S4]
//   x = hc_mix(attn_hc, hidden)                                          [S1]
//   y = layer is GDN ? fn_gdn_layer(x)                                   [S2]
//                    : fn_qsa_attention(x, fn_indexer_select(x))         [S2, S3]
//                      (fn_indexer_select on EVERY QSA call: its key state
//                       must see every token, dense or not)
//   hidden = hc_inject(attn_hc, hidden, y)                               [S1]
//   x = hc_mix(mlp_hc, hidden)                                           [S1]
//   ids = ignis_moe_router(router_L, x)                                  [kern]
//   lookahead = ignis_moe_router(router_L+1, x).logits  (none at the last) [kern]
//   ignis_residency_step(L, phase, ids, lookahead)                       [resid]
//   y = combine(ignis_moe_experts_*(x) , ignis_moe_shared_expert(x))     [kern]
//   hidden = hc_inject(mlp_hc, hidden, y)                                [S1]
//
// and after the last layer the final mixer (a mix with no inject, which is
// also the model's final norm) and the head.
//
// Conventions every entry point keeps:
// - Activations are BF16, token-major: row r of a `[rows][width]` buffer is
//   one token's features, contiguous (the 27B's feature-major `[hidden, T]`
//   buffers have the same bytes). The hyper-connection residual is
//   `[rows][streams * hidden]`, stream s at columns [s * hidden, (s+1) * hidden).
// - A call covers a Batch: `lanes` sequences of `tokens` consecutive tokens
//   each, lane-major (`rows = lanes * tokens`). Prefill: one lane, a chunk of
//   tokens, run eagerly. Decode: 1..lane_count lanes, one token each, captured
//   in a CUDA graph -- so everything that changes between rounds (positions,
//   slots, visible-token counts) is read from DEVICE memory the caller
//   refreshes before the launch, and a decode call's launches never depend on
//   a host value that changes between rounds.
// - Scratch: every temporary comes from `scratch` (a ninfer::DeviceArena the
//   load reserved, ADR 0030: nothing allocates while serving). An entry point
//   opens `auto scope = scratch.scope();` and returns everything it took
//   before it returns; its peak over any call it accepts is at most its
//   `*_scratch_bytes()`, which the plan sums. Outputs go to caller-provided
//   buffers, never to the arena.
// - Every entry point runs on `stream` and returns 0, or -1 with
//   fn_set_error() naming what failed. Nothing synchronizes the host.
// - Norm conventions (the checkpoint's): Qwen4ExpTextRMSNorm multiplies by
//   (1 + weight), grouped per `hidden`-wide stream where it spans several
//   (hc_norm, the PLE norms); the GDN gated norm multiplies by `weight` and
//   gates with sigmoid(z) (output_gate_type "sigmoid", not the 27B's SiLU).

#pragma once

#include "ignis_model.h"
#include "ignis_moe.h"
#include "ignis_seq.h"

#include "core/arena.h"
#include "ninfer/ops/rope.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

struct ignis_seq_pool;

namespace ignis::flash_next {

// ---------------------------------------------------------------------------
// Geometry: every count and shape, read once off the load's ignis_topology
// (crates/core/src/compute.rs ModelConfig::qwen38_flash_next is its source).
// A kernel may specialize on a value at compile time, but then refuses at
// load any other (as model.cu does for the GDN head dim and conv kernel).
struct Geometry {
  int32_t layers = 0;
  int32_t hidden = 0;
  int32_t vocab = 0;
  // Attention (QSA) layers.
  int32_t q_heads = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  int32_t rotary_dim = 0;
  // GDN layers.
  int32_t gdn_qk_heads = 0;
  int32_t gdn_value_heads = 0;
  int32_t gdn_head_dim = 0;
  int32_t gdn_conv_kernel = 0;
  // MoE block.
  int32_t experts = 0;
  int32_t experts_per_token = 0;
  int32_t expert_intermediate = 0;
  int32_t shared_intermediate = 0;
  // Hyper-connections.
  int32_t streams = 0;
  int32_t hc_rank = 0;
  // Indexer.
  int32_t indexer_heads = 0;
  int32_t indexer_head_dim = 0;
  int32_t indexer_kv_heads = 0;
  int32_t compress_ratio = 0;
  int32_t indexer_budget = 0;
  // N-gram embedding.
  int32_t ngram_size = 0;
  int32_t ngram_heads = 0;  // (ngram_size - 1) * heads_per_ngram
  int32_t ngram_embed_dim = 0;
  int32_t ngram_conv_kernel = 0;
  int32_t ngram_layer = 0;
  float rms_norm_eps = 0.0F;

  int32_t residual_width() const { return streams * hidden; }
  // The most visible tokens attention reads densely; also the widest
  // selection: budget/compress blocks of compress tokens plus a tail of up to
  // compress-1 (2051 on Flash-Next).
  int32_t dense_threshold() const { return indexer_budget + compress_ratio - 1; }
  int32_t selection_width() const { return dense_threshold(); }
  // The n-gram conv's past columns: (kernel - 1) * dilation, dilation = ngram_size.
  int32_t ngram_conv_state_columns() const { return (ngram_conv_kernel - 1) * ngram_size; }
  // One table row's bytes: INT4 codes of a head_dim-wide row, then one fp16
  // scale per 32 values (layout.md section 7.1: 90 on Flash-Next).
  int32_t ngram_head_dim() const { return ngram_embed_dim / ngram_heads; }
  int32_t ngram_row_bytes() const { return ngram_head_dim() / 2 + ngram_head_dim() / 32 * 2; }

  static Geometry from(const ignis_topology &t) {
    Geometry g;
    g.layers = static_cast<int32_t>(t.num_layers);
    g.hidden = static_cast<int32_t>(t.hidden);
    g.vocab = static_cast<int32_t>(t.vocab);
    g.q_heads = static_cast<int32_t>(t.num_q_heads);
    g.kv_heads = static_cast<int32_t>(t.num_kv_heads);
    g.head_dim = static_cast<int32_t>(t.head_dim);
    g.rotary_dim = static_cast<int32_t>(t.rotary_dim);
    g.gdn_head_dim = static_cast<int32_t>(t.gdn_head_dim);
    g.gdn_qk_heads = t.gdn_head_dim == 0 ? 0 : static_cast<int32_t>(t.gdn_state_cols / t.gdn_head_dim);
    g.gdn_value_heads = static_cast<int32_t>(t.gdn_value_heads);
    g.gdn_conv_kernel = static_cast<int32_t>(t.gdn_conv_kernel);
    g.experts = static_cast<int32_t>(t.moe.num_experts);
    g.experts_per_token = static_cast<int32_t>(t.moe.experts_per_token);
    g.expert_intermediate = static_cast<int32_t>(t.moe.expert_intermediate);
    g.shared_intermediate = static_cast<int32_t>(t.moe.shared_expert_intermediate);
    g.streams = static_cast<int32_t>(t.hyper.streams);
    g.hc_rank = static_cast<int32_t>(t.hyper.rank);
    g.indexer_heads = static_cast<int32_t>(t.indexer.heads);
    g.indexer_head_dim = static_cast<int32_t>(t.indexer.head_dim);
    g.indexer_kv_heads = static_cast<int32_t>(t.indexer.kv_heads);
    g.compress_ratio = static_cast<int32_t>(t.indexer.compress_ratio);
    g.indexer_budget = static_cast<int32_t>(t.indexer.budget);
    g.ngram_size = static_cast<int32_t>(t.ngram.ngram_size);
    g.ngram_heads = static_cast<int32_t>((t.ngram.ngram_size - 1) * t.ngram.heads_per_ngram);
    g.ngram_embed_dim = static_cast<int32_t>(t.ngram.embed_dim);
    g.ngram_conv_kernel = static_cast<int32_t>(t.ngram.conv_kernel);
    g.ngram_layer = static_cast<int32_t>(t.ngram.layer);
    g.rms_norm_eps = t.rms_norm_eps;
    return g;
  }
};

// ---------------------------------------------------------------------------
// Weights, as the binder (S1) hands them: device pointers into the artifact's
// arena, names from docs/specs/flash-next/layout.md section 6, not fused.

// How a linear weight is stored (layout.md 6.1). FP8 is the converter's
// default; every linear also has a BF16 route, because a part whose FP8 cost
// the conversion flags is re-converted to BF16 (coordinator, 2026-10-05).
enum class WeightFormat : int32_t {
  // FP8_E4M3FN_ROW_BF16S / row-scale-v1: codes [rows][cols], padding to 256
  // bytes, BF16 scales [rows].
  Fp8RowScale = 0,
  // BF16 / contiguous-le-v1: [rows][cols].
  Bf16 = 1,
};

// One linear weight, y = W x: `rows` outputs of `cols` inputs.
struct Linear {
  const void *data = nullptr;
  int32_t rows = 0;
  int32_t cols = 0;
  WeightFormat format = WeightFormat::Fp8RowScale;
};

// y[r][o] = sum_c W[o][c] * x[r][c] for `rows` rows of BF16 x, fp32
// accumulation, BF16 out (or fp32 with y_f32): kern's ignis_fp8_linear for
// FP8, our BF16 route otherwise (the vendored BF16 linear only admits its
// registered 27B problems). [S1: kernel/src/flash_next/linear.cu, its next
// commit; until it lands a slice's test may shim it over ignis_fp8_linear.]
int32_t fn_linear(const Linear &w, const void *x, int32_t rows, void *y, bool y_f32,
                  ninfer::DeviceArena &scratch, cudaStream_t stream);

// A GDN layer's own weights (between its HC mix and inject).
struct GdnWeights {
  Linear in_proj_qkv;  // [q 2048 + k 2048 + v 6144, hidden]
  Linear in_proj_z;    // [6144, hidden]
  Linear in_proj_a;    // [value_heads, hidden] (FP8: 48 rows reach the converter's FP8 minimum)
  Linear in_proj_b;    // [value_heads, hidden]
  const void *conv = nullptr;     // BF16 conv1d [10240][kernel]
  const void *a_log = nullptr;    // BF16 [value_heads]
  const void *dt_bias = nullptr;  // BF16 [value_heads]
  const void *norm = nullptr;     // BF16 [gdn_head_dim], plain weight, sigmoid gate
  Linear out_proj;     // [hidden, 6144]
};

// The QSA indexer's weights.
struct IndexerWeights {
  Linear qk_proj;                 // [(heads + kv_heads) * indexer_head_dim, hidden]: q heads, then the key
  const void *q_norm = nullptr;   // BF16 [indexer_head_dim], (1 + w)
  const void *k_norm = nullptr;   // BF16 [indexer_head_dim], (1 + w)
};

// A QSA layer's attention weights.
struct QsaWeights {
  // [q_heads * 2 * head_dim, hidden]: per head its query (head_dim rows) then
  // its output gate (head_dim rows) -- the checkpoint's view(.., -1, 2 *
  // head_dim).chunk(2), NOT the 27B's fused query_key_gate_value order.
  Linear q_proj;
  Linear k_proj;                  // [kv_heads * head_dim, hidden]
  Linear v_proj;                  // [kv_heads * head_dim, hidden]
  const void *q_norm = nullptr;   // BF16 [head_dim], (1 + w)
  const void *k_norm = nullptr;   // BF16 [head_dim], (1 + w)
  Linear o_proj;                  // [hidden, q_heads * head_dim]
  IndexerWeights indexer;
};

// One hyper-connection mix (a layer's attn_ or mlp_hyper_connection, or the
// final hyper_connection_mixer, whose block_inject is null).
struct HcWeights {
  const void *hc_norm = nullptr;       // BF16 [streams * hidden], (1 + w), grouped per stream
  Linear mix_down;                     // [hc_rank, streams * hidden]
  Linear mix_up;                       // [streams * hidden, hc_rank]
  const void *block_inject = nullptr;  // BF16 [streams][streams * hidden]
};

// A layer's MoE block weights, for kern's ignis_moe.h (experts are residency's).
struct MoeWeights {
  const void *router = nullptr;              // BF16 [experts][hidden]
  const void *shared_gate = nullptr;         // FP8 row-scale [shared_intermediate, hidden]
  const void *shared_up = nullptr;           // FP8 row-scale [shared_intermediate, hidden]
  const void *shared_down = nullptr;         // FP8 row-scale [hidden, shared_intermediate]
  const void *shared_expert_gate = nullptr;  // BF16 [hidden]
};

// The n-gram embedding's device weights (decoder layer `ngram_layer` only).
struct NgramWeights {
  Linear key_proj;                    // [streams * hidden, ngram_embed_dim]
  Linear value_proj;                  // [hidden, ngram_embed_dim]
  const void *norm_key = nullptr;     // BF16 [streams * hidden], (1 + w), grouped
  const void *norm_query = nullptr;   // BF16 [streams * hidden], (1 + w), grouped
  const void *norm_conv = nullptr;    // BF16 [streams * hidden], (1 + w), grouped
  const void *conv = nullptr;         // BF16 depthwise conv1d [streams * hidden][ngram_conv_kernel]
};

// ---------------------------------------------------------------------------
// A verify call (spec flash-next/07 phase C, verify.h): `lanes` lanes of `tokens` = k + 1 columns
// each -- the anchor, then the drafts -- at positions[l] .. positions[l] + k, none of them
// committed yet. What the ops do differently on it, and where they record what the commit needs:
// - fn_gdn_layer leaves the lanes' recurrent state and conv taps as they are: its recurrence is the
//   vendored replay record (columns from valid_columns[l] on are no transition), and the layer's
//   raw k, v, {g, beta} and conv inputs land in its records below;
// - fn_indexer_select copies the call's raw keys -- its tail and pooled blocks advance in place --
//   to indexer_keys;
// - fn_qsa_attention decodes every column's listed hq rows, never a lane's visible rows.
// Rows are lane-major throughout: row = lane * tokens + column.
struct VerifyRecords {
  const int32_t *valid_columns = nullptr;  // DEVICE [lanes]: extent + 1, in [1, tokens]
  // GDN layer `ordinal`'s records start at these + ordinal * gdn_layer_bytes: key BF16
  // [rows][16][128], value BF16 [rows][48][128], {g, beta} fp32 pairs [rows][48], conv inputs BF16
  // [rows][10240] (the vendored replay record's layouts, and our fold's).
  void *gdn_key = nullptr;
  void *gdn_value = nullptr;
  void *gdn_gate = nullptr;
  void *gdn_conv = nullptr;
  std::size_t gdn_layer_bytes = 0;
  // Attention layer `ordinal`'s raw indexer keys, BF16 [rows][indexer_head_dim], at indexer_keys +
  // ordinal * indexer_layer_bytes.
  void *indexer_keys = nullptr;
  std::size_t indexer_layer_bytes = 0;
};

// ---------------------------------------------------------------------------
// A call's sequences.
struct Batch {
  int32_t lanes = 0;   // sequences in the call
  int32_t tokens = 0;  // consecutive tokens per sequence; rows = lanes * tokens
  // DEVICE [lanes]: each sequence's seq-pool slot (its KV block-table row, its
  // GDN, indexer and n-gram sections).
  const int32_t *slots = nullptr;
  // DEVICE [lanes]: each sequence's position of its first token in the call
  // (its frontier before the call). Token t of a lane is at positions[l] + t.
  const int32_t *positions = nullptr;
  // HOST: an upper bound of positions[l] + tokens over the call's lanes. Exact
  // in prefill (eager: a call may branch on it, e.g. skip the indexer while
  // dense); in decode the graph's own bound, so a decode call never branches
  // on it.
  int32_t max_visible = 0;
  // HOST: a verify call's records (above), or null for every other call.
  const VerifyRecords *verify = nullptr;
  int32_t rows() const { return lanes * tokens; }
};

// ---------------------------------------------------------------------------
// The lane state sections (S1 reserves them in the seq pool at load; ADR
// 0030). The existing sections keep their owners' types:
// - KV: `ignis_seq_pool::kv_pool` (ninfer::PagedKVPool, page-major, 64-token
//   pages, block-table rows = slots), one plane pair (BF16) or quad
//   (hq-e8-2b: K codes, K meta, V codes, V meta) per attention layer, in
//   attention-layer order; the hq residual window as the 27B's
//   (`ignis_seq_pool::hq_residual_plane`). S1 sizes the pool by the
//   topology's attention-layer count instead of kIgnisGqaLayerCount.
// - GDN conv taps and recurrent state: `ignis_seq_pool::gdn_pool`
//   (ninfer::LinearAttentionStatePool), one layer per GDN layer, in GDN-layer
//   order, exactly as kernel/src/gdn_layer.cu reads it.
// New sections, defined here:

// One attention layer's indexer state.
struct IndexerLayerState {
  // Compressed block keys, paged with the KV: physical KV page p of a lane
  // holds the compressed keys of its 64 / compress_ratio blocks at
  // block_keys + (p * blocks_per_page + b) * indexer_head_dim (BF16), so
  // cloning, prefix sharing, cancellation and snapshots move them with the
  // KV page. A block's key is written once its compress_ratio raw keys exist:
  // bf16(mean_fp32(raw keys)), then k_norm, then rope at the block's FIRST
  // token's position (the checkpoint's pooling).
  void *block_keys = nullptr;
  int32_t blocks_per_page = 0;
  // The incomplete block's raw keys, per slot: [slot][compress_ratio - 1]
  // [indexer_head_dim] BF16; how many are valid is position % compress_ratio.
  void *tail_keys = nullptr;
};

// The n-gram embedding's per-lane state: its dilated conv's past input
// columns, [slot][ngram_conv_state_columns][streams * hidden] BF16, oldest
// first. (Its hashing context, the last ngram_size - 1 token ids, is host
// state: ignis_core::ngram::NgramContext.)
struct NgramState {
  void *conv_columns = nullptr;
};

// What every entry point reads the load and the lanes' state through. [S1]
struct Context {
  Geometry g;
  int32_t kv_format = 0;  // enum ignis_kv_format, fixed for the load (ADR 0022)
  ninfer::ops::RopeFrequencies rope{};  // the text table (rotary_dim of head_dim)
  ignis_seq_pool *pool = nullptr;       // KV and GDN sections
  // [attention layers], indexed by attention-layer ordinal.
  const IndexerLayerState *indexer = nullptr;
  NgramState ngram;
};

// ---------------------------------------------------------------------------
// The op families' entry points.

// [S2] One GDN layer's core: y = out_proj(gated_norm(recurrence(conv(qkv)),
// sigmoid(z))) for x (the layer's HC mix), updating the lanes' conv taps and
// recurrent state of GDN layer `gdn_ordinal`. x, y: BF16 [rows][hidden].
int32_t fn_gdn_layer(const Context &ctx, int32_t gdn_ordinal, const GdnWeights &w,
                     const Batch &batch, const void *x, void *y,
                     ninfer::DeviceArena &scratch, cudaStream_t stream);
std::size_t fn_gdn_layer_scratch_bytes(const Geometry &g, int32_t rows);

// The tokens one query row attends to, written by fn_indexer_select:
// tokens[row][i] for i < counts[row] are visible positions in ascending
// order, the rest -1. While a row's visible tokens are at most
// dense_threshold() they are all of them.
struct Selection {
  int32_t *tokens = nullptr;  // DEVICE [rows][selection_width]
  int32_t *counts = nullptr;  // DEVICE [rows]
  // HOST: set by fn_indexer_select when the call is dense for every row
  // (batch.max_visible <= dense_threshold(), so only in prefill): `tokens`
  // and `counts` are then NOT written, and attention reads every visible
  // token causally. A decode graph's call is never dense: its lists are
  // always written.
  bool dense = false;
};

// [S3] Appends the call's tokens to attention layer `attn_ordinal`'s indexer
// state (raw keys to the tail, completed blocks' compressed keys to their
// pages), then selects, for every row, the blocks the checkpoint would:
// scores sum_h relu(q_h . k_block) / sqrt(indexer_head_dim) over the row's
// complete visible blocks, the top budget/compress_ratio of them, plus its
// incomplete block's tokens. Causal per row. x: BF16 [rows][hidden] (the
// layer's HC mix). Called on EVERY QSA layer call, prefill and decode: the
// key projection and the state update always run, because a block key needs
// every token's raw key and none can be rebuilt later. While
// batch.max_visible <= dense_threshold() (prefill) it skips the queries,
// scoring and selection and sets out.dense.
int32_t fn_indexer_select(const Context &ctx, int32_t attn_ordinal, const IndexerWeights &w,
                          const Batch &batch, const void *x, Selection &out,
                          ninfer::DeviceArena &scratch, cudaStream_t stream);
// The scores need the visibility bound: fp32 per row per complete block.
std::size_t fn_indexer_select_scratch_bytes(const Geometry &g, int32_t rows, int32_t max_visible);

// [S2] One QSA layer's attention sublayer: q/gate, k, v projections, q/k
// norms, rope on the first rotary_dim of each head, the new K/V appended to
// the lanes' pages in the load's KV format, attention over `selection`'s
// tokens -- or causally over every visible token when selection.dense
// (GQA q_heads / kv_heads; hq-e8-2b rows decoded into a BF16 scratch of at
// most selection_width rows per lane first -- coordinator, 2026-10-05) --
// times sigmoid(gate), o_proj. x, y: BF16 [rows][hidden].
int32_t fn_qsa_attention(const Context &ctx, int32_t attn_ordinal, const QsaWeights &w,
                         const Batch &batch, const void *x, const Selection &selection, void *y,
                         ninfer::DeviceArena &scratch, cudaStream_t stream);
std::size_t fn_qsa_attention_scratch_bytes(const Geometry &g, int32_t rows);

// [S4] Adds the n-gram embedding to every stream of `hidden` (BF16
// [rows][streams * hidden], before layer ngram_layer's attention mix):
// `rows_int4` holds each token's ngram_heads table rows (layout.md 7.1,
// ngram_row_bytes each, [rows][ngram_heads][ngram_row_bytes]) as the host
// gathered them (ignis_core::ngram::GatherPlan), already on the device.
// Updates the lanes' n-gram conv state.
int32_t fn_ngram_add(const Context &ctx, const NgramWeights &w, const Batch &batch,
                     const void *rows_int4, void *hidden, ninfer::DeviceArena &scratch,
                     cudaStream_t stream);
std::size_t fn_ngram_add_scratch_bytes(const Geometry &g, int32_t rows);

// ---------------------------------------------------------------------------
// Expert residency (spec 03, resid) is a leaf object of its own, declared in
// resid's kernel/include/ignis_residency.h, not here. Agreed with resid
// (2026-10-05):
// - `ignis_residency_create` / `ignis_residency_free`: it owns the pinned host
//   pool, the K-class pools, the staging ring, the tables and its own prefetch
//   stream and events; the Flash-Next model holds the pointer and frees it in
//   its drop path (no singleton).
// - Per layer, between the router and the expert op, on the compute stream:
//   `ignis_residency_step(r, layer, phase, ids, tokens, lookahead_logits,
//   stream)` -- `ids` the router's [rows][experts_per_token]; `lookahead_logits`
//   the fp32 [rows][experts] logits of the NEXT layer's router run on this
//   layer's MoE input (ignis_moe_router's `logits` output), or null at the last
//   layer; residency ranks its own top-W from them. It forks and joins its
//   prefetch stream inside (capture-safe).
// - `ignis_residency_slot_table(r, layer)`: the layer's ignis_moe_slot table,
//   what ignis_moe_experts_* read.
// - `ignis_residency_plan_bytes`: its plan lines (ADR 0030).

// [S1, kernel/src/flash_next/errors.cu] The message of the most recent failed
// call on this thread; every entry point above reports through fn_set_error.
void fn_set_error(std::string message);
const char *fn_last_error();

}  // namespace ignis::flash_next
