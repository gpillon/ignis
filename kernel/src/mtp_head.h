#pragma once
// GitHub #620 (spec #618): the Qwen 3.8-27B's MTP head -- OURS.
//
// The checkpoint's one MTP layer, bound from the container's 12 `mtp/` objects
// (model_internal.h `MtpHeadWeights`). A head row at position `p` takes the
// target's post-final-norm hidden `h_p` and the token `x_{p+1}` after it:
//
//   e = embed(x_{p+1})                       the target's token embedding
//   x = W_fc [RMSNorm(e; embedding_norm); RMSNorm(h_p; hidden_norm)]
//   one gated full-attention layer over x    24 query / 4 KV heads of 256,
//     (input norm, the fused projection's   q/k norm, the target's rotary table
//     rows query | key | gate | value)      at position p, sigmoid output gate
//   a dense SwiGLU MLP (17,408)
//   RMSNorm(x; final_norm)                  the head's hidden, which the
//                                           target's output head scores
//
// Every RMSNorm has the 1 + w gain the target's have. The row's key and value
// land in the head's own attention layer (the pool's last, ignis_seq_internal.h)
// at position `p`, and its attention reads the head's rows 0..p. The head's
// matrices run on the op layer every 27B matrix runs on (the W8 `ops::linear`,
// whose registry carries the head's five geometries); the fused projections
// are not registered at these shapes, so the layer composes linear, a split,
// residual adds and the SwiGLU. Nothing here is the reference engine's MTP.
//
// Who writes which row: the entry at `p` is built when `x_{p+1}` is consumed,
// so a step that consumes positions `[a, b)` writes the head's rows
// `[a - 1, b - 1)` -- the first from the sequence's continuation hidden (the
// target's hidden at `a - 1`, kept in the pool), the rest from the step's own
// hiddens -- and leaves the target's hidden at `b - 1` as the next
// continuation hidden. A prefill's first chunk has no row at -1. A verify
// round (GitHub #621) is such a step over its committed run; its chain's own
// rows are scratch (below, "the head in a verify round").

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

struct ignis_model;
struct ignis_seq_pool;
struct ignis_seq;

namespace ignis::mtp {

// The head's geometry: the 27B's attention layer and MLP.
inline constexpr std::int32_t kHeadDim = 256;
inline constexpr std::int32_t kQueryHeads = 24;
inline constexpr std::int32_t kKvHeads = 4;
inline constexpr std::int32_t kQueryWidth = kHeadDim * kQueryHeads;  // 6144
inline constexpr std::int32_t kKvWidth = kHeadDim * kKvHeads;        // 1024
inline constexpr std::int32_t kProjectionRows = 2 * kQueryWidth + 2 * kKvWidth;  // 14336
inline constexpr std::int32_t kFfn = 17408;

// The most rows one head call takes in a prefill chunk: a chunk's rows run in
// calls of this many, so what the head adds to the prefill scratch does not
// grow with the load's chunk width.
inline constexpr std::int32_t kPrefillRows = 128;

// The bytes of every buffer one head call of `rows` rows bumps from its arena
// (each rounded to the arena's alignment) beside its attention's workspace --
// the stem, the layer, the MLP and the head's hidden. The attention workspace
// is the target's GQA layers' at the same widths, which the arena already
// holds (kernel/src/model.cu).
std::size_t call_bytes(std::int64_t hidden, std::int64_t rows);

// What a prefill chunk of `chunk` tokens adds to the program scratch on an MTP
// load: one call of at most kPrefillRows rows, its input hiddens and
// embeddings, and the continuation hidden's readout room. The CPU plan
// restates it (crates/core/src/mtp_head.rs).
std::size_t prefill_scratch_bytes(std::int64_t hidden, std::int64_t chunk);

// What a round of `columns` columns adds to the decode-round scratch on an MTP
// load: one call over every column, its inputs and its positions.
std::size_t round_scratch_bytes(std::int64_t hidden, std::int64_t columns);

// The head over a prefill chunk of `tokens` columns of `seq`, which stands at
// the chunk's start: `ids` (device I32 [tokens]) are the chunk's tokens and
// `final_residual` (BF16 [hidden, tokens]) the target's last residual of each.
// Writes the head's rows the chunk licenses and the continuation hidden of the
// chunk's last column. With `span_hidden` (host BF16 [tokens][hidden], or null)
// it also copies out the target's post-final-norm hidden of every column --
// the readout `ignis_prefill_options::out_span_hidden`. Enqueue only, on the
// model's stream, from `model.scratch`. 0, or -1 with `*error` set.
int32_t prefill_chunk(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, const ninfer::Tensor &ids,
                      const ninfer::Tensor &final_residual, std::int32_t tokens, std::uint16_t *span_hidden,
                      std::string *error);

// The head over a plain decode round of `width` lanes, graph-safe: each lane's
// row at its position less one, from its continuation hidden and the token
// the round consumes (`model.decode_graph_token_ids`), its slot from
// `model.decode_graph_slots`; then each lane's continuation hidden becomes
// `normalized` (BF16 [hidden, width]: the round's post-final-norm hidden).
// Enqueue only, from `model.decode_graph_scratch`. 0, or -1 with `*error`.
int32_t decode_round(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, const ninfer::Tensor &normalized,
                     std::string *error);

// The head's row at `seq`'s last position, eagerly and at width 1 -- the
// first draft's input -- from its continuation hidden and the token
// `next_token` after it: writes that row at the head's frontier (a position
// no committed state holds yet, which the sequence's next step writes again)
// and leaves the head's hidden in `out_hidden` (BF16 [hidden, 1]). For the
// proposal readout (kernel/include/ignis_mtp_readout.h). 0, or -1 with
// `*error`.
int32_t frontier_row(ignis_model &model, ignis_seq_pool &pool, ignis_seq &seq, std::int32_t next_token,
                     ninfer::DeviceArena &arena, ninfer::Tensor &out_hidden, std::string *error);

// GitHub #621: the head in a verify round. A lane at frontier `P` (its
// anchor, the pending token, at `P`) with extent `e` and a committed run of
// `c` tokens (1 <= c <= e + 1):
//
//   the chain  (`propose`, inside the round's pass and its graph): step 0 the
//              head's row at P - 1 from the continuation hidden and the
//              anchor, step s > 0 its row at P - 1 + s from the head's own
//              hidden at step s - 1 and the draft d_s that step chose; step s
//              chooses d_{s+1}, the argmax of the target's output head over
//              the head's hidden (the fixed draft, at every temperature). A
//              lane needs steps 0 .. e - 1; a step past them writes nothing
//              (a lane at extent 0 none), so no row lands past the lane's
//              budget or its context;
//   the commit (`commit_rows`, after the cut, eager): the rows the run
//              licenses, [P - 1, P + c - 1), each again from what the
//              sequence consumed -- the continuation hidden and the anchor,
//              then the target's post-final-norm hidden of each committed
//              column and the token after it -- so no row the chain built
//              from the head's own hidden stays below the frontier;
//   the continuation hidden (`commit_continuation`, the round's last device
//              step): the target's post-final-norm hidden of column c - 1.
//
// Rows the chain wrote past the commit are past the head's frontier (P + c
// - 1), never read before a later step writes them again; under hq-e8-2b
// their ring bits are cleared (`round_rows`, kernel/src/step.cu).

// The head's rows a verify round touches, for a lane at frontier `base`
// with extent `extent` that commits `committed` tokens: the chain's are
// [first, chain_end), the commit's [first, committed_end). Host arithmetic.
struct RoundRows {
  std::uint64_t first = 0;
  std::uint64_t chain_end = 0;
  std::uint64_t committed_end = 0;
};
RoundRows round_rows(std::uint64_t base, std::uint32_t extent, std::uint32_t committed);

// The chain over the round's `width` lanes, from the verify round's staging
// (`IgnisVerifyRound`: anchors, base positions, extents; the round's slots in
// `model.decode_graph_slots`): every lane's drafts into `verify.drafts`
// (I32 [k, width], lane-major), its steps at width `width`. Graph-safe;
// scratch from `model.decode_graph_scratch` in its own scope, the step's
// scores in `verify.logits` (which the pass writes again after it). Enqueue
// only. 0, or -1 with `*error`.
//
// GitHub #626: with `step_scores` (device BF16 `[window][vocab, width]`;
// null in the round and in every captured graph) step s scores into its own
// `[vocab, width]` block instead, so every step's scores outlive the next --
// the proposal readout's (kernel/include/ignis_mtp_readout.h
// `ignis_mtp_readout_chain`): the same kernels on the same inputs, only the
// scores' address moves, and the round's logits are left as they were.
int32_t propose(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, std::string *error,
                std::uint16_t *step_scores = nullptr);

// The commit's rows, eager, after the round's pass: lane b's rows at
// `positions[b * (k + 1) + j]` (host I32 [k + 1, width]: base_b - 1 + min(j,
// committed_b - 1)), its first `committed[b]` (host I32 [width]) columns
// valid, from the round's own `verify.hidden` and `verify.verify_ids`. Both
// host arrays stay alive until the stream synchronizes. Only the rows' keys
// and values: the head's hidden at a committed row is no draft's. 0, or -1
// with `*error`.
int32_t commit_rows(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, const std::int32_t *positions,
                    const std::int32_t *committed, std::string *error);

// Each lane's continuation hidden from the run's last committed column,
// `verify.selected_hidden` (the pre-norm residual the commit selected),
// normalized as the pass normalized it. 0, or -1 with `*error`.
int32_t commit_continuation(ignis_model &model, ignis_seq_pool &pool, std::uint32_t width, std::string *error);

// The round's three small device steps, exposed for their op test
// (kernel/tests/test_mtp_round.cu). Enqueue only; throw on a launch error.
namespace detail {

// Step `step` of the chain: lane b's row `positions[b]` = base[b] - 1 + step
// and `valid[b]` = 1 when the lane needs that step (step < extents[b]); 0
// and 0 (the masked form's empty row) otherwise.
void chain_rows(const std::int32_t *base_positions, const std::int32_t *extents, std::int32_t step,
                std::int32_t *positions, std::int32_t *valid, std::int32_t batch, cudaStream_t stream);

// drafts[b * window + step] = tokens[b].
void write_drafts(const std::int32_t *tokens, std::int32_t *drafts, std::int32_t window, std::int32_t step,
                  std::int32_t batch, cudaStream_t stream);

// The commit's input hiddens, BF16 [hidden, lane_columns, batch]: lane b's
// column 0 is its continuation hidden (`table` column `slots[b]`), column j >
// 0 the round's normalized column j - 1 (`normalized`, the same shape).
void commit_hiddens(const std::uint16_t *normalized, const std::uint16_t *table, const std::int32_t *slots,
                    std::uint16_t *out, std::int32_t hidden, std::int32_t lane_columns, std::int32_t batch,
                    cudaStream_t stream);

}  // namespace detail

}  // namespace ignis::mtp
