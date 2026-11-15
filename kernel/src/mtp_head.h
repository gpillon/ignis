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
// continuation hidden. A prefill's first chunk has no row at -1.

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

}  // namespace ignis::mtp
