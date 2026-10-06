// ignis kernel leaf -- the Flash-Next MTP head's step (spec flash-next/07 phase D, GitHub #307;
// OURS, ADR 0043): the checkpoint's own draft layer, run on the trunk's state.
//
// One step over a Batch of entries, entry r built from a 4-stream stack S_r and a token t_r and
// stored at the batch's position for row r (the KV index of the stack's own position: the entry
// built from (S_p, t[p+1]) sits at p, so the head's frontier is the trunk's):
//
//   e   = RMSNorm(embed_tokens(t), pre_fc_norm_embedding)                     [hidden]
//   n_s = RMSNorm_s(S[s], pre_fc_norm_hidden[s])            s = 0..streams-1   (grouped, conv. a)
//   X_s = bf16(fc_hidden(n_s) + fc_embedding(e))            the two BF16 linears summed in fp32
//   X  -> one Flash-Next decoder layer (QSA on the head's own attention section + MoE whose
//         experts are all resident) -> S'    [streams * hidden], the block's pre-mixer stack
//   logits = lm_head(hyper_connection_mixer(S'))            the trunk's head
//
// Every RMSNorm is x * rsqrt(mean(x^2) + eps) * (1 + w) in fp32, rounded once (the prototype's
// `rms`, tools/flash-next-mtp/mtp.py). A chained draft feeds S' back as the next step's stack
// (convention chain a). The layer is the trunk's layer sequence (flash_next_internal.h) with the
// experts' slot table fixed at load, so no residency step runs.

#pragma once

#include "bind.h"
#include "flash_next_internal.h"

#include "ignis_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::flash_next::mtp {

// Where a step reads and writes, every buffer [rows] of the program's activations.
struct Buffers {
  void *residual = nullptr;      // BF16 [rows][streams * hidden]: the stacks in, S' out
  void *x = nullptr;             // BF16 [rows][hidden]
  void *y = nullptr;             // BF16 [rows][hidden]
  float *inj = nullptr;          // fp32 [rows][streams]
  // The MoE block's (ignis_moe.h).
  ignis_moe_workspace workspace{};
  int64_t *acc = nullptr;
  int32_t *router_ids = nullptr;
  float *router_weights = nullptr;
  float *router_logits = nullptr;
  void *shared_h = nullptr;
  float *shared_out = nullptr;
};

// The step's own scratch beside the layer's ops: the embedding's stacks, the normed inputs and
// the two projections.
std::size_t combine_scratch_bytes(const Geometry &g, int32_t rows);

// X = the head's input for `batch.rows()` entries: `tokens` DEVICE [rows], the stacks in
// `b.residual` (overwritten by X).
int32_t combine(const Geometry &g, const MtpWeights &w, const Linear &embed, const int32_t *tokens,
                const Batch &batch, const Buffers &b, ninfer::DeviceArena &scratch, cudaStream_t stream);

// The head's layer over X in `b.residual`, in place, on attention section `attention_ordinal`.
// `experts`: DEVICE [experts * 2] slots, every projection resident. Decode route up to the
// workspace's decode tokens, the prefill route past them.
int32_t layer(const Context &ctx, int32_t attention_ordinal, const MtpWeights &w, const ignis_moe_slot *experts,
              const Batch &batch, const Buffers &b, ninfer::DeviceArena &scratch, cudaStream_t stream);

// The argmax of each of `rows` BF16 logits rows into `out` DEVICE [rows] (the lowest id among
// equal maxima: the draft a greedy proposal names).
int32_t argmax(const Geometry &g, const void *logits, int32_t rows, int32_t *out, cudaStream_t stream);

// After an alignment over `columns` columns per lane: each lane's first draft -- the pick at its
// last committed column, commit[l] - 1 -- into drafts[l][0] and chain_tokens[l], and that column's
// S' (`residual`'s row) into chain_stack[l]. `width` is the drafts' row stride (the window).
int32_t first_drafts(const Geometry &g, const int32_t *picks, const void *residual, const int32_t *commit,
                     int32_t lanes, int32_t columns, int32_t width, int32_t *drafts, int32_t *chain_tokens,
                     void *chain_stack, cudaStream_t stream);

// drafts[l][step] = chain_tokens[l] for every lane.
int32_t append_drafts(const int32_t *chain_tokens, int32_t lanes, int32_t width, int32_t step, int32_t *drafts,
                      cudaStream_t stream);

}  // namespace ignis::flash_next::mtp
