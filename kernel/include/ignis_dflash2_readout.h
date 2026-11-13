/* ignis kernel leaf: the DFlash2 proposal readout -- a test-only measurement
 * seam (GitHub #267, docs/specs/runtime/08-dflash2-sampled-acceptance.md).
 *
 * NOT part of the serving path, and no serving code calls it. It exists so a
 * replay of recorded agent turns can read, at any sequence state, what the
 * drafter would propose if a verify round started there -- without running
 * the round, which would commit what the target samples instead of the
 * recorded text. The precedent is `ignis_prefill_options::out_span_logits`:
 * a readout beside the program, never a second way to generate.
 *
 * `ignis_dflash2_readout` runs the drafter's forward
 * (`ignis_dflash2_propose`, kernel/src/dflash2_drafter.h) for one sequence,
 * eagerly, at width 1: the query block is the sequence's pending token and
 * `extent` mask tokens at its frontier `position + j`, over its own window.
 * That is the forward the sequence's next verify round runs at `extent` --
 * the same ops on the same inputs, so the same drafts a width-1 round would
 * verify. `window` is the caller's draft window, which sizes its buffers: it
 * must be the load's, since the leaf writes the load's. It copies out:
 *
 *   out_drafts            I32 [k]         the drafts the round verifies: the
 *                                         selector walk's, or under the sampled
 *                                         draft a sampling lane's drawn ones
 *   out_candidate_ids     I32 [k][16]     each draft column's top-16 token ids
 *   out_candidate_logits  F32 [k][16]     their proposal-head logits (the
 *                                         selector's unary term), same order
 *   out_scores            F32 [k][16][16] the selector's transition scores:
 *                                         `[s][pred][succ]`, the score of
 *                                         successor candidate `succ` at column
 *                                         `s` after predecessor candidate
 *                                         `pred` of column `s - 1` (every
 *                                         `pred` row of column 0 follows the
 *                                         anchor, so they are equal)
 *   out_drafter_logits    BF16 [k][vocab] the proposal head's logits over the
 *                                         whole vocabulary, or NULL to skip
 *   out_q                 F32 [k][16]     GitHub #577: the drafter's distribution
 *                                         q over each column's candidates that
 *                                         its draft was drawn from (one-hot on
 *                                         the walk's draft where nothing is
 *                                         drawn), or NULL to skip
 *
 * GitHub #577: `sampling` is the lane's sampling, as the round would take
 * it (NULL: a greedy lane). Under the sampled draft (ADR 0090) a sampling
 * lane draws its drafts as the round does -- from the drafter's distribution
 * at its temperature times the load's factor, with the uniforms of its seed
 * at its frontier -- so the drafts read are the drafts that round verifies,
 * and `out_q` names the distribution of each. Under the fixed draft, and for
 * a greedy lane, the drafts are the walk's and q is one-hot on them.
 *
 * `k` is the load's draft window. Columns past `extent` are still computed
 * (the forward always runs `k` draft columns; the block's masked tail
 * repeats its last valid position) and copied out as they are.
 *
 * It writes nothing any sequence reads: the round's staging buffers it fills
 * (anchors, base positions, valid columns, the lane's slot, the drafts, and
 * under the sampled draft the lane's sampling config and its drawn q) are
 * restaged by every round before use, and the drafter's scratch holds
 * nothing across calls. The sequence's window, KV and frontiers are read
 * only. A sequence carrying an extent-0 round's anchor taps (GitHub #157,
 * `ignis_seq::dflash2_pending`) is refused: its window is one column behind
 * what a round would read, and closing that gap would write the window.
 * Likewise a sequence whose window frontier trails its program frontier.
 *
 * Returns 0, or -1 with `ignis_dflash2_readout_last_error()` set: a null
 * argument, a size this leaf does not recognize (of the readout or of its
 * `sampling`), a load without the drafter,
 * a `window` that is not the load's, a pool built for another backend, a
 * sequence from another pool, an `extent` past the window, a sequence with
 * no pending token, or a kernel error. Synchronous: the outputs are
 * filled when it returns. Not thread-safe with any other call on the model.
 */
#ifndef IGNIS_DFLASH2_READOUT_H
#define IGNIS_DFLASH2_READOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ignis_model;
struct ignis_seq_pool;
struct ignis_seq;
struct ignis_sampling_params; /* kernel/include/ignis_step.h */

/* The selector's candidates per draft column (`kDflash2SelectorTopK`). */
#define IGNIS_DFLASH2_READOUT_TOP_K 16

struct ignis_dflash2_readout {
  uint32_t size;   /* sizeof(struct ignis_dflash2_readout) */
  uint32_t window; /* the load's draft window `k`, which sizes the outputs */
  uint32_t extent;
  int32_t *out_drafts;
  int32_t *out_candidate_ids;
  float *out_candidate_logits;
  float *out_scores;
  uint16_t *out_drafter_logits; /* or NULL */
  /* GitHub #577: appended after every field above. */
  const struct ignis_sampling_params *sampling; /* the lane's, or NULL: greedy */
  float *out_q;                                 /* or NULL */
};

#ifdef __cplusplus
/* Beside the `size` check, both sides pin the layout (here, and
 * crates/core/src/dflash2_readout.rs's mirror test). */
static_assert(sizeof(struct ignis_dflash2_readout) == 72 &&
                  offsetof(struct ignis_dflash2_readout, window) == 4 &&
                  offsetof(struct ignis_dflash2_readout, out_drafts) == 16 &&
                  offsetof(struct ignis_dflash2_readout, out_drafter_logits) == 48 &&
                  offsetof(struct ignis_dflash2_readout, sampling) == 56 &&
                  offsetof(struct ignis_dflash2_readout, out_q) == 64,
              "ignis_dflash2_readout drifted from crates/core/src/dflash2_readout.rs IgnisDflash2Readout");
#endif

int32_t ignis_dflash2_readout(struct ignis_model *model, struct ignis_seq_pool *pool,
                              const struct ignis_seq *seq, const struct ignis_dflash2_readout *readout);

/* The most recent verify round's target logits, BF16 `[lanes][k + 1][vocab]`
 * for its first `lanes` lanes, `k` the caller's `window`, which must be the
 * load's draft window (it sizes `out_logits`): row `j` of lane `i` is the target's
 * distribution after that lane's column `j` -- its anchor, then its drafts in
 * order (columns past the lane's extent repeat the anchor at the last valid
 * position). What the round's accept read, as it read it; valid until the
 * next verify round on this model. Any load with a draft window (under the
 * drafter or verify-only), on either 27B body (GitHub #339: the round's head
 * is the trellis one on a trellis body); `lanes` at most the load's decode
 * lanes (GitHub #336), which the round's logits are sized for. Synchronous.
 * Returns 0, or -1 with `ignis_dflash2_readout_last_error()` set. */
int32_t ignis_verify_readout_logits(struct ignis_model *model, uint32_t lanes, uint32_t window,
                                    uint16_t *out_logits);

const char *ignis_dflash2_readout_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_DFLASH2_READOUT_H */
