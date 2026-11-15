/* ignis kernel leaf: the Qwen 3.8-27B MTP head's proposal readout -- a
 * test-only measurement seam (GitHub #620, spec #618).
 *
 * Beside the program, not in it: no serving path reaches these entry points
 * (crates/core/tests/replay_seams_test_only.rs holds the Rust side to that).
 * They read what the head proposes at a sequence's frontier and score a hidden
 * through the load's output head, so a test can hold the head to a
 * double-precision recomputation from the stored weights.
 *
 * The head's first draft at a sequence's frontier is its row at the
 * sequence's last position, built from the continuation hidden and the
 * pending token (kernel/src/mtp_head.h), scored by the target's output head;
 * the draft is that row's argmax. Computing the row writes its key and value
 * at the head's frontier -- the one position of the head's layer no committed
 * state holds yet, which the sequence's next step writes again from the same
 * inputs -- and nothing else.
 *
 * GitHub #626: the chain readout (`ignis_mtp_readout_chain`, below) reads
 * every draft the head's chain proposes at the frontier, with each step's
 * scores: the replay of recorded agent turns under MTP stands on it, beside
 * the teacher-forced round.
 */
#ifndef IGNIS_MTP_READOUT_H
#define IGNIS_MTP_READOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ignis_model;
struct ignis_seq_pool;
struct ignis_seq;

struct ignis_mtp_readout {
  uint32_t size; /* sizeof(struct ignis_mtp_readout) */
  /* The head's first draft: the argmax of its scores (the lowest id on a
   * tie, as the vendored argmax takes it). */
  int32_t out_draft;
  /* Host F32 [vocab], or NULL: the first draft's scores, the target's
   * output head over the head's hidden, promoted from BF16. */
  float *out_scores;
  /* Host BF16 [hidden], or NULL: the head's hidden at the frontier, after
   * the head's own final norm -- what the output head scores. */
  uint16_t *out_head_hidden;
  /* Host BF16 [hidden], or NULL: the sequence's continuation hidden, the
   * target's post-final-norm hidden at its last position. */
  uint16_t *out_continuation;
};
#ifdef __cplusplus
static_assert(sizeof(struct ignis_mtp_readout) == 32,
              "ignis_mtp_readout drifted from crates/core/src/mtp_readout.rs IgnisMtpReadout");
#endif

/* The head's first draft at `seq`'s frontier, eagerly. Refused (-1, see
 * ignis_mtp_readout_last_error) without the MTP head, on a pool of another
 * drafter, and for a sequence with no pending token or no position yet. */
int32_t ignis_mtp_readout(struct ignis_model *model, struct ignis_seq_pool *pool, struct ignis_seq *seq,
                          struct ignis_mtp_readout *readout);

/* `hidden` (host BF16 [hidden]) scored by the load's output head into
 * `out_scores` (host F32 [vocab]): the scores a head hidden of these bits
 * gets, on the load's own head -- the 16 GB artifact's trellis head, which no
 * host code decodes, included. Refused without the MTP head. */
int32_t ignis_mtp_readout_scores(struct ignis_model *model, const uint16_t *hidden, float *out_scores);

/* The head's attention state of `seq` at positions [first_position,
 * first_position + count): each position's key and value as the head's layer
 * stores them (after its key norm and rotation), host BF16
 * `[count][kv_heads][head_dim]` into `out_keys` and `out_values`. A BF16 pool
 * only, and only positions the head holds (below the head's frontier, one
 * behind the sequence's own); a sequence that claims a shared prefix is read
 * through its logical pages, the shared ones then its own, as the KV capture
 * seam it reads through reads them (kernel/include/ignis_kv_capture.h). */
int32_t ignis_mtp_readout_state(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                                int32_t first_position, int32_t count, uint16_t *out_keys, uint16_t *out_values);

/* GitHub #621: under hq-e8-2b, the validity words of `seq`'s residual
 * window rings, `IGNIS_MTP_READOUT_RING_WORDS` host U32 each: the target's
 * (every target attention layer's, `out_target_words`) and the MTP head's own
 * (`out_head_words`), bit `r % 32` of word `r / 32` ring slot `r`: the words
 * a snapshot carries at the end of the residual window section (#622).
 * Refused without the head or under BF16 KV. */
#define IGNIS_MTP_READOUT_RING_WORDS 16
int32_t ignis_mtp_readout_ring(const struct ignis_seq_pool *pool, const struct ignis_seq *seq,
                               uint32_t *out_target_words, uint32_t *out_head_words);

/* GitHub #621: the last verify round's target post-final-norm hidden of
 * every column, for its `lanes` lanes at the load's window `k`: host BF16
 * `[lanes][k + 1][hidden]`, lane b's column j the hidden at its frontier plus
 * j -- the pass's own normalization of the residual it kept, the hiddens the
 * round's commit built the head's rows from. Read before the next round or
 * prefill. Refused without the MTP head or a draft window, and past the
 * load's lanes. */
int32_t ignis_mtp_readout_round_hidden(struct ignis_model *model, uint32_t lanes, uint16_t *out_hidden);

/* GitHub #626: the head's chain at a sequence's frontier, read without
 * running a round. `ignis_mtp_readout_chain` stages `seq` as lane 0 of a
 * width-1 verify round at `extent` -- its pending token the anchor, its
 * position the base, its slot -- and runs the round's own chain
 * (kernel/src/mtp_head.h `propose`) once, eagerly: the same steps on the
 * same inputs as the chain inside the sequence's next width-1 round at that
 * extent, so the same drafts, each the argmax of its step's scores (the
 * lowest id on a tie). Only where each step's scores land differs: its own
 * block of the readout's scratch, not the round's logits, which it leaves
 * as the last round left them. It copies out
 *
 *   out_drafts  I32 [extent]          the chain's drafts, in order
 *   out_scores  BF16 [extent][vocab]  each step's scores, the target's output
 *                                     head over the head's hidden at the step
 *
 * `window` is the caller's draft window and must be the load's; `extent` is
 * 1 .. window, and the steps past it are not read (the chain masks them).
 *
 * What it writes: the round's staging buffers it stages (anchors, base
 * positions, extents, the lane's slot, the drafts), which every round
 * stages again before use, and the head's rows at positions [position - 1,
 * position - 1 + extent): the head's frontier and the rows past it, which no
 * committed state holds and every later step writes before it reads them.
 * Under hq-e8-2b those appends also overwrite rows of the head's residual
 * window and set their ring bits, which older keys inside the window still
 * name, so the leaf saves the slot's head residual planes and head ring
 * words before the chain and puts them back after it: the next step reads
 * the window it would have read with no readout. The target's state is read
 * only.
 *
 * Refused (-1, see ignis_mtp_readout_last_error) without the MTP head, on a
 * pool of another drafter, for a sequence of another pool or with no
 * pending token or no position, a window not the load's, an extent of 0 or
 * past the window or one that runs past the sequence's capacity, and a size
 * this leaf does not recognize. Synchronous; not thread-safe with any other
 * call on the model. */
struct ignis_mtp_chain_readout {
  uint32_t size;   /* sizeof(struct ignis_mtp_chain_readout) */
  uint32_t window; /* the load's draft window `k` */
  uint32_t extent; /* the chain's steps read, 1 .. k */
  uint32_t reserved;
  int32_t *out_drafts;  /* host I32 [extent] */
  uint16_t *out_scores; /* host BF16 [extent][vocab] */
};
#ifdef __cplusplus
static_assert(sizeof(struct ignis_mtp_chain_readout) == 32 && offsetof(struct ignis_mtp_chain_readout, extent) == 8 &&
                  offsetof(struct ignis_mtp_chain_readout, out_drafts) == 16 &&
                  offsetof(struct ignis_mtp_chain_readout, out_scores) == 24,
              "ignis_mtp_chain_readout drifted from crates/core/src/mtp_readout.rs IgnisMtpChainReadout");
#endif

int32_t ignis_mtp_readout_chain(struct ignis_model *model, struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                const struct ignis_mtp_chain_readout *readout);

/* The message from the most recent failing call on this thread; never NULL. */
const char *ignis_mtp_readout_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_MTP_READOUT_H */
