/* ignis kernel leaf: the degenerate step ABI (ADR 0009, GitHub #54, P1-18).
 *
 * The program layer (ours, not vendored) runs the model's per-token pipeline
 * on top of the ADR 0010 vendored ops. `ignis_prefill` processes a token
 * span for one sequence starting at a position; `ignis_decode` runs one
 * round over a batch of current tokens (batch 1 today -- the array
 * parameter is present so a later batched decode round, G3, is a leaf
 * change only). Neither takes a sequence handle yet (P1-19 adds KV pages /
 * GDN slot / conv taps / position tracking); with every decoder layer
 * skipped there is no per-sequence state to own.
 *
 * `skip_layers` bypasses every decoder layer, running only embedding ->
 * final RMSNorm -> W8G32 output head -> argmax (the degenerate program this
 * ticket verifies). It is test-only: no production caller sets it to 0
 * (false) yet, because the GQA/GDN layer bodies do not exist yet
 * (P1-21/P1-22). Streams are internal to the leaf (owned by the model
 * handle, kernel/src/model_internal.h) -- no host activation pointer or
 * stream crosses this boundary.
 *
 * `ignis_program_prefill` takes a `struct ignis_prefill_options` (ADR 0016,
 * P2-02, GitHub #84): its default (chunked) route cuts a span into
 * `ignis_model_load`'s prefill-chunk-wide traversals; its per-token route
 * is the original per-token loop, retained test-only as a self-oracle.
 *
 * Rust bindings: crates/core/src/step.rs (keep 1:1).
 */
#ifndef IGNIS_STEP_H
#define IGNIS_STEP_H

#include <stdint.h>

#include "ignis_model.h"
#include "ignis_seq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Sampling parameters (greedy only, G1; temperature / top-p / top-k /
 * penalties / seed are G3). */
struct ignis_sampling_params {
  int32_t greedy; /* nonzero: argmax (the only supported mode today) */
};

/* Which internal path ignis_program_prefill takes over a span's chunks
 * (ADR 0016, P2-02, GitHub #84). Chunked is the production route: the leaf
 * cuts the span into prefill-chunk-wide traversals and synchronizes once
 * per chunk. Per-token is retained test-only -- it is the G1 loop
 * unchanged (one traversal per token, one synchronization per layer) --
 * so the same prompt prefilled both ways is a self-oracle for the chunk
 * loop, the state carry across chunk boundaries and (from P2-03/P2-04
 * onward) the multi-token kernel routes. */
enum ignis_prefill_route {
  IGNIS_PREFILL_ROUTE_CHUNKED = 0,
  IGNIS_PREFILL_ROUTE_PER_TOKEN = 1,
};

/* The compute policy every NVFP4 projection dispatches under (ADR 0016).
 * `ENGINE_DEFAULT` is whatever policy the program layer's dispatch sites
 * use today; `A16_ONLY` forces the A16 route so a test can compare routes
 * on identical inputs. Turning the engine's own default to AllowA4 is
 * P2-03 (GitHub #63) -- until then both values dispatch identically. */
enum ignis_prefill_compute_policy {
  IGNIS_PREFILL_COMPUTE_POLICY_ENGINE_DEFAULT = 0,
  IGNIS_PREFILL_COMPUTE_POLICY_A16_ONLY = 1,
};

/* Extensible per-call options for ignis_program_prefill (ADR 0016, amending
 * ADR 0009's "not an ABI change" claim). `size` must be
 * `sizeof(struct ignis_prefill_options)`; a NULL options pointer means the
 * production defaults (chunked route, engine default policy). A future
 * phase (G3 sampling, G4 snapshot controls) appends fields and bumps a new
 * recognized size -- it never adds a parameter or a `_ex` entry point. */
struct ignis_prefill_options {
  uint32_t size;           /* sizeof(struct ignis_prefill_options) */
  int32_t route;           /* enum ignis_prefill_route */
  int32_t compute_policy;  /* enum ignis_prefill_compute_policy */
};

/* Prefill a token span for one sequence starting at `start_position`
 * (unread while `skip_layers` is set -- no RoPE/KV runs in the degenerate
 * program). Produces the span's last position's argmax id in
 * `*out_token_id` and, if `out_logits` is non-null, that position's full
 * vocab-length logits (promoted from the device's BF16 storage to host
 * `float`, caller-owned buffer of at least `vocab` entries). Returns 0 on
 * success, -1 on a null/invalid argument or a kernel error (see
 * ignis_step_last_error). */
int32_t ignis_prefill(struct ignis_model *model, const int32_t *token_ids, uint64_t num_tokens,
                       uint64_t start_position, int32_t skip_layers,
                       const struct ignis_sampling_params *sampling, int32_t *out_token_id,
                       float *out_logits);

/* One decode round over a batch of current tokens (`batch_size` entries, one
 * per sequence), producing one argmax id per sequence in `out_token_ids`
 * (`batch_size` entries) and, if `out_logits` is non-null, `batch_size *
 * vocab` logits (sequence `i`'s row at `out_logits + i * vocab`). Returns 0
 * on success, -1 on a null/invalid argument or a kernel error. */
int32_t ignis_decode(struct ignis_model *model, const int32_t *token_ids, uint64_t batch_size,
                      int32_t skip_layers, const struct ignis_sampling_params *sampling,
                      int32_t *out_token_ids, float *out_logits);

/* The message from the most recent failing ignis_prefill/ignis_decode call
 * on this thread (thread-local; overwritten by the next call; empty string
 * if none failed yet). Never NULL. */
const char *ignis_step_last_error(void);

/* Runtime counters for the real program entry points.  `kernel_count` is
 * the number of program-layer dispatches in the latest step: the leaf's
 * stable, graph-independent dispatch counter until G3 groups those calls in
 * CUDA graphs. */
struct ignis_program_stats {
  uint64_t vram_bytes;
  uint64_t last_step_micros;
  uint64_t kernel_count;
};

/* Run the complete 64-layer program for every token in a prompt span.  The
 * span starts exactly at `start_position`; it advances the sequence's KV,
 * GDN, convolution and position state once per token.  No token is emitted:
 * the greedy successor is retained on `seq` for ignis_program_decode.
 *
 * `options` selects the route and compute policy (ADR 0016, P2-02, GitHub
 * #84); NULL means the production defaults (chunked route). A non-null
 * `options->size` that this leaf does not recognize is rejected. Under the
 * default chunked route the span is cut into `ignis_model_load`'s
 * `prefill_chunk_tokens`-wide chunks internally -- the caller never needs
 * to know the chunk width -- and the leaf synchronizes its stream once per
 * chunk, not once per layer; a chunk that fails is reported with its span
 * offset and leaves `seq` at its pre-chunk position (KV pages, GDN slot,
 * conv taps, position and pending token all unchanged), so the caller's
 * retry path stays correct. The per-token route (test-only) is today's
 * unchanged per-token loop. After a chunked prefill, `seq`'s state is
 * exactly what the per-token route would have left.
 *
 * If `out_logits` is non-null, it receives the span's *last* position's
 * full vocab-length logits (promoted from the device's BF16 storage to host
 * `float`, caller-owned buffer of at least `vocab` entries) -- the same
 * position whose argmax becomes the successor `ignis_program_decode` will
 * first emit. Debug-only (GitHub #72: confirming a near-tie argmax flip on
 * the canary suite needs the real logits, not just the winning id); every
 * earlier position in the span still runs argmax-only, at no extra cost. */
int32_t ignis_program_prefill(struct ignis_model *model, struct ignis_seq_pool *pool,
                              struct ignis_seq *seq, const int32_t *token_ids,
                              uint64_t num_tokens, uint64_t start_position,
                              const struct ignis_sampling_params *sampling,
                              const struct ignis_prefill_options *options,
                              float *out_logits);

/* Complete one greedy decode round for a batch of sequence handles.  Each
 * output is the successor made ready by prefill/the prior round; that token
 * is consumed before return to make the next successor ready. */
int32_t ignis_program_decode(struct ignis_model *model, struct ignis_seq_pool *pool,
                             struct ignis_seq *const *sequences, uint64_t batch_size,
                             const struct ignis_sampling_params *sampling,
                             int32_t *out_token_ids);

/* Read program counters and allocated device footprint. */
int32_t ignis_program_stats(const struct ignis_model *model,
                            const struct ignis_seq_pool *pool,
                            struct ignis_program_stats *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_STEP_H */
