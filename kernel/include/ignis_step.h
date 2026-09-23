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
 * `ignis_sampling_params` grew real sampling (P3-03, GitHub #99): the
 * size-prefixed struct (ADR 0016) now carries temperature, top-k, top-p,
 * presence/frequency penalties and a seed alongside the original `greedy`
 * flag. Device-side sampling (`ninfer::ops::sample`, vendored per ADR 0010)
 * replaces `ninfer::ops::argmax` everywhere a token is chosen: greedy
 * (`greedy` nonzero, or `temperature <= 0`) is bit-identical to argmax by
 * that op's own contract, so this is not a behavior change for any existing
 * caller. `ignis_program_decode` takes one `ignis_sampling_params` per
 * sequence (an array parallel to `sequences`, capped at
 * `IGNIS_DECODE_MAX_BATCH`) because lanes sharing a decode round carry
 * independent temperatures, seeds and penalty histories; the other three
 * entry points keep a single sampling pointer (one sequence, or -- for
 * `ignis_prefill`/`ignis_decode` -- one degenerate step at a time). RNG is
 * counter-based (keyed by `seed` and the sequence's own position, not by
 * batch row), so a sequence's output depends only on its own seed, never on
 * which lanes shared its round. Presence/frequency penalties read and
 * update a per-sequence, per-vocab-entry count buffer that lives in the
 * sequence handle's pool slot (`kernel/include/ignis_seq_internal.h`) --
 * there is no separate persistent RNG state to carry, since the counter-based
 * generator needs none.
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

/* Sampling parameters (ADR 0016's size-prefixed options struct; P3-03,
 * GitHub #99). `size` must be `sizeof(struct ignis_sampling_params)` -- the
 * leaf rejects a size it does not recognize, same rule as
 * `ignis_prefill_options`.
 *
 * `greedy` nonzero or `temperature <= 0` both select argmax (bit-identical
 * to G1's `ninfer::ops::argmax`, per the vendored sampler's own contract);
 * every other field is then unread. Otherwise: `top_k` <= 0 or > 19 keeps
 * the sampler's own top-20 cap; `top_p` disables at >= 1; `presence_penalty`
 * / `frequency_penalty` read this sequence's per-vocab-entry occurrence
 * count (zero when the leaf has no penalty state for the call, e.g. the
 * degenerate G1 entry points); `seed` plus the sequence's own logical
 * position key the counter-based RNG draw, independent of which other
 * sequences share the round or their arrival order. `min_p` is not
 * exposed at this ABI (disabled) -- not one of this ticket's six
 * parameters.
 *
 * P6-06 (GitHub #242, ADR 0034) appends the **permitted token set**, the
 * same way. `permitted_count` of 0 is today's unconstrained draw, bit for
 * bit. A nonzero count restricts this lane's draw to `permitted_ids` — at
 * most `IGNIS_MAX_PERMITTED_TOKENS`, each a valid vocabulary id, duplicates
 * allowed and harmless — by driving every other column of the lane's logits
 * out of reach before the sampler runs. It therefore *composes* with the
 * parameters above rather than replacing them: `greedy` takes the set's
 * argmax, a temperature draw is a draw from the set, `top_k`/`top_p` cut the
 * set further, and the penalties still read the sequence's own counts. No
 * logits cross this ABI for it, which is the whole reason it is here and not
 * in the host (ADR 0034).
 *
 * `ignis_program_prefill` and `ignis_program_decode` both read it, and they
 * have to: a decode round returns the successor the *previous* call made
 * ready, so the first token of a constrained run is the one the prefill
 * itself draws. A run of K constrained tokens is one prefill carrying the
 * first set and K-1 rounds carrying the rest. The degenerate `ignis_prefill`
 * / `ignis_decode` entry points reject a nonzero count rather than ignoring
 * it, because a constraint silently dropped is a wrong answer that looks
 * like a right one.
 *
 * P5-04 (GitHub #153) appends the verify round's per-lane inputs (ADR 0016:
 * a field append and a size bump, never a parameter). Both are read only by
 * a decode call whose `ignis_decode_options::speculative_window` is nonzero;
 * every other entry point ignores them. `remaining_tokens` is the lane's
 * remaining generation budget in tokens, counting the anchor this round
 * emits -- the round proposes at most `remaining_tokens - 1` drafts -- and 0
 * means no budget. `stop_ids` (caller-owned, `stop_id_count` entries, valid
 * for the call) cut the committed run at the first stop id inclusive: the
 * sequence's state never runs past the text the caller emits. */
/* The most ids one lane's permitted set may carry (P6-06, GitHub #242).
 * Ten digits and a handful of forced literals is what the constrained
 * decode needs; the cap is what lets the set live in a fixed per-lane
 * staging row instead of an allocation per round, and a larger set is
 * rejected rather than truncated. */
#define IGNIS_MAX_PERMITTED_TOKENS 32

struct ignis_sampling_params {
  uint32_t size;    /* sizeof(struct ignis_sampling_params) */
  int32_t greedy;   /* nonzero: argmax, ignoring every field below */
  float temperature;
  int32_t top_k;    /* an ignis extension over the OpenAI-compatible surface */
  float top_p;
  float presence_penalty;
  float frequency_penalty;
  uint64_t seed;
  uint32_t remaining_tokens; /* P5-04: this lane's budget, anchor included; 0 = none */
  uint32_t stop_id_count;    /* P5-04: entries in `stop_ids` (0: no stop id) */
  const int32_t *stop_ids;   /* P5-04: caller-owned; NULL when the count is 0 */
  uint32_t permitted_count;    /* P6-06: entries in `permitted_ids` (0: unconstrained) */
  const int32_t *permitted_ids; /* P6-06: caller-owned; NULL when the count is 0 */
};

/* The largest `batch_size` `ignis_program_decode` accepts: the decode
 * round's sampling parameters and logical positions are staged in
 * model-owned device buffers at stable addresses (so a future decode CUDA
 * graph, P3-05/#102, can replay reading them), sized once at model load for
 * this many lanes -- matching the engine's own N=8 resident decode-lane
 * concurrency and the decode graphs' exact widths 1..8. */
#define IGNIS_DECODE_MAX_BATCH 8

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
struct ignis_media_embedding;

struct ignis_prefill_options {
  uint32_t size;           /* sizeof(struct ignis_prefill_options) */
  int32_t route;           /* enum ignis_prefill_route */
  int32_t compute_policy;  /* enum ignis_prefill_compute_policy */
  /* GitHub #178: a span of a multimodal prompt (fields appended and the size
   * bumped -- one recognized size, ADR 0016: no compatibility wrapper is
   * kept). NULL `mrope_positions` is a text span, whose media fields must be
   * empty. Non-NULL -- a load with vision, the
   * chunked route -- it is the span's axis-major [3, num_tokens] positions,
   * which its GQA layers rotate at (MRoPE: pair i on axis i mod 3), and
   * `rope_delta` becomes the sequence's: every later decode round rotates at
   * `position + rope_delta`, while the position stays the KV index and the
   * sampler's key. `media` (or NULL) is an embedding from ignis_media_encode:
   * its columns `media_first_column ..` replace the embedded rows at the
   * `media_column_count` span-relative, strictly increasing
   * `media_scatter_indices`. */
  const int32_t *mrope_positions;
  int32_t rope_delta;
  uint32_t media_column_count;
  const struct ignis_media_embedding *media;
  const int32_t *media_scatter_indices;
  uint32_t media_first_column;
  /* P6-06 (GitHub #242): the drawn token's probability within this call's
   * permitted set, or NULL. 0 when the call declared no set. One float, not
   * a logits row -- the first digit of a constrained run is drawn here, and
   * its confidence has to come back with it (ADR 0034). */
  float *out_permitted_prob;
  /* GitHub #260 (ADR 0038): the attention readout, or none when
   * `attention_gqa_ordinal` is negative (or `out_attention_scores` NULL) --
   * a call that asks for none allocates nothing and launches nothing new.
   * Read on the span's last chunk, at its last position: query head
   * `attention_query_head` of GQA layer `attention_gqa_ordinal`, after its
   * norm and rotary embedding, dotted with the keys at absolute positions
   * [attention_key_begin, attention_key_begin + attention_key_count) exactly
   * as that layer's attention read them -- the cache's pages under BF16, the
   * prompt route's materialized planes under hq-e8-2b -- and scaled by
   * 1/sqrt(head_dim): one pre-softmax score per key into
   * `out_attention_scores` (host, `attention_key_count` floats). No full
   * attention row and no logits row crosses. `*out_attention_read` is set to
   * 1 when every score was written from those keys and to 0 when the keys
   * were not there to read (a small-T route, which materializes none; a span
   * outside the one band the hq prompt route materializes; a span past the
   * history) -- the prefill itself still succeeds, and the caller fails the
   * question rather than answering from any other copy of the keys. */
  int32_t attention_gqa_ordinal;
  int32_t attention_query_head;
  int64_t attention_key_begin;
  int64_t attention_key_count;
  float *out_attention_scores;
  int32_t *out_attention_read;
  /* GitHub #263 (ADR 0039): the head set read beside the attention readout,
   * or none when `attention_set_count` is 0 -- and then the readout arms its
   * one layer and launches what it launched before. Head `i` is query head
   * `attention_set_query_heads[i]` of GQA layer
   * `attention_set_gqa_ordinals[i]` (at most 384 heads); each GQA layer
   * holding one, and the readout's own, is armed and read the same way, in
   * one fused launch per layer. Head `i`'s argmax key over the span, skipping
   * the `attention_excluded_count` span-relative keys of
   * `attention_excluded` (at most 32; the fallback cells), lands in
   * `out_attention_set_argmax[i]` (host, span-relative, the larger index on a
   * tie). `*out_attention_read` is 1 only when every armed layer read: a
   * partial set is never reported. Needs the readout itself armed. */
  uint32_t attention_set_count;
  const int32_t *attention_set_gqa_ordinals;
  const int32_t *attention_set_query_heads;
  uint32_t attention_excluded_count;
  const int32_t *attention_excluded;
  int32_t *out_attention_set_argmax;
};

/* The media encode step (GitHub #178): one media item's BF16 patch rows plus
 * its grid and host-computed encoder control, in; an opaque, leaf-owned,
 * device-resident embedding of `[hidden, patches / 4]` merged columns, out --
 * the 27-block vision encoder and 2x2 merger, run once per item. Host inputs
 * only, like token ids: every array is caller-owned and read during the call.
 *
 * `patches` is row-major `[t*h*w][3*2*16*16]` BF16 bits in 2x2 merge-block
 * order; `position_ids` `[2*P]` (every patch's row, then every patch's
 * column); `cu_seqlens` `[t+1]` segment bounds; `position_table_indices` /
 * `_weights` `[4*P]`, four bilinear corners per patch. `h` and `w` are even.
 *
 * The load's embedding pool (GitHub #243) holds as many embeddings at a time
 * as their columns fit. An item wider than the load's envelope is refused
 * with -1; an item that would fit an empty pool but not the free pages left
 * is refused with IGNIS_MEDIA_ENCODE_POOL_FULL, which says "release
 * something and call again" rather than "this can never work" -- the caller
 * owns the eviction policy, this side owns only the pages. The encoder runs
 * out of the load's scratch arena, which prefill steps share (GitHub #212),
 * so it is never called from inside one. Returns 0 and the handle, -1 (see
 * ignis_media_last_error) on a load without vision or any invalid input, or
 * IGNIS_MEDIA_ENCODE_POOL_FULL. */

/* An embedding's width in merged columns per pool page (GitHub #243). One
 * page is this many `[hidden]` BF16 columns: 1,280 KiB at 5120 hidden, whose
 * 10,240-byte column keeps every page 256-aligned. */
#define IGNIS_MEDIA_EMBEDDING_PAGE_COLUMNS 128

/* ignis_media_encode: the item fits the pool but not the pages free right
 * now. Distinct from -1 because it is the only failure a caller can clear by
 * releasing another embedding and retrying. */
#define IGNIS_MEDIA_ENCODE_POOL_FULL (-2)
struct ignis_media_encode_input {
  uint32_t size; /* sizeof(struct ignis_media_encode_input) */
  uint32_t grid_t;
  uint32_t grid_h;
  uint32_t grid_w;
  const uint16_t *patches;
  const int32_t *position_ids;
  const int32_t *cu_seqlens;
  const int32_t *position_table_indices;
  const float *position_table_weights;
};

int32_t ignis_media_encode(struct ignis_model *model, const struct ignis_media_encode_input *input,
                           struct ignis_media_embedding **out_embedding);

/* The embedding's merged columns. 0 for NULL. */
uint32_t ignis_media_embedding_columns(const struct ignis_media_embedding *embedding);

/* Release an embedding, freeing the load's reservation for the next item.
 * NULL is a no-op; the model must still be live. */
void ignis_media_embedding_release(struct ignis_media_embedding *embedding);

/* The message from the most recent failing ignis_media_encode on this thread.
 * Never NULL. */
const char *ignis_media_last_error(void);

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
 * stable, graph-independent dispatch counter, unaffected by whether that
 * step replayed a graph. For a decode round that is the model's layer
 * count, at every batch width and on either path (GitHub #111: the
 * round is one batch-wide traversal, so its dispatch count no longer scales
 * with the width -- it did before, one complete traversal per lane).
 * `graph_launches` (P3-05, GitHub #102) is the number of `cudaGraphLaunch`
 * calls the most recent `ignis_program_decode` call made -- 1 when it
 * replayed a captured decode graph, 0 when it enqueued the same traversal
 * directly (including every `ignis_program_prefill` call, which never
 * replays a graph). `decode_graph_ready_mask` has bit (w-1) set when a
 * decode graph for exact width w (1..IGNIS_DECODE_MAX_BATCH) is captured
 * and replayable; a clear bit means that width always falls back to
 * eager. */
struct ignis_program_stats {
  uint64_t vram_bytes;
  uint64_t last_step_micros;
  uint64_t kernel_count;
  uint64_t graph_launches;
  uint32_t decode_graph_ready_mask;
  /* P5-04 (GitHub #153): bit (w-1) set when a *verify* graph for exact batch
   * width w is captured at the load's draft window. Always 0 on a load with
   * no window. */
  uint32_t verify_graph_ready_mask;
};

/* Captures one CUDA graph per exact decode batch width 1..IGNIS_DECODE_MAX_BATCH
 * (P3-05, GitHub #102, ADR 0019). Must be called once, after `pool` is
 * created and before any concurrent `ignis_program_decode` call -- capture
 * is not thread-safe with replay. Every width is attempted independently: a
 * capture failure for one width is logged (see ignis_step_last_error for the
 * last one) and leaves that width's bit clear in `*out_ready_mask`, but does
 * not fail this call or any other width -- a decode round at a width whose
 * graph failed to capture always falls back to the eager per-lane loop, so
 * a capture failure degrades performance and never refuses service.
 * `*out_capture_micros`, if non-null, receives the wall-clock cost of this
 * call (every width's capture, sequentially) so the startup cost is
 * measured and reported, not assumed. Returns 0 unless `model` or `pool` is
 * null (in which case no width is attempted and `*out_ready_mask` is left
 * unset).
 *
 * P5-04 (GitHub #153): on a load with a draft window this call also
 * captures one *verify* graph per exact width at that window, after the
 * decode graphs; a verify capture failure degrades that width to the eager
 * verify traversal, never refuses service, and is reported through
 * `ignis_program_stats::verify_graph_ready_mask` (the `*out_ready_mask` here
 * stays the decode graphs' own). */
int32_t ignis_decode_graph_capture(struct ignis_model *model, struct ignis_seq_pool *pool,
                                   uint64_t *out_capture_micros, uint32_t *out_ready_mask);

/* The message from the most recent width whose capture failed inside the
 * latest ignis_decode_graph_capture call on this thread (its own channel,
 * separate from ignis_step_last_error -- each ABI surface owns its own).
 * Never NULL; empty if every width captured, or before the first call. */
const char *ignis_decode_graph_last_error(void);

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
 * On a model loaded with IGNIS_SPECULATIVE_DFLASH2 (P5-03, GitHub #152) the
 * chunked route also taps the target's layer 5/19/33/47/61 outputs for the
 * span's last min(2048, num_tokens) positions into chunk-scoped scratch --
 * never persisted, never a state section -- and appends their projected
 * context to `seq`'s drafter window before each chunk's synchronize, leaving
 * the window's frontier at the span's end. `pool` must then have been built
 * with the same backend, and a plain model refuses a drafter pool: both are
 * rejected before any device work. The per-token route does not touch the
 * window.
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

/* Per-call options for ignis_program_decode (ADR 0016; P5-04, GitHub #153,
 * spec 05). `size` must be `sizeof(struct ignis_decode_options)`; a NULL
 * pointer means the production defaults -- today's round, one token per
 * lane.
 *
 * `speculative_window == 0` is today's round. `speculative_window == k` runs
 * the verify round: it must equal the window the model was loaded with
 * (`ignis_model_load_options::draft_tokens`, the one the verify graphs were
 * captured at) -- any other value is rejected naming both, never padded.
 *
 * The drafts come through `drafts`, an internal seam this ticket fills from
 * a test's fake drafter and P5-05 fills from the DFlash2 drafter: host
 * `[batch_size][speculative_window]`, row-major, lane i's proposals in
 * order; `draft_counts[i]` (NULL: `speculative_window` for every lane) says
 * how many of lane i's entries are proposals. A NULL `drafts` proposes
 * nothing, so every lane runs at extent 0 -- a fallback step inside the same
 * round, one committed token, exactly what the round emits today.
 *
 * `out_committed_counts` (`batch_size` entries, required) receives how many
 * tokens lane i committed this round, 1..k+1; `out_token_ids` is then
 * `[batch_size][speculative_window + 1]` and lane i's committed run is its
 * first `out_committed_counts[i]` entries: the anchor (the successor the
 * prior round made ready) followed by the accepted drafts, cut at the first
 * stop id inclusive.
 *
 * P5-05 (GitHub #155): on a load with the DFlash2 drafter the leaf proposes
 * every lane's drafts itself, from the lane's own window, so `drafts` and
 * `draft_counts` must be NULL there; a lane's extent is then `min(k,
 * remaining_tokens - 1, remaining context - 1)`. After the commit the
 * committed columns' feature taps are appended to each lane's window; a lane
 * at extent 0 appends nothing. `out_extents` (`batch_size` entries, or
 * NULL) receives each lane's extent -- the drafts this round verified for
 * it, whichever side proposed them. */
struct ignis_decode_options {
  uint32_t size;               /* sizeof(struct ignis_decode_options) */
  uint32_t speculative_window; /* 0: today's round; k: the verify round at the load's window */
  const int32_t *drafts;       /* [batch_size][speculative_window], or NULL */
  const uint32_t *draft_counts; /* [batch_size], or NULL (every lane proposes the window) */
  int32_t *out_committed_counts; /* [batch_size]; required when speculative_window > 0 */
  uint32_t *out_extents;  /* [batch_size], or NULL: each lane's extent this round */
  /* P6-06 (GitHub #242): each lane's committed token's probability within its
   * own permitted set, or 0 for a lane that declared none. NULL asks for
   * nothing. One float per lane rather than a logits row -- see
   * `ignis_sampling_params::permitted_ids`. Filled only for the anchor, so a
   * verify round reports the anchor's and says nothing about its drafts. */
  float *out_permitted_probs; /* [batch_size], or NULL */
};

/* Complete one decode round for a batch of sequence handles.  Each output is
 * the successor made ready by prefill/the prior round; that token is
 * consumed before return to make the next successor ready.
 *
 * `sampling` is an array of `batch_size` entries parallel to `sequences`
 * (P3-03, GitHub #99): sequence `i` is sampled with `sampling[i]`, so lanes
 * sharing this round may carry independent temperatures, seeds and penalty
 * histories. `batch_size` must not exceed `IGNIS_DECODE_MAX_BATCH`. Returns
 * -1 (see ignis_step_last_error) on a null argument, an oversized batch, or
 * an unrecognized `sampling[i].size`.
 *
 * `options` (P5-04, GitHub #153; NULL = today's round) selects the verify
 * round -- see `struct ignis_decode_options`. With a window of k, lane i's
 * extent is `min(k, draft_counts[i], remaining_tokens - 1, remaining
 * context - 1)` draft columns beyond the anchor; the 64 layers traverse the
 * batch's `k+1` columns per lane once, with the GDN layers recording
 * ReplaySSM transitions instead of advancing and the GQA layers appending
 * every valid column; the vendored accept kernel (greedy, or the
 * distribution-preserving branch at temperature > 0, with its stateless RNG
 * keyed by seed, position and purpose) licenses the accepted prefix plus
 * one correction or bonus token; the run is cut at the first stop id; the
 * KV frontier moves to the committed length, the ReplaySSM fold rebuilds
 * each lane's GDN slot and conv taps from its committed records, and the
 * accepted final hidden state is kept for the drafter (P5-05). The round is
 * atomic as before: no lane's state advances unless the whole round
 * succeeds. With NULL options the width-1 path is today's, unchanged. */
int32_t ignis_program_decode(struct ignis_model *model, struct ignis_seq_pool *pool,
                             struct ignis_seq *const *sequences, uint64_t batch_size,
                             const struct ignis_sampling_params *sampling,
                             int32_t *out_token_ids,
                             const struct ignis_decode_options *options);

/* Read program counters and allocated device footprint. */
int32_t ignis_program_stats(const struct ignis_model *model,
                            const struct ignis_seq_pool *pool,
                            struct ignis_program_stats *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_STEP_H */
