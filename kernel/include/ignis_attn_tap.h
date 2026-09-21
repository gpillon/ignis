/* ignis kernel leaf: attention-input tap -- test-only diagnostic seam
 * (docs/findings/2026-09-21-one-attention-head-points.md).
 *
 * NOT part of the public flat C ABI. It exists to measure one thing in the
 * engine rather than in a PyTorch vehicle: whether a GQA head's attention
 * from a chosen query position onto the image tokens points at the target.
 * The fused attention kernels never materialize attention weights, so this
 * captures what they are computed from -- the query and key rows after the
 * q/k norm and the rotation, exactly as `run_gqa_layer` hands them to the
 * attention op -- and leaves the scoring to the host.
 *
 * What it captures, per armed GQA layer, during a prefill:
 *   - the rotated **key** rows of every prefilled position, all KV heads;
 *   - the rotated **query** rows of each requested position, all Q heads.
 * Both are BF16 bit patterns (uint16_t), row-major as the op lays them out:
 * one position's query row is kIgnisGqaQHeads * 256 elements, head-major;
 * one position's key row is 4 * 256.
 *
 * **The keys are the ones before the cache, not the ones in it.** Under a
 * BF16 KV pool they are identical to what attention reads. Under hq-e8-2b
 * the attention kernel reads the codec's decode of them instead; this tap
 * does not see that. So an hq run through this tap measures everything the
 * codec did to the hidden states of *earlier* layers, and not what it does
 * to the armed layer's own scores. The difference is the codec's route
 * error (docs/findings/2026-09-12-hq-attention-route-agreement.md), and it
 * is stated wherever a number from this tap is reported.
 *
 * Positions are **cache positions** (the sequence index `run_gqa_layer`
 * appends at, `seq->gqa_positions`), not rotary positions. A multimodal
 * prompt's image tokens keep their prompt indices here; their MRoPE
 * positions are irrelevant to where a row is stored.
 *
 * Cost when disarmed: one relaxed load of a flag per GQA layer call. That
 * branch *does* sit in the production prefill path -- unlike
 * `ignis_kv_capture_rows`, which nothing in production references -- and it
 * is the only part of this seam that does. Arming it is reachable only
 * through crates/core's non-default `attn-tap` cargo feature. Armed, every
 * armed layer adds a synchronous device-to-host copy per chunk: a test
 * speed, not a serving one.
 *
 * Not thread-safe and not per-sequence: the arm is process-wide, and a
 * prefill of any sequence while armed is captured. The caller arms, runs
 * exactly one sequence's prefill, and disarms.
 */
#ifndef IGNIS_ATTN_TAP_H
#define IGNIS_ATTN_TAP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Arm the tap.
 *
 * `gqa_ordinals[n_layers]`: which GQA layers, as 0..15 ordinals (backbone
 * layer `4 * ordinal + 3`), each at most once.
 * `query_positions[n_queries]`: cache positions whose query rows to keep.
 * `max_positions`: the length of each layer's key buffer, in positions; a
 * prefill that writes a key row at or past it fails the layer call loudly
 * rather than truncating.
 * `q_out`: caller-allocated, `n_layers * n_queries * 24 * 256` uint16_t,
 *   indexed [layer][query][q_head][dim] in the order given.
 * `k_out`: caller-allocated, `n_layers * max_positions * 4 * 256` uint16_t,
 *   indexed [layer][position][kv_head][dim]. Rows never written stay as the
 *   caller left them.
 *
 * Both buffers must stay valid until `ignis_attn_tap_disarm`. Returns 0, or
 * -1 with `ignis_attn_tap_last_error` on a null pointer, an empty or
 * out-of-range ordinal list, a duplicate ordinal, a negative position, or a
 * non-positive `max_positions`. Arming while armed replaces the arm. */
int32_t ignis_attn_tap_arm(const int32_t *gqa_ordinals, int32_t n_layers,
                           const int64_t *query_positions, int32_t n_queries,
                           int64_t max_positions, uint16_t *q_out, uint16_t *k_out);

/* Disarm, and report what was captured: for each armed layer, how many key
 * rows were written (`rows_written[n_layers]`, may be null), and for each
 * requested query position, whether its row was seen in *every* armed layer
 * (`queries_seen[n_queries]`, 0/1, may be null). A query position the
 * prefill never reached reads 0 there, which is the caller's to refuse. */
int32_t ignis_attn_tap_disarm(int64_t *rows_written, int32_t *queries_seen);

const char *ignis_attn_tap_last_error(void);

#ifdef __cplusplus
} /* extern "C" */

#include <cuda_runtime.h>

/* The hook `run_gqa_layer` calls after `qk_norm_rope`. Returns 0 when
 * disarmed or when the layer is not armed, without touching the stream. */
int32_t ignis_attn_tap_record(uint32_t gqa_ordinal, int64_t start_position, int32_t tokens,
                              const void *rotated_query, const void *rotated_key,
                              cudaStream_t stream);
#endif

#endif /* IGNIS_ATTN_TAP_H */
