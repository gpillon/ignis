/* ignis kernel leaf: one GQA layer in the device-resident program
 * (ADR 0009, GitHub #57, P1-21).
 *
 * The program layer composes the ADR 0010 vendored attention, norm, RoPE,
 * projection, and MLP operations. Its only activation arguments are device
 * BF16 tensors; the sequence owns the paged KV allocation consumed below.
 */
#ifndef IGNIS_GQA_LAYER_H
#define IGNIS_GQA_LAYER_H

#include <stdint.h>

#include "ignis_model.h"
#include "ignis_seq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runs one full-attention (GQA) layer for `num_tokens` sequential tokens of
 * one sequence. `layer` is a zero-based GQA layer index. `in_residual` and
 * `out_residual` are device-resident BF16 `[hidden, num_tokens]` tensors;
 * no host activation buffer crosses this ABI. The leaf advances the GQA
 * layer's sequence frontier once per token, emits those positions into device
 * scratch, appends K/V to `seq`'s own paged-cache planes, and writes the final
 * residual to `out_residual`.
 *
 * Returns zero on success and -1 on invalid arguments, a non-GQA layer, or a
 * leaf/kernel error (read `ignis_gqa_layer_last_error`). */
int32_t ignis_gqa_layer_step(struct ignis_model *model, struct ignis_seq_pool *pool,
                             struct ignis_seq *seq, uint32_t layer,
                             const void *in_residual, void *out_residual,
                             uint64_t num_tokens);

/* Thread-local message from the most recent failed GQA-layer call. Never
 * NULL; overwritten by the next failed call. */
const char *ignis_gqa_layer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_GQA_LAYER_H */
