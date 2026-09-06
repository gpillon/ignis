/* ignis kernel leaf: one GDN (linear-attention) layer in the program
 * (ADR 0009, GitHub #58, P1-22).
 *
 * Runs a single GDN layer device-resident for `num_tokens` sequential tokens of
 * one sequence, on top of the ADR 0010 vendored ops (ninfer::ops). The layer
 * follows the reference's text context: input RMSNorm -> fused GDN input
 * projection + causal conv (rolling taps on the sequence's conv state) -> GDN
 * gating projection + gating -> per-head fp32 recurrence on the sequence's
 * GDN slot -> gated RMSNorm with z -> output projection + residual -> MLP
 * tail. It is verified against the P1-20 f64 layer reference (crates/artifact
 * f64_reference.rs, evaluate_layer on the GDN layer).
 *
 * The layer's GDN state (the conv taps and the fp32 recurrent slot) is drawn
 * from `seq`'s slot in `pool` (the sequence-state pools, GitHub #55) and
 * carries across the `num_tokens` tokens; releasing and re-allocating the
 * sequence resets it (a fresh slot reads zero, see ignis_seq.h).
 *
 * `in_residual` / `out_residual` are device-resident BF16 `[hidden,
 * num_tokens]` feature-major buffers: no host activation pointer crosses this
 * boundary. `out_residual` is written in place and receives the layer's final
 * residual. The model handle owns the layer's stream and scratch
 * (kernel/src/model_internal.h) -- no stream or host pointer crosses this ABI.
 *
 * Rust bindings: crates/core/src/gdn_layer.rs (keep 1:1).
 */
#ifndef IGNIS_GDN_LAYER_H
#define IGNIS_GDN_LAYER_H

#include <stdint.h>

#include "ignis_model.h"
#include "ignis_seq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runs one GDN layer (the layer's full attention + MLP tail) for `num_tokens`
 * sequential tokens of one sequence. `model` supplies the layer's weights, the
 * stream, and the scratch; `pool` + `seq` supply the sequence's GDN state
 * (conv taps + fp32 recurrent slot, addressed by `seq`'s slot); `layer` is the
 * zero-based layer index (a GDN layer, e.g. 4 for the 27B's BF16
 * output-projection exception). `in_residual` / `out_residual` are device
 * BF16 `[hidden, num_tokens]` feature-major buffers (out_residual is updated in
 * place and receives the final residual). Returns 0 on success, -1 on a
 * null/invalid argument, an unknown layer, or a kernel error (see
 * ignis_gdn_layer_last_error). */
int32_t ignis_gdn_layer_step(struct ignis_model *model, struct ignis_seq_pool *pool,
                             struct ignis_seq *seq, uint32_t layer, const void *in_residual,
                             void *out_residual, uint64_t num_tokens);

/* The message from the most recent failing ignis_gdn_layer_step call on this
 * thread (thread-local; overwritten by the next call; empty string if none
 * failed yet). Never NULL. */
const char *ignis_gdn_layer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_GDN_LAYER_H */