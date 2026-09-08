// ignis kernel leaf - P2-02 (GitHub #84): the layer-body entry points a
// chunk loop dispatches directly, without the per-call stream
// synchronization or sequence-position advance `ignis_gqa_layer_step` /
// `ignis_gdn_layer_step` (kernel/include/ignis_gqa_layer.h /
// ignis_gdn_layer.h) each do for their own (per-token, GPU-test) callers.
//
// A chunk's caller (kernel/src/step.cu) dispatches every layer's body for
// the whole chunk on the model's stream, synchronizes once, and only then
// advances `seq`'s position state -- so a failure anywhere in the chunk
// leaves `seq` exactly at its pre-chunk position (P2-02's failure
// semantics). Each function here does the same argument and topology
// validation its ABI counterpart does; it just stops short of the
// sync/advance.
//
// Not part of the public flat C ABI: this header is leaf-internal, shared
// between kernel/src/gqa_layer.cu, kernel/src/gdn_layer.cu and
// kernel/src/step.cu.
#pragma once

#include "model_internal.h"
#include "ignis_seq_internal.h"

#include "ninfer/ops/linear.h"

#include <cstdint>

// Runs one GQA layer's body (input norm -> fused q/k/gate/v projection ->
// q/k norm + RoPE -> KV append -> attention -> out-projection + residual ->
// post norm -> SwiGLU -> down + residual) for `num_tokens` sequential
// tokens, enqueued on the model's stream with no synchronization. `layer`
// is the model's absolute layer index. Returns 0 on success, -1 on a
// null/invalid argument, a non-GQA layer, or a leaf/kernel error (see
// ignis_gqa_layer_last_error). On success the caller still owes a stream
// synchronization and `seq->gqa_positions[ignis_gqa_relative_layer(layer)]
// += num_tokens` before the advanced position is valid.
int32_t ignis_gqa_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens);

// Runs one GDN layer's body for `num_tokens` sequential tokens, enqueued on
// the model's stream with no synchronization. `layer` is the model's
// absolute layer index. Returns 0 on success, -1 on a null/invalid
// argument, a non-GDN layer, or a leaf/kernel error (see
// ignis_gdn_layer_last_error). The layer's GDN state (conv taps + fp32
// recurrent slot) is drawn from `seq`'s slot and updated in place by the
// enqueued work itself -- unlike the GQA position counter, it has no
// separate host-side advance for the caller to defer.
int32_t ignis_gdn_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens);

// The Qwen 3.8 topology's GQA layers sit at index 3, 7, 11, ... (every 4th);
// this is the same relative-index formula `ignis_gqa_layer_step` and
// `ignis_gqa_layer_run_body` use internally, exposed so a chunk loop can
// advance `seq->gqa_positions` itself once its one chunk-wide
// synchronization succeeds.
inline uint32_t ignis_gqa_relative_layer(uint32_t layer) {
  return (layer - 3) / 4;
}

// The widest compute policy `qtype`'s own registered arm admits (only
// NVFP4 admits AllowA4; every other registered qtype in the real 27B
// artifact -- BF16_CTRL, W8G32_F16S -- admits only A16Only). Used only for
// *sizing*: P2-01 (GitHub #83) reserves every layer's scratch at this
// policy per weight (kernel/src/model.cu's own
// `compute_program_scratch_bytes`) so the reservation is an upper bound
// regardless of the token count a given call actually resolves to.
inline ninfer::ops::LinearPolicy ignis_widest_linear_policy_for(ninfer::QType qtype) {
  return qtype == ninfer::QType::NVFP4 ? ninfer::ops::LinearPolicy::AllowA4
                                       : ninfer::ops::LinearPolicy::A16Only;
}

// The policy `linear_swiglu`'s MLP-tail call needs for `tokens` tokens of a
// `qtype` weight (P2-02, GitHub #84). Unlike every other NVFP4 dispatch
// this program makes (attn_input_proj, gdn_input_proj, linear_add, GQA
// attention, the GDN recurrence -- all of which admit A16Only at every
// positive T, unchanged here), `linear_swiglu`'s NVFP4 registration admits
// its no-policy (A16Only) overload only through T=16
// (kernel/vendor/include/ninfer/ops/linear_swiglu.h): a chunk wider than
// that has no registered A16Only route to fall back to, so AllowA4 is not
// a performance choice this ticket could defer for those calls. Below
// T=17 this returns plain A16Only, not the sizing helper's unconditional
// AllowA4: AllowA4's real NVFP4 activation quantization is measurably less
// precise even at small T (GitHub #84's own gate run: T=4 against the f64
// reference moved the relative L2 error from within tolerance to ~11%),
// so forcing it below the width where it is actually required would
// regress decode (always T=1) and every per-token-route call for no
// reason -- the per-token route stays bit-for-bit what it always was.
inline ninfer::ops::LinearPolicy ignis_linear_swiglu_policy_for(ninfer::QType qtype,
                                                                std::int32_t tokens) {
  if (qtype != ninfer::QType::NVFP4 || tokens <= 16) {
    return ninfer::ops::LinearPolicy::A16Only;
  }
  return ninfer::ops::LinearPolicy::AllowA4;
}
