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

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cstdint>

// P2-03 (GitHub #85): the compute-policy mode a program call resolves to
// (ADR 0016's `compute_policy` field, resolved once at the prefill entry):
// `kEngineDefault` adopts the reference's text-model policy verbatim —
// AllowA4 on every NVFP4 projection, the artifact's other qtypes
// (BF16_CTRL, W8G32_F16S) admitting only A16; `kA16Only` forces A16Only on
// every weight (the test-only route-comparison override, ADR 0016). The
// mode only sets what the vendored dispatch is *allowed* to admit: the
// actual route a call takes is the dispatch's own per-projection token
// thresholds' decision, so the engine encodes no threshold of its own
// (GitHub #85's acceptance).
enum class LinearPolicyMode { kEngineDefault, kA16Only };

// The policy a `qtype` weight is dispatched under a `mode` (P2-03, GitHub
// #85): `kA16Only` forces the A16 route everywhere it is registered (the
// one op whose A16 registration domain is width-capped — `linear_swiglu`,
// T<=16 — is handled by `ignis_linear_swiglu_policy_for` below, never by
// this helper); otherwise the reference's text-model policy — AllowA4 for
// NVFP4, A16Only for the artifact's other qtypes (they admit nothing
// else).
inline ninfer::ops::LinearPolicy ignis_policy_for(ninfer::QType qtype, LinearPolicyMode mode) {
  if (mode == LinearPolicyMode::kA16Only || qtype != ninfer::QType::NVFP4) {
    return ninfer::ops::LinearPolicy::A16Only;
  }
  return ninfer::ops::LinearPolicy::AllowA4;
}

// The `linear_swiglu` counterpart of `ignis_policy_for` (P2-03, GitHub
// #85): this op's NVFP4 A16 route registration domain tops out at T=16
// (kernel/vendor/include/ninfer/ops/linear_swiglu.h) — unlike every other
// projection this program dispatches, which admit A16Only at every
// positive T. The cap is an op-registration fact, not an engine threshold:
// under `kEngineDefault` the ticket passes AllowA4 and the dispatch decides
// the route per its own token thresholds (acceptance 2, no width test);
// under the test-only `kA16Only` override the A16 route is forced where it
// is registered, and above the T=16 cap the only runnable route is AllowA4
// (a wider chunk simply cannot run this op at A16) — the exact pre-#85
// behavior #84's helper encoded, so the route-comparison test's A16 arm
// reproduces the pre-#85 numerics on the same span.
inline ninfer::ops::LinearPolicy ignis_linear_swiglu_policy_for(ninfer::QType qtype,
                                                                LinearPolicyMode mode,
                                                                std::int32_t tokens) {
  if (qtype != ninfer::QType::NVFP4) {
    return ninfer::ops::LinearPolicy::A16Only;
  }
  if (mode == LinearPolicyMode::kEngineDefault) {
    return ninfer::ops::LinearPolicy::AllowA4;
  }
  return tokens <= 16 ? ninfer::ops::LinearPolicy::A16Only
                      : ninfer::ops::LinearPolicy::AllowA4;
}

// Runs one GQA layer's body (input norm -> fused q/k/gate/v projection ->
// q/k norm + RoPE -> KV append -> attention -> out-projection + residual ->
// post norm -> SwiGLU -> down + residual) for `num_tokens` sequential
// tokens, enqueued on the model's stream with no synchronization. `layer`
// is the model's absolute layer index. Returns 0 on success, -1 on a
// null/invalid argument, a non-GQA layer, or a leaf/kernel error (see
// ignis_gqa_layer_last_error). On success the caller still owes a stream
// synchronization and `seq->gqa_positions[ignis_gqa_relative_layer(layer)]
// += num_tokens` before the advanced position is valid. `mode` is the
// call's compute-policy mode (P2-03, GitHub #85: ADR 0016's
// `compute_policy` field, resolved by the entry point that dispatches this
// body): every NVFP4 projection in the layer is dispatched under the
// policy `ignis_policy_for` resolves for it.
int32_t ignis_gqa_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode);

// The same body + one stream synchronization + the deferred
// `gqa_positions` advance, for a caller that wants `ignis_gqa_layer_step`'s
// synchronous contract but a non-default compute-policy mode (P2-03,
// GitHub #85: the program's per-token route threads ADR 0016's
// `compute_policy` override through it). Internal, not part of the flat C
// ABI: `ignis_gqa_layer_step` is the ABI entry point and calls this one
// with `kEngineDefault`.
int32_t ignis_gqa_layer_step_mode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode);

// Runs one GDN layer's body for `num_tokens` sequential tokens, enqueued on
// the model's stream with no synchronization. `layer` is the model's
// absolute layer index. Returns 0 on success, -1 on a null/invalid
// argument, a non-GDN layer, or a leaf/kernel error (see
// ignis_gdn_layer_last_error). The layer's GDN state (conv taps + fp32
// recurrent slot) is drawn from `seq`'s slot and updated in place by the
// enqueued work itself -- unlike the GQA position counter, it has no
// separate host-side advance for the caller to defer. `mode` is the
// call's compute-policy mode (P2-03, GitHub #85): every NVFP4 projection
// in the layer is dispatched under the policy `ignis_policy_for` resolves
// for it.
int32_t ignis_gdn_layer_run_body(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode);

// The GDN counterpart of `ignis_gqa_layer_step_mode` (no position advance:
// the GDN layer's state is updated in place by the enqueued work, so only
// the stream synchronization is owed after the body).
int32_t ignis_gdn_layer_step_mode(ignis_model *model, ignis_seq_pool *pool, ignis_seq *seq,
                                  uint32_t layer, const void *in_residual, void *out_residual,
                                  uint64_t num_tokens, LinearPolicyMode mode);

// P3-05 (GitHub #102, ADR 0019): one GQA layer's body for a single decode
// lane inside a captured (or capturing) decode graph. Unlike
// `ignis_gqa_layer_run_body`, this takes no `ignis_seq*` -- every address it
// touches is either fixed for the model's lifetime (weights,
// `pool->kv_pool.block_tables()`, the pool-wide block-table matrix) or a
// stable staging address `model` owns (`decode_graph_token_ids`,
// `decode_graph_slots`, `sampling_decode_positions`), so the exact same
// kernel launches are correct whether this call happens during capture (any
// staged content) or during replay (this round's real content, refreshed by
// the caller's H2D copy before `cudaGraphLaunch`). `lane` selects which
// column of those staging buffers this call reads -- a capture-time-fixed
// index, not a sequence identity. Scratch comes from
// `model->decode_graph_scratch`, never `model->scratch` (ADR 0019: a
// prefill chunk between two replays must not alias what a replay rereads).
// The envelope is fixed at `model->max_context_tokens` (a safe
// over-approximation every round, per the op's own "host launch-resource
// promise" contract) rather than derived from any lane's actual position.
// Returns 0 on success, -1 on a kernel/copy error (see
// ignis_gqa_layer_last_error).
int32_t ignis_gqa_layer_run_body_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                                       uint32_t lane, const void *in_residual, void *out_residual,
                                       LinearPolicyMode mode);

// The GDN counterpart of `ignis_gqa_layer_run_body_graph`: uses
// `causal_conv1d_silu_snapshot` / `gated_delta_net_snapshot` with
// `initial_state_slots == snapshot_base_slots == decode_graph_slots + lane`
// (in-place update of the pool slot named by this round's staged value) in
// place of `ignis_gdn_layer_run_body`'s direct `seq->slot`-addressed calls.
int32_t ignis_gdn_layer_run_body_graph(ignis_model *model, ignis_seq_pool *pool, uint32_t layer,
                                       uint32_t lane, const void *in_residual, void *out_residual,
                                       LinearPolicyMode mode);

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
