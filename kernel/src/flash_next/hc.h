// ignis kernel leaf -- Flash-Next's hyper-connections (spec flash-next/04,
// GitHub #302; OURS, ADR 0043). S1-private: the layer driver's own ops.
//
// transformers' Qwen4ExpTextGatedResidual, per token, `hidden` the
// [streams * hidden] residual:
//   normed = hc_norm(hidden)                 grouped RMSNorm, (1 + w), per stream
//   m      = sigmoid(up(silu(down(normed) / streams)))
//   x      = mean_s(m_s * normed_s)          the sublayer's input, [hidden]
//   inj    = 2 * sigmoid(block_inject(normed) / streams)        [streams]
// and after the sublayer's output y: hidden_s += y * inj_s. The final mixer is
// the mix with no inject weights. The reference runs these modules in BF16:
// every module output is rounded to BF16 where it is, and the arithmetic
// inside a module is fp32 (fn_hc_mix rounds at the same points).

#pragma once

#include "flash_next_internal.h"

namespace ignis::flash_next {

// x (BF16 [rows][hidden]) = the mix of `hidden` (BF16 [rows][streams * hidden]);
// `inj` (fp32 [rows][streams], the BF16-rounded injection weights) unless
// w.block_inject is null (the final mixer) or `inj` is null.
int32_t fn_hc_mix(const Geometry &g, const HcWeights &w, const void *hidden, int32_t rows, void *x,
                  float *inj, ninfer::DeviceArena &scratch, cudaStream_t stream);
std::size_t fn_hc_mix_scratch_bytes(const Geometry &g, int32_t rows);

// The decode route's norm folded into its mix_down launch (GitHub #306, the fusion study): two
// launches per mix of up to three rows instead of three, the same bits. Read when a mix is
// launched (a decode graph keeps the route it was captured with); on unless
// IGNIS_FN_HC_FUSED=0 is set at the first mix.
void fn_hc_set_decode_fused(bool on);

// hidden_s += y * inj_s for every stream, in place, in BF16 (each product and
// each sum rounded, as the reference's bf16 ops do).
int32_t fn_hc_inject(const Geometry &g, const void *y, const float *inj, int32_t rows, void *hidden,
                     cudaStream_t stream);

// The inject a sublayer leaves for the next mix (GitHub #306, step 2): hidden_s += y * inj_s, y
// the sublayer's BF16 output -- or, with `acc`, the MoE combine's, ignis_moe_combine(acc, shared,
// x, w_gate) written to `y` first. `inj` is the injection weights of the mix before the sublayer.
struct PendingInject {
  void *y = nullptr;           // BF16 [rows][hidden]
  const float *inj = nullptr;  // fp32 [rows][streams]; null: nothing pending
  int64_t *acc = nullptr;      // the combine's operands, or null for none
  const float *shared = nullptr;
  const void *x = nullptr;
  const void *w_gate = nullptr;
  bool pending() const { return inj != nullptr; }
};

// fn_hc_mix of `hidden` after `pending` is applied to it. On the fused decode route at one row
// (fusion.h's HcNorm and Inject; past it the fold loses to the launches it saves) one down launch rebuilds each stream as the inject (and the
// combine) would leave it and the up launch stores it, the combine's accumulator zeroed: the same
// bits as ignis_moe_combine, fn_hc_inject and fn_hc_mix one after another, which is what runs
// anywhere else. `inj` (this mix's) must not be `pending.inj`, nor `pending.y` the mix's x.
int32_t fn_hc_mix_after(const Geometry &g, const HcWeights &w, const PendingInject &pending, void *hidden,
                        int32_t rows, void *x, float *inj, ninfer::DeviceArena &scratch, cudaStream_t stream);

// `pending` applied on its own (its combine, then fn_hc_inject): before anything else reads the
// residual. Nothing pending: nothing launched.
int32_t fn_hc_flush(const Geometry &g, const PendingInject &pending, void *hidden, int32_t rows,
                    cudaStream_t stream);

}  // namespace ignis::flash_next
