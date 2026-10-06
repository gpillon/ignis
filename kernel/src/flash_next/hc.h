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

// hidden_s += y * inj_s for every stream, in place, in BF16 (each product and
// each sum rounded, as the reference's bf16 ops do).
int32_t fn_hc_inject(const Geometry &g, const void *y, const float *inj, int32_t rows, void *hidden,
                     cudaStream_t stream);

}  // namespace ignis::flash_next
