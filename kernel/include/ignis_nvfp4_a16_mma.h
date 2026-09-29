/* ignis kernel leaf: an A16 NVFP4 linear on the tensor cores for wide calls --
 * OURS, not vendored.
 *
 * The vendored A16 route for an NVFP4 weight (`nvfp4_dispatch`, A16 branch) is
 * a GEMV family: one launch per 32-column slice of the activation, each on the
 * FMA/ALU pipes. That is the right shape for decode, where T is 1..32. It is
 * the wrong one for a prompt: the DFlash2 drafter's context append projects
 * every prompt token through `feature_projection` [5120, 25600] and each
 * drafter layer's `query_key_value` [6144, 5120] with A16-only weights (the
 * vendored registry admits no activation quantization for the DFlash2
 * geometries), so a 1,024-token prefill chunk became 32 GEMV launches per
 * matrix with the tensor pipe at 0%. Measured 2026-09-29 on an RTX 5090:
 * 21% of prefill device time
 * (docs/findings/2026-09-29-gpu-resources-prefill-vs-decode.md).
 *
 * This route keeps the A16 compute profile -- BF16 activations, the NVFP4
 * weight decoded as it is stored, FP32 accumulation -- and moves the product
 * onto BF16 tensor-core MMA. The weight tile is decoded into shared memory as
 * BF16: an E2M1 code (2 significant bits) times its E4M3 group scale
 * (4 significant bits) has at most six and fits BF16's range, so the decoded
 * element is exact; the per-tensor `1 / weight_scale_divisor` is applied
 * once, in FP32, in the epilogue. It is held to the same Linear criterion as
 * the GEMVs, not to their bits: the accumulation order differs, so a sum
 * that rounds can round to a different BF16.
 *
 * It lives in `kernel/include` for the reason `ignis_dflash2_topk.h` does:
 * the leaf's own CTest has to drive the function the engine ships.
 *
 * ADR 0010: this carries no port claim. It is not a vendored file, a patch to
 * one, or a translation of one; it shares only the Linear numerical contract
 * (`ninfer/ops/linear.h`), and the vendored A16 test criterion is what holds
 * it to that contract.
 */
#ifndef IGNIS_NVFP4_A16_MMA_H
#define IGNIS_NVFP4_A16_MMA_H

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

/* Whether an A16 call with this weight and column count takes the MMA route:
 * a registered NVFP4 problem at 64 columns or more, 128 or more when it has
 * fewer than 4,096 output rows. Narrower calls stay on the vendored GEMVs,
 * which win there (the thresholds are measured; see
 * kernel/src/nvfp4_a16_mma.cu). */
bool ignis_nvfp4_a16_mma_applies(std::int32_t output_rows, std::int32_t input_rows,
                                 std::int32_t tokens);

/* `out[N, T] = W[N, K] * x[K, T]` for an NVFP4 `W`, A16 compute. It checks
 * only the shape rule: the Linear semantics (contiguous, 16-byte-aligned BF16
 * `x` and `out`, matching extents) and the weight's NVFP4 layout are
 * `ops::linear`'s to validate, and this entry point assumes them. It is
 * public so the leaf's CTest can hold `ops::linear` to it bit for bit; the
 * engine reaches it only through `ops::linear`. Throws for a shape
 * `ignis_nvfp4_a16_mma_applies` rejects. */
void ignis_nvfp4_a16_mma(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out,
                         cudaStream_t stream);

#endif /* IGNIS_NVFP4_A16_MMA_H */
