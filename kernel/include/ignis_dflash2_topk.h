/* ignis kernel leaf: the DFlash2 drafter's per-column top-k -- OURS, not
 * vendored, and the first op this engine replaces rather than calls.
 *
 * The vendored op (`ninfer::ops::dflash2_topk`,
 * kernel/vendor/src/ops/kernel/dflash2_draft.cuh) gives one warp to one
 * column: every lane strides the whole vocabulary keeping a private sorted
 * list, then k warp-reduction rounds drain the heads. At this engine's shape
 * -- k=16 over `[248046, 7]` -- that is seven active warps on a 170-SM card,
 * and the private list (64 entries at 20 registers per thread) lives in local
 * memory, so every insertion comparison is a memory access. Measured
 * 2026-09-18 on an RTX 5090: 3,145 us per decode round, 16.4% of all decode
 * kernel time, against the 2.0 us its 3.47 MB of input costs at this card's
 * bandwidth.
 *
 * This implementation splits the row axis across blocks and merges, so the
 * work is spread over hundreds of blocks instead of seven warps, and keeps
 * each thread's running list in registers by making k a compile-time constant
 * and the insertion a fully unrolled compare-and-shift with static indices.
 *
 * It is a drop-in for the vendored call and is held to the vendored op's
 * contract exactly (`ninfer/ops/dflash2_topk.h`): ties break toward the
 * SMALLER row id, `ids` is largest-first, `values` carries the corresponding
 * `logits` entries bit-exactly, and the selection is the iterative
 * largest-first removal of the BF16-ordered values. The vendored op is the
 * oracle for that: kernel/tests/test_dflash2_topk.cu requires the two to
 * agree bit-for-bit, which an exact deterministic selection admits.
 *
 * It lives in `kernel/include` rather than beside its `.cu` for the reason
 * `ignis_gqa_workspace.h` does: the leaf's own CTest
 * (kernel/tests/test_dflash2_topk.cu) has to call the function the engine
 * actually ships, or it would be testing a second copy of it.
 *
 * ADR 0010: this carries no port claim. It is not a vendored file, it is not
 * a patch to one, and it is not a translation of one -- only the numerical
 * contract is shared, which is the point.
 */
#ifndef IGNIS_DFLASH2_TOPK_H
#define IGNIS_DFLASH2_TOPK_H

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

/* Bytes of transient workspace `ignis_dflash2_topk` needs for a call at this
 * shape, for the caller to bump out of its own arena. Zero when the shape
 * falls back to the vendored op. */
std::size_t ignis_dflash2_topk_workspace_bytes(std::int32_t rows, std::int32_t columns,
                                               std::int32_t k);

/* The vendored op's signature, plus the workspace. `workspace` must hold
 * `ignis_dflash2_topk_workspace_bytes` and is written in full; it is read by
 * nothing else and needs no initialization.
 *
 * A shape this implementation does not specialize is forwarded to the
 * vendored op unchanged, so the engine's behaviour never depends on which
 * one ran. */
void ignis_dflash2_topk(const ninfer::Tensor &logits, std::int32_t k, ninfer::Tensor &ids,
                        ninfer::Tensor &values, void *workspace, std::size_t workspace_bytes,
                        cudaStream_t stream);

#endif /* IGNIS_DFLASH2_TOPK_H */
