// ignis kernel leaf: the target argmax of a mixed round (GitHub #718, spec #711) -- an ADAPTED op
// (ADR 0051 tier 2), leaf-internal: the vendored argmax's tiled route restated
// (kernel/src/tiled_argmax.cu names its upstream, as kernel/NOTICE does), with its output zeroed by
// a kernel of ours where the vendored launcher zeroes it by a memset. A graph captured over it has a
// kernel node where the vendored op leaves a memset node, so a mixed round's graph (kernel/src/
// decode_graph.cu `ignis_verify_graph_run_batch`, under the mixed graph's build) holds no memset
// node. The answer is the vendored op's bit for bit (kernel/tests/test_tiled_argmax.cu).
#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::tiled_argmax {

// `ninfer::ops::argmax`'s contract (ninfer/ops/argmax.h): out[t] is the lowest row of the largest
// value among rows 0..valid_rows-1 of column t of the contiguous BF16 [physical_rows, T] `logits`,
// into the contiguous I32 [T] `out`, enqueued on `stream`. Where the vendored launcher takes its
// tiled route (two tiles of 512 rows or more), the same route: `out` zeroed, then a CTA a tile and
// column whose winner an atomic compare-and-swap keeps, at the vendored launcher's block size for the
// shape. Elsewhere (one tile) the vendored op itself, which zeroes nothing. Throws
// std::invalid_argument on operands the vendored op refuses.
void argmax(const ninfer::Tensor &logits, ninfer::Tensor &out, std::int32_t valid_rows, cudaStream_t stream);

} // namespace ignis::tiled_argmax
