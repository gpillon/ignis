// ignis kernel leaf - GitHub #711 (spec #711): a device range zeroed by a kernel -- OURS.
// A graph captured over it holds one kernel node where `cudaMemsetAsync` leaves a memset node, so a
// mixed round's graphs under the mixed graph's build hold no memset node on a KV cache whose
// attention workspace is zeroed (kernel/src/gqa_layer.cu, `ignis_gqa_workspace_needs_zeroing`), as
// the target argmax's zeroing by a kernel gives them on every cache (kernel/src/tiled_argmax.h).
// Held to the bytes it zeroes and the node it records by kernel/tests/test_device_zero.cu.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ignis::device_zero {

// `bytes` bytes from `data` set to 0 on `stream`, at any alignment: nothing for 0 bytes, otherwise
// one kernel launch. Returns the launch's error.
cudaError_t zero(void *data, std::size_t bytes, cudaStream_t stream);

} // namespace ignis::device_zero
