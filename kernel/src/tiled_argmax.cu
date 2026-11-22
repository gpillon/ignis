// ignis kernel leaf: the target argmax of a mixed round with its zeroing by a kernel (GitHub #718,
// spec #711; kernel/src/tiled_argmax.h) -- an ADAPTED op (ADR 0051 tier 2): the vendored tiled route
// restated, the zeroing ours.
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (the
// commit kernel/vendor/manifest.json pins, ADR 0010), Apache-2.0 (kernel/vendor/LICENSE), restated:
// src/ops/kernel/argmax.cuh `argmax_better`, `argmax_warp_reduce`, `argmax_block_reduce` and
// `argmax_tiled_atomic_kernel` (a CTA a tile of rows and a column, its winner by value then lowest
// row, kept in the column's output by an atomic compare-and-swap against the value at the row it
// holds), at one row a thread (`kArgmaxItemsPerThread`); src/ops/launcher/argmax.cu `argmax_launch`
// (the one-tile test at 512 rows a tile), `tiled_block_for` (512 threads a CTA, 128 for the
// registered 248,320 / 248,077-row vocabulary and 256 for the 131,072-row one past 8 columns) and
// `argmax_tiled_atomic_launch` (its grid of tiles by columns, a launch every 65,535 columns, as
// src/ops/common/token_slices.h `for_each_token_slice` slices them). Changed here: each slice's
// output is zeroed by a kernel of ours before the tiles' launch, where the vendored launcher zeroes
// it by cudaMemsetAsync, so that a graph captured over the op holds two kernel nodes and no memset
// node; the one-tile route is the vendored op itself, called and not copied (`ninfer::ops::argmax`);
// a failed launch throws naming the op. The vendored op stays in the subtree as it is, every other
// argmax's and the mixed round's without its graph build, and is this one's oracle
// (kernel/tests/test_tiled_argmax.cu, kernel/NOTICE).

#include "tiled_argmax.h"

#include "ninfer/ops/argmax.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::tiled_argmax {

namespace {

// The vendored launcher's blocks and its one-tile route's (`kArgmaxBlock`, one row a thread).
constexpr int kBlock = 512;
constexpr std::int32_t kFullPhysicalRows = 248320;
constexpr std::int32_t kFullValidRows = 248077;
constexpr std::int32_t kShortlistRows = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock = 128;
constexpr int kShortlistAggregateBlock = 256;
// CUDA's grid.y limit, the columns a launch takes.
constexpr std::int32_t kGridYLimit = 65535;
// The zeroing kernel's threads a CTA, a column a thread.
constexpr int kZeroBlock = 128;

__device__ __forceinline__ bool better(float value, std::int32_t index, float best_value, std::int32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void warp_reduce(float &value, std::int32_t &index) {
  constexpr unsigned int kMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(kMask, value, offset);
    const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
    if (better(other_value, other_index, value, index)) {
      value = other_value;
      index = other_index;
    }
  }
}

__device__ __forceinline__ void block_reduce(float &value, std::int32_t &index) {
  __shared__ float warp_values[16];
  __shared__ std::int32_t warp_indices[16];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  warp_reduce(value, index);
  if (lane == 0) {
    warp_values[warp] = value;
    warp_indices[warp] = index;
  }
  __syncthreads();
  value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
  index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
  if (warp == 0) {
    warp_reduce(value, index);
  }
}

__launch_bounds__(kBlock) __global__ void tiled_atomic_kernel(const __nv_bfloat16 *logits, std::int32_t *out,
                                                              std::int32_t valid_rows, std::int32_t physical_rows) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.y);
  const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
  const std::int32_t tile_start = static_cast<std::int32_t>(blockIdx.x) * blockDim.x;
  float best_value = -CUDART_INF_F;
  std::int32_t best_index = INT32_MAX;
  const std::int32_t v = tile_start + static_cast<std::int32_t>(threadIdx.x);
  if (v < valid_rows) {
    const float value = __bfloat162float(logits[base + v]);
    if (better(value, v, best_value, best_index)) {
      best_value = value;
      best_index = v;
    }
  }
  block_reduce(best_value, best_index);
  if (threadIdx.x != 0 || best_index == INT32_MAX) {
    return;
  }
  int current = out[t];
  while (true) {
    const float current_value = __bfloat162float(logits[base + current]);
    if (!better(best_value, best_index, current_value, current)) {
      break;
    }
    const int observed = atomicCAS(reinterpret_cast<int *>(out + t), current, best_index);
    if (observed == current) {
      break;
    }
    current = observed;
  }
}

// Ours: the slice's output zeroed, a column a thread, which the tiles' winners then replace.
__global__ void zero_kernel(std::int32_t *out, std::int32_t count) {
  const std::int32_t i = static_cast<std::int32_t>(blockIdx.x) * blockDim.x + static_cast<std::int32_t>(threadIdx.x);
  if (i < count) {
    out[i] = 0;
  }
}

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t columns) {
  if (columns <= kSmallAggregateColumns) {
    return kBlock;
  }
  if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
    return kFullAggregateBlock;
  }
  if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
    return kShortlistAggregateBlock;
  }
  return kBlock;
}

void check_launch(const char *what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("ignis::tiled_argmax: ") + what + ": " + cudaGetErrorString(err));
  }
}

} // namespace

void argmax(const ninfer::Tensor &logits, ninfer::Tensor &out, std::int32_t valid_rows, cudaStream_t stream) {
  const std::int32_t physical_rows = logits.ne[0];
  const std::int32_t columns = logits.ne[1];
  const auto tiles = [](std::int32_t rows, int block) { return (rows + block - 1) / block; };
  // The one-tile route, and every operand the vendored op refuses, are the vendored op's.
  if (logits.dtype != ninfer::DType::BF16 || out.dtype != ninfer::DType::I32 || logits.ne[2] != 1 ||
      logits.ne[3] != 1 || out.ne[0] != columns || out.ne[1] != 1 || out.ne[2] != 1 || out.ne[3] != 1 ||
      physical_rows <= 0 || valid_rows <= 0 || valid_rows > physical_rows || columns <= 0 ||
      !logits.is_contiguous() || !out.is_contiguous() || logits.data == nullptr || out.data == nullptr ||
      tiles(valid_rows, kBlock) < 2) {
    ninfer::ops::argmax(logits, out, valid_rows, stream);
    return;
  }
  const int block = tiled_block_for(physical_rows, valid_rows, columns);
  const auto *rows = static_cast<const __nv_bfloat16 *>(logits.data);
  auto *winners = static_cast<std::int32_t *>(out.data);
  for (std::int32_t offset = 0; offset < columns; offset += kGridYLimit) {
    const std::int32_t count = columns - offset < kGridYLimit ? columns - offset : kGridYLimit;
    zero_kernel<<<static_cast<unsigned int>(tiles(count, kZeroBlock)), kZeroBlock, 0, stream>>>(winners + offset,
                                                                                                count);
    check_launch("its zeroing");
    const dim3 grid(static_cast<unsigned int>(tiles(valid_rows, block)), static_cast<unsigned int>(count));
    tiled_atomic_kernel<<<grid, block, 0, stream>>>(rows + static_cast<std::int64_t>(offset) * physical_rows,
                                                   winners + offset, valid_rows, physical_rows);
    check_launch("its tiles");
  }
}

} // namespace ignis::tiled_argmax
