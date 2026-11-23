// ignis kernel leaf - GitHub #711 (spec #711): a device range zeroed by a kernel -- OURS
// (kernel/src/device_zero.h).
#include "device_zero.h"

#include <algorithm>
#include <cstdint>

namespace ignis::device_zero {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kMaxBlocks = 1024;
constexpr std::size_t kWord = sizeof(uint4);

// The bytes before the first 16-byte boundary one a thread, then 16 bytes a thread a step, then the
// bytes past the last whole word one a thread, each over the grid's threads.
__global__ void zero_bytes_kernel(unsigned char *data, std::size_t bytes) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t misaligned = reinterpret_cast<std::uintptr_t>(data) % kWord;
  const std::size_t lead = misaligned == 0 ? 0 : (kWord - misaligned < bytes ? kWord - misaligned : bytes);
  for (std::size_t i = first; i < lead; i += stride) {
    data[i] = 0;
  }
  const std::size_t words = (bytes - lead) / kWord;
  uint4 *aligned = reinterpret_cast<uint4 *>(data + lead);
  for (std::size_t i = first; i < words; i += stride) {
    aligned[i] = make_uint4(0, 0, 0, 0);
  }
  for (std::size_t i = lead + words * kWord + first; i < bytes; i += stride) {
    data[i] = 0;
  }
}

} // namespace

cudaError_t zero(void *data, std::size_t bytes, cudaStream_t stream) {
  if (bytes == 0) {
    return cudaSuccess;
  }
  const std::size_t steps = bytes / kWord + 1;
  const auto blocks = static_cast<unsigned>(std::min((steps + kThreads - 1) / kThreads, kMaxBlocks));
  zero_bytes_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<unsigned char *>(data), bytes);
  return cudaGetLastError();
}

} // namespace ignis::device_zero
