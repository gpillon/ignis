// GitHub #711 (spec #711): a device range zeroed by a kernel (kernel/src/device_zero.h) -- OURS.
//
// From a buffer of 0xCD bytes, the op must zero exactly its range, at sizes across its three parts
// (the bytes before a 16-byte boundary, the whole words, the bytes past them) and at offsets that
// misalign it, every byte around the range left 0xCD: run eagerly, and captured into a graph that
// holds one kernel node and no memset node, replayed twice. That graph is what a mixed round's
// attention workspace records under the mixed graph's build (kernel/src/gqa_layer.cu).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "device_zero.h"
#include "graph_nodes.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  } else {
    std::printf("ok   %s\n", what.c_str());
  }
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

// The 0xCD bytes left past the range.
constexpr std::size_t kGuard = 64;

// Whether `host` holds zeros at [offset, offset + bytes) and 0xCD everywhere else.
bool zeroed_exactly(const std::vector<unsigned char> &host, std::size_t offset, std::size_t bytes) {
  for (std::size_t i = 0; i < host.size(); ++i) {
    const bool inside = i >= offset && i < offset + bytes;
    if (host[i] != (inside ? 0 : 0xcd)) {
      return false;
    }
  }
  return true;
}

void run_case(std::size_t offset, std::size_t bytes, cudaStream_t stream) {
  const std::size_t total = offset + bytes + kGuard;
  unsigned char *device = nullptr;
  cuda(cudaMalloc(&device, total), "cudaMalloc");
  std::vector<unsigned char> host(total);
  const std::string label = std::to_string(bytes) + " bytes at offset " + std::to_string(offset);
  const auto poisoned = [&](const auto &run) {
    cuda(cudaMemsetAsync(device, 0xcd, total, stream), "cudaMemsetAsync(poison)");
    run();
    cuda(cudaMemcpyAsync(host.data(), device, total, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
    cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return zeroed_exactly(host, offset, bytes);
  };
  check(poisoned([&] { cuda(ignis::device_zero::zero(device + offset, bytes, stream), "zero"); }),
        label + ", eagerly: its range zeroed, every byte around it as it was");
  cudaGraph_t graph = nullptr;
  cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
  cuda(ignis::device_zero::zero(device + offset, bytes, stream), "zero (captured)");
  cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  std::size_t count = 0;
  cuda(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
  const std::uint32_t kernels = ignis::graph_nodes::of_type(graph, cudaGraphNodeTypeKernel);
  const std::uint32_t memsets = ignis::graph_nodes::of_type(graph, cudaGraphNodeTypeMemset);
  check(count == 1 && kernels == 1 && memsets == 0,
        label + ": its graph holds one kernel node and no memset node (" + std::to_string(count) + " nodes, " +
            std::to_string(kernels) + " kernels, " + std::to_string(memsets) + " memsets)");
  cudaGraphExec_t exec = nullptr;
  cuda(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
  for (int replay = 0; replay < 2; ++replay) {
    check(poisoned([&] { cuda(cudaGraphLaunch(exec, stream), "cudaGraphLaunch"); }),
          label + ", from its graph, replay " + std::to_string(replay + 1) + ": its range zeroed, nothing else");
  }
  cuda(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
  cuda(cudaFree(device), "cudaFree");
}

} // namespace

int main() {
  try {
    cudaStream_t stream = nullptr;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    // Inside one word, one word, a word and a byte, past a block's words, and past every thread of
    // the grid at once (1,024 blocks of 256 threads, 16 bytes each, take 4 MiB a step).
    const std::size_t sizes[] = {1, 15, 16, 17, 4097, (std::size_t{9} << 20) + 3};
    const std::size_t offsets[] = {0, 1, 7};
    for (const std::size_t bytes : sizes) {
      for (const std::size_t offset : offsets) {
        run_case(offset, bytes, stream);
      }
    }
    cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  } catch (const std::exception &e) {
    std::printf("FAIL %s\n", e.what());
    ++failures;
  }
  std::printf("%s ignis_kernel_device_zero_test (%d failures)\n", failures == 0 ? "OK" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
