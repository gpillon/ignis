// GitHub #718 (spec #711): the target argmax of a mixed round with its zeroing by a kernel
// (kernel/src/tiled_argmax.h), an adapted op, against the vendored argmax it stands in for -- OURS,
// not vendored.
//
// The oracle is `ninfer::ops::argmax`. At the 27B's verify logits, [248,320, T] BF16 with every row
// valid (the round passes its vocabulary as both counts), and at each column count of the twelve
// mixes (kernel/src/round_layout.h kMixTable at 15 and 7 drafts: 24 to 64 columns), the op must
// write the vendored op's answer bit for bit: run eagerly, and captured into a graph that holds two
// kernel nodes and no memset node, replayed twice. Both start from an output of 0xCD bytes, so
// what zeroes it is what is held. Then each block size the restated launcher picks from the
// vendored one at a shape it does not take in the round: 8 columns (512 threads), the registered
// 248,320 / 248,077-row vocabulary (128) and the 131,072-row one (256). The logits are hashed BF16
// with each column's maximum at two rows in tiles apart (the lower one wins) and, where rows are
// padded, larger values past the valid ones; the vendored answer is checked against a host scan.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "round_layout.h"
#include "tiled_argmax.h"

#include "ninfer/ops/argmax.h"

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

std::uint16_t bf16_of(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof bits);
  return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

float f32_of(std::uint16_t bits) {
  const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &widened, sizeof value);
  return value;
}

// Column t's rows hashed in [-24, -12); its maximum at two rows apart, the lower one first among
// equals; and, past the valid rows, larger values the argmax must not see.
std::vector<std::uint16_t> make_logits(std::int32_t physical, std::int32_t valid, std::int32_t columns) {
  std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical) * columns);
  for (std::int32_t t = 0; t < columns; ++t) {
    const std::size_t base = static_cast<std::size_t>(t) * physical;
    for (std::int32_t row = 0; row < physical; ++row) {
      const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u + static_cast<std::uint32_t>(t + 1) * 1013904223u;
      logits[base + row] = bf16_of(-24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f));
    }
    std::int32_t first = (17 + t * 7919) % valid;
    std::int32_t second = ((valid - 1 - t * 65537) % valid + valid) % valid;
    if (first == second) {
      second = (second + 1) % valid;
    }
    logits[base + first] = bf16_of(32.0f + static_cast<float>(t % 7));
    logits[base + second] = logits[base + first];
    if (valid < physical) {
      logits[base + valid] = bf16_of(64.0f);
      logits[base + physical - 1] = bf16_of(96.0f);
    }
  }
  return logits;
}

std::vector<std::int32_t> host_argmax(const std::vector<std::uint16_t> &logits, std::int32_t physical,
                                      std::int32_t valid, std::int32_t columns) {
  std::vector<std::int32_t> best(static_cast<std::size_t>(columns), 0);
  for (std::int32_t t = 0; t < columns; ++t) {
    const std::size_t base = static_cast<std::size_t>(t) * physical;
    for (std::int32_t row = 1; row < valid; ++row) {
      if (f32_of(logits[base + row]) > f32_of(logits[base + best[t]])) {
        best[t] = row;
      }
    }
  }
  return best;
}

// One shape: the vendored answer, ours eagerly, ours from a graph replayed twice.
void run_case(std::int32_t physical, std::int32_t valid, std::int32_t columns, const std::string &label,
              cudaStream_t stream) {
  const std::vector<std::uint16_t> host = make_logits(physical, valid, columns);
  const std::size_t out_bytes = static_cast<std::size_t>(columns) * sizeof(std::int32_t);
  void *logits_device = nullptr;
  void *out_device = nullptr;
  cuda(cudaMalloc(&logits_device, host.size() * sizeof(std::uint16_t)), "cudaMalloc(logits)");
  cuda(cudaMalloc(&out_device, out_bytes), "cudaMalloc(out)");
  cuda(cudaMemcpy(logits_device, host.data(), host.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
       "cudaMemcpy(logits)");
  // A copy from pageable memory may return before it lands, and `stream` does not wait on it.
  cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(logits)");
  const ninfer::Tensor logits(logits_device, ninfer::DType::BF16, {physical, columns, 1, 1});
  ninfer::Tensor out(out_device, ninfer::DType::I32, {columns, 1, 1, 1});
  const auto poisoned_run = [&](auto &&enqueue) {
    cuda(cudaMemsetAsync(out_device, 0xcd, out_bytes, stream), "cudaMemsetAsync(poison)");
    enqueue();
    cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    std::vector<std::int32_t> got(static_cast<std::size_t>(columns));
    cuda(cudaMemcpy(got.data(), out_device, out_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(out)");
    return got;
  };

  const std::vector<std::int32_t> vendored =
      poisoned_run([&] { ninfer::ops::argmax(logits, out, valid, stream); });
  check(vendored == host_argmax(host, physical, valid, columns), label + ": the vendored argmax is the host scan's");
  const std::vector<std::int32_t> eager =
      poisoned_run([&] { ignis::tiled_argmax::argmax(logits, out, valid, stream); });
  check(eager == vendored, label + ": eagerly, the vendored argmax bit for bit");

  cudaGraph_t graph = nullptr;
  cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
  ignis::tiled_argmax::argmax(logits, out, valid, stream);
  cuda(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  std::size_t count = 0;
  cuda(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
  std::vector<cudaGraphNode_t> nodes(count);
  cuda(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
  std::size_t kernels = 0;
  std::size_t memsets = 0;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeCount;
    cuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    kernels += type == cudaGraphNodeTypeKernel ? 1 : 0;
    memsets += type == cudaGraphNodeTypeMemset ? 1 : 0;
  }
  check(count == 2 && kernels == 2 && memsets == 0,
        label + ": its graph holds two kernel nodes and no memset node (" + std::to_string(count) + " nodes, " +
            std::to_string(kernels) + " kernels, " + std::to_string(memsets) + " memsets)");
  cudaGraphExec_t exec = nullptr;
  cuda(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
  for (int replay = 1; replay <= 2; ++replay) {
    const std::vector<std::int32_t> replayed =
        poisoned_run([&] { cuda(cudaGraphLaunch(exec, stream), "cudaGraphLaunch"); });
    check(replayed == vendored, label + ": replay " + std::to_string(replay) + ", the vendored argmax bit for bit");
  }
  cuda(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
  cuda(cudaFree(out_device), "cudaFree(out)");
  cuda(cudaFree(logits_device), "cudaFree(logits)");
  std::printf("%s: %s\n", label.c_str(), failures == 0 ? "ok" : "failures so far");
}

} // namespace

int main() {
  try {
    cudaStream_t stream = nullptr;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    // The 27B's verify logits at each mix's columns.
    constexpr std::int32_t kVocab = 248320;
    std::set<std::int32_t> mixed_columns;
    for (const auto &mix : ignis::round_layout::kMixTable) {
      mixed_columns.insert(static_cast<std::int32_t>(ignis::round_layout::layout_of(mix[0], 15, mix[1], 7).columns()));
    }
    check(mixed_columns == std::set<std::int32_t>{24, 32, 40, 48, 56, 64}, "the twelve mixes take 24 to 64 columns");
    for (const std::int32_t columns : mixed_columns) {
      run_case(kVocab, kVocab, columns, "the 27B's verify logits at " + std::to_string(columns) + " columns", stream);
    }
    // The launcher's other blocks.
    run_case(kVocab, kVocab, 8, "8 columns (512 threads)", stream);
    run_case(kVocab, 248077, 24, "the registered 248,077 valid rows at 24 columns (128 threads)", stream);
    run_case(131072, 131072, 24, "the 131,072-row vocabulary at 24 columns (256 threads)", stream);
    cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
  } catch (const std::exception &e) {
    std::printf("FAIL %s\n", e.what());
    ++failures;
  }
  std::printf("%s ignis_kernel_tiled_argmax_test (%d failures)\n", failures == 0 ? "OK" : "FAILED", failures);
  return failures == 0 ? 0 : 1;
}
