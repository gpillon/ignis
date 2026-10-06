// The Flash-Next forward's side branch (kernel/src/flash_next/side_branch.h) -- OURS, GitHub #306.
//
//   order     work queued on the branch is done before what the layer's stream queues after
//             join_side: a slow write on the branch, read back after the join;
//   failure   inside a capture, an op on the branch that fails after queuing work (as the
//             shared expert's would) still leaves a capture that ends and replays once the
//             caller joins -- the forward's every return path;
//   control   the same capture without the join fails to end: the join is what closes it.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "flash_next/side_branch.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

#define CUDA_OK(expr)                                                                            \
  do {                                                                                           \
    const cudaError_t status_ = (expr);                                                          \
    if (status_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(status_));              \
      std::exit(EXIT_FAILURE);                                                                   \
    }                                                                                            \
  } while (0)

// Spins ~`cycles` before it writes, so a reader not ordered after it would see the old value.
__global__ void slow_write(int *dst, int value, long long cycles) {
  const long long start = clock64();
  while (clock64() - start < cycles) {
  }
  *dst = value;
}

__global__ void copy_one(const int *src, int *dst) { *dst = *src; }

}  // namespace

int main() {
  using ignis::flash_next::join_side;
  using ignis::flash_next::run_beside;
  using ignis::flash_next::SideBranch;

  int devices = 0;
  CUDA_OK(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return EXIT_FAILURE;
  }
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  SideBranch side;
  std::string error;
  check(side.create(&error), "create: " + error);
  int *d = nullptr;
  CUDA_OK(cudaMalloc(&d, 2 * sizeof(int)));

  // order
  {
    CUDA_OK(cudaMemsetAsync(d, 0, 2 * sizeof(int), stream));
    const int32_t rc = run_beside(side, stream, [&](cudaStream_t s) {
      slow_write<<<1, 1, 0, s>>>(d, 42, 20'000'000);
      return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }, &error);
    check(rc == 0, "order: the launch on the branch");
    check(join_side(side, stream), "order: the join");
    copy_one<<<1, 1, 0, stream>>>(d, d + 1);
    int seen = 0;
    CUDA_OK(cudaMemcpyAsync(&seen, d + 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    check(seen == 42, "order: the layer's stream reads the branch's write after the join (got " +
                          std::to_string(seen) + ")");
  }

  // failure: the op queued work and then failed; the caller joins and returns its error
  {
    CUDA_OK(cudaMemsetAsync(d, 0, 2 * sizeof(int), stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    error.clear();
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int32_t rc = run_beside(side, stream, [&](cudaStream_t s) {
      slow_write<<<1, 1, 0, s>>>(d, 7, 1000);
      return -1;
    }, &error);
    check(rc == -1 && error.empty(), "failure: the op's own failure comes back, not the branch's");
    check(join_side(side, stream), "failure: the join");
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    check(ended == cudaSuccess, std::string("failure: the capture ends once joined: ") + cudaGetErrorString(ended));
    if (ended == cudaSuccess) {
      cudaGraphExec_t exec = nullptr;
      CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
      CUDA_OK(cudaGraphLaunch(exec, stream));
      int seen = 0;
      CUDA_OK(cudaMemcpyAsync(&seen, d, sizeof(int), cudaMemcpyDeviceToHost, stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      check(seen == 7, "failure: the replay runs the branch's work");
      cudaGraphExecDestroy(exec);
      cudaGraphDestroy(graph);
    }
  }

  // control: without the join the capture cannot end
  {
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int32_t rc = run_beside(side, stream, [&](cudaStream_t s) {
      slow_write<<<1, 1, 0, s>>>(d, 9, 1000);
      return 0;
    }, &error);
    check(rc == 0, "control: the launch on the branch");
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    check(ended != cudaSuccess, "control: a capture with the branch unjoined must fail to end");
    if (graph != nullptr) cudaGraphDestroy(graph);
    (void)cudaGetLastError();
  }

  cudaFree(d);
  side.destroy();
  cudaStreamDestroy(stream);
  if (g_failed != 0) {
    std::fprintf(stderr, "test_flash_next_side_branch: %d failure(s)\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("test_flash_next_side_branch: OK\n");
  return EXIT_SUCCESS;
}
