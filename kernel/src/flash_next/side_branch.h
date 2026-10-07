// ignis kernel leaf -- the Flash-Next forward's side branch (OURS, GitHub #306): a stream beside
// the layer's for work that reads only the sublayer's input, forked from the layer's stream after
// everything queued on it and joined back before what reads the branch's output. The MoE's shared
// expert runs there. Capturable: a fork and its join are edges of the captured graph.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ignis::flash_next {

struct SideBranch {
  cudaStream_t stream = nullptr;
  cudaEvent_t fork = nullptr;
  cudaEvent_t join = nullptr;

  // The stream and both events, or false with *error and nothing left to free.
  bool create(std::string *error) {
    cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
    if (e != cudaSuccess) {
      *error = std::string("the side branch: ") + cudaGetErrorString(e);
      destroy();
      return false;
    }
    return true;
  }

  void destroy() {
    if (join != nullptr) cudaEventDestroy(join);
    if (fork != nullptr) cudaEventDestroy(fork);
    if (stream != nullptr) cudaStreamDestroy(stream);
    *this = SideBranch{};
  }
};

// Queues `launch(side.stream)` on the branch, forked from `stream`, and records the branch's join
// whatever `launch` returned. Returns `launch`'s result (its error is the op's own), or -1 with
// *error when the fork or the join's record failed. On every return but a failed fork the caller
// must `join_side` before it returns itself, or the branch stays open (a capture then fails to
// end).
template <typename Launch>
int32_t run_beside(const SideBranch &side, cudaStream_t stream, Launch &&launch, std::string *error) {
  if (cudaEventRecord(side.fork, stream) != cudaSuccess ||
      cudaStreamWaitEvent(side.stream, side.fork, 0) != cudaSuccess) {
    *error = std::string("the side branch's fork: ") + cudaGetErrorString(cudaGetLastError());
    return -1;
  }
  const int32_t launched = launch(side.stream);
  if (cudaEventRecord(side.join, side.stream) != cudaSuccess) {
    *error = std::string("the side branch's join: ") + cudaGetErrorString(cudaGetLastError());
    return -1;
  }
  return launched;
}

// Makes `stream` wait for everything queued on the branch so far.
inline bool join_side(const SideBranch &side, cudaStream_t stream) {
  return cudaStreamWaitEvent(stream, side.join, 0) == cudaSuccess;
}

}  // namespace ignis::flash_next
