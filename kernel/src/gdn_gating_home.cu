// ignis kernel leaf: the GDN gating's branch home (kernel/src/gdn_gating_home.h; spec #672,
// GitHub #681) -- OURS. The product is the vendored op's, called and not copied.

#include "gdn_gating_home.h"

#include "ninfer/ops/gdn_gating_proj.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ignis::gdn_gating {

namespace {

void check(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("the gating's branch: ") + what + ": " + cudaGetErrorString(err));
  }
}

} // namespace

Beside::Beside(const ignis::flash_next::SideBranch &branch, cudaStream_t stream, const ninfer::Tensor &h,
               const ninfer::Weight &ab, const ninfer::Tensor &a_log, const ninfer::Tensor &dt_bias,
               ninfer::DeviceArena &scratch, ninfer::Tensor &g, ninfer::Tensor &beta)
    : branch_(branch), stream_(stream) {
  if (branch.stream == nullptr) {
    throw std::invalid_argument("the gating's branch: no stream (the capture makes it)");
  }
  // The op's own workspace query at this call's columns: the span is the caller's until its scope of
  // `scratch` ends, and the op suballocates it as its own arena.
  const std::size_t bytes = std::max<std::size_t>(
      ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(ab.n / 2, ab.k, h.ne[1], h.ne[1]), 1);
  const ninfer::DeviceSpan span = scratch.alloc_bytes(bytes);
  check(cudaEventRecord(branch.fork, stream), "the fork");
  check(cudaStreamWaitEvent(branch.stream, branch.fork, 0), "the fork");
  open_ = true;
  try {
    ninfer::DeviceArena workspace(span);
    ninfer::ops::gdn_gating_proj(h, ab, a_log, dt_bias, workspace, g, beta, branch.stream);
    check(cudaEventRecord(branch.join, branch.stream), "the join");
  } catch (...) {
    (void)cudaEventRecord(branch.join, branch.stream);
    join_quietly();
    throw;
  }
}

Beside::~Beside() {
  join_quietly();
}

void Beside::join() {
  open_ = false;
  check(cudaStreamWaitEvent(stream_, branch_.join, 0), "the join");
}

void Beside::join_quietly() noexcept {
  if (open_) {
    open_ = false;
    (void)cudaStreamWaitEvent(stream_, branch_.join, 0);
  }
}

} // namespace ignis::gdn_gating
