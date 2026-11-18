// GitHub #681 (spec #672): the 27B's GDN gating on its branch (kernel/src/gdn_gating_home.h)
// against the chain's, the vendored op on the round's stream -- OURS, not vendored.
//
// The gating's oracle is `ninfer::ops::gdn_gating_proj` at the 27B's shape (the a|b parent
// [96, 5120], 48 heads), which runs one kernel at one column and a split-K pair at 2..8; the branch
// must leave its g and beta byte for byte, at every width the decode round runs (1..8).
//
//   branch   `ignis::gdn_gating::Beside`, the vendored op on a stream of its own forked after the
//            input is written and joined back, captured into a graph and replayed, then run
//            eagerly, while the round's stream allocates from the same scratch arena and fills
//            what it got with 0xFF before the join, as the input projection does beside it. Its
//            workspace must be a span of that arena the round's later allocations do not reach.
//            At 1..8 columns and at 16 and 64, the verify round's at one and at eight lanes, where
//            the op takes its cooperative tensor-core form.
//   switch   IGNIS_GDN_GATING as a capture reads it: unset is kUnset; `chain`, `branch`; any other
//            value refused by name; a card of compute capability 12 takes the switch's home and
//            every other the chain.
//
// The inputs are hashed BF16, a few a rows scaled so that softplus takes its linear branch too.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy GPU fails this
// test rather than skipping it.

#include "gdn_gating_home.h"
#include "moe_fixture.h"

#include "ninfer/ops/gdn_gating_proj.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace moe_test;

namespace {

// The 27B's gating (crates/core/src/compute.rs `ModelConfig::qwen38_27b`).
constexpr int32_t kHeads = 48, kHidden = 5120, kRows = 2 * kHeads;

std::vector<uint16_t> bf16_values(uint32_t stream, std::size_t n, float amplitude) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = f32_to_bf16(hash_uniform(stream, i, amplitude));
    if (i % 61 == 0) v[i] = 0x0000u;
    if (i % 67 == 0) v[i] = 0x8000u;
  }
  return v;
}

std::vector<float> f32_values(uint32_t stream, std::size_t n, float amplitude) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = hash_uniform(stream, i, amplitude);
  return v;
}

ninfer::Weight bf16_weight(void *data, int32_t rows, int32_t cols) {
  ninfer::Weight weight{};
  weight.qtype = ninfer::QType::BF16_CTRL;
  weight.layout = ninfer::QuantLayout::Contiguous;
  weight.payload = data;
  weight.payload_bytes = static_cast<uint64_t>(rows) * cols * sizeof(uint16_t);
  weight.qdata = data;
  weight.ndim = 2;
  weight.shape[0] = rows;
  weight.shape[1] = cols;
  weight.padded_shape[0] = rows;
  weight.padded_shape[1] = cols;
  weight.n = rows;
  weight.k = cols;
  return weight;
}

// One round's gating inputs at `columns` columns, uploaded.
struct Gating {
  int32_t columns;
  DeviceBytes h, ab, a_log, dt_bias;
  explicit Gating(int32_t T)
      : columns(T), h(static_cast<std::size_t>(kHidden) * T * 2), ab(static_cast<std::size_t>(kRows) * kHidden * 2),
        a_log(kHeads * 4), dt_bias(kHeads * 4) {
    upload(h, bf16_values(300 + static_cast<uint32_t>(T), static_cast<std::size_t>(kHidden) * T, 2.0f));
    std::vector<uint16_t> w = bf16_values(31, static_cast<std::size_t>(kRows) * kHidden, 0.05f);
    // Four a rows eight times as large: their sums pass softplus's threshold of 20 in some columns.
    for (std::size_t i = 0; i < static_cast<std::size_t>(4) * kHidden; ++i) w[i] = f32_to_bf16(8.0f * bf16_to_f32(w[i]));
    upload(ab, w);
    upload(a_log, f32_values(33, kHeads, 2.0f));
    upload(dt_bias, f32_values(35, kHeads, 1.0f));
  }
  ninfer::Tensor th() const { return ninfer::Tensor(h.p, ninfer::DType::BF16, {kHidden, columns, 1, 1}); }
  ninfer::Weight tab() const { return bf16_weight(ab.p, kRows, kHidden); }
  ninfer::Tensor ta() const { return ninfer::Tensor(a_log.p, ninfer::DType::FP32, {kHeads, 1, 1, 1}); }
  ninfer::Tensor tdt() const { return ninfer::Tensor(dt_bias.p, ninfer::DType::FP32, {kHeads, 1, 1, 1}); }
};

// The gates as bits.
struct Gates {
  std::vector<uint32_t> g, beta;
};

struct GateBuffers {
  int32_t columns;
  DeviceBytes g, beta;
  explicit GateBuffers(int32_t T) : columns(T), g(static_cast<std::size_t>(kHeads) * T * 4), beta(static_cast<std::size_t>(kHeads) * T * 4) {
    MOE_CUDA(cudaMemset(g.p, 0, g.bytes));
    MOE_CUDA(cudaMemset(beta.p, 0, beta.bytes));
    MOE_CUDA(cudaDeviceSynchronize());
  }
  ninfer::Tensor tg() const { return ninfer::Tensor(g.p, ninfer::DType::FP32, {kHeads, columns, 1, 1}); }
  ninfer::Tensor tbeta() const { return ninfer::Tensor(beta.p, ninfer::DType::FP32, {kHeads, columns, 1, 1}); }
  Gates read() const {
    return {download<uint32_t>(g.p, static_cast<std::size_t>(kHeads) * columns),
            download<uint32_t>(beta.p, static_cast<std::size_t>(kHeads) * columns)};
  }
};

std::size_t workspace_bytes(int32_t T) {
  return std::max<std::size_t>(ninfer::ops::gdn_gating_proj_workspace_capacity_bytes(kHeads, kHidden, T, T), 1);
}

// The chain's gating: the vendored op on the round's stream.
Gates vendored_gating(const Gating &in, cudaStream_t stream) {
  GateBuffers out(in.columns);
  ninfer::DeviceArena ws(workspace_bytes(in.columns) + 4096);
  const ninfer::Tensor h = in.th(), a_log = in.ta(), dt_bias = in.tdt();
  const ninfer::Weight ab = in.tab();
  ninfer::Tensor g = out.tg(), beta = out.tbeta();
  ninfer::ops::gdn_gating_proj(h, ab, a_log, dt_bias, ws, g, beta, stream);
  MOE_CUDA(cudaStreamSynchronize(stream));
  return out.read();
}

template <typename T>
void same_bits(const std::vector<T> &got, const std::vector<T> &want, const std::string &label) {
  std::size_t wrong = 0, first = 0;
  for (std::size_t i = 0; i < want.size() && i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (wrong == 0) first = i;
      ++wrong;
    }
  }
  check(got.size() == want.size() && wrong == 0,
        label + ": " + std::to_string(wrong) + " of " + std::to_string(want.size()) + " elements differ" +
            (wrong == 0 ? std::string() : ", the first at " + std::to_string(first)));
}

void same_gates(const Gates &got, const Gates &want, const std::string &label) {
  same_bits(got.g, want.g, label + " g");
  same_bits(got.beta, want.beta, label + " beta");
}

// --- branch ---------------------------------------------------------------------------------

// The branch home at `T` columns: the round's stream writes h from a staging copy, the gating
// forks after it, the round's stream then takes a span of the same arena and fills it, and the
// branch joins. Captured and replayed (`graph`), or run as it is enqueued.
Gates branch(const Gating &in, const ignis::flash_next::SideBranch &side, bool graph, cudaStream_t stream) {
  const int32_t T = in.columns;
  GateBuffers out(T);
  DeviceBytes h(static_cast<std::size_t>(kHidden) * T * 2);
  MOE_CUDA(cudaMemset(h.p, 0, h.bytes));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::size_t ws = workspace_bytes(T);
  ninfer::DeviceArena scratch(2 * ws + 8192);
  const ninfer::Tensor th(h.p, ninfer::DType::BF16, {kHidden, T, 1, 1});
  const ninfer::Tensor a_log = in.ta(), dt_bias = in.tdt();
  const ninfer::Weight ab = in.tab();
  ninfer::Tensor g = out.tg(), beta = out.tbeta();

  if (graph) MOE_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  {
    ninfer::DeviceArena::Scope scope = scratch.scope();
    MOE_CUDA(cudaMemcpyAsync(h.p, in.h.p, h.bytes, cudaMemcpyDeviceToDevice, stream));
    const std::size_t before = scratch.used();
    const ignis::gdn_gating::Gating gating{th, ab, a_log, dt_bias, g, beta};
    ignis::gdn_gating::Beside beside(side, stream, gating, scratch);
    check(scratch.used() - before >= ws, "branch, T=" + std::to_string(T) + ": the gating's workspace is the caller's (" +
                                             std::to_string(scratch.used() - before) + " of " + std::to_string(ws) +
                                             " bytes held)");
    // What the input projection does beside the branch: its own workspace from the same arena.
    const ninfer::DeviceSpan projection = scratch.alloc_bytes(ws);
    MOE_CUDA(cudaMemsetAsync(projection.data, 0xFF, projection.bytes, stream));
    beside.join();
  }
  if (graph) {
    cudaGraph_t captured = nullptr;
    MOE_CUDA(cudaStreamEndCapture(stream, &captured));
    cudaGraphExec_t exec = nullptr;
    MOE_CUDA(cudaGraphInstantiate(&exec, captured, 0));
    MOE_CUDA(cudaGraphDestroy(captured));
    MOE_CUDA(cudaGraphLaunch(exec, stream));
    MOE_CUDA(cudaStreamSynchronize(stream));
    MOE_CUDA(cudaGraphExecDestroy(exec));
  }
  MOE_CUDA(cudaStreamSynchronize(stream));
  return out.read();
}

void the_branch_home_leaves_the_chains_gates(cudaStream_t stream) {
  ignis::flash_next::SideBranch side;
  std::string error;
  check(side.create(&error), "the branch's stream and events" + (error.empty() ? std::string() : ": " + error));
  for (const int32_t T : {1, 2, 3, 4, 5, 6, 7, 8, 16, 64}) {
    const Gating in(T);
    const Gates want = vendored_gating(in, stream);
    same_gates(branch(in, side, /*graph=*/true, stream), want, "branch, T=" + std::to_string(T) + ", replayed");
    same_gates(branch(in, side, /*graph=*/false, stream), want, "branch, T=" + std::to_string(T) + ", eager");
  }
  side.destroy();
}

// --- switch ---------------------------------------------------------------------------------

void set_environment(const char *name, const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA(name, value);
#else
  if (value != nullptr) {
    setenv(name, value, 1);
  } else {
    unsetenv(name);
  }
#endif
}

void the_switch_names_the_home() {
  using ignis::gdn_gating::Home;
  using ignis::gdn_gating::home_on;
  using ignis::gdn_gating::switch_at_capture;
  constexpr const char *kSwitch = "IGNIS_GDN_GATING";
  set_environment(kSwitch, nullptr);
  check(switch_at_capture() == ignis::gdn_gating::kUnset, "unset, the kept home");
  set_environment(kSwitch, "chain");
  check(switch_at_capture() == Home::Chain, "IGNIS_GDN_GATING=chain, the round of earlier releases");
  set_environment(kSwitch, "branch");
  check(switch_at_capture() == Home::Branch, "IGNIS_GDN_GATING=branch");
  for (const char *bad : {"0", "1", "Chain", "fused"}) {
    set_environment(kSwitch, bad);
    bool refused = false;
    try {
      (void)switch_at_capture();
    } catch (const std::invalid_argument &e) {
      refused = std::string(e.what()).find(kSwitch) != std::string::npos;
    }
    check(refused, std::string("IGNIS_GDN_GATING=") + bad + " is refused by name");
  }
  set_environment(kSwitch, nullptr);
  for (const Home named : {Home::Chain, Home::Branch}) {
    check(home_on(named, 12) == named, "compute capability 12 takes the switch's home");
    for (const int major : {7, 8, 9, 10, 11, 13}) {
      check(home_on(named, major) == Home::Chain,
            "a card of compute capability " + std::to_string(major) + ".x keeps the chain");
    }
  }
}

} // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "FAIL: no CUDA device\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  the_branch_home_leaves_the_chains_gates(stream);
  the_switch_names_the_home();
  MOE_CUDA(cudaStreamDestroy(stream));
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
