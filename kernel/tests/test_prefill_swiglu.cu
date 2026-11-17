// GitHub #651 (spec #646): the 27B prefill's gate and up product with SwiGLU in its epilogue
// (kernel/src/prefill_swiglu.h) -- an ADAPTED op (ADR 0051).
//
// At the 27B's gate and up shape (an NVFP4 [34816, 5120] weight, x [5120, T] BF16), at 1,024,
// 1,280, 2,048 and 4,096 tokens, the op's BF16 [17408, T] output is the split route's byte for
// byte: the vendored W4A4 product into [34816, T] (`ops::linear` under AllowA4), then
// `ops::silu_mul` over its gate and up halves, what the 27B's prefill ran at every width but
// 1,024 before the op. The same on a stream a descriptor cache serves (kernel/src/
// tma_descriptor_cache.h), where Windows reads each width's block from a slot uploaded once.
//
// The positive control: at 1,024 tokens the vendored fused kernel, which multiplies gate and up
// in fp32 before one rounding, writes other bytes than the split route, so a byte compare here
// tells the two roundings apart. Beside it, the widths the op takes and the measurement switch
// IGNIS_PREFILL_SWIGLU's parse.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing GPU fails this test.

#include "prefill_swiglu.h"
#include "tma_descriptor_cache.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/quantized_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace {

namespace ps = ignis::prefill_swiglu;
using ninfer::DType;
using ninfer::Tensor;

constexpr std::int32_t kInput = 5120;
constexpr std::int32_t kGateUp = 34816;
constexpr std::int32_t kIntermediate = kGateUp / 2;
constexpr std::int32_t kWidths[] = {1024, 1280, 2048, 4096};
constexpr std::int32_t kMaxTokens = 4096;

// The widths the op takes: the vendored W4A4 TMA route's, from 1,024 tokens in whole 256-token
// tiles; every other width keeps the split route.
static_assert(ps::takes(1024) && ps::takes(1280) && ps::takes(2048) && ps::takes(4096) && ps::takes(8192));
static_assert(!ps::takes(1) && !ps::takes(128) && !ps::takes(768) && !ps::takes(1023) && !ps::takes(1152) &&
              !ps::takes(2000));

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

std::uint64_t g_state = 0x9E3779B97F4A7C15ull;
std::uint32_t next_u32() {
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  return static_cast<std::uint32_t>(g_state >> 16);
}
float next_normal() {
  const float u1 = (next_u32() + 1.0f) / 4294967296.0f;
  const float u2 = next_u32() / 4294967296.0f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

void *device_alloc(std::size_t bytes) {
  void *p = nullptr;
  cuda(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

std::vector<std::uint16_t> host_bf16(const Tensor &t, std::size_t count) {
  std::vector<std::uint16_t> host(count);
  cuda(cudaMemcpy(host.data(), t.data, count * 2, cudaMemcpyDeviceToHost), "cudaMemcpy");
  return host;
}

std::size_t differing(const std::vector<std::uint16_t> &a, const std::vector<std::uint16_t> &b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

void set_switch(const char *value) {
#ifdef _WIN32
  SetEnvironmentVariableA("IGNIS_PREFILL_SWIGLU", value);
#else
  if (value != nullptr) {
    setenv("IGNIS_PREFILL_SWIGLU", value, 1);
  } else {
    unsetenv("IGNIS_PREFILL_SWIGLU");
  }
#endif
}

// Whether the switch's parse refuses `value`, naming the variable.
bool refused(const char *value) {
  set_switch(value);
  try {
    (void)ps::route_for_load();
  } catch (const std::invalid_argument &e) {
    return std::string(e.what()).find("IGNIS_PREFILL_SWIGLU") != std::string::npos;
  }
  return false;
}

}  // namespace

int main() {
  // The switch: unset or `fused` the op, `split` today's route; `quant` (the quantizing epilogue,
  // not in this build) and anything else refused by name.
  set_switch(nullptr);
  check(ps::route_for_load() == ps::Route::Fused, "IGNIS_PREFILL_SWIGLU unset: fused");
  set_switch("fused");
  check(ps::route_for_load() == ps::Route::Fused, "IGNIS_PREFILL_SWIGLU=fused");
  set_switch("split");
  check(ps::route_for_load() == ps::Route::Split, "IGNIS_PREFILL_SWIGLU=split");
  check(refused("quant"), "IGNIS_PREFILL_SWIGLU=quant refused, naming the variable");
  check(refused("Fused"), "IGNIS_PREFILL_SWIGLU=Fused refused, naming the variable");
  set_switch(nullptr);

  cuda(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t plain = nullptr;
  cudaStream_t served = nullptr;
  cuda(cudaStreamCreate(&plain), "cudaStreamCreate");
  cuda(cudaStreamCreate(&served), "cudaStreamCreate");

  // x ~ N(0, 1); the weight's E2M1 codes uniform and its E4M3 scales in [0.25, 2), the input
  // divisor the activations' own (448 x 6 over their largest magnitude).
  std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kMaxTokens) * kInput);
  float amax = 0.0f;
  for (auto &v : host_x) {
    const __nv_bfloat16 b = __float2bfloat16(next_normal());
    amax = std::max(amax, std::fabs(__bfloat162float(b)));
    v = *reinterpret_cast<const std::uint16_t *>(&b);
  }
  ninfer::test::quantized_weight::PatternedWeightOptions options;
  options.weight_scale_divisor = 64.0f;
  options.input_scale_divisor = 448.0f * 6.0f / amax;
  ninfer::test::quantized_weight::PackedWeight packed = ninfer::test::quantized_weight::make_patterned_weight(
      ninfer::QType::NVFP4, kGateUp, kInput, 651U, options);
  for (std::uint64_t i = 0; i < packed.code_plane_bytes; ++i) {
    packed.payload[i] = static_cast<std::uint8_t>(next_u32());
  }
  for (std::uint64_t i = 0; i < packed.scale_plane_bytes; ++i) {
    packed.payload[packed.scale_plane_offset + i] = static_cast<std::uint8_t>(0x28 + next_u32() % 0x18);
  }
  void *weight_payload = device_alloc(packed.payload.size());
  cuda(cudaMemcpy(weight_payload, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice),
       "cudaMemcpy");
  const ninfer::Weight gate_up = packed.device_weight(weight_payload);

  void *x_data = device_alloc(host_x.size() * 2);
  cuda(cudaMemcpy(x_data, host_x.data(), host_x.size() * 2, cudaMemcpyHostToDevice), "cudaMemcpy");
  void *projected_data = device_alloc(static_cast<std::size_t>(kGateUp) * kMaxTokens * 2);
  const std::size_t out_bytes = static_cast<std::size_t>(kIntermediate) * kMaxTokens * 2;
  void *split_data = device_alloc(out_bytes);
  void *fused_data = device_alloc(out_bytes);
  ninfer::WorkspaceArena workspace(std::max(
      ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::NVFP4, kGateUp, kInput,
                                                   ninfer::ops::LinearPolicy::AllowA4, 1, kMaxTokens),
      ninfer::ops::linear_swiglu_workspace_capacity_bytes(ninfer::QType::NVFP4, kGateUp, kInput,
                                                          ninfer::ops::LinearPolicy::AllowA4, 1, kMaxTokens)));

  ignis::tma_descriptors::Cache cache(4, served);
  for (const std::int32_t tokens : kWidths) {
    const std::string at = " at " + std::to_string(tokens) + " tokens";
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * tokens;
    const Tensor x(x_data, DType::BF16, {kInput, tokens});
    Tensor projected(projected_data, DType::BF16, {kGateUp, tokens});
    Tensor split(split_data, DType::BF16, {kIntermediate, tokens});
    Tensor fused(fused_data, DType::BF16, {kIntermediate, tokens});

    // The split route: the product, then SwiGLU over its BF16 gate and up halves.
    cuda(cudaMemset(split_data, 0, out_bytes), "cudaMemset");
    workspace.reset();
    ninfer::ops::linear(x, gate_up, projected, ninfer::ops::LinearPolicy::AllowA4, workspace, plain);
    ninfer::ops::silu_mul(projected.slice(0, 0, kIntermediate), projected.slice(0, kIntermediate, kIntermediate),
                          split, plain);
    cuda(cudaStreamSynchronize(plain), "split route");
    const std::vector<std::uint16_t> expected = host_bf16(split, count);
    std::size_t finite = 0;
    std::size_t nonzero = 0;
    for (const std::uint16_t v : expected) {
      finite += (v & 0x7f80u) != 0x7f80u;
      nonzero += (v & 0x7fffu) != 0;
    }
    check(finite == count && nonzero > count / 2,
          "the split route's output is finite and mostly nonzero" + at + " (" + std::to_string(nonzero) + " of " +
              std::to_string(count) + " nonzero)");

    for (const auto &[stream, name] : {std::pair{plain, "a plain stream"}, std::pair{served, "a served stream"}}) {
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ps::gate_up_swiglu(x, gate_up, fused, workspace, stream);
      cuda(cudaStreamSynchronize(stream), "fused op");
      const std::size_t n = differing(host_bf16(fused, count), expected);
      check(n == 0, std::string("the fused op on ") + name + " writes the split route's bytes" + at + " (" +
                        std::to_string(n) + " differ)");
    }

    if (tokens == 1024) {
      // The positive control: the vendored dispatch at 1,024 is its fused kernel.
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ninfer::ops::linear_swiglu(x, gate_up, fused, ninfer::ops::LinearPolicy::AllowA4, workspace, plain);
      cuda(cudaStreamSynchronize(plain), "vendored fused kernel");
      const std::size_t n = differing(host_bf16(fused, count), expected);
      check(n > 0, "the vendored fused kernel at 1,024 tokens writes other bytes (" + std::to_string(n) +
                       " of " + std::to_string(count) + " differ)");
    }
  }
  // Each width's descriptor block, once: the same widths again read their slots.
  for (const std::int32_t tokens : kWidths) {
    const Tensor x(x_data, DType::BF16, {kInput, tokens});
    Tensor fused(fused_data, DType::BF16, {kIntermediate, tokens});
    workspace.reset();
    ps::gate_up_swiglu(x, gate_up, fused, workspace, served);
  }
  cuda(cudaStreamSynchronize(served), "fused op");
  const std::uint64_t blocks = ignis::tma_descriptors::kThroughPointer ? 4 : 0;
  check(cache.counts().uploads == blocks && cache.counts().resident == blocks,
        "on a served stream each width's block is uploaded once, twice over the four widths (" +
            std::to_string(cache.counts().uploads) + " uploads)");

  cuda(cudaStreamDestroy(plain), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(served), "cudaStreamDestroy");
  if (g_failed != 0) {
    std::printf("%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
