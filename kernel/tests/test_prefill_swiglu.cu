// GitHub #651, #653 (spec #646): the 27B prefill's gate and up product with SwiGLU in its epilogue
// (kernel/src/prefill_swiglu.h) -- an ADAPTED op (ADR 0051) -- in its two forms.
//
// At the 27B's MLP shapes (an NVFP4 gate and up [34816, 5120] weight, x [5120, T] BF16, an NVFP4
// down [5120, 17408] weight), at 1,024, 1,280, 2,048 and 4,096 tokens, against the split route,
// what the 27B's prefill ran at every width but 1,024 before the op: the vendored W4A4 product
// into [34816, T] (`ops::linear` under AllowA4), `ops::silu_mul` over its gate and up halves, the
// down product's quantizer (`launch_nvfp4_w4a4_quantize_for_products`) and the down product
// (`ops::linear_add` under AllowA4):
//   - the BF16 epilogue's [17408, T] output is the SiLU-mul's, byte for byte (GitHub #651);
//   - the NVFP4 epilogue's codes and scales are the down product's quantizer's, in the layout that
//     quantizer writes, and the layers' MLP under Quant leaves the residual the split route's down
//     product leaves, byte for byte, without writing the BF16 matrix (GitHub #653);
// in the default K-tile-major layout, on a plain stream and on one a descriptor cache serves
// (kernel/src/tma_descriptor_cache.h), where Windows reads each block from a slot uploaded once,
// and token-major on a stream a load names so (kernel/src/w4a4_activation_layout.h).
//
// The positive control: at 1,024 tokens the vendored fused kernel, which multiplies gate and up
// in fp32 before one rounding, writes other bytes than the split route, so a byte compare there
// tells the two roundings apart. There, under Quant, the layers' MLP whose down product is off the
// W4A4 route runs the BF16 epilogue: the split route's BF16 bytes. Beside it, the widths the op
// takes, the layers' entry under each route (the op at 1,024 under Quant and Fused, the vendored
// dispatch under Split and at a width the op does not take), the descriptor blocks the MLP uploads
// on a served stream (two a width, the op's and the down product's, as the plan counts), and the
// measurement switch IGNIS_PREFILL_SWIGLU's parse.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing GPU fails this test.

#include "prefill_swiglu.h"
#include "tma_descriptor_cache.h"
#include "w4a4_activation_layout.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/quantized_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
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
namespace nd = ninfer::ops::detail;
namespace layout = ignis::w4a4_activation_layout;
using ninfer::DType;
using ninfer::Tensor;
using ninfer::ops::LinearPolicy;

constexpr std::int32_t kInput = 5120;
constexpr std::int32_t kGateUp = 34816;
constexpr std::int32_t kIntermediate = kGateUp / 2;
constexpr std::int32_t kWidths[] = {1024, 1280, 2048, 4096};
constexpr std::int32_t kMaxTokens = 4096;
// The down product's NVFP4 input per token: codes and one E4M3 scale per 16 values.
constexpr std::size_t kCodeBytesPerToken = kIntermediate / 2;
constexpr std::size_t kScaleBytesPerToken = kIntermediate / 16;

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

template <class T> std::vector<T> host_copy(const void *device, std::size_t count) {
  std::vector<T> host(count);
  cuda(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
  return host;
}

std::vector<std::uint16_t> host_bf16(const Tensor &t, std::size_t count) {
  return host_copy<std::uint16_t>(t.data, count);
}

template <class T> std::size_t differing(const std::vector<T> &a, const std::vector<T> &b) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

// An NVFP4 [rows, cols] weight: its E2M1 codes uniform, its E4M3 scales in [0.25, 2), the input
// divisor `input_divisor`.
ninfer::Weight nvfp4_weight(std::int32_t rows, std::int32_t cols, std::uint32_t seed, float input_divisor) {
  ninfer::test::quantized_weight::PatternedWeightOptions options;
  options.weight_scale_divisor = 64.0f;
  options.input_scale_divisor = input_divisor;
  ninfer::test::quantized_weight::PackedWeight packed =
      ninfer::test::quantized_weight::make_patterned_weight(ninfer::QType::NVFP4, rows, cols, seed, options);
  for (std::uint64_t i = 0; i < packed.code_plane_bytes; ++i) {
    packed.payload[i] = static_cast<std::uint8_t>(next_u32());
  }
  for (std::uint64_t i = 0; i < packed.scale_plane_bytes; ++i) {
    packed.payload[packed.scale_plane_offset + i] = static_cast<std::uint8_t>(0x28 + next_u32() % 0x18);
  }
  void *payload = device_alloc(packed.payload.size());
  cuda(cudaMemcpy(payload, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
  return packed.device_weight(payload);
}

// The largest magnitude among BF16 values.
float amax_bf16(const std::vector<std::uint16_t> &values) {
  float amax = 0.0f;
  for (const std::uint16_t v : values) {
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, 4);
    amax = std::max(amax, std::fabs(f));
  }
  return amax;
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
  // The switch: unset or `quant` the NVFP4 epilogue where it applies, `fused` the BF16 epilogue,
  // `split` today's route; anything else refused by name.
  set_switch(nullptr);
  check(ps::route_for_load() == ps::Route::Quant, "IGNIS_PREFILL_SWIGLU unset: quant");
  set_switch("quant");
  check(ps::route_for_load() == ps::Route::Quant, "IGNIS_PREFILL_SWIGLU=quant");
  set_switch("fused");
  check(ps::route_for_load() == ps::Route::Fused, "IGNIS_PREFILL_SWIGLU=fused");
  set_switch("split");
  check(ps::route_for_load() == ps::Route::Split, "IGNIS_PREFILL_SWIGLU=split");
  check(refused("Quant"), "IGNIS_PREFILL_SWIGLU=Quant refused, naming the variable");
  check(refused("bf16"), "IGNIS_PREFILL_SWIGLU=bf16 refused, naming the variable");
  set_switch(nullptr);

  cuda(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t plain = nullptr;
  cudaStream_t served = nullptr;
  cudaStream_t token = nullptr;
  cuda(cudaStreamCreate(&plain), "cudaStreamCreate");
  cuda(cudaStreamCreate(&served), "cudaStreamCreate");
  cuda(cudaStreamCreate(&token), "cudaStreamCreate");
  // A stream no load names reads K-tile-major, the default; `token` is named token-major, as a
  // load under IGNIS_W4A4_ACTIVATION_LAYOUT=token names its stream.
  const layout::ForStream token_major(token, layout::Layout::Token);

  // x ~ N(0, 1), the gate and up's input divisor the activations' own (448 x 6 over their
  // largest magnitude).
  std::vector<std::uint16_t> host_x(static_cast<std::size_t>(kMaxTokens) * kInput);
  float amax = 0.0f;
  for (auto &v : host_x) {
    const __nv_bfloat16 b = __float2bfloat16(next_normal());
    amax = std::max(amax, std::fabs(__bfloat162float(b)));
    v = *reinterpret_cast<const std::uint16_t *>(&b);
  }
  const ninfer::Weight gate_up = nvfp4_weight(kGateUp, kInput, 651U, 448.0f * 6.0f / amax);

  void *x_data = device_alloc(host_x.size() * 2);
  cuda(cudaMemcpy(x_data, host_x.data(), host_x.size() * 2, cudaMemcpyHostToDevice), "cudaMemcpy");
  void *projected_data = device_alloc(static_cast<std::size_t>(kGateUp) * kMaxTokens * 2);
  const std::size_t out_bytes = static_cast<std::size_t>(kIntermediate) * kMaxTokens * 2;
  void *split_data = device_alloc(out_bytes);
  void *fused_data = device_alloc(out_bytes);
  ninfer::WorkspaceArena workspace(std::max(
      {ninfer::ops::linear_workspace_capacity_bytes(ninfer::QType::NVFP4, kGateUp, kInput, LinearPolicy::AllowA4, 1,
                                                    kMaxTokens),
       ninfer::ops::linear_swiglu_workspace_capacity_bytes(ninfer::QType::NVFP4, kGateUp, kInput,
                                                           LinearPolicy::AllowA4, 1, kMaxTokens),
       ninfer::ops::linear_add_workspace_capacity_bytes(ninfer::QType::NVFP4, kInput, kIntermediate,
                                                        LinearPolicy::AllowA4, 1, kMaxTokens)}));

  // The down weight's input divisor is the SwiGLU output's own, as a calibration sets it: the split
  // route's output at the widest width, whose first tokens every narrower width's are.
  float h_amax = 0.0f;
  {
    const Tensor x(x_data, DType::BF16, {kInput, kMaxTokens});
    Tensor projected(projected_data, DType::BF16, {kGateUp, kMaxTokens});
    Tensor split(split_data, DType::BF16, {kIntermediate, kMaxTokens});
    workspace.reset();
    ninfer::ops::linear(x, gate_up, projected, LinearPolicy::AllowA4, workspace, plain);
    ninfer::ops::silu_mul(projected.slice(0, 0, kIntermediate), projected.slice(0, kIntermediate, kIntermediate),
                          split, plain);
    cuda(cudaStreamSynchronize(plain), "split route");
    h_amax = amax_bf16(host_bf16(split, static_cast<std::size_t>(kIntermediate) * kMaxTokens));
  }
  const ninfer::Weight down = nvfp4_weight(kInput, kIntermediate, 653U, 448.0f * 6.0f / h_amax);

  // The residual the down product adds into, ~ N(0, 1), restored before every run.
  const std::size_t residual_bytes = static_cast<std::size_t>(kInput) * kMaxTokens * 2;
  std::vector<std::uint16_t> host_seed(static_cast<std::size_t>(kInput) * kMaxTokens);
  for (auto &v : host_seed) {
    const __nv_bfloat16 b = __float2bfloat16(next_normal());
    v = *reinterpret_cast<const std::uint16_t *>(&b);
  }
  void *seed_data = device_alloc(residual_bytes);
  cuda(cudaMemcpy(seed_data, host_seed.data(), residual_bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  void *reference_data = device_alloc(residual_bytes);
  void *residual_data = device_alloc(residual_bytes);
  const auto reseed = [&](void *residual) {
    cuda(cudaMemcpy(residual, seed_data, residual_bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy");
  };

  // The down product's NVFP4 input: the split route's quantizer's and the epilogue's.
  const std::size_t code_bytes = kCodeBytesPerToken * kMaxTokens;
  const std::size_t scale_bytes = kScaleBytesPerToken * kMaxTokens;
  const nd::Nvfp4W4a4Workspace expected_h{static_cast<std::uint8_t *>(device_alloc(code_bytes)),
                                          static_cast<std::uint8_t *>(device_alloc(scale_bytes))};
  const nd::Nvfp4W4a4Workspace got_h{static_cast<std::uint8_t *>(device_alloc(code_bytes)),
                                     static_cast<std::uint8_t *>(device_alloc(scale_bytes))};

  // Two blocks a width: the op's and the down product's.
  ignis::tma_descriptors::Cache cache(2 * std::size(kWidths), served);
  for (const std::int32_t tokens : kWidths) {
    const std::string at = " at " + std::to_string(tokens) + " tokens";
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * tokens;
    const std::size_t residual_count = static_cast<std::size_t>(kInput) * tokens;
    const Tensor x(x_data, DType::BF16, {kInput, tokens});
    Tensor projected(projected_data, DType::BF16, {kGateUp, tokens});
    Tensor split(split_data, DType::BF16, {kIntermediate, tokens});
    Tensor fused(fused_data, DType::BF16, {kIntermediate, tokens});
    Tensor reference(reference_data, DType::BF16, {kInput, tokens});
    Tensor residual(residual_data, DType::BF16, {kInput, tokens});

    // The split route on a plain stream: the product, SwiGLU over its BF16 gate and up halves, then
    // the down product into the residual.
    cuda(cudaMemset(split_data, 0, out_bytes), "cudaMemset");
    workspace.reset();
    ninfer::ops::linear(x, gate_up, projected, LinearPolicy::AllowA4, workspace, plain);
    ninfer::ops::silu_mul(projected.slice(0, 0, kIntermediate), projected.slice(0, kIntermediate, kIntermediate),
                          split, plain);
    reseed(reference_data);
    workspace.reset();
    ninfer::ops::linear_add(split, down, reference, LinearPolicy::AllowA4, workspace, plain);
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
    const std::vector<std::uint16_t> expected_residual = host_bf16(reference, residual_count);
    std::size_t residual_finite = 0;
    std::size_t moved = 0;
    for (std::size_t i = 0; i < residual_count; ++i) {
      residual_finite += (expected_residual[i] & 0x7f80u) != 0x7f80u;
      moved += expected_residual[i] != host_seed[i];
    }
    check(residual_finite == residual_count && moved > residual_count / 2,
          "the split route's residual is finite and mostly moved by the down product" + at);

    struct Leg {
      cudaStream_t stream;
      const char *name;
      nd::Nvfp4W4a4ActivationLayout layout;
    };
    for (const Leg &leg : {Leg{plain, "a plain stream", nd::Nvfp4W4a4ActivationLayout::KTileMajor},
                           Leg{served, "a served stream", nd::Nvfp4W4a4ActivationLayout::KTileMajor},
                           Leg{token, "a token-major stream", nd::Nvfp4W4a4ActivationLayout::TokenMajor}}) {
      const std::string on = std::string(" on ") + leg.name + at;

      // The split route's down product input on this stream, in the layout its quantizer writes.
      cuda(cudaMemset(expected_h.codes, 0, code_bytes), "cudaMemset");
      cuda(cudaMemset(expected_h.scales, 0, scale_bytes), "cudaMemset");
      const nd::Nvfp4W4a4ActivationLayout wrote =
          nd::launch_nvfp4_w4a4_quantize_for_products(split, down, expected_h, leg.stream);
      cuda(cudaStreamSynchronize(leg.stream), "down quantizer");
      const std::vector<std::uint8_t> expected_codes =
          host_copy<std::uint8_t>(expected_h.codes, kCodeBytesPerToken * tokens);
      const std::vector<std::uint8_t> expected_scales =
          host_copy<std::uint8_t>(expected_h.scales, kScaleBytesPerToken * tokens);
      std::size_t live = 0;
      std::size_t saturated = 0;
      for (const std::uint8_t s : expected_scales) {
        live += s != 0;
        saturated += (s & 0x7fu) == 0x7eu;
      }
      check(wrote == leg.layout && live > expected_scales.size() / 2 && saturated < expected_scales.size() / 100,
            std::string("the down product's quantizer writes ") +
                (leg.layout == nd::Nvfp4W4a4ActivationLayout::KTileMajor ? "K-tile-major" : "token-major") +
                ", its scales mostly live and unsaturated" + on + " (" + std::to_string(saturated) + " saturated)");

      // The BF16 epilogue.
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ps::gate_up_swiglu(x, gate_up, fused, workspace, leg.stream);
      cuda(cudaStreamSynchronize(leg.stream), "fused op");
      std::size_t n = differing(host_bf16(fused, count), expected);
      check(n == 0, "the BF16 epilogue writes the split route's bytes" + on + " (" + std::to_string(n) + " differ)");

      // The NVFP4 epilogue: the down product's input, in the layout that product reads.
      cuda(cudaMemset(got_h.codes, 0xa5, code_bytes), "cudaMemset");
      cuda(cudaMemset(got_h.scales, 0xa5, scale_bytes), "cudaMemset");
      workspace.reset();
      const nd::Nvfp4W4a4ActivationLayout epilogue =
          ps::gate_up_swiglu_nvfp4(x, gate_up, down, got_h, workspace, leg.stream);
      cuda(cudaStreamSynchronize(leg.stream), "fused op, NVFP4 epilogue");
      const std::size_t codes_differ =
          differing(host_copy<std::uint8_t>(got_h.codes, kCodeBytesPerToken * tokens), expected_codes);
      const std::size_t scales_differ =
          differing(host_copy<std::uint8_t>(got_h.scales, kScaleBytesPerToken * tokens), expected_scales);
      check(epilogue == wrote && codes_differ == 0 && scales_differ == 0,
            "the NVFP4 epilogue writes the down quantizer's codes and scales in its layout" + on + " (" +
                std::to_string(codes_differ) + " code and " + std::to_string(scales_differ) + " scale bytes differ)");

      // The layers' MLP under Quant: the residual the split route's down product leaves, and the BF16
      // matrix never written.
      reseed(residual_data);
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ps::mlp(ps::Route::Quant, x, gate_up, LinearPolicy::AllowA4, down, LinearPolicy::AllowA4, fused, residual,
              workspace, leg.stream);
      cuda(cudaStreamSynchronize(leg.stream), "MLP under Quant");
      n = differing(host_bf16(residual, residual_count), expected_residual);
      const std::vector<std::uint16_t> untouched = host_bf16(fused, count);
      const bool unwritten =
          std::all_of(untouched.begin(), untouched.end(), [](std::uint16_t v) { return v == 0xffffu; });
      check(n == 0 && unwritten, "under Quant the MLP leaves the split route's residual, the BF16 matrix unwritten" +
                                     on + " (" + std::to_string(n) + " differ)");
    }

    if (tokens == 1024) {
      // The positive control: the vendored dispatch at 1,024 is its fused kernel.
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ninfer::ops::linear_swiglu(x, gate_up, fused, LinearPolicy::AllowA4, workspace, plain);
      cuda(cudaStreamSynchronize(plain), "vendored fused kernel");
      const std::vector<std::uint16_t> vendored = host_bf16(fused, count);
      const std::size_t n = differing(vendored, expected);
      check(n > 0, "the vendored fused kernel at 1,024 tokens writes other bytes (" + std::to_string(n) + " of " +
                       std::to_string(count) + " differ)");

      // The layers' entry under each route: Quant and Fused run the op at 1,024, Split the vendored
      // dispatch.
      for (const ps::Route route : {ps::Route::Quant, ps::Route::Fused, ps::Route::Split}) {
        cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
        workspace.reset();
        ps::linear_swiglu(route, x, gate_up, fused, LinearPolicy::AllowA4, workspace, plain);
        cuda(cudaStreamSynchronize(plain), "routed entry");
        const std::vector<std::uint16_t> got = host_bf16(fused, count);
        if (route == ps::Route::Split) {
          check(differing(got, vendored) == 0, "under Split the layers' entry runs the vendored dispatch at 1,024 tokens");
        } else {
          check(differing(got, expected) == 0, std::string("under ") + (route == ps::Route::Quant ? "Quant" : "Fused") +
                                                   " the layers' entry runs the op at 1,024 tokens");
        }
      }

      // Under Quant, the layers' MLP whose down product is off the W4A4 route (A16 here, as an
      // exception tensor held in FP8 is elsewhere) runs the BF16 epilogue, then that down product.
      reseed(reference_data);
      workspace.reset();
      ninfer::ops::linear_add(split, down, reference, LinearPolicy::A16Only, workspace, plain);
      reseed(residual_data);
      cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
      workspace.reset();
      ps::mlp(ps::Route::Quant, x, gate_up, LinearPolicy::AllowA4, down, LinearPolicy::A16Only, fused, residual,
              workspace, plain);
      cuda(cudaStreamSynchronize(plain), "MLP under Quant, down off the route");
      const std::size_t bf16_differ = differing(host_bf16(fused, count), expected);
      const std::size_t residual_differ =
          differing(host_bf16(residual, residual_count), host_bf16(reference, residual_count));
      check(bf16_differ == 0 && residual_differ == 0,
            "under Quant with the down product off the W4A4 route the MLP runs the BF16 epilogue at 1,024 tokens (" +
                std::to_string(bf16_differ) + " BF16 and " + std::to_string(residual_differ) +
                " residual values differ)");
    }
  }
  {
    // A width the op does not take keeps the vendored dispatch under Fused.
    constexpr std::int32_t kRagged = 1152;
    const std::size_t count = static_cast<std::size_t>(kIntermediate) * kRagged;
    const Tensor x(x_data, DType::BF16, {kInput, kRagged});
    Tensor split(split_data, DType::BF16, {kIntermediate, kRagged});
    Tensor fused(fused_data, DType::BF16, {kIntermediate, kRagged});
    cuda(cudaMemset(fused_data, 0xff, out_bytes), "cudaMemset");
    workspace.reset();
    ninfer::ops::linear_swiglu(x, gate_up, split, LinearPolicy::AllowA4, workspace, plain);
    workspace.reset();
    ps::linear_swiglu(ps::Route::Fused, x, gate_up, fused, LinearPolicy::AllowA4, workspace, plain);
    cuda(cudaStreamSynchronize(plain), "routed entry");
    check(differing(host_bf16(fused, count), host_bf16(split, count)) == 0,
          "under Fused the layers' entry keeps the vendored dispatch at 1,152 tokens");
  }
  // Each width's two blocks, once: the MLP at the same widths again reads their slots.
  for (const std::int32_t tokens : kWidths) {
    const Tensor x(x_data, DType::BF16, {kInput, tokens});
    Tensor fused(fused_data, DType::BF16, {kIntermediate, tokens});
    Tensor residual(residual_data, DType::BF16, {kInput, tokens});
    reseed(residual_data);
    workspace.reset();
    ps::mlp(ps::Route::Quant, x, gate_up, LinearPolicy::AllowA4, down, LinearPolicy::AllowA4, fused, residual,
            workspace, served);
  }
  cuda(cudaStreamSynchronize(served), "MLP under Quant");
  const std::uint64_t blocks = ignis::tma_descriptors::kThroughPointer ? 2 * std::size(kWidths) : 0;
  check(cache.counts().uploads == blocks && cache.counts().resident == blocks,
        "on a served stream the MLP's two blocks a width are uploaded once, over the four widths (" +
            std::to_string(cache.counts().uploads) + " uploads)");

  cuda(cudaStreamDestroy(plain), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(served), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(token), "cudaStreamDestroy");
  if (g_failed != 0) {
    std::printf("%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
