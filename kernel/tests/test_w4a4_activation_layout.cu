// GitHub #652 (spec #646): the layout the 27B's W4A4 prefill products read their activations in
// (kernel/src/w4a4_activation_layout.h) -- OURS.
//
//   - The measurement switch IGNIS_W4A4_ACTIVATION_LAYOUT, as a load reads it: unset, empty or
//     `ktile` names the K-tile-major layout, `token` the token-major one, and any other value is
//     refused, naming the variable.
//   - The quantizer the products take (`launch_nvfp4_w4a4_quantize_for_products`), at the 27B's
//     three K (5,120, 6,144 and 17,408) and 1,024, 1,280, 2,048 and 4,096 tokens: on a stream no
//     load names, it writes the stock quantizer's codes and scales K-tile-major, codes
//     [K/128][T][64 bytes] and scales [K/256][T][16 bytes], and says so; on a stream named
//     token-major it writes the stock bytes and says so; and below the TMA route's widths (768
//     tokens, and 1,100, no whole number of 256-token tiles) it writes the stock bytes whatever
//     the stream names, as the products there read them.
//   - The 27B's five prefill products, through the ops its prefill runs them by (attn_input_proj
//     [14336, 5120], gdn_input_proj [16384, 5120], linear [34816, 5120], the gate and up as
//     linear_swiglu's split route runs it, and linear_add [5120, 6144] and [5120, 17408]), at the
//     four widths: on a stream named K-tile-major and served by a descriptor cache
//     (kernel/src/tma_descriptor_cache.h), as a Windows load's prefill is, every output is the one
//     written on a stream named token-major, byte for byte, while the workspace the op quantized
//     its activations into holds other bytes.
// The inputs vary per token and per 16-wide group of K, so a code or scale read from the wrong
// place changes an output.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing GPU fails this test.

#include "tma_descriptor_cache.h"
#include "w4a4_activation_layout.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
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

namespace {

namespace nd = ninfer::ops::detail;
namespace layout = ignis::w4a4_activation_layout;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::LinearPolicy;

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

// Sets (or, with null, removes) a variable in the operating system's environment, as a host
// process's own runtime does (Rust's std::env::set_var).
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

void check_switch() {
  constexpr const char *kSwitch = "IGNIS_W4A4_ACTIVATION_LAYOUT";
  set_environment(kSwitch, nullptr);
  check(layout::layout_for_load() == layout::Layout::KTile, "unset: K-tile-major");
  set_environment(kSwitch, "");
  check(layout::layout_for_load() == layout::Layout::KTile, "empty: K-tile-major");
  set_environment(kSwitch, "ktile");
  check(layout::layout_for_load() == layout::Layout::KTile, "`ktile`: K-tile-major");
  set_environment(kSwitch, "token");
  check(layout::layout_for_load() == layout::Layout::Token, "`token`: token-major");
  set_environment(kSwitch, "line");
  std::string refusal;
  try {
    (void)layout::layout_for_load();
  } catch (const std::invalid_argument &e) {
    refusal = e.what();
  }
  check(refusal.find(kSwitch) != std::string::npos && refusal.find("line") != std::string::npos,
        "`line` refused, naming the variable and the value: " + refusal);
  set_environment(kSwitch, nullptr);
}

std::uint64_t g_state = 0x2545F4914F6CDD1Dull;
std::uint32_t next_word() {
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  return static_cast<std::uint32_t>(g_state >> 16);
}

template <class T>
T *device_copy_of(const std::vector<T> &host) {
  void *p = nullptr;
  cuda(cudaMalloc(&p, host.size() * sizeof(T)), "cudaMalloc");
  cuda(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
  return static_cast<T *>(p);
}

std::vector<std::uint8_t> host_bytes(const void *device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  cuda(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
  return host;
}

// A BF16 activation [k, tokens], each 16-wide group of K at its own power of two per token, so
// every group of every token gets its own scale.
Tensor activation(std::int32_t k, std::int32_t tokens) {
  std::vector<__nv_bfloat16> host(static_cast<std::size_t>(k) * tokens);
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t i = 0; i < k; ++i) {
      const int exponent = static_cast<int>((static_cast<unsigned>(i / 16) * 7U + static_cast<unsigned>(t) * 3U) % 9U) - 4;
      const float unit = static_cast<float>(next_word() % 2001U) / 1000.0f - 1.0f;
      host[static_cast<std::size_t>(t) * k + i] = __float2bfloat16(std::ldexp(unit, exponent));
    }
  }
  return Tensor(device_copy_of(host), DType::BF16, {k, tokens});
}

// The NVFP4 weight [n, k] the products read (blockscale-k16-m128x4-v1): random E2M1 codes, E4M3
// scales from 0.5 to just under 1, in one payload as nvfp4_format.cpp requires.
Weight nvfp4_weight(std::int32_t n, std::int32_t k) {
  const std::size_t code_bytes = static_cast<std::size_t>(n) * k / 2;
  const std::size_t scale_offset = (code_bytes + 255) / 256 * 256;
  const std::size_t scale_bytes = static_cast<std::size_t>(n) * k / 16;
  std::vector<std::uint8_t> payload(scale_offset + scale_bytes + sizeof(float), 0);
  for (std::size_t i = 0; i < code_bytes; ++i) payload[i] = static_cast<std::uint8_t>(next_word());
  for (std::size_t i = 0; i < scale_bytes; ++i) {
    payload[scale_offset + i] = static_cast<std::uint8_t>(0x30 | (next_word() & 0x0f));
  }
  const float one = 1.0f;
  std::memcpy(payload.data() + scale_offset + scale_bytes, &one, sizeof(one));
  auto *device = device_copy_of(payload);
  Weight w;
  w.payload = device;
  w.payload_bytes = payload.size();
  w.qtype = QType::NVFP4;
  w.group_size = 16;
  w.shape[0] = n;
  w.shape[1] = k;
  w.padded_shape[0] = n;
  w.padded_shape[1] = k;
  w.ndim = 2;
  w.qdata = device;
  w.scales = device + scale_offset;
  w.n = n;
  w.k = k;
  w.group = 16;
  w.layout = ninfer::QuantLayout::BlockScaleK16M128x4;
  w.scale_dtype = DType::FP8_E4M3FN;
  w.weight_scale_divisor = 16.0f;
  w.input_scale_divisor = 64.0f;
  return w;
}

Tensor bf16_matrix(std::int32_t rows, std::int32_t tokens) {
  void *p = nullptr;
  cuda(cudaMalloc(&p, static_cast<std::size_t>(rows) * tokens * sizeof(__nv_bfloat16)), "cudaMalloc");
  return Tensor(p, DType::BF16, {rows, tokens});
}

// The quantizer's bytes on `stream`, and the layout it says it wrote.
struct Quantized {
  nd::Nvfp4W4a4ActivationLayout layout;
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> scales;
};

Quantized quantize(const Tensor &x, const Weight &w, bool for_products, cudaStream_t stream) {
  const std::int32_t k = x.ne[0];
  const std::int32_t tokens = x.ne[1];
  const std::size_t code_bytes = static_cast<std::size_t>(k) * tokens / 2;
  const std::size_t scale_bytes = static_cast<std::size_t>(k) * tokens / 16;
  void *codes = nullptr;
  void *scales = nullptr;
  cuda(cudaMalloc(&codes, code_bytes), "cudaMalloc");
  cuda(cudaMalloc(&scales, scale_bytes), "cudaMalloc");
  cuda(cudaMemset(codes, 0, code_bytes), "cudaMemset");
  cuda(cudaMemset(scales, 0, scale_bytes), "cudaMemset");
  const nd::Nvfp4W4a4Workspace workspace{static_cast<std::uint8_t *>(codes), static_cast<std::uint8_t *>(scales)};
  Quantized out{nd::Nvfp4W4a4ActivationLayout::TokenMajor, {}, {}};
  if (for_products) {
    out.layout = nd::launch_nvfp4_w4a4_quantize_for_products(x, w, workspace, stream);
  } else {
    nd::launch_nvfp4_w4a4_quantize(x, w, workspace, stream);
  }
  cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  out.codes = host_bytes(codes, code_bytes);
  out.scales = host_bytes(scales, scale_bytes);
  cuda(cudaFree(codes), "cudaFree");
  cuda(cudaFree(scales), "cudaFree");
  return out;
}

// The stock (token-major) quantizer's bytes moved to the K-tile-major places: a 16-wide group g of
// token t, 8 bytes of codes at [t][g * 8] and a scale at [t][g], goes to K tile g / 8 and to the
// pair of K tiles g / 16.
Quantized k_tile_major(const Quantized &token_major, std::int32_t k, std::int32_t tokens) {
  Quantized out{nd::Nvfp4W4a4ActivationLayout::KTileMajor,
                std::vector<std::uint8_t>(token_major.codes.size()),
                std::vector<std::uint8_t>(token_major.scales.size())};
  const std::int32_t groups = k / 16;
  for (std::int32_t t = 0; t < tokens; ++t) {
    for (std::int32_t g = 0; g < groups; ++g) {
      const std::size_t from = static_cast<std::size_t>(t) * groups + g;
      const std::size_t code_to = (static_cast<std::size_t>(g / 8) * tokens + t) * 64 + (g % 8) * 8;
      std::memcpy(out.codes.data() + code_to, token_major.codes.data() + from * 8, 8);
      out.scales[(static_cast<std::size_t>(g / 16) * tokens + t) * 16 + g % 16] = token_major.scales[from];
    }
  }
  return out;
}

bool same(const Quantized &a, const Quantized &b) {
  return a.layout == b.layout && a.codes == b.codes && a.scales == b.scales;
}

void check_quantizer(cudaStream_t unnamed, cudaStream_t token) {
  for (std::int32_t k : {5120, 6144, 17408}) {
    const Weight w = nvfp4_weight(128, k);
    for (std::int32_t tokens : {1024, 1280, 2048, 4096}) {
      const Tensor x = activation(k, tokens);
      const Quantized stock = quantize(x, w, false, unnamed);
      const std::string at = "K " + std::to_string(k) + ", " + std::to_string(tokens) + " tokens";
      check(same(quantize(x, w, true, unnamed), k_tile_major(stock, k, tokens)),
            at + ", a stream no load names: the stock bytes K-tile-major");
      check(same(quantize(x, w, true, token), stock), at + ", a stream named token-major: the stock bytes");
      cuda(cudaFree(x.data), "cudaFree");
    }
    for (std::int32_t tokens : {768, 1100}) {
      const Tensor x = activation(k, tokens);
      check(same(quantize(x, w, true, unnamed), quantize(x, w, false, unnamed)),
            "K " + std::to_string(k) + ", " + std::to_string(tokens) +
                " tokens, off the TMA route: the stock bytes");
      cuda(cudaFree(x.data), "cudaFree");
    }
    cuda(cudaFree(const_cast<void *>(w.payload)), "cudaFree");
  }
}

// One of the five products, through the op the 27B's prefill runs it by.
enum class Op { AttentionInput, GdnInput, GateUp, OutputProjection, Down };

struct Product {
  Op op;
  const char *name;
  std::int32_t n;
  std::int32_t k;
};

constexpr Product kProducts[] = {
    {Op::AttentionInput, "attn_input_proj [14336, 5120]", 14336, 5120},
    {Op::GdnInput, "gdn_input_proj [16384, 5120]", 16384, 5120},
    {Op::GateUp, "linear [34816, 5120]", 34816, 5120},
    {Op::OutputProjection, "linear_add [5120, 6144]", 5120, 6144},
    {Op::Down, "linear_add [5120, 17408]", 5120, 17408},
};

std::size_t workspace_bytes(const Product &p, std::int32_t tokens) {
  switch (p.op) {
  case Op::AttentionInput:
    return ninfer::ops::attn_input_proj_workspace_capacity_bytes(QType::NVFP4, p.n, p.k, LinearPolicy::AllowA4,
                                                                 tokens, tokens);
  case Op::GdnInput:
    return ninfer::ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, p.n, p.k, LinearPolicy::AllowA4,
                                                                tokens, tokens);
  case Op::GateUp:
    return ninfer::ops::linear_workspace_capacity_bytes(QType::NVFP4, p.n, p.k, LinearPolicy::AllowA4, tokens,
                                                        tokens);
  case Op::OutputProjection:
  case Op::Down:
    return ninfer::ops::linear_add_workspace_capacity_bytes(QType::NVFP4, p.n, p.k, LinearPolicy::AllowA4,
                                                            tokens, tokens);
  }
  return 0;
}

// What one run of a product wrote: its outputs, and the workspace it quantized its activations into.
struct Run {
  std::vector<std::vector<std::uint8_t>> outputs;
  std::vector<std::uint8_t> workspace;
};

Run run(const Product &p, const Weight &w, const Tensor &x, const std::vector<Tensor> &outputs,
        const std::vector<std::uint8_t> &residual, std::int32_t tokens, cudaStream_t stream) {
  for (const Tensor &out : outputs) cuda(cudaMemset(out.data, 0, out.bytes()), "cudaMemset");
  if (!residual.empty()) {
    cuda(cudaMemcpy(outputs[0].data, residual.data(), residual.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
  }
  ninfer::DeviceBuffer storage(workspace_bytes(p, tokens));
  cuda(cudaMemset(storage.p, 0, storage.bytes), "cudaMemset");
  ninfer::WorkspaceArena arena(ninfer::DeviceSpan{storage.p, storage.bytes});
  std::vector<Tensor> o = outputs;
  switch (p.op) {
  case Op::AttentionInput:
    ninfer::ops::attn_input_proj(x, w, o[0], o[1], o[2], o[3], LinearPolicy::AllowA4, arena, stream);
    break;
  case Op::GdnInput:
    ninfer::ops::gdn_input_proj(x, w, o[0], o[1], LinearPolicy::AllowA4, arena, stream);
    break;
  case Op::GateUp:
    ninfer::ops::linear(x, w, o[0], LinearPolicy::AllowA4, arena, stream);
    break;
  case Op::OutputProjection:
  case Op::Down:
    ninfer::ops::linear_add(x, w, o[0], LinearPolicy::AllowA4, arena, stream);
    break;
  }
  cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  Run r;
  for (const Tensor &out : outputs) r.outputs.push_back(host_bytes(out.data, out.bytes()));
  r.workspace = host_bytes(storage.p, storage.bytes);
  return r;
}

// The outputs of a product, as the op takes them: q, gate, k, v; qkv, z; the gate and up; the
// residual.
std::vector<Tensor> outputs_of(const Product &p, std::int32_t tokens) {
  switch (p.op) {
  case Op::AttentionInput:
    return {bf16_matrix(6144, tokens), bf16_matrix(6144, tokens), bf16_matrix(1024, tokens),
            bf16_matrix(1024, tokens)};
  case Op::GdnInput:
    return {bf16_matrix(10240, tokens), bf16_matrix(6144, tokens)};
  case Op::GateUp:
  case Op::OutputProjection:
  case Op::Down:
    return {bf16_matrix(p.n, tokens)};
  }
  return {};
}

bool any_nonzero(const std::vector<std::uint8_t> &bytes) {
  for (std::uint8_t b : bytes) {
    if (b != 0) return true;
  }
  return false;
}

void check_products(cudaStream_t token, cudaStream_t k_tile) {
  for (const Product &p : kProducts) {
    const Weight w = nvfp4_weight(p.n, p.k);
    for (std::int32_t tokens : {1024, 1280, 2048, 4096}) {
      const std::string at = std::string(p.name) + ", " + std::to_string(tokens) + " tokens";
      const Tensor x = activation(p.k, tokens);
      const std::vector<Tensor> outputs = outputs_of(p, tokens);
      std::vector<std::uint8_t> residual;
      if (p.op == Op::OutputProjection || p.op == Op::Down) {
        const Tensor r = activation(p.n, tokens);
        residual = host_bytes(r.data, r.bytes());
        cuda(cudaFree(r.data), "cudaFree");
      }
      const Run token_major = run(p, w, x, outputs, residual, tokens, token);
      const Run tiled = run(p, w, x, outputs, residual, tokens, k_tile);
      bool written = true;
      for (const auto &out : token_major.outputs) written = written && any_nonzero(out);
      check(written, at + ": token-major, every output written");
      check(tiled.outputs == token_major.outputs, at + ": K-tile-major, the token-major outputs byte for byte");
      check(tiled.workspace != token_major.workspace,
            at + ": the activations it read, K-tile-major, are not the token-major bytes");
      for (const Tensor &out : outputs) cuda(cudaFree(out.data), "cudaFree");
      cuda(cudaFree(x.data), "cudaFree");
    }
    cuda(cudaFree(const_cast<void *>(w.payload)), "cudaFree");
  }
}

}  // namespace

int main() {
  cuda(cudaSetDevice(0), "cudaSetDevice");
  check_switch();

  cudaStream_t unnamed = nullptr;
  cudaStream_t token = nullptr;
  cudaStream_t served = nullptr;
  cuda(cudaStreamCreate(&unnamed), "cudaStreamCreate");
  cuda(cudaStreamCreate(&token), "cudaStreamCreate");
  cuda(cudaStreamCreate(&served), "cudaStreamCreate");
  {
    const layout::ForStream token_major(token, layout::Layout::Token);
    const layout::ForStream k_tile_major(served, layout::Layout::KTile);
    // As a Windows load's prefill: its products' descriptors in slots (Linux never asks them).
    const ignis::tma_descriptors::Cache slots(16, served);
    check_quantizer(unnamed, token);
    check_products(token, served);
  }
  cuda(cudaStreamDestroy(unnamed), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(token), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(served), "cudaStreamDestroy");
  if (g_failed != 0) {
    std::printf("%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
