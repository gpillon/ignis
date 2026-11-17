// ignis kernel leaf: the 27B prefill's MLP gate and up product with SwiGLU in its epilogue -- an
// ADAPTED op (ADR 0051; spec #646, GitHub #651), leaf-internal. The kernel and its upstream:
// kernel/src/prefill_swiglu.cuh.
//
// It takes the place of two passes at every width the vendored W4A4 TMA route takes, from 1,024
// tokens in whole 256-token tiles (`takes`): today's split route writes the [34816, T] BF16
// product and a SiLU-mul pass reads it back, and at exactly 1,024 tokens the vendored fused
// kernel, which rounds once fewer, ran instead, so a 1,024-token chunk and a 2,048-token one
// computed the MLP with other bits. The op rounds as the split route does and writes its bytes,
// at every width it takes (kernel/tests/test_prefill_swiglu.cu); the vendored fused kernel leaves
// the 27B's path.
//
// Chosen once per load (model.cu `prefill_swiglu_route`) by the measurement switch
// IGNIS_PREFILL_SWIGLU: unset, or `fused`, the op; `split` today's route, the vendored dispatch
// unchanged (the vendored fused kernel at exactly 1,024 tokens). Every call it does not take keeps
// the vendored dispatch: decode and verify rounds, a narrower or ragged chunk, the A16 policy, a
// weight that is not the 27B's NVFP4 gate and up.
//
// It quantizes its own input with the vendored W4A4 quantizer and builds its own TMA descriptors,
// in the activations' token-major layout. On Windows the descriptor block goes through the load's
// device slots (kernel/src/tma_descriptor_cache.h), one block per call as the split route's
// product had, so the VRAM plan's `tma_descriptors` line counts it unchanged.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::prefill_swiglu {

// What IGNIS_PREFILL_SWIGLU names at a load.
enum class Route {
  // The vendored dispatch: the product, then SiLU-mul (the vendored fused kernel at 1,024).
  Split,
  // The op at every width `takes` admits.
  Fused,
};

// The switch in the process environment (kernel/src/process_environment.h), read once per load:
// unset, empty or `fused`, Fused; `split`, Split. Throws std::invalid_argument, naming the
// variable, on any other value (`quant`, the quantizing epilogue, is not in this build).
Route route_for_load();

// The vendored W4A4 TMA route's widths: at least 1,024 tokens, a whole number of 256-token tiles.
inline constexpr std::int32_t kMinTokens = 1024;
inline constexpr std::int32_t kTileTokens = 256;
constexpr bool takes(std::int32_t tokens) {
  return tokens >= kMinTokens && (tokens % kTileTokens) == 0;
}

// The op: x [5120, T] BF16 through the 27B's NVFP4 gate and up [34816, 5120] into out [17408, T]
// BF16, out = BF16(silu(BF16(gate)) * BF16(up)), the split route's bytes. Quantizes x into a scope
// of `workspace` (the W4A4 activation scratch, which the vendored workspace query already holds at
// every width the op takes), then one kernel. Throws std::invalid_argument on a call it does not
// take. Enqueues on `stream` and does not synchronize.
void gate_up_swiglu(const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                    ninfer::WorkspaceArena &workspace, cudaStream_t stream);

// `ninfer::ops::linear_swiglu` under the load's route (kernel/src/linear_swiglu.cu): the same
// checks, then the op where `route` is Fused, the policy AllowA4, the weight NVFP4 and the width
// one `takes` admits; the vendored dispatch otherwise.
void linear_swiglu(Route route, const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                   ninfer::ops::LinearPolicy policy, ninfer::WorkspaceArena &workspace, cudaStream_t stream);

} // namespace ignis::prefill_swiglu
