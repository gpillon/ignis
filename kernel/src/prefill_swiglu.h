// ignis kernel leaf: the 27B prefill's MLP gate and up product with SwiGLU in its epilogue -- an
// ADAPTED op (ADR 0051; spec #646, GitHub #651, #653), leaf-internal. The kernel and its upstream:
// kernel/src/prefill_swiglu.cuh.
//
// It takes the place of the split route's passes at every width the vendored W4A4 TMA route
// takes, from 1,024 tokens in whole 256-token tiles (`takes`): today's split route writes the
// [34816, T] BF16 product, a SiLU-mul pass reads it back and writes the [17408, T] BF16 matrix,
// and the down product's quantizer reads that; at exactly 1,024 tokens the vendored fused kernel,
// which rounds once fewer, ran instead, so a 1,024-token chunk and a 2,048-token one computed the
// MLP with other bits. The op rounds as the split route does and writes its bytes, at every width
// it takes (kernel/tests/test_prefill_swiglu.cu); the vendored fused kernel leaves the 27B's path.
// Its epilogue has two forms: BF16 writes the SwiGLU output, the [17408, T] matrix; NVFP4
// quantizes each token's values straight into the down product's NVFP4 input under the down
// weight's input divisor, the codes and scales that product's own quantizer would write, so the
// matrix is never written.
//
// Chosen once per load (model.cu `prefill_swiglu_switch`) by the measurement switch
// IGNIS_PREFILL_SWIGLU: unset, or `quant`, the NVFP4 epilogue wherever the down product runs on
// the W4A4 TMA route, the BF16 one at every other call the op takes (a down weight held as an
// exception tensor in FP8, a down product under the test-only A16 override); `fused` the BF16
// epilogue at every call the op takes; `split` today's route, the vendored dispatch unchanged (the
// vendored fused kernel at exactly 1,024 tokens). Every call the op does not take keeps the
// vendored dispatch: decode and verify rounds, a narrower or ragged chunk, the A16 policy, a weight
// that is not the 27B's NVFP4 gate and up.
//
// It quantizes its own input through the vendored W4A4 products' one quantize call
// (`launch_nvfp4_w4a4_quantize_for_products`, ops/linear/nvfp4/nvfp4_w4a4_plan.h) and reads it in
// the layout that call wrote, the one the load names for the stream (kernel/src/
// w4a4_activation_layout.h): K-tile-major unless IGNIS_W4A4_ACTIVATION_LAYOUT=token names it
// token-major. Its NVFP4 epilogue writes the same layout and the down product reads it, so every
// product on the route reads what was written for it, under either switch. It builds its own TMA
// descriptors; on Windows the descriptor block goes through the load's device slots
// (kernel/src/tma_descriptor_cache.h), one block per call as the split route's gate and up
// product had, so the VRAM plan's `tma_descriptors` line counts it unchanged.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::prefill_swiglu {

// What IGNIS_PREFILL_SWIGLU names at a load.
enum class Route {
  // The vendored dispatch: the product, then SiLU-mul (the vendored fused kernel at 1,024).
  Split,
  // The op's BF16 epilogue at every width `takes` admits.
  Fused,
  // The op's NVFP4 epilogue where the down product runs on the W4A4 TMA route, its BF16 one at
  // every other width `takes` admits.
  Quant,
};

// The switch in the process environment (kernel/src/process_environment.h), read once per load:
// unset, empty or `quant`, Quant; `fused`, Fused; `split`, Split. Throws std::invalid_argument,
// naming the variable, on any other value.
Route route_for_load();

// The vendored W4A4 TMA route's widths: at least 1,024 tokens, a whole number of 256-token tiles.
inline constexpr std::int32_t kMinTokens = 1024;
inline constexpr std::int32_t kTileTokens = 256;
constexpr bool takes(std::int32_t tokens) {
  return tokens >= kMinTokens && (tokens % kTileTokens) == 0;
}

// The op, BF16 epilogue: x [5120, T] BF16 through the 27B's NVFP4 gate and up [34816, 5120] into
// out [17408, T] BF16, out = BF16(silu(BF16(gate)) * BF16(up)), the split route's bytes.
// Quantizes x into a scope of `workspace` (the W4A4 activation scratch, which the vendored
// workspace query already holds at every width the op takes), then one kernel. Throws
// std::invalid_argument on a call it does not take. Enqueues on `stream` and does not synchronize.
void gate_up_swiglu(const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                    ninfer::WorkspaceArena &workspace, cudaStream_t stream);

// The op, NVFP4 epilogue: the same values quantized under `down`'s input divisor into `h`, the
// down product's NVFP4 input (codes T x 8704 bytes, scales T x 1088), in the layout it returns,
// the one its own input was quantized in on `stream`: byte for byte what the down product's
// quantizer writes from the split route's BF16 output. `down` is the 27B's NVFP4 [5120, 17408].
// Throws std::invalid_argument on a call it does not take. Enqueues on `stream` and does not
// synchronize.
ninfer::ops::detail::Nvfp4W4a4ActivationLayout gate_up_swiglu_nvfp4(const ninfer::Tensor &x,
                                                                    const ninfer::Weight &gate_up,
                                                                    const ninfer::Weight &down,
                                                                    ninfer::ops::detail::Nvfp4W4a4Workspace h,
                                                                    ninfer::WorkspaceArena &workspace,
                                                                    cudaStream_t stream);

// `ninfer::ops::linear_swiglu` under the load's route (kernel/src/linear_swiglu.cu): the same
// checks, then the op's BF16 epilogue where `route` is Fused or Quant, the policy AllowA4, the
// weight NVFP4 and the width one `takes` admits; the vendored dispatch otherwise.
void linear_swiglu(Route route, const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                   ninfer::ops::LinearPolicy policy, ninfer::WorkspaceArena &workspace, cudaStream_t stream);

// The layers' MLP under the load's route: residual += down(swiglu(gate_up(x))). Under Quant, where
// both policies are AllowA4, the width one `takes` admits and both weights the 27B's NVFP4 ones
// (the down product then on the W4A4 TMA route), the NVFP4 epilogue into a scope of `workspace`,
// then the down product reading what it wrote; `fused` is not written. Everywhere else
// `linear_swiglu` into `fused`, then `ninfer::ops::linear_add` under `down_policy`. Enqueues on
// `stream` and does not synchronize.
void mlp(Route route, const ninfer::Tensor &x, const ninfer::Weight &gate_up,
         ninfer::ops::LinearPolicy gate_up_policy, const ninfer::Weight &down, ninfer::ops::LinearPolicy down_policy,
         ninfer::Tensor &fused, ninfer::Tensor &residual, ninfer::WorkspaceArena &workspace, cudaStream_t stream);

} // namespace ignis::prefill_swiglu
