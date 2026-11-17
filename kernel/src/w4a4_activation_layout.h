// ignis kernel leaf: the layout the 27B's W4A4 prefill products read their activations in -- OURS
// (GitHub #652, spec #646), leaf-internal.
//
// On the RTX 5090 the NVFP4 W4A4 TMA products (the vendored `nvfp4_w4a4_tma.cu`, every product of
// a prefill chunk of a multiple of 256 tokens from 1,024) are bound by the L2's tag lookups, not by
// the tensor cores: stored token by token, a 128-wide slice of K reaches a 256-token tile as 256
// pieces of 64 bytes, half an L2 line each, and its scales as 256 pieces of 16 bytes. K-tile-major,
// the same slice is 16 KiB in one piece and its scales 4 KiB:
//
//   activation codes:  [K / 128][tokens][64 bytes]
//   activation scales: [K / 256][tokens][16 bytes]
//
// The weights keep their layout: the decode and verify routes read them too. The vendored
// launchers (`launch_nvfp4_w4a4_quantize_for_products`, ops/linear/nvfp4/nvfp4_w4a4_plan.h, under a
// recorded patch, ADR 0031) ask `k_tile_major` once per call, quantize in the layout it answers
// and hand the same layout to the product, so a product always reads what its quantizer wrote;
// the bytes it writes are the same in either. ignis's fused gate and up op
// (kernel/src/prefill_swiglu.h) quantizes its own input with the stock quantizer and reads it
// token-major whatever the load names.
//
// A load decides the layout once (`layout_for_load`) and names it for the stream its prefill runs
// on (`ForStream`); a stream no load names reads K-tile-major, the default.
#pragma once

#include <cuda_runtime.h>

namespace ignis::w4a4_activation_layout {

enum class Layout { KTile, Token };

// The measurement switch IGNIS_W4A4_ACTIVATION_LAYOUT in the process environment
// (kernel/src/process_environment.h), read once per load: unset, empty or `ktile`, K-tile-major;
// `token`, the token-major layout the products read before GitHub #652. Throws
// std::invalid_argument, naming the variable, on any other value.
Layout layout_for_load();

// While it lives, the W4A4 TMA products launched on `stream` read their activations in `layout`,
// which their quantizer writes. Throws std::invalid_argument when another already names `stream`.
class ForStream {
public:
  ForStream(cudaStream_t stream, Layout layout);
  ~ForStream();
  ForStream(const ForStream &) = delete;
  ForStream &operator=(const ForStream &) = delete;

private:
  cudaStream_t stream_ = nullptr;
};

// What the vendored launchers ask: whether the W4A4 TMA products launched on `stream` read their
// activations K-tile-major -- true unless a live `ForStream` names `stream` token-major.
bool k_tile_major(cudaStream_t stream);

} // namespace ignis::w4a4_activation_layout
