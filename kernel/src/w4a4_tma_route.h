// ignis kernel leaf: the widths and the cards the vendored NVFP4 W4A4 TMA route runs at -- OURS
// (spec #646), leaf-internal.
//
// The vendored dispatch (ops/linear/nvfp4/nvfp4_w4a4.cu) sends a W4A4 product of at least 1,024
// tokens in whole 256-token tiles to its TMA route, whose sm_120a code runs on a card of compute
// capability 12. Every site of ours that plans for that route or stands in for it asks here: the
// widths the fused SwiGLU op takes (kernel/src/prefill_swiglu.h), the descriptor slots a load plans
// (kernel/src/tma_descriptor_cache.h), and the layout and the SwiGLU route a load reports
// (kernel/src/model.cu).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ignis::w4a4_tma {

inline constexpr std::int64_t kMinTokens = 1024;
inline constexpr std::int64_t kTileTokens = 256;

// Whether a product of `tokens` tokens runs on the route: at least 1,024, a whole number of
// 256-token tiles.
constexpr bool takes(std::int64_t tokens) {
  return tokens >= kMinTokens && (tokens % kTileTokens) == 0;
}

// Whether a prefill whose chunks are at most `widest_tokens` wide reaches the route at any width.
constexpr bool reaches(std::int64_t widest_tokens) {
  return widest_tokens >= kMinTokens;
}

// How many of the route's widths a prefill whose chunks are at most `widest_tokens` wide takes:
// one per 256-token tile from 1,024 tokens, 0 where it never reaches the route.
constexpr std::size_t widths_up_to(std::int64_t widest_tokens) {
  return reaches(widest_tokens) ? static_cast<std::size_t>((widest_tokens - kMinTokens) / kTileTokens + 1) : 0;
}

// Whether a card of compute capability `major` runs the route's sm_120a code.
constexpr bool runs_on(int major) {
  return major == 12;
}

} // namespace ignis::w4a4_tma
