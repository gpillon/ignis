// ignis kernel leaf: the 27B's BF16 exception matrices in FP8 (GitHub #341, spec prefill-5090/01
// lever 1) -- OURS. What the two dispatchers (attn_input_proj.cu, linear_add.cu) and the load share:
// the weight's layout, the routes and the activation workspace.
//
// The weight is QType::FP8_E4M3FN_ROW_BF16S / QuantLayout::RowScale, converted from the
// artifact's BF16 at load (ignis_fp8_quantize_rows):
//   - an output projection (attention/output in GQA layers 3 and 7, gdn/output in GDN layer 4),
//     [5120][6144]: one row-scale-v1 payload (ignis_fp8_linear.h);
//   - the fused attention/query_key_gate_value (GQA layers 3, 7, 11, 15, 19, 23), [14336][5120]:
//     four payloads one after another, each at a multiple of 256 bytes -- q 6144 | k 1024 |
//     gate 6144 | v 1024 rows, the artifact's own row order -- so every route takes a projection
//     as its own weight.
//
// The routes, by the call's width T and policy:
//   - T <= IGNIS_FP8_GEMV_MAX_TOKENS, or A16Only: BF16 activations (W8A16, ignis_fp8_linear's GEMV
//     or tensor-core route) -- a decode round, a one-lane verify round;
//   - T above it under AllowA8: x quantized per token to E4M3 in the call's workspace, then the
//     W8A8 GEMM (ignis_image_gemm.h) -- a prefill chunk, a verify round of two lanes or more, a
//     one-lane round at the extended shape; up to 16 tokens on the GEMM's 16-token tile (GitHub
//     #693, w8a8_tile16).
#ifndef IGNIS_FP8_EXCEPTION_H
#define IGNIS_FP8_EXCEPTION_H

#include "ignis_fp8_linear.h"
#include "ignis_image_gemm.h"
#include "process_environment.h"

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::fp8_exception {

inline constexpr std::int32_t kHidden = 5120;
inline constexpr std::int32_t kQkgvRows = 14336;
inline constexpr std::int32_t kOutputCols = 6144;
inline constexpr std::int32_t kQkgvSegments = 4;
// q, k, gate, v: the order the artifact stores them in, and the blocks the binder converts them in
// (QKGV_FP8_BLOCKS, crates/artifact/src/inventory.rs).
inline constexpr std::uint32_t kQkgvSegmentRows[kQkgvSegments] = {6144, 1024, 6144, 1024};
// The widest call on the BF16-activation route under AllowA8.
inline constexpr std::int32_t kGemvTokens = IGNIS_FP8_GEMV_MAX_TOKENS;

inline std::uint64_t round_up_256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

// The byte offset of each q | k | gate | v payload, and (at [4]) the whole weight's bytes.
struct QkgvLayout {
  std::uint64_t at[kQkgvSegments + 1] = {};
};
inline QkgvLayout qkgv_layout() {
  QkgvLayout l;
  for (int i = 0; i < kQkgvSegments; ++i) {
    l.at[i + 1] = l.at[i] + round_up_256(ignis_fp8_payload_bytes(kQkgvSegmentRows[i], kHidden));
  }
  return l;
}

// Whether a call of width `tokens` under `policy` quantizes its activations.
inline bool quantizes(ninfer::ops::LinearPolicy policy, std::int32_t tokens) {
  return policy == ninfer::ops::LinearPolicy::AllowA8 && tokens > kGemvTokens;
}

// GitHub #678 (spec #672): whether the q | k | gate | v projection's GEMV calls take the grouped
// GEMV's paired form (ignis_fp8_linear_grouped_paired) or its one-row form, as the measurement
// switch IGNIS_FP8_GEMV_PAIRED names it at each call, so a graph keeps the form its capture read:
// unset or `1` paired, `0` one-row. The output projection's residual add keeps the one-row form,
// which measured faster there. Throws std::invalid_argument, naming the variable, on any other
// value; the load reads it once, so such a value refuses the load.
inline bool gemv_paired() { return ignis::named_switch("IGNIS_FP8_GEMV_PAIRED", true, {{"0", false}, {"1", true}}); }

// GitHub #693 (spec #684): whether a call that quantizes its activations takes the W8A8 GEMM's
// 16-token tile at up to 16 tokens (ignis_image_gemm_w8a8_grouped_tile16 and _add_tile16) or its
// default 128-token tile at every width, as the measurement switch IGNIS_FP8_W8A8_TILE16 names it
// at each call, so a graph keeps the tile its capture read: unset or `1` the 16-token tile, `0` the
// default one. The outputs are the same bits either way. Throws std::invalid_argument, naming the
// variable, on any other value; the load reads it once, so such a value refuses the load.
inline bool w8a8_tile16() { return ignis::named_switch("IGNIS_FP8_W8A8_TILE16", true, {{"0", false}, {"1", true}}); }

// Whether a quantizing call of width `tokens` runs on the 16-token tile.
inline bool takes_tile16(std::int32_t tokens) {
  return w8a8_tile16() && tokens <= IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS;
}

struct W8a8Workspace {
  void *codes = nullptr;
  float *scales = nullptr;
};

template <class Arena> W8a8Workspace allocate_w8a8_workspace(Arena &arena, std::int32_t tokens, std::int32_t cols) {
  if (tokens <= 0 || cols <= 0) throw std::invalid_argument("fp8 W8A8 workspace: T and K must be positive");
  const ninfer::DeviceSpan codes = arena.alloc_bytes(static_cast<std::size_t>(tokens) * cols, 256);
  const ninfer::DeviceSpan scales = arena.alloc_bytes(static_cast<std::size_t>(tokens) * 4, 256);
  return {codes.data, static_cast<float *>(scales.data)};
}

// The workspace a call of up to `max_tokens` needs: none on the BF16-activation route.
inline std::size_t workspace_capacity_bytes(ninfer::ops::LinearPolicy policy, std::int32_t max_tokens,
                                            std::int32_t cols) {
  if (!quantizes(policy, max_tokens)) return 0;
  ninfer::WorkspaceLayoutBuilder layout;
  (void)allocate_w8a8_workspace(layout, max_tokens, cols);
  return layout.peak_bytes(1);
}

// A row-scale weight's descriptor checks: shape, layout, alignment and the payload's extent.
inline void require_weight(const ninfer::Weight &w, std::int32_t rows, std::int32_t cols, std::uint64_t bytes,
                           const char *what) {
  if (w.qtype != ninfer::QType::FP8_E4M3FN_ROW_BF16S || w.layout != ninfer::QuantLayout::RowScale ||
      w.qdata == nullptr || (reinterpret_cast<std::uintptr_t>(w.qdata) & 15) != 0 || w.n != rows || w.k != cols ||
      w.payload_bytes < bytes) {
    throw std::invalid_argument(std::string(what) + ": not an FP8 row-scale [" + std::to_string(rows) + "][" +
                                std::to_string(cols) + "] weight of " + std::to_string(bytes) + " bytes");
  }
}

}  // namespace ignis::fp8_exception

#endif  // IGNIS_FP8_EXCEPTION_H
