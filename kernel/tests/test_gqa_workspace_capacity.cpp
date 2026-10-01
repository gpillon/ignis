// The vendored A1 capacity query against the bytes A1 actually bumps -- OURS,
// not vendored (GitHub #296).
//
// `ninfer::ops::gqa_attention_workspace_capacity_bytes` is the only source the
// GQA layer has for how much transient arena one prefill chunk's attention
// needs, and the layer hands A1 an arena of exactly that size
// (`ignis_gqa_attention_workspace_bytes`, kernel/include/ignis_gqa_workspace.h).
// An answer short by even one byte is a `bad_alloc` thrown mid-layer: GitHub
// #296 is a serving conversation past 262,144 keys whose last prompt chunk (330
// tokens) failed that way on every retry, for good.
//
// What is pinned: for every prompt width a serving chunk can have (17..1024 --
// the 1..16 widths are #123's, covered by ignis_kernel_hq_route_agreement_test),
// on an hq-e8-2b (U8) cache, the query at exactly that width covers a replay
// of the allocations `gqa_attention`'s Prompt route makes
// (kernel/vendor/src/ops/wrapper/gqa_attention.cpp, `allocate_hq_prompt_scratch`
// and the key-split partials), in its order and at its 256-byte alignment --
// once with the envelope inside one scratch band and once past it, where the
// route chains a carry state across bands instead of splitting keys.
//
// The banded arm is red against the reference's query: it adds the carry as
// raw arithmetic, `(2 * head_dim + 8) * q_heads * width`, while the op aligns
// each of the carry's three tensors, so the answer is short by the padding
// after `carry_m` whenever `96 * width` is not a multiple of 256. The partials
// of a split launch are far larger and hide it, so the shortfall shows only
// at a width `gqa_prefill_split_count` keeps at one split -- on a 170-SM RTX
// 5090, 321..448 and 769..896 among the widths here. A machine whose SM count
// gives every width here a split would not reproduce it, which is why the
// sweep covers every width rather than naming 330.
//
// Host-only layout arithmetic: the one device call is
// `gqa_prefill_split_count`'s SM-count query, which creates no context and
// touches no device memory. ADR 0006 / docs/agents/testing.md: no
// SKIP_RETURN_CODE.

#include "ignis_gqa_workspace.h"

#include "core/layout.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
constexpr std::int32_t kKvHeads = 4;
// The serving chunk's ceiling (`DEFAULT_SERVING_CHUNK_TOKENS`,
// crates/core/src/concrete.rs): every prefill chunk is at most this wide.
constexpr std::int32_t kWidestChunk = 1024;

// The bytes `gqa_attention` bumps for one B=1 Prompt call on a U8 cache at
// `width` over `envelope`, replayed through the same layout builder the
// arena's own dry runs use, in the op's order.
std::size_t prompt_call_bytes(std::int32_t width, ninfer::ops::GqaExecutionEnvelope envelope) {
  ninfer::WorkspaceLayoutBuilder layout;
  const std::int32_t span = static_cast<std::int32_t>(
      std::min(envelope.max_visible_keys, ninfer::ops::kGqaHqPromptScratchBandKeys));
  (void)layout.alloc(ninfer::DType::BF16, {kHeadDim, kKvHeads, span, 1});
  (void)layout.alloc(ninfer::DType::BF16, {kHeadDim, kKvHeads, span, 1});
  const bool banded = envelope.max_visible_keys > ninfer::ops::kGqaHqPromptScratchBandKeys;
  if (banded) {
    (void)layout.alloc(ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
    (void)layout.alloc(ninfer::DType::FP32, {kQHeads, width, 1, 1});
    (void)layout.alloc(ninfer::DType::FP32, {kQHeads, width, 1, 1});
  } else {
    const std::int32_t splits = ninfer::ops::detail::gqa_prefill_split_count(width, kQHeads);
    if (splits > 1) {
      (void)layout.alloc(ninfer::DType::FP32, {kHeadDim, kQHeads, width, splits});
      (void)layout.alloc(ninfer::DType::FP32, {kQHeads, width, splits});
      (void)layout.alloc(ninfer::DType::FP32, {kQHeads, width, splits});
    }
  }
  return layout.peak_bytes(1);
}

void sweep(std::uint32_t visible_keys, const char *label) {
  int short_widths = 0;
  std::string first_short;
  for (std::int32_t width = kIgnisGqaQueryVerifyCapWidth; width <= kWidestChunk; ++width) {
    const ninfer::ops::GqaExecutionEnvelope envelope{.min_visible_keys = 1,
                                                     .max_visible_keys = visible_keys};
    if (ninfer::ops::detail::gqa_attention_resolve_route(kQHeads, width, 1, ninfer::DType::U8,
                                                         envelope) !=
        ninfer::ops::detail::GqaAttentionRoute::Prompt) {
      continue;
    }
    const std::size_t reserved =
        ignis_gqa_attention_workspace_bytes(ninfer::DType::U8, envelope, width);
    const std::size_t bumped = prompt_call_bytes(width, envelope);
    if (reserved < bumped) {
      if (short_widths == 0) {
        first_short = "width " + std::to_string(width) + ": reserved " + std::to_string(reserved) +
                      ", the call bumps " + std::to_string(bumped);
      }
      ++short_widths;
    }
  }
  check(short_widths == 0, std::string(label) + ": " + std::to_string(short_widths) +
                               " prompt width(s) get less arena than the call bumps, first " +
                               first_short);
}

} // namespace

int main() {
  // --- 1. one scratch band: the key-split partials ride beside the planes -----
  sweep(200'000, "single band (200,000 visible keys)");
  sweep(ninfer::ops::kGqaHqPromptScratchBandKeys, "single band (exactly one band)");

  // --- 2. past one band: the carry chains the bands (GitHub #296) -------------
  sweep(ninfer::ops::kGqaHqPromptScratchBandKeys + 1, "banded (one key past the band)");
  sweep(337'408, "banded (337,408 visible keys)");
  sweep(344'064, "banded (344,064 visible keys, the #296 conversation)");

  if (g_failed != 0) {
    std::fprintf(stderr, "gqa workspace capacity test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("gqa workspace capacity test: ok\n");
  return 0;
}
