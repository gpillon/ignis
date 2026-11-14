// The GQA attention's transient arena -- OURS, not vendored: A1's capacity
// query against the bytes A1 actually bumps (GitHub #296), and the arena of
// ignis's prompt attention, which serves every hq-e8-2b prefill chunk since
// GitHub #335 (ADR 0061, spec 27b-16gb/03 acceptance 1).
//
// #335 first. The prompt attention keeps no prompt planes, so what one call
// bumps stops growing with the context at one band: the same bytes at 32,768
// keys as at 1,048,576, for every prompt width. What a load reserves for the
// layer (`ignis_gqa_attention_capacity_bytes`, kernel/src/model.cu) covers
// every width's call. And against A1's own query -- what the load reserved
// before -- at 524,288 keys and 1,024-token chunks: A1's reservation holds its
// two BF16 prompt planes, and the load's cannot hold even one, both bounds
// replayed through the layout builder at A1's span (GitHub #603). The saving
// is printed with the card's SM count, not gated: it moves with the SM count
// through A1's key-split partials, 1,020,474,240 bytes on the RTX 5090 (ADR
// 0061), 997,861,152 on the RTX 5080, where the 1.0e9 floor it replaced
// failed though nothing was wrong.
//
// #296, below, still pins A1's query: A1's hq Prompt route is the oracle the
// route agreement test holds the prompt attention against, and that test
// sizes it through `ignis_gqa_a1_workspace_bytes`.
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
// on an hq-e8-2b (U8) cache, A1's query at exactly that width covers a replay
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
// Host-only layout arithmetic: the device calls are the SM-count query,
// `gqa_prefill_split_count`'s and the one the arena line prints, which create
// no context and touch no device memory. ADR 0006 / docs/agents/testing.md: no
// SKIP_RETURN_CODE.

#include "ignis_gqa_workspace.h"
#include "ignis_hq_prompt_attention.h"

#include "core/layout.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <cuda_runtime.h>

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

// The keys A1's hq Prompt route materializes into each of its two BF16 prompt
// planes over `envelope`: the history, up to one scratch band
// (`allocate_hq_prompt_scratch`).
std::int32_t prompt_plane_span(ninfer::ops::GqaExecutionEnvelope envelope) {
  return static_cast<std::int32_t>(
      std::min(envelope.max_visible_keys, ninfer::ops::kGqaHqPromptScratchBandKeys));
}

// The bytes `gqa_attention` bumps for one B=1 Prompt call on a U8 cache at
// `width` over `envelope`, replayed through the same layout builder the
// arena's own dry runs use, in the op's order.
std::size_t prompt_call_bytes(std::int32_t width, ninfer::ops::GqaExecutionEnvelope envelope) {
  ninfer::WorkspaceLayoutBuilder layout;
  const std::int32_t span = prompt_plane_span(envelope);
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
        ignis_gqa_a1_workspace_bytes(ninfer::DType::U8, envelope, width);
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

// ---- GitHub #335: the prompt attention's arena ------------------------------

void no_prompt_planes() {
  // Every width: one band's worth of context and the envelope's ceiling bump
  // the same bytes, and so does everything between.
  const std::uint32_t contexts[] = {kIgnisHqPromptBandKeys, 200'000, 262'144, 262'145,
                                    344'064, 524'288, 1'048'576};
  int moved = 0;
  std::string first_moved;
  for (std::int32_t width = 1; width <= kWidestChunk; ++width) {
    const std::size_t at_band = ignis_hq_prompt_workspace_bytes(width, contexts[0]);
    for (const std::uint32_t context : contexts) {
      const std::size_t bytes = ignis_hq_prompt_workspace_bytes(width, context);
      if (bytes != at_band && moved++ == 0) {
        first_moved = "width " + std::to_string(width) + ": " + std::to_string(at_band) +
                      " bytes at one band, " + std::to_string(bytes) + " at " +
                      std::to_string(context) + " keys";
      }
    }
  }
  check(moved == 0, "the prompt attention's arena grows with the context past one band (" +
                        std::to_string(moved) + " cases, first " + first_moved + ")");
  std::printf("prompt attention, 1,024 tokens: %zu bytes at any context past one band\n",
              ignis_hq_prompt_workspace_bytes(kWidestChunk, contexts[0]));
}

void the_load_covers_every_call() {
  // A load reserves the layer's attention once, for widths 1..chunk at its
  // context; every call at a narrower width and history must fit in it.
  const std::uint32_t contexts[] = {4'096, 65'536, 524'288};
  for (const std::uint32_t context : contexts) {
    const ninfer::ops::GqaExecutionEnvelope load{.min_visible_keys = 1, .max_visible_keys = context};
    const std::size_t reserved = ignis_gqa_attention_capacity_bytes(ninfer::DType::U8, load, 1, 1,
                                                                    kWidestChunk);
    int short_calls = 0;
    std::string first;
    for (std::int32_t width = 1; width <= kWidestChunk; ++width) {
      for (const std::uint32_t visible : {static_cast<std::uint32_t>(width), context / 2, context}) {
        if (visible < static_cast<std::uint32_t>(width)) continue;
        const ninfer::ops::GqaExecutionEnvelope call{.min_visible_keys = 1, .max_visible_keys = visible};
        const std::size_t needed = ignis_gqa_attention_workspace_bytes(ninfer::DType::U8, call, width);
        if (needed > reserved && short_calls++ == 0) {
          first = "width " + std::to_string(width) + " at " + std::to_string(visible) + " keys needs " +
                  std::to_string(needed) + ", the load reserved " + std::to_string(reserved);
        }
      }
    }
    check(short_calls == 0, "a call needs more arena than the load at " + std::to_string(context) +
                                " keys reserved (" + std::to_string(short_calls) + ", first " + first + ")");
  }
}

// The bytes the first `planes` of A1's BF16 prompt planes take over
// `envelope`, as `allocate_hq_prompt_scratch` bumps them.
std::size_t prompt_plane_bytes(int planes, ninfer::ops::GqaExecutionEnvelope envelope) {
  ninfer::WorkspaceLayoutBuilder layout;
  for (int plane = 0; plane < planes; ++plane) {
    (void)layout.alloc(ninfer::DType::BF16, {kHeadDim, kKvHeads, prompt_plane_span(envelope), 1});
  }
  return layout.peak_bytes(1);
}

// The SM count of the card the test runs on, read as `gqa_prefill_split_count`
// reads it; 0 when the query fails.
int card_sm_count() {
  int device = 0;
  int count = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
    return 0;
  }
  return count;
}

void the_planes_are_gone_at_524288() {
  // What the load reserved for one hq layer's attention before #335 -- A1's
  // own query over the serving widths: its two prompt planes and its key-split
  // partials -- against what it reserves now. A1's must hold both planes, or
  // the comparison says nothing; the load's must not hold one. The saving is
  // printed, not gated: the partials grow with the SM count.
  const ninfer::ops::GqaExecutionEnvelope load{.min_visible_keys = 1, .max_visible_keys = 524'288};
  const std::size_t before = ninfer::ops::gqa_attention_workspace_capacity_bytes(
      kQHeads, ninfer::DType::U8, load, 1, 1, kWidestChunk);
  const std::size_t now =
      ignis_gqa_attention_capacity_bytes(ninfer::DType::U8, load, 1, 1, kWidestChunk);
  const std::size_t planes = prompt_plane_bytes(2, load);
  const std::size_t one_plane = prompt_plane_bytes(1, load);
  std::printf("hq attention arena at 524,288 keys, 1,024-token chunks, %d SMs: %zu bytes with prompt "
              "planes, %zu now, %lld fewer; A1's two prompt planes %zu bytes, one %zu\n",
              card_sm_count(), before, now,
              static_cast<long long>(before) - static_cast<long long>(now), planes, one_plane);
  check(before >= planes, "A1's reservation at 524,288 keys (" + std::to_string(before) +
                              " bytes) does not hold its two prompt planes (" +
                              std::to_string(planes) + ")");
  check(now < one_plane, "the reservation at 524,288 keys (" + std::to_string(now) +
                             " bytes) can hold a prompt plane (" + std::to_string(one_plane) + ")");
  // BF16 keeps A1, untouched.
  check(ignis_gqa_attention_capacity_bytes(ninfer::DType::BF16, load, 1, 1, kWidestChunk) ==
            ninfer::ops::gqa_attention_workspace_capacity_bytes(kQHeads, ninfer::DType::BF16, load, 1,
                                                                1, kWidestChunk),
        "a BF16 load's reservation moved");
}

} // namespace

int main() {
  // --- 0. GitHub #335: no prompt planes -----------------------------------
  no_prompt_planes();
  the_load_covers_every_call();
  the_planes_are_gone_at_524288();

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
