/* ignis kernel leaf: how much transient arena one A1 (`gqa_attention`) call
 * needs, with the one correction this engine applies to the vendored capacity
 * query (P4-05, GitHub #123).
 *
 * Not part of the public flat C ABI. It lives in `kernel/include` rather than
 * beside `kernel/src/gqa_layer.cu` for the same reason `ignis_seq_internal.h`
 * does: the leaf's own CTest
 * (kernel/tests/test_hq_route_agreement.cu) must size its workspace through
 * the function the engine actually ships, or the narrow-width arm there would
 * be testing a second copy of the arithmetic and the shipped one could break
 * unseen.
 */
#ifndef IGNIS_GQA_WORKSPACE_H
#define IGNIS_GQA_WORKSPACE_H

#include "ninfer/ops/gqa_attention.h"
/* The vendored route resolver, read-only: the correction below is gated on
 * the op's own answer about which kernel a width takes, rather than on a
 * width threshold restated here. */
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

/* The Q-head geometry the 27B text backbone runs: 24 query heads over 4 KV
 * heads of 256. */
inline constexpr std::int32_t kIgnisGqaQHeads = 24;

/* One past the vendored capacity query's own verify cap
 * (`kMaximumVerifyTokens` = 16, kernel/vendor/src/ops/wrapper/
 * gqa_attention.cpp): the narrowest width at which that query accounts for an
 * hq prompt call's two transient riders together rather than separately. */
inline constexpr std::int32_t kIgnisGqaQueryVerifyCapWidth = 17;

/* The transient arena A1 needs for one B=1 call at `width` over `envelope`.
 *
 * An hq-e8-2b Prompt call holds two transient riders at once: the
 * rotated-frame BF16 span planes it materializes the visible history into,
 * and the key-split partials. `gqa_attention_workspace_capacity_bytes` adds
 * them together only for widths above its own 16-token verify cap; at or
 * below that cap it combines them with `max` instead, and the answer comes
 * back short by one split-partial set. This engine asks per call, at exactly
 * the width it is about to run, so a 9..16-token prefill chunk -- a short
 * prompt, or the tail of a longer span -- would get an arena the op then
 * overruns, and the op's own bump throws `bad allocation` mid-layer. (The
 * reference's own engine does not meet this: it queries once over a whole
 * width interval, where the widths above the cap dominate the answer.)
 *
 * The correction is to ask over an interval wide enough that the answer
 * bounds this width, which is the query's documented contract -- it "returns
 * the transient arena capacity required for every W in the inclusive
 * interval". The split partials grow with the width at a fixed split count
 * across this range, so the answer at `kIgnisGqaQueryVerifyCapWidth` covers
 * any narrower width, and the envelope is raised with it because the query
 * refuses an interval its envelope cannot hold. Only a narrow Prompt width
 * pays the difference, and the difference is one split-partial set (under
 * 2 MB at this geometry).
 *
 * Deliberately a correction at this caller rather than a patch to the
 * vendored query (ADR 0010): it changes no kernel, no numerics and no route
 * -- only how many bytes this caller reserves before making the call.
 *
 * A decode round's own query is not this one: it runs B lanes of one token,
 * which the resolver sends to the small-T route where no such rider exists,
 * so kernel/src/gqa_layer.cu's graph path asks the vendored query directly
 * at its own batch size. */
inline std::size_t ignis_gqa_attention_workspace_bytes(ninfer::DType cache_dtype,
                                                       ninfer::ops::GqaExecutionEnvelope envelope,
                                                       std::int32_t width) {
  ninfer::ops::GqaExecutionEnvelope query_envelope = envelope;
  std::int32_t query_width = width;
  const bool prompt_route = ninfer::ops::detail::gqa_attention_resolve_route(
                                kIgnisGqaQHeads, width, /*batch_size=*/1, cache_dtype, envelope) ==
                            ninfer::ops::detail::GqaAttentionRoute::Prompt;
  if (cache_dtype == ninfer::DType::U8 && prompt_route &&
      width < kIgnisGqaQueryVerifyCapWidth) {
    query_width = kIgnisGqaQueryVerifyCapWidth;
    query_envelope.max_visible_keys =
        std::max(query_envelope.max_visible_keys, static_cast<std::uint32_t>(query_width));
  }
  return ninfer::ops::gqa_attention_workspace_capacity_bytes(
      kIgnisGqaQHeads, cache_dtype, query_envelope, /*batch_size=*/1, width, query_width);
}

#endif /* IGNIS_GQA_WORKSPACE_H */
