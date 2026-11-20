// GitHub #687 (spec #684): which batch rounds run their one lane's attention in chunks of the
// small-T tile (kernel/src/round_shape.h `one_lane_in_chunks`), from a table -- OURS, host-only.
//
// The batch body (kernel/src/gqa_layer.cu `run_gqa_layer_batch`) hands every round's attention to
// the vendored wrapper, `ninfer::ops::gqa_attention`, with one exception the extended round shape
// brought: its one lane at 9..16 columns, which the wrapper's resolver would send to its prompt
// route, runs the wrapper's chunk loop at one lane. Two tables:
//
//   1. A load at seven drafts or fewer is main's. No round of such a load is at an extended shape,
//      and none takes the exception, whatever its lanes, columns or cache format: the small-T tile
//      is 8 columns under hq-e8-2b and 6 under BF16, so a one-lane verify round of 7 or 8 columns
//      over BF16 is past the tile and still the wrapper's (its prompt route, which this test
//      names). A round that took the chunk loop there would run other kernels than main's on the
//      format every correctness oracle loads. A base round of a load of two shapes is the same
//      round.
//   2. The extended round. Over hq-e8-2b and BF16, its one lane at 9..16 columns runs in chunks;
//      two lanes or more are the wrapper's own chunked small-T route.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "round_shape.h"

#include "core/dtype.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using ignis::round_shape::one_lane_in_chunks;
using ninfer::DType;
using ninfer::ops::detail::GqaAttentionRoute;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

constexpr std::int32_t kQHeads = 24;
// The widest round of the base shape, the widest of an extended one, and the lanes a round has at
// most at each.
constexpr std::int32_t kBaseColumns = IGNIS_BASE_SHAPE_DRAFT_TOKENS + 1;
constexpr std::int32_t kExtendedColumns = IGNIS_DFLASH2_MAX_DRAFT_TOKENS + 1;
constexpr std::int32_t kMostLanes = 8;
constexpr std::int32_t kExtendedLanes = IGNIS_EXTENDED_SHAPE_MAX_LANES;
// An envelope as a round declares it: the load's context.
constexpr ninfer::ops::GqaExecutionEnvelope kEnvelope{1, 262144};

const char *name_of(DType dtype) { return dtype == DType::U8 ? "hq-e8-2b" : "BF16"; }

GqaAttentionRoute wrapper_route(std::int32_t columns, std::int32_t lanes, DType dtype) {
  return ninfer::ops::detail::gqa_attention_resolve_route(kQHeads, columns, lanes, dtype, kEnvelope);
}

// Table 1: no round of a load at seven drafts or fewer, and no base round, leaves the wrapper.
void check_the_base_shape_is_main() {
  check(kBaseColumns == 8 && kExtendedColumns == 16 && kExtendedLanes == 4,
        "the base shape is 8 columns a lane, an extended one at most 16 at 4 lanes");
  for (const DType dtype : {DType::U8, DType::BF16}) {
    const std::int32_t tile = ninfer::ops::detail::gqa_small_t_chunk_tokens(dtype);
    check(tile == (dtype == DType::U8 ? 8 : 6), std::string(name_of(dtype)) + ": the small-T tile is " +
                                                    std::to_string(tile) + " columns");
    for (std::int32_t lanes = 1; lanes <= kMostLanes; ++lanes) {
      for (std::int32_t columns = 1; columns <= kBaseColumns; ++columns) {
        check(!one_lane_in_chunks(/*extended_round=*/false, lanes, columns, tile),
              std::string(name_of(dtype)) + ", " + std::to_string(lanes) + " lane(s) of " + std::to_string(columns) +
                  " columns: a round of a load at seven drafts or fewer is the vendored wrapper's");
      }
    }
  }
  // What main runs where BF16's tile ends: one lane of 7 and of 8 columns is the wrapper's prompt
  // route, two lanes or more its chunked small-T; over hq-e8-2b one lane is one small-T tile.
  for (const std::int32_t columns : {7, 8}) {
    check(wrapper_route(columns, 1, DType::BF16) == GqaAttentionRoute::Prompt,
          "BF16, one lane of " + std::to_string(columns) + " columns: the wrapper's prompt route");
    check(wrapper_route(columns, 2, DType::BF16) == GqaAttentionRoute::ChunkedSmallT,
          "BF16, two lanes of " + std::to_string(columns) + " columns: the wrapper's chunked small-T route");
    check(wrapper_route(columns, 1, DType::U8) == GqaAttentionRoute::SmallT,
          "hq-e8-2b, one lane of " + std::to_string(columns) + " columns: one small-T tile");
  }
}

// Table 2: the extended round's one lane.
void check_the_extended_round() {
  for (const DType dtype : {DType::U8, DType::BF16}) {
    const std::int32_t tile = ninfer::ops::detail::gqa_small_t_chunk_tokens(dtype);
    for (std::int32_t columns = kBaseColumns + 1; columns <= kExtendedColumns; ++columns) {
      const std::string at = std::string(name_of(dtype)) + ", " + std::to_string(columns) + " columns a lane";
      check(one_lane_in_chunks(/*extended_round=*/true, 1, columns, tile),
            at + ": an extended round's one lane runs in chunks of the tile");
      // Why the exception exists, and why it stops at one lane.
      check(wrapper_route(columns, 1, dtype) == GqaAttentionRoute::Prompt,
            at + ": the wrapper would send one lane to its prompt route");
      for (std::int32_t lanes = 2; lanes <= kExtendedLanes; ++lanes) {
        check(!one_lane_in_chunks(true, lanes, columns, tile), at + ", " + std::to_string(lanes) +
                                                                   " lanes: the wrapper's own call");
        check(wrapper_route(columns, lanes, dtype) == GqaAttentionRoute::ChunkedSmallT,
              at + ", " + std::to_string(lanes) + " lanes: the wrapper chunks by the tile itself");
      }
    }
  }
}

} // namespace

int main() {
  check_the_base_shape_is_main();
  check_the_extended_round();
  if (g_failed != 0) {
    std::fprintf(stderr, "one_lane_chunks: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("one_lane_chunks: all checks passed\n");
  return 0;
}
