// GitHub #687, #688 (spec #684): which batch rounds run their attention in chunks of the small-T tile,
// and on which route (kernel/src/extended_attention.h), from tables -- OURS, host-only.
//
// The batch body (kernel/src/gqa_layer.cu `run_gqa_layer_batch`) hands every round's attention to
// the vendored wrapper, `ninfer::ops::gqa_attention`, or, over hq-e8-2b on an RTX 50 or RTX PRO card,
// to ignis's route in one pass, with one exception the extended round shape brought: its 9..16
// columns a lane run the one chunk loop, on ignis's route or on the vendored chunks. Three tables:
//
//   1. A load at seven drafts or fewer is main's. No round of such a load is at an extended shape,
//      and none takes the chunk loop, whatever its lanes, columns or cache format: the small-T tile
//      is 8 columns under hq-e8-2b and 6 under BF16, so a one-lane verify round of 7 or 8 columns
//      over BF16 is past the tile and still the wrapper's (its prompt route, which this test
//      names). A round that took the chunk loop there would run other kernels than main's on the
//      format every correctness oracle loads. A base round of a load of two shapes is the same
//      round.
//   2. The extended round. Over hq-e8-2b and BF16, at 1..4 lanes, its 9..16 columns run in chunks:
//      the loop stands in for the wrapper's own chunked small-T route at two lanes or more, and for
//      its prompt route at one lane, whose workspace a decode load does not reserve.
//   3. The route of its chunks: ignis's exactly where the load takes ignis's route for its base
//      rounds, the route serves the call (the 27B's 24 heads over hq-e8-2b, past one tile and
//      within two) and IGNIS_EXTENDED_ATTENTION does not name `vendored`; the vendored chunks in
//      every other case.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE -- but nothing here needs a GPU.

#include "extended_attention.h"
#include "hq_decode_attention.h"
#include "ignis_model.h"

#include "core/dtype.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using ignis::RouteSwitch;
using ignis::extended_attention::in_chunks;
using ignis::extended_attention::Route;
using ignis::extended_attention::route_of;
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

// Table 1: no round of a load at seven drafts or fewer, and no base round, takes the chunk loop.
void check_the_base_shape_is_main() {
  check(kBaseColumns == 8 && kExtendedColumns == 16 && kExtendedLanes == 4,
        "the base shape is 8 columns a lane, an extended one at most 16 at 4 lanes");
  for (const DType dtype : {DType::U8, DType::BF16}) {
    const std::int32_t tile = ninfer::ops::detail::gqa_small_t_chunk_tokens(dtype);
    check(tile == (dtype == DType::U8 ? 8 : 6), std::string(name_of(dtype)) + ": the small-T tile is " +
                                                    std::to_string(tile) + " columns");
    for (std::int32_t columns = 1; columns <= kBaseColumns; ++columns) {
      check(!in_chunks(/*extended_round=*/false, columns, tile),
            std::string(name_of(dtype)) + ", " + std::to_string(columns) +
                " columns a lane: a round of a load at seven drafts or fewer takes no chunk loop");
    }
  }
  // What main runs where BF16's tile ends: one lane of 7 and of 8 columns is the wrapper's prompt
  // route, two lanes or more its chunked small-T; over hq-e8-2b one lane is one small-T tile.
  for (const std::int32_t columns : {7, 8}) {
    check(wrapper_route(columns, 1, DType::BF16) == GqaAttentionRoute::Prompt,
          "BF16, one lane of " + std::to_string(columns) + " columns: the wrapper's prompt route");
    for (std::int32_t lanes = 2; lanes <= kMostLanes; ++lanes) {
      check(wrapper_route(columns, lanes, DType::BF16) == GqaAttentionRoute::ChunkedSmallT,
            "BF16, " + std::to_string(lanes) + " lanes of " + std::to_string(columns) +
                " columns: the wrapper's chunked small-T route");
    }
    check(wrapper_route(columns, 1, DType::U8) == GqaAttentionRoute::SmallT,
          "hq-e8-2b, one lane of " + std::to_string(columns) + " columns: one small-T tile");
  }
}

// Table 2: the extended round, at every lane count it runs.
void check_the_extended_round() {
  for (const DType dtype : {DType::U8, DType::BF16}) {
    const std::int32_t tile = ninfer::ops::detail::gqa_small_t_chunk_tokens(dtype);
    for (std::int32_t columns = kBaseColumns + 1; columns <= kExtendedColumns; ++columns) {
      const std::string at = std::string(name_of(dtype)) + ", " + std::to_string(columns) + " columns a lane";
      check(in_chunks(/*extended_round=*/true, columns, tile), at + ": an extended round runs in chunks of the tile");
      // What the loop stands in for.
      check(wrapper_route(columns, 1, dtype) == GqaAttentionRoute::Prompt,
            at + ": the wrapper would send one lane to its prompt route");
      for (std::int32_t lanes = 2; lanes <= kExtendedLanes; ++lanes) {
        check(wrapper_route(columns, lanes, dtype) == GqaAttentionRoute::ChunkedSmallT,
              at + ", " + std::to_string(lanes) + " lanes: the wrapper would chunk by the tile itself");
      }
    }
  }
}

// Table 3: the route of the chunks.
void check_the_route() {
  // The calls ignis's route serves in chunks: 24 heads over hq-e8-2b, a second chunk of 1..8 columns.
  for (std::int32_t columns = 1; columns <= kExtendedColumns + 1; ++columns) {
    const bool extended = columns > kBaseColumns && columns <= kExtendedColumns;
    check(ignis::hq_decode_attention::serves_chunked(kQHeads, columns, DType::U8) == extended,
          "hq-e8-2b, " + std::to_string(columns) + " columns a lane: the route serves it in chunks " +
              (extended ? "(a second chunk of " + std::to_string(columns - kBaseColumns) + ")" : "never"));
    check(!ignis::hq_decode_attention::serves_chunked(kQHeads, columns, DType::BF16),
          "BF16, " + std::to_string(columns) + " columns a lane: never on the route");
    check(!ignis::hq_decode_attention::serves_chunked(16, columns, DType::U8),
          "16 query heads, " + std::to_string(columns) + " columns a lane: never on the route");
  }
  // load takes ours | the switch | the route serves the call -> the chunks' route.
  struct Row {
    bool load_ours;
    RouteSwitch named;
    bool serves;
    Route want;
    const char *what;
  };
  const Row rows[] = {
      {true, RouteSwitch::Unset, true, Route::kOurs, "the load's route ours, the switch unset"},
      {true, RouteSwitch::Ours, true, Route::kOurs, "the load's route ours, `ours`"},
      {true, RouteSwitch::Vendored, true, Route::kVendored, "the load's route ours, `vendored`"},
      {true, RouteSwitch::Unset, false, Route::kVendored, "the load's route ours, a call it does not serve"},
      {false, RouteSwitch::Unset, true, Route::kVendored, "the load's route vendored (another card, a BF16 load)"},
      {false, RouteSwitch::Ours, true, Route::kVendored, "the load's route vendored, `ours`"},
      {false, RouteSwitch::Vendored, false, Route::kVendored, "the load's route vendored, `vendored`"},
  };
  for (const Row &row : rows) {
    check(route_of(row.load_ours, row.named, row.serves) == row.want,
          std::string(row.what) + ": the chunks on " + (row.want == Route::kOurs ? "ignis's route" : "the vendored chunks"));
  }
}

} // namespace

int main() {
  check_the_base_shape_is_main();
  check_the_extended_round();
  check_the_route();
  if (g_failed != 0) {
    std::fprintf(stderr, "one_lane_chunks: %d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("one_lane_chunks: all checks passed\n");
  return 0;
}
