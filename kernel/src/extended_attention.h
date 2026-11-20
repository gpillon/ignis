// ignis kernel leaf - GitHub #688 (spec #684): the attention of a round at a load's extended shape
// (kernel/src/round_shape.h), 9..16 columns a lane at 1..4 lanes -- OURS, leaf-internal.
//
// Such a round's columns are past the vendored small-T tile (8 columns over hq-e8-2b, 6 over BF16),
// so its attention runs in chunks of the tile, each chunk appending its own columns' K and V and
// attending from its first column of the lane's, the next chunk reading what the one before
// appended: the vendored chunked small-T route. One loop runs the chunks for every such call
// (`attend`), on one of two routes:
//   - kOurs, ignis's warp-specialized route over hq-e8-2b (kernel/src/hq_decode_attention.h
//     `attend_chunk`), wherever the load takes that route for its base rounds (an RTX 50 or RTX
//     PRO card's hq-e8-2b load, `ignis_model_hq_decode_attention`), unless the measurement switch
//     IGNIS_EXTENDED_ATTENTION=vendored names the other;
//   - kVendored, the vendored chunk itself (`ninfer::ops::detail::gqa_attention_small_t_launch`)
//     everywhere else: another card, a BF16 load, and under the switch.
// The two write the same bytes, output and KV append (kernel/tests/test_hq_verify_chunked_exact.cu,
// which also holds this loop on the vendored chunks to the vendored wrapper's own chunked route,
// `ninfer::ops::gqa_attention`, at 2..4 lanes). A round of a load without an extended shape, and a
// base round, never comes here: its attention is the call it was on main, on every KV format.
#pragma once

#include "process_environment.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ignis::extended_attention {

// Whether a batch round's attention runs in chunks of the small-T tile (`small_t_tile` columns):
// an extended round's, whose `lane_columns` are past the tile. No other round's: a base round and
// every round of a load without an extended shape is the vendored wrapper's or, over hq-e8-2b on an
// RTX 50 or RTX PRO card, ignis's route in one pass, whatever its lanes and its KV format.
inline bool in_chunks(bool extended_round, std::int32_t lane_columns, std::int32_t small_t_tile) {
  return extended_round && lane_columns > small_t_tile;
}

// The route of each chunk.
enum class Route : std::uint8_t { kVendored, kOurs };

// The route an extended round's chunks take: ignis's where the load takes ignis's route for its base
// rounds (`load_takes_ours`, kernel/src/model_internal.h `hq_decode_attention_ours`), the call is one
// the route serves in chunks (`route_serves`, `ignis::hq_decode_attention::serves_chunked`) and
// IGNIS_EXTENDED_ATTENTION (`named`) does not name `vendored`; the vendored chunks otherwise.
inline Route route_of(bool load_takes_ours, RouteSwitch named, bool route_serves) {
  return load_takes_ours && named != RouteSwitch::Vendored && route_serves ? Route::kOurs : Route::kVendored;
}

// IGNIS_EXTENDED_ATTENTION, read at each decode graph capture (kernel/src/decode_graph.cu): unset (or
// empty) or `ours`, the extended round's chunks on ignis's route where the load takes it; `vendored`,
// on the vendored chunks. Throws std::invalid_argument, naming the variable, on any other value.
inline RouteSwitch read_switch() { return route_switch("IGNIS_EXTENDED_ATTENTION"); }

// The chunk loop, for a call `in_chunks` admits: q [256, Hq, W, B], k and v [256, Hkv, W, B], gate
// and out as q, positions I32 [W, B], valid columns empty or I32 [B], table rows I32 [B]. Per chunk
// of the tile, in order from column 0: a workspace scope, the chunk's split capacity
// (`gqa_attention_split_capacity`) and its three partial tensors, BF16 [256, Hq, C, S * B] and FP32
// [Hq, C, S * B] twice, then the chunk on `route` at its first column. Every chunk's launches are
// programmatic dependents of the launch before them, so a chunk's attention reads what the chunk
// before it appended. The workspace is the round's: the vendored query at one chunk's widths at one
// lane, at the call's lanes and width past it (kernel/src/gqa_layer.cu). Throws on a call the route
// refuses; enqueues on `stream` and does not synchronize.
void attend(Route route, const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
            const ninfer::Tensor &positions, const ninfer::Tensor &valid_columns,
            const ninfer::Tensor &kv_table_rows, const ninfer::Tensor &gate, float scale,
            const ninfer::PagedKVBatchLayerView &cache, ninfer::ops::GqaExecutionEnvelope envelope,
            ninfer::WorkspaceArena &workspace, ninfer::Tensor &out, cudaStream_t stream);

} // namespace ignis::extended_attention
