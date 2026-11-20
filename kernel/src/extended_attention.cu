// ignis kernel leaf: the chunk loop of an extended round's attention (kernel/src/extended_attention.h;
// spec #684, GitHub #687, #688) -- restated from the vendored wrapper, ours otherwise.
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (the
// commit kernel/vendor/manifest.json pins, ADR 0010), Apache-2.0 (kernel/vendor/LICENSE), restated
// and not changed: src/ops/wrapper/gqa_attention.cpp `launch_chunked_small_t` (per chunk of the
// small-T tile a workspace scope, the chunk's split capacity and allocate_small_t_workspace's three
// partial tensors in their order and shapes, then the launcher at the chunk's first column). Called
// and not copied: gqa_small_t_chunk_tokens, gqa_attention_split_capacity and
// gqa_attention_small_t_launch (src/ops/launcher/gqa_attention.h). Changed here: the loop runs at
// every lane count, one lane included, which the vendored resolver sends to its prompt route
// instead, and a chunk runs on ignis's route (kernel/src/hq_decode_attention.h `attend_chunk`) where
// the round takes it (kernel/NOTICE).
#include "extended_attention.h"

#include "hq_decode_attention.h"

#include "ops/launcher/gqa_attention.h"

#include <algorithm>

namespace ignis::extended_attention {

void attend(Route route, const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
            const ninfer::Tensor &positions, const ninfer::Tensor &valid_columns,
            const ninfer::Tensor &kv_table_rows, const ninfer::Tensor &gate, float scale,
            const ninfer::PagedKVBatchLayerView &cache, ninfer::ops::GqaExecutionEnvelope envelope,
            ninfer::WorkspaceArena &workspace, ninfer::Tensor &out, cudaStream_t stream) {
  const std::int32_t head_dim = q.ne[0];
  const std::int32_t q_heads = q.ne[1];
  const std::int32_t width = q.ne[2];
  const std::int32_t lanes = q.ne[3];
  const std::int32_t tile = ninfer::ops::detail::gqa_small_t_chunk_tokens(cache.dtype);
  for (std::int32_t begin = 0; begin < width; begin += tile) {
    const std::int32_t count = std::min(tile, width - begin);
    auto chunk_scope = workspace.scope();
    const std::int32_t splits =
        ninfer::ops::detail::gqa_attention_split_capacity(q_heads, count, cache.dtype, envelope);
    ninfer::Tensor acc = workspace.alloc(ninfer::DType::BF16, {head_dim, q_heads, count, splits * lanes});
    ninfer::Tensor m = workspace.alloc(ninfer::DType::FP32, {q_heads, count, splits * lanes});
    ninfer::Tensor l = workspace.alloc(ninfer::DType::FP32, {q_heads, count, splits * lanes});
    if (route == Route::kOurs) {
      ignis::hq_decode_attention::attend_chunk(q, k, v, positions, valid_columns, kv_table_rows, gate, scale, cache,
                                               envelope, begin, count, acc, m, l, out, stream);
    } else {
      ninfer::ops::detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows, gate, scale,
                                                        cache, envelope, begin, count, acc, m, l, out, stream);
    }
  }
}

} // namespace ignis::extended_attention
