// ignis kernel leaf -- fn_indexer_select, the program's entry point to the Flash-Next QSA
// indexer (spec flash-next/04, GitHub #302, slice S3): OURS (ADR 0043). The projection runs on
// fn_linear; the stages (indexer.cu) append the call's keys to the lane's indexer section on every
// call -- a block key needs its tokens' raw keys, which exist only while their x passes through
// the projection, so the state is never skipped -- and select every row's tokens. A separate
// object from the stages, so the stage tests link without the program.

#include "indexer.h"

#include "ignis_seq_internal.h"

namespace ignis::flash_next {

int32_t fn_indexer_select(const Context &ctx, int32_t attn_ordinal, const IndexerWeights &w,
                          const Batch &batch, const void *x, Selection &out,
                          ninfer::DeviceArena &scratch, cudaStream_t stream) {
  namespace ix = indexer;
  const Geometry &g = ctx.g;
  if (const ix::Status st = ix::check_geometry(g)) {
    fn_set_error(st);
    return -1;
  }
  if (ctx.pool == nullptr || ctx.indexer == nullptr) {
    fn_set_error("fn_indexer_select: no seq pool or indexer sections");
    return -1;
  }
  const IndexerLayerState &state = ctx.indexer[attn_ordinal];
  if (state.blocks_per_page != ix::kBlocksPerPage) {
    fn_set_error("fn_indexer_select: indexer pages hold 16 blocks of 4 tokens");
    return -1;
  }
  auto scope = scratch.scope();
  const int32_t qk_cols = (g.indexer_heads + g.indexer_kv_heads) * g.indexer_head_dim;
  void *qk = scratch.alloc_bytes(static_cast<size_t>(batch.rows()) * qk_cols * sizeof(__nv_bfloat16)).data;
  if (fn_linear(w.qk_proj, x, batch.rows(), qk, false, scratch, stream) != 0) return -1;

  ix::Paged paged;
  paged.block_tables = static_cast<const int32_t *>(ctx.pool->kv_pool.block_tables().data);
  paged.logical_pages = static_cast<int32_t>(ctx.pool->kv_pool.logical_page_capacity());
  paged.block_keys = static_cast<__nv_bfloat16 *>(state.block_keys);
  paged.tail_keys = static_cast<__nv_bfloat16 *>(state.tail_keys);
  const ix::Rope rope = ix::rope_from(ctx.rope);
  ix::Status st = ix::append_keys(g, paged, rope, w.k_norm, batch, qk, stream);
  if (st == nullptr) st = ix::select(g, paged, rope, w.q_norm, batch, qk, out, scratch, stream);
  if (st != nullptr) {
    fn_set_error(st);
    return -1;
  }
  return 0;
}

std::size_t fn_indexer_select_scratch_bytes(const Geometry &g, int32_t rows, int32_t max_visible) {
  const std::size_t qk = static_cast<std::size_t>(rows) * (g.indexer_heads + g.indexer_kv_heads) *
                         g.indexer_head_dim * sizeof(__nv_bfloat16);
  return (qk + 255) / 256 * 256 + indexer::select_scratch_bytes(g, rows, max_visible);
}

}  // namespace ignis::flash_next
