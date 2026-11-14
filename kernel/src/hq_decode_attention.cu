// ignis kernel leaf: the launcher of the 27B's warp-specialized hq-e8-2b attention route in the
// decode and verify rounds, and the rule that picks it at a load -- OURS (spec #570, GitHub #572;
// kernel/src/hq_decode_attention.h).
//
// The launch restates the vendored call it stands in for, so nothing above the kernel can tell the
// two apart. From the vendored reference (kernel/vendor, ADR 0010), restated and not changed:
//   src/ops/wrapper/gqa_attention.cpp   gqa_attention's small-T branch: one workspace scope, the
//                                       split capacity, and allocate_small_t_workspace's three
//                                       partial tensors in its order and shapes;
//   src/ops/launcher/gqa_attention_decode.cu
//                                       gqa_attention_small_t_launch_for: the hq route's grid and
//                                       arguments (gqa_attention_decode_hq_routes.cuh), then the
//                                       reducer's grid, block and arguments.
// Called and not copied: gqa_attention_split_capacity and gqa_attention_resolve_route
// (ops/launcher/gqa_attention.h), and the reducer itself,
// gqa_attention_small_t_reduce_output_kernel (ops/kernel/gqa_attention_decode.cuh), instantiated
// here for the specializations the vendored launcher instantiates too, from the same header and for
// the same architectures. Changed here: the partials come from ignis's route
// (kernel/src/hq_decode_attention_ws.cu), not the vendored kernel.
#include "hq_decode_attention.h"

#include "process_environment.h"

#include "core/pdl.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/launcher/gqa_attention.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ignis::hq_decode_attention {
namespace {

using ninfer::DType;
using ninfer::Tensor;
using ninfer::ops::Gqa27Geometry;
using ninfer::ops::kGqaHeadDim;

constexpr const char *kSwitch = "IGNIS_HQ_DECODE_ATTENTION";

void require(bool ok, const char *what) {
  if (!ok) throw std::invalid_argument(std::string("hq_decode_attention: ") + what);
}

bool shaped(const Tensor &t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2,
            std::int32_t n3) {
  return t.data != nullptr && t.dtype == dtype && t.ne[0] == n0 && t.ne[1] == n1 && t.ne[2] == n2 &&
         t.ne[3] == n3 && t.is_contiguous();
}

// The vendored launcher's reducer launch for the 27B geometry over hq-e8-2b: kDChunk 64, no INT8
// scales, no column offset (the batch body's calls are one pass from column 0).
template <bool MultiBatch, bool Masked>
cudaError_t reduce(const Tensor &acc, const Tensor &m, const Tensor &l, const Tensor &positions,
                   const std::int32_t *valid_columns, const Tensor &gate, std::int32_t width,
                   std::int32_t batch, std::int32_t splits, Tensor &out, cudaStream_t stream) {
  constexpr int kReduceBlock = 256;
  constexpr int kDChunk = 64;
  const dim3 grid(Gqa27Geometry::QHeads, (kGqaHeadDim + kDChunk - 1) / kDChunk, width * batch);
  return ninfer::pdl::launch_dependent(
      {grid, dim3(kReduceBlock), 0, stream},
      ninfer::ops::gqa_attention_small_t_reduce_output_kernel<Gqa27Geometry, kDChunk, false, MultiBatch,
                                                              Masked, false>,
      static_cast<const __nv_bfloat16 *>(acc.data), static_cast<const float *>(m.data),
      static_cast<const float *>(l.data), static_cast<const std::int32_t *>(positions.data),
      valid_columns, static_cast<const __nv_bfloat16 *>(gate.data), width, width,
      std::int32_t{0}, batch, splits, static_cast<__nv_bfloat16 *>(out.data));
}

}  // namespace

Switch switch_for_load() {
  const std::string named = ignis::process_environment(kSwitch);
  if (named.empty()) return Switch::Unset;
  if (named == "ours") return Switch::Ours;
  if (named == "vendored") return Switch::Vendored;
  throw std::invalid_argument(std::string(kSwitch) + " is `ours` or `vendored`, not `" + named + "`");
}

bool serves(std::int32_t q_heads, std::int32_t width, std::int32_t batch, DType cache_dtype,
            ninfer::ops::GqaExecutionEnvelope envelope) {
  return q_heads == Gqa27Geometry::QHeads && cache_dtype == DType::U8 &&
         ninfer::ops::detail::gqa_attention_resolve_route(q_heads, width, batch, cache_dtype, envelope) ==
             ninfer::ops::detail::GqaAttentionRoute::SmallT;
}

void attend(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &positions,
            const Tensor &valid_columns, const Tensor &kv_table_rows, const Tensor &gate, float scale,
            const ninfer::PagedKVBatchLayerView &cache, ninfer::ops::GqaExecutionEnvelope envelope,
            ninfer::WorkspaceArena &workspace, Tensor &out, cudaStream_t stream) {
  constexpr std::int32_t kQHeads = Gqa27Geometry::QHeads;
  constexpr std::int32_t kKvHeads = Gqa27Geometry::KVHeads;
  const std::int32_t width = q.ne[2];
  const std::int32_t batch = q.ne[3];
  require(serves(q.ne[1], width, batch, cache.dtype, envelope),
          "a call of the 27B's 24 query heads over hq-e8-2b on the small-T route (1..8 columns a lane)");
  require(shaped(q, DType::BF16, kGqaHeadDim, kQHeads, width, batch), "q is BF16 [256, 24, W, B]");
  require(shaped(k, DType::BF16, kGqaHeadDim, kKvHeads, width, batch), "k is BF16 [256, 4, W, B]");
  require(shaped(v, DType::BF16, kGqaHeadDim, kKvHeads, width, batch), "v is BF16 [256, 4, W, B]");
  require(shaped(out, DType::BF16, kGqaHeadDim, kQHeads, width, batch), "out is BF16 [256, 24, W, B]");
  require(shaped(gate, DType::BF16, kGqaHeadDim, kQHeads, width, batch), "gate is BF16 [256, 24, W, B]");
  require(shaped(positions, DType::I32, width, batch, 1, 1), "positions are I32 [W, B]");
  require(shaped(kv_table_rows, DType::I32, batch, 1, 1, 1), "the table rows are I32 [B]");
  require(valid_columns.data == nullptr || shaped(valid_columns, DType::I32, batch, 1, 1, 1),
          "the valid columns are empty or I32 [B]");
  require(cache.block_tables.data != nullptr && cache.block_tables.dtype == DType::I32,
          "the cache's block tables are I32");

  auto scope = workspace.scope();
  const std::int32_t splits =
      ninfer::ops::detail::gqa_attention_split_capacity(kQHeads, width, cache.dtype, envelope);
  Tensor acc = workspace.alloc(DType::BF16, {kGqaHeadDim, kQHeads, width, splits * batch});
  Tensor m = workspace.alloc(DType::FP32, {kQHeads, width, splits * batch});
  Tensor l = workspace.alloc(DType::FP32, {kQHeads, width, splits * batch});

  const auto *masked = static_cast<const std::int32_t *>(valid_columns.data);
  PartialLaunch launch;
  launch.grid = dim3(kKvHeads, splits, batch);
  launch.q = static_cast<const __nv_bfloat16 *>(q.data);
  launch.k = static_cast<const __nv_bfloat16 *>(k.data);
  launch.v = static_cast<const __nv_bfloat16 *>(v.data);
  launch.positions = static_cast<const std::int32_t *>(positions.data);
  launch.codes_k = static_cast<std::uint8_t *>(cache.k_pages.data);
  launch.codes_v = static_cast<std::uint8_t *>(cache.v_pages.data);
  launch.meta_k = static_cast<std::uint8_t *>(cache.k_scale_pages.data);
  launch.meta_v = static_cast<std::uint8_t *>(cache.v_scale_pages.data);
  launch.residual_k = static_cast<__nv_bfloat16 *>(cache.residual_k.data);
  launch.residual_v = static_cast<__nv_bfloat16 *>(cache.residual_v.data);
  launch.ring_valid = static_cast<std::uint32_t *>(cache.ring_valid.data);
  launch.block_tables = static_cast<const std::int32_t *>(cache.block_tables.data);
  launch.valid_columns = masked;
  launch.table_rows = static_cast<const std::int32_t *>(kv_table_rows.data);
  launch.table_stride = cache.block_tables.ne[0];
  launch.tokens = width;
  launch.full_width = width;
  launch.column_begin = 0;
  launch.logical_capacity = static_cast<std::int32_t>(envelope.max_visible_keys);
  launch.scale = scale;
  launch.partial_acc = static_cast<__nv_bfloat16 *>(acc.data);
  launch.partial_m = static_cast<float *>(m.data);
  launch.partial_l = static_cast<float *>(l.data);
  launch_partials(launch, stream);

  // The vendored launcher's profile: one lane or many, the dense mask or a lane's valid prefix.
  const auto reducer = batch == 1 ? (masked == nullptr ? reduce<false, false> : reduce<false, true>)
                                  : (masked == nullptr ? reduce<true, false> : reduce<true, true>);
  const cudaError_t err = reducer(acc, m, l, positions, masked, gate, width, batch, splits, out, stream);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(std::string("hq_decode_attention: the reducer's launch: ") +
                             cudaGetErrorString(err));
  }
}

} // namespace ignis::hq_decode_attention
