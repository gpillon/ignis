// ignis kernel leaf: the launcher of the 27B prefill's gate and up product with SwiGLU in its
// epilogue, and the switch that picks it at a load -- an ADAPTED op (ADR 0051; spec #646, GitHub
// #651; kernel/src/prefill_swiglu.h). The switch is our own.
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1 (the
// commit kernel/vendor/manifest.json pins, ADR 0010), Apache-2.0 (kernel/vendor/LICENSE), restated:
//   src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu
//       make_descriptors (the four tensor maps, their boxes and swizzles) and
//       launch_nvfp4_linear_swiglu_w4a4_tma (the shared-memory attribute set once, the grid, and on
//       Windows the per-launch device copy of the descriptor block);
//   src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
//       nvfp4_linear_swiglu_dispatch's TMA arm: one workspace scope, the W4A4 activation scratch,
//       the vendored quantizer, and alpha = 1 / (input divisor x weight divisor).
// Called and not copied: allocate_nvfp4_w4a4_workspace, launch_nvfp4_w4a4_quantize,
// nvfp4_make_tma_2d and validate_nvfp4_weight. Changed here: the kernel (prefill_swiglu.cuh) and
// its two stages; on Windows the descriptor block is the load's resident copy where its slots
// serve the stream (kernel/src/tma_descriptor_cache.h), upstream's per-launch copy elsewhere.
#include "prefill_swiglu.h"

#include "prefill_swiglu.cuh"
#include "process_environment.h"
#include "tma_descriptor_cache.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::prefill_swiglu {
namespace {

namespace nd = ninfer::ops::detail;
using Geometry = nd::Nvfp4MlpGateUpGeometry;

constexpr const char *kSwitch = "IGNIS_PREFILL_SWIGLU";
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kPairN = GateUpSchedule::kBlockN / 2;
static_assert(GateUpSchedule::kBlockM == kTileTokens, "the op's token tile is the TMA route's");

nd::Nvfp4W4a4TmaDescriptors make_descriptors(const std::uint8_t *activation_codes,
                                             const std::uint8_t *activation_scales,
                                             const std::uint8_t *weight_codes, const std::uint8_t *weight_scales,
                                             std::int32_t tokens) {
  constexpr std::uint32_t kCodeColumns = 64;
  constexpr std::uint32_t kScaleColumns = 16;
  constexpr std::uint64_t kWeightScaleBytes =
      static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;
  nd::Nvfp4W4a4TmaDescriptors d{};
  d.a_codes = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    Geometry::kCodeBytesPerRow, tokens, Geometry::kCodeBytesPerRow, kCodeColumns,
                                    GateUpSchedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_64B,
                                    "prefill swiglu: activation codes TMA");
  d.b_codes = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow,
                                    kCodeColumns, kPairN, CU_TENSOR_MAP_SWIZZLE_64B,
                                    "prefill swiglu: weight codes TMA");
  d.a_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns,
                                     GateUpSchedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     "prefill swiglu: activation scales TMA");
  d.b_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
                                     kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     "prefill swiglu: weight scales TMA");
  return d;
}

void launch(const nd::Nvfp4W4a4TmaDescriptors &descriptors, std::int32_t tokens, float alpha,
            __nv_bfloat16 *out, cudaStream_t stream) {
  constexpr std::size_t kSharedBytes = sizeof(nd::Nvfp4LinearSwiGluTmaSharedStorage<GateUpSchedule>);
  static const bool kConfigured = [] {
    CUDA_CHECK(cudaFuncSetAttribute(swiglu_tma_kernel<Geometry, GateUpSchedule>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)));
    return true;
  }();
  (void)kConfigured;
  const dim3 grid(kIntermediate / kPairN, tokens / GateUpSchedule::kBlockM);
#ifdef _WIN32
  // The load's resident copy of the block where its slots serve the stream, else upstream's
  // allocation, copy and free around the launch.
  if (const void *resident = ignis::tma_descriptors::device_copy(&descriptors, sizeof(descriptors), stream)) {
    swiglu_tma_kernel<Geometry, GateUpSchedule><<<grid, GateUpSchedule::kThreads, kSharedBytes, stream>>>(
        static_cast<const nd::Nvfp4W4a4TmaDescriptors *>(resident), alpha, out);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  nd::Nvfp4W4a4TmaDescriptors *device = nullptr;
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&device), sizeof(descriptors), stream));
  const cudaError_t copied =
      cudaMemcpyAsync(device, &descriptors, sizeof(descriptors), cudaMemcpyHostToDevice, stream);
  if (copied == cudaSuccess) {
    swiglu_tma_kernel<Geometry, GateUpSchedule><<<grid, GateUpSchedule::kThreads, kSharedBytes, stream>>>(
        device, alpha, out);
  }
  const cudaError_t launched = copied != cudaSuccess ? copied : cudaGetLastError();
  CUDA_CHECK(cudaFreeAsync(device, stream));
  CUDA_CHECK(launched);
#else
  swiglu_tma_kernel<Geometry, GateUpSchedule><<<grid, GateUpSchedule::kThreads, kSharedBytes, stream>>>(
      descriptors, alpha, out);
  CUDA_CHECK(cudaGetLastError());
#endif
}

bool aligned16(const void *p) {
  return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
}

} // namespace

Route route_for_load() {
  const std::string named = ignis::process_environment(kSwitch);
  if (named.empty() || named == "fused") return Route::Fused;
  if (named == "split") return Route::Split;
  throw std::invalid_argument(std::string(kSwitch) + " is `split` or `fused`, not `" + named + "`");
}

void gate_up_swiglu(const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                    ninfer::WorkspaceArena &workspace, cudaStream_t stream) {
  const std::int32_t tokens = x.ne[1];
  if (!takes(tokens)) {
    throw std::invalid_argument("prefill swiglu: " + std::to_string(tokens) +
                                " tokens is no width the W4A4 TMA route takes");
  }
  if (x.dtype != ninfer::DType::BF16 || out.dtype != ninfer::DType::BF16 || x.ne[0] != Geometry::kInputRows ||
      out.ne[0] != kIntermediate || out.ne[1] != tokens || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[2] != 1 ||
      out.ne[3] != 1 || !x.is_contiguous() || !out.is_contiguous() || !aligned16(x.data) || !aligned16(out.data)) {
    throw std::invalid_argument("prefill swiglu: x is [5120, T] and out [17408, T], contiguous BF16");
  }
  if (gate_up.qtype != ninfer::QType::NVFP4 || gate_up.n != Geometry::kOutputRows ||
      gate_up.k != Geometry::kInputRows) {
    throw std::invalid_argument("prefill swiglu: the gate and up weight is the 27B's NVFP4 [34816, 5120]");
  }
  nd::validate_nvfp4_weight(gate_up, "prefill swiglu");
  auto scope = workspace.scope();
  const nd::Nvfp4W4a4Workspace scratch = nd::allocate_nvfp4_w4a4_workspace(workspace, tokens, Geometry::kInputRows);
  nd::launch_nvfp4_w4a4_quantize(x, gate_up, scratch, stream);
  const float alpha = 1.0F / (gate_up.input_scale_divisor * gate_up.weight_scale_divisor);
  launch(make_descriptors(scratch.codes, scratch.scales, static_cast<const std::uint8_t *>(gate_up.qdata),
                          static_cast<const std::uint8_t *>(gate_up.scales), tokens),
         tokens, alpha, static_cast<__nv_bfloat16 *>(out.data), stream);
}

} // namespace ignis::prefill_swiglu
