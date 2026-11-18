// ignis kernel leaf: the launcher of the 27B prefill's gate and up product with SwiGLU in its
// epilogue, the layers' MLP around it, and the switch that picks it at a load -- an ADAPTED op
// (ADR 0051; spec #646, GitHub #651, #653; kernel/src/prefill_swiglu.h). The switch is our own.
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
// Called and not copied: allocate_nvfp4_w4a4_workspace, launch_nvfp4_w4a4_quantize_for_products,
// launch_nvfp4_w4a4_tma_linear_add, nvfp4_make_tma_2d, validate_nvfp4_weight and ops::linear_add.
// Changed here: the kernel (prefill_swiglu.cuh) and its two stages; the input quantized in the
// layout the stream's W4A4 products read (GitHub #652, #653), and the tensor maps of the K-tile-major
// one, as the vendored product's recorded patch builds them (ops/linear/nvfp4/nvfp4_w4a4_tma.cuh
// make_nvfp4_w4a4_tma_k_tile_descriptors) but for the weight codes' 64-row box of a pair; the
// NVFP4 epilogue and the down product it feeds (GitHub #653); on Windows the descriptor block is
// the load's resident copy where its slots serve the stream (kernel/src/tma_descriptor_cache.h),
// upstream's per-launch copy elsewhere.
#include "prefill_swiglu.h"

#include "prefill_swiglu.cuh"
#include "process_environment.h"
#include "tma_descriptor_cache.h"

#include "core/device.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::prefill_swiglu {
namespace {

namespace nd = ninfer::ops::detail;
using Geometry = nd::Nvfp4MlpGateUpGeometry;
using DownGeometry = nd::Nvfp4Residual17408Geometry;
using Layout = nd::Nvfp4W4a4ActivationLayout;

constexpr const char *kSwitch = "IGNIS_PREFILL_SWIGLU";
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kPairN = GateUpSchedule::kBlockN / 2;
static_assert(GateUpSchedule::kBlockM == kTileTokens, "the op's token tile is the TMA route's");
static_assert(DownGeometry::kInputRows == kIntermediate, "the down product reads the SwiGLU output");

nd::Nvfp4W4a4TmaDescriptors make_descriptors(Layout layout, const std::uint8_t *activation_codes,
                                             const std::uint8_t *activation_scales,
                                             const std::uint8_t *weight_codes, const std::uint8_t *weight_scales,
                                             std::int32_t tokens) {
  constexpr std::uint32_t kCodeColumns = 64;
  constexpr std::uint32_t kScaleColumns = 16;
  constexpr std::uint64_t kWeightScaleBytes =
      static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;
  nd::Nvfp4W4a4TmaDescriptors d{};
  d.b_codes = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow,
                                    kCodeColumns, kPairN, CU_TENSOR_MAP_SWIZZLE_64B,
                                    "prefill swiglu: weight codes TMA");
  if (layout == Layout::KTileMajor) {
    // Codes [K/128][T][64 bytes] as 128-byte rows of two tokens, a K tile of a token tile in one box;
    // scales [K/256][T][16 bytes] as 128-byte rows of eight, a pair of K tiles in one box; the weight
    // scales, their bytes unchanged, in boxes of eight 128-byte rows.
    constexpr std::uint64_t kKTiles = Geometry::kInputRows / 128;
    constexpr std::uint64_t kKTilePairs = Geometry::kInputRows / 256;
    const auto columns = static_cast<std::uint64_t>(tokens);
    d.a_codes = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                      128, kKTiles * columns / 2, 128, 128, GateUpKTileSchedule::kBlockM / 2,
                                      CU_TENSOR_MAP_SWIZZLE_128B, "prefill swiglu: K-tile-major activation codes TMA");
    d.a_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_scales),
                                       CU_TENSOR_MAP_DATA_TYPE_UINT8, 128, kKTilePairs * columns / 8, 128, 128,
                                       GateUpKTileSchedule::kBlockM / 8, CU_TENSOR_MAP_SWIZZLE_NONE,
                                       "prefill swiglu: K-tile-major activation scales TMA");
    d.b_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                       128, kWeightScaleBytes / 128, 128, 128, 8, CU_TENSOR_MAP_SWIZZLE_NONE,
                                       "prefill swiglu: K-tile-major weight scales TMA");
    return d;
  }
  d.a_codes = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                    Geometry::kCodeBytesPerRow, tokens, Geometry::kCodeBytesPerRow, kCodeColumns,
                                    GateUpSchedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_64B,
                                    "prefill swiglu: activation codes TMA");
  d.a_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                     Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns,
                                     GateUpSchedule::kBlockM, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     "prefill swiglu: activation scales TMA");
  d.b_scales = nd::nvfp4_make_tma_2d(const_cast<std::uint8_t *>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
                                     kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE,
                                     "prefill swiglu: weight scales TMA");
  return d;
}

// What one launch writes: `out` under Store::Bf16, `codes`, `scales` and `divisor` under Store::Nvfp4.
struct Outputs {
  __nv_bfloat16 *out = nullptr;
  std::uint8_t *codes = nullptr;
  std::uint8_t *scales = nullptr;
  float divisor = 0.0F;
};

template <class Schedule, Store kStore>
void launch(const nd::Nvfp4W4a4TmaDescriptors &descriptors, std::int32_t tokens, float alpha, const Outputs &o,
            cudaStream_t stream) {
  const auto kernel = swiglu_tma_kernel<Geometry, Schedule, kStore>;
  constexpr std::size_t kSharedBytes = sizeof(nd::Nvfp4LinearSwiGluTmaSharedStorage<Schedule>);
  static const bool kConfigured = [&] {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(kSharedBytes)));
    return true;
  }();
  (void)kConfigured;
  const dim3 grid(kIntermediate / kPairN, tokens / Schedule::kBlockM);
#ifdef _WIN32
  // The load's resident copy of the block where its slots serve the stream, else upstream's
  // allocation, copy and free around the launch.
  if (const void *resident = ignis::tma_descriptors::device_copy(&descriptors, sizeof(descriptors), stream)) {
    kernel<<<grid, Schedule::kThreads, kSharedBytes, stream>>>(
        static_cast<const nd::Nvfp4W4a4TmaDescriptors *>(resident), alpha, o.out, o.codes, o.scales, o.divisor);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  nd::Nvfp4W4a4TmaDescriptors *device = nullptr;
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&device), sizeof(descriptors), stream));
  const cudaError_t copied =
      cudaMemcpyAsync(device, &descriptors, sizeof(descriptors), cudaMemcpyHostToDevice, stream);
  if (copied == cudaSuccess) {
    kernel<<<grid, Schedule::kThreads, kSharedBytes, stream>>>(device, alpha, o.out, o.codes, o.scales, o.divisor);
  }
  const cudaError_t launched = copied != cudaSuccess ? copied : cudaGetLastError();
  CUDA_CHECK(cudaFreeAsync(device, stream));
  CUDA_CHECK(launched);
#else
  kernel<<<grid, Schedule::kThreads, kSharedBytes, stream>>>(descriptors, alpha, o.out, o.codes, o.scales,
                                                             o.divisor);
  CUDA_CHECK(cudaGetLastError());
#endif
}

// Quantizes x into `scratch` in the layout the W4A4 products on `stream` read, then the kernel
// reading that layout, whose NVFP4 epilogue writes it too. Returns the layout.
template <Store kStore>
Layout run(const ninfer::Tensor &x, const ninfer::Weight &gate_up, nd::Nvfp4W4a4Workspace scratch,
           const Outputs &o, cudaStream_t stream) {
  const std::int32_t tokens = x.ne[1];
  const Layout layout = nd::launch_nvfp4_w4a4_quantize_for_products(x, gate_up, scratch, stream);
  const float alpha = 1.0F / (gate_up.input_scale_divisor * gate_up.weight_scale_divisor);
  const nd::Nvfp4W4a4TmaDescriptors descriptors =
      make_descriptors(layout, scratch.codes, scratch.scales, static_cast<const std::uint8_t *>(gate_up.qdata),
                       static_cast<const std::uint8_t *>(gate_up.scales), tokens);
  if (layout == Layout::KTileMajor) {
    launch<GateUpKTileSchedule, kStore>(descriptors, tokens, alpha, o, stream);
  } else {
    launch<GateUpSchedule, kStore>(descriptors, tokens, alpha, o, stream);
  }
  return layout;
}

bool aligned16(const void *p) {
  return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & 15U) == 0;
}

// The checks every entry of the op makes on its input and its gate and up weight.
void check_input(const ninfer::Tensor &x, const ninfer::Weight &gate_up) {
  const std::int32_t tokens = x.ne[1];
  if (!takes(tokens)) {
    throw std::invalid_argument("prefill swiglu: " + std::to_string(tokens) +
                                " tokens is no width the W4A4 TMA route takes");
  }
  if (x.dtype != ninfer::DType::BF16 || x.ne[0] != Geometry::kInputRows || x.ne[2] != 1 || x.ne[3] != 1 ||
      !x.is_contiguous() || !aligned16(x.data)) {
    throw std::invalid_argument("prefill swiglu: x is [5120, T], contiguous BF16");
  }
  if (gate_up.qtype != ninfer::QType::NVFP4 || gate_up.n != Geometry::kOutputRows ||
      gate_up.k != Geometry::kInputRows) {
    throw std::invalid_argument("prefill swiglu: the gate and up weight is the 27B's NVFP4 [34816, 5120]");
  }
  nd::validate_nvfp4_weight(gate_up, "prefill swiglu");
}

void check_down(const ninfer::Weight &down) {
  if (down.qtype != ninfer::QType::NVFP4 || down.n != DownGeometry::kOutputRows ||
      down.k != DownGeometry::kInputRows) {
    throw std::invalid_argument("prefill swiglu: the down weight is the 27B's NVFP4 [5120, 17408]");
  }
  nd::validate_nvfp4_weight(down, "prefill swiglu: down");
}

} // namespace

Route route_for_load() {
  return ignis::named_switch(kSwitch, Route::Quant,
                             {{"split", Route::Split}, {"fused", Route::Fused}, {"quant", Route::Quant}});
}

void gate_up_swiglu(const ninfer::Tensor &x, const ninfer::Weight &gate_up, ninfer::Tensor &out,
                    ninfer::WorkspaceArena &workspace, cudaStream_t stream) {
  check_input(x, gate_up);
  const std::int32_t tokens = x.ne[1];
  if (out.dtype != ninfer::DType::BF16 || out.ne[0] != kIntermediate || out.ne[1] != tokens || out.ne[2] != 1 ||
      out.ne[3] != 1 || !out.is_contiguous() || !aligned16(out.data)) {
    throw std::invalid_argument("prefill swiglu: out is [17408, T], contiguous BF16");
  }
  auto scope = workspace.scope();
  const nd::Nvfp4W4a4Workspace scratch = nd::allocate_nvfp4_w4a4_workspace(workspace, tokens, Geometry::kInputRows);
  (void)run<Store::Bf16>(x, gate_up, scratch, Outputs{static_cast<__nv_bfloat16 *>(out.data)}, stream);
}

nd::Nvfp4W4a4ActivationLayout gate_up_swiglu_nvfp4(const ninfer::Tensor &x, const ninfer::Weight &gate_up,
                                                   const ninfer::Weight &down, nd::Nvfp4W4a4Workspace h,
                                                   ninfer::WorkspaceArena &workspace, cudaStream_t stream) {
  check_input(x, gate_up);
  check_down(down);
  if (!aligned16(h.codes) || !aligned16(h.scales)) {
    throw std::invalid_argument("prefill swiglu: the down product's input is 16-byte aligned");
  }
  auto scope = workspace.scope();
  const nd::Nvfp4W4a4Workspace scratch =
      nd::allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
  return run<Store::Nvfp4>(x, gate_up, scratch, Outputs{nullptr, h.codes, h.scales, down.input_scale_divisor},
                           stream);
}

void mlp(Route route, const ninfer::Tensor &x, const ninfer::Weight &gate_up,
         ninfer::ops::LinearPolicy gate_up_policy, const ninfer::Weight &down, ninfer::ops::LinearPolicy down_policy,
         ninfer::Tensor &fused, ninfer::Tensor &residual, ninfer::WorkspaceArena &workspace, cudaStream_t stream) {
  const std::int32_t tokens = x.ne[1];
  const bool quantizes = route == Route::Quant && gate_up_policy == ninfer::ops::LinearPolicy::AllowA4 &&
                         down_policy == ninfer::ops::LinearPolicy::AllowA4 && takes(tokens) &&
                         gate_up.qtype == ninfer::QType::NVFP4 && down.qtype == ninfer::QType::NVFP4 &&
                         down.n == DownGeometry::kOutputRows && down.k == DownGeometry::kInputRows;
  if (!quantizes) {
    linear_swiglu(route, x, gate_up, fused, gate_up_policy, workspace, stream);
    ninfer::ops::linear_add(fused, down, residual, down_policy, workspace, stream);
    return;
  }
  check_input(x, gate_up);
  check_down(down);
  if (residual.dtype != ninfer::DType::BF16 || residual.ne[0] != DownGeometry::kOutputRows ||
      residual.ne[1] != tokens || residual.ne[2] != 1 || residual.ne[3] != 1 || !residual.is_contiguous() ||
      !aligned16(residual.data)) {
    throw std::invalid_argument("prefill swiglu: the residual is [5120, T], contiguous BF16");
  }
  // The op's input scratch first, as gate_up_swiglu's, then the down product's input, which the
  // scratch the layer reserves for the two products' own quantizers holds.
  auto scope = workspace.scope();
  const nd::Nvfp4W4a4Workspace scratch = nd::allocate_nvfp4_w4a4_workspace(workspace, tokens, Geometry::kInputRows);
  const nd::Nvfp4W4a4Workspace h = nd::allocate_nvfp4_w4a4_workspace(workspace, tokens, DownGeometry::kInputRows);
  const Layout layout =
      run<Store::Nvfp4>(x, gate_up, scratch, Outputs{nullptr, h.codes, h.scales, down.input_scale_divisor}, stream);
  const float alpha = 1.0F / (down.input_scale_divisor * down.weight_scale_divisor);
  nd::launch_nvfp4_w4a4_tma_linear_add(nd::Nvfp4Problem::Residual17408, h.codes, h.scales, layout,
                                       static_cast<const std::uint8_t *>(down.qdata),
                                       static_cast<const std::uint8_t *>(down.scales),
                                       static_cast<__nv_bfloat16 *>(residual.data), tokens, alpha, stream);
}

} // namespace ignis::prefill_swiglu
