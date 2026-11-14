// ignis kernel leaf: the warp-specialized hq-e8-2b attention route's one engine instantiation and
// its launch -- OURS (spec #570, GitHub #572; the route is kernel/src/hq_verify_ws.cuh, #571).
//
// In the sm_120a set (kernel/CMakeLists.txt, IGNIS_KERNEL_SM120A_SRCS): the route declares more
// static shared memory per block than an sm_8x block takes and waits on mbarrier.try_wait, so this
// file compiles for the build's arch-specific Blackwell entries alone and a device none of them
// runs on finds no image of it (`image_here`), never the stand-in the header compiles below
// compute capability 12.0. Only the route's kernel is instantiated here: the vendored reducer that
// follows it is launched from kernel/src/hq_decode_attention.cu, built for every architecture of
// the list as the vendored launcher that also instantiates it is.
#include "hq_decode_attention.h"

#include "hq_verify_ws.cuh"

#include "core/pdl.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace ignis::hq_decode_attention {
namespace {

// The batch body appends the round's own rows (A1's GqaAppendInput), never reads a cache alone.
using Geometry = ninfer::ops::Gqa27Geometry;
using Input = ninfer::ops::GqaAppendInput;

}  // namespace

bool image_here() {
  cudaFuncAttributes attributes{};
  const cudaError_t err =
      cudaFuncGetAttributes(&attributes, ignis_hq_verify::hq_verify_ws_kernel<Geometry, Input>);
  if (err != cudaSuccess) {
    // No image for this device (cudaErrorNoKernelImageForDevice, cudaErrorInvalidDeviceFunction),
    // or no device: consumed here, so no later launch check reads it as its own.
    (void)cudaGetLastError();
    return false;
  }
  // The route's block holds its ring and its query rows in static shared memory; the stand-in an
  // older architecture compiles declares none.
  return attributes.sharedSizeBytes > 0;
}

void launch_partials(const PartialLaunch &launch, cudaStream_t stream) {
  const cudaError_t err = ninfer::pdl::launch_dependent(
      {launch.grid, dim3(ignis_hq_verify::kHqVerifyWsThreads), 0, stream},
      ignis_hq_verify::hq_verify_ws_kernel<Geometry, Input>, launch.q, Input{launch.k, launch.v},
      launch.positions,
      ninfer::ops::GqaTcKVHq{launch.codes_k, launch.codes_v, launch.meta_k, launch.meta_v,
                             launch.residual_k, launch.residual_v, launch.ring_valid},
      launch.block_tables, launch.valid_columns, launch.table_rows, launch.table_stride,
      launch.tokens, launch.full_width, launch.column_begin, launch.logical_capacity, launch.scale,
      launch.partial_acc, launch.partial_m, launch.partial_l);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    throw std::runtime_error(std::string("hq_decode_attention: the route's launch: ") +
                             cudaGetErrorString(err));
  }
}

} // namespace ignis::hq_decode_attention
