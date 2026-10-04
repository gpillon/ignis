// ignis kernel leaf: the MoE ops' error channel, record sizes and the trellis decode seam -- OURS
// (kernel/include/ignis_moe.h).

#include "moe_common.cuh"
#include "trellis_decode.cuh"

#include <cuda_runtime.h>

#include <string>

namespace ignis_moe {

namespace {
thread_local std::string g_last_error = "no error";
}

const char *last_error_cstr() { return g_last_error.c_str(); }

void set_error(const std::string &message) { g_last_error = message; }

int32_t fail(const std::string &message) {
  set_error(message);
  return -1;
}

int32_t check_launch(const char *op) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return fail(std::string(op) + ": launch failed: " + cudaGetErrorString(err));
  }
  return 0;
}

namespace {

// One warp per tile: each lane reads its two words, decodes its eight weights and stores them
// at their (row, column) -- the expert ops' own decoder, writing instead of multiplying.
__global__ void trellis_reconstruct_kernel(const uint32_t *trellis, int k2, int tiles_k,
                                           int tiles_n, __half *w) {
  const int lane = threadIdx.x & 31;
  const int warps = (blockDim.x >> 5) * gridDim.x;
  const int words = ignis_trellis::tile_words(k2);
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(k2, lane);
  const int out = tiles_n * 16;
  const int r0 = (lane % 4) * 2;
  const int c0 = lane / 4;
  for (int tile = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); tile < tiles_k * tiles_n;
       tile += warps) {
    const uint32_t *t = trellis + static_cast<size_t>(tile) * words;
    uint32_t frag[4];
    ignis_trellis::decode_fragment(t[plan.w0], t[plan.w1], plan, frag);
    const int tk = tile / tiles_n;
    const int tn = tile % tiles_n;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = tk * 16 + r0 + ((i & 1) ? 8 : 0);
      const int c = tn * 16 + c0 + ((i & 2) ? 8 : 0);
      const __half2 v = *reinterpret_cast<const __half2 *>(&frag[i]);
      w[static_cast<size_t>(r) * out + c] = __low2half(v);
      w[static_cast<size_t>(r + 1) * out + c] = __high2half(v);
    }
  }
}

}  // namespace
}  // namespace ignis_moe

using namespace ignis_moe;

extern "C" int32_t ignis_moe_record_bytes(uint32_t projection, uint32_t k2, uint64_t *bytes) {
  if (bytes == nullptr) return fail("ignis_moe_record_bytes: bytes is NULL");
  if (k2 != 4 && k2 != 5 && k2 != 6 && k2 != 8) {
    return fail("ignis_moe_record_bytes: k2 must be 4, 5, 6 or 8");
  }
  uint64_t in = 0, out = 0;
  if (projection == IGNIS_MOE_PROJ_GATE_UP) {
    in = kHidden;
    out = kGateUpOut;
  } else if (projection == IGNIS_MOE_PROJ_DOWN) {
    in = kInter;
    out = kHidden;
  } else {
    return fail("ignis_moe_record_bytes: unknown projection");
  }
  const uint64_t data = trellis_bytes(static_cast<uint32_t>(in), static_cast<uint32_t>(out), k2) + 2 * in + 2 * out;
  *bytes = (data + 4095) / 4096 * 4096;
  return 0;
}

extern "C" int32_t ignis_moe_trellis_reconstruct(const void *trellis, uint32_t k2, uint32_t in,
                                                 uint32_t out, void *w_f16, void *stream) {
  if (trellis == nullptr || w_f16 == nullptr) {
    return fail("ignis_moe_trellis_reconstruct: NULL pointer");
  }
  if (k2 != 4 && k2 != 5 && k2 != 6 && k2 != 8) {
    return fail("ignis_moe_trellis_reconstruct: k2 must be 4, 5, 6 or 8");
  }
  if (in == 0 || out == 0 || in % 16 != 0 || out % 16 != 0) {
    return fail("ignis_moe_trellis_reconstruct: in and out must be positive multiples of 16");
  }
  if ((reinterpret_cast<uintptr_t>(trellis) & 3) != 0) {
    return fail("ignis_moe_trellis_reconstruct: trellis must be 4-byte aligned");
  }
  const int tiles = static_cast<int>(in / 16 * (out / 16));
  const int warps_per_block = 8;
  const int blocks = (tiles + warps_per_block - 1) / warps_per_block;
  trellis_reconstruct_kernel<<<blocks < 4096 ? blocks : 4096, warps_per_block * 32, 0,
                               static_cast<cudaStream_t>(stream)>>>(
      static_cast<const uint32_t *>(trellis), static_cast<int>(k2), static_cast<int>(in / 16),
      static_cast<int>(out / 16), static_cast<__half *>(w_f16));
  return check_launch("ignis_moe_trellis_reconstruct");
}

extern "C" const char *ignis_moe_last_error(void) { return last_error_cstr(); }
