// ignis kernel leaf: the image model's FP8 W8A8 GEMM (spec image-model/02 acceptance 2, GitHub
// #319) -- OURS (kernel/include/ignis_image_gemm.h; ADR 0051 tier 1). The kernel is
// fp8_gemm_kernel.cuh's; this unit instantiates the tile configurations the C ABI runs: the default
// one and (GitHub #693) the 16-token one of the *_tile16 calls. The other configurations exist for
// the bench's sweep and are built into the test and the bench, never into the leaf
// (kernel/tests/image_gemm_sweep.cu).

#include "fp8_gemm_kernel.cuh"
#include "ignis_image_gemm.h"

#include <atomic>

static_assert(IGNIS_FP8_MAX_SEGMENTS == ignis_image::kMaxSegments, "the grouped GEMM takes ignis_fp8_segment's count");

namespace ignis_image {
namespace {

// GitHub #693 (spec #684): the tile of a call of at most 16 tokens -- the 27B's FP8 exception
// matrices at 9..16 tokens (kernel/src/fp8_exception.h). The default tile holds 128 tokens: such a
// call fills an eighth of it and launches 56 CTAs for the q | k | gate | v rows and 20 for an
// output projection, each running the whole tile's MMAs along K. 16 tokens x 64 output rows, two
// warps of 16 x 32, launches 224 and 80. An output takes the same k32 steps in the same order (BK
// 64 in both), at the same place in its m16n8k32 fragment, and the same epilogue, so the same bits.
using GemmTile16Config = TileConfig<16, 64, 64, 4, 1, 2, 1>;
static_assert(GemmTile16Config::BM == IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS, "the tile holds the calls' tokens");
static_assert(GemmTile16Config::BK == GemmDefaultConfig::BK, "the k32 steps of the default tile, in its order");
std::atomic<uint64_t> g_tile16_launches{0};

constexpr int kMaxDevices = 64;
std::atomic<bool> g_prepared[kMaxDevices];

int32_t require_prepared(const char *op) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices || !g_prepared[device].load()) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_image_gemm_prepare at load");
  }
  return 0;
}

// The grouped GEMM's checks and launch on configuration C.
template <class C>
int32_t grouped(const char *op, const ignis_fp8_segment *segments, uint32_t count, uint32_t cols, const void *codes,
                const float *scales, uint32_t tokens, int y_mode, void *stream) {
  if (segments == nullptr) return fail(std::string(op) + ": NULL pointer");
  if (count == 0 || count > static_cast<uint32_t>(kMaxSegments)) {
    return fail(std::string(op) + ": 1.." + std::to_string(kMaxSegments) + " segments");
  }
  if (require_prepared(op) != 0) return -1;
  const void *weights[kMaxSegments];
  uint32_t rows[kMaxSegments];
  void *ys[kMaxSegments];
  for (uint32_t i = 0; i < count; ++i) {
    weights[i] = segments[i].weight;
    rows[i] = segments[i].rows;
    ys[i] = segments[i].y;
  }
  return gemm_w8a8_launch_segments<C>(op, weights, rows, ys, count, cols, codes, scales, tokens, y_mode, stream);
}

// The 16-token tile's refusal of a wider call, before anything is launched.
int32_t tile16_tokens(const char *op, uint32_t tokens) {
  if (tokens > IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS) {
    return fail(std::string(op) + ": the 16-token tile takes at most " +
                std::to_string(IGNIS_IMAGE_GEMM_TILE16_MAX_TOKENS) + " tokens");
  }
  return 0;
}

// `rc`, with a launch of the 16-token tile counted where it is 0.
int32_t tile16_counted(int32_t rc) {
  if (rc == 0) g_tile16_launches.fetch_add(1, std::memory_order_relaxed);
  return rc;
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_gemm_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = gemm_w8a8_set_attributes<GemmDefaultConfig>();
  if (err == cudaSuccess) err = gemm_w8a8_set_attributes<GemmTile16Config>();
  if (err != cudaSuccess) return fail(std::string("ignis_image_gemm_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_image_gemm_prepare: device id out of range");
  g_prepared[device].store(true);
  return 0;
}

extern "C" const char *ignis_image_gemm_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_gemm_w8a8(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  if (require_prepared("ignis_image_gemm_w8a8") != 0) return -1;
  return gemm_w8a8_launch<GemmDefaultConfig>(weight, rows, cols, codes, scales, tokens, y, y_f32, stream);
}

extern "C" int32_t ignis_image_gemm_w8a8_grouped(const ignis_fp8_segment *segments, uint32_t count, uint32_t cols,
                                                 const void *codes, const float *scales, uint32_t tokens,
                                                 uint32_t y_f32, void *stream) {
  return grouped<GemmDefaultConfig>("ignis_image_gemm_w8a8_grouped", segments, count, cols, codes, scales, tokens,
                                    y_f32 != 0 ? kYF32 : kYBf16, stream);
}

extern "C" int32_t ignis_image_gemm_w8a8_add(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                             const float *scales, uint32_t tokens, void *residual, void *stream) {
  const char *op = "ignis_image_gemm_w8a8_add";
  if (require_prepared(op) != 0) return -1;
  return gemm_w8a8_launch_segments<GemmDefaultConfig>(op, &weight, &rows, &residual, 1, cols, codes, scales, tokens,
                                                      kYBf16Add, stream);
}

extern "C" int32_t ignis_image_gemm_w8a8_grouped_tile16(const ignis_fp8_segment *segments, uint32_t count,
                                                        uint32_t cols, const void *codes, const float *scales,
                                                        uint32_t tokens, uint32_t y_f32, void *stream) {
  const char *op = "ignis_image_gemm_w8a8_grouped_tile16";
  if (tile16_tokens(op, tokens) != 0) return -1;
  return tile16_counted(grouped<GemmTile16Config>(op, segments, count, cols, codes, scales, tokens,
                                                  y_f32 != 0 ? kYF32 : kYBf16, stream));
}

extern "C" int32_t ignis_image_gemm_w8a8_add_tile16(const void *weight, uint32_t rows, uint32_t cols,
                                                    const void *codes, const float *scales, uint32_t tokens,
                                                    void *residual, void *stream) {
  const char *op = "ignis_image_gemm_w8a8_add_tile16";
  if (tile16_tokens(op, tokens) != 0 || require_prepared(op) != 0) return -1;
  return tile16_counted(gemm_w8a8_launch_segments<GemmTile16Config>(op, &weight, &rows, &residual, 1, cols, codes,
                                                                    scales, tokens, kYBf16Add, stream));
}

extern "C" uint64_t ignis_image_gemm_tile16_launches(void) { return g_tile16_launches.load(std::memory_order_relaxed); }
