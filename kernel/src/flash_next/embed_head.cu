// ignis kernel leaf -- Flash-Next's token embedding and output head (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). See embed_head.h.

#include "embed_head.h"

#include "hc.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <string>

namespace ignis::flash_next {

namespace {

constexpr int32_t kThreads = 256;

// One block per token: its embedding row, decoded once, written to every
// stream. An id outside [0, vocab) traps rather than reading another row.
__global__ void embed_rows(const uint8_t *__restrict__ weight, bool fp8, int64_t scales_at,
                           const int32_t *__restrict__ ids, int32_t vocab, int32_t hidden,
                           int32_t streams, __nv_bfloat16 *__restrict__ residual) {
  const int64_t row = blockIdx.x;
  const int32_t id = ids[row];
  if (id < 0 || id >= vocab) {
    __trap();
  }
  float scale = 1.0F;
  if (fp8) {
    scale = __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(weight + scales_at)[id]);
  }
  for (int32_t h = static_cast<int32_t>(threadIdx.x); h < hidden; h += kThreads) {
    __nv_bfloat16 value;
    if (fp8) {
      const auto code = weight[static_cast<int64_t>(id) * hidden + h];
      __nv_fp8_e4m3 e;
      e.__x = code;
      value = __float2bfloat16(static_cast<float>(e) * scale);
    } else {
      value = reinterpret_cast<const __nv_bfloat16 *>(weight)[static_cast<int64_t>(id) * hidden + h];
    }
    for (int32_t s = 0; s < streams; ++s) {
      residual[(row * streams + s) * hidden + h] = value;
    }
  }
}

}  // namespace

int32_t fn_embed(const Geometry &g, const Linear &embed, const int32_t *ids, int32_t rows,
                 void *residual, cudaStream_t stream) {
  if (embed.data == nullptr || ids == nullptr || residual == nullptr || rows <= 0) {
    fn_set_error("fn_embed: null operand or no rows");
    return -1;
  }
  if (embed.rows != g.vocab || embed.cols != g.hidden) {
    fn_set_error("fn_embed: embed_tokens is [" + std::to_string(embed.rows) + "," +
                 std::to_string(embed.cols) + "], not [vocab, hidden]");
    return -1;
  }
  const bool fp8 = embed.format == WeightFormat::Fp8RowScale;
  // row-scale-v1: codes [rows][cols], zero padding to a multiple of 256, BF16 scales [rows].
  const int64_t codes = static_cast<int64_t>(embed.rows) * embed.cols;
  const int64_t scales_at = (codes + 255) / 256 * 256;
  embed_rows<<<rows, kThreads, 0, stream>>>(static_cast<const uint8_t *>(embed.data), fp8, scales_at,
                                            ids, g.vocab, g.hidden, g.streams,
                                            static_cast<__nv_bfloat16 *>(residual));
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    fn_set_error(std::string("fn_embed: launch failed: ") + cudaGetErrorString(err));
    return -1;
  }
  return 0;
}

std::size_t fn_head_scratch_bytes(const Geometry &g, int32_t rows) {
  const std::size_t x = (static_cast<std::size_t>(rows) * g.hidden * 2 + 255) / 256 * 256;
  return x + fn_hc_mix_scratch_bytes(g, rows);
}

int32_t fn_head(const Geometry &g, const HcWeights &final_mixer, const Linear &head,
                const void *residual, int32_t rows, void *logits, ninfer::DeviceArena &scratch,
                cudaStream_t stream) {
  auto scope = scratch.scope();
  void *x = scratch.alloc_bytes(static_cast<std::size_t>(rows) * g.hidden * 2).data;
  if (fn_hc_mix(g, final_mixer, residual, rows, x, nullptr, scratch, stream) != 0) {
    return -1;
  }
  return fn_linear(head, x, rows, logits, /*y_f32=*/false, scratch, stream);
}

}  // namespace ignis::flash_next
