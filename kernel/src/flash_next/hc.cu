// ignis kernel leaf -- Flash-Next's hyper-connection mix and inject (spec
// flash-next/04, GitHub #302; OURS, ADR 0043). See hc.h for the math.

#include "hc.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace ignis::flash_next {

namespace {

// Rows a mix processes at once: its scratch is a few of these rows' streams,
// so a prefill chunk of any width needs at most this many rows of it.
constexpr int32_t kWaveRows = 1024;
constexpr int32_t kThreads = 256;

__device__ __forceinline__ float bf(float v) {
  return __bfloat162float(__float2bfloat16(v));
}

__device__ float block_sum(float v, float *shared) {
  for (int32_t offset = 16; offset > 0; offset /= 2) {
    v += __shfl_xor_sync(0xffffffffU, v, offset);
  }
  const int32_t warp = static_cast<int32_t>(threadIdx.x) / 32;
  const int32_t lane = static_cast<int32_t>(threadIdx.x) % 32;
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0F;
  for (int32_t w = 0; w < kThreads / 32; ++w) {
    total += shared[w];
  }
  __syncthreads();
  return total;
}

// normed[r] = grouped RMSNorm of hidden[r], one group per stream, (1 + w).
__global__ void hc_norm(const __nv_bfloat16 *__restrict__ hidden, const __nv_bfloat16 *__restrict__ w,
                        __nv_bfloat16 *__restrict__ normed, int32_t streams, int32_t width,
                        float eps) {
  __shared__ float partial[kThreads / 32];
  const std::int64_t row = blockIdx.x;
  for (int32_t s = 0; s < streams; ++s) {
    const std::int64_t base = (row * streams + s) * width;
    float squares = 0.0F;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += kThreads) {
      const float v = __bfloat162float(hidden[base + i]);
      squares = fmaf(v, v, squares);
    }
    const float inv = rsqrtf(block_sum(squares, partial) / static_cast<float>(width) + eps);
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += kThreads) {
      const float v = __bfloat162float(hidden[base + i]) * inv;
      const float scale = 1.0F + __bfloat162float(w[s * width + i]);
      normed[base + i] = __float2bfloat16(v * scale);
    }
  }
}

// act = silu(bf16(down / streams)), each step rounded as the BF16 module does.
__global__ void hc_down_activation(const __nv_bfloat16 *__restrict__ down,
                                   __nv_bfloat16 *__restrict__ act, std::int64_t count,
                                   float streams) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float d = bf(__bfloat162float(down[i]) / streams);
    act[i] = __float2bfloat16(d / (1.0F + __expf(-d)));
  }
}

// x[r][h] = mean_s(bf16(sigmoid(up) * normed)), and, with block-inject
// outputs, inj[r][s] = bf16(2 * bf16(sigmoid(bf16(raw / streams)))).
__global__ void hc_reduce(const __nv_bfloat16 *__restrict__ up, const __nv_bfloat16 *__restrict__ normed,
                          const __nv_bfloat16 *__restrict__ inject_raw, __nv_bfloat16 *__restrict__ x,
                          float *__restrict__ inj, int32_t streams, int32_t width) {
  const std::int64_t row = blockIdx.x;
  for (int32_t h = static_cast<int32_t>(threadIdx.x); h < width; h += kThreads) {
    float sum = 0.0F;
    for (int32_t s = 0; s < streams; ++s) {
      const std::int64_t i = (row * streams + s) * width + h;
      const float m = bf(1.0F / (1.0F + __expf(-__bfloat162float(up[i]))));
      sum += bf(m * __bfloat162float(normed[i]));
    }
    x[row * width + h] = __float2bfloat16(sum / static_cast<float>(streams));
  }
  if (inject_raw != nullptr && static_cast<int32_t>(threadIdx.x) < streams) {
    const int32_t s = static_cast<int32_t>(threadIdx.x);
    const float scaled = bf(__bfloat162float(inject_raw[row * streams + s]) / static_cast<float>(streams));
    const float gate = bf(1.0F / (1.0F + __expf(-scaled)));
    inj[row * streams + s] = bf(2.0F * gate);
  }
}

// hidden[r][s][h] = bf16(hidden + bf16(y[r][h] * inj[r][s])).
__global__ void hc_inject_kernel(const __nv_bfloat16 *__restrict__ y, const float *__restrict__ inj,
                                 __nv_bfloat16 *__restrict__ hidden, int32_t streams, int32_t width,
                                 std::int64_t count) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::int64_t per_row = static_cast<std::int64_t>(streams) * width;
  const std::int64_t row = i / per_row;
  const int32_t s = static_cast<int32_t>((i % per_row) / width);
  const int32_t h = static_cast<int32_t>(i % width);
  const float injection = bf(__bfloat162float(y[row * width + h]) * inj[row * streams + s]);
  hidden[i] = __float2bfloat16(__bfloat162float(hidden[i]) + injection);
}

std::size_t aligned(std::size_t bytes) {
  return (bytes + 255) / 256 * 256;
}

bool launched(const char *what) {
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    fn_set_error(std::string(what) + ": launch failed: " + cudaGetErrorString(err));
    return false;
  }
  return true;
}

}  // namespace

std::size_t fn_hc_mix_scratch_bytes(const Geometry &g, int32_t rows) {
  const auto wave = static_cast<std::size_t>(std::min(rows, kWaveRows));
  const auto width = static_cast<std::size_t>(g.residual_width());
  const auto rank = static_cast<std::size_t>(g.hc_rank);
  const auto streams = static_cast<std::size_t>(g.streams);
  // normed, up: [wave][streams * hidden]; down, act: [wave][rank]; the
  // block-inject outputs: [wave][streams]. All BF16.
  return 2 * aligned(wave * width * 2) + 2 * aligned(wave * rank * 2) + aligned(wave * streams * 2);
}

int32_t fn_hc_mix(const Geometry &g, const HcWeights &w, const void *hidden, int32_t rows, void *x,
                  float *inj, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  if (hidden == nullptr || x == nullptr || w.hc_norm == nullptr || rows <= 0) {
    fn_set_error("fn_hc_mix: null operand or no rows");
    return -1;
  }
  const bool with_inject = w.block_inject != nullptr && inj != nullptr;
  const int32_t width = g.residual_width();
  auto scope = scratch.scope();
  const int32_t wave = std::min(rows, kWaveRows);
  auto *normed = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * width * 2).data);
  auto *up = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * width * 2).data);
  auto *down = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.hc_rank * 2).data);
  auto *act = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.hc_rank * 2).data);
  auto *inject_raw = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(static_cast<std::size_t>(wave) * g.streams * 2).data);
  const Linear block_inject{w.block_inject, g.streams, width, WeightFormat::Bf16};

  for (int32_t first = 0; first < rows; first += wave) {
    const int32_t n = std::min(wave, rows - first);
    const auto *in = static_cast<const __nv_bfloat16 *>(hidden) + static_cast<std::int64_t>(first) * width;
    hc_norm<<<n, kThreads, 0, stream>>>(in, static_cast<const __nv_bfloat16 *>(w.hc_norm), normed,
                                         g.streams, g.hidden, g.rms_norm_eps);
    if (!launched("fn_hc_mix: norm")) {
      return -1;
    }
    if (fn_linear(w.mix_down, normed, n, down, false, scratch, stream) != 0) {
      return -1;
    }
    const std::int64_t down_count = static_cast<std::int64_t>(n) * g.hc_rank;
    hc_down_activation<<<static_cast<uint32_t>((down_count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        down, act, down_count, static_cast<float>(g.streams));
    if (!launched("fn_hc_mix: activation")) {
      return -1;
    }
    if (fn_linear(w.mix_up, act, n, up, false, scratch, stream) != 0) {
      return -1;
    }
    if (with_inject && fn_linear(block_inject, normed, n, inject_raw, false, scratch, stream) != 0) {
      return -1;
    }
    hc_reduce<<<n, kThreads, 0, stream>>>(up, normed, with_inject ? inject_raw : nullptr,
                                           static_cast<__nv_bfloat16 *>(x) + static_cast<std::int64_t>(first) * g.hidden,
                                           with_inject ? inj + static_cast<std::int64_t>(first) * g.streams : nullptr,
                                           g.streams, g.hidden);
    if (!launched("fn_hc_mix: reduce")) {
      return -1;
    }
  }
  return 0;
}

int32_t fn_hc_inject(const Geometry &g, const void *y, const float *inj, int32_t rows, void *hidden,
                     cudaStream_t stream) {
  if (y == nullptr || inj == nullptr || hidden == nullptr || rows <= 0) {
    fn_set_error("fn_hc_inject: null operand or no rows");
    return -1;
  }
  const std::int64_t count = static_cast<std::int64_t>(rows) * g.residual_width();
  hc_inject_kernel<<<static_cast<uint32_t>((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(y), inj, static_cast<__nv_bfloat16 *>(hidden), g.streams,
      g.hidden, count);
  return launched("fn_hc_inject") ? 0 : -1;
}

}  // namespace ignis::flash_next
