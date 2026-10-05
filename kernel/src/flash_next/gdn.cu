// ignis kernel leaf -- the Flash-Next GDN layer core (spec flash-next/04, GitHub #302, slice S2):
// OURS (ADR 0043), no port claim. See gdn.h for the oracle and the call shapes.
//
// Kernels here: the gating (g, beta from the BF16 projections and the layer's BF16 A_log and
// dt_bias), the sigmoid-gated norm, and the copy of one slot's state between the pool and a
// scratch image. Everything else is fn_linear and the vendored convolution and recurrence.
//
// Numerics: every BF16 rounding the checkpoint makes is made here at the same point (the four
// projections, beta = b.sigmoid() in BF16, the gated norm's three roundings); g and the norm's
// statistics are fp32 like the checkpoint's. The convolution rounds conv + SiLU once and the
// recurrence keeps an fp32 state, where the checkpoint rounds the convolution and the SiLU
// separately and runs its chunked rule in fp32: those two ops are the vendored ones, qualified
// against fp64 by their own contracts.

#include "gdn.h"

#include "ignis_seq_internal.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"

#include <cuda_bf16.h>

#include <cmath>
#include <exception>
#include <string>

namespace ignis::flash_next {

namespace gdn {

namespace {

constexpr int32_t kThreads = 256;

// g = -exp(A_log) * softplus(a + dt_bias) in fp32 (torch's softplus: x past 20 is x itself) and
// beta = sigmoid(b) rounded to BF16, as the checkpoint's b.sigmoid() is; [rows][kValueHeads] each.
__global__ void gating_kernel(const __nv_bfloat16 *__restrict__ a, const __nv_bfloat16 *__restrict__ b,
                              const __nv_bfloat16 *__restrict__ a_log,
                              const __nv_bfloat16 *__restrict__ dt_bias, float *__restrict__ g,
                              float *__restrict__ beta, int32_t n) {
  const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  const int32_t h = i % kValueHeads;
  const float x = __bfloat162float(a[i]) + __bfloat162float(dt_bias[h]);
  const float softplus = x > 20.0F ? x : log1pf(expf(x));
  g[i] = -expf(__bfloat162float(a_log[h])) * softplus;
  const float sig = 1.0F / (1.0F + expf(-__bfloat162float(b[i])));
  beta[i] = __bfloat162float(__float2bfloat16(sig));
}

// The checkpoint's Qwen4ExpTextRMSNormGated with a sigmoid gate, one warp per (token, value
// head): n = bf16(o * rsqrt(mean(o^2) + eps)), m = bf16(weight * n), out = bf16(m * sigmoid(z)).
__global__ void gated_norm_kernel(const __nv_bfloat16 *__restrict__ o, const __nv_bfloat16 *__restrict__ z,
                                  const __nv_bfloat16 *__restrict__ weight, float eps,
                                  __nv_bfloat16 *__restrict__ out, int32_t units) {
  const int32_t unit = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) / 32);
  const int32_t lane = static_cast<int32_t>(threadIdx.x % 32);
  if (unit >= units) return;
  const int64_t at = static_cast<int64_t>(unit) * kHeadDim + lane * 4;
  const uint2 ov = *reinterpret_cast<const uint2 *>(o + at);
  const uint2 zv = *reinterpret_cast<const uint2 *>(z + at);
  const uint2 wv = *reinterpret_cast<const uint2 *>(weight + lane * 4);
  const auto *oh = reinterpret_cast<const __nv_bfloat16 *>(&ov);
  const auto *zh = reinterpret_cast<const __nv_bfloat16 *>(&zv);
  const auto *wh = reinterpret_cast<const __nv_bfloat16 *>(&wv);
  float x[4];
  float sum = 0.0F;
  for (int32_t i = 0; i < 4; ++i) {
    x[i] = __bfloat162float(oh[i]);
    sum = fmaf(x[i], x[i], sum);
  }
  for (int32_t offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffffU, sum, offset);
  const float r = rsqrtf(sum / static_cast<float>(kHeadDim) + eps);
  uint2 result;
  auto *rh = reinterpret_cast<__nv_bfloat16 *>(&result);
  for (int32_t i = 0; i < 4; ++i) {
    const float n = __bfloat162float(__float2bfloat16(x[i] * r));
    const float m = __bfloat162float(__float2bfloat16(__bfloat162float(wh[i]) * n));
    const float gate = 1.0F / (1.0F + expf(-__bfloat162float(zh[i])));
    rh[i] = __float2bfloat16(m * gate);
  }
  *reinterpret_cast<uint2 *>(out + at) = result;
}

// One slot's state between the pool (slots of `vectors` 16-byte vectors each) and a scratch
// image, the slot read from the device; a slot outside the pool traps rather than touch another
// lane's state.
__global__ void slot_copy_kernel(uint4 *__restrict__ pool, uint4 *__restrict__ image, int64_t vectors,
                                 const int32_t *__restrict__ slots, int32_t slot_count, bool to_image) {
  const int32_t slot = slots[0];
  if (slot < 0 || slot >= slot_count) __trap();
  uint4 *at = pool + static_cast<int64_t>(slot) * vectors;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < vectors;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (to_image) {
      image[i] = at[i];
    } else {
      at[i] = image[i];
    }
  }
}

constexpr std::size_t kConvSlotBytes =
    static_cast<std::size_t>(kConvStateTaps) * kConvChannels * sizeof(__nv_bfloat16);
constexpr std::size_t kRecurrentSlotBytes =
    static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim * sizeof(float);

cudaError_t slot_copy(void *pool, void *image, std::size_t bytes, const int32_t *slots, int32_t slot_count,
                      bool to_image, cudaStream_t stream) {
  const auto vectors = static_cast<int64_t>(bytes / sizeof(uint4));
  const auto blocks = static_cast<unsigned>(std::min<int64_t>((vectors + kThreads - 1) / kThreads, 256));
  slot_copy_kernel<<<blocks, kThreads, 0, stream>>>(static_cast<uint4 *>(pool), static_cast<uint4 *>(image),
                                                    vectors, slots, slot_count, to_image);
  return cudaGetLastError();
}

bool shaped(const Linear &l, int32_t rows, int32_t cols) {
  return l.data != nullptr && l.rows == rows && l.cols == cols;
}

// Bytes of one aligned arena allocation, as DeviceArena::alloc_bytes takes them.
std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The call's activation buffers, in allocation order.
std::size_t activation_bytes(int32_t rows) {
  const auto r = static_cast<std::size_t>(rows);
  return 2 * aligned(r * kConvChannels * sizeof(__nv_bfloat16)) +
         aligned(r * kValueWidth * sizeof(__nv_bfloat16)) +
         2 * aligned(r * kValueHeads * sizeof(__nv_bfloat16)) + 2 * aligned(r * kValueHeads * sizeof(float));
}

}  // namespace

const char *check_geometry(const Geometry &g) {
  if (g.gdn_qk_heads != kQkHeads || g.gdn_value_heads != kValueHeads || g.gdn_head_dim != kHeadDim ||
      g.gdn_conv_kernel != kConvKernel) {
    return "gdn: the layer is written for 16 q/k heads, 48 value heads of 128 and a conv kernel of 4";
  }
  if (g.hidden <= 0 || g.hidden % 64 != 0) return "gdn: hidden must be a positive multiple of 64";
  return nullptr;
}

int32_t run(const Geometry &g, const State &state, const GdnWeights &w, const Batch &batch,
            const void *x, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  if (const char *why = check_geometry(g)) {
    fn_set_error(why);
    return -1;
  }
  const int32_t rows = batch.rows();
  const bool per_lane = batch.tokens == 1;
  if (rows <= 0 || batch.slots == nullptr) {
    fn_set_error("gdn: an empty batch, or no slots");
    return -1;
  }
  if (per_lane ? batch.lanes > kMaxLanes : batch.lanes != 1) {
    fn_set_error("gdn: a call is 1.." + std::to_string(kMaxLanes) +
                 " lanes of one token, or one lane of several (got " + std::to_string(batch.lanes) + " x " +
                 std::to_string(batch.tokens) + ")");
    return -1;
  }
  if (x == nullptr || y == nullptr || state.conv == nullptr || state.recurrent == nullptr || state.slots <= 0) {
    fn_set_error("gdn: null activations or state");
    return -1;
  }
  if (!shaped(w.in_proj_qkv, kConvChannels, g.hidden) || !shaped(w.in_proj_z, kValueWidth, g.hidden) ||
      !shaped(w.in_proj_a, kValueHeads, g.hidden) || !shaped(w.in_proj_b, kValueHeads, g.hidden) ||
      !shaped(w.out_proj, g.hidden, kValueWidth) || w.conv == nullptr || w.a_log == nullptr ||
      w.dt_bias == nullptr || w.norm == nullptr) {
    fn_set_error("gdn: a weight is missing or has the wrong shape");
    return -1;
  }

  try {
    auto scope = scratch.scope();
    const auto r = static_cast<std::size_t>(rows);
    // wide0: the projected qkv, then q | k | v split from the convolution, then the gated norm's
    // output; wide1: the convolution's output, then the recurrence's.
    auto *wide0 = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kConvChannels * 2).data);
    auto *wide1 = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kConvChannels * 2).data);
    auto *z = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueWidth * 2).data);
    auto *a = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueHeads * 2).data);
    auto *b = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueHeads * 2).data);
    auto *gf = static_cast<float *>(scratch.alloc_bytes(r * kValueHeads * 4).data);
    auto *beta = static_cast<float *>(scratch.alloc_bytes(r * kValueHeads * 4).data);

    if (fn_linear(w.in_proj_qkv, x, rows, wide0, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_z, x, rows, z, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_a, x, rows, a, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_b, x, rows, b, false, scratch, stream) != 0) {
      return -1;
    }
    const int32_t gates = rows * kValueHeads;
    gating_kernel<<<(gates + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        a, b, static_cast<const __nv_bfloat16 *>(w.a_log), static_cast<const __nv_bfloat16 *>(w.dt_bias), gf,
        beta, gates);

    const ninfer::Tensor conv_weight(const_cast<void *>(w.conv), ninfer::DType::BF16,
                                     {kConvChannels, kConvKernel, 1, 1});
    const ninfer::Tensor lane_slots(const_cast<int32_t *>(batch.slots), ninfer::DType::I32,
                                    {batch.lanes, 1, 1, 1});
    void *conv_image = nullptr;
    void *recurrent_image = nullptr;
    if (per_lane) {
      ninfer::Tensor conv_states(state.conv, ninfer::DType::BF16, {kConvChannels, kConvStateTaps, state.slots, 1});
      ninfer::Tensor out(wide1, ninfer::DType::BF16, {kConvChannels, 1, rows, 1});
      ninfer::ops::causal_conv1d_silu_snapshot(
          ninfer::Tensor(wide0, ninfer::DType::BF16, {kConvChannels, 1, rows, 1}), conv_weight, conv_states,
          ninfer::Tensor{}, lane_slots, lane_slots, out, stream);
    } else {
      conv_image = scratch.alloc_bytes(kConvSlotBytes).data;
      recurrent_image = scratch.alloc_bytes(kRecurrentSlotBytes).data;
      cudaError_t err = slot_copy(state.conv, conv_image, kConvSlotBytes, batch.slots, state.slots, true, stream);
      if (err == cudaSuccess) {
        err = slot_copy(state.recurrent, recurrent_image, kRecurrentSlotBytes, batch.slots, state.slots, true,
                        stream);
      }
      if (err != cudaSuccess) {
        fn_set_error(std::string("gdn: state gather failed: ") + cudaGetErrorString(err));
        return -1;
      }
      ninfer::Tensor conv_state(conv_image, ninfer::DType::BF16, {kConvChannels, kConvStateTaps});
      ninfer::Tensor out(wide1, ninfer::DType::BF16, {kConvChannels, rows, 1, 1});
      ninfer::ops::causal_conv1d_silu(ninfer::Tensor(wide0, ninfer::DType::BF16, {kConvChannels, rows, 1, 1}),
                                      conv_weight, conv_state, out, stream);
    }

    // q | k | v out of the convolution's [rows][10240], each contiguous for the recurrence.
    auto *q = wide0;
    auto *k = q + r * kKeyWidth;
    auto *v = k + r * kKeyWidth;
    const std::size_t pitch = static_cast<std::size_t>(kConvChannels) * 2;
    cudaError_t err = cudaMemcpy2DAsync(q, kKeyWidth * 2, wide1, pitch, kKeyWidth * 2, r,
                                        cudaMemcpyDeviceToDevice, stream);
    if (err == cudaSuccess) {
      err = cudaMemcpy2DAsync(k, kKeyWidth * 2, wide1 + kKeyWidth, pitch, kKeyWidth * 2, r,
                              cudaMemcpyDeviceToDevice, stream);
    }
    if (err == cudaSuccess) {
      err = cudaMemcpy2DAsync(v, kValueWidth * 2, wide1 + 2 * kKeyWidth, pitch, kValueWidth * 2, r,
                              cudaMemcpyDeviceToDevice, stream);
    }
    if (err != cudaSuccess) {
      fn_set_error(std::string("gdn: the q/k/v split failed: ") + cudaGetErrorString(err));
      return -1;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    if (per_lane) {
      ninfer::Tensor states(state.recurrent, ninfer::DType::FP32, {kHeadDim, kHeadDim, kValueHeads, state.slots});
      ninfer::Tensor out(wide1, ninfer::DType::BF16, {kHeadDim, kValueHeads, 1, rows});
      ninfer::ops::gated_delta_net_snapshot(
          ninfer::Tensor(q, ninfer::DType::BF16, {kHeadDim, kQkHeads, 1, rows}),
          ninfer::Tensor(k, ninfer::DType::BF16, {kHeadDim, kQkHeads, 1, rows}),
          ninfer::Tensor(v, ninfer::DType::BF16, {kHeadDim, kValueHeads, 1, rows}),
          ninfer::Tensor(gf, ninfer::DType::FP32, {kValueHeads, 1, rows, 1}),
          ninfer::Tensor(beta, ninfer::DType::FP32, {kValueHeads, 1, rows, 1}), scale, /*normalize_qk=*/true,
          states, ninfer::Tensor{}, lane_slots, lane_slots, out, stream);
    } else {
      ninfer::Tensor state_image(recurrent_image, ninfer::DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
      ninfer::Tensor out(wide1, ninfer::DType::BF16, {kHeadDim, kValueHeads, rows, 1});
      ninfer::ops::gated_delta_net(
          ninfer::Tensor(q, ninfer::DType::BF16, {kHeadDim, kQkHeads, rows, 1}),
          ninfer::Tensor(k, ninfer::DType::BF16, {kHeadDim, kQkHeads, rows, 1}),
          ninfer::Tensor(v, ninfer::DType::BF16, {kHeadDim, kValueHeads, rows, 1}),
          ninfer::Tensor(gf, ninfer::DType::FP32, {kValueHeads, rows, 1, 1}),
          ninfer::Tensor(beta, ninfer::DType::FP32, {kValueHeads, rows, 1, 1}), scale, /*normalize_qk=*/true,
          scratch, state_image, state_image, out, stream);
      err = slot_copy(state.conv, conv_image, kConvSlotBytes, batch.slots, state.slots, false, stream);
      if (err == cudaSuccess) {
        err = slot_copy(state.recurrent, recurrent_image, kRecurrentSlotBytes, batch.slots, state.slots, false,
                        stream);
      }
      if (err != cudaSuccess) {
        fn_set_error(std::string("gdn: state scatter failed: ") + cudaGetErrorString(err));
        return -1;
      }
    }

    const int32_t units = rows * kValueHeads;
    gated_norm_kernel<<<(units * 32 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        wide1, z, static_cast<const __nv_bfloat16 *>(w.norm), g.rms_norm_eps, wide0, units);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fn_set_error(std::string("gdn: launch failed: ") + cudaGetErrorString(err));
      return -1;
    }
    return fn_linear(w.out_proj, wide0, rows, y, false, scratch, stream);
  } catch (const std::exception &e) {
    fn_set_error(std::string("gdn: ") + e.what());
    return -1;
  }
}

}  // namespace gdn

int32_t fn_gdn_layer(const Context &ctx, int32_t gdn_ordinal, const GdnWeights &w, const Batch &batch,
                     const void *x, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream) {
  if (ctx.pool == nullptr) {
    fn_set_error("fn_gdn_layer: no seq pool");
    return -1;
  }
  gdn::State state;
  try {
    const auto layer = static_cast<std::uint32_t>(gdn_ordinal);
    state.conv = ctx.pool->gdn_pool.conv_slot(layer, 0).data;
    state.recurrent = static_cast<float *>(ctx.pool->gdn_pool.recurrent_slot(layer, 0).data);
    state.slots = ctx.pool->gdn_pool.slot_count();
  } catch (const std::exception &e) {
    fn_set_error(std::string("fn_gdn_layer: GDN layer ") + std::to_string(gdn_ordinal) + ": " + e.what());
    return -1;
  }
  return gdn::run(ctx.g, state, w, batch, x, y, scratch, stream);
}

std::size_t fn_gdn_layer_scratch_bytes(const Geometry &g, int32_t rows) {
  (void)g;
  if (rows <= 0) return 0;
  std::size_t bytes = gdn::activation_bytes(rows);
  if (rows > 1) {
    // A prefill call's state images and the chunked recurrence's workspace (none at one token).
    bytes += gdn::aligned(gdn::kConvSlotBytes) + gdn::aligned(gdn::kRecurrentSlotBytes) +
             ninfer::ops::gated_delta_net_workspace_capacity_bytes(gdn::kQkHeads, gdn::kValueHeads, true, 1, rows) +
             256;
  }
  return bytes;
}

}  // namespace ignis::flash_next
