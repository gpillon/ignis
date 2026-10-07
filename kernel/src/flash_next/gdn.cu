// ignis kernel leaf -- the Flash-Next GDN layer core (spec flash-next/04, GitHub #302, slice S2):
// OURS (ADR 0043), no port claim. See gdn.h for the oracle and the call shapes.
//
// Kernels here: the causal convolution (written into q | k | v directly), the gating (g, beta
// from the BF16 projections and the layer's BF16 A_log and dt_bias), the sigmoid-gated norm, and
// the copy of one slot's recurrent state between the pool and a scratch image. Everything else is
// fn_linear and the vendored recurrence.
//
// The convolution is ours rather than the vendored causal_conv1d_silu: that op reads its weight
// tap-major ([4][channels], the 27B artifact's gdn/convolution), while Flash-Next's artifact keeps
// the checkpoint's conv1d.weight [channels][1][4] channel-major (layout.md 6.2, the contract's
// GdnWeights::conv). Reading the taps in place costs nothing; a transposed copy would be a plan
// line of its own.
//
// Numerics: every BF16 rounding the checkpoint makes is made here at the same point (the four
// projections, the convolution's BF16 output and its SiLU's, beta = b.sigmoid() in BF16, the
// gated norm's three roundings); g and the statistics are fp32 like the checkpoint's. The
// recurrence keeps an fp32 state where the checkpoint runs its chunked rule in fp32: that op is
// the vendored one, qualified against fp64 by its own contract.

#include "gdn.h"

#include "ignis_seq_internal.h"

#include "ninfer/ops/gated_delta_net.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

namespace ignis::flash_next {

namespace gdn {

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kConvChunk = 64;  // tokens one convolution thread walks

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// The checkpoint's causal_conv1d_fn / causal_conv1d_update with SiLU: per channel c and token t,
// bf16(silu(bf16(sum_j w[c][j] u[t - 3 + j]))) over u = the lane's three taps then its tokens --
// F.conv1d in BF16 rounds the sum, SiLU rounds again. One thread per (lane, channel, chunk of
// kConvChunk tokens); the output goes straight to the recurrence's q [rows][2048] | k [rows][2048]
// | v [rows][6144]. The taps (the pool's [slot][3][channels], oldest first) are read and then
// rewritten by the chunk-0 thread alone: no other thread of the lane reads them, since a chunk
// past the first starts kConvChunk >= 3 tokens in. A verify call (keep_taps) reads them and
// leaves them: the commit's fold writes the taps of the columns it keeps.
__global__ void conv_kernel(const __nv_bfloat16 *__restrict__ qkv, const __nv_bfloat16 *__restrict__ weight,
                            __nv_bfloat16 *__restrict__ taps, const int32_t *__restrict__ slots, int32_t slot_count,
                            int32_t tokens, int32_t chunks, bool keep_taps, __nv_bfloat16 *__restrict__ q,
                            __nv_bfloat16 *__restrict__ k, __nv_bfloat16 *__restrict__ v) {
  const int32_t c = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= kConvChannels) return;
  const int32_t lane = static_cast<int32_t>(blockIdx.y) / chunks;
  const int32_t chunk = static_cast<int32_t>(blockIdx.y) % chunks;
  const int32_t slot = slots[lane];
  if (slot < 0 || slot >= slot_count) __trap();
  __nv_bfloat16 *tap = taps + static_cast<int64_t>(slot) * kConvStateTaps * kConvChannels + c;
  const int64_t row0 = static_cast<int64_t>(lane) * tokens;
  const __nv_bfloat16 *x = qkv + row0 * kConvChannels + c;
  // Where channel c lands: q, k or v (each a warp-aligned run of channels).
  __nv_bfloat16 *out;
  int32_t width;
  if (c < kKeyWidth) {
    out = q + row0 * kKeyWidth + c;
    width = kKeyWidth;
  } else if (c < 2 * kKeyWidth) {
    out = k + row0 * kKeyWidth + (c - kKeyWidth);
    width = kKeyWidth;
  } else {
    out = v + row0 * kValueWidth + (c - 2 * kKeyWidth);
    width = kValueWidth;
  }
  const uint2 wv = *reinterpret_cast<const uint2 *>(weight + static_cast<int64_t>(c) * kConvKernel);
  const auto *wh = reinterpret_cast<const __nv_bfloat16 *>(&wv);
  const float w0 = __bfloat162float(wh[0]), w1 = __bfloat162float(wh[1]), w2 = __bfloat162float(wh[2]),
              w3 = __bfloat162float(wh[3]);
  const int32_t t0 = chunk * kConvChunk;
  const int32_t t1 = min(tokens, t0 + kConvChunk);
  float old[kConvStateTaps] = {0.0F, 0.0F, 0.0F};
  if (chunk == 0) {
    for (int32_t j = 0; j < kConvStateTaps; ++j) old[j] = __bfloat162float(tap[static_cast<int64_t>(j) * kConvChannels]);
  }
  // u[i] = the input i - 3 tokens before t0: a token of the call, or one of the old taps.
  float u[kConvStateTaps];
  for (int32_t i = 0; i < kConvStateTaps; ++i) {
    const int32_t t = t0 - kConvStateTaps + i;
    u[i] = t >= 0 ? __bfloat162float(x[static_cast<int64_t>(t) * kConvChannels]) : old[kConvStateTaps + t];
  }
  for (int32_t t = t0; t < t1; ++t) {
    const float xt = __bfloat162float(x[static_cast<int64_t>(t) * kConvChannels]);
    const float s = round_bf16(fmaf(w3, xt, fmaf(w2, u[2], fmaf(w1, u[1], w0 * u[0]))));
    out[static_cast<int64_t>(t) * width] = __float2bfloat16_rn(s / (1.0F + expf(-s)));
    u[0] = u[1];
    u[1] = u[2];
    u[2] = xt;
  }
  if (chunk == 0 && !keep_taps) {
    for (int32_t j = 0; j < kConvStateTaps; ++j) {
      const int32_t t = tokens - kConvStateTaps + j;
      const float value = t >= 0 ? __bfloat162float(x[static_cast<int64_t>(t) * kConvChannels]) : old[kConvStateTaps + t];
      tap[static_cast<int64_t>(j) * kConvChannels] = __float2bfloat16_rn(value);
    }
  }
}

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
  beta[i] = round_bf16(sig);
}

// The checkpoint's Qwen4ExpTextRMSNormGated with a sigmoid gate, one warp per (token, value
// head): n = bf16(o / sqrt(mean(o^2) + eps)), m = bf16(weight * n), out = bf16(m * sigmoid(z)).
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
  const float r = 1.0F / sqrtf(sum / static_cast<float>(kHeadDim) + eps);
  uint2 result;
  auto *rh = reinterpret_cast<__nv_bfloat16 *>(&result);
  for (int32_t i = 0; i < 4; ++i) {
    const float n = round_bf16(x[i] * r);
    const float m = round_bf16(__bfloat162float(wh[i]) * n);
    const float gate = 1.0F / (1.0F + expf(-__bfloat162float(zh[i])));
    rh[i] = __float2bfloat16_rn(m * gate);
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
            const void *x, void *y, ninfer::DeviceArena &scratch, cudaStream_t stream, int32_t ordinal) {
  if (const char *why = check_geometry(g)) {
    fn_set_error(why);
    return -1;
  }
  const int32_t rows = batch.rows();
  const bool per_lane = batch.tokens == 1;
  const VerifyRecords *record = batch.verify;
  if (rows <= 0 || batch.slots == nullptr) {
    fn_set_error("gdn: an empty batch, or no slots");
    return -1;
  }
  if (record != nullptr) {
    // A verify call: the vendored replay record's domain.
    if (batch.lanes > kMaxLanes || batch.tokens < kMinRecordColumns || batch.tokens > kMaxRecordColumns ||
        record->valid_columns == nullptr || record->gdn_key == nullptr || rows > record->rows) {
      fn_set_error("gdn: a verify call is 1.." + std::to_string(kMaxLanes) + " lanes of " +
                   std::to_string(kMinRecordColumns) + ".." + std::to_string(kMaxRecordColumns) +
                   " columns with its records, in their " + std::to_string(record->rows) + " rows (got " +
                   std::to_string(batch.lanes) + " x " + std::to_string(batch.tokens) + ")");
      return -1;
    }
  } else if (per_lane ? batch.lanes > kMaxLanes : batch.lanes != 1) {
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
    // qkv: the projection, then the recurrence's output; split: the convolution's q | k | v, then
    // the gated norm's output.
    auto *qkv = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kConvChannels * 2).data);
    auto *split = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kConvChannels * 2).data);
    auto *z = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueWidth * 2).data);
    auto *a = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueHeads * 2).data);
    auto *b = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(r * kValueHeads * 2).data);
    auto *gf = static_cast<float *>(scratch.alloc_bytes(r * kValueHeads * 4).data);
    auto *beta = static_cast<float *>(scratch.alloc_bytes(r * kValueHeads * 4).data);
    __nv_bfloat16 *q = split;
    __nv_bfloat16 *k = q + r * kKeyWidth;
    __nv_bfloat16 *v = k + r * kKeyWidth;

    if (fn_linear(w.in_proj_qkv, x, rows, qkv, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_z, x, rows, z, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_a, x, rows, a, false, scratch, stream) != 0 ||
        fn_linear(w.in_proj_b, x, rows, b, false, scratch, stream) != 0) {
      return -1;
    }
    const int32_t gates = rows * kValueHeads;
    gating_kernel<<<(gates + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        a, b, static_cast<const __nv_bfloat16 *>(w.a_log), static_cast<const __nv_bfloat16 *>(w.dt_bias), gf,
        beta, gates);
    // A verify call's conv inputs, before the recurrence's output overwrites qkv: the fold's taps.
    if (record != nullptr &&
        cudaMemcpyAsync(static_cast<unsigned char *>(record->gdn_conv) + static_cast<std::size_t>(ordinal) * record->gdn_layer_bytes,
                        qkv, r * kConvChannels * 2, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      fn_set_error(std::string("gdn: recording the conv inputs failed: ") + cudaGetErrorString(cudaGetLastError()));
      return -1;
    }
    const int32_t chunks = (batch.tokens + kConvChunk - 1) / kConvChunk;
    const dim3 conv_grid((kConvChannels + kThreads - 1) / kThreads, static_cast<unsigned>(batch.lanes * chunks));
    conv_kernel<<<conv_grid, kThreads, 0, stream>>>(qkv, static_cast<const __nv_bfloat16 *>(w.conv),
                                                     static_cast<__nv_bfloat16 *>(state.conv), batch.slots,
                                                     state.slots, batch.tokens, chunks, record != nullptr, q, k, v);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fn_set_error(std::string("gdn: launch failed: ") + cudaGetErrorString(err));
      return -1;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    if (record != nullptr) {
      // Every lane's k + 1 columns from its slot's state, which stays as it is; the transitions'
      // raw inputs go to the layer's records for the commit's fold.
      const auto layer_at = [&](void *plane) {
        return static_cast<unsigned char *>(plane) + static_cast<std::size_t>(ordinal) * record->gdn_layer_bytes;
      };
      const int32_t t = batch.tokens;
      const int32_t lanes = batch.lanes;
      ninfer::Tensor states(state.recurrent, ninfer::DType::FP32, {kHeadDim, kHeadDim, kValueHeads, state.slots});
      ninfer::Tensor key_record(layer_at(record->gdn_key), ninfer::DType::BF16, {kHeadDim, kQkHeads, t, lanes});
      ninfer::Tensor value_record(layer_at(record->gdn_value), ninfer::DType::BF16, {kHeadDim, kValueHeads, t, lanes});
      ninfer::Tensor gate_record(layer_at(record->gdn_gate), ninfer::DType::FP32, {2, kValueHeads, t, lanes});
      ninfer::Tensor out(qkv, ninfer::DType::BF16, {kHeadDim, kValueHeads, t, lanes});
      ninfer::ops::gated_delta_net_replay_record(
          ninfer::Tensor(q, ninfer::DType::BF16, {kHeadDim, kQkHeads, t, lanes}),
          ninfer::Tensor(k, ninfer::DType::BF16, {kHeadDim, kQkHeads, t, lanes}),
          ninfer::Tensor(v, ninfer::DType::BF16, {kHeadDim, kValueHeads, t, lanes}),
          ninfer::Tensor(gf, ninfer::DType::FP32, {kValueHeads, t, lanes, 1}),
          ninfer::Tensor(beta, ninfer::DType::FP32, {kValueHeads, t, lanes, 1}), scale, states,
          ninfer::Tensor(const_cast<int32_t *>(record->valid_columns), ninfer::DType::I32, {lanes, 1, 1, 1}),
          ninfer::Tensor(const_cast<int32_t *>(batch.slots), ninfer::DType::I32, {lanes, 1, 1, 1}), key_record,
          value_record, gate_record, out, stream);
    } else if (per_lane) {
      const ninfer::Tensor lane_slots(const_cast<int32_t *>(batch.slots), ninfer::DType::I32,
                                      {batch.lanes, 1, 1, 1});
      ninfer::Tensor states(state.recurrent, ninfer::DType::FP32, {kHeadDim, kHeadDim, kValueHeads, state.slots});
      ninfer::Tensor out(qkv, ninfer::DType::BF16, {kHeadDim, kValueHeads, 1, rows});
      ninfer::ops::gated_delta_net_snapshot(
          ninfer::Tensor(q, ninfer::DType::BF16, {kHeadDim, kQkHeads, 1, rows}),
          ninfer::Tensor(k, ninfer::DType::BF16, {kHeadDim, kQkHeads, 1, rows}),
          ninfer::Tensor(v, ninfer::DType::BF16, {kHeadDim, kValueHeads, 1, rows}),
          ninfer::Tensor(gf, ninfer::DType::FP32, {kValueHeads, 1, rows, 1}),
          ninfer::Tensor(beta, ninfer::DType::FP32, {kValueHeads, 1, rows, 1}), scale, /*normalize_qk=*/true,
          states, ninfer::Tensor{}, lane_slots, lane_slots, out, stream);
    } else {
      void *image = scratch.alloc_bytes(kRecurrentSlotBytes).data;
      err = slot_copy(state.recurrent, image, kRecurrentSlotBytes, batch.slots, state.slots, true, stream);
      if (err != cudaSuccess) {
        fn_set_error(std::string("gdn: state gather failed: ") + cudaGetErrorString(err));
        return -1;
      }
      ninfer::Tensor state_image(image, ninfer::DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
      ninfer::Tensor out(qkv, ninfer::DType::BF16, {kHeadDim, kValueHeads, rows, 1});
      ninfer::ops::gated_delta_net(
          ninfer::Tensor(q, ninfer::DType::BF16, {kHeadDim, kQkHeads, rows, 1}),
          ninfer::Tensor(k, ninfer::DType::BF16, {kHeadDim, kQkHeads, rows, 1}),
          ninfer::Tensor(v, ninfer::DType::BF16, {kHeadDim, kValueHeads, rows, 1}),
          ninfer::Tensor(gf, ninfer::DType::FP32, {kValueHeads, rows, 1, 1}),
          ninfer::Tensor(beta, ninfer::DType::FP32, {kValueHeads, rows, 1, 1}), scale, /*normalize_qk=*/true,
          scratch, state_image, state_image, out, stream);
      err = slot_copy(state.recurrent, image, kRecurrentSlotBytes, batch.slots, state.slots, false, stream);
      if (err != cudaSuccess) {
        fn_set_error(std::string("gdn: state scatter failed: ") + cudaGetErrorString(err));
        return -1;
      }
    }

    const int32_t units = rows * kValueHeads;
    gated_norm_kernel<<<(units * 32 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        qkv, z, static_cast<const __nv_bfloat16 *>(w.norm), g.rms_norm_eps, split, units);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fn_set_error(std::string("gdn: launch failed: ") + cudaGetErrorString(err));
      return -1;
    }
    return fn_linear(w.out_proj, split, rows, y, false, scratch, stream);
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
  // The pool's slots must be the layout the kernels address: [slot][3][10240] taps,
  // [slot][48][128][128] fp32 states.
  const ninfer::LinearAttentionStatePoolSpec &spec = ctx.pool->gdn_pool.spec;
  if (spec.conv_channels != gdn::kConvChannels || spec.conv_width != gdn::kConvStateTaps ||
      spec.value_heads != gdn::kValueHeads || spec.value_head_dim != gdn::kHeadDim ||
      spec.key_head_dim != gdn::kHeadDim || spec.conv_dtype != ninfer::DType::BF16) {
    fn_set_error("fn_gdn_layer: the seq pool's GDN state is not 10240 x 3 BF16 taps and 48 x 128 x 128 fp32");
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
  return gdn::run(ctx.g, state, w, batch, x, y, scratch, stream, gdn_ordinal);
}

std::size_t fn_gdn_layer_scratch_bytes(const Geometry &g, int32_t rows) {
  (void)g;
  if (rows <= 0) return 0;
  std::size_t bytes = gdn::activation_bytes(rows);
  if (rows > 1) {
    // A prefill call's recurrent state image and the chunked recurrence's workspace (a decode
    // call of several lanes takes neither: counted anyway, the rows alone cannot tell them apart).
    bytes += gdn::aligned(gdn::kRecurrentSlotBytes) +
             ninfer::ops::gated_delta_net_workspace_capacity_bytes(gdn::kQkHeads, gdn::kValueHeads, true, 1, rows) +
             256;
  }
  return bytes;
}

}  // namespace ignis::flash_next
