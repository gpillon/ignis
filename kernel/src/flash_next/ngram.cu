// ignis kernel leaf -- Flash-Next's n-gram embedding, device side (spec
// flash-next/04 stories 21-24, slice S4 of GitHub #302). OURS (ADR 0043): no
// ninfer provenance; the oracle is transformers' Qwen4ExpTextPLELayer.forward
// (kernel/tests/fixtures/flash_next_ngram/record.py records it).
//
// The host hashes each token to `ngram_heads` table rows and gathers them
// (ignis_core::ngram); this file takes the gathered INT4 rows and adds the
// layer's output to every hyper-connection stream, as the decoder does before
// layer `ngram_layer`'s attention mix:
//
//   e   = dequant(rows)                      BF16 [rows][heads * head_dim]
//   k   = norm_key(key_proj(e))              grouped per stream, (1 + w)
//   v   = value_proj(e)                      BF16 [rows][hidden]
//   q   = norm_query(hidden)                 grouped per stream, (1 + w)
//   g_s = sum(k_s * q_s) / sqrt(hidden);  g_s = sign(g_s) * sqrt(max(|g_s|, 1e-6))
//   gv_s = sigmoid(g_s) * v                  [rows][streams * hidden]
//   c   = silu(dwconv(norm_conv(gv)))        kernel 4, dilation ngram_size,
//                                            over the lane's past 9 columns
//   hidden += gv + c
//
// Every value is rounded to BF16 where the reference (a BF16 module) rounds
// it: the dequantized row, each projection, each norm, each elementwise
// product, the gate's sum, quotient, root and sigmoid, the conv, the silu, the
// sum and the residual add. The projections are kern's FP8 linear (or the BF16
// route) through S1's fn_linear: the reference multiplies BF16-rounded
// weights, the FP8 linear scales an exact fp32 sum, the one difference a test
// tolerates.
//
// A call runs in windows of at most kWindowRows rows (all lanes, a run of
// tokens each), carrying each lane's conv state from one window to the next,
// so the scratch is a few MB whatever the prefill chunk; a decode call (one
// token per lane) is one window, graph-capturable: lanes' slots and positions
// are read from device memory.

#include "flash_next_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

namespace ignis::flash_next {
namespace {

// Rows (lane x token pairs) per window.
constexpr int32_t kWindowRows = 256;
constexpr int32_t kThreads = 256;
// The conv this file specializes on (the checkpoint's: ple_conv_kernel_size 4,
// dilation = ngram_size 3, so 9 past columns). Refused at call time otherwise.
constexpr int32_t kConvKernel = 4;
constexpr int32_t kDilation = 3;
constexpr int32_t kStateColumns = (kConvKernel - 1) * kDilation;
// Every table row's values are grouped by 32 under one fp16 scale.
constexpr int32_t kScaleGroup = 32;

__device__ __forceinline__ float to_bf16(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float load(const __nv_bfloat16 *p) { return __bfloat162float(*p); }

// The token row of window row `r`: windows hold `n` consecutive tokens of
// every lane, lane-major, starting at token `t0` of each.
__device__ __forceinline__ std::size_t source_row(int32_t r, int32_t n, int32_t tokens, int32_t t0) {
  const int32_t lane = r / n;
  return static_cast<std::size_t>(lane) * tokens + t0 + r % n;
}

template <int kWarps>
__device__ float block_sum(float value, float *shared) {
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xFFFFFFFFu, value, offset);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  float total = 0.0F;
  if (threadIdx.x < 32) {
    total = threadIdx.x < kWarps ? shared[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_xor_sync(0xFFFFFFFFu, total, offset);
    if (threadIdx.x == 0) shared[0] = total;
  }
  __syncthreads();
  return shared[0];
}

// e[r][h * head_dim + i] = bf16((nibble - 8) * scale): byte i / 2 of row h
// holds value i in its low nibble when i is even, its high one when odd; the
// fp16 scales follow the codes (layout.md 7.1).
__global__ void dequant_rows(const uint8_t *rows_int4, int32_t tokens, int32_t t0, int32_t n,
                             int32_t heads, int32_t head_dim, int32_t row_bytes, __nv_bfloat16 *e) {
  const int32_t r = blockIdx.x;
  const int32_t width = heads * head_dim;
  const uint8_t *token = rows_int4 + source_row(r, n, tokens, t0) * heads * row_bytes;
  for (int32_t v = threadIdx.x; v < width; v += blockDim.x) {
    const uint8_t *row = token + static_cast<std::size_t>(v / head_dim) * row_bytes;
    const int32_t i = v % head_dim;
    const uint8_t byte = row[i / 2];
    const int32_t nibble = (i & 1) ? (byte >> 4) : (byte & 0xF);
    const uint8_t *scale_at = row + head_dim / 2 + 2 * (i / kScaleGroup);
    const uint16_t bits = static_cast<uint16_t>(scale_at[0] | (scale_at[1] << 8));
    const float scale = __half2float(__ushort_as_half(bits));
    e[static_cast<std::size_t>(r) * width + v] = __float2bfloat16_rn(static_cast<float>(nibble - 8) * scale);
  }
}

// One block per (window row, stream): the grouped key and query norms, the
// gate and the inverse RMS of the gated value the conv norm needs.
__global__ void __launch_bounds__(kThreads) gate_rows(
    const __nv_bfloat16 *k, const __nv_bfloat16 *v, const __nv_bfloat16 *hidden,
    const __nv_bfloat16 *norm_key, const __nv_bfloat16 *norm_query, int32_t tokens, int32_t t0,
    int32_t n, int32_t width, float eps, float sqrt_width, float *gates, float *conv_rms) {
  __shared__ float shared[kThreads / 32];
  const int32_t r = blockIdx.x;
  const int32_t s = blockIdx.y;
  const int32_t streams = gridDim.y;
  const __nv_bfloat16 *kr = k + (static_cast<std::size_t>(r) * streams + s) * width;
  const __nv_bfloat16 *qr = hidden + (source_row(r, n, tokens, t0) * streams + s) * width;
  const __nv_bfloat16 *vr = v + static_cast<std::size_t>(r) * width;
  const __nv_bfloat16 *wk = norm_key + static_cast<std::size_t>(s) * width;
  const __nv_bfloat16 *wq = norm_query + static_cast<std::size_t>(s) * width;

  float kk = 0.0F, qq = 0.0F;
  for (int32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float kf = load(kr + i), qf = load(qr + i);
    kk += kf * kf;
    qq += qf * qf;
  }
  kk = block_sum<kThreads / 32>(kk, shared);
  qq = block_sum<kThreads / 32>(qq, shared);
  const float rk = 1.0F / sqrtf(kk / static_cast<float>(width) + eps);
  const float rq = 1.0F / sqrtf(qq / static_cast<float>(width) + eps);

  float dot = 0.0F;
  for (int32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float kn = to_bf16(load(kr + i) * rk * (1.0F + load(wk + i)));
    const float qn = to_bf16(load(qr + i) * rq * (1.0F + load(wq + i)));
    dot += to_bf16(kn * qn);
  }
  dot = block_sum<kThreads / 32>(dot, shared);
  // The reference's BF16 steps: the sum, the quotient, max(|g|, 1e-6), the
  // root, the sign, the sigmoid.
  const float g = to_bf16(to_bf16(dot) / sqrt_width);
  const float root = to_bf16(sqrtf(to_bf16(fmaxf(fabsf(g), 1e-6F))));
  const float signed_root = g > 0.0F ? root : (g < 0.0F ? -root : 0.0F);
  const float gate = to_bf16(1.0F / (1.0F + expf(-signed_root)));

  float gg = 0.0F;
  for (int32_t i = threadIdx.x; i < width; i += blockDim.x) {
    const float gv = to_bf16(gate * load(vr + i));
    gg += gv * gv;
  }
  gg = block_sum<kThreads / 32>(gg, shared);
  if (threadIdx.x == 0) {
    gates[static_cast<std::size_t>(r) * streams + s] = gate;
    conv_rms[static_cast<std::size_t>(r) * streams + s] = 1.0F / sqrtf(gg / static_cast<float>(width) + eps);
  }
}

// One thread per (lane, channel): walks the window's tokens in order with the
// lane's last kStateColumns conv inputs in registers, adds gv + silu(conv) to
// the channel of every token's hidden row, and writes the columns back.
__global__ void __launch_bounds__(kThreads) conv_add(
    const __nv_bfloat16 *v, const float *gates, const float *conv_rms,
    const __nv_bfloat16 *norm_conv, const __nv_bfloat16 *conv, const int32_t *slots,
    const int32_t *positions, __nv_bfloat16 *state, __nv_bfloat16 *hidden, int32_t tokens,
    int32_t t0, int32_t n, int32_t width, int32_t streams) {
  const int32_t channels = streams * width;
  const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t lane = blockIdx.y;
  if (c >= channels) return;
  const int32_t s = c / width;
  const int32_t i = c % width;
  __nv_bfloat16 *columns = state + static_cast<std::size_t>(slots[lane]) * kStateColumns * channels;

  // A sequence's first token starts from zeros (the reference pads its conv
  // input), whatever the slot held before.
  const bool fresh = t0 == 0 && positions[lane] == 0;
  float ring[kStateColumns];
#pragma unroll
  for (int m = 0; m < kStateColumns; ++m) {
    ring[m] = fresh ? 0.0F : load(columns + static_cast<std::size_t>(m) * channels + c);
  }
  float w[kConvKernel];
#pragma unroll
  for (int tap = 0; tap < kConvKernel; ++tap) w[tap] = load(conv + static_cast<std::size_t>(c) * kConvKernel + tap);
  const float norm_scale = 1.0F + load(norm_conv + c);

  for (int32_t j = 0; j < n; ++j) {
    const int32_t r = lane * n + j;
    const std::size_t g_at = static_cast<std::size_t>(r) * streams + s;
    const float gv = to_bf16(gates[g_at] * load(v + static_cast<std::size_t>(r) * width + i));
    const float gvn = to_bf16(gv * conv_rms[g_at] * norm_scale);
    // torch's conv1d (cross-correlation) over [past 9 | this token]: tap k
    // reads the input kDilation * (kConvKernel - 1 - k) tokens back.
    float acc = 0.0F;
#pragma unroll
    for (int tap = 0; tap < kConvKernel - 1; ++tap) acc += w[tap] * ring[tap * kDilation];
    acc += w[kConvKernel - 1] * gvn;
    const float conv_out = to_bf16(acc);
    const float act = to_bf16(conv_out / (1.0F + expf(-conv_out)));
    const float add = to_bf16(gv + act);
    __nv_bfloat16 *h = hidden + (source_row(r, n, tokens, t0) * streams + s) * width + i;
    *h = __float2bfloat16_rn(load(h) + add);
#pragma unroll
    for (int m = 0; m < kStateColumns - 1; ++m) ring[m] = ring[m + 1];
    ring[kStateColumns - 1] = gvn;
  }
#pragma unroll
  for (int m = 0; m < kStateColumns; ++m) {
    columns[static_cast<std::size_t>(m) * channels + c] = __float2bfloat16_rn(ring[m]);
  }
}

std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// A launch's own failure, named, before the next launch can mask it.
bool launched(const char *what) {
  if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
    fn_set_error(std::string("fn_ngram_add: ") + what + ": launch failed: " + cudaGetErrorString(err));
    return false;
  }
  return true;
}

// The rows of one window and the scratch they take.
struct WindowBytes {
  std::size_t e, k, v, gates, rms;
  std::size_t total() const { return aligned(e) + aligned(k) + aligned(v) + aligned(gates) + aligned(rms); }
};

WindowBytes window_bytes(const Geometry &g, int32_t rows) {
  const std::size_t r = static_cast<std::size_t>(rows);
  return WindowBytes{
      r * g.ngram_embed_dim * 2,
      r * g.residual_width() * 2,
      r * g.hidden * 2,
      r * g.streams * 4,
      r * g.streams * 4,
  };
}

std::string check(const Context &ctx, const NgramWeights &w, const Batch &batch) {
  const Geometry &g = ctx.g;
  if (g.ngram_heads <= 0 || g.ngram_embed_dim % g.ngram_heads != 0 || g.ngram_head_dim() % kScaleGroup != 0) {
    return "n-gram geometry: " + std::to_string(g.ngram_heads) + " heads of an embedding of " +
           std::to_string(g.ngram_embed_dim);
  }
  if (g.ngram_conv_kernel != kConvKernel || g.ngram_size != kDilation) {
    return "the n-gram conv is specialized on kernel 4 and dilation 3, the topology has kernel " +
           std::to_string(g.ngram_conv_kernel) + " and n-gram size " + std::to_string(g.ngram_size);
  }
  if (w.key_proj.rows != g.residual_width() || w.key_proj.cols != g.ngram_embed_dim ||
      w.value_proj.rows != g.hidden || w.value_proj.cols != g.ngram_embed_dim) {
    return "n-gram projections: key [" + std::to_string(w.key_proj.rows) + ", " +
           std::to_string(w.key_proj.cols) + "], value [" + std::to_string(w.value_proj.rows) + ", " +
           std::to_string(w.value_proj.cols) + "]";
  }
  if (w.norm_key == nullptr || w.norm_query == nullptr || w.norm_conv == nullptr || w.conv == nullptr) {
    return "an n-gram norm or conv weight is missing";
  }
  if (ctx.ngram.conv_columns == nullptr) return "the lanes' n-gram conv state is missing";
  if (batch.lanes <= 0 || batch.lanes > kWindowRows || batch.tokens <= 0 || batch.slots == nullptr ||
      batch.positions == nullptr) {
    return "a batch of " + std::to_string(batch.lanes) + " lanes of " + std::to_string(batch.tokens) +
           " tokens";
  }
  return {};
}

}  // namespace

std::size_t fn_ngram_add_scratch_bytes(const Geometry &g, int32_t rows) {
  return window_bytes(g, std::min(rows, kWindowRows)).total();
}

int32_t fn_ngram_add(const Context &ctx, const NgramWeights &w, const Batch &batch,
                     const void *rows_int4, void *hidden, ninfer::DeviceArena &scratch,
                     cudaStream_t stream) {
  if (const std::string bad = check(ctx, w, batch); !bad.empty()) {
    fn_set_error("fn_ngram_add: " + bad);
    return -1;
  }
  try {
    const Geometry &g = ctx.g;
    auto scope = scratch.scope();
    const int32_t per_lane = std::min(batch.tokens, std::max(1, kWindowRows / batch.lanes));
    const WindowBytes bytes = window_bytes(g, batch.lanes * per_lane);
    auto *e = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes.e).data);
    auto *k = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes.k).data);
    auto *v = static_cast<__nv_bfloat16 *>(scratch.alloc_bytes(bytes.v).data);
    auto *gates = static_cast<float *>(scratch.alloc_bytes(bytes.gates).data);
    auto *rms = static_cast<float *>(scratch.alloc_bytes(bytes.rms).data);
    auto *state = static_cast<__nv_bfloat16 *>(ctx.ngram.conv_columns);
    auto *h = static_cast<__nv_bfloat16 *>(hidden);
    const auto *rows = static_cast<const uint8_t *>(rows_int4);
    const float sqrt_width = std::sqrt(static_cast<float>(g.hidden));

    for (int32_t t0 = 0; t0 < batch.tokens; t0 += per_lane) {
      const int32_t n = std::min(per_lane, batch.tokens - t0);
      const int32_t window = batch.lanes * n;
      dequant_rows<<<window, kThreads, 0, stream>>>(rows, batch.tokens, t0, n, g.ngram_heads,
                                                     g.ngram_head_dim(), g.ngram_row_bytes(), e);
      if (!launched("dequant")) return -1;
      if (fn_linear(w.key_proj, e, window, k, false, scratch, stream) != 0 ||
          fn_linear(w.value_proj, e, window, v, false, scratch, stream) != 0) {
        return -1;  // fn_linear named the failure
      }
      gate_rows<<<dim3(window, g.streams), kThreads, 0, stream>>>(
          k, v, h, static_cast<const __nv_bfloat16 *>(w.norm_key),
          static_cast<const __nv_bfloat16 *>(w.norm_query), batch.tokens, t0, n, g.hidden,
          g.rms_norm_eps, sqrt_width, gates, rms);
      if (!launched("gate")) return -1;
      const int32_t channels = g.residual_width();
      conv_add<<<dim3((channels + kThreads - 1) / kThreads, batch.lanes), kThreads, 0, stream>>>(
          v, gates, rms, static_cast<const __nv_bfloat16 *>(w.norm_conv),
          static_cast<const __nv_bfloat16 *>(w.conv), batch.slots, batch.positions, state, h,
          batch.tokens, t0, n, g.hidden, g.streams);
      if (!launched("conv")) return -1;
    }
    return 0;
  } catch (const std::exception &error) {
    fn_set_error(std::string("fn_ngram_add: ") + error.what());
    return -1;
  }
}

}  // namespace ignis::flash_next
