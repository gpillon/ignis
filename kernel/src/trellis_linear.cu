// ignis kernel leaf: the dense trellis linear's entry points, preparation, workspace, the
// activation rotation and the reconstruct seam -- OURS (kernel/include/ignis_trellis_linear.h).
// The decode and prefill kernels are in trellis_linear_decode.cu and trellis_linear_prefill.cu.
//
// The rotation (rotate_kernel) is the first launch of every call: one warp per (weight, token,
// 128-input block) multiplies x by suh, applies the 128-wide Hadamard in fp32 and stores the
// block in fp16 scaled by the power of two that puts its largest entry in [2^(E-1), 2^E), with
// the exact inverse scale beside it. Per-block scales keep a token with one loud block from
// flushing its quiet ones, and need no pass over the whole row first. The launch lets its
// dependent start at once (programmatic dependent launch), so the multiply's weight loads are in
// flight while the activations rotate.

#include "moe_common.cuh"
#include "trellis_linear_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace ignis_tl {

namespace {
thread_local std::string g_last_error = "no error";
constexpr int kMaxDevices = 64;
std::mutex g_mutex;
DeviceState g_state[kMaxDevices];
}  // namespace

int32_t fail(const std::string &message) {
  g_last_error = message;
  return -1;
}

int32_t check_launch(const char *op) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return fail(std::string(op) + ": launch failed: " + cudaGetErrorString(err));
  return 0;
}

int32_t require_prepared(const char *op, DeviceState *state) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  std::lock_guard<std::mutex> lock(g_mutex);
  if (device < 0 || device >= kMaxDevices || g_state[device].sms == 0) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_trellis_prepare at load");
  }
  *state = g_state[device];
  return 0;
}

namespace {

// ---- the debug SM cap's admission ------------------------------------------------------------

__global__ void smid_probe_kernel(uint32_t *ids) {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) ids[blockIdx.x] = smid;
  // Hold the SM long enough that the grid spreads over all of them.
  const long long start = clock64();
  while (clock64() - start < 200000) {
  }
}

// The SM ids of the device, from enough CTAs to land on every SM, ascending; the cap admits the
// first `cap` of them.
int32_t admit_first(int sms, uint32_t cap, uint32_t (&admit)[kMaxSmIds / 32]) {
  const int blocks = 8 * sms;
  uint32_t *d_ids = nullptr;
  if (cudaMalloc(&d_ids, blocks * sizeof(uint32_t)) != cudaSuccess) return fail("ignis_trellis_prepare: SM probe alloc");
  smid_probe_kernel<<<blocks, 32>>>(d_ids);
  std::vector<uint32_t> ids(blocks);
  const cudaError_t err = cudaMemcpy(ids.data(), d_ids, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaFree(d_ids);
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_prepare: SM probe: ") + cudaGetErrorString(err));
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.size() < cap) {
    return fail("ignis_trellis_prepare: IGNIS_TRELLIS_SM_CAP=" + std::to_string(cap) + " but the probe saw " +
                std::to_string(ids.size()) + " SMs");
  }
  for (auto &w : admit) w = 0;
  for (uint32_t i = 0; i < cap; ++i) {
    if (ids[i] >= static_cast<uint32_t>(kMaxSmIds)) return fail("ignis_trellis_prepare: SM id beyond the cap's bitmap");
    admit[ids[i] >> 5] |= 1u << (ids[i] & 31);
  }
  return 0;
}

// ---- the activation rotation -----------------------------------------------------------------

// Element k (0..127) of a token's 128-input block in fragment order: the decode kernel's lane
// with column index c = (k % 8) / 2 reads, for each of the block's eight 16-input tiles kt, the
// four fp16 values (k = 16 kt + 2c, +1, +8, +9) -- its m16n8k16 B registers b0 / b1 -- as one
// 8-byte word.
__device__ __forceinline__ int frag_position(int k) {
  const int j = k & 15;
  return ((j & 7) >> 1) * 32 + (k >> 4) * 4 + (j >> 3) * 2 + (j & 1);
}

// PROTOTYPE kBlock8 (IGNIS_PROTO_FAKEQUANT=w8a8blk, the prefill route only): A8 per (token,
// 128-input block) under the block's own power of two. The block's largest entry sits in
// [2^(E-1), 2^E) with E = kScaleExp = 9; the INT8 value is the mantissa two binades down, max in
// [64, 128): q = clamp(rn(v 2^(shift - 2)), +-127), stored as 4 q (exact in fp16) beside the same
// inverse scale.
template <bool kBlock8>
__global__ void __launch_bounds__(kThreads) rotate_kernel(const __grid_constant__ RotateParams p) {
  grid_dependents_launch();
  const int lane = threadIdx.x & 31;
  const int nb = p.in / 128;
  const int item = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (item >= p.weights * p.tokens * nb) return;
  // item = (w * nb + blk) * tokens + t: the [weight][block][token] order of xr and inv.
  const int t = item % p.tokens;
  const int blk = (item / p.tokens) % nb;
  const int w = item / (p.tokens * nb);
  const int k0 = blk * 128 + 4 * lane;
  float x[4], s[4], v[4];
  load_bf16x4(p.x + static_cast<size_t>(t) * p.in + k0, x);
  load_half4(p.suh[w] + k0, s);
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] = x[q] * s[q];
  ignis_moe::warp_hadamard128(v);
  float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  m = ignis_moe::warp_max(m);
  int e = 0;
  if (m > 0.0f && m <= 3.0e38f) frexpf(m, &e);  // m in [2^(e-1), 2^e)
  int shift = p.scale_exp - e;
  if (m == 0.0f || !(m <= 3.0e38f)) shift = 0;
  shift = min(shift, 126);
  const float scale = ldexpf(1.0f, shift);
  __half h[4];
  if constexpr (kBlock8) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int i8 = max(-127, min(127, __float2int_rn(v[q] * scale * 0.25f)));
      h[q] = __int2half_rn(4 * i8);
    }
  } else {
#pragma unroll
    for (int q = 0; q < 4; ++q) h[q] = __float2half_rn(v[q] * scale);
  }
  if (p.frag_order) {
    __half *dst = p.xr + static_cast<size_t>(item) * 128;
    // Elements 4 lane + {0, 1} and + {2, 3} are pairs that sit together in fragment order.
    *reinterpret_cast<__half2 *>(dst + frag_position(4 * lane)) = __halves2half2(h[0], h[1]);
    *reinterpret_cast<__half2 *>(dst + frag_position(4 * lane + 2)) = __halves2half2(h[2], h[3]);
  } else {
    __half *dst = p.xr + (static_cast<size_t>(w) * p.tokens + t) * p.in + k0;
    uint2 packed;
    const __half2 a = __halves2half2(h[0], h[1]);
    const __half2 b = __halves2half2(h[2], h[3]);
    packed.x = *reinterpret_cast<const uint32_t *>(&a);
    packed.y = *reinterpret_cast<const uint32_t *>(&b);
    *reinterpret_cast<uint2 *>(dst) = packed;
  }
  if (lane == 0) p.inv[item] = ldexpf(1.0f, -shift);
}

// PROTOTYPE (IGNIS_PROTO_FAKEQUANT=w8a8tok, the prefill route only, one weight): A8 per token. One
// CTA per token rotates its row twice -- the same fp32 arithmetic as rotate_kernel -- first for
// the amax over the whole K, then to store q = clamp(rn(v 127 / amax), +-127) in fp16 (exact)
// with amax / 127 as every block's inverse scale.
__global__ void __launch_bounds__(kThreads) rotate_token_kernel(const __grid_constant__ RotateParams p) {
  grid_dependents_launch();
  __shared__ float s_max[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nb = p.in / 128;
  const int t = blockIdx.x;
  auto rotated = [&](int blk, float (&v)[4]) {
    const int k0 = blk * 128 + 4 * lane;
    float x[4], s[4];
    load_bf16x4(p.x + static_cast<size_t>(t) * p.in + k0, x);
    load_half4(p.suh[0] + k0, s);
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = x[q] * s[q];
    ignis_moe::warp_hadamard128(v);
  };
  float m = 0.0f;
  for (int blk = warp; blk < nb; blk += kWarps) {
    float v[4];
    rotated(blk, v);
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
  }
  m = ignis_moe::warp_max(m);
  if (lane == 0) s_max[warp] = m;
  __syncthreads();
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kWarps; ++i) amax = fmaxf(amax, s_max[i]);
  const bool finite = amax > 0.0f && amax <= 3.0e38f;
  const float to_q = finite ? 127.0f / amax : 0.0f;
  const float inv = finite ? amax * (1.0f / 127.0f) : 0.0f;
  for (int blk = warp; blk < nb; blk += kWarps) {
    float v[4];
    rotated(blk, v);
    __half h[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) h[q] = __int2half_rn(max(-127, min(127, __float2int_rn(v[q] * to_q))));
    __half *dst = p.xr + static_cast<size_t>(t) * p.in + blk * 128 + 4 * lane;
    uint2 packed;
    const __half2 a = __halves2half2(h[0], h[1]);
    const __half2 b = __halves2half2(h[2], h[3]);
    packed.x = *reinterpret_cast<const uint32_t *>(&a);
    packed.y = *reinterpret_cast<const uint32_t *>(&b);
    *reinterpret_cast<uint2 *>(dst) = packed;
    if (lane == 0) p.inv[static_cast<size_t>(blk) * p.tokens + t] = inv;
  }
}

// ---- reconstruct -----------------------------------------------------------------------------

// One warp per tile: each lane reads its window's words, decodes its eight weights and stores
// them at their (row, column) -- the decode the multiply runs, writing instead of multiplying.
template <int K2>
__global__ void reconstruct_kernel(const uint32_t *trellis, int stride, int tiles_k, int tiles_n, __half *w) {
  const int lane = threadIdx.x & 31;
  const int warps = (blockDim.x >> 5) * gridDim.x;
  constexpr int words = ignis_trellis::tile_words(K2);
  const int out = tiles_n * 16;
  const int r0 = (lane % 4) * 2;
  const int c0 = lane / 4;
  for (int tile = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); tile < tiles_k * tiles_n; tile += warps) {
    const int tk = tile / tiles_n;
    const int tn = tile % tiles_n;
    const uint32_t *t = trellis + (static_cast<size_t>(tk) * stride + tn) * words;
    uint32_t frag[4];
    if constexpr (K2 >= 10) {
      const ignis_trellis::LanePlan3 plan = ignis_trellis::lane_plan3(K2, lane);
      ignis_trellis::decode_fragment3<K2>(t[plan.w0], t[plan.w1], t[plan.w2], plan, frag);
    } else {
      const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
      ignis_trellis::decode_fragment<K2>(t[plan.w0], t[plan.w1], plan, frag);
    }
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

// ---- argument checks -------------------------------------------------------------------------

bool aligned(const void *p, uintptr_t a) { return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (a - 1)) == 0; }

bool valid_k2(uint32_t k2) { return k2 == 4 || k2 == 5 || k2 == 6 || k2 == 8 || k2 == 10 || k2 == 12; }

int32_t check_weight(const char *op, const char *name, const ignis_trellis_weight *w) {
  const std::string who = std::string(op) + ": " + name;
  if (w == nullptr) return fail(who + " is NULL");
  if (!valid_k2(w->k2)) return fail(who + ": k2 must be 4, 5, 6, 8, 10 or 12");
  if (w->in == 0 || w->out == 0 || w->in % 128 != 0 || w->out % 128 != 0) {
    return fail(who + ": in and out must be positive multiples of 128");
  }
  if (w->out / 128 > kArrivalBlocks) {
    return fail(who + ": out must be at most " + std::to_string(kArrivalBlocks * 128));
  }
  const uint32_t stride = w->tile_stride == 0 ? w->out / 16 : w->tile_stride;
  if (stride < w->out / 16) return fail(who + ": tile_stride is narrower than out / 16");
  if (!aligned(w->trellis, 16) || !aligned(w->suh, 8) || !aligned(w->svh, 8)) {
    return fail(who + ": trellis must be 16-byte aligned, suh and svh 8-byte aligned");
  }
  return 0;
}

Weight device_weight(const ignis_trellis_weight &w) {
  Weight d;
  d.trellis = static_cast<const uint32_t *>(w.trellis);
  d.suh = static_cast<const __half *>(w.suh);
  d.svh = static_cast<const __half *>(w.svh);
  d.k2 = static_cast<int>(w.k2);
  d.stride = static_cast<int>(w.tile_stride == 0 ? w.out / 16 : w.tile_stride);
  return d;
}

bool wide_call(const ignis_trellis_weight *gate, const ignis_trellis_weight *up) {
  return gate->k2 >= 10 || (up != nullptr && up->k2 >= 10);
}

int32_t check_call(const char *op, const DeviceState &st, const void *x, uint32_t tokens, const void *y, uint32_t ldy,
                   uint32_t out, const ignis_trellis_workspace *ws, uint32_t in, uint32_t weights, bool wide) {
  if (x == nullptr || y == nullptr) return fail(std::string(op) + ": NULL activations");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be positive");
  if (!aligned(x, 16)) return fail(std::string(op) + ": x must be 16-byte aligned");
  if (!aligned(y, 8) || ldy % 4 != 0 || ldy < out) {
    return fail(std::string(op) + ": y must be 8-byte aligned and ldy a multiple of 4, at least out");
  }
  if (ws == nullptr || !aligned(ws->base, 256)) return fail(std::string(op) + ": workspace base must be 256-byte aligned");
  const uint64_t need = layout(in, out, weights, tokens, decode_resident(st, tokens, wide)).bytes;
  if (ws->bytes < need) {
    return fail(std::string(op) + ": workspace holds " + std::to_string(ws->bytes) + " bytes, the call needs " +
                std::to_string(need));
  }
  return 0;
}

const char *fq_name(int fq) {
  return fq == kFqW8 ? "w8" : fq == kFqW8A8Token ? "w8a8tok" : fq == kFqW8A8Block ? "w8a8blk" : "off";
}

// PROTOTYPE: each matrix shape the fake quant touches, printed the first time it does (the
// evidence of what the prefill route carries: the body's matrices, never the head).
void proto_log_shape(int fq, const ignis_trellis_weight &w, uint32_t tokens) {
  static std::mutex m;
  static std::set<std::tuple<uint32_t, uint32_t, uint32_t>> seen;
  std::lock_guard<std::mutex> lock(m);
  if (seen.insert(std::make_tuple(w.in, w.out, w.k2)).second) {
    std::fprintf(stderr, "ignis proto: fake quant %s on the prefill route: in %u out %u k2 %u (tokens %u)\n",
                 fq_name(fq), w.in, w.out, w.k2, tokens);
  }
}

// The rotation of `weights` weights' activations, into the call's layout.
int32_t rotate(const char *op, const void *x, const ignis_trellis_weight *const *w, int weights, uint32_t tokens,
               bool decode, const Layout &l, const ignis_trellis_workspace *ws, cudaStream_t stream, int fq = kFqOff) {
  RotateParams r{};
  r.fq = fq;
  r.x = static_cast<const __nv_bfloat16 *>(x);
  for (int i = 0; i < weights; ++i) r.suh[i] = static_cast<const __half *>(w[i]->suh);
  r.weights = weights;
  r.tokens = static_cast<int>(tokens);
  r.in = static_cast<int>(w[0]->in);
  r.scale_exp = kScaleExp;
  r.frag_order = decode ? 1 : 0;
  char *base = static_cast<char *>(ws->base);
  r.xr = reinterpret_cast<__half *>(base + l.xr);
  r.inv = reinterpret_cast<float *>(base + l.inv);
  if (launch_rotate(r, stream) != 0) return -1;
  return check_launch(op);
}

int32_t run(const char *op, const DeviceState &st, const ignis_trellis_weight *gate, const ignis_trellis_weight *up,
            const void *x, uint32_t tokens, void *y, uint32_t ldy, int epilogue, const ignis_trellis_workspace *ws,
            void *stream_ptr) {
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  const int weights = up == nullptr ? 1 : 2;
  const uint32_t in = gate->in, out = gate->out;
  const Layout l = layout(in, out, static_cast<uint32_t>(weights), tokens, decode_resident(st, tokens, wide_call(gate, up)));
  char *base = static_cast<char *>(ws->base);
  uint32_t *counters = reinterpret_cast<uint32_t *>(base);
  const ignis_trellis_weight *ws_list[2] = {gate, up};
  if (tokens <= static_cast<uint32_t>(kDecodeMaxTokens)) {
    if (rotate(op, x, ws_list, weights, tokens, true, l, ws, stream) != 0) return -1;
    DecodeParams p{};
    p.w[0] = device_weight(*gate);
    if (up != nullptr) p.w[1] = device_weight(*up);
    p.weights = weights;
    p.tokens = static_cast<int>(tokens);
    p.in = static_cast<int>(in);
    p.out = static_cast<int>(out);
    p.nb = static_cast<int>(in / 128);
    p.splits = l.splits;
    p.units = weights * static_cast<int>(out / 128) * l.splits;
    p.epilogue = epilogue;
    p.xr = reinterpret_cast<const __half *>(base + l.xr);
    p.inv = reinterpret_cast<const float *>(base + l.inv);
    p.partials = reinterpret_cast<float *>(base + l.partials);
    p.counters = counters;
    p.arrivals = reinterpret_cast<uint32_t *>(base + l.arrivals);
    p.y = static_cast<__nv_bfloat16 *>(y);
    p.ldy = static_cast<int>(ldy);
    p.capped = st.cap != 0;
    for (int i = 0; i < kMaxSmIds / 32; ++i) p.admit[i] = st.admit[i];
    if (launch_decode(st, p, stream) != 0) return -1;
    return check_launch(op);
  }
  // Prefill: one weight at a time through the same rotated-activation region. The SwiGLU's gate
  // pass stores BF16 gate values into h; the up pass reads them back.
  for (int i = 0; i < weights; ++i) {
    const ignis_trellis_weight *w = ws_list[i];
    if (st.fq != kFqOff) proto_log_shape(st.fq, *w, tokens);
    if (rotate(op, x, &w, 1, tokens, false, l, ws, stream, st.fq) != 0) return -1;
    PrefillParams p{};
    p.w8 = st.fq != kFqOff ? 1 : 0;
    p.w = device_weight(*w);
    p.tokens = static_cast<int>(tokens);
    p.in = static_cast<int>(in);
    p.out = static_cast<int>(out);
    p.nb = static_cast<int>(in / 128);
    p.tiles = static_cast<int>((tokens + kPrefillRows - 1) / kPrefillRows * (out / 128));
    p.epilogue = weights == 1 ? epilogue : (i == 0 ? kStore : kSwiGluUp);
    p.xr = reinterpret_cast<const __half *>(base + l.xr);
    p.inv = reinterpret_cast<const float *>(base + l.inv);
    p.counters = counters;
    p.y = static_cast<__nv_bfloat16 *>(y);
    p.ldy = static_cast<int>(ldy);
    p.capped = st.cap != 0;
    for (int j = 0; j < kMaxSmIds / 32; ++j) p.admit[j] = st.admit[j];
    if (launch_prefill(st, p, stream) != 0) return -1;
    if (check_launch(op) != 0) return -1;
  }
  return 0;
}

}  // namespace

int32_t launch_rotate(const RotateParams &p, cudaStream_t stream) {
  const int items = p.weights * p.tokens * (p.in / 128);
  if (!p.frag_order && p.weights == 1 && p.fq == kFqW8A8Token) {
    rotate_token_kernel<<<p.tokens, kThreads, 0, stream>>>(p);
  } else if (!p.frag_order && p.fq == kFqW8A8Block) {
    rotate_kernel<true><<<(items + kWarps - 1) / kWarps, kThreads, 0, stream>>>(p);
  } else {
    rotate_kernel<false><<<(items + kWarps - 1) / kWarps, kThreads, 0, stream>>>(p);
  }
  return 0;
}

}  // namespace ignis_tl

using namespace ignis_tl;

extern "C" int32_t ignis_trellis_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_trellis_prepare: device index out of range");
  DeviceState st;
  err = cudaDeviceGetAttribute(&st.sms, cudaDevAttrMultiProcessorCount, device);
  if (err != cudaSuccess || st.sms <= 0) return fail("ignis_trellis_prepare: no multiprocessor count");
  if (prepare_decode(&st) != 0 || prepare_prefill(&st) != 0) return -1;
  const char *env = std::getenv("IGNIS_TRELLIS_SM_CAP");
  if (env != nullptr && env[0] != '\0') {
    char *end = nullptr;
    const unsigned long cap = std::strtoul(env, &end, 10);
    if (end == env || *end != '\0' || cap > static_cast<unsigned long>(st.sms)) {
      return fail(std::string("ignis_trellis_prepare: IGNIS_TRELLIS_SM_CAP must be 0..") + std::to_string(st.sms) +
                  ", got '" + env + "'");
    }
    st.cap = static_cast<uint32_t>(cap);
    if (st.cap != 0 && admit_first(st.sms, st.cap, st.admit) != 0) return -1;
  }
  // PROTOTYPE: the prefill route's fake quant; a value it does not know refuses the prepare.
  const char *fq = std::getenv("IGNIS_PROTO_FAKEQUANT");
  const std::string mode = fq == nullptr ? "" : fq;
  if (mode.empty() || mode == "off") {
    st.fq = kFqOff;
  } else if (mode == "w8") {
    st.fq = kFqW8;
  } else if (mode == "w8a8tok") {
    st.fq = kFqW8A8Token;
  } else if (mode == "w8a8blk") {
    st.fq = kFqW8A8Block;
  } else {
    return fail("ignis_trellis_prepare: IGNIS_PROTO_FAKEQUANT must be off, w8, w8a8tok or w8a8blk, got '" + mode + "'");
  }
  std::fprintf(stderr, "ignis proto: IGNIS_PROTO_FAKEQUANT=%s\n", fq_name(st.fq));
  std::lock_guard<std::mutex> lock(g_mutex);
  g_state[device] = st;
  return 0;
}

extern "C" uint32_t ignis_trellis_sm_cap(void) {
  DeviceState st;
  return require_prepared("ignis_trellis_sm_cap", &st) == 0 ? st.cap : 0;
}

extern "C" uint64_t ignis_trellis_workspace_bytes(uint32_t in, uint32_t out, uint32_t weights, uint32_t tokens) {
  if (in == 0 || out == 0 || in % 128 != 0 || out % 128 != 0 || out / 128 > kArrivalBlocks || tokens == 0 ||
      weights < 1 || weights > 2) {
    return 0;
  }
  DeviceState st;
  if (require_prepared("ignis_trellis_workspace_bytes", &st) != 0) return 0;
  // The K <= 4 kernel has the most resident CTAs, so the most splits: an upper bound for any K.
  return layout(in, out, weights, tokens, decode_resident(st, tokens, false)).bytes;
}

extern "C" int32_t ignis_trellis_workspace_init(const ignis_trellis_workspace *ws, void *stream) {
  if (ws == nullptr || ws->base == nullptr || ws->bytes < 256) {
    return fail("ignis_trellis_workspace_init: no workspace");
  }
  // All of it: the counters and arrival counts must start at zero, the rest is written before
  // it is read.
  const cudaError_t err = cudaMemsetAsync(ws->base, 0, ws->bytes, static_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_workspace_init: ") + cudaGetErrorString(err));
  return 0;
}

extern "C" int32_t ignis_trellis_linear(const ignis_trellis_weight *w, const void *x, uint32_t tokens, void *y,
                                        uint32_t ldy, uint32_t epilogue, const ignis_trellis_workspace *workspace,
                                        void *stream) {
  const char *op = "ignis_trellis_linear";
  if (check_weight(op, "weight", w) != 0) return -1;
  if (epilogue != IGNIS_TRELLIS_STORE && epilogue != IGNIS_TRELLIS_ADD) return fail("ignis_trellis_linear: unknown epilogue");
  DeviceState st;
  if (require_prepared(op, &st) != 0) return -1;
  if (check_call(op, st, x, tokens, y, ldy, w->out, workspace, w->in, 1, wide_call(w, nullptr)) != 0) return -1;
  return run(op, st, w, nullptr, x, tokens, y, ldy, epilogue == IGNIS_TRELLIS_ADD ? kAdd : kStore, workspace, stream);
}

extern "C" int32_t ignis_trellis_linear_swiglu(const ignis_trellis_weight *gate, const ignis_trellis_weight *up,
                                               const void *x, uint32_t tokens, void *h, uint32_t ldh,
                                               const ignis_trellis_workspace *workspace, void *stream) {
  const char *op = "ignis_trellis_linear_swiglu";
  if (check_weight(op, "gate", gate) != 0 || check_weight(op, "up", up) != 0) return -1;
  if (gate->in != up->in || gate->out != up->out) return fail("ignis_trellis_linear_swiglu: gate and up shapes differ");
  DeviceState st;
  if (require_prepared(op, &st) != 0) return -1;
  if (check_call(op, st, x, tokens, h, ldh, gate->out, workspace, gate->in, 2, wide_call(gate, up)) != 0) return -1;
  return run(op, st, gate, up, x, tokens, h, ldh, kSwiGlu, workspace, stream);
}

extern "C" int32_t ignis_trellis_reconstruct(const ignis_trellis_weight *w, void *w_f16, void *stream) {
  const char *op = "ignis_trellis_reconstruct";
  if (check_weight(op, "weight", w) != 0) return -1;
  if (w_f16 == nullptr) return fail("ignis_trellis_reconstruct: NULL output");
  const Weight d = device_weight(*w);
  const int tiles_k = static_cast<int>(w->in / 16), tiles_n = static_cast<int>(w->out / 16);
  const long long tiles = static_cast<long long>(tiles_k) * tiles_n;
  const int grid = static_cast<int>(std::min<long long>((tiles + kWarps - 1) / kWarps, 65536));
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  __half *out = static_cast<__half *>(w_f16);
  switch (d.k2) {
  case 4: reconstruct_kernel<4><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  case 5: reconstruct_kernel<5><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  case 6: reconstruct_kernel<6><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  case 8: reconstruct_kernel<8><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  case 10: reconstruct_kernel<10><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  default: reconstruct_kernel<12><<<grid, kThreads, 0, s>>>(d.trellis, d.stride, tiles_k, tiles_n, out); break;
  }
  return check_launch(op);
}

extern "C" const char *ignis_trellis_last_error(void) { return g_last_error.c_str(); }
