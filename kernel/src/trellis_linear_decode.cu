// ignis kernel leaf: the dense trellis linear's decode / verify route, 1..64 tokens -- OURS
// (kernel/include/ignis_trellis_linear.h), built on our own Flash-Next register decode
// (moe_decode.cu): the same one-word-per-lane loads with shuffles, the same decode straight into
// m16n8k16 fragments.
//
// One persistent launch takes work units by ticket. A unit is 128 output columns of one weight
// (one 16-column tile per warp) over one split of the inputs, a run of 128-input blocks:
//
//   per block   each warp's eight weight tiles are in registers two blocks ahead (every lane one
//               or two words of each tile, the decode window gathered by shuffles); the block's
//               rotated activations, all tokens, are staged in shared memory by cp.async, in the
//               fragment order the rotation wrote, a token's row padded off the bank stride; the
//               tile is decoded once and is the A operand of one m16n8k16 per eight tokens (the
//               B operand), with fp16 accumulation folded into fp32 times the activations'
//               inverse scale every 32 inputs.
//   at the end  a unit that holds its output block alone (one split, no SwiGLU) rotates it back
//               through shared memory, applies svh and the epilogue and stores BF16. Otherwise
//               it stores its fp32 partial sums; the last unit to arrive on the output block (an
//               arrival count per block) sums every split in split order, rotates, applies svh
//               and the epilogue -- for SwiGLU, the gate's and the up's -- and zeroes the count.
//
// Deterministic by construction: every sum has a fixed order, whichever CTA runs a unit or
// arrives last. The CTA that finishes last resets the ticket, so the launch replays from a
// graph. Capped (IGNIS_TRELLIS_SM_CAP), the grid still covers every SM and the CTAs of SMs the
// cap does not admit leave at once, so the work runs on the admitted SMs at full occupancy.

#include "moe_common.cuh"
#include "trellis_linear_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ignis_tl {
namespace {

using ignis_moe::cp_async16;
using ignis_moe::cp_async_commit;
using ignis_moe::cp_async_wait;
using ignis_moe::warp_hadamard128;

constexpr int kORow = 132;  // fp32 per row of the output staging, off the bank stride
// fp16 per token of a staged activation block: 128 and 8 of padding, so the eight tokens a warp's
// lanes read at one tile land on different banks (a token's row is 68 words, 4 mod 32).
constexpr int kXRow = 136;

// Dynamic shared memory at `tokens` rows: two activation stages (tokens x kXRow fp16) and their
// inverse scales, overlaid at the end of a unit by the [tokens][kORow] fp32 output staging.
__host__ __device__ inline size_t decode_smem(int tokens) {
  const size_t stages = 2 * static_cast<size_t>(tokens) * kXRow * 2 + 2 * static_cast<size_t>(tokens) * 4;
  const size_t staging = static_cast<size_t>(tokens) * kORow * 4;
  return stages > staging ? stages : staging;
}

struct Smem {
  __half *a[2];
  float *inv[2];
  float *o;
};

__device__ __forceinline__ Smem carve(unsigned char *base, int tokens) {
  Smem s;
  s.a[0] = reinterpret_cast<__half *>(base);
  s.a[1] = reinterpret_cast<__half *>(base + tokens * kXRow * 2);
  s.inv[0] = reinterpret_cast<float *>(base + 2 * tokens * kXRow * 2);
  s.inv[1] = s.inv[0] + tokens;
  s.o = reinterpret_cast<float *>(base);
  return s;
}

// Block `blk` of weight `wsel`'s rotated activations and inverse scales into stage `buf`.
__device__ __forceinline__ void stage_block(const DecodeParams &p, const Smem &s, int buf, int wsel, int blk) {
  const int tokens = p.tokens;
  const size_t item = static_cast<size_t>(wsel) * p.nb + blk;
  const __half *src = p.xr + item * tokens * 128;
  for (int i = threadIdx.x; i < tokens * 16; i += kThreads) cp_async16(s.a[buf] + (i >> 4) * kXRow + (i & 15) * 8, src + i * 8);
  const float *isrc = p.inv + item * tokens;
  for (int i = threadIdx.x; i < tokens; i += kThreads) cp_async4(s.inv[buf] + i, isrc + i);
}

// Registers per lane per tile: one word up to K = 4 (a tile is at most 32 words), two above.
template <int K2> constexpr int kRegs = K2 >= 10 ? 2 : 1;

template <int K2> struct PlanOf {
  using type = ignis_trellis::LanePlan;
};
template <> struct PlanOf<10> {
  using type = ignis_trellis::LanePlan3;
};
template <> struct PlanOf<12> {
  using type = ignis_trellis::LanePlan3;
};

template <int K2> __device__ __forceinline__ typename PlanOf<K2>::type plan_for(int lane) {
  if constexpr (K2 >= 10) {
    return ignis_trellis::lane_plan3(K2, lane);
  } else {
    return ignis_trellis::lane_plan(K2, lane);
  }
}

// The lane's words of the warp's eight tiles of one block, tile kt at base + kt * row_words.
template <int K2>
__device__ __forceinline__ void load_block(const uint32_t *base, size_t row_words, uint32_t (&r)[8][kRegs<K2>]) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const uint32_t *t = base + kt * row_words;
    r[kt][0] = lane < words ? load_stream(t + lane) : 0u;
    if constexpr (kRegs<K2> == 2) r[kt][1] = lane + 32 < words ? load_stream(t + lane + 32) : 0u;
  }
}

// One tile's B fragments from the warp's words of it: word w is lane w % 32's register w / 32.
template <int K2>
__device__ __forceinline__ void tile_fragment(const uint32_t (&r)[kRegs<K2>], const typename PlanOf<K2>::type &plan,
                                              uint32_t (&frag)[4]) {
  if constexpr (K2 >= 10) {
    auto word = [&](int w) {
      const uint32_t lo = __shfl_sync(0xFFFFFFFFu, r[0], w & 31);
      const uint32_t hi = __shfl_sync(0xFFFFFFFFu, r[1], w & 31);
      return (w >> 5) != 0 ? hi : lo;
    };
    const uint32_t w0 = word(plan.w0), w1 = word(plan.w1), w2 = word(plan.w2);
    ignis_trellis::decode_fragment3<K2>(w0, w1, w2, plan, frag);
  } else {
    const uint32_t w0 = __shfl_sync(0xFFFFFFFFu, r[0], plan.w0);
    const uint32_t w1 = __shfl_sync(0xFFFFFFFFu, r[0], plan.w1);
    ignis_trellis::decode_fragment<K2>(w0, w1, plan, frag);
  }
}

// acc += (W_block^T . x'_block) * inv for every group of eight tokens, from one stage. The weight
// tile is the MMA's A operand (16 output columns x 16 inputs: the decoded B-order fragment, its
// two n8 halves swapped into A's row order), the tokens its N: one MMA per tile serves eight
// tokens. fp16 accumulation, folded into fp32 times the block's inverse scale every 32 inputs,
// as the prefill route does (the scale keeps a fold's 32 products below fp16's range).
// acc[n] = {(column g, token 8n + 2c), (g, 8n + 2c + 1), (g + 8, 8n + 2c), (g + 8, 8n + 2c + 1)}.
template <int K2, int NG>
__device__ __forceinline__ void compute_block(const uint32_t (&r)[8][kRegs<K2>], const typename PlanOf<K2>::type &plan,
                                              const __half *x, const float *inv, int tokens, float (&acc)[NG][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
  float i0[NG], i1[NG];
#pragma unroll
  for (int n = 0; n < NG; ++n) {
    const int t0 = 8 * n + 2 * c;
    i0[n] = t0 < tokens ? inv[t0] : 0.0f;
    i1[n] = t0 + 1 < tokens ? inv[t0 + 1] : 0.0f;
  }
  uint32_t acc16[NG][2];
#pragma unroll
  for (int n = 0; n < NG; ++n) acc16[n][0] = acc16[n][1] = 0u;
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    uint32_t frag[4];
    tile_fragment<K2>(r[kt], plan, frag);
    const uint32_t a[4] = {frag[0], frag[2], frag[1], frag[3]};
#pragma unroll
    for (int n = 0; n < NG; ++n) {
      const int t = 8 * n + g;
      uint32_t b0 = 0u, b1 = 0u;
      if (t < tokens) {
        const uint2 v = *reinterpret_cast<const uint2 *>(x + t * kXRow + c * 32 + kt * 4);
        b0 = v.x;
        b1 = v.y;
      }
      mma_f16acc(acc16[n], a, b0, b1);
    }
    if (kt & 1) {
#pragma unroll
      for (int n = 0; n < NG; ++n) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc16[n][0]));
        const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc16[n][1]));
        acc[n][0] = fmaf(lo.x, i0[n], acc[n][0]);
        acc[n][1] = fmaf(lo.y, i1[n], acc[n][1]);
        acc[n][2] = fmaf(hi.x, i0[n], acc[n][2]);
        acc[n][3] = fmaf(hi.y, i1[n], acc[n][3]);
        acc16[n][0] = acc16[n][1] = 0u;
      }
    }
  }
}

// The output rotation, svh and the store of one token's 128 columns, lane holding v[4] at
// columns 4 lane .. 4 lane + 3 of block cb.
__device__ __forceinline__ void rotate_out(float (&v)[4], const __half *svh, int cb) {
  warp_hadamard128(v);
  float s[4];
  load_half4(svh + cb * 128 + 4 * (threadIdx.x & 31), s);
#pragma unroll
  for (int q = 0; q < 4; ++q) v[q] *= s[q];
}

// Token t's sums of weight `wsel`, block cb, lane's four columns: every split in split order.
__device__ __forceinline__ void sum_splits(const DecodeParams &p, int t, int wsel, int cb, float (&v)[4]) {
  const size_t outw = static_cast<size_t>(p.weights) * p.out;
  const float *col = p.partials + static_cast<size_t>(t) * outw + static_cast<size_t>(wsel) * p.out + cb * 128 +
                     4 * (threadIdx.x & 31);
  const size_t split_stride = static_cast<size_t>(p.tokens) * outw;
  v[0] = v[1] = v[2] = v[3] = 0.0f;
  for (int s = 0; s < p.splits; ++s) {
    const float4 q = __ldcg(reinterpret_cast<const float4 *>(col + s * split_stride));
    v[0] += q.x;
    v[1] += q.y;
    v[2] += q.z;
    v[3] += q.w;
  }
}

// The last arrival on output block cb: reduce, rotate, svh, epilogue.
__device__ void reduce_block(const DecodeParams &p, int cb) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int t = warp; t < p.tokens; t += kWarps) {
    float g[4];
    sum_splits(p, t, 0, cb, g);
    rotate_out(g, p.w[0].svh, cb);
    if (p.epilogue == kSwiGlu) {
      float u[4];
      sum_splits(p, t, 1, cb, u);
      rotate_out(u, p.w[1].svh, cb);
#pragma unroll
      for (int q = 0; q < 4; ++q) g[q] = ignis_moe::silu(g[q]) * u[q];
    }
    store_bf16x4(p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane, g, p.epilogue == kAdd);
  }
}

template <int NG>
__device__ __forceinline__ void finish(const DecodeParams &p, const Smem &s, const float (&acc)[NG][4], int wsel, int cb,
                                       int split, int *s_last) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int tokens = p.tokens;
  const int col = warp * 16 + g;  // and col + 8
  if (p.splits == 1 && p.epilogue != kSwiGlu) {
    // The CTA holds its block alone: through shared memory, one token per warp.
#pragma unroll
    for (int n = 0; n < NG; ++n) {
      const int t0 = 8 * n + 2 * c;
      if (t0 < tokens) {
        s.o[t0 * kORow + col] = acc[n][0];
        s.o[t0 * kORow + col + 8] = acc[n][2];
      }
      if (t0 + 1 < tokens) {
        s.o[(t0 + 1) * kORow + col] = acc[n][1];
        s.o[(t0 + 1) * kORow + col + 8] = acc[n][3];
      }
    }
    __syncthreads();
    for (int t = warp; t < tokens; t += kWarps) {
      const float4 q = *reinterpret_cast<const float4 *>(s.o + t * kORow + 4 * lane);
      float v[4] = {q.x, q.y, q.z, q.w};
      rotate_out(v, p.w[0].svh, cb);
      store_bf16x4(p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane, v, p.epilogue == kAdd);
    }
    __syncthreads();
    return;
  }
  const size_t outw = static_cast<size_t>(p.weights) * p.out;
  float *part = p.partials + static_cast<size_t>(split) * tokens * outw + static_cast<size_t>(wsel) * p.out + cb * 128 + col;
#pragma unroll
  for (int n = 0; n < NG; ++n) {
    const int t0 = 8 * n + 2 * c;
    if (t0 < tokens) {
      part[t0 * outw] = acc[n][0];
      part[t0 * outw + 8] = acc[n][2];
    }
    if (t0 + 1 < tokens) {
      part[(t0 + 1) * outw] = acc[n][1];
      part[(t0 + 1) * outw + 8] = acc[n][3];
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t expected = static_cast<uint32_t>(p.splits * (p.epilogue == kSwiGlu ? 2 : 1));
    *s_last = atomicAdd(&p.arrivals[cb], 1u) == expected - 1 ? 1 : 0;
  }
  __syncthreads();
  if (*s_last) {
    __threadfence();
    reduce_block(p, cb);
    if (threadIdx.x == 0) p.arrivals[cb] = 0;
  }
}

template <int K2, int NG>
__device__ void unit(const DecodeParams &p, const Smem &s, int wsel, int cb, int split, int *s_last) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const Weight &w = p.w[wsel];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int b0 = split * p.nb / p.splits;
  const int n = (split + 1) * p.nb / p.splits - b0;
  const size_t row_words = static_cast<size_t>(w.stride) * words;
  const size_t block_words = 8 * row_words;
  const uint32_t *base = w.trellis + (static_cast<size_t>(b0) * 8 * w.stride + cb * 8 + warp) * words;
  const typename PlanOf<K2>::type plan = plan_for<K2>(lane);

  uint32_t ra[8][kRegs<K2>], rb[8][kRegs<K2>];
  load_block<K2>(base, row_words, ra);
  if (n > 1) load_block<K2>(base + block_words, row_words, rb);
  grid_dependency_wait();
  stage_block(p, s, 0, wsel, b0);
  cp_async_commit();
  if (n > 1) stage_block(p, s, 1, wsel, b0 + 1);
  cp_async_commit();

  float acc[NG][4];
#pragma unroll
  for (int n = 0; n < NG; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;

  for (int i = 0; i < n; i += 2) {
    cp_async_wait<1>();
    __syncthreads();
    compute_block<K2, NG>(ra, plan, s.a[0], s.inv[0], p.tokens, acc);
    if (i + 2 < n) load_block<K2>(base + (i + 2) * block_words, row_words, ra);
    __syncthreads();
    if (i + 2 < n) stage_block(p, s, 0, wsel, b0 + i + 2);
    cp_async_commit();
    if (i + 1 >= n) break;
    cp_async_wait<1>();
    __syncthreads();
    compute_block<K2, NG>(rb, plan, s.a[1], s.inv[1], p.tokens, acc);
    if (i + 3 < n) load_block<K2>(base + (i + 3) * block_words, row_words, rb);
    __syncthreads();
    if (i + 3 < n) stage_block(p, s, 1, wsel, b0 + i + 3);
    cp_async_commit();
  }
  cp_async_wait<0>();
  __syncthreads();
  finish<NG>(p, s, acc, wsel, cb, split, s_last);
}

// kWide: the kernel that also serves K = 5 and 6, whose two words per lane and tile cost the
// registers of a second CTA per SM; calls of K <= 4 only take the narrow one.
// NG: groups of eight tokens (1, 2, 4 or 8: up to 8, 16, 32, 64 tokens).
template <int NG, bool kWide>
__global__ void __launch_bounds__(kThreads, !kWide && NG <= 2 ? 2 : 1) decode_kernel(const __grid_constant__ DecodeParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ int s_ticket, s_next, s_last;
  const Smem s = carve(smem_raw, p.tokens);
  if (sm_admitted(p.capped, p.admit)) {
    if (threadIdx.x == 0) s_ticket = static_cast<int>(atomicAdd(&p.counters[0], 1u));
    __syncthreads();
    while (s_ticket < p.units) {
      const int ticket = s_ticket;
      if (threadIdx.x == 0) s_next = static_cast<int>(atomicAdd(&p.counters[0], 1u));
      // Splits of one output block are consecutive tickets, and SwiGLU's gate and up of a block
      // are neighbours: the last arrival finds its partial sums fresh in the L2.
      const int split = ticket % p.splits;
      const int rest = ticket / p.splits;
      const int wsel = rest % p.weights;
      const int cb = rest / p.weights;
      const int k2 = p.w[wsel].k2;
      if constexpr (kWide) {
        dispatch_k2(k2, [&](auto k) { unit<decltype(k)::value, NG>(p, s, wsel, cb, split, &s_last); });
      } else {
        dispatch_k2(k2, [&](auto k) {
          constexpr int K2 = decltype(k)::value;
          if constexpr (K2 < 10) {
            unit<K2, NG>(p, s, wsel, cb, split, &s_last);
          } else {
            __trap();
          }
        });
      }
      __syncthreads();
      if (threadIdx.x == 0) s_ticket = s_next;
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&p.counters[1], 1u) == gridDim.x - 1) {
      p.counters[0] = 0;
      p.counters[1] = 0;
    }
  }
}

template <int NG, bool kWide> int32_t prepare_one(DeviceState *st, int cls) {
  const size_t smem = decode_smem(8 * NG);
  cudaError_t err = cudaFuncSetAttribute(decode_kernel<NG, kWide>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem));
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_prepare: decode attributes: ") + cudaGetErrorString(err));
  int ctas = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, decode_kernel<NG, kWide>, kThreads, smem);
  if (err != cudaSuccess || ctas < 1) return fail("ignis_trellis_prepare: the decode kernel does not fit an SM");
  st->decode_ctas[kWide ? 1 : 0][cls] = ctas;
  return 0;
}

template <bool kWide> cudaError_t launch_one(cudaLaunchConfig_t &cfg, int cls, const DecodeParams &p) {
  switch (cls) {
  case 0: return cudaLaunchKernelEx(&cfg, decode_kernel<1, kWide>, p);
  case 1: return cudaLaunchKernelEx(&cfg, decode_kernel<2, kWide>, p);
  case 2: return cudaLaunchKernelEx(&cfg, decode_kernel<4, kWide>, p);
  default: return cudaLaunchKernelEx(&cfg, decode_kernel<8, kWide>, p);
  }
}

}  // namespace

int32_t prepare_decode(DeviceState *st) {
  if (prepare_one<1, false>(st, 0) != 0 || prepare_one<2, false>(st, 1) != 0 || prepare_one<4, false>(st, 2) != 0 ||
      prepare_one<8, false>(st, 3) != 0 || prepare_one<1, true>(st, 0) != 0 || prepare_one<2, true>(st, 1) != 0 ||
      prepare_one<4, true>(st, 2) != 0 || prepare_one<8, true>(st, 3) != 0) {
    return -1;
  }
  return 0;
}

int32_t launch_decode(const DeviceState &st, const DecodeParams &p, cudaStream_t stream) {
  const int cls = decode_class(static_cast<uint32_t>(p.tokens));
  const bool wide = p.w[0].k2 >= 10 || (p.weights == 2 && p.w[1].k2 >= 10);
  const int resident = st.decode_ctas[wide ? 1 : 0][cls] * st.sms;
  // Capped, every SM gets its full share of CTAs and the unadmitted ones leave.
  const int grid = p.capped ? resident : (p.units < resident ? p.units : resident);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = decode_smem(p.tokens);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = ignis::pdl::available() ? 1 : 0;
  const cudaError_t err = wide ? launch_one<true>(cfg, cls, p) : launch_one<false>(cfg, cls, p);
  if (err != cudaSuccess) return fail(std::string("ignis_trellis: decode launch: ") + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_tl
