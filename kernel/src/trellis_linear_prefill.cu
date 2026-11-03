// ignis kernel leaf: the dense trellis linear's prefill route, more than 64 tokens -- OURS
// (kernel/include/ignis_trellis_linear.h), built on our own Flash-Next prefill
// (moe_prefill.cu): trellis tiles staged by cp.async and decoded from shared memory straight into
// MMA B fragments.
//
// One persistent launch takes 128-token x 128-column output tiles by ticket, the columns of a
// row of tiles consecutive so the CTAs running at once share their activation rows in the L2.
// A tile walks all inputs 64 at a time through a four-stage cp.async pipeline: the rotated
// activations (fp16, row-major, from the rotation launch) for its 128 tokens, the four trellis
// tile rows of its 128 columns, and the block's inverse scales. Each warp owns 16 columns and
// all 128 rows: it decodes its tile once per 16 inputs (the next tile row's while this one's
// MMAs run) and runs eight m16n8k16 pairs on it with fp16 accumulation, folded into fp32 (times
// the activations' inverse scale) every 32 inputs -- GeForce runs fp16-accumulate MMA at twice
// the fp32-accumulate rate, and the scale's bound keeps every fold's partial inside fp16
// (trellis_linear_internal.cuh, kScaleExp).
// The epilogue rotates each row's 128 columns back through shared memory, applies svh and stores
// BF16: plain, the residual add, or SwiGLU's up pass, which reads the gate pass's BF16 values
// back from h.
//
// Deterministic: a tile's sums have a fixed order whichever CTA runs it. The CTA that finishes
// last resets the ticket. Capped, as the decode route (trellis_linear_decode.cu).

#include "moe_common.cuh"
#include "trellis_linear_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ignis_tl {
namespace {

using ignis_moe::cp_async_commit;
using ignis_moe::cp_async_wait;
using ignis_moe::ldmatrix_x4;
using ignis_moe::warp_hadamard128;
constexpr int kBM = kPrefillRows;    // tokens per output tile
constexpr int kMB = kBM / 16;        // its m16 blocks
constexpr int kStages = 4;
constexpr int kBK = 64;              // inputs per stage: four 16-input tile rows, two fp16 folds
constexpr int kARow = kBK + 8;       // fp16 per staged activation row, off the bank stride
constexpr int kAStage = kBM * kARow; // fp16
constexpr int kMaxTileWords = 48;    // K = 6
constexpr int kBStage = 4 * 8 * kMaxTileWords;  // u32
constexpr int kERow = 132;           // fp32 per epilogue staging row

struct Stage {
  __half a[kAStage];
  uint32_t b[kBStage];
  float inv[kBM];
};

constexpr size_t kSmem = sizeof(Stage) * kStages;
static_assert(16 * kERow * 4 <= sizeof(Stage) * kStages, "the epilogue staging overlays the stages");

__device__ __forceinline__ void cp_async16_zfill(void *smem, const void *gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(size));
}

// Stage `ks` (inputs 64 ks .. 64 ks + 63) of tile (row0, cb) into `st`.
template <int K2>
__device__ __forceinline__ void load_stage(const PrefillParams &p, Stage &st, int ks, int row0, int cb) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const int k0 = ks * kBK;
  // Activations: kBM rows x 128 bytes, eight 16-byte chunks a row; rows past the call zero.
#pragma unroll
  for (int i = threadIdx.x; i < kBM * 8; i += kThreads) {
    const int r = i >> 3;
    const int q = i & 7;
    const int t = row0 + r;
    const bool valid = t < p.tokens;
    const __half *src = p.xr + static_cast<size_t>(valid ? t : 0) * p.in + k0 + 8 * q;
    cp_async16_zfill(st.a + r * kARow + 8 * q, src, valid);
  }
  // The four tile rows of the tile's eight 16-column tiles: contiguous words each.
#pragma unroll
  for (int i = threadIdx.x; i < 4 * words * 2; i += kThreads) {
    const int kk = i / (words * 2);
    const int chunk = i % (words * 2);
    const uint32_t *src = p.w.trellis + (static_cast<size_t>(k0 / 16 + kk) * p.w.stride + cb * 8) * words + 4 * chunk;
    cp_async16_zfill(st.b + kk * 8 * words + 4 * chunk, src, true);
  }
  // The block's inverse scales of the tile's rows.
  const int blk = k0 / 128;
  if (threadIdx.x < kBM) {
    const int t = row0 + threadIdx.x;
    if (t < p.tokens) {
      cp_async4(st.inv + threadIdx.x, p.inv + static_cast<size_t>(blk) * p.tokens + t);
    } else {
      st.inv[threadIdx.x] = 0.0f;
    }
  }
}

// A lane's decode window in a tile, computed once per output tile.
template <int K2> struct Plan {
  ignis_trellis::LanePlan3 p;
  __device__ __forceinline__ Plan() {
    const int lane = threadIdx.x & 31;
    if constexpr (K2 >= 10) {
      p = ignis_trellis::lane_plan3(K2, lane);
    } else {
      const ignis_trellis::LanePlan q = ignis_trellis::lane_plan(K2, lane);
      p = ignis_trellis::LanePlan3{q.w0, q.w1, 0, q.off};
    }
  }
};

// The warp's B fragments of tile row kk of a stage.
template <int K2>
__device__ __forceinline__ void decode_tile(const Stage &st, int kk, const Plan<K2> &plan, uint32_t (&frag)[4]) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const uint32_t *tile = st.b + (kk * 8 + (threadIdx.x >> 5)) * words;
  if constexpr (K2 >= 10) {
    ignis_trellis::decode_fragment3<K2>(tile[plan.p.w0], tile[plan.p.w1], tile[plan.p.w2], plan.p, frag);
  } else {
    const ignis_trellis::LanePlan q{plan.p.w0, plan.p.w1, plan.p.off};
    ignis_trellis::decode_fragment<K2>(tile[q.w0], tile[q.w1], q, frag);
  }
}

// Tile row kk's A fragments, every m16 block of the stage's activations.
__device__ __forceinline__ void load_a(const Stage &st, int kk, uint32_t (&a)[kMB][4]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int m = 0; m < kMB; ++m) ldmatrix_x4(a[m], st.a + (16 * m + (lane & 15)) * kARow + 16 * kk + (lane >> 4) * 8);
}

// The 16 MMAs of a tile row: every m16 block against the row's fragments.
__device__ __forceinline__ void mma_row(const uint32_t (&a)[kMB][4], const uint32_t (&frag)[4], uint32_t (&acc16)[kMB][2][2]) {
#pragma unroll
  for (int m = 0; m < kMB; ++m) {
    mma_f16acc(acc16[m][0], a[m], frag[0], frag[1]);
    mma_f16acc(acc16[m][1], a[m], frag[2], frag[3]);
  }
}

// The fold: 32 inputs of fp16 partials into fp32, times the block's inverse scale per row, and
// the fp16 accumulators zeroed for the next 32.
__device__ __forceinline__ void fold(const Stage &st, uint32_t (&acc16)[kMB][2][2], float (&acc)[kMB][2][4]) {
  const int g = (threadIdx.x & 31) >> 2;
#pragma unroll
  for (int m = 0; m < kMB; ++m) {
    const float i0 = st.inv[16 * m + g];
    const float i1 = st.inv[16 * m + g + 8];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(&acc16[m][h][0]));
      const float2 hi = __half22float2(*reinterpret_cast<const __half2 *>(&acc16[m][h][1]));
      acc[m][h][0] = fmaf(lo.x, i0, acc[m][h][0]);
      acc[m][h][1] = fmaf(lo.y, i0, acc[m][h][1]);
      acc[m][h][2] = fmaf(hi.x, i1, acc[m][h][2]);
      acc[m][h][3] = fmaf(hi.y, i1, acc[m][h][3]);
      acc16[m][h][0] = acc16[m][h][1] = 0u;
    }
  }
}

template <int K2>
__device__ void tile(const PrefillParams &p, Stage *stages, int tm, int cb) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int row0 = tm * kBM;
  const int steps = p.in / kBK;
  float acc[kMB][2][4];
  uint32_t acc16[2][kMB][2][2];
#pragma unroll
  for (int m = 0; m < kMB; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[m][h][i] = 0.0f;
      acc16[0][m][h][0] = acc16[0][m][h][1] = acc16[1][m][h][0] = acc16[1][m][h][1] = 0u;
    }

  // The pipeline: at the top of step ks, stages ks and ks + 1 are in shared memory and visible,
  // ks + 2 is in flight, and ks + 3 goes into the buffer ks - 1 left. Each tile row's fragments
  // are decoded while the previous row's MMAs run (the next stage's first row at the end of a
  // step), and the first fold of a step runs under its last rows' MMAs. One barrier per step.
  grid_dependency_wait();
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < steps) load_stage<K2>(p, stages[s], s, row0, cb);
    cp_async_commit();
  }
  cp_async_wait<kStages - 3>();
  __syncthreads();
  const Plan<K2> plan;
  uint32_t next[4];
  decode_tile<K2>(stages[0], 0, plan, next);
  for (int ks = 0; ks < steps; ++ks) {
    const Stage &st = stages[ks % kStages];
    if (ks + kStages - 1 < steps) load_stage<K2>(p, stages[(ks + kStages - 1) % kStages], ks + kStages - 1, row0, cb);
    cp_async_commit();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const uint32_t frag[4] = {next[0], next[1], next[2], next[3]};
      uint32_t a[kMB][4];
      load_a(st, kk, a);
      if (kk < 3) {
        decode_tile<K2>(st, kk + 1, plan, next);
      } else if (ks + 1 < steps) {
        decode_tile<K2>(stages[(ks + 1) % kStages], 0, plan, next);
      }
      mma_row(a, frag, acc16[kk >> 1]);
      if (kk == 2) fold(st, acc16[0], acc);
    }
    fold(st, acc16[1], acc);
    cp_async_wait<kStages - 3>();
    __syncthreads();
  }
  cp_async_wait<0>();
  __syncthreads();

  // Epilogue, one 16-row block at a time through shared memory.
  float *e = reinterpret_cast<float *>(stages);
#pragma unroll
  for (int m = 0; m < kMB; ++m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = warp * 16 + 8 * h + 2 * c;
      *reinterpret_cast<float2 *>(e + g * kERow + col) = make_float2(acc[m][h][0], acc[m][h][1]);
      *reinterpret_cast<float2 *>(e + (g + 8) * kERow + col) = make_float2(acc[m][h][2], acc[m][h][3]);
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int r = 2 * warp + rr;
      const int t = row0 + 16 * m + r;
      if (t < p.tokens) {
        const float4 q = *reinterpret_cast<const float4 *>(e + r * kERow + 4 * lane);
        float v[4] = {q.x, q.y, q.z, q.w};
        warp_hadamard128(v);
        float s[4];
        load_half4(p.w.svh + cb * 128 + 4 * lane, s);
        __nv_bfloat16 *y = p.y + static_cast<size_t>(t) * p.ldy + cb * 128 + 4 * lane;
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= s[i];
        if (p.epilogue == kSwiGluUp) {
          float gate[4];
          load_bf16x4(y, gate);
#pragma unroll
          for (int i = 0; i < 4; ++i) v[i] = ignis_moe::silu(gate[i]) * v[i];
        }
        store_bf16x4(y, v, p.epilogue == kAdd);
      }
    }
    __syncthreads();
  }
}

template <int K2>
__global__ void __launch_bounds__(kThreads, 1) prefill_kernel(const __grid_constant__ PrefillParams p) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __shared__ int s_ticket, s_next;
  Stage *stages = reinterpret_cast<Stage *>(smem_raw);
  if (sm_admitted(p.capped, p.admit)) {
    if (threadIdx.x == 0) s_ticket = static_cast<int>(atomicAdd(&p.counters[0], 1u));
    __syncthreads();
    const int cbs = p.out / 128;
    while (s_ticket < p.tiles) {
      const int ticket = s_ticket;
      if (threadIdx.x == 0) s_next = static_cast<int>(atomicAdd(&p.counters[0], 1u));
      tile<K2>(p, stages, ticket / cbs, ticket % cbs);
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

template <int K2> int32_t prepare_one(int *ctas) {
  cudaError_t err = cudaFuncSetAttribute(prefill_kernel<K2>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmem));
  if (err != cudaSuccess) return fail(std::string("ignis_trellis_prepare: prefill attributes: ") + cudaGetErrorString(err));
  int n = 0;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, prefill_kernel<K2>, kThreads, kSmem);
  if (err != cudaSuccess || n < 1) return fail("ignis_trellis_prepare: the prefill kernel does not fit an SM");
  *ctas = *ctas == 0 || n < *ctas ? n : *ctas;
  return 0;
}

template <int K2> cudaError_t launch_one(cudaLaunchConfig_t &cfg, const PrefillParams &p) {
  return cudaLaunchKernelEx(&cfg, prefill_kernel<K2>, p);
}

}  // namespace

int32_t prepare_prefill(DeviceState *st) {
  int ctas = 0;
  if (prepare_one<4>(&ctas) != 0 || prepare_one<5>(&ctas) != 0 || prepare_one<6>(&ctas) != 0 ||
      prepare_one<8>(&ctas) != 0 || prepare_one<10>(&ctas) != 0 || prepare_one<12>(&ctas) != 0) {
    return -1;
  }
  st->prefill_ctas = ctas;
  return 0;
}

int32_t launch_prefill(const DeviceState &st, const PrefillParams &p, cudaStream_t stream) {
  const int resident = st.prefill_ctas * st.sms;
  const int grid = p.capped ? resident : (p.tiles < resident ? p.tiles : resident);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = kSmem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = ignis::pdl::available() ? 1 : 0;
  cudaError_t err = cudaSuccess;
  switch (p.w.k2) {
  case 4: err = launch_one<4>(cfg, p); break;
  case 5: err = launch_one<5>(cfg, p); break;
  case 6: err = launch_one<6>(cfg, p); break;
  case 8: err = launch_one<8>(cfg, p); break;
  case 10: err = launch_one<10>(cfg, p); break;
  default: err = launch_one<12>(cfg, p); break;
  }
  if (err != cudaSuccess) return fail(std::string("ignis_trellis: prefill launch: ") + cudaGetErrorString(err));
  return 0;
}

}  // namespace ignis_tl
