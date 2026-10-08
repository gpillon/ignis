// ignis kernel leaf: Flash-Next's routed experts for 1..4 decode tokens, each SM streaming its
// work items' weights through shared memory -- OURS (kernel/include/ignis_moe.h: a workspace
// whose decode_route is IGNIS_MOE_DECODE_TICKETS; GitHub #306, the decode fusion roadmap's
// step 8).
//
// The register ticket kernel (moe_decode.cu) holds a unit's weights in registers, so a CTA asks
// DRAM for its next unit only after it has multiplied the current one: the card reads in waves
// with the bus idle between them, and every unit pays its operand loads and rotation in full
// before it multiplies. Here one CTA per SM runs two roles:
//
//   producer  one warp takes the work items by ticket (one ahead), reads their slots and issues
//             every byte an item needs -- its trellis tiles, its channel scales, its activations
//             or the gate/up sums it consumes -- as cp.async copies into one of two stages of
//             shared memory, then signals the stage's `full` barrier; it refills a stage as soon
//             as the compute warps release it, so the next item's bytes are in flight while the
//             current one is multiplied.
//   compute   sixteen warps take the stages in order: prepare the item's fp16 operand from the
//             stage, decode the trellis tiles from shared memory straight into m16n8k16 B
//             fragments (16 tiles per warp), release the stage, and add the item's result.
//
// Work items, 256 tiles (65,536 weights) each, all gate/up items first, then all down items:
//
//   gate/up (expert u, block b in 0..4, k-split s in 0..9)
//       gate block b and up block b (256 columns, one 16-column tile per warp) over the inputs
//       256 s .. 256 s + 255: the tokens' inputs times suh, the 128-wide Hadamard, one
//       power-of-two scale per token into fp16; the pre-rotation sums go into the int64
//       fixed-point gate/up accumulator, and the item counts one arrival on (u, b).
//   down (expert u, h block j in 0..4, column block c in 0..4)
//       its producer waits for (u, j)'s ten arrivals and stages gate and up block j's sums; the
//       compute warps rotate both, apply svh and SwiGLU (h's block j), rotate h_j o suh_down into
//       fp16, multiply the block's 8 k-tiles against 512 down columns (two tiles per warp), apply
//       the output Hadamard per 128 columns, svh and each selecting token's routing weight, and
//       add the result into the fixed-point output accumulator. Each of h block j's five readers
//       adds 16 to (u, j)'s counter once it has its copy; the fifth zeroes the block's sums and
//       the counter.
//
// The reduction is order-independent where it crosses items (integer sums) and fixed inside an
// item, so the result is deterministic. Against the register kernel only where partial sums
// round differs: the gate/up k-split is 256 inputs instead of 640 (an fp32 MMA chain of 16
// k-tiles instead of 40 before the conversion to fixed point), and down is summed over h's five
// blocks in fixed point instead of in one fp32 chain over 640 inputs, the output Hadamard taken
// per block (it is linear). Each fp16 operand rounds as before: the scale is a power of two, now
// per 256 (gate/up) or 128 (down) inputs instead of per 640, which moves no rounding of a normal
// number.
//
// Deadlock-free: a CTA takes tickets in increasing order and works them in order, every gate/up
// ticket precedes every down ticket, and nothing a gate/up item does waits on another CTA -- so
// a down item's producer waits only on gate/up items held by running CTAs that reach them
// first. The last CTA out resets the ticket counter, so the launch replays from a CUDA graph.

#include "moe_decode_common.cuh"
#include "moe_trace.h"
#include "trellis_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ignis_moe {
namespace {

constexpr int kComputeWarps = 16;
constexpr int kComputeThreads = kComputeWarps * 32;
constexpr int kThreads = kComputeThreads + 32;  // + the producer warp
constexpr int kStages = 2;
constexpr int kItemTiles = 256;
constexpr int kGuSplit = 256;                              // gate/up item inputs
constexpr int kGuSplits = kHidden / kGuSplit;              // 10
constexpr int kGuKTiles = kGuSplit / 16;                   // 16
constexpr int kDnCols = 512;                               // down item columns
constexpr int kDnColBlocks = kHidden / kDnCols;            // 5
constexpr int kDnKTiles = 128 / 16;                        // 8: one h block
constexpr int kGuItems = kGateUpBlocks * kGuSplits;        // per expert: 50
constexpr int kDnItems = kGateUpBlocks * kDnColBlocks;     // per expert: 25
constexpr int kAStride = kGuSplit + 8;                     // fp16 per A row, padded off the bank stride
constexpr uint32_t kGuArrivals = kGuSplits;                // gate/up items per (u, b)
constexpr uint32_t kReaderStep = 16;                       // a down reader's count on (u, b)
constexpr uint32_t kLastReader = kGuArrivals + (kDnColBlocks - 1) * kReaderStep;
constexpr int kMaxWeightBytes = kItemTiles * 16 * 8;       // 32 KiB: 256 tiles at K = 4

static_assert(kGuKTiles * 16 == kGuSplit && kComputeWarps * kGuKTiles == kItemTiles, "gate/up item shape");
static_assert(kDnCols / 16 == 2 * kComputeWarps && kComputeWarps * 2 * kDnKTiles == kItemTiles, "down item shape");
static_assert(kStagedMaxTokens * kTopK <= kDecodeMaxUnique, "unique experts of a call");

// Bytes past the weights in a stage: a gate/up item's suh slice and x slice, or a down item's
// gate and up sums, suh_down block, svh_down columns and the gate/up svh of its h block.
__host__ __device__ constexpr int stage_extra_bytes(int tokens) { return 2048 * tokens + 1792; }
__host__ __device__ constexpr int stage_bytes(int tokens) {
  return (kMaxWeightBytes + stage_extra_bytes(tokens) + 127) / 128 * 128;
}
// Dynamic shared memory of a launch for `tokens`: the stages and A. (A down item's fp32 output
// exchange reuses its own stage's weights once they are multiplied.)
__host__ __device__ constexpr int dynamic_bytes(int tokens) {
  return kStages * stage_bytes(tokens) + (tokens * kAStride * 2 + 127) / 128 * 128;
}

// Offsets inside a stage's extras.
constexpr int kGuSuh = 0;  // fp16 [256]; x bf16 [tokens][256] follows at 512
constexpr int kGuX = 512;
__host__ __device__ constexpr int dn_suh(int tokens) { return 2048 * tokens; }  // sums int64 [tokens][256] first
__host__ __device__ constexpr int dn_svh(int tokens) { return 2048 * tokens + 256; }
__host__ __device__ constexpr int dn_svh_gu(int tokens) { return 2048 * tokens + 256 + 1024; }

enum Kind : int { kGateUp = 0, kDown = 1, kEnd = 2 };

struct Header {
  int kind;
  int ticket;
  int u;
  int block;  // gate/up: b; down: the h block j
  int sub;    // gate/up: the k-split s; down: the column block c
  int k2;
  unsigned long long issued;  // kTrace: when the producer issued the copies
};

struct Shared {
  int n_unique;
  int unique_id[kDecodeMaxUnique];
  uint32_t unique_sel[kDecodeMaxUnique];
  float unique_w[kDecodeMaxUnique][kDecodeMaxTokens];
  int slot_of[kDecodeMaxTokens * kTopK];
  int first[kDecodeMaxTokens * kTopK];
  unsigned long long full[kStages];
  unsigned long long empty[kStages];
  Header header[kStages];
  float scale[kStagedMaxTokens];
};

// The card's per-block shared-memory ceiling (sm_120: 99 KiB) holds the widest launch.
static_assert(dynamic_bytes(kStagedMaxTokens) + sizeof(Shared) <= 99 * 1024, "shared memory of a 4-token launch");

struct Params {
  const __nv_bfloat16 *x;
  int tokens;
  const int32_t *ids;
  const float *weights;
  const ignis_moe_slot *slots;
  DecodeCounters *counters;
  int cap;             // the workspace's decode tokens: the row stride of gate_up
  long long *gate_up;  // int64 [unique][cap][1280], fixed point, zero between calls
  long long *acc;
  unsigned long long *trace;  // kTrace only (moe_trace.h)
};

// ---- shared-memory barriers and copies ----------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(unsigned long long *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(unsigned long long *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
// Arrives on `bar` once every cp.async the thread issued before it has landed.
__device__ __forceinline__ void mbar_arrive_on_copies(unsigned long long *bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long *bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT;\n"
      "}\n" ::"r"(smem_addr(bar)),
      "r"(parity)
      : "memory");
}
// The compute warps' own barrier (the producer warp never joins it).
__device__ __forceinline__ void compute_sync() {
  asm volatile("bar.sync 1, %0;\n" ::"n"(kComputeThreads) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// `chunks` 16-byte copies, lane-strided: chunk i from src(i) to dst + 16 i.
template <typename Src>
__device__ __forceinline__ void copy_chunks(char *dst, int chunks, Src src) {
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < chunks; i += 32) cp_async16(dst + 16 * i, src(i));
}

// ---- the producer ---------------------------------------------------------------------------

// Stage item `h` (its header already filled in) for `tokens` tokens: the weights and extras.
template <int K2>
__device__ void stage_item(const Shared &s, const Params &p, const Header &h, char *stage) {
  constexpr int tile_bytes = 16 * K2;
  const int tokens = p.tokens;
  char *extra = stage + kMaxWeightBytes;
  if (h.kind == kGateUp) {
    const ignis_moe_slot slot = load_slot(p.slots, s.unique_id[h.u], IGNIS_MOE_PROJ_GATE_UP);
    const char *rec = static_cast<const char *>(slot.record);
    const RecordPlanes planes = record_planes(slot, kHidden, kGateUpOut, K2);
    // 16 rows (k-tiles), each two runs of 8 tiles: gate block b, then up block b.
    constexpr int run_chunks = 8 * tile_bytes / 16;
    constexpr int row_chunks = 2 * run_chunks;
    const int kt0 = h.sub * kGuKTiles;
    copy_chunks(stage, kGuKTiles * row_chunks, [&](int i) {
      const int r = i / row_chunks, j = i % row_chunks;
      const int run = j / run_chunks, off = j % run_chunks;
      const int tile = (kt0 + r) * (kGateUpOut / 16) + run * (kInter / 16) + 8 * h.block;
      return rec + static_cast<size_t>(tile) * tile_bytes + 16 * off;
    });
    const char *suh = reinterpret_cast<const char *>(planes.suh + h.sub * kGuSplit);
    copy_chunks(extra + kGuSuh, kGuSplit * 2 / 16, [&](int i) { return suh + 16 * i; });
    copy_chunks(extra + kGuX, tokens * kGuSplit * 2 / 16, [&](int i) {
      const int t = i / (kGuSplit * 2 / 16), off = i % (kGuSplit * 2 / 16);
      return reinterpret_cast<const char *>(p.x + static_cast<size_t>(t) * kHidden + h.sub * kGuSplit) + 16 * off;
    });
  } else {
    const ignis_moe_slot gu = load_slot(p.slots, s.unique_id[h.u], IGNIS_MOE_PROJ_GATE_UP);
    const ignis_moe_slot dn = load_slot(p.slots, s.unique_id[h.u], IGNIS_MOE_PROJ_DOWN);
    const char *rec = static_cast<const char *>(dn.record);
    const RecordPlanes planes = record_planes(dn, kInter, kHidden, K2);
    const RecordPlanes gu_planes = record_planes(gu, kHidden, kGateUpOut, gu.k2);
    // 8 rows (h block j's k-tiles), each one run of 32 tiles (column block c).
    constexpr int row_chunks = 32 * tile_bytes / 16;
    copy_chunks(stage, kDnKTiles * row_chunks, [&](int i) {
      const int r = i / row_chunks, off = i % row_chunks;
      const int tile = (h.block * kDnKTiles + r) * (kHidden / 16) + 32 * h.sub;
      return rec + static_cast<size_t>(tile) * tile_bytes + 16 * off;
    });
    // Block j's gate and up sums: int64 [tokens][gate 128 | up 128].
    copy_chunks(extra, tokens * 2 * 128 * 8 / 16, [&](int i) {
      const int t = i / 128, half = i / 64 % 2, off = i % 64;
      const long long *row = p.gate_up + (static_cast<size_t>(h.u) * p.cap + t) * kGateUpOut;
      return reinterpret_cast<const char *>(row + half * kInter + 128 * h.block) + 16 * off;
    });
    const char *suh = reinterpret_cast<const char *>(planes.suh + 128 * h.block);
    copy_chunks(extra + dn_suh(tokens), 128 * 2 / 16, [&](int i) { return suh + 16 * i; });
    const char *svh = reinterpret_cast<const char *>(planes.svh + kDnCols * h.sub);
    copy_chunks(extra + dn_svh(tokens), kDnCols * 2 / 16, [&](int i) { return svh + 16 * i; });
    copy_chunks(extra + dn_svh_gu(tokens), 2 * 128 * 2 / 16, [&](int i) {
      const int half = i / 16, off = i % 16;
      return reinterpret_cast<const char *>(gu_planes.svh + half * kInter + 128 * h.block) + 16 * off;
    });
  }
}

template <bool kTrace>
__device__ void producer(Shared &s, const Params &p, char *stages, int stage_stride, int n_gate_up, int n_total) {
  const int lane = threadIdx.x & 31;
  // The first ticket is the CTA's own; each later one is fetched an item before it is needed
  // (lane 0 holds it until then), so the atomic's round trip overlaps an item's copies.
  int ticket = static_cast<int>(blockIdx.x);
  int pending = 0;
  if (lane == 0) pending = static_cast<int>(atomicAdd(&p.counters->ticket, 1u) + gridDim.x);
  for (int n = 0;; ++n) {
    const int st = n % kStages;
    if (n >= kStages) mbar_wait(&s.empty[st], static_cast<uint32_t>((n / kStages - 1) & 1));
    Header h{};
    h.ticket = ticket;
    if (ticket >= n_total) {
      h.kind = kEnd;
    } else if (ticket < n_gate_up) {
      h.kind = kGateUp;
      h.u = ticket / kGuItems;
      h.block = ticket / kGuSplits % kGateUpBlocks;
      h.sub = ticket % kGuSplits;
    } else {
      const int d = ticket - n_gate_up;
      h.kind = kDown;
      h.u = d / kDnItems;
      h.block = d / kDnColBlocks % kGateUpBlocks;
      h.sub = d % kDnColBlocks;
    }
    if (h.kind != kEnd) {
      const int proj = h.kind == kGateUp ? IGNIS_MOE_PROJ_GATE_UP : IGNIS_MOE_PROJ_DOWN;
      h.k2 = static_cast<int>(load_slot(p.slots, s.unique_id[h.u], proj).k2);
      if (h.kind == kDown) {
        // h block j is whole once its ten gate/up items have arrived.
        if (lane == 0) {
          const uint32_t *count = &p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block];
          while (ld_acquire(count) < kGuArrivals) __nanosleep(32);
        }
        __syncwarp();
      }
      if constexpr (kTrace) h.issued = global_ns();
      char *stage = stages + static_cast<size_t>(st) * stage_stride;
      dispatch_k2(static_cast<uint32_t>(h.k2), [&](auto k2) { stage_item<decltype(k2)::value>(s, p, h, stage); });
    }
    if (lane == 0) {
      s.header[st] = h;
      mbar_arrive(&s.full[st]);
    }
    mbar_arrive_on_copies(&s.full[st]);
    if (h.kind == kEnd) return;
    int fetched = 0;
    if (lane == 0) fetched = static_cast<int>(atomicAdd(&p.counters->ticket, 1u) + gridDim.x);
    ticket = __shfl_sync(0xFFFFFFFFu, pending, 0);
    pending = fetched;
  }
}

// ---- the compute warps ----------------------------------------------------------------------

// acc[n] += A(tokens x 16 kt) . W(16 kt x 16) for the warp's column tiles, kt k-tiles from row
// `kt0` of A; tile (r, n) of the stage at word (r * row_tiles + col[n]) * tile_words.
template <int K2, int NT, int KT>
__device__ __forceinline__ void mma_stage(const uint32_t *stage, int row_tiles, const int (&col)[NT], const __half *a,
                                          int tokens, float (&acc)[NT][2][4]) {
  constexpr int words = ignis_trellis::tile_words(K2);
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
  const bool row = g < tokens;
  const ignis_trellis::LanePlan plan = ignis_trellis::lane_plan(K2, lane);
#pragma unroll
  for (int r = 0; r < KT; ++r) {
    uint32_t af[4] = {0u, 0u, 0u, 0u};
    if (row) {
      af[0] = *reinterpret_cast<const uint32_t *>(&a[g * kAStride + r * 16 + 2 * c]);
      af[2] = *reinterpret_cast<const uint32_t *>(&a[g * kAStride + r * 16 + 2 * c + 8]);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const uint32_t *tile = stage + (r * row_tiles + col[n]) * words;
      uint32_t frag[4];
      ignis_trellis::decode_fragment<K2>(tile[plan.w0], tile[plan.w1], plan, frag);
      mma_f16(acc[n][0], af, frag[0], frag[1]);
      mma_f16(acc[n][1], af, frag[2], frag[3]);
    }
  }
}

// The gate/up item's operand: warp t < tokens rotates its token's 256 inputs (x o suh, two
// 128-wide Hadamards) and writes them into A as fp16 under one power-of-two scale.
__device__ void prepare_gate_up(Shared &s, const char *extra, int tokens, __half *a) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp < tokens) {
    const int t = warp;
    const __half *suh = reinterpret_cast<const __half *>(extra + kGuSuh);
    const __nv_bfloat16 *x = reinterpret_cast<const __nv_bfloat16 *>(extra + kGuX) + t * kGuSplit;
    float v[2][4];
    float m = 0.0f;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int k = blk * 128 + 4 * lane + q;
        v[blk][q] = __bfloat162float(x[k]) * __half2float(suh[k]);
      }
      warp_hadamard128(v[blk]);
#pragma unroll
      for (int q = 0; q < 4; ++q) m = fmaxf(m, fabsf(v[blk][q]));
    }
    const float scale = fp16_operand_scale(warp_max(m));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
      for (int q = 0; q < 4; ++q) a[t * kAStride + blk * 128 + 4 * lane + q] = __float2half_rn(v[blk][q] * scale);
    }
    if (lane == 0) s.scale[t] = scale;
  }
}

// The down item's operand: warp t < tokens reads gate and up block j's sums, rotates both,
// applies svh and SwiGLU (h's block j), then h_j o suh_down, rotated, into A as fp16 under one
// power-of-two scale.
__device__ void prepare_down(Shared &s, const char *extra, int tokens, __half *a) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp < tokens) {
    const int t = warp;
    const long long *sums = reinterpret_cast<const long long *>(extra) + t * 256;
    const __half *svh_gu = reinterpret_cast<const __half *>(extra + dn_svh_gu(tokens));
    const __half *suh = reinterpret_cast<const __half *>(extra + dn_suh(tokens));
    float gate[4], up[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      gate[q] = from_fixed(sums[4 * lane + q]);
      up[q] = from_fixed(sums[128 + 4 * lane + q]);
    }
    warp_hadamard128(gate);
    warp_hadamard128(up);
    float v[4];
    float m = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float g = gate[q] * __half2float(svh_gu[4 * lane + q]);
      const float u = up[q] * __half2float(svh_gu[128 + 4 * lane + q]);
      v[q] = silu(g) * u * __half2float(suh[4 * lane + q]);
    }
    warp_hadamard128(v);
#pragma unroll
    for (int q = 0; q < 4; ++q) m = fmaxf(m, fabsf(v[q]));
    const float scale = fp16_operand_scale(warp_max(m));
#pragma unroll
    for (int q = 0; q < 4; ++q) a[t * kAStride + 4 * lane + q] = __float2half_rn(v[q] * scale);
    if (lane == 0) s.scale[t] = scale;
  }
}

template <int K2>
__device__ void gate_up_item(Shared &s, const Params &p, const Header &h, const char *stage, __half *a,
                             unsigned long long *empty, UnitStamps &st, bool trace) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  prepare_gate_up(s, stage + kMaxWeightBytes, tokens, a);
  compute_sync();
  if (trace) st.ready = global_ns();
  float acc[1][2][4] = {};
  const int col[1] = {warp};
  mma_stage<K2, 1, kGuKTiles>(reinterpret_cast<const uint32_t *>(stage), 2 * 8, col, a, tokens, acc);
  __syncwarp();
  if (lane == 0) mbar_arrive(empty);  // the stage is free for the producer
  if (trace) st.mma = global_ns();
  const int g = lane >> 2;
  const int c = lane & 3;
  if (g < tokens) {
    const float inv = 1.0f / s.scale[g];
    const int col0 = (warp < 8 ? 128 * h.block + 16 * warp : kInter + 128 * h.block + 16 * (warp - 8)) + 2 * c;
    long long *dst = p.gate_up + (static_cast<size_t>(h.u) * p.cap + g) * kGateUpOut + col0;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      add_fixed(dst + hf * 8, acc[0][hf][0] * inv);
      add_fixed(dst + hf * 8 + 1, acc[0][hf][1] * inv);
    }
  }
  __threadfence();
  compute_sync();
  if (threadIdx.x == 0) atomicAdd(&p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block], 1u);
}

template <int K2>
__device__ void down_item(Shared &s, const Params &p, const Header &h, char *stage, __half *a,
                          unsigned long long *empty, UnitStamps &st, bool trace) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  uint32_t *count = &p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block];
  // This reader has its copy of the block's sums (the stage is full): count it now, read the
  // count after the multiply.
  uint32_t before = 0;
  if (threadIdx.x == 0) before = atomicAdd(count, kReaderStep);
  prepare_down(s, stage + kMaxWeightBytes, tokens, a);
  compute_sync();
  if (trace) st.ready = global_ns();
  float acc[2][2][4] = {};
  const int col[2] = {warp, warp + kComputeWarps};
  mma_stage<K2, 2, kDnKTiles>(reinterpret_cast<const uint32_t *>(stage), 2 * kComputeWarps, col, a, tokens, acc);
  if (trace) st.mma = global_ns();
  // The pre-rotation sums are exchanged through the stage's weights, multiplied by every warp now;
  // the stage goes back to the producer once they and its svh_down have been read.
  compute_sync();
  float *y = reinterpret_cast<float *>(stage);
  const __half *svh = reinterpret_cast<const __half *>(stage + kMaxWeightBytes + dn_svh(tokens));
  const int g = lane >> 2;
  const int c = lane & 3;
  if (g < tokens) {
    const float inv = 1.0f / s.scale[g];
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int cc = col[n] * 16 + hf * 8 + 2 * c;
        y[g * kDnCols + cc] = acc[n][hf][0] * inv;
        y[g * kDnCols + cc + 1] = acc[n][hf][1] * inv;
      }
    }
  }
  compute_sync();
  // Output rotation per 128 columns: warp (t, block) for every token that selected u.
  if (warp < 4 * tokens) {
    const int t = warp / 4;
    const int blk = warp % 4;
    if (s.unique_sel[h.u] >> t & 1u) {
      float v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = y[t * kDnCols + blk * 128 + 4 * lane + q];
      warp_hadamard128(v);
      const float wt = s.unique_w[h.u][t];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int cc = blk * 128 + 4 * lane + q;
        add_fixed(p.acc + static_cast<size_t>(t) * kHidden + kDnCols * h.sub + cc,
                  wt * (v[q] * __half2float(svh[cc])));
      }
    }
  }
  // The fifth reader of h block j zeroes its sums and the count for the next call.
  if (warp == 0) {
    before = __shfl_sync(0xFFFFFFFFu, before, 0);
    if (before == kLastReader) {
      for (int i = lane; i < tokens * 256; i += 32) {
        const int t = i / 256, j = i % 256;
        p.gate_up[(static_cast<size_t>(h.u) * p.cap + t) * kGateUpOut + (j < 128 ? 0 : kInter) + 128 * h.block + j % 128] = 0;
      }
      if (lane == 0) *count = 0u;
    }
  }
  compute_sync();
  if (lane == 0) mbar_arrive(empty);
}

template <bool kTrace>
__global__ void __launch_bounds__(kThreads, 1) experts_decode_staged_kernel(Params p, int stage_stride) {
  __shared__ Shared s;
  extern __shared__ __align__(128) char dyn[];
  char *stages = dyn;
  __half *a = reinterpret_cast<__half *>(dyn + kStages * stage_stride);
  if constexpr (kTrace) {
    if (threadIdx.x == 0) trace_cta(p.trace)[0] = global_ns();
  }
  if (threadIdx.x == 0) {
    for (int i = 0; i < kStages; ++i) {
      mbar_init(&s.full[i], 33);  // the producer's 32 lanes' copies + its lane 0's header
      mbar_init(&s.empty[i], kComputeWarps);
    }
  }
  build_unique(s, p.ids, p.weights, p.tokens);  // ends with __syncthreads
  const int n_gate_up = s.n_unique * kGuItems;
  const int n_total = n_gate_up + s.n_unique * kDnItems;
  const int warp = threadIdx.x >> 5;
  if (warp == kComputeWarps) {
    if constexpr (kTrace) {
      if ((threadIdx.x & 31) == 0) trace_cta(p.trace)[1] = global_ns();
    }
    producer<kTrace>(s, p, stages, stage_stride, n_gate_up, n_total);
  } else {
    for (int n = 0;; ++n) {
      const int st = n % kStages;
      UnitStamps stamps;
      if constexpr (kTrace) stamps.begin = global_ns();
      mbar_wait(&s.full[st], static_cast<uint32_t>((n / kStages) & 1));
      const Header h = s.header[st];
      if (h.kind == kEnd) break;
      if constexpr (kTrace) stamps.extra = global_ns() - stamps.begin;
      char *stage = stages + static_cast<size_t>(st) * stage_stride;
      dispatch_k2(static_cast<uint32_t>(h.k2), [&](auto k2) {
        if (h.kind == kGateUp) {
          gate_up_item<decltype(k2)::value>(s, p, h, stage, a, &s.empty[st], stamps, kTrace);
        } else {
          down_item<decltype(k2)::value>(s, p, h, stage, a, &s.empty[st], stamps, kTrace);
        }
      });
      if constexpr (kTrace) {
        stamps.end = global_ns();
        if (threadIdx.x == 0 && h.ticket < kTraceUnits) {
          trace_unit(p.trace, h.ticket, h.kind, h.u, h.block, h.sub, stamps);
          p.trace[static_cast<size_t>(h.ticket) * kTraceWords + 7] = h.issued;
        }
      }
    }
  }
  __syncthreads();
  // The last CTA out leaves the ticket counter as it found it, so the launch replays.
  if (threadIdx.x == 0) {
    if constexpr (kTrace) {
      trace_cta(p.trace)[2] = global_ns();
      trace_cta(p.trace)[3] = sm_id();
    }
    __threadfence();
    if (atomicAdd(&p.counters->done, 1u) == gridDim.x - 1) {
      p.counters->ticket = 0;
      p.counters->done = 0;
      __threadfence();
    }
  }
}

}  // namespace

int32_t prepare_decode_staged(int *grid) {
  const int bytes = dynamic_bytes(kStagedMaxTokens);
  int device = 0, sms = 0, per_sm = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(experts_decode_staged_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  }
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(experts_decode_staged_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  }
  if (err == cudaSuccess) {
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, experts_decode_staged_kernel<false>, kThreads, bytes);
  }
  if (err != cudaSuccess) return fail(std::string("ignis_moe_prepare (staged decode): ") + cudaGetErrorString(err));
  if (per_sm <= 0) return fail("ignis_moe_prepare (staged decode): the staged decode kernel fits no CTA on an SM");
  *grid = sms * per_sm;
  return 0;
}

int32_t decode_staged(int grid, const __nv_bfloat16 *x, int tokens, const int32_t *ids, const float *weights,
                      const ignis_moe_slot *slots, DecodeCounters *counters, int cap, long long *gate_up, long long *acc,
                      unsigned long long *trace, cudaStream_t stream) {
  Params p;
  p.x = x;
  p.tokens = tokens;
  p.ids = ids;
  p.weights = weights;
  p.slots = slots;
  p.counters = counters;
  p.cap = cap;
  p.gate_up = gate_up;
  p.acc = acc;
  p.trace = trace;
  const int stride = stage_bytes(tokens);
  const int bytes = dynamic_bytes(tokens);
  if (trace != nullptr) {
    experts_decode_staged_kernel<true><<<grid, kThreads, bytes, stream>>>(p, stride);
  } else {
    experts_decode_staged_kernel<false><<<grid, kThreads, bytes, stream>>>(p, stride);
  }
  return check_launch("ignis_moe_experts_decode (staged)");
}

}  // namespace ignis_moe
