// ignis kernel leaf: Flash-Next's routed experts for 1..4 decode tokens, each SM streaming its
// work items' weights through shared memory -- OURS (kernel/include/ignis_moe.h: a workspace
// whose decode_route is IGNIS_MOE_DECODE_TICKETS; GitHub #306, the decode fusion roadmap's
// step 8).
//
// The register ticket kernel (moe_decode.cu) holds a unit's weights in registers, so a CTA asks
// DRAM for its next unit only after it has multiplied the current one: the card reads in waves
// with the bus idle between them, and every unit pays its operand loads, its rotation and its
// reduction in full on the warps that multiply. Here one CTA per SM runs three roles, each on
// its own warps, handing work items along through shared memory:
//
//   producer  one warp takes the work items by ticket (one ahead), reads their slots and issues
//             every byte an item needs from the record -- its trellis tiles and channel scales --
//             and the tokens' inputs, as cp.async copies into one of two stages; the stage's
//             `full` barrier completes when they land. It refills a stage as soon as the other
//             roles release it, so the next item's bytes are in flight while one is multiplied.
//   aux       four warps prepare each item's fp16 operand from its stage into one of two A
//             buffers (`a_full`), and reduce the previous item's result from the output buffer
//             (`o_full`, then `o_empty`): the sums, the fences and the arrival counts.
//   mma       sixteen warps decode the trellis tiles from the stage straight into m16n8k16 B
//             fragments, 16 tiles per warp, and leave the raw sums in the output buffer. They
//             wait on nothing but their operands.
//
// Work items, 256 tiles (65,536 weights) each, all gate/up items first, then all down items:
//
//   gate/up (expert u, block b in 0..4, k-split s in 0..9)
//       gate block b and up block b (256 columns, one 16-column tile per mma warp) over the
//       inputs 256 s .. 256 s + 255: the tokens' inputs times suh, the 128-wide Hadamard, one
//       power-of-two scale per token into fp16. The aux warps add the pre-rotation sums into the
//       int64 fixed-point gate/up accumulator and count one arrival on (u, b).
//   down (expert u, h block j in 0..4, column block c in 0..4)
//       its producer waits for (u, j)'s ten arrivals; the aux warps read gate and up block j's
//       sums, rotate both, apply svh and SwiGLU (h's block j), rotate h_j o suh_down into fp16;
//       the mma warps multiply the block's 8 k-tiles against 512 down columns (two tiles per
//       warp); the aux warps apply the output Hadamard per 128 columns, svh and each selecting
//       token's routing weight, and add the result into the fixed-point output accumulator. Each
//       of h block j's five readers adds 16 to (u, j)'s counter once it has read the sums; the
//       fifth zeroes them and the counter.
//
// The reduction is order-independent where it crosses items (integer sums) and fixed inside an
// item, so the result is deterministic. Against the register kernel only where partial sums
// round differs: the gate/up k-split is 256 inputs instead of 640 (an fp32 MMA chain of 16
// k-tiles instead of 40 before the conversion to fixed point), and down is summed over h's five
// blocks in fixed point instead of in one fp32 chain over 640 inputs, the output Hadamard taken
// per block (it is linear). Each fp16 operand scale is a power of two, now per 256 (gate/up) or
// 128 (down) inputs instead of per 640, which moves no rounding of a normal number; the ~1e-7
// difference in the gate/up sums reaches the output only by flipping the fp16 rounding of a few
// of h's entries (test_moe_decode_routes).
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

constexpr int kMmaWarps = 16;
constexpr int kAuxWarps = 8;
constexpr int kAuxThreads = kAuxWarps * 32;
constexpr int kAuxWarp0 = kMmaWarps;                   // aux warps 16..23
constexpr int kProducerWarp = kMmaWarps + kAuxWarps;  // warps 24..27
constexpr int kProducerWarps = 4;
constexpr int kProducerThreads = kProducerWarps * 32;
constexpr int kThreads = (kProducerWarp + kProducerWarps) * 32;
constexpr int kSlots = 4;      // items staged at once
constexpr int kInFlight = 2;   // items whose copies may be in flight at once
constexpr int kRingBytes = 78 * 1024;
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
__host__ __device__ constexpr int weight_bytes(int k2) { return kItemTiles * 16 * k2; }

static_assert(kMmaWarps * kGuKTiles == kItemTiles, "gate/up item shape");
static_assert(kDnCols / 16 == 2 * kMmaWarps && kMmaWarps * 2 * kDnKTiles == kItemTiles, "down item shape");
constexpr int kMaxUnique = kStagedMaxTokens * kTopK;  // distinct experts of a call
static_assert(kStagedMaxTokens <= kAuxWarps && kMaxUnique <= 64, "tokens of a call");

// An item's bytes past its weights: a gate/up item's suh slice and x slice, or a down item's
// suh_down block, svh_down columns and the gate/up svh of its h block.
constexpr int kGuSuh = 0;  // fp16 [256], then x bf16 [tokens][256]
constexpr int kGuX = 512;
constexpr int kDnSuh = 0;    // fp16 [128]
constexpr int kDnSvh = 256;  // fp16 [512]
constexpr int kDnSvhGu = 256 + 1024;  // fp16 [gate 128 | up 128]
// An item's footprint in the staging ring.
__host__ __device__ constexpr int item_bytes(int k2, bool gate_up, int tokens) {
  return (weight_bytes(k2) + (gate_up ? kGuX + 512 * tokens : kDnSvhGu + 512) + 127) / 128 * 128;
}
__host__ __device__ constexpr int abuf_bytes(int tokens) { return (tokens * kAStride * 2 + 127) / 128 * 128; }
// Dynamic shared memory of a launch for `tokens`: the ring, two A buffers and the output buffer.
__host__ __device__ constexpr int dynamic_bytes(int tokens) {
  return kRingBytes + 2 * abuf_bytes(tokens) + tokens * kDnCols * 4;
}
static_assert(item_bytes(8, false, kStagedMaxTokens) <= kRingBytes / 2 &&
                  item_bytes(8, true, kStagedMaxTokens) <= kRingBytes / 2,
              "two of the largest items fit the ring");

enum Kind : int { kGateUp = 0, kDown = 1, kEnd = 2 };

struct Header {
  int kind;
  int ticket;
  int u;
  int block;  // gate/up: b; down: the h block j
  int sub;    // gate/up: the k-split s; down: the column block c
  int k2;
  int off;    // its bytes in the ring: [off, off + size)
  int size;
  unsigned long long issued;  // kTrace: when the producer issued the copies
};

// What the aux warps' prepare leaves beside an item's A buffer for the mma warps and for its own
// reduction later: the header, each token's operand scale and, for a down item, svh_down's 512
// columns (its stage is released before the reduction).
struct Meta {
  Header h;
  float scale[kStagedMaxTokens];
  alignas(16) __half svh[kDnCols];
};

struct Shared {
  int n_unique;
  int unique_id[kMaxUnique];
  uint32_t unique_sel[kMaxUnique];
  float unique_w[kMaxUnique][kStagedMaxTokens];
  ignis_moe_slot slot[kMaxUnique][2];  // each distinct expert's slots, read once
  unsigned long long full[kSlots];   // producer -> aux, mma: the item's bytes landed
  unsigned long long empty[kSlots];  // aux, mma -> producer: the item's bytes are read
  unsigned long long a_full[2];      // aux -> mma: A buffer and meta ready
  unsigned long long o_full;         // mma -> aux: the output buffer holds an item's sums
  unsigned long long o_empty;        // aux -> mma: the output buffer is read
  int arrived;                       // aux: the down item's h block is whole
  Header header[kSlots];
  Meta meta[2];
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

// Ticket t's item: every gate/up item first (expert-major, then h block, then k-split), then every
// down item (expert-major, then h block, then column block). Interleaving each expert's down items
// a few experts behind its gate/up items measured slower at 1-3 tokens (2026-10-08: 1 token 22.2-
// 32.3 us at a lag of 8 to 2 experts against 21.3 all gate/up first): a down item taken early waits
// on gate/up items still in flight.
__device__ __forceinline__ void decode_ticket(int t, int n_gate_up, Header &h) {
  if (t < n_gate_up) {
    h.kind = kGateUp;
    h.u = t / kGuItems;
    h.block = t / kGuSplits % kGateUpBlocks;
    h.sub = t % kGuSplits;
  } else {
    const int d = t - n_gate_up;
    h.kind = kDown;
    h.u = d / kDnItems;
    h.block = d / kDnColBlocks % kGateUpBlocks;
    h.sub = d % kDnColBlocks;
  }
}

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
__device__ __forceinline__ bool mbar_test(unsigned long long *bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(done)
      : "r"(smem_addr(bar)), "r"(parity)
      : "memory");
  return done != 0;
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
// The aux warps' own barrier, and the producer warps'.
__device__ __forceinline__ void aux_sync() { asm volatile("bar.sync 1, %0;\n" ::"n"(kAuxThreads) : "memory"); }
__device__ __forceinline__ void producer_sync() { asm volatile("bar.sync 2, %0;\n" ::"n"(kProducerThreads) : "memory"); }

// `chunks` 16-byte copies spread over the producer warps' threads: chunk i from src(i) to dst + 16 i.
template <typename Src>
__device__ __forceinline__ void copy_chunks(char *dst, int chunks, Src src) {
  for (int i = threadIdx.x - kProducerWarp * 32; i < chunks; i += kProducerThreads) cp_async16(dst + 16 * i, src(i));
}

__device__ __forceinline__ uint32_t ld_acquire(const uint32_t *p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// The distinct experts of the call, as build_unique (moe_decode_common.cuh) leaves them (only the
// fields the staged kernel reads), and both slots of each: warp 0 holds the 10 x tokens selections
// two to a lane, loads each one's slots right behind its id -- two dependent round trips to global
// memory in all -- and finds first appearances with warp matches.
__device__ void unique_experts(Shared &s, const int32_t *ids, const float *weights, const ignis_moe_slot *slots,
                               int tokens) {
  for (int j = threadIdx.x; j < kMaxUnique * kStagedMaxTokens; j += blockDim.x) (&s.unique_w[0][0])[j] = 0.0f;
  if (threadIdx.x < kMaxUnique) s.unique_sel[threadIdx.x] = 0u;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int n = tokens * kTopK;  // <= 40
    const bool v0 = lane < n, v1 = 32 + lane < n;
    // Distinct sentinels for the empty places, so they match nothing.
    const int e0 = v0 ? ids[lane] : -1 - lane;
    const int e1 = v1 ? ids[32 + lane] : -100 - lane;
    const float w0 = v0 ? weights[lane] : 0.0f;
    const float w1 = v1 ? weights[32 + lane] : 0.0f;
    ignis_moe_slot g0{}, d0{}, g1{}, d1{};
    if (v0) {
      g0 = load_slot(slots, e0, IGNIS_MOE_PROJ_GATE_UP);
      d0 = load_slot(slots, e0, IGNIS_MOE_PROJ_DOWN);
    }
    if (v1) {
      g1 = load_slot(slots, e1, IGNIS_MOE_PROJ_GATE_UP);
      d1 = load_slot(slots, e1, IGNIS_MOE_PROJ_DOWN);
    }
    const unsigned lt = (1u << lane) - 1u;
    // Selection lane: its first appearance is the lowest lane holding the same id.
    const unsigned m0 = __match_any_sync(0xFFFFFFFFu, e0);
    const int first0 = __ffs(m0) - 1;
    const unsigned b0 = __ballot_sync(0xFFFFFFFFu, v0 && first0 == lane);
    const int u0 = __popc(b0 & lt);  // its distinct index, if first
    // Selection 32 + lane: first unless one of the first 32 or a lower lane of its own half holds
    // the same id.
    int earlier = -1;  // the lane of the first 32 holding the same id
    for (int k = 0; k < 32; ++k) {
      if (__shfl_sync(0xFFFFFFFFu, e0, k) == e1 && earlier < 0) earlier = k;
    }
    const unsigned m1 = __match_any_sync(0xFFFFFFFFu, e1);
    const int first1 = __ffs(m1) - 1;
    const unsigned b1 = __ballot_sync(0xFFFFFFFFu, v1 && earlier < 0 && first1 == lane);
    const int u1 = __popc(b0) + __popc(b1 & lt);
    // Every selection's distinct index, from its first appearance.
    const int s0 = __shfl_sync(0xFFFFFFFFu, u0, first0);
    const int from_first = __shfl_sync(0xFFFFFFFFu, u0, earlier < 0 ? 0 : earlier);
    const int from_own = __shfl_sync(0xFFFFFFFFu, u1, first1);
    const int s1 = earlier >= 0 ? from_first : from_own;
    if (v0 && first0 == lane) {
      s.unique_id[u0] = e0;
      s.slot[u0][IGNIS_MOE_PROJ_GATE_UP] = g0;
      s.slot[u0][IGNIS_MOE_PROJ_DOWN] = d0;
    }
    if (v1 && earlier < 0 && first1 == lane) {
      s.unique_id[u1] = e1;
      s.slot[u1][IGNIS_MOE_PROJ_GATE_UP] = g1;
      s.slot[u1][IGNIS_MOE_PROJ_DOWN] = d1;
    }
    if (v0) {
      s.unique_w[s0][lane / kTopK] = w0;
      atomicOr(&s.unique_sel[s0], 1u << (lane / kTopK));
    }
    if (v1) {
      s.unique_w[s1][(32 + lane) / kTopK] = w1;
      atomicOr(&s.unique_sel[s1], 1u << ((32 + lane) / kTopK));
    }
    if (lane == 0) s.n_unique = __popc(b0) + __popc(b1);
  }
  __syncthreads();
}

// ---- the producer ---------------------------------------------------------------------------

// Stage item `h` for `tokens` tokens -- the weights, then the item's extras -- as 16-byte copies
// spread over the producer warps (a few hundred bytes per contiguous run: one bulk copy per run
// costs the copy engine more than the threads' own copies).
template <int K2>
__device__ void stage_item(const Shared &s, const Params &p, const Header &h, char *stage) {
  constexpr int tile_bytes = 16 * K2;
  const int tokens = p.tokens;
  char *extra = stage + weight_bytes(K2);
  if (h.kind == kGateUp) {
    const ignis_moe_slot slot = s.slot[h.u][IGNIS_MOE_PROJ_GATE_UP];
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
    const ignis_moe_slot gu = s.slot[h.u][IGNIS_MOE_PROJ_GATE_UP];
    const ignis_moe_slot dn = s.slot[h.u][IGNIS_MOE_PROJ_DOWN];
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
    const char *suh = reinterpret_cast<const char *>(planes.suh + 128 * h.block);
    copy_chunks(extra + kDnSuh, 128 * 2 / 16, [&](int i) { return suh + 16 * i; });
    const char *svh = reinterpret_cast<const char *>(planes.svh + kDnCols * h.sub);
    copy_chunks(extra + kDnSvh, kDnCols * 2 / 16, [&](int i) { return svh + 16 * i; });
    copy_chunks(extra + kDnSvhGu, 2 * 128 * 2 / 16, [&](int i) {
      const int half = i / 16, off = i % 16;
      return reinterpret_cast<const char *>(gu_planes.svh + half * kInter + 128 * h.block) + 16 * off;
    });
  }
}

// The producer warps: the first takes the tickets, keeps the in-flight limit and allocates the
// ring; then all of them issue the item's copies.
template <bool kTrace>
__device__ void producer(Shared &s, const Params &p, char *ring, int n_gate_up, int n_total) {
  const int lane = threadIdx.x & 31;
  const bool lead = threadIdx.x >> 5 == kProducerWarp;
  // The first ticket is the CTA's own; each later one is fetched an item before it is needed
  // (lane 0 holds it until then), so the atomic's round trip overlaps an item's copies.
  int ticket = static_cast<int>(blockIdx.x);
  int pending = 0;
  if (lead && lane == 0) pending = static_cast<int>(atomicAdd(&p.counters->ticket, 1u) + gridDim.x);
  int head = 0;    // the ring's next free byte
  int oldest = 0;  // the oldest item the consumers have not released
  for (int n = 0;; ++n) {
    const int slot = n % kSlots;
    unsigned long long t_begin = 0, t_flight = 0;
    if (lead) {
      if constexpr (kTrace) t_begin = global_ns();
      // At most kInFlight items' copies in flight; the first item alone, so every SM's first
      // bytes land first.
      if (n == 1) {
        mbar_wait(&s.full[0], 0u);
      } else if (n >= kInFlight) {
        mbar_wait(&s.full[(n - kInFlight) % kSlots], static_cast<uint32_t>(((n - kInFlight) / kSlots) & 1));
      }
      if constexpr (kTrace) t_flight = global_ns();
      Header h{};
      h.ticket = ticket;
      if (ticket >= n_total) {
        h.kind = kEnd;
      } else {
        decode_ticket(ticket, n_gate_up, h);
      }
      if (h.kind != kEnd) {
        const int proj = h.kind == kGateUp ? IGNIS_MOE_PROJ_GATE_UP : IGNIS_MOE_PROJ_DOWN;
        h.k2 = static_cast<int>(s.slot[h.u][proj].k2);
        h.size = item_bytes(h.k2, h.kind == kGateUp, p.tokens);
        h.off = head + h.size > kRingBytes ? 0 : head;
      }
      // Wait for the slot's previous item and for every live item whose bytes the new one would
      // overwrite -- a wrap to the ring's start can land on younger items than the oldest -- and,
      // since the consumers release in order, for every item older than those.
      int last = n - kSlots;
      for (int m = oldest; m < n; ++m) {
        const Header &o = s.header[m % kSlots];
        if (h.kind != kEnd && h.off < o.off + o.size && o.off < h.off + h.size) last = m;
      }
      for (; oldest <= last; ++oldest) {
        mbar_wait(&s.empty[oldest % kSlots], static_cast<uint32_t>((oldest / kSlots) & 1));
      }
      if (h.kind != kEnd) head = h.off + h.size;
      if constexpr (kTrace) h.issued = global_ns();
      __syncwarp();
      if (lane == 0) s.header[slot] = h;
    }
    producer_sync();  // the item's header is every producer thread's to read
    const Header h = s.header[slot];
    if (h.kind != kEnd) {
      dispatch_k2(static_cast<uint32_t>(h.k2), [&](auto k2) { stage_item<decltype(k2)::value>(s, p, h, ring + h.off); });
    }
    if (lead && lane == 0) mbar_arrive(&s.full[slot]);
    mbar_arrive_on_copies(&s.full[slot]);
    if constexpr (kTrace) {
      if (lead && lane == 0 && h.kind != kEnd && h.ticket < kTraceUnits) {
        unsigned long long *r = p.trace + static_cast<size_t>(h.ticket) * kTraceWords;
        r[12] = t_begin;
        r[13] = t_flight;
        r[14] = h.issued;
        r[15] = global_ns();
      }
    }
    if (h.kind == kEnd) return;
    if (lead) {
      int fetched = 0;
      if (lane == 0) fetched = static_cast<int>(atomicAdd(&p.counters->ticket, 1u) + gridDim.x);
      ticket = __shfl_sync(0xFFFFFFFFu, pending, 0);
      pending = fetched;
    }
    producer_sync();  // the header slot is read before the lead reuses it
  }
}

// ---- the mma warps --------------------------------------------------------------------------

// acc[n] += A(tokens x 16 KT) . W(16 KT x 16) for the warp's column tiles; tile (r, n) of the
// stage at word (r * row_tiles + col[n]) * tile_words.
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

// The warp's raw sums of its column tiles into the output buffer, row-major [tokens][cols].
template <int NT>
__device__ __forceinline__ void store_sums(float *out, int cols, const int (&col)[NT], int tokens, const float (&acc)[NT][2][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c = lane & 3;
  if (g < tokens) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int cc = col[n] * 16 + hf * 8 + 2 * c;
        *reinterpret_cast<float2 *>(&out[g * cols + cc]) = make_float2(acc[n][hf][0], acc[n][hf][1]);
      }
    }
  }
}

template <bool kTrace>
__device__ void mma_role(Shared &s, const Params &p, const char *ring, char *abufs, int abuf_stride, float *out) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tokens = p.tokens;
  for (int n = 0;; ++n) {
    const int st = n % kSlots;
    const int b = n & 1;
    UnitStamps stamps;
    if constexpr (kTrace) stamps.begin = global_ns();
    mbar_wait(&s.a_full[b], static_cast<uint32_t>((n >> 1) & 1));
    const Header h = s.meta[b].h;
    if (h.kind == kEnd) break;
    mbar_wait(&s.full[st], static_cast<uint32_t>((n / kSlots) & 1));  // done: orders the item's bytes
    if constexpr (kTrace) stamps.ready = global_ns();
    const uint32_t *stage = reinterpret_cast<const uint32_t *>(ring + h.off);
    const __half *a = reinterpret_cast<const __half *>(abufs + b * abuf_stride);
    dispatch_k2(static_cast<uint32_t>(h.k2), [&](auto k2) {
      constexpr int K2 = decltype(k2)::value;
      if (h.kind == kGateUp) {
        float acc[1][2][4] = {};
        const int col[1] = {warp};
        mma_stage<K2, 1, kGuKTiles>(stage, 16, col, a, tokens, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.empty[st]);
        if constexpr (kTrace) stamps.mma = global_ns();
        if (n > 0) mbar_wait(&s.o_empty, static_cast<uint32_t>((n - 1) & 1));
        store_sums<1>(out, 256, col, tokens, acc);
      } else {
        float acc[2][2][4] = {};
        const int col[2] = {warp, warp + kMmaWarps};
        mma_stage<K2, 2, kDnKTiles>(stage, 2 * kMmaWarps, col, a, tokens, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(&s.empty[st]);
        if constexpr (kTrace) stamps.mma = global_ns();
        if (n > 0) mbar_wait(&s.o_empty, static_cast<uint32_t>((n - 1) & 1));
        store_sums<2>(out, kDnCols, col, tokens, acc);
      }
    });
    __syncwarp();
    if (lane == 0) mbar_arrive(&s.o_full);
    if constexpr (kTrace) {
      if (threadIdx.x == 0 && h.ticket < kTraceUnits) {
        stamps.end = global_ns();
        stamps.extra = 0;
        unsigned long long *r = p.trace + static_cast<size_t>(h.ticket) * kTraceWords;
        r[0] = (static_cast<unsigned long long>(sm_id()) << 32) | blockIdx.x;
        r[1] = static_cast<unsigned long long>(h.kind) | static_cast<unsigned long long>(h.u) << 8 |
               static_cast<unsigned long long>(h.block) << 16 | static_cast<unsigned long long>(h.sub) << 24;
        r[2] = stamps.begin;
        r[3] = stamps.ready;
        r[4] = stamps.mma;
        r[5] = stamps.end;
        r[7] = h.issued;
      }
    }
  }
}

// ---- the aux warps --------------------------------------------------------------------------

// The gate/up item's operand: aux warp t < tokens rotates its token's 256 inputs (x o suh, two
// 128-wide Hadamards) and writes them into A as fp16 under one power-of-two scale.
__device__ void prepare_gate_up(Meta &m, const char *extra, int tokens, __half *a) {
  const int lane = threadIdx.x & 31;
  const int t = (threadIdx.x >> 5) - kAuxWarp0;
  if (t >= tokens) return;
  const __half *suh = reinterpret_cast<const __half *>(extra + kGuSuh);
  const __nv_bfloat16 *x = reinterpret_cast<const __nv_bfloat16 *>(extra + kGuX) + t * kGuSplit;
  float v[2][4];
  float mx = 0.0f;
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int k = blk * 128 + 4 * lane + q;
      v[blk][q] = __bfloat162float(x[k]) * __half2float(suh[k]);
    }
    warp_hadamard128(v[blk]);
#pragma unroll
    for (int q = 0; q < 4; ++q) mx = fmaxf(mx, fabsf(v[blk][q]));
  }
  const float scale = fp16_operand_scale(warp_max(mx));
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
    for (int q = 0; q < 4; ++q) a[t * kAStride + blk * 128 + 4 * lane + q] = __float2half_rn(v[blk][q] * scale);
  }
  if (lane == 0) m.scale[t] = scale;
}

// The down item's operand, two aux warps per token: warp 2t + 1 reads up block j's sums, rotates
// them and applies svh, leaving them in token t's A row (as scratch); warp 2t does the same for
// gate, applies SwiGLU (h's block j), then h_j o suh_down, rotated, into A as fp16 under one
// power-of-two scale. The aux threads copy svh_down's 512 columns into the meta for the
// reduction.
__device__ void prepare_down(Meta &m, const Params &p, const char *extra, int tokens, __half *a) {
  static_assert(2 * kStagedMaxTokens <= kAuxWarps, "two aux warps per token");
  static_assert(128 * 4 <= kAStride * 2, "a token's up block fits its A row");
  const int lane = threadIdx.x & 31;
  const int aux = threadIdx.x - kAuxWarp0 * 32;
  const int t = aux >> 6;
  const bool up_warp = (aux >> 5) & 1;
  const Header &h = m.h;
  const __half *svh = reinterpret_cast<const __half *>(extra + kDnSvh);
  for (int i = aux; i < kDnCols; i += kAuxThreads) m.svh[i] = svh[i];
  const __half *svh_gu = reinterpret_cast<const __half *>(extra + kDnSvhGu);
  float *row = reinterpret_cast<float *>(a + t * kAStride);  // token t's A row, as scratch first
  float v[4];
  if (t < tokens) {
    const long long *sums =
        p.gate_up + (static_cast<size_t>(h.u) * p.cap + t) * kGateUpOut + (up_warp ? kInter : 0) + 128 * h.block;
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = from_fixed(__ldcg(sums + 4 * lane + q));
    warp_hadamard128(v);
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] *= __half2float(svh_gu[(up_warp ? 128 : 0) + 4 * lane + q]);
    if (up_warp) {
#pragma unroll
      for (int q = 0; q < 4; ++q) row[4 * lane + q] = v[q];
    }
  }
  aux_sync();
  float u[4];
  if (t < tokens && !up_warp) {
#pragma unroll
    for (int q = 0; q < 4; ++q) u[q] = row[4 * lane + q];
  }
  aux_sync();  // the scratch is read before A overwrites it
  if (t < tokens && !up_warp) {
    const __half *suh = reinterpret_cast<const __half *>(extra + kDnSuh);
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = silu(v[q]) * u[q] * __half2float(suh[4 * lane + q]);
    warp_hadamard128(v);
    float mx = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; ++q) mx = fmaxf(mx, fabsf(v[q]));
    const float scale = fp16_operand_scale(warp_max(mx));
#pragma unroll
    for (int q = 0; q < 4; ++q) a[t * kAStride + 4 * lane + q] = __float2half_rn(v[q] * scale);
    if (lane == 0) m.scale[t] = scale;
  }
}

// Item `m`'s reduction from the output buffer.
__device__ void reduce_item(const Shared &s, const Meta &m, const Params &p, const float *out) {
  const int lane = threadIdx.x & 31;
  const int aux = threadIdx.x - kAuxWarp0 * 32;
  const int warp = aux >> 5;
  const int tokens = p.tokens;
  const Header &h = m.h;
  if (h.kind == kGateUp) {
    // 256 pre-rotation sums per token: columns c < 128 are gate block b's, the rest up's.
    for (int i = aux; i < tokens * 256; i += kAuxThreads) {
      const int t = i / 256, c = i % 256;
      const int col = (c < 128 ? 0 : kInter) + 128 * h.block + (c & 127);
      const float inv = 1.0f / m.scale[t];
      add_fixed(p.gate_up + (static_cast<size_t>(h.u) * p.cap + t) * kGateUpOut + col, out[t * 256 + c] * inv);
    }
    __threadfence();
    aux_sync();
    if (aux == 0) atomicAdd(&p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block], 1u);
    return;
  }
  // Down: the output rotation per 128 columns, warp-strided over (token, block) for every token
  // that selected u.
  for (int task = warp; task < 4 * tokens; task += kAuxWarps) {
    const int t = task / 4;
    const int blk = task % 4;
    if (!(s.unique_sel[h.u] >> t & 1u)) continue;
    float v[4];
    const float inv = 1.0f / m.scale[t];
#pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = out[t * kDnCols + blk * 128 + 4 * lane + q] * inv;
    warp_hadamard128(v);
    const float wt = s.unique_w[h.u][t];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int cc = blk * 128 + 4 * lane + q;
      add_fixed(p.acc + static_cast<size_t>(t) * kHidden + kDnCols * h.sub + cc, wt * (v[q] * __half2float(m.svh[cc])));
    }
  }
  // This reader read block j's sums in its prepare: counted now, off the operand's path. The
  // fifth reader zeroes the sums and the count for the next call.
  __shared__ uint32_t before;
  if (aux == 0) before = atomicAdd(&p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block], kReaderStep);
  aux_sync();
  if (before == kLastReader) {
    for (int i = aux; i < tokens * 256; i += kAuxThreads) {
      const int t = i / 256, c = i % 256;
      p.gate_up[(static_cast<size_t>(h.u) * p.cap + t) * kGateUpOut + (c < 128 ? 0 : kInter) + 128 * h.block + (c & 127)] = 0;
    }
    if (aux == 0) p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block] = 0u;
  }
}

// Item n - 1's reduction, once the mma warps have left its sums; then the output buffer is free.
template <bool kTrace>
__device__ void reduce_previous(Shared &s, const Params &p, int n, const float *out, unsigned long long *rec) {
  mbar_wait(&s.o_full, static_cast<uint32_t>((n - 1) & 1));
  unsigned long long t0 = 0;
  if constexpr (kTrace) t0 = global_ns();
  reduce_item(s, s.meta[(n - 1) & 1], p, out);
  aux_sync();
  if ((threadIdx.x & 31) == 0) mbar_arrive(&s.o_empty);
  if constexpr (kTrace) {
    if (threadIdx.x == kAuxWarp0 * 32 && rec != nullptr) {
      rec[10] = t0;
      rec[11] = global_ns();
    }
  }
}

// Each item n: prepare it while the mma warps multiply item n - 1, then reduce item n - 1. A down
// item's h block must be whole first: while it is not, the reduction of item n - 1 goes first --
// its arrival may be one of the ten the block waits for.
template <bool kTrace>
__device__ void aux_role(Shared &s, const Params &p, const char *ring, char *abufs, int abuf_stride, const float *out) {
  const int lane = threadIdx.x & 31;
  const int aux = threadIdx.x - kAuxWarp0 * 32;
  const int tokens = p.tokens;
  unsigned long long *rec = nullptr;  // kTrace: item n - 1's record
  for (int n = 0;; ++n) {
    const int st = n % kSlots;
    const int b = n & 1;
    Meta &m = s.meta[b];
    unsigned long long t_wait = 0, t_prep = 0;
    if constexpr (kTrace) t_wait = global_ns();
    mbar_wait(&s.full[st], static_cast<uint32_t>((n / kSlots) & 1));
    const Header h = s.header[st];
    bool reduced = n == 0;
    if (h.kind == kDown) {
      const uint32_t *count = &p.counters->gate_up_arrivals[h.u * kGateUpBlocks + h.block];
      if (aux == 0) s.arrived = ld_acquire(count) >= kGuArrivals;
      aux_sync();
      if (!s.arrived) {
        if (!reduced) reduce_previous<kTrace>(s, p, n, out, rec);
        reduced = true;
        if (aux == 0) {
          while (ld_acquire(count) < kGuArrivals) __nanosleep(32);
        }
      }
    }
    if constexpr (kTrace) t_prep = global_ns();
    if (aux == 0) m.h = h;
    aux_sync();
    const char *extra = ring + h.off + weight_bytes(h.k2);
    __half *a = reinterpret_cast<__half *>(abufs + b * abuf_stride);
    if (h.kind == kGateUp) {
      prepare_gate_up(m, extra, tokens, a);
    } else if (h.kind == kDown) {
      prepare_down(m, p, extra, tokens, a);
    }
    aux_sync();
    if (lane == 0) {
      mbar_arrive(&s.a_full[b]);
      if (h.kind != kEnd) mbar_arrive(&s.empty[st]);
    }
    if constexpr (kTrace) {
      if (aux == 0 && h.kind != kEnd && h.ticket < kTraceUnits) {
        unsigned long long *r = p.trace + static_cast<size_t>(h.ticket) * kTraceWords;
        r[6] = t_prep - t_wait;
        r[8] = t_prep;
        r[9] = global_ns();
      }
    }
    if (!reduced) reduce_previous<kTrace>(s, p, n, out, rec);
    if constexpr (kTrace) {
      rec = h.kind != kEnd && h.ticket < kTraceUnits ? p.trace + static_cast<size_t>(h.ticket) * kTraceWords : nullptr;
    }
    if (h.kind == kEnd) return;
  }
}

template <bool kTrace>
__global__ void __launch_bounds__(kThreads, 1) experts_decode_staged_kernel(Params p) {
  __shared__ Shared s;
  extern __shared__ __align__(128) char dyn[];
  const int abuf_stride = abuf_bytes(p.tokens);
  char *ring = dyn;
  char *abufs = dyn + kRingBytes;
  float *out = reinterpret_cast<float *>(abufs + 2 * abuf_stride);
  if constexpr (kTrace) {
    if (threadIdx.x == 0) trace_cta(p.trace)[0] = global_ns();
  }
  if (threadIdx.x == 0) {
    for (int i = 0; i < kSlots; ++i) {
      mbar_init(&s.full[i], kProducerThreads + 1);  // each producer thread's copies + the header
      mbar_init(&s.empty[i], kMmaWarps + kAuxWarps);  // every mma and aux warp has read the item
    }
    for (int i = 0; i < 2; ++i) mbar_init(&s.a_full[i], kAuxWarps);
    mbar_init(&s.o_full, kMmaWarps);
    mbar_init(&s.o_empty, kAuxWarps);
  }
  unique_experts(s, p.ids, p.weights, p.slots, p.tokens);  // ends with __syncthreads
  const int n_gate_up = s.n_unique * kGuItems;
  const int n_total = n_gate_up + s.n_unique * kDnItems;
  const int warp = threadIdx.x >> 5;
  if (warp >= kProducerWarp) {
    if constexpr (kTrace) {
      if (threadIdx.x == kProducerWarp * 32) trace_cta(p.trace)[1] = global_ns();
    }
    producer<kTrace>(s, p, ring, n_gate_up, n_total);
  } else if (warp >= kAuxWarp0) {
    aux_role<kTrace>(s, p, ring, abufs, abuf_stride, out);
  } else {
    mma_role<kTrace>(s, p, ring, abufs, abuf_stride, out);
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
  const int bytes = dynamic_bytes(tokens);
  if (trace != nullptr) {
    experts_decode_staged_kernel<true><<<grid, kThreads, bytes, stream>>>(p);
  } else {
    experts_decode_staged_kernel<false><<<grid, kThreads, bytes, stream>>>(p);
  }
  return check_launch("ignis_moe_experts_decode (staged)");
}

}  // namespace ignis_moe
