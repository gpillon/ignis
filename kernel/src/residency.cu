// ignis kernel leaf: Flash-Next's expert residency, the device side -- OURS (see
// kernel/include/ignis_residency.h; the policy is crates/core/src/residency/policy.rs).
//
// One layer step is these launches, on the caller's stream unless named:
//   resolve_demand    one CTA: classify the selection, stamp hits, place misses (free slot,
//                     else the LRU victim in decode, else the staging ring in prefill), write
//                     the slot table and the demand copy jobs;
//   resolve_prefetch  one CTA: place the lookahead's prefetches and their copy jobs -- on the
//                     caller's stream in a whole step, on the prefetch stream in a split one;
//   copy              a small grid copying the demand jobs from the mapped host pool, timing
//                     itself into its phase's stall;
//   copy              the same over the prefetch jobs, on the prefetch stream, beside the
//                     expert op, after the demand copy.
// Everything the resolve decides lives in device memory, so the host never waits on it.

#include "ignis_residency.h"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error = "no error";

int32_t fail(const std::string &message) {
  g_error = message;
  return -1;
}

// A failure is reported through the return value and consumed from the thread's last error:
// left there, it would be read back by the next launch check and blamed on that launch.
#define RESIDENCY_CUDA(expr)                                                                      \
  do {                                                                                            \
    const cudaError_t status_ = (expr);                                                           \
    if (status_ != cudaSuccess) {                                                                 \
      (void)cudaGetLastError();                                                                   \
      return fail(std::string(#expr) + ": " + cudaGetErrorString(status_));                       \
    }                                                                                             \
  } while (0)

constexpr uint32_t kClasses = IGNIS_RESIDENCY_CLASSES;
constexpr uint32_t kTopK = IGNIS_MOE_TOP_K;
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kMaxKeysPerLayer = 2 * IGNIS_MOE_EXPERTS;  // one thread per key
static_assert(kMaxKeysPerLayer <= kThreads, "the resolve classifies one key per thread");
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kKeyBits = 20;  // keys and slot indices fit below 2^20
constexpr unsigned long long kNoSlot = ~0ull;
constexpr unsigned long long kNoTime = ~0ull;  // no demand copy in flight
constexpr uint8_t kPrefetched = 1;
constexpr uint8_t kStaged = 2;
constexpr uint32_t kLists = IGNIS_RESIDENCY_LISTS;
enum List : uint32_t { kHits = 0, kPrefetchHits, kMisses, kEvictions, kPrefetches, kDropped };

struct Job {
  const uint4 *src;
  uint4 *dst;
  unsigned long long vectors;  // 16-byte units
};

struct Counters {
  unsigned long long hits[kClasses][2];
  unsigned long long misses[kClasses][2];
  unsigned long long prefetch_issued;
  unsigned long long prefetch_used;
  unsigned long long bytes_moved[2];
  unsigned long long stall_nanos[2];
};
static_assert(sizeof(Counters) == sizeof(ignis_residency_counters), "counters mirror the ABI");

struct Report {
  uint32_t status;
  uint32_t count[kLists];
  uint32_t reserved;
  unsigned long long bytes_moved;
};
static_assert(sizeof(Report) == sizeof(ignis_residency_report), "report mirrors the ABI");

// Device-resident scalars of the policy's state.
struct State {
  unsigned long long clock;
  uint32_t used[kClasses];
  uint32_t half_layer[2];      // the layer a ring half is staged for, or kNone
  unsigned long long half_fill[2];
  uint32_t half_count[2];
  uint32_t n_demand;
  uint32_t n_prefetch;
  // The layer of the last committed step, or kNone: a ring half tagged with the current layer
  // holds this step's lookahead only when the last step was another layer; after a step of the
  // same layer (a restarted forward) it holds that step's own, released, entries.
  uint32_t last_layer;
  // What a step's demand half leaves its prefetch half (resolve_prefetch): the refusal, the
  // report's list counts and the bytes the demand half moved.
  uint32_t step_status;
  uint32_t step_count[kLists];
  unsigned long long step_moved;
  // The demand copy in flight: its blocks' earliest start (kNoTime between copies) and how many
  // blocks are done; its last block adds the span to the stall and resets both.
  unsigned long long copy_start;
  uint32_t copy_done;
};

// Everything the kernels read, by value.
struct Dev {
  uint32_t layers, experts, width, prefill_width;  // a decode and a prefill step's lookahead width
  uint32_t capacity[kClasses];
  unsigned long long record_bytes[kClasses];
  unsigned long long budget;
  unsigned long long half_bytes;
  const uint8_t *cls;        // [keys] class of each projection
  const uint8_t *k2;         // [keys]
  const unsigned long long *host_off;  // [keys]
  const uint8_t *host;       // the pool as the device sees it
  int32_t *slot_of;          // [keys] slot in its class, or -1
  uint8_t *flags;            // [keys] kPrefetched | kStaged
  ignis_moe_slot *table;     // [keys]
  uint8_t *pool[kClasses];
  uint32_t *owner[kClasses];               // [capacity] key or kNone
  unsigned long long *stamp[kClasses];     // [capacity]
  uint8_t *ring;
  uint32_t *half_keys;       // [2][experts * 2]
  State *st;
  Job *demand;               // [experts * 2]
  Job *prefetch;             // [experts * 2]
  Counters *counters;
  Report *report;            // [layers], NULL without a report
  uint32_t *report_entries;  // [layers][kLists][4 * experts]
  ignis_residency_mirror *mirror;  // mapped host memory, or NULL
};


// ---- block-wide helpers (every thread calls them) ----------------------------------------------

__device__ unsigned long long block_min(unsigned long long v, unsigned long long *scratch) {
  for (int o = 16; o > 0; o >>= 1) {
    const unsigned long long w = __shfl_down_sync(0xffffffffu, v, o);
    v = w < v ? w : v;
  }
  const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < (blockDim.x >> 5) ? scratch[lane] : kNoSlot;
    for (int o = 16; o > 0; o >>= 1) {
      const unsigned long long w = __shfl_down_sync(0xffffffffu, v, o);
      v = w < v ? w : v;
    }
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// The slot a projection of class `c` takes: a free one if any (the lowest index), else -- when
// `may_evict` -- the unpinned slot with the smallest (stamp, key). -1 when there is none. A free
// slot's ordering value is its index (< 2^20); an occupied one's is stamp << 20 | key with
// stamp >= 1, so every free slot orders first; a slot stamped `now` is pinned.
__device__ int32_t find_slot(const Dev &d, uint32_t c, unsigned long long now, bool may_evict,
                             unsigned long long *scratch) {
  unsigned long long best = kNoSlot;
  for (uint32_t s = threadIdx.x; s < d.capacity[c]; s += blockDim.x) {
    const uint32_t o = d.owner[c][s];
    unsigned long long v;
    if (o == kNone) {
      v = s;
    } else if (!may_evict) {
      v = kNoSlot;
    } else {
      const unsigned long long stamp = d.stamp[c][s];
      v = stamp == now ? kNoSlot : (stamp << kKeyBits) | o;
    }
    best = v < best ? v : best;
  }
  best = block_min(best, scratch);
  if (best == kNoSlot) return -1;
  if (best < (1ull << kKeyBits)) return static_cast<int32_t>(best);
  return d.slot_of[best & ((1u << kKeyBits) - 1)];
}

__device__ void report_push(const Dev &d, uint32_t *entries, uint32_t list, uint32_t *count,
                            uint32_t entry) {
  if (entries != nullptr) entries[list * 4 * d.experts + *count] = entry;
  ++*count;
}

// Places `key` in slot `slot` of class `c` (one thread): evicts the slot's owner, writes the
// slot table, queues the copy.
__device__ void place(const Dev &d, uint32_t key, uint32_t c, int32_t slot,
                      unsigned long long now, bool prefetched, Job *jobs, uint32_t *n_jobs,
                      uint32_t *entries, uint32_t *count) {
  const uint32_t old = d.owner[c][slot];
  if (old == kNone) {
    ++d.st->used[c];
  } else {
    d.slot_of[old] = -1;
    d.flags[old] = 0;
    d.table[old] = ignis_moe_slot{nullptr, 0, 0};
    report_push(d, entries, kEvictions, &count[kEvictions], old);
  }
  d.owner[c][slot] = key;
  d.stamp[c][slot] = now;
  d.slot_of[key] = slot;
  d.flags[key] = prefetched ? kPrefetched : 0;
  uint8_t *dst = d.pool[c] + static_cast<unsigned long long>(slot) * d.record_bytes[c];
  d.table[key] = ignis_moe_slot{dst, d.k2[key], 0};
  jobs[(*n_jobs)++] = Job{reinterpret_cast<const uint4 *>(d.host + d.host_off[key]),
                          reinterpret_cast<uint4 *>(dst), d.record_bytes[d.cls[key]] / 16};
}

// Stages `key` in ring half `h`, tagged `layer` (one thread).
__device__ void stage(const Dev &d, uint32_t key, uint32_t h, uint32_t layer, Job *jobs,
                      uint32_t *n_jobs) {
  const unsigned long long bytes = d.record_bytes[d.cls[key]];
  const unsigned long long off = d.st->half_fill[h];
  if (off + bytes > d.half_bytes) __trap();  // the plan sizes a half for a whole layer
  d.st->half_fill[h] = off + bytes;
  d.st->half_layer[h] = layer;
  d.half_keys[h * 2 * d.experts + d.st->half_count[h]++] = key;
  uint8_t *dst = d.ring + h * d.half_bytes + off;
  d.flags[key] = kStaged;
  d.table[key] = ignis_moe_slot{dst, d.k2[key], 0};
  jobs[(*n_jobs)++] = Job{reinterpret_cast<const uint4 *>(d.host + d.host_off[key]),
                          reinterpret_cast<uint4 *>(dst), bytes / 16};
}

using BlockScan = cub::BlockScan<uint32_t, kThreads>;

// A step is two kernels: resolve_demand classifies the selection and places its misses (what
// the expert op waits for); resolve_prefetch then takes the lookahead's candidates. Run back to
// back they are the policy's one step. Split, the prefetch half runs on residency's prefetch
// stream beside the expert op: its candidates never evict a projection the step selected (the
// demand half stamped them `now`), so it changes nothing the op reads.
__global__ void __launch_bounds__(kThreads)
    resolve_demand(Dev d, uint32_t layer, uint32_t phase, const int32_t *ids, uint32_t tokens,
                   const int32_t *lookahead, uint32_t rows, uint32_t stride) {
  __shared__ uint32_t s_sel[IGNIS_MOE_EXPERTS / 32];
  __shared__ uint32_t s_list[kMaxKeysPerLayer];
  __shared__ uint32_t s_need[kClasses], s_pinned[kClasses], s_hits[kClasses];
  __shared__ unsigned long long s_scratch[32];
  __shared__ uint32_t s_status, s_n, s_prefetch_hits;
  __shared__ BlockScan::TempStorage s_scan;

  const uint32_t t = threadIdx.x;
  const uint32_t E = d.experts, nk = 2 * E;
  const bool decode = phase == IGNIS_RESIDENCY_DECODE;
  State *st = d.st;
  Report *report = d.report != nullptr ? d.report + layer : nullptr;
  uint32_t *entries =
      d.report_entries != nullptr ? d.report_entries + layer * kLists * 4 * d.experts : nullptr;
  const bool repeat = st->last_layer == layer;

  if (t < IGNIS_MOE_EXPERTS / 32) s_sel[t] = 0;
  if (t < kClasses) s_need[t] = s_pinned[t] = s_hits[t] = 0;
  if (t == 0) s_status = s_n = s_prefetch_hits = 0;
  __syncthreads();

  // 1. The selection, and the lookahead when the step was given it whole, validated before
  //    anything changes.
  for (uint32_t i = t; i < tokens * kTopK; i += blockDim.x) {
    const int32_t e = ids[i];
    if (e < 0 || static_cast<uint32_t>(e) >= E) {
      s_status = IGNIS_RESIDENCY_STATUS_INVALID;
    } else {
      atomicOr(&s_sel[e >> 5], 1u << (e & 31));
    }
  }
  if (lookahead != nullptr) {
    const uint32_t width = decode ? d.width : d.prefill_width;
    for (uint32_t row = t; row < rows; row += blockDim.x) {
      uint32_t rank = 0;
      for (uint32_t j = 0; j < stride && rank < width; ++j) {
        const int32_t e = lookahead[static_cast<unsigned long long>(row) * stride + j];
        if (e < 0) continue;
        if (static_cast<uint32_t>(e) >= E) {
          s_status = IGNIS_RESIDENCY_STATUS_INVALID;
          break;
        }
        ++rank;
      }
    }
  }
  __syncthreads();
  if (s_status != 0) {
    if (t == 0 && report != nullptr) {
      report->status = s_status;
      for (uint32_t l = 0; l < kLists; ++l) report->count[l] = 0;
      report->bytes_moved = 0;
    }
    if (t == 0) {
      st->n_demand = st->n_prefetch = 0;
      st->step_status = s_status;
    }
    return;
  }

  // 2. Classify each key of the layer: one thread per key.
  const uint32_t key = layer * nk + t;
  bool selected = false, resident = false, staged = false;
  uint32_t c = 0;
  if (t < nk) {
    selected = (s_sel[(t >> 1) >> 5] >> ((t >> 1) & 31)) & 1u;
    if (selected) {
      c = d.cls[key];
      resident = d.slot_of[key] >= 0;
      staged = !resident && !repeat && (d.flags[key] & kStaged) &&
               st->half_layer[layer & 1] == layer;
      if (resident) atomicAdd(&s_pinned[c], 1u);
      if (!resident && !staged) atomicAdd(&s_need[c], 1u);
    }
  }
  const bool hit = selected && (resident || staged);
  const bool miss = selected && !hit;
  __syncthreads();

  // 3. A decode step whose class cannot hold its misses is refused whole.
  if (decode && t == 0) {
    for (uint32_t k = 0; k < kClasses; ++k) {
      if (s_need[k] > d.capacity[k] - s_pinned[k]) {
        s_status = 1 + k;
        break;
      }
    }
  }
  __syncthreads();
  if (s_status != 0) {
    if (t == 0 && report != nullptr) {
      report->status = s_status;
      for (uint32_t l = 0; l < kLists; ++l) report->count[l] = 0;
      report->bytes_moved = 0;
    }
    if (t == 0) {
      st->n_demand = st->n_prefetch = 0;
      st->step_status = s_status;
    }
    return;
  }

  // 4. Committed: the clock, and the ring halves staged for any other layer -- or, after a step
  //    of this same layer, for this one too -- are released.
  const unsigned long long now = st->clock + 1;
  for (uint32_t h = 0; h < 2; ++h) {
    const uint32_t tag = st->half_layer[h];
    if (tag != kNone && (tag != layer || repeat)) {
      for (uint32_t i = t; i < st->half_count[h]; i += blockDim.x) {
        const uint32_t k = d.half_keys[h * nk + i];
        d.table[k] = ignis_moe_slot{nullptr, 0, 0};
        d.flags[k] = 0;
      }
    }
  }
  __syncthreads();
  if (t == 0) {
    st->clock = now;
    st->last_layer = layer;
    for (uint32_t h = 0; h < 2; ++h) {
      if (st->half_layer[h] != kNone && (st->half_layer[h] != layer || repeat)) {
        st->half_layer[h] = kNone;
        st->half_count[h] = 0;
        st->half_fill[h] = 0;
      }
    }
    st->n_demand = 0;
    st->n_prefetch = 0;
  }

  // 5. Hits: stamped (the LRU refresh); a prefetch's first use is a prefetch hit.
  bool prefetch_hit = false;
  if (hit) {
    if (resident) {
      d.stamp[c][d.slot_of[key]] = now;
      prefetch_hit = d.flags[key] & kPrefetched;
      d.flags[key] &= static_cast<uint8_t>(~kPrefetched);
    } else {
      prefetch_hit = true;  // staged for this layer by the previous step's lookahead
    }
    atomicAdd(&s_hits[c], 1u);
    if (prefetch_hit) atomicAdd(&s_prefetch_hits, 1u);
  }
  uint32_t pos_hit, pos_pf, pos_miss, total;
  BlockScan(s_scan).ExclusiveSum(hit ? 1u : 0u, pos_hit, total);
  __syncthreads();
  if (hit && entries) entries[kHits * 2 * nk + pos_hit] = key;
  BlockScan(s_scan).ExclusiveSum(prefetch_hit ? 1u : 0u, pos_pf, total);
  __syncthreads();
  if (prefetch_hit && entries) entries[kPrefetchHits * 2 * nk + pos_pf] = key;
  BlockScan(s_scan).ExclusiveSum(miss ? 1u : 0u, pos_miss, total);
  __syncthreads();
  if (miss) s_list[pos_miss] = key;
  if (t == 0) s_n = total;
  __syncthreads();

  // Per-step tallies, kept by thread 0.
  uint32_t count[kLists] = {0, 0, 0, 0, 0, 0};
  unsigned long long moved = 0;
  uint32_t n_demand = 0;

  // 6. Misses, in key order: a free slot, else the LRU victim (decode) or the ring (prefill).
  const uint32_t n_miss = s_n;
  for (uint32_t m = 0; m < n_miss; ++m) {
    const uint32_t k = s_list[m];
    const uint32_t mc = d.cls[k];
    const bool room = st->used[mc] < d.capacity[mc];
    int32_t slot = -1;
    if (decode || room) slot = find_slot(d, mc, now, decode, s_scratch);
    if (t == 0) {
      const unsigned long long bytes = d.record_bytes[mc];
      if (slot >= 0) {
        place(d, k, mc, slot, now, false, d.demand, &n_demand, entries, count);
        report_push(d, entries, kMisses, &count[kMisses], k);
      } else {
        stage(d, k, layer & 1, layer, d.demand, &n_demand);
        report_push(d, entries, kMisses, &count[kMisses], k | IGNIS_RESIDENCY_STAGING_BIT);
      }
      atomicAdd(&d.counters->misses[mc][phase], 1ull);
      moved += bytes;
    }
    __syncthreads();
  }

  // 7. Tallies; the prefetch half, if one follows, adds its own.
  if (t < kClasses && s_hits[t] != 0) atomicAdd(&d.counters->hits[t][phase], s_hits[t]);
  if (t == 0) {
    d.counters->prefetch_used += s_prefetch_hits;
    d.counters->bytes_moved[phase] += moved;
    st->n_demand = n_demand;
    count[kHits] = 0;
    count[kPrefetchHits] = s_prefetch_hits;
    for (uint32_t k = 0; k < kClasses; ++k) count[kHits] += s_hits[k];
    st->step_status = 0;
    for (uint32_t l = 0; l < kLists; ++l) st->step_count[l] = count[l];
    st->step_moved = moved;
    if (report != nullptr) {
      report->status = 0;
      for (uint32_t l = 0; l < kLists; ++l) report->count[l] = count[l];
      report->bytes_moved = moved;
    }
  }

  // 8. The host mirror, at the last layer of a step (it has no lookahead, so no prefetch half):
  //    the totals, each word stored whole.
  if (d.mirror != nullptr && layer + 1 == d.layers) {
    __syncthreads();
    const unsigned long long *from = reinterpret_cast<const unsigned long long *>(d.counters);
    unsigned long long *to = reinterpret_cast<unsigned long long *>(&d.mirror->counters);
    for (uint32_t i = t; i < sizeof(Counters) / 8; i += blockDim.x) to[i] = from[i];
    if (t < kClasses) d.mirror->in_use[t] = st->used[t];
  }
}

// The step's lookahead (`layer` + 1 < layers): candidates in rank order, both projections of
// each, prefetched within the decode budget. Nothing when the demand half refused the step. A
// lookahead the demand half did not validate (the split step) with an id outside [0, experts)
// prefetches nothing and reports IGNIS_RESIDENCY_STATUS_INVALID; the demand half stands.
__global__ void __launch_bounds__(kThreads)
    resolve_prefetch(Dev d, uint32_t layer, uint32_t phase, const int32_t *lookahead,
                     uint32_t rows, uint32_t stride) {
  __shared__ uint32_t s_pos[IGNIS_MOE_EXPERTS];
  __shared__ uint32_t s_order[IGNIS_MOE_EXPERTS];  // a candidate expert's place in s_list / 2
  __shared__ uint32_t s_list[kMaxKeysPerLayer];
  __shared__ uint8_t s_skip[kMaxKeysPerLayer];     // s_list[i] is resident or staged now
  __shared__ unsigned long long s_scratch[32];
  __shared__ uint32_t s_status, s_n_cand;

  const uint32_t t = threadIdx.x;
  const uint32_t E = d.experts, nk = 2 * E;
  const bool decode = phase == IGNIS_RESIDENCY_DECODE;
  State *st = d.st;
  if (st->step_status != 0) return;
  Report *report = d.report != nullptr ? d.report + layer : nullptr;
  uint32_t *entries =
      d.report_entries != nullptr ? d.report_entries + layer * kLists * 4 * d.experts : nullptr;

  if (t < IGNIS_MOE_EXPERTS) s_pos[t] = s_order[t] = kNone;
  if (t == 0) s_status = s_n_cand = 0;
  __syncthreads();

  // The candidates' first-occurrence rank positions.
  const uint32_t width = decode ? d.width : d.prefill_width;
  for (uint32_t row = t; row < rows; row += blockDim.x) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < stride && rank < width; ++j) {
      const int32_t e = lookahead[static_cast<unsigned long long>(row) * stride + j];
      if (e < 0) continue;
      if (static_cast<uint32_t>(e) >= E) {
        s_status = IGNIS_RESIDENCY_STATUS_INVALID;
        break;
      }
      atomicMin(&s_pos[e], rank * rows + row);
      ++rank;
    }
  }
  __syncthreads();
  if (s_status != 0) {
    if (t == 0 && report != nullptr) report->status = s_status;
    return;
  }

  const unsigned long long now = st->clock;  // the demand half's commit
  uint32_t count[kLists];
  for (uint32_t l = 0; l < kLists; ++l) count[l] = st->step_count[l];
  unsigned long long moved = 0;
  uint32_t n_prefetch = 0;

  // Each candidate's place in rank order and, read all at once, whether its projections are
  // resident or staged; the walk below keeps that exact, clearing a later candidate's flag when
  // an earlier one evicts it.
  if (t < E && s_pos[t] != kNone) {
    uint32_t order = 0;
    for (uint32_t e = 0; e < E; ++e) order += s_pos[e] < s_pos[t];
    s_order[t] = order;
    for (uint32_t j = 0; j < 2; ++j) {
      const uint32_t k = (layer + 1) * nk + 2 * t + j;
      s_list[2 * order + j] = k;
      s_skip[2 * order + j] = d.slot_of[k] >= 0 || (d.flags[k] & kStaged);
    }
    atomicAdd(&s_n_cand, 1u);
  }
  __syncthreads();
  const uint32_t n_cand = 2 * s_n_cand;
  unsigned long long spent = 0;
  for (uint32_t i = 0; i < n_cand; ++i) {
    if (s_skip[i]) continue;  // block-uniform: last written before a barrier
    const uint32_t k = s_list[i];
    const uint32_t kc = d.cls[k];
    const unsigned long long bytes = d.record_bytes[kc];
    if (decode && d.budget != IGNIS_RESIDENCY_NO_BUDGET && spent + bytes > d.budget) {
      if (t == 0) report_push(d, entries, kDropped, &count[kDropped], k);
      __syncthreads();
      continue;
    }
    const bool room = st->used[kc] < d.capacity[kc];
    int32_t slot = -1;
    if (decode || room) slot = find_slot(d, kc, now, decode, s_scratch);
    if (t == 0) {
      if (slot >= 0) {
        const uint32_t old = d.owner[kc][slot];
        place(d, k, kc, slot, now, true, d.prefetch, &n_prefetch, entries, count);
        report_push(d, entries, kPrefetches, &count[kPrefetches], k);
        // A later candidate it evicted is no longer resident at its turn.
        if (old != kNone && old / nk == layer + 1 && s_order[(old % nk) >> 1] != kNone) {
          s_skip[2 * s_order[(old % nk) >> 1] + (old & 1)] = 0;
        }
      } else if (!decode) {
        stage(d, k, (layer + 1) & 1, layer + 1, d.prefetch, &n_prefetch);
        report_push(d, entries, kPrefetches, &count[kPrefetches], k | IGNIS_RESIDENCY_STAGING_BIT);
      } else {
        report_push(d, entries, kDropped, &count[kDropped], k);
      }
      if (slot >= 0 || !decode) {
        ++d.counters->prefetch_issued;
        moved += bytes;
      }
    }
    if (slot >= 0 || !decode) spent += bytes;
    __syncthreads();
  }

  if (t == 0) {
    d.counters->bytes_moved[phase] += moved;
    st->n_prefetch = n_prefetch;
    if (report != nullptr) {
      for (uint32_t l = 0; l < kLists; ++l) report->count[l] = count[l];
      report->bytes_moved = st->step_moved + moved;
    }
  }
}

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// All blocks walk the `n` jobs in order and share each one.
__device__ void copy_all(const Job *jobs, uint32_t n) {
  const unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long threads = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (uint32_t j = 0; j < n; ++j) {
    const Job job = jobs[j];
    for (unsigned long long v = tid; v < job.vectors; v += threads) job.dst[v] = job.src[v];
  }
}

// Copies every queued job (the prefetch copy).
__global__ void copy_jobs(const Job *jobs, const uint32_t *n_jobs) { copy_all(jobs, *n_jobs); }

// The demand copy, what the expert op waits for: copies every queued job, then adds its device
// time -- the earliest block's start to the last block's end -- to `*stall` and stores the new
// total in `*mirror` too, when there is one. A copy with no job takes no time.
__global__ void copy_jobs_timed(const Job *jobs, const uint32_t *n_jobs, State *st, unsigned long long *stall,
                                unsigned long long *mirror) {
  const uint32_t n = *n_jobs;
  if (n == 0) return;
  if (threadIdx.x == 0) atomicMin(&st->copy_start, globaltimer());
  copy_all(jobs, n);
  __syncthreads();
  if (threadIdx.x != 0) return;
  __threadfence();
  if (atomicAdd(&st->copy_done, 1u) != gridDim.x - 1) return;
  const unsigned long long start = atomicExch(&st->copy_start, kNoTime);
  const unsigned long long total = *stall + (globaltimer() - start);
  *stall = total;
  if (mirror != nullptr) *mirror = total;
  st->copy_done = 0;
}

// Each row's top `width` experts by BF16-rounded logit, ties to the lower id, best first: the
// router's own ranking (ignis_moe.h). One warp per row.
__global__ void rank_lookahead(const float *logits, uint32_t rows, uint32_t experts,
                               uint32_t width, int32_t *out) {
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  constexpr uint32_t kPerLane = IGNIS_MOE_EXPERTS / 32;
  float v[kPerLane];
  uint32_t taken = 0;
  for (uint32_t i = 0; i < kPerLane; ++i) {
    const uint32_t e = lane + 32 * i;
    v[i] = e < experts
               ? __bfloat162float(__float2bfloat16_rn(logits[static_cast<unsigned long long>(row) * experts + e]))
               : 0.0f;
    if (e >= experts) taken |= 1u << i;
  }
  for (uint32_t w = 0; w < width; ++w) {
    float best = 0.0f;
    uint32_t best_e = kNone;
    for (uint32_t i = 0; i < kPerLane; ++i) {
      if (taken & (1u << i)) continue;
      const uint32_t e = lane + 32 * i;
      if (best_e == kNone || v[i] > best || (v[i] == best && e < best_e)) {
        best = v[i];
        best_e = e;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_down_sync(0xffffffffu, best, o);
      const uint32_t oe = __shfl_down_sync(0xffffffffu, best_e, o);
      if (oe != kNone && (best_e == kNone || ob > best || (ob == best && oe < best_e))) {
        best = ob;
        best_e = oe;
      }
    }
    best_e = __shfl_sync(0xffffffffu, best_e, 0);
    if (lane == 0) out[static_cast<unsigned long long>(row) * width + w] = best_e == kNone ? -1 : static_cast<int32_t>(best_e);
    if (best_e != kNone && (best_e & 31) == lane) taken |= 1u << (best_e >> 5);
  }
}

// ---- host side ----------------------------------------------------------------------------------

constexpr uint64_t kAlign = 256;

uint64_t round_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

// A step's lookahead width: a prefill step's own, else the decode width.
uint32_t width_of(const ignis_residency_desc &d, uint32_t phase) {
  return phase == IGNIS_RESIDENCY_PREFILL ? d.prefill_lookahead_width : d.lookahead_width;
}

// The device buffers besides pools and ring, in carve order: (bytes, element alignment).
struct Carve {
  uint64_t cls, k2, host_off, slot_of, flags, table, owner[kClasses], stamp[kClasses],
      half_keys, state, demand, prefetch, counters, report, report_entries, ranked;
  uint64_t total;
};

Carve carve(const ignis_residency_desc &d) {
  const uint64_t keys = static_cast<uint64_t>(d.layers) * d.experts * 2;
  Carve c{};
  uint64_t at = 0;
  auto take = [&](uint64_t bytes) {
    const uint64_t here = at;
    at += round_up(bytes);
    return here;
  };
  c.cls = take(keys);
  c.k2 = take(keys);
  c.host_off = take(keys * 8);
  c.slot_of = take(keys * 4);
  c.flags = take(keys);
  c.table = take(keys * sizeof(ignis_moe_slot));
  for (uint32_t k = 0; k < kClasses; ++k) c.owner[k] = take(static_cast<uint64_t>(d.capacity[k]) * 4);
  for (uint32_t k = 0; k < kClasses; ++k) c.stamp[k] = take(static_cast<uint64_t>(d.capacity[k]) * 8);
  c.half_keys = take(2ull * 2 * d.experts * 4);
  c.state = take(sizeof(State));
  c.demand = take(2ull * d.experts * sizeof(Job));
  c.prefetch = take(2ull * d.experts * sizeof(Job));
  c.counters = take(sizeof(Counters));
  c.report = take(d.report ? sizeof(Report) * d.layers : 0);
  c.report_entries = take(d.report ? static_cast<uint64_t>(d.layers) * kLists * 4 * d.experts * 4 : 0);
  c.ranked = take(static_cast<uint64_t>(d.max_tokens) * d.lookahead_width * 4);
  c.total = at;
  return c;
}

int32_t check_desc(const ignis_residency_desc *d) {
  if (d == nullptr) return fail("residency: no descriptor");
  if (d->layers == 0) return fail("residency: layers must be at least 1");
  if (d->experts == 0 || d->experts > IGNIS_MOE_EXPERTS) {
    return fail("residency: experts must be in 1.." + std::to_string(IGNIS_MOE_EXPERTS));
  }
  if (static_cast<uint64_t>(d->layers) * d->experts * 2 >= (1ull << kKeyBits)) {
    return fail("residency: layers * experts * 2 must stay below 2^20");
  }
  for (uint32_t k = 0; k < kClasses; ++k) {
    if (d->capacity[k] >= (1u << kKeyBits)) return fail("residency: a class capacity must stay below 2^20");
    if (d->record_bytes[k] == 0 || d->record_bytes[k] % 16 != 0) {
      return fail("residency: record bytes must be a non-zero multiple of 16");
    }
  }
  if (d->max_tokens == 0) return fail("residency: max_tokens must be at least 1");
  if (d->staging_half_bytes % 16 != 0) {
    return fail("residency: staging_half_bytes must be a multiple of 16 (the copies store 16-byte units)");
  }
  if (d->lookahead_width > IGNIS_MOE_EXPERTS) return fail("residency: lookahead_width exceeds the experts");
  if (d->prefill_lookahead_width > d->lookahead_width) {
    // The ranked lookahead's scratch is sized for lookahead_width rows' worth.
    return fail("residency: prefill_lookahead_width exceeds lookahead_width");
  }
  if (d->copy_blocks > 1024) return fail("residency: copy_blocks must be at most 1024");
  return 0;
}

}  // namespace

struct ignis_residency {
  ignis_residency_desc desc{};
  std::vector<uint8_t> k2, cls;
  std::vector<uint64_t> host_off;
  uint8_t *host = nullptr;
  uint8_t *host_dev = nullptr;
  uint8_t *pool[kClasses] = {};
  uint8_t *ring = nullptr;
  uint8_t *tables = nullptr;
  Carve layout{};
  Dev dev{};
  cudaStream_t prefetch_stream = nullptr;
  // fork: the demand resolve (split) or copy (whole) is done; demanded: the demand copy is (a
  // split step's prefetch copy waits on it, so the two never share the link); join: the
  // prefetch branch is.
  cudaEvent_t fork = nullptr, demanded = nullptr, join = nullptr;
  bool forked = false;
  // A split step's open branch: its layer and phase, and whether its join is recorded yet.
  bool branch_open = false;
  bool join_recorded = false;
  uint32_t branch_layer = 0, branch_phase = 0;
  bool stepped = false;
  ignis_residency_mirror *mirror_host = nullptr;  // registered by ignis_residency_set_mirror
};

namespace {

// The step's demand copy on `s`, timed into its phase's stall (and the mirror's).
int32_t demand_copy(ignis_residency *r, uint32_t phase, cudaStream_t s) {
  auto *mirror = r->dev.mirror != nullptr
                     ? reinterpret_cast<unsigned long long *>(&r->dev.mirror->counters.stall_nanos[phase])
                     : nullptr;
  copy_jobs_timed<<<r->desc.copy_blocks, 256, 0, s>>>(r->dev.demand, &r->dev.st->n_demand, r->dev.st,
                                                &r->dev.counters->stall_nanos[phase], mirror);
  RESIDENCY_CUDA(cudaGetLastError());
  return 0;
}

}  // namespace

extern "C" {

int32_t ignis_residency_plan_bytes(const ignis_residency_desc *desc, ignis_residency_plan *plan) {
  if (check_desc(desc) != 0) return -1;
  if (plan == nullptr) return fail("residency: no plan to write");
  plan->pools = 0;
  for (uint32_t k = 0; k < kClasses; ++k) {
    plan->pools += round_up(static_cast<uint64_t>(desc->capacity[k]) * desc->record_bytes[k]);
  }
  plan->staging = round_up(2 * desc->staging_half_bytes);
  plan->tables = carve(*desc).total;
  plan->total = plan->pools + plan->staging + plan->tables;
  return 0;
}

int32_t ignis_residency_create(const ignis_residency_desc *desc, const uint8_t *k2,
                               const uint64_t *pool_offsets, ignis_residency **out) {
  if (check_desc(desc) != 0) return -1;
  if (k2 == nullptr || pool_offsets == nullptr || out == nullptr) {
    return fail("residency: the K map, the pool offsets and the out pointer are required");
  }
  *out = nullptr;
  const ignis_residency_desc &d = *desc;
  const uint64_t keys = static_cast<uint64_t>(d.layers) * d.experts * 2;
  auto r = new ignis_residency();
  r->desc = d;
  if (r->desc.copy_blocks == 0) r->desc.copy_blocks = 16;
  r->k2.assign(k2, k2 + keys);
  r->host_off.assign(pool_offsets, pool_offsets + keys);
  r->cls.resize(keys);
  uint64_t heaviest = 0;
  for (uint32_t l = 0; l < d.layers; ++l) {
    uint64_t layer_bytes = 0;
    for (uint64_t i = 0; i < 2ull * d.experts; ++i) {
      const uint64_t key = static_cast<uint64_t>(l) * d.experts * 2 + i;
      const uint8_t v = r->k2[key];
      if (v != 4 && v != 5 && v != 6 && v != 8) {
        delete r;
        return fail("residency: k2 of key " + std::to_string(key) + " is " + std::to_string(v) +
                    ", not 4, 5, 6 or 8");
      }
      const uint32_t projection = static_cast<uint32_t>(i & 1);
      const uint32_t c = projection * 4 + (v == 4 ? 0 : v == 5 ? 1 : v == 6 ? 2 : 3);
      r->cls[key] = static_cast<uint8_t>(c);
      const uint64_t bytes = d.record_bytes[c];
      if (r->host_off[key] % 16 != 0 || r->host_off[key] + bytes > d.host_pool_bytes) {
        delete r;
        return fail("residency: record of key " + std::to_string(key) +
                    " is unaligned or outside the host pool");
      }
      layer_bytes += bytes;
    }
    heaviest = std::max(heaviest, layer_bytes);
  }
  if (d.staging_half_bytes < heaviest) {
    // A prefill can touch a whole layer; a smaller half could overflow mid-chunk.
    delete r;
    return fail("residency: staging_half_bytes " + std::to_string(d.staging_half_bytes) +
                " is below the heaviest layer's " + std::to_string(heaviest));
  }
  auto cleanup = [&](int32_t rc) {
    ignis_residency_free(r);
    return rc;
  };

  cudaError_t e = cudaHostAlloc(reinterpret_cast<void **>(&r->host), d.host_pool_bytes,
                                cudaHostAllocMapped | cudaHostAllocPortable);
  if (e != cudaSuccess) {
    r->host = nullptr;
    return cleanup(fail(std::string("residency: cudaHostAlloc of the expert pool: ") + cudaGetErrorString(e)));
  }
  e = cudaHostGetDevicePointer(reinterpret_cast<void **>(&r->host_dev), r->host, 0);
  if (e != cudaSuccess) return cleanup(fail(std::string("residency: cudaHostGetDevicePointer: ") + cudaGetErrorString(e)));
  for (uint32_t k = 0; k < kClasses; ++k) {
    const uint64_t bytes = static_cast<uint64_t>(d.capacity[k]) * d.record_bytes[k];
    if (bytes == 0) continue;
    e = cudaMalloc(reinterpret_cast<void **>(&r->pool[k]), bytes);
    if (e != cudaSuccess) return cleanup(fail(std::string("residency: cudaMalloc of a class pool: ") + cudaGetErrorString(e)));
  }
  if (d.staging_half_bytes > 0) {
    e = cudaMalloc(reinterpret_cast<void **>(&r->ring), 2 * d.staging_half_bytes);
    if (e != cudaSuccess) return cleanup(fail(std::string("residency: cudaMalloc of the staging ring: ") + cudaGetErrorString(e)));
  }
  r->layout = carve(r->desc);
  e = cudaMalloc(reinterpret_cast<void **>(&r->tables), r->layout.total);
  if (e != cudaSuccess) return cleanup(fail(std::string("residency: cudaMalloc of the tables: ") + cudaGetErrorString(e)));

  // Initial contents, staged on the host and copied once.
  std::vector<uint8_t> image(r->layout.total, 0);
  std::memcpy(image.data() + r->layout.cls, r->cls.data(), keys);
  std::memcpy(image.data() + r->layout.k2, r->k2.data(), keys);
  std::memcpy(image.data() + r->layout.host_off, r->host_off.data(), keys * 8);
  std::memset(image.data() + r->layout.slot_of, 0xFF, keys * 4);  // -1: nowhere
  for (uint32_t k = 0; k < kClasses; ++k) {
    std::memset(image.data() + r->layout.owner[k], 0xFF, static_cast<uint64_t>(d.capacity[k]) * 4);
  }
  State state{};
  state.half_layer[0] = state.half_layer[1] = kNone;
  state.last_layer = kNone;
  state.copy_start = kNoTime;
  std::memcpy(image.data() + r->layout.state, &state, sizeof(State));
  e = cudaMemcpy(r->tables, image.data(), image.size(), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return cleanup(fail(std::string("residency: initializing the tables: ") + cudaGetErrorString(e)));

  Dev &v = r->dev;
  v.layers = d.layers;
  v.experts = d.experts;
  v.width = d.lookahead_width;
  v.prefill_width = d.prefill_lookahead_width;
  for (uint32_t k = 0; k < kClasses; ++k) {
    v.capacity[k] = d.capacity[k];
    v.record_bytes[k] = d.record_bytes[k];
    v.pool[k] = r->pool[k];
    v.owner[k] = reinterpret_cast<uint32_t *>(r->tables + r->layout.owner[k]);
    v.stamp[k] = reinterpret_cast<unsigned long long *>(r->tables + r->layout.stamp[k]);
  }
  v.budget = d.prefetch_budget_bytes;
  v.half_bytes = d.staging_half_bytes;
  v.cls = r->tables + r->layout.cls;
  v.k2 = r->tables + r->layout.k2;
  v.host_off = reinterpret_cast<const unsigned long long *>(r->tables + r->layout.host_off);
  v.host = r->host_dev;
  v.slot_of = reinterpret_cast<int32_t *>(r->tables + r->layout.slot_of);
  v.flags = r->tables + r->layout.flags;
  v.table = reinterpret_cast<ignis_moe_slot *>(r->tables + r->layout.table);
  v.ring = r->ring;
  v.half_keys = reinterpret_cast<uint32_t *>(r->tables + r->layout.half_keys);
  v.st = reinterpret_cast<State *>(r->tables + r->layout.state);
  v.demand = reinterpret_cast<Job *>(r->tables + r->layout.demand);
  v.prefetch = reinterpret_cast<Job *>(r->tables + r->layout.prefetch);
  v.counters = reinterpret_cast<Counters *>(r->tables + r->layout.counters);
  v.report = d.report ? reinterpret_cast<Report *>(r->tables + r->layout.report) : nullptr;
  v.report_entries = d.report ? reinterpret_cast<uint32_t *>(r->tables + r->layout.report_entries) : nullptr;

  e = cudaStreamCreateWithFlags(&r->prefetch_stream, cudaStreamNonBlocking);
  if (e != cudaSuccess) return cleanup(fail(std::string("residency: the prefetch stream: ") + cudaGetErrorString(e)));
  e = cudaEventCreateWithFlags(&r->fork, cudaEventDisableTiming);
  if (e == cudaSuccess) e = cudaEventCreateWithFlags(&r->demanded, cudaEventDisableTiming);
  if (e == cudaSuccess) e = cudaEventCreateWithFlags(&r->join, cudaEventDisableTiming);
  if (e != cudaSuccess) return cleanup(fail(std::string("residency: the fork/join events: ") + cudaGetErrorString(e)));
  *out = r;
  return 0;
}

void ignis_residency_free(ignis_residency *r) {
  if (r == nullptr) return;
  cudaDeviceSynchronize();
  if (r->join) cudaEventDestroy(r->join);
  if (r->demanded) cudaEventDestroy(r->demanded);
  if (r->fork) cudaEventDestroy(r->fork);
  if (r->prefetch_stream) cudaStreamDestroy(r->prefetch_stream);
  if (r->mirror_host) cudaHostUnregister(r->mirror_host);
  if (r->tables) cudaFree(r->tables);
  if (r->ring) cudaFree(r->ring);
  for (uint32_t k = 0; k < kClasses; ++k) {
    if (r->pool[k]) cudaFree(r->pool[k]);
  }
  if (r->host) cudaFreeHost(r->host);
  delete r;
}

void *ignis_residency_host_pool(ignis_residency *r) { return r ? r->host : nullptr; }

const ignis_moe_slot *ignis_residency_slot_table(ignis_residency *r, uint32_t layer) {
  if (r == nullptr || layer >= r->desc.layers) {
    fail("residency: no such layer");
    return nullptr;
  }
  return r->dev.table + static_cast<uint64_t>(layer) * r->desc.experts * 2;
}

int32_t ignis_residency_warm_start(ignis_residency *r, const uint32_t *keys, uint32_t n,
                                   uint32_t *admitted) {
  if (r == nullptr || admitted == nullptr || (n > 0 && keys == nullptr)) return fail("residency: warm start arguments");
  if (r->stepped) return fail("residency: the warm start comes before the first step");
  const ignis_residency_desc &d = r->desc;
  const uint64_t nkeys = static_cast<uint64_t>(d.layers) * d.experts * 2;
  RESIDENCY_CUDA(cudaDeviceSynchronize());
  State state{};
  RESIDENCY_CUDA(cudaMemcpy(&state, r->dev.st, sizeof(State), cudaMemcpyDeviceToHost));
  std::vector<int32_t> slot_of(nkeys);
  RESIDENCY_CUDA(cudaMemcpy(slot_of.data(), r->dev.slot_of, nkeys * 4, cudaMemcpyDeviceToHost));
  // The model's rule: hottest first, a class with room admits, a repeat or a resident key is
  // skipped; then the least hot admitted gets the oldest stamp.
  std::vector<uint32_t> chosen;
  std::vector<uint32_t> room(kClasses);
  for (uint32_t k = 0; k < kClasses; ++k) room[k] = d.capacity[k] - state.used[k];
  std::vector<uint8_t> seen(nkeys, 0);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t key = keys[i];
    if (key >= nkeys || slot_of[key] >= 0 || seen[key]) continue;
    seen[key] = 1;
    const uint32_t c = r->cls[key];
    if (room[c] == 0) continue;
    --room[c];
    chosen.push_back(key);
  }
  for (uint32_t c = 0; c < kClasses; ++c) {
    std::vector<uint32_t> owner(d.capacity[c]);
    std::vector<unsigned long long> stamp(d.capacity[c]);
    if (d.capacity[c] == 0) continue;
    RESIDENCY_CUDA(cudaMemcpy(owner.data(), r->dev.owner[c], owner.size() * 4, cudaMemcpyDeviceToHost));
    RESIDENCY_CUDA(cudaMemcpy(stamp.data(), r->dev.stamp[c], stamp.size() * 8, cudaMemcpyDeviceToHost));
    unsigned long long clock = state.clock;
    for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
      ++clock;
      if (r->cls[*it] != c) continue;
      uint32_t slot = 0;
      while (owner[slot] != kNone) ++slot;
      owner[slot] = *it;
      stamp[slot] = clock;
      slot_of[*it] = static_cast<int32_t>(slot);
      ++state.used[c];
      uint8_t *dst = r->pool[c] + static_cast<uint64_t>(slot) * d.record_bytes[c];
      RESIDENCY_CUDA(cudaMemcpy(dst, r->host + r->host_off[*it], d.record_bytes[c], cudaMemcpyHostToDevice));
      const ignis_moe_slot entry{dst, r->k2[*it], 0};
      RESIDENCY_CUDA(cudaMemcpy(r->dev.table + *it, &entry, sizeof(entry), cudaMemcpyHostToDevice));
    }
    RESIDENCY_CUDA(cudaMemcpy(r->dev.owner[c], owner.data(), owner.size() * 4, cudaMemcpyHostToDevice));
    RESIDENCY_CUDA(cudaMemcpy(r->dev.stamp[c], stamp.data(), stamp.size() * 8, cudaMemcpyHostToDevice));
  }
  state.clock += chosen.size();
  RESIDENCY_CUDA(cudaMemcpy(r->dev.slot_of, slot_of.data(), nkeys * 4, cudaMemcpyHostToDevice));
  RESIDENCY_CUDA(cudaMemcpy(r->dev.st, &state, sizeof(State), cudaMemcpyHostToDevice));
  *admitted = static_cast<uint32_t>(chosen.size());
  return 0;
}

int32_t ignis_residency_join(ignis_residency *r, void *stream) {
  if (r == nullptr) return fail("residency: no residency");
  if (!r->forked) return 0;
  // Cleared first: after a failed capture the join event belongs to a dead capture and the
  // wait fails; its copies never ran, so there is nothing left to join, and every later step
  // must not fail on it again.
  r->forked = false;
  r->branch_open = false;
  if (!r->join_recorded) {
    // A split step whose caller never took its prefetch half: the branch holds only what the
    // caller put on it.
    r->join_recorded = true;
    RESIDENCY_CUDA(cudaEventRecord(r->join, r->prefetch_stream));
  }
  RESIDENCY_CUDA(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), r->join, 0));
  return 0;
}

int32_t ignis_residency_step_ranked(ignis_residency *r, uint32_t layer, uint32_t phase,
                                    const int32_t *ids, uint32_t tokens, const int32_t *lookahead,
                                    uint32_t rows, uint32_t stride, void *stream) {
  if (r == nullptr) return fail("residency: no residency");
  const ignis_residency_desc &d = r->desc;
  if (layer >= d.layers) return fail("residency: layer " + std::to_string(layer) + " is out of range");
  if (phase != IGNIS_RESIDENCY_DECODE && phase != IGNIS_RESIDENCY_PREFILL) return fail("residency: phase must be decode or prefill");
  if (ids == nullptr || tokens == 0 || tokens > d.max_tokens) {
    return fail("residency: a step takes 1.." + std::to_string(d.max_tokens) + " rows of ids");
  }
  if (lookahead != nullptr && (rows == 0 || rows > d.max_tokens || stride == 0)) {
    return fail("residency: a lookahead takes 1.." + std::to_string(d.max_tokens) + " rows of a non-zero stride");
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (ignis_residency_join(r, stream) != 0) return -1;
  r->stepped = true;
  const bool look = lookahead != nullptr && layer + 1 < d.layers && width_of(d, phase) > 0;
  // Validated with the selection only when there is a next layer to look at: the last layer's
  // lookahead is ignored, as the policy ignores it.
  resolve_demand<<<1, kThreads, 0, s>>>(r->dev, layer, phase, ids, tokens, look ? lookahead : nullptr,
                                        look ? rows : 0, look ? stride : 0);
  RESIDENCY_CUDA(cudaGetLastError());
  if (look) {
    resolve_prefetch<<<1, kThreads, 0, s>>>(r->dev, layer, phase, lookahead, rows, stride);
    RESIDENCY_CUDA(cudaGetLastError());
  }
  if (demand_copy(r, phase, s) != 0) return -1;
  if (look) {
    RESIDENCY_CUDA(cudaEventRecord(r->fork, s));
    RESIDENCY_CUDA(cudaStreamWaitEvent(r->prefetch_stream, r->fork, 0));
    copy_jobs<<<d.copy_blocks, 256, 0, r->prefetch_stream>>>(r->dev.prefetch, &r->dev.st->n_prefetch);
    RESIDENCY_CUDA(cudaGetLastError());
    RESIDENCY_CUDA(cudaEventRecord(r->join, r->prefetch_stream));
    r->forked = true;
    r->join_recorded = true;
  }
  return 0;
}

int32_t ignis_residency_step_demand(ignis_residency *r, uint32_t layer, uint32_t phase,
                                    const int32_t *ids, uint32_t tokens, void *stream,
                                    void **lookahead_stream) {
  if (r == nullptr) return fail("residency: no residency");
  const ignis_residency_desc &d = r->desc;
  if (lookahead_stream != nullptr) *lookahead_stream = nullptr;
  if (layer >= d.layers) return fail("residency: layer " + std::to_string(layer) + " is out of range");
  if (phase != IGNIS_RESIDENCY_DECODE && phase != IGNIS_RESIDENCY_PREFILL) return fail("residency: phase must be decode or prefill");
  if (ids == nullptr || tokens == 0 || tokens > d.max_tokens) {
    return fail("residency: a step takes 1.." + std::to_string(d.max_tokens) + " rows of ids");
  }
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  if (ignis_residency_join(r, stream) != 0) return -1;
  r->stepped = true;
  const bool look = lookahead_stream != nullptr && layer + 1 < d.layers && width_of(d, phase) > 0;
  resolve_demand<<<1, kThreads, 0, s>>>(r->dev, layer, phase, ids, tokens, nullptr, 0, 0);
  RESIDENCY_CUDA(cudaGetLastError());
  if (look) {
    RESIDENCY_CUDA(cudaEventRecord(r->fork, s));
    RESIDENCY_CUDA(cudaStreamWaitEvent(r->prefetch_stream, r->fork, 0));
    r->forked = true;
    r->join_recorded = false;
  }
  if (demand_copy(r, phase, s) != 0) return -1;
  if (look) {
    RESIDENCY_CUDA(cudaEventRecord(r->demanded, s));
    r->branch_open = true;
    r->branch_layer = layer;
    r->branch_phase = phase;
    *lookahead_stream = r->prefetch_stream;
  }
  return 0;
}

int32_t ignis_residency_step_prefetch_ranked(ignis_residency *r, const int32_t *lookahead,
                                             uint32_t rows, uint32_t stride) {
  if (r == nullptr) return fail("residency: no residency");
  const ignis_residency_desc &d = r->desc;
  if (!r->branch_open) return fail("residency: no split step's lookahead branch is open");
  if (lookahead == nullptr || rows == 0 || rows > d.max_tokens || stride == 0) {
    return fail("residency: a lookahead takes 1.." + std::to_string(d.max_tokens) + " rows of a non-zero stride");
  }
  r->branch_open = false;
  const cudaStream_t p = r->prefetch_stream;
  resolve_prefetch<<<1, kThreads, 0, p>>>(r->dev, r->branch_layer, r->branch_phase, lookahead, rows, stride);
  RESIDENCY_CUDA(cudaGetLastError());
  RESIDENCY_CUDA(cudaStreamWaitEvent(p, r->demanded, 0));
  copy_jobs<<<d.copy_blocks, 256, 0, p>>>(r->dev.prefetch, &r->dev.st->n_prefetch);
  RESIDENCY_CUDA(cudaGetLastError());
  RESIDENCY_CUDA(cudaEventRecord(r->join, p));
  r->join_recorded = true;
  return 0;
}

int32_t ignis_residency_step_prefetch(ignis_residency *r, const float *lookahead_logits, uint32_t tokens) {
  if (r == nullptr) return fail("residency: no residency");
  const ignis_residency_desc &d = r->desc;
  if (!r->branch_open) return fail("residency: no split step's lookahead branch is open");
  if (lookahead_logits == nullptr || tokens == 0 || tokens > d.max_tokens) {
    return fail("residency: a lookahead takes 1.." + std::to_string(d.max_tokens) + " rows of logits");
  }
  int32_t *ranked = reinterpret_cast<int32_t *>(r->tables + r->layout.ranked);
  const uint32_t threads = 256;
  const uint32_t blocks = (tokens * 32 + threads - 1) / threads;
  const uint32_t width = width_of(d, r->branch_phase);
  rank_lookahead<<<blocks, threads, 0, r->prefetch_stream>>>(lookahead_logits, tokens, d.experts,
                                                            width, ranked);
  RESIDENCY_CUDA(cudaGetLastError());
  return ignis_residency_step_prefetch_ranked(r, ranked, tokens, width);
}

int32_t ignis_residency_step(ignis_residency *r, uint32_t layer, uint32_t phase, const int32_t *ids,
                             uint32_t tokens, const float *lookahead_logits, void *stream) {
  if (r == nullptr) return fail("residency: no residency");
  const ignis_residency_desc &d = r->desc;
  const uint32_t width = width_of(d, phase);
  const bool look = lookahead_logits != nullptr && layer + 1 < d.layers && width > 0;
  if (!look) return ignis_residency_step_ranked(r, layer, phase, ids, tokens, nullptr, 0, 0, stream);
  if (tokens == 0 || tokens > d.max_tokens) {
    return fail("residency: a step takes 1.." + std::to_string(d.max_tokens) + " rows of ids");
  }
  int32_t *ranked = reinterpret_cast<int32_t *>(r->tables + r->layout.ranked);
  const uint32_t threads = 256;
  const uint32_t blocks = (tokens * 32 + threads - 1) / threads;
  rank_lookahead<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
      lookahead_logits, tokens, d.experts, width, ranked);
  RESIDENCY_CUDA(cudaGetLastError());
  return ignis_residency_step_ranked(r, layer, phase, ids, tokens, ranked, tokens, width, stream);
}

int32_t ignis_residency_read_counters(ignis_residency *r, ignis_residency_counters *out) {
  if (r == nullptr || out == nullptr) return fail("residency: counters arguments");
  RESIDENCY_CUDA(cudaDeviceSynchronize());
  RESIDENCY_CUDA(cudaMemcpy(out, r->dev.counters, sizeof(*out), cudaMemcpyDeviceToHost));
  return 0;
}

int32_t ignis_residency_set_mirror(ignis_residency *r, ignis_residency_mirror *host) {
  if (r == nullptr || host == nullptr) return fail("residency: mirror arguments");
  if (r->stepped) return fail("residency: the mirror comes before the first step");
  if (r->mirror_host != nullptr) return fail("residency: already mirrored");
  RESIDENCY_CUDA(cudaDeviceSynchronize());
  RESIDENCY_CUDA(cudaHostRegister(host, sizeof(*host), cudaHostRegisterMapped));
  r->mirror_host = host;
  void *dev = nullptr;
  RESIDENCY_CUDA(cudaHostGetDevicePointer(&dev, host, 0));
  RESIDENCY_CUDA(cudaMemcpy(&host->counters, r->dev.counters, sizeof(host->counters), cudaMemcpyDeviceToHost));
  const char *used = reinterpret_cast<const char *>(r->dev.st) + offsetof(State, used);
  RESIDENCY_CUDA(cudaMemcpy(host->in_use, used, sizeof(State::used), cudaMemcpyDeviceToHost));
  r->dev.mirror = static_cast<ignis_residency_mirror *>(dev);
  return 0;
}

int32_t ignis_residency_read_occupancy(ignis_residency *r, uint32_t out[IGNIS_RESIDENCY_CLASSES]) {
  if (r == nullptr || out == nullptr) return fail("residency: occupancy arguments");
  RESIDENCY_CUDA(cudaDeviceSynchronize());
  const char *used = reinterpret_cast<const char *>(r->dev.st) + offsetof(State, used);
  RESIDENCY_CUDA(cudaMemcpy(out, used, sizeof(State::used), cudaMemcpyDeviceToHost));
  return 0;
}

int32_t ignis_residency_last_report(ignis_residency *r, uint32_t layer,
                                    ignis_residency_report *head, uint32_t *entries,
                                    uint32_t capacity) {
  if (r == nullptr || head == nullptr) return fail("residency: report arguments");
  if (r->dev.report == nullptr) return fail("residency: created without a report");
  if (layer >= r->desc.layers) return fail("residency: no such layer");
  RESIDENCY_CUDA(cudaDeviceSynchronize());
  RESIDENCY_CUDA(cudaMemcpy(head, r->dev.report + layer, sizeof(*head), cudaMemcpyDeviceToHost));
  if (head->status != 0) return 0;
  const uint64_t stride = 4ull * r->desc.experts;
  const uint32_t *base = r->dev.report_entries + static_cast<uint64_t>(layer) * kLists * stride;
  for (uint32_t l = 0; l < kLists; ++l) {
    if (head->count[l] > capacity) return fail("residency: a report list is longer than the capacity given");
    if (head->count[l] > 0 && entries == nullptr) return fail("residency: no entries buffer");
    if (head->count[l] > 0) {
      RESIDENCY_CUDA(cudaMemcpy(entries + static_cast<uint64_t>(l) * capacity, base + l * stride,
                                head->count[l] * 4ull, cudaMemcpyDeviceToHost));
    }
  }
  return 0;
}

int32_t ignis_residency_get_layout(ignis_residency *r, ignis_residency_layout *out) {
  if (r == nullptr || out == nullptr) return fail("residency: layout arguments");
  for (uint32_t k = 0; k < kClasses; ++k) {
    out->pool[k] = r->pool[k];
    out->pool_bytes[k] = static_cast<uint64_t>(r->desc.capacity[k]) * r->desc.record_bytes[k];
  }
  out->ring = r->ring;
  out->ring_bytes = 2 * r->desc.staging_half_bytes;
  return 0;
}

const char *ignis_residency_last_error(void) { return g_error.c_str(); }

}  // extern "C"
