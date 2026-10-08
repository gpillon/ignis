// ignis kernel leaf: the routed-expert decode's phase trace -- OURS, a diagnostic for the MoE
// microbenchmark (kernel/tests/bench_moe.cu), not part of the ABI in kernel/include/ignis_moe.h.
//
// ignis_moe_experts_decode_trace runs a decode route's kernel instantiated with %globaltimer
// stamps (nanoseconds, comparable across SMs). The trace holds IGNIS_MOE_TRACE_UNITS unit
// records, then one record per CTA of the launch, each IGNIS_MOE_TRACE_WORDS u64:
//   unit (at its ticket):  0 (SM id << 32) | CTA   1 kind (0 gate/up, 1 down) | u << 8 | block << 16 | split << 24
//                          2 begin   3 ready (the unit's operands in place: gate/up its rotated
//                          activations; down h, after the wait, rotated)   4 multiplied
//                          5 end (sums added and the arrival counted; for the eighth gate/up
//                          arrival, after it published h)   6 down: ns waiting for h; gate/up: 1
//                          on the arrival that ran the SwiGLU   7 0
//   CTA (at IGNIS_MOE_TRACE_UNITS + blockIdx.x):  0 entry   1 first ticket known   2 exit   3 SM id
// Ticket order is the route's own; a record left zero was not reached.
#ifndef IGNIS_MOE_TRACE_H
#define IGNIS_MOE_TRACE_H

#include "ignis_moe.h"

#include <stdint.h>

#define IGNIS_MOE_TRACE_WORDS 8
// Units of the largest decode call (8 tokens, 80 distinct experts, 40 gate/up + 20 down units
// each, at the finest split any route uses: room for 4x that).
#define IGNIS_MOE_TRACE_UNITS (4 * 80 * 60)
// CTAs a traced launch may have.
#define IGNIS_MOE_TRACE_CTAS 1024

#ifdef __cplusplus
extern "C" {
#endif

// Bytes the trace buffer of one call needs.
#define IGNIS_MOE_TRACE_BYTES ((IGNIS_MOE_TRACE_UNITS + IGNIS_MOE_TRACE_CTAS) * IGNIS_MOE_TRACE_WORDS * 8)

int32_t ignis_moe_experts_decode_trace(const void *x, uint32_t tokens, const int32_t *ids, const float *weights,
                                       const struct ignis_moe_slot *slots,
                                       const struct ignis_moe_workspace *workspace, int64_t *acc,
                                       unsigned long long *trace, void *stream);

#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
namespace ignis_moe {
constexpr int kTraceWords = IGNIS_MOE_TRACE_WORDS;
constexpr int kTraceUnits = IGNIS_MOE_TRACE_UNITS;

__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ unsigned int sm_id() {
  unsigned int id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  return id;
}

// One unit's stamps; thread 0 writes the record at the unit's end.
struct UnitStamps {
  unsigned long long begin = 0, ready = 0, mma = 0, end = 0, extra = 0;
};

__device__ __forceinline__ void trace_unit(unsigned long long *trace, int ticket, int kind, int u, int block, int split,
                                           const UnitStamps &st) {
  if (threadIdx.x != 0 || ticket >= kTraceUnits) return;
  unsigned long long *r = trace + static_cast<size_t>(ticket) * kTraceWords;
  r[0] = (static_cast<unsigned long long>(sm_id()) << 32) | blockIdx.x;
  r[1] = static_cast<unsigned long long>(kind) | static_cast<unsigned long long>(u) << 8 |
         static_cast<unsigned long long>(block) << 16 | static_cast<unsigned long long>(split) << 24;
  r[2] = st.begin;
  r[3] = st.ready;
  r[4] = st.mma;
  r[5] = st.end;
  r[6] = st.extra;
  r[7] = 0;
}

__device__ __forceinline__ unsigned long long *trace_cta(unsigned long long *trace) {
  return trace + static_cast<size_t>(kTraceUnits + blockIdx.x) * kTraceWords;
}
}  // namespace ignis_moe
#endif

#endif  // IGNIS_MOE_TRACE_H
