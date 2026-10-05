// ignis kernel leaf: the routed-expert decode's phase trace -- OURS, a diagnostic for the MoE
// microbenchmark (kernel/tests/bench_moe.cu), not part of the ABI in kernel/include/ignis_moe.h.
//
// ignis_moe_experts_decode_trace runs the ticket route's kernel instantiated with global-timer
// stamps: every unit writes one record of kTraceWords u64 at trace[ticket * kTraceWords]:
//   0 CTA (blockIdx.x)   1 SM id   2 kind (0 gate/up, 1 down)
//   3 begin   4 ready (gate/up: activations rotated; down: its first h block in)
//   5 mma done (gate/up: its sums added; down: its last block multiplied)   6 end
//   7 down: nanoseconds spent waiting for h blocks (0 for gate/up)
// Times are %globaltimer nanoseconds, comparable across SMs. `trace` holds at least
// tokens * 10 * (40 + 20) records.
#ifndef IGNIS_MOE_TRACE_H
#define IGNIS_MOE_TRACE_H

#include "ignis_moe.h"

#include <stdint.h>

#define IGNIS_MOE_TRACE_WORDS 8

#ifdef __cplusplus
namespace ignis_moe {
constexpr int kTraceWords = IGNIS_MOE_TRACE_WORDS;
}  // namespace ignis_moe
extern "C" {
#endif

int32_t ignis_moe_experts_decode_trace(const void *x, uint32_t tokens, const int32_t *ids, const float *weights,
                                       const struct ignis_moe_slot *slots,
                                       const struct ignis_moe_workspace *workspace, int64_t *acc,
                                       unsigned long long *trace, void *stream);

#ifdef __cplusplus
}
#endif

#endif  // IGNIS_MOE_TRACE_H
