// ignis kernel leaf: what the dense prefill shares with its test and bench -- OURS (GitHub #509,
// kernel/include/ignis_dense_prefill.h): the GEMM's tile plan, the per-device state the plan
// reads, and a launch at a plan the caller names, so the test holds every tile height, depth and
// raster to the integer reference and the bench times each one against the plan's own choice.
// Not part of the flat C ABI.
#ifndef IGNIS_DENSE_PREFILL_INTERNAL_CUH
#define IGNIS_DENSE_PREFILL_INTERNAL_CUH

#include "ignis_dense_prefill.h"

#include <cstdint>

namespace ignis_dense {

// The GEMM's instantiations: tile heights (tokens) and main-loop depths (cp.async stages).
constexpr int kTileHeights[2] = {256, 128};
constexpr int kDepths[2] = {3, 4};

// Token tiles fastest (0) or output tiles fastest (1).
enum Raster : int { kTokensFastest = 0, kOutputsFastest = 1 };

struct Plan {
  int bm = 256;      // 256 or 128
  int stages = 3;    // 3 or 4
  int raster = kTokensFastest;
};

// What ignis_dense_prefill_prepare records for the current device: the SM count and, per
// instantiation [bm == 128][stages == 4], the GEMM's resident CTAs per SM.
struct State {
  int sms = 0;  // 0: not prepared
  int resident[2][2] = {};
};

// The current device's state; -1 (with the error set) when it was not prepared.
int32_t state(const char *op, State *out);

// The tile plan of a call (the rule ignis_dense_prefill_gemm runs):
//   bm      128 when its waves of resident CTAs times its rows, weighed by kTile128RowCostPercent
//           (dense_prefill.cu) for what a row costs in the smaller tile (GitHub #530), stay under
//           256's, else 256 (the larger tile reuses each operand twice as much);
//   stages  4 for a 128-token tile or a K of 8192 or more (the longest main loops), else 3;
//   raster  output tiles fastest when the activations outweigh the matrix (tokens > out), so
//           the larger operand is the one each wave reads once; token tiles fastest otherwise.
Plan plan(const State &st, uint32_t tokens, uint32_t in, uint32_t out);

// The GEMM at `plan` (ignis_dense_prefill_gemm, or with `acc` non-null ignis_dense_prefill_
// accumulate), after the same argument checks.
int32_t gemm_at(const char *op, const Plan &plan, const ignis_trellis_weight *w, const void *dense,
                const ignis_dense_prefill_activations *a, uint32_t tokens, const ignis_dense_prefill_output *out,
                int32_t *acc, uint32_t ldacc, void *stream);

}  // namespace ignis_dense

#endif  // IGNIS_DENSE_PREFILL_INTERNAL_CUH
