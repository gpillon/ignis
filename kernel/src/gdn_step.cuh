/* ignis kernel leaf - Ticket 05 (kernel-abi-01): GDN linear-attention step.
 *
 * Provenance: faithful adaptation of the reference Gated DeltaNet (Gated
 * Delta Rule) recurrent transition
 *   F:/ai/q38/ninfer/src/ops/linear_attention/gated_delta_net/recurrent.cuh
 *   (apply_gdn_transition; the fp32-state invariant in core/linear_attention_
 *   state.cpp, the 128-dim state / chunk constants in common.h). The reference
 *   kernel is a warp-tiled FP32-state machine driven by the engine's per-layer
 *   GEMM/Geometry dispatch (separate q/k/v/g/beta tensors, slot-indexed state
 *   pool, a 1/sqrt(128) readout). That dispatch layer is C++ state, which the
 *   flat C ABI of ADR 0001 forbids across the boundary, so it is not ported;
 *   the readout is out of the step's contract (the ABI updates the state, it
 *   does not emit a readout). The core recurrence -- the gated delta rule
 *     S <- alpha*S + beta_p*(v - alpha*S^T k) outer k^T,  alpha = exp(g),
 *     beta_p = sigmoid(beta)  (the (0,1) gate)
 *   -- is ported 1:1; the state is carried as bf16 in/out per the ABI contract
 *   (the full FP32-state pool is a later precision/perf gate) with fp32
 *   internal math.
 *
 * Contract (see kernel/include/ignis_kernel.h, ticket 05):
 *   x         : bf16 [batch][state_dim]. For each batch b, the feature is
 *               decomposed (in order) into the step's inputs:
 *                 k   = x[b][0 .. state_cols)            (the key,  d_k)
 *                 v   = x[b][state_cols .. state_cols+state_rows)  (value, d_v)
 *                 g   = x[b][state_cols + state_rows]    (gate pre-decay, <= 0)
 *                 beta= x[b][state_cols + state_rows + 1](beta pre-activation)
 *               So state_dim = state_cols + state_rows + 2. The same feature is
 *               applied to every GDN layer of the batch (the full model uses
 *               per-layer projections; the flat ABI shares the feature).
 *   state_in  : bf16 [batch][num_gdn_layers][state_rows][state_cols], the
 *               carried recurrent state. state_rows = d_v (rows),
 *               state_cols = d_k (cols); S[dv][d] with dv in [0,d_v), d in
 *               [0,d_k).
 *   state_out : bf16 [batch][num_gdn_layers][state_rows][state_cols], the
 *               updated state (state_in may alias state_out).
 *
 * Update per (batch b, layer l, dv row dv, d_k col d), with alpha = exp(g),
 * beta_p = sigmoid(beta), y[dv] = sum_d S[b][l][dv][d]*k[d] (the per-dv row of
 * S^T k), and delta[dv] = beta_p*(v[dv] - alpha*y[dv]):
 *     S_out[b][l][dv][d] = alpha*S_in[b][l][dv][d] + delta[dv]*k[d]
 *
 * Device-only header. Instantiated by prefill_gdn_surface.cu.
 */
#ifndef IGNIS_GDN_STEP_CUH
#define IGNIS_GDN_STEP_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ignis {

// GDN linear-attention step, batched. One block per (dv row, batch*layer);
// each thread owns one d_k column. A block reduce over the d_k columns yields
// y[dv] = sum_d S[dv][d]*k[d]; the gated delta rule then updates the S row.
// The block covers a single dv row (the delta is shared across the row's d_k
// columns), so one thread per d_k column is enough.
__global__ void gdn_step_kernel(const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ state_in,
                                __nv_bfloat16* __restrict__ state_out, int batch,
                                int num_layers, int state_rows, int state_cols,
                                int state_dim) {
  const int dv = static_cast<int>(blockIdx.x);  // d_v row in [0, state_rows)
  const int bl = static_cast<int>(blockIdx.y);  // b*num_layers + l
  const int d  = static_cast<int>(threadIdx.x);  // d_k column in [0, state_cols)
  if (dv >= state_rows || bl >= batch * num_layers || d >= state_cols) return;

  const int b = bl / num_layers;

  // Decompose x[b] into the step's inputs (k, v, g, beta) -- see the header.
  const __nv_bfloat16* xb = x + static_cast<std::int64_t>(b) * state_dim;
  const float k_d   = __bfloat162float(xb[d]);                 // key column d
  const float v_dv  = __bfloat162float(xb[state_cols + dv]);  // value row dv
  const float g     = __bfloat162float(xb[state_cols + state_rows]);
  const float beta  = __bfloat162float(xb[state_cols + state_rows + 1]);
  const float alpha = expf(g);                 // gate pre-decay -> (0, 1]
  const float beta_p = 1.0f / (1.0f + expf(-beta));  // sigmoid(beta)

  // Carried state element S[b][l][dv][d] (bf16 in, fp32 internal).
  const std::int64_t s_off =
      static_cast<std::int64_t>(bl) * (state_rows * state_cols) +
      static_cast<std::int64_t>(dv) * state_cols + d;
  const float s_in_d = __bfloat162float(state_in[s_off]);

  // y[dv] = sum_d' S[dv][d']*k[d'] -- a block reduce over the d_k columns
  // (one slot per column; slot 0 carries the broadcast sum).
  extern __shared__ float red[];  // [state_cols]
  red[d] = s_in_d * k_d;
  __syncthreads();
  if (d == 0) {
    float y = 0.0f;
    for (int i = 0; i < state_cols; ++i) {
      y += red[i];
    }
    red[0] = y;  // broadcast via slot 0
  }
  __syncthreads();
  const float y      = red[0];
  const float delta = beta_p * (v_dv - alpha * y);

  // Gated delta rule: S_out[dv][d] = alpha*S_in[dv][d] + delta*k[d].
  const float s_out_d = alpha * s_in_d + delta * k_d;
  state_out[s_off] = __float2bfloat16_rn(s_out_d);
}

}  // namespace ignis

#endif  // IGNIS_GDN_STEP_CUH