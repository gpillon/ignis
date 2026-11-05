// ignis kernel leaf: the GDN prefill's chunked delta rule on our own kernels, in the MMA forms a
// GeForce card runs at full rate -- OURS (GitHub #514, spec #503 "The GDN prefill"), leaf-internal.
//
// What it computes is the vendored `ninfer::ops::gated_delta_net` over a prefill chunk with
// `normalize_qk` (the layer body's call, kernel/src/gdn_layer.cu): per value head, from the state S
// (value x key, fp32) the sequence carries,
//
//   q, k <- q / |q|, k / |k|     S_t = a_t S_{t-1} + b_t (v_t - a_t S_{t-1} k_t) k_t^T
//   o_t  = scale S_t q_t         (a_t = exp(g_t), b_t = beta_t)
//
// and the state after the last token. The vendored chunked kernels run it in three passes over
// 64-token chunks (a prepare that builds T^-1, W and U; the state passing, which writes every
// chunk's state to memory; an output pass that reads them back) in TF32 and in bf16 with an fp32
// accumulator, the forms a GeForce card halves or quarters. This op restructures it:
//
//   - A small prepare, parallel over (32-token chunk, q/k head): the norms, K K^T and Q K^T, the
//     decays, and per value head the triangular solve T = (I + A)^-1 on the CUDA cores in fp32
//     (A = strictly lower beta_r (k_r . k_c) exp(G_r - G_c), G the chunk's inclusive cumulative
//     g). It publishes T diag(beta) and the masked decayed Q K^T (P) per chunk and value head.
//   - One recurrence, sequential over the chunks: a warp per 16 value rows of a head holds its
//     slice of S in registers for the whole prompt and per chunk computes
//       R = V - diag(exp(G)) K S^T,  U = T diag(beta) R,
//       O = diag(exp(G)) Q S^T + P U  (the output, written here: no state goes through memory),
//       S <- exp(G_C) S + U^T diag(exp(G_C - G)) K.
//   - Every MMA is f16 x f16 with an f16 accumulator over a span of 32 products, each span folded
//     into fp32 on the CUDA cores (as the FP16 band folds a 32-key tile, hq_prompt_attention.cu);
//     an f16 operand carries a power-of-two row scale that keeps every span below 2^15.
//
// The tail of a chunk that is not a multiple of 32 is masked inside the kernels (beta = 0 and
// zero rows), so no other kernel runs over it. The vendored path stays this op's oracle (ADR
// 0031), the route of a prefill below kMinTokens and the measurement switch's other route:
// kernel/tests/test_gdn_prefill.cu holds the two to each other and to an fp64 recurrence.

#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ignis::gdn_prefill {

// The narrowest prefill the layer body sends here: below it the vendored op runs its recurrent
// kernel alone, on the CUDA cores (it runs no chunked kernel under 64 tokens either).
inline constexpr std::int32_t kMinTokens = 64;

// The route per GPU generation, the one switch: our kernels on the two generations the build's
// cubins load on, Ampere and Ada (compute capability 8.x), which they were written for, and
// Blackwell (12.x), where the RTX 5090 measured them 2.4-2.6x faster than the vendored kernels on
// the 27B's prefill, its time to first token 3-4% shorter at 8K and 64K, and its KLD against BF16
// unmoved within the noise on both 27B bodies (GitHub #514, ADR 0080).
constexpr bool runs_by_default(int compute_capability_major) {
  return compute_capability_major == 8 || compute_capability_major == 12;
}

// The route a load takes (true: ours): `runs_by_default`, unless the process environment's
// IGNIS_GDN_PREFILL names one, `ours` or `vendored` -- a measurement switch (timing and KLD of
// both routes on one card), not an operator setting. Read from the operating system's environment
// at each call (kernel/src/model.cu calls it once per load): on Windows the C runtime's getenv
// holds a copy taken at start, which a later SetEnvironmentVariable (Rust's std::env::set_var)
// does not reach. Throws std::invalid_argument, naming the variable, on any other value.
bool route_for_load(int compute_capability_major);

// The workspace bytes `chunked` takes at `tokens` (any width; 0 for none).
std::size_t workspace_bytes(std::int32_t qk_heads, std::int32_t value_heads, std::int32_t tokens);

// `q`, `k` BF16 [128, qk_heads, T] (not normalized), `v` BF16 [128, value_heads, T], `g` (the log
// decay) and `beta` FP32 [value_heads, T], `scale` the readout's 1/sqrt(128), `state_in` and
// `state_out` FP32 [128 keys, 128 values, value_heads] (the pool slot's layout; they may alias),
// `out` BF16 [128, value_heads, T]. value_heads a multiple of qk_heads, at most 8 per q/k head;
// every tensor contiguous; `workspace` at least `workspace_bytes(..)`, 256-byte aligned. Throws
// std::invalid_argument on anything else; enqueues on `stream` and does not synchronize.
void chunked(const ninfer::Tensor &q, const ninfer::Tensor &k, const ninfer::Tensor &v,
             const ninfer::Tensor &g, const ninfer::Tensor &beta, float scale, const ninfer::Tensor &state_in,
             ninfer::Tensor &state_out, ninfer::Tensor &out, void *workspace, std::size_t workspace_bytes,
             cudaStream_t stream);

// The recurrence's warps per CTA on this device (2, 4 or 8), chosen from its SM count and the
// occupancy query (ADR 0077, #478); for the tests and the report.
std::int32_t recurrence_warps();

} // namespace ignis::gdn_prefill
