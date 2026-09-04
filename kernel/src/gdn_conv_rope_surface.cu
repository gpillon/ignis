// ignis kernel leaf - Ticket 28 (kernel-abi 06, GitHub #28): GDN causal
// conv + GQA RoPE C ABI surface.
//
// Implements the C ABI functions declared in include/ignis_kernel.h for
// the kernel-abi 06 ticket: the GDN 4-tap depthwise causal conv + SiLU
// (the `gdn/convolution` tensor) and the GQA split-half NeoX RoPE (the Q/K
// projections). Style follows the ticket-05 surface (prefill_gdn_surface.cu)
// and the ticket-06 surface (norms_sampling_surface.cu): host pointers
// with internal H2D/D2H copies, a stream handle (null = stream 0), and a
// 0/-1 int return code. The device kernels live in the sibling .cuh files
// (provenance in kernel/NOTICE).

#include "ignis_kernel.h"

#include "gdn_causal_conv.cuh"
#include "rope_qk.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

int kernel_abi06_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

// Ticket 28 (kernel-abi 06, GitHub #28): GDN 4-tap depthwise causal conv +
// SiLU. `projected`: bf16 [tokens][channels] (the projected q/k/v rows,
// the GEMM output — the `z` rows are NOT part of `channels`, they bypass
// the conv entirely). `conv_weight`: bf16 [4][channels] (the 4 taps
// w0..w3, tap-major — the artifact's `gdn/convolution` tensor).
// `state_in` / `state_out`: bf16 [channels][3] (the rolling 3-tap conv
// state s0,s1,s2 per channel; `state_out` receives the updated state —
// the last 3 consumed taps — and `state_in` may alias `state_out`).
// `out`: bf16 [tokens][channels] (the conv'd + SiLU'd q/k/v).
// stream: null = stream 0. Returns 0 on success, -1 on error.
extern "C" int ignis_gdn_causal_conv(const void* projected, const void* conv_weight,
                                     const void* state_in, void* state_out, void* out,
                                     std::int64_t tokens, std::int64_t channels,
                                     void* stream) {
  if (tokens <= 0 || channels <= 0 || projected == nullptr || conv_weight == nullptr ||
      state_in == nullptr || state_out == nullptr || out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t proj_elems = tokens * channels;
  const std::int64_t wt_elems = 4 * channels;
  const std::int64_t state_elems = channels * 3;

  __nv_bfloat16* d_proj = nullptr;
  __nv_bfloat16* d_wt = nullptr;
  __nv_bfloat16* d_state_in = nullptr;
  __nv_bfloat16* d_state_out = nullptr;
  __nv_bfloat16* d_out = nullptr;

  cudaError_t err = cudaMalloc(&d_proj, static_cast<size_t>(proj_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_wt, static_cast<size_t>(wt_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_state_in, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_state_out, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_out, static_cast<size_t>(proj_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_proj, projected, static_cast<size_t>(proj_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(d_wt, conv_weight, static_cast<size_t>(wt_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(d_state_in, state_in,
                     static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    // One thread per channel (the spec's "one thread per channel"); a
    // 256-thread block covers 256 channels.
    dim3 grid(static_cast<unsigned>((channels + 255) / 256));
    ignis::gdn_causal_conv_kernel<<<grid, 256, 0, s>>>(
        d_proj, d_wt, d_state_in, d_state_out, d_out, static_cast<int>(tokens),
        static_cast<int>(channels));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(out, d_out, static_cast<size_t>(proj_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(state_out, d_state_out,
                     static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_proj);
  cudaFree(d_wt);
  cudaFree(d_state_in);
  cudaFree(d_state_out);
  cudaFree(d_out);
  return kernel_abi06_report(err);
}

// Ticket 28 (kernel-abi 06, GitHub #28): the GQA split-half NeoX RoPE
// (in-place on q/k). `q`: bf16 [batch][seq][num_q_heads][head_dim];
// `k`: bf16 [batch][seq][num_kv_heads][head_dim]; `inv_freq`: fp32
// [rotary_dim/2] (the per-pair frequencies — the reference's
// `rope_linear_frequencies` table, θ = 1e7 in v1; computed once at
// construction, host-side, a deterministic table). For a pair (a = x[p],
// b = x[p + rotary_dim/2]): out[p] = a*cos - b*sin, out[p + rotary_dim/2]
// = b*cos + a*sin (cos/sin = sincosf(pos * inv_freq[p]), the fp32
// unscaled route, the reference's attention_factor 1.0 bit-stable path).
// stream: null = stream 0. Returns 0 on success, -1 on error.
extern "C" int ignis_rope_qk(void* q, void* k, const void* inv_freq, std::int64_t batch,
                             std::int64_t seq, std::int64_t num_q_heads,
                             std::int64_t num_kv_heads, std::int64_t head_dim,
                             std::int64_t rotary_dim, std::int32_t pos, void* stream) {
  if (batch <= 0 || seq <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim || rotary_dim > 256 ||
      q == nullptr || k == nullptr || inv_freq == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t q_elems = batch * seq * num_q_heads * head_dim;
  const std::int64_t k_elems = batch * seq * num_kv_heads * head_dim;
  const std::int64_t freq_elems = rotary_dim / 2;

  __nv_bfloat16* d_q = nullptr;
  __nv_bfloat16* d_k = nullptr;
  float* d_freq = nullptr;

  cudaError_t err = cudaMalloc(&d_q, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess)
    err = cudaMalloc(&d_k, static_cast<size_t>(k_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_freq, static_cast<size_t>(freq_elems) * sizeof(float));
  if (err == cudaSuccess)
    err = cudaMemcpy(d_q, q, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(d_k, k, static_cast<size_t>(k_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess)
    err = cudaMemcpy(d_freq, inv_freq, static_cast<size_t>(freq_elems) * sizeof(float),
                     cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    // One warp per (batch, seq, head) row; a 256-thread block (8 warps)
    // covers 8 rows (grid-strided over the rows).
    const std::int64_t rows = batch * seq * (num_q_heads + num_kv_heads);
    dim3 grid(static_cast<unsigned>((rows + 7) / 8));
    ignis::rope_qk_kernel<<<grid, 256, 0, s>>>(
        d_q, d_k, d_freq, static_cast<int>(batch), static_cast<int>(seq),
        static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), static_cast<int>(rotary_dim), static_cast<int>(pos));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess)
    err = cudaMemcpy(q, d_q, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16),
                      cudaMemcpyDeviceToHost);
  if (err == cudaSuccess)
    err = cudaMemcpy(k, d_k, static_cast<size_t>(k_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_freq);
  return kernel_abi06_report(err);
}