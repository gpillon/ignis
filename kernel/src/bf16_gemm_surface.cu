// ignis kernel leaf - Ticket 29 (kernel-abi 10, GitHub #29): bf16 GEMM (the
// logits path for the W8-dequantized lm_head) C ABI surface.
//
// Implements the C ABI function `ignis_bf16_gemm` declared in
// include/ignis_kernel.h: a plain-bf16 rowsplit GEMM (no tensor cores / no
// cuBLASLt, ADR 0001/0005). The W8-dequantized lm_head (A1's artifact
// dequant produces a bf16 weight) cannot go through the NVFP4 GEMM surface
// (kernel-abi 01/05), so the logits path gets its own GEMM. Style follows
// the ticket-03 decode surface (decode_surface.cu) and the kernel-abi-05
// prefill surface (nvfp4_gemm_prefill_surface.cu): host pointers with
// internal H2D/D2H copies, a stream handle (null = stream 0), and a 0/-1 int
// return code. The device kernel lives in the sibling bf16_gemm.cuh.

#include "ignis_kernel.h"

#include "bf16_gemm.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

int bf16_gemm_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

// Ticket 29 (kernel-abi 10, GitHub #29): bf16 GEMM (the logits path for the
// W8-dequantized lm_head).
//   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
// act: bf16 [tokens][k]. wt: bf16 [m][k] (the W8-dequantized lm_head).
// bias (nullable) and out: bf16 [m] / bf16 [tokens][m]. `tokens == 1` is the
// GEMV special case (the decode logits path); `tokens > 1` is the
// batched-prefill logits path (B1). m, k and tokens must be positive (no
// alignment constraint — plain bf16 planes, no NVFP4 group scales). stream:
// null = stream 0. Returns 0 on success, -1 on error.
extern "C" int ignis_bf16_gemm(const void* act, const void* wt, const void* bias,
                               void* out, std::int64_t tokens, std::int64_t m,
                               std::int64_t k, void* stream) {
  // Argument validation BEFORE any CUDA call (the invalid-argument contract:
  // a clean -1, no device work). `bias` is nullable (the lm_head may carry
  // no bias); `act`, `wt`, `out` and the dimensions are not.
  if (tokens <= 0 || m <= 0 || k <= 0 || act == nullptr || wt == nullptr ||
      out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t act_elems = tokens * k;
  const std::int64_t wt_elems = m * k;
  const std::int64_t out_elems = tokens * m;

  __nv_bfloat16* d_act = nullptr;
  __nv_bfloat16* d_wt = nullptr;
  __nv_bfloat16* d_bias = nullptr;
  __nv_bfloat16* d_out = nullptr;

  cudaError_t err = cudaMalloc(&d_act, static_cast<size_t>(act_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_wt, static_cast<size_t>(wt_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_act, act, static_cast<size_t>(act_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_wt, wt, static_cast<size_t>(wt_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess && bias != nullptr) {
    err = cudaMalloc(&d_bias, m * sizeof(__nv_bfloat16));
    if (err == cudaSuccess) {
      err = cudaMemcpy(d_bias, bias, m * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
  }
  if (err == cudaSuccess) {
    // Rowsplit grid: grid.x over m-row tiles, grid.y over token tiles (16x16
    // block; one thread per output element — the proven rowsplit baseline,
    // ADR 0005).
    dim3 grid(static_cast<unsigned>((m + 15) / 16), static_cast<unsigned>((tokens + 15) / 16));
    dim3 block(16, 16);
    ignis::bf16_gemm_kernel<<<grid, block, 0, s>>>(
        d_act, d_wt, d_bias, d_out, static_cast<std::int32_t>(tokens),
        static_cast<std::int32_t>(m), static_cast<std::int32_t>(k));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_act);
  cudaFree(d_wt);
  cudaFree(d_bias);
  cudaFree(d_out);
  return bf16_gemm_report(err);
}