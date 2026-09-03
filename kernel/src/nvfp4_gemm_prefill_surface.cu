// ignis kernel leaf - Ticket 22 (kernel-abi 05, GitHub #22): multi-token NVFP4
// GEMM (prefill / FFN-projection path) C ABI surface.
//
// Implements the C ABI function declared in include/ignis_kernel.h for the
// kernel-abi 05 ticket: the multi-token NVFP4 GEMM `ignis_nvfp4_gemm_prefill`.
// Style follows the ticket-01 leaf (hello.cu), the ticket-03 decode surface
// (decode_surface.cu), and the kernel-abi-01 prefill surface
// (prefill_gdn_surface.cu): host pointers with internal H2D/D2H copies, a
// stream handle (null = stream 0), and a 0/-1 int return code. The device
// kernel lives in the sibling .cuh (provenance in kernel/NOTICE).

#include "ignis_kernel.h"

#include "nvfp4_gemm_prefill.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

int gemm_prefill_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

// Ticket 22 (kernel-abi 05, GitHub #22): multi-token NVFP4 GEMM (the prefill /
// FFN-projection path).
//   out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k]
// act: bf16 [tokens][k]. wt_codes: E2M1 [m][k/2] bytes. wt_scales: E4M3
// [m][k/16] bytes. bias (nullable) and out: bf16. k must be a multiple of 16
// (the NVFP4 group scale); m and tokens must be positive. stream: null =
// stream 0. Returns 0 on success, -1 on error.
extern "C" int ignis_nvfp4_gemm_prefill(const void* act, const void* wt_codes,
                                        const void* wt_scales, const void* bias, void* out,
                                        std::int64_t tokens, std::int64_t m, std::int64_t k,
                                        void* stream) {
  // out[tokens][m] = bias[m] + sum_k act[tokens][k] * W[m][k].
  // act: bf16 [tokens][k]. wt_codes: E2M1 [m][k/2] bytes. wt_scales: E4M3
  // [m][k/16] bytes. bias (nullable) and out: bf16. k must be a multiple of 16
  // (the NVFP4 group scale).
  if (tokens <= 0 || m <= 0 || k <= 0 || (k % 16) != 0 ||
      act == nullptr || wt_codes == nullptr || wt_scales == nullptr || out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t act_elems = tokens * k;
  const std::int64_t code_bytes = m * (k / 2);
  const std::int64_t scale_bytes = m * (k / 16);
  const std::int64_t out_elems = tokens * m;

  __nv_bfloat16* d_act = nullptr;
  std::uint8_t* d_codes = nullptr;
  std::uint8_t* d_scales = nullptr;
  __nv_bfloat16* d_bias = nullptr;
  __nv_bfloat16* d_out = nullptr;

  cudaError_t err = cudaMalloc(&d_act, static_cast<size_t>(act_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_codes, static_cast<size_t>(code_bytes));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_scales, static_cast<size_t>(scale_bytes));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_act, act, static_cast<size_t>(act_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_codes, wt_codes, static_cast<size_t>(code_bytes), cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_scales, wt_scales, static_cast<size_t>(scale_bytes),
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
    // block; one thread per output element).
    dim3 grid(static_cast<unsigned>((m + 15) / 16), static_cast<unsigned>((tokens + 15) / 16));
    dim3 block(16, 16);
    ignis::nvfp4_gemm_prefill_kernel<<<grid, block, 0, s>>>(
        d_act, d_codes, d_scales, d_bias, d_out, static_cast<std::int32_t>(tokens),
        static_cast<std::int32_t>(m), static_cast<std::int32_t>(k));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_act);
  cudaFree(d_codes);
  cudaFree(d_scales);
  cudaFree(d_bias);
  cudaFree(d_out);
  return gemm_prefill_report(err);
}