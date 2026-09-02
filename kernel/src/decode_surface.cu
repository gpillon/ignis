// ignis kernel leaf - Ticket 03: flat C ABI decode-step surface (ADR 0001).
//
// Implements the C ABI functions declared in include/ignis_kernel.h for the
// decode step: NVFP4 GEMM (GEMV path) + GQA attention decode. Style follows the
// ticket-01 leaf (hello.cu): host pointers with internal H2D/D2H copies, a
// stream handle (null = stream 0), and a 0/-1 int return code. The device
// kernels live in the sibling .cuh files (provenance in kernel/NOTICE).

#include "ignis_kernel.h"

#include "gqa_attention_decode.cuh"
#include "nvfp4_gemm_decode.cuh"

#include <cstdio>
#include <cstdlib>
#include <math.h>

namespace {

int decode_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

constexpr int kGemmThreads = 256;

}  // namespace

extern "C" int ignis_nvfp4_gemm_decode(const void* act, const void* wt_codes,
                                       const void* wt_scales, const void* bias, void* out,
                                       std::int64_t m, std::int64_t k, void* stream) {
  // NVFP4 decode GEMV: out[m] = bias[m] + sum_k x[k] * W[m,k].
  // act: bf16 [k]. wt_codes: E2M1 [m][k/2] bytes. wt_scales: E4M3 [m][k/16].
  // bias (nullable) and out: bf16 [m].
  if (m <= 0 || k <= 0 || (k % 2) != 0 || (k % 16) != 0) return -1;
  if (out == nullptr) return -1;

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t code_bytes = m * (k / 2);
  const std::int64_t scale_bytes = m * (k / 16);

  __nv_bfloat16* d_act = nullptr;
  std::uint8_t* d_codes = nullptr;
  std::uint8_t* d_scales = nullptr;
  __nv_bfloat16* d_bias = nullptr;
  __nv_bfloat16* d_out = nullptr;

  cudaError_t err = cudaMalloc(&d_out, m * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_act, k * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_codes, static_cast<size_t>(code_bytes));
  if (err == cudaSuccess) err = cudaMalloc(&d_scales, static_cast<size_t>(scale_bytes));
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_act, act, k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
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
    ignis::nvfp4_gemm_decode_kernel<<<static_cast<unsigned>(m), kGemmThreads, 0, s>>>(
        d_act, d_codes, d_scales, d_bias, d_out, static_cast<int>(m), static_cast<int>(k));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, m * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_act);
  cudaFree(d_codes);
  cudaFree(d_scales);
  cudaFree(d_bias);
  cudaFree(d_out);
  return decode_report(err);
}

extern "C" int ignis_gqa_attention_decode(const void* q, const void* kv_cache,
                                          const void* block_table, void* out,
                                          std::int64_t num_q_heads, std::int64_t num_kv_heads,
                                          std::int64_t head_dim, std::int64_t seq_len,
                                          std::int64_t block_size, std::int64_t num_blocks,
                                          float softmax_scale, void* stream) {
  // GQA decode attention, single token. q: bf16 [num_q_heads][head_dim].
  // kv_cache: bf16, two paged planes (K first, V second), each
  // [num_blocks][num_kv_heads][block_size][head_dim] (kv_head-major within a
  // page; head_dim fastest). block_table: i32
  // [num_blocks], logical block -> physical page id. out: bf16
  // [num_q_heads][head_dim].
  if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || block_size <= 0 ||
      num_blocks <= 0 || seq_len <= 0 || (num_q_heads % num_kv_heads) != 0 ||
      seq_len > num_blocks * block_size ||
      out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t q_elems = num_q_heads * head_dim;
  const std::int64_t plane_elems = num_blocks * block_size * num_kv_heads * head_dim;
  const std::int64_t kv_elems = 2 * plane_elems;
  // The ABI contract (see ignis_kernel.h) says softmax_scale <= 0 selects the
  // default 1/sqrt(head_dim).
  const float scale =
      (softmax_scale > 0.0f) ? softmax_scale : 1.0f / sqrtf(static_cast<float>(head_dim));

  __nv_bfloat16* d_q = nullptr;
  __nv_bfloat16* d_kv = nullptr;
  std::int32_t* d_table = nullptr;
  __nv_bfloat16* d_out = nullptr;

  cudaError_t err = cudaMalloc(&d_out, q_elems * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_kv, kv_elems * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_table, num_blocks * sizeof(std::int32_t));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_q, q, q_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_kv, kv_cache, kv_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_table, block_table, num_blocks * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    ignis::gqa_attention_decode_kernel<<<static_cast<unsigned>(num_q_heads),
                                          static_cast<unsigned>(head_dim), 0, s>>>(
        d_kv, d_table, d_q, d_out, static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), static_cast<int>(seq_len), static_cast<int>(block_size),
        static_cast<int>(num_blocks), scale);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_q);
  cudaFree(d_kv);
  cudaFree(d_table);
  cudaFree(d_out);
  return decode_report(err);
}