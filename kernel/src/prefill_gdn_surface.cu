// ignis kernel leaf - Ticket 05 (kernel-abi-01): prefill + GDN C ABI surface.
//
// Implements the C ABI functions declared in include/ignis_kernel.h for the
// kernel-abi-01 ticket: GQA prefill attention (batched) + GDN linear-attention
// step (batched). Style follows the ticket-01 leaf (hello.cu) and the
// ticket-03 decode surface (decode_surface.cu): host pointers with internal
// H2D/D2H copies, a stream handle (null = stream 0), and a 0/-1 int return
// code. The device kernels live in the sibling .cuh files (provenance in
// kernel/NOTICE).

#include "ignis_kernel.h"

#include "gdn_step.cuh"
#include "gqa_attention_prefill.cuh"

#include <cstdio>
#include <cstdlib>
#include <math.h>

namespace {

int kernel_abi_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

// Ticket 05 (kernel-abi-01): GQA prefill attention (batched, multi-token).
extern "C" int ignis_gqa_attention_prefill(const void* q, const void* kv_cache,
                                           const void* block_table, void* out,
                                           std::int64_t batch, std::int64_t seq_len,
                                           std::int64_t num_q_heads,
                                           std::int64_t num_kv_heads, std::int64_t head_dim,
                                           std::int64_t block_size, std::int64_t num_blocks,
                                           float softmax_scale, void* stream) {
  // q: bf16 [batch][seq_len][num_q_heads][head_dim]. kv_cache: bf16, two paged
  // planes (K then V), each [batch][num_blocks][num_kv_heads][block_size]
  // [head_dim] (kv_head-major within a page). block_table: i32 [batch][num_blocks]. out: bf16
  // [batch][seq_len][num_q_heads][head_dim]. seq_len must be <= num_blocks *
  // block_size (the paged cache holds the whole prefill). softmax_scale <= 0
  // selects the default 1/sqrt(head_dim).
  if (batch <= 0 || seq_len <= 0 || num_q_heads <= 0 || num_kv_heads <= 0 ||
      head_dim <= 0 || block_size <= 0 || num_blocks <= 0 ||
      (num_q_heads % num_kv_heads) != 0 || seq_len > num_blocks * block_size ||
      q == nullptr || kv_cache == nullptr || block_table == nullptr ||
      out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t plane_elems =
      num_blocks * block_size * num_kv_heads * head_dim;  // per batch
  const std::int64_t q_elems   = batch * seq_len * num_q_heads * head_dim;
  const std::int64_t kv_elems  = 2 * batch * plane_elems;  // K plane + V plane, both batched
  const std::int64_t table_elems = batch * num_blocks;
  // The ABI contract says softmax_scale <= 0 selects the default 1/sqrt(head_dim).
  const float scale =
      (softmax_scale > 0.0f) ? softmax_scale : 1.0f / sqrtf(static_cast<float>(head_dim));

  __nv_bfloat16* d_q      = nullptr;
  __nv_bfloat16* d_kv     = nullptr;
  std::int32_t* d_table   = nullptr;
  __nv_bfloat16* d_out    = nullptr;

  cudaError_t err = cudaMalloc(&d_q, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_kv, static_cast<size_t>(kv_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) err = cudaMalloc(&d_table, table_elems * sizeof(std::int32_t));
  if (err == cudaSuccess) err = cudaMalloc(&d_out, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_q, q, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_kv, kv_cache, static_cast<size_t>(kv_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_table, block_table, table_elems * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    // One block per (q head, position, batch); one thread per head_dim element.
    // Dynamic shared memory: head_dim floats for the per-key block reduce.
    dim3 grid(static_cast<unsigned>(num_q_heads), static_cast<unsigned>(seq_len),
              static_cast<unsigned>(batch));
    const unsigned threads = static_cast<unsigned>(head_dim);
    const unsigned smem    = static_cast<unsigned>(head_dim * sizeof(float));
    ignis::gqa_attention_prefill_kernel<<<grid, threads, smem, s>>>(
        d_kv, d_table, d_q, d_out, static_cast<int>(batch), static_cast<int>(seq_len),
        static_cast<int>(num_q_heads), static_cast<int>(num_kv_heads),
        static_cast<int>(head_dim), static_cast<int>(block_size),
        static_cast<int>(num_blocks), scale);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(q_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_q);
  cudaFree(d_kv);
  cudaFree(d_table);
  cudaFree(d_out);
  return kernel_abi_report(err);
}

// Ticket 05 (kernel-abi-01): GDN (linear-attention) recurrent step, batched.
extern "C" int ignis_gdn_step(const void* x, const void* state_in, void* state_out,
                              std::int64_t batch, std::int64_t num_gdn_layers,
                              std::int64_t state_rows, std::int64_t state_cols,
                              std::int64_t state_dim, void* stream) {
  // x: bf16 [batch][state_dim]. state_in / state_out: bf16
  // [batch][num_gdn_layers][state_rows][state_cols]. state_dim must be
  // state_cols + state_rows + 2 (the k / v / g / beta feature block -- see
  // gdn_step.cuh). state_in may alias state_out.
  if (batch <= 0 || num_gdn_layers <= 0 || state_rows <= 0 || state_cols <= 0 ||
      state_dim != state_cols + state_rows + 2 ||
      x == nullptr || state_in == nullptr || state_out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t x_elems     = batch * state_dim;
  const std::int64_t state_elems = batch * num_gdn_layers * state_rows * state_cols;

  __nv_bfloat16* d_x            = nullptr;
  __nv_bfloat16* d_state_in     = nullptr;
  __nv_bfloat16* d_state_out   = nullptr;

  cudaError_t err = cudaMalloc(&d_x, static_cast<size_t>(x_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_state_in, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_state_out, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_x, x, static_cast<size_t>(x_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_state_in, state_in, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    // One block per (dv row, batch*layer); one thread per d_k column. Dynamic
    // shared memory: state_cols floats for the per-row block reduce.
    dim3 grid(static_cast<unsigned>(state_rows), static_cast<unsigned>(batch * num_gdn_layers));
    const unsigned threads = static_cast<unsigned>(state_cols);
    const unsigned smem    = static_cast<unsigned>(state_cols * sizeof(float));
    ignis::gdn_step_kernel<<<grid, threads, smem, s>>>(
        d_x, d_state_in, d_state_out, static_cast<int>(batch), static_cast<int>(num_gdn_layers),
        static_cast<int>(state_rows), static_cast<int>(state_cols),
        static_cast<int>(state_dim));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(state_out, d_state_out, static_cast<size_t>(state_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_x);
  cudaFree(d_state_in);
  cudaFree(d_state_out);
  return kernel_abi_report(err);
}