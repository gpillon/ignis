// ignis kernel leaf - Ticket 06 (kernel-abi-02): norms / embedding / greedy
// sampling C ABI surface.
//
// Implements the C ABI functions declared in include/ignis_kernel.h for
// kernel-abi-02: RMSNorm / LayerNorm (ignis_rmsnorm), embedding gather
// (ignis_embedding), and greedy sampling (ignis_greedy_sample). Style
// follows the ticket-03 decode surface (decode_surface.cu) and the
// ticket-05 surface (prefill_gdn_surface.cu): host pointers with internal
// H2D/D2H copies, a stream handle (null = stream 0), and a 0/-1 int
// return code. The device kernels live in the sibling .cuh files
// (provenance in kernel/NOTICE).

#include "ignis_kernel.h"

#include "argmax.cuh"
#include "embed_gather.cuh"
#include "rmsnorm.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace {

int kernel_abi_report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

// Ticket 06 (kernel-abi-02): RMSNorm (center == null) / LayerNorm (center
// != null, "centered first"). x / out: bf16 [n]. weight (nullable): bf16
// [n] (null = unit scale). center (nullable): bf16 [n] (present =>
// LayerNorm mode). eps: the ABI contract says eps <= 0 selects 1e-6.
extern "C" int ignis_rmsnorm(const void* x, const void* weight,
                             const void* center, void* out, std::int64_t n,
                             float eps, void* stream) {
  if (n <= 0 || x == nullptr || out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  // The ABI contract says eps <= 0 selects the default 1e-6.
  const float e = (eps > 0.0f) ? eps : 1e-6f;
  const std::int64_t elems = n;

  __nv_bfloat16* d_x     = nullptr;
  __nv_bfloat16* d_w     = nullptr;
  __nv_bfloat16* d_c     = nullptr;
  __nv_bfloat16* d_out   = nullptr;

  cudaError_t err = cudaMalloc(&d_x, static_cast<size_t>(elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess && weight != nullptr) {
    err = cudaMalloc(&d_w, static_cast<size_t>(elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess && center != nullptr) {
    err = cudaMalloc(&d_c, static_cast<size_t>(elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_out, static_cast<size_t>(elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_x, x, static_cast<size_t>(elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess && weight != nullptr) {
    err = cudaMemcpy(d_w, weight, static_cast<size_t>(elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess && center != nullptr) {
    err = cudaMemcpy(d_c, center, static_cast<size_t>(elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    // One 1024-thread block; the kernel grid-strides over n (see
    // rmsnorm.cuh). weight / center pass through as-is (null = absent).
    const dim3 grid(1);
    const dim3 threads(1024);
    ignis::rmsnorm_kernel<<<grid, threads, 0, s>>>(
        d_x, d_w, d_c, d_out, static_cast<int>(n), e);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_x);
  if (weight != nullptr) cudaFree(d_w);
  if (center != nullptr) cudaFree(d_c);
  cudaFree(d_out);
  return kernel_abi_report(err);
}

// Ticket 06 (kernel-abi-02): dense embedding gather. out[row] =
// table[id[row]]. table: bf16 [vocab][hidden]. id: i32 [batch]. out: bf16
// [batch][hidden]. id values must be in [0, vocab) (the caller's
// contract).
extern "C" int ignis_embedding(const void* table, const void* id, void* out,
                               std::int64_t batch, std::int64_t vocab,
                               std::int64_t hidden, void* stream) {
  if (batch <= 0 || vocab <= 0 || hidden <= 0 ||
      table == nullptr || id == nullptr || out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t table_elems = vocab * hidden;  // [vocab][hidden]
  const std::int64_t id_elems    = batch;           // i32 ids
  const std::int64_t out_elems   = batch * hidden;  // [batch][hidden]

  __nv_bfloat16* d_table = nullptr;
  std::int32_t* d_id     = nullptr;
  __nv_bfloat16* d_out   = nullptr;

  cudaError_t err =
      cudaMalloc(&d_table, static_cast<size_t>(table_elems) * sizeof(__nv_bfloat16));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_id, static_cast<size_t>(id_elems) * sizeof(std::int32_t));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_table, table,
                     static_cast<size_t>(table_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_id, id, static_cast<size_t>(id_elems) * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    // Grid-stride gather (1:1 with the reference's grid_for(n) = max(1,
    // div_up(n, 128)) launcher helper, 128-thread blocks).
    const unsigned grid = static_cast<unsigned>((out_elems + 127) / 128);
    ignis::embed_gather_kernel<<<std::max(1u, grid), 128, 0, s>>>(
        d_id, d_table, d_out, static_cast<int>(hidden), static_cast<int>(batch));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(out_elems) * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_table);
  cudaFree(d_id);
  cudaFree(d_out);
  return kernel_abi_report(err);
}

// Ticket 06 (kernel-abi-02): greedy sampling. out[t] = argmax over
// logits[t]. logits: f32 [batch][vocab]. out: i32 [batch]. Ties resolve
// to the lowest index (the deterministic v1 floor, ADR 0007).
extern "C" int ignis_greedy_sample(const void* logits, void* out,
                                   std::int64_t batch, std::int64_t vocab,
                                   void* stream) {
  if (batch <= 0 || vocab <= 0 || logits == nullptr || out == nullptr) {
    return -1;
  }

  const cudaStream_t s = static_cast<cudaStream_t>(stream);  // null = stream 0
  const std::int64_t logits_elems = batch * vocab;  // f32
  const std::int64_t out_elems    = batch;          // i32

  float*     d_logits = nullptr;
  std::int32_t* d_out = nullptr;

  cudaError_t err =
      cudaMalloc(&d_logits, static_cast<size_t>(logits_elems) * sizeof(float));
  if (err == cudaSuccess) {
    err = cudaMalloc(&d_out, static_cast<size_t>(out_elems) * sizeof(std::int32_t));
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(d_logits, logits,
                     static_cast<size_t>(logits_elems) * sizeof(float),
                     cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    // One 512-thread block per row (the reference's kArgmaxBlock, see
    // argmax.cuh); the block grid-strides over the row's vocab.
    ignis::argmax_kernel<<<static_cast<unsigned>(batch), 512, 0, s>>>(
        d_logits, d_out, static_cast<int>(batch), static_cast<int>(vocab));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(out, d_out, static_cast<size_t>(out_elems) * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost);
  }
  if (err == cudaSuccess) err = cudaStreamSynchronize(s);
  cudaFree(d_logits);
  cudaFree(d_out);
  return kernel_abi_report(err);
}