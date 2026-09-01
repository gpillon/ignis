// Ticket 01: kernel leaf smoke test (skeleton).
// Proven kernel ports (GEMM, attention, GDN) arrive with ticket 03+.

#include "ignis_kernel.h"

#include <cstdio>
#include <cstdlib>

extern "C" uint32_t ignis_kernel_hello(void) {
  return 42u;
}

namespace {

constexpr int kThreads = 256;

__global__ void vector_sum_kernel(const float* a, const float* b, float* c, int n) {
  int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) c[i] = a[i] + b[i];
}

int report(cudaError_t err) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "[ignis-kernel] CUDA error: %s\n", cudaGetErrorString(err));
  return -1;
}

}  // namespace

extern "C" int ignis_kernel_vector_sum(const float* a, const float* b, float* c, size_t n) {
  if (n == 0) return 0;
  const size_t bytes = n * sizeof(float);
  float* da = nullptr;
  float* db = nullptr;
  float* dc = nullptr;
  cudaError_t err = cudaMalloc(&da, bytes);
  if (err == cudaSuccess) err = cudaMalloc(&db, bytes);
  if (err == cudaSuccess) err = cudaMalloc(&dc, bytes);
  if (err == cudaSuccess) err = cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    vector_sum_kernel<<<blocks, kThreads>>>(da, db, dc, static_cast<int>(n));
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return report(err);
}