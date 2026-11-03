// Shared helpers for the dense trellis linear's tests and bench -- OURS (spec 27b-16gb/02, GitHub
// #337): device-side counter-hash trellis words and checksums at the 27B's real shapes, the fp64
// reference of a trellis linear on the device, and the tolerance the header states.
//
// The fp64 reference multiplies the rotated activations x' = (x o suh) H, computed on the host in
// fp64, by the inner matrix the kernel decoder reconstructs (held bit for bit to exllamav3 by
// test_trellis_dense_decode), on the device in fp64; the output rotation and svh are applied on the
// host in fp64. It also returns, per output, the scale s of the tolerance: |svh| times the root mean
// square over the output's 128-column block of sqrt(sum_k (x'_k inner_kj)^2).
#ifndef IGNIS_TRELLIS_TEST_COMMON_H
#define IGNIS_TRELLIS_TEST_COMMON_H

#include "ignis_trellis_linear.h"
#include "moe_fixture.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace trellis_test {

using moe_test::check;
using moe_test::DeviceBytes;
using moe_test::download;
using moe_test::upload;

#define TL_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_trellis_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

inline uint32_t tile_words16(uint32_t k2) { return 8 * k2; }  // int16 words per tile

// ---- device: counter-hash words and checksums (record.py's trellis_words / checksum_u16) ------

__device__ inline uint32_t d_lowbias32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

static __global__ void hash_words_kernel(uint16_t *w, uint64_t n, uint32_t stream) {
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    w[i] = static_cast<uint16_t>(d_lowbias32(static_cast<uint32_t>(i) * 0x9E3779B9u + stream * 0x85EBCA6Bu) & 0xFFFFu);
  }
}

static __global__ void checksum_kernel(const uint16_t *v, uint64_t n, unsigned long long *sum) {
  unsigned long long local = 0;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    local += static_cast<unsigned long long>(v[i]) * static_cast<unsigned long long>(d_lowbias32(static_cast<uint32_t>(i)) | 1u);
  }
  atomicAdd(sum, local);  // integer: exact and order-independent mod 2^64
}

inline void device_hash_words(void *w, uint64_t n, uint32_t stream) {
  hash_words_kernel<<<4096, 256>>>(static_cast<uint16_t *>(w), n, stream);
  MOE_CUDA(cudaGetLastError());
}

inline uint64_t device_checksum(const void *v, uint64_t n) {
  DeviceBytes sum(8);
  MOE_CUDA(cudaMemset(sum.p, 0, 8));
  checksum_kernel<<<4096, 256>>>(static_cast<const uint16_t *>(v), n, static_cast<unsigned long long *>(sum.p));
  MOE_CUDA(cudaGetLastError());
  MOE_CUDA(cudaDeviceSynchronize());
  return download<uint64_t>(sum.p, 1)[0];
}

// ---- the fp64 reference ------------------------------------------------------------------------

// y2[t][j] = sum_k xr[t][k] * inner[k][j], s2[t][j] = sum_k (xr[t][k] * inner[k][j])^2, fp64, for
// the `rows` rows of xr (T x in fp64) against inner (in x out fp16). 16 x 16 output tiles.
static __global__ void ref_gemm_kernel(const double *xr, const __half *inner, int rows, int in, int out, double *y2,
                                       double *s2) {
  __shared__ double xs[16][17];
  __shared__ double ws[16][17];
  const int tj = threadIdx.x, ti = threadIdx.y;
  const int j = blockIdx.x * 16 + tj;
  const int t = blockIdx.y * 16 + ti;
  double acc = 0.0, sq = 0.0;
  for (int k0 = 0; k0 < in; k0 += 16) {
    xs[ti][tj] = (blockIdx.y * 16 + ti) < rows ? xr[static_cast<size_t>(blockIdx.y * 16 + ti) * in + k0 + tj] : 0.0;
    ws[ti][tj] = static_cast<double>(__half2float(inner[static_cast<size_t>(k0 + ti) * out + blockIdx.x * 16 + tj]));
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const double p = xs[ti][k] * ws[k][tj];
      acc += p;
      sq += p * p;
    }
    __syncthreads();
  }
  if (t < rows && j < out) {
    y2[static_cast<size_t>(t) * out + j] = acc;
    s2[static_cast<size_t>(t) * out + j] = sq;
  }
}

struct Reference {
  std::vector<double> y;  // [rows][out], the fp64 product
  std::vector<double> s;  // [rows][out], the tolerance's scale
};

// x (BF16 bits [rows][in]) times suh, the 128-wide Hadamard, in fp64.
inline std::vector<double> rotate_in(const std::vector<uint16_t> &x, const std::vector<uint16_t> &suh, int rows, int in) {
  std::vector<double> xr(static_cast<size_t>(rows) * in);
  for (int t = 0; t < rows; ++t) {
    for (int k = 0; k < in; ++k) {
      xr[static_cast<size_t>(t) * in + k] =
          static_cast<double>(moe_test::bf16_to_f32(x[static_cast<size_t>(t) * in + k])) *
          static_cast<double>(moe_test::f16_to_f32(suh[k]));
    }
    moe_test::hadamard128_inplace(xr.data() + static_cast<size_t>(t) * in, in);
  }
  return xr;
}

// The reference of `rows` token rows (BF16 bits of x) through the inner matrix `d_inner` (device,
// fp16 [in][out]) and the scales.
inline Reference reference(const std::vector<uint16_t> &x, int rows, const void *d_inner, int in, int out,
                           const std::vector<uint16_t> &suh, const std::vector<uint16_t> &svh) {
  const std::vector<double> xr = rotate_in(x, suh, rows, in);
  DeviceBytes d_xr(xr.size() * 8), d_y(static_cast<size_t>(rows) * out * 8), d_s(static_cast<size_t>(rows) * out * 8);
  upload(d_xr, xr);
  const dim3 block(16, 16), grid(out / 16, (rows + 15) / 16);
  ref_gemm_kernel<<<grid, block>>>(static_cast<const double *>(d_xr.p), static_cast<const __half *>(d_inner), rows, in,
                                   out, static_cast<double *>(d_y.p), static_cast<double *>(d_s.p));
  MOE_CUDA(cudaGetLastError());
  MOE_CUDA(cudaDeviceSynchronize());
  Reference r;
  r.y = download<double>(d_y.p, static_cast<size_t>(rows) * out);
  const std::vector<double> s2 = download<double>(d_s.p, static_cast<size_t>(rows) * out);
  r.s.resize(r.y.size());
  for (int t = 0; t < rows; ++t) {
    double *y = r.y.data() + static_cast<size_t>(t) * out;
    moe_test::hadamard128_inplace(y, out);
    for (int b = 0; b < out; b += 128) {
      double mean = 0.0;
      for (int j = 0; j < 128; ++j) mean += s2[static_cast<size_t>(t) * out + b + j];
      const double rms = std::sqrt(mean / 128.0);
      for (int j = b; j < b + 128; ++j) {
        const double sv = static_cast<double>(moe_test::f16_to_f32(svh[j]));
        y[j] *= sv;
        r.s[static_cast<size_t>(t) * out + j] = std::fabs(sv) * rms;
      }
    }
  }
  return r;
}

// ---- the tolerance -------------------------------------------------------------------------------

// The header's bound, |y - Y| <= 2^-8 |Y| + 2^-9 s (2^-8 s below compute capability 9.0, as the
// header states), and how much of it an output used: the largest |y - Y| / bound over a call
// (below 1 passes) and the relative RMS error, for the report.
struct Agreement {
  double worst = 0.0;  // max |y - Y| / bound
  double rel_rms = 0.0;
  size_t bad = 0;
  size_t first_bad = 0;
};

// The exponent of the bound's second term on the card the test runs on (device 0, as every
// trellis test uses): -9, or -8 below compute capability 9.0.
inline int accumulation_exponent() {
  static const int exponent = [] {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess) return -9;
    return major < 9 ? -8 : -9;
  }();
  return exponent;
}

inline double bound(double y, double s) { return std::ldexp(std::fabs(y), -8) + std::ldexp(s, accumulation_exponent()); }

inline Agreement agree(const std::vector<double> &got, const std::vector<double> &want, const std::vector<double> &bounds) {
  Agreement a;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(got[i] - want[i]);
    const double b = bounds[i];
    const double ratio = b > 0.0 ? err / b : (err > 0.0 ? std::numeric_limits<double>::infinity() : 0.0);
    if (!(ratio <= 1.0)) {
      if (a.bad == 0) a.first_bad = i;
      ++a.bad;
    }
    if (ratio > a.worst || std::isnan(ratio)) a.worst = std::isnan(ratio) ? std::numeric_limits<double>::infinity() : ratio;
    num += err * err;
    den += want[i] * want[i];
  }
  a.rel_rms = den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
  return a;
}

inline std::vector<double> bf16_to_f64(const std::vector<uint16_t> &v) {
  std::vector<double> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = static_cast<double>(moe_test::bf16_to_f32(v[i]));
  return o;
}

}  // namespace trellis_test

#endif  // IGNIS_TRELLIS_TEST_COMMON_H
