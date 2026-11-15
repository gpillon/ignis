// ignis kernel leaf: the 27B MTP head's W8G32 matrices on an RTX 30 or RTX 40 card -- OURS
// (GitHub #627, spec #618; kernel/src/w8_imma.h), written from the PTX ISA and our own kernels
// (kernel/src/dense_prefill.cu's integer MMA). The narrow calls run the vendored W8 SIMT route,
// called and not copied (kernel/vendor/src/ops/linear/w8/w8_rowsplit_gemm_simt.cu).
//
// Two kernels, each its own launch on the caller's stream:
//   w8_imma_split_kernel    the activation's planes: a thread per 8 inputs of a token, the four
//                           threads of a group reduce its amax with two shuffles;
//   w8_imma_gemm_kernel<NI> 16 output rows x 8 NI tokens a CTA, 8 warps over the K groups
//                           (warp w takes groups w, w + 8, ...), two m16n8k32 s8 MMAs per group
//                           and token tile, the fp32 accumulators summed over the warps through
//                           shared memory in a fixed order, then BF16 out.
// Neither has a tensor-core form but the s8 x s8 -> s32 MMA: the full-rate rule's (spec #503),
// which mk/cuda-build-check.sh holds them to on the 86 cubin.
//
// The fragments read the K of a group in an order of their own. A group's integer sum does not
// depend on the order of its 32 products, so the MMA's k index 4 c .. 4 c + 3 (c = lane % 4) is
// the group's bytes 8 c .. 8 c + 3 and its k index 16 + 4 c .. the bytes 8 c + 4 .. 8 c + 7, in
// the matrix's rows and in the activation's tokens alike: each lane reads one 8-byte run of a row
// or a token per group, and a warp's load is 8 whole 32-byte sectors.

#include "w8_imma.h"

#include "ops/linear/w8/w8_launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ignis::w8_imma {
namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRows = 16;   // a CTA's output rows: one m16 tile
constexpr int kGroup = 32;  // W8G32's group, one k32 MMA
constexpr int kSplitThreads = 256;
constexpr std::size_t kAlign = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

void require(bool ok, const std::string &what) {
  if (!ok) {
    throw std::invalid_argument("w8_imma: " + what);
  }
}

void check_launch(const char *what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("w8_imma: ") + what + ": " + cudaGetErrorString(err));
  }
}

// D = A B: A 16 x 32 s8 (row), B 32 x 8 s8 (col), int32, exact.
__device__ __forceinline__ void mma_s8(int32_t (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// An integer below 2^22 in magnitude as fp32, exactly: the bits of 1.5 2^23 + v, less 1.5 2^23,
// two full-rate CUDA-core operations where the conversion instruction runs at a quarter of them.
__device__ __forceinline__ float small_int_to_float(int32_t v) {
  return __int_as_float(v + 0x4B400000) - 12582912.0f;
}

__device__ __forceinline__ uint32_t pack4(const int (&v)[8], int first) {
  return (static_cast<uint32_t>(v[first] & 0xFF)) | (static_cast<uint32_t>(v[first + 1] & 0xFF) << 8) |
         (static_cast<uint32_t>(v[first + 2] & 0xFF) << 16) | (static_cast<uint32_t>(v[first + 3] & 0xFF) << 24);
}

// Token blockIdx.y, inputs 8 i .. 8 i + 7 (i the thread's index across the CTAs of the token). k is
// a multiple of 256, so a warp is wholly inside the token's inputs or past them, and the group's
// four threads are four lanes of one warp.
__global__ void __launch_bounds__(kSplitThreads) w8_imma_split_kernel(const __nv_bfloat16 *__restrict__ x,
                                                                      std::int32_t k, std::int32_t tokens,
                                                                      std::int8_t *__restrict__ hi,
                                                                      std::int8_t *__restrict__ lo,
                                                                      float *__restrict__ scale) {
  const std::int32_t t = static_cast<std::int32_t>(blockIdx.y);
  const std::int32_t k0 = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (k0 >= k) return;
  const std::size_t at = static_cast<std::size_t>(t) * static_cast<std::size_t>(k) + static_cast<std::size_t>(k0);
  const uint4 raw = *reinterpret_cast<const uint4 *>(x + at);
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
  float v[8];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __uint_as_float(words[i] << 16);
    v[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000u);
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
  const bool live = amax > 0.0f && isfinite(amax);
  const float to_q = live ? static_cast<float>(kSplitRange) / amax : 0.0f;
  int h[8], l[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int q = min(max(__float2int_rn(v[i] * to_q), -kSplitRange), kSplitRange);
    h[i] = (q + 128) >> 8;
    l[i] = q - 256 * h[i];
  }
  *reinterpret_cast<uint2 *>(hi + at) = make_uint2(pack4(h, 0), pack4(h, 4));
  *reinterpret_cast<uint2 *>(lo + at) = make_uint2(pack4(l, 0), pack4(l, 4));
  if ((threadIdx.x & 3) == 0) {
    scale[static_cast<std::size_t>(k0 / kGroup) * static_cast<std::size_t>(tokens) + static_cast<std::size_t>(t)] =
        live ? amax / static_cast<float>(kSplitRange) : 0.0f;
  }
}

// Rows 16 blockIdx.y .. + 15, tokens 8 NI blockIdx.x .. + 8 NI - 1 (token tiles fastest, so the
// tiles of a row block read its rows from L2). Lane (g, c) = (lane / 4, lane % 4) holds the A
// fragment of rows g and g + 8, the B fragment of token 8 j + g of tile j, and the accumulators of
// rows g, g + 8 and tokens 8 j + 2 c, + 1 (the m16n8 layout). Tokens past `tokens` read zeros and
// are not written.
template <int NI>
__global__ void __launch_bounds__(kThreads) w8_imma_gemm_kernel(
    const std::uint8_t *__restrict__ codes, const __half *__restrict__ wscale, std::int32_t padded_k,
    const std::int8_t *__restrict__ hi, const std::int8_t *__restrict__ lo, const float *__restrict__ ascale,
    std::int32_t k, std::int32_t n, std::int32_t tokens, __nv_bfloat16 *__restrict__ out) {
  __shared__ float partial[kWarps][NI * 4][32];
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int g8 = lane >> 2;
  const int c = lane & 3;
  const std::int32_t t0 = static_cast<std::int32_t>(blockIdx.x) * 8 * NI;
  const std::int32_t r0 = static_cast<std::int32_t>(blockIdx.y) * kRows;
  const std::int32_t groups = k / kGroup;
  const std::int32_t scale_stride = padded_k / kGroup;
  const std::uint8_t *row_a = codes + static_cast<std::size_t>(r0 + g8) * static_cast<std::size_t>(padded_k) + 8 * c;
  const std::uint8_t *row_b = row_a + 8 * static_cast<std::size_t>(padded_k);
  const __half *scale_a = wscale + static_cast<std::size_t>(r0 + g8) * static_cast<std::size_t>(scale_stride);
  const __half *scale_b = scale_a + 8 * static_cast<std::size_t>(scale_stride);

  float acc[NI][4];
#pragma unroll
  for (int j = 0; j < NI; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
  }
#pragma unroll 2
  for (std::int32_t g = warp; g < groups; g += kWarps) {
    const uint2 wa = __ldcs(reinterpret_cast<const uint2 *>(row_a + static_cast<std::size_t>(g) * kGroup));
    const uint2 wb = __ldcs(reinterpret_cast<const uint2 *>(row_b + static_cast<std::size_t>(g) * kGroup));
    const uint32_t a[4] = {wa.x, wb.x, wa.y, wb.y};
    const float sa = __half2float(__ldg(scale_a + g));
    const float sb = __half2float(__ldg(scale_b + g));
    const float *ag = ascale + static_cast<std::size_t>(g) * static_cast<std::size_t>(tokens);
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const std::int32_t tb = t0 + 8 * j + g8;
      uint2 bh = make_uint2(0u, 0u);
      uint2 bl = make_uint2(0u, 0u);
      if (tb < tokens) {
        const std::size_t at = static_cast<std::size_t>(tb) * static_cast<std::size_t>(k) +
                               static_cast<std::size_t>(g) * kGroup + 8 * c;
        bh = __ldg(reinterpret_cast<const uint2 *>(hi + at));
        bl = __ldg(reinterpret_cast<const uint2 *>(lo + at));
      }
      int32_t sh[4] = {0, 0, 0, 0};
      int32_t sl[4] = {0, 0, 0, 0};
      mma_s8(sh, a, bh.x, bh.y);
      mma_s8(sl, a, bl.x, bl.y);
      const std::int32_t tc = t0 + 8 * j + 2 * c;
      const float s0 = tc < tokens ? __ldg(ag + tc) : 0.0f;
      const float s1 = tc + 1 < tokens ? __ldg(ag + tc + 1) : 0.0f;
      const float scales[4] = {sa * s0, sa * s1, sb * s0, sb * s1};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float sum = fmaf(small_int_to_float(sh[i]), 256.0f, small_int_to_float(sl[i]));
        acc[j][i] = fmaf(sum, scales[i], acc[j][i]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NI; ++j) {
#pragma unroll
    for (int i = 0; i < 4; ++i) partial[warp][j * 4 + i][lane] = acc[j][i];
  }
  __syncthreads();
  for (int e = static_cast<int>(threadIdx.x); e < NI * 4 * 32; e += kThreads) {
    const int slot = e >> 5;
    const int from = e & 31;
    float sum = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) sum += partial[w][slot][from];
    const int j = slot >> 2;
    const int i = slot & 3;
    const std::int32_t row = r0 + (from >> 2) + 8 * (i >> 1);
    const std::int32_t token = t0 + 8 * j + 2 * (from & 3) + (i & 1);
    if (token < tokens) {
      out[static_cast<std::size_t>(token) * static_cast<std::size_t>(n) + static_cast<std::size_t>(row)] =
          __float2bfloat16_rn(sum);
    }
  }
}

template <int NI>
void launch_gemm(const ninfer::Weight &w, const detail::Planes &p, std::int32_t tokens, ninfer::Tensor &out,
                 cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>((tokens + 8 * NI - 1) / (8 * NI)), static_cast<unsigned>(w.n / kRows));
  w8_imma_gemm_kernel<NI><<<grid, kThreads, 0, stream>>>(
      static_cast<const std::uint8_t *>(w.qdata), static_cast<const __half *>(w.scales), w.padded_shape[1], p.hi, p.lo,
      p.scale, w.k, w.n, tokens, static_cast<__nv_bfloat16 *>(out.data));
  check_launch("the integer MMA");
}

bool aligned_to(const void *p, std::uintptr_t alignment) {
  return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) & (alignment - 1)) == 0;
}

void require_call(const ninfer::Tensor &x, const ninfer::Weight &w, const ninfer::Tensor &out) {
  require(serves(w), "the matrix is not W8G32 row-split with rows a multiple of 16 and inputs of 256");
  require(x.dtype == ninfer::DType::BF16 && x.ne[0] == w.k && x.ne[1] > 0 && x.ne[2] == 1 && x.ne[3] == 1 &&
              x.is_contiguous() && aligned_to(x.data, 16),
          "x is not a contiguous, 16-byte aligned BF16 [" + std::to_string(w.k) + ", T]");
  require(out.dtype == ninfer::DType::BF16 && out.ne[0] == w.n && out.ne[1] == x.ne[1] && out.ne[2] == 1 &&
              out.ne[3] == 1 && out.is_contiguous() && aligned_to(out.data, 16),
          "out is not a contiguous, 16-byte aligned BF16 [" + std::to_string(w.n) + ", T]");
}

}  // namespace

bool route_for_load(int compute_capability_major) {
  return takes_ours(route_switch("IGNIS_MTP_HEAD_LINEAR"), compute_capability_major);
}

bool serves(const ninfer::Weight &w) {
  return w.qtype == ninfer::QType::W8G32_F16S && w.layout == ninfer::QuantLayout::RowSplit && w.qdata != nullptr &&
         w.scales != nullptr && w.n > 0 && w.n % kRows == 0 && w.k > 0 && w.k % (kWarps * kGroup) == 0 &&
         w.padded_shape[1] >= w.k && w.padded_shape[1] % kGroup == 0;
}

std::size_t split_bytes(std::int32_t k, std::int32_t tokens) {
  const std::size_t plane = static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
  return 2 * aligned(plane) + aligned(static_cast<std::size_t>(k / kGroup) * static_cast<std::size_t>(tokens) * 4);
}

namespace detail {

Planes planes(void *room, std::int32_t k, std::int32_t tokens) {
  auto *base = static_cast<unsigned char *>(room);
  const std::size_t plane = aligned(static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens));
  return Planes{reinterpret_cast<std::int8_t *>(base), reinterpret_cast<std::int8_t *>(base + plane),
                reinterpret_cast<float *>(base + 2 * plane)};
}

void split(const ninfer::Tensor &x, void *room, std::size_t room_bytes, cudaStream_t stream) {
  const std::int32_t k = x.ne[0];
  const std::int32_t tokens = x.ne[1];
  require(x.dtype == ninfer::DType::BF16 && k > 0 && k % (kWarps * kGroup) == 0 && tokens > 0 && x.ne[2] == 1 &&
              x.ne[3] == 1 && x.is_contiguous() && aligned_to(x.data, 16),
          "the activation is not a contiguous, 16-byte aligned BF16 [k, T] with k a multiple of 256");
  require(tokens <= 65535, "the activation has more than 65535 tokens");
  require(aligned_to(room, kAlign) && room_bytes >= split_bytes(k, tokens),
          "the room for the activation's planes is not 256-byte aligned or holds fewer than " +
              std::to_string(split_bytes(k, tokens)) + " bytes");
  const Planes p = planes(room, k, tokens);
  const dim3 grid(static_cast<unsigned>((k / 8 + kSplitThreads - 1) / kSplitThreads), static_cast<unsigned>(tokens));
  w8_imma_split_kernel<<<grid, kSplitThreads, 0, stream>>>(static_cast<const __nv_bfloat16 *>(x.data), k, tokens,
                                                           p.hi, p.lo, p.scale);
  check_launch("the activation's split");
}

void imma(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out, void *room, std::size_t room_bytes,
          cudaStream_t stream) {
  require_call(x, w, out);
  split(x, room, room_bytes, stream);
  const Planes p = planes(room, w.k, x.ne[1]);
  const std::int32_t tokens = x.ne[1];
  if (tokens <= 8) {
    launch_gemm<1>(w, p, tokens, out, stream);
  } else if (tokens <= 16) {
    launch_gemm<2>(w, p, tokens, out, stream);
  } else if (tokens <= 32) {
    launch_gemm<4>(w, p, tokens, out, stream);
  } else {
    launch_gemm<8>(w, p, tokens, out, stream);
  }
}

}  // namespace detail

void linear(const ninfer::Tensor &x, const ninfer::Weight &w, ninfer::Tensor &out, void *room, std::size_t room_bytes,
            cudaStream_t stream) {
  require_call(x, w, out);
  const std::int32_t tokens = x.ne[1];
  if (tokens <= kSimtMaxTokens) {
    // The vendored dispatch's own split between its two SIMT column tiles.
    if (tokens <= 4) {
      ninfer::ops::detail::launch_w8_simt_r8_c4(x, w, out, stream);
    } else {
      ninfer::ops::detail::launch_w8_simt_r8_c8(x, w, out, stream);
    }
    return;
  }
  detail::imma(x, w, out, room, room_bytes, stream);
}

}  // namespace ignis::w8_imma
