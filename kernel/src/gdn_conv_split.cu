// ignis kernel leaf: the GDN prefill's causal conv, writing q, k and v split
// (kernel/src/gdn_conv_split.h; GitHub #507, spec #503) -- an ADAPTED op (ADR 0051 tier 2).
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1
// (the commit kernel/vendor/manifest.json pins), Apache-2.0 (kernel/vendor/LICENSE):
//   src/ops/kernel/causal_conv1d.cuh    causal_conv1d_prefill_pairs_kernel (one thread per token
//                                       and channel pair, the four taps read from the chunk or
//                                       from the state, the fp32 accumulation in that order, SiLU,
//                                       the paired BF16 rounding) and
//                                       causal_conv1d_prefill_state_kernel (the taps after the
//                                       chunk, its old values held in registers before any store);
//   src/ops/launcher/causal_conv1d.cu   causal_conv1d_prefill_launch's grids and block widths.
// From the vendored reference, called and not copied: `ninfer::ops::silu` and `div_up`
// (ops/common/math.cuh).
//
// Changes from upstream (ours to change, no parity claim):
//   - The output pair lands in its own plane, query, key or value by its channel range, instead
//     of one [C, T] plane that the layer body then split with three strided copies (the copies
//     were a visible share of a long prefill, spec #503).
//   - One form for every T: upstream's wrapper sends T = 1, 2..16 and 17..64 to three other
//     kernels. They compute the same expression, so the planes are the same bits
//     (kernel/tests/test_gdn_conv_split.cu runs every form against this one).
//   - The pair form only, the state in place only: the layer calls nothing else.
//
// Its numerical contract is the vendored op's, which is its oracle (ADR 0031): bit equality.

#include "gdn_conv_split.h"

#include "ops/common/math.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kPairBlock = 256;
constexpr int kChannelBlock = 256;

__device__ __forceinline__ void acc_pair(__nv_bfloat162 w, __nv_bfloat162 x, float &acc0, float &acc1) {
  acc0 += __low2float(w) * __low2float(x);
  acc1 += __high2float(w) * __high2float(x);
}

// Upstream's causal_conv1d_prefill_pairs_kernel, its store split by channel range. Every range
// is a whole number of pairs, and with the 27B's widths a whole number of blocks, so a block's
// stores all go to one plane.
__global__ void gdn_split_conv_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                      const __nv_bfloat16 *conv_state, __nv_bfloat16 *query,
                                      __nv_bfloat16 *key, __nv_bfloat16 *value, std::int32_t C,
                                      std::int32_t T, std::int32_t query_pairs, std::int32_t key_pairs) {
  const std::int64_t C2 = static_cast<std::int64_t>(C / 2);
  const std::int64_t c_blocks = ninfer::ops::div_up(C2, static_cast<std::int64_t>(blockDim.x));
  const std::int64_t block = static_cast<std::int64_t>(blockIdx.x);
  const std::int32_t t = static_cast<std::int32_t>(block / c_blocks);
  const std::int64_t pair_base = (block - static_cast<std::int64_t>(t) * c_blocks) * blockDim.x;
  const std::int64_t p = pair_base + threadIdx.x;
  if (t >= T || p >= C2) { return; }

  const auto *x2 = reinterpret_cast<const __nv_bfloat162 *>(x);
  const auto *weight2 = reinterpret_cast<const __nv_bfloat162 *>(weight);
  const auto *state2 = reinterpret_cast<const __nv_bfloat162 *>(conv_state);

  const std::int64_t in_idx = static_cast<std::int64_t>(t) * C2 + p;
  const __nv_bfloat162 x0 = (t >= 3) ? x2[static_cast<std::int64_t>(t - 3) * C2 + p]
                                     : state2[static_cast<std::int64_t>(t) * C2 + p];
  const __nv_bfloat162 x1 = (t >= 2) ? x2[static_cast<std::int64_t>(t - 2) * C2 + p]
                                     : state2[static_cast<std::int64_t>(t + 1) * C2 + p];
  const __nv_bfloat162 x2v = (t >= 1) ? x2[static_cast<std::int64_t>(t - 1) * C2 + p]
                                      : state2[static_cast<std::int64_t>(t + 2) * C2 + p];
  const __nv_bfloat162 x3 = x2[in_idx];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  acc_pair(weight2[p], x0, acc0, acc1);
  acc_pair(weight2[C2 + p], x1, acc0, acc1);
  acc_pair(weight2[2 * C2 + p], x2v, acc0, acc1);
  acc_pair(weight2[3 * C2 + p], x3, acc0, acc1);
  const __nv_bfloat162 out = __floats2bfloat162_rn(ninfer::ops::silu(acc0), ninfer::ops::silu(acc1));

  const std::int64_t qp = query_pairs;
  const std::int64_t kp = key_pairs;
  if (p < qp) {
    reinterpret_cast<__nv_bfloat162 *>(query)[static_cast<std::int64_t>(t) * qp + p] = out;
  } else if (p < qp + kp) {
    reinterpret_cast<__nv_bfloat162 *>(key)[static_cast<std::int64_t>(t) * kp + p - qp] = out;
  } else {
    const std::int64_t vp = C2 - qp - kp;
    reinterpret_cast<__nv_bfloat162 *>(value)[static_cast<std::int64_t>(t) * vp + p - qp - kp] = out;
  }
}

// Upstream's causal_conv1d_prefill_state_kernel, in place: the taps after the chunk are its last
// three columns, or the old taps where the chunk is shorter than three. Each thread loads its
// channel's old taps before it stores any, and the launch follows the conv's on the stream, so
// the conv has read them all.
__global__ void gdn_split_conv_state_kernel(const __nv_bfloat16 *x, __nv_bfloat16 *conv_state, std::int32_t C,
                                            std::int32_t T) {
  const std::int64_t start = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  const std::int64_t C64 = static_cast<std::int64_t>(C);

  for (std::int64_t c64 = start; c64 < C64; c64 += stride) {
    const std::int32_t c = static_cast<std::int32_t>(c64);
    const __nv_bfloat16 old0 = conv_state[c];
    const __nv_bfloat16 old1 = conv_state[C64 + c];
    const __nv_bfloat16 old2 = conv_state[2 * C64 + c];

    for (std::int32_t s = 0; s < 3; ++s) {
      const std::int32_t seq_pos = T + s;
      __nv_bfloat16 v;
      if (seq_pos == 0) {
        v = old0;
      } else if (seq_pos == 1) {
        v = old1;
      } else if (seq_pos == 2) {
        v = old2;
      } else {
        v = x[static_cast<std::int64_t>(seq_pos - 3) * C64 + c];
      }
      conv_state[static_cast<std::int64_t>(s) * C64 + c] = v;
    }
  }
}

[[noreturn]] void refuse(const std::string &why) {
  throw std::invalid_argument("gdn split conv: " + why);
}

void require_bf16_plane(const ninfer::Tensor &t, std::int32_t rows, std::int32_t cols, const char *label) {
  if (t.dtype != ninfer::DType::BF16 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1) {
    refuse(std::string(label) + " must be BF16 [" + std::to_string(rows) + ", " + std::to_string(cols) + "], is [" +
           std::to_string(t.ne[0]) + ", " + std::to_string(t.ne[1]) + ", " + std::to_string(t.ne[2]) + ", " +
           std::to_string(t.ne[3]) + "]");
  }
  if (!t.is_contiguous() || t.data == nullptr) { refuse(std::string(label) + " must be contiguous and non-null"); }
  if ((reinterpret_cast<std::uintptr_t>(t.data) & (alignof(__nv_bfloat162) - 1)) != 0) {
    refuse(std::string(label) + " must be 4-byte aligned (the pair form)");
  }
}

} // namespace

namespace ignis::gdn_conv {

void split_conv(const ninfer::Tensor &x, const ninfer::Tensor &weight, ninfer::Tensor &conv_state,
                ninfer::Tensor &query, ninfer::Tensor &key, ninfer::Tensor &value, cudaStream_t stream) {
  const std::int32_t C = x.ne[0];
  const std::int32_t T = x.ne[1];
  if (C <= 0 || T <= 0) { refuse("x must have positive channels and T"); }
  const std::int32_t cq = query.ne[0], ck = key.ne[0], cv = value.ne[0];
  if (static_cast<std::int64_t>(cq) + ck + cv != C) {
    refuse("the query, key and value widths (" + std::to_string(cq) + " + " + std::to_string(ck) + " + " +
           std::to_string(cv) + ") must add up to x's " + std::to_string(C) + " channels");
  }
  if (((cq | ck | cv) & 1) != 0) { refuse("every width must be even (the pair form)"); }
  if (query.ne[1] != T || key.ne[1] != T || value.ne[1] != T) {
    refuse("the query, key and value planes must have x's T (" + std::to_string(T) + ")");
  }
  require_bf16_plane(x, C, T, "x");
  require_bf16_plane(weight, C, 4, "weight");
  require_bf16_plane(conv_state, C, 3, "conv_state");
  require_bf16_plane(query, cq, T, "query");
  require_bf16_plane(key, ck, T, "key");
  require_bf16_plane(value, cv, T, "value");

  const std::int64_t c_blocks = ninfer::ops::div_up(static_cast<std::int64_t>(C / 2), static_cast<std::int64_t>(kPairBlock));
  const std::int64_t grid = c_blocks * static_cast<std::int64_t>(T);
  if (grid > std::numeric_limits<int>::max()) { refuse("the grid exceeds CUDA's limit"); }
  gdn_split_conv_kernel<<<static_cast<int>(grid), kPairBlock, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data), static_cast<const __nv_bfloat16 *>(weight.data),
      static_cast<const __nv_bfloat16 *>(conv_state.data), static_cast<__nv_bfloat16 *>(query.data),
      static_cast<__nv_bfloat16 *>(key.data), static_cast<__nv_bfloat16 *>(value.data), C, T, cq / 2, ck / 2);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { throw std::runtime_error(std::string("gdn split conv: launch failed: ") + cudaGetErrorString(err)); }

  const int state_grid =
      static_cast<int>(std::max<std::int64_t>(1, ninfer::ops::div_up(static_cast<std::int64_t>(C), static_cast<std::int64_t>(kChannelBlock))));
  gdn_split_conv_state_kernel<<<state_grid, kChannelBlock, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(x.data), static_cast<__nv_bfloat16 *>(conv_state.data), C, T);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("gdn split conv: state launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace ignis::gdn_conv
