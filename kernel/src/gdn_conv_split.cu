// ignis kernel leaf: the GDN layer's causal conv, writing q, k and v split
// (kernel/src/gdn_conv_split.h; GitHub #507, spec #503; GitHub #680, spec #672, the decode
// round's form; GitHub #681, spec #672, the gating beside it) -- an ADAPTED op (ADR 0051 tier 2).
//
// Upstream: the reference, gpillon/ninfer, commit a00648cb828457986cf5b4b4f712b4cbcd7af0d1
// (the commit kernel/vendor/manifest.json pins), Apache-2.0 (kernel/vendor/LICENSE):
//   src/ops/kernel/causal_conv1d.cuh    causal_conv1d_prefill_pairs_kernel (one thread per token
//                                       and channel pair, the four taps read from the chunk or
//                                       from the state, the fp32 accumulation in that order, SiLU,
//                                       the paired BF16 rounding),
//                                       causal_conv1d_prefill_state_kernel (the taps after the
//                                       chunk, its old values held in registers before any store)
//                                       and, for the decode round's form,
//                                       causal_conv1d_batched_sequence_snapshot_kernel (one thread
//                                       per channel and lane, the lane's taps read from its slot,
//                                       the fp32 accumulation in that order, SiLU, the BF16
//                                       rounding, the new taps written to the slot);
//   src/ops/launcher/causal_conv1d.cu   causal_conv1d_prefill_launch's grids and block widths,
//                                       and launch_batched_sequence_snapshot's at one column.
//   src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
//                                       for the gating beside the decode round's conv (GitHub
//                                       #681): bf16_gdn_gating_proj_gemv_kernel (a CTA of 256
//                                       threads per a or b row, the pair loop's fp32 fma chain,
//                                       the block reduction, the gate), which the 27B's route
//                                       table runs at one column, and
//                                       bf16_gdn_gating_proj_small_t_partial_kernel (a warp per
//                                       row and 512-wide K slice, the eight-wide fma chains per
//                                       column, the warp reduction) with
//                                       bf16_gdn_gating_proj_small_t_reduce_kernel (the ten
//                                       slices summed in order, the gate), which it runs at 2..8.
// From the vendored reference, called and not copied: `ninfer::ops::silu`, `softplus`, `sigmoid`,
// `bf16x2_to_float2`, `bf16x2_bits_to_float2` and `div_up` (ops/common/math.cuh), `load_vec`
// (ops/common/memory.cuh), `warp_reduce_sum` and `block_reduce_sum` (ops/common/warp.cuh).
//
// Changes from upstream (ours to change, no parity claim):
//   - The output pair lands in its own plane, query, key or value by its channel range, instead
//     of one [C, T] plane that the layer body then split with three strided copies (the copies
//     were a visible share of a long prefill, spec #503).
//   - One form for every T: upstream's wrapper sends T = 1, 2..16 and 17..64 to three other
//     kernels. They compute the same expression, so the planes are the same bits
//     (kernel/tests/test_gdn_conv_split.cu runs every form against this one).
//   - The pair form only, the state in place only: the layer calls nothing else.
//   - The decode round's form (GitHub #680): the snapshot kernel at one column per lane, every
//     lane's initial and base slot the same (the decode graph's in-place form), its output
//     stored into the query, key or value plane by channel range, as above, instead of the
//     [C, B] plane the decode graph split with three strided copies past one column; no valid
//     column mask (the decode round has none). Upstream sends one lane to another kernel
//     (causal_conv1d_snapshot_decode_kernel), which computes the same expression.
//   - The decode round's form stores into planes whose columns may lie apart (GitHub #681), so
//     that it also writes the one [C, B] plane of the decode graph's copying form.
//   - The gating beside it (GitHub #681): upstream's two forms of the 27B's gating, run in CTAs
//     of the conv's grid ahead of the conv's. At one column the CTA is upstream's gemv CTA
//     without its programmatic launch (the conv is a plain launch). At 2..8 one CTA of ten warps
//     takes a row: each warp computes one K slice's partial as upstream's partial kernel's warp
//     does, its columns read straight from the input rather than staged in shared memory, and
//     the CTA sums the ten partials in slice order and takes the gate as upstream's reduce
//     kernel does, instead of a second launch through a workspace. Every sum is in upstream's
//     order, so the gates are its bits (kernel/tests/test_gdn_gating_home.cu).
//
// Its numerical contract is the vendored op's, which is its oracle (ADR 0031): bit equality.

#include "gdn_conv_split.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include "core/tensor.h"

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

// The decode round's conv: the pool, the round's lanes and the three planes it stores into, each
// plane's columns `*_ld` elements apart.
struct SlotConv {
  const __nv_bfloat16 *x;
  const __nv_bfloat16 *weight;
  __nv_bfloat16 *conv_states;
  const std::int32_t *lane_slots;
  __nv_bfloat16 *query;
  __nv_bfloat16 *key;
  __nv_bfloat16 *value;
  std::int32_t C;
  std::int32_t query_width;
  std::int32_t key_width;
  std::int64_t slot_stride;
  std::int64_t query_ld;
  std::int64_t key_ld;
  std::int64_t value_ld;
};

// Upstream's causal_conv1d_batched_sequence_snapshot_kernel at one column, unmasked, every lane's
// initial and base slot its own `lane_slots` entry, its store split by channel range: channel `c64`
// of lane `lane`. The accumulation is upstream's statement for statement: the bits are its bits.
__device__ __forceinline__ void conv_slot_channel(const SlotConv &a, std::int32_t lane, std::int64_t c64) {
  const std::int64_t C64 = static_cast<std::int64_t>(a.C);

  __nv_bfloat16 *slot = a.conv_states + static_cast<std::int64_t>(a.lane_slots[lane]) * a.slot_stride;
  __nv_bfloat16 s0 = slot[c64];
  __nv_bfloat16 s1 = slot[C64 + c64];
  __nv_bfloat16 s2 = slot[2 * C64 + c64];
  const float w0 = __bfloat162float(a.weight[c64]);
  const float w1 = __bfloat162float(a.weight[C64 + c64]);
  const float w2 = __bfloat162float(a.weight[2 * C64 + c64]);
  const float w3 = __bfloat162float(a.weight[3 * C64 + c64]);

  const __nv_bfloat16 x0 = a.x[static_cast<std::int64_t>(lane) * C64 + c64];
  float acc = 0.0f;
  acc += w0 * __bfloat162float(s0);
  acc += w1 * __bfloat162float(s1);
  acc += w2 * __bfloat162float(s2);
  acc += w3 * __bfloat162float(x0);
  const __nv_bfloat16 out = __float2bfloat16_rn(ninfer::ops::silu(acc));
  s0 = s1;
  s1 = s2;
  s2 = x0;
  slot[c64] = s0;
  slot[C64 + c64] = s1;
  slot[2 * C64 + c64] = s2;

  const std::int64_t qw = a.query_width;
  const std::int64_t kw = a.key_width;
  if (c64 < qw) {
    a.query[static_cast<std::int64_t>(lane) * a.query_ld + c64] = out;
  } else if (c64 < qw + kw) {
    a.key[static_cast<std::int64_t>(lane) * a.key_ld + c64 - qw] = out;
  } else {
    a.value[static_cast<std::int64_t>(lane) * a.value_ld + c64 - qw - kw] = out;
  }
}

// One CTA per channel block and lane (blockIdx.y).
__global__ void gdn_split_conv_slots_kernel(SlotConv a) {
  const std::int32_t lane = static_cast<std::int32_t>(blockIdx.y);
  const std::int64_t c64 = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (c64 >= static_cast<std::int64_t>(a.C)) { return; }
  conv_slot_channel(a, lane, c64);
}

// GitHub #681: the 27B's gating, upstream's geometry: 48 heads, a 5120-wide input, the a and b
// rows of one [96, 5120] parent, and the two forms its route table runs at one column and at 2..8.
constexpr int kGatingHeads = 48;
constexpr int kGatingK = 5120;
constexpr int kGatingRows = 2 * kGatingHeads;
constexpr int kGatingMaxColumns = 8;
constexpr int kGatingSliceK = 512;
constexpr int kGatingSlices = kGatingK / kGatingSliceK;
// Upstream's gemv CTA (one column); a warp per K slice (2..8 columns).
constexpr int kOneColumnThreads = 256;
constexpr int kSlicedThreads = kGatingSlices * ninfer::ops::kWarpSize;

struct GatingRows {
  const __nv_bfloat16 *x;
  const __nv_bfloat16 *a_weight;
  const __nv_bfloat16 *b_weight;
  const float *A_log;
  const float *dt_bias;
  float *g;
  float *beta;
  std::int32_t t;
};

// Upstream's bf16_gdn_gating_proj_gemv_kernel for logical row `global_row`, without its
// programmatic launch's wait and publish: the CTA's pair loop, its block reduction, the gate.
__device__ __forceinline__ void gating_row_one_column(const GatingRows &a, int global_row) {
  using namespace ninfer::ops;
  __shared__ float warp_sums[kOneColumnThreads / kWarpSize];
  const bool is_b = global_row >= kGatingHeads;
  const int row = is_b ? global_row - kGatingHeads : global_row;
  const auto *weight = is_b ? a.b_weight : a.a_weight;

  float acc = 0.0f;
  constexpr int kPairs = kGatingK / 2;
  const std::int64_t row_base = static_cast<std::int64_t>(row) * kGatingK;
  const auto *x2 = reinterpret_cast<const __nv_bfloat162 *>(a.x);
  const auto *w2 = reinterpret_cast<const __nv_bfloat162 *>(weight + static_cast<std::int64_t>(row_base));
  for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
    const float2 wf = bf16x2_to_float2(w2[p]);
    const float2 xf = bf16x2_to_float2(x2[p]);
    acc = fmaf(wf.x, xf.x, acc);
    acc = fmaf(wf.y, xf.y, acc);
  }

  acc = block_reduce_sum<kOneColumnThreads>(acc, warp_sums);
  if (threadIdx.x == 0) {
    if (is_b) {
      a.beta[row] = sigmoid(acc);
    } else {
      const float sp = softplus(acc + a.dt_bias[row]);
      a.g[row] = -expf(a.A_log[row]) * sp;
    }
  }
}

// Upstream's split-K form for logical row `logical_row` at `a.t` columns (2..8): warp `split`
// computes what upstream's bf16_gdn_gating_proj_small_t_partial_kernel's warp computes for this
// row and K slice, statement for statement, and the CTA then sums the slices of each column as
// bf16_gdn_gating_proj_small_t_reduce_kernel does, in slice order from zero, and takes the gate.
__device__ __forceinline__ void gating_row_sliced(const GatingRows &a, int logical_row) {
  using namespace ninfer::ops;
  constexpr int kVecsPerCol = kGatingSliceK / 8;
  __shared__ float partial[kGatingSlices][kGatingMaxColumns];

  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int split = static_cast<int>(threadIdx.x) >> 5;
  const int ncols = a.t;
  const int k0 = split * kGatingSliceK;
  const bool is_b = logical_row >= kGatingHeads;
  const int row = is_b ? logical_row - kGatingHeads : logical_row;
  const __nv_bfloat16 *wrow =
      (is_b ? a.b_weight : a.a_weight) + static_cast<std::int64_t>(row) * kGatingK + k0;

  float acc[kGatingMaxColumns];
#pragma unroll
  for (int tt = 0; tt < kGatingMaxColumns; ++tt) { acc[tt] = 0.0f; }

  for (int vec = lane; vec < kVecsPerCol; vec += 32) {
    const uint4 wv = load_vec<uint4>(wrow + vec * 8);
    const float2 wf0 = bf16x2_bits_to_float2(wv.x);
    const float2 wf1 = bf16x2_bits_to_float2(wv.y);
    const float2 wf2 = bf16x2_bits_to_float2(wv.z);
    const float2 wf3 = bf16x2_bits_to_float2(wv.w);

#pragma unroll
    for (int tt = 0; tt < kGatingMaxColumns; ++tt) {
      if (tt < ncols) {
        const uint4 xv = load_vec<uint4>(a.x + static_cast<std::int64_t>(tt) * kGatingK + k0 + vec * 8);
        const float2 xf0 = bf16x2_bits_to_float2(xv.x);
        const float2 xf1 = bf16x2_bits_to_float2(xv.y);
        const float2 xf2 = bf16x2_bits_to_float2(xv.z);
        const float2 xf3 = bf16x2_bits_to_float2(xv.w);
        acc[tt] = fmaf(wf0.x, xf0.x, acc[tt]);
        acc[tt] = fmaf(wf0.y, xf0.y, acc[tt]);
        acc[tt] = fmaf(wf1.x, xf1.x, acc[tt]);
        acc[tt] = fmaf(wf1.y, xf1.y, acc[tt]);
        acc[tt] = fmaf(wf2.x, xf2.x, acc[tt]);
        acc[tt] = fmaf(wf2.y, xf2.y, acc[tt]);
        acc[tt] = fmaf(wf3.x, xf3.x, acc[tt]);
        acc[tt] = fmaf(wf3.y, xf3.y, acc[tt]);
      }
    }
  }

#pragma unroll
  for (int tt = 0; tt < kGatingMaxColumns; ++tt) {
    if (tt < ncols) {
      float sum = warp_reduce_sum(acc[tt]);
      if (lane == 0) { partial[split][tt] = sum; }
    }
  }
  __syncthreads();

  const int token = static_cast<int>(threadIdx.x);
  if (token >= ncols) { return; }
  float total = 0.0f;
#pragma unroll
  for (int s = 0; s < kGatingSlices; ++s) { total += partial[s][token]; }
  const std::int64_t out_index = static_cast<std::int64_t>(token) * kGatingHeads + row;
  if (is_b) {
    a.beta[out_index] = sigmoid(total);
  } else {
    const float sp = softplus(total + a.dt_bias[row]);
    a.g[out_index] = -expf(a.A_log[row]) * sp;
  }
}

// GitHub #681: the gating and the decode round's conv in one grid, the gating's 96 CTAs first (a
// row each: the longer work, so it is scheduled first), then the conv's, `conv_blocks` per lane.
template <bool OneColumn>
__global__ void gdn_split_conv_gating_kernel(SlotConv conv, GatingRows gating, std::int32_t conv_blocks) {
  const auto block = static_cast<std::int32_t>(blockIdx.x);
  if (block < kGatingRows) {
    if constexpr (OneColumn) {
      gating_row_one_column(gating, block);
    } else {
      gating_row_sliced(gating, block);
    }
    return;
  }
  const std::int32_t conv_block = block - kGatingRows;
  const std::int32_t lane = conv_block / conv_blocks;
  const std::int64_t c64 =
      static_cast<std::int64_t>(conv_block - lane * conv_blocks) * blockDim.x + threadIdx.x;
  if (c64 >= static_cast<std::int64_t>(conv.C)) { return; }
  conv_slot_channel(conv, lane, c64);
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

// GitHub #681: a plane of the decode round's form, whose columns may lie apart: each column
// contiguous, `nb[1]` a whole number of elements and at least the column.
void require_bf16_columns(const ninfer::Tensor &t, std::int32_t rows, std::int32_t cols, const char *label) {
  if (t.dtype != ninfer::DType::BF16 || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1) {
    refuse(std::string(label) + " must be BF16 [" + std::to_string(rows) + ", " + std::to_string(cols) + "], is [" +
           std::to_string(t.ne[0]) + ", " + std::to_string(t.ne[1]) + ", " + std::to_string(t.ne[2]) + ", " +
           std::to_string(t.ne[3]) + "]");
  }
  if (t.data == nullptr || t.nb[0] != 2 || t.nb[1] < static_cast<std::int64_t>(rows) * 2 || t.nb[1] % 2 != 0) {
    refuse(std::string(label) + " must be non-null, each column contiguous and the columns whole elements apart");
  }
}

void require_contiguous(const ninfer::Tensor &t, ninfer::DType dtype, std::int32_t rows, std::int32_t cols,
                        const char *label) {
  if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != cols || t.ne[2] != 1 || t.ne[3] != 1 || !t.is_contiguous() ||
      t.data == nullptr) {
    refuse(std::string(label) + " must be contiguous and non-null, [" + std::to_string(rows) + ", " +
           std::to_string(cols) + "]");
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

void split_conv_slots(const ninfer::Tensor &x, const ninfer::Tensor &weight, ninfer::Tensor &conv_states,
                      const ninfer::Tensor &lane_slots, ninfer::Tensor &query, ninfer::Tensor &key,
                      ninfer::Tensor &value, cudaStream_t stream, const Gating *gating) {
  constexpr int kBlock = 256;
  const std::int32_t C = x.ne[0];
  const std::int32_t B = x.ne[1];
  if (C <= 0 || B <= 0) { refuse("x must have positive channels and lanes"); }
  const std::int32_t cq = query.ne[0], ck = key.ne[0], cv = value.ne[0];
  if (cq <= 0 || ck <= 0 || cv <= 0 || static_cast<std::int64_t>(cq) + ck + cv != C) {
    refuse("the query, key and value widths (" + std::to_string(cq) + " + " + std::to_string(ck) + " + " +
           std::to_string(cv) + ") must add up to x's " + std::to_string(C) + " channels");
  }
  if (query.ne[1] != B || key.ne[1] != B || value.ne[1] != B) {
    refuse("the query, key and value planes must have x's lanes (" + std::to_string(B) + ")");
  }
  require_bf16_plane(x, C, B, "x");
  require_bf16_plane(weight, C, 4, "weight");
  require_bf16_columns(query, cq, B, "query");
  require_bf16_columns(key, ck, B, "key");
  require_bf16_columns(value, cv, B, "value");
  if (conv_states.dtype != ninfer::DType::BF16 || conv_states.ne[0] != C || conv_states.ne[1] != 3 ||
      conv_states.ne[2] < B || conv_states.ne[3] != 1 || !conv_states.is_contiguous() || conv_states.data == nullptr) {
    refuse("conv_states must be contiguous BF16 [" + std::to_string(C) + ", 3, slots >= " + std::to_string(B) + "]");
  }
  if (lane_slots.dtype != ninfer::DType::I32 || lane_slots.ne[0] != B || lane_slots.ne[1] != 1 ||
      lane_slots.ne[2] != 1 || lane_slots.ne[3] != 1 || !lane_slots.is_contiguous() || lane_slots.data == nullptr) {
    refuse("lane_slots must be contiguous I32 [" + std::to_string(B) + "]");
  }

  const SlotConv conv{static_cast<const __nv_bfloat16 *>(x.data),
                      static_cast<const __nv_bfloat16 *>(weight.data),
                      static_cast<__nv_bfloat16 *>(conv_states.data),
                      static_cast<const std::int32_t *>(lane_slots.data),
                      static_cast<__nv_bfloat16 *>(query.data),
                      static_cast<__nv_bfloat16 *>(key.data),
                      static_cast<__nv_bfloat16 *>(value.data),
                      C,
                      cq,
                      ck,
                      static_cast<std::int64_t>(C) * 3,
                      query.nb[1] / 2,
                      key.nb[1] / 2,
                      value.nb[1] / 2};
  if (gating == nullptr) {
    const dim3 grid(static_cast<unsigned int>(ninfer::ops::div_up(static_cast<std::int64_t>(C), static_cast<std::int64_t>(kBlock))),
                    static_cast<unsigned int>(B));
    gdn_split_conv_slots_kernel<<<grid, kBlock, 0, stream>>>(conv);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("gdn split conv: slot form launch failed: ") + cudaGetErrorString(err));
    }
    return;
  }

  // GitHub #681: the gating beside the conv, at the 27B's shape alone.
  if (B > kGatingMaxColumns) {
    refuse("the gating takes at most " + std::to_string(kGatingMaxColumns) + " lanes, not " + std::to_string(B));
  }
  const ninfer::Weight &ab = gating->ab;
  if (ab.qtype != ninfer::QType::BF16_CTRL || ab.layout != ninfer::QuantLayout::Contiguous || ab.n != kGatingRows ||
      ab.k != kGatingK || ab.qdata == nullptr || (reinterpret_cast<std::uintptr_t>(ab.qdata) & 15u) != 0) {
    refuse("the gating's a|b parent must be a contiguous, 16-byte aligned BF16 [" + std::to_string(kGatingRows) + ", " +
           std::to_string(kGatingK) + "]");
  }
  require_contiguous(gating->h, ninfer::DType::BF16, kGatingK, B, "the gating's h");
  if ((reinterpret_cast<std::uintptr_t>(gating->h.data) & 15u) != 0) { refuse("the gating's h must be 16-byte aligned"); }
  require_contiguous(gating->a_log, ninfer::DType::FP32, kGatingHeads, 1, "the gating's A_log");
  require_contiguous(gating->dt_bias, ninfer::DType::FP32, kGatingHeads, 1, "the gating's dt_bias");
  require_contiguous(gating->g, ninfer::DType::FP32, kGatingHeads, B, "the gating's g");
  require_contiguous(gating->beta, ninfer::DType::FP32, kGatingHeads, B, "the gating's beta");

  const auto *a_weight = static_cast<const __nv_bfloat16 *>(ab.qdata);
  const GatingRows rows{static_cast<const __nv_bfloat16 *>(gating->h.data),
                        a_weight,
                        a_weight + static_cast<std::int64_t>(kGatingHeads) * kGatingK,
                        static_cast<const float *>(gating->a_log.data),
                        static_cast<const float *>(gating->dt_bias.data),
                        static_cast<float *>(gating->g.data),
                        static_cast<float *>(gating->beta.data),
                        B};
  const int threads = B == 1 ? kOneColumnThreads : kSlicedThreads;
  const auto conv_blocks = static_cast<std::int32_t>(
      ninfer::ops::div_up(static_cast<std::int64_t>(C), static_cast<std::int64_t>(threads)));
  const auto blocks = static_cast<unsigned int>(kGatingRows + conv_blocks * B);
  if (B == 1) {
    gdn_split_conv_gating_kernel<true><<<blocks, threads, 0, stream>>>(conv, rows, conv_blocks);
  } else {
    gdn_split_conv_gating_kernel<false><<<blocks, threads, 0, stream>>>(conv, rows, conv_blocks);
  }
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("gdn split conv: the gating form's launch failed: ") + cudaGetErrorString(err));
  }
}

} // namespace ignis::gdn_conv
