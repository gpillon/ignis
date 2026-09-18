// ignis kernel leaf: the DFlash2 drafter's per-column top-k, ours (see
// kernel/include/ignis_dflash2_topk.h for why it exists and what it owes the vendored
// op it replaces).
//
// Two passes over the row axis:
//
//   partial   grid (splits, columns), 128 threads. Block (s, c) selects the
//             top-k of its own contiguous slice of column c and writes those
//             k entries to the workspace. `splits` is chosen so the grid
//             fills the card rather than from the column count.
//   merge     grid (columns), 128 threads. Column c's `splits * k` partial
//             entries are selected down to the final k, which is correct
//             because a global top-k element is the top-k element of the one
//             slice that contains it.
//
// Both passes share the same two primitives, so the selection rule is written
// once:
//
//   insert()            a fully unrolled compare-and-shift into a thread's
//                       private sorted-best-first list. Static indices, so
//                       the list stays in registers; k is a template
//                       parameter for exactly that reason.
//   warp_drain_topk()   k rounds of warp argmax over the lanes' list heads,
//                       the winning lane advancing its cursor -- the vendored
//                       op's own merge, which is where the tie rule and the
//                       largest-first order live.
//
// A block drains its 128 lanes in two warp rounds (each warp to k, then warp
// 0 over the 4 * k survivors) rather than with a block-wide reduction, so one
// __syncthreads() covers the whole block.

#include "ignis_dflash2_topk.h"

#include "ninfer/ops/dflash2_topk.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

// The one shape this engine calls with (kDflash2SelectorTopK,
// kernel/src/model_internal.h). Any other k is the vendored op's.
constexpr std::int32_t kSpecializedK = 16;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
// Rows per block, picked so a 248k-row column spreads over ~120 blocks and
// the grid fills a 170-SM card even at one column. Smaller slices would add
// merge work for no more parallelism.
constexpr std::int64_t kRowsPerSplit = 2048;
constexpr std::int32_t kMaxSplits = 1024;

struct Entry {
  float value;
  int row;
};

// The vendored op's order, verbatim in intent: better means larger value, and
// on a tie the smaller row id. `-0.0f == 0.0f` in IEEE compare, so the two
// zeroes tie and the row decides, which is what the contract says.
__device__ __forceinline__ bool better(const Entry &a, const Entry &b) {
  return a.value > b.value || (a.value == b.value && a.row < b.row);
}

__device__ __forceinline__ Entry sentinel() {
  return Entry{-INFINITY, 0x7fffffff};
}

// Insert `e` into the sorted-best-first list, dropping the worst entry. Fully
// unrolled with static indices so `list` stays in registers: the vendored
// op's data-dependent `while` loop is what forces its list to local memory.
template <int K>
__device__ __forceinline__ void insert(Entry (&list)[K], Entry e) {
#pragma unroll
  for (int i = 0; i < K; ++i) {
    const Entry held = list[i];
    const bool take = better(e, held);
    list[i] = take ? e : held;
    e = take ? held : e;
  }
}

// K rounds of warp argmax over the lanes' list heads. Lane 0 writes the
// column's k winners, best first, to `out` (stride 1). Every lane must call
// this: it is warp-collective, and it consumes `list`.
//
// The head is always `list[0]` and the winning lane shifts its list down by
// one, rather than the vendored op's cursor into the list. That is the whole
// reason this implementation is fast: `list[cursor]` is a dynamic index, and
// a register array indexed dynamically is not a register array -- it is local
// memory, which is what makes the vendored kernel's 20-register frame spill.
// Every index here is a constant under `#pragma unroll`.
template <int K>
__device__ __forceinline__ void warp_drain_topk(Entry (&list)[K], int lane, Entry *out) {
#pragma unroll 1
  for (int slot = 0; slot < K; ++slot) {
    Entry head = list[0];
    for (int offset = 16; offset > 0; offset /= 2) {
      Entry other;
      other.value = __shfl_down_sync(0xffffffffu, head.value, offset);
      other.row = __shfl_down_sync(0xffffffffu, head.row, offset);
      if (better(other, head)) { head = other; }
    }
    head.value = __shfl_sync(0xffffffffu, head.value, 0);
    head.row = __shfl_sync(0xffffffffu, head.row, 0);
    if (lane == 0) { out[slot] = head; }
    // Rows are unique within a column, so exactly one lane holds the winner
    // and exactly one list advances. The sentinel guard matters only for a
    // column with fewer than K rows, where every remaining slot is a
    // sentinel whichever lane yields it.
    if (head.row != 0x7fffffff && list[0].row == head.row) {
#pragma unroll
      for (int i = 0; i < K - 1; ++i) { list[i] = list[i + 1]; }
      list[K - 1] = sentinel();
    }
  }
}

// The block's own k, from the lanes' private lists: each warp to k in shared
// memory, then warp 0 over the 4 * k survivors.
template <int K>
__device__ __forceinline__ void block_drain_topk(Entry (&list)[K], Entry *shared, Entry *out) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  warp_drain_topk<K>(list, lane, shared + warp * K);
  __syncthreads();
  if (warp == 0) {
    Entry merged[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { merged[i] = sentinel(); }
    for (int i = lane; i < kWarps * K; i += 32) { insert<K>(merged, shared[i]); }
    warp_drain_topk<K>(merged, lane, out);
  }
}

template <int K>
__global__ void __launch_bounds__(kThreads) topk_partial_kernel(
    const __nv_bfloat16 *__restrict__ logits, std::int32_t rows, std::int32_t splits,
    Entry *__restrict__ partial) {
  const int split = static_cast<int>(blockIdx.x);
  const int column = static_cast<int>(blockIdx.y);
  // Contiguous slices, balanced to within one row, so the read is coalesced
  // and every row belongs to exactly one block.
  const std::int64_t begin = static_cast<std::int64_t>(rows) * split / splits;
  const std::int64_t end = static_cast<std::int64_t>(rows) * (split + 1) / splits;
  const std::int64_t base = static_cast<std::int64_t>(rows) * column;

  Entry list[K];
#pragma unroll
  for (int i = 0; i < K; ++i) { list[i] = sentinel(); }
  for (std::int64_t row = begin + threadIdx.x; row < end; row += kThreads) {
    insert<K>(list, Entry{__bfloat162float(logits[base + row]), static_cast<int>(row)});
  }

  __shared__ Entry shared[kWarps * K];
  block_drain_topk<K>(list, shared, partial + (static_cast<std::int64_t>(column) * splits + split) * K);
}

template <int K>
__global__ void __launch_bounds__(kThreads) topk_merge_kernel(
    const Entry *__restrict__ partial, const __nv_bfloat16 *__restrict__ logits, std::int32_t rows,
    std::int32_t splits, std::int32_t *__restrict__ ids, __nv_bfloat16 *__restrict__ values) {
  const int column = static_cast<int>(blockIdx.x);
  const Entry *column_partial = partial + static_cast<std::int64_t>(column) * splits * K;
  const int count = splits * K;

  Entry list[K];
#pragma unroll
  for (int i = 0; i < K; ++i) { list[i] = sentinel(); }
  for (int i = threadIdx.x; i < count; i += kThreads) { insert<K>(list, column_partial[i]); }

  __shared__ Entry shared[kWarps * K];
  __shared__ Entry winners[K];
  block_drain_topk<K>(list, shared, winners);
  __syncthreads();

  // `values` carries the logits entry bit-exactly. Re-reading the winning row
  // is what makes that true by construction rather than by trusting a
  // float round trip -- the BF16 the column holds is copied, never rebuilt.
  if (threadIdx.x < K) {
    const Entry winner = winners[threadIdx.x];
    const std::int64_t slot = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(K) * column;
    ids[slot] = winner.row;
    values[slot] = winner.row == 0x7fffffff
        ? __float2bfloat16(winner.value)
        : logits[static_cast<std::int64_t>(rows) * column + winner.row];
  }
}

std::int32_t split_count(std::int32_t rows) {
  const std::int64_t splits = (static_cast<std::int64_t>(rows) + kRowsPerSplit - 1) / kRowsPerSplit;
  return static_cast<std::int32_t>(std::max<std::int64_t>(1, std::min<std::int64_t>(splits, kMaxSplits)));
}

bool specialized(std::int32_t k) {
  return k == kSpecializedK;
}

} // namespace

std::size_t ignis_dflash2_topk_workspace_bytes(std::int32_t rows, std::int32_t columns,
                                               std::int32_t k) {
  if (!specialized(k) || rows <= 0 || columns <= 0) { return 0; }
  return static_cast<std::size_t>(split_count(rows)) * static_cast<std::size_t>(columns) *
         static_cast<std::size_t>(k) * sizeof(Entry);
}

void ignis_dflash2_topk(const ninfer::Tensor &logits, std::int32_t k, ninfer::Tensor &ids,
                        ninfer::Tensor &values, void *workspace, std::size_t workspace_bytes,
                        cudaStream_t stream) {
  const auto rows = static_cast<std::int32_t>(logits.ne[0]);
  const auto columns = static_cast<std::int32_t>(logits.ne[1]);
  const std::size_t needed = ignis_dflash2_topk_workspace_bytes(rows, columns, k);
  if (needed == 0 || workspace == nullptr || workspace_bytes < needed) {
    ninfer::ops::dflash2_topk(logits, k, ids, values, stream);
    return;
  }

  const std::int32_t splits = split_count(rows);
  auto *partial = static_cast<Entry *>(workspace);
  topk_partial_kernel<kSpecializedK>
      <<<dim3(static_cast<unsigned>(splits), static_cast<unsigned>(columns)), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(logits.data), rows, splits, partial);
  topk_merge_kernel<kSpecializedK><<<static_cast<unsigned>(columns), kThreads, 0, stream>>>(
      partial, static_cast<const __nv_bfloat16 *>(logits.data), rows, splits,
      static_cast<std::int32_t *>(ids.data), static_cast<__nv_bfloat16 *>(values.data));
}
