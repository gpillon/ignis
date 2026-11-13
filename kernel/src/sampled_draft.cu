// The sampled draft's kernels (GitHub #576, spec #574, ADR 0090): the draft's
// draw and the verify round's accept for a load under the sampled draft
// (kernel/src/sampled_draft.h), and the test-only entries over explicit
// uniforms (kernel/include/ignis_sampled_draft.h). The arithmetic is
// kernel/src/sampled_draft.cuh's. GitHub #577 adds the teacher-forced round's
// mode of the draw and the accept (test-only, null in every serving round)
// and its verdict's sweep.
//
// The accept builds each column's target distribution with the vendored
// sampler's device primitives (ops/kernel/sampling_device.cuh) on its
// multi-block route, the route the vendored accept takes for the 27B's
// vocabulary: per column a partial top-k over 512-row tiles, then a group
// merge whose last block sorts the column's candidates and normalizes them,
// as the sampler's own `sampling_normalize_support` does. Only the verdict
// differs, so a column's p is the vendored accept's to the bit.

#include "sampled_draft.cuh"
#include "sampled_draft.h"

#include "ignis_sampled_draft.h"

#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using ninfer::ops::SamplingConfig;
using ninfer::ops::SamplingWorkspace;
using ninfer::ops::div_up;
using ninfer::ops::kSamplerBlock;
using ninfer::ops::kSamplerCandidateCap;
using ninfer::ops::kSamplerGroupBlock;
using ninfer::ops::kSamplerGroupItemsPerThread;
using ninfer::ops::kSamplerItemsPerThread;
using ninfer::ops::kSamplerPartialTileItems;
using ninfer::ops::kSamplerPartialsPerGroup;

constexpr int kTopK = kIgnisSampledDraftCandidates;

void check_launch(const char *what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " launch failed: " + cudaGetErrorString(err));
  }
}

__device__ inline SamplingWorkspace workspace_row(SamplingWorkspace workspace, std::size_t row_stride,
                                                  std::int32_t row) {
  const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
  const auto at = [offset](auto *ptr) {
    using T = std::remove_pointer_t<decltype(ptr)>;
    return ptr == nullptr ? nullptr
                          : reinterpret_cast<T *>(reinterpret_cast<unsigned char *>(ptr) + offset);
  };
  workspace.partial_keys = at(workspace.partial_keys);
  workspace.dist_idx = at(workspace.dist_idx);
  workspace.dist_prob = at(workspace.dist_prob);
  workspace.dist_support = at(workspace.dist_support);
  workspace.group_done = at(workspace.group_done);
  workspace.speculative_finalize_count = at(workspace.speculative_finalize_count);
  return workspace;
}

__device__ inline int clamped_extent(const std::int32_t *current_extents, int row, int k) {
  const int extent = current_extents[row];
  return extent < 0 ? 0 : (extent > k ? k : extent);
}

// One thread per lane: the draws of a sampling lane, column by column, each
// from q given the draft drawn before it. GitHub #577 (test-only): with
// `forced` (`[k + 1, B]`, a teacher-forced round's recorded tokens) a column
// whose recorded token is one of its candidates drafts that token instead of
// the draw, and the next column's q follows it: the forced round drafts along
// the recording, keeping the q a live round that drew it would have read.
__global__ void sampled_draft_draw_kernel(const float *scores, const std::int32_t *candidates,
                                          const SamplingConfig *configs,
                                          const std::int32_t *base_positions, std::int32_t k,
                                          std::int32_t lanes, float temperature_factor, float top_p,
                                          const std::int32_t *forced, std::int32_t *drafts,
                                          std::int32_t *q_ids, float *q_probs) {
  const int lane = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (lane >= lanes) {
    return;
  }
  const SamplingConfig cfg = configs[lane];
  if (!(cfg.temperature > 0.0f)) {
    return;
  }
  const float temperature = cfg.temperature * temperature_factor;
  const int base = base_positions[lane];
  int pred = 0;
  for (int s = 0; s < k; ++s) {
    const std::int64_t slot = static_cast<std::int64_t>(lane) * k + s;
    float q[kTopK];
    ignis_drafter_distribution(scores + kTopK * (pred + kTopK * slot), kTopK, temperature, top_p, q);
    const float u = ninfer::ops::sampling_uniform(cfg.seed, base + s + 1, kIgnisSamplePurposeDraft, 0u);
    int picked = ignis_draw_draft(q, kTopK, u);
    for (int i = 0; i < kTopK; ++i) {
      q_ids[kTopK * slot + i] = candidates[kTopK * slot + i];
      q_probs[kTopK * slot + i] = q[i];
    }
    if (forced != nullptr) {
      const std::int32_t recorded = forced[static_cast<std::int64_t>(lane) * (k + 1) + s];
      for (int i = 0; i < kTopK; ++i) {
        if (candidates[kTopK * slot + i] == recorded) {
          picked = i;
          break;
        }
      }
    }
    drafts[slot] = candidates[kTopK * slot + picked];
    pred = picked;
  }
}

// Column `col` of a sampling lane: the adjusted logits of one 512-row tile,
// sorted, the column's top `cap` of it kept (the vendored partial top-k's
// step, on the vendored primitives).
__launch_bounds__(kSamplerBlock) __global__ void sampled_target_partial_kernel(
    const __nv_bfloat16 *logits, const std::int32_t *drafts, const std::int32_t *current_extents,
    const SamplingConfig *configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace, std::size_t row_stride) {
  const int row = static_cast<int>(blockIdx.z);
  const int col = static_cast<int>(blockIdx.y);
  const int partial = static_cast<int>(blockIdx.x);
  if (col > clamped_extent(current_extents, row, k)) {
    return;
  }
  const SamplingConfig cfg = configs[row];
  if (!(cfg.temperature > 0.0f)) {
    return;
  }
  workspace = workspace_row(workspace, row_stride, row);
  if (partial == 0 && threadIdx.x == 0) {
    workspace.group_done[col] = 0;
  }

  __shared__ typename ninfer::ops::SamplingPartialSort::TempStorage sort_storage;
  unsigned long long keys[kSamplerItemsPerThread];
  const int cap = ninfer::ops::sampling_candidate_cap(cfg, token_domain);
  const std::int64_t base = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
  const std::int32_t *row_drafts = drafts + row * k;
  const int tile_start = partial * kSamplerPartialTileItems;
  // Column col is reached only when the col drafts before it were taken, so
  // they are its round-local penalty overlay.
#pragma unroll
  for (int item = 0; item < kSamplerItemsPerThread; ++item) {
    const int v = tile_start + item * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (v < token_domain) {
      const float x = ninfer::ops::sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg,
                                                           row_drafts, col);
      keys[item] = ninfer::ops::sampling_sort_key(x, v);
    } else {
      keys[item] = 0ull;
    }
  }
  ninfer::ops::SamplingPartialSort(sort_storage).Sort(keys, ninfer::ops::SamplingKeyGreater{});
#pragma unroll
  for (int item = 0; item < kSamplerItemsPerThread; ++item) {
    const int rank = static_cast<int>(threadIdx.x) * kSamplerItemsPerThread + item;
    if (rank < cap) {
      workspace.partial_keys[ninfer::ops::sampling_partial_offset(workspace, col, partial, rank)] = keys[item];
    }
  }
}

// One group of a column's partials merged; the column's last group to finish
// merges the groups and writes the column's target distribution -- support
// ids in the sampler's order and their renormalized probabilities -- into
// the workspace for the accept.
__launch_bounds__(kSamplerGroupBlock) __global__ void sampled_target_finalize_kernel(
    const std::int32_t *current_extents, const SamplingConfig *configs, std::int32_t token_domain,
    std::int32_t cols, std::int32_t partial_blocks, std::int32_t group_count,
    SamplingWorkspace workspace, std::size_t row_stride) {
  const int row = static_cast<int>(blockIdx.z);
  const int group = static_cast<int>(blockIdx.x);
  const int col = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (col > clamped_extent(current_extents, row, cols - 1)) {
    return;
  }
  const SamplingConfig cfg = configs[row];
  if (!(cfg.temperature > 0.0f)) {
    return;
  }
  workspace = workspace_row(workspace, row_stride, row);

  __shared__ typename ninfer::ops::SamplingGroupSort::TempStorage sort_storage;
  __shared__ float cand_val[kSamplerCandidateCap];
  __shared__ int cand_idx[kSamplerCandidateCap];
  __shared__ float prob[kSamplerCandidateCap];
  __shared__ int n_support;
  __shared__ int is_last_group;
  unsigned long long keys[kSamplerGroupItemsPerThread];
  const int cap = ninfer::ops::sampling_candidate_cap(cfg, token_domain);

  const int group_begin = group * kSamplerPartialsPerGroup;
  int group_partials = partial_blocks - group_begin;
  group_partials = group_partials < 0 ? 0 : (group_partials > kSamplerPartialsPerGroup ? kSamplerPartialsPerGroup : group_partials);
  const int group_n = group_partials * cap;
#pragma unroll
  for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
    const int p = item * static_cast<int>(blockDim.x) + tid;
    keys[item] = p < group_n ? workspace.partial_keys[ninfer::ops::sampling_partial_offset(
                                   workspace, col, group_begin + p / cap, p % cap)]
                             : 0ull;
  }
  ninfer::ops::SamplingGroupSort(sort_storage).Sort(keys, ninfer::ops::SamplingKeyGreater{});
#pragma unroll
  for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
    const int rank = tid * kSamplerGroupItemsPerThread + item;
    if (rank < cap) {
      workspace.partial_keys[ninfer::ops::sampling_partial_offset(workspace, col, partial_blocks + group,
                                                                  rank)] = keys[item];
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
    is_last_group = done == group_count ? 1 : 0;
  }
  __syncthreads();
  if (!is_last_group) {
    return;
  }
  __threadfence();

  const int final_n = group_count * cap;
#pragma unroll
  for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
    const int p = item * static_cast<int>(blockDim.x) + tid;
    keys[item] = p < final_n ? workspace.partial_keys[ninfer::ops::sampling_partial_offset(
                                   workspace, col, partial_blocks + p / cap, p % cap)]
                             : 0ull;
  }
  __syncthreads();
  ninfer::ops::SamplingGroupSort(sort_storage).Sort(keys, ninfer::ops::SamplingKeyGreater{});
#pragma unroll
  for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
    const int rank = tid * kSamplerGroupItemsPerThread + item;
    if (rank < cap) {
      cand_val[rank] = ninfer::ops::sampling_key_float(keys[item]);
      cand_idx[rank] = ninfer::ops::sampling_key_index(keys[item]);
    }
  }
  __syncthreads();
  ninfer::ops::sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
  if (tid == 0) {
    workspace.dist_support[col] = n_support;
    for (int j = 0; j < n_support; ++j) {
      workspace.dist_idx[ninfer::ops::sampling_dist_offset(col, j)] = cand_idx[j];
      workspace.dist_prob[ninfer::ops::sampling_dist_offset(col, j)] = prob[j];
    }
    workspace.group_done[col] = 0;
  }
}

// One thread per lane: the verdicts of the round, then its outputs, as the
// vendored accept writes them. GitHub #577 (test-only): with `forced`
// (`[k + 1, B]`, a teacher-forced round's recorded tokens, drafted along the
// recording by the draw) a sampling lane takes each recorded token by the
// forced verdict (`ignis_forced_sampled_verdict`) and emits the recorded token
// where it stops, so its run is the recording's; a token outside the
// vocabulary (the pad after a stop) is never counted in the penalty row.
__global__ void sampled_accept_kernel(const std::int32_t *target_tokens, const std::int32_t *drafts,
                                      const std::int32_t *current_extents, std::int32_t *lengths,
                                      std::int32_t *anchors, std::int32_t *licensed_tokens,
                                      std::int32_t *licensed_counts, std::int32_t *accepted,
                                      const SamplingConfig *configs, const std::int32_t *q_ids,
                                      const float *q_probs, const std::int32_t *forced,
                                      std::int32_t token_domain, std::int32_t k, std::int32_t batch,
                                      SamplingWorkspace workspace, std::size_t row_stride) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= batch) {
    return;
  }
  const int cols = k + 1;
  const int extent = clamped_extent(current_extents, row, k);
  const SamplingConfig cfg = configs[row];
  const std::int32_t *row_drafts = drafts + row * k;
  const std::int32_t *row_forced = forced == nullptr ? nullptr : forced + static_cast<std::int64_t>(row) * cols;
  std::int32_t *row_tokens = licensed_tokens + row * cols;
  int a = 0;
  int tstar = 0;
  int length = lengths[row];
  if (!(cfg.temperature > 0.0f)) {
    // A greedy lane: today's rule, the vendored greedy branch's -- the
    // longest prefix the target argmax agrees with, then the argmax.
    const std::int32_t *row_targets = target_tokens + row * cols;
    while (a < extent && row_targets[a] == row_drafts[a]) {
      ++a;
    }
    tstar = row_targets[a];
  } else {
    workspace = workspace_row(workspace, row_stride, row);
    for (int i = 0; i <= extent; ++i) {
      const int n = workspace.dist_support[i];
      const int *support = workspace.dist_idx + ninfer::ops::sampling_dist_offset(i, 0);
      const float *probs = workspace.dist_prob + ninfer::ops::sampling_dist_offset(i, 0);
      if (i == extent) {
        if (row_forced != nullptr) {
          // Every recorded token taken: the recorded one after them.
          tstar = row_forced[extent];
          break;
        }
        // Every draft taken: the bonus token from the last column, the
        // vendored accept's draw.
        const float u = ninfer::ops::sampling_uniform(cfg.seed, length + extent + 1,
                                                      ninfer::ops::kSamplePurposeSpeculativeBonus, 0u);
        tstar = ninfer::ops::sampling_pick_from_support(support, probs, n, -1, u);
        break;
      }
      const std::int32_t draft = row_drafts[i];
      const float one = 1.0f;
      const std::int32_t *drafter_ids = &draft;
      const float *drafter_probs = &one;
      int n_drafter = 1;
      if (q_ids != nullptr) {
        const std::int64_t slot = static_cast<std::int64_t>(row) * k + i;
        drafter_ids = q_ids + kTopK * slot;
        drafter_probs = q_probs + kTopK * slot;
        n_drafter = kTopK;
      }
      const float u_accept =
          ninfer::ops::sampling_uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeSampledAccept, 0u);
      const float u_refusal =
          ninfer::ops::sampling_uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeSampledRefusal, 0u);
      if (row_forced != nullptr) {
        // The draw drafted the recorded token wherever it is a candidate; the
        // live draft there is the one the lane's own stream draws from the q
        // it kept.
        const std::int32_t recorded = row_forced[i];
        bool taken = false;
        if (draft == recorded) {
          const float u_draft =
              ninfer::ops::sampling_uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeDraft, 0u);
          const std::int32_t live_draft =
              q_ids == nullptr ? draft : drafter_ids[ignis_draw_draft(drafter_probs, n_drafter, u_draft)];
          std::int32_t live_token = 0;
          const bool live_taken = ignis_sampled_verdict(support, probs, n, drafter_ids, drafter_probs, n_drafter,
                                                        live_draft, u_accept, u_refusal, &live_token);
          const float u_coin =
              ninfer::ops::sampling_uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeForcedCoin, 0u);
          taken = ignis_forced_sampled_verdict(support, probs, n, drafter_ids, drafter_probs, n_drafter, recorded,
                                               live_draft, live_taken, live_token, u_coin);
        }
        if (taken) {
          a = i + 1;
          continue;
        }
        tstar = recorded;
        break;
      }
      std::int32_t token = 0;
      if (ignis_sampled_verdict(support, probs, n, drafter_ids, drafter_probs, n_drafter, draft, u_accept,
                                u_refusal, &token)) {
        a = i + 1;
        continue;
      }
      tstar = token;
      break;
    }
  }
  for (int i = 0; i <= k; ++i) {
    row_tokens[i] = 0;
  }
  for (int i = 0; i < a; ++i) {
    row_tokens[i] = row_drafts[i];
  }
  row_tokens[a] = tstar;
  const int produced = a + 1;
  licensed_counts[row] = produced;
  accepted[row] = a;
  anchors[row] = tstar;
  lengths[row] = length + produced;
  if (cfg.temperature > 0.0f && cfg.token_counts != nullptr) {
    for (int i = 0; i < produced; ++i) {
      if (row_tokens[i] >= 0 && row_tokens[i] < token_domain) {
        atomicAdd(&cfg.token_counts[row_tokens[i]], 1);
      }
    }
  }
}

// --- the test-only sweeps ---------------------------------------------------

__global__ void verdict_sweep_kernel(const std::int32_t *target_ids, const float *target_probs, int n_target,
                                     const std::int32_t *drafter_ids, const float *drafter_probs, int n_drafter,
                                     std::int32_t draft, const float *grid, std::uint32_t cells,
                                     std::int32_t *out) {
  const std::uint64_t point = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (point >= static_cast<std::uint64_t>(cells) * cells) {
    return;
  }
  std::int32_t token = 0;
  const bool taken = ignis_sampled_verdict(target_ids, target_probs, n_target, drafter_ids, drafter_probs,
                                           n_drafter, draft, grid[point / cells], grid[point % cells], &token);
  out[point] = taken ? IGNIS_SAMPLED_DRAFT_ACCEPTED : token;
}

__global__ void forced_sweep_kernel(const std::int32_t *target_ids, const float *target_probs, int n_target,
                                    const std::int32_t *drafter_ids, const float *drafter_probs, int n_drafter,
                                    std::int32_t recorded, std::int32_t live_draft, bool live_taken,
                                    std::int32_t live_token, const float *grid, std::uint32_t cells,
                                    std::uint8_t *out) {
  const std::uint32_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point < cells) {
    out[point] = ignis_forced_sampled_verdict(target_ids, target_probs, n_target, drafter_ids, drafter_probs,
                                              n_drafter, recorded, live_draft, live_taken, live_token, grid[point])
                     ? 1
                     : 0;
  }
}

__global__ void draw_sweep_kernel(const float *q, int n, const float *grid, std::uint32_t cells,
                                  std::int32_t *out) {
  const std::uint32_t point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point < cells) {
    out[point] = ignis_draw_draft(q, n, grid[point]);
  }
}

__global__ void drafter_distribution_kernel(const float *scores, int n, float temperature, float top_p,
                                            float *q) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ignis_drafter_distribution(scores, n, temperature, top_p, q);
  }
}

thread_local std::string g_last_error;

int32_t fail(const std::string &message) {
  g_last_error = message;
  return -1;
}

// A device copy of `count` host elements.
template <typename T>
ninfer::DeviceBuffer to_device(const T *host, std::size_t count) {
  ninfer::DeviceBuffer buffer(std::max<std::size_t>(count * sizeof(T), 1));
  buffer.copy_from_host(host, count * sizeof(T));
  return buffer;
}

void synchronize(const char *what) {
  check_launch(what);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
  }
}

} // namespace

void ignis_sampled_draft_draw(const ninfer::Tensor &scores, const ninfer::Tensor &candidates,
                              const ninfer::ops::SamplingConfig *configs, const std::int32_t *base_positions,
                              float temperature_factor, float top_p, ninfer::Tensor &drafts,
                              std::int32_t *q_ids, float *q_probs, cudaStream_t stream,
                              const std::int32_t *forced) {
  const std::int32_t k = drafts.ne[0];
  const std::int32_t lanes = drafts.ne[1];
  if (scores.ne[0] != kTopK || scores.ne[1] != kTopK || scores.ne[2] != k || scores.ne[3] != lanes ||
      candidates.ne[0] != kTopK || candidates.ne[1] != k || candidates.ne[2] != lanes) {
    throw std::invalid_argument("ignis_sampled_draft_draw: the selector's lattice is not [16, 16, k, B]");
  }
  if (configs == nullptr || base_positions == nullptr || q_ids == nullptr || q_probs == nullptr) {
    throw std::invalid_argument("ignis_sampled_draft_draw: null argument");
  }
  constexpr int kBlock = 32;
  sampled_draft_draw_kernel<<<div_up(lanes, kBlock), kBlock, 0, stream>>>(
      static_cast<const float *>(scores.data), static_cast<const std::int32_t *>(candidates.data), configs,
      base_positions, k, lanes, temperature_factor, top_p, forced, static_cast<std::int32_t *>(drafts.data),
      q_ids, q_probs);
  check_launch("ignis_sampled_draft_draw");
}

bool ignis_sampled_accept_serves(std::int32_t token_domain, std::int32_t k) {
  return ninfer::ops::make_sampling_workspace_layout(token_domain, k + 1).multiblock;
}

std::size_t ignis_sampled_accept_workspace_bytes(std::int32_t token_domain, std::int32_t k,
                                                 std::int32_t batch) {
  return ninfer::ops::make_sampling_workspace_layout(token_domain, k + 1).bytes *
         static_cast<std::size_t>(batch);
}

void ignis_sampled_accept(const ninfer::Tensor &target_tokens, const ninfer::Tensor &logits,
                          const ninfer::Tensor &drafts, const ninfer::Tensor &current_extents,
                          ninfer::Tensor &lengths, ninfer::Tensor &anchors, ninfer::Tensor &licensed_tokens,
                          ninfer::Tensor &licensed_counts, ninfer::Tensor &accepted, std::int32_t token_domain,
                          const ninfer::ops::SamplingConfig *configs, const std::int32_t *q_ids,
                          const float *q_probs, ninfer::DeviceArena &workspace, cudaStream_t stream,
                          const std::int32_t *forced) {
  const std::int32_t k = drafts.ne[0];
  const std::int32_t batch = drafts.ne[1];
  const std::int32_t cols = k + 1;
  const std::int32_t physical_rows = logits.ne[0];
  if (k < 1 || batch < 1 || logits.ne[1] != cols || logits.ne[2] != batch || target_tokens.ne[0] != cols ||
      target_tokens.ne[1] != batch || licensed_tokens.ne[0] != cols || licensed_tokens.ne[1] != batch) {
    throw std::invalid_argument("ignis_sampled_accept: the round's tensors are not [k+1, B]");
  }
  if (token_domain <= 0 || token_domain > physical_rows || configs == nullptr ||
      (q_ids == nullptr) != (q_probs == nullptr)) {
    throw std::invalid_argument("ignis_sampled_accept: invalid token domain, configs or q");
  }
  const ninfer::ops::SamplingWorkspaceLayout layout = ninfer::ops::make_sampling_workspace_layout(token_domain, cols);
  if (!layout.multiblock) {
    throw std::invalid_argument("ignis_sampled_accept: a vocabulary of " + std::to_string(token_domain) +
                                " does not take the sampler's multi-block route");
  }
  ninfer::DeviceArena::Scope scope = workspace.scope();
  const ninfer::DeviceSpan span = workspace.alloc_bytes(layout.bytes * static_cast<std::size_t>(batch));
  const SamplingWorkspace scratch = layout.bind(span);
  const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
  const std::int32_t groups = ninfer::ops::sampler_group_count(partial_blocks);
  const auto *extents = static_cast<const std::int32_t *>(current_extents.data);
  const dim3 partial_grid(static_cast<unsigned int>(partial_blocks), static_cast<unsigned int>(cols),
                          static_cast<unsigned int>(batch));
  sampled_target_partial_kernel<<<partial_grid, kSamplerBlock, 0, stream>>>(
      static_cast<const __nv_bfloat16 *>(logits.data), static_cast<const std::int32_t *>(drafts.data), extents,
      configs, token_domain, physical_rows, cols, k, scratch, layout.bytes);
  check_launch("ignis_sampled_accept: the partial top-k");
  const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(cols),
                        static_cast<unsigned int>(batch));
  sampled_target_finalize_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
      extents, configs, token_domain, cols, partial_blocks, groups, scratch, layout.bytes);
  check_launch("ignis_sampled_accept: the group merge");
  constexpr int kBlock = 32;
  sampled_accept_kernel<<<div_up(batch, kBlock), kBlock, 0, stream>>>(
      static_cast<const std::int32_t *>(target_tokens.data), static_cast<const std::int32_t *>(drafts.data),
      extents, static_cast<std::int32_t *>(lengths.data), static_cast<std::int32_t *>(anchors.data),
      static_cast<std::int32_t *>(licensed_tokens.data), static_cast<std::int32_t *>(licensed_counts.data),
      static_cast<std::int32_t *>(accepted.data), configs, q_ids, q_probs, forced, token_domain, k, batch, scratch,
      layout.bytes);
  check_launch("ignis_sampled_accept: the verdicts");
}

extern "C" int32_t ignis_sampled_draft_verdicts(const int32_t *target_ids, const float *target_probs,
                                                uint32_t target_count, const int32_t *drafter_ids,
                                                const float *drafter_probs, uint32_t drafter_count,
                                                int32_t draft, const float *grid, uint32_t cells,
                                                int32_t *out_tokens) {
  if (target_ids == nullptr || target_probs == nullptr || drafter_ids == nullptr || drafter_probs == nullptr ||
      grid == nullptr || out_tokens == nullptr) {
    return fail("ignis_sampled_draft_verdicts: null argument");
  }
  if (target_count == 0 || drafter_count == 0 || drafter_count > IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES ||
      cells == 0) {
    return fail("ignis_sampled_draft_verdicts: an empty distribution or grid, or a drafter past " +
                std::to_string(IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES) + " candidates");
  }
  try {
    const std::size_t points = static_cast<std::size_t>(cells) * cells;
    ninfer::DeviceBuffer d_target_ids = to_device(target_ids, target_count);
    ninfer::DeviceBuffer d_target_probs = to_device(target_probs, target_count);
    ninfer::DeviceBuffer d_drafter_ids = to_device(drafter_ids, drafter_count);
    ninfer::DeviceBuffer d_drafter_probs = to_device(drafter_probs, drafter_count);
    ninfer::DeviceBuffer d_grid = to_device(grid, cells);
    ninfer::DeviceBuffer d_out(points * sizeof(std::int32_t));
    constexpr int kBlock = 256;
    verdict_sweep_kernel<<<static_cast<unsigned int>((points + kBlock - 1) / kBlock), kBlock>>>(
        static_cast<const std::int32_t *>(d_target_ids.p), static_cast<const float *>(d_target_probs.p),
        static_cast<int>(target_count), static_cast<const std::int32_t *>(d_drafter_ids.p),
        static_cast<const float *>(d_drafter_probs.p), static_cast<int>(drafter_count), draft,
        static_cast<const float *>(d_grid.p), cells, static_cast<std::int32_t *>(d_out.p));
    synchronize("ignis_sampled_draft_verdicts");
    d_out.copy_to_host(out_tokens, points * sizeof(std::int32_t));
    return 0;
  } catch (const std::exception &e) {
    return fail(std::string("ignis_sampled_draft_verdicts: ") + e.what());
  }
}

extern "C" int32_t ignis_sampled_draft_forced_verdicts(const int32_t *target_ids, const float *target_probs,
                                                       uint32_t target_count, const int32_t *drafter_ids,
                                                       const float *drafter_probs, uint32_t drafter_count,
                                                       int32_t recorded, int32_t live_draft, int32_t live_taken,
                                                       int32_t live_token, const float *grid, uint32_t cells,
                                                       uint8_t *out_taken) {
  if (target_ids == nullptr || target_probs == nullptr || drafter_ids == nullptr || drafter_probs == nullptr ||
      grid == nullptr || out_taken == nullptr) {
    return fail("ignis_sampled_draft_forced_verdicts: null argument");
  }
  if (target_count == 0 || drafter_count == 0 || drafter_count > IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES ||
      cells == 0) {
    return fail("ignis_sampled_draft_forced_verdicts: an empty distribution or grid, or a drafter past " +
                std::to_string(IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES) + " candidates");
  }
  try {
    ninfer::DeviceBuffer d_target_ids = to_device(target_ids, target_count);
    ninfer::DeviceBuffer d_target_probs = to_device(target_probs, target_count);
    ninfer::DeviceBuffer d_drafter_ids = to_device(drafter_ids, drafter_count);
    ninfer::DeviceBuffer d_drafter_probs = to_device(drafter_probs, drafter_count);
    ninfer::DeviceBuffer d_grid = to_device(grid, cells);
    ninfer::DeviceBuffer d_out(static_cast<std::size_t>(cells));
    constexpr int kBlock = 256;
    forced_sweep_kernel<<<div_up(static_cast<int>(cells), kBlock), kBlock>>>(
        static_cast<const std::int32_t *>(d_target_ids.p), static_cast<const float *>(d_target_probs.p),
        static_cast<int>(target_count), static_cast<const std::int32_t *>(d_drafter_ids.p),
        static_cast<const float *>(d_drafter_probs.p), static_cast<int>(drafter_count), recorded, live_draft,
        live_taken != 0, live_token, static_cast<const float *>(d_grid.p), cells,
        static_cast<std::uint8_t *>(d_out.p));
    synchronize("ignis_sampled_draft_forced_verdicts");
    d_out.copy_to_host(out_taken, static_cast<std::size_t>(cells));
    return 0;
  } catch (const std::exception &e) {
    return fail(std::string("ignis_sampled_draft_forced_verdicts: ") + e.what());
  }
}

extern "C" int32_t ignis_sampled_draft_draws(const float *q, uint32_t count, const float *grid, uint32_t cells,
                                             int32_t *out_candidates) {
  if (q == nullptr || grid == nullptr || out_candidates == nullptr) {
    return fail("ignis_sampled_draft_draws: null argument");
  }
  if (count == 0 || count > IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES || cells == 0) {
    return fail("ignis_sampled_draft_draws: an empty distribution or grid, or past " +
                std::to_string(IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES) + " candidates");
  }
  try {
    ninfer::DeviceBuffer d_q = to_device(q, count);
    ninfer::DeviceBuffer d_grid = to_device(grid, cells);
    ninfer::DeviceBuffer d_out(static_cast<std::size_t>(cells) * sizeof(std::int32_t));
    constexpr int kBlock = 256;
    draw_sweep_kernel<<<div_up(static_cast<int>(cells), kBlock), kBlock>>>(
        static_cast<const float *>(d_q.p), static_cast<int>(count), static_cast<const float *>(d_grid.p), cells,
        static_cast<std::int32_t *>(d_out.p));
    synchronize("ignis_sampled_draft_draws");
    d_out.copy_to_host(out_candidates, static_cast<std::size_t>(cells) * sizeof(std::int32_t));
    return 0;
  } catch (const std::exception &e) {
    return fail(std::string("ignis_sampled_draft_draws: ") + e.what());
  }
}

extern "C" int32_t ignis_sampled_drafter_distribution(const float *scores, uint32_t count, float temperature,
                                                      float top_p, float *out_q) {
  if (scores == nullptr || out_q == nullptr) {
    return fail("ignis_sampled_drafter_distribution: null argument");
  }
  if (count == 0 || count > IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES) {
    return fail("ignis_sampled_drafter_distribution: " + std::to_string(count) + " candidates, not 1.." +
                std::to_string(IGNIS_SAMPLED_DRAFT_MAX_CANDIDATES));
  }
  try {
    ninfer::DeviceBuffer d_scores = to_device(scores, count);
    ninfer::DeviceBuffer d_q(count * sizeof(float));
    drafter_distribution_kernel<<<1, 1>>>(static_cast<const float *>(d_scores.p), static_cast<int>(count),
                                          temperature, top_p, static_cast<float *>(d_q.p));
    synchronize("ignis_sampled_drafter_distribution");
    d_q.copy_to_host(out_q, count * sizeof(float));
    return 0;
  } catch (const std::exception &e) {
    return fail(std::string("ignis_sampled_drafter_distribution: ") + e.what());
  }
}

extern "C" const char *ignis_sampled_draft_last_error(void) {
  return g_last_error.c_str();
}
