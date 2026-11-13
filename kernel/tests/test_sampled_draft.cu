// GitHub #576 (spec #574, ADR 0090): the sampled draft's kernels -- OURS, not
// vendored (kernel/src/sampled_draft.h, kernel/src/sampled_draft.cuh).
//
// The rule's arithmetic is held to a double-precision oracle elsewhere: the
// host reference by the CPU suite, these device functions by the GPU test
// crates/core/tests/sampled_draft_gpu.rs, over a grid of explicit uniforms.
// What this test holds is everything around that arithmetic, on a card of any
// generation the build covers:
//
//   distribution  the drafter's q on the device: the oracle's closed forms
//                 (softmax at a temperature, the sampler's inclusive top-p,
//                 ties toward the smaller index) and, at temperature 0, the
//                 selector walk's first maximum.
//   target        the accept's per-column target distribution p, read back
//                 from its workspace, is the vendored sampler's builder's bit
//                 for bit -- the penalties with the round-local overlay of
//                 the drafts before the column, top-k, temperature, top-p --
//                 on a vocabulary that spans several merge groups.
//   greedy        a greedy lane, at extent 0, at a short extent and at the
//                 window, writes exactly what the vendored accept writes.
//   bonus         so does a sampling lane at extent 0: its one token is the
//                 vendored accept's bonus draw, and its penalty count the same.
//   verdicts      a sampling lane with drafts writes what the host walk of the
//                 same verdicts writes over the p it built, with the uniforms
//                 the counter scheme gives at the rule's own purposes -- with
//                 q one-hot on the draft (a verify-only load) and with q
//                 explicit, half of it outside the target's support -- every
//                 produced token counted once.
//   draw          the draft kernel: a sampling lane's drafts and its kept q are
//                 the host walk's over the lattice, each column after the
//                 draft drawn before it; a greedy lane keeps the walk's drafts.
//   forced        GitHub #577 (test-only): a teacher-forced round's mode of
//                 both. The draw drafts each recorded token that is one of its
//                 column's candidates (its draw elsewhere) and keeps the q
//                 after it; the accept takes recorded tokens by the forced
//                 verdict (ignis_forced_sampled_verdict) exactly as the host
//                 walk of the same functions does, licenses the recording's
//                 run, counts the recorded tokens and never the pad, and
//                 leaves greedy lanes as without the recording.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or busy
// GPU fails this test rather than skipping it. Not sm120a-only: an RTX 30 or
// RTX 40 card runs it.

#include "sampled_draft.cuh"
#include "sampled_draft.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/common/sampling_workspace.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

#define CUDA_FATAL(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

using ninfer::ops::SamplingConfig;

constexpr int kTopK = kIgnisSampledDraftCandidates;

template <typename T>
struct Device {
  T *p = nullptr;
  std::size_t n = 0;
  explicit Device(std::size_t count) : n(count) { CUDA_FATAL(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); }
  explicit Device(const std::vector<T> &host) : Device(host.size()) { upload(host); }
  ~Device() { cudaFree(p); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void upload(const std::vector<T> &host) {
    CUDA_FATAL(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(n);
    CUDA_FATAL(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }
};

ninfer::Tensor tensor(void *data, ninfer::DType dtype, std::initializer_list<std::int32_t> shape) {
  return ninfer::Tensor(data, dtype, shape);
}

// --- the counter scheme's uniforms, as the device draws them ---------------

__global__ void uniforms_kernel(const unsigned long long *seeds, const int *positions, const int *purposes,
                                int count, float *out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    out[i] = ninfer::ops::sampling_uniform(seeds[i], positions[i], purposes[i], 0u);
  }
}

float uniform(unsigned long long seed, int position, int purpose) {
  Device<unsigned long long> seeds(std::vector<unsigned long long>{seed});
  Device<int> positions(std::vector<int>{position});
  Device<int> purposes(std::vector<int>{purpose});
  Device<float> out(1);
  uniforms_kernel<<<1, 1>>>(seeds.p, positions.p, purposes.p, 1, out.p);
  CUDA_FATAL(cudaGetLastError());
  return out.download()[0];
}

// --- arm 1: the drafter's distribution ---------------------------------------

__global__ void distribution_kernel(const float *scores, int n, float temperature, float top_p, float *q) {
  ignis_drafter_distribution(scores, n, temperature, top_p, q);
}

std::vector<float> device_distribution(const std::vector<float> &scores, float temperature, float top_p) {
  Device<float> d_scores(scores);
  Device<float> d_q(scores.size());
  distribution_kernel<<<1, 1>>>(d_scores.p, static_cast<int>(scores.size()), temperature, top_p, d_q.p);
  CUDA_FATAL(cudaGetLastError());
  return d_q.download();
}

void distribution_arm() {
  // The oracle's column: softmax at temperature 1 of 1, 8, 2, 4 and 1
  // sixteenths, out of order and with a tie (crates/core/tests/sampled_draft_oracle.rs).
  const double weights[5] = {1.0, 8.0, 2.0, 4.0, 1.0};
  std::vector<float> scores;
  for (const double w : weights) {
    scores.push_back(static_cast<float>(std::log(w / 16.0)));
  }
  double sqrt_total = 0.0;
  for (const double w : weights) {
    sqrt_total += std::sqrt(w);
  }
  struct Case {
    float temperature;
    float top_p;
    double expected[5];
  };
  const Case cases[] = {
      {1.0f, 1.0f, {1.0 / 16, 8.0 / 16, 2.0 / 16, 4.0 / 16, 1.0 / 16}},
      {1.0f, 0.9f, {1.0 / 15, 8.0 / 15, 2.0 / 15, 4.0 / 15, 0.0}},
      {0.5f, 0.95f, {0.0, 64.0 / 84, 4.0 / 84, 16.0 / 84, 0.0}},
      {2.0f, 0.95f, {1.0 / sqrt_total, std::sqrt(8.0) / sqrt_total, std::sqrt(2.0) / sqrt_total, 2.0 / sqrt_total, 1.0 / sqrt_total}},
  };
  for (const Case &c : cases) {
    const std::vector<float> q = device_distribution(scores, c.temperature, c.top_p);
    for (int i = 0; i < 5; ++i) {
      check(std::fabs(q[i] - c.expected[i]) <= 1e-6,
            "distribution at temperature " + std::to_string(c.temperature) + ", top-p " + std::to_string(c.top_p) +
                ", candidate " + std::to_string(i) + ": " + std::to_string(q[i]) + " against " +
                std::to_string(c.expected[i]));
      if (c.expected[i] == 0.0) {
        check(q[i] == 0.0f, "a candidate past the top-p cut keeps no mass");
      }
    }
  }
  // Four equal scores at top-p 0.5: the inclusive cut keeps the first two.
  const std::vector<float> tied = device_distribution({3.0f, 3.0f, 3.0f, 3.0f}, 1.0f, 0.5f);
  check(tied == std::vector<float>({0.5f, 0.5f, 0.0f, 0.0f}), "the inclusive cut breaks a tie toward the smaller index");
  // Temperature 0: all the mass on the first maximum, the walk's choice.
  const std::vector<float> walk = device_distribution({0.5f, 2.0f, -1.0f, 2.0f}, 0.0f, 0.95f);
  check(walk == std::vector<float>({0.0f, 1.0f, 0.0f, 0.0f}), "temperature 0 is one-hot on the first maximum");
}

// --- the round fixture ---------------------------------------------------------

constexpr int kVocab = 40000; // 79 partial tiles, four merge groups
constexpr int kWindow = 3;
constexpr int kCols = kWindow + 1;
constexpr int kLanes = 6;

std::uint16_t to_bf16(float x) {
  const __nv_bfloat16 b = __float2bfloat16(x);
  std::uint16_t bits;
  std::memcpy(&bits, &b, sizeof(bits));
  return bits;
}

float from_bf16(std::uint16_t bits) {
  std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float x;
  std::memcpy(&x, &wide, sizeof(x));
  return x;
}

struct Round {
  std::vector<std::uint16_t> logits;  // BF16 [vocab, k+1, B]
  std::vector<std::int32_t> targets;  // I32 [k+1, B]
  std::vector<std::int32_t> drafts;   // I32 [k, B]
  std::vector<std::int32_t> extents;  // I32 [B]
  std::vector<std::int32_t> lengths;  // I32 [B]
  std::vector<SamplingConfig> configs; // host view; token_counts patched per run
  std::vector<std::vector<std::int32_t>> counts; // each lane's penalty row, or empty
};

// Peaked columns: each column of each lane has one dominant token and a few
// runners-up, so a draft of the dominant token is often taken and the run goes
// past column 0.
Round make_round() {
  Round r;
  std::mt19937 rng(576);
  std::normal_distribution<float> noise(0.0f, 1.0f);
  std::uniform_int_distribution<int> any(0, kVocab - 1);
  r.logits.resize(static_cast<std::size_t>(kVocab) * kCols * kLanes);
  std::vector<int> dominant(kCols * kLanes);
  for (int b = 0; b < kLanes; ++b) {
    for (int c = 0; c < kCols; ++c) {
      const std::size_t base = (static_cast<std::size_t>(b) * kCols + c) * kVocab;
      for (int v = 0; v < kVocab; ++v) {
        r.logits[base + v] = to_bf16(noise(rng));
      }
      for (int j = 0; j < 6; ++j) {
        r.logits[base + any(rng)] = to_bf16(7.0f + 0.25f * j);
      }
      dominant[b * kCols + c] = any(rng);
      r.logits[base + dominant[b * kCols + c]] = to_bf16(12.0f);
    }
  }
  r.targets.resize(kCols * kLanes);
  for (int b = 0; b < kLanes; ++b) {
    for (int c = 0; c < kCols; ++c) {
      const std::size_t base = (static_cast<std::size_t>(b) * kCols + c) * kVocab;
      int best = 0;
      for (int v = 1; v < kVocab; ++v) {
        if (from_bf16(r.logits[base + v]) > from_bf16(r.logits[base + best])) {
          best = v;
        }
      }
      r.targets[b * kCols + c] = best;
    }
  }
  // Drafts: mostly each column's dominant token, one lane with a stranger.
  r.drafts.resize(kWindow * kLanes);
  for (int b = 0; b < kLanes; ++b) {
    for (int i = 0; i < kWindow; ++i) {
      r.drafts[b * kWindow + i] = dominant[b * kCols + i];
    }
  }
  r.drafts[1 * kWindow + 1] = 7; // lane 1: greedy, disagrees at its second draft
  r.drafts[5 * kWindow + 0] = 11; // lane 5: a draft the target hardly gives anything
  r.extents = {3, 2, 0, 3, 2, 3};
  r.lengths = {10, 40, 7, 99, 1000, 5};
  // Lanes 0 and 1 greedy; 2 samples at extent 0; 3 samples with penalties; 4
  // at top-k 5 and top-p 0.8; 5 a cold one with a wide support.
  r.configs.resize(kLanes);
  r.configs[0].temperature = 0.0f;
  r.configs[1].temperature = 0.0f;
  r.configs[2] = SamplingConfig{1.0f, 0, 0.95f, 0.0f, 0.2f, 0.1f, 11ull, nullptr};
  r.configs[3] = SamplingConfig{0.8f, 20, 1.0f, 0.0f, 1.5f, 0.5f, 12ull, nullptr};
  r.configs[4] = SamplingConfig{1.2f, 5, 0.8f, 0.0f, 0.0f, 0.0f, 13ull, nullptr};
  r.configs[5] = SamplingConfig{2.0f, 0, 1.0f, 0.0f, 0.0f, 0.0f, 14ull, nullptr};
  r.counts.resize(kLanes);
  for (const int b : {2, 3}) {
    r.counts[b].assign(kVocab, 0);
    // Lane 3 has already said its first draft twice and its second once.
    if (b == 3) {
      r.counts[b][r.drafts[3 * kWindow + 0]] = 2;
      r.counts[b][r.drafts[3 * kWindow + 1]] = 1;
    }
  }
  return r;
}

struct Outputs {
  std::vector<std::int32_t> licensed, counts, accepted, anchors, lengths;
  std::vector<std::vector<std::int32_t>> penalty;
  // Our accept's per-lane target distributions, column by column: ids, probabilities.
  std::vector<std::vector<std::vector<std::int32_t>>> p_ids;
  std::vector<std::vector<std::vector<float>>> p_probs;
};

enum class Accept { Vendored, Ours };

// One accept over a fresh copy of `round`. `q_ids`/`q_probs` are ours' kept q
// ([16, k, B]), or empty for a one-hot q.
Outputs run_accept(const Round &round, Accept which, const std::vector<std::int32_t> &q_ids = {},
                   const std::vector<float> &q_probs = {}, const std::vector<std::int32_t> &forced = {}) {
  std::unique_ptr<Device<std::int32_t>> d_forced;
  if (!forced.empty()) {
    d_forced = std::make_unique<Device<std::int32_t>>(forced);
  }
  Device<std::uint16_t> logits(round.logits);
  Device<std::int32_t> targets(round.targets);
  Device<std::int32_t> drafts(round.drafts);
  Device<std::int32_t> extents(round.extents);
  Device<std::int32_t> lengths(round.lengths);
  Device<std::int32_t> anchors(kLanes);
  Device<std::int32_t> licensed(kCols * kLanes);
  Device<std::int32_t> licensed_counts(kLanes);
  Device<std::int32_t> accepted(kLanes);
  std::vector<std::unique_ptr<Device<std::int32_t>>> penalty(kLanes);
  std::vector<SamplingConfig> configs = round.configs;
  for (int b = 0; b < kLanes; ++b) {
    if (!round.counts[b].empty()) {
      penalty[b] = std::make_unique<Device<std::int32_t>>(round.counts[b]);
      configs[b].token_counts = penalty[b]->p;
    }
  }
  Device<SamplingConfig> d_configs(configs);
  std::unique_ptr<Device<std::int32_t>> d_q_ids;
  std::unique_ptr<Device<float>> d_q_probs;
  if (!q_ids.empty()) {
    d_q_ids = std::make_unique<Device<std::int32_t>>(q_ids);
    d_q_probs = std::make_unique<Device<float>>(q_probs);
  }
  const ninfer::Tensor t_targets = tensor(targets.p, ninfer::DType::I32, {kCols, kLanes, 1, 1});
  const ninfer::Tensor t_logits = tensor(logits.p, ninfer::DType::BF16, {kVocab, kCols, kLanes, 1});
  const ninfer::Tensor t_drafts = tensor(drafts.p, ninfer::DType::I32, {kWindow, kLanes, 1, 1});
  const ninfer::Tensor t_extents = tensor(extents.p, ninfer::DType::I32, {kLanes, 1, 1, 1});
  ninfer::Tensor t_lengths = tensor(lengths.p, ninfer::DType::I32, {kLanes, 1, 1, 1});
  ninfer::Tensor t_anchors = tensor(anchors.p, ninfer::DType::I32, {kLanes, 1, 1, 1});
  ninfer::Tensor t_licensed = tensor(licensed.p, ninfer::DType::I32, {kCols, kLanes, 1, 1});
  ninfer::Tensor t_counts = tensor(licensed_counts.p, ninfer::DType::I32, {kLanes, 1, 1, 1});
  ninfer::Tensor t_accepted = tensor(accepted.p, ninfer::DType::I32, {kLanes, 1, 1, 1});
  const std::size_t workspace_bytes = std::max<std::size_t>(
      ninfer::ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(kVocab, kWindow, kWindow, kLanes,
                                                                             kLanes),
      256);
  ninfer::DeviceArena arena(workspace_bytes);
  if (which == Accept::Vendored) {
    ninfer::ops::speculative_accept_greedy_drafts(t_targets, t_logits, t_drafts, t_extents, t_lengths, t_anchors,
                                                  t_licensed, t_counts, t_accepted, kVocab, d_configs.p, arena,
                                                  nullptr);
  } else {
    check(ignis_sampled_accept_workspace_bytes(kVocab, kWindow, kLanes) <= workspace_bytes,
          "our accept fits the vendored accept's workspace");
    ignis_sampled_accept(t_targets, t_logits, t_drafts, t_extents, t_lengths, t_anchors, t_licensed, t_counts,
                         t_accepted, kVocab, d_configs.p, d_q_ids ? d_q_ids->p : nullptr,
                         d_q_probs ? d_q_probs->p : nullptr, arena, nullptr, d_forced ? d_forced->p : nullptr);
  }
  CUDA_FATAL(cudaDeviceSynchronize());
  Outputs out;
  out.licensed = licensed.download();
  out.counts = licensed_counts.download();
  out.accepted = accepted.download();
  out.anchors = anchors.download();
  out.lengths = lengths.download();
  out.penalty.resize(kLanes);
  for (int b = 0; b < kLanes; ++b) {
    if (penalty[b]) {
      out.penalty[b] = penalty[b]->download();
    }
  }
  if (which == Accept::Ours) {
    // The workspace's first allocation is ours: the sampler's layout of k+1
    // columns, once per lane.
    const ninfer::ops::SamplingWorkspaceLayout layout = ninfer::ops::make_sampling_workspace_layout(kVocab, kCols);
    std::vector<unsigned char> bytes(layout.bytes * kLanes);
    CUDA_FATAL(cudaMemcpy(bytes.data(), arena.base(), bytes.size(), cudaMemcpyDeviceToHost));
    out.p_ids.resize(kLanes);
    out.p_probs.resize(kLanes);
    for (int b = 0; b < kLanes; ++b) {
      if (!(round.configs[b].temperature > 0.0f)) {
        continue;
      }
      const unsigned char *row = bytes.data() + layout.bytes * b;
      const auto *support = reinterpret_cast<const std::int32_t *>(row + layout.dist_support.region.offset);
      const auto *ids = reinterpret_cast<const std::int32_t *>(row + layout.dist_idx.region.offset);
      const auto *probs = reinterpret_cast<const float *>(row + layout.dist_prob.region.offset);
      for (int c = 0; c <= round.extents[b]; ++c) {
        const int n = support[c];
        out.p_ids[b].emplace_back(ids + c * ninfer::ops::kSamplerCandidateCap,
                                  ids + c * ninfer::ops::kSamplerCandidateCap + n);
        out.p_probs[b].emplace_back(probs + c * ninfer::ops::kSamplerCandidateCap,
                                    probs + c * ninfer::ops::kSamplerCandidateCap + n);
      }
    }
  }
  return out;
}

// --- arm 2: the target distribution, against the vendored builder -------------

__global__ void vendored_build_kernel(const __nv_bfloat16 *logits, const std::int32_t *drafts,
                                      const SamplingConfig *configs, int lane, int col, int *out_n,
                                      int *out_ids, float *out_probs) {
  __shared__ float merge_val[ninfer::ops::kSamplerBlock * ninfer::ops::kSamplerFastCandidates];
  __shared__ int merge_idx[ninfer::ops::kSamplerBlock * ninfer::ops::kSamplerFastCandidates];
  __shared__ float cand_val[ninfer::ops::kSamplerCandidateCap];
  __shared__ int cand_idx[ninfer::ops::kSamplerCandidateCap];
  __shared__ float prob[ninfer::ops::kSamplerCandidateCap];
  __shared__ int n_support;
  const std::int64_t base = (static_cast<std::int64_t>(lane) * kCols + col) * kVocab;
  ninfer::ops::sampling_build_truncated_block_fast(logits, base, kVocab, configs[lane], merge_val, merge_idx,
                                                   cand_val, cand_idx, prob, &n_support, drafts + lane * kWindow,
                                                   col);
  if (threadIdx.x == 0) {
    *out_n = n_support;
    for (int j = 0; j < n_support; ++j) {
      out_ids[j] = cand_idx[j];
      out_probs[j] = prob[j];
    }
  }
}

void target_arm(const Round &round, const Outputs &ours) {
  Device<std::uint16_t> logits(round.logits);
  Device<std::int32_t> drafts(round.drafts);
  std::vector<std::unique_ptr<Device<std::int32_t>>> penalty(kLanes);
  std::vector<SamplingConfig> configs = round.configs;
  for (int b = 0; b < kLanes; ++b) {
    if (!round.counts[b].empty()) {
      penalty[b] = std::make_unique<Device<std::int32_t>>(round.counts[b]);
      configs[b].token_counts = penalty[b]->p;
    }
  }
  Device<SamplingConfig> d_configs(configs);
  Device<int> n(1);
  Device<int> ids(ninfer::ops::kSamplerCandidateCap);
  Device<float> probs(ninfer::ops::kSamplerCandidateCap);
  int columns = 0;
  for (int b = 0; b < kLanes; ++b) {
    if (!(round.configs[b].temperature > 0.0f)) {
      continue;
    }
    for (int c = 0; c <= round.extents[b]; ++c) {
      vendored_build_kernel<<<1, ninfer::ops::kSamplerBlock>>>(reinterpret_cast<const __nv_bfloat16 *>(logits.p),
                                                               drafts.p, d_configs.p, b, c, n.p, ids.p, probs.p);
      CUDA_FATAL(cudaGetLastError());
      const int support = n.download()[0];
      const std::vector<int> all_ids = ids.download();
      const std::vector<int> want_ids(all_ids.begin(), all_ids.begin() + support);
      const std::vector<float> all_probs = probs.download();
      const std::vector<float> want_probs(all_probs.begin(), all_probs.begin() + support);
      const std::string where = "lane " + std::to_string(b) + " column " + std::to_string(c);
      check(ours.p_ids[b][c] == want_ids, where + ": the support is not the vendored builder's");
      check(ours.p_probs[b][c] == want_probs, where + ": the probabilities are not the vendored builder's, bit for bit");
      ++columns;
    }
  }
  check(columns == 1 + 4 + 3 + 4, "every sampling column was compared"); // lanes 2..5: extents 0, 3, 2, 3
  // Lane 4's top-k of 5 and top-p of 0.8 cut its support.
  for (const auto &column : ours.p_ids[4]) {
    check(!column.empty() && column.size() <= 5, "a top-k of 5 holds at most five tokens");
  }
}

// --- arm 4: the host walk of the verdicts --------------------------------------

struct Lane {
  std::vector<std::int32_t> licensed;
  std::int32_t accepted = 0, anchor = 0, length = 0;
};

Lane host_lane(const Round &round, const Outputs &ours, int b, const std::vector<std::int32_t> &q_ids,
               const std::vector<float> &q_probs) {
  const SamplingConfig &cfg = round.configs[b];
  const int extent = round.extents[b];
  const int length = round.lengths[b];
  Lane lane;
  lane.licensed.assign(kCols, 0);
  int a = 0;
  int tstar = 0;
  for (int i = 0; i <= extent; ++i) {
    const std::vector<std::int32_t> &ids = ours.p_ids[b][i];
    const std::vector<float> &probs = ours.p_probs[b][i];
    if (i == extent) {
      const float u = uniform(cfg.seed, length + extent + 1, ninfer::ops::kSamplePurposeSpeculativeBonus);
      // The vendored support pick, walked on the host.
      float mass = 0.0f;
      for (const float p : probs) {
        mass += p;
      }
      const float goal = u * mass;
      float acc = 0.0f;
      tstar = ids.back();
      for (std::size_t j = 0; j < ids.size(); ++j) {
        acc += probs[j];
        if (goal < acc) {
          tstar = ids[j];
          break;
        }
      }
      break;
    }
    const std::int32_t draft = round.drafts[b * kWindow + i];
    const float one = 1.0f;
    const std::int32_t *drafter_ids = &draft;
    const float *drafter_probs = &one;
    int n_drafter = 1;
    if (!q_ids.empty()) {
      drafter_ids = q_ids.data() + kTopK * (b * kWindow + i);
      drafter_probs = q_probs.data() + kTopK * (b * kWindow + i);
      n_drafter = kTopK;
    }
    std::int32_t token = 0;
    if (ignis_sampled_verdict(ids.data(), probs.data(), static_cast<int>(ids.size()), drafter_ids, drafter_probs,
                              n_drafter, draft, uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeSampledAccept),
                              uniform(cfg.seed, length + i + 1, kIgnisSamplePurposeSampledRefusal), &token)) {
      a = i + 1;
      continue;
    }
    tstar = token;
    break;
  }
  for (int i = 0; i < a; ++i) {
    lane.licensed[i] = round.drafts[b * kWindow + i];
  }
  lane.licensed[a] = tstar;
  lane.accepted = a;
  lane.anchor = tstar;
  lane.length = length + a + 1;
  return lane;
}

void verdict_arm(const Round &round, const std::vector<std::int32_t> &q_ids, const std::vector<float> &q_probs,
                 const std::string &label) {
  const Outputs ours = run_accept(round, Accept::Ours, q_ids, q_probs);
  int runs_past_column_0 = 0;
  for (const int b : {3, 4, 5}) {
    const Lane want = host_lane(round, ours, b, q_ids, q_probs);
    const std::string where = label + " lane " + std::to_string(b);
    check(std::vector<std::int32_t>(ours.licensed.begin() + b * kCols, ours.licensed.begin() + (b + 1) * kCols) ==
              want.licensed,
          where + ": the licensed run is not the host walk's");
    check(ours.accepted[b] == want.accepted && ours.counts[b] == want.accepted + 1, where + ": accepted and licensed counts");
    check(ours.anchors[b] == want.anchor && ours.lengths[b] == want.length, where + ": the anchor and the frontier");
    runs_past_column_0 += want.accepted > 0 ? 1 : 0;
    if (!round.counts[b].empty()) {
      std::vector<std::int32_t> counted = round.counts[b];
      for (int i = 0; i <= want.accepted; ++i) {
        ++counted[want.licensed[i]];
      }
      check(ours.penalty[b] == counted, where + ": every produced token counted once in the penalty row");
    }
  }
  check(runs_past_column_0 > 0, label + ": no lane took a draft, so the walk past column 0 went unexercised");
  // Lane 5's stranger draft (token 11) has no mass in p: always refused.
  check(ours.accepted[5] == 0 && ours.licensed[5 * kCols] != 11, label + ": a draft outside the support was taken");
}

// --- arm 6: the draft kernel ---------------------------------------------------

void draw_arm() {
  constexpr int kLanesDraw = 3;
  std::mt19937 rng(5760);
  std::normal_distribution<float> noise(0.0f, 1.5f);
  std::vector<float> scores(kTopK * kTopK * kWindow * kLanesDraw);
  for (float &s : scores) {
    s = noise(rng);
  }
  std::vector<std::int32_t> candidates(kTopK * kWindow * kLanesDraw);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    candidates[i] = static_cast<std::int32_t>(1000 + 7 * i);
  }
  const std::vector<std::int32_t> walk(kWindow * kLanesDraw, -5);
  std::vector<SamplingConfig> configs(kLanesDraw);
  configs[0] = SamplingConfig{1.0f, 0, 1.0f, 0.0f, 0.0f, 0.0f, 101ull, nullptr};
  configs[1].temperature = 0.0f; // greedy: keeps the walk's drafts
  configs[2] = SamplingConfig{0.6f, 20, 0.95f, 0.0f, 0.0f, 0.0f, 202ull, nullptr};
  const std::vector<std::int32_t> base_positions = {17, 3, 250};
  const float factor = 0.9f;
  const float top_p = 0.95f;

  Device<float> d_scores(scores);
  Device<std::int32_t> d_candidates(candidates);
  Device<std::int32_t> d_drafts(walk);
  Device<SamplingConfig> d_configs(configs);
  Device<std::int32_t> d_base(base_positions);
  Device<std::int32_t> d_q_ids(std::vector<std::int32_t>(kTopK * kWindow * kLanesDraw, -9));
  Device<float> d_q(std::vector<float>(kTopK * kWindow * kLanesDraw, -1.0f));
  const ninfer::Tensor t_scores = tensor(d_scores.p, ninfer::DType::FP32, {kTopK, kTopK, kWindow, kLanesDraw});
  const ninfer::Tensor t_candidates = tensor(d_candidates.p, ninfer::DType::I32, {kTopK, kWindow, kLanesDraw, 1});
  ninfer::Tensor t_drafts = tensor(d_drafts.p, ninfer::DType::I32, {kWindow, kLanesDraw, 1, 1});
  ignis_sampled_draft_draw(t_scores, t_candidates, d_configs.p, d_base.p, factor, top_p, t_drafts, d_q_ids.p, d_q.p,
                           nullptr);
  CUDA_FATAL(cudaDeviceSynchronize());
  const std::vector<std::int32_t> drafts = d_drafts.download();
  const std::vector<std::int32_t> q_ids = d_q_ids.download();
  const std::vector<float> q = d_q.download();

  for (int s = 0; s < kWindow; ++s) {
    check(drafts[1 * kWindow + s] == -5, "a greedy lane keeps the walk's drafts");
    for (int i = 0; i < kTopK; ++i) {
      check(q_ids[kTopK * (kWindow + s) + i] == -9, "a greedy lane's q is left as it was");
    }
  }
  int moved = 0;
  for (const int lane : {0, 2}) {
    int pred = 0;
    for (int s = 0; s < kWindow; ++s) {
      const int slot = lane * kWindow + s;
      float want[kTopK];
      ignis_drafter_distribution(scores.data() + kTopK * (pred + kTopK * slot), kTopK,
                                 configs[lane].temperature * factor, top_p, want);
      const float u = uniform(configs[lane].seed, base_positions[lane] + s + 1, kIgnisSamplePurposeDraft);
      const int picked = ignis_draw_draft(want, kTopK, u);
      const std::string where = "lane " + std::to_string(lane) + " column " + std::to_string(s);
      check(drafts[slot] == candidates[kTopK * slot + picked], where + ": the draft is not the host walk's draw");
      for (int i = 0; i < kTopK; ++i) {
        check(q_ids[kTopK * slot + i] == candidates[kTopK * slot + i], where + ": the kept q's ids");
        check(std::fabs(q[kTopK * slot + i] - want[i]) <= 1e-6f, where + ": the kept q's probabilities");
      }
      // The walk's own choice after the same predecessor: how often the draw moved off it.
      float greedy[kTopK];
      ignis_drafter_distribution(scores.data() + kTopK * (pred + kTopK * slot), kTopK, 0.0f, top_p, greedy);
      moved += greedy[picked] == 1.0f ? 0 : 1;
      pred = picked;
    }
  }
  check(moved > 0, "no draw left the walk's draft: the draw went unexercised");
}


// --- arm 7: the teacher-forced round (GitHub #577) -----------------------------

// The draw drafts along a recording: each recorded token that is one of its
// column's candidates, its draw elsewhere, the next column's q after it.
void forced_draw_arm() {
  constexpr int kLanesDraw = 2;
  std::mt19937 rng(5770);
  std::normal_distribution<float> noise(0.0f, 1.5f);
  std::vector<float> scores(kTopK * kTopK * kWindow * kLanesDraw);
  for (float &s : scores) {
    s = noise(rng);
  }
  std::vector<std::int32_t> candidates(kTopK * kWindow * kLanesDraw);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    candidates[i] = static_cast<std::int32_t>(2000 + 5 * i);
  }
  const std::vector<std::int32_t> walk(kWindow * kLanesDraw, -5);
  std::vector<SamplingConfig> configs(kLanesDraw);
  configs[0] = SamplingConfig{1.0f, 20, 0.95f, 0.0f, 0.0f, 0.0f, 303ull, nullptr};
  configs[1].temperature = 0.0f; // greedy: keeps the walk's drafts, forced or not
  const std::vector<std::int32_t> base_positions = {41, 9};
  const float factor = 0.9f;
  const float top_p = 0.95f;
  // Lane 0 records candidate 3 of column 0, candidate 9 of column 1, then a
  // token no candidate of column 2 holds; lane 1's recording is the same.
  std::vector<std::int32_t> forced(kCols * kLanesDraw, -1);
  for (int b = 0; b < kLanesDraw; ++b) {
    forced[b * kCols + 0] = candidates[kTopK * (0 * kWindow + 0) + 3];
    forced[b * kCols + 1] = candidates[kTopK * (0 * kWindow + 1) + 9];
    forced[b * kCols + 2] = 77777;
    forced[b * kCols + 3] = 88888;
  }

  Device<float> d_scores(scores);
  Device<std::int32_t> d_candidates(candidates);
  Device<std::int32_t> d_drafts(walk);
  Device<SamplingConfig> d_configs(configs);
  Device<std::int32_t> d_base(base_positions);
  Device<std::int32_t> d_forced(forced);
  Device<std::int32_t> d_q_ids(std::vector<std::int32_t>(kTopK * kWindow * kLanesDraw, -9));
  Device<float> d_q(std::vector<float>(kTopK * kWindow * kLanesDraw, -1.0f));
  const ninfer::Tensor t_scores = tensor(d_scores.p, ninfer::DType::FP32, {kTopK, kTopK, kWindow, kLanesDraw});
  const ninfer::Tensor t_candidates = tensor(d_candidates.p, ninfer::DType::I32, {kTopK, kWindow, kLanesDraw, 1});
  ninfer::Tensor t_drafts = tensor(d_drafts.p, ninfer::DType::I32, {kWindow, kLanesDraw, 1, 1});
  ignis_sampled_draft_draw(t_scores, t_candidates, d_configs.p, d_base.p, factor, top_p, t_drafts, d_q_ids.p, d_q.p,
                           nullptr, d_forced.p);
  CUDA_FATAL(cudaDeviceSynchronize());
  const std::vector<std::int32_t> drafts = d_drafts.download();
  const std::vector<float> q = d_q.download();

  for (int s = 0; s < kWindow; ++s) {
    check(drafts[1 * kWindow + s] == -5, "forced: a greedy lane keeps the walk's drafts");
  }
  int pred = 0;
  for (int s = 0; s < kWindow; ++s) {
    const int slot = s;
    float want[kTopK];
    ignis_drafter_distribution(scores.data() + kTopK * (pred + kTopK * slot), kTopK, configs[0].temperature * factor,
                               top_p, want);
    int picked = ignis_draw_draft(want, kTopK, uniform(configs[0].seed, base_positions[0] + s + 1,
                                                       kIgnisSamplePurposeDraft));
    for (int i = 0; i < kTopK; ++i) {
      if (candidates[kTopK * slot + i] == forced[s]) {
        picked = i;
      }
    }
    const std::string where = "forced: column " + std::to_string(s);
    check(drafts[slot] == candidates[kTopK * slot + picked], where + ": not the recording's draft (or the draw)");
    for (int i = 0; i < kTopK; ++i) {
      check(std::fabs(q[kTopK * slot + i] - want[i]) <= 1e-6f, where + ": the kept q is not the one after the draft before");
    }
    pred = picked;
  }
  check(drafts[0] == forced[0] && drafts[1] == forced[1], "forced: the recorded candidates are drafted");
  check(drafts[2] != forced[2], "forced: a recorded token no candidate holds cannot be drafted");
}

// The accept's forced verdicts, walked on the host with the same functions.
Lane host_forced_lane(const Round &round, const Outputs &ours, int b, const std::vector<std::int32_t> &q_ids,
                      const std::vector<float> &q_probs, const std::vector<std::int32_t> &forced) {
  const SamplingConfig &cfg = round.configs[b];
  const int extent = round.extents[b];
  const int length = round.lengths[b];
  Lane lane;
  lane.licensed.assign(kCols, 0);
  int a = 0;
  std::int32_t tstar = 0;
  for (int i = 0; i <= extent; ++i) {
    const std::int32_t recorded = forced[b * kCols + i];
    if (i == extent) {
      tstar = recorded;
      break;
    }
    const std::vector<std::int32_t> &ids = ours.p_ids[b][i];
    const std::vector<float> &probs = ours.p_probs[b][i];
    const std::int32_t draft = round.drafts[b * kWindow + i];
    bool taken = false;
    if (draft == recorded) {
      const std::int32_t *drafter_ids = q_ids.data() + kTopK * (b * kWindow + i);
      const float *drafter_probs = q_probs.data() + kTopK * (b * kWindow + i);
      const int pos = length + i + 1;
      const std::int32_t live_draft =
          drafter_ids[ignis_draw_draft(drafter_probs, kTopK, uniform(cfg.seed, pos, kIgnisSamplePurposeDraft))];
      std::int32_t live_token = 0;
      const bool live_taken = ignis_sampled_verdict(
          ids.data(), probs.data(), static_cast<int>(ids.size()), drafter_ids, drafter_probs, kTopK, live_draft,
          uniform(cfg.seed, pos, kIgnisSamplePurposeSampledAccept), uniform(cfg.seed, pos, kIgnisSamplePurposeSampledRefusal),
          &live_token);
      taken = ignis_forced_sampled_verdict(ids.data(), probs.data(), static_cast<int>(ids.size()), drafter_ids,
                                           drafter_probs, kTopK, recorded, live_draft, live_taken, live_token,
                                           uniform(cfg.seed, pos, kIgnisSamplePurposeForcedCoin));
    }
    if (!taken) {
      tstar = recorded;
      break;
    }
    a = i + 1;
  }
  for (int i = 0; i < a; ++i) {
    lane.licensed[i] = forced[b * kCols + i];
  }
  lane.licensed[a] = tstar;
  lane.accepted = a;
  return lane;
}

void forced_accept_arm(const Round &round, const std::vector<std::int32_t> &q_ids, const std::vector<float> &q_probs) {
  // Each lane records its own drafts, then: lane 2 (extent 0) one token;
  // lane 3 (penalties) its three drafts and the pad after them, as after a
  // stop; lane 4 a token at its second column no draft holds; lane 5 its
  // drafts and a token after them.
  std::vector<std::int32_t> forced(kCols * kLanes, 0);
  for (int b = 0; b < kLanes; ++b) {
    for (int i = 0; i < kWindow; ++i) {
      forced[b * kCols + i] = round.drafts[b * kWindow + i];
    }
    forced[b * kCols + kWindow] = 321 + b;
  }
  forced[2 * kCols + 0] = 123;
  forced[3 * kCols + 3] = -1;
  forced[4 * kCols + 1] = 999;
  const Outputs plain = run_accept(round, Accept::Ours, q_ids, q_probs);
  const Outputs ours = run_accept(round, Accept::Ours, q_ids, q_probs, forced);
  for (const int b : {0, 1}) {
    check(std::vector<std::int32_t>(ours.licensed.begin() + b * kCols, ours.licensed.begin() + (b + 1) * kCols) ==
                  std::vector<std::int32_t>(plain.licensed.begin() + b * kCols,
                                            plain.licensed.begin() + (b + 1) * kCols) &&
              ours.accepted[b] == plain.accepted[b],
          "forced: greedy lane " + std::to_string(b) + " is not as without the recording");
  }
  int taken_any = 0;
  for (const int b : {2, 3, 4, 5}) {
    const Lane want = host_forced_lane(round, ours, b, q_ids, q_probs, forced);
    const std::string where = "forced lane " + std::to_string(b);
    check(std::vector<std::int32_t>(ours.licensed.begin() + b * kCols,
                                    ours.licensed.begin() + b * kCols + want.accepted + 1) ==
              std::vector<std::int32_t>(want.licensed.begin(), want.licensed.begin() + want.accepted + 1),
          where + ": the licensed run is not the host walk's");
    check(ours.accepted[b] == want.accepted && ours.counts[b] == want.accepted + 1, where + ": accepted and licensed counts");
    for (int i = 0; i <= want.accepted; ++i) {
      check(want.licensed[i] == forced[b * kCols + i], where + ": the run is not the recording's");
    }
    taken_any += want.accepted;
    if (!round.counts[b].empty()) {
      std::vector<std::int32_t> counted = round.counts[b];
      for (int i = 0; i <= want.accepted; ++i) {
        if (want.licensed[i] >= 0) {
          ++counted[want.licensed[i]];
        }
      }
      check(ours.penalty[b] == counted, where + ": every recorded token counted once, the pad never");
    }
  }
  check(ours.accepted[2] == 0 && ours.licensed[2 * kCols] == 123, "forced lane 2: extent 0 emits its recorded token");
  check(ours.accepted[4] <= 1, "forced lane 4: no draft holds its second recorded token, so the run stops there");
  check(taken_any > 0, "forced: no lane took a recorded token, so the walk past column 0 went unexercised");
}

} // namespace

int main() {
  int devices = 0;
  CUDA_FATAL(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return EXIT_FAILURE;
  }

  distribution_arm();

  const Round round = make_round();
  const Outputs vendored = run_accept(round, Accept::Vendored);
  const Outputs ours = run_accept(round, Accept::Ours);

  target_arm(round, ours);

  // Arm 3 and 5: the greedy lanes and the sampling lane at extent 0 write the
  // vendored accept's outputs.
  for (const int b : {0, 1, 2}) {
    const std::string where = "lane " + std::to_string(b);
    check(std::vector<std::int32_t>(ours.licensed.begin() + b * kCols, ours.licensed.begin() + (b + 1) * kCols) ==
              std::vector<std::int32_t>(vendored.licensed.begin() + b * kCols,
                                        vendored.licensed.begin() + (b + 1) * kCols),
          where + ": the licensed run is not the vendored accept's");
    check(ours.counts[b] == vendored.counts[b] && ours.accepted[b] == vendored.accepted[b] &&
              ours.anchors[b] == vendored.anchors[b] && ours.lengths[b] == vendored.lengths[b],
          where + ": the counts, the anchor or the frontier is not the vendored accept's");
    check(ours.penalty[b] == vendored.penalty[b], where + ": the penalty row is not the vendored accept's");
  }
  check(vendored.accepted[0] == 3 && vendored.accepted[1] == 1,
        "the greedy lanes took their agreeing drafts (the fixture's own check)");

  // Arm 4: the verdicts, with q one-hot and with an explicit q.
  verdict_arm(round, {}, {}, "one-hot q");
  std::vector<std::int32_t> q_ids(kTopK * kWindow * kLanes);
  std::vector<float> q_probs(kTopK * kWindow * kLanes, 0.0f);
  for (int b = 0; b < kLanes; ++b) {
    for (int i = 0; i < kWindow; ++i) {
      const int slot = b * kWindow + i;
      // The draft and the target's top token a quarter of the mass each (the
      // draft half when it is the top token), and half spread over fourteen
      // ids the target never ranks.
      const bool top = round.targets[b * kCols + i] == round.drafts[slot];
      for (int j = 0; j < kTopK; ++j) {
        q_ids[kTopK * slot + j] = 30000 + 16 * slot + j;
        q_probs[kTopK * slot + j] = 0.5f / 14.0f;
      }
      q_ids[kTopK * slot] = round.drafts[slot];
      q_probs[kTopK * slot] = top ? 0.5f : 0.25f;
      q_ids[kTopK * slot + 1] = round.targets[b * kCols + i];
      q_probs[kTopK * slot + 1] = top ? 0.0f : 0.25f;
    }
  }
  verdict_arm(round, q_ids, q_probs, "explicit q");
  forced_accept_arm(round, q_ids, q_probs);

  draw_arm();
  forced_draw_arm();

  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("sampled draft: all checks passed\n");
  return EXIT_SUCCESS;
}
