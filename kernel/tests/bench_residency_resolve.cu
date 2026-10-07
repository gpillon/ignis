// Expert residency's per-layer cost at Flash-Next's geometry -- OURS, and NOT a CTest test (a
// timing is a finding, not a pass/fail; docs/agents/testing.md). Spec flash-next/03, GitHub #301.
//
// A residency with the real catalog shape (48 layers x 512 experts, K drawn per projection),
// real pool capacities (21.5 GB worth of slots split evenly by the record bytes) but 4 KiB
// records, so the timing is the resolve's decision work and a token copy, not PCIe. After a
// warm-up, steady-state decode rounds of 1 and 3 lanes walk the 48 layers with a skewed
// selection (most experts drawn from a hot quarter) and a W = 16 lookahead, and one 4096-token
// prefill chunk walks them once. Prints the median device time per layer step (event pairs
// around ignis_residency_step_ranked, prefetch join included), and for decode the split step's
// demand half alone (ignis_residency_step_demand: what the expert op waits for; the prefetch
// half runs on the branch it forks).
//
//   ignis_residency_resolve_bench [--rounds 64]

#include "ignis_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

#define CUDA_OK(expr)                                                                            \
  do {                                                                                           \
    const cudaError_t status_ = (expr);                                                          \
    if (status_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(status_));                     \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

#define RES_OK(expr)                                                                             \
  do {                                                                                           \
    if ((expr) != 0) {                                                                           \
      std::fprintf(stderr, "%s: %s\n", #expr, ignis_residency_last_error());                    \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

constexpr uint32_t kLayers = 48, kExperts = 512, kTopK = IGNIS_MOE_TOP_K, kWidth = 16;
constexpr uint64_t kRecord = 4096;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char **argv) {
  int rounds = 64;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
  }
  std::mt19937 rng(301);
  const uint64_t keys = static_cast<uint64_t>(kLayers) * kExperts * 2;
  std::vector<uint8_t> k2(keys);
  std::vector<uint64_t> offsets(keys);
  const uint8_t ks[4] = {4, 5, 6, 8};
  for (uint64_t i = 0; i < keys; ++i) {
    k2[i] = ks[rng() % 4];
    offsets[i] = i * kRecord;
  }
  ignis_residency_desc desc{};
  desc.layers = kLayers;
  desc.experts = kExperts;
  // ~21.5 GB of real records over 8 classes: about 2,600 slots a class at the 2.5-bit mean.
  for (int c = 0; c < 8; ++c) {
    desc.capacity[c] = 2600;
    desc.record_bytes[c] = kRecord;
  }
  desc.max_tokens = 4096;
  desc.lookahead_width = kWidth;
  desc.prefill_lookahead_width = kTopK;  // as Flash-Next loads it
  desc.prefetch_budget_one_row_bytes = 4 * kRecord;
  desc.staging_half_bytes = kExperts * 2 * kRecord;
  desc.host_pool_bytes = keys * kRecord;
  desc.copy_blocks = 16;
  desc.report = 0;
  ignis_residency *r = nullptr;
  RES_OK(ignis_residency_create(&desc, k2.data(), offsets.data(), &r));

  cudaStream_t s;
  CUDA_OK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t a, b;
  CUDA_OK(cudaEventCreate(&a));
  CUDA_OK(cudaEventCreate(&b));
  int32_t *d_ids = nullptr, *d_look = nullptr;
  CUDA_OK(cudaMalloc(&d_ids, 4096ull * kTopK * 4));
  CUDA_OK(cudaMalloc(&d_look, 4096ull * kWidth * 4));
  auto draw = [&](uint32_t rows, std::vector<int32_t> &ids, std::vector<int32_t> &look) {
    ids.assign(static_cast<size_t>(rows) * kTopK, 0);
    look.assign(static_cast<size_t>(rows) * kWidth, 0);
    for (uint32_t r_ = 0; r_ < rows; ++r_) {
      std::vector<int32_t> row;
      while (row.size() < kTopK) {
        const int32_t e = (rng() % 4 == 0) ? static_cast<int32_t>(rng() % kExperts)
                                           : static_cast<int32_t>(rng() % (kExperts / 4));
        if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
      }
      std::copy(row.begin(), row.end(), ids.begin() + static_cast<size_t>(r_) * kTopK);
      for (uint32_t w = 0; w < kWidth; ++w) look[static_cast<size_t>(r_) * kWidth + w] = static_cast<int32_t>(rng() % kExperts);
    }
  };

  auto run = [&](uint32_t rows, uint32_t phase, int n_rounds, bool timed, bool split = false) {
    std::vector<double> times;
    std::vector<int32_t> ids, look;
    for (int round = 0; round < n_rounds; ++round) {
      for (uint32_t layer = 0; layer < kLayers; ++layer) {
        draw(rows, ids, look);
        CUDA_OK(cudaMemcpyAsync(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, s));
        CUDA_OK(cudaMemcpyAsync(d_look, look.data(), look.size() * 4, cudaMemcpyHostToDevice, s));
        CUDA_OK(cudaEventRecord(a, s));
        if (split) {
          void *side = nullptr;
          RES_OK(ignis_residency_step_demand(r, layer, phase, d_ids, rows, s, &side));
          CUDA_OK(cudaEventRecord(b, s));
          if (side != nullptr) RES_OK(ignis_residency_step_prefetch_ranked(r, d_look, rows, kWidth));
          RES_OK(ignis_residency_join(r, s));
          CUDA_OK(cudaStreamSynchronize(s));
        } else {
          RES_OK(ignis_residency_step_ranked(r, layer, phase, d_ids, rows,
                                             layer + 1 < kLayers ? d_look : nullptr, rows, kWidth, s));
          RES_OK(ignis_residency_join(r, s));
          CUDA_OK(cudaEventRecord(b, s));
          CUDA_OK(cudaEventSynchronize(b));
        }
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, a, b));
        if (timed) times.push_back(ms * 1000.0);
      }
    }
    return times;
  };

  run(3, IGNIS_RESIDENCY_DECODE, 32, false);  // warm the pools
  for (uint32_t lanes : {1u, 3u}) {
    const auto t = run(lanes, IGNIS_RESIDENCY_DECODE, rounds, true);
    std::printf("decode, %u lane(s): %.1f us per layer step (median of %zu), %.2f ms per token over %u layers\n",
                lanes, median(t), t.size(), median(t) * kLayers / 1000.0, kLayers);
    const auto h = run(lanes, IGNIS_RESIDENCY_DECODE, rounds, true, true);
    std::printf("decode, %u lane(s), split: %.1f us per demand half (median of %zu), %.2f ms per token\n", lanes,
                median(h), h.size(), median(h) * kLayers / 1000.0);
  }
  const auto p = run(4096, IGNIS_RESIDENCY_PREFILL, 1, true);
  std::printf("prefill, 4096 tokens: %.1f us per layer step (median of %zu)\n", median(p), p.size());

  cudaFree(d_ids);
  cudaFree(d_look);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaStreamDestroy(s);
  ignis_residency_free(r);
  return 0;
}
