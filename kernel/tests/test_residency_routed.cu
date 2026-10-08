// GitHub #306, step 5 -- OURS: expert residency's routed demand step
// (ignis_residency_step_demand_routed: the router's selection made inside the demand resolve's
// launch) held to the router and the demand step it replaces (ignis_moe_router then
// ignis_residency_step_demand), on two residencies created alike and stepped alike:
//   - the selection: ids, weights and the logits, bit for bit, every step;
//   - the step: each report's status, list counts and lists (as sets), each layer's slot table
//     (which keys are resident, at which K) and the counters, equal;
//   - the launches: the routed step captures one kernel fewer (logits, resolve, copy).
// Steps of one, two and three tokens over two layers whose pools are small enough to evict, a
// token whose logits all tie (the lower ids win) and one whose input is NaN (ids 0..9, NaN
// weights). Same no-SKIP_RETURN_CODE rule as every GPU test here (ADR 0006).

#include "ignis_moe.h"
#include "ignis_residency.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

#define CUDA_OK(expr)                                                                            \
  do {                                                                                           \
    const cudaError_t status_ = (expr);                                                          \
    if (status_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,                         \
                   cudaGetErrorString(status_));                                                 \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

#define API_OK(expr, error)                                                                      \
  do {                                                                                           \
    if ((expr) != 0) {                                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr, error());                \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

int g_failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

constexpr uint32_t kLayers = 2;
constexpr uint32_t kExperts = IGNIS_MOE_EXPERTS;
constexpr uint32_t kHidden = IGNIS_MOE_HIDDEN;
constexpr uint32_t kTopK = IGNIS_MOE_TOP_K;
constexpr uint32_t kMaxTokens = 3;
constexpr uint64_t kRecord = 4096;

uint32_t lcg(uint32_t &state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

std::vector<uint16_t> random_bf16(std::size_t n, float scale, uint32_t seed) {
  std::vector<uint16_t> out(n);
  for (auto &v : out) v = bf16_bits((static_cast<float>(lcg(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale);
  return out;
}

template <typename T>
std::vector<T> download(const void *p, std::size_t n) {
  std::vector<T> v(n);
  CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

struct Outcome {
  std::vector<int32_t> ids;
  std::vector<uint32_t> weights, logits;  // bits
  ignis_residency_report head{};
  std::vector<std::set<uint32_t>> lists;
  std::vector<std::pair<bool, uint32_t>> table;  // resident, K, every key of both layers
};

}  // namespace

int main() {
  CUDA_OK(cudaSetDevice(0));
  API_OK(ignis_moe_prepare(), ignis_moe_last_error);
  ignis_residency_desc desc{};
  desc.layers = kLayers;
  desc.experts = kExperts;
  for (int c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) desc.record_bytes[c] = kRecord;
  desc.capacity[0] = desc.capacity[4] = 24;  // gate/up and down at K = 4: a three-token step evicts
  desc.max_tokens = kMaxTokens;
  desc.lookahead_width = 4;
  desc.prefill_lookahead_width = 4;
  desc.prefetch_budget_one_row_bytes = IGNIS_RESIDENCY_NO_BUDGET;
  desc.staging_half_bytes = 2ull * kExperts * kRecord;
  desc.host_pool_bytes = static_cast<uint64_t>(kLayers) * 2 * kExperts * kRecord;
  desc.copy_blocks = 4;
  desc.report = 1;
  const std::vector<uint8_t> k2(static_cast<std::size_t>(kLayers) * 2 * kExperts, 4);
  std::vector<uint64_t> offsets(k2.size());
  for (std::size_t i = 0; i < offsets.size(); ++i) offsets[i] = i * kRecord;
  ignis_residency *r[2] = {};
  for (auto &one : r) API_OK(ignis_residency_create(&desc, k2.data(), offsets.data(), &one), ignis_residency_last_error);

  const auto w_router = random_bf16(static_cast<std::size_t>(kExperts) * kHidden, 0.05F, 7);
  void *d_w = nullptr, *d_x = nullptr;
  int32_t *d_ids[2] = {};
  float *d_weights[2] = {}, *d_logits[2] = {};
  CUDA_OK(cudaMalloc(&d_w, w_router.size() * 2));
  CUDA_OK(cudaMemcpy(d_w, w_router.data(), w_router.size() * 2, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMalloc(&d_x, static_cast<std::size_t>(kMaxTokens) * kHidden * 2));
  for (int i = 0; i < 2; ++i) {
    CUDA_OK(cudaMalloc(&d_ids[i], kMaxTokens * kTopK * 4));
    CUDA_OK(cudaMalloc(&d_weights[i], kMaxTokens * kTopK * 4));
    CUDA_OK(cudaMalloc(&d_logits[i], static_cast<std::size_t>(kMaxTokens) * kExperts * 4));
  }
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // One step on residency i: the router and the demand step (0), or the routed step (1).
  const auto step = [&](int i, uint32_t layer, uint32_t tokens, cudaStream_t s) {
    if (i == 0) {
      if (ignis_moe_router(d_x, tokens, d_w, d_ids[0], d_weights[0], d_logits[0], s) != 0) return -1;
      return ignis_residency_step_demand(r[0], layer, IGNIS_RESIDENCY_DECODE, d_ids[0], tokens, s, nullptr);
    }
    if (ignis_moe_router_logits(d_x, tokens, d_w, d_logits[1], s) != 0) return -1;
    return ignis_residency_step_demand_routed(r[1], layer, IGNIS_RESIDENCY_DECODE, d_logits[1], d_ids[1],
                                              d_weights[1], tokens, s, nullptr);
  };
  const auto outcome = [&](int i, uint32_t layer, uint32_t tokens) {
    Outcome o;
    o.ids = download<int32_t>(d_ids[i], tokens * kTopK);
    o.weights = download<uint32_t>(d_weights[i], tokens * kTopK);
    o.logits = download<uint32_t>(d_logits[i], static_cast<std::size_t>(tokens) * kExperts);
    std::vector<uint32_t> entries(static_cast<std::size_t>(IGNIS_RESIDENCY_LISTS) * 4 * kExperts);
    API_OK(ignis_residency_last_report(r[i], layer, &o.head, entries.data(), 4 * kExperts), ignis_residency_last_error);
    for (int l = 0; l < IGNIS_RESIDENCY_LISTS; ++l) {
      const auto first = entries.begin() + static_cast<std::ptrdiff_t>(l) * 4 * kExperts;
      o.lists.emplace_back(first, first + o.head.count[l]);
    }
    for (uint32_t L = 0; L < kLayers; ++L) {
      std::vector<ignis_moe_slot> table(2 * kExperts);
      CUDA_OK(cudaMemcpy(table.data(), ignis_residency_slot_table(r[i], L), table.size() * sizeof(ignis_moe_slot),
                         cudaMemcpyDeviceToHost));
      for (const auto &e : table) o.table.emplace_back(e.record != nullptr, e.k2);
    }
    return o;
  };

  struct Step {
    uint32_t layer, tokens;
    int kind;  // 0 random, 1 every logit tied (x zero), 2 a NaN input
  };
  const Step steps[] = {{0, 1, 0}, {1, 1, 0}, {0, 3, 0}, {1, 3, 0}, {0, 2, 0}, {1, 1, 1},
                        {0, 3, 0}, {1, 1, 2}, {0, 3, 0}, {1, 3, 0}, {0, 1, 0}, {1, 2, 0}};
  uint32_t seed = 11;
  int n = 0;
  for (const Step &s : steps) {
    const std::string at = "step " + std::to_string(n++) + " (layer " + std::to_string(s.layer) + ", " +
                           std::to_string(s.tokens) + " token(s))";
    std::vector<uint16_t> x = random_bf16(static_cast<std::size_t>(s.tokens) * kHidden, 1.0F, seed++);
    if (s.kind == 1) std::fill(x.begin(), x.end(), uint16_t{0});
    if (s.kind == 2) x[17] = 0x7FC0;  // NaN
    CUDA_OK(cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    for (int i = 0; i < 2; ++i) {
      CUDA_OK(cudaMemset(d_ids[i], 0xFF, kMaxTokens * kTopK * 4));
      CUDA_OK(cudaMemset(d_weights[i], 0xFF, kMaxTokens * kTopK * 4));
      if (step(i, s.layer, s.tokens, stream) != 0) {
        std::fprintf(stderr, "%s: %s / %s\n", at.c_str(), ignis_moe_last_error(), ignis_residency_last_error());
        return 1;
      }
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    const Outcome a = outcome(0, s.layer, s.tokens), b = outcome(1, s.layer, s.tokens);
    check(b.logits == a.logits, at + ": the logits are the router's");
    check(b.ids == a.ids, at + ": the ids are the router's");
    check(b.weights == a.weights, at + ": the weights are the router's, bit for bit");
    if (s.kind == 1) {
      bool lowest = true;
      for (uint32_t k = 0; k < kTopK; ++k) lowest = lowest && b.ids[k] == static_cast<int32_t>(k);
      check(lowest, at + ": tied logits pick the lowest ids");
    }
    if (s.kind == 2) {
      bool nan = true;
      for (uint32_t k = 0; k < kTopK; ++k) nan = nan && std::isnan(*reinterpret_cast<const float *>(&b.weights[k]));
      check(nan, at + ": a NaN input gives NaN weights");
    }
    check(b.head.status == a.head.status && b.head.bytes_moved == a.head.bytes_moved,
          at + ": the report's status and bytes are the demand step's");
    check(b.lists == a.lists, at + ": the report's lists are the demand step's");
    check(b.table == a.table, at + ": the slot tables are the demand step's");
  }
  ignis_residency_counters c[2] = {};
  for (int i = 0; i < 2; ++i) API_OK(ignis_residency_read_counters(r[i], &c[i]), ignis_residency_last_error);
  check(std::memcmp(c[0].hits, c[1].hits, sizeof(c[0].hits)) == 0 &&
            std::memcmp(c[0].misses, c[1].misses, sizeof(c[0].misses)) == 0 &&
            std::memcmp(c[0].bytes_moved, c[1].bytes_moved, sizeof(c[0].bytes_moved)) == 0 &&
            c[0].prefetch_used == c[1].prefetch_used,
        "the counters are the demand steps'");
  check(c[0].misses[0][IGNIS_RESIDENCY_DECODE] > 24, "the steps evicted (the comparison has power)");

  // Launches: the router (logits, select), the resolve and the demand copy, against the logits,
  // the routed resolve and the copy.
  int kernels[2] = {};
  for (int i = 0; i < 2; ++i) {
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int rc = step(i, 0, 1, stream);
    CUDA_OK(cudaStreamEndCapture(stream, &graph));
    check(rc == 0, "the step captures");
    std::size_t count = 0;
    CUDA_OK(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &count));
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type;
      CUDA_OK(cudaGraphNodeGetType(node, &type));
      kernels[i] += type == cudaGraphNodeTypeKernel ? 1 : 0;
    }
    CUDA_OK(cudaGraphDestroy(graph));
  }
  check(kernels[0] == 4 && kernels[1] == 3, "4 kernels unrouted, 3 routed (got " + std::to_string(kernels[0]) + ", " +
                                                std::to_string(kernels[1]) + ")");
  check(ignis_residency_step_demand_routed(r[1], 0, IGNIS_RESIDENCY_DECODE, nullptr, d_ids[1], d_weights[1], 1, stream,
                                           nullptr) != 0,
        "a routed step without logits is refused");

  for (auto *one : r) ignis_residency_free(one);
  CUDA_OK(cudaStreamDestroy(stream));
  if (g_failures != 0) {
    std::fprintf(stderr, "residency routed: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("residency routed: %zu steps, the routed step equal to the router and the demand step\n",
              sizeof(steps) / sizeof(steps[0]));
  return 0;
}
