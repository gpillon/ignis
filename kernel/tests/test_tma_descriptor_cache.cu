// GitHub #650 (spec #646): the W4A4 TMA products' descriptor blocks kept on the device
// (kernel/src/tma_descriptor_cache.h) -- OURS.
//
// The two vendored launchers the 27B's prefill runs its W4A4 products through
// (`launch_nvfp4_w4a4_tma_linear`, whose `launch_tma` every W4A4 TMA product shares, and
// `launch_nvfp4_linear_swiglu_w4a4_tma`), at the 27B's [5120, 6144] and [34816, 5120] shapes:
//   - on a stream a cache serves, a product writes the bytes it writes on a stream none serves
//     (upstream's per-launch upload), launch after launch;
//   - with a slot for every block, a block is uploaded once and every later launch of the same
//     block reads it, and no product allocates from the device's memory pool;
//   - with fewer slots than blocks, launched back to back with no synchronize, every launch
//     rewrites a slot an earlier, still queued launch read, and the bytes are still the same: the
//     rewrite is ordered on the stream after its readers (a map rewritten under a kernel still
//     reading it hangs that kernel, which the test's CTest timeout turns into a failure);
//   - a launch inside a graph capture takes upstream's path: captured on a stream a cache serves,
//     it gives the graph it gives on a stream none serves, and the cache gains no block.
// Windows takes the descriptors through a pointer; Linux passes them by value and never asks the
// cache, so there the counts stay zero and the bytes are the same as plainly.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing GPU fails this test.

#include "tma_descriptor_cache.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace nd = ninfer::ops::detail;
using ignis::tma_descriptors::Cache;

int g_failed = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++g_failed;
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

// One W4A4 TMA product's operands: random FP4 codes and finite E4M3 scales.
struct Operand {
  std::uint8_t *codes = nullptr;
  std::uint8_t *scales = nullptr;
};

std::uint64_t g_state = 0x9E3779B97F4A7C15ull;
std::uint8_t next_byte() {
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  return static_cast<std::uint8_t>(g_state >> 24);
}

std::uint8_t *device_bytes(std::size_t bytes, bool scale) {
  std::vector<std::uint8_t> host(bytes);
  for (auto &b : host) {
    // E4M3 0x30..0x3f: 0.5 to just under 1, never the NaN 0x7f.
    b = scale ? static_cast<std::uint8_t>(0x30 | (next_byte() & 0x0f)) : next_byte();
  }
  void *p = nullptr;
  cuda(cudaMalloc(&p, bytes), "cudaMalloc");
  cuda(cudaMemcpy(p, host.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  return static_cast<std::uint8_t *>(p);
}

Operand operand(std::int64_t rows, std::int64_t k) {
  return {device_bytes(static_cast<std::size_t>(rows * k / 2), false),
          device_bytes(static_cast<std::size_t>(rows * k / 16), true)};
}

__nv_bfloat16 *output(std::size_t count) {
  void *p = nullptr;
  cuda(cudaMalloc(&p, count * sizeof(__nv_bfloat16)), "cudaMalloc");
  return static_cast<__nv_bfloat16 *>(p);
}

std::vector<std::uint8_t> host_copy(const void *device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  cuda(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy");
  return host;
}

// A product: one of the two launchers over one activation, writing one output.
struct Product {
  bool swiglu = false;
  Operand activation;
  Operand weight;
  __nv_bfloat16 *out = nullptr;
  std::size_t out_count = 0;
};

constexpr std::int32_t kTokens = 256;
constexpr float kAlpha = 1.0f / 64.0f;

void launch(const Product &p, cudaStream_t stream) {
  if (p.swiglu) {
    nd::launch_nvfp4_linear_swiglu_w4a4_tma(p.activation.codes, p.activation.scales, p.weight.codes,
                                            p.weight.scales, p.out, kTokens, kAlpha, stream);
  } else {
    nd::launch_nvfp4_w4a4_tma_linear(nd::Nvfp4Problem::Residual6144, p.activation.codes,
                                     p.activation.scales, p.weight.codes, p.weight.scales, p.out,
                                     kTokens, kAlpha, stream);
  }
}

// What the products wrote, one buffer each.
std::vector<std::vector<std::uint8_t>> outputs(const std::vector<Product> &products) {
  std::vector<std::vector<std::uint8_t>> out;
  for (const Product &p : products) {
    out.push_back(host_copy(p.out, p.out_count * sizeof(__nv_bfloat16)));
  }
  return out;
}

void clear(const std::vector<Product> &products) {
  for (const Product &p : products) {
    cuda(cudaMemset(p.out, 0, p.out_count * sizeof(__nv_bfloat16)), "cudaMemset");
  }
  cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

// The device's default memory pool's high-water mark of bytes in use, reset by `reset`.
std::uint64_t pool_used_high(bool reset) {
  int device = 0;
  cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaMemPool_t pool = nullptr;
  cuda(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
  // The attribute's value is a cuuint64_t, the driver API's 64-bit unsigned.
  unsigned long long value = 0;
  if (reset) {
    cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &value), "reset UsedMemHigh");
  }
  cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &value), "read UsedMemHigh");
  return value;
}

// The uploads Windows makes for `windows` uploads' worth of misses; Linux never asks the cache.
std::uint64_t uploads_here(std::uint64_t windows) {
  return ignis::tma_descriptors::kThroughPointer ? windows : 0;
}

// `rounds` rounds of every product back to back on `stream`, with no synchronize between them.
void run(const std::vector<Product> &products, int rounds, cudaStream_t stream) {
  for (int r = 0; r < rounds; ++r) {
    for (const Product &p : products) launch(p, stream);
  }
  cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

// The node types of the graph one launch of `p` captures on `stream`, sorted; or the error the
// capture ended with, as a negative entry.
std::vector<int> captured(const Product &p, cudaStream_t stream) {
  cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
  launch(p, stream);
  cudaGraph_t graph = nullptr;
  const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
  if (ended != cudaSuccess) {
    (void)cudaGetLastError();
    return {-static_cast<int>(ended)};
  }
  std::size_t count = 0;
  cuda(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes");
  std::vector<cudaGraphNode_t> nodes(count);
  cuda(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes");
  std::vector<int> types;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    cuda(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
    types.push_back(static_cast<int>(type));
  }
  cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
  std::sort(types.begin(), types.end());
  return types;
}

std::string text(const std::vector<int> &types) {
  std::string out;
  for (int t : types) out += (out.empty() ? "" : ",") + std::to_string(t);
  return out;
}

}  // namespace

int main() {
  cuda(cudaSetDevice(0), "cudaSetDevice");
  cudaStream_t plain = nullptr;
  cudaStream_t served = nullptr;
  cuda(cudaStreamCreate(&plain), "cudaStreamCreate");
  cuda(cudaStreamCreate(&served), "cudaStreamCreate");

  // Three [5120, 6144] products over three activations of one weight: three distinct blocks.
  using Linear = nd::Nvfp4Residual6144Geometry;
  const Operand linear_weight = operand(Linear::kOutputRows, Linear::kInputRows);
  std::vector<Product> linear;
  for (int i = 0; i < 3; ++i) {
    Product p;
    p.activation = operand(kTokens, Linear::kInputRows);
    p.weight = linear_weight;
    p.out_count = static_cast<std::size_t>(Linear::kOutputRows) * kTokens;
    p.out = output(p.out_count);
    linear.push_back(p);
  }
  // Two [34816, 5120] gate and up products, SwiGLU in the epilogue: two distinct blocks.
  using GateUp = nd::Nvfp4MlpGateUpGeometry;
  const Operand gate_up_weight = operand(GateUp::kOutputRows, GateUp::kInputRows);
  std::vector<Product> swiglu;
  for (int i = 0; i < 2; ++i) {
    Product p;
    p.swiglu = true;
    p.activation = operand(kTokens, GateUp::kInputRows);
    p.weight = gate_up_weight;
    p.out_count = static_cast<std::size_t>(GateUp::kOutputRows / 2) * kTokens;
    p.out = output(p.out_count);
    swiglu.push_back(p);
  }

  // Upstream's path: a stream no cache serves.
  clear(linear);
  clear(swiglu);
  run(linear, 1, plain);
  run(swiglu, 1, plain);
  const auto linear_plain = outputs(linear);
  const auto swiglu_plain = outputs(swiglu);

  {
    // A slot for every block: each is uploaded once, every later launch reads it.
    Cache cache(3, served);
    check(cache.bytes() == 3 * ignis::tma_descriptors::kSlotBytes, "a cache of 3 slots holds 3 x 512 bytes");
    clear(linear);
    pool_used_high(true);
    run(linear, 3, served);
    check(outputs(linear) == linear_plain, "[5120, 6144] x3, three rounds, served: the plain bytes");
    check(cache.counts().uploads == uploads_here(3), "three distinct blocks over three rounds: three uploads");
    check(cache.counts().resident == uploads_here(3), "three blocks resident");
    check(pool_used_high(false) == 0, "no product allocates from the device's memory pool");
  }
  {
    // Fewer slots than blocks, back to back: every launch rewrites a slot a queued launch read.
    Cache cache(2, served);
    clear(linear);
    run(linear, 3, served);
    check(outputs(linear) == linear_plain, "[5120, 6144] x3 through 2 slots, back to back: the plain bytes");
    check(cache.counts().uploads == uploads_here(9), "three blocks cycled through two slots: every launch uploads");
    check(cache.counts().resident == uploads_here(2), "two blocks resident");
  }
  {
    Cache cache(2, served);
    clear(swiglu);
    run(swiglu, 3, served);
    check(outputs(swiglu) == swiglu_plain, "[34816, 5120] SwiGLU x2, three rounds, served: the plain bytes");
    check(cache.counts().uploads == uploads_here(2), "two distinct SwiGLU blocks: two uploads");
  }
  {
    Cache cache(1, served);
    clear(swiglu);
    run(swiglu, 3, served);
    check(outputs(swiglu) == swiglu_plain, "[34816, 5120] SwiGLU x2 through 1 slot, back to back: the plain bytes");
    check(cache.counts().uploads == uploads_here(6), "two SwiGLU blocks cycled through one slot: every launch uploads");
  }
  {
    // Inside a capture, upstream's path: the same graph as on a stream no cache serves.
    Cache cache(4, served);
    for (const Product &p : {linear[0], swiglu[0]}) {
      const std::vector<int> plain_graph = captured(p, plain);
      const std::vector<int> served_graph = captured(p, served);
      check(served_graph == plain_graph, std::string(p.swiglu ? "SwiGLU" : "[5120, 6144]") +
                                             " captured on a served stream: upstream's graph (" +
                                             text(served_graph) + " / " + text(plain_graph) + ")");
#ifdef _WIN32
      const bool copies = std::find(plain_graph.begin(), plain_graph.end(),
                                    static_cast<int>(cudaGraphNodeTypeMemcpy)) != plain_graph.end();
      check(copies, "upstream's captured launch copies its descriptors in the graph");
#endif
    }
    check(cache.counts().uploads == 0 && cache.counts().resident == 0, "a capture adds no block to the cache");
  }

  cuda(cudaStreamDestroy(plain), "cudaStreamDestroy");
  cuda(cudaStreamDestroy(served), "cudaStreamDestroy");
  if (g_failed != 0) {
    std::printf("%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
