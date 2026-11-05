// GitHub #335 (ADR 0061, spec 27b-16gb/03): one prefill chunk's hq-e8-2b attention at a long
// history, A1's own Prompt route against ignis's prompt attention without prompt planes -- OURS,
// and NOT a CTest test (a timing is a finding, not a pass/fail; docs/agents/testing.md).
//
// It builds one sequence through the production pool and view builders, appends a history of
// --ctx minus --width keys through A2 (`gqa_kv_append`, pieces of 8,192; Gaussian rows, which is
// what the codec is calibrated on), and times the chunk's attention call on both routes over the
// same cache, the residual window on as the engine serves it (or off with --codec-only).
//
// Output, per route: the median call time over --iters launches, and the cost per key and query
// token of one GQA layer and of the 27B's 16 (the unit of the cold-prefill finding,
// docs/findings/2026-10-13-cold-prefill-27b-vs-flash-next.md).
//
//   ignis_hq_prompt_bench --ctx 46080 [--width 1024] [--iters 10] [--codec-only] [--band KEYS]
//                         [--splits S] [--ignis-only] [--fp16 | --int8]
//   ignis_hq_prompt_bench --prompt 65536 [--width 4096] [--iters 5] [--bands fp16,fp8,int8]
//
// --fp16 runs the prompt attention's FP16-accumulate fallback band instead of the 8-bit one,
// --int8 the integer band (GitHub #510).
//
// --prompt (GitHub #510) times a whole prompt instead of one chunk: each band in --bands (by
// default every band the card runs; fp8 needs compute capability 12.0) prefills the prompt from
// position 0 at chunk --width through the op, which appends each chunk after it attends, and each
// call is timed from its start to the end of its attention and merge (the append and the gate,
// which are no attention, left out). The bands alternate, a walk each, for --iters rounds; the
// table gives each chunk's median and the prompt's total, for one GQA layer and for the 27B's 16.

#include "ignis_gqa_workspace.h"
#include "ignis_hq_prompt_attention.h"
#include "ignis_seq.h"
#include "ignis_seq_internal.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

#define CUDA_FATAL(expr)                                                                        \
  do {                                                                                          \
    const cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "FATAL: %s failed: %s\n", #expr, cudaGetErrorString(_err));          \
      std::exit(EXIT_FAILURE);                                                                  \
    }                                                                                           \
  } while (0)

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads = 24;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kGqaLayers = 16;

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  ~DeviceBytes() { cudaFree(p); }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

// N(0, 1) rows from a counter hash, on the device: a whole prompt's rows at no host cost.
__global__ void gaussian_rows_kernel(__nv_bfloat16 *dst, std::int64_t n, std::uint64_t seed) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  auto mix = [](std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  };
  const std::uint64_t r = mix(seed * 0x100000001B3ull ^ static_cast<std::uint64_t>(i));
  const float u1 = (static_cast<float>(r >> 40) + 0.5F) * (1.0F / 16777216.0F);
  const float u2 = static_cast<float>(mix(r) >> 40) * (1.0F / 16777216.0F);
  dst[i] = __float2bfloat16(sqrtf(-2.0F * logf(u1)) * cospif(2.0F * u2));
}

std::vector<std::uint16_t> gaussian_bf16(std::size_t n, std::mt19937 &rng) {
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> out(n);
  for (auto &bits : out) {
    const __nv_bfloat16 v = __float2bfloat16(normal(rng));
    std::memcpy(&bits, &v, sizeof(bits));
  }
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// --prompt: every band in `bands` walks the whole prompt, `rounds` times in turn.
int bench_prompt(std::int32_t prompt, std::int32_t width, std::int32_t rounds, const std::string &bands) {
  std::vector<std::pair<std::string, IgnisHqPromptPrecision>> runs;
  int device = 0, major = 0;
  CUDA_FATAL(cudaGetDevice(&device));
  CUDA_FATAL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  const std::string list = bands.empty() ? (major >= 12 ? "fp16,fp8,int8" : "fp16,int8") : bands;
  for (std::size_t from = 0; from <= list.size();) {
    const std::size_t to = std::min(list.find(',', from), list.size());
    const std::string name = list.substr(from, to - from);
    if (name == "fp16") runs.emplace_back(name, IgnisHqPromptPrecision::kFp16);
    else if (name == "fp8") runs.emplace_back(name, IgnisHqPromptPrecision::kFp8);
    else if (name == "int8") runs.emplace_back(name, IgnisHqPromptPrecision::kInt8);
    else {
      std::fprintf(stderr, "unknown band %s (fp16, fp8, int8)\n", name.c_str());
      return 2;
    }
    from = to + 1;
  }
  if (width < 17 || prompt < width || prompt % width != 0 || rounds < 1) {
    std::fprintf(stderr, "need --width >= 17 dividing --prompt, --iters >= 1\n");
    return 2;
  }
  const std::int32_t chunks = prompt / width;
  constexpr auto page = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = kKvHeads;
  spec.head_dim = kHeadDim;
  spec.kv_format = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = (static_cast<std::uint32_t>(prompt) + page - 1) / page;
  spec.max_context_tokens = static_cast<std::uint32_t>(prompt);
  spec.slot_count = 1;
  spec.gdn_num_layers = 2;
  spec.kv_num_layers = kIgnisGqaLayerCount;
  spec.gdn_conv_channels = 6;
  spec.gdn_value_heads = 2;
  spec.gdn_head_dim = 4;
  spec.vocab = 32;
  ignis_seq_pool *pool = nullptr;
  ignis_seq *seq = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0 || ignis_seq_alloc(pool, spec.max_context_tokens, &seq) != 0) {
    std::fprintf(stderr, "FATAL: pool or sequence allocation failed\n");
    return 1;
  }
  const ninfer::PagedKVLayerView layer = ignis_kv_layer_view(pool, seq, 0);
  const auto q_elems = static_cast<std::size_t>(kHeadDim) * kQHeads * width;
  const auto kv_elems = static_cast<std::size_t>(kHeadDim) * kKvHeads * width;
  DeviceBytes qd(q_elems * 2), kd(kv_elems * 2), vd(kv_elems * 2), gd(q_elems * 2), od(q_elems * 2),
      pd(static_cast<std::size_t>(width) * 4);
  CUDA_FATAL(cudaMemset(gd.p, 0, q_elems * 2));
  ninfer::DeviceArena workspace(ignis_hq_prompt_workspace_bytes(width, static_cast<std::uint32_t>(prompt)));
  cudaEvent_t e0, e1;
  CUDA_FATAL(cudaEventCreate(&e0));
  CUDA_FATAL(cudaEventCreate(&e1));
  const float scale = 1.0F / 16.0F;
  auto fill = [](void *dst, std::size_t n, std::uint64_t seed) {
    gaussian_rows_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(static_cast<__nv_bfloat16 *>(dst),
                                                                         static_cast<std::int64_t>(n), seed);
    CUDA_FATAL(cudaGetLastError());
  };
  // ms[band][chunk][round]
  std::vector<std::vector<std::vector<double>>> ms(runs.size(), std::vector<std::vector<double>>(chunks));
  std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
  for (std::int32_t round = 0; round < rounds; ++round) {
    for (std::size_t b = 0; b < runs.size(); ++b) {
      for (std::int32_t c = 0; c < chunks; ++c) {
        const std::int32_t start = c * width;
        fill(qd.p, q_elems, 3u * c + 1);
        fill(kd.p, kv_elems, 3u * c + 2);
        fill(vd.p, kv_elems, 3u * c + 3);
        for (std::int32_t t = 0; t < width; ++t) positions[static_cast<std::size_t>(t)] = start + t;
        CUDA_FATAL(cudaMemcpy(pd.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
        const ninfer::Tensor q(qd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
        const ninfer::Tensor k(kd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, width, 1});
        const ninfer::Tensor v(vd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, width, 1});
        const ninfer::Tensor gate(gd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
        ninfer::Tensor out(od.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
        const ninfer::Tensor pos(pd.p, ninfer::DType::I32, {width, 1, 1, 1});
        IgnisHqPromptOptions options;
        options.precision = runs[b].second;
        options.after_attention = [&](const IgnisHqPromptKeys &) { CUDA_FATAL(cudaEventRecord(e1, nullptr)); };
        CUDA_FATAL(cudaDeviceSynchronize());
        CUDA_FATAL(cudaEventRecord(e0, nullptr));
        ignis_hq_prompt_attention(q, k, v, pos, start, gate, scale, layer, workspace, out, nullptr, options);
        CUDA_FATAL(cudaEventSynchronize(e1));
        float t = 0.0F;
        CUDA_FATAL(cudaEventElapsedTime(&t, e0, e1));
        ms[b][static_cast<std::size_t>(c)].push_back(t);
        CUDA_FATAL(cudaDeviceSynchronize());
      }
    }
  }
  std::printf("prompt %d at chunk %d, %d round(s): ms a call of one layer (median), prep to merge\n", prompt,
              width, rounds);
  std::printf("%6s %8s", "chunk", "start");
  for (const auto &r : runs) std::printf(" %10s", r.first.c_str());
  std::printf("\n");
  std::vector<double> total(runs.size(), 0.0);
  for (std::int32_t c = 0; c < chunks; ++c) {
    std::printf("%6d %8d", c, c * width);
    for (std::size_t b = 0; b < runs.size(); ++b) {
      const double t = median(ms[b][static_cast<std::size_t>(c)]);
      total[b] += t;
      std::printf(" %10.3f", t);
    }
    std::printf("\n");
  }
  for (std::size_t b = 0; b < runs.size(); ++b) {
    std::printf("%-5s prompt %d chunk %d: %.4f s for one layer, %.4f s for 16 layers", runs[b].first.c_str(),
                prompt, width, total[b] / 1000.0, kGqaLayers * total[b] / 1000.0);
    if (runs[0].first == "fp16") std::printf(" (%.2fx fp16)", total[0] / total[b]);
    std::printf("\n");
  }
  ignis_seq_release(pool, seq);
  ignis_seq_pool_free(pool);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  std::int32_t ctx = 46080, width = 1024, iters = 10, splits = 0, prompt = 0;
  std::uint32_t band = kIgnisHqPromptBandKeys;
  bool codec_only = false, ignis_only = false, fp16 = false, int8 = false;
  bool width_named = false, iters_named = false;
  std::string bands;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::atoi(argv[++i]) : 0; };
    if (a == "--ctx") ctx = next();
    else if (a == "--width") { width = next(); width_named = true; }
    else if (a == "--iters") { iters = next(); iters_named = true; }
    else if (a == "--band") band = static_cast<std::uint32_t>(next());
    else if (a == "--splits") splits = next();
    else if (a == "--codec-only") codec_only = true;
    else if (a == "--ignis-only") ignis_only = true;
    else if (a == "--fp16") fp16 = true;
    else if (a == "--int8") int8 = true;
    else if (a == "--prompt") prompt = next();
    else if (a == "--bands") bands = i + 1 < argc ? argv[++i] : "";
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (prompt > 0) return bench_prompt(prompt, width_named ? width : 4096, iters_named ? iters : 5, bands);
  if (fp16 && int8) {
    std::fprintf(stderr, "--fp16 and --int8 name two bands\n");
    return 2;
  }
  if (width < 17 || ctx < width || iters < 1) {
    std::fprintf(stderr, "need --width >= 17, --ctx >= --width, --iters >= 1\n");
    return 2;
  }
  const std::int32_t start = ctx - width;

  constexpr auto page = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = kKvHeads;
  spec.head_dim = kHeadDim;
  spec.kv_format = IGNIS_KV_FORMAT_HQ_E8_2B;
  spec.kv_page_group_count = (static_cast<std::uint32_t>(ctx) + page - 1) / page;
  spec.max_context_tokens = static_cast<std::uint32_t>(ctx);
  spec.slot_count = 1;
  spec.gdn_num_layers = 2;
  spec.kv_num_layers = kIgnisGqaLayerCount;
  spec.gdn_conv_channels = 6;
  spec.gdn_value_heads = 2;
  spec.gdn_head_dim = 4;
  spec.vocab = 32;
  ignis_seq_pool *pool = nullptr;
  ignis_seq *seq = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0 || ignis_seq_alloc(pool, spec.max_context_tokens, &seq) != 0) {
    std::fprintf(stderr, "FATAL: pool or sequence allocation failed\n");
    return 1;
  }
  ninfer::PagedKVLayerView layer = ignis_kv_layer_view(pool, seq, 0);
  if (codec_only) {
    layer.residual_k = {};
    layer.residual_v = {};
    layer.ring_valid = {};
  }

  std::mt19937 rng(335);
  constexpr std::int32_t kPiece = 8192;
  for (std::int32_t at = 0; at < start; at += kPiece) {
    const std::int32_t tokens = std::min(kPiece, start - at);
    const auto rows = static_cast<std::size_t>(kHeadDim) * kKvHeads * tokens;
    const std::vector<std::uint16_t> hk = gaussian_bf16(rows, rng), hv = gaussian_bf16(rows, rng);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t t = 0; t < tokens; ++t) positions[static_cast<std::size_t>(t)] = at + t;
    DeviceBytes kd(rows * 2), vd(rows * 2), pd(positions.size() * 4);
    CUDA_FATAL(cudaMemcpy(kd.p, hk.data(), rows * 2, cudaMemcpyHostToDevice));
    CUDA_FATAL(cudaMemcpy(vd.p, hv.data(), rows * 2, cudaMemcpyHostToDevice));
    CUDA_FATAL(cudaMemcpy(pd.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
    ninfer::ops::gqa_kv_append(ninfer::Tensor(kd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, tokens, 1}),
                               ninfer::Tensor(vd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, tokens, 1}),
                               ninfer::Tensor(pd.p, ninfer::DType::I32, {tokens, 1, 1, 1}), layer,
                               nullptr);
    CUDA_FATAL(cudaStreamSynchronize(nullptr));
  }

  const auto q_elems = static_cast<std::size_t>(kHeadDim) * kQHeads * width;
  const auto kv_elems = static_cast<std::size_t>(kHeadDim) * kKvHeads * width;
  const std::vector<std::uint16_t> hq = gaussian_bf16(q_elems, rng), hk = gaussian_bf16(kv_elems, rng),
                                   hv = gaussian_bf16(kv_elems, rng);
  std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
  for (std::int32_t t = 0; t < width; ++t) positions[static_cast<std::size_t>(t)] = start + t;
  DeviceBytes qd(q_elems * 2), kd(kv_elems * 2), vd(kv_elems * 2), gd(q_elems * 2), od(q_elems * 2),
      pd(positions.size() * 4), rowsd(4);
  CUDA_FATAL(cudaMemcpy(qd.p, hq.data(), q_elems * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(kd.p, hk.data(), kv_elems * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(vd.p, hv.data(), kv_elems * 2, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemset(gd.p, 0, q_elems * 2));
  CUDA_FATAL(cudaMemcpy(pd.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemset(rowsd.p, 0, 4));
  const ninfer::Tensor q(qd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
  const ninfer::Tensor k(kd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, width, 1});
  const ninfer::Tensor v(vd.p, ninfer::DType::BF16, {kHeadDim, kKvHeads, width, 1});
  const ninfer::Tensor gate(gd.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
  ninfer::Tensor out(od.p, ninfer::DType::BF16, {kHeadDim, kQHeads, width, 1});
  const ninfer::Tensor pos(pd.p, ninfer::DType::I32, {width, 1, 1, 1});
  const float scale = 1.0F / 16.0F;

  // A1's batched view of the one sequence, as run_gqa_layer builds it.
  ninfer::PagedKVBatchLayerView batch;
  ignis_kv_fill_layer_planes(batch, pool, 0);
  if (!codec_only) ignis_kv_fill_residual(batch, pool, 0, seq->slot, 1);
  const ninfer::Tensor table = seq->kv.block_table();
  batch.block_tables = table.view({table.ne[0], 1});
  const ninfer::Tensor rows(rowsd.p, ninfer::DType::I32, {1, 1, 1, 1});
  const ninfer::ops::GqaExecutionEnvelope envelope{1, static_cast<std::uint32_t>(ctx)};

  IgnisHqPromptOptions options;
  options.band_keys = band;
  options.splits = splits;
  if (fp16) options.precision = IgnisHqPromptPrecision::kFp16;
  if (int8) options.precision = IgnisHqPromptPrecision::kInt8;
  const double pairs = static_cast<double>(width) * start + 0.5 * width * (width + 1.0);
  cudaEvent_t e0, e1;
  CUDA_FATAL(cudaEventCreate(&e0));
  CUDA_FATAL(cudaEventCreate(&e1));
  auto time = [&](const char *name, std::size_t workspace_bytes, auto &&call) {
    ninfer::DeviceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    std::vector<double> us;
    for (int i = 0; i < iters + 2; ++i) {
      CUDA_FATAL(cudaEventRecord(e0, nullptr));
      call(workspace);
      CUDA_FATAL(cudaEventRecord(e1, nullptr));
      CUDA_FATAL(cudaEventSynchronize(e1));
      float ms = 0.0F;
      CUDA_FATAL(cudaEventElapsedTime(&ms, e0, e1));
      if (i >= 2) us.push_back(1000.0 * ms);
    }
    const double t = median(us);
    std::printf("%-8s ctx %7d width %4d%s: %10.1f us per call, %.4f ns per key-token per layer, "
                "%.3f ns for 16 layers, workspace %.1f MiB\n",
                name, ctx, width, codec_only ? " codec only" : "", t, 1000.0 * t / pairs,
                kGqaLayers * 1000.0 * t / pairs, workspace_bytes / 1048576.0);
  };
  if (!ignis_only) {
    time("A1", ignis_gqa_a1_workspace_bytes(ninfer::DType::U8, envelope, width),
         [&](ninfer::DeviceArena &ws) {
           ninfer::ops::gqa_attention(q, k, v, pos, ninfer::Tensor{}, rows, gate, scale, batch, envelope, ws,
                                      out, nullptr);
         });
  }
  time(fp16 ? "ignis16" : int8 ? "ignis_i8" : "ignis", ignis_hq_prompt_workspace_bytes(width, static_cast<std::uint32_t>(ctx), band),
       [&](ninfer::DeviceArena &ws) {
         ignis_hq_prompt_attention(q, k, v, pos, start, gate, scale, layer, ws, out, nullptr, options);
       });
  CUDA_FATAL(cudaDeviceSynchronize());
  ignis_seq_release(pool, seq);
  ignis_seq_pool_free(pool);
  return 0;
}
