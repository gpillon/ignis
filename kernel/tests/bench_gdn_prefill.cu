// The GDN prefill on our full-rate kernels against the vendored chunked kernels, per call and per
// prompt -- OURS (GitHub #514, spec #503; kernel/src/gdn_prefill.h). A tool, NOT a CTest: timing
// wants the card to itself (make gpu-status first).
//
// For each chunk width T (--tokens, default 2048,4096,8192) it times, on one stream with CUDA
// events, best of --reps rounds of --inner calls, the 27B's GDN recurrence of one layer (16 q/k
// heads, 48 value heads of 128) both ways, the state in place as the layer body runs it:
//   vendored  ninfer::ops::gated_delta_net with normalize_qk (l2norm, the three chunked kernels,
//             the recurrent kernel over a remainder);
//   ours      ignis::gdn_prefill::chunked (the prepare and the recurrence), its warps per CTA
//             printed.
// Then a prompt's 48 GDN layers at --prompts (default 8192,65536) tokens in chunks of T: the
// whole chunks' calls plus the remainder's, times 48, both ways and their ratio.
//
//   ignis_kernel_gdn_prefill_bench [--tokens 2048,4096,8192] [--prompts 8192,65536] [--reps 5] [--inner 10]

#include "gdn_prefill.h"
#include "moe_fixture.h"

#include "ninfer/ops/gated_delta_net.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

constexpr int32_t kQkHeads = 16, kValueHeads = 48, kDim = 128, kLayers = 48;
const float kScale = 1.0f / std::sqrt(static_cast<float>(kDim));

std::vector<int32_t> int_list(const char *arg) {
  std::vector<int32_t> out;
  for (const char *p = arg; *p != '\0';) {
    out.push_back(static_cast<int32_t>(std::strtol(p, const_cast<char **>(&p), 10)));
    if (*p == ',') ++p;
  }
  return out;
}

struct Buffers {
  DeviceBytes q, k, v, g, beta, out, state, workspace;
  explicit Buffers(int32_t T)
      : q(static_cast<std::size_t>(T) * kQkHeads * kDim * 2), k(q.bytes), v(static_cast<std::size_t>(T) * kValueHeads * kDim * 2),
        g(static_cast<std::size_t>(T) * kValueHeads * 4), beta(g.bytes), out(v.bytes),
        state(static_cast<std::size_t>(kDim) * kDim * kValueHeads * 4),
        workspace(ignis::gdn_prefill::workspace_bytes(kQkHeads, kValueHeads, T)) {
    std::vector<uint16_t> qk(static_cast<std::size_t>(T) * kQkHeads * kDim), vv(static_cast<std::size_t>(T) * kValueHeads * kDim);
    for (std::size_t i = 0; i < qk.size(); ++i) qk[i] = f32_to_bf16(hash_uniform(1, i, 1.0f));
    upload(q, qk);
    for (std::size_t i = 0; i < qk.size(); ++i) qk[i] = f32_to_bf16(hash_uniform(2, i, 1.0f));
    upload(k, qk);
    for (std::size_t i = 0; i < vv.size(); ++i) vv[i] = f32_to_bf16(hash_uniform(3, i, 2.0f));
    upload(v, vv);
    std::vector<float> gg(static_cast<std::size_t>(T) * kValueHeads), bb(gg.size());
    for (std::size_t i = 0; i < gg.size(); ++i) {
      gg[i] = -0.05f - 0.05f * hash_uniform(4, i, 1.0f);
      bb[i] = 0.5f + 0.4f * hash_uniform(5, i, 1.0f);
    }
    upload(g, gg);
    upload(beta, bb);
    MOE_CUDA(cudaMemset(state.p, 0, state.bytes));
  }
};

template <typename F> double best_ms(F &&call, int reps, int inner, cudaStream_t stream) {
  cudaEvent_t a, b;
  MOE_CUDA(cudaEventCreate(&a));
  MOE_CUDA(cudaEventCreate(&b));
  call();
  MOE_CUDA(cudaStreamSynchronize(stream));
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    MOE_CUDA(cudaEventRecord(a, stream));
    for (int i = 0; i < inner; ++i) call();
    MOE_CUDA(cudaEventRecord(b, stream));
    MOE_CUDA(cudaEventSynchronize(b));
    float ms = 0.0f;
    MOE_CUDA(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, static_cast<double>(ms) / inner);
  }
  MOE_CUDA(cudaEventDestroy(a));
  MOE_CUDA(cudaEventDestroy(b));
  return best;
}

struct Timing {
  double vendored = 0.0, ours = 0.0;
};

Timing time_call(int32_t T, int reps, int inner, ninfer::DeviceArena &arena, cudaStream_t stream) {
  Buffers b(T);
  ninfer::Tensor q(b.q.p, ninfer::DType::BF16, {kDim, kQkHeads, T, 1});
  ninfer::Tensor k(b.k.p, ninfer::DType::BF16, {kDim, kQkHeads, T, 1});
  ninfer::Tensor v(b.v.p, ninfer::DType::BF16, {kDim, kValueHeads, T, 1});
  ninfer::Tensor g(b.g.p, ninfer::DType::FP32, {kValueHeads, T, 1, 1});
  ninfer::Tensor beta(b.beta.p, ninfer::DType::FP32, {kValueHeads, T, 1, 1});
  ninfer::Tensor out(b.out.p, ninfer::DType::BF16, {kDim, kValueHeads, T, 1});
  ninfer::Tensor state(b.state.p, ninfer::DType::FP32, {kDim, kDim, kValueHeads, 1});
  Timing t;
  t.vendored = best_ms(
      [&] { ninfer::ops::gated_delta_net(q, k, v, g, beta, kScale, /*normalize_qk=*/true, arena, state, state, out, stream); },
      reps, inner, stream);
  t.ours = best_ms(
      [&] {
        ignis::gdn_prefill::chunked(q, k, v, g, beta, kScale, state, state, out, b.workspace.p, b.workspace.bytes, stream);
      },
      reps, inner, stream);
  return t;
}

} // namespace

int main(int argc, char **argv) {
  std::vector<int32_t> tokens = {2048, 4096, 8192}, prompts = {8192, 65536};
  int reps = 5, inner = 10;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i];
    if (flag == "--tokens") tokens = int_list(argv[i + 1]);
    else if (flag == "--prompts") prompts = int_list(argv[i + 1]);
    else if (flag == "--reps") reps = std::atoi(argv[i + 1]);
    else if (flag == "--inner") inner = std::atoi(argv[i + 1]);
  }
  int device = 0;
  MOE_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  MOE_CUDA(cudaGetDeviceProperties(&prop, device));
  cudaStream_t stream = nullptr;
  MOE_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const int32_t widest = *std::max_element(tokens.begin(), tokens.end());
  ninfer::DeviceArena arena(
      ninfer::ops::gated_delta_net_workspace_capacity_bytes(kQkHeads, kValueHeads, /*normalize_qk=*/true, 1, widest));
  std::printf("gdn_prefill bench: %s, sm_%d%d, %d SMs; ours runs %d warps per CTA\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, ignis::gdn_prefill::recurrence_warps());
  std::printf("%8s %14s %14s %8s\n", "T", "vendored ms", "ours ms", "ratio");
  std::map<int32_t, Timing> per_call;
  const auto timing = [&](int32_t T) -> Timing {
    auto it = per_call.find(T);
    if (it == per_call.end()) it = per_call.emplace(T, time_call(T, reps, inner, arena, stream)).first;
    return it->second;
  };
  for (const int32_t T : tokens) {
    const Timing t = timing(T);
    std::printf("%8d %14.4f %14.4f %8.2f\n", T, t.vendored, t.ours, t.vendored / t.ours);
  }
  std::printf("\n48 GDN layers per prompt (chunks of T, the remainder last):\n");
  std::printf("%8s %8s %14s %14s %8s\n", "prompt", "T", "vendored ms", "ours ms", "ratio");
  for (const int32_t P : prompts) {
    for (const int32_t T : tokens) {
      const int32_t whole = P / T, rest = P % T;
      Timing sum;
      const Timing w = timing(T);
      sum.vendored = whole * w.vendored;
      sum.ours = whole * w.ours;
      if (rest > 0) {
        const Timing r = timing(rest);
        sum.vendored += r.vendored;
        // The layer body sends a remainder under kMinTokens to the vendored op either way.
        sum.ours += rest < ignis::gdn_prefill::kMinTokens ? r.vendored : r.ours;
      }
      std::printf("%8d %8d %14.3f %14.3f %8.2f\n", P, T, kLayers * sum.vendored, kLayers * sum.ours,
                  sum.vendored / sum.ours);
    }
  }
  MOE_CUDA(cudaStreamDestroy(stream));
  return EXIT_SUCCESS;
}
