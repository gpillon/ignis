// Flash-Next's hyper-connection mix and inject at the model's geometry -- OURS, and NOT a CTest
// test (a timing is a finding, not a pass/fail; docs/agents/testing.md). GitHub #306.
//
// The real weight formats (FP8 row-scale mix_down / mix_up, BF16 hc_norm and block_inject) at
// 4 streams of 2560, rank 320. For each row count, `--calls` mixes (and, separately, injects)
// are captured into one CUDA graph, as the decode round replays them; the graph is replayed
// after a warm-up and the median replay is divided by the calls. Decode lanes (1-5, 8) and
// prefill chunks (256, 2048: two 1024-row waves). The calls cycle through `--sets` distinct
// weight sets (~6.6 MB each, 24 by default: more than the L2 holds), so the weights stream from
// DRAM as the round's 97 distinct mixes do. Timing wants the card to itself.
//
//   ignis_kernel_flash_next_hc_bench [--calls 48] [--replays 20] [--sets 24] [--rows N] [--fused 0|1]
//                                    [--fold 0|1]
//
// --rows times one row count only (for a per-kernel profile of one shape). --fused 0 times the
// decode route with its norm in a launch of its own (fn_hc_set_decode_fused; GitHub #306).
// The "+inject" and "+combine" columns time fn_hc_mix_after with the previous sublayer's inject
// pending, without and with the MoE combine (GitHub #306, step 2); --fold 0 runs them unfolded
// (the combine, the inject and the mix one after another: fusion.h's Inject off).

#include "flash_next/fusion.h"
#include "flash_next/hc.h"

#include "ignis_fp8_linear.h"
#include "ignis_moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kStreams = 4;
constexpr int kHidden = 2560;
constexpr int kWidth = kStreams * kHidden;
constexpr int kRank = 320;

uint32_t lcg(uint32_t &state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

void *upload_bytes(const void *host, std::size_t bytes) {
  void *dev = nullptr;
  CUDA_OK(cudaMalloc(&dev, bytes));
  CUDA_OK(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));
  return dev;
}

void *random_bf16(std::size_t n, float scale, uint32_t seed) {
  std::vector<uint16_t> v(n);
  for (auto &x : v) {
    x = bf16_bits((static_cast<float>(lcg(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale);
  }
  return upload_bytes(v.data(), n * 2);
}

// FP8 row-scale (layout.md 6.1): finite codes, BF16 scales at the next multiple of 256 bytes.
void *random_fp8(int32_t rows, int32_t cols, uint32_t seed) {
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scales_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> payload(scales_at + static_cast<std::size_t>(rows) * 2, 0);
  for (std::size_t i = 0; i < codes; ++i) {
    uint8_t code = static_cast<uint8_t>(lcg(seed) >> 24);
    payload[i] = (code & 0x7F) == 0x7F ? static_cast<uint8_t>(code - 1) : code;
  }
  for (int32_t r = 0; r < rows; ++r) {
    const uint16_t bits = bf16_bits(1e-4F + static_cast<float>(lcg(seed) % 100) * 1e-6F);
    std::memcpy(&payload[scales_at + static_cast<std::size_t>(r) * 2], &bits, 2);
  }
  return upload_bytes(payload.data(), payload.size());
}

template <typename F>
double graph_us_per_call(int calls, int replays, cudaStream_t stream, F &&call) {
  cudaGraph_t graph = nullptr;
  CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < calls; ++i) {
    if (!call()) {
      std::fprintf(stderr, "a call failed: %s\n", ignis::flash_next::fn_last_error());
      std::exit(1);
    }
  }
  CUDA_OK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t exec = nullptr;
  CUDA_OK(cudaGraphInstantiate(&exec, graph, 0));
  cudaEvent_t start, stop;
  CUDA_OK(cudaEventCreate(&start));
  CUDA_OK(cudaEventCreate(&stop));
  for (int i = 0; i < 3; ++i) {
    CUDA_OK(cudaGraphLaunch(exec, stream));
  }
  std::vector<float> ms(static_cast<std::size_t>(replays));
  for (auto &m : ms) {
    CUDA_OK(cudaEventRecord(start, stream));
    CUDA_OK(cudaGraphLaunch(exec, stream));
    CUDA_OK(cudaEventRecord(stop, stream));
    CUDA_OK(cudaEventSynchronize(stop));
    CUDA_OK(cudaEventElapsedTime(&m, start, stop));
  }
  std::sort(ms.begin(), ms.end());
  CUDA_OK(cudaEventDestroy(start));
  CUDA_OK(cudaEventDestroy(stop));
  CUDA_OK(cudaGraphExecDestroy(exec));
  CUDA_OK(cudaGraphDestroy(graph));
  return 1e3 * ms[ms.size() / 2] / calls;
}

}  // namespace

int main(int argc, char **argv) {
  int calls = 48;
  int replays = 20;
  int only_rows = 0;
  int sets = 24;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--calls") == 0) {
      calls = std::max(1, std::atoi(argv[i + 1]));
    } else if (std::strcmp(argv[i], "--replays") == 0) {
      replays = std::max(1, std::atoi(argv[i + 1]));
    } else if (std::strcmp(argv[i], "--sets") == 0) {
      sets = std::max(1, std::atoi(argv[i + 1]));
    } else if (std::strcmp(argv[i], "--rows") == 0) {
      only_rows = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--fused") == 0) {
      ignis::flash_next::fn_hc_set_decode_fused(std::atoi(argv[i + 1]) != 0);
    } else if (std::strcmp(argv[i], "--fold") == 0) {
      ignis::flash_next::set_fused(ignis::flash_next::Fusion::Inject, std::atoi(argv[i + 1]) != 0);
    }
  }
  // ignis_moe_prepare prepares the FP8 linear too; the unfolded combine needs the MoE ops'.
  if (ignis_moe_prepare() != 0) {
    std::fprintf(stderr, "ignis_moe_prepare: %s\n", ignis_moe_last_error());
    return 1;
  }
  ignis::flash_next::Geometry g;
  g.streams = kStreams;
  g.hidden = kHidden;
  g.hc_rank = kRank;
  g.rms_norm_eps = 1e-6F;
  std::vector<ignis::flash_next::HcWeights> w(static_cast<std::size_t>(sets));
  for (int i = 0; i < sets; ++i) {
    const auto seed = static_cast<uint32_t>(16 * i);
    auto &set = w[static_cast<std::size_t>(i)];
    set.hc_norm = random_bf16(kWidth, 0.2F, seed + 1);
    set.mix_down = {random_fp8(kRank, kWidth, seed + 2), kRank, kWidth, ignis::flash_next::WeightFormat::Fp8RowScale};
    set.mix_up = {random_fp8(kWidth, kRank, seed + 3), kWidth, kRank, ignis::flash_next::WeightFormat::Fp8RowScale};
    set.block_inject = random_bf16(static_cast<std::size_t>(kStreams) * kWidth, 0.02F, seed + 4);
  }

  cudaStream_t stream = nullptr;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::printf("flash-next HC mix, 4 x 2560 streams, rank 320, FP8 mix_down/mix_up; %d calls per graph over %d weight sets\n",
              calls, sets);
  std::printf("%6s %12s %12s %12s %14s %12s %12s\n", "rows", "mix us", "final us", "inject us", "mix us / row",
              "+inject us", "+combine us");
  for (int rows : {1, 2, 3, 4, 5, 8, 256, 2048}) {
    if (only_rows > 0 && rows != only_rows) {
      continue;
    }
    ninfer::DeviceArena scratch(ignis::flash_next::fn_hc_mix_scratch_bytes(g, rows) + (1u << 20));
    void *hidden = random_bf16(static_cast<std::size_t>(rows) * kWidth, 2.0F, 0x4C0U + rows);
    void *x = nullptr;
    void *y = random_bf16(static_cast<std::size_t>(rows) * kHidden, 1.0F, 0x1E7U + rows);
    float *inj = nullptr;
    CUDA_OK(cudaMalloc(&x, static_cast<std::size_t>(rows) * kHidden * 2));
    CUDA_OK(cudaMalloc(&inj, static_cast<std::size_t>(rows) * kStreams * 4));
    const int n = rows > 8 ? std::max(1, calls / 8) : calls;
    int call = 0;
    const double mix = graph_us_per_call(n, replays, stream, [&] {
      const auto &set = w[static_cast<std::size_t>(call++ % sets)];
      return ignis::flash_next::fn_hc_mix(g, set, hidden, rows, x, inj, scratch, stream) == 0;
    });
    const double fin = graph_us_per_call(n, replays, stream, [&] {
      auto set = w[static_cast<std::size_t>(call++ % sets)];
      set.block_inject = nullptr;
      return ignis::flash_next::fn_hc_mix(g, set, hidden, rows, x, nullptr, scratch, stream) == 0;
    });
    const double inject = graph_us_per_call(n, replays, stream, [&] {
      return ignis::flash_next::fn_hc_inject(g, y, inj, rows, hidden, stream) == 0;
    });
    // The mix with the previous sublayer's inject pending, and with the MoE combine too: its
    // operands at decode shapes (the accumulator is zeroed by the first call; the timing does not
    // depend on its values), the pending injection weights apart from the mix's own.
    double after[2] = {0.0, 0.0};
    if (rows <= 8) {
      std::vector<float> ones(static_cast<std::size_t>(rows) * kStreams, 1.0F);
      auto *inj_prev = static_cast<float *>(upload_bytes(ones.data(), ones.size() * 4));
      void *acc = nullptr;
      CUDA_OK(cudaMalloc(&acc, static_cast<std::size_t>(rows) * kHidden * 8));
      CUDA_OK(cudaMemset(acc, 0, static_cast<std::size_t>(rows) * kHidden * 8));
      std::vector<float> zeros(static_cast<std::size_t>(rows) * kHidden, 0.0F);
      auto *shared = static_cast<float *>(upload_bytes(zeros.data(), zeros.size() * 4));
      void *cx = random_bf16(static_cast<std::size_t>(rows) * kHidden, 1.0F, 0x2E7U + rows);
      void *gate = random_bf16(kHidden, 0.05F, 0x3E7U);
      for (int combine = 0; combine < 2; ++combine) {
        ignis::flash_next::PendingInject pending;
        pending.y = y;
        pending.inj = inj_prev;
        if (combine == 1) {
          pending.acc = static_cast<int64_t *>(acc);
          pending.shared = shared;
          pending.x = cx;
          pending.w_gate = gate;
        }
        after[combine] = graph_us_per_call(n, replays, stream, [&] {
          const auto &set = w[static_cast<std::size_t>(call++ % sets)];
          return ignis::flash_next::fn_hc_mix_after(g, set, pending, hidden, rows, x, inj, scratch, stream) == 0;
        });
      }
      CUDA_OK(cudaFree(inj_prev));
      CUDA_OK(cudaFree(acc));
      CUDA_OK(cudaFree(shared));
      CUDA_OK(cudaFree(cx));
      CUDA_OK(cudaFree(gate));
    }
    std::printf("%6d %12.2f %12.2f %12.2f %14.3f %12.2f %12.2f\n", rows, mix, fin, inject, mix / rows, after[0],
                after[1]);
    CUDA_OK(cudaFree(hidden));
    CUDA_OK(cudaFree(x));
    CUDA_OK(cudaFree(y));
    CUDA_OK(cudaFree(inj));
  }
  CUDA_OK(cudaStreamDestroy(stream));
  return 0;
}
