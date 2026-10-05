// GitHub #302 (spec flash-next/04): Flash-Next's token embedding and head
// (kernel/src/flash_next/embed_head.cu) -- OURS, not vendored.
//
// - The embedding, FP8 row-scale and BF16, bit for bit: each element is
//   bf16(e4m3(code) * scale) (the converter's quantized-reference decode,
//   layout.md 6.1) or the stored BF16, the same in every stream.
// - The head is the final mixer then lm_head: bit for bit the two steps run
//   apart (fn_hc_mix, fn_linear), so its wiring of rows and buffers is right;
//   both steps have their own numeric tests.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "flash_next/embed_head.h"
#include "flash_next/hc.h"

#include "ignis_fp8_linear.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

void cuda_ok(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

uint32_t lcg(uint32_t &state) {
  state = state * 1664525U + 1013904223U;
  return state;
}

uint16_t bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  const uint32_t rounding = 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

float bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float v;
  std::memcpy(&v, &u, 4);
  return v;
}

template <typename T>
T *upload(const std::vector<T> &host) {
  T *dev = nullptr;
  cuda_ok(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
  cuda_ok(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return dev;
}

constexpr int32_t kVocab = 1000, kHidden = 2560, kStreams = 4, kRank = 320;

ignis::flash_next::Geometry geometry() {
  ignis::flash_next::Geometry g;
  g.vocab = kVocab;
  g.hidden = kHidden;
  g.streams = kStreams;
  g.hc_rank = kRank;
  g.rms_norm_eps = 1e-6F;
  return g;
}

void embed_case(bool fp8) {
  const std::string label = fp8 ? "FP8 embedding" : "BF16 embedding";
  const auto g = geometry();
  uint32_t seed = fp8 ? 11U : 12U;
  const std::size_t codes = static_cast<std::size_t>(kVocab) * kHidden;
  const std::size_t scales_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> payload(fp8 ? scales_at + kVocab * 2 : codes * 2);
  std::vector<uint16_t> expect_row(codes);  // the decoded table, as BF16 bits
  if (fp8) {
    for (int32_t r = 0; r < kVocab; ++r) {
      const uint16_t scale = bf16_bits(0.001F * static_cast<float>(1 + lcg(seed) % 50));
      std::memcpy(&payload[scales_at + static_cast<std::size_t>(r) * 2], &scale, 2);
    }
    for (std::size_t i = 0; i < codes; ++i) {
      uint8_t code = static_cast<uint8_t>(lcg(seed) >> 24);
      if ((code & 0x7F) == 0x7F) code = static_cast<uint8_t>(code - 1);
      payload[i] = code;
      __nv_fp8_e4m3 e;
      e.__x = code;
      uint16_t scale;
      std::memcpy(&scale, &payload[scales_at + (i / kHidden) * 2], 2);
      expect_row[i] = bf16_bits(static_cast<float>(e) * bf16_value(scale));
    }
  } else {
    for (std::size_t i = 0; i < codes; ++i) {
      expect_row[i] = bf16_bits(static_cast<float>(static_cast<int32_t>(lcg(seed) >> 20) - 2048) / 4096.0F);
    }
    std::memcpy(payload.data(), expect_row.data(), codes * 2);
  }
  const std::vector<int32_t> ids = {0, kVocab - 1, 123, 123, 517};
  auto *d_weight = upload(payload);
  auto *d_ids = upload(ids);
  void *d_residual = nullptr;
  cuda_ok(cudaMalloc(&d_residual, ids.size() * kStreams * kHidden * 2), "cudaMalloc residual");
  const ignis::flash_next::Linear embed{d_weight, kVocab, kHidden,
                                        fp8 ? ignis::flash_next::WeightFormat::Fp8RowScale
                                            : ignis::flash_next::WeightFormat::Bf16};
  check(ignis::flash_next::fn_embed(g, embed, d_ids, static_cast<int32_t>(ids.size()), d_residual, nullptr) == 0,
        label + ": runs: " + ignis::flash_next::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> residual(ids.size() * kStreams * kHidden);
  cuda_ok(cudaMemcpy(residual.data(), d_residual, residual.size() * 2, cudaMemcpyDeviceToHost), "download");
  int mismatched = 0;
  for (std::size_t t = 0; t < ids.size(); ++t) {
    for (int32_t s = 0; s < kStreams; ++s) {
      for (int32_t h = 0; h < kHidden; ++h) {
        mismatched += residual[(t * kStreams + s) * kHidden + h] !=
                              expect_row[static_cast<std::size_t>(ids[t]) * kHidden + h]
                          ? 1
                          : 0;
      }
    }
  }
  check(mismatched == 0, label + ": " + std::to_string(mismatched) + " elements differ from the decode");
  cudaFree(d_weight);
  cudaFree(d_ids);
  cudaFree(d_residual);
}

void head_case() {
  const auto g = geometry();
  uint32_t seed = 21U;
  auto random = [&](std::size_t n, float scale) {
    std::vector<uint16_t> v(n);
    for (auto &x : v) x = bf16_bits((static_cast<float>(lcg(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale);
    return v;
  };
  const int32_t rows = 3;
  ignis::flash_next::HcWeights mixer;
  mixer.hc_norm = upload(random(kStreams * kHidden, 0.2F));
  mixer.mix_down = {upload(random(static_cast<std::size_t>(kRank) * kStreams * kHidden, 0.02F)), kRank,
                    kStreams * kHidden, ignis::flash_next::WeightFormat::Bf16};
  mixer.mix_up = {upload(random(static_cast<std::size_t>(kStreams) * kHidden * kRank, 0.1F)),
                  kStreams * kHidden, kRank, ignis::flash_next::WeightFormat::Bf16};
  const ignis::flash_next::Linear head{upload(random(static_cast<std::size_t>(kVocab) * kHidden, 0.05F)),
                                       kVocab, kHidden, ignis::flash_next::WeightFormat::Bf16};
  auto *d_residual = upload(random(static_cast<std::size_t>(rows) * kStreams * kHidden, 2.0F));
  void *d_logits = nullptr;
  void *d_x = nullptr;
  void *d_apart = nullptr;
  cuda_ok(cudaMalloc(&d_logits, static_cast<std::size_t>(rows) * kVocab * 2), "cudaMalloc logits");
  cuda_ok(cudaMalloc(&d_x, static_cast<std::size_t>(rows) * kHidden * 2), "cudaMalloc x");
  cuda_ok(cudaMalloc(&d_apart, static_cast<std::size_t>(rows) * kVocab * 2), "cudaMalloc apart");
  ninfer::DeviceArena scratch(64u << 20);
  check(ignis::flash_next::fn_head(g, mixer, head, d_residual, rows, d_logits, scratch, nullptr) == 0,
        std::string("head runs: ") + ignis::flash_next::fn_last_error());
  check(ignis::flash_next::fn_hc_mix(g, mixer, d_residual, rows, d_x, nullptr, scratch, nullptr) == 0 &&
            ignis::flash_next::fn_linear(head, d_x, rows, d_apart, false, scratch, nullptr) == 0,
        "the two steps run apart");
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> logits(static_cast<std::size_t>(rows) * kVocab), apart(logits.size());
  cuda_ok(cudaMemcpy(logits.data(), d_logits, logits.size() * 2, cudaMemcpyDeviceToHost), "download");
  cuda_ok(cudaMemcpy(apart.data(), d_apart, apart.size() * 2, cudaMemcpyDeviceToHost), "download");
  check(logits == apart, "the head is the final mixer then lm_head, bit for bit");
}

}  // namespace

int main() {
  int devices = 0;
  cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  if (ignis_fp8_linear_prepare() != 0) {
    std::fprintf(stderr, "FATAL: ignis_fp8_linear_prepare: %s\n", ignis_fp8_linear_last_error());
    return 1;
  }
  embed_case(true);
  embed_case(false);
  head_case();
  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next embed/head test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next embed/head test: ok\n");
  return 0;
}
