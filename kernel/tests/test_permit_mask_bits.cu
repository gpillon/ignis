// The permitted set's second form -- OURS, not vendored (GitHub #586, spec
// #287, kernel/src/permitted_tokens.h): `ignis_permit_mask_bits`, the mask
// over the whole vocabulary a **repair** restricts a position to.
//
// The kernel's whole contract is bit-exact, so the oracle is the host walking
// the same bits: a column whose bit is set comes back with its BF16 bits
// untouched, a column whose bit is clear comes back as the short form's
// masked value (`-1e30` in BF16), and nothing past the row is written. No
// tolerance anywhere -- a mask is never approximately applied.
//
// What the arms are for:
//
//   production   the 27B's 248,320 columns (3,880 whole words) under a random
//                mask with whole words clear and whole words set among them,
//                the shape every repair runs at.
//   ragged       a vocabulary that ends mid-word, its last word carrying set
//                bits past the row: the kernel must read only the row's own
//                bits, and must not write past the row (a guard column after
//                it is checked).
//   extremes     one column permitted -- the first, a word's last bit, the
//                next word's first and the row's last -- and every column
//                permitted, which leaves the row exactly as it was.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "permitted_tokens.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

struct DeviceBytes {
  void *p = nullptr;
  explicit DeviceBytes(std::size_t bytes) {
    if (bytes > 0) { CUDA_FATAL(cudaMalloc(&p, bytes)); }
  }
  ~DeviceBytes() {
    if (p != nullptr) { cudaFree(p); }
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
};

std::uint16_t bf16_bits(float value) {
  const __nv_bfloat16 b = __float2bfloat16(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &b, sizeof(bits));
  return bits;
}

bool permitted(const std::vector<std::uint64_t> &mask, std::int32_t v) {
  return ((mask[static_cast<std::size_t>(v) / 64] >> (v % 64)) & 1ull) != 0;
}

// Runs the kernel over `row` under `mask` and checks every column, plus a
// guard column right after the row that the kernel must leave alone.
void run_arm(const char *name, std::vector<std::uint16_t> row, const std::vector<std::uint64_t> &mask) {
  const auto vocab = static_cast<std::int32_t>(row.size());
  const std::uint16_t guard = bf16_bits(12345.0f);
  std::vector<std::uint16_t> host(row);
  host.push_back(guard);
  DeviceBytes d_row(host.size() * sizeof(std::uint16_t));
  DeviceBytes d_mask(mask.size() * sizeof(std::uint64_t));
  CUDA_FATAL(cudaMemcpy(d_row.p, host.data(), host.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  CUDA_FATAL(cudaMemcpy(d_mask.p, mask.data(), mask.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
  check(ignis_permit_mask_bits(d_row.p, vocab, static_cast<const std::uint64_t *>(d_mask.p), nullptr) == 0,
        std::string(name) + ": launch");
  CUDA_FATAL(cudaDeviceSynchronize());
  std::vector<std::uint16_t> out(host.size());
  CUDA_FATAL(cudaMemcpy(out.data(), d_row.p, out.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

  const std::uint16_t masked = bf16_bits(-1.0e30f);
  std::int32_t wrong = 0;
  std::int32_t first_wrong = -1;
  for (std::int32_t v = 0; v < vocab; ++v) {
    const std::uint16_t want = permitted(mask, v) ? row[static_cast<std::size_t>(v)] : masked;
    if (out[static_cast<std::size_t>(v)] != want) {
      if (first_wrong < 0) { first_wrong = v; }
      ++wrong;
    }
  }
  check(wrong == 0, std::string(name) + ": " + std::to_string(wrong) + " columns wrong, the first at " +
                        std::to_string(first_wrong));
  check(out.back() == guard, std::string(name) + ": the column past the row was written");
  std::printf("  %s: %d columns, %s\n", name, vocab, wrong == 0 ? "exact" : "WRONG");
}

std::vector<std::uint16_t> random_row(std::int32_t vocab, std::mt19937 &rng) {
  std::normal_distribution<float> logit(0.0f, 4.0f);
  std::vector<std::uint16_t> row(static_cast<std::size_t>(vocab));
  for (auto &bits : row) { bits = bf16_bits(logit(rng)); }
  return row;
}

std::vector<std::uint64_t> empty_mask(std::int32_t vocab) {
  return std::vector<std::uint64_t>((static_cast<std::size_t>(vocab) + 63) / 64, 0ull);
}

std::vector<std::uint64_t> one_column(std::int32_t vocab, std::int32_t v) {
  auto mask = empty_mask(vocab);
  mask[static_cast<std::size_t>(v) / 64] |= 1ull << (v % 64);
  return mask;
}

}  // namespace

int main() {
  std::mt19937 rng(0x586u);
  std::uniform_int_distribution<std::uint64_t> bits;

  // production: the 27B's vocabulary, a random mask with whole words of
  // either kind in it.
  {
    constexpr std::int32_t kVocab = 248320;
    auto mask = empty_mask(kVocab);
    for (std::size_t w = 0; w < mask.size(); ++w) {
      mask[w] = (w % 17 == 0) ? 0ull : (w % 23 == 0) ? ~0ull : bits(rng);
    }
    run_arm("production 248320", random_row(kVocab, rng), mask);
  }

  // ragged: a row ending mid-word, its last word's bits past the row set.
  {
    constexpr std::int32_t kVocab = 1000;  // 15 whole words and 40 bits
    auto mask = empty_mask(kVocab);
    for (auto &word : mask) { word = bits(rng); }
    mask.back() |= ~0ull << (kVocab % 64);
    run_arm("ragged 1000", random_row(kVocab, rng), mask);
  }

  // extremes: one column at each edge a word index could get wrong, and all.
  {
    constexpr std::int32_t kVocab = 4100;
    const auto row = random_row(kVocab, rng);
    for (const std::int32_t v : {0, 63, 64, kVocab - 1}) {
      run_arm(("one column " + std::to_string(v)).c_str(), row, one_column(kVocab, v));
    }
    auto all = empty_mask(kVocab);
    for (auto &word : all) { word = ~0ull; }
    run_arm("every column", row, all);
  }

  if (g_failed != 0) {
    std::fprintf(stderr, "ignis_permit_mask_bits: %d checks failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("ignis_permit_mask_bits: every arm exact\n");
  return EXIT_SUCCESS;
}
