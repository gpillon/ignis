// GitHub #564 (spec #555, ADR 0087): the MTP head's combine (kernel/src/flash_next/mtp.cu) takes an
// image's vision rows -- OURS, not vendored; the reference has no MTP head, so the rule is ours.
//
// An entry whose next token is an image placeholder embeds the vision row the trunk took for that
// placeholder, not the placeholder id's row:
// - with runs of columns over their rows, combine's input X equals, bit for bit and in every row,
//   the combine of the same stacks over tokens whose embedding rows are those columns -- two runs,
//   as a pool page's end splits them, from a VRAM page and from a page in mapped host memory (the
//   host residency's pool), read in place;
// - every row no run names is the plain call's, bit for bit;
// - without the runs every placeholder row is another X (the placeholder id's row), so the first
//   check can fail.
// Synthetic BF16 weights at Flash-Next's widths; the projections' numerics are fn_linear's own
// test's, the columns' copy into every stream fn_embed_columns' (test_flash_next_embed_head.cu).
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "flash_next/embed_head.h"
#include "flash_next/mtp.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace fnx = ignis::flash_next;

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

template <typename T>
T *upload(const std::vector<T> &host) {
  T *dev = nullptr;
  cuda_ok(cudaMalloc(&dev, host.size() * sizeof(T)), "cudaMalloc");
  cuda_ok(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
  return dev;
}

constexpr int32_t kVocab = 1000, kHidden = 2560, kStreams = 4;
constexpr int32_t kPlaceholder = 999;
constexpr std::size_t kRowElems = static_cast<std::size_t>(kStreams) * kHidden;

fnx::Geometry geometry() {
  fnx::Geometry g;
  g.vocab = kVocab;
  g.hidden = kHidden;
  g.streams = kStreams;
  g.rms_norm_eps = 1e-6F;
  return g;
}

std::vector<uint16_t> random(uint32_t &seed, std::size_t n, float scale) {
  std::vector<uint16_t> v(n);
  for (auto &x : v) x = bf16_bits((static_cast<float>(lcg(seed) >> 8) / 16777216.0F * 2.0F - 1.0F) * scale);
  return v;
}

// The head's combine weights and the token embedding, BF16, and one set of stacks.
struct Head {
  fnx::Geometry g = geometry();
  std::vector<uint16_t> table;  // embed_tokens [vocab][hidden]
  fnx::Linear embed;
  fnx::MtpWeights w;
  std::vector<uint16_t> stacks;  // [rows][streams * hidden]
};

Head make_head(int32_t rows) {
  uint32_t seed = 41U;
  Head h;
  h.table = random(seed, static_cast<std::size_t>(kVocab) * kHidden, 0.5F);
  h.embed = {upload(h.table), kVocab, kHidden, fnx::WeightFormat::Bf16};
  h.w.fc_embedding = {upload(random(seed, static_cast<std::size_t>(kHidden) * kHidden, 0.02F)), kHidden, kHidden,
                      fnx::WeightFormat::Bf16};
  h.w.fc_hidden = {upload(random(seed, static_cast<std::size_t>(kHidden) * kHidden, 0.02F)), kHidden, kHidden,
                   fnx::WeightFormat::Bf16};
  h.w.norm_embedding = upload(random(seed, kHidden, 0.1F));
  h.w.norm_hidden = upload(random(seed, kRowElems, 0.1F));
  h.stacks = random(seed, static_cast<std::size_t>(rows) * kRowElems, 2.0F);
  return h;
}

// X for `tokens` over the head's stacks, with `media`'s runs (or none).
std::vector<uint16_t> combine(const Head &h, const std::vector<int32_t> &tokens, const std::vector<fnx::MediaRun> *media,
                              const std::string &label) {
  const auto rows = static_cast<int32_t>(tokens.size());
  auto *d_residual = upload(h.stacks);
  auto *d_tokens = upload(tokens);
  fnx::mtp::Buffers b;
  b.residual = d_residual;
  fnx::Batch batch;
  batch.lanes = 1;
  batch.tokens = rows;
  ninfer::DeviceArena scratch(16u << 20);
  check(fnx::mtp::combine(h.g, h.w, h.embed, d_tokens, batch, b, scratch, nullptr, media) == 0,
        label + ": runs: " + fnx::fn_last_error());
  cuda_ok(cudaDeviceSynchronize(), "sync");
  std::vector<uint16_t> x(static_cast<std::size_t>(rows) * kRowElems);
  cuda_ok(cudaMemcpy(x.data(), d_residual, x.size() * 2, cudaMemcpyDeviceToHost), "download");
  cudaFree(d_residual);
  cudaFree(d_tokens);
  return x;
}

bool same_row(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, std::size_t row) {
  return std::memcmp(&a[row * kRowElems], &b[row * kRowElems], kRowElems * 2) == 0;
}

// A pool page's columns, in a VRAM buffer or in mapped host memory, as the device pointer the
// program reads in place.
struct Page {
  const uint16_t *device = nullptr;
  void *host = nullptr;

  Page(const std::vector<uint16_t> &columns, bool mapped) {
    if (mapped) {
      cuda_ok(cudaHostAlloc(&host, columns.size() * 2, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
      std::memcpy(host, columns.data(), columns.size() * 2);
      void *d = nullptr;
      cuda_ok(cudaHostGetDevicePointer(&d, host, 0), "cudaHostGetDevicePointer");
      device = static_cast<const uint16_t *>(d);
    } else {
      device = upload(columns);
    }
  }
  ~Page() {
    if (host != nullptr) {
      cudaFreeHost(host);
    } else {
      cudaFree(const_cast<uint16_t *>(device));
    }
  }
  Page(const Page &) = delete;
  Page &operator=(const Page &) = delete;
};

void media_case(bool mapped) {
  const std::string label = mapped ? "the head's media rows (mapped host pages)" : "the head's media rows (VRAM pages)";
  // Entries 1, 2, 4 and 6 have an image placeholder next; the reference names the ids whose
  // embedding rows the columns are.
  const std::vector<int32_t> placed = {5, kPlaceholder, kPlaceholder, 17, kPlaceholder, 23, kPlaceholder};
  const std::vector<int32_t> stand_in = {5, 101, 202, 17, 303, 23, 404};
  const std::vector<int32_t> media_rows = {1, 2, 4, 6};
  const auto rows = static_cast<int32_t>(placed.size());
  const Head h = make_head(rows);
  const auto column_of = [&](int32_t id) {
    return std::vector<uint16_t>(h.table.begin() + static_cast<std::ptrdiff_t>(id) * kHidden,
                                 h.table.begin() + static_cast<std::ptrdiff_t>(id + 1) * kHidden);
  };
  // Two pages, a page's end between the second and third column: the first holds an unrelated
  // column before the run's two, the second the run's last two.
  std::vector<uint16_t> first_page;
  for (const int32_t id : {7, 101, 202}) {
    const auto c = column_of(id);
    first_page.insert(first_page.end(), c.begin(), c.end());
  }
  std::vector<uint16_t> second_page;
  for (const int32_t id : {303, 404}) {
    const auto c = column_of(id);
    second_page.insert(second_page.end(), c.begin(), c.end());
  }
  const Page page_a(first_page, mapped);
  const Page page_b(second_page, mapped);
  auto *d_rows = upload(media_rows);
  const std::vector<fnx::MediaRun> runs = {
      {page_a.device + kHidden, d_rows, 2},
      {page_b.device, d_rows + 2, 2},
  };

  const auto reference = combine(h, stand_in, nullptr, label + " (the stand-in ids)");
  const auto with_columns = combine(h, placed, &runs, label);
  const auto without = combine(h, placed, nullptr, label + " (no runs)");
  for (int32_t r = 0; r < rows; ++r) {
    check(same_row(with_columns, reference, static_cast<std::size_t>(r)),
          label + ": entry " + std::to_string(r) + " is not, bit for bit, the entry of the id its column is the row of");
  }
  for (int32_t r = 0; r < rows; ++r) {
    const bool named = r == 1 || r == 2 || r == 4 || r == 6;
    if (named) {
      check(!same_row(with_columns, without, static_cast<std::size_t>(r)),
            label + ": entry " + std::to_string(r) + " is the placeholder id's with its column too");
    } else {
      check(same_row(with_columns, without, static_cast<std::size_t>(r)),
            label + ": entry " + std::to_string(r) + ", which no run names, moved");
    }
  }
  cudaFree(d_rows);
}

}  // namespace

int main() {
  int devices = 0;
  cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  media_case(false);
  media_case(true);
  if (g_failed != 0) {
    std::fprintf(stderr, "flash-next MTP combine test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("flash-next MTP combine test: ok\n");
  return 0;
}
