// The attention readout (GitHub #260, ADR 0038) -- OURS, not vendored
// (kernel/src/attention_readout.h).
//
// One query head's pre-softmax scores over a span of keys, from the two
// places the GQA layer's attention reads keys: the BF16 cache's pages through
// the sequence's block table, and the hq-e8-2b prompt route's materialized
// key plane (rotated frame, rows at absolute positions). The oracle is a
// host restatement in double precision of the same definition:
// `q . k / sqrt(256)`, with the query rotated by the codec's
// `R = H * diag(signs) / 16` on the hq side.
//
// What the arms are for:
//
//   pages        a block table that is not the identity and a span that
//                starts mid-page, so a kernel that ignored the indirection
//                or the page offset reads the wrong rows.
//   plane        the hq plane read against the host-rotated query; then the
//                plane filled with the rotation of known keys, where the
//                score must come back as the frame-free q . k -- which is the
//                property the readout rests on (the codec's rotation is
//                orthonormal), checked on the device's own rotation.
//   head         a query head other than the first of its KV group, so a
//                kernel that took the wrong query column or KV head fails.
//   unread       every guard that leaves the readout unread instead of
//                scoring the wrong keys: no prompt-route plane, keys past the
//                history, keys past the plane's band -- and the one geometry
//                it refuses outright.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "attention_readout.h"

#include "core/tensor.h"

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
#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kPage = 64;
const float kScale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

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

template <typename T> struct Device {
  T *p = nullptr;
  std::size_t n = 0;
  explicit Device(const std::vector<T> &host) : n(host.size()) {
    CUDA_FATAL(cudaMalloc(&p, n * sizeof(T)));
    CUDA_FATAL(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  explicit Device(std::size_t count) : n(count) {
    CUDA_FATAL(cudaMalloc(&p, n * sizeof(T)));
    CUDA_FATAL(cudaMemset(p, 0, n * sizeof(T)));
  }
  ~Device() { cudaFree(p); }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  std::vector<T> read() const {
    std::vector<T> host(n);
    CUDA_FATAL(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }
};

std::uint16_t to_bf16(float value) {
  const __nv_bfloat16 b = __float2bfloat16(value);
  std::uint16_t bits;
  std::memcpy(&bits, &b, sizeof(bits));
  return bits;
}

double from_bf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// The codec's per-coordinate sign, restated from its definition
// (kernel/vendor/src/ops/kernel/hq_codec.cuh `hq_engine_sign`).
double codec_sign(int d) {
  std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  return (x & 1u) ? 1.0 : -1.0;
}

// R v = H diag(signs) v / 16, natural-order Walsh-Hadamard, in double.
std::vector<double> rotate(const std::vector<double> &v) {
  std::vector<double> x(kHeadDim);
  for (int d = 0; d < kHeadDim; ++d) {
    x[d] = v[d] * codec_sign(d);
  }
  for (int len = 1; len < kHeadDim; len <<= 1) {
    for (int base = 0; base < kHeadDim; base += 2 * len) {
      for (int i = base; i < base + len; ++i) {
        const double a = x[i];
        const double b = x[i + len];
        x[i] = a + b;
        x[i + len] = a - b;
      }
    }
  }
  for (double &a : x) {
    a /= 16.0;
  }
  return x;
}

struct Scene {
  int tokens = 3;           // the chunk's width; the query is its last column
  int query_head = 10;      // KV head 1, not the first of its group
  std::int64_t visible = 1100;
  std::int64_t key_begin = 37;
  std::int64_t key_count = 1000;
  std::vector<std::uint16_t> query; // [kQHeads * 256, tokens]
};

Scene make_scene(std::mt19937 &rng) {
  Scene scene;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  scene.query.resize(static_cast<std::size_t>(kQHeads) * kHeadDim * scene.tokens);
  for (auto &q : scene.query) {
    q = to_bf16(normal(rng));
  }
  return scene;
}

std::vector<double> query_of(const Scene &scene) {
  std::vector<double> q(kHeadDim);
  const std::size_t base = (static_cast<std::size_t>(scene.tokens) - 1) * kQHeads * kHeadDim +
                           static_cast<std::size_t>(scene.query_head) * kHeadDim;
  for (int d = 0; d < kHeadDim; ++d) {
    q[d] = from_bf16(scene.query[base + d]);
  }
  return q;
}

struct Run {
  int32_t rc = 0;
  bool read = false;
  std::vector<float> scores;
};

Run run(const Scene &scene, const ninfer::PagedKVBatchLayerView &cache, bool hq_prompt_scratch,
        const void *workspace, std::int64_t span, int32_t kv_heads = kKvHeads) {
  Device<std::uint16_t> query(scene.query);
  Device<float> scores(static_cast<std::size_t>(scene.key_count));
  Run out;
  AttentionReadoutTarget target;
  target.query_head = scene.query_head;
  target.key_begin = scene.key_begin;
  target.key_count = scene.key_count;
  target.device_scores = scores.p;
  target.read = &out.read;
  const char *error = nullptr;
  out.rc = ignis_attention_readout_run(target, query.p, kQHeads, kv_heads, scene.tokens,
                                       scene.visible, cache, hq_prompt_scratch, workspace, span,
                                       kScale, nullptr, &error);
  CUDA_FATAL(cudaDeviceSynchronize());
  out.scores = scores.read();
  return out;
}

double worst_relative(const std::vector<float> &got, const std::vector<double> &want) {
  double worst = 0.0;
  for (std::size_t k = 0; k < want.size(); ++k) {
    worst = std::max(worst, std::fabs(got[k] - want[k]) / std::max(1.0, std::fabs(want[k])));
  }
  return worst;
}

void the_pages_are_read_through_the_block_table(std::mt19937 &rng) {
  std::printf("pages: BF16 keys through a permuted block table\n");
  const Scene scene = make_scene(rng);
  const int pages = static_cast<int>((scene.visible + kPage - 1) / kPage);
  // Physical page for each logical page: reversed, so no logical page is
  // its own physical one.
  std::vector<std::int32_t> table(pages);
  for (int p = 0; p < pages; ++p) {
    table[p] = pages - 1 - p;
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(pages) * kKvHeads * kPage * kHeadDim);
  for (auto &k : plane) {
    k = to_bf16(normal(rng));
  }
  Device<std::uint16_t> k_pages(plane);
  Device<std::int32_t> block_table(table);
  ninfer::PagedKVBatchLayerView cache;
  cache.dtype = ninfer::DType::BF16;
  cache.k_pages = ninfer::Tensor(k_pages.p, ninfer::DType::BF16, {kHeadDim, kKvHeads * kPage * pages, 1, 1});
  cache.block_tables = ninfer::Tensor(block_table.p, ninfer::DType::I32, {pages, 1, 1, 1});

  const Run got = run(scene, cache, /*hq_prompt_scratch=*/false, nullptr, 0);
  check(got.rc == 0 && got.read, "the BF16 readout reads");
  const int kv_head = scene.query_head / (kQHeads / kKvHeads);
  const std::vector<double> q = query_of(scene);
  std::vector<double> want(static_cast<std::size_t>(scene.key_count));
  for (std::int64_t k = 0; k < scene.key_count; ++k) {
    const std::int64_t position = scene.key_begin + k;
    // Page-major: [physical page][kv head][64 positions][256].
    const std::size_t row = ((static_cast<std::size_t>(table[position / kPage]) * kKvHeads + kv_head) * kPage +
                             static_cast<std::size_t>(position % kPage)) *
                            kHeadDim;
    double dot = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
      dot += q[d] * from_bf16(plane[row + d]);
    }
    want[static_cast<std::size_t>(k)] = dot * kScale;
  }
  const double worst = worst_relative(got.scores, want);
  std::printf("  worst relative error %.2e over %lld keys\n", worst,
              static_cast<long long>(scene.key_count));
  check(worst < 1e-4, "the BF16 scores are q . k / 16 of the table's rows");
}

void the_plane_is_read_in_the_codec_frame(std::mt19937 &rng) {
  std::printf("plane: hq keys in the rotated frame\n");
  const Scene scene = make_scene(rng);
  const std::int64_t span = scene.visible;
  const int kv_head = scene.query_head / (kQHeads / kKvHeads);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  // Every row of the plane is the rotation of a known key, rounded to BF16
  // as the codec's rows are.
  std::vector<std::vector<double>> keys(static_cast<std::size_t>(span), std::vector<double>(kHeadDim));
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(kKvHeads) * span * kHeadDim);
  for (int h = 0; h < kKvHeads; ++h) {
    for (std::int64_t p = 0; p < span; ++p) {
      std::vector<double> k(kHeadDim);
      for (auto &x : k) {
        x = normal(rng);
      }
      if (h == kv_head) {
        keys[static_cast<std::size_t>(p)] = k;
      }
      const std::vector<double> rk = rotate(k);
      for (int d = 0; d < kHeadDim; ++d) {
        plane[(static_cast<std::size_t>(h) * span + p) * kHeadDim + d] = to_bf16(static_cast<float>(rk[d]));
      }
    }
  }
  Device<std::uint16_t> workspace(plane);
  ninfer::PagedKVBatchLayerView cache;
  cache.dtype = ninfer::DType::U8;

  const Run got = run(scene, cache, /*hq_prompt_scratch=*/true, workspace.p, span);
  check(got.rc == 0 && got.read, "the hq readout reads");
  const std::vector<double> q = query_of(scene);
  const std::vector<double> rq = rotate(q);
  std::vector<double> in_frame(static_cast<std::size_t>(scene.key_count));
  std::vector<double> frame_free(static_cast<std::size_t>(scene.key_count));
  for (std::int64_t k = 0; k < scene.key_count; ++k) {
    const std::int64_t position = scene.key_begin + k;
    double rotated = 0.0;
    double plain = 0.0;
    for (int d = 0; d < kHeadDim; ++d) {
      rotated += rq[d] * from_bf16(plane[(static_cast<std::size_t>(kv_head) * span + position) * kHeadDim + d]);
      plain += q[d] * keys[static_cast<std::size_t>(position)][d];
    }
    in_frame[static_cast<std::size_t>(k)] = rotated * kScale;
    frame_free[static_cast<std::size_t>(k)] = plain * kScale;
  }
  const double exact = worst_relative(got.scores, in_frame);
  const double frame = worst_relative(got.scores, frame_free);
  std::printf("  worst relative error %.2e against R q . plane, %.2e against q . k (BF16 plane)\n", exact,
              frame);
  check(exact < 1e-4, "the hq scores are the rotated query against the plane's rows");
  // The plane's rows are BF16 roundings of R k: the frame-free score holds to
  // that rounding, and a query left unrotated would miss it by the whole
  // score.
  check(frame < 2e-2, "the device's rotation is the codec's: R q . R k == q . k");
}

void a_read_that_cannot_be_made_is_left_unread(std::mt19937 &rng) {
  std::printf("unread: the guards\n");
  Scene scene = make_scene(rng);
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(kKvHeads) * scene.visible * kHeadDim, to_bf16(1.0F));
  Device<std::uint16_t> workspace(plane);
  ninfer::PagedKVBatchLayerView hq;
  hq.dtype = ninfer::DType::U8;

  Run got = run(scene, hq, /*hq_prompt_scratch=*/false, workspace.p, scene.visible);
  check(got.rc == 0 && !got.read, "no prompt-route plane: unread, not an error");

  got = run(scene, hq, /*hq_prompt_scratch=*/true, workspace.p, scene.key_begin + scene.key_count - 1);
  check(got.rc == 0 && !got.read, "keys past the plane's band: unread");

  Scene past = scene;
  past.visible = past.key_begin + past.key_count - 1;
  got = run(past, hq, /*hq_prompt_scratch=*/true, workspace.p, past.visible);
  check(got.rc == 0 && !got.read, "keys past the history: unread");

  got = run(scene, hq, /*hq_prompt_scratch=*/true, workspace.p, scene.visible, /*kv_heads=*/8);
  check(got.rc != 0 && !got.read, "a head geometry it was not built for: refused");
}

} // namespace

int main() {
  int devices = 0;
  CUDA_FATAL(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return EXIT_FAILURE;
  }
  std::mt19937 rng(20260922);
  the_pages_are_read_through_the_block_table(rng);
  the_plane_is_read_in_the_codec_frame(rng);
  a_read_that_cannot_be_made_is_left_unread(rng);
  if (g_failed > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("all attention readout checks passed\n");
  return EXIT_SUCCESS;
}
