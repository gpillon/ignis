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
//   set          GitHub #263 (ADR 0039): the fused launch over a head set,
//                on the plane and on the pages -- each head's argmax against
//                the host's, an excluded key scoring above everything
//                skipped, an exact tie won by the larger key, the pointing
//                head's row equal to the single-head kernel's whether or not
//                it is one of the set, and the guards.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU fails this test rather than skipping it.

#include "attention_readout.h"

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

// ── GitHub #263 (ADR 0039): the head set, in one fused launch ────────────

// The heads of a set one layer holds, across all four KV heads and more than
// one per KV head; 10 is the pointing head (KV head 1).
const std::vector<int> kSetHeads = {0, 5, 10, 11, 13, 17, 23};

struct SetRun {
  int32_t rc = 0;
  bool read = false;
  std::vector<float> scores;             // the pointing head's row, if asked
  std::vector<unsigned long long> best;  // packed, one per head of `heads`
  std::vector<float> neighbours;         // GitHub #264: four per head, NaN off the grid
};

// GitHub #264: the image grid the span is walked as. 1000 keys over 32
// columns is 31 full rows and a short one of 8 -- so the last row's keys have
// nothing below them, which is a case the grid has to get right.
constexpr std::int32_t kGridCols = 32;

// One fused launch: `heads` with slots 0.., the pointing head's row when
// `row_head` >= 0, `excluded` span-relative.
SetRun run_set(const Scene &scene, const std::vector<int> &heads, int row_head,
               const std::vector<std::int32_t> &excluded, const ninfer::PagedKVBatchLayerView &cache,
               bool hq_prompt_scratch, const void *workspace, std::int64_t span) {
  Device<std::uint16_t> query(scene.query);
  Device<float> scores(static_cast<std::size_t>(scene.key_count));
  Device<unsigned long long> best(heads.size());
  Device<float> neighbours(4 * heads.size());
  SetRun out;
  AttentionReadoutTarget target;
  target.query_head = row_head;
  target.key_begin = scene.key_begin;
  target.key_count = scene.key_count;
  target.device_scores = row_head >= 0 ? scores.p : nullptr;
  target.set_heads = static_cast<int32_t>(heads.size());
  for (std::size_t i = 0; i < heads.size(); ++i) {
    target.set_query_head[i] = heads[i];
    target.set_slot[i] = static_cast<int32_t>(i);
  }
  target.device_set_best = best.p;
  target.device_set_neighbours = neighbours.p;
  target.grid_cols = kGridCols;
  target.excluded_count = static_cast<int32_t>(excluded.size());
  std::copy(excluded.begin(), excluded.end(), target.excluded);
  target.read = &out.read;
  const char *error = nullptr;
  out.rc = ignis_attention_readout_run(target, query.p, kQHeads, kKvHeads, scene.tokens,
                                       scene.visible, cache, hq_prompt_scratch, workspace, span,
                                       kScale, nullptr, &error);
  CUDA_FATAL(cudaDeviceSynchronize());
  out.scores = scores.read();
  out.best = best.read();
  out.neighbours = neighbours.read();
  return out;
}

std::vector<double> query_of_head(const Scene &scene, int head) {
  Scene one = scene;
  one.query_head = head;
  return query_of(one);
}

// A key row's value `d`, head `kv`, position `p`, as each layout stores it.
using KeyAt = std::function<double(int kv, std::int64_t position, int d)>;

// Each head's argmax over the span minus `excluded` against the host's
// double-precision scores: the same key, or one whose score ties the
// maximum to float accumulation -- and on an exact tie, the larger index.
void check_set(const Scene &scene, const SetRun &got, const std::vector<int> &heads,
               const std::vector<std::int32_t> &excluded, const KeyAt &key, bool hq,
               const std::string &label) {
  check(got.rc == 0 && got.read, label + ": the fused readout reads");
  for (std::size_t i = 0; i < heads.size(); ++i) {
    const std::vector<double> q0 = query_of_head(scene, heads[i]);
    const std::vector<double> q = hq ? rotate(q0) : q0;
    const int kv = heads[i] / (kQHeads / kKvHeads);
    std::vector<double> want(static_cast<std::size_t>(scene.key_count));
    double top = -1e300;
    for (std::int64_t k = 0; k < scene.key_count; ++k) {
      double dot = 0.0;
      for (int d = 0; d < kHeadDim; ++d) {
        dot += q[d] * key(kv, scene.key_begin + k, d);
      }
      want[static_cast<std::size_t>(k)] = dot * kScale;
      if (std::find(excluded.begin(), excluded.end(), k) == excluded.end()) {
        top = std::max(top, want[static_cast<std::size_t>(k)]);
      }
    }
    const unsigned long long packed = got.best[i];
    const auto index = static_cast<std::int64_t>(packed & 0xffffffffULL);
    const bool scored = packed != 0 && index < scene.key_count;
    check(scored, label + ": head " + std::to_string(heads[i]) + " published a key");
    if (!scored) {
      continue;
    }
    const bool not_excluded = std::find(excluded.begin(), excluded.end(), index) == excluded.end();
    check(not_excluded, label + ": head " + std::to_string(heads[i]) + " landed on an excluded key " +
                            std::to_string(index));
    const double at = want[static_cast<std::size_t>(index)];
    check(at >= top - 1e-4 * std::max(1.0, std::fabs(top)),
          label + ": head " + std::to_string(heads[i]) + " peaks at key " + std::to_string(index) +
              " scoring " + std::to_string(at) + ", the maximum is " + std::to_string(top));

    // GitHub #264: the peak's own score, and the four keys around it in the
    // image grid -- against the same double-precision restatement. A
    // neighbour the grid does not have is NaN, never a score.
    const float peak = ignis_attention_unpack_score(static_cast<std::uint32_t>(packed >> 32));
    const auto close = [&](float got, double expected, const std::string &what) {
      const double diff = std::fabs(static_cast<double>(got) - expected);
      check(diff <= 1e-3 * std::max(1.0, std::fabs(expected)),
            label + ": head " + std::to_string(heads[i]) + " " + what + ": read " +
                std::to_string(got) + ", the host says " + std::to_string(expected));
    };
    close(peak, at, "peak score");
    const std::int64_t col = index % kGridCols;
    const std::int64_t row = index / kGridCols;
    const std::int64_t grid_rows = (scene.key_count + kGridCols - 1) / kGridCols;
    const std::array<std::int64_t, 4> neighbour = {
        col > 0 ? index - 1 : -1,
        col + 1 < kGridCols ? index + 1 : -1,
        row > 0 ? index - kGridCols : -1,
        row + 1 < grid_rows ? index + kGridCols : -1,
    };
    for (std::size_t j = 0; j < neighbour.size(); ++j) {
      const float read = got.neighbours[4 * i + j];
      const std::int64_t k = neighbour[j] < scene.key_count ? neighbour[j] : -1;
      if (k < 0) {
        check(std::isnan(read), label + ": head " + std::to_string(heads[i]) + " neighbour " +
                                    std::to_string(j) + " is off the grid, read " +
                                    std::to_string(read));
      } else {
        close(read, want[static_cast<std::size_t>(k)],
              "neighbour " + std::to_string(j) + " (key " + std::to_string(k) + ")");
      }
    }
  }
}

// The set's keys, with three rows planted in every KV head: an excluded key
// scoring far above everything (the argmax must skip it), and two identical
// rows scoring above the rest (an exact tie, which the larger index wins).
struct Planted {
  std::int64_t excluded_key = 500;
  std::int64_t tie_low = 200;
  std::int64_t tie_high = 700;
};

void the_set_is_read_in_one_launch_off_the_plane(std::mt19937 &rng) {
  std::printf("set/plane: hq keys, every head of the set\n");
  const Scene scene = make_scene(rng);
  const Planted planted;
  const std::int64_t span = scene.visible;
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(kKvHeads) * span * kHeadDim);
  for (auto &k : plane) {
    k = to_bf16(normal(rng));
  }
  // Planted rows: along the sum of the KV head's rotated set queries, so every
  // head of the group scores them high.
  for (int kv = 0; kv < kKvHeads; ++kv) {
    std::vector<double> direction(kHeadDim, 0.0);
    for (const int h : kSetHeads) {
      if (h / (kQHeads / kKvHeads) == kv) {
        const std::vector<double> rq = rotate(query_of_head(scene, h));
        for (int d = 0; d < kHeadDim; ++d) {
          direction[d] += rq[d];
        }
      }
    }
    const auto plant = [&](std::int64_t k, double gain) {
      const std::int64_t position = scene.key_begin + k;
      for (int d = 0; d < kHeadDim; ++d) {
        plane[(static_cast<std::size_t>(kv) * span + position) * kHeadDim + d] =
            to_bf16(static_cast<float>(gain * direction[d]));
      }
    };
    plant(planted.excluded_key, 8.0);
    plant(planted.tie_low, 4.0);
    plant(planted.tie_high, 4.0);
  }
  Device<std::uint16_t> workspace(plane);
  ninfer::PagedKVBatchLayerView cache;
  cache.dtype = ninfer::DType::U8;
  const std::vector<std::int32_t> excluded = {0, static_cast<std::int32_t>(planted.excluded_key),
                                              static_cast<std::int32_t>(scene.key_count - 1)};
  const KeyAt key = [&](int kv, std::int64_t position, int d) {
    return from_bf16(plane[(static_cast<std::size_t>(kv) * span + position) * kHeadDim + d]);
  };

  const SetRun got = run_set(scene, kSetHeads, scene.query_head, excluded, cache, true, workspace.p, span);
  check_set(scene, got, kSetHeads, excluded, key, /*hq=*/true, "plane");
  for (std::size_t i = 0; i < kSetHeads.size(); ++i) {
    check((got.best[i] & 0xffffffffULL) == static_cast<unsigned long long>(planted.tie_high),
          "plane: head " + std::to_string(kSetHeads[i]) +
              " breaks the planted tie toward the larger key");
  }
  // The pointing head's row, written in the same pass, is the single-head
  // kernel's -- the excluded key included.
  const Run alone = run(scene, cache, /*hq_prompt_scratch=*/true, workspace.p, span);
  double worst = 0.0;
  for (std::size_t k = 0; k < alone.scores.size(); ++k) {
    worst = std::max(worst, static_cast<double>(std::fabs(got.scores[k] - alone.scores[k])) /
                                std::max(1.0, static_cast<double>(std::fabs(alone.scores[k]))));
  }
  std::printf("  pointing row against the single-head kernel: worst relative %.2e\n", worst);
  check(worst < 1e-5, "plane: the pointing head's row is the single-head kernel's");
}

void the_set_is_read_in_one_launch_off_the_pages(std::mt19937 &rng) {
  std::printf("set/pages: BF16 keys through a permuted block table\n");
  const Scene scene = make_scene(rng);
  const int pages = static_cast<int>((scene.visible + kPage - 1) / kPage);
  std::vector<std::int32_t> table(pages);
  for (int p = 0; p < pages; ++p) {
    table[p] = pages - 1 - p;
  }
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(pages) * kKvHeads * kPage * kHeadDim);
  for (auto &k : plane) {
    k = to_bf16(normal(rng));
  }
  const auto row_of = [&](int kv, std::int64_t position) {
    return ((static_cast<std::size_t>(table[position / kPage]) * kKvHeads + kv) * kPage +
            static_cast<std::size_t>(position % kPage)) *
           kHeadDim;
  };
  Device<std::uint16_t> k_pages(plane);
  Device<std::int32_t> block_table(table);
  ninfer::PagedKVBatchLayerView cache;
  cache.dtype = ninfer::DType::BF16;
  cache.k_pages = ninfer::Tensor(k_pages.p, ninfer::DType::BF16, {kHeadDim, kKvHeads * kPage * pages, 1, 1});
  cache.block_tables = ninfer::Tensor(block_table.p, ninfer::DType::I32, {pages, 1, 1, 1});
  const std::vector<std::int32_t> excluded = {0, 1, 223, static_cast<std::int32_t>(scene.key_count - 1)};
  const KeyAt key = [&](int kv, std::int64_t position, int d) {
    return from_bf16(plane[row_of(kv, position) + d]);
  };
  // The pointing head outside the set: its row still comes back, and it
  // takes no argmax slot.
  const std::vector<int> heads = {0, 5, 13, 17, 23};
  const SetRun got = run_set(scene, heads, scene.query_head, excluded, cache, false, nullptr, 0);
  check_set(scene, got, heads, excluded, key, /*hq=*/false, "pages");
  const Run alone = run(scene, cache, /*hq_prompt_scratch=*/false, nullptr, 0);
  double worst = 0.0;
  for (std::size_t k = 0; k < alone.scores.size(); ++k) {
    worst = std::max(worst, static_cast<double>(std::fabs(got.scores[k] - alone.scores[k])) /
                                std::max(1.0, static_cast<double>(std::fabs(alone.scores[k]))));
  }
  check(worst < 1e-5, "pages: the pointing head outside the set still gets its row");

  // A layer that holds only heads of the set writes no row at all.
  const SetRun set_only = run_set(scene, heads, -1, excluded, cache, false, nullptr, 0);
  check_set(scene, set_only, heads, excluded, key, /*hq=*/false, "pages, no pointing head");
}

void a_set_that_cannot_be_read_is_left_unread(std::mt19937 &rng) {
  std::printf("set/unread: the guards\n");
  const Scene scene = make_scene(rng);
  std::vector<std::uint16_t> plane(static_cast<std::size_t>(kKvHeads) * scene.visible * kHeadDim, to_bf16(1.0F));
  Device<std::uint16_t> workspace(plane);
  ninfer::PagedKVBatchLayerView hq;
  hq.dtype = ninfer::DType::U8;
  SetRun got = run_set(scene, kSetHeads, scene.query_head, {}, hq, /*hq_prompt_scratch=*/false,
                       workspace.p, scene.visible);
  check(got.rc == 0 && !got.read, "no prompt-route plane: the set is unread, not an error");
  bool untouched = true;
  for (const unsigned long long b : got.best) {
    untouched = untouched && b == 0;
  }
  check(untouched, "and no head published anything");
  got = run_set(scene, {0, 5, 5}, scene.query_head, {}, hq, true, workspace.p, scene.visible);
  check(got.rc != 0 && !got.read, "a head named twice: refused");
  got = run_set(scene, {0, 24}, scene.query_head, {}, hq, true, workspace.p, scene.visible);
  check(got.rc != 0 && !got.read, "a query head the layer does not have: refused");
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
  the_set_is_read_in_one_launch_off_the_plane(rng);
  the_set_is_read_in_one_launch_off_the_pages(rng);
  a_set_that_cannot_be_read_is_left_unread(rng);
  if (g_failed > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("all attention readout checks passed\n");
  return EXIT_SUCCESS;
}
