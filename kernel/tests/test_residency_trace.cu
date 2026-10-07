// Expert residency's device side held to the CPU policy model -- OURS (spec flash-next/03,
// acceptance 4, GitHub #301).
//
// Replays kernel/tests/fixtures/residency/trace_v1.txt -- a routing trace and, step by step,
// what crates/core's ResidencyModel did with it (written by
// crates/core/tests/expert_residency_fixture.rs) -- through ignis_residency_step_ranked on
// synthetic records, and checks after every step:
//   - the outcome: status, hits, prefetch hits, misses (and where each landed), evictions,
//     prefetches, dropped candidates and bytes moved equal the model's, as sets;
//   - the copies: every selected projection's slot-table entry points at its own record's bytes;
//   - the slot table: every entry is ABSENT or points at a slot boundary of its own class's pool
//     or into the staging ring, no two entries share a slot, an evicted projection is ABSENT,
//     and a staged projection is ABSENT again once a step of another layer has run;
//   - at the end, the counters add up to the trace;
//   - the stall: a step (or captured round) with no miss adds nothing to stall_nanos, none touches
//     the other phase, and the copying steps' additions are each phase's whole stall, above 0.
// `graph` runs every whole decode round (all layers in order, one token count) as one CUDA graph,
// captured once per token count and replayed with new ids in the same buffers: decode residency
// is graph-capturable across the layer steps of a round (spec acceptance 3); the rest runs
// eagerly. Each layer's outcome is read from its own report. `split` (alone or after `graph`)
// runs every step as ignis_residency_step_demand plus, on the stream it forks, a prefetch half:
// ignis_residency_step_prefetch on logits that rank as the fixture's lookahead where a row of it
// can be written so (an eager step whose rows hold `width` distinct ids), else
// ignis_residency_step_prefetch_ranked. The split step's outcomes are the whole step's.
// Every mode then checks that a whole step at the last layer ignores its lookahead, as the policy
// does, out-of-range ids included, and that a prefill width of 0 is no lookahead.
// Same no-SKIP_RETURN_CODE rule as every GPU test here (ADR 0006).

#include "ignis_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#ifndef IGNIS_RESIDENCY_FIXTURE_PATH
#error "IGNIS_RESIDENCY_FIXTURE_PATH must name kernel/tests/fixtures/residency/trace_v1.txt"
#endif

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

#define RES_OK(expr)                                                                             \
  do {                                                                                           \
    if ((expr) != 0) {                                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr,                         \
                   ignis_residency_last_error());                                                \
      std::exit(1);                                                                              \
    }                                                                                            \
  } while (0)

int g_failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    if (g_failures < 40) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Expect {
  uint32_t status = 0;
  std::vector<uint32_t> list[IGNIS_RESIDENCY_LISTS];  // key | staging bit
  uint64_t bytes = 0;
};

struct Step {
  uint32_t layer = 0, phase = 0, tokens = 0, rows = 0, stride = 0;
  std::vector<int32_t> ids, lookahead;
  Expect expect;
};

struct Fixture {
  uint32_t layers = 0, experts = 0, top_k = 0, width = 0, prefill_width = 0;
  uint64_t record_bytes[8] = {};
  uint32_t capacity[8] = {};
  uint64_t budget = 0, budget_row = 0;
  std::vector<uint8_t> k2;
  std::vector<uint32_t> warm;
  std::vector<Step> steps;
};

void expect_word(std::ifstream &in, const char *word) {
  std::string w;
  in >> w;
  if (w != word) {
    std::fprintf(stderr, "fixture: expected '%s', read '%s'\n", word, w.c_str());
    std::exit(1);
  }
}

Fixture load(const char *path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "fixture %s is missing\n", path);
    std::exit(1);
  }
  Fixture f;
  int version = 0;
  expect_word(in, "ignis-residency-fixture");
  in >> version;
  if (version != 3) {
    std::fprintf(stderr, "fixture: version %d, this test reads 3\n", version);
    std::exit(1);
  }
  expect_word(in, "layers");
  in >> f.layers;
  expect_word(in, "experts");
  in >> f.experts;
  expect_word(in, "top_k");
  in >> f.top_k;
  expect_word(in, "record_bytes");
  for (auto &b : f.record_bytes) in >> b;
  expect_word(in, "capacity");
  for (auto &c : f.capacity) in >> c;
  expect_word(in, "width");
  in >> f.width;
  expect_word(in, "prefill_width");
  in >> f.prefill_width;
  expect_word(in, "budget");
  in >> f.budget;
  expect_word(in, "budget_row");
  in >> f.budget_row;
  expect_word(in, "k2");
  f.k2.resize(static_cast<size_t>(f.layers) * f.experts * 2);
  for (auto &k : f.k2) {
    unsigned v = 0;
    in >> v;
    k = static_cast<uint8_t>(v);
  }
  expect_word(in, "warm");
  uint32_t n = 0;
  in >> n;
  f.warm.resize(n);
  for (auto &k : f.warm) in >> k;
  expect_word(in, "steps");
  in >> n;
  static const char *kLists[IGNIS_RESIDENCY_LISTS] = {"hits",       "prefetch_hits", "misses",
                                                      "evictions",  "prefetches",    "dropped"};
  for (uint32_t i = 0; i < n; ++i) {
    Step s;
    expect_word(in, "step");
    in >> s.layer >> s.phase >> s.tokens;
    s.ids.resize(static_cast<size_t>(s.tokens) * f.top_k);
    for (auto &e : s.ids) in >> e;
    in >> s.rows >> s.stride;
    s.lookahead.resize(static_cast<size_t>(s.rows) * s.stride);
    for (auto &e : s.lookahead) in >> e;
    expect_word(in, "expect");
    in >> s.expect.status;
    if (s.expect.status == 0) {
      for (int l = 0; l < IGNIS_RESIDENCY_LISTS; ++l) {
        expect_word(in, kLists[l]);
        uint32_t count = 0;
        in >> count;
        const bool admitted = l == 2 || l == 4;
        for (uint32_t j = 0; j < count; ++j) {
          uint32_t key = 0, staging = 0;
          in >> key;
          if (admitted) in >> staging;
          s.expect.list[l].push_back(key | (staging ? IGNIS_RESIDENCY_STAGING_BIT : 0));
        }
      }
      expect_word(in, "bytes");
      in >> s.expect.bytes;
    }
    f.steps.push_back(std::move(s));
  }
  if (!in) {
    std::fprintf(stderr, "fixture %s is truncated\n", path);
    std::exit(1);
  }
  return f;
}

uint32_t class_of(const Fixture &f, uint32_t key) {
  const uint32_t k2 = f.k2[key];
  return (key & 1) * 4 + (k2 == 4 ? 0 : k2 == 5 ? 1 : k2 == 6 ? 2 : 3);
}

uint8_t pattern(uint32_t key, uint64_t i) {
  return static_cast<uint8_t>((key * 2654435761u + i * 40503u) >> 13);
}

// A decode round captured as one CUDA graph: per layer, the ids and lookahead buffers it reads.
struct Round {
  std::vector<int32_t *> ids, lookahead;
  cudaGraphExec_t exec = nullptr;
};

}  // namespace

int main(int argc, char **argv) {
  bool graph_mode = false, split_mode = false;
  for (int i = 1; i < argc; ++i) {
    graph_mode = graph_mode || std::string(argv[i]) == "graph";
    split_mode = split_mode || std::string(argv[i]) == "split";
  }
  const Fixture f = load(IGNIS_RESIDENCY_FIXTURE_PATH);
  const uint32_t nk = f.experts * 2;
  const uint64_t keys = static_cast<uint64_t>(f.layers) * nk;

  // Records packed in key order, as the binder's expert pool lays them out.
  std::vector<uint64_t> offsets(keys);
  uint64_t pool_bytes = 0, heaviest = 0;
  for (uint32_t l = 0; l < f.layers; ++l) {
    uint64_t layer_bytes = 0;
    for (uint32_t i = 0; i < nk; ++i) {
      const uint32_t key = l * nk + i;
      offsets[key] = pool_bytes;
      const uint64_t b = f.record_bytes[class_of(f, key)];
      pool_bytes += b;
      layer_bytes += b;
    }
    heaviest = std::max(heaviest, layer_bytes);
  }
  uint32_t max_rows = 1;
  for (const Step &s : f.steps) max_rows = std::max({max_rows, s.tokens, s.rows});

  ignis_residency_desc desc{};
  desc.layers = f.layers;
  desc.experts = f.experts;
  for (int c = 0; c < 8; ++c) {
    desc.capacity[c] = f.capacity[c];
    desc.record_bytes[c] = f.record_bytes[c];
  }
  desc.max_tokens = max_rows;
  desc.lookahead_width = f.width;
  desc.prefill_lookahead_width = f.prefill_width;
  desc.prefetch_budget_bytes = f.budget;
  desc.prefetch_budget_row_bytes = f.budget_row;
  desc.staging_half_bytes = heaviest;
  desc.host_pool_bytes = pool_bytes;
  desc.copy_blocks = 8;
  desc.report = 1;
  ignis_residency_plan plan{};
  RES_OK(ignis_residency_plan_bytes(&desc, &plan));
  std::printf("residency: %u layers x %u experts, %u steps, plan %llu B (pools %llu, staging %llu, tables %llu)%s%s\n",
              f.layers, f.experts, static_cast<unsigned>(f.steps.size()),
              static_cast<unsigned long long>(plan.total), static_cast<unsigned long long>(plan.pools),
              static_cast<unsigned long long>(plan.staging), static_cast<unsigned long long>(plan.tables),
              graph_mode ? ", whole decode rounds as CUDA graphs" : "", split_mode ? ", split steps" : "");

  ignis_residency *r = nullptr;
  RES_OK(ignis_residency_create(&desc, f.k2.data(), offsets.data(), &r));
  auto *host = static_cast<uint8_t *>(ignis_residency_host_pool(r));
  for (uint32_t key = 0; key < keys; ++key) {
    const uint64_t b = f.record_bytes[class_of(f, key)];
    for (uint64_t i = 0; i < b; ++i) host[offsets[key] + i] = pattern(key, i);
  }
  uint32_t admitted = 0;
  RES_OK(ignis_residency_warm_start(r, f.warm.data(), static_cast<uint32_t>(f.warm.size()), &admitted));
  // The host mirror, on a page of its own: read with no call, held to the readouts below.
  alignas(4096) static unsigned char mirror_page[4096];
  auto *mirror = reinterpret_cast<volatile ignis_residency_mirror *>(mirror_page);
  RES_OK(ignis_residency_set_mirror(r, reinterpret_cast<ignis_residency_mirror *>(mirror_page)));
  auto check_mirror = [&](const std::string &at) {
    ignis_residency_counters want{};
    RES_OK(ignis_residency_read_counters(r, &want));
    uint32_t occupancy[IGNIS_RESIDENCY_CLASSES];
    RES_OK(ignis_residency_read_occupancy(r, occupancy));
    const volatile uint64_t *got = reinterpret_cast<const volatile uint64_t *>(&mirror->counters);
    const uint64_t *wanted = reinterpret_cast<const uint64_t *>(&want);
    for (size_t i = 0; i < sizeof(want) / 8; ++i) {
      check(got[i] == wanted[i], at + ": mirror counter word " + std::to_string(i) + " reads " +
                                     std::to_string(got[i]) + ", the device " + std::to_string(wanted[i]));
    }
    for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
      check(mirror->in_use[c] == occupancy[c], at + ": mirror class " + std::to_string(c) + " reads " +
                                                   std::to_string(mirror->in_use[c]) + " slots in use, the device " +
                                                   std::to_string(occupancy[c]));
    }
  };
  check_mirror("the mirror after the warm start");
  ignis_residency_layout layout{};
  RES_OK(ignis_residency_get_layout(r, &layout));

  cudaStream_t stream;
  CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int32_t *d_ids = nullptr, *d_look = nullptr;
  CUDA_OK(cudaMalloc(&d_ids, static_cast<size_t>(max_rows) * f.top_k * 4));
  CUDA_OK(cudaMalloc(&d_look, static_cast<size_t>(max_rows) * std::max(f.width, 1u) * 4));
  float *d_logits = nullptr;
  CUDA_OK(cudaMalloc(&d_logits, static_cast<size_t>(max_rows) * f.experts * 4));
  std::vector<float> logits;  // host staging of d_logits, alive across the async copy
  uint32_t logits_steps = 0;
  std::map<uint32_t, Round> rounds;  // by token count

  const uint32_t capacity = 4 * f.experts;
  std::vector<uint32_t> entries(static_cast<size_t>(IGNIS_RESIDENCY_LISTS) * capacity);
  std::vector<ignis_moe_slot> table(keys);
  std::vector<uint8_t> record;
  std::vector<uint32_t> staged_prev;  // keys staged by the previous step
  uint32_t prev_layer = UINT32_MAX;
  uint64_t total_bytes = 0, total_prefetch_hits = 0, total_prefetches = 0;
  uint64_t want_hits[8][2] = {}, want_misses[8][2] = {};
  uint32_t graph_rounds = 0;

  // The stall: what steps `from`..`to` (one phase) added to stall_nanos, against their misses. A
  // copying step must add something only where %globaltimer ticks finer than its copy (the 5090
  // reads ~1.6 us for a 4 KB one); elsewhere it may read 0, so a coarse tick is reported, and
  // only each phase's total must grow.
  uint64_t stall_seen[2] = {}, stall_steps[2] = {}, stall_min = UINT64_MAX, stall_zero = 0;
  auto stall_now = [&](uint64_t out[2]) {
    ignis_residency_counters c{};
    RES_OK(ignis_residency_read_counters(r, &c));
    out[0] = c.stall_nanos[0];
    out[1] = c.stall_nanos[1];
  };
  auto check_stall = [&](const std::string &at, size_t from, size_t to, const uint64_t before[2]) {
    uint64_t after[2];
    stall_now(after);
    const uint32_t phase = f.steps[from].phase;
    bool copied = false;
    for (size_t i = from; i < to; ++i) copied = copied || (f.steps[i].expect.status == 0 && !f.steps[i].expect.list[2].empty());
    const uint64_t added = after[phase] - before[phase];
    check(after[1 - phase] == before[1 - phase], at + ": the stall of the other phase moved");
    if (copied) {
      if (added == 0) ++stall_zero;
      stall_min = std::min(stall_min, added);
      stall_seen[phase] += added;
      ++stall_steps[phase];
    } else {
      check(added == 0, at + ": no miss, yet the stall grew by " + std::to_string(added) + " ns");
    }
  };

  // One step's outcome against the model's.
  auto compare = [&](size_t si) {
    const Step &s = f.steps[si];
    const std::string at = "step " + std::to_string(si) + " (layer " + std::to_string(s.layer) +
                           (s.phase ? ", prefill" : ", decode") + ")";
    ignis_residency_report head{};
    RES_OK(ignis_residency_last_report(r, s.layer, &head, entries.data(), capacity));
    check(head.status == s.expect.status,
          at + ": status " + std::to_string(head.status) + ", the model " + std::to_string(s.expect.status));
    if (head.status != 0 || s.expect.status != 0) return;
    static const char *kNames[IGNIS_RESIDENCY_LISTS] = {"hits", "prefetch hits", "misses",
                                                         "evictions", "prefetches", "dropped"};
    for (int l = 0; l < IGNIS_RESIDENCY_LISTS; ++l) {
      std::vector<uint32_t> got(entries.begin() + static_cast<size_t>(l) * capacity,
                                entries.begin() + static_cast<size_t>(l) * capacity + head.count[l]);
      std::vector<uint32_t> want = s.expect.list[l];
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      if (got != want) {
        std::string g, w;
        for (uint32_t k : got) g += " " + std::to_string(k & ~IGNIS_RESIDENCY_STAGING_BIT) + ((k & IGNIS_RESIDENCY_STAGING_BIT) ? "s" : "");
        for (uint32_t k : want) w += " " + std::to_string(k & ~IGNIS_RESIDENCY_STAGING_BIT) + ((k & IGNIS_RESIDENCY_STAGING_BIT) ? "s" : "");
        check(false, at + ": " + kNames[l] + " GPU [" + g + " ] model [" + w + " ]");
      }
    }
    check(head.bytes_moved == s.expect.bytes,
          at + ": bytes " + std::to_string(head.bytes_moved) + ", the model " + std::to_string(s.expect.bytes));
    total_bytes += s.expect.bytes;
    total_prefetch_hits += s.expect.list[1].size();
    total_prefetches += s.expect.list[4].size();
    for (uint32_t k : s.expect.list[0]) ++want_hits[class_of(f, k)][s.phase];
    for (uint32_t k : s.expect.list[2]) ++want_misses[class_of(f, k & ~IGNIS_RESIDENCY_STAGING_BIT)][s.phase];
  };

  // The slot table: every entry ABSENT or inside its class pool on a slot boundary or inside the
  // ring, no two sharing a slot, every live entry holding its own record's bytes. With
  // `si` (an eager step), also that step's evictions and the previous step's staging.
  auto check_table = [&](const std::string &at, const Step *s) {
    CUDA_OK(cudaMemcpy(table.data(), ignis_residency_slot_table(r, 0), keys * sizeof(ignis_moe_slot),
                       cudaMemcpyDeviceToHost));
    std::set<const void *> used;
    uint32_t in_pools[IGNIS_RESIDENCY_CLASSES] = {};
    for (uint32_t key = 0; key < keys; ++key) {
      const ignis_moe_slot &e = table[key];
      if (e.record == nullptr) {
        check(e.k2 == 0, at + ": an ABSENT entry with k2 " + std::to_string(e.k2));
        continue;
      }
      const uint32_t c = class_of(f, key);
      check(e.k2 == f.k2[key], at + ": key " + std::to_string(key) + " has k2 " + std::to_string(e.k2));
      const auto *p = static_cast<const uint8_t *>(e.record);
      const auto *pool = static_cast<const uint8_t *>(layout.pool[c]);
      const auto *ring = static_cast<const uint8_t *>(layout.ring);
      const bool in_pool = pool != nullptr && p >= pool && p < pool + layout.pool_bytes[c] &&
                           (p - pool) % f.record_bytes[c] == 0;
      const bool in_ring = ring != nullptr && p >= ring && p + f.record_bytes[c] <= ring + layout.ring_bytes;
      check(in_pool || in_ring, at + ": key " + std::to_string(key) + " points outside its class pool and the ring");
      if (in_pool) ++in_pools[c];
      check(used.insert(e.record).second, at + ": two entries share " + std::to_string(reinterpret_cast<uintptr_t>(e.record)));
      const uint64_t b = f.record_bytes[c];
      record.resize(b);
      CUDA_OK(cudaMemcpy(record.data(), e.record, b, cudaMemcpyDeviceToHost));
      check(std::memcmp(record.data(), host + offsets[key], b) == 0,
            at + ": key " + std::to_string(key) + " does not hold its record");
    }
    // The occupancy readout counts what the slot table holds in the pools.
    uint32_t occupancy[IGNIS_RESIDENCY_CLASSES];
    RES_OK(ignis_residency_read_occupancy(r, occupancy));
    for (uint32_t c = 0; c < IGNIS_RESIDENCY_CLASSES; ++c) {
      check(occupancy[c] == in_pools[c], at + ": class " + std::to_string(c) + " reads " +
                                             std::to_string(occupancy[c]) + " slots in use, the table holds " +
                                             std::to_string(in_pools[c]));
    }
    if (s == nullptr || s->expect.status != 0) return;
    // Selected: live now.
    for (int l : {0, 2}) {
      for (uint32_t k : s->expect.list[l]) {
        const uint32_t key = k & ~IGNIS_RESIDENCY_STAGING_BIT;
        check(table[key].record != nullptr, at + ": selected key " + std::to_string(key) + " is ABSENT");
      }
    }
    // Evicted: ABSENT, unless the same step brought it back (a miss or a prefetch).
    std::set<uint32_t> back;
    for (int l : {2, 4}) {
      for (uint32_t k : s->expect.list[l]) back.insert(k & ~IGNIS_RESIDENCY_STAGING_BIT);
    }
    for (uint32_t k : s->expect.list[3]) {
      if (back.count(k) == 0) check(table[k].record == nullptr, at + ": evicted key " + std::to_string(k) + " is not ABSENT");
    }
    // Staged by the previous step: released once a step of another layer -- or of the same
    // layer again -- has run, unless this step selected it again.
    if (prev_layer != UINT32_MAX) {
      std::set<uint32_t> now_live;
      for (int l : {0, 2, 4}) {
        for (uint32_t k : s->expect.list[l]) now_live.insert(k & ~IGNIS_RESIDENCY_STAGING_BIT);
      }
      for (uint32_t k : staged_prev) {
        const bool lookahead_for_this = k / nk == s->layer && prev_layer != s->layer;
        if (!lookahead_for_this && now_live.count(k) == 0) {
          check(table[k].record == nullptr, at + ": key " + std::to_string(k) + " staged by the previous step is not ABSENT");
        }
      }
    }
  };
  auto remember_staging = [&](const Step &s) {
    staged_prev.clear();
    if (s.expect.status == 0) {
      for (int l : {2, 4}) {
        for (uint32_t k : s.expect.list[l]) {
          if (k & IGNIS_RESIDENCY_STAGING_BIT) staged_prev.push_back(k & ~IGNIS_RESIDENCY_STAGING_BIT);
        }
      }
    }
    prev_layer = s.layer;
  };
  // A whole decode round from `si`: every layer in order, one token count.
  auto is_round = [&](size_t si) {
    if (si + f.layers > f.steps.size()) return false;
    for (uint32_t l = 0; l < f.layers; ++l) {
      const Step &s = f.steps[si + l];
      if (s.phase != IGNIS_RESIDENCY_DECODE || s.layer != l || s.tokens != f.steps[si].tokens) return false;
      if ((l + 1 < f.layers) != (s.rows > 0) || (s.rows > 0 && (s.rows != s.tokens || s.stride != f.width))) return false;
    }
    return true;
  };

  // Logits whose ranking (BF16, ties to the lower id) is `host` row by row, when every row's
  // first `width` entries are distinct ids: the listed ids at width, width - 1, ..., 1, every
  // other expert at -8.
  auto as_logits = [&](const std::vector<int32_t> &host, uint32_t rows, uint32_t stride) {
    if (stride != f.width || f.width == 0) return false;
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t j = 0; j < f.width; ++j) {
        const int32_t e = host[static_cast<size_t>(row) * stride + j];
        if (e < 0 || static_cast<uint32_t>(e) >= f.experts) return false;
        for (uint32_t i = 0; i < j; ++i) {
          if (host[static_cast<size_t>(row) * stride + i] == e) return false;
        }
      }
    }
    logits.assign(static_cast<size_t>(rows) * f.experts, -8.0f);
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t j = 0; j < f.width; ++j) {
        logits[static_cast<size_t>(row) * f.experts + host[static_cast<size_t>(row) * stride + j]] =
            static_cast<float>(f.width - j);
      }
    }
    CUDA_OK(cudaMemcpyAsync(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice, stream));
    return true;
  };

  // One step, whole or split (the prefetch half on the stream the demand half forked). `host`,
  // the lookahead's host copy, lets an eager split step take the logits variant.
  auto run_step = [&](uint32_t layer, uint32_t phase, const int32_t *ids, uint32_t tokens,
                      const int32_t *look, uint32_t rows, uint32_t stride,
                      const std::vector<int32_t> *host = nullptr) {
    if (!split_mode) {
      RES_OK(ignis_residency_step_ranked(r, layer, phase, ids, tokens, look, rows, stride, stream));
      return;
    }
    const bool from_logits = look != nullptr && host != nullptr && layer + 1 < f.layers &&
                             as_logits(*host, rows, stride);
    void *side = nullptr;
    RES_OK(ignis_residency_step_demand(r, layer, phase, ids, tokens, stream, look ? &side : nullptr));
    const uint32_t width = phase == IGNIS_RESIDENCY_PREFILL ? f.prefill_width : f.width;
    check((side != nullptr) == (look != nullptr && layer + 1 < f.layers && width > 0),
          "split: a branch opens exactly when the step looks ahead");
    if (side != nullptr && from_logits) {
      RES_OK(ignis_residency_step_prefetch(r, d_logits, rows));
      ++logits_steps;
    } else if (side != nullptr) {
      RES_OK(ignis_residency_step_prefetch_ranked(r, look, rows, stride));
    }
  };

  // A capture that fails after a step with a lookahead (ended without the join, so its fork is
  // unjoined) leaves residency able to step: the join reports the dead capture's event once and
  // forgets it, and the fixture then replays from its first step as if nothing had happened --
  // the captured step never ran.
  {
    const Step &s = f.steps[0];
    check(s.rows > 0 && s.layer + 1 < f.layers, "the fixture's first step has a lookahead");
    CUDA_OK(cudaMemcpy(d_ids, s.ids.data(), s.ids.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_look, s.lookahead.data(), s.lookahead.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    run_step(s.layer, s.phase, d_ids, s.tokens, d_look, s.rows, s.stride);
    cudaGraph_t g = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &g);
    check(ended != cudaSuccess, "a capture with an unjoined prefetch fork must fail to end");
    if (g) cudaGraphDestroy(g);
    cudaGetLastError();  // the capture's error is not the test's
    // May report the dead event; must forget it either way, and leave no last error behind for
    // the next step's launch check to read as its own.
    const int32_t joined = ignis_residency_join(r, stream);
    std::printf("join after the failed capture: %s\n", joined == 0 ? "ok" : ignis_residency_last_error());
    check(cudaPeekAtLastError() == cudaSuccess,
          std::string("the join leaves a last error behind: ") + cudaGetErrorString(cudaPeekAtLastError()));
    check(ignis_residency_join(r, stream) == 0, std::string("a second join still fails: ") + ignis_residency_last_error());
    // A normal step now runs; the replay below starts from this state.
    CUDA_OK(cudaStreamSynchronize(stream));
  }

  size_t si = 0;
  while (si < f.steps.size() && g_failures <= 40) {
    if (graph_mode && is_round(si)) {
      const uint32_t tokens = f.steps[si].tokens;
      Round &round = rounds[tokens];
      if (round.exec == nullptr) {
        for (uint32_t l = 0; l < f.layers; ++l) {
          int32_t *ids = nullptr, *look = nullptr;
          CUDA_OK(cudaMalloc(&ids, static_cast<size_t>(tokens) * f.top_k * 4));
          if (l + 1 < f.layers) CUDA_OK(cudaMalloc(&look, static_cast<size_t>(tokens) * f.width * 4));
          round.ids.push_back(ids);
          round.lookahead.push_back(look);
        }
        RES_OK(ignis_residency_join(r, stream));  // nothing outstanding enters the capture
        cudaGraph_t g = nullptr;
        CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (uint32_t l = 0; l < f.layers; ++l) {
          run_step(l, IGNIS_RESIDENCY_DECODE, round.ids[l], tokens, round.lookahead[l],
                   round.lookahead[l] ? tokens : 0, round.lookahead[l] ? f.width : 0);
        }
        RES_OK(ignis_residency_join(r, stream));
        CUDA_OK(cudaStreamEndCapture(stream, &g));
        CUDA_OK(cudaGraphInstantiate(&round.exec, g, 0));
        CUDA_OK(cudaGraphDestroy(g));
      }
      RES_OK(ignis_residency_join(r, stream));
      uint64_t before[2];
      stall_now(before);
      for (uint32_t l = 0; l < f.layers; ++l) {
        const Step &s = f.steps[si + l];
        CUDA_OK(cudaMemcpyAsync(round.ids[l], s.ids.data(), s.ids.size() * 4, cudaMemcpyHostToDevice, stream));
        if (round.lookahead[l]) {
          CUDA_OK(cudaMemcpyAsync(round.lookahead[l], s.lookahead.data(), s.lookahead.size() * 4,
                                  cudaMemcpyHostToDevice, stream));
        }
      }
      CUDA_OK(cudaGraphLaunch(round.exec, stream));
      CUDA_OK(cudaStreamSynchronize(stream));
      for (uint32_t l = 0; l < f.layers; ++l) compare(si + l);
      check_stall("round from step " + std::to_string(si), si, si + f.layers, before);
      check_table("round from step " + std::to_string(si), nullptr);
      if (f.steps[si + f.layers - 1].expect.status == 0) check_mirror("round from step " + std::to_string(si));
      remember_staging(f.steps[si + f.layers - 1]);
      ++graph_rounds;
      si += f.layers;
      continue;
    }
    const Step &s = f.steps[si];
    const bool has_look = s.rows > 0;
    uint64_t before[2];
    stall_now(before);
    CUDA_OK(cudaMemcpyAsync(d_ids, s.ids.data(), s.ids.size() * 4, cudaMemcpyHostToDevice, stream));
    if (has_look) {
      CUDA_OK(cudaMemcpyAsync(d_look, s.lookahead.data(), s.lookahead.size() * 4, cudaMemcpyHostToDevice, stream));
    }
    run_step(s.layer, s.phase, d_ids, s.tokens, has_look ? d_look : nullptr, s.rows, s.stride, &s.lookahead);
    RES_OK(ignis_residency_join(r, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    compare(si);
    check_stall("step " + std::to_string(si), si, si + 1, before);
    check_table("step " + std::to_string(si) + " (layer " + std::to_string(s.layer) + ")", &s);
    if (s.layer + 1 == f.layers && s.expect.status == 0) check_mirror("step " + std::to_string(si));
    remember_staging(s);
    ++si;
  }

  ignis_residency_counters counters{};
  RES_OK(ignis_residency_read_counters(r, &counters));
  uint64_t moved = counters.bytes_moved[0] + counters.bytes_moved[1];
  check(moved == total_bytes, "counters: bytes moved " + std::to_string(moved) + ", the trace " + std::to_string(total_bytes));
  check(counters.prefetch_used == total_prefetch_hits, "counters: prefetches used");
  check(counters.prefetch_issued == total_prefetches, "counters: prefetches issued");
  for (int c = 0; c < 8; ++c) {
    for (int p = 0; p < 2; ++p) {
      check(counters.hits[c][p] == want_hits[c][p], "counters: hits of class " + std::to_string(c));
      check(counters.misses[c][p] == want_misses[c][p], "counters: misses of class " + std::to_string(c));
    }
  }
  if (graph_mode) check(graph_rounds > 2, "graph mode replayed too few captured rounds");
  check(counters.stall_nanos[0] == stall_seen[0] && counters.stall_nanos[1] == stall_seen[1],
        "counters: the stall is what the steps added");
  check(stall_steps[0] > 0 && stall_steps[1] > 0, "the trace copies misses in both phases");
  check(stall_seen[0] > 0 && stall_seen[1] > 0, "the copies of each phase add up to some stall");
  std::printf("stall: decode %.1f us over %llu copying steps, prefill %.1f us over %llu, smallest %llu ns%s\n",
              stall_seen[0] / 1e3, static_cast<unsigned long long>(stall_steps[0]), stall_seen[1] / 1e3,
              static_cast<unsigned long long>(stall_steps[1]), static_cast<unsigned long long>(stall_min),
              stall_zero ? (" -- " + std::to_string(stall_zero) + " copying steps read 0 ns: a coarse %globaltimer").c_str()
                         : "");

  if (split_mode) {
    // A prefetch half needs its demand half's open branch; a branch the caller never took a
    // prefetch half on still joins.
    check(ignis_residency_step_prefetch_ranked(r, d_look, 1, f.width) != 0,
          "a prefetch half with no open branch must be refused");
    const std::vector<int32_t> first(f.steps[0].ids.begin(), f.steps[0].ids.begin() + f.top_k);
    CUDA_OK(cudaMemcpy(d_ids, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
    void *side = nullptr;
    RES_OK(ignis_residency_step_demand(r, 0, IGNIS_RESIDENCY_DECODE, d_ids, 1, stream, &side));
    check(side != nullptr, "a split step of layer 0 opens a branch");
    RES_OK(ignis_residency_join(r, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    check(ignis_residency_step_prefetch_ranked(r, d_look, 1, f.width) != 0,
          "the join closes the branch");

    std::printf("split: %u prefetch halves from logits\n", logits_steps);
    check(logits_steps > 0, "split: some prefetch halves ran from logits");

    // A split step's ranked lookahead with an id outside [0, experts): the demand half stands,
    // nothing is prefetched, and the report says INVALID over the demand half's lists.
    std::set<int32_t> distinct(first.begin(), first.end());
    std::vector<int32_t> bad(f.width, static_cast<int32_t>(f.experts));
    CUDA_OK(cudaMemcpy(d_look, bad.data(), bad.size() * 4, cudaMemcpyHostToDevice));
    RES_OK(ignis_residency_step_demand(r, 0, IGNIS_RESIDENCY_PREFILL, d_ids, 1, stream, &side));
    RES_OK(ignis_residency_step_prefetch_ranked(r, d_look, 1, f.width));
    RES_OK(ignis_residency_join(r, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    ignis_residency_report head{};
    RES_OK(ignis_residency_last_report(r, 0, &head, entries.data(), capacity));
    check(head.status == IGNIS_RESIDENCY_STATUS_INVALID, "split: an out-of-range lookahead id reports INVALID");
    check(head.count[2] + head.count[0] == 2 * distinct.size() && head.count[4] == 0 && head.count[5] == 0,
          "split: the INVALID report holds the demand half's hits and misses, and no prefetch");
  }

  // The last layer has no next layer: a whole step there ignores its lookahead, as the policy
  // does, an out-of-range id included (a prefill step, so no class refuses it).
  {
    const std::vector<int32_t> first(f.steps[0].ids.begin(), f.steps[0].ids.begin() + f.top_k);
    const std::vector<int32_t> bad(f.width, static_cast<int32_t>(f.experts));
    CUDA_OK(cudaMemcpy(d_ids, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_look, bad.data(), bad.size() * 4, cudaMemcpyHostToDevice));
    RES_OK(ignis_residency_step_ranked(r, f.layers - 1, IGNIS_RESIDENCY_PREFILL, d_ids, 1, d_look, 1, f.width,
                                       stream));
    RES_OK(ignis_residency_join(r, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    ignis_residency_report head{};
    RES_OK(ignis_residency_last_report(r, f.layers - 1, &head, entries.data(), capacity));
    check(head.status == 0, "a last-layer step ignores its lookahead, out-of-range ids included (status " +
                                std::to_string(head.status) + ")");
    // That step wrote the mirror: the totals the after-free check compares it with.
    RES_OK(ignis_residency_read_counters(r, &counters));
    moved = counters.bytes_moved[0] + counters.bytes_moved[1];
  }

  for (auto &[k, round] : rounds) {
    if (round.exec) cudaGraphExecDestroy(round.exec);
    for (auto *p : round.ids) cudaFree(p);
    for (auto *p : round.lookahead) cudaFree(p);
  }
  cudaFree(d_ids);
  cudaFree(d_look);
  cudaFree(d_logits);
  cudaStreamDestroy(stream);
  ignis_residency_free(r);
  // Unregistered by free, still the caller's: the last step's totals stay readable.
  if (f.steps.back().layer + 1 == f.layers && f.steps.back().expect.status == 0) {
    const uint64_t final_moved = mirror->counters.bytes_moved[0] + mirror->counters.bytes_moved[1];
    check(final_moved == moved, "the mirror after free: bytes moved " + std::to_string(final_moved) +
                                    ", the device " + std::to_string(moved));
  }

  // A prefetch that evicts a later candidate of the same lookahead: the candidate is no longer
  // resident at its turn, so it is prefetched again, as the policy walks them one at a time.
  // Two layers of 16 experts, one class per projection with 12 slots: the warm start holds
  // layer 1's experts B (hotter) and A; a decode step of layer 0 misses ten experts, which fills
  // both classes, and looks ahead at [X, A]. X evicts the least recent, A; A then evicts B.
  {
    constexpr uint32_t kE = 16, kA = 12, kB = 13, kX = 14;
    auto key = [](uint32_t layer, uint32_t expert, uint32_t projection) { return (layer * kE + expert) * 2 + projection; };
    ignis_residency_desc d2{};
    d2.layers = 2;
    d2.experts = kE;
    for (int c = 0; c < 8; ++c) d2.record_bytes[c] = 4096;
    d2.capacity[0] = d2.capacity[4] = 12;  // gate/up and down at K = 2
    d2.max_tokens = 1;
    d2.lookahead_width = 2;
    d2.prefill_lookahead_width = 2;
    d2.prefetch_budget_bytes = IGNIS_RESIDENCY_NO_BUDGET;
    d2.staging_half_bytes = 2 * kE * 4096;
    d2.host_pool_bytes = 2 * 2 * kE * 4096;
    d2.copy_blocks = 4;
    d2.report = 1;
    const std::vector<uint8_t> k2(2 * 2 * kE, 4);
    std::vector<uint64_t> offsets2(2 * 2 * kE);
    for (size_t i = 0; i < offsets2.size(); ++i) offsets2[i] = i * 4096;
    ignis_residency *r2 = nullptr;
    RES_OK(ignis_residency_create(&d2, k2.data(), offsets2.data(), &r2));
    const std::vector<uint32_t> warm = {key(1, kB, 0), key(1, kB, 1), key(1, kA, 0), key(1, kA, 1)};
    uint32_t warmed = 0;
    RES_OK(ignis_residency_warm_start(r2, warm.data(), static_cast<uint32_t>(warm.size()), &warmed));
    check(warmed == 4, "eviction check: the warm start admits A and B");
    cudaStream_t s2;
    CUDA_OK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    std::vector<int32_t> ids2(IGNIS_MOE_TOP_K);
    for (uint32_t i = 0; i < IGNIS_MOE_TOP_K; ++i) ids2[i] = static_cast<int32_t>(i);
    const std::vector<int32_t> look2 = {static_cast<int32_t>(kX), static_cast<int32_t>(kA)};
    int32_t *d_ids2 = nullptr, *d_look2 = nullptr;
    CUDA_OK(cudaMalloc(&d_ids2, ids2.size() * 4));
    CUDA_OK(cudaMalloc(&d_look2, look2.size() * 4));
    CUDA_OK(cudaMemcpy(d_ids2, ids2.data(), ids2.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_look2, look2.data(), look2.size() * 4, cudaMemcpyHostToDevice));
    if (split_mode) {
      void *side = nullptr;
      RES_OK(ignis_residency_step_demand(r2, 0, IGNIS_RESIDENCY_DECODE, d_ids2, 1, s2, &side));
      RES_OK(ignis_residency_step_prefetch_ranked(r2, d_look2, 1, 2));
    } else {
      RES_OK(ignis_residency_step_ranked(r2, 0, IGNIS_RESIDENCY_DECODE, d_ids2, 1, d_look2, 1, 2, s2));
    }
    RES_OK(ignis_residency_join(r2, s2));
    CUDA_OK(cudaStreamSynchronize(s2));
    ignis_residency_report head{};
    std::vector<uint32_t> lists(static_cast<size_t>(IGNIS_RESIDENCY_LISTS) * 4 * kE);
    RES_OK(ignis_residency_last_report(r2, 0, &head, lists.data(), 4 * kE));
    auto list = [&](int l) {
      std::set<uint32_t> out(lists.begin() + static_cast<size_t>(l) * 4 * kE,
                             lists.begin() + static_cast<size_t>(l) * 4 * kE + head.count[l]);
      return out;
    };
    check(head.status == 0, "eviction check: status " + std::to_string(head.status));
    check(list(4) == std::set<uint32_t>{key(1, kX, 0), key(1, kX, 1), key(1, kA, 0), key(1, kA, 1)},
          "eviction check: X and then the A it evicted are prefetched");
    check(list(3) == std::set<uint32_t>{key(1, kA, 0), key(1, kA, 1), key(1, kB, 0), key(1, kB, 1)},
          "eviction check: X evicts A, A evicts B");
    std::vector<ignis_moe_slot> t2(2 * kE);
    CUDA_OK(cudaMemcpy(t2.data(), ignis_residency_slot_table(r2, 1), t2.size() * sizeof(ignis_moe_slot),
                       cudaMemcpyDeviceToHost));
    check(t2[2 * kA].record != nullptr && t2[2 * kA + 1].record != nullptr, "eviction check: A is resident");
    check(t2[2 * kB].record == nullptr && t2[2 * kB + 1].record == nullptr, "eviction check: B is ABSENT");

    // A prefill width of 0 is no lookahead: a prefill step opens no branch and prefetches nothing
    // from a whole step's lookahead, while a decode step of the same residency still looks ahead.
    {
      ignis_residency_desc d3 = d2;
      d3.prefill_lookahead_width = 0;
      ignis_residency *r3 = nullptr;
      RES_OK(ignis_residency_create(&d3, k2.data(), offsets2.data(), &r3));
      void *side = nullptr;
      RES_OK(ignis_residency_step_demand(r3, 0, IGNIS_RESIDENCY_PREFILL, d_ids2, 1, s2, &side));
      check(side == nullptr, "a prefill width of 0 opens no lookahead branch");
      RES_OK(ignis_residency_step_ranked(r3, 0, IGNIS_RESIDENCY_PREFILL, d_ids2, 1, d_look2, 1, 2, s2));
      RES_OK(ignis_residency_join(r3, s2));
      CUDA_OK(cudaStreamSynchronize(s2));
      ignis_residency_report h3{};
      RES_OK(ignis_residency_last_report(r3, 0, &h3, lists.data(), 4 * kE));
      check(h3.status == 0 && h3.count[4] == 0, "a prefill width of 0 prefetches nothing");
      RES_OK(ignis_residency_step_demand(r3, 0, IGNIS_RESIDENCY_DECODE, d_ids2, 1, s2, &side));
      check(side != nullptr, "a decode step of the same residency still looks ahead");
      RES_OK(ignis_residency_join(r3, s2));
      CUDA_OK(cudaStreamSynchronize(s2));
      ignis_residency_free(r3);
    }
    cudaFree(d_ids2);
    cudaFree(d_look2);
    cudaStreamDestroy(s2);
    ignis_residency_free(r2);
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("residency trace: %zu steps equal the policy model%s%s (warm start admitted %u, %u captured rounds)\n",
              f.steps.size(), graph_mode ? " through graphs" : "", split_mode ? " as split steps" : "", admitted,
              graph_rounds);
  return 0;
}
