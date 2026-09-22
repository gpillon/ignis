// ignis kernel leaf: attention-input tap -- test-only diagnostic seam
// (docs/findings/2026-09-21-one-attention-head-points.md). See
// kernel/include/ignis_attn_tap.h for what it captures, what it does not
// (the keys are the ones before the cache), and why its disarmed cost -- one
// flag load per GQA layer call -- is the only part of it on the production
// path.
//
// Style follows kv_capture.cu: explicit pointers and sizes, int32 return
// codes (0 ok, -1 error via ignis_attn_tap_last_error), validate before
// touching the GPU, and name the value that failed.

#include "ignis_attn_tap.h"
#include "ignis_gqa_workspace.h"
#include "ignis_seq_internal.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kQHeads = kIgnisGqaQHeads;
constexpr std::int32_t kKvHeads = 4;
constexpr std::int32_t kHeadDim = 256;
constexpr std::int64_t kQRow = static_cast<std::int64_t>(kQHeads) * kHeadDim;
constexpr std::int64_t kKRow = static_cast<std::int64_t>(kKvHeads) * kHeadDim;

thread_local std::string g_last_error;

void set_error(std::string message) {
  g_last_error = std::move(message);
}

struct Tap {
  std::atomic<bool> armed{false};
  std::int32_t slot_of[kIgnisGqaLayerCount];
  std::int32_t n_layers = 0;
  std::vector<std::int64_t> queries;
  std::int64_t max_positions = 0;
  std::uint16_t *q_out = nullptr;
  std::uint16_t *k_out = nullptr;
  std::vector<std::int64_t> rows_written;         // [slot]
  std::vector<std::uint8_t> seen;                 // [slot * n_queries + query]
  std::uint16_t *kc_out = nullptr;                // consumed keys, or null
  std::vector<std::int64_t> consumed_rows;        // [slot]
  std::vector<std::int64_t> consumed_start;       // [slot], -1 = none
};

Tap g_tap;

} // namespace

extern "C" int32_t ignis_attn_tap_arm(const int32_t *gqa_ordinals, int32_t n_layers,
                                      const int64_t *query_positions, int32_t n_queries,
                                      int64_t max_positions, uint16_t *q_out, uint16_t *k_out) {
  if (gqa_ordinals == nullptr || q_out == nullptr || k_out == nullptr ||
      (n_queries > 0 && query_positions == nullptr)) {
    set_error("ignis_attn_tap_arm: null argument");
    return -1;
  }
  if (n_layers <= 0 || n_layers > kIgnisGqaLayerCount) {
    set_error("ignis_attn_tap_arm: n_layers is " + std::to_string(n_layers) +
              ", expected 1.." + std::to_string(kIgnisGqaLayerCount));
    return -1;
  }
  if (n_queries < 0) {
    set_error("ignis_attn_tap_arm: n_queries is negative: " + std::to_string(n_queries));
    return -1;
  }
  if (max_positions <= 0) {
    set_error("ignis_attn_tap_arm: max_positions must be positive, got " +
              std::to_string(max_positions));
    return -1;
  }
  std::int32_t slot_of[kIgnisGqaLayerCount];
  for (auto &slot : slot_of) {
    slot = -1;
  }
  for (std::int32_t i = 0; i < n_layers; ++i) {
    const std::int32_t ordinal = gqa_ordinals[i];
    if (ordinal < 0 || ordinal >= kIgnisGqaLayerCount) {
      set_error("ignis_attn_tap_arm: GQA ordinal " + std::to_string(ordinal) +
                " is outside 0.." + std::to_string(kIgnisGqaLayerCount - 1));
      return -1;
    }
    if (slot_of[ordinal] != -1) {
      set_error("ignis_attn_tap_arm: GQA ordinal " + std::to_string(ordinal) +
                " is listed twice");
      return -1;
    }
    slot_of[ordinal] = i;
  }
  for (std::int32_t q = 0; q < n_queries; ++q) {
    if (query_positions[q] < 0) {
      set_error("ignis_attn_tap_arm: query position " + std::to_string(q) + " is negative: " +
                std::to_string(query_positions[q]));
      return -1;
    }
  }

  g_tap.armed.store(false, std::memory_order_relaxed);
  for (std::int32_t i = 0; i < kIgnisGqaLayerCount; ++i) {
    g_tap.slot_of[i] = slot_of[i];
  }
  g_tap.n_layers = n_layers;
  g_tap.queries.assign(query_positions, query_positions + n_queries);
  g_tap.max_positions = max_positions;
  g_tap.q_out = q_out;
  g_tap.k_out = k_out;
  g_tap.rows_written.assign(static_cast<std::size_t>(n_layers), 0);
  g_tap.seen.assign(static_cast<std::size_t>(n_layers) * static_cast<std::size_t>(n_queries), 0);
  g_tap.kc_out = nullptr;
  g_tap.consumed_rows.assign(static_cast<std::size_t>(n_layers), 0);
  g_tap.consumed_start.assign(static_cast<std::size_t>(n_layers), -1);
  g_tap.armed.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t ignis_attn_tap_arm_consumed(uint16_t *kc_out) {
  if (kc_out == nullptr) {
    set_error("ignis_attn_tap_arm_consumed: null argument");
    return -1;
  }
  if (!g_tap.armed.load(std::memory_order_acquire)) {
    set_error("ignis_attn_tap_arm_consumed: arm the tap first");
    return -1;
  }
  if (g_tap.queries.empty()) {
    set_error("ignis_attn_tap_arm_consumed: the arm names no query position");
    return -1;
  }
  g_tap.kc_out = kc_out;
  return 0;
}

extern "C" int32_t ignis_attn_tap_disarm(int64_t *rows_written, int32_t *queries_seen,
                                         int64_t *consumed_rows, int64_t *consumed_chunk_start) {
  const bool was_armed = g_tap.armed.exchange(false, std::memory_order_acq_rel);
  if (!was_armed) {
    set_error("ignis_attn_tap_disarm: the tap was not armed");
    return -1;
  }
  if (rows_written != nullptr) {
    for (std::int32_t i = 0; i < g_tap.n_layers; ++i) {
      rows_written[i] = g_tap.rows_written[static_cast<std::size_t>(i)];
    }
  }
  if (queries_seen != nullptr) {
    const auto n_queries = static_cast<std::int32_t>(g_tap.queries.size());
    for (std::int32_t q = 0; q < n_queries; ++q) {
      std::int32_t all = 1;
      for (std::int32_t i = 0; i < g_tap.n_layers; ++i) {
        if (g_tap.seen[static_cast<std::size_t>(i) * n_queries + q] == 0) {
          all = 0;
        }
      }
      queries_seen[q] = all;
    }
  }
  if (consumed_rows != nullptr) {
    for (std::int32_t i = 0; i < g_tap.n_layers; ++i) {
      consumed_rows[i] = g_tap.consumed_rows[static_cast<std::size_t>(i)];
    }
  }
  if (consumed_chunk_start != nullptr) {
    for (std::int32_t i = 0; i < g_tap.n_layers; ++i) {
      consumed_chunk_start[i] = g_tap.consumed_start[static_cast<std::size_t>(i)];
    }
  }
  g_tap.q_out = nullptr;
  g_tap.k_out = nullptr;
  g_tap.kc_out = nullptr;
  return 0;
}

extern "C" const char *ignis_attn_tap_last_error(void) {
  return g_last_error.c_str();
}

int32_t ignis_attn_tap_record(uint32_t gqa_ordinal, int64_t start_position, int32_t tokens,
                              const void *rotated_query, const void *rotated_key,
                              cudaStream_t stream) {
  if (!g_tap.armed.load(std::memory_order_acquire)) {
    return 0;
  }
  if (gqa_ordinal >= static_cast<uint32_t>(kIgnisGqaLayerCount)) {
    return 0;
  }
  const std::int32_t slot = g_tap.slot_of[gqa_ordinal];
  if (slot < 0) {
    return 0;
  }
  if (tokens <= 0 || start_position < 0) {
    set_error("ignis_attn_tap_record: a chunk of " + std::to_string(tokens) +
              " tokens at position " + std::to_string(start_position));
    return -1;
  }
  const std::int64_t end = start_position + tokens;
  if (end > g_tap.max_positions) {
    set_error("ignis_attn_tap_record: GQA ordinal " + std::to_string(gqa_ordinal) +
              " writes key rows up to position " + std::to_string(end) +
              ", past the armed max_positions " + std::to_string(g_tap.max_positions));
    return -1;
  }

  // Keys: the whole chunk, straight into its positions.
  std::uint16_t *k_dst =
      g_tap.k_out + (static_cast<std::int64_t>(slot) * g_tap.max_positions + start_position) * kKRow;
  cudaError_t err = cudaMemcpyAsync(k_dst, rotated_key,
                                    static_cast<std::size_t>(tokens) * kKRow * sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_attn_tap_record: cudaMemcpyAsync(keys) failed: ") +
              cudaGetErrorString(err));
    return -1;
  }

  // Queries: only the rows asked for that fall in this chunk.
  const auto n_queries = static_cast<std::int64_t>(g_tap.queries.size());
  for (std::int64_t q = 0; q < n_queries; ++q) {
    const std::int64_t position = g_tap.queries[static_cast<std::size_t>(q)];
    if (position < start_position || position >= end) {
      continue;
    }
    const auto *src = static_cast<const std::uint16_t *>(rotated_query) +
                      (position - start_position) * kQRow;
    std::uint16_t *q_dst = g_tap.q_out + (static_cast<std::int64_t>(slot) * n_queries + q) * kQRow;
    err = cudaMemcpyAsync(q_dst, src, static_cast<std::size_t>(kQRow) * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_attn_tap_record: cudaMemcpyAsync(query) failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
    g_tap.seen[static_cast<std::size_t>(slot) * static_cast<std::size_t>(n_queries) +
               static_cast<std::size_t>(q)] = 1;
  }

  // Synchronous by design. Stream order already makes these copies read the
  // rows before anything later on this stream overwrites the scratch they
  // live in; what it does not do is finish the copies into the caller's
  // (pageable) host buffers before this function reports the rows as seen.
  // A test that reads the buffers after disarm must find them complete.
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_attn_tap_record: cudaStreamSynchronize failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  g_tap.rows_written[static_cast<std::size_t>(slot)] += tokens;
  return 0;
}

int32_t ignis_attn_tap_record_consumed(uint32_t gqa_ordinal, int64_t start_position,
                                       int32_t tokens, bool hq_prompt, const void *workspace,
                                       int64_t span, cudaStream_t stream) {
  if (!g_tap.armed.load(std::memory_order_acquire) || g_tap.kc_out == nullptr) {
    return 0;
  }
  if (gqa_ordinal >= static_cast<uint32_t>(kIgnisGqaLayerCount)) {
    return 0;
  }
  const std::int32_t slot = g_tap.slot_of[gqa_ordinal];
  if (slot < 0) {
    return 0;
  }
  const std::int64_t end = start_position + tokens;
  const std::int64_t query = g_tap.queries.front();
  if (query < start_position || query >= end) {
    return 0;  // not the chunk whose attention the query row belongs to
  }
  // Only the hq prompt route materializes a scratch plane, and only a single
  // band of it holds the whole history after the call.
  if (!hq_prompt || workspace == nullptr || span < end) {
    return 0;
  }
  if (end > g_tap.max_positions) {
    set_error("ignis_attn_tap_record_consumed: " + std::to_string(end) +
              " consumed rows exceed the armed max_positions " +
              std::to_string(g_tap.max_positions));
    return -1;
  }
  // scratch_k is [kv_head][span][256]; the capture is [position][kv_head][256].
  const auto *base = static_cast<const std::uint16_t *>(workspace);
  for (std::int32_t head = 0; head < kKvHeads; ++head) {
    std::uint16_t *dst = g_tap.kc_out +
                         (static_cast<std::int64_t>(slot) * g_tap.max_positions * kKvHeads + head) *
                             kHeadDim;
    const std::uint16_t *src = base + static_cast<std::int64_t>(head) * span * kHeadDim;
    const cudaError_t err = cudaMemcpy2DAsync(
        dst, static_cast<std::size_t>(kKRow) * sizeof(std::uint16_t), src,
        static_cast<std::size_t>(kHeadDim) * sizeof(std::uint16_t),
        static_cast<std::size_t>(kHeadDim) * sizeof(std::uint16_t),
        static_cast<std::size_t>(end), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      set_error(std::string("ignis_attn_tap_record_consumed: cudaMemcpy2DAsync failed: ") +
                cudaGetErrorString(err));
      return -1;
    }
  }
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    set_error(std::string("ignis_attn_tap_record_consumed: cudaStreamSynchronize failed: ") +
              cudaGetErrorString(err));
    return -1;
  }
  g_tap.consumed_rows[static_cast<std::size_t>(slot)] = end;
  g_tap.consumed_start[static_cast<std::size_t>(slot)] = start_position;
  return 0;
}
