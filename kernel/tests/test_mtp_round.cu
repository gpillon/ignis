// GitHub #621 (spec #618): the 27B MTP head's verify round -- OURS. The
// round's three small device steps (kernel/src/mtp_head.h `detail`) and the
// rows it touches (`round_rows`), against a host restatement:
//
// - `chain_rows`: chain step s writes lane b's row base_b - 1 + s exactly
//   when the lane needs it (s < extent_b), and the masked form's empty row
//   (position 0, valid 0) otherwise -- a lane at extent 0 writes nothing,
//   and no lane writes past its extent, so none writes past its budget or
//   its context; on a vision load (GitHub #624) the row rotates at the
//   lane's first verify column's rotation position (base + rope delta) less
//   one, plus s, read from that column alone, while its row stays the KV
//   position;
// - `write_drafts`: step s's choices land at drafts[b * window + s];
// - `commit_hiddens`: lane b's column 0 is its slot's continuation hidden,
//   column j > 0 the round's normalized column j - 1;
// - `round_rows`: the chain's rows [P - 1, P - 1 + e), the commit's
//   [P - 1, P - 1 + c), at every extent and committed count a round allows.
//
// The head's forward itself is held to its weights in double precision on
// the real artifact (crates/core/tests/mtp_head_gpu.rs) and the round end to
// end by crates/core/tests/mtp_round_gpu.rs.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE, so a missing or
// busy GPU or a kernel error fails this test.

#include "mtp_head.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void expect(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

void cuda(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "FATAL: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

template <class T>
T *device_copy(const std::vector<T> &host) {
  T *device = nullptr;
  cuda(cudaMalloc(&device, host.size() * sizeof(T)), "cudaMalloc");
  cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  return device;
}

template <class T>
std::vector<T> host_copy(const T *device, std::size_t count) {
  std::vector<T> host(count);
  cuda(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  return host;
}

void chain_rows_follow_each_lanes_extent() {
  constexpr std::int32_t kWindow = 7;
  // Lanes at extent 0 (a budget or a context cut), 1, a middle one and the
  // full window, at frontiers where a row past the extent would cross a page.
  const std::vector<std::int32_t> base = {10, 64, 127, 300, 1};
  const std::vector<std::int32_t> extents = {0, 1, 4, kWindow, 2};
  const auto batch = static_cast<std::int32_t>(base.size());
  std::int32_t *d_base = device_copy(base);
  std::int32_t *d_extents = device_copy(extents);
  std::int32_t *d_positions = device_copy(std::vector<std::int32_t>(batch, -7));
  std::int32_t *d_valid = device_copy(std::vector<std::int32_t>(batch, -7));
  for (std::int32_t step = 0; step < kWindow; ++step) {
    ignis::mtp::detail::chain_rows(d_base, d_extents, step, d_positions, d_valid, batch, nullptr);
    cuda(cudaDeviceSynchronize(), "chain_rows");
    const auto positions = host_copy(d_positions, batch);
    const auto valid = host_copy(d_valid, batch);
    for (std::int32_t b = 0; b < batch; ++b) {
      const bool needed = step < extents[b];
      const std::string at = "step " + std::to_string(step) + " lane " + std::to_string(b);
      expect(valid[b] == (needed ? 1 : 0), at + ": valid " + std::to_string(valid[b]));
      expect(positions[b] == (needed ? base[b] - 1 + step : 0), at + ": row " + std::to_string(positions[b]));
      // A needed row never reaches the lane's anchor-tail position, the
      // last its context grants.
      if (needed) {
        expect(positions[b] < base[b] + extents[b] - 1, at + ": row past the lane's extent");
      }
    }
  }
  cudaFree(d_base);
  cudaFree(d_extents);
  cudaFree(d_positions);
  cudaFree(d_valid);
}

// GitHub #624: lanes whose rope deltas are 0, negative (an image's grid
// takes fewer positions than tokens) and positive; the verify columns past
// each lane's first hold a sentinel, which the chain must never read.
void chain_rows_rotate_at_each_lanes_rope_delta() {
  constexpr std::int32_t kWindow = 3;
  constexpr std::int32_t kColumns = kWindow + 1;
  const std::vector<std::int32_t> base = {10, 64, 600, 300};
  const std::vector<std::int32_t> extents = {kWindow, 2, 0, 1};
  // Each lane's first column: base + delta, deltas 0, -37, -500, +12.
  const std::vector<std::int32_t> first_column = {10, 27, 100, 312};
  const auto batch = static_cast<std::int32_t>(base.size());
  std::vector<std::int32_t> rope(static_cast<std::size_t>(kColumns * batch), -9999);
  for (std::int32_t b = 0; b < batch; ++b) {
    rope[static_cast<std::size_t>(b * kColumns)] = first_column[b];
  }
  // What each step's rotation is, by lane: needed rows only, 0 otherwise.
  const std::int32_t expected[kWindow][4] = {{9, 26, 0, 311}, {10, 27, 0, 0}, {11, 0, 0, 0}};
  std::int32_t *d_base = device_copy(base);
  std::int32_t *d_extents = device_copy(extents);
  std::int32_t *d_rope = device_copy(rope);
  std::int32_t *d_positions = device_copy(std::vector<std::int32_t>(batch, -7));
  std::int32_t *d_valid = device_copy(std::vector<std::int32_t>(batch, -7));
  std::int32_t *d_rotation = device_copy(std::vector<std::int32_t>(batch, -7));
  for (std::int32_t step = 0; step < kWindow; ++step) {
    ignis::mtp::detail::chain_rows(d_base, d_extents, step, d_positions, d_valid, batch, nullptr, d_rope, kColumns,
                                   d_rotation);
    cuda(cudaDeviceSynchronize(), "chain_rows with rope positions");
    const auto positions = host_copy(d_positions, batch);
    const auto rotation = host_copy(d_rotation, batch);
    for (std::int32_t b = 0; b < batch; ++b) {
      const bool needed = step < extents[b];
      const std::string at = "rope step " + std::to_string(step) + " lane " + std::to_string(b);
      expect(rotation[b] == expected[step][b], at + ": rotation " + std::to_string(rotation[b]));
      expect(positions[b] == (needed ? base[b] - 1 + step : 0), at + ": row " + std::to_string(positions[b]));
    }
  }
  cudaFree(d_base);
  cudaFree(d_extents);
  cudaFree(d_rope);
  cudaFree(d_positions);
  cudaFree(d_valid);
  cudaFree(d_rotation);
}

void each_steps_choices_land_in_the_lanes_draft_column() {
  constexpr std::int32_t kWindow = 5;
  constexpr std::int32_t kBatch = 8;
  std::int32_t *d_drafts = device_copy(std::vector<std::int32_t>(kWindow * kBatch, -1));
  std::vector<std::int32_t> tokens(kBatch);
  std::int32_t *d_tokens = device_copy(tokens);
  for (std::int32_t step = 0; step < kWindow; ++step) {
    for (std::int32_t b = 0; b < kBatch; ++b) {
      tokens[b] = 1000 * b + step;
    }
    cuda(cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice), "tokens");
    ignis::mtp::detail::write_drafts(d_tokens, d_drafts, kWindow, step, kBatch, nullptr);
  }
  cuda(cudaDeviceSynchronize(), "write_drafts");
  const auto drafts = host_copy(d_drafts, kWindow * kBatch);
  for (std::int32_t b = 0; b < kBatch; ++b) {
    for (std::int32_t step = 0; step < kWindow; ++step) {
      expect(drafts[b * kWindow + step] == 1000 * b + step,
             "lane " + std::to_string(b) + " draft " + std::to_string(step));
    }
  }
  cudaFree(d_drafts);
  cudaFree(d_tokens);
}

void the_commits_hiddens_are_the_continuation_then_the_rounds_columns() {
  // A hidden wider than one block of the copy, and three lanes on slots out
  // of order, so a column read from the wrong lane, slot or offset shows.
  constexpr std::int32_t kHidden = 300;
  constexpr std::int32_t kLaneColumns = 4;
  constexpr std::int32_t kSlots = 5;
  const std::vector<std::int32_t> slots = {3, 0, 4};
  const auto batch = static_cast<std::int32_t>(slots.size());
  const std::int32_t columns = kLaneColumns * batch;
  std::vector<std::uint16_t> normalized(static_cast<std::size_t>(kHidden) * columns);
  for (std::size_t i = 0; i < normalized.size(); ++i) {
    normalized[i] = static_cast<std::uint16_t>(i * 7 + 1);
  }
  std::vector<std::uint16_t> table(static_cast<std::size_t>(kHidden) * kSlots);
  for (std::size_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<std::uint16_t>(0x8000u + i);
  }
  std::uint16_t *d_normalized = device_copy(normalized);
  std::uint16_t *d_table = device_copy(table);
  std::int32_t *d_slots = device_copy(slots);
  std::uint16_t *d_out = device_copy(std::vector<std::uint16_t>(normalized.size(), 0xFFFF));
  ignis::mtp::detail::commit_hiddens(d_normalized, d_table, d_slots, d_out, kHidden, kLaneColumns, batch, nullptr);
  cuda(cudaDeviceSynchronize(), "commit_hiddens");
  const auto out = host_copy(d_out, normalized.size());
  for (std::int32_t b = 0; b < batch; ++b) {
    for (std::int32_t j = 0; j < kLaneColumns; ++j) {
      const std::int32_t column = b * kLaneColumns + j;
      bool same = true;
      for (std::int32_t i = 0; i < kHidden; ++i) {
        const std::uint16_t want = j == 0 ? table[static_cast<std::size_t>(slots[b]) * kHidden + i]
                                          : normalized[static_cast<std::size_t>(column - 1) * kHidden + i];
        same = same && out[static_cast<std::size_t>(column) * kHidden + i] == want;
      }
      expect(same, "lane " + std::to_string(b) + " column " + std::to_string(j));
    }
  }
  cudaFree(d_normalized);
  cudaFree(d_table);
  cudaFree(d_slots);
  cudaFree(d_out);
}

void a_rounds_rows_are_the_chains_and_the_commits() {
  for (std::uint64_t base : {1ull, 2ull, 63ull, 64ull, 1000ull}) {
    for (std::uint32_t extent = 0; extent <= 7; ++extent) {
      for (std::uint32_t committed = 1; committed <= extent + 1; ++committed) {
        const ignis::mtp::RoundRows rows = ignis::mtp::round_rows(base, extent, committed);
        const std::string at = "base " + std::to_string(base) + " extent " + std::to_string(extent) + " committed " +
                               std::to_string(committed);
        expect(rows.first == base - 1, at + ": first");
        // The chain's steps 0 .. e - 1, one row each.
        expect(rows.chain_end - rows.first == extent, at + ": chain rows");
        // The commit writes row P - 1 .. P + c - 2; the head's frontier after
        // it is P + c - 1, one behind the sequence's.
        expect(rows.committed_end == base + committed - 1, at + ": committed rows");
        // Every chain row past the commit is at or past the new frontier,
        // and none reaches the round's last column's position P + e.
        expect(rows.chain_end <= base + extent, at + ": a chain row past the round's columns");
      }
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  chain_rows_follow_each_lanes_extent();
  chain_rows_rotate_at_each_lanes_rope_delta();
  each_steps_choices_land_in_the_lanes_draft_column();
  the_commits_hiddens_are_the_continuation_then_the_rounds_columns();
  a_rounds_rows_are_the_chains_and_the_commits();
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("mtp round: every check passed\n");
  return 0;
}
