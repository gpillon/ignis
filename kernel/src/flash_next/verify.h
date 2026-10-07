// ignis kernel leaf -- the Flash-Next verify round (spec flash-next/07 phase C, GitHub #307; OURS,
// ADR 0043): k+1 columns per lane through the trunk without committing them, the vendored accept,
// then a commit of exactly the run each lane keeps.
//
// A round of `lanes` lanes at window k (window_for) is two device passes with the host between them:
//
//   pass  (one graph per width)  save what the forward advances in place and the commit must undo
//                                -- each lane's indexer tails, n-gram conv columns, hq ring words
//                                and the ring rows its k+1 positions overwrite; the forward over
//                                Batch{lanes, k+1, verify = &records} (flash_next_internal.h: GDN
//                                runs the vendored replay record and leaves its state alone, the
//                                indexer copies its raw keys out); the head on every column; the
//                                target argmax and the vendored speculative_accept_greedy_drafts.
//   host                         reads the accept back, cuts each lane's run at its first stop id,
//                                and stages each lane's committed column count c.
//   commit (one graph per width) GDN: our fold of the first c records into each lane's recurrent
//                                state (the vendored recurrence's own transition, so the state is
//                                bit for bit what c one-token rounds leave) and conv taps
//                                tail3(taps || conv inputs[0, c)); the indexer tails rebuilt from
//                                the saved tail and the recorded keys; the n-gram conv columns
//                                tail9(saved || inputs[0, c)); the ring rows and words of the
//                                rejected positions restored -- a rejected column is a column that
//                                was never drafted.
//
// What needs no commit: KV rows and pooled indexer blocks at positions past the new frontier sit
// where nothing reads them until the frontier reaches them again, and that append rewrites them
// (positions are never read past a lane's frontier); residency is a cache.
//
// On an MTP load (phase D, mtp.h) the commit also drafts the next round, between the fold and the
// restore: the head's alignment over the pass's k + 1 stacks and licensed tokens (its entries at
// the columns' own positions, the committed ones kept), then k - 1 chained steps from each lane's
// last committed column, past its frontier. The head's attention section is one more attention
// layer of the pool, saved and restored with the trunk's -- its ring rows over the chain's
// positions too, so a lane's saved range is 2k positions there. Between the alignment and the
// chain the head's own section goes back to the new frontier (restore_head): the alignment wrote
// it at the rejected columns too, and a chain step reads it.

#pragma once

#include "flash_next_internal.h"

#include "core/arena.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ignis_seq_pool;

namespace ignis::flash_next::verify {

// The most rows a verify call carries (lanes * (k + 1)): the MoE decode route's, the GDN replay
// record's batch and the QSA listed hq decode's bound.
inline constexpr uint32_t kMaxRows = 8;
// The widest window a load names (the 27B's draft-token range, spec 05).
inline constexpr uint32_t kMaxWindow = 7;
// The lanes a round serves (IGNIS_DECODE_MAX_BATCH).
inline constexpr uint32_t kMaxLanes = 8;
// The attention layers the save and restore kernels address (Flash-Next's 12, plus the MTP head's).
inline constexpr int32_t kMaxAttentionLayers = 16;
// The GDN layers our fold addresses (Flash-Next's 36).
inline constexpr int32_t kMaxGdnLayers = 48;

// The window a round of `lanes` lanes runs at: the load's draft tokens, cut so that lanes * (k + 1)
// stays within the row budget (0 or more than kMaxRows: kMaxRows). 0: that width runs today's
// one-token round.
uint32_t window_for(uint32_t draft_tokens, uint32_t row_budget, uint32_t lanes);

// The decode route's widest call on a load: its plain rounds' lanes, or a verify round's rows.
uint32_t decode_rows(uint32_t decode_lanes, uint32_t draft_tokens, uint32_t row_budget);

// The positions past a lane's frontier a round of window k writes: its k + 1 columns, and on an MTP
// load the head's chained steps up to 2k.
uint32_t written_positions(uint32_t window, bool mtp);

// Every device byte a load with a draft window keeps for its rounds (ADR 0030), by line.
struct Plan {
  std::size_t staging = 0;   // per-round inputs and the accept's outputs
  std::size_t accept = 0;    // the vendored accept's workspace
  std::size_t gdn_records = 0;
  std::size_t indexer_records = 0;
  std::size_t saved = 0;     // the lane state the pass saves for the commit
  std::size_t total() const { return staging + accept + gdn_records + indexer_records + saved; }
};
Plan plan(const Geometry &g, int32_t kv_format, uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget,
          int32_t attention_layers, int32_t gdn_layers, bool mtp);

// The round's buffers, at stable addresses (a captured graph replays them). Allocated once at load.
struct State {
  uint32_t draft_tokens = 0;
  uint32_t row_budget = 0;
  uint32_t lanes = 0;  // the load's decode lanes
  int32_t attention_layers = 0;
  int32_t gdn_layers = 0;
  bool mtp = false;
  Plan sizes;

  // Host-staged per round: I32 [lanes] each, drafts I32 [lanes][k] (lane-major, the accept's [k, B]).
  std::unique_ptr<ninfer::DeviceBuffer> valid_columns;  // extent + 1
  std::unique_ptr<ninfer::DeviceBuffer> extents;
  std::unique_ptr<ninfer::DeviceBuffer> lengths;        // the accept RNG's position base: the frontier
  std::unique_ptr<ninfer::DeviceBuffer> anchors;        // the anchor in, the correction / bonus out
  std::unique_ptr<ninfer::DeviceBuffer> drafts;
  std::unique_ptr<ninfer::DeviceBuffer> commit;         // c, staged between the pass and the commit
  // Written by the pass: I32 [rows] each, lane-major; I32 [lanes] each.
  std::unique_ptr<ninfer::DeviceBuffer> target_tokens;
  std::unique_ptr<ninfer::DeviceBuffer> licensed;
  std::unique_ptr<ninfer::DeviceBuffer> licensed_counts;
  std::unique_ptr<ninfer::DeviceBuffer> accepted;
  std::unique_ptr<ninfer::DeviceArena> accept_workspace;
  // An MTP load's drafting (mtp.h), null otherwise: the head's picks per row I32 [rows], each
  // lane's chain token I32 [lanes] and stack BF16 [lanes][streams * hidden], the chain's positions
  // I32 [k - 1][lanes] (host-staged with the commit counts) and the drafts out I32 [lanes][k].
  std::unique_ptr<ninfer::DeviceBuffer> picks;
  std::unique_ptr<ninfer::DeviceBuffer> chain_tokens;
  std::unique_ptr<ninfer::DeviceBuffer> chain_stack;
  std::unique_ptr<ninfer::DeviceBuffer> chain_positions;
  std::unique_ptr<ninfer::DeviceBuffer> drafts_out;

  // The records the ops write (flash_next_internal.h VerifyRecords) and what the pass saves.
  std::unique_ptr<ninfer::DeviceBuffer> gdn_records;
  std::unique_ptr<ninfer::DeviceBuffer> indexer_records;
  std::unique_ptr<ninfer::DeviceBuffer> saved;
  VerifyRecords records;

  // One pass and one commit graph per width (index width - 1), captured at the width's window.
  std::array<cudaGraphExec_t, kMaxLanes> pass_exec{};
  std::array<cudaGraphExec_t, kMaxLanes> commit_exec{};
  std::array<bool, kMaxLanes> ready{};

  uint32_t window(uint32_t width) const { return window_for(draft_tokens, row_budget, width); }
  // What the buffers hold, read off them (ignis_model_stats).
  std::size_t device_bytes() const;
  // Bit (w - 1) set when width w's pass and commit graphs are both captured.
  uint32_t ready_mask() const;

  State() = default;
  State(const State &) = delete;
  State &operator=(const State &) = delete;
  ~State();
};

// Why a load of these options cannot run the verify round, or empty: what bind refuses before any
// allocation, so a plan and a load refuse alike.
std::string refusal(const Geometry &g, uint32_t lanes, uint32_t draft_tokens, uint32_t row_budget,
                    int32_t attention_layers, int32_t gdn_layers);

// Allocates the state for a load (`lanes` decode lanes), or null and *error.
std::unique_ptr<State> create(const Geometry &g, int32_t kv_format, uint32_t lanes, uint32_t draft_tokens,
                              uint32_t row_budget, int32_t attention_layers, int32_t gdn_layers, bool mtp,
                              std::string *error);

// The lane-state addresses the save, restore and fold kernels read and write: one pool's sections.
struct Sections {
  int32_t attention_layers = 0;
  int32_t gdn_layers = 0;
  void *tails[kMaxAttentionLayers] = {};       // the layer's tail keys, slot 0
  int64_t tail_slot_elements = 0;              // BF16 elements per slot ((compress - 1) * key dim)
  void *residual_k[kMaxAttentionLayers] = {};  // the layer's hq side plane, slot 0 (null: none)
  void *residual_v[kMaxAttentionLayers] = {};
  uint32_t *ring = nullptr;                    // ring words, slot 0 (null: none)
  void *ngram_conv = nullptr;                  // [slot][columns][channels] BF16
  int32_t ngram_columns = 0;
  int32_t ngram_channels = 0;
  float *recurrent[kMaxGdnLayers] = {};        // the GDN layer's recurrent state, slot 0
  void *conv[kMaxGdnLayers] = {};              // the GDN layer's conv taps, slot 0
};
// The sections of `pool`, or *error naming what it lacks.
bool sections_of(const ignis_seq_pool &pool, const Geometry &g, int32_t gdn_layers, Sections *out,
                 std::string *error);

// The pass's first step: every lane's state the forward advances in place, into `state.saved`.
// `slots`, `positions`: DEVICE [lanes] (the round's staged rows). Graph-safe.
int32_t save(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
             const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error);

// The pass's last step: the target argmax over `logits` (BF16 [lanes * (window + 1)][vocab]) and the
// vendored accept. `configs`: the round's device SamplingConfig [lanes]. Graph-safe.
int32_t accept(State &state, const Geometry &g, uint32_t lanes, uint32_t window, const void *logits,
               const void *configs, cudaStream_t stream, std::string *error);

// The commit, with each lane's committed column count in `state.commit` (DEVICE [lanes], 1..window+1),
// in two halves an MTP load drafts between: the GDN fold, then the restore of everything else.
// Graph-safe.
int32_t fold(const State &state, const Sections &sections, uint32_t lanes, uint32_t window, const int32_t *slots,
             cudaStream_t stream, std::string *error);
int32_t restore(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
                const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error);

// An MTP load's drafting, between the head's alignment and its chain: the head's section (the
// pool's last attention section) at each lane's new frontier -- its indexer tail, and under
// hq-e8-2b the ring words and the ring rows of the alignment's rejected columns. Without it a
// chain step at q reads a rejected column's ring row as position q - 512 + i (a ring slot carries
// no position) and pools a block from the alignment's tail. The restore after the chain puts the
// section back again, over the chain's positions too. Graph-safe.
int32_t restore_head(const State &state, const Sections &sections, const Geometry &g, uint32_t lanes, uint32_t window,
                     const int32_t *slots, const int32_t *positions, cudaStream_t stream, std::string *error);

}  // namespace ignis::flash_next::verify
