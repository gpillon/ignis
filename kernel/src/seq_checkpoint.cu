// ignis kernel leaf - GitHub #186 (ADR 0029): prompt checkpoints on the
// device.
//
// A **prompt checkpoint** is a finished request's whole state at its
// generation opener, kept so a later request whose prompt extends it resumes
// there instead of prefilling the conversation again. This file owns the
// three things a shared prefix cannot own by itself: the mutable state *at
// the opener* (a prefix's image stands up to a page short of it), a copy of
// the partial page the opener ends inside (the publisher is still writing
// it, so it can never be shared), and one reference to the prefix holding
// every whole page below it.
//
// Since GitHub #215 (ADR 0030) none of the three allocates: the state goes
// into a retained slot the caller names, reserved at load, and the partial
// page into one KV page of the pool.
//
// What is moved and what it is made of are not decided here. The sections
// come from the leaf's one state-section table through
// ignis_seq_prefix_internal.h, and the pages from the vendored paged pool.
// This file owns lifetimes and refusals -- the same division of labour
// kernel/src/seq_prefix.cu keeps for publish and claim.
//
// The refusals are the interesting part, because a wrong one hands a request
// state for history it does not have:
//
//   * a capture demands the sequence's frontier be *exactly* the opener. The
//     state a claimant receives is the state there, and a sequence that has
//     run past it no longer has that state to give.
//   * a capture demands the whole pages below the opener *be* a prefix's.
//     That is what puts the opener inside a page the sequence alone writes --
//     the page this copies. A sequence that resumed from an earlier point and
//     prefilled past it satisfies this by publishing a **chained** prefix at
//     its own opener's page floor first (GitHub #187, ignis_seq_prefix_publish),
//     or -- since GitHub #306 -- by letting the capture lend those pages to a
//     **pages-only link** itself, which spares the prefill its cut at the
//     floor. Either way a conversation's every turn, and every iteration of an
//     agent's tool loop, leaves a checkpoint of its own.
//   * a capture changes nothing about the sequence, including on failure. It
//     is a bet the caller may lose, and a lost bet must cost nothing. A
//     successful one may lend the pages below the opener to a link: they stay
//     in the sequence's allocation and row, and move to the link only when the
//     sequence is released (ignis_seq_settle_loan), by host bookkeeping alone.

#include "ignis_seq.h"
#include "ignis_seq_checkpoint_internal.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

// Wall time `work` took with the device work already complete, in
// microseconds. The synchronize is inside the measurement on purpose: what a
// caller pays for a claim is the point at which the claimant can be stepped,
// not the point at which the copies were enqueued.
template <typename Work> double timed(Work &&work) {
  const auto started = std::chrono::steady_clock::now();
  work();
  const cudaError_t err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize after a checkpoint transfer "
                                         "failed: ") +
                             cudaGetErrorString(err));
  }
  const std::chrono::duration<double, std::micro> elapsed =
      std::chrono::steady_clock::now() - started;
  return elapsed.count();
}

// The fault armed by ignis_seq_inject_capture_fault, if any.
std::atomic<bool> g_capture_fault{false};

bool checkpoint_belongs_to(const ignis_seq_pool &pool,
                           const ignis_seq_checkpoint &checkpoint) {
  return checkpoint.prefix != nullptr && ignis_seq_prefix_belongs_to(*checkpoint.prefix, pool);
}

// The pool page holding the checkpoint's copy of its partial page, or -1 for
// an opener on a page boundary.
std::int32_t tail_page_of(const ignis_seq_checkpoint &checkpoint) {
  return checkpoint.tail.valid() ? checkpoint.tail.page_ids()[0] : -1;
}

} // namespace

void ignis_seq_inject_capture_fault() { g_capture_fault.store(true); }

void ignis_seq_capture_fault_point() {
  if (g_capture_fault.exchange(false)) {
    throw std::runtime_error("a fault injected at the commit point (test seam)");
  }
}

extern "C" int32_t
ignis_seq_checkpoint_snapshot_size(const struct ignis_seq_pool *pool,
                                   const struct ignis_seq_checkpoint *checkpoint,
                                   uint64_t *out_bytes) {
  if (out_bytes != nullptr) {
    *out_bytes = 0;
  }
  if (pool == nullptr || checkpoint == nullptr || out_bytes == nullptr) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_snapshot_size: null argument");
    return -1;
  }
  if (!checkpoint_belongs_to(*pool, *checkpoint)) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_snapshot_size: the checkpoint was not captured from this pool");
    return -1;
  }
  try {
    *out_bytes =
        ignis_seq_materialized_blob_bytes(*pool, ninfer::pages_for_tokens(checkpoint->tokens));
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_snapshot_size: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_checkpoint_snapshot(const struct ignis_seq_pool *pool,
                                                   const struct ignis_seq_checkpoint *checkpoint,
                                                   void *dst, uint64_t dst_bytes) {
  if (pool == nullptr || checkpoint == nullptr || dst == nullptr) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_snapshot: null argument");
    return -1;
  }
  if (!checkpoint_belongs_to(*pool, *checkpoint)) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_snapshot: the checkpoint was not captured from this pool");
    return -1;
  }
  try {
    // The prefix chain's whole pages, then the opener's partial page from the
    // checkpoint's own copy -- the layout a sequence standing at the opener
    // would have packed. A checkpoint on a page boundary has no partial page.
    ignis_seq_write_materialized_blob(*pool, checkpoint->prefix, tail_page_of(*checkpoint),
                                      static_cast<std::uint32_t>(checkpoint->retained_slot),
                                      checkpoint->progress,
                                      ninfer::pages_for_tokens(checkpoint->tokens), dst,
                                      dst_bytes);
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_snapshot: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_checkpoint_capture(struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                                 uint32_t opener_tokens, uint32_t retained_slot,
                                                 struct ignis_seq_checkpoint **out_checkpoint) {
  if (out_checkpoint != nullptr) {
    *out_checkpoint = nullptr;
  }
  if (pool == nullptr || seq == nullptr || out_checkpoint == nullptr) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: the sequence was not drawn from this pool");
    return -1;
  }
  if (const char *refusal = ignis_seq_clone_refusal(*pool)) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_capture: ") + refusal);
    return -1;
  }
  if (opener_tokens == 0) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: opener_tokens must be positive");
    return -1;
  }
  const auto page_size      = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  const std::uint32_t below = opener_tokens / page_size;
  // GitHub #306: pages already lent to a link (an earlier capture at this
  // opener) are a prefix's as far as a claimant is concerned.
  const std::uint32_t lent =
      seq->lent_to == nullptr ? 0 : static_cast<std::uint32_t>(seq->lent_to->lent.size());
  const std::uint32_t covered = seq->shared_pages + lent;
  if (seq->prefix == nullptr && covered == 0 && below == 0) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: sequence slot " + std::to_string(seq->slot) +
        " holds no shared prefix and the opener is inside its first page, so there is no whole "
        "page below the opener for a checkpoint to hold");
    return -1;
  }
  if (!ignis_seq_at_chunk_boundary(*seq)) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: sequence slot " +
                             std::to_string(seq->slot) + " is mid-chunk (program frontier " +
                             std::to_string(seq->position) +
                             "); its state sections are not consistent with one another");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }
  if (seq->position != static_cast<std::uint64_t>(opener_tokens)) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: the sequence stands at " + std::to_string(seq->position) +
        " tokens, not at the " + std::to_string(opener_tokens) +
        " being captured; a checkpoint is captured at the chunk boundary that lands on the "
        "generation opener");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }
  if (below < covered) {
    // The opener lies inside pages a prefix already holds: the page it ends
    // inside is not this sequence's to copy, and no later write of it is.
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: the " + std::to_string(opener_tokens) +
        "-token opener covers " + std::to_string(below) + " whole pages, but sequence slot " +
        std::to_string(seq->slot) + " has " + std::to_string(covered) +
        " in a prefix already; the opener must not fall inside them");
    return -1;
  }
  if (lent != 0 && below != covered) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: sequence slot " +
                             std::to_string(seq->slot) +
                             " has lent pages to a link already, and lends once");
    return -1;
  }
  // The whole pages below the opener must be a prefix's before a checkpoint
  // can stand on them, and **this is not relaxed** -- two things break if it
  // is, the first silently. The partial page copied below would be some
  // earlier page of the sequence's own, not the opener's; and the pages
  // between the chain and the opener would be the capturing sequence's own,
  // back in the pool when its request ends while the checkpoint still points a
  // claimant at them (the defect GitHub #186 fixed in 565d634).
  //
  // GitHub #187 made it true by publishing a chained prefix at the opener's
  // page floor first, at a chunk cut of its own. GitHub #306 (ADR 0029 as
  // amended 2026-10-07) makes it true here, at the opener: the `handed` pages
  // between what the sequence shares and the opener's floor are **lent** to a
  // **pages-only link** chained over its head -- no image, since no sequence
  // stands at its end, and no handle, since nothing claims it on its own. They
  // stay where they are, in the sequence's allocation and row, until the
  // sequence is released and the link takes them (ignis_seq_settle_loan).
  const std::uint32_t handed = below - covered;
  if (seq->kv.mapped_page_count() <= lent + handed) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: sequence slot " +
                             std::to_string(seq->slot) + " maps no page of its own past the " +
                             std::to_string(below) + " whole pages below the opener");
    return -1;
  }
  if (const std::string refusal = ignis_seq_retained_slot_refusal(*pool, retained_slot);
      !refusal.empty()) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: " + refusal);
    return -1;
  }
  // GitHub #215: an opener on a page boundary ends inside no page, so only
  // one that does not takes a page of the pool for its copy.
  const bool partial = opener_tokens % page_size != 0;
  if (partial && !pool->kv_pool.can_reserve(1)) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: the KV pool has no page left for the "
                             "copy of the page the opener ends inside");
    return -1;
  }

  try {
    // Every fallible step happens first, against the entry's own page and a
    // slot nothing holds, while the sequence is only read. A failure -- a
    // copy, a synchronize, an allocation -- leaves it exactly as it was, the
    // page goes back with the entry, and the caller simply did not get a
    // checkpoint.
    auto entry         = std::make_unique<ignis_seq_checkpoint>();
    entry->tokens      = opener_tokens;
    entry->image_bytes = pool->retained_image_bytes();
    if (partial) {
      entry->tail = pool->kv_pool.reserve(1);
      entry->tail.materialize_pages(1);
      entry->image_bytes += ignis_seq_checkpoint_page_bytes(*pool);
    }
    std::unique_ptr<ignis_seq_prefix> link;
    if (handed != 0) {
      link           = std::make_unique<ignis_seq_prefix>();
      link->tokens   = below * page_size;
      const auto own = seq->kv.page_ids();
      link->lent.assign(own.begin(), own.begin() + handed);
    }

    // The page the opener ends inside: the sequence's own first page past the
    // whole pages below the opener.
    const std::int32_t own_page = seq->kv.page_ids()[lent + handed];
    timed([&] {
      ignis_seq_capture_state(*pool, *seq, retained_slot, entry->progress);
      if (partial) {
        ignis_seq_copy_kv_page(*pool, own_page, tail_page_of(*entry));
      }
    });
    ignis_seq_capture_fault_point();

    // The commit point: nothing below can fail. The link takes a reference of
    // its own on the head the sequence stands on (the sequence keeps its own),
    // and the sequence, as lender, holds the link until it hands the pages
    // over. The checkpoint takes one reference to whatever holds the whole
    // pages below the opener: that reference, not the pages themselves, is
    // what outlives the request that warmed them.
    if (link != nullptr) {
      link->lender = seq;
      link->parent = seq->prefix;
      if (link->parent != nullptr) {
        ++link->parent->refcount;
      }
      link->refcount = 1;
      seq->lent_to   = link.release();
    }
    entry->prefix = seq->lent_to != nullptr ? seq->lent_to : seq->prefix;
    ++entry->prefix->refcount;
    entry->retained_slot               = static_cast<std::int32_t>(retained_slot);
    pool->retained_held[retained_slot] = true;
    *out_checkpoint                    = entry.release();
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_capture: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_alloc_from_checkpoint(struct ignis_seq_pool *pool,
                                                    uint32_t context_tokens,
                                                    struct ignis_seq_checkpoint *checkpoint,
                                                    struct ignis_seq **out_seq) {
  if (out_seq != nullptr) {
    *out_seq = nullptr;
  }
  if (pool == nullptr || checkpoint == nullptr || out_seq == nullptr) {
    ignis_seq_set_last_error("ignis_seq_alloc_from_checkpoint: null argument");
    return -1;
  }
  // The shared pages, a fresh zeroed tail, the block-table row -- everything
  // a sibling claim already does. The prefix's *own* mutable image is not
  // cloned: the checkpoint's, captured further along at the opener, goes
  // over the slot immediately below.
  const int32_t rc =
      ignis_seq_alloc_against_prefix(pool, context_tokens, checkpoint->prefix, false, out_seq);
  if (rc != 0) {
    return rc;
  }
  ignis_seq *seq = *out_seq;
  try {
    const std::int32_t own_page = seq->kv.page_ids()[0];
    checkpoint->last_claim_micros = timed([&] {
      ignis_seq_clone_state(*pool, static_cast<std::uint32_t>(checkpoint->retained_slot),
                            checkpoint->progress, *seq);
      // The partial page the opener ends inside, into the first page this
      // sequence owns. `ignis_seq_alloc_against_prefix` zeroed that page a
      // moment ago; this writes the capture's copy over it, which carries the
      // history and nothing past the opener.
      if (checkpoint->tail.valid()) {
        ignis_seq_copy_kv_page(*pool, tail_page_of(*checkpoint), own_page);
      }
    });
    ++checkpoint->claim_count;
    return 0;
  } catch (const std::exception &e) {
    // The sequence exists but its state is half-written, so it is released
    // rather than handed back: a partially claimed sequence would decode
    // from a slot whose sections disagree with its pages.
    ignis_seq_release(pool, seq);
    *out_seq = nullptr;
    ignis_seq_set_last_error(std::string("ignis_seq_alloc_from_checkpoint: ") + e.what());
    return -1;
  }
}

extern "C" void ignis_seq_checkpoint_release(struct ignis_seq_pool *pool,
                                              struct ignis_seq_checkpoint *checkpoint) {
  // `pool` is taken for the symmetry every other release in this ABI has,
  // and it earns it here for the reason ignis_seq_prefix_release takes one: a
  // handle returned against one pool and released against another would free
  // pages out of a pool that never lent them.
  if (checkpoint == nullptr) {
    return;
  }
  if (pool != nullptr && checkpoint->prefix != nullptr &&
      !ignis_seq_prefix_belongs_to(*checkpoint->prefix, *pool)) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_release: the checkpoint was not captured from "
                             "this pool; nothing was released");
    return;
  }
  // The retained slot comes back, and the tail page goes with the entry; the
  // prefix under it loses one holder, and its pages come back only if that
  // was the last.
  if (pool != nullptr && checkpoint->retained_slot >= 0) {
    pool->retained_held[static_cast<std::size_t>(checkpoint->retained_slot)] = false;
  }
  ignis_seq_prefix_drop_reference(checkpoint->prefix);
  delete checkpoint;
}

extern "C" int32_t ignis_seq_checkpoint_stats(const struct ignis_seq_checkpoint *checkpoint,
                                               struct ignis_seq_checkpoint_stats *out_stats) {
  if (checkpoint == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->tokens = checkpoint->tokens;
  // The whole head below the opener, the prefix's chain included (GitHub
  // #187): what a claimant shares, which is what this number is read as.
  out_stats->pages = checkpoint->prefix == nullptr
                         ? 0
                         : ignis_seq_prefix_total_pages(*checkpoint->prefix);
  out_stats->image_bytes       = checkpoint->image_bytes;
  out_stats->claim_count       = checkpoint->claim_count;
  out_stats->last_claim_micros = checkpoint->last_claim_micros;
  return 0;
}
