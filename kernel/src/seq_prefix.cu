// ignis kernel leaf - P4-10 (GitHub #126, ADR 0024): device prefix reuse.
//
// One prompt head, two mechanisms. The **KV pages are read-only history**, so
// a published prefix owns them and every claimant's block-table row addresses
// those same physical pages -- one charge to the pool, however many
// claimants, and the pages come back when the last holder releases. The
// **mutable sections are cloned device-to-device** -- the GDN recurrent
// state, the conv taps, the penalty-count row -- from a device-resident image
// the publish captured at the prefix's end. Nothing here crosses PCIe, which
// is the difference between this and restoring a sibling's snapshot.
//
// What each half is made of is not decided here: it comes from the leaf's one
// state-section table (ignis_seq_sections.h), read through
// ignis_seq_prefix_internal.h. This file moves bytes, owns lifetimes, and
// decides what to refuse -- the same division of labour kernel/src/seq.cu
// keeps for snapshot and restore.
//
// The refusals are the interesting part, because a wrong one corrupts a
// sequence silently:
//
//   * a publish demands the sequence's frontier be *exactly* the prefix. The
//     mutable state handed to a claimant is the state at the prefix's end,
//     and a sequence that has run past it no longer has that state to give.
//   * a prefix is a whole number of KV pages. That is what puts a claimant's
//     first write on the first page it owns, so no claimant ever writes a
//     shared page -- the invariant the whole mechanism rests on.
//   * a sequence claims at most one prefix, and a publisher cannot publish a
//     head it does not own.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The whole block-table row of `seq`: the prefix's physical pages first, then
// the sequence's own, in logical page order.
//
// The vendored `PagedKVAllocation::bind_row` publishes an allocation's pages
// from logical index 0, which is right for a sequence that owns its whole
// history and wrong for one whose head belongs to a prefix. So the row is
// written here instead, once, after the bind -- and nothing re-publishes it
// afterwards, because a sequence materializes its whole reservation at
// allocation time and never grows it.
void publish_shared_row(ignis_seq_pool &pool, const ignis_seq &seq) {
  std::vector<std::int32_t> ids;
  ids.reserve(ignis_seq_logical_page_count(seq));
  if (seq.prefix != nullptr) {
    const auto shared = seq.prefix->kv.page_ids();
    ids.insert(ids.end(), shared.begin(), shared.end());
  }
  const auto own = seq.kv.page_ids();
  ids.insert(ids.end(), own.begin(), own.end());
  if (ids.empty()) {
    return;
  }
  const ninfer::Tensor row = pool.kv_pool.block_table_row(seq.slot);
  const cudaError_t err    = cudaMemcpy(row.data, ids.data(),
                                        ids.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy(block-table row) failed: ") +
                             cudaGetErrorString(err));
  }
}

// The prefix's page count, from the allocation that owns them.
std::uint32_t prefix_pages(const ignis_seq_prefix &prefix) {
  return prefix.kv.mapped_page_count();
}

// Run `transfer` and return the wall time it took, in microseconds, with the
// device work already complete. The synchronize is inside the measurement on
// purpose: what a caller pays for a clone is the point at which the claimant
// can be stepped, not the point at which the copies were enqueued.
double timed_transfer(ignis_seq_pool &pool, ignis_seq_prefix &prefix, ignis_seq &seq,
                      ignis_seq_prefix_direction direction) {
  const auto started = std::chrono::steady_clock::now();
  ignis_seq_prefix_transfer(pool, prefix, seq, direction);
  const cudaError_t err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize after a prefix transfer failed: ") +
                             cudaGetErrorString(err));
  }
  const std::chrono::duration<double, std::micro> elapsed =
      std::chrono::steady_clock::now() - started;
  return elapsed.count();
}

} // namespace

void ignis_seq_prefix_drop_reference(ignis_seq_pool *pool, ignis_seq_prefix *prefix) {
  if (prefix == nullptr) {
    return;
  }
  if (prefix->refcount > 0) {
    --prefix->refcount;
  }
  if (prefix->refcount != 0) {
    return;
  }
  // The last holder let go. `~PagedKVAllocation` returns the shared pages and
  // the entitlement to `pool` -- which is why a prefix's pages are charged to
  // the pool once and released once, whatever the claimant count did in
  // between.
  (void)pool;
  delete prefix;
}

extern "C" int32_t ignis_seq_prefix_publish(struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                             uint32_t prefix_tokens,
                                             struct ignis_seq_prefix **out_prefix) {
  if (out_prefix != nullptr) {
    *out_prefix = nullptr;
  }
  if (pool == nullptr || seq == nullptr || out_prefix == nullptr) {
    ignis_seq_set_last_error("ignis_seq_prefix_publish: null argument");
    return -1;
  }
  if (!ignis_seq_belongs_to(*pool, *seq)) {
    ignis_seq_set_last_error("ignis_seq_prefix_publish: the sequence was not drawn from this pool");
    return -1;
  }
  if (seq->prefix != nullptr) {
    ignis_seq_set_last_error(
        "ignis_seq_prefix_publish: sequence slot " + std::to_string(seq->slot) +
        " already claims a shared prefix, so its head is not its own to publish");
    return -1;
  }
  const auto page_size = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  if (prefix_tokens == 0 || prefix_tokens % page_size != 0) {
    ignis_seq_set_last_error("ignis_seq_prefix_publish: a shared prefix is a whole number of " +
                             std::to_string(page_size) + "-token KV pages, got " +
                             std::to_string(prefix_tokens) +
                             " (a partial page would be written by both its owner and its "
                             "claimants)");
    return -1;
  }
  if (!ignis_seq_at_chunk_boundary(*seq)) {
    ignis_seq_set_last_error("ignis_seq_prefix_publish: sequence slot " +
                             std::to_string(seq->slot) + " is mid-chunk (program frontier " +
                             std::to_string(seq->position) +
                             "); its state sections are not consistent with one another");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }
  if (seq->position != static_cast<std::uint64_t>(prefix_tokens)) {
    // Not a formality: what a claimant receives is the mutable state at the
    // prefix's end. A sequence standing anywhere else has state that belongs
    // to a longer history, and a claimant seeded from it would decode from a
    // GDN slot that has seen tokens its KV pages do not carry.
    ignis_seq_set_last_error(
        "ignis_seq_prefix_publish: the sequence stands at " + std::to_string(seq->position) +
        " tokens, not at the " + std::to_string(prefix_tokens) +
        " being published; a prefix is published at the chunk boundary that lands on it");
    return IGNIS_SEQ_ERR_NOT_AT_BOUNDARY;
  }

  const std::uint32_t pages = prefix_tokens / page_size;
  if (pages > seq->kv.mapped_page_count()) {
    ignis_seq_set_last_error("ignis_seq_prefix_publish: the sequence maps " +
                             std::to_string(seq->kv.mapped_page_count()) + " KV pages, fewer than the " +
                             std::to_string(pages) + " being published");
    return -1;
  }
  const std::uint32_t tail = seq->kv.page_entitlement() - pages;
  if (tail == 0) {
    ignis_seq_set_last_error(
        "ignis_seq_prefix_publish: the whole reservation of sequence slot " +
        std::to_string(seq->slot) + " is the prefix, leaving it no page of its own to write");
    return -1;
  }

  try {
    // Fallible work first, while nothing the pool owns has moved: the image
    // allocation and the capture. A failure here leaves the sequence and the
    // pool exactly as they were.
    auto entry                = std::make_unique<ignis_seq_prefix>();
    const std::uint64_t bytes = ignis_seq_prefix_clone_bytes(ignis_seq_prefix_clone_layout(*pool));
    entry->clone_image        = ninfer::DeviceBuffer(static_cast<std::size_t>(bytes));
    entry->tokens             = prefix_tokens;
    timed_transfer(*pool, *entry, *seq, IGNIS_SEQ_PREFIX_CAPTURE);

    // From here the steps are balanced against each other and cannot fail:
    // the tail entitlement handed back below is exactly the one re-reserved,
    // and the row released is exactly the one re-bound.
    ninfer::PagedKVAllocation shared = std::move(seq->kv);
    shared.unbind_row();
    // The publisher has written exactly `prefix_tokens`, so every page past
    // the prefix is still the zeroed page `ignis_seq_alloc` materialized --
    // trimming them loses no history, and hands back the pages the tail
    // reservation immediately takes.
    shared.trim_pages(pages);
    shared.set_page_entitlement(pages);
    entry->kv = std::move(shared);

    seq->kv = pool->kv_pool.reserve(tail);
    seq->kv.materialize_pages(tail);
    pool->kv_pool.zero_pages(seq->kv.page_ids());
    seq->kv.bind_row(seq->slot);
    seq->prefix       = entry.get();
    seq->shared_pages = pages;
    publish_shared_row(*pool, *seq);

    // Two holders from the start: the handle this call returns, and the
    // publishing sequence, which is now a claimant of its own prefix.
    entry->refcount = 2;
    *out_prefix     = entry.release();
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_prefix_publish: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_alloc_shared(struct ignis_seq_pool *pool, uint32_t context_tokens,
                                           struct ignis_seq_prefix *prefix,
                                           struct ignis_seq **out_seq) {
  if (out_seq != nullptr) {
    *out_seq = nullptr;
  }
  if (pool == nullptr || prefix == nullptr || out_seq == nullptr) {
    ignis_seq_set_last_error("ignis_seq_alloc_shared: null argument");
    return -1;
  }
  if (!prefix->kv.valid() || !prefix->kv.belongs_to(pool->kv_pool)) {
    ignis_seq_set_last_error("ignis_seq_alloc_shared: the prefix was not published from this pool");
    return -1;
  }
  if (context_tokens == 0) {
    ignis_seq_set_last_error("ignis_seq_alloc_shared: context_tokens must be positive");
    return -1;
  }
  const std::uint32_t total  = ninfer::pages_for_tokens(context_tokens);
  const std::uint32_t shared = prefix_pages(*prefix);
  if (total <= shared) {
    ignis_seq_set_last_error(
        "ignis_seq_alloc_shared: a reservation of " + std::to_string(context_tokens) +
        " tokens is " + std::to_string(total) + " pages, which the " + std::to_string(shared) +
        "-page prefix leaves no page of its own");
    return -1;
  }
  const std::uint32_t tail = total - shared;
  if (pool->free_slots.empty()) {
    ignis_seq_set_last_error("ignis_seq_alloc_shared: sequence pool exhausted (no free slot)");
    return -1;
  }
  if (!pool->kv_pool.can_reserve(tail)) {
    ignis_seq_set_last_error("ignis_seq_alloc_shared: sequence pool exhausted (KV pages)");
    return -1;
  }

  // Peek (not pop) the slot, exactly as ignis_seq_alloc does: on any failure
  // below `seq`'s destructor unwinds the reservation, so `free_slots` must
  // stay untouched until every step has actually succeeded.
  const std::int32_t slot = pool->free_slots.back();
  try {
    auto seq = std::make_unique<ignis_seq>();
    seq->kv  = pool->kv_pool.reserve(tail);
    seq->kv.materialize_pages(tail);
    seq->kv.bind_row(slot);
    // Only its OWN pages are zeroed. The prefix's pages are the history it
    // claims -- zeroing them would erase the very thing this call exists to
    // hand over, and they belong to other holders besides.
    pool->kv_pool.zero_pages(seq->kv.page_ids());
    seq->slot         = slot;
    seq->prefix       = prefix;
    seq->shared_pages = shared;
    publish_shared_row(*pool, *seq);

    // The GDN slot, the conv taps and the penalty counts, device to device.
    // No zeroing first and no cudaMemset of the count row either: the clone
    // writes every byte of each, and a fresh zero would only be overwritten.
    prefix->last_clone_micros = timed_transfer(*pool, *prefix, *seq, IGNIS_SEQ_PREFIX_CLONE);
    ++prefix->clone_count;

    ++prefix->refcount;
    pool->free_slots.pop_back();
    *out_seq = seq.release();
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_alloc_shared: ") + e.what());
    return -1;
  }
}

extern "C" void ignis_seq_prefix_release(struct ignis_seq_pool *pool,
                                          struct ignis_seq_prefix *prefix) {
  ignis_seq_prefix_drop_reference(pool, prefix);
}

extern "C" int32_t ignis_seq_prefix_stats(const struct ignis_seq_prefix *prefix,
                                           struct ignis_seq_prefix_stats *out_stats) {
  if (prefix == nullptr || out_stats == nullptr) {
    return -1;
  }
  out_stats->tokens            = prefix->tokens;
  out_stats->pages             = prefix->kv.mapped_page_count();
  out_stats->refcount          = prefix->refcount;
  out_stats->clone_image_bytes = prefix->clone_image.bytes;
  out_stats->clone_count       = prefix->clone_count;
  out_stats->last_clone_micros = prefix->last_clone_micros;
  return 0;
}
