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
//   * a capture demands the whole pages below the opener already *be* the
//     sequence's shared prefix. That is what puts the opener inside a page
//     the sequence alone writes -- the page this copies. A sequence that
//     resumed from an earlier point and prefilled past it satisfies this by
//     publishing a **chained** prefix at its own opener's page floor first
//     (GitHub #187, ignis_seq_prefix_publish): the pages it warmed itself,
//     over the ones it claimed. That is how a conversation's second turn --
//     and every later iteration of an agent's tool loop -- leaves a
//     checkpoint of its own, without this file learning to own pages.
//   * a capture changes nothing about the sequence, including on failure. It
//     is a bet the caller may lose, and a lost bet must cost nothing.

#include "ignis_seq.h"
#include "ignis_seq_checkpoint_internal.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"

#include <cuda_runtime.h>

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

std::vector<std::int32_t> checkpoint_prefix_pages(const ignis_seq_checkpoint &checkpoint) {
  std::vector<const ignis_seq_prefix *> chain;
  for (const ignis_seq_prefix *at = checkpoint.prefix; at != nullptr; at = at->parent) {
    chain.push_back(at);
  }
  std::vector<std::int32_t> pages;
  for (auto at = chain.rbegin(); at != chain.rend(); ++at) {
    const auto ids = (*at)->kv.page_ids();
    pages.insert(pages.end(), ids.begin(), ids.end());
  }
  return pages;
}

void zero_snapshot_gaps(unsigned char *base, const std::vector<ignis_seq_section> &sections,
                        std::uint64_t total_bytes) {
  std::uint64_t cursor = sizeof(ignis_seq_snapshot_header) +
                         sections.size() * sizeof(ignis_seq_section);
  for (const ignis_seq_section &section : sections) {
    if (cursor < section.offset) {
      std::memset(base + cursor, 0, static_cast<std::size_t>(section.offset - cursor));
    }
    cursor = section.offset + section.bytes;
  }
  if (cursor < total_bytes) {
    std::memset(base + cursor, 0, static_cast<std::size_t>(total_bytes - cursor));
  }
}

const ignis_seq_section &section_named(const std::vector<ignis_seq_section> &sections,
                                       std::int32_t kind) {
  for (const ignis_seq_section &section : sections) {
    if (section.kind == kind) {
      return section;
    }
  }
  throw std::logic_error(std::string("state-section table has no ") +
                         ignis_seq_section_name(kind) + " section");
}

bool checkpoint_belongs_to(const ignis_seq_pool &pool,
                           const ignis_seq_checkpoint &checkpoint) {
  return checkpoint.prefix != nullptr && checkpoint.prefix->kv.belongs_to(pool.kv_pool);
}

void pack_checkpoint_pages(const ignis_seq_pool &pool, const ignis_seq_checkpoint &checkpoint,
                           std::uint32_t page_count, void *dst) {
  const std::vector<std::int32_t> prefix_pages = checkpoint_prefix_pages(checkpoint);
  if (page_count < prefix_pages.size() || page_count > prefix_pages.size() + 1) {
    throw std::logic_error("checkpoint materialization extent does not match its prefix and tail");
  }
  auto *out = static_cast<unsigned char *>(dst);
  const auto *tail = static_cast<const unsigned char *>(checkpoint.tail_page.p);
  std::size_t tail_offset = 0;
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    const std::size_t bytes = static_cast<std::size_t>(plane.nb[3]);
    const auto *base = static_cast<const unsigned char *>(plane.data);
    for (std::int32_t page : prefix_pages) {
      const cudaError_t err = cudaMemcpyAsync(out, base + static_cast<std::int64_t>(page) * plane.nb[3],
                                              bytes, cudaMemcpyDeviceToHost, nullptr);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync(checkpoint prefix page) failed: ") +
                                 cudaGetErrorString(err));
      }
      out += bytes;
    }
    if (page_count > prefix_pages.size()) {
      const cudaError_t err = cudaMemcpyAsync(out, tail + tail_offset, bytes,
                                              cudaMemcpyDeviceToHost, nullptr);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync(checkpoint tail page) failed: ") +
                                 cudaGetErrorString(err));
      }
      out += bytes;
    }
    tail_offset += bytes;
  }
}

} // namespace

extern "C" int32_t ignis_seq_checkpoint_image_bytes(const struct ignis_seq_pool *pool,
                                                     uint64_t *out_bytes) {
  if (pool == nullptr || out_bytes == nullptr) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_image_bytes: null argument");
    return -1;
  }
  try {
    *out_bytes = ignis_seq_checkpoint_image_bytes_of(*pool);
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_image_bytes: ") + e.what());
    return -1;
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
    const std::uint32_t pages = ninfer::pages_for_tokens(checkpoint->tokens);
    *out_bytes = ignis_seq_snapshot_bytes(ignis_seq_section_table(*pool, pages));
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
    const std::uint32_t pages = ninfer::pages_for_tokens(checkpoint->tokens);
    const std::vector<ignis_seq_section> sections = ignis_seq_section_table(*pool, pages);
    const ignis_seq_snapshot_header header = ignis_seq_snapshot_header_for(*pool, pages, sections);
    if (dst_bytes < header.total_bytes) {
      ignis_seq_set_last_error("ignis_seq_checkpoint_snapshot: destination is smaller than the blob");
      return -1;
    }
    auto *base = static_cast<unsigned char *>(dst);
    std::memcpy(base, &header, sizeof(header));
    std::memcpy(base + sizeof(header), sections.data(), sections.size() * sizeof(ignis_seq_section));
    zero_snapshot_gaps(base, sections, header.total_bytes);
    const std::vector<ignis_seq_section> clone = ignis_seq_prefix_clone_layout(*pool);
    for (const ignis_seq_section &section : sections) {
      unsigned char *at = base + section.offset;
      if (section.kind == IGNIS_SEQ_SECTION_KV_PAGES) {
        pack_checkpoint_pages(*pool, *checkpoint, pages, at);
      } else if (section.kind == IGNIS_SEQ_SECTION_PROGRESS) {
        std::memcpy(at, &checkpoint->progress, sizeof(checkpoint->progress));
      } else {
        const ignis_seq_section &source = section_named(clone, section.kind);
        const cudaError_t err = cudaMemcpyAsync(at,
                                                static_cast<const unsigned char *>(checkpoint->image.p) + source.offset,
                                                static_cast<std::size_t>(section.bytes),
                                                cudaMemcpyDeviceToHost, nullptr);
        if (err != cudaSuccess) {
          throw std::runtime_error(std::string("cudaMemcpyAsync(checkpoint state) failed: ") +
                                   cudaGetErrorString(err));
        }
      }
    }
    const cudaError_t err = cudaStreamSynchronize(nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamSynchronize after checkpoint snapshot failed: ") +
                               cudaGetErrorString(err));
    }
    return 0;
  } catch (const std::exception &e) {
    ignis_seq_set_last_error(std::string("ignis_seq_checkpoint_snapshot: ") + e.what());
    return -1;
  }
}

extern "C" int32_t ignis_seq_checkpoint_capture(struct ignis_seq_pool *pool, struct ignis_seq *seq,
                                                 uint32_t opener_tokens,
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
  if (opener_tokens == 0) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: opener_tokens must be positive");
    return -1;
  }
  if (seq->prefix == nullptr) {
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: sequence slot " + std::to_string(seq->slot) +
        " holds no shared prefix, so nothing owns the whole pages below the opener; publish "
        "one at the opener's page boundary first");
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
  const auto page_size          = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize);
  const std::uint32_t below     = opener_tokens / page_size;
  if (below != seq->shared_pages) {
    // Not a formality, and **do not relax this condition**: two separate
    // things break if you do, and the first breaks silently.
    //
    //   1. **The captured image would be corrupt.** The capture below copies
    //      `seq->kv.page_ids()[0]` as the partial tail page -- the sequence's
    //      own first page, which is the opener's page only while `below ==
    //      shared_pages`. With the opener further up, index 0 is some earlier
    //      page of the sequence's own tail, so the checkpoint would carry the
    //      wrong page of history and every claimant would resume from it.
    //      Nothing reports that: it is answered, not refused.
    //   2. **The pages between would have no holder.** A checkpoint holds one
    //      prefix reference, one mutable image and exactly one tail page, and
    //      ignis_seq_alloc_from_checkpoint hands a claimant the prefix chain's
    //      whole pages plus that single copied page. Pages above the chain and
    //      below the opener are the *capturing sequence's own*: they go back
    //      to the pool when that request ends, and the checkpoint would point
    //      a later claimant at history somebody else is now writing. That
    //      breaks the discipline GitHub #186 (565d634) built the checkpoint
    //      on: every page a checkpoint promises is held by something that
    //      outlives the request that warmed it.
    //
    // The right move is the one GitHub #187 took, and it makes this condition
    // true instead of weakening it: the sequence publishes a **chained**
    // prefix at its own opener's page floor first (ignis_seq_prefix_publish),
    // which hands those in-between pages to an entry that outlives the
    // request and moves `shared_pages` up to `below`. Then what a claimant
    // shares is the whole chain and what it copies is the one page below the
    // opener that this sequence owns alone -- which is what the capture is
    // built on.
    ignis_seq_set_last_error(
        "ignis_seq_checkpoint_capture: the " + std::to_string(opener_tokens) +
        "-token opener covers " + std::to_string(below) + " whole pages, but sequence slot " +
        std::to_string(seq->slot) + " shares " + std::to_string(seq->shared_pages) +
        "; the opener must fall inside the sequence's own first page (a sequence that resumed "
        "from an earlier checkpoint and prefilled past it publishes a chained prefix at the "
        "opener's page floor first -- ignis_seq_prefix_publish, GitHub #187)");
    return -1;
  }
  if (seq->kv.mapped_page_count() == 0) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_capture: sequence slot " +
                             std::to_string(seq->slot) + " maps no page of its own to capture");
    return -1;
  }

  try {
    // Every fallible step happens here, against buffers of this call's own.
    // The sequence is read and never touched, so a failure -- an allocation,
    // a copy, a synchronize -- leaves it exactly as it was and the caller
    // simply did not get a checkpoint.
    auto entry            = std::make_unique<ignis_seq_checkpoint>();
    const std::uint64_t state_bytes =
        ignis_seq_prefix_clone_bytes(ignis_seq_prefix_clone_layout(*pool));
    entry->image     = ninfer::DeviceBuffer(static_cast<std::size_t>(state_bytes));
    entry->tail_page = ninfer::DeviceBuffer(
        static_cast<std::size_t>(ignis_seq_checkpoint_page_bytes(*pool)));
    entry->tokens    = opener_tokens;

    const std::int32_t own_page = seq->kv.page_ids()[0];
    timed([&] {
      ignis_seq_state_transfer(*pool, static_cast<unsigned char *>(entry->image.p),
                               entry->progress, *seq, IGNIS_SEQ_PREFIX_CAPTURE);
      ignis_seq_checkpoint_page_transfer(*pool, own_page,
                                         static_cast<unsigned char *>(entry->tail_page.p),
                                         IGNIS_SEQ_PREFIX_CAPTURE);
    });

    // Nothing below can fail. The checkpoint takes one reference to the
    // prefix under it: that reference, not the pages themselves, is what
    // outlives the request that warmed them.
    entry->prefix = seq->prefix;
    ++entry->prefix->refcount;
    *out_checkpoint = entry.release();
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
      ignis_seq_state_transfer(*pool, static_cast<unsigned char *>(checkpoint->image.p),
                               checkpoint->progress, *seq, IGNIS_SEQ_PREFIX_CLONE);
      // The partial page the opener ends inside, into the first page this
      // sequence owns. `ignis_seq_alloc_against_prefix` zeroed that page a
      // moment ago; this overwrites the part of it that is history, and the
      // rest stays zero for the tail the claimant prefills itself.
      ignis_seq_checkpoint_page_transfer(*pool, own_page,
                                         static_cast<unsigned char *>(checkpoint->tail_page.p),
                                         IGNIS_SEQ_PREFIX_CLONE);
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
      !checkpoint->prefix->kv.belongs_to(pool->kv_pool)) {
    ignis_seq_set_last_error("ignis_seq_checkpoint_release: the checkpoint was not captured from "
                             "this pool; nothing was released");
    return;
  }
  // The device images go with the entry; the prefix under it loses one
  // holder, and its pages come back only if that was the last.
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
  out_stats->image_bytes =
      static_cast<std::uint64_t>(checkpoint->image.bytes) + checkpoint->tail_page.bytes;
  out_stats->claim_count       = checkpoint->claim_count;
  out_stats->last_claim_micros = checkpoint->last_claim_micros;
  return 0;
}
