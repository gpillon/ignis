// GitHub #622 (spec #618, ADR 0095): a sequence pool built for the 27B's MTP
// head -- OURS.
//
// The head's state travels with the sequence wherever its state goes (ADR
// 0024's "by all or by none"): its layer in every KV page, under hq-e8-2b its
// layer of the residual window and its own ring words, and the continuation
// hidden, a section of its own. On the device, under both KV formats:
// - a snapshot and its restore carry all of it, a window at a time as whole,
//   and a restored sequence snapshots to the source's bytes;
// - a blob of a pool built for the head and one of any other pool are each
//   refused by the other by name ("the MTP head's state");
// - a prefix's last page is every holder's own: the publisher and each
//   claimant get a copy of the page as it stood at the publish, the page
//   below it is the last one shared, and a holder's write of the head's entry
//   at the prefix's last position reaches no other holder -- through a chained
//   prefix too, whose claimant reads its publisher's entry, part of its own
//   history; the copy goes with the prefix's handle;
// - a checkpoint copies the page of its opener's last position even on a
//   page boundary, lends the pages below it, and hands each claimant its own
//   copy and the opener's continuation hidden and ring words, through a host
//   retained slot;
// - a spilled prefix and a spilled checkpoint restore with every page.
// The program writes these sections; here the test writes known bytes into
// them and moves the frontier by hand, as a program's chunk would.
//
// GPU test (ADR 0006): no SKIP_RETURN_CODE, a missing device fails.

#include "ignis_model.h"
#include "ignis_seq.h"
#include "ignis_seq_checkpoint_internal.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_prefix_internal.h"
#include "ignis_seq_sections.h"
#include "seq_window_test_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string &label) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", label.c_str());
    ++g_failed;
  }
}

std::vector<unsigned char> read(const void *device, std::size_t bytes) {
  std::vector<unsigned char> host(bytes);
  cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost);
  return host;
}

constexpr uint32_t kPage = 64;
// Two target attention layers and the head's.
constexpr uint32_t kLayers = 3;

// Every plane's bytes of physical page `page`.
std::vector<unsigned char> page_bytes(const ignis_seq_pool &pool, int32_t page) {
  std::vector<unsigned char> out;
  for (std::size_t i = 0; i < pool.kv_pool.plane_count(); ++i) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(i);
    const auto bytes = static_cast<std::size_t>(plane.nb[3]);
    const auto page_bytes = read(static_cast<const unsigned char *>(plane.data) + page * plane.nb[3], bytes);
    out.insert(out.end(), page_bytes.begin(), page_bytes.end());
  }
  return out;
}

void fill_page(const ignis_seq_pool &pool, int32_t page, unsigned char value) {
  for (std::size_t i = 0; i < pool.kv_pool.plane_count(); ++i) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(i);
    cudaMemset(static_cast<unsigned char *>(plane.data) + page * plane.nb[3], value + static_cast<int>(i),
               static_cast<std::size_t>(plane.nb[3]));
  }
}

// The planes of the head's layer, in the pool's order.
std::vector<std::size_t> head_planes(const ignis_seq_pool &pool) {
  std::vector<std::size_t> planes;
  const int32_t layer = pool.mtp_head_layer();
  if (pool.kv_format == IGNIS_KV_FORMAT_HQ_E8_2B) {
    for (const auto role : {IGNIS_KV_PLANE_K, IGNIS_KV_PLANE_K_META, IGNIS_KV_PLANE_V, IGNIS_KV_PLANE_V_META}) {
      planes.push_back(ignis_kv_plane_index(pool.kv_format, layer, role));
    }
  } else {
    for (const auto role : {IGNIS_KV_PLANE_K, IGNIS_KV_PLANE_V}) {
      planes.push_back(ignis_kv_plane_index(pool.kv_format, layer, role));
    }
  }
  return planes;
}

// The head's row at `offset` of physical page `page`: every head plane, every
// KV head.
std::vector<unsigned char> head_row(const ignis_seq_pool &pool, int32_t page, int32_t offset) {
  std::vector<unsigned char> out;
  for (const std::size_t index : head_planes(pool)) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    for (int32_t head = 0; head < plane.ne[2]; ++head) {
      const auto *at = static_cast<const unsigned char *>(plane.data) + page * plane.nb[3] + offset * plane.nb[1] +
                       head * plane.nb[2];
      const auto row = read(at, static_cast<std::size_t>(plane.nb[1]));
      out.insert(out.end(), row.begin(), row.end());
    }
  }
  return out;
}

void set_head_row(const ignis_seq_pool &pool, int32_t page, int32_t offset, unsigned char value) {
  for (const std::size_t index : head_planes(pool)) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    for (int32_t head = 0; head < plane.ne[2]; ++head) {
      cudaMemset(static_cast<unsigned char *>(plane.data) + page * plane.nb[3] + offset * plane.nb[1] +
                     head * plane.nb[2],
                 value, static_cast<std::size_t>(plane.nb[1]));
    }
  }
}

bool all(const std::vector<unsigned char> &bytes, unsigned char value) {
  for (const unsigned char b : bytes) {
    if (b != value) {
      return false;
    }
  }
  return true;
}

// The head's slot state: the continuation hidden, and under hq-e8-2b the
// target's and the head's ring words.
struct HeadState {
  std::vector<unsigned char> continuation, ring, head_ring;
  bool operator==(const HeadState &other) const {
    return continuation == other.continuation && ring == other.ring && head_ring == other.head_ring;
  }
};

HeadState head_state_of(const ignis_seq_pool &pool, int32_t slot) {
  HeadState s;
  s.continuation = read(pool.mtp_continuation_of(slot), pool.mtp_continuation_slot_bytes());
  if (pool.has_hq_residual()) {
    s.ring = read(pool.hq_ring_words(slot), kIgnisHqRingWords * sizeof(uint32_t));
    s.head_ring = read(pool.hq_head_ring_words(slot), kIgnisHqRingWords * sizeof(uint32_t));
  }
  return s;
}

void write_head_state(const ignis_seq_pool &pool, int32_t slot, unsigned char value) {
  cudaMemset(pool.mtp_continuation_of(slot), value, pool.mtp_continuation_slot_bytes());
  if (pool.has_hq_residual()) {
    cudaMemset(pool.hq_ring_words(slot), value + 1, kIgnisHqRingWords * sizeof(uint32_t));
    cudaMemset(pool.hq_head_ring_words(slot), value + 2, kIgnisHqRingWords * sizeof(uint32_t));
  }
}

// What a program's completed chunk leaves: every frontier at `position`.
void stand_at(ignis_seq &seq, uint64_t position) {
  seq.position = position;
  seq.pending_token = 5;
  for (auto &f : seq.gqa_positions) f = static_cast<uint32_t>(position);
  for (auto &f : seq.gdn_positions) f = static_cast<uint32_t>(position);
}

// `seq`'s logical pages: its prefix chain's shared ones, then its own.
std::vector<int32_t> pages_of(const ignis_seq &seq) {
  std::vector<int32_t> pages = ignis_seq_prefix_chain_page_ids(seq.prefix);
  const auto own = seq.kv.page_ids();
  pages.insert(pages.end(), own.begin(), own.end());
  return pages;
}

std::vector<unsigned char> snapshot(ignis_seq_pool *pool, const ignis_seq *seq, const std::string &what) {
  uint64_t bytes = 0;
  if (ignis_seq_snapshot_size(pool, seq, &bytes) != 0) {
    check(false, what + ": snapshot size: " + ignis_seq_last_error());
    return {};
  }
  std::vector<unsigned char> blob(bytes);
  check(ignis_seq_snapshot(pool, seq, blob.data(), blob.size(), nullptr) == 0,
        what + ": snapshot: " + ignis_seq_last_error());
  return blob;
}

bool lists(const std::vector<unsigned char> &blob, int32_t kind) {
  ignis_seq_snapshot_header header{};
  std::memcpy(&header, blob.data(), sizeof(header));
  for (uint32_t i = 0; i < header.section_count; ++i) {
    ignis_seq_section record{};
    std::memcpy(&record, blob.data() + sizeof(header) + i * sizeof(record), sizeof(record));
    if (record.kind == kind) {
      return true;
    }
  }
  return false;
}

// A small 27B-shaped pool: two target attention layers and the head's, two
// GDN layers, 16 pages, 4 lanes, two device retained slots and a host one.
ignis_seq_pool_spec small_spec(int32_t kv_format, int32_t backend, uint32_t layers) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads = 2;
  spec.head_dim = 256;
  spec.kv_format = kv_format;
  spec.kv_page_group_count = 16;
  spec.max_context_tokens = 320;
  spec.slot_count = 4;
  spec.gdn_num_layers = 2;
  spec.gdn_conv_channels = 10240;
  spec.gdn_value_heads = 48;
  spec.gdn_head_dim = 128;
  spec.vocab = 1024;
  spec.kv_num_layers = layers;
  spec.speculative_backend = backend;
  spec.retained_slot_count = 2;
  spec.retained_host_slot_count = 1;
  return spec;
}

ignis_seq_pool *create(const ignis_seq_pool_spec &spec, const std::string &what) {
  ignis_seq_pool *pool = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0) {
    check(false, what + ": pool create: " + ignis_seq_last_error());
    return nullptr;
  }
  return pool;
}

uint32_t free_pages(const ignis_seq_pool *pool) {
  struct ignis_seq_pool_stats stats{};
  ignis_seq_pool_stats(pool, &stats);
  return stats.kv_free_pages;
}

// A blob of one kind of pool restored into the other is refused by name.
void check_named_refusal(ignis_seq_pool *into, const std::vector<unsigned char> &blob, const std::string &what) {
  ignis_seq *target = nullptr;
  if (ignis_seq_alloc(into, 256, &target) != 0) {
    check(false, what + ": alloc the target: " + ignis_seq_last_error());
    return;
  }
  const int32_t rc = ignis_seq_restore(into, target, blob.data(), blob.size(), nullptr);
  const std::string message = ignis_seq_last_error();
  check(rc == IGNIS_SEQ_ERR_BAD_SNAPSHOT && message.find("the MTP head's state") != std::string::npos,
        what + " is refused by name: got " + std::to_string(rc) + " \"" + message + "\"");
  check(target->position == 0, what + ": the target sequence is unchanged");
  ignis_seq_release(into, target);
}

void run(int32_t kv_format) {
  const std::string fmt = kv_format == IGNIS_KV_FORMAT_HQ_E8_2B ? "hq-e8-2b" : "bf16";
  const auto at = [&](const std::string &what) { return fmt + ": " + what; };
  ignis_seq_pool *pool = create(small_spec(kv_format, IGNIS_SPECULATIVE_MTP, kLayers), at("MTP pool"));
  if (pool == nullptr) {
    return;
  }
  check(pool->has_mtp_head() && pool->holder_lag() == 1, at("the pool is built for the head"));
  check(ignis_seq_pool_snapshot_format_version(pool) == kIgnisSeqSnapshotFormatVersion,
        at("a pool built for the head writes the 27B's layout version"));

  // --- a slot's head state is zero for its next sequence --------------------
  {
    ignis_seq *first = nullptr;
    check(ignis_seq_alloc(pool, 256, &first) == 0, at(std::string("alloc: ") + ignis_seq_last_error()));
    const int32_t reused = first->slot;
    write_head_state(*pool, reused, 0x5A);
    cudaDeviceSynchronize();
    ignis_seq_release(pool, first);
    ignis_seq *again = nullptr;
    check(ignis_seq_alloc(pool, 256, &again) == 0 && again->slot == reused, at("the freed slot comes back"));
    const HeadState zero = head_state_of(*pool, reused);
    check(all(zero.continuation, 0) && all(zero.ring, 0) && all(zero.head_ring, 0),
          at("the continuation hidden and both rings' words are zero for the slot's next sequence"));
    ignis_seq_release(pool, again);
  }

  // --- a snapshot carries the head's state, whole and window by window ------
  ignis_seq *a = nullptr;
  check(ignis_seq_alloc(pool, 256, &a) == 0, at(std::string("alloc A: ") + ignis_seq_last_error()));
  const std::vector<int32_t> a_pages = pages_of(*a);
  for (std::size_t i = 0; i < 2; ++i) {
    fill_page(*pool, a_pages[i], static_cast<unsigned char>(0x10 * (i + 1)));
  }
  // The head's entry at 127, the last position, waits for the token after it.
  set_head_row(*pool, a_pages[1], kPage - 1, 0);
  write_head_state(*pool, a->slot, 0x20);
  stand_at(*a, 128);
  cudaDeviceSynchronize();
  const HeadState a_head = head_state_of(*pool, a->slot);
  const std::vector<unsigned char> a_page1 = page_bytes(*pool, a_pages[1]);
  const std::vector<unsigned char> blob = snapshot(pool, a, at("A"));
  check(lists(blob, IGNIS_SEQ_SECTION_MTP_CONTINUATION), at("the blob lists the continuation hidden"));
  for (const uint64_t window : seq_window_sizes()) {
    check(seq_take_windows(pool, blob.size(), window,
                           [&](unsigned char *dst, uint64_t n, const ignis_seq_transfer &t) {
                             return ignis_seq_snapshot(pool, a, dst, n, &t);
                           }) == blob,
          at("the windows are the whole call's bytes (window " + std::to_string(window) + ")"));
  }
  {
    ignis_seq *e = nullptr;
    check(ignis_seq_alloc(pool, 256, &e) == 0, at(std::string("alloc E: ") + ignis_seq_last_error()));
    check(ignis_seq_restore(pool, e, blob.data(), blob.size(), nullptr) == 0,
          at(std::string("restore: ") + ignis_seq_last_error()));
    check(head_state_of(*pool, e->slot) == a_head, at("the restored continuation hidden and ring words are A's"));
    const std::vector<int32_t> e_pages = pages_of(*e);
    check(page_bytes(*pool, e_pages[0]) == page_bytes(*pool, a_pages[0]) && page_bytes(*pool, e_pages[1]) == a_page1,
          at("the restored pages carry every layer, the head's included"));
    check(snapshot(pool, e, at("E")) == blob, at("the restored sequence snapshots to A's bytes"));
    ignis_seq_release(pool, e);
    ignis_seq *w = nullptr;
    check(ignis_seq_alloc(pool, 256, &w) == 0, at(std::string("alloc W: ") + ignis_seq_last_error()));
    check(seq_restore_windows(pool, w, blob, 4096 * 3 + 1000) == 0,
          at(std::string("a restore fed window by window: ") + ignis_seq_last_error()));
    check(head_state_of(*pool, w->slot) == a_head && snapshot(pool, w, at("W")) == blob,
          at("the windowed restore holds A's state and snapshots to A's bytes"));
    ignis_seq_release(pool, w);
  }

  // --- a blob of another kind of pool is refused by name, both ways ---------
  {
    ignis_seq_pool *plain = create(small_spec(kv_format, IGNIS_SPECULATIVE_NONE, kLayers - 1), at("plain pool"));
    ignis_seq_pool *dflash2 = create(small_spec(kv_format, IGNIS_SPECULATIVE_DFLASH2, kLayers - 1), at("DFlash2 pool"));
    if (plain != nullptr && dflash2 != nullptr) {
      check_named_refusal(plain, blob, at("an MTP blob on a pool with no drafter"));
      check_named_refusal(dflash2, blob, at("an MTP blob on a DFlash2 pool"));
      for (ignis_seq_pool *other : {plain, dflash2}) {
        ignis_seq *s = nullptr;
        check(ignis_seq_alloc(other, 256, &s) == 0, at("alloc on the other pool"));
        stand_at(*s, 100);
        const std::vector<unsigned char> theirs = snapshot(other, s, at("the other pool's sequence"));
        check(!lists(theirs, IGNIS_SEQ_SECTION_MTP_CONTINUATION), at("a pool without the head lists no head section"));
        check_named_refusal(pool, theirs,
                            at(other == plain ? "a blob with no drafter on an MTP pool" : "a DFlash2 blob on an MTP pool"));
        ignis_seq_release(other, s);
      }
    }
    ignis_seq_pool_free(plain);
    ignis_seq_pool_free(dflash2);
  }

  // --- a prefix's last page is every holder's own ---------------------------
  const uint32_t free_before_publish = free_pages(pool);
  ignis_seq_prefix *prefix = nullptr;
  check(ignis_seq_prefix_publish(pool, a, 128, 0, &prefix) == 0,
        at(std::string("publish 128 tokens: ") + ignis_seq_last_error()));
  if (prefix == nullptr) {
    ignis_seq_release(pool, a);
    ignis_seq_pool_free(pool);
    return;
  }
  check(free_before_publish - free_pages(pool) == 1, at("the publish takes one page, the copy of the last one"));
  struct ignis_seq_prefix_stats pstats{};
  check(ignis_seq_prefix_stats(prefix, &pstats) == 0 && pstats.pages == 2 && pstats.tokens == 128,
        at("the prefix holds its shared page and the copy of its last"));
  const std::vector<int32_t> a_after = pages_of(*a);
  const int32_t last_copy = prefix->last.page_ids()[0];
  check(a->shared_pages == 1 && a_after[0] == a_pages[0] && a->kv.page_ids()[0] == a_after[1] &&
            a_after[1] != last_copy,
        at("the publisher shares page 0 and holds page 1 as its own, not the prefix's copy"));
  check(page_bytes(*pool, a_after[1]) == a_page1, at("the publisher's own page 1 is the page it published"));
  check(head_state_of(*pool, a->slot) == a_head, at("the publisher's head state is untouched"));
  check(page_bytes(*pool, last_copy) == a_page1, at("the prefix's copy of its last page is the published page"));
  // The publisher's next step writes the head's entry at 127 from its token.
  set_head_row(*pool, a_after[1], kPage - 1, 0xA1);
  stand_at(*a, 160);
  cudaDeviceSynchronize();

  ignis_seq *c = nullptr;
  check(ignis_seq_alloc_shared(pool, 320, prefix, &c) == 0, at(std::string("claim: ") + ignis_seq_last_error()));
  if (c != nullptr) {
    const std::vector<int32_t> c_pages = pages_of(*c);
    check(c->shared_pages == 1 && c_pages[0] == a_pages[0] && c_pages[1] != a_after[1] && c_pages[1] != last_copy,
          at("the claimant shares page 0 and holds its own page 1"));
    check(page_bytes(*pool, c_pages[1]) == a_page1 && all(head_row(*pool, c_pages[1], kPage - 1), 0),
          at("the claimant's page 1 is the published page: no head entry at 127, not the publisher's"));
    check(head_state_of(*pool, c->slot) == a_head && c->position == 128,
          at("the claimant holds the continuation hidden and ring words at the prefix, and stands there"));
    set_head_row(*pool, c_pages[1], kPage - 1, 0xC1);
    cudaDeviceSynchronize();
    check(all(head_row(*pool, a_after[1], kPage - 1), 0xA1) && page_bytes(*pool, last_copy) == a_page1,
          at("the claimant's entry at 127 reaches neither the publisher nor the prefix's copy"));
    ignis_seq *d = nullptr;
    check(ignis_seq_alloc_shared(pool, 256, prefix, &d) == 0, at("a second claim"));
    if (d != nullptr) {
      check(all(head_row(*pool, pages_of(*d)[1], kPage - 1), 0), at("a second claimant's entry at 127 is its own too"));
      ignis_seq_release(pool, d);
    }

    // A snapshot of the claimant materializes the shared page and its own.
    const std::vector<unsigned char> c_blob = snapshot(pool, c, at("C"));
    ignis_seq *r = nullptr;
    check(ignis_seq_alloc(pool, 256, &r) == 0 &&
              ignis_seq_restore(pool, r, c_blob.data(), c_blob.size(), nullptr) == 0,
          at(std::string("restore C's blob: ") + ignis_seq_last_error()));
    if (r != nullptr) {
      check(all(head_row(*pool, pages_of(*r)[1], kPage - 1), 0xC1), at("C's blob carries C's own entry at 127"));
      ignis_seq_release(pool, r);
    }

    // --- a chained prefix: its claimant reads its publisher's entry --------
    const std::vector<int32_t> c_now = pages_of(*c);
    for (std::size_t i = 2; i < 4; ++i) {
      fill_page(*pool, c_now[i], static_cast<unsigned char>(0x60 + i));
    }
    set_head_row(*pool, c_now[3], kPage - 1, 0);
    stand_at(*c, 256);
    cudaDeviceSynchronize();
    const std::vector<unsigned char> c_page3 = page_bytes(*pool, c_now[3]);
    ignis_seq_prefix *chained = nullptr;
    check(ignis_seq_prefix_publish(pool, c, 256, 1, &chained) == 0,
          at(std::string("a chained publish at 256: ") + ignis_seq_last_error()));
    if (chained != nullptr) {
      check(c->shared_pages == 3 && ignis_seq_prefix_total_pages(*chained) == 3,
            at("the chained head shares three pages, the fourth every holder's own"));
      ignis_seq *f = nullptr;
      check(ignis_seq_alloc_shared(pool, 320, chained, &f) == 0,
            at(std::string("claim the chained prefix: ") + ignis_seq_last_error()));
      if (f != nullptr) {
        const std::vector<int32_t> f_pages = pages_of(*f);
        check(f_pages[0] == a_pages[0] && f_pages[1] == c_now[1] && f_pages[2] == c_now[2] && f_pages[3] != c_now[3],
              at("the chained claimant shares the prefix's page, the chained publisher's two, and holds page 3"));
        check(all(head_row(*pool, f_pages[1], kPage - 1), 0xC1),
              at("and reads the chained publisher's entry at 127, which its history is"));
        check(page_bytes(*pool, f_pages[3]) == c_page3, at("its page 3 is the chained publish's last page"));
        ignis_seq_release(pool, f);
      }
      ignis_seq_prefix_release(pool, chained);
    }
    ignis_seq_release(pool, c);
  }

  // --- a spilled prefix restores with every page ----------------------------
  {
    uint64_t bytes = 0;
    check(ignis_seq_prefix_snapshot_size(pool, prefix, &bytes) == 0, at("prefix blob size"));
    std::vector<unsigned char> prefix_blob(bytes);
    check(ignis_seq_prefix_snapshot(pool, prefix, prefix_blob.data(), prefix_blob.size(), nullptr) == 0,
          at(std::string("prefix blob: ") + ignis_seq_last_error()));
    check(seq_take_windows(pool, prefix_blob.size(), 4096,
                           [&](unsigned char *dst, uint64_t n, const ignis_seq_transfer &t) {
                             return ignis_seq_prefix_snapshot(pool, prefix, dst, n, &t);
                           }) == prefix_blob,
          at("the prefix blob, window by window"));
    ignis_seq *carrier = nullptr;
    check(ignis_seq_alloc(pool, 256, &carrier) == 0 &&
              ignis_seq_restore(pool, carrier, prefix_blob.data(), prefix_blob.size(), nullptr) == 0,
          at(std::string("restore the prefix blob: ") + ignis_seq_last_error()));
    if (carrier != nullptr) {
      const std::vector<int32_t> pages = pages_of(*carrier);
      check(carrier->position == 128 && page_bytes(*pool, pages[1]) == a_page1 &&
                head_state_of(*pool, carrier->slot) == a_head,
            at("the carrier stands at 128 on the published pages and head state"));
      ignis_seq_release(pool, carrier);
    }
  }

  // --- the copy goes with the prefix's handle --------------------------------
  {
    const uint32_t free_before = free_pages(pool);
    ignis_seq_prefix_release(pool, prefix);
    check(free_pages(pool) - free_before == 1,
          at("releasing the handle gives the copy of the last page back while the publisher stands on the prefix"));
    ignis_seq_release(pool, a);
  }

  // --- a checkpoint on a page boundary copies its last position's page -----
  ignis_seq *g = nullptr;
  check(ignis_seq_alloc(pool, 256, &g) == 0, at(std::string("alloc G: ") + ignis_seq_last_error()));
  const std::vector<int32_t> g_pages = pages_of(*g);
  fill_page(*pool, g_pages[0], 0x30);
  fill_page(*pool, g_pages[1], 0x40);
  set_head_row(*pool, g_pages[1], kPage - 1, 0);
  write_head_state(*pool, g->slot, 0x50);
  stand_at(*g, 128);
  cudaDeviceSynchronize();
  const HeadState g_head = head_state_of(*pool, g->slot);
  const std::vector<unsigned char> g_page1 = page_bytes(*pool, g_pages[1]);
  ignis_seq_checkpoint *checkpoint = nullptr;
  check(ignis_seq_checkpoint_capture(pool, g, 128, 2, &checkpoint) == 0,
        at(std::string("capture at 128 into the host retained slot: ") + ignis_seq_last_error()));
  if (checkpoint != nullptr) {
    check(checkpoint->tail.valid() && page_bytes(*pool, checkpoint->tail.page_ids()[0]) == g_page1,
          at("a checkpoint on a page boundary copies the page of its last position"));
    check(g->lent_to != nullptr && g->lent_to->lent.size() == 1 && g->lent_to->lent[0] == g_pages[0],
          at("and lends only the page below it"));
    set_head_row(*pool, g_pages[1], kPage - 1, 0x61);
    cudaDeviceSynchronize();
    ignis_seq *h = nullptr;
    check(ignis_seq_alloc_from_checkpoint(pool, 256, checkpoint, &h) == 0,
          at(std::string("claim the checkpoint: ") + ignis_seq_last_error()));
    if (h != nullptr) {
      const std::vector<int32_t> h_pages = pages_of(*h);
      check(h->shared_pages == 1 && h_pages[0] == g_pages[0] && h_pages[1] != g_pages[1],
            at("the claimant shares page 0 and holds its own page 1"));
      check(page_bytes(*pool, h_pages[1]) == g_page1, at("its page 1 is the capture's, not the capturer's later entry"));
      check(head_state_of(*pool, h->slot) == g_head && h->position == 128,
            at("it holds the opener's head state through the host retained slot, and stands at the opener"));
      ignis_seq_release(pool, h);
    }
    ignis_seq *tight = nullptr;
    check(ignis_seq_alloc_from_checkpoint(pool, 128, checkpoint, &tight) != 0 && tight == nullptr,
          at("a claim with no room past an opener on a page boundary is refused"));
    uint64_t bytes = 0;
    check(ignis_seq_checkpoint_snapshot_size(pool, checkpoint, &bytes) == 0, at("checkpoint blob size"));
    std::vector<unsigned char> ckpt_blob(bytes);
    check(ignis_seq_checkpoint_snapshot(pool, checkpoint, ckpt_blob.data(), ckpt_blob.size(), nullptr) == 0,
          at(std::string("checkpoint blob: ") + ignis_seq_last_error()));
    ignis_seq *k = nullptr;
    check(ignis_seq_alloc(pool, 256, &k) == 0 &&
              ignis_seq_restore(pool, k, ckpt_blob.data(), ckpt_blob.size(), nullptr) == 0,
          at(std::string("restore the checkpoint blob: ") + ignis_seq_last_error()));
    if (k != nullptr) {
      const std::vector<int32_t> k_pages = pages_of(*k);
      check(page_bytes(*pool, k_pages[0]) == page_bytes(*pool, g_pages[0]) && page_bytes(*pool, k_pages[1]) == g_page1 &&
                head_state_of(*pool, k->slot) == g_head,
            at("the checkpoint blob carries both pages and the opener's head state"));
      ignis_seq_release(pool, k);
    }
    ignis_seq_checkpoint_release(pool, checkpoint);
  }
  ignis_seq_release(pool, g);

  // --- a checkpoint inside a page copies that page, as before ---------------
  {
    ignis_seq *p = nullptr;
    check(ignis_seq_alloc(pool, 256, &p) == 0, at("alloc P"));
    stand_at(*p, 65);
    ignis_seq_checkpoint *inside = nullptr;
    check(ignis_seq_checkpoint_capture(pool, p, 65, 2, &inside) == 0,
          at(std::string("capture at 65: ") + ignis_seq_last_error()));
    if (inside != nullptr) {
      check(inside->tail.valid() && p->lent_to != nullptr && p->lent_to->lent.size() == 1,
            at("an opener at 65 copies page 1, which holds its last position, and lends page 0"));
      ignis_seq_checkpoint_release(pool, inside);
    }
    ignis_seq_release(pool, p);
    ignis_seq *q = nullptr;
    check(ignis_seq_alloc(pool, 256, &q) == 0, at("alloc Q"));
    stand_at(*q, 64);
    ignis_seq_checkpoint *first_page = nullptr;
    check(ignis_seq_checkpoint_capture(pool, q, 64, 2, &first_page) != 0 && first_page == nullptr,
          at("an opener whose last position is in the first page has no whole page below it"));
    ignis_seq_release(pool, q);
  }

  struct ignis_seq_pool_stats stats{};
  check(ignis_seq_pool_stats(pool, &stats) == 0 && stats.kv_free_pages == 16, at("every page came back"));
  ignis_seq_pool_free(pool);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device\n");
    return 1;
  }
  run(IGNIS_KV_FORMAT_BF16);
  run(IGNIS_KV_FORMAT_HQ_E8_2B);
  if (g_failed != 0) {
    std::fprintf(stderr, "seq mtp sections test: %d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("seq mtp sections test: ok\n");
  return 0;
}
