// Leaf-level sequence state-transfer test (P4-06, GitHub #124, ADR 0024).
//
// Ours, not vendored. It covers the three things `ignis_seq.h`'s snapshot
// surface promises and the flat ABI alone cannot show:
//
//   1. the state-section table enumerates every section of a sequence with
//      its size and its shareable-or-clone class, and the blob a snapshot
//      writes is exactly that table (kernel/include/ignis_seq_sections.h);
//   2. a whole sequence round-trips -- a dirtied sequence, released and
//      re-allocated as a zeroed one, restores to a byte-identical image;
//   3. every refusal refuses: a stale version, a foreign buffer, a size
//      that disagrees with the blob's own header, a geometry the target
//      does not match, a target too small, and a mid-chunk source -- each
//      leaving the target sequence untouched.
//
// It also reports the transfer's measured cost at the real 27B geometry for
// a short and a full-context sequence, which is what the ticket's "measured
// rather than assumed" asks for: the numbers print on every run and are
// recorded in docs/findings/2026-09-12-sequence-snapshot-transfer-cost.md.
//
// The sequence's history is set here by writing `position` / `gqa_positions`
// through ignis_seq_internal.h and filling its device state with a known
// pattern, rather than by running a prefill: what is under test is the
// transfer of bytes, which does not depend on the bytes having come from a
// real forward pass. The end-to-end claim -- that a restored sequence
// continues decoding to the *same tokens* -- is proven one level up, in
// crates/core/tests/seq_snapshot_gpu.rs, against the real model.
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE is set on this test
// (kernel/tests/CMakeLists.txt), so a missing/busy GPU fails it, never skips.

#include "ignis_seq.h"
#include "ignis_seq_internal.h"
#include "ignis_seq_sections.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char *label) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
  }
}

void expect_rc(int32_t rc, int32_t want, const char *label) {
  if (rc != want) {
    std::fprintf(stderr, "FAIL: %s (rc=%d, want %d: %s)\n", label, rc, want,
                 ignis_seq_last_error());
    ++failures;
  }
}

bool cuda_unavailable(cudaError_t err) {
  return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

// A deterministic byte pattern, so a restored section can be compared
// against the source it came from without keeping a host copy of every
// section separately.
std::vector<unsigned char> pattern(std::size_t bytes, std::uint32_t seed) {
  std::vector<unsigned char> host(bytes);
  std::uint32_t x = seed | 1U;
  for (unsigned char &b : host) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    b = static_cast<unsigned char>(x);
  }
  return host;
}

void fill_device(void *dst, std::size_t bytes, std::uint32_t seed) {
  if (bytes == 0) {
    return;
  }
  const std::vector<unsigned char> host = pattern(bytes, seed);
  CUDA_CHECK(cudaMemcpy(dst, host.data(), bytes, cudaMemcpyHostToDevice));
}

// Put `seq` at a completed chunk boundary `tokens` in: the program frontier
// and every layer's own, GDN included (a GDN layer's state is updated in
// place, so its counter is the only thing that can say it has consumed the
// same tokens the KV pages have).
void set_frontier(ignis_seq &seq, std::uint64_t tokens) {
  seq.position = tokens;
  for (std::uint32_t &frontier : seq.gqa_positions) {
    frontier = static_cast<std::uint32_t>(tokens);
  }
  for (std::uint32_t &frontier : seq.gdn_positions) {
    frontier = static_cast<std::uint32_t>(tokens);
  }
}

// Give `seq` a history: a chunk boundary `tokens` in, a pending token, and a
// known pattern in every section of its device state.
void give_history(ignis_seq_pool &pool, ignis_seq &seq, std::uint64_t tokens,
                  std::uint32_t seed) {
  set_frontier(seq, tokens);
  seq.pending_token = static_cast<std::int32_t>(1000 + seed);

  std::uint32_t salt = seed;
  for (std::size_t plane_index = 0; plane_index < pool.kv_pool.plane_count(); ++plane_index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(plane_index);
    auto *base                  = static_cast<unsigned char *>(plane.data);
    for (std::int32_t page_id : seq.kv.page_ids()) {
      fill_device(base + static_cast<std::int64_t>(page_id) * plane.nb[3],
                  static_cast<std::size_t>(plane.nb[3]), ++salt);
    }
  }
  for (std::uint32_t layer = 0; layer < pool.gdn_pool.layer_count(); ++layer) {
    const ninfer::Tensor conv = pool.gdn_pool.conv_slot(layer, seq.slot);
    const ninfer::Tensor rec  = pool.gdn_pool.recurrent_slot(layer, seq.slot);
    fill_device(conv.data, conv.bytes(), ++salt);
    fill_device(rec.data, rec.bytes(), ++salt);
  }
  fill_device(pool.token_counts_for(seq.slot),
              static_cast<std::size_t>(pool.vocab) * sizeof(std::int32_t), ++salt);
}

// The whole blob for `seq`, sized by the leaf's own query.
std::vector<unsigned char> snapshot_of(const ignis_seq_pool &pool, const ignis_seq &seq,
                                       const char *label) {
  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(&pool, &seq, &bytes), 0, label);
  std::vector<unsigned char> blob(static_cast<std::size_t>(bytes));
  expect_rc(ignis_seq_snapshot(&pool, &seq, blob.data(), bytes), 0, label);
  return blob;
}

ignis_seq_snapshot_header header_of(const std::vector<unsigned char> &blob) {
  ignis_seq_snapshot_header header{};
  std::memcpy(&header, blob.data(), sizeof(header));
  return header;
}

void reseal(std::vector<unsigned char> &blob, ignis_seq_snapshot_header &header) {
  header.header_checksum = 0;
  header.header_checksum = ignis_seq_fnv1a(&header, sizeof(header));
  std::memcpy(blob.data(), &header, sizeof(header));
}

double ms_since(std::chrono::steady_clock::time_point began) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began)
      .count();
}

// A small, fast geometry -- the same shape test_seq_alloc.cpp uses, so the
// two tests exercise one pool spec between them.
ignis_seq_pool_spec small_spec() {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = 2;
  spec.head_dim            = 8;
  spec.kv_format           = IGNIS_KV_FORMAT_BF16;
  spec.kv_page_group_count = 8;
  spec.max_context_tokens  = 128; // pages_for_tokens(128) == 2
  spec.slot_count          = 3;
  spec.gdn_num_layers      = 2;
  spec.gdn_conv_channels   = 6;
  spec.gdn_value_heads     = 2;
  spec.gdn_head_dim        = 4;
  spec.vocab               = 32;
  return spec;
}

// The real Qwen 3.8-27B sequence geometry (crates/core/src/compute.rs's
// `ModelConfig::qwen38_27b`), which is what makes the cost figures below
// the engine's own rather than a scaled-down proxy.
ignis_seq_pool_spec qwen38_27b_spec(int32_t kv_format, std::uint32_t context_tokens,
                                    std::uint32_t slot_count) {
  ignis_seq_pool_spec spec{};
  spec.num_kv_heads        = 4;
  spec.head_dim            = 256;
  spec.kv_format           = kv_format;
  spec.kv_page_group_count = ninfer::pages_for_tokens(context_tokens) * slot_count;
  spec.max_context_tokens  = context_tokens;
  spec.slot_count          = slot_count;
  spec.gdn_num_layers      = 48;
  spec.gdn_conv_channels   = 10240; // q 2048 + k 2048 + v 6144
  spec.gdn_value_heads     = 48;
  spec.gdn_head_dim        = 128;
  spec.vocab               = 248320;
  return spec;
}

// ---- 1. the section table --------------------------------------------------

void check_section_table() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "table: pool create");
  ignis_seq *seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, 128, &seq), 0, "table: alloc");
  give_history(*pool, *seq, 100, 0x51u);

  const std::uint32_t pages = ignis_seq_snapshot_page_count(*seq);
  expect(pages == ninfer::pages_for_tokens(100),
        "table: a snapshot captures the pages the history occupies");
  expect(pages == 2, "table: 100 tokens over 64-token pages is 2 pages");

  const std::vector<ignis_seq_section> sections = ignis_seq_section_table(*pool, pages);
  expect(sections.size() == kIgnisSeqSectionCount, "table: every state section is listed");

  const std::int32_t want_kind[] = {IGNIS_SEQ_SECTION_KV_PAGES, IGNIS_SEQ_SECTION_GDN_CONV,
                                    IGNIS_SEQ_SECTION_GDN_RECURRENT,
                                    IGNIS_SEQ_SECTION_PENALTY_COUNTS,
                                    IGNIS_SEQ_SECTION_PROGRESS};
  // The classes ADR 0024 assigns: KV pages are read-only history and are
  // shared by refcount for prefix reuse; everything a restored sequence
  // writes from its first step must be cloned per sequence.
  const std::int32_t want_transfer[] = {IGNIS_SEQ_SECTION_SHAREABLE, IGNIS_SEQ_SECTION_CLONE,
                                        IGNIS_SEQ_SECTION_CLONE, IGNIS_SEQ_SECTION_CLONE,
                                        IGNIS_SEQ_SECTION_CLONE};
  for (std::size_t i = 0; i < sections.size() && i < 5; ++i) {
    expect(sections[i].kind == want_kind[i], "table: section kind in blob order");
    expect(sections[i].transfer == want_transfer[i], "table: section transfer class");
    expect(sections[i].bytes > 0, "table: every section of a live sequence has bytes");
  }

  // Every section's size is the leaf's own account of that piece of state,
  // not a restatement: each is checked against the pool that holds it.
  expect(sections[0].bytes == ninfer::paged_kv_host_image_bytes(pool->kv_pool, pages),
        "table: the KV section is the pages' host image");
  expect(sections[1].bytes == pool->gdn_pool.conv_host_image_bytes(),
        "table: the conv section is every layer's taps");
  expect(sections[2].bytes == pool->gdn_pool.recurrent_host_image_bytes(),
        "table: the recurrent section is every layer's state matrices");
  expect(sections[3].bytes ==
            static_cast<std::uint64_t>(spec.vocab) * sizeof(std::int32_t),
        "table: the penalty-count section is one int32 per vocab entry");
  expect(sections[4].bytes == sizeof(ignis_seq_progress_image),
        "table: the progress section is the scalars");

  // Sections do not overlap, and the reported size covers all of them.
  std::uint64_t previous_end = sizeof(ignis_seq_snapshot_header) +
                               sections.size() * sizeof(ignis_seq_section);
  for (const ignis_seq_section &section : sections) {
    expect(section.offset >= previous_end, "table: sections do not overlap the ones before them");
    expect(section.offset % kIgnisSeqSectionAlign == 0, "table: every section is aligned");
    previous_end = section.offset + section.bytes;
  }

  std::uint64_t reported = 0;
  expect_rc(ignis_seq_snapshot_size(pool, seq, &reported), 0, "table: snapshot size query");
  expect(reported >= previous_end, "table: the reported size covers every section");
  expect(reported == ignis_seq_snapshot_bytes(sections),
        "table: the reported size is the table's own total");

  // The blob is that table: header, records, payload.
  const std::vector<unsigned char> blob = snapshot_of(*pool, *seq, "table: snapshot");
  expect(blob.size() == reported, "table: a snapshot writes exactly the reported size");
  const ignis_seq_snapshot_header header = header_of(blob);
  expect(header.magic == kIgnisSeqSnapshotMagic, "table: the blob carries its own magic");
  expect(header.format_version == ignis_seq_snapshot_format_version(),
        "table: the blob's version is the one the ABI reports");
  expect(header.total_bytes == reported, "table: the blob records its own size");
  expect(header.section_count == sections.size(), "table: the blob records every section");
  expect(header.kv_page_count == pages, "table: the blob records its KV page count");
  std::vector<ignis_seq_section> records(header.section_count);
  std::memcpy(records.data(), blob.data() + sizeof(header),
              records.size() * sizeof(ignis_seq_section));
  for (std::size_t i = 0; i < records.size(); ++i) {
    expect(records[i].kind == sections[i].kind && records[i].transfer == sections[i].transfer &&
              records[i].offset == sections[i].offset && records[i].bytes == sections[i].bytes,
          "table: the blob's records are the leaf's table");
  }

  // The progress section really is the sequence's scalars.
  ignis_seq_progress_image progress{};
  std::memcpy(&progress, blob.data() + sections[4].offset, sizeof(progress));
  expect(progress.position == 100, "table: the progress section carries the frontier");
  expect(progress.pending_token == seq->pending_token,
        "table: the progress section carries the pending token");
  expect(progress.gqa_positions[0] == 100,
        "table: the progress section carries every GQA frontier");
  expect(progress.gdn_positions[kIgnisGdnLayerCount - 1] == 100,
        "table: the progress section carries every GDN frontier");

  // A destination one byte short is refused rather than truncated.
  std::vector<unsigned char> tight(static_cast<std::size_t>(reported) - 1);
  expect_rc(ignis_seq_snapshot(pool, seq, tight.data(), tight.size()), -1,
           "table: a destination below the reported size is refused");

  // The blob is the same bytes whatever was in the destination before: a
  // host tier reuses its pinned regions, and the gaps between sections are
  // part of the blob's extent.
  std::vector<unsigned char> dirty(static_cast<std::size_t>(reported), 0xEE);
  expect_rc(ignis_seq_snapshot(pool, seq, dirty.data(), dirty.size()), 0,
           "table: snapshot into a used buffer");
  expect(dirty == blob, "table: a snapshot into a dirty buffer is byte-identical to a clean one");

  ignis_seq_release(pool, seq);
  ignis_seq_pool_free(pool);
}

// ---- 2. the round trip -----------------------------------------------------

void check_round_trip(const ignis_seq_pool_spec &spec, std::uint32_t context_tokens,
                      std::uint64_t history_tokens, const char *label) {
  std::printf("round trip (%s)\n", label);
  ignis_seq_pool *pool = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "round trip: pool create");

  ignis_seq *source = nullptr;
  expect_rc(ignis_seq_alloc(pool, context_tokens, &source), 0, "round trip: alloc source");
  give_history(*pool, *source, history_tokens, 0xA5u);
  const std::vector<unsigned char> blob = snapshot_of(*pool, *source, "round trip: snapshot");
  const std::int32_t pending            = source->pending_token;

  // Release the source: its slot and pages go back to the pool, and the
  // next allocation gets them zeroed. Nothing of the sequence survives on
  // the device -- only the blob.
  ignis_seq_release(pool, source);

  ignis_seq *target = nullptr;
  expect_rc(ignis_seq_alloc(pool, context_tokens, &target), 0, "round trip: alloc target");
  expect(target->position == 0 && target->pending_token == -1,
        "round trip: a fresh sequence starts at zero");

  expect_rc(ignis_seq_restore(pool, target, blob.data(), blob.size()), 0,
           "round trip: restore");
  expect(target->position == history_tokens, "round trip: the frontier is restored");
  expect(target->pending_token == pending, "round trip: the pending token is restored");
  for (std::uint32_t frontier : target->gqa_positions) {
    expect(static_cast<std::uint64_t>(frontier) == history_tokens,
          "round trip: every GQA frontier is restored");
  }
  for (std::uint32_t frontier : target->gdn_positions) {
    expect(static_cast<std::uint64_t>(frontier) == history_tokens,
          "round trip: every GDN frontier is restored");
  }

  // The strongest statement available at this level: the restored sequence
  // snapshots to the same bytes. A section that did not round-trip -- or
  // one carried by snapshot and forgotten by restore -- shows up here.
  const std::vector<unsigned char> again = snapshot_of(*pool, *target, "round trip: re-snapshot");
  expect(again.size() == blob.size(), "round trip: the restored sequence costs the same");
  expect(again == blob, "round trip: the restored sequence is byte-identical to the source");

  ignis_seq_release(pool, target);
  ignis_seq_pool_free(pool);
}

// ---- 3. the refusals -------------------------------------------------------

// A refused restore must leave the target exactly as it was, so every case
// below snapshots the target first and re-snapshots it after.
void expect_refused(ignis_seq_pool *pool, ignis_seq *target,
                    const std::vector<unsigned char> &blob, std::uint64_t src_bytes,
                    const char *label) {
  const std::vector<unsigned char> before = snapshot_of(*pool, *target, label);
  expect_rc(ignis_seq_restore(pool, target, blob.data(), src_bytes), IGNIS_SEQ_ERR_BAD_SNAPSHOT,
           label);
  const std::vector<unsigned char> after = snapshot_of(*pool, *target, label);
  if (before != after) {
    std::fprintf(stderr, "FAIL: %s left the target sequence changed\n", label);
    ++failures;
  }
}

void check_refusals() {
  const ignis_seq_pool_spec spec = small_spec();
  ignis_seq_pool *pool           = nullptr;
  expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "refusals: pool create");

  ignis_seq *source = nullptr;
  expect_rc(ignis_seq_alloc(pool, 128, &source), 0, "refusals: alloc source");
  give_history(*pool, *source, 100, 0x11u);
  const std::vector<unsigned char> blob = snapshot_of(*pool, *source, "refusals: snapshot");

  ignis_seq *target = nullptr;
  expect_rc(ignis_seq_alloc(pool, 128, &target), 0, "refusals: alloc target");
  give_history(*pool, *target, 64, 0x22u);

  // A stale format version, resealed so the checksum agrees -- the version
  // is refused on its own merits, not because the blob looks corrupt.
  {
    std::vector<unsigned char> stale = blob;
    ignis_seq_snapshot_header header = header_of(stale);
    header.format_version += 1;
    reseal(stale, header);
    expect_refused(pool, target, stale, stale.size(), "refusals: a stale format version");
  }

  // A buffer that is not a snapshot at all.
  {
    std::vector<unsigned char> foreign = blob;
    ignis_seq_snapshot_header header    = header_of(foreign);
    header.magic                        = 0xDEADBEEFDEADBEEFULL;
    reseal(foreign, header);
    expect_refused(pool, target, foreign, foreign.size(), "refusals: a foreign buffer");
  }

  // A header edited without resealing: the checksum catches it.
  {
    std::vector<unsigned char> tampered = blob;
    ignis_seq_snapshot_header header     = header_of(tampered);
    header.kv_page_count                 = 1;
    std::memcpy(tampered.data(), &header, sizeof(header));
    expect_refused(pool, target, tampered, tampered.size(), "refusals: an edited header");
  }

  // A size that disagrees with the blob's own record, in either direction.
  expect_refused(pool, target, blob, blob.size() - 1, "refusals: a short byte count");
  {
    std::vector<unsigned char> padded = blob;
    padded.push_back(0);
    expect_refused(pool, target, padded, padded.size(), "refusals: a long byte count");
  }
  {
    std::vector<unsigned char> stub(sizeof(ignis_seq_snapshot_header) / 2, 0);
    expect_rc(ignis_seq_restore(pool, target, stub.data(), stub.size()),
             IGNIS_SEQ_ERR_BAD_SNAPSHOT, "refusals: a buffer shorter than a header");
  }

  // A section record edited to claim a different layout, resealed so only
  // the records disagree: the records are checked against the table this
  // leaf would build, so the layout is refused even with a valid header.
  {
    std::vector<unsigned char> relaid = blob;
    std::vector<ignis_seq_section> records(header_of(relaid).section_count);
    std::memcpy(records.data(), relaid.data() + sizeof(ignis_seq_snapshot_header),
                records.size() * sizeof(ignis_seq_section));
    records[1].bytes += kIgnisSeqSectionAlign;
    std::memcpy(relaid.data() + sizeof(ignis_seq_snapshot_header), records.data(),
                records.size() * sizeof(ignis_seq_section));
    expect_refused(pool, target, relaid, relaid.size(), "refusals: an edited section layout");
  }

  // A target with less room than the blob carries.
  {
    ignis_seq *small = nullptr;
    expect_rc(ignis_seq_alloc(pool, 64, &small), 0, "refusals: alloc a one-page target");
    expect_rc(ignis_seq_restore(pool, small, blob.data(), blob.size()),
             IGNIS_SEQ_ERR_BAD_SNAPSHOT, "refusals: a target that maps fewer pages");
    expect(small->position == 0, "refusals: the too-small target is unchanged");
    ignis_seq_release(pool, small);
  }

  // A pool of different geometry. Everything else about the blob is valid,
  // so this is the geometry check and nothing else.
  {
    ignis_seq_pool_spec wide = spec;
    wide.num_kv_heads        = 4;
    ignis_seq_pool *wide_pool = nullptr;
    expect_rc(ignis_seq_pool_create(&wide, &wide_pool), 0, "refusals: wide pool create");
    ignis_seq *wide_seq = nullptr;
    expect_rc(ignis_seq_alloc(wide_pool, 128, &wide_seq), 0, "refusals: wide alloc");
    expect_rc(ignis_seq_restore(wide_pool, wide_seq, blob.data(), blob.size()),
             IGNIS_SEQ_ERR_BAD_SNAPSHOT, "refusals: a foreign KV geometry");
    expect(wide_seq->position == 0, "refusals: the foreign-geometry target is unchanged");
    ignis_seq_release(wide_pool, wide_seq);
    ignis_seq_pool_free(wide_pool);
  }

  // A sequence that is not this pool's is a plain bad argument, not a bad
  // blob: the caller paired the wrong two handles.
  {
    ignis_seq_pool *other = nullptr;
    expect_rc(ignis_seq_pool_create(&spec, &other), 0, "refusals: second pool create");
    std::vector<unsigned char> elsewhere(blob.size());
    expect_rc(ignis_seq_snapshot(other, source, elsewhere.data(), elsewhere.size()), -1,
             "refusals: a sequence from another pool");
    ignis_seq_pool_free(other);
  }

  // Mid-chunk: one GQA layer's frontier ahead of the program's, which is
  // the state the per-layer entry point leaves behind between layers. The
  // size query refuses it too -- a caller cannot even price a snapshot it
  // would not be allowed to take.
  {
    std::vector<unsigned char> scratch(blob.size());
    std::uint64_t bytes = 0;

    source->gqa_positions[0] += 1;
    expect_rc(ignis_seq_snapshot_size(pool, source, &bytes), IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
             "refusals: sizing a sequence mid-chunk at a GQA layer");
    expect(bytes == 0, "refusals: a refused size query reports nothing");
    expect_rc(ignis_seq_snapshot(pool, source, scratch.data(), scratch.size()),
             IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
             "refusals: snapshotting a sequence mid-chunk at a GQA layer");
    source->gqa_positions[0] -= 1;

    // The same, one GDN layer in. Layers 0..2 are GDN, so this is the state
    // a chunk is in before it reaches its first GQA layer at all -- and a
    // GDN layer leaves no trace in the KV pages, so without its own counter
    // this would look like a boundary and capture GDN state running ahead of
    // the KV history.
    source->gdn_positions[0] += 1;
    expect_rc(ignis_seq_snapshot_size(pool, source, &bytes), IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
             "refusals: sizing a sequence mid-chunk at a GDN layer");
    expect_rc(ignis_seq_snapshot(pool, source, scratch.data(), scratch.size()),
             IGNIS_SEQ_ERR_NOT_AT_BOUNDARY,
             "refusals: snapshotting a sequence mid-chunk at a GDN layer");
    source->gdn_positions[0] -= 1;

    expect_rc(ignis_seq_snapshot_size(pool, source, &bytes), 0,
             "refusals: the same sequence at the boundary again");
  }

  ignis_seq_release(pool, target);
  ignis_seq_release(pool, source);
  ignis_seq_pool_free(pool);
}

// ---- 4. both KV formats ----------------------------------------------------

// A snapshot's size must be right under both storage formats: bytes per
// sequence-token differ by 7.11x, so a size computed from anything but the
// pool's own planes would be wrong for one of them (ADR 0022).
void check_both_formats() {
  const std::uint32_t context = 1024;
  std::uint64_t sizes[2]      = {0, 0};
  std::uint64_t page_bytes[2] = {0, 0};
  const int32_t formats[2]    = {IGNIS_KV_FORMAT_BF16, IGNIS_KV_FORMAT_HQ_E8_2B};

  for (int i = 0; i < 2; ++i) {
    const ignis_seq_pool_spec spec = qwen38_27b_spec(formats[i], context, 1);
    ignis_seq_pool *pool           = nullptr;
    expect_rc(ignis_seq_pool_create(&spec, &pool), 0, "formats: pool create");
    ignis_seq *seq = nullptr;
    expect_rc(ignis_seq_alloc(pool, context, &seq), 0, "formats: alloc");
    set_frontier(*seq, context);

    struct ignis_seq_pool_stats stats{};
    expect_rc(ignis_seq_pool_stats(pool, &stats), 0, "formats: pool stats");
    page_bytes[i] = stats.kv_page_bytes;
    expect_rc(ignis_seq_snapshot_size(pool, seq, &sizes[i]), 0, "formats: snapshot size");

    // The KV part of the size is exactly what this format's pages cost.
    const std::uint64_t pages = ninfer::pages_for_tokens(context);
    const std::vector<ignis_seq_section> sections = ignis_seq_section_table(*pool, pages);
    expect(sections[0].bytes == pages * stats.kv_page_bytes,
          "formats: the KV section is the pool's own page bytes");
    expect(stats.kv_bytes_per_token * context == sections[0].bytes,
          "formats: the KV section matches the reported per-token cost");

    ignis_seq_release(pool, seq);
    ignis_seq_pool_free(pool);
  }

  // Everything but KV is identical between the two, so the whole difference
  // in snapshot size is the KV image -- 7.11x denser under hq-e8-2b.
  const std::uint64_t pages     = ninfer::pages_for_tokens(context);
  const std::uint64_t kv_delta  = pages * (page_bytes[0] - page_bytes[1]);
  expect(sizes[0] > sizes[1], "formats: a BF16 snapshot is the larger one");
  expect(sizes[0] - sizes[1] == kv_delta,
        "formats: the size difference is exactly the KV image difference");
  expect(page_bytes[0] == 65536ULL * ninfer::kPagedKVPageSize,
        "formats: BF16 costs 65,536 bytes per sequence-token");
  expect(page_bytes[1] == 9216ULL * ninfer::kPagedKVPageSize,
        "formats: hq-e8-2b costs 9,216 bytes per sequence-token");

  std::printf("snapshot size at %u tokens (27B geometry): bf16 %llu B, hq-e8-2b %llu B\n", context,
              static_cast<unsigned long long>(sizes[0]),
              static_cast<unsigned long long>(sizes[1]));
}

// ---- 5. the measured transfer cost -----------------------------------------

// The tier's whole justification is that a restore is orders of magnitude
// cheaper than the re-prefill it replaces, and the ticket asks for that to
// be measured rather than assumed. Both directions are timed over pinned
// host memory -- the transport the host tier uses (GitHub #125) -- at a
// short and a full-context sequence of the real 27B geometry.
void report_transfer_cost(std::uint32_t context_tokens, const char *label) {
  const ignis_seq_pool_spec spec = qwen38_27b_spec(IGNIS_KV_FORMAT_HQ_E8_2B, context_tokens, 1);
  ignis_seq_pool *pool           = nullptr;
  if (ignis_seq_pool_create(&spec, &pool) != 0) {
    std::fprintf(stderr, "FAIL: cost (%s): pool create: %s\n", label, ignis_seq_last_error());
    ++failures;
    return;
  }
  ignis_seq *seq = nullptr;
  expect_rc(ignis_seq_alloc(pool, context_tokens, &seq), 0, "cost: alloc");
  set_frontier(*seq, context_tokens);

  std::uint64_t bytes = 0;
  expect_rc(ignis_seq_snapshot_size(pool, seq, &bytes), 0, "cost: snapshot size");

  void *pinned            = nullptr;
  const cudaError_t alloc = cudaMallocHost(&pinned, static_cast<std::size_t>(bytes));
  if (alloc != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cost (%s): cudaMallocHost(%llu): %s\n", label,
                 static_cast<unsigned long long>(bytes), cudaGetErrorString(alloc));
    ++failures;
    ignis_seq_release(pool, seq);
    ignis_seq_pool_free(pool);
    return;
  }

  // One untimed pass first: the first transfer over a fresh pinned region
  // pays page-table work that a steady-state eviction does not.
  expect_rc(ignis_seq_snapshot(pool, seq, pinned, bytes), 0, "cost: warmup snapshot");

  constexpr int kReps = 3;
  double snapshot_ms  = 0;
  double restore_ms   = 0;
  for (int rep = 0; rep < kReps; ++rep) {
    auto began = std::chrono::steady_clock::now();
    expect_rc(ignis_seq_snapshot(pool, seq, pinned, bytes), 0, "cost: snapshot");
    snapshot_ms += ms_since(began);

    began = std::chrono::steady_clock::now();
    expect_rc(ignis_seq_restore(pool, seq, pinned, bytes), 0, "cost: restore");
    restore_ms += ms_since(began);
  }
  snapshot_ms /= kReps;
  restore_ms /= kReps;

  const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  std::printf("transfer cost %s (%u tokens, hq-e8-2b): %.2f MiB, snapshot %.2f ms (%.1f GB/s), "
              "restore %.2f ms (%.1f GB/s)\n",
              label, context_tokens, mib, snapshot_ms,
              static_cast<double>(bytes) / (snapshot_ms * 1e6), restore_ms,
              static_cast<double>(bytes) / (restore_ms * 1e6));

  CUDA_CHECK(cudaFreeHost(pinned));
  ignis_seq_release(pool, seq);
  ignis_seq_pool_free(pool);
}

} // namespace

int main() {
  int count                   = 0;
  const cudaError_t count_err = cudaGetDeviceCount(&count);
  if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
    std::cout << "SKIP: no usable CUDA device\n";
    return 77;
  }
  if (count_err != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
    return 1;
  }

  check_section_table();
  // Twice: the small BF16 geometry, and the real 27B geometry under
  // hq-e8-2b, whose KV pages are four planes per layer of fixed-budget code
  // and metadata rows rather than two of raw values. A round trip that packs
  // and unpacks one correctly says nothing about the other.
  check_round_trip(small_spec(), 128, 100, "bf16, small geometry");
  check_round_trip(qwen38_27b_spec(IGNIS_KV_FORMAT_HQ_E8_2B, 256, 2), 256, 200,
                   "hq-e8-2b, 27B geometry");
  check_refusals();
  check_both_formats();
  // 128 tokens is the "short sequence" the spec prices at the snapshot's
  // floor (the GDN slot plus the conv taps and the penalty-count row);
  // 40,960 is the engine's own default context, where KV dominates.
  report_transfer_cost(128, "short");
  report_transfer_cost(40960, "full-context");

  if (failures != 0) {
    std::fprintf(stderr, "sequence snapshot test: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("sequence snapshot test: ok\n");
  return 0;
}
