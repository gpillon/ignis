def patch(path, reps):
    s = open(path, encoding='utf-8').read()
    for old, new in reps:
        assert old in s, (path, old[:80])
        s = s.replace(old, new, 1)
    open(path, 'w', encoding='utf-8', newline='').write(s)


# ── minor: a running bound, so a drifted layout is refused before a byte
#    is enqueued rather than after every copy is ─────────────────────────
patch('kernel/include/ignis_seq_checkpoint_internal.h', [
    ("""  const bool capture      = direction == IGNIS_SEQ_PREFIX_CAPTURE;
  unsigned char *packed   = image;
  const std::size_t count = pool.kv_pool.plane_count();
  for (std::size_t index = 0; index < count; ++index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    const std::size_t bytes     = static_cast<std::size_t>(plane.nb[3]);""",
     """  const bool capture      = direction == IGNIS_SEQ_PREFIX_CAPTURE;
  unsigned char *packed   = image;
  const std::size_t count = pool.kv_pool.plane_count();
  // The bound is checked *before* each copy is enqueued, not after the loop:
  // an out-of-bounds `cudaMemcpyAsync` that has already been issued is not
  // something a later throw can take back. The sizing function and this loop
  // agree today by construction -- both read the PageMajor page stride off
  // each plane -- so this is here for the day one of them stops, and a
  // refused capture is a bet not taken where an overrun is someone else's
  // memory.
  const std::uint64_t budget = ignis_seq_checkpoint_page_bytes(pool);
  for (std::size_t index = 0; index < count; ++index) {
    const ninfer::Tensor &plane = pool.kv_pool.plane(index);
    const std::size_t bytes     = static_cast<std::size_t>(plane.nb[3]);
    if (static_cast<std::uint64_t>(packed - image) + bytes > budget) {
      throw std::logic_error("prompt checkpoint tail page: plane " + std::to_string(index) +
                             " would move past the " + std::to_string(budget) +
                             " bytes the pool prices a page at; the page layout and its "
                             "sizing have drifted apart");
    }"""),

    ("""    packed += bytes;
  }
  // The sizing function and this loop must agree about what a page costs, or
  // a capture writes past the buffer it was given. They agree today by
  // construction -- both read the PageMajor page stride off each plane -- so
  // this is here for the day one of them stops: a refused capture is a bet
  // not taken, an overrun is someone else's memory.
  const std::uint64_t moved    = static_cast<std::uint64_t>(packed - image);
  const std::uint64_t expected = ignis_seq_checkpoint_page_bytes(pool);
  if (moved != expected) {
    throw std::logic_error("prompt checkpoint tail page: moved " + std::to_string(moved) +
                           " bytes for a page the pool prices at " + std::to_string(expected) +
                           "; the page layout and its sizing have drifted apart");
  }
}""",
     """    packed += bytes;
  }
  // And the whole page has to have been moved, not only part of one: a plane
  // set that shrank would otherwise leave the rest of the image stale.
  const std::uint64_t moved = static_cast<std::uint64_t>(packed - image);
  if (moved != budget) {
    throw std::logic_error("prompt checkpoint tail page: moved " + std::to_string(moved) +
                           " bytes for a page the pool prices at " + std::to_string(budget) +
                           "; the page layout and its sizing have drifted apart");
  }
}"""),
])

# ── finding 2: the kernel test asserts the property instead of staging it ─
patch('kernel/tests/test_seq_checkpoint.cpp', [
    ("""  dirty_state(*pool, *seq, 1, 0x57u);
  // The penalty-count row is zero at the opener in a real request: nothing
  // has been sampled yet. Set explicitly here because `dirty_state` above
  // leaves the rest of the slot patterned.
  CUDA_CHECK(cudaMemset(pool->token_counts_for(seq->slot), 0,
                        static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t)));
  return seq;""",
     """  dirty_state(*pool, *seq, 1, 0x57u);
  // Note what is NOT done here: the penalty-count row is left exactly as the
  // sequence's own life left it. `ignis_seq_alloc` zeroed it and nothing has
  // sampled since, which is the state a real request is in at its generation
  // opener -- the scheduler only ever asks for a capture from an intermediate,
  // greedy chunk. Zeroing it here would stage the property the test asserts.
  return seq;"""),

    ("""  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "counts: publisher");

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "counts: capture");""",
     """  ignis_seq_prefix *prefix = nullptr;
  ignis_seq *publisher     = publisher_at_opener(pool, &prefix, "counts: publisher");

  // The property first, on the sequence itself and before anything is
  // captured: a request that has reached its opener has sampled nothing, so
  // its count row is untouched from allocation. Nothing in this test put it
  // there.
  const std::size_t counts_bytes =
      static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t);
  const std::vector<unsigned char> at_opener =
      read_device(pool->token_counts_for(publisher->slot), counts_bytes);
  expect(std::all_of(at_opener.begin(), at_opener.end(),
                     [](unsigned char b) { return b == 0; }),
         "counts: a sequence standing at its opener has sampled nothing");

  ignis_seq_checkpoint *checkpoint = nullptr;
  expect_rc(ignis_seq_checkpoint_capture(pool, publisher, kOpener, &checkpoint), 0,
            "counts: capture");"""),

    ("""  fill_device(pool->token_counts_for(publisher->slot),
              static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t), 0xB1u);

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0,
            "counts: claim");
  const std::vector<unsigned char> counts =
      read_device(pool->token_counts_for(claimant->slot),
                  static_cast<std::size_t>(pool->vocab) * sizeof(std::int32_t));""",
     """  fill_device(pool->token_counts_for(publisher->slot), counts_bytes, 0xB1u);

  ignis_seq *claimant = nullptr;
  expect_rc(ignis_seq_alloc_from_checkpoint(pool, kContext, checkpoint, &claimant), 0,
            "counts: claim");
  const std::vector<unsigned char> counts =
      read_device(pool->token_counts_for(claimant->slot), counts_bytes);"""),
])

# ── minors: comment width, the zero-token claim, departure 6's issue number ─
patch('crates/runtime/src/cuda_leaf.rs', [
    ("""    // Read only for `RuntimeStats::free_vram_bytes` (GitHub #186), never
    // to size the KV pool — see the module doc; otherwise held purely so
    // the device context outlives `artifact` and every loaded model, since dropping it would invalidate their device
    // memory.""",
     """    // Read only for `RuntimeStats::free_vram_bytes` (GitHub #186), never to
    // size the KV pool — see the module doc; otherwise held purely so the
    // device context outlives `artifact` and every loaded model, since
    // dropping it would invalidate their device memory."""),
])

patch('crates/runtime/src/lib.rs', [
    ("""            // A full-prompt match carries no tail: the claim already put the
            // sequence where its prompt ends, with the pending token the
            // publisher computed, so there is nothing left to warm.""",
     """            // A full-prompt match carries no tail: the claim already put the
            // sequence where its prompt ends, with the pending token the
            // entry carried, so there is nothing left to warm. True of a
            // shared prefix, whose image is the publisher's state at the
            // prefix's end, and of a prompt checkpoint (GitHub #186), whose
            // progress section carries the pending token the capturing
            // sequence had at its opener."""),
])

patch('crates/server/src/api.rs', [
    ("""    // prefix's identity is token ids alone until #180's media-aware match key
    // lands, so two prompts differing only in their images would share a
    // checkpoint. The opener is therefore reported on the text-only path only.""",
     """    // prefix's identity is token ids alone until the media-aware match key
    // lands (#189 defines the key, #193 fills its media slot), so two prompts
    // differing only in their images would share a checkpoint. The opener is
    // therefore reported on the text-only path only."""),
])

patch('crates/core/src/checkpoint.rs', [
    ("""//! #189 (identity)
//! replaces [`CheckpointEntry::tokens`] with a hash chain over the token ids
//! plus media identity, and adds the compatibility identity a blob is refused
//! on. None of those need the shape here to change.""",
     """//! #189 (identity)
//! replaces [`CheckpointEntry::tokens`] with a hash chain over the token ids
//! and adds the compatibility identity a blob is refused on, with #193
//! filling the media slot in that key. None of those need the shape here to
//! change."""),
])
print('ok')
