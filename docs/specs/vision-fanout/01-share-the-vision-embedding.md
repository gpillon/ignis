# 01 - share the vision embedding across a fan-out

The device-resident vision embedding is encoded **once per request**. A
fan-out of N questions over one image encodes the same picture N times, at
4.17 s each on a 4096x4096 screenshot
(`docs/findings/2026-09-20-number-width-and-decide-e2e.md`).

The CPU half is already cached: `MediaCache` keys prepared patches by their
content digest, and every follower in a fan-out reports
`media.cache_hits: 1` with `media.preprocess_seconds: 0.000`. What has no
cache at any tier is the thing that costs the most - the embedding those
patches encode to. `RuntimeCompute.media` keys it by `RequestId` and drops it
in `prefill_multimodal_job` the moment the request's last placeholder is
prefilled, so nothing carries it to the sibling asking the next question
about the same bytes.

## What the measurement says

Per question, 4096x4096 screenshot, 16,384 vision tokens, from
`ignis.request.admitted`:

| stage | time | shared today? |
|---|---|---|
| media preprocess | 0.17 s first, 0.00 s after | yes, by content digest |
| **vision encode** | **4.17 s** | **no** |
| attention prefill, 16,506 tokens | ~2.4 s | no |

End to end, a `/v1/decide` fan-out over one image: 1 question 6.84 s, 2
questions 13.50 s, 4 questions 27.92 s - 1x, 2x, 4x, with the token counts
exactly proportional too.

## Why this is not the prefix machinery's job

GitHub #240's unmet acceptance 2 is about the *prompt*: an image `state`
stays in the user turn, so no retained prefix covers it and every question
re-prefills the image's placeholder run. Fixing that recovers the ~2.4 s row.

It would leave the 4.17 s exactly where it is. A shared prefix shares **KV
pages**, and the embedding is not KV - it is the encoder's output, consumed
by the prefill that scatters it into the placeholder columns. The two are
independent misses and this slice is the larger one.

## Shape

**Revised 2026-09-20, during implementation.** Two findings in the code
changed the shape below; what the first one said is kept at the end under
*What the first shape got wrong*, because the reasoning is the point.

- Key the live embedding by **`(content digest, grid)`**, which `MediaItem`
  already carries, rather than by `RequestId`. Not the whole `MediaKey`: that
  also carries the item's prompt offset, which two siblings do not share when
  the question precedes the image.
- Refcount it, and **let the entry outlive its last holder**. The overlap a
  refcount would exploit does not exist - `ignis_core::concrete` admits one
  request holding multi-tick prefill progress at a time, and the embedding is
  given up at the item's last placeholder, before the next sibling's first
  chunk runs. The lingering *is* the mechanism, not a budget option on top of
  one.
- Bound it with a **paged pool** in the leaf, not with a count of slots. An
  embedding is `merged_tokens x hidden x 2 B`, and at 5120 hidden a column is
  10,240 B: a 320x240 thumbnail is 0.78 MiB and a 4096x4096 screenshot is
  160 MiB. Envelope-wide slots would reserve 320 MiB to hold a thumbnail. One
  reservation carved into 128-column pages holds as many embeddings as their
  own columns fit.
- The leaf owns the bytes and answers `IGNIS_MEDIA_ENCODE_POOL_FULL`; the
  runtime owns the policy and releases the least recently unheld entry before
  asking again. The pool is floored at one envelope-wide item, which is what
  makes that loop terminate.
- The budget is the owner's call, taken 2026-09-20:
  **`--vision-embedding-pool-mib`, default one envelope-wide item** - exactly
  the reservation GitHub #177 always took, so a load that says nothing does
  not move the VRAM plan.

See ADR 0035 for the decision and for the alternatives that were rejected.

## Acceptance

1. Two questions over one image encode it **once**: the second question's
   `media.encode_seconds` is 0 and its answer is unchanged.
2. A fan-out of N questions over one image costs roughly one encode plus N
   prefills, not N of each - measured end to end, against the 1x/2x/4x
   baseline above.
3. **Revised.** The embedding is released when the pool needs its room, not
   when its last holder finishes - holding it past the last holder is the
   whole slice. What must still hold: `live_media()` (entries a request is
   prefilling against) returns to zero after a fan-out, resident bytes never
   exceed the plan's line, and a fan-out over a *different* image **replaces**
   rather than grows.
4. The VRAM plan carries the pool as its own line, and the load refuses
   rather than overcommits if the budget cannot hold it (ADR 0030).
5. A text-only load is untouched: no new reservation, no new lookup on the
   prefill path.

## Out of scope, filed separately

The pool makes it *possible* to hold N frames. Nothing encodes a frame before
a request asks for one - an encode happens only inside a prefill chunk - so
"at time N, N+x frames ready" needs a **warm-ahead trigger** (a call that
encodes without asking a question) and a **pin** against the eviction order.
That is new API surface and is its own ticket.

Nor does any of this make a *new* frame cheaper: an unseen picture costs
4.17 s of tower whatever the pool holds. That is spec 02's axis.

## What the first shape got wrong

The original shape said the entry "must **not** outlive the last holder by
default", and offered a cache size as the owner's budget knob. Both were
wrong for the same reason: they assumed a fan-out's questions overlap in
time. They do not, so a refcount that drops at zero holders recovers 0% of
the 4.17 s. The knob was mis-framed too - a count of envelope-wide slots is
the expensive way to hold a thumbnail, and bytes-with-pages is the cheap one.

## References

- Finding: `docs/findings/2026-09-20-number-width-and-decide-e2e.md`
  (the split, and the cache-hit evidence that the CPU half already works).
- ADR 0035 (this slice's decision), ADR 0030 (the VRAM plan), ADR 0024 (what
  a snapshot may hold).
- GitHub #240 (the prompt half of the same fan-out cost), #235.
- `crates/runtime/src/lib.rs::prefill_multimodal_job`,
  `crates/server/src/media.rs` (the digest-keyed cache this mirrors),
  `kernel/src/vision_encode.cu` and `kernel/src/step.cu` (the pool's writer
  and reader).
