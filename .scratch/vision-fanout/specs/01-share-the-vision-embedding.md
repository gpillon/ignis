# 01 - share the vision embedding across a fan-out

The device-resident vision embedding is encoded **once per request**. A
fan-out of N questions over one image encodes the same picture N times, at
4.17 s each on a 4096x4096 screenshot
(`docs/findings/2026-09-20-number-width-and-decide-e2e.md`).

The CPU half is already cached: `MediaCache` keys prepared patches by their
content digest, and every follower in a fan-out reports
`media.cache_hits: 1` with `media.preprocess_seconds: 0.000`. What has no
cache at any tier is the thing that costs the most — the embedding those
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
questions 13.50 s, 4 questions 27.92 s — 1x, 2x, 4x, with the token counts
exactly proportional too.

## Why this is not the prefix machinery's job

GitHub #240's unmet acceptance 2 is about the *prompt*: an image `state`
stays in the user turn, so no retained prefix covers it and every question
re-prefills the image's placeholder run. Fixing that recovers the ~2.4 s row.

It would leave the 4.17 s exactly where it is. A shared prefix shares **KV
pages**, and the embedding is not KV — it is the encoder's output, consumed
by the prefill that scatters it into the placeholder columns. The two are
independent misses and this slice is the larger one.

## Shape

- Key the live embedding by the item's **content digest**, which
  `MediaItem` already carries, rather than by `RequestId`.
- Refcount it: a fan-out's questions overlap in time, and the entry must
  outlive whichever of them finishes first. It must **not** outlive the last
  holder by default — see the budget below.
- Bound it. One embedding is `merged_tokens x hidden x 2 B`: 160 MiB for a
  4096x4096 image at the default vision budget, 320 MiB at the
  `--vision-max-tokens` ceiling. ADR 0030 says VRAM is planned, not
  discovered, so this needs a line in the plan and a stated eviction rule,
  not a `HashMap` that grows.
- The **budget is the owner's call**, and it is the one thing this slice
  cannot decide for itself: a cache that holds one item is enough for a
  fan-out over one picture and useless for two concurrent callers; a cache
  that holds four costs 640 MiB of a 28.5 GB budget.

## Acceptance

1. Two questions over one image encode it **once**: the second question's
   `media.encode_seconds` is 0 and its answer is unchanged.
2. A fan-out of N questions over one image costs roughly one encode plus N
   prefills, not N of each — measured end to end, against the 1x/2x/4x
   baseline above.
3. The embedding is released when its last holder finishes: live media count
   returns to zero after a fan-out, and a second fan-out over a *different*
   image does not grow it without bound.
4. The VRAM plan carries the cache's reservation as its own line, and the
   load refuses rather than overcommits if the budget cannot hold it
   (ADR 0030).
5. A text-only load is untouched: no new reservation, no new lookup on the
   prefill path.

## References

- Finding: `docs/findings/2026-09-20-number-width-and-decide-e2e.md`
  (the split, and the cache-hit evidence that the CPU half already works).
- ADR 0030 (the VRAM plan), ADR 0024 (what a snapshot may hold).
- GitHub #240 (the prompt half of the same fan-out cost), #235.
- `crates/runtime/src/lib.rs::prefill_multimodal_job`,
  `crates/server/src/media.rs` (the digest-keyed cache this mirrors).
