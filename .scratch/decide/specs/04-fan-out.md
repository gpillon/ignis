# 04 - fan-out over one shared state

GitHub: #240

One `state` and N questions become **N internal requests** over a shared prefix.
Nothing new in the scheduler: the prefix reuse and batched prefill that exist
already do the work (option A of the grill; option B is GitHub #235).

- Each question becomes a request carrying the whole prompt, with the evidence
  in the **system block**, so the shared `state` is a shared token prefix a
  sibling can actually claim.

  *(Corrected 2026-09-20, during #240. This line said "evidence first", which
  is true of the bytes and false of the engine: a decision is never a live
  publisher and captures no checkpoint, so its sibling can claim only a
  retained prefix, and a retained prefix is cut at the page floor of the
  system block and never reaches outside it. Evidence first in the user turn
  is shared and re-prefilled N times -
  `docs/findings/2026-09-20-the-evidence-belongs-in-the-system-block.md`.)*
- **The first question is sequenced.** If all N are handed to the scheduler at
  once, none finds a published prefix and all N prefill the whole state - for an
  image that is N x 16K tokens instead of 16K. One question goes first,
  publishes the prefix, and the rest follow.
- The image is not re-encoded by the followers: `prefill_multimodal_job` encodes
  only when a chunk carries media columns, and a claimant standing past them
  carries none.

  *(Not implemented, and acceptance 2 with it. An image `state` stays in the
  user turn, so there is no claim to stand past. The two things that argue
  for leaving it there - a media-in-system policy this route would be
  reversing alone, and `prefix_floor` excluding a media item the page floor
  lands inside - are in `decide::messages_for`; neither is a law, so this is
  a design fork rather than a closed question.)*
- **Group cancellation.** The fan-out is a unit of cancellation even though it
  is not a unit of scheduling: the handler's future dropping cancels every
  internal request still alive. Twenty independent requests is exactly the shape
  in which one forgets to cancel nineteen.
- Failure after the GPU is per-question: a lane that dies or times out puts an
  error in that question's place in `answers` and leaves the others' paid-for
  answers alone. (Validation failures are all-or-nothing and happen earlier -
  spec 03.)

## Acceptance

1. Twenty questions over one text `state` prefill the state once, not twenty
   times - asserted on the prefill token count, not on wall time. *(Met, with
   one caveat the spec did not anticipate: the retained prefix is cut at a
   **page floor**, so a `state` whose system block is under one page is
   shared in full on paper and not at all in practice - Jev's own documented
   one-sentence example among them.)*
2. Twenty questions over one image `state` encode the image once.
3. A client disconnecting mid-fan-out leaves no internal request running.
4. One question failing at runtime still returns the other nineteen answers,
   with an error in its own slot.
5. The fan-out's internal requests carry the parent's class.

## References

- ADR 0034, ADR 0029 (cross-request state reuse).
- GitHub #235 (the one-request, N-suffix alternative, and why the sequencing
  above is the argument in its favour).
- `crates/runtime/src/lib.rs::prefill_multimodal_job` (the encode condition).
