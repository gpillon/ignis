# 04 - fan-out over one shared state

GitHub: #240

One `state` and N questions become **N internal requests** over a shared prefix.
Nothing new in the scheduler: the prefix reuse and batched prefill that exist
already do the work (option A of the grill; option B is GitHub #235).

- Each question becomes a request carrying the whole prompt, evidence first, so
  the shared `state` is a shared token prefix.
- **The first question is sequenced.** If all N are handed to the scheduler at
  once, none finds a published prefix and all N prefill the whole state - for an
  image that is N x 16K tokens instead of 16K. One question goes first,
  publishes the prefix, and the rest follow.
- The image is not re-encoded by the followers: `prefill_multimodal_job` encodes
  only when a chunk carries media columns, and a claimant standing past them
  carries none.
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
   times - asserted on the prefill token count, not on wall time.
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
