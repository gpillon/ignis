# 02 - the decision request in the scheduler

GitHub: #238

A request kind that **ends where prefill ends**. It never takes a decode lane,
never enters `Running`, generates nothing, and finishes with its readout.

- Terminates at `prefill_complete`; its finish event carries the readout.
- `Compute::release` on finish, like any other completed request.
- Context reservation is the prompt alone - `api.rs`'s check reserves prompt +
  `max_tokens`, which is wrong here.
- **The empty-last-chunk trap.** `runtime/src/lib.rs`'s prefill loop runs the
  model only `if !job.tokens.is_empty()`. An exact repeat of a decision claims a
  checkpoint at its opener, leaves no tokens to prefill, and produces no forward
  pass and therefore no logits. A readout job's final chunk must carry at least
  one token: trim any prefix or checkpoint claim to `len - 1`.
- Class: inherited from the request, defaulting to `Agent` rather than
  `Interactive` (`CONTEXT.md`, **lane tag**, and its stated reasons).

## Acceptance

1. A decision request never appears in a decode round - asserted against a
   `Compute` that fails the test if `decode_step` is called for it.
2. Its finish event carries the readout, and `Compute::release` is called
   exactly once.
3. Two identical decision requests in a row: the second still performs a forward
   pass. A test that replays the same prompt and asserts a non-empty final chunk
   is what guards the trap above.
4. A decision whose prompt alone exceeds the context is refused with the context
   error; a decision whose prompt fits is admitted regardless of any
   `max_tokens`.
5. With no class on the request, the admitted class is `Agent`.

## References

- ADR 0034, ADR 0029 (retained state, whose claim the trim interacts with).
- `crates/core/src/scheduler.rs` (`Compute`, `PrefillJob`, `PrefillOutcome`),
  `crates/runtime/src/lib.rs` (`prefill_step`).
- Spec 01 (the seam this rides on).
