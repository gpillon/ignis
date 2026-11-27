---
status: accepted
---

# Performance-first as the organizing principle

ignis is our engine, a new architecture, not a recreation of ninfer: framed as a port that matched the reference,
it would have had the reference as its ceiling and its deliverable. Performance is the first objective above a
correctness floor that is never traded away, the self-check of ADR 0007 (sane output for the same model, greedy at a
fixed seed). Above the floor, performance is the tie-breaker for every scope, feature and kernel decision, within
each use of the north-star (ADR 0083), and it is never dropped to simplify.

## Consequences

- The architecture is ours, not a port: the Rust core (scheduler, KV, admission, API) is new, not a line-for-line
  port of the reference's.
- Performance trade-offs are decided by the north-star (ADR 0083), not by matching the reference.
- The ported CUDA kernels, proven and low-risk, are a temporary starting point: ignis re-implements them, guided by
  the north-star. ignis is functional and its own kernels wait on no dogfooding milestone: they come one measured
  family at a time, under ADRs 0010, 0031 and 0051.
- The reference is for inspiration only: it is consulted in extreme necessity, to solve a problem someone has
  already solved. Borrowing pieces from other inference engines is fine as long as the north-star is kept.
- Every optimization and every re-implementation is gated again. The gate is ADR 0082's, where the 99% performance
  gate against the reference once stood; the image model has no performance gate, its speed target taken from the
  literature (ADR 0050).
