# ADR 0005 — Performance-first as the organizing principle

## Status

Accepted (2026-09-02, grilling session).

## Context

v1 could have been framed as "port the reference stack and match its parity,"
which would make the reference (ninfer) the ceiling and the deliverable. The
project owner stated explicitly: **ignis is OUR engine (a new architecture), not
a recreation of ninfer.** The reference is a *reference for inspiration only* —
consult it only when stuck on a problem someone has already solved. Performance
is the #1 objective.

## Decision

- **Performance is the #1 objective above a non-negotiable correctness floor.**
  The floor (self-check: the engine produces *sane* output for the same model,
  greedy, fixed seed) is never traded away; above it, performance is the
  tie-breaker for every scope, feature, and kernel decision.
- **The 99% acceptance is a PERFORMANCE gate (≥ 99% of the reference's
  speed), not a token-agreement gate.** See ADR 0007.
- **Kernel policy: port the proven CUDA "for now," re-implement later.** v1
  uses the proven ported CUDA kernels as a *temporary* starting point (low
  risk, proven). We **re-implement** them — guided by the north-star — in a
  later milestone (after ignis is already functional and dogfooding). The
  reference is consulted **only in extreme necessity** (to solve a problem
  someone already solved), and borrowing pieces from other inference engines is
  fine as long as we stay on the north-star.
- **Every optimization / re-implementation is re-gated by the 99% performance
  gate** (and the per-class performance checks) — we never drop performance to
  "simplify."

## Consequences

- The architecture is OURS, not a port: the Rust core (scheduler, KV,
  admission, API) is new, not a line-for-line port of the reference's
  architecture.
- The original kernel-port rule "no hand rewrites of proven kernels" is revised
  to **"port proven for now; re-implement later, guided by the north-star."**
- Performance trade-offs are decided by the **north-star** ("the best local
  coding engine: max performance + agent parallelism saturating the GPU in
  prefill and decode"), not by "matching the reference."