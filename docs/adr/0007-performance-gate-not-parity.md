# ADR 0007 — The 99% gate is a performance gate, not a parity gate

## Status

Accepted (2026-09-02, grilling session). **Supersedes ADR 0003.**

## Context

ADR 0003 defined acceptance as "≥ 99% token agreement against the reference
(ninfer) baseline." The project owner clarified that this framing is wrong for
this project: ignis is *our* engine, not a recreation of ninfer, so acceptance
should **not** be "match ninfer's output." The 99% figure refers
*exclusively* to the **first performance gate**, not to token-agreement with
the reference.

## Decision

- **The 99% acceptance is a PERFORMANCE gate: "≥ 99% of the reference's
  speed" (throughput / latency), not "≥ 99% token agreement."**
- **Correctness is self-checked, not reference-matched:** the engine must
  produce *sane* output for the same model (greedy, fixed seed); we do *not*
  require it to 99%-match or byte-match the reference's output.
- **It is the FIRST of a ladder of performance gates.** For now, only one gate
  (the 99%) is defined; later gates (e.g., "beat the reference") are to be
  defined as needed.
- The reference (ninfer) is used **as a speed reference only**, and *only* in
  extreme necessity; borrowing pieces from other inference engines is also
  fine, as long as we stay on the north-star.

## Consequences

- "Parity" is no longer the acceptance word; **"≥ 99% of reference
  performance"** is. Token-level divergence reports are no longer the
  acceptance artifact; the **performance report** (tok-s, ttft vs reference)
  plus a **self-consistency check** (sane output) is.
- The `bench` harness's acceptance criterion changes from "token-agreement +
  divergence report" to "performance ≥ 99% of reference + per-class
  ttft/tok-s check + self-consistency."