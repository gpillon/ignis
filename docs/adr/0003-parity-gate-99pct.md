# ADR 0003 — 99% token-agreement parity gate

## Status

Superseded by ADR 0007 (2026-09-02): the 99% is a **performance** gate (≥ 99%
of reference speed), not token-agreement. Correctness is self-checked (sane
output, same model, greedy, fixed seed), not reference-matched.

## Context

v1 acceptance is "parity with the reference engine" (prefill/decode tok/s within
10%, ttft/tok-s within tolerance on trace-replay load). Byte-identical greedy
output across different builds is not guaranteed: CUDA toolkit version drift,
kernel porting subtleties, and GEMM accumulation-order changes can flip argmax
near ties. A hard byte-identity gate would make acceptance brittle and would
conflate "architecture works" with "every floating-point op is bit-identical".

Options:
- (a) byte-identical greedy output on the canary suite (divergence = bug, always)
- (b) ≥ 99% token agreement against the reference baseline, with a documented
  divergence report

## Decision

(b), with teeth: agreement is measured greedy + fixed seed on the canary suite and
the trace-replay load; every accepted run ships a divergence report (which prompt,
which token, which stream). Divergences are investigated as bug signals — the gate
tolerates them statistically, it does not bless them. If a divergence pattern
repeats across canaries, it blocks acceptance regardless of the aggregate rate.

## Consequences

- Acceptance is reproducible and not flaky on version drift.
- The divergence report is part of the bench harness output from v1.
- "Parity" in docs and telemetry always means "≥ 99% agreement + per-class
  performance within tolerance", never byte identity.