# ADR 0022 — two KV formats, with BF16 retained as the correctness oracle

## Status

Accepted (2026-09-11) — GitHub #65, phase 4. Spec:
`.scratch/runtime/specs/04-reference-feature-floor.md`.

## Context

ignis has carried BF16 KV since G1 because the review chose it as the first KV
dtype and put hq-e8-2b at G4 (§7 decision 5). The reference has run hq-e8-2b in
the owner's production profile the whole time, so every gate verdict to date
records the format difference as a known inequality rather than measuring
through it (ADR 0015; `crates/bench/src/g2.rs:416`).

The capacity argument for hq is decisive at this geometry. A sequence-token
costs 65,536 bytes of BF16 KV against 9,216 bytes of hq-e8-2b, a factor of
7.11, so eight lanes at a 40,960-token context need 20 GiB under BF16 and about
3.02 GB under hq. `N-lane concurrency` in `CONTEXT.md` is a short-context
promise until that changes.

The complication is not the codec, which is vendored and whose format keeps the
paged-KV contract's fixed-bytes-per-token property. It is that every
correctness oracle this project owns was calibrated on BF16: the teacher-forced
canary floor (ADR 0014), the f64 layer references, and the chunked-versus-
per-token self-oracle. Replacing the KV format outright would re-derive all of
those tolerances under a lossy format in the same change that introduces the
lossy format — and #96 is this project's own record of what a tolerance nobody
re-derived can hide, in that case roughly 80x of unused slack carried over from
an unrelated oracle.

An external oracle was considered for hq: recording the reference's hq-mode
output and checking ignis against it. That is token-agreement, which ADR 0007
exists to refuse, and it would make a second engine's quantization choices the
definition of ignis being correct.

## Decision

Both KV formats ship. The format is a **model-load option**, fixed for the life
of a load.

- **hq-e8-2b is the serving default**, and the format both engines run in for
  the G4 gate. *In force since P4-05 (GitHub #123).* P4-04 (GitHub #122)
  shipped the format as a load option with `bf16` as the CLI default, because
  hq could not serve a token until its attention routes landed — an hq forward
  pass was refused by name until then. #123 wired the prefill and decode
  routes and their decode graphs and flipped the default, which is what made
  the sentence above true rather than planned.
- **BF16 is retained**, and is the format every correctness oracle runs
  against. The GPU profile keeps its correctness checks on BF16.
- **hq earns its own acceptance in-house**: op-level codec error measured on
  real KV rows, and the hq attention route checked against the BF16 route on
  identical keys and values. The tolerance is derived from the measured codec
  error and never copied from another oracle.
- The KV pool is sized by a byte budget; token capacity is **derived** from the
  format in force and reported at load. No token count is compiled in.

## Consequences

- Two attention route families are live, and both are in the GPU profile. The
  cost is real and is accepted in exchange for keeping a non-lossy reference
  path inside the engine.
- A format change is a restart, not a runtime switch. One decode-graph capture
  set per process; widths 1..8 unchanged (ADR 0019 / 0020).
- The recorded KV inequality beside the G2 and G3 verdicts is retired only by a
  live/live run with hq on both sides, which is a G4 gate cell.
- Anything that later depends on the exact bytes of the KV cache — an exact-key
  side store, the host tier's snapshot layout, a drafter's KV carry at G5 — must
  be written against the section table (ADR 0024), not against one format.
- Keeping BF16 means the engine can always answer "is this hq's fault?" by
  reloading. That diagnostic is the main reason retention beats replacement.
