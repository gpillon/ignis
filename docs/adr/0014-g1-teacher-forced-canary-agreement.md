# ADR 0014 — G1's cross-engine canary floor is teacher-forced next-token agreement

## Status

Accepted (2026-09-07, GitHub #76). **Clarifies ADR 0007** and corrects the G1
acceptance wording in `.scratch/runtime/specs/01-device-resident-forward.md`,
which had drifted into a parity requirement ADR 0007 explicitly rejects.

Sources: GitHub #72 (root cause), GitHub #76 (this decision), the measurement
in `crates/server/tests/oracle_teacher_forced_gpu.rs`.

## Context

G1's spec asked for "≥ 95% first-32-token agreement with the reference." As
implemented, that was scored **free-running**: ignis generated its own 32-token
continuation of each canary prompt and the comparer diffed that token stream
against the reference's recorded continuation, position by position.

Free-running comparison has a structural flaw as a correctness floor. The
candidate feeds *its own* emitted token back in, so the two engines stop
sharing a prefix the moment they differ once. One divergence at position 0
decorrelates every later position, and the metric collapses from "is the
forward pass right?" to "did the whole continuation stay identical?"

That is exactly what happened. GitHub #72 established that `rust-sort` and
`explain-reverse` diverge at position 0 on an **exact BF16 logit tie** (19.5
vs 19.5, and 22.875 vs 22.875) between ignis's pick and the oracle's. Both
continuations are correct, fluent answers. The suite scored 52% — not because
the forward pass is broken, but because two coin flips cascaded.

Two further facts rule out the obvious "fix":

- ignis's argmax **is** the reference's argmax. `kernel/vendor/src/ops/kernel/argmax.cuh`
  is vendored verbatim under the manifest (ADR 0010) and diffs identical
  against the reference. Its tie-break is lowest token id.
- The reference's own greedy sampler documents itself as "bit-identical to
  argmax()", with the same lowest-id rule, on both its plain and speculative
  decode paths.

So there is no tie-break convention to adopt: we already have the reference's.
Where the two engines' logits differ at all, they differ in the last bits of a
BF16 output head, which is expected and permitted.

**What G1 is actually for.** G1 is a *sanity floor*. It exists to catch gross
implementation errors — wrong tensor layouts, missing ops, broken state
wiring, wrong RoPE positions, corrupted activations, an output head wired
backwards. It is **not** a proof of numerical or token-level equivalence with
the reference, and the reference's generated continuation is **not** the
definition of correctness. ADR 0007 already said this ("correctness is
self-checked, not reference-matched"); the G1 spec's wording silently
contradicted it.

## Decision

- **The G1 cross-engine canary metric is teacher-forced next-token
  agreement.** For each oracle canary position, ignis is fed the *oracle's
  own* token prefix for that position, and its greedy next-token argmax is
  compared with the recorded oracle token. Every position is scored
  independently, so a single divergence cannot cascade.
- **The floor is unchanged: overall teacher-forced agreement ≥ 95%**,
  aggregated over total matched positions across the suite (not a per-canary
  average).
- **No special-casing of ties.** A mismatch is a mismatch, including the two
  known BF16 exact-tie positions. No tie waiver, no epsilon comparison, no
  denominator adjustment, no canary removal, no threshold reduction.
- **ignis's argmax does not change.** It is the vendored reference kernel and
  already matches the reference's lowest-token-id tie-break.
- **Free-running comparison stays as a diagnostic**, reported by
  `ignis-bench oracle compare` and clearly labelled informational. It is **no
  longer a hard G1 gate** — it measures continuation similarity, not whether
  the forward pass is grossly broken.
- **This does not replace the numeric checks.** The vendored op tests, the f64
  layer/program references, deterministic execution across loads, and the GPU
  profile remain separate hard checks. Teacher-forced agreement is an
  additional high-level sanity check layered on top of them, and is not a
  claim of numerical equivalence.

## Empirical basis (2026-09-07, free RTX 5090, real artifact)

Same fixture (`crates/bench/tests/fixtures/oracle_canary.json`), same prompts,
thinking disabled to match how the oracle was recorded:

| Metric | Agree / compared | Result |
|---|---|---|
| Free-running (the old gate) | 53 / 102 | 52% |
| Teacher-forced next token | 99 / 102 | **97.1%** |
| The floor | 97 / 102 | 95% |

Per canary, teacher-forced: `rust-hello` 21/21, `math-greedy` 32/32,
`rust-sort` 30/32, `explain-reverse` 16/17. Two consecutive runs were
bit-identical.

Three mismatches remain, and none is waived:

- `rust-sort` position 0 and `explain-reverse` position 0 — the two
  already-diagnosed exact BF16 ties (GitHub #72). They are counted as
  mismatches and the suite still clears the floor.
- `rust-sort` position 23 — a genuine disagreement: ignis picks token 198 at
  logit 20.25 where the oracle's token 25 sits at 19.25 on ignis's own logits.
  Roughly eight units in the last place at BF16. Worth a diagnostic follow-up,
  **not** a G1 blocker under the accepted floor.

## Consequences

- The G1 spec's user story 5 and its testing decisions, `ROADMAP.md`'s G1 row,
  `CONTEXT.md`'s "Canary oracle" entry and the PENDING/REVIEW references are
  reworded to the teacher-forced definition. ADR 0007 is **not** weakened: this
  ADR moves G1 *toward* it, not away.
- The executable gate is `crates/server/tests/oracle_teacher_forced_gpu.rs`,
  under the GPU profile (ADR 0006). Its scoring math lives in
  `ignis_bench::oracle` (`score_teacher_forced`, `overall_teacher_forced_agreement`,
  `G1_AGREEMENT_FLOOR`) so the arithmetic is CPU-unit-tested without a GPU.
- `ignis-bench oracle compare` no longer fails the process below 95%. It
  prints the free-running figure as a diagnostic.
- Teacher forcing needs the next-token result the engine already computes:
  `prefill_program`'s existing `out_logits` (GitHub #72) plus the step ABI's
  span + start-position contract are sufficient. **No production-path change
  was required.**
- A future engine change that breaks layouts, positions or state wiring will
  drop teacher-forced agreement sharply, which is the signal G1 wants. A
  change that merely shifts the last bits of the output head will not.
