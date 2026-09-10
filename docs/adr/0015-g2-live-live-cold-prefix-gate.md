# ADR 0015 — G2 is judged live/live against the reference, on cold-prefix samples

## Status

Accepted (2026-09-07, grilling session for GitHub #63). **Clarifies ADR 0007**
(performance gates, not parity) and sets the measurement method the later
performance gates (G3, G4, G5) are expected to follow.

**Amended (2026-09-10) by ADR 0021** (GitHub #116): a live/live comparison
must pool at least two independent process launches per engine within the
session, not rely on within-launch repetition alone — #116 found that one
process launch's throughput can sit consistently 5-17% away from another
launch of the exact same binary, on the exact same GPU, moments apart. Every
decision below still holds; ADR 0021 adds the launch-pooling requirement on
top of it.

Sources: `.scratch/runtime/specs/02-real-prefill.md` (GitHub #63, the spec
this ADR serves), `.scratch/REVIEW-2026-09-05.md` §6 (Phase 2), the
reference's published context-length tables.

## Context

G2's gate is "TTFT at 8K / 32K prompts ≤ 1.5× the reference's MTP0". Turning
that sentence into a verdict needs three decisions that the roadmap left open.

**Where the reference's number comes from.** The reference publishes a
context-length table (prefill tok/s, server TTFT, decode tok/s at 7,680 /
64,512 / 130,048 / 260,096 prompt tokens). Reading the gate off that table is
cheap and reproducible, and wrong for this project:

- there is no 32K cell in it, so half the gate would be interpolated;
- the numbers were measured on a different day, driver, thermal state and
  build than the run being judged;
- the two engines would be measured by two different instruments — their
  harness and ours — with different definitions of when the clock starts.

The alternative is to measure both engines in the same session with the same
harness. It costs GPU-exclusive time (ADR 0006: the reference is the owner's
own coding-agent backend and must be stopped for ignis to run, then
restarted), and it produces a number that is only valid for that session. It
is also the only way to answer the question the gate is actually asking:
*today, on this machine, is ignis's prefill within striking distance of the
engine the owner otherwise runs?*

**What the reference is configured as.** The owner's production profile is
hq-e8-2b KV, 1,024-token prefill chunk, CUDA graphs, prefix reuse enabled.
ignis at G2 has BF16 KV and no graphs. Measuring the reference in a
handicapped configuration ("same KV format") would compare a hypothetical
against a hypothetical; measuring it as the owner runs it compares against
the bar the owner actually experiences. Performance-first (ADR 0005) points
at the second.

**What a TTFT sample must not measure.** The reference's production profile
has prefix reuse on, plus a host KV tier. Repeating the *same* prompt across
samples turns samples 2..N into cache hits: the measured quantity stops being
prefill and becomes restore. Prefix-cache performance is a real property of
an engine and a legitimate thing to measure — it is what G4 is about — but it
is not what a *real prefill* gate is asking, and it would silently flatter
whichever engine has it. Disabling prefix reuse for the run was considered;
it perturbs the production profile this ADR just chose to keep, and it does
not cover the other caches (host KV tier, drafter checkpoints).

## Decision

- **G2 is judged live/live.** The verdict is computed from two records
  produced in the **same measurement session, on the same machine, by the
  same harness** (`ignis-bench`): one driving ignis, one driving the
  reference. The gate check refuses to produce a verdict from records that do
  not share a session identifier.
- **The threshold is a ratio: ignis's median TTFT ÷ the reference's median
  TTFT ≤ 1.5, applied per cell** (8K and 32K post-template prompt tokens).
  Both cells must pass.
- **The reference is measured in the owner's production profile**, hq-e8-2b
  KV included. The KV-format difference is **recorded next to the verdict**
  as a known inequality, not corrected for, not used to adjust the ratio.
- **Every sample is a cold prefix.** Each sample — the warmup included — uses
  its own deterministically generated prompt, distinct from the **first
  content token**, at the same exact post-template token count. A shared
  template header may precede the divergence point only insofar as the engine
  cannot form a reusable prefix from it.
- **Cold prefill is verified, not assumed.** Each sample reads back the
  engine's own computed-prefill-token count and asserts it equals the prompt
  length. A sample that falls short is **void** and fails its cell.
- **The cell statistic is the median of five samples after one warmup**, with
  a small output budget, greedy, thinking disabled, streaming (TTFT is the
  arrival of the first content delta).
- **A committed reference record is a fixture for regression and sanity
  only.** It detects "the reference moved" and "our harness changed"; it is
  never the live side of a gate, and the gate check refuses to treat it as
  one.

## Consequences

- Running G2 costs a GPU-exclusive window with the reference stopped, started
  and stopped again (ADR 0006's runbook), because both engines must be
  measured back to back. That cost is accepted.
- A G2 verdict is only valid for its session. Re-running the gate after a
  change means re-measuring both sides, not comparing against the last
  recorded reference number.
- `ignis-bench` grows a `ttft` subcommand (cells, samples, cold-prefix prompt
  generation, computed-prefill verification) and a G2 gate check over two
  records. Both are ordinary bench code with CPU tests; only the run itself
  needs the GPU.
- The threshold is a floor, not an ambition. It says "not meaningfully worse
  than the reference at prefill"; the project's target remains at least as
  fast as the reference, and G3/G4 tighten to 99%.
- Later gates should follow this method (live/live, same harness, verified
  cold conditions, ratio against a stated profile) unless a specific gate has
  a reason not to. Where a gate measures a *cache* feature rather than a
  compute path, the cold-prefix rule is the thing to revisit — the rest of the
  method still applies.
- The reference's published tables remain useful as a sanity check (an ignis
  or reference number wildly off them means the harness is wrong), and as the
  only source for context lengths nobody has measured live.
