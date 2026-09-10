# ADR 0021 — a live/live gate pools at least two independent process launches per engine

## Status

Accepted (2026-09-10) — GitHub #116. Amends ADR 0015: its live/live method
stands: same session, same harness, cold samples, ratio against a stated
profile. This adds one more requirement the method needs to actually deliver
what it promises — a ratio decided by the engines, not by which process each
one happened to be that session.

## Context

Investigating #111's C=1 acceptance criterion turned up a confound in the G3
instrument itself: `ignis-bench g3` samples agree to better than 1.5% against
one `ignis-server` process, but two *separate* process launches of the exact
same binary, same tree, same fixture, disagree by up to 17% on C=1 and ITL
p95 (#116). Repeating the finding under controlled conditions on a free 5090
confirmed it and narrowed down what it is not:

- **Not the code.** Both the pre-#111 and post-#111 trees produced both a
  fast-launch and a slow-launch reading.
- **Not slow drift over a launch's lifetime.** One launch was sampled four
  times over 20+ minutes, including a 3-minute idle gap: 72.9, 68.7, 72.6,
  73.4 tok/s — one low outlier, no trend. A second launch, sampled three
  times immediately after the first was stopped, held tight at 68.8, 69.5,
  69.8 — a different, narrower band from the first, reached instantly rather
  than converged into.
- **Not thermal or clock carryover from the previous process.** If the
  second launch's band came from residual heat or boost state left over by
  the one just stopped, its first sample would resemble the first launch's
  level. It did not — it landed in its own band immediately.
- **Not the GPU's sustained clock.** Restricting to samples where
  `nvidia-smi` reported `utilization.gpu > 30%` — genuine compute activity,
  not idle-between-requests polling — the two launches drove the SM clock to
  the *same* place: 2,823 vs 2,824 MHz average, 2,910 MHz peak, both. No
  `clocks_event_reasons` throttle bit was ever set on either.

That last point is the one that matters for where to look next. If busy-time
GPU clock is identical between a "fast" and a "slow" launch, the GPU is not
running any faster or slower during the work it does — whatever decides a
launch's throughput band is deciding it somewhere else, most plausibly the
host-side per-token path (kernel-launch dispatch, `cudaStreamSynchronize`
wait behavior, OS thread scheduling, the async runtime's wakeup latency) —
each only a fraction of a millisecond, but a decode token is already only
about 13–15 ms wide at these rates, so a fraction of a millisecond of added
per-token host overhead is a few percent of throughput, and it only has to
happen consistently for one launch's lifetime to produce a stable, distinct
band. This project has no tooling to trace that path further (Nsight Systems
correlating host and device timelines is the right instrument and is not
part of this repo's toolchain), so the exact mechanism stays a well-supported
hypothesis, not a proof.

What is provable, and load-bearing for the gate, is simpler: **the unit that
picks a throughput band is one process launch**, and a live/live session
launches each engine exactly once. ADR 0015's within-launch repetition (five
samples, one warmup) cancels request-to-request noise; it cannot cancel a
bias that is constant for the whole launch. A gate ratio computed from one
launch per side is therefore, some fraction of the time, actually comparing
"today's ignis-server launch" against "today's reference launch" rather than
the two engines — and #116's 17% spread is larger than the tolerance on every
G3 cell.

## Decision

A live/live gate measures **at least two independent process launches per
engine**, within the one GPU-exclusive session ADR 0015 already requires, and
computes the verdict from the pooled result rather than from a single launch
pair.

- "Independent launch" means the server process is stopped and started again
  — not a second `ignis-bench` invocation against the same running process.
  Model load plus decode-graph capture costs seconds, not minutes, so this is
  cheap next to the session's own cost.
- The simplest pooling rule that uses what this investigation actually
  established: report every launch's cells, and require the *cell statistic
  taken across all launches* (not the best one, not one arbitrarily chosen
  launch) to meet the threshold. Two launches per engine is the floor — it
  catches "one side got an unlucky launch," which is the failure mode #116
  demonstrated. Within-launch spread is not a fixed number to check against —
  #116 measured one launch at 1.4% and another, on the same tree, at 6.4% —
  so a cell whose across-launch spread looks larger than its own launches'
  internal spread is itself a finding worth recording, not silently averaged
  away.
- This is a session-cost, not a code, requirement: no `ignis-bench` interface
  changes are forced by this ADR. Running the existing `g3` /  `ttft`
  subcommands twice per engine and passing both pairs through the existing
  gate check already satisfies it; an operator convenience for aggregating
  more than one launch's records is left to whoever finds the manual repeats
  tedious enough to file it.

## Consequences

- Every live/live gate run costs roughly twice what it did per engine
  (two launches instead of one), on top of ADR 0015's existing GPU-exclusive
  window. Model load being cheap keeps this proportionate.
- A gate run that reports only one launch pair per engine is not a valid
  reading of this method from here on — the same way ADR 0015 already
  refuses a verdict computed from records that do not share a session id.
- `.scratch/runtime/specs/03-serving-loop.md`'s G3 procedure states the
  two-launch requirement for its three cells.
- The mechanism behind the per-launch band is not resolved by this ADR and
  is not ignis's own defect: both trees produced both bands, and the GPU's
  own sustained clock is identical across bands. Nailing the host-side cause
  further needs instrumentation this project does not have; the protocol
  change here is the mitigation that does not depend on finding it.
