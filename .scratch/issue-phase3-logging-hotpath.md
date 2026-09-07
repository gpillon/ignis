## Problem Statement

Once diagnostic logging is flowing through `crates/logging` (Phases 1-2), the biggest remaining risk is that logging itself degrades inference latency/throughput — either through per-token/per-layer emission, unbounded queue growth, or blocking I/O on the prefill/decode path. Ignis's whole reason to exist is performance (ADR 0005, ADR 0007's G4 gate); a logging system that regresses it, even slightly, is a regression the project cannot accept silently.

## Solution

Establish and verify, with the existing G4 performance gate, that normal operation never logs at INFO-or-above per token/layer/kernel/allocation on the prefill/decode/CUDA-graph path, and that the logging system's I/O is decoupled from that path through a bounded, two-priority queue that can never grow without limit and never silently drops an ERROR/WARN event under normal operation.

## User Stories

1. As the owner, I want normal INFO-level operation to never produce one log record per generated token, per attention/transformer layer, per CUDA kernel launch, per memory copy, or per scheduler iteration, so that turning logging on doesn't turn into an accidental profiler.
2. As the owner, I want high-frequency internal state (e.g. KV-cache fill level) to be logged only on meaningful state transitions (e.g. "KV cache pressure crossed 90%"), not on every allocation, so that the signal-to-noise ratio of INFO logs stays high.
3. As the owner, I want TRACE/DEBUG-level high-frequency diagnostics to exist for when I need them but be disabled (and effectively free) during normal production operation, so that I can turn on deep visibility for a debugging session without needing a rebuild.
4. As the owner, I want log record creation and physical I/O to be decoupled so that a slow sink (a full disk, a stalled pipe) cannot add latency to a decode step, so that logging failures degrade gracefully instead of stalling inference.
5. As the owner, I want the logging queue to be bounded with a hard cap, so that a burst of log volume cannot grow memory usage without limit and eventually OOM the process.
6. As the owner, I want ERROR/WARN events to not be silently discarded under normal backpressure, so that I don't lose the one log line that would have told me why something failed, while a burst of low-priority DEBUG/TRACE noise is allowed to drop.
7. As the owner, I want graceful shutdown to flush pending important log events before the process exits, so that the last thing that happened before a clean shutdown isn't lost.
8. As the owner, I want shutdown to never block indefinitely waiting on the logging queue, so that a stuck sink can't turn a graceful shutdown into a hang.
9. As the owner, I want every logging/tracing change that touches the prefill/decode/CUDA-graph path to be verified against the G4 performance gate (≥99% of reference performance) before merge, so that "we designed it to be cheap" is backed by a measurement, not just an assumption.

## Implementation Decisions

- Audit `crates/core`, `crates/runtime`, and the prefill/decode/CUDA-graph-adjacent Rust code (not `kernel/`, which is out of scope per ADR on §33/34) for any log call site that would fire per-token, per-layer, per-kernel-launch, or per-allocation at INFO or above; demote to DEBUG/TRACE or convert to a state-transition event (e.g. KV pressure crossing a threshold) as appropriate.
- Two-channel bounded queue in `crates/logging`, both fed by the same `tracing` event stream via the JSON/pretty `Layer`s from Phase 1: a DEBUG/TRACE channel that drops the oldest entry when full (bounded, lossy under pressure by design), and an INFO/WARN/ERROR channel that is bounded but blocks briefly (rather than dropping) when full — acceptable because these events are rare by construction (state transitions, lifecycle events, errors), never emitted from the true per-token inner loop.
- Likely built on `tracing-appender`'s non-blocking writer machinery (or an equivalent) rather than a bespoke channel implementation from scratch, consistent with ADR 0011's choice to lean on the `tracing` ecosystem rather than reinvent it — final call on off-the-shelf vs custom is the implementer's, as long as the two-priority/bounded/no-silent-ERROR-drop properties hold.
- Shutdown: on `ignis.process.stopping`, flush the INFO/WARN/ERROR channel with a bounded timeout (e.g. a small fixed budget); do not wait indefinitely on a stuck sink. `ignis.process.stopped` is the last event emitted.
- Gate: any change under this issue that touches prefill/decode/CUDA-graph code must be run through `ignis-bench gate` (the existing G4 harness, ADR 0007) before merge, comparing against the pre-change baseline — not just a design review.

## Testing Decisions

- Non-GPU tests (`crates/logging`, `MemorySink`-equivalent) covering: the DEBUG/TRACE channel drops the oldest entry rather than blocking or growing past its bound; the INFO/WARN/ERROR channel never silently drops under a simulated burst within its bound; shutdown flush emits all pending INFO/WARN/ERROR events queued before the flush call, within the timeout budget; shutdown does not hang when the sink is artificially stalled past the timeout.
- A static/structural check (grep-based or a small custom lint, per spec §40) flagging any newly introduced INFO-or-above call site inside files/modules known to be on the per-token/per-layer hot path, to make regression toward hot-path logging visible in review rather than only caught by the performance gate after the fact.
- **GPU test: this issue's acceptance criterion is a passing `ignis-bench gate` run (G4, ≥99% of reference performance) with the Phase 1-3 logging system fully wired into `ignis-server`, following ADR 0006 (exclusive GPU testing) — run when the 5090 is free, fails rather than skips if it isn't.**

## Out of Scope

- Any new log call site beyond what Phase 2 already migrated — this issue is about the queue/backpressure/shutdown machinery and auditing for hot-path violations, not further adoption.
- Trace/span correlation (Phase 4) — spans are a separate concern from this issue's queue/backpressure work, though Phase 4's spans will need the same "no per-token" discipline this issue establishes.
- The C/C++/CUDA/FFI logging bridge (§33/34) — still N/A; this issue covers Rust-side hot-path logging only.

## Further Notes

- Depends on Phase 1 and Phase 2 landing first.
- This is the phase where "no impact on inference latency/throughput" (spec §1, §26, §27) gets a real measurement, not just a design argument — see ADR 0011's context note on why this was made an explicit gate rather than a review checklist item.
- Full spec: `.scratch/Ignis Structured Logging Specification.md` §25-28. Roadmap: `.scratch/logging-roadmap.md`.
