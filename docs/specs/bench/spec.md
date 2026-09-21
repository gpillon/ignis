# Bench harness — spec

The Rust bench crate (`ignis-bench`) is the trace-replay harness + the
v1 acceptance gate. It re-sends a recorded "1 main agent + N subagents"
load trace (JSONL) against the running engine and produces the performance
report. Per `docs/design/ignis-v1.md` §4 (acceptance) and §6 (bench crate).

The reference is recorded with the **same** harness so comparisons are
apples-to-apples (eager CUDA graph on both sides). The reference is a
*speed reference only* (ADR 0005).

## v1 scope (priority order)

1. **Trace replay harness** — `bench-01`. Re-send a recorded JSONL load trace
   ("1 main agent + N subagents") against the engine; drive the scheduler
   with a realistic concurrent load; collect per-request ttft / tok-s.
2. **Canary suite + performance report + v1 performance gate (99%)** —
   `bench-02`. Run the canary suite (fixed, high-signal prompts) for
   divergence detection; produce the **performance report** (tok-s, ttft vs
   reference) + a self-consistency check. **The v1 acceptance: ≥ 99% of the
   reference's performance (throughput / latency) on the trace-replay load,
   with a per-class ttft / tok-s check. NOT token-agreement** — correctness
   is self-checked (sane output, same model, greedy, fixed seed). The
   divergence report is shipped.

## Acceptance

- The harness re-sends the recorded trace and produces per-request ttft /
  tok-s.
- The canary suite detects divergences; the performance report is produced.
- **v1 gate: ≥ 99% of the reference's performance** on the trace-replay
  load (ADR 0007), with a per-class ttft / tok-s check; a self-consistency
  check passes; the divergence report is shipped.

## References

- Design: `docs/design/ignis-v1.md` §4 (acceptance), §6 (bench crate).
- ADRs: 0005 (performance-first, reference = speed-only), 0006 (exclusive
  GPU testing), 0007 (performance gate, not parity).
- Upstream: `server-01` (HTTP to replay against), the full v1 feature set
  (the gate measures the whole engine).