# 02 — canary suite + performance report + v1 gate (99%)

Status: needs-triage
GitHub: #20
Blocked by: #19 (bench-01) (and the full v1 set: #16 (core-05), #18 (core-07), #15 (server-02), #10 (kernel-abi-03))

Run the **canary suite** (fixed, high-signal prompts) for divergence
detection; produce the **performance report** (tok-s, ttft vs reference) +
a self-consistency check. This is the **v1 acceptance gate**
(`docs/design/ignis-v1.md` §4, ADR 0007):

- **≥ 99% of the reference's performance** (throughput / latency) on the
  trace-replay load, with a per-class ttft / tok-s check.
- **NOT token-agreement** — correctness is self-checked (sane output, same
  model, greedy, fixed seed).
- The **divergence report** is shipped.

## Acceptance

- The canary suite detects divergences; the performance report is produced.
- **v1 gate: ≥ 99% of the reference's performance** (ADR 0007); the
  self-consistency check passes; the divergence report is shipped.
