# 01 — trace replay harness (JSONL load trace)

Status: needs-triage
Blocked by: server-01

Re-send a recorded "1 main agent + N subagents" load trace (JSONL) against
the running engine (`docs/design/ignis-v1.md` §4, §6). Drive the scheduler
with a realistic concurrent load; collect per-request ttft / tok-s. The
reference is recorded with the **same** harness so comparisons are
apples-to-apples (eager CUDA graph on both sides).

## Acceptance

- The harness replays the recorded trace and produces per-request ttft /
  tok-s.
- A realistic "1 main + ~10 subagents" concurrent load is driven into the
  scheduler.