# 03 — gate-run: run the v1 gate end-to-end (trace capture, replay, canary, gate, dogfood)

GitHub: #24

The bench-02 harness is code-complete (bench-01: `replay` / `canary` /
`report`; bench-02: the shipped v1 gate artifact — `ignis-bench gate`
+ `canary --out`). What remains for the v1 verdict (ADR 0007) is the
**recorded side** (`bench/traces/README.md`): a real reference trace
plus a reference run recorded with the *same* harness. This spec is
the operational ticket that runs the v1 gate end-to-end and closes
GitHub #20. It also adds the one missing harness piece: a **stable
capture tool** that records a live agent session as a bench trace
(deliberate harness tooling, not a one-shot script).

**Scope:**

- **Capture tool** (a new `ignis-bench` subcommand, `crates/bench`):
  `record` — a capture-proxy mode:
  `ignis-bench record --listen <proxy> --target <engine-url> --out
  <load>-trace.jsonl [--class <policy>]`. It accepts OpenAI
  chat-completions from a live agent client, records each request as a
  trace line — `id`, `class` (`main` | `sub`), `t_arrive_ms` (offset
  from session start), `prompt` (the actual request content — the
  engine sees the same input the reference saw, per the trace format in
  `crates/bench/src/trace.rs`), `max_tokens`, `stream` — and forwards
  to the target engine. Class policy: `first-is-main` (the first
  request of the session is the main agent, the rest are subagents) or
  a client marker. On session end it writes `<load>-trace.jsonl`.
  CPU-testable against a mock target.
- **The run** (formalizes `bench/traces/README.md`):
  0. Preflight: the kernel-abi-04 compute adapter is merged (the
     server serves a real model); the GPU is free (ADR 0006); the
     reference (ninfer) stack can run locally (`F:\ai\q38`).
  1. **Record**: a real "1 main + ~10 subagents" agent session against
     the *reference* (ninfer) stack, through the capture endpoint →
     `<load>-trace.jsonl`. The synthetic fixture
     `crates/bench/tests/fixtures/main_plus_10.jsonl` is **not** a
     reference (per the README).
  2. **Baseline**: `ignis-bench replay --trace <load>-trace.jsonl
     --endpoint <ninfer:8080> --label ninfer --out
     <load>-ninfer.json`
  3. **ignis run**: `ignis-bench replay --trace <load>-trace.jsonl
     --endpoint <ignis:8000> --label ignis --out <load>-ignis.json`
     (`ignis-server` with `IGNIS_ARTIFACT` → the production adapter)
  4. **Canary**: `ignis-bench canary --endpoint <ignis:8000> --out
     <load>-canary.json` (greedy + fixed seed; exit 0 =
     self-consistent)
  5. **Gate**: `ignis-bench gate --ours <load>-ignis.json --ref
     <load>-ninfer.json --canary <load>-canary.json --out
     <load>-v1-gate.json` — exit 0 = the v1 verdict passes
  6. **Dogfood** (the end-to-end verification, design §4 — both the
     scripted floor and the agent session, per the 2026-09-03
     decision):
     (i) a scripted smoke — a fixed prompt through
     `/v1/chat/completions`, output self-checked (a sane completion);
     (ii) a final "1 main + 10 subagents" agent session against
     `ignis-server` on a real task — completions are sane and
     performance is within the gate (v1 is the dogfood target for the
     developer's own coding agent, design §3/§4).
  7. **Artifacts + bookkeeping**: commit the full `<load>-*` artifact
     set under `bench/traces/`; update `.scratch/PENDING.md`
     (bench-02 → Resolved); close GitHub #20 and this issue.

## Acceptance

- **The capture tool ships**: `ignis-bench record` records a live
  session into a valid bench trace (a unit test against a mock target
  checks the line shape + the arrival offsets; a recorded trace loads
  through `ignis-bench replay`).
- **A real reference trace**: `<load>-trace.jsonl` recorded from a
  real "1 main + ~10 subagents" agent session against the reference
  (ninfer) stack — not a synthetic fixture.
- **The v1 verdict artifact**: `<load>-v1-gate.json` with an
  exit-0 verdict — **≥ 99% of the reference's speed per class (per-
  class ttft / tok-s) AND the canaries are self-consistent** (ADR
  0007: a performance gate, *not* token-agreement; the divergence
  report ships).
- **The dogfood passes**: the scripted smoke (sane output) plus a
  real "1 main + 10 subagents" session against `ignis-server`
  completes with sane output, performance within the gate.
- **The artifacts are committed**: the full `<load>-*` set (including
  the divergence report) under `bench/traces/`; `.scratch/PENDING.md`
  updated; **GitHub #20 closed**.

References: ADR 0005 (the reference is a *speed reference* only —
apples-to-apples load, no token parity), ADR 0006 (exclusive GPU —
the whole run needs the 5090 free), ADR 0007 (the 99% performance
gate + the self-consistency floor). The run depends on kernel-abi-04
(GitHub #22): the gate is only measurable once the server serves a
real model.