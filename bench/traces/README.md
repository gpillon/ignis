# bench/traces — recorded traces + gate artifacts (the v1 gate input)

Repo-root layout per `docs/archive/ignis-v1.md` §6: the *recorded* load
traces (JSONL) and the artifacts the `ignis-bench` harness produces from
them. The v1 99% performance gate (ADR 0007) is only runnable on
**recorded** data: a trace recorded against the reference stack + a
reference run recorded with the *same harness* (apples-to-apples — eager
CUDA graph on both sides, design §4). The synthetic fixture
(`crates/bench/tests/fixtures/main_plus_10.jsonl`) is **not** a
reference — it only shapes the test load.

## Layout

| File | Produced by | Contents |
|---|---|---|
| `<load>-trace.jsonl` | recorded against the reference stack | the "1 main + N subagents" load trace (replayed against both engines) |
| `<load>-ignis.json` | `ignis-bench replay --out` | per-request metrics of the ignis run |
| `<load>-ninfer.json` | `ignis-bench replay --out` | per-request metrics of the reference run (the baseline) |
| `<load>-canary.json` | `ignis-bench canary --out` | the divergence report (canary self-consistency) |
| `<load>-v1-gate.json` | `ignis-bench gate --out` | the shipped v1 acceptance artifact (performance report + divergence report + verdict) |

The trace lines follow the `TraceLine` shape (`crates/bench/src/trace.rs`):
`{"id", "class" ("main"|"sub"), "t_arrive_ms", "prompt", "max_tokens", "stream"}`
— one main agent + N subagents, arrivals staggered over time.

**`<load>-trace.jsonl` is never committed** (it contains real working
content — repository text, paths, whatever the recorded session touched;
see `.gitignore`). What ships in git instead is `<load>-trace.meta.json`:
the trace's SHA-256 and its shape (request count, class split, arrival
span, prompt-length distribution) — enough for a later reader to audit a
gate run without the trace itself. Both a gate run's own record (`g4.rs`'s
`trace_sha256`) and this file carry the same hash, so a reader can prove
which trace a given verdict replayed.

## Procedure (GPU-exclusive, ADR 0006)

1. **Record the trace** against the reference stack as JSONL (the prompts
   and arrival offsets both engines will see — the reference side is the
   speed reference only, ADR 0005). A live coding-agent client works, or
   `bench/sim/simulate-session.ps1` for a scripted "1 main + N subagents"
   session when standing one up by hand is too slow to reproduce — see
   `bench/sim/README.md`.
2. **Reference run (the baseline):** start the reference engine, then
   `ignis-bench replay --trace bench/traces/<load>-trace.jsonl --endpoint <ref-url> --label ninfer --out bench/traces/<load>-ninfer.json`
3. **ignis run:** start `ignis-server`, then
   `ignis-bench replay --trace bench/traces/<load>-trace.jsonl --endpoint <ignis-url> --label ignis --out bench/traces/<load>-ignis.json`
4. **Canary suite:** `ignis-bench canary --endpoint <ignis-url> --out bench/traces/<load>-canary.json`
   (greedy + fixed seed; exits non-zero when a canary diverges).
5. **The v1 gate** (the 99% performance gate **and** the self-consistency
   check, ADR 0007):
   `ignis-bench gate --ours bench/traces/<load>-ignis.json --ref bench/traces/<load>-ninfer.json --canary bench/traces/<load>-canary.json --out bench/traces/<load>-v1-gate.json`
   — exits 0 only when the v1 verdict passes (≥ 99% of the reference's
   speed per class **and** self-consistent canaries). A performance-only
   comparison (no canary section) is `ignis-bench report`.

## Replaying an agentic trace faithfully (opt-in)

A trace line's `prompt` may be the raw JSON request body a real agent client
sent (messages + tools), as in a trace recorded from a live coding session.
The default replay wraps that text as **one flat user message**, so two
servers answer a JSON blob and their outputs diverge: the work is not the
same. Every baseline in this directory was recorded that way, so the default
stays; two flags, both off unless given, change it for a study such as "with
parallel agents, does MTP pay?" (GitHub #329 item 14):

- `--as-recorded` sends a line whose `prompt` parses as a chat body (an
  object with `messages`) as that body: messages, `tools`, `tool_choice` and
  every other recorded field. The harness still sets `max_tokens` (the
  line's), `stream`/`stream_options`, `temperature: 0`, `seed: 0`, the
  `class` extension and `enable_thinking` where set, and drops the recorded
  `model`. Any other line is sent flat, as before; the replay prints "N of M
  lines sent as recorded" so a trace of non-chat lines does not pass for a
  faithful one. Compare an as-recorded run only with another as-recorded
  run, never with a flat baseline. The body is greedy (`temperature: 0`),
  not the client's own sampling, so a speculative-decoding study ("does MTP
  pay?") reads an **upper bound** on acceptance: greedy drafts match more
  often than sampled ones. A reply made only of tool calls has no text
  chunk to time, so it counts in the request totals and the ttft but is left
  out of the decode rate.
- `--retry-engine-full` waits out a `503` `engine_full` (more concurrent
  requests than the server has lanes) and resends instead of recording a
  failure: it waits the `Retry-After` seconds (1 s without the header),
  bounded to `--retry-max-wait S` (default 30, at most 60: the server's
  `Retry-After` is 1 to 60 s), at most `--retry-max N` resends (default 100)
  and `--retry-total-wait S` of waiting in all (default 600) per request: a
  request with a `max_tokens` of tens of thousands holds its lane for
  minutes, so it is the budget that ends a request which keeps being
  refused. The run file then carries `retries` and `retry_wait_ms` on the
  requests that queued (omitted when zero); `ttft_ms` and `total_ms` time
  the admitted attempt, so queueing is the waiting and service is the rest.
  `retry_wait_ms` is the waiting slept, not the refused round trips.
  `report` and `gate` print the retries and the wait per class when there
  are any, since a run that queued for minutes passes the speed gate.

```
ignis-bench replay --trace bench/traces/<load>-trace.jsonl --endpoint <url> --conc 11 \
  --as-recorded --retry-engine-full --label ignis --out <run>.json
```
