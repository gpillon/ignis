# bench/traces — recorded traces + gate artifacts (the v1 gate input)

Repo-root layout per `docs/design/ignis-v1.md` §6: the *recorded* load
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

## Procedure (GPU-exclusive, ADR 0006)

1. **Record the trace** against the reference stack as JSONL (the prompts
   and arrival offsets both engines will see — the reference side is the
   speed reference only, ADR 0005).
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