# The G3 ITL cell's length is decided by a three-way race the fixture does not control

- Kind: discovery
- Status: current
- Observed: 2026-09-13
- Last verified: 2026-09-13
- Scope: bench / G3 ITL cell, EOS suppression, reference engine capabilities
- Related: [#139](https://github.com/gpillon/ignis/issues/139), [#146](https://github.com/gpillon/ignis/issues/146), [#114](https://github.com/gpillon/ignis/issues/114), [ADR 0015](../adr/0015-g2-live-live-cold-prefix-gate.md), [ADR 0021](../adr/0021-live-live-launch-pooling.md), [spec 03 serving loop](../../.scratch/runtime/specs/03-serving-loop.md), [CONTEXT.md glossary](../../CONTEXT.md)
- Superseded by: none

## Question

Three G3 ITL legs against the reference's second launch refused a verdict, each
because `itl-decode-0` ended before the final prefill window closed (#139). Is
the corpus window selection responsible, is the EOS suppression reaching the
engine, and is this the same defect as the decode-lane test flakes (#146)?

## Evidence

**The corpus cut is deterministic.**
`ttft::generate_cell_prompts_from_corpus_with_suffix` places prompt `index` at
`offset = index * (corpus.len() / count)`, wrapping, and `land_corpus_window`
walks at most `ROTATIONS = 8` fixed rotations from there. No seed, no clock, no
session input, so lane 0 receives the same prompt on every invocation against
the same corpus, template and `decode_prompt_tokens`.
`F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids` holds 65,536 ids, which at a
4,096-token window leaves 16 non-overlapping windows.

**The suppression is sent and has no effect.** The artifact's
`generation_config.json` declares `"eos_token_id": [248046, 248044]` (at byte
13,049,250 of `qwen3_8_27b_nvfp4full-v2.ninfer`), so `eos_token_ids()` is
non-empty and `client::request_body_suppressing_eos` puts both
`ignore_eos: true` and `logit_bias: {"248046": -100.0, "248044": -100.0}` on the
wire. In the reference's own tree (`F:\ai\q38\ninfer`):

- `ignore_eos` appears nowhere in `src/serve/openai_schema.cpp`. Its only
  occurrence in the repository is `src/serve/http_server.cpp:524`
  (`params["ignore_eos"] = false`), a hardcoded field of the `/props` blob,
  never read off a request. Unknown body fields are accepted silently, so no
  `400` reports the drop.
- `logit_bias` is parsed into `SamplingParams::logit_bias`
  (`src/serve/openai_schema.cpp:413-423`) and never applied;
  `src/serve/request.h:107-109` states it "remains parsed for wire
  compatibility; the current public engine sampler has no bias input, so it
  does not affect generation".

ignis honours `ignore_eos` end to end (`crates/server/src/api.rs:217` ->
`DecodeParams` -> `crates/runtime/src/lib.rs:514`) and has no `logit_bias`
support at all.

**The reference's request log agrees.** In
`.scratch/g4-logs/ninfer-launch2-requests.jsonl`, the three lane-0 requests
(ids 19, 38, 57) report `finish_reason: "stop_token"` at 2,857 / 2,670 / 2,747
completion tokens against a requested 4,032; three sibling lanes report
`output_limit` at 4,032 and three report `cancelled`. The `sampling` block those
entries record carries only temperature, top_p, top_k, penalties and seed.

**Lane finishes across the four G3 records** (`.scratch/g4-logs/`):

| record | lane-0 | lanes 1-3 |
|---|---|---|
| `ninfer-launch1-g3.json` | 2,451 `window` | 2,441 / 2,456 / 2,446 `window` |
| `ninfer-launch2-g3.json` | 2,856 `stop_token` | 3,729 / 3,744 / 3,734 `window` |
| `ninfer-launch2-g3-retry.json` | 2,669 `stop_token` | 4,032 `cap` (x3) |
| `ninfer-launch2-g3-retry2.json` | 2,746 `stop_token` | 3,528 / 3,533 / 3,538 `window` |

**Recomputing each record against the span every lane shared** (the maximum of
the lanes' first tokens to the minimum of their ends) and counting the prefill
windows that close inside it:

| record | shared span | prefill windows covered | p95 as recorded |
|---|---:|---:|---:|
| `ninfer-launch1-g3.json` | 103,921.6 ms | 10/10 | 234.74 ms |
| `ninfer-launch2-g3.json` | 101,129.9 ms | 8/10 | 236.31 ms |
| `ninfer-launch2-g3-retry.json` | 86,679.3 ms | 6/10 | 235.07 ms |
| `ninfer-launch2-g3-retry2.json` | 96,165.2 ms | 7/10 | 235.91 ms |
| `ignis-g3-launch1.json` | 74,264.7 ms | 10/10 | 241.79 ms |
| `ignis-g3-launch2.json` | 74,309.1 ms | 10/10 | 241.80 ms |

**The decode-lane tests flake on scheduling, and stop once paced.** On this
machine (20 cores), `cargo test -p ignis-bench --lib` under 16 busy-loop
processes failed 12 of 20 runs before the fixtures were paced against the
prefiller series — `itl_decode_lanes_are_stopped_after_the_final_prefill_window`
and `lanes_that_end_on_the_cap_...`, plus two corpus/pooling tests that read the
same unpaced fixture. After pacing, 25 of 25 runs under the same load were
green. Two flakes seen during that sweep belong to neither issue and are not
addressed here: `ignis_server::telemetry::tests::evicted_and_restored_report_the_tier_s_wall_time`
(fails under a whole-workspace run, 6/6 green alone) and
`ignis_bench::client::tests::results_are_sorted_by_id_and_tok_s_is_computed`
(its `MockEndpoint` hands out canned outcomes in order while `replay` runs at
concurrency 4, so under load `main` draws the wrong one: `tok_s 70` against an
expected 85.2).

## Finding

**Observed.** A decode lane can be ended by any of three things, and the fixture
pins none of them: the harness cancelling it when the prefiller series ends, the
engine's own EOS, or `ITL_DECODE_MAX_TOKENS`. Against the reference the EOS
suppression is inert — neither knob is honoured — and against a fast enough
engine the cap binds first; `ninfer-launch2-g3-retry.json` shows three lanes
ending at the cap on a live run.

**Observed.** Which of the three fires is a function of how long the prefiller
series happens to take relative to the lane's own trajectory, not of anything
the cell states. Launch 1 did not avoid the EOS: lane 0 ended there at 2,451
tokens, a few hundred short of the 2,669-2,856 band where its EOS lands.

**Inference.** The old pooling guard, which refused any lane that ended before
the final prefill window closed, therefore refused runs for reasons that carry
no information about either engine's speed. Worse, the cap arm of the race
tightens as an engine gets faster, so the failure mode grows against ignis
precisely as ignis approaches the goal the gate exists to measure.

**Observed.** Where the old guard accepted a record, the shared-span reading
accepts it with full coverage — the guard's condition (every lane outlives the
last prefill window) implies it. Where the old guard refused, the shared span
still covers 6 to 8 of the 10 prefill windows, and the p95 those three refused
records already carried sits within 0.7% of launch 1's.

**Inference.** The refusals were correct about the cell's *claim* (all four
lanes decoding throughout) and needlessly severe about the *data*. Measuring
inside the shared span keeps the claim true and keeps the leg.

**Observed.** #146's flakes are a different mechanism at the same seam: the test
fixtures emitted canned tokens with no pacing, so whether a lane reached the cap
or was cancelled depended on how the OS scheduled the lane threads against the
prefiller series.

## Implications

- The reference cannot be asked to keep a measurement lane alive. Any cell whose
  validity depends on a lane outliving its own EOS is measuring ignis under
  suppression and the reference without it.
- Patching the reference to honour `ignore_eos` would change the engine that
  every committed G3 reading was taken against, and ADR 0021 pools two launches
  of one binary — so the fix belongs in the instrument.
- A lane the harness cancelled was generating right up to the cancellation, so
  its span ends there and not at its last observed token; treating the last
  token as the end makes the window's close a race between a lane's final SSE
  chunk and the harness's own store.
- Percentiles now stand on the prefill windows actually covered. A record must
  say how many, and a gate must refuse a window too short to hold a
  distribution.

## Limits and unknowns

- Why lane 0's specific corpus window drives the model to an end-of-answer
  around token 2,700 is not established; only that the window is deterministic
  and the behaviour reproduced three times in a narrow band.
- Whether the reference would honour a `stop`-token or `min_tokens` request
  instead was not tested.
- The coverage recomputation above is an offline reading of committed records,
  not a re-run: no leg was re-measured against either engine for this finding.
- `ITL_MIN_COVERED_PREFILLERS = 2` is a floor chosen to refuse an anecdote, not
  a number derived from the percentile's error at a given coverage.

## Follow-ups

- [#139](https://github.com/gpillon/ignis/issues/139) — the ITL cell measures
  the shared span and reports its coverage.
- [#146](https://github.com/gpillon/ignis/issues/146) — the decode-lane test
  fixtures pace against the series rather than the scheduler.
- The G4 leg blocked on #139 needs re-running on both engines with the amended
  cell before ADR 0021's two-launch pooling can close.
