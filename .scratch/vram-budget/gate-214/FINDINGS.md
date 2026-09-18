# Gate #214 — run record

2026-09-18, branch `vram-budget` @ c9e3efe. The durable result and every
number that outlives this run live in the finding:
[`docs/findings/2026-09-18-vram-budget-flat-memory-gate.md`](../../../docs/findings/2026-09-18-vram-budget-flat-memory-gate.md).
This file is what the run itself did — commands, departures, and where the raw
evidence is.

Units: binary MiB/GiB, as Task Manager's per-process WDDM counters report them.
`mem.csv` timestamps are local (UTC+2); server log timestamps are UTC.

## Acceptance criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Process commit flat after the first request, within a stated tolerance | **PASS** — spread 36 MiB over 24.5 min, tolerance ±64 MiB (§Tolerances) |
| 2 | Process shared equal to the KV-RAM arena and not moving | **PASS** — 8266–8268 MiB, spread 2 MiB; arena 8192 + 74 MiB pre-arena baseline |
| 3 | No dedicated/shared trading at constant commit | **PASS** — 0 samples (baseline: 23) |
| 4 | `vram_plan` total matches Task Manager's dedicated right after load | **PASS** — 28558 vs 28574 MiB, +16 MiB, tolerance ±32 MiB |
| 5 | Every request of the raw trace served, zero `leaf_error` storms | **PARTIAL, accepted by the owner** — 71/157 served, all 2xx (zero 4xx in the run), **0** `leaf_error`; stopped and then accepted on the owner's call (§Departures) |
| 6 | Record KV capacity, retained slots in use, publish/capture skips, comparison table | **Done** — §Recorded numbers, and the finding's comparison table |
| 7 | A failed check files a follow-up with the evidence, no tuning | **Done** — #218 for the redaction that hid the KV token capacity (fixed); #219 for the 86 unserved trace requests, which the owner then closed as not planned, accepting the coverage. Neither was tuned around |

## The run

Launch, from this worktree:

```
make start ARTIFACT=F:/ai/opencode/inference/models/qwen3_8_27b_nvfp4full-v2.ninfer \
           ARGS="--vision" LOG_LEVEL=debug METRICS=1
```

Resolved flags: `--kv-format hq-e8-2b --max-context 262144 --prefill-chunk 1024
--kv-host-pool-bytes 8G --request-timeout 1800 --spec dflash2 --draft-tokens 7
--ui --metrics --system-message-policy merge --developer-message-policy inplace
--vision`. Ready in 11.2 s.

Load, the **raw** trace with no merged copy:

```
ignis-bench replay-raw --trace bench/traces/reuse-191-trace.jsonl \
  --endpoint http://127.0.0.1:8000 --label gate-214 --reuse on \
  --model qwen3.8-27b --preserve-thinking --max-gap 10
```

beside `.scratch/vram-analysis/parallel_lane_load.py` with 4 workers on the
`@agent` lane, in rounds of 10 turns: 216 turns, prompts 50K→87K tokens.

Sampling: `.scratch/vram-analysis/gpumem.ps1 -IntervalSec 3`, 306 samples,
01:03:51 → 01:29:34 local, 294 of them with the model loaded. Window under
load 01:05:02 → 01:29:29 = 24.5 min (the ticket asks for at least 14).

GPU exclusive: `make gpu-status` before the run; a stale `ignis-server`
(pid 21744) from the previous session was stopped by the owner first. Card back
to 2506 MiB (desktop only) after `make stop`.

## Recorded numbers

- **KV token capacity 447,296** — 6989 pages × 64 (`KV_PAGE_TOKENS`,
  `crates/core/src/kv_format.rs:30`). Derived during the run, because the log
  redacted it (#218, fixed afterwards on this branch and verified live: a
  later start logged `token_capacity 453824` = 7091 × 64, and
  `bytes_per_token 9216`, both in clear — the capacity differs only because
  that start derived its budget from a different amount of free VRAM).
  `hq-e8-2b`, `max_context_tokens` 262,144, pool 4,122,279,936 B on the
  `kv_pool` line — 128 KiB under the plan's `kv_pool_bytes` of 4,122,411,008,
  which is the plan's line for the pool rather than the pool's own budget.
- **Retained slots: 8**, one per decode lane (the default), from 505
  `ignis.scheduler.retained_slots` DEBUG events:

  | slots in use | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | samples | 1 | 11 | 27 | 45 | 65 | 95 | 155 | 106 |

- **Publish/capture skips: 20** — 15 `capture_skipped_no_slot`,
  5 `publish_skipped_no_slot`, 0 `capture_skipped_no_page`. **Every one of
  them happened at 8 slots of 8**: the pool skips only when it is full, never
  as a way of avoiding work it had room for.
- **`/metrics` at 01:28:53**: retained reused tokens 3,688,114 device /
  10,134,354 kv_ram; hits 27/168; misses 261/94; spills 0/225; discards 19/216;
  restores 27/168; `ignis_prefix_reused_tokens_total` 407,040;
  `ignis_kv_cache_evictions_total` 65.
- **Requests**: 289 admitted, 288 done. By client: 216 parallel turns and 1
  canary completed, leaving **71 completed raw-trace requests**; the 289th
  admission is the 72nd trace request, in flight when the run was stopped,
  which is also why 70 responses streamed rather than 71. Zero `leaf_error`
  lines (2026-09-17: 345K lines, 105 MB of log, in about a minute, on these
  same bodies).
- **HTTP**: 543 × 200, 20 × 503, **zero 4xx**. The histogram is the evidence
  for criterion 5, not the failure lines: tower-http classifies only 5xx as a
  failure, so a `render_failed` or `invalid_role` 400 would be invisible there.
  70 responses streamed (`on_eos`), matching the trace requests; the remaining
  200s are `/v1/models` readiness and progress polls.
- **20 × HTTP 503 "engine full"** at 01:15:26, all parallel-load turns
  (`parallel.jsonl`). Documented admission backpressure
  (`crates/server/src/api.rs:641-670`), triggered by the measurement itself:
  two 4-worker rounds briefly overlapped, so 8 conversations plus the replay
  hit the lanes at once. No trace request was refused.
- **214 × `api.rs:797`** ("reasoning but no content", `finish_reason: length`)
  — the parallel load's 400-token cap cutting the thinking block. Served
  responses, not errors.

## Tolerances

Stated from this run's own sample noise, as the ticket asks:

- **Commit flat: ±64 MiB** after the first request. Observed spread 36 MiB over
  289 samples — live bookkeeping plus a sampling margin. The band is set at
  roughly twice the observed spread, so it is calibrated by this run rather
  than independently derived; what makes it usable as a verdict is the
  distance to the failing case, not its precision — the baseline's spread on
  the same basis is 2547 MiB, two orders of magnitude out.
- **Shared equals the arena: 8192 + 74 ± 16 MiB.** Observed 8266–8268. The
  74 MiB is what the process already showed at 01:04:28, before the arena was
  pinned.
- **Plan vs Task Manager: ±32 MiB.** Observed +16 MiB.
- **Trading: exactly 0** samples with |Δcommit| ≤ 8 MiB, |Δdedicated| ≥ 50 MiB
  and dedicated/shared moving in opposite directions.

## Departures

1. **71/157 raw-trace requests.** `--max-gap 10` replays the trace's own idle
   gaps, so the full 157 needed about 40 more minutes of exclusive GPU. At
   71/157 the memory window was already 10 minutes past what the ticket asks
   and the verdict had saturated, so the owner chose to stop. What the
   remaining 86 would have added is render coverage, not memory coverage.
   Filed as #219, which is what criterion 7 asks of a check left partial —
   disclosure here is not a substitute for the ticket. **The owner then
   accepted 71/157 as sufficient and closed #219 as not planned**, so the
   coverage is a decision on the record rather than an open gap.
2. **`ARGS="--vision"`** instead of a `VISION=1` Makefile default: that knob is
   commit 726a1b7 on `issue-191`, not on this branch. Same flag either way.
3. **Explicit `ARTIFACT=`**: the worktree has no `models/` directory.
4. **`LOG_LEVEL=debug`**: `ignis.scheduler.retained_slots` is a DEBUG event and
   criterion 6 asks for the slots in use. Change-driven (ADR 0025), so the
   whole run's log is 1.22 MiB — no measurable perturbation.
5. **`METRICS=1`**: the retained-state lifecycle counters exist only on
   `/metrics`. One extra listener, scraped once at the end.
6. **Bench client from `issue-191`**: `target-bench191/…/ignis-bench.exe`
   (built 2026-09-17 12:16, has `--max-gap`). `replay-raw` and the trace live
   on that branch; the client speaks HTTP only, so nothing was merged.
7. **Parallel load in rounds of 10 turns**, relaunched to keep pressure on for
   the whole window; history resets between rounds. Two rounds overlapped once,
   which is what produced the 20 × 503.
8. **No image requests.** The trace has none, so `--vision` is measured as a
   reservation, not as an encode path.

## Workspace state

No code changed — this is a measurement. `cargo test --workspace` at c9e3efe,
after the run: **1386 passed, 0 failed, 1 ignored**.

## Files

- `mem.csv` — the WDDM per-process/adapter counters, 3 s apart
- `server.log` — the whole run's server log (DEBUG)
- `metrics.txt` — `/metrics` at 01:28:53
- `parallel.jsonl`, `parallel.out` — the 4 `@agent` conversations
- `replay.log` — the raw-trace replay (cut short; no `replay.json` written)
- `make-start.log` — the launch
