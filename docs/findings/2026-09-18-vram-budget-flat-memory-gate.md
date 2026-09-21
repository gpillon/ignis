# VRAM budget holds memory flat under agent load

- Kind: experiment
- Status: current
- Observed: 2026-09-18
- Last verified: 2026-09-18
- Scope: serving / device memory reservation, KV-RAM pinned arena, retained slots, WDDM paging
- Related: https://github.com/gpillon/ignis/issues/214, https://github.com/gpillon/ignis/issues/207, [ADR 0030](../adr/0030-device-memory-reserved-at-load.md), [ADR 0029](../adr/0029-cross-request-state-reuse.md), [spec](../specs/vram-budget/01-vram-budget.md), [run record](../../.scratch/vram-budget/gate-214/FINDINGS.md), https://github.com/gpillon/ignis/issues/218, https://github.com/gpillon/ignis/issues/219
- Superseded by: none

## Question

The 2026-09-17 measurement (`.scratch/vram-analysis/REPORT.md`) found ignis's
process commit growing 3.16 GiB in 14 minutes under a live agent load, after
which Windows paged device allocations to system RAM and TTFT went erratic.
With #207's slices in (budget and load plan, retained slots, vision sharing the
prefill scratch, KV-RAM as one pinned arena), does the same load still grow the
process, and does the printed plan match what the operating system reports?

## Evidence

One launch on `vram-budget` @ c9e3efe, RTX 5090 (32,607 MiB) with the card
exclusive and ~1.6 GiB of desktop. `make start` defaults plus `--vision`,
262,144 context, hq-e8-2b, DFlash2/7, prompt reuse on, 8 GiB KV-RAM.

Load: the **raw** `bench/traces/reuse-191-trace.jsonl` replayed with
`ignis-bench replay-raw --reuse on --preserve-thinking --max-gap 10` — the
bodies carry two leading `system` messages and no merged copy was used —
beside four parallel multi-turn `@agent` conversations
(`.scratch/vram-analysis/parallel_lane_load.py`, prompts 50K→87K tokens).
71 trace requests and 216 agent turns were served in the window.

Sampling: `.scratch/vram-analysis/gpumem.ps1` every 3 s — the per-process WDDM
counters Task Manager shows — for 24.5 minutes under load. Raw csv, server log,
`/metrics` scrape and the run record:
`.scratch/vram-budget/gate-214/`.

Per-process counters, MiB:

| Moment | Dedicated | Shared | Commit |
|---|---:|---:|---:|
| right after the load | 28574 | 8266 | 36840 |
| after the first request | 28576 | 8266 | 36842 |
| maximum over the run | 28610 | 8268 | 36878 |
| last sample under load | 28578 | 8268 | 36846 |

Against the 2026-09-17 baseline, same script and same criteria
(`ignis-run2-mem.final.csv`, 14.1 min):

| Quantity | 2026-09-17 | 2026-09-18 |
|---|---:|---:|
| window under load | 14.1 min | 24.5 min |
| requests served | 55 | 287 |
| first request's cost (commit) | +686 MiB | **+2 MiB** |
| commit spread *after* the first request | 2547 MiB | **36 MiB** |
| dedicated spread *after* the first request | 1969 MiB | **34 MiB** |
| commit, load → end | +2734 MiB | **+6 MiB** |
| process shared | 74 → 976 MiB, climbing | **8266 → 8268 MiB** |
| dedicated/shared trading at constant commit | 23 samples | **0** |

Both spreads are measured on the same basis — every sample after the first
request — so the one-off cost of that request is excluded from both rather
than inflating the baseline's. The first request's +686 MiB is the csv's
figure; `REPORT.md` states it as +670 from a slightly different sample pair.

`ignis.runtime.vram_plan` (mode `derived`) totals 29,945,673,472 B = 28,558 MiB:
weights 17,724, kv_pool 3,931, workspace 2,117, lane_state 1,822,
retained_slots 1,822, cuda_context 432, media_embedding 320, verify_round 145,
decode_graph 83, drafter_round 74, residual 84, sampling 4 MiB.

HTTP over the whole run: 543 × 200, 20 × 503, **zero 4xx**, zero `leaf_error`
lines. The 20 × 503 are the documented admission backpressure, all of them
parallel-load turns during a moment when the measurement itself doubled its own
workers.

## Finding

**Observed.**

- Process commit is flat: +6 MiB from load to the end of a 24.5-minute window,
  spread 36 MiB over 289 samples. The tolerance stated for this run is ±64 MiB,
  taken from the sample noise itself; the baseline missed it by two orders of
  magnitude.
- The process's "Shared GPU memory" equals the KV-RAM arena and does not move:
  8266–8268 MiB, which is the 8,192 MiB pinned arena plus the 74 MiB the
  process already showed before the arena was pinned.
- There is no sample where commit stays constant while dedicated and shared
  trade places — the signature of WDDM paging that the 2026-09-17 run showed 23
  times.
- The printed plan total (28,558 MiB) matches Task Manager's dedicated figure
  right after the load (28,574 MiB) to 16 MiB, 0.06%.
- KV capacity under this plan is 447,296 tokens (6,989 pages × 64), against a
  262,144 `max_context`. Retained slots are 8, one per decode lane; they
  saturate and stay saturated, and 20 retained operations were skipped for want
  of a slot (15 capture, 5 publish) rather than served with a fresh
  `cudaMalloc`. Every skip happened with all 8 slots held, so the pool skips
  only when it is full.
- Reuse keeps working while slots are saturated: 3,688,114 tokens reused from
  device-resident state and 10,134,354 from KV-RAM, 195 hits, 225 spills, 0
  device spills.

**Inferred.**

- The first-request delta is the sharpest single indicator of the change in
  kind: 686 MiB on 2026-09-17 (lazy CUDA init, first prefix image, first
  checkpoint) against 2 MiB now. Nothing of consequence is allocated on the
  request path any more; what is left is live bookkeeping.
- Saturated slots plus continuing reuse is the designed trade working as
  intended — a full pool degrades to skipping, not to allocating, and not to
  refusing.

## Implications

- The per-process "Shared GPU memory" figure is now a meaningful assertion
  about the KV-RAM arena rather than a paging symptom. A future run that shows
  it moving is reporting a regression, and the arena size is the number to
  compare it against.
- `vram_plan` can be trusted as the prediction of the process's dedicated
  footprint to within tens of MiB, so a plan that fits is evidence the load
  will fit — on this card, with this desktop.
- Memory is no longer a candidate explanation for erratic TTFT (#204) at these
  settings; a future TTFT investigation should look elsewhere first.
- The vision encoder sharing the prefill scratch (#212) shows up in the plan as
  a single `workspace_bytes` of 2,219,837,184 B rather than a separate 2.38 GiB
  arena, which is where the KV pool's extra pages came from.

## Limits and unknowns

- One card, one desktop size, one launch. The plan is derived from memory free
  at start, so a different desktop footprint produces a different plan; nothing
  here establishes behaviour when the desktop grows *while* serving (explicitly
  out of scope in the spec).
- 71 of the trace's 157 requests were replayed. The window was 10 minutes past
  what the ticket requires and the memory verdict had saturated, so the run was
  stopped by the owner, who then accepted that coverage and closed the
  follow-up as not planned. The remaining 86 requests would have added render
  coverage, not memory coverage; what stays unestablished is whether some body
  among them renders differently for a reason this run could not see.
- No image request was served. `--vision` is exercised here as a reservation
  that the plan accounts for, not as an encode path; the encode path has its
  own GPU tests.
- `oversubscribed` was false for the whole run. Nothing here measures what the
  explicit-budget or allow-oversubscription modes do under the same load.
- The 20 × 503 came from the measurement over-driving itself, so this run does
  not establish where the admission ceiling actually sits.
- The KV token capacity quoted here was derived (`page_count` ×
  `KV_PAGE_TOKENS`) rather than read, because the log redacted it during the
  run. A later start, after the fix, logs it directly and agrees with the
  derivation — but that is a different plan on a different amount of free
  VRAM, not this run's number re-read.

## Follow-ups

- https://github.com/gpillon/ignis/issues/218 — `ignis.runtime.kv_pool` logged
  `token_capacity` and `bytes_per_token` as `[REDACTED]`, because
  `is_sensitive_key` treated the word `token` as a credential wherever it
  appeared. Fixed on this branch after the run and verified on a later start,
  which logged `token_capacity 453824` and `bytes_per_token 9216` in clear —
  confirming the derivation used here (that start's plan holds 7,091 pages
  rather than 6,989, from a different amount of free VRAM).
- https://github.com/gpillon/ignis/issues/219 — the 86 raw-trace requests this
  run did not reach. Render coverage only; the memory criteria do not depend on
  them. Closed as not planned: the owner accepted 71/157 rather than spend
  another ~40 minutes of exclusive GPU on it.
- The run record lists the run's departures and the exact commands:
  [`.scratch/vram-budget/gate-214/FINDINGS.md`](../../.scratch/vram-budget/gate-214/FINDINGS.md).
