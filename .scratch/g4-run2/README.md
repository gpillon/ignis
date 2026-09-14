# G4 gate records, run 2 (GitHub #65)

The second run of gate G4. The first (#128, PR #140, session
`g4-20260912T162610Z`, records under `.scratch/g4-logs/`) reached a verdict
whose primary cells were unreadable: it filed **#137** (the bench SSE reader
ignored `delta.reasoning_content`), **#138** (an undeclared 30 s request
deadline killed the 128K needle cell on both engines), **#139** / **#146**
(the G3 ITL fixture refused the reference's leg), **#143** (a 0.99 threshold
narrower than the cell's own noise), **#144** (the canary budget under
thinking) and **#147** (a trace request that generated nothing and was scored
as ok). All eight issues are closed and merged; this run is the follow-up
"two launches per engine with a fixed harness" that #128's verdict asked for.

The verdict itself lives in `.scratch/REVIEW-2026-09-05.md` §6 Phase 4.

## The session

One GPU-exclusive session (ADR 0006), session id **`g4-20260913T163557Z`**,
against `F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer` on a free
RTX 5090, `ninfer-serve` stopped for every ignis leg and started fresh for
every reference leg. Nothing else touched the card: this session ran as a
single agent leg in the main worktree, which is the condition
[`docs/findings/2026-09-13-gpu-profile-card-fits-one-run.md`](../../docs/findings/2026-09-13-gpu-profile-card-fits-one-run.md)
names as the one #128 did not meet.

**Both engines at hq-e8-2b, matched capacity.** Each launch resolved
**465,984 tokens / 7,281 pages**: ignis from its 4 GiB auto budget
(`ignis.runtime.kv_pool`), the reference from an explicit `--kv-capacity
465984`. Both at `--max-context 262144` and `--prefill-chunk 1024`, CUDA
graphs and prefix reuse on for both.

**Four independent process launches** (ADR 0021), stopped and restarted
between each, in order: reference 1, ignis 1, reference 2, ignis 2.

## The trace

Recorded fresh this session (ADR 0015 / ADR 0021: the trace is regenerated
alongside the reference's own live run, never read from a committed file)
through `ignis-bench record` in front of the reference, driven by
`bench/sim/simulate-session.ps1` with `bench/sim/g4-gate-session.json`.
11 requests (1 main + 10 sub), arrivals over 21.2 s, prompts 21 KB-83 KB of
real current-repository content, 12m18s to last completion, **11/11
status=200**, and no error, warn or panic line in the reference's log across
the whole recording. Hash and shape: `bench/traces/g4-load-trace.meta.json`
(`00fd08b810c44ed6ec1a422e4a0c476af786bccdc0219e733d346bd5a0c4ab01`). The
trace file itself is not committed — it carries real working content.

## Records

| file | what it is |
|---|---|
| `ninfer-launch1-g4.json` | Reference launch 1, trace replay + needle cells. main 23.9 tok/s, sub 27.1 tok/s, needle@64K and @128K both RETRIEVED. |
| `ignis-launch1-g4.json` | ignis launch 1, same. main 30.5 tok/s, sub 26.4 tok/s, both needles RETRIEVED. |
| `ninfer-launch2-g4.json` | Reference launch 2. main 25.5 tok/s, sub 28.7 tok/s, both needles RETRIEVED. |
| `ignis-launch2-g4.json` | ignis launch 2. main 31.8 tok/s, sub 27.6 tok/s, both needles RETRIEVED. |
| `g4-verdict.json` | `ignis-bench g4-gate` pooled over all four launches: main **1.264 PASS**, sub **0.969 FAIL**, aggregate **1.013 PASS**, needle@64K and @128K **PASS**. Verdict FAIL, on the `sub` cell alone. |
| `ninfer-launch1-g3.json`, `ignis-launch1-g3.json` | The G3 cells under hq on both sides, launch pair 1. |
| `ninfer-launch2-g3.json`, `ignis-launch2-g3.json` | The same, launch pair 2. |
| `g3-verdict-pair1.json` | C=1 0.990, C=4 3.197, ITL p95 0.986. |
| `g3-verdict-pair2.json` | C=1 0.982, C=4 3.836, ITL p95 1.028 (flagged: the reference's window covered 6 of 10 prefill windows). |
| `ignis-launch1-canary.json` | Canary self-consistency against ignis at **default flags** (thinking on): 4/4 `sane=true deterministic=true`, `self-consistency: PASS`. |
| `dogfood-request.json`, `dogfood-raw-sse.txt` | The dogfood cell: one real tool-calling turn against ignis launch 2, captured as raw SSE. |
| `*-server.log`, `*-server.log.err` | Each launch's own engine log. |
| `gpu-profile.log` | The GPU profile's own log (gitignored). `kernel/build.ps1 -Test` 40/40, then the serialized `--ignored` Rust sweep 39 passed / 0 failed / 0 skipped under `IGNIS_GPU_PROFILE=1`, exit 0, first attempt, 711.4 s. |
| `kvab/` | The KV-format A/B (the 2x2 over engine and format) — see its own README. |

## Reading the two FAILs

**`g4-gate` exits 1 on the `sub` cell at 0.969**, against spec 04's 0.99
per-class floor. That is the gate's own arithmetic on a cell that is now a
real measurement: both launches agree in direction (0.975 and 0.962), ignis's
across-launch spread is 2.2% and the reference's 2.8%, so it is not one
unlucky launch. It is filed, not waived.

**`g3-gate` exits 1 on C=1 in both pairs (0.990, 0.982), and that exit code
is not this cell's verdict at G4.** Spec 04 reads the G3 cells inside G4 as a
**regression band** — every ratio reported, the cell failing only below 0.90
on throughput or above 1.10 on ITL p95 — because the 0.99 threshold sits
inside the cell's own launch-to-launch noise (#143). Every ratio in both
pairs is inside that band.

## The KV-format A/B (the 2x2), also live/live

Eight more legs in their own session (`kvab-20260913T194359Z`), two launches per
(engine, format) cell at a matched 65,536-token capacity — see
[`kvab/README.md`](kvab/README.md). Pooled:

| engine / format | C=1 | C=4 | ITL p50 | ITL p95 |
|---|---:|---:|---:|---:|
| ignis hq-e8-2b | 54.2 | 29.4 | 194.38 | 262.81 |
| ignis BF16 | 58.6 | 38.2 | 169.08 | 224.78 |
| ninfer hq-e8-2b | 51.3 | 9.6 | 167.56 | 254.80 |
| ninfer BF16 | 57.8 | 19.7 | 152.18 | 218.33 |

hq to BF16 costs ITL p95 **-14.5%** on ignis and **-14.3%** on the reference —
the format's toll, not ignis's, now measured the way a gate measures. Written up
as [`docs/findings/2026-09-13-hq-vs-bf16-live-live.md`](../../docs/findings/2026-09-13-hq-vs-bf16-live-live.md).
