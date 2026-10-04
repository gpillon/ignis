# 06 - measuring the Flash-Next that was built: quality, speed, memory, load

GitHub: #304 (master #298).

Specs 01-05 each carry their own acceptance, but each measures its own part. This
ticket measures the whole: what the owner gets when ignis serves Flash-Next.
It measures it once, at the end, on the merged build. Every number is measured
on the 5090 and set against what the study estimated and what the 27B does on
the same harness.

ADRs:
- 0043 (Accepted 2026-10-04): the checkpoint is the oracle;
- 0044 (Accepted 2026-10-04);
- 0022 (BF16 as the oracle format);
- 0030 (plan lines).

## Decided for the autonomous run (2026-10-04)

The owner asked for this ticket on 2026-10-04: one final ticket measuring what
was implemented. It runs after specs 04 and 05 are merged, on local main, with
nothing else on the card. Benchmarks are error detectors run once at phase end,
not a loop.

## Problem Statement

The study estimated Flash-Next's behaviour on this machine from simulations, a
layer-streamed torch pipeline and a GPU skeleton of the prefill. The owner
decides how to use the model (and whether the phase-2 switch is worth building)
from what the built engine actually does. Five per-spec acceptances do not add
up to that picture.

## Solution

One measurement campaign on the merged build, written up as a finding in
`docs/findings/` with an index row. Each table puts the measured value beside
its estimate and beside the 27B.

## User Stories

1. As the owner, I want Flash-Next's quality against the BF16 checkpoint and against the 27B on the same questions, so that I know what the compression costs and what Flash-Next buys over the 27B.
2. As the owner, I want decode speed at 1, 2 and 3 lanes at short and long context, so that I know how many agents it serves well.
3. As the owner, I want time to first token for cold and warm prompts of 1K, 4K, 8K and 32K, and for an agent turn with and without reuse, so that I know what a request feels like.
4. As the owner, I want VRAM and RAM used, with the plan lines, so that I know what is left for the desktop and other work.
5. As the owner, I want the load time of the server from start to ready, cold and with the page cache warm, so that the phase-2 switch is estimated from a measurement.
6. As the owner, I want every gap between a measurement and its estimate explained or turned into an issue, so that nothing is discovered later.

## Implementation Decisions

**Quality** (BF16 KV for the oracle checks, then hq-e8-2b):
- G1 teacher-forced argmax agreement against the quantized reference spec 01
  recorded;
- KLD per domain on the stored 2048- and 8192-token windows against the BF16
  top-64 references, BF16 KV and hq-e8-2b, beside spec 01's quantization-only
  numbers;
- MMLU-Pro proxy on the 281 questions, paired (McNemar) against the BF16
  checkpoint (73.7%), the study's recipe (73.0%) and the 27B (68.3% on the same
  questions).

**Speed** (hq-e8-2b, the defaults of spec 04):
- decode tok/s at 1, 2 and 3 lanes with 2K, 32K and 128K of context, warm
  expert cache, against the simulation (102 / 186 tok/s at 1 / 3 lanes) and the
  27B on the same bench;
- TTFT for 1K, 4K, 8K and 32K prompts, cold and warm expert cache, against the
  prefill bench (cold 4K: 2.65 s);
- the agent turn, 30K history plus 1K new, with reuse and with
  `--prompt-reuse off`, against spec 05's estimates (0.6-1.6 s, ~6.5 s);
- a three-agent swarm replay: per-turn TTFT, reuse sources, decode tok/s.

**Memory and I/O:**
- peak VRAM with the desktop, and every VRAM and host plan line;
- expert-cache hit rate per domain and PCIe bytes per decode token, live,
  against the trace replay of spec 03;
- n-gram hot-row hit rate and NVMe reads per second.

**Load:** wall time from `make start` to ready, with a cold page cache (after a
reboot or a 27B session) and a warm one. This is the input of the phase-2
switch estimate (14-15 s cold and 6-8 s warm, estimated).

**Write-up:** a finding in `docs/findings/` with an index row. Every number is
labelled as measured; every miss against a spec floor or an estimate gets a
cause or a follow-up issue.

## Testing Decisions

This ticket ships measurements, not code. Any harness change it needs (a bench
cell, a replay driver option) ships with its own test, as every change does.
The runs follow the GPU profile's rules: `make gpu-status` first, the shared GPU
lock, one run on the card at a time.

## Acceptance

1. The finding exists in `docs/findings/` with its index row, and covers quality, speed, memory/I-O and load as listed above.
2. Quality: G1, KLD per domain (2048 and 8192, BF16 and hq-e8-2b) and MMLU-Pro (paired against BF16, the study's recipe and the 27B) are reported.
3. Speed: decode at 1/2/3 lanes × 2K/32K/128K context, TTFT for 1K/4K/8K/32K cold and warm, the 30K+1K agent turn with and without reuse, and a three-agent swarm replay are reported against their estimates and the 27B.
4. Memory and I/O: peak VRAM with the desktop, the plan lines, live hit rate and PCIe bytes per token, n-gram hot-row hit rate and NVMe read rate.
5. Load time to ready, cold and warm page cache.
6. Every miss against a spec floor or an estimate is explained in the finding or has a follow-up issue linked from it.
7. `cargo test` passes workspace-wide.

## Out of Scope

- Fixing what the measurements find: follow-up issues.
- The phase-2 model switch itself (see `phase2-model-switch-notes.md`).
- Public benchmark suites and other engines.

## Further Notes

- Blocked by specs 04 and 05.
- The study's estimates live in
  `F:/ai/opencode/inference/.scratch/flash-next-compression-2026-10-03/`
  (`RISULTATI_3.md`, `review/PLACEMENT.md`, `review/MEMORY_PLAN.md`,
  `review/PREFILL_4K.md`).
