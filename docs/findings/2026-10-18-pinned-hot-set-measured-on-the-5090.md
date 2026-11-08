# A pinned hot set measured on the 5090: the host pool gives back the pinned bytes exactly and greedy text is unchanged; 1 GiB costs nothing measurable, 2 GiB 0.2-1.1% of three-lane decode, 4 GiB 4.3% of one-lane and 5.9% of three-lane decode

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: core / Flash-Next expert residency with a pinned hot set (`vram.pinned_expert_bytes`, 1, 2 and 4 GiB): the host plan's expert pool, the process's RAM, the expert cache's decode hit rate, decode tok/s and load time on the rtx5090 profile, and one load with MTP on
- Topic: flash-next
- Related: [GitHub #329](https://github.com/gpillon/ignis/issues/329) (item 12),
  [ADR 0055](../adr/0055-a-pinned-hot-set-is-opt-in.md),
  [A partially exclusive expert pool on the 5090](2026-10-18-partial-exclusive-expert-pool.md),
  [spec flash-next/03](../specs/flash-next/03-expert-residency.md)
- Superseded by: none
- Summary: Served on the 5090, a pinned expert hot set gives back exactly its bytes of host RAM, greedy text unchanged: 1 GiB costs nothing measurable, 2 GiB 0.1-0.2% of one-lane and 0.2-1.1% of three-lane decode, 4 GiB 4.3% and 5.9%.

## Question

The replay study predicted that pinning the calibration-hottest 1-2 GB of
experts in VRAM, outside the LRU and with no host copy, gives that RAM back
for under 1% of decode. Built behind `--vram-pinned-expert-bytes` (default 0),
does a served load on the RTX 5090 do that?

## Evidence

**Setup.** Branch `expert-pin` (ignis 0.7.0 + the pin), release build, the
served artifact `qwen3_8_flash_next_trellis_a25-v2.ninfer`, the rtx5090
profile (589,824 tokens, 3 lanes, yarn:4), MTP off,
`--model-idle-unload 0 --model-loop-guard off`. The machine had ~42-44 GB
available, short of the default host plan, so every launch ran the same lean
host lines: `--reuse-kv-host-pool-bytes 0 --reuse-retained-host 0
--kv-disk-bytes 0 --ngram-hot-bytes 0 --model-host-margin-bytes 1536M`. The
VRAM plan gave the expert cache 18,115,480,912 B (the study replayed 18.44 GB),
expected hit rate 0.596.

**Launches.** The baseline binary (main at `0a810c0b`) once; the pin binary at
`0`, `1G` and `2G` (GiB) three times each, in three rounds (0/1G/2G, 2G/1G/0,
0/1G/2G). Each launch: load, one unmeasured three-lane round of the prompts,
then one lane and three lanes, three reps each: greedy, `ignore_eos`,
thinking off, 320 tokens, three distinct prompts. Decode tok/s per request is
`(tokens − 1) / (last − first content chunk)`; the hit rate and the demand
stall are the `/metrics` decode counters around each phase. Reps 1-2 are
averaged (rep 0 warms the phase). The first 1 GiB launch was disturbed: its
reps ran 98-102 tok/s at one lane and 118-130 at three with unchanged hit
rates, and the later launches ran at the others' speed. Another agent's
kernel build (nvcc) ran on the CPU over it, 13:27-13:31, which that agent
reported afterwards; the launch ended at 13:31. It is left out of the speed
figures. Script and raw data: `.scratch/expert-pin/` in the
`expert-pin` worktree (`measure.py`, one JSON and one log per launch,
`summary.txt`), copied to `.scratch/orchestra-2026-10-18/expert-pin/session1/`
in the main checkout.

**What was pinned.**

| field | experts | projections | pinned bytes | of which K = 4 | load staging |
|---|---:|---:|---:|---:|---:|
| `1G` | 443 | 886 | 1,075,085,312 | 812 projections | 282,034,176 |
| `2G` | 912 | 1,824 | 2,149,367,808 | 1,467 projections | 496,091,136 |

The hottest experts are mostly K = 4: at 2 GiB the gate/up K4 LRU keeps 303
of its 1,075 slots and the down K4 449 of 1,144.

**RAM.** The host plan's `expert_pool` line and the process's working set
after the load:

| field | `expert_pool` | working set | less than at 0 |
|---|---:|---:|---:|
| baseline, `0` | 37,795,446,784 | 38.162 GB | — |
| `1G` | 36,720,361,472 | 37.087 GB | 1.075 GB |
| `2G` | 35,646,078,976 | 36.013 GB | 2.149 GB |

The same three launches of each agree to 1 MB. The pinned records stream
through `load_staging` while the experts load (the plan's residency-phase
peak is the pool plus it).

**Decode.**

| | 1 lane tok/s | hit | stall ms/round | 3 lanes tok/s | hit | stall ms/round |
|---|---:|---:|---:|---:|---:|---:|
| baseline (main) | 105.65 | 96.285% | 2.152 | 140.07 | 95.113% | 8.165 |
| `0` | 105.90 | 96.285% | 2.166 | 139.76 | 95.106% | 8.197 |
| `1G` | 106.17 | 96.274% | 2.168 | 140.19 | 95.075% | 8.254 |
| `2G` | 105.76 | 96.218% | 2.213 | 138.19 | 94.967% | 8.474 |

Per launch at three lanes: `0` 139.90 / 139.70 / 139.68, `1G` 141.03 /
139.35, `2G` 138.15 / 138.15 / 138.28.

**Text.** Every launch's 12 completions (320 tokens each) equal the baseline
binary's, at `0`, `1G` and `2G`.

**Load.** The residency phase took 17.5-18.3 s at `0`, 19.9-20.9 s at `1G` and
19.9-20.4 s at `2G`, with the page cache warm.

**Second session: 4 GiB (15:34-15:46, for the owner's choice between `2G`
and `4G`).** The same script (`measure.py`, copied unchanged), the same lean
host lines, prompts, reps and averaging, on the branch's binary with the
review's fixes (the pin keeps the slots of the widest decode step, which with
MTP off is the lanes as before; a pin it cannot reach is refused; a kernel
variable renamed). `0`, `2G` and `4G` three launches each, in three rounds
(4G/2G/0, 0/2G/4G, 4G/2G/0), the control at an explicit
`--vram-pinned-expert-bytes 0`, since the rtx5090 profile now sets `2G`. The
VRAM plan gave the cache 18,185,223,504 B (70 MB more than the first session:
the desktop held less VRAM); 48.0-48.1 GB was available before each launch,
no build was running when any launch started (checked before each, logged in
`session2.log` for all but the first), and the GPU lock was held in timing
mode, which asks other agents to defer builds. Script, raw data and
`summary-session2.txt` (`summarize.py` reproduces the first session's
figures from its JSON): `.scratch/expert-pin-4g/` in the `expert-pin`
worktree, copied to `.scratch/orchestra-2026-10-18/expert-pin/session2-4g/`
in the main checkout.

| field | experts | projections | pinned bytes | of which K = 4 | load staging |
|---|---:|---:|---:|---:|---:|
| `4G` | 1,995 | 3,990 | 4,296,007,680 | 1,977 projections | 867,164,160 |

At 4 GiB the gate/up K4 LRU keeps 78 of its 1,077 slots, the minimum three
lanes' decode step needs (3 x (10 + 16)), and the down K4 LRU 168 of 1,146;
the walk then pins K = 3 experts (972 gate/up and 877 down projections). The
host plan's `expert_pool` was 33,499,439,104 B and the working set 33.866 GB:
4,296,007,680 B and 4.296 GB less than at `0` (37,795,446,784 B, 38.162 GB).

| session 2 | 1 lane tok/s | hit | stall ms/round | 3 lanes tok/s | hit | stall ms/round |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 105.95 | 96.331% | 2.142 | 139.78 | 95.163% | 8.092 |
| `2G` | 105.77 | 96.275% | 2.186 | 139.54 | 95.041% | 8.352 |
| `4G` | 101.35 | 95.797% | 2.594 | 131.48 | 94.417% | 9.717 |

Per launch at three lanes: `0` 140.04 / 139.26 / 140.04, `2G` 138.97 /
138.96 / 140.69, `4G` 130.54 / 130.93 / 132.98; at one lane `4G` 100.94 /
101.48 / 101.63 against `0`'s 105.90-105.99. Every launch's 12 completions
equal the baseline binary's. The residency phase took 17.4-17.5 s at `0`,
19.7-19.9 s at `2G` and 18.9-19.1 s at `4G`.

**MTP on (second review, 2026-10-18 18:26), acceptance only.** One launch
of the review's binary, the profile's `2G`, the same lean host lines plus
`--spec-backend mtp --spec-draft-tokens 3`; one unmeasured three-lane round,
then one rep at one lane and one at three, 320 greedy tokens. That binary's
split gives every class the widest decode step's slots first, here an 8-row
verify round's 208 where the lanes gave 78 (with MTP off it splits as
before, so the figures above stand). The VRAM plan gave the cache
16,477,948,752 B (the MTP head's lines come out of it); the load was
accepted and pinned 912 experts, 2,149,367,808 B, the gate/up K4 LRU keeping
352 slots and the down K4 484, and the `expert_pool` line was
35,646,078,976 B as with MTP off. All 8 requests ended `length` (the
measured ones at 320 tokens), with no `ignis.runtime.leaf_error` and no
ERROR line. Log and JSON: `.scratch/pin-review2/R2-mtp-pin2G.{log,json}` in
the `expert-pin` worktree, copied to
`.scratch/orchestra-2026-10-18/expert-pin/review2-mtp/` in the main checkout.

## Finding

**Observed.** The pin gives back exactly its bytes: the plan's expert pool
and the process's working set drop by 1,075,085,312 B at `1G` and
2,149,367,808 B at `2G`. Greedy text is unchanged, so the pinned slots hold the
right records. With the field at 0 the binary decodes as main does (within
0.3%, hit rate equal to 0.007 points).

**Observed.** `1G` costs nothing measurable: tok/s within ±0.3% of `0` at one
and three lanes, the hit rate 0.01-0.03 points lower. `2G` costs 0.1% at one
lane and 1.1% at three (138.2 against 139.8 tok/s, every launch below every
`0` launch), the hit rate 0.07 and 0.14 points lower, and the three-lane
demand stall 0.28 ms per round higher (8.47 against 8.20).

**Compared with the study.** The study replayed 2.0 GB at +0.6-0.9% of a
three-lane round and 0.2-0.3% of a one-lane one, and 1.0 GB at 0.2% and 0.
At 2 GiB (2.15 GB) the card measures 1.1% and 0.1%: the same order, the
three-lane cost at the top of the study's bracket. Its hit-rate deltas at
2 GB (−0.04 and −0.08 points) are close to these (−0.07 and −0.14).

**Observed (second session).** `4G` (4.30 GB pinned) costs 4.3% of one-lane
and 5.9% of three-lane decode (101.35 against 105.95, 131.48 against 139.78
tok/s; every `4G` launch below every `0` launch at both lane counts), the hit
rate 0.53 and 0.75 points lower and the demand stall 0.45 and 1.63 ms per
round higher. Its RAM comes back exactly (4.296 GB) and greedy text is
unchanged. In the same session `2G` measured 0.2% at both lane counts, two of
its three-lane launches 0.6% below `0`'s mean and the third above it: with
the first session's 1.1%, `2G` costs 0.2-1.1% of three-lane decode.

**Compared with the study at 4 GB.** It predicted 3.8-5.3% of a three-lane
round and 1.6-2.1% of a one-lane one for 4.00 GB (1,833 experts). The card
measures 5.9% and 4.3%: above the bracket's top at three lanes, and twice it
at one. Part of the gap is the larger pin (4.30 GB, 1,995 experts) on a cost
curve that rises steeply here (the study's demand +1.0-1.6% at 2 GB, +8-10% at
4 GB) (inferred); the rest is not separated here.

## Implications

- `--vram-pinned-expert-bytes 1G` is RAM for free on this host; `2G` costs
  0.2-1.1% of three-lane decode for 2.149 GB; `4G` 4.3% of one-lane and 5.9%
  of three-lane decode for 4.296 GB, about 0.7 GB of RAM per % of three-lane
  decode against 2-11 GB per % at `2G`. Past 2 GiB the K = 4 LRU pools are
  what the pin eats (72% of the gate/up K4 slots at 2 GiB, all but the
  three-lane minimum at 4 GiB), and the cost climbs as the study said, faster
  than it predicted.
- Loading with a pin takes ~2 s longer, from reading the pinned records apart
  and the pool in runs around them.
- The field's default stays 0; the owner accepted ADR 0055 with the rtx5090
  profile at `2G` for Flash-Next, and chooses between `2G` and `4G` from the
  second session.

## Limits and unknowns

- One prompt set, three prompts of 320 tokens, repeated: decode only, no
  prefill and no long session. Three launches per setting; the first `1G`
  launch was set aside as disturbed.
- Lean host lines (no n-gram hot rows, no KV-RAM arena, no retained host slots,
  a 1.5 GiB margin), the same on every launch, because the machine had too
  little RAM for the default plan that day. N-gram rows came from the page
  cache or the drive.
- The hit rates are this workload's, higher than the replay's teacher-forced
  chunks; the deltas are what compares.
- Only 1, 2 and 4 GiB were measured, at one cache size (18.12-18.19 GB), and
  with MTP off; with MTP on one `2G` load was checked (accepted, every request
  `length`), not timed. The two sessions ran about two hours apart on two binaries; their
  `0` rows agree within 0.05%, so the rows compare within a session.

## Follow-ups

- None needed for the feature. If a larger pin is ever wanted, the study's
  untried options (a split that re-plans the remaining pools, ranking by
  traffic per byte) are where to start.
