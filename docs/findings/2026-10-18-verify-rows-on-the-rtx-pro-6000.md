# On the RTX PRO 6000 a Flash-Next verify round past 8 rows costs more than its drafts return, so the `rtxpro6000` profile keeps 8

- Kind: experiment
- Status: current
- Observed: 2026-10-18
- Last verified: 2026-10-18
- Scope: serving / the `rtxpro6000` profile's Flash-Next MTP row budget (`spec.draft_rows`, ADR 0056), the verify round's time by width on a card that holds every expert in VRAM
- Related: https://github.com/gpillon/ignis/issues/329 (item 5), [Flash-Next verify rows past 8](2026-10-18-flash-next-verify-rows-past-8.md), [the rtxpro6000 profile measured](2026-10-18-rtxpro6000-profile-measured.md), [spec flash-next/07](../specs/flash-next/07-mtp-speculation.md), `.scratch/rows-card/`
- Superseded by: none

## Question

A verify round carries at most 8 rows (lanes x (drafts + 1)), so the
`rtxpro6000` profile's MTP drafts nothing from five lanes on: 0.97x
speculation off at 8 lanes. Branch `verify-rows` raises the budget to 32
(`spec.draft_rows`). On the RTX PRO 6000, where decode is not bound by PCIe,
does a wider round pay at 4 and 8 lanes and for simulated agents, and what
does it cost in VRAM?

## Evidence

**Setup.**
- Card and host: one RTX PRO 6000 Blackwell Server Edition (96 GB) on an
  8-thread host with 64 GiB of RAM, in a Kubernetes cluster.
- Server: branch `verify-rows` at 33c15274 (`git archive`), built in the
  cluster's dev pod (`cargo build --release --target
  x86_64-unknown-linux-gnu -p ignis-server --features ignis-server/cuda`,
  62 kernel objects rebuilt), binary sha256 `72db0d86…62869`. It ran inside
  the develop image of its base 0a810c0b
  (`sha256:63a38192c3d43953e86f5176b7cd3166f5d0a464e056a3f709ea99d5f250661f`).
- Flags: `--profile rtxpro6000 --model-loop-guard off --model-idle-unload
  0`, n-gram `auto` (the whole table, `ignis_ngram_reads_total` 0 in every
  cell), pod limit 56 GiB.
- Knobs: changed with `PATCH /v1/config`
  (`{"spec":{"qwen38flashnext":{"draft_rows":N}}}`), which reloads in place.
  Each reload's `ignis.runtime.flash_next_plan` names the budget, e.g.
  `mtp draft tokens 3 rows 16 (k at 1/2/3 lanes: 3/3/3)`.

**Client.** `sweep.py` from the profile campaign, in a CPU pod of the same
namespace:
- decode cells: N lanes fired together, greedy, `ignore_eos`, 512 tokens,
  8 code or 8 prose prompts, 3 reps, the median of the aggregate over the
  window where all N decode;
- agents: `--type agents --turns 8 --seed 7`, the earlier campaign's
  schedule (each turn a seeded 96-768 tokens, a seeded tool pause and
  result);
- a one-lane warm-up after every reload.

The output past 8 rows drifts from speculation off at near-ties (the
`verify-rows` finding), so throughput is compared, not text.

**8 lanes** (aggregate tok/s; acceptance = accepted / drafted; window = the
drafts a lane verifies at 8 lanes):

| speculation | rows | window | code | prose | x off code / prose | acceptance code / prose |
|---|---:|---:|---:|---:|---|---|
| off | -- | -- | 426.7 | 423.9 | 1 / 1 | -- |
| MTP 3 | 8 | 0 | 414.6 | 413.6 | 0.97 / 0.98 | no drafts |
| MTP 3 | 16 | 1 | 231.9 | 212.9 | 0.54 / 0.50 | 0.948 / 0.795 |
| MTP 1 | 16 | 1 | 233.0 | 211.6 | 0.55 / 0.50 | 0.949 / 0.791 |
| MTP 3 | 24 | 2 | 299.6 | 247.0 | 0.70 / 0.58 | 0.900 / 0.648 |
| MTP 2 | 24 | 2 | 299.7 | 244.7 | 0.70 / 0.58 | 0.903 / 0.651 |
| MTP 3 | 32 | 3 | 350.3 | 261.6 | 0.82 / 0.62 | 0.864 / 0.553 |

A cell's three reps span at most 4.1% (6.1% for 4 lanes code at rows 16,
208-221). Rows 8, re-measured after the other configurations about 40
minutes later, ran 413.8 / 413.4.

**4 lanes:**

| speculation | rows | window | code | prose | x off code / prose | acceptance code / prose |
|---|---:|---:|---:|---:|---|---|
| off | -- | -- | 331.8 | 329.3 | 1 / 1 | -- |
| MTP 3 | 8 | 1 | 404.0 | 369.3 | 1.22 / 1.12 | 0.941 / 0.774 |
| MTP 3 | 16 | 3 | 208.9 | 155.6 | 0.63 / 0.47 | 0.844 / 0.521 |

**One lane** (the warm-up cell, code, 256 tokens, 2 reps): MTP 3 ran
241.6-241.8 tok/s in rep 1 and 211.6-211.8 in rep 2 at rows 8, 16, 24 and
32 alike (a lane alone verifies 3 drafts at every budget); MTP 2 228.4 /
212.3, MTP 1 188.8 / 184.7, off 128.8 / 128.7. In rep 2 MTP 2 is at
least MTP 3 (212.3 against 211.7), so this cell does not separate 3 from 2.

**Simulated agents** (tokens over the makespan; the same work in every
run: 10,240 tokens in 32 turns at 4 agents, 19,136 in 64 at 8; no turn
errored):

| speculation | rows | 4 agents | x off | 8 agents | x off | median turn 4 / 8 |
|---|---:|---:|---:|---:|---:|---|
| off | -- | 149.5 (68.5 s) | 1 | 213.8 (89.5 s) | 1 | 3.79 / 5.37 s |
| MTP 3 | 8 | 185.3 (55.3 s) | 1.24 | 224.2 (85.3 s) | 1.05 | 2.58 / 5.43 s |
| MTP 3 | 16 | 158.3 (64.7 s) | 1.06 | 147.2 (130.0 s) | 0.69 | 3.93 / 8.71 s |
| MTP 3 | 32 | 145.6 (70.3 s) | 0.97 | 174.0 (110.0 s) | 0.81 | 4.41 / 6.89 s |

Rows 8 against off matches the earlier campaign on another build (1.21x and
1.05x).

**The round's time by width** (derived: a lane's tokens per round, 1 +
accepted / rounds from the speculative counters, over the per-lane rate):

| rows in a round | cells | ms a round |
|---:|---|---:|
| 1 | off, 1 lane | 7.8 |
| 4 | off, 4 lanes; MTP 3, 1 lane | 12.2; 13.1 |
| 8 | off, 8 lanes; MTP 3 rows 8, 8 lanes and 4 lanes | 19.0; 19.4-19.6 |
| 16 | 8 lanes x 2 (rows 16); 4 lanes x 4 (rows 16) | 66.6-67.7; 63.6-64.8 |
| 24 | 8 lanes x 3 | 73.6-74.6 |
| 32 | 8 lanes x 4 | 79.2-80.2 |

For 8 lanes to break even with the 8-row step (19.6 ms a token a lane), a
round must take at most its tokens per lane x 19.6 ms. That is 38.2 / 35.2
ms at 16 rows (code / prose), 54.9 / 45.2 at 24 and 69.6 / 51.8 at 32,
against 67-80 ms measured.

**VRAM** (the plan event at each reload; MB = 10^6 bytes; the expert cache
was the whole catalog, 37,795 MB, in every load):

| speculation | rows | verify round | program (workspace + sampling + decode graph) | lines over rows 8 | KV pool tokens |
|---|---:|---:|---:|---:|---:|
| MTP 3 | 8 | 16.4 MB | 2,392.0 MB | -- | 11,442,240 |
| MTP 3 | 16 | 27.1 | 2,430.4 | +49.2 MB | 11,437,952 (budget +29.4 MB) |
| MTP 1 | 16 | 25.0 | 2,430.4 | +47.0 | 11,432,000 |
| MTP 3 | 24 | 37.9 | 2,469.1 | +98.6 | 11,421,184 (budget +2.1 MB) |
| MTP 2 | 24 | 36.8 | 2,469.1 | +97.5 | 11,420,928 |
| MTP 3 | 32 | 48.6 | 2,507.3 | +147.6 | 11,409,984 |
| off | -- | 0 | 2,392.0 | -- | 12,647,232 |

The budget was 100,127.7 MB in the reloads unless marked (the cold load,
at rows 8, had 100,362.2 MB and 11,493,504 tokens). The KV pool takes the
difference at 4,576 bytes a pool token: 32 rows cost 32,256 tokens against
8 at the same budget (0.28% of the pool), 24 rows ~21,500, 16 rows ~10,700.
The plan event's +147.6 MB at 32 rows matches the `verify-rows` finding's
host-only plan figure (+147.5 MB) within 0.1 MB.

**Errors.** 856 requests over one cold load and seven reloads, every one
`finish_reason: length` at its `max_tokens`; no `ignis.runtime.leaf_error`,
no WARN or ERROR line, no restart.

## Finding

Past 8 rows a verify round on this card takes 64-68 ms at 16 rows, 74 ms
at 24 and 80 ms at 32, against 19.5 ms at 8. No acceptance comes near
paying for that: the drafts would need the round under 35-38 ms at 16 rows.
So every wider budget loses to rows 8 wherever it changes the round:
- 8 lanes: 0.51-0.84x;
- 4 lanes: 0.42-0.52x;
- the simulated agents: 0.66-0.85x.

A lane alone, which verifies 3 drafts at any budget, is unchanged. The cost
follows the window, not the draft count: MTP 1 at 16 rows and MTP 2 at 24
match MTP 3 at the same rows within 1%. The VRAM is small (47-148 MB, at
most 0.28% of the KV pool, as planned); time is what the wider round costs.

Inferred, not profiled: the round's time is one jump past 8 rows (+44-48 ms
from 8 to 16) and then 5.3-7.1 ms per further 8 rows (each 8-lane cell from
16 to 24 to 32 rows). Up to 8 it climbs ~1.5-1.7 ms a
row.
- The later steps are of the size one more MoE launch of 8 tokens would
  add (the route a wider round keeps), though an 8-token MoE pass was not
  timed on its own.
- The jump is fixed. It sits where the `verify-rows` finding lists the
  routes a call of 9 to 32 tokens switches to: the linears' tensor-core and
  WMMA routes, the hyper-connection prefill route, the grouped GEMVs split
  into separate linears.

## Implications

- **Recommendation for the `rtxpro6000` profile: keep `spec.draft_rows`
  unset (0 = 8 rows) and `draft_tokens: 3`.** Rows 8 is the measured best
  at one, four and eight lanes and for both agent counts. The draft count
  has no reason to move from this run: at rows 8 it matters only at one or
  two lanes, where the earlier campaign found 3 and 4 tied over code and
  prose and ahead of 2 and 5. This run's one-lane warm-up (code only, 256
  tokens, 2 reps) does not separate 3 from 2: 241.6 / 228.4 / 188.8 tok/s
  for 3 / 2 / 1 in rep 1, 211.7 / 212.3 / 184.7 in rep 2.
- The knob is worth revisiting on this card only once a 16-row round runs
  under ~35 ms. Either of the `verify-rows` finding's options 1 (launches of
  8 rows on every route that switches at 8) or 2 (decode-phase GEMVs wider
  than 8) targets exactly the routes that the inferred jump points to.
- At 8 lanes, rows 8 runs 2-3% below speculation off with no drafts at all,
  as the earlier campaign saw (0.97-0.98x). The cause was not examined.

## Limits and unknowns

- One card, one build, greedy only, 512 tokens; 8 code and 8 prose
  prompts; one agents schedule (seed 7). Sampled decoding was not run.
- 4 lanes were measured at rows 8 and 16 only. At 24 and 32 a 4-lane round
  is the same 16 rows (window 3), so it is expected to match rows 16, but
  this was not measured.
- Which route carries the 44-48 ms jump is inferred from the shape of the
  curve, not from a profile.
- The text was not compared (the drift past 8 rows is the `verify-rows`
  finding's subject).

## Follow-ups

- A profile (nsys or ncu) of one 16-row round against one 8-row round on
  this card, to name the route behind the jump, before choosing between the
  `verify-rows` finding's options 1 and 2.
- After either lands: the 8-lane and 4-lane cells and the agents above, at
  rows 8 against 16 and 32.
