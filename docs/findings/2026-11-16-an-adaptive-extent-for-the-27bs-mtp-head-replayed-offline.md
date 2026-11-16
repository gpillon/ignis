# An adaptive extent for the 27B's MTP head, replayed offline: the head's confidence commits 7.5% more than the best fixed window on the 16 GB leg only where a round stops its chain and narrows its verify, 3.3% where only the chain stops, nothing in today's round, and the acceptance history 1.6%

- Kind: experiment
- Status: current
- Observed: 2026-11-16
- Last verified: 2026-11-16
- Scope: speculative decoding / the 27B's MTP head (spec #618, step 3): an extent that adapts, replayed offline
  over #628's round records of #267's 200 agent turns and its RTX 5090 cost table, on the 16 GB artifact in a
  13 GiB budget (2 lanes) and the NVFP4 artifact (8 lanes), at one lane and the load's, 4,096 and 65,536 tokens
  deep; the head's confidence and the lane's acceptance history against the best fixed draft window; how a
  round of several lanes pays for extents that differ
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/629,
  https://github.com/gpillon/ignis/issues/618,
  https://github.com/gpillon/ignis/issues/628,
  https://github.com/gpillon/ignis/issues/630,
  [the head measured on the RTX 5090](2026-11-15-the-27bs-mtp-head-measured-on-the-rtx-5090.md) (the records,
  the cost table, the default window),
  [ADR 0095](../adr/0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md),
  [DFlash2 sampled acceptance](2026-09-24-dflash2-sampled-acceptance.md) (lever 1, DFlash2's adaptive extent
  at eight lanes)
- Superseded by: none
- Summary: RTX 5090 costs, 200 agent turns, 16 GB: a confidence stop (chain probability < 0.56) commits 7.5% over the best fixed window if stopping the chain and narrowing the verify cost under 0.29-0.43 ms a round; 3.3% if only the chain stops.

## Question

Spec #618's step 3: the head stops drafting where another draft is not worth its cost, built only if the
measurement says it commits at least 5% more tokens per second than the best fixed draft window, on the 16 GB
leg. #628 measured the head at fixed windows (the round records over #267's 200 agent turns, the round's cost
by window, lanes and context) and kept the default at 7. Two policies are candidates, each reading only its
lane's own signals and no clock:

- **the head's confidence**: a lane's chain stops when the probability of its drafts falls under a threshold;
- **the lane's acceptance history** at each draft position, the reference engine's approach without its
  clock: the costs are the measured table's.

Replayed offline over the same records and priced at the measured costs, against the best fixed window: does
either clear the gate, by which policy and threshold, and what does a round of several lanes do with extents
that differ?

## Evidence

**Nothing measured anew.** The inputs are #628's, read as they are: its round records at a window of 7 on each
leg (54,846 rounds on the 16 GB artifact, 54,647 on NVFP4; per round the head's drafts, its probability of
each, and how many the recording accepted), its direct replays at windows 4, 5 and 6 of jobs 0 to 21, and its
cost table (the head's round and the verify substrate's alone at each window 1 to 7 and the plain round, at
one lane and the load's, 4,096 and 65,536 tokens deep; a cell the mean of two launches' medians), with each
launch's own table beside it; #267's manifest and traffic weights. No GPU run, no engine change. The analysis
is `tools/mtp-replay/extent.py`, tracked with its CPU check `test_extent.py` (12 tests, runnable without
pytest); its two runs, the commands (`run.sh`) and the plan reading below (`lanes_vram.py`) are in the main
checkout's `.scratch/618/629/` (untracked).

**A round under a policy.** Every round of the window-7 walk keeps its anchor. A policy gives each lane-round
its chain steps and its extent `e`, and the round commits `min(accepted, e) + 1`: #628's derivation of windows
1 to 6. It reads a policy that cuts accepted drafts low, since the walk that policy would make starts its next
round inside the run it cut, where acceptance is higher.

**The policies**, each also naming the load's draft window (4 to 7), which caps its extent:
- *the fixed window*, the baseline;
- *the oracle*: each round at the extent its accepted run needed; a ceiling no policy reaches;
- *the head's confidence*: a lane's chain stops at the first draft whose signal falls under a threshold, the
  signal the head's probability of that draft or the chain's product of them (the head's own chance that the
  run reaches it); the round keeps the draft under the threshold, or verifies only those before it. The
  threshold runs 0.00 to 0.99 by 0.01, then 0.991 to 0.999;
- *the lane's acceptance history*: per lane and draft position `j`, two counts decayed by `d` at each
  observation, of the rounds whose run reached `j - 1` and tested `j` and of those that accepted it; the
  estimate of accepting `j` is `(accepted + m p) / (reached + m)`, `p` a prior and `m` its weight; the chance
  the run reaches `e` their product; each round the extent with the most expected tokens over its price. A
  lane learns only the positions its extent tested and forgets at the start of a turn. The reference engine
  keeps two moving averages a position and one width for the batch, calibrated by a clock; this is ignis's own
  design of the same idea, priced by the table: `d` 0.3 to 0.995, `p` 0.5 to 0.9, `m` 1 or 4 (16 and 64,
  checked after, read lower: 1.006 at best on the 16 GB leg).

**How a round pays**, from the table (C the head's round at window w, V the substrate's, H = C - V the head's
part, a chain step the table's slope of H):

| pricing (the tool's name) | the round costs | what it needs |
|---|---|---|
| a width the lanes share (`shapes`) | V(the lanes' widest extent) + H(chain steps), every lane at that width | a graph per window and lane count; for a confidence stop, the chain stopped on the device and the verify's width chosen after it, inside the round |
| the chain stops (`chain`) | C(window) - (window - the widest chain) x a chain step; each lane verifies its own extent, the verify keeps the window | today's graph per lane count at the window, with a chain that skips its steps past the lanes' widest extent, which the leaf does not have |
| each lane its own width (`ragged`) | V(the lanes' mean extent) + H(the widest chain) | more captured shapes, a lane's own width in one round: no leaf has it, an estimate |

Today's round is none of these: it runs its window's chain steps and verify columns whatever its lanes'
extents (#621, #628), so a shorter extent saves nothing and commits fewer tokens; under it the best a policy
does is never to stop, window 7, 0.991 of each cell's best fixed window on the 16 GB leg.

At one lane the shared width and a lane's own width price alike. The table holds neither what a skipped step
still costs nor what choosing a width inside a round costs: every price is the measured round less what is not
run, a ceiling. A
chain step is 0.77 to 1.08 ms on the 16 GB leg, a verify column 0.20 to 0.55 ms.

**Lanes and rates.** At the load's lanes the lanes walk the rounds from as many job starts, each lane every
round once, and a lane-round takes its whole round's time. Committed tokens per second (a lane's) are, per
traffic cell, the tokens over the time of its lane-rounds, the cells pooled by #267's weights as #628 pooled
tokens a round; for a fixed window that is #628's T(w) / C(w). The unweighted figure is all tokens over all
time.

**The comparison.** Each cell's baseline is its best fixed window as #628 rated it: windows 4 to 6 corrected
by the subsets' direct over derived ratio (1.034, 1.028, 1.018 on 16 GB; 1.035, 1.017, 1.017 on NVFP4), the
others as derived. A policy's figure is its derivation, uncorrected: the conservative side of the pair.
Beside it, a model: the policy's rate times `1 + kappa x` the fraction of its rounds that cut an accepted run,
kappa fitted through zero on the subsets' three ratios against the fraction of rounds each window cuts (0.092
on 16 GB, 0.081 on NVFP4, reading the ratios within 0.9%). The rule, #628's: the geometric mean over a leg's
four cells of the policy's rate over the cell's best fixed window, with one config for the whole leg, as a
configuration field would have it. The gate is 1.05 on the 16 GB leg.

### The 16 GB leg

Each cell's best fixed window, committed tokens per second a lane: 222.3 (window 5, one lane, 4K), 194.0 (7,
two lanes, 4K), 179.1 (6, one lane, 64K), 141.6 (6, two lanes, 64K), #628's figures.

| policy, its leg config | pricing | gain (geometric mean) | unweighted | model-corrected | per cell: 1 x 4K, 2 x 4K, 1 x 64K, 2 x 64K |
|---|---|---:|---:|---:|---|
| **confidence**: chain probability < 0.56, keep, window 7 | shared width | **1.075** | 1.068 | 1.084 | 1.084, 1.066, 1.100, 1.049 |
| confidence: chain probability < 0.52, keep, window 7 | own width | 1.066 | 1.058 | 1.076 | 1.085, 1.050, 1.101, 1.028 |
| confidence: chain probability < 0.38, window 7 | chain stops | **1.033** | 1.026 | 1.039 | 1.058, 1.012, 1.070, 0.996 |
| history: `d` 0.8, `p` 0.8, `m` 4, window 7 | shared width | 1.016 | 1.009 | 1.024 | 1.013, 1.016, 1.028, 1.004 |
| history: the same | chain stops | 0.997 | 0.988 | 1.002 | 1.000, 0.995, 1.014, 0.979 |
| history: `p` 0.9 | own width | 0.998 | 0.989 | 1.001 | 1.004, 0.993, 1.015, 0.979 |
| oracle | shared width | 1.225 | 1.217 | | 1.276, 1.180, 1.292, 1.157 |
| oracle | chain stops | 1.156 | 1.147 | | 1.203, 1.116, 1.209, 1.099 |
| oracle | own width | 1.244 | 1.236 | | 1.276, 1.222, 1.292, 1.189 |

- **The two numbers**: over the four cells (geometric mean, tokens per second a lane), the chosen confidence
  policy under a shared width commits 195.4 where each cell's best fixed window commits 181.9 (192.4 against
  180.2 unweighted).
- **The sweep** (leg gain; chain probability, keeping the draft, window 7; threshold 0 is window 7):

| threshold | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.56 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared width | 0.991 | 1.011 | 1.032 | 1.050 | 1.064 | 1.072 | **1.075** | 1.074 | 1.070 | 1.060 | 1.039 | 1.018 | 0.969 |
| the chain stops | 0.991 | 1.008 | 1.022 | 1.031 | 1.033 | 1.028 | 1.023 | 1.017 | 1.001 | 0.981 | 0.948 | 0.921 | 0.865 |

  and the history's decay (shared width, `p` 0.8, `m` 4): 1.004 at 0.3, 1.010 at 0.5, 1.014 at 0.7, **1.016**
  at 0.8, 1.015 at 0.9, 1.013 at 0.95, 1.010 at 0.98, 1.006 at 0.995.
- **The chosen confidence policy** (shared width): 241.1, 206.8, 197.0 and 148.5 tokens per second a lane; a mean
  extent of 3.82 at one lane and 5.14 at two (the widest of the two lanes'); 11.8% of its one-lane rounds cut
  an accepted run (6.4% at two lanes). The threshold is inside its grid: the cells' own bests are 0.50 at one
  lane and 0.66 to 0.69 at two, together 0.3% more than the one threshold. The chain's product beats each
  draft's own probability (1.069 at 0.69), keeping the draft under the threshold beats dropping it (1.053), and
  a window of 6 under it reads 1.039.
- **The break-even**: the overhead every round could carry before the leg's gain falls to 1.05, 0.433 ms under
  a shared width (a cell's own: 0.49, 0.28, 0.87 and none at two lanes 64K, already under 1.05), 0.274 ms under
  a lane's own width. When only the chain stops there is none: the gain is under the gate before any.
- **The history's** decay of 0.8 (about five observations) and prior are inside their grids; whatever the
  pricing it moves the extent from round to round by what the last rounds did, which tells little about the
  next: its best is within 1.6% of the best fixed window.
- **Robustness.** Chosen on the even jobs and read on the odd, and the reverse: 1.074 and 1.076 (confidence,
  shared width), 1.033 and 1.035 (the chain stops), 1.014 and 1.018 (history, shared width). The same sweep on each
  launch's own table and on the table's least-squares lines over windows 1 to 7 (the trellis's verify curve
  has kinks: its substrate at window 5 costs more than at 6 and 7):

| cost table | confidence, shared width (threshold, break-even) | confidence, the chain stops | history, shared width |
|---|---:|---:|---:|
| the two launches' mean | 1.075 (0.56, 0.43 ms) | 1.033 | 1.016 |
| the first launch | 1.074 (0.56, 0.42 ms) | 1.035 | 1.017 |
| the second launch | 1.066 (0.56, 0.29 ms) | 1.023 | 1.007 |
| least-squares lines | 1.068 (0.58, 0.32 ms) | 1.016 | 1.008 |

- **The walk.** The chosen configs at one lane 4K, capped at each subset's window w, rated on the direct walk at
  w and on the window-7 walk of the same jobs: the confidence policy reads 1.030, 1.022 and 1.011 higher on the
  direct walk at windows 4, 5 and 6, the history 1.034, 1.026 and 1.011, a fixed window 1.034, 1.028 and 1.018.
  The derivation reads a policy that cuts low by about what it reads a fixed window that cuts: the size the
  model-corrected column gives.

### The NVFP4 leg (beside it)

Best fixed: 184.4 (window 6, one lane, 4K), 132.1 (4, eight lanes, 4K), 156.9 (6, one lane, 64K), 64.3 (5,
eight lanes, 64K).

| policy, its leg config | pricing | gain | unweighted | model-corrected | per cell |
|---|---|---:|---:|---:|---|
| confidence: chain probability < 0.68, keep, window 7 | shared width | 1.030 | 1.020 | 1.037 | 1.094, 0.965, 1.088, 0.978 |
| confidence: < 0.43 | own width | 1.026 | 1.017 | 1.033 | 1.088, 0.968, 1.085, 0.971 |
| confidence: < 0.30, window 6 | chain stops | 0.998 | 0.987 | 1.016 | 1.048, 0.943, 1.045, 0.959 |
| history: `d` 0.9, `p` 0.9, `m` 4, window 7 | shared width | 0.999 | 0.989 | 1.011 | 1.009, 0.986, 1.013, 0.989 |
| oracle | shared width | 1.131 | 1.120 | | 1.303, 0.978, 1.294, 0.991 |

At one lane the confidence stop gains 9% on NVFP4 too; at eight lanes the widest of eight lanes' extents is
near the window (a mean of 6.72), so a width the lanes share saves next to nothing, and even the oracle at a
shared width falls short of the best fixed window there (DFlash2's lever 1 found the same at eight lanes: +2-3% for
its oracle at agent contexts).

### Several lanes: what each option holds, and what it does to a lane's text

A captured graph's own memory is in no line of the VRAM plan; it is part of the process's residual (WDDM's
count after load less every other line, #628's served starts). MTP's residual less window 0's, over the
verify graphs MTP adds (one per lane count at window 7), is 13.1 MB for 2 graphs on the 16 GB load (one start:
its second read 42.7 MB under its other lines) and 43.2 MB for 8 on NVFP4: 6.6 and 5.4 MB a graph, 4.8 and 4.7
KB a graph node. Capture took about 22 and 17 ms a graph.

| option | captured verify graphs, 16 GB (2 lanes) | NVFP4 (8 lanes) | 16 GB gain (confidence) | a lane's text |
|---|---|---|---:|---|
| the chain stops, the verify keeps the window (today's graphs) | 2 | 8 | 1.033 | as today: its verify keeps the load's width whatever its neighbours' extents |
| a width the lanes share | 14 (12 more): at most 78.7 MB, about 8,000 pool tokens, 0.26 s more capture | 56 (48 more): at most 258.9 MB, about 26,400 pool tokens, 0.82 s | 1.075 | its width follows its neighbours' confidence |
| more captured shapes: each lane its own width | 56 exact shapes (7 + 49): about 354 MB | 7^8, about 5.8 million: not capturable | 1.066 (estimate) | the round's columns follow its neighbours' |

- The plan's residual line (124,474,800 bytes) covers today's graphs (45.7 MB at 2 lanes, 110 MB at 8); 12 more
  graphs would bring the 16 GB load to its edge (124.4 MB) and 48 more take NVFP4's past it (about 369 MB), so
  more shapes need a line of their own. In the 13 GiB budget the 16 GB load's pool would fall from 232,320 to
  about 224,300 tokens, above the 196,608 its 131,072 context and the spare reservation ask: the context would
  stay.
- A lane alone in its rounds walks the same extents and writes the same text under every option: its signal is
  its own. With neighbours, ADR 0095 measured that a verify round at another extent, and the head's call at
  another width (the W8 small-T route splits its K by the width), part the text at near ties. When only the
  chain stops, a lane's round keeps the shape it has today; under a shared width or a lane's own, its width, or
  the round's, depends on what its neighbours draft, beyond the lane count it already depends on (an inference
  from ADR 0095's measurements, not run here).

### Same records, same extents

Two runs of the whole sweep on both legs, in separate processes, wrote the same bytes: the 16 GB figures
`a0267f29005853fa7774f23776e40a23c8f09c9638d40f4dae1f1bfd52ceb46c`, NVFP4's
`c1d4aa74d0cec07f3490a39c0cdd1c3b98c91bd33a2a73bef82fec197be2b1c2` (SHA-256 of each JSON), and in them every
config's extents, lane-round by lane-round, by their own SHA-256 (the chosen 16 GB confidence config's
`69469d60c8ae...` at one lane, `ae79756a0a6d...` at two). `test_extent.py` holds the same on a synthetic walk.

## Finding

Observed:
- **The head's confidence is the better policy, by far.** On the 16 GB leg a lane's chain that stops at the
  first draft whose chain probability falls under 0.56 commits 7.5% more tokens per second than each cell's
  best fixed window (6.8% unweighted, 8.4% by the correction's model), when the round then verifies only the
  width its lanes' chains reached; 3.3% when only the chain stops and the verify keeps the load's window; and
  nothing in today's round, which runs its window's steps whatever the extent (its best is window 7, 0.991).
  The acceptance history commits at most 1.6% more under any pricing.
- **The gate is cleared only by the narrowed round**: 1.075, and 1.066 to 1.074 on each launch's table and the
  smoothed one; held out on the other half of the jobs 1.074 and 1.076; per cell 1.049 to 1.100, the two lanes
  64K deep the narrowest; thresholds from 0.3 to about 0.8 all clear it. A chain that stops with the verify at
  the window reads 1.016 to 1.035: under the gate on every table, before its skipped steps' own cost.
- **The narrowed round needs two costs the table does not hold**: the chain's stop on the device at the first
  draft under the threshold, and the verify's width chosen after it, inside the round. The gain stays at 5% or
  more if the two together cost at most 0.29 to 0.43 ms a round (the second launch's table to the two launches'
  mean): about 1 to 3% of a 16 GB round.
- **The ceiling**: a round at the extent its accepted run needed would commit 22.5% more (a shared width), 15.6%
  (the chain stops), 24.4% (a lane's own width); the confidence stop takes a third of the shared width's
  ceiling.
- **NVFP4**: the confidence stop gains 9% at one lane and loses 2 to 3.5% at eight lanes, 3.0% for the leg; at
  eight lanes the widest lane's extent is near the window.
- **Several lanes**: a width the lanes share needs 12 more captured verify graphs on the 16 GB load (at most
  78.7 MB, a line of its own in the plan) and makes a lane's width follow its neighbours'; each lane its own
  width needs 56 exact shapes at two lanes and cannot be captured at eight.
- **The same records give the same extents and the same figures**, byte for byte, in two runs.

Inferred:
- The head's probability carries per round what the history cannot: a run's survival looks close to
  independent from round to round once the cell is known, so the history settles near a fixed window.
- What makes the narrowed round worth more than the chain's stop is the verify columns it does not take: at
  one lane 4K a column is 0.20 ms beside a 0.80 ms chain step, and the stop's mean extent of 3.8 leaves three
  of the window's seven columns unverified.
- The toolkit the leaf builds with (CUDA 13.1) has conditional graph nodes, which could stop the chain and
  pick the verify's width on the device without a host round trip; the leaf uses none today, and neither
  their cost nor a host read-back's between two launches is measured here.

## Implications

- **The verdict on the gate (16 GB leg): conditional.** The head's confidence clears it, 7.5% over the best fixed
  window, in a round that stops its chain on the device and verifies only the width its lanes reached, if those
  two cost under 0.29 to 0.43 ms a round together; neither is measured. A chain that stops with the verify at
  the window does not clear it (3.3%), today's round gains nothing, and no history policy clears it. Until
  those costs are measured on the RTX 5090, the fixed draft window ships (7, #628's default) and the
  instruments stay, for the 12 GB spec (#630) to ask again on its own artifact:
  `tools/mtp-replay/extent.py` over `mtp_replay`'s records and `mtp_round_cost`'s table.
- **If the measurement says go**, the policy is the confidence stop on the chain's probability, keeping the draft
  under the threshold, under the load's draft window as its ceiling; its configuration fields, not added here,
  would be the extent's rule beside `spec.draft_rule` (the window, as today, or the confidence stop) and its
  threshold: 0.56 measured on the 16 GB artifact under `rtx-16gb` (0.68 on NVFP4 under `rtx5090`, whose leg
  does not clear the gate), each a profile's default.
- **For several lanes**, the figures favour a width the lanes share (12 more graphs, a plan line, a lane's text
  following its neighbours' confidence, the spec's determinism kept for a lane alone in its rounds) over a lane's
  own width, which saves less at two lanes and cannot be captured at eight. Which of the two, and whether a
  lane's text may follow its neighbours', is the owner's to decide with the ticket that would build it.
- **What to measure first**, before any controller: the cost of stopping the chain on the device and choosing
  a verify width inside the round after it (a read-back and a second launch, or conditional nodes), and what a
  skipped chain step still costs.

## Limits and unknowns

- Derived, not walked: every policy is rated on the window-7 walk's anchors; the walk a policy makes would start
  rounds elsewhere. The derivation reads it low by about the correction's model (about 1% for the chosen configs, the
  walk check agreeing), and the records are not the same at an anchor two walks share (drafts differ at 7 to
  11% of the shared anchors on 16 GB, 23 to 30% on NVFP4: likely because the head's continuation hidden and
  state come from a verify column at another width; `explore2.py` beside the runs).
- Priced, not run: the costs are the table's rounds less what a policy does not run; no round of a policy ran.
  What a skipped step, a width chosen inside a round and a round of lanes at their own widths cost is not
  measured. The cost of capturing and holding more graphs is inferred from the residual, one 16 GB start of two.
- One card's costs (the RTX 5090, Windows, WDDM); the 16 GB leg is a 13 GiB budget on it, not a 16 GB card, whose
  bandwidth moves a chain step's share and so the stop's value.
- One workload: #267's 200 agent turns sampled at temperature 1; greedy traffic accepts more, which favours
  longer extents and leaves less to stop.
- Lanes are modelled as independent walks from different job starts, all at one context; served lanes
  differ in context and come and go.
- The history policy is one design (decayed counts per position under a prior); the reference engine's
  hold-offs and probes were not replayed.

## Follow-ups

- The cost of a verify width chosen inside the round, and of a skipped chain step, on the RTX 5090: the owner's
  call, before any ticket builds the controller.
- The 12 GB card class asks again on its own artifact: https://github.com/gpillon/ignis/issues/630.
