# The 27B's mixed rounds while spec #711 was built, on the RTX 5090: the NVFP4 body's mixed rows inside the extended round's bounds over a uniform cell's window, the trellis body's bit for bit a uniform round's of the same columns, the trellis body's cost table and the memory the mixed graphs' captures took

- Kind: experiment
- Status: current
- Observed: 2026-11-23
- Last verified: 2026-11-23
- Scope: decode / the 27B's DFlash2 mixed verify rounds at 15 drafts on the RTX 5090 (Windows): the numerical class
  test on the NVFP4 body (FP8 exception weights) and on the trellis body (BF16), 2026-11-22 and 2026-11-23; the cost
  tables' timing of #721 on both bodies; the device memory the twelve mixes' graphs took at capture
- Topic: decode-27b
- Related: https://github.com/gpillon/ignis/issues/717,
  https://github.com/gpillon/ignis/issues/720,
  https://github.com/gpillon/ignis/issues/721,
  https://github.com/gpillon/ignis/issues/714,
  https://github.com/gpillon/ignis/issues/711,
  [ADR 0106](../adr/0106-the-27bs-dflash2-verify-round-may-carry-lanes-of-both-shapes-and-a-mixed-round-is-held-to-its-base-rounds-numerical-class.md),
  [ADR 0104](../adr/0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md),
  [the 27B's mixed rounds served](2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md)
- Superseded by: none
- Summary: RTX 5090, 27B at 15 drafts, twelve mixes: NVFP4 mixed rows inside ADR 0104's bounds over 224-row windows (widest 110.8 of 256 ulps, 0.0158 of 0.02 nats); trellis mixed rows bit-identical to uniform rounds of the same columns.

## Question

Spec #711 lets a DFlash2 verify round of the 27B at 15 drafts carry lanes of both round shapes, a **mixed round**
(ADR 0106), one of twelve mixes of `w` extended lanes beside `n` base lanes. While it was built, on the RTX 5090:

- are a mixed round's rows in its base round's numerical class, by the bounds ADR 0104 holds the extended round to
  (#717), and over what window of a lane's text can they be read;
- what are a trellis body's mixed rows held to (#720);
- what does each round of the cost table cost on the trellis body, and how far do the legs of that measurement
  agree on either body (#721);
- how much device memory do the mixed graphs' captures take (#714).

These measurements stood in ADR 0106 until 2026-11-27, when the ADR took the short format (#742) and they moved
here unchanged; the served gate of the same spec is
[its own finding](2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md), which also holds the NVFP4 body's
cost table.

## Evidence

### Conditions

RTX 5090 under Windows. The class is read by
`crates/core/tests/extended_round_class_gpu.rs`, whose opening comment states the method and whose constants carry
each bound beside its measured values: the NVFP4 body under FP8 exception weights and the trellis body (the `sc30`
artifact) under BF16, each over hq-e8-2b, DFlash2 at 15 drafts, 8 lanes, a context of 262,144. Speculation off
writes the greedy text of seven prompts, one a lane; each of the twelve mixes is then teacher-forced to that text
at its 2 to 7 lanes in two kinds of round, **forced** (offered the text as its drafts, eager) and **graph**
(replaying the mix's captured graph on the drafter's own drafts), its last lanes extended. The raw logs are not in
the repository.

```text
cargo test -p ignis-core --features cuda --test extended_round_class_gpu -- --ignored --nocapture --test-threads=1
```

### The NVFP4 body: the first reading, and why a lane is read over a uniform cell's window

The first reading rode each mixed leg while every lane's text, at most 224 tokens, held a whole round of its shape,
so a lane beside a faster one held 51 to 85 rows on prose. On graph rounds the prose prompt's lane extended at the
mix 3+2 read its text 0.0256 nats a token below the base rows' over 79 rows, past the bound of 0.02. A diagnostic
found the mix adds nothing to it:

- over the same 79 rows that prompt's lane of the uniform extended graph cell reads 0.0257 at 4 lanes and 0.0201 at
  one, its rows bit-identical to the mixed lane's over the first 54, where the two legs' rounds pair, and the
  excess sits in a few near-tie rows, one of them 0.78 nats of the lane's 2.02;
- a mean over a short window of prose moves so in the uniform cells too: that lane of the uniform extended graph
  cell reads 0.0348 below its base rows over its first 32 rows at one lane and 0.0303 at four, 0.0242 and 0.0177
  over 64, inside 0.02 for good only from row 87 at one lane and from row 180 at four, and 0.0015 and 0.0124 apart
  over its 210, the window over which ADR 0104 measured and confirmed its bounds;
- a longer leg (texts of 512 tokens, the other lanes on essays) brought the 3+2 lane inside, 0.0130 over 324 rows;
  the near tie read over the mixed cells' own base rows (9.0, so 18) then refused a pick moved 22.0 ulps apart, a
  move the uniform extended graph rows make too and ADR 0104's reading over its kind's base rows passes. The base
  round's own near tie widens on longer texts (to 42 ulps on the long answers, ADR 0104).

### The NVFP4 body: the mixed rows over a uniform cell's window (2026-11-22)

Each lane held over the rows after its text's first 224 tokens, against the base rows a leg of base rounds at its
lane count gives over the same window, at least 64 of them; the near tie read over the base rows of the kind's
uniform cells at 1 to 4 lanes (both kinds' widest moved gap 12.0 ulps, so 24). NVFP4 body, FP8 exception weights,
hq-e8-2b, over 12,096 forced and 11,887 graph mixed rows at 2 to 7 lanes (224 a lane forced, 217 to 223 on graph
rounds), each leg 28 forced rounds or 45 to 88 graph rounds at its mix:

| bound | margin | forced mixed rounds | graph mixed rounds |
|---|---|---|---|
| widest row, mixed against base | 256 ulps | 1.0 | 110.8 |
| a lane's median row | 32 ulps | 0.0-0.5 | 0.0-14.9 |
| mean log-probability, apart | 0.02 nats | 0.0005 | 0.0158 (0.0229 over its first 64 rows) |
| class ratio, median / 90th percentile / widest | 1.25 / 1.25 / 1.5 | 1.00 / 1.00 / 1.00 | 1.00 / 0.99 / 1.00 |
| the same, by lane count | | 1.00 / 1.00 / 1.00 | 0.99-1.01 / 0.95-1.02 / 0.89-1.00 |
| picks moved from the plain rows, mixed and base | 1.25 x + 2 | 385 and 385 | 311 and 286 (70 and 57 at 4 lanes, bound 73) |
| widest moved gap, mixed over the mixed cells' base; between them | 2 x the uniform cells' 12.0, so 24 | 9.0 over 9.0; none | 10.0 over 10.0; 12.0 |
| control's median, over the right rows' and the typical bound | 2 x | 14.0x, 5.0x | 14.0x, 5.0x |

- Forced, every mixed row is its base row within 1 ulp, both groups' (the uniform shapes' forced rows already agree
  so at 2 to 4 lanes); on graph rounds a lane's row at one text index can stand at another column of its round in
  the mixed leg than in the base leg, each leg's drafts and accepted counts its own, as between a uniform graph
  cell's two shapes.
- The furthest likelihood is the essay prompt's lane extended at the mix 3+2, 0.0158 over its 218 rows, which reads
  0.0229 over its first 64: the short window's excess again, inside the bound over the window.
- The near tie's reading was not put to the test on this run: over the mixed cells' own base rows it would have
  been 18 and 20, which the mixed rows' 9.0 and 10.0 pass too.
- The same run's uniform cells print what the first reading's did, line for line.

### The trellis body: the NVFP4 body's method, and its diagnostic (2026-11-22)

Read on the trellis cell of the same test (the `sc30` artifact, BF16 exception weights, the same 8 lanes and
context) by the method above, the mixed rows sat inside every bound of the row and the likelihood and past five
relative clauses of the class:

- forced rounds at 2 lanes, the widest mixed row 23.5 ulps from speculation off's against the base leg's 14.3, past
  1.5 times it;
- graph rounds, 4, 8 and 9 picks moved from the plain rows at 3, 4 and 5 lanes against the base legs' 0, 3 and 3,
  and 30 against 13 over the kind's lane counts.

A diagnostic found no defect in the trellis mixed path:

- every mixed row, 12,096 forced and 11,808 from the graph, is bit-identical to the same prompt's row in a uniform
  round of its own shape at the round's columns: on the trellis body a lane's rows follow its round's columns, not
  its lane count, the trellis projection routing by the call's token count;
- forced, the base legs of 3 to 7 lanes (24 to 56 columns) give each prompt's rows the same bits and the leg of 2
  (16) others, so the test read the 1+1 mix's 24 columns against a base leg of 16;
- the graph rounds' moved picks are six text positions at exact ties, gaps of 0 or 1 ulp, counted once for each mix
  that holds the lane, positions the uniform rounds move too;
- the argmax is not in it: the rows are the same under `IGNIS_MIXED_GRAPH_BUILD=0` (12,096 and 11,808 of them, the
  moved picks 49 and 30 both ways).

### The trellis body: bit for bit to uniform legs at one width (2026-11-23)

A lane's rows are the same bits in every round of 24 to 56 columns, of either kind: forced, the base legs of 3 to
7 lanes and every mix of those columns; on graph rounds the base round at 3 and at 7 lanes, every mix of those
columns against the base round at 7 and the extended at 3. A graph round of 64 columns gives its own where the
drafter's drafts move: two prompts' lanes leave the 56-column round's at their rows 6 and 17, the three mixes of 64
columns bit-identical to the uniform rounds of 64. The extended shape has no round of 24, 40 or 56 columns, and has
a round of a mix's own columns only beside an even count of base lanes. So the uniform legs ran at one width, on the
mixed prompts' texts, every lane riding until each holds 224 rows: 7 base lanes (56 columns) and 3 extended (48)
for the nine mixes of 24 to 56 columns, 8 base lanes (the eighth on a prompt again) and 4 extended for the three of
64.

Every mixed row of the trellis cell is its uniform leg's, bit for bit: 12,096 forced rows (each mix 28 rounds) and
12,096 from the graph (224 a lane, each mix 39 to 88 rounds) over the 54 lanes of the twelve mixes, the cell's
uniform section printing what its run of 2026-11-22 did, line for line.

### What the mixed checks add to the test's time

They ride each cell's load after its uniform checks, with no load of their own:

- the NVFP4 cell five to six minutes more: 323 s on 2026-11-22, 36 s of it speculation off's texts; the NVFP4 test
  431 s, where the first reading's took 304 to 314;
- the trellis cell, bit for bit, under three minutes: 162 s on 2026-11-23, 32 s of it the texts; the trellis test
  272 s, 120 s without the mixed checks and 450 s with the method of the NVFP4 cell.

### The cost table on both bodies (#721, 2026-11-23)

Through the round identity test's timing (`IGNIS_ROUND_IDENTITY_TIMING_TABLE`) at a served context
(`IGNIS_ROUND_IDENTITY_TIMING_CONTEXT=3584`: each timed lane's prompt behind a long text of the guide's, prefilled
to 3,584 tokens and ridden 1,536 more, a fresh set each time a lane reaches that, the model loaded at 262,144
tokens of context): at each lane count of 1 to 8, the base round and each count of extended lanes 64 columns hold,
each cell after a capture again, 600 greedy rounds of fresh lanes forced to its shapes (its last lanes extended),
on the GPU clock with the round timing; the NVFP4 body on `nvfp4-dflash2-w15` (FP8 exception weights, hq-e8-2b, the
mixed graph build), the trellis body on `trellis-24gb-w15-rule`; four legs each, riding the cells forward, in
reverse, in reverse and forward. Each cell's period is the median of its 256-round blocks' medians over the four
legs, eight of them, since a shared machine only ever slows a block, so one slow block or leg must not move a cell.
An entry is its cell's period over the base round's at its lane count. The NVFP4 body's rows
at 1 to 7 lanes are in [the served finding](2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md); its base
round at 8 lanes took 30,753 us, 0.3% more than at 7 (30,647 us). The trellis body's, each row's entries at 1, 2,
... lanes extended:

| lanes | trellis base round | trellis entries |
|---|---|---|
| 1 | 14,875 us | 1.364 |
| 2 | 20,191 us | 1.214, 1.407 |
| 3 | 24,586 us | 1.162, 1.502, 1.629 |
| 4 | 28,785 us | 1.291, 1.391, 1.55, 1.606 |
| 5 | 37,291 us | 1.082, 1.204, 1.243 |
| 6 | 40,415 us | 1.117, 1.154 |
| 7 | 45,270 us | 1.036 |
| 8 | 47,018 us | none |

- The legs' own readings of an entry span at most 0.007 on the NVFP4 body and 0.012 on the trellis body (0.006 past
  one lane), and dropping any one leg moves no entry by more than 0.002 on the NVFP4 body and 0.004 on the trellis
  body.
- A round with extended lanes costs what a base round of its columns costs to within 2.5% on either body.
- On the NVFP4 body the step from 56 columns to 64 costs nothing or less, its 56-column rounds at about 2,775 MHz
  and its 64-column ones at about 2,815, in either order, near the card's 575 W limit. The trellis body's rows do
  not fall: its smallest step, from the base round of 7 lanes to 1+6, is 0.036.

### The memory the mixed graphs' captures took (#714, 2026-11-22)

The twelve mixes' graphs at 8 lanes, captured with the mixed graph build, as the device's free memory before the
captures less after them: 69,206,016 B (66 MiB, two runs) on the NVFP4 body and at most 73,400,320 B (70 MiB) on
the trellis body; one mix's capture took 4 to 10 MiB, in the driver's 2 MiB pages.

## Finding

Observed:

- **On the NVFP4 body every mixed row sits inside the bounds ADR 0104 holds the extended round to**, over a uniform
  cell's window of 224 rows a lane: the widest mixed row 110.8 ulps from its base row (the bound 256), a lane's
  median row at most 14.9 (32), the furthest likelihood 0.0158 nats a token (0.02), the class ratios 0.89 to 1.02
  (1.25 and 1.5), and forced, every mixed row its base row within 1 ulp.
- A lane read over a shorter window fails the likelihood bound with no fault of the mix: over 79 rows of prose the
  mixed lane reads 0.0256 and the uniform extended lane 0.0257, bit-identical rows where the legs' rounds pair.
- **On the trellis body a lane's rows follow its round's columns, not its lane count**, so a base leg at the lane
  count is no reference there; every mixed row, 12,096 forced and 12,096 from the graph, is bit for bit its
  prompt's row in a uniform round of its shape at one width (24 to 56 columns, or 64).
- The trellis body's cost table rises along every row; the two bodies' legs agree within 0.012 on any entry, and a
  round with extended lanes costs within 2.5% of a base round of its columns on either body.
- The twelve mixes' graphs take 66 MiB on the NVFP4 body and at most 70 MiB on the trellis body.

Inferred:

- The likelihood excess of a short window is the window's: a few near-tie rows of prose carry it (one row 0.78 nats
  of a lane's 2.02), in the uniform cells as in the mixed ones.
- The trellis cell's five class clauses past their margins under the NVFP4 body's method came from the reference,
  a base leg of other columns, not from the mixed path: the rows and the moved picks are the same without the mixed
  graph build, and bit-identical to uniform rounds of the same columns.

## Implications

- The class test holds an NVFP4 mixed lane over a uniform cell's window and reads the near tie over the uniform
  cells' base rows, and holds a trellis mixed row bit for bit to a uniform round of its shape at its round's columns
  (ADR 0106).
- A change that splits the trellis body's rounds of 24 to 56 columns into rounds of different bits fails the
  trellis cell, and the uniform legs then follow the split.
- The VRAM plan's `mixed_graphs` line holds the twelve at each body's largest reading and each mix fewer at 5 MiB
  less (`kernel/src/mixed_graphs.h`).

## Limits and unknowns

- One machine, an RTX 5090 under Windows.
- The bounds are the RTX 5090's on the NVFP4 body under FP8 exception weights over hq-e8-2b.
- That a trellis lane's rows are the same bits in every round of 24 to 56 columns was read on this card alone.
- The cost tables are figures at about 3.5-5K tokens of context, and the ratio grows with context; the trellis
  body's was measured with the model loaded at 262,144 tokens of context, where the `rtx-24gb` profile serves
  1,048,576, and on no 16 GB or 24 GB card.

## Follow-ups

- None of its own: the cards these were not measured on are
  [the served finding](2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md)'s follow-up.
