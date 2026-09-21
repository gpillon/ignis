# The grid a bigram can name is 9 x 9, and that is too coarse to point with

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision readout primitives, answer alphabet, one-pass pointing
- Related: `crates/core/tests/grid_label_rectangle.rs`, `crates/core/src/decision.rs`,
  `docs/specs/decide/11-point-in-one-pass.md`, ADR 0034,
  `2026-09-19-constrained-digit-readout-points.md`,
  `2026-09-21-scalar-readout-legend-width.md`
- Superseded by: none

## Question

A readout reads one position and names one vocabulary entry, so the only way
to answer a **point** without generating anything is to make the whole answer
one token: a label that names a cell of a 2-D grid.

The answer alphabet has a naming that is already compositional. `AX` reads as
row `A`, column `X`, so a grid of uppercase bigrams needs a legend of one
sentence rather than one line per cell — which is the difference that matters,
because spec 08 measured a legend the model has to *read* collapsing at width
(4/5 correct at 20 and 32 cells, 0/6 at 100).

But a label is admitted only if it encodes to exactly one token that decodes
back to itself, and **114 of the 676 uppercase bigrams fail that** in this
tokenizer (ADR 0034). A grid with a hole in it is not a coarser grid: the
missing cell's logit belongs to some other label, which is the one thing the
alphabet rule exists to prevent. So the grid has to fit entirely inside the
admitted set.

How big is that grid?

## Evidence

`crates/core/tests/grid_label_rectangle.rs`, against the real
`qwen3_8_27b_nvfp4full-v2` artifact. No GPU, no materialization: a directory
walk and the tokenizer's own round-trip, ~0.6 s.

Alphabet: **624 labels = 62 singles + 562 bigrams**, matching ADR 0034's
count exactly (676 - 114).

The admitted map, `.` admitted and `x` refused, rows down and columns across:

```
     ABCDEFGHIJKLMNOPQRSTUVWXYZ
  A ..........................
  B ................x........x
  C .........x......x........x
  D ................x........x
  E .........x..............x.
  F .........x......x....x...x
  G .........xx.....x........x
  H .........x................
  I ........................x.
  J .....xxx...x.x..x.....xxxx
  K .........x......x......x.x
  L .......x.x......x.....xx.x
  M ..........................
  N ..........................
  O .........x......x.......xx
  P ................x........x
  Q ...x.x..xxx...x......xxxxx
  R ................x........x
  S ..........................
  T .........x......x.........
  U .........x....x.x.....x...
  V .......x.x......x...x.xxxx
  W .........x......x...xx..xx
  X ......x..xx..xx.x...xxx..x
  Y .x.x.x.xxxx.....xx..xx.x..
  Z .xxx..x..xxxx..xx.xxxx....
```

The holes are not scattered. Columns `J`, `Q` and `Z` are refused across most
rows, and rows `J`, `Q`, `V`, `W`, `X`, `Y`, `Z` are refused across most
columns — the second half of the alphabet in both directions. Only four rows
(`A`, `M`, `N`, `S`) are clean across all 26 columns.

Exhaustive over all 26^4 contiguous rectangles, and greedy-with-every-seed
over non-contiguous ones:

| grid | size | cells | per cell (% of the side) |
|---|---|---|---|
| largest contiguous square | **9 x 9** | 81 | **11.11%** |
| largest contiguous rectangle | 21 x 4 | 84 | — |
| largest free (non-contiguous) | **18 x 18** | 324 | **5.56%** |

The free rectangle's letters are rows `ABCDEFGHIKMNOPRSTW` and columns
`ABCDEFGHILMNOPRSTW` — the alphabet minus eight letters in each direction,
and the two sets differ by one letter (`K` in the rows, `L` in the columns),
so a *symmetric* legend is at most 17 x 17 unless an exact search says
otherwise. Greedy is not exact for maximum edge biclique; this is a lower
bound on the free rectangle and an exact answer for the contiguous one.

Single uppercase letters: **26 of 26 admitted**, with no holes at all.

## Finding

**A one-position bigram grid cannot answer a point.** The measured pointing
error of the digit chain is 2.1% of the side worst case
(`2026-09-19-constrained-digit-readout-points.md`). A 9 x 9 contiguous grid
is 11.11% per cell and an 18 x 18 free grid is 5.56% — 5x and 2.6x coarser
than the answer it would replace, before any question about whether the model
can do the grid arithmetic at all. On the committed fixture, `small`'s button
is 2.2% of the side tall, so an argmax cell misses it **by construction** at
either width.

**The contiguity cost is real and it is a factor of four in cells.** A
contiguous range is a legend of one sentence ("rows A-I top to bottom"); the
free rectangle needs the omissions enumerated ("the letters A-Z omitting J,
L, Q, U, V, X, Y, Z"), which is still one sentence but is a list the model
has to hold rather than a range it can count. 81 cells against 324 is the
price of not knowing which of those two the model handles — and that is a
measurement nobody has taken.

**Single letters have no holes, and that is where the resolution is.** 26 of
26 admitted means a **strip** readout — "which of the 26 equal vertical
strips contains the target" — is nameable with a trivially compositional
legend and no rectangle problem at all. One strip is 3.8% of the side as an
argmax, and the readout returns the whole 26-way distribution, so a centroid
over adjacent strips is not bounded by 3.8%. The price is that x and y are
then **two positions rather than one**, which is a different design and a
different experiment.

**The failure pattern is a property of BPE merge frequency, not of this
grid.** The clean rows are the letters that begin common English words and
the refused ones are the rare pairs, which is why the holes concentrate in
`J`, `Q`, `V`, `W`, `X`, `Y`, `Z`. Any future compositional naming over this
tokenizer will hit the same wall, and it is worth checking with this test
rather than assuming — one artifact swap moves every number above.

## Implications

- The chain-free single-readout point (candidate C1 of
  `docs/specs/decide/11-point-in-one-pass.md`) is **not viable as a point
  answer**. It stays viable as the *coarse* pass of a two-pass box, where
  11% of the side is plenty to seed a scaffold.
- The lead candidate becomes the **two-position strip readout**: x from one
  26-way distribution, y from another, decoded by centroid. It needs the
  multi-position readout head, and it needs the study's E2 to establish that
  a second readout conditioned on a placeholder first answer is not damaged —
  "which vertical strip" and "which horizontal strip" are self-contained
  questions, which is exactly the cross-axis case E2 exists to test.
- Nothing here says the model can *read* any grid. This is a property of the
  tokenizer alone: it bounds what could work, and bounds nothing else.

## Limits and unknowns

- The free rectangle is a greedy lower bound. An exact maximum edge biclique
  over 26 x 26 would settle whether 18 x 18 is the best, and whether a
  symmetric row/column alphabet reaches 18 or stops at 17.
- Only uppercase bigrams were considered, because that is the alphabet
  `candidate_labels()` builds. A mixed-case or letter-digit naming would have
  a different map and is not compositional in the same obvious way.
- Whether the model can map a visual position onto a declared grid at *any*
  width is untested here and is the study's E1.
- One artifact (`qwen3_8_27b_nvfp4full-v2`). Every number is that
  tokenizer's.
