# A scalar reads its legend, and only while the legend is narrow

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: server / `/v1/decide` readout primitives, prompt width
- Related: [ADR 0034](../adr/0034-the-leaf-answers-without-generating.md),
  [typed option logit readout](2026-09-19-typed-option-logit-readout.md),
  [constrained digit readout](2026-09-19-constrained-digit-readout-points.md),
  [number width and decide e2e](2026-09-20-number-width-and-decide-e2e.md),
  [GitHub #252](https://github.com/gpillon/ignis/issues/252)
- Superseded by: the *primitive* is superseded by
  [a number that closes its own object](2026-09-21-a-number-that-closes-its-own-object.md)
  (GitHub #255) — the measurement below is not, and is why that one generates
  rather than reads. Status stays `current` for the finding; what was dropped
  is the implementation ([spec 08](../specs/decide/08-scalar-readout.md), never
  merged).

## Question

`number` answers with a decode round per digit: a three-digit value costs one
prefill and three rounds on a lane, and the third digit is one the model
itself reports as a guess (p = 0.15-0.57 in the pointing finding). A readout
costs one prefill and no rounds, but it reads one position, so it can name one
vocabulary entry — and this tokenizer has **no multi-digit token at all**
(`\p{N}` is isolated in the pre-tokenizer, so `10` is two tokens).

Can a scalar be read in one pass anyway, by declaring a grid of ordered cells
and naming each cell with a single-token label? And if it can, what bounds it?

## Evidence

A live server on the served 27B (`qwen3.8-27b`, hq-e8-2b, no vision), driven
over HTTP against `/v1/decide`. Every item's truth is computable from its own
evidence — a count, a sum, a maximum, a percentage — so no row depends on what
the model knows about the world. The harness, the payloads and every response
are under
[`.scratch/scalar-readout/`](../../.scratch/scalar-readout/): `run.py`
(the sweep), `shuffle.py` and `width.py` (the permuted legends), `narrow.py`,
`stepped.py`, and one `.jsonl` per run.

The grid is shipped as a `score` question, so the prompt is `DIRECT_SYSTEM`
and the options are `{description, letter}` pairs — the shape
`classify_option_ceiling_gpu.rs` measured. A cell's *description* is its value;
its *name* is its index, which is what `probabilities` is keyed by.

### The main sweep: 14 items, three grids, two `number` baselines

`g100` and `g256` describe cell `i` with the text `"i"`; `g10` describes it
with the decade `"i0-i9"`. `num2`/`num4` are `number` at two and four digits.

| variant | median \|err\| of the argmax cell | median \|err\| of `score` (the mean) | exact hits |
|---|---:|---:|---:|
| g100 | 0.00 | 1.24 | 9/14 |
| g256 | 0.00 | 0.80 | 9/14 |
| g10 | 3.50 | 4.00 | 0/14 |
| num2 | 0.66 | — | 7/14 |
| num4 | 1440.00 | — | 5/14 |

`num2`'s misses are the documented width trap, not model error: every
single-digit truth came back multiplied by ten (7 → 70, 3 → 30, 4 → 40,
5 → 50, 1 → 10). Read with that correction `number` is 13/14, better than any
grid.

The grids' own misses are not spread evenly. They are the rows that need the
model to *compute* — `qty_total` (47 → 12), `open_pct` (58.33 → 3),
`fail_pct` (25 → 3) — and on those the distribution is flat, the top cell
holding 0.118-0.294 against 0.90-0.99 on the rows it gets right.

### Is it reading the legend, or the slot?

In `g100` a cell's value, its index and the ordinal of its letter are the same
number, so an answer of 7 is consistent with two mechanisms that have opposite
consequences. Permuting the descriptions separates them: with `"7"` at index
42, a model that reads the legend answers 42 and one that reads the slot
answers 7.

Permuted, 100 cells, on the eight items the identity grid got right:
**four LEGEND, zero SLOT**, four "neither". The four LEGEND rows carry
p = 0.566-0.990 and are the *lookups* (`qty_max` 19, `waiting` 88,
`rolled_back` 24, `score_min` 3). The four "neither" rows picked a cell whose
text is a plausible wrong number (7 → `8`, 4 → `5`), which is a counting
error, not a mechanism failure.

Permuted, 20 cells, on the six items that a 100-cell legend had lost:
**six of six correct**, p = 0.443-0.996.

### Where the width breaks

The same permuted test walked across widths, on the items that need counting
(`width.py`, one seed per width, a cell that cannot name the truth skipped):

| cells | correct |
|---:|---:|
| 20 | 4/5 |
| 32 | 4/5 |
| 48 | 2/6 |
| 64 | 2/6 |
| 100 | 0/6 |

`p_top` tracks it: 0.35-0.96 at 20 and 32 cells, 0.21-0.69 at 100.

### A coarse grid over a range that cannot be narrowed

A percentage's range is 0-100, so `stepped.py` made it coarse instead of
narrow — the same three items at step 5, 10 and 25 (21, 11 and 5 cells):

| item | truth | step 5 | step 10 | step 25 | 100 cells |
|---|---:|---:|---:|---:|---:|
| open_pct | 58.33 | 60 | 60 | 50 | 3 |
| fail_pct | 25.00 | 20 | 20 | 25 | 3 |
| eng_pct | 55.56 | 70 | 60 | 50 | 2 |

### Cost

| variant | input tokens | median s | output tokens |
|---|---:|---:|---:|
| 10-cell grid | 239 | 0.065 | 0 |
| 21-cell grid | ~340 | ~0.08 | 0 |
| 100-cell grid | 1,011 | 0.204 | 0 |
| 256-cell grid | 2,571 | 0.340 | 0 |
| `number`, 2 digits | 132 | 0.130 | 2 |
| `number`, 4 digits | 135 | 0.183 | 4 |

## Finding

**Observed.** A grid of ordered cells named by single-token labels reads a
scalar in one prefill with no decoded token. The model reads the *legend* and
not the slot: on a permuted legend it never once answered by position, and at
20 cells it answered correctly six times out of six. Accuracy falls with the
legend's width — 4/5 at 20 and 32 cells, 2/6 at 48, 0/6 at 100 — and it falls
on the rows that require computation before the lookup, while direct lookups
survive 100 cells at p >= 0.90. The expected value the endpoint returns as
`score` is **worse than the argmax cell** on every set measured here: 1.24
against 0.00 median error on the main sweep, and low-biased on every stepped
percentage (48.61 against an argmax of 60, 16.67 against 20).

**Inferred.** The width cost is a retrieval cost, not a reasoning one: the
same question the model answers correctly over 20 cells it answers with a
plausible neighbour over 100. That is why narrowing the range restores the
answer and coarsening a range that cannot be narrowed (step 5 over 0-100)
moves a percentage from noise to the right neighbourhood.

**Not established.** That a scalar readout is *more accurate* than `number`.
It is not. `number` reads 58 and 25 exactly where a step-5 grid reads 60 and
20, and `fail_count` is wrong at every grid width (3 for a truth of 4) where
`number` is right. What the readout buys is cost and composition — zero
decoded tokens, no lane, no residency, and no width trap — not precision.

### Narrowing is not monotone, and a near-tie stays a near-tie

Shipped as `scalar` and asked over an **ordered** grid inside the ceiling,
`crates/server/tests/decide_scalar_gpu.rs` reads all six of its items exactly:

| item | truth | grid | confidence |
|---|---:|---|---:|
| `err_one` | 1 | 0-10, step 1 | 0.978 |
| `high_count` | 3 | 0-12, step 1 | 0.854 |
| `qty_max` | 19 | 0-31, step 1 (the full ceiling) | 0.783 |
| `rolled_back` | 24 | 0-31, step 1 | 0.879 |
| `score_min` | 3 | 0-10, step 1 | 0.714 |
| `waiting` | 88 | 0-100, step 4 | 0.937 |

A seventh item was dropped from that fixture rather than asserted, and it is
the more interesting row. `open_count` — seven open tickets of twelve — read
**6 over a thirteen-cell grid** where the hundred-cell ordered grid read 7.

That is not the width effect reversing. The hundred-cell run put that item's
top cell at 0.355 against 0.244 for its neighbour — the model was never sure
of it — and an item that close flips on any prompt change. It is recorded
here because it bounds the claim above: narrowing the grid restores the rows
the width cost, and it does not manufacture certainty the model does not
have. A fixture that asserted it would be asserting more than was measured.

## Implications

- A scalar primitive is worth shipping **beside** `number`, not in place of
  it, for a caller who can declare a range and wants the answer without
  spending a decode round per digit or a lane for the duration.
- Its cell count must be capped, and the cap is a measurement rather than a
  preference. 32 is the widest grid measured to hold.
- Such a primitive must return the **argmax cell** as its value. Returning the
  probability-weighted mean, which is what `score` does, is measurably worse
  here — a scalar is not a score, and the distinction is in the data.
- `confidence` and the top cell's probability separate the rows to trust from
  the rows that are well-formed noise (0.90-0.99 against 0.118-0.294), which
  is the same signal ADR 0034 exports answer mass for.

## Limits and unknowns

- The width walk is five or six items with one seed per width, and 20 cells
  scored 6/6 under `narrow.py`'s seed against 4/5 under `width.py`'s. The
  break between 32 and 48 is clear; the exact cap is not sharp to one cell.
- Every permuted run is a *conservative bound* on the shipped shape, which
  presents cells in ascending order. Ordered grids were measured at 5, 11 and
  21 cells (percentages) and at 100 and 256 (the main sweep), never permuted
  and ordered at the same width on the same items.
- One model, one artifact, one tokenizer. The 114 rejected bigrams and the
  absence of multi-digit tokens are this tokenizer's, computed at load.
- Confidence is not accuracy. `qty_max` is right at 0.783 and `open_count`
  is wrong at a top cell of 0.355, but nothing here establishes a threshold:
  six items cannot place one, and the ADR's own abstention figure was
  measured on `choice`, not on a grid.
- Fourteen authored items is a fixture, not a corpus. The counts are small
  (4-16 entries) and the percentages are ratios of them.

## Follow-ups

- `number` at four digits contradicts `numbers.rs::DIGITS`, which documents
  that two or more digits of headroom make the model pad on the left: it
  returned 4700 for 47, 2400 for 24 and 3000 for 3, while padding correctly
  for 3 → `3` and 88 → `88`. The documented rule is unreliable and the
  doc comment should not be leaned on until somebody re-measures it.
- Whether an ordered grid holds past 32 cells, which is the only way the cap
  moves.
- Whether a purpose-built system text that declares the range — rather than
  `DIRECT_SYSTEM`'s "choose exactly one listed option" — buys accuracy back on
  the computed rows. Untested, and it is a new prompt, so it is a new
  measurement.
