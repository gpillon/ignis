# ADR 0040 — the attention readout carries each peak's neighbours

## Status

Accepted (2026-09-23, owner — spec `docs/specs/decide/15-the-head-box-reads-inside-the-cell.md`,
GitHub #264). **Extends ADR 0039**, which extends ADR 0038. Everything both
decided holds: the job names exactly what to read, exactly that comes back,
the keys are the ones attention read, and a read the leaf cannot make is a
failed question, never a partial set.

Source: `docs/findings/2026-09-23-the-head-box-is-quantised-to-the-cell.md`.

## Context

ADR 0039 brings back one number per head of the set — the image cell it
peaks on. The host then places each head at that **cell's centre**, spans
the kept cells and grows the span half a cell.

Every edge of the resulting box therefore sits on cell geometry, and spec
14's acceptance found the one regime where that is fatal. At 1024 px an image
cell is 32 px and a labelled button is 1.44 cells tall — 199 of 240 are under
two cells. The smallest box the rule can express in y is one whole cell, the
usual one is two, and the metric is capped before any score is read:
snapping the *true* box to the grid scores 82.5% at IoU >= 0.5, and covering
the cells the true box touches scores 52.5%, against the digit chain's 79%.
The head box lost that one cell of the table, and `box`'s default stayed on
the chain.

A head's score map is smooth around its peak, so the peak's position *inside*
its cell is recoverable — but only from scores the seam does not carry. The
whole rows would be 6.3 MB at 4096 px, which is what ADR 0039 exists to
avoid.

## Decision

- **The outcome carries each peak's four in-line neighbours.** Beside the
  pointing head's score row and the set's argmax indices, a readout that
  named a set brings back, per head and in the set's order: that head's
  **score at its own argmax**, and the scores at the argmax's **left, right,
  up and down neighbours in the image grid**, in that order. Four scores per
  axis-pair is what parabolic interpolation needs and no more — a 3x3
  neighbourhood was measured and read no better. At 4096 px that is 1.5 KB
  on top of ADR 0039's 384 bytes, against 6.3 MB for the rows.
- **The peak's own score is not copied twice.** It is the high half of the
  packed value ADR 0039's fused kernel already reduces the argmax with, and
  the host undoes that order-preserving packing to read it.
- **The query names the image grid's columns.** The span is walked
  row-major, so the left neighbour of a key in column 0 does not exist and is
  never the previous row's last key. The leaf is told the columns so it can
  say so; without them it would have to guess the image's shape from a flat
  span.
- **A neighbour off the grid is absent, not a sentinel.** The leaf writes
  **NaN**, the runtime turns that into `None`, and the host rule reads an
  absent neighbour as no sub-cell offset on that axis — the other axis still
  reads. A score of "minus infinity" or "zero" would be a number the rule
  could interpolate, and would move a box off the object at the image's
  border.
- **The excluded keys are ordinary neighbours.** A head may not *peak* on a
  fallback cell (ADR 0039), because that is where a head parks when it finds
  nothing; the score there is still the attention's, and a peak beside one is
  a peak beside a real score. This is what every number behind the rule was
  measured with.
- **One small gather launch per armed layer.** ADR 0039's fused kernel is
  unchanged and still publishes the packed (score, index) per head. A second
  launch, in the same layer's scope and on the same stream — so the fused
  reduction has finished — reads this layer's argmaxes and scores the at most
  four neighbours of each: at most 96 dot products a layer, against rows the
  layer's plane still holds. It is launched only for a layer holding a head
  of the set, so a job with no readout, or with the pointing head alone, pays
  nothing and launches exactly what it launched before.
- **The arithmetic stays on the host.** The leaf ships scores; the host reads
  the sub-cell position from them. The rule can then change without a kernel
  change, and the device path is checked against the test-only tap the same
  way ADR 0039's argmax is.

## Consequences

- `ignis_prefill_options` grows by appended fields only (ADR 0016):
  `attention_grid_cols`, `out_attention_set_peak`,
  `out_attention_set_neighbours`. A set named without them, or with a grid of
  no columns, is refused before any GPU time — a caller that meant a set and
  would silently get no way to read inside a cell.
- The plan's readout reservation grows from 3 KB to 9 KB for the largest set
  (384 heads). Still nothing against the score row's 64 KB at 4096 px.
- Two launches per armed layer instead of one — eighteen instead of nine on
  the served set. The gather's work is negligible beside the fused pass, but
  the launches are not free, which is why the cost is measured inside the
  prefill at both grids rather than argued.
- A head `point` and a head `box` now answer to **half a cell** per axis
  rather than one: `uncertainty` is the reading's resolution, and the reading
  resolves inside the cell. A documented number changes meaning; no field is
  added or removed.
- The host rule and its Python reference
  (`tools/pointing-scenes/ensemble_score.py`) stay one rule, held together by
  golden cases the reference writes.

## Alternatives

- **Carry every head's score row.** 6.3 MB at 4096 px per question, for a
  reading that uses five numbers a head. This is the cost ADR 0039 exists to
  avoid.
- **Compute the sub-cell offset on the device** and carry two floats a head.
  Smaller still, but it moves the rule into the kernel: changing a constant
  would mean changing the leaf, and the rule could no longer be held to the
  reference as a pure host function.
- **Publish the neighbours from the fused kernel itself**, with no second
  launch. At the time a block scores key `k` it does not yet know whether `k`
  will win, so it would have to publish every key's score — the rows again —
  or the grid would need a device-wide barrier the launch does not have.
- **Keep cell centres and tune the growth instead.** Measured: the best a
  cell-quantised box can do on 1024 px buttons is 82.5%, and every constant
  growth trades buttons against photographs because the two regimes want
  opposite corrections. Reading inside the cell is what removes the trade.
