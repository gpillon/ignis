# The dense prefill's tile plan weighs a 128-row tile's rows at 1.10: on the RTX 5090 the plan comes within 1.2% of the fastest tile plan at every chunk width, from 4.9%

- Kind: experiment
- Status: current
- Observed: 2026-11-07
- Last verified: 2026-11-07
- Scope: kernel / the dense prefill's tile plan (`plan()` in `kernel/src/dense_prefill.cu`, `kTile128RowCostPercent`), the 27B trellis body's GEMMs at chunk widths 2048 to 8192
- Related: [GitHub #530](https://github.com/gpillon/ignis/issues/530), [GitHub #509](https://github.com/gpillon/ignis/issues/509), [spec #503](https://github.com/gpillon/ignis/issues/503), [ADR 0080](../adr/0080-on-sm8x-the-trellis-prefill-decodes-each-matrix-once-per-chunk-and-multiplies-on-the-integer-mma.md)
- Superseded by: none

## Question

The dense prefill's GEMM runs at one of two tile heights, BM 128 or 256 token
rows. `plan()` picked between them by one cost, waves of resident CTAs times
the tile's rows, so BM 128 won whenever it was strictly cheaper by that count:
it costed a token row in a 128-row tile as it costs in a 256-row tile. Does it,
on the RTX 5090, and what weight on a 128-row tile's rows makes the plan pick
the faster tile height across the body?

## Evidence

- **Method.** `kernel/build/tests/ignis_kernel_dense_prefill_bench --tokens
  2048,4096,8192 --sweep`, twice, and `--tokens 6656 --sweep`, twice, on the
  RTX 5090 (170 SMs, one resident CTA per SM at both tile heights and both
  depths), from the leaf built for `86;120a`. The sweep times each body entry's
  GEMM at all eight plans (two tile heights x two depths x two rasters) and
  marks the plan's own row.
- **The measure.** The plan's loss at a width: the sum over the body's entries,
  each weighted by its calls in the bench's K mix (twice for gate and up), of
  the plan's row less the fastest of the eight rows, as a share of the dense
  body the bench prints for that width (decode then GEMM, plus A8). The plan's
  own row is used, not the bench's first timing of the plan, which runs at a
  higher clock right after the decode and so hid the loss behind its "fastest"
  label.
- **Before**, `main` at 0bb6e820:

  | T | pass 1 | pass 2 | body (ms) |
  |---|---:|---:|---:|
  | 2048 | 4.82% | 4.92% | 193.8, 194.2 |
  | 4096 | 0.46% | 1.06% | 353.4, 347.3 |
  | 8192 | 2.51% | 2.22% | 690.0, 690.5 |
  | 6656 | 4.19% | 4.23% | 577.0, 578.6 |

  #530's own sweep on the same card, earlier the same day, gave 4.9-5.2%,
  0.7-1.0%, 2.2-2.7% and 4.1%.
- **Each tile height's fastest row**, summed per shape over its K entries
  (both passes):
  - at 2048, `mlp gate+up` takes 7.8-8.9% longer on BM 128 than on BM 256,
    where waves x rows called BM 128 7% cheaper (1,664 rows against 1,792);
  - at 2048, `gqa k, v` takes 82-83% longer on BM 256, and `gdn z` 12-14%;
  - at 6656, `mlp down` takes 9.2-9.3% longer on BM 128.
- **The weight, replayed on the before rows.** The pick at 170 SMs with a
  128-row tile's waves x rows weighed by E, read off the same rows:

  | T | E 1.05 | E 1.08 to 1.18 | E 1.20 |
  |---|---:|---:|---:|
  | 2048 | 4.82%, 4.92% | 0.88%, 0.77% | 1.76%, 1.16% |
  | 4096 | 0.46%, 1.06% | 0.06%, 0.64% | 0.06%, 0.64% |
  | 8192 | 1.67%, 0.79% | 1.12%, 0.15% | 1.12%, 0.15% |
  | 6656 | 3.16%, 3.20% | 0.20%, 0.05% | 0.20%, 0.05% |

  Every E from 1.08 to 1.18 picks the same rows; at 1.20 `gdn z` at 2048 goes
  to BM 256.
- **After**, the plan with E = 1.10 (`kTile128RowCostPercent = 110`):

  | T | pass 1 | pass 2 | body (ms) |
  |---|---:|---:|---:|
  | 2048 | 0.89% | 1.15% | 182.3, 185.3 |
  | 4096 | 0.47% | 0.35% | 342.4, 344.6 |
  | 8192 | 0.16% | 0.22% | 672.1, 673.4 |
  | 6656 | 0.18% | 0.35% | 553.2, 555.8 |

- **Two passes set aside.** The first two after-change passes at 2048 to 8192
  ran on a card that was slower on identical machine code: `gdn qkv` K 3 at
  4096, BM 256, 3 stages, tokens fastest, took 804 us in both, against
  665-667 us in the passes before and 653-660 us in the passes after; the
  bodies were 10-20% longer. A GPU sampler running beside them showed the SM
  clock at 2,360-2,525 MHz and the power at its cap, as in the clean passes;
  the cause was not identified. Their plan's loss was 0.93% and 1.19% at 2048,
  1.82% and 0.46% at 4096, 0.52% and 0.28% at 8192, the 1.82% coming from
  depths and rasters at BM 256, where the plan did not change.
- **The outputs.** `ignis_kernel_dense_prefill_test` holds every one of the
  eight plans to the op's own plan, every BF16 byte, for every body shape at
  1, 300, 2048 and 2049 tokens through the store, the add, SwiGLU and the
  split, and the planner's picks for the body's shapes at 2048 to 8192 tokens
  on 82, 128 and 170 SMs. The `86` and `120a` machine code of both archives
  is byte-identical before and after the change (#482's method), so the change
  is the host's pick alone.

## Finding

Observed:

- On the RTX 5090 a token row costs more in a 128-row tile than in a 256-row
  tile: the plan that costed them alike lost 4.8-4.9% of the dense body at
  2048 tokens, 2.2-2.5% at 8192 and 4.2% at 6656, against the fastest of the
  eight plans.
- Weighing a 128-row tile's waves x rows by 1.10 brings the plan within 1.2%
  of the fastest plans at every width measured, in both passes, and shortens
  the body by about 5% at 2048 tokens, 4% at 6656 and 2.5% at 8192.
- BM 256 everywhere would not do: BM 128 stays the faster tile height for
  `gqa k, v` and `gdn z` at 2048 tokens, and the weighed cost keeps it there.

Inferred, from #530's analysis of the warp tiles and not measured here: the
128-row tile issues 6 `ldmatrix` per 16 MMAs against 8 per 32 and loads 1.33x
the shared-memory bytes per MAC, so on a card with full-rate INT8 its shared
memory and L2 traffic stop hiding under the MMAs.

## Implications

- `plan()` picks BM 128 only where `cost128 x 110 < cost256 x 100`, waves x
  rows at each height (`kTile128RowCostPercent` in `kernel/src/dense_prefill.cu`).
  The constant is a planning constant of the kernel route, not a config field.
- The change is speed alone: every plan writes the same bytes, both tile
  heights hold one CTA per SM, and the dense buffer, the activation workspace
  and `y` do not depend on the tile height.
- The dense prefill is the default route on RTX 30 and RTX 40 cards, and on
  the RTX 5090 the route `--model-prefill-linear int8` names.

## Limits and unknowns

- Measured on the RTX 5090 only. The picks at 82 and 128 SMs follow the same
  rule (the leaf test pins them), but no card with those SM counts measured
  them, and a card with less memory bandwidth per SM may want a larger weight.
- E is one number for every shape and width. At 170 SMs the replay's plateau
  is bounded by two shapes at 2048 tokens: waves x rows favour BM 128 by 1.077
  for `mlp gate+up`, which wants BM 256, and by 1.20 for `gdn z`, which wants
  BM 128.
- The remaining loss is the depth and the raster, not the tile height: at 2048
  tokens `mlp down` runs 1-5% faster at 3 stages than at the 4 its K takes, and
  `gdn z` on BM 128 3-5%.
- The weights are synthetic counter-hash words; the GEMM's time does not
  depend on their values. A raster that takes the L2 size into account is not
  covered.

## Follow-ups

- none
