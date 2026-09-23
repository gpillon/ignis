# The head set holds through `/v1/decide`: spec 14's acceptance

- Kind: experiment
- Status: current
- Observed: 2026-09-23
- Last verified: 2026-09-23
- Scope: serving / `/v1/decide` `point` and `box`, the attention readout with a head set (ADR 0039), one-pass pointing; hq-e8-2b KV
- Related: `docs/specs/decide/14-point-and-box-from-the-head-set.md` (#263),
  ADR 0039, `crates/server/tests/decide_point_acceptance_gpu.rs` (the run),
  `crates/server/tests/attention_readout_gpu.rs` (leaf against tap),
  `crates/server/tests/attention_set_cost_gpu.rs` (cost),
  `crates/core/tests/anchored_reading.rs` (the host rule against the
  reference), `tools/pointing-scenes/` (generators, `ensemble_score.py`),
  `2026-09-23-an-anchored-head-set-points-and-boxes.md` (the study),
  `2026-09-22-the-head-points-through-decide.md` (spec 13's acceptance),
  `.scratch/head-set-263/acceptance/point-acceptance-*.json` (raw, on disk), #262
- Superseded by: none

## Question

Spec 14 ships the anchored head set as `point`'s default: the prefill that
forces `{"x":` reads the pointing head L39.h10 and 96 heads of GQA layers
31-63 in the leaf, one fused launch per armed layer, and the host keeps the
set's cells near the pointing head's point and answers their extent's
centre. A head `box` is that extent. Every number behind it came from a
test-only tap over harness dumps. Does the shipped path — HTTP, the served
render, the scheduler, the leaf's own device argmax — clear the floors spec
14 wrote down before any run, on scenes never used to choose anything, and
which method should `box` default to by the rule written beside them?

## Pre-registration

In spec 14 (commit e8e5a14, before this run), four sets through
`/v1/decide`, served artifact, hq-e8-2b with the residual window, `point`
with no `method`:

| set | command | `point` inside | head `box` IoU >= 0.5 |
|---|---|---|---|
| E1 buttons, 1024 px | `scenes.py --varied --seed 20260940` (240) | >= 223 | reported |
| E2 buttons, 4096 px | `scenes.py --varied --side 4096 --seed 20260941` (240) | >= 230 | reported |
| E3 rectangles, 1024 px | `rectangles.py --seed 20260942 --n 120` | >= 108 | >= 84 |
| E4 rectangles, 4096 px | `rectangles.py --side 4096 --seed 20260943 --n 60` | >= 54 | >= 42 |

`box`'s default becomes `head` if and only if the head box's IoU >= 0.5 rate
is at least the chain box's on **every** set.

The load was `make start`'s default shape: vision, hq-e8-2b, a 1024-token
prefill chunk, prompt reuse on, the dflash2 drafter at 7, a fresh load every
40 scenes at 1024 px and every 8 at 4096 px (#262). Each scene was asked five
questions, each its own request: `point` (the default, the head set),
`box` with `"method": "head"`, `point` through a second router over the same
engine whose server has no head set (the pointing head alone — spec 13's
answer as a caller gets it, a separate prefill of the same prompt, not the
same pass), and the chain's `point` and `box`. The default point and the
chain point ran in alternating order for timing.

## Result

Every floor holds.

| set | `point` inside: head set / pointing head alone / chain | head `box` / chain `box` IoU >= 0.5 | median distance from the target's centre, per diagonal: set / head alone / chain |
|---|---|---|---|
| E1 | **231** / 227 / 220 of 240 | 102 / 190 | 0.057 / 0.223 / 0.013 |
| E2 | **240** / 234 / 197 of 240 | 238 / 129 | 0.015 / 0.176 / 0.063 |
| E3 | **119** / 79 / 116 of 120 | **111** / 85 | 0.030 / 0.466 / 0.004 |
| E4 | **59** / 53 / 58 of 60 | **53** / 37 | 0.011 / 0.473 / 0.022 |

- **The set fixes the pointing head's corner.** On large objects the
  pointing head alone lands inside 79 of 120 and 53 of 60 (median distance
  near half the diagonal: it reads the corner); the set lands inside 119 and
  59 at 0.03 and 0.01 of the diagonal. On buttons the set's point sits at
  0.06 (1024 px) and 0.015 (4096 px) of the diagonal where the pointing head
  sat at 0.22 and 0.18.
- **At 4096 px the set beats the chain outright** on buttons (240 against
  197; the chain still drifts down at the model's largest grid, as spec 13
  found) and matches it on large objects. At 1024 px the chain is finer on
  what it hits (0.004-0.013 of the diagonal against 0.03-0.06, one image
  token being 32 px) and the set hits more.
- **Against the pointing head alone, scene by scene**, the set turns 9, 6,
  41 and 7 of its misses into hits (E1-E4) and loses 5, 0, 1 and 1 of its
  hits; four of the set's nine misses on E1 are misses of the pointing head
  alone too. Where the other misses land was not studied.

**`box`'s default stays the chain.** The head box's IoU >= 0.5 rate is at
least the chain box's on E2 (99% against 54%), E3 (93% against 71%) and E4
(88% against 62%), and not on E1 (43% against 79%): on 1024 px buttons, 186
of 240 targets are under two token rows tall, and an extent spanned on
32 px cells frames them loosely (57 of those 186 at IoU >= 0.5, the chain
149). By the rule written before the run, `box` answers with the chain unless
asked for `"method": "head"`, which stays the better choice for anything
larger than a couple of tokens and at 4096 px. Seen on the way and not
studied: the chain box misses on many scenes where the chain point hits (31
of its 50 misses on E1), with a corner's leading digit off or the corners
inverted.

**Wall time.** A default point costs what a spec 13 head point cost: 187 ms
asked first and 135 ms asked second at 1024 px, against the chain point's 320
and 270 ms; at 4096 px 6.0 and 2.3 s against 6.2 and 2.5 s, the prefill
dominating.

## The pieces below the endpoint

- **The leaf's argmax is the tap's.** With the tap armed on all nine layers
  the set arms, over the 1024 px fixture (5 scenes) and the 4096 px fixture
  (3), under BF16 and hq-e8-2b, every one of the 96 heads' device argmax
  equals the argmax of the tap's host-side `q . k / 16` over the span minus
  the fallback cells — no near-tie needed the tolerance — and the pointing
  head's row, now written by the fused launch, still equals the tap's to
  2.2e-6 relative (`attention_readout_gpu.rs`).
- **The host rule is the reference's.** `ignis_core::pointing::read_anchored`
  reproduces `ensemble_score.py read_anchored` to 1e-6 px on seven real dumps
  (buttons and rectangles at 1024 px, the owner's Doom screenshot on its
  15x27 grid, a 4096 px rectangle), NumPy's median and its linear quantile
  interpolation (`_lerp`, from above past half the gap) ported as written.
  The reference's own `argmax_excluding` breaks a tie toward the first index
  where the kernel breaks it toward the last (spec 14's rule); on real scores
  no tie came up.
- **Fallback cells on 128x128.** Five 4096 px blank priors
  (`rectangles.py --blank --side 4096 --seed 20260944`, BF16) put 16% of all
  heads' peaks on the first cell and 2.4% each on (1,0) and the last; nothing
  but the first and the last reaches the 5% rule, so the table lists no extra
  cells for that grid. The same `select` run reproduces the 96 heads exactly.
- **Cost.** On the same prompt, prompt reuse off, 25 interleaved repetitions
  and paired differences, arming the set adds 0.21 ms to the 4096 px reading
  chunk (0.40 ms inside the nine armed layers — the microbenchmark said
  0.38-0.40) and 0.15 ms inside the armed layers at 1024 px, where the whole
  chunk's difference (-0.29 ms) is under its noise. The whole-chunk median
  moves by more than a millisecond at 4096 px from run to run, which is why
  the test pairs.

## Owner's screenshots (reported, not floored)

The owner's seven Doom and Wolfenstein captures and their 34 questions (the
study's, boxes eyeballed; local to the owner's clone), through the same
harness: the head set inside **33 of 34**, the pointing head alone 30, the
chain 32 — the study's numbers, through the shipped path. Boxes: the head
box at IoU >= 0.5 on 22 (median 0.54), the chain box on 27 (median 0.79);
the targets are mostly one to three tokens tall, the buttons' regime. The
set's one miss (a vase 26 px wide) is a miss of the pointing head alone too.

## Reproduce

```text
python tools/pointing-scenes/scenes.py --varied --seed 20260940 --out .scratch/head-set-263/E1
python tools/pointing-scenes/scenes.py --varied --side 4096 --seed 20260941 --out .scratch/head-set-263/E2
python tools/pointing-scenes/rectangles.py --seed 20260942 --n 120 --out .scratch/head-set-263/E3
python tools/pointing-scenes/rectangles.py --side 4096 --seed 20260943 --n 60 --out .scratch/head-set-263/E4
IGNIS_POINT_SCENES=<abs path to a set> IGNIS_POINT_OUT=<dir> \
  cargo test -p ignis-server --features cuda --test decide_point_acceptance_gpu \
  -- --ignored --test-threads=1 --nocapture
```

(under the GPU profile's preflight; see `docs/agents/testing.md`). The seeds
are spent: a new acceptance needs new ones.
