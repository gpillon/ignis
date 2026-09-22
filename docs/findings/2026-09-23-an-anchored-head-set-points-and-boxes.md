# An anchored set of 96 heads points better than the pointing head and the chain, and draws a box, in one pass

- Kind: experiment
- Status: current
- Observed: 2026-09-23
- Last verified: 2026-09-23
- Scope: serving / `/v1/decide` `point` and `box` read from attention in one pass; kernel / attention readout cost
- Related: https://github.com/gpillon/ignis/issues/263, [The heads outline the object](2026-09-22-the-heads-outline-the-object.md), [spec 13](../specs/decide/13-point-by-attention-head.md), [spec 14](../specs/decide/14-point-and-box-from-the-head-set.md), [ADR 0038](../adr/0038-the-seam-carries-one-attention-head.md), `tools/pointing-scenes/` (`ensemble_score.py`, `rectangles.py`, `pointing-heads-4bdc7b13.json`, `bench_readout.cu`)
- Superseded by: none

## Question

The heads that find the asked object each read a different part of it
([the previous finding](2026-09-22-the-heads-outline-the-object.md)). Read
together, do they give a point at the object's centre and its box in the one
pass `point` already runs — and what does reading them cost?

## Evidence

Harness dumps of `crates/server/tests/attention_head_point_gpu.rs` (all 384
heads, served render, the query after `{"x":`), scored on the CPU. Every test
below was pre-registered — rule frozen, criteria written — before its data
was generated (`.scratch/vision-study/PREREG-3.md`, `-4`, `-5`); the numbers
reproduce with `tools/pointing-scenes/ensemble_score.py score` and the
recorded set `pointing-heads-4bdc7b13.json`.

**The head set.** The 96 heads at layer 31 and deeper that put at least 0.8
of their mass over the boxes on the asked one (and at least 0.3 on the boxes
together), measured on 12 scenes of a red rectangle beside a blue one under
BF16: 5, 9, 12, 14, 16, 16, 15, 8 and 1 heads in layers 31 to 63.
**Fallback cells** — where heads peak when nothing matches (blank priors) —
are excluded from every head's argmax: the first and last image cell, and on
the 32x32 grid also (0,1) and (6,31).

**Unanchored (R2):** the 10%-90% extent of the 96 heads' argmax cells, its
centre the point. Pre-registered, hq: large rectangles among distractors
(T2, `rectangles.py --seed 20260931`) inside 59 of 60, box IoU >= 0.5 on 95%
— pass; labelled buttons (T1, `scenes.py --varied --seed 20260930`) inside
194 of 240 against the pointing head's 227 — **fail**. About a fifth of the
set is colour-selective but not label-selective (L59.x, L63.h18, L55.h11 hit
the button on 4-20% of scenes) and the extent swallows the distractor they
land on.

**Anchored (R4):** the pointing head (L39.h10, TAG's region rule) names the
object; only head cells within twice the median distance to its point are
kept; the box is their 10%-90% extent grown half a cell, the point its
centre. The radius was chosen among 1.5, 2 and 3 on T1/T2 and the earlier
rounds, then frozen, and tested on fresh sets. KV hq, served render:

| set | pointing head alone | **anchored set** | chain | anchored box IoU >= 0.5 |
|---|---|---|---|---|
| T1' buttons, 1024 px (`scenes.py --varied --seed 20260932`, 240) | 223 | **229** | 213 | 37% (median 0.45) |
| of which named by label (126) | 115 | **123** | 112 | |
| T2' rectangles among distractors, 1024 px (`rectangles.py --seed 20260933`, 60) | 40 | **58** | 60 | **90%** (median 0.73) |
| owner's Doom and Wolfenstein screenshots (7 images, 34 questions) | 30 | **33** | 32 | 65% (median 0.55) |
| buttons, 4096 px (`scenes.py --varied --side 4096 --seed 20260934 --n 10`) | 7 | **9** | 7 | 90% (median 0.85) |
| T2' scenes 0-9 upscaled to 4096 px | 8 | **10** | 10 | 100% (median 0.94) |

Under BF16 the anchored set is 236 of 240 and 60 of 60 on T1' and T2'.
Distance of the point from the target's centre over the target's diagonal
(median, hq): 0.23 -> 0.06 on T1', 0.46 -> 0.04 on T2', 0.21 -> 0.09 on the
screenshots (chain 0.02, 0.004, 0.06). All three pre-registered criteria for
R4 passed (T1' >= pointing head - 3; T2' inside >= 90%; T2' box IoU >= 0.5
on >= 70%), and so did the screenshot and 4096 px predictions.

**A learned linear combination (LP),** the owner's alternative: ridge
regression per image cell from the 384 z-scored head scores to the fraction
of the cell the asked object covers, fitted on the development scenes only.
The signal is real — with the targets permuted across scenes it collapses
(buttons 1-2% inside) — and in the development domain it fills the object
where single heads outline it. It does not transfer: on the screenshots it
is inside 20 of 34 (24 anchored on the pointing head), against the anchored
set's 33. Its largest weights are in layers 3-19, heads that find objects
without telling them apart, and on flat synthetic shapes that is enough.

**Cost** (`tools/pointing-scenes/bench_readout.cu`, RTX 5090, the engine's
readout kernel verbatim and a fused per-layer variant, the set's real heads
per layer, one copy to the host at the end as the engine makes it):

| image tokens | pointing head today | 96 launches of today's kernel | fused per layer, argmax on the device |
|---|---|---|---|
| 1,024 (1024 px) | 0.04-0.05 ms | 0.80-1.12 ms | **0.14-0.19 ms** |
| 4,096 (2048 px) | 0.04-0.05 ms | 1.05-1.19 ms | **0.20-0.25 ms** |
| 16,384 (4096 px) | 0.04-0.05 ms | 1.60-1.63 ms | **0.38-0.40 ms** |

Against a point's measured wall time (143 ms at 1024 px, 2,337 ms at 4096 px
on the second question over an image,
[the head points through decide](2026-09-22-the-head-points-through-decide.md))
the fused read adds about 0.1% and 0.02%. What crosses to the host is 8 bytes
per head beside the pointing head's row (5-66 KB), against 0.4-6.3 MB if
every head's row were copied. The first fused version made one `atomicMax`
per key per head on 16 addresses and cost 3-5x more; reducing within the
block first fixed it.

## Finding

Observed:

- Anchored on the pointing head, the selective heads give a point closer to
  the target's centre than the pointing head (median 0.04-0.09 of the
  diagonal against 0.21-0.46) and inside the target at least as often as
  the pointing head and the chain on every set measured, at 1024 and 4096 px
  and on real screenshots — in the same one pass, for well under a
  millisecond.
- The same read yields a box: IoU >= 0.5 on 90-100% of large objects, 65%
  of the screenshot questions and 37-46% of buttons (a button is about one
  token tall, the extent's resolution).
- The anchor is what makes it work: unanchored, colour-only heads pull the
  extent onto a distractor of the right colour.
- A linear map learned on synthetic scenes does not generalize to real
  images; the anchored set, which learns nothing but which heads to read,
  does.

Inferred:

- The pointing head's misses are the anchored set's misses: it decides which
  object. On the screenshots the one miss left (a 26x48 px vase) is the
  pointing head's.

## Implications

- `point` should be read from the anchored head set instead of the pointing
  head alone, and `box` can be answered in the same one pass (spec 14).
- The readout seam must carry a set of heads, which ADR 0038 ("one attention
  head") does not allow as written.
- Learning head weights is worth revisiting only with annotated real images.

## Limits and unknowns

- The synthetic scenes are flat colours and one generator's buttons; 4096 px
  is 10 + 10 scenes, and the rectangles there are upscaled, not native.
- The screenshots are 34 questions on 7 images with boxes eyeballed by the
  person who scored them; the images stay local to the clone that ran it.
- The head set was chosen on 12 scenes of one kind (red beside blue);
  fallback cells are measured only on the 32x32 grid.
- The cost is a microbenchmark of the kernel, not measured inside the
  prefill; the fused kernel's argmax was not checked against the naive one.
- The chain's box was not measured, so the box numbers have no chain
  comparison yet.

## Follow-ups

- Spec 14 (`docs/specs/decide/14-point-and-box-from-the-head-set.md`) and
  its GitHub issue.
