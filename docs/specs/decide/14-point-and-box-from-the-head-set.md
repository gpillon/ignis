# 14 - point and box in one pass, from the anchored head set

GitHub: #263

Spec 13 (#260, ADR 0038) made `point` a one-pass answer read from one
attention head, the **pointing head** L39.h10. A study of *how* that head and
its neighbours look at an image
(`docs/findings/2026-09-22-the-heads-outline-the-object.md`) found that the
pointing head reads a fixed **part** of the object — on anything larger than
a couple of image tokens, its bottom-right corner — and that 96 heads from
layer 31 on each read a different part of the object the question asked
for. Read together and filtered around the pointing head's point, they give
the object's centre and its box in the same pass
(`docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`). This
spec makes that the way `/v1/decide` answers `point` by default, and adds a
one-pass `box`.

## Problem Statement

A caller asking `/v1/decide` for a `point` gets, since spec 13, the pointing
head's answer: one prefill, no decode round. It lands inside small labelled
buttons as often as spec 13 promised, but it is not *the centre* of what was
asked for:

- On a large object the point sits on the object's **corner** — the owner
  asked for the lava pool in a Doom screenshot and got its bottom-right
  corner, where the digit chain gave its middle. On 60 large rectangles
  among distractors the pointing head is inside only 40 times (the corner
  cell straddles the edge); the chain 60.
- Its distance from the target's centre is a quarter to a half of the
  target's diagonal (median 0.23 on buttons, 0.46 on large objects), so a
  click lands on the object's rim, not on it.
- When there is nothing to find, it falls back to the image's **last token**
  — the screen's bottom-right corner — with a map so flat (`region.share`
  about 0.03) that it is not an answer at all.

And a caller asking for a `box` still pays for the digit chain: one prefill
and 25 decode rounds, because spec 13 found that one head's region is not a
box.

## Solution

`point` keeps answering in one pass, from the same prefill, but reads **a set
of heads** instead of one:

1. the **pointing head** (L39.h10, unchanged) names *which* object — its
   region rule gives an anchor point and, as today, the answer's confidence;
2. the **head set** — 96 heads in GQA layers 31 to 63, each of which reads a
   different part of the asked object — gives one cell each: where that head
   looks hardest, with the **fallback cells** (where heads go when nothing
   matches) excluded;
3. the cells near the anchor are kept, and their **extent** — the 10%-90%
   span on each axis, grown half a cell — is the object's box; the point is
   its centre.

Measured on scenes never used to choose anything (KV hq-e8-2b, served
render): inside **229 of 240** labelled buttons (pointing head 223, chain
213), **58 of 60** large objects among distractors (40, chain 60), **33 of
34** questions on the owner's Doom and Wolfenstein screenshots (30, chain
32); at 4096 px 9 of 10 buttons (7, chain 7) and 10 of 10 large objects. The
point's distance from the target's centre drops to 0.04-0.09 of its
diagonal. Reading the set costs **0.14-0.4 ms** of GPU per point.

`box` gains `"method": "head"`: the same extent, in the same one pass (IoU
>= 0.5 on 90% of large objects, 65% of the screenshot questions). Whether it
becomes `box`'s default is decided by this spec's acceptance, which measures
the chain's box beside it under a rule written below.

## User Stories

1. As an agent calling `/v1/decide`, I want a `point` to land on the middle of
   the thing I named, so that my click hits the object and not its rim.
2. As an agent working on game or desktop screenshots, I want large targets
   (a monster, a window, a pool of lava) pointed at their centre, so that the
   answer is usable without asking again with the chain.
3. As an agent, I want `point` to keep costing one prefill and no decode
   round, so that locating something stays as cheap as a yes/no question.
4. As an agent clicking small labelled buttons, I want the point inside the
   button at least as often as spec 13's head gave, so that the upgrade costs
   me nothing where the old answer already worked.
5. As an agent working at 4096 px, I want the same behaviour at the model's
   largest grid, so that I do not downscale my screenshots.
6. As a caller, I want the `point` answer to carry the object's **extent** in
   pixels of my image, so that I know how big the thing is and can pick a
   click point of my own inside it.
7. As a caller, I want to ask for a `box` in one pass, so that a bounding box
   costs a prefill instead of 25 decode rounds.
8. As a caller, I want a head `box` in the same pixels and 0-999 scale as the
   chain's, so that switching method changes no code on my side.
9. As a caller, I want every `point` and `box` answer to name the `method`
   that produced it, so that I can tell a one-pass answer from a chain one.
10. As a caller, I want `region.share` kept on head answers, so that I can
    still treat a near-flat map — nothing found — as no answer.
11. As a caller who sends no `method` on `point`, I want the anchored set on a
    load calibrated for it, the pointing head alone on a load calibrated only
    for that, and the chain otherwise, so that I always get the best measured
    answer the load has.
12. As a caller asking `box` with `"method": "head"` on a load without a head
    set, I want a 422 before any GPU time, so that I never get a chain box I
    did not ask for.
13. As a caller sending an unknown `method`, or a `method` on a primitive
    that has one way of being answered, I want the same validation errors as
    today.
14. As a caller asking several questions over one screenshot (fan-out), I
    want head points and head boxes to share the image's prefix like every
    other decision, so that ten questions do not prefill it ten times.
15. As a caller whose state carries no image, I want `state_carries_no_image`
    from head boxes as from head points.
16. As an operator, I want the head set read without a decode lane and
    without residency, like spec 13's point, so that pointing does not compete
    with chat for decode slots.
17. As an operator, I want the added cost of reading 96 heads measured inside
    the prefill, so that I know it stays well under a millisecond.
18. As an operator, I want the load to log whether `point` answers by head
    set, pointing head or chain, and whether `box` accepts `head`, so that I
    know before the first request.
19. As a maintainer, I want the head set and its fallback cells to be a
    calibrated constant keyed to the artifact's content hash beside the
    pointing head, so that a set chosen for one model is never read on
    another.
20. As a maintainer, I want only one score row (the pointing head's) and one
    index per head of the set to cross the `Compute` seam, so that the seam
    stays as narrow as ADR 0034 and ADR 0038 made it.
21. As a maintainer, I want the fused kernel's per-head argmax held to the
    test-only attention tap on the same prompt, so that the device path is
    checked against an independent oracle.
22. As a maintainer, I want the reading rule to be a pure host function with
    golden cases generated by the Python reference, so that the Rust port is
    held to the rule every number was measured with.
23. As a maintainer, I want the mock backend to produce a deterministic
    pointing-head map and head-set argmax, so that the whole path is covered
    by CPU tests (ADR 0006).
24. As a maintainer, I want a job that asks for no attention readout to pay
    nothing, and a job that asks for the pointing head alone to pay what it
    pays today, so that nothing else slows down.
25. As a maintainer, I want a read the leaf cannot make on any armed layer to
    fail the question, never to give a partial set, so that a wrong point
    cannot look like a right one.
26. As a maintainer recalibrating for a new artifact, I want the head-set
    selection written down with the tool that ran it, so that it is a repeat
    and not a study.
27. As a maintainer, I want the attention-head harness to stop failing its hq
    self-check on prompts too short to have codec rows, and to score
    non-square images by their own width and height, so that it can measure
    real screenshots.
28. As a reviewer of the OpenAPI document, I want `extent`, `box`'s `method`
    and the head box's fields described at `/v1`, so that the contract is
    readable without the source (ADR 0036).
29. As the owner, I want the acceptance measured once, on fresh seeds, with
    floors written here before the run, so that the new default is not tuned
    to its own test.
30. As the owner, I want `box`'s default decided by a rule written before the
    measurement, so that the data picks it and not a preference.
31. As the owner, I want the chain's point and box and the pointing head
    alone reported beside the set on the same scenes, so that the gain is a
    number and not a claim.

## Implementation Decisions

- **Vocabulary.** The **pointing head** keeps its name and its job: it names
  which object, and its map's region and share stay the answer's confidence.
  New: the **head set** (the heads read together, one cell each), the
  **fallback cells** (image cells heads peak on when nothing matches), the
  **extent** (the box the kept head-set cells span), and the **anchored
  reading** (the rule below). `CONTEXT.md` gains *Head set*, *Fallback cells*
  and *Extent*, and *Pointing head* and *Attention readout* are rewritten to
  match.

- **`point`'s default reading changes, its method name does not.** `method:
  "head"` keeps meaning "one pass off the calibrated attention"; on a load
  whose calibration has a head set it now means the anchored reading, on a
  load with a pointing head only it is spec 13's reading unchanged, and
  without either `point` answers by chain. Omitted `method` picks the best of
  those the load has. The chain path is unchanged.

- **The answer.** A head `point` carries `pixels` and `normalized` of the
  extent's centre, `uncertainty` of one image cell per axis (the resolution,
  as today), `region` (the pointing head's cells and share, as today) and a
  new optional **`extent`** — `x0`, `y0`, `x1`, `y1` in pixels of the
  submitted image — present only when the head set was read. A head `box`
  carries `method: "head"`, `pixels` of the extent (rounded, clamped to the
  image), `normalized` on the question's `digits` scale, `uncertainty` of one
  cell per edge (a cell's width for `x0`/`x1`, its height for `y0`/`y1`), the
  pointing head's `region`, and no `digits`. A chain `box` gains `method:
  "chain"` and is otherwise unchanged; `digits` becomes optional on `box` as
  it already is on `point`.

- **`box` accepts `method`.** `"chain"` (today's answer) or `"head"` (the
  extent). Omitted, `box` stays the chain until the acceptance's rule below
  says otherwise. `"head"` on a load without a head set is refused at
  validation (422, `pointing_head_unavailable`, before any prefill). Every
  other primitive still refuses `method` (`method_unsupported`), and an
  unknown value is still `method_unknown`.

- **The attention readout carries a head set** — a new ADR (0039) extends ADR
  0038, whose title and first decision say "one attention head". A prefill
  job's readout names, as today, the span (the image's placeholder positions)
  and the pointing head (GQA ordinal, query head), and now optionally the
  **head set** — a list of (GQA ordinal, query head) — and the **excluded
  positions** (the fallback cells, span-relative). The outcome carries, as
  today, one score per key of the span for the pointing head, and now one
  **argmax key index per head of the set**, in the set's order, over the span
  minus the excluded positions. No other row crosses: at 4096 px that is
  64 KB plus 768 bytes, where copying every head's row would be 6.3 MB.
  `ignis_prefill_options` grows by appended fields only (ADR 0016).

- **The leaf reads each armed layer inside its own scope**, exactly where and
  how ADR 0038 reads the pointing head's: right after that layer's attention,
  from the hq prompt route's plane (query rotated into the codec's frame) or
  the BF16 pages. Every GQA layer holding a head of the set or the pointing
  head is armed — layers 31, 35, ..., 63 for the served set. The rules that
  make the pointing head's read possible hold for every armed layer: a single
  band, the prompt route (the scheduler's nine-token tail already keeps the
  reading chunk on it), the image inside the history. If **any** armed layer
  cannot read, the readout is unread and the question fails, as ADR 0038
  fails it; a partial set is never read.

- **One fused kernel launch per armed layer** (ours, not vendored). From the
  prototype (`tools/pointing-scenes/bench_readout.cu`): a grid of
  (key blocks) x (the 4 KV heads); each block rotates the armed query heads
  of its KV head into shared memory once; each warp scores one key row
  against every armed query head sharing that KV head (the row is read once
  per layer, not once per head); each warp keeps its best (score, index) per
  head in registers, the block reduces them, and **one** `atomicMax` per
  block per head publishes it. The pointing head's score row is written in
  the same pass. The first prototype published one `atomicMax` per key per
  head on 16 addresses and was 3-5x slower — do not. The packing that makes
  a 64-bit `atomicMax` an argmax (from the prototype):

  ```cuda
  // order-preserving float -> uint, index in the low half: atomicMax is an argmax
  __device__ unsigned long long pack(float s, uint32_t idx) {
    uint32_t u = __float_as_uint(s);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long)u << 32) | idx;
  }
  ```

  Ties go to the larger index. The per-head results live in the prefill
  chunk's scratch scope (8 bytes per head, zeroed before the first armed
  layer) and are copied to the host once, at the end of the chunk, beside the
  pointing head's scores and behind the same synchronization. The plan's
  readout term grows by that buffer (under 1 KB for 96 heads).

- **The anchored reading runs on the host**, a pure function of the pointing
  head's scores, the set's argmax indices, the grid and the image size.
  Constants are part of the decision. From the Python reference
  (`tools/pointing-scenes/ensemble_score.py`, `read_anchored`, the rule every
  number here was measured with):

  ```python
  cw, ch = width / cols, height / rows
  ax, ay = tag(anchor_scores, rows, cols)          # spec 13's region rule, grid units
  anchor = (ax * cw, ay * ch)
  cells  = [((i % cols + .5) * cw, (i // cols + .5) * ch) for i in head_argmax]
  d      = [dist(c, anchor) for c in cells]
  kept   = [c for c, di in zip(cells, d) if di <= 2.0 * max(median(d), max(cw, ch))]
  x0 = quantile(kept.x, .1) - cw/2 ; x1 = quantile(kept.x, .9) + cw/2   # linear interpolation
  y0 = quantile(kept.y, .1) - ch/2 ; y1 = quantile(kept.y, .9) + ch/2   #   between order statistics
  extent = clamp((x0, y0, x1, y1), image) ; point = centre(extent)
  ```

  The kept set is never empty (at least half the cells are within the median
  distance). Quantiles interpolate linearly between order statistics at
  position `(n - 1) * q` — NumPy's default — and the Rust port must match it.

- **The calibration grows, keyed as before.** The compiled-in table maps an
  artifact **content hash** to the pointing head and, optionally, a head set
  and its fallback cells per grid. The served NVFP4 27B's row
  (`4bdc7b13...6542513`) is recorded in
  `tools/pointing-scenes/pointing-heads-4bdc7b13.json` and reproduced by
  `ensemble_score.py select` from the dumps named there:

  | GQA layer (ordinal) | query heads |
  |---|---|
  | 31 (7) | 1, 10, 12, 17, 23 |
  | 35 (8) | 2, 4, 6, 8, 16, 17, 18, 22, 23 |
  | 39 (9) | 0, 2, 7, 8, 10, 11, 12, 15, 16, 17, 22, 23 |
  | 43 (10) | 1, 3, 5, 6, 7, 9, 14, 15, 17, 18, 19, 20, 22, 23 |
  | 47 (11) | 0, 1, 2, 3, 4, 5, 9, 10, 13, 14, 16, 17, 18, 20, 21, 23 |
  | 51 (12) | 0, 2, 4, 6, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
  | 55 (13) | 0, 2, 4, 6, 7, 8, 9, 10, 11, 13, 16, 18, 20, 22, 23 |
  | 59 (14) | 0, 1, 2, 3, 4, 5, 8, 14 |
  | 63 (15) | 18 |

  96 heads; the pointing head (39, 10) is one of them. Every armed layer but
  59 and 63 touches all four KV heads. Fallback cells: the first and the last
  image cell on every grid, and on the **32x32** grid also cells 1 and 223
  ((0,1) and (6,31)). The **128x128** grid (4096 px) is measured as part of
  this work (`rectangles.py --blank --side 4096`, then `select --blanks`) and
  recorded if it finds more than the first and the last; on any other grid
  only those two are excluded.

- **Selection rule for a head set** (for the table and for recalibration):
  heads at GQA layer 31 or deeper whose median softmax mass over the boxes of
  distractor scenes is at least 0.3, of which at least 0.8 on the asked box.
  Layer 31 is where the question binds to the asked object: layers 7-27 have
  no such head.

- **Failure is a failed question, never a wrong point** (ADR 0038), now for
  every armed layer.

- **Observability** is unchanged: a head box counts as a `box` decision, a
  head point as a `point`. A `method` label stays out (ADR 0017).

- **Documentation**: ADR 0039; `CONTEXT.md` as above; the OpenAPI document
  gains `extent`, `box`'s `method` and the head box; the head answer's doc
  comment stops saying the point sits where the label begins — that was the
  pointing head alone.

- **The attention-head harness** (`attention_head_point_gpu.rs`), the tool
  the head set was chosen with, gets two fixes found by this study: its hq
  consumed-key self-check treats a prompt with no codec rows (a short prompt,
  all rows exact) as nothing to check instead of failing on the median of an
  empty set; and it scores a scene by its own width and height when the
  manifest gives them (`study.size`, or `width`/`height`), including the
  chain's `y`, instead of the square `side`.

## Testing Decisions

A good test asserts what a caller or an operator can observe — the answer,
its method, its extent, the rounds that did not run, the refusals — and holds
the device path to an **independent** oracle, never to itself.

- **Anchored reading (CPU, pure function).** Golden cases generated with
  `ensemble_score.py` from real harness dumps (buttons, large rectangles, a
  screenshot, a 128x128 scene): pointing-head scores, set argmax, grid and
  image size in; point and extent out, to float tolerance. Plus table cases:
  every head on one cell; heads split between the target and a distractor
  far away (the distractor's cells are dropped); an extent past the image
  edge (clamped); a non-square grid and image; ties in the median. Prior art:
  the region-rule tests beside `read_head_map`.

- **Fused argmax against the tap (GPU profile, `attn-tap`).** On the committed
  1024 px pointing fixture and the 4096 px fixture, under BF16 and hq-e8-2b:
  for every head of the set, the leaf's argmax equals the argmax of the tap's
  host-side `q · k / 16` over the span minus the excluded positions (a
  different index is accepted only where the two scores tie to float
  accumulation), and the pointing head's row still equals the tap's. Prior
  art: the leaf-against-tap test #260 added.

- **The decide endpoint over the mock (CPU).** The mock returns a
  deterministic pointing-head map and head-set argmax (documented, like
  today's peak). At `/v1/decide`: a default `point` carries `method: "head"`,
  the extent's centre and `extent`, after zero decode rounds; a load
  calibrated with a pointing head only answers without `extent`; `box` with
  `"method": "head"` answers the extent with `method: "head"` and no
  `digits`; a chain `box` carries `method: "chain"`; `box` `head` on a load
  without a set is a 422 before any prefill; unknown methods and `method` on
  other primitives keep their codes; no image gives
  `state_carries_no_image`; a fan-out mixing head points, head boxes, chain
  boxes and readouts over one image answers all of them. Prior art: the
  decide endpoint and fan-out tests over the mock (#239, #240, #260).

- **Nothing for everyone else.** A prefill job with no readout allocates and
  launches nothing new; a job with the pointing head and no set launches
  what it launches today (one layer armed).

- **Cost in the prefill (GPU).** The reading chunk's GPU time with the set
  armed against the pointing head alone, same prompt, via the chunk profile
  (`IGNIS_CHUNK_PROFILE`), at 1024 and 4096 px.

## Acceptance

Closed against these, in one piece of work: the seam, the fused leaf under
both KV formats, the host rule, the endpoint and the acceptance are one
design and are not split.

1. **The leaf's reads are the attention's.** Under BF16 and hq-e8-2b, on the
   1024 px and 4096 px pointing fixtures, the fused kernel's argmax for every
   head of the set equals the tap's (ties to float accumulation excepted) and
   the pointing head's scores still equal the tap's; if any armed layer
   cannot read, the question fails.
2. **The anchored reading equals the reference.** The host rule reproduces
   `ensemble_score.py read_anchored` on the golden cases, and its table cases
   pass on CPU.
3. **`/v1/decide` over the mock** answers as Testing Decisions lists: head
   `point` with `extent` after zero decode rounds; head `box`; chain `box`
   with `method`; the refusals; the fan-out.
4. **Calibration.** The table maps the served artifact's content hash to the
   pointing head, the 96-head set above and its fallback cells (32x32 as
   above; 128x128 measured and recorded); the load logs what `point` and
   `box` will answer with; the GPU-profile artifact test still fails, naming
   the procedure, when the artifact changes.
5. **Cost.** Arming the set adds at most **0.5 ms** of GPU time to the reading
   chunk at 1024 px and at most **1 ms** at 4096 px (microbenchmark:
   0.14-0.19 and 0.38-0.40 ms), and nothing to a job that asks for no
   readout.
6. **The pre-registered acceptance holds** — measured once, through
   `/v1/decide`, served artifact, hq-e8-2b with the residual window, `point`
   with no `method`, sets never used before (seeds reserved in
   `tools/pointing-scenes/README.md`):

   | set | command | `point` inside | head `box` IoU >= 0.5 |
   |---|---|---|---|
   | E1 buttons, 1024 px | `scenes.py --varied --seed 20260940` (240) | **>= 223** | reported |
   | E2 buttons, 4096 px | `scenes.py --varied --side 4096 --seed 20260941` (240) | **>= 230** | reported |
   | E3 rectangles, 1024 px | `rectangles.py --seed 20260942 --n 120` | **>= 108** | **>= 84** |
   | E4 rectangles, 4096 px | `rectangles.py --side 4096 --seed 20260943 --n 60` | **>= 54** | **>= 42** |

   E1's floor is the set's 229 on T1' minus the slack of six every earlier
   floor used; E2's is spec 13's floor for the pointing head alone, which
   the set must not fall below; E3 and E4 are the pre-registered 90% and 70%
   that T2' passed at 97% and 90%. Reported beside each, not asserted: the
   pointing head alone (its scores come back with the set's), the chain
   `point`, the chain `box` (with `"method": "chain"`), each method's median
   distance from the target's centre, and the wall time of a head point and a
   chain point on the same prompt; and the owner's screenshot questions if
   the clone has them (`.scratch/vision-study/screens/`).
   **`box`'s default:** it becomes `head` if and only if the head box's
   IoU >= 0.5 rate is at least the chain box's on **every** one of E1-E4;
   otherwise it stays the chain and `head` stays opt-in. Recorded as a
   finding with a README row, whichever way it goes.
7. The harness fixes (short-prompt hq self-check, non-square scoring) are in,
   and ADR 0039, `CONTEXT.md` and the OpenAPI document are updated.

## Out of Scope

- ~~**The Playground's Decide tab** drawing the extent and offering `method`
  on `box`.~~ Shipped: `decide-ui-260-256` merged into this branch, and the
  tab now offers both methods on both primitives — naming the two defaults
  apart, because a `point`'s absent `method` follows the load and a `box`'s
  follows `BOX_DEFAULT_METHOD` — draws a head point's extent as a dashed
  outline with the crosshair at its centre and writes its corners beside the
  table, and outlines a head box's per-edge band rather than fading it out,
  since that figure is the reading's resolution and not a spread.
- **A learned combination of heads.** A ridge map over all 384 heads per cell
  has a real signal but, fitted on synthetic scenes, does not transfer to
  real screenshots (20 of 34 inside); it needs annotated real images first.
- **Refusing on low `region.share`** (answering "not found" instead of the
  fallback's point): the share is exposed, a threshold needs its own
  measurement on scenes where the object is absent.
- **Where between layer 27 and 31 the question binds** (a residual-stream
  probe; the GDN layers have no attention map), and why cell (6,31) is a
  fixed peak — research, not needed to ship.
- **`box` in two passes** (the earlier plan with the head as pass 1): the
  one-pass box here is measured first.
- **A `method` label on the decision metrics** (ADR 0017), **multi-image
  states**, and **an exact copy of the pointing head's KV head** (spec 13's
  reasons stand).

## Further Notes

- **Where the numbers come from.** Every table in this spec is in
  `docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`, and
  the understanding behind it (parts, boundary, fallback, binding at layer
  31, mirrored images) in `2026-09-22-the-heads-outline-the-object.md`. The
  pre-registrations, the per-scene dumps and the round-by-round results are
  raw material in `.scratch/vision-study/` of the clone that ran them.
- **The anchor decides which object.** The set's misses are the pointing
  head's misses; without the anchor, about a fifth of the set (colour-only
  heads such as L59.x, L63.h18, L55.h11) pulls the extent onto a distractor
  of the right colour and buttons fall to 194 of 240.
- **Why the rule is not simpler.** Averaging the set's maps does not give the
  centre (the set is top-right heavy: its mean map's centroid sits at
  (0.6, 0.33) of the object); the extent of the kept cells does, because
  each head marks a boundary part and the quantiles span them.
- **Fallback cells matter a little.** Excluding only the first and last cell
  on the 32x32 grid costs one large object in 60 and 2-4 points of box rate;
  the extra cells are measured, not guessed, which is why the table carries
  them per grid.
- **Scale.** Each armed layer's plane is freshly written by its own attention
  and is in L2 when the read runs at 1024 px; the kernel is bound by reading
  4 KV heads' keys once per layer (2.6 MB at 1024 tokens, 34 MB at 16,384).
- **Screenshots.** The owner's seven Doom and Wolfenstein captures and the 34
  questions (with boxes eyeballed and checked on drawn copies) are local to
  the owner's clone; they are a report, not a floor.
- **Recalibration** of the set for a new artifact: `tools/pointing-scenes/README.md`,
  "Recalibrating the head set", after the pointing head's own procedure.
