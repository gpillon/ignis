# The heads outline the object, each reading its own part, and bind to the question from layer 31

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-23
- Scope: model / how the served 27B's attention reads image tokens at the `point` scaffold; `/v1/decide` pointing
- Related: https://github.com/gpillon/ignis/issues/263, [spec 13](../specs/decide/13-point-by-attention-head.md) (GitHub #260), [ADR 0038](../adr/0038-the-seam-carries-one-attention-head.md), [How the head map is read](2026-09-22-how-the-head-map-is-read.md), [An anchored head set points and boxes](2026-09-23-an-anchored-head-set-points-and-boxes.md), [spec 14](../specs/decide/14-point-and-box-from-the-head-set.md)
- Superseded by: none

## Question

The owner noticed that the pointing head (L39.h10) answers "almost always the
bottom-right corner". Is that a bug, a positional bias, or something the
model does — and what does it say about how this model's vision represents
an object?

## Evidence

All attention numbers are from `crates/server/tests/attention_head_point_gpu.rs`
with every GQA layer armed (16 layers x 24 query heads = 384 maps per scene),
served render, query at the position after the forced `{"x":`, KV under
BF16 **and** hq-e8-2b (the two agree: L39.h10's argmax identical on 32 of 49
scenes, map correlation median 0.93, the same picture everywhere). The
competing hypotheses and what each predicts were written down before the
dumps were read. Raw material, scripts and per-round results:
`.scratch/vision-study/` (`PREDICTIONS*.md`, `RESULTS*.md`, `dumps/`), local
to the clone that ran it.

- **The owner's case, live server:** the Doom screenshot
  (`A-screenshot-of-Doom.webp`, 850x478), "dove sta lava pool" and three
  rephrasings: the head answers (582, 303) every time, the chain (405-408,
  274-276). The pool spans about x 238-576, y 236-318.
- **Rectangles, live server:** a red rectangle on grey, 4 shapes x 5
  places. The head lands on a corner or an edge of the rectangle 20 of 20
  times (15 at its bottom-right, relative position (0.95-1.25, 0.53-0.98));
  the chain lands at the centre (0.50 +- 0.02) 20 of 20. A small disc (about
  2 cells): both inside 9 of 9.
- **Flips of Doom** (orig / hflip / vflip / rot180, 4 questions each): L39.h10's
  peak on the lava mapped back to the original image's cells:

  | image as the model saw it | peak as shown | same peak in the original |
  |---|---|---|
  | original | bottom-right of the pool (r9, c18) | (r9, c18) |
  | mirrored left-right | bottom-**left** of the pool | (r9, c17) |
  | mirrored top-bottom | **top**-right of the pool | (r9, c17) |
  | rotated 180 | diffuse map, peak moves | (r7, c11) / (r8, c8) |

- **Boundary against interior** (15 large filled objects, mass per cell on
  the object's boundary ring over its interior): L39.h10 3.4x, the 8 heads
  with the most mass on the object 2.8x. With the interior textured (noise,
  checker, stripes, gradient; 12 scenes) L39.h10 stays at 1.7-6.5x and the
  top 8 at 2.5-12.7x.
- **Which part each head reads** (median peak relative to the object, 0 =
  left/top, 1 = right/bottom; heads with the most mass on large objects):

  | part | heads |
  |---|---|
  | top-left corner | L19.h5 (0.02, 0.02) |
  | top-right corner | L51.h19, L51.h16, L47.h20, L55.h7, L55.h20, L59.h0, L35.h6 |
  | left edge, upper half | L39.h23 (0.08, 0.31), L43.h14 (0.08, 0.19) |
  | top edge, middle | L51.h22 (0.52, 0.27) |
  | bottom-right corner | **L39.h10** (0.96, 0.97; v >= 0.8 on 15 of 15 under BF16) |

  The mean map of the top k heads has its centroid at (0.60-0.65,
  0.32-0.35): averaging does not give the centre.
- **Size sweep** (a square at the image centre, 32-768 px): from 128 px up
  L39.h10's peak is the square's bottom-right cell every time. On a target
  of one or two cells a corner and the centre are the same cell.
- **Nothing to find** (grey, black, white, noise and gradient fields, asked
  for a red rectangle): L39.h10 peaks on the **last** image cell (r31, c31)
  on 5 of 6, with the peak cell holding only 0.02-0.04 of the softmax.
  Over all 384 heads the commonest peaks are (r0, c0) 13%, (r31, c31) 10%,
  (r0, c1) 9% and (r6, c31) 6% — the last one also lights up in maps of
  scenes that have an object, and on noise.
- **Two identical discs:** the one met first in raster order gets more
  of L39.h10's mass on 4 of 4 (0.22-0.24 against 0.12-0.18); the chain also
  picks the first on 3 of 4.
- **When the question binds to the object** (12 scenes, a red rectangle asked
  for beside a blue one; heads with at least 0.8 of their mass over the two
  boxes on the asked one, and at least 0.3 on the two):

  | GQA layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | **31** | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | selective heads (BF16) | 2 | 0 | 0 | 0 | 0 | 0 | 0 | **5** | 9 | 12 | 14 | 16 | 16 | 15 | 8 | 1 |

  hq gives the same row to within two heads per layer. L19.h5 puts 0.81 of
  its mass on the two boxes and 0.71 of that on the asked one: it finds
  objects without telling them apart. L39.h10 is at 0.92-0.93 on every scene.

## Finding

Observed:

- The "bottom-right corner" has two sources. On a large object it is the
  **object's** corner — the part of the object L39.h10 reads. When there is
  nothing to find it is the **image's** last token, with a near-flat map
  whose `region.share` (about 0.03) already says so.
- The peak follows **content**, not token order and not rotary distance to
  the question: mirroring the image moves it with the pool in both
  directions, and mirrored top-bottom it sits at the top of the image.
- Heads that find the object put most of their mass on its **boundary** —
  corners and edges — whatever fills the interior, and each reads a **fixed
  part** of it. The object's extent is spread across heads; no head reads its
  centre.
- Heads from layer 7 to 27 find objects but not *the asked* object; from
  layer 31 on most of them are selective, and layer 3 has two colour-matching
  heads. Between layer 27 and 31 sit three GDN layers, which have no
  attention map to read.

Inferred:

- The digit chain's centre is **computed** while it writes the digits; it is
  not stored at any image token a head could read.
- A pointing head calibrated on buttons about one token tall cannot show
  where on an object it looks; that is why spec 13's acceptance, all on such
  buttons, never saw the corner.
- Several heads read together should recover the extent — the premise of
  [the anchored head set](2026-09-23-an-anchored-head-set-points-and-boxes.md).

## Implications

- A `point` read from one head is a good "which object" and a poor "where is
  its centre" on anything larger than a couple of tokens.
- `region.share` from the pointing head separates "found it" from "fell back
  to the last token"; a caller should treat a share near 0.03 as no answer.
- The first and last image tokens (and, on the 32x32 grid, (0,1) and (6,31))
  are where heads go when nothing matches; a reading that combines heads
  should exclude them.

## Limits and unknowns

- Synthetic flat shapes and one real screenshot; per-group counts are small
  (4-20 scenes). Doom's boxes are eyeballed.
- The rot180 map going diffuse is unexplained.
- Why (6,31) is a fixed peak is not measured (a ViT "register"-like token is
  a hypothesis; key norms would test it).
- Where between layer 27 and 31 the binding happens needs a probe on the
  residual stream (the GDN layers have no map).

## Follow-ups

- Spec 14 (`docs/specs/decide/14-point-and-box-from-the-head-set.md`) reads
  the selective heads together, anchored on the pointing head.
