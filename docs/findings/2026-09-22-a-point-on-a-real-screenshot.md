# A point on a real screenshot costs 60-90 ms over HTTP, and the head is the cheap half of it

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / `/v1/decide` `point`, one-pass pointing (ADR 0038), end-to-end latency over HTTP
- Related: `docs/specs/decide/13-point-by-attention-head.md` (#260),
  `2026-09-22-the-head-points-through-decide.md` (the acceptance, in process),
  `2026-09-22-how-the-head-map-is-read.md` (where the head's point lands),
  `.scratch/decide-point-e2e/` (the two scripts, `phases.json`, the marked
  images -- raw, on disk)
- Superseded by: none

## Question

The acceptance measured `point` in process, on synthetic scenes, against
floors. What an agent actually meets is different: a real screenshot, one
question, an HTTP round trip to a release build. How long does that take, and
what does the head's one pass buy once the whole request is counted?

Not an acceptance: two images, one target, no pre-registration. It is an
illustration of the shipped path at a size the study never used.

## Method

`make start VISION=1 MAX_CONTEXT=131072` on the served NVFP4 27B artifact
(hq-e8-2b, 1024-token prefill chunk, prompt reuse on, dflash2 at 7), release
build, GPU otherwise idle. `MAX_CONTEXT` was cut from the Makefile's 262144
because the desktop's own 3.7 GiB left the VRAM plan 27 MB short; nothing in
a 530-token prompt touches the difference.

A local Python client posts `/v1/decide` with the screenshot as one
`image_url` part and one `point` question, and times the round trip with
`perf_counter`. Two images the owner supplied: an 850x478 WebP frame of Doom
and the same frame at 640x359 as a PNG. The question is "Where is the
player's face?" -- the status-bar portrait.

Three phases, because the cost is not one number:

- **A, first sight**: the first request carrying that image. The vision
  encode and the whole prompt's prefill are in it.
- **B, the same question again**: the prefix is retained; nothing is new.
- **C, a different question about the same image**: five further questions
  (the ammo counter, the health percentage, the armour value, the weapon, the
  lava), so the image's prefix is shared and only the question's tail is
  fresh. This is the fan-out an agent does.

Each phase both ways: no `method` (this load answers by head) and
`"method": "chain"`.

## What it costs

| image | method | A first sight | B same question | C a new question |
|---|---|---|---|---|
| 850x478, 530 prompt tokens | head | **506 ms** | 92 ms | 83 ms |
| | chain | 244 ms * | 168 ms | 174 ms |
| 640x359, 345 prompt tokens | head | 120 ms * | 68 ms | **60 ms** |
| | chain | 190 ms * | 194 ms | 200 ms |

`*` not a first sight: that image's encode was already paid by the row above
it. Only the 850x478 head row's **506 ms** is a genuinely new image on a warm
server.

So the two numbers worth quoting are **~500 ms for the first question about a
screenshot the server has never seen**, and **60-90 ms for every question
after it**. The head generates **0 tokens**; the chain generates 9.

**What the one pass buys, on the same prompt**: 76-91 ms at 850x478 and
126-140 ms at 640x359 -- the chain's nine decode rounds. On a repeated
question that is 45% of the answer at the larger size and 65% at the smaller,
against the 2-50% the in-process acceptance measured on synthetic scenes at
1024 and 4096 px. The saving is absolute and the prompt here is small, so it
is a larger share of a smaller total.

The chain is *slower on the smaller image* (200 against 174 ms) while the
head is faster (60 against 83). The head's half of that is the shape to
expect -- its cost is the prefill and follows the token count -- and the
chain's is not explained here: nine decode rounds whose count does not follow
the prompt, and the digits it writes differ between the two images, so
drafter acceptance is a plausible cause and is not measured.

**Cost against size**, head only, the same frame re-encoded at eight widths,
median of six warm requests each:

| width | prompt tokens | median |
|---|---|---|
| 1280 | 1,005 | 138 ms |
| 1024 | 701 | 90 ms |
| 850 | 530 | 97 ms |
| 768 | 461 | 88 ms |
| 704 | 389 | 69 ms |
| 640 | 345 | 72 ms |
| 512 | 269 | 64 ms |
| 426 | 229 | 44 ms |

Roughly 0.1 ms a prompt token over a ~20 ms floor. The head found the face at
seven of those eight widths; at 512 px it sits about 3 px left of the face's
edge, on the panel bezel -- a tenth of a cell out, and outside all the same.

## Where the points land

Both methods land **inside the face** on both images, and they land in
different parts of it:

| image | head | chain |
|---|---|---|
| 850x478 (face ~x 402-448, y 402-471) | (425, 462) -- the mouth and chin | (423, 439) -- the centre, on the nose |
| 640x359 (face ~x 303-337, y 303-355) | (304, 343) -- the left jaw, on the face's edge | (320, 325) -- the brow and eyes |

The head's answer is one cell -- `uncertainty` 31.5 x 31.9 px at 850x478 and
32.0 x 32.6 at 640x359 -- with `region.cells` 1 and `region.share` 0.47 and
0.42. The chain's `uncertainty` is 13.4 x 3.7 px, and its trace shows the
place it is least sure of (0.43 and 0.17 on the last digit of each axis).

The head sitting on the target's **edge** rather than its centre is the
behaviour spec 13 documents for labelled targets, on a target that carries no
label. It matters more here than on the study's buttons: the cell is ~32 px
of the *submitted* image at both sizes, so the face is 1.4 cells wide at
850x478 and 1.1 at 640x359 -- the head has barely one cell on the target
either way, and half a cell of offset is the difference between the jaw and
the bezel beside it.

## What it means

- **Downscaling a screenshot does not buy the head anything and costs it
  resolution.** It saves 20-30 ms of prefill and shrinks the target in cells
  one for one, because a cell is a fixed ~32 px of whatever was submitted. An
  agent should send the screenshot it has.
- **The first question about an image is 5-6x the ones after it** (506 ms
  against 83-92 ms, the one row that is a true first sight), so a
  caller with several questions about one screenshot should send them as one
  fan-out, or at least over one retained prefix, rather than as separate
  requests.
- **The head's saving is the chain's decode rounds and nothing else**, which
  is the same conclusion the in-process acceptance reached; what is new is
  that on a small prompt those rounds are the majority of the answer.
- **A face is not a labelled button**, and n here is two images and one
  target. What this measures well is the latency; the two points landing
  inside are an illustration, not a rate.
