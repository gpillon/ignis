# The head points through `/v1/decide`: spec 13's acceptance holds at both sizes

- Kind: experiment
- Status: current
- Observed: 2026-09-22
- Last verified: 2026-09-22
- Scope: serving / `/v1/decide` `point`, the attention readout (ADR 0038), one-pass pointing; hq-e8-2b KV
- Related: `docs/specs/decide/13-point-by-attention-head.md` (#260),
  ADR 0038, `crates/server/tests/decide_point_acceptance_gpu.rs` (the run),
  `crates/server/tests/attention_readout_gpu.rs` (leaf against tap),
  `tools/pointing-scenes/` (the generator),
  `2026-09-22-how-the-head-map-is-read.md`,
  `2026-09-22-the-codec-costs-the-head-its-read.md`,
  `.scratch/pointing-acceptance/point-acceptance-scenes-d{,4096}.json` and
  `run-d.log`, `run-d4096*.log` (raw, on disk), #262
- Superseded by: none

## Question

Spec 13 ships L39.h10 as the default way `/v1/decide` answers `point`: the
prefill that forces `{"x":` reads the head's scores over the image in the
leaf, and the host turns the map into a point. Every earlier number came from
a test-only tap. Does the shipped path — HTTP, the served render, the
scheduler's chunking, the leaf's own readout — clear the floors spec 13 wrote
down before any run, on scenes never used to choose the head or the rule?

## Pre-registration

In spec 13 (commit c7ceaf7, before this run): sets **D** (240 varied scenes,
1024 px, `scenes.py --varied --seed 20260925`) and **D4096** (240, 4096 px,
seed 20260926), asked through `/v1/decide` with no `method`, the served
artifact, hq-e8-2b with the residual window. The head inside the target on
**at least 221 of 240 at 1024 px and 230 of 240 at 4096 px** — the engine's
227 and 236 on set C and C4096 minus six. Reported, not asserted: the chain
on the same scenes, both methods' distance from the target's centre, and the
wall time of a head point and a chain point on the same prompt.

The load was `make start`'s default shape: vision, hq-e8-2b, a 1024-token
prefill chunk, prompt reuse on, the dflash2 drafter at 7. Each scene was
asked both ways, in alternating order (even scenes head first), because the
second question over an image finds its embedding already encoded.

## Evidence

**The leaf reads what the tap reads.** Before the run, the leaf's scores were
held to the tap's host-side `q · k / 16` on the same prefill, driven by the
scheduler over the real runtime (`attention_readout_gpu.rs`): worst
difference 0.7-1.1e-6 relative under BF16 and 1.4-2.2e-6 under hq-e8-2b, on
the three 4096 px fixture scenes and five 1024 px scenes, and the region rule
read the same point from both, to the cell's thousandth.

**The acceptance.**

| | D (1024 px) | D4096 |
|---|---|---|
| head inside | **228 / 240** | **237 / 240** |
| pre-registered floor | 221 | 230 |
| chain inside | 216 | 182 |
| both inside / head only / chain only / neither | 207 / 21 / 9 / 3 | 179 / 58 / 3 / 0 |
| head inside, colour / label targets | 110 of 115 / 118 of 125 | 117 of 119 / 120 of 121 |
| median max-axis distance from the centre, 0-999 (head / chain) | 39.0 / 2.0 | 27.6 / 13.9 |
| the same, hits only | 38.0 / 2.0 | 27.6 / 10.7 |
| head median signed error x, y | −35.1, 0.0 | −25.4, −3.7 |
| chain median signed error y | −1.0 | +13.8 |

**Both floors are met.** The head lands inside on 228 and 237 of 240, one
above each of the engine's own set-C and C4096 numbers (227, 236).

**The chain drifts at 4096 px, as measured before.** 182 of 240, with a
median of +13.8/999 down on y; 58 scenes are inside for the head alone and 3
for the chain alone. At 1024 px the chain is centred (−1.0 on y).

**Where the head's point sits.** On the hits, at a median 28% across the
button at 1024 px (quartiles 24-32%) and 34% at 4096 px (30-37%), near its
vertical middle: the start of the label, as the map-reading finding found.
That is the whole of its distance from the centre. `uncertainty` is one token
cell, 32 px on both axes of these unresized images. The region is one cell on
165 and 137 of 240 scenes.

**The share is a confidence.** The region's share of the head's attention
over the image is a median 0.290 on hits against 0.173 on misses at 1024 px
(AUC 0.80), and 0.152 against 0.052 at 4096 px (AUC 0.95, three misses).

**The committed fixture has one of those misses.** Under hq-e8-2b the head
reads the 4096 px `large` scene at (1520, 3312), left of a button spanning
x 2600-3700, with a region share of 0.025; under BF16 it lands inside, and it
lands inside `medium` and `small` under both. The leaf and the tap agree on
that point to the thousandth of a cell, so it is the model's map under the
codec, not the readout. `decide_point_gpu.rs` therefore holds the head to at
most one miss in three and no miss at a share of 0.10 or more, rather than to
every scene.

**Wall time, head against chain on the same prompt** (median, in process,
the test build):

| | first question over the image | second question over it |
|---|---|---|
| 1024 px, head / chain | 198 / 339 ms | 143 / 285 ms |
| 4096 px, head / chain | 6,066 / 6,220 ms | 2,337 / 2,499 ms |

The head saves the chain's decode rounds — ~140-160 ms at either size. At
1024 px that is 42-50% of a question; at 4096 px the ~2.3 s prefill of 16.4K
tokens and the vision encode on the first question (~3.7 s) dominate, and the
saving is 2-7%.

**How the run went.** D ran in one load. D4096 took three attempts with no
code change in between that touches what a head point reads: the first load
died of host memory at scene 86 because the scheduler never drops a finished
request, whose input holds a 4096 px image's ~200 MB of patch rows (#262);
the second, reloading every 40 scenes, died at scene 99 when a concurrent
workspace build exhausted the machine's commit limit; the third reloaded
every 20 scenes with nothing else running and completed. The first two
attempts' partial outcomes were visible in their logs before the third ran.

## Finding

- **Spec 13's acceptance holds**: the one-pass head, through `/v1/decide`
  with no `method`, lands inside the target on 228 of 240 at 1024 px and 237
  of 240 at 4096 px against pre-registered floors of 221 and 230. The chain
  lands 216 and 182.
- **The leaf's readout is the tap's**: the production path's scores equal
  the test-only oracle's to about one part in a million under both KV
  formats, so every tap-measured number carries over.
- **The head is coarser and better placed than the chain at once**: inside
  more often at both sizes, but 28-39/999 from the centre where the chain's
  hits are 2-11, because it marks the start of the label.
- **What one pass buys is the decode rounds**: ~150 ms a point, roughly half
  a 1024 px question, a few percent of a 4096 px one.

## Limits and unknowns

- Synthetic scenes from one generator, with labelled buttons; at 4096 px the
  generator draws the 1024 px scene four times larger, so a real 4K screen
  with small UI is unmeasured for both methods.
- Wall times are from the test build (opt-level 2) over in-process HTTP, not
  a release server; the difference between the methods is device time and
  should carry, the absolute figures less so.
- The share's AUC at 4096 px rests on three misses.
