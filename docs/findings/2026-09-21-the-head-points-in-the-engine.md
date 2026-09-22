# The head points in the engine; the guard sits on its threshold

- Kind: experiment
- Status: current
- Observed: 2026-09-21
- Last verified: 2026-09-21
- Scope: serving / decision primitives, one-pass pointing, `/v1/decide` `point`; kernel test seams
- Related: `crates/server/tests/attention_head_point_gpu.rs`,
  `kernel/include/ignis_attn_tap.h`, `crates/core/src/attn_tap.rs`,
  `.scratch/latent-probe/engine_score.py` and `results/engine/` (raw, on disk),
  `2026-09-21-one-attention-head-points.md` (the vehicle result this checks),
  `2026-09-12-hq-attention-route-agreement.md`
- Superseded by: none

## Question

A PyTorch vehicle found that one attention head, **L39.h10**, read at the
position after the forced `{"x":`, lands inside the target button more often
than the ten-round digit chain, and that as a guard on the chain it takes
218 of 240 to 239 (`2026-09-21-one-attention-head-points.md`). The vehicle
was not the engine: NF4 weights where the engine serves NVFP4, BF16 KV where
it serves hq-e8-2b, and a render the endpoint never sends. Does the head
point in the engine?

## Evidence

**The tap.** The fused attention kernels never materialize weights, so a
test-only seam (`kernel/include/ignis_attn_tap.h`, behind the non-default
`attn-tap` cargo feature) copies what they are computed from: the query and
key rows `run_gqa_layer` hands to the attention op, after the q/k norm and
the rotation. The host forms `q . k / 16` for every GQA layer and query head
over the image positions. Its one production trace is a disarmed flag check
per GQA prefill call. **Under hq-e8-2b the keys are the ones given to the
codec, not decoded from it**: an hq run measures what the codec did to the
earlier layers, not to L39's own scores.

**The harness** (`attention_head_point_gpu.rs`, written by the parallel
session on this branch) runs the 480 scenes the vehicle used, at 1024 px,
under two renders:

- **vehicle** — reproduces the vehicle's prompt, xhigh reasoning paragraph
  and open think block included, and asserts both ends. Its first render is
  **byte-identical** to the vehicle's (679 characters), and on three smoke
  scenes the engine's chain lands within 2 units of the vehicle's.
- **served** — what `/v1/decide` sends: thinking off, the instruction as
  `{"instruction":…}` through the endpoint's own serializer.

**Pre-registered** before the first run, vehicle render, same 480 scenes:
L39.h10's region centre inside on at least 230 of set A and 227 of set B
(within about six of the vehicle), and the guard at d = 60/999 at least 237
on both — first under BF16 KV, then hq-e8-2b.

### The matrix

Inside the button, of 240 — head (L39.h10, region centre) / chain / guard:

| render | KV | set A (target blue) | set B (colour or label) |
|---|---|---|---|
| vehicle | BF16 | **236** / 218 / **238** | **232** / 213 / 236 |
| vehicle | hq-e8-2b* | **230** / 217 / 235 | **232** / 213 / **238** |
| served | BF16 | 235 / 218 / 239 | 232 / 213 / 237 |
| *the vehicle* | *NF4, BF16* | *236 / 218 / 239* | *233 / 213 / 239* |

\* keys before the codec, as above.

**Cross-validation picks L39.h10 on every fold of all six arms**, and the
ranking behind it is the vehicle's: L39.h10, then L43.h9, then L35.h6 or
L35.h16. The head indices match; the orientation check holds (row-major
226-233, transposed 7-14).

**The committed fixture at 4096 px** — the size the endpoint serves by
default, and the first C5 data at it: L39.h10 inside on **3 of 3 under both
renders** (a 128 x 128 grid). The chain is 3 of 3 served and 2 of 3 under the
vehicle render, where `small`'s y lands 3 px outside a 90 px-tall button and
the guard, 20/999 away, keeps it.

## Finding

**The verdict, split, because the two criteria say different things.**

- **The head's criterion passes on all four vehicle arms** — 236 and 232
  under BF16, 230 and 232 under hq — within one of the vehicle under BF16.
  The one-pass point transfers to the engine.
- **The guard's criterion fails on two of four** — set B under BF16 by one
  scene (236), set A under hq by two (235). Its margin was the tighter by
  construction: two scenes below the vehicle's 239, against six for the head.

**The guard sits on its threshold, not below it.** Across the six arms at
1024 px it lands 238, 236, 235, 238, 239, 237: a median of about 237.5,
which is the value pre-registered. The vehicle's 239 was the optimistic end,
and a criterion set at the median passes about half the time — which is
what it did. What the owner should read is the effect, not the coin: **the
guard takes the chain from 218 and 213 to about 237, from about 90% to about
99%**, stably, and a little less than the vehicle promised.

**The head alone beats the chain in every arm**: 230-236 against 213-218.

**NVFP4 does not move the chain and does move the map.** The engine's chain
lands inside on exactly 218 and 213 — and on the **same** scenes: its inside
predicate agrees with the vehicle's on 240 of 240 in both sets, its points
within 5 units of 999 on about 210 of them and 3 at the median. L39.h10's
map, meanwhile, reshuffles on the smallest targets: on
set B it loses three scenes the vehicle had and gains two, every one of them
a button 8-13% wide and 2.1-3.3% tall. The guard lost where two of the three
fell on scenes the chain also misses and one (scene0093) put the head on
another element while the chain was right. That is the calibration for
anything still to be measured in the vehicle: its chain numbers transfer
exactly, its attention numbers to within a few small targets.

**The failures are rules for later, not rescues now.** On scene0093 the chain
was right with a first-digit probability of 0.85/0.58, and a guard that also
weighs that probability might have kept it. Sets A and B are now the guard's
development data: any second guard is designed on them and measured **once**,
on a new set with its threshold fixed first.

## Implications

- **C5 is real in the engine**, as a one-pass point and as a guard on the
  chain, and the head choice is stable across weights, KV format and render.
- **The production configuration is measured by no arm yet.** Production is
  the served render under hq-e8-2b, and under hq this tap scores L39 with the
  keys before the codec. The next pre-registered criterion belongs exactly
  there: served render, hq with the codec's encode-and-decode applied to the
  armed layer's keys inside the tap (`hq_encode_row_warp` /
  `hq_decode_row_thread` with `hq_dither_row_seed`, as
  `kernel/tests/test_hq_codec_kv_rows.cu` does), on a new set C. Before any
  second guard.
- **4096 px needs its own sweep.** Three fixture scenes say the head points
  there; they say nothing about `d`, which is in 0-999 units and was chosen
  on a 32 x 32 grid.
- **The engine-side cost stays one GEMV per decision.** The harness is no
  measure of it — it scores all 384 heads on the host and copies every
  armed layer's keys synchronously, and still runs a 240-scene set in about
  105 s end to end, chain and model load included.

## Limits and unknowns

- *Measured since*, in `2026-09-22-the-codec-costs-the-head-its-read.md`:
  the codec on L39's own keys costs the head 20 scenes of 240 at 1024 px
  and fails set C's pre-registered floors; at 4096 px it costs 3, and there
  the chain drifts down. The bullet below is kept as written.
- **The hq rows leave out the armed layer's own quantization**: its scores
  use pre-codec keys. That is likely the kinder side of the codec but not a
  proven bound — noise on the keys can move a region either way. The route-agreement finding puts the codec's attention
  error at ~0.19-0.23 median relative L2 with near-tie flips; for a map read
  by its region centre that should matter less than for an argmax, and it is
  unmeasured here.
- **1024 px for the criterion, three scenes at 4096.** The served default is
  4096; `d` and the head's small-target behaviour there are open.
- **Synthetic scenes from one generator family**, the same 480 the vehicle
  used — now development data for the guard.
- **One head, calibrated on labelled synthetic scenes**, keyed in the harness
  to this artifact by constant. A production path needs it keyed to the
  artifact the way the answer alphabet is.
- The tap's disarmed check sits in the production prefill path on this
  branch; it is one relaxed flag load per GQA layer call and was not
  profiled.
