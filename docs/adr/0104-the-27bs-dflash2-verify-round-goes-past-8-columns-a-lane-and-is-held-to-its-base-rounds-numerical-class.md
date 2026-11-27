---
status: accepted
amends: [ADR-0090, ADR-0095]
amended-by: [ADR-0106]
---

# The 27B's DFlash2 verify round goes past 8 columns a lane, and is held to its base round's numerical class

The 27B's DFlash2 drafted at most 7 tokens a round: `MAX_DRAFT_TOKENS = 7`, the range the reference gave its
draft-tokens option, taken as it was and never measured past (#666). On structured output that range, not the
drafter, is what a round waits on: measured on main (6d671912, the NVFP4 artifact, one lane, served) a greedy JSON
answer of 7,229 tokens commits 7.93 tokens a round against a ceiling of 8. Asked for 15 drafts, the drafter, trained
at a block of 8, still has positions 8 to 15 taken in 90 to 36% of that answer's rounds, and a 16-column round costs
17.17 ms against 14.93 at 8 columns for 13.4 tokens a round against 7.9; on sampled free text almost nothing past
position 7 is taken, so a window fixed at 15 would cost it 7-8% (spec #684, measured 2026-11-19 on the RTX 5090). The
owner decided spec #684 on 2026-11-19: the window grows for DFlash2, and each round's shape is chosen per round.

- **The draft window.** A DFlash2 load of the 27B takes 1 to 15 drafts, 15 the edge of every vendored op's domain
  the round reaches; the reference's 1..7 range is superseded for DFlash2. The 27B's MTP head keeps 0 to 7 and the
  verify substrate alone 1 to 7 (ADR 0095's window is the head's and stands), and so does Flash-Next. A load at 7
  drafts or fewer is main's: one shape, its plan, graphs and text. Every profile's default stays 7; the `rtx5090`
  profile's becomes 15 only in the commit that records the served gate's pass (#696).
- **Two round shapes.** Past 7 drafts a load holds a **base shape**, 8 columns a lane at every lane count, today's
  round, and an **extended shape**, the window + 1 columns a lane at 1 to 4 lanes, so that a round never carries more
  than 64 columns. A round has one shape for all its lanes, chosen before it launches and handed to it; the shapes
  share one set of device buffers.
- **The shape rule** (what a lane asks): a lane starts at the base shape, asks the extended one after a base round
  that took `spec.shape_enter_accepted` drafts of a full window (default 7, every one), and keeps asking while its
  extended rounds take at least `spec.shape_stay_accepted` (default 6). A round the budget cut short changes nothing.
  The rule reads only what the host holds before the next round launches, never the drafter's confidence: handing
  the previous round's confidence back was replayed at +0.2 points over the accept history alone. A threshold the
  load's window cannot satisfy is refused at load.
- **The shape policy** (the lanes' asks to the round's shape): `spec.shape_policy`, `aggregate` by default, `all`
  selectable. Under `all` a round is extended only when every lane in it asks, so no lane pays for another. Under
  `aggregate` one lane's ask decides at one lane, and at 2 to 4 lanes the round is extended when the drafts the asking
  lanes are expected to take past position 7 exceed what the wider round costs every lane in it, from each lane's
  recent rounds and a period ratio per lane count (measured 1.15, 1.23 and 1.31 at 1, 2 and 4 lanes); a lane's first
  extended round after it asks is granted. If `aggregate` does not raise the pair cells' aggregate over `all` in the
  served gate, the owner decides the default.
- **Rules are judged on served arms.** Two offline replays of round records ranked rules wrongly against the served
  arms: a shape change is a text change, and acceptance depends on the shape history. A second rule a replay proposed
  (enter after two full rounds, leave after two short ones) was served and refuted: it kept the JSON answer's gain and
  lost greedy code's and the agent turn's. A rule changes on a served A/B, never on a replay alone.

**The extended round is held numerically equivalent to its base round, not bit-exact to it and not to a plain
decode.** Past 8 columns the round's attention runs in two chunks, its FP8 exception linears quantize their
activations on another tile and the GDN gating projection leaves its small-T route, so its rows are the base round's
up to summation order. ADR 0056's bounds cannot be the oracle: they hold a Flash-Next round to speculation off, and
on this model main's own base round already sits past them. Over 19,720 teacher-forced rows of three long answers
the base round's rows sit a median 18.5 BF16 ulps from a plain round's (99th percentile 97.9, widest 161.0, against
ADR 0056's 64) and move 239 picks, 16 past its 16-ulp near tie; the extended round's sit at 18.6 (97.8, widest 200.5)
and move 213. Whether a drafted greedy decode should be held to a plain one on the 27B is #597's question, which this
decision does not close; the extended round must not widen it. So the test
(`crates/core/tests/extended_round_class_gpu.rs`, ADR 0056's method) offers speculation off's own greedy text as
each round's drafts, so each column holds the text and its row stands after the same tokens as a plain round's, and
holds every extended row to the base row at the same text index and lane count:

- a row within **256 BF16 ulps** at the base row's largest magnitude, and a lane's median row within **32**;
- each lane's mean log-probability of the text within **0.02 nats** a token of the base rows';
- **the class**: the extended rows' distance from the plain rows over the base rows', at each lane count of a kind
  of round and over all of them together, at most **1.25** at the median and at the 90th percentile and **1.5** at
  the widest row; no more picks moved from the plain rows' than **1.25 times** the base rows' **plus 2**;
- **the class's near tie**, over every lane count of a kind together: a pick the extended round moves, from the plain
  row's or from the base row's, only where the reference row rates the two within ADR 0056's **16 ulps** or within
  **twice** the widest gap at which the base rows move a pick from the plain rows', whichever is more. The base round
  itself moves picks past 16 ulps (to 42 on the long answers), so ADR 0056's near tie alone would hold the extended
  round tighter than its base round is; and the widest gap is one pick's of the 11 to 25 a lane count moves, so it is
  read over the kind's lane counts together;
- **a control**: an extended row against the plain row one index on, another context, whose median must sit past
  **twice** the right rows' median and twice the typical bound, or the bounds could not tell a right row from a wrong
  one.

It runs at 1 to 4 lanes in two kinds of round: rounds offered the text as their drafts, which run eagerly, and the
served rounds replaying each shape's captured graph on their drafter's own drafts, teacher-forced to the text, whose
rows up to the accepted count stand after it. Measured on the RTX 5090 (NVFP4 body, FP8 exception weights,
hq-e8-2b), by the prototype at one lane on eager rounds and by the test on 2026-11-20 over 3,360 forced and 3,205
graph rows:

| bound | margin | prototype, one lane | forced rounds, 1-4 lanes | graph rounds, 1-4 lanes |
|---|---|---|---|---|
| widest row, extended against base | 256 ulps | 168.0 | 96.9 (0.0-1.0 at 2-4 lanes) | 115.2 |
| a lane's median row | 32 ulps | 14.0-22.3 | 0.0-16.1 | 0.0-16.0 |
| mean log-probability, apart | 0.02 nats | 0.0089 | 0.0089 | 0.0124 |
| class ratio, median / 90th percentile / widest | 1.25 / 1.25 / 1.5 | 0.86-1.25 | 1.00 / 1.00 / 1.00 | 1.02 / 0.99 / 1.03 |
| picks moved from the plain rows, extended and base | 1.25 x + 2 | 15 and 14 | 90 and 89 | 76 and 73 |
| widest moved gap, extended over base; between the shapes | 2 x, floor 16 | 6.0 over 11.0; 8.0 | 12.0 over 12.0; 8.0 | 22.0 over 12.0; 17.0 |
| the same, at one lane count, the furthest | | | 12.0 over 11.0 (1 lane) | 22.0 over 9.0 (4 lanes) |
| control's median, over the right rows' and the typical bound | 2 x | 14x | 14.0x, 5.2x | 13.8x, 5.2x |

At each lane count the distance ratios read 0.94 to 1.06 and the moved picks at most 26 against a bound of 28. At 2
to 4 lanes the two shapes' forced rows agree within 1 ulp, and the graph rows of the code prompt within 1 ulp too,
while the other prompts' graph rows differ there as much as at one lane: observed, not explained. A graph cell's
rounds narrow as its lanes' texts run out, so its base and extended rows at one index can come from rounds of
different widths.

Two readings were the owner's, beside the margins. The test held a pick moved between the shapes to ADR 0056's 16 ulps
at first; its first run failed once, a pick moved at 17 ulps on a one-lane graph round (the base rows' widest gap there
12.0), and the class's near tie took its place: a bound changed after a failure, inside the class it states. And the
near tie is read over a kind's lane counts together: at the graph rounds' 4 lanes alone the extended rows move a pick at
22.0 ulps where the base rows' widest is 9.0, 2.44 times, past the margin of 2 there. The margins (1.25, 1.5, 1.25 + 2,
2) sit above the widest ratios measured but that one. The owner confirmed the margins, the class's near tie in place of
16 ulps, and its reading over a kind's lane counts together on 2026-11-20.

**What this amends.** In ADR 0056, the method of a verify round past 8 rows held teacher-forced, row by row, reaches
the 27B's DFlash2 round, its reference the base round rather than speculation off. In ADR 0090, the sampled draft draws
and accepts at up to 15 drafts a lane, the sampler's 16 columns, and the drafter's q it keeps is 128 bytes a draft a lane:
896 at seven drafts, 1,920 at fifteen. In ADR 0095, the 27B's two drafters no longer share a range: DFlash2 takes 1 to
15 and the MTP head keeps 0 to 7, so a window named beside one is that drafter's own.

## Considered options

- **A window fixed at 15**: free text pays the wider round and takes no more drafts, 7-8%.
- **Lanes of both shapes in one round**, the only way to an extended shape past 4 lanes: studied and costed, not built.
- **Bit-exactness to the base round**: 16-column kernels in the 8-column round's reduction order everywhere a route
  switches at 8, ADR 0056's option 2, a kernel redesign for a seam that is a tolerance on the base round already.
- **The extended round held to a plain decode by a bound of its own**: main's base round fails ADR 0056's bounds on
  this model, so any such bound would be set by the base round and say nothing the class comparison does not.

## Consequences

- A greedy answer can differ at a near tie when a round changes shape, as it already can between a drafted and a
  plain decode; the guide says so beside the draft window.
- The test stands on test-only seams outside the serving round: a teacher-forced round may name its round shape and
  its drafts (the decode options' `forced_shape_window` and `forced_drafts`, which the leaf reads only beside
  `forced_tokens`, never from the environment or the model, and which no captured graph carries), and the readouts
  read the shape their window names, refusing the verify logits after a round of the other shape.
- The bounds are the RTX 5090's, on the NVFP4 body under FP8 exception weights over hq-e8-2b. A kernel change on
  the extended round, another card or a trellis body re-measures them before a default moves there.
