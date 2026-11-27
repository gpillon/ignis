---
status: accepted
---

# The 27B's DFlash2 verify round goes past 8 columns a lane, and is held to its base round's numerical class

The 27B's DFlash2 drafted at most 7 tokens a round, the range the reference gave its draft-tokens option, taken as it
was and never measured past. On structured output that range, not the drafter, is what a round waits on: a greedy
JSON answer commits 7.93 tokens a round against a ceiling of 8, and 13.4 when asked for 15 drafts, in a round about
15% longer. On sampled free text almost nothing past position 7 is taken, so a window fixed at 15 would cost it 7-8%
([finding](../findings/2026-11-20-the-27bs-draft-window-of-15-served-on-the-rtx-5090.md)). So the window grows for
DFlash2, and each round's shape is chosen per round.

- **The draft window.** A DFlash2 load of the 27B takes 1 to 15 drafts, 15 the edge of every vendored op's domain
  the round reaches; the reference's 1..7 range is superseded for DFlash2 alone. The 27B's MTP head (ADR 0095), the
  verify substrate alone and Flash-Next keep a window of at most 7. A load at 7 drafts or fewer is what it was: one
  shape, its plan, graphs and text. Every profile's default stays 7 but `rtx5090`'s, 15 since its served gate passed.
- **Two round shapes.** Past 7 drafts a load holds a **base shape**, 8 columns a lane at every lane count, the round
  of a load at 7 drafts, and an **extended shape**, the window + 1 columns a lane, so that a round never carries more
  than 64 columns: a round with every lane extended runs at 1 to 4 lanes. A round's shape is each lane's, chosen
  before it launches and handed to it: every lane at one shape, or lanes of both, a mixed round, the only way to an
  extended lane past 4 lanes (ADR 0106). The shapes share one set of device buffers.
- **The shape rule and the shape policy** (`kernel/src/round_shape.h`, its twin `crates/core/src/shape_grant.rs`).
  The rule says what a lane asks: it starts at the base shape, asks the extended one after a base round whose full
  window was taken up to a threshold, and keeps asking while its extended rounds take enough. It reads only what the
  host holds before the next round launches, never the drafter's confidence. The policy (`spec.shape_policy`) turns
  the lanes' asks into the round's shape: `all` extends a round only when every lane in it asks, so no lane pays for
  another; `aggregate`, the default, which the served gate kept over `all`, at one lane when the lane asks, and at
  2 to 4 lanes when the drafts the asking lanes are expected to take past position 7 exceed what the wider round
  costs every lane in it; `mixed` grants lane by lane and alone runs a mixed round (ADR 0106, which says where it is
  the default).
- **Rules are judged on served arms.** Two offline replays of round records ranked rules wrongly against the served
  arms: a shape change is a text change, and acceptance depends on the shape history. A rule changes on a served A/B,
  never on a replay alone.

**The extended round is held numerically equivalent to its base round, not bit-exact to it and not to a plain
decode.** Past 8 columns the round's attention runs in two chunks, its FP8 exception linears quantize their
activations on another tile and the GDN gating projection leaves its small-T route, so its rows are the base round's
up to summation order. ADR 0056's bounds cannot be the oracle: they hold a Flash-Next round to speculation off, and
on this model the base round itself already sits past them. Whether a drafted greedy decode should be held to a
plain one on the 27B is #597's question, which this decision does not close; the extended round must not widen it.

So the test (`crates/core/tests/extended_round_class_gpu.rs`) follows ADR 0056's method: both shapes teacher-forced
to speculation off's own greedy text, on rounds offered the text as their drafts and on the served rounds replaying
each shape's captured graph, every extended row held to the base row at the same text index and lane count. Beside
bounds on the row, a lane's median row and the text's likelihood, it holds **the class**: the extended rows sit no
further from the plain rows than the base rows do, and move no more picks from them, beyond a margin, at each lane
count of a kind of round and over all of them together; a pick moves, from the plain row's or from the base row's,
only within the class's near tie; and a control, an extended row against the plain row one index on, must sit far
outside, or the bounds could not tell a right row from a wrong one. The bounds and margins are constants of the test,
each with what was measured beside it.

The margins are the owner's, and so are two readings of the near tie. It is the class's, ADR 0056's near tie or a
margin times the widest gap at which the base rows move a pick from the plain rows', whichever is more: the base
round itself moves picks past ADR 0056's, which alone would hold the extended round tighter than its base round is.
The test held a pick moved between the shapes to that one at first and failed once: a bound changed after a failure,
inside the class it states. And it is read over a kind's lane counts together: the widest gap is one pick's of the
few a lane count moves, and at the graph rounds' 4 lanes alone the extended rows move a pick at 2.44 times the base
rows' widest gap, the one ratio measured above its margin.

## Considered Options

- **A window fixed at 15**: free text pays the wider round and takes no more drafts, 7-8%.
- **The drafter's confidence handed to the shape rule**: replayed at +0.2 points over the accept history alone.
- **A rule a replay proposed** (enter after two full rounds, leave after two short ones): served and refuted. It kept
  the JSON answer's gain and lost greedy code's and the agent turn's.
- **Bit-exactness to the base round**: 16-column kernels in the 8-column round's reduction order everywhere a route
  switches at 8, the wider kernels ADR 0056 rejects, a kernel redesign for a seam that is a tolerance on the base
  round already.
- **The extended round held to a plain decode by a bound of its own**: the base round fails ADR 0056's bounds on this
  model, so any such bound would be set by the base round and say nothing the class comparison does not.

## Consequences

- A greedy answer can differ at a near tie when a lane's round changes shape or mix, as it already can between a
  drafted and a plain decode; the guide says so beside the draft window.
- The test stands on test-only seams outside the serving round: a teacher-forced round may name its round shape and
  its drafts, which the leaf reads only beside the forced tokens, never from the environment or the model, and which
  no captured graph carries, and the readouts read the shape their window names, refusing the verify logits after a
  round of the other shape.
- The bounds are the RTX 5090's, on the NVFP4 body under FP8 exception weights over hq-e8-2b. A kernel change on the
  extended round, another card or a trellis body re-measures them before a default moves there.
- The class reaches the mixed round, held as the extended round is (ADR 0106).
