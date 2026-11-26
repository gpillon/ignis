---
status: accepted
amends: [ADR-0017, ADR-0030, ADR-0104]
---

# The 27B's DFlash2 verify round may carry lanes of both shapes, and a mixed round is held to its base round's numerical class

[ADR 0104](0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md)
gives the 27B's DFlash2 verify round past 7 drafts two shapes, the base shape (8 columns a lane) and the extended shape
(the window + 1 columns a lane at 1 to 4 lanes), and "one shape for all its lanes"; lanes of both shapes in one round
were "studied and costed, not built". So one predictable lane drags every other lane into 16 columns or is refused:
served on the build of #684, a JSON answer beside three conversations at four lanes ran as if the extended shape did
not exist, its lane committing 9.4 tokens a round against 13.4 alone, and at two lanes a conversation beside a JSON
answer paid for 32 columns where it needs 8 (spec #711, measured 2026-11-20 on the RTX 5090). Spec #711 builds the
round ADR 0104 left out; the owner narrowed it on 2026-11-22 (#711, two decisions found building #713).

- **A mixed round.** A DFlash2 verify round of the 27B at 15 drafts may carry `w` lanes at the extended shape (1 to 3,
  the extended group) beside `n` at the base shape (1 to 6, the base group), at 2 to 7 lanes and at most 64 columns:
  one of twelve mixes, (1,1) to (1,6), (2,1) to (2,4), (3,1) and (3,2). A round is uniform base, uniform extended or one
  of the mixes; a round of 8 lanes is at the base shape. The extended lanes come first in the round's rows, each lane's
  columns and drafts at its own group's stride, the base group's after the extended group's; its outputs go back in
  the caller's order and nothing moves on the device. The shape a round is handed before it launches is each lane's.
- **At 15 drafts only** (the owner's first decision). Each group's GDN records lie in its own shape's records over the
  one backing both shapes share, base lane `s` at row `2w + s` of a base layout laid at the extended layout's stride;
  only at a window of 15 is a group's record row its column there, and below it the base group's records and the
  extended layout's collide. A load at 8 to 14 drafts holds no mix: its rounds are uniform and it captures no mixed
  graph. No record region of their own, no new buffer.
- **A mix (w, n) takes 2w + n GDN slots** (the owner's second decision). The base group's fold runs the vendored
  `gdn_replay_fold` as it is, behind `2w` leading rows that commit no column (the op's own no-op), and the op wants
  distinct state slots on every row, those rows included, on slots no base lane holds. So a mix is captured and runs
  only where the load's lanes hold `w + n` and the pool's GDN slots, its decode lanes and its device retained slots,
  hold `2w + n`; the 27B's default 8 lanes hold all twelve, and the `rtx-16gb` profile, 2 lanes and no retained slot,
  holds none. No vendored file changes.
- **The grant** (`spec.shape_policy: mixed`, #715). The shape rule is unchanged: it says what each lane asks. At one
  lane the lane's ask decides, as under `aggregate`. At 2 to 7 lanes the round starts from the uniform base round and
  the asking lanes are granted the extended shape one by one: the highest expected extra tokens first, at a tie the lane
  refused longest, then the caller's order; a lane that asks with no extended round among its recent rounds after those
  with an estimate, granted when the columns allow, so that its estimate has a round to read. A lane is granted while
  the round with it extended too stays within 64 columns, and when its expected extra tokens exceed (that round's period
  ratio over the current round's, less one) times the tokens every lane of the round is expected to take, each lane's
  expectations read over its own last `spec.shape_estimate_rounds` verify rounds as under `aggregate`; a step to a round
  the cost table says costs less is granted. A refused lane does not end the grant, only the columns do, so a round of 8
  lanes is at the base shape, and a step to a mix the load holds no graph of: the grant stops at the last mix the load
  holds, so a pool that holds some of them keeps their gain, and a round at a lane count where the load holds no mix
  takes `aggregate`'s choice. A lane under a grammar is granted like any other (#716). The grant is a pure function in
  the core, held to tables, and the leaf runs its twin each round, held to the same rows. `all` and `aggregate` are
  unchanged and never run a mixed round. Granting every ask in the caller's order, as the prototype did, bought nothing
  on a moderately predictable answer beside a conversation (+0.6%, inside the spread, the conversation 5.0% slower): the
  grant has to refuse such a lane.
- **Its default.** `mixed` on the `rtx5090` profile for the NVFP4 body, `aggregate` on every other profile. The served
  gate (#722,
  [finding](../findings/2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md)) asked a JSON
  answer beside three conversations at four lanes at least 10% faster together than the build of #684, no conversation
  lane more than 15% below it, and no one-lane or free-text cell more than 2% below: on the RTX 5090 it read +14.48%,
  the slowest conversation lane -7.9%, and missed the 2% line by the letter on two sampled one-lane cells (-2.05% and
  -2.45%) and a cold first token 7 ms later. The owner made `mixed` the `rtx5090` profile's default on 2026-11-23 all
  the same: one lane forms no mix, and seeded, a request alone wrote the same text in the same rounds under both
  builds. Beside the 27B's MTP head the policy is ignored, as under `off` and as every draft setting is where its
  speculation does not run (edited 2026-11-26, #729). It is the NVFP4 body's too, the one the gate served: the trellis bodies keep
  `aggregate`, under `rtx5090` as under their own profiles, and a trellis body at 15 drafts runs `mixed` where the
  operator names it (a flag, the environment, a file or a patch); a run on a 16 GB or 24 GB card would be a new
  decision. A load given `mixed` that holds no mix runs `aggregate`, and its startup line names both, the policy it was
  given as the operator's (`shape_policy_given`) or a profile's (`shape_policy_profile`): a trellis body under the
  profile's, one whose pool's lanes and GDN slots hold no mix (which a WARN at load names, with the lanes and the slots,
  where the operator named `mixed`), and one whose plan asked for none under `IGNIS_MIXED_ROUNDS=0`; a load whose latest
  capture ran under that switch takes `aggregate`'s choice too. A load of DFlash2 at another window than 15 runs
  `aggregate` with a WARN at load naming its window and whose `mixed` it was, and its line names the operator's alone;
  at 7 drafts or fewer `mixed` changes nothing else, the load holding one shape, and the 27B's MTP head ignores it, as
  it ignores every shape policy (edited 2026-11-26, #729).
- **The cost table.** `spec.shape_period_ratios` becomes a table by lane count (1 to 7) and extended lanes, each entry
  the period of the round with that many lanes extended over the base round at the same lane count; `aggregate` reads
  its uniform entries at 1 to 4 lanes and `mixed` reads every entry. Its default, `auto`, is the loaded body's own
  table, resolved at load where the body is known (`PeriodTableSetting::of_body`), as `model.exception_weights` and
  `model.rope_scaling` resolve theirs under ADR 0046's one declaration: a table named whole stands on either body, the
  four-value form's mixed entries are the loaded body's, and the four-value form at its earlier default reads as `auto`,
  so a file an earlier build generated names no table. Both bodies' tables are measured on the RTX 5090 at a served
  context (#721, below): every entry a round forced to its shapes over a base round at its lane count measured in the
  same session, never read off a line between lane counts, since the base round's cost rises in steps with its columns
  and on this card not always by much (8 lanes cost 0.3% more than 7 on the NVFP4 body). Where a mixed entry measured
  below the entry before it in its row, above its row's uniform entry or under 1, the default holds it at that bound
  (`PeriodTable::held`), so that under a default no step of the grant is free; the uniform entries, which `aggregate`
  reads, stand as they are: the NVFP4 body's #690's served measurement, which this one reproduced, the trellis body's
  this one's. The four-value form stays accepted as the uniform entries.
- **The mixed graphs, a sixteenth line of the VRAM plan** (#714). A mixed round replays a verify graph and a drafter
  append graph of its mix, both or neither, captured at load for every mix the load holds where mixed rounds are asked;
  the grant never takes a mix with no graph, and a round forced at one (the test seam) runs at the uniform shape
  `aggregate` would choose, counted, never eagerly and never refused. The graphs are the plan's line `mixed_graphs`, in
  the leaf's reservations and their mirror in the core, a measured size of each body: the twelve at 8 lanes at what the
  RTX 5090 measured their captures to take with the mixed graph build, as the device's free memory before them less
  after them, 69,206,016 B (66 MiB, two runs) on the NVFP4 body and at most 73,400,320 B (70 MiB) on the trellis body,
  and 5 MiB less a mix fewer (one mix's capture took 4 to 10 MiB), in whole 2 MiB pages; 0 on a load that captures none.
  The driver keeps the graphs' pages after a teardown, so a reload adds the line back. This amends
  [ADR 0030](0030-device-memory-reserved-at-load.md) as
  [ADR 0101](0101-on-windows-the-vram-plan-reserves-the-w4a4-tma-products-descriptor-slots-as-a-line-of-their-own.md)
  did for the fifteenth line, by a new ADR rather than an edit in place, which only the owner decides.
- **The shape counts, a label of a family ADR 0017 has** (#719). `shape="mixed"` joins
  `ignis_speculative_shape_rounds_total`'s label set beside `base` and `extended`, with no new family: like the other
  shapes it counts each lane's rounds, a lane of a mixed round once as mixed and at neither of the others, so the three
  series sum to `ignis_speculative_rounds_total` and a share of them is a share of lane-rounds. That is the unit user
  story 27 asks for (how often rounds mix), taken over a literal reading of "the shape counts read each lane's own",
  which would count a mixed round's lanes at their own shapes and no round as mixed; the per-position drafted and
  accepted families keep counting each lane's own drafts, an extended lane's up to its window and a base lane's up to
  its own. The series renders at 0 on a load that runs no mixed round, and the Monitor's *Speculation* card shows its
  share beside the extended rounds'. This amends [ADR 0017](0017-prometheus-metrics.md) by a new ADR, as for ADR 0030.

**A mixed round is held to its base round's numerical class as ADR 0104 holds the extended round: its method, its
margins and its window.** Each group runs its own shape's routes on its slice and the ops that see only columns run once
over all `16w + 8n` of them, so a mixed lane's rows are its base round's up to summation order, as an extended round's
are. The test (`crates/core/tests/extended_round_class_gpu.rs`, on ADR 0104's NVFP4 load, its graphs captured again with
the mixes') offers speculation off's own greedy text as each mixed round's drafts, placed by group, and replays each
mix's captured graph on the drafter's own drafts, both teacher-forced to the text, and holds every mixed row, of either
group, to the base row at the same text index and lane count by ADR 0104's bounds: a row within 256 BF16 ulps at the
base row's largest magnitude, a lane's median row within 32, each lane's mean log-probability of the text within 0.02
nats a token, the class at each lane count and over a kind's lane counts together (1.25 at the median and the 90th
percentile, 1.5 at the widest row, no more moved picks than 1.25 times the base rows' plus 2), the class's near tie and
the control one index off over a kind's lane counts together. Every one of the twelve mixes runs in both kinds of round,
its last lanes extended so that the round's rows are not the caller's order, its lanes on seven prompts of their own.
The owner decided on 2026-11-22 (#717), after a first reading failed on one lane, that the mixed rows are held exactly
as ADR 0104 holds the extended round: legs long enough for each lane to reach the uniform cells' row counts, every
round at its mix; the near tie read the way ADR 0104 reads it; the margins unchanged. How the test does so, the owner
confirmed with this ADR on 2026-11-23 (#722):

- **A lane is held over a uniform cell's window.** A mixed lane holds the rows after its text's first 224 tokens, as a
  uniform cell's lane does, against the base rows a leg of base rounds at its lane count gives over the same window, at
  least 64 of them. Its leg rides at its mix, every round at the mix and the lane count, until every lane holds the rows
  its base leg gave it: a lane that commits more a round rides on past them on its text, its rows there not read, and
  a lane whose text runs out first fails. Speculation off writes each text's rows over the window one lane at a time,
  and the text past it, which no row is read from, the lanes together.
- **The near tie is the uniform cells'**: read, as ADR 0104 reads it, over the base rows of the kind's uniform cells
  at 1 to 4 lanes (both kinds' widest moved gap 12.0 ulps, so 24), not over the mixed cells' own base rows nor pooled
  with them.

The first reading rode each mixed leg while every lane's text, at most 224 tokens, held a whole round of its shape, so a
lane beside a faster one held 51 to 85 rows on prose. On graph rounds the prose prompt's lane extended at the mix 3+2
read its text 0.0256 nats a token below the base rows' over 79 rows, past 0.02. A diagnostic found the mix adds nothing
to it: over the same 79 rows that prompt's lane of the uniform extended graph cell reads 0.0257 at 4 lanes and 0.0201
at one, its rows bit-identical to the mixed lane's over the first 54, where the two legs' rounds pair, and the excess
sits in a few near-tie rows, one of them 0.78 nats of the lane's 2.02. A mean over a short window of prose moves so in
the uniform cells too: that lane of the uniform extended graph cell reads 0.0348 below its base rows over its first 32
rows at one lane and 0.0303 at four, 0.0242 and 0.0177 over 64, inside 0.02 for good only from row 87 at one lane and
from row 180 at four, and 0.0015 and 0.0124 apart over its 210, the window over which ADR 0104 measured and confirmed its
bounds. A longer leg (texts of 512 tokens, the other lanes on essays) brought the 3+2 lane inside, 0.0130 over 324 rows;
the near tie read over the mixed cells' own base rows (9.0, so 18) then refused a pick moved 22.0 ulps apart, a move the
uniform extended graph rows make too and ADR 0104's reading over its kind's base rows passes. So the mixed lane is read
over the window, and against the near tie, that the extended round is: rows past that window would hold the mixed round
to bounds the extended round was never read against, the base round's own near tie widening on longer texts (to 42 ulps
on the long answers, ADR 0104).

Measured on the RTX 5090 (NVFP4 body, FP8 exception weights, hq-e8-2b) on 2026-11-22, over 12,096 forced and 11,887
graph mixed rows at 2 to 7 lanes (224 a lane forced, 217 to 223 on graph rounds), each leg 28 forced rounds or 45 to 88
graph rounds at its mix:

| bound | margin | forced mixed rounds | graph mixed rounds |
|---|---|---|---|
| widest row, mixed against base | 256 ulps | 1.0 | 110.8 |
| a lane's median row | 32 ulps | 0.0-0.5 | 0.0-14.9 |
| mean log-probability, apart | 0.02 nats | 0.0005 | 0.0158 (0.0229 over its first 64 rows) |
| class ratio, median / 90th percentile / widest | 1.25 / 1.25 / 1.5 | 1.00 / 1.00 / 1.00 | 1.00 / 0.99 / 1.00 |
| the same, by lane count | | 1.00 / 1.00 / 1.00 | 0.99-1.01 / 0.95-1.02 / 0.89-1.00 |
| picks moved from the plain rows, mixed and base | 1.25 x + 2 | 385 and 385 | 311 and 286 (70 and 57 at 4 lanes, bound 73) |
| widest moved gap, mixed over the mixed cells' base; between them | 2 x the uniform cells' 12.0, so 24 | 9.0 over 9.0; none | 10.0 over 10.0; 12.0 |
| control's median, over the right rows' and the typical bound | 2 x | 14.0x, 5.0x | 14.0x, 5.0x |

Forced, every mixed row is its base row within 1 ulp, both groups' (the uniform shapes' forced rows already agree so at
2 to 4 lanes); on graph rounds a lane's row at one text index can stand at another column of its round in the mixed leg
than in the base leg, each leg's drafts and accepted counts its own, as between a uniform graph cell's two shapes. The
furthest likelihood is the essay prompt's lane extended at the mix 3+2, 0.0158 over its 218 rows, which reads 0.0229
over its first 64: the short window's excess again, inside the bound over the window. The near tie's reading was not put
to the test on this run: over the mixed cells' own base rows it would have been 18 and 20, which the mixed rows' 9.0 and
10.0 pass too. The same run's uniform cells print what the first reading's did, line for line.

**On the trellis body a mixed row is held, bit for bit, to the row of a uniform round of its lane's shape at its
round's columns**, and the uniform rounds to the class by ADR 0104's cells, its margins unchanged (the owner's decision
of 2026-11-23, #720). Read on the trellis cell of the same test (the `sc30` artifact, BF16 exception weights, the same
8 lanes and context; 2026-11-22), the method above sat inside every bound of the row and the likelihood and past five
relative clauses of the class: forced rounds at 2 lanes, the widest mixed row 23.5 ulps from speculation off's against
the base leg's 14.3, past 1.5 times it; graph rounds, 4, 8 and 9 picks moved from the plain rows at 3, 4 and 5 lanes
against the base legs' 0, 3 and 3, and 30 against 13 over the kind's lane counts. A diagnostic found no defect in the
trellis mixed path. Every mixed row, 12,096 forced and 11,808 from the graph, is bit-identical to the same prompt's row
in a uniform round of its own shape at the round's columns: on the trellis body a lane's rows follow its round's
columns, not its lane count, the trellis projection routing by the call's token count. Forced, the base legs of 3 to 7
lanes (24 to 56 columns) give each prompt's rows the same bits and the leg of 2 (16) others, so the test read the 1+1
mix's 24 columns against a base leg of 16. The graph rounds' moved picks are six text positions at exact ties, gaps
of 0 or 1 ulp, counted once for each mix that holds the lane, positions the uniform rounds move too; the argmax is not
in it, the rows the same under `IGNIS_MIXED_GRAPH_BUILD=0` (12,096 and 11,808 of them, the moved picks 49 and 30 both
ways). So a base leg at the lane count is no reference on that body. The test holds the decision so, as the owner
confirmed with this ADR on 2026-11-23 (#722):

- **Uniform legs at one width stand for a mix's round.** On the RTX 5090 a lane's rows are the same bits in every
  round of 24 to 56 columns, of either kind (forced, the base legs of 3 to 7 lanes and every mix of those columns; on
  graph rounds the base round at 3 and at 7 lanes, every mix of those columns against the base round at 7 and the
  extended at 3), and a graph round of 64 columns gives its own where the drafter's drafts move (two prompts' lanes
  leave the 56-column round's at their rows 6 and 17, the three mixes of 64 columns bit-identical to the uniform rounds
  of 64); the extended shape has no round of 24, 40 or 56 columns. So, in each kind, the uniform legs of each shape on
  the mixed prompts' texts, every lane riding until each holds 224 rows so that no round narrows, as no mixed round
  does: 7 base lanes (56 columns) and 3 extended (48) for the nine mixes of 24 to 56 columns, 8 base lanes (the eighth
  on a prompt again) and 4 extended for the three of 64.
- **Each mixed lane rides at its mix to 224 rows, every one its prompt's row in the uniform leg of its shape**, bit for
  bit; a lane short of them fails. No trellis mixed row is read against a base row, so none of ADR 0104's margins is
  read there; the NVFP4 cell holds its mixed rows as above.

Measured on the RTX 5090 on 2026-11-23, every mixed row of the trellis cell is its uniform leg's, bit for bit: 12,096
forced rows (each mix 28 rounds) and 12,096 from the graph (224 a lane, each mix 39 to 88 rounds) over the 54 lanes of
the twelve mixes, the cell's uniform section printing what its run of 2026-11-22 did, line for line. A change that
splits the rounds of 24 to 56 columns fails it, and the uniform legs then follow the split; the extended shape has a
round of a mix's own columns only beside an even count of base lanes.

**The cost table, measured** (#721). On the RTX 5090 on 2026-11-23, through the round identity test's timing
(`IGNIS_ROUND_IDENTITY_TIMING_TABLE`) at a served context (`IGNIS_ROUND_IDENTITY_TIMING_CONTEXT=3584`: each timed
lane's prompt behind a long text of the guide's, prefilled to 3,584 tokens and ridden 1,536 more, a fresh set each time
a lane reaches that, the model loaded at 262,144 tokens of context): at each lane count of 1 to 8, the base round and
each count of extended lanes 64 columns hold, each cell after a capture again, 600 greedy rounds of fresh lanes forced
to its shapes (its last lanes extended), on the GPU clock with the round timing; the NVFP4 body on `nvfp4-dflash2-w15`
(FP8 exception weights, hq-e8-2b, the mixed graph build), the trellis body on `trellis-24gb-w15-rule`; four legs each,
riding the cells forward, in reverse, in reverse and forward. Each cell's period is the median of its 256-round blocks'
medians over the four legs, eight of them, since a shared machine only ever slows a block, so one slow block or leg
must not move a cell. An entry is its cell's period over the base round's at its lane count; the legs' own readings of
an entry span at most 0.007 on the NVFP4 body and 0.012 on the trellis body (0.006 past one lane), and dropping any one
leg moves no entry by more than 0.002 on the NVFP4 body and 0.004 on the trellis body. The NVFP4 body's uniform entries
stay #690's served measurement, 1.128, 1.29, 1.363 and 1.344, which this one read within 0.011 (1.125, 1.295, 1.352
and 1.342), so that `aggregate` there reads what it read before; its mixed entries and every entry of the trellis
body's are this measurement's. Each row's entries at 1, 2, ... lanes extended, a held entry beside its measured value:

| lanes | NVFP4 base round | NVFP4 entries | trellis base round | trellis entries |
|---|---|---|---|---|
| 1 | 15,471 us | 1.128 | 14,875 us | 1.364 |
| 2 | 17,186 us | 1.159, 1.29 | 20,191 us | 1.214, 1.407 |
| 3 | 20,101 us | 1.112, 1.249, 1.363 | 24,586 us | 1.162, 1.502, 1.629 |
| 4 | 22,422 us | 1.132, 1.217, 1.344 (1.345), 1.344 | 28,785 us | 1.291, 1.391, 1.55, 1.606 |
| 5 | 25,485 us | 1.083, 1.196, 1.196 (1.188) | 37,291 us | 1.082, 1.204, 1.243 |
| 6 | 27,592 us | 1.109, 1.109 (1.106) | 40,415 us | 1.117, 1.154 |
| 7 | 30,647 us | 1 (0.999) | 45,270 us | 1.036 |
| 8 | 30,753 us | none | 47,018 us | none |

A round with extended lanes costs what a base round of its columns costs to within 2.5% on either body. On the NVFP4
body the step from 56 columns to 64 costs nothing or less: its 56-column rounds ran at about 2,775 MHz and its
64-column ones at about 2,815, in either order, near the card's 575 W limit, so 3+1 measured above the uniform round
of 4 lanes (1.345, where the served entry is 1.344 and this measurement read 1.342), 3+2 below 2+3, 2+4 below 1+5 and
1+6 at the base round of 7 lanes (0.999); those four are held. The trellis body's rows do not fall: its smallest step,
from the base round of 7 lanes to 1+6, is 0.036. The table is a figure at about 3.5-5K tokens: the ratio grows with
context (#690's finding), so on a shorter conversation the wider round costs every lane somewhat less than its entry
says, and on a longer one somewhat more.

**What this amends.** In ADR 0104, a round's shape is each lane's, a round may carry both, and its numerical class
reaches the mixed round; "lanes of both shapes in one round" is built, the only way to an extended shape past 4 lanes.
In ADR 0030, the plan's sixteenth line, `mixed_graphs`. In ADR 0017, the shape counts' `mixed` label.

## Considered options

- **One shape for all of a round's lanes, as ADR 0104 has it**: at 3 to 8 lanes one predictable lane loses its
  extended rounds, and at 2 its neighbour pays its 16 columns.
- **Mixed rounds below 15 drafts, on GDN records of their own**: a new device buffer for windows the `rtx5090` profile
  does not run; the owner kept 15 alone.
- **A recorded patch of the vendored fold admitting one slot on several rows that commit nothing**: a two-lane pool
  without a retained slot would hold (1,1); the owner kept the op as it is, and such a pool holds no mix.
- **An eager mixed round where its graph is missing**: every launch of the round paid every round, for a mix the load
  could not capture; the uniform choice is a round the load already replays.
- **Capturing a mix's graphs when a round first needs it**: the KV pool is fixed at load, and VRAM taken later under
  WDDM risks the decode collapse; the plan holds them from the start.
- **A mixed lane held on the rows its leg reaches while every text holds a round**, the first reading: a lane beside a
  faster one read its likelihood over 51 to 85 rows, where the uniform cells' own lanes sit past 0.02 too.
- **A looser likelihood margin for a short lane**: a margin set by a window the extended round was never held on.
- **Every row a long leg reaches**: bounds read on texts ADR 0104 did not measure them over, and gigabytes of rows on the
  host.
- **The near tie over the mixed cells' own base rows**: on the longer leg they moved a pick at 9.0 where the uniform
  cells' move one at 12.0, so the extended round's own move at 22.0 failed there.

## Consequences

- A greedy answer can differ at a near tie where a lane's round changes shape or mix, as ADR 0104 says of the two
  shapes.
- The test stands on test-only seams: a teacher-forced round may name each lane's shape (the decode options'
  `forced_lane_shapes`, `TeacherForcing::lane_shapes`), which no serving path sets, and the verify logits readout of a
  mixed round (`ignis_verify_readout_mixed_logits`, `dflash2_readout::mixed_verify_logits`) reads each lane's rows at
  its group's stride and refuses after any other round, as the readout of one shape refuses after a mixed one.
- The mixed checks add no load: they ride the NVFP4 cell's load after its uniform ones, five to six minutes more (323 s
  on 2026-11-22, 36 s of it speculation off's texts; the NVFP4 test 431 s, where the first reading's took 304 to 314),
  and the trellis cell's, bit for bit, under three minutes (162 s on 2026-11-23, 32 s of it the texts; the trellis test
  272 s, 120 s without them and 450 s with the method of the NVFP4 cell). A
  mixed prompt must run past the 1,408 tokens a leg may ride on, or the test fails before its legs.
- The bounds are the RTX 5090's on the NVFP4 body under FP8 exception weights over hq-e8-2b. A trellis body runs mixed
  rounds from #720, held to the round identity test's tokens at 8 lanes and, on its class test's cell, bit for bit to
  the uniform rounds' rows, which follow a round's columns there.
- The cost tables are the RTX 5090's at about 3.5-5K tokens of context. The ratio grows with context, so on a longer
  conversation the wider round costs every lane somewhat more than its entry says, and `aggregate` and the grant take it
  a little more readily than it pays there; on another card both tables are unmeasured, the trellis body's indicative
  for the 16 GB and 24 GB cards (its measurement loaded the model at 262,144 tokens of context, where the `rtx-24gb`
  profile serves 1,048,576).
