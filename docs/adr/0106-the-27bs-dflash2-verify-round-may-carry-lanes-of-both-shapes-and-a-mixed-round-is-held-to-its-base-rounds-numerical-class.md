---
status: proposed
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
- **Its default.** `aggregate` on every profile. The `rtx5090` profile's becomes `mixed` only once the served gate
  passes: a JSON answer beside three conversations at four lanes at least 10% faster together than the build of #684, no
  conversation lane more than 15% below it. The trellis bodies keep `aggregate` until a run on a 16 GB or 24 GB card;
  given `mixed`, a trellis body at 15 drafts runs it. A load given `mixed` that holds no mix runs `aggregate`, and its
  startup line names both: a load of DFlash2 at 8 to 14 drafts, one whose pool's lanes and GDN slots hold no mix (which
  a WARN at load names, with the lanes and the slots), and one whose plan asked for none under `IGNIS_MIXED_ROUNDS=0`; a
  load whose latest capture ran under that switch takes `aggregate`'s choice too. At 7 drafts or fewer `mixed` is taken
  and changes nothing, the load holding one shape; the 27B's MTP head refuses it by name, as it refuses every shape
  policy (#690).
- **The cost table.** `spec.shape_period_ratios` becomes a table by lane count (1 to 7) and extended lanes, each entry
  the period of the round with that many lanes extended over the base round at the same lane count; `aggregate` reads
  its uniform entries at 1 to 4 lanes and `mixed` reads every entry. Its default, `auto`, is the loaded body's own
  table, resolved at load where the body is known (`PeriodTableSetting::of_body`), as `model.exception_weights` and
  `model.rope_scaling` resolve theirs under ADR 0046's one declaration: a table named whole stands on either body, the
  four-value form's mixed entries are the loaded body's, and the four-value form at its earlier default reads as `auto`,
  so a file an earlier build generated names no table. The NVFP4 body's: its uniform entries the RTX 5090's measured
  1.128, 1.290, 1.363 and 1.344; four of its mixed entries, 1+1, 1+3, 2+2 and 1+5, those mixes' periods measured with
  their graph build, and each of the other eight the measured base round of the mix's columns over the base round at its
  lane count, a mixed round with its graph build having been measured to cost what a base round of its columns costs
  (the three mixes of 64 columns, where no base round of 8 lanes was measured, one 8-column step past 56). The trellis
  body's keeps the NVFP4 body's uniform entries, so that `aggregate` there reads what it read before mixed rounds; its
  mixed entries scale the NVFP4 body's excess over 1 by 2.09, what the trellis body's base round's 8 more columns cost
  at one and two lanes against the NVFP4 body's (1.218 over 1.104), each held no higher than the uniform entry at its
  lane count and no lower than the mix of one fewer extended lane, so that no step of the grant is free. Both are
  provisional until measured against measured rounds of the same columns, never read off a line between lane counts: the
  base round's cost rises in steps with its columns (5 lanes 24,668 us and 7 lanes 29,770, where a line between lane
  counts said 23.9 and 27.7 ms). The four-value form stays accepted as the uniform entries.
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
round at its mix; the near tie read the way ADR 0104 reads it; the margins unchanged. How the test does so is proposed
here, for the owner to confirm with this ADR:

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

**What this amends.** In ADR 0104, a round's shape is each lane's, a round may carry both, and its numerical class
reaches the mixed round; "lanes of both shapes in one round" is built, the only way to an extended shape past 4 lanes.
In ADR 0030, the plan's sixteenth line, `mixed_graphs`.

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
  on 2026-11-22, 36 s of it speculation off's texts; the NVFP4 test 431 s, where the first reading's took 304 to 314). A
  mixed prompt must run past the 1,408 tokens a leg may ride on, or the test fails before its legs.
- The bounds are the RTX 5090's on the NVFP4 body under FP8 exception weights over hq-e8-2b. A trellis body runs mixed
  rounds from #720, held to the round identity test's tokens at 8 lanes; its class test's mixed section, read once on
  its cell (2026-11-22), sits inside every bound of the row and the likelihood and past five relative clauses of the
  class, which waits on the owner's reading.
