---
status: proposed
amends: [ADR-0030, ADR-0104]
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
  the asking lanes are granted the extended shape one by one: the highest expected extra tokens first, at a tie the
  lane refused longest; a lane that asks with no extended round among its recent rounds after those with an estimate,
  granted when the columns allow, so that its estimate has a round to read. A lane is granted while the round with it
  extended too stays within 64 columns, and when its expected extra tokens exceed (that round's period ratio over the
  current round's, less one) times the tokens every lane of the round is expected to take, each lane's expectations
  read over its own last `spec.shape_estimate_rounds` verify rounds as under `aggregate`. A lane under a grammar is
  granted like any other. `all` and `aggregate` are unchanged and never run a mixed round. Granting every ask in the
  caller's order, as the prototype did, bought nothing on a moderately predictable answer beside a conversation (+0.6%,
  inside the spread, the conversation 5.0% slower): the grant has to refuse such a lane.
- **Its default.** `aggregate` on every profile. The `rtx5090` profile's becomes `mixed` only in the commit that
  records the served gate's pass (#722: a JSON answer beside three conversations at four lanes at least 10% faster
  together than the build of #684, no conversation lane more than 15% below it). The trellis bodies keep `aggregate`
  until a run on a 16 GB or 24 GB card. `mixed` on a load without a mix (7 drafts or fewer, 8 to 14, the MTP head,
  Flash-Next) is accepted and runs as `aggregate`.
- **The cost table.** `spec.shape_period_ratios` becomes a table by lane count (1 to 7) and extended lanes, each entry
  the period of the round with that many lanes extended over the base round at the same lane count; `aggregate` reads
  its uniform entries at 1 to 4 lanes, today's measured 1.128, 1.290, 1.363 and 1.344, and `mixed` reads every entry.
  Its defaults are each body's, measured on the RTX 5090 with the final build, the NVFP4 body's (#721) and the trellis
  body's (#720; indicatively a trellis body's extended round costs 1.35 to 1.40 times its base round at one lane,
  against 1.13 on the NVFP4 body); until those land the mixed entries are the prototype's measured mixes. Every entry is
  measured against measured rounds, never read off a line between lane counts: the base round's cost rises in steps
  with its columns (5 lanes 24,668 us and 7 lanes 29,770, where a line between lane counts said 23.9 and 27.7 ms). The
  four-value form stays accepted as the uniform entries.
- **The mixed graphs, a sixteenth line of the VRAM plan** (#714). A mixed round replays a verify graph and a drafter
  append graph of its mix, both or neither, captured at load for every mix the load holds where mixed rounds are asked;
  a mix with no graph runs its rounds at the uniform shape `aggregate` would choose, counted and logged, never eagerly
  and never refused. The graphs are the plan's line `mixed_graphs`, in the leaf's reservations and their mirror in the
  core: 5 MiB a mix and 4 MiB beside them, in whole 2 MiB pages, as the RTX 5090 measured them at capture and on the
  process's WDDM counter, 64 MiB for the twelve (at most 114 KV pages fewer at the `rtx5090` default) and 0 on a load
  that captures none. The driver keeps the graphs' pages after a teardown, so a reload adds the line back. This amends
  [ADR 0030](0030-device-memory-reserved-at-load.md) as
  [ADR 0101](0101-on-windows-the-vram-plan-reserves-the-w4a4-tma-products-descriptor-slots-as-a-line-of-their-own.md)
  did for the fifteenth line, by a new ADR rather than an edit in place, which only the owner decides.

**A mixed round is held to its base round's numerical class, with ADR 0104's method and margins.** Each group runs its
own shape's routes on its slice and the ops that see only columns run once over all `16w + 8n` of them, so a mixed
lane's rows are its base round's up to summation order, as an extended round's are. The test
(`crates/core/tests/extended_round_class_gpu.rs`, on ADR 0104's NVFP4 load, its graphs captured again with the
mixes') offers speculation off's own greedy text as each mixed round's drafts, placed by group, and replays each mix's
captured graph on the drafter's own drafts, both teacher-forced to the text, and holds every mixed row, of either group,
to the base row at the same text index and lane count by ADR 0104's bounds: a row within 256 BF16 ulps at the base
row's largest magnitude, a lane's median row within 32, each lane's mean log-probability of the text within 0.02 nats
a token, the class at each lane count and over a kind's lane counts together (1.25 at the median and the 90th
percentile, 1.5 at the widest row, no more moved picks than 1.25 times the base rows' plus 2), the class's near tie
and the control one index off over a kind's lane counts together. Every one of the twelve mixes runs in both kinds of
round, its last lanes extended so that the round's rows are not the caller's order, its lanes on seven prompts of their
own, and its leg rides the mix while every lane's text holds a whole round of its shape, so that every row is a round's
at that mix and lane count. Measured on the RTX 5090 (NVFP4 body, FP8 exception weights, hq-e8-2b) on 2026-11-22, over
8,040 forced and 8,060 graph mixed rows at 2 to 7 lanes:

| bound | margin | forced mixed rounds | graph mixed rounds |
|---|---|---|---|
| widest row, mixed against base | 256 ulps | 1.0 | 115.2 |
| a lane's median row | 32 ulps | 0.0-0.5 | 0.0-15.0 |
| mean log-probability, apart | 0.02 nats | 0.0010 | **0.0256** on one lane (below) |
| class ratio, median / 90th percentile / widest | 1.25 / 1.25 / 1.5 | 1.00 / 1.00 / 1.00 | 1.01 / 1.00 / 1.00 |
| the same, by lane count | | 1.00 / 1.00 / 1.00 | 1.00-1.03 / 0.97-1.01 / 1.00-1.13 |
| picks moved from the plain rows, mixed and base | 1.25 x + 2 | 248 and 248 | 108 and 97 (30 and 25 at 6 lanes, bound 33) |
| widest moved gap, mixed over base; between them | 2 x, floor 16 | 12.0 over 12.0; none | 9.0 over 9.0; 11.0 |
| control's median, over the right rows' and the typical bound | 2 x | 13.5x, 5.0x | 14.4x, 5.2x |

Forced, every mixed row is its base row within 1 ulp, both groups' (the uniform shapes' forced rows already agree so at
2 to 4 lanes); on graph rounds a lane's row at one text index can stand at another column of its round in the mixed leg
than in the base leg, each leg's drafts and accepted counts its own, as between a uniform graph cell's two shapes. One
bound fails, on one lane, in both runs of 2026-11-22: on graph rounds the prose prompt's lane, extended at the mix 3+2,
reads its text's mean log-probability 0.0256 nats a token below the base rows', over 79 rows (its leg ended with the
table prompt's text); the next furthest is the same lane at 2+1 and 3+1, 0.0194 over 51. Over the first 64
rows that lane reads 0.0177, as the same prompt's lane of ADR 0104's uniform extended graph cell at 4 lanes does there
(0.0124 over its 210 rows), and that cell's lane at one lane reads 0.0242 there: the excess is the extended round's on
a short window of an uncertain text, which a mixed lane's shorter leg reads whole, not the mix's. Whether the likelihood
is read over a window every lane reaches, or the margin stands, is the owner's to decide with this ADR; the test holds
0.02, as ADR 0104 confirmed it, and fails on that lane until then.

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

## Consequences

- A greedy answer can differ at a near tie where a lane's round changes shape or mix, as ADR 0104 says of the two
  shapes.
- The test stands on test-only seams: a teacher-forced round may name each lane's shape (the decode options'
  `forced_lane_shapes`, `TeacherForcing::lane_shapes`), which no serving path sets, and the verify logits readout of a
  mixed round (`ignis_verify_readout_mixed_logits`, `dflash2_readout::mixed_verify_logits`) reads each lane's rows at
  its group's stride and refuses after any other round, as the readout of one shape refuses after a mixed one.
- The mixed checks add no load: they ride the NVFP4 cell's load after its uniform ones, three to four minutes more. A
  mixed cell's lane is held on at least 32 rows where a uniform cell's is on 64, its leg ending with its fastest lane's
  text (the fewest measured: 104 rows forced, 51 on graph rounds).
- The bounds are the RTX 5090's on the NVFP4 body under FP8 exception weights over hq-e8-2b. A trellis body's mixed
  rounds are held by #720 before they run there.
