---
status: accepted
---

# The 27B's DFlash2 verify round may carry lanes of both shapes, and a mixed round is held to its base round's numerical class

A DFlash2 verify round of the 27B at 15 drafts may carry lanes at the extended shape beside lanes at the base shape, a
**mixed round**: of the two shapes
[ADR 0104](0104-the-27bs-dflash2-verify-round-goes-past-8-columns-a-lane-and-is-held-to-its-base-rounds-numerical-class.md)
gives a load past 7 drafts, the shape a round is handed before it launches is each lane's. With one shape for all a
round's lanes, one predictable lane drags every other lane into 16 columns or is refused: at 3 to 8 lanes it loses its
extended rounds, and at 2 its neighbour pays its 16 columns. A mixed round is held to its base round's numerical class
as ADR 0104 holds the extended round: its method, its margins and its window.

- **The rounds.** A round is uniform base, uniform extended or one of twelve mixes: 1 to 3 lanes at the extended shape
  beside 1 to 6 at the base shape, at 2 to 7 lanes and at most 64 columns. A round of 8 lanes is at the base shape,
  and a mix is the only way to an extended shape past 4 lanes.
- **At 15 drafts only.** Each group's GDN records lie in its own shape's records over the one backing both shapes
  share, with no record region of their own and no new buffer, and only at a window of 15 is a group's record row its
  column there: below it the base group's records and the extended layout's collide. A load at 8 to 14 drafts holds no
  mix: its rounds are uniform and it captures no mixed graph. Records of their own would be a new device buffer for
  windows the `rtx5090` profile does not run, so the owner kept 15 alone.
- **A mix of `w` extended lanes beside `n` base lanes takes `2w + n` GDN slots.** The base group's fold runs the
  vendored op as it is, behind `2w` leading rows that commit no column, and the op wants distinct state slots on every
  row, those rows included, on slots no base lane holds. So a mix is captured and runs only where the load's lanes
  hold `w + n` and the pool's GDN slots, its decode lanes and its device retained slots, hold `2w + n`: the 27B's
  default 8 lanes hold all twelve, and 2 lanes with no retained slot, the `rtx-16gb` profile's, none. A recorded patch
  of the fold admitting one slot on several rows that commit nothing would give that pool a mix; the owner kept the op
  as it is, and no vendored file changes.
- **The grant.** The shape policy `mixed` leaves the shape rule, what each lane asks, unchanged, and at 2 to 7 lanes
  grants the asking lanes the extended shape one by one: a lane is granted while the round with it extended too stays
  within 64 columns, and when its expected extra tokens exceed what that round costs every lane in it over the round
  without it. Granting every ask in the caller's order bought nothing on a moderately predictable answer beside a
  conversation: the grant has to refuse such a lane. It stops at the last mix the load holds a graph of, so a pool
  that holds some of them keeps their gain. A lane under a grammar is granted like any other. `all` and `aggregate`
  are unchanged and never run a mixed round.
- **The cost table** the grant reads (`spec.shape_period_ratios`), by lane count and extended lanes, is by default
  the loaded body's own measured table, resolved at load where the body is known, under ADR 0046's one declaration.
  Where a mixed entry measured below the entry before it in its row, above its row's uniform entry or under 1, the
  default holds it at that bound, so that under a default no step of the grant is free; the uniform entries, which
  `aggregate` reads, stand as they are, so that it reads what it read before.
- **Its default.** `mixed` on the `rtx5090` profile for the NVFP4 body, the one the gate served, and `aggregate` on
  every other profile; the trellis bodies keep `aggregate`, under `rtx5090` as under their own profiles, and a trellis
  body at 15 drafts runs `mixed` where the operator names it. A run on a 16 GB or 24 GB card would be a new decision.
  The served gate ([finding](../findings/2026-11-23-the-27bs-mixed-rounds-served-on-the-rtx-5090.md)) missed by the
  letter its line of no one-lane or free-text cell more than 2% below the build before it, and the owner made `mixed`
  the default all the same: one lane forms no mix, and seeded, a request alone wrote the same text in the same rounds
  under both builds. A load given `mixed` that holds no mix, or whose DFlash2 window is not 15, runs `aggregate`, and
  its load says so; beside the 27B's MTP head the policy is ignored, as under `off`, as every draft setting is where
  its speculation does not run.
- **The mixed graphs.** A mixed round replays a verify graph and a drafter append graph of its mix, both or neither,
  captured at load for every mix the load holds where mixed rounds are asked, and held from the start as a line of the
  VRAM plan ([ADR 0030](0030-device-memory-reserved-at-load.md)), `mixed_graphs`, at a measured size of each body:
  captured when a round first needs them they would take VRAM after the KV pool is fixed, which under WDDM risks the
  decode collapse. The grant never takes a mix with no graph, and a round forced at one (the test seam) runs at the
  uniform shape `aggregate` would choose, counted, never refused and never eagerly: eager, every launch of the round
  would pay every round, for a mix the load could not capture, and the uniform choice is a round the load already
  replays. The driver keeps the graphs' pages after a teardown, so a reload adds the line back.
- **The shape counts.** `shape="mixed"` joins the label set of `ignis_speculative_shape_rounds_total`, a family under
  [ADR 0017](0017-prometheus-metrics.md), with no new family: a lane of a mixed round counts once, as mixed and at
  neither of the others, so the three series sum to `ignis_speculative_rounds_total` and a share of them is a share of
  lane-rounds, which says how often rounds mix; counted at its lanes' own shapes, no round would count as mixed. The
  per-position drafted and accepted families keep counting each lane's own drafts, an extended lane's up to its window
  and a base lane's up to its own.
- **The class on the NVFP4 body.** Each group runs its own shape's routes on its slice and the ops that see only
  columns run once over all of them, so a mixed lane's rows are its base round's up to summation order, as an extended
  round's are. The test (`crates/core/tests/extended_round_class_gpu.rs`) holds every mixed row, of either group, to
  the base row at the same text index and lane count exactly as ADR 0104 holds the extended round, the margins
  unchanged: each lane over a uniform cell's window, on a leg long enough for every lane to reach a uniform cell's row
  count, every round at its mix; the near tie read as ADR 0104 reads it, over the base rows of the kind's uniform
  cells at 1 to 4 lanes, not over the mixed cells' own base rows nor pooled with them. On a shorter window of prose
  the likelihood sits past its margin in the uniform cells too, and a looser margin for a short lane would be set by a
  window the extended round was never held on; every row a longer leg reaches would read the bounds on texts ADR 0104
  did not measure them over, the base round's own near tie widening there, with gigabytes of rows on the host; and the
  mixed cells' own base rows move a pick at a narrower gap than the uniform cells', so the extended round's own move,
  which ADR 0104's reading passes, fails against them.
- **The class on the trellis body.** A mixed row is held, bit for bit, to the row of a uniform round of its lane's
  shape at its round's columns, and the uniform rounds to the class by ADR 0104's cells, its margins unchanged. There
  a lane's rows follow its round's columns, not its lane count, the trellis projection routing by the call's token
  count, so a base leg at the lane count is no reference: read by the NVFP4 body's method, a mixed path with no defect
  sat past five relative clauses of the class. No trellis mixed row is read against a base row, so none of ADR 0104's
  margins is read there.

The measurements behind the class, the cost tables and the graphs' size are a
[finding](../findings/2026-11-23-the-27bs-mixed-rounds-numerical-class-on-the-rtx-5090.md).

## Consequences

- A greedy answer can differ at a near tie where a lane's round changes shape or mix, as ADR 0104 says of the two
  shapes.
- The bounds are the RTX 5090's on the NVFP4 body under FP8 exception weights over hq-e8-2b. A trellis body's mixed
  rounds are held to the round identity test's tokens at 8 lanes and, on its class test's cell, bit for bit to the
  uniform rounds' rows, which follow a round's columns there.
- The cost tables are the RTX 5090's at about 3.5-5K tokens of context. The ratio grows with context, so on a shorter
  conversation the wider round costs every lane somewhat less than its entry says and on a longer one somewhat more,
  and there `aggregate` and the grant take it a little more readily than it pays; on another card both tables are
  unmeasured, the trellis body's indicative for the 16 GB and 24 GB cards.
