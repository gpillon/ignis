---
status: accepted
---

# On a card smaller than the RTX 5090 the GDN gating planner takes the largest split it holds, and its test caps the SM count

The vendored GDN gating planner admits a split-K schedule only for a grid the card holds resident (ADR
0077). What it does where a route's own schedule is not resident is decided here, on ADR
0077's terms, because the reference's answer was wrong three ways on a card with fewer SMs than the RTX 5090's. The
projection fell back straight to the unsplit route, which carries the tensor core's truncation over all of K in one
FP32 accumulator, skipping the smaller splits the card holds. The 27B's fused norm/control route at 1 to 8 columns
took cooperative split40 without asking, and threw on a card below 120 SMs. And the capacity query asked each route
only at its last width, which can be far below what a narrower call on a smaller split allocates, so the engine's
reservation held it only through the slack of buffers sized at the chunk.

- **A card that cannot hold a route's own split-K schedule** takes the largest smaller split of the model's family
  it holds (the 27B's split8, split4, split2; the 35B's split16, split8, split4, split2), never one larger than the
  route's own, and the unsplit route only when none fits. The 27B's norm route takes split40 only where it is
  resident, and the composed route otherwise, as the 35B's split32 route already fell back.
- **The capacity query covers every width**: each cooperative route is asked at its last width in the interval and
  at every multiple of its column tile inside it. These widths hold every peak: residency changes only where the
  column-tile count does, and a fixed schedule's workspace grows with the width.
- **The gating test runs a smaller card's routes on the RTX 5090** under a test-only cap on the SM count the
  planner's residency test multiplies by: a host setter beside the planner that no engine path and no wrapper
  calls, while the CTAs per SM still come from the occupancy query on the card. ADR 0077's terms admit it because
  with no cap the RTX 5090 takes the reference's branch everywhere: no route falls back there and every capacity
  answer is unchanged, which the test checks. The test skips nothing.

## Considered Options

- **A fixed fallback list, split4, split2, unsplit**: it picks the same
  routes on every card the issue models, but it stands for a rule, the family's smaller splits in turn, and for the
  35B it would skip split8.
- **Keeping the last-width capacity and the engine's slack**: once a route runs smaller splits inside it, its peak
  sits just before a column-tile boundary, where the argument from the slack no longer reaches.
- **Showing a smaller card's routes only on such a card**: none is at hand where the gating test runs; the cap
  runs, on the RTX 5090, the planner's choice, the capacity, the launch and the workspace high-water of an 84-SM
  and a 36-SM card.

## Consequences

- The planner, its header and the gating test carry these parts in their recorded patches. On the RTX 5090 every
  branch is the reference's, so `kernel/vendor/VENDOR.md`'s table of Ignis-patched behaviours gains no row.
- On a card that falls back, a GDN layer's scratch grows to the query's exact peak, by about 2.7 MB for the RTX
  5080's 27B at a 1,024-token chunk. On the RTX 5090 the reservation does not move.
