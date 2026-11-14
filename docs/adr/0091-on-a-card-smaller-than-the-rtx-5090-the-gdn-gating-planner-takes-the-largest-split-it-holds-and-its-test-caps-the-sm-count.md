---
status: accepted
amends: [ADR-0077]
---

# On a card smaller than the RTX 5090 the GDN gating planner takes the largest split it holds, and its test caps the SM count

ADR 0077 admitted the GDN gating planner's portability patch (#478): the
planner asks CUDA's occupancy query for each cooperative kernel's CTAs per SM
and admits a split-K schedule only for a grid the card holds resident. What it
did where a route's own schedule is not resident stayed the reference's, and
on a card with fewer SMs than the RTX 5090's 170 that was wrong three ways
(GitHub #602, found by a fork running an RTX 5080). The projection fell back
straight to the unsplit route, which carries the tensor core's truncation over
all of K in one FP32 accumulator, skipping the smaller splits the card holds:
on an RTX 5080 (84 SMs) every full 1,024-token prefill chunk of the 16 GB
profile ran unsplit where split4 fits. The 27B's fused norm/control route at 1
to 8 columns took cooperative split40 without asking, and threw on a card
below 120 SMs. The capacity query asked each route only at its last width, so
on the RTX 5080 the 27B's control interval through 4,097 columns answered
30,720 bytes while an 896-column call allocates 2,752,512 on split8; the
engine's reservation held it only through the slack of buffers sized at the
chunk.

The owner decided on 2026-11-14 that, on ADR 0077's terms:

- **A card that cannot hold a route's own split-K schedule** takes the largest
  smaller split of the model's family it holds (the 27B's split8, split4,
  split2; the 35B's split16, split8, split4, split2), never one larger than
  the route's own, and the unsplit route only when none fits. The 27B's norm
  route takes split40 only where it is resident, and the composed route
  otherwise, as the 35B's split32 route already fell back.
- **The capacity query covers every width**: each cooperative route is asked
  at its last width in the interval and at every multiple of its column tile
  inside it. These widths hold every peak: residency changes only where the
  column-tile count does, and a fixed schedule's workspace grows with the
  width.
- **The gating test runs a smaller card's routes on the RTX 5090** under a
  test-only cap on the SM count the planner's residency test multiplies by: a
  host setter beside the planner that no engine path and no wrapper calls,
  while the CTAs per SM still come from the occupancy query on the card. The
  terms admit it because with no cap the 5090 takes the reference's branch
  everywhere: no route falls back there and every capacity answer is
  unchanged, which the test checks.

## Considered Options

- **The fork's fixed fallback list, split4, split2, unsplit**: it picks the
  same routes on every card the issue models, but it stands for a rule, the
  family's smaller splits in turn, and for the 35B it would skip split8.
- **Keeping the last-width capacity and the engine's slack**: once a route
  runs smaller splits inside it, its peak sits just before a column-tile
  boundary, where the argument from the slack no longer reaches.
- **Showing a smaller card's routes only on such a card**: none is at hand
  where the gating test runs; the cap runs, on the 5090, the planner's choice,
  the capacity, the launch and the workspace high-water of an 84-SM and a
  36-SM card.

## Consequences

- The planner, its header (patched for the first time) and the gating test
  carry #602 parts in their recorded patches. On the 5090 every branch is the
  reference's, so `kernel/vendor/VENDOR.md`'s table of Ignis-patched
  behaviours gains no row.
- On a card that falls back, a GDN layer's scratch grows to the query's exact
  peak: by 2,721,792 bytes for the RTX 5080's 27B at a 1,024-token chunk. On
  the RTX 5090 the reservation does not move.
- The test runs its cases at the card's own SM count, then under caps of 84
  (the RTX 5080) and 36 (the RTX 5060 Ti) below it. Each line names its
  pass's SM count, a `FALLBACK` line names each case the planner runs on a
  fallback, and nothing is skipped any more.
