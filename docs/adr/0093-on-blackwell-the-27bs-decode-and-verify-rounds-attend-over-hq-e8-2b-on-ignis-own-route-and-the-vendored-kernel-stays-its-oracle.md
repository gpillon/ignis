---
status: accepted
---

# On Blackwell the 27B's decode and verify rounds attend over hq-e8-2b on ignis's own route, and the vendored kernel stays its oracle

On Blackwell the 27B's attention over hq-e8-2b in the decode and verify rounds runs on a route of ignis's own, an
adapted op (ADR 0051 tier 2) in `kernel/src/` beside the vendored kernel, which is not edited and stays its oracle:
ADR 0031's option (b), ignis's own implementation with the vendored op as the oracle. The vendored kernel had
already taken ADR 0031's option (a), a recorded patch, its throughput form, and
[its finding](../findings/2026-09-24-hq-verify-decode-cost.md) left it latency-bound at 8 warps per SM, 3.9 to 4.9
times its MMA floor, naming the next step: more warps per SM. The route runs one block of 20 warps per SM, twelve
decoding keys and values into a ring of shared-memory tiles while eight run the matrix products.

- **The vendored kernel stays in the subtree with both its forms**: its reference hq route is the oracle, and its
  throughput form is the comparison and the route of every card and every call that does not take ours.
- **The route's contract is the reference route's bytes.** Each score keeps the reference's matrix products in its
  key order, each row maximum is the exact maximum of its halves and each row sum is rebuilt in the reference's
  order. `ignis_kernel_hq_verify_exact_test` holds every partial byte and every cache byte the route writes to the
  reference route's, beside the throughput form, and `ignis_hq_codec_kv_rows_test` holds its lean decoder to the
  group and per-thread decodes on the codec's corpora. With the bytes the same, no quality gate is re-run.
- **What is taken and what is ours.** The kernel takes from the reference's hq route, at the commit the manifest
  pins, its per-lane offsets, neutral partials, bounds, split arithmetic, the fused append's per-row body and the
  epilogue; the warp roles, the ring and the split of the matrix products are ignis's, and the decoder is ignis's
  own. The launcher restates the vendored small-T call it stands in for, launches the route on the vendored hq
  route's grid and arguments, and then launches the vendored split reducer, unchanged. The file headers and
  `kernel/NOTICE` name the upstream, its commit, its paths and its licence.
- **A load chooses it, by card.** An hq-e8-2b load of the 27B on a card of compute capability 12 takes it for every
  decode and verify round, captured or eager, when the binary holds the route's image for that card. An RTX 30 or
  40 card, a 12.x card the `120a` code does not run on, a build without the card's arch-specific target, a BF16
  load and Flash-Next keep the vendored route, as do the prefill's attention and a prefill tail of 1 to 8 tokens; a
  build for an older card compiles the route as a trap that nothing launches. The load logs the route it runs.
- **The vendored route is a measurement switch away on Blackwell**: `IGNIS_HQ_DECODE_ATTENTION=vendored` (`ours`,
  or unset, is the default; any other value refuses the load, naming the variable), read once per load, not a
  configuration field.
- **Its win is shown per op, as ADR 0031 asks**
  ([finding](../findings/2026-11-14-the-warp-specialized-hq-attention-served-on-the-27b.md)): on the RTX 5090 the
  kernel alone takes 0.58 to 0.84 of the throughput form's time, with 19.95 resident warps per SM against 7.90, and
  served, rounds a second rise by 1.5% to 10.2% from 7K to 142K tokens of context at one, four and eight lanes, with
  the one-lane text and tokens a round unchanged.

## Considered Options

- **A second recorded patch to the vendored kernel**, option (a): the warp roles, the ring of barriers and one
  block per SM are a different decomposition, not one recognizable as the reference's kernel, which ADR 0031 sends
  to option (b).
- **The prototype's second route**, the accumulators spread over more warps without separating the decode: it won
  three short multi-lane cells by 1 to 5%, and one route needs no selection rule or threshold to keep tuned (owner).
- **Taking the vendored route out of the call site**: it is the oracle's home, the route of RTX 30 and 40 cards,
  and the comparison a measurement of the two needs.

## Consequences

- The vendored file stays under the manifest with its recorded patch, so a reference bump still brings the
  vendored kernel's fixes to the cards and calls that run it, while the Blackwell route is ignis's to maintain. A
  change to the vendored kernel's partials contract (the layout, the masks, the neutral partials, the fused append,
  the residual window, the rotated frame) has to reach the route too; the exactness test sees it as a byte
  difference.
- Every Blackwell card class takes the route, the 16 GB artifact included. Its correctness does not depend on the
  SM count, since its split capacity comes from the visible keys alone, but its speed is measured on the RTX 5090
  alone.
- The VRAM plan does not move: the route allocates exactly the vendored call's tensors.
