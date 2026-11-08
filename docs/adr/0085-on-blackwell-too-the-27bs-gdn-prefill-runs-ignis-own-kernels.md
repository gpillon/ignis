---
status: accepted
amends: [ADR-0080]
---

# On Blackwell too, the 27B's GDN prefill runs ignis's own kernels

ADR 0080 gave the 27B's GDN prefill on 8.x ignis's own kernels (#514) and kept Blackwell on the vendored chunked kernels: on the RTX 5090 ours were faster by 3-4% at 8K and 64K with the KLD intact (1.0036 trellis, 0.9816 NVFP4), but two entailment gate cases and the vision verify probe moved, so the switch waited for #490. Blackwell took ours in #536 (spec #531) by #514's rule, which replaces the vendored chunked kernels there only if the RTX 5090 measures ours faster with its KLD holding. It did, on Blackwell's default routes (the fused prefill, the FP8 band) and with one binary for both routes, as [the finding](../findings/2026-11-08-blackwell-gdn-prefill-on-ignis-kernels.md) records with its figures: the GDN prefill kernels alone ran 2.4 to 2.6 times faster; the time to first token was shorter on both 27B entries over alternated legs, on the NVFP4 entry within each regime of its legs (the finding says why its registered rule's second part failed there); and the full-vocabulary KLD against BF16 stayed within its interval on both. So the GDN prefill has no option: it is the route on every card that runs the 27B.

#514 made the switch once and reverted it, because three tests of the GPU profile moved. None was a quality loss of ours, and each has its rule since:
- **Two gate cases of the entailment regression set**, 3 and 17, crossed the decision line. The set had been tiered on the vendored route alone, and any change of the numerics moves its cases: the int8 prompt-attention band, a route that passed the KLD gate, moved them more than ours did and lost a gate case of its own. Since #535 a case is a gate only when it is clear of the decision line on every route measured, so 3, 10 and 17 are watch cases.
- **The multimodal speculative test's near-tie probe** failed a split between the decode round's pick and the verify round's. Both its probes took the decode round's shape, while the verify round runs other activation routes at its width (ADR 0016); with both rounds on one route the text was identical. Since #534 the split is judged by the two round shapes, each computed outside the rounds.
- **The shared vision workspace's recorded tokens** are exact tokens, which any change of the numerics moves. They were re-recorded on the new default.

Its speed is unmeasured on the RTX PRO 6000 and the 16 GB RTX 50 cards, which run the RTX 5090's SASS. A greedy answer on Blackwell can differ from an earlier release's at a near-tie: correctness is self-checked, not byte-matched (ADR 0007). Flash-Next keeps its own vendored GDN layers. `IGNIS_GDN_PREFILL=vendored|ours` is the switch for a measurement, on every card, not a setting; the vendored kernels stay its other route and the leaf test's oracle.
