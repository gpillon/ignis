# Findings archive

The findings that no longer stand as they are, kept as history: the
`superseded` ones next to what replaced them, the `archived` ones, closed lines
of research, with why they closed. They are listed by topic, newest first, and
their files stay in this directory, since the guide and the other records link
them. The usable findings are in [README.md](README.md);
[`docs/agents/findings.md`](../agents/findings.md) is the convention.

`mk/findings-index.py` generates this file (`make records`): it is never edited
by hand.

## 27B decode round, kernels and speculation (`decode-27b`)

- 2026-09-24 [Copy drafting at width 8 buys 0–8% on real agent traffic: thinking dominates the output and DFlash2 already rides the copyable spans](2026-09-24-copy-drafting-on-agent-traces.md), archived: Closed, not built: on 13,102 real agent turns a copy drafter inside the width-8 verify buys 0-8% at one lane, under the +8% bar, because thinking is 73.5% of the output and DFlash2 already rides the copyable spans.

## Flash-Next: MoE experts, residency, MTP, verify rows (`flash-next`)

- 2026-10-08 [Flash-Next routed experts at decode: staging each SM's work items through shared memory takes a layer from 25.4 to 20.9 µs at one token alone, but served it is within noise and moves every text, so it ships off (NO-GO)](2026-10-08-flash-next-routed-experts-staged.md), archived: Closed as a NO-GO: staging routed-expert work through shared memory takes a layer from 25.4 to 20.9 us alone, but served it is within noise and not bit-exact, so it moves every text and ships off behind IGNIS_FN_MOE_STAGED=1.

## KV cache, formats and state tiers (`kv-and-state`)

- 2026-10-08 [Live moves a window at a time: the copies cost the other lanes almost nothing; what a move in's rounds pay is expert-cache misses](2026-10-08-live-moves-windowed.md), superseded by [expert misses after a live move are the text's](2026-10-08-expert-misses-after-a-move-are-the-text.md): Windowed at 12 MiB out and 16 MiB in, a live move stalls no lane (max +3-11 ms, was +314 ms) and the copies cost the rounds almost nothing; the move in's +45-58% ITL p50 was later shown to be the lanes' text, not the move.
- 2026-10-08 [A live move's PCIe contention on Flash-Next: the windowed disk move out holds the other lanes, a move in slows them, a synchronous KV-RAM move stalls them ~0.32 s](2026-10-08-live-move-pcie-contention.md), superseded by [Live moves a window at a time](2026-10-08-live-moves-windowed.md): A synchronous KV-RAM move of a 1.13 GB Flash-Next sequence stalled two decoding lanes ~0.32 s each way at ~3.5 GB/s and a move in cost them +50% ITL p50; windowed moves then removed the stall and explained the slowdown.

## Host, VRAM and GPU generations (`host-and-gpu`)

- 2026-10-31 [A Metal or AMD backend for the leaf: the roads, a seam at the segment, and trellis decode's ALU budget on an M4 Pro](2026-10-31-metal-and-amd-backends-seam-and-roads.md), archived: Exploration not followed (#439 parked as an idea, nothing run on Apple or AMD): a Metal or AMD backend would be native kernels behind today's C ABI, seamed at the segment, after a spike tests trellis decode on an M4 Pro.

## Decide: options, number, scalar, prompt and reuse (`decide-options`)

- 2026-09-21 [A scalar reads its legend, and only while the legend is narrow](2026-09-21-scalar-readout-legend-width.md), archived: Not implemented (spec 08): a grid readout reads its legend only while it is narrow (4/5 at 20 and 32 cells, 0/6 at 100), which is why the `scalar` that shipped generates and closes its own object instead.
- 2026-09-20 [A fixed-width `number` is wrong at exactly one width, and an image `state` shares nothing](2026-09-20-number-width-and-decide-e2e.md), superseded by [the number prompt declares an alignment](2026-09-21-the-number-prompt-declares-an-alignment.md), [a shared vision embedding](2026-09-20-shared-vision-embedding-fan-out.md): Both halves are replaced: a fixed-width `number` is wrong at many widths until the prompt names an alignment (47 read 4700 at five digits), and an image fan-out no longer re-encodes its image per question.

## Decide: point and box, attention heads (`decide-pointing`)

- 2026-09-21 [The point is assembled as it is written, not read out of the latent](2026-09-21-the-point-is-assembled-as-it-is-written.md), archived: Negative (spec 12): where x's first digit is written the latent holds x to about one digit (20.9 units against a 25.0 baseline), so a probe lands inside 97 of 240 against the chain's 218: each forced digit decides the point.
- 2026-09-21 [The grid a bigram can name is 9 x 9, and that is too coarse to point with](2026-09-21-the-grid-a-bigram-can-name.md), archived: Negative (spec 11 E0): 114 of 676 uppercase bigrams are not single tokens, so a one-label grid is at most 9 x 9 contiguous (11.1% of the side per cell), far coarser than the digit chain's 2.1%; single letters have no holes.
- 2026-09-21 [A declared grid is read to one part in ten, however finely it is declared](2026-09-21-a-declared-grid-is-read-to-one-part-in-ten.md), archived: Negative (spec 11 E1): asked which of N strips holds the target, the model errs by 9-15% of the axis at every width (11-26 strips) against the chain's 0.08%; and answer mass (0.98) certifies an answer's shape, not its correctness.

## Decide: locate (`decide-locate`)

- 2026-09-26 [Attention finds the paraphrased line and loses the quoted one: `locate` is a no-go](2026-09-26-locate-attention-no-go.md), superseded by [locate by a head vote is a go](2026-09-27-locate-by-head-vote-go.md), [the shortlist `locate`, spec 22's acceptance](2026-09-28-locate-shortlist-acceptance-partial.md): Replaced by the head vote: one head (L39.h12) at the copy scaffold found paraphrased lines better than labels (91.0% against 87.2%) but missed 43% of lexical questions, 73.3% to 91.3% overall, so spec 18 said no-go.

## Vision (`vision`)

- 2026-10-30 [The 27B's vision tower staged per encode and its embedding pool host-mapped: a text-only load's KV, bit-identical embeddings, +3 to +8 ms a 1,024-token encode under WDDM](2026-10-30-vision-tower-staged-and-pool-host-mapped.md), superseded by [The vision residency's acceptance](2026-10-30-vision-residency-acceptance.md): The prototype that staged the 27B's vision tower per encode and host-mapped its pool kept a 16 GB load's text-only KV pool, bit-identical, at +3 to +8 ms a 1,024-token encode; the built residency's acceptance replaces it.

## Harness, bench, build and transport (`harness-and-build`)

- 2026-09-19 [The codebase as an n-gram memory: the summary exists, the injection cannot use it](2026-09-19-codebase-as-ngram-memory.md), archived: Closed negative result: a use site picks its own definition out of 200 candidates 41.3% of the time at layer 19 of 64, but no injection made that a better prediction than a shuffled-value control, so the UVA gather fast path was not built.
