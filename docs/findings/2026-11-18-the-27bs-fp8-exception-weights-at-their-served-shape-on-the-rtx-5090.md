# The 27B's FP8 exception weights at their served shape on the RTX 5090: KLD 1.033x BF16's, inside the leg's 1.05, and the context a start keeps follows the VRAM free at start

- Kind: experiment
- Status: current
- Observed: 2026-11-18
- Last verified: 2026-11-18
- Scope: quality / the 27B's NVFP4 artifact with its nine exception tensors held in FP8 or BF16, KLD against the BF16
  checkpoint on set A at the shapes the RTX 5090 serves; serving / the VRAM plan at the `rtx5090` profile's shape and
  the context a start with no config file keeps
- Topic: artifacts-and-gates
- Related: [GitHub #682](https://github.com/gpillon/ignis/issues/682),
  [spec #672](https://github.com/gpillon/ignis/issues/672),
  [ADR 0065](../adr/0065-the-bf16-exception-tensors-can-be-held-in-fp8.md),
  [ADR 0082](../adr/0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md),
  [the BF16 exception tensors in FP8](2026-10-21-the-bf16-exception-tensors-in-fp8.md),
  [the decide regression sets under both exception weights](2026-11-18-the-decide-regression-sets-under-both-exception-weights.md)
- Superseded by: none
- Summary: RTX 5090, 27B: FP8 exception weights at their served shape read a KLD 1.0328x BF16's (limit 1.05), top-1 -0.21 points; 917,504 tokens hold unadapted with 28,483 MiB free at start under FP8 and 28,993 under BF16.

Written on 2026-11-27 from the ADR that held this measurement, when it was absorbed into ADR 0065 and left the tree
(it is `0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md` in
`docs/adr/REMOVED.md`). Nothing was measured again. The records were in the main checkout's `.scratch/672/kld/`.

## Question

`model.exception_weights: auto` holds the 27B's exception tensors in FP8 on a card of compute capability 12.0
(ADR 0065). Before that became the default, two questions:

- Does FP8 pass ADR 0082's KLD leg for a route made a default, the KLD of FP8 at most 1.05x BF16's and top-1 within
  0.5 points, by the point estimate, at the shapes the RTX 5090 serves?
- What context does a default start keep on the RTX 5090 under each exception weights?

## Evidence

**The KLD leg.** On the RTX 5090, set A was dumped four times from one binary in one session: chunk 2,048, hq-e8-2b,
each model loaded at the named context with its table, and each dump scored against the BF16 checkpoint over the full
vocabulary.

| dump | KLD | top-1 |
|---|---:|---:|
| BF16, 786,432 tokens, yarn:3 | 0.05211 | 92.40% |
| BF16, 917,504 tokens, yarn:3.5 | 0.05357 | 92.26% |
| FP8, 917,504 tokens, yarn:3 | 0.05312 | 92.37% |
| FP8, 917,504 tokens, yarn:3.5 | 0.05382 | 92.19% |

- **FP8 at its served shape against BF16 at its adapted one** (917,504 and yarn:3.5 against 786,432 and yarn:3):
  1.0328 (window-bootstrap 95% CI 1.0061-1.0611), top-1 -0.21 points.
- **The rope's share:** FP8 at yarn:3 against the same BF16 reads 1.0195 (1.0008-1.0408), -0.02 points, and FP8 at
  yarn:3.5 against FP8 at yarn:3 reads 1.0130. The BF16 reference runs no YaRN over these 4,096-token windows, so a
  wider table costs KLD on its own: BF16 at yarn:3.5 reads 2.8% above BF16 at yarn:3.
- **At one context and table** (both at 917,504 and yarn:3.5): 1.0046 (0.9807-1.0257), -0.08 points.

**The plan.** At the `rtx5090` profile's shape (917,504 tokens, eight lanes, DFlash2, vision) the VRAM plan needs
28,608,687,108 bytes with FP8 and 29,142,942,212 with BF16, one sequence of KV included, 534.3 MB apart.

**What a start kept.** On 2026-11-18 the card had 28,549-28,564 MiB of its 32,607 MiB free at start (nvidia-smi read
3,624-3,639 MiB used). Started with no config file under the profile's 1,200 MiB headroom, BF16 adapted to 786,432
tokens and yarn:3, and FP8 kept 917,504 and yarn:3.5 with 65 MiB to spare. Under a 1,500 MiB headroom on the same
desktop BF16 adapted to 720,896 tokens (yarn:2.75) and FP8 to 786,432 (yarn:3). #675 saw BF16 keep 917,504 on a
desktop holding about 0.5 GB less.

## Finding

Observed:

- FP8 passes the leg at the shapes the RTX 5090 serves: 1.0328 against a limit of 1.05, top-1 -0.21 points against
  0.5, by the point estimate. The interval's upper end, 1.0611, is above the limit.
- At one context and one table FP8 reads 0.5% above BF16, with an interval that holds 1; a wider rope table costs
  KLD on its own, 2.8% on BF16.

Inferred, from the plan's bytes and the profile's headroom:

- Under the 1,200 MiB headroom a start with no config file keeps 917,504 tokens unadapted when the card has at least
  28,483 MiB free at start with FP8, and 28,993 MiB with BF16; with less it adapts. Under a 1,500 MiB headroom both
  thresholds rise by 300 MiB.
- What a default install keeps therefore follows what the desktop holds.

## Implications

- Greedy text changes against a BF16 load: the weights and the prefill's activations are other numbers.
- On the RTX 5090 the FP8 default moves the rope table only where BF16 had adapted its context: from yarn:3 to
  yarn:3.5 at the profile's headroom on a desktop that held BF16 to 786,432, and nowhere when BF16 already kept
  917,504.
- A check that compares two runs of the 27B names its exception weights and pins its context.

## Limits and unknowns

- One session on one card. The RTX PRO 6000 runs the same kernels and takes the same default; its KLD was not
  measured.
- The leg is read by the point estimate, as ADR 0082 states it; the interval of the served-shape ratio crosses 1.05.
- The thresholds are the plan's arithmetic at one shape, with one sequence of KV; another profile shape or headroom
  moves them.

## Follow-ups

- None.
