---
status: accepted
amends: [ADR-0065]
---

# On compute capability 12.0 the 27B holds its exception tensors in FP8 by default

[ADR 0065](0065-the-bf16-exception-tensors-can-be-held-in-fp8.md) let the 27B hold its nine BF16 exception tensors in
FP8 row-scale, converted at load, and kept `bf16` the default. FP8 freed 534.6 MB for the KV pool and made cold prefill
2.8-4.1% faster at 14K-105K at the same 4K KLD, but two pre-registered decision fixtures read differently under it, and
the scalar balance's lost sign, lost with BF16 activations too, was read as the FP8 weights' own error. The
measurement of 2026-11-17 (#667) replaced that reason. Over 17 deterministic routes (the rope table, the GDN prefill,
the prompt attention band, prompt reuse) the balance's sign is a coin in BF16 too, which reads it right on 2 of 9
routes and wrong at its served shape. On 116 synthetic entailment items FP8 is as right as BF16, with a margin shift
of +0.16 [+0.04, +0.27], and FP8 moves the entailment set about as far as one harmless switch of route does. #675
re-registered both sets on arms that name their exception weights, context and rope before the default moved
([the finding](../findings/2026-11-18-the-decide-regression-sets-under-both-exception-weights.md)). The owner decided
the default on 2026-11-17, with yarn:3.5 on the RTX 5090.

- **`model.exception_weights` is `auto` by default.** `auto` is `fp8` on a card that runs the block-scaled FP8 MMA,
  compute capability 12.0 (the RTX 5090 and the RTX PRO 6000), and `bf16` on any other card and when no card is read.
  A named `bf16` is kept on every card. A named `fp8` is refused below 12.0, naming the card, as before. The 16 GB and
  12 GB RTX 50 cards resolve `fp8` too, but their trellis artifacts hold no BF16 exception tensor, so nothing converts.
- **The default lives in config resolution**, beside the card reading ([ADR 0063](0063-the-profile-follows-the-card-class.md),
  [ADR 0046](0046-one-declaration-per-config-field.md)). `auto` resolves into the load's weights, while `GET
  /v1/config` and a file `config generate` writes keep `auto`. The engine's own default stays `bf16`: a test or a tool
  that binds the model directly keeps the stored numerics unless it names FP8.
- **The load says what it holds**: `ignis.runtime.exception_weights` names the weights, the setting and where the
  setting came from (`exception_weights auto -> fp8 (default)`).

**The gate** is [ADR 0082](0082-the-gate-is-kld-against-bf16-dflash2-on-against-off-and-the-loop-rate.md)'s KLD leg for
a route made a default: the KLD of FP8 at most 1.05x BF16's, top-1 within 0.5 points, by the point estimate. On the
RTX 5090 (2026-11-18), set A was dumped four times from one binary in one session: chunk 2,048, hq-e8-2b, each model
loaded at the named context with its table, and each dump scored against the BF16 checkpoint over the full vocabulary.

| dump | KLD | top-1 |
|---|---:|---:|
| BF16, 786,432 tokens, yarn:3 | 0.05211 | 92.40% |
| BF16, 917,504 tokens, yarn:3.5 | 0.05357 | 92.26% |
| FP8, 917,504 tokens, yarn:3 | 0.05312 | 92.37% |
| FP8, 917,504 tokens, yarn:3.5 | 0.05382 | 92.19% |

- **FP8 at its served shape against BF16 at its adapted one** (917,504 and yarn:3.5 against 786,432 and yarn:3): 1.0328
  (window-bootstrap 95% CI 1.0061-1.0611), top-1 -0.21 points. It passes.
- **The rope's share:** FP8 at yarn:3 against the same BF16 reads 1.0195 (1.0008-1.0408), -0.02 points, and FP8 at
  yarn:3.5 against FP8 at yarn:3 reads 1.0130. The BF16 reference runs no YaRN over these 4,096-token windows, so a
  wider table costs KLD on its own: BF16 at yarn:3.5 reads 2.8% above BF16 at yarn:3.
- **At one context and table** (both at 917,504 and yarn:3.5): 1.0046 (0.9807-1.0257), -0.08 points.

**The context the default leaves on the RTX 5090.** At the `rtx5090` profile's shape (917,504 tokens, eight lanes,
DFlash2, vision) the VRAM plan needs 28,608,687,108 bytes with FP8 and 29,142,942,212 with BF16, one sequence of KV
included, 534.3 MB apart. Under the profile's 1,200 MiB headroom a start with no config file keeps 917,504 tokens
unadapted when the card has at least 28,483 MiB free at start with FP8, and 28,993 MiB with BF16; with less it adapts.
What a default install keeps therefore follows what the desktop holds. On 2026-11-18 the card had 28,549-28,564 MiB of
its 32,607 MiB free at start (nvidia-smi read 3,624-3,639 MiB used): BF16 adapted to 786,432 tokens and yarn:3, and FP8
kept 917,504 and yarn:3.5 with 65 MiB to spare. #675 saw BF16 keep 917,504 on a desktop holding about 0.5 GB less.
Under a 1,500 MiB headroom both thresholds rise by 300 MiB, and on the same desktop BF16 adapted to 720,896 tokens
(yarn:2.75) and FP8 to 786,432 (yarn:3).

## Considered options

- **FP8 opt-in, as ADR 0065 left it.** It leaves 534.6 MB of the RTX 5090's KV pool and 3% of cold prefill unused for
  a decision read the measurement no longer supports.
- **FP8 by default on every card that loads the NVFP4 artifact.** That is compute capability 12.0 alone, so it is this
  decision. On an RTX 30 or 40 card, the W8A8 GEMM the FP8 prefill needs does not run.

## Consequences

- Greedy text changes against a BF16 load, as ADR 0065 says: the weights and the prefill's activations are other
  numbers. On the RTX 5090 the rope table moves only where BF16 had adapted its context: from yarn:3 to yarn:3.5 at the
  profile's headroom on a desktop that held BF16 to 786,432, and nowhere when BF16 already kept 917,504. The release
  notes say both.
- A config file `config generate` wrote before 0.11.0 names `bf16`, which is kept: such a file holds the stored
  weights until it says `auto`.
- `config::resolve` and `resolve_with` inject an RTX 5090 reading, so `auto` is `fp8` in every test that resolves a
  config; a test that holds a recorded number or text through config resolution names its weights.
- The leg ran on the RTX 5090. The RTX PRO 6000 runs the same kernels and takes the same default, its KLD not measured.
