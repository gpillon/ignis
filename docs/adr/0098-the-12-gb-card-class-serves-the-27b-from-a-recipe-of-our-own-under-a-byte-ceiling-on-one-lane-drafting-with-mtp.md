---
status: proposed
amends: [ADR-0063, ADR-0064, ADR-0079]
---

# The 12 GB card class serves the 27B from a recipe of our own under a byte ceiling, on one lane, drafting with MTP

A 12 GB card was the 16 GB class's (ADR 0063: every total below 20 GiB), and the 16 GB entry does not fit it: one
lane of that artifact holds 12,218,298,952 bytes before its KV pool, and `rtx-16gb`'s headroom alone is 1.26 GB.
The 27B stays the model: the family has no smaller sibling (checked 2026-11-15), and on the one public harness that
runs agent tasks below 3 bits per weight the loss grows with the task's length and steps down between 8.0 and 9.2 GB
of weights, while single-turn code barely moves
([the online research](../findings/2026-11-15-the-12-gb-card-class-online-research.md)). The owner's grill of
2026-11-15 (spec GitHub #630) decided:

- **A card class of its own.** A card whose NVML total is below 14 GiB is the 12 GB class (a 12 GB card reads about
  11.9 GiB), with its profile, `rtx-12gb`, which `--profile auto` picks there, and its catalog entry,
  `qwen3.8-27b-12g` (`card_class: 12`), which answers to `qwen3.8-27b` there. The 16 GB class becomes 14 GiB up to
  20 GiB; the larger classes do not move. The class is memory alone, as every class: an RTX 5070, a 12 GB RTX 4070
  family card, an RTX 3060 12 GB or an RTX 3080 Ti are one class, and the GPU generation decides routes. It has no
  lower bound: a card of 10 or 11 GB is read as the class and refused by its plan, which names the bytes it misses.
- **A recipe of our own, bound by bytes.** The body is trellis at integer K 2 to 5 with a 4-bit head, the classes
  the load binds today, under a ceiling of 9.2 GB of text-scope device weights; its bits per weight, about 2.8, are
  what that leaves. No public quant between 2.2 and 3.0 bits binds with those classes. The artifact keeps the DFlash2
  module and the 8-bit MTP head.
- **MTP is this class's default drafter**
  ([ADR 0095](0095-the-27b-has-two-drafters-and-its-mtp-head-keeps-attention-state-over-the-whole-context-and-a-continuation-hidden.md),
  at the window #628 measured, 7), where every other class drafts with DFlash2: DFlash2 holds about 0.7 GB more,
  which on this card is the context. Measured (#635, candidate A in 12 GiB), DFlash2 also keeps 131,072 tokens there:
  its plan's lines are 707,931,720 bytes more than the head's and its KV pool holds 150,720 tokens against the head's
  214,144. In that budget the head's edge is pool; on a real card the pool is the context: on the plan's arithmetic,
  from the DFlash2 load's logged lines, DFlash2 keeps 49,152 tokens on an RTX 50 card with nothing else on it, where
  the head keeps 131,072, and beside a desktop of about 0.75 GB its plan holds no context of the ladder.
- **The profile is sized for a desktop on the card, one user**: one decode lane, a context of 131,072 that adapts in
  steps of 16,384 (`model.adapt_context_step`; 65,536 elsewhere), `vram.qwen38.headroom_bytes` 512M (1200M
  elsewhere), a 512-token prefill chunk with the solo width at the chunk, vision on in host memory at 4,096 tokens.
  The step is the owner's call of 2026-11-16, after the gate: a real card's budget is derived from the memory its
  desktop leaves, the context is the largest of the ladder's that budget holds, and at 32,768 the real-card cases
  wasted up to a step of it, which 16,384 recovers at no cost.
  Every value is a default the load adapts and the operator overrides; two lanes, a 256-token chunk and DFlash2 stay
  configuration choices. The chunk's price is measured (#636, candidate A in 12 GiB): 512 buys 4,608 tokens of KV
  pool over 1,024, not the estimated 17,000, for 7% on the first token, and 256 buys none with vision on, 25%
  slower at 8,192 tokens; the context is 131,072 at all three.
- **The gate's three legs are measured and reported, and the owner admits the artifact on the numbers** (ADR 0082: a
  failing leg stops the work and the owner decides). The limit a re-quantization of the 16 GB body has registered is
  0.0484; the 16 GB body reads 0.0403 and the estimate for this recipe is about 0.056. Measured on 2026-11-16
  (#635), against the 16 GB body's 0.04177 on that session's engine and shape, whose limit is 0.0501: the optimizer's
  recipe (candidate A) reads 0.04308, 1.03 times, and the GGUF allocation's (B) 0.05084, 1.22 times. The estimate of
  0.056 was high ([the finding](../findings/2026-11-15-the-12-gb-card-class-online-research.md)). The owner admitted
  candidate A on 2026-11-16 as `trellis-ig28`, the ceiling staying 9.2 GB and the chunk 512.

## Considered Options

- **A smaller quant to pay for a second lane.** A lane costs about 0.87 GB, 0.28 bits per weight of the body, and
  lands the recipe where long agent tasks give way.
- **Kernels for another weight format.** On that harness a 9.2 GB file of the other format and the 9.85 GB trellis
  score the same within noise: the per-tensor allocation is what pays, and it translates to K.
- **Flash-Next on the 12 GB card.** More per byte of VRAM on agent suites, at 48 to 64 GB of host memory and under a
  licence that restricts commercial serving. A later phase.
- **Keeping the class inside the 16 GB class and adapting.** The class decides the artifact (ADR 0064), and no
  adaptation changes an artifact.
- **The 65,536-token step of the other classes, or 32,768.** The spec took 32,768, so that a desktop leaving room for
  about 100,000 tokens kept 98,304 and not 65,536. On a real card's derived budget either throws away up to a step
  of the context the free memory holds: beside a desktop of about 0.5 GB 32,768 keeps 65,536 where 16,384 keeps
  81,920, and beside about 0.3 GB 98,304 where 16,384 keeps 114,688; the finer step costs nothing.
- **The 1200M headroom of the other classes.** With it no context is left: 512M is a choice, not a placeholder.

## Consequences

- The headroom and the context are the RTX 5090's simulation, under an explicit 12 GiB budget and that less 1 GiB
  (131,072 and 98,304 tokens, measured by the plan on candidate A), until a 12 GB card reads its CUDA
  context, its residual at one lane and its speed.
- A real 12 GB card's context is the plan's arithmetic on those lines until one confirms it (#639). Its budget is
  what NVML reads free before the start less 512 MiB: 131,072 tokens need 12,027 MiB free on Blackwell's routes and
  12,112 MiB on an RTX 30 or 40 card's, whose dense prefill buffer is 89 MB more, and each 16,384-token step below
  about 153 MiB less. With nothing else on a card of about 11.9 GiB (about 11.75 GiB free) that is 131,072 tokens
  on Blackwell's routes and 114,688 on the 8.x routes; beside a desktop of about 0.3 GB 114,688 and 98,304; about
  0.5 GB 81,920 on both; about 0.75 GB 65,536 and 49,152; about 1 GB 32,768 on both; about 1.3 GB the start is
  refused, 10 MiB short of 16,384. The spec's 98,304 beside a desktop's gigabyte was the RTX 5090's 12 GiB less
  1 GiB, an explicit budget with no headroom; a real card's derived budget leaves that desktop 32,768. The guide
  states the table (`configuration/vram-and-kv-pool.md`) and what an operator can change: the desktop,
  `--vram-budget-bytes` (at most the 512 MiB of headroom, three steps, since a budget above free memory is
  refused), and the second lane and DFlash2 as costs.
- The class's catalog entry, `qwen3.8-27b-12g` (#638), pins `trellis-ig28` at
  `gpillon/Qwen3.8-27B-trellis-2.8bpw-ignis`, revision `15333a0829f055e7ee25d6d5ef84fd8d521518c6`, public under
  Apache 2.0: the artifact and its conversion record, 12,754,973,515 bytes. On a 12 GB card it is the one built-in
  text or image model that loads; the uncensored variant and Swift 1.5 have no build of the class, and they and
  every 16 GB entry are refused there by name before a fetch or a teardown. An artifact named by its path still
  loads, judged by the plan alone.
- Two entries now answer to `qwen3.8-27b`, the 16 GB 27B and this one, so ADR 0064's one claimant of an alias
  becomes one claimant of each card class within a layer, and where several entries that answer to an id fit the
  card, the one of the largest class answers, never the catalog's order: the 16 GB entry on a 16 or 24 GB card, an
  RTX 3090 or 4090 and a card of no class, this one on a 12 GB card only. It loads by its own id on every class. An
  operator entry still claims no alias a built-in entry claims, whatever its class; two of its own entries of
  different classes may answer to one id.
- The entry states no generation, as the 16 GB 27B's, where ADR 0079 has a trellis build state `blackwell` until it
  is verified below Blackwell: its layout is that build's (the trellis K classes 2 to 5 and the 4-bit head the RTX
  30 and 40 routes bind, every other object the 32 GB artifact's), and the class is memory alone. Its quality on
  those routes is measured as Swift 1.5's build's was (#529, #513): on 2026-11-17 the RTX 5090, running them
  (`IGNIS_MODEL_PREFILL_LINEAR=int8`, `IGNIS_MODEL_PREFILL_ATTENTION=int8`, the 512-token chunk), read a KLD
  against BF16 of 0.04353, 1.0103 times the same artifact's on Blackwell's routes (95% interval 0.974 to 1.035;
  Swift's, 1.0148), under ADR 0082's 1.05 for a route's numerics, with top-1 93.23% against 93.35%
  ([the finding](../findings/2026-11-15-the-12-gb-card-class-online-research.md)). So the entry stands on both
  grounds ADR 0079 admitted Swift 1.5's build on, its layout, read as above, and its quality on those routes; an
  sm_86 gate leg, or a 12 GB RTX 30 or 40 card (#639), reads it on such a card.
- A limit above 0.0484 for this class, if the owner admits one, is registered in the spec that admits it.
