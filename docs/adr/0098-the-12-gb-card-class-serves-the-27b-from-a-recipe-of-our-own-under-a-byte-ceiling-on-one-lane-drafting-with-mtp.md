---
status: accepted
---

# The 12 GB card class serves the 27B from a recipe of our own under a byte ceiling, on one lane, drafting with MTP

A 12 GB card is a card class of its own, with its own profile and its own catalog entry of the 27B. It was the
16 GB class's, whose entry does not fit it: one lane of that artifact holds about 12.2 GB before its KV pool, and
that profile's headroom alone is over a gigabyte. The 27B stays the model: the family has no smaller sibling, and on
the one public harness that runs agent tasks below 3 bits per weight the loss grows with the task's length and steps
down between 8.0 and 9.2 GB of weights, while single-turn code barely moves
([the online research](../findings/2026-11-15-the-12-gb-card-class-online-research.md)).

- **A card class of its own.** A card below the 16 GB class is the 12 GB class, and `--profile auto` picks its
  profile there (ADR 0063); the larger classes do not move. The class is memory alone, as every class: an RTX 5070,
  a 12 GB RTX 4070 family card, an RTX 3060 12 GB or an RTX 3080 Ti are one class, and the GPU generation decides
  routes. It has no lower bound: a card of 10 or 11 GB is read as the class and refused by its plan, which names the
  bytes it misses.
- **A recipe of our own, bound by bytes.** The body is trellis at integer K 2 to 5 with a 4-bit head, the classes
  the load binds today, under a ceiling of 9.2 GB of text-scope device weights; its bits per weight, about 2.8, are
  what that leaves. No public quant between 2.2 and 3.0 bits binds with those classes. The artifact keeps the
  DFlash2 module and the 8-bit MTP head.
- **MTP is this class's default drafter** (ADR 0095), where every other class drafts with DFlash2: DFlash2 holds
  about 0.7 GB more, which on this card is the context.
- **The profile is sized for a desktop on the card, one user**: one decode lane, a context that adapts in finer
  steps than the other classes', less headroom than theirs, a narrow prefill chunk with the solo width at the
  chunk, vision on in host memory. The finer step is the owner's call: a real card's budget is derived from the
  memory its desktop leaves and the context is the largest of the ladder's that budget holds, so a coarser step
  throws away up to a step of it, which the finer one recovers at no cost. The headroom is a choice, not a
  placeholder: with the other classes' no context is left. Every value is a default the load adapts and the
  operator overrides (ADR 0063); two lanes, a narrower chunk and DFlash2 stay configuration choices.
- **The gate's three legs are measured and reported, and the owner admits the artifact on the numbers** (ADR 0082:
  a failing leg stops the work and the owner decides), as the owner admitted the optimizer's recipe,
  `trellis-ig28`, the ceiling staying 9.2 GB. The KLD limit is the one a re-quantization of the 16 GB body has
  registered there; a limit above it for this class, if the owner admits one, is registered in the spec that admits
  it.
- **The entry answers to the 27B's default id on a 12 GB card only**, and loads by its own id on every class (ADR
  0064 holds the rule for several entries that answer to one id). On a 12 GB card it and the uncensored variant's
  build of the class are the built-in text or image models that load: Swift 1.5 has no build of the class, and it
  and every 16 GB entry are refused there by name before a fetch or a teardown. An artifact named by its path still
  loads, judged by the plan alone.
- **The entry states no generation**, as the 16 GB 27B's, where ADR 0079 has a trellis build state `blackwell`
  until it is verified below Blackwell. It stands on both grounds ADR 0079 admitted Swift 1.5's build on: its
  layout is the 16 GB build's, and its quality on the RTX 30 and 40 routes, read on the RTX 5090 running them, is
  within ADR 0082's limit for a route's numerics. A gate leg on such a card, or a 12 GB RTX 30 or 40 card, reads
  it there.

## Considered Options

- **A smaller quant to pay for a second lane.** A lane costs about 0.87 GB, 0.28 bits per weight of the body, and
  lands the recipe where long agent tasks give way.
- **Kernels for another weight format.** On that harness a 9.2 GB file of the other format and the 9.85 GB trellis
  score the same within noise: the per-tensor allocation is what pays, and it translates to K.
- **Flash-Next on the 12 GB card.** More per byte of VRAM on agent suites, at 48 to 64 GB of host memory and under a
  licence that restricts commercial serving. A later phase.
- **Keeping the class inside the 16 GB class and adapting.** The class decides the artifact (ADR 0064), and no
  adaptation changes an artifact.
- **The context step of the other classes, or half of it.** On a real card's derived budget either throws away up
  to a step of the context the free memory holds; the finer step costs nothing.

## Consequences

- The profile's headroom and context come from the RTX 5090 held to the class's budget. What a real 12 GB card keeps
  is the plan's arithmetic on those lines until cards confirm it; the guide states it beside each desktop, with what
  an operator can change (`docs/user/configuration/vram-and-kv-pool.md`).
