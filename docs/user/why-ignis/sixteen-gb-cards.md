---
title: "The 27B on a 16 GB card"
description: "A 3-bit trellis body, a profile chosen by the card's memory, and four quality gates passed: the 27B on an RTX 5060 Ti 16 GB, 5070 Ti or 5080."
sidebar:
  order: 4.5
---

The 27B's served NVFP4 artifact is 18 GiB on disk, and its text weights alone
are 16 GiB: it cannot load on a 16 GB card at any context. Since 0.8.0 ignis ships a second encoding of the same
model for that class of card, `qwen3.8-27b-16g`: the body re-encoded in a trellis code at
about 3 bits per weight, **12.5 GiB on disk**, with the same DFlash2 drafter.
On a 16 GB card the server picks the matching profile and the matching
catalog entry by itself, and the artifact passed **all four quality gates**
against the NVFP4 it stands in for.

:::caution[Not yet measured on a 16 GB card]
The project has no 16 GB card. Every figure below was measured on an RTX 5090
held to a 16 GiB budget; the speeds for the 5060 Ti 16 GB, 5070 Ti and 5080 are
an estimate from those measurements. If you have one of these cards,
[one command measures it](../guides/measure-a-16gb-card.md), and its report is
how the profile gets real numbers.
:::

## How it works

**A body that fits** ([ADR 0060](../../adr/0060-the-27b-has-a-trellis-body-for-small-cards.md)).
The 16 GB entry's linear layers are a trellis code at about 3 bits
per weight (a repack of a public 3.0 bpw EXL3 quantization), the code ignis
already uses for Flash-Next's experts, read by dense kernels of its own at
55-65% of the card's memory bandwidth. The body runs with 16-bit activations
where the NVFP4 runs 4-bit ones, which is why it ends up closer to BF16 than
the artifact it stands in for. The DFlash2 drafter is the same.

**A profile chosen by the card** ([ADR 0063](../../adr/0063-the-profile-follows-the-card-class.md)).
With `--profile auto`, the default, the server reads the card's memory and
picks `rtx50-16gb` under 20 GiB: 262,144 tokens of context, two decode lanes,
DFlash2 with 7 drafts, vision off (the tower and its encoder peak are VRAM the
card does not have; `--vision-enabled` turns it on at 1,024 tokens an image). The VRAM budget is the card's free memory less the
profile's 1,200 MiB of headroom, so whatever the desktop holds comes out of it.
When the plan does not fit that budget, the load lowers the context step by step
instead of refusing at once, and `GET /v1/models` names what it adapted
([when a profile does not fit](../concepts/profiles.md#when-a-profile-does-not-fit)).

**A catalog entry that knows its class** ([ADR 0064](../../adr/0064-a-catalog-entry-states-its-card-class.md)).
Each catalog entry states the smallest card class it loads on. On a 16 GB
card, `qwen3.8-27b` resolves to the 16 GB entry, and a model too large for the
card is refused before anything is torn down.

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| KLD from BF16 **0.0403, 0.77x the NVFP4's** at chunk 1,024 (0.82x at 2,048; gate 1.2x) | the 2026-09-24 KLD suite, through ignis's engine path | [Quality gates](../../findings/2026-10-24-27b-16gb-quality-gates.md) |
| MMLU-Pro **65.84%** against BF16's 67.97%, exact McNemar p 0.31 | through the engine's logits | same |
| DFlash2 commits **0.996x** the NVFP4's tokens a round (3.370 against 3.383, 95% interval 0.993-0.999; floor 0.90) | 200 recorded agent turns, teacher-forced | same |
| Reasoning loops **9 of 32** against the NVFP4's 14 of 32 (gate: not significantly more at one-sided Fisher p < 0.05; p 0.94) | the loop study's four prompts, eight seeds each | same |
| A 16 GiB and a 15 GiB budget: **262,144 tokens on two lanes**, nothing adapted; a 13 GiB budget adapts to 65,536 | RTX 5090, the budget named by `--vram-budget-bytes` | [Capacity on the 5090](../../findings/2026-10-24-27b-16gb-capacity-on-the-5090.md) |
| **416 / 216 tok/s** at one lane (8K / 253K of context), 684 / 275 across two | RTX 5090 held to 16 GiB, uncapped | [Speed estimate](../../findings/2026-10-24-27b-16gb-speed-estimate.md) |
| Estimated, one lane at 8K: 5080 **243-307**, 5070 Ti **204-270**, 5060 Ti 16 GB **103-153** tok/s | an estimate, not a measurement | same |

The gates were registered before the run. One figure sits outside them: the
thinking budget forced a close on 15 of 32 turns against 8 of 32 on NVFP4.

## See it, control it

- **The model**: `ignis-server model download qwen3.8-27b-16g`
  ([the catalog](../reference/models.md#qwen38-27b-16g)).
- **The profile**: `--profile auto` (the default) or `--profile rtx50-16gb`
  ([profiles](../concepts/profiles.md#rtx50-16gb)).
- **The adaptation**: [`model.adapt_context_step`](../reference/configuration.md#modeladapt_context_step)
  and [`vram.adapt_spare_tokens`](../reference/configuration.md#vramadapt_spare_tokens);
  `GET /v1/models` reports `decode_lanes` and `adapted`.
- **A report of your card**: `ignis-bench small-card`
  ([measure a 16 GB card](../guides/measure-a-16gb-card.md)).
- **Playground**: the [Server view](../playground/server.md) lists the 16 GB
  entry and downloads it.

## Limits

- Two lanes, not eight, and no image input unless you turn it on, then at
  1,024 vision tokens an image.
- Prefill is slower than on NVFP4: the trellis kernels' prefill reaches about
  half the card's tensor peak.
- The speeds are an estimate until someone measures a real card.
- The capacity figures are budgets named by hand on a 5090. On a real card
  the budget is what is free less the 1,200 MiB headroom, so the context a load
  keeps there depends on what else holds the card's memory.

## Read more

- [Profiles](../concepts/profiles.md): what each card class runs, and why.
- [Models and artifacts](../concepts/models-and-artifacts.md): the families and their encodings.
- [A 125B MoE on a 32 GB card](flash-next.md): the same trellis code on Flash-Next's experts.
