---
title: "Context that follows the card"
description: "Rope scaling picks itself from the context: none up to 262,144 tokens, YaRN at the exact ratio beyond."
sidebar:
  order: 15
---

How long a context a card can hold depends on how much VRAM is left for the KV
pool, and the 5090 profile runs the 27B at 917,504 tokens (896K) with vision and
DFlash2 on. Both models were trained over 262,144 positions, so a context past
that needs a rescaled rotary table. ignis computes it for you: `auto` leaves
positions alone up to 262,144 and applies YaRN at `max_context / 262,144` beyond,
so changing the context changes the table with it and a mismatch cannot happen
silently.

:::caution[A principled default, not a measured win]
No probe so far shows YaRN improving or hurting quality. The 27B's 320K needle,
and both Flash-Next legs, failed to separate `auto` from `none`. `auto` removes a
way to misconfigure a long context; it is not claimed to improve recall.
:::

## How it works

Before ADR 0057 each profile spelled a matching `yarn:F` per model, and a factor
that did not match the context was silent: a lower one leaves positions
extrapolated, a higher one degrades short contexts for nothing. Now `auto`
resolves per model at load from its effective `model.max_context`. Within 262,144
it is no scaling at all, the original table and not "YaRN at 1". Beyond it, it
is YaRN at the exact ratio with the default temperature and ramp: 2 at 524,288,
3 at 786,432, 3.5 at the 5090's 917,504, 4 at 1,048,576. Past 64 times the
envelope (16,777,216 tokens) the start is refused, naming the context.

Flash-Next applies YaRN's attention factor on queries and keys alike, from one
table that serves its attention, its sparse-attention indexer and its MTP head
([ADR 0059](../../adr/0059-flash-next-applies-yarn-on-q-and-k.md)). `none` and
`yarn:F[,t=..][,bf=..][,bs=..]` override `auto` from any source, a model's own
section included. Because the factor follows the context, a check that compares
two runs of the 27B must pin `max_context`: when free VRAM changes the
context a load can hold, the factor changes with it. On the RTX 5090 that is a
few hundred MiB: under the profile's 1,200 MiB headroom the 27B keeps 917,504
tokens (yarn:3.5) with at least 28,993 MiB free at start under BF16 exception
weights, and from 0.11.0, under the FP8 default, with 28,483 MiB; a desktop
that leaves less adapts it to 786,432 (yarn:3) ([ADR 0103](../../adr/0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md)).

Which contexts fit on which card, and what they cost in KV, is in
[Profiles](../concepts/profiles.md) and [Context and speculation](../configuration/context-and-speculation.md).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| Trained envelope 262,144 positions; factor = `max_context` / 262,144; widest factor 64 | the rule, not a measurement | [ADR 0057](../../adr/0057-rope-scaling-is-auto-by-default.md) |
| The 5090 profile's 27B context 917,504 tokens with vision and DFlash2 on and a KV pool of 944,832 tokens; 950,272 is refused by 59 MB | RTX 5090, the plan raised until it refuses | [Prefill attention without prompt planes](../../findings/2026-10-21-prefill-attention-without-prompt-planes.md) |
| From 0.11.0, 917,504 tokens (yarn:3.5) held unadapted with at least 28,483 MiB free at start under the FP8 exception weights, 28,993 MiB under BF16; with 28,549 MiB free BF16 adapted to 786,432 (yarn:3), and under a 1,500 MiB headroom BF16 to 720,896 (yarn:2.75) and FP8 to 786,432 (yarn:3) | RTX 5090, `rtx5090` profile (1,200 MiB headroom), no config file, 2026-11-18 | [ADR 0103](../../adr/0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md) |
| A needle at 320,076 tokens is recalled both with `yarn:4` and with `none` | 27B, one needle | [YaRN needle at 320K does not discriminate](../../findings/2026-09-19-yarn-needle-320k-not-discriminating.md) |
| Flash-Next at about 512K on the 5090: `auto` 5/5 needles and 1/2 chains, `none` 4/5 and 1/2. At 1M on the RTX PRO 6000 with MTP: `auto` 4/5 and 0/2, `none` 5/5 and 1/2 | one seed, one filler, one question per cell | [Flash-Next YaRN in the reference engines](../../findings/2026-10-19-flash-next-yarn-in-the-reference-engines.md) |

The last two rows are the whole evidence on quality, and neither separates the
tables: the `none` control did not degrade past 262,144 on these probes, so they
cannot say what `auto` is worth. The ADR itself calls the 27B's factor on the
5090 profile unmeasured against the `yarn:4` it replaces, and Flash-Next's
factors unmeasured at long context.

## See it, control it

- **The setting**: [`model.rope_scaling`](../reference/configuration.md#modelrope_scaling)
  (`auto`, `none` or `yarn:F[,...]`; `auto` by default, patchable with a reload),
  and [`model.max_context`](../reference/configuration.md#modelmax_context).
- **At load**: the `ignis.model.loaded` line names the table that ran
  (`rope_scaling`) beside the setting (`rope_scaling_setting`).
- **Live**: `GET /v1/config` shows the setting as configured, and the
  Playground's [Server view](../playground/server.md) shows the running
  configuration.

## Limits

- A literal needle at 320K does not discriminate. A probe of long-context quality
  has to ask about relative position, and no such probe has yet separated the
  tables.
- Context costs KV. The 5090's 1,048,576 tokens for the 27B is 1.07 GB short of
  fitting, which is why its profile stops at 917,504.
- Flash-Next prefill at 1M tokens takes longer than the default request timeout;
  raise it for such a prompt ([Deadlines](../api/deadlines.md)).

## Read more

- [Context and speculation](../configuration/context-and-speculation.md): the flag and its neighbours.
- [Profiles](../concepts/profiles.md): the context each card class runs.
- [Every byte planned at load](every-byte-planned.md): why 917,504 and not 1,048,576.
