---
title: "The VRAM plan"
description: "How a load decides every byte of device memory before the first request, why a start is refused instead of running out of memory later, and what Windows does differently."
sidebar:
  order: 4
---

When the server loads a model it does not allocate as it goes. It first works
out, from the card's free memory, how every byte of VRAM will be used, prints
that plan, and reserves it. Nothing is allocated on the device while requests
are served. This page explains the plan, what it counts, and why a start is
refused rather than started and left to fail
([ADR 0030](../../adr/0030-device-memory-reserved-at-load.md)).

## The budget

The plan starts from a **VRAM budget**: the device memory the whole ignis
process may hold, weights included, which is the figure Task Manager shows for
the process. It is either named (`--vram-budget-bytes`) or derived:

```text
budget = device memory free at start  -  headroom
```

The **headroom** (`--vram-headroom-bytes`, 1,536 MiB by default) is what the
plan leaves to everything else on the card: the desktop, a browser, other
programs. Because the budget is derived from free memory, whatever the desktop
already holds is counted out before the plan begins, and the same flags can give
a smaller plan on a busier machine.

## What the plan counts

The plan carves the budget into lines, in a fixed order, and prints them as the
`ignis.runtime.vram_plan` event (`ignis.runtime.flash_next_plan` for
Flash-Next). The lines include:

- the **weights** (the 27B's vision tower among them only under
  `--vision-residency device`);
- **workspaces**: the prefill scratch, sized for `--model-prefill-chunk`; with
  vision on, the image encoder's workspace shares it and the line is the larger
  of the two (under `--vision-residency host` the encoder's share includes the
  slots its weights are copied through,
  [Vision](../configuration/vision.md#where-the-vision-tower-and-the-embedding-pool-live));
- the **lane state** of every decode lane (on the 27B, about 144 MiB per lane
  for the linear-attention layers);
- the **decode and drafter graphs**' staging buffers;
- the **retained slots** that live on the device (none by default) and, under
  the compressed KV format, the residual window of every lane and slot;
- the **media embedding** pool for images (0 under `--vision-residency host`,
  which keeps it in pinned host RAM);
- the **KV pool**, which takes whatever is left on the 27B, or is reserved first
  on a Flash-Next that streams its experts.

On Flash-Next a few more lines appear: the **expert cache** (what the KV pool
leaves, refused below 12 GiB), the **prefill staging** ring and the residency
tables. The pool policy, resident or offloaded, is chosen by whether every
weight fits on the device ([the KV cache](kv-cache-and-prompt-reuse.md),
[VRAM budget and the KV pool](../configuration/vram-and-kv-pool.md)).

Because the KV pool is the remainder, it is the line that moves when something
else changes. Raising `--model-max-context`, turning vision on, adding lanes or
a draft head, or a busier desktop all come out of the pool, and the pool's size
in pages and tokens is printed in the same event. `make config` prints the plan
a start will use before it runs.

On the RTX 5090 profile the 27B's 917,504-token context fits with vision and
DFlash2 on, and 950,272 is refused by 59 MB
([finding](../../findings/2026-10-21-prefill-attention-without-prompt-planes.md)).
That is the plan working as intended: the context is sized to the card, not
guessed.

## Why refuse instead of run

A budget that cannot hold the minimum refuses the start. The minimum is the
weights, the workspaces, every lane, every device retained slot, and one
sequence at the maximum context with a KV page per retained slot beside it. The
refusal names the shortfall and the flags that lower it (a smaller
`--model-max-context`, `--model-prefill-chunk`, `--vram-kv-pool-bytes` or
`--vram-headroom-bytes`).

The alternative is to start and fail later, and the failure would be at the
worst time. Before the plan existed, a default start serving four parallel agent
conversations grew the process by 2,734 MiB in 14 minutes, because requests
allocated device memory as they went and each prefix publish allocated its own
228 MiB image. After it, the same kind of run grew the process by 6 MiB in
24.5 minutes, and the printed plan matched the OS's figure to within 16 MiB
(README, measured on a 5090). A refused start costs a message at load; an
allocation that fails or pages at minute forty costs a request.

The same rule applies to host memory. A text model also plans its pinned host
RAM and refuses the load, in GiB, with how far short it fell and the line to
shrink, before anything is pinned
([host RAM](../configuration/host-ram.md)).

## Windows and WDDM

On Windows, the display driver model (WDDM) does not fail an allocation when
the card is full. It pages device allocations out to system RAM instead, which
Task Manager shows as "Shared GPU memory". The process keeps running, and every
step it takes slows, erratically. `cudaMemGetInfo` cannot see it. Other systems
simply fail the allocation.

Two things follow:

- **The headroom is what keeps the model on the card.** The desktop's own GPU
  clients (a browser drawing a page, for example) allocate as they run. In a
  measurement on a 5090 driving a Windows desktop, a 1 GiB headroom let a
  browser push WDDM into paging the model out: Flash-Next decoded at
  4.6 tokens per second and the 27B at about 40. At 1.5 GiB neither model moved,
  which is why the default is 1,536 MiB
  ([finding](../../findings/2026-10-09-vram-headroom-wddm-paging.md)). The
  27B's RTX 5090 profile lowers its own headroom to 1,200 MiB to fit vision and
  its context, a trade that profile measured
  ([profiles](profiles.md)).
- **Oversubscription is refused unless asked for.** A plan that needs more than
  is free is a refusal. `--vram-allow-oversubscription` lets a start proceed
  with an explicit `--vram-budget-bytes` above free memory, with a warning; on
  Windows that means paging, and the slowdown is the price.

A nonzero "Shared GPU memory" is not always paging. The KV-RAM arena
(`--reuse-kv-host-pool-bytes`) is pinned host memory held whole for the life of
the load, and it is the figure Windows reports as shared GPU memory
([context, KV and speculation](../configuration/context-and-speculation.md)).

## Seeing the plan

- The `ignis.runtime.vram_plan` log event, with the pool policy
  (`kv_pool_policy`), `kv_pool_pages` and `kv_pool_tokens`.
- The Playground's Monitor shows the plan beside what is in use
  ([the Playground](../playground/index.md)).
- Prometheus gauges carry the plan lines, the KV pool and KV-RAM; they read zero
  while no model is loaded ([telemetry](../guides/telemetry.md)).
