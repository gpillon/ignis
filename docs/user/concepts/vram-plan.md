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
- **workspaces**: the prefill scratch, sized for the wider of
  `--model-prefill-chunk` and
  [the solo width](../configuration/context-and-speculation.md#the-solo-width) and
  for the prefill routes the load runs (the dense prefill and ignis's gated
  delta net prefill hold less of it than the fused prefill and the vendored
  kernels); with vision on, the image encoder's workspace shares it and the line is the larger
  of the two (under `--vision-residency host` the encoder's share includes the
  slots its weights are copied through,
  [Vision](../configuration/vision.md#where-the-vision-tower-and-the-embedding-pool-live));
- on the 27B's 16 GB trellis entries, the **trellis workspace** its matrices
  share and, when the load runs the dense prefill (`--model-prefill-linear
  int8`, the default on an RTX 30 or RTX 40 card), the **dense prefill buffer** (`trellis_dense`): the largest matrix
  decoded to INT8, 89 MB, which every matrix of a prefill chunk is decoded into
  in turn, so it costs the KV pool about that much and no more;
- from 0.11.0, on Windows, on the 27B's NVFP4 entries on an RTX 50 or RTX PRO
  card, the **TMA descriptor slots** (`tma_descriptors`): 512 bytes for each of a prompt
  chunk's matrix products (247 on the 27B) at every chunk width from 1,024
  tokens up to the wider of `--model-prefill-chunk` and the solo width, in
  steps of 256: 0.6 MB at a 2,048-token chunk, 3.7 MB at 8,192. They keep each
  product's tensor maps on the device instead of allocating and copying them
  before every product, as 0.10.0 does, so the first request after a load
  allocates nothing a later one does not
  ([ADR 0101](../../adr/0101-on-windows-the-vram-plan-reserves-the-w4a4-tma-products-descriptor-slots-as-a-line-of-their-own.md)).
  0 on Linux, which passes the maps by value, on every
  other entry and card, and below a 1,024-token chunk;
- the **lane state** of every decode lane (on the 27B, about 144 MiB per lane
  for the linear-attention layers);
- the **decode and drafter graphs**' staging buffers;
- from 0.11.0, on a 27B DFlash2 load of the NVFP4 artifact at 15 drafts that
  asks for mixed rounds (rounds whose lanes take both
  [round shapes](speculative-decoding.md#past-seven-drafts-two-round-shapes);
  for now only the measurement switch `IGNIS_MIXED_ROUNDS=1` asks,
  [Testing](../contributing/testing.md)), the **mixed rounds' graphs**
  (`mixed_graphs`): a verify graph and a drafter graph for each mix of extended
  and base lanes the load's lanes and its pool hold (a mix of `w` extended
  lanes beside `n` base lanes takes `2w + n` of the pool's slots, the lanes'
  and the device retained slots'), 5 MiB a mix and 2 MiB more, in 2 MiB pages,
  as the RTX 5090 measured them: all twelve at 8 lanes, 62 MiB, at most 111
  KV pages. 0 on every other load. The driver keeps their pages once the load
  is gone, so a reload adds them back as it adds back the CUDA context;
- the **drafter**, when one is loaded: DFlash2's weights, its window in every
  lane and retained slot (40 MiB each) and its round, or the 27B's MTP head's
  weights (0.45 GB), one call over the widest round in the decode graph's line,
  one call of a prefill chunk's in the workspace, and about 10 KB a lane and a
  retained slot (under `hq-e8-2b`, the head's share of the residual window
  too). Under the MTP head every KV page holds one more attention layer, so the
  pool's bytes buy about 6% fewer tokens
  ([the 27B's two drafters](speculative-decoding.md#the-27bs-two-drafters));
- the **retained slots** that live on the device (none by default, two for
  the 27B under `rtx-24gb`, with prompt reuse on; the event's `retained_device_slots` counts them
  and `retained_slots_bytes` is their line) and, under
  the compressed KV format, the residual window of every lane and slot;
- the **media embedding** pool for images (0 under `--vision-residency host`,
  which keeps it in pinned host RAM);
- the **CUDA context**, from 0.11.0 measured on the card when the first load
  creates it ([Every byte planned at load](../why-ignis/every-byte-planned.md#how-it-works));
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
([finding](../../findings/2026-10-21-prefill-attention-without-prompt-planes.md)),
when the card has enough free at start. From 0.11.0 the 27B holds its exception
tensors in FP8 on the RTX 50 and RTX PRO cards
([`model.exception_weights`](../reference/configuration.md#modelexception_weights)), and its plan at
917,504 tokens needs 534.3 MB less than with BF16: under the profile's
1,200 MiB headroom the context fits unadapted with at least 28,483 MiB free at
start, where BF16 needs 28,993 MiB. On 2026-11-18, with 28,549 MiB free at
start, the FP8 load kept 917,504 tokens
and a BF16 one adapted to 786,432 ([ADR 0103](../../adr/0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md)).
That is the plan working as intended: the context is sized to the card, not
guessed.

### What the 27B's drafter holds

The same 27B load under its two drafters and with no speculation, read off the
three plans on the RTX 5090 (from 0.11.0, where the 27B takes its MTP head;
vision on, as every profile turns it on;
[finding](../../findings/2026-11-15-the-27bs-mtp-head-measured-on-the-rtx-5090.md)):

| | NVFP4, `rtx5090`, 8 lanes | 16 GB artifact, `rtx-16gb`, 2 lanes, 13 GiB budget |
|---|---:|---:|
| weights: DFlash2's 1,083 MB against the head's 451 MB | -631.6 MB | -631.6 MB |
| drafter window in the lanes' state (40 MiB a lane) | -335.5 MB | -83.9 MB |
| drafter round | -78.0 MB | -44.7 MB |
| workspace | 0 | -52.4 MB |
| decode graph, the head's residual window | +37.7 MB | +9.4 MB |
| **MTP against DFlash2, outside the KV pool** | **-1.01 GB** | **-0.80 GB** |
| MTP against no speculation, outside the KV pool | +0.71 GB | +0.56 GB |
| KV pool tokens: MTP, DFlash2, none | 1,039,040; 988,544; 1,182,336 | 232,320; 161,664; 307,200 |
| context loaded: MTP, DFlash2, none | 917,504 each | 131,072; 65,536; 262,144 |

The workspace is one arena for the prefill's scratch and the image encoder's:
on the NVFP4 load at 2,048-token chunks the encoder's share sizes it under all
three, and on the 16 GB load at 1,024-token chunks the prefill's does, where
DFlash2's taps add 92 MB to it and the head 40 MB. The head's layer in every
KV page costs the pool 6% of its tokens beside that, and an MTP load keeps a
page a lane back for the holders of shared prefixes. Read under WDDM's own
counter right after load, every one of these loads held less than its plan.

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
  (`kv_pool_policy`), `kv_pool_pages` and `kv_pool_tokens`, and
  `kv_holder_pages`: on the 27B's MTP load, the pages of the pool admission
  keeps back for the head's holders, a page a lane and a page a retained slot
  (0 on every other load). Each holder of a shared prefix keeps its own copy
  of the prefix's last page there; the plan holds those pages beside one
  context and a page per retained slot, so requests are admitted against the
  pool less them and the pool is never short of a page they asked for.
- The Playground's Monitor shows the plan beside what is in use
  ([the Playground](../playground/index.md)).
- Prometheus gauges carry the plan lines, the KV pool and KV-RAM; they read zero
  while no model is loaded ([telemetry](../guides/telemetry.md)).
