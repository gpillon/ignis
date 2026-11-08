---
title: "A 125B MoE on a 32 GB card"
description: "Flash-Next's experts in a 2.5-bit trellis code, the hot ones in VRAM and the rest streamed from pinned RAM, at about 100 tok/s on one lane."
sidebar:
  order: 4
---

Qwen3.8-Flash-Next is a 125B-parameter mixture of experts. It does not fit in
32 GB, and on most engines that ends the discussion. ignis stores its experts
in a **2.5-bit trellis code**, keeps the ones in use in a VRAM cache and the
rest in pinned host RAM, and decodes at **about 100 tok/s on one lane** on an
RTX 5090 with a PCIe 3.0 host. On an RTX PRO 6000, where the experts fit, the
same model with its MTP head reaches 258 tok/s on one lane of code and 427
tok/s across eight.

## How it works

Each of Flash-Next's 48 layers holds 512 small MLPs, the experts, and a router
picks ten per token. Only about 6B parameters work on a given token, which is
why a model this size can decode fast at all. The experts are re-quantized to
a trellis code with a K per expert: 2, 2.5, 3 or 4 bits, chosen from
calibration error, 2.5 bits per weight on average. At that rate the 120.8B
expert parameters take 37.7 GB, pinned in host RAM
([ADR 0044](../../adr/0044-experts-in-trellis-with-a-k-per-expert.md)).

On a 32 GB card a **VRAM expert cache** holds the experts in use, replaced
least-recently-used, and a token whose experts are not there waits for them to
cross PCIe. So decode on this card is bound by the link, and the engine works
on that. A router lookahead copies the next layer's experts ahead of need,
under a per-step budget. An opt-in pinned hot set keeps the calibration-hottest
experts out of the LRU for the life of the load
([ADR 0055](../../adr/0055-a-pinned-hot-set-is-opt-in.md)). The KV pool is
reserved first and the cache takes what is left, so the cache and the pool
trade against each other, and the pool follows residency
([ADR 0045](../../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)).

On a card that holds every expert, around 96 GB, the experts are read from the
artifact straight into their VRAM slots and never copied again; no host pool
exists
([ADR 0054](../../adr/0054-a-resident-load-keeps-its-experts-only-in-vram.md)).
There the decode is limited by weight streaming as the 27B's is, so Flash-Next's
MTP head pays; on the 5090 it is off by default, because a verify round selects
experts for every column and so copies more of them.

Where each piece sits, and why: [Models and artifacts](../concepts/models-and-artifacts.md#qwen38-flash-next).

## The numbers

| Figure | Measured on | Source |
|---|---|---|
| **101.5 tok/s**, 9.85 ms a token, one lane | RTX 5090, PCIe Gen 3 host, the 3-lane default, one lane decoding; the decode fusion then took a one-lane token on harder prompts from 10.49 to 9.83 ms | [Flash-Next decode fusion](../../findings/2026-10-08-flash-next-decode-fusion.md) |
| 101.7-102.1 tok/s with one lane decoding at the 3-lane default, 109.7 tok/s across three; the shared KV pool gives the expert cache 1.03 GiB more than a pool of three full contexts (+4.1% and +10.1%) | RTX 5090, 262,144 tokens a lane, greedy prose, 2026-10-08 | [KV pool follows residency](../../findings/2026-10-08-flash-next-kv-pool-follows-residency.md) |
| 2.5 bits a weight on average (K of 2, 2.5, 3 or 4); 37.7 GB of experts pinned | calibration-derived allocation | [ADR 0044](../../adr/0044-experts-in-trellis-with-a-k-per-expert.md) |
| MTP on the 5090: 1.02-1.30x on one lane, 7-15% slower at two and three lanes at the default row budget | RTX 5090 | [MTP is PCIe-bound](../../findings/2026-10-07-flash-next-mtp-speculation-is-pcie-bound.md) |
| **258.2 tok/s** one lane of code, **427.5 tok/s** across eight lanes of code (MTP, 3 drafts), 512 tokens | RTX PRO 6000 Server Edition | [RTX PRO 6000 profile](../../findings/2026-10-18-rtxpro6000-profile-measured.md) |
| MTP up to 1.85x speculation off on one lane of code | same, every lane decoding | same |
| Lanes keep 15.6-18.3 tok/s while a prompt prefills at decode share 25 | RTX 5090 | [Prefill chunk and decode share](../../findings/2026-10-07-flash-next-prefill-chunk-and-decode-share.md) |

:::note[The link sets the speed]
Every figure on the 5090 is for a PCIe 3.0 x16 host (about 12 GB/s). The
decode is copy-bound, so a faster link moves it. The RTX PRO 6000 figures are
a different regime: the experts are resident and the bound is the weight
stream.
:::

## See it, control it

- **Lanes**: [`spec.decode_lanes`](../reference/configuration.md#specdecode_lanes),
  default 3; fewer lanes give the expert cache more VRAM.
- **Speculation**: [`spec.backend`](../reference/configuration.md#specbackend)
  `mtp` and [`spec.draft_rows`](../reference/configuration.md#specdraft_rows),
  which can go past 8 verify rows, numerically equivalent
  ([ADR 0056](../../adr/0056-speculation-past-eight-verify-rows-is-numerically-equivalent.md)).
- **The cache**: [`vram.pinned_expert_bytes`](../reference/configuration.md#vrampinned_expert_bytes)
  for the hot set, and
  [`vram.allow_expert_cache_below_floor`](../reference/configuration.md#vramallow_expert_cache_below_floor)
  for the 12 GiB floor below which a start is refused.
- **Host RAM**: serving needs about 46-47 GB available; see [host RAM](../configuration/host-ram.md).
- **Metrics**: `ignis_expert_cache_hits_total`, `ignis_expert_cache_misses_total`,
  `ignis_expert_prefetches_issued_total`, `ignis_expert_residency_stall_seconds_total`
  ([metrics](../reference/metrics.md)); the load prints `ignis.runtime.flash_next_plan`.
- **Playground**: the [Monitor](../playground/monitor.md) and the
  [Server view](../playground/server.md).

## Limits

- On a 32 GB card the decode is bound by PCIe, not by the GPU. MTP is off by
  default there and helps only one user or one agent at a time.
- The load needs about 46-47 GB of available RAM, plus the n-gram table's
  read path ([n-gram cache](../configuration/ngram-cache.md)).
- Below a 12 GiB expert cache the start is refused, because decode slows
  sharply there.
- The 258 and 427 tok/s are one run each on one server; the finding states
  its own conditions.

## Read more

- [Models and artifacts](../concepts/models-and-artifacts.md): the families and where their parts live.
- [Flash-Next MTP](../configuration/flash-next-mtp.md): turning speculation on.
- [Speculative decoding](../concepts/speculative-decoding.md#why-mtp-is-off-on-the-rtx-5090-and-on-for-the-rtx-pro-6000): why the two cards differ.
- [Every byte planned at load](every-byte-planned.md): how the cache and the pool are carved.
