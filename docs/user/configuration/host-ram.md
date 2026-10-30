---
title: "Host RAM"
description: "What each model needs in host memory, and how the host plan refuses a start that does not fit."
sidebar:
  order: 9
---

## Host RAM for Flash-Next

Serving Flash-Next needs about **46-47 GB of available RAM**: 38 GB of pinned
experts (2.1 GB less under the `rtx5090` profile's pinned hot set, below), 1 GB of n-gram hot rows, ~0.5 GB of staging (KV-disk's 64 MiB
included: `--qwen38flashnext-kv-disk-bytes 0` drops it), ~1 GB of retained host
slots, the 2 GiB KV-RAM arena and the 4 GiB margin the load refuses to dip into
(`--model-host-margin-bytes`: it never lets reuse push Windows into paging).
"Available" is free memory plus the page cache the system can drop (Windows'
free plus standby, Linux's `MemAvailable`); in a Linux container on cgroup v2
it is capped by the container's memory limit (every `memory.max` above the
process, less what is charged there and not page cache), so a load inside a
smaller limit is refused rather than OOM-killed. On cgroup v1 the cap is not
read: give such a container a limit near the machine's RAM.

**On a card that holds every expert** (around 96 GB: the VRAM plan's
`kv_pool_policy: resident`) there are no pinned experts at all: every record is
read from the artifact straight into its VRAM slot, a layer at a time, and no
step ever copies an expert in, so the plan's `expert_pool` line is 0. The
layer buffers the load reads the experts through are a line of their own,
`load_staging` (~4.7 GB): they are freed before the n-gram hot rows are read,
so the plan charges them only while the experts load, and holds at its peak
the larger of that phase and the lines kept while serving. The same load then
needs about **9 GB available** (the 4.7 GB of layer buffers, more than the
~4.3 GB of lines it keeps, and the 4 GiB margin), and `--ngram-hot-bytes auto`
holds the whole n-gram table (28.8 GB) from about 37 GB available (36 GB of
lines and `auto`'s 256 MiB headroom), which a 64 GB machine has. Both figures
are arithmetic over the plan's logged lines.
Measured on an RTX PRO 6000 (2026-10-18): the pinned memory went from 41.1 to
3.3 GB, decode and greedy text unchanged, the expert cache's hit rate 1.0
instead of 0.994. On a 64 GiB host in a 56 GiB container `auto` planned the
whole table; the experts are read through the page cache, so with the table
in RAM a reload read them in ~31 s, and in 5 s only when the cache held them
all ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)).

A load below that is refused
in GiB, with how far short it fell and the plan line to shrink:

```
not enough RAM: the load holds 39.2 GiB and keeps 4 GiB free for the system, so it needs 43.2 GiB
available, but only 38.8 GiB is (4.5 GiB short). Even the experts alone (35.2 GiB) do not fit: close
other applications. Plan: experts 35.2 GiB, n-gram hot rows 1 GiB, staging 64 MiB, retained host slots
992 MiB, KV-RAM arena 2 GiB; margin 4 GiB (--qwen38flashnext-model-host-margin-bytes).
```

A device-only load lists its layer buffers apart, as `expert layer buffers
4.4 GiB (while the experts load)`, and one that cannot hold them is refused
naming them (`crossing_line: load_staging`): no flag shrinks them, so close
other applications.

The same refusal in exact bytes is the `ignis.runtime.flash_next_host_plan_refused`
log event (every line, the plan's peak as `planned_bytes`, `available_bytes`,
`margin_bytes`, `needed_bytes`, `short_bytes`, `crossing_line`) and, after a
switch, `GET /v1/models`'s `last_switch.host_plan`. The knobs that lower it,
with what each costs:

| Knob (`make`) | Flag | Saves | Costs |
|---|---|---|---|
| `RETAINED_HOST` | `--qwen38flashnext-reuse-retained-host <n>` | ~125 MiB per slot below the default 8 | Fewer retained prompt checkpoints and shared prefixes: a returning agent or a subagent sharing a system block prefills what no slot kept. `0` retains nothing on the host. |
| `FN_KV_HOST_POOL_BYTES` | `--qwen38flashnext-reuse-kv-host-pool-bytes <n>` | up to the 2 GiB arena (`0` turns it off) | The KV-RAM tier holds fewer suspended requests: under VRAM pressure a request that cannot be parked there is evicted and prefilled again. |
| `NGRAM_HOT_BYTES` | `--ngram-hot-bytes <n>` | up to the 1 GiB default (`0` holds none) | Each n-gram row not held in RAM is read from NVMe when a step needs it: more reads per token, and slower prefill. |
| `HOST_MARGIN_BYTES` | `--model-host-margin-bytes <n>` | what it lowers the 4 GiB default by (`0` leaves none) | Less room for the rest of the machine to grow into: the plan's lines are pinned and cannot be paged, so once other applications outgrow what is left, Windows pages *them* out. |
| `PINNED_EXPERT_BYTES` | `--vram-pinned-expert-bytes <n>` | about `<n>` of the experts line (`2G` under the `rtx5090` profile, else off) | The pinned hot set (below): the LRU loses those slots. Measured on the 5090: nothing at `1G`, up to ~1% of three-lane decode at `2G`, 5.9% at `4G`; the load takes ~2 s longer. |

Check `make config MODEL=flash-next` for the lines a start will use.

**The pinned hot set** (`--vram-pinned-expert-bytes`, `IGNIS_VRAM_PINNED_EXPERT_BYTES`,
`vram.pinned_expert_bytes`; 0 by default, `2G` under the `rtx5090` profile, ADR 0055). On a card whose expert cache
holds only part of the experts, as on a 32 GB one, the load can keep its
calibration-hottest experts in the cache for its whole life: whole experts,
hottest first, until they reach the bytes named. They sit outside the cache's
LRU, are never evicted or copied, and have no copy in RAM: the `expert_pool`
line, and the process's RAM, shrink by their bytes. The trade is the LRU's
slots they take. On the RTX 5090 (2026-10-18,
[finding](../../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md)) `1G`
gave back 1.075 GB for no measurable decode cost, `2G` 2.149 GB for 0.1-0.2%
of one-lane and 0.2-1.1% of three-lane decode, and `4G` 4.296 GB for 4.3% and
5.9%; greedy output is the same at each. Past 2 GiB the cost climbs, as the
replay study
([finding](../../findings/2026-10-18-partial-exclusive-expert-pool.md)) predicted
(3.8-5.3% of three-lane decode at 4 GB), and faster: the hottest experts are
mostly 4-bit, and pinning them takes most of those classes' LRU. The
records the pin keeps reach VRAM
through a few buffers held while the experts load (the plan's `load_staging`
line, ~0.3-0.9 GB). The load logs `ignis.runtime.flash_next_pinned_experts`
with what it pinned; the Monitor's expert slots and `ignis_expert_cache_slots`
count the LRU's slots only. A pin the cache cannot hold while every K class
keeps the slots of the widest decode step (a verify round's rows under MTP)
is refused at start, naming the most it can pin and where the value was set:
the flag, the environment, a config file or the profile. A card that holds every expert keeps none in RAM already
and ignores the field; the 27B refuses it.

A host with RAM to spare can go the other way: `--ngram-hot-bytes auto`
(`make MODEL=flash-next NGRAM_HOT_BYTES=auto`) gives the hot rows what the
plan leaves, and with about 28 GB more available than the figure above
(~75 GB) it holds the whole table and the NVMe is out of the decode path. The
`rtxpro6000` profile sets `auto`. On a cloud block volume (EBS gp3, 16,000
IOPS) the whole table measured 1.24x the 1 GiB default's decode at one and
eight lanes and 2.3x its prefill (a 32K-token TTFT of 7.4 s instead of 16.9 s,
[finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)). A local
NVMe was not measured.

## Host RAM for the 27B

The 27B keeps its token embedding (1.35 GB, a 5,440-byte row per token) in
pinned host memory rather than VRAM (ADR 0062): the card reads one row per
token over PCIe and gives the 1.35 GB to the KV pool. So the 27B plans host
memory too, the way Flash-Next does: the table, the host retained slots
(232.5 MB each, 16 by default: 3.72 GB), the KV-RAM arena (2 GiB, 8 GiB under
`make`), the vision tower and the embedding pool under the `host` vision
residency (below) and KV-disk's staging when the tier is on must leave the
4 GiB margin of the available RAM. Under the `make` defaults on a 32 GB card,
whose profile turns vision on (297 MB of tower and a 320 MiB pool), that is
about **18.6 GB available**; without vision, about 18 GB. A load below it is
refused before anything is pinned, naming the 27B's own flags:

```
not enough RAM: the load holds 12.7 GiB and keeps 4 GiB free for the system, so it needs 16.7 GiB
available, but only 14 GiB is (2.7 GiB short). The model and its retained slots fit, the KV-RAM arena
does not: a smaller one (--qwen38-reuse-kv-host-pool-bytes, 0 turns it off) or close other
applications. Plan: token embedding 1.3 GiB, retained host slots 3.5 GiB, KV-RAM arena 8 GiB; margin
4 GiB (--qwen38-model-host-margin-bytes).
```

The same refusal in exact bytes is the `ignis.runtime.qwen27b_host_plan_refused`
event, and a load that fits logs `ignis.runtime.qwen27b_host_plan`. The knobs
are Flash-Next's, under the `qwen38` scope: `--qwen38-reuse-retained-host`,
`--qwen38-reuse-kv-host-pool-bytes`, `--qwen38-kv-disk-bytes` and
`--qwen38-model-host-margin-bytes`.

A vision load under `--vision-residency host` pins the vision tower beside the
table and maps the embedding pool in pinned RAM of its own, each on a line of
its own: `vision_tower`, 297 MB with its page padding, and `vision_pool`, the
pool `--vision-embedding-pool-mib` sizes (`vision_tower_bytes` and
`vision_pool_bytes` in both events). A start that cannot hold them is refused
naming `--vision-residency device` and turning vision off as the ways out
([Where the vision tower and the embedding pool live](vision.md#where-the-vision-tower-and-the-embedding-pool-live)).
So is a start the plan let through but whose pin the system then declined
(it may refuse to lock memory it reports available): its load fails with
`materialize weights:`, the system's error, the bytes the plan had counted,
the tower's and the pool's among them, and the same ways out.
