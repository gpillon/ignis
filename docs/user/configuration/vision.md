---
title: "Vision"
description: "The 27B's vision tower: on, off, and its token budget."
sidebar:
  order: 10
---

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vision-enabled` | `IGNIS_VISION_ENABLED` | off | Load the vision tower and reserve its workspace. Without it every `image_url` part is refused with `vision_disabled`, whatever the artifact holds. |
| `--vision-max-tokens <n>` | `IGNIS_VISION_MAX_TOKENS` | `32768` with `--vision-enabled` (`1024` under the `rtx50-16gb` profile) | Merged vision tokens per request, 1..1,048,576. An image whose natural size is past the cap is downscaled to fit (aspect ratio kept, one token = 32 x 32 pixels), not refused, unless its aspect ratio is too extreme for the cap (a side cannot go below 32 pixels); the `ignis.request.admitted` line reports `media.vision_tokens` (admitted) and `media.natural_vision_tokens`. Several images whose total is past the cap are still refused with `media_budget_exceeded`. |
| `--vision-embedding-pool-mib <n>` | `IGNIS_VISION_EMBEDDING_POOL_MIB` | one envelope-wide embedding | Encoded images kept for reuse across requests, 1..65536, counted where `--vision-residency` puts the pool ([below](#where-the-vision-tower-and-the-embedding-pool-live)). |
| `--vision-residency <host\|device>` | `IGNIS_VISION_RESIDENCY` | `device` | Needs `--vision-enabled`. Where the vision tower's weights and the embedding pool live: `device` in VRAM, `host` in pinned host RAM, the weights copied to the GPU for each image ([below](#where-the-vision-tower-and-the-embedding-pool-live)). Read at load: a change takes effect at the next one. |
| `--media-cache-mib <n>` | `IGNIS_MEDIA_CACHE_MIB` | `1024` with `--vision-enabled` | Prepared images kept for reuse; `0` disables, max 65536. |
| `--media-allow-private-network` | `IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` | off | Needs `--vision-enabled`. Fetch image URLs on private, loopback and link-local addresses. |

## Where the vision tower and the embedding pool live

The 27B's vision tower is 296 MB of weights (295,711,648 bytes). Its embedding
pool holds encoded images for reuse (`--vision-embedding-pool-mib`, one
envelope-wide image unless named: 10 MiB at a 1,024-token envelope, 40 MiB at
4,096, 320 MiB at the default 32,768). The vision residency
(`--vision-residency`, `vision.residency`, `IGNIS_VISION_RESIDENCY`) says where
a vision load keeps both:

- **`device`** (the default): in VRAM for the life of the load, as every
  vision load before it. An image encodes at full speed, and the tower and the
  pool come out of the KV pool whether or not an image ever arrives. It is the
  choice for a card with VRAM to spare on a host short of RAM, and for the
  lowest encode latency.
- **`host`**: in pinned host RAM. The tower sits beside the token embedding
  ([Host RAM for the 27B](host-ram.md#host-ram-for-the-27b)), and for each image
  the encode copies it to the GPU a block at a time, through three 25 MB slots
  it holds in the prefill scratch, while the block before computes. The pool is
  mapped: the encode writes each page of an image's embedding on the GPU and
  copies it there, and the prefill reads the image's columns from host RAM
  where they sit. Between images neither takes VRAM: the VRAM plan's
  `weights_bytes` leaves the tower out, its `media_embedding_bytes` is 0, and
  its `workspace_bytes` grows by the slots (75 MB) only where the image
  encoder, not the prefill chunk, sets the scratch's size. Under the
  `rtx50-16gb` profile at a 1,024- or 4,096-token envelope it does not grow,
  and a vision load keeps the KV pool of a text-only one. An encode takes a few
  milliseconds more: about 6 to 8 ms on a 1,024-token image (about 40 ms under
  `device`) on a PCIe Gen 3 host under Windows, and nothing measurable on a
  16,384-token one
  ([finding](../../findings/2026-10-30-vision-tower-staged-and-pool-host-mapped.md)).
  The answers do not change: the image's embedding is the same to the bit. A
  fan-out of questions over one image encodes it once, so only the first pays.

`--vision-embedding-pool-mib` keeps its meaning under both, with the same
default and floor: it sizes the pool where the residency puts it, VRAM under
`device` and pinned host RAM under `host`.

A `host` start whose host RAM cannot hold the tower and the pool beside the
host margin is refused before anything is loaded; it never moves them to the
device on its own, so the context a start gets never depends on how much RAM
happened to be free. The host plan names the tower and the pool on lines of
their own, and the message says how much is missing and the two ways out:

```
not enough RAM: the load holds … and keeps 4 GiB free for the system, so it needs … available, but
only … is (503 MiB short). The token embedding fits, the vision tower (283 MiB) and its embedding
pool (320 MiB) do not: keep them in VRAM (vision.residency: device, --vision-residency device), turn
vision off, or close other applications. Plan: token embedding 1.3 GiB, vision tower 283 MiB,
embedding pool 320 MiB, …
```

On Windows, Task Manager counts the pinned tower and pool as shared GPU
memory, as it does the token embedding: that is host RAM, not VRAM
oversubscription.

The `ignis.runtime.vram_plan` and `ignis.runtime.qwen27b_host_plan` log events
carry the load's `vision_residency` (a text-only load has none), and the host
plan event the tower's and the pool's lines as `vision_tower_bytes` and
`vision_pool_bytes`. The reasons are in
[ADR 0075](../../adr/0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md).
