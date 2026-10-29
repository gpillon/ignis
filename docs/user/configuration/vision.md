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
| `--vision-embedding-pool-mib <n>` | `IGNIS_VISION_EMBEDDING_POOL_MIB` | one envelope-wide embedding | Encoded images kept for reuse across requests, 1..65536. |
| `--media-cache-mib <n>` | `IGNIS_MEDIA_CACHE_MIB` | `1024` with `--vision-enabled` | Prepared images kept for reuse; `0` disables, max 65536. |
| `--media-allow-private-network` | `IGNIS_MEDIA_ALLOW_PRIVATE_NETWORK` | off | Needs `--vision-enabled`. Fetch image URLs on private, loopback and link-local addresses. |
