---
title: "VRAM budget and the KV pool"
description: "How the VRAM plan sizes the KV pool and retained state, and the knobs that move it."
sidebar:
  order: 6
---

Laid out at load and printed as the `ignis.runtime.vram_plan` event
(`ignis.runtime.flash_next_plan` on Flash-Next,
[ADR 0030](../../adr/0030-device-memory-reserved-at-load.md)). A plan that cannot
hold one full context refuses the start.

## The KV pool

The pool is sized by whether every weight is on the device
([ADR 0045](../../adr/0045-the-kv-pool-follows-residency-and-state-goes-down-to-disk.md)),
and the plan event says which branch it took (`kv_pool_policy`, with
`kv_pool_pages` and `kv_pool_tokens`; `make config` prints the same before a
start):

- **Resident** — the 27B always, and Flash-Next on a card that holds every
  expert (around 96 GB): the pool takes the rest of the VRAM budget. On
  Flash-Next that rest includes the 1.57 GB a prefill staging ring would
  take: every expert has its slot, so nothing stages and the plan reserves
  no ring (`prefill_staging_bytes` 0). That is ~5,825 pages, ~373K tokens,
  more than a plan with the ring (arithmetic from the plan, not measured on
  the card).
- **Offloaded** — Flash-Next on a card whose experts stream over PCIe, as on
  a 32 GB 5090: the pool is reserved first, **524,288 tokens** shared by the
  lanes, and the expert cache takes the rest. The pool never holds more than
  every lane's whole context (so one lane keeps a one-context pool and the
  larger cache) and never less than one `--model-max-context` sequence and a page
  per retained slot (so a 524,288-token context starts).

`--vram-kv-pool-bytes` replaces either size. On a resident load it becomes the pool
and the rest of the budget stays unused.

A request reserves its prompt plus its cap when it is admitted: its
`max_tokens`, or `--model-default-max-tokens`. When a request cannot fit, a
lower-ranked sequence (an `agent` one, for an `interactive` request) moves to
KV-RAM or [KV-disk](kv-disk.md) and later resumes exactly where it stopped.
The move goes a window at a time between decode rounds, so the other
requests keep generating while it happens, and the request that needed the
room starts once the move has landed (on Flash-Next, about a second for a
whole-context sequence). Otherwise the request waits until room returns. No request is refused for it,
and with KV-disk on (Flash-Next's default) none loses work for it; without it
(the 27B's default) KV-RAM can still give a moved sequence up for a newer one,
which is then prefilled again, as before. Rank is the class, then the order of
arrival: an `agent` request never moves an `interactive` one, nor an older
`agent` one, and a sequence that moved comes back before anything that ranks
below it.

Below 12 GiB the expert cache refuses the start: decode slows sharply there.
The message names the knobs that lift it: a smaller `--vram-kv-pool-bytes`,
`--model-prefill-chunk` or `--vram-headroom-bytes`, or fewer `--reuse-retained-device`.
`--vram-allow-expert-cache-below-floor` starts anyway, with an
`ignis.runtime.expert_cache_below_floor` warning carrying the cache, the floor
and those knobs. A cache too small for one decode step's experts in every
class is refused whatever the flag says.

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--vram-headroom-bytes <b>` | `IGNIS_VRAM_HEADROOM_BYTES` | `1536M` | Derives the budget: the device memory free at start minus this. Not with `--vram-budget-bytes`. |
| `--vram-budget-bytes <b>` | `IGNIS_VRAM_BUDGET_BYTES` | derived | The device memory the whole process may hold, weights included. More than is free refuses the start. |
| `--vram-allow-oversubscription` | `IGNIS_VRAM_ALLOW_OVERSUBSCRIPTION` | off | With `--vram-budget-bytes` only: start above free memory (or below the plan's minimum) with a warning instead of a refusal. On Windows that pages. Both models. |
| `--vram-adapt-spare-tokens <tokens>` | `IGNIS_VRAM_ADAPT_SPARE_TOKENS` | `65536` | The 27B only: at more than one decode lane, a context its load adapts is kept only when the KV pool holds it and a spare reservation of this many tokens, room for a second request ([Adaptation](index.md)); `0`, one sequence is enough. At one lane the plan's own floor decides. A Flash-Next start refuses it. |
| `--vram-allow-expert-cache-below-floor` | `IGNIS_VRAM_ALLOW_EXPERT_CACHE_BELOW_FLOOR` | off | Flash-Next only: start with an expert cache below its 12 GiB floor, with a warning ([above](#the-kv-pool)). The 27B, which has no expert cache, refuses it. (`make` knob `ALLOW_EXPERT_CACHE_BELOW_FLOOR=1`.) |
| `--vram-pinned-expert-bytes <b>` | `IGNIS_VRAM_PINNED_EXPERT_BYTES` | `0`; `2G` under the `rtx5090` profile | Flash-Next only: the calibration-hottest experts held in the expert cache for the whole load, outside its LRU, with no copy in RAM, up to these bytes: the RAM the experts take shrinks by as much ([Host RAM for Flash-Next](host-ram.md#host-ram-for-flash-next)). `0` pins none. A pin that cannot leave every class the slots of the widest decode step (under MTP a verify round's lanes x (drafts + 1)) is refused, naming the most it can pin. The 27B refuses it when named, and passes the profile's value over. (`make` knob `PINNED_EXPERT_BYTES`.) |
| `--reuse-prompt <on\|off>` | `IGNIS_REUSE_PROMPT` | `on` | `off`: no prompt checkpoint is captured or reused, and no prefix is shared unless `--reuse-retained-device` or `--reuse-retained-host` gives slots for it. |
| `--reuse-retained-device <n>` | `IGNIS_REUSE_RETAINED_DEVICE` | `0`; `2` for the 27B under the `rtx-24gb` profile, `0` there with `--reuse-prompt off` | Retained slots in VRAM: reserved in the VRAM plan, copied device to device (~0.3 ms), and handed out before any host slot. A card with VRAM to spare can take `--reuse-retained-device 16 --reuse-retained-host 0`. |
| `--reuse-retained-host <n>` | `IGNIS_REUSE_RETAINED_HOST` | two per decode lane of the 27B (16 at its 8); `0` with `--reuse-prompt off` | Retained slots in one pinned host block reserved at start, ~222 MiB each at the default load (3.5 GiB for 16): no VRAM, so the KV pool gets it; a capture and a claim each cost a PCIe copy (~15–19 ms). Both kinds hold the images of retained checkpoints and shared prefixes. When none is free, retained state gives one up — checkpoints before prefixes, `agent` before `interactive`, then least recently used; when nothing can, the publish or capture is skipped. `--retained-slots` was replaced by these two and now refuses the start. |
| `--reuse-retained-interactive-ttl <secs>` | `IGNIS_REUSE_RETAINED_INTERACTIVE_TTL` | `300` | Idle seconds after which a main-conversation checkpoint in KV-RAM ranks as a subagent's. Needs `--reuse-prompt on`. |
