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

**The 27B's MTP head** (`--spec-backend mtp`, from 0.11.0) changes the 27B's pool twice.
Its attention state is one more layer in every page, so a page holds the same
64 tokens in 17/16 of the bytes and the same budget buys about 6% fewer tokens
(DFlash2 keeps its own state outside the pool, in 40 MiB a lane and a retained
slot that the head does not take). And every holder of a shared prefix keeps
its own copy of the prefix's last page, so the floor holds a page a lane and a
page a retained slot more: one `--model-max-context` sequence, a page per
retained slot, and those (`kv_holder_pages` in the plan event). Requests are
admitted against the pool less them, so `kv_pool_pages` and `kv_pool_tokens`
are that many pages more than a request can reserve
([the 27B's two drafters](../concepts/speculative-decoding.md#the-27bs-two-drafters)).

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

## The 12 GB class

From 0.11.0 a card below 14 GiB starts on the
[`rtx-12gb` profile](../concepts/profiles.md#rtx-12gb), which derives the
27B's budget from 512 MiB of headroom (`vram.qwen38.headroom_bytes`): the
memory free at the start less 512 MiB, where the larger classes leave
1,200 MiB and the default 1,536 MiB. The load runs one lane drafting with MTP,
so the KV pool takes what the weights, the workspace and that lane leave, and
must hold one 131,072-token context. When it cannot, the load lowers the
context 32,768 tokens at a time and says so in the plan event's `adapted`:
`max_context 131072 -> 98304 (adapt_context_step 32768, one lane)`. A card of
10 or 11 GB is read as the class, and its plan refuses the start after every
step, naming the bytes it misses.

The plan of a 12 GB load on the class's own artifact, measured on the RTX
5090 under an explicit 12 GiB budget
(`--profile rtx-12gb --vram-budget-bytes 12884901888`), as its
`ignis.runtime.vram_plan` event gives it:

| Line | Bytes |
|---|---:|
| `weights`: the text scope and the MTP head (the vision tower is in host RAM) | 9,651,141,632 |
| `cuda_context`, measured | 452,161,536 |
| `workspace`: at the 512-token chunk, the vision encoder's share sets it | 302,433,536 |
| `lane_state` | 154,947,584 |
| `residual` | 124,474,800 |
| `hq_residual_window` | 37,879,936 |
| `trellis_workspace` | 32,134,400 |
| `verify_round` | 19,017,436 |
| `decode_graph` | 12,802,060 |
| `sampling` | 579,048 |
| `kv_pool`: 3,346 pages, 214,144 tokens | 2,096,906,240 |
| total, nothing adapted | 12,884,478,208 |

A token of KV costs 9,792 bytes there, not 9,216: the MTP head keeps its own
attention layer in the pool. The pool's floor at 131,072 tokens is 2,053
pages: one sequence, a page for each of the lane's two retained slots, and
the three the head's copies keep back. In an 11 GiB budget the same lines
leave 1,633 pages, 104,512 tokens, so the load takes the next step, 98,304,
and its plan's `adapted` says
`max_context 131072 -> 98304 (adapt_context_step 32768, one lane)`. On the
routes an RTX 30 or 40 card runs, the dense prefill's 89 MB buffer takes 142
pages of the pool ([the profile](../concepts/profiles.md#rtx-12gb)).
