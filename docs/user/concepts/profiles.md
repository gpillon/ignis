---
title: "Profiles and defaults"
description: "What a hardware profile is, where it sits between the config file and the built-in defaults, what each built-in profile chooses and why, and how to define your own."
sidebar:
  order: 8
---

Most settings have a sensible default that works anywhere. A few do not: the
longest context a card can hold, how many lanes it can feed, whether a draft
head pays for its memory. Those depend on the card. A **profile** is a named
bundle of such defaults for one kind of card. It is a smarter default, never an
override: anything you set yourself wins over it.

## Where a profile sits

Every field is resolved from five places, highest first
([configuration](../configuration/index.md)):

```text
flag  >  environment variable  >  config file  >  profile  >  built-in default
```

So a profile fills in only what you left unsaid. `--model-max-context 131072`
beats the profile's context; a profile's value beats the hardcoded default; and a
field the profile does not mention keeps its hardcoded default. An empty value
counts as unset.

Choose a profile with `--profile <name>`, `IGNIS_PROFILE`, or `profile:` in the
config file. With none of them the profile is `auto`: the start reads the
card's total memory from NVML and takes the built-in profile of its class,
`rtx-16gb` under 20 GiB, `rtx-24gb` from
20 GiB up to 28 GiB, `rtx5090` from
28 to 40 GiB, `rtxpro6000` from 88 to 104 GiB. The log line
`ignis.config.source` names the profile chosen and
the total it was chosen by. To skip the layer entirely and run on the hardcoded
defaults, name `none`. A name that is not defined is refused with the list of
the ones that are.

:::caution
Under `auto`, a card of any other size, or a total NVML cannot read, is
refused, naming `--profile`: name the profile that fits the card, or `none`. A
build without CUDA has no card, so `auto` there is `none`.
:::

Profile values can be scoped to a model family in the same shape as the file:
`qwen38` for the 27B, `qwen38flashnext` for Flash-Next. A field only one family
has, such as the number of decode lanes or the pinned expert bytes, is passed over
by the other family, not refused. `GET /v1/config/origins` says where each value
in force came from, `profile` included.

## The built-in profiles

A card's profile follows from its memory, not its generation; what it loads
follows from both. Which cards run ignis, and how far each is verified, is
[What it runs on](../start/index.md#what-it-runs-on); which model each of them
loads is the table of
[Where each model loads](../reference/models.md#where-each-model-loads).
No profile here carries a speed for an RTX 30 or RTX 40 card.

`rtx5090` and `rtxpro6000` carry numbers measured on the card they name,
with the finding for each choice linked below; `rtx-16gb` is derived from
them, since no 16 GB card exists in the project; `rtx-24gb` takes the 5090's
values for the 27B, at a context measured on the 5090 held to a 24 GiB
budget.

### `rtx5090`

The default: a 32 GB RTX 5090.

| Field | 27B | Flash-Next |
|---|---|---|
| Context (`model.max_context`) | 917,504 | 589,824 |
| Vision | on, tower and pool in host RAM (`vision.residency: host`) | off; from 0.11.0, with `--vision-enabled` or after a switch from the 27B serving images, tower and pool in host RAM ([Flash-Next](../configuration/vision.md#flash-next)) |
| Speculation | DFlash2, 7 draft tokens | off |
| Decode lanes | 8 (the default) | 3 |
| Prefill chunk | 2,048 | 8,192 |
| VRAM headroom | 1,200 MiB | 1,536 MiB (the default) |
| Pinned hot experts | not applicable | 2 GiB |

Why each:

- **The 27B's context and headroom.** Raising `max_context` under vision until
  the plan refused it, then trading headroom down from 1,536 to 1,200 MiB, reached
  917,504 tokens with DFlash2 still on; 950,272 is refused by 59 MB. The
  alternative trade, turning speculation off, reached about 864K tokens, and the
  profile chose to keep DFlash2. Past the checkpoint's trained 262,144 positions
  the `auto` rope scaling applies YaRN at `context / 262,144`: a factor of 3.5 for
  the 27B and 2.25 for Flash-Next
  ([ADR 0057](../../adr/0057-rope-scaling-is-auto-by-default.md),
  [ADR 0059](../../adr/0059-flash-next-applies-yarn-on-q-and-k.md)).
- **The vision tower and pool in host RAM.** The 5090 stages the 296 MB tower
  for each image and maps the embedding pool in pinned host RAM, so a vision
  load keeps the KV pool of a text-only one ([Vision](../configuration/vision.md#where-the-vision-tower-and-the-embedding-pool-live)).
  On Windows, Task Manager counts the pinned tower and pool as shared GPU
  memory, as it does the token embedding: that is host RAM, not VRAM
  oversubscription ([the WDDM note](vram-plan.md#windows-and-wddm)).
- **The 27B's prefill chunk.** With vision on, the image encoder sizes the
  workspace, so a 2,048-token chunk costs no VRAM over 1,024. It prefilled
  1.9 and 2.2 percent faster at 46K and 105K, with a lower KLD against BF16 and
  decode unchanged; the cost is that a decoding lane's longest pause while a
  prompt prefills grows by about 65 percent
  ([finding](../../findings/2026-10-21-the-27b-prefill-chunk-at-2048.md)).
  Neither Blackwell profile names a
  [solo width](../configuration/context-and-speculation.md#the-solo-width):
  `auto` is the prefill chunk on these cards.
- **Flash-Next's prefill chunk.** 8,192 under `rtx5090` and `rtxpro6000`, as
  `make` always started it: each chunk streams the experts once, so a narrower
  one costs its TTFT nearly in proportion (2,048 took 1.86 times as long at
  34K) and under load only halves a decoding lane's longest pause, which the
  decode share already paces
  ([finding](../../findings/2026-10-07-flash-next-prefill-chunk-and-decode-share.md)).
  The wider scratch costs the expert cache 1.03 GiB against 1,024: from the
  ~16.6 GiB measured at 1,024 it keeps ~15.6. `rtx-24gb` and `rtx-16gb` give
  it 4,096 and 2,048, placeholders until Flash-Next runs on those cards.
- **Flash-Next's context and lanes.** 589,824 is 512K plus two 32K
  allowances, the floor asked for on both the expert cache (at least 15 GiB; it
  measured about 16.6) and the context. Three lanes is Flash-Next's own default.
- **Speculation off for Flash-Next.** Decode on this card is bound by the experts
  copied over PCIe, which MTP does not reduce
  ([speculative decoding](speculative-decoding.md)).
- **Pinned hot experts.** Keeping the calibration-hottest 2 GiB of experts in the
  cache gives back 2.149 GB of host RAM for 0.1 to 0.2 percent of one-lane and
  0.2 to 1.1 percent of three-lane decode, with greedy text unchanged. At 4 GiB
  the cost reached 4.3 and 5.9 percent
  ([finding](../../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md),
  [ADR 0055](../../adr/0055-a-pinned-hot-set-is-opt-in.md)).

### `rtxpro6000`

The RTX PRO 6000 Blackwell series (Workstation, Max-Q and Server Edition, 96 GB
each), the same architecture as the 5090. Measured on a Server Edition under
Linux ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)).

| Field | 27B | Flash-Next |
|---|---|---|
| Context | 1,048,576 | 1,048,576 |
| Vision | on, tower and pool in VRAM (`vision.residency: device`) | off; from 0.11.0, with `--vision-enabled` or after a switch from the 27B serving images, tower and pool in VRAM |
| Speculation | DFlash2, 7 draft tokens | MTP, 3 draft tokens |
| Decode lanes | 8 (the default) | 8 |
| Prefill chunk | 2,048 | 8,192 |
| VRAM headroom | default | default |
| N-gram hot rows | not applicable | `auto` |

Why each:

- **A million tokens of context for both.** It is the 27B's whole compressed-KV
  attention envelope, and the plan holds it. `auto` rope scaling makes both
  models YaRN at 4. No request that long was run.
- **Headroom left at the default.** Nothing has to be traded for vision on 96 GB.
- **The vision tower and pool stay in VRAM.** On 96 GB they are about 0.8% of
  the KV pool, and the host may be short of RAM to pin them, so this profile
  takes `device`. `--vision-residency host` overrides it.
- **Eight lanes.** The expert cache holds nearly every expert, so a lane costs
  its state and its share of the KV pool, and the aggregate peaks at eight.
- **MTP for Flash-Next, at the default row budget.** Few experts cross PCIe here,
  so speculation pays at one to four active lanes and with simulated agents. A
  wider row budget lost on this card, so the profile keeps 8 rows
  ([speculative decoding](speculative-decoding.md)).
- **`auto` n-gram hot rows.** The whole n-gram table in RAM keeps the disk out of
  every decode round, and `auto` takes it wherever the host plan leaves room,
  never refusing a start the 1 GiB default would make
  ([n-gram cache](../configuration/ngram-cache.md)).
- **No pinned hot set.** The profile leaves `vram.pinned_expert_bytes` at its
  default, 0.

### `rtx-16gb`

The 16 GB class: the RTX 5060 Ti 16 GB, 5070 Ti and 5080, and the RTX 4060 Ti
16 GB and RTX 4080. `rtx50-16gb`, its former name, is accepted as an alias of
`rtx-16gb`. It serves the 27B
from the catalog's 16 GB entry, `qwen3.8-27b-16g`; the profile names no
model, and a start with no artifact resolves the default id `qwen3.8-27b` to
that entry ([Get a model](../guides/models.md)).

| Field | 27B |
|---|---|
| Context | 262,144, the positions it was trained on |
| Vision | on, 4,096 tokens an image, tower and pool in host RAM |
| Speculation | DFlash2, 7 draft tokens, the drafter's head `auto` (the full head) |
| Decode lanes | 2 |
| Prefill chunk | 1,024 (the default), with no lane decoding too ([the solo width](../configuration/context-and-speculation.md#the-solo-width) pinned) |
| VRAM headroom | 1,200 MiB |

Why each:

- **Vision on, at 4,096 tokens.** The profile caps an image at 4,096 merged
  tokens, about 2,048 x 2,048 pixels (a larger one is downscaled). The vision
  tower and the embedding pool sit in pinned host RAM (`vision.residency:
  host`), so with no image in flight vision costs no context: a vision load
  keeps the KV pool of a text-only one, measured at 7,956 pages at 1,024 and at
  4,096 tokens on a 16 GB budget. `--vision-enabled false` turns it off and
  `--vision-max-tokens` sets your own envelope. The pinned host RAM it needs is
  [host RAM for the 27B](../configuration/host-ram.md#host-ram-for-the-27b).
- **Two lanes.** Eight would leave no room for the context.
- **The 1,024-token chunk.** The prefill scratch is sized by the chunk, and
  2,048 would cost about 0.4 GB of KV pool. The solo width is pinned to it for
  the same reason: on an RTX 40 card in a 16 GiB budget less 1 GiB, `auto`
  would take 2,048 and the KV pool from 386,048 tokens to 352,768.
- **The headroom** is the 5090's 27B value until a real card says otherwise
  ([measure yours](../guides/measure-a-16gb-card.md)).
- **On a 16 GB RTX 40 card** the 27B prefills on the routes those cards run
  ([Context and speculation](../configuration/context-and-speculation.md)):
  the dense prefill's 89 MB buffer is a line of the VRAM plan, and the prefill
  scratch it holds is smaller. On the RTX 5090 held to a 16 GB card's budget
  less 1 GiB of desktop, those routes left the KV pool 6,656 tokens smaller
  than an RTX 50 card's (386,048 against 392,704) and the context loaded at
  262,144 with nothing adapted: the buffer costs pool, not context.

### `rtx-24gb`

The 24 GB class: the RTX 3090 and the RTX 4090, from 20 GiB up to 28 GiB of
total memory. No catalog entry states this class: the 32 GB entries stay above
it and the 16 GB entries fit it, so it serves the 27B from `qwen3.8-27b-16g`,
as `rtx-16gb` does. The profile names no model, and a start with no artifact
resolves the default id `qwen3.8-27b` to that entry
([Get a model](../guides/models.md)).

| Field | 27B |
|---|---|
| Context | 1,048,576 |
| Vision | on, tower and pool in host RAM (`vision.residency: host`) |
| Speculation | DFlash2, 7 draft tokens |
| Decode lanes | 8 (the default) |
| Prefill chunk | 2,048, with no lane decoding too ([the solo width](../configuration/context-and-speculation.md#the-solo-width) pinned) |
| VRAM headroom | 1,200 MiB |
| Retained slots | 2 on the device, beside the 16 in host RAM |

Why each:

- **The 5090's values for the 27B.** The headroom, the 2,048-token chunk,
  DFlash2 at 7 drafts and vision in host RAM are what `rtx5090` gives the
  27B; the lanes and the drafter's head stay at their defaults.
- **A million tokens of context.** Measured as the 16 GB numbers were: the
  16 GB entry on the RTX 5090 held to a 24 GiB budget, on the routes a card
  below compute capability 12.0 runs (the dense prefill with its 89 MB buffer,
  the integer band, ignis's gated delta net prefill). 1,048,576 tokens load
  with nothing adapted, on eight lanes, with a KV pool of 1,139,776 tokens
  beside the two device retained slots (1,148,544 on the fused prefill and the
  FP16 band, 1,199,040 without the slots). The
  cap is the 27B's compressed-KV attention envelope, as on `rtxpro6000`, not
  the memory; `auto` rope scaling makes the 27B YaRN at 4. No request that
  long was run.
- **Two retained slots on the device.** The 27B's prompt checkpoints and
  shared prefixes go to them first, and a capture or a claim is then a copy
  on the card instead of one over PCIe
  ([retained slots](kv-cache-and-prompt-reuse.md#what-is-kept)). Each is about
  232 MB, a lane's state, and comes out of the KV pool, not the context: the
  context is capped by the attention envelope, and the pool had about 1.4 GB
  past it. With `--reuse-prompt off` they are not reserved. The 16 GB
  profiles keep none, since their context is set by memory.
- **The pool past the context**, 91,200 tokens or about 0.84 GB once the
  device slots and the dense buffer have theirs, is room for what a real
  card's desktop holds;
  past that the load adapts
  ([when a profile does not fit](#when-a-profile-does-not-fit)).
- **No wider chunk while no lane decodes.** The
  [solo width](../configuration/context-and-speculation.md#the-solo-width) is
  pinned to the 2,048-token chunk, so those 91,200 tokens stay the desktop's.
  In the same 24 GiB simulation `auto` would have taken 4,096 for 13,312 of
  them (8,192 does not fit); `--model-prefill-chunk-solo auto` asks for it.

## When a profile does not fit

A profile's values, and the built-in defaults, are what the 27B asks for, not
what it insists on. When the VRAM plan cannot hold them in the memory the
start finds free, the load lowers them one at a time and plans again: the
drafter's head first, then the context in steps, then the lanes. A
[solo width](../configuration/context-and-speculation.md#the-solo-width) is
settled before any of them: `auto` takes a wider one only where nothing else
has to give, and one you named drops to the prefill chunk first. A value you
name yourself never adapts, and the `ignis.runtime.vram_plan` log event lists
every step taken ([Configuration](../configuration/index.md)); so does
`GET /v1/models`, in `adapted`, beside the context and the decode lanes the
load runs ([the API](../api/index.md)).

## Defining your own

A config file's `profiles:` section holds profiles in the file's own shape, one
per name, and `profile:` selects one:

```yaml
profile: workstation
profiles:
  workstation:
    vram:
      headroom_bytes: 2G
    model:
      qwen38:
        max_context: 262144
    spec:
      qwen38:
        backend: dflash2
        draft_tokens: 7
```

A built-in profile is only the default contents of the same mechanism, so one you
define resolves exactly as they do. A profile in the file with a built-in's name
replaces it for that file. A good profile is found the way the built-in ones
were: start from a candidate context, raise it until the load is refused, back
off, and write down what you measured
([the VRAM plan](vram-plan.md)).

`ignis-server config print` shows what a start would run, `ignis-server help
--fields` lists every field a profile can set, and the Server view of the
Playground edits the running configuration
([the Playground](../playground/index.md)).
