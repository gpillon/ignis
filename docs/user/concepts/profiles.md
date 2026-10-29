---
title: "Profiles and defaults"
description: "What a hardware profile is, where it sits between the config file and the built-in defaults, what the rtx5090 and rtxpro6000 profiles choose and why, and how to define your own."
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
`rtx50-16gb` under 20 GiB, `rtx5090` from 28 to 40 GiB, `rtxpro6000` from 88
to 104 GiB. The log line `ignis.config.source` names the profile chosen and
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

`rtx5090` and `rtxpro6000` carry numbers measured on the card they name,
with the finding for each choice linked below; `rtx50-16gb` is derived from
them, since no 16 GB card exists in the project.

### `rtx5090`

The default: a 32 GB RTX 5090.

| Field | 27B | Flash-Next |
|---|---|---|
| Context (`model.max_context`) | 917,504 | 589,824 |
| Vision | on | not applicable |
| Speculation | DFlash2, 7 draft tokens | off |
| Decode lanes | 8 (the default) | 3 |
| Prefill chunk | 2,048 | 1,024 (the default) |
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
- **The 27B's prefill chunk.** With vision on, the image encoder sizes the
  workspace, so a 2,048-token chunk costs no VRAM over 1,024. It prefilled
  1.9 and 2.2 percent faster at 46K and 105K, with a lower KLD against BF16 and
  decode unchanged; the cost is that a decoding lane's longest pause while a
  prompt prefills grows by about 65 percent
  ([finding](../../findings/2026-10-21-the-27b-prefill-chunk-at-2048.md)).
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
| Vision | on | not applicable |
| Speculation | DFlash2, 7 draft tokens | MTP, 3 draft tokens |
| Decode lanes | 8 (the default) | 8 |
| Prefill chunk | 2,048 | 1,024 (the default) |
| VRAM headroom | default | default |
| N-gram hot rows | not applicable | `auto` |

Why each:

- **A million tokens of context for both.** It is the 27B's whole compressed-KV
  attention envelope, and the plan holds it. `auto` rope scaling makes both
  models YaRN at 4. No request that long was run.
- **Headroom left at the default.** Nothing has to be traded for vision on 96 GB.
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

### `rtx50-16gb`

The 16 GB class: the RTX 5060 Ti 16 GB, 5070 Ti and 5080. It serves the 27B
from the catalog's 16 GB entry, `qwen3.8-27b-16g`; the profile names no
model, and a start with no artifact resolves the default id `qwen3.8-27b` to
that entry ([Get a model](../guides/models.md)).

| Field | 27B |
|---|---|
| Context | 262,144, the positions it was trained on |
| Vision | off; 1,024 tokens an image when turned on |
| Speculation | DFlash2, 7 draft tokens, the drafter's head `auto` (the full head) |
| Decode lanes | 2 |
| Prefill chunk | 1,024 (the default) |
| VRAM headroom | 1,200 MiB |

Why each:

- **Vision off.** The vision tower's weights and its encoder peak are VRAM a
  16 GB card does not have.
- **1,024 vision tokens when you turn it on.** With `--vision-enabled` the
  profile caps an image at 1,024 merged tokens, about 1,024 x 1,024 pixels
  (a larger one is downscaled). At the 32,768 default the encoder's workspace
  and embedding pool take most of the card's context; at 1,024 a 5060 Ti kept
  all 262,144, at about a tenth of its KV pool. `--vision-max-tokens` sets
  your own.
- **Two lanes.** Eight would leave no room for the context.
- **The 1,024-token chunk.** With vision off the workspace is the text scratch,
  and 2,048 would cost about 0.4 GB of KV pool.
- **The headroom** is the 5090's 27B value until a real card says otherwise
  ([measure yours](../guides/measure-a-16gb-card.md)).

## When a profile does not fit

A profile's values, and the built-in defaults, are what the 27B asks for, not
what it insists on. When the VRAM plan cannot hold them in the memory the
start finds free, the load lowers them one at a time and plans again: the
drafter's head first, then the context in steps, then the lanes. A value you
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
