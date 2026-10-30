---
title: "Configuration"
description: "Fields, groups, flags, env vars and the config file; precedence, per-model scopes, profiles and live changes."
sidebar:
  order: 1
---

Every setting is one **field** in a **group** — `server`, `model`, `vram`,
`reuse`, `switch`, `spec`, `vision`, `media`, `image`, `speech`, `download`, `ngram`, `kv_disk` —
and has the same name on every surface
([ADR 0046](../../adr/0046-one-declaration-per-config-field.md)):

| Surface | Spelling | Example |
|---|---|---|
| flag | `--<group>-<field>` | `--reuse-kv-host-pool-bytes 8G` |
| env var | `IGNIS_<GROUP>_<FIELD>` | `IGNIS_REUSE_KV_HOST_POOL_BYTES=8G` |
| config file | `<group>: { <field>: … }` | `reuse: { kv_host_pool_bytes: 8G }` |

`ignis-server help --fields` prints every field — flag, env var, file key,
default, rule, the models that take it — from the table the server itself is
built from; it is the reference, and the tables below add the why. (`--format
json` gives the same list to a tool.) A bool's flag may stand bare
(`--server-ui` is `--server-ui true`) or take its value (`--server-ui false`).

**Precedence**, highest first: flag, env var, config file, profile, built-in
default. An empty value is an unset one.

**Per model.** A field both models take but want sized differently has a
family-scoped variant beside it — `qwen38` for Qwen3.8-27B, `qwen38flashnext`
for Qwen3.8-Flash-Next: `--qwen38flashnext-reuse-kv-host-pool-bytes 1G`,
`IGNIS_QWEN38FLASHNEXT_REUSE_KV_HOST_POOL_BYTES`, or in the file
`reuse: { qwen38flashnext: { kv_host_pool_bytes: 1G } }`. Within each source
the family's value wins over the general one, and a model switch takes the
target model's own. A field only one model has (`--vision-enabled`, the
Flash-Next knobs) is refused at start when set for the other model, and
dropped with a log line when a switch moves to the other model.

**The config file.** JSON or YAML, named by `--config <path>` /
`IGNIS_CONFIG` (it must exist). With neither, a start looks for
`./ignis.config.yaml`, `./ignis.config.json`, then `%APPDATA%\ignis\config.yaml`
(Windows) or `$XDG_CONFIG_HOME/ignis/config.yaml` / `~/.config/ignis/config.yaml`
(Linux); the first found is used. The startup line `ignis.config.source` says
which (`explicit`, `discovered` or `none`). An unknown key or a bad value in it
refuses the start naming the key.

```sh
ignis-server config generate --out ignis.config.yaml   # every default, plus any flags given
ignis-server config print                              # what would run: file + env + flags
ignis-server config patch --model-max-context 131072   # change the file in place
```

**Profiles.** `--profile <name>` / `IGNIS_PROFILE` / the file's `profile:`
names a bundle of hardware-shaped defaults, below the file and above the
built-in defaults. Three are built in, and the default, `--profile auto`,
picks one by the card's total memory as NVML reports it: under 20 GiB
(a 16 GB card) `rtx50-16gb`, 28 to 40 GiB (a 5090) `rtx5090`, 88 to 104 GiB
(an RTX PRO 6000) `rtxpro6000`. A card of any other size, or a total NVML
cannot read, is refused, naming `--profile` — at a start, and at `config
generate|print|patch` and `model list|download` too, which resolve exactly as a
start would, except that the `model` commands need no card: to them a total
NVML cannot read (no NVIDIA driver) is no card, as on a build without CUDA; a
profile you name (`--profile`, `IGNIS_PROFILE`, the
file's `profile:`, `none` included) always wins and the card is not consulted
for the choice. The start's log line `ignis.config.source` names the profile
chosen and the total it was chosen by; `GET /v1/config` and the Playground show
the profile it resolved to. A build without CUDA has no card, so `auto` there is
`none`.
- `rtx5090` holds numbers measured on that card. Under it
  Flash-Next pins its calibration-hottest 2 GiB of experts in VRAM
  (`--vram-pinned-expert-bytes 2G`, [the pinned hot set](host-ram.md#host-ram-for-flash-next)):
  2.1 GB less RAM for at most ~1% of three-lane decode. `--vram-pinned-expert-bytes 0`
  turns it off. The 27B prefills 2,048 tokens a chunk (`--model-prefill-chunk`):
  with vision on that costs no VRAM, and long prompts prefill ~2% faster
  ([finding](../../findings/2026-10-21-the-27b-prefill-chunk-at-2048.md)).
- `rtxpro6000` is the RTX PRO 6000 Blackwell series (Workstation, Max-Q,
  Server Edition, 96 GB each), measured on a Server Edition under Linux
  ([finding](../../findings/2026-10-18-rtxpro6000-profile-measured.md)). Both
  models get 1,048,576 tokens of context (both with YaRN at 4, which `auto` rope scaling picks for them). Flash-Next gets 8 decode
  lanes, MTP speculation with 3 draft tokens
  ([Flash-Next speculation (MTP)](flash-next-mtp.md)) and `--ngram-hot-bytes auto`: the
  whole n-gram table in RAM when the host plan leaves room for it
  ([Flash-Next n-gram startup cache](ngram-cache.md)). Everything else is as on the
  5090, except that the 27B's VRAM headroom stays at its default. `--profile
  auto` picks it on this card; `--profile rtxpro6000` names it.
- `rtx50-16gb` is the 16 GB class (RTX 5060 Ti 16 GB, 5070 Ti, 5080): derived,
  not measured, since no 16 GB card exists in the project. The 27B at
  262,144 tokens of context on two decode lanes, with DFlash2 at 7 draft
  tokens and the drafter's head `auto`, vision on at 4,096 tokens an image (the tower and pool in host RAM, so
  no context lost while no image is in flight), the VRAM headroom of the
  5090's 27B (1200M) and a 1,024-token prefill chunk.
  It names no model: a start with no artifact takes the catalog's 16 GB entry
  of the default id.

**Adaptation.** A profile's values, and the built-in defaults, are what the
27B asks for, not what it insists on: when the VRAM plan cannot hold them in
the memory the start finds free, the load lowers them, in this order, and
plans again after each step:

1. the drafter's head: a `shortlist` from a profile falls to `full` (the
   shortlist is bound beside the full head and costs ~357 MB);
2. `--model-max-context`, down `--model-adapt-context-step` tokens at a time
   (65,536) to one step, keeping the largest context at which the KV pool
   holds one whole sequence and a spare reservation of
   `--vram-adapt-spare-tokens` (65,536) for a second request (at one lane,
   the sequence alone);
3. `--spec-decode-lanes` to 1, after which the context steps down again from
   the profile's value.

A start that fits as configured changes nothing, even when its pool has no
room for a second request: a budget one byte smaller can therefore run a
smaller context that has that room. A value you name (a flag, an env var, the
config file, a live patch -- even one equal to the default) never adapts. With
the lanes named, a context with room for the spare reservation still wins
first; only when none has it does the load keep the largest context the plan
holds, and the log says `one sequence only`. The `ignis.runtime.vram_plan` log
event lists every step taken in `adapted`, empty when none was -- on the 5090
under `rtx5090`, a 23.3 GB budget gave `decode_lanes 8 -> 1, max_context
917504 -> 327680 (adapt_context_step 65536, one lane)`. `GET /v1/models`
reports the same steps in `adapted`, beside the context the load runs
(`max_model_len`) and its decode lanes (`decode_lanes`), while `GET /v1/config`
keeps the values configured. When not even one step's context on one lane
fits, the start is refused with that plan's deficit and every step it tried.
DFlash2 is never switched off; a budget named with
`--vram-allow-oversubscription` never adapts, nor does a named
`--vram-kv-pool-bytes` below one sequence (refused by name, as before).
Flash-Next does not adapt: its expert cache takes what the plan leaves. `make`
leaves the 27B's context to the profile, so it adapts; `MAX_CONTEXT=n` names
one, which does not.

A file's `profiles: { <name>: { <group>: { … } } }` defines one for any other
card. `auto` picks the class's profile by name, the lookup a named profile
goes through, so a file whose `profiles:` redefines `rtx50-16gb`, `rtx5090` or
`rtxpro6000` replaces what `auto` picks for that class. A profile called `none`
or `auto` is refused when the file is read: both words mean something to
`--profile`, so such a profile could never be reached.

`config generate` writes every resolved value as a value of the FILE, the
profile's among them: a file generated under `auto` carries the numbers of the
card class it was generated on, and a value in a file never adapts to the card
that reads it. Generate with `--profile none` for a file that does not depend on
the card.

**At runtime.** `GET /v1/config` shows the running configuration (fields
declared visible only: never the API key); `GET /v1/config?format=yaml` shows
it as the config file it amounts to, the text `config print` writes.
Both are in the file's shape: a group key holds the shared value, and a model
family's section (the loaded model's too) sits beside it wherever that family's
value differs, so what the loaded model runs with is its section's, else the
group's.
`PATCH /v1/config` takes a partial document in the file's shape: a change to
the deadlines, the message policies, the thinking defaults or the switch
knobs applies at once, and so does a value the loaded model does not run with
— one in the other model's family section, or a group value the loaded model
overrides in its own — which that model uses when it next loads; a change to
anything the loaded model's load is built from reloads it with the whole
patch, as a model switch does. A `null` removes a key: in a family's section,
so the group's value applies to that model again; at a group key, so the
profile's or the built-in default applies — a flag or an env var that sets the
field still outranks the file, and the answer names it. A field only one
model takes (`spec.draft_rows`) is set and removed in that model's section,
whichever key names it. Turning speculation off takes one step: a patch
with `spec.backend: off` that names no `spec.draft_tokens` or
`spec.draft_head` beside it removes the ones the file or an earlier patch holds
at that same key, as if it said `null` for each, so
`{"spec":{"qwen38flashnext":{"backend":"off"}}}` is enough; naming a count
beside `off` in the same patch is still refused, and so is a count a flag or an
env var sets (a patch cannot remove those). Either way it is
written to the config file in use once it has taken, a removed key and the
section it empties taken out. `GET /v1/config/fields` says what each of those fields
is: its key, kind, default, rule, the model families that take it, and whether
a change is live, needs a reload, or can only be made at start — the same
declarations as `ignis-server help --fields --format json`, minus the fields
`GET /v1/config` hides. `GET /v1/config/origins` says where each value in force
comes from, for the shared value and for each model family: the default, the
profile, the file, an env var, a flag or a live change, whether it was set for
that family itself, and the flag, env var, file key or profile that set it.
A family's own value in the file or from a live change is the one a group
change does not reach and a `null` removes.
