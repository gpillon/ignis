---
status: accepted
amended-by: [ADR-0075, ADR-0079]
---

# ADR 0063 — the profile follows the card class, and its defaults adapt to the budget

## Current decision

`--profile` defaults to `auto`, which picks the built-in profile by the card's total memory alone: `rtx-16gb` for the
16 GB class (`rtx50-16gb` is a permanent alias), `rtx-24gb` for the 24 GB class, 20 GiB up to 28 GiB (the RTX 3090
and RTX 4090), `rtx5090` at ~32 GB and `rtxpro6000` at ~96 GB. Any other total is refused, naming `--profile`, and a
named profile always wins. The GPU generation is a second axis beside the class
([ADR 0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md)): RTX 30 and RTX 40 cards are in
scope beside SM120, which the Status below excluded.

At load, `auto` first picks the 27B's solo width on an RTX 30 or RTX 40 card: the widest of its ladder the plan holds
with nothing else adapted and room for a second request (GitHub #489, ADR 0018). Then, only when the plan is refused
below its minimum, a profile's defaults adapt in a fixed order, as built (2026-10-23, GitHub #336): the drafter's
head, whose `auto` is the full head on every class, so this step never fires on a built-in profile; the context, down
by `model.adapt_context_step`, keeping the largest whose KV pool holds one full sequence and
`vram.adapt_spare_tokens`; then `decode_lanes` 2 → 1 and the context ladder again. A value the operator names never
adapts, except a solo width the plan refuses, which narrows to the prefill chunk. Each step is named in
`ignis.runtime.vram_plan`'s `adapted`.

`rtx-16gb` serves the 27B's 16 GB entry at 262,144 tokens on two lanes, with DFlash2, and vision on at a 4,096-token
envelope under `host` residency
([ADR 0075](0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md));
`rtx5090` and `rtx-24gb` take `host`, `rtxpro6000` `device`. The 16 GB and 24 GB profiles pin the 27B's solo width to
their prefill chunk, 1,024 and 2,048.

## History

### Status

Accepted (2026-10-21, after the owner's review: "accetto tutto"). Proposed the same day. Owner decisions in the 16 GB grill:
- the profile is `auto` by card class;
- one 16 GB profile in phase 1;
- per-card profiles only if real-card measurements justify them;
- no display detection: "con i parametri di default si adatterà al caso migliore, come fa adesso espandendo il pool
  KV, o con gli esperti";
- architectures other than SM120 are out of scope.

The adaptation order and its 65,536-token floor (point 3), and the `rtx50-16gb` name, are the agent's proposals. The
owner approved them as a whole ("ok tutto") without changing them.

**Amended 2026-10-29 (owner, GitHub #407):** `rtx50-16gb` keeps vision off and sets `vision.max_tokens` to 1,024, the
envelope a start that turns vision on takes. On a 5060 Ti (Linux, reported on #407 and #360) `--vision-enabled` at
the 32,768 default adapted the context to 65,536, and at 1,024 kept all 262,144 at a KV pool of 275,456 tokens against
308,672 text-only. Vision stays off so a text-only start, and a Windows card whose desktop already costs it context,
pays nothing for it.

**Amended 2026-10-30 (owner, GitHub #409, [ADR 0075](0075-the-vision-tower-is-staged-and-the-embedding-pool-host-mapped-unless-the-load-asks-for-the-device.md)):**
once the vision residency is built and passes its gate, `rtx50-16gb` turns vision on with `vision.max_tokens` 4,096,
since under `host` residency vision costs it no context while no image is in flight. `rtx5090` takes `host` and
`rtxpro6000` `device`.

**Amended by [ADR 0079](0079-a-catalog-entry-states-the-oldest-gpu-generation-it-loads-on.md)** (spec #474, GitHub
#476, #480): a 24 GB card class, the RTX 3090 and RTX 4090, from 20 GiB up to but not including 28 GiB, which `auto`
starts under a new built-in profile, `rtx-24gb`; the 16 GB profile is named `rtx-16gb`, with `rtx50-16gb` kept as a
permanent alias. The class stays memory alone, and the GPU generation is an axis beside it; the RTX 30 (Ampere,
compute capability 8.6) and RTX 40 (Ada, 8.9) cards are now in scope beside SM120.

**Amended 2026-11-06 (GitHub #489, [ADR 0018](0018-chunk-level-prefill-decode-interleaving.md) as amended the same
day):** the solo width comes before point 3's steps.
- At `auto`, the default, the load first plans at the profile's values with each width of the solo ladder in turn,
  8192, 4096 and 2048 above the prefill chunk, for the 27B on an RTX 30 or RTX 40 card. It keeps the first the plan
  fits there with nothing else adapted for it and, at more than one lane, the KV pool still holding one full sequence
  and `vram.adapt_spare_tokens` beside it: the criterion an adapted context is kept by, so room for a second request
  comes before a wider chunk. A width it passes over is no step and is not named in `adapted`. When none is kept, the
  solo width is the prefill chunk and the steps run as before. The ladder is one function
  (`config::arch::solo_ladder`); on Blackwell and for Flash-Next it is empty.
- A width the operator names is the one exception to "an explicit value never adapts": when the plan refuses it
  below its minimum, its drop to the prefill chunk is the first step (`prefill_chunk_solo 8192 -> 2048`), and the
  steps follow at the prefill chunk. It only ever narrows a chunk, never a context or a lane. Flash-Next adapts
  nothing, its solo width included.
- A named width narrower than a prefill chunk also named is refused naming both; narrower than a chunk a profile or
  the default gave (Flash-Next's 8192 under `rtx5090`, for a width named for every model), it is that chunk.
- The width the load runs is logged (`ignis.runtime.prefill_chunk_solo`: `prefill_chunk_solo auto -> 8192`) and
  reported by `GET /v1/models` (`prefill_chunk_solo`); `GET /v1/config` keeps the setting.
- `rtx-16gb` and `rtx-24gb` pin the 27B's solo width to their chunk, 1024 and 2048 (owner, 2026-11-06): on those
  classes the KV pool comes before a wider chunk. On the RTX 5090 held to their budgets `auto` would keep 2048 on
  the 16 GB class (33,280 tokens of pool) and 4096 on the 24 GB one (13,312). The Blackwell profiles name none. `auto`
  stays the default for the 8.x cards no profile pins.
- The profiles set Flash-Next's prefill chunk: 8192 under `rtx5090` and `rtxpro6000`, measured; 4096 under
  `rtx-24gb` and 2048 under `rtx-16gb`, placeholders until Flash-Next runs on those cards (GitHub #526).

Related:
- [ADR 0030](0030-device-memory-reserved-at-load.md): device memory is reserved at load;
- [ADR 0046](0046-one-declaration-per-config-field.md): one declaration per config field;
- [ADR 0057](0057-rope-scaling-is-auto-by-default.md): `auto` by default, as for rope;
- [ADR 0060](0060-the-27b-has-a-trellis-body-for-small-cards.md): the 16 GB body;
- [ADR 0064](0064-a-catalog-entry-states-its-card-class.md): the catalog entry's card class.

### Context

Today the profile is chosen by name only (`--profile`, `IGNIS_PROFILE`, the config file), and its default is
`rtx5090`. Nothing reads the card:
- the total memory NVML reports is thrown away;
- there is no GPU name and no compute capability.

So a 16 GB card would start under the 5090's profile, without a word.

The budget a load gets is already the free memory at start minus the headroom: whatever the desktop holds is counted.
After that:
- the KV pool takes what is left (27B);
- the expert cache takes what is left (Flash-Next).

`max_context`, the lanes and speculation are fixed, so a short budget is refused with the byte deficit. On a 16 GB card
the free memory at start varies with what else holds the card, and 256K with DFlash2 fits only when little else does.

### Decision

1. **`--profile` defaults to `auto`.**
   - It picks the built-in profile by the card's total memory: ~32 GB → `rtx5090`, ~96 GB → `rtxpro6000`, ~16 GB → the
     new `rtx50-16gb`.
   - On any other total the start is refused, with a message that names `--profile`.
   - A named profile always wins, `none` included.
   - On today's cards `auto` resolves to today's profiles.
2. **`rtx50-16gb`** defaults to:
   - the 27B's 16 GB catalog entry (ADR 0064);
   - `decode_lanes` 2;
   - `max_context` 262,144;
   - DFlash2 on, at the family's default draft tokens (to be tuned on a real card: speculation yields less on a
     compute-poor card);
   - the drafter's proposal head `auto`;
   - vision off, with a 1,024-token envelope for a start that turns it on (amended 2026-10-29); vision on at 4,096
     under `host` residency once ADR 0075 is built (amended 2026-10-30).
   - `decode_lanes` becomes one field for both families, 1..8.
3. **A profile's defaults adapt to the budget at load, in a fixed order.** An explicit value from any source never
   adapts and is refused as today. When the plan does not fit, the load tries, in turn:
   1. the drafter's proposal head as its `auto` rule chooses;
   2. `max_context` down in steps, 262,144 → 196,608 → 131,072 → 65,536. It keeps the largest at which one full
      sequence and one more 64K reservation fit the pool;
   3. `decode_lanes` 2 → 1.
   - DFlash2 is never switched off by adaptation.
   - Below 65,536 the start is refused with the deficit, as today.
   - Each step taken is named in the VRAM plan's log event (`adapted: max_context 262144 -> 131072`).

**As built** (2026-10-23, GitHub #336), adaptation and the drafter's head, where the decision left a choice or rested
on a wrong premise:

- **`draft_head: auto` is the full head on every card class.** Spec 04 proposed `shortlist` "when the plan needs the
  bytes it frees, or on the 16 GB class with the pool short of two full reservations". It frees none: the shortlist
  head is bound beside the full head (`ProposalHead::Shortlist`, "costs its ~357 MB"). The measured plans: text +
  DFlash2 18,289,864,964 B, with the shortlist 18,646,905,344 B (+357,040,380). On the one card that measured it, it
  was slower too: −7.4% tokens/round, −5.5% tok/s (finding 2026-09-24-upstream-quick-wins-ab). Both clauses of the
  rule were about bytes, so with the facts right `auto` is `full`. The rule is revisited in phase 2, when a real
  16 GB card measures the trade.
  - `auto` is a third spelling of the field (`full | shortlist | auto`), not a third head:
    `ignis_core::ProposalHeadSetting` beside `ProposalHead`, as `RopeSetting` is beside `RopeScaling` (ADR 0057).
  - It resolves where the load's speculation is built (`derive_speculation`). `GET /v1/config` keeps `auto`.
  - It is valid with `dflash2` and with `mtp`, whose only head is `full`. Without a backend it is refused, as any head
    is.
  - A load whose head is `auto` logs `ignis.runtime.draft_head`: `draft_head auto -> full`, once, where the load
    derives its shape from the resolution (`load::load_shape`). A CPU test reads it through that function, in its
    own test binary (`tests/draft_head_auto_log.rs`); the GPU leg reads it from a load through `load::load_model`.
- **Step 1 drops a `shortlist` only.** A head the profile or the default gave falls to `full` when it is `shortlist`.
  Since `auto` is `full`, it never fires on today's built-in profiles; a test profile that sets `shortlist`
  exercises it.
- **The context ladder generalizes §3's.** 262,144 → 196,608 → 131,072 → 65,536 is "down 65,536 at a time". From any
  value `v` the candidates are `v`, then every multiple of the context step (65,536 by default) below it, down to one
  step: 917,504 → 851,968 → … on the 5090, 1,048,576 → … on the RTX PRO 6000. The owner, 2026-10-21: with `auto` the
  5090's context finds the largest that fits.
- **The criterion, at more than one lane:** a context `c` is kept when the plan fits and its KV pool holds `c` plus
  the spare reservation (65,536 tokens by default): one full sequence and room for a second request. The plan's own
  floor, a tail page per retained slot, still applies. At one lane a spare reservation has no lane to serve, so the
  plan's floor alone decides.
- **Only a plan refused below its minimum adapts** (`VramPlanError::BelowMinimum`).
  - A start whose plan fits as configured takes no step, and its `adapted` is empty. On the 5090, a start that fits
    today is byte-identical.
  - That holds whatever room the pool leaves: the criterion is an adapted context's, not the configured one's
    (decision 5). So a budget that just holds the configured context keeps it with no room for a second request,
    while a budget one byte smaller is refused, adapts, and can run a smaller context that has the room.
  - A budget above free memory is refused as before. So is a named `--vram-kv-pool-bytes` below one sequence and its
    tail pages (`VramPlanError::PoolBelowFloor`): no adaptation lowers a pool the operator named, and the refusal
    names it.
  - A budget with `--vram-allow-oversubscription` never refuses, so it never adapts.
- **Step 3 runs the ladder again.** When the ladder reaches 65,536 without meeting the criterion, the lanes drop to 1
  (if not explicit and above 1). The ladder then runs again from the profile's value, against the one-lane
  criterion. The lines that follow the lanes are planned again at one lane. So is the 27B's default of host
  retained slots (two a lane), unless the operator named it.
  - When the lanes cannot drop, the load keeps the largest context the plan itself holds, and the step says so:
    `(…, one sequence only: no room for adapt_spare_tokens N, and decode_lanes is explicit)`. A context that meets
    the criterion still wins first. With the default fields that fallback is the floor whenever it happens (above
    262,144 tokens the pool is the same at every step, below it the pool grows as the context falls); a spare
    reservation larger than the step can leave it higher.
- **The refusal** carries the last, smallest attempt's deficit, in the plan's own words. After it come every
  adaptation tried, in order: `; adaptation tried, in order: draft_head shortlist -> full, max_context 262144 ->
  196608 -> 131072 -> 65536, decode_lanes 2 -> 1, max_context 262144 -> …`. With nothing adaptable it reads exactly
  as before.
- **The 27B only.** Flash-Next's plan sizes an expert cache from the rest; its 16 GB story is phase 2.
- **The step and the spare reservation are config fields** (the review of #336 replaced decision 9's constants: a
  threshold that decides whether or how a load starts is a field, its value the default). Both are the 27B's alone (a
  Flash-Next start refuses them named), visible, patchable with a reload:
  - `model.adapt_context_step` (`--model-adapt-context-step`, `IGNIS_MODEL_ADAPT_CONTEXT_STEP`), default 65,536: the
    ladder's step and its floor. A nonzero multiple of 64, a whole number of KV pages. A start whose plan does not
    hold one step is refused.
  - `vram.adapt_spare_tokens` (`--vram-adapt-spare-tokens`, `IGNIS_VRAM_ADAPT_SPARE_TOKENS`), default 65,536: the
    spare reservation; 0, one sequence is enough.
  - `adaptation::adapt` and `adaptation::contexts` take them as inputs (`adaptation::Ladder`, from
    `Config::adapt_ladder`, carried as `EngineShape::adapt_ladder`).
  - At their defaults neither refuses a start that code without adaptation accepted: each such start was refused
    already.
- **Where it lives.**
  - The ladder is a pure function, `adaptation::adapt`. Its inputs are the values, which of them may adapt, and a
    closure that plans at an attempt.
  - The closure lives in the planner's caller, `runtime::qwen27b_load`. At each attempt it plans the leaf's lines
    again: the hq prompt planes follow the context up to 262,144 keys, the rope table, the lanes, and the head's
    bound scope. It then calls `plan_vram`, unchanged.
  - Which values may adapt is decided by source alone (`Resolution::adapts`, `Config::adaptable`, carried as
    `EngineShape::adaptable`). The profile's and the hardcoded default's adapt. A flag, env var, file value or live
    patch never does, a file's value equal to the default included: a file `config generate` wrote names the
    numbers of the card it was generated on. That differs on purpose from ADR 0046's `explicit`, where a value equal
    to the default counts as unnamed.
  - Only the resolution sets it (`EngineShape::from(&Config)`). A shape built by hand -- `EngineShape::default()`,
    every GPU gate's -- adapts nothing (`Adaptable::NONE`) and names its rope table, so a gate the card cannot hold
    is refused as before, never served at a context it did not ask for.
- **The values the load kept are the load's everywhere after the plan.**
  - They reach the leaf, the scheduler (admission, `max_model_len` in `GET /v1/models` and the tokenize route, the
    lanes Decide fans out to) and KV-disk's identity (its rope table).
  - An `auto` rope table is resolved again for an adapted context (`EngineShape::at`, from the `RopeSetting` the
    shape carries); a spelled-out table stays.
  - The vision processor is built before the load at the configured context, to check the model contract, and the
    load serves it at the context it runs (`load::serve_vision`, `media::processor_at`): its request budget and item
    bound are the adapted envelope. The shape's encoder item bound follows too (`EngineShape::at`); the leaf always
    capped the encoder's workspace at the context.
  - `ignis.model.loaded` names the kept values, and `/metrics` the adapted plan's lines.
  - `GET /v1/config` keeps the values configured.
- **`ignis.runtime.vram_plan` gains `adapted`**: the steps taken, in order, comma-separated; empty when none. A
  context step names the fields it was taken by and the criterion that kept it: `max_context 917504 -> 786432
  (adapt_context_step 65536, adapt_spare_tokens 65536)`, at one lane `(adapt_context_step 65536, one lane)`.
- **`make` leaves the 27B's context to the profile** (`MAX_CONTEXT` empty in `mk/config.mk`): a `make start` on a
  5090 whose free memory cannot hold 917,504 tokens adapts the context rather than dropping to one lane, as the owner
  asked on 2026-10-21. `MAX_CONTEXT=n` names one, which never adapts.
- **`rtx50-16gb`** gains `spec.qwen38.decode_lanes: 2` and `spec.qwen38.draft_head: auto`, scoped to the 27B. The
  other profiles leave the head unset, which is `full`.
- **Measured on the 5090** (`crates/server/tests/adaptation_gpu.rs`, explicit budgets, 2026-10-23). Each budget was
  derived from the plan's own refusal at 1 GiB, whose `BelowMinimum` carries what the plan needs.
  - Under `rtx5090` the 917,504-token plan needs 30,607,308,032 bytes, more than the 30.41 GB free beside a 2.6 GB
    desktop.
  - A budget 100 MB below that loads at `max_context 917504 -> 786432 (adapt_context_step 65536,
    adapt_spare_tokens 65536)`, serves `max_model_len` 786,432 and decodes. The first run (30,143,774,720 bytes, the
    free memory's cap) kept a pool of 868,224 tokens, the review's re-run (30,507,308,032) 907,648.
  - A 23,349,097,728-byte budget, where no eight-lane context keeps a spare reservation, loads at `decode_lanes 8 ->
    1, max_context 917504 -> 327680 (adapt_context_step 65536, one lane)` with 2 host retained slots (pool 334,528
    tokens) and decodes.
  - The 27B with 2 and with 1 lane named, at 262,144 tokens inside a budget 256 MB above its plan, loads with
    `adapted` empty and decodes.
  - Under `rtx50-16gb`, through the server's load path, the 27B (text only, DFlash2 at 7) needs 20,991,684,452 bytes
    at 262,144 tokens on two lanes (weights line 16,939,004,164). A budget 256 MiB above it loads with `adapted`
    empty, two lanes, 4 host retained slots and a pool of 291,520 tokens, logs `draft_head auto -> full`, and decodes.
  - The spec's 17,179,869,184-byte budget (16 GiB) cannot be used: it is below today's NVFP4 27B's weights line alone,
    17,234,723,840 bytes. The 16 GB body is #337/#338.

### Considered options

- **Detect whether the card drives a display** (NVML `display_active`) and choose the context from it. Rejected by
  the owner. The free memory at start already says what the desktop holds, and adaptation answers it.
- **A profile per 16 GB card** (5060 Ti, 5070 Ti, 5080). Deferred to phase 2. Memory-shaped fields are the same on all
  three; bandwidth-shaped ones (lanes, draft tokens) need a real card to tune.
- **Refuse instead of adapting.** Rejected: a 16 GB card's free memory varies run to run, and refusing what a smaller
  context would serve helps no one.

### Consequences

- A load may run a smaller context than its profile names, and says so in the log.
- The order (drafter head, then context, then lanes) puts speed before context length, on purpose.
- The plan's refusal message must say which adaptations it already tried.
