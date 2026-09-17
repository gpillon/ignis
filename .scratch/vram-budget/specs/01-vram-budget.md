# 01 — Device memory reserved at load within a VRAM budget; render refusals and instruction-message policies

GitHub: #207 (absorbs #204); slices 1 #208, 2 #209, 3 #210, 4 #211, 5 #215, 6 #212, 7 #213, 8 #214 (blocking edges live on GitHub). ADRs: 0030 (this feature), 0029 (clarified by it), 0023
(eviction order the retained slots follow), 0024 (sequence state transfer).
Evidence: `.scratch/vram-analysis/REPORT.md` (live measurement and code analysis),
`.scratch/vram-analysis/research-multi-system-messages.md` (how other engines treat
instruction messages).

## Problem Statement

**Memory.** Under Task Manager, ninfer holds a fixed dedicated and a fixed
"Shared GPU memory" figure. ignis loads at the same size (27.65 GiB at the
Makefile defaults) and then grows. A live run (trace #191 replay plus four
parallel `@agent` conversations) raised the process commit by 3.16 GiB in
14 minutes on a 31.84 GiB card with 1.6 GiB of desktop. Windows then paged
device allocations to system RAM: dedicated and shared traded places at a
constant commit, and TTFT went erratic (#204). Causes, at source:

- Every prefix publish (`kernel/src/seq_prefix.cu:207`) and every checkpoint
  capture (`kernel/src/seq_checkpoint.cu:244-246`) `cudaMalloc`s a
  238,823,424-byte mutable-state image.
  - The retained pool (`crates/core/src/checkpoint.rs:1152`, derived at
    `crates/server/src/runtime.rs:222-239`) is a ledger that charges
    checkpoint images only.
  - Prefix images are charged to nothing, and a KV page shortfall
    (`crates/core/src/concrete.rs:1021-1073`) is the only thing that
    releases them.
- KV-RAM makes one `cudaHostAlloc` per blob (`kernel/src/seq.cu:901`, via
  `crates/core/src/seq.rs:1032`), so the pinned memory Windows reports as
  shared moves with load.
- The vision encoder workspace (2,219,837,184 B) has its own arena beside the
  prefill scratch (~1.34 GB, `kernel/src/model.cu:975-1042`). The two are
  never live together.
- `cudaMemGetInfo` cannot see WDDM paging, and an oversubscribed `cudaMalloc`
  does not fail on Windows.

**Rendering.** qwen-code sends two leading `system` messages: the agent
prompt, then a hook-injected line.
- The Qwen 3.8 template raises "System message must be at the beginning".
  `crates/server/src/artifact_template.rs:180` turns any render failure into
  `RenderedPrompt::default()`: an empty prompt that is admitted with 0
  tokens.
- The decode then fails on a sequence that was never prefilled, and
  `concrete.rs:2974` only records the error. The request stays decoding and
  is retried every tick: 345K `ignis.runtime.leaf_error` lines and 105 MB of
  log in about a minute, with the lane held.
- Every request of the #191 trace hits this.
- Separately, `Role::parse(..).unwrap_or(Role::User)` (`artifact_template.rs:141`)
  silently turns a `developer` message, or any unknown role, into a user
  message.

## Solution

- **Memory: nothing on the device is allocated while serving.**
  - A **VRAM budget** for the whole process, weights included, is either
    derived (memory free at start minus a **VRAM headroom**) or explicit.
  - The load lays out every reservation inside it, prints the plan, and
    refuses to start when the plan does not fit.
  - Retained state lives in **retained slots** reserved at load.
  - KV-RAM is one pinned arena held from start.
  - The vision encoder shares the prefill scratch.
- **Rendering: a prompt that cannot be rendered is refused before admission.**
  - A decode error never loops.
  - `system` and `developer` messages beyond the first each follow an
    explicit, operator-selectable policy whose defaults serve qwen-code and
    OpenAI clients without losing prefix reuse.

## User Stories

1. As the owner, I want Task Manager to show ignis at a fixed dedicated and a fixed shared figure for the life of a load, so that a long agent session never degrades into paging.
2. As the owner, I want ignis started with no memory flags to take what is free minus 1 GiB for my desktop, so that the default just works on my box.
3. As the owner, I want to state exactly how much VRAM ignis may use and have the start refuse when that much is not free, so that ignis never silently pages beside my other GPU software.
4. As the owner, I want to accept paging explicitly for a stated budget and get a warning instead of a refusal, so that I can run oversubscribed on purpose.
5. As the owner, I want the startup log to print the whole device plan (weights, workspaces, lanes, retained slots, KV, headroom, what was free), so that I can see where the VRAM went.
6. As the owner, I want a plan that cannot hold one sequence at `--max-context` refused at start with the shortfall and the knobs that shrink it, so that a misconfiguration fails fast and legibly.
7. As an agent in a tool loop, I want prefix and checkpoint reuse to keep working when retained slots are full, by the oldest retained state giving its slot up, so that a busy engine still reuses.
8. As a live request, I want a full set of retained slots never to delay or refuse me, so that reuse bookkeeping never costs my present.
9. As the owner, I want KV-RAM held whole from start and shown as a fixed shared figure, and the start refused when the host cannot pin it, so that host memory is as predictable as device memory.
10. As the owner, I want `--vision` to cost only what the encoder needs beyond the prefill scratch, so that image input does not eat ~1.25 GiB of KV for nothing.
11. As a qwen-code session, I want a leading run of `system` messages accepted and joined into the system prompt, so that my hook lines do not break requests.
12. As an operator who wants the template's own rules, I want `--system-message-policy strict` to refuse any `system` message that is not first with a 400 naming its index, so that nothing out of distribution reaches the model.
13. As an OpenAI client, I want `developer` messages accepted where I put them by default, and a leading `developer` treated as the system prompt, so that modern SDK requests work and keep prefix reuse.
14. As an operator, I want to choose how `developer` messages are placed or refused, and to be warned at start when my choice re-renders history, so that I trade reuse only knowingly.
15. As any client, I want a request whose messages cannot be rendered answered with a 400 that says why, so that I never get an empty completion or a stuck request.
16. As the owner, I want a decode error to end the affected request with an error instead of retrying it every tick, so that one bad request cannot flood the log and hold a lane.
17. As the owner, I want the memory measurement of 2026-09-17 repeated after the change with a flat commit and a fixed shared figure, so that the fix is proven on the workload that exposed it.

## Implementation Decisions

### Slice 1 — render refusal and no decode hot loop

- **Render failures refuse the request.** `apply_chat_template` returns a
  rendering error instead of a default `RenderedPrompt`. The API answers 400
  `invalid_request_error` carrying the template's message, with code
  `render_failed`. The same applies to a tokenizer encode failure and to the
  multimodal path, which already maps to `invalid_media`: keep that code for
  media and use `render_failed` for a template error.
- **Unknown roles refuse the request.** A role that is not `system`, `user`,
  `assistant`, `tool` or `developer` gets a 400 with code `invalid_role`,
  never a silent `User`. `developer` is handled by slice 2; until slice 2
  lands, it is refused with `invalid_role`.
- **Admission refuses a zero-token prompt** as defence in depth, with the
  same 400 path.
- **A decode leaf error ends the batch's requests with `Error`.** It never
  leaves them decoding, and the same holds for the verify round. Follow the
  #166 prefill precedent: bounded, logged once per request, lane released.
  Record whether a retry is ever meaningful for decode; the default is no
  retry.

### Slice 2 — instruction-message policies

Terms: **instruction message**, **system prompt**, **system message
policy**, **developer message policy** (CONTEXT.md). Every accepted
instruction message reaches the model as `<|im_start|>system\n…<|im_end|>`.
The model has no `developer` role (the Qwen 3.8 template raises "Unexpected
message role."). Rendering happens in Rust before the template sees the
messages: the template receives a message list it accepts, so ignis keeps
running the artifact's own template.

**How messages are grouped and normalized:**
- **Leading run:** the maximal run of `system` messages at index 0. A
  `developer` message interrupts it.
- **Leading developer:** a `developer` at index 0 with no `system` before it
  is the system prompt, in every developer mode except `reject`.
- **Joining** uses `"\n\n"` between trimmed texts. Instruction messages carry
  text only; media in one is refused, as today.

**System message policy** (`--system-message-policy`,
`IGNIS_SYSTEM_MESSAGE_POLICY`, make `SYSTEM_MESSAGE_POLICY`):

| Value | Leading run | A later `system` |
|---|---|---|
| `merge` (default) | joined into the system prompt | its own system block in place |
| `strict` | only index 0 allowed; a second one → 400 `system_message_position` naming the index | 400 `system_message_position` |

**Developer message policy** (`--developer-message-policy`,
`IGNIS_DEVELOPER_MESSAGE_POLICY`, make `DEVELOPER_MESSAGE_POLICY`):

| Value | Rendering | History re-rendered when one arrives mid-conversation |
|---|---|---|
| `inplace` (default) | each is its own system block where it stands | no |
| `into-system` | all joined into the system prompt | **yes**: the head changes |
| `after-system` | all gathered into one system block right after the system prompt | **yes**: everything after the system prompt changes |
| `one-after-system` | exactly one, immediately after the system prompt; any other → 400 `developer_message_position` | no |
| `reject` | any → 400 `developer_message_position` | – |

- **No mode reorders messages across roles.** A `[system, developer,
  system, user]` head keeps its order: the first `system` is the prompt, the
  `developer` follows its policy, and the second `system` counts as "later".
- **The render boundaries follow the normalized list.** The **generation
  opener**, the **system block end** where a **retained prefix** is cut, and
  the last-user-query offset are all computed on it. With `merge`, a leading
  hook line is therefore part of the system block, and a hook line that
  changes invalidates at most the pages from where it starts.
- **Startup warning** for `into-system` and `after-system`: "re-renders
  history when a developer message arrives mid-conversation; prefix reuse is
  lost from that point".
- **Departure from the reference, recorded for the #191 gate.** ninfer
  renders a later system message in place and never joins the leading run.
  With `merge`, ignis tokenizes the hook line's join differently: Qwen merges
  `".\n\n"` into one token. Their token prefixes therefore diverge at the end
  of the first system text.

### Slice 3 — VRAM budget and the load plan

Terms: **VRAM budget**, **VRAM headroom**, **VRAM oversubscription**.

**Flags:**
- `--vram-headroom-bytes <size>` (`IGNIS_VRAM_HEADROOM_BYTES`, make
  `VRAM_HEADROOM`, default 1G) selects the derived mode.
- `--vram-budget-bytes <size>` (`IGNIS_VRAM_BUDGET_BYTES`, make
  `VRAM_BUDGET`) selects the explicit mode.
- Giving both is a configuration error, from any mix of CLI and env.
- `--allow-vram-oversubscription` (`IGNIS_ALLOW_VRAM_OVERSUBSCRIPTION`, make
  `ALLOW_VRAM_OVERSUBSCRIPTION`) is valid only with `--vram-budget-bytes`;
  alone, it is a configuration error.
- Sizes parse like the other `*-bytes` flags (`config.rs` `parse_bytes`).

**The budget is the whole process footprint.** Free memory is read at start,
before the weights. The CUDA context is either measured by reading free
memory before and after context creation or accounted in the plan. The
derived budget is `free − headroom`.
- **Explicit mode, `budget > free`:** refuse the start, or warn and proceed
  with the oversubscription flag. The warning says whether this system can
  page (Windows WDDM) or will fail the allocation.

**The plan**, computed before the first large allocation:
1. weights (the artifact's size);
2. workspace = `max(prefill scratch, vision encoder workspace)` (slice 6) +
   media embedding + sampling + decode-graph scratch + verify round +
   drafter round;
3. every lane's state: `N_DECODE_LANES` × the sequence-pool slot size,
   GDN + conv + penalty counts + drafter window and checkpoint;
4. retained slots (slices 4–5; until then, the current checkpoint budget as one line);
5. KV pool = the rest, or `--kv-pool-bytes` when given.

- **Minimum:** the KV of one sequence at `--max-context`. When the minimum
  does not fit, refuse, naming the shortfall and the knobs (`--max-context`,
  `--vision` off, `--retained-slots`, headroom). With the oversubscription
  flag it only warns.
- **Log:** one `ignis.runtime.vram_plan` event carrying every line above in
  bytes, plus `mode`, `free_at_start_bytes`, `headroom_bytes` or
  `budget_bytes`, and `oversubscribed`. It replaces
  `ignis.runtime.retained_pool`; `ignis.runtime.kv_pool` stays.
- **Measure the residual.** About 250–290 MiB of the 27.65 GiB load is
  unexplained (graph capture, allocator rounding, lazy kernel modules).
  Measure it once, then plan for it as an explicit line, so the planned total
  matches what Task Manager shows within a stated tolerance.

### Slice 4 — prefactor: retained slots in the pool, device allocation counter

A prefactor that makes slice 5 easy. It has no behaviour change for requests.

- **The sequence pool gains retained slots** beside the lane slots:
  `slot_count = N_DECODE_LANES + retained_slots`.
  - Lane counts and decode/verify widths stay at `N_DECODE_LANES`.
  - Never write the literal 8: use the compile-time constant, and keep it
    findable, because the owner intends to replace slots-in-the-pool with a
    dedicated structure later (ADR 0030).
  - An allocator hands retained slots out and takes them back.
  - The count is an internal load option here, reserved in the plan
    (slice 3).
- **A device allocation counter in the leaf, readable by tests.**
  - It counts device and pinned-host allocations and frees after load.
  - A GPU test records today's non-zero counts per event kind (prefix
    publish, checkpoint capture, KV-RAM blob) over a fixed request mix.
  - Slices 5, 6 and 7 flip those counts to zero. This is how "serving
    allocates nothing" becomes a test rather than a belief.

### Slice 5 — retained state in retained slots

Term: **retained slot**. Builds on slice 4's pool slots and counter.

- **Flag:** `--retained-slots <n>` (`IGNIS_RETAINED_SLOTS`, make
  `RETAINED_SLOTS`), default `N_DECODE_LANES`. With `--prompt-reuse off` it
  defaults to 0. `--retained-pool-bytes` is removed and its use is a
  configuration error pointing to `--retained-slots`.
- **What a slot holds:**
  - A prefix publish and a checkpoint capture take a slot instead of
    `cudaMalloc`. The images are the slot's GDN + conv + penalty +
    drafter sections, copied with the pool's existing slot copy.
  - A checkpoint's partial tail page (589,824 B) takes one KV page from the
    pool instead of a `cudaMalloc`.
  - A shared prefix still claimed by live siblings holds its slot until its
    last reference drops; slots are refcounted like the prefix.
- **When no slot is free:**
  - Ask retained state for one, in ADR 0023 order: checkpoints before
    retained prefixes, lowest class, LRU. The victim spills to KV-RAM exactly
    as `reclaim_retained_until` does today.
  - A slot held by a claimed prefix, or by a chain link under one, is not a
    victim.
  - When nothing gives one up, the publish or capture is skipped: the request
    runs, shares nothing further and leaves no reuse. Never wait, never
    refuse.
- **One accounting.** The checkpoint byte ledger disappears, and the slot
  count is the only bound. `request_done` keeps `reuse_source` and
  `reused_prompt_tokens`. Add a log/metric for `retained_slots_in_use` and
  for skips (`publish_skipped_no_slot`, `capture_skipped_no_slot`).
- **Chain accounting:** every link of a chained prefix is one slot, so a
  checkpoint and the prefix under it cost two slots. Tests pin that.

**As built (#215), where it departs from or sharpens the above:**

- **The tail page is charged exactly, and the plan pays for it.** A
  retained checkpoint's tail page counts as a used KV page for as long as its
  device image lives. It is still retained state:
  - The page path gives a checkpoint up for its tail page even while a live
    request stands on the prefix below it, because that page is the
    checkpoint's alone.
  - The only exception is a checkpoint an admitted request is about to be
    built from. Its claimant cannot take that page back, so a pool of exactly
    one context would leave a full-context claimant waiting forever.
  - The plan's KV minimum is therefore `pages(max_context) + retained_slots`,
    and the error names `--retained-slots`.
  - A capture with no page for its tail is skipped with a third reason,
    `capture_skipped_no_page`.
  - An opener on a page boundary takes no tail page.
- **A slot comes back when the backend handle drops.** The scheduler's slot
  ledger is keyed like the handles. The kernel keeps an occupancy bit per
  retained slot. It refuses a publish or capture into a held or out-of-range
  slot, and it refuses a claim on a prefix whose handle is gone. A shared
  prefix therefore holds its slot until its last reference goes. That is the
  refcount above, with no second count.
- **Victims on the slot path skip one more case.** A device checkpoint that an
  admitted request has claimed but not yet built from keeps its slot, because
  that request's first chunk copies the image out of it. A checkpoint standing
  on a prefix that a live request holds *is* a candidate, since its own slot
  comes back whatever the prefix does.
- **The two victim orders differ.** The slot path uses ADR 0023's order:
  checkpoints first, then class, then LRU. The page path
  (`reclaim_retained_until`) keeps #188's LRU-only order for retained
  prefixes.
- **`--prompt-reuse off` also stops sibling prefix sharing.** A publish needs
  a slot, and reuse off reserves none. Live siblings then no longer share a
  published block either, where before #215 they did.
- **Slots per conversation.** Each link of a chain costs one slot, so each turn
  costs its link plus its checkpoint. A two-turn conversation holds 4 slots,
  and a request with a system block before an image holds 3 (block, head,
  checkpoint). At the default of `N_DECODE_LANES` slots, about two
  conversations keep full reuse. A long tool loop plateaus: once its own
  claimed chain holds every slot, further publishes and captures are skipped.
  The #214 replay should read the skip lines with that in mind.
- **One copy primitive.** The only device image copy is
  `ignis_seq_copy_slot_state` (slot to slot). `ignis_seq_state_transfer`,
  the prefix transfer, the image clone and `ignis_seq_checkpoint_image_bytes`
  are removed, which answers #211's open question. The allocation counter's
  `PrefixImage` / `CheckpointImage` / `CheckpointTailPage` kinds stay, and
  over the baseline mix they read zero.
- **Logs, no new metric.**
  - `ignis.scheduler.retained_slots` (DEBUG) is emitted whenever the count
    held changes.
  - `ignis.request.retained_slot_skipped` (INFO) carries `reason` and
    `retained_slots_in_use` / `retained_slots` as they were at the skip. It
    is logged whenever prompt reuse is on, including `--retained-slots 0`.
  - No Prometheus series was added.

### Slice 6 — vision encoder shares the prefill scratch

- **One arena for both.** The vision encoder workspace and the prefill
  scratch become a single arena sized `max(prefill scratch, vision encoder
  workspace)`. Media encode already runs inside a prefill step and resets
  only its own workspace (`vision_encode.cu:148-150`), while prefill uses
  scoped `model->scratch`. Make both use the one arena, never live at the
  same time.
- **The media embedding (335,544,320 B) stays its own reservation.** Prefill
  chunks read the live embedding from it.
- **Expected saving:** ~1.25 GiB (the prefill scratch). Record the measured
  value in `vram_plan`.

### Slice 7 — KV-RAM as one pinned arena

- **One arena for the tier.** Use the vendored `HostPinnedArena`
  (`kernel/vendor/src/core/arena.cu`): one `cudaHostAlloc(…,
  cudaHostAllocDefault)` of `--kv-host-pool-bytes` at start, which is the
  same call ninfer's KV-RAM cache uses. Blobs for evicted sequences, spilled
  checkpoints and spilled prefixes are placed first-fit inside it and freed
  back to it. No `cudaHostAlloc` or `cudaFreeHost` happens while serving.
- **A failed start allocation refuses the start**, naming the size and
  `--kv-host-pool-bytes` (0 disables the tier).
- **A blob with no fitting hole** is handled like today's full budget: the
  host tier's victim order runs; if that frees no fitting hole, the spill or
  evict does not happen.
- **The host tier's byte ledger** (`crates/core/src/host.rs`) stays the
  admission accounting. The arena is where the bytes are, and a test pins
  that the two agree.

### Slice 8 — gate: repeat the 2026-09-17 measurement

- **The run:** `make start` defaults, with a Makefile default of
  `SYSTEM_MESSAGE_POLICY=merge` so the raw #191 trace replays with no merged
  copy. Replay `bench/traces/reuse-191-trace.jsonl` (`--max-gap 10`) plus
  `.scratch/vram-analysis/parallel_lane_load.py` (4 × `@agent`), sampled by
  `.scratch/vram-analysis/gpumem.ps1`, for at least as long as the original
  14 minutes.
- **Pass:**
  - process commit flat after the first request, within a stated tolerance;
  - process shared equal to the KV-RAM arena, within a stated tolerance, and
    not moving;
  - no dedicated/shared trading at constant commit;
  - `vram_plan` total matching Task Manager within tolerance;
  - zero `leaf_error` storms;
  - every request of the trace served.
- **Record** the new KV token capacity and the retained-slot skips. An error
  detector, not an optimization step. GPU exclusive: `make gpu-status`
  first.

## Testing Decisions

Good tests check what the next layer observes: the HTTP status and error
code, the rendered prompt text and token ids, the plan numbers, the slot
counts, the request's final state. They never check the order of CUDA calls.

- **CPU, server/artifact:**
  - A render failure gives 400 `render_failed`, and no request reaches the
    engine.
  - An unknown role gives 400 `invalid_role`.
  - A zero-token prompt is refused.
  - Every system and developer mode produces the expected rendered text for:
    - `[system A, system B, user]`
    - `[developer, user]`
    - `[system, user, developer, assistant, user]`
    - `[system, developer, system, user]`
    - tools present
  - The generation opener, the system block end and the last-user-query
    offset are exact token prefixes on the real frontend for each mode.
  - Configuration errors: headroom plus budget; oversubscription without a
    budget; `--retained-pool-bytes`.
  - Prior art: `crates/artifact/tests/real_frontend.rs`,
    `crates/server/tests/openai_http*.rs`, `config.rs` tests.
- **CPU, core (mock `Compute`):**
  - A decode error ends the request with `Error` and releases the lane, with
    no repeated retries.
  - Retained slots:
    - full slots evict retained state in ADR 0023 order;
    - a claimed prefix is never a victim;
    - no victim skips the publish or capture without delaying admission;
    - a checkpoint plus its prefix cost two slots;
    - `--prompt-reuse off` reserves none.
  - The plan arithmetic: derived and explicit modes, minimum refusal,
    oversubscription warning.
  - Prior art: `crates/core/tests/prefix_reuse.rs`, `host_tier.rs`,
    `checkpoint_*`.
- **GPU** (fails, never skips, when the card is busy; `--test-threads=1`):
  - **Reuse bit-exactness** through retained slots equals today's
    `cudaMalloc` images: turn N+1 reusing turn N generates what the split
    cold prefill generates, on a DFlash2 load.
  - **Media encode** out of the shared arena generates the same tokens as
    before, for an image request.
  - **KV-RAM** evict/restore and spill/restore through the pinned arena
    continue exactly.
  - **No request-time device allocation.** After warm-up, a fixed request mix
    leaves `cudaMemGetInfo` free memory unchanged; a hook or counter on the
    leaf's allocation wrappers is acceptable if cleaner.
  - Prior art: `seq_snapshot_gpu.rs`, `prefix_reuse_gpu.rs`,
    `vision_prefix_reuse_gpu.rs`, `vision_mixed_load_gpu.rs`.

`cargo test` passes workspace-wide, and
`cargo check --workspace --features cuda --tests` too.

## Out of Scope

- A dedicated retained-image structure replacing slots in the sequence pool
  (owner's later change).
- Runtime detection of WDDM paging (the plan prevents it; detecting it after
  the fact is a separate question).
- Changing the lane count (`N_DECODE_LANES` stays compile-time).
- Re-deriving the budget while serving when the desktop grows.
- Tier 2 KV-disk.
- An `inplace` value for the system message policy (ninfer's exact
  behaviour). The enum leaves room for it.

## Further Notes

- **ninfer reference points:**
  - `src/serve/request_memory.h:15-17`: no request-time device allocation;
  - `src/core/arena.cu:274-286`: the pinned KV-RAM arena;
  - `layouts_impl.h:645-654`: workspace `max` including `vision_encode`;
  - `linear_state_slots.h`: `2 × max_concurrency` checkpoint slots;
  - `src/targets/qwen3_6/impl/frontend/chat_template.cpp:316-458`: later
    system messages rendered in place.
- **Other engines:** vLLM, SGLang and llama.cpp reject two system messages
  under the Qwen 3.5+ template, and Ollama on Qwen 3.8 joins every system and
  developer message with `"\n\n"`
  (`.scratch/vram-analysis/research-multi-system-messages.md`).
- **Measured at the Makefile defaults** (2026-09-17, 726a1b7):
  - load 28,314 MiB;
  - KV pool 4.0 GiB (7281 pages, ~291K tokens);
  - vision reservation 2,555,381,504 B;
  - free after load 2.39 GiB;
  - derived retained budget 744,624,128 B (3 checkpoint images).
- **Plan estimate at 1 GiB headroom** (1.6 GiB desktop): budget ~29.2 GiB,
  fixed part ~22.4 GiB, 8 retained slots 1.78 GiB, KV ~5.0 GiB (~364K
  tokens).
