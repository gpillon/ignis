# 03 — The KV pool policy, page-wise reservations, the default `max_tokens`, and the KV-disk tier

GitHub: the ticket that links here. ADRs: 0045 (this feature, amended
2026-10-08), 0030 (the VRAM plan it amends), 0029 (the residency tiers; Tier 2
is built here), 0024 (the blob, now moved a window at a time), 0023 (the
eviction priority, one tier further down, and a page shortage that takes what
is not decoding first), 0017 (the metric contract), 0022 (pages derived from
the format). Spec core-05 (the reservation rule it replaces). Evidence:
[Flash-Next on the 5090](../../findings/2026-10-06-flash-next-on-the-5090.md)
(its lanes-at-262K update), the F: disk bench (`.scratch/disk-bench/RESULTS.md`,
untracked), GitHub #205.

## Problem Statement

- **Flash-Next's KV pool is every lane's whole context.** On the 5090 that is
  786K tokens (3.46 GiB) at the default three lanes. The VRAM it takes is the
  expert cache's, and the cache sets decode speed: 13.38 GiB and 85-89 tok/s at
  three lanes, against 15.74 GiB and 98.5-99.3 tok/s at one.
- **The pool cannot be named in the quantity the owner reasons in.**
  `--kv-pool-bytes` is a byte figure, validated against the 27B's geometry, and
  Flash-Next ignores it.
- **The 12 GiB expert cache floor has no way past it,** even for an operator
  who knows the cost.
- **A reservation is the whole bound, held from admission.** A request reserves
  its prompt plus its `max_tokens`, or the whole `--max-context` without one
  (core-05), and the pages are mapped at once. On a 524,288-token pool two
  agent requests without `max_tokens` take every page, mostly pages neither
  will ever write.
- **A request that names no cap may generate to the end of the context.**
- **Live work is lost when KV-RAM fills.** KV-RAM discards an evicted live
  sequence to make room, and that request re-prefills from zero.
- **Flash-Next can barely evict.** Its host can spare 0.5-2 GiB of KV-RAM
  beside 37.8 GB of pinned experts, and with no arena it cannot evict at all.
  The host has NVMe to spare instead.

## Solution

- **A KV pool policy chosen by residency.**
  - When every weight is on the device, the pool takes the rest of the budget
    (the 27B, and Flash-Next on a card that holds every expert).
  - When experts stream (Flash-Next on the 5090), the pool is reserved first:
    524,288 tokens by default, capped at every lane's context and never below
    one `--max-context` sequence plus a page per retained slot. The expert
    cache takes the rest.
- **`--kv-pool-bytes` names the pool on both models,** in bytes or in tokens
  (`512Ktok`).
- **`--allow-expert-cache-below-floor` starts below the 12 GiB floor** with a
  warning.
- **A default generation cap.** A request that names none generates at most
  `--default-max-tokens`, 38,912 on both models; `0` is today's behaviour.
- **Reservations grow by pages.** Admission reserves the prompt plus a
  2,048-token step, and a lane takes the next step as it generates. When the
  pool runs out, the lowest-ranked sequence below the requester moves down a
  tier, mid-generation if need be, and resumes bit-exact later.
- **Tier 2, KV-disk, below KV-RAM.**
  - A blob that KV-RAM gives up goes to disk, and so does a device victim that
    KV-RAM cannot take.
  - A blob comes back from disk straight to the device.
  - With the tier on, no live work is discarded for room. When no tier has
    room, the request that needed it waits, and a growing lane parks.
  - On by default for Flash-Next (16 GiB); off by default for the 27B.

ADR 0045 holds the reasons and the rejected alternatives. This spec holds the
seams, the defaults, the acceptance and the phases.

## User Stories

1. As the owner running Flash-Next on the 5090, I want the KV pool sized as a
   number of tokens the lanes share, so that VRAM no lane is using goes to the
   expert cache and decode gets faster.
2. As the owner on a card that holds every expert, I want the KV pool to take
   the rest of the budget, so that a resident model serves the most context the
   card can hold.
3. As the owner, I want the 27B's plan unchanged, so that nothing I have
   measured on it moves.
4. As an operator, I want to name the KV pool in tokens, so that one number
   means the same context on either model and either KV format.
5. As an operator, I want `--kv-pool-bytes` honoured on Flash-Next, so that a
   flag I pass is never silently ignored.
6. As an operator, I want to start Flash-Next below the expert cache floor when
   I choose to, with a warning, so that a smaller card or a crowded desktop can
   still run it knowingly.
7. As an agent's user, I want an evicted conversation to resume where it
   stopped, not re-prefill, even when RAM cannot hold it, so that a burst
   beyond the pool costs a disk crossing, not minutes of prefill.
8. As an agent's user, I want my request to wait when no tier has room, never
   to fail and never to cost another request its work.
9. As the owner, I want the disk tier never to stall the decoding lanes, so
   that an eviction is invisible to whoever is generating.
10. As the owner, I want the n-gram reads during a prefill served before the
    tier's IO, so that a spill does not slow a prompt.
11. As the owner, I want a torn, corrupt or foreign file never restored, so that
    the disk can only cost time, never correctness.
12. As the owner, I want the tier to leave my volume a margin and never block a
    start, so that a full F: degrades reuse instead of breaking the server.
13. As the owner, I want files from a crashed run removed and two servers on one
    model directory kept apart, so that the tier neither accumulates nor
    deletes another process's state.
14. As the owner, I want a disk restore taken only when it beats re-prefilling,
    so that the tier never makes a request slower.
15. As an operator, I want the disk tier on `/metrics` and the Monitor, so that
    I can see what it holds and what it saved.
16. As the owner, I want the 27B free of disk writes unless I turn the tier on.
17. As an agent's user, I want three agents that send no `max_tokens` to run
    at once on the default Flash-Next load, not two.
18. As the owner, I want a request to hold the pages it has written plus a
    small step, not the pages it might write, so that the pool serves the
    lanes that are actually generating.
19. As an agent's user, I want a conversation moved off the device in the
    middle of its answer to resume exactly where it stopped, token for token.
20. As the owner, I want an agent's growth never to move an interactive
    conversation, and two equal conversations never to keep swapping each
    other in and out.
21. As an operator, I want a request with no cap to stop at a sane length, and
    one flag to restore the old behaviour.
22. As the owner, I want a live move to slow the other lanes' decode by no more
    than a bound I have confirmed.

## Defaults

Items marked *(owner)* are the owner's decisions of 2026-10-07 and 2026-10-08.
Items marked *(agent)* are proposals the owner confirms or changes; ADR 0045
gives the reason for each.

| Point | Default | |
|---|---|---|
| Offloaded pool | `max(floor, min(524288, decode_lanes × --max-context))` tokens | *(owner)* size, *(agent)* cap and floor |
| Resident test | budget ≥ fixed lines + residency's fixed lines + every expert projection + floor pool | *(agent)* |
| `--kv-pool-bytes` | bytes, or `<n>[K\|M]tok`; honoured on both models | *(agent)* |
| Floor opt-in | `--allow-expert-cache-below-floor`, WARN; class minimum stays a refusal | *(owner)* |
| Default cap | `--default-max-tokens 38912` on both models, clamped to the context; `0` = up to the context | *(owner)* value and `0`, *(agent)* name |
| Growth step | 32 pages (2,048 tokens). Admission reserves the prompt plus one step; a lane keeps at least one step of room ahead | *(owner)* growth, *(agent)* size |
| Page-shortage victim | retained state; then only what ranks below the requester (class, then submission): `Agent` first, not decoding first, latest-submitted first | *(owner)* not decoding first, never the requester; *(agent)* rank gate |
| Entry rule | enter when free pages ≥ reservation + 4 steps for itself and each resident sequence above it; a restore never moves anything | *(agent)* |
| Last resort | every resident sequence parked and no tier takes a victim: the lowest-ranked is re-queued, with an ERROR | *(agent)* |
| PCIe contention | move out: ITL p50 +10%; move in: ITL p50 +25%; either: max within baseline + 150 ms | *(agent)*, owner to confirm |
| KV-disk on | Flash-Next `--kv-disk-bytes 16G`; 27B `0` | *(owner)* build, *(agent)* sizes |
| Location | `--kv-disk-path model` (beside the artifact), `auto`, or a directory: the n-gram cache's rule | *(owner)*, confirmed 2026-10-08 |
| Volume margin | 10 GiB, `KV_DISK_VOLUME_MARGIN_BYTES` | *(agent)* |
| Placement | one file per blob, a byte ledger in eviction-priority order | *(agent)* |
| Restore floor | Flash-Next 8,192 tokens, 27B 16,384 tokens | *(agent)* |
| Transfer | 32 MiB windows, two-window pinned staging (64 MiB), two IO threads | *(agent)* |
| n-gram first | no new tier request while a prefill's n-gram gather is pending | *(agent)* |
| Restart | nothing reused; per-process directory, stale ones removed at start | *(agent)* |
| Integrity | CRC32 per window, header page written last as the commit | *(agent)* |
| No room | the request waits in the admission queue, a growing lane parks; the in-flight cap's 503 is unchanged | *(owner)* |

## Seam

What changes, by module. Nothing here is a new process-wide singleton: the
tier, its directory, its workers and its staging belong to the loaded model's
compute, and its drop path frees them.

- **The plan — `crates/core/src/vram.rs`, `crates/core/src/residency/plan.rs`.**
  - One pure function decides the branch and the pool's pages for both models.
    `plan_vram` gains a residency input:
    - whole (the 27B), or
    - offloaded with the expert pool's bytes, residency's fixed bytes, the
      floor and its override.
  - Per-page and per-slot costs come in as arguments, with the arena size as
    the closure `VramRequest::kv_arena_bytes` already is. No FFI runs in the
    plan, so every branch is a unit test.
  - `plan_expert_cache` takes what the pool leaves. `BelowFloor` becomes a
    warning when the override is on. `BelowClassMinimum` stays an error.
- **Flash-Next's pool — `crates/core/src/flash_next.rs`.**
  - `EngineOptions` carries the pool's pages from the plan.
  - `pool_budget()` stops multiplying the context by the lanes.
- **The Flash-Next load — `crates/server/src/runtime.rs`
  (`flash_next_scheduler_with_ngram_cache`).**
  - Plans through the shared function, honours `shape.kv_pool_bytes` and the
    floor override, and adds the plan event fields.
  - Builds the KV-disk store when the tier is on, and puts its staging line on
    the host plan (`plan_host`).
  - Passes the default cap into `SchedulerConfig`.
- **The 27B load (`scheduler`).** The same plan call with a whole residency,
  the KV-disk store when named, and the default cap.
- **Config — `crates/server/src/config.rs`.**
  - `--kv-pool-bytes` parses into bytes or tokens; the 27B-geometry validation
    moves out of `resolve_kv_pool_bytes`.
  - New: `--allow-expert-cache-below-floor`, `--default-max-tokens`,
    `--kv-disk-bytes`, `--kv-disk-path`, and their environment variables.
  - Per-model defaults and refusals live in `served_model_for` and
    `EngineShape::for_family`.
  - Help text: `--kv-pool-bytes`, `--decode-lanes` (a lane no longer holds its
    own whole context) and the new flags.
- **The default cap — `crates/core/src/concrete.rs`.**
  - `SchedulerConfig::default_max_tokens` (`0` = none), read by
    `generation_budget` when the request names no cap and is neither
    constrained nor prefill-only. It is clamped to what the prompt leaves of
    `max_sequence_tokens`.
  - `submit` writes the resolved cap into `params.max_tokens`. The scheduler's
    hard cap and the backend's own check (`RuntimeCompute::decode_step`) then
    read one number.
  - The HTTP handlers do not change. `one_cap` still decides what the client
    sent, `ignore_eos` still requires an explicit `max_tokens`, and
    `/v1/responses` echoes the `max_output_tokens` the client sent.
- **The scheduler, KV-disk — `crates/core/src/concrete.rs`,
  `crates/core/src/host.rs`, a new `crates/core/src/disk.rs`,
  `crates/core/src/checkpoint.rs`, `crates/core/src/prefix.rs`.**
  - The disk ledger: entries, byte budget, victim order. Its order is the
    KV-RAM order, extended.
  - `make_host_room_for_bytes` demotes a KV-RAM victim to the disk instead of
    discarding it, when the tier is on.
  - `evict_one_victim` / `snapshot_and_evict` go straight to disk when KV-RAM
    cannot make room.
  - Transfers are states a request lives in: spilling (still charged on the
    device) and restoring (charged, not yet schedulable).
  - `restore_pass` restores from the disk.
  - The retained match walks three tiers and applies each tier's floor.
  - `ReuseSource::Disk`.
  - With the tier on, no `KvRamVictim::Live` is ever discarded.
- **The scheduler, page-wise reservations — `crates/core/src/concrete.rs`,
  `crates/core/src/admission.rs`.**
  - A request carries its *bound* (prompt plus cap: what `ContextExceeded` and
    `Oversized` are decided on) apart from its *reservation*
    (`resources.kv_pages`: what `kv_used_pages` charges).
  - `reservation()` returns the prompt, or the claimant's tail, plus
    `min(cap, step)`. `requeue_request`, `snapshot_and_evict`'s
    `restore_pages` and the claim loop follow the same rule.
  - A growth pass before each round. A lane whose room ahead is below a step
    asks for one, highest-ranked first. A lane with room below one round's
    largest append and no page is *parked*: a flag on the `Running` request
    that only the decode round reads.
  - A pure page-shortage victim order in `admission.rs`, beside
    `choose_retained_lane_victim`: the rank gate, then `Agent` first, not
    decoding first, latest-submitted first. The lane-shortage order is not
    touched.
  - The entry rule, in `restore_pass` and at materialization (beside
    `fits_for_materialization`), with entries taken in rank order.
  - The last resort, and a `Parked` fact.
- **The `Compute` seam — `crates/core/src/scheduler.rs`, `crates/core/src/mock.rs`.**
  - KV-disk calls: start a spill to disk, from the device or from a KV-RAM
    blob; advance in-flight transfers by at most one window each; report
    finished and failed transfers; start a restore from disk; discard a disk
    blob; ask whether a blob of N bytes fits the disk.
  - Growth: `grow(request, context_tokens)`, refused with nothing changed when
    the pages are not free.
  - `MockCompute` implements a fake disk with configurable latency, capacity
    and failures, and mirrors the leaf's page entitlement so that a scheduler
    test catches the two ledgers drifting.
- **The runtime — `crates/runtime/src/lib.rs`, plus a `kv_disk` module.**
  - The store: directory lifecycle and lock, file format, the two IO workers,
    the staging windows, CRC.
  - `RuntimeCompute` implements the new seam calls: the disk calls, and
    `grow` through `crates/core/src/seq.rs` (`Seq::grow`).
  - `StepLeaf` gains windowed snapshot and restore: whole-sequence,
    checkpoint and prefix blobs, read or fed a window at a time on the leaf's
    transfer stream, with completion polled.
  - Sequences are allocated at their reservation, and a restore maps the
    blob's tokens plus a step.
- **The leaf — `kernel/`.**
  - The snapshot and restore calls, for sequences, checkpoints and prefixes,
    take a transfer-options struct (ADR 0016, ADR 0024's "partial extent"):
    a byte window and the stream to use. A null options pointer is today's
    whole-blob call.
  - A restore keeps a cursor. A sequence whose restore is incomplete refuses
    every step and can be released.
  - `ignis_seq_alloc`, `ignis_seq_alloc_shared` and
    `ignis_seq_alloc_from_checkpoint` map the tokens they are given, which is
    now the reservation, not the bound. The block-table row stays sized for
    `max_context_tokens`.
  - New `ignis_seq_grow(pool, seq, context_tokens)`. It raises the entitlement
    with the vendored `PagedKVAllocation::set_page_entitlement`, maps the new
    pages with `materialize_pages` (which publishes the block-table range),
    and zeroes them with `zero_pages`, as `ignis_seq_alloc` does. It is called
    between rounds, on the model stream.
  - It refuses with nothing changed when the pages are not free, past
    `max_context_tokens`, or on a sequence with an incomplete restore.
- **Direct IO — `crates/artifact/src/direct.rs`.** A `DirectWriter` twin of
  `DirectReader`: positional, unbuffered, aligned writes.
- **The n-gram table — `crates/core/src/ngram_table.rs`.** Exposes whether a
  prefill gather is pending, which the tier's workers check before issuing a
  request.
- **Metrics — `crates/core/src/types.rs` (`SchedEvent`),
  `crates/server/src/telemetry.rs`, `crates/server/src/metrics.rs`, ADR 0017's
  table.**
  - New facts: a disk spill (with its source tier), a disk failure (write or
    read), and a lane parked. The tick carries the disk's used bytes.
  - The retained-state facts carry `ReuseSource::Disk`.
  - `ignis_kv_lane_parks_total`, on every load.
- **The Playground Monitor — `web/src/monitor/`** (snapshot contract, derive,
  view, their vitest suites) and `web/mockMetrics.ts`.
- **Make and docs.**
  - `mk/config.mk` and the `Makefile`: new knobs `DEFAULT_MAX_TOKENS`,
    `KV_DISK_BYTES`, `KV_DISK_PATH` and `ALLOW_EXPERT_CACHE_BELOW_FLOOR`.
  - `make config` prints the pool policy, the pool's tokens, the default cap,
    and the disk directory and budget.
  - `docs/user/README.md`, and `CONTEXT.md` (updated with ADR 0045).

## Acceptance criteria

Each criterion belongs to one phase (see Phases).

### The pool policy

1. **The 27B is resident and unchanged.**
   - For every 27B plan in `vram.rs`'s tests, the plan is byte-identical to
     today's: the pool takes the rest.
   - `kv_pool_policy` reads `resident`.
2. **Flash-Next offloaded, at the 5090's plan.** Take the lines of
   `ignis.runtime.flash_next_plan` in the 5090 finding, 262,144-token context,
   8 host retained slots.
   - The pool is 4,104 pages at `--decode-lanes 1`: today's one-lane pool,
     byte for byte.
   - It is 8,192 pages at `--decode-lanes 2` through `8`.
   - The expert cache is the budget less every line, the pool and residency's
     fixed bytes.
   - At `--decode-lanes 3` the cache is ~1.03 GiB larger than today's plan.
3. **Flash-Next resident.**
   - With a simulated budget that holds every expert projection (for example
     96 GB), the expert cache holds the whole expert pool and the KV pool takes
     the rest.
   - At a budget exactly on the threshold the load is resident; one byte less,
     it is offloaded.
4. **The floor.**
   - A pool below one `--max-context` sequence plus one page per retained slot
     refuses the start, naming the knobs and the model's own bytes per token.
   - At `--max-context 524288` the default becomes the floor (8,200 pages), not
     a refusal.
5. **`--kv-pool-bytes` spellings.**
   - It accepts a byte count with an optional `K`/`M`/`G` suffix, and
     `<n>tok` / `<n>Ktok` / `<n>Mtok` (binary).
   - The flag wins over `IGNIS_KV_POOL_BYTES`.
   - A malformed value is refused at config, naming the flag. The config no
     longer validates against the 27B's geometry.
   - On Flash-Next a named pool is checked against Flash-Next's own per-token
     cost at the plan. Offloaded, it replaces the default. Resident, it
     replaces the minimum pool in the residency test and becomes the pool, and
     the rest of the budget stays unused, as on the 27B.
   - `512Ktok` gives an 8,192-page pool on either model.
6. **The floor opt-in.**
   - Below 12 GiB the start is refused by default, and the message names
     `--allow-expert-cache-below-floor`.
   - With the flag the load starts and emits a WARN,
     `ignis.runtime.expert_cache_below_floor`, carrying the cache's bytes, the
     floor and the knobs that would lift it.
   - The class-minimum refusal is unchanged, flag or no flag.
   - A 27B start with the flag is refused, naming the model.
7. **The plan says which branch it took.**
   - `ignis.runtime.vram_plan` and `ignis.runtime.flash_next_plan` carry
     `kv_pool_policy` (`resident` or `offloaded`), `kv_pool_pages` and
     `kv_pool_tokens`.
   - `make config` prints the policy and the pool's tokens.
   - An explicit `--vram-budget-bytes` above free memory refuses a Flash-Next
     start unless `--allow-vram-oversubscription` is given, as on the 27B.
8. **GPU: Flash-Next decode does not regress (A-B-A).** Setup:
   - A is the commit before this change, B this change, then A again. All
     three legs run at the defaults (3 decode lanes, 262,144-token context,
     `make config MODEL=flash-next`'s flags).
   - The harness is the lanes-at-262K finding's: greedy, `ignore_eos`, 1,800
     tokens, `max_tokens` set.

   With one active lane and with three, B's decode tok/s and decode hit rate
   are each at least the lower of the two A legs. The expected direction is up.
   The result is recorded in a finding.
9. **GPU: three agents without `max_tokens` on the default plan.** Three
   concurrent greedy requests, class `agent`, no `max_tokens`, prompts of
   ~30K tokens each.
   - All three hold a decode lane at once: none waits for pages and none is
     refused. Each one's bound is its prompt plus the default cap.
   - The harness cancels them once each has decoded 1,800 tokens with all
     three on lanes, so the leg costs minutes, not the cap.
   - `ignis_kv_ram_evictions_total` stays 0.
   - The finding records the aggregate tok/s against AC 8's A leg.

### The default `max_tokens`

10. **The flag.**
    - `--default-max-tokens <n|0>` and `IGNIS_DEFAULT_MAX_TOKENS` default to
      38,912 on Flash-Next and on the 27B. The flag wins over the
      environment.
    - A malformed value is refused at config, naming the flag. `0` means no
      default: a request without a cap runs up to the context, as today.
    - A value past `--max-context` is accepted and acts as the context.
11. **What it caps** (CPU over `MockCompute`, and the HTTP tests).
    - A chat request with neither `max_tokens` nor `max_completion_tokens`
      generates at most 38,912 tokens and ends `finish_reason: "length"`.
      A `/v1/responses` request without `max_output_tokens` ends `incomplete`
      with `reason: "max_output_tokens"`.
    - With thinking on, the reasoning tokens count inside the cap. The default
      thinking budget still forces its close at 6,144 and keeps its 2,048-token
      answer reserve.
    - A prompt that leaves less than 38,912 of the context gets what is left,
      and is not refused. A prompt that fills the context alone is refused, as
      today.
    - An explicit cap wins, larger or smaller. One past the context is refused
      as today, naming its field.
    - `ignore_eos` without an explicit `max_tokens` is still a 400.
    - A decision, a constrained decode and a prefill-only request are
      unchanged.
    - With `--default-max-tokens 0`, a request without a cap runs to the
      context.
    - The resolved cap is in the request's `params.max_tokens` after `submit`,
      so it is enforced exactly as an explicit `max_tokens` is today,
      speculation included.
    - Both models.

### KV-disk: configuration and lifecycle

12. **Flags and defaults.**
    - `--kv-disk-bytes` and `IGNIS_KV_DISK_BYTES` default to 16 GiB on
      Flash-Next and `0` on the 27B; `0` turns the tier off.
    - `--kv-disk-path` and `IGNIS_KV_DISK_PATH` take `model`, `auto` or a
      directory, resolved as `ngram_cache::CacheLocation` resolves them.
      `auto` is `LOCALAPPDATA/ignis/cache/kv-disk` on Windows, and
      `XDG_CACHE_HOME/ignis/kv-disk` or `HOME/.cache/ignis/kv-disk` on Linux.
    - The flag wins over the environment.
    - A 27B load at its defaults creates no directory and writes nothing.
13. **Directory lifecycle.**
    - A load writes under `<location>/ignis-kv-disk/<pid>-<nonce>/` and holds a
      lock file there, open exclusively, for its life.
    - Two processes on one location: the second leaves the first's directory
      untouched.
    - At start, a directory whose lock can be taken (its owner is gone) is
      removed.
    - A clean shutdown removes the process's own directory.
    - No file written by another process is ever read.
14. **The budget and the volume.**
    - At start the effective budget is `min(--kv-disk-bytes, volume free −
      10 GiB)`. It is printed on `ignis.kv_disk.ready`, with the directory.
    - A WARN is emitted when the effective budget is below the flag. At zero or
      less, the tier is off with a WARN, and the start proceeds.
    - A write that would cross the margin, or that fails, is refused, and the
      blob stays where it was. `ignis_kv_disk_failures_total{op="write"}`
      counts it.
    - `ignis.kv_disk.write_refused` logs once per entry into the refusing
      state, not once per write.

### KV-disk: file format and integrity

15. **The file format.** A file is a 4 KiB header page, written last, followed
    by the blob's windows. The last window is padded to `DIRECT_IO_ALIGNMENT`
    for the unbuffered write, and the header holds the true length. The header
    carries:
    - the served model id;
    - the blob identity: structural artifact hash, KV format, layout version,
      drafter presence and window;
    - the sidecar's payload SHA-256 when one is present;
    - the RoPE scaling;
    - the blob's kind, match key, token count and byte length;
    - a CRC32 per window, and the header's own CRC32.

    Unit tests:
    - A file with no header (a torn write) is refused and never restored.
    - So is a file with any single flipped byte, in any window or in the
      header.
    - So is a header that differs from the load in any one identity field,
      each tried in turn.
    - A refused live blob's request re-prefills, with an ERROR and
      `ignis_kv_disk_failures_total{op="read"}`. A refused retained blob is
      discarded.

### KV-disk: the windowed transfer

16. **Kernel test: windowed equals whole.**
    - A blob produced window by window equals the whole-blob call, byte for
      byte.
    - Cover window sizes of 4 KiB, 32 MiB, and one that does not divide the
      blob; both KV formats; both models' pools; a sequence holding a shared
      prefix (materialized); a checkpoint blob; and a prefix blob.
    - A restore fed window by window gives a sequence byte-identical to a whole
      restore.
    - A sequence with an incomplete restore refuses every step and releases
      cleanly.
    - A null options pointer is today's call, and its test is unchanged.

### KV-disk: the scheduler (CPU, `MockCompute` with a fake disk)

17. **The chain.**
    - A device victim goes to KV-RAM when it fits.
    - A KV-RAM victim goes to disk instead of being discarded: a live one
      always, a retained one when it outranks the disk's lowest entry.
    - A device victim goes straight to disk when the arena is 0, or when it
      cannot make room.
    - The disk discards in the extended order: retained before live, then
      class, then probation before protected, then least-recently-used, with
      the Interactive TTL.
    - With the tier on, no `SnapshotDropped` is ever emitted (but for the last
      resort of AC 33).
18. **No room.**
    - Fill every tier with live or higher-ranked state. A new request under the
      in-flight cap is then held in `Admitted`: not refused, and nothing is
      discarded.
    - It is admitted at the first advance after a lane completes.
    - A request beyond the in-flight cap is still refused `Full`, as today.
19. **The restore floor.**
    - A disk match is taken only when it beats the best device and KV-RAM
      match by the family's floor: 8,192 tokens on Flash-Next, 16,384 on the
      27B.
    - A tie goes to the tier above.
    - A live disk blob is restored whenever room returns, with no floor.
20. **The model thread never waits on the disk.** With a fake disk that takes
    1 s per window:
    - The other lanes' decode rounds keep advancing during a spill and during
      a restore.
    - Each advance copies at most one window per transfer.
    - The victim's pages stay charged until its file commits.
    - A restoring request is not scheduled before its last window lands.
    - A write that fails leaves the victim resumable on the device, and the
      request that wanted the room still waiting.
21. **Cancel mid-transfer.** A request cancelled mid-spill or mid-restore
    releases its pages, its file is deleted, and the ledgers' used bytes return
    to their value before the transfer.
22. **The n-gram table goes first.** While a prefill gather is pending (a fake
    table flag in the CPU test), no new tier request is issued. A request
    already issued completes its window.

### KV-disk on the GPU

23. **Flash-Next: a forced overflow restores bit-exact.** Setup:
    - `--decode-lanes 2`, `--kv-pool-bytes` at the floor,
      `--kv-host-pool-bytes 0`, the disk tier on.
    - Request A (class `agent`, greedy, explicit `max_tokens` with
      `ignore_eos`) has a prompt over half the context, so no second such
      prompt fits beside it, whether reservations are whole or grow. A
      prefills and decodes alone.
    - Request B (`interactive`), also over half the context, arrives, moves A
      straight to disk, and runs alone to completion.
    - A restores from disk and finishes alone.

    A's greedy tokens are identical to a run of A alone with the same prefill
    chunking. Every decode round is at width 1 in both runs, so a width change
    cannot explain a difference (finding 2026-09-14, batched decode width
    drift).

    A second leg checks the chain, not tokens. Setup:
    - `--decode-lanes 3`, the pool at the floor, a KV-RAM arena that holds
      one blob.
    - A and B (`agent`, prompts of ~30% of the context, explicit `max_tokens`
      of ~10% with `ignore_eos`) decode together.
    - C (`interactive`, a prompt of ~80%) arrives and needs both of them out.

    Expected:
    - The first of A and B to move lands in KV-RAM. It is demoted to disk to
      make room for the second: `ignis_kv_disk_spills_total{from="kv_ram"}`
      is at least 1.
    - C runs alone to completion. Then A and B come back and finish.
    - Every request generates its full `max_tokens`. No `Requeued` is
      emitted, and `ignis_kv_ram_evictions_total` stays 0.
    - C's tokens equal its lone run's, since it decoded alone throughout. A's
      and B's are not compared: they shared rounds at width 2.
24. **The 27B: the same overflow.** The test of 23 runs on the 27B with
    `--kv-disk-bytes` named: the mechanism is generic.
25. **Contention on F:, measured.** On Flash-Next, force a ≥ 1 GB spill to disk
    twice: once during a prefill chunk with an uncovered n-gram table, and once
    while two lanes decode. Compare against the same work without the spill.
    - The chunk's wall time and its n-gram gather time are recorded, and so are
      the decoding lanes' ITL p50 and p99.
    - Starting bounds, for the owner to confirm: chunk wall time and ITL p50
      each within +10% of the run without the spill.
    - The result goes in a finding.

### Page-wise reservations

26. **The reservation (CPU).**
    - At admission a request reserves `ceil((prompt + min(cap, 2,048)) / 64)`
      pages. A claimant of a shared prefix or a checkpoint reserves its tail
      in place of the prompt.
    - Its bound, prompt plus cap, is what `ContextExceeded` and `Oversized`
      are decided on, as before.
    - A request whose bound is within one step of its prompt reserves the
      bound and never grows: a decision, a constrained decode, a
      `max_tokens` up to 2,048.
    - After every advance, `kv_used_pages` equals the leaf's entitled pages.
      `MockCompute` mirrors the leaf's ledger.
    - A request alone on a pool at the floor (retained state given up, every
      other sequence moved) reaches its bound.
27. **Growth and parking (CPU).**
    - Before a round, a lane whose room ahead is below one step (2,048 tokens)
      is granted one more step, capped at its bound. While pages are free it
      never parks.
    - Without a page, a lane whose room is below one round's largest append
      (1, or the draft window plus one under speculation) parks:
      - it is left out of the round and keeps its pages;
      - it emits a `Parked` fact, counted once per entry on
        `ignis_kv_lane_parks_total` (a row in ADR 0017's table);
      - it decodes again at the first advance with room.
    - While a disk move it triggered is in flight (fake disk, 1 s per
      window), the grower keeps decoding on its room. It parks only if the room
      runs out first.
    - A cancelled parked lane releases its pages.
28. **Kernel test: grown equals whole.**
    - Take a sequence allocated at prompt plus one step and grown to N tokens
      with `ignis_seq_grow`, fed the same prefill and decode as one allocated
      whole at N. Its KV bytes, its block table read by logical page, and its
      state are identical to the whole one's.
    - The new pages are zeroed, as at allocation.
    - A refused growth leaves the sequence and the pool unchanged, and the
      sequence usable. The refusals: pool short, past `max_context_tokens`,
      an incomplete restore.
    - Cover both KV formats; both models' pools; a sequence on a shared prefix
      and one from a checkpoint, whose tails grow; and a snapshot taken after
      growth, restored into its tokens plus one step.
    - `test_seq_alloc` and `test_seq_snapshot` carry it. The whole-allocation
      tests are unchanged.
29. **GPU: grown equals whole, both models.** A greedy request decoded at
    width 1 with a page-wise reservation, growing at least five times, gives
    the same tokens as on the commit before this phase. The 27B leg runs its
    default decode route, so a verify round's append (the draft window plus
    one) is covered by the room ahead.

### Live moves

30. **The victim of a page shortage** (pure tests in `admission.rs`, then
    CPU).
    - Retained state on the device goes first, as today.
    - Then only live sequences that rank below the requester. Rank is class,
      then submission order.
    - Among those, in order: `Agent` before `Interactive`; sequences not in the
      decode round (waiting for a lane, at a chunk boundary, parked) before
      lane holders; the latest-submitted first.
    - Never the requester, a sequence mid-transfer, a protection donor, or a
      lane reserved for an earlier Interactive request.
    - A lane shortage (the head's lane deal, `try_evict_for_head`) keeps
      today's order, unchanged.
    - The case that proves the gate: an `Agent` lane's growth never moves an
      `Interactive` sequence. It parks instead.
31. **Triggers (CPU).**
    - A growing lane moves AC 30's victim into KV-RAM, else KV-disk, at a
      round or chunk boundary. A victim holding a lane resumes as `Running`; a
      lane-less one resumes as `Prefilling` from its progress.
    - An `Interactive` admission moves an `Agent` sequence to fit its prompt
      plus a step.
    - An `Agent` admission behind resident `Agent` sequences moves none of
      them. It waits in `Admitted` and is admitted when room returns.
    - A moved sequence's output continues unbroken on `MockCompute`: no
      `Requeued`, and its generated count is continuous.
32. **The entry rule (CPU).**
    - A restore never moves anything.
    - A moved sequence and a newcomer enter the device only when the free
      pages cover their reservation plus four steps, for themselves and for
      each resident live sequence that ranks above them. For a sequence that
      can grow less than four steps, the room counted is what it can still
      grow.
    - Entries are taken in rank order. A moved `Agent` sequence comes back
      before a later `Agent` newcomer, even one that would fit sooner.
    - Take two equal-class sequences whose bounds cannot both fit, with every
      lane advancing one token a round. Between a restore of the younger and
      its next move, the younger advances at least three steps (6,144
      tokens).
    - Take the default pool (8,192 pages, 262,144-token context, three lanes),
      with three requests generating to their bound. At most one sequence is
      ever off the device, and it moves at most once.
33. **Progress and the last resort (CPU).**
    - The top-ranked live sequence never parks behind a lower-ranked one.
    - Suppose every resident live sequence is parked and no tier can take a
      victim: a fake disk refusing every write, KV-RAM full of live blobs, or
      no tier at all. Then the lowest-ranked parked sequence is re-queued,
      with an ERROR naming the refusing tier and a `SnapshotDropped`.
    - With a disk that accepts writes, the last resort never happens.
    - A request alone on a pool at the floor grows to its bound without
      parking.
34. **GPU: a live move resumes bit-exact, both models.** Every compared round
    is at width 1.
    - *At the compute seam.* A greedy sequence is reserved its prompt plus a
      step and grows as it decodes. It is moved down mid-generation and
      restored with its tokens plus a step: once right after a growth, once
      mid-step, through KV-RAM and through KV-disk (windowed). Its tokens equal
      the same request reserved whole and never moved.
    - *Through the scheduler, with a victim that is not decoding.* Setup:
      - `--decode-lanes 2`, the pool at the floor, the disk tier on.
      - A (`interactive`, a short prompt, explicit `max_tokens` with
        `ignore_eos`) decodes alone.
      - B (`agent`, a long prompt) is still mid-prompt when A's growth runs
        the pool out. A high `--decode-share` makes sure of that.

      B moves, as `Prefilling`, to a tier. A finishes alone. B comes back,
      finishes its prompt and decodes alone. A's and B's tokens each equal
      their lone runs with the same prefill chunking.

### PCIe contention

35. **GPU: a live move's PCIe contention, measured (Flash-Next).** Setup, with
    the sizes left to the implementer and `--kv-pool-bytes` cut so that the
    moves happen:
    - C (`agent`) has a prompt of at least 236K tokens, so its blob is at
      least 1 GB, and decodes.
    - A and B (`interactive`, short prompts, explicit `max_tokens` of
      different lengths with `ignore_eos`) decode beside it.
    - A's or B's growth moves C out (device to host) while both decode. C
      comes back in (host to device) while at least one of them still decodes.
    - Two legs. C goes through KV-RAM, with an arena that holds its blob (the
      synchronous path). Then it goes through KV-disk, with
      `--kv-host-pool-bytes 0` (the windowed path).

    Measured for each move: its bytes, duration and GB/s, and the decoding
    lanes' ITL p50 and max during it. Each is compared against a baseline
    window of the same length, at the same decode width, with no transfer in
    flight: right after the move out, and right before the move in.

    Starting thresholds, for the owner to confirm:
    - Move out: ITL p50 within +10% of the baseline.
    - Move in: ITL p50 within +25%, since it shares the expert stream's
      direction.
    - Either: ITL max within the baseline's max + 150 ms. That covers one
      synchronous KV-RAM copy of a whole-context blob, ~0.1 s at ~12 GB/s.

    No work is lost: A, B and C each generate their full `max_tokens`, no
    `Requeued` is emitted, and `ignis_kv_ram_evictions_total` and
    `ignis_kv_disk_failures_total{op="read"}` stay 0.

    The finding records these numbers and why the two directions differ. The
    expert stream moves 66-74 MB per decode token, ~6.6-7.4 GB/s of the
    link's ~12 GB/s at ~100 tok/s. A move out runs against that direction; a
    move in shares it. The finding also records any time A or B parked.

### Observability, docs and the 27B

36. **The KV-disk metric contract.**
    - The six retained-state families carry `tier="disk"`, and the request log
      spells the reuse source `disk`.
    - New series: `ignis_kv_disk_bytes{state="capacity"|"used"}`,
      `ignis_kv_disk_spills_total{from="device"|"kv_ram"}` and
      `ignis_kv_disk_failures_total{op="write"|"read"}`.
    - All of them render from the first scrape, zeros included, on a load with
      the tier. A load without it never renders them.
    - `ignis_kv_cache_evictions_total` counts device to KV-RAM only, and
      `ignis_kv_ram_evictions_total` counts only live snapshots dropped and
      lost. A demotion to disk is in neither.
    - ADR 0017's table and its disk paragraph say so.
    - The router exposition test covers the new series. Fact traffic is the
      same with metrics off and on.
37. **The Monitor.**
    - Its Disk row is live when the scrape carries the tier: live spills,
      retained discards, and the demotions into the disk.
    - A capacity and used bar shows the disk beside KV-RAM's.
    - The row reads "off on this load" when the series are absent.
    - `assessHealth` weighs a disk read failure like a KV-RAM live drop.
    - vitest covers a fixture scrape with the tier and one without, and the
      mock simulator feeds the row.
38. **Make and docs.** Each phase lands its own part.
    - **P1.**
      - Knobs `DEFAULT_MAX_TOKENS` and `ALLOW_EXPERT_CACHE_BELOW_FLOOR`.
        `make config` prints them with the pool policy and its tokens.
      - The `--decode-lanes` help text no longer says each lane holds its own
        whole context.
      - `docs/user/README.md` documents the token form of `--kv-pool-bytes`,
        the policy and the floor opt-in. It documents `--default-max-tokens`
        in its flag table, and in the chat completions section with this
        text, or close to it:

        > A request that sends no `max_tokens` (nor `max_completion_tokens`,
        > nor `max_output_tokens` on `/v1/responses`) generates at most
        > `--default-max-tokens` tokens, 38,912 by default, its reasoning
        > included. It ends with `finish_reason: "length"` when it gets there.
        > Before, it could generate to the end of the context. Send
        > `max_tokens` for more, or start the server with
        > `--default-max-tokens 0` for the old behaviour.
    - **P2.**
      - Knobs `KV_DISK_BYTES` and `KV_DISK_PATH`. `make config` prints them
        with the disk directory and budget.
      - The README documents the tier and its flags.
    - **P3.** The README says how a request takes pages, in this sense:

      > A request reserves its prompt plus 2,048 tokens when it is admitted,
      > and takes pages as it generates. When the pool runs out, the
      > lowest-ranked sequence moves to KV-RAM or KV-disk and later resumes
      > exactly where it stopped. No request is refused or loses work for it.
      > A request may wait, and a lane may pause, until room returns.
39. **The 27B at its defaults**, checked at the end of every phase.
    - Config resolution is identical but for the new fields: the default cap
      at 38,912, the disk off.
    - The plan is byte-identical (AC 1).
    - No disk directory is created.
    - Its serving changes only in that its requests take the default cap
      (after P1) and grow their reservations (after P3).
    - The 27B GPU profile is green.
40. **The suites are green**, at the end of every phase.
    - `cargo test` passes workspace-wide.
    - `cargo check --workspace --features cuda --tests` is clean.
    - `npm test` passes in `web/`.
    - The phase's GPU tests are green on a free 5090, the card checked free
      first (AGENTS.md).

## Implementation Decisions

- **The policy is one function for both models.** Flash-Next's hand-rolled
  plan in `runtime.rs` goes through `plan_vram`. The 27B gets
  `Residency::Whole` and so takes today's branch by construction.
- **Tokens are converted to pages by the plan, never by the config.** The
  config cannot know the model, which is what made today's validation wrong
  for Flash-Next.
- **The default cap has one home, the scheduler.** `generation_budget` applies
  it, and `submit` writes the result into `params.max_tokens`. Every entry
  point gets it (chat, responses, the Playground), and `refusal` and `submit`
  cannot disagree.
- **The disk ledger mirrors `HostTier`** (entries, owner class,
  probation/protected, use tick). The victim order is shared code, not a copy,
  so the two tiers cannot drift.
- **Transfers are scheduler states, not blocking calls.** The KV-RAM path
  stays synchronous: it runs at ~12 GB/s and its cost is measured (ADR 0024).
  Only disk transfers are pumped. AC 35 measures whether a synchronous live
  move is too long a stall.
- **The staging is reserved at load** (ADR 0030: serving allocates nothing).
  It is a host-plan line on Flash-Next and part of the tier's open on the 27B.
  The arena's spans are aligned to `DIRECT_IO_ALIGNMENT`, so that KV-RAM to
  disk writes straight from them.
- **The tier's IO threads are the tier's own,** not the n-gram reader's: one
  pool per purpose, so a slow write never occupies a gather's thread.
- **The bound and the reservation are two fields of a request.** The bound
  decides refusals at submit and caps growth. The reservation is what the pool
  and the leaf hold.
- **One pass per advance, in this order.**
  1. The growth pass: resident lanes, highest-ranked first.
  2. Entries, restores and admissions alike, in rank order, under the entry
     rule.
  3. The decode round, without the parked lanes.
- **Growth and admission share the move path.** A move is
  `snapshot_and_evict` into KV-RAM, or P2's spill to disk. Growth is one more
  caller beside admission and the head's lane deal.
- **Parked is a flag on a `Running` request, not a state.** It keeps its lane
  and its pages; only the decode round reads it.
- **The new numbers are named constants, not flags:** `KV_GROWTH_STEP_PAGES
  = 32` and `KV_ENTRY_HEADROOM_STEPS = 4`. They become flags if the owner asks.

## Testing Decisions

- Pure plan tests in `vram.rs` and `residency/plan.rs` cover every branch,
  boundary and spelling (ACs 1-7).
- Config resolution tests in `config.rs` cover ACs 5, 6, 10, 12 and 39.
- The default cap: scheduler tests in `concrete.rs` over `MockCompute`, and
  the HTTP tests for the three request shapes (AC 11).
- CPU scheduler tests in `crates/core/tests/` run over `MockCompute` with a
  fake disk and a mirrored page ledger (ACs 17-22, 26-27, 30-33).
- The page-shortage victim order is a pure function, tested key by key in
  `admission.rs` (AC 30), as `retained_lane_is_better_victim` is.
- Existing tests that assert a whole reservation at admission, or a
  same-class newcomer evicting an older lane for pages, are updated to the new
  rule, not deleted. Each says which rule it now pins.
- The store's unit tests run on a temp directory: the format, CRC, lock and
  margin (ACs 13-15). The free space is injected, not read from the real
  volume.
- The kernel CTests are `test_seq_snapshot`, extended to windows (AC 16) and
  to grown sequences, and `test_seq_alloc`, extended to growth (AC 28).
- GPU tests are `--ignored`, under the GPU profile (ACs 8-9, 23-25, 29,
  34-35). Their measurements go to findings, per `docs/findings/README.md`.
- web: vitest for the snapshot, derive and view of the disk row (AC 37).

## Phases

Three phases, each a separate agent, each ending with ACs 39 and 40. Phases
are the coordinator's work packages, not tickets.

| Phase | Owns | Closes | Depends on |
|---|---|---|---|
| **P1 — the pool policy, the floor opt-in, the default cap** | `crates/core/src/vram.rs`, `crates/core/src/residency/plan.rs`, `crates/core/src/flash_next.rs`; `crates/server/src/runtime.rs` (the plan calls, the cap into `SchedulerConfig`); `crates/server/src/config.rs` (`--kv-pool-bytes` token form, `--allow-expert-cache-below-floor`, `--default-max-tokens`); `crates/core/src/concrete.rs` (only `SchedulerConfig::default_max_tokens`, `generation_budget`, the write in `submit`); `mk/config.mk`, `Makefile`; `docs/user/README.md` (its rows) | 1-11, 38 (P1), 39, 40 | nothing |
| **P2 — Tier 2, KV-disk** | `kernel/` (windowed snapshot and restore, `test_seq_snapshot`); `crates/artifact/src/direct.rs`; `crates/runtime/src/lib.rs` and a new `kv_disk` module; `crates/core/src/{scheduler.rs, mock.rs, disk.rs (new), host.rs, concrete.rs (the chain), checkpoint.rs, prefix.rs, ngram_table.rs, types.rs}`; `crates/server/src/{telemetry.rs, metrics.rs}`; `crates/server/src/config.rs` (`--kv-disk-*` only); `crates/server/src/runtime.rs` (the store, the staging line); `web/src/monitor/`, `web/mockMetrics.ts`; ADR 0017's table; make knobs and README rows for the tier | 12-25, 36, 37, 38 (P2), 39, 40 | nothing; it can run beside P1 |
| **P3 — page-wise reservations and live moves** | `kernel/src/seq.cu`, `kernel/src/seq_prefix.cu`, `kernel/src/seq_checkpoint.cu`, `kernel/include/ignis_seq.h` (allocation at the reservation, `ignis_seq_grow`), `test_seq_alloc`; `crates/core/src/seq.rs` (`Seq::grow`); `crates/core/src/{concrete.rs (reservation, growth, parking, entry rule, last resort), admission.rs (rank and the page-shortage order), scheduler.rs and mock.rs (`grow`, the mirrored ledger), types.rs (`Parked`)}`; `crates/runtime/src/lib.rs` (`grow`, allocation and restore at the reservation); `crates/server/src/{telemetry.rs, metrics.rs}` (`ignis_kv_lane_parks_total`); ADR 0017's row; the README paragraph | 26-35, 38 (P3), 39, 40 | P2 and P1 |

- **P1 first slice.** The default cap (ACs 10-11, and the README text of AC
  38) needs no GPU. It can be closed on its own before the plan work.
- **P1 and P2 share four files** (`config.rs`, `runtime.rs`, `mk/config.mk` with
  the `Makefile`, and the README). Each touches its own flags, functions and
  rows there, so the hunks are additive, and whichever merges second rebases.
  P1's `concrete.rs` hunk is in `SchedulerConfig` and `generation_budget`,
  away from P2's chain.
- **P2's internal order.**
  1. The kernel windows and `DirectWriter`.
  2. The store.
  3. The seam and the scheduler chain, CPU-tested.
  4. The wiring, the metrics and the Monitor.
  5. The GPU ACs.
- **Why P3 follows P2.**
  - Both rewrite `kernel/src/seq.cu`, `concrete.rs`'s eviction and restore
    paths, and the `Compute` seam with its `MockCompute`. Run together, they
    would conflict at every hunk.
  - A live move to disk needs P2's transfer states.
  - P3's CPU tests reuse P2's fake disk.
  - It follows P1 for the default cap, which its tests and AC 9's load
    assume.

## Out of Scope

- **Changing `--prefill-chunk`'s defaults** (8,192 on Flash-Next), MTP, and
  the n-gram table itself.
- **Raising `max_in_flight` above the lane count** so that a burst queues
  instead of getting a 503. The fan-out width in `decide.rs` reads the same
  number. Left to the owner, unchanged on 2026-10-08.
- **Moving sequences at a pool low-water mark** (pre-emptive). One step of
  room ahead per lane is the mitigation.
- **Windowing the KV-RAM moves.** A follow-up if AC 35's synchronous leg
  misses its bound.
- **Changing the lane-shortage order** (the head's lane deal).
- **Flags for the growth step and the entry headroom.** They are named
  constants until the owner asks.
- **A Monitor view of parked lanes.** The counter is on `/metrics`.
- **Reusing disk blobs across a restart or a model switch.** The header
  already carries what that would need.
- **Deduplicating prefix bytes across blobs** (ADR 0024's chained blobs).
- **Dropping the host expert pool or the prefill staging ring on a resident
  Flash-Next load.** Spec flash-next/03 owns residency.
- **A completion-port IO backend.**
