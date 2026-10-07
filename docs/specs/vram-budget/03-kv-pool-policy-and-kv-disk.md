# 03 — The KV pool policy, the expert cache floor's opt-in, and the KV-disk tier

GitHub: the ticket that links here. ADRs: 0045 (this feature), 0030 (the VRAM
plan it amends), 0029 (the residency tiers; Tier 2 is built here), 0024 (the
blob, now moved a window at a time), 0023 (the eviction priority, one tier
further down), 0017 (the metric contract), 0022 (pages derived from the
format). Evidence:
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
- **Tier 2, KV-disk, below KV-RAM.**
  - A blob that KV-RAM gives up goes to disk, and so does a device victim that
    KV-RAM cannot take.
  - A blob comes back from disk straight to the device.
  - With the tier on, no live work is discarded for room. When no tier has
    room, the request that needed it waits.
  - On by default for Flash-Next (16 GiB); off by default for the 27B.

ADR 0045 holds the reasons and the rejected alternatives. This spec holds the
seams, the defaults and the acceptance.

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

## Defaults

Items marked *(owner)* are the owner's decisions of 2026-10-07. Items marked
*(agent)* are proposals the owner confirms or changes; ADR 0045 gives the
reason for each.

| Point | Default | |
|---|---|---|
| Offloaded pool | `max(floor, min(524288, decode_lanes × --max-context))` tokens | *(owner)* size, *(agent)* cap and floor |
| Resident test | budget ≥ fixed lines + residency's fixed lines + every expert projection + floor pool | *(agent)* |
| `--kv-pool-bytes` | bytes, or `<n>[K\|M]tok`; honoured on both models | *(agent)* |
| Floor opt-in | `--allow-expert-cache-below-floor`, WARN; class minimum stays a refusal | *(owner)* |
| KV-disk on | Flash-Next `--kv-disk-bytes 16G`; 27B `0` | *(owner)* build, *(agent)* sizes |
| Location | `--kv-disk-path model` (beside the artifact), `auto`, or a directory | *(agent)* |
| Volume margin | 10 GiB, `KV_DISK_VOLUME_MARGIN_BYTES` | *(agent)* |
| Placement | one file per blob, a byte ledger in eviction-priority order | *(agent)* |
| Restore floor | Flash-Next 8,192 tokens, 27B 16,384 tokens | *(agent)* |
| Transfer | 32 MiB windows, two-window pinned staging (64 MiB), two IO threads | *(agent)* |
| n-gram first | no new tier request while a prefill's n-gram gather is pending | *(agent)* |
| Restart | nothing reused; per-process directory, stale ones removed at start | *(agent)* |
| Integrity | CRC32 per window, header page written last as the commit | *(agent)* |
| No room | the request waits in the admission queue; the in-flight cap's 503 is unchanged | *(owner)* |

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
- **The 27B load (`scheduler`).** The same plan call with a whole residency,
  and the KV-disk store when named.
- **Config — `crates/server/src/config.rs`.**
  - `--kv-pool-bytes` parses into bytes or tokens; the 27B-geometry validation
    moves out of `resolve_kv_pool_bytes`.
  - New: `--allow-expert-cache-below-floor`, `--kv-disk-bytes`,
    `--kv-disk-path`, and their environment variables.
  - Per-model defaults and refusals live in `served_model_for` and
    `EngineShape::for_family`.
  - Help text: `--kv-pool-bytes`, `--decode-lanes` (a lane no longer holds its
    own whole context) and the new flags.
- **The scheduler — `crates/core/src/concrete.rs`, `crates/core/src/host.rs`,
  a new `crates/core/src/disk.rs`, `crates/core/src/checkpoint.rs`,
  `crates/core/src/prefix.rs`.**
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
- **The `Compute` seam — `crates/core/src/scheduler.rs`.**
  - New calls: start a spill to disk, from the device or from a KV-RAM blob;
    advance in-flight transfers by at most one window each; report finished
    and failed transfers; start a restore from disk; discard a disk blob; ask
    whether a blob of N bytes fits the disk.
  - `MockCompute` implements a fake disk with configurable latency, capacity
    and failures.
- **The runtime — `crates/runtime/src/lib.rs`, plus a `kv_disk` module.**
  - The store: directory lifecycle and lock, file format, the two IO workers,
    the staging windows, CRC.
  - `RuntimeCompute` implements the new seam calls.
  - `StepLeaf` gains windowed snapshot and restore: whole-sequence,
    checkpoint and prefix blobs, read or fed a window at a time on the leaf's
    transfer stream, with completion polled.
- **The leaf — `kernel/`.**
  - The snapshot and restore calls, for sequences, checkpoints and prefixes,
    take a transfer-options struct (ADR 0016, ADR 0024's "partial extent"):
    a byte window and the stream to use. A null options pointer is today's
    whole-blob call.
  - A restore keeps a cursor. A sequence whose restore is incomplete refuses
    every step and can be released.
- **Direct IO — `crates/artifact/src/direct.rs`.** A `DirectWriter` twin of
  `DirectReader`: positional, unbuffered, aligned writes.
- **The n-gram table — `crates/core/src/ngram_table.rs`.** Exposes whether a
  prefill gather is pending, which the tier's workers check before issuing a
  request.
- **Metrics — `crates/core/src/types.rs` (`SchedEvent`),
  `crates/server/src/telemetry.rs`, `crates/server/src/metrics.rs`, ADR 0017's
  table.**
  - New facts: a disk spill (with its source tier) and a disk failure (write
    or read). The tick carries the disk's used bytes.
  - The retained-state facts carry `ReuseSource::Disk`.
- **The Playground Monitor — `web/src/monitor/`** (snapshot contract, derive,
  view, their vitest suites) and `web/mockMetrics.ts`.
- **Make and docs.**
  - `mk/config.mk` and the `Makefile`: new knobs `KV_DISK_BYTES`,
    `KV_DISK_PATH` and `ALLOW_EXPERT_CACHE_BELOW_FLOOR`.
  - `make config` prints the pool policy, the pool's tokens, and the disk
    directory and budget.
  - `docs/user/README.md`, and `CONTEXT.md` (updated with ADR 0045).

## Acceptance criteria

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
   - On Flash-Next a named pool replaces the default on both branches, and is
     checked against Flash-Next's own per-token cost at the plan.
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
9. **GPU: agent-shaped load on the default plan.** Three concurrent requests
   without `max_tokens`, so each reserves the whole context:
   - Every request completes. None is refused, and none loses work:
     `ignis_kv_ram_evictions_total` and
     `ignis_kv_disk_failures_total{op="read"}` stay 0.
   - The finding records how many ran at once, the spills to each tier, the
     transfers' times and the aggregate tok/s against A.
   - No speed bound: this is the trade in ADR 0045's first consequence, left
     to the owner.

### KV-disk: configuration and lifecycle

10. **Flags and defaults.**
    - `--kv-disk-bytes` and `IGNIS_KV_DISK_BYTES` default to 16 GiB on
      Flash-Next and `0` on the 27B; `0` turns the tier off.
    - `--kv-disk-path` and `IGNIS_KV_DISK_PATH` take `model`, `auto` or a
      directory, resolved as `ngram_cache::CacheLocation` resolves them.
      `auto` is `LOCALAPPDATA/ignis/cache/kv-disk` on Windows, and
      `XDG_CACHE_HOME/ignis/kv-disk` or `HOME/.cache/ignis/kv-disk` on Linux.
    - The flag wins over the environment.
    - A 27B load at its defaults creates no directory and writes nothing.
11. **Directory lifecycle.**
    - A load writes under `<location>/ignis-kv-disk/<pid>-<nonce>/` and holds a
      lock file there, open exclusively, for its life.
    - Two processes on one location: the second leaves the first's directory
      untouched.
    - At start, a directory whose lock can be taken (its owner is gone) is
      removed.
    - A clean shutdown removes the process's own directory.
    - No file written by another process is ever read.
12. **The budget and the volume.**
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

13. **The file format.** A file is a 4 KiB header page, written last, followed
    by the blob's windows. The header carries:
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

14. **Kernel test: windowed equals whole.**
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

15. **The chain.**
    - A device victim goes to KV-RAM when it fits.
    - A KV-RAM victim goes to disk instead of being discarded: a live one
      always, a retained one when it outranks the disk's lowest entry.
    - A device victim goes straight to disk when the arena is 0, or when it
      cannot make room.
    - The disk discards in the extended order: retained before live, then
      class, then probation before protected, then least-recently-used, with
      the Interactive TTL.
    - With the tier on, no `SnapshotDropped` is ever emitted.
16. **No room.**
    - Fill every tier with live or higher-ranked state. A new request under the
      in-flight cap is then held in `Admitted`: not refused, and nothing is
      discarded.
    - It is admitted at the first advance after a lane completes.
    - A request beyond the in-flight cap is still refused `Full`, as today.
17. **The restore floor.**
    - A disk match is taken only when it beats the best device and KV-RAM
      match by the family's floor: 8,192 tokens on Flash-Next, 16,384 on the
      27B.
    - A tie goes to the tier above.
    - A live disk blob is restored whenever room returns, with no floor.
18. **The model thread never waits on the disk.** With a fake disk that takes
    1 s per window:
    - The other lanes' decode rounds keep advancing during a spill and during
      a restore.
    - Each advance copies at most one window per transfer.
    - The victim's pages stay charged until its file commits.
    - A restoring request is not scheduled before its last window lands.
    - A write that fails leaves the victim resumable on the device, and the
      request that wanted the room still waiting.
19. **Cancel mid-transfer.** A request cancelled mid-spill or mid-restore
    releases its pages, its file is deleted, and the ledgers' used bytes return
    to their value before the transfer.
20. **The n-gram table goes first.** While a prefill gather is pending (a fake
    table flag in the CPU test), no new tier request is issued. A request
    already issued completes its window.

### KV-disk on the GPU

21. **Flash-Next: a forced overflow restores bit-exact.** Setup:
    - `--decode-lanes 2`, `--kv-pool-bytes` at the floor,
      `--kv-host-pool-bytes 0`, the disk tier on.
    - Request A (class `agent`, no `max_tokens`, so it reserves the whole
      context) prefills and decodes alone.
    - Request B (`interactive`) arrives, evicts A straight to disk, and runs
      alone to completion.
    - A restores from disk and finishes alone.

    A's greedy tokens are identical to a run of A alone with the same prefill
    chunking. Every decode round is at width 1 in both runs, so a width change
    cannot explain a difference (finding 2026-09-14, batched decode width
    drift).

    A second leg uses a small KV-RAM arena: A goes to KV-RAM, a third request
    C demotes it to disk, and the same equality holds.
22. **The 27B: the same overflow.** The test of 21 runs on the 27B with
    `--kv-disk-bytes` named: the mechanism is generic.
23. **Contention on F:, measured.** On Flash-Next, force a ≥ 1 GB spill to disk
    twice: once during a prefill chunk with an uncovered n-gram table, and once
    while two lanes decode. Compare against the same work without the spill.
    - The chunk's wall time and its n-gram gather time are recorded, and so are
      the decoding lanes' ITL p50 and p99.
    - Starting bounds, for the owner to confirm: chunk wall time and ITL p50
      each within +10% of the run without the spill.
    - The result goes in a finding.

### Observability, docs and the 27B

24. **The metric contract.**
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
25. **The Monitor.**
    - Its Disk row is live when the scrape carries the tier: live spills,
      retained discards, and the demotions into the disk.
    - A capacity and used bar shows the disk beside KV-RAM's.
    - The row reads "off on this load" when the series are absent.
    - `assessHealth` weighs a disk read failure like a KV-RAM live drop.
    - vitest covers a fixture scrape with the tier and one without, and the
      mock simulator feeds the row.
26. **Make and docs.**
    - New knobs: `KV_DISK_BYTES`, `KV_DISK_PATH` and
      `ALLOW_EXPERT_CACHE_BELOW_FLOOR`. `make config` prints them, with the pool
      policy.
    - `docs/user/README.md` documents the flags, the token form, the policy and
      the tier.
    - The `--decode-lanes` help text no longer says each lane holds its own
      whole context.
27. **The 27B is unchanged at its defaults.**
    - Config resolution is identical but for the new fields.
    - The plan is byte-identical (AC 1).
    - No disk directory is created.
    - The 27B GPU profile is green.
28. **The suites are green.**
    - `cargo test` passes workspace-wide.
    - `cargo check --workspace --features cuda --tests` is clean.
    - `npm test` passes in `web/`.
    - The GPU tests above are green on a free 5090, the card checked free
      first (AGENTS.md).

## Implementation Decisions

- **The policy is one function for both models.** Flash-Next's hand-rolled
  plan in `runtime.rs` goes through `plan_vram`. The 27B gets
  `Residency::Whole` and so takes today's branch by construction.
- **Tokens are converted to pages by the plan, never by the config.** The
  config cannot know the model, which is what made today's validation wrong
  for Flash-Next.
- **The disk ledger mirrors `HostTier`** (entries, owner class,
  probation/protected, use tick). The victim order is shared code, not a copy,
  so the two tiers cannot drift.
- **Transfers are scheduler states, not blocking calls.** The KV-RAM path
  stays synchronous: it runs at ~12 GB/s and its cost is measured (ADR 0024).
  Only disk transfers are pumped.
- **The staging is reserved at load** (ADR 0030: serving allocates nothing).
  It is a host-plan line on Flash-Next and part of the tier's open on the 27B.
  The arena's spans are aligned to `DIRECT_IO_ALIGNMENT`, so that KV-RAM to
  disk writes straight from them.
- **The tier's IO threads are the tier's own,** not the n-gram reader's: one
  pool per purpose, so a slow write never occupies a gather's thread.

## Testing Decisions

- Pure plan tests in `vram.rs` and `residency/plan.rs` cover every branch,
  boundary and spelling (ACs 1-7).
- Config resolution tests in `config.rs` cover ACs 5, 6, 10 and 27.
- CPU scheduler tests in `crates/core/tests/` run over `MockCompute` with a
  fake disk (ACs 15-20).
- The store's unit tests run on a temp directory: the format, CRC, lock and
  margin (ACs 11-13). The free space is injected, not read from the real
  volume.
- The kernel CTest is `test_seq_snapshot`, extended to windows (AC 14).
- GPU tests are `--ignored`, under the GPU profile (ACs 8-9, 21-23). Their
  measurements go to findings, per `docs/findings/README.md`.
- web: vitest for the snapshot, derive and view of the disk row (AC 25).

## Out of Scope

- **Changing `--prefill-chunk`'s defaults** (8,192 on Flash-Next), MTP, and
  the n-gram table itself.
- **Reservations that grow page by page.** They supersede core-05. This is
  ADR 0045's first consequence, left to the owner.
- **Raising `max_in_flight` above the lane count** so that a burst queues
  instead of getting a 503. The fan-out width in `decide.rs` reads the same
  number. Left to the owner.
- **Reusing disk blobs across a restart or a model switch.** The header
  already carries what that would need.
- **Deduplicating prefix bytes across blobs** (ADR 0024's chained blobs).
- **Dropping the host expert pool or the prefill staging ring on a resident
  Flash-Next load.** Spec flash-next/03 owns residency.
- **A completion-port IO backend.**
