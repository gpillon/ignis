# ADR 0045 — the KV pool is sized by whether the model is resident, and evicted state goes down to disk

## Status

Accepted (2026-10-07). The owner decided the pool policy and its 524,288-token
default, kept the expert cache floor as a refusal with an opt-in to start below
it, and asked for Tier 2 to be built. Items marked *(agent proposal)* are
defaults the owner confirms or changes; the others are the owner's. Spec:
`docs/specs/vram-budget/03-kv-pool-policy-and-kv-disk.md`.
**Amends ADR 0030** (the KV pool line, the expert cache floor), **ADR 0029**
(Tier 2 is built), **ADR 0024** (a transfer may move a window at a time),
**ADR 0023** (the eviction priority reaches one tier further down; a page
shortage takes what is not decoding first), **ADR 0017** (`tier="disk"` is
exported on a load that has the tier) and **spec core-05** (a reservation
grows).

**Amended 2026-10-08 (owner).** Four decisions replace the open point this ADR
first left to the owner (two whole-context reservations fill the default
pool):

- Reservations grow by pages, and a live sequence moves down a tier
  mid-generation and resumes bit-exact. Built now, in the same spec.
- A server default `max_tokens` of 38,912 on both models, `0` for today's
  behaviour.
- The KV-disk location follows the n-gram cache's rule (confirmed).
- A PCIe contention criterion for a live move.

Kept: the in-flight cap, refusals at plan time, and the spec's F: contention
thresholds as starting values. The numbers in the new sections marked
*(agent proposal)* are for the owner to confirm.

Sources: [Flash-Next on the 5090](../findings/2026-10-06-flash-next-on-the-5090.md)
(its 2026-10-07 update: lanes at 262K), the F: disk bench
(`.scratch/disk-bench/RESULTS.md`, 2026-09-29; untracked, local to the owner's
clone), GitHub #205 (the owner's Tier 2 header decision and its RoPE comment).

## Context

- **The 27B's pool takes the rest** of the VRAM budget (ADR 0030). Every weight
  is on the device, so the rest has no other use.
- **Flash-Next's pool is every lane's whole context.** The expert cache takes
  the rest (ADR 0030 as amended by spec flash-next/03), and the pool is sized
  `ceil(max_context / 64) × decode_lanes + retained_slots` pages
  (`EngineOptions::pool_budget`). At the defaults, 262,144 tokens × 3 lanes,
  that is 786K tokens: 3.46 GiB with the lanes' state.
- **The pool and the expert cache trade byte for byte, and the cache sets
  decode speed.** Measured at 262K per lane, one lane active:

  | `--decode-lanes` | KV pool | expert cache | decode tok/s | decode hit rate |
  |---:|---:|---:|---:|---:|
  | 3 | 3.46 GiB | 13.38 GiB | 85.3-89.5 | 96.25% |
  | 1 | 1.15 GiB | 15.74 GiB | 98.5-99.3 | 97.55% |

  The lane count is today's only knob between the two, and it moves VRAM only
  by taking a whole context away.
- **`--kv-pool-bytes` names the 27B's pool only.** A Flash-Next load ignores
  it, and the config validates it against the 27B's geometry whichever model
  loads (`resolve_kv_pool_bytes`).
- **Below 12 GiB the expert cache refuses the start**
  (`EXPERT_CACHE_FLOOR_BYTES`), with no way past it.
- **Tier 2 was prepared, not built** (ADR 0029). When KV-RAM must make room it
  discards: retained state first, then an evicted live sequence. That request
  re-prefills from zero (`ignis_kv_ram_evictions_total`, "the most expensive
  event in the system", ADR 0017). With `--kv-host-pool-bytes 0`, a Flash-Next
  load cannot evict at all: its snapshot buffers come from the arena.
- **The host is RAM-poor and NVMe-rich.**
  - Flash-Next pins 37.8 GB of experts. With ~48 GB available, the KV-RAM
    arena fits at 0.5-2 GiB; on 2026-10-07 the host plan refused 2 GiB by
    0.66 GB.
  - A whole-context Flash-Next lane's blob is ~1.25 GB: 4,224 B/token paged
    (KV 3,456, indexer keys 768) plus a 130 MB state image.
  - F: is an NVMe behind the chipset at Gen3 x4, ~96% full. Unbuffered, it
    reads ~2.9-3.0 GB/s and writes 1.2-1.5 GB/s, with garbage-collection
    stalls up to ~150 ms.
  - A full 8,192-token Flash-Next chunk takes ~3.2 s, ~1.7 s of it expert
    copies; even a traversal of a few tokens re-streams 0.13-0.66 s of experts
    ([the agent turn tail](../findings/2026-10-07-flash-next-agent-turn-tail.md)).
- **The n-gram table reads its rows from the same NVMe** while a prompt
  prefills (#306).
- **A reservation is whole and never grows** (spec core-05). At admission a
  request reserves `ceil((prompt + effective_max) / 64)` pages, and without a
  `max_tokens` `effective_max` is what the prompt leaves of `--max-context`.
  The leaf maps every page at `ignis_seq_alloc`.
  - The vendored allocator already separates the two: `PagedKVPool` counts an
    entitlement (`can_reserve`, `set_page_entitlement`) apart from the pages
    it maps (`materialize_pages`). Each block-table row is already sized for
    `--max-context`.
- **The 27B already moves live sequences.** `snapshot_and_evict` snapshots a
  decode lane (`ResumePhase::Running`) or a resident lane-less request
  (`ResumePhase::Prefilling`) into KV-RAM, and `restore_pass` brings it back.
  Today it runs only when an admission needs pages or a prefilled head needs
  a lane. A lane holder goes before a lane-less request (ADR 0023, #190). Every decode lane carries the same use
  tick, so least-recently-used reduces to lane id.
- **PCIe is full duplex, and the expert stream uses one direction.** At
  ~100 tok/s Flash-Next moves 66-74 MB per decode token host to device, that
  is ~6.6-7.4 GB/s of the link's ~12 GB/s. A sequence moving out (device to
  host) runs against the opposite direction; a sequence moving in shares the
  experts' direction.
- **Qwen recommends 38,912 tokens of output** for complex tasks, reasoning
  included.

## Decision

### The KV pool follows residency

**The pool is sized by whether every weight is on the device (owner).**

- **Resident.** Every weight is on the device: the 27B always, and Flash-Next
  when the budget holds every expert projection beside the minimum pool (a
  card around 96 GB). The expert cache, if any, holds the whole expert pool,
  and **the KV pool takes the rest**: today's 27B rule, unchanged.
- **Offloaded.** Flash-Next's experts stream over PCIe, as on the 5090. **The
  KV pool is reserved first at a default size, and the expert cache takes the
  rest.**
- **The test is plan arithmetic**, done before anything is allocated. A load is
  resident when the budget holds its fixed lines, residency's fixed lines,
  every expert projection and the minimum pool.
- **The offloaded default is 524,288 tokens (owner)**, capped and floored
  *(agent proposal)*:

  `pages = max(floor, min(8192, decode_lanes × ceil(max_context / 64)))`,
  with `floor = ceil(max_context / 64) + retained_slots`.

  - The cap: a pool larger than every lane's context can hold only retained
    pages. At `--decode-lanes 1` it keeps today's 262K pool, so a one-lane
    load's expert cache does not shrink.
  - The floor wins, so `--max-context 524288` starts instead of being refused.
- **The pool is shared by the lanes and paged** (64-token pages on both
  models, as today). A lane now costs its state, not a context, and a request
  costs the pages it has used, not the ones it might (below).
- **`--max-context` bounds every request.**
- **Invariant:** the pool holds one `--max-context` sequence and one page per
  retained slot (ADR 0030's floor, now on both models). A smaller pool refuses
  the start, naming the knobs.

### Reservations grow by pages, and a live sequence moves down a tier

**A request reserves what it has used plus a step, and takes pages as it
generates. When the pool runs out, the lowest-ranked sequence moves down a
tier and later resumes where it stopped: no work lost, never a 503 (owner,
2026-10-08).** Both models, one mechanism.

- **The bound and the reservation are two numbers.**
  - The *bound* is the prompt plus the request's generation cap (below),
    never past `--max-context`. A submission is still refused against it:
    `ContextExceeded` past the context, `Oversized` past the pool alone.
    Because the pool holds one `--max-context` sequence beside the retained
    pages, **a sequence alone always fits**: with retained state given up and
    every other sequence moved, any one bound fits.
  - The *reservation* is what the pool has given the request. At admission it
    is the prompt (the tail past a claimed prefix, as today) plus **one
    growth step**.
- **The growth step is 32 pages, 2,048 tokens, on both models** *(agent
  proposal)*.
  - From its first round a decode lane keeps **at least one step of room
    ahead** of its position. When its room drops below a step, it asks for one
    more, capped at its bound.
  - The step is small against the pool: 32 of 8,192 pages, so three lanes
    over-reserve at most ~2.3% (two steps each).
  - It is large against a move. A step is ~20 s of decode at ~100 tok/s, so a
    victim's disk spill (~1 s for a whole-context Flash-Next blob) completes
    while the lane still decodes on its room.
  - A prompt never grows: admission reserves all of it.
  - A request whose bound is within one step of its prompt (a decision, a
    constrained decode, a small `max_tokens`) reserves its whole bound at
    admission and never grows, exactly as today.
- **A lane that runs out parks.** When its room is below one round's largest
  append (1 token, or the draft window plus one under speculation) and no page
  can be had yet, the lane is held out of decode rounds. It keeps its pages
  and resumes at the first advance with room. A parked lane is not decoding,
  so it is among the first victims of a higher-ranked need.
- **Two triggers, both reactive** (ADR 0023: eviction runs on the refusal
  path): a lane asking for its next step, and an admission materializing.
  A restore never triggers a move (below).
- **A need is met in this order.**
  1. Free pages.
  2. Retained state on the device, given up first (ADR 0023 as amended by
     0029).
  3. A live sequence that **ranks below the requester**, moved down a tier:
     KV-RAM, else KV-disk, at a round or chunk boundary.
     - **Rank:** class first (`Interactive` above `Agent`), then submission
       order (earlier above later). A moved sequence keeps its rank.
     - **Order among eligible victims:** `Agent` before `Interactive`; then
       sequences not in the decode round before lane holders (owner: a
       prefilled request waiting for a lane, a half-prefilled one at a chunk
       boundary, a parked lane); then the latest-submitted.
     - **Never** the requester, a sequence mid-transfer, or one the admission
       machine protects (a protection donor, a lane reserved for an earlier
       Interactive request).
  4. Nothing left: an admission waits in the queue (as when no tier has
     room), and a growing lane parks.
- **Why rank gates the victim** *(agent proposal)*. "Never the requester" on
  its own would let an `Agent` lane's growth move an `Interactive` lane. It
  would also let two equal lanes move each other in turn.
  - With rank, an `Agent` need never moves an `Interactive` sequence.
  - Within a class it is first in, first out: an older sequence may move a
    younger one, never the reverse. A newcomer of a class therefore waits
    behind the sequences of its class already admitted, and is not placed by
    moving one of them, as an admission does today.
  - The top-ranked live sequence can move every other one, so it always
    progresses, and each sequence eventually becomes the top.
- **A lane shortage keeps today's order** (ADR 0023 as amended by #190): a
  lane holder before a lane-less request, since moving a lane-less request
  frees no lane. Only a shortage of pages takes what is not decoding first.
- **Entry rule, the anti-thrash hysteresis** *(agent proposal)*.
  - A restore never moves anything. A moved sequence comes back only into
    free room, retained pages counted as reclaimable: `restore_pass` already
    breaks when there is none.
  - Restores and admissions share one rule. A sequence enters the device when
    the free pages cover its reservation plus **four growth steps** (8,192
    tokens), for it and for every resident live sequence that ranks above it.
    For a sequence closer than that to its bound, the room counted is what it
    can still grow.
  - Entries are taken in rank order. A moved sequence therefore comes back
    before any newcomer that ranks below it, and a stream of short newcomers
    cannot keep it out.
  - Why it is enough: everyone above a returning sequence can grow about four
    steps before the pool is short again, so two equal sequences that cannot
    both fit swap at most once per ~8,192 tokens of the younger one's
    progress. At the default pool there is no cycle at all: 8,192 pages are
    exactly two 262,144-token sequences, so at three lanes at most one
    whole-context sequence is ever below the device, and it returns when one
    of the two finishes.
- **The last resort.** Suppose every resident live sequence is parked and no
  tier can take a victim: the disk refuses writes, or no tier exists. Then the
  lowest-ranked parked sequence is discarded and re-queued (re-prefill,
  today's KV-RAM loss), with an ERROR naming the tier that refused, as a
  `SnapshotDropped`. On a load with KV-disk and room on it, this never
  happens.
- **The 27B without KV-disk keeps today's guarantee, no stronger.** KV-RAM
  still discards a live blob to take a newer victim (re-prefill). That can now
  follow a growth as well as an admission.
- **The leaf** *(agent proposal)*.
  - `ignis_seq_alloc` and its two siblings (`_shared`,
    `_from_checkpoint`) map the reservation they are given, prompt plus a
    step, instead of the bound.
  - A new `ignis_seq_grow(pool, seq, context_tokens)` raises the entitlement
    (`set_page_entitlement`) and maps the new pages (`materialize_pages`),
    zeroes them as an allocation does, and publishes the block-table range,
    between rounds on the model stream.
  - It refuses with the pool unchanged when the pages are not free, past
    `--max-context`, and on a sequence whose windowed restore is incomplete.
  - A restore maps the blob's tokens plus one step.
  - A sequence grown step by step is byte-identical to one mapped whole.
- **Mechanism already built.** The move is the 27B's `snapshot_and_evict`
  (KV-RAM, synchronous) or the disk tier's windowed spill (Tier 2, below).
  The return is `restore_pass`. Growth adds a third caller beside admission
  and the head's lane deal.

### A server default for `max_tokens`

**A request that names no cap generates at most 38,912 tokens, its reasoning
included (owner, 2026-10-08).**

- `--default-max-tokens <n|0>` (`IGNIS_DEFAULT_MAX_TOKENS`, make
  `DEFAULT_MAX_TOKENS`) *(agent proposal: the name)*. The default is 38,912
  on both models; `0` is today's behaviour, up to the context. The flag wins
  over the environment.
- **It is the request's generation cap when the request sends none** of
  `max_tokens`, `max_completion_tokens` (chat) or `max_output_tokens`
  (`/v1/responses`).
  - It is clamped to what the prompt leaves of `--max-context`. A long prompt
    is never refused for it; only a prompt that fills the context alone is.
  - Reaching it ends the request `finish_reason: "length"` (`incomplete` with
    `max_output_tokens` on `/v1/responses`).
  - Reasoning tokens count inside it, as they do inside `max_tokens`. The
    default thinking budget (6,144) and its 2,048-token answer reserve sit
    well within it.
- **An explicit cap always wins.** An explicit cap is still bounded by
  `--max-context`: past it, it is refused as today. `ignore_eos` still needs an
  explicit `max_tokens`. A decision or a constrained decode keeps its own
  budget.
- **One home** *(agent proposal)*: `SchedulerConfig::default_max_tokens`, read
  by `generation_budget`. Every entry point gets it, and the request's
  `remaining_work`, which the admission machine's frontier distance reads,
  becomes finite.

### `--kv-pool-bytes` names the pool on both models

**The existing flag, honoured on both models and both branches, with a token
form *(agent proposal)*.**

- Named, it replaces the policy's size. On the offloaded branch it replaces the
  default. On the resident branch it replaces the minimum pool in the
  residency test and becomes the pool, and the rest of the budget is left
  unused, as a named pool leaves it on the 27B today.
- It accepts `<n>tok`, with `K`/`M` as binary multipliers: `512Ktok` is 524,288
  tokens. `IGNIS_KV_POOL_BYTES` takes the same spellings. A bare count or a
  `K`/`M`/`G` suffix stays bytes.
- **Why tokens.** Bytes per token differ by model and format: Flash-Next 4,224
  paged, the 27B 9,216 under hq-e8-2b and 65,536 under BF16. A byte figure is
  therefore a different context on every load. "512K tokens" means the same on
  every one, and it is the quantity the owner decided in. ADR 0022 still holds:
  the pages are what the plan derives, whatever the spelling.
- **Validation moves to the load's plan**, where the model is known. The config
  only parses.
- **Not a new flag.** One quantity gets one flag with two spellings, rather
  than two flags that must refuse each other.

### The expert cache floor stays, with an opt-in

**Below `EXPERT_CACHE_FLOOR_BYTES` (12 GiB) the start is refused by default
(owner).**

- `--allow-expert-cache-below-floor` (`IGNIS_ALLOW_EXPERT_CACHE_BELOW_FLOOR`)
  starts anyway. It emits a WARN naming the cache, the floor and the knobs that
  would lift it (owner), on the pattern of ADR 0030's
  `--allow-vram-oversubscription`.
- The class-minimum refusal is not overridable. A cache that cannot hold every
  K class's minimum (one decode step's selection and lookahead) cannot run a
  step.
- A 27B load refuses the flag, as it refuses `--decode-lanes`.

### Tier 2: KV-disk

**When KV-RAM must give a blob up, the blob goes to disk instead of nowhere.
When KV-RAM cannot take a device victim, the victim goes straight to disk
(owner).**

- **Generic.** Both models share one mechanism under the `Compute` seam.
- **Defaults *(agent proposal)*.** On for Flash-Next at 16 GiB; it holds about
  13 whole-context lanes or about 65 conversations of 30K tokens. Off for the
  27B, for four reasons:
  - its RAM is not taken by experts, and make gives it an 8 GiB KV-RAM arena;
  - its prefill runs ~8-9.6K tokens/s, so a crossing from disk pays only
    beyond ~16K tokens (below);
  - F: is nearly full;
  - its defaults stay byte-identical.

  One flag turns it on.
- **Flags.**
  - `--kv-disk-bytes <bytes|0>` (`IGNIS_KV_DISK_BYTES`); `0` turns the tier
    off, as `--kv-host-pool-bytes 0` does KV-RAM.
  - `--kv-disk-path <model|auto|dir>` (`IGNIS_KV_DISK_PATH`), with the n-gram
    cache's `CacheLocation` semantics: beside the artifact by default, the
    OS's per-user cache directory for `auto`, or a named directory (owner,
    confirmed 2026-10-08).
- **The budget is a ceiling, not a reservation *(agent proposal)*.** Nothing
  is preallocated, unlike the KV-RAM arena.
  - At start the effective budget is `min(flag, volume free − 10 GiB)`. It is
    printed, with a WARN when it is below the flag.
  - A write that would take the volume under the 10 GiB margin is refused, and
    so is a write that fails. The blob stays where it was.
  - The tier never refuses a start: it is a cache, like the n-gram one.
- **One file per blob *(agent proposal)*.** The files live in a directory the
  process owns. The filesystem places them, so there is no span to fit, unlike
  the pinned arena's first-fit. The tier is a byte ledger walked in the
  eviction priority's order, least recently used last within it.
- **Lazy.** A blob reaches disk only when the tier above gives it up, never as
  a copy of state still resident. It is written only from a snapshot point.
- **The eviction priority goes one tier down.**
  - Leaving KV-RAM, on a load with KV-disk, is a move to KV-disk.
  - Leaving KV-disk is a discard, in the same order: retained state before
    evicted live sequences, then request class, then probation before
    protected, then least-recently-used, with the Interactive TTL.
  - A spill into the disk displaces only what ranks below it.
- **No lost work.** With the tier on, an evicted live sequence is never
  discarded to make room, in KV-RAM or on disk. Only a failed read loses one.
- **When no tier has room, the request waits.** No victim is taken. The request
  that needed the room stays in the admission queue, held as `make_room`
  already holds a head, and is admitted when room frees. It is not refused,
  there is no 503 for it, and nothing is discarded. A growing lane in the same
  position parks.
  - Unchanged: the in-flight cap. `max_in_flight` is the lane count, and a
    request beyond it is a 503 `engine_full`. That cap concerns lanes, not
    tiers.
- **Restore goes straight to the device**, never through KV-RAM.
  - A live blob's file is deleted once it is restored.
  - A retained blob's file stays, because a claim never consumes (ADR 0029).
- **Restore floor *(agent proposal)*.** A disk match must beat the best match
  on the device and in KV-RAM by the family's floor, so a tie goes up. Live
  blobs have no floor: they come back whenever room returns.
  - **Flash-Next: 8,192 tokens**, one prefill chunk. Skipping them skips a full
    chunk, ~3.2 s. At ~2.9 GB/s even a whole-context blob reads in ~0.45 s, so
    the floor buys a margin of about 7x.
  - **27B: 16,384 tokens.** A blob is ~230 MB of image plus 9,216 B/token; it
    reads in ~0.08 s + 3.2 µs/token. Prefill costs ~0.11 ms/token. Break-even
    is ~750 + 0.03 × (blob tokens): ~8.6K tokens at a 262K blob and ~16K at a
    524K one, make's default context.
- **IO *(agent proposal)*.**
  - **Unbuffered positional IO** (`FILE_FLAG_NO_BUFFERING` on Windows,
    `O_DIRECT` on Linux), as the n-gram reader does (`DirectReader`, plus a
    writer twin). It runs on the tier's own two worker threads, a positional
    request each.
  - That gives the overlap without completion ports. Ports would need
    `windows-sys` ≥ 0.59, which does not build on this host's windows-gnu
    toolchain.
  - **Windows of 32 MiB** move through a pinned staging of two windows:
    64 MiB, a host-plan line present only with the tier. One window crosses
    PCIe while the other is written or read.
  - KV-RAM to disk writes straight from the arena span, with no staging.
  - **The model thread never waits on the disk, nor on a window copy.** It
    issues a window's copy on the leaf's transfer stream between steps and
    polls for it at the next advance.
  - **A device victim keeps its pages until its file is committed.** A failed
    write therefore leaves it intact on the device, and the request that
    wanted the room keeps waiting.
  - **A restored request rejoins** (a lane, or its prefill) at the first
    advance after its last window lands.
  - **The n-gram table goes first.** While a prefill's n-gram gather is
    pending, the tier issues no new request; one already issued finishes its
    window. Decode gathers do not hold the tier: they are small, and the tier
    keeps at most one request in flight per direction.
- **Restart *(agent proposal)*.** v1 reuses nothing across a restart. Each
  process writes into its own directory and holds a lock in it. At start, any
  directory whose lock is free (its owner is gone) is removed; at a clean
  shutdown, the process removes its own. Persistence is not nearly free:
  - Only retained state would be worth keeping; a live sequence dies with its
    connection.
  - Keeping it means rebuilding the retained index at start from the file
    headers: match key, kind, lineage, and at most two checkpoints per lineage.
  - The identity would then have to hold across loads, which #205 widens.
  - A wiped tier is a cache. A kept one is every conversation's state at rest,
    and it outlives the process and its crashes.
- **Integrity.** Every file opens with one 4 KiB header page, written last as
  the commit.
  - It carries GitHub #205's identity: the served model id; the blob identity
    (structural artifact hash, KV format, layout version, drafter presence and
    window); the sidecar's payload SHA-256 when present; and the RoPE scaling
    (the owner's #205 comment).
  - It also carries the blob's kind, match key, token count and length, a
    CRC32 per window, and its own CRC32.
  - Writing the full identity in v1 costs bytes, not work. A later
    persistence step, or phase 2's model switch, then needs no format change.
  - A file is never restored when its header is missing (a torn write), when
    any window fails its CRC, or when its identity differs from the load's. A
    live blob's request then re-prefills (an ERROR, counted). A retained blob
    is discarded.
  - CRC32 (crc32fast, already a dependency in `Cargo.lock`), not SHA-256. The
    check guards against torn and corrupt files, not adversaries: the
    directory is the process's own. It must also keep pace with 3 GB/s reads
    on a CPU without SHA extensions.

### Observability *(agent proposal)*

- **`tier="disk"` is exported** on the six retained-state families
  (`ReuseSource::Disk`; request-log spelling `disk`).
- New series:
  - `ignis_kv_disk_bytes{state="capacity"|"used"}` (gauge), beside
    `ignis_kv_ram_arena_bytes`;
  - `ignis_kv_disk_spills_total{from="device"|"kv_ram"}`: live snapshots
    written to the disk;
  - `ignis_kv_disk_failures_total{op="write"|"read"}`: a refused or failed
    write (nothing lost), or a read that failed its check;
  - `ignis_kv_lane_parks_total`: a decode lane parked for want of a page, once
    per entry into the parked state. It renders on every load, both models.
- **A live move is counted where it lands.** A move into KV-RAM is
  `ignis_kv_cache_evictions_total`, one straight to disk
  `ignis_kv_disk_spills_total{from="device"}`, whether an admission or a
  growth caused it.
- **The disk families render only on a load with the tier**, zeros included.
  A load without it never renders them, as a 27B load never renders the
  expert residency families.
- **The two unlabelled live counters keep their meaning.**
  `ignis_kv_cache_evictions_total` is device to KV-RAM only; a device victim
  that goes straight to disk is `ignis_kv_disk_spills_total{from="device"}`.
  `ignis_kv_ram_evictions_total` stays "dropped and lost"; a demotion to disk
  is not one.
- **The Playground Monitor's disk row goes live.** ADR 0017 reserved it as
  inert.

## Considered options

- **Rest-to-pool on Flash-Next too.** It starves the expert cache, which sets
  decode speed.
- **A whole context per lane (today).** It pays the cache's VRAM for contexts
  no lane is using.
- **A new `--kv-pool-tokens` flag.** Two flags for one quantity, each refusing
  the other.
- **Shrinking the default pool when the floor does not fit.** That silently
  cuts concurrency; the refusal names the knob instead.
- **Lowering the 12 GiB floor.** The floor is where decode collapses, so the
  operator must choose to go below it, knowingly.
- **One preallocated arena file for the disk.** NTFS zero-fills up to any write
  past the valid data length, and writing the whole budget at start takes
  ~15 s at 1.2 GB/s.
- **Staging a whole blob in pageable RAM.** ~1.25 GB at a time, on a host that
  cannot spare it.
- **Restoring disk → KV-RAM → device.** It needs arena room the host does not
  have, and it is still one PCIe crossing.
- **Writing to disk eagerly, at request end or on every spill.** It writes
  state that may never be needed and wears the drive.
- **SHA-256 per blob.** ~0.5 GB/s without SHA extensions, slower than the read
  it guards.
- **Persistence across a restart in v1.** See Restart above.
- **Disk on by default on the 27B.** See Defaults above.
- **Raising the in-flight cap so that a burst beyond the lanes queues.** That
  changes HTTP backpressure, is not a tier decision, and is left to the owner.
- **Accepting two lanes (owner, rejected 2026-10-08).** With whole
  reservations, two requests without `max_tokens` fill the default pool, and
  the third waits on the tiers. The third lane is decode capacity the card
  already pays for in lane state.
- **Raising the pool default (rejected 2026-10-08).** A third whole context
  costs the expert cache ~1.03 GiB, the trade the policy exists to undo, and
  it still holds pages no lane writes.
- **Only the default `max_tokens`.** It fixes the clients that send no cap,
  and it alone would let three of them run at once on the default pool. A
  large explicit cap, or `--default-max-tokens 0`, would still hold pages it
  may never write.
- **A newcomer moves an older sequence of its class (today's admission).** A
  stream of newcomers could then keep moving a sequence that has done the
  work. First in, first out within a class bounds that.
- **A minimum residency after a restore.** The sequence that needs the page
  would park behind a lower-ranked one: an inversion. The entry rule instead
  brings a sequence back only with room for everyone above it to grow.
- **Moving sequences at a pool low-water mark.** That is pre-emptive, which
  ADR 0023 rules out. One step of room ahead per lane gives a disk move the
  same time, without moving anything that nobody needs yet.
- **Growing a page at a time.** The pool packs no better, a move's spill has
  no time to finish, and the scheduler asks the leaf 32 times as often.
- **Discarding a live sequence for a growing lane.** That loses work. It is
  only the last resort, when every tier refuses.
- **Windowing the KV-RAM moves too.** Kept synchronous for now: ~12 GB/s, and
  a measured cost (ADR 0024). The PCIe criterion in the spec decides whether
  a whole-blob stall is too long.
- **Applying the default `max_tokens` in each HTTP handler.** Three surfaces
  would each re-derive the context clamp. `generation_budget` already does it
  once for every entry point.

## Consequences

- **Concurrency follows what lanes use, not what they might.** Before this
  amendment, two whole-context reservations filled the default Flash-Next
  pool, and a third request without `max_tokens` went through the tiers.
  - Now each request reserves its prompt plus a step. With the default cap of
    38,912, three agent requests without `max_tokens` fit at once even as
    whole bounds, for prompts up to ~135K tokens each.
  - When growth does fill the pool, the youngest sequence of the lowest class
    moves down and comes back when room returns.
- **Behaviour change for clients that send no cap.** Such a request stops at
  38,912 tokens with `finish_reason: "length"`; it used to run to the end of
  the context. A client that wants more sends `max_tokens`, and an operator
  restores the old behaviour with `--default-max-tokens 0`.
- **A same-class newcomer waits behind the sequences already admitted.**
  Today an admission may evict an older decode lane of the same class to make
  room; with rank it may not. Scheduler tests that assert either a whole
  reservation at admission or that eviction change, each to the new rule.
- **A grower stalls only when a move outlasts its room.**
  - A disk spill holds the victim's pages until the file commits (above),
    ~1 s for a whole-context Flash-Next blob at 1.2-1.5 GB/s. One step of room
    ahead covers that.
  - A KV-RAM move is synchronous. A whole-context Flash-Next blob stalls the
    model thread, and so every lane's round, ~0.1 s at ~12 GB/s.
- **PCIe contention of a live move.** A move out (device to host) runs against
  the opposite direction of the expert stream and should barely touch decode.
  A move in shares the experts' direction: at ~100 tok/s that direction
  already carries ~6.6-7.4 GB/s of the link's ~12 GB/s. The spec measures the
  other lanes' inter-token latency during each, against a baseline, with
  starting thresholds for the owner to confirm.
- **The expert cache grows by ~1.03 GiB at three lanes** (13.38 → ~14.4 GiB).
  - At two lanes and at one, the pool is today's within 8 pages.
  - By the one-lane/three-lane slope, three lanes should decode ~4-5% faster
    with one lane active. That is inferred, not measured; the spec measures
    it.
- **`--decode-lanes` sells a context to the cache only below two lanes.** Its
  help text changes.
- **Flash-Next's plan goes through the 27B's `plan_vram`.**
  - An explicit budget above free memory is now refused there too, unless
    `--allow-vram-oversubscription`; Flash-Next's own path never asked.
  - `--kv-pool-bytes` is honoured there.
- **Flash-Next can evict with `--kv-host-pool-bytes 0`**, now to disk.
- **The host plan gains a 64 MiB pinned line** (`kv_disk_staging`) on a load
  with the tier.
- **The model thread pays a few µs per window** to issue a copy and poll it.
  The copies' PCIe contention is the live-move point above: a spill's windows
  run against the expert stream, a restore's share its direction.
- **The 27B changes too, at its defaults.** Its requests take the default cap
  and grow their reservations; its plan does not change. Without KV-disk (its
  default), a live sequence moved into KV-RAM can still be discarded there
  for a newer victim, as today.
- **The tier's writes contend on F:** with cargo builds and with the n-gram
  reads. They happen only at eviction, and the spec measures them.
- **Amends ADR 0024's "one call per direction".** A transfer may move a window
  at a time, through the options struct that ADR 0024 reserved for "a partial
  extent". The blob is the same bytes. ADR 0024's chained blobs
  (deduplication) are still deferred: every disk blob is materialized.
- **`ignis_kv_ram_evictions_total` stays at zero on a load with KV-disk.** A
  KV-RAM live victim goes to disk instead. Only the last resort, when every
  tier refuses, counts there.
- **v1's disk tier cannot outlive a restart or a model switch.** Phase 2's
  idea (`docs/specs/flash-next/phase2-model-switch-notes.md`) starts from the
  header this ADR already writes.
