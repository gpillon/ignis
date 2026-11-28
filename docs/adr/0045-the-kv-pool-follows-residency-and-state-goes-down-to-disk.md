---
status: accepted
---

# The KV pool is sized by whether the model is resident, and evicted state goes down to disk

The KV pool and Flash-Next's expert cache trade VRAM byte for byte, and the cache sets decode speed, so a load sizes
its pool by whether every weight is on the device: a **resident** load gives the pool the rest of the budget, which
has no other use, and an **offloaded** one, its experts streaming over PCIe, reserves a default pool first and gives
the cache the rest. A request reserves its whole bound at admission, and when the pool runs out the lowest-ranked
sequence below the requester moves down a tier and later resumes bit-exact where it stopped: no work lost, never a
503, on both models by one mechanism. **KV-disk**, the tier below KV-RAM, takes what KV-RAM gives up of a
checkpoint, which went nowhere before, and the live victim KV-RAM cannot take. A retained prefix KV-RAM gives up, and
a device checkpoint KV-RAM cannot take, are discarded: nothing brings a prefix back from the disk, and retained state
never waits for a file to commit. How the modules carry it out is in
`docs/design/kv-tiers-and-live-moves.md`.

## The pool

- **Resident** is the 27B always, and Flash-Next when the budget holds its fixed lines, residency's tables, every
  expert projection and the minimum pool, a card around 96 GB; such a load stages nothing, so no prefill staging ring
  is in the test (ADR 0054). The test is plan arithmetic, done before anything is allocated.
- **Offloaded**, the default is the owner's, `OFFLOADED_KV_POOL_TOKENS`, capped at every lane's whole context, since
  a larger pool can hold only retained pages and a one-lane load so keeps its expert cache, and floored at the
  invariant, which wins, so a `--model-max-context` at the default or past it starts instead of being refused.
- **The invariant**, on both models: the pool holds one `--model-max-context` sequence and one page per retained
  slot (ADR 0030's floor), and a smaller pool refuses the start, naming the knobs. `--model-max-context` bounds
  every request, so with retained state given up and every other sequence moved, a sequence alone always fits.
- **The pool is shared by the lanes and paged**: a lane costs its state, not a context.
- **`--vram-kv-pool-bytes` names the pool on both models** and replaces the policy's size; on a resident load it
  also replaces the minimum pool in the residency test, and the rest of the budget is left unused. It takes bytes or
  tokens: a byte figure is a different context on every model and KV format, a token count the same on each, and
  the quantity the owner decided in. The pages are still what the plan derives (ADR 0022), and the load's plan
  validates the size, where the model is known; the config only parses.
- **The expert cache's floor** (`EXPERT_CACHE_FLOOR_BYTES`) refuses the start unless
  `--vram-allow-expert-cache-below-floor`, which warns, on the pattern of ADR 0030's
  `--vram-allow-oversubscription`, and which a 27B load refuses. The class-minimum refusal is not overridable: a
  cache that cannot hold every K class's minimum cannot run a step.

## Reservations and the default cap

- **The bound** is the prompt plus the request's generation cap, never past `--model-max-context`. A request
  reserves it whole at admission, it never grows, and the leaf maps a sequence whole.
- **A request that names no cap generates at most `--model-default-max-tokens` tokens**, its reasoning included, on
  both models: the output Qwen recommends for complex tasks by default (`DEFAULT_MAX_TOKENS`), `0` for none. It is
  clamped to what the prompt leaves of the context, so a long prompt is never refused for it, only one that fills
  the context alone. An explicit cap always wins, but past the context it is refused, not clamped; `ignore_eos` still
  needs one, and a decision or a constrained decode keeps its own budget. It has one home,
  `SchedulerConfig::default_max_tokens`, read by `generation_budget`:
  every entry point gets it, a request's remaining work becomes finite for the admission machine, and the same
  number sizes the reservation. It fixes the clients that send no cap, and it alone lets three of them run at once
  on the default pool.

## Only an admission moves a live sequence

- **A need is met in order**: free pages; retained state on the device; what requests waiting below the entry
  claimed; then a live sequence that ranks below the requester, moved down a tier, KV-RAM else KV-disk, at a round
  or chunk boundary, mid-generation if need be, and only when that makes the whole room. With nothing left the
  admission waits in the queue, no lane ever parks, and every sequence on the device keeps decoding to its end: no
  refusal, no 503, nothing discarded. The in-flight cap's 503 is unchanged: it concerns lanes, not tiers.
- **The trigger is reactive** (ADR 0023) and an admission is the only one: a restore never moves anything.
- **Rank gates the victim**: class first, `Interactive` above `Agent`, then submission order, and a moved sequence
  keeps its rank. "Never the requester" alone would let an `Agent` need move an `Interactive` lane, and two equal
  lanes move each other in turn. With rank an `Agent` need never moves an `Interactive` sequence; within a class an
  older sequence may move a younger one and never the reverse, so a newcomer waits behind the sequences of its class
  already admitted; and the top-ranked sequence can move every other, so it always progresses, and each eventually
  becomes the top. A shortage of pages takes what is not decoding first; a shortage of lanes keeps its own order
  (ADR 0023), since moving a lane-less request frees no lane.
- **A victim frees its reservation and the links of its prefix chain** that only retained state and lower-ranked
  waiting claims hold: counted by its reservation alone, a request that had published or lent its prompt's pages
  could not be moved for them.
- **A waiting claim is a bet not yet cashed**: what a request still waiting for its admission claimed goes as
  retained state goes, after it, for an entry that outranks the claimant, and only when that makes the room.
  Before, the entry waited for the claim, the claimant behind the entry, and nothing ran again. So ADR 0029's
  "retained state never delays or refuses an admission" holds of such a claim too, and a claim reports its reuse
  and its hit only with the chunk that lands it.
- **Entries are taken in rank order**, each when its whole reservation fits, so a moved sequence comes back before
  any newcomer that ranks below it and a stream of short newcomers cannot keep it out.
- **A move goes a window at a time** on the leaf's transfer stream (ADR 0024), and the model thread never waits on
  a window's copy or on the disk.

## KV-disk

- **One mechanism under the `Compute` seam, for both models**, on by default on Flash-Next and off on the 27B
  (`kv_disk::default_bytes`): the 27B's RAM is not taken by experts and `make` gives it a large KV-RAM arena above
  the disk, its prefill is fast enough that a crossing
  from disk pays only past its floor, the development volume is nearly full, and its defaults stay byte-identical.
  One flag turns it on.
- **No lost work.** With the tier on, an evicted live sequence is never discarded to make room, in KV-RAM or on
  disk; only a failed read loses one. When no tier has room the request waits and no victim is taken. Without the
  tier the 27B keeps its earlier guarantee, no stronger: KV-RAM still discards a live blob to take a newer victim.
- **The eviction priority goes one tier down** (ADR 0023): leaving KV-RAM, on a load with the tier, is a move to
  KV-disk, leaving KV-disk is a discard in the same order, and a spill displaces only what ranks below it.
- **Lazy**: a blob reaches disk only when the tier above gives it up, never as a copy of state still resident.
- **The budget is a ceiling, not a reservation**: nothing is preallocated, and a write that would cross the
  volume's margin, or that fails, is refused with the blob left where it was. The tier never refuses a start: it
  is a cache, like the n-gram one.
- **A restore goes straight to the device**, never through KV-RAM, and a retained match on disk must beat the best
  match on the device and in KV-RAM by the family's floor (`restore_floor_tokens`), so a tie goes up.
- **v1 reuses nothing across a restart**, so the tier cannot outlive a restart or a model switch. Each file still
  opens with a header page, written last as the commit, that carries the load's full identity (GitHub #205) and a
  CRC32 per window (`docs/formats/kv-disk-file.md`): a torn, corrupt or foreign file is never restored, and a later
  step that keeps files needs no format change.

## Considered Options

- **Reservations that grow page by page** (the growth branch). Preferred, since concurrency would follow what lanes
  use and not what they are capped at, but not without measuring: growth puts a leaf call on the serving path at
  every page, on decode, the number everything else is weighed against. Its gate, P0, grew one page at a time, the
  worst case for a branch that grows by 32-page steps, and by the owner's rule passed only if, at every lane count on
  both models, decode tok/s was at most
  0.5% below the mean of the two whole-mapping legs and ITL p99 no worse than the higher of them plus their
  difference, the run's own noise. It moved the 27B's ITL p99 past the rule
  ([the finding](../findings/2026-10-08-kv-page-growth-gate.md)), so the fixed branch above was built, the owner
  confirmed it, and nothing of growth is in the code. Growth was never gated again at the 32-page step the finding
  names as the first thing to measure.
- **Rest-to-pool on Flash-Next too.** It starves the expert cache, which sets decode speed.
- **A whole context per lane**, the rule before. It pays the cache's VRAM for contexts no lane is using.
- **A new `--kv-pool-tokens` flag.** Two flags for one quantity, each refusing the other.
- **Shrinking the default pool when the floor does not fit.** That silently cuts concurrency; the refusal names the
  knob instead.
- **Lowering the expert cache's floor.** It is where decode collapses, so the operator must choose to go below it,
  knowingly.
- **Accepting two lanes, or raising the pool default.** With whole-context reservations two requests without a cap
  fill the default pool; but the third lane is decode capacity the card already pays for in lane state, and a third
  whole context costs the expert cache the trade the policy exists to undo, for pages no lane writes.
- **Applying the default cap in each HTTP handler.** Three surfaces would each re-derive the context clamp.
- **A newcomer moves an older sequence of its class**, as before. A stream of newcomers could keep moving a
  sequence that has done the work.
- **Moving sequences at a pool low-water mark.** Pre-emptive, which ADR 0023 rules out.
- **Keeping the KV-RAM moves synchronous**, one call per direction (ADR 0024). A move stalls every lane for its
  length, about 0.32 s as measured, so a move goes a window at a time.
- **Raising the in-flight cap so that a burst beyond the lanes queues.** That changes HTTP backpressure, is not a
  tier decision, and is left to the owner.
- **Writing to disk eagerly**, at request end or on every spill. It writes state that may never be needed and wears
  the drive.
- **Restoring disk to KV-RAM to device.** It needs arena room the host does not have, and it is still one PCIe
  crossing.
- **One preallocated arena file for the disk.** NTFS zero-fills up to any write past the valid data length, and
  writing the whole budget at start takes about 15 s at 1.2 GB/s.
- **Persistence across a restart in v1.** Not nearly free: only retained state would be worth keeping, a live
  sequence dying with its connection; the retained index would be rebuilt from the file headers and the identity
  would have to hold across loads; and a wiped tier is a cache, where a kept one is every conversation's state at
  rest, outliving the process and its crashes.

The alternatives weighed for how a move and the disk's IO are carried out are with the mechanism, in
`docs/design/kv-tiers-and-live-moves.md`, and the sizes and rates behind the disk's choices in
[its arithmetic](../findings/2026-10-07-kv-disk-arithmetic-on-the-rtx-5090-host.md).

## Consequences

- **A client that sends no cap** stops at the default with `finish_reason: "length"`, where it ran to the end of
  the context; the 27B's requests take the default cap too, and its plan does not change.
- **A same-class newcomer waits** behind the sequences already admitted, where an admission could evict an older
  decode lane of its class.
- **Concurrency follows what lanes are capped at**: a large explicit cap, or a default cap of `0`, still holds
  pages it may never write.
- **Flash-Next's plan goes through the 27B's `plan_vram`**, so an explicit budget above free memory is refused
  there too, unless `--vram-allow-oversubscription`.
- **Flash-Next can evict with KV-RAM off**, to disk, and `ignis_kv_ram_evictions_total` stays at zero on a load
  with KV-disk.
- **ADR 0024's chained blobs are still deferred**: every disk blob is materialized.
