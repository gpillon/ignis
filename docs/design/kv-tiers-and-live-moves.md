# The KV tiers under pressure: a page shortage, a live move, KV-disk (ADR 0045)

How an admission that finds the KV pool short gets its pages, how a live
sequence moves down a tier and comes back, and how KV-disk stores what it is
given. ADR 0045 is the decision record: reservations are whole, only an
admission moves a live sequence, rank gates the victim, no work is lost. This
page is the map across the modules that carry it out; each rule is stated in
full beside its code, in the place named here.

| Module | What it holds |
|---|---|
| `crates/core/src/admission.rs` | `Rank`, the page shortage's victim order (`page_shortage_victims`) |
| `crates/core/src/concrete.rs` | the admission's passes: `reclaim_retained`, `make_room`, `snapshot_and_evict`, `restore_pass`, `disk_pass` |
| `crates/core/src/prefix.rs` | what a prefix chain gives back (`pages_a_holder_frees`, `links_let_go`) |
| `crates/core/src/host.rs`, `crates/core/src/disk.rs` | the KV-RAM and KV-disk ledgers, one eviction order held twice |
| `crates/runtime/src/lib.rs` | `TransferPace`, the leaf's windowed transfer (`snapshot_window`, `restore_window`, the fences) |
| `crates/runtime/src/disk_transfer.rs`, `crates/runtime/src/kv_disk.rs` | KV-disk's transfers, and its files and IO threads ([the file's layout](../formats/kv-disk-file.md)) |

## A need for pages

The triggers are reactive (ADR 0023): the one trigger is an admission
materializing its reservation. A restore never triggers anything. The need is
met in this order, each step only as far as the one before fell short:

1. Free pages.
2. Retained state on the device, given up first (the first-victim pass).
3. What requests still waiting for their admission claimed, when they rank
   below the entry ([below](#a-waiting-claim)).
4. A live sequence that ranks below the requester, moved down a tier at a
   round or chunk boundary: into KV-RAM, else straight to KV-disk when KV-RAM
   cannot take it. Among the eligible, `Agent` goes before `Interactive`, then
   a sequence not in the decode round (a prefilled request waiting for a lane,
   a half-prefilled one at a chunk boundary) before a lane holder, then the
   latest-submitted: every decode lane carries the same use tick, so least
   recently used among them would reduce to lane id. Never the requester, a
   sequence mid-transfer, or one the admission machine protects: a protection
   donor, a lane reserved for an earlier `Interactive` request.
5. Nothing left: the admission waits in the queue, held as `make_room` already
   holds a head, and is admitted when room frees.

On an admission nothing is given up unless steps 2 to 4 together cover the
need: otherwise no retained state or claim is given up and nothing moves,
since what moved would only come back into the room it left. When they do,
step 2 goes first, until the entry fits (`reclaim_retained`). The test
(`room_can_be_made`) runs before anything is given up and counts all that
would come back once steps 2 to 4 had all gone, the most step 2 and
`make_room` can free, so an entry it turns away could not have entered in that
advance. A claim
that makes the room alone goes before anything moves. A shortage of lanes is a
different path with its own order (`evict_one_victim`): there a lane holder
goes before a lane-less request, since moving a lane-less request frees no
lane.

## What a victim frees

A victim counts as freeing its reservation and the pages of its prefix chain
that only retained state, and the waiting claims below the requester, hold
beside it (`PrefixCache::pages_a_holder_frees`). A request that publishes a
shared prefix, or lends the pages below its opener at its capture, hands those
pages to the prefix cache and its reservation shrinks by them; counted by its
reservation alone, a need that only those pages would cover moved nothing and
waited behind a request that could have moved: every 27B request that
published a head, and every Flash-Next one that captured. The chain changes
what a victim counts as freeing, never which victims are eligible or their
order ([step 4](#a-need-for-pages)).

- The walk goes from the entry the victim stands on toward the chain's root. A
  link counts while its holders, but for the one reference from below, are
  retained state: the device checkpoints standing on it and its own retention.
  The first link anything else holds ends the walk, since every link above it
  is held through it: a live claimant, another chain's link, or the claimant of
  a checkpoint not yet built from that does not rank below the requester.
- What the victim published, what it lent and what it claimed from a finished
  request count alike: a moved sequence comes back owning all of its pages.
- Where two victims share a link each sees the other's reference, so neither
  counts it. No page is counted twice, and a need only a set of victims would
  free together waits.
- The chain is read only on an advance where an admission is short of pages
  after retained state was given up, and only when the victims' reservations
  alone fall short: a few links per eligible victim, no pass over the pool.
- The move lands, then the first-victim pass frees the chain. A move a window
  at a time ends in a later advance; until its last window lands the victim
  keeps its pages and its chain, and no advance judges the need covered. In the
  advance it lands the victim's references drop, and the admission's own
  first-victim pass gives up the retained state left holding the chain. On a
  synchronous backend, such as the mock, the move completes in the call and
  that pass follows at once.

## A waiting claim

A request waiting for its admission holds what it claimed: a reference on the
shared prefix it resumes from, and the device checkpoint it chose, tail page
included, until its first chunk copies it. That room is neither retained
state nor a live sequence; while it yielded to nothing, an entry ranked above
the claimant waited for it, the claimant waited behind that entry, and nothing
ran again.

- For an admission, or a moved sequence coming back, the claims below it go
  after retained state, the lowest-ranked claimant first, each only if its
  going gives pages back: alone, or, where claims stand together on a link,
  with the others below the entry. A claim on a live sibling's prefix, or on a
  block that a checkpoint another claim keeps chains over, frees nothing and
  stays.
- A claim goes only for an entry that ranks above its claimant: a moved
  sequence never takes the claim of a waiting request above it. A need that
  fits takes no claim, and no claim goes for an entry that would wait anyway.
- The claimant starts over (`Request::give_up_claim`): its reservation is its
  whole bound again, and it looks again for state to resume from when it next
  comes up. The checkpoint it chose may have gone to KV-RAM, from where it
  comes back past the restore floor, or it prefills.
- A claim reports its reuse, its tokens and its hit with the chunk that lands
  it, so `cached_tokens`, the request log's reuse and the hit count say only
  what a request resumed from.
- The count is exact over chains (`PrefixCache::links_let_go`): every link
  that nothing else holds once retained state and the claims all let go, a
  head two claims reach through links of their own included. So a device that
  holds only retained state and waiting claims always gives one bound its
  room. It is read only on an advance where an entry is short of pages after
  retained state was given up.

## A move through KV-RAM

On a leaf with a transfer stream, which both leaves have, a move goes a window
at a time; the synchronous whole-blob call remains for a backend without one,
and for retained state.

- A move out copies the sequence straight into its arena span. The victim
  keeps its pages, its resident slot and its lane, in no round, until the last
  window lands, and only then is released.
- A move in draws its sequence at the start, charged, and is schedulable once
  its last window lands.
- A cancel abandons the move after its window in flight.
- One move each way at a time, and none onto the device beside a disk restore:
  a disk restore and a KV-RAM one never share the link.
- A failed move off the device pauses moves off it for a moment
  (`KV_RAM_MOVE_BACKOFF`), so the victim decodes instead of retrying the same
  copy every advance.
- The pace is `TransferPace`: one window in flight each way, a new one at most
  once an advance, constants and not flags. Flash-Next's windows are sized so
  that the copies leave a round's time outside its expert stall where it was;
  the 27B moves in one window, since its decode puts nothing on the link. The
  same pace feeds a KV-disk restore and slices a KV-disk spill's copies off the
  device.
- The bound a move in is held to is the transfer's, its rounds' time outside
  the expert stall (the owner's): what those rounds miss of the expert cache
  is the text they generate, not the move, and the rounds after a restore are
  held to a same-text unmoved run
  ([the finding](../findings/2026-10-08-expert-misses-after-a-move-are-the-text.md)).
  A move out runs against the direction of Flash-Next's expert stream and a
  move in shares it
  ([measured](../findings/2026-10-08-live-move-pcie-contention.md)).
- The whole-blob copy runs at the link's speed: Flash-Next's block keys go a
  run of pages per copy, where a page at a time held it to a fraction of that
  ([measured](../findings/2026-10-08-live-moves-windowed.md)).

## Entry

Restores and admissions share one rule (`restore_pass`, `make_entry_room`):
a sequence enters when its whole reservation fits beside the room held for
every entry that outranks it and still waits, and entries are taken in rank
order. So a moved sequence comes back before any newcomer that ranks below it,
and a stream of short newcomers cannot keep it out; a newcomer that moved
sequences for its room keeps that room while it waits for the last move. A
restore moves nothing live: it takes free room, with retained state and the
claims below it given up when that is what makes the room. A live blob has no
restore floor; it comes back whenever room returns.

## KV-disk

- **Lazy.** A blob reaches the disk only when the tier above gives it up,
  never as a copy of state still resident, and only from a snapshot point.
- **The ledger** (`crates/core/src/disk.rs`) is a byte ledger walked in the
  eviction priority's order. On a load with the tier, a checkpoint leaving
  KV-RAM moves to the disk and a retained prefix leaving it is discarded;
  leaving the disk is a discard, in KV-RAM's order. A spill
  displaces only what ranks below it, and a live blob is never discarded for
  room.
- **One file per blob**, in a directory the process owns: the filesystem
  places the files, so there is no span to fit, unlike the pinned arena's
  first-fit.
- **A spill from the device** keeps the victim's pages until its file is
  committed, so a failed write leaves it intact and the request that wanted
  the room keeps waiting. A spill from KV-RAM writes straight from the arena
  span, with no staging.
- **A restore goes straight to the device**, never through KV-RAM. A live
  blob's file is deleted once it is restored; a retained blob's file stays,
  because a claim never consumes (ADR 0029). The restored request rejoins, a
  lane or its prefill, at the first advance after its last window lands. A
  retained match on disk is taken only when it beats the best match on the
  device and in KV-RAM by the family's floor (`restore_floor_tokens`), so a tie
  goes up.
- **IO** is unbuffered and positional, on the tier's own two threads, a
  request each: that gives the overlap without completion ports, which would
  need a `windows-sys` release that does not build on the development host's
  windows-gnu toolchain. Windows move through a pinned staging of two, one
  crossing PCIe while the other is written or read. The model thread issues a
  window's copy on the transfer stream between steps and polls for it at the
  next advance; it never waits on the disk or on a copy, and pays a few
  microseconds per window to issue a copy and poll it.
- **The n-gram table goes first.** While a prefill's n-gram gather is pending
  the tier issues no new request; one already issued finishes its window.
  Decode gathers do not hold the tier: they are small, and the tier keeps at
  most one request in flight per direction.
- **A write that fails, or that would cross the volume's margin, is refused**
  and the blob stays where it was; the tier then takes no new write for a
  moment (`KV_DISK_WRITE_BACKOFF`).

## Alternatives not taken

- **Synchronous KV-RAM moves**, the first design, which assumed the link's
  full speed. Measured, a move of a whole Flash-Next sequence stalled every
  lane for its length
  ([live moves a window at a time](../findings/2026-10-08-live-moves-windowed.md)).
- **A move in's window waited for between rounds**, tried. It costs the
  window's copy time on every round and saved nothing measurable beside it.
- **One preallocated arena file for the disk.** NTFS zero-fills up to any
  write past the valid data length, and writing the whole budget at start
  takes seconds.
- **Staging a whole blob in pageable RAM.** More than a gigabyte at a time, on
  a host that cannot spare it.
- **Restoring disk to KV-RAM to device.** It needs arena room the host does
  not have, and it is still one PCIe crossing.
- **SHA-256 per blob**: see [the file's layout](../formats/kv-disk-file.md).

The sizes and rates are in
[the disk's arithmetic](../findings/2026-10-07-kv-disk-arithmetic-on-the-rtx-5090-host.md).

## Where a move is counted

A live move is counted where it lands: into KV-RAM it is
`ignis_kv_cache_evictions_total`, straight to disk
`ignis_kv_disk_spills_total{from="device"}`. `ignis_kv_ram_evictions_total`
stays "dropped and lost": a demotion to disk is not one, so on a load with
KV-disk, where a KV-RAM live victim goes to disk instead, it stays at zero.
