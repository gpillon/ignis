---
status: accepted
---

# A pinned hot set of experts: a field off by default, which the rtx5090 profile sets

An offloaded Flash-Next load, whose VRAM expert cache holds only part of the experts (about half on the 5090),
keeps every projection in the pinned host pool so that a miss can copy it in: about 38 GB, most of a 64 GB host's
RAM, where the KV-RAM arena and the n-gram hot rows compete for the rest. A projection that never leaves VRAM needs
no host copy. So `vram.pinned_expert_bytes` (Flash-Next only, changed through a reload) pins the
calibration-hottest whole experts in the cache, outside its LRU and with no copy in RAM, until their bytes reach
it. Its default, 0, pins none: the owner asked for an opt-in whose default leaves the load's behaviour, plans, logs
and outputs byte for byte what they are without the feature. The `rtx5090` profile sets it for Flash-Next, the
owner's choice from the measurement.

The LRU by itself keeps no such set for long, but pinning the calibration-hottest experts anyway is cheap up to a
point. On the 5090 ([measured](../findings/2026-10-18-pinned-hot-set-measured-on-the-5090.md), after the
[study](../findings/2026-10-18-partial-exclusive-expert-pool.md)) a 2 GiB pin gives back 2.1 GB of RAM for at most
1.1% of three-lane decode, greedy text unchanged, and 1 GiB costs nothing measurable. 4 GiB costs 4.3% at one lane
and 5.9% at three: the pin takes K = 4 slots mostly, and there it leaves the gate/up K = 4 LRU its minimum.

- **The set** is whole experts in the warm start's order, the calibration's selections, taken out of the pools the
  cache split already made, as the study measured it, not a new split of the rest.
- **Each class keeps for its LRU the slots of the widest decode step the load runs**: the class minimum of its
  decode rows, which under MTP is a verify round's lanes × (drafts + 1), up to the row budget, not the lanes. The
  split gives every class that minimum first, the plan's split and the load's alike, at any pin, 0 included: one
  rule, so no class is split below it, pinned or not. It is a correctness rule, not a tuning: a class below it
  cannot hold what one verify round selects and looks ahead.
- **A budget the walk cannot reach while keeping those slots is refused at plan time**, before anything loads,
  naming the most it can pin and where the budget was set: the flag, the environment variable, the file key or the
  profile.
- **One computation.** The server's plan and the load build the pools and the pinned set from the same call with
  the same options.
- **A selected pinned key is a hit and nothing else**: never stamped, evicted, staged, prefetched or copied. The
  CPU policy model pins the same way, and the device equals it step for step, on random traces and on the recorded
  trace replayed with pins.
- **No live resize of the expert cache may evict a pinned projection**: it has no copy to come back from. Pools
  are fixed per load (ADR 0030), so none does today.
- **Device-only loads (ADR 0054) ignore the field**: they keep no expert in RAM already.
- **The cache's `capacity` series counts the LRU's slots only.** The pinned slots are in the event a load that
  asked for a pin logs, not in a series of their own.

## Considered Options

- **On by default.** Taken as a profile value, not as the field's default, which stays 0. The field is listed like
  every config field (ADR 0046), and being Flash-Next only, the 27B passes the profile's value over.
- **Pin what the LRU never evicts.** Small (0.4-0.5 GB), still shrinking with the length of the trace, and not
  known at load.
- **A fully exclusive pool**, where RAM holds only what VRAM does not and victims are written back. About 18 GB
  back for 20-70% of a round on a PCIe Gen 3 link, the study's inference. Not decided here.

## Consequences

- The host pool and the process's RAM are smaller by exactly the pinned bytes.
- A load with a pin takes about 2 s longer to load its experts. Under the `rtx5090` profile every default
  Flash-Next load pays it, logs the event and shows a `capacity` lower by the pinned slots.
