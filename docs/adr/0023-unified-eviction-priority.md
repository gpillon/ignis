---
status: accepted
---

# One eviction priority across GPU residency and the host tier

What loses residency is decided by one priority model across the tiers, the device, KV-RAM and KV-disk, whose shared
key is the request class, `Agent` before `Interactive`. Two independent policies, the admission machine's and a host
tier with no notion of class, would let a request's class change meaning the moment its state crosses the PCIe bus:
an accident of where the machinery was written, not a tier policy. A single ordering shared by every level was
rejected, so it is one model and not one ordering. On the device something is being served, so the admission
machinery's protection and entitlement (ADR 0004) must outrank class: a protected lane mid-stream is not a victim
because its owner is an agent. Below the device nothing is being served, protection has no meaning, and class is
the only thing left that says whose work costs most to lose. Eviction is reactive: it runs on the path where a need
finds no room, never pre-emptively.

**Retained state goes first, on the device and in the tiers below** (ADR 0029). Prompt checkpoints and retained
prefixes are state no live request needs, and the loss is asymmetric: an evicted live sequence is work that will
certainly be redone, a retained entry is a bet that it will be needed. On the device it never causes an admission
refusal or a wait. A retained entry carries the class of the request that produced it.

**What leaves the device.**

- Retained state, before any live sequence is considered, prompt checkpoints before retained prefixes (ADR 0029). It
  spills to KV-RAM, where a spill displaces only retained entries that rank strictly below it in KV-RAM's order and
  never a live snapshot. When even all of those are not enough, nothing is discarded there and the device discards
  the newcomer: the room is planned before anything goes, so a spill that fails costs no other entry.
- Then a live sequence, eligible only at a chunk or round boundary, protection ranking first: protected lanes are
  last. One that holds a shared prefix is a victim like any other, since its snapshot materializes the prefix's pages
  (ADR 0024) and it resumes rather than re-prefilling.
- A shortage of lanes takes a lane holder before a lane-less request, since moving a lane-less request frees no lane,
  so a lane-less prefilling request is chosen only when every lane is protected. Then class, least recently used,
  and the lane id as the last tie-break.
- A shortage of pages (ADR 0045) moves only state that ranks below its requester, rank being class and then
  submission order, the earlier above the later; never the requester, a sequence mid-transfer, or one the admission
  machine protects, a protection donor or a lane reserved for an earlier `Interactive` request. Among those it may
  move: `Agent` before `Interactive`, then sequences not in the decode round before lane holders, then the
  latest-submitted, since every decode lane carries the same use tick and least recently used among them was only
  the lane id. A moved sequence goes to KV-RAM, else to KV-disk.

**What leaves KV-RAM.** Retained entries before evicted live sequences, then class, then probation before protected,
then least recently used. An `Interactive` entry idle past `--reuse-retained-interactive-ttl` ranks as an `Agent`'s
probation entry: class alone would let conversations nobody returns to hold KV-RAM against every subagent. `Agent`
entries have no TTL.

**What leaves KV-disk** (ADR 0045). On a load with KV-disk, leaving KV-RAM is a move to the disk, a spill into the
disk displaces only what ranks below it, and leaving the disk is a discard in KV-RAM's order. An evicted live
sequence is then never discarded to make room, in KV-RAM or on the disk: only a failed read loses one. When no tier
has room, the request that needed it waits. Without KV-disk, a live sequence in KV-RAM can still be discarded there
for a newer victim, and re-prefills.

## Consequences

- The asymmetry between the device and the tiers below is deliberate and has to be tested as such. The case that
  proves it: an `Interactive` snapshot in probation outlives an `Agent` snapshot in protected.
- The policy is only as good as the class, so the class has to arrive from the workload. Where no request states
  one, every request is `Interactive` and the orderings collapse to what they were before class: a safe default, not
  a broken one.
- A third request class, if one is ever added, changes two orderings rather than one.
- The ordering is expected to change as workloads teach more, and it is recorded here so that a change is a decision
  rather than a drift. One structured retention score in place of these orderings is open (#199).
