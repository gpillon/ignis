---
status: accepted
---

# Sequence state transfer: an opaque versioned blob, and device-to-device cloning for prefix reuse

A sequence is not one buffer: it is sections of state, KV pages, the GDN recurrent slot, conv taps, a position and
more, consistent together at a completed chunk boundary. One place in the leaf, the section table, knows what a
sequence is made of, what each section costs, and whether it is shareable read-only or must be cloned per sequence.
The table is leaf-internal: the ABI (ADR 0009) exposes a snapshot size and a format version, not a layout, and the
blob is opaque. Prefix reuse clones on the device: KV pages are shared by refcount, owned by the leaf, where the
block table already lives, and the mutable sections are copied device-to-device. Snapshot, restore and clone share
the one description, so a new section is carried by all three or by none, and a section added to only one of them is
a visible omission rather than a silent one.

**The blob.**

- It carries its own header, which names its compatibility identity: the artifact content hash, the KV format, the
  layout version and the drafter configuration (ADR 0029). Restore validates it and refuses a stale or foreign
  layout, or a blob taken under another load, rather than writing it into a live sequence. A blob does not survive a
  version change, and that is the intent: it is rejected, not reinterpreted. Nothing further is exposed until a
  consumer needs it.
- Each model family has its own layout version, and a blob belongs to one drafter: its identity names the drafter,
  so a load refuses a blob taken under another drafter or under none (ADR 0095).
- Adding a section is an edit to the table, which makes re-earning the snapshot-point permission an act rather than
  a memory: a new section does not inherit the permission the others earned at the chunk boundary, and must re-earn
  it.
- Transfer is whole-sequence, and partial restore is not offered: GDN state is not recomputable without re-running
  the prefix that the restore exists to avoid. A transfer may move the blob a window at a time (ADR 0045), through
  the options struct: the windows laid end to end are the blob's bytes, and a null options pointer is the whole-blob
  call.
- A sequence holding a shared prefix is snapshotted, not refused (ADR 0029): its blob materializes the shared pages,
  a copy that owns every byte it carries and outlives the prefix, at the cost of prefix bytes duplicated per blob.
  Chained blobs, a reference to a retained prefix plus a tail, stay deferred: every blob, a disk one included, is
  materialized (ADR 0045). Restoring into a sequence that holds a shared prefix is still refused, since its pages are
  other claimants' history: a blob is restored into a fresh sequence.
- A retained checkpoint or prefix spills as the blob its sequence would have written, so it restores like any other.
- Per-sequence state lives where these calls can reach it, in the pool: they take a pool and a sequence, never a
  model, so a section must be addressable from those two alone. The drafter's window is the sequence pool's for that
  reason, and the MTP head's state travels as sections too, its snapshot point earned at a completed chunk boundary
  and after a committed round, not inherited (ADR 0095).

**Prefix reuse.**

- A shared or retained prefix is published at a whole-page boundary, which makes it a scheduling decision: the
  mutable state a claimant clones is the state at the prefix's end, so the publisher's prefill has to stop exactly
  there. A prompt whose length is not a whole number of KV pages pays one extra prefill chunk, in exchange for every
  sibling skipping its head.
- A prompt checkpoint publishes at its generation opener instead, wherever that falls, and the partial page it ends
  inside is copied, not shared; a prefix may be published over another (ADR 0029). On a pool built for the MTP head
  the page holding the publish point's last position is each holder's own copy (ADR 0095).
- A claim never consumes: retained state is cloned out, so one entry serves any number of claimants.
- The multimodal `rope_delta` is a progress scalar, which snapshot and restore carry, and the one progress scalar
  the clone path leaves behind: a multimodal claimant gets its delta from the tail it always prefills (ADR 0029).

**The state-transfer and prefix calls are entry points.** ADR 0016's rule, that the step ABI grows by fields on an
options struct and never by new entry points, is about steps. It does not reach the calls that create and destroy
the objects a step runs against, and these are of that kind: a snapshot size and a format version are queries with
no knob, and publish, claim and release are lifetime operations on a leaf-owned object. `ignis_seq_alloc_shared` is
a second constructor rather than a flag, since a sequence that claims a prefix is built differently, not stepped
differently. A snapshot control, a policy, a stream or a partial extent, still goes in an options struct, and is not
another entry point.

## Considered Options

- **The section table in the ABI.** It would let the host tier compute layouts, and would put the leaf's internal
  geometry into a contract that every future section change breaks.
- **Prefix reuse as the restore of a sibling's snapshot.** One mechanism instead of two, but it moves the mutable
  sections host-ward and back, two PCIe crossings for bytes that never needed to leave the card, while the read-only
  KV pages can simply be shared.
- **Sharing pages without cloning the mutable sections** saves nothing, because prefill must traverse every layer to
  produce them. Recorded so it is not re-proposed as an optimization.
- **A feature-gated readback seam** for the tests that inspect a blob: the GPU profile does not build it, so such a
  test would never run.

## Consequences

- The leaf owns the physical pages and their refcounts: the Rust pool's refcounts are admission accounting, not
  truth.
- The host tier cannot compute its own buffer layouts, and does not need to: it asks for a size, allocates pinned
  memory, and stores bytes it does not interpret.
- A GPU test may read the blob layout; a production consumer may not. Such a test spells out the layout for one
  format version and asserts that version first, so a layout change fails the test instead of being misread.
- The feature taps a verify round at extent 0 leaves on the sequence handle are deliberately not a section: a
  sequence transferred there would lose acceptance, never text, and the scheduler never reaches the case, since
  extent 0 is a request's last round. If a sequence is ever continued from there across a transfer, the taps become
  a section.
- Vision state is not a section: the chunk that continues an evicted media item after restore encodes it again.
- The costs are asymmetric, and are measured rather than assumed: the device clone is about fifty times cheaper than
  the same bytes one way over this host's PCIe link, and a full-context snapshot about a hundred times cheaper than
  the re-prefill it replaces ([snapshot](../findings/2026-09-12-sequence-snapshot-transfer-cost.md),
  [clone](../findings/2026-09-12-device-prefix-clone-cost.md)).
