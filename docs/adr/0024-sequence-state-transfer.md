# ADR 0024 — sequence state transfer: an opaque versioned blob, and device-to-device cloning for prefix reuse

## Status

Accepted (2026-09-11) — GitHub #65, phase 4. Spec:
`.scratch/runtime/specs/04-reference-feature-floor.md`. Applies ADR 0009
(step-level device-resident ABI) to the two features that move sequence state.

## Context

Two phase-4 features move a sequence's state, and both were written above an
ABI that cannot: `ignis_seq_snapshot` and `ignis_seq_restore` are declared and
return `IGNIS_SEQ_ERR_NOT_IMPLEMENTED` (`kernel/src/seq.cu:217`), while
`crates/core/src/host.rs` and `crates/core/src/prefix.rs` already model a host
tier and a refcounted prefix cache in Rust.

A sequence is not one buffer. It is KV pages, a GDN recurrent slot of 144 MiB,
conv taps, a position and last token, and since #99 a penalty-count row. The
G3 session established that all of these are consistent together at a completed
chunk boundary, and recorded a caveat with it: a *new* state section does not
inherit that snapshot-point permission and must re-earn it
(`.scratch/DEFERRED-DECISIONS.md` item 5).

Two questions followed. First, how much of that structure crosses the C ABI.
Exposing a section table to Rust would let the host tier compute layouts, and
would also put the leaf's internal geometry into a contract that every future
section change breaks. Second, how prefix reuse transfers state. Expressing it
as "restore a sibling's snapshot" reuses one mechanism, but it would move the
mutable sections host-ward and back — two PCIe crossings for bytes that never
needed to leave the card — while the read-only KV pages can simply be shared.

## Decision

**The section table is leaf-internal.** One place in the leaf knows what a
sequence is made of, what each section costs, and whether it is shareable
read-only or must be cloned per sequence. Adding a section is an edit to that
table, which is what makes re-earning the snapshot-point permission an act
rather than a memory.

**The ABI exposes a size and a version, not a layout.** A caller asks for a
sequence's snapshot size and gets a number plus a format version; the blob is
opaque and carries its own header. Restore validates that header and refuses a
stale or foreign layout rather than writing it into a live sequence. Nothing
further is exposed until a consumer actually needs it.

**Transfer is whole-sequence.** One call per direction. Partial restore is not
offered: GDN state is not recomputable without re-running the prefix that the
restore exists to avoid.

**Prefix reuse clones on the device.** KV pages are shared by refcount, owned
by the leaf, where the block table already lives. Mutable sections — the GDN
slot, the conv taps — are copied device-to-device through the same internal
section machinery the snapshot path uses. The host tier's pinned-memory path is
a different transport over the same description.

**Lifetime calls are entry points** (amends ADR 0016 for this family). ADR 0016
rules that the step ABI grows by fields on an options struct, never by new
entry points, and that a call which plausibly needs per-call modulation takes
an options pointer from its first version. That rule is about *steps* — a
forward pass whose route, compute policy or sampling would otherwise multiply
parameters and `_ex` variants. It does not reach the calls that create and
destroy the objects a step runs against: `ignis_seq_pool_create`,
`ignis_seq_alloc` and `ignis_seq_release` predate it and take no options
struct, because there is nothing about an allocation to modulate per call.

The state-transfer and prefix calls this ADR introduces are of that second
kind. A snapshot size and a format version are queries with no knob; publish,
claim and release are lifetime operations on a leaf-owned object. So they are
entry points, and `ignis_seq_alloc_shared` is a second constructor rather than
a flag on the first — a sequence that claims a prefix is built differently,
not stepped differently.

What ADR 0016 still governs, unchanged: a *snapshot control* — a policy, a
stream, a partial extent — goes in an options struct when a phase needs one,
and is not a fifth entry point.

## Consequences

- The two parallel page ledgers end. The leaf owns physical pages and their
  refcounts; `KvPool`'s refcounts in Rust become admission accounting, not
  truth.
- **A prefix is published at a page boundary, which makes it a scheduling
  decision.** The mutable state a claimant clones is the state at the prefix's
  *end*, so the publishing request's prefill has to stop exactly there. A
  prompt whose length is not a whole number of KV pages therefore pays one
  extra prefill chunk — its shareable head, then its remainder — in exchange
  for every sibling skipping that head entirely (P4-10, GitHub #126).
- **A sequence that holds a shared prefix cannot be snapshotted.** Its leading
  pages belong to the prefix, so there is no whole-sequence blob to write: the
  leaf refuses with its own code and the sequence is released and re-prefilled
  rather than evicted to the host tier. The alternative — copying another
  request's history into this request's blob — is the corruption the refusal
  exists to prevent.
- Clone, snapshot and restore share one description, so a new section is
  carried by all three or by none. A section added to only one of them is now a
  visible omission rather than a silent one.
- The host tier cannot compute its own buffer layouts, and does not need to: it
  asks for a size, allocates pinned memory, and stores bytes it does not
  interpret.
- A blob does not survive a version change. That is the intent — a snapshot
  taken before a layout change is unusable and must be rejected, not
  reinterpreted.
- Sharing pages without cloning the mutable sections saves nothing, because
  prefill must traverse every layer to produce them. This is recorded so it is
  not re-proposed as an optimization.
- Costs are asymmetric and worth measuring rather than assuming: a
  device-to-device clone of the GDN slot is roughly 0.09 ms, while a
  full-context snapshot is about 528 MB and near 21 ms per direction over
  pinned PCIe — still two orders of magnitude cheaper than the ~4.6 s
  re-prefill it replaces.
  - **Measured (P4-06, GitHub #124):** 508 MiB and **~45 ms per direction**,
    not 21 ms. This host caps the 5090 at PCIe Gen 3 x16, so both directions
    run at ~12 GB/s. The asymmetry is around 100x rather than ~220x, which
    leaves the decision unchanged; the estimate above was made at PCIe 5.0
    rates. See
    [Sequence snapshot transfer cost](../findings/2026-09-12-sequence-snapshot-transfer-cost.md).
  - **Measured (P4-10, GitHub #126):** the device-to-device clone is **~0.25 ms**
    for the 148 MiB of mutable state at this geometry, not 0.09 ms. The state
    is strided per GDN layer in the pool and packed in the prefix's image, so
    the copy runs at ~620 GB/s rather than at the card's flat-copy bandwidth.
    Against the same 148 MiB over PCIe — ~12 ms per direction at this host's
    measured 12 GB/s — the clone is ~48x cheaper one way and ~96x cheaper than
    the round trip a snapshot-and-restore would pay, and it is four orders of
    magnitude cheaper than re-prefilling the head. The decision to clone on
    the device rather than route through the host tier therefore stands on a
    wider margin than the estimate claimed. See
    [Device prefix clone cost](../findings/2026-09-12-device-prefix-clone-cost.md).
