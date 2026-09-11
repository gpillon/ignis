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

## Consequences

- The two parallel page ledgers end. The leaf owns physical pages and their
  refcounts; `KvPool`'s refcounts in Rust become admission accounting, not
  truth.
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
