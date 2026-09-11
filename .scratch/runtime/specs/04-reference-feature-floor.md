# runtime 04 — reference feature floor: hq-e8-2b KV, state transfer, device prefix reuse, tagged lanes (gate G4)

GitHub: #65 (phase 4 master; blocked by #64, gate G3 — closed 2026-09-10)

Source: `.scratch/REVIEW-2026-09-05.md` §6 (Phase 4), `.scratch/ROADMAP.md`
(the G4 row and the phase 3–5 candidate decomposition),
`.scratch/DEFERRED-DECISIONS.md` (the ten items the G3 session deferred here),
and the grilling session of 2026-09-11 that turned them into this spec. Builds
directly on `.scratch/runtime/specs/03-serving-loop.md` (G3).

ADRs respected: 0005 (performance-first above a sane-output floor), 0006
(exclusive GPU testing), 0007 (performance gates, correctness self-checked),
0009 (step-level device-resident ABI), 0010 (vendored reference kernels),
0014 (teacher-forced canary floor), 0015 (live/live gate on cold samples),
0016 (extensible options structs at the step ABI), 0018 (chunk-level
interleaving), 0019 / 0020 (decode graph slot indirection, batch-wide round),
0021 (a live/live gate pools at least two launches per engine).
ADRs introduced by this work: **0022** (two KV formats, BF16 retained as the
correctness oracle), **0023** (one eviction priority across GPU residency and
the host tier), **0024** (sequence state transfer: an opaque versioned
snapshot blob, and device-to-device cloning for prefix reuse).

Reference: ninfer at the manifest's pinned commit
(`kernel/vendor/manifest.json`).

## Problem Statement

G3 delivered a serving loop that meets its cells. What it does not have is the
feature floor the reference actually runs on, and four gaps separate the two.

**The KV format is the one thing every gate records as an inequality.** ignis
carries BF16; the reference runs hq-e8-2b. At this geometry a sequence-token
costs 65,536 bytes of BF16 KV (16 GQA layers × 4 KV heads × 256 × K+V × 2
bytes), so eight lanes at a 40,960-token context would need 20 GiB next to
~19 GB of weights. `N-lane concurrency` in `CONTEXT.md` is therefore a
short-context promise today, and the G2 and G3 verdicts each carry the format
difference as a recorded inequality rather than as a measurement. The hq
kernels are already vendored and have never been run: `hq_codec.cuh`,
`gqa_attention_decode_hq*` and `gqa_attention_prefill_hq*` came in with P1-15,
vendored compiled but untested.

**A sequence cannot leave the GPU.** `ignis_seq_snapshot` and
`ignis_seq_restore` exist in the ABI and both return
`IGNIS_SEQ_ERR_NOT_IMPLEMENTED` (`kernel/src/seq.cu:217`). Everything above
them is written: `crates/core/src/host.rs` models a bounded two-tier host tier
with probation, protected and LRU eviction, and `crates/core/src/prefix.rs`
refcounts blocks so siblings can share a prefix. Neither moves a byte of
device state. The glossary says both features work; the engine does neither.

**Nothing tells the engine who is asking.** The reference separates main,
agents and classifier traffic; `admission.rs` collapsed that to `Interactive`
and `Agent` and documented the collapse, but no request can state its class —
the field exists in the scheduler with no way in from HTTP. The G4 load is a
burst of subagents behind one main agent, and the gate is scored per class.

**The gate has never been run.** bench-02 and bench-03 shipped `replay`,
`canary`, `report`, `gate` and `record`; `bench/traces/` holds a README and no
trace. The procedure those specs describe predates ADR 0015 and ADR 0021: it
records one reference run, commits it, and compares against it later, which is
precisely the shape `ignis-bench g2` and `g3-gate` now refuse.

## Solution

**Two KV formats, chosen at load (ADR 0022).** hq-e8-2b becomes the serving
default; BF16 stays, and stays the format every correctness oracle runs
against. The codec keeps the paged-KV contract's fixed-bytes-per-token
property — every (token, KV head) row occupies exactly 64 bytes of code plane
and 8 bytes of metadata — so page addressing, capacity math and CUDA-graph
address stability are unchanged. The pool is sized by a byte budget, not a
token count, and its token capacity is derived from the format in force.

**One description of what a sequence is made of (ADR 0024).** The leaf owns a
state-section table: the KV pages, the GDN slot, the conv taps, the position
and last token, the penalty-count row, and whatever a later phase adds. It is
internal. The ABI exposes only what a caller needs in order to move state — a
snapshot size and a format version — and the blob itself is opaque with its own
header, so a restore refuses a stale or foreign layout instead of corrupting a
sequence. Three consumers share the machinery underneath: snapshot to host,
restore from host, and clone on device.

**Prefix reuse never leaves the card (ADR 0024).** KV pages are read-only
history and are shared by refcount, owned by the leaf where the block table
already lives. The GDN slot and the conv taps are mutable per-sequence state
and are cloned device-to-device, roughly 0.09 ms for 144 MiB at this card's
bandwidth. Routing that through pinned host memory would pay two PCIe
crossings to move bytes that never needed to leave the GPU.

**One eviction priority, two levels (ADR 0023).** GPU residency and the host
tier stop being separate policies. Choosing what leaves the GPU: eligibility
and protection first, then request class, then least-recently-used. Choosing
what the host tier discards: request class first, then probation before
protected, then least-recently-used. The asymmetry is deliberate — on the GPU
something is actively being served, so protection outranks class; in the host
tier nothing is being served, so class is the only thing left that says whose
work hurts most to lose. An `Interactive` snapshot is not discarded ahead of
an `Agent` one merely because it sits in probation.

**A lane says what it is.** The request carries its class as an ignis
extension field, the way `top_k` came in at #101; absent or unrecognized maps
to `Interactive`. Two classes, not three: the recorded trace will show whether
a classifier role exists before one is built for it.

**The gate is live/live, like every gate since G2.** The recorded trace is the
load both engines replay; the reference's *run* is measured in the same
session, never read from a committed file. Two process launches per engine per
ADR 0021, pooled into one verdict.

## User Stories

1. As the engine owner, I want the KV format chosen at model load, so that the serving profile and the oracle profile are the same binary with a different flag.
2. As the engine owner, I want BF16 KV retained and every correctness oracle kept on it, so that a lossy format and a re-derived tolerance never land in the same change.
3. As the engine owner, I want hq's correctness established against ignis's own BF16 route on identical inputs, so that the check is a property of this engine rather than an agreement with another one.
4. As the engine owner, I want the hq tolerance derived from measured codec error, so that it is never a constant copied from an unrelated oracle (#96).
5. As the engine owner, I want the KV pool sized from a byte budget with an auto default and an explicit CLI override, so that capacity is an operator decision and not a constant compiled into the allocator.
6. As the engine owner, I want token capacity derived from the format in force and reported at load, so that a format change shows up as a number rather than as a surprise under load.
7. As the engine owner, I want the standard target profile under hq to hold at least 8 × 40,960 resident tokens, so that N-lane concurrency stops being a short-context promise.
8. As the engine owner, I want the decode graphs captured per exact width 1..8 under hq exactly as under BF16, so that the format change costs no graph machinery.
9. As the engine owner, I want the leaf to own the state-section table, so that exactly one place knows what a sequence is made of.
10. As the engine owner, I want the ABI to expose a snapshot size and a version and nothing more, so that section layout does not leak across the C boundary to consumers that do not need it.
11. As the engine owner, I want the snapshot blob self-describing enough to refuse a foreign or stale layout, so that a restore fails loudly instead of corrupting a sequence.
12. As the engine owner, I want a new state section to be an explicit act against that table, so that the snapshot-point permission is re-earned rather than inherited.
13. As the engine owner, I want snapshot and restore to move a whole sequence in one call per direction over pinned host memory, so that the tier has one cost to measure and one failure to handle.
14. As the engine owner, I want a half-prefilled sequence to be evictable at a chunk boundary, so that work already done survives the eviction G3 could only pause.
15. As the engine owner, I want eviction to run on the admission-refusal path rather than pre-emptively, so that the tier costs nothing while capacity is sufficient.
16. As the engine owner, I want the host tier bounded by a byte budget, so that a tier full of short sequences is priced correctly against one full-context sequence.
17. As the engine owner, I want prefix reuse to share KV pages by refcount inside the leaf, so that the block table and the pages' owner are the same component.
18. As the engine owner, I want mutable per-sequence state cloned device-to-device for prefix reuse, so that a shared prefix costs one intra-card copy and no PCIe traffic.
19. As the engine owner, I want the same internal section machinery behind clone, snapshot and restore, so that a new section is carried by all three or by none.
20. As the engine owner, I want one priority model across GPU residency and the host tier, so that a request's class means the same thing wherever it is standing.
21. As the engine owner, I want a request to declare its class over HTTP, so that the scheduler's class field is reachable from the workload that has the information.
22. As the engine owner, I want the class carried in the request log, so that a per-class gate cell is attributable without a second run.
23. As the engine owner, I want tool-call and thinking streams hardened against a real agent session, so that the dogfood measures the engine rather than the parser.
24. As the engine owner, I want the recorded trace kept out of git with only its hash committed, so that a measurement fixture does not publish a real working session.
25. As the engine owner, I want both engines measured in one session over at least two launches each, so that the verdict is decided by the engines and not by which process each one happened to be.
26. As the engine owner, I want the G3 cells re-run with hq on both sides, so that the inequality recorded beside the G2 and G3 verdicts is retired with a measurement.
27. As the engine owner, I want long-context retrieval checked under hq, so that a lossy KV format cannot pass on throughput while quietly losing the middle of a prompt.

## Implementation Decisions

**Scope boundary.** This phase builds the reference's feature floor. It does
not pack several requests' prefill into one traversal, does not run prefill and
decode concurrently on separate streams, does not add speculative decoding, and
does not build an exact-key side store unless evidence demands one.

**hq-e8-2b (ADR 0022).** Per (token, KV head) row: 64 bytes of code plane, 8
bytes of metadata. Per sequence-token across the 16 GQA layers, both roles and
4 KV heads: 9,216 bytes against BF16's 65,536, a factor of 7.11. The codec's
escalation path (re-encode at alpha/2, then alpha/4) is bounded, deterministic
and host-free, so graph capture stays safe. Format is fixed for the life of a
model load: one capture set per process, widths 1..8 unchanged.

**The pool is a byte budget.** `KvPool` already auto-sizes from a byte budget;
G4 keeps that, adds an explicit CLI override, and derives token capacity from
the format in force. No token target is compiled in. The gate's requirement is
a property of the *standard target profile* — under hq it must hold at least
8 × 40,960 = 327,680 resident tokens, about 3.02 GB at 9,216 bytes per token —
not a constant the allocator enforces.

**The exact-key side store is not built on spec.** The review lists it beside
hq. It is not in the vendored tree, and nothing in this repo can read the
reference's implementation. hq ships without it; if long-context retrieval
comes back short, that result is the evidence that opens a ticket, and the
ticket decides the shape and the size. Nothing is predeclared for it.

**State transfer (ADR 0024).** The section table is leaf-internal. The ABI
gains a snapshot-size query and a version; the blob is opaque and carries its
own header. Restore validates that header and refuses a mismatch. Snapshot and
restore move a whole sequence: partial restore is not offered, because GDN
state is not recomputable without re-running the prefix the restore exists to
avoid.

**What a snapshot costs.** The floor is the GDN slot at 144 MiB plus conv taps
and the penalty-count row (248,320 × 4 bytes ≈ 0.95 MiB), paid regardless of
prompt length. KV dominates at long context: 40,960 hq tokens are about 378 MB,
so a full-context snapshot is roughly 528 MB, near 21 ms per direction at
pinned PCIe rates. Re-prefilling those 40,960 tokens costs about 4.6 s at the
G3-measured chunk rate, so restore beats re-prefill by roughly two orders of
magnitude. That asymmetry is the tier's whole justification and is worth
measuring rather than assuming.

**Eviction (ADR 0023).** Eviction runs on the admission-refusal path. A
sequence is eligible only at a chunk boundary; an in-flight chunk finishes
first. The lane is freed as soon as the snapshot lands. Both tiers full means
admission refuses, exactly as today. The host tier is bounded in bytes.

**Prefix reuse (ADR 0024).** The leaf owns physical pages and their refcounts;
`crates/core/src/kv.rs`'s refcounts become admission accounting rather than
truth, which resolves today's two parallel ledgers over the same pages. A
claimant receives shared KV pages and a device-to-device clone of the mutable
sections. Sharing pages without cloning that state saves nothing, since prefill
must traverse every layer to produce it — stated here so it is not
re-proposed.

**Multi-active prefill is beside the gate, not in it.** One active prefill
ships. N in-flight prefills add no prefill throughput (the chunks stay serial
on the one stream) and buy fairness, which is what the per-class TTFT cell
measures. If that cell passes with one prefill lane, the work is unnecessary;
if it fails, the failure names its own fix. Rationale: `DEFERRED-DECISIONS.md`
item 3 and GitHub #92.

**Tagged lanes.** Two classes. The request carries its class as an ignis
extension field; absent or unrecognized maps to `Interactive`. The class
appears in the request log so the gate's per-class split is attributable. A
third class waits for a workload that has one.

## Gate G4

The bench-03 99% performance gate (ADR 0007) on a recorded "1 main + N
subagents" trace, hq-e8-2b on both sides, measured live/live (ADR 0015) over at
least two launches per engine (ADR 0021).

| cell | measurement | verdict |
|---|---|---|
| per class (`main`, `sub`) | trace replay: TTFT and tok/s per class | pooled ratio ≥ 0.99 against the live reference |
| C=1, C=4, ITL p95 | the G3 cells re-run with hq on both engines | the G3 thresholds, retiring the recorded KV inequality |
| needle retrieval @ 64K / 128K | correctness floor under hq, not a ratio | the planted fact is retrieved |
| canary self-consistency | greedy, fixed seed, sane output | exit 0 |

**Method.** One GPU-exclusive session. Both engines launched twice,
independently — stopped and restarted, not a second bench invocation against a
live process. Every launch's cells are reported; the verdict is the pooled cell
statistic across launches. Per-pair spread is recorded as diagnostic, and a
cell whose across-launch spread exceeds its within-launch spread is itself a
finding. No single-pair threshold is invented here: ADR 0021 defines the pooled
statistic as the gate and defines nothing else.

**The trace.** Recorded from a real agent session against the reference stack
through `ignis-bench record`. It is not committed: it contains real working
content. Its SHA-256 and its shape (request count, class split, arrival span,
prompt-length distribution) are committed with the gate artifacts, and both run
records carry the hash, so a later reader can prove both engines replayed the
same load.

## Testing Decisions

- hq op-level correctness: vendor the reference's hq op tests if they exist in
  the reference tree; otherwise a codec round-trip test with an error bound
  measured on real KV rows. The vendored `test_gqa_attention.cpp` has no hq arm
  to re-enable — its patch dropped I8, and hq was never in it.
- hq route agreement: the hq attention route against the BF16 route on
  identical keys and values at 27B geometry, with the tolerance derived from
  the measured codec error.
- Graph replay under hq: replay matches eager at every width 1..8, and an
  interleaved prefill chunk between two replays changes nothing — the same test
  #102 shipped for BF16.
- Snapshot round-trip: a GPU test that snapshots a sequence, releases it,
  restores it into a fresh handle and continues decoding to the same tokens a
  never-evicted sequence produces from the same seed.
- Blob rejection: a restore given a blob with a mismatched version or size
  fails and leaves the target sequence untouched.
- Prefix reuse: a sibling claiming a cached prefix produces the same tokens as
  a sibling that prefilled the prefix itself, and the shared pages are charged
  to the pool once.
- Eviction priority: CPU tests over both orderings, including the case the
  policy exists for — an `Interactive` snapshot in probation outliving an
  `Agent` snapshot in protected.
- Capacity: admission's view and the leaf's reported pool are cross-checked
  under hq, and the reported token capacity changes with the byte budget and
  with the format.
- The bench cells, needle retrieval included, are CPU-testable end to end
  against a stub endpoint; only the gate run itself needs the GPU (ADR 0006).

## Out of Scope

- Packed prefill: several requests' prefill tokens in one traversal. Its phase
  is decided after #92 reports (`DEFERRED-DECISIONS.md` item 1).
- True prefill/decode overlap on separate streams (roadmap phase 6, ADR 0018).
- More than one active prefill, and priority preemption of a prefill: built only
  if the per-class TTFT cell fails without them.
- The exact-key side store: built only if long-context retrieval under hq comes
  back short.
- Warmup / readiness split: a follow-up ticket, filed and tracked, gating
  nothing in G4.
- A third request class: waits for a workload that has one.
- Speculative decoding (phase 5); vision (past G5).

## Further Notes

- `DEFERRED-DECISIONS.md` item 5 carries a caveat this phase must honour: a new
  state section does not inherit the snapshot-point permission. The section
  table in ADR 0024 is what makes honouring it an act rather than a memory.
- The KV format difference recorded beside the G2 and G3 verdicts is retired by
  the second gate cell, not by argument. Until that run exists, the inequality
  stands as written.
- `crates/bench/src/g2.rs:416` is the note recording that inequality, and the
  only mention of hq anywhere in the ignis tree today.
