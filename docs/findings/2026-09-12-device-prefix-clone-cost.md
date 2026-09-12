# A prefix clone costs 0.33 ms on the device, and how the copy is shaped decides it

- Kind: experiment
- Status: current
- Observed: 2026-09-12
- Last verified: 2026-09-12
- Scope: kernel / device prefix reuse, sequence state transfer
- Related: [GitHub #126](https://github.com/gpillon/ignis/issues/126),
  [ADR 0024](../adr/0024-sequence-state-transfer.md),
  [Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md),
  [Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)
- Superseded by: none

## Question

ADR 0024 splits prefix reuse in two: KV pages are shared in place, and the
mutable sections are cloned device-to-device at "roughly 0.09 ms for 144 MiB
at this card's bandwidth". The half that crosses PCIe was measured when the
host tier landed; this is the half that never leaves the card, and the ticket
asks for it measured rather than assumed.

Two things were open. What does the clone actually cost, and is the decision
to clone on the device rather than route a sibling's snapshot through pinned
host memory still the right one at the real number?

## Evidence

`kernel/tests/test_seq_prefix.cpp` prints the figure on every run. It builds a
sequence pool at the real Qwen 3.8-27B geometry (16 GQA layers of 4 KV heads
x 256, 48 GDN layers of 48 value heads x 128x128, vocab 248,320) under
hq-e8-2b KV, publishes a 20,480-token prefix from one sequence, and times
`ignis_seq_alloc_shared` claiming it. One untimed claim first, then the mean
of three.

RTX 5090, no other process holding the card.

| copy shape | cloned bytes | clone | effective |
| --- | ---: | ---: | ---: |
| one copy per layer per section (96 copies) | 147.76 MiB | 1.513 ms | 102 GB/s |
| one 2D copy per section (3 copies) | 147.76 MiB | 0.329 ms | 471 GB/s |

The cloned state is the mutable half of a sequence and nothing else: 144 MiB
of GDN recurrent state (48 layers x 48 heads x 128x128 fp32), 2.81 MiB of conv
taps, and 0.95 MiB of penalty counts. It does not grow with the prefix. The
prefix in this cell covers 320 KV pages — 180 MiB of history that is shared in
place and never copied at all.

The two rows differ only in how the copy is issued. A slot's GDN state is the
same region of every layer's tensor, strided by the pool's layer stride, and
the prefix's image holds those regions packed. Issuing that as one
`cudaMemcpy2DAsync` per section rather than one `cudaMemcpyAsync` per layer is
the whole difference.

## Finding

**Observed.** The clone costs 0.33 ms for 147.76 MiB at the real geometry,
3.6x ADR 0024's 0.09 ms estimate.

**Observed.** Issuing the same bytes as 96 per-layer copies instead of 3
strided ones costs 4.6x. At 3 MiB per layer-slot the per-copy cost is ~16 us,
which is dispatch, not bandwidth.

**Inference.** 471 GB/s is well under the card's flat-copy bandwidth, and the
remaining gap is the strided source: the pool side of the copy reads 48 rows
of 3 MiB at a layer stride rather than one contiguous 144 MiB run. Nothing
here suggests a further shape would help much — the estimate's implied ~1.6
TB/s is a contiguous-copy figure, and this copy is not contiguous on one side
by construction, because the pool interleaves slots inside each layer.

**The decision holds by a wider margin than the estimate claimed.** One PCIe
direction for the same 148 MiB is ~12 ms on this host
([Sequence snapshot transfer cost](2026-09-12-sequence-snapshot-transfer-cost.md)),
so routing prefix reuse through the pinned-memory tier would cost ~24 ms for
the round trip against 0.33 ms on the card — 73x. Against the work it
replaces, the margin is far larger still: re-prefilling this cell's
20,480-token head is at least ~2.0 s at the measured chunk rate
([Prefill chunk wall time](2026-09-11-prefill-chunk-wall-time.md)), about
6,000x the clone.

## Implications

- **A claim is free at scheduling granularity.** 0.33 ms is a third of one
  decode round's budget and four orders of magnitude under the prefill it
  replaces, so admission has no reason to price a claim differently from a
  plain allocation.
- **The floor is the GDN slot, not the prefix.** A 64-token prefix and a
  40,960-token prefix cost the same clone, because the cloned half does not
  grow with the history. What grows with the prefix is the saving.
- **Shape the copy, not the bytes.** The 4.6x between the two rows is the
  only tuning knob found here, and it was worth taking. Any future state
  section large enough to matter should be moved the same way: one strided
  copy over the pool's layout, not one per layer.
- **hq-e8-2b does not change this number.** The cloned sections carry no KV,
  so the KV format changes only how much history a given prefix shares, never
  what the clone costs.

## Limits and unknowns

- One machine, one geometry, the mean of three claims after a warm-up claim.
  Nothing here establishes variance across driver versions, and the figure was
  taken with the card otherwise idle — a clone issued on the default stream
  while a decode round is in flight has not been measured.
- The measurement times `ignis_seq_alloc_shared` end to end, so it includes
  the sequence's own page reservation and zeroing alongside the clone. Those
  are a few pages at this geometry and the per-layer-versus-2D comparison
  isolates the copy, but the absolute number is "what a claim costs", not
  "what the memcpy costs".
- The prefix's state is written directly rather than produced by a forward
  pass. Transfer cost does not depend on the content of the bytes, and the
  end-to-end claim that a claimant generates what a sibling generates is
  proven separately (`crates/core/tests/prefix_reuse_gpu.rs`).

## Follow-ups

- The G4 gate replays a "1 main + N subagents" trace, where the saving this
  finding prices is what the per-class TTFT cell should show. That run, not
  this synthetic cell, is the production measurement.
