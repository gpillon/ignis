# ADR 0039 — the attention readout carries a head set

## Status

Accepted (2026-09-23, owner — spec `docs/specs/decide/14-point-and-box-from-the-head-set.md`,
GitHub #263). **Extends ADR 0038**, whose title and first decision say "one
attention head". Everything ADR 0038 decided for that head holds for every
head this one adds: the job names exactly what to read, exactly that comes
back, the keys are the ones attention read, and a read the leaf cannot make
is a failed question.

Sources: `docs/findings/2026-09-22-the-heads-outline-the-object.md`,
`docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`, and the
acceptance recorded for spec 14.

## Context

ADR 0038 made `point` a one-pass answer read from one head, the pointing
head L39.h10. A study of how that head and its neighbours look at an image
found that it reads a fixed **part** of the object — on anything larger than
a couple of image tokens, its bottom-right corner — so its point sits on the
rim of a large object, and falls back to the image's last token when there is
nothing to find. The same study found 96 heads in GQA layers 31 to 63 that
each read a different part of the object the question asked for; the cells
they peak on, kept near the pointing head's point, span the object, and
their extent's centre is the object's centre. That reading needs, per head,
one number — the key it peaks on — and needs it from nine GQA layers at once.

The seam as ADR 0038 left it could carry one head's full score row from one
layer. Carrying 96 rows would be 6.3 MB at 4096 px, where the reading uses
96 indices; and launching the one-head kernel 96 times reads each layer's
keys up to 16 times.

## Decision

- **The readout names a head set beside the pointing head.** A prefill job's
  `AttentionQuery` keeps the span and the pointing head, and optionally names
  a **head set** — a list of (GQA ordinal, query head), each at most once, at
  most 384 — and the **excluded positions**: span-relative keys no head of the
  set may peak on (the calibration's **fallback cells**, at most 32). The
  outcome keeps the pointing head's score row and adds one **argmax key
  index per head of the set**, in the set's order, over the span minus the
  excluded keys, the larger index on a tie. At 4096 px that is 64 KB and 384
  bytes. `ignis_prefill_options` grows by appended fields only (ADR 0016).
- **Every armed layer is read where ADR 0038 reads the pointing head's.**
  Each GQA layer holding a head of the set, or the pointing head, is armed:
  right after that layer's attention, inside its scope, from the hq prompt
  route's plane (query rotated into the codec's frame) or the BF16 pages. The
  rules that make one layer's read possible hold for each — one band, the
  prompt route, the image inside the history — and **the readout is read
  only when every armed layer read**: a partial set is never reported, and
  the question fails as ADR 0038 fails it.
- **One fused launch per armed layer, ours.** A grid of key blocks by the
  four KV heads; each block rotates the armed query heads of its KV head
  once, each warp scores one key row against all of them (the row is read
  once per layer, not once per head), keeps its best (score, key) per head in
  registers, and each block publishes one 64-bit `atomicMax` per head of an
  order-preserving float-to-uint packing with the key index in the low half.
  The pointing head's row is written in the same pass. The per-head results
  are 8 bytes each in the prefill chunk's scratch, zeroed once per chunk and
  copied to the host beside the pointing head's scores, behind the same
  synchronization; a vision load's plan reserves them for the largest set a
  readout may name (3 KB). A job whose readout names no set arms one layer
  and launches ADR 0038's kernel, unchanged, and a job with no readout pays
  nothing.
- **The reading is the host's.** The anchored reading — keep the set's cells
  within twice the median distance of the pointing head's point, span the
  kept cells' 10%-90% quantiles grown half a cell, answer the extent and its
  centre — is a pure function (`ignis_core::pointing::read_anchored`), held to
  the Python reference every number was measured with.
- **The set is calibrated beside the head.** The compiled-in table maps an
  artifact's content hash to the pointing head and, optionally, a head set
  with its fallback cells per grid (the first and the last image cell on any
  grid, and the cells measured on a grid). A load with a head set answers a
  `point` by the anchored reading and a `box` with `"method": "head"` by the
  extent; a load with a pointing head only answers spec 13's `point` and
  refuses a head `box`; a load with neither answers both by chain.

## Considered options

**Copy every armed head's row and reduce on the host.** Simplest seam: the
existing readout, 96 times. Rejected: 6.3 MB per point at 4096 px across the
seam for 384 bytes of answer, and the keys read up to 16 times per layer.

**Launch the one-head kernel once per head.** No new kernel. Rejected on the
prototype (`tools/pointing-scenes/bench_readout.cu`): 0.8-1.6 ms per point
against the fused kernel's 0.14-0.4 ms, still under a percent of the
prefill — acceptable, but for nothing.

**Publish one `atomicMax` per key per head.** The first fused prototype did,
and was 3-5x slower than the per-block reduction: 16 addresses under
contention from every key.

**A learned combination of heads** (a ridge map over all 384 heads per
cell). Has signal in-domain; fitted on synthetic scenes it does not transfer
to real screenshots (20 of 34 inside). Needs annotated real images first.

## Consequences

- `AttentionQuery` is no longer `Copy`: the set rides it as shared slices.
  Every CPU `Compute` produces the set's indices: `MockCompute` peaks head
  `i` on the pointing head's key plus `i % 3`, skipping excluded keys (ADR
  0006).
- A head `box` is read in a **point's** pass — the point's system text and
  forced `{"x":` — because that is the prompt the set was chosen and
  measured on; so a fan-out's head boxes share the image's prefix with its
  head points.
- A new artifact needs its head set recalibrated after its pointing head
  (`tools/pointing-scenes/README.md`, "Recalibrating the head set"); until
  one is recorded the load answers `point` by the pointing head alone.
- `method` stays off the decision metrics (ADR 0017): a head box counts as a
  `box`, a head point as a `point`.
