---
status: accepted
---

# The `Compute` seam carries an attention readout

A `point` answered by ADR 0034's constrained decode costs a prefill and a decode round per digit, while a better
answer is already inside that prefill: at the position after the forced `{"x":` one attention head attends to the
image token the target's label begins on ([finding](../findings/2026-09-21-one-attention-head-points.md)), and the
heads of the later layers each mark a part of the object, so the cells they peak on near that head's point span it
([finding](../findings/2026-09-23-an-anchored-head-set-points-and-boxes.md)), where that head alone reads a fixed
part of the object, on anything larger than a couple of image tokens its bottom-right corner
([finding](../findings/2026-09-22-the-heads-outline-the-object.md)). So the seam carries a third thing
beside a token and ADR 0034's readout, an **attention readout**, held to the same rule: the job names exactly what
to read, and exactly that comes back.

- **What crosses is what the host's reading uses, and no more.** Of the pointing head: one pre-softmax score per key
  of one span, at the job's last position. Of each head of a head set read beside it: the one key it peaks on over
  the span less the excluded positions (the calibration's fallback cells), its score there, and the scores of that
  key's left, right, up and down neighbours in the image grid, which is what reading the peak's position inside its
  cell needs (a 3x3 neighbourhood read no better). No full attention row and no logits row crosses, and of a head
  set over an image no head's row (a text span's rows are ADR 0041's). A job that asks for no readout allocates and
  launches nothing new, and one that names no set costs what the pointing head alone costs.
- **The leaf scores the keys where attention read them, as it read them**: inside each armed GQA layer, right after
  its attention and inside its scope, under hq-e8-2b from the prompt route's materialized keys and under BF16 from
  the cache's pages. The accuracy was measured on exactly those keys
  ([finding](../findings/2026-09-22-the-codec-costs-the-head-its-read.md)), and a fan-out's second question over one
  image finds that image's keys in the cache, written by an earlier request, so the scores cannot come from a copy
  taken at write time. The kernels are ours, not vendored, and the readout's room comes from the prefill chunk's own
  scratch, reserved by the load's plan, an image's only on a vision load (a text span's room is ADR 0041's).
- **A read the leaf cannot make is a failed question, never a wrong point and never a partial set.** The small-T
  route materializes no keys and a banded prompt keeps only its last band: the leaf then reports the readout unread,
  the prefill itself stands, and the scheduler ends that request with an error. The readout is read only when every
  armed layer read. The scheduler keeps a readout's last chunk long enough to take the prompt route: the chunk cut,
  the reuse trim and the retained prefix's reach all hold back that tail.
- **The heads are calibrated constants keyed to the artifact's content hash.** They were chosen on labelled scenes,
  the pointing head by cross-validation and the set by a selection rule, so they cannot be computed at load, and
  they are never read on an artifact they were not
  chosen for. The head set is calibrated beside its pointing head, with its fallback cells per grid: the first and
  the last image cell on any grid, and the cells measured on a grid. A load with a head set answers a `point` by the
  anchored reading and a `box` with `"method": "head"` by the extent; a load with a pointing head only answers a
  `point` from that head and refuses a head `box`; a load with neither answers both by the digit chain and says so
  at startup.
- **A head point or a head box is a decision.** It reads at its last prefill position and generates nothing, so
  every rule that makes a readout a request kind (no decode lane, no residency, no checkpoint, a zero generation
  budget) holds for it unchanged. A head `box` is read in a point's pass, the point's system text and forced
  `{"x":`, because that is the prompt the set was chosen and measured on; so a fan-out's head boxes share the
  image's prefix with its head points.
- **The arithmetic is the host's.** The leaf ships scores and key indices, and the anchored reading and the sub-cell
  position are pure host functions held to the Python reference every number was measured with, by golden cases the
  reference writes: the rule can change without a kernel change. A neighbour off the grid is absent, not a sentinel:
  a score of minus infinity or zero would be a number the rule could interpolate, and would move a box off the
  object at the image's border. A set named without its neighbours' room or its grid's columns is refused before any
  GPU time: the caller meant a set, and would silently get no way to read inside a cell. A head may not peak on a
  fallback cell, where a head parks when it finds nothing, but as a neighbour that cell is an ordinary one, its
  score still the attention's: every number behind the rule was measured so.

## Considered Options

- **Ship the test-only tap**, a process-wide global that copies every key of an armed layer to the host: rejected.
  It is the wrong cost and
  the wrong keys, the ones before the codec, taken at write time, and under prefix reuse it does not have the
  image's keys at all. It stays test-only, as the independent check the leaf is held to.
- **Score on the host from the logits of digit tokens**: free through ADR 0034's readout, and rejected as the point
  itself: the chain's leading digit is near-certain and its expectation equals its argmax
  ([finding](../findings/2026-09-19-constrained-digit-readout-points.md)), so it carries no more position than the
  chain's first round. It remains a candidate for a second signal
  beside the head.
- **An exact copy of the head's KV head**: worth a few scenes over the consumed keys, and rejected for now: it
  exists only for keys the request itself wrote, so it breaks exactly where a fan-out shares an image's prefix.
- **Copy every head's row and reduce on the host**: megabytes across the seam for a few hundred bytes of answer,
  and each layer's keys read once per head where one fused launch per armed layer reads them once.
- **Launch the one-head kernel once per head**: no new kernel, and several times the fused launch's cost, which is
  still under a percent of the prefill: acceptable, but for nothing.
- **A learned combination of heads**: it has signal in-domain, and fitted on synthetic scenes it does not transfer to
  real screenshots; it needs annotated real images first.
- **Compute the sub-cell offset on the device**: smaller still, but it moves the rule into the kernel, where it
  could no longer be held to the reference as a pure host function.
- **Publish the neighbours from the fused launch itself**: a block scoring a key does not yet know whether it wins,
  so it would publish every key's score, the rows again, or need a device-wide barrier the launch does not have.
  A small gather launch per armed layer reads them instead.
- **Keep cell centres and tune the box's growth**: a box on cell geometry cannot express a button of a cell and a
  half, and every constant growth trades buttons against photographs, the two regimes wanting opposite corrections
  ([finding](../findings/2026-09-23-the-head-box-is-quantised-to-the-cell.md)).

## Consequences

- Every CPU-only `Compute` has to produce the readout, as ADR 0034 asks of every reading the seam carries: the mock
  returns a deterministic one.
- The prefill options grow by appended fields only (ADR 0016).
- A new artifact needs its pointing head recalibrated, and then its head set (`tools/pointing-scenes/README.md`);
  `crates/core/tests/pointing_head_artifact_gpu.rs` fails in the GPU profile until a pointing head is recorded, and
  until a set is the load answers `point` by the pointing head alone.
- `method` stays off the decision metrics, where a label would be a contract change (ADR 0017): a head point counts
  as a `point`, a head box as a `box`.
- Read from a head set, a head `point` and a head `box` answer to half a cell per axis: `uncertainty` is the
  reading's resolution, and the reading resolves inside the cell.
