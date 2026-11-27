---
status: accepted
---

# The attention readout reads a text span, in whole rows

A `locate` asks which segment of a text `state`, a line or an array element, an instruction names, and a vote of
attention heads answers it in a prefill: each head names the segment whose share of its softmax over the span the
question raised most above a content-free twin of the prompt, and the most-voted segment wins
([finding](../findings/2026-09-27-locate-by-head-vote-go.md)). A share is a softmax mass per segment, which the one
key per head that ADR 0038 brings back cannot give. So the text path carries the attention readout the image path
has, and over a text span a head set is read in whole rows: every head's pre-softmax score against every key of
the span. Everything ADR 0038 decides holds for a text span: the job names exactly what to read, exactly that comes
back, the keys are the ones attention read, a claimed prefix's included, and a read the leaf cannot make is a failed
question, never a partial set.

- **The rows cross because the reading needs them**, where ADR 0038 refuses an image's rows, and for a bounded
  number of heads. With rows a grid and neighbours are neither needed nor gathered; the argmax still rides every
  set and still says the set was read whole. The fused launch writes the rows it already forms, so there is no new
  launch.
- **The arithmetic stays on the host**: the softmax per head, the mass per segment, the baseline subtracted, one
  vote per head and the tie to the best-ranked head are a pure function held to the Python that measured the vote
  by golden cases, so the rule can change without a kernel change, as ADR 0038 keeps for a peak.
- **The room is reserved at load, text or vision (ADR 0030).** A load of an artifact calibrated for `locate`
  reserves, in the one scratch arena, one score per key of the widest span a `locate` reads in one prefill, the
  set's results and the rows, counted as the prefill scratch's rather than as vision's; a vision load shares it, the
  image's scores needing only what the text scores do not already hold. The VRAM plan states it, and a prefill that
  asks for no readout allocates and launches nothing new.
- **The vote is a calibrated constant keyed to the artifact's content hash**, beside the pointing head: its heads
  and its ceiling, the longest span it was measured on. A load without an entry refuses `locate` and says so at
  load.

Since ADR 0042 the vote is no longer `locate`'s default: it is served unchanged as one of two methods, and the
readout decided here reads the shortlist's heads too, over that method's windows, which is the span the room is
sized for.

## Considered Options

- **Reduce per segment on the device**, sending the segment boundaries in and bringing back a mass per head and
  segment: smaller, but it needs each head's normaliser over the whole span before any mass is known, moves the rule
  into the kernel and ties the seam to one reading. Rejected while the rows are small beside a prefill of thousands
  of tokens; worth revisiting if a reading ever needs many more heads.
- **Bring back the argmax only and vote on it**: no seam change, and measured far below the vote
  ([finding](../findings/2026-09-26-locate-attention-no-go.md)). A head's peak key is not its mass.
- **Serve a labelled `choice` over the segments in `locate`'s place**: it writes labels into the state, breaking the
  prefix a `locate` shares with every other question over it, and stops at 256 segments.

## Consequences

- Every CPU-only `Compute` produces rows for a set read in rows; the mock's are deterministic.
- A voted `locate` costs two prefills of one state: the question's and, once per target in a request, its
  content-free twin's, which claims the question's retained state like any sibling. `usage.input_tokens` counts both.
- `ignis_decisions_total` gains `type="locate"`, counted once per question and never for its baseline, with no
  answer mass (ADR 0017).
- The load options gained the text span's key count in what was the struct's tail padding, so their size did not
  move: the append ADR 0016 allows under a layout pin, accepted because the ABI has one consumer, built with the
  leaf from the same header (a caller compiled against the older header would hand over its padding as the count).
- The calibration table holds the heads and the ceiling only: the copy scaffold and the baseline are the served
  reading's, not calibrated values. A recalibration whose procedure chose another scaffold or no baseline would be a
  new reading, and a new spec, not a new table row.
- A new artifact needs its vote recalibrated on fresh development sets (`tools/locate-sets/README.md`), then accepted
  on a fresh set through `/v1/decide`.
