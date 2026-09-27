# ADR 0041 — the attention readout reads a text span, in whole rows

## Status

Accepted (2026-09-27, owner — spec `docs/specs/decide/18-locate-by-attention.md`
phase B, read as spec 19's track L registered it, GitHub #275). **Extends
ADR 0038 and ADR 0039**, whose readout was an image's: everything both
decided holds for a text span — the job names exactly what to read, exactly
that comes back, the keys are the ones attention read, and a read the leaf
cannot make is a failed question, never a partial set.

Sources: `docs/findings/2026-09-27-a-head-vote-finds-the-line.md` (the
reading), `docs/findings/2026-09-27-locate-by-head-vote-go.md` (the go on
set D), and the acceptance on set F recorded in
`docs/findings/2026-09-27-locate-through-decide.md`.

## Context

A `locate` asks which **segment** of a text `state` — a line, an array
element — an instruction names. Spec 18's single head was a no-go; spec 19's
**head vote** passed spec 18's rule 2 on set D: 32 heads over GQA layers
35-63, each naming the segment whose share of its softmax over the span the
question raised most above a **content-free** twin of the prompt, the
most-voted segment winning.

Three things stood in the way of serving that. The runtime refused a text
job with a readout (`ignis.runtime.attention_without_image`), because the
span was always an image's. The score room was reserved only on a vision
load, sized for one image item. And what the vote reads is not what the seam
carried: ADR 0039 brings back one key per head of a set — its argmax — and a
share is a softmax mass *per segment*, which a single key cannot give.

## Decision

- **The text path carries the readout.** `StepLeaf::prefill` takes the
  `AttentionRead` the multimodal path has; the runtime serves a text job with
  one instead of refusing it, and the leaf arms it on the chunked route
  exactly as for an image (`prefill_program_attention`). The last-chunk tail
  rule, the one band the hq prompt route materializes and the keys read where
  attention read them — a claimed prefix's included — are unchanged.
- **A head set may be read in whole rows.** `SetQuery` names what it reads of
  each head: an image's **peaks** (ADR 0039's argmax, ADR 0040's neighbours
  over a grid) or a text span's **rows** — every head's `q · k / sqrt(256)`
  against every key of the span, `[heads][keys]`, in the set's order. At the
  served vote's 32 heads over 4,554 keys that is 583 KB, against 96 heads'
  6.3 MB of rows that ADR 0039 refused at 4096 px; it crosses because the
  reading needs it. `ignis_prefill_options` grows one appended field,
  `out_attention_set_rows` (ADR 0016); with rows, a grid and neighbours are
  neither needed nor gathered. At most 32 heads a readout reads in rows.
- **The fused kernel writes the rows it already forms.** ADR 0039's launch
  scores every key against every armed head of its layer; with a rows buffer
  it stores each score at its head's slot as well as reducing the argmax.
  No new launch, and the neighbour gather runs only when a grid named
  somewhere to put them. The argmax still rides every set and still says the
  set was read whole.
- **The arithmetic stays on the host.** Softmax per head over the span,
  mass per segment, the baseline subtracted, one vote per head, the tie to
  the best-ranked head — `ignis_core::locate::read_vote`, a pure function
  held to `tools/locate-sets/score.py` by golden cases, three of them real
  questions of set D. The rule can change without a kernel change, as
  ADR 0040 kept for the sub-cell peak.
- **The room is reserved at load, text or vision (ADR 0030).** A load of an
  artifact calibrated for `locate` passes `attention_text_max_keys` — the
  vote's measured ceiling, 4,554 — and reserves, in the one scratch arena,
  one score per key, the set's results and 32 rows: 610,560 bytes on the
  served 27B, counted as the prefill scratch's rather than as vision's. A
  vision load shares it: the image's scores need only what the text scores
  do not already hold. The VRAM plan states it; a prefill that asks for no
  readout allocates and launches nothing new.
- **The vote is a calibrated constant keyed to the artifact's content hash**,
  beside the pointing head: 32 heads and the ceiling, `ignis_core::locate`.
  A load without an entry refuses `locate` (`locate_uncalibrated`) and says
  so at load (`ignis.decide.locate`).

## Considered options

**Reduce per segment on the device** (spec 18's R3 shape): send the
segment boundaries in and bring back `heads x segments` masses. Smaller —
32 floats per segment instead of per token — but it needs each head's
normaliser over the whole span before any mass is known (a second pass, or
rescaled partial sums), moves the rule into the kernel, and ties the seam to
one reading. Rejected while the rows cost half a megabyte on a prefill of
thousands of tokens; worth revisiting if a reading ever needs many more
heads.

**Bring back the argmax only and vote on it** (spec 18's R2). No seam change,
and measured: 48.3% top-1 in cross-validation on A+B against the vote's
90-91.5%. A head's peak key is not its mass.

**Serve the labelled `choice`** in `locate`'s place. It writes labels into
the state — breaking the prefix a `locate` shares with every other question
over it — and stops at 256 segments. Spec 18 keeps it out of scope as a
shipped method.

## Consequences

- Every CPU `Compute` produces rows for a set read in rows: `MockCompute`
  peaks every head a third of the way into the span, higher the longer the
  prompt it is read at the end of — so a test's question, whose instruction
  outruns its twin's `N/A`, votes for its peak's segment (ADR 0006).
- A `locate` costs two prefills of one state: the question's and, once per
  target in a request, its content-free twin's, which claims the question's
  retained state like any sibling. `usage.input_tokens` counts both.
- `ignis_decisions_total` gains `type="locate"`, counted once per question
  and never for its baseline, with no answer mass (ADR 0017).
- `ignis_model_load_options` gains `attention_text_max_keys` without growing:
  the `uint32_t` lands in what was the struct's tail padding, so its size
  stays 48 bytes and ADR 0016's "one recognized size" cannot tell the two
  layouts apart. Accepted because the ABI has one consumer, built with the
  leaf from the same header; a caller compiled against the old header would
  hand over its padding as the key count. The next field appended to this
  struct bumps its size, as ADR 0016 has it.
- The calibration table holds the heads and the ceiling only: the copy
  scaffold and the baseline are the served reading's, not calibrated values.
  A recalibration whose procedure chose another scaffold or no baseline
  would be a new reading, and a new spec, not a new table row.
- A new artifact needs its vote recalibrated: spec 19 phase 0's procedure on
  fresh development sets (`tools/locate-sets/README.md`), then spec 18's
  acceptance on a fresh set through `/v1/decide`.
