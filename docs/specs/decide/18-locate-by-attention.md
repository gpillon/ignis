# 18 - locate: a place in the text evidence, read from attention

GitHub: #274 (phase A), #275 (phase B)

`point` answers "where in the image?" in one pass by reading where the
model's attention already goes (spec 13, 14, 15). This spec asks the same
thing of **text**: *which line of this log is the root cause, which item of
this list is the one I mean, which sentence of this evidence supports the
claim?* The answer is a **segment** of the caller's own `state`, read from
the attention of calibrated heads at one position of one prefill. No label
is written into the state, no option is listed and no token is decoded.

The candidates this was chosen from, and the prior art it leans on, are in
`docs/findings/2026-09-26-decision-classes-beyond-the-seven.md`.

The work is two tickets. **Phase A** calibrates the heads and decides, by
rules written here before any run, whether the attention reading is good
enough to ship. **Phase B** ships it, and is blocked by phase A. A no-go at
the end of phase A ends the work: the spec is then marked **NOT
IMPLEMENTED**, as specs 08 and 12 are, with its finding.

## Problem Statement

An agent that wants to know *where* something is in text has two options
today, and both are poor.

- **Generate it**, through `/v1/chat/completions`. That means a decode round
  per token and a parser that has to survive whatever the model writes.
  Asking for a line number fails in its own way: the model does not count
  lines well.
- **Build a `choice` over labelled segments.** This is Jev's own "line
  search" recipe: prefix every line with an answer label and ask which
  label. It is one readout and it works, but it has three costs:
  1. **The labels change the state's bytes.** A `choice` asked about the
     labelled log and a `noul` asked about the plain one share no prefix.
     Spec 17 is about making every kind share the state; this route undoes
     it for the one question that most needs a long state.
  2. **It stops at the measured ceiling of 256 options.** A build log has
     thousands of lines.
  3. **The caller has to split, label and map back.** That is the parser
     the endpoint exists to remove.

The model already knows where the answer is before it writes anything. When
it is about to name or quote a line, some of its attention heads look at
that line. The engine already reads one head's attention over a span of keys
(ADR 0038), and a head set's argmax over the span (ADR 0039). Today it only
does so over an image.

## Solution

A new primitive, **`locate`**.

- The caller sends the `state` it would send for any other question, plus an
  instruction. Optionally, `within` points at the part of the state to
  search.
- The server splits the target into **segments**: the lines of a string, or
  the elements of an array. It renders the prompt **without changing the
  state's bytes**, forces an answer scaffold, and reads the calibrated
  heads' attention at the scaffold's last position over the state's keys.
- It turns what the heads read into a **share** for each segment, and
  answers with the winning segment and a short ranking.

The whole question is one prefill: no decode round and no lane, the same
class as a head `point`. It needs no labels, so a `locate` and a `choice`
over one state can share that state's prefix. It has no option ceiling
beyond the span it can read.

Three things are unknown and are phase A's job:
- which reading of the attention works on text;
- which heads to read;
- how long a state it holds up to.

## User Stories

1. As an agent reading a build or service log, I want to ask "which line is
   the root cause?" and get the line's index and text in one prefill, so that
   finding it costs what a yes/no question costs.
2. As an agent holding a JSON list (tickets, search results, memories,
   messages), I want the index of the element that answers my instruction,
   so that I can act on it without asking the model to write it back.
3. As a caller checking a claim against evidence, I want the sentence that
   supports it, so that a `noul` "is it supported?" can come with its
   citation.
4. As a caller whose `state` is an object, I want to name the part to search
   with a JSON pointer (`within`), so that I do not have to reshape my state
   for one question.
5. As a caller asking several questions over one state, I want a `locate`
   and my other questions to share the state's prefix, so that locating does
   not force a second copy of a long state. (This is shared across kinds
   once spec 17's layout ships; see Implementation Decisions.)
6. As a caller, I want segments numbered the way I would number them — the
   0-based index of `split("\n")` for a string, the array index for an
   array — so that I can map the answer back without the server's help.
7. As a caller, I want a ranking of the best few segments with their shares,
   so that I can take the second one when the first is not what I needed.
8. As a caller, I want a `confidence` documented as what it is — a share of
   attention, with the measured separation between hits and misses — so
   that I do not read it as a calibrated probability.
9. As a caller, I want a state longer than `locate` was measured on refused
   before any GPU time, naming the ceiling, so that I never get an
   unmeasured answer.
10. As a caller, I want `criteria`, `digits` and `method` refused on a
    `locate`, and `within` refused on every other type, so that a field I
    wrote is never silently ignored.
11. As a caller on an artifact nobody calibrated `locate` for, I want a 422
    before any prefill, so that I never get an answer from heads chosen for
    another model.
12. As an operator, I want a `locate` to hold no decode lane and no residency,
    like a head `point`, so that it does not compete with chat traffic.
13. As an operator, I want the load to log whether `locate` is available, so
    that I know before the first request.
14. As a maintainer, I want the heads to be a calibrated constant keyed to
    the artifact's content hash, with the procedure and harness written
    down, so that recalibration is a repeat and not a new study.
15. As a maintainer, I want the leaf's text-span scores held to the test-only
    attention tap as an independent oracle, under BF16 and hq-e8-2b, and
    also when the state's keys were written by an earlier request, so that
    prefix reuse cannot quietly corrupt the answer.
16. As a maintainer, I want segmentation and every reading rule to be pure
    host functions with table-driven CPU tests, and the mock to produce a
    known attention peak, so that the endpoint is covered without a GPU
    (ADR 0006).
17. As the owner, I want the go/no-go, the reading choice and the length
    ceiling decided by rules written here before phase A runs, and the
    acceptance measured once on fresh sets with floors written before the
    run, so that the shipped default is not tuned to its own test.
18. As the owner, I want the attention route compared with the labelled
    `choice` route on the same questions, so that the price of not writing
    labels is a number.

## Implementation Decisions

### The wire

- `{"type": "locate", "instructions": …, "within": "/log"}`. `within` is an
  RFC 6901 JSON Pointer into the `state`; absent, it is the root.
  - The target must be a **string**, whose segments are its lines split on
    `\n` exactly (a `\r` stays part of its line), or a **non-empty array**,
    whose segments are its elements.
  - Anything else is a 422 that names the pointer and what it found there.
- **The `state` must be JSON** (a string, object or array). A content-parts
  `state` is a 422, `locate_needs_json_state`. Parts are Out of Scope.
- **Refused fields.** `criteria`, `digits` and `method` on a `locate` are
  422s, and so is `within` on any other type. This is the policy `digits`
  already follows: a caller who wrote a field meant something by it.
- **Empty segments.** A segment whose content is empty or all whitespace
  owns no key. It keeps its index and gets a share of 0. Fewer than two
  segments that own keys is a 422.
- **The answer:**

  ```json
  {"type": "locate", "segment": 17, "value": "pg: FATAL too many clients",
   "confidence": 0.62,
   "ranking": [{"segment": 17, "share": 0.62}, {"segment": 4, "share": 0.11}]}
  ```

  - `value` is the segment exactly as the caller sent it: the line as a
    string, or the array element as JSON.
  - `ranking` is the top five segments by share.
  - `confidence` is the winner's share.
  - Shares are not calibrated probabilities, and the documentation says so
    together with phase A's measured separation between hits and misses.
    Shares may sum to less than one: attention that falls outside every
    segment belongs to none.
- `usage.output_tokens` is 0.

### The prompt

- **The state is rendered as it is for every other kind, byte for byte.**
  This is the reason to read attention instead of labels, and it holds
  whatever reading phase A picks.
- **Layout L1 (spec 17) from the start.**
  - System: `{"evidence": …}` alone.
  - User: the `locate` kind text, then `{"instruction": …}`.
  - Assistant: opens with the forced scaffold as prompt.

  `locate` has no L0 prompt to stay compatible with, and L1 is the one
  under which its prefix is shared with other kinds. Until spec 17 moves
  the other kinds to L1, a `locate` shares nothing with them, which is
  exactly today's state for mixed kinds. Sharing among `locate` questions
  over one state works from the start.
- **The query sits inside the answer's scaffold**, as it does for `point`:
  the pointing study found the instruction's own tokens do not transfer.
  Phase A measures two scaffolds, each with its own kind text:
  - **S1 — index-shaped.** `{"line":` for a string, `{"item":` for an
    array. The model is about to name the segment.
  - **S2 — copy-shaped.** `{"quote":"` for either. The model is about to
    copy the segment, which is where copying and retrieval heads look at
    their source (Wu et al., arXiv 2404.15574).

  Phase A picks one, and the other is never served.

### Segments to keys

- **The renderer records each segment's byte range in the rendered prompt as
  it writes it.** Nothing re-finds a segment by searching the text: two
  identical lines are two segments.
- The tokenizer's offsets map byte ranges to token indices.
  - **A token belongs to the segment holding most of its bytes; on a tie,
    to the earlier one.** Separator bytes (the `\n` escape between lines,
    the `,` between elements, JSON quoting) belong to no segment.
  - A token such as `\nERROR` therefore counts for the line it begins, and
    `x\n` for the line it ends.
- The readout's key span runs from the first segment-owned token to the
  last. Keys inside the span that belong to no segment are read, but credit
  nobody.
- All of this is one pure host function.

### The readings phase A chooses between

Each reading comes with its seam cost, and the choice rule is under
Phase A.

| | Reading | What crosses the seam | Seam change |
|---|---|---|---|
| **R1** | One head: softmax of its scores over the span, summed per segment | today's per-key scores | none |
| **R2** | A head set votes: each head's argmax key credits its segment; share = votes / heads | today's per-head argmax | none (the "grid" is one row, so the up and down neighbours are off-grid) |
| **R3** | K ≤ 32 heads' softmax mass summed per segment (QRHead-style), on the device | K × segments floats | the fused readout kernel gains a per-segment reduction; segment boundaries go in |

Any of the three may subtract a **content-free baseline** (ICR's, QRHead's
and contextual calibration's "N/A"):
- a second prefill of the same state, with the instruction replaced by
  `N/A`, read the same way and subtracted per segment before normalising;
- it is one extra prefill per (`state`, `within`) in a request, shared by
  every `locate` over that target;
- its suffix claims the state's prefix like any sibling;
- it counts in `usage.input_tokens`.

Phase A decides whether the baseline is used, as part of the reading.

### The engine

- **The text prefill carries the attention readout.** `StepLeaf::prefill`
  gains the optional `AttentionRead` that `prefill_multimodal` has.
  - The C ABI's attention fields already sit on the options struct both
    paths share, and the leaf arms them on the chunked route whatever the
    media.
  - The runtime stops refusing a text job with a readout
    (`ignis.runtime.attention_without_image`). It keeps refusing a span
    outside the prompt.
  - A request that asks for no readout still allocates nothing and
    launches nothing new.
- **The score room.** The load reserves
  `min(vision_item_max_tokens, max_context_tokens)` scores today: zero or
  near zero on a load without vision, and one image's worth with it. It
  becomes `max(vision item bound, LOCATE_MAX_KEYS)`, capped by the context,
  and is reserved at load (ADR 0030).
  - `LOCATE_MAX_KEYS` is the measured length ceiling from phase A, and the
    endpoint refuses a longer span with a 422 that names it.
  - Under R3, the heads × segments buffer is reserved the same way, with a
    segment cap equal to what phase A measured.
- **Unchanged constraints**, inherited from the head `point`:
  - the last chunk is at least `ATTENTION_MIN_CHUNK_TOKENS` wide, so the hq
    prompt route materializes the keys;
  - the span must lie inside the one band that route materializes
    (262,144 keys);
  - the keys read are the ones attention read: the cache's pages under
    BF16, and under hq-e8-2b the prompt route's planes with the residual
    window wired. In a fan-out those keys were written by an earlier
    request, and only the cache still has them.
- **A `locate` is a readout-class decision:** one prefill, no decode round,
  no lane, no residency. It finishes on its last prefill chunk.
- **Failure is a failed question, never a wrong segment.** If the leaf
  cannot read the span, the question fails with its own error.

### Calibration is a constant keyed to the artifact

- The heads (R1: one; R2: the set; R3: K) and the chosen scaffold, reading
  and baseline flag are a compiled-in table keyed to the artifact's
  **content hash**, beside the pointing table in
  `crates/core/src/pointing.rs`.
- Another load refuses `locate` (422, `locate_uncalibrated`), and the load
  logs which it is.
- There is **no fallback method.** Serving the labelled `choice` in its
  place is Out of Scope.

### Observability and documentation

- `ignis_decisions_total{type="locate"}` is one more value of an existing
  label, and ADR 0017's contract table carries it. A `locate` has no answer
  tokens, so it observes no answer mass, as a head `point` does not.
- **ADR 0041** — *the attention readout reads a text span*. It records:
  - the image-only refusal removed;
  - the score room sized for text;
  - under R3, the per-segment reduction as a new thing crossing the seam.
- `CONTEXT.md` gains *Segment* and *Locate*, and *Attention readout* stops
  saying "an image's placeholders". The OpenAPI document gains `locate`,
  `within` and the answer's fields (ADR 0036).

## Phase A — calibration and go/no-go

### The sets

- A generator in `tools/locate-sets/` (Python), deterministic per `--seed`,
  writes into `.scratch/`.
- **Every set is 240 questions: 80 logs, 80 records and 80 prose.**
- **`logs.py`** — synthetic service logs of 60, 250 and 1,000 lines, which
  is about 1K, 4K and 16K state tokens.
  - One line is the target, among 3 to 8 distractor lines of the same level
    and shape.
  - Half the questions share a rare token with the target (**lexical**).
    The other half share no content word with it (**paraphrase**). This
    split is the guard against ICR's measured lexical bias: a string
    matcher passes the first half and fails the second.
  - One question in six has its target removed (**absent**).
- **`records.py`** — JSON arrays of 20, 80 and 300 records with 5 to 8
  fields. The question selects one by a field value, lexical or paraphrased,
  and one in six is absent.
- **`prose.py`** — HotpotQA distractor dev (CC BY-SA 4.0).
  - The script downloads the dev file from its official URL into
    `.scratch/`; it is never committed.
  - One line per sentence of the ten paragraphs.
  - Top-1 is correct if it is any gold supporting-fact sentence.
- The roles, as for the pointing sets:
  - **A** and **B**: development — heads, reading, scaffold, kind text;
  - **C**: check — go/no-go, reading confirmation, length ceiling, the
    floors;
  - **D**: phase B's acceptance, used once.

  The seeds are recorded in `tools/locate-sets/README.md`. A set used to
  choose something is spent for judging it.

### The harness

`crates/server/tests/attention_head_locate_gpu.rs`, `attn-tap`, GPU profile.

It is the pointing harness's shape, run with:
- the served L1 render and both scaffolds;
- every GQA layer armed;
- the consumed hq keys, with the harness's own self-check (a capture that
  fails it is not a measurement).

It dumps, per head, the scores at the last position over the state's keys,
plus the same for the content-free baseline.

`tools/locate-sets/score.py` scores R1, R2 and R3, each with and without the
baseline, per scaffold, with 5-fold cross-validation over A+B:
- R1 picks one head;
- R2 selects its set by a written mass rule in the style of spec 14;
- R3 selects its top-K heads by QRHead's score.

The pointing head L39.h10 is scored on text as one more R1 candidate.

### The baseline route

The same questions, asked through today's `/v1/decide` as a `choice` over
answer-labelled segments (Jev's line search), wherever a question has at
most 256 segments. It is measured on C, reported, and never shipped.

### The rules, written before the first run

1. **Reading choice** (on A+B, by cross-validation):
   - the best top-1 among R1 and R2, which need no seam change;
   - R3 only if it beats that by **at least 3 points** of top-1;
   - the baseline and the scaffold are chosen by the same measure.
2. **Go** (on C) if the chosen reading's top-1, on questions with at most
   256 segments, is:
   - **within 5 points of the labelled route's top-1** overall;
   - **within 10 points** on the paraphrase half alone.

   Otherwise **no-go**:
   - the finding records why;
   - this spec is marked NOT IMPLEMENTED;
   - the labelled recipe is documented as the way to locate today.
3. **Length ceiling** (on C): `LOCATE_MAX_KEYS` is the longest measured
   length whose top-1 is within 5 points of the reading's own top-1 at the
   shortest length.
4. **Floors for phase B**: C's top-1 per family (logs, records, prose),
   minus a slack of 3% of the family's question count, written into this
   spec before D runs.

### Reported with it, not asserted

- top-3 recall;
- the AUC of `confidence` between present and absent questions (whether a
  `found` flag would be meaningful later);
- the wall time of a `locate` and of the labelled `choice` on the same
  states;
- the content-free baseline's prefill cost.

All of it goes in a finding with a README row.

## Testing Decisions

A good test asserts what a caller or an operator can observe — the segment,
the refusals, the rounds that did not run — and holds the device path to an
**independent** oracle.

- **Segmentation (CPU, pure).** Table-driven:
  - lines, empty and whitespace lines, `\r\n`, identical lines, non-ASCII
    text, escapes;
  - arrays of strings and of objects, a `within` into nested objects;
  - a token straddling a separator, and the majority rule and its tie;
  - fewer than two segments that own keys.
- **Readings (CPU, pure).** For the chosen reading and its baseline:
  - a single peak;
  - a flat map;
  - mass entirely outside the segments;
  - ties;
  - baseline subtraction that goes negative;
  - golden cases written by `score.py`, as the anchored reading has.
- **The endpoint over the mock (CPU).** The mock's attention map has a known
  peak. At `/v1/decide`:
  - the answer is the peak's segment, with `value`, `ranking` and
    `confidence`, and zero decode rounds;
  - every 422 in the wire section fires before any prefill;
  - an uncalibrated artifact refuses;
  - a fan-out mixing `locate`, `choice` and `noul` over one state answers
    all of them;
  - with the baseline, one extra prefill per target, not per question.
- **The scheduler (CPU).** A text job with an attention readout is served
  rather than refused, keeps a last chunk of at least
  `ATTENTION_MIN_CHUNK_TOKENS`, finishes on that chunk and never takes a
  lane.
- **Leaf equivalence (GPU profile, `attn-tap`).** On text prompts, under
  BF16 and under hq-e8-2b with the residual window, the leaf's scores (and
  R2's argmax, or R3's per-segment mass) equal the tap's host-side
  computation for the same heads and position, to within float
  accumulation. This includes a prompt whose state was claimed from a prefix
  an earlier request published.
- **Artifact key (GPU profile).** The served artifact's hash is in the
  table. The test fails when it is not, naming the recalibration procedure.
- **Prompt pinning (CPU).** The endpoint's `locate` render is byte-identical
  to the render phase A's harness measured. The heads were chosen on those
  bytes, and a reworded prompt is an unmeasured one (ADR 0034).
- **Room and cost.** A load without vision reserves the text room, and the
  VRAM plan's test states the new bytes. A prefill with no readout launches
  nothing new.

## Acceptance

### Phase A

1. **Phase A's finding exists** with a README row: the reading, heads,
   scaffold and baseline chosen by rule 1; the go/no-go by rule 2, with the
   labelled route beside it; `LOCATE_MAX_KEYS` by rule 3; the floors by
   rule 4, written into this spec. The sets' seeds and the recalibration
   steps are written down beside the tools. On a no-go the work ends here:
   the spec is marked NOT IMPLEMENTED and phase B is closed as not planned.
2. `cargo test` passes workspace-wide.

### Phase B

3. **The leaf reads text.** Leaf equivalence holds under both KV formats,
   including over a claimed prefix. A text job with a readout is served.
4. **`/v1/decide` over the mock** passes the Testing Decisions list, and
   segmentation and the reading have their table-driven CPU tests.
5. **The calibration is keyed to the artifact.** The GPU-profile test
   asserts the served hash, and the load logs whether `locate` is
   available.
6. **The served prompt is the calibrated one.** A prompt-pinning test holds
   the `locate` render (L1, the kind text and the scaffold phase A chose)
   byte-identical to the render phase A's harness measured.
7. **Nothing for everyone else.** A prefill with no readout allocates and
   launches nothing new. The only new reservation is the text score room
   (and R3's buffer), stated in the VRAM plan.
8. **The pre-registered acceptance holds.** Set D, used once, through
   `/v1/decide`, on the served artifact, hq-e8-2b with the residual window:
   top-1 at or above rule 4's floor for each family. Reported beside it:
   top-3, the present/absent AUC, and the wall time against the labelled
   route. Recorded as a finding.
9. ADR 0041, `CONTEXT.md` (*Segment*, *Locate*, *Attention readout*), ADR
   0017's contract table and the OpenAPI document are updated.
10. `cargo test` passes workspace-wide.

## Out of Scope

- **The labelled route as a shipped method** (`"method": "labels"`), and any
  fallback for uncalibrated loads. It is measured here as the baseline and
  nothing more.
- **Content-parts states**, including text beside an image.
- **Several segments as one answer** (all the lines that match), and spans
  finer than a segment (character offsets inside a line). The first is
  `multi`'s shape over segments; the second is a sub-segment reading, like
  spec 15's, and wants its own measurement.
- **`rank` as a contract.** `ranking` gives the top five, but only top-1 is
  held to a floor. A ranking primitive promises more than that, and wants
  its own sets (ICR's and QRHead's BEIR shape).
- **A `found` flag or an abstention**, until the present/absent AUC says it
  would mean something.
- **The Playground's Decide tab** learning `locate`: a follow-up, as the tab
  followed spec 13.
- **Moving the other kinds to L1.** That is spec 17's to decide.

## Further Notes

- **Why attention and not labels, in one line:** the labels are the only
  thing that makes the labelled route work, and they are the thing that
  breaks sharing, caps the width at 256 and puts a mapping on the caller.
- **Why the lexical/paraphrase split is the one that matters.** ICR's own
  paper reports lexical bias and weak entity matching. A head that matches
  strings would pass a set built from questions that quote their answer, and
  this spec's go rule would say yes to a string matcher. The paraphrase half
  exists so that it cannot.
- **What the pointing work predicts, and does not.**
  - L39.h10 reads a fixed *part* of an image object, where a label begins.
    A text head reading the first token of its segment is the analogue, and
    segment-level credit makes it harmless.
  - The heads may also fall back to sink tokens (the first token of the
    evidence, the newlines) when nothing matches, as image heads do on the
    fallback cells. The absent questions exist to see that.
- **Recalibration** for a new artifact is phase A with fresh seeds and the
  rules above. It needs no new study, and `tools/locate-sets/README.md`
  writes the steps down as `tools/pointing-scenes/README.md` does for
  `point`.

## References

- Finding: `docs/findings/2026-09-26-decision-classes-beyond-the-seven.md`
  (the candidates, and the prior art: ICR arXiv 2410.02642, QRHead arXiv
  2506.09944, AT2 arXiv 2504.13752, Retrieval Heads arXiv 2404.15574).
- ADR 0034 (a decision reads, and does not generate), ADR 0038/0039/0040
  (the attention readout, the head set, the neighbours), ADR 0030 (memory
  reserved at load), ADR 0017 (the metrics contract), ADR 0036 (the
  OpenAPI document).
- Specs 13-15 (the head `point` and `box`, whose shape this copies), spec 17
  (layout L1), spec 03 (validation before the GPU).
- `crates/server/tests/attention_head_point_gpu.rs` and
  `tools/pointing-scenes/`, the harness and tooling this copies.
