# 19 - where in the text: a span read from attention (research)

> **RESEARCH.** No server behaviour changes. The outcome is a recommendation:
> write a `locate` spec for spans, revive spec 18's line-level phase B, or
> neither. Written 2026-09-27, after spec 18 phase A's no-go.

GitHub: (to be filed)

## Why a second study

Spec 18 asked whether one prefill's attention can say *which line* of a text
answers an instruction, and its pre-registered rule said no
(`docs/findings/2026-09-26-locate-attention-no-go.md`): the best reading, one
head (L39.h12) at the copy scaffold `{"quote":"` with the content-free
baseline, read 73.3% of set C against the labelled `choice`'s 91.3%. The same
run left five leads that the rule was not built to follow:

1. **A line is the coarse question.** Attention is read per *key*, i.e. per
   token. Over an unsegmented text it can name the exact place — the order id
   inside the log line, the date inside the sentence — which a `choice` cannot
   do by construction: its options are segments, at most 256 of them, and a
   choice over single tokens would stop at 256 tokens of state.
2. **One head was the reading.** For `box`, many heads each read a different
   *part* of the object, and the answer came from all of them together —
   spec 14's anchored set: the argmax cells of 96 selected heads, filtered
   around the pointing head's cell, their 10%-90% extent the box
   (`docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`).
   Spec 18's head sets were a vote (R2) and a sum (R3); neither was that, and
   no learned combination was tried.
3. **The attention weights are half of what attention does.** Each key comes
   with a value, and what a head actually moves into the answer position is
   the weighted sum of values, not of weights. A weight on a key whose value
   is near zero moves nothing (Kobayashi et al., EMNLP 2020, "Attention is Not
   Only a Weight"). Spec 18 read weights only.
4. **Lexical questions failed, and the literature predicts the opposite.** ICR
   reports attention *over*-attracted to words the query repeats; here a
   question that repeats a rare word of its target missed 43% of the time,
   far from the target, while paraphrases hit 91%. Something is taking the
   attention; the likeliest candidate is the instruction's own copy of the
   word, which the span never included. Unmeasured.
5. **L39.h10 reads a part, again.** Under the index scaffold `{"line":` the
   pointing head landed on the line *after* the target on 56 of 67 logs (46 of
   67 on set A): in images it reads where a label begins or an object's
   bottom-right corner, in text the boundary after the target. A head with a
   fixed offset is not a wrong head — it is a head whose reading needs its
   offset, and a head that marks where the target *ends* is half of a span.

This spec studies those five, in the order that costs least first.

## Questions

- **Q1 — span.** Over an unsegmented text, does the attention at one position
  land *inside* the answer's span (token hit), and can two readings give its
  edges (span F1, exact match)? Against what the only other one-pass-free
  route gives: generating the answer and searching for it in the state.
- **Q2 — combination.** Does a combination of heads beat the best single head
  — an anchored set in one dimension, or per-head weights learned on the
  development sets (AT2's shape, arXiv 2504.13752) — and does it close the
  lexical gap?
- **Q3 — values.** Does weighting each key by its value's norm
  (`α_j · ||v_j||`) sharpen the reading, per head or combined?
- **Q4 — the lexical paradox.** Where does the attention go on a lexical
  question: how much of the full row lands on the instruction's tokens, the
  scaffold, the sinks, against a paraphrase? Does reading from the
  instruction's own tokens (ICR's direction), or placing the instruction
  before the state, change it?
- **Q5 — part heads.** Which heads have a stable offset from the target (the
  token before it, its first token, its last, the boundary after it), and do
  a start head and an end head together give the span — a box in one
  dimension?

## What exists, and what is spent

- **Instruments** (spec 18, commits 6fa64a2 and 7180882):
  - `tools/locate-sets/`: the logs / records / prose generators, the
    lexical / paraphrase / absent split, `score.py`, `labelled.py`;
  - `crates/server/tests/attention_head_locate_gpu.rs`: the L1 render
    through the endpoint's template provider, every GQA head's scores at the
    scaffold's last position over the state's key span, consumed hq keys
    self-checked, about 11 minutes per 240-question set;
  - `crates/server/tests/support/locate.rs`: byte ranges to tokens (the
    majority rule), reusable with one token per "segment".
- **Dumps** in `.scratch/locate/dumps/` (untracked, this machine): sets A, B
  and C, span-only scores, four prefills per question.
- **Sets.** A (20261010) and B (20261011) are development and may be reused
  as such. **C (20261012) is spent**: its per-question failures were read
  while writing spec 18's finding. **D (20261013) has never been run**: it
  is the check set for track L below, and nothing else may touch it.
- **The tap** (`kernel/include/ignis_attn_tap.h`, feature `attn-tap`):
  query rows at any list of positions, key rows at every position, all 16
  GQA layers; the hq consumed keys only for the chunk holding the **first**
  query position. **It does not capture values.** `ignis_kv_capture.h`
  (feature `kv-capture`) reads K and V rows back from the paged cache after a
  prefill, BF16 pools only.
- **The model.** 16 of its 64 layers have attention; the other 48 are GDN
  and have no keys to read. Everything here is about those 16 × 24 heads.

## Plan

### Phase 0 — offline, on A+B's dumps (no GPU)

Cheap, and it decides what phase 1 dumps.

0. **Literature pass** (the `research` skill) before any reading is
   designed: attention-based answer-span extraction, value-aware attention
   attribution, learned head combinations for attribution (AT2, QRHead,
   ContextCite), and any report of the lexical effect going the other way.
   Written into this spec's References with what each measured.
1. **Q5 at line level.** Per head and scaffold, the distribution of (winner
   segment − target) on A+B; heads with a stable non-zero offset; R1 with a
   per-head offset calibrated in cross-validation. At token level, where in
   the target line each head's peak key falls (first token, last, the
   separator after).
2. **Q2 at line level.** On the 384 heads' per-segment shares (and the `-na`
   ones), 5-fold CV over A+B: logistic regression (segment is target or not),
   the anchored set in one dimension, and the best single head as reference;
   reported per split. Spec 18's rule 1 table is the baseline to beat.

### Phase 1 — the instruments (GPU)

1. **Full-row dumps.** The harness reads every key of the prompt, not only
   the state's span, and labels each token's region (state segment, kind
   text, instruction, scaffold, template). Q4 needs it; the span readings
   need nothing more.
2. **Several query positions**: the scaffold's last token (as today), the
   instruction's tokens (ICR's direction), and, teacher-forced, the first
   tokens of the gold quote after `{"quote":"` — where end heads, if any,
   should show. The last chunk is cut to hold every query position, because
   the consumed hq capture covers only the first query's chunk; BF16 is the
   control where that cannot be kept.
3. **Values.** A value capture beside the key capture in the tap, test-only
   like the rest of it (or the `kv-capture` readback on a BF16 pool if that
   is enough): `||v||` per key and KV head, dumped with the scores.
4. **Span sets**, generated like spec 18's, each with lexical / paraphrase /
   absent and gold **character spans**:
   - logs and records whose target is a *value* inside a line or a record (an
     id, a code, an amount), at 1K / 4K / 16K tokens;
   - SQuAD 2.0 dev (CC BY-SA 4.0): an article's paragraphs as one text, the
     answer's character spans gold, unanswerable questions as absent;
   - HotpotQA distractor dev, extractive answers only (the answer string
     inside a gold sentence).

   Roles: two development sets (E1, E2) and one check set (E3), fresh seeds
   recorded in `tools/locate-sets/README.md`.
5. **The comparator for spans.** The model's own answer, generated greedily
   with thinking off through `/v1/chat/completions`, then found in the state
   by string search (the first occurrence; a string found more than once is
   counted as ambiguous and reported). This is what a caller does today to
   get a position, and its cost is decode rounds. The labelled `choice`
   stays the line-level reference.

### Phase 2 — readings, chosen on E1+E2 by cross-validation

- **Span readings**: one head's argmax key, with its calibrated offset; the
  anchored set in one dimension (centre and 10%-90% extent of the selected
  heads' argmaxes around an anchor); per-head weights learned on E1+E2
  (logistic on "key inside the gold span"), over α and over α·||v||; and the
  **edge pair**, a start head and an end head read as a one-dimensional box.
- **Metrics**: token hit (the reading's key inside the gold span), span F1
  and exact match (token level), character distance to the span; each by
  split, by family and by length; the present/absent AUC of the confidence.
- **Q4**: the full-row mass on the instruction against the state, lexical
  against paraphrase; the ICR-direction read; the instruction-before-state
  render as a diagnostic only (it breaks L1's shared prefix).

### Phase 3 — one pre-registered check per track

Written into this spec, as spec 18 did, **before** E3 or D runs. Proposed
here, to be confirmed by the owner:

- **Track S (spans).** Go — a `locate` spec for spans gets written — if on
  E3 the chosen span reading's token hit is within 5 points of the
  generation route's, and its span F1 within 10, at one prefill (plus the
  baseline's, if chosen) against the generation's decode rounds.
- **Track L (lines, revived).** If a phase 0 or phase 2 reading beats spec
  18's rule-1 choice on A+B's CV, it is judged once on set D against the
  labelled `choice` with spec 18's rule 2 verbatim (within 5 points overall,
  10 on the paraphrase half). A go reopens spec 18's phase B (#275).

A track whose phase 0 or phase 2 answer is already clear stops there: no
check is run to confirm what the development sets have settled.

## Acceptance

1. Phase 0's results, and the literature pass, in a finding with a README
   row (updated as the study goes, or one finding per phase).
2. The harness, generators and scorer extended, each change with its test,
   and the recalibration notes in `tools/locate-sets/README.md` kept current.
3. Phase 3's rules written here before any check set runs; each run check
   recorded as a finding with the recommendation that follows from it.
4. `cargo test` passes workspace-wide.

## Out of Scope

- Any server or production-kernel change. The tap and the value capture are
  test-only, as spec 18's were.
- Training beyond per-head weights: no fine-tuning, no added module (the
  GUI-Actor shape).
- Several spans as one answer, and spans that cross a segment the caller
  would call separate (two log lines).

## Traps already paid for

- The consumed hq keys are captured for the first query position's chunk
  only; the residual window keeps the chunk, the 32 sinks and 512 ring rows
  exact and decodes the rest.
- Every generated set is checked word by word (`common.py`): a bank entry
  that leaks a word fails the generator, not the measurement. HotpotQA's
  paraphrase is weaker (no *rare* word shared), and its draw depends on the
  file's row order.
- A set used to choose anything is spent for judging it.
- Write edit scripts with the Write tool, not shell heredocs: heredocs eat
  backslashes (`\n` escapes in Rust strings broke once in spec 18).
- Three sets of span-only dumps are 11 GB; full-row dumps with values will
  be larger. Check F: before a run.

## References

- Spec 18 and its finding `docs/findings/2026-09-26-locate-attention-no-go.md`.
- Specs 13-15 and `docs/findings/2026-09-22-the-heads-outline-the-object.md`,
  `docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`.
- `docs/findings/2026-09-26-decision-classes-beyond-the-seven.md`: ICR
  (arXiv 2410.02642), QRHead (arXiv 2506.09944), AT2 (arXiv 2504.13752),
  retrieval heads (arXiv 2404.15574), ContextCite (arXiv 2409.00729).
- Kobayashi, Kuribayashi, Yokoi, Inui, "Attention is Not Only a Weight:
  Analyzing Transformers with Vector Norms", EMNLP 2020.
- SQuAD 2.0 (Rajpurkar, Jia, Liang, 2018), CC BY-SA 4.0; HotpotQA (Yang et
  al., 2018), CC BY-SA 4.0.
