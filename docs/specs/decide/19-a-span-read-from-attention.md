# 19 - where in the text: a span read from attention (research)

> **RESEARCH.** No server behaviour changes. The outcome is a recommendation:
> write a `locate` spec for spans or for profiles, revive spec 18's
> line-level phase B, build the hybrid of track H, or none of them. Written
> 2026-09-27, after spec 18 phase A's no-go.
>
> **Read this first if you are the agent picking this up.** § Why a second
> study says what is known; § The frame is the idea that organises the rest;
> § Questions and § Plan say what to do; § For the researcher is the map —
> where every tool, dump and number is, how to run them, and the traps
> already paid for.

GitHub: #276

## Why a second study

Spec 18 asked whether one prefill's attention can say *which line* of a text
answers an instruction, and its pre-registered rule said no
(`docs/findings/2026-09-26-locate-attention-no-go.md`). The best reading, one
head (L39.h12) at the copy scaffold `{"quote":"` with the content-free
baseline subtracted, read 73.3% of set C against the labelled `choice`'s
91.3%. The same run left leads the rule was not built to follow:

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
   Spec 18's head sets were a vote (R2) and a sum (R3), and neither beat one
   head; no anchored or learned combination was tried.
3. **The attention weights are half of what attention does.** Each key comes
   with a value, and what a head moves into the answer position is the
   weighted sum of values, not of weights: a weight on a key whose value is
   near zero moves nothing (Kobayashi et al., EMNLP 2020). And this model's
   attention is **output-gated** — `attn_output * sigmoid(gate)`, the gate
   computed from `q_proj` — so a head whose gate is shut contributes nothing
   whatever its weights say (found by the pointing study's PyTorch vehicle,
   `.scratch/latent-probe/attention.py` in the `point-one-pass` worktree).
   Spec 18 read weights only, and chose heads by what they attended to, not
   by whether the model listened to them.
4. **Lexical questions failed, and the literature predicts the opposite.** ICR
   reports attention biased *towards* lexical overlap; here a question that
   repeats a rare word of its target missed 43% of the time, far from the
   target (31 of 34 misses on logs and records more than two segments away,
   often on an early line), while paraphrases hit 91%. The likeliest
   candidate is the instruction's own copy of the word, which the dumped
   span never included; an early-line bias (sinks, position) is the other.
   Unmeasured.
5. **L39.h10 reads a part, again.** Under the index scaffold `{"line":` the
   pointing head landed on the line *after* the target on 56 of 67 logs of
   C (46 of 67 on A). In images it reads where a label begins, or a large
   object's bottom-right corner; in text, the boundary after the target. A
   head with a fixed offset is not a wrong head — its reading needs its
   offset — and a head that marks where the target *ends* is half of a span.
   A +1 offset is also exactly what an induction head does (attend to the
   token *after* an earlier match; Olsson et al. 2022), which is worth
   checking before calling it anything else.
6. **The shortlist is almost always right.** The chosen reading's top 3
   segments held the target on 94.5% of C's present questions — above the
   labelled route's top-1. A second, small question over the shortlist could
   decide what one head could not (track H).

## The frame: a profile over the input

*The owner's idea, 2026-09-27.* Put the state's tokens on the x axis. Any
signal one prefill produces for each token is an ordinate `y = f(x)`: a
**profile** of the input. Every reading so far is one choice of `f` and one
way of reading it — spec 18's R1 is one head's attention weight at the
scaffold's last token, summed per line and read at its maximum; spec 14's
anchored set is where many heads' profiles peak. So the study has two
organising questions:

1. **Which ordinates are worth reading** — heads, combinations of heads,
   values, gates, the lift over the content-free baseline, profiles read
   from other query positions; and
2. **what a profile's shape says** — its peaks, their heights and widths —
   so that `locate`'s answer can be the profile's peaks with their values.

### Candidate ordinates

- `α_h(x)`: one head's attention weight from a query position (384 profiles
  per query position).
- **Lift**: `log α_h^q(x) − log α_h^{N/A}(x)`, what the question adds to what
  the head does with no question at all (sinks, first lines, position). Spec
  18 subtracted shares per line; a token-level log-ratio is a different
  quantity (contextual calibration's move, Zhao et al. 2021).
- **Value- and gate-weighted**: `α_h(x) · ||v_h(x)||`, and
  `g_h · α_h(x) · ||W_O v_h(x)||` with the head's output gate `g_h` at the
  query.
- **Combinations**: `w · A(x)` with per-head weights learned on development
  sets (AT2); a per-position classifier over the 384-vector of head values at
  `x` (nonlinear); anchored sets (spec 14).
- **Role curves**: a *start* (initiator) curve, an *end* (terminator) curve,
  an *inside* curve — the way extractive QA models read a span's edges from
  start and end scores and take the best `i ≤ j` (BERT's span decoding,
  Devlin et al. 2019). L39.h10 under the index scaffold is the first
  terminator candidate; initiators are to be found.
- **Query-position variants**: the scaffold's last token, the instruction's
  tokens (ICR), the teacher-forced quote's tokens.
- **The KV alone**, with the limit below.

### What the KV can and cannot say

Under layout L1 the state comes before the question, and the model is
causal: every key and value of the state is computed before the question
exists, so they are **the same for every question over that state** — which
is exactly why the prefix can be shared. The KV carries what the text is
(which tokens are salient, what each would contribute if attended), not what
this question wants. The question reaches the state only through queries at
later positions — the attention above — and through what later layers do
with what those queries read.

So a KV-only ordinate is a **prior** over the state: useful as a weight or a
normaliser (`||v||`; a salience profile shared by every question), or made
question-specific by a query. The instruction-before-state render of Q4 is
the one layout in which the state's KV depends on the question; measuring
there says how much question-conditioned KV would add, at the cost of the
shared prefix.

### What makes an ordinate interesting

Measured on development sets only:

1. **Localisation**: its peaks sit on gold spans — per question, the
   position-level AUROC and average precision of `y(x)` for "x inside a gold
   span", before any peak picking.
2. **Separation**: peak height separates present from absent questions, and
   gold peaks from other peaks.
3. **Length invariance**: a height means the same at 1K and at 16K tokens
   (softmax mass shrinks as 1/N; lift and calibrated heights may not).
4. **Complementarity**: a second ordinate adds to the first (initiator with
   terminator; a lexical-robust one with a paraphrase-robust one).
5. **Cost**: what would cross the seam. One head's per-key scores already
   do (ADR 0038); many heads combined on the device is spec 18's R3-style
   kernel change.

### The answer this would give

`locate` would return peaks, not a segment: for each chosen ordinate, the
local maxima above a threshold, each with its position (a span, from
initiator and terminator curves or from the peak's width) and a height in
[0, 1] calibrated on development sets as the probability that the peak sits
on a gold span. A caller then reads several answers (every peak above a
threshold), none (no peak above it: absent), or a ranking (by height), and
several ordinates give several such vectors — start, end, relevance. In one
prefill that would answer three shapes spec 18 kept out of scope, and which
`docs/findings/2026-09-26-decision-classes-beyond-the-seven.md` lists among
the classes `/v1/decide` lacks: several spans (`multi`), a ranking (`rank`)
and "not there" (a `found` flag).

## Questions

- **Q1 — span.** Over an unsegmented text, does the attention at one position
  land *inside* the answer's span (token hit), and can two readings give its
  edges (span F1, exact match)? Against the route a caller has today:
  generating the answer and searching for it in the state.
- **Q2 — combination.** Does a combination of heads beat the best single head
  — an anchored set in one dimension, or per-head weights learned on the
  development sets (AT2's shape) — and does it close the lexical gap?
- **Q3 — values and gates.** Does weighting each key by its value
  (`α_j · ||v_j||`, or the head's output contribution `α_j · W_O v_j`), or
  each head by its output gate at the query, sharpen the reading?
- **Q4 — the lexical paradox.** Where does the attention go on a lexical
  question: how much of the full row lands on the instruction's tokens, the
  scaffold, the sinks, the first lines, against a paraphrase? Does reading
  from the instruction's own tokens (ICR's direction), or placing the
  instruction before the state, change it?
- **Q5 — initiators and terminators.** Which heads, or combinations, have a
  stable offset from the target — the token before it, its first token, its
  last, the boundary after it? Is L39.h10 a terminator, and where are the
  initiators? Are they induction-like? Do an initiator and a terminator
  together give the span — a box in one dimension?
- **Q6 — the hybrid (track H).** Attention proposes the top k segments;
  a `choice` whose options *carry those segments' text* (the state is not
  relabelled, so the prefix stays shared) picks one. Does it reach the
  labelled route's accuracy at a fraction of its prompt, and past 256
  segments?
- **Q7 — profiles and peaks (track P).** Which ordinates of § The frame
  localise, separate and hold across lengths, and does reading a profile's
  peaks with calibrated heights answer several-span, ranked and absent
  questions in one prefill?

## Plan

### Phase 0 — offline, on A+B's dumps (no GPU)

Cheap, and it decides what phase 1 dumps.

0. **Literature pass** (the `research` skill) before any reading is
   designed. Start from § References; look specifically for: training-free
   answer-span extraction from attention; value- and gate-aware attention
   attribution; learned head combinations for attribution; attention sinks
   and position bias in long contexts; copy/induction heads and offsets;
   heads that mark the start or the end of a copied span; token-level
   relevance profiles read as peaks (attribution curves, peak picking,
   calibrated heights, several answers or none); any report of repeated query
   words *hurting* attention-based retrieval. Record what each measured, on
   which model, in a finding.
1. **Q5 at line level.** Per head and scaffold, the distribution of (winner
   segment − target) on A+B; heads with a stable non-zero offset; R1 with a
   per-head offset calibrated in cross-validation. At token level, where in
   the target line each head's peak key falls: first token, last token, the
   `\n` escape after it, the next line's first token. The dumps hold every
   key of the state's span, separators included, so this is readable now.
2. **Q2 at line level.** On the 384 heads' per-segment shares (and the `-na`
   ones), 5-fold CV over A+B: logistic regression (segment is target or not),
   the anchored set in one dimension, and the best single head as the
   reference; reported per split. Spec 18's rule 1 table is the bar.
3. **Q6 upper bound.** The top-k recall of the best readings on A+B, k = 1,
   2, 3, 5: the ceiling any shortlist-then-choose hybrid can reach.
4. **Q4, what can already be seen.** On A+B's lexical misses, where the
   winner falls (early lines, other ERROR lines, the question's own words in
   other lines) and whether the content-free baseline moved it.
5. **Span-aligned averages (Q5, Q7).** For every head and scaffold, the mean
   profile aligned on the target's first token and on its last token (−16 to
   +16 tokens), raw and lift — an event-related average. A head whose average
   peaks at the start is an initiator candidate; one that peaks just after
   the end, a terminator (L39.h10 under the index scaffold is the
   hypothesis). Combinations follow from the same table.
6. **Ordinate quality at token level (Q7).** Per question, the position-level
   AUROC and average precision of each head's `α` and lift profile for "key
   inside the target line", averaged by split and by length: the first table
   of ordinates, before any peak is picked. The dumps hold both profiles
   (`s1`/`s2` and their `-na`) over every key of the span.

### Phase 1 — the instruments (GPU)

1. **Full-row dumps.** The harness reads every key of the prompt, not only
   the state's span, and labels each token's region (state segment, kind
   text, instruction, scaffold, template). Q4 needs it; nothing else changes.
2. **Several query positions**: the scaffold's last token (as today), the
   instruction's tokens (ICR's direction), and, teacher-forced, the first
   tokens of the gold quote after `{"quote":"` — where end heads, if any,
   should show. Cut the last chunk to hold every query position: the
   consumed hq capture covers only the first query's chunk. BF16 is the
   control where that cannot be kept.
3. **Values and gates.** Either a value capture beside the key capture in
   the tap (test-only, like the rest of it) with `||v||` per key and KV
   head, or the `kv-capture` readback on a BF16 pool; the output gate per
   head at the query position needs one more tap point (or the vehicle, see
   below). Decide by what phase 0 says is worth it.
4. **Span sets**, generated like spec 18's, each with lexical / paraphrase /
   absent and gold **character spans**:
   - logs and records whose target is a *value* inside a line or a record
     (an id, a code, an amount), at 1K / 4K / 16K tokens;
   - SQuAD 2.0 dev (CC BY-SA 4.0,
     `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json`,
     4.4 MB, reachable 2026-09-27): an article's paragraphs as one text, the
     answer's character spans gold, unanswerable questions as absent;
   - HotpotQA distractor dev, extractive answers only (the answer string
     inside a gold sentence);
   - for Q7, questions with **several or no** gold spans: HotpotQA's
     supporting facts (two or more per question), and logs asked "which
     lines report …" with zero to three matching lines.

   Roles: two development sets (E1, E2) and one check set (E3), fresh seeds
   recorded in `tools/locate-sets/README.md`.
5. **The comparator for spans.** The model's own answer, generated greedily
   with thinking off through `/v1/chat/completions`, then found in the state
   by string search (the first occurrence; a string found more than once is
   counted ambiguous and reported). That is what a caller does today to get
   a position, and it costs decode rounds. The labelled `choice` stays the
   line-level reference.

### Phase 2 — readings, chosen on E1+E2 by cross-validation

- **Span readings**: one head's argmax key, with its calibrated offset; the
  anchored set in one dimension (centre and 10%-90% extent of the selected
  heads' argmaxes around an anchor); per-head weights learned on E1+E2
  (logistic on "key inside the gold span") over α, α·||v|| and gate-weighted
  α; and the **edge pair**, a start head and an end head read as a
  one-dimensional box.
- **Metrics**: token hit (the reading's key inside the gold span), span F1
  and exact match (token level), character distance to the span; each by
  split, by family and by length; the present/absent AUC of the confidence.
- **Q4**: the full-row mass on the instruction against the state, lexical
  against paraphrase; the ICR-direction read; the instruction-before-state
  render as a diagnostic only (it breaks L1's shared prefix).
- **Q6**: the hybrid on E1+E2 through a live server — pass 1 the attention
  shortlist (from the dumps), pass 2 a `choice` over the k segments' text
  as options over the *unlabelled* state.
- **Q7, profiles**: the ordinates ranked by the criteria of § The frame; for
  the best few, a peak reading (smoothing, peak picking, a minimum
  separation, a threshold) and heights calibrated as P(peak on a gold span),
  all chosen on E1+E2; measured as peak precision and recall against the
  gold spans, top-peak hit, the absent AUC, and a ranking metric (NDCG)
  where a question has several golds. Initiator and terminator curves read
  together as spans.

### Phase 3 — one pre-registered check per track

Written into this spec, as spec 18 did, **before** E3 or D runs. Proposed
here, to be confirmed by the owner before the run:

- **Track S (spans).** Go — a `locate` spec for spans gets written — if on
  E3 the chosen span reading's token hit is within 5 points of the
  generation route's, and its span F1 within 10, at one prefill (plus the
  baseline's, if chosen) against the generation's decode rounds.
- **Track L (lines, revived).** If a phase 0 or phase 2 reading beats spec
  18's rule-1 choice on A+B's CV, it is judged once on set D against the
  labelled `choice` with spec 18's rule 2 verbatim (within 5 points overall,
  10 on the paraphrase half). A go reopens #275 (spec 18 phase B).
  **Registered 2026-09-27, after phase 0 and before C or D was read with
  it** (`docs/findings/2026-09-27-a-head-vote-finds-the-line.md`). Phase 0
  found such a reading, the **head vote**:
  - *The reading.* The copy scaffold (`s2`) and its content-free prefill
    (`s2-na`). Each of K heads names its R1 winner — the segment with its
    largest share of the head's softmax over the span, less the `-na`
    prefill's share of the same segment — and the answer is the segment
    with the most votes; a tie goes to the tied segment named by the
    best-ranked head. The confidence is the winner's share of the votes.
    `tools/locate-sets/score.py` reading `vote`.
  - *The heads*, K = 32, best first, fitted on all of A+B by the procedure
    below (`.scratch/locate/phase0/vote-choice.json`): L39.h12, L47.h20,
    L59.h16, L55.h17, L59.h17, L59.h7, L59.h6, L55.h0, L59.h8, L59.h10,
    L63.h5, L39.h15, L59.h2, L55.h21, L39.h23, L55.h9, L39.h10, L63.h0,
    L35.h16, L55.h14, L51.h6, L47.h17, L51.h12, L59.h12, L55.h18, L47.h15,
    L43.h13, L59.h9, L63.h1, L59.h14, L47.h18, L43.h9.
  - *The procedure.* The heads are the K with the most training hits as R1
    (score.py's order); scaffold, baseline and K ∈ {1, 3, 5, 8, 16, 32} are
    the configuration with the most cross-validated hits, ties to fewer
    heads, then no baseline, then S1. On A+B: plain 5-fold CV 368/402 =
    91.5%; the whole procedure under nested CV 362/402 = 90.0%; K = 5, 8 and
    16 within three questions of K = 32. Spec 18's rule-1 choice: 70.4%.
  - *The rule, spec 18's rule 2 verbatim.* **Go** if on D the vote's top-1
    is within 5 points of the labelled `choice`'s overall, and within 10 on
    the paraphrase half, on D's present questions with at most 256 segments.
    Rules 3 and 4 (the length ceiling and the per-family floors) are
    computed as spec 18 computes them, for phase B.
  - *What a go does.* It reopens #275 with the vote as the reading in place
    of R1 — a head set on the seam, as ADR 0039 did for `box` — and its
    floors and ceiling from D. A no-go ends track L; the finding records it.
  - *Not binding, reported beside it:* the conditional logit over every
    head's per-segment lift, both scaffolds (four prefills), 93.3% in nested
    CV on A+B; and C, read with the frozen vote as a replication only —
    C chose nothing here, but its failures were read in spec 18, so it
    cannot judge.
- **Track H (hybrid).** Judged on D the same way as track L, with its cost
  (two prefills of one shared state against the labelled route's one of a
  labelled state) reported beside it. *After phase 0:* the vote reaches the
  labelled route's level on A+B by itself, so track H runs only if track L
  is a no-go (the vote's top-3 recall on A+B is 98.5% in CV, the hybrid's ceiling).
- **Track P (profiles).** Its rule is proposed at the end of phase 2, from
  what the development sets show — there is no honest bar to write for
  several-span recall or absent detection before any profile is measured —
  and confirmed by the owner before E3 runs. The comparator for several
  spans is the generation route asked for all of them.

A track whose answer is already clear on the development sets stops there:
no check is run to confirm what they have settled.

## Acceptance

1. Phase 0's results and the literature pass in a finding with a README row
   (updated as the study goes, or one finding per phase).
2. The harness, generators and scorer extended, each change with its test,
   and `tools/locate-sets/README.md` kept current (sets, seeds, commands,
   recalibration).
3. Phase 3's rules written here before any check set runs; each check run
   recorded as a finding with the recommendation that follows from it.
4. `cargo test` passes workspace-wide.

## Out of Scope

- Any server or production-kernel change. The tap and any value or gate
  capture are test-only, as spec 18's were.
- Training beyond per-head weights: no fine-tuning, no added module (the
  GUI-Actor shape).
- *Shipping* several spans as one answer: the study measures multi-peak
  profiles, a later spec would ship them. Spans that cross segments a caller
  would call separate (two log lines) stay out.

## For the researcher

### Where things are

| What | Where |
|---|---|
| The spec 18 study, rules and banner | `docs/specs/decide/18-locate-by-attention.md` |
| Its finding (all numbers below) | `docs/findings/2026-09-26-locate-attention-no-go.md` |
| Set generators, scorer, labelled client, their tests, README | `tools/locate-sets/` (`generate.py`, `logs.py`, `records.py`, `prose.py`, `common.py`, `score.py`, `labelled.py`, `test_score.py`, `test_sets.py`) |
| The GPU harness (dump every head over a text state) | `crates/server/tests/attention_head_locate_gpu.rs` |
| Render, segments, byte ranges → tokens, chunking (pure) | `crates/server/tests/support/locate.rs`, tests in `crates/server/tests/locate_segments.rs` |
| Token byte offsets | `ignis_artifact::Tokenizer::encode_with_offsets` (`crates/artifact/src/frontend.rs`) |
| The attention tap (Q and K rows, consumed hq keys) | `crates/core/src/attn_tap.rs`, `kernel/include/ignis_attn_tap.h`, feature `attn-tap` |
| K/V readback from the paged cache (BF16) | `Seq::capture_kv_rows_for_test` (`crates/core/src/seq.rs`, `role` 0 = K, 1 = V), `kernel/include/ignis_kv_capture.h`, feature `kv-capture` |
| Which hq row came from where (fresh / sink / ring / codec) | `crates/core/src/hq_ring.rs` (`prompt_source`, `ring_before_chunk`) |
| The image-side precedent: one head, head set, anchored reading | specs 13-15; `crates/core/src/pointing.rs` (`read_anchored`, `Calibration`, `SERVED_NVFP4_27B_SET`, `ATTENTION_MIN_CHUNK_TOKENS`); `tools/pointing-scenes/` (`ensemble_score.py select/score/golden`, `c5_score.py`); `crates/server/tests/attention_head_point_gpu.rs` |
| The served attention readout (what a shipped `locate` would extend) | ADR 0038 / 0039 / 0040, `kernel/src/attention_readout.cu`, `ignis_core::pointing::AttentionQuery` |
| The PyTorch vehicle (weights, values, gates, any layer, no kernel work) | `../.inference-qwen-worktrees/point-one-pass/.scratch/latent-probe/` (`attention.py`, `attention_score.py`, `vehicle4096.py`), venv `F:/ai/ngram-venv`, model `Y:/models/Qwen3.8-27B` |

### The dumps spec 18 left (phase 0's input)

- `.scratch/locate/{A,B,C,D}/manifest.json`: the sets (`questions[]`: `id`,
  `family`, `kind`, `split`, `absent`, `segments`, `state`, `instruction`,
  `targets`, `distractors`, `event`).
- `.scratch/locate/dumps/<set>-hq.{bin,jsonl,json,features.npz}` for A, B, C.
  - `.bin`: f16 little-endian; per question, per variant in the order `s1`,
    `s1-na`, `s2`, `s2-na`: `[GQA ordinal 0..16][query head 0..24][span key]`,
    scores `q · k / 16` before softmax, at the scaffold's last token.
  - `.jsonl`, one row per question: `span` [start, length] in prompt tokens;
    `keys`, per segment `[a, b)` relative to the span or `null` (a segment
    that owns no key); `variants.<name>` with `offset` (f16 elements into the
    bin), `prompt_tokens`, `last_chunk`, `prefill_ms`, `hq` (self-check
    stats); `l39_h10`, the pointing head's R1 winner per variant.
  - `.json`: the answer alphabet, the first render of each (scaffold, unit),
    the self-check failures (none).
  - `.features.npz`: `score.load`'s cache (per head per segment shares and
    argmax segments).
- `.scratch/locate/dev.json` (rule 1: every config's CV, per fold),
  `check.json` (C, per question), `C-labelled.json` (the labelled route on
  C, with timings and tokens).
- **C is spent**: its failures were read. Use A+B for anything that chooses.
  **D has never been run**: keep it for tracks L and H.

### Numbers to beat or explain (spec 18)

- Rule 1 CV on A+B (402 present questions): R1 S2 +baseline 70.4%, R1 S2
  66.2%, R3 S2 +baseline 59.5%, L39.h10 S2 +baseline 58.5%, R1 S1 57.2%, R2
  S2 +baseline 48.3%. Heads the folds chose under S2: L39.h12, L47.h20,
  L59.h16; under S1 every fold chose L35.h16.
- C, present, ≤ 256 segments: reading 118/161 (lexical 47/83, paraphrase
  71/78); labelled `choice` 147/161 (79/83, 68/78). By family: logs 32 vs 44
  of 47, records 31 vs 44 of 47, prose 55 vs 59 of 67.
- C, all present: 145/201, top-3 94.5%; by length tier 73.5% (≤ 2.0K keys),
  72.9% (≤ 4.6K), 67.5% (≤ 17.9K).
- Confidence AUC present/absent: 0.69 over C, 0.62 on the labelled route's
  188 questions, against its 0.82.
- L39.h10 as R1 on C's present questions, per family, hit / next line /
  elsewhere: under the index scaffold logs 1 / 56 / 10, records 38 / 7 / 22,
  prose 31 / 4 / 32; under the copy scaffold logs 35 / 5 / 27, records
  13 / 1 / 53, prose 53 / 1 / 13. What a head reads depends on the query
  position as much as on the head: a profile is a (head, query position)
  pair.
- Cost: labelled `choice` median 273 ms / 1,840 prompt tokens against a
  `noul` over the plain state 223 ms / 1,396; logs 4,519 vs 2,834 tokens.
  The baseline's own prefill: 55 tokens once the state's prefix is shared.
- Harness: about 11 minutes per 240-question set of four prefills (up to
  18K tokens each; one armed 17.6K-token prefill about 2 s).

### Running things

- **GPU first**: `make gpu-status` (the card fits one run; the loser dies
  without a diagnostic), and the shared lock
  `bash ../.inference-qwen-worktrees/.swarm/gpu-lock.sh try <name> "<what>"`,
  `... release <name>` after. Never a second GPU process, in any worktree.
- **The harness** (absolute paths — cargo runs a test in its crate's
  directory):

  ```text
  IGNIS_LOCATE_SET=<abs>/.scratch/locate/A IGNIS_LOCATE_OUT=<abs>/.scratch/locate/dumps \
  cargo test -p ignis-server --features cuda,attn-tap --test attention_head_locate_gpu \
    -- --ignored --test-threads=1 --nocapture
  ```

  `IGNIS_LOCATE_KV=bf16|hq` (default hq, consumed keys), `IGNIS_LOCATE_CHUNK`,
  `IGNIS_LOCATE_LIMIT=<n>` for a smoke run, `IGNIS_LOCATE_RESUME=1` to
  continue a cut dump (byte-identical to an uninterrupted run).
- **A live server** for anything through `/v1/decide` or chat: `make build`
  if the binary is stale, `make start` (a daemon; redirect its output to a
  file, never pipe it), `make stop` after. `make config` prints the command.
- **Scoring**: `python tools/locate-sets/score.py dev|check ...`; the tool
  tests `python tools/locate-sets/test_score.py` and `test_sets.py`.
- **The gate**: `cargo test` workspace-wide (CPU; about 1,813 tests). Never
  run a `--features cuda` build while it runs in the same checkout: both
  write `ignis-server.exe`.

### Facts about this model and engine that matter here

- 16 of 64 layers have attention (backbone layer `4o + 3` for GQA ordinal
  `o`); 24 query heads each, 4 KV heads (query head `h` reads KV head
  `h / 6`), head dimension 256. The other 48 layers are Gated DeltaNet: no
  keys, a recurrent state instead.
- The tokenizer (NFC, byte-level BPE, `trim_offsets: false`) isolates every
  digit (`\p{N}`) — a timestamp is a token per digit — and gives exact byte
  offsets. A JSON string state's newlines are the two-character escape `\n`,
  which belongs to no segment.
- The served KV format is hq-e8-2b: the keys attention reads are the current
  chunk, the first 32 (sinks) and the 512 before the chunk exact, the rest
  decoded by the codec (median relative error 0.37). A chunk of 8 tokens or
  fewer takes a route that materializes no keys (`ATTENTION_MIN_CHUNK_TOKENS`
  = 9).
- `/v1/decide` renders a JSON state as `{"evidence": …}` in the system
  message (layout L0 today; L1 puts nothing kind-specific before it, spec
  17). An array whose every element is an object with a string `type` is
  read as content parts, not evidence. Numbers are re-serialized (`1.50`
  becomes `1.5`).
- The artifact is `F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer`
  (sha256 `abb1e120…fffd9af`, re-downloaded from Hugging Face 2026-09-26 when
  it had disappeared).

### How the owner works (what spec 18 learned)

- Measure first, design after: on a research track try the bounded,
  reversible thing and bring back the number.
- Stop measuring once the decision is clear; state the real expected gain,
  not the effort.
- Pre-register a rule before a check set runs, and let the rule decide; a
  no-go ends the ship question, not the study — surface the anomalies.
- Durable results go in `docs/findings/` with a README row
  (`docs/agents/findings.md`); raw output stays in `.scratch/`.
- Commit with explicit paths (other sessions leave untracked files in the
  checkout); never reformat (the repo is not rustfmt-clean).

### Traps already paid for

- The consumed hq keys are captured for the first query position's chunk
  only.
- Every generated question is checked word by word (`common.py`); a bank
  entry that leaks a word fails the generator. HotpotQA's paraphrase is
  weaker (no *rare* word shared: only 12 of 7,405 questions share no content
  word), and its draw depends on the file's row order (the Hugging Face copy,
  sha256 `c20b638c…4e972f7c6`).
- An absolute selectivity bar (0.5, spec 14's style) keeps no head on text,
  where the best head's selectivity is about 0.4: make bars relative.
- A set used to choose anything is spent for judging it.
- Shell heredocs eat backslashes (`\n` in Rust strings broke once); Python
  in text mode on Windows writes CRLF — open with `newline=''`.
- The PyTorch vehicle's render once differed from the served one (a
  reasoning paragraph, an open think block, a plain-text instruction):
  compare its render byte for byte with the harness's `renders` before
  trusting a number from it. Its load from `Y:` takes about nine minutes;
  keep every measurement of a run in one process.
- MSVC's `getenv` does not see Rust's `set_var`: a kernel environment knob
  must be set before the test process starts.
- Three sets of span-only dumps are 11 GB; full-row dumps with values will be
  larger. Check free space on F: before a run.

## References

In the repo:
- Spec 18 and `docs/findings/2026-09-26-locate-attention-no-go.md`.
- Specs 11-15; `docs/findings/2026-09-21-one-attention-head-points.md`,
  `docs/findings/2026-09-22-the-heads-outline-the-object.md`,
  `docs/findings/2026-09-23-an-anchored-head-set-points-and-boxes.md`,
  `docs/findings/2026-09-22-how-the-head-map-is-read.md`,
  `docs/findings/2026-09-21-qwen-vl-grounding-primary-sources.md`.
- `docs/findings/2026-09-26-decision-classes-beyond-the-seven.md` (the
  prior art behind spec 18), and spec 17 (layout L1).

The literature pass (`docs/findings/2026-09-27-span-from-attention-literature.md`,
§ 10) checked the list below and corrects it: ICR's lexical bias is read from
the question's own tokens, not from a position after it, so point 4 of § Why
a second study contradicts nothing; ICR calibrates by subtraction, and no
attention-relevance paper uses the log-ratio lift; Kobayashi's norm includes
`W_O`; this model's output gate is a 256-vector per head, not a scalar; an
induction head's +1 is one token, not one line; TAG is "Tuning-free
Attention-driven Grounding".

Outside (verify each in the literature pass; these are starting points):
- ICR, attention-based in-context reranking: arXiv 2410.02642 (content-free
  "N/A" calibration, lexical bias).
- QRHead, query-focused retrieval heads: arXiv 2506.09944.
- AT2, attribution with learned per-head coefficients: arXiv 2504.13752.
- Retrieval heads (copy-paste heads in long context): arXiv 2404.15574.
- ContextCite, ablation-based attribution: arXiv 2409.00729.
- TAG, training-free grounding from attention: arXiv 2412.10840.
- Kobayashi, Kuribayashi, Yokoi, Inui, "Attention is Not Only a Weight:
  Analyzing Transformers with Vector Norms", EMNLP 2020.
- Olsson et al., "In-context Learning and Induction Heads" (Anthropic,
  2022): arXiv 2209.11895.
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks":
  arXiv 2309.17453.
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts":
  arXiv 2307.03172.
- Devlin et al., BERT (NAACL 2019): arXiv 1810.04805 — span decoding from
  start and end scores, the shape of initiator and terminator curves.
- Zhao et al., "Calibrate Before Use" (ICML 2021): arXiv 2102.09690 — the
  content-free input as a calibration, the lift's origin.
- Abnar and Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020):
  arXiv 2005.00928 — attention across layers rather than per head.
- SQuAD 2.0 (Rajpurkar, Jia, Liang, 2018), CC BY-SA 4.0; HotpotQA (Yang et
  al., 2018), CC BY-SA 4.0.
