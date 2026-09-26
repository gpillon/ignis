# Attention finds the paraphrased line and loses the quoted one: `locate` is a no-go

- Kind: experiment
- Status: current
- Observed: 2026-09-26
- Last verified: 2026-09-26
- Scope: `/v1/decide` / attention readout over a text state (spec 18 phase A), labelled `choice` route
- Related: [GitHub #274](https://github.com/gpillon/ignis/issues/274) (phase A), [#275](https://github.com/gpillon/ignis/issues/275) (phase B); [spec 18](../specs/decide/18-locate-by-attention.md); [decision classes beyond the seven](2026-09-26-decision-classes-beyond-the-seven.md); [ADR 0038](../adr/0038-the-seam-carries-one-attention-head.md); [`tools/locate-sets/`](../../tools/locate-sets/README.md); [`attention_head_locate_gpu.rs`](../../crates/server/tests/attention_head_locate_gpu.rs)
- Superseded by: none

## Question

Spec 18 asks whether the served 27B's attention, read at one position of one
prefill, can answer "which line / which item / which sentence" of a **text**
state well enough to ship `locate`: a primitive that needs no labels in the
state, so it shares the state's prefix with other questions and has no
256-option ceiling. Its phase A was decided by rules written before any run:

1. the reading, heads, scaffold and baseline are chosen on sets A+B by
   cross-validation (R3 only if it wins by at least 3 points);
2. **go** on set C if the chosen reading's top-1 is within 5 points of the
   labelled `choice` route's overall, and within 10 on the paraphrase half,
   on the questions both can answer (at most 256 segments);
3. and 4. the length ceiling and the per-family floors, for phase B.

## Evidence

### What was measured

- **Sets** (`tools/locate-sets/`): A (seed 20261010) and B (20261011) for
  development, C (20261012) to judge; D (20261013) is phase B's and was not
  run. 240 questions each: 80 synthetic logs (60 / 250 / 1,000 lines, about
  1.1K / 4.4K / 17.7K state tokens), 80 JSON record arrays (20 / 80 / 300
  records, 0.9K / 3.4K / 12.9K) and 80 HotpotQA distractor-dev questions
  (about 1.3K). Half lexical (the question shares a word only the target
  has), half paraphrase (no content word shared; for HotpotQA, no *rare* one,
  since only 12 of its 7,405 questions share none at all), and 13 of 80 per
  family with the target removed. 201 present questions per set.
- **Harness** (`attention_head_locate_gpu.rs`, commit 6fa64a2, made before
  the full runs): the L1 render through `/v1/decide`'s own template provider,
  four prefills per question (the index scaffold `{"line":`/`{"item":`, the
  copy scaffold `{"quote":"`, and each with the instruction replaced by `N/A`),
  all 16 GQA layers × 24 heads scored at the scaffold's last position over the
  state's key span, on the served NVFP4 artifact with hq-e8-2b and the
  **consumed** keys. The capture's self-check passed on every layer of all
  2,880 prefills: exact rows at a median relative L2 of 0.0017, decoded rows
  at 0.37, the codec's own band. About 11 minutes per set.
- **Labelled route** (`labelled.py`): the same C questions through a live
  `/v1/decide` (`make start`, the served configuration), each content segment
  prefixed with the endpoint's own answer label and one `choice` over those
  labels — Jev's line search. 161 of C's 201 present questions have at most
  256 segments; the 1,000-line logs and 300-record arrays do not.

### Rule 1 on A+B (5-fold CV, 402 present questions)

| Reading | Scaffold | Baseline | top-1 |
|---|---|---|---|
| **R1** (one head) | **S2** (copy) | **yes** | **283/402 = 70.4%** |
| R1 | S2 | no | 66.2% |
| R3 (top-K heads' mass) | S2 | yes | 59.5% |
| L39.h10 (the pointing head, as R1) | S2 | yes | 58.5% |
| R1 | S1 (index) | no | 57.2% |
| R2 (head-set vote) | S2 | yes | 48.3% |
| R2 | S1 | no | 30.1% |

Rule 1 chose **R1, S2, with the content-free baseline**; R3's best was 10.9
points below it. Fitted on all of A+B the head is **L39.h12**; the folds had
picked L39.h12 twice, L47.h20 twice and L59.h16 once — several heads read
text about equally well. R2's set rule keeps every head whose selectivity
(mass on the target over mass on all segments) is at least half the best
head's; the first version, an absolute 0.5 like spec 14's, kept no head at
all on the smoke run's six questions (best 0.40) and was changed before the
full runs.

### Rule 2 on C: no-go

| C, present, ≤ 256 segments | attention (R1 L39.h12 S2 +base) | labelled `choice` | gap |
|---|---|---|---|
| **overall** | 118/161 = **73.3%** | 147/161 = **91.3%** | **+18.0** (bar 5) |
| **paraphrase** | 71/78 = **91.0%** | 68/78 = **87.2%** | −3.8 (bar 10) |
| lexical | 47/83 = 56.6% | 79/83 = 95.2% | +38.6 |
| logs | 32/47 | 44/47 | +25.5 |
| records | 31/47 | 44/47 | +27.7 |
| prose | 55/67 | 59/67 | +6.0 |

The paraphrase half passes — the attention reading is *better* than the
labels there — and the overall bar fails by 13 points, all of it on the
lexical half. **By rule 2 the answer is no-go.**

Over all of C's present questions (lengths included) the reading is 145/201
(72.1%): lexical 59/102 (57.8%), paraphrase 86/99 (86.9%). Its lexical misses
are not near misses: 31 of the 34 on logs and records land more than two
segments from the target, often on an early line of the state, while the
target is in the reading's top 3 on 94.5% of all present questions.

### The pointing head on text

L39.h10, which points in images, is 70/201 (S1) and 101/201 (S2) on C as a
single head. Under the index scaffold on logs it lands on the line **after**
the target 56 times of 67 (46 of 67 on A): the query about to name a line
number reads where the next line begins, the text analogue of the head
reading where a button's label begins. Segment-level credit does not absorb
it, and a shifted reading was not among the pre-registered ones.

### Reported, not binding on a no-go

- **Rule 3**: top-1 by length tier 73.5% (spans up to 2.0K keys), 72.9%
  (4.6K), 67.5% (17.9K); `LOCATE_MAX_KEYS` would have been 4,570.
- **Rule 4**: the floors would have been logs ≥ 31/47, records ≥ 30/47,
  prose ≥ 53/67.
- **Confidence**: the reading's winner share separates present from absent
  questions at an AUC of 0.69 over all of C (medians: hit 0.54, miss 0.30,
  absent 0.32). On the 188 questions the labelled route was asked (161
  present, 27 absent) it is 0.62, against the labelled `choice`'s 0.82.
- **Cost**: no `locate` was timed — there is none to time until phase B. What
  stands in for it is a `noul` over the unlabelled state: one prefill of the
  same state and one readout, the cost class a `locate` would be in. Over the
  188 questions the labelled `choice` took a median 273 ms and 1,840 prompt
  tokens, the `noul` 223 ms and 1,396 (+32% tokens for the labels); on logs
  4,519 against 2,834 tokens (+59%) and 296 against 220 ms. The chosen
  reading would add a second prefill, the content-free baseline: a median 55
  tokens once the state's prefix is shared, the whole state when it is not.

## Finding

Read at the copy scaffold, one calibrated attention head finds a
**paraphrased** line, record or sentence in one prefill better than the
labelled `choice` does (91.0% against 87.2%), with no labels in the state —
51/55 and 20/23 of C's paraphrase questions up to 4.6K keys, 15/21 at 17.9K,
a length the labels cannot reach. It fails a **lexical** question — one that repeats a
rare word of its target — 43% of the time, and not by a line: the reading
goes somewhere unrelated. Overall that puts it 18 points behind the labelled
route, and spec 18's pre-registered rule says no-go.

## Implications

- Spec 18 is **NOT IMPLEMENTED**, and phase B (#275) does not go ahead.
- **The way to locate today is the labelled `choice`** (documented in
  `docs/user/README.md`, "Finding a line or an item"): 91.3% top-1 on C, up
  to 256 segments, at 59% more prompt tokens than the plain state on logs
  (32% over the three families), and no prefix shared with other kinds of
  question over that state.
- The failure is the opposite of the lexical bias ICR reports for its
  attention ranking: here the paraphrase half is the easy one. A follow-up on
  `locate` should start from the lexical half, not from the heads.
- The instruments are reusable as they are: the harness and sets for any
  text-attention question, `score.py` for any reading over its dumps, and set
  D is still unspent.

## Limits and unknowns

- **Why lexical questions fail is not measured.** The likeliest reading is
  that the instruction itself carries the rare word, so the copy heads attend
  to *that* occurrence — outside the span — and what is left inside the span
  is noise. The dump holds only the span's scores, so the mass on the
  instruction's tokens was never read; the hypothesis is untested.
- One kind text per scaffold, written once and not iterated.
- The prose paraphrase split is weaker than the other two families' (no rare
  word shared, rather than no content word).
- Each prompt is a fresh prefill. A fan-out that claims the state from an
  earlier request reads the same keys from the cache, where all but the
  residual window is decoded — as here for every row older than the window.
- The comparison covers 161 questions (one question is 0.6 points); the gap
  it measured is 30 times that.

## Follow-ups

- If `locate` is revisited: dump the scores over the instruction's tokens as
  well (the harness's span is one parameter), to confirm or kill the
  mechanism above before designing anything; the candidate fixes (an
  instruction that paraphrases its own rare words, a layout that masks the
  instruction's copy) each need fresh sets and a new pre-registration.
- A `found` flag stays out of scope: at an AUC of 0.62-0.69 the confidence would
  not mean much.
