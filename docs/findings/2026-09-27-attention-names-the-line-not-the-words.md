# Attention names the line that holds the answer, not the answer's words

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` / attention readout over a text state — spec 19 phases 1 and 2 (spans, profiles, the full row), development sets E1+E2
- Related: [GitHub #276](https://github.com/gpillon/ignis/issues/276); [spec 19](../specs/decide/19-a-span-read-from-attention.md); [phase 0: a vote of heads finds the line](2026-09-27-a-head-vote-finds-the-line.md); [the literature pass](2026-09-27-span-from-attention-literature.md); [`attention_span_locate_gpu.rs`](../../crates/server/tests/attention_span_locate_gpu.rs), [`tools/locate-sets/`](../../tools/locate-sets/README.md) (`spans.py`, `generation.py`, `spanscore.py`)
- Superseded by: none

## Question

Phase 0 found that a vote of heads names the *line* an instruction asks for.
Spec 19's phases 1 and 2 ask the finer questions: can one prefill's attention
name the exact **words** that answer — a charge id inside a log line, a
record's id, a SQuAD answer — against the route a caller has today
(generate the answer, find it in the state); can a profile's peaks answer
several spans or none; and where does the full row's mass go on a lexical
question (the literature pass's hypothesis H1: the match is made at the
instruction's own copy of the word)?

## Evidence

### What was measured

- **Sets** (`spans.py`): E1 (seed 20261020) and E2 (20261021), 300
  questions each, golds as character spans — `logvalue` (a value inside the
  target ERROR line, 1.1K / 4.4K / 17.7K-token logs), `recvalue` (a record's
  id or SKU, found by its city, title or name), `squad` (SQuAD 2.0 dev,
  about 1K and 4K tokens of an article; unanswerable questions are absent),
  `hotspan` (HotpotQA answers inside a gold sentence), `logmulti` (zero to
  three whole matching lines) and `hotfacts` (every supporting sentence).
  494 present questions with a gold; E3 (20261022) is written and unread.
- **Harness** (`attention_span_locate_gpu.rs`, commit `5e26b3a`): the L1
  render with the span kind text and the copy scaffold `{"quote":"`; per
  question, every head's scores over the **full row** at the scaffold's last
  token with the instruction and with `N/A`, the instruction's own tokens'
  mean weights, and — the first gold forced after the scaffold — the state's
  keys at the quote's first, middle and last tokens. Served NVFP4 artifact,
  consumed hq-e8-2b keys; the capture's self-check passed on every layer of
  every prefill. About 12.5 minutes a set.
- **Comparator** (`generation.py`): the same render through
  `/v1/chat/completions`, greedy, thinking off, the answer searched for in
  the state. Its prompt token counts equal the harness's render on every
  question checked.
- **Scoring** (`spanscore.py`) on the key grid: each gold character span
  mapped onto keys through the evidence writer and the dump's key byte
  ranges; readings chosen by 5-fold CV over E1+E2 (and nested); the
  generation route's first answer mapped onto the same keys. Raw output in
  `.scratch/locate/spans/` (`E12-report.json`).

### One prefill against generation, on the key grid

Token hit: the reading's key inside a gold span (for generation, its answer
overlapping one). Span F1 and exact match at key level.

| E1+E2, present (494) | one prefill, best reading | generation route |
|---|---|---|
| **token hit** | **278 = 56.3%** (nested CV 56.3%) | **85.2%** |
| span F1 / exact match | 22.0 / 4.5 | 74.0 / 65.2 |
| logvalue (100) | **98.0** | 89.0 |
| recvalue (100) | **5.0** | 98.0 |
| squad (100) | 57.0 | 91.0 |
| hotspan (100) | 61.0 | 61.0 |
| logmulti (44) | 22.7 | 100.0 |
| hotfacts (50) | 94.0 | 76.0 |
| cost | one prefill (+ the baseline's) | a prefill and 1-400 decode rounds, 0.2-2.5 s |

- The best reading is **the segment first, then its best key**: the 16
  heads with the most training token hits each name their argmax key, the
  segment holding most of those keys wins, and inside it the key with the
  largest summed z-scored score (the weight, not the lift). Every fold of the
  nested CV chose it (K = 8 or 16). The best single head reads 47.6%, the
  best key vote 51.0%, the best summed profile 54.5% — all on the weight;
  the lift is behind at every K.
- Growing a span from that key while the summed profile stays near its peak
  reaches F1 22.0 at best over the thresholds tried.
- The reading's agreement separates present from absent at an AUC of 0.67.
- **The segment is found.** The same heads' line-level vote — each head's
  segment with the largest share lift, spec 19's track L reading — names the
  segment holding the answer on 89.9% of these questions in CV (K = 16):
  logvalue 100, squad 98, hotfacts 94, logmulti 86, recvalue 83, hotspan
  78.
- **Where the key reading goes wrong.** On `recvalue` 86 of the 100 answer
  keys sit in the wrong record, on an `id` or `sku` value — a token of the
  answer's *type* in another record. On `logvalue` the value is the only
  token of its type near the words the question matches, and the key reading
  beats generation (98 against 89).

### Several answers or none (track P)

A vote of the K best line heads, every segment with at least a share `bar`
of the votes an answer, K and `bar` chosen in CV: set F1 53.1 on `logmulti`
and `hotfacts` (exact set 15%), and at most 35% of the absent `logmulti`
questions read as empty. The generation route's first quote was a gold line,
exactly, on every present `logmulti` question, and it quoted nothing on 15
of the 16 absent ones.

### The full row (Q4)

Softmax mass of the scaffold's full row, averaged over the 8 heads with the
most lift token hits, with the instruction (and with `N/A`):

| | evidence | instruction | kind text | template | scaffold |
|---|---|---|---|---|---|
| lexical (248) | 0.832 (0.730) | 0.071 (0.091) | 0.004 (0.063) | 0.078 | 0.015 |
| paraphrase (246) | 0.861 (0.733) | 0.049 (0.091) | 0.003 (0.061) | 0.070 | 0.017 |

The instruction moves the row's mass off the prompt's own words onto the
state, on both halves. A lexical question keeps 2.2 points more on the
instruction; its rare-word tokens hold 2.5% of these heads' mass (1.6% on
the paraphrase questions that have one). Over all 384 heads the instruction
takes 13.3% against 11.7%.

### While the quote is written (forced queries)

With the first gold forced after the scaffold, the heads whose argmax key is
most often on each part of the gold:

| query | gold's first key | anywhere in the gold | gold's last key | just after it |
|---|---|---|---|---|
| scaffold's last token | L59.h3 32% | L51.h23 47% | L47.h17 14% | L47.h20 28% |
| quote's first token | L43.h12 52% | L51.h20 70% | L51.h6 22% | L47.h20 40% |
| quote's last token | — | L7.h8 82% | **L7.h8 81%** | **L15.h21 73%** |

At the quote's last token — about to close it — early-layer heads mark the
copied span's last key (L7.h8, L43.h16, L11.h14) and the key after it
(L15.h21, L15.h23, L15.h19): the span's edges, exactly, once its words are
being written.

## Finding

Observed:

- At the copy scaffold, one prefill's attention names the **segment** that
  holds the answer on 89.9% of span questions, but the answer's own **key**
  on 56.3%, against 85.2% for generating the answer and finding it; the
  span's extent is far behind (F1 22 against 74).
- The key reading succeeds where the answer is the only token of its kind
  among the words the question matches (log values 98%, beating
  generation's 89%) and fails where it is not: asked for a record's id by its
  city, the heads point at an id — in another record (86 of 100).
- Several-or-none answers read from the vote's peaks reach a set F1 of 53;
  absent questions are rarely read as empty.
- With the instruction the scaffold's row moves mass from the prompt's words
  to the state; a lexical question keeps slightly more on the instruction
  (+2.2 points) and on its copy of the rare word (2.5%).
- While the answer is written, early-layer heads mark its first and last
  keys and the key after it.

Inferred, not measured:

- The heads at the scaffold look for the **next token to write** — a token
  of the answer's form — and the segment signal comes from where the
  question's match raises the mass (the lift of shares). Which instance of
  the form is right is decided in the residual stream, not in these heads'
  argmax. That is consistent with L39.h10 reading a timestamp's digit under
  the index scaffold (phase 0).
- The lexical paradox of spec 18 is not the instruction taking the mass: the
  shift is two points. Phase 0's account stands — one head's misses land on
  its content-free prefill's favourite lines.

## Implications

- **Track S (spans) stops on the development sets.** Its registered bar
  (token hit within 5 points of generation, span F1 within 10) is missed by
  29 and 52 points; by spec 19's own rule no E3 check is run.
- **Track P (profiles) stops too**: several-span F1 53 and near-blind
  absent detection, where generation is exact.
- **Track L is unaffected, and generalizes**: the line vote finds the
  segment holding an answer on 89.9% of these different questions (SQuAD
  98%), as it found the line on spec 18's sets.
- A caller who wants the words has two routes this study did not measure: the
  line from attention (one prefill) followed by generation over that line
  alone, or generation with the early-layer heads marking which occurrence
  of an ambiguous answer is the one being copied (the generation route's
  answer occurs more than once in the state on 43 of the 200 present
  `hotspan` and `squad` questions).
- **Values and gates (Q3) were not measured.** The key reading fails by
  picking the wrong instance of the right form; a value or gate weighting of
  the same keys would not know which instance the question means
  (inferred).

## Limits and unknowns

- E1+E2 chose the readings; the nested estimate (56.3%) is the honest one
  for the procedure. E3 was not read.
- `logvalue`, `recvalue` and `logmulti` are synthetic; SQuAD states are a
  paragraph with its article's neighbours, not whole articles; `hotspan`
  keeps only answers found verbatim in a gold sentence.
- The generation route's token hit counts an overlap with a gold span, the
  attention reading's a single key inside one: both are "a gold key named",
  not the same event.
- The forced queries use the gold's own text; the model's own quote often
  differs (its exact match with the gold is 33% on `squad` and 30% on
  `hotspan`, usually a longer answer).
- One artifact, the served render, consumed hq keys, fresh prefills.

## Follow-ups

- Spec 19: tracks S and P closed on the development sets; track L's check on
  set D is registered and waits for the owner's confirmation.
- If a span product is wanted: measure the line-then-generate route (a
  `choice`-free line from attention, then generation over that line) and
  its cost, against generation over the whole state.
