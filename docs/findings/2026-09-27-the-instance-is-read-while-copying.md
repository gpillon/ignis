# At the copy scaffold attention names the kind of line; the instance is read while copying

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: attention over a long text state at `/v1/decide`'s `locate` scaffold and along a teacher-forced quote — all 384 GQA heads, full-prompt rows, real cluster logs (set R) and synthetic logs (set L2); **exploratory**, on sets already read
- Related: [locate at length](2026-09-27-locate-at-length.md) (the misses this explains); [spec 20](../specs/decide/20-locate-at-length.md); [attention names the line, not the words](2026-09-27-attention-names-the-line-not-the-words.md); [`tools/locate-sets/research.py`](../../tools/locate-sets/research.py), [`research_report.py`](../../tools/locate-sets/research_report.py), [`instances.py`](../../tools/locate-sets/instances.py)
- Superseded by: none

## Question

On real logs the served vote finds the right kind of line and not the
instance ([locate at length](2026-09-27-locate-at-length.md): 28/50 against
generation's 46/50). Why? Three things were not known: what the vote's
heads look at when they miss, what the other 352 heads do at length, and
how much of the row leaves the state at length. The owner added a fourth:
what the heads do while the answer is being written.

## Evidence

- **Instruments** (branch `locate-long-context`, commit `176cea5`, never
  merged): a control file read at every `locate` names the heads read in
  rows (any 32, so all 384 in twelve requests that claim the state's
  prefix), full-prompt rows, and text forced after `{"quote":"`; each dump
  is reduced as it lands (segment shares, top keys, log-mass before / in /
  after the state, every tail score). Key *i* of the span is token *i* of
  the JSON-escaped state — checked on every question — and the prompt's
  tail is rebuilt token for token (65 of 65), so the tail splits into the
  closing template, the kind text, the instruction, the turn and the
  scaffold.
- **Sets**: R's 50 present questions (real logs, 4K-200K tokens, 10 per
  tier) and 15 present questions of L2 (synthetic, 1,000 / 6,000 / 12,000
  lines, 5 each). Both were read before: every number here is exploratory.

### What the vote's heads look at when they miss (R, the served 32)

The target against its **twin** — the line the vote chose when it missed,
else the line most like the target — with each line's tokens split into
its own and the shared ones:

| mean over heads | tokens t_own / t_shared / w_own / w_shared | mass per token, t_own / t_shared / w_own / w_shared | peak on t_own / w_shared |
|---|---|---|---|
| hits (28) | 17 / 105 / 11 / 105 | 0.0032 / 0.0014 / 0.0006 / 0.0006 | 11% / 8% |
| misses (22) | 10 / 77 / 20 / 80 | 0.0018 / 0.0006 / 0.0009 / 0.0011 | 3% / 21% |

- In misses the target's own tokens are still the densest per token, but
  the twin takes more mass in total and most peaks.
- **Near-duplicates decide it**: the miss rate by the number of lines at
  token Jaccard ≥ 0.5 with the target is 29% (none, 7 questions), 26% (1-5,
  19), 53% (6-50, 17), 86% (> 50, 7). Misses have a median 11 such lines,
  hits 1; the chosen line has a median Jaccard 0.60 with the target, and
  precedes it in 15 of 22 misses.
- Every line's first tokens (the pod label on R, the timestamp on L2) take
  about three times their share of the positive lift (R 23-26% on 7-8% of
  the tokens, L2 47-51% on 28%) — **the same in hits and misses**, so not
  the cause.

### All 384 heads at length (one-head reading, the lift, k = 0)

| heads over 50% top-1 | 4K / 16K / 50K / 100K / 200K tokens (R) | 1K / 6K / 12K lines (L2) |
|---|---|---|
| at the scaffold (k = 0) | 51 / 24 / 8 / 2 / 1 | 15 / 10 / 13 |
| 8 target tokens copied (k = 8) | 13 / 27 / 2 / 1 / 0 | 86 / 67 / 77 |

On real logs no head keeps the instance past 50K; the best at 200K are end
markers (L39.h23, L47.h17, L47.h1 at 0.5-0.6). On synthetic logs, eight
copied tokens — a line's timestamp, level and service — are enough for most
heads.

### Where the row goes (all 384 heads, full row, k = 0)

- About half of the softmax mass is on the state at every length: R 0.50 /
  0.52 / 0.52 / 0.53 / 0.52 from 4K to 200K, L2 0.49 / 0.51 / 0.52; the kind
  text takes ~0.15, the instruction ~0.10, the scaffold ~0.11, the turn
  ~0.07, the prompt before the state 0.02-0.05.
- By layer: L3-L31 put 12-44% on the state and most of the rest on the kind
  text (20-36%) and the instruction (12-24%); L35-L59 put 59-91% on the
  state; L63 26-28% on the scaffold. With length the late layers move
  slightly *towards* the state (L55 0.83 → 0.89, L59 0.79 → 0.86) and L35 /
  L39 towards the instruction (0.10 → 0.17, 0.07 → 0.12).
- So the state's share of attention is fixed, and each line's share falls
  with the number of lines; nothing leaks to the question.

### While the answer is written (R, the served 32, teacher-forced)

The target's first *k* tokens, as the model would write them inside the
JSON string, forced after `{"quote":"`; each head's segment of largest share
in the **question's** prefill alone:

| plurality of the 32 on the target | k = 1 | 2 | 4 | 12 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| R, the vote's hits (28) | 0.64 | 0.71 | 0.82 | 0.68 | 0.79 | 0.86 | 0.86 | 0.78 |
| R, the vote's misses (22) | 0.09 | 0.23 | 0.27 | 0.32 | **0.55** | 0.50 | 0.57 | 0.61 |

- The misses move to the instance as it is written, around 24 tokens — the
  median number of tokens after which the target's prefix matches no other
  line of its window is 26 (p90 46).
- **Method**: the content-free twin must not carry the forced text. Given
  the same prefix it copies the same line, and the lift cancels the very
  pointer being measured (the served vote along the quote reads 0.46-0.66
  for this reason, and is not reported as a result).

## Finding

At the copy scaffold the heads that locate a line name its **kind** — the
lines the answer could begin with — and among near-duplicates the instance
is a coin toss that worsens with their number (29% misses with none, 86%
with more than fifty) and with length (on real logs 51 heads read the
instance at 4K, one at 200K). Attention does not leak to the question at
length: half of the row stays on the state at every length, late layers a
little more, so each line's share falls as the state grows. The instance is
read **while the answer is copied**: as the target's prefix is written, the
heads move onto it, the vote's misses around the point where the prefix
stops matching any other line; on synthetic logs, whose lines differ in
their first tokens, eight tokens suffice.

## Implications

- One-pass reading at the scaffold cannot be tuned into instance accuracy
  on real logs by choosing other heads: no head holds it past 50K.
- The copy is where the instance is resolved: a `locate` that copies the
  line under a constraint to the state's lines and stops at the first
  unique prefix is the mechanism this measures, not a workaround.
- Research: these are hypotheses on read sets. The confirmatory versions —
  miss rate against near-duplicate count, heads over 50% against length,
  the state's share against length, the plurality's crossing point against
  the unique-prefix length — need a fresh real-log set registered first.

## Limits and unknowns

- Exploratory throughout: R and L2 were read before these analyses; R is 50
  present questions from one cluster, written by one author after reading
  the windows; L2's subset is 15 questions.
- Teacher forcing uses the gold line tokenized alone, which may split
  tokens differently from free generation at the prefix's end.
- The trajectory reads the 32 served heads only; the heads that copy may be
  others (k = 8 over all heads is the one cross-section).
- One artifact (the served NVFP4 27B), hq-e8-2b keys.

## Follow-ups

- A fresh real-log set with the four hypotheses above registered.
- The trajectory over all 384 heads at a few *k*, to name the copy heads.
