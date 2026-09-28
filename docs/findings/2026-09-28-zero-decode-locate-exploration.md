# Zero-decode locate on real logs: read the lines' closing keys, then re-read a short list

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` over long text states with no token generated — other readings of the same attention rows, other heads, other read positions, the labelled `choice` over a folded or shortened text, text rewrites; real logs (cluster set R, public+cluster set R2) at 4K-200K tokens and the short sets A-D; **exploratory** (R2 was already read by spec 21, and several choices below were made looking at it)
- Related: [locating a line in real logs](2026-09-28-locating-a-line-in-real-logs.md) (the failure this starts from: the served vote 34.5% on R2); [the literature](2026-09-28-zero-decode-locate-literature.md) (the candidate list this tests); [the instance is read while copying](2026-09-27-the-instance-is-read-while-copying.md); [locate at length](2026-09-27-locate-at-length.md) (the end-marking heads); tools `tools/locate-sets/zd_*.py`, `folded_locate.py --route choice`
- Superseded by: none

## Question

The served `locate` reads 32 heads' attention at the copy scaffold, less a
content-free twin, and takes a plurality vote. On real logs past 4K tokens
it reads 34.5% (R2) and 56% (R), losing the instance among near-duplicates;
generating the line reads 86-92%, and folding the log then generating reads
94.8%. The owner asked for other **zero-decode** algorithms — no token
written — using the heads or anything else the engine has: a better reading
of the same rows, other heads, other read positions, other signals (logits,
labels), text rewrites, and the end-marking heads as a line of research of
their own.

## Evidence

- **Instruments** (branch `locate-long-context`, never merged): the research
  hooks of `176cea5` plus two controls — `cut_tail` (read the rows at an
  earlier prompt position, e.g. an instruction token) and `answers` (read
  logits at the scaffold instead of rows). One server, ceiling lifted.
- **Offline cache** (`zd_cache.py`): for every question already dumped, each
  head's softmax share per segment (all 384 heads: A-D, R, R2, 15 of L2) and,
  from key-level rows, per-segment features at the line's keys (`sum`), its
  last key, its closing separator, the next line's first key, its first key
  and its max (the 32 served heads on R/R2/L2; all 384 on A-D).
- **Readings compared** (`zd_offline.py`): the served vote re-read; the same
  heads' lifts standardized and summed; a conditional logit over heads (and
  over heads x key positions) fitted on other sets; the **end reading** — a
  line scored by its closing keys (last key + separator + next line's first
  key), lift standardized per head and summed; BM25 over the instruction's
  words and its fusion with attention; contextual calibration (other
  questions on the same state as the baseline).
- **End heads** (`zd_heads.py`): 32 heads chosen **on A+B only** (short sets)
  by their single-head top-1 at a line's last key, next first key or
  separator (`L47.h20, L47.h4, L51.h4, L47.h17, L47.h1, L51.h12, ...`, mostly
  L39-L51), read at length on R and R2.
- **Two-pass zero decode**: (a) **shortlist + re-read** (`zd_shortlist.py`):
  pass 1 ranks the whole log's lines by a one-pass reading, pass 2 asks a
  labelled `choice` over the first K lines only (in order), optionally with
  the question also written before them; (b) **folded + labelled choice**
  (`folded_locate.py --route choice --values`): the spec-21 fold, a labelled
  `choice` over the templates, then over the chosen template's rows; (c)
  **chunked labelled choice** (`zd_chunked.py`): the whole log in 256-line
  labelled chunks, then one `choice` over the chunks' winners.
- **Mechanism**: where heads look on short sets (all 384, key level), and
  where the served heads' top key sits along R2's teacher-forced copy.

### One pass over the whole log (top-1; recall@16)

| reading | heads | trained on | R2 (58) | R (50) |
|---|---|---|---|---|
| served vote (sum of a line's keys) | served 32 | — | 34.5 (91.4) | 56 (96) |
| standardized sum | served 32 | — | 36.2 | 60 |
| conditional logit, sum | all 384 | A+B+other | 43.1 | 64 |
| separator only | served 32 | — | 50.0 | 62 |
| **end reading** (last + separator + next first) | served 32 | — | **60.3** (94.8) | **68** (94) |
| logit over heads x {sum, last, next, sep} | served 32 | A+B+other | 62.1 (93.1) | 74 (100) |
| **end reading** | **end heads 32** (chosen on A+B) | — | **72.4** (94.8) | **74** (100) |
| the end heads read as a sum (vote) | end heads 32 | — | 27.6 | 52 |
| logit over 56 heads x 4 positions | served + end | A+B+other | 60.3 | 78 |
| BM25 alone | — | — | 50.0 (60.3) | 44 |
| logit end reading + 0.5 BM25 | served 32 | A+B+other; α from R's sweep | 70.7 (96.6) | 80 (α in-sample) |

On the short sets (C and D, test; ~60-1,000 segments) the order is the other
way round: the served vote 91.5 and 92.0, the 384-head logit 94.5 and 94.5,
the sum and the end together 92.5 and 95.0 — but the end reading **alone**
80.6 and 86.1, the separator alone 52-53. The sum over a short line works; at
length its body's share is diluted among near-duplicates and the end holds.
On synthetic logs at length (L2, 250-12,000 lines) the end reading reads
93-97 against the vote's 91. Equal-weighting all 384 heads reads 43-54%.
Contextual calibration reads within 2 points of the N/A twin everywhere (R2 36.4 vs 34.5 on the sum,
60.0 vs 58.2 on the end).

### One reading for every length (top-1)

| reading | C (short) | D (short) | R2 (long) | R (long) |
|---|---|---|---|---|
| served vote | 91.5 | 92.0 | 34.5 | 56 |
| served heads, end reading | 80.6 | 86.1 | 60.3 | 68 |
| served heads, z(sum) + z(end) | **93.5** | **95.0** | 69.0 | 68 |
| **end heads, end reading** | 92.0 | 91.5 | **72.4** | **74** |
| end heads, z(sum) + z(end) | 90.0 | 87.6 | 65.5 | 74 |

The end heads were chosen on A and B; C and D are their first short test.
Nothing here was fitted on a long set.

### Two passes, no token written (top-1)

| method | R2 (58) | R (50) | cost |
|---|---|---|---|
| served vote (one pass) | 20 | 28 | whole-log prefill |
| generation on the whole log (reference, decodes) | 50 | 46 | whole-log prefill + decode |
| folded + generation (spec 21, decodes) | 55 | 45 | ~1.3 s |
| **folded + labelled choice** | **52** | **42** | ~0.1-3 s, no long prefill |
| folded + choice, top-3 templates, tournament | 52 | 44 | +2 short passes |
| chunked labelled choice (256-line chunks) | 48 | — | whole log in chunks + 1 pass |
| **shortlist 16 (served end reading) + choice** | **54** | 45 | whole-log prefill + ~1 s |
| shortlist 16 (served end reading) + choice, question first | 55 | 45 | same |
| shortlist 16 (served logit end) + choice, question first | 53 | 48 | same |
| shortlist 8 (end heads + BM25) + choice | 56 | 47 | same |
| **shortlist 16 (end heads) + choice** | **55** | **49** | whole-log prefill + ~0.3-2 s |
| **shortlist 16 (end heads + BM25) + choice** | **57** | **49** | same |
| shortlist 32 (end heads) + choice | 55 | 47 | same |
| shortlist 64 + choice | 48-51 | — | same |

By near-duplicates and by window (R2):

| method | siblings 0 | 1-5 | 6+ | 16K | 50K | 100K | 200K |
|---|---|---|---|---|---|---|---|
| served vote | 9/23 | 8/19 | 3/16 | 5/7 | 2/9 | 12/36 | 1/6 |
| end heads, end reading | 21/23 | 15/19 | 6/16 | 5/7 | 6/9 | 27/36 | 4/6 |
| shortlist 16 + choice, question first | 23/23 | 19/19 | 13/16 | 7/7 | 9/9 | 33/36 | 6/6 |
| folded + choice | 20/23 | 18/19 | 14/16 | 7/7 | 9/9 | 32/36 | 4/6 |

### Where the heads look

- Short sets A-D, every head, single-head top-1 by read position: a line's
  **first** key is the worst place to read (best head 0.25, served mean 0.13;
  no head reads better there than at the sum); **61 heads** read better at the
  line's **last key** than at its sum, **44** at the **next line's first key**,
  19 at the separator itself (none above 0.50).
- Along R2's teacher-forced copy (top key inside the target): the 25 served
  heads that are not among its seven end-marking heads (spec 20) sit on the
  **copy point** — the next token
  to copy — 24% at 1-4 tokens written, 50% at 24-64 (an induction pointer);
  the seven end-marking heads sit on the target's **last keys** 39% at 1-4 tokens
  written, the copy point 5%: they point at where the copy will end before it
  has started.

### Negative or flat

- Text rewrites at length (R2, one pass): a unique id before each line —
  served vote 18 (20 without), served end reading 38 (35), end heads 39
  (42) — flat; an explicit ` <eol>` after each line — the served vote 29
  (20 without: a marker inside the line's keys helps the sum), but the end
  reading falls (served 24 against 35, end heads 30 against 42): the closing
  keys become the same tokens on every line. The raw text read at its own
  ends stays the best one pass.
- Reading at the instruction's tokens (ICR-style, `cut_tail`; the served
  heads; the 16 R2 questions on windows up to 50K, ~20 positions each, 9-24 s
  a question on the retained prefix): pooling every token reads 9/16 against
  the scaffold's 8/16 (sum; the end reading 7/16 both) — not ICR's 10-17
  points. But one token usually reads the line: the best token per question,
  chosen after the fact, reads 14-15/16; the token whose reading has the
  largest margin, chosen without the answer, reads 11/16 (sum). At the
  question's own tokens the heads land in the line's body (the sum reads,
  the end does not). A hint on 16 questions, for "kind x value" follow-ups.
- A fitted logit over many heads does not travel from short to long sets
  (56 heads: 60.3 on R2 against 72.4 for the untrained end heads).
- Unioning the top-3 templates at level 2 (R2 46) and 64-line shortlists
  (48-51): the labelled `choice` gets worse with more near-duplicate
  candidates.

## Finding

At length the line a question asks for is marked at its **end**, not in its
body: reading a line's closing keys — its last token, the separator, the
next line's first token — with heads chosen for that on short texts finds it
72-74% of the time in one pass over real logs, against 34.5-56% for the
served vote's sum over the line, without fitting anything on long logs —
and on short texts it reads as the vote does (92.0 and 91.5 against 91.5
and 92.0), so it could be one reading for every length. A line's first key is where no head reads it, as a causal model predicts (its
key has seen none of the line), and the end-marking heads point at the end of
the line from the first tokens the model copies. One pass still loses the
instance among many near-duplicates (6/16 with six or more). A second, short
pass resolves most of them without writing a token: the labelled `choice`
over the first 16 lines of the end heads' ranking reads 55-57/58 on R2 and
49/50 on R (93-98%, above generating the line from the whole log, 50 and
46), and over the folded log's templates and rows 42-52 (84-90%) with no
long prefill at all. The `choice` is a short-list reader: over 64
near-duplicates, or 256 lines of raw log, it falls to 80-88%.

## Implications

- **The vote's reading, not its heads' attention, was the weak link at
  length.** The same served rows read at the lines' ends gain 26 points on
  R2 (12 on R); end heads gain 38 (18). A `locate` past ~4.5K keys needs
  the end reading and the end heads, calibrated as the vote was.
- **A zero-decode `locate` for long logs is two passes**: one reading of the
  whole log (the retained prefix pays it once per state), then a labelled
  `choice` over 16 candidates. It reads at least as well as folding and
  copying (spec 22) on these sets, with no decode; folding + `choice` is the
  cheap variant (no long prefill) at a few points less.
- **For paper 1**: end reading vs sum reading vs length is the mechanism
  story (end heads, the copy pointer, no start heads); for paper 2, shortlist
  + short re-read and fold + labelled choice are the zero-decode methods to
  set against copy and generation.
- The owner's hypothesis — end heads mark the end of what will be copied —
  has its first direct evidence (39% on the target's last keys before the
  copy starts, 5% on the copy point).

## Limits and unknowns

- **Exploratory.** R2 was read by spec 21 and again here; K (8 / 16 / 32),
  the BM25 weight, the end-reading form and the end-head rule were chosen
  with R and R2 in view, and the best row of a table of many is an optimistic
  estimate. None of it is confirmed: that needs registered hypotheses on a
  set nobody has read (spec 22's R3 is the natural one).
- One artifact; 58 + 50 real-log questions; the end heads were chosen on
  synthetic short sets (A+B) only, which is their strength, but one choice.
- Latencies were measured with several experiments sharing the GPU; they
  are upper bounds, not the method's cost.
- Not tried: value-norm-weighted attention and hidden-state probes (both
  need new engine taps); first-token logits (subsumed by the labelled
  `choice` over short lists; the `answers` control exists).

## Follow-ups

- A confirmatory spec on a fresh set: (1) end heads' end reading beats the
  served vote at length; (2) shortlist-K + labelled choice is non-inferior to
  folded copy; (3) folded + labelled choice as a no-long-prefill option; K
  and the heads frozen before the set is built.
- A served change to register and confirm: the end heads' end reading, or
  the served heads' z(sum) + z(end), as the one-pass reading at every length
  (table "One reading for every length").
- `found` for the two-pass routes: the `choice`'s confidence on the short
  list against absent questions.
