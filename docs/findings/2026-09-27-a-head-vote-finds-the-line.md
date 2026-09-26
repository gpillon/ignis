# A vote of heads finds the line one head lost

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` / attention readout over a text state — spec 19 phase 0 on spec 18's development dumps (sets A+B), replicated on set C
- Related: [GitHub #276](https://github.com/gpillon/ignis/issues/276); [spec 19](../specs/decide/19-a-span-read-from-attention.md); [spec 18's no-go](2026-09-26-locate-attention-no-go.md); [the literature pass](2026-09-27-span-from-attention-literature.md); [`tools/locate-sets/profiles.py`](../../tools/locate-sets/profiles.py), [`score.py`](../../tools/locate-sets/score.py)
- Superseded by: none

## Question

Spec 18 read the line a text instruction asks for from **one** attention head
and missed 43% of the lexical questions, far from the target
([no-go](2026-09-26-locate-attention-no-go.md)). Spec 19's phase 0 asks what
its dumps already say, with no GPU: which per-token signals (*ordinates*)
localise the target, where each head's peak sits against it (a start, an end,
the line after), whether a combination of heads beats the best one, how high
a shortlist's ceiling is, and where the lexical misses go.

## Evidence

### What was measured

- **Input**: spec 18's dumps of sets A and B (402 present questions, 78
  absent; logs of 60 / 250 / 1,000 lines, JSON record arrays of 20 / 80 /
  300, HotpotQA prose): every GQA head's pre-softmax scores at the scaffold's
  last token over the state's keys, for the index scaffold (`s1`), the copy
  scaffold (`s2`) and each with the instruction replaced by `N/A` (`-na`),
  served NVFP4 artifact, consumed hq-e8-2b keys.
- **Tools**: `tools/locate-sets/profiles.py` (`extract`, then `report`),
  tested on a synthetic dump with planted heads (`test_profiles.py`); the
  `vote` reading in `score.py`. Raw output in `.scratch/locate/phase0/`
  (`quick.json`, `votes.json`, `comb.json`, `vote-choice.json`,
  `C-vote-replication.json`).
- **Ordinates**, per head: the weight (`s1`, `s2`), the **lift**
  `log a_q − log a_N/A` (`lift1`, `lift2`), and the content-free weight alone
  (`na1`, `na2`) as the control — the question-independent prior.
- **Selection** is always by cross-validation on A+B, with `score.py`'s
  folds. C was read only after the head vote was registered in spec 19
  (commit `1d1d999`).
- The rare-word keys (the tokens of the word a lexical question shares with
  its target only) were found by re-tokenizing the evidence and checking the
  result against the dump's own span and segment keys: exact on all logs and
  records, and on the prose questions without non-ASCII text (200 lexical
  questions matched).

### Token level: the best single head per ordinate

| Ordinate | Best head (by CV token hit) | token hit, CV | AUROC | AP |
|---|---|---|---|---|
| `s1` | L35.h16 | 156/402 = 38.8% | 0.84 | 0.22 |
| `s2` | L47.h20 | 235/402 = 58.5% | 0.88 | 0.31 |
| `lift1` | L39.h12 | 186/402 = 46.3% | 0.85 | 0.28 |
| `lift2` | L39.h12 | 246/402 = 61.2% | 0.85 | 0.29 |
| `na1` (prior) | L35.h16 | 53/402 = 13.2% | 0.69 | 0.07 |
| `na2` (prior) | L39.h15 | 58/402 = 14.4% | 0.70 | 0.08 |

Token hit is the head's argmax key falling inside the gold segment; AUROC
and AP are per question over every key of the span, "inside a gold segment"
as the label. A head's peak is sharp (AUROC 0.85-0.88) but covers little of
the line (AP ≤ 0.31). The index scaffold reads records well and logs badly:
`lift1` hits 80.6% of records and 28.4% of logs; `lift2` 66.4% and 50.7%.

### Where the peaks sit

On logs and records (268 single-target questions), where the argmax key of
each head falls against the target:

- **No head marks the target's first token.** The best is L51.h20 at 6.0%
  (`s1`); under the lift no head passes 2.6%. In the span-aligned averages
  (the mean profile around the target's first key) a few heads *rise* at the
  first key (L51.h20, L43.h17), but not to their maximum.
- **Many heads mark the boundary after it.** Under `s1`, L35.h18 (43%),
  L43.h14 (41%), L31.h3 (37%) peak on the next segment's first key; on logs,
  L43.h14 and L35.h18 land on the next line's first token or the `\n` escape
  before it (78 and 59 of 131). Under `s2` L51.h4 does so on 66% of logs.
  These peaks are structure, not the question: under the lift they fall to
  12-15%.
- **L39.h10 under the index scaffold reads a number, not a boundary.** On
  logs it lands on the **last digit of the next line's timestamp** (offset
  +4 from that line's first key) on 62 of 131 questions, and on that line's
  first token or the `\n` before it on 16. The model is about to write a
  line number, and the pointing head reads the first number after the
  target.
- **A record's last token** holds the peak of L35.h19 and L35.h4 under
  `s1`, and of L39.h15 and L47.h3 under `s2`, on 23-25% of records.

### Line level: offsets, votes and learned weights

5-fold CV over A+B's 402 present questions; spec 18's rule-1 choice (R1,
one head, `s2` with the baseline) is the reference.

| Reading | top-1 | lexical | paraphrase | logs | records | prose |
|---|---|---|---|---|---|---|
| R1 (spec 18's choice) | 283 = 70.4% | 63.5 | 77.2 | 64.2 | 70.9 | 76.1 |
| R1 with a per-head offset | 283 = 70.4% | — | — | — | — | — |
| R1 with offset, `s1` + baseline | 227 = 56.5% | 59.0 | 54.0 | 83.6 | 63.4 | 22.4 |
| **Head vote**, `s2` + baseline, K = 5 | 366 = 91.0% | 92.0 | 90.1 | 92.5 | 94.0 | 86.6 |
| **Head vote**, K = 8 | 367 = 91.3% | 93.5 | 89.1 | 97.8 | 93.3 | 82.8 |
| **Head vote**, K = 32 | 368 = 91.5% | 92.0 | 91.1 | 96.3 | 91.0 | 87.3 |
| Head vote, `s2` without baseline, K = 8 | 313 = 77.9% | 69.0 | 86.6 | 68.7 | 82.1 | 82.8 |
| Head vote, `s1` + baseline, K = 8 | 200 = 49.8% | 51.0 | 48.5 | 6.0 | 80.6 | 62.7 |
| Head vote, the whole choice in nested CV | 362 = 90.0% | 92.0 | 88.1 | 93.3 | 93.3 | 83.6 |
| Conditional logit, `lift2` (384 weights) | 366 = 91.0% | 95.0 | 87.1 | 97.8 | 97.8 | 77.6 |
| Conditional logit, both scaffolds' lift and shares | 375 = 93.3% | 95.5 | 91.1 | 100.0 | 97.0 | 82.8 |

- The **head vote**: the K heads with the most training hits as R1 each name
  their winner segment (its largest share, less the `-na` share), and the
  most-voted segment is the answer. The nested row chooses scaffold,
  baseline and K ∈ {1, 3, 5, 8, 16, 32} inside each training fold; it chose
  `s2` with the baseline every time and K = 5 to 32. Fitted on all of A+B
  it is `s2`, baseline, K = 32 (heads in spec 19, track L).
- The **conditional logit** learns one weight per head over the per-segment
  lift, softmax over a question's segments (AT2's shape at the line level),
  λ by an inner CV; the last row needs all four prefills.
- An **anchored** vote (spec 14's rule in one dimension) is within a
  question of the plain vote (91.0% at K = 8).
- **The offset reading does not help the copy scaffold**: CV keeps offset 0.
  Under the index scaffold the boundary heads with offset +1 read 83.6% of
  logs, and nothing else well.
- **Top-k** (the ceiling of a shortlist-then-`choice` hybrid): R1 70.4 /
  88.8 / 94.8 / 96.5% at k = 1, 2, 3, 5; the vote 91.5 / 96.5 / 98.5 /
  99.3%; the full logit 93.3 / 98.3 / 99.0 / 99.5%.
- **Confidence**: the vote's agreement (the winner's share of the votes)
  separates present from absent questions at an AUC of 0.77 / 0.81 / 0.78
  (K = 5 / 8 / 32) in CV, against 0.69 for R1 on C.

### The lexical misses

- R1's lexical misses in CV (73 of 200) land more than two segments away on
  33, in the first tenth of the state on 25 (paraphrase: 15 of 46), on a
  segment sharing a content word with the instruction on 25; the baseline
  moved the winner on 7 of them.
- **The chosen head does not look at the rare word.** Under `lift2` the best
  rank L39.h12 gives any rare-word key is a median 37th; the rare key is its
  argmax on 5.5% of lexical questions. The head that takes the rare word
  most often is L47.h20 (19.5% under `s2`), then L59.h20, L59.h10 (~11%).
- **The vote does not have the paradox**: lexical 92.0%, paraphrase 91.1%.
  The heads that carry the lexical match are other heads than the one R1
  chose.
- The span's log-normaliser moves the same way on both halves (L39.h12,
  `logZ_q − logZ_na`: +2.57 lexical, +2.87 paraphrase): what little the
  span-only dumps can say about mass leaving the span does not separate them.
- **The misses land where the content-free prefill already looks** — the
  literature pass's hypothesis H4
  ([`2026-09-27-span-from-attention-literature.md`](2026-09-27-span-from-attention-literature.md)).
  With L39.h12 fitted on all of A+B (in-sample, descriptive), the winning
  line of a lexical miss is among the three lines the `-na` prefill gives
  the most mass on 35 of 67 (median rank 3), against 29 of 133 lexical hits
  (the target, median rank 12) and 8 of 31 paraphrase misses (median 8).
  The question adds mass to lines the head favours anyway, and the
  baseline's subtraction does not remove it.
- **No inhibition of the shared word** (H2): the head's lift on the target's
  rare-word keys is positive on the misses too (median z 2.0, hits 2.4). And
  the target is not ignored (H1's strong form): its best key's lift is z 2.7
  on lexical misses against 3.1 on lexical hits and 3.3 on paraphrase hits
  — weaker, not flat. What the instruction's own copy of the word does
  (H1) needs the full row, which phase 1 dumps.

### Replication on C (registered first, not binding)

The frozen vote (`s2`, baseline, the 32 heads fitted on A+B), read on C with
`score.py check` after its registration: C chose nothing here, but its
failures were read in spec 18, so this is a replication, not the check.

| C | vote | labelled `choice` | R1 (spec 18) |
|---|---|---|---|
| present, ≤ 256 segments | **147/161 = 91.3%** | 147/161 = 91.3% | 118/161 = 73.3% |
| paraphrase | 73/78 = 93.6% | 68/78 = 87.2% | 71/78 |
| lexical | 74/83 = 89.2% | 79/83 = 95.2% | 47/83 |
| logs / records / prose | 45 / 45 / 57 of 47 / 47 / 67 | 44 / 44 / 59 | 32 / 31 / 55 |
| all present (201) | 184 = 91.5%, top-3 99.0% | — | 145 = 72.1% |
| by length: ≤ 2.0K / ≤ 4.6K / ≤ 17.9K keys | 88.5 / 97.9 / 92.5% | only up to 256 segments | 73.5 / 72.9 / 67.5% |
| confidence AUC, present / absent | 0.79 (all of C) | 0.82 (its 188) | 0.69 |

The vote finds 45 targets R1 missed and loses 6 R1 found. Spec 18's rule 2
applied to these numbers would read go (gap 0.0 overall, −6.4 on the
paraphrase half); by its own terms that is for set D to say.

## Finding

Observed:

- On the development sets, a **majority vote of the heads with the most
  training hits**, each read at the copy scaffold less its content-free
  prefill, finds the line, record or sentence 90.0-91.5% of the time in
  cross-validation, against 70.4% for spec 18's single head; on C, read
  after the reading was frozen, it ties the labelled `choice` at 91.3% and
  holds 92.5% at 17.9K keys, a length the labels cannot reach.
- The lexical failure was the single head's: under the vote lexical and
  paraphrase questions are read alike on A+B (92.0 / 91.1%). The head R1
  chose does not attend to the rare word the question shares with its
  target; other heads do.
- Learned per-head weights (AT2's shape) add up to 1.8 points over the vote,
  at the cost of every head and, for the best of them, four prefills.
- Among 384 heads none peaks on the target's first token on more than 6% of
  the questions; many peak on the boundary after the target, and those
  peaks are mostly structural (they shrink under the lift). L39.h10 under the index scaffold reads the next
  number after the target, the timestamp of the line after it.
- The index scaffold is a poor query position for text: every reading under
  it trails the copy scaffold's, and on logs its heads read the line after.

Inferred, not measured:

- The vote works because each head's errors are its own: the heads fail on
  different questions, and a plurality survives any one head's miss. The
  data are consistent with that (R1's top-3 already holds the target on
  94.8%) but no per-head error correlation was computed.
- Nothing here says why a lexical question moves the single head. The
  instruction-copy mechanism of spec 18's finding is still untested: the
  dumps hold no key outside the state.

## Implications

- **Track L of spec 19 has a candidate that beat the bar on the development
  sets and on C.** It is registered in spec 19 (the reading, the 32 heads,
  spec 18's rule 2 verbatim) and judged once on set D. A go reopens #275
  with the vote as its reading; the engine side is spec 14's shape (a head
  set on the seam, ADR 0039), with 32 heads over L35-L63 and one extra
  prefill for the content-free baseline (about 55 tokens once the state's
  prefix is shared).
- **Track H is not needed unless L fails**: the vote's top-3 ceiling (98.5%)
  is what a shortlist-then-`choice` hybrid could add to, and the vote alone
  is at the labelled route's level.
- **A `found` flag is back in reach**: vote agreement separates absent from
  present at 0.77-0.81, near the labelled route's 0.82. It needs its own
  calibration and bar before any use.
- **For spans (track S, phase 1)**, the copy scaffold and the lift are the
  ordinates to dump; the index scaffold can be dropped; a head combination
  should be the default reading, not the single head. No initiator head was
  found, so a span's start will have to come from somewhere other than one
  head's argmax — the vote's segment, the peak's width, or the
  teacher-forced quote's first token.
- **Values and gates** (spec 19 Q3) are not what limited the line reading;
  they move to a lower priority for phase 1.

## Limits and unknowns

- A and B chose everything; the plain-CV rows can flatter a configuration
  chosen among 24. The nested row (90.0%) is the honest estimate of the
  procedure, and C's replication agrees with it. D decides.
- One artifact (the served NVFP4 27B), one render per scaffold, hq-e8-2b
  with the consumed keys, fresh prefills (no shared prefix read from the
  cache); the same caveats as spec 18.
- K = 32 won by one question over K = 8 (four fewer layers) on A+B; the
  engine cost of either is small by spec 14's measure, but it has not been
  measured for text spans of up to 18K keys.
- The rare-word analysis covers the lexical questions whose re-tokenization
  matched the dump (all logs and records, part of prose).
- Nothing here measures the attention outside the state's span (the
  instruction, the scaffold), values, or gates.

## Follow-ups

- Set D: the harness dump and the labelled route, then `score.py check` with
  the registered vote (spec 19, track L), once the owner confirms the rule.
- Phase 1 of spec 19 for spans and profiles, with the vote as the default
  reading and the full row dumped at the copy scaffold.
