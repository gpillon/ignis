# Zero-decode locate for very long texts: logs fold, heads narrow, a labelled choice decides; prose read by heads in windows

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` with no token generated over logs to ~1M tokens (compressed, then attention heads and a labelled `choice` at each level) and prose to 1M tokens (heads in windows, a labelled `choice` over their shortlist, several pointers); a `kind` switch (`log` / `prose` / `auto`); exploratory on R, R2, RX, P, then **confirmatory** on fresh sets R4 and P2 ([spec 23](../specs/decide/23-zero-decode-locate-confirmatory.md))
- Related: [zero-decode locate on real logs](2026-09-28-zero-decode-locate-exploration.md) (round 1: end heads, shortlist + choice); [locating a line in real logs](2026-09-28-locating-a-line-in-real-logs.md) (the fold); tools `zd_logpipe.py`, `zd_windows.py`, `zd_prose.py`, `zd_hop.py`, `prosehay.py`, `zd_rx.py`, `zd_rxq.py`, `z23_r4.py`, `z23_judge.py`
- Superseded by: none

## Question

The owner's direction after round 1: for line-structured logs, combine
compression, a head shortlist and the labelled `choice`; for prose, where
compression does not apply, let the heads find the passages and a `choice`
decide, returning **several pointers** when the answer spans several places;
prefer the heads (the publication is about them) without being bound to
them; and design for **very long** texts — past the engine's 262K context.
Which pipeline, per kind of text, and how is the kind told?

## Evidence

### Spec 23, confirmatory (registered in `c5ef3f3`, judge `fd3712f`, both before the sets existed)

- **R4**: a fresh capture of the owner's cluster (147 pods, six hours, read
  only, later than every earlier capture); windows of 100K (two), 200K (two;
  one had no eligible line) and ~1M tokens (two); 25 targets drawn by rule,
  23 questions (two dropped by the check, not replaced).
- **P2**: `prosehay.py` seed 20261090, no question shared with A-F or P;
  16K-200K (four windows per tier) and 1M (two); 108 questions.

| spec 23 | rule | result | |
|---|---|---|---|
| HL1 `log` pipeline on R4 | ≥ 85% | **21/23 = 91.3%** | pass |
| HL2 vs folded + `choice` | at least equal | 21 vs 16 | pass |
| HP1 `prose` pick is a gold sentence | ≥ 85% overall and ≥ 80% per group | 88.0% overall; ≤200K 91.7%; **1M 7/12 = 58.3%** | **fail** |
| HP2 paragraph pointer F1 | ≥ 0.75 | 0.771 (sentence F1 0.661) | pass |
| HP3 a gold sentence in the heads' first 16 | ≥ 95% | 100% (every gold sentence: 74.1%) | pass |
| HA `auto` | every window right | 23/23 (logs 0.96-0.98, prose 0.04-0.07) | pass |

Wall time on R4 (the `log` pipeline): median 2.6 s, p90 6.2 s.

HP1 fails on its 1M group (12 questions). Read after the judgement: the
heads' first 16 held a gold sentence for all twelve; of the five wrong picks
three are another sentence of a gold paragraph (one, "The breed is named for
Jocelyn Lucas.", answers the question though HotpotQA marks a different
sentence) and two are hard distractors of two-hop questions. By paragraph the
pick is right 104/108 (96.3%; 1M 10/12).

The second hop, tried on P2 after the judgement (exploratory): paragraph
pointer F1 77.1 → 83.5 with the `choice`'s pointers plus the hop's other
paragraph (≤200K 78.8 → 85.4; 1M 63.9 → 68.8) — the gain P showed (+6)
holds on the fresh set.


### Logs: fold, then heads and a labelled choice at each level (exploratory)

`zd_logpipe.py`: the log folded (`compress.fold`); level 1 = templates with
their values, level 2 = the chosen template's rows (values first). At each
level the end heads (32, chosen on short sets A+B, round 1) read the lines
by the end reading, and/or a labelled `choice` decides. No long prefill:
level 1 of a 1M-token log is 25-34K tokens (30-40x shorter).

| level 1 → level 2 | R (50) | R2 (58) | RX (41, ~1M tokens) |
|---|---|---|---|
| heads top-1 → heads top-1 (heads only) | 30 | 29 | 17 |
| labelled choice over all templates → choice over rows | 42 | 53 | 33 |
| **heads' first 5 templates → choice; rows: heads' first 16 → choice** | **45** | **55** | **36** |
| the same with the sum+end reading | 45 | 56 | 36 |
| wall time of the bold row (median / p90) | 0.8 / 2.9 s | 1.3 / 4.0 s | 5.0 / 16 s |

Level 2 is where the `choice` is needed: of the 55 R2 questions whose
template level 1 got right, the heads alone name the row 38 times and the
`choice` over their first 16 all 55 (R: 36 and 45 of 47; RX: 20 and 36 of
38). RX (the cluster's windows widened to ~1M tokens) holds 10
of R2's questions that still single their line out and 31 new ones written
for it (targets drawn by rule, one dropped). For comparison on R2: folded +
generation (spec 21, decodes) 55, whole-log end-head shortlist + choice
(round 1, a long prefill) 55-57, the served vote 20.

### Prose: heads in windows, then a labelled choice with several pointers (exploratory)

Set P (`prosehay.py`, seed 20261070): HotpotQA questions whose gold
supporting sentences (2-4 per question, in two paragraphs) are spread
through a haystack of other Wikipedia paragraphs — titles as lines, one
sentence per line — at 16K / 64K / 128K / 200K tokens (four windows each)
and 500K / 1M (two each, read in sub-windows of at most 210K tokens cut at
paragraph breaks and merged by standardized score).

| reading, served heads | ≤200K (96) | 500K (12) | 1M (12) |
|---|---|---|---|
| a gold sentence first (sum reading) | 85.4 | 25.0 | 83.3 |
| a gold sentence in the first 4 / 16 | 100 / 100 | 83 / 100 | 100 / 100 |
| every gold sentence in the first 16 / 64 | 78 / 93 | 83 / 92 | 58 / 67 |
| the end reading, first | 73 | 42 | 75 |

On prose the sum reading beats the end reading (the opposite of logs), the
served heads read as well as heads chosen on prose, and there is no fall with
length up to 200K (83% at 16K, 92% at 200K): prose has few near-duplicates.
Past one window the first place is unreliable (each window has its own best)
but the first 16 always hold a gold sentence — the `choice` decides:

| the labelled `choice` over the heads' first 16, in their paragraphs | ≤200K (96) | 500K+1M (24) |
|---|---|---|
| its pick is a gold sentence | **91.7** | **91.7** |
| paragraph pointers, p ≥ 0.05 (τ chosen on dev): precision / recall / F1 | 94.8 / 79.2 / 83.3 (test) | 87.4 / 79.2 / 80.4 |
| + a **second hop**: the heads read again with the pick in the instruction, the best line of another paragraph added | 86.8 / 92.7 / **89.2** (test) | — |

A yes/no (`noul`) per candidate flags the gold sentences' neighbours as well
(F1 60-64); the `choice`'s own probabilities are the better pointer set. The
second hop finds the bridge paragraph of a two-hop question that the first
reading misses (paragraph F1 76.6 → 85.4 for best + one other paragraph).

Wall time, per question on the retained prefix: 0.4 s (16K) to 4.3 s (200K) of
head reading per (sub)window, plus ~0.3-1 s for the `choice`; the first
question over a text pays its prefill (2 s at 16K, 68 s at 200K, ~6 min at 1M
in five sub-windows).

### Telling the kind

Fold the first 2,000 non-empty lines; the share of lines in templates of two
or more: logs 0.41-1.00 (median 0.96-0.98, R/R2/C/D), long prose 0.04-0.08
(P), short prose 0-0.65 (C/D), JSON records 0.10-1.00 (median 0.94). At 0.5:
1 of 198 log windows and 2 of 184 prose windows misread; 18 of 160 record
arrays read as prose.

## Finding

Folding a log and letting the heads narrow while a labelled `choice`
decides finds the asked line with no token generated and no long prefill:
on a fresh cluster capture of 100K to 1M tokens it reads 21/23 (91.3%),
registered in advance (spec 23), in a median 2.6 s, above the `choice`
alone over the folded log (16/23); on the earlier sets it reads 45/50,
55/58 and 36/41 at ~1M tokens, where the heads alone read 30, 29 and 17.
In prose the heads' sum reading puts a gold sentence among the first 16 every
time on the fresh set, to 1M tokens, and the `choice` over those 16 picks one
91.7% of the time up to 200K; at 1M it falls to 58% of 12 questions — the
registered hypothesis fails there — mostly to another sentence of the right
paragraph. Several pointers are best taken from the `choice`'s own
probabilities (paragraph F1 0.77 on the fresh set, 0.83 on P), and a second
head reading with the first pointer in the instruction finds a two-hop
question's other paragraph (P: 0.89). The kind of text is told by the fold
itself: the share of lines in shared templates separates logs (≥0.96) from
prose (≤0.07) on every fresh window.

## Implications

- **A `locate` with `kind: log | prose | auto`** is supported by the data:
  `log` = fold → heads over templates → `choice` among 5 → heads' 16 rows →
  `choice` → unfold (no long prefill; median 0.8-2.6 s on R, R2, R4, 5 s on RX);
  `prose` = heads (served, sum reading) in windows ≤210K → first 16 in their
  paragraphs → `choice` → pointers at p ≥ 0.05, optionally a second hop;
  `auto` = the fold share at 0.5. JSON records likely want their own kind.
- **The heads narrow, the `choice` decides.** In both kinds the heads are
  the step that makes very long texts tractable (a 1M-token log to 5 templates
  and 16 rows; a 1M-token text to 16 sentences), and the labelled `choice`
  is the step that picks among near-duplicates. Neither alone reaches the
  combination.
- **For the paper**: the end heads mark a log line's end; the sum heads find
  a prose sentence; the first depends on near-duplicate density, the second
  does not fall with length.
- **Cost of very long prose**: without compression the text is prefilled
  once (≈6 min per 1M tokens) and every sub-window's prefix must stay
  retained for later questions to be cheap; batching questions per text is
  what makes it affordable.

## Limits and unknowns

- One artifact. The prose sets are HotpotQA's gold sentences in a haystack of
  unrelated Wikipedia paragraphs (topic shifts make the haystack easier than
  one long document; the gold labels are strict). R4 and RX questions were
  written by one author who saw the windows and targets.
- The exploratory numbers (R, R2, RX, P) come from choices made on them;
  only spec 23's are confirmatory.
- Latencies are the research server's, one client, host-side choice
  rendering included.
- Not tried: a document-structure hierarchy for prose (sections → paragraphs
  → sentences), JSON records as their own kind, `found` (absent answers) for
  the new pipelines.
