# Zero-decode locate, third round: paragraphs at 1M, "not found", JSON records

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` with no token generated, after [spec 23](../specs/decide/23-zero-decode-locate-confirmatory.md): the paragraph as the prose pointer at ~1M tokens; an answer that the text does not hold, for logs, prose and records; long JSON arrays of records as a kind of their own. **Exploratory throughout**: P and P2 were read before (P2 is spec 23's judged set), R/R2/R4 are earlier sets, and the new sets (P2abs, J, J2) were built and read in this round.
- Related: [very long logs and prose](2026-09-28-zero-decode-locate-very-long-logs-and-prose.md) (round 2, spec 23); [zero-decode locate on real logs](2026-09-28-zero-decode-locate-exploration.md) (round 1); tools `zd_para.py`, `zd_notfound.py`, `zd_verify.py`, `zd_prose_nf.py`, `zd_records.py`
- Superseded by: none

## Question

Spec 23 left three things open: prose at ~1M tokens (its registered
weak point, HP1 7/12), a `locate` that can answer "not here", and JSON
records, which `auto` sends to the `log` pipeline. The owner asked for all
three, with the 1M prose kept light (it will be studied again).

## Evidence

### 1. Prose at ~1M tokens: the paragraph as the unit

`zd_para.py`. The heads' ranking (served heads, sum reading, windows ≤210K,
unchanged) turned into a paragraph ranking — a paragraph ranks where its best
sentence does — holds a gold paragraph among its first 8 on every question
of P, P2 and P's 500K/1M windows. A labelled `choice` over those 8 whole
paragraphs, each labelled at its title ("Which paragraph helps answer ..."):

| ≥500K tokens (P 500K+1M 24, P2 1M 12 = 36) | right |
|---|---|
| spec 23's sentence `choice` (16), the sentence | 29/36 |
| the same, the sentence's paragraph | 32/36 |
| **paragraph `choice` over the first 8** | **35/36** (P2 1M 11/12, P 24/24) |
| paragraph `choice`, then a sentence `choice` inside the paragraphs it points at | 31/36 (sentence) |
| sentence `choice` over 8 instead of 16 | 30/36 (sentence) |

Up to 200K nothing changes (P2 paragraph pick 96.9% both ways). The
paragraph pointer set does **not** improve: the paragraph `choice` puts
nearly all its mass on one paragraph (paragraph recall 67-72%), so
paragraph F1 at p ≥ 0.05 is 77.4 on P2 (spec 23's sentence pointers 77.1)
and 76.0 on P's long windows (80.4). A better first pointer, not a better
set.

### 2. "Not found"

Absent questions, three kinds:
- **logs**: R and R2's 20 authored absent questions, and **deleted** ones —
  every present question of R2 (58) and R4 (23) asked over its log with the
  target line removed (the sets' rule made the target's normalized text
  unique and the question's words together in no other line, so nothing
  answers it; its near-duplicates stay — the hardest absent);
- **prose** (`P2abs`): each P2 question of the 16K/64K/128K windows over its
  window without its two gold paragraphs (its eight HotpotQA distractor
  paragraphs, retrieved for it, stay; 72), and 12 questions of another 200K
  window over a 200K window whose text holds none of their gold sentences
  (`cross`);
- **records** (J, below): a value no record holds (12).

Three readings, all in the same request as the pipeline's final `choice`:
an option "no line/sentence/record answers" added to the `choice`
(`p_none`); a yes/no "Is there a line in the evidence that answers ..."
(`found`); and the `choice`'s own top probability.

| where the check looks | logs R+R2 (present 108 / absent 20 / deleted 58) | logs R4 (23 / – / 23) |
|---|---|---|
| `none` in the level-2 `choice` (rows as values only) | keeps 37 of 100 found; AUC 0.67 / 0.70 | AUC 0.81 |
| the answer's original line alone, `none` + yes/no (mean > 0.5) | keeps 100/100; flags 13/20, 36/58; AUC 0.96 / 0.94 | keeps 21/21; flags 20/23 |
| **the shortlisted rows' original lines, final `choice` + `none` + `found`** (mean > 0.5) | **keeps 101/101; flags 16/20, 41/58**; AUC of the mean 0.97 / 0.93 (`p_none` 0.98 / 0.94, `1-found` 0.93 / 0.92) | **keeps 21/21; flags 20/23**; AUC of the mean 0.92 (`p_none` 0.88, `1-found` 0.94) |

A folded level-2 row is its values only: whether it answers cannot be read
from it, and `none` wins on present questions. Shown as their original
lines, the same 16 rows make the final `choice` both as accurate (101 vs
100 of 108) and able to say "none". `none` alone loses five of R4's
questions ("When did ...", `p_none` 0.5-0.8 with `found` 0.7-0.9); the mean
of the two keeps them — a rule chosen after seeing R4, which is therefore
no held-out check of it.

| prose (P2 ≤200K present, same tiers) | deleted (72) | cross (12) |
|---|---|---|
| `none` in the sentence `choice` (16 in paragraphs), > 0.5 | keeps 66/67; flags 45/72; AUC 0.937 | keeps 21/21; flags 12/12; AUC 1.0 |
| the same, mean with `found` > 0.5 | keeps 64/67; flags 60/72 | keeps 19/21; flags 12/12 |
| `none` in the paragraph `choice` (8), > 0.5 | keeps 69/70; flags 49/72; AUC 0.898 | keeps 21/23; flags 12/12 |
| the pick alone in its paragraph | AUC 0.85; keeps 50/67 | AUC 0.98 |

In prose the shortlist is already shown in its paragraphs, and `none` in the
`choice` works as it is; checking the pick alone is worse, because a
HotpotQA gold sentence often answers only together with the other paragraph.

Records (J): `none` in the final `choice` keeps 57/58 and flags 12/12
(AUC 1.0) — an absent value is easy there.

### 3. JSON records as a kind

`zd_records.py`, set **J** (seed 20261130): spec 18's record kinds
(employees by city, tickets by title, products by name) in arrays of 1,000
(~55K tokens) and 3,500 records (~195K), three arrays each; five targets
per array whose selecting value is unique, a lexical and a paraphrase
question each (spec 18's checks, against every other record), and one
absent value per array — 60 present, 12 absent. The heads read the
**native JSON array** (the segments are the records, as the product
segments a JSON state).

| heads over the array, one pass | top-1 | first 16 |
|---|---|---|
| end heads, end reading | 86.7 | 96.7 |
| end heads, end + sum | 90.0 | 98.3 |
| served heads, sum reading | 85.0 | 96.7 |
| served heads, end reading | 68.3 | 95.0 |
| end heads, sum reading | 56.7 | 98.3 |

| pipeline | lexical (30) | paraphrase (30) | latency |
|---|---|---|---|
| **records: end heads' first 16 (array order, spaced JSON, labelled) → `choice`** | **30** | **28** | first question: the prefill (8 s at 55K, 45-48 s at 195K); then 1.0-3.3 s + 0.3 s |
| the same from the served heads' sum reading | 30 | 27 | |
| the `log` pipeline over the records as JSON lines (what `auto` does today) | 28 | 24 | median 1.3-1.7 s, no long prefill |

The fold does not suit records: its template key (a line's token count)
splits records whose values have different word counts (12-200 templates
per array), and a template summary caps the selecting values out of view;
level 1 names the target's template 53 of 60 times. Records are read like
log lines — at their end, by the end heads — but hold few near-duplicates
per question, so one window's heads and a `choice` suffice.

**J2** (seed 20261140, 10,000 records, ~550K tokens, three arrays, 30
present and 6 absent questions; windows of ≤200K tokens cut at record
boundaries, the end heads only, scores standardized per window and merged):
the end reading puts the target first 90.0% (end + sum 93.3%, sum alone
40.0%) and among the first 16 96.7%; the `choice` over those 16 picks it
29/30 (lexical 15/15, paraphrase 14/15 — the miss is outside the 16);
`none` in it keeps 28/29 and flags 6/6 (AUC 0.99). The first question over
an array pays each window's prefill (~45 s per 200K); a later one ~2.9 s
per window plus 0.3 s.

**Telling records**: a state that is a JSON array whose every element is an
object is `records` — all 400 record arrays of sets A-F and every J array,
no log or prose state (there are no JSON-lines strings in any set). It needs
no statistic: today's `auto` (the fold's share) sent 18 of 160 short record
arrays to prose.

## Finding

At ~1M tokens of prose the first pointer is right far more often at the
paragraph than at the sentence: a labelled `choice` over the heads' first
8 paragraphs picks a gold one 35 of 36 times past 500K tokens (the
sentence `choice`'s own paragraph: 32; its sentence: 29), though its
pointer set is no better. A `locate` can say "not found" with no token
generated, provided the last `choice` sees its candidates in a form that
can be checked: an option "none" beside the candidates, read together with
a yes/no in the same request, keeps every log answer the pipeline found and
flags 16 of 20 authored and 41 of 58 near-duplicate absent questions when
the 16 shortlisted rows are shown as their original lines — where over the
folded values it fails (37 of 100 kept); in prose, over the shortlist in
its paragraphs, `none` flags 45-60 of 72 hard absent questions and all 12
easy ones, losing 1-3 present. JSON records want neither the fold nor the
prose reading: the end heads over the native array, then a `choice` over
their first 16, find 58 of 60 targets at up to 3,500 records and 29 of 30
at 10,000 (~550K tokens, windows), against 52 of 60 for the `log`
pipeline, and a record array is told by its shape alone.

## Implications

- **`kind: records`** is supported: a JSON array of objects (told
  structurally, not by the fold); no compression; the **end heads, end
  reading** over the array in windows of ≤200K tokens cut at record
  boundaries; the first 16 in array order as spaced-JSON lines; a labelled
  `choice`. The same shape as `prose` with the log's reading.
- **Not found, one design for every kind**: the last `choice` of each
  pipeline gains an option "none" and a yes/no (`found`) in the same request,
  over its candidates shown readable — logs: the 16 rows' **original
  lines** (which also makes the last `choice` read the lines rather than
  their values: 101 vs 100 of 108); prose: the 16 sentences in their
  paragraphs (as today); records: the 16 records. The answer carries
  `found` (a probability) and names no segment when it is below 0.5; the
  measured rule is the mean of the two for logs and `none` alone for prose
  and records (exploratory choices).
- **Prose pointer unit**: a `paragraph` pointer (the paragraph `choice` over
  the first 8) is the better single answer at ~1M tokens; sentence pointers
  stay the better set. Worth a value of its own (`unit: sentence |
  paragraph`) before it is a default.
- All three need a confirmatory spec on fresh sets (prose at 1M with more
  than 12 questions, fresh absent questions, real JSON records) before spec
  22 takes them.

## Limits and unknowns

- Exploratory: P2 was judged by spec 23 and every rule here (paragraph
  `choice`, where the check looks, the mean rule, the records pipeline) was
  chosen on the data it is reported on. A confirmatory spec on fresh sets is
  the next step before any of it is a default.
- The "deleted" absent is one construction (the target removed, its
  near-duplicates and distractors kept); authored absent questions are only
  20 (logs). Records' absent questions are easy (a value no record has).
- The records are synthetic (spec 18's generator: one selecting value per
  question); real records (API responses, database dumps, JSON-lines logs)
  and questions over several fields were not tried.
- One artifact; the research server's latencies; one client.
