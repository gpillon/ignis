# The shortlist `locate` through `/v1/decide`: spec 22's acceptance, partial

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` as served on the NVFP4 27B under `make start`'s defaults (hq-e8-2b, 262,144 context, dflash2), the defaults of [spec 22](../specs/decide/22-locate-by-copy-over-a-folded-state.md) (GitHub #278) on its registered sets R3 (real logs), P3 (prose) and J3 (JSON records), and set F under both methods. **Partial**: runs 1, 2, 5 and 6 whole; run 3 on P3's present questions to 200K tokens and 2 of its 12 questions at ~1M; run 4 on J3's 1,000- and 3,500-record arrays; P3's absent questions, J3's 10,000-record arrays and R3's Apache and Proxifier windows after a fix are not run (the owner stopped the runs to save GPU time).
- Related: [spec 22](../specs/decide/22-locate-by-copy-over-a-folded-state.md), [ADR 0042](../adr/0042-locate-copies-over-a-folded-state.md) (stays Proposed until the acceptance is whole), [very long logs and prose](2026-09-28-zero-decode-locate-very-long-logs-and-prose.md), [paragraphs, not found, records](2026-09-28-zero-decode-locate-paragraphs-not-found-records.md), [locate through `/v1/decide`](2026-09-27-locate-through-decide.md); tools `accept22.py`, `r3_judge.py`, `r3set.py`
- Superseded by: none

## Question

Does the served shortlist — heads narrow, a labelled `choice` decides, `found`
says when nothing answers — do on fresh sets, registered before any route was
asked on them, what the research measured, and is the vote unchanged beside
it? Spec 22's rules 1-11, judged once.

## Evidence

Data: `.scratch/locate/accept22/` in the main checkout (runs' records,
`r3-judge-partial.json`, the server's JSON log); sets under `.scratch/locate/`
(R3, R3src, zd/P3, zd/J3), hashes in spec 22's banner. One `locate` per
request, sequentially, R3 first on a freshly started server.

| rule | measured | floor | verdict |
|---|---|---|---|
| 1 logs: R3 right | **107/120 (89.2%)** | 85% | pass |
| 2 heads' part: defaults' top-1 vs the `choice` alone over the fold | 107 vs 97 | ≥ | pass |
| 3 latency: median first question of R3's 100K windows | **2.05 s** (18 windows) | ≤ 3.0 s | pass |
| 4 prose: P3 right to 200K; paragraph pointer F1 | **83/96 (86.5%)**; 0.789 (98 questions) | 85%; 0.75 | pass (1M: 2 of 12 asked) |
| 5 records: J3 right | **57/60 (95.0%)** on 1,000 and 3,500 records | 90% | pass on what ran; 10,000 not run |
| 6 not found, logs | kept 107/107; target-removed flagged 101/120 (84.2%); authored flagged 42/42 | 95%; 60%; 65% | pass |
| 7 not found, prose | kept 83/85 (97.6%) | 95% | present part pass; absent questions not run |
| 8 not found, records | kept 57/57; absent flagged 12/12 | 95%; 90% | pass on what ran |
| 9 `auto` | every P3 and J3 state and set F's record arrays told right; R3 **275/282** | every one | **fail**: `h-apache-200k` read as prose |
| 10 the vote unchanged (set F, `vote` + `none`) | 240 questions, **no difference** from the recorded run | none | pass |
| 11 short states (set F, defaults) | logs 41/43, records 46/46, prose 61/67 | 39, 44, 56 | pass |

Reported beside the rules (`r3-judge-partial.json`):

- R3 by source: the cluster 38/39, LogHub 69/81; by tier 16K 8/8, 50K 7/7,
  100K 39/46, 200K 46/51, ~1M 7/8; by sibling bin 0: 24/28, 1-5: 25/27,
  6-50: 32/33, >50: 26/32. Level-1 accuracy (the fold's template) 100/117.
  The `choice` alone over the fold (run 2) 97/120.
- R3 wall time: median 1.10 s per question, p90 5.3 s; the first question of
  a 200K window 3.05 s median (p90 35 s); `found`'s AUC 0.975 against the
  target-removed questions and 1.0 against the authored ones.
- P3 by tier: 16K 20/24, 64K 22/24, 128K 18/24, 200K 23/24; sentence pointer
  F1 0.65; a gold sentence among the 16 candidates on 98/98. Wall time median
  0.8 s (16K) to 7.6 s (200K); **a ~1M-token question took 360 s**, the
  second as the first: its five windows' prefixes were not kept between
  requests.
- J3: lexical 30/30, paraphrase 27/30; 1,000 records 28/30, 3,500 29/30;
  median 1.1 s and 3.5 s.
- Set F under the defaults: present right logs 65/67, records 67/67, prose
  61/67; absent flagged logs 13/13, records 13/13, prose 6/13.
- Host time: `auto` 12-40 ms per state; the plan (fold and renders before the
  first prefill) median 0.12 s on R3, 2.6 s on P3 (to 34 s at ~1M: every
  window's prompts are tokenized per request).

The `auto` failure: every Apache line opens with `[Sun Dec 04 04:47:44
2005]`, which the fold — as `compress.py` measured it — read as a source
label, so every line keyed a template of its own and the share of lines in
shared templates fell under 0.5; Proxifier's `[10.30 16:49:06]` did the same
(read as a log anyway, with thousands of templates). A bracket that holds a
time is now a timestamp, not a label (commit a038c35, held to a new golden
case); after it, `auto` reads all four Apache and Proxifier windows as logs
(shares 0.999-1.0, 10-24 templates). **This fix was made after R3's answers
were seen**: those four windows are no longer a blind judgement, and they are
not re-asked here.

## Finding

Observed: on fresh sets registered in advance, the defaults hold every rule
spec 22 set that could be judged on the runs made — logs 89.2% (the vote read
34.5% of a comparable set), prose 86.5% to 200K, records 95%, "not found"
keeping every found log and record answer while flagging 84% of the hardest
absent log questions and every authored one — in a median 1-2 s for logs,
and the vote is unchanged beside them. Rule 9 failed on one window, through a
fold rule that treated a bracketed timestamp as a source label; the rule is
corrected, not re-judged.

Inferred: the shortlist is the better default for a `locate` at every
measured length and kind; its weak point is cost, not accuracy, at ~1M
tokens of prose or records read without a fold (every window prefilled per
request).

## Implications

- Spec 22's acceptance is not whole: ADR 0042 stays Proposed. What remains is
  a verification, not a design question: P3's absent questions, P3's ~1M
  group, J3's 10,000-record arrays, R3's Apache and Proxifier windows under
  the corrected fold, and the judge over all of it.
- Keeping a long text's window prefixes between requests would turn a ~1M
  prose question from 360 s into a few seconds; tokenizing each window once
  per text would cut the plan's host time.

## Limits and unknowns

- Partial runs, listed above; the unrun parts may move rules 4, 5, 7 and 9.
- One artifact; R3's questions were written by one author (three sessions)
  who saw the windows and targets, before any answer.
- The fold's bracket fix was chosen after seeing R3; the other 38 R3 windows
  are unaffected by it (no bracketed time in their lines).

## Follow-ups

- A ticket for the rest of the acceptance and the prefix reuse at ~1M.
