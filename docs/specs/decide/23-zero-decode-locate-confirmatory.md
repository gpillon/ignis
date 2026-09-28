# Spec 23 — zero-decode `locate` for very long logs and prose (research, confirmatory)

Status: **RUN** (registered in `c5ef3f3`, judge `fd3712f`, before sets R4 and P2 existed; judged once on 2026-09-28): HL1, HL2, HP2, HP3 and HA pass; **HP1 fails** on its 1M group (7/12) — [finding](../../findings/2026-09-28-zero-decode-locate-very-long-logs-and-prose.md). Branch
`locate-long-context` (experiment). Follows the exploratory finding
[zero-decode locate on real logs](../../findings/2026-09-28-zero-decode-locate-exploration.md)
and its second round (logs folded then read by heads and a labelled
`choice`; prose read by heads in windows then a labelled `choice` with
several pointers), whose choices were made on R, R2, RX (1M-token logs) and P
(long HotpotQA prose). This spec freezes those choices and judges them once,
on two sets nobody has read.

## The pipelines (frozen)

No token is generated anywhere: every step is a prefill that reads either
attention rows (`locate`'s readout, the research hooks of `bb35244`) or the
logits of answer labels (the labelled `choice`).

**`kind: log`** (`tools/locate-sets/zd_logpipe.py`, configuration
`heads-end5+choice / L2 choice-end`):

1. Fold the log (`compress.fold`, SIM 0.5, every slot's distinct values).
2. Level 1: the **end heads** (`.scratch/locate/zd/endheads.json`, 32 heads
   chosen on sets A+B) read over the templates; each template scored by the
   **end reading** — each head's lift (question less content-free twin) at
   the line's last key, its separator and the next line's first key,
   standardized over the lines, summed over heads; the first 5 templates, in
   order, go to a labelled `choice`.
3. Level 2: the chosen template's rows (values first, time last). Past 16
   rows, the end heads' end reading over the rows keeps the first 16 (in
   document order); a labelled `choice` picks the row.
4. Unfold: the row's first original line is the answer.

**`kind: prose`** (`zd_windows.py` then `zd_prose.py multi`):

1. The text in sub-windows of at most 210,000 tokens, cut at empty lines.
2. The **served 32 heads** read each sub-window; each line scored by the
   **sum reading** (each head's lift over the line's keys, standardized over
   the lines, summed over heads), standardized within its sub-window and
   merged; title lines (`# ...`) and empty lines excluded.
3. The first 16 lines, shown in their paragraphs (title and every sentence,
   the 16 labelled, the rest as context), one labelled `choice`.
4. **Pointers**: every label whose probability is at least 0.05 (at least
   the best one).

**`kind: auto`**: fold the first 2,000 non-empty lines; if at least half of
them fall in templates of two or more lines, `log`, else `prose`.

## The sets (built after this commit)

- **R4** (logs): a fresh capture of the owner's cluster — `kubectl logs
  --since=6h --timestamps` of its running pods, **read only**, later than any
  earlier capture — merged by `prodset.py timeline`; windows of 100K and 200K
  tokens (two each) and ~1M tokens (two), cut by `r2set.py`'s rule. Targets
  drawn by `zd_rxq.py draw`'s rule (sampled eligible lines, round-robin over
  the sibling bins 0, 1-5, 6-50, > 50), five per window. Questions written by
  Claude after seeing the targets and before any model answer, checked by
  `zd_rxq.py build`'s rules; a target no question can single out is dropped,
  not replaced. Data never enters the repository.
- **P2** (prose): `prosehay.py --seed 20261090`, excluding every HotpotQA
  question of sets A-F and P; tiers 16K / 64K / 128K / 200K (four windows
  each) and 1M (two windows); six questions per window; all of it is test.

## Hypotheses and rules (present questions)

- **HL1**: the `log` pipeline's answer is the target line on at least **85%**
  of R4.
- **HL2**: it is at least the labelled `choice` over all the templates then
  the rows (folded + `choice`, spec 21's fold with the `choice` route).
- **HP1**: the `prose` pipeline's best pointer (the `choice`'s pick) is a gold
  sentence on at least **85%** of P2, and at least **80%** within each of
  the ≤200K and 1M groups.
- **HP2**: paragraph-level pointer F1 (pointers mapped to their paragraphs
  against the gold sentences' paragraphs) averages at least **0.75** on P2.
- **HP3**: a gold sentence is among the served heads' first 16 on at least
  **95%** of P2.
- **HA**: `auto` names every R4 window `log` and every P2 window `prose`.

Reported beside them, no rule: wall time per question (median, p90), the
heads-only variants, all-gold recall within the first 16, sentence-level F1.
Each hypothesis is judged once; a failed one is reported as failed.

## Limits registered in advance

One artifact; R4's questions written by one author who saw the windows and
targets; P2 is HotpotQA's gold sentences in a haystack of other Wikipedia
paragraphs (the gold sentences are strict: a neighbouring sentence that also
helps counts against precision); the latency is the research server's, one
client at a time.
