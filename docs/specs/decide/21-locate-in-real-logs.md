# Spec 21 — locating a line in real logs (research, confirmatory)

Status: **RUN** (registered 2026-09-27 in `e17c5e8`, judge `c5d3386`, before set R2 was asked; judged
once on 2026-09-28): **every hypothesis passes** (H1 at its bar, 20.4 points) — [finding](../../findings/2026-09-28-locating-a-line-in-real-logs.md). Branch
`locate-long-context` (experiment). Follows spec 20 and its two findings
([locate at length](../../findings/2026-09-27-locate-at-length.md),
[the instance is read while copying](../../findings/2026-09-27-the-instance-is-read-while-copying.md)),
whose mechanisms were **exploratory** — read on set R, which also tuned the
folded method below. This spec judges them once, on a set nobody has asked.

## The set: R2

`tools/locate-sets/r2set.py` (seed 20261040), built before any run:

- **Sources**: the owner's cluster — `kubectl logs --since=6h --timestamps`
  of its 147 running pods, merged by time (`prodset.py timeline`), later
  than every window of R — cut into two windows per tier of 16K / 50K /
  100K / 200K tokens; and the public LogHub 2k samples, one window per
  system of 16K / 50K / 100K tokens (Apache has no eligible line).
- **Targets by rule**: a line's **siblings** are the lines whose word sets,
  times, numbers, ids and hashes removed, have a Jaccard of at least 0.5
  with it; lines with an exact twin (after that removal) are excluded;
  targets are drawn round-robin over the sibling bins 0, 1-5, 6-50, > 50.
- **Questions** were written for the drawn targets and checked by
  `r2set.py build`: lexical (a rare shared word), combo (the shared words,
  together, in no other line), paraphrase (no shared content word, only for
  targets without siblings). A target no question could single out under
  the checks was **dropped, not replaced** (9, each with its reason). Ten
  questions are **absent** (the target removed).
- 68 questions, 58 present: 34 LogHub, 24 cluster; 24 combo, 23 paraphrase,
  11 lexical; siblings 0 / 1-5 / 6+ = 23 / 19 / 16. The cluster's data never
  enters the repository; manifest sha256
  `3a35a887f65250623c6e0d99fcc101dbcaed379bdb339bd5a2e436166ace090f`.

## Runs

One server from this branch (`176cea5` + the folding tools), the ceiling
lifted, rows dumped:

1. `long.py ask --found`: the served vote, a sibling `noul`, and the
   generation route on the whole state.
2. `research.py --plan heads0 traj`: all 384 heads over the full prompt at
   the scaffold, and the 32 served heads along the target's teacher-forced
   first k tokens (k = 1 ... 64; the content-free twin carries the forced
   text, so the trajectory is read on the question's prefill alone).
3. `folded_locate.py --values`, with `--route vote` and `--route generate`:
   fold (`compress.py`, SIM 0.5, every slot's distinct values up to a
   600-character budget), level 1 over the templates, level 2 over the
   chosen template's rows (values first, time last), unfold.

## Hypotheses and rules (present questions unless said)

Mechanism:

- **H1, near-duplicates**: the served vote's miss rate on targets with 6+
  siblings exceeds that on targets with none by **at least 20 points**.
- **H2, one pass at length**: the fraction of the 384 heads whose one-head
  reading (lift) hits the target falls with the state's length — Spearman
  rho < 0 over present questions, p < 0.05 (one-sided).
- **H3, the state's share**: averaged over heads, the state's share of the
  row differs between R2's shortest and longest tier by **at most 5
  points**.
- **H4, the copy**: (a) on the vote's misses, the served heads' plurality
  on the target (question prefill alone) is **at least 20 points** higher at
  k = 32 than at k = 1; (b) over present questions with a trajectory, the
  first k at which that plurality lands on the target and stays through
  k = 64 correlates positively with the number of tokens after which the
  target's prefix matches no other line (Spearman rho > 0, p < 0.05,
  one-sided; a question never landing counts as k = 96).

Method:

- **M1**: folded + generation top-1 is **within 5 points** of the
  generation route on the whole state (non-inferiority).
- **M2**: folded + generation top-1 exceeds the served vote's by **at
  least 10 points**.
- Reported beside them, no rule: folded + vote, level-1 accuracy, prompt
  tokens and wall time of every route, and the `noul`'s present/absent AUC.

Each hypothesis is judged once. A failed one is reported as failed, and
the exploratory finding it came from is marked so.

## Limits registered in advance

One artifact; the questions were written by one author, after seeing the
windows but before any answer; LogHub windows are single-source and mostly
100K; the 6+ bin is 16 questions; H2's length and source are confounded
(LogHub tops out at 100K).
