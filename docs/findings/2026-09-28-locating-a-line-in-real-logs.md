# Locating a line in real logs: near-duplicates break one pass, copying resolves the instance, folding makes it cheap

- Kind: experiment
- Status: current
- Observed: 2026-09-28
- Last verified: 2026-09-28
- Scope: `/v1/decide` `locate` and the generation route over real logs (one production cluster, public LogHub samples), 16K-200K tokens; spec 21's registered hypotheses on a fresh set; the folded method (owner's idea)
- Related: [spec 21](../specs/decide/21-locate-in-real-logs.md) (registered in `e17c5e8`, judge `c5d3386`, both before R2 was asked); [the instance is read while copying](2026-09-27-the-instance-is-read-while-copying.md) (the exploratory results judged here); [locate at length](2026-09-27-locate-at-length.md); [`compress.py`](../../tools/locate-sets/compress.py), [`folded_locate.py`](../../tools/locate-sets/folded_locate.py), [`r2_judge.py`](../../tools/locate-sets/r2_judge.py)
- Superseded by: none

## Question

The exploratory work on set R said three things about finding a line in a
real log: one pass of attention finds the *kind* of line and loses the
instance among near-duplicates, more so with length; the row's share of
the state is fixed, so each line's share falls; the instance is resolved
while the line is copied. And a method tuned on R — fold near-duplicates
into templates, locate the template, then the instance among its values,
unfold — read R nearly as well as generating on the whole log, at a
fraction of the cost. Do they hold on a set nobody had asked?

## Evidence

- **Set R2** (spec 21): 33 windows — 8 of the cluster's last six hours
  (16K / 50K / 100K / 200K tokens, later than anything R held) and 15 public
  LogHub systems (50K-100K) — with targets drawn by rule across sibling bins;
  68 questions (58 present, 10 absent), 9 drawn targets dropped for
  reasons recorded before any run. One server from branch
  `locate-long-context`; three runs (spec 21 § Runs); `r2_judge.py` applied
  once.

| spec 21 | rule | R2 | |
|---|---|---|---|
| H1 near-duplicates | miss rate, 6+ siblings minus none, ≥ 20 points | 81% - 61% = 20.4 | pass (at the bar) |
| H2 one pass at length | heads on the target fall with length, rho < 0, p < 0.05 | rho -0.29, p 0.012 | pass |
| H3 the state's share | shortest vs longest tier within 5 points | 0.51 / 0.49 / 0.49 / 0.52; 1.5 | pass |
| H4a the copy | misses' plurality on the target, k = 32 minus k = 1, ≥ 20 points | 0.03 → 0.50 | pass |
| H4b the copy and uniqueness | first settled k vs unique-prefix length, rho > 0, p < 0.05 | rho 0.59, p 4e-7 | pass |
| M1 folded + generation | within 5 points of generation on the whole log | 94.8% vs 86.2% | pass |
| M2 folded + generation | at least 10 points over the served vote | 94.8% vs 34.5% | pass |

- **Top-1 over 58 present questions**: the served vote 20, folded + vote
  28, generation on the whole log 50, **folded + generation 55**. By source
  and tier the folded generation is never below the whole-log generation
  (cluster 200K 6/6 vs 5/6; LogHub 100K 26/29 vs 22/29); its three misses are
  all at level 1 (a template not chosen).
- **Cost**: level 1 is a median 7.4x shorter than the window (1.3x to 51x;
  ~10K tokens for a 100K window). A whole-log question pays its window's
  prefill first (median 47 s here, 139 s at 200K on R), then ~2.4 s per
  question on the retained prefix; folded + generation answers in a median
  1.3 s (p90 3.9 s) with no long prefill at all.
- **Found**: the sibling `noul`'s present/absent AUC is 0.79 (R: 0.93;
  10 absent questions each).

## Finding

On a fresh set of real logs, cluster and public, every mechanism the
exploratory work proposed holds as registered: a one-pass reading of
attention loses the target among its near-duplicates (81% misses with six
or more, against 61% with none — itself high, because length alone costs
the one pass too), and loses it more the longer the log; half of the
attention row stays on the log at every length, so each line's share falls;
and the instance is resolved while the line is copied — the heads settle
on the target at the point where its written prefix stops matching any
other line (rho 0.59). Folding the log into templates and their values, and
letting the model copy first the template and then the instance, finds the
line 94.8% of the time against 86.2% for generating on the whole log and
34.5% for the served vote, in about a second instead of a long prefill.

## Implications

- **For the product**: a `locate` for logs is fold, then copy under
  constraint, then unfold — two short generations (a constrained decode
  could stop each at its first unique prefix) over texts ~7x shorter,
  answered as the original line's index. The one-pass vote stays where its
  measured ceiling puts it (≤ 4.5K keys).
- **For the paper**: near-duplicate density, not length alone, names what
  one-pass attention cannot do; copying is the mechanism that resolves it,
  measurably tied to prefix uniqueness; and a reversible, content-aware
  compression turns that into a method that beats generation on the whole
  context.
- The `found` signal needs its own work: 0.79-0.93 AUC on 10 absent
  questions per set is not yet a flag.

## Limits and unknowns

- One model (the served NVFP4 27B, hq-e8-2b keys); one cluster plus 15
  LogHub samples; 58 present questions, written by one author who saw the
  windows (not the answers); H1 passed at its bar exactly.
- The folded method was tuned on R (value summaries, values-first rows,
  quote matching) and judged once here; SIM = 0.5 and the value budget were
  not varied on R2.
- LogHub windows are single-source and mostly 100K; the 200K tier is the
  cluster's only (6 questions).
- The generation routes match their quote to a line by exact, contained or
  word-overlap search, not a constrained decode.
- Questions needing context across lines ("the error after the deploy")
  were not asked; folding removes that context by design.

## Follow-ups

- A spec for `locate` by folding and constrained copy, with its acceptance
  on a fresh set.
- `found`: a larger absent set, and the `noul` on the folded text.
