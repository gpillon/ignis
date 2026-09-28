# Locate at length: the vote holds on synthetic logs, not on real ones

- Kind: experiment
- Status: current
- Observed: 2026-09-27
- Last verified: 2026-09-27
- Scope: `/v1/decide` `locate` past `LOCATE_MAX_KEYS` — the served head vote, an end-marker reading of it, the generation route and a sibling `noul`, on synthetic logs to 212K keys and real cluster logs to 200K tokens
- Related: [spec 20](../specs/decide/20-locate-at-length.md) (registered in commit `78e061a` before L2 and R were asked); [locate through `/v1/decide`](2026-09-27-locate-through-decide.md); [the go on set D](2026-09-27-locate-by-head-vote-go.md); [attention names the line, not the words](2026-09-27-attention-names-the-line-not-the-words.md); [`tools/locate-sets/long_judge.py`](../../tools/locate-sets/long_judge.py)
- Superseded by: none

## Question

`locate` refuses a target past 4,554 keys, the length set D's rule 3 fixed.
The logs it is meant for are far longer. Does the vote hold there, what do
its misses look like, and what reads a line at that length?

## Evidence

- **Runs**: a server built from branch `locate-long-context` with the
  ceiling lifted and both prefills' rows of the 32 served heads dumped
  (two env overrides, not merged), `make start` defaults otherwise
  (hq-e8-2b, chunk 1,024, DFlash2). Each question: a `locate` (plus, on L2
  and R, a sibling `noul`, "Is there a line in the evidence that answers
  this question: …"), then the same render through
  `/v1/chat/completions`, greedy, thinking off (the generation route).
- **Set L** (seed 20261030, read before anything was registered): synthetic
  logs (`logs.py`'s events, 3-8 ERROR distractors) of 250 / 1,000 / 3,000 /
  6,000 / 12,000 lines (4.5K-212K keys), 20 present + 4 absent per length,
  depth stratified.
- **Set L2** (seed 20261032): L's design, fresh.
- **Set R**: real logs — `kubectl logs` of 147 running pods of one
  production cluster, merged by time with each line prefixed by its pod's
  short name, cut into three windows per tier of 4K / 16K / 50K / 100K /
  200K tokens; 60 hand-written questions (10 present + 2 absent per tier;
  25 paraphrase, 19 lexical, 6 **combo** — only common words, whose
  combination no other line holds). Kept outside the repository.

| top-1 of present questions | served vote | end reading | generation |
|---|---|---|---|
| L, 4.5K / 18K / 53K / 106K / 212K | 19 / 15 / 18 / 19 / 16 | 20 / 20 / 20 / 20 / 17 | 20 each |
| L2, same tiers | 19 / 19 / 18 / 17 / 18 (91) | 20 / 19 / 19 / 17 / 19 (94) | 99/100 |
| R, 4K / 16K / 50K / 100K / 200K | 8 / 7 / 5 / 4 / 4 (28/50) | 8 / 6 / 5 / 5 / 4 (28) | 9 / 10 / 10 / 9 / 8 (46/50) |

- **The misses on synthetic logs are the line after the target**: 12 of
  L's 13. On development logs (A+B) seven of the 32 heads vote the line
  after more than the target (L47.h17 4% / 74%, L51.h12 6% / 87%, L43.h9
  11% / 80%, L47.h15, L51.h6, L39.h15, L39.h23) and peak on the separator or
  the next line's first keys; on records and prose the same heads vote the
  target ~80%. They mark where the target **ends**. Among all 384 heads the
  best end markers are outside the vote (L47.h3 peaks on the target's last
  two keys or the five after it on 75% of A+B logs, 77% of records, 56% of
  prose); no head marks the first keys (4% at best).
- **The end reading** (those seven heads name the segment ending just
  before their peak): A-D 185 / 192 / 185 / 190 against 180 / 187 / 184 /
  185, L 97 against 87, L2 94 against 91, R 28 against 28. Spec 20's rule 1
  passes, 122 against 119.
- **The misses on real logs are another instance of the same kind of line**:
  of the vote's 22 misses on R, 14 land more than five lines away, almost
  all on a line of the target's own template — the same exporter's other
  write failure, "1 decision added" for "2 decisions added", the same socket
  error of another session, the same controller's stats line for another
  application — and 5 more on the line after. The six combo questions of
  the 200K tier's reconciliation storm are all missed, at a confidence of
  0.12-0.19. Top-3 38/50, top-5 43/50.
- **Ceiling** (spec 20's rule 2, the end reading within 10 points of the
  shortest tier): L2 to 53K keys, R to 4K. Candidate `LOCATE_MAX_KEYS`
  3,995.
- **Found**: the `noul`'s P(yes) separates present from absent questions at
  an AUC of **0.999** on L2 (80 / 20) and **0.93** on R (50 / 10), against
  the vote's agreement 0.91 / 0.72 and the end reading's 0.95 / 0.75. On
  absent questions the vote and the generation route both answer the
  nearest distractor.
- **Cost**: the first question over a state pays its prefill (R: 139 s at
  200K tokens; L2's 212K keys 147 s with the sibling `noul`, whose
  L0 prompt does not share the `locate`'s L1 prefix: 26 s at 106K without
  it on L, 50 s with it on L2); later questions over the retained state take
  0.2-4.4 s (R medians). The generation route writes a whole line (median
  90 tokens on R) in 0.4-1.9 s; the prefix that already names one line of
  its window is a median 26 tokens (p90 46).

## Finding

On synthetic logs the head vote holds to 212K keys — 91 of L2's 100 — and
its misses are the line after the target, which seven heads that mark the
target's end cause; reading them as end markers recovers most (94 on L2, 97
on L). On real cluster logs it does not: 56% over 4K-200K, falling past 4K,
because at the copy scaffold the heads find the right *kind* of line and
not the instance among near-duplicates — which real logs are made of and
the synthetic sets were not. Generating the line does both (92% on R, 99%
on L2) at every length measured, and a `noul` asking whether the line is
there at all separates absent questions far better (AUC 0.93 on R) than any
agreement of the vote. `LOCATE_MAX_KEYS` measured on real logs stays at the
4K the synthetic set D had set.

## Implications

- Do not lift `LOCATE_MAX_KEYS` for the vote: real logs put its ceiling
  where set D did.
- The route that reads a line in a long real log is generation — copying
  the line out, which a decode constrained to the state's lines could stop
  at the first unique prefix (median 26 tokens on R) and answer as an index.
- A `found` signal is in reach through a sibling `noul`; it needs the
  `locate`'s L1 prefix to avoid a second full prefill at length.
- The end-marker reading is a small, safe gain for the vote (spec 20 rule 1)
  where the vote is served (≤ 4.5K keys).

## Limits and unknowns

- R is small (10 present per tier) and from one cluster; its questions
  were written after reading the windows, by one author.
- L and L2 are logs only; records and prose past 18K keys are unmeasured.
- The end-marker heads and `M = 5` were chosen on A+B; the reading's gain
  is 3 questions of 150 on the fresh sets.
- The unique-prefix length is computed from the target's tokens, not from
  a constrained decode; a constrained decode was not built.
- One `noul` wording; the absent sets are 20 (L2) and 10 (R).

## Follow-ups

- A spec for `locate` by constrained quote, with its own acceptance on a
  fresh real-log set, and a `found` beside it.
