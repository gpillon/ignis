# Spec 20 — locate at length (research)

Status: **RUN** (registered 2026-09-27 in `78e061a` before sets L2 and R were asked; judged the same day,
[finding](../../findings/2026-09-27-locate-at-length.md)): rule 1 passes (end 122 vs served 119), the
ceiling is 53K keys on L2 and 4K on R (candidate 3,995: no lift), generation reads 99/100 and 46/50, the
`noul` separates absent questions at AUC 0.999 / 0.93.
Branch `locate-long-context` (experiment; the two env overrides below are
never merged). Owner's request: `locate` refuses past `LOCATE_MAX_KEYS`
(4,554 keys), while the logs it is meant for are far longer.

## What is already known (set L, read, spent)

Set L (`tools/locate-sets/long.py generate --seed 20261030`): synthetic logs
of 250 / 1,000 / 3,000 / 6,000 / 12,000 lines (4.5K-212K keys), 20 present
and 4 absent questions per length, depth stratified. Served through
`/v1/decide` with the ceiling lifted:

- the served vote reads 19 / 15 / 18 / 19 / 16 of 20; the generation route
  (the same render, greedy) reads 20 of 20 at every length;
- 12 of the vote's 13 misses are the **line after** the target;
- on development logs (A+B) seven of the 32 heads vote the line after more
  often than the target (L47.h17 4% / 74%, L51.h12 6% / 87%, L43.h9 11% /
  80%, ...), and peak on the separator or the next line's first keys; on
  records and prose the same heads vote the target ~80%. They mark **where
  the target ends** (the owner's reading, 2026-09-27).

Offline, on the dumps: reading those seven heads as end markers (below)
gives A 185, B 192, C 185, D 190 (served 180, 187, 184, 185) and L 97/100
(served 87). Chosen on A+B only; C, D and L are replications, not checks.

## The readings (registered)

`tools/locate-sets/long_judge.py`, on the rows of the 32 served heads:

- **end** (primary): the seven end-marker heads (`ENDERS`) each name the
  segment that ends just before their peak key (argmax of the per-key lift);
  a peak on a separator or within the first `M = 5` keys of a segment names
  the previous owned segment. The other 25 vote as served; plurality wins.
- **snap** (secondary): an end-marker head's vote for `s` moves to `s - 1`
  when the other heads gave `s - 1` more votes than `s`.
- **served** (reference) and the **generation route** (comparator).

## The sets

- **L2**: `long.py generate --seed 20261032` — L's design, fresh. Manifest
  sha256 `957ecc4c4293e8a35df33da40cd5ed4db45d27f5f6720df7d01ac14c2716745c`.
- **R**: real logs. `kubectl logs --timestamps --tail=4000` of the owner's
  cluster's running pods (read only), merged into one time-ordered stream
  with each line prefixed by its pod's short name (`prodset.py timeline`);
  three windows per tier of 4K / 16K / 50K / 100K / 200K tokens
  (`prodset.py windows --seed 20261031`); 10 present + 2 absent hand-written
  questions per tier, each checked by `prodset.py build`: lexical (a rare
  word), paraphrase (no shared content word) and **combo** (only common
  words, whose combination no other line holds). **Never committed** (it
  holds the cluster's data); manifest sha256
  `e08a1be09181d5243f5bab8d0d33d54fd6649649da6b874fdb65bd6390e55a43`.

## The run

A server built from this branch, `make start` with
`IGNIS_LOCATE_MAX_KEYS_EXPERIMENT=262144` and `IGNIS_LOCATE_DUMP_DIR`, then
`long.py ask --found` per set: each question a `locate` plus a sibling `noul`
("Is there a line in the evidence that answers this question: …") in one
request, then the generation route.

## Rules

1. **Reading**: `end`'s top-1 over the present questions of L2 and R
   together is at least the served vote's. Otherwise the served vote stays.
2. **Ceiling**, per set: tiers in length order; a tier passes when `end`'s
   top-1 is within 10 points of the shortest tier's (L2: two questions of
   20, R: one of 10); the ceiling is the longest tier with it and every
   shorter tier passing. The `LOCATE_MAX_KEYS` candidate is the smaller of
   the two sets' ceiling-tier spans. Small sets: a candidate, not a table
   entry, until the owner decides.
3. **Comparator**: the generation route's top-1 per tier, reported.
4. **Found** (exploratory, no bar): the present/absent AUC of the `noul`'s
   P(yes), the served vote's agreement and `end`'s agreement.

## Limits registered in advance

Logs only (records and prose past 18K keys are unmeasured); one artifact;
R is small (10 per tier) and from one cluster; the `noul` wording is one
unmeasured prompt.
