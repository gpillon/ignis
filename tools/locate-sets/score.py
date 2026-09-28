"""Score spec 18 phase A's dumps and apply its pre-registered rules.

`crates/server/tests/attention_head_locate_gpu.rs` dumps, per question and per
prefill (`s1`, `s1-na`, `s2`, `s2-na`), every GQA head's pre-softmax scores at
the scaffold's last position over the state's key span, with each segment's
keys. From those this computes, per head:

- its **share** of each segment: the softmax over the span, summed over the
  segment's keys (mass on keys no segment owns belongs to none);
- the segment its **argmax key** falls in (none when it is a separator).

and reads a question three ways (spec 18, "The readings phase A chooses
between"):

- **R1**, one head: the segment with the largest share;
- **R2**, a head set voting: each head's argmax credits its segment, and the
  share is votes over heads;
- **R3**, K <= 32 heads: shares summed per segment (QRHead-style);

each with or without the **content-free baseline** -- the `-na` prefill's
reading of the same state, subtracted per segment before the winner is taken
(the confidence is then the clipped difference over its sum).

**Choosing, on the development sets only** (`dev`), by 5-fold cross-validation
over A+B's present questions, folds by a hash of the question id:

- R1 picks the head with the most training hits (ties: larger mean target share);
- R2 keeps every head whose **selectivity** -- its mass on the target segment
  over its mass on all segments, averaged over the training questions -- is at
  least half the most selective head's: spec 14's selectivity rule in text,
  made relative because text has tens to a thousand segments where an image
  had a few boxes (an absolute 0.5, the first version of this rule, kept no
  head at all on the first six development questions, whose best head sat at
  0.40);
- R3 takes the top K heads by **QRHead's score** (mean mass on the target) with
  K in {8, 16, 32}, K chosen by training hits;
- the pointing head L39.h10 is scored as one more R1 candidate, never chosen.

Rule 1 then takes the best top-1 among R1 and R2 (both scaffolds, with and
without the baseline), and R3 only if its best beats that by at least 3
points. Ties go to the cheaper and the simpler: no baseline, then R1, then S1.

**Judging, on the check set** (`check`), with the choice fitted on all of A+B:
rule 2 (go/no-go against the labelled `choice` route of `labelled.py`), rule 3
(`LOCATE_MAX_KEYS`) and rule 4 (the per-family floors), plus top-3 recall,
the present/absent AUC of `confidence`, and the costs.

Absent questions have no right segment: top-1 counts present questions only,
and absent ones enter only the AUC.

    python score.py dev --dumps A-hq.json B-hq.json --out dev.json
    python score.py check --choice dev.json --dumps C-hq.json --labelled C-labelled.json --out check.json

Needs NumPy.
"""

import argparse
import hashlib
import json
import math
import os

import numpy as np

LAYERS, HEADS = 16, 24
N_HEADS = LAYERS * HEADS
POINTING = 9 * HEADS + 10            # L39.h10: GQA ordinal 9, query head 10
VARIANTS = ("s1", "s1-na", "s2", "s2-na")
SELECTIVITY_OF_BEST = 0.5
R3_KS = (8, 16, 32)
R3_MARGIN = 3.0                      # rule 1, points of top-1
FOLDS = 5
GO_OVERALL, GO_PARAPHRASE = 5.0, 10.0    # rule 2, points
CEILING_POINTS = 5.0                 # rule 3, points
FLOOR_SLACK = 0.03                   # rule 4, of the family's question count
MAX_LABELLED = 256                   # the labelled route's measured ceiling
LONG = {"logs": {60: 0, 250: 1, 1000: 2}, "records": {20: 0, 80: 1, 300: 2}}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1)


def head_name(h):
    return f"L{4 * (h // HEADS) + 3}.h{h % HEADS}"


# ---------------------------------------------------------------------------
# The dump, as per-head shares and argmax segments
# ---------------------------------------------------------------------------

def features(scores, keys):
    """`scores` [N_HEADS, span] -> shares [N_HEADS, segments] (0 for a
    segment that owns no key) and argmax segment [N_HEADS] (-1: none)."""
    s = scores.astype(np.float64)
    top = s.max(axis=1, keepdims=True)
    e = np.exp(s - top)
    total = e.sum(axis=1)
    cum = np.concatenate([np.zeros((s.shape[0], 1)), np.cumsum(e, axis=1)], axis=1)
    shares = np.zeros((s.shape[0], len(keys)), dtype=np.float32)
    owner = np.full(s.shape[1], -1, dtype=np.int32)
    for j, k in enumerate(keys):
        if k is not None:
            shares[:, j] = (cum[:, k[1]] - cum[:, k[0]]) / total
            owner[k[0]:k[1]] = j
    return shares, owner[s.argmax(axis=1)]


def load(json_path, cache=True):
    """One dump as a list of question dicts with per-variant features,
    cached beside the dump (`.features.npz`) since reading the scores is the
    slow part."""
    meta = read_json(json_path)
    base = os.path.dirname(json_path)
    with open(os.path.join(base, meta["rows_file"]), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    cache_path = os.path.splitext(json_path)[0] + ".features.npz"
    if cache and os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(json_path):
        stored = np.load(cache_path, allow_pickle=True)
        feats = stored["feats"].tolist()
    else:
        raw = np.memmap(os.path.join(base, meta["bin_file"]), dtype="<f2", mode="r")
        feats = []
        for row in rows:
            span = row["span"][1]
            per = {}
            for v in VARIANTS:
                off = row["variants"][v]["offset"]
                block = np.asarray(raw[off:off + N_HEADS * span]).reshape(N_HEADS, span)
                per[v] = features(block, row["keys"])
            feats.append(per)
        if cache:
            np.savez(cache_path, feats=np.array(feats, dtype=object))
    for row, per in zip(rows, feats):
        row["feat"] = per
        row["set"] = meta["set"]
    return meta, rows


def tier(row):
    """The length tier: 0, 1, 2 for the three lengths of logs and records;
    prose (about 1K tokens) is tier 0."""
    return LONG.get(row["family"], {}).get(row["segments"], 0)


# ---------------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------------

def owning(row):
    return np.array([k is not None for k in row["keys"]])


def reading(row, method, scaffold, baseline):
    """Per-segment scores of one reading (method = ("r1", h) | ("r2", heads)
    | ("r3", heads) | ("vote", heads)), before normalisation."""
    kind, heads = method
    if kind == "vote":
        return vote_reading(row, heads, scaffold, baseline)

    def one(variant):
        shares, argseg = row["feat"][variant]
        if kind == "r1":
            return shares[heads].astype(np.float64)
        if kind == "r3":
            return shares[heads].sum(axis=0).astype(np.float64) / len(heads)
        votes = np.zeros(shares.shape[1])
        for h in heads:
            if argseg[h] >= 0:
                votes[argseg[h]] += 1
        return votes / max(1, len(heads))

    s = one(scaffold)
    if baseline:
        s = s - one(scaffold + "-na")
    s = np.where(owning(row), s, -np.inf)
    return s


def vote_reading(row, heads, scaffold, baseline):
    """Spec 19's head vote (phase 0, GitHub #276): each head of `heads`, best
    first, names its R1 winner -- the segment with its largest share, less
    the -na prefill's share with a baseline -- and a segment scores the votes
    it gets. A tie goes to the tied segment named by the best-ranked head: a
    voted segment's score is its votes less a millionth of its best voter's
    rank, too little to outweigh a whole vote. The confidence is then the
    winner's share of the votes."""
    shares = row["feat"][scaffold][0].astype(np.float64)
    if baseline:
        shares = shares - row["feat"][scaffold + "-na"][0]
    own = owning(row)
    s = np.where(own[None, :], shares[heads], -np.inf)
    votes = np.zeros(shares.shape[1])
    best = np.zeros(shares.shape[1])
    for rank, w in enumerate(s.argmax(axis=1)):
        if votes[w] == 0:
            best[w] = rank
        votes[w] += 1.0
    votes = np.where(votes > 0, votes - 1e-6 * best, 0.0)
    return np.where(own, votes, -np.inf)


def winner(s):
    """The first segment with the largest score (ties to the earlier)."""
    return int(np.argmax(s))


def confidence(s, baseline):
    """The winner's share: the score itself without a baseline, the clipped
    difference over its sum with one."""
    w = winner(s)
    if not baseline:
        return float(s[w])
    clipped = np.clip(np.where(np.isfinite(s), s, 0), 0, None)
    total = clipped.sum()
    return float(clipped[w] / total) if total > 0 else 0.0


def hit(row, s):
    return winner(s) in row["targets"]


def target_share(row, variant):
    shares, _ = row["feat"][variant]
    return shares[:, row["targets"]].sum(axis=1)


# ---------------------------------------------------------------------------
# Selection on training questions
# ---------------------------------------------------------------------------

def r1_hits(rows, scaffold, baseline):
    """[N_HEADS] training hits of every head as R1."""
    hits = np.zeros(N_HEADS)
    for row in rows:
        shares, _ = row["feat"][scaffold]
        s = shares.astype(np.float64)
        if baseline:
            s = s - row["feat"][scaffold + "-na"][0]
        s = np.where(owning(row)[None, :], s, -np.inf)
        w = s.argmax(axis=1)
        hits += np.isin(w, row["targets"])
    return hits


def select(rows, reading_kind, scaffold, baseline):
    """The method a reading uses, chosen on `rows` (present questions)."""
    if reading_kind == "r1":
        hits = r1_hits(rows, scaffold, baseline)
        mean = np.mean([target_share(r, scaffold) for r in rows], axis=0)
        order = np.lexsort((-mean, -hits))
        return ("r1", int(order[0]))
    if reading_kind == "l39":
        return ("r1", POINTING)
    if reading_kind == "r2":
        sel = []
        for row in rows:
            shares, _ = row["feat"][scaffold]
            on = shares.sum(axis=1)
            sel.append(np.where(on > 0, target_share(row, scaffold) / np.maximum(on, 1e-30), 0))
        selectivity = np.mean(sel, axis=0)
        bar = SELECTIVITY_OF_BEST * selectivity.max()
        return ("r2", [int(h) for h in np.flatnonzero(selectivity >= bar)])
    if reading_kind == "r3":
        score = np.mean([target_share(r, scaffold) for r in rows], axis=0)
        ranked = [int(h) for h in np.argsort(-score, kind="stable")]
        best = None
        for k in R3_KS:
            method = ("r3", ranked[:k])
            n = sum(hit(r, reading(r, method, scaffold, baseline)) for r in rows)
            if best is None or n > best[0]:
                best = (n, method)
        return best[1]
    raise ValueError(reading_kind)


def fold_of(row):
    return int(hashlib.sha256(row["id"].encode() + row["set"].encode()).hexdigest(), 16) % FOLDS


CONFIGS = [(k, s, b) for k in ("r1", "r2", "r3", "l39") for s in ("s1", "s2") for b in (False, True)]


def config_name(kind, scaffold, baseline):
    return f"{kind.upper()} {scaffold}{' +base' if baseline else ''}"


def cross_validate(rows):
    present = [r for r in rows if not r["absent"]]
    out = {}
    for kind, scaffold, baseline in CONFIGS:
        hits, per_fold = 0, []
        by = {}
        for fold in range(FOLDS):
            train = [r for r in present if fold_of(r) != fold]
            test = [r for r in present if fold_of(r) == fold]
            method = select(train, kind, scaffold, baseline)
            if method[0] != "r1" and not method[1]:
                n, got = 0, [False] * len(test)
            else:
                got = [hit(r, reading(r, method, scaffold, baseline)) for r in test]
                n = sum(got)
            hits += n
            per_fold.append({"method": describe(method), "hits": n, "of": len(test)})
            for r, g in zip(test, got):
                for key in (r["family"], r["split"], f"tier{tier(r)}"):
                    by.setdefault(key, [0, 0])
                    by[key][0] += g
                    by[key][1] += 1
        out[config_name(kind, scaffold, baseline)] = {
            "kind": kind, "scaffold": scaffold, "baseline": baseline,
            "hits": hits, "of": len(present), "top1": 100.0 * hits / len(present),
            "by": {k: {"hits": v[0], "of": v[1]} for k, v in sorted(by.items())},
            "folds": per_fold,
        }
    return out


def describe(method):
    kind, heads = method
    if kind == "r1":
        return {"reading": "r1", "head": heads, "name": head_name(heads)}
    return {"reading": kind, "heads": heads, "names": [head_name(h) for h in heads]}


def rule1(cv):
    """The reading, scaffold and baseline rule 1 chooses, and why."""
    def key(item):
        name, c = item
        return (-c["top1"], c["baseline"], c["kind"] != "r1", c["scaffold"] != "s1")
    r12 = sorted([(n, c) for n, c in cv.items() if c["kind"] in ("r1", "r2")], key=key)
    r3 = sorted([(n, c) for n, c in cv.items() if c["kind"] == "r3"], key=key)
    best12, best3 = r12[0], r3[0]
    if best3[1]["top1"] >= best12[1]["top1"] + R3_MARGIN:
        return best3, f"R3 beats the best of R1/R2 by {best3[1]['top1'] - best12[1]['top1']:.1f} points (>= {R3_MARGIN})"
    return best12, (f"best of R1/R2; R3's best ({best3[0]}, {best3[1]['top1']:.1f}) is "
                    f"{best3[1]['top1'] - best12[1]['top1']:+.1f} points, under the {R3_MARGIN}-point bar")


# ---------------------------------------------------------------------------
# The check set
# ---------------------------------------------------------------------------

def auc(pos, neg):
    """P(a present question's confidence > an absent one's), ties half."""
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def floor_of(hits, n):
    """Rule 4's floor for a family: its check-set hits less a slack of 3% of
    its question count, and the fewest hits that clear it."""
    slack = FLOOR_SLACK * n
    return {"slack": slack, "floor": hits - slack, "at_least": math.ceil(hits - slack - 1e-9)}


def pct(h, n):
    return 100.0 * h / n if n else float("nan")


def check(choice, rows, labelled):
    kind, scaffold, baseline = choice["kind"], choice["scaffold"], choice["baseline"]
    method = (choice["method"]["reading"], choice["method"].get("head", choice["method"].get("heads")))
    per = []
    for r in rows:
        s = reading(r, method, scaffold, baseline)
        order = [int(j) for j in np.argsort(-np.where(np.isfinite(s), s, -1e30), kind="stable")]
        per.append({
            "id": r["id"], "family": r["family"], "split": r["split"], "absent": r["absent"],
            "tier": tier(r), "owning": int(owning(r).sum()), "span": r["span"][1],
            "winner": order[0], "top3": order[:3], "targets": r["targets"],
            "hit": (not r["absent"]) and order[0] in r["targets"],
            "hit3": (not r["absent"]) and bool(set(order[:3]) & set(r["targets"])),
            "confidence": confidence(s, baseline),
            "l39": {v: r["l39_h10"][v] for v in ("s1", "s2")},
            "prompt_tokens": {v: r["variants"][v]["prompt_tokens"] for v in VARIANTS},
            "suffix_tokens": r["variants"][scaffold + "-na"]["prompt_tokens"] - (r["span"][0] + r["span"][1]),
        })
    present = [p for p in per if not p["absent"]]
    lab = {q["id"]: q for q in labelled["questions"]} if labelled else {}

    # Rule 2: the reading against the labelled route, on the questions both answer.
    both = [p for p in present if p["owning"] <= MAX_LABELLED and p["id"] in lab]
    def compare(subset):
        h_read = sum(p["hit"] for p in subset)
        h_lab = sum(lab[p["id"]]["hit"] for p in subset)
        return {"n": len(subset), "reading": h_read, "labelled": h_lab,
                "reading_top1": pct(h_read, len(subset)), "labelled_top1": pct(h_lab, len(subset)),
                "gap": pct(h_lab, len(subset)) - pct(h_read, len(subset))}
    overall = compare(both)
    paraphrase = compare([p for p in both if p["split"] == "paraphrase"])
    families = {f: compare([p for p in both if p["family"] == f]) for f in ("logs", "records", "prose")}
    go = labelled is not None and overall["gap"] <= GO_OVERALL and paraphrase["gap"] <= GO_PARAPHRASE

    # Rule 3: the length ceiling.
    tiers = {}
    for t in (0, 1, 2):
        sub = [p for p in present if p["tier"] == t]
        tiers[t] = {"n": len(sub), "hits": sum(p["hit"] for p in sub), "top1": pct(sum(p["hit"] for p in sub), len(sub)),
                    "max_span": max((p["span"] for p in per if p["tier"] == t), default=0)}
    ceiling_tier = 0
    for t in (1, 2):
        if tiers[t]["n"] and tiers[t]["top1"] >= tiers[0]["top1"] - CEILING_POINTS:
            ceiling_tier = t
    max_keys = tiers[ceiling_tier]["max_span"]

    # Rule 4: the per-family floors, on the questions a locate would serve.
    floors = {}
    for f in ("logs", "records", "prose"):
        sub = [p for p in present if p["family"] == f and p["span"] <= max_keys]
        h = sum(p["hit"] for p in sub)
        floors[f] = {"n": len(sub), "hits": h, **floor_of(h, len(sub))}

    report = {
        "choice": choice,
        "top1": {"hits": sum(p["hit"] for p in present), "of": len(present),
                 "pct": pct(sum(p["hit"] for p in present), len(present))},
        "top3": {"hits": sum(p["hit3"] for p in present), "of": len(present),
                 "pct": pct(sum(p["hit3"] for p in present), len(present))},
        "by": {}, "rule2": {"overall": overall, "paraphrase": paraphrase, "families": families,
                            "go": go, "bars": [GO_OVERALL, GO_PARAPHRASE]},
        "rule3": {"tiers": tiers, "ceiling_tier": ceiling_tier, "LOCATE_MAX_KEYS": max_keys},
        "rule4": floors,
        "auc_present_absent": auc([p["confidence"] for p in present], [p["confidence"] for p in per if p["absent"]]),
        "confidence_median": {
            "hit": float(np.median([p["confidence"] for p in present if p["hit"]] or [np.nan])),
            "miss": float(np.median([p["confidence"] for p in present if not p["hit"]] or [np.nan])),
            "absent": float(np.median([p["confidence"] for p in per if p["absent"]] or [np.nan])),
        },
        "l39_h10": {v: sum(1 for p, r in zip(per, rows) if not r["absent"] and p["l39"][v] in r["targets"])
                    for v in ("s1", "s2")},
        "baseline_suffix_tokens_median": float(np.median([p["suffix_tokens"] for p in per])),
        "questions": per,
    }
    for key in ("family", "split"):
        for value in sorted({p[key] for p in present}):
            sub = [p for p in present if p[key] == value]
            report["by"][value] = {"hits": sum(p["hit"] for p in sub), "of": len(sub),
                                   "pct": pct(sum(p["hit"] for p in sub), len(sub))}
    if labelled:
        report["labelled"] = {k: v for k, v in labelled.items() if k != "questions"}
        timed = [lab[p["id"]] for p in both]
        report["wall_ms_median"] = {
            "labelled_choice": float(np.median([q["choice_ms"] for q in timed])) if timed else None,
            "noul_same_state": float(np.median([q["noul_ms"] for q in timed])) if timed else None,
        }
    return report


# ---------------------------------------------------------------------------
# Golden cases for the served reading (GitHub #275)
# ---------------------------------------------------------------------------

def f16_hex(block):
    """`block`'s scores as f16 little-endian bytes, hex: what the Rust port
    reads back exactly (every f16 is exact in f32)."""
    return np.ascontiguousarray(block, dtype="<f2").tobytes().hex()


def vote_case(name, heads, keys, q, na):
    """One golden case of the head vote: `q` and `na` are the vote's heads'
    [K, span] scores (best first) at the copy scaffold and at its content-free
    prefill, `keys` each segment's [a, b) or None. Out: the segment each head
    votes for, the votes, the winner, its share of the votes, and the ranking
    -- the voted segments by votes, ties to the best-ranked voter, at most
    five -- all from `features` and `vote_reading`, the arithmetic every
    number in the findings was measured with."""
    q, na = np.asarray(q, dtype="<f2"), np.asarray(na, dtype="<f2")
    K = q.shape[0]
    row = {"keys": keys, "feat": {"s2": features(q, keys), "s2-na": features(na, keys)}}
    s = vote_reading(row, list(range(K)), "s2", True)
    diff = row["feat"]["s2"][0].astype(np.float64) - row["feat"]["s2-na"][0]
    own = owning(row)
    voted = [int(v) for v in np.where(own[None, :], diff, -np.inf).argmax(axis=1)]
    votes = [int(v) for v in np.bincount(voted, minlength=len(keys))]
    order = [int(j) for j in np.argsort(-np.where(np.isfinite(s), s, -1e30), kind="stable")]
    ranking = [[j, votes[j] / K] for j in order if votes[j] > 0][:5]
    w = winner(s)
    return {"name": name, "heads": heads, "span": int(q.shape[1]),
            "keys": [None if k is None else [int(k[0]), int(k[1])] for k in keys],
            "q": f16_hex(q), "na": f16_hex(na), "voted": voted, "votes": votes, "winner": w,
            "confidence": votes[w] / K, "reference_confidence": confidence(s, True), "ranking": ranking}


def synthetic_cases(seed):
    """Small cases on the rule's edges, planted on f16 noise: a clear
    majority, a two-way tie the best-ranked voter breaks, a flat row, a
    baseline sharper than the question (every difference negative), mass on
    the separators alone, and segments that own no key."""
    rng = np.random.default_rng(seed)
    cases = []

    def layout(widths, gaps=1, empty=()):
        keys, at = [], 0
        for j, width in enumerate(widths):
            if j in empty:
                keys.append(None)
                continue
            keys.append([at, at + width])
            at += width + gaps
        return keys, at - gaps

    def noise(k, span):
        return rng.normal(0, 1, size=(k, span))

    def plant(block, head, key_range, lift):
        a, b = key_range
        block[head, a:b] += lift

    heads5 = ["L39.h12", "L47.h20", "L59.h16", "L55.h17", "L59.h17"]
    keys, span = layout([3, 4, 2, 5, 3, 4, 2, 3])
    q, na = noise(5, span), noise(5, span)
    for h in (0, 1, 2):
        plant(q, h, keys[3], 9.0)
    cases.append(vote_case("synthetic/majority", heads5, keys, q, na))

    keys, span = layout([4, 4, 4, 4, 4, 4])
    q, na = noise(5, span), noise(5, span)
    for h, j in ((0, 4), (3, 4), (1, 1), (2, 1), (4, 2)):
        plant(q, h, keys[j], 10.0)
    cases.append(vote_case("synthetic/tie-to-the-best-ranked-voter", heads5, keys, q, na))

    keys, span = layout([2, 6, 3, 1, 4])
    q, na = noise(3, span), noise(3, span)
    q[1, :] = 0.5
    na[1, :] = 0.5
    cases.append(vote_case("synthetic/flat-row", heads5[:3], keys, q, na))

    keys, span = layout([3, 3, 3, 3, 3])
    q, na = noise(3, span), noise(3, span)
    for h in range(3):
        plant(q, h, keys[2], 4.0)
        plant(na, h, keys[2], 7.0)
    cases.append(vote_case("synthetic/baseline-sharper-than-the-question", heads5[:3], keys, q, na))

    keys, span = layout([2, 2, 2, 2], gaps=3)
    q, na = noise(3, span), noise(3, span)
    separators = [k for k in range(span) if not any(r and r[0] <= k < r[1] for r in keys)]
    q[:, separators] += 12.0
    cases.append(vote_case("synthetic/mass-on-the-separators", heads5[:3], keys, q, na))

    keys, span = layout([3, 0, 4, 2, 0, 3], empty=(1, 4))
    q, na = noise(5, span), noise(5, span)
    for h in range(5):
        plant(q, h, keys[5], 6.0 + h)
    cases.append(vote_case("synthetic/segments-that-own-no-key", heads5, keys, q, na))
    return cases


def golden(args):
    """Golden cases for the served vote (`crates/core/tests/locate_reading.rs`):
    real questions of a dump, read with the registered heads, and the
    synthetic edges."""
    choice = read_json(args.choice)["choice"]
    heads, names = choice["method"]["heads"], choice["method"]["names"]
    meta = read_json(args.dump)
    base = os.path.dirname(args.dump)
    with open(os.path.join(base, meta["rows_file"]), encoding="utf-8") as f:
        rows = {r["id"]: r for r in (json.loads(line) for line in f)}
    raw = np.memmap(os.path.join(base, meta["bin_file"]), dtype="<f2", mode="r")
    cases = []
    for qid in args.ids:
        row = rows[qid]
        span = row["span"][1]

        def block(variant):
            off = row["variants"][variant]["offset"]
            return np.asarray(raw[off:off + N_HEADS * span]).reshape(N_HEADS, span)[heads]

        keys = [None if k is None else tuple(k) for k in row["keys"]]
        cases.append(vote_case(f"{meta['set']}/{qid}", names, keys, block("s2"), block("s2-na")))
    cases += synthetic_cases(args.seed)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        json.dump({"source": "tools/locate-sets/score.py golden", "choice": choice["config"], "cases": cases}, f)
    print(f"{len(cases)} golden cases -> {args.out}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev", help="cross-validate every reading on the development dumps, apply rule 1")
    d.add_argument("--dumps", nargs="+", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("check", help="judge the chosen reading on the check dump, apply rules 2-4")
    c.add_argument("--choice", required=True, help="the dev report")
    c.add_argument("--dumps", nargs="+", required=True)
    c.add_argument("--labelled", help="labelled.py's results on the same set")
    c.add_argument("--out", required=True)
    g = sub.add_parser("golden", help="golden cases of the head vote for the served port")
    g.add_argument("--choice", required=True, help="the vote's choice file (vote-choice.json)")
    g.add_argument("--dump", required=True, help="a harness dump's .json")
    g.add_argument("--ids", nargs="+", required=True, help="the dump's questions to write")
    g.add_argument("--seed", type=int, default=20261027, help="the synthetic cases' noise")
    g.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "golden":
        golden(args)
        return

    if args.cmd == "dev":
        rows = []
        for path in args.dumps:
            rows += load(path)[1]
        cv = cross_validate(rows)
        (name, best), why = rule1(cv)
        present = [r for r in rows if not r["absent"]]
        method = select(present, best["kind"], best["scaffold"], best["baseline"])
        choice = {"config": name, "kind": best["kind"], "scaffold": best["scaffold"],
                  "baseline": best["baseline"], "cv_top1": best["top1"], "why": why,
                  "method": describe(method), "fitted_on": sorted({r["set"] for r in rows})}
        for n, c in sorted(cv.items(), key=lambda kv: -kv[1]["top1"]):
            by = "  ".join(f"{k} {v['hits']}/{v['of']}" for k, v in c["by"].items())
            print(f"{n:16} {c['hits']:4}/{c['of']} = {c['top1']:5.1f}   {by}")
        print(f"\nrule 1: {name} -- {why}")
        print(f"fitted on A+B: {choice['method']}")
        write_json(args.out, {"cv": cv, "choice": choice})
        return

    choice = read_json(args.choice)["choice"]
    rows = []
    for path in args.dumps:
        rows += load(path)[1]
    labelled = read_json(args.labelled) if args.labelled else None
    report = check(choice, rows, labelled)
    write_json(args.out, report)
    r2 = report["rule2"]
    print(f"chosen: {choice['config']}  ({choice['method'].get('name') or len(choice['method'].get('heads', []))})")
    print(f"C top-1 {report['top1']['hits']}/{report['top1']['of']} = {report['top1']['pct']:.1f}  "
          f"top-3 {report['top3']['pct']:.1f}   by {json.dumps(report['by'])}")
    for label, block in (("overall", r2["overall"]), ("paraphrase", r2["paraphrase"])):
        print(f"rule 2 {label:10}: reading {block['reading']}/{block['n']} ({block['reading_top1']:.1f}) "
              f"labelled {block['labelled']}/{block['n']} ({block['labelled_top1']:.1f}) gap {block['gap']:+.1f}")
    print(f"rule 2: {'GO' if r2['go'] else 'NO-GO'}")
    print(f"rule 3: tiers {json.dumps(report['rule3']['tiers'])} -> LOCATE_MAX_KEYS {report['rule3']['LOCATE_MAX_KEYS']}")
    print(f"rule 4: {json.dumps(report['rule4'])}")
    print(f"AUC present/absent {report['auc_present_absent']}; confidence medians {report['confidence_median']}")
    print(f"L39.h10 as R1 on C: {report['l39_h10']}; wall {report.get('wall_ms_median')}")


if __name__ == "__main__":
    main()
