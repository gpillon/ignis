"""EXPERIMENT (branch locate-long-context, never merged): read the rows the
server dumped (`IGNIS_LOCATE_DUMP_DIR`) for `long.py`'s questions, offline.

Each dump is `<id>.bin` -- f32, the question's rows then the content-free
baseline's, `[32 heads][span]` each, in the calibrated heads' order -- and
`<id>.json` (span, each segment's key range, the served winner and votes).

What it reports, per length:

- the served vote re-read from the rows (a check that the dump is the answer);
- where each head's vote lands against the target (0, +1, -1, near, far);
- the target's votes against the best other segment's;
- a few other readings of the same rows (exploratory, chosen on this set).

    python long_rows.py --set <L> --served L-served.json --dumps <dir> --out L-rows.json
"""

import argparse
import json
import os
from collections import Counter

import numpy as np


def load(dumps, qid):
    with open(os.path.join(dumps, qid + ".json"), encoding="utf-8") as f:
        meta = json.load(f)
    raw = np.fromfile(os.path.join(dumps, qid + ".bin"), dtype="<f4")
    heads, span = meta["heads"], meta["span"]
    q = raw[:heads * span].reshape(heads, span).astype(np.float64)
    na = raw[heads * span:].reshape(heads, span).astype(np.float64)
    return meta, q, na


def softmax_rows(s):
    e = np.exp(s - s.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def seg_sum(p, keys):
    """[heads, span] -> [heads, segments] summed over each segment's keys
    (nan for a segment that owns none)."""
    cum = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    out = np.full((p.shape[0], len(keys)), np.nan)
    for j, k in enumerate(keys):
        if k is not None:
            out[:, j] = cum[:, k[1]] - cum[:, k[0]]
    return out


def vote(scores_by_head):
    """Each head's argmax segment (nan-safe), the plurality winner (ties to
    the best-ranked head's), and the votes."""
    voted = np.nanargmax(scores_by_head, axis=1)
    votes = Counter(voted.tolist())
    first = {}
    for rank, v in enumerate(voted.tolist()):
        first.setdefault(v, rank)
    winner = min(votes, key=lambda s: (-votes[s], first[s]))
    return winner, voted, votes


def readings(q, na, keys):
    pq, pn = seg_sum(softmax_rows(q), keys), seg_sum(softmax_rows(na), keys)
    lift = pq - pn
    out = {}
    out["vote"] = vote(lift)[0]
    out["vote_nobase"] = vote(pq)[0]
    ratio = np.log(np.maximum(pq, 1e-30)) - np.log(np.maximum(pn, 1e-30))
    out["vote_logratio"] = vote(ratio)[0]
    out["sum_lift"] = int(np.nanargmax(np.nansum(lift, axis=0)))
    # a vote for line s+1 also names s (the boundary heads): pairs, then the
    # pair's first line
    w, voted, votes = vote(lift)
    pair = Counter()
    for s, n in votes.items():
        pair[s] += n
        pair[s - 1] += n
    out["vote_pair_first"] = max(pair, key=lambda s: (pair[s], votes.get(s, 0)))
    return out, voted, lift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.served, encoding="utf-8") as f:
        rows = json.load(f)["questions"]
    by_len = {}
    per_head = {}
    for r in rows:
        if not r["locate"].get("dump"):
            continue
        meta, q, na = load(args.dumps, r["id"])
        keys = meta["keys"]
        got, voted, lift = readings(q, na, keys)
        served = r["locate"]["answer"]["segment"]
        L = r["segments"]
        b = by_len.setdefault(L, {"n": 0, "served_matches_reread": 0, "hits": Counter(),
                                   "votes_on_target": [], "votes_best_other": [], "offsets": Counter()})
        if r["absent"]:
            continue
        t = r["targets"][0]
        b["n"] += 1
        b["served_matches_reread"] += int(got["vote"] == served)
        for name, seg in got.items():
            b["hits"][name] += int(seg == t)
        c = Counter(voted.tolist())
        b["votes_on_target"].append(c.get(t, 0))
        b["votes_best_other"].append(max([n for s, n in c.items() if s != t], default=0))
        for h, v in enumerate(voted.tolist()):
            d = v - t
            key = str(d) if abs(d) <= 1 else ("near" if abs(d) <= 5 else "far")
            b["offsets"][key] += 1
            ph = per_head.setdefault(L, np.zeros((len(voted), 2)))
            ph[h, 0] += int(d == 0)
            ph[h, 1] += 1
    out = {}
    for L in sorted(by_len):
        b = by_len[L]
        n = b["n"]
        tot = sum(b["offsets"].values())
        out[L] = {
            "present": n,
            "served_matches_reread": b["served_matches_reread"],
            "hits": dict(b["hits"]),
            "votes_on_target_median": float(np.median(b["votes_on_target"])),
            "votes_best_other_median": float(np.median(b["votes_best_other"])),
            "head_vote_offsets_pct": {k: round(100 * v / tot, 1) for k, v in sorted(b["offsets"].items())},
            "per_head_target_rate": [round(x, 2) for x in (per_head[L][:, 0] / per_head[L][:, 1]).tolist()],
        }
        print(f"{L:6} lines, {n} present: served==reread {b['served_matches_reread']}/{n}; "
              + " ".join(f"{k} {v}" for k, v in sorted(b["hits"].items()))
              + f" | votes on target {out[L]['votes_on_target_median']:.0f} vs best other "
              f"{out[L]['votes_best_other_median']:.0f} | offsets {out[L]['head_vote_offsets_pct']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
