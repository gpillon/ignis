"""EXPERIMENT (branch locate-long-context): spec 20's judge, registered
before sets L2 and R were asked (`docs/specs/decide/20-locate-at-length.md`).

Reads `long.py ask`'s rows and the server's dumps (both prefills' rows of
the 32 served heads) and scores, per length tier:

- **served**: the vote as `/v1/decide` answers it today;
- **end** (registered, primary): the seven **end-marker heads** -- the heads
  that on development logs vote the line *after* the target more often than
  the target (A+B, `ENDERS`) -- are read as marking where the target ends:
  each names the segment ending just before its peak key (the argmax of its
  per-key lift, the question's softmax less the content-free one's). A peak
  on a separator, or within the first `M` keys of a segment, names the
  previous owned segment; a peak deeper inside a segment names that segment.
  The other 25 heads vote as served; the plurality wins, ties to the
  best-ranked head;
- **snap** (registered, secondary): an end-marker head's vote for `s` moves
  to `s - 1` when the other heads gave `s - 1` more votes than `s`;
- **generation**: the model writes the line out (the comparator).

Then the rules of spec 20 § Rules: the reading rule, the ceiling per set,
and the present/absent AUC of the `found` signals.

    python long_judge.py --set L2 <L2-served.json> <dumps> --set R <R-served.json> <dumps> --out judge.json
"""

import argparse
import json
from collections import Counter

import numpy as np

import long_rows as LR

# The served 32 heads' order is ignis_core::locate's; these are the seven
# whose A+B logs votes land on target+1 more often than on the target.
ENDERS = ("L39.h15", "L39.h23", "L51.h6", "L47.h17", "L51.h12", "L47.h15", "L43.h9")
SERVED = ("L39.h12", "L47.h20", "L59.h16", "L55.h17", "L59.h17", "L59.h7", "L59.h6", "L55.h0", "L59.h8",
          "L59.h10", "L63.h5", "L39.h15", "L59.h2", "L55.h21", "L39.h23", "L55.h9", "L39.h10", "L63.h0",
          "L35.h16", "L55.h14", "L51.h6", "L47.h17", "L51.h12", "L59.h12", "L55.h18", "L47.h15", "L43.h13",
          "L59.h9", "L63.h1", "L59.h14", "L47.h18", "L43.h9")
FLAG = [SERVED.index(h) for h in ENDERS]
M = 5                       # chosen on A+B (end1/3/5: 371/372/377 of 402)
CEILING_POINTS = 10.0       # a tier passes within 10 points of the shortest tier


def plurality(voted):
    votes, first = Counter(), {}
    for rank, v in enumerate(voted):
        votes[v] += 1
        first.setdefault(v, rank)
    return min(votes, key=lambda s: (-votes[s], first[s])), votes


def end_segment(peak, own, start):
    j = own[peak]
    if j >= 0 and start[peak] >= M:
        return int(j)
    k = peak - 1 if j < 0 else peak - start[peak] - 1
    while k >= 0 and own[k] < 0:
        k -= 1
    return int(own[k]) if k >= 0 else int(max(j, 0))


def readings(meta, q, na):
    keys, span = meta["keys"], meta["span"]
    pq, pn = LR.softmax_rows(q), LR.softmax_rows(na)
    lift = LR.seg_sum(pq, keys) - LR.seg_sum(pn, keys)
    voted = np.nanargmax(lift, axis=1).tolist()
    served, votes = plurality(voted)
    own = np.full(span, -1)
    start = np.full(span, -1)
    for j, k in enumerate(keys):
        if k:
            own[k[0]:k[1]] = j
            start[k[0]:k[1]] = np.arange(k[1] - k[0])
    keylift = pq - pn
    end_voted = list(voted)
    for i in FLAG:
        end_voted[i] = end_segment(int(np.argmax(keylift[i])), own, start)
    end, end_votes = plurality(end_voted)
    others = Counter(v for i, v in enumerate(voted) if i not in FLAG)
    snap, _ = plurality([v - 1 if i in FLAG and others.get(v - 1, 0) > others.get(v, 0) else v
                         for i, v in enumerate(voted)])
    return {"served": served, "end": end, "snap": snap,
            "served_confidence": votes[served] / len(voted), "end_confidence": end_votes[end] / len(voted)}


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def judge_set(rows, dumps):
    tiers = {}
    found = {"noul": ([], []), "served_confidence": ([], []), "end_confidence": ([], [])}
    for r in rows:
        if r["locate"]["status"] != 200 or not r["locate"].get("dump"):
            continue
        meta, q, na = LR.load(dumps, r["id"])
        got = readings(meta, q, na)
        assert got["served"] == r["locate"]["answer"]["segment"], r["id"]
        p_yes = (r["locate"].get("found") or {}).get("noul")
        side = 1 if r["absent"] else 0
        if p_yes is not None:
            found["noul"][side].append(p_yes)
        found["served_confidence"][side].append(got["served_confidence"])
        found["end_confidence"][side].append(got["end_confidence"])
        if r["absent"]:
            continue
        t = tiers.setdefault(r["segments"], {"n": 0, "served": 0, "end": 0, "snap": 0, "generation": 0,
                                             "max_span": 0})
        t["n"] += 1
        t["max_span"] = max(t["max_span"], meta["span"])
        for name in ("served", "end", "snap"):
            t[name] += got[name] in r["targets"]
        t["generation"] += bool(set(r.get("generation", {}).get("segments", [])) & set(r["targets"]))
    order = sorted(tiers)
    base = 100.0 * tiers[order[0]]["end"] / tiers[order[0]]["n"]
    ceiling = None
    for tier in order:
        pct = 100.0 * tiers[tier]["end"] / tiers[tier]["n"]
        if pct < base - CEILING_POINTS:
            break
        ceiling = tier
    total = {name: sum(t[name] for t in tiers.values()) for name in ("n", "served", "end", "snap", "generation")}
    return {"tiers": tiers, "total": total, "ceiling_tier": ceiling,
            "ceiling_max_span": tiers[ceiling]["max_span"] if ceiling is not None else None,
            "found_auc": {k: auc(v[0], v[1]) for k, v in found.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=3, action="append", metavar=("NAME", "SERVED", "DUMPS"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {}
    for name, served, dumps in args.set:
        with open(served, encoding="utf-8") as f:
            rows = json.load(f)["questions"]
        report[name] = judge_set(rows, dumps)
    served = sum(r["total"]["served"] for r in report.values())
    end = sum(r["total"]["end"] for r in report.values())
    report["rule_reading"] = {"end": end, "served": served, "pass": end >= served}
    spans = [r["ceiling_max_span"] for k, r in report.items() if k != "rule_reading"]
    report["LOCATE_MAX_KEYS_candidate"] = min(spans) if all(s is not None for s in spans) else None
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for name, r in report.items():
        if name in ("rule_reading", "LOCATE_MAX_KEYS_candidate"):
            continue
        print(f"== {name}: total {r['total']}; ceiling tier {r['ceiling_tier']} ({r['ceiling_max_span']} keys); "
              f"found AUC {r['found_auc']}")
        for tier, t in sorted(r["tiers"].items()):
            print(f"   {tier:>7}: n {t['n']:2}  served {t['served']:2}  end {t['end']:2}  snap {t['snap']:2}  "
                  f"generation {t['generation']:2}  (max span {t['max_span']})")
    print(f"reading rule: end {end} vs served {served} -> {'PASS' if end >= served else 'FAIL'}; "
          f"LOCATE_MAX_KEYS candidate {report['LOCATE_MAX_KEYS_candidate']}")


if __name__ == "__main__":
    main()
