"""EXPERIMENT (branch locate-long-context): spec 23's judge, committed before
sets R4 and P2 are asked (`docs/specs/decide/23-zero-decode-locate-confirmatory.md`).

    python z23_judge.py --r4 R4-logpipe.json --p2 <P2 dir> --p2-multi P2-multi.json \
        --p2-rank P2-rank.json --auto auto.json --out z23-judge.json
"""

import argparse
import json
import os

import numpy as np


def para_of(lines, i):
    while i > 0 and lines[i - 1] != "":
        i -= 1
    return i


def log_hits(data, l1_key):
    hits = []
    for r in data.values():
        t = r["targets"][0]
        ci = r["l1"][l1_key]["segment"]
        l2 = r["l2"].get(str(ci))
        row = None if l2 is None else (l2.get("choice") or l2.get("end_choice") or {}).get("row")
        hits.append(row is not None and t in l2["members"][row])
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r4", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--p2-multi", required=True)
    ap.add_argument("--p2-rank", required=True)
    ap.add_argument("--auto", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {}

    r4 = json.load(open(args.r4, encoding="utf-8"))["questions"]
    pipe = log_hits(r4, "end5_choice")
    folded_choice = log_hits(r4, "choice")
    out["HL1"] = {"n": len(pipe), "top1": float(np.mean(pipe)), "pass": bool(np.mean(pipe) >= 0.85)}
    out["HL2"] = {"pipeline": int(sum(pipe)), "folded_choice": int(sum(folded_choice)),
                  "pass": bool(sum(pipe) >= sum(folded_choice))}

    qs = {q["id"]: q for q in json.load(open(os.path.join(args.p2, "manifest.json"), encoding="utf-8"))["questions"]}
    rows = json.load(open(args.p2_multi, encoding="utf-8"))["questions"]
    best = {r["id"]: r["probs"][0][0] in r["targets"] for r in rows}
    groups = {"<=200K": [v for k, v in best.items() if qs[k]["segments"] <= 200_000],
              "1M": [v for k, v in best.items() if qs[k]["segments"] >= 1_000_000]}
    out["HP1"] = {"n": len(best), "all": float(np.mean(list(best.values()))),
                  "groups": {g: {"n": len(v), "top1": float(np.mean(v)) if v else None} for g, v in groups.items()}}
    out["HP1"]["pass"] = bool(out["HP1"]["all"] >= 0.85 and all(
        v["top1"] is not None and v["top1"] >= 0.80 for v in out["HP1"]["groups"].values()))
    f1s, sent = [], []
    for r in rows:
        lines = qs[r["id"]]["state"].split("\n")
        got = {s for s, p in r["probs"] if p >= 0.05} or {r["probs"][0][0]}
        gp = {para_of(lines, t) for t in r["targets"]}
        pp = {para_of(lines, s) for s in got}
        p, rc = len(pp & gp) / len(pp), len(pp & gp) / len(gp)
        f1s.append(0 if p + rc == 0 else 2 * p * rc / (p + rc))
        t = set(r["targets"])
        p, rc = len(got & t) / len(got), len(got & t) / len(t)
        sent.append(0 if p + rc == 0 else 2 * p * rc / (p + rc))
    out["HP2"] = {"paragraph_f1": float(np.mean(f1s)), "sentence_f1": float(np.mean(sent)),
                  "pass": bool(np.mean(f1s) >= 0.75)}
    rank = json.load(open(args.p2_rank, encoding="utf-8"))
    any16 = [any(t in set(rank[k]["served:sum"][:16]) for t in qs[k]["targets"]) for k in rank]
    all16 = [all(t in set(rank[k]["served:sum"][:16]) for t in qs[k]["targets"]) for k in rank]
    out["HP3"] = {"n": len(any16), "any16": float(np.mean(any16)), "all16": float(np.mean(all16)),
                  "pass": bool(np.mean(any16) >= 0.95)}
    auto = json.load(open(args.auto, encoding="utf-8"))
    wrong = [w for w, k in auto["R4"].items() if k != "log"] + [w for w, k in auto["P2"].items() if k != "prose"]
    out["HA"] = {"windows": len(auto["R4"]) + len(auto["P2"]), "wrong": wrong, "pass": not wrong}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1)
    for key in ("HL1", "HL2", "HP1", "HP2", "HP3", "HA"):
        print(key, "PASS" if out[key]["pass"] else "FAIL", {k: v for k, v in out[key].items() if k != "pass"})


if __name__ == "__main__":
    main()
