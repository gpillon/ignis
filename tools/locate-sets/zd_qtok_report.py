"""EXPERIMENT: read `zd_qtok.py`'s dumps — is the line found better at the
instruction's tokens than at the scaffold, alone or pooled (ICR)?

    python zd_qtok_report.py --set <R2> --dir <qtok dir> --cache <zd cache dir>
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

from zd_offline import rank_of, vote_scores, zsum

STOPWORD = set("which what line the a an of to in on for that this is was says say about there any".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cache", default="F:/ai/opencode/inference/.scratch/locate/zd/cache")
    ap.add_argument("--cache-set", default="R2")
    args = ap.parse_args()
    man = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    cache = {r["id"]: r for r in pickle.load(open(os.path.join(args.cache, args.cache_set + ".pkl"), "rb"))}
    res = {}

    def add(k, v, t):
        res.setdefault(k, []).append(rank_of(v, t))

    for path in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        qid = os.path.basename(path)[:-4]
        z = np.load(path, allow_pickle=True)
        log = [r for r in json.loads(str(z["log"])) if "n" in r]
        t = man[qid]["targets"]
        na0 = {f: cache[qid]["kf"][f][1].astype(float) for f in ("sum", "last", "next1", "sep")}
        per = []
        for r in log:
            n = r["n"]
            f = {k: (z[f"p{n}/{k}/q"].astype(float), z[f"p{n}/{k}/na"].astype(float)) for k in ("sum", "last", "next1", "sep")}
            end_q = f["last"][0] + f["sep"][0] + f["next1"][0]
            end_na0 = na0["last"] + na0["sep"] + na0["next1"]
            per.append({"piece": r["piece"], "n": n,
                        "sum_na0": zsum(f["sum"][0] - na0["sum"]),
                        "end_na0": zsum(end_q - end_na0),
                        "sum_raw": zsum(f["sum"][0]),
                        "end_raw": zsum(end_q)})
        scaf, toks = per[0], per[1:]
        content = [p for p in toks if p["piece"].strip("Ġ?\".,").lower() not in STOPWORD and len(p["piece"].strip("Ġ?\".,")) > 1]
        for key in ("sum_na0", "end_na0", "sum_raw", "end_raw"):
            add(f"scaffold/{key}", scaf[key].sum(0), t)
            add(f"best-token/{key}", max((p[key].sum(0) for p in toks), key=lambda v: -rank_of(v, t)), t)  # oracle
            add(f"pool-all/{key}", sum(p[key].sum(0) for p in toks), t)
            if content:
                add(f"pool-content/{key}", sum(p[key].sum(0) for p in content), t)
                add(f"pool-content+scaf/{key}", sum(p[key].sum(0) for p in content) + len(content) * scaf[key].sum(0), t)
            add(f"last-token/{key}", toks[-1][key].sum(0), t)
    n = len(next(iter(res.values())))
    print(f"{n} questions")
    for k, v in sorted(res.items()):
        v = np.array(v)
        print(f"  {k:28s} top-1 {100 * np.mean(v == 0):5.1f}  @4 {100 * np.mean(v < 4):5.1f}  @16 {100 * np.mean(v < 16):5.1f}")


if __name__ == "__main__":
    main()
