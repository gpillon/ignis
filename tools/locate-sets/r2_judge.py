"""EXPERIMENT (branch locate-long-context): spec 21's judge, written before
set R2's results were read (`docs/specs/decide/21-locate-in-real-logs.md`).

    python r2_judge.py --set <R2> --served R2-served.json --research <research/R2> \
        --folded-vote R2-folded-vote.json --folded-gen R2-folded-gen.json --out R2-judge.json
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from tokenizers import Tokenizer

TRAJ_K = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
NEVER = 96


def bin_of(s):
    return "0" if s == 0 else ("1-5" if s <= 5 else "6+")


def winners(seg_q, seg_na=None):
    s = seg_q.astype(np.float64) - (seg_na.astype(np.float64) if seg_na is not None else 0.0)
    return np.where(np.isnan(s), -np.inf, s).argmax(axis=1)


def plurality(won):
    return int(np.bincount(won).argmax())


def unique_prefix(tok, lines, t):
    """Tokens of the target (JSON-escaped, as written) until its prefix
    matches no other line's."""
    enc = lambda line: tok.encode(json.dumps(line, ensure_ascii=False)[1:-1]).ids
    target = enc(lines[t])
    others = [enc(line) for i, line in enumerate(lines) if i != t]
    for k in range(1, len(target) + 1):
        if not any(o[:k] == target[:k] for o in others):
            return k
    return len(target)


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--research", required=True)
    ap.add_argument("--folded-vote", required=True)
    ap.add_argument("--folded-gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = {q["id"]: q for q in json.load(f)["questions"]}
    with open(args.served, encoding="utf-8") as f:
        served = {r["id"]: r for r in json.load(f)["questions"]}
    load = lambda p: {r["id"]: r for r in json.load(open(p, encoding="utf-8"))["questions"]}
    fvote, fgen = load(args.folded_vote), load(args.folded_gen)
    present = [q for q in manifest.values() if not q["absent"]]
    out = {}

    def vote_hit(qid):
        a = served[qid]["locate"].get("answer", {})
        return a.get("type") == "locate" and a.get("segment") in manifest[qid]["targets"]

    def gen_hit(qid):
        return bool(set(served[qid].get("generation", {}).get("segments", [])) & set(manifest[qid]["targets"]))

    # H1
    rate = {}
    for b in ("0", "1-5", "6+"):
        sub = [q["id"] for q in present if bin_of(q["siblings"]) == b]
        rate[b] = {"n": len(sub), "miss_rate": 1 - np.mean([vote_hit(i) for i in sub]) if sub else None}
    gap = 100 * (rate["6+"]["miss_rate"] - rate["0"]["miss_rate"])
    out["H1"] = {"bins": rate, "gap_points": gap, "pass": gap >= 20}

    # H2, H3, H4 from the research dumps
    frac, length, share_by_tier, traj = [], [], {}, {}
    for path in sorted(glob.glob(os.path.join(args.research, "*.npz"))):
        qid = os.path.splitext(os.path.basename(path))[0]
        q = manifest[qid]
        t = q["targets"][0]
        z = np.load(path, allow_pickle=True)
        if "heads0/g0/seg_q" in z.files:
            won = np.concatenate([winners(z[f"heads0/g{g}/seg_q"], z[f"heads0/g{g}/seg_na"]) for g in range(12)])
            frac.append(float((won == t).mean()))
            length.append(json.loads(str(z["heads0/g0/meta"]))["span"])
            shares = []
            for g in range(12):
                lse = z[f"heads0/g{g}/lse"]
                m = lse.max(axis=1, keepdims=True)
                p = np.exp(lse - m)
                shares.append(p[:, 1] / p.sum(axis=1))
            share_by_tier.setdefault(q["segments"], []).append(float(np.concatenate(shares).mean()))
        by_k = {}
        for k in TRAJ_K:
            if f"traj/k{k}/seg_q" in z.files:
                by_k[k] = plurality(winners(z[f"traj/k{k}/seg_q"])) == t
        traj[qid] = by_k
    rho, p = spearmanr(length, frac)
    out["H2"] = {"n": len(frac), "rho": float(rho), "p_one_sided": float(p / 2) if rho < 0 else 1 - float(p / 2),
                 "pass": bool(rho < 0 and p / 2 < 0.05)}
    tiers = sorted(share_by_tier)
    diff = 100 * abs(np.mean(share_by_tier[tiers[-1]]) - np.mean(share_by_tier[tiers[0]]))
    out["H3"] = {"share_by_tier": {str(t): float(np.mean(v)) for t, v in share_by_tier.items()},
                 "diff_points": float(diff), "pass": bool(diff <= 5)}
    misses = [qid for qid in traj if not vote_hit(qid) and 1 in traj[qid] and 32 in traj[qid]]
    at1 = np.mean([traj[q][1] for q in misses]) if misses else None
    at32 = np.mean([traj[q][32] for q in misses]) if misses else None
    out["H4a"] = {"n": len(misses), "k1": at1, "k32": at32,
                  "pass": bool(misses) and 100 * (at32 - at1) >= 20}
    kstar, uniq = [], []
    for qid, by_k in traj.items():
        if not by_k:
            continue
        ks = sorted(by_k)
        first = NEVER
        for i, k in enumerate(ks):
            if all(by_k[j] for j in ks[i:]):
                first = k
                break
        q = manifest[qid]
        kstar.append(first)
        uniq.append(unique_prefix(tok, q["state"].split("\n"), q["targets"][0]))
    rho4, p4 = spearmanr(uniq, kstar)
    out["H4b"] = {"n": len(kstar), "rho": float(rho4), "p_one_sided": float(p4 / 2) if rho4 > 0 else 1 - float(p4 / 2),
                  "pass": bool(rho4 > 0 and p4 / 2 < 0.05)}

    # M1, M2 and the report
    n = len(present)
    top1 = {
        "served_vote": sum(vote_hit(q["id"]) for q in present),
        "generation": sum(gen_hit(q["id"]) for q in present),
        "folded_vote": sum(fvote[q["id"]]["hit"] for q in present),
        "folded_generation": sum(fgen[q["id"]]["hit"] for q in present),
    }
    pct = {k: 100 * v / n for k, v in top1.items()}
    out["M1"] = {"folded_generation": pct["folded_generation"], "generation": pct["generation"],
                 "pass": pct["folded_generation"] >= pct["generation"] - 5}
    out["M2"] = {"folded_generation": pct["folded_generation"], "served_vote": pct["served_vote"],
                 "pass": pct["folded_generation"] - pct["served_vote"] >= 10}
    out["top1"] = {k: f"{v}/{n}" for k, v in top1.items()}
    out["level1"] = {"folded_vote": sum(fvote[q["id"]]["level1_hit"] for q in present),
                     "folded_generation": sum(fgen[q["id"]]["level1_hit"] for q in present)}
    by = {}
    for q in present:
        key = f"{q['source']}/{q['segments']}"
        b = by.setdefault(key, {"n": 0, "served_vote": 0, "generation": 0, "folded_vote": 0, "folded_generation": 0})
        b["n"] += 1
        b["served_vote"] += vote_hit(q["id"])
        b["generation"] += gen_hit(q["id"])
        b["folded_vote"] += fvote[q["id"]]["hit"]
        b["folded_generation"] += fgen[q["id"]]["hit"]
    out["by_source_tier"] = by
    noul = lambda qids: [(served[i]["locate"].get("found") or {}).get("noul") for i in qids]
    pos = [v for v in noul([q["id"] for q in present]) if v is not None]
    neg = [v for v in noul([q["id"] for q in manifest.values() if q["absent"]]) if v is not None]
    out["noul_auc"] = auc(pos, neg)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    for key in ("H1", "H2", "H3", "H4a", "H4b", "M1", "M2"):
        print(key, "PASS" if out[key]["pass"] else "FAIL", {k: v for k, v in out[key].items() if k != "pass"})
    print("top-1", out["top1"], "| level 1", out["level1"], "| noul AUC", out["noul_auc"])
    for key, b in sorted(by.items()):
        print(f"  {key:16s} {b}")


if __name__ == "__main__":
    main()
