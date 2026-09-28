"""EXPERIMENT (branch locate-long-context, exploratory): zero-decode readings
of the already-dumped rows, offline (`zd_cache.py`'s cache).

Every reading turns one question's rows into a score per segment; top-1 and
recall@K come from the ranking. Readings:

- `vote32` the served vote re-read (32 heads, lift, plurality; ties to the
  better-ranked head, then the summed lift), `vote384` every head;
- `z32` / `z384` the heads' lifts standardized over the question's segments
  and summed; `borda32` each head's rank;
- `logit` a conditional logit over every head's standardized lift, fitted on
  other sets (`--train`), read on the rest;
- key-level readings on the served heads' rows (`sep`, `last`, `max`,
  `first`, `next1`, and `end` = last + sep + next1: a line's closing keys);
- `bm25` the instruction's words against each segment's, and `hyb` the
  reciprocal-rank fusion of `z384` (or `z32`) and `bm25`.

    python zd_offline.py --cache <dir> --train A B L2 R --test R2 C D
"""

import argparse
import math
import os
import pickle
import re
from collections import Counter

import numpy as np
from scipy.optimize import minimize

WORD = re.compile(r"[A-Za-z0-9_]+")
KS = (1, 2, 4, 8, 16, 32, 64, 128)
STOP = set("the a an of to in on for which what line that this is was were with by from at and or it as be "
           "says say said shows show about there any one does do did has have had record item sentence".split())


def load(cache, name):
    with open(os.path.join(cache, name + ".pkl"), "rb") as f:
        return pickle.load(f)


def served_index(r):
    from zd_cache import SERVED
    return SERVED


def vote_scores(lift):
    """Plurality of each head's argmax, ties by the best-ranked head, then
    by the summed lift — a full ranking."""
    lift = np.where(np.isnan(lift), -np.inf, lift)
    won = lift.argmax(axis=1)
    S = lift.shape[1]
    votes = np.bincount(won, minlength=S).astype(float)
    first = np.full(S, lift.shape[0], float)
    for rank, s in enumerate(won):
        first[s] = min(first[s], rank)
    total = np.where(np.isfinite(lift), lift, 0).sum(axis=0)
    total = (total - total.min()) / (np.ptp(total) + 1e-12)
    return votes * 1e3 - first + 0.5 * total


def zsum(lift):
    x = np.where(np.isnan(lift), np.nan, lift)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True) + 1e-12
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=-1e3)


def borda(lift):
    x = np.where(np.isnan(lift), -np.inf, lift)
    return np.argsort(np.argsort(x, axis=1), axis=1).sum(axis=0).astype(float)


def bm25(texts, query, k1=1.2, b=0.75):
    docs = [Counter(w.lower() for w in WORD.findall(t)) for t in texts]
    q = [w.lower() for w in WORD.findall(query) if w.lower() not in STOP]
    N = len(docs)
    avg = np.mean([sum(d.values()) for d in docs]) or 1.0
    df = Counter(w for d in docs for w in d)
    out = np.zeros(N)
    for w in set(q):
        if not df[w]:
            continue
        idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
        for i, d in enumerate(docs):
            f = d.get(w, 0)
            if f:
                L = sum(d.values())
                out[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))
    return out


def rrf(*scores, k=60):
    out = 0
    for s in scores:
        rank = np.argsort(np.argsort(-s))
        out = out + 1.0 / (k + rank + 1)
    return out


def rank_of(scores, targets):
    order = np.argsort(-scores, kind="stable")
    pos = {s: i for i, s in enumerate(order)}
    return min(pos[t] for t in targets)


class Logit:
    """score_s = sum_h w_h z_hs (+ b * bm25z); softmax over segments."""

    def __init__(self, heads, lam=1.0, use_bm25=False):
        self.heads, self.lam, self.use_bm25 = heads, lam, use_bm25

    def feats(self, r):
        X = zsum(r["Q"][self.heads].astype(np.float64) - r["NA"][self.heads])
        if self.use_bm25:
            b = bm25(r["texts"], r["instruction"])[: X.shape[1]]
            b = (b - b.mean()) / (b.std() + 1e-12)
            X = np.vstack([X, b[None]])
        return X

    def fit(self, rows):
        data = [(self.feats(r), r["targets"]) for r in rows if "Q" in r]
        d = data[0][0].shape[0]

        def f(w):
            loss, grad = 0.0, np.zeros(d)
            for X, t in data:
                s = w @ X
                m = s.max()
                p = np.exp(s - m)
                Z = p.sum()
                p /= Z
                tt = [x for x in t if x < X.shape[1]]
                pt = p[tt].sum()
                loss -= math.log(pt + 1e-300)
                grad -= (X[:, tt] @ p[tt]) / (pt + 1e-300) - X @ p
            loss /= len(data)
            grad /= len(data)
            return loss + self.lam * (w @ w) / 2, grad + self.lam * w

        w0 = np.full(d, 0.01)
        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w = res.x
        return self

    def score(self, r):
        return self.w @ self.feats(r)


class LogitKF(Logit):
    """The conditional logit over (head, key feature) pairs: each head's
    lift at a segment's keys (`sum`), its last key, the next segment's first
    key, its separator — standardized over the question's segments."""

    FEATS = ("sum", "last", "next1", "sep")

    def __init__(self, heads, lam=1.0, feats=FEATS):
        super().__init__(heads, lam)
        self.feats_used = feats

    def feats(self, r):
        kh = r["kheads"]
        rows = [kh.index(h) for h in self.heads]
        return np.vstack([zsum(r["kf"][f][0][rows].astype(np.float64) - r["kf"][f][1][rows])
                          for f in self.feats_used])

    def fit(self, rows):
        return super().fit([dict(r, Q=True) for r in rows if "kf" in r and all(h in r["kheads"] for h in self.heads)])


def readings(r, served, logit=None, logit_kf=None):
    out = {}
    S = len(r["texts"]) if "Q" not in r else r["Q"].shape[1]
    if "Q" in r:
        lift = r["Q"].astype(np.float64) - r["NA"]
        out["vote32"] = vote_scores(lift[served])
        out["vote384"] = vote_scores(lift)
        out["z32"] = zsum(lift[served]).sum(axis=0)
        out["z384"] = zsum(lift).sum(axis=0)
        out["borda32"] = borda(lift[served])
        if logit is not None:
            out["logit"] = logit.score(r)
    if "kf" in r:
        kh = r["kheads"]
        rows = [kh.index(h) for h in served] if len(kh) != len(served) else list(range(len(served)))
        kf = {k: (q[rows].astype(np.float64), n[rows].astype(np.float64)) for k, (q, n) in r["kf"].items()}
        if "vote32" not in out:
            out["vote32"] = vote_scores(kf["sum"][0] - kf["sum"][1])
            out["z32"] = zsum(kf["sum"][0] - kf["sum"][1]).sum(axis=0)
        for k in ("sep", "last", "max", "first", "next1"):
            out["k_" + k] = vote_scores(kf[k][0] - kf[k][1])
        end = [kf["last"][i] + kf["sep"][i] + kf["next1"][i] for i in (0, 1)]
        out["k_end"] = vote_scores(end[0] - end[1])
        out["k_endz"] = zsum(end[0] - end[1]).sum(axis=0)
        both = [kf["sum"][i] + kf["sep"][i] + kf["next1"][i] for i in (0, 1)]
        out["k_sum+end"] = vote_scores(both[0] - both[1])
        for name, model in (logit_kf or {}).items():
            if all(h in kh for h in model.heads):
                out[name] = model.score(r)
    b = bm25(r["texts"], r["instruction"])
    if len(b) == S:
        out["bm25"] = b + 1e-9 * np.arange(S)[::-1]
        base = out.get("z384", out.get("z32"))
        if base is not None:
            out["hyb"] = rrf(base, out["bm25"])
            out["hyb_vote"] = rrf(out["vote32"], out["bm25"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="F:/ai/opencode/inference/.scratch/locate/zd/cache")
    ap.add_argument("--train", nargs="*", default=["A", "B", "L2", "R"])
    ap.add_argument("--test", nargs="+", default=["R2", "C", "D"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--export", help="write each test question's top-256 ranking per reading here (JSON)")
    args = ap.parse_args()
    from zd_cache import SERVED
    served = SERVED
    logit, kfm = None, {}
    if args.train:
        rows = [r for s in args.train for r in load(args.cache, s)]
        logit = Logit(list(range(384)), lam=args.lam).fit(rows)
        top = np.argsort(-np.abs(logit.w))[:12]
        print("logit fitted on", args.train, "top heads:",
              ", ".join(f"L{4 * (h // 24) + 3}.h{h % 24}:{logit.w[h]:+.2f}" for h in top))
        kfm = {"lkf32": LogitKF(list(served), lam=args.lam).fit(rows)}
        if all(s in "ABCD" for s in args.train):
            kfm["lkf384"] = LogitKF(list(range(384)), lam=args.lam).fit(rows)
        w = kfm["lkf32"].w.reshape(len(LogitKF.FEATS), -1)
        print("lkf32 weight mass by feature:", {f: round(float(np.abs(w[i]).sum()), 2) for i, f in enumerate(LogitKF.FEATS)})
    exported = {}
    for s in args.test + [x for x in args.train if x not in args.test]:
        rows = load(args.cache, s)
        ranks, bins = {}, {}
        for r in rows:
            rd = readings(r, served, logit if s not in args.train else None, kfm if s not in args.train else None)
            if args.export and s not in args.train:
                exported[r["id"]] = {k: [int(i) for i in np.argsort(-v, kind="stable")[:256]] for k, v in rd.items()}
            b = r.get("siblings")
            b = None if b is None else ("0" if b == 0 else "1-5" if b <= 5 else "6+")
            for k, v in rd.items():
                ranks.setdefault(k, []).append(rank_of(v, r["targets"]))
                if b is not None:
                    bins.setdefault(k, {}).setdefault(b, []).append(rank_of(v, r["targets"]) == 0)
        print(f"\n== {s} ({len(rows)} present){' [train]' if s in args.train else ''}")
        print(f"{'reading':12s} " + " ".join(f"@{k:<4d}" for k in KS) + "   top-1 by siblings 0 / 1-5 / 6+")
        for k, rk in ranks.items():
            rk = np.array(rk)
            line = " ".join(f"{100 * np.mean(rk < K):5.1f}" for K in KS)
            bb = bins.get(k, {})
            sib = " / ".join(f"{100 * np.mean(bb[x]):3.0f}({len(bb[x])})" for x in ("0", "1-5", "6+") if x in bb)
            print(f"{k:12s} {line}   {sib}")

    if args.export:
        import json
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({"train": args.train, "rankings": exported}, f)


if __name__ == "__main__":
    main()
