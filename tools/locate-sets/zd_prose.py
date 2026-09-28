"""EXPERIMENT (branch locate-long-context, exploratory): long prose, read by
the heads (`zd_windows.py`), then — optionally — a labelled `choice` over
the heads' shortlist, answering with several pointers.

`rank`: merge each question's sub-windows (readings standardized within a
sub-window, then concatenated) and report, by tier and split, any-target
top-1 and recall@K, all-targets recall@K; write the rankings.

`choice`: for each question, the first K sentences of a ranking, shown in
their paragraphs (title and every sentence, the candidates labelled, the
rest as context), one labelled `choice`; the answer is every label whose
probability reaches `tau` (at least the best one). Scored by any-hit@1,
precision, recall and F1 of the pointer set.

    python zd_prose.py rank --set <P> --dir <Pw> --out P-rank.json
    python zd_prose.py choice --set <P> --rank P-rank.json --reading served:sum --k 16 --out P-choice.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np

import folded_locate as FL
from zd_offline import zsum

KS = (1, 4, 8, 16, 32, 64)


def readings(z):
    f = {k: (z[f"{k}/q"].astype(float), z[f"{k}/na"].astype(float)) for k in ("sum", "last", "next1", "sep")}
    end = [f["last"][i] + f["sep"][i] + f["next1"][i] for i in (0, 1)]
    s = zsum(f["sum"][0] - f["sum"][1]).sum(0)
    e = zsum(end[0] - end[1]).sum(0)
    return {"sum": s, "end": e, "both": s + e}


def std(v):
    ok = v > -1e5
    out = np.full_like(v, -1e3)
    out[ok] = (v[ok] - v[ok].mean()) / (v[ok].std() + 1e-12)
    return out


def rank(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    by = {}
    for path in glob.glob(os.path.join(args.dir, "*.npz")):
        qid, sub, heads = os.path.basename(path)[:-4].rsplit(".", 2)
        by.setdefault((qid, heads), []).append(path)
    out, table = {}, {}
    for (qid, heads), paths in sorted(by.items()):
        q = qs[qid]
        n = len(q["state"].split("\n"))
        merged = {r: np.full(n, -1e4) for r in ("sum", "end", "both")}
        subs = set()
        for path in paths:
            z = np.load(path, allow_pickle=True)
            meta = json.loads(str(z["meta"]))
            subs.add(meta["subs"])
            a = meta["first_line"]
            for r, v in readings(z).items():
                v = np.where(np.isfinite(v) & (v > -1e2), v, -1e4)
                merged[r][a:a + len(v)] = std(v)
        if len(paths) != max(subs):
            continue
        lines = q["state"].split("\n")
        mask = np.array([not line or line.startswith("# ") for line in lines])
        for r in merged:
            merged[r][mask] = -1e5
        t = set(q["targets"])
        for r, v in merged.items():
            order = np.argsort(-v, kind="stable")
            pos = {int(s): i for i, s in enumerate(order)}
            first = min(pos[x] for x in t)
            last = max(pos[x] for x in t)
            key = f"{heads}:{r}"
            out.setdefault(qid, {})[key] = [int(i) for i in order[:256]]
            for group in ("all", f"{q['split']}", f"{q['segments'] // 1000}K"):
                row = table.setdefault((key, group), {"n": 0, "any": np.zeros(len(KS)), "all": np.zeros(len(KS))})
                row["n"] += 1
                row["any"] += [first < k for k in KS]
                row["all"] += [last < k for k in KS]
    json.dump(out, open(args.out, "w"))
    print(f"{'reading':22s} {'group':6s}   n  " + " ".join(f"any@{k:<3d}" for k in KS) + "  " + " ".join(f"all@{k:<3d}" for k in KS))
    for (key, group), row in sorted(table.items(), key=lambda x: (x[0][1], x[0][0])):
        n = row["n"]
        print(f"{key:22s} {group:6s} {n:4d}  " + " ".join(f"{100 * a / n:6.1f}" for a in row["any"]) + "  "
              + " ".join(f"{100 * a / n:6.1f}" for a in row["all"]))


def render(lines, cand):
    """The candidates' paragraphs in document order, candidates labelled."""
    paras = []
    for c in sorted(cand):
        a = c
        while a > 0 and lines[a - 1] != "":
            a -= 1
        b = c
        while b + 1 < len(lines) and lines[b + 1] != "":
            b += 1
        if not paras or paras[-1][1] < a:
            paras.append([a, b])
    out, labels, n = [], {}, 0
    for a, b in paras:
        for i in range(a, b + 1):
            if i in cand:
                label = FL.ALPHABET[n]
                n += 1
                labels[label] = i
                out.append(f"{label}: {lines[i]}")
            else:
                out.append(f"   {lines[i]}")
        out.append("")
    return "\n".join(out).rstrip("\n"), labels


def choice(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    ranks = json.load(open(args.rank, encoding="utf-8"))
    rows = []
    for qid, r in sorted(ranks.items()):
        q = qs[qid]
        if args.split and q["split"] != args.split:
            continue
        lines = q["state"].split("\n")
        cand = set(i for i in r[args.reading][:args.k] if lines[i] and not lines[i].startswith("# "))
        text, labels = render(lines, cand)
        instruction = q["instruction"]
        body = {"state": text, "questions": {"q": {"type": "choice", "instructions": instruction,
                                                   "criteria": {label: None for label in labels}}}}
        status, payload, ms = FL.post(args.url, body)
        a = payload.get("answers", {}).get("q", {}) if status == 200 else {}
        if a.get("type") != "choice":
            print(qid, "error", str(payload)[:200])
            continue
        probs = sorted(((labels[k], p) for k, p in a["probabilities"].items()), key=lambda x: -x[1])
        rows.append({"id": qid, "targets": q["targets"], "segments": q["segments"], "split": q["split"],
                     "recall_k": len(set(q["targets"]) & cand) / len(q["targets"]), "probs": probs, "ms": ms})
        print(qid, "top", probs[0][0], "hit" if probs[0][0] in q["targets"] else "miss",
              f"p={probs[0][1]:.2f} {ms:.0f} ms", flush=True)
    json.dump({"reading": args.reading, "k": args.k, "questions": rows}, open(args.out, "w"))
    score(rows)


def multi(args):
    """One request per question over the shortlist in context: a labelled
    `choice` and one yes/no (`noul`) per candidate — "line X helps answer
    the question" — asked together, so they share the prefix."""
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    ranks = json.load(open(args.rank, encoding="utf-8"))
    rows = []
    for qid, r in sorted(ranks.items()):
        q = qs[qid]
        if args.split and q["split"] != args.split:
            continue
        if args.ids and qid not in args.ids:
            continue
        lines = q["state"].split("\n")
        cand = set(i for i in r[args.reading][:args.k] if lines[i] and not lines[i].startswith("# "))
        text, labels = render(lines, cand)
        questions = {"best": {"type": "choice", "instructions": q["instruction"],
                              "criteria": {label: None for label in labels}}}
        for label in labels:
            questions[f"n_{label}"] = {"type": "noul", "instructions":
                                       f"The line labelled {label} helps answer this question: {q['question']}"}
        status, payload, ms = FL.post(args.url, {"state": text, "questions": questions})
        ans = payload.get("answers", {}) if status == 200 else {}
        if ans.get("best", {}).get("type") != "choice":
            print(qid, "error", str(payload)[:200])
            continue
        probs = sorted(((labels[k], p) for k, p in ans["best"]["probabilities"].items()), key=lambda x: -x[1])
        yes = {labels[k[2:]]: v.get("noul") for k, v in ans.items() if k.startswith("n_") and "noul" in v}
        rows.append({"id": qid, "targets": q["targets"], "segments": q["segments"], "split": q["split"],
                     "recall_k": len(set(q["targets"]) & cand) / len(q["targets"]), "probs": probs, "yes": yes,
                     "ms": ms})
        print(qid, "best", probs[0][0], "hit" if probs[0][0] in q["targets"] else "miss",
              "yes>0.5:", sorted(s for s, p in yes.items() if p and p >= 0.5), "targets", q["targets"],
              f"{ms:.0f} ms", flush=True)
    json.dump({"reading": args.reading, "k": args.k, "questions": rows}, open(args.out, "w"))
    score(rows)
    score_multi(rows)


def score_multi(rows, taus=(0.3, 0.5, 0.7, 0.9)):
    for tau in taus:
        P, R, F = [], [], []
        for r in rows:
            got = {s for s, p in r["yes"].items() if p is not None and p >= tau} | {r["probs"][0][0]}
            t = set(r["targets"])
            p, rc = len(got & t) / len(got), len(got & t) / len(t)
            P.append(p)
            R.append(rc)
            F.append(0 if p + rc == 0 else 2 * p * rc / (p + rc))
        print(f"  best + yes>={tau:.1f}: precision {100 * np.mean(P):.1f} recall {100 * np.mean(R):.1f} "
              f"F1 {100 * np.mean(F):.1f}  (mean pointers {np.mean([len({s for s, p in r['yes'].items() if p and p >= tau} | {r['probs'][0][0]}) for r in rows]):.1f})")


def score(rows, taus=(0.05, 0.1, 0.2, 0.3, 0.5)):
    n = len(rows)
    print(f"n {n}  any-hit@1 {100 * np.mean([r['probs'][0][0] in r['targets'] for r in rows]):.1f}"
          f"  shortlist all-target recall {100 * np.mean([r['recall_k'] for r in rows]):.1f}")
    for tau in taus:
        P, R, F = [], [], []
        for r in rows:
            got = {s for s, p in r["probs"] if p >= tau} or {r["probs"][0][0]}
            t = set(r["targets"])
            p = len(got & t) / len(got)
            rc = len(got & t) / len(t)
            P.append(p)
            R.append(rc)
            F.append(0 if p + rc == 0 else 2 * p * rc / (p + rc))
        print(f"  tau {tau:.2f}: precision {100 * np.mean(P):.1f} recall {100 * np.mean(R):.1f} F1 {100 * np.mean(F):.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("rank")
    a.add_argument("--set", required=True)
    a.add_argument("--dir", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("choice")
    b.add_argument("--set", required=True)
    b.add_argument("--rank", required=True)
    b.add_argument("--reading", default="served:sum")
    b.add_argument("--k", type=int, default=16)
    b.add_argument("--split")
    b.add_argument("--out", required=True)
    b.add_argument("--url", default="http://127.0.0.1:8000")
    m = sub.add_parser("multi")
    m.add_argument("--set", required=True)
    m.add_argument("--rank", required=True)
    m.add_argument("--reading", default="served:sum")
    m.add_argument("--k", type=int, default=16)
    m.add_argument("--split")
    m.add_argument("--ids", nargs="*")
    m.add_argument("--out", required=True)
    m.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    {"rank": rank, "choice": choice, "multi": multi}[args.cmd](args)


if __name__ == "__main__":
    main()
