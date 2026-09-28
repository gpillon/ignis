"""EXPERIMENT (branch locate-long-context, exploratory): a second hop for
prose, still zero decode. The first pointer (the labelled `choice` over the
heads' shortlist, `zd_prose.py multi`) is written into the instruction, the
heads read the text again (`zd_windows.py` over the manifest this writes),
and the best sentence of another paragraph is the second pointer.

    python zd_hop.py manifest --set <P> --multi P-multi.json --out <P-hop>
    (zd_windows.py --set <P-hop> ...; zd_prose.py rank --set <P-hop> ...)
    python zd_hop.py score --set <P> --multi P-multi.json --rank2 P-hop-rank.json
"""

import argparse
import json
import os

import numpy as np

HOP = "Which sentence helps answer this question: {question} We already know: \"{known}\""


def para_of(lines, i):
    while i > 0 and lines[i - 1] != "":
        i -= 1
    return i


def manifest(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    rows = json.load(open(args.multi, encoding="utf-8"))["questions"]
    out = []
    for r in rows:
        q = dict(qs[r["id"]])
        lines = q["state"].split("\n")
        q["instruction"] = HOP.format(question=q["question"], known=lines[r["probs"][0][0]])
        q["first"] = r["probs"][0][0]
        out.append(q)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"questions": out}, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"))
    print(len(out), "questions")


def score(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    rows = {r["id"]: r for r in json.load(open(args.multi, encoding="utf-8"))["questions"]}
    rank2 = json.load(open(args.rank2, encoding="utf-8"))
    rank1 = json.load(open(args.rank1, encoding="utf-8"))
    res = {}
    for qid, r in rows.items():
        if qid not in rank2:
            continue
        q = qs[qid]
        lines = q["state"].split("\n")
        t = set(q["targets"])
        gp = {para_of(lines, x) for x in t}
        best = r["probs"][0][0]

        def second(order):
            return next((s for s in order if para_of(lines, s) != para_of(lines, best)), None)

        for name, sents in (("best only", [best]),
                            ("best + first-pass heads, other paragraph", [best, second(rank1[qid][args.reading])]),
                            ("best + second-hop heads, other paragraph", [best, second(rank2[qid][args.reading])])):
            sents = [s for s in sents if s is not None]
            got_p = {para_of(lines, s) for s in sents}
            got = set(sents)
            for level, g, gt in (("sentence", got, t), ("paragraph", got_p, gp)):
                p, rc = len(g & gt) / len(g), len(g & gt) / len(gt)
                res.setdefault((name, level), []).append((p, rc, 0 if p + rc == 0 else 2 * p * rc / (p + rc)))
    for (name, level), v in sorted(res.items(), key=lambda x: (x[0][1], x[0][0])):
        v = np.array(v)
        print(f"{level:9s} {name:44s} n {len(v):3d}  P {100 * v[:, 0].mean():5.1f}  R {100 * v[:, 1].mean():5.1f}  "
              f"F1 {100 * v[:, 2].mean():5.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("manifest")
    a.add_argument("--set", required=True)
    a.add_argument("--multi", required=True)
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("--set", required=True)
    b.add_argument("--multi", required=True)
    b.add_argument("--rank1", required=True)
    b.add_argument("--rank2", required=True)
    b.add_argument("--reading", default="served:sum")
    args = ap.parse_args()
    {"manifest": manifest, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
