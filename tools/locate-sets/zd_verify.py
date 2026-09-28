"""EXPERIMENT (branch locate-long-context, exploratory): "not found" as a
check of the `log` pipeline's answer on its **original line**.

`zd_notfound.py` showed the "none" option at level 2 wins on present
questions too: a level-2 row is its values only, so whether it answers
cannot be read from it. Here the pipeline's answer (the picked row's first
original line, unfolded) is shown alone and asked, in one request:
- `none`: a `choice` between that line (A) and "no line answers";
- `yes`: a yes/no "The line answers this question: ...".

    python zd_verify.py --set <R2> --nf R2-nf.json --out R2-verify.json
    python zd_verify.py --report R2-verify.json R-verify.json --test R4-verify.json
"""

import argparse
import json
import os

import numpy as np

from zd_notfound import NONE, auc
from zd_qtok import post

YES = "The line in the evidence answers this question: {}"


def run(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    nf = json.load(open(args.nf, encoding="utf-8"))
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for key, rec in nf.items():
        if key in out:
            continue
        q = qs[rec["id"]]
        lines = q["state"].split("\n")
        if rec["variant"] == "deleted":
            gone = set(q["targets"])
            lines = [line for i, line in enumerate(lines) if i not in gone]
        answer = lines[rec["answer_lines"][0]]
        body = {"state": f"A: {answer}", "questions": {
            "none": {"type": "choice", "instructions": q["instruction"], "criteria": {"A": None, "none": NONE}},
            "yes": {"type": "noul", "instructions": YES.format(q["instruction"])}}}
        status, payload, ms = post(args.url, body)
        ans = payload.get("answers", {}) if status == 200 else {}
        if ans.get("none", {}).get("type") != "choice":
            print(key, "error", str(payload)[:200])
            continue
        out[key] = {"variant": rec["variant"], "hit": rec["hit"], "segments": rec["segments"],
                    "p_none": ans["none"]["probabilities"].get("none", 0.0), "yes": ans["yes"].get("noul"), "ms": ms}
        json.dump(out, open(args.out, "w", encoding="utf-8"))
        print(key, rec["variant"], "hit" if rec["hit"] else "-", f"none {out[key]['p_none']:.3f} yes {out[key]['yes']:.3f}",
              flush=True)


def report(args):
    def load(paths):
        items = []
        for p in paths:
            items += list(json.load(open(p, encoding="utf-8")).values())
        return items
    rules = {"p_none": lambda r: r["p_none"], "1-yes": lambda r: 1 - r["yes"], "mean": lambda r: (r["p_none"] + 1 - r["yes"]) / 2}
    train, test = load(args.report), load(args.test or [])
    for name, items in (("train", train), ("test", test)):
        if not items:
            continue
        pres = [r for r in items if r["variant"] == "present"]
        print(f"== {name}: present {len(pres)} (pipeline hit {sum(r['hit'] for r in pres)})")
        for v in ("absent", "deleted"):
            sub = [r for r in items if r["variant"] == v]
            if not sub:
                continue
            says = lambda r: r["p_none"] > 0.5
            print(f"  none argmax: present right {sum(r['hit'] and not says(r) for r in pres)}/{len(pres)}, "
                  f"{v} flagged {sum(says(r) for r in sub)}/{len(sub)}")
            for rn, f in rules.items():
                print(f"    {rn:7s} AUC vs {v}: {auc([f(r) for r in sub], [f(r) for r in pres]):.3f}"
                      f"  (vs {v}, present hits only: {auc([f(r) for r in sub], [f(r) for r in pres if r['hit']]):.3f})")
    if test:
        print("== thresholds from train (max present-right + absent-flagged), applied to test")
        for rn, f in rules.items():
            def acc(items, tau):
                pres = [r for r in items if r["variant"] == "present"]
                absn = [r for r in items if r["variant"] != "present"]
                return (sum(r["hit"] and f(r) < tau for r in pres) / len(pres), sum(f(r) >= tau for r in absn) / len(absn))
            taus = sorted({f(r) for r in train}) + [9.0]
            best = max(taus, key=lambda t: sum(acc(train, t)))
            tp, ta = acc(train, best)
            ep, ea = acc(test, best)
            print(f"  {rn:7s} tau {best:.3f}: train present-right {100 * tp:.1f} absent {100 * ta:.1f} | "
                  f"test present-right {100 * ep:.1f} absent {100 * ea:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set")
    ap.add_argument("--nf")
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
