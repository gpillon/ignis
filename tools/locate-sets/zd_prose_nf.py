"""EXPERIMENT (branch locate-long-context, exploratory): "not found" for the
`prose` pipeline (spec 23: served heads' sum reading, the first 16
sentences in their paragraphs, a labelled `choice`).

`build`: absent questions over P2's windows —
- `deleted`: a question asked over its own window with its gold paragraphs
  removed (its eight HotpotQA distractor paragraphs, retrieved for it, stay:
  the hard absent); 16K / 64K / 128K windows;
- `cross`: a question of another window of the same tier, asked over a
  200K window whose text does not hold any of its gold sentences.
Head rows then come from `zd_windows.py` over the new manifest.

`ask`: per question (P2's present ones and the absent ones), two requests over
the heads' ranking —
- sentences: the first 16 in their paragraphs; `plain` choice, `none`
  (plus an option "no sentence answers"), `found` (yes/no);
- paragraphs: the first 8 whole paragraphs (`zd_para.py`); the same three.

    python zd_prose_nf.py build --set P2 --out P2abs
    python zd_windows.py --set P2abs --heads served.json --out P2absw --window 210000
    python zd_prose_nf.py rank --set P2abs --dir P2absw --out P2abs-rank.json
    python zd_prose_nf.py ask --present P2 P2-rank.json --absent P2abs P2abs-rank.json --out P2-nf.json
    python zd_prose_nf.py report --out P2-nf.json
"""

import argparse
import glob
import json
import os
import random

import numpy as np

import folded_locate as FL
from zd_notfound import auc
from zd_para import paragraphs, para_rank, render_paras
from zd_prose import readings, render, std

NONE_S = "No sentence of the evidence answers the criterion"
NONE_P = "No paragraph of the evidence answers the criterion"
FOUND = "Is there a sentence in the evidence that answers this question: {}"


def build(args):
    m = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))
    r = random.Random(args.seed)
    out = []
    by_window = {}
    for q in m["questions"]:
        by_window.setdefault(q["window"], []).append(q)
    for wid, qs in sorted(by_window.items()):
        tier = qs[0]["segments"]
        lines = qs[0]["state"].split("\n")
        if tier in args.deleted:
            for q in qs:
                of, spans = paragraphs(lines)
                gone = set()
                for p in {of[t] for t in q["targets"]}:
                    a, b = spans[p]
                    gone |= set(range(a, b + 2))  # the paragraph and the empty line after it
                kept = [line for i, line in enumerate(lines) if i not in gone]
                while kept and kept[-1] == "":
                    kept.pop()
                out.append(dict(q, id=q["id"] + "~del", window=q["id"] + "~del", state="\n".join(kept),
                                targets=[], absent=True, variant="deleted"))
        if tier in args.cross:
            others = [q for q in m["questions"] if q["segments"] == tier and q["window"] != wid]
            text = set(lines)
            r.shuffle(others)
            n = 0
            for q in others:
                gold = {q["state"].split("\n")[t] for t in q["targets"]}
                if gold & text:
                    continue
                out.append(dict(q, id=f"{q['id']}@{wid}", window=wid, state=qs[0]["state"], targets=[],
                                absent=True, variant="cross"))
                n += 1
                if n == args.per_window:
                    break
    os.makedirs(args.out, exist_ok=True)
    json.dump({"seed": args.seed, "from": args.set, "questions": out},
              open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"))
    print(len(out), "absent questions:", {v: sum(q["variant"] == v for q in out) for v in ("deleted", "cross")})


def rank(args):
    qs = {q["id"]: q for q in json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]}
    out = {}
    for path in glob.glob(os.path.join(args.dir, "*.npz")):
        qid, sub, heads = os.path.basename(path)[:-4].rsplit(".", 2)
        q = qs[qid]
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if meta["subs"] != 1:
            raise SystemExit(f"{qid}: {meta['subs']} sub-windows (this tool reads single windows)")
        lines = q["state"].split("\n")
        v = readings(z)["sum"]
        v = std(np.where(np.isfinite(v) & (v > -1e2), v, -1e4))
        mask = np.array([not line or line.startswith("# ") for line in lines])
        v[mask] = -1e5
        out.setdefault(qid, {})[f"{heads}:sum"] = [int(i) for i in np.argsort(-v, kind="stable")[:256]]
    json.dump(out, open(args.out, "w"))
    print(len(out), "rankings")


def request(url, text, labels, instruction, none, question):
    crit = {label: None for label in labels}
    body = {"state": text, "questions": {
        "plain": {"type": "choice", "instructions": instruction, "criteria": crit},
        "none": {"type": "choice", "instructions": instruction, "criteria": dict(crit, none=none)},
        "found": {"type": "noul", "instructions": FOUND.format(question)}}}
    status, payload, ms = FL.post(url, body)
    ans = payload.get("answers", {}) if status == 200 else {}
    if ans.get("plain", {}).get("type") != "choice":
        raise RuntimeError(f"{status} {str(payload)[:300]}")
    plain = sorted(((labels[k], p) for k, p in ans["plain"]["probabilities"].items()), key=lambda x: -x[1])
    nprobs = ans["none"]["probabilities"]
    return {"plain": plain, "p_none": nprobs.get("none", 0.0), "none_best": max(nprobs, key=nprobs.get),
            "found": ans.get("found", {}).get("noul"), "ms": ms}


def ask(args):
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    todo = []
    for set_dir, rank_path in (args.present, args.absent):
        qs = {q["id"]: q for q in json.load(open(os.path.join(set_dir, "manifest.json"), encoding="utf-8"))["questions"]}
        ranks = json.load(open(rank_path, encoding="utf-8"))
        todo += [(qs[qid], r) for qid, r in sorted(ranks.items())
                 if qid in qs and qs[qid]["segments"] <= args.max_segments]
    for q, r in todo:
        if q["id"] in out:
            continue
        lines = q["state"].split("\n")
        of, spans = paragraphs(lines)
        order = [i for i in r[args.reading] if lines[i] and not lines[i].startswith("# ")]
        rec = {"variant": q.get("variant", "present"), "segments": q["segments"], "targets": q["targets"],
               "gold": sorted({of[t] for t in q["targets"]})}
        text, labels = render(lines, set(order[:16]))
        rec["sent"] = request(args.url, text, labels, q["instruction"], NONE_S, q["question"])
        chosen = para_rank(order, of)[:8]
        text, labels = render_paras(lines, spans, chosen)
        rec["para"] = request(args.url, text, labels, f"Which paragraph helps answer this question: {q['question']}",
                              NONE_P, q["question"])
        out[q["id"]] = rec
        json.dump(out, open(args.out, "w", encoding="utf-8"))
        print(q["id"], rec["variant"], f"sent none {rec['sent']['p_none']:.3f} found {rec['sent']['found']:.3f} | "
              f"para none {rec['para']['p_none']:.3f} found {rec['para']['found']:.3f}", flush=True)


def verify(args):
    """The sentence `choice`'s pick checked alone in its paragraph (as
    `zd_verify.py` checks a log line): a `choice` between the pick (A) and
    "no sentence answers", and a yes/no."""
    qs = {}
    for set_dir in args.sets:
        qs.update({q["id"]: q for q in json.load(open(os.path.join(set_dir, "manifest.json"), encoding="utf-8"))["questions"]})
    data = json.load(open(args.out, encoding="utf-8"))
    done = json.load(open(args.verify, encoding="utf-8")) if os.path.exists(args.verify) else {}
    for qid, rec in data.items():
        if qid in done:
            continue
        q = qs[qid]
        lines = q["state"].split("\n")
        of, spans = paragraphs(lines)
        pick = rec["sent"]["plain"][0][0]
        a, b = spans[of[pick]]
        text = "\n".join(f"A: {lines[i]}" if i == pick else f"   {lines[i]}" for i in range(a, b + 1))
        body = {"state": text, "questions": {
            "none": {"type": "choice", "instructions": q["instruction"], "criteria": {"A": None, "none": NONE_S}},
            "yes": {"type": "noul", "instructions": f"The sentence labelled A helps answer this question: {q['question']}"}}}
        status, payload, ms = FL.post(args.url, body)
        ans = payload.get("answers", {}) if status == 200 else {}
        if ans.get("none", {}).get("type") != "choice":
            print(qid, "error", str(payload)[:200])
            continue
        done[qid] = {"variant": rec["variant"], "segments": rec["segments"], "hit": pick in rec["targets"],
                     "para_hit": of[pick] in rec["gold"], "p_none": ans["none"]["probabilities"].get("none", 0.0),
                     "yes": ans["yes"].get("noul"), "ms": ms}
        json.dump(done, open(args.verify, "w", encoding="utf-8"))
        print(qid, rec["variant"], f"none {done[qid]['p_none']:.3f} yes {done[qid]['yes']:.3f}", flush=True)
    pres = [r for r in done.values() if r["variant"] == "present"]
    rules = {"p_none": lambda r: r["p_none"], "1-yes": lambda r: 1 - r["yes"], "mean": lambda r: (r["p_none"] + 1 - r["yes"]) / 2}
    for v in ("deleted", "cross"):
        sub = [r for r in done.values() if r["variant"] == v]
        tiers = {r["segments"] for r in sub}
        same = [r for r in pres if r["segments"] in tiers]
        print(f"== verify vs {v} ({len(sub)}; present of the same tiers {len(same)}, pick right {sum(r['hit'] for r in same)},"
              f" paragraph right {sum(r['para_hit'] for r in same)})")
        for rn, f in rules.items():
            print(f"  {rn:7s} AUC {auc([f(r) for r in sub], [f(r) for r in same]):.3f}   > 0.5: present right "
                  f"{sum(r['hit'] and f(r) <= 0.5 for r in same)}/{len(same)} (para {sum(r['para_hit'] and f(r) <= 0.5 for r in same)}),"
                  f" absent flagged {sum(f(r) > 0.5 for r in sub)}/{len(sub)}")


def report(args):
    data = json.load(open(args.out, encoding="utf-8"))
    pres = [r for r in data.values() if r["variant"] == "present"]
    for unit in ("sent", "para"):
        hit = (lambda r: r["sent"]["plain"][0][0] in r["targets"]) if unit == "sent" else \
            (lambda r: r["para"]["plain"][0][0] in r["gold"])
        print(f"== {unit}: present {len(pres)}, pick right {sum(hit(r) for r in pres)}")
        rules = {"p_none": lambda r: r[unit]["p_none"], "1-found": lambda r: 1 - r[unit]["found"],
                 "1-maxp": lambda r: 1 - r[unit]["plain"][0][1],
                 "p_none+1-found": lambda r: r[unit]["p_none"] + 1 - r[unit]["found"]}
        says = lambda r: r[unit]["none_best"] == "none"
        for v in ("deleted", "cross"):
            sub = [r for r in data.values() if r["variant"] == v]
            if not sub:
                continue
            # against the present questions of the same tiers: a score that
            # follows length would otherwise pass for one that finds absence
            tiers = {r["segments"] for r in sub}
            same = [r for r in pres if r["segments"] in tiers]
            print(f"  vs {v} ({len(sub)}; present of its tiers {len(same)}, pick right {sum(hit(r) for r in same)}): "
                  f"none argmax: present right {sum(hit(r) and not says(r) for r in same)}/{len(same)}, "
                  f"absent flagged {sum(says(r) for r in sub)}/{len(sub)}")
            for name, f in rules.items():
                print(f"    {name:16s} AUC {auc([f(r) for r in sub], [f(r) for r in same]):.3f}")
            for name in ("1-found", "p_none"):
                f = rules[name]
                for tau in (0.3, 0.5, 0.7):
                    print(f"    {name} >= {tau}: present right {sum(hit(r) and f(r) < tau for r in same)}/{len(same)}, "
                          f"absent flagged {sum(f(r) >= tau for r in sub)}/{len(sub)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=20261120)
    b.add_argument("--deleted", type=int, nargs="*", default=[16000, 64000, 128000])
    b.add_argument("--cross", type=int, nargs="*", default=[200000])
    b.add_argument("--per-window", type=int, default=3)
    r = sub.add_parser("rank")
    r.add_argument("--set", required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--out", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--present", nargs=2, required=True)
    a.add_argument("--absent", nargs=2, required=True)
    a.add_argument("--reading", default="served:sum")
    a.add_argument("--max-segments", type=int, default=200000)
    a.add_argument("--out", required=True)
    a.add_argument("--url", default="http://127.0.0.1:8000")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--sets", nargs="+", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--verify", required=True)
    v.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    {"build": build, "rank": rank, "ask": ask, "report": report, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    main()
