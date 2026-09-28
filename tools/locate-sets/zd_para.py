"""EXPERIMENT (branch locate-long-context, exploratory): prose pointers by
paragraph — the weak point of spec 23 (a 1M-token text).

`offline`: from a `zd_prose.py multi` result, the paragraph of the
sentence `choice`'s pick, and the pick by the paragraphs' summed
probability; paragraph recall of the heads' ranking (a paragraph ranked by
its best sentence).

`run`: over the heads' ranking, per question —
- `sent8`: the sentence `choice` over the first 8 sentences (spec 23 uses 16);
- `para`: a labelled `choice` over the first `--kp` paragraphs (whole
  paragraphs, labelled at their title), the pointers every paragraph at
  p >= 0.05;
- `hier`: then a sentence `choice` over every sentence of the paragraphs
  `para` pointed at.

    python zd_para.py offline --set P2 --multi P2-multi.json
    python zd_para.py run --set P2 --rank P2-rank.json --out P2-para.json
"""

import argparse
import json
import os

import numpy as np

import folded_locate as FL
from zd_prose import render

TAU = 0.05


def paragraphs(lines):
    """Line -> paragraph index, and each paragraph's [first, last] line."""
    of, spans, start = {}, [], None
    for i, line in enumerate(lines + [""]):
        if line and start is None:
            start = i
        if not line and start is not None:
            spans.append((start, i - 1))
            start = None
    for p, (a, b) in enumerate(spans):
        for i in range(a, b + 1):
            of[i] = p
    return of, spans


def group(segments):
    return "<=200K" if segments <= 200000 else f"{segments // 1000}K"


def f1(got, gold):
    hit = len(got & gold)
    p, r = hit / max(1, len(got)), hit / len(gold)
    return 0.0 if hit == 0 else 2 * p * r / (p + r)


def load(set_dir):
    return {q["id"]: q for q in json.load(open(os.path.join(set_dir, "manifest.json"), encoding="utf-8"))["questions"]}


def para_rank(order, of):
    seen, out = set(), []
    for i in order:
        p = of.get(i)
        if p is not None and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def offline(args):
    qs = load(args.set)
    table = {}
    for r in json.load(open(args.multi, encoding="utf-8"))["questions"]:
        q = qs[r["id"]]
        lines = q["state"].split("\n")
        of, _ = paragraphs(lines)
        gold = {of[t] for t in q["targets"]}
        probs = r["probs"]
        mass = {}
        for s, p in probs:
            mass[of[s]] = mass.get(of[s], 0) + p
        by_mass = max(mass, key=mass.get)
        ptrs = {p for p, m in mass.items() if m >= TAU} or {by_mass}
        sent_ptrs = {of[s] for s, p in probs if p >= TAU} or {of[probs[0][0]]}
        row = {"sent": probs[0][0] in q["targets"], "para_of_pick": of[probs[0][0]] in gold,
               "para_by_mass": by_mass in gold, "F1_sent_ptrs": f1(sent_ptrs, gold), "F1_mass_ptrs": f1(ptrs, gold)}
        for g in ("all", group(q["segments"])):
            t = table.setdefault(g, {})
            for k, v in row.items():
                t.setdefault(k, []).append(float(v))
    for g, t in sorted(table.items()):
        print(g, f"n={len(t['sent'])}", "  ".join(f"{k} {100 * np.mean(v):.1f}" for k, v in t.items()))
    if args.rank:
        ranks = json.load(open(args.rank, encoding="utf-8"))
        rec = {}
        for qid, r in ranks.items():
            q = qs.get(qid)
            if q is None:
                continue
            lines = q["state"].split("\n")
            of, _ = paragraphs(lines)
            gold = {of[t] for t in q["targets"]}
            pr = para_rank(r[args.reading], of)
            for g in ("all", group(q["segments"])):
                t = rec.setdefault(g, {})
                for k in (1, 2, 4, 8, 16):
                    t.setdefault(f"any@{k}", []).append(bool(gold & set(pr[:k])))
                    t.setdefault(f"all@{k}", []).append(gold <= set(pr[:k]))
        for g, t in sorted(rec.items()):
            print("paragraph recall", g, f"n={len(t['any@1'])}", "  ".join(f"{k} {100 * np.mean(v):.1f}" for k, v in t.items()))


def ask(url, text, labels, instruction):
    body = {"state": text, "questions": {"q": {"type": "choice", "instructions": instruction,
                                               "criteria": {label: None for label in labels}}}}
    status, payload, ms = FL.post(url, body)
    a = payload.get("answers", {}).get("q", {}) if status == 200 else {}
    if a.get("type") != "choice":
        raise RuntimeError(str(payload)[:300])
    return sorted(((labels[k], p) for k, p in a["probabilities"].items()), key=lambda x: -x[1]), ms


def render_paras(lines, spans, chosen):
    out, labels = [], {}
    for n, p in enumerate(sorted(chosen)):
        a, b = spans[p]
        label = FL.ALPHABET[n]
        labels[label] = p
        out.append(f"{label}: {lines[a]}")
        out += [f"   {lines[i]}" for i in range(a + 1, b + 1)]
        out.append("")
    return "\n".join(out).rstrip("\n"), labels


def render_sentences(lines, spans, chosen):
    out, labels, n = [], {}, 0
    for p in sorted(chosen):
        a, b = spans[p]
        for i in range(a, b + 1):
            if lines[i].startswith("# "):
                out.append(f"   {lines[i]}")
                continue
            label = FL.ALPHABET[n]
            n += 1
            labels[label] = i
            out.append(f"{label}: {lines[i]}")
        out.append("")
    return "\n".join(out).rstrip("\n"), labels


def run(args):
    qs = load(args.set)
    ranks = json.load(open(args.rank, encoding="utf-8"))
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for qid, r in sorted(ranks.items()):
        q = qs[qid]
        if qid in out or (args.min_segments and q["segments"] < args.min_segments):
            continue
        lines = q["state"].split("\n")
        of, spans = paragraphs(lines)
        order = [i for i in r[args.reading] if lines[i] and not lines[i].startswith("# ")]
        rec = {"targets": q["targets"], "segments": q["segments"], "gold": sorted({of[t] for t in q["targets"]})}
        text, labels = render(lines, set(order[:8]))
        rec["sent8"], rec["sent8_ms"] = ask(args.url, text, labels, q["instruction"])
        chosen = para_rank(order, of)[:args.kp]
        text, labels = render_paras(lines, spans, chosen)
        rec["para"], rec["para_ms"] = ask(args.url, text, labels,
                                          f"Which paragraph helps answer this question: {q['question']}")
        rec["para_of"] = {str(p): [spans[p][0], spans[p][1]] for p in chosen}
        pointed = {p for p, v in rec["para"] if v >= TAU} or {rec["para"][0][0]}
        text, labels = render_sentences(lines, spans, pointed)
        rec["hier"], rec["hier_ms"] = ask(args.url, text, labels, q["instruction"])
        rec["hier_of"] = {str(s): of[s] for s, _ in rec["hier"]}
        out[qid] = rec
        json.dump(out, open(args.out, "w", encoding="utf-8"))
        gold = set(rec["gold"])
        print(qid, q["segments"], "sent8", rec["sent8"][0][0] in q["targets"], "para", rec["para"][0][0] in gold,
              "hier", rec["hier"][0][0] in q["targets"], f"{rec['para_ms'] + rec['hier_ms']:.0f} ms", flush=True)
    report(out, qs)


def report(out, qs):
    table = {}
    for qid, rec in out.items():
        q = qs[qid]
        lines = q["state"].split("\n")
        of, _ = paragraphs(lines)
        gold = set(rec["gold"])
        t = set(q["targets"])
        para_ptrs = {p for p, v in rec["para"] if v >= TAU} or {rec["para"][0][0]}
        hier_ptrs = {of[s] for s, v in rec["hier"] if v >= TAU} or {of[rec["hier"][0][0]]}
        s8_ptrs = {of[s] for s, v in rec["sent8"] if v >= TAU} or {of[rec["sent8"][0][0]]}
        row = {"sent8 pick": rec["sent8"][0][0] in t, "sent8 para": of[rec["sent8"][0][0]] in gold,
               "sent8 F1": f1(s8_ptrs, gold), "para pick": rec["para"][0][0] in gold,
               "para F1": f1(para_ptrs, gold), "para recall": len(para_ptrs & gold) / len(gold),
               "hier pick": rec["hier"][0][0] in t, "hier F1": f1(hier_ptrs, gold),
               "shortlist any": bool(gold & {int(p) for p in rec["para_of"]}),
               "ms": (rec["para_ms"] + rec["hier_ms"]) / 1000}
        for g in ("all", group(q["segments"])):
            d = table.setdefault(g, {})
            for k, v in row.items():
                d.setdefault(k, []).append(float(v))
    for g, d in sorted(table.items()):
        n = len(d["ms"])
        print(g, f"n={n}", "  ".join(f"{k} {np.mean(v) if k == 'ms' else 100 * np.mean(v):.1f}" for k, v in d.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("offline")
    a.add_argument("--set", required=True)
    a.add_argument("--multi", required=True)
    a.add_argument("--rank")
    a.add_argument("--reading", default="served:sum")
    b = sub.add_parser("run")
    b.add_argument("--set", required=True)
    b.add_argument("--rank", required=True)
    b.add_argument("--reading", default="served:sum")
    b.add_argument("--kp", type=int, default=8)
    b.add_argument("--min-segments", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--url", default="http://127.0.0.1:8000")
    c = sub.add_parser("report")
    c.add_argument("--set", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "report":
        report(json.load(open(args.out, encoding="utf-8")), load(args.set))
    else:
        {"offline": offline, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
