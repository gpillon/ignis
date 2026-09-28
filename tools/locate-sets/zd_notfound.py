"""EXPERIMENT (branch locate-long-context, exploratory): "not found" for the
`log` pipeline (spec 23's `heads-end5+choice / L2 choice-end`).

Each question is asked as written (present or authored absent) and, for a
present one, again over the log **without its target line** (`deleted`: the
sets' rule makes the target's normalized text unique in its window and the
question's words together in no other line, so the question has no answer
there — the hardest absent, its near-duplicates still in place).

At both levels one request asks, over the same labelled text:
- `plain`: the labelled `choice` (the pipeline as registered);
- `none`: the same `choice` plus one option "no line answers";
- `found`: a yes/no "Is there a line in the evidence that answers this
  question: ...".
Level 2 follows `plain`'s pick. Scores are compared offline (`--report`).

    python zd_notfound.py --set <R2> --heads endheads.json --out R2-nf.json [--deleted]
    python zd_notfound.py --report R2-nf.json R-nf.json --test R4-nf.json
"""

import argparse
import json
import os

import numpy as np

import compress as C
import folded_locate as FL
from zd_logpipe import head_rank
from zd_qtok import post

K1, K2 = 5, 16
NONE = "No line of the evidence answers the criterion"
FOUND = "Is there a line in the evidence that answers this question: {}"


def ask(url, lines, instruction):
    """plain / none / found over `lines`, labelled; one request."""
    labels = FL.ALPHABET[:len(lines)]
    state = "\n".join(f"{label}: {line}" for label, line in zip(labels, lines))
    crit = {label: None for label in labels}
    body = {"state": state, "questions": {
        "plain": {"type": "choice", "instructions": instruction, "criteria": crit},
        "none": {"type": "choice", "instructions": instruction, "criteria": dict(crit, none=NONE)},
        "found": {"type": "noul", "instructions": FOUND.format(instruction)}}}
    status, payload, ms = post(url, body)
    ans = payload.get("answers", {}) if status == 200 else {}
    if ans.get("plain", {}).get("type") != "choice" or ans.get("none", {}).get("type") != "choice":
        raise RuntimeError(f"{status} {str(payload)[:300]}")
    index = {label: i for i, label in enumerate(labels)}
    plain = sorted(((index[k], p) for k, p in ans["plain"]["probabilities"].items()), key=lambda x: -x[1])
    none = ans["none"]["probabilities"]
    return {"plain": plain, "p_none": none.get("none", 0.0),
            "none_best": max(none, key=none.get) if none else None,
            "found": ans.get("found", {}).get("noul"), "ms": ms}


def one(args, heads, lines, instruction, target):
    f = C.fold(lines, C.SIM, values=True)
    tc = next((i for i, c in enumerate(f.clusters) if target is not None and target in c.members), None)
    rec = {"target_cluster": tc, "clusters": len(f.clusters)}
    if len(f.level1) > K1:
        rank, rec["l1_heads_ms"] = head_rank(args, heads, f.level1, instruction)
        cand = sorted(rank["end"][:K1])
    else:
        cand, rec["l1_heads_ms"] = list(range(len(f.level1))), 0.0
    rec["l1_cand"] = cand
    rec["l1"] = ask(args.url, [f.level1[i] for i in cand], instruction)
    ci = cand[rec["l1"]["plain"][0][0]]
    rec["l1_pick"] = ci
    texts, members = C.level2(f, ci)
    if len(texts) > K2:
        rank, rec["l2_heads_ms"] = head_rank(args, heads, texts, instruction)
        rows = sorted(rank["end"][:K2])
    else:
        rows, rec["l2_heads_ms"] = list(range(len(texts))), 0.0
    # asked even over one row: "none" and "found" still have something to say
    rec["l2"] = ask(args.url, [texts[i] for i in rows], instruction)
    row = rows[rec["l2"]["plain"][0][0]]
    rec["answer_lines"] = members[row]
    rec["hit"] = target is not None and target in members[row]
    if getattr(args, "raw_final", False):
        # the same shortlisted rows shown as their original lines (each row's
        # first line, unfolded), in document order
        raw = sorted(members[i][0] for i in rows)
        rec["l2raw"] = ask(args.url, [lines[i] for i in raw], instruction)
        rec["l2raw_lines"] = raw
        pick = raw[rec["l2raw"]["plain"][0][0]]
        rec["hit_raw"] = target is not None and any(target in members[i] and members[i][0] == pick for i in rows)
    return rec


def run(args):
    heads = json.load(open(args.heads, encoding="utf-8"))["heads"]
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    out = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    for q in questions:
        todo = [("absent" if q["absent"] else "present", q["id"])]
        if args.deleted and not q["absent"]:
            todo.append(("deleted", q["id"] + ":del"))
        for variant, key in todo:
            if key in out:
                continue
            lines = q["state"].split("\n")
            target = q["targets"][0] if variant == "present" else None
            if variant == "deleted":
                gone = set(q["targets"])
                lines = [line for i, line in enumerate(lines) if i not in gone]
            rec = one(args, heads, lines, q["instruction"], target)
            rec.update(variant=variant, segments=q["segments"], id=q["id"])
            out[key] = rec
            json.dump(out, open(args.out, "w", encoding="utf-8"))
            print(key, variant, "hit" if rec["hit"] else "-", f"L1 none {rec['l1']['p_none']} found {rec['l1']['found']}"
                  f" | L2 none {rec['l2']['p_none']} found {rec['l2']['found']}", flush=True)


def scores(r):
    """Per item, the 'absent' score of each rule (higher = more absent)."""
    l1, l2 = r["l1"], r["l2"]
    g = lambda v, d=0.0: d if v is None else v
    return {
        "none L2": g(l2["p_none"]),
        "none max(L1,L2)": max(g(l1["p_none"]), g(l2["p_none"])),
        "1-found L2": 1 - g(l2["found"], 1.0),
        "1-found L1": 1 - g(l1["found"], 1.0),
        "1-maxp L2": 1 - l2["plain"][0][1],
        "none L2 + 1-found L2": g(l2["p_none"]) + 1 - g(l2["found"], 1.0),
    }


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def report(args):
    def load(paths):
        items = []
        for p in paths:
            items += list(json.load(open(p, encoding="utf-8")).values())
        return items
    train = load(args.report)
    test = load(args.test) if args.test else []
    for name, items in (("train", train), ("test", test)):
        if not items:
            continue
        pres = [r for r in items if r["variant"] == "present"]
        absn = [r for r in items if r["variant"] != "present"]
        print(f"== {name}: present {len(pres)} (pipeline hit {sum(r['hit'] for r in pres)}), "
              f"authored absent {sum(r['variant'] == 'absent' for r in items)}, deleted {sum(r['variant'] == 'deleted' for r in items)}")
        # none argmax at L2 (or L1): the `none` choice itself
        for lvl in ("l2", "l1|l2"):
            says = lambda r: (r["l2"]["none_best"] == "none") or (lvl == "l1|l2" and r["l1"]["none_best"] == "none")
            ok_p = sum(r["hit"] and not says(r) for r in pres)
            for v in ("absent", "deleted"):
                sub = [r for r in absn if r["variant"] == v]
                if sub:
                    print(f"  none argmax {lvl:6s}: present right {ok_p}/{len(pres)}; {v} flagged {sum(says(r) for r in sub)}/{len(sub)}")
        for rule in scores(pres[0]):
            sp = [scores(r)[rule] for r in pres]
            line = f"  {rule:22s}"
            for v in ("absent", "deleted"):
                sa = [scores(r)[rule] for r in absn if r["variant"] == v]
                if sa:
                    line += f" AUC vs {v} {auc(sa, sp):.3f}"
            print(line)
    if test:
        # thresholds chosen on train (max balanced accuracy), applied on test
        print("== thresholds from train, applied to test")
        for rule in scores(train[0]):
            def acc(items, tau):
                pres = [r for r in items if r["variant"] == "present"]
                absn = [r for r in items if r["variant"] != "present"]
                ok_p = sum(r["hit"] and scores(r)[rule] < tau for r in pres) / len(pres)
                ok_a = sum(scores(r)[rule] >= tau for r in absn) / len(absn)
                return ok_p, ok_a
            taus = sorted({scores(r)[rule] for r in train}) + [9.0]
            best = max(taus, key=lambda t: sum(acc(train, t)))
            tp, ta = acc(train, best)
            ep, ea = acc(test, best)
            print(f"  {rule:22s} tau {best:.3f}: train present-right {100 * tp:.1f} absent {100 * ta:.1f} | "
                  f"test present-right {100 * ep:.1f} absent {100 * ea:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set")
    ap.add_argument("--heads")
    ap.add_argument("--out")
    ap.add_argument("--deleted", action="store_true")
    ap.add_argument("--raw-final", action="store_true", help="also ask over the shortlisted rows' original lines")
    ap.add_argument("--report", nargs="*")
    ap.add_argument("--test", nargs="*")
    ap.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    ap.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
