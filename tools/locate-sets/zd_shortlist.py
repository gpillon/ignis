"""EXPERIMENT (branch locate-long-context, exploratory): shortlist, then
re-read short — zero decode in two passes.

Pass 1 is a reading of the whole log's rows (`zd_offline.py --export`: the
served heads' rows at the scaffold, read at the lines' ends or by a fitted
logit). Pass 2 asks the model again over the shortlist alone — the first K
lines of that ranking, in their original order — by a labelled `choice`
(`folded_locate.choice`), by the served vote (`locate`), or by a labelled
`choice` with the question also written before the lines (`qfirst`).

    python zd_shortlist.py --set <R2> --rankings rank-R2.json --out R2-shortlist.json \
        --readings lkf32 k_endz --ks 16 64 --routes choice vote qfirst
"""

import argparse
import json
import os

import folded_locate as FL


def qfirst(url, lines, instruction):
    labels = FL.ALPHABET[:len(lines)]
    text = "\n".join(f"{label}: {line}" for label, line in zip(labels, lines))
    body = {"state": {"question": instruction, "lines": text},
            "questions": {"q": {"type": "choice", "instructions": instruction,
                                "criteria": {label: None for label in labels}}}}
    status, payload, ms = FL.post(url, body)
    a = payload.get("answers", {}).get("q", {}) if status == 200 else {}
    if a.get("type") != "choice":
        return {"status": status, "error": payload.get("error") or a, "ms": ms}
    return {"segment": labels.index(a["choice"]), "confidence": a["confidence"], "ms": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--readings", nargs="+", default=["lkf32", "k_endz"])
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--routes", nargs="+", default=["choice", "vote", "qfirst"])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    rankings = json.load(open(args.rankings, encoding="utf-8"))["rankings"]
    rows, tally = [], {}
    for q in questions:
        if q["absent"] or q["id"] not in rankings:
            continue
        lines = q["state"].split("\n")
        row = {"id": q["id"], "targets": q["targets"], "runs": {}}
        for reading in args.readings:
            ranking = rankings[q["id"]][reading]
            for k in args.ks:
                cand = sorted(ranking[:k])
                recall = any(t in cand for t in q["targets"])
                for route in args.routes:
                    FL.ROUTE = route
                    if route == "qfirst":
                        got = qfirst(args.url, [lines[i] for i in cand], q["instruction"])
                    else:
                        got = FL.locate(args.url, [lines[i] for i in cand], q["instruction"])
                    seg = cand[got["segment"]] if "segment" in got else None
                    hit = seg in q["targets"]
                    key = f"{reading}/k{k}/{route}"
                    print(f"    {key} {'H' if hit else ('r' if recall else '-')} {got.get('ms', 0):.0f} ms", flush=True)
                    row["runs"][key] = {"segment": seg, "hit": hit, "recall": recall, "ms": got.get("ms"),
                                        "error": got.get("error")}
                    t = tally.setdefault(key, [0, 0, 0, 0.0])
                    t[0] += 1
                    t[1] += hit
                    t[2] += recall
                    t[3] += got.get("ms") or 0
        rows.append(row)
        print(q["id"], " ".join(f"{k}:{'H' if v['hit'] else ('r' if v['recall'] else '-')}" for k, v in row["runs"].items()),
              flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"questions": rows}, f, indent=1)
    for key, (n, hit, rec, ms) in tally.items():
        print(f"{key:24s} top-1 {hit}/{n} = {100 * hit / n:5.1f}%   shortlist recall {100 * rec / n:5.1f}%   {ms / n:6.0f} ms")


if __name__ == "__main__":
    main()
