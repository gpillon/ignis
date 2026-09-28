"""EXPERIMENT (branch locate-long-context, exploratory): the labelled
`choice` over the whole log, in chunks of 256 labelled lines, then one
`choice` over the chunks' winners — zero decode, no folding, one pass over
the text (split) plus a short one.

    python zd_chunked.py --set <R2> --out R2-chunked.json [--chunk 256]
"""

import argparse
import json
import os

import folded_locate as FL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    FL.MAX_OPTIONS = args.chunk
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    rows = []
    for q in questions:
        if q["absent"]:
            continue
        lines = q["state"].split("\n")
        got = FL.choice(args.url, lines, q["instruction"])
        seg = got.get("segment")
        # did the target's chunk pick it?
        t = q["targets"][0]
        start = (t // args.chunk) * args.chunk
        row = {"id": q["id"], "targets": q["targets"], "segment": seg, "hit": seg in q["targets"],
               "ms": got.get("ms"), "chunks": got.get("chunks"), "error": got.get("error"),
               "chunk_hit": t in (got.get("winners") or [seg])}
        rows.append(row)
        print(f"{q['id']} lines {len(lines)} -> {seg} {'HIT' if row['hit'] else 'miss'} (chunk {'ok' if row['chunk_hit'] else '--'}) target {t} "
              f"| {got.get('ms', 0) / 1e3:.1f} s", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"chunk": args.chunk, "questions": rows}, f, indent=1)
    n = len(rows)
    print(f"chunked choice: {sum(r['hit'] for r in rows)}/{n} (target's chunk right {sum(r['chunk_hit'] for r in rows)}); median {sorted(r['ms'] or 0 for r in rows)[n // 2] / 1e3:.1f} s")


if __name__ == "__main__":
    main()
