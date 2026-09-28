"""EXPERIMENT (branch locate-long-context, exploratory): one head set's rows
over a long set, reduced to key features (`zd_cache.key_features`) as they
land — e.g. the end-marking heads chosen on the short sets, read at length.

    python zd_heads.py --set <R2> --heads endheads.json --out <dir>
"""

import argparse
import glob
import json
import os
import time

import numpy as np

import long_rows as LR
from zd_cache import key_features
from zd_qtok import post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--heads", required=True, nargs="+", help="one or more head-set files, read in turn per question")
    ap.add_argument("--out", required=True, nargs="+", help="one output directory per head set")
    ap.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    ap.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--transform", choices=("none", "ids", "endmark"), default="none",
                    help="rewrite each line first: a unique id in front, or an explicit end marker")
    args = ap.parse_args()
    sets = [json.load(open(h, encoding="utf-8"))["heads"] for h in args.heads]
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    for out in args.out:
        os.makedirs(out, exist_ok=True)
    for q in questions:
        if q["absent"]:
            continue
        lines = q["state"].split("\n")
        if args.transform == "ids":
            lines = [f"#{i:04d} {line}" for i, line in enumerate(lines)]
        elif args.transform == "endmark":
            lines = [f"{line} <eol>" for line in lines]
        for heads, out in zip(sets, args.out):
            path = os.path.join(out, q["id"] + ".npz")
            if os.path.exists(path):
                continue
            with open(args.control, "w", encoding="utf-8") as f:
                json.dump({"heads": heads, "tag": "heads"}, f)
            before = set(glob.glob(os.path.join(args.raw, "*.json")))
            body = {"state": "\n".join(lines), "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
            status, payload, ms = post(args.url, body)
            with open(args.control, "w", encoding="utf-8") as f:
                json.dump({}, f)
            new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
            if len(new) != 1:
                print(q["id"], "no dump", status, str(payload)[:200], flush=True)
                continue
            base = os.path.splitext(new[0])[0]
            meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
            keys = [tuple(k) if k else None for k in meta["keys"]]
            fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
            arrays = {f"{k}/q": fq[k].astype(np.float32) for k in fq}
            arrays.update({f"{k}/na": fn[k].astype(np.float32) for k in fn})
            arrays["argmax"] = qr.argmax(axis=1).astype(np.int32)
            arrays["meta"] = np.array(json.dumps({"heads": heads, "keys": meta["keys"], "span": meta["span"], "ms": ms,
                                                  "winner": meta.get("winner"), "transform": args.transform}))
            np.savez_compressed(path, **arrays)
            for ext in (".json", ".bin"):
                os.remove(base + ext)
            print(f"{q['id']} {os.path.basename(out)} {ms / 1e3:.1f} s winner {meta.get('winner')} targets {q['targets']}",
                  flush=True)


if __name__ == "__main__":
    main()
