"""EXPERIMENT (branch locate-long-context, exploratory): head rows over very
long texts, window by window.

A state longer than `--window` tokens is cut at paragraph breaks (empty
lines) into sub-windows of at most that many tokens; each sub-window is its
own `locate` state. Windows are the outer loop and questions the inner one,
so every question after the first claims the sub-window's prefix. Each dump
is reduced to key features (`zd_cache.key_features`) with the sub-window's
first line, so the readings can be merged over the whole text offline.

    python zd_windows.py --set <P> --heads served.json proseheads.json --out <dir> [--window 120000]
"""

import argparse
import glob
import json
import os

import numpy as np

import long_rows as LR
from zd_cache import key_features
from zd_qtok import post


def sub_windows(lines, cost, budget):
    """[(first line, last line + 1)] cut at empty lines, each within budget."""
    out, start, total, last_break = [], 0, 0, None
    for i, line in enumerate(lines):
        if line == "":
            last_break = i
        total += cost[i]
        if total > budget and last_break is not None and last_break > start:
            out.append((start, last_break))
            start = last_break + 1
            total = sum(cost[start:i + 1])
            last_break = None
    out.append((start, len(lines)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--heads", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=120_000)
    ap.add_argument("--windows", nargs="*", help="only these window ids")
    ap.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    ap.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    sets = [(os.path.splitext(os.path.basename(h))[0], json.load(open(h, encoding="utf-8"))["heads"]) for h in args.heads]
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    by_window = {}
    for q in questions:
        by_window.setdefault(q["window"], []).append(q)
    os.makedirs(args.out, exist_ok=True)
    for wid, qs in by_window.items():
        if args.windows and wid not in args.windows:
            continue
        lines = qs[0]["state"].split("\n")
        cost = [len(tok.encode(json.dumps(line, ensure_ascii=False)[1:-1]).ids) + 1 for line in lines]
        subs = sub_windows(lines, cost, args.window)
        for si, (a, b) in enumerate(subs):
            state = "\n".join(lines[a:b])
            for q in qs:
                for name, heads in sets:
                    path = os.path.join(args.out, f"{q['id']}.s{si:02d}.{name}.npz")
                    if os.path.exists(path):
                        continue
                    with open(args.control, "w", encoding="utf-8") as f:
                        json.dump({"heads": heads, "tag": name}, f)
                    before = set(glob.glob(os.path.join(args.raw, "*.json")))
                    body = {"state": state, "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
                    status, payload, ms = post(args.url, body)
                    with open(args.control, "w", encoding="utf-8") as f:
                        json.dump({}, f)
                    new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
                    if len(new) != 1:
                        print(q["id"], si, name, "no dump", status, str(payload)[:300], flush=True)
                        continue
                    base = os.path.splitext(new[0])[0]
                    meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
                    keys = [tuple(k) if k else None for k in meta["keys"]]
                    fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
                    arrays = {f"{k}/q": fq[k].astype(np.float32) for k in ("sum", "last", "next1", "sep", "max")}
                    arrays.update({f"{k}/na": fn[k].astype(np.float32) for k in ("sum", "last", "next1", "sep", "max")})
                    arrays["meta"] = np.array(json.dumps({"first_line": a, "end_line": b, "sub": si, "subs": len(subs),
                                                          "heads": heads, "span": meta["span"], "ms": ms,
                                                          "winner": meta.get("winner")}))
                    np.savez_compressed(path, **arrays)
                    for ext in (".json", ".bin"):
                        os.remove(base + ext)
                    print(f"{q['id']} s{si}/{len(subs)} {name} {ms / 1e3:.1f} s", flush=True)


if __name__ == "__main__":
    main()
