"""EXPERIMENT (branch locate-long-context, exploratory): very long logs —
R2's cluster windows widened to ~1M tokens with the capture's lines around
them, the same questions re-checked against the whole widened window.

A question stays when it still singles its line out: lexical and combo
questions by r2set's rule (the shared words, together, in no other line);
paraphrase questions when the target's text, times/numbers/ids removed,
occurs once. The cluster's data stays in `.scratch/`.

    python zd_rx.py --r2 <prod/R2> --windows r2-windows.json --out <RX> [--tokens 1000000]
"""

import argparse
import json
import os

from common import content_stems
from r2set import STRIP, cost_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    windows = {w["id"]: w for w in json.load(open(args.windows, encoding="utf-8"))["windows"] if w["source"] == "cluster"}
    questions = [q for q in json.load(open(os.path.join(args.r2, "manifest.json"), encoding="utf-8"))["questions"]
                 if q["source"] == "cluster" and not q["absent"]]
    timelines = {}
    out = []
    for wid, w in windows.items():
        if w["file"] not in timelines:
            with open(w["file"], encoding="utf-8") as f:
                timelines[w["file"]] = [line for line in f.read().rstrip("\n").split("\n") if len(line) <= 1200]
        stream = timelines[w["file"]]
        a, b = w["start"], w["end"]
        total = w["tokens"]
        step = 200
        while total < args.tokens and (a > 0 or b < len(stream)):
            na, nb = max(0, a - step), min(len(stream), b + step)
            total += sum(cost_of(tok, stream[na:a])) + sum(cost_of(tok, stream[b:nb]))
            a, b = na, nb
        lines = stream[a:b]
        norm = [STRIP.sub(" ", line) for line in lines]
        count = {}
        for n in norm:
            count[n] = count.get(n, 0) + 1
        stems = [content_stems(line) for line in lines]
        kept = dropped = 0
        for q in questions:
            if q["window"] != wid:
                continue
            t = q["targets"][0] + (w["start"] - a)
            assert lines[t] == q["state"].split("\n")[q["targets"][0]]
            if q["split"] in ("combo", "lexical"):
                shared = content_stems(q["instruction"]) & stems[t]
                ok = bool(shared) and not any(shared <= s for i, s in enumerate(stems) if i != t)
            else:
                ok = count[norm[t]] == 1
            if not ok:
                dropped += 1
                continue
            kept += 1
            out.append(dict(q, id=q["id"].replace("r2-", "rx-"), state="\n".join(lines), targets=[t],
                            segments=args.tokens, window=f"x-{wid}", tokens=total, depth=t / len(lines)))
        print(f"{wid}: {len(lines)} lines ~{total} tokens, kept {kept} dropped {dropped}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"questions": out}, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"))
    print(len(out), "questions")


if __name__ == "__main__":
    main()
