"""EXPERIMENT (branch locate-long-context): spec 23's set R4 — a fresh
cluster capture (read only, `.scratch/`), windows of 100K / 200K / ~1M tokens
(two each), five targets per window drawn by `zd_rxq.py`'s rule, questions
written before any model answer and checked by the same rules.

    python z23_r4.py windows --timeline timeline3.txt --out r4-targets.json
    python z23_r4.py show --targets r4-targets.json
    python z23_r4.py build --targets r4-targets.json --questions r4-questions.json --out <R4>
"""

import argparse
import json
import os
import random

from common import content_stems, paraphrase_clean
from r2set import BINS, STRIP, WORD, cost_of, cut

SEED = 20261100
TIERS = ((100_000, 2), (200_000, 2), (1_000_000, 2))


def stream_of(path):
    with open(path, encoding="utf-8") as f:
        return [line for line in f.read().rstrip("\n").split("\n") if len(line) <= 1200]


def windows(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    r = random.Random(SEED)
    stream = stream_of(args.timeline)
    cost = cost_of(tok, stream)
    out, targets = [], []
    for tier, count in TIERS:
        for w in range(count):
            a, b, total = cut(stream, cost, tier, r)
            wid = f"r4-{tier // 1000:04}k-{w}"
            lines = stream[a:b]
            norm = [STRIP.sub(" ", line) for line in lines]
            seen = {}
            for n in norm:
                seen[n] = seen.get(n, 0) + 1
            sets = [set(WORD.findall(n.lower())) for n in norm]
            eligible = [i for i in range(len(lines)) if seen[norm[i]] == 1 and 20 <= len(lines[i]) <= 600]
            r.shuffle(eligible)
            sample = eligible[:120]
            sib = {i: sum(1 for j, s in enumerate(sets) if j != i and sets[i] and len(sets[i] & s) / len(sets[i] | s) >= 0.5)
                   for i in sample}
            by_bin = [[i for i in sample if lo <= sib[i] <= hi] for lo, hi in BINS]
            picks = []
            while len(picks) < args.per_window and any(by_bin):
                for pool in by_bin:
                    if pool and len(picks) < args.per_window:
                        picks.append(pool.pop())
            for i in picks:
                near = sorted(((len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])), j)
                               for j in range(len(lines)) if j != i), reverse=True)[:3]
                targets.append({"window": wid, "line": i, "siblings": sib[i], "text": lines[i],
                                "near": [[round(s, 2), j, lines[j]] for s, j in near]})
            out.append({"id": wid, "tier": tier, "start": a, "end": b, "tokens": total, "lines": len(lines)})
            print(wid, len(lines), "lines", total, "tokens; targets", [(p, sib[p]) for p in picks], flush=True)
    json.dump({"seed": SEED, "timeline": args.timeline, "windows": out, "targets": targets},
              open(args.out, "w", encoding="utf-8"), indent=1)


def show(args):
    for n, t in enumerate(json.load(open(args.targets, encoding="utf-8"))["targets"]):
        print(f"[{n}] {t['window']} line {t['line']} siblings {t['siblings']}\n  T: {t['text'][:args.width]}")
        for s, j, text in t["near"]:
            print(f"   ~{s} {j}: {text[:args.width]}")


def build(args):
    meta = json.load(open(args.targets, encoding="utf-8"))
    stream = stream_of(meta["timeline"])
    wins = {w["id"]: w for w in meta["windows"]}
    asked = json.load(open(args.questions, encoding="utf-8"))["questions"]
    rows, problems = [], []
    for q in asked:
        t = meta["targets"][q["target"]]
        w = wins[t["window"]]
        lines = stream[w["start"]:w["end"]]
        target = lines[t["line"]]
        assert target == t["text"]
        if q["split"] == "paraphrase":
            if not paraphrase_clean(q["instruction"], target) or t["siblings"]:
                problems.append(f"[{q['target']}] paraphrase check")
                continue
        else:
            shared = content_stems(q["instruction"]) & content_stems(target)
            also = [i for i, line in enumerate(lines) if i != t["line"] and shared <= content_stems(line)]
            if not shared or also:
                problems.append(f"[{q['target']}] {q['split']}: shared {sorted(shared)} also in {also[:5]}")
                continue
        rows.append({"id": f"{t['window']}-{q['target']:02d}", "source": "cluster", "split": q["split"],
                     "absent": False, "segments": w["tier"], "siblings": t["siblings"], "window": t["window"],
                     "state": "\n".join(lines), "instruction": q["instruction"], "targets": [t["line"]],
                     "distractors": [], "depth": t["line"] / len(lines)})
    for p in problems:
        print("DROP", p)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"seed": SEED, "questions": rows}, open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8"))
    print(f"{len(rows)} questions, {len(problems)} dropped")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--timeline", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--per-window", type=int, default=5)
    s = sub.add_parser("show")
    s.add_argument("--targets", required=True)
    s.add_argument("--width", type=int, default=260)
    b = sub.add_parser("build")
    b.add_argument("--targets", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    {"windows": windows, "show": show, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
