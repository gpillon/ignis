"""EXPERIMENT (branch locate-long-context, exploratory): new targets for the
~1M-token log windows of `zd_rx.py`, drawn by rule and shown for writing
questions, then checked as r2set checks them.

`draw`: in each widened window, eligible lines (20-600 characters, their
text with times/numbers/ids removed unique) sampled by seed; each sampled
line's siblings counted (r2set's Jaccard rule, against the whole window);
targets drawn round-robin over the bins 0, 1-5, 6-50, > 50. `show` prints
each target with its three nearest lines. `build` checks the written
questions (lexical/combo: the shared words, together, in no other line;
paraphrase: no content word shared, only for targets without siblings) and
appends them to the RX manifest as `rxn-*`.

    python zd_rxq.py draw --rx <RX> --r2 <prod/R2> --windows r2-windows.json --out rxq-targets.json
    python zd_rxq.py show --targets rxq-targets.json
    python zd_rxq.py build --targets rxq-targets.json --questions rxq-questions.json --rx <RX>
"""

import argparse
import json
import os
import random
import re

from common import content_stems, paraphrase_clean
from r2set import BINS, STRIP, WORD

SEED = 20261080


def window_states(rx):
    qs = json.load(open(os.path.join(rx, "manifest.json"), encoding="utf-8"))["questions"]
    return {q["window"]: q["state"] for q in qs}


def widened(args):
    """Every widened window's lines (also those whose R2 questions were all dropped)."""
    import zd_rx  # noqa: F401  (same widening rule)
    from tokenizers import Tokenizer
    from r2set import cost_of
    tok = Tokenizer.from_file("F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    windows = {w["id"]: w for w in json.load(open(args.windows, encoding="utf-8"))["windows"] if w["source"] == "cluster"}
    out, streams = {}, {}
    for wid, w in windows.items():
        if w["file"] not in streams:
            with open(w["file"], encoding="utf-8") as f:
                streams[w["file"]] = [line for line in f.read().rstrip("\n").split("\n") if len(line) <= 1200]
        stream = streams[w["file"]]
        a, b, total = w["start"], w["end"], w["tokens"]
        while total < 1_000_000 and (a > 0 or b < len(stream)):
            na, nb = max(0, a - 200), min(len(stream), b + 200)
            total += sum(cost_of(tok, stream[na:a])) + sum(cost_of(tok, stream[b:nb]))
            a, b = na, nb
        out[f"x-{wid}"] = stream[a:b]
    return out


def draw(args):
    r = random.Random(SEED)
    wins = widened(args)
    targets = []
    for wid, lines in sorted(wins.items()):
        norm = [STRIP.sub(" ", line) for line in lines]
        count = {}
        for n in norm:
            count[n] = count.get(n, 0) + 1
        sets = [set(WORD.findall(n.lower())) for n in norm]
        eligible = [i for i in range(len(lines)) if count[norm[i]] == 1 and 20 <= len(lines[i]) <= 600]
        r.shuffle(eligible)
        sample = eligible[:120]
        sib = {}
        for i in sample:
            a = sets[i]
            sib[i] = sum(1 for j, b in enumerate(sets) if j != i and a and len(a & b) / len(a | b) >= 0.5)
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
        print(wid, len(lines), "lines; targets", [(p, sib[p]) for p in picks], flush=True)
    json.dump({"seed": SEED, "targets": targets}, open(args.out, "w", encoding="utf-8"), indent=1)


def show(args):
    for n, t in enumerate(json.load(open(args.targets, encoding="utf-8"))["targets"]):
        print(f"[{n}] {t['window']} line {t['line']} siblings {t['siblings']}\n  T: {t['text'][:args.width]}")
        for s, j, text in t["near"]:
            print(f"   ~{s} {j}: {text[:args.width]}")


def build(args):
    targets = json.load(open(args.targets, encoding="utf-8"))["targets"]
    asked = json.load(open(args.questions, encoding="utf-8"))["questions"]
    manifest = json.load(open(os.path.join(args.rx, "manifest.json"), encoding="utf-8"))
    states = {q["window"]: q["state"] for q in manifest["questions"]}
    wins = None
    added, problems = [], []
    for q in asked:
        t = targets[q["target"]]
        if t["window"] not in states:
            wins = wins or widened(args)
            states[t["window"]] = "\n".join(wins[t["window"]])
        lines = states[t["window"]].split("\n")
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
        added.append({"id": f"rxn-{t['window']}-{q['target']:02d}", "source": "cluster", "split": q["split"],
                      "absent": False, "segments": 1_000_000, "siblings": t["siblings"], "window": t["window"],
                      "state": states[t["window"]], "instruction": q["instruction"], "targets": [t["line"]],
                      "distractors": [], "depth": t["line"] / len(lines)})
    for p in problems:
        print("DROP", p)
    manifest["questions"] = [q for q in manifest["questions"] if not q["id"].startswith("rxn-")] + added
    json.dump(manifest, open(os.path.join(args.rx, "manifest.json"), "w", encoding="utf-8"))
    print(f"added {len(added)}, dropped {len(problems)}; manifest now {len(manifest['questions'])}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--rx", required=True)
    d.add_argument("--r2", required=True)
    d.add_argument("--windows", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--per-window", type=int, default=4)
    s = sub.add_parser("show")
    s.add_argument("--targets", required=True)
    s.add_argument("--width", type=int, default=260)
    b = sub.add_parser("build")
    b.add_argument("--targets", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--rx", required=True)
    b.add_argument("--windows", required=True)
    args = ap.parse_args()
    {"draw": draw, "show": show, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
