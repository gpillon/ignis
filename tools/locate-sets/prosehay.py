"""EXPERIMENT (branch locate-long-context, exploratory): long prose for
`locate` — HotpotQA supporting facts spread through a long haystack of
Wikipedia paragraphs, with every gold sentence a target (several pointers).

A window holds a few questions: each question's own ten paragraphs (its two
gold ones and its eight retrieved distractors) and filler paragraphs from
other dev questions, until the window's token budget; paragraphs are
shuffled, each is its title on a line of its own then one sentence per line,
and paragraphs are separated by an empty line. A title never repeats inside
a window (HotpotQA reuses article intros across questions). Questions used
by sets A-D are excluded. Half the windows of each tier are `dev`, half
`test`: every choice (heads, K, thresholds) is made on dev only.

Tiers past the engine's context (500K, 1M) are read in sub-windows.

    python prosehay.py --out <dir> [--seed 20261070]
"""

import argparse
import json
import os
import random

from common import nfc
from prose import INSTRUCTION, classify, load

LOC = "F:/ai/opencode/inference/.scratch/locate"


def paragraph_lines(title, sentences):
    return [f"# {nfc(title)}"] + [nfc(" ".join(s.split())) for s in sentences if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261070)
    ap.add_argument("--tiers", type=int, nargs="+", default=[16_000, 64_000, 128_000, 200_000, 500_000, 1_000_000])
    ap.add_argument("--windows", type=int, default=4, help="per tier; the first half dev, the rest test")
    ap.add_argument("--per-window", type=int, default=6)
    ap.add_argument("--exclude", nargs="*", default=[], help="more manifests whose questions are not drawn")
    ap.add_argument("--tier-windows", nargs="*", default=[], help="tier:count overrides, e.g. 1000000:2")
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    r = random.Random(args.seed)
    examples = load(f"{LOC}/hotpot")
    used = set()
    for s in "ABCDF":
        path = f"{LOC}/{s}/manifest.json"
        if os.path.exists(path):
            used |= {q.get("event") for q in json.load(open(path, encoding="utf-8"))["questions"]}
    for path in args.exclude:
        used |= {q.get("event") for q in json.load(open(path, encoding="utf-8"))["questions"]}
    per_tier = {int(t): int(c) for t, c in (x.split(":") for x in args.tier_windows)}
    pool = [e for e in examples if e["_id"] not in used and e["supporting_facts"]]
    r.shuffle(pool)
    # filler paragraphs: every context paragraph of the questions not hosted
    hosted_n = sum(per_tier.get(t, args.windows) for t in args.tiers) * args.per_window
    filler_src = pool[hosted_n:]
    fillers = [(t, s) for e in filler_src for t, s in e["context"]]
    r.shuffle(fillers)
    cost = lambda lines: len(tok.encode(json.dumps("\n".join(lines), ensure_ascii=False)[1:-1]).ids)
    windows, questions, fi = [], [], 0
    host = iter(pool)
    for tier in args.tiers:
        count = per_tier.get(tier, args.windows)
        for w in range(count):
            wid = f"p{tier // 1000:04}k-{w}"
            split = "dev" if w < count // 2 else "test"
            hosted, paras, titles = [], [], set()
            while len(hosted) < args.per_window:
                e = next(host)
                if any(t in titles for t, _ in e["context"]):
                    continue
                hosted.append(e)
                for t, s in e["context"]:
                    titles.add(t)
                    paras.append((t, s))
            total = sum(cost(paragraph_lines(t, s)) + 2 for t, s in paras)
            while total < tier:
                t, s = fillers[fi % len(fillers)]
                fi += 1
                if t in titles:
                    continue
                titles.add(t)
                paras.append((t, s))
                total += cost(paragraph_lines(t, s)) + 2
            r.shuffle(paras)
            lines, where = [], {}
            for t, s in paras:
                if lines:
                    lines.append("")
                lines.append(f"# {nfc(t)}")
                for i, sentence in enumerate(s):
                    if sentence.strip():
                        where[(t, i)] = len(lines)
                        lines.append(nfc(" ".join(sentence.split())))
            state = "\n".join(lines)
            windows.append({"id": wid, "tier": tier, "split": split, "lines": len(lines), "tokens": total})
            for n, e in enumerate(hosted):
                gold = sorted({where[(t, i)] for t, i in e["supporting_facts"] if (t, i) in where})
                if not gold:
                    continue
                question = nfc(e["question"])
                questions.append({
                    "id": f"{wid}-{n}", "family": "prose", "window": wid, "split": split,
                    "kind": classify(question, lines, gold), "absent": False, "segments": tier,
                    "state": state, "instruction": INSTRUCTION.format(question=question),
                    "question": question, "answer": e["answer"], "targets": gold, "distractors": [],
                    "event": e["_id"]})
            print(f"{wid} {split:4s} {len(lines):6d} lines ~{total:8d} tokens, {len(hosted)} questions", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "windows": windows, "questions": questions}, f)
    print(len(questions), "questions")


if __name__ == "__main__":
    main()
