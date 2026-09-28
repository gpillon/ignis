"""EXPERIMENT (branch locate-long-context): set R2, spec 21's confirmatory
real-log set — targets chosen by rule, stratified by near-duplicates.

Two sources: the owner's cluster (a fresh `kubectl logs --since` capture
merged by `prodset.py timeline`, never committed) and the public LogHub
2k samples (one system per window, lines as they are). For each window:

- every line's **siblings**: the other lines whose word sets — times,
  numbers, ids and hashes removed — have a Jaccard of at least 0.5 with it;
- lines whose text, times and numbers removed, occurs twice are excluded
  (no question can single one out);
- targets are drawn, seeded, round-robin over the sibling bins 0, 1-5,
  6-50, > 50, as far as the window has them.

`show` prints each target with its three nearest siblings, for writing a
question that singles it out; `build` checks the questions as
`prodset.py build` does, with the combo rule for every sibling-rich target
(the shared words, all together, in no other line) and paraphrases only
for targets without siblings.
"""

import argparse
import json
import os
import random
import re

from common import content_stems, paraphrase_clean, rare_shared

WORD = re.compile(r"[A-Za-z0-9_]+")
BINS = ((0, 0), (1, 5), (6, 50), (51, 10 ** 9))
STRIP = re.compile(r"\d{4}[-/]\d\d[-/]\d\d[T ][\d:.,]+\S*|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                   r"|\b\d+(\.\d+)*\b|\b[0-9a-f]{7,}\b", re.I)


def cost_of(tok, lines):
    return [len(tok.encode(json.dumps(line, ensure_ascii=False)[1:-1] + "\\n").ids) for line in lines]


def cut(lines, cost, budget, r):
    for _ in range(200):
        start = r.randrange(max(1, len(lines) - 10))
        total, end = 0, start
        while end < len(lines) and total + cost[end] <= budget:
            total += cost[end]
            end += 1
        if total >= 0.95 * budget:
            return start, end, total
    return None


def siblings(lines):
    """Near-duplicates on the text with times, numbers, ids and hashes
    removed: two lines of one template differ in those, and nothing else."""
    sets = [set(WORD.findall(STRIP.sub(" ", line).lower())) for line in lines]
    out = []
    for i, a in enumerate(sets):
        n = 0
        for j, b in enumerate(sets):
            if i != j and a and len(a & b) / len(a | b) >= 0.5:
                n += 1
        out.append(n)
    return out


def windows(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    r = random.Random(args.seed)
    out = []
    with open(args.timeline, encoding="utf-8") as f:
        stream = [line for line in f.read().rstrip("\n").split("\n") if len(line) <= 1200]
    cost = cost_of(tok, stream)
    for tier in args.cluster_tiers:
        for w in range(args.per_tier):
            got = cut(stream, cost, tier, r)
            out.append({"id": f"c{tier // 1000:03}k-{w}", "source": "cluster", "file": args.timeline,
                        "tier": tier, "start": got[0], "end": got[1], "tokens": got[2]})
    for path in sorted(os.listdir(args.loghub)):
        with open(os.path.join(args.loghub, path), encoding="utf-8", errors="replace") as f:
            lines = [line for line in f.read().rstrip("\n").split("\n") if line.strip()]
        c = cost_of(tok, lines)
        total = sum(c)
        tier = max((t for t in args.loghub_tiers if t <= total), default=None)
        if tier is None:
            continue
        got = cut(lines, c, tier, r)
        out.append({"id": f"p-{path.split('_')[0].lower()}", "source": "loghub", "file": os.path.join(args.loghub, path),
                    "tier": tier, "start": got[0], "end": got[1], "tokens": got[2]})
    for w in out:
        lines = window_lines(w)
        sib = siblings(lines)
        norm = [STRIP.sub(" ", line) for line in lines]
        seen = {}
        for n in norm:
            seen[n] = seen.get(n, 0) + 1
        eligible = [i for i in range(len(lines)) if seen[norm[i]] == 1 and 20 <= len(lines[i]) <= 600]
        by_bin = [[i for i in eligible if lo <= sib[i] <= hi] for lo, hi in BINS]
        for b in by_bin:
            r.shuffle(b)
        want = args.cluster_targets if w["source"] == "cluster" else args.loghub_targets
        picks = []
        while len(picks) < want and any(by_bin):
            for b, pool in enumerate(by_bin):
                if pool and len(picks) < want:
                    i = pool.pop()
                    picks.append({"line": i, "siblings": sib[i], "bin": b})
        w["targets"] = picks
        w["lines"] = len(lines)
        print(f"{w['id']:14s} {w['tier']:>6} tokens {w['tokens']:>6} lines {len(lines):>5} targets "
              + " ".join(f"{p['line']}(s{p['siblings']})" for p in picks))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "windows": out}, f, indent=1)


def window_lines(w):
    with open(w["file"], encoding="utf-8", errors="replace") as f:
        lines = f.read().rstrip("\n").split("\n")
    if w["source"] == "cluster":
        lines = [line for line in lines if len(line) <= 1200]
    else:
        lines = [line for line in lines if line.strip()]
    return lines[w["start"]:w["end"]]


def show(args):
    with open(args.windows, encoding="utf-8") as f:
        meta = json.load(f)
    for w in meta["windows"]:
        if args.window and w["id"] not in args.window:
            continue
        lines = window_lines(w)
        sets = [set(WORD.findall(STRIP.sub(" ", line).lower())) for line in lines]
        print(f"=== {w['id']} ({w['lines']} lines)")
        for p in w["targets"]:
            i = p["line"]
            near = sorted(((len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])), j)
                           for j in range(len(lines)) if j != i), reverse=True)[:3]
            print(f"  T {i} (siblings {p['siblings']}): {lines[i][:args.width]}")
            for s, j in near:
                print(f"     ~{s:.2f} {j}: {lines[j][:args.width]}")


def build(args):
    with open(args.windows, encoding="utf-8") as f:
        meta = {w["id"]: w for w in json.load(f)["windows"]}
    with open(args.questions, encoding="utf-8") as f:
        asked = json.load(f)["questions"]
    rows, problems = [], []
    for n, q in enumerate(asked):
        w = meta[q["window"]]
        lines = window_lines(w)
        t = q["target"]
        target, rest = lines[t], lines[:t] + lines[t + 1:]
        sib = next((p["siblings"] for p in w["targets"] if p["line"] == t), None)
        if q["split"] == "lexical" and not rare_shared(q["instruction"], target, rest):
            problems.append(f"{q['window']}:{t} lexical question shares no rare word")
        if q["split"] == "paraphrase":
            if not paraphrase_clean(q["instruction"], target):
                problems.append(f"{q['window']}:{t} paraphrase shares a content word")
            if sib:
                problems.append(f"{q['window']}:{t} paraphrase on a target with {sib} siblings")
        if q["split"] in ("combo", "lexical"):
            shared = content_stems(q["instruction"]) & content_stems(target)
            also = [i for i, line in enumerate(lines) if i != t and shared <= content_stems(line)]
            if not shared or also:
                problems.append(f"{q['window']}:{t} {q['split']}: shared {sorted(shared)} also in {also[:5]}")
        absent = bool(q.get("absent"))
        state = rest if absent else lines
        rows.append({"id": f"r2-{q['window']}-{n:02}", "source": w["source"], "split": q["split"], "absent": absent,
                     "segments": w["tier"], "siblings": sib, "window": q["window"], "depth": t / len(lines),
                     "state": "\n".join(state), "instruction": q["instruction"],
                     "targets": [] if absent else [t], "distractors": []})
    for p in problems:
        print("PROBLEM", p)
    if problems and not args.force:
        raise SystemExit(f"{len(problems)} problems")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "questions": rows}, f, indent=1)
    print(f"{len(rows)} questions -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--timeline", required=True)
    w.add_argument("--loghub", required=True)
    w.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--cluster-tiers", type=int, nargs="+", default=[16_000, 50_000, 100_000, 200_000])
    w.add_argument("--loghub-tiers", type=int, nargs="+", default=[16_000, 50_000, 100_000])
    w.add_argument("--per-tier", type=int, default=2)
    w.add_argument("--cluster-targets", type=int, default=4)
    w.add_argument("--loghub-targets", type=int, default=3)
    w.add_argument("--out", required=True)
    s = sub.add_parser("show")
    s.add_argument("--windows", required=True)
    s.add_argument("--window", nargs="*")
    s.add_argument("--width", type=int, default=200)
    b = sub.add_parser("build")
    b.add_argument("--windows", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--seed", type=int, default=20261040)
    b.add_argument("--out", required=True)
    b.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"windows": windows, "show": show, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
