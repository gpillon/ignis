"""EXPERIMENT (branch locate-long-context): a locate set from real logs.

The logs are read from a directory of `kubectl logs` captures (one file per
pod) that lives in `.scratch/` and is never committed; so is every set this
writes. Nothing here names a source: a source's label is derived from its
file name at run time.

- `timeline`: every capture merged into one time-ordered stream, each line
  prefixed with its pod's short name (`[name] `), as stern or Loki would show
  it. Lines without a parsable ISO timestamp keep their capture's order.
- `windows`: for each length tier, contiguous windows of the stream cut to a
  token budget (the render's own tokens: the state is a JSON string), and for
  each window the **candidate targets**: lines whose template -- numbers,
  times, ids, hashes and addresses folded -- occurs once in the window.
- `build`: joins hand-written questions (`questions.json`: window, target
  line, split, instruction) with their windows into a `long.py` manifest,
  checking each: the target's template is unique in its window; a lexical
  question shares a rare word with the target and no other line; a
  paraphrase question shares no content word with it (`common.py`'s rules);
  a **combo** question shares only common words with it, and no other line
  holds all of them (the dense case real logs are made of).
  An absent question is the same window with the target line removed.
"""

import argparse
import glob
import json
import os
import random
import re

from common import content_stems, paraphrase_clean, rare_shared

TIERS = (4_000, 16_000, 50_000, 100_000, 200_000)
ISO = re.compile(r"(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d(?:\.\d+)?)")


def label_of(path):
    """A pod's short name: the file is `<ns>__<pod>.log`; drop the replica
    set and pod hashes."""
    pod = os.path.basename(path).split("__", 1)[-1].rsplit(".log", 1)[0]
    parts = pod.split("-")
    while len(parts) > 1 and (re.fullmatch(r"[a-z0-9]{5}|[a-z0-9]{8,10}|\d+", parts[-1])
                              and re.search(r"\d", parts[-1]) or parts[-1].count(".") > 1):
        parts.pop()
    return "-".join(parts)[:32]


def template(line):
    t = ISO.sub("<T>", line)
    t = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<U>", t)
    t = re.sub(r"\b\d+(\.\d+){3}(:\d+)?\b", "<IP>", t)
    t = re.sub(r"\b[0-9a-f]{7,}\b", "<H>", t)
    t = re.sub(r"\d+(\.\d+)?", "<N>", t)
    return t


def timeline(args):
    rows = []
    for path in sorted(glob.glob(os.path.join(args.logs, "*.log"))):
        label = label_of(path)
        last = ""
        with open(path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f.read().split("\n")):
                line = line.rstrip()
                # `kubectl logs --timestamps`: the capture's own RFC 3339
                # prefix orders the stream and is not shown
                prefix = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z) (.*)", line)
                if prefix:
                    stamp, line = prefix.group(1), prefix.group(2)
                if not line.strip():
                    continue
                m = ISO.search(line)
                if not prefix:
                    stamp = f"{m.group(1)}T{m.group(2)}" if m else last
                last = stamp
                rows.append((stamp, label, i, f"[{label}] {line}"))
    rows.sort()
    with open(args.out, "w", encoding="utf-8") as f:
        for stamp, label, i, line in rows:
            f.write(line + "\n")
    print(f"{len(rows)} lines -> {args.out}")


def windows(args):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    with open(args.timeline, encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    lines = [line for line in lines if len(line) <= args.max_chars]
    # the render's tokens per line: the line JSON-escaped, plus its `\n`
    cost = [len(tok.encode(json.dumps(line, ensure_ascii=False)[1:-1] + "\\n").ids) for line in lines]
    r = random.Random(args.seed)
    out = []
    for tier in args.tiers:
        for w in range(args.per_tier):
            for _ in range(100):
                start = r.randrange(len(lines))
                total, end = 0, start
                while end < len(lines) and total + cost[end] <= tier:
                    total += cost[end]
                    end += 1
                if total >= 0.97 * tier:
                    break
            else:
                raise SystemExit(f"no window of {tier} tokens")
            window = lines[start:end]
            counts = {}
            for line in window:
                counts[template(line)] = counts.get(template(line), 0) + 1
            cands = [i for i, line in enumerate(window) if counts[template(line)] == 1]
            out.append({"id": f"w{tier // 1000:03}k-{w}", "tier": tier, "start": start, "end": end,
                        "tokens": total, "lines": len(window), "candidates": cands})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"timeline": args.timeline, "max_chars": args.max_chars, "seed": args.seed,
                   "windows": out}, f, indent=1)
    for w in out:
        print(f"{w['id']}: lines {w['start']}..{w['end']} ({w['lines']} lines, {w['tokens']} tokens), "
              f"{len(w['candidates'])} unique-template lines")


def window_lines(meta, w):
    with open(meta["timeline"], encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    lines = [line for line in lines if len(line) <= meta["max_chars"]]
    return lines[w["start"]:w["end"]]


def show(args):
    with open(args.windows, encoding="utf-8") as f:
        meta = json.load(f)
    w = next(w for w in meta["windows"] if w["id"] == args.window)
    lines = window_lines(meta, w)
    for i in w["candidates"]:
        print(f"{i:5d} {lines[i][:args.width]}")


def build(args):
    with open(args.windows, encoding="utf-8") as f:
        meta = json.load(f)
    with open(args.questions, encoding="utf-8") as f:
        asked = json.load(f)["questions"]
    by_id = {w["id"]: w for w in meta["windows"]}
    rows, problems = [], []
    for n, q in enumerate(asked):
        w = by_id[q["window"]]
        lines = window_lines(meta, w)
        t = q["target"]
        target = lines[t]
        rest = lines[:t] + lines[t + 1:]
        same = [i for i, line in enumerate(lines) if template(line) == template(target)]
        if q.get("unique", "template") == "template" and len(same) != 1:
            problems.append(f"{q['window']}:{t} template occurs {len(same)} times")
        if q["split"] == "lexical" and not rare_shared(q["instruction"], target, rest):
            problems.append(f"{q['window']}:{t} lexical question shares no rare word")
        if q["split"] == "paraphrase" and not paraphrase_clean(q["instruction"], target):
            problems.append(f"{q['window']}:{t} paraphrase shares a content word")
        if q["split"] == "combo":
            # real logs' commonest question: every shared word is common,
            # only their combination singles the target out
            shared = content_stems(q["instruction"]) & content_stems(target)
            also = [i for i, line in enumerate(lines) if i != t and shared <= content_stems(line)]
            if not shared or also or rare_shared(q["instruction"], target, rest):
                problems.append(f"{q['window']}:{t} combo: shared {sorted(shared)}, also in {also[:5]}")
        absent = bool(q.get("absent"))
        state = rest if absent else lines
        rows.append({"id": f"prod-{w['tier'] // 1000:03}k-{n:02}", "family": "prodlogs", "split": q["split"],
                     "absent": absent, "segments": w["tier"], "lines": len(state), "window": q["window"],
                     "depth": t / len(lines), "state": "\n".join(state), "instruction": q["instruction"],
                     "targets": [] if absent else [t], "distractors": []})
    for p in problems:
        print("PROBLEM", p)
    if problems and not args.force:
        raise SystemExit(f"{len(problems)} problems")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": meta["seed"], "source": "prod logs (never committed)", "questions": rows}, f, indent=1)
    print(f"{len(rows)} questions -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timeline")
    t.add_argument("--logs", required=True)
    t.add_argument("--out", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--timeline", required=True)
    w.add_argument("--tokenizer", required=True)
    w.add_argument("--seed", type=int, required=True)
    w.add_argument("--tiers", type=int, nargs="+", default=list(TIERS))
    w.add_argument("--per-tier", type=int, default=3)
    w.add_argument("--max-chars", type=int, default=1200)
    w.add_argument("--out", required=True)
    s = sub.add_parser("show")
    s.add_argument("--windows", required=True)
    s.add_argument("--window", required=True)
    s.add_argument("--width", type=int, default=260)
    b = sub.add_parser("build")
    b.add_argument("--windows", required=True)
    b.add_argument("--questions", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"timeline": timeline, "windows": windows, "show": show, "build": build}[args.cmd](args)


if __name__ == "__main__":
    main()
