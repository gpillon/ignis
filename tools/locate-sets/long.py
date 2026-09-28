"""EXPERIMENT (branch locate-long-context, never merged): does `locate` hold
past the length it was measured to (`LOCATE_MAX_KEYS`, 4,554 keys)?

One family, the one long states are made of: synthetic service logs
(`logs.py`, unchanged events and routine traffic, 3 to 8 ERROR distractors
whatever the length), at 250 / 1,000 / 3,000 / 6,000 / 12,000 lines -- about
4.4K / 17.7K / 53K / 106K / 212K state tokens. The only thing that moves is
the length; the target's **depth** is stratified (two present questions per
decile of the log), half lexical and half paraphrase, and one question in
six absent.

`ask` puts each question to a server built from this branch and started
with `IGNIS_LOCATE_MAX_KEYS_EXPERIMENT` (the ceiling lifted) and
`IGNIS_LOCATE_DUMP_DIR` (both prefills' rows of the 32 heads dumped per
question): first as a `locate` through `/v1/decide`, then as the same render
through `/v1/chat/completions`, greedy, thinking off -- what the model
itself answers when it writes the line out (the generation route).

    python long.py generate --seed 20261030 --out .scratch/locate/long/L
    python long.py ask --set .scratch/locate/long/L --dumps <IGNIS_LOCATE_DUMP_DIR> --out L-served.json
    python long.py report --set .scratch/locate/long/L --served L-served.json --dumps <dir> --out L-report.json
"""

import argparse
import glob
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request

from common import assign_absent, paraphrase_clean, rare_shared
from generation import occurrences, parse
import logs

LENGTHS = (250, 1000, 3000, 6000, 12000)
PER_LENGTH = 24
KIND_LINE = ("Find the one line of the evidence that the instruction asks for. Answer with only "
             "a JSON object {\"quote\": \"<that line, copied exactly>\"}.")
FOUND = "Is there a line in the evidence that answers this question: {}"


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------

def one_log_at(r, length, event, split, absent, target_at):
    """`logs.one_log` with the target's line chosen by the caller."""
    others = [key for key in logs.EVENTS if key != event]
    distractors = r.sample(others, r.randint(3, 8))
    free = [i for i in range(length) if i != target_at]
    distractor_at = r.sample(free, len(distractors))
    special = {target_at: event, **dict(zip(distractor_at, distractors))}
    clock = r.randint(0, 86399)
    lines = []
    for i in range(length):
        clock += r.randint(0, 20)
        if i in special:
            level, message = "ERROR", logs.EVENTS[special[i]][0](r)
        else:
            level, message = logs._routine(r)
        lines.append(f"{logs._stamp(clock)} {level} {r.choice(logs.SERVICES)}: {message}")
    question = logs.EVENTS[event][1 if split == "lexical" else 2]
    target = lines[target_at]
    rest = lines[:target_at] + lines[target_at + 1:]
    if split == "lexical" and not rare_shared(question, target, rest):
        return None
    if split == "paraphrase" and not paraphrase_clean(question, target):
        return None
    if absent:
        level, message = logs._routine(r)
        lines[target_at] = f"{target[:5]} {level} {r.choice(logs.SERVICES)}: {message}"
    return {"state": "\n".join(lines), "instruction": question,
            "targets": [] if absent else [target_at], "target_at": target_at,
            "distractors": sorted(distractor_at), "event": event}


def generate(args):
    r = random.Random(args.seed)
    rows = []
    for length in args.lengths:
        absent = assign_absent(r, args.per)
        present_seen = 0
        for i in range(args.per):
            split = ("lexical", "paraphrase")[i % 2]
            is_absent = i in absent
            # present questions: two per decile of depth, in order
            decile = (present_seen // 2) % 10 if not is_absent else r.randrange(10)
            if not is_absent:
                present_seen += 1
            event = r.choice(sorted(logs.EVENTS))
            for _ in range(50):
                at = min(length - 1, int((decile + r.random()) / 10 * length))
                row = one_log_at(r, length, event, split, is_absent, at)
                if row is not None:
                    break
            else:
                raise SystemExit(f"{event!r} ({split}) failed its word checks 50 times")
            rows.append({"id": f"long-{length:05}-{i:02}", "family": "logs", "split": split,
                         "absent": is_absent, "segments": length, "depth": at / length, **row})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "lengths": list(args.lengths), "per_length": args.per,
                   "questions": rows}, f, indent=1)
    print(f"{len(rows)} questions -> {args.out}")


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

def post(url, path, body, timeout):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, payload = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.load(error)
    return status, payload, (time.perf_counter() - started) * 1e3


def ask_locate(args, q):
    before = set(glob.glob(os.path.join(args.dumps, "*.json")))
    questions = {"q": {"type": "locate", "instructions": q["instruction"]}}
    if args.found:
        # the "is it there at all?" question, over the same state: a sibling
        # of the locate in one request, so it claims the state's prefix
        questions["f"] = {"type": "noul", "instructions": FOUND.format(q["instruction"])}
    body = {"state": q["state"], "questions": questions}
    status, payload, ms = post(args.url, "/v1/decide", body, args.timeout)
    row = {"status": status, "ms": ms}
    if status == 200:
        row["answer"] = payload["answers"]["q"]
        if args.found:
            row["found"] = payload["answers"].get("f")
        row["input_tokens"] = payload["usage"]["input_tokens"]
    else:
        row["code"] = payload.get("error", {}).get("code")
        row["message"] = payload.get("error", {}).get("message")
    new = sorted(set(glob.glob(os.path.join(args.dumps, "*.json"))) - before)
    if len(new) == 1:
        seq = os.path.splitext(new[0])[0]
        for ext in (".json", ".bin"):
            os.replace(seq + ext, os.path.join(args.dumps, q["id"] + ext))
        row["dump"] = q["id"]
    else:
        row["dump"] = None
        row["dump_files"] = new
    return row


def ask_generation(args, q):
    body = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": "{\"evidence\":" + json.dumps(q["state"], ensure_ascii=False,
                                                                      separators=(",", ":")) + "}"},
            {"role": "user", "content": KIND_LINE + "\n\n{\"instruction\":"
             + json.dumps(q["instruction"], ensure_ascii=False, separators=(",", ":")) + "}"},
        ],
        "temperature": 0,
        "max_tokens": 200,
        "enable_thinking": False,
    }
    status, payload, ms = post(args.url, "/v1/chat/completions", body, args.timeout)
    row = {"status": status, "ms": ms}
    if status != 200:
        row["code"] = payload.get("error", {}).get("code")
        return row
    content = payload["choices"][0]["message"].get("content") or ""
    quotes = parse(content, False)
    row["content"] = content
    row["prompt_tokens"] = payload.get("usage", {}).get("prompt_tokens")
    if quotes:
        match, spans = occurrences(q["state"].split("\n"), quotes[0])
        row["match"] = match
        row["segments"] = sorted({s["segment"] for s in spans})
    else:
        row["match"] = "no quote"
        row["segments"] = []
    return row


def ask(args):
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    done = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {row["id"]: row for row in json.load(f)["questions"]}
    questions = manifest["questions"]
    if args.lengths:
        questions = [q for q in questions if q["segments"] in args.lengths]
    questions = questions[:args.limit] if args.limit else questions
    rows = dict(done)
    for q in questions:
        if q["id"] in done:
            continue
        row = {k: q.get(k) for k in ("id", "split", "absent", "segments", "targets", "depth", "distractors")}
        row["locate"] = ask_locate(args, q)
        if not args.no_generation:
            row["generation"] = ask_generation(args, q)
        rows[q["id"]] = row
        with open(args.out + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"seed": manifest["seed"], "questions": list(rows.values())}, f, indent=1)
        os.replace(args.out + ".tmp", args.out)
        loc, gen = row["locate"], row.get("generation", {})
        a = loc.get("answer", {})
        print(f"  {q['id']} {q['split']:10} {'absent ' if q['absent'] else 'present'} depth {q['depth']:.2f} "
              f"target {q['targets']} | locate {loc['status']} {a.get('segment', loc.get('code'))} "
              f"conf {a.get('confidence', 0):.2f} {loc['ms']:.0f} ms | gen {gen.get('match')} {gen.get('segments')} "
              f"{gen.get('ms', 0):.0f} ms", flush=True)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def report(args):
    with open(args.served, encoding="utf-8") as f:
        rows = json.load(f)["questions"]
    out = {"by_length": {}, "by_depth": {}}
    for length in sorted({r["segments"] for r in rows}):
        sub = [r for r in rows if r["segments"] == length]
        present = [r for r in sub if not r["absent"]]
        def loc_seg(r):
            a = r["locate"].get("answer", {})
            return a.get("segment") if a.get("type") == "locate" else None
        def top3(r):
            a = r["locate"].get("answer", {})
            return [e["segment"] for e in a.get("ranking", [])[:3]]
        hits = [r for r in present if loc_seg(r) in r["targets"]]
        plus1 = [r for r in present if loc_seg(r) is not None and loc_seg(r) - r["targets"][0] == 1]
        gen_hits = [r for r in present if r["targets"][0] in r.get("generation", {}).get("segments", [])]
        out["by_length"][length] = {
            "present": len(present),
            "locate_top1": len(hits),
            "locate_top3": sum(bool(set(top3(r)) & set(r["targets"])) for r in present),
            "locate_plus1": len(plus1),
            "generation_top1": len(gen_hits),
            "errors": [r["id"] for r in sub if r["locate"]["status"] != 200],
            "conf_hit_median": statistics.median([r["locate"]["answer"]["confidence"] for r in hits]) if hits else None,
            "locate_ms_median": statistics.median([r["locate"]["ms"] for r in sub]),
            "input_tokens_median": statistics.median([r["locate"].get("input_tokens", 0) for r in sub]),
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for length, v in out["by_length"].items():
        print(f"{length:6} lines: locate {v['locate_top1']}/{v['present']} (top-3 {v['locate_top3']}, +1 {v['locate_plus1']}) "
              f"| generation {v['generation_top1']}/{v['present']} | {v['locate_ms_median']:.0f} ms, "
              f"{v['input_tokens_median']:.0f} tokens")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    g.add_argument("--per", type=int, default=PER_LENGTH)
    a = sub.add_parser("ask")
    a.add_argument("--set", required=True)
    a.add_argument("--url", default="http://127.0.0.1:8000")
    a.add_argument("--dumps", required=True, help="the server's IGNIS_LOCATE_DUMP_DIR")
    a.add_argument("--out", required=True)
    a.add_argument("--timeout", type=float, default=1800)
    a.add_argument("--limit", type=int)
    a.add_argument("--lengths", type=int, nargs="+")
    a.add_argument("--no-generation", action="store_true")
    a.add_argument("--found", action="store_true", help="ask a noul beside each locate: is the line there at all?")
    r = sub.add_parser("report")
    r.add_argument("--served", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"generate": generate, "ask": ask, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
