"""Spec #555's phase gate (GitHub #560): Flash-Next's decode with vision on and no image in flight at least 99%
of vision off, one lane and three, measured once at the end of the phase.

    python tools/flash-next-vision/decode_gate.py run --url http://127.0.0.1:8000 --label <leg> --vision on|off
                                                  --out <dir> [--server-log <ignis-server.log>]
                                                  [--max-tokens 512] [--rounds 2]
    python tools/flash-next-vision/decode_gate.py verdict --legs <off-1> <on-1> <on-2> <off-2> [--out <json>]

A leg is one server started with `make start MODEL=flash-next`, every knob the same in every leg but
`VISION=0` or `VISION=1` (docs/agents/testing/vision-canary.md has the commands). `run` measures it, text
only, through `POST /v1/chat/completions`, streaming: every request greedy (`temperature` 0), thinking off,
`ignore_eos` with `max_tokens` 512, so every request decodes exactly 512 tokens in every leg.
- C=1: the nine `PROMPTS` one after another; a request's decode rate is its tokens after the first over the
  time from its first content chunk to its last; the cell is their mean.
- C=3: the nine prompts through three clients at once (three waves of three, Flash-Next's three lanes on the
  RTX 5090), `--rounds` times; the cell is the mean over rounds of the completion tokens over the round's wall
  time.
It writes `<out>/<label>.json`: both cells, every request's figures and text, the served model, and with
`--server-log` the last `ignis.runtime.flash_next_plan` and `ignis.runtime.flash_next_host_plan` the log holds
(the leg's own when the log was started with it, as `make start` does).

`verdict` takes four legs in ABBA order (off, on, on, off: a drift common to both arms falls on both) and
passes when, at C=1 and at C=3 alike, each adjacent pair's on/off ratio is at least `GATE` (the rule is on the
pairs, not on legs far apart in time), the expert cache is the same in every leg, and every prompt's text is
the same in every leg (vision on changes nothing a text prompt computes). The pooled ratio, the mean of the
on legs over the mean of the off legs, is reported beside them. Standard library only.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

GATE = 0.99
LANES = 3
SCHEMA = "ignis-fn-vision-decode-leg/1"
TIMEOUT_SECONDS = 1800
PLAN_EVENTS = ("ignis.runtime.flash_next_plan", "ignis.runtime.flash_next_host_plan")
# Nine coding asks with nothing in common: the kind of request decode is measured on (#335's, #417's).
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string such as 'P3DT4H12M' into a "
    "datetime.timedelta, with a docstring and a few doctest examples.",
    "Explain, step by step, how a paged KV cache with 64-token pages is allocated and freed in an LLM inference "
    "server, and what fragmentation it avoids.",
    "Write a bash script that finds the ten largest files under a directory, prints them with human-readable "
    "sizes, and exits non-zero if any is larger than 1 GiB.",
    "Write a TypeScript component that renders a sortable table of {name, size, modified} rows with keyboard "
    "navigation.",
    "Write a SQL schema for a multi-tenant issue tracker with labels, comments and audit history, then three "
    "example queries.",
    "Write a C function that reverses a singly linked list in place, and a small test that builds a list of five "
    "nodes, reverses it and prints it.",
    "Explain the difference between a mutex and a semaphore, then show a short Rust example of each.",
    "Write a Python class implementing an LRU cache with O(1) get and put, and unit tests for its eviction order.",
    "Write a Go HTTP handler that accepts a JSON body with a list of integers and returns their median, with "
    "input validation and a table-driven test.",
]


class Refused(Exception):
    """Legs the verdict cannot judge."""


def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def stream(url, model, prompt, max_tokens):
    """One streamed request: (completion tokens, first and last content chunk times, end time, text)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "enable_thinking": False, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="POST")
    first = last = None
    text, usage = [], None
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    text.append(piece)
    end = time.perf_counter()
    tokens = (usage or {}).get("completion_tokens", 0)
    return {"tokens": tokens, "first": first, "last": last, "started": started, "end": end, "text": "".join(text)}


def last_plans(path):
    """The last attributes of each plan event in a JSON-lines server log."""
    found = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"event_name"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event_name") in PLAN_EVENTS:
                found[event["event_name"]] = event.get("attributes", {})
    return found


def run(args):
    model = http_json(args.url.rstrip("/") + "/v1/models")["data"][0]
    stream(args.url, model["id"], "Say hello.", 8)
    leg = {"schema": SCHEMA, "label": args.label, "vision": args.vision == "on", "model": model["id"],
           "served": model, "max_tokens": args.max_tokens, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "texts": {}}
    requests = []
    for i, prompt in enumerate(PROMPTS):
        r = stream(args.url, model["id"], prompt, args.max_tokens)
        span = (r["last"] - r["first"]) if r["first"] is not None else 0
        rate = (r["tokens"] - 1) / span if span > 0 and r["tokens"] > 1 else 0.0
        requests.append({"prompt": i, "tokens": r["tokens"], "tok_s": rate,
                         "ttft_s": (r["first"] - r["started"]) if r["first"] is not None else None})
        leg["texts"][str(i)] = r["text"]
        print(f"{args.label} c1 #{i}: {r['tokens']} tokens, {rate:.1f} tok/s", flush=True)
    leg["c1"] = {"tok_s": sum(r["tok_s"] for r in requests) / len(requests), "requests": requests}
    rounds = []
    for k in range(args.rounds):
        lock, texts = threading.Lock(), {}

        def one(i):
            r = stream(args.url, model["id"], PROMPTS[i], args.max_tokens)
            with lock:
                texts[str(i)] = r["text"]
            return r
        t0 = time.perf_counter()
        with ThreadPoolExecutor(LANES) as pool:
            results = list(pool.map(one, range(len(PROMPTS))))
        wall = time.perf_counter() - t0
        total = sum(r["tokens"] for r in results)
        rounds.append({"tokens": total, "wall_s": wall, "tok_s": total / wall,
                       "texts_as_c1": all(texts[i] == leg["texts"][i] for i in texts)})
        print(f"{args.label} c3 round {k}: {total} tokens in {wall:.2f} s = {total / wall:.1f} tok/s", flush=True)
    leg["c3"] = {"tok_s": sum(r["tok_s"] for r in rounds) / len(rounds), "rounds": rounds}
    if args.server_log:
        plans = last_plans(args.server_log)
        leg["plan"] = plans.get(PLAN_EVENTS[0])
        leg["host_plan"] = plans.get(PLAN_EVENTS[1])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.label}.json"), "w", encoding="utf-8") as f:
        json.dump(leg, f, indent=1)
    print(f"{args.label}: C=1 {leg['c1']['tok_s']:.1f} tok/s, C=3 {leg['c3']['tok_s']:.1f} tok/s", flush=True)
    return 0


def verdict(paths):
    """Four legs, off-on-on-off, judged by the module docstring's rule."""
    legs = [json.load(open(p, encoding="utf-8")) for p in paths]
    if [leg["vision"] for leg in legs] != [False, True, True, False]:
        raise Refused("the legs must be off, on, on, off; they are "
                      + ", ".join("on" if leg["vision"] else "off" for leg in legs))
    out = {"gate": GATE, "legs": [leg["label"] for leg in legs], "reasons": []}
    for cell in ("c1", "c3"):
        rates = [leg[cell]["tok_s"] for leg in legs]
        pairs = [(legs[0], legs[1]), (legs[3], legs[2])]
        ratios = [on[cell]["tok_s"] / off[cell]["tok_s"] for off, on in pairs]
        out[cell] = {"tok_s": rates, "pairs": ratios, "pooled": (rates[1] + rates[2]) / (rates[0] + rates[3])}
        for (off, on), ratio in zip(pairs, ratios):
            if ratio < GATE:
                out["reasons"].append(f"{cell}: {on['label']} is {ratio:.2%} of {off['label']}, below {GATE:.0%}")
    caches = [(leg.get("plan") or {}).get("expert_cache_bytes") for leg in legs]
    out["expert_cache_bytes"] = caches
    out["expert_cache_equal"] = None if None in caches else len(set(caches)) == 1
    if out["expert_cache_equal"] is None:
        out["reasons"].append("a leg has no plan: pass its server log to `run --server-log`")
    elif not out["expert_cache_equal"]:
        out["reasons"].append(f"the expert cache differs between legs: {caches}")
    differing = sorted(i for i in legs[0]["texts"] if len({leg["texts"].get(i) for leg in legs}) != 1)
    out["texts_identical"] = not differing
    if differing:
        out["reasons"].append(f"prompts {', '.join(differing)} decode other text in some leg")
    out["pass"] = not out["reasons"]
    return out


def run_verdict(args):
    out = verdict(args.legs)
    for cell in ("c1", "c3"):
        c = out[cell]
        print(f"{cell}: tok/s {', '.join(f'{r:.1f}' for r in c['tok_s'])}; on/off pairs "
              f"{', '.join(f'{r:.2%}' for r in c['pairs'])}; pooled {c['pooled']:.2%}", flush=True)
    print(f"expert cache equal: {out['expert_cache_equal']}; texts identical: {out['texts_identical']}", flush=True)
    print("PASS" if out["pass"] else "FAIL: " + "; ".join(out["reasons"]), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0 if out["pass"] else 1


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="measure one leg against a running server")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--vision", required=True, choices=("on", "off"))
    r.add_argument("--out", required=True)
    r.add_argument("--server-log")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--rounds", type=int, default=2)
    r.set_defaults(fn=run)
    v = sub.add_parser("verdict", help="four legs, off on on off, against the gate")
    v.add_argument("--legs", nargs=4, required=True)
    v.add_argument("--out")
    v.set_defaults(fn=run_verdict)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
