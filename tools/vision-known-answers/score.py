"""Images with known answers, scored through a running server's HTTP API (GitHub #560, spec #555's seam 1).

    python tools/vision-known-answers/score.py run --url http://127.0.0.1:8000 --images <dir> --label <label>
                                               --out <dir> [--manifest <manifest.json>] [--model <id>]
                                               [--max-tokens 64]
    python tools/vision-known-answers/score.py compare --flash-next <label>.jsonl --qwen27b <label>.jsonl
                                               [--out <json>]

`run` sends every image of the manifest (`manifest.json` beside this file: the vision canary's four images and
the public ones `fetch.py` downloads) with its question to `POST /v1/chat/completions`, one at a time: the
question then the image as a `data:` URI, `temperature` 0, `enable_thinking` false, `max_tokens` 64. It writes
`<out>/<label>.jsonl` (per image: the reply, its verdict, the tokens, the seconds, the SHA-256 of the bytes
sent) and `<out>/<label>.summary.json` (the served model, the accuracy, per kind, the canaries, the alarms).
A canary answered wrongly prints an ALARM line: it goes to the owner (#560).

The matching rule, fixed before any run: a reply is right when, for every needle group of the image, at least
one of the group's spellings appears in the lowercased reply as a whole word -- not preceded or followed by a
letter or a digit, so "red" is not found in "covered", nor "47" in "147". The canaries' needles are their GPU
test's (`crates/server/tests/flash_next_vision_canary_gpu.rs`: 47; red; 3 or three; config.yaml and not
found). Nothing else is normalised and nothing is judged: a reply naming the answer among others counts as
right, and the questions ask for one word or one number to keep that rare.

`compare` pairs two runs on the images both scored (Flash-Next's and the 27B's, one server at a time): each
accuracy, the images only one got right, the paired bootstrap 95% interval of the accuracy difference
(Flash-Next less the 27B, images resampled, 10,000 draws, seed 560) and the exact McNemar p. Its alarms, for
the owner: a canary Flash-Next answered wrongly, and Flash-Next clearly below the 27B, meaning the interval
lies wholly below zero.

Standard library only: any Python 3.9+.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = "ignis-vision-known-answers/1"
RUN_SCHEMA = "ignis-vision-known-answers-run/1"
KINDS = ("object", "colour", "count", "text", "scene")
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 560
TIMEOUT_SECONDS = 600


class Refused(Exception):
    """An input the scorer cannot use."""


def matches(reply, needles):
    """The matching rule (module docstring): every group has a spelling present as a whole word."""
    text = reply.lower()

    def present(spelling):
        return re.search(r"(?<![0-9a-z])" + re.escape(spelling.lower()) + r"(?![0-9a-z])", text) is not None
    return all(any(present(s) for s in group) for group in needles)


def image_path(item, images):
    """Where an entry's scored image is: a repository file for a canary, `fetch.py`'s copy otherwise."""
    repo = item.get("source", {}).get("repo")
    return os.path.join(REPO_ROOT, repo) if repo else os.path.join(images, item["id"], "scored.png")


def request_body(item, image, mime, model, max_tokens):
    url = f"data:{mime};base64," + base64.b64encode(image).decode()
    return {"model": model, "temperature": 0, "enable_thinking": False, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": item["question"]},
                                                      {"type": "image_url", "image_url": {"url": url}}]}]}


def http_json(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"},
                                     method="GET" if body is None else "POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def load_manifest(path):
    manifest = json.load(open(path, encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise Refused(f"{path} is {manifest.get('schema')!r}, not {SCHEMA}")
    return manifest


def run(args):
    manifest = load_manifest(args.manifest)
    model = args.model or http_json(args.url.rstrip("/") + "/v1/models")["data"][0]["id"]
    os.makedirs(args.out, exist_ok=True)
    rows, alarms = [], []
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(args.out, f"{args.label}.jsonl"), "w", encoding="utf-8") as out:
        for item in manifest["images"]:
            path = image_path(item, args.images)
            if not os.path.isfile(path):
                raise Refused(f"{item['id']}: {path} does not exist (run fetch.py first)")
            image = open(path, "rb").read()
            mime = MIME[os.path.splitext(path)[1].lower()]
            t0 = time.perf_counter()
            reply = http_json(args.url.rstrip("/") + "/v1/chat/completions",
                              request_body(item, image, mime, model, args.max_tokens))
            seconds = time.perf_counter() - t0
            answer = reply["choices"][0]["message"].get("content") or ""
            right = matches(answer, item["needles"])
            usage = reply.get("usage") or {}
            row = {"id": item["id"], "kind": item["kind"], "right": right, "answer": answer,
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "seconds": round(seconds, 3), "image_sha256": hashlib.sha256(image).hexdigest()}
            out.write(json.dumps(row) + "\n")
            out.flush()
            rows.append(row)
            print(f"{item['id']}: {'right' if right else 'WRONG'} -- {answer!r}", flush=True)
            if item["kind"] == "canary" and not right:
                alarms.append(f"canary {item['id']} answered wrongly: {answer!r}")
                print(f"ALARM: canary {item['id']} answered wrongly: {answer!r}", flush=True)
    by_kind = {}
    for row in rows:
        k = by_kind.setdefault(row["kind"], {"images": 0, "right": 0})
        k["images"] += 1
        k["right"] += int(row["right"])
    summary = {"schema": RUN_SCHEMA, "label": args.label, "url": args.url, "model": model, "started": started,
               "manifest_sha256": hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(),
               "max_tokens": args.max_tokens, "images": len(rows), "right": sum(r["right"] for r in rows),
               "accuracy": sum(r["right"] for r in rows) / len(rows), "by_kind": by_kind,
               "canaries": {r["id"]: r["right"] for r in rows if r["kind"] == "canary"}, "alarms": alarms}
    with open(os.path.join(args.out, f"{args.label}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"{args.label} ({model}): {summary['right']}/{summary['images']} right", flush=True)
    return 0


def read_run(path):
    return {row["id"]: row for row in map(json.loads, open(path, encoding="utf-8").read().splitlines())}


def mcnemar_p(b, c):
    """The exact two-sided McNemar p of `b` and `c` discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(flash_next, qwen27b, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """The paired comparison of two runs' `.jsonl` (module docstring)."""
    a, b = read_run(flash_next), read_run(qwen27b)
    ids = [i for i in a if i in b]
    if not ids:
        raise Refused("the two runs share no image")
    x = [int(a[i]["right"]) for i in ids]
    y = [int(b[i]["right"]) for i in ids]
    d = [p - q for p, q in zip(x, y)]
    rng = random.Random(seed)
    n = len(ids)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low, high = means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
    fn_only, b27_only = d.count(1), d.count(-1)
    out = {"images": n, "flash_next": {"right": sum(x), "accuracy": sum(x) / n},
           "qwen27b": {"right": sum(y), "accuracy": sum(y) / n},
           "difference": (sum(x) - sum(y)) / n, "flash_next_only": fn_only, "qwen27b_only": b27_only,
           "bootstrap": {"low": low, "high": high, "resamples": resamples, "seed": seed},
           "mcnemar_p": mcnemar_p(fn_only, b27_only), "alarms": []}
    for i in ids:
        if a[i]["kind"] == "canary" and not a[i]["right"]:
            out["alarms"].append(f"Flash-Next answered canary {i} wrongly: {a[i]['answer']!r}")
    if high < 0:
        out["alarms"].append(f"Flash-Next is clearly below the 27B: {sum(x)}/{n} against {sum(y)}/{n}, the "
                             f"difference's 95% interval [{low:+.3f}, {high:+.3f}] lies below zero")
    out["by_kind"] = {}
    for i in ids:
        k = out["by_kind"].setdefault(a[i]["kind"], {"images": 0, "flash_next": 0, "qwen27b": 0})
        k["images"] += 1
        k["flash_next"] += int(a[i]["right"])
        k["qwen27b"] += int(b[i]["right"])
    return out


def run_compare(args):
    out = compare(args.flash_next, args.qwen27b)
    print(json.dumps({k: v for k, v in out.items() if k != "by_kind"}, indent=1))
    for alarm in out["alarms"]:
        print(f"ALARM: {alarm}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="score one running server")
    r.add_argument("--url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    r.add_argument("--images", required=True, help="fetch.py's --root")
    r.add_argument("--model", help="the model id to ask for (default: the server's first in /v1/models)")
    r.add_argument("--max-tokens", type=int, default=64)
    r.set_defaults(fn=run)
    c = sub.add_parser("compare", help="Flash-Next's run against the 27B's, paired")
    c.add_argument("--flash-next", required=True)
    c.add_argument("--qwen27b", required=True)
    c.add_argument("--out")
    c.set_defaults(fn=run_compare)
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
