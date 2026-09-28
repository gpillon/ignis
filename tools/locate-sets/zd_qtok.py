"""EXPERIMENT (branch locate-long-context, exploratory): the served heads'
rows read at the instruction's own tokens, not only at the scaffold
(ICR-style multi-position reading; "kind x value").

For each present question: one `locate` per position — the scaffold's last
token (cut 0), then each token of the instruction's text — through the
server's `cut_tail` control (tokens dropped from the prompt's end, the
content-free twin cut the same). Each dump is reduced to the key features of
`zd_cache.key_features` and deleted.

    python zd_qtok.py --set <R2> --out <dir> [--ids ...]
"""

import argparse
import glob
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
from tokenizers import Tokenizer

import long_rows as LR
from zd_cache import key_features

KIND = ("Find the one line of the evidence that the instruction asks for. Answer with only a JSON object "
        "{\"quote\": \"<that line, copied exactly>\"}.")
TEMPLATE_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
OPENING = 3  # {" quote ":"


def post(url, body, timeout=3600):
    """One `/v1/decide` request; an answer turned away with `engine_full`
    (another client holds every lane) is asked again, up to six times."""
    for attempt in range(6):
        request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, payload = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, payload = error.code, json.load(error)
        full = any(isinstance(a, dict) and a.get("code") == "engine_full"
                   for a in (payload.get("answers") or {}).values())
        if not full:
            break
        time.sleep(5 * (attempt + 1))
    return status, payload, (time.perf_counter() - started) * 1e3


def positions(tok, instruction):
    """(cut_tail, token text) for the scaffold and each instruction token."""
    head = KIND + "\n\n{\"instruction\":"
    value = json.dumps(instruction, ensure_ascii=False)
    text = head + value + "}" + TEMPLATE_TAIL
    enc = tok.encode(text, add_special_tokens=False)
    offsets, pieces = enc.offsets, enc.tokens
    n = len(enc.ids)
    lo, hi = len(head) + 1, len(head) + len(value) - 1  # the instruction's characters
    out = [(0, "<scaffold>")]
    for i, (a, b) in enumerate(offsets):
        if b > lo and a < hi:
            out.append((n - (i + 1) + OPENING, pieces[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    ap.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--max-tier", type=int, help="only questions whose window is at most this many tokens")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    os.makedirs(args.out, exist_ok=True)
    for q in questions:
        if q["absent"] or (args.ids and q["id"] not in args.ids) or (args.max_tier and q["segments"] > args.max_tier):
            continue
        path = os.path.join(args.out, q["id"] + ".npz")
        if os.path.exists(path):
            continue
        arrays, log = {}, []
        for n, (cut, piece) in enumerate(positions(tok, q["instruction"])):
            with open(args.control, "w", encoding="utf-8") as f:
                json.dump({"cut_tail": cut, "tag": f"cut{cut}"}, f)
            before = set(glob.glob(os.path.join(args.raw, "*.json")))
            body = {"state": q["state"], "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
            status, payload, ms = post(args.url, body)
            new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
            if len(new) != 1:
                log.append({"cut": cut, "piece": piece, "status": status, "error": f"{len(new)} dumps"})
                continue
            base = os.path.splitext(new[0])[0]
            meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
            keys = [tuple(k) if k else None for k in meta["keys"]]
            fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
            for k in ("sum", "last", "next1", "sep"):
                arrays[f"p{n}/{k}/q"] = fq[k].astype(np.float16)
                arrays[f"p{n}/{k}/na"] = fn[k].astype(np.float16)
            arrays[f"p{n}/argmax"] = qr.argmax(axis=1).astype(np.int32)
            log.append({"n": n, "cut": cut, "piece": piece, "status": status, "ms": ms,
                        "winner": meta.get("winner")})
            for ext in (".json", ".bin"):
                os.remove(base + ext)
        arrays["log"] = np.array(json.dumps(log))
        np.savez_compressed(path, **arrays)
        with open(args.control, "w", encoding="utf-8") as f:
            json.dump({}, f)
        hits = [r.get("winner") in q["targets"] for r in log if "winner" in r]
        print(f"{q['id']} {len(log)} positions, {sum(r.get('ms', 0) for r in log) / 1e3:.0f} s, "
              f"vote hits at scaffold {hits[:1]} and at instruction tokens {sum(hits[1:])}/{len(hits) - 1}", flush=True)


if __name__ == "__main__":
    main()
