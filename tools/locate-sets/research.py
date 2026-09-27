"""EXPERIMENT (branch locate-long-context, exploratory research): what the
heads do at length, where the row's mass goes outside the state, and how the
vote moves while the answer is being copied.

Drives a server built from this branch, started with
`IGNIS_LOCATE_MAX_KEYS_EXPERIMENT`, `IGNIS_LOCATE_DUMP_DIR` and
`IGNIS_LOCATE_EXPERIMENT_FILE` (the control file this script writes before
each request: heads, forced quote text, full row). Every request is a
single `locate`; after the first one over a state the rest claim its prefix
and pay only the tail. Each raw dump is reduced as it lands and deleted:

- `seg_q`, `seg_na` [heads, segments] f16: softmax share of each segment
  over the state's span (the question's prefill, the content-free one's);
- `top_keys`, `top_scores` [heads, 24]: each head's best span keys (question);
- with a full row: `lse` [heads, 3] — log-sum-exp of the question's scores
  over the prompt before the span, the span, and after it — and `tail`
  [heads, width - span_end] f16, every score after the span (template, kind
  text, instruction, scaffold, forced text), plus the same for the baseline.

Configurations per question (`--plan`):
- `heads0`: all 384 heads, full row, at the scaffold (k = 0), 12 requests;
- `heads8`: all 384 heads, full row, 8 tokens of the target copied;
- `traj`: the 32 served heads over the span with k = 1 .. 64 target tokens
  forced after the scaffold (the answer being written, teacher-forced).

    python research.py --set <R> --dumps-raw <dir> --control <file> --out <dir> --plan heads0 traj
"""

import argparse
import glob
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

import long_rows as LR

ALL_HEADS = [f"L{4 * g + 3}.h{h}" for g in range(16) for h in range(24)]
TRAJ_K = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def post(url, body, timeout):
    request = urllib.request.Request(url + "/v1/decide", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, payload = response.status, json.load(response)
    except urllib.error.HTTPError as error:
        status, payload = error.code, json.load(error)
    return status, payload, (time.perf_counter() - started) * 1e3


def reduce(raw_base, keys):
    with open(raw_base + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    raw = np.fromfile(raw_base + ".bin", dtype="<f4")
    heads, qw, nw = meta["heads"], meta["q_width"], meta["na_width"]
    q = raw[:heads * qw].reshape(heads, qw).astype(np.float64)
    na = raw[heads * qw:heads * qw + heads * nw].reshape(heads, nw).astype(np.float64)
    s0 = meta["span_start"] if meta["full_row"] else 0
    s1 = s0 + meta["span"]
    out = {"meta": meta}
    qs, nas = q[:, s0:s1], na[:, s0:s1]
    out["seg_q"] = LR.seg_sum(LR.softmax_rows(qs), keys).astype(np.float16)
    out["seg_na"] = LR.seg_sum(LR.softmax_rows(nas), keys).astype(np.float16)
    top = np.argsort(-qs, axis=1)[:, :24]
    out["top_keys"] = top.astype(np.int32)
    out["top_scores"] = np.take_along_axis(qs, top, axis=1).astype(np.float16)
    if meta["full_row"]:
        def lse(x):
            if x.shape[1] == 0:
                return np.full(x.shape[0], -np.inf)
            m = x.max(axis=1, keepdims=True)
            return (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))
        out["lse"] = np.stack([lse(q[:, :s0]), lse(q[:, s0:s1]), lse(q[:, s1:])], axis=1)
        out["lse_na"] = np.stack([lse(na[:, :s0]), lse(na[:, s0:s1]), lse(na[:, s1:])], axis=1)
        out["tail"] = q[:, s1:].astype(np.float16)
        out["tail_na"] = na[:, s1:].astype(np.float16)
    return out


def escaped_tokens(tok, line):
    """The target line as the model writes it inside the JSON string, token
    by token (text pieces)."""
    text = json.dumps(line, ensure_ascii=False)[1:-1]
    enc = tok.encode(text)
    offsets = enc.offsets
    return [text[a:b] for a, b in offsets], text


def ask(args, q, control, tag):
    control = dict(control, tag=tag)
    with open(args.control, "w", encoding="utf-8") as f:
        json.dump(control, f)
    before = set(glob.glob(os.path.join(args.dumps_raw, "*.json")))
    body = {"state": q["state"], "questions": {"q": {"type": "locate", "instructions": q["instruction"]}}}
    status, payload, ms = post(args.url, body, args.timeout)
    new = sorted(set(glob.glob(os.path.join(args.dumps_raw, "*.json"))) - before)
    answer = payload.get("answers", {}).get("q") if status == 200 else payload.get("error")
    if len(new) != 1:
        return {"tag": tag, "status": status, "answer": answer, "ms": ms, "error": f"{len(new)} dumps"}, None
    base = os.path.splitext(new[0])[0]
    with open(base + ".json", encoding="utf-8") as f:
        keys = json.load(f)["keys"]
    reduced = reduce(base, keys)
    for ext in (".json", ".bin"):
        os.remove(base + ext)
    return {"tag": tag, "status": status, "answer": answer, "ms": ms}, reduced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--dumps-raw", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", nargs="+", default=["heads0", "traj"])
    ap.add_argument("--ids", nargs="*", help="only these question ids")
    ap.add_argument("--present-only", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.present_only:
        questions = [q for q in questions if not q["absent"]]
    os.makedirs(args.out, exist_ok=True)
    for q in questions:
        path = os.path.join(args.out, q["id"] + ".npz")
        if os.path.exists(path):
            continue
        log, arrays = [], {}
        pieces = escaped_tokens(tok, q["state"].split("\n")[q["targets"][0]])[0] if q["targets"] else []
        for plan in args.plan:
            if plan in ("heads0", "heads8"):
                k = 0 if plan == "heads0" else 8
                if k and not pieces:
                    continue
                prefix = "".join(pieces[:k])
                for g in range(12):
                    row, red = ask(args, q, {"heads": ALL_HEADS[32 * g:32 * g + 32], "full_row": True,
                                             "quote_prefix": prefix}, f"{plan}-g{g}")
                    log.append(row)
                    if red:
                        for name, value in red.items():
                            if name != "meta":
                                arrays[f"{plan}/g{g}/{name}"] = value
                        arrays[f"{plan}/g{g}/meta"] = np.array(json.dumps(red["meta"]))
            elif plan == "traj" and pieces:
                for k in TRAJ_K:
                    if k >= len(pieces):
                        break
                    row, red = ask(args, q, {"quote_prefix": "".join(pieces[:k])}, f"traj-k{k}")
                    row["k"] = k
                    log.append(row)
                    if red:
                        arrays[f"traj/k{k}/seg_q"] = red["seg_q"]
                        arrays[f"traj/k{k}/seg_na"] = red["seg_na"]
                        arrays[f"traj/k{k}/top_keys"] = red["top_keys"]
        arrays["log"] = np.array(json.dumps(log))
        np.savez_compressed(path, **arrays)
        winners = [(r["tag"], (r.get("answer") or {}).get("segment")) for r in log if r["tag"].startswith("traj")]
        print(f"  {q['id']} target {q['targets']}: {len(log)} requests, "
              f"{sum(r['ms'] for r in log) / 1e3:.0f} s; traj {winners}", flush=True)


if __name__ == "__main__":
    main()
