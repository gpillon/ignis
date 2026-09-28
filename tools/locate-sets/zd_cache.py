"""EXPERIMENT (branch locate-long-context, exploratory): one cache of
everything already dumped, for testing zero-decode readings offline.

Per question of sets A-D (spec 18/19, short), L2 (synthetic long), R (real,
exploratory) and R2 (real, spec 21):

- `Q`, `NA` [384, segments] f32: each head's softmax share of each segment
  over the state's span, the question's prefill and the content-free one's
  (every head; A-D from their key rows, L2/R/R2 from `research.py heads0`);
- `kf` {feature: (q [H, segments], na [H, segments])} from key-level rows —
  the served 32 heads for L2/R/R2, every head for A-D: `sum`, `first`,
  `last`, `max` (a segment's keys), `sep` (the keys between it and the next
  segment: its closing separator), `next1` (the next segment's first key);
  and `argmax` [H] each head's peak key with `keys` to place it;
- the texts, instruction, targets, siblings (r2set's rule), span length.

    python zd_cache.py --out <dir>
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

import long_rows as LR
from r2set import siblings

LOC = "F:/ai/opencode/inference/.scratch/locate"
WT = "F:/ai/opencode/.inference-qwen-worktrees/locate-long-context/.scratch"
SERVED = json.load(open(f"{LOC}/phase0/vote-choice.json", encoding="utf-8"))["choice"]["method"]["heads"]


def key_features(q, keys, span):
    """Per-segment features of rows `q` [H, span] (pre-softmax scores)."""
    p = LR.softmax_rows(q)
    cum = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, axis=1)], axis=1)
    owned = [(j, k) for j, k in enumerate(keys) if k is not None]
    S = len(keys)
    out = {name: np.full((p.shape[0], S), np.nan) for name in ("sum", "first", "last", "max", "sep", "next1")}
    for n, (j, (a, b)) in enumerate(owned):
        nxt = owned[n + 1][1][0] if n + 1 < len(owned) else span
        out["sum"][:, j] = cum[:, b] - cum[:, a]
        out["first"][:, j] = p[:, a]
        out["last"][:, j] = p[:, b - 1]
        out["max"][:, j] = p[:, a:b].max(axis=1)
        out["sep"][:, j] = cum[:, nxt] - cum[:, b]
        out["next1"][:, j] = p[:, nxt] if nxt < span else 0.0
    return out


def texts_of(q):
    if isinstance(q["state"], list):
        return [json.dumps(item, ensure_ascii=False) for item in q["state"]]
    return q["state"].split("\n")


def base_row(q, set_name):
    texts = texts_of(q)
    line_based = isinstance(q["state"], str) and q.get("family", "logs") != "prose" and set_name != "L2"
    return {
        "id": q["id"], "set": set_name, "family": q.get("family", q.get("source", "logs")),
        "source": q.get("source", "synthetic" if set_name in "ABCD" or set_name == "L2" else "cluster"),
        "split": q.get("split"), "absent": q["absent"], "targets": q["targets"], "texts": texts,
        "instruction": q["instruction"], "tier": q.get("segments"),
        "siblings": q.get("siblings") if q.get("siblings") is not None else
        (siblings(texts)[q["targets"][0]] if line_based and q["targets"] else None),
    }


def cache_short(set_name):
    """A-D: every head's key rows, variants s2 and s2-na."""
    manifest = {q["id"]: q for q in json.load(open(f"{LOC}/{set_name}/manifest.json", encoding="utf-8"))["questions"]}
    meta = json.load(open(f"{LOC}/dumps/{set_name}-hq.json", encoding="utf-8"))
    raw = np.memmap(f"{LOC}/dumps/{meta['bin_file']}", dtype="<f2", mode="r")
    out = []
    for line in open(f"{LOC}/dumps/{set_name}-hq.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["absent"]:
            continue
        span = r["span"][1]
        keys = [tuple(k) if k else None for k in r["keys"]]
        rows = {}
        for v in ("s2", "s2-na"):
            off = r["variants"][v]["offset"]
            rows[v] = np.asarray(raw[off:off + 384 * span], dtype=np.float64).reshape(384, span)
        row = base_row(manifest[r["id"]], set_name)
        fq, fn = key_features(rows["s2"], keys, span), key_features(rows["s2-na"], keys, span)
        row.update(span=span, keys=keys, Q=fq["sum"].astype(np.float32), NA=fn["sum"].astype(np.float32),
                   kf={k: (fq[k].astype(np.float32), fn[k].astype(np.float32)) for k in fq},
                   kheads=list(range(384)), argmax=rows["s2"].argmax(axis=1))
        out.append(row)
    return out


def cache_long(set_name, manifest_path, npz_dir, dump_dirs):
    manifest = {q["id"]: q for q in json.load(open(manifest_path, encoding="utf-8"))["questions"]}
    out = []
    for qid, q in manifest.items():
        if q["absent"]:
            continue
        npz = os.path.join(npz_dir, qid + ".npz") if npz_dir else None
        dump = next((d for d in dump_dirs if os.path.exists(os.path.join(d, qid + ".bin"))), None)
        if not (npz and os.path.exists(npz)) and not dump:
            continue
        row = base_row(q, set_name)
        if npz and os.path.exists(npz):
            z = np.load(npz, allow_pickle=True)
            if "heads0/g0/seg_q" in z.files:
                row["Q"] = np.concatenate([z[f"heads0/g{g}/seg_q"] for g in range(12)]).astype(np.float32)
                row["NA"] = np.concatenate([z[f"heads0/g{g}/seg_na"] for g in range(12)]).astype(np.float32)
                row["lse"] = np.concatenate([z[f"heads0/g{g}/lse"] for g in range(12)])
                row["lse_na"] = np.concatenate([z[f"heads0/g{g}/lse_na"] for g in range(12)])
        if dump:
            meta, qr, nar = LR.load(dump, qid)
            keys = [tuple(k) if k else None for k in meta["keys"]]
            span = meta["span"]
            if meta.get("full_row"):
                s0 = meta["span_start"]
                qr, nar = qr[:, s0:s0 + span], nar[:, s0:s0 + span]
            fq, fn = key_features(qr, keys, span), key_features(nar, keys, span)
            row.update(span=span, keys=keys, kheads=SERVED,
                       kf={k: (fq[k].astype(np.float32), fn[k].astype(np.float32)) for k in fq},
                       argmax=qr.argmax(axis=1))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{LOC}/zd/cache")
    ap.add_argument("--sets", nargs="+", default=["A", "B", "C", "D", "L2", "R", "R2"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for s in args.sets:
        if s in "ABCD":
            rows = cache_short(s)
        elif s == "L2":
            rows = cache_long(s, f"{LOC}/long/L2/manifest.json", f"{LOC}/research/L2", [f"{LOC}/long/dumps2"])
        elif s == "R":
            rows = cache_long(s, f"{WT}/prod/R/manifest.json", f"{LOC}/research/R", [f"{LOC}/long/dumps2"])
        elif s == "R2":
            rows = cache_long(s, f"{WT}/prod/R2/manifest.json", f"{LOC}/research/R2", [f"{LOC}/research/raw2"])
        with open(os.path.join(args.out, s + ".pkl"), "wb") as f:
            pickle.dump(rows, f, protocol=4)
        print(s, len(rows), "questions;", sum("Q" in r for r in rows), "with 384 heads;",
              sum("kf" in r for r in rows), "with key rows", flush=True)


if __name__ == "__main__":
    main()
