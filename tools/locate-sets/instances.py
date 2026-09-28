"""EXPERIMENT (branch locate-long-context, exploratory): why the vote picks
another instance of the target's kind of line (spec 20's real-log misses).

For each present question of a `long.py` set with dumps (the 32 served
heads' rows, both prefills), the key grid is mapped to tokens (key i of the
span is token i of the JSON-escaped state, checked per question), and the
target is set against its **twin**: the line the vote chose when it missed,
else the other line most like the target (token Jaccard). Tokens of the two
lines split into:

- `t_own`: in the target and not in the twin (what tells them apart),
- `t_shared` / `w_shared`: in both (the template),
- `w_own`: in the twin and not in the target,

and every head's softmax mass (question prefill), and its lift (less the
content-free prefill's), is summed per class; the peak key is classed too.
Hits against misses is the contrast.

    python instances.py --set <R> --served R-served.json --dumps <dir> --tokenizer tokenizer.json --out instances.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np

import long_rows as LR

CLASSES = ("t_own", "t_shared", "w_own", "w_shared", "other")


def span_tokens(tok, state, span):
    system = '{"evidence":' + json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "}"
    enc = tok.encode(system)
    offsets, all_ids = enc.offsets, enc.ids      # each access builds a new list
    start, end = len('{"evidence":"'), len(system) - 2
    idx = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end]
    if len(idx) != span:
        raise SystemExit(f"key grid mismatch: {len(idx)} tokens against a span of {span}")
    return [all_ids[i] for i in idx], [system[offsets[i][0]:offsets[i][1]] for i in idx]


def twin_of(ids, keys, t, chosen):
    if chosen != t:
        return chosen
    mine = set(ids[keys[t][0]:keys[t][1]])
    best, score = None, -1.0
    for j, k in enumerate(keys):
        if j == t or not k:
            continue
        other = set(ids[k[0]:k[1]])
        jac = len(mine & other) / max(1, len(mine | other))
        if jac > score:
            best, score = j, jac
    return best


def classes(ids, keys, t, w, span):
    lab = np.full(span, 4, dtype=np.int8)
    tset, wset = set(ids[keys[t][0]:keys[t][1]]), set(ids[keys[w][0]:keys[w][1]])
    for i in range(*keys[t]):
        lab[i] = 1 if ids[i] in wset else 0
    for i in range(*keys[w]):
        lab[i] = 3 if ids[i] in tset else 2
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=int, default=0, help="print the top keys of this many misses")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    with open(f"{args.set}/manifest.json", encoding="utf-8") as f:
        manifest = {q["id"]: q for q in json.load(f)["questions"]}
    with open(args.served, encoding="utf-8") as f:
        rows = [r for r in json.load(f)["questions"] if not r["absent"]]
    agg = {k: defaultdict(list) for k in ("hit", "miss")}
    per_q = []
    shown = 0
    for r in rows:
        meta, q, na = LR.load(args.dumps, r["id"])
        keys, span = meta["keys"], meta["span"]
        ids, texts = span_tokens(tok, manifest[r["id"]]["state"], span)
        t = r["targets"][0]
        chosen = r["locate"]["answer"]["segment"]
        w = twin_of(ids, keys, t, chosen)
        lab = classes(ids, keys, t, w, span)
        pq, pn = LR.softmax_rows(q), LR.softmax_rows(na)
        side = "hit" if chosen == t else "miss"
        counts = np.array([(lab == c).sum() for c in range(5)])
        mass = np.stack([pq[:, lab == c].sum(axis=1) for c in range(5)], axis=1)       # [heads, 5]
        lift = np.stack([(pq - pn)[:, lab == c].sum(axis=1) for c in range(5)], axis=1)
        peak = np.argmax(pq - pn, axis=1)
        peak_class = lab[peak]
        a = agg[side]
        a["mass"].append(mass)
        a["lift"].append(lift)
        a["counts"].append(counts)
        a["peak"].append(np.bincount(peak_class, minlength=5))
        per_q.append({"id": r["id"], "side": side, "split": r["split"], "tier": r["segments"], "t": t, "twin": w,
                      "counts": counts.tolist(), "mass_mean": mass.mean(axis=0).tolist(),
                      "lift_mean": lift.mean(axis=0).tolist(),
                      "peak_classes": np.bincount(peak_class, minlength=5).tolist()})
        if side == "miss" and shown < args.show:
            shown += 1
            top = np.argsort(-(pq - pn).sum(axis=0))[:12]
            print(f"--- {r['id']} ({r['split']}) target {t} chose {chosen}")
            print("    top lifted keys:", " | ".join(f"{CLASSES[lab[k]]}:{texts[k]!r}" for k in top))
    out = {}
    for side, a in agg.items():
        if not a["mass"]:
            continue
        mass = np.concatenate(a["mass"]).mean(axis=0)
        lift = np.concatenate(a["lift"]).mean(axis=0)
        counts = np.stack(a["counts"]).mean(axis=0)
        peak = np.stack(a["peak"]).sum(axis=0)
        out[side] = {"questions": len(a["mass"]),
                     "mass": dict(zip(CLASSES, mass.round(4).tolist())),
                     "mass_per_token": dict(zip(CLASSES, (mass / np.maximum(counts, 1)).round(6).tolist())),
                     "lift": dict(zip(CLASSES, lift.round(4).tolist())),
                     "tokens": dict(zip(CLASSES, counts.round(1).tolist())),
                     "peak_share": dict(zip(CLASSES, (peak / peak.sum()).round(3).tolist()))}
        print(f"== {side} ({len(a['mass'])} questions)")
        for k in ("tokens", "mass", "mass_per_token", "lift", "peak_share"):
            print(f"   {k:15s}", out[side][k])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "questions": per_q}, f, indent=1)


if __name__ == "__main__":
    main()
