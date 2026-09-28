"""EXPERIMENT (branch locate-long-context, exploratory): a zero-decode
`locate` for long logs with no long prefill — fold, then heads and the
labelled `choice` at each level.

Per question it records everything the configurations below need, so they
are compared offline (`--report`) without asking the model again:

- level 1 (the folded templates): the head set's rows read at every
  template (end reading, sum, both), a labelled `choice` over all the
  templates, and one over the heads' first 3 and first 5;
- level 2, for every template among the heads' first 3 and the `choice`'s
  pick: the rows (values first); past 16 rows the heads' ranking of them;
  a labelled `choice` over the first 16 (or all).

Configurations: L1 = heads top-1 | choice over all | heads top-k then
choice; L2 = heads top-1 | choice over the heads' top 16. The answer is the
original line (unfolded).

    python zd_logpipe.py --set <R> --heads endheads.json --out R-logpipe.json
    python zd_logpipe.py --report R-logpipe.json
"""

import argparse
import glob
import json
import os

import numpy as np

import compress as C
import folded_locate as FL
import long_rows as LR
from zd_cache import key_features
from zd_offline import zsum
from zd_qtok import post

K2 = 16


def head_rank(args, heads, lines, instruction):
    """The heads' readings of `lines` (end, sum, both), best first."""
    if len(lines) < 2:
        return {"end": [0], "sum": [0], "both": [0]}, 0.0
    with open(args.control, "w", encoding="utf-8") as f:
        json.dump({"heads": heads, "tag": "logpipe"}, f)
    before = set(glob.glob(os.path.join(args.raw, "*.json")))
    body = {"state": "\n".join(lines), "questions": {"q": {"type": "locate", "instructions": instruction}}}
    status, payload, ms = post(args.url, body)
    with open(args.control, "w", encoding="utf-8") as f:
        json.dump({}, f)
    new = sorted(set(glob.glob(os.path.join(args.raw, "*.json"))) - before)
    if len(new) != 1:
        raise RuntimeError(f"no dump: {status} {str(payload)[:300]}")
    base = os.path.splitext(new[0])[0]
    meta, qr, nar = LR.load(os.path.dirname(base), os.path.basename(base))
    for ext in (".json", ".bin"):
        os.remove(base + ext)
    keys = [tuple(k) if k else None for k in meta["keys"]]
    fq, fn = key_features(qr, keys, meta["span"]), key_features(nar, keys, meta["span"])
    end = zsum((fq["last"] + fq["sep"] + fq["next1"]) - (fn["last"] + fn["sep"] + fn["next1"])).sum(0)
    sm = zsum(fq["sum"] - fn["sum"]).sum(0)
    rank = lambda v: [int(i) for i in np.argsort(-v, kind="stable")]
    return {"end": rank(end), "sum": rank(sm), "both": rank(end + sm)}, ms


def run(args):
    heads = json.load(open(args.heads, encoding="utf-8"))["heads"]
    questions = json.load(open(os.path.join(args.set, "manifest.json"), encoding="utf-8"))["questions"]
    out = {"heads": args.heads, "questions": {}}
    if os.path.exists(args.out):
        out = json.load(open(args.out, encoding="utf-8"))
    folds = {}
    for q in questions:
        if q["absent"] or q["id"] in out["questions"]:
            continue
        lines = q["state"].split("\n")
        key = hash(q["state"])
        if key not in folds:
            folds[key] = C.fold(lines, C.SIM, values=True)
        f = folds[key]
        t = q["targets"][0]
        tc = next(i for i, c in enumerate(f.clusters) if t in c.members)
        rec = {"target_cluster": tc, "clusters": len(f.clusters), "l1": {}, "l2": {}}
        rec["l1"]["heads"], rec["l1"]["heads_ms"] = head_rank(args, heads, f.level1, q["instruction"])
        FL.ROUTE = "choice"
        one = FL.locate(args.url, f.level1, q["instruction"])
        rec["l1"]["choice"] = {"segment": one.get("segment"), "ms": one.get("ms", 0),
                               "ranking": [e["segment"] for e in one.get("ranking", [])][:10]}
        for k in (3, 5):
            for reading in ("end", "both"):
                cand = sorted(rec["l1"]["heads"][reading][:k])
                got = FL.locate(args.url, [f.level1[i] for i in cand], q["instruction"])
                rec["l1"][f"{reading}{k}_choice"] = {"segment": cand[got["segment"]] if "segment" in got else None,
                                                     "ms": got.get("ms", 0)}
        todo = set(rec["l1"]["heads"]["end"][:3]) | set(rec["l1"]["heads"]["both"][:3])
        todo |= {rec["l1"]["choice"]["segment"]} | {v["segment"] for k, v in rec["l1"].items() if k.endswith("_choice")}
        for ci in sorted(x for x in todo if x is not None):
            texts, members = C.level2(f, ci)
            l2 = {"rows": len(texts), "members": members}
            if len(texts) > K2:
                l2["heads"], l2["heads_ms"] = head_rank(args, heads, texts, q["instruction"])
                for reading in ("end", "both"):
                    cand = sorted(l2["heads"][reading][:K2])
                    got = FL.locate(args.url, [texts[i] for i in cand], q["instruction"])
                    l2[f"{reading}_choice"] = {"row": cand[got["segment"]] if "segment" in got else None,
                                               "ms": got.get("ms", 0)}
            else:
                got = FL.locate(args.url, texts, q["instruction"])
                l2["choice"] = {"row": got.get("segment"), "ms": got.get("ms", 0)}
            rec["l2"][str(ci)] = l2
        out["questions"][q["id"]] = dict(rec, targets=q["targets"], segments=q["segments"])
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        print(f"{q['id']} tc {tc} | heads end top3 {rec['l1']['heads']['end'][:3]} choice {rec['l1']['choice']['segment']}",
              flush=True)


def l2_answer(l2, rule, reading):
    if rule == "heads":
        if "heads" not in l2:
            return None if l2["rows"] > 1 else 0
        return l2["heads"][reading][0]
    if "choice" in l2:
        return l2["choice"]["row"]
    return l2[f"{reading}_choice"]["row"]


def report(args):
    data = json.load(open(args.report, encoding="utf-8"))["questions"]
    configs = {}
    for qid, r in data.items():
        t = r["targets"][0]
        l1 = r["l1"]
        picks = {"heads-end": l1["heads"]["end"][0], "heads-both": l1["heads"]["both"][0],
                 "choice-all": l1["choice"]["segment"]}
        for k in (3, 5):
            for reading in ("end", "both"):
                picks[f"heads-{reading}{k}+choice"] = l1[f"{reading}{k}_choice"]["segment"]
        for name, ci in picks.items():
            l1_ok = ci == r["target_cluster"]
            configs.setdefault((name, "L1 only"), []).append(l1_ok)
            l2 = r["l2"].get(str(ci))
            if l2 is None:
                continue
            for rule in ("heads", "choice"):
                for reading in ("end", "both"):
                    row = l2_answer(l2, rule, reading)
                    hit = row is not None and t in l2["members"][row]
                    configs.setdefault((name, f"L2 {rule}-{reading}"), []).append(hit)
    for (l1, l2), v in sorted(configs.items()):
        print(f"{l1:22s} {l2:18s} {sum(v):3d}/{len(v)}")
    # the chosen configuration's wall time: heads over the templates, the
    # choice among their first 5, then heads over the rows (past 16) and the
    # choice among the rows
    ms = []
    for r in data.values():
        l1 = r["l1"]
        ci = l1["end5_choice"]["segment"]
        l2 = r["l2"].get(str(ci), {})
        total = l1["heads_ms"] + l1["end5_choice"]["ms"] + l2.get("heads_ms", 0)
        total += (l2.get("choice") or l2.get("end_choice") or {}).get("ms", 0)
        ms.append(total)
    ms = sorted(ms)
    print(f"heads-end5+choice / L2 choice-end: median {ms[len(ms) // 2] / 1e3:.2f} s, p90 {ms[int(0.9 * len(ms))] / 1e3:.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set")
    ap.add_argument("--heads")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--raw", default="F:/ai/opencode/inference/.scratch/locate/zd/raw")
    ap.add_argument("--control", default="F:/ai/opencode/inference/.scratch/locate/zd/control.json")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    if args.report:
        report(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
