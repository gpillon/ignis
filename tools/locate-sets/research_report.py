"""EXPERIMENT (branch locate-long-context, exploratory): read `research.py`'s
reduced dumps.

- **heads**: every one of the 384 heads as a one-head reading (its segment
  of largest lift, question less content-free), at the scaffold (k = 0) and
  8 target tokens into the copy (k = 8): hit rate per length tier;
- **mass**: where each head's softmax goes over the whole prompt — before
  the state, the state, and the tail split into the closing template, the
  kind text, the instruction, the turn and the scaffold (with the forced
  text) — per layer and per tier;
- **trajectory**: the served vote's answer, and its heads' share on the
  target, as the target's first k tokens are forced after the scaffold.

    python research_report.py --dir <out>/R --set <R> --out report.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from tokenizers import Tokenizer

KIND = ("Find the one line of the evidence that the instruction asks for. Answer with only "
        "a JSON object {\"quote\": \"<that line, copied exactly>\"}.")
REGIONS = ("before", "state", "close", "kind", "instruction", "turn", "scaffold")
ALL_HEADS = [f"L{4 * g + 3}.h{h}" for g in range(16) for h in range(24)]


def tail_labels(tok, instruction, forced, width):
    """Region index (2..6) of every tail position, from the rendered tail."""
    instr = json.dumps(instruction, ensure_ascii=False, separators=(",", ":"))
    parts = [('"}<|im_end|>\n<|im_start|>user\n', 2), (KIND + "\n\n", 3),
             ('{"instruction":' + instr + "}", 4),
             ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", 5), ('{"quote":"' + forced, 6)]
    labels, text = [], ""
    for part, lab in parts:
        before = len(tok.encode(text).ids)
        text += part
        after = len(tok.encode(text).ids)
        labels += [lab] * (after - before)
    if len(labels) != width:
        return None
    return np.array(labels)


def head_winners(seg_q, seg_na):
    lift = seg_q.astype(np.float64) - seg_na.astype(np.float64)
    lift = np.where(np.isnan(lift), -np.inf, lift)
    return lift.argmax(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    with open(os.path.join(args.set, "manifest.json"), encoding="utf-8") as f:
        manifest = {q["id"]: q for q in json.load(f)["questions"]}
    hits = {k: defaultdict(lambda: np.zeros(384)) for k in ("heads0", "heads8")}
    counts = {k: defaultdict(int) for k in ("heads0", "heads8")}
    mass = defaultdict(list)            # tier -> [384, 7] per question
    traj = defaultdict(lambda: defaultdict(list))
    unmatched = 0
    for path in sorted(glob.glob(os.path.join(args.dir, "*.npz"))):
        qid = os.path.splitext(os.path.basename(path))[0]
        q = manifest[qid]
        if q["absent"]:
            continue
        t, tier = q["targets"][0], q["segments"]
        z = np.load(path, allow_pickle=True)
        for plan in ("heads0", "heads8"):
            if f"{plan}/g0/seg_q" not in z.files:
                continue
            won = np.concatenate([head_winners(z[f"{plan}/g{g}/seg_q"], z[f"{plan}/g{g}/seg_na"]) for g in range(12)])
            hits[plan][tier] += won == t
            hits[plan]["all"] += won == t
            counts[plan][tier] += 1
            counts[plan]["all"] += 1
        if "heads0/g0/lse" in z.files:
            per_head = []
            ok = True
            for g in range(12):
                meta = json.loads(str(z[f"heads0/g{g}/meta"]))
                lse = z[f"heads0/g{g}/lse"]
                tail = z[f"heads0/g{g}/tail"].astype(np.float64)
                labels = tail_labels(tok, q["instruction"], meta["quote_prefix"], tail.shape[1])
                if labels is None:
                    ok = False
                    break
                top = np.maximum(lse.max(axis=1), tail.max(axis=1))[:, None]
                parts = np.zeros((tail.shape[0], 7))
                parts[:, 0] = np.exp(lse[:, 0] - top[:, 0])
                parts[:, 1] = np.exp(lse[:, 1] - top[:, 0])
                for r in range(2, 7):
                    parts[:, r] = np.exp(tail[:, labels == r] - top).sum(axis=1)
                per_head.append(parts / parts.sum(axis=1, keepdims=True))
            if ok:
                mass[tier].append(np.concatenate(per_head))
            else:
                unmatched += 1
        log = json.loads(str(z["log"]))
        for row in log:
            if row["tag"].startswith("traj"):
                k = row["k"]
                seg = (row.get("answer") or {}).get("segment")
                traj[tier][k].append(seg == t)
                traj["all"][k].append(seg == t)
                if f"traj/k{k}/seg_q" in z.files:
                    won = head_winners(z[f"traj/k{k}/seg_q"], z[f"traj/k{k}/seg_na"])
                    traj[("heads_on_target", tier)][k].append(float((won == t).mean()))
    report = {"heads": {}, "mass": {}, "trajectory": {}, "unmatched_tails": unmatched}
    for plan in ("heads0", "heads8"):
        report["heads"][plan] = {}
        for tier, h in hits[plan].items():
            n = counts[plan][tier]
            rate = h / max(n, 1)
            best = np.argsort(-rate)[:12]
            report["heads"][plan][str(tier)] = {
                "n": n, "best": [(ALL_HEADS[i], round(float(rate[i]), 2)) for i in best],
                "heads_over_half": int((rate > 0.5).sum()), "rate": rate.round(3).tolist()}
    for tier, rows in sorted(mass.items(), key=lambda kv: str(kv[0])):
        m = np.stack(rows)                  # [questions, 384, 7]
        per_layer = m.mean(axis=0).reshape(16, 24, 7).mean(axis=1)
        report["mass"][str(tier)] = {"n": len(rows), "by_layer": {f"L{4 * g + 3}": dict(zip(REGIONS, per_layer[g].round(3).tolist())) for g in range(16)},
                                     "all_heads": dict(zip(REGIONS, m.mean(axis=(0, 1)).round(3).tolist()))}
    for tier, by_k in traj.items():
        report["trajectory"][str(tier)] = {str(k): round(float(np.mean(v)), 3) for k, v in sorted(by_k.items())}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for plan in ("heads0", "heads8"):
        for tier, r in report["heads"][plan].items():
            print(f"{plan} tier {tier:>7} n {r['n']:3}: heads >50% {r['heads_over_half']:3}; best {r['best'][:6]}")
    for tier, r in report["mass"].items():
        print(f"mass tier {tier:>7} (n {r['n']}): all heads {r['all_heads']}")
    for tier, r in report["trajectory"].items():
        print(f"trajectory {tier}: {r}")
    print(f"tails not matched: {unmatched}")


if __name__ == "__main__":
    main()
