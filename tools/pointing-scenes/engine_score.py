"""Score the engine's attention-tap run against the pre-registered criterion.

`crates/server/tests/attention_head_point_gpu.rs` writes, per scene set, the
pre-softmax scores `q . k / 16` of every GQA head from the `last` position
onto the image tokens, plus the chain's answer, as a raw f16 file and a JSON
beside it. This turns those into the same maps the vehicle's scorers used
(`exp(s - max)`, then per-map min-max — identical in shape to the vehicle's
softmax restricted to the image columns) and reports:

- **L39.h10** (GQA ordinal 9, head 10), region centre and argmax, per kind;
- the **guard** at d = 60/999, with the engine's own chain;
- which head **cross-validation** picks in the engine, and whether it is
  still L39.h10 (if not, suspect head ordering before the model);
- the orientation check, and precision medians.

And the verdict against the criterion fixed before the run:
vehicle render, KV BF16, 1024 px — L39.h10 region centre >= 230/240 on set A
and >= 227/240 on set B, guard >= 237 on both.
"""

import argparse
import json
import os

import numpy as np

from c5_score import cells, inside, norm, point_argmax, region_tag

CRITERION = {"A": {"head": 230, "guard": 237}, "B": {"head": 227, "guard": 237}}
ORD, HEAD = 9, 10            # L39.h10
D = 60


def load(json_path, bin_path=None):
    meta = json.load(open(json_path))
    if bin_path is None:
        bin_path = os.path.splitext(json_path)[0] + ".bin"
    n = len(meta["rows"])
    gh, gw = meta["grid"]
    raw = np.fromfile(bin_path, dtype=np.float16)
    scores = raw.reshape(n, 16, 24, gh * gw).astype(np.float32)
    return meta, scores


def to_maps(scores):
    """exp(s - max) per map: proportional to the softmax over the image
    columns, which is all a per-map normalization can see."""
    m = scores - scores.max(axis=-1, keepdims=True)
    return np.exp(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+", help="one JSON per scene set")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="set name per JSON for the criterion (A, B), else none")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    labels = args.labels or [None] * len(args.json)

    for path, label in zip(args.json, labels):
        meta, scores = load(path)
        side = meta["side"]
        grid = tuple(meta["grid"])
        cx, cy = cells(grid, side)
        sc = meta["rows"]
        N = len(sc)
        boxes = np.array([s["box"] for s in sc], dtype=np.float64)
        centres = np.array([s["centre"] for s in sc], dtype=np.float64)
        kinds = np.array([s.get("kind") or "blue" for s in sc])
        chain = np.array([s["chain"] for s in sc], dtype=np.float64) / 999 * side
        maps = norm(to_maps(scores))                           # [N, 16, 24, P]
        print(f"\n=== {os.path.basename(path)}: {N} scenes, {side}px, grid {grid}, "
              f"render {meta.get('render')}, kv {meta.get('kv')} ===")

        ch_ok = np.array([inside(p, b) for p, b in zip(chain, boxes)])
        head = maps[:, ORD, HEAD]
        reg = np.array([region_tag(m, grid, cx, cy)[0] for m in head])
        arg = np.array([point_argmax(m, cx, cy) for m in head])
        h_ok = np.array([inside(p, b) for p, b in zip(reg, boxes)])
        a_ok = np.array([inside(p, b) for p, b in zip(arg, boxes)])
        dist = np.hypot(*(chain - reg).T) / side * 999
        out = np.where((dist <= D)[:, None], chain, reg)
        g_ok = np.array([inside(p, b) for p, b in zip(out, boxes)])

        def per_kind(v):
            return {k: f"{int(v[kinds == k].sum())}/{int((kinds == k).sum())}"
                    for k in np.unique(kinds)}

        print(f"  chain            {int(ch_ok.sum()):3d}/{N}  {per_kind(ch_ok)}")
        print(f"  L39.h10 region   {int(h_ok.sum()):3d}/{N}  {per_kind(h_ok)}")
        print(f"  L39.h10 argmax   {int(a_ok.sum()):3d}/{N}")
        print(f"  guard d={D}      {int(g_ok.sum()):3d}/{N}  {per_kind(g_ok)}   "
              f"chain kept on {int((dist <= D).sum())}")
        print(f"  vs chain: both {int((h_ok & ch_ok).sum())}, only head {int((h_ok & ~ch_ok).sum())}, "
              f"only chain {int((~h_ok & ch_ok).sum())}, neither {int((~h_ok & ~ch_ok).sum())}")
        e = np.abs(reg - centres) / side * 999
        print(f"  precision, region centre: median x {np.median(e[:, 0]):.1f} y {np.median(e[:, 1]):.1f}")

        # which head does cross-validation pick in the engine?
        flat = maps.reshape(N, 16 * 24, -1)
        ok = np.zeros((16 * 24, N), dtype=bool)
        for h in range(16 * 24):
            for n in range(N):
                ok[h, n] = inside(point_argmax(flat[n, h], cx, cy), boxes[n])
        rng = np.random.default_rng(0)
        order = rng.permutation(N)
        picks, cv = [], np.zeros(N, dtype=bool)
        for f in range(args.folds):
            test = order[f::args.folds]
            train = np.setdiff1d(order, test)
            h = int(np.argmax(ok[:, train].mean(axis=1)))
            picks.append(h)
            for n in test:
                cv[n] = inside(region_tag(flat[n, h], grid, cx, cy)[0], boxes[n])
        names = [f"L{4 * (h // 24) + 3}.h{h % 24}" for h in picks]
        print(f"  CV top-1 head per fold: {names}  -> {int(cv.sum())}/{N}")
        best = np.argsort(-ok.mean(axis=1))[:5]
        print("  best heads on this set:", ", ".join(
            f"L{4 * (h // 24) + 3}.h{h % 24} ({int(ok[h].sum())})" for h in best))

        # orientation
        t = head.reshape(N, *grid).transpose(0, 2, 1).reshape(N, -1)
        t_ok = sum(inside(point_argmax(m, cx, cy), b) for m, b in zip(t, boxes))
        print(f"  orientation (L39.h10 argmax): row-major {int(a_ok.sum())}, transposed {t_ok}")

        if label in CRITERION:
            c = CRITERION[label]
            passed = h_ok.sum() >= c["head"] and g_ok.sum() >= c["guard"]
            print(f"  CRITERION set {label}: head >= {c['head']} and guard >= {c['guard']} -> "
                  f"head {int(h_ok.sum())}, guard {int(g_ok.sum())}: "
                  f"{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
