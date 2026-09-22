"""Score C5's attention maps on acceptance 3, against the chain's 218/240.

Every estimator here is reported under how much it was allowed to know, from
nothing to everything:

- **all heads, no labels** — the mean map over every GQA head;
- **TAG, no labels** — per scene, the top-10 heads by the attention mass they
  put on the image, as `arXiv:2412.10840` selects them;
- **cross-validated heads** — heads ranked by their accuracy on the training
  folds and averaged on the held-out one: labels choose *which* heads, and
  nothing else is fitted;
- **best single head** — chosen on the same scenes it is scored on, and so
  an upper bound, labelled as one.

A map becomes a point two ways: its argmax cell's centre, and TAG's rule —
normalize to [0, 1], keep >= 0.5, take the connected region with the highest
mean relevance, and its relevance-weighted centre.
"""

import argparse
import json

import numpy as np


def cells(grid, side):
    gh, gw = grid
    k = np.arange(gh * gw)
    return (k % gw + 0.5) * side / gw, (k // gw + 0.5) * side / gh


def point_argmax(m, cx, cy):
    k = int(np.argmax(m))
    return cx[k], cy[k]


def point_tag(m, grid, cx, cy):
    return region_tag(m, grid, cx, cy)[0]


def region_tag(m, grid, cx, cy, thr=0.5):
    """TAG's rule: the point, and the region it came from as cell indices."""
    gh, gw = grid
    lo, hi = float(m.min()), float(m.max())
    if hi <= lo:
        k = int(np.argmax(m))
        return (cx[k], cy[k]), np.array([k])
    r = (m - lo) / (hi - lo)
    on = (r >= thr).reshape(gh, gw)
    rr = r.reshape(gh, gw)
    seen = np.zeros_like(on)
    best, best_score = None, -1.0
    for i in range(gh):
        for j in range(gw):
            if on[i, j] and not seen[i, j]:
                stack, comp = [(i, j)], []
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    comp.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + da, b + db
                        if 0 <= u < gh and 0 <= v < gw and on[u, v] and not seen[u, v]:
                            seen[u, v] = True
                            stack.append((u, v))
                score = float(np.mean([rr[a, b] for a, b in comp]))
                if score > best_score:
                    best, best_score = comp, score
    w = np.array([rr[a, b] for a, b in best])
    ks = np.array([a * gw + b for a, b in best])
    return (float((cx[ks] * w).sum() / w.sum()),
            float((cy[ks] * w).sum() / w.sum())), ks


def region_box(ks, grid, side):
    """The pixel box spanned by a set of cells."""
    gh, gw = grid
    rows, cols = ks // gw, ks % gw
    return (cols.min() * side / gw, rows.min() * side / gh,
            (cols.max() + 1) * side / gw, (rows.max() + 1) * side / gh)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inside(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def norm(m):
    """Per-map min-max, so heads with different scales average fairly."""
    lo = m.min(axis=-1, keepdims=True)
    hi = m.max(axis=-1, keepdims=True)
    return (m - lo) / np.maximum(hi - lo, 1e-12)


def score_maps(maps, boxes, grid, side):
    """maps [N, P] -> (inside by argmax, inside by TAG rule, errors)."""
    cx, cy = cells(grid, side)
    a = t = 0
    for m, b in zip(maps, boxes):
        a += inside(point_argmax(m, cx, cy), b)
        t += inside(point_tag(m, grid, cx, cy), b)
    return a, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/c5-attention.npz")
    ap.add_argument("--out", default="results/c5-score.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz)
    att = z["att"].astype(np.float32)          # [N, L, H, Q, P]
    layers = [int(v) for v in z["layers"]]
    grid = tuple(int(v) for v in z["grid"])
    side = int(z["side"])
    queries = [str(q) for q in z["queries"]]
    boxes = z["boxes"]
    N, L, H, Q, P = att.shape
    cx, cy = cells(grid, side)
    print(f"{N} scenes, {L} GQA layers x {H} heads, queries {queries}, "
          f"grid {grid}, side {side}px; the chain is 218/240, the residual probe 97/240")

    report = {"scenes": N, "rows": []}

    def emit(label, q, a, t):
        row = {"estimator": label, "query": q, "inside_argmax": int(a), "inside_tag": int(t)}
        report["rows"].append(row)
        print(f"  {label:44s} {q:9s} argmax {a:3d}/{N}   TAG rule {t:3d}/{N}")

    # Per-head argmax accuracy, needed by the oracle and by the CV selection.
    head_ok = np.zeros((Q, L, H, N), dtype=bool)
    for qi in range(Q):
        for li in range(L):
            for hi in range(H):
                for n in range(N):
                    head_ok[qi, li, hi, n] = inside(
                        point_argmax(att[n, li, hi, qi], cx, cy), boxes[n])

    for qi, q in enumerate(queries):
        print(f"\nquery `{q}`")
        # 1. every head, no labels
        m = norm(att[:, :, :, qi, :]).mean(axis=(1, 2))
        emit("all heads, mean of normalized maps", q, *score_maps(m, boxes, grid, side))

        # 2. per layer, no labels — which of the 16 carry anything
        per_layer = []
        for li, layer in enumerate(layers):
            m = norm(att[:, li, :, qi, :]).mean(axis=1)
            a, t = score_maps(m, boxes, grid, side)
            per_layer.append((layer, a, t))
        best_layer = max(per_layer, key=lambda r: max(r[1], r[2]))
        print("  per layer (all heads):", " ".join(
            f"L{l}:{max(a, t)}" for l, a, t in per_layer))
        emit(f"best layer L{best_layer[0]} (chosen on these scenes)", q,
             best_layer[1], best_layer[2])

        # 3. TAG: per scene, top-10 heads by attention mass on the image
        mass = att[:, :, :, qi, :].sum(axis=-1).reshape(N, L * H)
        flat = norm(att[:, :, :, qi, :]).reshape(N, L * H, P)
        top = np.argsort(-mass, axis=1)[:, :10]
        m = np.stack([flat[n, top[n]].mean(axis=0) for n in range(N)])
        emit("TAG: top-10 heads by image mass, no labels", q,
             *score_maps(m, boxes, grid, side))

        # 3b. TAG with the attention sinks removed, still label-free. A sink
        #     attends to the same place in every scene whatever the target
        #     is, so its argmax cell barely moves across scenes. Measure that
        #     spread per head (distinct argmax cells over the scenes, as a
        #     fraction), drop the stiffest heads, then take the top-10 by mass
        #     as TAG does — optionally weighted by the head's output gate,
        #     when the run recorded one.
        arg = att[:, :, :, qi, :].argmax(axis=-1).reshape(N, L * H)          # [N, LH]
        spread = np.array([len(np.unique(arg[:, h])) / N for h in range(L * H)])
        gate = z["gate"][:, :, :, qi].reshape(N, L * H) if "gate" in z.files else None
        for keep in (0.5, 0.25):
            alive = spread >= np.quantile(spread, 1 - keep)
            for weighted in ((False, True) if gate is not None else (False,)):
                score_h = mass * (gate if weighted else 1.0)
                score_h = np.where(alive[None, :], score_h, -np.inf)
                top = np.argsort(-score_h, axis=1)[:, :10]
                m = np.stack([flat[n, top[n]].mean(axis=0) for n in range(N)])
                emit(f"TAG, sinks removed (keep {int(keep * 100)}% most mobile)"
                     + (", gate-weighted" if weighted else ""), q,
                     *score_maps(m, boxes, grid, side))

        # 4. cross-validated head selection
        rng = np.random.default_rng(0)
        order = rng.permutation(N)
        for k in (1, 5, 10, 20, 40):
            a_tot = t_tot = 0
            for f in range(args.folds):
                test = order[f::args.folds]
                train = np.setdiff1d(order, test)
                acc = head_ok[qi][:, :, train].mean(axis=-1).reshape(L * H)
                chosen = np.argsort(-acc)[:k]
                m = flat[test][:, chosen].mean(axis=1)
                a, t = score_maps(m, boxes[test], grid, side)
                a_tot += a
                t_tot += t
            emit(f"CV head selection, top {k} heads", q, a_tot, t_tot)

        # 4b. the same CV estimator, scored for precision and as a box.
        #     Precision in 0-999 units so it sits beside the chain's median of
        #     0.8; the box is the thresholded region's extent, at a few
        #     thresholds because TAG's 0.5 was tuned for points.
        centres = z["centres"]
        for k in (5, 10):
            errs, ious = [], {0.3: [], 0.5: [], 0.7: []}
            for f in range(args.folds):
                test = order[f::args.folds]
                train = np.setdiff1d(order, test)
                acc = head_ok[qi][:, :, train].mean(axis=-1).reshape(L * H)
                chosen = np.argsort(-acc)[:k]
                for n in test:
                    m = flat[n, chosen].mean(axis=0)
                    px, py = point_argmax(m, cx, cy)
                    (rx, ry), _ = region_tag(m, grid, cx, cy)
                    errs.append((abs(px - centres[n][0]) / side * 999,
                                 abs(py - centres[n][1]) / side * 999,
                                 abs(rx - centres[n][0]) / side * 999,
                                 abs(ry - centres[n][1]) / side * 999))
                    for thr in ious:
                        _, ks = region_tag(m, grid, cx, cy, thr)
                        ious[thr].append(iou(region_box(ks, grid, side), boxes[n]))
            e = np.array(errs)
            row = {"estimator": f"CV top {k}: precision and box", "query": q,
                   "median_err_x": round(float(np.median(e[:, 0])), 1),
                   "median_err_y": round(float(np.median(e[:, 1])), 1),
                   "p90_err_x": round(float(np.percentile(e[:, 0], 90)), 1),
                   "p90_err_y": round(float(np.percentile(e[:, 1], 90)), 1),
                   "region_median_err_x": round(float(np.median(e[:, 2])), 1),
                   "region_median_err_y": round(float(np.median(e[:, 3])), 1),
                   "region_p90_err_x": round(float(np.percentile(e[:, 2], 90)), 1),
                   "region_p90_err_y": round(float(np.percentile(e[:, 3], 90)), 1)}
            for thr, v in ious.items():
                v = np.array(v)
                row[f"box_iou_median@{thr}"] = round(float(np.median(v)), 3)
                row[f"box_iou>0.5@{thr}"] = int((v > 0.5).sum())
            report["rows"].append(row)
            print(f"  CV top {k:2d} precision (0-999), argmax: median x {row['median_err_x']:5.1f} "
                  f"y {row['median_err_y']:5.1f}  p90 x {row['p90_err_x']:5.1f} "
                  f"y {row['p90_err_y']:5.1f}")
            print(f"             region centre: median x {row['region_median_err_x']:5.1f} "
                  f"y {row['region_median_err_y']:5.1f}  p90 x {row['region_p90_err_x']:5.1f} "
                  f"y {row['region_p90_err_y']:5.1f}   (one merged cell = {999 / grid[1]:.0f} units)")
            print("             box from the region: " + "  ".join(
                f"thr {thr}: median IoU {row[f'box_iou_median@{thr}']:.2f}, "
                f"IoU>0.5 on {row[f'box_iou>0.5@{thr}']}/{N}" for thr in ious))

        # 4c. orientation: the merged image tokens are taken as row-major over
        #     the grid. If that were wrong every map would be transposed, so
        #     score the same CV estimator with the map transposed. A right
        #     layout wins by a wide margin; a wrong one loses by as much.
        gh, gw = grid
        for k in (5,):
            straight = transposed = 0
            for f in range(args.folds):
                test = order[f::args.folds]
                train = np.setdiff1d(order, test)
                acc = head_ok[qi][:, :, train].mean(axis=-1).reshape(L * H)
                chosen = np.argsort(-acc)[:k]
                m = flat[test][:, chosen].mean(axis=1)
                mt = m.reshape(-1, gh, gw).transpose(0, 2, 1).reshape(len(test), -1)
                straight += score_maps(m, boxes[test], grid, side)[0]
                transposed += score_maps(mt, boxes[test], grid, side)[0]
            report["rows"].append({"estimator": "orientation check, CV top 5", "query": q,
                                   "row_major": int(straight), "transposed": int(transposed)})
            print(f"  orientation (CV top 5, argmax): row-major {straight}/{N}, "
                  f"transposed {transposed}/{N}")

        # 5. the oracle
        ok = head_ok[qi].sum(axis=-1)
        li, hi = np.unravel_index(int(np.argmax(ok)), ok.shape)
        emit(f"best single head L{layers[li]}.h{hi} (selected on test)", q,
             int(ok[li, hi]), int(ok[li, hi]))

    json.dump(report, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
