"""The anchored head ensemble (spec 14): choose its heads, and read a point and a box with it.

Reads dumps of `crates/server/tests/attention_head_point_gpu.rs` run with every
GQA layer armed (`IGNIS_POINT_LAYERS=all`): `<stem>.json` + `<stem>.bin`, the
latter f16 scores `[scene][armed GQA layer][query head 0..24][image position]`.
Needs NumPy.

  select  choose the head set and the fallback cells for an artifact:
          ensemble_score.py select --distractors <dump.json> <scene dir>
                                   --blanks <dump.json> <scene dir> --out heads.json
  score   read every scene of a dump with the anchored ensemble, beside the anchor
          alone (the spec 13 point) and the chain the harness also ran:
          ensemble_score.py score <dump.json> <scene dir> --heads heads.json [--out per-scene.json]
  golden  write golden cases for the host rule's port (crates/core/tests/anchored_reading.rs):
          ensemble_score.py golden --heads heads.json --case <dump.json> <scene dir> <index> ...
                                   --out anchored_reading.json

`read_anchored` below is the reading rule spec 14 ships, and the reference its
host implementation is held to.
"""

import argparse
import json
from pathlib import Path

import numpy as np

# spec 14's constants
SELECT_MIN_LAYER = 31        # the query binds to the asked object from here on
SELECT_MIN_SELECTIVITY = 0.8  # mass on the asked box / (asked + distractors)
SELECT_MIN_MASS = 0.3         # mass on all the boxes together
FALLBACK_SHARE = 0.05         # a cell >= 5% of all heads' peaks on blank priors
KEEP_RADIUS = 2.0             # keep head cells within 2x the median distance to the anchor
EXTENT_QUANTILE = 0.1         # the extent is the 10%-90% quantile span of the kept cells
REGION_THRESHOLD = 0.5        # TAG's region rule, as crates/core/src/pointing.rs


def load(dump_json):
    meta = json.loads(Path(dump_json).read_text())
    raw = np.fromfile(Path(dump_json).with_suffix(".bin"), dtype="<f2").astype(np.float32)
    return meta, raw.reshape(meta["shape"])


def scene_size(scene, side):
    study = scene.get("study") or {}
    if "size" in study:
        return tuple(study["size"])
    return scene.get("width", side), scene.get("height", side)


def tag(scores, rows, cols):
    """TAG's region rule, as `read_head_map` in crates/core/src/pointing.rs: (x, y) in grid units."""
    s = np.asarray(scores, dtype=np.float64)
    mass = np.exp(s - s.max())
    lo, hi = mass.min(), mass.max()
    if hi <= lo:
        return 0.5, 0.5
    norm = (mass - lo) / (hi - lo)
    on = norm >= REGION_THRESHOLD
    seen = np.zeros(len(norm), bool)
    best = None
    for start in range(len(norm)):
        if not on[start] or seen[start]:
            continue
        seen[start] = True
        stack, region = [start], []
        while stack:
            k = stack.pop()
            region.append(k)
            r, c = divmod(k, cols)
            for n in ((k + cols) if r + 1 < rows else None, (k - cols) if r > 0 else None,
                      (k + 1) if c + 1 < cols else None, (k - 1) if c > 0 else None):
                if n is not None and on[n] and not seen[n]:
                    seen[n] = True
                    stack.append(n)
        mean = norm[region].mean()
        if best is None or mean > best[0]:  # strictly greater: the first in raster order keeps a tie
            best = (mean, region)
    region = np.array(best[1])
    w = norm[region]
    return float((((region % cols) + 0.5) * w).sum() / w.sum()), float((((region // cols) + 0.5) * w).sum() / w.sum())


def read_anchored(anchor_scores, head_argmax, rows, cols, width, height):
    """Spec 14's reading rule.

    anchor_scores: the anchor head's scores over the image span (row-major grid).
    head_argmax:   one image-cell index per head of the set -- each head's argmax
                   over the span with the fallback cells excluded.
    Returns (anchor point, point, extent) in pixels of the submitted image.
    """
    cw, ch = width / cols, height / rows
    ax, ay = tag(anchor_scores, rows, cols)
    anchor = np.array([ax * cw, ay * ch])
    idx = np.asarray(head_argmax)
    cells = np.stack([(idx % cols + 0.5) * cw, (idx // cols + 0.5) * ch], 1)
    d = np.hypot(*(cells - anchor).T)
    scale = max(float(np.median(d)), max(cw, ch))
    kept = cells[d <= KEEP_RADIUS * scale]
    q = EXTENT_QUANTILE
    x0 = np.quantile(kept[:, 0], q) - cw / 2
    y0 = np.quantile(kept[:, 1], q) - ch / 2
    x1 = np.quantile(kept[:, 0], 1 - q) + cw / 2
    y1 = np.quantile(kept[:, 1], 1 - q) + ch / 2
    extent = (max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1))
    point = ((extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2)
    return tuple(anchor), point, extent


def argmax_excluding(scores, excluded):
    s = np.array(scores, dtype=np.float64)
    if excluded:
        s[list(excluded)] = -np.inf
    return int(np.argmax(s))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    return inter / (area(a) + area(b) - inter) if inter > 0 else 0.0


def box_mask(box, rows, cols, width, height):
    """Cells whose centre lies within half a cell of the box."""
    cw, ch = width / cols, height / rows
    c = (np.arange(cols) + 0.5) * cw
    r = (np.arange(rows) + 0.5) * ch
    inx = (c >= box[0] - cw / 2) & (c <= box[2] + cw / 2)
    iny = (r >= box[1] - ch / 2) & (r <= box[3] + ch / 2)
    return (iny[:, None] & inx[None, :]).reshape(-1)


def softmax(s):
    p = np.exp(s - s.max())
    return p / p.sum()


def select(args):
    meta, S = load(args.distractors[0])
    man = json.loads((Path(args.distractors[1]) / "manifest.json").read_text())
    by_id = {s["id"]: s for s in man["scenes"]}
    rows, cols = meta["grid"]
    layers = meta["layers"]
    heads = []
    for li, layer in enumerate(layers):
        if layer < SELECT_MIN_LAYER:
            continue
        for h in range(S.shape[2]):
            asked, other = [], []
            for n, row in enumerate(meta["rows"]):
                sc = by_id[row["id"]]
                w, hh = scene_size(sc, man["side"])
                others = (sc.get("study") or {}).get("others") or [(sc.get("study") or {}).get("other")]
                others = [o for o in others if o]
                if not others:
                    continue
                p = softmax(S[n, li, h].astype(np.float64))
                asked.append(p[box_mask(sc["blue_box"], rows, cols, w, hh)].sum())
                other.append(sum(p[box_mask(o, rows, cols, w, hh)].sum() for o in others))
            a, o = float(np.median(asked)), float(np.median(other))
            if a + o >= SELECT_MIN_MASS and a / (a + o) >= SELECT_MIN_SELECTIVITY:
                heads.append([layer, h])
    fallback = {}
    if args.blanks:
        bmeta, B = load(args.blanks[0])
        brows, bcols = bmeta["grid"]
        counts = np.zeros(brows * bcols)
        # a dump may mix blanks with other scenes: count only the blanks when it names any
        ids = [r["id"] for r in bmeta["rows"]]
        blank = [n for n, i in enumerate(ids) if i.startswith(("blank", "prior"))] or list(range(len(ids)))
        for n in blank:
            for li in range(B.shape[1]):
                for h in range(B.shape[2]):
                    counts[int(np.argmax(B[n, li, h]))] += 1
        cells = set(np.flatnonzero(counts >= FALLBACK_SHARE * counts.sum()).tolist()) | {0, brows * bcols - 1}
        fallback[f"{brows}x{bcols}"] = sorted(int(c) for c in cells)
    out = {"anchor": [39, 10], "heads": heads, "fallback_cells": fallback,
           "fallback_default": "the first and the last image cell on any grid not listed",
           "selected_on": str(args.distractors[0]), "rule": {
               "min_layer": SELECT_MIN_LAYER, "min_selectivity": SELECT_MIN_SELECTIVITY, "min_mass": SELECT_MIN_MASS,
               "fallback_share": FALLBACK_SHARE}}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(heads)} heads over layers {sorted({l for l, _ in heads})}; fallback cells {fallback}")


def score(args):
    cfg = json.loads(Path(args.heads).read_text())
    meta, S = load(args.dump)
    man = json.loads((Path(args.scene_dir) / "manifest.json").read_text())
    by_id = {s["id"]: s for s in man["scenes"]}
    rows, cols = meta["grid"]
    layers = meta["layers"]
    excluded = set(cfg["fallback_cells"].get(f"{rows}x{cols}", [])) | {0, rows * cols - 1}
    al, ah = cfg["anchor"]
    per, sums = [], {}
    for n, row in enumerate(meta["rows"]):
        sc = by_id[row["id"]]
        w, h = scene_size(sc, man["side"])
        cw, ch = w / cols, h / rows
        argmax = [argmax_excluding(S[n, layers.index(l), q], excluded) for l, q in cfg["heads"]]
        anchor, point, extent = read_anchored(S[n, layers.index(al), ah], argmax, rows, cols, w, h)
        chain = (row["chain"][0] / 999 * w, row["chain"][1] / 999 * h)
        b = sc["blue_box"]
        inside = lambda p: b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
        dist = lambda p: float(np.hypot(p[0] - (b[0] + b[2]) / 2, p[1] - (b[1] + b[3]) / 2) / np.hypot(b[2] - b[0], b[3] - b[1]))
        rec = {"id": row["id"], "kind": sc.get("kind"), "anchor": anchor, "point": point, "extent": extent,
               "chain": chain, "box": b, "inside": {"anchor": inside(anchor), "ensemble": inside(point), "chain": inside(chain)},
               "distance": {"anchor": dist(anchor), "ensemble": dist(point), "chain": dist(chain)},
               "iou": iou(extent, b)}
        per.append(rec)
    n = len(per)
    print(f"{Path(args.dump).name}: {n} scenes, grid {rows}x{cols}, {len(cfg['heads'])} heads, excluded cells {sorted(excluded)}")
    for m in ("anchor", "ensemble", "chain"):
        k = sum(r["inside"][m] for r in per)
        print(f"  {m:9s} inside {k:4d}/{n} ({k / n:5.1%})  centre distance / diagonal median {np.median([r['distance'][m] for r in per]):.3f}")
    ious = np.array([r["iou"] for r in per])
    print(f"  ensemble box IoU median {np.median(ious):.2f}, >= 0.5 on {np.mean(ious >= 0.5):.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps(per, indent=1))


def golden(args):
    """Golden cases for the host rule (crates/core/tests/anchored_reading.rs): per case the
    anchor's scores, the set's argmax over the span minus the fallback cells, the grid and the
    image size in; `read_anchored`'s anchor, point and extent out."""
    cfg = json.loads(Path(args.heads).read_text())
    cases = []
    for dump, scene_dir, index in args.case:
        meta, S = load(dump)
        man = json.loads((Path(scene_dir) / "manifest.json").read_text())
        by_id = {s["id"]: s for s in man["scenes"]}
        rows, cols = meta["grid"]
        layers = meta["layers"]
        excluded = set(cfg["fallback_cells"].get(f"{rows}x{cols}", [])) | {0, rows * cols - 1}
        n = int(index)
        row = meta["rows"][n]
        w, h = scene_size(by_id[row["id"]], man["side"])
        al, ah = cfg["anchor"]
        anchor_scores = S[n, layers.index(al), ah]
        argmax = [argmax_excluding(S[n, layers.index(l), q], excluded) for l, q in cfg["heads"]]
        anchor, point, extent = read_anchored(anchor_scores, argmax, rows, cols, w, h)
        cases.append({"name": f"{meta['set']}/{row['id']}", "rows": rows, "cols": cols, "width": w, "height": h,
                      "excluded": sorted(excluded), "anchor_scores": [float(s) for s in anchor_scores],
                      "set_argmax": argmax, "anchor": list(anchor), "point": list(point), "extent": list(extent)})
    Path(args.out).write_text(json.dumps({"source": "tools/pointing-scenes/ensemble_score.py golden",
                                          "heads": Path(args.heads).name, "cases": cases}))
    print(f"{len(cases)} golden cases -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--heads", required=True)
    g.add_argument("--case", nargs=3, action="append", required=True, metavar=("DUMP", "SCENES", "INDEX"))
    g.add_argument("--out", required=True)
    g.set_defaults(fn=golden)
    s = sub.add_parser("select")
    s.add_argument("--distractors", nargs=2, required=True, metavar=("DUMP", "SCENES"))
    s.add_argument("--blanks", nargs=2, metavar=("DUMP", "SCENES"))
    s.add_argument("--out", required=True)
    s.set_defaults(fn=select)
    c = sub.add_parser("score")
    c.add_argument("dump")
    c.add_argument("scene_dir")
    c.add_argument("--heads", required=True)
    c.add_argument("--out")
    c.set_defaults(fn=score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
