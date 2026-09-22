"""Large coloured rectangles among distractors, and blank priors (spec 14).

The pointing studies before spec 14 measured small labelled buttons only, where
an object's corner and its centre are the same image token. This generator
draws the regime that told them apart: a target rectangle of one colour among
one to three distractor rectangles of other colours, asked for by colour
("Where is the green rectangle?"). Its `manifest.json` has the fields
`crates/server/tests/attention_head_point_gpu.rs` reads (`blue_box` is the
harness's name for the target's box whatever its colour) plus
`study.others`, the distractors' boxes, which `ensemble_score.py select`
scores selectivity against.

`--blank` writes blank priors instead: grey, black, white, noise and a
gradient, each asked for a red rectangle that is not there. They show where
heads look when nothing matches -- the fallback cells `ensemble_score.py
select` excludes.

Deterministic per `--seed`. Sizes scale with `--side` (1024 px: target short
side 96-480 px, distractors 64-360 px). Needs Pillow and NumPy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {"red": (220, 30, 30), "green": (40, 170, 60), "blue": (30, 60, 220),
           "yellow": (235, 200, 20), "purple": (140, 50, 180)}
GREY = (200, 200, 200)


def rectangles(out, n, side, seed):
    rng = np.random.default_rng(seed)
    k = side / 1024

    def rbox(lo, hi):
        w, h = int(rng.integers(round(lo * k), round(hi * k))), int(rng.integers(round(lo * k), round(hi * k)))
        m = round(16 * k)
        x0, y0 = int(rng.integers(m, side - m - w)), int(rng.integers(m, side - m - h))
        return (x0, y0, x0 + w, y0 + h)

    def apart(a, b):
        pad = 24 * k
        return a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1]

    names = list(COLOURS)
    scenes = []
    for i in range(n):
        colour = names[i % len(names)]
        n_dist = int(rng.integers(1, 4))
        for _ in range(1000):
            target = rbox(96, 480)
            others = [rbox(64, 360) for _ in range(n_dist)]
            boxes = [target] + others
            if all(apart(boxes[a], boxes[b]) for a in range(len(boxes)) for b in range(a + 1, len(boxes))):
                break
        else:
            raise SystemExit(f"scene {i}: could not place {n_dist + 1} rectangles apart")
        others_c = rng.choice([c for c in names if c != colour], size=n_dist, replace=False)
        im = Image.new("RGB", (side, side), GREY)
        d = ImageDraw.Draw(im)
        d.rectangle([target[0], target[1], target[2] - 1, target[3] - 1], fill=COLOURS[colour])
        for b, c in zip(others, others_c):
            d.rectangle([b[0], b[1], b[2] - 1, b[3] - 1], fill=COLOURS[str(c)])
        sid = f"rect{i:04d}"
        im.save(out / f"{sid}.png")
        scenes.append({"id": sid, "image": f"{sid}.png", "blue_box": list(target),
                       "blue_centre": [(target[0] + target[2]) // 2, (target[1] + target[3]) // 2],
                       "instruction": f"Where is the {colour} rectangle?", "kind": "rectangle",
                       "study": {"colour": colour, "others": [list(b) for b in others],
                                 "others_colours": [str(c) for c in others_c]}})
    return scenes


def blanks(out, side, seed):
    rng = np.random.default_rng(seed)
    g = np.tile(np.linspace(0, 255, side), (side, 1))
    fields = {"grey": np.full((side, side, 3), GREY), "black": np.zeros((side, side, 3)),
              "white": np.full((side, side, 3), 255), "noise": rng.integers(0, 256, (side, side, 3)),
              "gradient": np.stack([g] * 3, -1)}
    scenes = []
    for name, arr in fields.items():
        sid = f"blank-{name}"
        Image.fromarray(arr.astype(np.uint8)).save(out / f"{sid}.png")
        scenes.append({"id": sid, "image": f"{sid}.png", "blue_box": [0, 0, 1, 1], "blue_centre": [0, 0],
                       "instruction": "Where is the red rectangle?", "kind": "blank", "study": {"blank": name}})
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blank", action="store_true", help="write the five blank priors instead")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = blanks(out, a.side, a.seed) if a.blank else rectangles(out, a.n, a.side, a.seed)
    (out / "manifest.json").write_text(json.dumps(
        {"side": a.side, "seed": a.seed, "source": f"rectangles.py{' --blank' if a.blank else ''}", "scenes": scenes},
        indent=1))
    print(f"{len(scenes)} scenes at {a.side}px in {out}")


if __name__ == "__main__":
    main()
