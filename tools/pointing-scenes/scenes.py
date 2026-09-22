"""N pointing scenes with the blue button somewhere different every time.

E-P1 of `docs/specs/decide/12-the-coordinate-in-the-latent.md` fits a probe,
and a fit needs samples that move. The committed fixture is three scenes
chosen to vary the button's *size*; this varies its **position**, which is
the quantity the probe has to recover, and keeps the rest of the scene the
same kind of thing the fixture is.

Smaller than the fixture on purpose. At 4096 px an image is 16,384 merged
columns and the reference PyTorch GDN implementation wants tens of GB per
forward; at 1024 px it is 1,024 columns. The engine's own width walk found
`point` inside the button at every size from 768 px up
(`docs/findings/2026-09-21-vision-tower-cost-at-width.md` §5), so this is a
regime the endpoint is measured in — but it is **not** the regime the
pointing finding's digit traces were taken in, and a probe fitted here is
about this one.

Deliberately not reusing `generate.py`: that script regenerates the
*committed* fixture, and the committed PNGs are what three GPU tests assert
against.
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

DISTRACTOR_COLOURS = ["#c0392b", "#27ae60", "#e67e22", "#8e44ad", "#16a085"]
DISTRACTOR_LABELS = ["Cancel", "Export", "Delete", "Archive", "Share", "Run",
                     "Stop", "Halt", "Reset", "Apply"]
BLUE = "#2563eb"


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def button(draw, box, fill, label, fsize):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=max(2, (y1 - y0) // 6), fill=fill)
    f = font(fsize)
    tw = draw.textlength(label, font=f)
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - fsize * 0.65), label,
              fill="#ffffff", font=f)


def chrome(draw, side, rng):
    """The same kind of app window the fixture draws, with its furniture
    jittered so the probe cannot learn the layout instead of the button."""
    bar = int(side * rng.uniform(0.04, 0.07))
    rail = int(side * rng.uniform(0.13, 0.20))
    draw.rectangle([0, 0, side, bar], fill="#2b2f3a")
    draw.rectangle([0, bar, rail, side], fill="#f0f1f4")
    draw.rectangle([rail, bar, side, side], fill="#ffffff")
    draw.text((side * 0.015, bar * 0.28), "Project settings", fill="#ffffff",
              font=font(max(8, int(side * 0.022))))
    rows = rng.randint(4, 7)
    for i in range(rows):
        y = int(bar * 1.4 + i * side * 0.045)
        if y + side * 0.027 < side:
            draw.rectangle([side * 0.015, y, rail * 0.9, y + side * 0.027],
                           fill="#dfe1e7")


def overlaps(a, b, pad):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def place(rng, side, w, h, taken):
    for _ in range(200):
        x0 = rng.randint(0, side - w)
        y0 = rng.randint(0, side - h)
        box = (x0, y0, x0 + w, y0 + h)
        if not any(overlaps(box, t, side * 0.01) for t in taken):
            return box
    return None


PALETTE = {"blue": "#2563eb", "red": "#c0392b", "green": "#27ae60",
           "orange": "#e67e22", "purple": "#8e44ad", "teal": "#16a085"}
LABELS = ["Save", "Cancel", "Export", "Delete", "Archive", "Share", "Run",
          "Stop", "Reset", "Apply", "Upload", "Print"]


def varied_scene(rng, side):
    """The target's colour and name vary, and half the time the instruction
    names its label instead of its colour: every distractor then shares the
    palette, so no single colour identifies the target."""
    image = Image.new("RGB", (side, side), "#ffffff")
    draw = ImageDraw.Draw(image)
    chrome(draw, side, rng)
    frac = rng.uniform(0.078, 0.269)
    w = int(side * frac)
    h = int(w * rng.uniform(0.22, 0.32))
    taken = []
    target = place(rng, side, w, h, taken)
    if target is None:
        return None
    taken.append(target)
    by_label = rng.random() < 0.5
    colours = list(PALETTE)
    t_colour = rng.choice(colours)
    labels = rng.sample(LABELS, 5)
    t_label = labels[0]
    others = []
    for i in range(rng.randint(2, 4)):
        ow = int(w * rng.uniform(0.7, 1.15))
        oh = int(h * rng.uniform(0.85, 1.15))
        box = place(rng, side, ow, oh, taken)
        if box is None:
            continue
        taken.append(box)
        # by colour: distractors never share the target's colour;
        # by label: they may, and some will, which is the point.
        pool = colours if by_label else [c for c in colours if c != t_colour]
        others.append((PALETTE[rng.choice(pool)], box, labels[1 + i]))
    for colour, box, label in others:
        button(draw, box, colour, label, max(6, int((box[3] - box[1]) * 0.42)))
    button(draw, target, PALETTE[t_colour], t_label, max(6, int(h * 0.42)))
    instruction = (f"click the {t_label} button" if by_label
                   else f"click the {t_colour} button")
    return image, target, instruction, ("label" if by_label else "colour")


def scene(rng, side):
    image = Image.new("RGB", (side, side), "#ffffff")
    draw = ImageDraw.Draw(image)
    chrome(draw, side, rng)

    # The blue button's size sweeps the fixture's own range, 7.8% to 26.9%
    # of the side, so the fit is not about one size either.
    frac = rng.uniform(0.078, 0.269)
    w = int(side * frac)
    h = int(w * rng.uniform(0.22, 0.32))
    taken = []
    blue = place(rng, side, w, h, taken)
    if blue is None:
        return None
    taken.append(blue)

    others = []
    for _ in range(rng.randint(2, 4)):
        ow = int(w * rng.uniform(0.7, 1.15))
        oh = int(h * rng.uniform(0.85, 1.15))
        box = place(rng, side, ow, oh, taken)
        if box is None:
            continue
        taken.append(box)
        others.append((rng.choice(DISTRACTOR_COLOURS), box,
                       rng.choice(DISTRACTOR_LABELS)))

    for colour, box, label in others:
        button(draw, box, colour, label,
               max(6, int((box[3] - box[1]) * 0.42)))
    button(draw, blue, BLUE, "Save", max(6, int(h * 0.42)))
    return image, blue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=".scratch/latent-probe/scenes")
    ap.add_argument("--varied", action="store_true",
                    help="vary the target's colour and name it by label half the time")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    made = 0
    while made < args.n:
        s = varied_scene(rng, args.side) if args.varied else scene(rng, args.side)
        if s is None:
            continue
        if args.varied:
            image, (x0, y0, x1, y1), instruction, kind = s
        else:
            (image, (x0, y0, x1, y1)), instruction, kind = s, "click the blue button", "blue"
        name = f"scene{made:04d}.png"
        image.save(os.path.join(args.out, name), optimize=True)
        rows.append({
            "id": f"scene{made:04d}", "image": name,
            "blue_box": [x0, y0, x1, y1],
            "blue_centre": [(x0 + x1) // 2, (y0 + y1) // 2],
            "blue_size": [x1 - x0, y1 - y0],
            "blue_fraction_of_side": round((x1 - x0) / args.side, 4),
            "instruction": instruction, "kind": kind,
        })
        made += 1
    json.dump({"side": args.side, "seed": args.seed, "scenes": rows},
              open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{made} scenes at {args.side}px in {args.out}")


if __name__ == "__main__":
    main()
