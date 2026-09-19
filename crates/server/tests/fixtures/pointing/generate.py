"""Regenerate the pointing scenes for `classify_pointing_gpu.rs`.

Three synthetic 4096x4096 app screenshots, each with a blue "Save" button at a
known centre beside distractors of other colours, so that "the blue one" is a
real choice rather than the only rectangle. The blue button is large (26.9% of
the side), medium (13.7%) and small (7.8%).

The committed PNGs are the fixture; this script records how they were made and
lets the set be extended. It is **not** reproducible byte for byte: the button
labels are drawn with whatever font the machine has, so re-running it changes
the images and the numbers measured against them.

    python crates/server/tests/fixtures/pointing/generate.py
"""

import json
import os

from PIL import Image, ImageDraw, ImageFont

SIDE = 4096
OUT = os.path.dirname(os.path.abspath(__file__))


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def button(draw, box, fill, label, fsize):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=max(8, (y1 - y0) // 6), fill=fill)
    f = font(fsize)
    tw = draw.textlength(label, font=f)
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1) / 2 - fsize * 0.65), label, fill="#ffffff", font=f)


def chrome(draw):
    """A plausible app window: title bar, sidebar, content panel."""
    draw.rectangle([0, 0, SIDE, 220], fill="#2b2f3a")
    draw.rectangle([0, 220, 700, SIDE], fill="#f0f1f4")
    draw.rectangle([700, 220, SIDE, SIDE], fill="#ffffff")
    draw.text((60, 60), "Project settings", fill="#ffffff", font=font(90))
    for i in range(6):
        y = 320 + i * 180
        draw.rectangle([60, y, 620, y + 110], fill="#dfe1e7")


SCENES = [
    {
        "id": "large",
        "blue": (2600, 3200, 3700, 3480),
        "others": [("#c0392b", (1200, 3200, 2300, 3480), "Cancel"),
                   ("#27ae60", (1200, 2600, 2300, 2880), "Export")],
    },
    {
        "id": "medium",
        "blue": (1000, 900, 1560, 1050),
        "others": [("#c0392b", (1700, 900, 2260, 1050), "Delete"),
                   ("#e67e22", (2400, 900, 2960, 1050), "Archive"),
                   ("#27ae60", (1000, 1200, 1560, 1350), "Share")],
    },
    {
        "id": "small",
        "blue": (3500, 600, 3820, 690),
        "others": [("#c0392b", (3100, 600, 3420, 690), "Stop"),
                   ("#27ae60", (3500, 760, 3820, 850), "Run"),
                   ("#e67e22", (3100, 760, 3420, 850), "Halt")],
    },
]


def main():
    manifest = []
    for scene in SCENES:
        image = Image.new("RGB", (SIDE, SIDE), "#ffffff")
        draw = ImageDraw.Draw(image)
        chrome(draw)
        x0, y0, x1, y1 = scene["blue"]
        for colour, box, label in scene["others"]:
            button(draw, box, colour, label, max(24, int((box[3] - box[1]) * 0.42)))
        button(draw, scene["blue"], "#2563eb", "Save", max(24, int((y1 - y0) * 0.42)))
        name = f"{scene['id']}.png"
        image.save(os.path.join(OUT, name), optimize=True)
        manifest.append({
            "id": scene["id"],
            "image": name,
            "width": SIDE,
            "height": SIDE,
            "blue_box": [x0, y0, x1, y1],
            "blue_centre": [(x0 + x1) // 2, (y0 + y1) // 2],
            "blue_size": [x1 - x0, y1 - y0],
            "blue_fraction_of_side": round((x1 - x0) / SIDE, 4),
        })
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"side": SIDE, "scenes": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
