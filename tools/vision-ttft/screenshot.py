"""Generate the multimodal TTFT cell's screenshot (GitHub #181).

A fixed ~1-megapixel "screenshot" of a code editor — title bar, file tree,
source, a terminal with a failing test — drawn with PIL and the Windows
system fonts, no randomness, so re-running it is a no-op. 1280x800 sits on
the processor's 32-pixel grid, so it is not resized: 40x25 merged blocks,
1,000 vision tokens.

    python tools/vision-ttft/screenshot.py crates/bench/tests/fixtures/vision_ttft/screenshot.png

The cell itself is `ignis-bench ttft --image <this png>`; see
`docs/agents/testing.md` for the live/live run.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 800

SOURCE = [
    ("use std::collections::HashMap;", (86, 156, 214)),
    ("", None),
    ("/// Tracks how many times each route was hit.", (106, 153, 85)),
    ("pub struct RouteCounter {", (78, 201, 176)),
    ("    hits: HashMap<String, u64>,", (212, 212, 212)),
    ("}", (212, 212, 212)),
    ("", None),
    ("impl RouteCounter {", (78, 201, 176)),
    ("    pub fn record(&mut self, route: &str) {", (220, 220, 170)),
    ("        *self.hits.entry(route.to_owned()).or_insert(0) += 1;", (212, 212, 212)),
    ("    }", (212, 212, 212)),
    ("", None),
    ("    pub fn busiest(&self) -> Option<(&str, u64)> {", (220, 220, 170)),
    ("        self.hits", (212, 212, 212)),
    ("            .iter()", (212, 212, 212)),
    ("            .min_by_key(|(_, count)| **count)", (206, 145, 120)),
    ("            .map(|(route, count)| (route.as_str(), *count))", (212, 212, 212)),
    ("    }", (212, 212, 212)),
    ("}", (212, 212, 212)),
]

TERMINAL = [
    ("$ cargo test -p routes", (204, 204, 204)),
    ("running 3 tests", (204, 204, 204)),
    ("test counts_each_hit ... ok", (106, 190, 106)),
    ("test empty_has_no_busiest ... ok", (106, 190, 106)),
    ("test busiest_is_the_most_hit_route ... FAILED", (230, 90, 90)),
    ("  left: Some((\"/health\", 1))", (230, 90, 90)),
    (" right: Some((\"/api/users\", 42))", (230, 90, 90)),
]

TREE = ["v routes", "  v src", "    lib.rs", "    counter.rs", "    server.rs", "  v tests", "    busiest.rs", "  Cargo.toml", "  README.md"]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(name, size)


def screenshot() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    ui, mono = font("segoeui.ttf", 15), font("consola.ttf", 17)

    # Title bar and tabs.
    draw.rectangle((0, 0, WIDTH, 32), fill=(50, 50, 52))
    draw.text((12, 7), "counter.rs - routes - Editor", fill=(220, 220, 220), font=ui)
    draw.rectangle((240, 32, WIDTH, 64), fill=(37, 37, 38))
    draw.rectangle((240, 32, 380, 64), fill=(30, 30, 30))
    draw.text((256, 40), "counter.rs", fill=(255, 255, 255), font=ui)
    draw.text((400, 40), "busiest.rs", fill=(150, 150, 150), font=ui)

    # File tree.
    draw.rectangle((0, 32, 240, HEIGHT), fill=(37, 37, 38))
    draw.text((16, 42), "EXPLORER", fill=(190, 190, 190), font=ui)
    for row, entry in enumerate(TREE):
        y = 76 + row * 26
        if entry.strip() == "counter.rs":
            draw.rectangle((0, y - 3, 240, y + 21), fill=(4, 57, 94))
        draw.text((16, y), entry, fill=(204, 204, 204), font=ui)

    # Source, with line numbers.
    for row, (text, colour) in enumerate(SOURCE):
        y = 80 + row * 24
        draw.text((252, y), f"{row + 1:>3}", fill=(110, 118, 129), font=mono)
        if colour:
            draw.text((304, y), text, fill=colour, font=mono)

    # Terminal panel.
    top = 80 + len(SOURCE) * 24 + 12
    draw.rectangle((240, top, WIDTH, HEIGHT - 24), fill=(24, 24, 24))
    draw.line((240, top, WIDTH, top), fill=(70, 70, 70), width=1)
    draw.text((256, top + 6), "TERMINAL", fill=(190, 190, 190), font=ui)
    for row, (text, colour) in enumerate(TERMINAL):
        draw.text((256, top + 34 + row * 22), text, fill=colour, font=mono)

    # Status bar.
    draw.rectangle((0, HEIGHT - 24, WIDTH, HEIGHT), fill=(0, 122, 204))
    draw.text((12, HEIGHT - 21), "main   1 error   Ln 15, Col 28   UTF-8   Rust", fill=(255, 255, 255), font=ui)
    return image


def main(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    screenshot().save(out, format="PNG", optimize=False, compress_level=9)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
