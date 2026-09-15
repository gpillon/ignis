"""Generate the committed Seam A image set (GitHub #176).

Deterministic synthetic pictures chosen to hit the processor's edges against
the real artifact's `preprocessor_config.json` (min 65,536 / max 16,777,216
pixels, factor 32). Content is smooth gradients plus a few hard shapes, so the
resize filter sees both flat and high-frequency regions while the PNGs stay
small.

    python tools/vision-fixtures/make_images.py crates/artifact/tests/fixtures/vision/images
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def picture(width: int, height: int, seed: int) -> Image.Image:
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255 * x / max(width - 1, 1)
    g = 255 * y / max(height - 1, 1)
    b = 127.5 + 127.5 * np.sin((x + 2 * y) / 9.0 + seed)
    rgb = np.stack([r, g, b], axis=-1)
    # Hard-edged shapes: a checker band and a solid disc.
    checker = ((x // 7 + y // 7) % 2 == 0) & (y > height * 0.55) & (y < height * 0.7)
    rgb[checker] = [250, 20, 30]
    disc = (x - width * 0.3) ** 2 + (y - height * 0.35) ** 2 < (min(width, height) * 0.18) ** 2
    rgb[disc] = [10, 200 - seed * 7 % 50, 90]
    return Image.fromarray(np.clip(rgb.round(), 0, 255).astype(np.uint8), "RGB")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Already on the 32-px grid and inside [min, max]: no resize.
    picture(640, 480, 1).save(out / "on_grid.png")
    # 6,000 pixels: upscaled to at least min_pixels.
    picture(100, 60, 2).save(out / "upscale.png")
    # 17,280,000 pixels: downscaled below max_pixels. Posterized so the PNG
    # stays small.
    picture(4800, 3600, 3).quantize(16).convert("RGB").save(out / "downscale.png", optimize=True)
    # Neither side on the grid, 3:1.
    picture(1001, 333, 4).save(out / "odd_aspect.png")
    # 195:1, just under the 200 limit.
    picture(3900, 20, 5).save(out / "strip.png")
    # 201:1, just over it (processor error).
    picture(4020, 20, 6).save(out / "strip_too_wide.png")
    # RGBA with a varying alpha channel: the reference drops alpha.
    rgba = picture(300, 200, 7).convert("RGBA")
    alpha = np.tile(np.linspace(0, 255, 300).astype(np.uint8), (200, 1))
    rgba.putalpha(Image.fromarray(alpha, "L"))
    rgba.save(out / "alpha.png")
    # A 480x320 frame stored with EXIF orientation 6 (rotate 90 CW on display).
    exif = Image.Exif()
    exif[0x0112] = 6
    picture(480, 320, 8).save(out / "exif_rotated.jpg", quality=90, exif=exif.tobytes())
    # More JPEGs on the exact path (8-bit YCbCr, 4:2:0 or 4:2:2, even
    # height): progressive, 4:2:2, restart markers every MCU row, odd width.
    picture(400, 304, 13).save(out / "progressive.jpg", quality=85, progressive=True)
    picture(320, 240, 14).save(out / "subsampled_422.jpg", quality=92, subsampling=1)
    picture(256, 200, 15).save(out / "restart.jpg", quality=80, restart_marker_rows=1)
    picture(321, 240, 16).save(out / "odd_width.jpg", quality=95)
    # Second picture for the two-images message.
    picture(320, 352, 9).save(out / "second.png")
    # Animated GIF: only the first frame is used.
    frames = [picture(256, 256, 10).quantize(64), picture(256, 256, 11).quantize(64)]
    frames[0].save(out / "animated.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    # Lossless WebP.
    picture(400, 300, 12).save(out / "lossless.webp", lossless=True)
    # Not an image at all.
    (out / "corrupt.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
