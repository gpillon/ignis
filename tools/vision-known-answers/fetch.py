"""The known-answer set's public images, from their primary source (GitHub #560, spec #555's seam 1).

    python -I tools/vision-known-answers/fetch.py --root <dir> [--manifest <manifest.json>] [--tower <dir>]

Every public entry of the manifest is a Wikimedia Commons file: the manifest records its page, the original's
URL, its SHA-1 and size as Commons reports them, its license and its author. `fetch` downloads each original
into its own directory, `<root>/<id>/original.<ext>`, holds it to the recorded SHA-1 and size (a mismatch is
refused and nothing is kept), and writes the copy the scorer sends, `<root>/<id>/scored.png`: the original
turned upright by its EXIF orientation, as Commons shows it, in RGB (a transparent one flattened on white), its
longer side brought down to `MAX_SIDE` pixels (Lanczos; a smaller image is not enlarged). The bound keeps every
item at most 1,600 merged tokens, so both families encode the same patches whatever their item bounds. A file
already there and matching is not downloaded again. The images are not part of the repository; their licenses are recorded per
image in the manifest.

`--tower <dir>` also writes the tower check's share of the photos (`crates/server/tests/
vision_tower_dump_gpu.rs` reads them from `IGNIS_TOWER_IMAGES`): `TOWER_COUNT` entries spread over the
manifest's order, each brought to a longer side taken in turn from `TOWER_SIDES`, as `<dir>/<id>.png`.

Downloaded files are untrusted data: they are opened as images and nothing else. Run it with `python -I`.
Needs Pillow.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SIDE = 1280
TOWER_COUNT = 12
TOWER_SIDES = (384, 640, 896, 1280)
USER_AGENT = "ignis-tests/0.1 (https://github.com/gpillon/ignis; known-answer set)"
# The licenses an image may carry (Commons' LicenseShortName, lowercased): public domain, CC0, CC BY and
# CC BY-SA of any version or port. Anything else -- non-commercial, no-derivatives, GFDL alone -- is refused.
LICENSES = re.compile(r"^(?:cc0(?: 1\.0)?|public domain|pd(?:-[a-z0-9.-]+)?|"
                      r"cc by(?:-sa)? [1-4]\.[05](?: [a-z]{2,3})?)$")


class Refused(Exception):
    """A download that is not the manifest's file."""


def license_allowed(name):
    return LICENSES.match(name.strip().lower()) is not None


def download(url):
    last = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError as e:  # network errors of any kind: retry with backoff
            last = e
            time.sleep(2 + 3 * attempt)
    raise Refused(f"{url}: {last!r}")


def derive(original, scored, max_side=MAX_SIDE):
    """`original` upright, RGB (a transparent one flattened on white), its longer side at most `max_side`, as a
    PNG at `scored`."""
    from PIL import Image, ImageOps
    with Image.open(original) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.new("RGB", rgba.size, (255, 255, 255))
            image.paste(rgba, mask=rgba.getchannel("A"))
        else:
            image = image.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
        image.save(scored + ".tmp", format="PNG")
    os.replace(scored + ".tmp", scored)


def fetch_one(item, root):
    """The scored copy of one public entry, downloading and checking its original first if need be."""
    source = item["source"]
    directory = os.path.join(root, item["id"])
    extension = os.path.splitext(source["url"])[1].lower() or ".bin"
    original = os.path.join(directory, "original" + extension)
    scored = os.path.join(directory, "scored.png")

    def pinned(data):
        return len(data) == source["bytes"] and hashlib.sha1(data).hexdigest() == source["sha1"]
    if not (os.path.isfile(original) and pinned(open(original, "rb").read())):
        data = download(source["url"])
        if not pinned(data):
            raise Refused(f"{item['id']}: {source['url']} gave {len(data)} bytes with sha1 "
                          f"{hashlib.sha1(data).hexdigest()}, the manifest records {source['bytes']} bytes with sha1 "
                          f"{source['sha1']}")
        os.makedirs(directory, exist_ok=True)
        with open(original + ".tmp", "wb") as f:
            f.write(data)
        os.replace(original + ".tmp", original)
        if os.path.exists(scored):
            os.remove(scored)
    if not os.path.isfile(scored):
        derive(original, scored)
    return scored


def tower_images(public, root, out):
    """The tower check's photos: `TOWER_COUNT` entries spread over `public`, each at a side of `TOWER_SIDES`."""
    os.makedirs(out, exist_ok=True)
    step = max(1, len(public) // TOWER_COUNT)
    chosen = public[::step][:TOWER_COUNT]
    for k, item in enumerate(chosen):
        derive(os.path.join(root, item["id"], "scored.png"), os.path.join(out, item["id"] + ".png"),
               TOWER_SIDES[k % len(TOWER_SIDES)])
    return [item["id"] for item in chosen]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="where each image gets its own directory")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--tower", help="also write the tower check's photos here")
    args = ap.parse_args(argv)
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    public = [item for item in manifest["images"] if "url" in item.get("source", {})]
    failed = []
    for item in public:
        if not license_allowed(item["source"]["license"]):
            failed.append(f"{item['id']}: license {item['source']['license']!r} is not one the set may use")
            continue
        try:
            print(fetch_one(item, args.root), flush=True)
        except Refused as e:
            failed.append(str(e))
        time.sleep(0.2)
    for line in failed:
        print(f"refused: {line}", file=sys.stderr, flush=True)
    if failed:
        return 2
    if args.tower:
        for item_id in tower_images(public, args.root, args.tower):
            print(f"tower: {item_id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
