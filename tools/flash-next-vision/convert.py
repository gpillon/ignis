"""Flash-Next's vision companion container (ADR 0087, docs/formats/flash-next-layout.md §14).

    python tools/flash-next-vision/convert.py build --main <main .ninfer> --work <absent or empty dir>
                                                   --source <cache dir> [--out <companion>] [--packer <exe>]
    python tools/flash-next-vision/convert.py fetch --source <cache dir>

`build` is the one command. It reads the pinned revision's 333 `model.visual.*` tensors, which are one
contiguous byte range of the checkpoint's first shard: the index and `config.json`, that shard's header by
HTTP range, then the range itself. No other byte of the checkpoint is read. The range is cached in
`--source` (`visual.bin`, its record `visual.json`) and held to its pin, the range's SHA-256; a later run
reads the cache and nothing from the hub. The tower is encoded with the 27B tower's recipe
(`tools/27b-nvfp4-converter`: its inventory, routes and encoders, unchanged) at Flash-Next's widths, where
the merger writes 2,560 columns, into the work tree's one unit `vision/`; `converter.json` records the
source range, the pair with `--main` (its whole-file SHA-256) and the tool's commit. Then the packer
(`ignis-artifact-pack --family flash-next-vision`) assembles the companion, pins it to `--main` and merges
the record into `<companion>.conversion.json`; the build reads the companion back against the bytes it
encoded and writes `<companion>.sha256`.

A second run gives the same bytes: the companion, its sidecar and its checksum. Nothing in the record
depends on the clock or on a local path (`converter.command` names `--main` and `--out` by file name and
the directories by placeholders), and the sidecar carries measured values only.

`fetch` fills the cache alone: the BF16 tower of shard 1, for anything that compares against it.

Exit status: 0 done, 2 refused (nothing written to `--work` or `--out`), 1 the packer failed.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RECIPE_TOOL = os.path.join(REPO_ROOT, "tools", "27b-nvfp4-converter")
sys.path.insert(1, RECIPE_TOOL)

import torch  # noqa: E402

import container  # noqa: E402  (the 27B NVFP4 converter's modules, flat files)
import encode  # noqa: E402
import inventory  # noqa: E402
import recipe  # noqa: E402
import work  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
PREFIX = "model.visual."
SCHEMA = "flash-next-vision-converter-v1"
IDENTITY = {"model_id": "qwen3.8-flash-next-vision", "weights_id": "vision-w8q4q5q6-bf16-de4b8e4"}
PART = 32 << 20
WORKERS = 8
DTYPE = {"BF16": torch.bfloat16}
WORD = {"BF16": 2}


@dataclasses.dataclass(frozen=True)
class Tower:
    """A tower's widths (the packer's `--geometry`) and, for the real one, the pin of its source range."""
    name: str
    hidden: int
    intermediate: int
    patch_size: int
    positions: int
    out_hidden: int
    shard: str = None
    span_bytes: int = None
    span_sha256: str = None

    def config(self):
        """The `vision_config` members the checkpoint must state."""
        return {"depth": inventory.VISION_LAYERS, "hidden_size": self.hidden, "intermediate_size": self.intermediate,
                "in_channels": 3, "patch_size": self.patch_size, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": self.positions, "out_hidden_size": self.out_hidden}

    def geometry(self):
        """The 27B recipe's geometry with this tower: its `vision/` objects are the companion's."""
        return dataclasses.replace(inventory.Geometry.qwen3_8_27b(), name=self.name, hidden=self.out_hidden,
                                   vision_hidden=self.hidden, vision_intermediate=self.intermediate,
                                   vision_patch_size=self.patch_size, vision_positions=self.positions)


GEOMETRIES = {
    "qwen3.8-flash-next": Tower("qwen3.8-flash-next", 1152, 4304, 16, 48 * 48, 2560,
                                shard="model-00001-of-00131.safetensors", span_bytes=897_862_112,
                                span_sha256="665127beaa6d168ab9ff4e22c6c3df3ea012bf0097550e6fb52d0a84103d4eb0"),
    "fixture": Tower("fixture", 32, 64, 4, 4 * 4, 64),
}


class Refused(Exception):
    """A check failed before anything was written to the work tree or the companion."""


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------------------------- the hub
class HttpHub:
    """The checkpoint's files at one revision on the Hugging Face hub: small files whole, shards by range."""

    def __init__(self, repo=REPO, revision=REVISION):
        import httpx
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self.client = httpx.Client(timeout=120, follow_redirects=True)

    def _get(self, name, headers, want):
        last = None
        for attempt in range(8):
            try:
                r = self.client.get(self.base + name, headers=headers)
                if r.status_code in (200, 206) and (want is None or len(r.content) == want):
                    return r.content
                last = IOError(f"{name}: status {r.status_code}, {len(r.content)} bytes")
            except Exception as e:  # network errors of any kind: retry with backoff
                last = e
            time.sleep(2 + 3 * attempt)
        raise IOError(f"{name}: {last!r}")

    def file(self, name):
        return self._get(name, {}, None)

    def range(self, name, start, end):
        return self._get(name, {"Range": f"bytes={start}-{end - 1}"}, end - start)


# ---------------------------------------------------------------------------------------------- the tower
def tower_objects(tower):
    """The companion's objects, in container order: the 27B recipe's `vision/` objects at `tower`'s widths."""
    return [o for o in inventory.objects(tower.geometry()) if o.unit == "vision"]


def tower_routes(tower):
    """object name -> (source tensor, its shape in the checkpoint, the recipe's transform)."""
    g = tower.geometry()
    table = recipe.recipes(g)
    out = {}
    for o in tower_objects(tower):
        r = table[o.name]
        (name, rows), = r.parts
        if r.route != "official" or rows is not None:
            raise Refused(f"{o.name}: the 27B recipe routes it {r.route}, not straight from one checkpoint tensor")
        shape = (g.vision_hidden, 3, 2, g.vision_patch_size, g.vision_patch_size) if r.transform == "flatten" else o.shape
        out[o.name] = (name, shape, r.transform)
    return out


def check_config(cfg, tower):
    """Refuses a checkpoint whose `vision_config` is not the tower's, naming the first member that differs."""
    vision = cfg.get("vision_config") or {}
    for key, want in tower.config().items():
        if vision.get(key) != want:
            raise Refused(f"config.json's vision_config.{key} is {vision.get(key)!r}, the {tower.name} tower's is {want}")


def locate(index, header_of, tower):
    """(shard, header base, {tensor: (dtype, shape, start, end)}, span start, span end) of the visual tensors,
    refusing a set that is not the recipe's (a missing, extra, mis-shaped or non-BF16 tensor), a tower over two
    shards or a range with another tensor inside it."""
    routes = tower_routes(tower)
    want = {name: shape for name, shape, _ in routes.values()}
    found = sorted(n for n in index if n.startswith(PREFIX))
    missing, extra = sorted(set(want) - set(found)), sorted(set(found) - set(want))
    if missing or extra:
        raise Refused("the checkpoint's visual tensors are not the tower's: "
                      + "; ".join([f"missing {n}" for n in missing] + [f"extra {n}" for n in extra]))
    shards = sorted({index[n] for n in found})
    if len(shards) != 1:
        raise Refused(f"the visual tensors are in {len(shards)} shards ({', '.join(shards)}), not one")
    shard = shards[0]
    if tower.shard is not None and shard != tower.shard:
        raise Refused(f"the visual tensors are in {shard}, the pin's are in {tower.shard}")
    base, header = header_of(shard)
    tensors = {}
    for name in found:
        meta = header.get(name)
        if meta is None:
            raise Refused(f"{shard}'s header has no {name}, which the index puts there")
        if meta["dtype"] != "BF16":
            raise Refused(f"{name} is {meta['dtype']}, not BF16")
        if tuple(meta["shape"]) != tuple(want[name]):
            raise Refused(f"{name} is {list(meta['shape'])} in {shard}, the {tower.name} tower's is {list(want[name])}")
        start, end = meta["data_offsets"]
        tensors[name] = ("BF16", tuple(meta["shape"]), base + start, base + end)
    lo = min(t[2] for t in tensors.values())
    hi = max(t[3] for t in tensors.values())
    inside = [n for n, meta in header.items() if n not in tensors and base + meta["data_offsets"][0] < hi
              and base + meta["data_offsets"][1] > lo]
    if inside or sum(t[3] - t[2] for t in tensors.values()) != hi - lo:
        raise Refused(f"the visual tensors are not one contiguous range of {shard}: {', '.join(inside) or 'gaps'}")
    if tower.span_bytes is not None and hi - lo != tower.span_bytes:
        raise Refused(f"the visual range of {shard} is {hi - lo} bytes, the pin's {tower.span_bytes}")
    return shard, tensors, lo, hi


def shard_header(hub, shard):
    """(the byte the shard's data starts at, its header without `__metadata__`), read by range."""
    n = int.from_bytes(hub.range(shard, 0, 8), "little")
    header = json.loads(hub.range(shard, 8, 8 + n))
    header.pop("__metadata__", None)
    return 8 + n, header


# ---------------------------------------------------------------------------------------------- the source cache
def fetch(hub, source, tower):
    """The visual range as bytes and its record, from the cache or (once) from the hub, held to the record
    and the pin. The cache is `visual.bin` and `visual.json` (the revision, the shard, the range, its
    SHA-256 and every tensor's place in it)."""
    os.makedirs(source, exist_ok=True)
    data_path, record_path = os.path.join(source, "visual.bin"), os.path.join(source, "visual.json")
    if os.path.isfile(record_path):
        record = json.load(open(record_path, encoding="utf-8"))
        if (record["repo"], record["revision"]) != (REPO, REVISION):
            raise Refused(f"{record_path} caches {record['repo']}@{record['revision']}, not {REPO}@{REVISION}")
        data = open(data_path, "rb").read() if os.path.isfile(data_path) else b""
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != record["bytes"] or digest != record["sha256"]:
            raise Refused(f"{data_path} is {len(data)} bytes with sha256 {digest}; {record_path} records "
                          f"{record['bytes']} bytes with sha256 {record['sha256']}")
        check_config(record["config"], tower)
        routes = tower_routes(tower)
        want = {name: list(shape) for name, shape, _ in routes.values()}
        got = {name: t["shape"] for name, t in record["tensors"].items()}
        if got != want:
            raise Refused(f"{record_path} caches another tower than the {tower.name} one")
    else:
        cfg = json.loads(hub.file("config.json"))
        check_config(cfg, tower)
        index = json.loads(hub.file("model.safetensors.index.json"))["weight_map"]
        headers = {}

        def header_of(shard):
            headers[shard] = shard_header(hub, shard)
            return headers[shard]
        shard, tensors, lo, hi = locate(index, header_of, tower)
        parts = [(a, min(a + PART, hi)) for a in range(lo, hi, PART)]
        with ThreadPoolExecutor(WORKERS) as pool:
            data = b"".join(pool.map(lambda p: hub.range(shard, *p), parts))
        digest = hashlib.sha256(data).hexdigest()
        record = {"repo": REPO, "revision": REVISION, "file": shard, "offset": lo, "bytes": hi - lo, "sha256": digest,
                  "config": {"vision_config": cfg["vision_config"]},
                  "tensors": {name: {"dtype": t[0], "shape": list(t[1]), "start": t[2] - lo, "end": t[3] - lo}
                              for name, t in sorted(tensors.items())}}
        with open(data_path + ".tmp", "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(data_path + ".tmp", data_path)
        work.write_json_atomic(record_path, record)
        log(f"fetched {hi - lo} bytes of {shard} [{lo}, {hi}): sha256 {digest}")
    if tower.span_sha256 is not None and record["sha256"] != tower.span_sha256:
        raise Refused(f"the visual range's sha256 is {record['sha256']}, the pin's {tower.span_sha256}")
    return data, record


def source_tensors(data, record):
    """name -> BF16 tensor, each a view of its bytes in the cached range."""
    buffer = bytearray(data)
    return {name: torch.frombuffer(buffer, dtype=DTYPE[t["dtype"]], offset=t["start"],
                                   count=(t["end"] - t["start"]) // WORD[t["dtype"]]).reshape(t["shape"])
            for name, t in record["tensors"].items()}


# ---------------------------------------------------------------------------------------------- the build
def tool_commit():
    """(commit, dirty): the commit the tool ran from, and whether the files that make the companion (this
    tool, the 27B recipe it runs, the packer's crate) differed from it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        dirty = git("status", "--porcelain", "--", "tools/flash-next-vision", "tools/27b-nvfp4-converter",
                    "crates/artifact")
        return git("rev-parse", "HEAD"), bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def command(args):
    """The command as run, with no local path: the containers by file name, the directories by placeholder."""
    out = ["tools/flash-next-vision/convert.py", "build", "--geometry", args.geometry,
           "--main", os.path.basename(args.main), "--work", "<work>", "--source", "<source>",
           "--out", os.path.basename(args.out), "--packer", "<packer>"]
    return " ".join(out)


def versions():
    import numpy
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__}


def default_packer():
    """IGNIS_ARTIFACT_PACK, else this checkout's release build of the packer, else its debug build."""
    if os.environ.get("IGNIS_ARTIFACT_PACK"):
        return os.environ["IGNIS_ARTIFACT_PACK"]
    for build in ("release", "debug"):
        for target in (os.path.join("x86_64-pc-windows-msvc", build), build):
            for exe in ("ignis-artifact-pack.exe", "ignis-artifact-pack"):
                path = os.path.join(REPO_ROOT, "target", target, exe)
                if os.path.isfile(path):
                    return path
    return None


def check_outputs(args):
    if not os.path.isfile(args.main):
        raise Refused(f"--main {args.main} is not a file")
    if os.path.exists(args.work) and (not os.path.isdir(args.work) or os.listdir(args.work)):
        raise Refused(f"--work {args.work} is not empty")
    for path in (args.out, args.out + ".conversion.json", args.out + ".pack-state.json"):
        if os.path.exists(path):
            raise Refused(f"{path} exists: the build writes a new companion")
    if args.packer is None or not os.path.isfile(args.packer):
        raise Refused("no packer: pass --packer, set IGNIS_ARTIFACT_PACK or build it "
                      "(cargo build --release -p ignis-artifact --bin ignis-artifact-pack)")


def encode_tower(tensors, tower):
    """(object, payload) of every companion object, in container order: the 27B recipe's encoders on the
    CPU, each matrix's group scales computed on the host (`encode.grouped`)."""
    routes = tower_routes(tower)
    for o in tower_objects(tower):
        name, _, transform = routes[o.name]
        t = tensors[name]
        if transform == "flatten":
            t = t.reshape(o.shape)
        payload, _ = encode.encode(o, t, "cpu")
        yield o, payload


def write_unit(directory, encoded):
    """The unit's files, its `tensors.json` and its DONE last; returns {object: payload sha256}."""
    os.makedirs(directory)
    entries, digests, by_object = [], {}, {}
    for o, payload in encoded:
        file = o.name[len("vision/"):].replace("/", ".") + ".bin"
        with open(os.path.join(directory, file), "wb") as f:
            f.write(payload)
        digests[file] = by_object[o.name] = hashlib.sha256(payload).hexdigest()
        entries.append({"name": o.name, "file": file, "format": o.format, "layout": o.layout,
                        "shape": list(o.shape), "bytes": len(payload)})
    work.write_json_atomic(os.path.join(directory, "tensors.json"), {"tensors": entries})
    work.mark_done(directory, digests)
    return by_object


def read_back(path, digests):
    """The packed companion against what was encoded: its identity, its objects in order, every payload."""
    c = container.Container(path)
    if c.identity != IDENTITY:
        raise RuntimeError(f"{path} has identity {c.identity}, not {IDENTITY}")
    names = [o["name"] for o in c.objects]
    if names != list(digests):
        raise RuntimeError(f"{path}'s objects are not the ones encoded")
    for name, digest in digests.items():
        if hashlib.sha256(c.read(name)).hexdigest() != digest:
            raise RuntimeError(f"{path}: {name}'s payload is not the one encoded")


def run_build(args, hub):
    tower = GEOMETRIES[args.geometry]
    args.out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.main)), "qwen3_8_flash_next_vision-v2.ninfer")
    # absolute: Windows does not start a relative path with forward slashes
    args.packer = os.path.abspath(args.packer) if args.packer else default_packer()
    check_outputs(args)
    t0 = time.time()
    data, record = fetch(hub, args.source, tower)
    tensors = source_tensors(data, record)
    log(f"source: {len(tensors)} tensors, {record['bytes']} bytes, {time.time() - t0:.0f} s")
    t1 = time.time()
    main_sha256 = work.file_digest(args.main)
    log(f"--main {os.path.basename(args.main)}: sha256 {main_sha256}, {time.time() - t1:.0f} s")

    t1 = time.time()
    os.makedirs(args.work, exist_ok=True)
    digests = write_unit(os.path.join(args.work, "vision"), encode_tower(tensors, tower))
    objs = tower_objects(tower)
    formats = {}
    for o in objs:
        formats[o.format] = formats.get(o.format, 0) + 1
    commit, dirty = tool_commit()
    converter = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {"repo": REPO, "revision": REVISION, "file": record["file"], "prefix": PREFIX,
                   "tensors": len(record["tensors"]), "offset": record["offset"], "bytes": record["bytes"],
                   "sha256": record["sha256"]},
        "pair": {"main": {"file": os.path.basename(args.main), "file_sha256": main_sha256}},
        "tower": {"layers": inventory.VISION_LAYERS, "hidden": tower.hidden, "intermediate": tower.intermediate,
                  "patch": tower.geometry().vision_patch, "positions": tower.positions,
                  "out_hidden": tower.out_hidden},
        "recipe": {"of": "the 27B tower's: tools/27b-nvfp4-converter's inventory, routes and encoders",
                   "objects": len(objs), "formats": dict(sorted(formats.items())),
                   "payload_bytes": sum(inventory.encoded_size(o) for o in objs)},
        "converter": {"commit": commit, "dirty": dirty, "command": command(args)},
        "versions": versions(),
    }
    work.write_json_atomic(os.path.join(args.work, "converter.json"), converter)
    log(f"work tree: {len(digests)} objects encoded, {time.time() - t1:.0f} s")

    cmd = [args.packer, "--family", "flash-next-vision", "--work", args.work, "--pair-main", args.main,
           "--out", args.out, "--geometry", args.geometry]
    done = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    if done.returncode != 0:
        log(f"the packer exited {done.returncode}")
        return 1
    read_back(args.out, digests)
    digest = work.file_digest(args.out)
    with open(args.out + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{digest}  {os.path.basename(args.out)}\n")
    log(f"{os.path.basename(args.out)}: {os.path.getsize(args.out)} bytes, sha256 {digest}, "
        f"{time.time() - t0:.0f} s in all")
    return 0


def run_fetch(args, hub):
    _, record = fetch(hub, args.source, GEOMETRIES[args.geometry])
    log(f"{record['file']} [{record['offset']}, {record['offset'] + record['bytes']}): sha256 {record['sha256']}")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the companion and its sidecar, from the pinned revision's visual tensors")
    b.add_argument("--main", required=True, help="the main container the companion is pinned to")
    b.add_argument("--work", required=True, help="the work tree (absent or empty); the packer consumes its unit")
    b.add_argument("--out", help="the companion (default: beside --main, under its fixed name)")
    b.add_argument("--packer", help="ignis-artifact-pack (default: IGNIS_ARTIFACT_PACK, else this checkout's build)")
    f = sub.add_parser("fetch", help="the visual range into the cache, alone")
    for s in (b, f):
        s.add_argument("--source", required=True, help="the cache of the visual range (visual.bin, visual.json)")
        s.add_argument("--geometry", choices=sorted(GEOMETRIES), default="qwen3.8-flash-next")
    b.set_defaults(fn=run_build)
    f.set_defaults(fn=run_fetch)
    return ap


def main(argv=None, hub=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args, hub or HttpHub())
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
