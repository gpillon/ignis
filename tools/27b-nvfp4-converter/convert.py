"""The 27B NVFP4 converter (spec #449, GitHub #468): a BF16 checkpoint of the 27B, the base's or a fine-tune's,
to the work tree the `qwen3.8-27b-nvfp4` packer family packs (docs/formats/27b-nvfp4-work-tree.md), under a
weights id of its own.

  python convert.py calibrate --checkpoint DIR --out CALIBRATION.json [--device cuda]
  python convert.py run --checkpoint DIR --source-repo REPO --source-revision REV --license ID \\
      --dflash2 DIR --ranking FILE --calibration CALIBRATION.json --weights-id ID --work DIR [--device cuda]
  python convert.py verify --artifact A.ninfer --checkpoint DIR --dflash2 DIR --ranking FILE \\
      --calibration CALIBRATION.json --weights-id ID [--device cuda] [--report R.json]

- `calibrate` measures the 247 activation divisors on the checkpoint (`calibrate.py`, the GPU step).
- `run` writes the work tree: every object of the inventory from its route (`recipe.py`), every one of the
  247 text NVFP4 parents encoded here from the BF16 (`encode.py`) beside the calibration's divisor, the
  drafter's 34 weight-only; then `converter.json`, the converter's half of the sidecar, last. The packer
  (`ignis-artifact-pack --family qwen3.8-27b-nvfp4`) assembles the container; Python never writes it.
- `verify` reads a packed artifact back against its sources: its identity, its directory against the
  inventory, and every object's payload against the one its sources give (each local parent re-encoded
  byte-equal, each divisor the calibration's word). It prints VERIFY PASS or VERIFY FAILED (exit 3).

Every refusal (exit 2) names what failed and comes before anything is written. At most one safetensors
shard is open at a time (`shards.py`).
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
import torch

import calibrate as calibration_mod
import checkpoint
import container
import encode
import inventory
import recipe
import shards
import work
from checkpoint import Refused

SCHEMA = "qwen3.8-27b-nvfp4-converter-v1"
CALIBRATION_SCHEMA = "qwen3.8-27b-nvfp4-calibration-v1"
RECIPE_ID = "qwen3_8_27b_nvfp4full-v2-local"
# The served artifact's weights id: a build of this layout under it would share its content hash, and with it
# the calibrations keyed by that hash (`locate`, the pointing heads; spec #449, "Identity").
SERVED_WEIGHTS_ID = "nvfp4full"
DIVISOR_FORMULA = "d_x = binary32(2688 / max|site input| over the fixed corpus)"
EXIT_REFUSED = 2
EXIT_VERIFY_FAILED = 3


def log(message):
    print(message, flush=True)


def versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for dist in ("torch", "transformers", "accelerate", "safetensors", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def tool_commit():
    """{tool, commit, dirty}: the commit the converter ran from, and whether its files differed from it."""
    here = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"tool": "tools/27b-nvfp4-converter", "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--", "."))}
    except (OSError, subprocess.CalledProcessError):
        return {"tool": "tools/27b-nvfp4-converter", "commit": None, "dirty": None}


def device_of(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Refused(f"--device {name}, and torch sees no CUDA device")
    return device


def peaks(device):
    """This process's peaks: the CUDA allocator's (on a CUDA device) and the host's working set and commit."""
    import psutil
    mem = psutil.Process().memory_info()
    out = {"host_peak_working_set_bytes": getattr(mem, "peak_wset", None),
           "host_peak_commit_bytes": getattr(mem, "peak_pagefile", None)}
    if device.type == "cuda":
        out.update(gpu=torch.cuda.get_device_name(device),
                   vram_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   vram_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return out


def log_step(step, seconds, measured):
    gib = lambda n: "-" if n is None else f"{n / 2 ** 30:.2f} GiB"  # noqa: E731
    vram = (f"; VRAM peak {gib(measured['vram_peak_allocated_bytes'])} allocated, "
            f"{gib(measured['vram_peak_reserved_bytes'])} reserved") if "gpu" in measured else ""
    log(f"{step}: {seconds:.0f} s{vram}; host peak {gib(measured['host_peak_working_set_bytes'])} working set, "
        f"{gib(measured['host_peak_commit_bytes'])} commit")


def is_empty(path):
    return not os.path.exists(path) or (os.path.isdir(path) and not os.listdir(path))


def free_bytes(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


# ---------------------------------------------------------------------------------------------------
# The calibration's record
# ---------------------------------------------------------------------------------------------------

def read_calibration(path, sources):
    """site -> divisor, and the calibration's provenance; refused unless it was measured on this checkpoint
    (its revision and every shard's LFS oid), over this corpus, at every one of the 247 sites."""
    doc = checkpoint.read_json(path, f"the calibration {path}")
    if doc.get("schema") != CALIBRATION_SCHEMA:
        raise Refused(f"the calibration {path} is not a {CALIBRATION_SCHEMA} record")
    measured_on = doc.get("checkpoint") or {}
    if measured_on.get("revision") != sources.revision:
        raise Refused(f"the calibration was measured on revision {measured_on.get('revision')}, "
                      f"the checkpoint is at {sources.revision}")
    theirs = {n: p.get("sha256") for n, p in (measured_on.get("shards") or {}).items()}
    if theirs != sources.shard_oids:
        differ = sorted(n for n in set(theirs) | set(sources.shard_oids) if theirs.get(n) != sources.shard_oids.get(n))
        raise Refused(f"the calibration was measured on another checkpoint: its {differ[0]} is not this one's")
    if (doc.get("corpus") or {}).get("sha256") != calibration_mod.corpus_sha256():
        raise Refused("the calibration was measured over another corpus than the fixed one")
    expected = set(calibration_mod.sites(sources.geometry))
    measured = doc.get("measured_sites") or {}
    checkpoint.refuse_if([f"it has no {n}" for n in sorted(expected - set(measured))]
                         + [f"it has {n}, which is no site of the recipe" for n in sorted(set(measured) - expected)],
                         "the calibration does not cover the 247 sites")
    out = {}
    for name in sorted(expected):
        d = measured[name].get("input_scale_divisor")
        amax = measured[name].get("amax")
        if not isinstance(d, float) or not math.isfinite(d) or d <= 0.0 or encode.binary32(d) != d:
            raise Refused(f"the calibration's divisor of {name} ({d!r}) is not a finite positive binary32")
        if not isinstance(amax, float) or encode.divisor_of(amax) != d:
            raise Refused(f"the calibration's divisor of {name} is not 2688 / its amax")
        out[name] = d
    provenance = {k: doc.get(k) for k in ("encoder_profile", "divisor_formula", "corpus", "gdn_kernel", "device",
                                          "versions", "converter")}
    provenance.update(sha256=checkpoint.file_sha256(path), sites=len(out))
    return out, provenance


# ---------------------------------------------------------------------------------------------------
# The objects, from their sources
# ---------------------------------------------------------------------------------------------------

def produce(sources, table, divisors, ids, device):
    """(object, payload, NVFP4 record or None) for every object of the inventory, in its order. The
    checkpoint's reader is closed before the drafter's opens."""
    objs = inventory.objects(sources.geometry)
    with shards.ShardReader(sources.checkpoint, sources.weight_map) as reader:
        for o in objs:
            if o.unit == "dflash2":
                break
            r = table[o.name]
            if r.route == "resource":
                with open(os.path.join(sources.checkpoint, r.parts[0][0]), "rb") as f:
                    yield o, f.read(), None
            elif r.route == "input_divisor":
                yield o, struct.pack("<f", divisors[o.name]), None
            elif r.transform == "ids":
                yield o, encode.direct(torch.from_numpy(ids).to(torch.int32), inventory.I32), None
            else:
                payload, record = encode.encode(o, recipe.materialize(r, o, reader, ids), device)
                yield o, payload, record
    with shards.ShardReader.single(os.path.join(sources.drafter, checkpoint.DRAFTER_FILE)) as reader:
        for o in objs:
            if o.unit != "dflash2":
                continue
            payload, record = encode.encode(o, recipe.materialize(table[o.name], o, reader), device)
            yield o, payload, None if record is None else dict(record, weight_scale_divisor=None)


def file_name(o):
    """The work file of an object inside its unit: the name past the unit, '/' -> '.', '.bin' appended."""
    rest = o.name[len(o.unit) + 1:]
    return rest if o.kind == "resource" else rest.replace("/", ".") + ".bin"


def tree_bytes(g, checkpoint_root):
    """The work tree's payload bytes."""
    return sum(inventory.encoded_size(o) if o.kind == "tensor"
               else os.path.getsize(os.path.join(checkpoint_root, o.name.split("/", 1)[1]))
               for o in inventory.objects(g))


def write_tree(work_dir, produced):
    """Writes every unit's files, its `tensors.json` and its DONE last; returns the NVFP4 records."""
    parents = {}
    unit = entries = digests = None

    def finish():
        if unit is None:
            return
        directory = os.path.join(work_dir, unit)
        if entries:
            work.write_json_atomic(os.path.join(directory, "tensors.json"), {"tensors": entries})
        work.mark_done(directory, digests)
        log(f"{unit}: {len(digests)} files")

    for o, payload, record in produced:
        if o.unit != unit:
            finish()
            unit, entries, digests = o.unit, [], {}
            os.makedirs(os.path.join(work_dir, unit))
        name = file_name(o)
        with open(os.path.join(work_dir, unit, name), "wb") as f:
            f.write(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()
        if o.kind == "tensor":
            entries.append({"name": o.name, "file": name, "format": o.format, "layout": o.layout,
                            "shape": list(o.shape), "bytes": len(payload)})
        if record is not None:
            parents[o.name] = record
    finish()
    return parents


# ---------------------------------------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------------------------------------

def pin(sources):
    """`sources` with every file hashed against its download record, the time logged."""
    t0 = time.time()
    sources.hash()
    files = len(sources.files) + len(sources.drafter_files or {})
    log(f"pins: {files} files checked against their download records in {time.time() - t0:.0f} s")
    return sources


def run_calibrate(args, g, device):
    if os.path.exists(args.out):
        raise Refused(f"{args.out} exists: the calibration writes a new file")
    sources = pin(checkpoint.check_sources(g, args.checkpoint))
    t0 = time.time()
    peaks_, tokens, kernel = calibration_mod.measure(args.checkpoint, g, device, args.gdn_kernel, log)
    sites = calibration_mod.divisors(peaks_)
    seconds = time.time() - t0
    measured = peaks(device)
    shards_ = {n: p for n, p in sources.files.items() if n.endswith(".safetensors")}
    record = {"schema": CALIBRATION_SCHEMA, "encoder_profile": encode.ENCODER_PROFILE,
              "divisor_formula": DIVISOR_FORMULA,
              "corpus": {"documents": len(calibration_mod.DOCUMENTS), "tokens": tokens,
                         "sha256": calibration_mod.corpus_sha256()},
              "checkpoint": {"revision": sources.revision, "shards": shards_},
              "gdn_kernel": kernel, "device": str(device), "seconds": round(seconds, 1), "peaks": measured,
              "versions": versions(), "converter": tool_commit(),
              "measured_sites": {name: dict(sites[name], module=module)
                                 for name, module in calibration_mod.sites(g).items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    work.write_json_atomic(args.out, record)
    log_step(f"calibration: {len(sites)} sites over {tokens} tokens", seconds, measured)


def check_recipe(g):
    table = recipe.recipes(g)
    checkpoint.refuse_if(recipe.problems(table, g), "the recipe is not the local-only one")
    return table


def run_convert(args, g, device, disk_free=None):
    if not is_empty(args.work):
        raise Refused(f"{args.work} is not empty: the converter writes a new work tree")
    if args.weights_id == SERVED_WEIGHTS_ID:
        raise Refused(f"--weights-id {SERVED_WEIGHTS_ID} is the served artifact's: a build takes a weights id of its own")
    sources = checkpoint.check_sources(g, args.checkpoint, args.source_revision, args.dflash2, args.ranking)
    divisors, calibration = read_calibration(args.calibration, sources)
    table = check_recipe(g)
    ids, ranking = checkpoint.draft_ids(args.ranking, args.checkpoint, g)
    need = tree_bytes(g, args.checkpoint)
    free = free_bytes(args.work) if disk_free is None else disk_free
    if free < need:
        raise Refused(f"the work tree takes {need} bytes and {args.work}'s disk has {free} free")
    pin(sources)

    t0 = time.time()
    parents = write_tree(args.work, produce(sources, table, divisors, ids, device))
    errors = [p["relative_frobenius_error"] for p in parents.values()]
    record = {
        "schema": SCHEMA,
        "status": "complete",
        "recipe_id": RECIPE_ID,
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": args.weights_id},
        "source": {"repo": args.source_repo, "revision": sources.revision, "license": args.license,
                   "files": sources.files},
        "drafter": {"repo": checkpoint.PINNED[g.name]["drafter"][0], "revision": checkpoint.PINNED[g.name]["drafter"][1],
                    "files": sources.drafter_files, "config": checkpoint.drafter_config(g)},
        "ranking": dict(sources.ranking_pin, **ranking, rows=g.draft_head_rows),
        "calibration": calibration,
        "recipe": dict(sorted(recipe.counts(table).items()), source_nvfp4_parents=0),
        "local_nvfp4": {"encoder_profile": encode.ENCODER_PROFILE, "parents": parents,
                        "relative_frobenius_error_max": max(errors),
                        "relative_frobenius_error_mean": sum(errors) / len(errors)},
        "geometry": g.name,
        "device": str(device),
        "converter": tool_commit(),
        "versions": versions(),
    }
    work.write_json_atomic(os.path.join(args.work, "converter.json"), record)
    log_step(f"conversion: {len(parents)} NVFP4 parents, the work tree {need} bytes", time.time() - t0, peaks(device))


def directory_problems(found, g, weights_id):
    """The packed artifact's identity and directory against the inventory, object by object."""
    out = []
    identity = {"model_id": inventory.MODEL_ID, "weights_id": weights_id}
    if found.identity != identity:
        out.append(f"the identity is {found.identity}, not {identity}")
    expected = inventory.objects(g)
    names = [o["name"] for o in found.objects]
    out += [f"the artifact has no {o.name}" for o in expected if o.name not in found.by_name]
    out += [f"the artifact has {n}, which the inventory has not" for n in names if n not in {o.name for o in expected}]
    for o in expected:
        d = found.by_name.get(o.name)
        if d is None:
            continue
        if o.kind == "resource":
            if d.get("kind") != "resource":
                out.append(f"{o.name} is not a resource")
            continue
        got = (d.get("format"), d.get("layout"), tuple(d.get("shape") or ()), d.get("bytes"))
        want = (o.format, o.layout, o.shape, inventory.encoded_size(o))
        if got != want:
            out.append(f"{o.name} is {got[0]} {got[1]} {list(got[2])} ({got[3]} bytes), "
                       f"the inventory's is {want[0]} {want[1]} {list(want[2])} ({want[3]} bytes)")
    if not out and names != [o.name for o in expected]:
        out.append("the objects are not in the inventory's order")
    return out


def run_verify(args, g, device):
    found = container.Container(args.artifact)
    sources = checkpoint.check_sources(g, args.checkpoint, None, args.dflash2, args.ranking)
    divisors, _ = read_calibration(args.calibration, sources)
    table = check_recipe(g)
    ids, _ = checkpoint.draft_ids(args.ranking, args.checkpoint, g)
    pin(sources)
    t0 = time.time()
    failures = directory_problems(found, g, args.weights_id)
    checked = {"objects": 0, "local_nvfp4": 0, "input_divisor": 0}
    if not failures:
        for o, payload, _ in produce(sources, table, divisors, ids, device):
            stored = found.read(o.name)
            route = table[o.name].route
            checked["objects"] += 1
            checked[route] = checked.get(route, 0) + 1
            if stored == payload:
                continue
            if route == "input_divisor":
                failures.append(f"{o.name}: stores divisor {struct.unpack('<f', stored)[0]!r}, "
                                f"the calibration's is {struct.unpack('<f', payload)[0]!r}")
            else:
                n = min(len(stored), len(payload))
                differ = int(np.count_nonzero(np.frombuffer(stored, np.uint8, n) != np.frombuffer(payload, np.uint8, n)))
                differ += abs(len(stored) - len(payload))
                what = "does not re-encode byte-equal from its source" if o.format == inventory.NVFP4 \
                    else "is not its source's"
                failures.append(f"{o.name}: {what} ({differ} of {len(payload)} bytes differ)")
    report = {"artifact": os.path.basename(args.artifact), "identity": found.identity, "checked": checked,
              "failures": failures, "verdict": "VERIFY FAILED" if failures else "VERIFY PASS"}
    if args.report:
        work.write_json_atomic(args.report, report)
    for f in failures[:20]:
        log(f"  {f}")
    if len(failures) > 20:
        log(f"  ... and {len(failures) - 20} more")
    log_step(f"verify: {checked['objects']} objects", time.time() - t0, peaks(device))
    log(report["verdict"])
    return EXIT_VERIFY_FAILED if failures else 0


def parser():
    p = argparse.ArgumentParser(description="The 27B NVFP4 converter (GitHub #468).")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s):
        s.add_argument("--checkpoint", required=True, help="the BF16 checkpoint, with its download records")
        s.add_argument("--device", default="cuda")
        s.add_argument("--geometry", default="qwen3.8-27b", choices=("qwen3.8-27b", "fixture"))

    s = sub.add_parser("calibrate", help="measure the 247 activation divisors (GPU)")
    common(s)
    s.add_argument("--out", required=True, help="the calibration record to write")
    s.add_argument("--gdn-kernel", default="reference", choices=("reference", "installed"))

    for name in ("run", "verify"):
        s = sub.add_parser(name)
        common(s)
        s.add_argument("--dflash2", required=True, help="the base's DFlash2 drafter, with its download records")
        s.add_argument("--ranking", required=True, help="ninfer's ranking.train.counts.i64")
        s.add_argument("--calibration", required=True)
        s.add_argument("--weights-id", required=True)
        if name == "run":
            s.add_argument("--source-repo", required=True, help="the checkpoint's hub repository")
            s.add_argument("--source-revision", required=True, help="the revision its download records must hold")
            s.add_argument("--license", required=True, help="the license's identifier, as the catalog names it")
            s.add_argument("--work", required=True, help="the work tree to write: absent or empty")
        else:
            s.add_argument("--artifact", required=True)
            s.add_argument("--report")
    return p


def main(argv=None, disk_free=None):
    args = parser().parse_args(argv)
    g = inventory.Geometry.named(args.geometry)
    try:
        device = device_of(args.device)
        if args.command == "calibrate":
            run_calibrate(args, g, device)
        elif args.command == "run":
            run_convert(args, g, device, disk_free)
        else:
            return run_verify(args, g, device)
    except Refused as e:
        log(f"refused: {e}")
        return EXIT_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
