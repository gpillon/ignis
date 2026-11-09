"""Flash-Next's encoded vision tower against its BF16 tower, with the 27B's as the bar (GitHub #560, spec #555's
seam 2, ADR 0087).

    python tools/flash-next-vision/tower_check.py fetch-27b --source <dir>
    python tools/flash-next-vision/tower_check.py reference --family <family> --source <dir> --dump <dir>
                                                  [--threads 6] [--device cpu]
    python tools/flash-next-vision/tower_check.py compare --dump <dir> [--out <json>] [--csv <csv>]

The served side is a dump: `crates/server/tests/vision_tower_dump_gpu.rs` encodes each image through the
family's own vision processor and media encode, the way the server does, and writes `<dump>/<family>/`:
`manifest.json` (`DUMP_SCHEMA`: the family, the artifact, the residency, the hidden width, and per item its id,
image, grid in patches `[t, h, w]` and merged tokens) and per item `<id>.patches.bf16` (the processor's BF16
patch rows, `[t*h*w][3*2*16*16]`, the encode's very input) and `<id>.embedding.bf16` (the embedding rows the
pool holds, `[tokens][hidden]`). Every file is little-endian, BF16 as its bits. The families are
`qwen3.8-flash-next` (hidden 2,560) and `qwen3.8-27b` (5,120).

`fetch-27b` caches the 27B's BF16 tower, the 333 `model.visual.*` tensors of the first shard of
`Qwen/Qwen3.8-27B` at the revision its served tower was converted from, by HTTP range (`convert.fetch`; the
rest of the checkpoint is never read). Flash-Next's is `convert.py fetch`'s cache.

`reference` runs, on the CPU, the checkpoint's own tower as transformers implements it (ADR 0043): the vision
model of the cached config's model type (`qwen4_exp` for Flash-Next, `qwen3_5` for the 27B), built from the
cached BF16 tensors, computing in float32, on each item's dumped patches; it writes `<id>.reference.f32`
(`[tokens][hidden]`) and `reference.json` beside the dump. Both sides read the same patches, so what differs is
the tower: its stored formats and the encode's arithmetic.

`compare` reads both families' dumps and references and writes one record:
- per image: the relative L2 of the served rows against the reference rows (`rel_l2`, the item's), each row's
  relative L2 (median, p90, max), and the cosine per row centred on the family's reference set mean (raw cosine
  ~0.995 is what any two rows of one tower give, so it is reported only beside the centred one);
- per image, two controls by the same measure, which an informative error sits far below: the served rows
  against the reference rows half the item away (`control_within`), and against the next image's rows
  (`control_across`). An error near its control is a layout or binding fault, not a quantisation error;
- per family: the pooled relative L2 over all images, and the median of the images';
- the bar: on the images both families encoded, the mean over images of Flash-Next's `rel_l2` less the 27B's,
  with a paired bootstrap interval (images resampled); Flash-Next's tower is "at most the 27B's" when that mean
  is not above zero. An excess is the owner's to weigh, with the interval saying whether it is beyond noise.

Exit status: 0 done, 2 refused (a missing or mismatched input, named).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert  # noqa: E402  (its hub, its cache and, through it, the 27B recipe's modules)
import torch  # noqa: E402

DUMP_SCHEMA = "ignis-vision-tower-dump/1"
REFERENCE_SCHEMA = "ignis-vision-tower-reference/1"
CHECK_SCHEMA = "ignis-vision-tower-check/1"
FLASH_NEXT, QWEN27B = "qwen3.8-flash-next", "qwen3.8-27b"
FAMILIES = (FLASH_NEXT, QWEN27B)
# The 27B's served tower is the converter's from this revision, byte for byte (finding 2026-11-03, the NVFP4
# converter reproduces the served 27B): its BF16 tower is the first shard's visual range, pinned by its bytes.
REPO_27B = "Qwen/Qwen3.8-27B"
REVISION_27B = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOWER_27B = convert.Tower("qwen3.8-27b", 1152, 4304, 16, 48 * 48, 5120, shard="model-00001-of-00018.safetensors",
                          span_bytes=921_460_192,
                          span_sha256="c623c5ec68429f48689a011872157782c85508517faaa66de38250e0ad4b145b")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 560


class Refused(Exception):
    """An input the check cannot use; nothing has been written."""


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------------------------- the dump
def bf16_to_f32(bits):
    """BF16 bits (uint16) as float32 values: the bits are float32's high half."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def read_bits(path, rows, width):
    data = np.fromfile(path, dtype="<u2")
    if data.size != rows * width:
        raise Refused(f"{path} holds {data.size} values, not {rows} x {width}")
    return data.reshape(rows, width)


def read_manifest(directory):
    path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(path):
        raise Refused(f"no dump at {directory}: {path} does not exist")
    manifest = json.load(open(path, encoding="utf-8"))
    if manifest.get("schema") != DUMP_SCHEMA:
        raise Refused(f"{path} is {manifest.get('schema')!r}, not {DUMP_SCHEMA}")
    return manifest


def item_patches(directory, manifest, item):
    t, h, w = item["grid"]
    return read_bits(os.path.join(directory, item["patches"]), t * h * w, manifest["patch_features"])


def item_served(directory, manifest, item):
    return bf16_to_f32(read_bits(os.path.join(directory, item["embedding"]), item["tokens"], manifest["hidden"]))


def item_reference(directory, manifest, item):
    path = os.path.join(directory, f"{item['id']}.reference.f32")
    if not os.path.isfile(path):
        raise Refused(f"{path} does not exist: run `reference --family {manifest['family']}` first")
    data = np.fromfile(path, dtype="<f4")
    if data.size != item["tokens"] * manifest["hidden"]:
        raise Refused(f"{path} holds {data.size} values, not {item['tokens']} x {manifest['hidden']}")
    return data.reshape(item["tokens"], manifest["hidden"])


# ---------------------------------------------------------------------------------------------- the reference
def vision_classes(model_type):
    """(config class, model class) of the checkpoint's vision tower, as transformers names its model type."""
    if model_type == "qwen4_exp":
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
        return Qwen4ExpVisionConfig, Qwen4ExpVisionModel
    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        return Qwen3_5VisionConfig, Qwen3_5VisionModel
    raise Refused(f"no reference for a vision tower of model type {model_type!r}")


def load_state(model, tensors):
    """`tensors` (the checkpoint's `model.visual.*` names) into `model` as float32, every name accounted for."""
    state = {name[len(convert.PREFIX):] if name.startswith(convert.PREFIX) else name: t.float()
             for name, t in tensors.items()}
    model.load_state_dict(state, strict=True)
    return model


def reference_model(source):
    """The checkpoint's vision tower from a `convert.fetch` cache, its weights the cached BF16 tensors held in
    float32, computing in float32."""
    data_path, record_path = os.path.join(source, "visual.bin"), os.path.join(source, "visual.json")
    for path in (data_path, record_path):
        if not os.path.isfile(path):
            raise Refused(f"{path} does not exist: fill the cache first (`convert.py fetch` or `fetch-27b`)")
    record = json.load(open(record_path, encoding="utf-8"))
    vision = dict(record["config"]["vision_config"])
    config_class, model_class = vision_classes(vision.pop("model_type", None))
    config = config_class(**vision)
    if not (config.rope_parameters or {}).get("rope_theta"):
        raise Refused(f"the {config_class.__name__} built from {record_path} has no rope_theta")
    config._attn_implementation = "sdpa"
    model = model_class(config)
    tensors = convert.source_tensors(open(data_path, "rb").read(), record)
    load_state(model, tensors)
    model.float().eval()
    model.source = {k: record[k] for k in ("repo", "revision", "file", "offset", "bytes", "sha256")}
    return model


def reference_rows(model, patches, grid):
    """The merger's output, float32 `[tokens][out_hidden]`, of `model` on one item's BF16 patch rows."""
    device = next(model.parameters()).device
    pixels = torch.from_numpy(bf16_to_f32(patches).copy()).to(device)
    with torch.no_grad():
        out = model(pixels, grid_thw=torch.tensor([list(grid)], dtype=torch.long, device=device))
    return out.pooler_output.float().cpu().numpy()


# ---------------------------------------------------------------------------------------------- the measure
def _rel(diff, ref):
    norm = np.linalg.norm(ref, axis=-1)
    return np.linalg.norm(diff, axis=-1) / np.where(norm > 0, norm, 1)


def _cos(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.sum(a * b, axis=-1) / np.where(na * nb > 0, na * nb, 1)


def _pair(served, reference, centre):
    """The relative L2 and the mean centred cosine of served rows against reference rows, row for row."""
    diff = served.astype(np.float64) - reference
    rel = float(np.linalg.norm(diff) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30))
    return {"rel_l2": rel, "centred_cosine_mean": float(np.mean(_cos(served - centre, reference - centre)))}


def measures(served, reference, centre, other=None):
    """One image's error (the module docstring's list); `other` is the next image's reference rows, for the
    across-image control."""
    served, reference = np.asarray(served, np.float32), np.asarray(reference, np.float32)
    if served.shape != reference.shape:
        raise Refused(f"served rows {served.shape} against reference rows {reference.shape}")
    diff = served.astype(np.float64) - reference
    rows = _rel(diff, reference.astype(np.float64))
    centred = _cos(served - centre, reference - centre)
    out = {"tokens": int(served.shape[0]),
           "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30)),
           "sq_err": float(np.sum(diff * diff)), "sq_ref": float(np.sum(reference.astype(np.float64) ** 2)),
           "row_rel_l2": {"median": float(np.median(rows)), "p90": float(np.percentile(rows, 90)),
                          "max": float(rows.max())},
           "centred_cosine": {"mean": float(centred.mean()), "min": float(centred.min())},
           "raw_cosine_mean": float(np.mean(_cos(served, reference)))}
    n = served.shape[0]
    out["control_within"] = _pair(served, np.roll(reference, n // 2, axis=0), centre) if n > 1 else None
    if other is not None:
        k = min(n, other.shape[0])
        out["control_across"] = _pair(served[:k], np.asarray(other[:k], np.float32), centre)
    return out


def bar(flash_next, qwen27b, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    """Flash-Next's per-image errors against the 27B's on the images both have (`{id: measures}` each): the mean
    paired difference of `rel_l2`, its bootstrap interval, and the pooled errors and their ratio."""
    ids = sorted(set(flash_next) & set(qwen27b))
    if not ids:
        raise Refused("no image both families encoded")
    a = np.array([flash_next[i]["rel_l2"] for i in ids])
    b = np.array([qwen27b[i]["rel_l2"] for i in ids])
    d = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
    means = d[draws].mean(axis=1)

    def pooled(m):
        return float(np.sqrt(sum(m[i]["sq_err"] for i in ids) / sum(m[i]["sq_ref"] for i in ids)))
    return {"images": len(ids), "mean_difference": float(d.mean()), "median_difference": float(np.median(d)),
            "flash_next_lower_on": int(np.sum(d <= 0)),
            "bootstrap": {"low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5)),
                          "resamples": resamples, "seed": seed},
            "flash_next_pooled_rel_l2": pooled(flash_next), "qwen27b_pooled_rel_l2": pooled(qwen27b),
            "pooled_ratio": pooled(flash_next) / pooled(qwen27b),
            "flash_next_at_most_27b": bool(d.mean() <= 0)}


# ---------------------------------------------------------------------------------------------- commands
def run_fetch_27b(args):
    _, record = convert.fetch(convert.HttpHub(REPO_27B, REVISION_27B), args.source, TOWER_27B, REPO_27B,
                              REVISION_27B)
    log(f"{args.source}: {record['bytes']} bytes of {REPO_27B}@{REVISION_27B} {record['file']}, "
        f"sha256 {record['sha256']}")
    return 0


def run_reference(args):
    directory = os.path.join(args.dump, args.family)
    manifest = read_manifest(directory)
    if manifest["family"] != args.family:
        raise Refused(f"{directory} holds a {manifest['family']} dump, not {args.family}")
    torch.set_num_threads(args.threads)
    model = reference_model(args.source).to(args.device)
    out_hidden = model.config.out_hidden_size
    if out_hidden != manifest["hidden"]:
        raise Refused(f"the tower in {args.source} writes {out_hidden} columns, the dump's {manifest['hidden']}")
    record = {"schema": REFERENCE_SCHEMA, "family": args.family, "source": model.source, "dtype": "float32",
              "device": args.device, "threads": args.threads, "torch": torch.__version__,
              "transformers": __import__("transformers").__version__, "items": {}}
    for item in manifest["items"]:
        started = time.perf_counter()
        rows = reference_rows(model, item_patches(directory, manifest, item), item["grid"])
        if rows.shape != (item["tokens"], manifest["hidden"]):
            raise Refused(f"{item['id']}: the reference gave {rows.shape}, the dump holds "
                          f"{(item['tokens'], manifest['hidden'])}")
        path = os.path.join(directory, f"{item['id']}.reference.f32")
        rows.astype("<f4").tofile(path)
        seconds = time.perf_counter() - started
        record["items"][item["id"]] = {"file": os.path.basename(path), "seconds": round(seconds, 3)}
        log(f"{args.family} {item['id']}: {item['tokens']} tokens, grid {item['grid']}, {seconds:.1f} s")
    with open(os.path.join(directory, "reference.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return 0


def family_check(directory):
    """(manifest, [per-image measures]) of one family's dump and reference."""
    manifest = read_manifest(directory)
    items = manifest["items"]
    total, count = None, 0
    for item in items:
        rows = item_reference(directory, manifest, item).astype(np.float64)
        total = rows.sum(0) if total is None else total + rows.sum(0)
        count += rows.shape[0]
    centre = (total / count).astype(np.float32)
    out = []
    for k, item in enumerate(items):
        other = item_reference(directory, manifest, items[(k + 1) % len(items)]) if len(items) > 1 else None
        m = measures(item_served(directory, manifest, item), item_reference(directory, manifest, item), centre, other)
        out.append({"id": item["id"], "image": item["image"], "grid": item["grid"], **m})
    return manifest, out


def run_compare(args):
    record = {"schema": CHECK_SCHEMA, "families": {}}
    by_family = {}
    for family in FAMILIES:
        directory = os.path.join(args.dump, family)
        if not os.path.isfile(os.path.join(directory, "manifest.json")):
            log(f"{family}: no dump in {directory}")
            continue
        manifest, items = family_check(directory)
        sq_err, sq_ref = sum(i["sq_err"] for i in items), sum(i["sq_ref"] for i in items)
        record["families"][family] = {
            "artifact": manifest["artifact"], "residency": manifest["residency"], "hidden": manifest["hidden"],
            "images": len(items), "pooled_rel_l2": float(np.sqrt(sq_err / sq_ref)),
            "median_rel_l2": float(np.median([i["rel_l2"] for i in items])), "items": items}
        by_family[family] = {i["id"]: i for i in items}
        log(f"{family}: {len(items)} images, pooled rel L2 {record['families'][family]['pooled_rel_l2']:.4g}, "
            f"median {record['families'][family]['median_rel_l2']:.4g}")
    if len(by_family) == 2:
        record["bar"] = bar(by_family[FLASH_NEXT], by_family[QWEN27B])
        b = record["bar"]
        log(f"bar: Flash-Next - 27B rel L2 mean {b['mean_difference']:+.4g} "
            f"[{b['bootstrap']['low']:+.4g}, {b['bootstrap']['high']:+.4g}] over {b['images']} images: "
            f"{'at most the 27B' if b['flash_next_at_most_27b'] else 'ABOVE the 27B: to the owner'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["family", "id", "tokens", "rel_l2", "row_rel_l2_median", "row_rel_l2_p90", "row_rel_l2_max",
                        "centred_cosine_mean", "centred_cosine_min", "raw_cosine_mean", "control_within_rel_l2",
                        "control_across_rel_l2"])
            for family, f_record in record["families"].items():
                for i in f_record["items"]:
                    w.writerow([family, i["id"], i["tokens"], i["rel_l2"], i["row_rel_l2"]["median"],
                                i["row_rel_l2"]["p90"], i["row_rel_l2"]["max"], i["centred_cosine"]["mean"],
                                i["centred_cosine"]["min"], i["raw_cosine_mean"],
                                (i["control_within"] or {}).get("rel_l2"),
                                (i.get("control_across") or {}).get("rel_l2")])
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch-27b", help="the 27B's BF16 tower into a cache, by range")
    f.add_argument("--source", required=True)
    f.set_defaults(fn=run_fetch_27b)
    r = sub.add_parser("reference", help="the BF16 tower's rows on one family's dumped patches")
    r.add_argument("--family", required=True, choices=FAMILIES)
    r.add_argument("--source", required=True, help="the family's cache of its BF16 tower")
    r.add_argument("--dump", required=True)
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--device", default="cpu")
    r.set_defaults(fn=run_reference)
    c = sub.add_parser("compare", help="the record: per image, per family, and the bar")
    c.add_argument("--dump", required=True)
    c.add_argument("--out")
    c.add_argument("--csv")
    c.set_defaults(fn=run_compare)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (Refused, convert.Refused) as e:
        print(f"refused: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
