"""Flash-Next's encoded vision tower against its BF16 tower, with the 27B's as the bar (GitHub #560, spec #555's
seam 2, ADR 0087).

    python tools/flash-next-vision/tower_check.py fetch-27b --source <dir>
    python tools/flash-next-vision/tower_check.py reference --family <family> --source <dir> --dump <dir>
                                                  [--threads 6] [--device cpu]
    python tools/flash-next-vision/tower_check.py compare --dump <dir> [--out <json>] [--csv <csv>]
    python tools/flash-next-vision/tower_check.py diagnose --family <family> --source <dir> --container <.ninfer>
                                                  --dump <dir> --out <json> [--variant <v> ...] [--items <id,...>]

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

`diagnose` says where an encoded tower's error comes from (GitHub #560: Flash-Next's was above the 27B's). It
runs the reference tower of `reference` with its matrices taken, kind by kind, from one of three sources: the
BF16 checkpoint (`bf16`), the container's stored words decoded as the kernels decode them (`stored`: a code times
its group's binary16 scale), or the BF16 tensor re-encoded by the recipe's own encoder in another format the
kernels serve (`Q4`, `Q5`, `Q6`, `W8`, `BF16`). The kinds are the seven matrices: `patch_embedding`, the blocks'
`qkv`, `attention_output`, `fc1` and `fc2`, `merger_fc1` and `merger_fc2`; every other object (biases, norms,
the position table) is BF16 and exact. A variant names a base source and its exceptions, later ones winning,
each over a kind (`block` for the four block matrices) and optionally a range of blocks: `stored+qkv=W8`,
`bf16+block@26=stored` (`Variant`). The default set is `bf16`, `stored`, and `bf16+<kind>=stored` for each
kind. Per variant, on the dump's items:
- each stage's error against the BF16 tower's (`embedding`, the patches with their positions; `block <b>`;
  `merger norm`, `merger fc1` before its GELU, `merger fc2`, the rows): the relative L2, the same error over
  the BF16 rows centred on their set mean (`centred_rel_l2`; a column every row shares is reproduced for free
  and inflates the plain one's denominator), and the share of the error's energy in one vector every row
  shares (`constant_error_share`);
- the served rows against the variant's (`vs_served_rel_l2`): with `stored`, near the kernels' rounding when
  the encode computes what the stored words say;
- its cost: the tower's bytes and the staging slot (the largest stage, `ignis_vision_staging_slot_bytes`);
- with `--against` another diagnosis (the 27B's), `compare`'s bar against that one's stored tower, item for
  item, plain and centred.
The record also holds the recipe check (each stored payload against the recipe's encoding of its BF16 tensor,
byte for byte, the differing objects named) and each source's weight error per kind (relative Frobenius).

Exit status: 0 done, 2 refused (a missing or mismatched input, named).
"""
import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import convert  # noqa: E402  (its hub, its cache and, through it, the 27B recipe's modules)
import container  # noqa: E402  (the 27B NVFP4 converter's, on the path through `convert`)
import encode  # noqa: E402
import inventory  # noqa: E402
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
DIAGNOSIS_SCHEMA = "ignis-vision-tower-diagnosis/1"
TOWERS = {FLASH_NEXT: convert.GEOMETRIES[FLASH_NEXT], QWEN27B: TOWER_27B, "fixture": convert.GEOMETRIES["fixture"]}
KINDS = ("patch_embedding", "qkv", "attention_output", "fc1", "fc2", "merger_fc1", "merger_fc2")
FORMATS = {"Q4": inventory.Q4, "Q5": inventory.Q5, "Q6": inventory.Q6, "W8": inventory.W8, "BF16": inventory.BF16}
DEFAULT_VARIANTS = ("bf16", "stored") + tuple(f"bf16+{kind}=stored" for kind in KINDS)
# kernel/src/vision_encode.cu: a staged tensor's span slack and the slot's alignment.
STAGING_SPAN_SLACK = 2 * 4096 + 256
STAGING_ALIGNMENT = 256


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


# ---------------------------------------------------------------------------------------------- the diagnosis
def dequantize(payload, fmt, shape):
    """The values a stored `[N, K]` payload holds, float32, as the kernels decode it: BF16 words as they are; a
    row-split W8/Q4/Q5/Q6 matrix's code (two's complement, its low four bits in the code plane, a Q5's fifth and
    a Q6's fifth and sixth in the high plane, element-major) times its group's binary16 scale, the K padding
    cut away (`kernel/vendor/tests/ops/quantized_weight.h`, `unpack_lowbit_code`)."""
    if fmt == inventory.BF16:
        return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).float().reshape(shape)
    if fmt not in inventory.GROUP:
        raise Refused(f"no decoder for {fmt}")
    n, k = shape
    group, width = inventory.GROUP[fmt]
    k_pad, groups, base_row, high_row, scale_row, high_offset, scale_offset, size = \
        inventory.row_split_geometry(fmt, (n, k))
    if len(payload) != size:
        raise Refused(f"a {fmt} {list(shape)} payload is {size} bytes, not {len(payload)}")
    words = np.frombuffer(payload, dtype=np.uint8)
    base = words[: n * base_row].reshape(n * groups, -1)
    if width == 8:
        codes = base.view(np.int8).astype(np.int32)
    else:
        codes = np.stack((base & 0x0F, base >> 4), axis=-1).reshape(n * groups, group).astype(np.int32)
        if width > 4:
            high_bits = width - 4
            high = words[high_offset: high_offset + n * high_row].reshape(n * groups, -1)
            planes = np.unpackbits(high, axis=1, bitorder="little").reshape(n * groups, group, high_bits)
            codes |= sum(planes[..., b].astype(np.int32) << (4 + b) for b in range(high_bits))
        codes = np.where(codes >= 1 << (width - 1), codes - (1 << width), codes)
    scales = words[scale_offset: scale_offset + n * scale_row].view("<f2").astype(np.float32)
    values = codes.reshape(n, groups, group).astype(np.float32) * scales.reshape(n, groups, 1)
    return torch.from_numpy(np.ascontiguousarray(values.reshape(n, k_pad)[:, :k]))


def object_kind(name):
    """The matrix kind of a tower object, None for the BF16 vectors and the position table."""
    if name.startswith("vision/layers/"):
        return {"attention/qkv": "qkv", "attention/output": "attention_output", "mlp/fc1": "fc1",
                "mlp/fc2": "fc2"}.get(name.split("/", 3)[3])
    return {"vision/patch_embedding": "patch_embedding", "vision/merger/fc1": "merger_fc1",
            "vision/merger/fc2": "merger_fc2"}.get(name)


def staging_stage(name):
    """The staging stage of a tower object (`visit_vision_stages`): the pre-block weights, one per block, the
    merger's norm and fc1, then its fc2."""
    if name.startswith("vision/layers/"):
        return 1 + int(name.split("/")[2])
    if name in ("vision/merger/fc2", "vision/merger/fc2_bias"):
        return 2 + inventory.VISION_LAYERS
    return 1 + inventory.VISION_LAYERS if name.startswith("vision/merger/") else 0


BLOCK_KINDS = ("qkv", "attention_output", "fc1", "fc2")


class Variant:
    """Where each matrix comes from: `<base>[+<rule>]...`, the base `bf16` or `stored`, a rule
    `<kind>[@<b>[-<b>]]=<source>`: a kind of KINDS or `block` (the four block matrices), the blocks it covers
    (all if none), and a source, `bf16`, `stored` or a format of FORMATS. A later rule wins over an earlier."""

    def __init__(self, text):
        self.text = text
        base, *rules = text.split("+")
        if base not in ("bf16", "stored"):
            raise Refused(f"variant {text!r}: its base is {base!r}, not bf16 or stored")
        self.base, self.rules = base, []
        for rule in rules:
            target, _, source = rule.partition("=")
            kind, _, span = target.partition("@")
            kinds = BLOCK_KINDS if kind == "block" else (kind,)
            blocks = None
            try:
                if span:
                    first, _, last = span.partition("-")
                    blocks = (int(first), int(last or first))
            except ValueError:
                blocks = (1, 0)
            if (not set(kinds) <= set(KINDS) or not (source in ("bf16", "stored") or source in FORMATS)
                    or blocks is not None and (not set(kinds) <= set(BLOCK_KINDS)
                                               or not 0 <= blocks[0] <= blocks[1] < inventory.VISION_LAYERS)):
                raise Refused(f"variant {text!r}: {rule!r} is not <kind>[@<b>[-<b>]]=<bf16|stored|"
                              f"{'|'.join(FORMATS)}>, a kind of {', '.join(KINDS)} or block, blocks of a "
                              f"block kind 0 to {inventory.VISION_LAYERS - 1}")
            self.rules.append((kinds, blocks, "bf16" if source == "BF16" else source))

    def source(self, name):
        """The source of object `name`: BF16 for the vectors and the position table, which no source changes."""
        kind = object_kind(name)
        if kind is None:
            return "bf16"
        block = int(name.split("/")[2]) if name.startswith("vision/layers/") else None
        out = self.base
        for kinds, blocks, source in self.rules:
            if kind in kinds and (blocks is None or blocks[0] <= block <= blocks[1]):
                out = source
        return out


class TowerWeights:
    """A tower's objects from each source: the BF16 checkpoint's tensors, the container's stored words decoded,
    or the BF16 tensors re-encoded by the recipe's encoder (decoded ones cached)."""

    def __init__(self, tower, tensors, stored):
        self.objects = convert.tower_objects(tower)
        self.routes = convert.tower_routes(tower)
        self.tensors, self.stored, self.cache = tensors, stored, {}
        for o in self.objects:
            entry = stored.by_name.get(o.name)
            if entry is None or tuple(entry["shape"]) != o.shape:
                raise Refused(f"{stored.path}: {o.name} is {'absent' if entry is None else entry['shape']}, "
                              f"the tower's is {list(o.shape)}")

    def param(self, o):
        return self.routes[o.name][0][len(convert.PREFIX):]

    def bf16(self, o):
        name, _, transform = self.routes[o.name]
        return self.tensors[name].reshape(o.shape) if transform == "flatten" else self.tensors[name]

    def format(self, o, source):
        return {"bf16": inventory.BF16, "stored": self.stored.by_name[o.name]["format"]}.get(source) or FORMATS[source]

    def bytes(self, o, source):
        if source == "stored":
            return self.stored.by_name[o.name]["bytes"]
        return inventory.encoded_size(inventory.Obj(o.name, self.format(o, source), o.shape))

    def value(self, o, source):
        """The object's float32 values from `source`."""
        if source == "bf16":
            return self.bf16(o).float()
        key = (o.name, source)
        if key not in self.cache:
            if source == "stored":
                payload = self.stored.read(o.name)
            else:
                payload = encode.encode(inventory.Obj(o.name, FORMATS[source], o.shape), self.bf16(o), "cpu")[0]
            self.cache[key] = dequantize(payload, self.format(o, source), o.shape)
        return self.cache[key]

    def recipe_check(self):
        """Each stored payload against the recipe's encoding of its BF16 tensor, byte for byte."""
        differ = [o.name for o in self.objects
                  if self.stored.read(o.name) != encode.encode(o, self.bf16(o), "cpu")[0]]
        return {"objects": len(self.objects), "equal": len(self.objects) - len(differ), "differ": differ}

    def weight_error(self, source, kinds=KINDS):
        """Per kind: the format, the objects, the pooled relative Frobenius error of `source` against BF16."""
        out = {}
        for kind in kinds:
            objs = [o for o in self.objects if object_kind(o.name) == kind]
            err = ref = worst = 0.0
            for o in objs:
                w = self.bf16(o).double()
                e = float(((self.value(o, source).double() - w) ** 2).sum())
                r = float((w ** 2).sum())
                err, ref, worst = err + e, ref + r, max(worst, (e / r) ** 0.5)
            out[kind] = {"format": self.format(objs[0], source), "objects": len(objs),
                         "rel_frobenius": (err / ref) ** 0.5, "max_object_rel_frobenius": worst}
        return out

    def cost(self, variant):
        """The tower's bytes and its staging slot under `variant`."""
        stages = [0] * (3 + inventory.VISION_LAYERS)
        total = 0
        for o in self.objects:
            if object_kind(o.name):
                size = self.bytes(o, variant.source(o.name))
            else:
                size = self.stored.by_name[o.name]["bytes"]
            total += size
            stages[staging_stage(o.name)] += size + STAGING_SPAN_SLACK
        largest = max(range(len(stages)), key=stages.__getitem__)
        slot = (stages[largest] + STAGING_ALIGNMENT - 1) // STAGING_ALIGNMENT * STAGING_ALIGNMENT
        return {"tower_bytes": total, "slot_bytes": slot, "largest_stage": largest}

    def formats(self, variant):
        """{kind: the formats `variant` gives its objects}."""
        out = {}
        for o in self.objects:
            if object_kind(o.name):
                out.setdefault(object_kind(o.name), set()).add(self.format(o, variant.source(o.name)))
        return {kind: sorted(f) for kind, f in out.items()}

    def assign(self, model, variant, current):
        """`model`'s matrices from `variant`; `current` ({object: source}) names what each holds now."""
        with torch.no_grad():
            for o in self.objects:
                source = variant.source(o.name)
                if current.get(o.name, "bf16") != source:
                    p = model.get_parameter(self.param(o))
                    p.copy_(self.value(o, source).reshape(p.shape))
                    current[o.name] = source


def stage_names():
    return (["embedding"] + [f"block {b}" for b in range(inventory.VISION_LAYERS)]
            + ["merger norm", "merger fc1", "merger fc2"])


def stage_rows(model, patches, grid, sink=None):
    """(every stage's output in `stage_names`' order, the rows), float32, of one forward of `model` on one item;
    with `sink`, each stage's output goes to `sink(stage, output)` as it is made and the list is empty, so an
    item's stages are never all held twice."""
    out, count = [], [0]

    def put(output):
        if sink is None:
            out.append(output.detach())
        else:
            sink(count[0], output.detach())
        count[0] += 1
    hooks = [model.blocks[0].register_forward_pre_hook(
        lambda _m, args, kwargs: put(args[0] if args else kwargs["hidden_states"]), with_kwargs=True)]
    hooks += [m.register_forward_hook(lambda _m, _i, output: put(output))
              for m in (*model.blocks, model.merger.norm, model.merger.linear_fc1)]
    try:
        rows = torch.from_numpy(reference_rows(model, patches, grid))
    finally:
        for h in hooks:
            h.remove()
    put(rows)
    return out, rows


class Tally:
    """Sums of one variant's error against the BF16 rows, per stage and per item, from which the record's
    measures are read once every item has run."""

    def __init__(self, stages):
        self.stages = [{"sq_err": 0.0, "sq_ref": 0.0, "sum_err": 0.0, "sum_ref": 0.0, "rows": 0} for _ in stages]
        self.items = {}

    def add_stage(self, stage, base, rows):
        """One item's output of one stage, against the BF16 tower's."""
        s, b = self.stages[stage], base.double()
        e = rows.double() - b
        s["sq_err"] += float((e * e).sum())
        s["sq_ref"] += float((b * b).sum())
        s["sum_err"] = s["sum_err"] + e.sum(0)
        s["sum_ref"] = s["sum_ref"] + b.sum(0)
        s["rows"] += b.shape[0]

    def add_item(self, item, base, rows, served):
        """One item's rows against the BF16 tower's and the served ones."""
        b, r = base.double(), rows.double()
        self.items[item] = {"sq_err": float(((r - b) ** 2).sum()), "sq_ref": float((b * b).sum()),
                            "sq_served": float(((served - r) ** 2).sum()), "sum_ref": b.sum(0), "rows": b.shape[0]}

    @staticmethod
    def _ratio(a, b):
        return float(np.sqrt(a / b)) if b > 0 else 0.0

    def record(self, names):
        stages = []
        for name, s in zip(names, self.stages):
            centre = s["sum_ref"] / s["rows"]
            centred = s["sq_ref"] - s["rows"] * float((centre * centre).sum())
            constant = float((s["sum_err"] * s["sum_err"]).sum()) / s["rows"]
            stages.append({"stage": name, "rel_l2": self._ratio(s["sq_err"], s["sq_ref"]),
                           "centred_rel_l2": self._ratio(s["sq_err"], centred),
                           "constant_error_share": constant / s["sq_err"] if s["sq_err"] > 0 else 0.0})
        last = self.stages[-1]
        centre = last["sum_ref"] / last["rows"]
        items = {}
        for item, m in self.items.items():
            centred = (m["sq_ref"] - 2 * float((centre * m["sum_ref"]).sum())
                       + m["rows"] * float((centre * centre).sum()))
            items[item] = {"rel_l2": self._ratio(m["sq_err"], m["sq_ref"]),
                           "centred_rel_l2": self._ratio(m["sq_err"], centred),
                           "vs_served_rel_l2": self._ratio(m["sq_served"], m["sq_ref"]),
                           "sq_err": m["sq_err"], "sq_ref": m["sq_ref"], "sq_ref_centred": centred}
        final = dict(stages[-1])
        del final["stage"]
        final["vs_served_rel_l2"] = self._ratio(sum(m["sq_served"] for m in self.items.values()),
                                                sum(m["sq_ref"] for m in self.items.values()))
        return {"final": final, "items": items, "stages": stages}


def against_bar(items, other):
    """`bar` of a variant's per-item errors against another diagnosis's stored tower's, plain and centred."""
    def centred(m):
        return {i: {"rel_l2": v["centred_rel_l2"], "sq_err": v["sq_err"], "sq_ref": v["sq_ref_centred"]}
                for i, v in m.items()}
    plain = bar(items, other)
    return {"images": plain["images"], "rel_l2": plain, "centred_rel_l2": bar(centred(items), centred(other))}


def run_diagnose(args):
    directory = os.path.join(args.dump, args.family)
    manifest = read_manifest(directory)
    if manifest["family"] != args.family:
        raise Refused(f"{directory} holds a {manifest['family']} dump, not {args.family}")
    items = manifest["items"]
    if args.items:
        wanted = args.items.split(",")
        unknown = sorted(set(wanted) - {i["id"] for i in items})
        if unknown:
            raise Refused(f"{directory} has no item {', '.join(unknown)}")
        items = [i for i in items if i["id"] in wanted]
    variants = {v: Variant(v) for v in (args.variant or DEFAULT_VARIANTS)}
    against = None
    if args.against:
        other = json.load(open(args.against, encoding="utf-8"))
        if other.get("schema") != DIAGNOSIS_SCHEMA or "stored" not in other.get("variants", {}):
            raise Refused(f"{args.against} is not a diagnosis with a stored variant")
        against = other["variants"]["stored"]["items"]
    torch.set_num_threads(args.threads)
    model = reference_model(args.source)
    if model.config.out_hidden_size != manifest["hidden"]:
        raise Refused(f"the tower in {args.source} writes {model.config.out_hidden_size} columns, "
                      f"the dump's {manifest['hidden']}")
    gc.collect()  # the model's build leaves its float32 copies of the tensors to the collector
    tensors = convert.source_tensors(open(os.path.join(args.source, "visual.bin"), "rb").read(),
                                     json.load(open(os.path.join(args.source, "visual.json"), encoding="utf-8")))
    weights = TowerWeights(TOWERS[args.geometry or args.family], tensors, container.Container(args.container))
    started = time.perf_counter()
    record = {"schema": DIAGNOSIS_SCHEMA, "family": args.family, "container": os.path.basename(args.container),
              "identity": weights.stored.identity, "source": model.source, "items": [i["id"] for i in items],
              "threads": args.threads, "torch": torch.__version__,
              "transformers": __import__("transformers").__version__}
    record["recipe"] = weights.recipe_check()
    log(f"recipe check: {record['recipe']['equal']} of {record['recipe']['objects']} objects equal"
        + (f"; differ: {', '.join(record['recipe']['differ'])}" if record["recipe"]["differ"] else ""))
    used = {}
    for v in variants.values():
        for o in weights.objects:
            if v.source(o.name) != "bf16":
                used.setdefault(v.source(o.name), set()).add(object_kind(o.name))
    used["stored"] = set(KINDS)
    record["weights"] = {source: weights.weight_error(source, [k for k in KINDS if k in kinds])
                         for source, kinds in sorted(used.items())}
    needed = {(o.name, v.source(o.name)) for v in variants.values() for o in weights.objects}
    weights.cache = {key: values for key, values in weights.cache.items() if key in needed}
    names = stage_names()
    tallies = {v: Tally(names) for v in variants}
    harness = [0.0, 0.0]
    current = {}
    for item in items:
        t0 = time.perf_counter()
        patches, grid = item_patches(directory, manifest, item), item["grid"]
        served = torch.from_numpy(item_served(directory, manifest, item)).double()
        weights.assign(model, Variant("bf16"), current)
        base, base_rows = stage_rows(model, patches, grid)
        if os.path.isfile(os.path.join(directory, f"{item['id']}.reference.f32")):
            ref = torch.from_numpy(item_reference(directory, manifest, item)).double()
            harness[0] += float(((base_rows.double() - ref) ** 2).sum())
            harness[1] += float((ref * ref).sum())
        for name, variant in variants.items():
            tally = tallies[name]
            if all(variant.source(o.name) == "bf16" for o in weights.objects):
                for stage, rows in enumerate(base):
                    tally.add_stage(stage, rows, rows)
                rows = base_rows
            else:
                weights.assign(model, variant, current)
                _, rows = stage_rows(model, patches, grid,
                                     lambda stage, out, tally=tally: tally.add_stage(stage, base[stage], out))
            tally.add_item(item["id"], base_rows, rows, served)
        del base
        log(f"{args.family} {item['id']}: {item['tokens']} tokens, {len(variants)} variants, "
            f"{time.perf_counter() - t0:.1f} s")
    if harness[1] > 0:
        record["harness"] = {"bf16_vs_reference_rel_l2": float(np.sqrt(harness[0] / harness[1]))}
    if against is not None:
        record["against"] = {"file": os.path.basename(args.against), "family": other["family"], "variant": "stored"}
    record["variants"] = {}
    for name, variant in variants.items():
        record["variants"][name] = {"formats": weights.formats(variant), **tallies[name].record(names),
                                    "cost": weights.cost(variant)}
        if against is not None:
            record["variants"][name]["bar"] = against_bar(record["variants"][name]["items"], against)
        f = record["variants"][name]["final"]
        log(f"{name}: rel L2 {f['rel_l2']:.4g}, centred {f['centred_rel_l2']:.4g}, constant share "
            f"{f['constant_error_share']:.3g}, served {f['vs_served_rel_l2']:.4g}")
    record["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
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
    d = sub.add_parser("diagnose", help="where the encoded tower's error comes from: variants of the matrices' "
                                        "sources, stage by stage")
    d.add_argument("--family", required=True, choices=FAMILIES)
    d.add_argument("--geometry", choices=sorted(TOWERS), help="the tower's widths (default: the family's)")
    d.add_argument("--source", required=True, help="the family's cache of its BF16 tower")
    d.add_argument("--container", required=True, help="the container holding the served tower's vision/ objects")
    d.add_argument("--dump", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--variant", action="append",
                   help=f"a variant (repeatable; default: {', '.join(DEFAULT_VARIANTS)})")
    d.add_argument("--items", help="the dump's items to run, by id, comma-separated (default: all)")
    d.add_argument("--against", help="another diagnosis (the 27B's): each variant's paired bar against its "
                                     "stored tower, on the items both ran")
    d.add_argument("--threads", type=int, default=6)
    d.set_defaults(fn=run_diagnose)
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
