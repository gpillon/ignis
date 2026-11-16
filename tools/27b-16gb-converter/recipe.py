"""A named recipe of the 27B's trellis artifacts (GitHub #631, spec #630): what the 16 GB converter
(`convert.py run`) and the local quant (`local_quant.py run`) take by name, `--recipe`, the 16 GB
artifact's, `sc30`, by default.

A recipe is named data, `recipes/<name>.json` beside the tool: the K of every body matrix (`tensors`), the
fields the quant's `quantization_config.json` states (`quantization_config`, the head's bits among them),
the ceiling in bytes of the text scope's device weights (`ceiling_bytes`), the pins of the files it reads
(`files`: bytes and sha256 by file name), the artifact's weights id (`weights_id`) and, for a recipe this
machine converts, exllamav3's job beside the K (`job`: the MTP layer's bits, the calibration size).
`load` refuses one the load cannot bind: a K outside 2 to 5, a fractional K, a head not at 4 bits.

A local conversion (GitHub #471) gives exllamav3 1.5.3 (`convert_model.py --recipe`) the recipe's K for
every body matrix: exllamav3 takes the head from `head_bits` and ignores a recipe entry for it, so its
recipe lists only the body.
"""
import hashlib
import json
import os
import re

import yaml

import exl3

SCHEMA = "qwen3.8-27b-trellis-recipe-v1"
# Where `load` finds a recipe by its name, and the one the tools take when none is named: the 16 GB artifact's.
RECIPES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
DEFAULT = "sc30"
REQUIRED = ("schema", "weights_id", "ceiling_bytes", "quantization_config", "tensors", "files")
OPTIONAL = ("job", "about")
# The K the load binds, integer only (spec #630: no fractional K, no K 1), and the head's bits.
K_RANGE = (2, 5)
HEAD_BITS = 4
SHA256 = re.compile(r"[0-9a-f]{64}")
BODY = "model.language_model.layers."
HEAD = "lm_head"


class Refused(Exception):
    """A conversion refused before it writes anything; the tools' commands exit 2 on it."""


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _problems(data):
    """What keeps the recipe `data` from being one the load binds, one line each."""
    problems = [f"no `{k}`" for k in REQUIRED if k not in data]
    problems += [f"`{k}`: not a field of a recipe" for k in sorted(set(data) - set(REQUIRED) - set(OPTIONAL))]
    if problems:
        return problems
    if data["schema"] != SCHEMA:
        problems.append(f"schema {data['schema']!r}, not {SCHEMA!r}")
    if not isinstance(data["weights_id"], str) or not data["weights_id"]:
        problems.append("`weights_id`: not a name")
    if not isinstance(data["ceiling_bytes"], int) or isinstance(data["ceiling_bytes"], bool) or data["ceiling_bytes"] <= 0:
        problems.append(f"`ceiling_bytes` {data['ceiling_bytes']!r}: not a number of bytes")
    config = data["quantization_config"]
    head = config.get("head_bits") if isinstance(config, dict) else None
    if head != HEAD_BITS:
        problems.append(f"the head at {head} bits (quantization_config.head_bits): the load binds the output "
                        f"head at {HEAD_BITS}")
    tensors = data["tensors"]
    if not isinstance(tensors, dict) or not tensors:
        problems.append("`tensors`: no matrix")
        tensors = {}
    for key, k in sorted(tensors.items()):
        if not key.startswith(BODY):
            problems.append(f"{key}: not a matrix of the body (the head's bits are quantization_config.head_bits)")
        elif not _number(k):
            problems.append(f"{key}: K {k!r}, not a number")
        elif not float(k).is_integer():
            problems.append(f"{key}: K {k}, a fractional K: the load binds integer K {K_RANGE[0]} to {K_RANGE[1]}")
        elif not K_RANGE[0] <= k <= K_RANGE[1]:
            problems.append(f"{key}: K {rate(k)}, outside {K_RANGE[0]} to {K_RANGE[1]}")
    files = data["files"]
    for name, pin in sorted((files if isinstance(files, dict) else {}).items()):
        if not (isinstance(pin, dict) and isinstance(pin.get("bytes"), int)
                and isinstance(pin.get("sha256"), str) and SHA256.fullmatch(pin["sha256"])):
            problems.append(f"{name}: its pin is not bytes and a sha256")
    job = data.get("job")
    if job is not None and not (isinstance(job, dict) and set(job) == {"mtp_bits", "calibration"}):
        problems.append("`job`: not exllamav3's MTP bits and calibration size")
    return problems


def load(name, root=None):
    """The recipe `name` (`<root>/<name>.json`, by default beside the tool), checked: Refused names each
    cause. Its K are ints; `name` and `sha256` (the file's) are added, `job` is None when it has none."""
    root = root or RECIPES
    path = os.path.join(root, f"{name}.json")
    if not name or os.path.basename(name) != name or not os.path.isfile(path):
        there = sorted(n[:-5] for n in os.listdir(root) if n.endswith(".json")) if os.path.isdir(root) else []
        raise Refused(f"no recipe {name!r} in {root} (its recipes: {', '.join(there) or 'none'})")
    with open(path, "rb") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise Refused(f"the recipe {name} is not JSON: {e}")
    problems = _problems(data) if isinstance(data, dict) else ["not a JSON object"]
    if problems:
        raise Refused(f"the recipe {name}:\n  " + "\n  ".join(problems))
    return {"name": name, "sha256": hashlib.sha256(raw).hexdigest(), "weights_id": data["weights_id"],
            "ceiling_bytes": data["ceiling_bytes"], "quantization_config": data["quantization_config"],
            "job": data.get("job"), "files": data["files"],
            "tensors": {key: int(k) for key, k in sorted(data["tensors"].items())}}


def ks(r):
    """Every matrix of the recipe `r` at its K: the body's `tensors`, the head at its bits."""
    return dict(r["tensors"], **{HEAD: r["quantization_config"]["head_bits"]})


def mismatches(r, held):
    """What keeps the matrices `held` ({key: K}, a quant's) from being the recipe `r`'s, one line each."""
    want = ks(r)
    out = []
    for key in sorted(set(held) | set(want)):
        if key not in want:
            out.append(f"{key}: in the quant, not in the recipe")
        elif key not in held:
            out.append(f"{key}: in the recipe, not in the quant")
        elif held[key] != want[key]:
            out.append(f"{key}: K {rate(held[key])}, the recipe's is {want[key]}")
    return out


def bpw(matrices):
    """The body's bits per weight, `{key: {"K", "shape"}}`'s average K weighted by each matrix's weights,
    the head apart as exllamav3's `bits` counts it."""
    body = [m for key, m in matrices.items() if key != HEAD]
    weights = sum(m["shape"][0] * m["shape"][1] for m in body)
    return round(sum(m["K"] * m["shape"][0] * m["shape"][1] for m in body) / weights, 4)


# The text objects the text scope's device weights leave out: the token embedding lives in host memory, the
# draft head is not bound with the text scope.
NOT_DEVICE = ("text/token_embedding", "text/draft_head", "text/draft_head_token_ids")


def text_scope_bytes(matrices, copied):
    """The text scope's device weights as the container stores them: every trellis record of `matrices`
    (`{key: {"K", "shape"}}`, the body and the head) and every object of `copied` ({name: bytes}) under
    `text/` but `NOT_DEVICE`. The load's plan line adds its arena's alignment to it."""
    trellis = sum(exl3.record_bytes(int(m["K"] * 2), *m["shape"]) for m in matrices.values())
    return trellis + sum(n for name, n in copied.items() if name.startswith("text/") and name not in NOT_DEVICE)


def check_ceiling(r, text_bytes):
    if text_bytes > r["ceiling_bytes"]:
        raise Refused(f"the text scope's device weights are {text_bytes:,} bytes, above the recipe {r['name']}'s "
                      f"ceiling of {r['ceiling_bytes']:,}")


def record(r, matrices, text_bytes):
    """What a conversion record says of its recipe: its name and file's sha256, its ceiling, the text
    scope's bytes and the body's bits per weight."""
    return {"name": r["name"], "sha256": r["sha256"], "ceiling_bytes": r["ceiling_bytes"],
            "text_scope_bytes": text_bytes, "bpw": bpw(matrices)}


def rate(k):
    """K as exllamav3 writes a rate: an int when whole (`allocation._as_rate`)."""
    return int(k) if float(k).is_integer() else float(k)


# What a local conversion takes from the recipe's `quantization_config` beside the head's bits.
JOB_CONFIG = ("bits", "codebook", "out_scales")


def job(r, shapes):
    """The local conversion of the recipe `r` (`local_quant.py run`): every matrix, `{key: {"K", "shape"}}`
    in key order, at its K and at its shape in `shapes` ({key: [out, in]}: the 27B's, a conversion
    record's), the head at its bits; and exllamav3's job beside it, the recipe's: bits, codebook, output
    scales, MTP bits, calibration size. Refused when the recipe states no job, or other matrices."""
    if r["job"] is None:
        raise Refused(f"the recipe {r['name']} states no exllamav3 job (`job`: the MTP layer's bits and the "
                      f"calibration size): it is a downloaded quant's, which is not made here")
    config = r["quantization_config"]
    missing = [k for k in JOB_CONFIG if k not in config]
    if missing:
        raise Refused(f"the recipe {r['name']}'s quantization_config states no {', '.join(missing)}: exllamav3 "
                      f"takes them")
    want = ks(r)
    problems = [f"{key}: in the record, not in the recipe" for key in sorted(set(shapes) - set(want))]
    problems += [f"{key}: in the recipe, not in the record" for key in sorted(set(want) - set(shapes))]
    if problems:
        raise Refused(f"the recipe {r['name']} is not the 27B's matrices:\n  " + "\n  ".join(problems))
    return {"matrices": {key: {"K": want[key], "shape": list(shapes[key])} for key in sorted(want)},
            "quantization_config": config, "head_bits": config["head_bits"], "mtp_bits": r["job"]["mtp_bits"],
            "bits": config["bits"], "codebook": config["codebook"], "out_scales": config["out_scales"],
            "calibration": r["job"]["calibration"]}


def exllamav3(r):
    """exllamav3's `--recipe`: `tensors`, every budgeted matrix (the body) at its K, exactly; the head at
    `head_bits`; `target_bpw`, the bits exllamav3 reports (it writes its own average in the output)."""
    return {"target_bpw": r["bits"], "head_bits": r["head_bits"],
            "tensors": {k: m["K"] for k, m in r["matrices"].items() if k != HEAD}}


def formats(r):
    """The recipe's histogram by container format, the head included."""
    out = {}
    for m in r["matrices"].values():
        f = exl3.FORMATS[int(m["K"] * 2)]
        out[f] = out.get(f, 0) + 1
    return dict(sorted(out.items()))


def write_yaml(path, ex):
    """exllamav3's recipe `ex` as the YAML `convert_model.py --recipe` reads (`yaml.safe_load`)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(ex, f, sort_keys=False)
