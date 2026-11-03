"""The per-tensor recipe of a local exllamav3 conversion at a published quant's bit widths (GitHub #471,
spec #449: Swift 1.5's 16 GB artifact).

A fine-tune of the 27B is converted by exllamav3 1.5.3 (`convert_model.py --recipe`) with every matrix at
the K it has in `SC_3.00bpw_H4`, the quant the 16 GB 27B artifact repacks: the allocation is the quant's,
not re-optimized for the fine-tune. The recipe is derived from the quant's `quantization_config.json`
alone, a pure step (no torch, no file):

- each matrix's K is its `tensor_storage` entry's `bits_per_weight`, which must be the K its stored trellis
  holds (`exl3.py`: the last dimension is 16·K) and one the 16 GB container stores (`exl3.FORMATS`);
- its shape is the trellis's, `[out, in]`;
- the head (`lm_head`) is at `head_bits`, 4: exllamav3 takes the head from `head_bits` and ignores a
  recipe entry for it, so exllamav3's recipe lists only the body;
- the rest of the recipe is the quant's: bits 3.0, the `mul1` codebook, output scales always, its MTP
  bits and its calibration size, checked as the 16 GB converter checks a quant (`convert.check_quant_config`).
"""
import yaml

import convert
import exl3

BODY = "model.language_model.layers."
HEAD = "lm_head"
# The K the 16 GB container stores, by 2·K (`exl3.FORMATS`).
STORED = {k2 / 2 for k2 in exl3.FORMATS}


def rate(k):
    """K as exllamav3 writes a rate: an int when whole (`allocation._as_rate`)."""
    return int(k) if float(k).is_integer() else float(k)


def derive(config):
    """The recipe of the quant whose quantization_config.json is `config`: every quantized matrix of the
    body and the head, `{key: {"K", "shape"}}` in key order, and the quant's recipe beside it. Refused
    (`convert.Refused`) names the matrix it cannot take."""
    convert.check_quant_config(config)
    matrices = {}
    for key, entry in sorted((config.get("tensor_storage") or {}).items()):
        if entry.get("quant_format") != "exl3":
            continue
        if not (key.startswith(BODY) or key == HEAD):
            raise convert.Refused(f"{key}: a quantized matrix outside the body and the head, which the recipe "
                                  f"does not give a K")
        shape = ((entry.get("stored_tensors") or {}).get(f"{key}.trellis") or {}).get("shape")
        if not shape or len(shape) != 3 or shape[2] % 8:
            raise convert.Refused(f"{key}: its trellis's shape {shape} holds no K")
        k = rate(shape[2] / 16)
        if entry.get("bits_per_weight") != k:
            raise convert.Refused(f"{key}: bits_per_weight {entry.get('bits_per_weight')}, its trellis stores K {k}")
        if k not in STORED:
            raise convert.Refused(f"{key}: K {k}, which the 16 GB artifact cannot store ({sorted(STORED)})")
        if key == HEAD and k != config["head_bits"]:
            raise convert.Refused(f"{key}: K {k}, the head is quantized at head_bits {config['head_bits']}")
        matrices[key] = {"K": k, "shape": [shape[1] * 16, shape[0] * 16]}
    if HEAD not in matrices:
        raise convert.Refused(f"the quant has no quantized {HEAD}")
    return {"matrices": matrices, "head_bits": config["head_bits"], "mtp_bits": config.get("mtp_bits"),
            "bits": config["bits"], "codebook": config["codebook"], "out_scales": config["out_scales"],
            "calibration": config.get("calibration")}


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
