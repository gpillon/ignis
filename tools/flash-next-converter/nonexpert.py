"""Non-expert tensors as work files (layout.md §6): each file holds exactly its container payload.

Used by the pass and by the reduced fixture, so both write the same bytes.
"""
import hashlib
import json
import os

import torch

import fp8
import layout


def write_tensor(d, name, t, prefix=""):
    """Writes one non-expert tensor (layer-local `name`) as its container payload into `d` and
    returns its tensors.json entry; the entry's name is `prefix + name`."""
    enc = layout.encoding_of(name, tuple(t.shape))
    if enc == "fp8":
        payload = fp8.encode(t)
        fmt, lay = "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"
    else:
        payload = t.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().tobytes()
        fmt, lay = "BF16", "contiguous-le-v1"
    fname = name + ".bin"
    with open(os.path.join(d, fname), "wb") as f:
        f.write(payload)
    return {"name": prefix + name, "file": fname, "format": fmt, "layout": lay, "shape": list(t.shape),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "fp8_measured_by_study": layout.measured_by_study(name) if enc == "fp8" else None}


def fp8_weights(d, device):
    """layer-local name -> the bf16 weight each FP8 payload of a directory stands for."""
    out = {}
    for t in json.load(open(os.path.join(d, "tensors.json")))["tensors"]:
        if t["format"].startswith("FP8"):
            raw = bytearray(open(os.path.join(d, t["file"]), "rb").read())
            name = t["name"].split(".", 2)[2] if t["name"].startswith("layers.") else t["name"]
            out[name] = fp8.decode(raw, t["shape"], device)
    return out
