"""The 16 GB artifact's object table (spec 27b-16gb/01): the served artifact's directory, in its
order, with the body's linears and the output head replaced by the public quant's trellis
matrices.

- A fused NVFP4 plane of the served artifact becomes the split matrices the quant carries, under
  their exllamav3 names (`gdn/in_proj_qkv` + `gdn/in_proj_z`, `attention/q_proj` / `k_proj` /
  `v_proj`, `mlp/gate_proj` + `mlp/up_proj`): exllamav3 quantized them separately, each with its
  own scales, so they cannot be fused by concatenation (ADR 0060).
- A plane the quant also holds whole keeps the served name (`gdn/output`, `attention/output`,
  `mlp/down`, `text/output_head`): the layer code picks the op by the tensor's format.
- The NVFP4 planes' `*_input_scale_divisor` scalars belong to their W4A4 activations and are
  dropped with them.
- Everything else is the served object, byte for byte: the token embedding, the norms and the GDN
  small set, the draft head, the MTP layer, the vision tower, the DFlash2 module, the frontend.

Only `text/layers/{i}/...` and `text/output_head` are replaced: the MTP layer's planes share the
suffixes but stay the served W8 objects.
"""
import re

UNITS = ("frontend", "text", "mtp", "vision", "dflash2")

# A served plane under `text/layers/{i}/` -> the objects replacing it: (suffix, the quant's module).
REPLACED = {
    "gdn/query_key_value_z": (("gdn/in_proj_qkv", "linear_attn.in_proj_qkv"),
                              ("gdn/in_proj_z", "linear_attn.in_proj_z")),
    "gdn/output": (("gdn/output", "linear_attn.out_proj"),),
    "attention/query_key_gate_value": (("attention/q_proj", "self_attn.q_proj"),
                                       ("attention/k_proj", "self_attn.k_proj"),
                                       ("attention/v_proj", "self_attn.v_proj")),
    "attention/output": (("attention/output", "self_attn.o_proj"),),
    "mlp/gate_up": (("mlp/gate_proj", "mlp.gate_proj"), ("mlp/up_proj", "mlp.up_proj")),
    "mlp/down": (("mlp/down", "mlp.down_proj"),),
}
DROPPED = frozenset({
    "gdn/input_projection/input_scale_divisor",
    "gdn/output_projection/input_scale_divisor",
    "attention/input_projection/input_scale_divisor",
    "attention/output_projection/input_scale_divisor",
    "mlp/gate_up_projection/input_scale_divisor",
    "mlp/down_projection/input_scale_divisor",
})
HEAD = ("text/output_head", "lm_head")
LAYER = re.compile(r"text/layers/(\d+)/(.+)")
QUANT_LAYER = "model.language_model.layers.{}."


class Planned:
    """One object of the new artifact. `copy` names the served object it is; `key` the quant's
    matrix it is built from. Exactly one of them is set."""

    def __init__(self, unit, name, kind, format=None, layout=None, shape=None, encoding=None, copy=None, key=None):
        self.unit, self.name, self.kind = unit, name, kind
        self.format, self.layout, self.shape, self.encoding = format, layout, shape, encoding
        self.copy, self.key = copy, key

    def __repr__(self):
        return f"Planned({self.name!r}, {self.format or self.encoding}, copy={self.copy!r}, key={self.key!r})"


def plan(served_objects, matrices):
    """(objects, dropped): the new artifact's objects in container order, and the served names
    left out. `served_objects` is the served directory; `matrices` maps every quantized key the
    plan may take (the quant's text body and `lm_head`) to its `exl3.Matrix`-like record
    (`format`, `layout`-free `shape` `[out, in]`). Every one of them must be taken exactly once."""
    left = dict(matrices)
    objects, dropped = [], []
    for o in served_objects:
        name = o["name"]
        unit = name.split("/", 1)[0]
        if unit not in UNITS:
            raise ValueError(f"served object {name} is outside the units {UNITS}")
        if o["kind"] == "resource":
            objects.append(Planned(unit, name, "resource", encoding=o["encoding"], copy=name))
            continue
        m = LAYER.fullmatch(name)
        suffix = m.group(2) if m else None
        if suffix in DROPPED:
            dropped.append(name)
            continue
        if suffix in REPLACED:
            prefix = f"text/layers/{m.group(1)}/"
            parts = [(prefix + new, QUANT_LAYER.format(m.group(1)) + module) for new, module in REPLACED[suffix]]
            objects.extend(_trellis(o, parts, left))
        elif name == HEAD[0]:
            objects.extend(_trellis(o, [HEAD], left))
        else:
            objects.append(Planned(unit, name, "tensor", o["format"], o["layout"], list(o["shape"]), copy=name))
    if left:
        raise ValueError(f"{len(left)} quantized matrices have no served plane to replace: {sorted(left)[:5]}")
    return objects, dropped


def _trellis(served, parts, left):
    """The trellis objects replacing the served plane `served`: their rows add up to its rows,
    each takes its columns."""
    rows, columns = served["shape"]
    out, total = [], 0
    for name, key in parts:
        if key not in left:
            raise ValueError(f"{served['name']} needs the quant's {key}, which it does not hold")
        m = left.pop(key)
        r, c = m.shape
        if c != columns:
            raise ValueError(f"{key} takes {c} inputs, {served['name']} {columns}")
        total += r
        out.append(Planned("text", name, "tensor", m.format, "trellis-tile16-v1", [r, c], key=key))
    if total != rows:
        raise ValueError(f"{[k for _, k in parts]} give {total} rows, {served['name']} has {rows}")
    return out
