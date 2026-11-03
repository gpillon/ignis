"""The recipe: where each object comes from and how it is made (docs/formats/27b-nvfp4-work-tree.md §4).

A port of ninfer's `recipe_nvfp4full.py` (with the official recipes of `qwen3_6_27b/recipe.py` and the
drafter's of `recipe_nvfp4full._build_dflash2_recipes`) by geometry, with one change: **no quantized
source**. Every one of the 247 text NVFP4 parents is encoded here from the BF16 checkpoint, and every one
of their 247 activation divisors is the calibration's; the served artifact copied 112 of each from a
third-party NVFP4 quant of the base, which no fine-tune has.

The routes, each object in exactly one:

- `local_nvfp4` (247): a text NVFP4 parent, its rows fused from the checkpoint's projections;
- `input_divisor` (247): a parent's activation divisor, the calibration's `d_x`;
- `bf16_exception` (9): a projection the positional pattern keeps in BF16;
- `base_direct` (401): the norms and the GDN small set as the checkpoint holds them;
- `official` (349): the W8 token embedding and output head, the Q4 draft head and its ids, the MTP layer
  and the vision tower;
- `dflash2` (66): the base's drafter, its NVFP4 matrices weight-only;
- `resource` (6): the frontend files, the checkpoint's bytes.
"""
from dataclasses import dataclass

import torch

import inventory

TEXT = "model.language_model."
EXPECTED_ROUTES = {"resource": 6, "local_nvfp4": 247, "input_divisor": 247, "bf16_exception": 9,
                   "base_direct": 401, "official": 349, "dflash2": 66}


@dataclass(frozen=True)
class Recipe:
    """An object's route and its parts: (source tensor, row ranges or None for all), fused along rows;
    `transform` is `fp32` (a cast), `conv` (`[C, 1, Ck]` stored `[Ck, C]`), `flatten` (to the object's
    shape) or `gather` (the rows the draft head's ids pick)."""
    route: str
    parts: tuple = ()
    transform: str = None


def _all(name):
    return (name, None)


def q_part(name, g, gate):
    """The query (or the output gate) rows of a head-interleaved `q_proj`: per head, `D` query rows then
    `D` gate rows."""
    D = g.head_dim
    return (name, tuple((h * 2 * D + (D if gate else 0), h * 2 * D + (D if gate else 0) + D)
                        for h in range(g.attention_heads)))


def _qkgv(p, g):
    q = p + "self_attn.q_proj.weight"
    return (q_part(q, g, False), _all(p + "self_attn.k_proj.weight"), q_part(q, g, True),
            _all(p + "self_attn.v_proj.weight"))


def recipes(g):
    """object name -> Recipe, for every object of the inventory."""
    out = {f"frontend/{f}": Recipe("resource", ((f, None),)) for f in inventory.FRONTEND_FILES}
    for i in range(inventory.TEXT_LAYERS):
        s, o = f"{TEXT}layers.{i}.", f"text/layers/{i}/"

        def matrix(name, parts, exception):
            if exception:
                out[o + name] = Recipe("bf16_exception", parts)
            else:
                out[o + name] = Recipe("local_nvfp4", parts)
                out[inventory.divisor_name(o + name)] = Recipe("input_divisor", ((o + name, None),))

        out[o + "input_norm"] = Recipe("base_direct", (_all(s + "input_layernorm.weight"),))
        out[o + "post_attention_norm"] = Recipe("base_direct", (_all(s + "post_attention_layernorm.weight"),))
        if inventory.is_gqa(i):
            matrix("attention/query_key_gate_value", _qkgv(s, g), i in inventory.QKGV_BF16_LAYERS)
            matrix("attention/output", (_all(s + "self_attn.o_proj.weight"),), i in inventory.ATTENTION_OUT_BF16_LAYERS)
            out[o + "attention/query_norm"] = Recipe("base_direct", (_all(s + "self_attn.q_norm.weight"),))
            out[o + "attention/key_norm"] = Recipe("base_direct", (_all(s + "self_attn.k_norm.weight"),))
        else:
            q = s + "linear_attn."
            matrix("gdn/query_key_value_z", (_all(q + "in_proj_qkv.weight"), _all(q + "in_proj_z.weight")), False)
            matrix("gdn/output", (_all(q + "out_proj.weight"),), i in inventory.GDN_OUT_BF16_LAYERS)
            out[o + "gdn/a_log"] = Recipe("base_direct", (_all(q + "A_log"),), "fp32")
            out[o + "gdn/dt_bias"] = Recipe("base_direct", (_all(q + "dt_bias"),), "fp32")
            out[o + "gdn/convolution"] = Recipe("base_direct", (_all(q + "conv1d.weight"),), "conv")
            out[o + "gdn/a_b_projection"] = Recipe("base_direct", (_all(q + "in_proj_a.weight"),
                                                                   _all(q + "in_proj_b.weight")))
            out[o + "gdn/norm"] = Recipe("base_direct", (_all(q + "norm.weight"),))
        matrix("mlp/gate_up", (_all(s + "mlp.gate_proj.weight"), _all(s + "mlp.up_proj.weight")), False)
        matrix("mlp/down", (_all(s + "mlp.down_proj.weight"),), False)
    out["text/final_norm"] = Recipe("base_direct", (_all(TEXT + "norm.weight"),))
    out["text/token_embedding"] = Recipe("official", (_all(TEXT + "embed_tokens.weight"),))
    out["text/output_head"] = Recipe("official", (_all("lm_head.weight"),))
    out["text/draft_head"] = Recipe("official", (_all("lm_head.weight"),), "gather")
    out["text/draft_head_token_ids"] = Recipe("official", (), "ids")

    m = "mtp.layers.0."
    for name, parts in (
            ("mtp/input_projection", (_all("mtp.fc.weight"),)),
            ("mtp/embedding_norm", (_all("mtp.pre_fc_norm_embedding.weight"),)),
            ("mtp/hidden_norm", (_all("mtp.pre_fc_norm_hidden.weight"),)),
            ("mtp/layer/input_norm", (_all(m + "input_layernorm.weight"),)),
            ("mtp/layer/attention/query_key_gate_value", _qkgv(m, g)),
            ("mtp/layer/attention/query_norm", (_all(m + "self_attn.q_norm.weight"),)),
            ("mtp/layer/attention/key_norm", (_all(m + "self_attn.k_norm.weight"),)),
            ("mtp/layer/attention/output", (_all(m + "self_attn.o_proj.weight"),)),
            ("mtp/layer/post_attention_norm", (_all(m + "post_attention_layernorm.weight"),)),
            ("mtp/layer/mlp/gate_up", (_all(m + "mlp.gate_proj.weight"), _all(m + "mlp.up_proj.weight"))),
            ("mtp/layer/mlp/down", (_all(m + "mlp.down_proj.weight"),)),
            ("mtp/final_norm", (_all("mtp.norm.weight"),))):
        out[name] = Recipe("official", parts)

    v = "model.visual."
    out["vision/patch_embedding"] = Recipe("official", (_all(v + "patch_embed.proj.weight"),), "flatten")
    out["vision/patch_embedding_bias"] = Recipe("official", (_all(v + "patch_embed.proj.bias"),))
    out["vision/position_embedding"] = Recipe("official", (_all(v + "pos_embed.weight"),))
    for b in range(inventory.VISION_LAYERS):
        for obj, src in (("attention/qkv", "attn.qkv.weight"), ("attention/qkv_bias", "attn.qkv.bias"),
                         ("attention/output", "attn.proj.weight"), ("attention/output_bias", "attn.proj.bias"),
                         ("mlp/fc1", "mlp.linear_fc1.weight"), ("mlp/fc1_bias", "mlp.linear_fc1.bias"),
                         ("mlp/fc2", "mlp.linear_fc2.weight"), ("mlp/fc2_bias", "mlp.linear_fc2.bias"),
                         ("norm1/weight", "norm1.weight"), ("norm1/bias", "norm1.bias"),
                         ("norm2/weight", "norm2.weight"), ("norm2/bias", "norm2.bias")):
            out[f"vision/layers/{b}/{obj}"] = Recipe("official", (_all(f"{v}blocks.{b}.{src}"),))
    for obj, src in (("fc1", "linear_fc1.weight"), ("fc1_bias", "linear_fc1.bias"), ("fc2", "linear_fc2.weight"),
                     ("fc2_bias", "linear_fc2.bias"), ("norm/weight", "norm.weight"), ("norm/bias", "norm.bias")):
        out[f"vision/merger/{obj}"] = Recipe("official", (_all(v + "merger." + src),))

    def drafter(name, *sources):
        out[name] = Recipe("dflash2", tuple(_all(s) for s in sources))

    drafter("dflash2/feature_projection", "fc.weight")
    drafter("dflash2/context_norm", "hidden_norm.weight")
    for l in range(inventory.DFLASH2_LAYERS):
        s, o = f"layers.{l}.", f"dflash2/layers/{l}/"
        drafter(o + "input_norm", s + "input_layernorm.weight")
        drafter(o + "attention/query_key_value", s + "self_attn.q_proj.weight", s + "self_attn.k_proj.weight",
                s + "self_attn.v_proj.weight")
        drafter(o + "attention/query_norm", s + "self_attn.q_norm.weight")
        drafter(o + "attention/key_norm", s + "self_attn.k_norm.weight")
        drafter(o + "attention/output", s + "self_attn.o_proj.weight")
        drafter(o + "attention/conv_base", s + "attention_conv.base_kernel")
        drafter(o + "attention/conv_proj", s + "attention_conv.kernel_projection.weight")
        drafter(o + "post_attention_norm", s + "post_attention_layernorm.weight")
        drafter(o + "mlp/gate_up", s + "mlp.gate_proj.weight", s + "mlp.up_proj.weight")
        drafter(o + "mlp/down", s + "mlp.down_proj.weight")
        drafter(o + "mlp/conv_base", s + "mlp_conv.base_kernel")
        drafter(o + "mlp/conv_proj", s + "mlp_conv.kernel_projection.weight")
    drafter("dflash2/final_norm", "norm.weight")
    drafter("dflash2/selector/hidden", "candidate_selector.hidden_projection.weight")
    drafter("dflash2/selector/predecessor", "candidate_selector.predecessor_codebook")
    drafter("dflash2/selector/successor", "candidate_selector.successor_codebook")
    return out


def counts(table):
    """route -> objects."""
    out = {}
    for r in table.values():
        out[r.route] = out.get(r.route, 0) + 1
    return out


def problems(table, g):
    """What makes `table` another recipe than the local-only one: its route counts, an object of the
    inventory with no route or a route with no object, an NVFP4 text parent not encoded locally, a divisor
    not bound to a local parent."""
    objs = inventory.objects(g)
    names = {o.name for o in objs}
    out = []
    got = counts(table)
    for route in sorted(set(got) | set(EXPECTED_ROUTES)):
        if got.get(route, 0) != EXPECTED_ROUTES.get(route, 0):
            out.append(f"{got.get(route, 0)} objects by route {route}, the local-only recipe has "
                       f"{EXPECTED_ROUTES.get(route, 0)} (no source parents, 247 local)")
    out += [f"{n} has no route" for n in sorted(names - set(table))]
    out += [f"{n} is routed and is no object of the inventory" for n in sorted(set(table) - names)]
    for o in objs:
        r = table.get(o.name)
        if r is None:
            continue
        if o.unit == "text" and o.format == inventory.NVFP4 and r.route != "local_nvfp4":
            out.append(f"{o.name} is an NVFP4 parent routed {r.route}, not encoded locally")
        if r.route == "input_divisor" and (table.get(r.parts[0][0]) or Recipe("")).route != "local_nvfp4":
            out.append(f"{o.name} is not the divisor of a local parent")
    return out


def materialize(recipe, obj, reader, draft_ids=None):
    """The tensor of a checkpoint or drafter object (not a resource, a divisor or the ids)."""
    pieces = []
    for name, rows in recipe.parts:
        t = reader.get(name)
        if rows is not None:
            t = torch.cat([t[a:b] for a, b in rows], dim=0)
        pieces.append(t)
    t = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    if recipe.transform == "fp32":
        t = t.to(torch.float32)
    elif recipe.transform == "conv":
        t = t.reshape(t.shape[0], t.shape[2]).transpose(0, 1).contiguous()
    elif recipe.transform == "flatten":
        t = t.reshape(obj.shape)
    elif recipe.transform == "gather":
        t = t.index_select(0, torch.from_numpy(draft_ids))
    return t
