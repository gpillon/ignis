"""A reduced work tree in the exact layout of layout.md, from synthetic data, CPU only.

For the packer's end-to-end tests: 2 layers of 8 experts spread over at least three K
classes (records at the real projection shapes, so the size table holds), a few FP8 and
BF16 tensors, a 1,000-row table shard, the hash buffers, a hot-row list and DONE markers.
The trellis words are random: this checks layout, not decode.
"""
import json
import os

import numpy as np
import torch

import fp8
import layout
import table

K2_CYCLE = (4, 5, 6, 8, 5, 4, 6, 4)


def _tensor(d, name, t, prefix=""):
    enc = layout.encoding_of(name, tuple(t.shape))
    if enc == "fp8":
        payload, fmt, lay = fp8.encode(t), "FP8_E4M3FN_ROW_BF16S", "row-scale-v1"
    else:
        payload = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
        fmt, lay = "BF16", "contiguous-le-v1"
    with open(os.path.join(d, name + ".bin"), "wb") as f:
        f.write(payload)
    return {"name": prefix + name, "file": name + ".bin", "format": fmt, "layout": lay, "shape": list(t.shape),
            "bytes": len(payload)}


def make(out, n_layers=2, n_experts=8, table_rows=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    work = os.path.join(out, "work")
    for L in range(n_layers):
        d = os.path.join(work, "layers", f"L{L:02d}")
        os.makedirs(d, exist_ok=True)

        def record(e, p, L=L):
            k2 = K2_CYCLE[(2 * e + (p == "dn") + L) % len(K2_CYCLE)]
            i, o = layout.SHAPES[p]
            return k2, {"trellis": rng.integers(-2 ** 15, 2 ** 15, (i // 16, o // 16, 8 * k2), dtype=np.int16),
                        "suh": rng.standard_normal(i).astype(np.float16),
                        "svh": rng.standard_normal(o).astype(np.float16)}
        layout.write_experts(d, n_experts, record)
        tensors = [_tensor(d, "linear_attn.in_proj_qkv.weight", torch.randn(64, 128, generator=g), f"layers.{L}."),
                   _tensor(d, "mlp.gate.weight", torch.randn(n_experts, 128, generator=g), f"layers.{L}."),
                   _tensor(d, "attn_hyper_connection.hc_norm.weight", torch.randn(512, generator=g), f"layers.{L}.")]
        layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": tensors})
        layout.mark_done(d)
    d = os.path.join(work, "global")
    os.makedirs(d, exist_ok=True)
    tensors = [_tensor(d, "embed_tokens.weight", torch.randn(100, 128, generator=g)),
               _tensor(d, "lm_head.weight", torch.randn(100, 128, generator=g))]
    layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": tensors})
    layout.mark_done(d)
    d = os.path.join(work, "ngram")
    os.makedirs(os.path.join(d, "table"), exist_ok=True)
    rows = (torch.randn(table_rows, table.DIM, generator=g) * 0.05).to(torch.bfloat16)
    table.encode_rows(rows).tofile(os.path.join(d, "table", "shard_000.int4"))
    layout.mark_done(os.path.join(d, "table"))
    np.array([23703573157769, 20109073645365, 8052911324071], dtype="<i8").tofile(os.path.join(d, "layer_multipliers.i64"))
    np.arange(16, dtype="<i8").tofile(os.path.join(d, "ngram_heads_vocab_sizes.i64"))
    np.arange(16, dtype="<i8").tofile(os.path.join(d, "ngram_heads_offsets.i64"))
    table.hot_rows(rng.integers(0, table_rows, 5000), 100).astype("<u4").tofile(os.path.join(d, "hot_rows.u32"))
    layout.write_json_atomic(os.path.join(d, "hot_rows.json"), {"rows": 100, "table_rows": table_rows, "row_bytes": 90,
                                                                  "shards": 1, "complete": True})
    layout.mark_done(d)
    d = os.path.join(work, "frontend")
    os.makedirs(d, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json", "video_preprocessor_config.json", "config.json"):
        with open(os.path.join(d, name), "w") as f:
            f.write("{}\n")
    layout.mark_done(d)
    layout.write_json_atomic(os.path.join(work, "converter.json"),
                             {"schema": "flash-next-converter-v1", "status": "fixture", "layers": n_layers,
                              "experts": n_experts, "table_rows": table_rows})
    return work


# ---------------------------------------------------------------- the reduced-geometry tree

REDUCED_SHAPES = {"gu": (256, 256), "dn": (128, 256)}
PLE_LAYER = 1


def reduced_config():
    """The packer's FlashNextGeometry::fixture() as a Qwen4Exp text config: 2 layers (GDN, then
    attention), hidden 256, 8 experts of 128, a 1,000-row table. The PLE sits on layer 1 as in the
    real model, but layer 1 is attention here, which transformers' config refuses: the config
    carries no PLE layer and make_reduced builds layer 1's PLE module beside it."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    return Qwen4ExpTextConfig(
        vocab_size=512, hidden_size=256, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        full_attention_interval=2, hc_count=4, hc_lowrank=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_conv_kernel_dim=4, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=64, indexer_compress_ratio=4,
        num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=128, shared_expert_intermediate_size=128, ple_layer_ids=[], ple_embed_dim=320,
        ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=490,
        make_ngram_vocab_size_divisible_by=1000, split_ngram_parts=2, partial_rotary_factor=0.25,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                         "mrope_section": [3, 3, 2], "mrope_interleaved": True})


def _randomize(module, g):
    for p in module.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.05


def make_reduced(work, seed=0):
    """A complete work tree at the reduced geometry, from the HF modeling code on CPU, through
    the converter's own writers (nonexpert.write_tensor, layout.write_experts, table.encode_rows):
    every per-layer tensor of the real inventory at reduced shapes, records in all 8 K classes."""
    import nonexpert
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
    cfg = reduced_config()
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    k_map = []
    table_weight = bufs = None
    for L in range(cfg.num_hidden_layers):
        d = os.path.join(work, "layers", f"L{L:02d}")
        os.makedirs(d, exist_ok=True)
        layer = mq.Qwen4ExpTextDecoderLayer(cfg, L)
        _randomize(layer, g)
        sd = layer.state_dict()
        if L == PLE_LAYER:
            ple = mq.Qwen4ExpTextPLELayer(cfg, L, 0)
            _randomize(ple, g)
            sd.update({"ple." + k: v for k, v in ple.state_dict().items()})
            emb = ple.ple_embedding
            bufs = {k: getattr(emb, k) for k in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")}
            table_weight = emb.ngram_embedding.weight.data
        entries = [nonexpert.write_tensor(d, name, t, prefix=f"layers.{L}.") for name, t in sd.items()
                   if layout.encoding_of(name, tuple(t.shape)) not in ("expert",)
                   and not name.startswith("ple.ple_embedding.")]
        layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": entries})
        picks = {"gu": [], "dn": []}

        def record(e, p, L=L):
            k2 = layout.K2_SET[(e + (p == "dn") + L) % len(layout.K2_SET)]
            picks[p].append(k2)
            i, o = REDUCED_SHAPES[p]
            return k2, {"trellis": rng.integers(-2 ** 15, 2 ** 15, (i // 16, o // 16, 8 * k2), dtype=np.int16),
                        "suh": rng.standard_normal(i).astype(np.float16),
                        "svh": rng.standard_normal(o).astype(np.float16)}
        layout.write_experts(d, cfg.num_experts, record, REDUCED_SHAPES)
        k_map.append({"layer": L, "gu": picks["gu"], "dn": picks["dn"]})
        layout.mark_done(d)
    d = os.path.join(work, "global")
    os.makedirs(d, exist_ok=True)
    mixer = mq.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    _randomize(mixer, g)
    tensors = [nonexpert.write_tensor(d, "embed_tokens.weight", torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g)),
               nonexpert.write_tensor(d, "lm_head.weight", torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g))]
    tensors += [nonexpert.write_tensor(d, "hyper_connection_mixer." + k, t) for k, t in mixer.state_dict().items()]
    layout.write_json_atomic(os.path.join(d, "tensors.json"), {"tensors": tensors})
    layout.mark_done(d)
    d = os.path.join(work, "ngram")
    os.makedirs(os.path.join(d, "table"), exist_ok=True)
    rows = table_weight.shape[0]
    half = rows // 2
    for n in range(2):
        table.encode_rows(table_weight[n * half:(n + 1) * half]).tofile(os.path.join(d, "table", f"shard_{n:03d}.int4"))
    layout.mark_done(os.path.join(d, "table"))
    for k, v in bufs.items():
        v.numpy().astype("<i8").tofile(os.path.join(d, f"{k}.i64"))
    hot = table.hot_rows(rng.integers(0, rows, 5000), 100)
    hot.astype("<u4").tofile(os.path.join(d, "hot_rows.u32"))
    layout.write_json_atomic(os.path.join(d, "hot_rows.json"), {
        "rows": int(hot.size), "bytes": int(hot.size * table.ROW_BYTES), "table_rows": rows, "row_bytes": 90,
        "shards": 2, "complete": True})
    layout.mark_done(d)
    d = os.path.join(work, "frontend")
    os.makedirs(d, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json", "video_preprocessor_config.json"):
        with open(os.path.join(d, name), "w", newline="\n") as f:
            f.write("{}\n")
    text = cfg.to_dict()
    text["ple_layer_ids"] = [PLE_LAYER + 1]
    with open(os.path.join(d, "config.json"), "w", newline="\n") as f:
        json.dump({"text_config": text}, f, indent=1, default=str)
    layout.mark_done(d)
    layout.write_json_atomic(os.path.join(work, "converter.json"), {
        "schema": "flash-next-converter-v1", "status": "fixture", "layers": cfg.num_hidden_layers,
        "experts": cfg.num_experts, "table_rows": rows, "geometry": "fixture", "k_map": {"layers": k_map}})
    return work
