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
    layout.write_json_atomic(os.path.join(d, "hot_rows.json"), {"rows": 100, "table_rows": table_rows, "row_bytes": 90})
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
