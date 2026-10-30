"""pytest: the tool's modules are flat files beside the tests; the synthetic sources they share.

`make_sources` builds, in a temporary directory, a reduced served artifact and a reduced public
quant with the real ones' names and structure: four layers (layer 3 GQA), hidden 128, a 256-row
head, the NVFP4 planes with their divisors, the BF16 exception planes, an MTP plane that shares a
body suffix, a vision tensor, a DFlash2 tensor and its graft sidecar; the quant's shards, index,
configs and Hugging Face download records. Payloads are seeded random bytes: the converter copies
bytes and never decodes them.
"""
import hashlib
import json
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import exl3  # noqa: E402

REVISION = "86b95530f9d365d90dd24d01a997a8de0cdf6c55"
BF16_REVISION = "0123456789abcdef0123456789abcdef01234567"
H = 128
LAYERS = 4
VOCAB = 256
# The quant's per-layer matrices: (module, out, in, k2).
GDN = [("linear_attn.in_proj_qkv", 256, H, 8), ("linear_attn.in_proj_z", 128, H, 6),
       ("linear_attn.out_proj", H, 128, 8)]
GQA = [("self_attn.q_proj", 256, H, 6), ("self_attn.k_proj", 128, H, 8), ("self_attn.v_proj", 128, H, 10),
       ("self_attn.o_proj", H, 128, 8)]
MLP = [("mlp.gate_proj", 256, H, 4), ("mlp.up_proj", 256, H, 4), ("mlp.down_proj", H, 256, 6)]


def _bytes(rng, n):
    return bytes(rng.getrandbits(8) for _ in range(n))


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, payload bytes), written in name order."""
    header, offset, body = {}, 0, []
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        body.append(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"".join(body))


def write_container(path, identity, objects):
    """A v2 container (the reader's framing) from (descriptor without offset, payload) pairs, in
    order, each at its layout's alignment: a test fixture only, as the packer writes the real one."""
    payload, directory = bytearray(), []
    for desc, data in objects:
        if desc["kind"] == "tensor":
            align = 4096 if desc["layout"] == "trellis-tile16-v1" else 256
            payload += bytes(-len(payload) % align)
        directory.append(dict(desc, offset=len(payload), bytes=len(data)))
        payload += data
    raw = json.dumps({"identity": identity, "objects": directory}).encode()
    head = b"NINFER\x00\x02" + struct.pack("<Q", len(raw)) + raw
    head += bytes(-len(head) % 4096)
    with open(path, "wb") as f:
        f.write(head + bytes(payload))


def quant_tensors(rng):
    """The reduced quant: name -> (dtype, shape, bytes)."""
    out = {}

    def linear(key, rows, columns, k2):
        out[f"{key}.trellis"] = ("I16", [columns // 16, rows // 16, k2 * 8], _bytes(rng, rows * columns // 16 * k2))
        out[f"{key}.suh"] = ("F16", [columns], _bytes(rng, 2 * columns))
        out[f"{key}.svh"] = ("F16", [rows], _bytes(rng, 2 * rows))
        out[f"{key}.mul1"] = ("I32", [], struct.pack("<I", exl3.MUL1))

    for i in range(LAYERS):
        mods = (GQA if (i + 1) % 4 == 0 else GDN) + MLP
        for module, rows, columns, k2 in mods:
            linear(f"model.language_model.layers.{i}.{module}", rows, columns, k2)
        out[f"model.language_model.layers.{i}.input_layernorm.weight"] = ("BF16", [H], _bytes(rng, 2 * H))
    linear("lm_head", VOCAB, H, 8)
    linear("mtp.fc", H, 2 * H, 6)
    out["model.language_model.embed_tokens.weight"] = ("BF16", [VOCAB, H], _bytes(rng, 2 * VOCAB * H))
    return out


def served_objects(rng):
    """The reduced served artifact: (descriptor, payload) pairs in its directory order."""
    objs = []

    def tensor(name, fmt, layout, shape, size):
        objs.append(({"name": name, "kind": "tensor", "format": fmt, "layout": layout, "shape": shape},
                     _bytes(rng, size)))

    nv, bf, cl = "NVFP4", "BF16", "contiguous-le-v1"
    bs, rs = "blockscale-k16-m128x4-v1", "row-split-k128-v1"
    for name in ("frontend/tokenizer.json", "frontend/chat_template.jinja"):
        objs.append(({"name": name, "kind": "resource", "encoding": "raw-bytes-v1"}, _bytes(rng, 37)))
    tensor("text/token_embedding", "W8G32_F16S", rs, [VOCAB, H], 300)
    for i in range(LAYERS):
        p = f"text/layers/{i}/"
        tensor(p + "input_norm", bf, cl, [H], 2 * H)
        if (i + 1) % 4 == 0:
            exception = i == 3
            tensor(p + "attention/query_key_gate_value", bf if exception else nv, cl if exception else bs, [512, H], 400)
            if not exception:
                tensor(p + "attention/input_projection/input_scale_divisor", "FP32", cl, [], 4)
            tensor(p + "attention/query_norm", bf, cl, [32], 64)
            tensor(p + "attention/output", nv, bs, [H, 128], 200)
            tensor(p + "attention/output_projection/input_scale_divisor", "FP32", cl, [], 4)
        else:
            tensor(p + "gdn/a_log", "FP32", cl, [4], 16)
            tensor(p + "gdn/query_key_value_z", nv, bs, [384, H], 300)
            tensor(p + "gdn/input_projection/input_scale_divisor", "FP32", cl, [], 4)
            tensor(p + "gdn/norm", bf, cl, [32], 64)
            # the layer-4-style quirk: one BF16 gdn/output, no divisor
            if i == 1:
                tensor(p + "gdn/output", bf, cl, [H, 128], 2 * H * 128)
            else:
                tensor(p + "gdn/output", nv, bs, [H, 128], 200)
                tensor(p + "gdn/output_projection/input_scale_divisor", "FP32", cl, [], 4)
        tensor(p + "post_attention_norm", bf, cl, [H], 2 * H)
        tensor(p + "mlp/gate_up", nv, bs, [512, H], 400)
        tensor(p + "mlp/gate_up_projection/input_scale_divisor", "FP32", cl, [], 4)
        tensor(p + "mlp/down", nv, bs, [H, 256], 300)
        tensor(p + "mlp/down_projection/input_scale_divisor", "FP32", cl, [], 4)
    tensor("text/final_norm", bf, cl, [H], 2 * H)
    tensor("text/output_head", "W8G32_F16S", rs, [VOCAB, H], 300)
    tensor("text/draft_head", "Q4G64_F16S", rs, [128, H], 200)
    tensor("text/draft_head_token_ids", "I32", cl, [128], 512)
    tensor("mtp/layer/mlp/gate_up", "W8G32_F16S", rs, [512, H], 400)
    tensor("vision/patch_embedding", "Q6G64_F16S", rs, [64, 128], 100)
    tensor("dflash2/feature_projection", nv, bs, [H, 256], 300)
    tensor("dflash2/context_norm", bf, cl, [H], 2 * H)
    return objs


def make_sources(root, seed=338):
    """Writes the served artifact (+ graft sidecar), the quant and a BF16 index under `root`;
    returns their paths."""
    rng = random.Random(seed)
    served = os.path.join(root, "served", "qwen3_8_27b_nvfp4full-v2.ninfer")
    os.makedirs(os.path.dirname(served))
    write_container(served, {"model_id": "qwen3.8-27b", "weights_id": "nvfp4full"}, served_objects(rng))
    graft = {"recipe_id": "qwen3_8_27b_nvfp4full-v2",
             "grafted_from": {"path": "base.ninfer", "repository": "someone/base", "bytes": 1},
             "source": {"dflash2": {"config": {"num_hidden_layers": 5}}},
             "objects": {"count": 9, "grafted": 2},
             "local_nvfp4": {"encoder_profile": "P", "parents": {
                 "dflash2/feature_projection": {"weight_scale_divisor": None, "relative_frobenius_error": 0.09},
                 "text/layers/0/mlp/down": {"weight_scale_divisor": None, "relative_frobenius_error": 0.1}}}}
    with open(served + ".graft.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(graft, f)

    quant = os.path.join(root, "quant")
    meta = os.path.join(quant, ".cache", "huggingface", "download")
    os.makedirs(meta)
    tensors = quant_tensors(rng)
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
              "model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for shard, part in shards.items():
        write_safetensors(os.path.join(quant, shard), {n: tensors[n] for n in part})
        weight_map.update({n: shard for n in part})
    files = {
        "model.safetensors.index.json": {"metadata": {}, "weight_map": weight_map},
        "quantization_config.json": {"quant_method": "exl3", "version": "1.4.2", "bits": 3.0, "head_bits": 4,
                                     "out_scales": "always", "codebook": "mul1"},
        "config.json": {"architectures": ["Qwen3_5ForConditionalGeneration"]},
    }
    for name, value in files.items():
        with open(os.path.join(quant, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f)
    download_records(quant, list(shards) + list(files), set(shards), REVISION)

    # The BF16 checkpoint: one shard and its index (the reconstruction step is a test double).
    bf16 = os.path.join(root, "bf16")
    os.makedirs(os.path.join(bf16, ".cache", "huggingface", "download"))
    write_safetensors(os.path.join(bf16, "model-00001-of-00001.safetensors"),
                      {"lm_head.weight": ("BF16", [VOCAB, H], _bytes(rng, 2 * VOCAB * H))})
    with open(os.path.join(bf16, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}, f)
    download_records(bf16, ["model-00001-of-00001.safetensors", "model.safetensors.index.json"],
                     {"model-00001-of-00001.safetensors"}, BF16_REVISION)
    return {"served": served, "quant": quant, "bf16": bf16}


def download_records(root, names, lfs, revision):
    """The Hugging Face download records of `names` under `root`: an LFS file's etag is its sha256."""
    meta = os.path.join(root, ".cache", "huggingface", "download")
    for name in names:
        with open(os.path.join(root, name), "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        etag = digest if name in lfs else "0" * 40
        with open(os.path.join(meta, name + ".metadata"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{revision}\n{etag}\n1792534371.28\n")


OVERRIDE_REPO = "someone/abliterated"
OVERRIDE_REVISION = "fedcba9876543210fedcba9876543210fedcba98"
PREFIX = "model.language_model.layers."


def make_override(root, src, seed=420, records=None):
    """An override directory under `root` for the sources `src`, as `replay.py run` writes it
    (records and `replay.json`), and the partial BF16 it was made from, as `replay.py fetch` writes
    it (a shard of the changed tensors, an index, `fetch.json`). `records`: key -> (rows, columns, k2)
    for each matrix it holds, default two body matrices with the quant's own K and shape (a different
    one makes a record the swap must refuse). Returns {"override": dir, "bf16": dir, "keys": [...]};
    its bytes differ from the quant's."""
    rng = random.Random(seed)
    if records is None:
        quant = exl3.Quant(src["quant"])
        records = {}
        for key in (PREFIX + "2.mlp.gate_proj", PREFIX + "3.self_attn.v_proj"):
            m = quant.matrix(key)
            records[key] = (m.rows, m.columns, m.k2)
    tensors = {}
    for key, (rows, columns, k2) in records.items():
        tensors[f"{key}.trellis"] = ("I16", [columns // 16, rows // 16, k2 * 8], _bytes(rng, rows * columns // 16 * k2))
        tensors[f"{key}.suh"] = ("F16", [columns], _bytes(rng, 2 * columns))
        tensors[f"{key}.svh"] = ("F16", [rows], _bytes(rng, 2 * rows))
        tensors[f"{key}.mul1"] = ("I32", [], struct.pack("<I", exl3.MUL1))
    ov = os.path.join(root, "override")
    os.makedirs(ov)
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(os.path.join(ov, shard), tensors)
    files = {"model.safetensors.index.json": {"metadata": {}, "weight_map": {n: shard for n in tensors}},
             "quantization_config.json": {"quant_method": "exl3", "bits": 3.0, "head_bits": 4,
                                          "out_scales": "always", "codebook": "mul1"}}
    for name, value in files.items():
        with open(os.path.join(ov, name), "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f)

    bf16 = os.path.join(root, "override-bf16")
    os.makedirs(bf16)
    weights = {f"{k}.weight": ("BF16", [r, c], _bytes(rng, 2 * r * c)) for k, (r, c, _) in records.items()}
    ref_shard = "model-00006-of-00018.safetensors"
    write_safetensors(os.path.join(bf16, ref_shard), weights)
    with open(os.path.join(bf16, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": {n: ref_shard for n in weights}}, f)
    digests = {n: hashlib.sha256(t[2]).hexdigest() for n, t in weights.items()}
    with open(os.path.join(bf16, "fetch.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": OVERRIDE_REPO, "revision": OVERRIDE_REVISION, "index_sha256": "1" * 64,
                   "shards": {ref_shard: {"bytes": 1 << 20, "lfs_oid": "2" * 64}},
                   "tensors": {n: {"shard": ref_shard, "bytes": len(t[2]), "dtype": "BF16", "shape": t[1],
                                   "sha256": digests[n]} for n, t in weights.items()}}, f)

    quant = exl3.Quant(ov)
    replay = {"schema": "qwen3.8-27b-trellis-replay-v1", "status": "complete",
              "sources": {"quant": {"revision": REVISION},
                          "abliterated": {"repo": OVERRIDE_REPO, "revision": OVERRIDE_REVISION, "tensors": digests}},
              "matrices": {k: {"record_sha256": hashlib.sha256(quant.record(quant.matrix(k))).hexdigest()}
                           for k in records}}
    with open(os.path.join(ov, "replay.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(replay, f)
    return {"override": ov, "bf16": bf16, "keys": sorted(records), "ref_shard": ref_shard}
