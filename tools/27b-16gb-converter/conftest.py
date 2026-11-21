"""pytest: the tool's modules are flat files beside the tests; the synthetic sources they share.

`make_sources` builds, in a temporary directory, a reduced served artifact and a reduced public
quant with the real ones' names and structure: four layers (layer 3 GQA), hidden 128, a 256-row
head, the NVFP4 planes with their divisors, the BF16 exception planes, an MTP plane that shares a
body suffix, a vision tensor, a DFlash2 tensor and its graft sidecar; the quant's shards, index,
configs and Hugging Face download records. Payloads are seeded random bytes: the converter copies
bytes and never decodes them. `make_replay_sources` adds what the abliterated replay reads, and
`ReplayExllamav3` runs the replay's GPU step on the CPU.
"""
import hashlib
import json
import os
import random
import shutil
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import container  # noqa: E402
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


def make_sources(root, seed=338, recipes=None):
    """Writes the served artifact (+ graft sidecar), the quant and a BF16 index under `root`, and the
    quant's recipe `sc30` in `recipes` (default `<root>/recipes`); returns their paths."""
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
    # The default recipe, `sc30`, of the reduced quant: its K and its files' pins (GitHub #631).
    recipes = recipes or os.path.join(root, "recipes")
    write_recipe(os.path.join(recipes, "sc30.json"), files=pin_dir(quant))
    return {"served": served, "quant": quant, "bf16": bf16, "recipes": recipes}


def body_k():
    """The reduced quant's body, each matrix at its K, from the tables above."""
    return {f"model.language_model.layers.{i}.{module}": k2 // 2
            for i in range(LAYERS) for module, _, _, k2 in (GQA if (i + 1) % 4 == 0 else GDN) + MLP}


def pin_dir(*roots):
    """{name: {bytes, sha256}} of every file directly under each of `roots` (a later root's file wins)."""
    out = {}
    for root in roots:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    out[name] = {"bytes": os.path.getsize(path), "sha256": hashlib.sha256(f.read()).hexdigest()}
    return out


def write_recipe(path, **fields):
    """A recipe file (`recipe.load`'s): the reduced quant's K, the head at 4, today's configuration, a
    ceiling of 1 GiB, exllamav3's job; `fields` replace any of them."""
    data = {"schema": "qwen3.8-27b-trellis-recipe-v1", "weights_id": "trellis-sc30", "ceiling_bytes": 1 << 30,
            "quantization_config": {"quant_method": "exl3", "codebook": "mul1", "out_scales": "always", "bits": 3.0,
                                    "head_bits": 4},
            "job": {"mtp_bits": 3, "calibration": {"rows": 250, "cols": 2048}},
            "tensors": body_k(), "files": {}}
    data.update(fields)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f)
    return path


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


FINE_TUNE_SERVED = "nvfp4full-swift-1.5"
FINE_TUNE_FILE = "qwen3_8_27b_nvfp4full-v2-ukisai-swift-1.5.ninfer"


def make_fine_tune_served(src, sidecar="conversion", weights_id=FINE_TUNE_SERVED, seed=465):
    """A fine-tune's served artifact beside `src`'s (GitHub #465): the same directory under its own
    weights id, with payloads of its own, and its sidecar: the `.conversion.json` the
    `qwen3.8-27b-nvfp4` packer family writes (the converter's record carried through, the recipe id,
    the artifact's bytes and object count, every local NVFP4 parent with its divisor), or with
    `sidecar="graft"` a copy of the served graft record. Returns `src` with it as "served"."""
    rng = random.Random(seed)
    vanilla = container.Container(src["served"])
    path = os.path.join(os.path.dirname(src["served"]), FINE_TUNE_FILE)
    objects = [({k: v for k, v in o.items() if k not in ("offset", "bytes")}, _bytes(rng, o["bytes"]))
               for o in vanilla.objects]
    write_container(path, {"model_id": "qwen3.8-27b", "weights_id": weights_id}, objects)
    if sidecar == "graft":
        with open(src["served"] + ".graft.json", "rb") as f, open(path + ".graft.json", "wb") as g:
            g.write(f.read())
    else:
        record = {"identity": {"model_id": "qwen3.8-27b", "weights_id": weights_id},
                  "source": {"repo": "someone/fine-tune", "revision": "a" * 40, "license": "swift-open-license-1.0"},
                  "recipe_id": "qwen3_8_27b_nvfp4full-v2-local",
                  "artifact": {"bytes": os.path.getsize(path)},
                  "objects": {"count": len(objects)},
                  "local_nvfp4": {"encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1", "parents": {
                      "dflash2/feature_projection": {"weight_scale_divisor": None, "relative_frobenius_error": 0.08},
                      "text/layers/0/mlp/down": {"weight_scale_divisor": 2688.0, "relative_frobenius_error": 0.07}}}}
        with open(path + ".conversion.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f)
    return dict(src, served=path)


def make_local_quant(src):
    """`src`'s quant as a local conversion leaves it (GitHub #465): no Hugging Face download record, and
    beside it the pins its conversion recorded, every file's bytes and sha256 (`{"files": {name:
    {bytes, sha256}}}`). Returns the pins' path."""
    shutil.rmtree(os.path.join(src["quant"], ".cache"))
    files = {}
    for name in sorted(os.listdir(src["quant"])):
        with open(os.path.join(src["quant"], name), "rb") as f:
            data = f.read()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path = src["quant"] + ".pins.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"files": files}, f)
    return path


# --- the abliterated replay (#421, #703) ---

CAL_TRACE = "cal_trace.safetensors"
# The reduced model's abliterated layers, the replay's whole range, as layers 17-51 are the 27B's.
REPLAY_LAYERS = (1, 3)
# The calibration size the reduced quants state.
CALIBRATION = {"rows": 4, "cols": 16}
# What `quantization_config.json` states of a quant at the 12 GB recipe, `ig28a`, beside `sc30`'s.
IG28A_CONFIG = {"quant_method": "exl3", "codebook": "mul1", "out_scales": "always", "bits": 2.79, "head_bits": 4}


def replayed_keys(first, last):
    """The matrices the replay takes in layers `first`..`last` of the reduced quant: each layer's output
    projection (the attention's at every fourth layer, from layer 3), then its down projection."""
    keys = []
    for layer in range(first, last + 1):
        out = "self_attn.o_proj" if (layer + 1) % 4 == 0 else "linear_attn.out_proj"
        keys += [f"{PREFIX}{layer}.{out}", f"{PREFIX}{layer}.mlp.down_proj"]
    return keys


def write_fetched(root, quant, keys, seed=421):
    """The directory `replay.py fetch` writes for the matrices `keys` of `quant` (an `exl3.Quant`): their
    BF16 weights in one shard named after the base's, its index and `fetch.json`."""
    rng = random.Random(seed)
    weights = {}
    for key in keys:
        m = quant.matrix(key)
        weights[key + ".weight"] = ("BF16", m.shape, _bytes(rng, 2 * m.rows * m.columns))
    os.makedirs(root)
    shard = "model-00006-of-00018.safetensors"
    write_safetensors(os.path.join(root, shard), weights)
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"metadata": {"source": f"{OVERRIDE_REPO}@{OVERRIDE_REVISION}"},
                   "weight_map": {n: shard for n in sorted(weights)}}, f)
    with open(os.path.join(root, "fetch.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"repo": OVERRIDE_REPO, "revision": OVERRIDE_REVISION, "index_sha256": "1" * 64,
                   "shards": {shard: {"bytes": 1 << 20, "lfs_oid": "2" * 64}},
                   "tensors": {n: {"shard": shard, "bytes": len(t[2]), "dtype": "BF16", "shape": t[1],
                                   "sha256": hashlib.sha256(t[2]).hexdigest()} for n, t in sorted(weights.items())}}, f)
    return root


def make_replay_sources(root, local=False):
    """The reduced sources of `make_sources` as the replay reads them: the quant states its calibration
    size, its calibration trace is at hand, and `abliterated` is the directory `replay.py fetch` writes for
    the replay's matrices of layers 1-3. A downloaded quant (the default) holds the trace, with its download
    record, and is the recipe `sc30`'s. A local one (`local`, GitHub #703) is the same matrices converted on
    this machine at `ig28a`'s 2.79 bits by exllamav3 1.5.3: no download record, its pins beside it
    (`pins`), and the same trace in a directory of its own (`trace`), which its recipe `ig28a` pins."""
    src = make_sources(root)
    quant = src["quant"]
    path = os.path.join(quant, "quantization_config.json")
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    config["calibration"] = CALIBRATION
    if local:
        config.update(bits=IG28A_CONFIG["bits"], version="1.5.3")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f)
    trace = os.path.join(root, "trace") if local else quant
    os.makedirs(trace, exist_ok=True)
    rows, cols = CALIBRATION["rows"], CALIBRATION["cols"]
    write_safetensors(os.path.join(trace, CAL_TRACE),
                      {"input_ids": ("I64", [rows, cols], bytes(i % 251 for i in range(8 * rows * cols)))})
    out = dict(src, abliterated=write_fetched(os.path.join(root, "abliterated"), exl3.Quant(quant),
                                              replayed_keys(*REPLAY_LAYERS)))
    if local:
        out.update(pins=make_local_quant(src), trace=trace, recipe="ig28a")
        write_recipe(os.path.join(src["recipes"], "ig28a.json"), weights_id="trellis-ig28a",
                     quantization_config=IG28A_CONFIG, files=pin_dir(trace))
    else:
        download_records(quant, ["quantization_config.json", CAL_TRACE], {CAL_TRACE}, REVISION)
        write_recipe(os.path.join(src["recipes"], "sc30.json"), files=pin_dir(quant))
        out.update(recipe="sc30")
    return out


class StubTensor:
    """A tensor of the stub torch: a numpy array."""

    def __init__(self, a):
        self.a = a

    @property
    def shape(self):
        return self.a.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def all(self):
        return StubTensor(self.a.all())

    def item(self):
        return self.a.item()


class ReplayExllamav3:
    """The replay's GPU step (`replay.run`) on the CPU: torch, exllamav3 and safetensors' `safe_open` are
    stubs, so nothing reaches a GPU. The stub model is `src`'s quant, its token embedding then one module per
    layer holding a linear per matrix of the quant; its calibration is the trace's size in zeros; its
    quantizer gives each matrix a record of its own at the K and seed it is asked for (the trellis filled
    with a number its key gives, the scales one). It also stands in for the GPU's name and peaks and for the
    reconstruction step (each matrix's error a number of its own), and fixes the converter's commit and the
    versions, so a record is the same wherever it is made; the replay's whole range is layers 1-3."""

    PEAKS = {"allocated": 3 << 30, "reserved": 4 << 30}

    def __init__(self, src):
        self.src, self.quant = src, exl3.Quant(src["quant"])
        self.calibration, self.quantized = None, []

    def same(self, a, b):
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    def install(self, monkeypatch):
        import contextlib
        import types

        import numpy as np
        import safetensors.numpy

        import convert
        import reconstruction
        import replay

        stub = self
        dtypes = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16, "I64": np.int64}

        class Handle:
            def __init__(self, path):
                self.path, self.header = path, exl3.read_header(path)

            def get_tensor(self, name):
                t = self.header[name]
                with open(self.path, "rb") as f:
                    f.seek(t.offset)
                    data = f.read(t.size)
                return StubTensor(np.frombuffer(data, dtype=dtypes[t.dtype]).reshape(t.shape))

        def safe_open(path, framework, device):
            assert (framework, device) == ("pt", "cpu")
            return Handle(path)

        class LinearFP16:
            def swap_cpu(self):
                pass

        class LinearEXL3:
            def __init__(self, tensors):
                self.tensors = tensors

            def swap_cpu(self):
                pass

        class Linear:
            def __init__(self, key):
                self.key, self.inner = key, LinearFP16()

            def convert_exl3(self, H_data, quant_args, verbose, save_reg, override_swap_device):
                m, k2 = stub.quant.matrix(self.key), int(quant_args["K"] * 2)
                fill = int(hashlib.sha256(self.key.encode()).hexdigest()[:4], 16) & 0x7FFF
                self.inner = LinearEXL3({
                    f"{self.key}.trellis": StubTensor(np.full((m.columns // 16, m.rows // 16, k2 * 8), fill, np.int16)),
                    f"{self.key}.suh": StubTensor(np.ones(m.columns, np.float16)),
                    f"{self.key}.svh": StubTensor(np.ones(m.rows, np.float16)),
                    f"{self.key}.mul1": StubTensor(np.array(exl3.MUL1, dtype=np.uint32).view(np.int32))})
                quant_args.update(g_scale=1.0 + fill / (1 << 16), q_fallback=False)
                stub.quantized.append((self.key, quant_args["seed"], quant_args["K"]))
                return fill / (1 << 20)

            def get_tensors(self):
                return dict(self.inner.tensors)

        class Module:
            def __init__(self, key, linears):
                self.key, self.linears, self.num_slices, self.caps = key, linears, 1, {}

            def __iter__(self):
                return iter(self.linears)

            def can_defer_load(self):
                return False

            def load(self, device, source=None, keep_source_weights=False):
                pass

            def unload(self):
                pass

            def prepare_for_device(self, x, params):
                return x

            def forward(self, x, params):
                return x

        layers = {}
        for key in self.quant.keys():
            if key.startswith(PREFIX):
                layers.setdefault(int(key[len(PREFIX):].split(".")[0]), []).append(Linear(key))
        modules = [Module("model.language_model.embed_tokens", [])]
        modules += [Module(f"{PREFIX}{i}", layers[i]) for i in range(max(layers) + 1)]

        class Collection:
            def __init__(self):
                self.tensor_file_map = {}

            def add_tensor_files(self, root):
                with open(os.path.join(root, "model.safetensors.index.json"), encoding="utf-8") as f:
                    weight_map = json.load(f)["weight_map"]
                self.tensor_file_map.update({n: os.path.join(root, s) for n, s in weight_map.items()})

            def begin_deferred_load(self):
                pass

            end_deferred_load = close = begin_deferred_load

            def set_new_tensors(self, tensors):
                pass

        def from_directory(root):
            config = types.SimpleNamespace(stc=Collection())
            config.stc.add_tensor_files(root)
            return config

        def get_file_calibration(args, tokenizer):
            stub.calibration = dict(args)
            return [StubTensor(np.zeros((1, args["cal_cols"]), np.int64)) for _ in range(args["cal_rows"])]

        def make_quant_args(qargs, idx, bits, devices):
            return {"seed": idx, "K": bits, "apply_out_scales": qargs["apply_out_scales"], "devices": devices}

        ex = types.ModuleType("exllamav3")
        ex.Config = types.SimpleNamespace(from_directory=from_directory)
        ex.Model = types.SimpleNamespace(from_config=lambda config: types.SimpleNamespace(
            modules=modules, calibration_all_experts=False))
        ex.Tokenizer = types.SimpleNamespace(from_config=lambda config: None)
        calibration = types.ModuleType("exllamav3.conversion.calibration_data")
        calibration.get_file_calibration = get_file_calibration
        cm = types.ModuleType("exllamav3.conversion.convert_model")
        cm.check_bad_rows = lambda bad_rows, rows: None
        cm.get_H_data = lambda qargs, linear, capture_H, state: (None, None)
        cm.make_quant_args = make_quant_args
        cm.num_ref_states = 2
        mods = types.ModuleType("exllamav3.modules")
        mods.Linear = Linear
        quant = types.ModuleType("exllamav3.modules.quant")
        quant.LinearEXL3, quant.LinearFP16 = LinearEXL3, LinearFP16
        quantize = types.ModuleType("exllamav3.modules.quant.exl3_lib.quantize")
        quantize.get_temp_buffers = types.SimpleNamespace(cache_clear=lambda: None)
        memory = types.ModuleType("exllamav3.util.memory")
        memory.free_mem = lambda: None
        torch = types.ModuleType("torch")
        torch.set_grad_enabled = lambda on: None
        torch.device = lambda index: f"cuda:{index}"
        torch.inference_mode = contextlib.nullcontext
        torch.isfinite = lambda t: StubTensor(np.isfinite(t.a))
        torch.cuda = types.SimpleNamespace(max_memory_allocated=lambda device: self.PEAKS["allocated"],
                                           max_memory_reserved=lambda device: self.PEAKS["reserved"],
                                           get_device_name=lambda device: "a test GPU")
        st = types.ModuleType("safetensors")
        st.safe_open, st.numpy = safe_open, safetensors.numpy
        for name, module in {"torch": torch, "safetensors": st, "exllamav3": ex,
                             "exllamav3.conversion": types.ModuleType("exllamav3.conversion"),
                             "exllamav3.conversion.calibration_data": calibration,
                             "exllamav3.conversion.convert_model": cm, "exllamav3.modules": mods,
                             "exllamav3.modules.quant": quant,
                             "exllamav3.modules.quant.exl3_lib": types.ModuleType("exllamav3.modules.quant.exl3_lib"),
                             "exllamav3.modules.quant.exl3_lib.quantize": quantize,
                             "exllamav3.util": types.ModuleType("exllamav3.util"),
                             "exllamav3.util.memory": memory}.items():
            monkeypatch.setitem(sys.modules, name, module)

        def rel_errors(q, reference, objects, device="cuda", log=print, override=None):
            source = "quant" if stub.same(q.root, stub.src["quant"]) else "replay"
            against = "abliterated" if stub.same(reference, stub.src["abliterated"]) else "base"
            offset = {("replay", "abliterated"): 0.3, ("quant", "base"): 0.2, ("quant", "abliterated"): 0.4}
            return {p.name: round(offset[source, against] + i / 1000, 6) for i, p in enumerate(objects)}

        monkeypatch.setattr(reconstruction, "rel_errors", rel_errors)
        monkeypatch.setattr(convert, "versions", lambda: {"python": "3.11.9", "torch": "2.13.0", "exllamav3": "1.5.3",
                                                          "safetensors": "0.6.2"})
        monkeypatch.setattr(convert, "tool_commit", lambda: {"tool": "tools/27b-16gb-converter", "commit": "0" * 40,
                                                             "dirty": False})
        monkeypatch.setattr(replay, "FIRST_LAYER", REPLAY_LAYERS[0])
        monkeypatch.setattr(replay, "LAST_LAYER", REPLAY_LAYERS[1])
        return self


# What the replay and the conversion wrote at 010f221d, before a local quant (#703), on the reduced
# downloaded sources of `make_replay_sources` with `ReplayExllamav3`: the base commit's converter
# (`git archive 010f221d tools/27b-16gb-converter`) with this conftest beside it, `replay.main` on
# `replay_argv`, then `convert.convert` taking that override (weights id trellis-sc30-abliterated).
# `replay.json` is the record; `converter-override.json` the conversion record's `override` section.
FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "010f221d")


def frozen(name):
    with open(os.path.join(FROZEN, name), encoding="utf-8") as f:
        return json.load(f)


def replay_argv(src, out, *extra):
    """`replay.py run`'s arguments for the sources `src` of `make_replay_sources`, into `out`."""
    return ["run", "--quant", src["quant"], "--bf16", src["bf16"], "--abliterated", src["abliterated"],
            "--out", out, *extra]


def replayed(tmp_path, monkeypatch, local=False, name="src"):
    """`replay.py run` on the reduced sources (downloaded, or `local` at its recipe with its pins and trace),
    on the CPU: (sources, stub, exit code, the records directory)."""
    import replay
    src = make_replay_sources(str(tmp_path / name), local=local)
    stub = ReplayExllamav3(src).install(monkeypatch)
    out = str(tmp_path / name / "override")
    extra = ("--recipe", src["recipe"], "--quant-pins", src["pins"], "--trace", src["trace"]) if local else ()
    code = replay.main(replay_argv(src, out, *extra), recipes=src["recipes"])
    return src, stub, code, out


def file_sha256(path):
    """The sha256 of the file `path`, read whole: the tests' own, not the converter's `work.file_digest`."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def edit_json(path, change):
    """Rewrites the JSON file `path` after `change` (a function of its value) edits the value in place."""
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    change(value)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f)


def flip_last_byte(path):
    """Changes the last byte of the file `path`: the same size, other bytes."""
    with open(path, "r+b") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
        f.seek(-1, os.SEEK_END)
        f.write(bytes([last[0] ^ 1]))
