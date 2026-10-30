import hashlib
import json
import os
import struct

import pytest

import exl3
import replay
from conftest import make_sources, write_safetensors

P = "model.language_model.layers."


def _linear(tensors, key, rows, columns, k2):
    tensors[f"{key}.trellis"] = ("I16", [columns // 16, rows // 16, k2 * 8], bytes(rows * columns // 16 * k2))
    tensors[f"{key}.suh"] = ("F16", [columns], bytes(2 * columns))
    tensors[f"{key}.svh"] = ("F16", [rows], bytes(2 * rows))
    tensors[f"{key}.mul1"] = ("I32", [], struct.pack("<I", exl3.MUL1))


def _quant(root, tensors):
    os.makedirs(root, exist_ok=True)
    write_safetensors(os.path.join(root, "model-00001-of-00001.safetensors"), tensors)
    with open(os.path.join(root, "model.safetensors.index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": {n: "model-00001-of-00001.safetensors" for n in tensors}}, f)
    with open(os.path.join(root, "quantization_config.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"quant_method": "exl3", "codebook": "mul1"}, f)
    return exl3.Quant(root)


def _quant_27b_layers(root, k2=None):
    """The 27B's 64 layers (full attention at every fourth, from layer 3), reduced to each layer's
    output projection and down projection; `k2` maps a key to its 2·K (default 6, K 3)."""
    k2 = k2 or {}
    tensors = {}
    for layer in range(64):
        out = "self_attn.o_proj" if (layer + 1) % 4 == 0 else "linear_attn.out_proj"
        for module, rows, columns in ((out, 128, 128), ("mlp.down_proj", 128, 256)):
            key = f"{P}{layer}.{module}"
            _linear(tensors, key, rows, columns, k2.get(key, 6))
    return _quant(root, tensors)


def test_the_plan_is_each_layers_output_projection_and_down_projection_at_the_quants_k(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    # The synthetic quant: layers 1-2 GDN (out_proj K4), layer 3 GQA (o_proj K4), down_proj K3.
    assert [(m.key, m.layer, m.seed, m.k2, m.shape) for m in planned] == [
        (P + "1.linear_attn.out_proj", 1, 2, 8, [128, 128]),
        (P + "1.mlp.down_proj", 1, 2, 6, [128, 256]),
        (P + "2.linear_attn.out_proj", 2, 3, 8, [128, 128]),
        (P + "2.mlp.down_proj", 2, 3, 6, [128, 256]),
        (P + "3.self_attn.o_proj", 3, 4, 8, [128, 128]),
        (P + "3.mlp.down_proj", 3, 4, 6, [128, 256]),
    ]


def test_on_the_27b_the_plan_is_the_70_matrices_of_layers_17_to_51(tmp_path):
    quant = _quant_27b_layers(str(tmp_path), k2={P + "23.self_attn.o_proj": 8})
    planned = replay.plan(quant)
    assert len(planned) == 70
    assert sorted({m.layer for m in planned}) == list(range(17, 52))
    attention = [m.layer for m in planned if m.key.endswith(".self_attn.o_proj")]
    assert attention == [19, 23, 27, 31, 35, 39, 43, 47, 51]
    assert sum(m.key.endswith(".linear_attn.out_proj") for m in planned) == 26
    assert sum(m.key.endswith(".mlp.down_proj") for m in planned) == 35
    assert all(m.seed == m.layer + 1 for m in planned)
    assert {m.key: m.k2 / 2 for m in planned if m.k2 != 6} == {P + "23.self_attn.o_proj": 4}


@pytest.mark.parametrize("change,message", [
    ("neither", "layer 20 has no output projection"),
    ("both", "layer 20 has two output projections"),
    ("no down", "no model.language_model.layers.20.mlp.down_proj"),
])
def test_a_layer_the_plan_cannot_read_is_refused(tmp_path, change, message):
    tensors = {}
    for layer in (19, 20):
        out = "self_attn.o_proj" if layer == 19 else "linear_attn.out_proj"
        _linear(tensors, f"{P}{layer}.{out}", 128, 128, 6)
        _linear(tensors, f"{P}{layer}.mlp.down_proj", 128, 256, 6)
    if change == "neither":
        tensors = {n: v for n, v in tensors.items() if ".20.linear_attn" not in n}
    elif change == "both":
        _linear(tensors, f"{P}20.self_attn.o_proj", 128, 128, 6)
    else:
        tensors = {n: v for n, v in tensors.items() if ".20.mlp" not in n}
    with pytest.raises(ValueError, match=message):
        replay.plan(_quant(str(tmp_path), tensors), first=19, last=20)


class Remote:
    """A model repository at one revision, served the way `replay.fetch` reads Hugging Face: the
    index whole, the file listing, shard bytes by range. `cut` shortens one tensor's range."""

    def __init__(self, root, shards, cut=None):
        self.root, self.cut, self.requests = root, cut, []
        os.makedirs(root)
        weight_map = {}
        for shard, tensors in shards.items():
            write_safetensors(os.path.join(root, shard), tensors)
            weight_map.update({n: shard for n in tensors})
        self.headers = {shard: exl3.read_header(os.path.join(root, shard)) for shard in shards}
        self.index = json.dumps({"metadata": {}, "weight_map": weight_map}).encode()
        self.tree = json.dumps([{"path": s, "size": os.path.getsize(os.path.join(root, s)),
                                 "lfs": {"oid": f"oid-{s}"}} for s in shards]).encode()

    def get(self, url, start=None, end=None):
        self.requests.append((url, start, end))
        if url == f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}":
            return self.tree
        head = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
        assert url.startswith(head), url
        name = url[len(head):]
        if name == "model.safetensors.index.json":
            return self.index
        with open(os.path.join(self.root, name), "rb") as f:
            data = f.read()[start:end]
        cut = self.cut and self.headers[name].get(self.cut)
        return data[:-1] if cut and start == cut.offset else data


REPO, REVISION = "someone/abliterated", "f" * 40


def _bf16(rows, columns, fill):
    return ("BF16", [rows, columns], bytes([fill]) * 2 * rows * columns)


def _remote_shards():
    return {
        "model-00001-of-00002.safetensors": {P + "1.linear_attn.out_proj.weight": _bf16(128, 128, 1),
                                              P + "1.mlp.down_proj.weight": _bf16(128, 256, 2),
                                              P + "1.mlp.up_proj.weight": _bf16(256, 128, 3)},
        "model-00002-of-00002.safetensors": {P + "2.linear_attn.out_proj.weight": _bf16(128, 128, 4),
                                              P + "2.mlp.down_proj.weight": _bf16(128, 256, 5)},
    }


def _planned(tmp_path):
    return replay.plan(exl3.Quant(make_sources(str(tmp_path / "src"))["quant"]), first=1, last=2)


def test_fetch_takes_only_the_planned_tensors_by_range_under_their_names(tmp_path):
    shards = _remote_shards()
    remote = Remote(str(tmp_path / "remote"), shards)
    out = str(tmp_path / "abliterated")
    replay.fetch(_planned(tmp_path), out, repo=REPO, revision=REVISION, get=remote.get, log=lambda _: None)

    with open(os.path.join(out, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    wanted = {n: s for s, tensors in shards.items() for n in tensors if not n.endswith("up_proj.weight")}
    assert weight_map == wanted
    for name, shard in wanted.items():
        t = exl3.read_header(os.path.join(out, shard))[name]
        with open(os.path.join(out, shard), "rb") as f:
            f.seek(t.offset)
            assert (t.dtype, t.shape, f.read(t.size)) == shards[shard][name]
    with open(os.path.join(out, "fetch.json"), encoding="utf-8") as f:
        record = json.load(f)
    assert (record["repo"], record["revision"]) == (REPO, REVISION)
    assert record["shards"]["model-00002-of-00002.safetensors"]["lfs_oid"] == "oid-model-00002-of-00002.safetensors"
    assert {n: t["sha256"] for n, t in record["tensors"].items()} == {
        n: hashlib.sha256(shards[s][n][2]).hexdigest() for n, s in wanted.items()}
    # the unplanned tensor's bytes were never requested
    up = remote.headers["model-00001-of-00002.safetensors"][P + "1.mlp.up_proj.weight"]
    assert all(start is None or end <= up.offset or start >= up.offset + up.size for _, start, end in remote.requests)


@pytest.mark.parametrize("change,message", [
    ("short", "down_proj.weight: 65535 bytes read from model-00002-of-00002.safetensors, its header says 65536"),
    ("shape", r"2\.linear_attn\.out_proj\.weight: BF16 \[128, 256\], 65536 bytes .*not BF16 \[128, 128\]"),
])
def test_fetch_refuses_a_tensor_not_of_its_headers_length_or_the_plans_shape(tmp_path, change, message):
    shards = _remote_shards()
    if change == "shape":
        shards["model-00002-of-00002.safetensors"][P + "2.linear_attn.out_proj.weight"] = _bf16(128, 256, 4)
    cut = P + "2.mlp.down_proj.weight" if change == "short" else None
    remote = Remote(str(tmp_path / "remote"), shards, cut=cut)
    with pytest.raises(ValueError, match=message):
        replay.fetch(_planned(tmp_path), str(tmp_path / "abliterated"), repo=REPO, revision=REVISION,
                     get=remote.get, log=lambda _: None)


def _records(planned, k2=None):
    """Each planned matrix's EXL3 tensors, as the GPU step hands them to `write_layer`, by layer."""
    import numpy as np
    k2 = k2 or {}
    layers = {}
    for m in planned:
        words = k2.get(m.key, m.k2) * 8
        layers.setdefault(m.layer, {}).update({
            f"{m.key}.trellis": np.full((m.columns // 16, m.rows // 16, words), m.layer, dtype=np.int16),
            f"{m.key}.suh": np.ones(m.columns, dtype=np.float16),
            f"{m.key}.svh": np.ones(m.rows, dtype=np.float16),
            f"{m.key}.mul1": np.array(exl3.MUL1, dtype=np.uint32).view(np.int32),
        })
    return layers


def _write(out, quant, layers):
    written = dict(replay.write_layer(out, layer, tensors) for layer, tensors in layers.items())
    replay.write_index(out, quant, written, version="1.5.3")


def test_the_records_are_an_exl3_directory_of_the_plans_matrices_at_the_quants_k_and_shape(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path / "src"))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    out = str(tmp_path / "records")
    _write(out, quant, _records(planned))

    records = exl3.Quant(out)
    assert records.keys() == sorted(m.key for m in planned)
    for m in planned:
        r, q = records.matrix(m.key), quant.matrix(m.key)
        assert (r.k2, r.shape) == (q.k2, q.shape)
        assert records.record(r)[:2] == bytes([m.layer, 0])
    assert records.config["quant_method"] == "exl3" and records.config["codebook"] == "mul1"
    assert records.config["version"] == "1.5.3"
    assert replay.check_records(out, planned) == []
    # the converter's override takes them like for like
    import convert
    import override
    assert override.Override(out, None, None, None).problems(convert.text_matrices(quant)[0]) == []


def test_records_not_the_plans_are_named(tmp_path):
    quant = exl3.Quant(make_sources(str(tmp_path / "src"))["quant"])
    planned = replay.plan(quant, first=1, last=3)
    layers = _records(planned, k2={P + "2.mlp.down_proj": 8})
    del layers[3][P + "3.self_attn.o_proj.trellis"]
    for s in ("suh", "svh", "mul1"):
        del layers[3][P + f"3.self_attn.o_proj.{s}"]
    out = str(tmp_path / "records")
    _write(out, quant, layers)
    assert replay.check_records(out, planned) == [
        f"{P}2.mlp.down_proj: K 4, the plan's is 3",
        f"{P}3.self_attn.o_proj: missing",
    ]
