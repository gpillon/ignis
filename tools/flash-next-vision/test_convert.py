"""The vision companion's builder on a synthetic checkpoint (CPU, no network): the companion it makes, a second
run's bytes, the reads it makes of the checkpoint, and its refusals."""
import hashlib
import json
import os

import pytest
import torch

import container  # the 27B NVFP4 converter's reader, on the path `convert` sets
import convert
import encode
import inventory
from conftest import OUT, PATCH, SHARDS, VH, make_checkpoint, visual_shapes, visual_tensors, write_main

MAIN = "qwen3_8_flash_next_fixture-v2.ninfer"
COMPANION = "qwen3_8_flash_next_vision-v2.ninfer"
# Flash-Next's own recipe (GitHub #556, the owner's decision on #560's alarm): every matrix W8, the rest BF16.
MATRICES = ("patch_embedding", "attention/qkv", "attention/output", "mlp/fc1", "mlp/fc2", "merger/fc1", "merger/fc2")


class LocalHub:
    """The hub's reads answered from a checkpoint directory, each one recorded."""

    def __init__(self, root):
        self.root = root
        self.reads = []

    def file(self, name):
        self.reads.append((name, None, None))
        with open(os.path.join(self.root, name), "rb") as f:
            return f.read()

    def range(self, name, start, end):
        self.reads.append((name, start, end))
        with open(os.path.join(self.root, name), "rb") as f:
            f.seek(start)
            return f.read(end - start)


class OfflineHub:
    def file(self, name):
        raise AssertionError(f"read {name} with the source cached")

    def range(self, name, start, end):
        raise AssertionError(f"read {name} [{start}, {end}) with the source cached")


def expected_names():
    """The companion's objects in container order, from the 27B served tower's names."""
    names = ["vision/patch_embedding", "vision/patch_embedding_bias", "vision/position_embedding"]
    for b in range(27):
        names += [f"vision/layers/{b}/{s}" for s in (
            "attention/qkv", "attention/qkv_bias", "attention/output", "attention/output_bias", "mlp/fc1",
            "mlp/fc1_bias", "mlp/fc2", "mlp/fc2_bias", "norm1/weight", "norm1/bias", "norm2/weight", "norm2/bias")]
    return names + [f"vision/merger/{s}" for s in ("fc1", "fc1_bias", "fc2", "fc2_bias", "norm/weight", "norm/bias")]


def kind(name):
    return name.split("/", 3)[3] if name.startswith("vision/layers/") else name[len("vision/"):]


def source_name(name):
    """The checkpoint tensor an object is made of (the 27B recipe's routes, restated)."""
    v = "model.visual."
    fixed = {"vision/patch_embedding": "patch_embed.proj.weight", "vision/patch_embedding_bias": "patch_embed.proj.bias",
             "vision/position_embedding": "pos_embed.weight", "vision/merger/fc1": "merger.linear_fc1.weight",
             "vision/merger/fc1_bias": "merger.linear_fc1.bias", "vision/merger/fc2": "merger.linear_fc2.weight",
             "vision/merger/fc2_bias": "merger.linear_fc2.bias", "vision/merger/norm/weight": "merger.norm.weight",
             "vision/merger/norm/bias": "merger.norm.bias"}
    if name in fixed:
        return v + fixed[name]
    _, _, b, rest = name.split("/", 3)
    src = {"attention/qkv": "attn.qkv.weight", "attention/qkv_bias": "attn.qkv.bias", "attention/output": "attn.proj.weight",
           "attention/output_bias": "attn.proj.bias", "mlp/fc1": "mlp.linear_fc1.weight",
           "mlp/fc1_bias": "mlp.linear_fc1.bias", "mlp/fc2": "mlp.linear_fc2.weight", "mlp/fc2_bias": "mlp.linear_fc2.bias",
           "norm1/weight": "norm1.weight", "norm1/bias": "norm1.bias", "norm2/weight": "norm2.weight",
           "norm2/bias": "norm2.bias"}[rest]
    return f"{v}blocks.{b}.{src}"


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def argv(tmp, run, exe, source="source"):
    return ["build", "--geometry", "fixture", "--main", str(tmp / "main" / MAIN), "--work", str(tmp / run / "work"),
            "--source", str(tmp / source), "--out", str(tmp / run / COMPANION), "--packer", exe]


@pytest.fixture
def sources(tmp_path):
    """(checkpoint dir, its visual tensors, the stand-in main's sha256)."""
    tensors = make_checkpoint(str(tmp_path / "checkpoint"))
    os.makedirs(tmp_path / "main")
    return str(tmp_path / "checkpoint"), tensors, write_main(str(tmp_path / "main" / MAIN))


def test_one_build_makes_the_companion_of_the_towers_objects_with_every_matrix_in_w8(tmp_path, sources, pack_exe):
    checkpoint, tensors, main_sha = sources
    hub = LocalHub(checkpoint)
    assert convert.main(argv(tmp_path, "run", pack_exe), hub=hub) == 0
    out = tmp_path / "run" / COMPANION
    c = container.Container(out)
    assert c.identity == {"model_id": "qwen3.8-flash-next-vision", "weights_id": "vision-w8-bf16-de4b8e4"}
    assert [o["name"] for o in c.objects] == expected_names()
    for o in c.objects:
        name = o["name"]
        assert o["format"] == ("W8G32_F16S" if kind(name) in MATRICES else "BF16"), name
        source = tensors[source_name(name)]
        if o["format"] == "BF16":
            assert c.read(name) == source.contiguous().view(torch.uint8).numpy().tobytes(), name
        else:
            matrix = source.reshape(VH, -1) if name == "vision/patch_embedding" else source
            assert c.read(name) == encode.grouped(matrix, o["format"], "cpu"), name
    assert c.by_name["vision/merger/fc2"]["shape"] == [OUT, 4 * VH]
    assert c.by_name["vision/merger/fc2_bias"]["shape"] == [OUT]
    assert c.by_name["vision/patch_embedding"]["shape"] == [VH, 3 * 2 * PATCH * PATCH]

    # The sidecar: the pair record, the source range, measured values only, no local path.
    text = open(str(out) + ".conversion.json", encoding="utf-8").read()
    side = json.loads(text)
    assert side["schema"] == "flash-next-vision-converter-v1" and side["status"] == "complete"
    assert side["pair"]["main"]["file"] == MAIN
    assert side["pair"]["main"]["file_sha256"] == main_sha
    assert side["pair"]["main"]["model_id"] == "qwen3.8-flash-next"
    assert side["recipe"]["formats"] == {"BF16": 222, "W8G32_F16S": 111}
    assert side["recipe"]["payload_bytes"] == sum(o["bytes"] for o in c.objects)
    span = b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in tensors.values())
    src = side["source"]
    assert (src["repo"], src["revision"]) == (convert.REPO, convert.REVISION)
    assert (src["file"], src["tensors"], src["bytes"]) == (SHARDS[0], 333, len(span))
    assert src["sha256"] == hashlib.sha256(span).hexdigest()
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
    assert not any(word in text.lower() for word in ("verdict", "pass", "fail", "limit"))
    assert open(str(out) + ".sha256").read() == f"{sha256(out)}  {COMPANION}\n"

    # Read of the checkpoint: the index, the configuration, shard 1's header and the visual span, nothing else.
    header = int.from_bytes(open(os.path.join(checkpoint, SHARDS[0]), "rb").read(8), "little")
    ranges = sorted((s, e) for f, s, e in hub.reads if s is not None)
    assert {f for f, _, _ in hub.reads} == {"config.json", "model.safetensors.index.json", SHARDS[0]}
    assert ranges[:2] == [(0, 8), (8, 8 + header)]
    assert ranges[2][0] == src["offset"] and ranges[-1][1] == src["offset"] + len(span)
    assert all(a[1] == b[0] for a, b in zip(ranges[2:], ranges[3:])), "the span is read once, in parts"


def test_a_second_build_gives_the_same_bytes_from_the_cached_source(tmp_path, sources, pack_exe, monkeypatch):
    checkpoint, _, _ = sources
    assert convert.main(argv(tmp_path, "one", pack_exe), hub=LocalHub(checkpoint)) == 0
    # the second run names the packer as a relative path with forward slashes, as a shell does
    monkeypatch.chdir(convert.REPO_ROOT)
    relative = os.path.relpath(pack_exe, convert.REPO_ROOT).replace("\\", "/")
    assert convert.main(argv(tmp_path, "two", relative), hub=OfflineHub()) == 0
    for suffix in ("", ".conversion.json", ".sha256"):
        one, two = tmp_path / "one" / (COMPANION + suffix), tmp_path / "two" / (COMPANION + suffix)
        assert open(one, "rb").read() == open(two, "rb").read(), suffix


def test_the_tower_at_flash_nexts_widths_is_the_27b_towers_objects_every_matrix_w8_and_its_merger_2560_wide():
    ours = convert.tower_objects(convert.GEOMETRIES["qwen3.8-flash-next"])
    served = [o for o in inventory.objects(inventory.Geometry.qwen3_8_27b()) if o.unit == "vision"]
    assert len(ours) == len(served) == 333
    for o, s in zip(ours, served):
        assert o.name == s.name
        assert o.format == (inventory.BF16 if s.format == inventory.BF16 else inventory.W8), o.name
        expected = {"vision/merger/fc2": (2560, 4608), "vision/merger/fc2_bias": (2560,)}.get(o.name, s.shape)
        assert o.shape == expected, o.name
    # #560's diagnosis costed the all-W8 tower at 481,445,216 bytes of payload.
    assert sum(inventory.encoded_size(o) for o in ours) == 481_445_216


def refuse(tmp_path, exe, capsys, hub=None):
    code = convert.main(argv(tmp_path, "run", exe), hub=hub or LocalHub(str(tmp_path / "checkpoint")))
    assert code == 2
    assert not (tmp_path / "run" / COMPANION).exists()
    return capsys.readouterr().err


def edited(change):
    tensors = visual_tensors()
    change(tensors)
    return tensors


@pytest.mark.parametrize("case, kwargs, named", [
    ("missing", {"tensors": edited(lambda t: t.pop("model.visual.blocks.26.norm2.bias"))},
     "model.visual.blocks.26.norm2.bias"),
    ("extra", {"tensors": edited(lambda t: t.update({"model.visual.blocks.27.norm1.bias": torch.zeros(VH, dtype=torch.bfloat16)}))},
     "model.visual.blocks.27.norm1.bias"),
    ("shape", {"tensors": edited(lambda t: t.update({"model.visual.merger.linear_fc2.weight": torch.zeros((4 * VH, 4 * VH), dtype=torch.bfloat16)}))},
     "model.visual.merger.linear_fc2.weight"),
    ("dtype", {"tensors": edited(lambda t: t.update({"model.visual.pos_embed.weight": t["model.visual.pos_embed.weight"].float()}))},
     "model.visual.pos_embed.weight"),
    ("width", {"cfg": __import__("conftest").config(out_hidden_size=4 * VH)}, "out_hidden_size"),
    ("shards", {"visual_shard": 1}, None),
])
def test_a_checkpoint_that_is_not_flash_nexts_tower_is_refused_by_name_writing_nothing(tmp_path, pack_exe, capsys, case, kwargs, named):
    os.makedirs(tmp_path / "main")
    write_main(str(tmp_path / "main" / MAIN))
    if case == "shards":
        # the tower split over two shards: half of it in the second
        tensors = visual_tensors()
        make_checkpoint(str(tmp_path / "checkpoint"), tensors=tensors)
        index_path = tmp_path / "checkpoint" / "model.safetensors.index.json"
        index = json.load(open(index_path))
        index["weight_map"]["model.visual.pos_embed.weight"] = SHARDS[1]
        json.dump(index, open(index_path, "w"))
        named = SHARDS[1]
    else:
        make_checkpoint(str(tmp_path / "checkpoint"), **kwargs)
    err = refuse(tmp_path, pack_exe, capsys)
    assert named in err, err
    assert not (tmp_path / "run" / "work").exists() or not os.listdir(tmp_path / "run" / "work")


def test_a_cached_source_whose_bytes_changed_is_refused(tmp_path, sources, pack_exe, capsys):
    checkpoint, _, _ = sources
    assert convert.main(argv(tmp_path, "one", pack_exe), hub=LocalHub(checkpoint)) == 0
    cached = tmp_path / "source" / "visual.bin"
    data = bytearray(open(cached, "rb").read())
    data[100] ^= 1
    open(cached, "wb").write(bytes(data))
    err = refuse(tmp_path, pack_exe, capsys, hub=OfflineHub())
    assert "visual.bin" in err and "sha256" in err, err


def test_a_work_tree_that_is_not_empty_is_refused(tmp_path, sources, pack_exe, capsys):
    os.makedirs(tmp_path / "run" / "work")
    open(tmp_path / "run" / "work" / "left-over", "w").write("x")
    err = refuse(tmp_path, pack_exe, capsys)
    assert "is not empty" in err, err
    assert os.listdir(tmp_path / "run" / "work") == ["left-over"]


def test_the_pinned_revisions_tower_is_pinned_by_its_bytes():
    pin = convert.GEOMETRIES["qwen3.8-flash-next"]
    assert pin.shard == "model-00001-of-00131.safetensors"
    assert (pin.span_bytes, len(pin.span_sha256)) == (897_862_112, 64)


def test_the_cache_holds_the_checkpoint_it_was_read_from_and_refuses_another(tmp_path, sources):
    # GitHub #560: the 27B's tower is cached the same way, as the bar Flash-Next's is checked against.
    checkpoint, tensors, _ = sources
    other = ("Qwen/Another", "0123456789abcdef0123456789abcdef01234567")
    data, record = convert.fetch(LocalHub(checkpoint), str(tmp_path / "source"), convert.GEOMETRIES["fixture"], *other)
    assert (record["repo"], record["revision"]) == other
    assert len(data) == sum(t.numel() * 2 for t in tensors.values())
    _, again = convert.fetch(OfflineHub(), str(tmp_path / "source"), convert.GEOMETRIES["fixture"], *other)
    assert again == record
    with pytest.raises(convert.Refused, match="caches Qwen/Another@0123456789abcdef"):
        convert.fetch(OfflineHub(), str(tmp_path / "source"), convert.GEOMETRIES["fixture"])
