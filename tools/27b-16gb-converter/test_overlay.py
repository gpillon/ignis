import hashlib
import json
import os
import random

import pytest

import overlay
from conftest import write_safetensors

PREFIX = "model.language_model.layers."
DOWN = PREFIX + "17.mlp.down_proj.weight"
OUT = PREFIX + "17.linear_attn.out_proj.weight"
NORM = PREFIX + "17.input_layernorm.weight"
UP = PREFIX + "18.mlp.up_proj.weight"
# The overlay's partial shard has the base shard's file name, as `replay.py fetch` writes it.
SHARD_A, SHARD_B = "model-00006-of-00018.safetensors", "model-00007-of-00018.safetensors"


def _bytes(rng, n):
    return bytes(rng.getrandbits(8) for _ in range(n))


def _base(root, seed=386):
    """A base BF16 checkpoint: two shards and their index."""
    rng = random.Random(seed)
    base = os.path.join(root, "base")
    os.makedirs(base)
    shards = {SHARD_A: {DOWN: ("BF16", [8, 16], _bytes(rng, 256)), NORM: ("BF16", [8], _bytes(rng, 16)),
                        OUT: ("BF16", [8, 4], _bytes(rng, 64))},
              SHARD_B: {UP: ("BF16", [16, 8], _bytes(rng, 256))}}
    weight_map = {}
    for shard, tensors in shards.items():
        write_safetensors(os.path.join(base, shard), tensors)
        weight_map.update({n: shard for n in tensors})
    with open(os.path.join(base, overlay.INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": weight_map}, f)
    return base


def _overlay(root, tensors=None, seed=422, change=None):
    """What `replay.py fetch` writes: the changed tensors in a shard of the base shard's name, an
    index and `fetch.json`; `change(record)` edits `fetch.json` before it is written."""
    rng = random.Random(seed)
    if tensors is None:
        tensors = {DOWN: ("BF16", [8, 16], _bytes(rng, 256)), OUT: ("BF16", [8, 4], _bytes(rng, 64))}
    d = os.path.join(root, "overlay")
    os.makedirs(d)
    write_safetensors(os.path.join(d, SHARD_A), tensors)
    with open(os.path.join(d, overlay.INDEX), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"weight_map": {n: SHARD_A for n in tensors}}, f)
    record = {"repo": "someone/abliterated", "revision": "f" * 40,
              "tensors": {n: {"shard": SHARD_A, "bytes": len(t[2]), "dtype": t[0], "shape": t[1],
                              "sha256": hashlib.sha256(t[2]).hexdigest()} for n, t in tensors.items()}}
    if change:
        change(record)
    with open(os.path.join(d, overlay.FETCH_RECORD), "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f)
    return d, tensors


def _read(path, name):
    """A tensor's bytes, read from its safetensors file."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        start, end = json.loads(f.read(n))[name]["data_offsets"]
        f.seek(8 + n + start)
        return f.read(end - start)


def test_the_overlays_tensors_are_read_from_it_and_every_other_from_the_base(tmp_path):
    base = _base(str(tmp_path))
    d, tensors = _overlay(str(tmp_path))
    ref = overlay.Reference(base, d)
    assert ref.replaced == sorted(tensors)
    assert (ref.repo, ref.revision) == ("someone/abliterated", "f" * 40)
    for name, (_, _, data) in tensors.items():
        assert _read(ref.files[name], name) == data
    # the base's own tensors stay the base's, the one in the shard the overlay's shares a name with too
    for name, shard in ((NORM, SHARD_A), (UP, SHARD_B)):
        assert ref.files[name] == os.path.join(base, shard)
    assert sorted(ref.files) == sorted([DOWN, OUT, NORM, UP])


def test_without_an_overlay_every_tensor_is_the_bases(tmp_path):
    base = _base(str(tmp_path))
    ref = overlay.Reference(base)
    assert ref.replaced == [] and ref.repo is None
    with open(os.path.join(base, overlay.INDEX), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    assert ref.files == {n: os.path.join(base, s) for n, s in weight_map.items()}


def _swap(name, value):
    def change(record):
        record["tensors"][name][value[0]] = value[1]
    return change


@pytest.mark.parametrize("make,message", [
    # a tensor the base does not have: the overlay cannot swap it like for like
    (lambda r: _overlay(r, tensors={PREFIX + "99.mlp.down_proj.weight": ("BF16", [8, 16], bytes(256))}),
     r"layers.99.mlp.down_proj.weight: the base has no tensor of that name"),
    (lambda r: _overlay(r, tensors={DOWN: ("BF16", [16, 8], bytes(256))}),
     r"layers.17.mlp.down_proj.weight: BF16 \[16, 8\], the base's is BF16 \[8, 16\]"),
    (lambda r: _overlay(r, tensors={DOWN: ("F16", [8, 16], bytes(256))}),
     r"layers.17.mlp.down_proj.weight: F16 \[8, 16\], the base's is BF16 \[8, 16\]"),
    # a tensor whose bytes are not the ones the fetch recorded
    (lambda r: _overlay(r, change=_swap(DOWN, ("sha256", "0" * 64))),
     r"layers.17.mlp.down_proj.weight: hashes to [0-9a-f]{64}, fetch.json has 0{64}"),
    # fetch.json and the index disagree on what the overlay holds
    (lambda r: _overlay(r, change=lambda rec: rec["tensors"].pop(OUT)),
     r"layers.17.linear_attn.out_proj.weight: the index maps it, fetch.json does not name it"),
    (lambda r: _overlay(r, change=lambda rec: rec["tensors"].__setitem__(UP, dict(rec["tensors"][DOWN]))),
     r"layers.18.mlp.up_proj.weight: fetch.json names it, the index does not map it"),
    (lambda r: _overlay(r, change=_swap(DOWN, ("shard", SHARD_B))),
     r"layers.17.mlp.down_proj.weight: the index has it in model-00006-of-00018.safetensors, "
     r"fetch.json in model-00007-of-00018.safetensors"),
])
def test_an_overlay_that_is_not_the_fetched_like_for_like_swap_is_refused(tmp_path, make, message):
    base = _base(str(tmp_path))
    d, _ = make(str(tmp_path))
    with pytest.raises(ValueError, match=message):
        overlay.Reference(base, d)


def test_an_overlay_with_no_fetch_record_is_refused(tmp_path):
    base = _base(str(tmp_path))
    d, _ = _overlay(str(tmp_path))
    os.remove(os.path.join(d, overlay.FETCH_RECORD))
    with pytest.raises(ValueError, match="has no fetch.json: its tensors cannot be pinned"):
        overlay.Reference(base, d)
