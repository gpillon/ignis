"""#469's comparison of a converter-built artifact with the served one (`compare.py`), on the synthetic sources."""
import collections
import json
import os
import shutil
import struct
import sys

import pytest

import checkpoint
import compare
import container
import convert
import inventory
import recipe
from conftest import ALL_FILES, G, bf16_parent, checkpoint_tensors, corrupt, decode_nvfp4, make_checkpoint

NINFER = "F:/ai/q38/ninfer"


def test_the_tiers_partition_the_inventory_by_the_recipes_routes():
    """Tier 1 is every object neither calibrated nor taken from the third-party quant (831), tier 2 the 135
    parents both recipes encode locally and their divisors, tier 3 the 112 the served recipe took from the
    quant (the MLP of layers 0-55) and their divisors."""
    tier = compare.tiers(recipe.served_recipes(G), recipe.recipes(G))
    assert collections.Counter(tier.values()) == {1: 831, 2: 270, 3: 224}
    routes = collections.Counter(recipe.recipes(G)[n].route for n, t in tier.items() if t == 1)
    assert routes == {"resource": 6, "base_direct": 401, "bf16_exception": 9, "official": 349, "dflash2": 66}
    third_party = {n for n, t in tier.items() if t == 3 and not n.endswith("/input_scale_divisor")}
    assert third_party == {f"text/layers/{i}/mlp/{m}" for i in range(56) for m in ("gate_up", "down")}


def test_rebuilding_tier_1_from_the_sources_names_an_object_that_differs(built, packed, tmp_path):
    """The machine-local test's machinery at the fixture's widths: the 831 tier-1 objects the recipe makes of
    the sources are the packed artifact's, and a copy with one byte of a vision matrix flipped is caught by
    that object's name."""
    sources = checkpoint.check_sources(G, built["checkpoint"], None, built["drafter"], built["ranking"])
    ids, _ = checkpoint.draft_ids(built["ranking"], built["checkpoint"], G)
    wanted = set(compare.tier_names(G, 1))
    done = compare.rebuild(container.Container(packed), sources, recipe.recipes(G), ids, wanted, "cpu")
    assert (done["objects"], done["differ"]) == (831, [])
    copy = corrupt(packed, tmp_path, "vision/layers/3/mlp/fc2", at=10)
    done = compare.rebuild(container.Container(copy), sources, recipe.recipes(G), ids, wanted, "cpu")
    assert done["differ"] == [{"name": "vision/layers/3/mlp/fc2", "route": "official", "bytes": 2688,
                               "differing_bytes": 1}]


@pytest.mark.skipif(not os.path.isdir(NINFER), reason=f"ninfer's checkout is not on this machine ({NINFER})")
def test_the_served_recipes_third_party_parents_are_ninfers():
    """The served artifact's record names none of the parents it took from the quant: ninfer's recipe does."""
    sys.path.insert(0, NINFER)
    try:
        from tools.convert.qwen3_8_27b import inventory_nvfp4full
    finally:
        sys.path.remove(NINFER)
    theirs = {spec.name for spec in inventory_nvfp4full.SOURCE_NVFP4_WEIGHT_SPECS}
    ours = {n for n, r in recipe.served_recipes(G).items() if r.route == "source_nvfp4"}
    assert ours == theirs and len(theirs) == 112
    local = {spec.name for spec in inventory_nvfp4full.LOCAL_NVFP4_WEIGHT_SPECS}
    assert {n for n, r in recipe.served_recipes(G).items() if r.route == "local_nvfp4"} == local


# ---------------------------------------------------------------------------------------------------
# `compare`: the packed fixture artifact against a "served" copy of it under the served recipe's record
# ---------------------------------------------------------------------------------------------------

SERVED = "qwen3_8_27b_nvfp4full-v2.ninfer"


def served_copy(packed, directory):
    """A copy of the packed artifact as the served one, beside a `.graft.json` as ninfer's graft wrote it: the
    served recipe, the artifact's bytes and objects, and only the drafter's parents recorded as encoded
    locally."""
    path = str(directory / SERVED)
    shutil.copyfile(packed, path)
    drafter = [o.name for o in inventory.objects(G) if o.unit == "dflash2" and o.format == inventory.NVFP4]
    sidecar = {"recipe_id": "qwen3_8_27b_nvfp4full-v2", "artifact": {"path": SERVED, "bytes": os.path.getsize(path)},
               "objects": {"count": 1325, "grafted": 66},
               "local_nvfp4": {"encoder_profile": "NVFP4_MAXABS_DIVISOR_RNE_V1",
                               "parents": {n: {"weight_scale_divisor": None} for n in drafter}}}
    with open(path + ".graft.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sidecar, f)
    return path


def rewrite_sidecar(served, **fields):
    with open(served + ".graft.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    with open(served + ".graft.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(sidecar, **fields), f)


def patch(path, name, at, data):
    """Writes `data` at byte `at` of `name`'s payload."""
    offset, _ = container.Container(path).span(name)
    with open(path, "r+b") as f:
        f.seek(offset + at)
        f.write(data)


def flip(path, name, at, masks):
    """XORs the bytes of `name`'s payload from byte `at` with `masks`."""
    stored = container.Container(path).read(name)
    patch(path, name, at, bytes(stored[at + i] ^ m for i, m in enumerate(masks)))


def word(path, name, at=-4):
    """The FP32 word at byte `at` of `name`'s payload (by default its last)."""
    return struct.unpack("<f", container.Container(path).read(name)[at:][:4])[0]


def compare_args(src, served, artifact, report):
    return ["compare", "--served", served, "--artifact", artifact, "--checkpoint", src["checkpoint"],
            "--report", report, "--device", "cpu", "--geometry", "fixture"]


def run_compare(src, served, artifact, tmp_path, capsys, expect=0):
    """Runs `compare`, its report in a directory that does not exist yet; returns the JSON report and the
    summary, which it also printed."""
    report = str(tmp_path / "reports" / "compare.json")
    assert convert.main(compare_args(src, served, artifact, report)) == expect
    with open(report, encoding="utf-8") as f:
        doc = json.load(f)
    with open(str(tmp_path / "reports" / "compare.txt"), encoding="utf-8") as f:
        summary = f.read()
    assert summary in capsys.readouterr().out
    return doc, summary


def error_against_bf16(src, artifact, name):
    """The oracle: a parent's values decoded from `artifact` against its BF16 rows, in binary64."""
    o = {x.name: x for x in inventory.objects(G)}[name]
    values, _ = decode_nvfp4(container.Container(artifact).read(name), *o.shape)
    parent = bf16_parent(checkpoint_tensors(src["checkpoint"]), name).double()
    return ((values.double() - parent).norm() / parent.norm()).item()


def test_compare_passes_tier_1_and_lists_the_third_party_parents(built, packed, tmp_path, capsys):
    """Tier 1 and tier 2 byte-equal; tier 3 exactly the parents the served recipe took from the quant, each
    one's reconstruction error against the BF16 on both sides (the same bytes here, so the same error)."""
    served = served_copy(packed, tmp_path)
    doc, summary = run_compare(built, served, packed, tmp_path, capsys)
    assert doc["verdict"] == "TIER 1 PASS" and summary.strip().endswith("TIER 1 PASS")
    assert {k: doc["tier1"][k] for k in ("objects", "byte_equal", "differ")} == \
        {"objects": 831, "byte_equal": 831, "differ": []}
    assert (doc["tier2"]["parents"], doc["tier2"]["parents_byte_equal"], doc["tier2"]["divisors_byte_equal"]) == \
        (135, 135, 135)
    third = doc["tier3"]["entries"]
    assert [e["name"] for e in third] == [f"text/layers/{i}/mlp/{m}" for i in range(56) for m in ("gate_up", "down")]
    for e in third:
        error = error_against_bf16(built, packed, e["name"])
        assert e["relative_frobenius_error"]["served"] == pytest.approx(error, rel=1e-9), e["name"]
        assert e["relative_frobenius_error"]["scratch"] == pytest.approx(error, rel=1e-9), e["name"]
        assert e["input_divisor"]["name"] == inventory.divisor_name(e["name"])
        assert e["input_divisor"]["relative_difference"] == 0.0
    assert (doc["served"]["recipe_id"], doc["artifact"]["recipe_id"]) == \
        ("qwen3_8_27b_nvfp4full-v2", "qwen3_8_27b_nvfp4full-v2-local")


def test_a_tier_1_difference_is_named_and_fails_the_comparison(built, packed, tmp_path, capsys):
    served = served_copy(packed, tmp_path)
    flip(served, "text/layers/7/input_norm", 6, [0xFF])
    flip(served, "dflash2/layers/2/mlp/down", 1000, [0x01, 0x10])
    doc, summary = run_compare(built, served, packed, tmp_path, capsys, expect=compare.EXIT_TIER1_FAILED)
    assert doc["verdict"] == "TIER 1 FAILED" and summary.strip().endswith("TIER 1 FAILED")
    assert doc["tier1"]["differ"] == [
        {"name": "text/layers/7/input_norm", "route": "base_direct", "bytes": 256, "differing_bytes": 1},
        {"name": "dflash2/layers/2/mlp/down", "route": "dflash2", "bytes": 4612, "differing_bytes": 2}]
    assert doc["tier1"]["byte_equal"] == 829
    assert "text/layers/7/input_norm" in summary and "dflash2/layers/2/mlp/down" in summary


def test_a_tier_2_parent_that_differs_is_measured(built, packed, tmp_path, capsys):
    """One of the 135 parents perturbed in the served copy: three codes (a byte's low nibble, another byte's
    two), one scale, the weight divisor doubled; and its activation divisor doubled."""
    name = "text/layers/9/gdn/query_key_value_z"
    site = inventory.divisor_name(name)
    n, k = {x.name: x for x in inventory.objects(G)}[name].shape
    scale_offset, divisor_offset, _ = inventory.block_scale_geometry((n, k))
    served = served_copy(packed, tmp_path)
    flip(served, name, 0, [0x01, 0x11])
    scale = container.Container(served).read(name)[scale_offset + 5]
    patch(served, name, scale_offset + 5, bytes([scale - 1 if scale & 0x7F else scale + 1]))
    d_w, d_x = word(packed, name), word(packed, site, 0)
    patch(served, name, divisor_offset, struct.pack("<f", 2 * d_w))
    patch(served, site, 0, struct.pack("<f", 2 * d_x))

    doc, summary = run_compare(built, served, packed, tmp_path, capsys)
    assert doc["verdict"] == "TIER 1 PASS"
    assert (doc["tier2"]["parents_byte_equal"], doc["tier2"]["divisors_byte_equal"]) == (134, 134)
    (e,) = [e for e in doc["tier2"]["entries"] if not e["byte_equal"]]
    assert e["name"] == name
    assert e["differing_words"] == {"codes": 3, "codes_of": n * k, "scales": 1, "scales_of": n * k // 16,
                                    "weight_divisor": 1, "padding_bytes": 0}
    assert e["weight_divisor"] == {"served": 2 * d_w, "scratch": d_w, "relative_difference": -0.5}
    assert e["relative_frobenius_error"]["served"] == pytest.approx(error_against_bf16(built, served, name), rel=1e-9)
    assert e["relative_frobenius_error"]["scratch"] == pytest.approx(error_against_bf16(built, packed, name), rel=1e-9)
    assert e["input_divisor"] == {"name": site, "byte_equal": False, "served": 2 * d_x, "scratch": d_x,
                                  "relative_difference": -0.5}
    assert name in summary and site in summary


def test_compare_keeps_one_shard_open_at_a_time(built, packed, handles, tmp_path, capsys):
    run_compare(built, served_copy(packed, tmp_path), packed, tmp_path, capsys)
    assert handles["peak"] == 1 and handles["open"] == 0
    assert set(handles["opened"]) == ALL_FILES - {"model.safetensors"}


def test_a_sidecar_of_an_unknown_recipe_is_refused(built, packed, tmp_path, capsys):
    served = served_copy(packed, tmp_path)
    rewrite_sidecar(served, recipe_id="qwen3_8_27b_nvfp4full-v1")
    report = str(tmp_path / "compare.json")
    assert convert.main(compare_args(built, served, packed, report)) == convert.EXIT_REFUSED
    assert "recipe qwen3_8_27b_nvfp4full-v1, which compare does not know" in capsys.readouterr().out
    assert not os.path.exists(report)


def test_a_sidecar_of_another_artifact_is_refused(built, packed, tmp_path, capsys):
    served = served_copy(packed, tmp_path)
    rewrite_sidecar(served, artifact={"bytes": 12345})
    assert convert.main(compare_args(built, served, packed, str(tmp_path / "c.json"))) == convert.EXIT_REFUSED
    assert "its sidecar records 12345 bytes" in capsys.readouterr().out


def test_a_checkpoint_the_artifact_was_not_converted_from_is_refused(built, packed, tmp_path, capsys):
    other = dict(built, checkpoint=make_checkpoint(str(tmp_path / "ckpt"), seed=1))
    served = served_copy(packed, tmp_path)
    assert convert.main(compare_args(other, served, packed, str(tmp_path / "c.json"))) == convert.EXIT_REFUSED
    assert "is not the one the artifact was converted from" in capsys.readouterr().out
