"""The tower check's measure, its controls, its bar and its reference (CPU, no network, no GPU): what an
encoded tower's error against its BF16 tower is read as, before any figure of the real towers exists."""
import json
import os

import numpy as np
import pytest
import torch

import tower_check
from conftest import OUT, PATCH, config, visual_tensors

RNG = np.random.default_rng(560)


def bits(values):
    """float32 values rounded to BF16, as their bits (round to nearest even)."""
    u = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def test_bf16_bits_decode_to_their_values():
    assert tower_check.bf16_to_f32(np.array([0x3F80, 0xC000, 0x0000, 0x3E80], dtype=np.uint16)).tolist() == \
        [1.0, -2.0, 0.0, 0.25]


def test_rows_equal_to_the_reference_measure_no_error_and_rows_of_another_place_do():
    reference = RNG.standard_normal((64, 32)).astype(np.float32)
    m = tower_check.measures(reference.copy(), reference, centre=reference.mean(0))
    assert m["rel_l2"] == 0 and m["row_rel_l2"]["max"] == 0
    assert m["centred_cosine"]["min"] == pytest.approx(1.0, abs=1e-6)
    # The control pairs every served row with the reference row half the item away: unrelated rows.
    assert m["control_within"]["rel_l2"] == pytest.approx(np.sqrt(2), rel=0.1)
    assert abs(m["control_within"]["centred_cosine_mean"]) < 0.1


def test_a_shared_direction_makes_raw_cosine_near_one_where_the_centred_cosine_sees_unrelated_rows():
    # The method memory's trap: rows dominated by one common direction look alike to a raw cosine
    # whatever their content; centred on the set's mean they are what they are.
    common = 50 * RNG.standard_normal(32).astype(np.float32)
    reference = common + RNG.standard_normal((200, 32)).astype(np.float32)
    unrelated = common + RNG.standard_normal((200, 32)).astype(np.float32)
    m = tower_check.measures(unrelated, reference, centre=reference.mean(0))
    assert m["raw_cosine_mean"] > 0.99
    assert abs(m["centred_cosine"]["mean"]) < 0.15
    close = reference + 0.05 * RNG.standard_normal((200, 32)).astype(np.float32)
    assert tower_check.measures(close, reference, centre=reference.mean(0))["centred_cosine"]["mean"] > 0.99


def test_the_bar_is_the_27b_towers_error_on_the_same_images_paired():
    images = [f"i{k}" for k in range(12)]
    low = {i: {"rel_l2": 0.010 + 0.001 * k, "sq_err": 1.0, "sq_ref": 1e4} for k, i in enumerate(images)}
    high = {i: {"rel_l2": 0.020 + 0.001 * k, "sq_err": 4.0, "sq_ref": 1e4} for k, i in enumerate(images)}
    within = tower_check.bar(low, high, seed=1)
    assert within["images"] == 12 and within["flash_next_at_most_27b"] is True
    assert within["mean_difference"] == pytest.approx(-0.010)
    assert within["bootstrap"]["high"] < 0
    excess = tower_check.bar(high, low, seed=1)
    assert excess["flash_next_at_most_27b"] is False and excess["bootstrap"]["low"] > 0
    # Only the images both towers encoded are compared.
    assert tower_check.bar(low, {i: high[i] for i in images[:5]}, seed=1)["images"] == 5


def fixture_record(tmp_path):
    """The fixture tower's BF16 tensors as a cached range (`visual.bin`, `visual.json`), as `convert.fetch`
    leaves it, under the vision config of Flash-Next's model type."""
    tensors = visual_tensors()
    data, entries, at = bytearray(), {}, 0
    for name, t in tensors.items():
        raw = t.view(torch.int16).numpy().tobytes()
        entries[name] = {"dtype": "BF16", "shape": list(t.shape), "start": at, "end": at + len(raw)}
        data += raw
        at += len(raw)
    vision = dict(config()["vision_config"], model_type="qwen4_exp")
    record = {"repo": "fixture", "revision": "0" * 40, "file": "shard", "offset": 0, "bytes": len(data),
              "sha256": "", "config": {"vision_config": vision}, "tensors": entries}
    os.makedirs(tmp_path / "source")
    (tmp_path / "source" / "visual.bin").write_bytes(bytes(data))
    (tmp_path / "source" / "visual.json").write_text(json.dumps(record))
    return tensors


def test_the_reference_is_the_checkpoints_own_tower_on_the_items_patches(tmp_path):
    tensors = fixture_record(tmp_path)
    model = tower_check.reference_model(str(tmp_path / "source"))
    grid = (1, 4, 6)
    patches = bits(RNG.uniform(-1, 1, (24, 3 * 2 * PATCH * PATCH)))
    rows = tower_check.reference_rows(model, patches, grid)
    assert rows.shape == (6, OUT) and rows.dtype == np.float32 and np.isfinite(rows).all()
    # The rows are the merger's: a 2 x 2 window of patches is one row, in the processor's merge-block
    # order, so changing the last window's patches changes the last row alone.
    changed = patches.copy()
    changed[20:] = bits(RNG.uniform(-1, 1, (4, changed.shape[1])))
    other = tower_check.reference_rows(model, changed, grid)
    assert not np.array_equal(other[5], rows[5])
    # Every patch attends to every other, so the other rows move too, but by less than the last.
    moved = np.linalg.norm(other - rows, axis=1) / np.linalg.norm(rows, axis=1)
    assert moved[:5].max() < moved[5]
    # A tensor the model does not have is refused, not ignored.
    tensors["model.visual.extra.weight"] = torch.zeros(2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="extra"):
        tower_check.load_state(torch.nn.Module(), tensors)


def write_dump(directory, family, items, hidden):
    """A dump as the GPU test writes it: per item its patches and the served embedding, BF16 bits."""
    os.makedirs(directory, exist_ok=True)
    manifest = {"schema": tower_check.DUMP_SCHEMA, "family": family, "artifact": "a.ninfer", "residency": "host",
                "hidden": hidden, "patch_features": 3 * 2 * PATCH * PATCH, "items": []}
    for item_id, (grid, patches, served) in items.items():
        (directory / f"{item_id}.patches.bf16").write_bytes(patches.tobytes())
        (directory / f"{item_id}.embedding.bf16").write_bytes(bits(served).tobytes())
        manifest["items"].append({"id": item_id, "image": f"{item_id}.png", "image_sha256": "0" * 64,
                                  "grid": list(grid), "tokens": int(np.prod(grid)) // 4,
                                  "patches": f"{item_id}.patches.bf16", "embedding": f"{item_id}.embedding.bf16"})
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_a_dump_and_its_reference_compare_to_a_record_per_image_and_the_bar(tmp_path):
    fixture_record(tmp_path)
    model = tower_check.reference_model(str(tmp_path / "source"))
    grids = {"a": (1, 4, 4), "b": (1, 2, 8), "c": (1, 6, 2)}
    dump = tmp_path / "dump"
    noise = {"qwen3.8-flash-next": 0.001, "qwen3.8-27b": 0.004}
    for family, scale in noise.items():
        items = {}
        for item_id, grid in grids.items():
            patches = bits(np.random.default_rng(len(item_id) + grid[2]).uniform(-1, 1, (int(np.prod(grid)), 96)))
            exact = tower_check.reference_rows(model, patches, grid)
            served = exact + scale * np.abs(exact).mean() * RNG.standard_normal(exact.shape).astype(np.float32)
            items[item_id] = (grid, patches, served)
        write_dump(dump / family, family, items, OUT)
        assert tower_check.main(["reference", "--family", family, "--source", str(tmp_path / "source"),
                                 "--dump", str(dump), "--threads", "2"]) == 0
    out = tmp_path / "check.json"
    assert tower_check.main(["compare", "--dump", str(dump), "--out", str(out), "--csv", str(tmp_path / "check.csv")]) == 0
    record = json.loads(out.read_text())
    fn = record["families"]["qwen3.8-flash-next"]
    assert [i["id"] for i in fn["items"]] == ["a", "b", "c"]
    assert all(i["rel_l2"] < 0.01 and i["control_within"]["rel_l2"] > 0.1 for i in fn["items"])
    assert record["bar"]["images"] == 3 and record["bar"]["flash_next_at_most_27b"] is True
    assert len((tmp_path / "check.csv").read_text().strip().splitlines()) == 1 + 6


# ---------------------------------------------------------------------------------------------- the diagnosis
import struct  # noqa: E402

import container  # noqa: E402  (the 27B NVFP4 converter's modules, on the path through `convert`)
import convert  # noqa: E402
import encode  # noqa: E402
import inventory  # noqa: E402


@pytest.mark.parametrize("fmt, qmax", [(inventory.Q4, 7), (inventory.Q5, 15), (inventory.Q6, 31), (inventory.W8, 127)])
@pytest.mark.parametrize("shape", [(8, 128), (4, 4304)])
def test_stored_grouped_words_decode_to_the_values_they_encode_exactly(fmt, qmax, shape):
    # Each group holds its scale times whole codes, the largest magnitude qmax: the encoder's binary16 scale
    # is exactly that scale and every code is exact, so the words decode to the very values, the padded K
    # (4,304 is stored 4,352 wide) cut away. -qmax sets a Q5 or Q6 code's high bits.
    group = inventory.GROUP[fmt][0]
    n, k = shape
    rng = np.random.default_rng(qmax + k)
    codes = rng.integers(-qmax, qmax + 1, size=(n, k))
    codes[:, ::group] = qmax
    codes[:, 1::group] = -qmax
    scales = 2.0 ** rng.integers(-12, -2, size=(n, (k + group - 1) // group))
    values = torch.tensor(codes * np.repeat(scales, group, axis=1)[:, :k], dtype=torch.float32)
    payload = encode.grouped(values, fmt, "cpu")
    assert torch.equal(tower_check.dequantize(payload, fmt, shape), values)


def test_stored_bf16_words_decode_to_their_values():
    values = torch.tensor([[1.0, -2.0], [0.25, 3.5]], dtype=torch.bfloat16)
    payload = encode.direct(values, inventory.BF16)
    assert tower_check.dequantize(payload, inventory.BF16, (2, 2)).tolist() == [[1.0, -2.0], [0.25, 3.5]]


# The fixture widths in each family's recipe: Flash-Next's own (every matrix W8) and the 27B's grouped one.
FIXTURE_FAMILY = {"fixture": "qwen3.8-flash-next", "fixture-27b": "qwen3.8-27b"}


def write_container(path, payloads, tower):
    """A `.ninfer` v2 container of a fixture tower's objects: `payloads` {object: bytes}, in its recipe's
    order and formats."""
    objects, at = [], 0
    for o in convert.tower_objects(tower):
        data = payloads[o.name]
        objects.append({"name": o.name, "kind": "tensor", "format": o.format, "layout": o.layout,
                        "shape": list(o.shape), "offset": at, "bytes": len(data)})
        at = container.align_up(at + len(data), 256)
    directory = json.dumps({"identity": {"model_id": "fixture-vision", "weights_id": "fixture"},
                            "objects": objects}).encode()
    start = container.align_up(16 + len(directory), 4096)
    blob = bytearray(at)
    for o in objects:
        blob[o["offset"]: o["offset"] + o["bytes"]] = payloads[o["name"]]
    with open(path, "wb") as f:
        f.write(b"NINFER\x00\x02" + struct.pack("<Q", len(directory)) + directory)
        f.write(b"\x00" * (start - 16 - len(directory)))
        f.write(bytes(blob))


def fixture_diagnosis(tmp_path, alter=None, geometry="fixture"):
    """The fixture tower's cache, its companion encoded in `geometry`'s recipe (`alter`ed: {object: payload}
    changed in place), and for each family a dump of two items whose served rows are the BF16 tower's own."""
    tensors = fixture_record(tmp_path)
    tower = tower_check.TOWERS[geometry]
    payloads = {o.name: p for o, p in convert.encode_tower(tensors, tower)}
    if alter is not None:
        alter(payloads)
    write_container(tmp_path / "vision.ninfer", payloads, tower)
    model = tower_check.reference_model(str(tmp_path / "source"))
    items = {}
    for item_id, grid in {"a": (1, 4, 4), "b": (1, 2, 6)}.items():
        patches = bits(np.random.default_rng(grid[2]).uniform(-1, 1, (int(np.prod(grid)), 96)))
        items[item_id] = (grid, patches, tower_check.reference_rows(model, patches, grid))
    for family in tower_check.FAMILIES:
        write_dump(tmp_path / "dump" / family, family, items, OUT)
    return tensors, payloads


def diagnose(tmp_path, *variants, geometry="fixture"):
    out = tmp_path / "diagnosis.json"
    argv = ["diagnose", "--family", FIXTURE_FAMILY[geometry], "--geometry", geometry, "--source",
            str(tmp_path / "source"), "--container", str(tmp_path / "vision.ninfer"), "--dump", str(tmp_path / "dump"),
            "--out", str(out), "--threads", "2"]
    for v in variants:
        argv += ["--variant", v]
    assert tower_check.main(argv) == 0
    return json.loads(out.read_text())


def stage(record, variant, name):
    return next(s for s in record["variants"][variant]["stages"] if s["stage"] == name)


def test_each_family_has_its_own_recipe_flash_next_every_matrix_w8_the_27b_its_grouped_formats():
    # GitHub #556: Flash-Next's tower is re-encoded with every matrix W8; the 27B's served tower keeps the
    # grouped recipe its container was built with, which its recipe check reads.
    grouped = [o for o in inventory.objects(inventory.Geometry.qwen3_8_27b()) if o.unit == "vision"]
    ours = convert.tower_objects(tower_check.TOWER_27B)
    assert [(o.name, o.format, o.shape) for o in ours] == [(o.name, o.format, o.shape) for o in grouped]
    flash_next = convert.tower_objects(tower_check.TOWERS["qwen3.8-flash-next"])
    assert [o.format for o in flash_next] == [inventory.BF16 if o.format == inventory.BF16 else inventory.W8
                                             for o in grouped]


def test_a_companion_of_flash_nexts_recipe_is_its_bf16_tower_encoded_every_matrix_in_w8(tmp_path):
    fixture_diagnosis(tmp_path)
    record = diagnose(tmp_path, "stored", "bf16+qkv=W8", "bf16+merger_fc1=Q4")
    assert record["recipe"] == {"objects": 333, "equal": 333, "differ": []}
    stored = record["weights"]["stored"]
    assert {kind: w["format"] for kind, w in stored.items()} == {kind: inventory.W8 for kind in tower_check.KINDS}
    # The stored words are the W8 encoding of the BF16 tensors: the same error as W8 re-encoded, below Q4's.
    assert stored["qkv"]["rel_frobenius"] == record["weights"]["W8"]["qkv"]["rel_frobenius"] > 0
    assert stored["merger_fc1"]["rel_frobenius"] < record["weights"]["Q4"]["merger_fc1"]["rel_frobenius"]


def test_a_companion_checked_against_another_familys_recipe_differs_in_every_matrix_that_recipe_moves(tmp_path):
    fixture_diagnosis(tmp_path, geometry="fixture-27b")
    record = diagnose(tmp_path, "stored", geometry="fixture-27b")
    assert record["recipe"] == {"objects": 333, "equal": 333, "differ": []}
    # Flash-Next's W8 companion where the 27B's grouped recipe is asked for: every matrix differs but the
    # merger's two, W8 in both recipes.
    fixture_diagnosis(tmp_path / "w8")
    os.replace(tmp_path / "w8" / "vision.ninfer", tmp_path / "vision.ninfer")
    record = diagnose(tmp_path, "stored", geometry="fixture-27b")
    assert record["recipe"]["equal"] == 333 - (1 + 27 * 4)
    assert "vision/patch_embedding" in record["recipe"]["differ"]
    assert "vision/merger/fc1" not in record["recipe"]["differ"]


def test_a_companion_of_the_27b_recipe_is_its_bf16_tower_encoded_and_the_bf16_variant_has_no_error(tmp_path):
    fixture_diagnosis(tmp_path, geometry="fixture-27b")
    record = diagnose(tmp_path, "bf16", "stored", geometry="fixture-27b")
    assert record["recipe"] == {"objects": 333, "equal": 333, "differ": []}
    bf16, stored = record["variants"]["bf16"], record["variants"]["stored"]
    assert bf16["final"]["rel_l2"] == 0 and all(s["rel_l2"] == 0 for s in bf16["stages"])
    # The served rows are the BF16 tower's, held as BF16 (a rounding of about 2^-9): the BF16 variant is
    # what was served, the stored tower is not.
    assert bf16["final"]["vs_served_rel_l2"] < 0.005
    assert stored["final"]["rel_l2"] > 0
    assert stored["final"]["vs_served_rel_l2"] > 1.5 * bf16["final"]["vs_served_rel_l2"]
    assert [s["stage"] for s in stored["stages"]][:2] == ["embedding", "block 0"]
    assert [s["stage"] for s in stored["stages"]][-3:] == ["merger norm", "merger fc1", "merger fc2"]
    weights = record["weights"]["stored"]
    assert weights["qkv"]["format"] == inventory.Q4 and weights["qkv"]["objects"] == 27
    assert 0 < weights["merger_fc1"]["rel_frobenius"] < weights["qkv"]["rel_frobenius"]


def test_a_companion_holding_another_tensor_is_named_by_the_recipe_check(tmp_path):
    def swap(payloads):
        payloads["vision/layers/3/mlp/fc1"] = payloads["vision/layers/4/mlp/fc1"]
    fixture_diagnosis(tmp_path, swap)
    record = diagnose(tmp_path, "stored")
    assert record["recipe"]["equal"] == 332 and record["recipe"]["differ"] == ["vision/layers/3/mlp/fc1"]


def test_one_kinds_stored_weights_move_the_stages_from_its_own_on(tmp_path):
    fixture_diagnosis(tmp_path)
    record = diagnose(tmp_path, "bf16+fc2=stored", "bf16+merger_fc2=stored")
    assert stage(record, "bf16+fc2=stored", "embedding")["rel_l2"] == 0
    assert stage(record, "bf16+fc2=stored", "block 0")["rel_l2"] > 0
    late = record["variants"]["bf16+merger_fc2=stored"]["stages"]
    assert all(s["rel_l2"] == 0 for s in late[:-1]) and late[-1]["rel_l2"] > 0


def test_more_bits_for_a_kind_give_less_error_and_cost_more_bytes(tmp_path):
    fixture_diagnosis(tmp_path)
    record = diagnose(tmp_path, "bf16+qkv=Q4", "bf16+qkv=Q6", "bf16+qkv=W8")
    errors = [record["variants"][f"bf16+qkv={f}"]["final"]["rel_l2"] for f in ("Q4", "Q6", "W8")]
    assert errors[0] > errors[1] > errors[2] > 0
    sizes = [record["variants"][f"bf16+qkv={f}"]["cost"]["tower_bytes"] for f in ("Q4", "Q6", "W8")]
    assert sizes[0] < sizes[1] < sizes[2]
    # A fixture qkv [96, 32] is 6,144 bytes in BF16 and 6,528 in Q4 (K padded to 128: 96 rows of 64 code
    # bytes, then 96 rows of two binary16 scales after the 256-aligned code plane), 27 of them.
    both = diagnose(tmp_path, "bf16+qkv=Q4", "bf16+qkv=BF16")["variants"]
    bf16, q4 = (both[f"bf16+qkv={f}"]["cost"]["tower_bytes"] for f in ("BF16", "Q4"))
    assert bf16 - q4 == 27 * (6144 - 6528)


def test_a_rule_over_a_range_of_blocks_moves_the_stages_from_its_first_block_on(tmp_path):
    fixture_diagnosis(tmp_path)
    record = diagnose(tmp_path, "bf16+block@26=stored", "bf16+fc1@3-4=stored", "stored+block@0-25=bf16")
    last = record["variants"]["bf16+block@26=stored"]["stages"]
    assert all(s["rel_l2"] == 0 for s in last[:27])
    assert stage(record, "bf16+block@26=stored", "block 26")["rel_l2"] > 0
    assert stage(record, "bf16+fc1@3-4=stored", "block 2")["rel_l2"] == 0
    assert stage(record, "bf16+fc1@3-4=stored", "block 3")["rel_l2"] > 0
    # Later rules win: every block in BF16 leaves the patch embedding's error alone before block 0.
    assert stage(record, "stored+block@0-25=bf16", "embedding")["rel_l2"] > 0
    # The rule's blocks alone change their format: two fixture fc1 [64, 32], W8 (8,704 bytes each: 64 rows of
    # 128 code bytes, K padded to 128, then of four binary16 scales) for BF16 (4,096 each).
    w8 = record["variants"]["bf16+fc1@3-4=stored"]["cost"]["tower_bytes"]
    bf16 = diagnose(tmp_path, "bf16")["variants"]["bf16"]["cost"]["tower_bytes"]
    assert w8 - bf16 == 2 * (8704 - 4096)


def test_a_variant_that_names_no_kind_or_a_merger_block_is_refused():
    for text in ("stored+mlp=W8", "fp32", "stored+merger_fc1@3=W8", "stored+qkv@5-2=W8", "stored+qkv=Q3"):
        with pytest.raises(tower_check.Refused):
            tower_check.Variant(text)


def test_against_another_diagnosis_each_variant_gets_the_bar_on_the_others_stored_tower(tmp_path):
    fixture_diagnosis(tmp_path, geometry="fixture-27b")
    first = tmp_path / "first.json"
    diagnose(tmp_path, "stored", geometry="fixture-27b")
    os.replace(tmp_path / "diagnosis.json", first)
    argv = ["diagnose", "--family", "qwen3.8-27b", "--geometry", "fixture-27b", "--source",
            str(tmp_path / "source"), "--container", str(tmp_path / "vision.ninfer"), "--dump", str(tmp_path / "dump"),
            "--out", str(tmp_path / "second.json"), "--threads", "2", "--against", str(first),
            "--variant", "stored", "--variant", "stored+qkv=W8"]
    assert tower_check.main(argv) == 0
    record = json.loads((tmp_path / "second.json").read_text())
    same = record["variants"]["stored"]["bar"]
    assert same["images"] == 2 and same["rel_l2"]["mean_difference"] == 0
    better = record["variants"]["stored+qkv=W8"]["bar"]
    assert better["rel_l2"]["mean_difference"] < 0 and better["rel_l2"]["flash_next_at_most_27b"] is True
    assert better["centred_rel_l2"]["mean_difference"] < 0
    assert record["against"] == {"file": "first.json", "family": "qwen3.8-27b", "variant": "stored"}
