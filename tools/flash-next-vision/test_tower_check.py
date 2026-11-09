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
