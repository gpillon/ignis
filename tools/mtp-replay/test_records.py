"""CPU tests of the MTP replay's record reader (records.py): the 1,016-byte record the driver
writes (crates/runtime/examples/mtp_replay.rs, whose Record::write asserts the same size), field
by field in its order.

python -m pytest tools/mtp-replay
"""
import json

import numpy as np

import records

# Record::write's order: the field, its dtype, its count.
DRIVER_LAYOUT = [
    ("job", "<u4", 1), ("turn", "<u4", 1), ("anchor", "<u4", 1), ("window", "<u4", 1), ("extent", "<u4", 1),
    ("accepted", "<u4", 1), ("committed", "<u4", 1),
    ("run", "<i4", 8),
    ("drafts", "<i4", 7),
    ("probability", "<f4", 7),
    ("cand_ids", "<i4", 7 * 16),
    ("cand_probs", "<f4", 7 * 16),
    ("agreed", "<u4", 1),
]


def driver_record(seed):
    """One record's bytes laid out the way Record::write lays them, and its fields."""
    rng = np.random.default_rng(seed)
    fields, parts = {}, []
    for name, dtype, count in DRIVER_LAYOUT:
        values = (rng.integers(-1, 248_320, count) if dtype == "<i4"
                  else rng.integers(0, 8, count) if dtype == "<u4"
                  else rng.random(count)).astype(dtype)
        fields[name] = values
        parts.append(values.tobytes())
    return b"".join(parts), fields


def test_the_record_is_the_drivers_1016_bytes_in_its_order():
    assert records.RECORD.itemsize == 1016
    assert sum(np.dtype(dtype).itemsize * count for _, dtype, count in DRIVER_LAYOUT) == 1016
    assert list(records.RECORD.names) == [name for name, _, _ in DRIVER_LAYOUT]
    offset = 0
    for name, dtype, count in DRIVER_LAYOUT:
        field, at = records.RECORD.fields[name]
        assert at == offset, name
        assert field.base == np.dtype(dtype), name
        assert int(np.prod(field.shape)) == count, name
        offset += np.dtype(dtype).itemsize * count


def test_a_driver_record_reads_back_field_by_field():
    raw, fields = driver_record(626)
    row = np.frombuffer(raw, dtype=records.RECORD)[0]
    for name, _, _ in DRIVER_LAYOUT:
        np.testing.assert_array_equal(np.ravel(row[name]), fields[name], err_msg=name)
    # The candidate tables keep the driver's row-major nesting: [position][candidate].
    assert row["cand_ids"].shape == (7, 16)
    assert row["cand_ids"][3, 5] == fields["cand_ids"][3 * 16 + 5]
    assert row["cand_probs"][6, 15] == fields["cand_probs"][6 * 16 + 15]


def test_acceptance_by_position_follows_the_accepted_count():
    rows = np.zeros(3, records.RECORD)
    rows["accepted"] = [0, 2, 7]
    np.testing.assert_array_equal(records.accepted_at(rows).sum(1), [0, 2, 7])
    assert records.accepted_at(rows)[1].tolist() == [True, True] + [False] * 5


def test_a_turns_last_round_tests_only_the_drafts_the_recording_reaches():
    # A turn of 10 tokens from 100: a full round at 100, then one at 108 with a token left after
    # its anchor (109), and one at 109 with none.
    manifest = {"jobs": [{"job": "j000", "turns": [{"turn": 5, "start": 100, "end": 110}]}]}
    rows = np.zeros(3, records.RECORD)
    rows["job"], rows["turn"], rows["window"], rows["extent"] = 0, 5, 7, 7
    rows["anchor"] = [100, 108, 109]
    rows["accepted"] = [3, 1, 0]
    tested = records.tested_at(rows, manifest)
    np.testing.assert_array_equal(tested.sum(1), [7, 1, 0])
    # Every accepted draft was tested; the last round's later drafts are untested, not rejected.
    assert not (records.accepted_at(rows) & ~tested).any()
    # A round at a narrower window or extent tests no more than it drafted.
    rows["window"] = 3
    np.testing.assert_array_equal(records.tested_at(rows, manifest).sum(1), [3, 1, 0])


def test_load_reads_the_finished_jobs_only(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    manifest = {"jobs": [{"job": "j000"}, {"job": "j001"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for job, rounds in (("j000", 3), ("j001", 2)):
        (out / f"{job}.rounds").write_bytes(b"".join(driver_record(rounds + i)[0] for i in range(rounds)))
    (out / "j000.done").write_bytes(b"")
    (out / "replay.json").write_text(json.dumps({"drafter": "mtp", "window": 7}))
    rows = records.load(str(out), records.read_manifest(str(tmp_path / "manifest.json")))
    assert len(rows) == 3, "j001 has no .done: a job the run did not finish is not read"
    assert records.replay(str(out))["drafter"] == "mtp"
