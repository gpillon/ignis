"""The draft head's ranking: the totals ignis carries (`data/ranking.totals.i64`) and the pins at the 27B's
geometry, on the CPU. Their equality with ninfer's file is machine-local (`test_real_sources.py`)."""
import os

import pytest

import checkpoint
import inventory
import work

G = inventory.Geometry.qwen3_8_27b()


def test_the_committed_totals_are_one_row_of_the_27bs_vocabulary_with_the_pinned_sha256():
    assert checkpoint.TOTALS == os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ranking.totals.i64")
    assert os.path.getsize(checkpoint.TOTALS) == 248320 * 8
    assert work.file_digest(checkpoint.TOTALS) == "1e8fb35cd76655acac1cc422e2ef36e7dbedae567e04527ae433cf30b7dbd59f"
    assert set(checkpoint.PINNED["qwen3.8-27b"]["ranking_sha256"]) == {
        "1e8fb35cd76655acac1cc422e2ef36e7dbedae567e04527ae433cf30b7dbd59f",
        "c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5"}


def test_the_committed_totals_are_recorded_as_ignis_own():
    assert checkpoint.check_ranking(checkpoint.TOTALS, G) == {
        "file": "ranking.totals.i64", "bytes": 1986560,
        "sha256": "1e8fb35cd76655acac1cc422e2ef36e7dbedae567e04527ae433cf30b7dbd59f",
        "origin": "ignis tools/27b-nvfp4-converter/data/ranking.totals.i64 (row 0 of ninfer's "
                  "ranking.train.counts.i64: the totals)"}


def test_a_ranking_matching_neither_pin_is_refused(tmp_path):
    other = tmp_path / "ranking.totals.i64"
    with open(checkpoint.TOTALS, "rb") as f:
        data = bytearray(f.read())
    data[0] ^= 1
    other.write_bytes(bytes(data))
    with pytest.raises(checkpoint.Refused) as refused:
        checkpoint.check_ranking(str(other), G)
    message = str(refused.value)
    assert message.startswith(f"the ranking {other} hashes to {work.file_digest(str(other))}")
    assert "1e8fb35cd76655acac1cc422e2ef36e7dbedae567e04527ae433cf30b7dbd59f" in message
    assert "c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5" in message
