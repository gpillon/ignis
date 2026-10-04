"""The expert records and the per-layer index (docs/specs/flash-next/layout.md §3-§4)."""
import numpy as np
import pytest

import layout


def test_record_bytes_are_the_contract_table():
    # layout.md §3, worked by hand: trellis in*out*K/8 + 2*(in+out), rounded up to 4096
    assert [layout.record_bytes("gu", k2) for k2 in (4, 5, 6, 8)] == [827392, 1032192, 1236992, 1646592]
    assert [layout.record_bytes("dn", k2) for k2 in (4, 5, 6, 8)] == [417792, 520192, 622592, 827392]


def test_record_bytes_refuses_a_k_outside_the_set():
    with pytest.raises(ValueError):
        layout.record_bytes("gu", 7)


def fake_record(proj, k2, seed):
    rng = np.random.default_rng(seed)
    i, o = layout.SHAPES[proj]
    return {"trellis": rng.integers(-2**15, 2**15, (i // 16, o // 16, 8 * k2), dtype=np.int16),
            "suh": rng.standard_normal(i).astype(np.float16),
            "svh": rng.standard_normal(o).astype(np.float16)}


def test_experts_file_round_trips_through_the_index(tmp_path):
    ks = {(0, "gu"): 4, (0, "dn"): 5, (1, "gu"): 8, (1, "dn"): 6, (2, "gu"): 5, (2, "dn"): 4}
    recs = {key: fake_record(key[1], k2, n) for n, (key, k2) in enumerate(ks.items())}
    layout.write_experts(tmp_path, 3, lambda e, p: (ks[(e, p)], recs[(e, p)]))
    index = layout.read_index(tmp_path / "experts.idx")
    assert [(e.expert, e.proj, e.k2) for e in index] == [(0, "gu", 4), (0, "dn", 5), (1, "gu", 8), (1, "dn", 6),
                                                         (2, "gu", 5), (2, "dn", 4)]
    assert index[0].offset == 0 and all(e.offset % 4096 == 0 for e in index)
    assert (tmp_path / "experts.bin").stat().st_size == sum(layout.record_bytes(e.proj, e.k2) for e in index)
    for e in index:
        back = layout.read_record(tmp_path / "experts.bin", e)
        want = recs[(e.expert, e.proj)]
        for name in ("trellis", "suh", "svh"):
            assert back[name].dtype == want[name].dtype
            assert np.array_equal(back[name].view(np.uint8), want[name].view(np.uint8)), name


def test_index_entry_is_sixteen_little_endian_bytes(tmp_path):
    rec = fake_record("dn", 5, 0)
    layout.write_experts(tmp_path, 1, lambda e, p: (5, rec) if p == "dn" else (4, fake_record("gu", 4, 1)))
    raw = (tmp_path / "experts.idx").read_bytes()
    assert len(raw) == 32
    # second entry: expert 0, proj 1 (dn), k2 5, size 520192, offset 827392 (after the gu-2 record)
    assert raw[16:32] == (0).to_bytes(2, "little") + bytes([1, 5]) + (520192).to_bytes(4, "little") \
        + (827392).to_bytes(8, "little")


def test_a_record_with_the_wrong_word_count_is_refused(tmp_path):
    bad = fake_record("gu", 4, 0)
    with pytest.raises(ValueError):
        layout.write_experts(tmp_path, 1, lambda e, p: (5, bad))


def test_rate_counts_scales_but_not_padding():
    # layout.md §5: K + 16(in+out)/(in*out); gate/up 0.01875, down 0.03125
    assert layout.rate("gu", 5) == pytest.approx(2.5 + 0.01875)
    assert layout.rate("dn", 4) == pytest.approx(2.0 + 0.03125)
    assert layout.stored_rate("dn", 4) == pytest.approx(417792 * 8 / (640 * 2560))
