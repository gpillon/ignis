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


def test_encodings_follow_the_contract_table():
    # layout.md §6.2, from the checkpoint's own names and shapes
    cases = {
        ("linear_attn.in_proj_qkv.weight", (10240, 2560)): "fp8",
        ("linear_attn.in_proj_a.weight", (48, 2560)): "fp8",
        ("self_attn.indexer.index_qk_proj.weight", (640, 2560)): "fp8",
        ("mlp.shared_expert.down_proj.weight", (2560, 640)): "fp8",
        ("attn_hyper_connection.input_mix_weight_down.weight", (320, 10240)): "fp8",
        ("ple.key_proj.weight", (10240, 2560)): "fp8",
        ("embed_tokens.weight", (248320, 2560)): "fp8",
        ("mlp.gate.weight", (512, 2560)): "bf16",
        ("mlp.shared_expert_gate.weight", (1, 2560)): "bf16",
        ("mlp_hyper_connection.block_inject_weight.weight", (4, 10240)): "bf16",
        ("attn_hyper_connection.hc_norm.weight", (10240,)): "bf16",
        ("linear_attn.conv1d.weight", (10240, 1, 4)): "bf16",
        ("linear_attn.A_log", (48,)): "bf16",
        ("mlp.experts.gate_up_proj", (512, 1280, 2560)): "expert",
    }
    for (name, shape), want in cases.items():
        assert layout.encoding_of(name, shape) == want, name
    assert layout.measured_by_study("linear_attn.out_proj.weight")
    assert not layout.measured_by_study("attn_hyper_connection.input_mix_weight_up.weight")


def test_the_fixture_tree_is_complete_and_readable(tmp_path):
    import fixture
    work = fixture.make(str(tmp_path))
    for L in range(2):
        d = tmp_path / "work" / "layers" / f"L{L:02d}"
        assert layout.is_done(d)
        index = layout.read_index(d / "experts.idx")
        assert len(index) == 16 and len({(e.proj, e.k2) for e in index}) >= 3
    raw = (tmp_path / "work" / "ngram" / "table" / "shard_000.int4").read_bytes()
    assert len(raw) == 1000 * 90
    assert layout.is_done(tmp_path / "work" / "ngram") and layout.is_done(tmp_path / "work" / "global")


def test_done_verification_catches_a_torn_or_changed_file(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.json").write_text("{}")
    layout.mark_done(tmp_path)
    assert layout.verify_done(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"x" * 99 + b"y")      # same size, other bytes
    assert not layout.verify_done(tmp_path)
    (tmp_path / "a.bin").write_bytes(b"x" * 50)              # torn
    assert not layout.verify_done(tmp_path)
    assert not layout.verify_done(tmp_path / "missing")
