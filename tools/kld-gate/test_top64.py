"""The sm_86 gate's top-64 tools on a synthetic set: `ref_top64.py` writes the BF16 stream's top ids and
log-probabilities, `score_top64.py` scores a dump against them with `kl_top64`'s formula."""
import json
import math
import os

import numpy as np
import pytest
import torch

import ref_top64
import score_top64

HERE = os.path.dirname(os.path.abspath(__file__))
LN = math.log
# The reference's distribution at every position: (0.5, 0.25, 0.25) over a three-token vocabulary, so its
# top 2 are tokens 0 and 1 (the tie between 1 and 2 goes to the lower id) and its remaining mass is 0.25.
REF_LP = [LN(0.5), LN(0.25)]
# The engine's logits, exact in BF16: uniform; then token 0 ahead by 1; then token 1 ahead by 2.
ENGINE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
TOKENS = [0, 2, 1]
# KL with the remaining mass one bucket, worked by hand: 0.5 ln 1.125; 0.5 ln 0.125 - 0.5 + ln(e + 2);
# 0.5 ln 0.125 - 0.5 + ln(e^2 + 2). With a single token outside the top 2 it is the exact KL.
KL = [0.05889151782819173, 0.011723943092132982, 0.6998239953819663]


def write_dump(path, logits):
    """A `span_logits` window file: `[len][vocab]` little-endian BF16 (the values here are exact in BF16)."""
    a = np.asarray(logits, dtype=np.float32)
    (a.view(np.uint32) >> 16).astype("<u2").tofile(path)


def write_set(set_dir, q_argmax):
    """A one-window set of three positions in the layout `ref_top64.py` writes, top 2 of 3."""
    set_dir.mkdir()
    manifest = {"windows": [{"index": 0, "kind": "code", "source": "a.rs", "part": 0, "length": 3, "valid": 3,
                             "first_position": 0}], "positions": 3, "top": 2, "vocab": 3}
    (set_dir / "manifest.json").write_text(json.dumps(manifest))
    np.asarray(TOKENS, dtype="<u4").tofile(set_dir / "tokens.u32")
    np.asarray([[0, 1]] * 3, dtype="<i4").tofile(set_dir / "bf16_top64_ids.i32")
    np.asarray([REF_LP] * 3, dtype="<f4").tofile(set_dir / "bf16_top64_lp.f32")
    np.asarray(q_argmax, dtype="<i4").tofile(set_dir / "q_argmax.i32")


def test_a_dump_is_scored_top_2_against_the_set_per_position(tmp_path):
    write_set(tmp_path / "set", q_argmax=[0, 0, 0])
    dump = tmp_path / "dump"
    dump.mkdir()
    write_dump(dump / "w000.bin", ENGINE)
    ref = score_top64.ReferenceSet(str(tmp_path / "set"))

    # Two rows a slice: the window's three positions cross a slice boundary.
    kl, top1, top1_q, nll = score_top64.score_window(ref, ref.windows[0], str(dump), 2)

    assert kl == pytest.approx(KL, abs=1e-6)
    # The engine's argmax of a tie is the first token: it agrees with the reference's top token, 0, but at
    # the last position, and the set's 5090 stream says 0 at all three.
    assert top1.tolist() == [True, True, False]
    assert top1_q.tolist() == [True, True, False]
    # The NLL of the next token, none at the window's last position: -ln(1/3), -ln(1/(e + 2)).
    assert nll[:2] == pytest.approx([LN(3), LN(math.e + 2)], abs=1e-6)
    assert math.isnan(nll[2])


def test_the_scorer_reports_the_figures_the_gate_reads(tmp_path):
    write_set(tmp_path / "set", q_argmax=[0, 0, 1])
    dump = tmp_path / "dump"
    dump.mkdir()
    write_dump(dump / "w000.bin", ENGINE)
    out = tmp_path / "result.json"

    score_top64.main([str(tmp_path / "set"), str(dump), "--json", str(out)])

    figures = json.loads(out.read_text())["figures"]
    assert figures["all"]["positions"] == 3
    assert figures["all"]["kld_top64"] == pytest.approx(sum(KL) / 3, abs=1e-6)
    assert figures["all"]["top1_bf16"] == pytest.approx(2 / 3)
    assert figures["all"]["top1_q5090"] == 1.0
    assert figures["code"]["kld_top64"] == figures["all"]["kld_top64"]


def test_a_dump_of_another_size_than_its_window_is_refused(tmp_path):
    write_set(tmp_path / "set", q_argmax=[0, 0, 0])
    dump = tmp_path / "dump"
    dump.mkdir()
    write_dump(dump / "w000.bin", ENGINE[:2])
    ref = score_top64.ReferenceSet(str(tmp_path / "set"))

    with pytest.raises(SystemExit, match="expected 18"):
        score_top64.score_window(ref, ref.windows[0], str(dump), 2)


def test_the_reference_side_writes_what_the_scorer_reads():
    # The reference's logits give (0.5, 0.25, 0.25) at every position; the engine's are ENGINE.
    ref_logits = torch.tensor([[LN(2.0), 0.0, 0.0]] * 3).to(torch.bfloat16)
    out = ref_top64.score_slice(ref_logits, torch.tensor(ENGINE), torch.tensor([2, 1, 1]), ks=(2, 3), top=2)

    assert out["bf16_ids"].tolist() == [[0, 1]] * 3, "the tie between 1 and 2 goes to the lower id"
    expected_lp = [LN(2.0) - LN(4.0), -LN(4.0)]
    bf16_lp = torch.log_softmax(ref_logits.float(), dim=-1)[0, :2].tolist()
    assert out["bf16_lp"].tolist() == [bf16_lp] * 3 and bf16_lp == pytest.approx(expected_lp, abs=1e-2)
    # Top 2 is the scorer's formula; top 3, the whole vocabulary, is the exact KL of (a).
    assert out["kl_top3"].tolist() == pytest.approx(out["kl"].tolist(), abs=1e-5)
    assert out["kl_top2"].tolist() == pytest.approx(score_top64.kl_top64(out["bf16_lp"].numpy(),
                                                                        out["q_at_bf16"].double().numpy()), abs=1e-6)
    assert out["q_argmax"].tolist() == [0, 0, 1]
    assert out["top1"].tolist() == [True, True, False]


def test_a_set_rebuilds_from_its_own_windows(tmp_path):
    set_dir = tmp_path / "set"
    write_set(set_dir, q_argmax=[0, 0, 0])
    (set_dir / "windows.json").write_text(json.dumps([TOKENS]))
    assert ref_top64.windows_meta(str(set_dir / "windows.json")) == [{"kind": "code", "source": "a.rs", "part": 0}]
    (tmp_path / "w.json").write_text(json.dumps([TOKENS]))
    (tmp_path / "w_meta.json").write_text(json.dumps([{"kind": "chat", "source": "turns", "part": None}]))
    assert ref_top64.windows_meta(str(tmp_path / "w.json"))[0]["kind"] == "chat", "the meta file beside it first"
    (tmp_path / "alone.json").write_text(json.dumps([TOKENS]))
    with pytest.raises(SystemExit, match="manifest.json"):
        ref_top64.windows_meta(str(tmp_path / "alone.json"))


def test_the_set_names_the_artifact_the_catalog_pins():
    entry = ref_top64.catalog_entry(ref_top64.CATALOG, "qwen3.8-27b-16g")
    assert entry["file"] == "qwen3_8_27b_trellis_sc30-v2.ninfer"
    assert entry["sha256"] == "2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c"
    assert entry["bytes"] == 13379877124
