"""The reference side of gate 1 on synthetic logits: an engine dump scored against a reference's head."""
import json
import math

import numpy as np
import pytest
import torch

import ref_kld

LN4, LN3 = math.log(4.0), math.log(3.0)


def write_dump(path, logits):
    """A `span_logits` window file: `[len][vocab]` little-endian BF16 (the values here are exact in BF16)."""
    a = np.asarray(logits, dtype=np.float32)
    (a.view(np.uint32) >> 16).astype("<u2").tofile(path)


def test_an_engine_dump_is_scored_per_position_against_the_reference(tmp_path):
    # Two-token vocabulary, the head the identity: the reference's logits are its hidden rows.
    # Positions 0, 2, 3: the reference says (0.2, 0.8); position 1: (0.8, 0.2).
    hidden = [torch.tensor([[0.0, LN4], [LN4, 0.0], [0.0, LN4], [0.0, LN4]])]
    head = torch.eye(2)
    # The engine says (0.5, 0.5) but at position 2: logits (0, 2), (0.119203, 0.880797).
    dump = tmp_path / "engine"
    dump.mkdir()
    write_dump(dump / "w000.bin", [[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    windows, meta = [[1, 0, 1, 1]], [{"kind": "code"}]

    (row,) = ref_kld.score_windows(windows, meta, hidden, head, {"engine": str(dump)})

    # KL(p || q) = 0.2 ln(0.2 / 0.5) + 0.8 ln(0.8 / 0.5), and 0.2 ln(0.2 / 0.119203) + 0.8 ln(0.8 / 0.880797)
    assert row["engine"]["kl"] == pytest.approx([0.192745, 0.192745, 0.026526, 0.192745], abs=1e-6)
    # The engine's argmax of a tie is the first token: it agrees with the reference only at positions 1 and 2.
    assert row["engine"]["top1"] == [0.0, 1.0, 1.0, 0.0]
    # The NLL of the next token (the window's last position has none): -ln 0.5, -ln 0.5, -ln 0.880797.
    assert row["engine"]["nll"] == pytest.approx([0.693147, 0.693147, 0.126928], abs=1e-6)
    # The reference's own: the next tokens are 0, 1, 1, so -ln 0.2, -ln 0.2, -ln 0.8.
    assert row["ref_nll"] == pytest.approx([1.609438, 1.609438, 0.223144], abs=1e-6)
    assert (row["window"], row["kind"]) == (0, "code")


def test_a_saved_reference_scores_a_dump_as_the_forward_it_was_saved_from(tmp_path):
    torch.manual_seed(0)
    windows, meta = [[3, 1, 4, 1, 5], [2, 7, 1]], [{"kind": "code"}, {"kind": "chat"}]
    hidden = [torch.randn(len(ids), 6).to(torch.bfloat16) for ids in windows]
    head = torch.randn(8, 6).to(torch.bfloat16)
    dump = tmp_path / "engine"
    dump.mkdir()
    for w, ids in enumerate(windows):
        write_dump(dump / f"w{w:03d}.bin", torch.randn(len(ids), 8).to(torch.bfloat16).float().numpy())
    saved = str(tmp_path / "reference.pt")

    ref_kld.save_hidden(saved, hidden, windows, {"ckpt": "H:/models/a-checkpoint", "overlay": None})
    loaded, reference = ref_kld.load_hidden(saved, windows)

    assert reference == {"ckpt": "H:/models/a-checkpoint", "overlay": None}
    assert ref_kld.score_windows(windows, meta, loaded, head, {"e": str(dump)}) == \
        ref_kld.score_windows(windows, meta, hidden, head, {"e": str(dump)})


def test_a_saved_reference_is_refused_for_other_windows(tmp_path):
    windows = [[3, 1, 4, 1, 5], [2, 7, 1]]
    saved = str(tmp_path / "reference.pt")
    ref_kld.save_hidden(saved, [torch.zeros(5, 2), torch.zeros(3, 2)], windows, {"ckpt": "c"})

    with pytest.raises(ValueError, match="other windows"):
        ref_kld.load_hidden(saved, [[3, 1, 4, 1, 5], [2, 7, 2]])
    with pytest.raises(ValueError, match="other windows"):
        ref_kld.load_hidden(saved, windows[:1])


def test_the_checkpoint_keeps_one_shard_open_at_a_time(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    layer = "model.language_model.layers.0."
    shards = {"model-00001-of-00002.safetensors": {layer + "a.weight": torch.full((2, 2), 1.0)},
              "model-00002-of-00002.safetensors": {layer + "b.weight": torch.full((2, 2), 2.0),
                                                   "lm_head.weight": torch.full((3, 2), 3.0)}}
    for name, tensors in shards.items():
        save_file(tensors, str(tmp_path / name))
    index = {"weight_map": {t: name for name, tensors in shards.items() for t in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    real, state = ref_kld.safe_open, {"open": 0, "peak": 0}

    class Counted:
        def __init__(self, path, **kwargs):
            self.inner = real(path, **kwargs)

        def __enter__(self):
            state["open"] += 1
            state["peak"] = max(state["peak"], state["open"])
            return self.inner.__enter__()

        def __exit__(self, *exc):
            state["open"] -= 1
            return self.inner.__exit__(*exc)

    monkeypatch.setattr(ref_kld, "safe_open", Counted)
    ck = ref_kld.Checkpoint(str(tmp_path))

    got = ck.tensors(layer)

    assert sorted(got) == ["a.weight", "b.weight"]
    assert torch.equal(got["a.weight"], torch.full((2, 2), 1.0)) and torch.equal(got["b.weight"], torch.full((2, 2), 2.0))
    assert torch.equal(ck.tensors("lm_head.")["weight"], torch.full((3, 2), 3.0))
    assert (state["peak"], state["open"]) == (1, 0)
