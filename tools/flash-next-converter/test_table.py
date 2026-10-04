"""The INT4 g32 n-gram table rows and the hot-row list (layout.md §7)."""
import numpy as np
import torch

import table


def test_row_is_ninety_bytes_nibbles_then_scales():
    x = torch.zeros(1, 160)
    x[0, 0], x[0, 1] = 7.0, -7.0     # group 0: amax 7 -> scale 1.0
    x[0, 32] = 0.5                   # group 1: amax 0.5 -> scale 0.5/7
    raw = table.encode_rows(x)
    assert raw.shape == (1, 90) and raw.dtype == np.uint8
    assert raw[0, 0] == (7 + 8) | ((-7 + 8) << 4)       # values 0 (low nibble) and 1 (high nibble)
    assert raw[0, 2] == 8 | (8 << 4)                     # zeros store q = 0 -> nibble 8
    s = raw[0, 80:90].copy().view(np.float16)
    assert s[0] == np.float16(1.0) and s[1] == np.float16(0.5 / 7) and s[2] == 0


def test_decode_matches_the_rounding_rule():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1000, 160, generator=g) * 0.05).to(torch.bfloat16)
    raw = table.encode_rows(x)
    back = table.decode_rows(raw)
    xs = x.float().view(1000, 5, 32)
    s = torch.from_numpy(raw[:, 80:90].copy().view(np.float16).astype(np.float32))
    q = torch.round(xs / s[:, :, None]).clamp(-7, 7)
    assert torch.equal(back.view(1000, 5, 32), q * s[:, :, None])
    assert torch.all(q.abs() <= 7)


def test_quantizing_a_gathered_row_equals_the_stored_row():
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(50, 160, generator=g) * 0.1).to(torch.bfloat16)
    stored = table.encode_rows(x)
    rows = [3, 17, 17, 42]
    assert np.array_equal(table.encode_rows(x[rows]), stored[rows])
    # the 16-head gathered embedding (tokens, 2560) quantizes row by row
    emb = x[torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])].reshape(1, 2560)
    assert torch.equal(table.fake_quant(emb), table.decode_rows(stored[:16]).reshape(1, 2560).to(torch.bfloat16))


def test_hot_rows_rank_by_count_then_row_and_are_deterministic():
    ids = np.array([[5, 9, 5, 2], [9, 5, 7, 7]])        # counts: 5 x3, 9 x2, 7 x2, 2 x1
    assert table.hot_rows(ids, cap_rows=10).tolist() == [5, 7, 9, 2]
    assert table.hot_rows(ids, cap_rows=2).tolist() == [5, 7]
    perm = ids.ravel()[np.random.default_rng(0).permutation(8)]
    assert table.hot_rows(perm, cap_rows=10).tolist() == [5, 7, 9, 2]
