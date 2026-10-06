"""The pure parts of the exllamav3 wrapper (the quantizer and decode need the GPU)."""
import torch

import trellis


def test_k_value_is_an_int_for_integer_k_and_a_float_for_the_half_step():
    assert [trellis.k_value(k2) for k2 in (4, 5, 6, 8)] == [2, 2.5, 3, 4]
    assert isinstance(trellis.k_value(6), int) and isinstance(trellis.k_value(5), float)


def test_shrink_adds_five_percent_of_the_target_at_equal_trace():
    H = torch.diag(torch.tensor([4.0, 0.0]))[None]       # trace 4
    T = torch.eye(2)[None] * 10                          # trace 20
    out = trellis.shrink(H, T)
    # H + 0.05 * T * (4 / 20) = H + 0.01 * T
    assert torch.allclose(out, H + 0.1 * torch.eye(2)[None])


def test_checksum_matches_a_hand_computed_case():
    # u16 bits 1 and 2 at indices 0 and 1: 1*(lowbias32(0)|1) + 2*(lowbias32(1)|1)
    v = torch.tensor([1, 2], dtype=torch.int16).view(torch.float16)
    l0 = int(trellis.lowbias32(torch.tensor(0))) | 1
    l1 = int(trellis.lowbias32(torch.tensor(1))) | 1
    assert l0 == 1                                       # lowbias32(0) == 0
    assert trellis.checksum_u16(v) == (l0 + 2 * l1) % (1 << 64)
