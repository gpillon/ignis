"""The K allocation (spec 01 "Allocation", layout.md §5)."""
import torch

import allocate
import layout


def curves(n, seed):
    """Distortion falling with K, a spread of expert energies, some unrouted experts."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, generator=g) * 0.2 + 0.05
    Dv = torch.stack([base * 2.0 ** (-2 * (k2 / 2 - 2)) for k2 in layout.K2_SET], 1)
    en = torch.rand(n, generator=g) ** 3 * 100
    en[:7] = 0
    return en, Dv


def test_mean_rate_with_scales_is_at_most_the_budget_and_uses_it():
    for proj in ("gu", "dn"):
        en, Dv = curves(512, 1)
        k2 = allocate.allocate(en, Dv, proj, 2.5)
        rates = [layout.rate(proj, k) for k in k2]
        mean = sum(rates) / len(rates)
        assert mean <= 2.5
        # the budget is used: no single expert could move up one class without exceeding it
        assert mean > 2.5 - (2.0 + 0.04) / 512
        assert set(k2) <= set(layout.K2_SET)


def test_every_expert_keeps_at_least_two_bits_even_unrouted():
    en, Dv = curves(64, 2)
    en[:] = 0
    k2 = allocate.allocate(en, Dv, "gu", 2.5)
    assert k2 == [4] * 64


def test_the_heaviest_experts_get_the_most_bits():
    en = torch.tensor([1.0] * 63 + [1e6])
    Dv = torch.tensor([[0.1, 0.05, 0.025, 0.006]] * 64)
    k2 = allocate.allocate(en, Dv, "dn", 2.5)
    assert k2[-1] == 8


def test_run8_layer1_histogram_shape_is_reachable():
    # a mean of exactly 2.5000 on down (run 8 layer 1) is reachable with the scale overhead counted
    en, Dv = curves(512, 3)
    k2 = allocate.allocate(en, Dv, "dn", 2.5)
    assert sum(layout.rate("dn", k) for k in k2) / 512 <= 2.5
