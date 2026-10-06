"""The reference readouts and the scorers spec 04 shares (layout.md §10)."""
import math

import pytest
import torch

import scoring


def test_top_k_breaks_ties_toward_the_lower_token_id():
    v = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0, 0.0]])
    ids, vals = scoring.top_k(v, 2)
    assert ids.tolist() == [[1, 2]] and vals.tolist() == [[3.0, 3.0]]
    ids, vals = scoring.top_k(v, 4)
    assert ids.tolist() == [[1, 2, 4, 3]]
    assert scoring.argmax(v).tolist() == [1]


def test_the_top64_scorer_is_the_exact_kl_when_the_reference_lives_in_its_top64():
    g = torch.Generator().manual_seed(0)
    ref = torch.full((3, 500), -60.0)
    ref[:, :40] = torch.randn(3, 40, generator=g)
    cand = ref + 0.3 * torch.randn(3, 500, generator=g)
    lr, lc = torch.log_softmax(ref, -1), torch.log_softmax(cand, -1)
    ids, lp = scoring.top_k(lr, 64)
    exact = scoring.kl_exact(lr, lc)
    approx = scoring.kl_top64(ids, lp, lc)
    assert torch.allclose(approx, exact, rtol=1e-4, atol=1e-7)


def test_the_top64_scorer_never_exceeds_the_exact_kl_and_tracks_it():
    g = torch.Generator().manual_seed(1)
    ref = torch.randn(4, 2000, generator=g) * 3
    cand = ref + 0.5 * torch.randn(4, 2000, generator=g)
    lr, lc = torch.log_softmax(ref, -1), torch.log_softmax(cand, -1)
    ids, lp = scoring.top_k(lr, 64)
    exact, approx = scoring.kl_exact(lr, lc), scoring.kl_top64(ids, lp, lc)
    assert torch.all(approx <= exact + 1e-6)      # grouping the tail can only lose divergence
    assert torch.all(approx > 0.5 * exact)


def test_mcnemar_exact_two_sided():
    assert scoring.mcnemar_p(0, 0) == 1.0
    assert scoring.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert scoring.mcnemar_p(3, 5) == pytest.approx(2 * (1 + 8 + 28 + 56) / 256)


def test_paired_counts_lost_and_gained_against_the_reference():
    ref_ok = [True, True, False, False, True]
    var_ok = [True, False, True, False, False]
    assert scoring.paired(ref_ok, var_ok)[:2] == (2, 1)


def test_db_of_a_relative_error():
    assert scoring.db(0.01) == pytest.approx(-20.0)
    assert math.isinf(scoring.db(0.0)) is False
