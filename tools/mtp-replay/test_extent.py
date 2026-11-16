"""CPU tests of the adaptive-extent replay (extent.py, GitHub #629): how a round is priced, where each policy
stops, how lanes share a round, the correction of the window-7 derivation, and that the same records give
the same extents.

python -m pytest tools/mtp-replay      or, without pytest:      python tools/mtp-replay/test_extent.py
"""
import numpy as np

import extent

K = extent.K


def cell(lanes=1, plain=10.0, mtp=None, verify=None, step=0.8, context=4096):
    """A cost-table cell in #628's layout."""
    mtp = mtp or [12.0 + 1.0 * w for w in range(K)]
    verify = verify or [11.0 + 0.2 * w for w in range(K)]
    return {"lanes": lanes, "context": context, "plain_ms": plain, "chain_step_ms": step,
            "by_window": {str(w + 1): {"mtp_ms": mtp[w], "verify_only_ms": verify[w]} for w in range(K)}}


def rounds(accepted, probability=None, turn_starts=(0,), jobs=None, cells=None):
    """Synthetic rounds: every draft tested, committed = accepted + 1."""
    accepted = np.asarray(accepted)
    n = len(accepted)
    probability = np.full((n, K), 0.9) if probability is None else np.asarray(probability, float)
    turn_start = np.zeros(n, bool)
    turn_start[list(turn_starts)] = True
    jobs = np.zeros(n, int) if jobs is None else np.asarray(jobs)
    job_start = np.concatenate([[True], jobs[1:] != jobs[:-1]])
    turn_start |= job_start
    cells = np.full(n, extent.CELLS.index("B-think")) if cells is None else np.asarray(cells)
    return extent.Rounds(accepted, accepted + 1, probability, np.ones((n, K), bool), cells, turn_start,
                         job_start, jobs)


def test_a_round_is_priced_by_its_steps_and_its_width():
    c = extent.Costs(cell())
    assert c.C[0] == c.V[0] == 10.0 and c.H[0] == 0.0
    one = np.array([[3]])
    # shapes: the window's own round when the chain ran as many steps as it verifies.
    assert np.isclose(c.round_ms("shapes", np.array([3]), one, K)[0], c.C[3])
    # A confidence stop that drops the draft under the threshold ran one step more than it verifies.
    assert np.isclose(c.round_ms("shapes", np.array([4]), one, K)[0], c.V[3] + c.H[4])
    # chain: the window's round less the steps the chain skipped.
    assert np.isclose(c.round_ms("chain", np.array([3]), one, K)[0], c.C[K] - 4 * 0.8)
    assert np.isclose(c.round_ms("chain", np.array([3]), one, 5)[0], c.C[5] - 2 * 0.8)
    # ragged: the substrate at the lanes' mean extent, between its windows, and the widest chain.
    two = np.array([[2, 5]])
    assert np.isclose(c.round_ms("ragged", np.array([5]), two, K)[0], (c.V[3] + c.V[4]) / 2 + c.H[5])


def test_the_linear_table_is_the_table_on_a_line():
    c = extent.Costs(cell())
    lin = c.linear()
    np.testing.assert_allclose(lin.C, c.C)
    np.testing.assert_allclose(lin.V, c.V)
    assert np.isclose(lin.step, 1.0 - 0.2)
    kinked = extent.Costs(cell(mtp=[12, 13, 14, 15, 17, 16, 18])).linear()
    assert np.all(np.diff(kinked.C[1:]) > 0)


def test_the_confidence_stop_is_the_first_draft_under_the_threshold():
    p = np.array([[0.9, 0.8, 0.3, 0.9, 0.9, 0.9, 0.9],
                  [0.95] * K,
                  [0.2] + [0.9] * (K - 1)])
    r = rounds([2, 7, 0], probability=p)
    s, e = extent.confidence(r, "draft", 0.5, False)
    assert s.tolist() == [3, 7, 1] and e.tolist() == [3, 7, 1]
    s, e = extent.confidence(r, "draft", 0.5, True)
    assert s.tolist() == [3, 7, 1] and e.tolist() == [2, 7, 0]
    # The chain's product: 0.9, 0.72 falls under 0.75 at the second draft, 0.95 ** n at the sixth.
    s, e = extent.confidence(r, "chain", 0.75, False)
    assert s.tolist() == [2, 6, 1]
    # No threshold is the fixed window; a narrower window caps the chain.
    assert extent.confidence(r, "chain", 0.0, False)[1].tolist() == [7, 7, 7]
    assert extent.confidence(r, "draft", 0.0, False, window=4)[1].tolist() == [4, 4, 4]


def test_a_fixed_window_commits_tokens_a_round_over_its_round():
    r = rounds([0, 3, 7, 5, 1, 2])
    c = extent.Costs(cell())
    for w in (0, 3, 7):
        o = extent.Outcome(r, *extent.lay_out(r, c, "shapes", w, *extent.fixed(r, w)))
        t = (np.minimum(r.accepted, w) + 1).mean()
        assert np.isclose(o.rate(), 1000 * t / c.C[w])
        assert np.isclose(o.rate(weighted=False), 1000 * t / c.C[w])
    # The oracle commits what window 7 commits at the price of each round's own run.
    o = extent.Outcome(r, *extent.lay_out(r, c, "shapes", K, *extent.oracle(r)))
    assert np.isclose(o.rate(), 1000 * (r.accepted + 1).sum() / c.C[r.accepted].sum())
    assert o.cut_fraction() == 0.0


def test_traffic_cells_pool_by_their_weights():
    # Two cells at the same price: the pooled rate is the cells' rates by their token weights.
    names = ("B-think", "C-content")
    cells = [extent.CELLS.index(names[0])] * 2 + [extent.CELLS.index(names[1])] * 2
    r = rounds([1, 1, 5, 5], cells=cells)
    c = extent.Costs(cell())
    o = extent.Outcome(r, *extent.lay_out(r, c, "shapes", K, *extent.fixed(r, K)))
    w = [extent.WEIGHTS[n] for n in names]
    rates = [1000 * 2 / c.C[K], 1000 * 6 / c.C[K]]
    assert np.isclose(o.rate(), sum(w) / (w[0] / rates[0] + w[1] / rates[1]))


def test_every_round_runs_once_on_every_lane_starting_at_a_job():
    r = rounds([1] * 10, jobs=[0, 0, 0, 1, 1, 2, 2, 2, 3, 3])
    at = extent.schedule(r, 2)
    assert at.shape == (10, 2)
    for lane in range(2):
        assert sorted(at[:, lane].tolist()) == list(range(10))
        assert r.job_start[at[0, lane]]
    assert at[0].tolist() == [0, 5]


def test_shapes_runs_every_lane_at_the_widest_extent_and_chain_each_at_its_own():
    # Two lanes from rounds 0 and 2: each step pairs a lane at 5 with one at 2.
    r = rounds([7, 1, 1, 7], jobs=[0, 0, 1, 1])
    c = extent.Costs(cell(lanes=2))
    steps = np.array([5, 2, 2, 5])
    _, e, ms = extent.lay_out(r, c, "shapes", K, steps, steps)
    assert e.tolist() == [[5, 5]] * 4
    assert np.allclose(ms, c.C[5])
    _, e, ms = extent.lay_out(r, c, "chain", K, steps, steps)
    assert e.tolist() == [[5, 2], [2, 5], [2, 5], [5, 2]]
    assert np.allclose(ms, c.C[K] - 2 * 0.8)
    _, e, ms = extent.lay_out(r, c, "ragged", K, steps, steps)
    assert e.tolist() == [[5, 2], [2, 5], [2, 5], [5, 2]]
    assert np.allclose(ms, (c.V[3] + c.V[4]) / 2 + c.H[5])


def test_the_history_widens_on_acceptance_narrows_on_rejection_and_forgets_at_a_turn():
    c = extent.Costs(cell())
    good = rounds([7] * 40)
    _, e, _ = extent.history(good, c, "shapes", K, [(0.9, 0.5, 1.0)])
    assert e[0, -1, 0] == K
    bad = rounds([0] * 40)
    _, e, _ = extent.history(bad, c, "shapes", K, [(0.9, 0.9, 1.0)])
    assert e[0, 0, 0] > 1 and e[0, -1, 0] == 1
    # A new turn starts again from the prior: the same extent as the first round.
    both = rounds([0] * 20 + [0] * 20, turn_starts=(0, 20))
    _, e, _ = extent.history(both, c, "shapes", K, [(0.9, 0.9, 1.0)])
    assert e[0, 20, 0] == e[0, 0, 0] and e[0, 19, 0] == 1


def test_the_history_learns_only_the_positions_its_extent_tested():
    # Held at extent 1 by a dear step, a lane sees position 1 alone: rounds that accept 1 and rounds that
    # accept 7 look the same to it, so they give the same extents.
    dear = extent.Costs(cell(mtp=[12.0 + 5.0 * w for w in range(K)]))
    one, seven = rounds([1] * 30), rounds([7] * 30)
    _, e1, _ = extent.history(one, dear, "shapes", K, [(0.9, 0.6, 1.0)])
    _, e7, _ = extent.history(seven, dear, "shapes", K, [(0.9, 0.6, 1.0)])
    assert set(e1[0, :, 0].tolist()) == {1} and np.array_equal(e1, e7)
    # At a cheap step the lane tests further and widens on what it sees.
    cheap = extent.Costs(cell(mtp=[12.0 + 0.01 * w for w in range(K)]))
    _, e, _ = extent.history(seven, cheap, "shapes", K, [(0.9, 0.6, 1.0)])
    assert e[0, -1, 0] == K


def test_the_same_records_give_the_same_extents():
    rng = np.random.default_rng(629)
    n = 300
    r = rounds(rng.integers(0, K + 1, n), probability=rng.random((n, K)), turn_starts=range(0, n, 37),
               jobs=np.repeat(np.arange(6), 50))
    c = extent.Costs(cell(lanes=2))
    runs = []
    for _ in range(2):
        s, e = extent.confidence(r, "chain", 0.4, False)
        a = extent.Outcome(r, *extent.lay_out(r, c, "shapes", K, s, e))
        at, e, ms = extent.history(r, c, "chain", K, [(0.8, 0.7, 4.0)])
        b = extent.Outcome(r, at, e[0], ms[0])
        runs.append((a.sha256, a.rate(), b.sha256, b.rate()))
    assert runs[0] == runs[1]


def test_the_correction_is_kappa_through_zero_over_the_cut_fraction():
    # Subsets whose direct tokens a round sit 1 + 0.1 * cut above the derivation give kappa 0.1.
    w7 = rounds([7, 0, 3, 5, 2, 6, 1, 4] * 10, jobs=np.repeat(np.arange(8), 10))
    subsets = {}
    for w in (4, 5, 6):
        derived = np.minimum(w7.accepted, w) + 1
        cut = (w7.accepted > w).mean()
        scale = 1 + 0.1 * cut
        # A direct walk with every round's commit scaled: same jobs, same cells.
        d = rounds(w7.accepted, jobs=w7.job)
        d.committed = derived * scale
        subsets[w] = d
    corr = extent.correction(w7, subsets)
    assert np.isclose(corr["kappa_weighted"], 0.1) and np.isclose(corr["kappa_unweighted"], 0.1)


def test_the_break_even_overhead_brings_the_rate_to_the_target():
    r = rounds([3, 4, 5, 6])
    c = extent.Costs(cell())
    o = extent.Outcome(r, *extent.lay_out(r, c, "shapes", K, *extent.fixed(r, K)))
    target = o.rate() / 1.1
    x = o.break_even_ms(target)
    assert x > 0 and np.isclose(o.rate(overhead_ms=x), target, rtol=1e-9)
    assert o.break_even_ms(o.rate() * 2) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
