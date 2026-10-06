"""The projection and the interval phase A's verdict is read from (CPU, no weights)."""
import math

import phase_a


def test_tau_chains_conditional_acceptance():
    assert phase_a.tau([0.5, 0.5, 0.5], 0) == 1.0
    assert phase_a.tau([0.5, 0.5, 0.5], 2) == 1 + 0.5 + 0.25
    assert math.isclose(phase_a.tau([0.8, 0.5], 2), 1 + 0.8 + 0.4)


def test_the_pre_registered_projection_is_the_spec_table():
    # Spec 07's "Expected speedup": alpha 0.7 everywhere, c = 3.8 ms, d = 0.9 ms, R1 = 15.3 ms.
    cost = {"cells": [{"shape": "consecutive", "lanes": L, "width": w, "time": {"median_ms": 10.0 + 2 * (w - 1) * L}}
                      for L in (1, 2, 3) for w in (1, 2, 3)]}
    out = phase_a.project([0.7] * 4, cost)
    assert math.isclose(out["pre_registered"][1], 1.70 * 15.3 / 20.0)
    assert math.isclose(out["pre_registered"][2], 2.19 * 15.3 / 24.7)
    assert out["verdict"] == "GO"
    one = out["measured"][1]["k"][1]
    assert math.isclose(one["c_ms_per_column"], 2.0)
    assert math.isclose(one["speedup"], 1.7 * 10.0 / (12.0 + 0.9))
    assert 3 not in out["measured"][1]["k"], "a width the cost table lacks is not projected"
    assert phase_a.project([0.3] * 4, cost)["verdict"] == "NO-GO"


def test_a_repeat_table_reads_each_width_against_its_own_group():
    cell = lambda texts, w, ms: {"texts": texts, "lanes": len(texts), "width": w, "time": {"median_ms": ms}}
    cost = {"cells": [cell([0], 1, 10.0), cell([0], 2, 13.0), cell([0], 1, 12.0),
                      cell([5], 1, 20.0), cell([5], 2, 21.0), cell([5], 1, 20.0),
                      cell([1, 2], 1, 30.0), cell([1, 2], 2, 40.0)]}
    table = phase_a.round_table(cost)
    assert table[(1, 1)] == 15.5                      # mean of the groups' width-1 rounds (11, 20)
    assert table[(1, 2)] == 15.5 + (2.0 + 1.0) / 2    # mean increment over each group's own base
    assert table[(2, 2)] == 40.0


def test_wilson_brackets_the_rate():
    p, lo, hi = phase_a.wilson(1000, 700)
    assert p == 0.7 and lo < 0.7 < hi and hi - lo < 0.06
    assert phase_a.wilson(0, 0) == [0.0, 0.0, 0.0]
