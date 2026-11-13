"""CPU tests of the draft-rule reader (draft_rule.py, GitHub #577): the drafter's q is the
host reference's (crates/core/src/sampled_draft.rs, held to its oracle), and the sampled
draft's per-column terms are the live round's probabilities worked by hand.

python -m pytest tools/dflash2-replay
"""
import numpy as np

import draft_rule

# The rule oracle's column (crates/core/tests/sampled_draft_oracle.rs): softmax at
# temperature 1 of 1, 8, 2, 4 and 1 sixteenths, out of order and with a tie.
WEIGHTS = np.array([1.0, 8.0, 2.0, 4.0, 1.0])
SCORES = np.log(WEIGHTS / 16.0)


def test_the_drafters_q_is_the_host_references_closed_forms():
    np.testing.assert_allclose(draft_rule.drafter_q(SCORES, 1.0, 1.0), WEIGHTS / 16.0)
    # 8, 12, 14 and 15 sixteenths: the fourth reaches 0.9, and of the two ones the first.
    np.testing.assert_allclose(draft_rule.drafter_q(SCORES, 1.0, 0.9), [1 / 15, 8 / 15, 2 / 15, 4 / 15, 0.0])
    # Squared, 64, 16, 4, 1 and 1 of 86: 84/86 is the first prefix past 0.95.
    np.testing.assert_allclose(draft_rule.drafter_q(SCORES, 0.5, 0.95), [0.0, 64 / 84, 4 / 84, 16 / 84, 0.0])
    # The inclusive cut breaks a tie toward the smaller index; temperature 0 is the walk.
    np.testing.assert_allclose(draft_rule.drafter_q(np.full(4, 3.0), 1.0, 0.5), [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_array_equal(draft_rule.drafter_q(np.array([0.5, 2.0, -1.0, 2.0]), 0.0, 0.95), [0, 1, 0, 0])


def one_column(recorded):
    """One round, one column, worked by hand: the target 1/2 on 501, 1/2 on 77; the
    drafter's 16 candidates 501, 77 and fourteen others, q 3/4 on 501 and 1/4 on 77."""
    p_ids = np.full((1, 1, 20), -1, np.int32)
    p = np.zeros((1, 1, 20))
    p_ids[0, 0, :2] = [501, 77]
    p[0, 0, :2] = [0.5, 0.5]
    cand = np.arange(1000, 1016, dtype=np.int32).reshape(1, 1, 16)
    cand[0, 0, :2] = [501, 77]
    scores = np.full((1, 1, 16, 16), -1e9)
    scores[0, 0, :, 0] = np.log(0.75)
    scores[0, 0, :, 1] = np.log(0.25)
    rec = np.array([[recorded]], np.int32)
    return draft_rule.sampled_terms(p_ids, p, rec, cand, scores, 1.0, 1.0)


def test_the_sampled_drafts_terms_are_a_live_rounds_probabilities():
    m, a = one_column(77)
    # A draft is taken with probability sum_x min(p, q) = 1/2 + 1/4.
    np.testing.assert_allclose(m, [[0.75]])
    # Given the target sampled 77, the round drafted it with min(1/2, 1/4) / (1/2).
    np.testing.assert_allclose(a, [[0.5]])
    # 501, which q holds more of than p, is always the draft when it is emitted.
    np.testing.assert_allclose(one_column(501)[1], [[1.0]])
    # A candidate the target gives nothing: the drafter's own probability, 0 here.
    np.testing.assert_allclose(one_column(1005)[1], [[0.0]])
    # A token neither holds.
    np.testing.assert_allclose(one_column(5)[1], [[0.0]])


def test_the_expectation_and_the_survival_follow_the_recorded_chain():
    a = np.array([[1.0, 0.5, 0.0]])
    m = np.array([[0.8, 0.6, 0.9]])
    # 0.8 + 1.0 * 0.6 + 1.0 * 0.5 * 0.9
    np.testing.assert_allclose(draft_rule.rb(a, m), [0.8 + 0.6 + 0.45])
    np.testing.assert_allclose(draft_rule.survival(a, m), [[0.8, 0.6, 0.45]])
