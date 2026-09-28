"""`r3set.py`'s target rule and question checks, and `accept22.py`'s order
of asking, on inputs whose answer is known (spec 22 § Set R3, § The runs).

    python tools/locate-sets/test_r3set.py
"""

import random
import unittest

import accept22
import r3set


class Targets(unittest.TestCase):
    def test_a_line_whose_normalized_text_repeats_is_never_a_target(self):
        lines = ["pod ready in 12 ms on node alpha", "pod ready in 13 ms on node alpha",
                 "disk full on volume data-7 of the storage node", "auth refused for service account deployer"]
        picks = r3set.pick_targets(lines, 4, random.Random(1), sampled=False)
        self.assertEqual(sorted(p["line"] for p in picks), [2, 3], "the two ready lines differ in a number only")

    def test_targets_are_drawn_round_robin_over_the_sibling_bins(self):
        # A line with no sibling, and five near-duplicates of one another.
        lines = ["kernel panic on host gamma while mounting the root filesystem"]
        lines += [f"GET /api/v1/items/{i} user alice{i} served by gateway" for i in range(6)]
        picks = r3set.pick_targets(lines, 2, random.Random(2), sampled=False)
        self.assertEqual({p["bin"] for p in picks}, {0, 1}, picks)
        self.assertEqual(next(p for p in picks if p["bin"] == 0)["line"], 0)


class Checks(unittest.TestCase):
    lines = ["[api] GET /v1/orders user=u42 status=200", "[api] GET /v1/orders user=u42 status=503",
             "[api] GET /v1/orders user=u7 status=503", "[db] vacuum finished on table orders"]

    def test_a_combo_question_must_single_the_target_out(self):
        ok = {"split": "combo", "instruction": "Which request of user u42 failed with 503?"}
        self.assertIsNone(r3set.check(ok, self.lines, 1, 2))
        vague = {"split": "combo", "instruction": "Which request failed with 503?"}
        self.assertIn("also in", r3set.check(vague, self.lines, 1, 2))

    def test_a_paraphrase_shares_no_word_and_needs_a_target_without_siblings(self):
        clean = {"split": "paraphrase", "instruction": "When did the database complete its cleanup pass?"}
        self.assertIsNone(r3set.check(clean, self.lines, 3, 0))
        self.assertIn("siblings", r3set.check(clean, self.lines, 3, 2))
        leaky = {"split": "paraphrase", "instruction": "Which vacuum run completed?"}
        self.assertIn("content word", r3set.check(leaky, self.lines, 3, 0))


class Asking(unittest.TestCase):
    def test_a_states_questions_are_asked_together_the_first_marked(self):
        manifest = {"windows": {"w": ["a", "b", "c"]}, "questions": [
            {"id": "p1", "window": "w", "remove": None},
            {"id": "p1~removed", "window": "w", "remove": 1},
            {"id": "a0", "window": "w", "remove": None},
            {"id": "p2~removed", "window": "w", "remove": 2},
        ]}
        asked = accept22.questions_of("R3", manifest)
        self.assertEqual([q["id"] for q, _, _, _ in asked], ["p1", "a0", "p1~removed", "p2~removed"])
        self.assertEqual([first for _, _, first, _ in asked], [True, False, True, True])
        self.assertEqual(asked[2][1], "a\nc", "the removed line is gone")


if __name__ == "__main__":
    unittest.main()
