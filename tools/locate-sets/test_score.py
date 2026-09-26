"""`score.py` on a synthetic dump whose answer is known.

One head (L27.h5) puts its attention on the target segment's keys; every
other head is noise. Cross-validation must find that head as R1, R2's
selectivity rule must keep it, rule 1 must choose a reading that finds every
target, and the check must say GO against a labelled route that does no
better. The feature arithmetic is held to a hand-computed case.

    python tools/locate-sets/test_score.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

import score

GOOD = 6 * score.HEADS + 5       # L27.h5


def write_dump(directory, name, n, seed, family="logs"):
    rng = np.random.default_rng(seed)
    rows, blocks, end = [], [], 0
    for i in range(n):
        segments = int(rng.integers(5, 40))
        keys, at = [], 0
        for j in range(segments):
            width = int(rng.integers(2, 9))
            keys.append([at, at + width])
            at += width + 1                      # one separator key after each
        span = at - 1
        absent = i % 6 == 5
        target = int(rng.integers(0, segments))
        variants = {}
        for v in score.VARIANTS:
            s = rng.normal(0, 1, size=(score.N_HEADS, span)).astype(np.float32)
            if not absent and not v.endswith("-na"):
                a, b = keys[target]
                s[GOOD, a:b] += 8.0
            variants[v] = {"offset": end, "prompt_tokens": span + 60}
            blocks.append(s.astype("<f2").ravel())
            end += s.size
        rows.append({
            "id": f"{family}-{i:03}", "family": family, "split": ("lexical", "paraphrase")[i % 2],
            "absent": absent, "segments": segments, "span": [10, span],
            "targets": [] if absent else [target], "keys": keys, "variants": variants,
            "l39_h10": {"s1": 0, "s2": 0},
        })
    np.concatenate(blocks).tofile(os.path.join(directory, f"{name}.bin"))
    with open(os.path.join(directory, f"{name}.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    meta = {"set": name, "rows_file": f"{name}.jsonl", "bin_file": f"{name}.bin", "alphabet": []}
    path = os.path.join(directory, f"{name}.json")
    score.write_json(path, meta)
    return path, rows


class ScoreTest(unittest.TestCase):
    def test_features_are_softmax_mass_per_segment(self):
        scores = np.log(np.array([[1.0, 2.0, 1.0, 4.0]] * score.N_HEADS, dtype=np.float32))
        shares, argseg = score.features(scores, [[0, 2], None, [3, 4]])
        np.testing.assert_allclose(shares[0], [3 / 8, 0.0, 4 / 8], rtol=1e-6)
        self.assertEqual(argseg[0], 2)
        # The argmax on a separator key credits nobody.
        shares, argseg = score.features(scores, [[0, 2], None, [3, 3]])
        self.assertEqual(argseg[0], -1)

    def test_a_baseline_can_go_negative_and_the_winner_is_still_the_largest(self):
        s = np.array([-0.2, 0.1, -np.inf, 0.3])
        self.assertEqual(score.winner(s), 3)
        self.assertAlmostEqual(score.confidence(s, True), 0.3 / 0.4)
        self.assertAlmostEqual(score.confidence(np.array([0.5, 0.5]), False), 0.5)

    def test_cross_validation_finds_the_pointing_head_and_rule_2_says_go(self):
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for name, seed in (("A", 1), ("B", 2)):
                path, _ = write_dump(d, name, 36, seed)
                rows += score.load(path, cache=False)[1]
            cv = score.cross_validate(rows)
            r1 = cv[score.config_name("r1", "s1", False)]
            self.assertEqual(r1["hits"], r1["of"])
            self.assertTrue(all(f["method"]["head"] == GOOD for f in r1["folds"]))
            r2 = cv[score.config_name("r2", "s2", False)]
            self.assertTrue(all(GOOD in f["method"]["heads"] for f in r2["folds"]))
            (name, best), _ = score.rule1(cv)
            self.assertEqual(best["top1"], 100.0)
            self.assertEqual((best["kind"], best["baseline"], best["scaffold"]), ("r1", False, "s1"))

            present = [r for r in rows if not r["absent"]]
            method = score.select(present, "r1", "s1", False)
            choice = {"config": name, "kind": "r1", "scaffold": "s1", "baseline": False,
                      "method": score.describe(method)}
            path, check_rows = write_dump(d, "C", 36, 3)
            loaded = score.load(path, cache=False)[1]
            labelled = {"questions": [{"id": r["id"], "hit": not r["absent"], "choice_ms": 1.0, "noul_ms": 1.0}
                                      for r in check_rows]}
            report = score.check(choice, loaded, labelled)
            self.assertTrue(report["rule2"]["go"])
            self.assertEqual(report["top1"]["pct"], 100.0)
            # Every present confidence above every absent one.
            self.assertEqual(report["auc_present_absent"], 1.0)

    def test_the_vote_counts_each_heads_winner(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = write_dump(d, "V", 12, 5)
            rows = score.load(path, cache=False)[1]
            for r in (r for r in rows if not r["absent"]):
                s = score.reading(r, ("vote", [GOOD, 0, 1]), "s1", True)
                self.assertIn(score.winner(s), r["targets"])

    def test_a_vote_tie_goes_to_the_segment_of_the_best_ranked_head(self):
        # Five heads over four segments (the third owns no key), no baseline
        # needed: heads 0 and 4 name segment 1, heads 1 and 2 segment 3,
        # head 3 segment 0. Two votes each for 1 and 3; head 0 ranks first.
        shares = np.zeros((score.N_HEADS, 4), dtype=np.float32)
        for h, seg in ((0, 1), (1, 3), (2, 3), (3, 0), (4, 1)):
            shares[h, seg] = 1.0
        row = {"keys": [[0, 1], [2, 3], None, [4, 5]],
               "feat": {"s2": (shares, None), "s2-na": (np.zeros_like(shares), None)}}
        s = score.reading(row, ("vote", [0, 1, 2, 3, 4]), "s2", True)
        self.assertEqual(score.winner(s), 1)
        self.assertEqual(score.winner(score.reading(row, ("vote", [1, 0, 2, 3, 4]), "s2", True)), 3)
        self.assertTrue(np.isneginf(s[2]))
        # the confidence is the winner's share of the votes
        self.assertAlmostEqual(score.confidence(s, True), 2 / 5, places=4)

    def test_a_floor_is_the_hits_less_three_percent_of_the_questions(self):
        # C's prose: 55 of 67 -> a floor of 52.99, so 53 clear it and 52 do not.
        floor = score.floor_of(55, 67)
        self.assertAlmostEqual(floor["floor"], 52.99)
        self.assertEqual(floor["at_least"], 53)
        # 80 questions leave a slack of 2.4: 60 hits need 58.
        self.assertEqual(score.floor_of(60, 80)["at_least"], 58)
        # A whole-number floor is its own bar.
        self.assertEqual(score.floor_of(50, 100)["at_least"], 47)

    def test_rule_1_takes_r3_only_past_its_margin(self):
        cv = {
            "R1 s1": {"kind": "r1", "scaffold": "s1", "baseline": False, "top1": 80.0},
            "R3 s2": {"kind": "r3", "scaffold": "s2", "baseline": False, "top1": 82.9},
        }
        self.assertEqual(score.rule1(cv)[0][0], "R1 s1")
        cv["R3 s2"]["top1"] = 83.0
        self.assertEqual(score.rule1(cv)[0][0], "R3 s2")


if __name__ == "__main__":
    unittest.main()
