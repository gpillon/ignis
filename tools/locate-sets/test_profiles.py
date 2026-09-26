"""`profiles.py` on a synthetic dump whose heads are known.

Four planted heads, every other one noise:

- an **initiator** (L27.h5) puts its peak on the target's first key, and only
  when the instruction is there (not in the `-na` prefills);
- a **terminator** (L43.h7) puts its peak on the next segment's first key;
- a **prior** (L11.h2) peaks on the first key of the state in every prefill,
  question or not;
- an **inside** head (L55.h3) raises every key of the target.

Phase 0's tables must find each of them where it was planted: the token
categories, the span-aligned averages, the line offset (+1 for the
terminator, recovered by the offset reading), the lift's AUROC against the
prior's, and a combination over heads that finds every target. The evidence
writer and the key map are held to hand-written cases.

    python tools/locate-sets/test_profiles.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

import profiles
import score

INIT = 6 * score.HEADS + 5        # L27.h5
TERM = 10 * score.HEADS + 7       # L43.h7
PRIOR = 2 * score.HEADS + 2       # L11.h2
INSIDE = 13 * score.HEADS + 3     # L55.h3


def write_dump(directory, name, n, seed):
    """A dump in the harness's layout (like `test_score.write_dump`) with the
    four planted heads; and a manifest with each question's segment texts."""
    rng = np.random.default_rng(seed)
    rows, blocks, questions, end = [], [], [], 0
    for i in range(n):
        segments = int(rng.integers(8, 30))
        keys, at = [], 0
        for j in range(segments):
            width = int(rng.integers(3, 9))
            keys.append([at, at + width])
            at += width + 1
        span = at - 1
        absent = i % 6 == 5
        target = int(rng.integers(1, segments - 1))
        family = ("logs", "records")[i % 2]
        for v in score.VARIANTS:
            s = rng.normal(0, 1, size=(score.N_HEADS, span)).astype(np.float32)
            s[PRIOR, 0] += 9.0
            if not absent and not v.endswith("-na"):
                a, b = keys[target]
                s[INIT, a] += 9.0
                s[TERM, keys[target + 1][0]] += 9.0
                s[INSIDE, a:b] += 4.0
            blocks.append(s.astype("<f2").ravel())
            end += s.size
        variants = {v: {"offset": 0, "prompt_tokens": span + 60} for v in score.VARIANTS}
        off = end - 4 * score.N_HEADS * span
        for k, v in enumerate(score.VARIANTS):
            variants[v]["offset"] = off + k * score.N_HEADS * span
        texts = [f"line {j} routine" for j in range(segments)]
        rows.append({
            "id": f"{family}-{i:03}", "family": family, "split": ("lexical", "paraphrase")[(i // 2) % 2],
            "absent": absent, "segments": segments, "span": [10, span],
            "targets": [] if absent else [target], "distractors": [], "keys": keys, "variants": variants,
            "l39_h10": {"s1": 0, "s2": 0},
        })
        questions.append({"id": rows[-1]["id"], "instruction": "Which line is the one?",
                          "state": "\n".join(texts)})
    np.concatenate(blocks).tofile(os.path.join(directory, f"{name}.bin"))
    with open(os.path.join(directory, f"{name}.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    meta = {"set": name, "rows_file": f"{name}.jsonl", "bin_file": f"{name}.bin", "alphabet": []}
    path = os.path.join(directory, f"{name}.json")
    score.write_json(path, meta)
    manifest = os.path.join(directory, f"{name}-manifest.json")
    score.write_json(manifest, {"questions": questions})
    return path, manifest


class EvidenceTest(unittest.TestCase):
    def test_lines_are_written_between_escapes_and_items_without_commas(self):
        text, segs = profiles.evidence('a "b"\n\nc')
        self.assertEqual(text, '{"evidence":"a \\"b\\"\\n\\nc"}')
        self.assertEqual([text[a:b] for a, b, _ in segs], ['a \\"b\\"', "", "c"])
        self.assertEqual([o for _, _, o in segs], [True, False, True])
        text, segs = profiles.evidence([{"id": 1, "x": "é"}, "s"])
        self.assertEqual(text, '{"evidence":[{"id":1,"x":"é"},"s"]}')
        self.assertEqual([text[a:b] for a, b, _ in segs], ['{"id":1,"x":"é"}', "s"])

    def test_keys_are_the_owned_tokens_and_separators_own_nothing(self):
        # three tokens "ab", "\\n", "cd" over segments "ab" and "cd"
        segs = [(0, 2, True), (4, 6, True)]
        span, keys = profiles.map_keys([(0, 2), (2, 4), (4, 6)], segs, 0)
        self.assertEqual(span, [0, 3])
        self.assertEqual(keys, [[0, 1], [2, 3]])
        # a token straddling two segments goes to the one holding more of it
        span, keys = profiles.map_keys([(0, 1), (1, 5), (5, 6)], [(0, 2, True), (3, 6, True)], 0)
        self.assertEqual(keys, [[0, 1], [1, 3]])


class AurocTest(unittest.TestCase):
    def test_a_perfect_ordinate_scores_one_and_its_reverse_zero(self):
        pos = np.array([False, True, True, False, False])
        y = np.array([[0.0, 5.0, 4.0, 1.0, 2.0], [5.0, 0.0, 1.0, 4.0, 3.0]], dtype=np.float32)
        auc, ap = profiles.auroc_ap(y, pos)
        np.testing.assert_allclose(auc, [1.0, 0.0])
        self.assertAlmostEqual(ap[0], 1.0)
        # positives at ranks 4 and 5 of 5: precision 1/4 and 2/5
        self.assertAlmostEqual(ap[1], (1 / 4 + 2 / 5) / 2, places=6)

    def test_ties_count_half(self):
        pos = np.array([True, False] * 200)
        y = np.ones((1, 400), dtype=np.float32)
        auc, _ = profiles.auroc_ap(y, pos)
        self.assertAlmostEqual(float(auc[0]), 0.5, delta=0.06)


class Phase0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        sets, dumps, manifests = [], [], []
        for name, seed in (("A", 1), ("B", 2)):
            dump, manifest = write_dump(d, name, 36, seed)
            profiles.extract(dump, manifest, None, os.path.join(d, "x"))
            sets.append(os.path.join(d, "x", name))
            dumps.append(dump)
            manifests.append(manifest)
        rows = profiles.load_all(sets, dumps, manifests)
        cls.present = [r for r in rows if not r["absent"]]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_planted_heads_land_in_their_categories(self):
        peaks = profiles.peak_places(self.present)["s2"]
        self.assertEqual(peaks["top"]["first"][0], (score.head_name(INIT), 1.0))
        self.assertEqual(peaks["top"]["next_first"][0], (score.head_name(TERM), 1.0))

    def test_span_aligned_averages_find_initiator_and_terminator(self):
        era = profiles.span_aligned(self.present)["logs+records"]["ordinates"]["lift2"]
        self.assertEqual(era["initiators"][0][0], score.head_name(INIT))
        self.assertEqual(era["terminators"][0][0], score.head_name(TERM))

    def test_the_lift_beats_the_prior_and_the_prior_localises_nothing(self):
        q = profiles.quality(self.present)
        self.assertEqual(q["lift2"]["ap_top"][0][0], score.head_name(INSIDE))
        self.assertGreater(q["lift2"]["by"]["all"]["auc"], 0.95)
        self.assertLess(q["na2"]["cv_token_hit"]["hits"], len(self.present) // 3)
        self.assertEqual(q["s2"]["cv_token_hit"]["hits"], len(self.present))

    def test_the_offset_reading_uses_the_terminator_one_segment_late(self):
        off = profiles.line_offsets(self.present)[score.config_name("r1", "s2", True)]
        self.assertIn((score.head_name(TERM), 1, 1.0), off["stable_nonzero"])
        self.assertEqual(off["cv_hits"], len(self.present))

    def test_a_vote_of_the_best_heads_finds_every_target(self):
        v = profiles.votes(self.present)
        self.assertEqual(v["nested_hits"], len(self.present))
        # the fitted reading's best voter is a head that finds the target
        self.assertIn(v["final"]["heads"][0], (score.head_name(INIT), score.head_name(INSIDE)))

    def test_a_vote_tie_goes_to_the_best_head(self):
        winners = np.array([4, 7, 7, 4, 9])
        self.assertEqual(profiles.vote_answer(winners, np.array([0, 1, 2, 3])), 4)
        self.assertEqual(profiles.vote_answer(winners, np.array([4, 1, 2, 3, 0])), 7)
        self.assertEqual(profiles.vote_answer(winners, np.array([4, 1, 0])), 9)

    def test_a_combination_over_heads_finds_every_target(self):
        comb = profiles.combinations(self.present, kinds=("lift2",))["lift2"]
        self.assertEqual(comb["hits"], len(self.present))
        self.assertEqual(comb["recall"][1], 100.0)


if __name__ == "__main__":
    unittest.main()
