"""`r3_judge.py` on runs whose verdict is known (spec 22 § Scoring, § The
rules): a present answer is right only when it names its target and keeps
`found` at 0.5 or more; an absent one is flagged below 0.5; the pick under
0.5 is the ranking's first; the paragraph F1 of a "not found" is 0.

    python tools/locate-sets/test_r3_judge.py
"""

import json
import os
import sys
import tempfile
import unittest

import r3_judge as J


def answer(segment, found=0.9, kind="log", pointers=None, confidence=0.8, method="shortlist"):
    named = found is None or found >= 0.5
    a = {"type": "locate", "kind": kind, "method": method, "compression": "none",
         "segment": segment if named else None, "value": "x" if named else None,
         "confidence": confidence if named else None,
         "ranking": [{"segment": segment, "share": confidence}, {"segment": 99, "share": 0.1}],
         "pointers": ([{"segment": s, "value": "x", "share": 0.5} for s in (pointers or [segment])] if named else [])}
    if found is not None:
        a["found"] = found
    return a


def row(qid, targets, a, **extra):
    return dict({"id": qid, "status": 200, "ms": 1000.0, "targets": targets, "absent": not targets, "answer": a,
                 "input_tokens": 100, "first_of_state": True}, **extra)


class Scoring(unittest.TestCase):
    def test_a_present_answer_is_right_only_when_found(self):
        self.assertTrue(J.right(row("a", [3], answer(3, 0.9))))
        self.assertFalse(J.right(row("a", [3], answer(3, 0.4))), "a not-found answer is a miss")
        self.assertFalse(J.right(row("a", [3], answer(4, 0.9))))
        self.assertTrue(J.right(row("a", [3], answer(3, None))), "a route without found names its pick")

    def test_the_pick_under_the_threshold_is_the_rankings_first(self):
        self.assertEqual(J.pick_of(row("a", [3], answer(3, 0.2))), 3)
        self.assertEqual(J.pick_of(row("a", [3], answer(5, 0.9))), 5)

    def test_an_absent_question_is_flagged_below_the_threshold(self):
        self.assertTrue(J.flagged(row("a", [], answer(3, 0.49))))
        self.assertFalse(J.flagged(row("a", [], answer(3, 0.5))))
        self.assertFalse(J.flagged(row("a", [], answer(3, None))), "no found flags nothing")

    def test_paragraph_f1(self):
        lines = ["# A", "a1", "a2", "", "# B", "b1", "", "# C", "c1"]
        both = row("p", [1, 5], answer(1, 0.9, pointers=[1, 5]))
        self.assertEqual(J.paragraph_f1(both, lines), 1.0)
        one = row("p", [1, 5], answer(2, 0.9, pointers=[2]))
        self.assertAlmostEqual(J.paragraph_f1(one, lines), 2 * 1.0 * 0.5 / 1.5)
        none = row("p", [1, 5], answer(1, 0.3))
        self.assertEqual(J.paragraph_f1(none, lines), 0.0, "not found has no pointers")


class Rules(unittest.TestCase):
    def run_judge(self, files):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, payload in files.items():
                paths[name] = os.path.join(tmp, name + ".json")
                with open(paths[name], "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            out = os.path.join(tmp, "out.json")
            argv = ["r3_judge.py"] + [x for name, path in paths.items() for x in (f"--{name}", path)] + ["--out", out]
            old, sys.argv = sys.argv, argv
            try:
                J.main()
            finally:
                sys.argv = old
            return json.load(open(out, encoding="utf-8"))

    def files(self, r3_right=18, prose_right=True):
        r3 = []
        for i in range(20):
            hit = i < r3_right
            r3.append(row(f"r{i}", [3], answer(3 if hit else 4, 0.9), variant="present", tier=100_000,
                          source="cluster", bin=0))
            r3.append(row(f"r{i}~removed", [], answer(3, 0.2), variant="removed", tier=100_000, first_of_state=True))
        r3.append(row("ra", [], answer(3, 0.1), variant="authored", tier=100_000, first_of_state=False))
        choice = {"questions": [{"id": f"r{i}", "hit": i < 15} for i in range(20)]}
        lines = ["# A", "a1", "", "# B", "b1"]
        p3m = {"questions": [{"id": f"p{i}", "state": "\n".join(lines)} for i in range(10)]}
        p3 = [row(f"p{i}", [1], answer(1 if prose_right else 4, 0.9, kind="prose"), segments=16_000) for i in range(10)]
        p3abs = [row(f"x{i}~del", [], answer(1, 0.1, kind="prose"), variant="deleted", segments=16_000) for i in range(4)]
        p3abs.append(row("y@w", [], answer(1, 0.1, kind="prose"), variant="cross", segments=200_000))
        j3 = [row(f"j{i}", [2], answer(2, 0.95, kind="records"), segments=10_000) for i in range(10)]
        j3 += [row("ja", [], answer(2, 0.1, kind="records"), segments=10_000)]
        recorded = [{"id": "f1", "status": 200, "absent": False, "targets": [2], "family": "logs",
                     "answer": {"segment": 2, "ranking": [{"segment": 2, "share": 0.5}, {"segment": 1, "share": 0.1}]}},
                    {"id": "f2", "status": 422, "code": "locate_too_long", "absent": False, "targets": [1], "family": "logs"}]
        vote = [row("f1", [2], answer(2, None, method="vote"), family="logs"),
                {"id": "f2", "status": 422, "code": "locate_too_long", "absent": False, "targets": [1], "family": "logs"}]
        defaults = [row("f1", [2], answer(2, 0.9, kind="log"), family="logs")]
        return {"r3": {"questions": r3}, "r3-choice": choice, "p3": {"questions": p3}, "p3abs": {"questions": p3abs},
                "p3-manifest": p3m, "j3": {"questions": j3}, "f-vote": {"questions": vote},
                "f-defaults": {"questions": defaults}, "f-recorded": {"questions": recorded}}

    def test_rules_on_a_run_that_passes_them(self):
        out = self.run_judge(self.files())
        rules = out["rules"]
        for name in ("1 logs", "2 the heads' part", "3 latency", "4 prose", "5 records", "6 not found, logs",
                     "7 not found, prose", "8 not found, records", "9 auto", "10 the vote unchanged"):
            self.assertTrue(rules[name]["pass_"], (name, rules[name]))
        self.assertEqual(rules["1 logs"]["hits"], 18)
        self.assertEqual(rules["2 the heads' part"]["choice_alone"], 15)
        # Set F's floors are counted against the recorded families (one
        # served log question here): 1 is not 39.
        self.assertFalse(rules["11 short states"]["pass_"])

    def test_a_failed_rule_is_reported_failed(self):
        out = self.run_judge(self.files(r3_right=16, prose_right=False))
        self.assertFalse(out["rules"]["1 logs"]["pass_"], "16/20 is under 85%")
        self.assertFalse(out["rules"]["4 prose"]["pass_"])
        self.assertFalse(out["all_pass"])


if __name__ == "__main__":
    unittest.main()
