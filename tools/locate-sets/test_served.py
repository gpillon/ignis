"""`served.py judge` on rows whose verdict is known.

The rule compares each family's top-1 rate with D's floor rate, on the
present questions a `locate` serves; a refusal past the ceiling is not a
miss, an absent question is not a question, and an error is a miss.

    python tools/locate-sets/test_served.py
"""

import unittest

import served


def row(i, family, hit=True, absent=False, status=200, code=None, confidence=0.5, split="lexical"):
    targets = [] if absent else [3]
    r = {"id": f"{family}-{i:03}", "family": family, "split": split, "absent": absent, "segments": 10,
         "targets": targets, "status": status, "ms": 100.0 + i, "input_tokens": 1000}
    if status == 200 and code is None:
        segment = 3 if hit and not absent else 7
        r["answer"] = {"type": "locate", "segment": segment, "value": "x", "confidence": confidence,
                       "ranking": [{"segment": segment, "share": confidence}, {"segment": 3, "share": 0.1}]}
    elif status == 200:
        r["answer"] = {"type": "error", "code": code, "message": "m"}
    else:
        r["code"] = code
    return r


def family(name, hits, misses, **extra):
    return [row(i, name, hit=i < hits, **extra) for i in range(hits + misses)]


class Judge(unittest.TestCase):
    def test_a_family_passes_at_the_floor_rate_exactly(self):
        # 82/86 is 41/43 exactly; 81/86 is under it.
        rows = family("logs", 82, 4) + family("records", 45, 0) + family("prose", 67, 0)
        self.assertTrue(served.judge(rows)["families"]["logs"]["pass"])
        rows = family("logs", 81, 5) + family("records", 45, 0) + family("prose", 67, 0)
        report = served.judge(rows)
        self.assertFalse(report["families"]["logs"]["pass"])
        self.assertFalse(report["pass"], "one family under its floor fails the check")

    def test_refusals_past_the_ceiling_are_not_counted(self):
        rows = family("logs", 43, 0) + [row(99, "logs", status=422, code=served.TOO_LONG)]
        rows += family("records", 45, 0) + family("prose", 67, 0)
        report = served.judge(rows)
        self.assertEqual(report["families"]["logs"]["of"], 43)
        self.assertEqual(report["refused_too_long"], {"present": 1, "absent": 0})
        self.assertTrue(report["pass"])

    def test_an_error_is_a_miss_and_an_absent_question_is_not_counted(self):
        rows = family("logs", 43, 0) + [row(98, "logs", status=200, code="attention_unread")]
        rows += [row(97, "logs", absent=True)]
        rows += family("records", 45, 0) + family("prose", 67, 0)
        report = served.judge(rows)
        self.assertEqual(report["families"]["logs"]["of"], 44)
        self.assertEqual(report["families"]["logs"]["hits"], 43)
        self.assertEqual(len(report["errors"]), 1)

    def test_top3_and_the_confidence_auc(self):
        rows = [row(0, "logs", hit=False, confidence=0.9), row(1, "logs", absent=True, confidence=0.2)]
        report = served.judge(rows)
        self.assertEqual(report["top3"], {"hits": 1, "of": 1}, "the target is second in the ranking")
        self.assertEqual(report["auc_present_absent"], 1.0)


if __name__ == "__main__":
    unittest.main()
