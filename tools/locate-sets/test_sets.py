"""The set generators and the labelled route's state, on what a set promises.

Every question of a set is lexical (a rare word shared with its target),
paraphrase (no content word shared) or absent (target removed), one in six
absent, and the same seed writes the same set. These hold the generators to
that without the HotpotQA download (prose is checked on a synthetic example),
and the labelled route to labelling exactly the segments it maps back.

    python tools/locate-sets/test_sets.py
"""

import random
import unittest

import common
import labelled
import logs
import prose
import records


class WordRulesTest(unittest.TestCase):
    def test_content_words_ignore_the_question_frame_and_suffixes(self):
        self.assertEqual(common.content_stems("Which line reports a timeout?"), {"timeo"})
        self.assertEqual(common.content_stems("rejected"), common.content_stems("rejection"))
        # Digits are content however short.
        self.assertEqual(common.content_stems("status 7"), {"statu", "7"})

    def test_a_rare_word_is_one_no_other_segment_has(self):
        target = "ERROR billing: checksum mismatch on block 7"
        others = ["INFO billing: checksum ok", "ERROR edge: disk full"]
        self.assertEqual(common.rare_shared("Which line reports a checksum mismatch?", target, others), {"misma"})
        self.assertEqual(common.rare_shared("Which line reports a checksum?", target, others), set())

    def test_a_paraphrase_shares_no_content_word(self):
        self.assertTrue(common.paraphrase_clean("Which line says stored data failed verification?",
                                                "checksum mismatch on block 7"))
        self.assertFalse(common.paraphrase_clean("Which line says a block was bad?", "checksum mismatch on block 7"))

    def test_one_question_in_six_is_absent(self):
        self.assertEqual(len(common.assign_absent(random.Random(1), 80)), 13)


class GeneratorTest(unittest.TestCase):
    def check_family(self, module, rows):
        self.assertEqual({r["split"] for r in rows}, {"lexical", "paraphrase"})
        for r in rows:
            state = r["state"]
            segments = state.split("\n") if isinstance(state, str) else [str(x) for x in state]
            self.assertEqual(len(segments), r["segments"], r["id"])
            self.assertIn(r["segments"], module.LENGTHS, r["id"])
            if r["absent"]:
                self.assertEqual(r["targets"], [], r["id"])
                continue
            (t,) = r["targets"]
            target = segments[t]
            if isinstance(state, list):
                target = records._text(state[t])
                segments = [records._text(x) for x in state]
            rest = segments[:t] + segments[t + 1:]
            if r["split"] == "lexical":
                self.assertTrue(common.rare_shared(r["instruction"], target, rest), r["id"])
            else:
                self.assertTrue(common.paraphrase_clean(r["instruction"], target), r["id"])

    def test_logs_keep_their_split_and_their_distractors(self):
        rows = logs.generate(random.Random("t/logs"), 12)
        self.check_family(logs, rows)
        for r in rows:
            lines = r["state"].split("\n")
            self.assertTrue(3 <= len(r["distractors"]) <= 8, r["id"])
            self.assertTrue(all(" ERROR " in lines[d] for d in r["distractors"]), r["id"])
            if not r["absent"]:
                self.assertIn(" ERROR ", lines[r["targets"][0]])

    def test_records_keep_their_split_and_are_never_content_parts(self):
        rows = records.generate(random.Random("t/records"), 12)
        self.check_family(records, rows)
        for r in rows:
            self.assertTrue(all(isinstance(x, dict) and "type" not in x for x in r["state"]), r["id"])
            self.assertTrue(all(5 <= len(x) <= 8 for x in r["state"]), r["id"])

    def test_a_seed_writes_the_same_questions(self):
        first = logs.generate(random.Random("s"), 6) + records.generate(random.Random("s"), 6)
        again = logs.generate(random.Random("s"), 6) + records.generate(random.Random("s"), 6)
        self.assertEqual(first, again)


class ProseTest(unittest.TestCase):
    EXAMPLE = {
        "_id": "x1",
        "question": "Which river flows through Brno?",
        "context": [["Brno", ["Brno is a city in Moravia.", " The Svratka flows through it."]],
                    ["Other", ["Rivers are long.", " Some are short."]]],
        "supporting_facts": [["Brno", 1]],
    }

    def test_paragraphs_become_lines_with_an_empty_line_between(self):
        lines, gold = prose.as_lines(self.EXAMPLE)
        self.assertEqual(lines, ["Brno is a city in Moravia.", "The Svratka flows through it.", "",
                                 "Rivers are long.", "Some are short."])
        self.assertEqual(gold, [1])

    def test_a_rare_shared_word_makes_a_question_lexical(self):
        lines, gold = prose.as_lines(self.EXAMPLE)
        self.assertEqual(prose.classify(self.EXAMPLE["question"], lines, gold), "lexical")
        self.assertEqual(prose.classify("Which water course is in Moravia?", lines, gold), "paraphrase")


class LabelledStateTest(unittest.TestCase):
    ALPHABET = ["A", "B", "C", "D"]

    def test_content_lines_are_labelled_and_empty_ones_kept(self):
        state, mapping = labelled.labelled_state("one\n\ntwo\n  \nthree", self.ALPHABET)
        self.assertEqual(state, "A: one\n\nB: two\n  \nC: three")
        self.assertEqual(mapping, {"A": 0, "B": 2, "C": 4})
        self.assertEqual(labelled.owning("one\n\ntwo\n  \nthree"), 3)

    def test_an_array_becomes_an_object_in_order(self):
        state, mapping = labelled.labelled_state([{"k": 2}, {"k": 1}], self.ALPHABET)
        self.assertEqual(list(state.items()), [("A", {"k": 2}), ("B", {"k": 1})])
        self.assertEqual(mapping, {"A": 0, "B": 1})


if __name__ == "__main__":
    unittest.main()
