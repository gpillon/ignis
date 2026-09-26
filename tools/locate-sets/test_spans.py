"""The span sets' promises (`spans.py`), with no download: SQuAD and HotpotQA
are replaced by a few hand-written rows.

- every gold span's text is the value the question asks for;
- a question is absent exactly when it has no gold;
- the lexical / paraphrase split holds on the segment holding each gold;
- a seed writes the same set.

    python tools/locate-sets/test_spans.py
"""

import json
import random
import re
import unittest
from unittest import mock

import prose
import records
import spans
from common import paraphrase_clean, rare_shared

SQUAD = [{"title": "Rivers", "paragraphs": [
    {"context": "The Danube flows through ten countries. It rises in the Black Forest.",
     "qas": [{"id": "q1", "question": "Where does the Danube rise?", "is_impossible": False,
              "answers": [{"text": "the Black Forest", "answer_start": 52},
                          {"text": "Black Forest", "answer_start": 56}]},
             {"id": "q5", "question": "Through how many countries does the Danube flow?", "is_impossible": False,
              "answers": [{"text": "ten", "answer_start": 25}]},
             {"id": "q2", "question": "Where does the Danube end in a lake?", "is_impossible": True,
              "answers": [], "plausible_answers": [{"text": "Black Forest", "answer_start": 56}]}]},
    {"context": "Its delta lies on the coast of Romania and is a wetland reserve.",
     "qas": [{"id": "q3", "question": "What kind of protected area is at the river's mouth?", "is_impossible": False,
              "answers": [{"text": "a wetland reserve", "answer_start": 46}]},
             {"id": "q4", "question": "Which fish breed in the reserve?", "is_impossible": True,
              "answers": []}]},
]}]


def hotpot(n):
    """HotpotQA-shaped rows: even ones ask with the entity's name (lexical),
    odd ones without it (paraphrase)."""
    rows = []
    for k in range(n):
        question = (f"Which city did the founder of Zorbex{k} move to?" if k % 2 == 0
                    else "Where did the person who started the company relocate?")
        rows.append({
            "_id": f"h{k}", "question": question, "answer": f"Quelport{k}",
            "context": [[f"Zorbex{k}", [f"Zorbex{k} was founded in 19{k:02}.", f"Its founder moved to Quelport{k}."]],
                        [f"Other{k}", ["An unrelated sentence.", "Another unrelated one."]]],
            "supporting_facts": [[f"Zorbex{k}", 0], [f"Zorbex{k}", 1]],
        })
    return rows


class SpanSetTest(unittest.TestCase):
    def check_split(self, q):
        lines = q["state"].split("\n") if isinstance(q["state"], str) else [records._text(x) for x in q["state"]]
        for t in q["targets"]:
            rest = [line for j, line in enumerate(lines) if j not in q["targets"]]
            if q["split"] == "lexical":
                self.assertTrue(rare_shared(q["instruction"], lines[t], rest), q["id"])
            else:
                self.assertTrue(paraphrase_clean(q["instruction"], lines[t]), q["id"])

    def test_log_values_are_the_asked_value_and_absent_means_no_gold(self):
        rows = spans.logvalue(random.Random("t/logvalue"), 12)
        self.assertEqual(sum(q["absent"] for q in rows), 2)
        for q in rows:
            self.assertEqual(q["absent"], not q["spans"])
            for s in q["spans"]:
                text = spans.text_of(q["state"], s)
                line = q["state"].split("\n")[s["segment"]]
                found = re.search(spans.LOG_VALUES[q["event"]][0], line)
                self.assertEqual((found.start(), found.end()), (s["start"], s["end"]), q["id"])
                self.assertEqual(q["quote"], text)
            if not q["absent"]:
                self.check_split(q)

    def test_record_values_are_the_records_id_or_sku(self):
        rows = spans.recvalue(random.Random("t/recvalue"), 18)
        for q in rows:
            self.assertEqual(q["absent"], not q["spans"])
            for s in q["spans"]:
                record = q["state"][s["segment"]]
                field = "sku" if q["kind"] == "products" else "id"
                self.assertEqual(spans.text_of(q["state"], s), str(record[field]))
            if not q["absent"]:
                self.check_split(q)

    def test_multi_line_questions_have_zero_to_three_whole_line_golds(self):
        rows = spans.logmulti(random.Random("t/logmulti"), 8)
        self.assertEqual([len(q["spans"]) for q in rows], [0, 1, 2, 3] * 2)
        for q in rows:
            lines = q["state"].split("\n")
            self.assertEqual(q["absent"], q["matches"] == 0)
            for s in q["spans"]:
                self.assertEqual((s["start"], s["end"]), (0, len(lines[s["segment"]])))
            if q["spans"]:
                self.check_split(q)

    def test_squad_golds_are_its_answers_and_unanswerable_is_absent(self):
        with mock.patch.object(spans, "load_squad", return_value=SQUAD):
            rows = spans.squad(random.Random("t/squad"), 3, "unused")
        for q in rows:
            self.assertEqual(q["absent"], not q["spans"])
            for s in q["spans"]:
                self.assertIn(spans.text_of(q["state"], s), ("the Black Forest", "Black Forest", "a wetland reserve", "ten"))
        # two answers of one question, two golds
        danube = [q for q in rows if q["event"] == "q1"]
        if danube:
            self.assertEqual(len(danube[0]["spans"]), 2)

    def test_hotpot_spans_sit_in_gold_sentences_and_an_absent_answer_is_gone(self):
        with mock.patch.object(prose, "load", return_value=hotpot(30)):
            rows = spans.hotspan(random.Random("t/hotspan"), 12, "unused")
            facts = spans.hotfacts(random.Random("t/hotfacts"), 6, "unused")
        for q in rows:
            if q["absent"]:
                self.assertNotIn(q["answer"], q["state"])
            for s in q["spans"]:
                self.assertEqual(spans.text_of(q["state"], s), q["answer"])
        for q in facts:
            self.assertEqual(q["absent"], not q["spans"])
            if not q["absent"]:
                self.assertEqual(len(q["spans"]), 2)

    def test_a_seed_writes_the_same_set(self):
        a = spans.logvalue(random.Random("s/logvalue"), 6) + spans.recvalue(random.Random("s/recvalue"), 6)
        b = spans.logvalue(random.Random("s/logvalue"), 6) + spans.recvalue(random.Random("s/recvalue"), 6)
        self.assertEqual(json.dumps(a), json.dumps(b))


if __name__ == "__main__":
    unittest.main()
