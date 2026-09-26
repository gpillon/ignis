"""The generation route's parsing and search (`generation.py`), with no
server.

    python tools/locate-sets/test_generation.py
"""

import unittest

import generation


class GenerationTest(unittest.TestCase):
    def test_a_reply_is_read_with_or_without_a_fence(self):
        self.assertEqual(generation.parse('{"quote": "ch_1a2b3c"}', False), ["ch_1a2b3c"])
        self.assertEqual(generation.parse('```json\n{"quote": "x y"}\n```', False), ["x y"])
        self.assertEqual(generation.parse('Sure: {"quote": "z"} done', False), ["z"])
        self.assertEqual(generation.parse('{"quotes": ["a", "", 3, "b"]}', True), ["a", "b"])
        self.assertEqual(generation.parse('{"quotes": []}', True), [])
        self.assertIsNone(generation.parse("no json here", False))
        self.assertIsNone(generation.parse('{"quote": ""}', False))

    def test_every_exact_occurrence_is_kept_in_segment_coordinates(self):
        texts = ["a job j12 held", "", "j12 again, j12"]
        match, spans = generation.occurrences(texts, "j12")
        self.assertEqual(match, "exact")
        self.assertEqual(spans, [{"segment": 0, "start": 6, "end": 9},
                                 {"segment": 2, "start": 0, "end": 3},
                                 {"segment": 2, "start": 11, "end": 14}])

    def test_the_lenient_search_folds_case_and_space_and_maps_back(self):
        texts = ["The  Black\tForest rises"]
        match, spans = generation.occurrences(texts, "black forest")
        self.assertEqual(match, "lenient")
        self.assertEqual(spans, [{"segment": 0, "start": 5, "end": 17}])
        self.assertEqual(texts[0][5:17], "Black\tForest")
        self.assertEqual(generation.occurrences(texts, "nowhere"), ("none", []))

    def test_the_user_text_is_the_kind_text_then_the_instruction_object(self):
        self.assertEqual(generation.user_text("K.", 'say "hi"'), 'K.\n\n{"instruction":"say \\"hi\\""}')
        self.assertEqual(generation.evidence_text("a\nb"), '{"evidence":"a\\nb"}')


if __name__ == "__main__":
    unittest.main()
