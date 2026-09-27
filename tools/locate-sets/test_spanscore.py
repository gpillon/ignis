"""`spanscore.py` on a synthetic span dump whose heads are known.

The dump has the harness's layout (`attention_span_locate_gpu.rs`), one key
per character of the state. Planted heads:

- **inside** (L47.h3, L55.h4, L59.h20) raise every gold key when the
  instruction is there;
- **end** (L51.h9) points, at the forced quote's last token, one key after
  the gold's last;
- **reader** (L23.h1) puts its mass on the instruction's tokens.

The scorer must map a gold character span onto keys through the JSON
escaping, find the gold with every reading, grow the whole gold span, see
the reader's mass on the instruction and the end head after the gold, and
put the generation route on the same grid.

    python tools/locate-sets/test_spanscore.py
"""

import json
import os
import tempfile
import unittest

import numpy as np

import profiles
import score
import spanscore

INSIDE = 11 * score.HEADS + 3      # L47.h3
INSIDES = (INSIDE, 13 * score.HEADS + 4, 14 * score.HEADS + 20)     # L47.h3, L55.h4, L59.h20
END = 12 * score.HEADS + 9         # L51.h9
READER = 5 * score.HEADS + 1       # L23.h1
H = score.N_HEADS
PREFIX, KIND, INSTR, TAIL, SCAFFOLD = 5, 4, 3, 2, 3


def one(rng, i):
    """A state of 4-7 lines of words, a gold value inside one line (after a
    quote character, so the escaping moves it), and its keys."""
    n = int(rng.integers(4, 8))
    lines = [" ".join(f"w{int(rng.integers(0, 99))}" for _ in range(int(rng.integers(3, 6)))) for _ in range(n)]
    t = int(rng.integers(0, n))
    value = f"v{i:03}"
    lines[t] = f'say "x" {value} {lines[t]}'
    start = lines[t].index(value)
    state = "\n".join(lines)
    text, segments = profiles.evidence(state)
    key_bytes, keys = [], []
    for j, (a, b, _) in enumerate(segments):
        if j:
            key_bytes.append([a - 2, a])            # the \n escape, a separator key
        first = len(key_bytes)
        key_bytes += [[c, c + 1] for c in range(a, b)]
        keys.append([first, len(key_bytes)])
    return state, lines, t, {"segment": t, "start": start, "end": start + len(value)}, key_bytes, keys


def write_dump(directory, name, n, seed):
    rng = np.random.default_rng(seed)
    rows, questions, blocks, end = [], [], [], 0
    for i in range(n):
        absent = i % 6 == 5
        state, lines, t, span, key_bytes, keys = one(rng, i)
        S = len(key_bytes)
        T = PREFIX + S + KIND + INSTR + TAIL + SCAFFOLD
        s0 = PREFIX
        regions = [[0, PREFIX, "template"], [PREFIX, PREFIX + S, "evidence"],
                   [PREFIX + S, PREFIX + S + KIND, "kind"], [PREFIX + S + KIND, PREFIX + S + KIND + INSTR, "instruction"],
                   [PREFIX + S + KIND + INSTR, T - SCAFFOLD, "template"]]
        gold_keys = []
        if not absent:
            mask, each = spanscore.span_keys(state, [span], key_bytes)
            gold_keys = each[0]
        q = rng.normal(0, 1, size=(H, T)).astype(np.float32)
        na = rng.normal(0, 1, size=(H, T)).astype(np.float32)
        instr = slice(PREFIX + S + KIND, PREFIX + S + KIND + INSTR)
        q[READER, instr] += 12.0
        if len(gold_keys):
            for h in INSIDES:
                q[h, s0 + gold_keys] += 8.0
        w = np.full((H, PREFIX + S + KIND + INSTR), 1.0 / (PREFIX + S + KIND + INSTR), dtype=np.float32)
        forced = []
        if not absent:
            for qi in range(4):
                f = rng.normal(0, 1, size=(H, S)).astype(np.float32)
                if qi == 3:
                    f[END, min(S - 1, gold_keys[-1] + 1)] += 12.0
                forced.append(f)
        offsets = {}
        for key, arr in (("q", q), ("w", w), ("na", na)):
            offsets[key] = end
            blocks.append(arr.astype("<f2").ravel())
            end += arr.size
        f_offsets = []
        for f in forced:
            f_offsets.append(end)
            blocks.append(f.astype("<f2").ravel())
            end += f.size
        qid = f"logvalue-{i:03}"
        spans = [] if absent else [span]
        rows.append({
            "id": qid, "family": "logvalue", "split": ("lexical", "paraphrase")[i % 2], "absent": absent,
            "unit": "line", "segments": len(lines), "targets": [] if absent else [t], "spans": spans,
            "quote": None if absent else f"v{i:03}", "span": [s0, S], "keys": keys, "key_bytes": key_bytes,
            "regions": regions, "scaffold": [T - SCAFFOLD, T],
            "q": {"scores_offset": offsets["q"], "keys": T, "weights_offset": offsets["w"],
                  "weight_keys": PREFIX + S + KIND + INSTR},
            "q-na": {"scores_offset": offsets["na"], "keys": T, "regions": regions},
            "q-forced": None if absent else {"queries": [T - 1, T, T + 1, T + 3], "scores_offsets": f_offsets,
                                             "keys": S, "quote_tokens": 4},
            "end": end,
        })
        questions.append({"id": qid, "family": "logvalue", "split": rows[-1]["split"], "absent": absent,
                          "state": state, "spans": spans})
    np.concatenate(blocks).tofile(os.path.join(directory, f"{name}.bin"))
    with open(os.path.join(directory, f"{name}.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    dump = os.path.join(directory, f"{name}.json")
    spanscore.write_json(dump, {"set": name, "rows_file": f"{name}.jsonl", "bin_file": f"{name}.bin"})
    manifest = os.path.join(directory, f"{name}-manifest.json")
    spanscore.write_json(manifest, {"questions": questions})
    return dump, manifest, questions


class SpanKeysTest(unittest.TestCase):
    def test_a_gold_after_an_escaped_quote_lands_on_its_own_keys(self):
        state = 'ab\nsay "x" v1 z'
        text, segments = profiles.evidence(state)
        key_bytes = [[c, c + 1] for c in range(len(text.encode()))]
        mask, each = spanscore.span_keys(state, [{"segment": 1, "start": 8, "end": 10}], key_bytes)
        self.assertEqual("".join(text[k] for k in each[0]), "v1")

    def test_an_element_is_its_own_json(self):
        state = [{"id": 7, "name": "é"}, {"id": 8}]
        text, _ = profiles.evidence(state)
        raw = text.encode()
        key_bytes = [[b, b + 1] for b in range(len(raw))]
        element = json.dumps(state[0], ensure_ascii=False, separators=(",", ":"))
        at = element.index("é")
        mask, each = spanscore.span_keys(state, [{"segment": 0, "start": at, "end": at + 1}], key_bytes)
        self.assertEqual(raw[each[0][0]:each[0][-1] + 1].decode(), "é")


class SpanScoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        d = cls.dir.name
        sets, dumps, manifests, gens = [], [], [], []
        for name, seed in (("E1", 1), ("E2", 2)):
            dump, manifest, questions = write_dump(d, name, 30, seed)
            spanscore.extract(dump, manifest, os.path.join(d, "x"))
            sets.append(os.path.join(d, "x", name))
            dumps.append(dump)
            manifests.append(manifest)
            # the generation route answers every present question with its gold
            gen = [{"id": q["id"], "found": [{"match": "exact", "spans": q["spans"][:1]}] if q["spans"] else []}
                   for q in questions]
            gens.append(os.path.join(d, f"{name}-generation.json"))
            spanscore.write_json(gens[-1], {"questions": gen})
        cls.rows = spanscore.load_all(sets, dumps, manifests, gens)

    @classmethod
    def tearDownClass(cls):
        cls.rows = None          # the dumps are memory-mapped
        cls.dir.cleanup()

    def test_every_reading_finds_the_gold_key(self):
        cv = spanscore.cv_readings(self.rows)
        for name in ("head lift K=1", "vote lift K=3 w=1", "sum lift K=3", "segment lift K=3"):
            self.assertEqual(cv[name]["hits"], cv[name]["of"], name)

    def test_the_span_grows_to_the_whole_gold(self):
        choice = spanscore.span_choice(self.rows, spanscore.cv_readings(self.rows))
        self.assertGreater(max(d["f1"] for d in choice["span_by_delta"].values()), 90.0)

    def test_the_reader_reads_the_instruction_and_the_end_head_looks_past_the_gold(self):
        table = spanscore.regions_table(self.rows, [READER])
        self.assertGreater(table["lexical"]["q"]["instruction"], 0.9)
        forced = spanscore.forced_table(self.rows)
        self.assertEqual(forced["last"]["after_last"][0], (score.head_name(END), 1.0))

    def test_a_set_answer_is_scored_against_the_gold_set(self):
        self.assertEqual(spanscore.set_scores({1, 2}, {2, 3})[:3], (0.5, 0.5, 0.5))
        self.assertEqual(spanscore.set_scores(set(), set()), (1.0, 1.0, 1.0, True))
        self.assertEqual(spanscore.set_scores({4}, set())[3], False)
        row = {"x": {"seg_top": np.array([[3, 0], [3, 1], [5, 0], [7, 0]])}}
        self.assertEqual(spanscore.multi_answer(row, np.array([0, 1, 2, 3]), 0.5), {3})
        self.assertEqual(spanscore.multi_answer(row, np.array([0, 1, 2, 3]), 0.25), {3, 5, 7})

    def test_the_line_level_winner_of_the_inside_heads_is_the_gold_segment(self):
        hits = spanscore.line_hits([r for r in self.rows if not r["absent"]])
        self.assertTrue(all(hits[:, h].all() for h in INSIDES))

    def test_a_forced_query_is_named_by_its_place_in_the_quote(self):
        self.assertEqual([spanscore.forced_label(p, 4) for p in (0, 1, 2, 4)], ["scaffold", "first", "middle", "last"])
        self.assertEqual([spanscore.forced_label(p, 1) for p in (0, 1)], ["scaffold", "last"])

    def test_the_generation_route_is_scored_on_the_same_keys(self):
        gen = spanscore.generation_on_keys(self.rows)
        self.assertEqual(gen["all"]["hit"], 100.0)
        self.assertEqual(gen["all"]["em"], 100.0)


if __name__ == "__main__":
    unittest.main()
