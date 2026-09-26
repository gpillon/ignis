"""HotpotQA distractor dev as `locate` questions (spec 18, family `prose`).

HotpotQA (Yang et al., 2018; CC BY-SA 4.0) pairs each question with ten
Wikipedia paragraphs, two of which hold its gold **supporting-fact**
sentences. Here the state is those paragraphs as one string, one line per
sentence, paragraphs separated by an empty line (an empty segment owns no key
and keeps its index). The instruction asks for a sentence that helps answer
the question, and top-1 is correct on any gold supporting sentence.

The dev file is downloaded into `.scratch/` and never committed. The official
URL is tried first; when it does not answer (it timed out on 2026-09-26), the
same 7,405 questions are read from the Hugging Face copy of the dataset
(`hotpotqa/hotpot_qa`, config `distractor`, split `validation`).

A set's prose questions depend on the file's **row order** as well as the
seed. Sets A-D were drawn from the Hugging Face copy, sha256
`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`; a draw from
the official file reproduces them only if its rows come in the same order,
which could not be checked while that URL was down.

The split is **classified**, not constructed: a question is lexical when it
shares a rare word (one no other line has) with a gold sentence, and
paraphrase otherwise -- it may share common words with its gold sentences,
but none that only they carry. That is weaker than the other families'
paraphrase (no content word shared at all), and it has to be: HotpotQA's
questions name their entities, and only 12 of the 7,405 share no content word
with any gold sentence. What it still guarantees is the property the split
exists for -- a rare-string matcher cannot single the gold sentence out. A
set draws 40 of each. An absent question has its gold sentences removed.
"""

import json
import os
import urllib.request

from common import assign_absent, nfc, rare_shared

OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MIRROR_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
              "distractor/validation-00000-of-00001.parquet")
INSTRUCTION = "Which sentence helps answer this question: {question}"


def _fetch(url, path, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response, open(path + ".part", "wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(path + ".part", path)


def load(cache_dir):
    """The dev set as HotpotQA's own JSON rows, downloading it on first use."""
    os.makedirs(cache_dir, exist_ok=True)
    official = os.path.join(cache_dir, "hotpot_dev_distractor_v1.json")
    mirror = os.path.join(cache_dir, "hotpot_dev_distractor.parquet")
    if not os.path.exists(official) and not os.path.exists(mirror):
        try:
            _fetch(OFFICIAL_URL, official, timeout=30)
        except OSError as error:
            print(f"prose: the official URL failed ({error}); using the Hugging Face copy")
            _fetch(MIRROR_URL, mirror, timeout=300)
    if os.path.exists(official):
        with open(official, encoding="utf-8") as f:
            return json.load(f)
    import pyarrow.parquet as pq

    rows = []
    for row in pq.read_table(mirror).to_pylist():
        context = row["context"]
        facts = row["supporting_facts"]
        rows.append({
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "context": [[t, s] for t, s in zip(context["title"], context["sentences"])],
            "supporting_facts": [[t, i] for t, i in zip(facts["title"], facts["sent_id"])],
        })
    return rows


def as_lines(example):
    """The paragraphs as lines, and the line index of every gold sentence."""
    gold_keys = {(title, index) for title, index in example["supporting_facts"]}
    lines, gold = [], []
    for p, (title, sentences) in enumerate(example["context"]):
        if p > 0:
            lines.append("")
        for s, sentence in enumerate(sentences):
            if (title, s) in gold_keys:
                gold.append(len(lines))
            lines.append(nfc(" ".join(sentence.split())))
    return lines, gold


def classify(question, lines, gold):
    rest = [line for i, line in enumerate(lines) if i not in gold]
    if any(rare_shared(question, lines[g], rest + [lines[h] for h in gold if h != g]) for g in gold):
        return "lexical"
    return "paraphrase"


def generate(r, count, cache_dir, exclude=frozenset(), prefix="prose"):
    examples = load(cache_dir)
    pools = {"lexical": [], "paraphrase": []}
    for example in examples:
        if example["_id"] in exclude:
            continue
        lines, gold = as_lines(example)
        if not gold:
            continue
        pools[classify(nfc(example["question"]), lines, gold)].append(example)
    for pool in pools.values():
        r.shuffle(pool)
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        split = ("lexical", "paraphrase")[i % 2]
        example = pools[split].pop()
        lines, gold = as_lines(example)
        question = nfc(example["question"])
        if i in absent:
            lines = [line for j, line in enumerate(lines) if j not in gold]
            gold = []
        rows.append({"id": f"{prefix}-{i:03}", "family": "prose", "split": split,
                     "absent": i in absent, "segments": len(lines),
                     "state": "\n".join(lines),
                     "instruction": INSTRUCTION.format(question=question),
                     "targets": gold, "distractors": [], "event": example["_id"]})
    return rows
