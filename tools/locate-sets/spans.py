"""Span sets: where in the text, not which line (spec 19 phase 1, GitHub #276).

Spec 18's sets ask for a segment. These ask for a **place**: every question's
gold is one or more character spans of the state -- the charge id inside a
log line, a record's SKU, the answer inside a paragraph -- and a question may
have several golds or none. Six families:

- `logvalue`: spec 18's logs (60 / 250 / 1,000 lines), the question asking
  for a value inside the target ERROR line (a charge, a job, a file, a host);
- `recvalue`: spec 18's record arrays (20 / 80 / 300), the question asking
  for the target record's id or SKU;
- `squad`: SQuAD 2.0 dev (Rajpurkar, Jia and Liang, 2018; CC BY-SA 4.0): a
  question's paragraph with its article's neighbours, one line per
  paragraph, about 1K or 4K tokens; its answers are the golds, and an
  unanswerable question is absent;
- `hotspan`: HotpotQA distractor dev, the questions whose answer string
  appears in a gold supporting sentence, laid out as spec 18's prose;
- `logmulti`: logs asked "which lines report ..." with zero to three
  matching lines, each a whole-line gold (spec 19's several-or-none);
- `hotfacts`: HotpotQA's supporting sentences, every one of them a gold.

A span is `{"segment": j, "start": a, "end": b}`: character offsets into
segment j's **plain** text -- a line of a string state, or an element's
compact JSON (`profiles.segment_texts`) -- so a scorer maps it to keys
through the evidence writer, whatever the escaping. `targets` lists the
segments that hold a gold, which keeps spec 18's line-level scoring working
on these sets, and `quote` is the first gold's text, what the harness forces
after `{"quote":"` for its teacher-forced prefill.

The lexical / paraphrase split is spec 18's (`common.py`), checked against
the segment holding the gold; an absent question has its gold removed (or,
for SQuAD, is one of its unanswerable questions).

    python tools/locate-sets/spans.py --seed 20261020 --out .scratch/locate/E1 --exclude .scratch/locate/A ...
"""

import argparse
import json
import os
import random
import re

import logs
import prose
import records
from common import assign_absent, nfc, paraphrase_clean, rare_shared

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_TIERS = (4_000, 16_000)            # characters, about 1K and 4K tokens
# A value question names what it wants; the suffix says the answer is the
# value alone, not the line around it (both routes read the same words).
VALUE_SUFFIX = " Answer with that value alone."
PER_FAMILY = {"logvalue": 60, "recvalue": 60, "squad": 60, "hotspan": 60, "logmulti": 30, "hotfacts": 30}

# logvalue: per event, the value's pattern in the ERROR message and the two
# questions asking for it. The word checks run on every generated question.
LOG_VALUES = {
    "provider": (r"ch_[0-9a-f]{6}",
                 "Which charge did the payment provider return a 503 for?",
                 "What is the reference of the card transaction that failed because the card processing company was unavailable?"),
    "oom": (r"(?<=process )\d+",
            "Which process was killed for running out of memory?",
            "Which task identifier was terminated for exhausting RAM?"),
    "deadlock": (r"(?<=transaction )\d+",
                 "Which transaction was rolled back after a deadlock?",
                 "Give the number of the database operation that was undone after two operations blocked each other."),
    "ratelimit": (r"k_[0-9a-f]{6}",
                  "For which key was the rate limit exceeded?",
                  "Which client credential was slowed down for sending too many calls?"),
    "migration": (r"(?<=migration )\d+",
                  "Which schema migration failed?",
                  "Give the number of the upgrade to the database layout that could not be applied."),
    "queue": (r"q_[0-9a-f]{4}",
              "Which message queue is full?",
              "Which buffer between services overflowed and lost data?"),
    "smtp": (r"user\d+@example\.org",
             "To which address did the SMTP relay refuse mail?",
             "Which recipient could not get an email because the outgoing server declined it?"),
    "config": (r"feature\.[0-9a-f]{4}",
               "Which config key is missing?",
               "Which setting was absent so standard values were used instead?"),
    "lock": (r"j\d+(?= held)",
             "Which job's lock was forced to release?",
             "Which task had an exclusive claim that lasted too long and was broken?"),
    "upload": (r"export-\d+\.csv",
               "Which file's S3 upload failed with access denied?",
               "Which file could not be sent to cloud storage for lack of permission?"),
    "segfault": (r"img_[0-9a-f]{6}\.png",
                 "Which image was being resized when the segfault happened?",
                 "Which picture was a native library processing when it crashed?"),
    "jwt": (r"s_[0-9a-f]{6}",
            "Which session has an invalid JWT signature?",
            "For which visit did a login token fail its authenticity test?"),
    "cron": (r"nightly-[0-9a-f]{4}",
             "Which cron job exited with status 137?",
             "Which periodic task ended abnormally?"),
    "index": (r"idx_[a-z]+",
              "Which index is corrupted?",
              "Which lookup structure was damaged and is being recreated?"),
    "stock": (r"(?<=SKU )\d+",
              "Which SKU has negative stock?",
              "Which article's inventory count dropped below zero?"),
    "websocket": (r"(?<=by client )\S+",
                  "Which client closed the websocket unexpectedly?",
                  "Which machine dropped a live browser link without warning?"),
    "dns": (r"(?<=DNS lookup for )\S+",
            "For which host did the DNS lookup time out?",
            "Which server's name resolution never got an answer?"),
    "webhook": (r"hooks\.example\.com/[0-9a-f]{4}",
                "Which webhook returned 410?",
                "Which callback endpoint no longer exists, so notifications were switched off?"),
}


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def span_of(segment, start, end):
    return {"segment": segment, "start": start, "end": end}


def text_of(state, span):
    """The gold's text, from the state as a caller sends it."""
    if isinstance(state, str):
        return state.split("\n")[span["segment"]][span["start"]:span["end"]]
    element = json.dumps(state[span["segment"]], ensure_ascii=False, separators=(",", ":"))
    return element[span["start"]:span["end"]]


def question_row(prefix, i, family, split, absent, state, instruction, spans, **extra):
    targets = sorted({s["segment"] for s in spans})
    segments = len(state.split("\n")) if isinstance(state, str) else len(state)
    row = {"id": f"{prefix}-{i:03}", "family": family, "split": split, "absent": absent,
           "segments": segments, "state": state, "instruction": instruction,
           "targets": targets, "spans": spans, "distractors": [], "event": None}
    row["quote"] = text_of(state, spans[0]) if spans else None
    row.update(extra)
    return row


def words_hold(question, split, target, others):
    if split == "lexical":
        return bool(rare_shared(question, target, others))
    return paraphrase_clean(question, target)


# --- logvalue --------------------------------------------------------------

def logvalue(r, count, prefix="logvalue"):
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        length = logs.LENGTHS[i % len(logs.LENGTHS)]
        split = ("lexical", "paraphrase")[i % 2]
        event = r.choice(sorted(LOG_VALUES))
        pattern, lexical, paraphrase = LOG_VALUES[event]
        question = (lexical if split == "lexical" else paraphrase) + VALUE_SUFFIX
        for _ in range(50):
            # spec 18's log with a present target, whose own checks use its
            # line question; the value question is checked here
            row = logs.one_log(r, length, event, split, False)
            if row is None:
                continue
            lines = row["state"].split("\n")
            t = row["targets"][0]
            found = re.search(pattern, lines[t])
            if found and words_hold(question, split, lines[t], lines[:t] + lines[t + 1:]):
                break
        else:
            raise SystemExit(f"logvalue: event {event!r} ({split}) failed its checks 50 times")
        spans = [span_of(t, found.start(), found.end())]
        if i in absent:
            level, message = logs._routine(r)
            lines[t] = f"{lines[t][:5]} {level} {r.choice(logs.SERVICES)}: {message}"
            spans = []
        rows.append(question_row(prefix, i, "logvalue", split, i in absent, "\n".join(lines), question, spans,
                                 distractors=row["distractors"], event=event))
    return rows


# --- recvalue --------------------------------------------------------------

def record_question(kind, value, split):
    """The value question for a record kind, and the field it asks for."""
    if kind == "employees":
        return ("id", f"What is the id of the employee who works from {value}?" if split == "lexical"
                else f"What is the staff number of the person based in the capital of {records.CAPITALS[value]}?")
    if kind == "tickets":
        return ("id", f"What is the id of the ticket titled \"{value}\"?" if split == "lexical"
                else f"What is the reference of the ticket about {records.TICKETS[value]}?")
    return ("sku", f"What is the SKU of the {value.lower()}?" if split == "lexical"
            else f"What is the catalogue code of the product that {records.PRODUCTS[value]}?")


def field_span(element_json, field):
    """Where a field's value sits in an element's compact JSON (a string's
    value without its quotes)."""
    key = f"\"{field}\":"
    at = element_json.index(key) + len(key)
    if element_json[at] == "\"":
        return at + 1, element_json.index("\"", at + 1)
    end = at
    while end < len(element_json) and element_json[end] not in ",}":
        end += 1
    return at, end


def recvalue(r, count, prefix="recvalue"):
    absent = assign_absent(r, count)
    kinds = ("employees", "tickets", "products")
    rows = []
    for i in range(count):
        length = records.LENGTHS[i % len(records.LENGTHS)]
        split = ("lexical", "paraphrase")[i % 2]
        kind = kinds[(i // 6) % len(kinds)]
        for _ in range(50):
            row = records.one_array(r, length, kind, split, False)
            if row is None:
                continue
            t = row["targets"][0]
            field, question = record_question(kind, row["event"], split)
            question += VALUE_SUFFIX
            texts = [records._text(x) for x in row["state"]]
            if words_hold(question, split, texts[t], texts[:t] + texts[t + 1:]):
                break
        else:
            raise SystemExit(f"recvalue: {kind} ({split}) failed its checks 50 times")
        state = row["state"]
        element = json.dumps(state[t], ensure_ascii=False, separators=(",", ":"))
        spans = [span_of(t, *field_span(element, field))]
        if i in absent:
            build, _, filler, _, _ = records._kind(random.Random(f"{prefix}/{i}"), kind)
            state[t] = build(r, t, filler())
            spans = []
        rows.append(question_row(prefix, i, "recvalue", split, i in absent, state, question, spans,
                                 kind=kind, event=row["event"]))
    return rows


# --- squad -----------------------------------------------------------------

def load_squad(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "dev-v2.0.json")
    if not os.path.exists(path):
        prose._fetch(SQUAD_URL, path, timeout=120)
    return read_json(path)["data"]


def squad_state(paragraphs, p, chars):
    """Paragraph p and its neighbours (after, before, after, ...) until the
    text reaches `chars`: (lines, the index of p among them)."""
    chosen, before, after = [p], p - 1, p + 1
    while sum(len(paragraphs[j]) for j in chosen) < chars and (before >= 0 or after < len(paragraphs)):
        if after < len(paragraphs):
            chosen.append(after)
            after += 1
        if sum(len(paragraphs[j]) for j in chosen) < chars and before >= 0:
            chosen.append(before)
            before -= 1
    chosen.sort()
    return [paragraphs[j] for j in chosen], chosen.index(p)


def nearest(text, needle, near):
    """The occurrence of `needle` in `text` closest to `near`, or None."""
    best, at = None, text.find(needle)
    while at >= 0:
        if best is None or abs(at - near) < abs(best - near):
            best = at
        at = text.find(needle, at + 1)
    return best


def squad(r, count, cache_dir, exclude=frozenset(), prefix="squad"):
    articles = load_squad(cache_dir)
    pools = {(split, impossible): [] for split in ("lexical", "paraphrase") for impossible in (False, True)}
    for a, article in enumerate(articles):
        paragraphs = [nfc(p["context"].replace("\n", " ")) for p in article["paragraphs"]]
        for p, paragraph in enumerate(article["paragraphs"]):
            for qa in paragraph["qas"]:
                if qa["id"] in exclude:
                    continue
                pools_key = qa["is_impossible"]
                pools_entry = (a, p, qa)
                question = nfc(qa["question"])
                rest = paragraphs[:p] + paragraphs[p + 1:]
                split = "lexical" if rare_shared(question, paragraphs[p], rest) else "paraphrase"
                pools[(split, pools_key)].append(pools_entry)
    for pool in pools.values():
        r.shuffle(pool)
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        split = ("lexical", "paraphrase")[i % 2]
        chars = SQUAD_TIERS[(i // 2) % len(SQUAD_TIERS)]
        while True:
            a, p, qa = pools[(split, i in absent)].pop()
            paragraphs = [nfc(x["context"].replace("\n", " ")) for x in articles[a]["paragraphs"]]
            lines, at = squad_state(paragraphs, p, chars)
            original = articles[a]["paragraphs"][p]["context"]
            spans = []
            for answer in qa["answers"]:
                start = nearest(lines[at], nfc(answer["text"]), len(nfc(original[:answer["answer_start"]])))
                if start is not None:
                    span = span_of(at, start, start + len(nfc(answer["text"])))
                    if span not in spans:
                        spans.append(span)
            if i in absent or spans:
                break
        rows.append(question_row(prefix, i, "squad", split, i in absent, "\n".join(lines), nfc(qa["question"]),
                                 spans, event=qa["id"], chars=chars))
    return rows


# --- hotspan and hotfacts --------------------------------------------------

def answer_spans(lines, gold, answer):
    """Every occurrence of `answer` inside a gold line."""
    spans = []
    for g in gold:
        at = lines[g].find(answer)
        while at >= 0:
            spans.append(span_of(g, at, at + len(answer)))
            at = lines[g].find(answer, at + 1)
    return spans


def hotspan(r, count, cache_dir, exclude=frozenset(), prefix="hotspan"):
    examples = prose.load(cache_dir)
    pools = {"lexical": [], "paraphrase": []}
    for example in examples:
        answer = nfc(" ".join(example["answer"].split()))
        if example["_id"] in exclude or answer.lower() in ("yes", "no") or not answer:
            continue
        lines, gold = prose.as_lines(example)
        if not gold or not answer_spans(lines, gold, answer):
            continue
        pools[prose.classify(nfc(example["question"]), lines, gold)].append(example)
    for pool in pools.values():
        r.shuffle(pool)
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        split = ("lexical", "paraphrase")[i % 2]
        while True:
            example = pools[split].pop()
            lines, gold = prose.as_lines(example)
            answer = nfc(" ".join(example["answer"].split()))
            spans = answer_spans(lines, gold, answer)
            if i in absent:
                lines = [line for j, line in enumerate(lines) if j not in gold]
                if any(answer in line for line in lines):
                    continue          # the answer is still in the state: not absent
                spans = []
            break
        rows.append(question_row(prefix, i, "hotspan", split, i in absent, "\n".join(lines),
                                 nfc(example["question"]), spans, event=example["_id"], answer=answer))
    return rows


def hotfacts(r, count, cache_dir, exclude=frozenset(), prefix="hotfacts"):
    examples = [e for e in prose.load(cache_dir) if e["_id"] not in exclude]
    r.shuffle(examples)
    absent = assign_absent(r, count)
    rows = []
    for i in range(count):
        while True:
            example = examples.pop()
            lines, gold = prose.as_lines(example)
            if len(gold) >= 2:
                break
        question = nfc(example["question"])
        split = prose.classify(question, lines, gold)
        spans = [span_of(g, 0, len(lines[g])) for g in gold]
        if i in absent:
            lines = [line for j, line in enumerate(lines) if j not in gold]
            spans = []
        rows.append(question_row(prefix, i, "hotfacts", split, i in absent, "\n".join(lines),
                                 prose.INSTRUCTION.replace("Which sentence helps", "Which sentences help")
                                 .format(question=question), spans, event=example["_id"]))
    return rows


# --- logmulti --------------------------------------------------------------

def plural(question):
    return (question.replace("Which line reports", "Which lines report")
            .replace("Which line says", "Which lines say"))


def logmulti(r, count, prefix="logmulti"):
    rows = []
    for i in range(count):
        length = logs.LENGTHS[(i // 8) % 2]              # 60 and 250 lines
        matches = i % 4                                   # zero to three
        split = ("lexical", "paraphrase")[(i // 4) % 2]
        event = r.choice(sorted(logs.EVENTS))
        question = plural(logs.EVENTS[event][1 if split == "lexical" else 2])
        for _ in range(50):
            others = [key for key in logs.EVENTS if key != event]
            distractors = r.sample(others, r.randint(3, 8))
            slots = r.sample(range(length), matches + len(distractors))
            special = {**{s: event for s in slots[:matches]}, **dict(zip(slots[matches:], distractors))}
            clock = r.randint(0, 86399)
            lines = []
            for j in range(length):
                clock += r.randint(0, 20)
                if j in special:
                    level, message = "ERROR", logs.EVENTS[special[j]][0](r)
                else:
                    level, message = logs._routine(r)
                lines.append(f"{logs._stamp(clock)} {level} {r.choice(logs.SERVICES)}: {message}")
            golds = sorted(slots[:matches])
            rest = [line for j, line in enumerate(lines) if j not in golds]
            if all(words_hold(question, split, lines[g], rest) for g in golds):
                break
        else:
            raise SystemExit(f"logmulti: event {event!r} ({split}) failed its checks 50 times")
        spans = [span_of(g, 0, len(lines[g])) for g in golds]
        rows.append(question_row(prefix, i, "logmulti", split, matches == 0, "\n".join(lines), question, spans,
                                 distractors=sorted(slots[matches:]), event=event, matches=matches))
    return rows


# ---------------------------------------------------------------------------

def generate(seed, cache, exclude_ids, per_family=PER_FAMILY):
    """One set, every family seeded from the set's seed (so a family's draws
    never move another's)."""
    def rng(name):
        return random.Random(f"{seed}/{name}")
    hot = os.path.join(cache, "hotpot")
    return (logvalue(rng("logvalue"), per_family["logvalue"])
            + recvalue(rng("recvalue"), per_family["recvalue"])
            + squad(rng("squad"), per_family["squad"], os.path.join(cache, "squad"), exclude_ids)
            + hotspan(rng("hotspan"), per_family["hotspan"], hot, exclude_ids)
            + logmulti(rng("logmulti"), per_family["logmulti"])
            + hotfacts(rng("hotfacts"), per_family["hotfacts"], hot, exclude_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.join(".scratch", "locate"),
                    help="where HotpotQA (hotpot/) and SQuAD (squad/) are downloaded")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="set directories whose HotpotQA and SQuAD questions this set must not reuse")
    args = ap.parse_args()
    exclude = set()
    for other in args.exclude:
        exclude |= {q["event"] for q in read_json(os.path.join(other, "manifest.json"))["questions"]
                    if q["family"] in ("prose", "squad", "hotspan", "hotfacts")}
    questions = generate(args.seed, args.cache, exclude)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"seed": args.seed, "kind": "spans", "per_family": PER_FAMILY, "exclude": sorted(exclude),
                "questions": questions}
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    summary = {}
    for q in questions:
        key = (q["family"], q["split"], q["absent"])
        summary[key] = summary.get(key, 0) + 1
    for key in sorted(summary):
        print(f"  {key[0]:9} {key[1]:10} {'absent' if key[2] else 'present':7} {summary[key]}")
    print(f"wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
