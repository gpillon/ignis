"""Cross-check ignis-grammar's verdicts with an independent JSON Schema
validator (GitHub #585, spec #287; docs/agents/testing/structured-output.md).

    python -m venv VENV
    VENV/Scripts/pip install -r tools/grammar-check/requirements.txt
    VENV/Scripts/python -I tools/grammar-check/validate.py OUT_DIR/crosscheck.jsonl

OUT_DIR is what the bench wrote (`cargo run --release -p ignis-grammar
--example grammar_bench -- ARTIFACT crates/grammar/tests/corpus/cases.json
OUT_DIR`).

Each line of the input carries a document, its case's schema (null for
json_object), where it came from (corpus_valid, corpus_invalid,
outside_subset, random) and the automaton's verdict. This script validates the
document with `jsonschema` (Draft 2020-12, format checking on) and reports the
confusion per source. The formats `jsonschema` cannot check without extra
packages get our own independent checks below, written from the RFCs, not from
the grammar's regular expressions.

What must hold: every document the automaton accepted is valid here
(soundness; a random document of the grammar's language most of all), and
every corpus document the automaton refused is invalid here, except the
`outside_subset` ones, which are valid JSON-schema documents the grammar does
not generate by design (keys out of schema order).
"""

import datetime
import ipaddress
import json
import re
import sys
import uuid
from decimal import Decimal
from collections import Counter, defaultdict

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker


def duration(s):
    # RFC 3339 appendix A, by hand: P, then weeks alone or date parts, then
    # an optional T with time parts; at least one part on each side used.
    if not isinstance(s, str):
        return True
    if not s.startswith("P") or len(s) < 3:
        return False
    body = s[1:]
    if "T" in body:
        date_part, time_part = body.split("T", 1)
        if not time_part:
            return False
    else:
        date_part, time_part = body, None

    def parts(text, units):
        out, num = [], ""
        for ch in text:
            if ch.isdigit() and ch.isascii():
                num += ch
            elif ch in units and num:
                out.append(ch)
                num = ""
            else:
                return None
        if num:
            return None
        order = [units.index(u) for u in out]
        if order != sorted(set(order)):
            return None
        return out

    if date_part:
        if date_part.endswith("W"):
            if time_part is not None or parts(date_part, "W") != ["W"]:
                return False
        else:
            got = parts(date_part, "YMD")
            if got is None:
                return False
            # RFC 3339: years-months-days are contiguous (no Y then D).
            if got == ["Y", "D"]:
                return False
    if time_part is not None:
        got = parts(time_part, "HMS")
        if not got or got == ["H", "S"]:
            return False
    return True


def ipv4(s):
    if not isinstance(s, str):
        return True
    try:
        ipaddress.IPv4Address(s)
        return True
    except ValueError:
        return False


def ipv6(s):
    if not isinstance(s, str):
        return True
    try:
        ipaddress.IPv6Address(s)
        return "%" not in s
    except ValueError:
        return False


def is_uuid(s):
    if not isinstance(s, str):
        return True
    try:
        uuid.UUID(s)
    except ValueError:
        return False
    return len(s) == 36 and all(s[i] == "-" for i in (8, 13, 18, 23))


def date(s):
    if not isinstance(s, str):
        return True
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", s):
        return False
    try:
        datetime.date.fromisoformat(s)
        return True
    except ValueError:
        return False


def checker():
    # Draft 2020-12's own checkers: a bare FormatChecker() merges every
    # draft's, and draft 3's `time` (no offset allowed) overrides 2020-12's.
    fc = FormatChecker(formats=())
    fc.checkers = dict(Draft202012Validator.FORMAT_CHECKER.checkers)
    fc.checks("duration")(duration)
    fc.checks("ipv4")(ipv4)
    fc.checks("ipv6")(ipv6)
    fc.checks("uuid")(is_uuid)
    fc.checks("date")(date)
    return fc


def no_constants(name):
    # Python's json accepts NaN and Infinity; JSON does not.
    raise json.JSONDecodeError(f"{name} is not JSON", name, 0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    fc = checker()
    missing = [f for f in ("date-time", "time", "email", "hostname") if f not in fc.checkers]
    validators = {}
    counts = defaultdict(Counter)
    float_only = Counter()
    bad = []

    def verdict(row, exact):
        # exact: numbers as Decimal, so multipleOf and the bounds are decimal
        # arithmetic, as JSON Schema defines them; otherwise Python floats,
        # as most validators run.
        pf = Decimal if exact else float
        try:
            doc = json.loads(row["text"], parse_constant=no_constants, parse_float=pf)
        except json.JSONDecodeError:
            return False
        if row["schema"] is None:
            return isinstance(doc, dict)
        key = (row["schema"], exact)
        v = validators.get(key)
        if v is None:
            v = validators[key] = Draft202012Validator(json.loads(row["schema"], parse_float=pf), format_checker=fc)
        return v.is_valid(doc)

    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        exact = verdict(row, True)
        if verdict(row, False) != exact:
            float_only[(row["case"], row["source"])] += 1
        theirs = "valid" if exact else "invalid"
        key = f"ours {row['ours']}, theirs {theirs}"
        counts[row["source"]][key] += 1
        sound_violation = row["ours"] == "valid" and theirs == "invalid"
        refused_valid = row["ours"] == "invalid" and theirs == "valid" and row["source"] != "outside_subset"
        if sound_violation or refused_valid:
            bad.append((row["case"], row["source"], key, row["text"][:300]))
    from importlib.metadata import version
    print(f"jsonschema {version('jsonschema')}; format checkers missing: {missing or 'none'}")
    for source, c in sorted(counts.items()):
        print(f"{source:<16} " + "; ".join(f"{k}: {n}" for k, n in sorted(c.items())))
    print("verdicts that change when the validator uses floats instead of decimals: "
          + (", ".join(f"{c}/{s}: {n}" for (c, s), n in sorted(float_only.items())) or "none"))
    print(f"disagreements: {len(bad)}")
    for b in bad[:40]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
