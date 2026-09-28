"""Golden cases for spec 22's host functions (GitHub #278): the Python the
research measured with, composed into the served route's pure steps, and
written as fixtures the Rust port is held to (`crates/core/tests/`).

Every case is synthetic — never a cluster line. Each family calls the
reference function itself rather than a re-statement of it:

- `fold`: `compress.fold(values=True)`, `summarize`, `level2`,
  `_common_affixes` at the settings R2 was judged with;
- `readings`: `zd_cache.key_features` (in f64, as `zd_logpipe.head_rank`
  reads it — not the f32 the npz files rounded to) and `zd_offline.zsum`,
  summed over heads, for the end reading (last + separator + next first key)
  and the sum reading;
- `windows`: the one rule spec 22 states — at the last empty segment within
  the window when there is one (`zd_windows.sub_windows`), else at the last
  segment that fits (`zd_records.cut`) — checked here against both
  references on the inputs each was written for;
- `merge`: `zd_prose.py rank`'s (prose) and `zd_records.py rank-windows`'
  (records, and a log read without a fold) standardization per window;
- `renders`: `zd_prose.render` (prose in its paragraphs) and the labelled
  lines `zd_notfound.ask` sends; `json.dumps(ensure_ascii=False)` (spaced
  JSON);
- `auto`: spec 22 § `auto` over `compress.fold`'s clusters;
- `found`: spec 22 § Not found.

    python golden22.py --out ../../crates/core/tests/fixtures
"""

import argparse
import json
import os

import numpy as np

import compress as C
from zd_cache import key_features
from zd_offline import zsum
from zd_prose import render as prose_render
from zd_prose import std as prose_std
from zd_records import cut as records_cut
from zd_windows import sub_windows

LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]


# ---------------------------------------------------------------------------
# template_fold

FOLD_INPUTS = {
    "labels": [
        "[svc-a] started worker 12",
        "[svc-a] started worker 13",
        "[svc-b] started worker 12",
        "no label on this line",
        "[] an empty label",
        "[not a label but a list] item one",
    ],
    "times in every shape": [
        "2026-09-28T12:00:00Z request served path=/a",
        "2026-09-28 12:00:01.123+02:00 request served path=/b",
        "2026/09/28 12:00 request served path=/c",
        "12:00:01 request served path=/d",
        "at 12:00:01.5 request served path=/e",
        "2026-09-28T12:00:00+0200 request served path=/f",
        "2026-09-28T12:00:00-07:00 request served path=/g",
        "x12:00:01 is no time",
        "2026-09-28 12:00:00",
        "2026-09-28 12:00:00",
    ],
    "masked variables": [
        "took 12ms for 3.5s of 45% at 10.0.0.1:8080",
        "took 7ms for 1.25s of 99% at 10.0.0.2:8080",
        "id deadbeef01 uuid 123e4567-e89b-12d3-a456-426614174000 count 1,234",
        "id cafebabe99 uuid 00000000-0000-0000-0000-000000000000 count 12/34",
        "quoted \"12\", (42) id=17 =17 {3}",
        "quoted \"13\", (43) id=18 =18 {4}",
        "sizes 12KB 3mb 4b 5h 6m 7us 8µs 9μs 10Kb 11ſ",
        "sizes 13KB 4mb 5b 6h 7m 8us 9µs 10μs 11Kb 12ſ",
        "hex DEADBEEF12 and 0x1f and abcdefgh",
    ],
    "the SIM boundary": [
        "alpha beta gamma delta",
        "alpha beta zeta eta",
        "alpha theta iota kappa",
        "one two three four",
        "one two three five",
        "one six seven eight",
        "one two nine ten",
    ],
    "the 600-character budget": (
        [f"user {'u' * 30}{i:03d} logged in from {'h' * 20}{i:02d}" for i in range(40)]
        + [f"job {chr(97 + i % 26)}{i} done" for i in range(12)]
    ),
    "long values and non-ASCII": [
        "Ünïcödé 名前 🚀 value=Grüße-aus-Köln-am-Rhein-und-mehr",
        "Ünïcödé 名前 🚀 value=Saluti-da-Milano-in-Lombardia-oggi",
        "Ünïcödé 名前 🛰 value=Привет-из-Москвы",
        "naïve café crème brûlée",
        "naïve café crème flambée",
    ],
    "repeats and affixes": [
        "GET /api/v1/items application=\"foo\" status=200",
        "GET /api/v1/items application=\"bar\" status=200",
        "GET /api/v1/items application=\"foo\" status=200",
        "GET /api/v1/items application=\"foo\" status=200",
        "GET /api/v1/items application=\"baz\" status=500",
        "2026-09-28T10:00:00Z GET /api/v1/items application=\"foo\" status=200",
    ],
    "whitespace and carriage returns": [
        "  leading spaces here",
        "trailing spaces here   ",
        "tabs\there\tand\tthere",
        "carriage return line\r",
        "carriage return line\r",
        "unit\x1cseparator\x1fsplit",
        "unit\x1cseparator\x1fsplat",
        "no break space",
    ],
    "records as spaced JSON": [
        json.dumps({"id": 1, "name": "Ada", "city": "Paris"}, ensure_ascii=False),
        json.dumps({"id": 2, "name": "Bo", "city": "Rome"}, ensure_ascii=False),
        json.dumps({"id": 3, "name": "Cy Young", "city": "New York"}, ensure_ascii=False),
        json.dumps({"id": 4, "name": "Dee", "tags": ["a", "b"], "ok": True, "n": None}, ensure_ascii=False),
    ],
    "string elements": ["apple pie", "apple tart", "banana split", "cherry"],
    "one line": ["just one line with 12 values"],
    "one template": ["tick 1", "tick 2", "tick 3", "tick 3"],
}


def fold_case(name, lines):
    f = C.fold(lines, C.SIM, values=True)
    level2 = []
    for ci in range(len(f.clusters)):
        texts, members = C.level2(f, ci)
        level2.append({"texts": texts, "members": members})
    return {
        "name": name,
        "lines": lines,
        "clusters": [{"label": c.label, "template": c.template, "members": c.members} for c in f.clusters],
        "level1": f.level1,
        "level2": level2,
    }


# ---------------------------------------------------------------------------
# readings

def synthetic_window(rng, owned):
    """Keys for segments that own (`owned[i]`) a run of 1-7 keys, one
    separator key between consecutive segments; the span ends at the last
    owned key, as `crate::locate::key_span` cuts it."""
    keys, p = [], 0
    for own in owned:
        if own:
            n = int(rng.integers(1, 8))
            keys.append([p, p + n])
            p += n + 1
        else:
            keys.append(None)
            p += 1
    span = max(k[1] for k in keys if k is not None)
    return keys, span


def rows(rng, heads, span, peaks=()):
    q = rng.normal(0.0, 2.0, size=(heads, span)).astype(np.float32)
    for at in peaks:
        q[:, at] += np.float32(6.0)
    return q


def reading_scores(q, na, keys, span, reading):
    """One window's reading: every head's lift standardized over the
    window's segments (`zsum`), summed over the heads."""
    k = [tuple(x) if x is not None else None for x in keys]
    fq = key_features(q.astype(np.float64), k, span)
    fn = key_features(na.astype(np.float64), k, span)
    if reading == "end":
        lift = (fq["last"] + fq["sep"] + fq["next1"]) - (fn["last"] + fn["sep"] + fn["next1"])
    else:
        lift = fq["sum"] - fn["sum"]
    return zsum(lift).sum(axis=0)


def reading_cases(seed):
    rng = np.random.default_rng(seed)
    cases = []
    shapes = [
        ("two segments", 2, [True, True]),
        ("an empty segment between", 4, [True, False, True, True]),
        ("empty first and last", 8, [False, True, True, True, True, False]),
        ("thirty-two heads", 32, [True] * 12 + [False] + [True] * 7),
        ("one head", 1, [True, True, True]),
    ]
    for name, heads, owned in shapes:
        keys, span = synthetic_window(rng, owned)
        owners = [k for k in keys if k is not None]
        target = owners[len(owners) // 2]
        q = rows(rng, heads, span, peaks=[target[1] - 1])
        na = rows(rng, heads, span)
        case = {"name": name, "heads": heads, "span": span, "keys": keys,
                "q": q.astype(float).ravel().tolist(), "na": na.astype(float).ravel().tolist()}
        for reading in ("end", "sum"):
            case[reading] = reading_scores(q, na, keys, span, reading).tolist()
        cases.append(case)
    # a head whose every lift is equal: zsum's +1e-12 keeps it at zero
    keys, span = synthetic_window(rng, [True, True, True])
    q = np.zeros((2, span), np.float32)
    na = np.zeros((2, span), np.float32)
    q[1] = rng.normal(0, 1, span).astype(np.float32)
    case = {"name": "a flat head", "heads": 2, "span": span, "keys": keys,
            "q": q.astype(float).ravel().tolist(), "na": na.astype(float).ravel().tolist()}
    for reading in ("end", "sum"):
        case[reading] = reading_scores(q, na, keys, span, reading).tolist()
    cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# windows

def cut_windows(costs, empty, budget):
    """Spec 22's windows: segment ranges `[first, end)` of at most `budget`
    keys each where the segments allow — cut at the last empty segment
    within the window when there is one (it belongs to no window), else
    before the segment that would not fit."""
    out, start, total, last_break = [], 0, 0, None
    for i, cost in enumerate(costs):
        if empty[i]:
            last_break = i
        if total + cost > budget and i > start:
            if last_break is not None and last_break > start:
                out.append((start, last_break))
                start = last_break + 1
                total = sum(costs[start:i + 1])
                last_break = None
                continue
            out.append((start, i))
            start, total, last_break = i, 0, None
        total += cost
    out.append((start, len(costs)))
    return out


class Costs:
    """A tokenizer stand-in for `zd_records.cut`: the cost of a record is
    looked up, less the +1 `cut` adds."""

    def __init__(self, costs):
        self.costs = costs
        self.at = 0

    def encode(self, _text):
        cost = self.costs[self.at]
        self.at += 1
        return type("E", (), {"ids": [0] * (cost - 1)})()


def window_cases(seed):
    rng = np.random.default_rng(seed)
    cases = []
    # prose: paragraph breaks often enough that every window has one
    for n, budget in ((40, 60), (120, 200), (9, 1000)):
        lines = []
        for i in range(n):
            lines.append("" if i % 5 == 4 else f"sentence {i}")
        costs = [1 if line == "" else int(rng.integers(3, 12)) for line in lines]
        empty = [line == "" for line in lines]
        got = cut_windows(costs, empty, budget)
        assert got == sub_windows(lines, costs, budget), (got, sub_windows(lines, costs, budget))
        cases.append({"name": f"prose {n} at {budget}", "costs": costs, "empty": empty, "budget": budget,
                      "windows": [list(w) for w in got]})
    # records and log lines: no empty segment, cut before what does not fit
    for n, budget in ((30, 100), (50, 57), (5, 10_000)):
        costs = [int(rng.integers(5, 16)) for _ in range(n)]
        empty = [False] * n
        got = cut_windows(costs, empty, budget)
        assert got == records_cut(list(range(n)), Costs(costs), budget), got
        cases.append({"name": f"records {n} at {budget}", "costs": costs, "empty": empty, "budget": budget,
                      "windows": [list(w) for w in got]})
    # a segment alone past the budget gets a window of its own
    costs = [3, 50, 3, 3]
    got = cut_windows(costs, [False] * 4, 10)
    assert got == records_cut(list(range(4)), Costs(costs), 10), got
    cases.append({"name": "one segment past the budget", "costs": costs, "empty": [False] * 4, "budget": 10,
                  "windows": [list(w) for w in got]})
    # no break inside a window: the rule falls back to the last segment that
    # fits (where `sub_windows` would have run past the budget)
    costs = [5, 5, 5, 1, 5, 5, 5, 5, 5, 5]
    empty = [False, False, False, True, False, False, False, False, False, False]
    cases.append({"name": "a break, then none", "costs": costs, "empty": empty, "budget": 12,
                  "windows": [list(w) for w in cut_windows(costs, empty, 12)]})
    # the break is the segment that overflows: cut there, it belongs to none
    costs = [5, 5, 3, 5]
    empty = [False, False, True, False]
    cases.append({"name": "the overflowing segment is the break", "costs": costs, "empty": empty, "budget": 11,
                  "windows": [list(w) for w in cut_windows(costs, empty, 11)]})
    return cases


# ---------------------------------------------------------------------------
# merge and shortlist

def merge_prose(n, windows):
    """`zd_prose.py rank`: each window's reading, a sum under -1e2 (or not
    finite) set to -1e4 and kept in the standardization, standardized with
    `std`; segments no window read stay at -1e4."""
    merged = np.full(n, -1e4)
    for first, v in windows:
        v = np.asarray(v, dtype=float)
        v = np.where(np.isfinite(v) & (v > -1e2), v, -1e4)
        merged[first:first + len(v)] = prose_std(v)
    return merged


def merge_records(n, windows):
    """`zd_records.py rank-windows`: each window's reading standardized over
    all its segments; segments no window read stay at -1e9."""
    merged = np.full(n, -1e9)
    for first, v in windows:
        v = np.asarray(v, dtype=float)
        v = np.where(np.isfinite(v), v, -1e9)
        merged[first:first + len(v)] = (v - v.mean()) / (v.std() + 1e-12)
    return merged


def shortlist(scores, candidate, k):
    """The first `k` candidates by score (stable: the earlier segment keeps a
    tie), in document order."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return sorted(int(i) for i in [i for i in order if candidate[i]][:k])


def merge_cases(seed):
    rng = np.random.default_rng(seed)
    cases = []
    # prose: titles, paragraph breaks, three windows, the sum reading
    lines = []
    for p in range(12):
        lines.append(f"# Title {p}")
        lines += [f"Sentence {p}.{s} of paragraph {p}." for s in range(int(rng.integers(2, 5)))]
        lines.append("")
    lines.pop()
    costs = [1 if line == "" else int(rng.integers(4, 10)) for line in lines]
    empty = [line == "" for line in lines]
    wins = cut_windows(costs, empty, 70)
    heads = 6
    readings = []
    for first, end in wins:
        owned = [lines[i] != "" for i in range(first, end)]
        keys, span = synthetic_window(rng, owned)
        q, na = rows(rng, heads, span, peaks=[keys[1][0]]), rows(rng, heads, span)
        readings.append([first, reading_scores(q, na, keys, span, "sum").tolist()])
    merged = merge_prose(len(lines), readings)
    candidate = [line != "" and not line.startswith("# ") for line in lines]
    cases.append({"name": "prose in windows", "rule": "prose", "segments": len(lines), "windows": readings,
                  "merged": merged.tolist(), "candidate": candidate, "k": 16,
                  "shortlist": shortlist(merged, candidate, 16), "lines": lines})
    # records: every record owns keys, the end reading
    n = 45
    costs = [int(rng.integers(8, 20)) for _ in range(n)]
    wins = cut_windows(costs, [False] * n, 200)
    readings = []
    for first, end in wins:
        keys, span = synthetic_window(rng, [True] * (end - first))
        q, na = rows(rng, heads, span, peaks=[keys[0][1] - 1]), rows(rng, heads, span)
        readings.append([first, reading_scores(q, na, keys, span, "end").tolist()])
    merged = merge_records(n, readings)
    candidate = [True] * n
    cases.append({"name": "records in windows", "rule": "records", "segments": n, "windows": readings,
                  "merged": merged.tolist(), "candidate": candidate, "k": 16,
                  "shortlist": shortlist(merged, candidate, 16)})
    # one window, fewer candidates than k: all of them
    keys, span = synthetic_window(rng, [True, False, True, True])
    q, na = rows(rng, 3, span), rows(rng, 3, span)
    v = reading_scores(q, na, keys, span, "end")
    merged = merge_records(4, [[0, v.tolist()]])
    candidate = [True, False, True, True]
    cases.append({"name": "fewer candidates than k", "rule": "records", "segments": 4, "windows": [[0, v.tolist()]],
                  "merged": merged.tolist(), "candidate": candidate, "k": 16,
                  "shortlist": shortlist(merged, candidate, 16)})
    # a tie keeps the earlier segment
    tie = [1.0, 3.0, 3.0, 2.0, 3.0]
    cases.append({"name": "ties keep the earlier segment", "rule": "records", "segments": 5, "windows": [[0, tie]],
                  "merged": merge_records(5, [[0, tie]]).tolist(), "candidate": [True] * 5, "k": 2,
                  "shortlist": shortlist(merge_records(5, [[0, tie]]), [True] * 5, 2)})
    return cases


# ---------------------------------------------------------------------------
# renders

def render_cases():
    lines = ["# Alpha", "Alpha one.", "Alpha two.", "Alpha three.", "", "# Beta", "Beta one.", "Beta two.",
             "", "# Gamma", "Gamma one.", "", "Loose line without a title."]
    cases = []
    for name, cand in (("one candidate", [2]), ("two in one paragraph", [1, 3]),
                       ("across paragraphs", [7, 10, 2]), ("a title-less paragraph", [12, 6])):
        text, labels = prose_render(lines, set(cand))
        cases.append({"name": name, "kind": "prose", "lines": lines, "candidates": cand, "text": text,
                      "labels": {k: v for k, v in labels.items()}})
    labelled = ["first line", "second: with a colon", "  indented", "ünïcödé"]
    cases.append({"name": "labelled lines", "kind": "labelled", "lines": labelled,
                  "text": ask_state(labelled)})
    # A fold's level 1: the kept templates' level-1 lines, in their order
    # (`zd_logpipe.py`: `[f.level1[i] for i in sorted(cand)]`).
    shapes = ["pod-{p} ready in {i}ms", "pod-{p} failed with exit code {i} after restart",
              "evicted pod-{p} from node for memory pressure", "scaled app web to {i} replicas now please",
              "[cron] job backup-{p} finished", "certificate for host{p}.example.org renews on day {i}"]
    log = [f"2026-09-28T10:00:{i:02d}Z " + shapes[i % 6].format(p=i % 4, i=i) for i in range(24)]
    log += ["2026-09-28T10:01:00Z pod-9 ready in 9ms", "2026-09-28T10:01:01Z pod-9 ready in 9ms"]
    f = C.fold(log, C.SIM, values=True)
    kept = [0, 2, 3] if len(f.clusters) > 3 else list(range(len(f.clusters)))
    cases.append({"name": "a fold's level 1", "kind": "level1", "lines": log, "kept": kept,
                  "text": ask_state([f.level1[i] for i in sorted(kept)])})
    # A fold's last `choice` over the kept rows' original lines
    # (`zd_notfound.py --raw-final`: `sorted(members[i][0] for i in rows)`).
    ci = 0
    texts, members = C.level2(f, ci)
    rows = [len(texts) - 1, 0, 1] if len(texts) > 2 else list(range(len(texts)))
    raw = sorted(members[i][0] for i in rows)
    cases.append({"name": "a fold's last choice", "kind": "raw_final", "lines": log, "cluster": ci, "rows": rows,
                  "raw": raw, "text": ask_state([log[i] for i in raw])})
    return cases


def ask_state(lines):
    """The state `zd_notfound.ask` (and `zd_records.ask` through it) sends:
    each candidate as `label: line`, one per line."""
    return "\n".join(f"{label}: {line}" for label, line in zip(LABELS, lines))


def records_render_case():
    """`zd_records.py ask`: the candidates in array order as spaced JSON,
    labelled (`ask_lines(url, [line(q["state"][i]) for i in cand], ...)`)."""
    from zd_records import line
    records = [{"id": 7, "name": "Ada Silva", "city": "Porto"}, {"id": 8, "name": "Bo", "tags": ["x", "y"], "ok": True},
               {"id": 9, "note": "quote \" and ünï"}, {"id": 10, "nested": {"a": None}}]
    cand = [3, 0, 2]
    # Each record as its JSON text: a reader of the fixture must see the keys
    # in the order they were written.
    return {"records": [json.dumps(r, ensure_ascii=False) for r in records], "candidates": cand,
            "text": ask_state([line(records[i]) for i in sorted(cand)])}


SPACED = [
    '{"id": 1, "name": "Ada", "city": "Paris"}',
    '{"z": 1, "a": 2, "m": {"y": [1, 2, {"k": "v"}], "b": null}}',
    '{"text": "line\\nbreak\\ttab \\"quote\\" back\\\\slash \\u0001 \\u007f \\u2028 ünï 名"}',
    '{"ok": true, "no": false, "none": null, "neg": -3, "dec": 1.5, "zero": 0, "e": [], "o": {}}',
    '["a", 1, [2, 3], {"x": "y"}]',
    '"a plain string"',
    '12',
    '{"price": 149.0, "qty": 2, "ratio": -0.25}',
]


# ---------------------------------------------------------------------------
# auto

def auto(state):
    if (isinstance(state, list) and len(state) >= 2 and all(isinstance(x, dict) for x in state)
            and not all(isinstance(x.get("type"), str) for x in state)):
        return "records", None
    if isinstance(state, str):
        segments = state.split("\n")
    else:
        segments = [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in state]
    content = [s for s in segments if s.strip()][:2000]
    f = C.fold(content, C.SIM, values=True)
    share = sum(len(c.members) for c in f.clusters if len(c.members) >= 2) / len(content)
    return ("log" if share >= 0.5 else "prose"), share


def auto_cases(seed):
    rng = np.random.default_rng(seed)
    log = "\n".join(f"2026-09-28T10:{i // 60:02d}:{i % 60:02d}Z pod-{i % 7} ready in {int(rng.integers(1, 900))}ms"
                    for i in range(300))
    words = ["river", "stone", "light", "garden", "winter", "music", "letter", "window", "harbour", "forest",
             "silver", "morning", "shadow", "voice", "paper", "market"]
    prose = "\n".join(" ".join(rng.choice(words, size=int(rng.integers(4, 14)))) + "." for _ in range(200))
    half = "\n".join([f"step {i} ok" for i in range(10)] + [f"{w} {v} unique" for w, v in zip(words, range(10))])
    # the first 2,000 fold as a log; all 4,100 would be read as prose
    long_log = "\n".join(f"tick {i}" for i in range(2000)) + "\n" + "\n".join(
        f"w{i}a w{i}b w{i}c" for i in range(2100))
    states = [
        ("a log", log),
        ("prose", prose),
        ("half and half", half),
        ("records", [{"id": i, "name": f"n{i}"} for i in range(5)]),
        ("records with a type key that is not all strings", [{"type": "a", "id": 1}, {"type": 2, "id": 2}]),
        ("content-parts shaped", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
        ("one object", [{"id": 1}]),
        ("strings", ["GET /a 200", "GET /b 200", "GET /c 500", "POST /d 201"]),
        ("numbers", [1, 2, 3, 4]),
        ("mixed", [{"id": 1}, "text", 3]),
        ("empty lines left out", "\n\n  \nalpha beta\nalpha gamma\n\ndelta epsilon zeta\n"),
        ("only the first 2000 segments", long_log),
    ]
    cases = []
    for name, state in states:
        kind, share = auto(state)
        cases.append({"name": name, "state": state, "kind": kind, "share": share})
    return cases


# ---------------------------------------------------------------------------
# found

def found_cases():
    out = []
    for route, p_none, p_yes in (("log", 0.1, 0.9), ("log", 0.6, 0.35), ("log", 0.5, 0.5), ("log", 0.2, 0.1),
                                 ("prose", 0.49, None), ("prose", 0.51, None), ("records", 0.0, None),
                                 ("records", 1.0, None)):
        found = (1 - p_none + p_yes) / 2 if route == "log" else 1 - p_none
        out.append({"route": route, "p_none": p_none, "p_yes": p_yes, "found": found, "found_is": found >= 0.5})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261180)
    args = ap.parse_args()
    source = "tools/locate-sets/golden22.py"

    def write(name, payload):
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict({"source": source}, **payload), f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(path)

    write("locate_fold.json", {"sim": C.SIM, "cases": [fold_case(n, l) for n, l in FOLD_INPUTS.items()]})
    write("locate_readings.json", {"arithmetic": "f64 key features (zd_logpipe), zsum per head, summed",
                                   "readings": reading_cases(args.seed), "windows": window_cases(args.seed + 1),
                                   "merges": merge_cases(args.seed + 2)})
    spaced = [{"json": text, "spaced": json.dumps(json.loads(text), ensure_ascii=False)} for text in SPACED]
    write("locate_renders.json", {"labels": LABELS, "renders": render_cases(), "spaced_json": spaced,
                                  "records": records_render_case(), "found": found_cases()})
    write("locate_auto.json", {"segments": 2000, "threshold": 0.5, "cases": auto_cases(args.seed + 3)})


if __name__ == "__main__":
    main()
