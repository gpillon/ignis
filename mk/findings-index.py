#!/usr/bin/env python3
"""The findings indexes, generated from each finding's own header, and their check.

    python mk/findings-index.py --write      regenerate docs/findings/README.md and ARCHIVE.md
    python mk/findings-index.py --check      refuse a finding or an index out of the convention
    python mk/findings-index.py --selftest   the tool's own test, on fixtures in a temporary directory

`make records` runs --write and `make records-check` runs --check, which CI
runs as well (`python` on Windows, `python3` on Linux; the standard library
only). The convention it holds the files to is docs/agents/findings.md: the
header fields of every finding, in their order, the Status and Topic
vocabularies, a one-line Summary of at most 240 characters, the links that tie
a corrected or superseded finding to what revised or replaced it, and the two
indexes, which nobody edits by hand.

README.md lists the usable findings, `current` and `corrected`, one section
per Topic in TOPICS order, newest first; ARCHIVE.md lists the `superseded` and
`archived` ones the same way. No finding's file ever moves: the guide and the
other records link them, and the guide's build refuses a dead link.
"""

import argparse
import datetime
import difflib
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
from pathlib import Path

KINDS = ("discovery", "research", "experiment")
STATUSES = ("current", "corrected", "superseded", "archived")
USABLE = ("current", "corrected")

# The Topic vocabulary, in the indexes' display order. A new topic is added
# here and to the table in docs/agents/findings.md, in the same change.
TOPICS = (
    ("decode-27b", "27B decode round, kernels and speculation"),
    ("flash-next", "Flash-Next: MoE experts, residency, MTP, verify rows"),
    ("prefill", "Prefill: chunk width, TTFT, prompt reuse"),
    ("kv-and-state", "KV cache, formats and state tiers"),
    ("artifacts-and-gates", "Model artifacts, the 16 GB trellis and quality gates"),
    ("host-and-gpu", "Host, VRAM and GPU generations"),
    ("decide-options", "Decide: options, number, scalar, prompt and reuse"),
    ("decide-pointing", "Decide: point and box, attention heads"),
    ("decide-locate", "Decide: locate"),
    ("reasoning-and-context", "Reasoning, loops and long context"),
    ("vision", "Vision"),
    ("speech", "Speech"),
    ("image", "Image"),
    ("harness-and-build", "Harness, bench, build and transport"),
)
TOPIC_SLUGS = tuple(slug for slug, _ in TOPICS)

# Every header field, in its order; `Corrected by` only when Status is corrected.
FIELDS = (
    "Kind",
    "Status",
    "Observed",
    "Last verified",
    "Scope",
    "Topic",
    "Related",
    "Superseded by",
    "Corrected by",
    "Summary",
)
SUMMARY_MAX = 240
INDEXES = ("README.md", "ARCHIVE.md")

NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
FIELD_RE = re.compile(r"^- ([A-Z][A-Za-z -]*?):(?:\s+(.*))?$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LINK_RE = re.compile(r"\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

REPO = Path(__file__).resolve().parents[1]
FINDINGS = REPO / "docs" / "findings"


class Finding:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.title = None
        self.fields = {}

    status = property(lambda self: self.fields.get("Status", ""))
    observed = property(lambda self: self.fields.get("Observed", ""))
    topic = property(lambda self: self.fields.get("Topic", ""))
    summary = property(lambda self: self.fields.get("Summary", ""))


def read(path):
    # utf-8-sig drops a byte-order mark; the newline swap keeps a CRLF
    # checkout from failing --check.
    with open(path, encoding="utf-8-sig") as f:
        return f.read().replace("\r\n", "\n")


def parse(path):
    """The finding at `path` and the problems of its title and header, which
    is the `- Field: value` list between the title and the first blank line;
    a line indented under a field continues it."""
    finding = Finding(path)
    problems = []
    try:
        lines = read(path).split("\n")
    except UnicodeDecodeError:
        return finding, ["not UTF-8 text"]
    if lines[0].startswith("# ") and lines[0][2:].strip():
        finding.title = lines[0][2:].strip()
    else:
        problems.append("line 1 is not the title, `# Title`")
    i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    header = []  # [name, value, line count]
    while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
        line = lines[i]
        m = FIELD_RE.match(line)
        if m:
            header.append([m.group(1), (m.group(2) or "").strip(), 1])
        elif line[:1] in (" ", "\t") and header:
            header[-1][1] = (header[-1][1] + " " + line.strip()).strip()
            header[-1][2] += 1
        else:
            problems.append("line %d is neither a `- Field: value` line nor the indented continuation of one" % (i + 1))
        i += 1
    if not header:
        problems.append("no `- Field: value` header under the title")
        return finding, problems

    names = [name for name, _, _ in header]
    seen = []
    for name, value, count in header:
        if name not in FIELDS:
            problems.append("unknown field `%s`" % name)
        elif name in seen:
            problems.append("field `%s` appears twice" % name)
        else:
            seen.append(name)
            finding.fields[name] = value
            if name == "Summary" and count > 1:
                problems.append("`Summary` runs over %d lines: it is one line" % count)

    status = finding.fields.get("Status")
    expected = [f for f in FIELDS if f != "Corrected by" or status == "corrected"]
    for name in expected:
        if name not in finding.fields:
            problems.append("missing field `%s`" % name)
    if "Corrected by" in finding.fields and status != "corrected":
        problems.append("`Corrected by` belongs only to a `corrected` finding")
    present = [f for f in expected if f in finding.fields]
    if [n for n in seen if n in expected] != present:
        problems.append("fields out of order: the order is %s" % ", ".join(expected))

    problems += check_values(finding)
    return finding, problems


def check_values(finding):
    problems = []
    f = finding.fields
    if "Kind" in f and f["Kind"] not in KINDS:
        problems.append("Kind `%s` is not one of %s" % (f["Kind"], ", ".join(KINDS)))
    status = f.get("Status")
    if status is not None and status not in STATUSES:
        problems.append("Status `%s` is not one of %s" % (status, ", ".join(STATUSES)))
    dates = {}
    for name in ("Observed", "Last verified"):
        if name in f:
            try:
                if not DATE_RE.match(f[name]):
                    raise ValueError
                dates[name] = datetime.date.fromisoformat(f[name])
            except ValueError:
                problems.append("%s `%s` is not a YYYY-MM-DD date" % (name, f[name]))
    if len(dates) == 2 and dates["Last verified"] < dates["Observed"]:
        problems.append("Last verified is earlier than Observed")
    for name in ("Scope", "Related", "Superseded by"):
        if name in f and not f[name]:
            problems.append("`%s` is empty (`none` when there is nothing)" % name)
    if "Topic" in f and f["Topic"] not in TOPIC_SLUGS:
        problems.append("Topic `%s` is not in the vocabulary (TOPICS in mk/findings-index.py)" % f["Topic"])
    if "Summary" in f:
        if not f["Summary"]:
            problems.append("`Summary` is empty")
        elif len(f["Summary"]) > SUMMARY_MAX:
            problems.append("`Summary` is %d characters, over %d" % (len(f["Summary"]), SUMMARY_MAX))

    superseded_by = f.get("Superseded by")
    if superseded_by and status in STATUSES:
        if status == "superseded":
            relative = [t for t in LINK_RE.findall(superseded_by) if not SCHEME_RE.match(t) and not t.startswith("#")]
            if not relative:
                problems.append("a superseded finding links its replacement under `Superseded by`, a relative link to a finding or an ADR")
        elif superseded_by != "none":
            problems.append("Status `%s` takes `Superseded by: none`" % status)
    corrected_by = f.get("Corrected by")
    if status == "corrected" and corrected_by is not None and not LINK_RE.search(corrected_by):
        problems.append("a corrected finding links what revised it under `Corrected by`")
    for name in ("Superseded by", "Corrected by"):
        for target in LINK_RE.findall(f.get(name) or ""):
            if SCHEME_RE.match(target) or target.startswith("#"):
                continue
            local = urllib.parse.unquote(target.split("#", 1)[0])
            if not (finding.path.parent / local).is_file():
                problems.append("`%s` links %s, which does not exist" % (name, target))
    return problems


def load(findings_dir):
    """Every finding in `findings_dir` (its *.md files but the two indexes),
    sorted by name, and the problems of each as "path: problem"."""
    findings, problems = [], []
    shown = findings_dir.parent.parent
    for path in sorted(findings_dir.iterdir()):
        if not path.is_file() or path.suffix != ".md" or path.name in INDEXES:
            continue
        finding, found = parse(path)
        if not NAME_RE.match(path.name):
            found.insert(0, "the file name is not YYYY-MM-DD-short-descriptive-slug.md")
        where = path.relative_to(shown).as_posix()
        problems += ["%s: %s" % (where, p) for p in found]
        findings.append(finding)
    problems += check_topic_table(findings_dir.parent / "agents" / "findings.md", shown)
    return findings, problems


def check_topic_table(convention, shown):
    """The convention's Topic table lists TOPICS, in their order."""
    if not convention.is_file():
        return []
    slugs = re.findall(r"^\| `([a-z0-9-]+)` \|", read(convention), re.M)
    if tuple(slugs) == TOPIC_SLUGS:
        return []
    return ["%s: the Topic table lists %s, not TOPICS in mk/findings-index.py (%s)" % (
        convention.relative_to(shown).as_posix(), ", ".join(slugs) or "nothing", ", ".join(TOPIC_SLUGS))]


HEADER_README = """\
# Findings

A finding is a durable, reusable, evidence-backed result of discovery,
research or an experiment; [`docs/agents/findings.md`](../agents/findings.md)
says when and how to write one. This index lists the usable findings, by topic
and newest first, each with its conclusion in one sentence. The superseded and
archived ones, kept as history, are in [ARCHIVE.md](ARCHIVE.md).

`mk/findings-index.py` generates this file and ARCHIVE.md from the header of
each finding (`make records`), and CI refuses them out of sync: they are never
edited by hand.

- `current`: usable as it is.
- `corrected`: the conclusion holds, but a number, a threshold or a part was
  revised later; the entry links what revised it, and its summary says what
  holds now.
- `superseded`: replaced by the finding or the record it links; in the archive.
- `archived`: a closed line of research (a negative result, a design not
  implemented, an exploration not followed); in the archive.
"""

HEADER_ARCHIVE = """\
# Findings archive

The findings that no longer stand as they are, kept as history: the
`superseded` ones next to what replaced them, the `archived` ones, closed lines
of research, with why they closed. They are listed by topic, newest first, and
their files stay in this directory, since the guide and the other records link
them. The usable findings are in [README.md](README.md);
[`docs/agents/findings.md`](../agents/findings.md) is the convention.

`mk/findings-index.py` generates this file (`make records`): it is never edited
by hand.
"""


def entry(finding):
    title = finding.title.replace("[", "\\[").replace("]", "\\]")
    line = "- %s [%s](%s)" % (finding.observed, title, finding.name)
    if finding.status == "corrected":
        line += ", corrected by " + finding.fields["Corrected by"]
    elif finding.status == "superseded":
        line += ", superseded by " + finding.fields["Superseded by"]
    elif finding.status == "archived":
        line += ", archived"
    return line + ": " + finding.summary


def sections(findings):
    out = []
    for slug, name in TOPICS:
        listed = [f for f in findings if f.topic == slug]
        if not listed:
            continue
        listed.sort(key=lambda f: (f.observed, f.name), reverse=True)
        out.append("\n## %s (`%s`)\n\n" % (name, slug) + "".join(entry(f) + "\n" for f in listed))
    return "".join(out)


def render(findings):
    """The two indexes' text, keyed by file name."""
    usable = [f for f in findings if f.status in USABLE]
    history = [f for f in findings if f.status not in USABLE]
    archive = sections(history) or "\nNo finding is superseded or archived.\n"
    return {"README.md": HEADER_README + sections(usable), "ARCHIVE.md": HEADER_ARCHIVE + archive}


def write(findings_dir, out=sys.stdout, err=sys.stderr):
    findings, problems = load(findings_dir)
    if problems:
        for p in problems:
            print(p, file=err)
        print("findings-index: %d problem(s): nothing written" % len(problems), file=err)
        return 1
    for name, text in render(findings).items():
        path = findings_dir / name
        if path.is_file() and read(path) == text:
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote %s (%d bytes)" % (path.relative_to(findings_dir.parent.parent).as_posix(), len(text.encode("utf-8"))), file=out)
    usable = sum(1 for f in findings if f.status in USABLE)
    print("findings-index: %d findings, %d usable, %d in the archive" % (len(findings), usable, len(findings) - usable), file=out)
    return 0


def check(findings_dir, out=sys.stdout, err=sys.stderr):
    findings, problems = load(findings_dir)
    if not problems:
        for name, text in render(findings).items():
            path = findings_dir / name
            where = path.relative_to(findings_dir.parent.parent).as_posix()
            if not path.is_file():
                problems.append("%s: missing: run `make records`" % where)
                continue
            have = read(path)
            if have != text:
                problems.append("%s: out of sync with the findings, or edited by hand: run `make records`" % where)
                diff = difflib.unified_diff(have.split("\n"), text.split("\n"), where, "generated", n=0, lineterm="")
                problems += ["    " + d[:160] for d in list(diff)[:12]]
    elif findings:
        problems.append("(the indexes are compared once every finding passes)")
    if problems:
        for p in problems:
            print(p, file=err)
        return 1
    print("findings-index: %d findings, indexes in sync" % len(findings), file=out)
    return 0


# ── the selftest ────────────────────────────────────────────────────────────


def fixture(**fields):
    """A finding's text: the default header below with the fields named
    overridden (lowercase, `_` for a space). None drops a field, a list
    continues it onto indented lines, `extra` inserts (position, (field,
    value)) pairs, and `order` lists the fields in the order to write them."""
    head = [
        ("Kind", "experiment"),
        ("Status", "current"),
        ("Observed", "2026-01-03"),
        ("Last verified", "2026-01-03"),
        ("Scope", "kernel / prefill chunking"),
        ("Topic", "prefill"),
        ("Related", "none"),
        ("Superseded by", "none"),
        ("Summary", "A chunk of 1,024 tokens costs 10 ms on the test card."),
    ]
    title = fields.pop("title", "A measured result")
    order = fields.pop("order", None)
    extra = fields.pop("extra", [])
    head = [(k, fields.pop(k.lower().replace(" ", "_"), v)) for k, v in head]
    if fields:
        raise KeyError(sorted(fields))
    for pos, pair in extra:
        head.insert(pos, pair)
    if order:
        values = dict(head)
        head = [(k, values[k]) for k in order]
    lines = ["# " + title if title is not None else "A title without its mark", ""]
    for k, v in head:
        if v is None:
            continue
        if isinstance(v, list):
            lines.append("- %s: %s" % (k, v[0]))
            lines += ["  " + more for more in v[1:]]
        else:
            lines.append(("- %s: %s" % (k, v)).rstrip())
    return "\n".join(lines + ["", "## Question", "", "Text.", ""])


NEW = "2026-01-03-new-measure.md"
FIRST = "2026-01-02-first-reading.md"
OLD = "2026-01-01-old-measure.md"
CLOSED = "2026-01-04-a-closed-road.md"
SPEECH = "2026-01-05-speech-at-the-limit.md"
FULL = "S" * (SUMMARY_MAX - 1) + "."


def clean_tree():
    return {
        "docs/adr/0001-a-decision.md": "# A decision\n",
        "docs/agents/findings.md": "# Findings\n\n| Topic | What it holds |\n|---|---|\n"
        + "".join("| `%s` | %s |\n" % pair for pair in TOPICS),
        "docs/findings/" + NEW: fixture(
            title="The new measure",
            related=["[ADR 0001](../adr/0001-a-decision.md),", "[the first reading](%s)" % FIRST],
        ),
        "docs/findings/" + FIRST: fixture(
            title="The first reading",
            status="corrected",
            observed="2026-01-02",
            last_verified="2026-01-03",
            extra=[(8, ("Corrected by", "[the new measure](%s#evidence)" % NEW))],
            summary="The chunk costs 10 ms, not the 12 ms first read.",
        ),
        "docs/findings/" + OLD: fixture(
            title="The old measure",
            status="superseded",
            observed="2026-01-01",
            last_verified="2026-01-01",
            superseded_by="[the new measure](%s), [ADR 0001](../adr/0001-a-decision.md)" % NEW,
            summary="A chunk cost 12 ms, before the fix.",
        ),
        "docs/findings/" + CLOSED: fixture(
            title="A closed road",
            status="archived",
            kind="research",
            observed="2026-01-04",
            last_verified="2026-01-04",
            topic="vision",
            summary="Not followed: the tower costs too much at this width.",
        ),
        "docs/findings/" + SPEECH: fixture(
            title="Speech at the limit",
            observed="2026-01-05",
            last_verified="2026-01-05",
            topic="speech",
            summary=FULL,
        ),
    }


def plant(root, tree):
    for rel, text in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


class Sink:
    def __init__(self):
        self.lines = []

    def write(self, s):
        self.lines += [l for l in s.split("\n") if l]

    def flush(self):
        pass


def edit(path, old, new):
    text = read(path)
    if old not in text:
        raise AssertionError("fixture edit: %r not in %s" % (old, path.name))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(old, new, 1))


def put(name, **fields):
    return lambda d: plant(d.parent.parent, {"docs/findings/" + name: fixture(**fields)})


# (what is wrong, how to break the clean tree, the file the problem names, a phrase of it)
CASES = [
    ("an index edited by hand", lambda d: edit(d / "README.md", "(`prefill`)", "(`prefill`, edited)"), "README.md", "out of sync"),
    ("an index missing", lambda d: (d / "ARCHIVE.md").unlink(), "ARCHIVE.md", "missing"),
    ("a stale index", put("2026-01-06-a-new-finding.md", observed="2026-01-06", last_verified="2026-01-06"), "README.md", "out of sync"),
    ("a field missing", put(NEW, title="The new measure", topic=None), NEW, "missing field `Topic`"),
    ("fields out of order", put(NEW, order=["Status", "Kind", "Observed", "Last verified", "Scope", "Topic", "Related", "Superseded by", "Summary"]), NEW, "out of order"),
    ("an unknown field", put(NEW, extra=[(2, ("Pre-registration", "da587d2"))]), NEW, "unknown field `Pre-registration`"),
    ("a field twice", put(NEW, extra=[(9, ("Summary", "Again."))]), NEW, "`Summary` appears twice"),
    ("a stray header line", lambda d: edit(d / NEW, "- Scope:", "Scope:"), NEW, "neither a `- Field: value` line"),
    ("no title", put(NEW, title=None), NEW, "not the title"),
    ("a file name off the pattern", lambda d: plant(d.parent.parent, {"docs/findings/Notes.md": fixture()}), "Notes.md", "file name"),
    ("a Kind outside the vocabulary", put(NEW, kind="benchmark"), NEW, "Kind `benchmark`"),
    ("a Status outside the vocabulary", put(NEW, status="current (pre-registration committed)"), NEW, "Status `current (pre-registration committed)`"),
    ("a Topic outside the vocabulary", put(NEW, topic="decode"), NEW, "Topic `decode`"),
    ("a date that is not one", put(NEW, observed="2026-13-01"), NEW, "Observed `2026-13-01`"),
    ("Last verified before Observed", put(NEW, last_verified="2026-01-02"), NEW, "earlier than Observed"),
    ("an empty Scope", put(NEW, scope=""), NEW, "`Scope` is empty"),
    ("the Summary missing", put(NEW, summary=None), NEW, "missing field `Summary`"),
    ("the Summary empty", put(NEW, summary=""), NEW, "`Summary` is empty"),
    ("the Summary on two lines", put(NEW, summary=["A chunk of 1,024 tokens", "costs 10 ms."]), NEW, "over 2 lines"),
    ("the Summary over 240 characters", put(NEW, summary=FULL + "x"), NEW, "241 characters"),
    ("superseded, by nothing", put(OLD, status="superseded"), OLD, "links its replacement"),
    ("superseded, by an issue only", put(OLD, status="superseded", superseded_by="[#1](https://github.com/o/r/issues/1)"), OLD, "links its replacement"),
    ("superseded, by a file that does not exist", put(OLD, status="superseded", superseded_by="[gone](2026-01-09-gone.md)"), OLD, "2026-01-09-gone.md, which does not exist"),
    ("corrected, with no Corrected by", put(FIRST, status="corrected"), FIRST, "missing field `Corrected by`"),
    ("corrected, by no link", put(FIRST, status="corrected", extra=[(8, ("Corrected by", "none"))]), FIRST, "links what revised it"),
    ("corrected, by a file that does not exist", put(FIRST, status="corrected", extra=[(8, ("Corrected by", "[gone](../adr/0009-gone.md)"))]), FIRST, "0009-gone.md, which does not exist"),
    ("current, with a Superseded by", put(NEW, superseded_by="[the old measure](%s)" % OLD), NEW, "Status `current` takes `Superseded by: none`"),
    ("archived, with a Superseded by", put(CLOSED, status="archived", topic="vision", superseded_by="[the new measure](%s)" % NEW), CLOSED, "Status `archived` takes `Superseded by: none`"),
    ("a Topic table off TOPICS", lambda d: edit(d.parent / "agents" / "findings.md", "| `image` | Image |\n", ""), "agents/findings.md", "Topic table lists"),
    ("a Corrected by on a current finding", put(NEW, extra=[(8, ("Corrected by", "[the old measure](%s)" % OLD))]), NEW, "belongs only to a `corrected` finding"),
]


def selftest():
    failed = []

    def fail(what):
        print("  FAIL: " + what, file=sys.stderr)
        failed.append(what)

    with tempfile.TemporaryDirectory(prefix="findings-index-") as tmp:
        base = Path(tmp)

        def fresh(name):
            root = base / re.sub(r"[^a-z0-9]+", "-", name.lower())
            plant(root, clean_tree())
            d = root / "docs" / "findings"
            out, err = Sink(), Sink()
            if write(d, out, err) != 0:
                fail("the clean tree does not write: %s" % err.lines)
            return d

        # The clean tree: it writes, it checks, and a second write changes nothing.
        d = fresh("clean")
        out, err = Sink(), Sink()
        if check(d, out, err) != 0:
            fail("the clean tree fails its check: %s" % err.lines)
        readme, archive = read(d / "README.md"), read(d / "ARCHIVE.md")
        out = Sink()
        write(d, out, Sink())
        if any(l.startswith("wrote") for l in out.lines):
            fail("a second --write rewrote an index: %s" % out.lines)
        def listed(text):
            return [m.group(1) for m in re.finditer(r"^- \d{4}-\d{2}-\d{2} \[[^\]]*\]\(([^)]+)\)", text, re.M)]

        def heading(text, slug):
            return text.find("(`%s`)\n" % slug)

        expect = [
            (listed(readme) == [NEW, FIRST, SPEECH], "README.md lists the current and corrected findings, newest first within a topic"),
            (listed(archive) == [OLD, CLOSED], "ARCHIVE.md lists the superseded and archived ones"),
            (0 < heading(readme, "prefill") < heading(readme, "speech"), "README.md follows the topic order"),
            (0 < heading(archive, "prefill") < heading(archive, "vision"), "ARCHIVE.md follows the topic order"),
            (heading(readme, "decode-27b") < 0 and heading(readme, "vision") < 0 and heading(archive, "speech") < 0, "an empty topic gets no section"),
            (", corrected by [the new measure](%s#evidence): " % NEW in readme, "a corrected entry links what revised it"),
            (", superseded by [the new measure](%s), [ADR 0001]" % NEW in archive, "a superseded entry links what replaced it"),
            (", archived: Not followed" in archive, "an archived entry says it is archived"),
            ("- 2026-01-03 [The new measure](%s): A chunk of 1,024" % NEW in readme, "an entry gives the date, the linked title and the summary"),
            (FULL in readme, "a Summary of exactly 240 characters passes"),
        ]
        for ok, what in expect:
            print("  %s  %s" % ("ok  " if ok else "FAIL", what))
            if not ok:
                failed.append(what)

        # The command line: exit 0 on the clean tree (each case below, 1).
        if main(["--check"], d, Sink(), Sink()) != 0:
            fail("--check exits non-zero on the clean tree")
        else:
            print("  ok    --check passes the clean tree")

        # --write refuses a broken finding and leaves the indexes alone.
        d = fresh("write refuses")
        put(NEW, topic="decode")(d)
        err = Sink()
        if main(["--write"], d, Sink(), err) != 1 or read(d / "README.md") != readme:
            fail("--write wrote over a finding with a problem")
        elif not any(NEW in l and "Topic" in l for l in err.lines):
            fail("--write did not name the problem: %s" % err.lines)
        else:
            print("  ok    --write refuses a finding with a problem and writes nothing")

        for what, breaks, where, phrase in CASES:
            d = fresh(what)
            breaks(d)
            err = Sink()
            code = main(["--check"], d, Sink(), err)
            hit = [l for l in err.lines if where in l.split(": ", 1)[0] and phrase in l]
            if code != 1 or not hit:
                fail("%s: exit %d, expected `%s: ...%s...`, got %s" % (what, code, where, phrase, err.lines))
            else:
                print("  ok    refused: %s" % what)

    if failed:
        print("selftest: %d failure(s)" % len(failed), file=sys.stderr)
        return 1
    print("selftest ok: %d checks" % (len(expect) + 2 + len(CASES)))
    return 0


def main(argv=None, findings_dir=None, out=None, err=None):
    parser = argparse.ArgumentParser(description="Generate and check the findings indexes (docs/agents/findings.md).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate docs/findings/README.md and ARCHIVE.md")
    mode.add_argument("--check", action="store_true", help="refuse a finding or an index out of the convention")
    mode.add_argument("--selftest", action="store_true", help="the tool's own test, on fixtures in a temporary directory")
    args = parser.parse_args(argv)
    findings_dir = findings_dir or FINDINGS
    out, err = out or sys.stdout, err or sys.stderr
    if args.selftest:
        return selftest()
    if args.write:
        return write(findings_dir, out, err)
    return check(findings_dir, out, err)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except AttributeError:
            pass
    sys.exit(main())
