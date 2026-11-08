#!/usr/bin/env python3
"""The ADR index, generated from each ADR's frontmatter, and its check (ADR 0081).

    python3 mk/adr-index.py --write      regenerate docs/adr/README.md
    python3 mk/adr-index.py --check      fail on a wrong frontmatter or a stale index
    python3 mk/adr-index.py --check --since REV
                                         and on an earlier ADR's text changed since REV
    python3 mk/adr-index.py --selftest   the check's own test, on fixtures in a temp directory

`make records` and `make records-check` run it with the host's Python
(`python` on Windows, `python3` elsewhere: the PYTHON hook of mk/os/<os>.mk),
and the records workflow runs it on every push that touches docs/adr/. It is
the standard library alone, so it needs no virtual environment.

Every `docs/adr/NNNN-*.md` opens with a frontmatter and a blank line before its
heading:

    ---
    status: accepted
    amends: [ADR-0016]
    supersedes: [ADR-0008]
    amended-by: [ADR-0045, ADR-0054]
    ---

`status` is one of `proposed`, `accepted`, `deprecated`, `superseded by
ADR-NNNN` (the /domain-modeling vocabulary); the three lists are optional, in
that order, ascending, and left out when empty. The check refuses anything else
in the block, a reference to an ADR that does not exist, a back-link missing on
either side (X `amends` Y with no `amended-by: X` on Y, X `supersedes` Y with no
`status: superseded by X` on Y, and each the other way), a heading from 0070 on
that is not the bare title, and an index that is not what --write would write.

--write writes the index even when the check would fail, so the table shows
what the files say, and prints the problems it found; --check is the gate.

With --since, the check also holds every ADR that existed at the merge base of
REV and HEAD, and was no longer `proposed` there, to its text at that base: a
decision that changes is a new ADR, and the old one changes only in its
frontmatter (ADR 0081). The text is compared as mk/changelog.sh compares it, the
frontmatter dropped and every link target and every path in backticks blanked,
so a status, a back-link or a moved file's path is not a change; the working
tree is compared, so an edit is refused before it is committed. A typo fix
passes when every commit since the base that changes the ADR's text names it in
its message, `[typo ADR-0032]` (or `[typo ADR-0032, ADR-0045]`), and nothing on
top of them is left uncommitted; the release notes then do not call it amended. An
ADR that existed at the base is never deleted.
`--since auto` is the merge base with origin/main, which `make records-check`
uses, and is skipped with a note in a clone with no origin/main or no git; the
records workflow passes the push's or the pull request's base.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

KEYS = ("status", "amends", "supersedes", "amended-by")
LIST_KEYS = KEYS[1:]
STATUS = re.compile(r"(?:proposed|accepted|deprecated|superseded by (ADR-\d{4}))")
REF_LIST = re.compile(r"\[(ADR-\d{4}(?:, ADR-\d{4})*)\]")
KEY_LINE = re.compile(r"([A-Za-z_-]+): (.*)")
ADR_FILE = re.compile(r"(\d{4})-[a-z0-9][a-z0-9-]*\.md")
# The short format (ADR 0070) takes the number from the file name only.
BARE_TITLE_FROM = 70
NUMBERED = re.compile(r"ADR[ -]?\d{4}\b")
# "ADR 0001 — title" in the long form; the index shows the title alone.
NUMBER_PREFIX = re.compile(r"ADR[ -]?\d{4}\s*(?:—|–|-|:)\s*")
# --since: the commit message's mark of a typo fix, and a frontmatter's status.
TYPO = re.compile(r"\[typo (ADR-\d{4}(?:, ADR-\d{4})*)\]")
FRONT_STATUS = re.compile(r"^status: (\S+)", re.MULTILINE)

INDEX = "README.md"

HEADER = """\
# Architecture decision records

An ADR records a decision that is hard to reverse, surprising without its
context and the result of a real trade-off, and why it was taken: the project's
memory of why ignis is built the way it is.

- **Writing one.** A new ADR is in the `/domain-modeling` skill's short format,
  its heading the bare title, with the frontmatter
  [ADR 0081](0081-an-adrs-status-is-in-its-frontmatter-and-a-changed-decision-is-a-new-adr.md)
  describes. [ADR 0070](0070-a-spec-is-a-github-issue-and-its-tickets-are-sub-issues.md)
  lists where the repo differs from the skill, and
  [docs/agents/domain.md](../agents/domain.md) gives the numbering rule.
- **A decision that changes** is recorded by a new ADR that amends or
  supersedes the old one, whose text then stays as it was: only its
  frontmatter changes (`amended-by`, or a status of `superseded by`).
  `make records-check` refuses any other change to an ADR that is no longer
  `proposed`, a typo fix aside when its commit message says `[typo ADR-NNNN]`.
- **This file is generated** from the frontmatter by `mk/adr-index.py`
  (`make records`), and `make records-check` fails when it is out of date:
  never edit it by hand.

| ADR | Title | Status | Amended by |
|---|---|---|---|
"""


class Adr:
    def __init__(self, path):
        self.path = path
        self.name = path.name
        self.number = path.name[:4]
        self.has_frontmatter = False
        self.status = None
        self.lists = {key: [] for key in LIST_KEYS}
        self.heading = None

    @property
    def superseded_by(self):
        match = STATUS.fullmatch(self.status or "")
        return match.group(1) if match else None

    @property
    def title(self):
        heading = self.heading or self.name
        prefix = NUMBER_PREFIX.match(heading)
        return heading[prefix.end():] if prefix else heading


def ref(number):
    return f"ADR-{number}"


def parse(path, problems):
    """One ADR's frontmatter and heading; what is wrong goes to `problems`."""
    adr = Adr(path)
    lines = path.read_text(encoding="utf-8").split("\n")

    def problem(message):
        problems.append(f"docs/adr/{adr.name}: {message}")

    def first_heading(start):
        for line in lines[start:]:
            if line.startswith("# "):
                return line[2:].strip()
        return None

    if not lines or lines[0] != "---":
        problem("no frontmatter: the file must open with a '---' line (ADR 0081)")
        adr.heading = first_heading(0)
        return adr
    try:
        end = lines.index("---", 1)
    except ValueError:
        problem("the frontmatter is never closed by a '---' line")
        adr.heading = first_heading(1)
        return adr
    adr.has_frontmatter = True

    seen = []
    for offset, line in enumerate(lines[1:end], start=2):
        match = KEY_LINE.fullmatch(line)
        if not match:
            problem(f"line {offset} is not a 'key: value' line: {line!r}")
            continue
        key, value = match.groups()
        if key not in KEYS:
            problem(f"unknown key {key!r} (the keys are {', '.join(KEYS)})")
            continue
        if key in seen:
            problem(f"{key} is given twice")
            continue
        seen.append(key)
        if key == "status":
            if STATUS.fullmatch(value):
                adr.status = value
            else:
                problem(f"status {value!r} is not one of proposed, accepted, deprecated, "
                        "'superseded by ADR-NNNN'")
            continue
        listed = REF_LIST.fullmatch(value)
        if not listed:
            problem(f"{key} {value!r} is not a list like [ADR-0001, ADR-0002] "
                    "(an empty one is left out)")
            continue
        numbers = [item[4:] for item in listed.group(1).split(", ")]
        if numbers != sorted(set(numbers)):
            problem(f"{key} is not in ascending order without repeats: {value}")
        adr.lists[key] = sorted(set(numbers))
    if "status" not in seen:
        problem("the frontmatter has no status")
    if seen != [key for key in KEYS if key in seen]:
        problem(f"the keys are out of order: {', '.join(seen)} (the order is {', '.join(KEYS)})")

    after = lines[end + 1:end + 3]
    if len(after) < 2 or after[0] != "" or not after[1].startswith("# "):
        problem("the frontmatter is not followed by one blank line and the '# ' heading")
        adr.heading = first_heading(end + 1)
    else:
        adr.heading = after[1][2:].strip()
    if adr.heading is None:
        problem("no '# ' heading")
    elif int(adr.number) >= BARE_TITLE_FROM and NUMBERED.match(adr.heading):
        problem(f"from ADR 0070 on the heading is the bare title, without its number: "
                f"{adr.heading!r}")
    return adr


def load(adr_dir):
    """Every ADR under `adr_dir` and every problem the check finds in them."""
    problems = []
    adrs = {}
    for path in sorted(adr_dir.glob("*.md")):
        if path.name == INDEX:
            continue
        if not ADR_FILE.fullmatch(path.name):
            problems.append(f"docs/adr/{path.name}: not an ADR file name (NNNN-slug.md, "
                            "the slug in lowercase letters, digits and hyphens)")
            continue
        adr = parse(path, problems)
        if adr.number in adrs:
            problems.append(f"docs/adr/{path.name}: ADR {adr.number} is also "
                            f"{adrs[adr.number].name}")
            continue
        adrs[adr.number] = adr

    for adr in adrs.values():
        def problem(message):
            problems.append(f"docs/adr/{adr.name}: {message}")

        targets = [(key, number) for key in LIST_KEYS for number in adr.lists[key]]
        if adr.superseded_by:
            targets.append(("status", adr.superseded_by[4:]))
        for key, number in targets:
            if number == adr.number:
                problem(f"{key} names the ADR itself")
            elif number not in adrs:
                problem(f"{key} names {ref(number)}, which does not exist")

        # A relation that disagrees is reported once, on the file whose side
        # of it is missing; a file with no frontmatter has its own problem.
        for number in adr.lists["amends"]:
            other = adrs.get(number)
            if other and other.has_frontmatter and adr.number not in other.lists["amended-by"]:
                problems.append(f"docs/adr/{other.name}: {ref(adr.number)} amends it, but its "
                                f"amended-by does not list {ref(adr.number)}")
        for number in adr.lists["amended-by"]:
            other = adrs.get(number)
            if other and other.has_frontmatter and adr.number not in other.lists["amends"]:
                problems.append(f"docs/adr/{other.name}: {ref(adr.number)} lists it under "
                                f"amended-by, but its amends does not list {ref(adr.number)}")
        for number in adr.lists["supersedes"]:
            other = adrs.get(number)
            if other and other.has_frontmatter and other.superseded_by != ref(adr.number):
                problems.append(f"docs/adr/{other.name}: {ref(adr.number)} supersedes it, but "
                                f"its status is not 'superseded by {ref(adr.number)}'")
        if adr.superseded_by:
            other = adrs.get(adr.superseded_by[4:])
            if other and other.has_frontmatter and adr.number not in other.lists["supersedes"]:
                problems.append(f"docs/adr/{other.name}: {ref(adr.number)} is superseded by it, "
                                f"but its supersedes does not list {ref(adr.number)}")
    return adrs, problems


def decision_text(text):
    """An ADR's text as mk/changelog.sh's adr_text compares it: the frontmatter
    and the blank lines after it dropped, every link target and every path in
    backticks blanked, and the trailing newlines gone, as `$(...)` drops them."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = 0
    if lines and lines[0] == "---" and "---" in lines[1:]:
        start = lines.index("---", 1) + 1
        while start < len(lines) and lines[start] == "":
            start += 1
    body = "\n".join(lines[start:])
    body = re.sub(r"\]\([^)\n]*\)", "]()", body)
    return re.sub(r"`[^` \n]*/[^` \n]*`", "``", body).rstrip("\n")


def marked(message):
    """The ADRs a commit message names as typo fixes, as ADR-NNNN."""
    return {item for match in TYPO.finditer(message) for item in match.group(1).split(", ")}


def git(root, *args, data=None):
    """git in `root`; the completed process, its output as bytes."""
    return subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True)


def frozen(adr_dir, adrs, since):
    """--since: the problems and the notes of holding the ADRs that existed at the
    merge base of `since` and HEAD to their text there (ADR 0081)."""
    root = adr_dir.parent.parent
    if shutil.which("git") is None or git(root, "rev-parse", "--git-dir").returncode:
        return [], ["adr-index: not a git checkout, so no earlier ADR's text was compared"]
    if since == "auto":
        if git(root, "rev-parse", "--verify", "--quiet", "origin/main^{commit}").returncode:
            return [], ["adr-index: no origin/main here, so no earlier ADR's text was compared"]
        since = "origin/main"
    found = git(root, "merge-base", since, "HEAD")
    if found.returncode:
        return [f"--since {since}: no merge base with HEAD "
                f"({found.stderr.decode('utf-8', 'replace').strip() or 'not a commit'})"], []
    base = found.stdout.decode().strip()
    folder = adr_dir.relative_to(root).as_posix()
    if git(root, "diff", "--quiet", base, "--", folder).returncode == 0:
        return [], []

    paths = [path for path in git(root, "ls-tree", "--name-only", base, f"{folder}/")
             .stdout.decode("utf-8").split("\n") if ADR_FILE.fullmatch(path.rsplit("/", 1)[-1])]
    # One git process for every blob at the base: `cat-file --batch` answers
    # each "<rev>:<path>" line with "<sha> blob <size>", the bytes and a newline.
    out = git(root, "cat-file", "--batch",
              data="".join(f"{base}:{path}\n" for path in paths).encode("utf-8")).stdout
    texts, at = {}, 0
    for path in paths:
        end = out.index(b"\n", at)
        header, at = out[at:end].split(), end + 1
        if len(header) == 3 and header[1] == b"blob":
            size = int(header[2])
            texts[path] = out[at:at + size].decode("utf-8", "replace")
            at += size + 1

    def typo_fix(number):
        """Whether every commit since the base that changes ADR `number`'s text
        names it as a typo fix: a commit that changes its frontmatter alone needs
        no mark, one marked commit does not cover another's rewrite, and an edit
        not committed yet is no fix."""
        pathspec = f":(glob){folder}/{number}-*.md"
        if git(root, "diff", "--quiet", "HEAD", "--", pathspec).returncode:
            return False
        log = git(root, "log", "--format=%x1e%H%x1f%B%x1f", "--name-only",
                  f"{base}..HEAD", "--", pathspec).stdout.decode("utf-8", "replace")
        def text_at(rev, name):
            blob = git(root, "show", f"{rev}:{name}").stdout
            return decision_text(blob.decode("utf-8", "replace"))

        fixed = False
        for entry in log.split("\x1e")[1:]:
            sha, message, names = entry.split("\x1f", 2)
            if all(text_at(f"{sha}^", name) == text_at(sha, name) for name in names.split()):
                continue
            if ref(number) not in marked(message):
                return False
            fixed = True
        return fixed

    problems, notes = [], []
    for path, was in texts.items():
        number = path.rsplit("/", 1)[-1][:4]
        front = was.replace("\r\n", "\n").split("\n---", 1)[0] if was.startswith("---") else ""
        status = FRONT_STATUS.search(front)
        if status and status.group(1) == "proposed":
            continue
        adr = adrs.get(number)
        if adr is None:
            problems.append(f"{path}: ADR {number} existed at {base[:8]} and is gone: an ADR "
                            "is never deleted, a later one supersedes it (ADR 0081)")
        elif decision_text(was) != decision_text(adr.path.read_text(encoding="utf-8")):
            if typo_fix(number):
                notes.append(f"adr-index: ADR {number}'s text changed since {base[:8]}, "
                             "as a typo fix its commit names")
            else:
                problems.append(
                    f"docs/adr/{adr.name}: its text changed since {base[:8]}, but an ADR "
                    "changes only in its frontmatter: a decision that changes is a new ADR "
                    "that amends or supersedes it (ADR 0081). A typo fix names it in its "
                    f"commit message, [typo ADR-{number}]")
    return problems, notes


def render(adrs):
    def link(number):
        other = adrs.get(number)
        return f"[{number}]({other.name})" if other else ref(number)

    rows = []
    for number in sorted(adrs):
        adr = adrs[number]
        if adr.superseded_by:
            status = f"superseded by {link(adr.superseded_by[4:])}"
        else:
            status = adr.status or "(none)"
        title = adr.title.replace("|", "\\|")
        amended = ", ".join(link(n) for n in adr.lists["amended-by"])
        cells = [f"[{number}]({adr.name})", title, status] + ([amended] if amended else [])
        rows.append("| " + " | ".join(cells) + (" |" if amended else " | |"))
    return HEADER + "\n".join(rows) + "\n"


def run(adr_dir, mode, out=sys.stdout, err=sys.stderr, since=None):
    """--write or --check against one docs/adr directory; returns the exit status."""
    adrs, problems = load(adr_dir)
    index = adr_dir / INDEX
    expected = render(adrs)
    if mode == "write":
        with open(index, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(expected)
        print(f"wrote docs/adr/{INDEX} ({len(adrs)} ADRs)", file=out)
        if problems:
            print(f"{len(problems)} problems the check refuses:", file=err)
            for line in problems:
                print(f"  {line}", file=err)
        return 0
    if not index.exists():
        problems.append(f"docs/adr/{INDEX} is missing: run make records")
    elif index.read_text(encoding="utf-8") != expected:
        problems.append(f"docs/adr/{INDEX} is not what the frontmatter generates: "
                        "run make records (never edit it by hand)")
    if since:
        more, notes = frozen(adr_dir, adrs, since)
        problems += more
        for line in notes:
            print(line, file=out)
    for line in problems:
        print(line, file=err)
    if problems:
        print(f"adr-index: {len(problems)} problems", file=err)
        return 1
    print(f"adr-index: {len(adrs)} ADRs, index in sync", file=out)
    return 0


# --- selftest ---------------------------------------------------------------

GOOD = {
    "0001-the-first.md": "---\nstatus: superseded by ADR-0002\namended-by: [ADR-0070]\n---\n\n"
                         "# ADR 0001 — the first decision\n\n## Status\n\nAccepted.\n",
    "0002-the-second.md": "---\nstatus: accepted\nsupersedes: [ADR-0001]\n---\n\n"
                          "# ADR 0002 — the second | with a pipe\n",
    "0070-a-short-one.md": "---\nstatus: proposed\namends: [ADR-0001]\n---\n\n"
                           "# A short one\n\nIts decision and why. See "
                           "[ADR 0002](0002-the-second.md).\n",
}

# Each case: what it does to a copy of GOOD, and a fragment the check must print.
CASES = [
    ("no frontmatter", {"0002-the-second.md": "# ADR 0002 — the second\n"},
     "no frontmatter"),
    ("an unclosed frontmatter", {"0002-the-second.md": "---\nstatus: accepted\n\n# x\n"},
     "never closed"),
    ("a status outside the vocabulary",
     {"0070-a-short-one.md": "---\nstatus: Accepted\namends: [ADR-0001]\n---\n\n# A short one\n"},
     "status 'Accepted' is not one of"),
    ("an unknown key",
     {"0002-the-second.md": "---\nstatus: accepted\nsupersedes: [ADR-0001]\nsuperseded-by: "
                            "[ADR-0070]\n---\n\n# ADR 0002 — the second\n"},
     "unknown key 'superseded-by'"),
    ("a YAML comment",
     {"0002-the-second.md": "---\n# why\nstatus: accepted\nsupersedes: [ADR-0001]\n---\n\n"
                            "# ADR 0002 — the second\n"},
     "is not a 'key: value' line"),
    ("a list out of order",
     {"0001-the-first.md": "---\nstatus: superseded by ADR-0002\namended-by: [ADR-0070, "
                           "ADR-0002]\n---\n\n# ADR 0001 — the first\n"},
     "not in ascending order"),
    ("an empty list",
     {"0002-the-second.md": "---\nstatus: accepted\namends: []\nsupersedes: [ADR-0001]\n"
                            "---\n\n# ADR 0002 — the second\n"},
     "is not a list like"),
    ("a reference spelled without ADR-",
     {"0002-the-second.md": "---\nstatus: accepted\nsupersedes: [0001]\n---\n\n"
                            "# ADR 0002 — the second\n"},
     "is not a list like"),
    ("keys out of order",
     {"0070-a-short-one.md": "---\namends: [ADR-0001]\nstatus: proposed\n---\n\n# A short one\n"},
     "out of order"),
    ("no status",
     {"0070-a-short-one.md": "---\namends: [ADR-0001]\n---\n\n# A short one\n"},
     "has no status"),
    ("no blank line before the heading",
     {"0002-the-second.md": "---\nstatus: accepted\nsupersedes: [ADR-0001]\n---\n"
                            "# ADR 0002 — the second\n"},
     "one blank line and the '# ' heading"),
    ("a reference to an ADR that does not exist",
     {"0070-a-short-one.md": "---\nstatus: proposed\namends: [ADR-0001, ADR-0009]\n---\n\n"
                             "# A short one\n"},
     "names ADR-0009, which does not exist"),
    ("a superseder that does not exist",
     {"0002-the-second.md": "---\nstatus: superseded by ADR-0082\nsupersedes: [ADR-0001]\n"
                            "---\n\n# ADR 0002 — the second\n"},
     "status names ADR-0082, which does not exist"),
    ("the ADR naming itself",
     {"0002-the-second.md": "---\nstatus: accepted\namends: [ADR-0002]\nsupersedes: [ADR-0001]"
                            "\n---\n\n# ADR 0002 — the second\n"},
     "names the ADR itself"),
    ("amends with no amended-by on the other side",
     {"0001-the-first.md": "---\nstatus: superseded by ADR-0002\n---\n\n# ADR 0001 — the first\n"},
     "0001-the-first.md: ADR-0070 amends it, but"),
    ("amended-by with no amends on the other side",
     {"0070-a-short-one.md": "---\nstatus: proposed\n---\n\n# A short one\n"},
     "0070-a-short-one.md: ADR-0001 lists it under amended-by"),
    ("supersedes with no superseded by on the other side",
     {"0001-the-first.md": "---\nstatus: accepted\namended-by: [ADR-0070]\n---\n\n"
                           "# ADR 0001 — the first\n"},
     "0001-the-first.md: ADR-0002 supersedes it, but"),
    ("superseded by with no supersedes on the other side",
     {"0002-the-second.md": "---\nstatus: accepted\n---\n\n# ADR 0002 — the second\n"},
     "0002-the-second.md: ADR-0001 is superseded by it"),
    ("a short-format heading carrying its number",
     {"0070-a-short-one.md": "---\nstatus: proposed\namends: [ADR-0001]\n---\n\n"
                             "# ADR 0070 — a short one\n"},
     "the heading is the bare title"),
    ("two files with one number",
     {"0002-another.md": "---\nstatus: accepted\n---\n\n# ADR 0002 — another\n"},
     "ADR 0002 is also"),
    ("a file name that is not an ADR's",
     {"81-misnamed.md": "---\nstatus: accepted\n---\n\n# Misnamed\n"},
     "not an ADR file name"),
]

# --since, in a git repo whose first commit is GOOD and an accepted ADR with a
# link and a path: what the second commit changes (None deletes the file), its
# message (None leaves the change uncommitted), whether the check passes, and a
# fragment it must print. A list of changes and a list of messages are commits
# made in turn.
THIRD = "---\nstatus: accepted\n---\n\n# ADR 0003 — the third\n\nSee [the finding]({}) and `{}`.\n"
SINCE_BASE = {**GOOD, "0003-the-third.md": THIRD.format("../findings/a.md", "docs/a.md")}
CHANGED = THIRD.format("../findings/a.md", "docs/a.md").replace("See", "Now see")
SINCE_CASES = [
    ("a new ADR that amends an earlier one, whose frontmatter alone changes",
     {"0003-the-third.md": THIRD.format("../findings/a.md", "docs/a.md").replace(
         "accepted\n", "deprecated\namended-by: [ADR-0071]\n"),
      "0071-a-new-one.md": "---\nstatus: accepted\namends: [ADR-0003]\n---\n\n# A new one\n"},
     "The third, amended by a new decision (#1)", True, "index in sync"),
    ("a link's target and a path in backticks that moved",
     {"0003-the-third.md": THIRD.format("../findings/archive/a.md", "docs/archive/a.md")},
     "The finding, archived (#1)", True, "index in sync"),
    ("an earlier ADR's text changed", {"0003-the-third.md": CHANGED},
     "The third, rewritten (#1)", False, "its text changed since"),
    ("an edit not committed yet", {"0003-the-third.md": CHANGED},
     None, False, "its text changed since"),
    ("a typo fix its commit names", {"0003-the-third.md": CHANGED},
     "A typo in the third, fixed [typo ADR-0003] (#1)", True, "as a typo fix its commit names"),
    ("a typo fix named in a list", {"0003-the-third.md": CHANGED},
     "Two typos, fixed [typo ADR-0002, ADR-0003] (#1)", True, "as a typo fix its commit names"),
    ("a typo fix among two marks of one message", {"0003-the-third.md": CHANGED},
     "Typos, fixed [typo ADR-0002] [typo ADR-0003] (#1)", True, "as a typo fix its commit names"),
    ("a typo mark that names another ADR", {"0003-the-third.md": CHANGED},
     "A typo, fixed [typo ADR-0002] (#1)", False, "its text changed since"),
    ("a typo mark on one commit and a rewrite in the next",
     [{"0003-the-third.md": CHANGED}, {"0003-the-third.md": CHANGED.replace("Now", "Then")}],
     ["A typo, fixed [typo ADR-0003] (#1)", "The third, rewritten (#1)"], False,
     "its text changed since"),
    ("a typo fix with an edit on top not committed yet",
     [{"0003-the-third.md": CHANGED}, {"0003-the-third.md": CHANGED.replace("Now", "Then")}],
     ["A typo, fixed [typo ADR-0003] (#1)", None], False, "its text changed since"),
    ("a typo fix beside a commit that changes the frontmatter alone",
     [{"0003-the-third.md": THIRD.format("../findings/a.md", "docs/a.md").replace(
         "accepted\n", "deprecated\namended-by: [ADR-0071]\n"),
       "0071-a-new-one.md": "---\nstatus: accepted\namends: [ADR-0003]\n---\n\n# A new one\n"},
      {"0003-the-third.md": CHANGED.replace("accepted\n", "deprecated\namended-by: [ADR-0071]\n")}],
     ["The third, amended by a new decision (#1)",
      "A typo in the third, fixed [typo ADR-0003] (#1)"],
     True, "as a typo fix its commit names"),
    ("a blank line at the end", {"0003-the-third.md": SINCE_BASE["0003-the-third.md"] + "\n"},
     "A blank line (#1)", True, "index in sync"),
    ("a proposed ADR's text, still being written",
     {"0070-a-short-one.md": GOOD["0070-a-short-one.md"] + "\nSharpened.\n"},
     "The proposal, sharpened (#1)", True, "index in sync"),
    ("an earlier ADR deleted", {"0003-the-third.md": None},
     "The third, gone (#1)", False, "is gone"),
    ("auto, in a clone with no origin/main", {"0003-the-third.md": CHANGED},
     "The third, rewritten (#1)", True, "no origin/main here"),
]


def selftest():
    failed = 0

    def check(ok, what):
        nonlocal failed
        if ok:
            print(f"  {what}")
        else:
            failed += 1
            print(f"  FAIL: {what}", file=sys.stderr)

    class Sink:
        def __init__(self):
            self.text = ""

        def write(self, text):
            self.text += text

    def fixture(root, files):
        adr_dir = pathlib.Path(root) / "docs" / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (adr_dir / name).write_text(text, encoding="utf-8", newline="\n")
        return adr_dir

    print("adr-index selftest")
    with tempfile.TemporaryDirectory() as root:
        adr_dir = fixture(root, GOOD)
        out, err = Sink(), Sink()
        check(run(adr_dir, "check", out, err) == 1 and "is missing" in err.text,
              "a missing index fails the check")
        check(run(adr_dir, "write", Sink(), Sink()) == 0, "--write writes the index")
        index = (adr_dir / INDEX).read_text(encoding="utf-8")
        out, err = Sink(), Sink()
        check(run(adr_dir, "check", out, err) == 0,
              "a consistent set and its index pass the check"
              + (f" -- {err.text.strip()}" if err.text else ""))
        check("| [0001](0001-the-first.md) | the first decision | superseded by "
              "[0002](0002-the-second.md) | [0070](0070-a-short-one.md) |" in index,
              "a row links the ADR, drops the number from a long-form heading, links the "
              "superseder and the amending ADR")
        check("| the second \\| with a pipe |" in index, "a pipe in a title is escaped")
        check("| [0070](0070-a-short-one.md) | A short one | proposed | |\n" in index,
              "a row with no amended-by leaves its last cell empty")
        run(adr_dir, "write", Sink(), Sink())
        check((adr_dir / INDEX).read_text(encoding="utf-8") == index, "--write is idempotent")
        (adr_dir / INDEX).write_text(index.replace("proposed", "accepted"), encoding="utf-8",
                                     newline="\n")
        out, err = Sink(), Sink()
        check(run(adr_dir, "check", out, err) == 1 and "never edit it by hand" in err.text,
              "a hand-edited index fails the check")
        crlf = (adr_dir / "0002-the-second.md").read_text(encoding="utf-8")
        (adr_dir / "0002-the-second.md").write_bytes(crlf.replace("\n", "\r\n").encode("utf-8"))
        (adr_dir / INDEX).write_bytes(index.replace("\n", "\r\n").encode("utf-8"))
        check(run(adr_dir, "check", Sink(), Sink()) == 0,
              "a CRLF checkout reads as its LF text")

    for what, change, expected in CASES:
        with tempfile.TemporaryDirectory() as root:
            adr_dir = fixture(root, GOOD)
            run(adr_dir, "write", Sink(), Sink())
            fixture(root, change)
            out, err = Sink(), Sink()
            status = run(adr_dir, "check", out, err)
            check(status == 1 and expected in err.text,
                  f"refused: {what}" + ("" if status == 1 and expected in err.text
                                        else f" -- got {status}: {err.text.strip()}"))
            out, err = Sink(), Sink()
            check(run(adr_dir, "write", out, err) == 0 and expected in err.text,
                  f"--write still writes, naming it: {what}")

    if shutil.which("git") is None:
        print("  (no git on PATH: the --since cases are skipped)")
    else:
        for what, change, message, passes, expected in SINCE_CASES:
            with tempfile.TemporaryDirectory() as root:
                adr_dir = fixture(root, SINCE_BASE)
                run(adr_dir, "write", Sink(), Sink())
                hooks = pathlib.Path(root) / "no-hooks"
                hooks.mkdir()
                config = ["-c", "user.name=selftest", "-c", "user.email=selftest@invalid",
                          "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
                          "-c", f"core.hooksPath={hooks.as_posix()}"]
                git(root, "init", "-q")
                git(root, *config, "add", "-A")
                git(root, *config, "commit", "-q", "-m", "The base")
                steps = (list(zip(change, message)) if isinstance(change, list)
                         else [(change, message)])
                for step, step_message in steps:
                    for name, text in step.items():
                        if text is None:
                            (adr_dir / name).unlink()
                        else:
                            (adr_dir / name).write_text(text, encoding="utf-8", newline="\n")
                    run(adr_dir, "write", Sink(), Sink())
                    if step_message:
                        git(root, *config, "add", "-A")
                        git(root, *config, "commit", "-q", "-m", step_message)
                commits = sum(1 for _, step_message in steps if step_message)
                since = ("auto" if what.startswith("auto")
                         else f"HEAD~{commits}" if commits else "HEAD")
                out, err = Sink(), Sink()
                status = run(adr_dir, "check", out, err, since=since)
                said = out.text + err.text
                ok = (status == 0) == passes and expected in said
                check(ok, f"{'passes' if passes else 'refused'}: {what}"
                      + ("" if ok else f" -- got {status}: {said.strip()}"))

    print()
    if failed:
        print(f"adr-index selftest: {failed} failed", file=sys.stderr)
        return 1
    print("adr-index selftest: ok")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_const", const="write", dest="mode")
    mode.add_argument("--check", action="store_const", const="check", dest="mode")
    mode.add_argument("--selftest", action="store_const", const="selftest", dest="mode")
    parser.add_argument("--since", metavar="REV",
                        help="with --check: refuse a change to the text of an ADR that existed "
                             "at the merge base of REV and HEAD ('auto': origin/main)")
    args = parser.parse_args(argv)
    if args.since and args.mode != "check":
        parser.error("--since goes with --check")
    if args.mode == "selftest":
        return selftest()
    adr_dir = pathlib.Path(__file__).resolve().parent.parent / "docs" / "adr"
    return run(adr_dir, args.mode, since=args.since)


if __name__ == "__main__":
    sys.exit(main())
