#!/usr/bin/env python3
"""The project board's Status from an issue's labels: the board-status workflow's step.

    python mk/board-status.py --repo gpillon/ignis --issue 524             set the issue's Status from its labels
    python mk/board-status.py --repo gpillon/ignis --issue 524 --dry-run   say what it would set, and set nothing
    python mk/board-status.py --selftest                                   the tool's own test, offline

.github/workflows/board-status.yml runs it on every issue opened, reopened,
labeled or unlabeled (GitHub #524), with a token that can write the board in
GH_TOKEN (`python` on Windows, `python3` on Linux; the standard library only).
The board is BOARD_OWNER's project number BOARD_NUMBER, a user's or an
organization's, which the run tells apart itself; the Status field and its
options are found by name, so a board rebuilt under another owner needs only
the two values.

The mapping is the table in docs/agents/issue-tracker.md, Project board, which
stays the authority: STATUS_OF below is a copy of it, and the selftest refuses
the two apart. The run reads the issue's labels, state and board item when it
starts, not from the event, so runs that land out of order still leave the
Status the issue's last labels name.
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# The table of docs/agents/issue-tracker.md, Project board: a label and the
# Status it gives. `idea` wins over any state label beside it. A label added
# or a Status renamed changes the table and this copy in the same change.
STATUS_OF = {
    "idea": "Parked",
    "needs-triage": "Inbox",
    "needs-info": "Inbox",
    "ready-for-agent": "Ready",
    "ready-for-human": "Ready",
}
WINS = "idea"
# The Statuses the coordinating session sets, which a run never moves an issue
# out of, and the one the board gives a closed issue.
KEPT = ("In progress", "Waiting on owner")
DONE = "Done"
STATUS_FIELD = "Status"

# How long a run waits for the board's own workflows to finish with the issue
# (unsettled below), how often it asks again, and how young an issue is that
# the board may not have added yet.
SETTLE_SECONDS = 60
POLL_SECONDS = 5
FRESH = datetime.timedelta(minutes=5)

REPO = Path(__file__).resolve().parents[1]
TRACKER = REPO / "docs" / "agents" / "issue-tracker.md"
GRAPHQL_URL = os.environ.get("GITHUB_GRAPHQL_URL") or "https://api.github.com/graphql"

BOARD_QUERY = """
query($owner: String!, $number: Int!, $field: String!) {
  repositoryOwner(login: $owner) {
    __typename
    ... on User { projectV2(number: $number) { ...board } }
    ... on Organization { projectV2(number: $number) { ...board } }
  }
}
fragment board on ProjectV2 {
  id
  title
  field(name: $field) {
    ... on ProjectV2SingleSelectField { id options { id name } }
  }
}
"""

ISSUE_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $field: String!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      state
      createdAt
      labels(first: 100) { nodes { name } }
      projectItems(first: 100) {
        nodes {
          id
          isArchived
          project { id }
          fieldValueByName(name: $field) {
            ... on ProjectV2ItemFieldSingleSelectValue { name }
          }
        }
      }
    }
  }
}
"""

SET_STATUS = """
mutation($project: ID!, $item: ID!, $field: ID!, $option: String!) {
  updateProjectV2ItemFieldValue(input: {projectId: $project, itemId: $item, fieldId: $field, value: {singleSelectOptionId: $option}}) {
    projectV2Item { id }
  }
}
"""


class Failure(Exception):
    """A run that cannot do its job: the board, its Status field or an option it needs is missing, or GitHub refused."""


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def read(path):
    # utf-8-sig drops a byte-order mark; the newline swap reads a CRLF checkout the same.
    with open(path, encoding="utf-8-sig") as f:
        return f.read().replace("\r\n", "\n")


def graphql(token):
    """A function that sends one GraphQL request with `token` and returns its data."""
    def send(query, variables):
        body = json.dumps({"query": query, "variables": variables}).encode()
        request = urllib.request.Request(GRAPHQL_URL, data=body, headers={
            "Authorization": "bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "ignis-board-status",
        })
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as e:
            raise Failure("GitHub's GraphQL API answered %d: %s" % (e.code, " ".join(e.read()[:300].decode(errors="replace").split())))
        except urllib.error.URLError as e:
            raise Failure("GitHub's GraphQL API is out of reach: %s" % e.reason)
        if answer.get("errors"):
            raise Failure("GitHub's GraphQL API: " + "; ".join(e.get("message", "?") for e in answer["errors"]))
        return answer["data"]
    return send


def read_only(api):
    """`api` with every mutation refused: what a dry run talks to."""
    def send(query, variables):
        if query.lstrip().startswith("mutation"):
            raise Failure("a dry run writes nothing")
        return api(query, variables)
    return send


def decide(state, labels, item):
    """(the Status to set, the labels that name it, False), or (None, why the
    run leaves the issue alone, whether that is a warning).

    `state` is the issue's GraphQL state, `labels` its label names, `item`
    None when the issue is not on the board, else {"status", "archived"} with
    status None when the item has none."""
    if state != "OPEN":
        return None, "closed: the board moves it to Done itself", False
    if item is None:
        return None, "not on the board", False
    if item["archived"]:
        return None, "archived on the board", False
    status = item["status"]
    if status in KEPT:
        return None, "%s is the coordinating session's to move" % status, False
    named = [WINS] if WINS in labels else sorted(l for l in labels if l in STATUS_OF)
    wants = sorted({STATUS_OF[l] for l in named})
    if not wants:
        return None, "no label names a Status", False
    if len(wants) > 1:
        return None, "its state labels disagree: " + ", ".join("`%s` %s" % (l, STATUS_OF[l]) for l in named), True
    why = ", ".join("`%s`" % l for l in named)
    if wants[0] == status:
        return None, "already %s, from %s" % (status, why), False
    return wants[0], why, False


def unsettled(issue, now):
    """Why the board's own workflows may still be about to write the issue's
    Status, or None. The board adds a new issue by itself and gives it Inbox
    ("Auto-add to project", "Item added to project"), and moves a reopened one
    out of Done ("Item reopened"), a moment after the event that starts the
    run: a Status set before them would be overwritten."""
    if issue["state"] != "OPEN":
        return None
    item, fresh = issue["item"], now - issue["created"] < FRESH
    if item is None and fresh:
        return "the board has not added it yet"
    if item is not None and item["status"] is None and fresh:
        return "the board has not given it its first Status yet"
    if item is not None and item["status"] == DONE:
        return "the board has not moved it out of Done yet"
    return None


def find_board(api, owner, number):
    """The board's id, its Status field's id and that field's options by name.
    Every Status this script names must be an option: a board whose In
    progress were spelled otherwise would see its tickets moved out of it."""
    data = api(BOARD_QUERY, {"owner": owner, "number": number, "field": STATUS_FIELD})
    holder = data.get("repositoryOwner")
    if not holder:
        raise Failure("no user or organization %s" % owner)
    board = holder.get("projectV2")
    if not board:
        raise Failure("%s %s has no project %d the token can see" % (holder["__typename"].lower(), owner, number))
    field = board.get("field") or {}
    if not field.get("id"):
        raise Failure("project %d of %s has no single-select field %s" % (number, owner, STATUS_FIELD))
    options = {o["name"]: o["id"] for o in field["options"]}
    missing = [s for s in sorted(set(STATUS_OF.values())) + list(KEPT) + [DONE] if s not in options]
    if missing:
        raise Failure("the %s field of project %d of %s has no option %s (it has %s)" % (
            STATUS_FIELD, number, owner, ", ".join(missing), ", ".join(options)))
    return {"id": board["id"], "field": field["id"], "options": options}


def read_issue(api, repo, number, board_id):
    owner, name = repo.split("/", 1)
    data = api(ISSUE_QUERY, {"owner": owner, "name": name, "number": number, "field": STATUS_FIELD})
    issue = (data.get("repository") or {}).get("issue")
    if not issue:
        raise Failure("no issue %s#%d" % (repo, number))
    item = None
    for node in issue["projectItems"]["nodes"]:
        if node["project"]["id"] == board_id:
            value = node.get("fieldValueByName") or {}
            item = {"id": node["id"], "archived": node["isArchived"], "status": value.get("name")}
    created = datetime.datetime.strptime(issue["createdAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    return {"state": issue["state"], "created": created, "labels": [l["name"] for l in issue["labels"]["nodes"]], "item": item}


def run(api, repo, number, owner, board_number, dry_run=False, out=sys.stdout, sleep=time.sleep, now=utcnow):
    board = find_board(api, owner, board_number)
    waited = 0
    while True:
        issue = read_issue(api, repo, number, board["id"])
        wait = unsettled(issue, now())
        if not wait or waited >= SETTLE_SECONDS:
            break
        print("#%d: %s; asking again in %d s" % (number, wait, POLL_SECONDS), file=out)
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS
    status, why, warn = decide(issue["state"], issue["labels"], issue["item"])
    if status is None:
        print("%s#%d left alone: %s" % ("::warning::" if warn else "", number, why), file=out)
        return 0
    was = issue["item"]["status"] or "no Status"
    if dry_run:
        print("#%d: would move %s -> %s, from %s (dry run)" % (number, was, status, why), file=out)
        return 0
    api(SET_STATUS, {"project": board["id"], "item": issue["item"]["id"], "field": board["field"], "option": board["options"][status]})
    print("#%d: %s -> %s, from %s" % (number, was, status, why), file=out)
    return 0


def doc_mapping(text):
    """The label -> Status table under issue-tracker.md's `## Project board`:
    its rows are `| `label`, `label` | Status |`."""
    parts = text.split("\n## Project board\n", 1)
    section = parts[1].split("\n## ", 1)[0] if len(parts) == 2 else ""
    mapping = {}
    for line in section.split("\n"):
        m = re.match(r"^\| *((?:`[a-z-]+`(?:, *)?)+) *\| *([A-Z][A-Za-z ]*?) *\|$", line)
        if m:
            for label in re.findall(r"`([a-z-]+)`", m.group(1)):
                mapping[label] = m.group(2)
    return mapping


# ---- selftest ----

NOW = datetime.datetime(2026, 11, 10, 12, 0, 0, tzinfo=datetime.timezone.utc)
OPTIONS = ["Inbox", "Parked", "Ready", "In progress", "Waiting on owner", "Done"]


class Sink:
    def __init__(self):
        self.lines = []

    def write(self, s):
        self.lines += [l for l in s.split("\n") if l]

    def flush(self):
        pass


def snapshot(state="OPEN", labels=(), status="Inbox", on_board=True, archived=False, age=3600):
    """The ISSUE_QUERY answer for one reading of an issue, `age` seconds old."""
    nodes = [{"id": "ITEM_other", "isArchived": False, "project": {"id": "PROJECT_other"}, "fieldValueByName": {"name": "Ready"}}]
    if on_board:
        nodes.append({"id": "ITEM_1", "isArchived": archived, "project": {"id": "PROJECT_1"},
                      "fieldValueByName": {"name": status} if status else None})
    created = (NOW - datetime.timedelta(seconds=age)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"repository": {"issue": {"state": state, "createdAt": created, "labels": {"nodes": [{"name": l} for l in labels]},
                                     "projectItems": {"nodes": nodes}}}}


class FakeGitHub:
    """GraphQL as the board and the issue answer it: each issue reading takes
    the next snapshot, and the last one stays."""

    def __init__(self, snapshots, owner_type="User", options=OPTIONS, board=True):
        self.snapshots, self.sets, self.readings = list(snapshots), [], 0
        field = {"id": "FIELD_status", "options": [{"id": "OPT_" + o.replace(" ", "_"), "name": o} for o in options]}
        project = {"id": "PROJECT_1", "title": "ignis", "field": field} if board else None
        self.owner = {"__typename": owner_type, "projectV2": project}

    def __call__(self, query, variables):
        if query is BOARD_QUERY:
            return {"repositoryOwner": self.owner}
        if query is ISSUE_QUERY:
            self.readings += 1
            return self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]
        if query is SET_STATUS:
            self.sets.append(variables["option"])
            return {"updateProjectV2ItemFieldValue": {"projectV2Item": {"id": variables["item"]}}}
        raise AssertionError("unexpected query")


def on(status="Inbox", archived=False):
    return {"status": status, "archived": archived}


# decide(): (state, labels, item) and the Status it sets, None for left alone.
DECISIONS = [
    ("OPEN", ["needs-triage"], on("Ready"), "Inbox", "needs-triage gives Inbox"),
    ("OPEN", ["needs-info", "bug"], on("Ready"), "Inbox", "needs-info gives Inbox, and a category changes nothing"),
    ("OPEN", ["ready-for-agent"], on("Inbox"), "Ready", "ready-for-agent gives Ready"),
    ("OPEN", ["ready-for-human", "quick"], on("Parked"), "Ready", "ready-for-human gives Ready"),
    ("OPEN", ["idea"], on("Inbox"), "Parked", "idea gives Parked"),
    ("OPEN", ["idea", "ready-for-human"], on("Ready"), "Parked", "idea wins over the state label beside it"),
    ("OPEN", ["idea", "needs-triage", "ready-for-agent"], on("Inbox"), "Parked", "idea wins over state labels that disagree"),
    ("OPEN", ["needs-triage", "needs-info"], on("Ready"), "Inbox", "two state labels that agree"),
    ("OPEN", ["ready-for-agent"], on(None), "Ready", "an item with no Status gets one"),
    ("OPEN", ["ready-for-agent"], on("Done"), "Ready", "an open issue still in Done follows its label"),
    ("OPEN", ["needs-triage", "ready-for-agent"], on("Inbox"), None, "state labels that disagree leave it alone"),
    ("OPEN", ["ready-for-agent"], on("Ready"), None, "already at its Status: nothing is written"),
    ("OPEN", ["ready-for-agent"], on("In progress"), None, "In progress is left alone"),
    ("OPEN", ["idea"], on("Waiting on owner"), None, "Waiting on owner is left alone, idea or not"),
    ("OPEN", ["enhancement"], on("Ready"), None, "no state label leaves it alone"),
    ("OPEN", ["wontfix"], on("Ready"), None, "wontfix names no Status"),
    ("OPEN", [], on("Ready"), None, "no label at all leaves it alone"),
    ("CLOSED", ["needs-triage"], on("Done"), None, "a closed issue is left alone"),
    ("OPEN", ["needs-triage"], None, None, "an issue not on the board is left alone"),
    ("OPEN", ["needs-triage"], on("Ready", archived=True), None, "an archived item is left alone"),
]


def flows():
    """(what, FakeGitHub, run options, the options set, the seconds slept, a Failure's phrase or None)."""
    return [
        ("a label on the user's board sets its option", FakeGitHub([snapshot(labels=["ready-for-agent"])]), {}, ["OPT_Ready"], 0, None),
        ("an organization's board is found the same way",
         FakeGitHub([snapshot(labels=["needs-info"], status="Ready")], owner_type="Organization"), {}, ["OPT_Inbox"], 0, None),
        ("a dry run writes nothing", FakeGitHub([snapshot(labels=["ready-for-agent"])]), {"dry_run": True}, [], 0, None),
        ("a closed issue: nothing written", FakeGitHub([snapshot(state="CLOSED", labels=["ready-for-agent"], status="Done")]), {}, [], 0, None),
        ("In progress: nothing written", FakeGitHub([snapshot(labels=["ready-for-agent"], status="In progress")]), {}, [], 0, None),
        ("an old issue off the board: no wait, nothing written", FakeGitHub([snapshot(labels=["idea"], on_board=False)]), {}, [], 0, None),
        ("a new issue waits for the board to add it, then gets its label's Status",
         FakeGitHub([snapshot(labels=["ready-for-agent"], on_board=False, age=10), snapshot(labels=["ready-for-agent"], status=None, age=15),
                     snapshot(labels=["ready-for-agent"], status="Inbox", age=20)]), {}, ["OPT_Ready"], 2 * POLL_SECONDS, None),
        ("a new issue the board never adds: the wait ends, nothing written",
         FakeGitHub([snapshot(labels=["ready-for-agent"], on_board=False, age=10)]), {}, [], SETTLE_SECONDS, None),
        ("a reopened issue waits for the board to move it out of Done",
         FakeGitHub([snapshot(labels=["ready-for-human"], status="Done"), snapshot(labels=["ready-for-human"], status="Inbox")]),
         {}, ["OPT_Ready"], POLL_SECONDS, None),
        ("a board without the In progress option is refused, nothing written",
         FakeGitHub([snapshot(labels=["ready-for-agent"])], options=[o for o in OPTIONS if o != "In progress"] + ["In Progress"]),
         {}, [], 0, "no option In progress"),
        ("a board that is not there is refused", FakeGitHub([snapshot()], board=False), {}, [], 0, "has no project 6"),
    ]


def selftest():
    failed = []

    def check(ok, what):
        print("  %s  %s" % ("ok  " if ok else "FAIL", what))
        if not ok:
            failed.append(what)

    for state, labels, item, want, what in DECISIONS:
        got = decide(state, labels, item)[0]
        check(got == want, "%s (%s, wanted %s)" % (what, got, want) if got != want else what)
    check(decide("OPEN", ["needs-triage", "ready-for-agent"], on("Inbox"))[2], "disagreeing state labels are a warning")

    checks = 0
    for what, github, options, sets, slept, refused in flows():
        naps = []
        try:
            run(github, "gpillon/ignis", 524, "gpillon", 6, out=Sink(), sleep=naps.append, now=lambda: NOW, **options)
            failure = None
        except Failure as e:
            failure = str(e)
        ok = github.sets == sets and sum(naps) == slept and (failure is None if refused is None else refused in (failure or ""))
        check(ok, what if ok else "%s: set %s slept %d failure %r" % (what, github.sets, sum(naps), failure))
        checks += 1
    try:
        read_only(FakeGitHub([snapshot()]))(SET_STATUS, {"project": "p", "item": "i", "field": "f", "option": "o"})
        check(False, "a dry run's API refuses a mutation")
    except Failure:
        check(True, "a dry run's API refuses a mutation")

    # issue-tracker.md stays the authority: its table and STATUS_OF agree, and
    # the comparison sees a row that moved.
    table = doc_mapping(read(TRACKER))
    check(table == STATUS_OF, "issue-tracker.md's Project board table is STATUS_OF" if table == STATUS_OF
          else "issue-tracker.md's Project board table %s is not STATUS_OF %s" % (table, STATUS_OF))
    drifted = "# T\n\n## Project board\n\n| Label | Status |\n|---|---|\n| `idea` | Parked |\n| `needs-triage`, `needs-info` | Inbox |\n" \
              "| `ready-for-agent` | Inbox |\n| `ready-for-human` | Ready |\n\n## Auth\n\n| `wontfix` | Done |\n"
    check(doc_mapping(drifted) == dict(STATUS_OF, **{"ready-for-agent": "Inbox"}), "the table's reader sees a row that moved, and stops at the section's end")

    total = len(DECISIONS) + 1 + checks + 3
    if failed:
        print("selftest: %d failure(s) of %d checks" % (len(failed), total), file=sys.stderr)
        return 1
    print("selftest ok: %d checks" % total)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Set an issue's Status on the project board from its labels (docs/agents/issue-tracker.md).")
    parser.add_argument("--selftest", action="store_true", help="the tool's own test, offline")
    parser.add_argument("--repo", help="the issue's repository, OWNER/NAME")
    parser.add_argument("--issue", type=int, help="the issue's number")
    parser.add_argument("--dry-run", action="store_true", help="say what it would set, and set nothing")
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.repo or "/" not in args.repo or not args.issue:
        parser.error("--repo OWNER/NAME and --issue N name the issue")
    owner, number, token = os.environ.get("BOARD_OWNER"), os.environ.get("BOARD_NUMBER", ""), os.environ.get("GH_TOKEN")
    if not owner or not number.isdigit():
        parser.error("BOARD_OWNER and BOARD_NUMBER name the board, as the workflow sets them (gpillon and 6)")
    if not token:
        parser.error("GH_TOKEN holds a token that can read the board, and write it unless --dry-run")
    api = graphql(token)
    try:
        return run(read_only(api) if args.dry_run else api, args.repo, args.issue, owner, int(number), dry_run=args.dry_run)
    except Failure as e:
        # On stdout, where the runner reads its workflow commands.
        print("::error::board-status: %s" % e)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except AttributeError:
            pass
    sys.exit(main())
