# Issue tracker: GitHub (gpillon/ignis)

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Make an issue a sub-issue of a parent**: `gh issue create --parent <parent> ...`, or `gh issue edit <parent> --add-sub-issue <child>` afterwards (`gh` 2.94+). Older `gh`: `gh api --method POST repos/<owner>/<repo>/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>` (database id, as in **Blocking** below). Without sub-issues, put `Part of #<parent>` at the top of the child body.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

## Pull requests as a triage surface

**PRs as a request surface: yes.** `/triage` reads this flag: an external PR is a feature request here.

PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh api --paginate 'repos/{owner}/{repo}/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`.
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue, then set its fields on the board (**Project board**
below).

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (see **Make an issue a sub-issue of a parent**). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). GitHub reports `issue_dependencies_summary.blocked_by` (open blockers only, the live gate). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (`issue_dependencies_summary.blocked_by > 0`, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.

## Issues from before 2026-10-25: the spec is a file

Until 2026-10-25 a spec lived as a file under `docs/specs/<feature>/`, and its
issue carried only a short body linking it (`**Spec:** [docs/specs/...](...)`).
For those issues the linked file is the spec and the ticket's brief: read it
after the issue body. `docs/specs/` is frozen (`docs/specs/README.md`): a new
spec is an issue.

## Closing work

Commits cite the ticket they deliver as `(#<n>)`. The coordinating session
closes a ticket with `gh issue close <n> --comment "..."` naming the commit on
`main` that delivers it, and comments on the ticket when work starts and when
it merges. Implementer agents never comment on or close an issue. A ticket whose
acceptance is the owner's (a live check, a GPU gate the owner runs) stays open
until the owner says so. With its start and merge comments the coordinating
session moves the ticket's Status on the board (**Project board** below).

## Project board

The public project [ignis](https://github.com/users/gpillon/projects/6) plans
these issues. An open issue joins it by itself in Inbox, and closing it moves
it to Done. The issue stays the source of truth and the board adds what no
label says: labels lead, the board follows them.

The session that publishes or triages an issue sets its fields; implementer
agents leave the board alone, as they leave the issue.

- **Area**: on every new issue, the one area it touches.
- **Release**: when the spec or the owner names one. A ticket takes its
  spec's Release and Area.
- **Status**: the labels set it, through a workflow, once the owner has set
  up its credential (**Status from the labels** below; `gh secret list` names
  `BOARD_TOKEN` or `BOARD_APP_KEY` once one is). Until then the session sets
  it by hand from the table there. The coordinating session moves a ticket to
  In progress when work starts, and to Waiting on owner when it merges with
  the owner's acceptance still open.
- **Start date** and **Target date**: on release work only, estimates along
  the blocking edges that the coordinating session moves when the plan moves.

One field per call, by name (`gh` needs the `project` scope:
`gh auth refresh -s project`):
`gh project item-edit 6 --owner gpillon --url <issue-url> --field Area --value "Platforms"`;
a date takes `--value YYYY-MM-DD`, and the option names of Area, Release and
Status come from `gh project field-list 6 --owner gpillon --format json`.
An issue not on the board yet joins with
`gh project item-add 6 --owner gpillon --url <issue-url>`. The board sets a
newly added issue's Status to Inbox and no label event follows an
`item-add`, so a session that adds an issue by hand sets its Status last,
from the table below.

### Status from the labels

`.github/workflows/board-status.yml` sets an issue's Status from its labels
whenever the issue is opened, reopened, labeled or unlabeled, with this table,
which stays the authority (`mk/board-status.py` holds the rules, and its
selftest refuses the script's copy of the table once it differs from this
one):

| Label | Status |
|---|---|
| `idea` | Parked |
| `needs-triage`, `needs-info` | Inbox |
| `ready-for-agent`, `ready-for-human` | Ready |

`idea` wins over any state label beside it. The run reads the issue as it is
when the run starts, not as the event saw it, and leaves the Status alone
when:

- it is In progress or Waiting on owner, which the coordinating session sets;
- the issue is closed: the board moves it to Done itself;
- the issue is not on the board, or its item there is archived;
- no label names a Status: `wontfix`, a category or a marker alone;
- its state labels name two Statuses, which the run reports as a warning:
  the label event that leaves one of them sets it;
- it already is the one the labels name.

On a new or a reopened issue the run first waits, up to a minute, for the
board's own workflows to add the issue, give it Inbox and take it out of Done,
so that they do not overwrite the Status it sets. A run on a fork is skipped.
Until the owner sets up its credential (below) the workflow does nothing and
says so in one notice per run.

### What the owner sets up

The Actions `GITHUB_TOKEN` cannot reach a project, so the workflow takes its
credential from the repository's settings (Secrets and variables, Actions):

- **The board**: the variables `BOARD_OWNER`, the user or organization that
  owns it, and `BOARD_NUMBER`, the project's number; unset, they are
  `gpillon` and `6`. The Status field and its options are found by name, so
  a board rebuilt elsewhere needs only these two values, and the options
  Inbox, Parked, Ready, In progress, Waiting on owner and Done: a board
  missing one fails the run, naming it.
- **The credential, today**: a classic personal access token with the
  `project` and `repo` scopes, as the secret `BOARD_TOKEN`: the credential
  GitHub's guide gives a user's board, where it gives an organization's a
  GitHub App.
- **After the move to an organization (#523)**: a GitHub App installed on
  the organization, with read and write on its projects and read on issues;
  its Client ID as the variable `BOARD_APP_ID` and a private key as the
  secret `BOARD_APP_KEY`. With both credentials set the App is used, and
  `BOARD_TOKEN` can go.

To check it, with a token in `GH_TOKEN` (`gh auth token`) and the board in
`BOARD_OWNER` and `BOARD_NUMBER`,
`python mk/board-status.py --repo gpillon/ignis --issue <n> --dry-run`
says what a run would set and sets nothing.
Live, on a test issue on the board, adding and removing each state label and
`idea` moves its Status within a minute, and each run of the workflow
(`gh run list --workflow board-status.yml`) says what it did.

## Auth

Before pushing: push over HTTPS with `gh` as `gpillon`. The token comes from
the repo-local credential helper `.git/config` names and stays in its file,
never in a URL or a committed file.
