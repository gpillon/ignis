# Issue tracker: GitHub (gpillon/ignis)

Issues for this repo live **exclusively on GitHub** (`github.com/gpillon/ignis`,
managed with the `gh` CLI). GitHub is the **single source of truth** for issue
tracking: status, blocking, labels, and closure.

Implementation specs (acceptance criteria, seam description, ADR references)
live under `docs/specs/<feature>/` in this repo. These are **specs, not
issues** — they do not track status or blocking.

`.scratch/` is **not tracked by git** (see `.gitignore`): it is the scratchpad
for temporary artifacts, experiments, raw logs and workflow output, local to
each clone. A finding may point at raw material there, but nothing durable
lives in it, and it is never an issue tracker.

## Division of responsibility

| Content | Where | Why |
|---------|-------|-----|
| Status (open / closed / in-progress) | GitHub Issue | Native close, CI triggers, external visibility |
| Blocking relationships | GitHub Issue body (`**Blocked by:** #X`) + native blocking from UI | Single authoritative source for dependencies |
| Owner / milestone / labels | GitHub Issue | Tracker metadata |
| Feature-level spec (the whole feature in one document) | `docs/specs/<feature>/spec.md` | What the tickets below it decompose |
| Implementation spec (seam, acceptance criteria, ADR refs) | `docs/specs/<feature>/NN-name.md` | Rich formatting, versioned with code, readable offline |
| Cross-cutting open items (span 2+ crates or external blockers) | `docs/PENDING.md` | No single GitHub issue owns them |

## Conventions

- One GitHub issue per work item; the issue title is the canonical title.
- **Issue body is short** (1-3 lines of context + `**Spec:**` link +
  `**Blocked by:**` references). The full spec text lives in the `docs/specs/`
  file, never duplicated in the issue body.
- The five triage labels (see `triage-labels.md`) are applied at issue
  creation on GitHub. The `docs/specs/` spec file may reference the GitHub
  issue number for traceability, but must not restate status/blocking.
- "Publish to the issue tracker" = `gh issue create` with a short body
  pointing to the `docs/specs/` spec file.
- "Fetch the relevant ticket" = `gh issue view <n>`; then read the linked
  `docs/specs/<feature-slug>/` spec file for implementation details.
- **PENDING.md hygiene:** at each integration step, prune resolved items
  from `docs/PENDING.md`. Never accumulate "X is resolved" paragraphs —
  that lives in git history.

## Auth

Repo-local git credential helper (see `.git/config`:
`credential.https://github.com.helper = store --file ...`). The token lives in
the user's credential file, **never** in URLs or committed files.

## PRs as a request surface

Disabled by default — external PRs are not part of the triage queue. A
maintainer who wants that can enable it here.