# Issue tracker: GitHub (gpillon/ignis)

Issues for this repo live **exclusively on GitHub** (`github.com/gpillon/ignis`,
managed with the `gh` CLI). GitHub is the **single source of truth** for issue
tracking: status, blocking, labels, and closure.

Implementation specs (acceptance criteria, seam description, ADR references)
live under `.scratch/<feature>/specs/` in this repo. These are **specs, not
issues** — they do not track status or blocking. `.scratch/` is also used for
temporary artifacts, experiments, and workflow output, but **never for issue
tracking**.

## Division of responsibility

| Content | Where | Why |
|---------|-------|-----|
| Status (open / closed / in-progress) | GitHub Issue | Native close, CI triggers, external visibility |
| Blocking relationships | GitHub Issue body (`**Blocked by:** #X`) + native blocking from UI | Single authoritative source for dependencies |
| Owner / milestone / labels | GitHub Issue | Tracker metadata |
| Implementation spec (seam, acceptance criteria, ADR refs) | `.scratch/<feature>/specs/NN-name.md` | Rich formatting, versioned with code, readable offline |
| Cross-cutting open items (span 2+ crates or external blockers) | `.scratch/PENDING.md` | No single GitHub issue owns them |

## Conventions

- One GitHub issue per work item; the issue title is the canonical title.
- **Issue body is short** (1-3 lines of context + `**Spec:**` link +
  `**Blocked by:**` references). The full spec text lives in the `.scratch/`
  file, never duplicated in the issue body.
- The five triage labels (see `triage-labels.md`) are applied at issue
  creation on GitHub. The `.scratch/` spec file may reference the GitHub
  issue number for traceability, but must not restate status/blocking.
- "Publish to the issue tracker" = `gh issue create` with a short body
  pointing to the `.scratch/` spec file.
- "Fetch the relevant ticket" = `gh issue view <n>`; then read the linked
  `.scratch/<feature-slug>/` spec file for implementation details.
- **PENDING.md hygiene:** at each integration step, prune resolved items
  from `.scratch/PENDING.md`. Never accumulate "X is resolved" paragraphs —
  that lives in git history.

## Auth

Repo-local git credential helper (see `.git/config`:
`credential.https://github.com.helper = store --file ...`). The token lives in
the user's credential file, **never** in URLs or committed files.

## PRs as a request surface

Disabled by default — external PRs are not part of the triage queue. A
maintainer who wants that can enable it here.