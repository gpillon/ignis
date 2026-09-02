# AGENTS.md

## Agent skills

### Issue tracker

Issues live **exclusively on GitHub** (`gpillon/ignis`, via the `gh` CLI).
GitHub is the single source of truth for issue tracking: status, blocking
relationships, labels, and closure. Each GitHub issue body is short
(1-3 lines of context) and links to the implementation spec.

Implementation specs (acceptance criteria, seam description, ADR references)
live under `.scratch/<feature>/specs/` in this repo. These are **specs, not
issues** — they do not track status or blocking (that is GitHub's job).
`.scratch/` is also used for temporary artifacts, experiments, and workflow
output, but **never for issue tracking**.

See `docs/agents/issue-tracker.md` for the full convention.

### Triage labels

The five canonical triage roles, mapped 1:1 to same-name labels
(`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`).
See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — root `CONTEXT.md` for the glossary, `docs/adr/` for ADRs.
See `docs/agents/domain.md`.

### Testing

Every code change ships with a test, and the task is not complete until
`cargo test` passes workspace-wide. See `docs/agents/testing.md`.