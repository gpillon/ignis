# AGENTS.md

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `gpillon/ignis` (via the `gh` CLI); local
specs/plans live under `.scratch/` in this repo. See `docs/agents/issue-tracker.md`.

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