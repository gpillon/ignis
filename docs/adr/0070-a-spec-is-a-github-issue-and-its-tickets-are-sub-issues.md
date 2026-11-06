# A spec is a GitHub issue and its tickets are sub-issues

The first `/setup-matt-pocock-skills` ran on 2026-09-02, before the GitHub
remote existed, and chose the local-markdown tracker (379c2ece); 25 minutes
later only issue status moved to GitHub (47b6c68a), that evening the specs
stayed files in a "dual-tracker" convention no decision recorded (108cddc8),
and on 2026-09-21 they moved from `.scratch/` to `docs/specs/` (b1ed4de7). The
skills moved on meanwhile, renaming the glossary to `GLOSSARY.md` upstream
(mattpocock/skills d80fa0f4) and writing tickets as sub-issues with native
blocking, so on 2026-10-25 the repo renamed its glossary (2944c880) and re-ran
the setup against GitHub (2205db6c): a spec is a GitHub issue written with
`/to-spec`, and its tickets are its sub-issues, written with `/to-tickets`.
The skills read the tracker in the shape they write it, and every place where
the repo still differs from them is listed below, as the owner decided on
2026-10-25 (#363), so that the next setup run or audit keeps it instead of
undoing it.

## Consequences

- `docs/specs/` is frozen and not migrated: an issue opened before 2026-10-25
  links its spec there. Migration was rejected because GitHub cannot backdate
  an issue, four of the 158 files exceed the 65,536-character body limit, about
  275 citations would move, and about half the specs were edited after their
  tickets closed.
- A byte-level, ABI or wire-format contract that code cites as normative lives
  in `docs/formats/`, a home the skills' taxonomy does not have (#369).
- Durable findings live in `docs/findings/` under `docs/agents/findings.md`,
  written with the `finding` skill, an extension beside the skills;
  `/research` saves where the repo already keeps such notes, so it writes
  there too.
- The `idea` marker (parked) and the `quick` marker (a fast owner decision)
  sit beside the five triage roles and the two categories
  (`docs/agents/triage-labels.md`).
- ADRs 0001-0069 keep their long form and the number in their heading; a new
  ADR is in the short format, its heading the bare title.
- qwen-code orchestrates a campaign of issues with the `issue-orchestrator`
  skill, which reads the same issues and sub-issues and leaves the specialist
  work to the skills; Claude Code runs `/implement-spec`.
- `/wayfinder` is adopted as it is written, with its five `wayfinder:` labels
  (#366).
- A `prototype/<name>` or `research/<name>` branch is pushed to `origin` and
  never merged; `main` keeps only the validated decision.
- A rejected enhancement is recorded in `.out-of-scope/<concept>.md`, the
  folder `/triage` keeps, created at the first rejection.
- The open issues with no category get `bug` or `enhancement` from a list an
  agent proposes and the owner approves before anything is applied (#367).
- The issues are planned on the public project board (gpillon/projects/6):
  Status, Release, Area and dates, set by the session that publishes,
  triages or coordinates (`docs/agents/issue-tracker.md`, Project board).
