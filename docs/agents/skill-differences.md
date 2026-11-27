# Where the repo differs from the skills

The agent skills this repo follows (`mattpocock/skills`, set up with
`/setup-matt-pocock-skills`) read the tracker and the docs in the shape they
write them. Each line below is a place where the repo differs on purpose: a
setup run or an audit against a skill keeps it. A new difference is a new line
here.

## Tracker

- The `idea` marker (parked) and the `quick` marker (a fast owner decision)
  sit beside the five triage roles and the two categories
  (`docs/agents/triage-labels.md`).
- The open issues with no category get `bug` or `enhancement` from a list an
  agent proposes and the owner approves before anything is applied (#367).
- A rejected enhancement is recorded in `.out-of-scope/<concept>.md`, the
  folder `/triage` keeps, created at the first rejection.
- `/wayfinder` is adopted as it is written, with its five `wayfinder:` labels
  (#366).
- The issues are planned on the public project board (gpillon/projects/6):
  Status, Release, Area and dates, set by the session that publishes,
  triages or coordinates (`docs/agents/issue-tracker.md`, Project board).
- Whichever agent runs a campaign of issues reads the same issues and
  sub-issues: qwen-code with the `issue-orchestrator` skill, which leaves the
  specialist work to the skills, or a coordinating session with
  `/implement-spec` (`docs/agents/orchestrated-builds.md`).

## Records

- A byte-level, ABI or wire-format contract that code cites as normative lives
  in `docs/formats/`, a home the skills' taxonomy does not have (#369).
- Durable findings live in `docs/findings/` under `docs/agents/findings.md`,
  written with the `finding` skill, an extension beside the skills;
  `/research` saves where the repo already keeps such notes, so it writes
  there too.
- A `prototype/<name>` or `research/<name>` branch is pushed to `origin` and
  never merged; `main` keeps only the validated decision.

## ADRs

The rules are in `docs/agents/domain.md`; these are the places where they
leave the `/domain-modeling` skill.

- ADRs 0001-0069 keep their long form and the number in their heading; a new
  ADR is in the short format, its heading the bare title.
- Every ADR opens with the status frontmatter the skill leaves optional, and
  `docs/adr/README.md` is an index generated from it.
- An ADR states what holds today: a decision that changes rewrites its ADR in
  place, under its number, where the skill's `superseded by` status has a new
  ADR replace the old one.
- A dead ADR is deleted once no live part of the repo cites it, and
  `docs/adr/REMOVED.md` lists it with where its decision went; the skill keeps
  a superseded ADR in the folder.
- A new ADR takes the next number above every number in `docs/adr/` and in
  that list; the skill's highest number in the folder plus one would use a
  removed last number again.
- The frontmatter's relations between ADRs, `amends`, `supersedes` and
  `amended-by`, are not the skill's and are on their way out (#742): each
  one's effect is written into the ADR it changed and the key dropped, and a
  new ADR takes none.
