# Domain docs

The repo is a single context: one glossary, `GLOSSARY.md` at the root, and one
set of decisions, `docs/adr/`, listed in its generated index
`docs/adr/README.md`. Before exploring an area, read the ADRs that touch it.

- A domain concept is named with the term `GLOSSARY.md` defines, in an issue
  title, a proposal, a hypothesis or a test name; a concept it lacks is a gap
  to note for `/domain-modeling`.
- An output that contradicts an ADR names the ADR and says why it is worth
  reopening.
- A new ADR is written in the `/domain-modeling` skill's short format, with the
  bare title as its heading: the release notes take the number from the file
  name.
- An ADR takes the next free number when it merges into `main`: one above
  every number in `docs/adr/` and in the list of removed ADRs,
  `docs/adr/REMOVED.md`, since a removed number is never used again. If `main`
  took that number first, the ADR is renumbered with every line that cites it
  (b61a224f).
- An ADR opens with a frontmatter that gives its `status`, `proposed`,
  `accepted` or `deprecated`, and nothing else: no relation to another ADR,
  since a change to a decision is written into its ADR and a superseded ADR is
  deleted (below), and `make records-check` refuses `amends`, `amended-by`,
  `supersedes` and `superseded by ADR-NNNN`. `mk/adr-index.py` documents the
  block and generates the index from it (`make records`).
- An ADR states what holds today, and why (ADR 0081). It carries no history
  and no description of how the mechanism works, and it does not restate a
  value the code or the config owns: it names the field or the constant.
- A length note goes in the final report. `make records-check` prints one, and
  still passes, for an ADR the change added or rewrote whose text is over the
  soft limit (`WORD_LIMIT` in `mk/adr-index.py`): shorten the ADR, move the
  detail to the home that owns it (`docs/agents/findings.md`, "Artifact
  boundaries"), or say in the report that it stays long.
  `make records-long` lists every ADR over the limit.
- A decision that changes rewrites its ADR in place: the new text replaces the
  old, the number stays, and the file is renamed, with the links that name it,
  only when the title changes. The earlier versions are the history of the
  file: `git log -p -- "docs/adr/NNNN-*"`.
- When a decision changes, the live places that cite its ADR are reread in the
  same change: the number stays valid, a comment that paraphrases the old
  decision may not. Live is everything but the historical set:
  `docs/findings/`, the release notes, and `docs/specs/` and `docs/archive/`,
  which each keep one page saying where their files went, with commit
  messages and issues. A folder nobody lists there is live (`HISTORICAL` in
  `mk/adr-index.py`). `make records-check` names each ADR whose text changed.
- When an ADR is written:
  - a new decision is a new ADR, `proposed` while it waits for the owner;
  - a change to a decision that the owner has confirmed, in a grilling for
    instance, rewrites its ADR at once: the ADR records what is decided, and
    the tickets bring the code to it;
  - a change that still waits for the owner stays in its issue or its branch,
    and rewrites the ADR once accepted: an accepted ADR never goes back to
    `proposed`, and `make records-check` and the records workflow refuse one
    that does.
- An ADR that holds no decision any more is deleted, for one of three reasons:
  - it was superseded;
  - it only amended another ADR and was absorbed into it;
  - it is not a decision by the `/domain-modeling` skill's three criteria,
    hard to reverse, surprising without its context and the result of a real
    trade-off: a default of the skills, a rule a check already holds, or a
    register that only grows. This one is a judgement: two agents that judge
    independently agree, or the ADR stays.
- Before an ADR is deleted, what it held has its home and no live part cites
  it. The ADR that now holds the decision carries why the old one died and any
  fact from it that still holds; the content of one that was not a decision is
  in the home that owns it, the agents' docs, the coding standards or the
  guide. Each citation in a live part moves to the ADR that holds the
  decision, or the code that cites it changes or goes.
- The change that deletes an ADR's file adds its line to
  `docs/adr/REMOVED.md`, in ascending order:
  ``| NNNN | its title | `NNNN-the-name-its-file-had.md` | where it went |``.
  Where it went names the ADR that holds the decision (`ADR NNNN`), the home
  that took the content (its path from the root, in backticks), or says that
  the decision was retired.
- `make records-check` and the records workflow hold the removal: they refuse
  a citation in a live part, by number or by path, of an ADR that has no file;
  a deleted ADR with no line in the list; a line for an ADR that still exists,
  or naming an ADR or a path that does not; and a new ADR on a removed number,
  naming the next free one. A test that builds ADRs of its own is listed in
  `FIXTURES` in `mk/adr-index.py`.
- The release notes call an ADR amended when a commit changes its text and
  removed when one deletes it (`mk/changelog.sh`), so a commit that changes no
  decision says so with a mark in its message, before the closing issue:
  `[typo ADR-NNNN]` (or a list, `[typo ADR-0032, ADR-0045]`) for a typo fix in
  the ADRs it names, and `[adr-cleanup]`, once, for a commit that brings ADRs
  to the short format or deletes one whose decision did not change, whichever
  ADRs it touches. A commit that changes a decision carries neither.
- Where a skill's convention and the repo's disagree,
  `docs/agents/skill-differences.md` lists where the repo differs; a new
  difference is a line there.
