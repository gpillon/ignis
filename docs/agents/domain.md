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
- An ADR takes the next free number when it merges into `main`; if `main` took
  that number first, the ADR is renumbered with every line that cites it
  (b61a224f).
- An ADR opens with a frontmatter that gives its `status`: `proposed`,
  `accepted`, `deprecated` or `superseded by ADR-NNNN`. `mk/adr-index.py`
  documents the block and generates the index from it (`make records`).
- An ADR states what holds today, and why (ADR 0081). It carries no history
  and no description of how the mechanism works, and it does not restate a
  value the code or the config owns: it names the field or the constant.
- A decision that changes rewrites its ADR in place: the new text replaces the
  old, the number stays, and the file is renamed, with the links that name it,
  only when the title changes. The earlier versions are the history of the
  file: `git log -p -- "docs/adr/NNNN-*"`.
- When a decision changes, the live places that cite its ADR are reread in the
  same change: the number stays valid, a comment that paraphrases the old
  decision may not. Live is everything but `docs/findings/` and the release
  notes; `make records-check` names each
  ADR whose text changed.
- When an ADR is written:
  - a new decision is a new ADR, `proposed` while it waits for the owner;
  - a change to a decision that the owner has confirmed, in a grilling for
    instance, rewrites its ADR at once: the ADR records what is decided, and
    the tickets bring the code to it;
  - a change that still waits for the owner stays in its issue or its branch,
    and rewrites the ADR once accepted: an accepted ADR never goes back to
    `proposed`, and `make records-check` and the records workflow refuse one
    that does.
- The release notes call an ADR amended when a commit changes its text
  (`mk/changelog.sh`), so a commit that changes no decision says so with a
  mark in its message, before the closing issue: `[typo ADR-NNNN]` (or a list,
  `[typo ADR-0032, ADR-0045]`) for a typo fix in the ADRs it names, and
  `[adr-cleanup]`, once, for a commit that brings ADRs to the short format,
  whichever ADRs it touches. A commit that changes a decision carries neither.
- Where a skill's convention and the repo's disagree, ADR 0070 lists where the
  repo differs.
