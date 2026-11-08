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
- An ADR changes through a new ADR that amends it (ADR 0081).
- Where a skill's convention and the repo's disagree, ADR 0070 lists where the
  repo differs.
