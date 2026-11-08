---
status: accepted
amends: [ADR-0070]
---

# An ADR's status is in its frontmatter, and a decision that changes is recorded by a new ADR

By 2026-11 the ADRs wrote their status three ways (a `## Status` section in
0001-0069, frontmatter in 0077, 0079 and 0080, nothing in the other short
ones), and the Status sections had grown into logs of amendments made in place,
222 lines in ADR 0017 and 188 in ADR 0045, so no list of the decisions could be
drawn from them and an ADR's first lines no longer said what held. The owner
decided on 2026-11-08 that every ADR opens with a frontmatter giving its
`status`, in the `/domain-modeling` vocabulary (`proposed`, `accepted`,
`deprecated`, `superseded by ADR-NNNN`), and the relations that change a
decision, `amends`, `supersedes` and `amended-by`, lists of `ADR-NNNN` the skill
does not have, which ADR 0070's list of the repo's deviations gains through
this ADR. A decision that changes is recorded by a new ADR that amends or
supersedes the old one; the old one then changes only in its frontmatter, never
in its text, which stays the record of what was decided then and why.

## Considered Options

- **Amending in place, as before**: it is what grew the Status sections, and
  the release notes marked every ADR it touched as amended.
- **Rewriting every ADR to its current decision**: it erases the reasons of the
  time, which are what an ADR is kept for, and would rewrite text that code,
  findings and the guide cite (ADR 0006 alone 447 times).

## Consequences

- 0001-0069 keep their `## Status` section as history; the frontmatter is the
  source of truth over it. An amendment an issue made in place before this rule
  stays in the prose.
- A relation that only extends or clarifies has no key and stays in the prose.
  A status names one superseder: an ADR that two others replaced names the main
  one and lists the other under `amended-by`.
- The ADRs whose decision had drifted through amendments in place got, once, in
  the 2026-11 cleanup, a "Current decision" section, the old text kept under
  "History".
- `docs/adr/README.md` is an index generated from the frontmatter by
  `mk/adr-index.py` (`make records`), never edited by hand. Its check
  (`make records-check`, and the `records` workflow) refuses a missing or
  invalid frontmatter, a reference to an ADR that does not exist, back-links
  that disagree, a heading from 0070 on that carries its number, and an index
  out of date.
- The release notes (`mk/changelog.sh`) call an ADR amended only when its text
  changed, the frontmatter and link targets aside, and say when its status
  changed.
