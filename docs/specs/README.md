# Specs

The specs written before 2026-10-25, one folder per feature, were here. They
left the main branch on 2026-11-27 and are whole under the tag
`records-2026-11-27`. Each is linked from its issue: an issue opened before
2026-10-25 has a short body that links its spec at the tag. Since that date a
spec is a GitHub issue and its tickets are its sub-issues
(`docs/agents/issue-tracker.md`).

A comment's `spec <feature>/<nn>` names `docs/specs/<feature>/<nn>-*.md` at the tag.

To open one, and to list them:

```sh
git show records-2026-11-27:docs/specs/<feature>/<file>.md
git ls-tree -r --name-only records-2026-11-27 docs/specs
```
