# Removed ADRs

An ADR that left the tree, and where its decision or its content went: the ADR
that holds it now, the home that took it, or that the decision was retired. A
removed number is never used again. The last text of a removed file is in git,
`git log -p -- docs/adr/<file>`, and every ADR as it was on 2026-11-27, before
the first one left, is at the tag `records-2026-11-27`:
`git show records-2026-11-27:docs/adr/<file>`. When an ADR is removed, and how
a line here is written, is in `docs/agents/domain.md`; `make records-check`
holds this list to the folder.

| ADR | Title | File | Where it went |
|---|---|---|---|
| 0003 | 99% token-agreement parity gate | `0003-parity-gate-99pct.md` | superseded by ADR 0007, which says why agreement with a reference engine's tokens is not what makes ignis correct, and keeps the one fact of it that holds: greedy output is not byte-identical across builds |
| 0070 | A spec is a GitHub issue and its tickets are sub-issues | `0070-a-spec-is-a-github-issue-and-its-tickets-are-sub-issues.md` | not a decision but the skills' default, which `docs/agents/issue-tracker.md` states; its list of where the repo differs from the skills is `docs/agents/skill-differences.md` |
| 0086 | The builds run beside the tests, and nothing is published before they pass | `0086-the-builds-run-beside-the-tests-and-nothing-is-published-before-they-pass.md` | absorbed into ADR 0032, whose pipeline it reordered; how the workflow, the container build and the test shards do it is in the comments of `.github/workflows/release.yml`, `Containerfile` and `mk/test-shard.sh` |
| 0100 | The 27B cuts a reusable prompt's tail once, at the generation opener, on every card | `0100-the-27b-cuts-a-reusable-prompts-tail-once-at-the-generation-opener-on-every-card.md` | absorbed into ADR 0029, whose rule for the generation opener's page it extended to every card; the measurement switch that gives the second cut back is in `docs/user/contributing/testing.md` |
