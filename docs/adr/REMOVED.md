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
| 0103 | On compute capability 12.0 the 27B holds its exception tensors in FP8 by default | `0103-on-compute-capability-12-0-the-27b-holds-its-exception-tensors-in-fp8-by-default.md` | absorbed into ADR 0065, whose field's default it set; the KLD leg that admitted the default and the context a start keeps on the RTX 5090 are `docs/findings/2026-11-18-the-27bs-fp8-exception-weights-at-their-served-shape-on-the-rtx-5090.md` |
