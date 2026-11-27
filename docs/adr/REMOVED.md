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
| 0014 | G1's cross-engine canary floor is teacher-forced next-token agreement | `0014-g1-teacher-forced-canary-agreement.md` | ADR 0082, which retired G1 as a gate and keeps its scoring for the canary test; the scoring rule is the opening comment of `crates/bench/src/oracle.rs` |
| 0015 | G2 is judged live/live against the reference, on cold-prefix samples | `0015-g2-live-live-cold-prefix-gate.md` | retired by ADR 0082; the measuring rule the bench instruments still follow is the opening comment of `crates/bench/src/lib.rs` |
| 0021 | a live/live gate pools at least two independent process launches per engine | `0021-live-live-launch-pooling.md` | retired as a gate's rule by ADR 0082, which keeps the two launches for a speed claim; its measurement is `docs/findings/2026-09-10-one-process-launch-picks-its-own-throughput-band.md` |
| 0039 | the attention readout carries a head set | `0039-the-seam-carries-a-head-set.md` | absorbed into ADR 0038, whose readout it extended from one head to a head set; how the set is read, reduced and reserved is in the comments of `crates/core/src/pointing.rs`, `kernel/src/attention_readout.h` and `kernel/include/ignis_step.h` |
| 0040 | the attention readout carries each peak's neighbours | `0040-the-readout-carries-the-peak-s-neighbours.md` | absorbed into ADR 0038, whose head set it extended with each peak's score and its four neighbours; how they are gathered and read is in the comments of `crates/core/src/pointing.rs`, `kernel/src/attention_readout.cu` and `kernel/include/ignis_step.h` |
| 0070 | A spec is a GitHub issue and its tickets are sub-issues | `0070-a-spec-is-a-github-issue-and-its-tickets-are-sub-issues.md` | not a decision but the skills' default, which `docs/agents/issue-tracker.md` states; its list of where the repo differs from the skills is `docs/agents/skill-differences.md` |
| 0086 | The builds run beside the tests, and nothing is published before they pass | `0086-the-builds-run-beside-the-tests-and-nothing-is-published-before-they-pass.md` | absorbed into ADR 0032, whose pipeline it reordered; how the workflow, the container build and the test shards do it is in the comments of `.github/workflows/release.yml`, `Containerfile` and `mk/test-shard.sh` |
