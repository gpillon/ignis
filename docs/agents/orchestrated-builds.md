# Orchestrated builds

How a coordinating session builds a spec's tickets with implementer agents
working in parallel (`/implement-spec`), and this repo's answer to each choice
the skill leaves open.

## Worktrees

The integration branch has its own worktree too.
Each worktree builds into its own target directory: two worktrees on one
`CARGO_TARGET_DIR` link each other's workspace crates, with no error unless a
signature happens to break.

## Agents

The notes the implementers share live in the coordinator's `.scratch/`.

## Checks

Implementers run the CPU-side suites only: `make ci`, or the narrower targets
their brief names. The coordinator runs every GPU leg (the profile, a bench
run, a gate leg) itself, under the GPU rule in `AGENTS.md`, Testing
(`docs/agents/testing.md`, the GPU test profile).

The local checks run on one host, so code behind the other host's `cfg`
compiles nowhere before `main`. Before the owner merges the integration
branch, the coordinator pushes its tip as `ci/<spec>`, with the owner's OK
since it is a push: the release workflow runs its Linux and Windows test jobs
on `ci/**` without touching `main`. The coordinator reports both results with
the branch and deletes `ci/<spec>` once the owner has merged.

## The end of a build

There is no pull request: the owner merges the integration branch into `main`
locally, and nothing is pushed until the owner says so.

An implementer's merge of the integration branch into its own takes a message
its brief gives it, such as `Merge 374-integration into the T2 branch before
the report (#377)`: the commit-message hook refuses git's default one.

## Cleanup

Before `git worktree remove --force`, the coordinator copies the worktree's
`.scratch/` into the main checkout's `.scratch/<branch>/` and checks that every
`.scratch/` path a finding or a report cites still resolves. `.scratch/` is
untracked, so removing the worktree deletes it without a warning.
