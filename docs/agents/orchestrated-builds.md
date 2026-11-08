# Orchestrated builds

How a coordinating session builds a spec's tickets with implementer agents
working in parallel (`/implement-spec`), and this repo's answer to each choice
the skill leaves open.

## Worktrees

The integration branch and each implementer's branch get their own worktree,
one per ticket, every implementer branch starting from the integration branch.
Each worktree builds into its own target directory: two worktrees on one
`CARGO_TARGET_DIR` link each other's workspace crates, with no error unless a
signature happens to break.

## Agents

Each ticket gets a fresh implementer whose brief points at the spec, the
ticket and any shared notes, which live in the coordinator's `.scratch/`. An
implementer is stopped once it reports; follow-up work goes to a new agent with
a short handoff in the scratchpad, never back to the finished one. At most
three agents run at once, reviews included.

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

There is no pull request. Each implementer's branch merges into the
integration branch, the coordinator reports that branch, and the owner merges
it into `main` locally; nothing is pushed until the owner says so. Tickets
close as `docs/agents/issue-tracker.md` says under "Closing work".

An implementer merges the integration branch into its own before it reports,
with a message its brief gives it, such as `Merge 374-integration into the T2
branch before the report (#377)`: the commit-message hook refuses git's
default one.

## Cleanup

Before `git worktree remove --force`, the coordinator copies the worktree's
`.scratch/` into the main checkout's `.scratch/<branch>/` and checks that every
`.scratch/` path a finding or a report cites still resolves. `.scratch/` is
untracked, so removing the worktree deletes it without a warning.
