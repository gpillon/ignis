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

Implementers run their own ticket's tests only: the Rust ones through
`cargo test`'s filter, the GPU ones through the profile script's single-test
mode (`-Test` / `--test`), which takes the GPU lock. The workspace suite runs
on `ci/<spec>` at the end of the build (below); the coordinator keeps the full
profile, which runs before `main` moves on a spec that touches the GPU
(below), the bench runs and the gate legs. Both run
on the card under the GPU rule in `AGENTS.md`, Testing
(`docs/agents/testing.md`, the GPU test profile): an agent whose own model is
served from the card hands every GPU step to the user.

The local checks run on one host, so code behind the other host's `cfg`
compiles nowhere before `main`: the run on `ci/<spec>` at the end of a build
is where it compiles first.

## The end of a build

There is no pull request. Once every ticket is merged and the review's fixes
are in, the coordinator lands the integration branch on `main` itself, without
asking:

1. It merges `main` into the integration branch and pushes the tip as
   `ci/<spec>`: the release workflow runs its Linux and Windows test jobs on
   `ci/**` and publishes nothing.
2. A red job is fixed on the integration branch, within the spec, and pushed
   again; a fix outside the spec goes to the owner first. A test that fails and
   then passes on a rerun of the same commit (`gh run rerun --failed`) is
   flaky: the coordinator opens an issue labelled `bug` and `needs-triage`
   that names the test, the job and its failing lines, and the green rerun
   counts. A test that fails its rerun too is a real failure.
3. When the spec touches the GPU (`docs/agents/testing.md`, "Done means",
   item 4: paths and a pattern read off `git diff --name-only main...`), the
   full GPU profile runs on that tip, `make gpu-profile`, and must be green; a
   spec that touches no GPU file skips this step. When step 1 runs again
   because `main` moved, the profile runs again if the commits it merged in
   touch the GPU too.
4. With every job of the run green, and the profile where step 3 runs it,
   `main` moves to that commit by fast-forward and is
   pushed by its SHA, which publishes the `develop` image and prerelease. If
   `main` moved meanwhile, step 1 runs again.
5. `ci/<spec>` is deleted, with its Actions caches.

An implementer's merge of the integration branch into its own takes a message
its brief gives it, such as `Merge 374-integration into the T2 branch before
the report (#377)`: the commit-message hook refuses git's default one.

## Cleanup

Before `git worktree remove --force`, the coordinator copies the worktree's
`.scratch/` into the main checkout's `.scratch/<branch>/` and checks that every
`.scratch/` path a finding or a report cites still resolves. `.scratch/` is
untracked, so removing the worktree deletes it without a warning.
