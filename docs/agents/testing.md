# Testing

How changes to this repo are verified. One rule governs everything
below: every code change ships with a test, and a task is not complete
until the test suite is green.

## The gate

- `cargo test` from the workspace root — all five crates
  (`artifact`, `core`, `server`, `bench`, `vendor`).
- Machine-local smoke tests skip gracefully when their fixture is
  absent (e.g. `crates/artifact/tests/real_artifact.rs` needs the
  artifact in `F:\ai\q38`). A skip counts as green; a failure does not.
- **Exception — compute work: a skip is never green.** For anything on the
  forward pass (the kernel leaf, the step ABI, the program), a test that
  self-skips on a busy GPU or on a kernel error proves nothing; that pattern
  is what let a broken forward stay green for two tickets
  (`.scratch/REVIEW-2026-09-05.md` §4.1). Those tests belong to the explicit
  **GPU profile**, which requires the GPU free and *fails* on any kernel
  error. The default `cargo test` stays CPU-only. The profile, its preflight
  and its runbook are GitHub #38.
- Kernel work (`kernel/`, CUDA) is not exercised by `cargo test`: build and
  run the leaf's own op-test executable (CTest) with `kernel/build.ps1 -Test`,
  and verify end to end through the GPU profile. The one kernel-adjacent thing
  `cargo test` *does* check is the vendored subtree's integrity
  (`crates/vendor`): a vendored file edited without a recorded patch turns the
  workspace red (ADR 0010, `kernel/vendor/VENDOR.md`).

## Where tests live

- Unit tests: `#[cfg(test)]` modules, in the file they test.
- Integration tests: `crates/<crate>/tests/`, one file per behavior,
  named for what it verifies (cf. `real_artifact.rs`).

## Writing the tests

- New behavior: its test lands with the change, before the task is
  reported done — and it exercises the new path, not the old world
  (a test that already passed before the change proves nothing about
  it).
- Bug fix: the regression test goes first — reproduce the bug, watch
  the test go red, then fix.
- The standard loop is red-green, driven by the `/tdd` skill.

## Done means

1. every new or changed behavior has a test that covers it
2. `cargo test` passes workspace-wide (machine-local skips allowed)
3. kernel changes: `kernel/build.ps1` clean + the leaf's op tests green +
   the GPU profile green on a free 5090 (never a skip)