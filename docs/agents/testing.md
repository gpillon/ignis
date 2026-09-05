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
  **GPU profile** (below), which requires the GPU free and *fails* on any
  kernel error, busy GPU, or missing fixture. The default `cargo test` stays
  CPU-only (GitHub #38).
- Kernel work (`kernel/`, CUDA) is not exercised by `cargo test`: build and
  run the leaf's own op-test executable (CTest) with `kernel/build.ps1 -Test`,
  and verify end to end through the GPU profile. The one kernel-adjacent thing
  `cargo test` *does* check is the vendored subtree's integrity
  (`crates/vendor`): a vendored file edited without a recorded patch turns the
  workspace red (ADR 0010, `kernel/vendor/VENDOR.md`).

## The GPU test profile (ADR 0006, GitHub #38)

ignis and ninfer (the reference engine, also the coding agent's own model
runner) cannot share the RTX 5090 — a single engine's footprint already
takes ~28 GB of the 32 GB card (ADR 0006). So every GPU-touching Rust test
is `#[ignore]`d and runs only when asked for explicitly: `cargo test` never
touches the GPU on its own.

A GPU test decides what a busy GPU, a kernel error, or a missing fixture
means by asking [`ignis_core::gpu_profile`] — never by returning early on
its own:

- **Outside the profile** (`IGNIS_GPU_PROFILE` unset): those conditions
  print `SKIP: ...` and the test returns — a quick local check without
  stopping ninfer.
- **Under the profile** (`IGNIS_GPU_PROFILE=1`): the *same* conditions are
  hard failures (`panic!`), never a skip. This is the only mode whose green
  result counts for a gate.

Use it as `gpu_profile::check_rc(rc, "ignis_<op>")` for a kernel return
code, `check_compute_err(&e, "...")` for a `ComputeError`, and
`skip_or_fail("...")` for a missing fixture or an absent GPU. Its own unit
tests (in `crates/core/src/gpu_profile.rs`) pin the skip/fail decision on
CPU with an injected rc — no GPU needed, so they run in the default suite.

There are no GPU-gated Rust tests in the tree right now: GitHub #39 deleted
the superseded forward pass and its `*_gpu.rs` tests. They come back with
the vendored ops and the step ABI (`.scratch/ROADMAP.md`, P1-07 onward),
and must use the helper above rather than reinventing a self-skip. Until
then the GPU-side coverage is the kernel leaf's own op-test executable
(`kernel/build.ps1 -Test`), which the same runbook applies to.

### Running the profile: `scripts/gpu-profile.ps1`

ADR 0006 calls the guard "a preflight check in the `bench`/test harness",
not a script a developer must remember to run standalone. So the profile
takes **two** things, not one: `IGNIS_GPU_PROFILE=1` *and* a preflight pass
on record. `scripts/gpu-preflight.ps1` writes a marker file
(`$env:TEMP\ignis-gpu-preflight.ok`) when it passes and clears it when it
refuses; `active()` reads that marker and **panics** if the variable is set
without a recent one. Setting the variable by hand therefore fails loudly
instead of running un-preflighted while ninfer may hold the card.

`crates/core` has no FFI and no GPU access of its own (GitHub #39 removed
the flat C-ABI surface), so the GPU is inspected by the script and the
verdict is carried across to Rust by that marker.

`scripts/gpu-profile.ps1` is the entry point that ties it together: it runs
the preflight, and only on a pass sets `IGNIS_GPU_PROFILE=1` and runs the
GPU-gated work (`kernel/build.ps1 -Test`, then
`cargo test --workspace -- --ignored`). It consumes the marker — both it and
the env var are cleared before the script exits, pass or fail — so one
preflight authorizes exactly one run. A pass also ages out after 30 minutes,
which only matters if a run was killed before it could clean up.
**This script is the normal, documented way to run the GPU profile.**

Runbook:

```powershell
# 1. Stop ninfer (frees the VRAM the GPU profile needs).
# 2. Preflight + profile in one step: refuses to proceed while the GPU is
#    held (and names the offending process); on a free GPU, runs the leaf's
#    op tests and the #[ignore]d Rust GPU tests (none exist yet -- see
#    above) under IGNIS_GPU_PROFILE=1, then clears it.
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu-profile.ps1
# -ThresholdMiB <n>    forwarded to gpu-preflight.ps1
# -SkipKernelBuild     Rust GPU tests only, skip kernel/build.ps1 -Test
# -SkipCargoTests      kernel leaf only, skip cargo test --workspace -- --ignored
# 3. Restart ninfer.
```

`scripts/gpu-preflight.ps1` still exists as the standalone check
(`scripts/gpu-profile.ps1` calls it) for a quick "is the GPU free" query
that doesn't run anything.

## The canary oracle fixture (P1-05, GitHub #41)

`crates/bench/tests/fixtures/oracle_canary.json` is the committed fixture
G1 is measured against: the reference engine's (`ninfer-serve`, greedy,
the `qwen3_8_27b_nvfp4full-v2.ninfer` artifact) completions on the canary
suite (`crates/bench/src/canary.rs::CANARIES`), 32 greedy tokens per
prompt. `ignis-bench oracle compare` diffs a candidate engine's tokens
against it (§"Oracle (two levels)", spec `01-device-resident-forward`).

Re-record it (needs the GPU and the reference stack, ADR 0006 — stop
`ignis-server` first):

```powershell
# 1. Start the reference engine on the same artifact, greedy, thinking off
#    (thinking on burns the whole token budget on the reasoning channel,
#    leaving no content tokens to compare).
F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe `
  F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --model-id qwen3.8-27b-nvfp4full-v2 --host 127.0.0.1 --port 8080 `
  --greedy --no-thinking --max-context 8192 --max-concurrency 1 --kv-capacity auto

# 2. Record the fixture.
cargo run -p ignis-bench -- oracle record `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer `
  --out crates/bench/tests/fixtures/oracle_canary.json --max-tokens 32

# 3. Self-check: compare the fixture against a fresh live recording (must be
#    100% — greedy + fixed seed on the same artifact is deterministic).
cargo run -p ignis-bench -- oracle compare `
  --fixture crates/bench/tests/fixtures/oracle_canary.json `
  --endpoint http://127.0.0.1:8080 `
  --artifact F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer

# 4. Stop ninfer-serve (frees the GPU) and commit the updated fixture.
```

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
   the GPU profile green on a free 5090 (`scripts/gpu-profile.ps1` —
   preflight passed, `IGNIS_GPU_PROFILE=1` for the run — never a skip)