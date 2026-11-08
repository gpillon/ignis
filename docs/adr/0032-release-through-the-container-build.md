# ADR 0032 — the Linux release is the container build, and it is one pipeline

## Status

Accepted (2026-09-19, owner decision — GitHub #225). **Revised (2026-09-21,
owner decision)**: the two-workflow split below is withdrawn — one workflow,
one Linux job, and a CPU-only test gate in front of every build. The core
decision (the Linux binaries *are* the container build) is unchanged.
**Revised (2026-10-18, GitHub #329 item 9)**: the CUDA binaries link no NVIDIA
driver library; the driver and NVML are opened at run time, so the image runs
its no-GPU commands (`model download`, `--help`) on a machine without a driver.

## Context

Until now `--features cuda` meant Windows. `crates/artifact/build.rs` ran
`kernel/build.ps1` through powershell, looked for `ignis_*.lib` and linked
`$CUDA_PATH/lib/x64`; `mk/os/linux.mk` said so in a comment and failed every
CUDA hook. There was no Linux build to package, and no release pipeline of any
kind: the only way to obtain ignis was to build it on the development machine.

The C++/CUDA was already portable — the vendored TMA sources carry the
non-Windows arm the reference itself runs on (ADR 0010). What was missing was
the build glue, and three host differences that only a real Linux link
surfaces: headers MSVC includes transitively (`<cstring>`, `<cerrno>`), PIC
objects for rustc's PIE link, and the fact that `ignis_vendor` and
`ignis_kernel` depend on each other — the vendored plans call
`ninfer::ops::linear`, whose qtype dispatcher is ours. The MSVC linker
re-scans the archives it is handed and had been resolving that silently; `ld`
walks its inputs once.

## Decision

**The Linux binaries and the container image are one build.** The
`Containerfile`'s `build` stage compiles the kernel leaf and the workspace
inside `nvidia/cuda:<v>-devel`; its `artifacts` stage holds exactly what the
`runtime` image runs, and CI exports the release tarball from that stage
(`--target artifacts --output type=local`). The tarball and the published
image therefore cannot carry different binaries, and no CI job installs a CUDA
toolkit of its own — the base image is the toolchain.

**One workflow, and the image is the same job as the Linux tarball**
(revised 2026-09-21; originally two workflows, `release.yml` and
`image.yml`). Both come out of the same `build` stage, so building them in
two jobs — let alone two workflows — bought nothing but a shared remote cache
scope to keep the kernel from being compiled twice. In one job the builder's
own local cache does that: the `artifacts` build compiles, the `runtime`
build that follows it is layer hits.

The price is the coupling the split was there to avoid: a GHCR push failure
now fails the `linux` job, and therefore withholds the Release. Accepted on
the owner's call — the registry push is the last step of a job whose
expensive half has already succeeded, and a re-run costs a cached build.

**Nothing is built before the tests pass** (2026-09-21; amended 2026-11-08
below: the builds run beside the tests, and nothing is *published* before
they pass). A `test` job runs
the CPU-only suite on both hosts — `cargo test --workspace` plus the
Playground's typecheck and vitest run — and `linux` and `windows` need it.
`cargo test` never touches the GPU on its own (GitHub #38,
`docs/agents/testing.md`), which is what makes it runnable on a runner with
no card: the default build is pure Rust and needs no CUDA toolkit. The GPU
profile is not and cannot be part of CI; correctness on the card stays on the
development machine.

**The Release waits for both hosts, and for nothing else.** `release` needs
`linux` and `windows`.

**A tag is only cut on a pipeline already green.** Every push to `main` or a
`ci/**` branch runs the tests and both builds and publishes nothing — that is
the dry run (amended 2026-10-07 below: a push to `main` now publishes a
development build). A `v*` tag publishes, and its version must equal
`workspace.package.version` (and `web/package.json`'s), or the run fails
before building anything. A tag's run is never cancelled by the concurrency
group.

**Nothing is run on a runner.** No runner has an NVIDIA GPU. The kernel is
compiled for SM120a and links no driver library: the binary reaches the
host's own `libcuda.so.1` through the CUDA runtime and opens
`libnvidia-ml.so.1` at run time (GitHub #329 — until then it was linked
against the toolkit's driver stubs, which kept it from starting at all where
there is no driver). Correctness stays on the machine with the 5090
(`docs/agents/testing.md`).

**The image carries no model and no driver.** A `.ninfer` container is tens of
GB, so it is mounted; the driver is injected by the container runtime.
`IGNIS_ARTIFACT` is deliberately not defaulted — unset, the server starts on
the CPU mock (ADR 0006), which makes running the image a smoke test on its
own.

## Amendment (2026-10-07, owner decision): main publishes a development build

A push to `main` (and a `workflow_dispatch` run on `main`) now publishes, in
addition to running the tests and both builds. It is **not a release**.

- The runtime image is pushed as `ghcr.io/gpillon/ignis:develop`, and only that
  tag: never `latest`, never a version. No per-commit tags — they would pile
  up in GHCR with nothing to prune them; the commit is in the image's
  `org.opencontainers.image.revision` label and in the prerelease body.
- A rolling GitHub prerelease, tag `develop`, holds the Linux tarball, the
  Windows zip and their `.sha256`. The `prerelease` job (needs `version`,
  `linux`, `windows`; `contents: write` for that job only) deletes the old
  release and its tag and creates both again at the run's commit, with
  `--prerelease --latest=false`, so it is never the repository's "latest".
  Delete-and-recreate rather than force-moving the tag: the release can never
  hold a mix of two runs' files. The tag is pushed by `GITHUB_TOKEN`, which
  starts no workflow, and `develop` does not match `v*.*.*`.
- The window: the main concurrency group cancels an older run, and a run
  cancelled between the delete and the end of the upload leaves no `develop`
  release (or one missing assets) until the next main run recreates it. Nothing
  depends on it.
- On main and `ci/**` the archives are named `ignis-develop-<sha7>-…` and
  `ignis-ci-<sha7>-…` instead of `ignis-v<version>-…`; a `v*` tag release is
  unchanged. `ci/**` still publishes nothing.
- A `v*` tag release is unchanged: same image tags, same `release` job, same
  body.

## Amendment (2026-10-12, owner decision): a push that changes no behaviour builds nothing

A push to `main` or `ci/**` whose changed files are all Markdown, `docs/`,
`bench/` or `scripts/` starts no run (`paths-ignore` on the push trigger). None
of them is compiled, embedded, read by a test or used by the workflow, so the
~56-minute run would rebuild the same binaries and move `develop` to an
identical build. Tags are never path-filtered: a release always runs.

- Consequence: `develop` (image and prerelease) stays at the last push that
  changed code, and the README.md inside its archives and image can lag the
  repository's until then.
- `tools/` and `mk/` still trigger: a test runs `tools/flash-next-converter`,
  the server embeds `mk/kv-pool-tokens.sh`, and the workflow runs
  `mk/version.sh`. A file that becomes embedded or read by a test must leave
  the ignore list.
- `workflow_dispatch` still builds on demand.

## Amendment (2026-11-07, GitHub #495): only gpillon/ignis publishes

A fork that keeps Actions on ran this workflow on its own push to `main` and
published a `develop` prerelease under its name, with ignis's notes, from a
commit ignis never checked. Every publishing job and step now starts its
condition with `github.repository == 'gpillon/ignis'`: the `release` and
`prerelease` jobs, the `linux` job's registry login, image push and summary,
and the guide's Pages upload and deployment in `docs.yml`.

- A fork keeps `version`, `test`, `linux` and `windows`: its contributors get
  CI on their own changes, and the archives stay as the run's artifacts.
- On `gpillon/ignis` nothing changes: `main` moves `develop`, a `v*` tag
  releases.
- `crates/server/tests/fork_publishing.rs` reads every workflow and refuses a
  publishing point (a job granting `contents` or `pages` write, a registry
  login, a release or Pages action, a `gh release` or push command, a `push`
  input) whose condition does not start with the guard, so a new one is held
  to it as it lands.

## Amendment (2026-11-08, owner decision, GitHub #552): the builds run beside the tests; nothing is published before they pass

A push to `main` took 46 to 64 minutes, most of it one wait after another:
the `test` legs (Windows 24 to 34 minutes, the critical path) and only then
the two builds (14 to 23 more). Inside a test leg ~80% was compiling, not
testing: rust-cache keeps only the dependencies, so every workspace crate and
every one of the ~330 test binaries was built again on every push (Linux ~10
minutes of ~12.5, Windows ~21 of ~25). The Linux build compiled the kernel
leaf from nothing on every push, its layer coming after `COPY . .`.

- **`linux` and `windows` need only `version`; whatever publishes needs
  `test`.** The 2026-09-21 rule becomes: nothing is *published* before the
  tests pass. `release` and `prerelease` need `test`, and the image is pushed
  by a job of its own, `image` (needs `test` and `linux`). A push whose tests
  fail still pays for its builds, in runner time a public repository does not
  pay for.
- **The image is still the `linux` job's compile.** When the run will
  publish, that job writes the `runtime` image to an OCI archive, and `image`
  copies it to each tag with `skopeo copy --all --preserve-digests`, its index
  and provenance attestation included, as the registry held them when the
  build step pushed. The tarball and the image cannot carry different builds;
  only the push moved.
- **The `test` job runs as four shards per host** (`mk/test-shard.sh`). A
  shard is `cargo test --workspace` with the target flags of its part, so
  features resolve as for the whole workspace and the shard compiles only its
  part. The parts are read off `cargo metadata` and hold every target `cargo
  test --workspace` runs exactly once; a target the flags cannot select as it
  does is refused (`mk/test-shard-selftest.sh`, run in shard 1).
- **The Linux build compiles the leaf in a layer that only `kernel/`
  invalidates, and the dependencies in one that only the manifests and the
  lock file invalidate** (cargo-chef, pinned), and keeps only the three
  shipped binaries from its cargo step: the target directory, rewritten into
  the 10 GB Actions cache on every push, crowded out the layers that are
  reused.
- **The Windows build's kernel cache restores only on an exact hit, and then
  dates the sources it hashes back**: ninja decides by date, the checkout
  dates every source later than the cached objects, and restored as is the
  leaf was compiled again in full on every push (274 commands, ~12.5 minutes).
  An exact hit proves the sources are the bytes the objects came from; a
  partial restore cannot, so there is none.
- **The test job builds with line tables only**, in place of full debuginfo:
  the compile took about half as long on Linux and a sixth less on Windows,
  with the suite unchanged.
- `crates/server/tests/ci_pipeline.rs` holds the workflow and the
  Containerfile to this shape.

## Considered Options

- **Install the CUDA toolkit on the Linux runner** and build with cargo
  directly, as the Windows job does — rejected: a second toolchain to keep in
  step with the image's, several GB of install per job, and two Linux builds
  per push that can disagree.
- **One workflow with the image as a third job** — rejected in 2026-09-19 on
  the owner's call (the image is not part of the Release's contract, and
  coupling their failures means one flaky registry push hides a good
  release), then **adopted in 2026-09-21** in a stronger form: not a third
  job, but the same job, because the two builds share every layer that costs
  anything. The failure coupling is real and accepted above.
- **Make the image a dependency of the Release** — still rejected: `release`
  needs `linux` and `windows`, and nothing else.
- **Gate the builds on the GPU profile too** — impossible: no runner has an
  NVIDIA GPU. CI runs the CPU-only suite, which is the whole of what a
  runner can honestly answer for.
- **Ship a CPU-mock image** (no `--features cuda`), which would build anywhere
  — rejected: an inference engine that cannot infer is not a release.
- **`cudart_static` on Linux** so the tarball needs no CUDA runtime —
  deferred: it diverges from the Windows link, which ships the redistributable
  `cudart64_*.dll` instead. The image is the self-contained option.

## Consequences

- `kernel/build.sh` is the Linux half of `kernel/build.ps1` and must stay in
  step with it: same generator, same Release + `120a` configure, same
  `kernel/build` output. `crates/artifact/build.rs` watches both on both
  hosts.
- The mutual dependency between the two archives is now declared in CMake, and
  `build.rs` names the first archive again on the ELF link line. A new call
  from the vendored substrate into `kernel/src` no longer breaks the Linux
  link silently.
- `.cargo/config.toml` still pins the MSVC target for every host — cargo has
  no host condition for `build.target` — so everything that builds on Linux
  passes `--target` explicitly. A bare `cargo` command there needs
  `CARGO_BUILD_TARGET`. GitHub #226. The `test` job is the newest thing that
  has to say so: without `--target x86_64-unknown-linux-gnu` its Linux leg
  dies in the first dependency with `can't find crate for 'core'`.
- The image is SM120a-only, like every ignis build. It will not run on another
  card.
- `make` on Linux covers the kernel and GPU hooks; background server control
  and the GPU profile are still Windows-only (GitHub #226).
