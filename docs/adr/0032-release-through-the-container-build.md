---
status: accepted
---

# The Linux release is the container build, and it is one pipeline

The Linux binaries and the container image are one build: the `Containerfile`'s `build` stage compiles them, its
`artifacts` stage holds exactly what the `runtime` image runs, and CI exports the release tarball from that stage.
The tarball and the published image therefore cannot carry different binaries, and no CI job installs a CUDA toolkit
of its own: the base image is the toolchain. It is one workflow, `release.yml`, and one job compiles both: they come
out of the same `build` stage, so two jobs, let alone two workflows, bought nothing but a shared remote cache to
keep the kernel from being compiled twice.

- **The builds run beside the tests, and nothing is published before the tests pass.** The CPU-only suite runs on
  both hosts, `cargo test --workspace` plus the Playground's typecheck and vitest run: `cargo test` never touches
  the GPU on its own, and the default build is pure Rust and needs no CUDA toolkit, which is what makes it runnable
  on a runner with no card. The suite may run as shards, which together run every target `cargo test --workspace`
  runs, exactly once. The builds wait for no test; whatever publishes waits for the tests: the release, which also
  waits for both hosts, the `develop` prerelease, and a job of its own that pushes the image, which is still the
  Linux job's compile. A push whose tests fail still pays for its builds, in runner time a public repository does
  not pay for. `crates/server/tests/ci_pipeline.rs` holds the workflow and the `Containerfile` to this shape.
- **Nothing is run on a runner.** No runner has an NVIDIA GPU, so the GPU profile is not and cannot be part of CI:
  the CPU-only suite is the whole of what a runner can answer for, and correctness on the card stays on the
  development machine (`docs/agents/testing.md`).
- **A tag is cut only on a pipeline already green.** A push to a `ci/**` branch runs the tests and both builds and
  publishes nothing: that is the dry run. A `v*` tag releases, and its version must equal the workspace's and
  `web/package.json`'s, or the run fails before building anything. A tag's run is never cancelled.
- **A push to `main` publishes a development build, never a release**, and so does a `workflow_dispatch` run on
  `main`: the image under the tag `develop` alone, never `latest`, never a version, and a rolling prerelease,
  `develop`, that holds the archives and is never the repository's latest release. There are no per-commit tags,
  which would pile up in the registry with nothing to prune them: the commit is in the image's revision label and
  in the prerelease's body.
- **A push that changes no behaviour builds nothing.** A push to `main` or `ci/**` whose changed files are all
  on the workflow's ignore list starts no run: nothing on that list is compiled, embedded, read by a test or
  used by the workflow, so the run would rebuild the same binaries and move `develop` to an identical build. A file
  that becomes embedded or read by a test must leave the ignore list. Tags are never path-filtered: a release
  always runs. `develop` therefore stays at the last push that changed code, and the `README.md` inside its archives
  and image can lag the repository's.
- **Only `gpillon/ignis` publishes.** A fork that keeps Actions on would publish, under its own name and with
  ignis's notes, a build from a commit ignis never checked. Every publishing job and step starts its condition with
  the repository's name, so a fork keeps the tests and both builds, with the archives as its run's artifacts, and
  publishes nothing; `crates/server/tests/fork_publishing.rs` holds every publishing point to it.
- **The image carries no model and no driver.** A `.ninfer` container is tens of GB, so it is mounted or
  downloaded: `IGNIS_MODEL_ARTIFACT` is not defaulted, and unset the server downloads its default model (ADR 0033).
  The driver is injected by the container runtime, and no binary links an NVIDIA driver library: the driver and NVML
  are opened at run time, so the image runs its no-GPU commands (`model download`, `--help`) on a machine without a
  driver, and with `--download-enabled false` it starts on the CPU mock, the image's smoke test.
- **The release builds compile the kernel leaf for every architecture the release supports**
  (`IGNIS_CUDA_ARCHITECTURES`), on both hosts, where a build from a checkout compiles the default alone. Every
  packaged binary first passes `mk/cuda-build-check.sh`.

## Considered Options

- **Install the CUDA toolkit on the Linux runner and build with cargo directly**, as the Windows job does:
  rejected. A second toolchain to keep in step with the image's, several GB of install per job, and two Linux builds
  per push that can disagree.
- **The image's push as a dependency of the release**: rejected. The image is not part of the release's contract,
  and one flaky registry push would hide a good release.
- **A CPU-mock image**, which would build anywhere: rejected. An inference engine that cannot infer is not a
  release.
- **`cudart_static` on Linux**, so the tarball needs no CUDA runtime: deferred. It diverges from the Windows link,
  which ships the redistributable runtime library instead; the image is the self-contained option.
- **Nothing built before the tests pass**, the earlier order: rejected. A push to `main` took 46 to 64 minutes, most
  of it one wait after another, the test legs and only then the two builds.

## Consequences

- `kernel/build.sh` is the Linux half of `kernel/build.ps1` and must stay in step with it.
- `.cargo/config.toml` pins the MSVC target for every host, since cargo has no host condition for `build.target`,
  so everything that builds on Linux passes `--target` explicitly, and a bare `cargo` command there needs
  `CARGO_BUILD_TARGET`.
