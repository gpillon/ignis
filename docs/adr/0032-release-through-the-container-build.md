# ADR 0032 — the Linux release is the container build, and the image ships on its own

## Status

Accepted (2026-09-19, owner decision — GitHub #225).

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

**Two workflows, not one.** `release.yml` builds the Windows and Linux
binaries and cuts the GitHub Release; `image.yml` builds and publishes the
image. They share one buildx cache scope (`linux-cuda`), so the kernel is
compiled once per push rather than once per workflow, but they fail
independently: a registry outage never withholds a release, and a broken
Windows build never withholds the image.

**The Release waits for both hosts, and for nothing else.** `release` needs
`linux` and `windows`; the image is deliberately not a dependency.

**A tag is only cut on a pipeline already green.** Every push to `main` or a
`ci/**` branch runs both workflows and publishes nothing — that is the dry
run. A `v*` tag publishes, and its version must equal
`workspace.package.version` (and `web/package.json`'s), or the run fails
before building anything. A tag's run is never cancelled by the concurrency
group.

**Nothing is run on a runner.** No runner has an NVIDIA GPU. The kernel is
compiled for SM120a and linked against the toolkit's driver stubs
(`lib64/stubs`), which carry the real SONAMEs, so the binary loads the host's
own `libcuda.so.1` and `libnvidia-ml.so.1` at run time. Correctness stays on
the machine with the 5090 (`docs/agents/testing.md`).

**The image carries no model and no driver.** A `.ninfer` container is tens of
GB, so it is mounted; the driver is injected by the container runtime.
`IGNIS_ARTIFACT` is deliberately not defaulted — unset, the server starts on
the CPU mock (ADR 0006), which makes running the image a smoke test on its
own.

## Considered Options

- **Install the CUDA toolkit on the Linux runner** and build with cargo
  directly, as the Windows job does — rejected: a second toolchain to keep in
  step with the image's, several GB of install per job, and two Linux builds
  per push that can disagree.
- **One workflow with the image as a third job** — rejected on the owner's
  call: the image is not part of the Release's contract, and coupling their
  failures means one flaky registry push hides a good release.
- **Make the image a dependency of the Release** — rejected for the same
  reason, in the other direction.
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
  `CARGO_BUILD_TARGET`. GitHub #226.
- The image is SM120a-only, like every ignis build. It will not run on another
  card.
- `make` on Linux covers the kernel and GPU hooks; background server control
  and the GPU profile are still Windows-only (GitHub #226).
