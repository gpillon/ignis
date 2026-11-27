---
status: accepted
---

# make is the development entry point, with per-OS hooks

The root `Makefile` is the one entry point for building, running and checking ignis: GNU make with POSIX sh recipes
on every OS (Git for Windows' `sh.exe` on Windows). The target graph is platform-neutral, and everything that
differs per OS (the target triple, the kernel leaf build, the GPU checks, background server control) sits behind
hook variables in `mk/os/<os>.mk`.

Three behaviours are deliberate:

- **`run` and `start` launch, they never build.** A missing or stale binary fails naming the changed files, and so
  does a binary built for the other backend (`CUDA=0` against `1`); `dev` is build-then-run. A GPU build rebuilds
  the kernel leaf and can take minutes, and starting the server should never do that implicitly.
- **A foreground session ends whole on Ctrl+C.** `run-ui` and `dev-ui` hold the server and Vite in one console, so
  the interrupt reaches both, and a killed session takes both down too. `start` is the single daemon: it outlives
  the terminal until `make stop`.
- **A CUDA `run` or `start` runs the GPU guard first**: the preflight plus a check for other ignis GPU work, leaving
  no preflight marker behind.

## Considered Options

- **PowerShell scripts only**: rejected. No dependency graph, so no "stale, rebuild first", and nothing to carry to
  Linux.
- **`just` or `cargo xtask`**: rejected. One more tool to install, and the flow crosses cargo, npm, CMake and
  PowerShell rather than living in Rust.

## Consequences

- `make`'s knobs are a second place engine defaults live; the server's own `--help` stays the source of truth for
  what a flag means.
