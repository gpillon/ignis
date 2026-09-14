# ADR 0027 — make is the development entry point, with per-OS hooks

## Status

Accepted (2026-09-14, owner decision).

## Decision

The root `Makefile` is the one entry point for building, running and checking
ignis. It is GNU make with **POSIX sh recipes on every OS** (Git for Windows'
`sh.exe` on Windows). The target graph is platform-neutral; everything that
differs per OS — the target triple, the kernel leaf build, the GPU checks,
background server control — sits behind hook variables in `mk/os/<os>.mk`.
Windows implements them (PowerShell helpers in `mk/windows/`); Linux is
scaffolded, and each hook it cannot run yet fails naming itself.

Four behaviours are deliberate:

- **`run` and `start` launch, they never build.** A missing or stale binary
  (a server source, a kernel source under `CUDA=1`, or `web/dist` newer than
  it) fails naming the changed files; so does a binary built for the other
  backend (`CUDA=0` vs `1`). `dev` is build-then-run.
- **A foreground session ends whole on Ctrl+C.** `run-ui`/`dev-ui` hold the
  server and Vite in one console, so the interrupt reaches both, inside a
  kill-on-close job object, so a killed session takes them down too.
  `start` is the single daemon: it outlives the terminal until `make stop`.
- **The GPU defaults are the G5 gate configuration** (262144-token context,
  hq-e8-2b, `--spec dflash2 --draft-tokens 7`), so a hand-run server matches
  what the gates measure; each is a knob (`SPEC=` turns speculation off).
- **A CUDA `run`/`start` runs the GPU guard first**: the preflight plus a check
  for other ignis GPU work, leaving no preflight marker behind.

## Considered Options

- **PowerShell scripts only** (`scripts/*.ps1`, as the GPU runbook had) —
  rejected: no dependency graph, so no "stale, rebuild first", and nothing to
  carry to Linux.
- **`just` / `cargo xtask`** — rejected: one more tool to install, and the
  flow crosses cargo, npm, CMake and PowerShell rather than living in Rust.
- **`run` builds when stale** — rejected: a GPU build rebuilds the kernel leaf
  and can take minutes; starting the server should never do that implicitly.

## Consequences

- GNU make for Windows (the Chocolatey build is 32-bit) runs a metacharacter-
  free line itself, and a 32-bit parent resolves `powershell` to the
  SysWOW64 copy, which cannot see `nvidia-smi`. Every PowerShell hook is
  therefore `exec powershell …`, which forces the line through sh. Do not
  drop the `exec`.
- Vite is started as `node vite.js`, never through `npm`: a batch file answers
  Ctrl+C with "Terminate batch job (Y/N)?" and would keep the session open.
- `make`'s knobs are a second place engine defaults live; the server's own
  `--help` stays the source of truth for what a flag means.
