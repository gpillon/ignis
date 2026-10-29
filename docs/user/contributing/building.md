---
title: "Building from source"
description: "The kernel leaf, the Rust workspace, the tests and the canary suite."
sidebar:
  order: 2
---

Two parts, in this order: the C++/CUDA kernel leaf, then the Rust workspace that
links it.

**Prerequisites**

- **Rust** — the workspace (Cargo).
- **A C++ toolchain** — MSVC C++ build tools (Visual Studio 2022, C++ workload)
  on Windows; GCC on Linux.
- **The NVIDIA CUDA Toolkit** (`nvcc`), target `SM120a`. Set `CUDA_PATH`
  (`CUDA_HOME` on Linux) if it is not on the default install path.
- **CMake + Ninja** — the kernel leaf builds with the Ninja generator.
- **Node.js**, for the Playground.

## 1. The kernel leaf

`kernel/build.ps1` (Windows) and `kernel/build.sh` (Linux) configure and build
the leaf with CMake + Ninja + `nvcc`, targeting `SM120a`
(`CMAKE_CUDA_ARCHITECTURES=120a`) in a Release build, into `kernel/build/` —
which `crates/*/build.rs` link. The PowerShell script imports the MSVC
environment itself, so no developer prompt is needed. In a shell that already
has one (`cl.exe` on `PATH`) it uses that environment as it is, so a toolchain
the Visual Studio installer does not list builds too.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File kernel\build.ps1
```

A second argument is an alternate build directory (`kernel\build.ps1 build-a`),
so a parallel workstream can verify new `.cu` files without contending on the
canonical `kernel/build/`.

## 2. The Rust workspace

```
# GPU-backed: the real model, needs the kernel leaf built
cargo build --release -p ignis-server --features cuda

# CPU-only mock: protocol and loop work, no GPU, no kernel leaf
cargo build --release -p ignis-server
```

`--features cuda` enables the production GPU backend. Without it, or without an
artifact, the server runs the deterministic CPU mock: the endpoints, streaming,
telemetry and scheduler behaviour are real, the generated text is not, and every
completion reports `finish_reason: "length"` because the mock has no EOS token.

The cargo build reuses an already-built `kernel/build/ignis_kernel.lib` when
present, so it does not recompile the C++ leaf each time. On the MSVC triple the
binary lands in `target/x86_64-pc-windows-msvc/release/`.

Windows is the development host; Linux builds the same engine and is what the
container image is built from. Some `make` targets are still Windows-only there
— background server control (`start` / `stop` / `dev-ui`) and the GPU test
profile; `mk/os/linux.mk` names them.

## Tests

```
cargo test          # workspace-wide, CPU-only and fast — never touches the GPU
```

GPU work is checked by a separate, **explicit** suite (the GPU profile): it
requires the card to be free and **fails** — never skips — when the GPU is busy
or a kernel errors ([ADR 0006](../../adr/0006-exclusive-gpu-testing.md)). The
kernel leaf additionally has its own CTest executable running each vendored op's
reference test at real 27B geometry. The card fits one run at a time: check
`make gpu-status` first.

## Canary check

`ignis-bench canary` runs the fixed high-signal prompt suite against a live
server and checks each output is *sane* and *deterministic* — greedy plus a
fixed seed means an identical output on a repeat run
([ADR 0007](../../adr/0007-performance-gate-not-parity.md)):

```bash
cargo run -p ignis-bench -- canary --endpoint http://127.0.0.1:8000
```
