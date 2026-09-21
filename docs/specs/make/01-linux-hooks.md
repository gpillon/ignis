# 01 — Linux hooks for the Makefile

GitHub: #167

ADRs: 0027 (make + per-OS hooks), 0006 (one GPU run at a time, the
preflight), 0010 (the vendored kernel leaf).

## Goal

On a Linux host with the CUDA toolkit, every target that works on Windows
works the same way: `make dev`, `make run`/`start`/`stop`/`status`,
`make run-ui`/`dev-ui`, `make gpu-status`/`gpu-guard`/`gpu-profile`,
`make kernel`/`kernel-test`, `make doctor`. The neutral graph (`Makefile`,
`mk/config.mk`) is already OS-free; the work is the hooks in
`mk/os/linux.mk` plus the Windows assumptions below the Makefile.

## Seam

`mk/os/linux.mk` provides the hook contract documented at the top of
`mk/os/windows.mk`. Windows helpers live in `mk/windows/*.ps1`; the Linux
ones go in `mk/linux/*.sh`, POSIX sh.

## Work below the Makefile (Windows-only today)

- `crates/artifact/build.rs` — runs `powershell kernel/build.ps1`, expects
  `ignis_kernel.lib` / `ignis_vendor.lib`, links CUDA from
  `$CUDA_PATH/lib/x64`. Linux: a `kernel/build.sh` (CMake + Ninja + nvcc,
  `CMAKE_CUDA_ARCHITECTURES=120a`, Release, into `kernel/build/`), the
  `lib*.a` names, `$CUDA_PATH/lib64` (default `/usr/local/cuda`), plus
  `stdc++` on the link line.
- `kernel/CMakeLists.txt` — MSVC-specific flags (`/GS-`, the static MSVC
  runtime, the cl.exe `.cpp` workarounds) need GCC/Clang equivalents behind
  a compiler check.
- `.cargo/config.toml` — pins `build.target = x86_64-pc-windows-msvc` for
  every host. The Makefile passes `--target` explicitly, but a plain `cargo
  test` on Linux still targets MSVC; the pin must become Windows-only.
- `crates/core/src/gpu_profile.rs` — reads the preflight marker from
  `std::env::temp_dir()`, which is already portable (`/tmp`).

## Hooks

- `KERNEL_BUILD` / `KERNEL_TEST` — `kernel/build.sh` and `kernel/build.sh
  --test` (CTest).
- `GPU_STATUS` — `nvidia-smi` memory/utilisation plus
  `nvidia-smi --query-compute-apps` (real per-process VRAM on Linux) and
  `pgrep` for `ninfer*`, `ignis-server`, `ignis-bench`, `*_gpu-*`.
- `GPU_GUARD` — a `scripts/gpu-preflight.sh` equivalent of the PowerShell
  preflight (ninfer running, VRAM threshold), plus the ignis-process check;
  it leaves no marker behind, and refuses while a recent marker says a
  profile run is in progress (same rules as `mk/windows/gpu.ps1`).
- `GPU_PROFILE` — `scripts/gpu-profile.sh`, same three stages as
  `gpu-profile.ps1`.
- `SERVER_START` / `SERVER_STOP` / `SERVER_STATUS` — `nohup` + `setsid`,
  pid file and logs in `$(RUNTIME_DIR)`, the `/v1/models` readiness loop,
  a Ctrl+C during the wait stops the server; stop sends SIGTERM (the server
  shuts down gracefully on it), then SIGKILL after a grace period.
- `DEV_UI` — server and `node web/node_modules/vite/bin/vite.js` in one
  process group under `trap 'kill 0' INT TERM EXIT`; either child exiting
  stops the other.
- `OS_REQUIRED_TOOLS` / `OS_OPTIONAL_TOOLS` / `OS_DOCTOR` — cmake, ninja,
  nvcc under `$CUDA_PATH`, nvidia-smi.

## Acceptance criteria

- On Linux, `make dev` builds the GPU server from a clean checkout and serves
  `/v1/models`; `make dev CUDA=0` does the same on the mock.
- `make run` on a stale binary fails naming the changed files, as on Windows.
- `make dev-ui`, then Ctrl+C: no `ignis-server` and no vite process remains.
  The same after `kill -9` of the make process.
- `make start` → `make status` reports it → `make stop` leaves nothing.
- `make gpu-guard` refuses with ninfer or another ignis GPU process running,
  and passes on a free card.
- `cargo test` passes workspace-wide on Linux, and still on Windows.
- `make help` / `make config` unchanged on Windows.
