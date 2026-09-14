# Linux hooks (scaffolded, not implemented yet: GitHub #167, spec
# .scratch/make/specs/01-linux-hooks.md). Same contract as mk/os/windows.mk. What already works on Linux through the neutral Makefile:
# the CPU mock (`make mock`, `make build CUDA=0`, `make run CUDA=0`), the web
# targets, `make test` / `check` / `clippy`, `make smoke`, `make canary`.
#
# Still to do before CUDA=1 works here -- each is a hook below that fails
# with a clear message today:
#   - kernel/build.sh (the CMake + Ninja + nvcc build of kernel/build.ps1),
#     and crates/artifact/build.rs calling it and linking lib*.a + libcudart
#     instead of powershell + *.lib
#   - a GPU guard/status equivalent of scripts/gpu-preflight.ps1 (nvidia-smi
#     plus pgrep for ninfer*, ignis-server, ignis-bench, *_gpu-*), writing the
#     marker ignis_core::gpu_profile reads, and a gpu-profile.sh
#   - background server control: pid file + nohup + a /v1/models readiness
#     loop (the Windows version is mk/windows/server.ps1)

EXE :=
TARGET_TRIPLE ?= x86_64-unknown-linux-gnu

# $(call todo,<what>): a recipe line that fails with a pointer to this file.
todo = @echo "error: $1 is not implemented on Linux yet (see mk/os/linux.mk)" >&2; exit 1

KERNEL_BUILD = $(call todo,the kernel leaf build)
KERNEL_TEST = $(call todo,the kernel leaf op tests)
KERNEL_BUILD_INPUTS := $(wildcard kernel/build.sh)

GPU_STATUS = $(call todo,gpu-status)
GPU_GUARD = $(call todo,the GPU guard)
GPU_PROFILE = $(call todo,the GPU profile)

SERVER_START = $(call todo,background start)
SERVER_STOP = $(call todo,background stop)
SERVER_STATUS = $(call todo,background status)
# sh: start the server and vite in one process group, `trap 'kill 0' INT TERM EXIT`.
DEV_UI = $(call todo,run-ui / dev-ui)

OS_REQUIRED_TOOLS :=
OS_OPTIONAL_TOOLS := cmake ninja nvcc nvidia-smi
OS_TOOL_HINT := CUDA=1 only, not wired on Linux yet
OS_DOCTOR =
