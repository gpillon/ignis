# Linux hooks. Same contract as mk/os/windows.mk (documented at the top of
# that file). GitHub #167, spec docs/specs/make/01-linux-hooks.md.
#
# Implemented here: the kernel leaf build and its op tests (kernel/build.sh),
# and the GPU status/guard pair (mk/linux/gpu.sh). Those are what a CUDA=1
# build and the container image need.
#
# Still to do (each is a hook below that fails with a clear message today):
#   - background server control: pid file + setsid/nohup and a /v1/models
#     readiness loop (the Windows version is mk/windows/server.ps1), and the
#     run-ui/dev-ui process group on top of it
#   - scripts/gpu-profile.sh, the three-stage explicit GPU test profile
#     (scripts/gpu-profile.ps1's counterpart), which writes the marker
#     ignis_core::gpu_profile reads
#
# `cargo` on Linux: .cargo/config.toml pins build.target to the MSVC triple
# (cargo has no host-conditional for it), so every cargo call here passes
# --target explicitly -- the neutral Makefile already does, through
# CARGO_TARGET. A bare `cargo test` outside make needs
# CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu in the environment.

EXE :=
TARGET_TRIPLE ?= x86_64-unknown-linux-gnu

# $(call todo,<what>): a recipe line that fails with a pointer to this file.
todo = @echo "error: $1 is not implemented on Linux yet (see mk/os/linux.mk)" >&2; exit 1

KERNEL_BUILD = bash kernel/build.sh
KERNEL_TEST = bash kernel/build.sh build --test
KERNEL_BUILD_INPUTS := $(wildcard kernel/build.sh)

GPU_STATUS = bash mk/linux/gpu.sh status
GPU_GUARD = bash mk/linux/gpu.sh guard $(GPU_THRESHOLD_MIB)
GPU_PROFILE = $(call todo,the GPU profile)

SERVER_START = $(call todo,background start)
SERVER_STOP = $(call todo,background stop)
SERVER_STATUS = $(call todo,background status)
# sh: start the server and vite in one process group, `trap 'kill 0' INT TERM EXIT`.
DEV_UI = $(call todo,run-ui / dev-ui)

# Not a todo, and not OS-specific either: mk/version.sh is text editing over
# four files and serves both hosts, so mk/os/windows.mk names the same one.
# mk/changelog.sh is git and gh, which are the same on either host.
VERSION_TOOL = bash mk/version.sh
CHANGELOG_TOOL = bash mk/changelog.sh

OS_REQUIRED_TOOLS :=
OS_OPTIONAL_TOOLS := cmake ninja nvidia-smi
OS_TOOL_HINT := CUDA=1 only: CUDA Toolkit under CUDA_HOME, CMake, Ninja -- README Prerequisites
OS_DOCTOR = echo "cuda:"; \
  cuda="$${CUDA_HOME:-$${CUDA_PATH:-/usr/local/cuda}}"; \
  if [ -x "$$cuda/bin/nvcc" ]; then echo "  ok       nvcc ($$cuda)"; \
  else echo "  optional nvcc not under $$cuda (set CUDA_HOME; CUDA=1 only)"; fi;
