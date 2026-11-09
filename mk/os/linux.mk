# Linux hooks. Same contract as mk/os/windows.mk (documented at the top of
# that file). GitHub #167, spec docs/specs/make/01-linux-hooks.md.
#
# Implemented here: the kernel leaf build and its op tests (kernel/build.sh),
# and the GPU status/guard pair (mk/linux/gpu.sh). Those are what a CUDA=1
# build and the container image need.
#
# Background server control uses mk/linux/server.py, with a detached session,
# a PID + process identity, and /ready 200 after warmup (GitHub #129, #493).
# Still to do (each is a hook below that fails with a clear message today):
#   - the run-ui/dev-ui process group
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

SERVER_CTL_ARGS = --pid '$(SERVER_PID)' --log '$(SERVER_LOG)' --url '$(SERVER_URL)' --timeout '$(READY_TIMEOUT)'
SERVER_START = $(PYTHON) mk/linux/server.py start $(SERVER_CTL_ARGS) -- '$(SERVER_BIN)' $(SERVER_FLAGS)
SERVER_STOP = $(PYTHON) mk/linux/server.py stop $(SERVER_CTL_ARGS) $(if $(filter 1,$(FORCE)),--force)
SERVER_STATUS = $(PYTHON) mk/linux/server.py status $(SERVER_CTL_ARGS)
# sh: start the server and vite in one process group, `trap 'kill 0' INT TERM EXIT`.
DEV_UI = $(call todo,run-ui / dev-ui)

# Not a todo, and not OS-specific either: mk/version.sh is text editing over
# four files and serves both hosts, so mk/os/windows.mk names the same one.
# mk/changelog.sh is git and gh, which are the same on either host.
VERSION_TOOL = bash mk/version.sh
CHANGELOG_TOOL = bash mk/changelog.sh

# Debian and Ubuntu name Python 3 `python3`, with no `python` by default.
PYTHON ?= python3

OS_REQUIRED_TOOLS := $(PYTHON)
OS_OPTIONAL_TOOLS := cmake ninja nvidia-smi
OS_TOOL_HINT := CUDA=1 only: CUDA Toolkit under CUDA_HOME, CMake, Ninja -- README Prerequisites
# nvcc, libcudart.so, nvml.h, nvtx3/ in either CUDA layout, and CMake against
# the kernel leaf's minimum (GitHub #498).
OS_DOCTOR = bash mk/linux/doctor.sh;
