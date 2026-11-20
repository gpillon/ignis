# Linux hooks. Same contract as mk/os/windows.mk (documented at the top of
# that file). GitHub #167, spec docs/specs/make/01-linux-hooks.md.
#
# Implemented here: the kernel leaf build and its op tests (kernel/build.sh),
# the GPU status/guard pair and the run under the GPU lock (mk/linux/gpu.sh,
# mk/linux/gpulock.py) and the explicit GPU test
# profile (scripts/gpu-profile.sh, the counterpart of scripts/gpu-profile.ps1,
# which writes the marker ignis_core::gpu_profile reads). Those are what a
# CUDA=1 build and the container image need.
#
# Background server control uses mk/linux/server.py, with a detached session,
# a PID + process identity, and /ready 200 after warmup (GitHub #129, #493).
# run-ui/dev-ui's foreground session, mk/linux/devui.py, writes the same
# record with its helpers (GitHub #226).
# Nothing is left to do here: every hook of the contract is implemented.
# The profile's Linux run on a real card is the one thing no CI runner can
# do (docs/user/contributing/releases.md); its selftest
# (mk/linux/gpu-profile-selftest.sh) runs it on stand-ins.
#
# `cargo` on Linux: .cargo/config.toml pins build.target to the MSVC triple
# (cargo has no host-conditional for it), so every cargo call here passes
# --target explicitly -- the neutral Makefile already does, through
# CARGO_TARGET. A bare `cargo test` outside make needs
# CARGO_BUILD_TARGET=x86_64-unknown-linux-gnu in the environment.

EXE :=
TARGET_TRIPLE ?= x86_64-unknown-linux-gnu

KERNEL_BUILD = bash kernel/build.sh
KERNEL_TEST = bash kernel/build.sh build --test
KERNEL_BUILD_INPUTS := $(wildcard kernel/build.sh)

GPU_STATUS = bash mk/linux/gpu.sh status
GPU_GUARD = bash mk/linux/gpu.sh guard $(GPU_THRESHOLD_MIB)
# The sh line in IGNIS_MK_GPU_CMD under the GPU lock (mk/linux/gpulock.py),
# waited for up to GPU_LOCK_WAIT seconds, then the guard.
GPU_RUN = bash mk/linux/gpu.sh run $(GPU_THRESHOLD_MIB) $(GPU_LOCK_WAIT)
GPU_LOCK_SELFTEST = bash mk/linux/gpu-lock-selftest.sh && bash mk/linux/gpu-profile-selftest.sh
# GPU_PROFILE_ARGS are the script's own flags (--skip-kernel-build, ...).
GPU_PROFILE = TARGET_TRIPLE='$(TARGET_TRIPLE)' bash scripts/gpu-profile.sh --wait-seconds $(GPU_LOCK_WAIT) $(GPU_PROFILE_ARGS)

SERVER_CTL_ARGS = --pid '$(SERVER_PID)' --log '$(SERVER_LOG)' --url '$(SERVER_URL)' --timeout '$(READY_TIMEOUT)'
# The guard runs first, from make: on Linux it never touches the marker a GPU
# profile reads, so a lock taken after it races nothing.
SERVER_GUARD = $(if $(GPU_GUARDED),gpu-guard)
# A CUDA server takes the GPU lock before it starts and holds it until it stops.
SERVER_GPU_LOCK = $(if $(GPU_LOCKED),--gpu-lock)
SERVER_START = $(PYTHON) mk/linux/server.py start $(SERVER_CTL_ARGS) $(SERVER_GPU_LOCK) -- '$(SERVER_BIN)' $(SERVER_FLAGS)
SERVER_STOP = $(PYTHON) mk/linux/server.py stop $(SERVER_CTL_ARGS) $(if $(filter 1,$(FORCE)),--force)
SERVER_STATUS = $(PYTHON) mk/linux/server.py status $(SERVER_CTL_ARGS)
# `make run` with CUDA=1: the lock taken for the recipe's shell, which then
# becomes the server (exec keeps the pid and its start time), so the server
# holds it until it exits.
SERVER_RUN = $(PYTHON) mk/linux/gpulock.py acquire --pid $$$$ --wait 0 --command 'make run' && exec env $(SERVER_ENV) '$(SERVER_BIN)' $(SERVER_FLAGS)
# The server and Vite as one foreground session (mk/linux/devui.py's header
# says how each dies with it). `exec` makes the script make's own child, with
# no shell between: it watches its parent, to stop both when make is killed
# by a signal make cannot pass on (kill -9).
DEV_UI = exec $(PYTHON) mk/linux/devui.py $(SERVER_CTL_ARGS) $(SERVER_GPU_LOCK) -- '$(SERVER_BIN)' $(SERVER_FLAGS)

# Not OS-specific: mk/version.sh is text editing over
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
