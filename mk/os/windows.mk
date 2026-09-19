# Windows hooks (implemented). The contract every mk/os/<os>.mk provides:
#
#   EXE                    executable suffix
#   TARGET_TRIPLE          the cargo target triple
#   KERNEL_BUILD           build the kernel leaf
#   KERNEL_TEST            run the kernel leaf's op tests (GPU)
#   KERNEL_BUILD_INPUTS    extra files whose change makes a CUDA server stale
#   GPU_STATUS             print VRAM use and the processes that may hold it
#   GPU_GUARD              exit non-zero while the GPU is held
#   GPU_PROFILE            run the explicit GPU test profile
#   SERVER_START           start $(SERVER_BIN) in the background, wait for ready
#   SERVER_STOP            stop it (FORCE=1: any ignis-server)
#   SERVER_STATUS          report the background server process
#   DEV_UI                 server + Vite as one foreground session; Ctrl+C
#                          (or the session dying) stops both
#   VERSION_TOOL           show / check / set / bump the release version
#                          (mk/version.sh on both hosts -- see below)
#   CHANGELOG_TOOL         draft the release notes for a commit range
#                          (mk/changelog.sh on both hosts -- see below)
#   OS_REQUIRED_TOOLS      tools `make doctor` requires
#   OS_OPTIONAL_TOOLS      tools `make doctor` only reports
#   OS_TOOL_HINT           where those tools come from
#   OS_DOCTOR              extra `make doctor` checks (sh, ending in `;`)

# GNU make on Windows falls back to cmd.exe when it finds no sh.exe on PATH;
# every recipe here is POSIX sh.
ifeq ($(findstring sh,$(notdir $(SHELL))),)
  $(error no POSIX sh found (SHELL=$(SHELL)): install Git for Windows and put its usr\bin on PATH)
endif

EXE := .exe
TARGET_TRIPLE ?= x86_64-pc-windows-msvc

# `exec` is load-bearing. GNU make for Windows (the Chocolatey build is 32-bit)
# runs a metacharacter-free command line itself, and a 32-bit parent resolves
# `powershell` to the SysWOW64 copy: a 32-bit PowerShell that cannot see
# System32\nvidia-smi.exe, so every GPU check fails. A shell builtin makes make
# hand the line to sh (64-bit), which starts the real 64-bit PowerShell. Since
# `exec` replaces the shell, a $(PS) hook must be the last command on its line.
PS := exec powershell -NoProfile -ExecutionPolicy Bypass

KERNEL_BUILD = $(PS) -File kernel/build.ps1
KERNEL_TEST = $(PS) -File kernel/build.ps1 build -Test
KERNEL_BUILD_INPUTS := $(wildcard kernel/build.ps1)

GPU_STATUS = $(PS) -File mk/windows/gpu.ps1 -Action status
GPU_GUARD = $(PS) -File mk/windows/gpu.ps1 -Action guard -ThresholdMiB $(GPU_THRESHOLD_MIB)
GPU_PROFILE = $(PS) -File scripts/gpu-profile.ps1 $(GPU_PROFILE_ARGS)

# server.ps1 reads its inputs from the environment: the flag string passes
# through sh and PowerShell untouched that way.
SERVER_CTL_ENV = IGNIS_MK_BIN='$(SERVER_BIN)' IGNIS_MK_ARGS='$(strip $(SERVER_FLAGS))' \
  IGNIS_MK_PID='$(SERVER_PID)' IGNIS_MK_LOG='$(SERVER_LOG)' IGNIS_MK_URL='$(SERVER_URL)' \
  IGNIS_MK_READY_TIMEOUT='$(READY_TIMEOUT)' IGNIS_MK_FORCE='$(FORCE)'
SERVER_START = $(SERVER_CTL_ENV) $(PS) -File mk/windows/server.ps1 -Action start
SERVER_STOP = $(SERVER_CTL_ENV) $(PS) -File mk/windows/server.ps1 -Action stop
SERVER_STATUS = $(SERVER_CTL_ENV) $(PS) -File mk/windows/server.ps1 -Action status
DEV_UI = $(SERVER_CTL_ENV) $(PS) -File mk/windows/devui.ps1

# The two hooks here that are not PowerShell: mk/version.sh is text editing
# over four files and mk/changelog.sh is git and gh, and the sh this file
# already requires (the $(error) above) is Git for Windows', which ships bash.
VERSION_TOOL = bash mk/version.sh
CHANGELOG_TOOL = bash mk/changelog.sh

OS_REQUIRED_TOOLS := powershell
OS_OPTIONAL_TOOLS := cmake ninja nvidia-smi
OS_TOOL_HINT := CUDA=1 only: VS 2022 C++ tools, CUDA Toolkit, CMake, Ninja -- README Prerequisites
OS_DOCTOR = echo "cuda:"; \
  cuda="$${CUDA_PATH:-C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1}"; \
  if [ -f "$$cuda/bin/nvcc.exe" ]; then echo "  ok       nvcc ($$cuda)"; \
  else echo "  optional nvcc not under $$cuda (set CUDA_PATH; CUDA=1 only)"; fi;
