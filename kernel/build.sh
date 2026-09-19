#!/usr/bin/env bash
# Build the ignis kernel leaf (CMake + Ninja + nvcc, SM120a) on Linux.
#
#   bash kernel/build.sh [build-dir] [--test]
#
# The Linux half of kernel/build.ps1: same generator, same build type, same
# architecture, same canonical kernel/build output directory. There is no
# MSVC env to import here, so the script is only toolchain discovery plus the
# two cmake calls.
#
# --test also runs the leaf's op-test executables through CTest. Those are GPU
# tests and they FAIL (never skip) when the GPU is busy or a kernel errors
# (ADR 0006), so run them on a free GPU: stop the reference runner first.

set -euo pipefail

kernel="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional build-dir override (default: the canonical kernel/build, which
# crates/artifact/build.rs links). A second build dir lets a parallel
# workstream verify new .cu files without contending on the canonical build.
build_dir="build"
run_tests=0
for arg in "$@"; do
    case "$arg" in
        --test) run_tests=1 ;;
        -*) echo "unknown option: $arg" >&2; exit 2 ;;
        *) build_dir="$arg" ;;
    esac
done
case "$build_dir" in
    /*) build_path="$build_dir" ;;
    *) build_path="$kernel/$build_dir" ;;
esac

# CUDA_HOME is the Linux spelling; CUDA_PATH is accepted too so a single
# environment can drive both hosts (crates/artifact/build.rs reads the same
# pair in the same order).
cuda="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
nvcc="$cuda/bin/nvcc"
if [ ! -x "$nvcc" ]; then
    echo "error: nvcc not found at $nvcc (set CUDA_HOME)" >&2
    exit 2
fi

for tool in cmake ninja; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found in PATH" >&2
        exit 2
    fi
done

# The launcher overrides mirror build.ps1: a ccache/sccache configured
# machine-wide must not sit between nvcc and these translation units, because
# a cache miss on a device-code TU is not what the leaf's staleness contract
# (crates/artifact/build.rs) assumes.
cmake -S "$kernel" -B "$build_path" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=120a \
    -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DCMAKE_C_COMPILER_LAUNCHER= \
    -DCMAKE_CXX_COMPILER_LAUNCHER= \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=

# IGNIS_KERNEL_BUILD_JOBS caps the parallel compile width. nvcc on the W4A4 /
# TMA translation units peaks well above 2 GB of RSS, so a small container or
# VM needs a lower width than its core count; unset means "all cores".
cmake --build "$build_path" --parallel ${IGNIS_KERNEL_BUILD_JOBS:+"$IGNIS_KERNEL_BUILD_JOBS"}

if [ "$run_tests" = 1 ]; then
    echo "[ignis-kernel] ctest (GPU: a skip is never green, ADR 0006)"
    ctest --test-dir "$build_path" --output-on-failure
fi
