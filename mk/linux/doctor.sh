#!/usr/bin/env bash
# The CUDA half of `make doctor` on Linux (mk/os/linux.mk's OS_DOCTOR): what
# a CUDA=1 build reads from the toolkit, checked before the compiler or the
# linker fails on it (GitHub #498). The toolkit is the root named as the
# first argument, else $CUDA_HOME, $CUDA_PATH or /usr/local/cuda, as
# crates/artifact/build.rs takes it; its files are looked for in both
# layouts, a toolkit's own (lib64, include) and an environment-installed
# one's (targets/<arch>-linux/lib and include):
#   - nvcc, the compiler;
#   - libcudart.so, the runtime the link names (-lcudart);
#   - nvml.h and nvtx3/nvToolsExt.h, which the kernel leaf includes
#     (kernel/src/device.cu, the vendored core/nvtx.h) and an incomplete CUDA
#     environment can lack;
#   - cmake, against the minimum kernel/CMakeLists.txt asks for.
# Every line is advice for CUDA=1, so it never fails the doctor. Tested by
# mk/linux/doctor-selftest.sh.

set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cuda="${1:-${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}}"
targets="$cuda/targets/$(uname -m)-linux"

# <what> <first dir> <second dir>: the first dir holding <what>, or the miss.
either() {
    local what="$1" dir
    for dir in "$2" "$3"; do
        if [ -e "$dir/$what" ]; then
            printf '  ok       %s (%s)\n' "$what" "$dir"
            return
        fi
    done
    printf '  optional %s not in %s or %s (CUDA=1 only)\n' "$what" "$2" "$3"
}

echo "cuda:"
if [ -x "$cuda/bin/nvcc" ]; then
    echo "  ok       nvcc ($cuda)"
else
    echo "  optional nvcc not under $cuda (set CUDA_HOME; CUDA=1 only)"
fi
either libcudart.so "$cuda/lib64" "$targets/lib"
either nvml.h "$cuda/include" "$targets/include"
either nvtx3/nvToolsExt.h "$cuda/include" "$targets/include"

echo "cmake:"
minimum="$(sed -n 's/^cmake_minimum_required(VERSION \([0-9.]*\)).*/\1/p' "$repo/kernel/CMakeLists.txt")"
version="$(cmake --version 2>/dev/null | sed -n '1s/^cmake version \([0-9][0-9.]*\).*/\1/p')"
if [ -z "$version" ]; then
    echo "  optional cmake not found ($minimum or newer; CUDA=1 only)"
elif [ "$(printf '%s\n' "$minimum" "$version" | sort -V | head -n 1)" = "$minimum" ]; then
    echo "  ok       cmake $version ($minimum or newer)"
else
    echo "  old      cmake $version, below the $minimum the kernel leaf needs (kernel/CMakeLists.txt; Ubuntu 22.04 packages 3.22; CUDA=1 only)"
fi
exit 0
