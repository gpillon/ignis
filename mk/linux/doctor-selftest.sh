#!/usr/bin/env bash
# The behavioural test of mk/linux/doctor.sh (`make doctor-selftest`), the
# CUDA half of `make doctor` on Linux (GitHub #498). It builds fake CUDA
# roots in both layouts (a toolkit's lib64, an environment's
# targets/<arch>-linux) and a fake cmake named by $CMAKE, and reads what the report
# says of each. It needs no CUDA toolkit and no GPU, and runs under any bash.
# It writes only under a temporary directory it removes.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

failed=0
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
arch="$(uname -m)"

# A fake cmake that says it is version $1; none at all for an empty $1.
bin="$tmp/bin"
with_cmake() {
    rm -rf "$bin"
    mkdir -p "$bin"
    if [ -n "$1" ]; then
        printf '#!/usr/bin/env bash\necho "cmake version %s"\necho\necho "CMake suite maintained and supported by Kitware (kitware.com/cmake)."\n' "$1" > "$bin/cmake"
        chmod +x "$bin/cmake"
    fi
}
# The report on root $1, with the fake cmake as $CMAKE: a path that is not
# there when there is none, so a cmake installed on the machine is not read.
report() { CMAKE="$bin/cmake" bash mk/linux/doctor.sh "$1" 2>&1; }

has() { # <label> <report> <line prefix>
    printf '%s\n' "$2" | grep -qF -- "$3" || fail "$1: no line with '$3' in:
$2"
}
# An executable stand-in at $1 (a script, which Git Bash's -x needs as well).
stand_in() { printf '#!/bin/sh\n' > "$1"; chmod +x "$1"; }

# A toolkit's layout: the libraries under lib64, the headers under include.
toolkit="$tmp/toolkit"
mkdir -p "$toolkit/bin" "$toolkit/lib64" "$toolkit/include/nvtx3"
stand_in "$toolkit/bin/nvcc"
: > "$toolkit/lib64/libcudart.so"
: > "$toolkit/include/nvml.h"
: > "$toolkit/include/nvtx3/nvToolsExt.h"
with_cmake 3.28.3
out="$(report "$toolkit")"
has "toolkit" "$out" "  ok       nvcc ($toolkit)"
has "toolkit" "$out" "  ok       libcudart.so ($toolkit/lib64)"
has "toolkit" "$out" "  ok       nvml.h ($toolkit/include)"
has "toolkit" "$out" "  ok       nvtx3/nvToolsExt.h ($toolkit/include)"
has "toolkit" "$out" "  ok       cmake 3.28.3"

# An environment's layout: no lib64, everything under targets/<arch>-linux.
env_root="$tmp/env"
targets="$env_root/targets/$arch-linux"
mkdir -p "$env_root/bin" "$targets/lib" "$targets/include/nvtx3"
stand_in "$env_root/bin/nvcc"
: > "$targets/lib/libcudart.so"
: > "$targets/include/nvml.h"
: > "$targets/include/nvtx3/nvToolsExt.h"
out="$(report "$env_root")"
has "environment" "$out" "  ok       libcudart.so ($targets/lib)"
has "environment" "$out" "  ok       nvml.h ($targets/include)"
has "environment" "$out" "  ok       nvtx3/nvToolsExt.h ($targets/include)"

# An incomplete one: nvcc alone. Each missing piece names where it was looked for.
bare="$tmp/bare"
mkdir -p "$bare/bin"
stand_in "$bare/bin/nvcc"
out="$(report "$bare")"
has "bare" "$out" "  optional libcudart.so not in $bare/lib64 or $bare/targets/$arch-linux/lib"
has "bare" "$out" "  optional nvml.h not in $bare/include or $bare/targets/$arch-linux/include"
has "bare" "$out" "  optional nvtx3/nvToolsExt.h not in $bare/include or $bare/targets/$arch-linux/include"
has "no toolkit" "$(report "$tmp/nowhere")" "  optional nvcc not under $tmp/nowhere"
# A directory under a file's name is not that file.
mkdir -p "$bare/lib64/libcudart.so"
has "a directory named libcudart.so" "$(report "$bare")" "  optional libcudart.so not in $bare/lib64"

# CMake against kernel/CMakeLists.txt's minimum, whatever it is.
minimum="$(sed -n 's/^cmake_minimum_required(VERSION \([0-9.]*\)).*/\1/p' kernel/CMakeLists.txt)"
[ "$minimum" = 3.28 ] || fail "kernel/CMakeLists.txt's minimum is $minimum: update this test's versions"
with_cmake 3.22.1
has "cmake 3.22.1" "$(report "$toolkit")" "  old      cmake 3.22.1, below the 3.28 the kernel leaf needs"
with_cmake 3.28.0
has "cmake 3.28.0" "$(report "$toolkit")" "  ok       cmake 3.28.0"
with_cmake 4.1.2
has "cmake 4.1.2" "$(report "$toolkit")" "  ok       cmake 4.1.2"
with_cmake 3.9.6
has "cmake 3.9.6" "$(report "$toolkit")" "  old      cmake 3.9.6"
with_cmake ""
has "no cmake" "$(report "$toolkit")" "  optional cmake not found"

# The report never fails the doctor: every line is advice for CUDA=1.
with_cmake 3.22.1
report "$bare" >/dev/null || fail "the report exited non-zero"

if [ "$failed" -ne 0 ]; then
    printf 'doctor-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "doctor-selftest: ok"
