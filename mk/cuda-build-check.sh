#!/usr/bin/env bash
# Refuses an ignis-server binary that is not a CUDA build (GitHub #408):
#
#   bash mk/cuda-build-check.sh <ignis-server binary>...
#
# A server built without the `cuda` feature still starts, reports ready and
# serves MockCompute's placeholder completions, so nothing downstream of the
# build notices; v0.9.0 and v0.9.1 shipped that way. The check reads the
# binary itself, not the command line that built it: the server's
# `cfg(not(feature = "cuda"))` load path, and only it, carries the warning
# below, so a binary holding it is one that falls back to the mock. Every
# release build runs it before packaging (the Containerfile's build stage, the
# release workflow's Windows job). It reads nothing but its arguments and
# writes nothing.

set -uo pipefail

# The text of the server's own WARN when a model is loaded without CUDA
# (crates/server/src/load.rs); crates/server/tests/cuda_build_recipes.rs
# holds the two equal.
marker='built without --features cuda'

if [ "$#" -eq 0 ]; then
    echo "usage: bash mk/cuda-build-check.sh <ignis-server binary>..." >&2
    exit 2
fi

status=0
for binary in "$@"; do
    if [ ! -s "$binary" ]; then
        echo "cuda-build-check: $binary: missing or empty" >&2
        status=1
    elif grep -aqF "$marker" "$binary"; then
        echo "cuda-build-check: $binary is not a CUDA build: it carries the MockCompute fallback ('$marker')" >&2
        status=1
    else
        echo "cuda-build-check: $binary: CUDA build"
    fi
done
exit "$status"
