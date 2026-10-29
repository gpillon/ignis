#!/usr/bin/env bash
# The behavioural test of mk/cuda-build-check.sh (`make
# cuda-build-check-selftest`), which is docs/agents/testing.md's rule applied
# to a script. The check reads only the files it is given, so each case is a
# temp file standing in for a server binary and the check's exit status: it
# builds nothing and reads no real binary. The release workflow's Linux test
# job runs it, since the check decides whether a release is packaged.

set -uo pipefail
check="$(dirname "$0")/cuda-build-check.sh"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0

# A stand-in for each kind of binary: bytes around the text, as in an
# executable.
printf '\177ELF\0\0 ignis.model.mock_compute \0built without --features cuda — MockCompute\0\0' > "$dir/mock"
printf '\177ELF\0\0 ignis.model.loaded \0libcudart.so.13\0\0' > "$dir/cuda"
: > "$dir/empty"
mkdir "$dir/a-directory"

# expect <exit status> <name> <argument>...
expect() {
    want=$1 name=$2
    shift 2
    bash "$check" "$@" > /dev/null 2>&1
    got=$?
    if [ "$got" = "$want" ]; then
        printf 'ok    %s\n' "$name"
    else
        printf 'FAIL  %s: expected exit %s, got %s\n' "$name" "$want" "$got"
        failed=1
    fi
}

expect 0 "a CUDA build passes" "$dir/cuda"
expect 1 "a build carrying the mock fallback is refused" "$dir/mock"
expect 1 "one refused binary among good ones refuses the run" "$dir/cuda" "$dir/mock" "$dir/cuda"
expect 1 "a missing binary is refused" "$dir/missing"
expect 1 "an empty binary is refused" "$dir/empty"
expect 1 "a directory is refused, not read as a pass" "$dir/a-directory"
expect 2 "no argument is a usage error"

exit "$failed"
