#!/usr/bin/env bash
# The behavioural test of mk/cuda-build-check.sh (`make
# cuda-build-check-selftest`), which is docs/agents/testing.md's rule applied
# to a script. The check reads only the files it is given, so each case is a
# temp file standing in for a server binary and the check's exit status: it
# builds nothing, reads no real binary and needs no CUDA toolkit. The release
# workflow's test job runs it on both hosts, since the check decides whether a
# release is packaged.

set -uo pipefail
check="$(dirname "$0")/cuda-build-check.sh"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0
# Each case names the list it asks for; the caller's must not leak in.
unset IGNIS_CUDA_ARCHITECTURES

# Little-endian integers, written as raw bytes.
byte() { printf "\\$(printf '%03o' "$1")"; }
le16() { byte $(($1 & 255)); byte $((($1 >> 8) & 255)); }
le32() { le16 $(($1 & 65535)); le16 $((($1 >> 16) & 65535)); }
le64() { le32 "$1"; le32 0; }

# One entry of a fatbin container, laid out as nvcc writes it: a 64-byte
# header (the kind, 1 for PTX and 2 for an ELF cubin, the machine code; the
# CUDA architecture as a number at byte 28, 86 for sm_86 and 120 for sm_120a)
# and an 8-byte payload.
entry() {
    le16 "$1"; le16 0x0101; le32 64; le64 8
    le32 0; le32 64; le32 0x10008; le32 "$2"
    for _ in 1 2 3 4 5 6 7 8; do le32 0; done
    printf 'payload!'
}
# A fatbin container: its header (the magic, version 1, header size 16, the
# entries' size), then the entries, each named <kind>:<arch>.
fatbin() {
    local spec
    printf '\120\355\125\272'; le16 1; le16 16; le64 $(($# * 72))
    for spec in "$@"; do entry "${spec%%:*}" "${spec##*:}"; done
}
elf=2 ptx=1
# A stand-in for a CUDA server binary: bytes around one fatbin per object, as
# in an executable. Each argument is one object's entries. The sm_120a-only
# objects carry no sm_86 code even in an `86;120a` build, so the check counts
# machine code across the whole binary, never per object.
server() {
    local object
    printf '\177ELF\0\0 ignis.model.loaded \0libcudart.so.13\0\0'
    for object in "$@"; do
        # shellcheck disable=SC2086 # one word per entry
        fatbin $object
        printf '\0\0\0\0'
    done
}

server "$elf:86 $ptx:86 $elf:120 $ptx:120" "$elf:120 $ptx:120" > "$dir/both"
server "$elf:120 $ptx:120" "$elf:120 $ptx:120" > "$dir/no-86"
server "$elf:86 $ptx:86 $ptx:120" > "$dir/no-120"
# PTX is not machine code: the driver would have to compile it on the card.
server "$ptx:86 $elf:120 $ptx:120" > "$dir/ptx-86"
# Bytes that start like a fatbin but do not walk as one (an entry of kind 9)
# carry nothing, whatever architecture they name.
{
    server "$elf:120"
    printf '\120\355\125\272'; le16 1; le16 16; le64 72
    le16 9; le16 0x0101; le32 64; le64 8; le32 0; le32 64; le32 0x10008; le32 86
} > "$dir/broken-86"
printf '\177ELF\0\0 ignis.model.mock_compute \0built without --features cuda — MockCompute\0\0' > "$dir/mock"
: > "$dir/empty"
mkdir "$dir/a-directory"

# expect <exit status> <IGNIS_CUDA_ARCHITECTURES, - for unset> <name> <argument>...
expect() {
    local want=$1 list=$2 name=$3 got
    shift 3
    if [ "$list" = - ]; then
        bash "$check" "$@" > /dev/null 2>&1
    else
        IGNIS_CUDA_ARCHITECTURES="$list" bash "$check" "$@" > /dev/null 2>&1
    fi
    got=$?
    if [ "$got" = "$want" ]; then
        printf 'ok    %s\n' "$name"
    else
        printf 'FAIL  %s: expected exit %s, got %s\n' "$name" "$want" "$got"
        failed=1
    fi
}

# What a release asks for: machine code for 8.6 and for 12.0.
expect 0 '86;120a' "a build with machine code for both passes" "$dir/both"
expect 1 '86;120a' "a build missing the 8.6 machine code is refused" "$dir/no-86"
expect 1 '86;120a' "a build missing the 12.0 machine code is refused" "$dir/no-120"
expect 1 '86;120a' "a build carrying the mock fallback is refused" "$dir/mock"
expect 1 '86;120a' "PTX for 8.6 does not stand in for its machine code" "$dir/ptx-86"
expect 1 '86;120a' "bytes that do not walk as a fatbin carry no machine code" "$dir/broken-86"
# A build from a checkout: unset or empty is the leaf's own default, 120a.
expect 0 - "the default list asks for 12.0 alone" "$dir/no-86"
expect 0 '' "an empty list is the default" "$dir/no-86"
expect 1 - "the default list refuses a build missing 12.0" "$dir/no-120"
expect 1 - "the default list refuses the mock" "$dir/mock"
# Any other list is read the same way.
expect 1 '89;120a' "a list naming 8.9 refuses a build carrying 8.6 alone" "$dir/both"
expect 2 '86-real;120a' "a list kernel/CMakeLists.txt refuses is a usage error" "$dir/both"
# The release's own requirement, --require, holds whatever list the build
# was given: a list that shrank does not shrink the check with it.
expect 0 '86;120a' "the release's requirement passes a build with both" --require '86;120a' "$dir/both"
expect 1 '120a' "the release's requirement refuses a build missing 8.6, whatever the build's list" --require '86;120a' "$dir/no-86"
expect 1 - "the release's requirement refuses a build missing 8.6 with no list set" --require '86;120a' "$dir/no-86"
expect 1 '86' "the release's requirement refuses a build missing 12.0, whatever the build's list" --require '86;120a' "$dir/no-120"
expect 1 '86;120a' "the release's requirement lets no mock through" --require '86;120a' "$dir/mock"
expect 1 '86;120a' "the build's own list is still checked beside the requirement" --require '120a' "$dir/no-86"
expect 0 - "an empty requirement adds nothing to the build's list" --require '' "$dir/no-86"
expect 2 - "a requirement kernel/CMakeLists.txt would refuse is a usage error" --require '86-real;120a' "$dir/both"
expect 2 - "--require with no list is a usage error" --require
# The binaries themselves.
expect 1 '86;120a' "one refused binary among good ones refuses the run" "$dir/both" "$dir/no-86" "$dir/both"
expect 1 - "a missing binary is refused" "$dir/missing"
expect 1 - "an empty binary is refused" "$dir/empty"
expect 1 - "a directory is refused, not read as a pass" "$dir/a-directory"
expect 2 - "no argument is a usage error"

exit "$failed"
