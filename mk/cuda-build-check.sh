#!/usr/bin/env bash
# Refuses an ignis-server binary that is not a CUDA build (GitHub #408), or
# a binary that carries no machine code for an architecture its build asked
# for (#482):
#
#   bash mk/cuda-build-check.sh <binary>...
#
# A server built without the `cuda` feature still starts, reports ready and
# serves MockCompute's placeholder completions, so nothing downstream of the
# build notices; v0.9.0 and v0.9.1 shipped that way. The check reads the
# binary itself, not the command line that built it: the server's
# `cfg(not(feature = "cuda"))` load path, and only it, carries the warning
# below, so a binary holding it is one that falls back to the mock.
#
# The architectures are the list kernel/CMakeLists.txt compiles the leaf for,
# IGNIS_CUDA_ARCHITECTURES, read from the same environment the build ran in:
# unset or empty, 120a alone, the leaf's own default. Each one must be there
# as machine code, an ELF cubin in one of the binary's fatbin containers (the
# blocks nvcc embeds, one per object), counted across the whole binary: the
# sm_120a-only objects carry no other architecture in any build. PTX does not
# count. The releases ask for `86;120a` (.github/workflows/release.yml), so a
# release binary without code for 8.6, or without code for 12.0, is refused.
#
# A binary it cannot read is refused too. Every release build runs it on the
# three binaries it packages, before packaging (the Containerfile's build
# stage, the release workflow's Windows job): the bench and the inspector link
# the leaf's device objects too, and carry no mock to fall back to.
# `make cuda-build-check-selftest` is its test. It reads nothing but its
# arguments and that variable, writes nothing, and needs bash and perl, both
# in Git for Windows and in every Ubuntu image, but no CUDA toolkit.

set -uo pipefail

# The text of the server's own WARN when a model is loaded without CUDA
# (crates/server/src/load.rs); crates/server/tests/cuda_build_recipes.rs
# holds the two equal.
marker='built without --features cuda'

if [ "$#" -eq 0 ]; then
    echo "usage: bash mk/cuda-build-check.sh <binary>..." >&2
    exit 2
fi

# The list as kernel/CMakeLists.txt reads it, and the machine code each entry
# asks for: `120a` is sm_120a, an ELF cubin for architecture 120.
list=${IGNIS_CUDA_ARCHITECTURES:-120a}
wanted=()
IFS=';' read -r -a entries <<< "$list"
for entry in "${entries[@]}"; do
    if ! [[ "$entry" =~ ^[1-9][0-9]+a?$ ]]; then
        echo "cuda-build-check: IGNIS_CUDA_ARCHITECTURES ($list): \`$entry\` is not an architecture such as 86 or 120a" >&2
        exit 2
    fi
    wanted+=("${entry%a}")
done

# Every architecture a binary carries machine code for, one number per line.
# A fatbin container is the magic 0xBA55ED50, version 1, a 16-byte header and
# the size of the entries that follow; each entry has a 16-bit kind (2 is an
# ELF cubin, 1 is PTX) at byte 0, its header size at byte 4, its payload size
# at byte 8 and its architecture at byte 28. A container whose entries do not
# walk to its exact end is not one, and counts for nothing.
machine_code() {
    perl -e '
        binmode STDIN;
        local $/;
        my $bytes = <STDIN>;
        my $length = length $bytes;
        my %found;
        while ($bytes =~ /\x50\xed\x55\xba\x01\x00\x10\x00/g) {
            my $at = $-[0];
            last if $at + 16 > $length;
            my ($low, $high) = unpack("V V", substr($bytes, $at + 8, 8));
            my $offset = $at + 16;
            my $end = $offset + $low + $high * 4294967296;
            next if $end > $length;
            my @elf;
            while ($offset + 32 <= $end) {
                my ($kind, $header) = unpack("v x2 V", substr($bytes, $offset, 8));
                my ($size_low, $size_high) = unpack("V V", substr($bytes, $offset + 8, 8));
                last if ($kind != 1 && $kind != 2) || $header < 32;
                push @elf, unpack("V", substr($bytes, $offset + 28, 4)) if $kind == 2;
                $offset += $header + $size_low + $size_high * 4294967296;
            }
            next unless $offset == $end;
            $found{$_} = 1 for @elf;
        }
        print "$_\n" for sort { $a <=> $b } keys %found;
    ' < "$1"
}

status=0
for binary in "$@"; do
    if [ ! -f "$binary" ] || [ ! -s "$binary" ]; then
        echo "cuda-build-check: $binary: missing, empty or not a file" >&2
        status=1
        continue
    fi
    grep -aqF "$marker" "$binary"
    case $? in
        0)
            echo "cuda-build-check: $binary is not a CUDA build: it carries the MockCompute fallback ('$marker')" >&2
            status=1
            continue ;;
        1) ;;
        *)
            echo "cuda-build-check: $binary: could not be read" >&2
            status=1
            continue ;;
    esac
    if ! carried=$(machine_code "$binary"); then
        echo "cuda-build-check: $binary: its machine code could not be read" >&2
        status=1
        continue
    fi
    missing=()
    for arch in "${wanted[@]}"; do
        grep -qx "$arch" <<< "$carried" || missing+=("$arch")
    done
    found=$(for arch in $carried; do printf 'sm_%s ' "$arch"; done)
    found=${found% }
    if [ "${#missing[@]}" -gt 0 ]; then
        for arch in "${missing[@]}"; do
            echo "cuda-build-check: $binary carries no machine code for compute capability $((arch / 10)).$((arch % 10)) (sm_$arch), which IGNIS_CUDA_ARCHITECTURES ($list) asks for; it has code for: ${found:-none}" >&2
        done
        status=1
        continue
    fi
    echo "cuda-build-check: $binary: CUDA build, machine code for ${found// /, } (IGNIS_CUDA_ARCHITECTURES: $list)"
done
exit "$status"
