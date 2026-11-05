#!/usr/bin/env bash
# Refuses an ignis-server binary that is not a CUDA build (GitHub #408), or
# a binary that carries no machine code for an architecture its build asked
# for, or the release requires (#482):
#
#   bash mk/cuda-build-check.sh [--require <list>] <binary>...
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
# count.
#
# `--require <list>`, in the same form, names architectures the binary must
# carry whatever list the build ran with: what the release requires, `86;120a`
# (.github/workflows/release.yml), written apart from the build's list, so an
# edit that drops an architecture from the build is refused here rather than
# checked against itself. The build's own list is checked beside it; an empty
# list requires nothing more. A release binary without code for 8.6, or
# without code for 12.0, is refused.
#
# A binary that carries machine code for 8.6 is also held to the full-rate rule
# (spec #503, #512): the RTX 30 and RTX 40 cards run some tensor-core forms at
# half the rate of others, and the kernels the sm_8x prefill work added use only
# the full-rate ones. The kernels the rule covers are listed in
# mk/full-rate-kernels.txt, one place, with #488's exceptions; the check dumps
# the sm_86 SASS with cuobjdump (IGNIS_CUOBJDUMP, else PATH, else
# $CUDA_PATH/bin or $CUDA_HOME/bin) and refuses a covered kernel that holds
# HMMA with an f32 accumulator, any BF16, TF32 or FP8/FP6/FP4 form, any
# warpgroup or tensor-memory MMA, or an MMA form it does not know, or that holds
# no integer MMA where the list requires it. A refusal names the kernel and the
# opcode. A listed kernel found in none of the binaries is refused (a rename
# must not skip the rule), and so is a dump in which the negative control, the
# existing window kernel's f32-accumulate form, shows no forbidden form: a
# check that cannot fail is not a pass. HMMA with an f16 accumulator is allowed
# and reported. The selftest runs all of it on SASS text fixtures.
#
# A binary it cannot read is refused too. Every release build runs it on the
# three binaries it packages, before packaging (the Containerfile's build
# stage, the release workflow's Windows job): the bench and the inspector link
# the leaf's device objects too, and carry no mock to fall back to.
# `make cuda-build-check-selftest` is its test. It reads nothing but its
# arguments, the list and those variables, and needs bash, perl and awk, all in
# Git for Windows and in every Ubuntu image, but no CUDA toolkit (cuobjdump is
# only needed for a binary that carries 8.6, and is faked in the selftest).

set -uo pipefail

# The text of the server's own WARN when a model is loaded without CUDA
# (crates/server/src/load.rs); crates/server/tests/cuda_build_recipes.rs
# holds the two equal.
marker='built without --features cuda'

usage() {
    echo "usage: bash mk/cuda-build-check.sh [--require <list>] <binary>..." >&2
    exit 2
}

required=
if [ "${1-}" = --require ]; then
    [ "$#" -ge 2 ] || usage
    required=$2
    shift 2
fi
[ "$#" -gt 0 ] || usage

# The list as kernel/CMakeLists.txt reads it, and the machine code each entry
# asks for: `120a` is sm_120a, an ELF cubin for architecture 120. Each wanted
# architecture keeps what asked for it first, for the refusal to name.
list=${IGNIS_CUDA_ARCHITECTURES:-120a}
wanted=()
asked=()
want() {
    local entry entries
    IFS=';' read -r -a entries <<< "$1"
    for entry in "${entries[@]}"; do
        if ! [[ "$entry" =~ ^[1-9][0-9]+a?$ ]]; then
            echo "cuda-build-check: $2: \`$entry\` is not an architecture such as 86 or 120a" >&2
            exit 2
        fi
        [[ " ${wanted[*]} " == *" ${entry%a} "* ]] && continue
        wanted+=("${entry%a}")
        asked+=("$2")
    done
}
want "$list" "IGNIS_CUDA_ARCHITECTURES ($list)"
want "$required" "the release's requirement (--require $required)"

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

# The full-rate rule (#503, #512). `full_rate_scan <list> <sass dump>` reads the
# dump of one binary's sm_86 SASS (cuobjdump -sass) and prints, one per line:
# FAIL <why> (the rule refused a kernel), NOTE <what> (allowed, reported), INFO
# <kernel and its MMA opcodes>, SEEN <kernel name> (a listed kernel the dump
# holds), WANT <kernel name> (a covered kernel of the list), CONTROL <name> (the
# negative control showed a forbidden form), WANTCONTROL <name>. A list it
# cannot read is exit status 2.
#
# A function is the kernel's mangled name, `_Z` or `_ZN`; its own name is the
# last length-prefixed identifier of a `_ZN` name (past any namespace), the first
# of a `_Z` name (the rest are its parameters' types), and the template
# arguments are left out, so every instantiation is held to the rule. An
# opcode is what the SASS prints: IMMA is integer; HMMA.<shape> alone is an f16
# accumulator; anything else with MMA in it (HMMA with F32, BF16, TF32, an FP8,
# FP6 or FP4 type; QMMA, OMMA, HGMMA, UTC..., a form the rule has never seen) is
# refused: the safe side is the finite one.
full_rate_scan() {
    awk -v list="$1" '
    function base(fn,   rest, name, n) {
        if (fn !~ /^_Z/) return fn
        name = fn
        rest = substr(fn, 3)
        if (substr(fn, 1, 3) == "_ZN") rest = substr(fn, 4)
        while (match(rest, /^[0-9]+/)) {
            n = substr(rest, 1, RLENGTH) + 0
            name = substr(rest, RLENGTH + 1, n)
            rest = substr(rest, RLENGTH + 1 + n)
            if (substr(fn, 1, 3) != "_ZN") break
        }
        return name
    }
    function kind(op) {
        if (op ~ /^IMMA(\.|$)/) return "int"
        if (op ~ /^HMMA\.[0-9]+(\.F16)?$/) return "f16"
        return "bad"
    }
    BEGIN {
        while ((getline line < list) > 0) {
            sub(/#.*/, "", line)
            n = split(line, f, " ")
            if (n == 0) continue
            if (n < 2 || f[1] !~ /^(mma|cuda|exempt|control)$/ || (f[2] in role)) {
                print "cuda-build-check: " list ": cannot read the line `" line "`" > "/dev/stderr"
                aborted = 1
                continue
            }
            role[f[2]] = f[1]
        }
        close(list)
        if (aborted) exit 2
    }
    /Function : / {
        fn = $3
        cur = (base(fn) in role) ? fn : ""
        if (cur != "" && !(cur in known)) { known[cur] = 1; order[++nfn] = cur }
        next
    }
    cur != "" {
        line = $0
        while (match(line, /[A-Z]*MMA[.A-Za-z0-9_]*/)) {
            op = substr(line, RSTART, RLENGTH)
            line = substr(line, RSTART + RLENGTH)
            if (!((cur, op) in count)) ops[cur] = ops[cur] " " op
            count[cur, op]++
        }
    }
    END {
        if (aborted) exit 2
        for (i = 1; i <= nfn; i++) {
            fn = order[i]; b = base(fn); r = role[b]
            seen[b] = 1
            nops = split(ops[fn], oplist, " "); shown = ""; ints = 0; bad = 0
            for (j = 1; j <= nops; j++) {
                op = oplist[j]
                shown = shown sprintf(" %s x%d", op, count[fn, op])
                k = kind(op)
                if (k == "int") ints++
                if (k == "bad") bad++
                if (k == "bad" && (r == "mma" || r == "cuda"))
                    print "FAIL kernel " b " (" fn ") holds " op ", which is not a full-rate form on an RTX 30 or RTX 40 card"
                if (k == "f16" && (r == "mma" || r == "cuda"))
                    print "NOTE kernel " b " holds " op " (f16 accumulate, allowed)"
            }
            if (r == "mma" && ints == 0)
                print "FAIL kernel " b " (" fn ") holds no integer MMA (IMMA), which a GEMM or band kernel must"
            if (r == "control" && bad > 0) hit[b] = 1
            key = r " " b ":" (shown == "" ? " (no MMA)" : shown)
            if (!(key in times)) infos[++ninfo] = key
            times[key]++
        }
        for (i = 1; i <= ninfo; i++) print "INFO " infos[i] (times[infos[i]] > 1 ? " (" times[infos[i]] " instantiations)" : "")
        for (b in role) {
            if (role[b] == "mma" || role[b] == "cuda") print "WANT " b
            if (role[b] == "control") print "WANTCONTROL " b
            if (b in seen) print "SEEN " b
            if (b in hit) print "CONTROL " b
        }
    }' "$2"
}

# cuobjdump: IGNIS_CUOBJDUMP, else PATH, else the toolkit the build ran with
# (CUDA_PATH arrives in Windows spelling on the Windows host).
find_cuobjdump() {
    local root exe
    if [ -n "${IGNIS_CUOBJDUMP-}" ]; then
        printf '%s\n' "$IGNIS_CUOBJDUMP"
        return 0
    fi
    command -v cuobjdump && return 0
    for root in "${CUDA_PATH-}" "${CUDA_HOME-}"; do
        [ -n "$root" ] || continue
        command -v cygpath > /dev/null 2>&1 && root=$(cygpath -u "$root")
        for exe in cuobjdump cuobjdump.exe; do
            if [ -x "$root/bin/$exe" ]; then
                printf '%s\n' "$root/bin/$exe"
                return 0
            fi
        done
    done
    return 1
}

full_rate_list=${IGNIS_FULL_RATE_LIST:-"$(dirname "$0")/full-rate-kernels.txt"}
sass_dump=
sass_ran=0
all_want=
all_seen=
all_control=
all_wantcontrol=

# Holds one binary's sm_86 machine code to the rule; the whole run's presence
# checks are in the end (a binary that links part of the leaf carries only part
# of the list). Returns 1 when the binary is refused.
full_rate_check() {
    local binary=$1 cuobjdump scan line refused=0
    if [ ! -f "$full_rate_list" ]; then
        echo "cuda-build-check: $full_rate_list: the list of full-rate kernels is missing" >&2
        return 1
    fi
    if ! cuobjdump=$(find_cuobjdump); then
        echo "cuda-build-check: $binary carries 8.6 machine code, which the full-rate rule reads with cuobjdump, and none is on PATH or under \$CUDA_PATH/bin or \$CUDA_HOME/bin" >&2
        return 1
    fi
    if [ -z "$sass_dump" ]; then
        sass_dump=$(mktemp) || return 1
        trap 'rm -f "$sass_dump"' EXIT
    fi
    if ! "$cuobjdump" -sass -arch sm_86 "$binary" > "$sass_dump" 2> /dev/null; then
        echo "cuda-build-check: $binary: cuobjdump could not dump its sm_86 code" >&2
        return 1
    fi
    scan=$(full_rate_scan "$full_rate_list" "$sass_dump")
    case $? in
        0) ;;
        2) return 1 ;;
        *)
            echo "cuda-build-check: $binary: its sm_86 code could not be scanned" >&2
            return 1 ;;
    esac
    sass_ran=1
    while IFS= read -r line; do
        case $line in
            FAIL\ *)
                echo "cuda-build-check: $binary: ${line#FAIL }" >&2
                refused=1 ;;
            NOTE\ *) echo "cuda-build-check: $binary: ${line#NOTE }" ;;
            INFO\ *) echo "cuda-build-check: sm_86 ${line#INFO }" ;;
            WANT\ *) all_want+=" ${line#WANT }" ;;
            SEEN\ *) all_seen+=" ${line#SEEN }" ;;
            CONTROL\ *) all_control+=" ${line#CONTROL }" ;;
            WANTCONTROL\ *) all_wantcontrol+=" ${line#WANTCONTROL }" ;;
        esac
    done <<< "$scan"
    return "$refused"
}

# After the last binary: every covered kernel is in one of them, and the
# negative control, which the rule would refuse, was refused.
full_rate_presence() {
    local name
    [ "$sass_ran" = 1 ] || return 0
    for name in $all_want; do
        [[ " $all_seen " == *" $name "* ]] && continue
        echo "cuda-build-check: kernel $name, which mk/full-rate-kernels.txt holds to the full-rate rule, is in the sm_86 code of none of the binaries; a renamed or removed kernel leaves the list with it" >&2
        status=1
    done
    for name in $all_wantcontrol; do
        [[ " $all_control " == *" $name "* ]] && continue
        echo "cuda-build-check: the negative control $name shows no forbidden MMA form in the sm_86 code, so the full-rate rule could not have refused anything: the dump is not read as it must be" >&2
        status=1
    done
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
    found=$(for arch in $carried; do printf 'sm_%s ' "$arch"; done)
    found=${found% }
    refused=0
    for i in "${!wanted[@]}"; do
        arch=${wanted[$i]}
        grep -qx "$arch" <<< "$carried" && continue
        echo "cuda-build-check: $binary carries no machine code for compute capability $((arch / 10)).$((arch % 10)) (sm_$arch), which ${asked[$i]} asks for; it has code for: ${found:-none}" >&2
        refused=1
    done
    if [ "$refused" = 1 ]; then
        status=1
        continue
    fi
    # 8.6 machine code is held to the full-rate rule (#503, #512).
    if grep -qx 86 <<< "$carried" && ! full_rate_check "$binary"; then
        status=1
        continue
    fi
    echo "cuda-build-check: $binary: CUDA build, machine code for ${found// /, } (IGNIS_CUDA_ARCHITECTURES: $list${required:+; required: $required})"
done
full_rate_presence
exit "$status"
