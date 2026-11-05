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


# The full-rate rule reads the sm_86 SASS of a binary that carries 8.6, which
# cuobjdump prints. The selftest has no CUDA toolkit, so IGNIS_CUOBJDUMP names a
# stand-in that prints the text file $FAKE_SASS (with no file it fails);
# the dumps are text, written here in the form cuobjdump -sass prints:
# a header, then per function its mangled name and its instructions.
cat > "$dir/cuobjdump" <<'FAKE'
#!/usr/bin/env bash
# A dump named <FAKE_SASS>.<the binary's file name> stands for that binary alone.
f=${FAKE_SASS-}.$(basename "${@: -1}")
[ -f "$f" ] && FAKE_SASS=$f
[ -n "${FAKE_SASS-}" ] && [ -f "$FAKE_SASS" ] || exit 1
cat "$FAKE_SASS"
FAKE
chmod +x "$dir/cuobjdump"
export IGNIS_CUOBJDUMP="$dir/cuobjdump"
# function <mangled name> <opcode>...: one function with one instruction per
# opcode, as cuobjdump -sass prints them.
function_sass() {
    local name=$1 op at=16
    shift
    printf '\t\tFunction : %s\n\t.headerflags\t@"EF_CUDA_SM86 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM86)"\n' "$name"
    printf '        /*0000*/                   MOV R1, c[0x0][0x28] ;                 /* 0x00000a0000017a02 */\n'
    printf '                                                                          /* 0x000fc40000000f00 */\n'
    for op in "$@"; do
        printf '        /*%04x*/                   %s R8, R4, R12, R8 ;     /* 0x0000000000000000 */\n' "$at" "$op"
        printf '                                                                          /* 0x000fe20000000000 */\n'
        at=$((at + 16))
    done
    printf '        /*%04x*/                   EXIT ;                                 /* 0x000000000000794d */\n' "$at"
    printf '\t\t.........................\n\n\n'
}
# A plain function (`_Z`, parameters after its name) and a template in an
# anonymous namespace (`_ZN`, template arguments after its name).
plain() { function_sass "_Z${#1}$1PKvi" "${@:2}"; }
templ() { function_sass "_ZN12_GLOBAL__N_1${#1}$1ILi128ELb1EEvv" "${@:2}"; }
# A dump in which every kernel of the list is as the rule wants it, and the
# kernels of #488 and the control are as they are in the build. `$swap` names
# one kernel whose opcodes are `$swap_ops` instead (none when empty), `$drop`
# one the dump leaves out, `$more` text appended.
good_dump() {
    printf 'Fatbin elf code:\n================\narch = sm_86\ncode version = [1,7]\nhost = linux\ncompile_size = 64bit\n\n\tcode for sm_86\n'
    local name
    for name in dense_prefill_decode_kernel dense_prefill_a8_kernel dense_prefill_a8_wide_kernel \
        tile_vt_group_kernel gdn_split_conv_kernel gdn_split_conv_state_kernel dense_prefill_gemm_kernel \
        fa_int8_kernel nvfp4_a16_mma_kernel output_kernel fa_kernel not_in_the_list_kernel; do
        [ "${drop-}" = "$name" ] && continue
        local ops
        case $name in
            dense_prefill_gemm_kernel) ops="IMMA.16832.S8.S8 IMMA.16832.S8.S8" ;;
            fa_int8_kernel) ops="IMMA.16832.S8.S8 IMMA.16832.U8.S8" ;;
            # #488's drafter, and the chunked delta rule, are BF16 and TF32.
            nvfp4_a16_mma_kernel) ops="HMMA.16816.F32.BF16" ;;
            output_kernel) ops="HMMA.1688.F32.TF32" ;;
            # The control: the window's BF16 form accumulates in f32.
            fa_kernel) ops="HMMA.16816.F32.BF16 IMMA.16832.S8.S8" ;;
            # Not in the list: no rule of this check.
            not_in_the_list_kernel) ops="HMMA.16816.F32.BF16" ;;
            *) ops="" ;;
        esac
        [ "${swap-}" = "$name" ] && ops=$swap_ops
        case $name in
            fa_int8_kernel | fa_kernel) templ "$name" $ops ;;
            *) plain "$name" $ops ;;
        esac
    done
    printf '%s' "${more-}"
}

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

good_dump > "$dir/good.sass"
export FAKE_SASS="$dir/good.sass"
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


# ---------------------------------------------------------------------------
# The full-rate rule (#503, #512), on the SASS a binary with 8.6 machine code
# dumps. `both` carries 8.6, so every case above ran it against `good`.
dump() { good_dump > "$dir/$1.sass"; }

# says <exit status> <regex of what the check prints, stdout and stderr> <name> <argument>...
# Run with IGNIS_CUDA_ARCHITECTURES=86;120a and `$FAKE_SASS` set.
says() {
    local want=$1 pattern=$2 name=$3 got output
    shift 3
    output=$(IGNIS_CUDA_ARCHITECTURES='86;120a' bash "$check" "$@" 2>&1)
    got=$?
    if [ "$got" = "$want" ] && grep -qE -- "$pattern" <<< "$output"; then
        printf 'ok    %s\n' "$name"
    else
        printf 'FAIL  %s: expected exit %s and /%s/, got exit %s:\n%s\n' "$name" "$want" "$pattern" "$got" "$output"
        failed=1
    fi
}
# rule <exit status> <regex> <name>: the dump $dir/$FAKE_SASS against the list.
rule() {
    FAKE_SASS="$dir/case.sass" says "$1" "$2" "$3" "$dir/both"
}

# A passing build: every covered kernel as the rule wants it, the exceptions
# and the control as they are in the build, a kernel outside the list that
# breaks the rule without being refused.
good_dump > "$dir/case.sass"
rule 0 'sm_86 mma fa_int8_kernel: IMMA' "a passing build is read and its kernels reported"
rule 0 'sm_86 exempt nvfp4_a16_mma_kernel: HMMA.16816.F32.BF16' "#488's drafter keeps its BF16 form and is reported"
rule 0 'sm_86 exempt output_kernel: HMMA.1688.F32.TF32' "#488's chunked delta rule keeps its TF32 form"
rule 0 'sm_86 control fa_kernel' "the negative control is read in a passing build"

# A forbidden form in a covered kernel is refused, and the refusal names the
# kernel and the opcode.
forbid() { # forbid <kernel> <opcodes> <opcode named>
    swap=$1 swap_ops=$2 good_dump > "$dir/case.sass"
    rule 1 "kernel $1 \(.*\) holds $3," "$1 holding $3 is refused, kernel and opcode named"
}
forbid dense_prefill_gemm_kernel "IMMA.16832.S8.S8 HMMA.16816.F32" "HMMA.16816.F32"
forbid dense_prefill_gemm_kernel "IMMA.16832.S8.S8 HMMA.16816.F32.BF16" "HMMA.16816.F32.BF16"
forbid dense_prefill_gemm_kernel "IMMA.16832.S8.S8 HMMA.1688.F32.TF32" "HMMA.1688.F32.TF32"
forbid dense_prefill_gemm_kernel "IMMA.16832.S8.S8 QMMA.16832.F32.E4M3.E4M3" "QMMA.16832.F32.E4M3.E4M3"
forbid dense_prefill_gemm_kernel "IMMA.16832.S8.S8 QMMA.16864.F32.E2M1.E2M1" "QMMA.16864.F32.E2M1.E2M1"
forbid fa_int8_kernel "IMMA.16832.S8.S8 HGMMA.64x64x16.F32" "HGMMA.64x64x16.F32"
forbid dense_prefill_a8_kernel "UTCHMMA" "UTCHMMA"
forbid gdn_split_conv_kernel "HMMA.16816.F32" "HMMA.16816.F32"
forbid fa_int8_kernel "IMMA.16832.S8.S8 BMMA.88128.POPC" "BMMA.88128.POPC"
# An f16 accumulate is allowed, and said.
swap=fa_int8_kernel swap_ops="IMMA.16832.S8.S8 HMMA.16816" good_dump > "$dir/case.sass"
rule 0 'fa_int8_kernel holds HMMA.16816 \(f16 accumulate, allowed\)' "HMMA with an f16 accumulate is allowed and reported"
# Each instantiation of a template is held to the rule.
more=$(function_sass "_ZN12_GLOBAL__N_114fa_int8_kernelILi64ELb0EEvv" HMMA.16816.F32) good_dump > "$dir/case.sass"
rule 1 'kernel fa_int8_kernel \(_ZN12_GLOBAL__N_114fa_int8_kernelILi64ELb0EEvv\) holds HMMA.16816.F32' "every instantiation of a covered template is held to the rule"

# A GEMM or band kernel holds integer MMA; a CUDA-core kernel needs none.
swap=dense_prefill_gemm_kernel swap_ops="" good_dump > "$dir/case.sass"
rule 1 'kernel dense_prefill_gemm_kernel .* holds no integer MMA' "a GEMM kernel with no MMA at all is refused"
swap=fa_int8_kernel swap_ops="HMMA.16816" good_dump > "$dir/case.sass"
rule 1 'kernel fa_int8_kernel .* holds no integer MMA' "a band kernel holding only f16 MMA is refused"
swap=dense_prefill_decode_kernel swap_ops="IMMA.16832.S8.S8" good_dump > "$dir/case.sass"
rule 0 'sm_86 cuda dense_prefill_decode_kernel: IMMA' "a CUDA-core kernel may hold integer MMA"

# A listed kernel the binaries do not carry is refused: a rename must not skip
# the rule. The names are read from the list, so one more line is one more kernel.
drop=dense_prefill_gemm_kernel good_dump > "$dir/case.sass"
rule 1 'kernel dense_prefill_gemm_kernel, which .*full-rate-kernels.txt .*none of the binaries' "a covered kernel missing from the build is refused"
drop=nvfp4_a16_mma_kernel good_dump > "$dir/case.sass"
rule 0 'sm_86 cuda' "an exempt kernel missing from the build is not refused"
# The binaries are read together: one that links part of the leaf is not
# refused for the kernels another carries.
cp "$dir/both" "$dir/other"
drop=dense_prefill_gemm_kernel good_dump > "$dir/case.sass.both"
good_dump > "$dir/case.sass.other"
FAKE_SASS="$dir/case.sass" says 0 'CUDA build' "two binaries carrying the list between them pass" "$dir/both" "$dir/other"
FAKE_SASS="$dir/case.sass" says 1 'dense_prefill_gemm_kernel, which' "a kernel in none of them is refused" "$dir/both" "$dir/both"
rm "$dir/case.sass.both" "$dir/case.sass.other"

# The negative control: the existing window kernel's f32-accumulate form. Held
# to the rule it must be refused, which proves the check can fail; a dump in
# which it shows no forbidden form is refused, since nothing could be.
good_dump > "$dir/case.sass"
printf 'mma fa_kernel\n' > "$dir/control.list"
FAKE_SASS="$dir/case.sass" IGNIS_FULL_RATE_LIST="$dir/control.list" says 1 'kernel fa_kernel \(.*\) holds HMMA.16816.F32.BF16, which is not a full-rate form' "the window kernel's f32-accumulate form fails the rule (the negative control)" "$dir/both"
swap=fa_kernel swap_ops="IMMA.16832.S8.S8" good_dump > "$dir/case.sass"
rule 1 'negative control fa_kernel shows no forbidden MMA form' "a dump in which the negative control passes is refused"
drop=fa_kernel good_dump > "$dir/case.sass"
rule 1 'negative control fa_kernel shows no forbidden MMA form' "a dump without the negative control's kernel is refused"

# The tool and the list.
rm -f "$dir/case.sass"
rule 1 'cuobjdump could not dump' "a cuobjdump that fails refuses the build"
good_dump > "$dir/case.sass"
FAKE_SASS="$dir/case.sass" IGNIS_CUOBJDUMP="$dir/no-such-cuobjdump" says 1 'could not dump' "a cuobjdump that is not there refuses the build" "$dir/both"
# PATH without the directory that holds cuobjdump, on a machine that has one.
clean=; IFS=: read -r -a path_dirs <<< "$PATH"
for p in "${path_dirs[@]}"; do [ -e "$p/cuobjdump" ] || [ -e "$p/cuobjdump.exe" ] || clean+="$p:"; done
FAKE_SASS="$dir/case.sass" IGNIS_CUOBJDUMP= PATH="$clean" CUDA_PATH= CUDA_HOME= says 1 'none is on PATH' "no cuobjdump anywhere refuses a build with 8.6" "$dir/both"
printf 'gemm dense_prefill_gemm_kernel\n' > "$dir/bad.list"
FAKE_SASS="$dir/case.sass" IGNIS_FULL_RATE_LIST="$dir/bad.list" says 1 'cannot read the line' "a list line with an unknown role refuses the build" "$dir/both"
printf 'mma a_kernel\ncuda a_kernel\n' > "$dir/dup.list"
FAKE_SASS="$dir/case.sass" IGNIS_FULL_RATE_LIST="$dir/dup.list" says 1 'cannot read the line' "a kernel listed twice refuses the build" "$dir/both"
FAKE_SASS="$dir/case.sass" IGNIS_FULL_RATE_LIST="$dir/missing.list" says 1 'list of full-rate kernels is missing' "a missing list refuses the build" "$dir/both"
# Only a binary with 8.6 machine code is held to the rule: the default list is
# 12.0 alone, and needs no cuobjdump.
IGNIS_CUDA_ARCHITECTURES=120a FAKE_SASS= IGNIS_CUOBJDUMP="$dir/no-such-cuobjdump" bash "$check" "$dir/no-86" > /dev/null 2>&1 && echo "ok    a build without 8.6 code does not run the rule" || { echo "FAIL  a build without 8.6 code does not run the rule"; failed=1; }

exit "$failed"
