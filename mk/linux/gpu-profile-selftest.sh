#!/usr/bin/env bash
# The behavioural test of scripts/gpu-profile.sh, the GPU_PROFILE hook behind
# `make gpu-profile` on Linux (GitHub #226). It copies the script,
# mk/linux/gpu.sh and mk/linux/gpulock.py into a temporary tree and runs them on stand-ins first on
# PATH: a `cargo` and a kernel/build.sh that record their arguments, the
# profile variable, whether the marker stood at that moment and whether the
# profile held the GPU lock, an
# `nvidia-smi` that reports a chosen VRAM figure and a `pgrep` that finds no
# holder. It checks the stage order, the flags of each stage, that the marker
# is there while the stages run and gone afterwards on a pass, a failing stage
# and a refusal, the exit codes, that the not-GPU legs are the ones
# scripts/gpu-profile.ps1 lists, that the lock is held for the whole run and
# waited for up to --wait-seconds without touching the marker (GitHub #707),
# and the single-test mode, --test: each named test listed, then run alone.
# It needs no GPU, no build and no CUDA; Linux only (the lock reads /proc).
# It writes only under a temporary
# directory it removes, and TMPDIR and IGNIS_GPU_LOCK_DIR point there, so a
# marker the real profile owns, or the machine's lock, is never read or
# removed.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

failed=0
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tree="$tmp/repo"
bin="$tmp/bin"
mkdir -p "$tree/scripts" "$tree/mk/linux" "$tree/kernel" "$bin" "$tmp/tmp"
cp scripts/gpu-profile.sh "$tree/scripts/"
cp mk/linux/gpu.sh mk/linux/gpulock.py "$tree/mk/linux/"
log="$tmp/calls.log"
marker="$tmp/tmp/ignis-gpu-preflight.ok"
export IGNIS_GPU_LOCK_DIR="$tmp/lock"
unset IGNIS_GPU_CARD
lock="$IGNIS_GPU_LOCK_DIR/gpu-0.lock"
resolved="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import gpulock; print(gpulock.lock_path())' "$tree/mk/linux")"
case "$resolved" in
    "$tmp"/*) ;;
    *) echo "gpu-profile-selftest: the lock resolves to $resolved, outside $tmp: not running" >&2; exit 1 ;;
esac
sleepers=()
trap 'for pid in "${sleepers[@]}"; do { kill -KILL "$pid" && wait "$pid"; } 2>/dev/null; done; rm -rf "$tmp"' EXIT

# The stand-ins. STUB_*_RC: the exit code of that stage's stand-in; STUB_USED:
# the MiB nvidia-smi reports; STUB_NO_ORACLE: --list names no oracle test;
# STUB_LISTED: the further tests --list names, comma-separated; STUB_FAIL_TEST: the one
# named test that fails (exit 5). Each records lock=parent when the GPU lock
# names its parent, the profile, as its holder.
cat > "$tree/kernel/build.sh" <<'EOF'
#!/usr/bin/env bash
echo "kernel $* | profile=${IGNIS_GPU_PROFILE:-} marker=$([ -f "$TMPDIR/ignis-gpu-preflight.ok" ] && echo yes || echo no) lock=$(grep -q "\"pid\": $PPID," "$IGNIS_GPU_LOCK_DIR/gpu-0.lock" 2>/dev/null && echo parent || echo no)" >> "$STUB_LOG"
exit "${STUB_KERNEL_RC:-0}"
EOF
cat > "$bin/cargo" <<'EOF'
#!/usr/bin/env bash
echo "cargo $* | profile=${IGNIS_GPU_PROFILE:-} marker=$([ -f "$TMPDIR/ignis-gpu-preflight.ok" ] && echo yes || echo no) lock=$(grep -q "\"pid\": $PPID," "$IGNIS_GPU_LOCK_DIR/gpu-0.lock" 2>/dev/null && echo parent || echo no)" >> "$STUB_LOG"
case " $* " in
    *" --list "*)
        [ -n "${STUB_NO_ORACLE:-}" ] || echo "gqa_and_gdn_references_cover_two_tokens: test"
        for name in ${STUB_LISTED//,/ }; do echo "$name: test"; done
        exit "${STUB_LIST_RC:-0}" ;;
    *" --exact "*" ${STUB_FAIL_TEST:-none} "*) exit 5 ;;
    *" --workspace "*) exit "${STUB_SWEEP_RC:-0}" ;;
    *) exit "${STUB_ORACLE_RC:-0}" ;;
esac
EOF
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *memory.used*) echo "${STUB_USED:-100}" ;;
    *) echo "stub-gpu" ;;
esac
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "$bin/pgrep"
chmod +x "$tree/kernel/build.sh" "$bin/cargo" "$bin/nvidia-smi" "$bin/pgrep"

# The profile with environment $1 (words, may be empty) and arguments after
# it; sets $code and $out, and leaves the calls in $log.
run() {
    local env_words="$1"
    shift
    : > "$log"
    # shellcheck disable=SC2086
    out="$(env $env_words STUB_LOG="$log" TMPDIR="$tmp/tmp" PATH="$bin:$PATH" bash "$tree/scripts/gpu-profile.sh" "$@" 2>&1)"
    code=$?
}
expect_code() { [ "$code" = "$2" ] || fail "$1: exit $code, expected $2:
$out"; }
has() { # <label> <text> <file-or-output>
    grep -qxF -- "$2" <<< "$3" || fail "$1: no line '$2' in:
$3"
}
no_marker() { [ ! -e "$marker" ] || fail "$1: the marker is still there"; }
no_lock() { [ ! -e "$lock" ] || fail "$1: the lock is still there: $(cat "$lock")"; }
calls() { cat "$log"; }
lines() { calls | wc -l | tr -d ' '; }

gnu="x86_64-unknown-linux-gnu"
listing="cargo test -p ignis-artifact --features cuda --target $gnu --test layer_reference_real -- --list | profile=1 marker=yes lock=parent"
oracle="cargo test -p ignis-artifact --features cuda --target $gnu --test layer_reference_real -- --ignored --nocapture | profile=1 marker=yes lock=parent"
sweep_skips="--skip gqa_and_gdn_references_cover_two_tokens"
for leg in $(sed -n 's/^    \([a-z_]*\)$/\1/p' <<< "$(sed -n '/^not_gpu_legs=(/,/^)/p' scripts/gpu-profile.sh)"); do
    sweep_skips="$sweep_skips --skip $leg"
done
sweep="cargo test --workspace --exclude ignis-speech --features cuda --target $gnu --no-fail-fast -- --ignored --test-threads=1 --nocapture $sweep_skips | profile=1 marker=yes lock=parent"
kernel="kernel build --test | profile=1 marker=yes lock=parent"

# A pass: the three stages in order, each with the profile on and the marker
# standing, then the marker gone and a stage summary.
run ""
expect_code "pass" 0
[ "$(calls)" = "$kernel
$listing
$oracle
$sweep" ] || fail "pass: the stages are not as expected:
$(calls)"
no_marker "pass"
no_lock "pass"
has "pass" "GPU profile: stage wall time" "$out"
grep -q 'cpu f64 layer oracle$' <<< "$out" || fail "pass: no timing for the oracle stage"
grep -q 'total$' <<< "$out" || fail "pass: no total in the summary"
# The legs the sweep skips by name are the ones the Windows script lists.
ps1_legs="$(sed -n '/^\$NotGpuLegs = @(/,/^)/p' scripts/gpu-profile.ps1 | sed -n 's/^ *"\([a-z_]*\)",\{0,1\}$/\1/p' | tr '\n' ' ')"
sh_legs="$(sed -n '/^not_gpu_legs=(/,/^)/p' scripts/gpu-profile.sh | sed -n 's/^ *\([a-z_]*\)$/\1/p' | tr '\n' ' ')"
[ -n "$ps1_legs" ] && [ "$ps1_legs" = "$sh_legs" ] || fail "the not-GPU legs differ from scripts/gpu-profile.ps1's: '$ps1_legs' against '$sh_legs'"

# The cargo target is TARGET_TRIPLE when the Makefile passes one.
run "TARGET_TRIPLE=aarch64-unknown-linux-gnu"
expect_code "target" 0
grep -q -- '--target aarch64-unknown-linux-gnu' "$log" || fail "target: TARGET_TRIPLE not passed:
$(calls)"
! grep -q -- "--target $gnu" "$log" || fail "target: the default leaked through"

# A failing stage ends the run with its own exit code, the later stages never
# start, and the marker is gone.
run "STUB_KERNEL_RC=5"
expect_code "kernel fails" 5
[ "$(lines)" = 1 ] || fail "kernel fails: later stages ran:
$(calls)"
no_marker "kernel fails"
no_lock "kernel fails"
has "kernel fails" "GPU profile: kernel/build.sh build --test failed (exit 5)." "$out"
has "kernel fails" "GPU profile: stage wall time" "$out"

run "STUB_LIST_RC=3"
expect_code "listing fails" 3
[ "$(lines)" = 2 ] || fail "listing fails: later stages ran:
$(calls)"
no_marker "listing fails"

run "STUB_ORACLE_RC=7"
expect_code "oracle fails" 7
[ "$(calls | tail -1)" = "$oracle" ] || fail "oracle fails: the sweep ran after it:
$(calls)"
no_marker "oracle fails"

run "STUB_SWEEP_RC=9"
expect_code "sweep fails" 9
[ "$(calls | tail -1)" = "$sweep" ] || fail "sweep fails: not the last call:
$(calls)"
no_marker "sweep fails"
has "sweep fails" "GPU profile: gpu tests (serialized) failed (exit 9)." "$out"

# An oracle name that no longer resolves stops the run before the sweep.
run "STUB_NO_ORACLE=1"
expect_code "oracle renamed" 1
[ "$(lines)" = 2 ] || fail "oracle renamed: stages ran after the listing:
$(calls)"
no_marker "oracle renamed"

# A refused preflight runs nothing, exits 1, and clears an older pass.
run "STUB_USED=9000"
expect_code "refused" 1
[ "$(lines)" = 0 ] || fail "refused: a stage ran:
$(calls)"
no_marker "refused"
no_lock "refused"
printf 'old\n' > "$marker"
run "STUB_USED=9000"
expect_code "refused over a marker" 1
no_marker "refused over a marker"
run "STUB_USED=200" --threshold-mib 100
expect_code "threshold flag" 1
run "STUB_USED=200" --threshold-mib 400
expect_code "threshold flag, under" 0

# The lock held by another process: the profile waits up to --wait-seconds,
# then fails naming the holder, before the preflight, so a marker another
# run owns stays where it is.
sleep 300 &
holder=$!
sleepers+=("$holder")
python3 "$tree/mk/linux/gpulock.py" acquire --pid "$holder" --wait 0 --command "make start" > /dev/null
printf 'another run\n' > "$marker"
began=$SECONDS
run "" --wait-seconds 2
expect_code "the lock held" 1
[ $((SECONDS - began)) -ge 2 ] || fail "the lock held: it gave up before its wait"
grep -qF "GPU lock: waiting up to 2s for the card, held ($lock) by pid $holder" <<< "$out" ||
    fail "the lock held: no wait naming the holder:
$out"
grep -qF "the card is held ($lock) by pid $holder" <<< "$out" || fail "the lock held: no refusal naming the holder:
$out"
[ "$(lines)" = 0 ] || fail "the lock held: a stage ran:
$(calls)"
[ "$(cat "$marker" 2>/dev/null)" = "another run" ] || fail "the lock held: the marker another run owns was touched"
rm -f "$marker"
( sleep 1; python3 "$tree/mk/linux/gpulock.py" release --pid "$holder" ) &
releaser=$!
run "" --wait-seconds 10 --skip-kernel-build
expect_code "the lock freed during the wait" 0
wait "$releaser"
[ "$(calls)" = "$listing
$oracle
$sweep" ] || fail "the lock freed during the wait:
$(calls)"
no_lock "the lock freed during the wait"

# The single-test mode: the named tests listed in one call, each then run
# alone with the sweep's flags, under the profile, the marker and the lock;
# no other stage.
named_list="cargo test --workspace --exclude ignis-speech --features cuda --target $gnu -- --list --ignored --exact a_gpu_test b_gpu_test | profile=1 marker=yes lock=parent"
named_a="cargo test --workspace --exclude ignis-speech --features cuda --target $gnu --no-fail-fast -- --ignored --exact --test-threads=1 --nocapture a_gpu_test | profile=1 marker=yes lock=parent"
named_b="cargo test --workspace --exclude ignis-speech --features cuda --target $gnu --no-fail-fast -- --ignored --exact --test-threads=1 --nocapture b_gpu_test | profile=1 marker=yes lock=parent"
run "STUB_LISTED=a_gpu_test,b_gpu_test" --test a_gpu_test --test b_gpu_test
expect_code "two tests" 0
[ "$(calls)" = "$named_list
$named_a
$named_b" ] || fail "two tests: the calls are not as expected:
$(calls)"
grep -q 'gpu test a_gpu_test$' <<< "$out" && grep -q 'gpu test b_gpu_test$' <<< "$out" ||
    fail "two tests: no timing for each test:
$out"
no_marker "two tests"
no_lock "two tests"
run "STUB_LISTED=a_gpu_test" --test a_gpu_test --test b_gpu_test
expect_code "a name that is no #[ignore]d test" 1
[ "$(lines)" = 1 ] || fail "a name that is no #[ignore]d test: tests ran:
$(calls)"
has "a name that is no #[ignore]d test" "GPU profile: no #[ignore]d test is named 'b_gpu_test' (cargo test -- --list --ignored --exact): not running the named tests." "$out"
run "STUB_LISTED=a_gpu_test,b_gpu_test STUB_FAIL_TEST=a_gpu_test" --test a_gpu_test --test b_gpu_test
expect_code "the first named test fails" 5
[ "$(calls | tail -1)" = "$named_a" ] || fail "the first named test fails: the second ran:
$(calls)"
no_marker "the first named test fails"
no_lock "the first named test fails"
run "" --test
expect_code "--test without a name" 2

# The skip flags.
run "" --skip-kernel-build
expect_code "skip kernel" 0
[ "$(calls)" = "$listing
$oracle
$sweep" ] || fail "skip kernel:
$(calls)"
run "" --skip-cargo-tests
expect_code "skip cargo" 0
[ "$(calls)" = "$kernel" ] || fail "skip cargo:
$(calls)"
run "" --skip-kernel-build --skip-cargo-tests
expect_code "skip both" 0
[ "$(lines)" = 0 ] || fail "skip both: a stage ran"
no_marker "skip both"

# A flag it does not know is a usage error, before anything else.
run "" --SkipKernelBuild
expect_code "unknown flag" 2
[ "$(lines)" = 0 ] || fail "unknown flag: a stage ran"

# The Makefile's hook runs this script with its target and flags.
grep -qF 'GPU_PROFILE = TARGET_TRIPLE='"'"'$(TARGET_TRIPLE)'"'"' bash scripts/gpu-profile.sh --wait-seconds $(GPU_LOCK_WAIT) $(GPU_PROFILE_ARGS)' mk/os/linux.mk ||
    fail "mk/os/linux.mk does not wire GPU_PROFILE to scripts/gpu-profile.sh"

if [ "$failed" -ne 0 ]; then
    printf 'gpu-profile-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "gpu-profile-selftest: ok"
