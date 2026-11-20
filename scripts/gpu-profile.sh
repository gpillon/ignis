#!/usr/bin/env bash
# Single entry point for the explicit GPU test profile on Linux (ADR 0006,
# GitHub #38, #226): the counterpart of scripts/gpu-profile.ps1, with the same
# three stages in the same order, the same flags on each cargo call and the
# same exit codes. Read that script's header for why the entry point
# preflights instead of leaving it to the developer; this one differs only
# where the two hosts do:
#
#   - The preflight is a step below, not a second script: the guard half is
#     mk/linux/gpu.sh (an ignis GPU process anywhere, a standing marker, or
#     more than the threshold resident -- under Linux nvidia-smi reports real
#     VRAM), and on a pass this script writes the marker
#     ignis_core::gpu_profile reads, at ${TMPDIR:-/tmp}/ignis-gpu-preflight.ok
#     (std::env::temp_dir()). It is deleted when the run ends, pass or fail,
#     so one preflight authorizes exactly one run.
#   - The GPU lock is mk/linux/gpulock.py, called with this script's pid; the
#     Windows script dot-sources mk/windows/gpulock.ps1.
#   - Every cargo call passes --target: .cargo/config.toml pins the MSVC
#     triple. TARGET_TRIPLE in the environment overrides the default; the
#     Makefile's hook (mk/os/linux.mk) passes its own.
#   - The kernel stage is kernel/build.sh. The Windows script has no other
#     host-specific step (no WDDM handling, no pdb sweep), so nothing is left
#     out on that account.
#
#   bash scripts/gpu-profile.sh
#   ... --threshold-mib 4096     # the guard's VRAM threshold (default 8192)
#   ... --skip-kernel-build      # skip kernel/build.sh build --test (Rust tests only)
#   ... --skip-cargo-tests       # skip both cargo stages (kernel leaf only)
#   ... --test <name> [--test <name> ...]
#                                # only these #[ignore]d tests, in place of the stages
#   ... --wait-seconds 900       # how long to wait for the GPU lock (default 900)
#
# The GPU lock (GitHub #707) is taken first and held for the whole run: while
# another process holds it the script waits up to --wait-seconds, then fails
# naming the holder, before the preflight, so a marker another run owns is
# never touched. The preflight and the marker are then as without the lock.
#
# --test: the single-test mode, for a ticket's own GPU tests. Each name is an
# #[ignore]d test's full name as `cargo test -- --list --ignored` prints it;
# one listing checks that every name resolves (a name that matches nothing
# would otherwise pass with no test run), then each runs alone, as stage 3
# runs, its wall time reported like a stage's.
#
# Stages, each timed and reported at the end (GitHub #135):
#   1. kernel/build.sh build --test    the leaf's own op tests (CTest)
#   2. cpu f64 layer oracle            CPU-only, full libtest parallelism
#   3. gpu tests (serialized)          --ignored, --test-threads=1 (ADR 0006)
#
# Exit codes: 1 if the lock stays held past the wait, the preflight refuses
# (its own message says why), or the CPU oracle's name or a --test name no
# longer resolves; 2 for a bad flag; otherwise the exit code
# of the first stage that fails, or 0 if all pass. IGNIS_GPU_PROFILE is only
# ever set inside this process, so nothing outlives the run but the marker
# and the lock, which the EXIT trap removes.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
threshold=8192
skip_kernel=0
skip_cargo=0
wait_seconds=900
tests=()
usage() {
    echo "usage: gpu-profile.sh [--threshold-mib N] [--skip-kernel-build] [--skip-cargo-tests] [--test NAME ...] [--wait-seconds N]" >&2
    exit 2
}
while [ $# -gt 0 ]; do
    case "$1" in
        --threshold-mib) threshold="${2:?--threshold-mib needs a number}"; shift ;;
        --skip-kernel-build) skip_kernel=1 ;;
        --skip-cargo-tests) skip_cargo=1 ;;
        --test) [ $# -ge 2 ] || usage; tests+=("$2"); shift ;;
        --wait-seconds) [ $# -ge 2 ] || usage; wait_seconds="$2"; shift ;;
        *) usage ;;
    esac
    shift
done

cargo="${CARGO:-cargo}"
target="${TARGET_TRIPLE:-x86_64-unknown-linux-gnu}"
marker="${TMPDIR:-/tmp}/ignis-gpu-preflight.ok"
lockctl() { "${PYTHON:-python3}" "$repo/mk/linux/gpulock.py" "$@"; }

# The CPU-only f64 layer oracle (GitHub #56): it runs as its own stage at
# libtest's default parallelism, and the serialized sweep skips it by name
# (GitHub #135).
cpu_oracle_test="gqa_and_gdn_references_cover_two_tokens"

# The `--ignored` tests of ignis-server that touch no GPU, which the sweep
# skips by name too. The same list is in scripts/gpu-profile.ps1: each binary
# has a test that its leg's name is listed there, so keep the two equal.
not_gpu_legs=(
    the_eleven_dictations_through_the_real_route_at_real_time_pace
    the_smallest_speech_entry_downloads_and_verifies_against_its_pins
    the_diarizer_adapter_steps_the_crates_stream_on_its_own_pool
    the_diarizer_keeps_up_with_a_conversation_at_real_time_pace
    every_item_of_the_conversations_sends_its_text_whole_at_its_end
    a_long_monologue_keeps_its_live_text_with_speakers_on
)

stage_names=()
stage_secs=()

now() { date +%s.%N; }

write_summary() {
    [ "${#stage_names[@]}" -gt 0 ] || return 0
    local i total=0
    echo
    echo "GPU profile: stage wall time"
    for i in "${!stage_names[@]}"; do
        printf '  %9.1fs  %s\n' "${stage_secs[$i]}" "${stage_names[$i]}"
        total="$(awk -v a="$total" -v b="${stage_secs[$i]}" 'BEGIN { print a + b }')"
    done
    printf '  %9.1fs  total\n' "$total"
}

# In an EXIT trap, so a run that failed a stage still reports where its time
# went, and neither the marker nor the lock outlives the run.
finish() {
    local code=$?
    trap - EXIT
    write_summary
    unset IGNIS_GPU_PROFILE
    rm -f "$marker"
    lockctl release --pid $$
    exit "$code"
}

# A GPU-gated step that failed: report it and leave with its own exit code.
fail_with_code() { # <what> <code>
    echo "GPU profile: $1 failed (exit $2)." >&2
    exit "$2"
}

# Run one stage, record its wall time, fail the run with the stage's own code.
invoke_stage() { # <name> <command ...>
    local name="$1" began code ended secs
    shift
    echo "GPU profile: $name"
    began="$(now)"
    "$@"
    code=$?
    ended="$(now)"
    secs="$(awk -v a="$began" -v b="$ended" 'BEGIN { print b - a }')"
    stage_names+=("$name")
    stage_secs+=("$secs")
    printf 'GPU profile: %s -- %.1fs\n' "$name" "$secs"
    if [ "$code" -ne 0 ]; then fail_with_code "$name" "$code"; fi
}

# The lock, before anything else: a wait that runs out leaves before the
# preflight. Held from here to the EXIT trap.
what="gpu-profile${tests[*]:+ --test ${tests[*]}}"
if ! lockctl acquire --pid $$ --wait "$wait_seconds" --command "$what"; then
    echo "GPU profile: the GPU lock is held (see above) -- not running the GPU profile." >&2
    exit 1
fi
trap finish EXIT

# The preflight: any earlier pass is cleared first, so a refusal never leaves
# an older one standing; the guard refuses a held card (the lock this run
# holds aside); a pass is recorded.
rm -f "$marker"
if ! bash "$repo/mk/linux/gpu.sh" guard "$threshold" $$; then
    echo "GPU profile: preflight refused (see above) -- not running the GPU profile." >&2
    exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$marker"

cd "$repo" || exit 1
export IGNIS_GPU_PROFILE=1
echo "GPU profile: preflight passed -- running the profile."

if [ "${#tests[@]}" -gt 0 ]; then
    # The sweep's build and flags, narrowed to the named tests: one listing
    # that each name resolves, then each test alone.
    listed="$("$cargo" test --workspace --exclude ignis-speech --features cuda --target "$target" -- --list --ignored --exact "${tests[@]}")"
    code=$?
    if [ "$code" -ne 0 ]; then fail_with_code "listing the named tests" "$code"; fi
    for name in "${tests[@]}"; do
        if ! grep -qxF -- "$name: test" <<< "$listed"; then
            echo "GPU profile: no #[ignore]d test is named '$name' (cargo test -- --list --ignored --exact): not running the named tests." >&2
            exit 1
        fi
    done
    for name in "${tests[@]}"; do
        invoke_stage "gpu test $name" \
            "$cargo" test --workspace --exclude ignis-speech --features cuda --target "$target" --no-fail-fast -- --ignored --exact --test-threads=1 --nocapture "$name"
    done
    echo "GPU profile: done. If you stopped the engine that serves your agent, restart it now (runbook step 3, docs/agents/testing.md)."
    exit 0
fi

if [ "$skip_kernel" = 0 ]; then
    invoke_stage "kernel/build.sh build --test" bash kernel/build.sh build --test
fi

if [ "$skip_cargo" = 0 ]; then
    # Stage 3 excludes the CPU oracle by test-name string; check the name still
    # resolves before relying on it (a rename would silently put the oracle
    # back inside the serialization).
    listed="$("$cargo" test -p ignis-artifact --features cuda --target "$target" --test layer_reference_real -- --list)"
    code=$?
    if [ "$code" -ne 0 ]; then fail_with_code "listing the CPU oracle's tests" "$code"; fi
    if ! grep -qF -- "$cpu_oracle_test" <<< "$listed"; then
        echo "GPU profile: '$cpu_oracle_test' is no longer a test in layer_reference_real, so stage 3's --skip would stop matching and re-run the CPU oracle inside the serialized sweep. Update cpu_oracle_test in this script." >&2
        exit 1
    fi

    # The CPU-only oracle, at full parallelism. `--features cuda` matches stage
    # 3's feature set (else cargo builds the dependency tree twice).
    invoke_stage "cpu f64 layer oracle" \
        "$cargo" test -p ignis-artifact --features cuda --target "$target" --test layer_reference_real -- --ignored --nocapture

    # --test-threads=1: one GPU-gated binary's tests must not load the artifact
    # concurrently onto the one card (ADR 0006, GitHub #75). --skip: the CPU
    # oracle ran above, and the server's non-GPU legs are not GPU tests.
    # --nocapture: the sweep's diagnostics are its point. --exclude
    # ignis-speech: its ignored tests are CPU legs on the study disk (spec
    # speech/01). --no-fail-fast: one failing binary must not hide every GPU
    # test after it (#424, #425); the stage still fails, with every failure
    # listed.
    skips=(--skip "$cpu_oracle_test")
    for leg in "${not_gpu_legs[@]}"; do skips+=(--skip "$leg"); done
    invoke_stage "gpu tests (serialized)" \
        "$cargo" test --workspace --exclude ignis-speech --features cuda --target "$target" --no-fail-fast -- --ignored --test-threads=1 --nocapture "${skips[@]}"
fi

echo "GPU profile: done. If you stopped the engine that serves your agent, restart it now (runbook step 3, docs/agents/testing.md)."
exit 0
