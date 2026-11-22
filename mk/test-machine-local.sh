#!/usr/bin/env bash
# The machine-local test targets, run for real (`make test-machine-local`,
# GitHub #709).
#
#   bash mk/test-machine-local.sh [cargo flags]          build and run them
#   bash mk/test-machine-local.sh --plan [cargo flags]   print the command, run nothing
#
# mk/machine-local-tests.txt lists the test targets that read a fixture only
# the owner's machine holds; where it is absent they print a skip and pass, so
# on CI they always skip. This runs every target listed there, and nothing
# else, the way `cargo test` runs it: one `cargo test --workspace` with a
# `--test` per target, as mk/test-shard.sh selects them, no `--ignored`, every
# binary run even after one fails (`--no-fail-fast`). It runs once per
# release, on the owner's machine, on the commit the tag will name, before the
# tag is pushed (docs/user/contributing/releases.md).
#
# CPU only, so it takes no GPU lock: IGNIS_TEST_CUDA, which opts a test of
# real_artifact.rs onto the card, is unset. Each test's output is shown
# (`--show-output`), and the lines that print a skip are gathered at the end:
# a skip stays green, so a fixture this machine lacks shows there and nowhere
# else.
#
# CARGO names the cargo to run (the Makefile's), IGNIS_MACHINE_LOCAL_LIST
# another list (mk/test-machine-local-selftest.sh).

set -uo pipefail

plan=0
if [ "${1:-}" = --plan ]; then
    plan=1
    shift
fi
cd "$(dirname "$0")/.."
list=${IGNIS_MACHINE_LOCAL_LIST:-mk/machine-local-tests.txt}
cargo=${CARGO:-cargo}

tests=()
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    line=$(printf '%s' "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    case "$line" in
        "" | "#"* | "exempt "*) continue ;;
        *.rs) name=${line##*/}; tests+=(--test "${name%.rs}") ;;
        *) echo "test-machine-local: $list: not a test target's path: $line" >&2; exit 2 ;;
    esac
done <"$list"
if [ ${#tests[@]} -eq 0 ]; then
    echo "test-machine-local: $list lists no target" >&2
    exit 2
fi

command=("$cargo" test --workspace --no-fail-fast "$@" "${tests[@]}" -- --show-output)
if [ "$plan" = 1 ]; then
    echo "${command[*]}"
    exit 0
fi

unset IGNIS_TEST_CUDA
log=$(mktemp)
trap 'rm -f "$log"' EXIT
echo "+ ${command[*]}"
"${command[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}

skips=$(grep -iE '^[[:space:]]*skip' "$log")
if [ -n "$skips" ]; then
    echo
    echo "test-machine-local: $(printf '%s\n' "$skips" | wc -l | tr -d ' ') line(s) printed a skip (a fixture this machine lacks, or an opt-in left unset):"
    printf '%s\n' "$skips" | sed 's/^[[:space:]]*/  /'
fi
if [ "$status" != 0 ]; then
    echo "test-machine-local: cargo test failed (exit $status)" >&2
fi
exit "$status"
