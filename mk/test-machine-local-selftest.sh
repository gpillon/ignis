#!/usr/bin/env bash
# The behavioural test of mk/test-machine-local.sh (`make
# test-machine-local-selftest`, GitHub #709). Each case is a list in a temp
# directory and the script's `--plan`, or a stand-in cargo that only records
# how it was run, so it builds nothing and runs no test. The one real file it
# reads is mk/machine-local-tests.txt, whose every listed target the plan must
# run. The release workflow's test job runs it on both hosts.

set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script="$here/test-machine-local.sh"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0

fail() {
    echo "FAIL: $1" >&2
    failed=1
}

# A list as one is written: comments, a blank line, an exemption, a CRLF line
# from a Windows checkout, an indented path.
printf '%s\n' '# a comment' '' 'crates/a/tests/first.rs' 'exempt crates/b/tests/exempt.rs: a reason' >"$dir/list"
printf 'crates/c/tests/second.rs\r\n' >>"$dir/list"
printf '  crates/c/tests/third.rs\n' >>"$dir/list"
plan=$(IGNIS_MACHINE_LOCAL_LIST="$dir/list" CARGO=cargo bash "$script" --plan --target x86_64-unknown-linux-gnu)
want="cargo test --workspace --no-fail-fast --target x86_64-unknown-linux-gnu --test first --test second --test third -- --show-output"
[ "$plan" = "$want" ] || fail "the plan of a written list is
  $plan
not
  $want"

# The real list: every listed target, and no exempt one, is a --test.
real="$here/machine-local-tests.txt"
listed=$(sed 's/\r$//' "$real" | grep -E '^[[:space:]]*crates/' | sed 's#.*/##; s/\.rs$//' | sort)
planned=$(CARGO=cargo bash "$script" --plan | tr ' ' '\n' | grep -A1 -x -- --test | grep -vx -- --test | grep -vx -- -- | sort)
[ -n "$listed" ] || fail "the real list lists no target"
[ "$listed" = "$planned" ] || fail "the plan of the real list runs $(echo $planned), the list names $(echo $listed)"
for exempt in $(sed 's/\r$//' "$real" | grep -E '^exempt ' | sed 's/^exempt //; s/:.*//; s#.*/##; s/\.rs$//'); do
    printf '%s\n' "$planned" | grep -qx -- "$exempt" && fail "the plan runs the exempt target $exempt"
done

# A line that is not a target's path is refused, and so is a list of none.
printf '%s\n' 'crates/a/tests/first' >"$dir/bad"
IGNIS_MACHINE_LOCAL_LIST="$dir/bad" bash "$script" --plan >/dev/null 2>&1
[ $? = 2 ] || fail "a line that is not a .rs path is not refused"
printf '%s\n' '# only a comment' >"$dir/none"
IGNIS_MACHINE_LOCAL_LIST="$dir/none" bash "$script" --plan >/dev/null 2>&1
[ $? = 2 ] || fail "a list of no target is not refused"

# The run: a stand-in cargo records its arguments and IGNIS_TEST_CUDA, prints
# a skip as a test would, and exits with $STUB_EXIT.
cat >"$dir/cargo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >"$STUB_DIR/args"
printf '%s\n' "${IGNIS_TEST_CUDA-unset}" >"$STUB_DIR/cuda"
echo "running 1 test"
echo "skip: F:/ai/q38/model.ninfer does not exist"
exit "$STUB_EXIT"
EOF
run() {
    STUB_DIR="$dir" STUB_EXIT=$1 IGNIS_TEST_CUDA=1 CARGO="$dir/cargo" IGNIS_MACHINE_LOCAL_LIST="$dir/list" \
        bash "$script" --target x86_64-unknown-linux-gnu >"$dir/out" 2>&1
}
run 0 || fail "a green cargo test fails the run: $(cat "$dir/out")"
[ "$(cat "$dir/args")" = "${want#cargo }" ] || fail "cargo ran as $(cat "$dir/args"), the plan says ${want#cargo }"
[ "$(cat "$dir/cuda")" = unset ] || fail "IGNIS_TEST_CUDA reached the tests: $(cat "$dir/cuda")"
grep -q '^  skip: F:/ai/q38/model.ninfer does not exist$' "$dir/out" || fail "the skip is not gathered at the end: $(cat "$dir/out")"
run 101
[ $? = 101 ] || fail "a red cargo test does not fail the run with its status"

if [ "$failed" = 0 ]; then
    echo "test-machine-local-selftest: ok"
fi
exit "$failed"
