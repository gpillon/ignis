#!/usr/bin/env bash
# The behavioural test of the version tool (`make version-selftest`), for
# both implementations: mk/linux/version.sh always, mk/windows/version.ps1
# whenever a powershell is on PATH. docs/agents/testing.md's rule -- every
# code change ships with a test -- and the two are checked against each other
# rather than separately, because the thing most likely to go wrong with two
# implementations of one tool is that they stop agreeing.
#
# It drives the real repo files and puts them back with git afterwards, so it
# refuses to start unless those four files are clean. It is NOT part of
# `make ci`: a check that writes to tracked files has no business running
# inside one.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo" || exit 1

failed=0
note() { printf '  %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

restore() {
    git checkout -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>/dev/null
}
trap restore EXIT

dirty="$(git status --porcelain -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>/dev/null)"
if [ -n "$dirty" ]; then
    echo "error: the version files have uncommitted changes -- this test rewrites them" >&2
    echo "$dirty" >&2
    exit 1
fi

# The implementations under test. Each is a command prefix taking the tool's
# own arguments; `run <impl> <args...>` is how the cases below call them.
implementations=("bash mk/linux/version.sh")
if command -v powershell >/dev/null 2>&1; then
    implementations+=("powershell -NoProfile -ExecutionPolicy Bypass -File mk/windows/version.ps1")
elif command -v pwsh >/dev/null 2>&1; then
    implementations+=("pwsh -NoProfile -File mk/windows/version.ps1")
else
    note "no powershell on PATH: the Windows implementation is not exercised here"
fi

run() {
    local impl="$1"; shift
    # shellcheck disable=SC2086 -- the prefix is ours, and word-split on purpose
    $impl "$@" 2>&1
}

# What the four files declare, as one line, read independently of the tool so
# a tool that lies about its own writes is caught.
declared() {
    local cargo lock web weblock
    cargo="$(sed -n '/^\[workspace\.package\]/,/^\[/p' Cargo.toml | sed -n 's/^version *= *"\(.*\)"/\1/p' | head -1)"
    lock="$(awk '/^name = "ignis-[a-z-]*"$/ { getline; sub(/^version = "/, ""); sub(/"$/, ""); print }' Cargo.lock | sort -u | paste -sd/ -)"
    web="$(sed -n 's/^  "version": "\(.*\)",$/\1/p' web/package.json | head -1)"
    weblock="$(sed -n 's/^  "version": "\(.*\)",$/\1/p' web/package-lock.json | head -1)"
    printf '%s %s %s %s' "$cargo" "$lock" "$web" "$weblock"
}

expect_all() {
    local want="$1" label="$2" got
    got="$(declared)"
    if [ "$got" = "$want $want $want $want" ]; then
        note "ok: $label -> $want"
    else
        fail "$label: the four files are [$got], wanted $want in all four"
    fi
}

expect_refused() {
    local output="$1" label="$2" needle="${3:-}"
    if [ -n "$output" ] && printf '%s' "$output" | grep -qi "error"; then
        if [ -z "$needle" ] || printf '%s' "$output" | grep -qi -- "$needle"; then
            note "ok: $label is refused"
            return
        fi
        fail "$label: refused, but the message does not mention '$needle': $output"
        return
    fi
    fail "$label: not refused (output: $output)"
}

for impl in "${implementations[@]}"; do
    echo "== ${impl##* } =="
    restore

    # An explicit version reaches all four files.
    run "$impl" set 9.9.9 >/dev/null
    expect_all 9.9.9 "set 9.9.9"

    # The three parts, each from the version before it.
    run "$impl" bump patch >/dev/null
    expect_all 9.9.10 "bump patch"
    run "$impl" bump minor >/dev/null
    expect_all 9.10.0 "bump minor"
    run "$impl" bump major >/dev/null
    expect_all 10.0.0 "bump major"

    # A prerelease is a version like any other, and patching one releases it
    # rather than stepping past it (`npm version patch`'s rule).
    run "$impl" set 10.1.0-rc.1 >/dev/null
    expect_all 10.1.0-rc.1 "set 10.1.0-rc.1"
    run "$impl" bump patch >/dev/null
    expect_all 10.1.0 "bump patch from a prerelease"

    # check agrees with the files while they agree with each other.
    if run "$impl" check | grep -q "ok: every file declares 10.1.0"; then
        note "ok: check passes on an agreeing tree"
    else
        fail "check did not pass on an agreeing tree"
    fi

    # Setting what every file already declares is the one refusal that is
    # about state rather than syntax.
    expect_refused "$(run "$impl" set 10.1.0)" "setting the version already declared" "already declares"

    # Syntax. `+build` is refused on purpose: npm drops build metadata, so it
    # could never reach the web files (see either script's header).
    expect_refused "$(run "$impl" set 1.2)" "a two-part version" "not a version"
    expect_refused "$(run "$impl" set 1.2.3+cuda13)" "build metadata" "not a version"
    expect_refused "$(run "$impl" set '')" "an empty version"
    expect_refused "$(run "$impl" bump quarterly)" "an unknown part"

    # A half-bumped tree is the state this tool exists to repair, so setting
    # the version it half-carries must work rather than be refused.
    restore
    run "$impl" set 9.9.9 >/dev/null
    git checkout -- web/package.json web/package-lock.json
    run "$impl" set 9.9.9 >/dev/null
    expect_all 9.9.9 "repairing a tree where only the web files drifted"

    restore
done

if [ "$failed" -ne 0 ]; then
    echo "version selftest: $failed check(s) failed" >&2
    exit 1
fi
echo "version selftest: all checks passed"
