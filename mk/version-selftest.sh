#!/usr/bin/env bash
# The behavioural test of mk/version.sh (`make version-selftest`), which is
# docs/agents/testing.md's rule applied to a script: every code change ships
# with a test. It found four real bugs in the tool the first time it ran --
# build metadata silently splitting the four files, a repair being refused, a
# clean-tree guard blocking the second bump of a session, and a success
# message printed over files that had not taken the version.
#
# It drives the real repo files and puts them back with git afterwards, so it
# refuses to start unless those four files are clean. It is NOT part of
# `make ci`: a check that writes to tracked files has no business inside one.
#
# Every assertion reads the four files directly rather than believing what
# the tool printed -- the point is to catch a tool that lies about its own
# writes, which is exactly what the first version did.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo" || exit 1

tool="bash mk/version.sh"
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

# What the four files declare, as one line, read independently of the tool.
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

# An explicit version reaches all four files.
$tool set 9.9.9 >/dev/null 2>&1
expect_all 9.9.9 "set 9.9.9"

# The three parts, each from the version before it.
$tool bump patch >/dev/null 2>&1
expect_all 9.9.10 "bump patch"
$tool bump minor >/dev/null 2>&1
expect_all 9.10.0 "bump minor"
$tool bump major >/dev/null 2>&1
expect_all 10.0.0 "bump major"

# A prerelease is a version like any other, and patching one releases it
# rather than stepping past it (`npm version patch`'s rule).
$tool set 10.1.0-rc.1 >/dev/null 2>&1
expect_all 10.1.0-rc.1 "set 10.1.0-rc.1"
$tool bump patch >/dev/null 2>&1
expect_all 10.1.0 "bump patch from a prerelease"

# check agrees with the files while they agree with each other.
if $tool check 2>&1 | grep -q "ok: every file declares 10.1.0"; then
    note "ok: check passes on an agreeing tree"
else
    fail "check did not pass on an agreeing tree"
fi

# Setting what every file already declares is the one refusal that is about
# state rather than syntax.
expect_refused "$($tool set 10.1.0 2>&1)" "setting the version already declared" "already declares"

# Syntax. `+build` is refused on purpose: npm drops build metadata, so it
# could never reach the web files (see mk/version.sh's header).
expect_refused "$($tool set 1.2 2>&1)" "a two-part version" "not a version"
expect_refused "$($tool set 1.2.3+cuda13 2>&1)" "build metadata" "not a version"
expect_refused "$($tool set '' 2>&1)" "an empty version"
expect_refused "$($tool bump quarterly 2>&1)" "an unknown part"

# A half-bumped tree is the state this tool exists to repair, so setting the
# version it half-carries must work rather than be refused.
restore
$tool set 9.9.9 >/dev/null 2>&1
git checkout -- web/package.json web/package-lock.json
$tool set 9.9.9 >/dev/null 2>&1
expect_all 9.9.9 "repairing a tree where only the web files drifted"

restore

if [ "$failed" -ne 0 ]; then
    echo "version selftest: $failed check(s) failed" >&2
    exit 1
fi
echo "version selftest: all checks passed"
