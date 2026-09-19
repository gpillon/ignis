#!/usr/bin/env bash
# The release version, for the Makefile (the Linux VERSION_TOOL hook,
# mk/os/linux.mk). The counterpart of mk/windows/version.ps1 -- same three
# actions, same files, same output.
#
#   show           print the version each file declares
#   check          exit 1 when they disagree (what .github/workflows/release.yml
#                  checks before it builds anything, runnable before you tag)
#   set <version>  write an explicit version, `x.y.z` with an optional
#                  `-prerelease` and `+build`
#   bump <part>    patch, minor or major, from what the workspace declares
#
# The version lives in four files and every one of them must agree: the
# release workflow refuses a tag whose Cargo.toml and web/package.json differ
# (the Playground ships inside the binary, so one version covers both), and a
# lockfile left behind makes the next build rewrite it under you.
#
#   Cargo.toml              [workspace.package] version -- the source of truth
#   Cargo.lock              every workspace member's entry
#   web/package.json        the Playground's own version
#   web/package-lock.json   its root and its "" package entry
#
# Only the first of those is edited here. `cargo update --workspace` owns the
# Cargo lockfile and `npm version` owns both web files -- hand-written JSON
# surgery on a lockfile is how every nested dependency ends up carrying the
# workspace's version number.
#
# Nothing here commits or tags: it edits the files and prints the commands,
# because a tag push builds and publishes a release image and that is the
# operator's call, not a side effect of a version bump.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

die() { echo "error: $*" >&2; exit 1; }

# `x.y.z`, optionally `-prerelease` and `+build` (semver 2.0.0's grammar,
# minus the leading-zero rule -- `01.0.0` is nobody's typo worth a rejection
# message).
version_re='^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$'

cargo_version() {
    sed -n '/^\[workspace\.package\]/,/^\[/p' Cargo.toml |
        sed -n 's/^version *= *"\(.*\)"/\1/p' | head -1
}

web_version() {
    sed -n 's/^  "version": "\(.*\)",$/\1/p' web/package.json | head -1
}

web_lock_version() {
    sed -n 's/^  "version": "\(.*\)",$/\1/p' web/package-lock.json | head -1
}

# The lockfile's entry for one workspace member -- the line after its name.
cargo_lock_version() {
    awk '/^name = "ignis-server"$/ { getline; sub(/^version = "/, ""); sub(/"$/, ""); print; exit }' Cargo.lock
}

show() {
    printf 'Cargo.toml            %s\n' "$(cargo_version)"
    printf 'Cargo.lock            %s\n' "$(cargo_lock_version)"
    printf 'web/package.json      %s\n' "$(web_version)"
    printf 'web/package-lock.json %s\n' "$(web_lock_version)"
}

check() {
    local cargo web lock web_lock failed=0
    cargo="$(cargo_version)"
    web="$(web_version)"
    lock="$(cargo_lock_version)"
    web_lock="$(web_lock_version)"
    [ -n "$cargo" ] || die "no version in Cargo.toml [workspace.package]"
    show
    if [ "$cargo" != "$web" ]; then
        echo "error: Cargo.toml ($cargo) and web/package.json ($web) disagree -- the release workflow refuses this" >&2
        failed=1
    fi
    if [ "$cargo" != "$lock" ]; then
        echo "error: Cargo.lock ($lock) is stale -- run 'cargo update --workspace --offline'" >&2
        failed=1
    fi
    if [ "$cargo" != "$web_lock" ]; then
        echo "error: web/package-lock.json ($web_lock) is stale" >&2
        failed=1
    fi
    [ "$failed" -eq 0 ] || exit 1
    echo "ok: every file declares $cargo"
}

# Refuse to edit a file that already carries changes: a bump is a mechanical
# rewrite, and mixing it into unrelated work is how a release commit ends up
# carrying something nobody reviewed.
require_clean() {
    local dirty
    dirty="$(git status --porcelain -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>/dev/null)"
    [ -z "$dirty" ] || die "the version files already have uncommitted changes:
$dirty"
}

write_version() {
    local new="$1" old
    old="$(cargo_version)"
    [ -n "$old" ] || die "no version in Cargo.toml [workspace.package]"
    [ "$new" != "$old" ] || die "the workspace already declares $new"

    # Only inside [workspace.package]: `version = "1"` appears under a dozen
    # dependencies in the same file.
    awk -v new="$new" '
        /^\[workspace\.package\]$/ { in_section = 1; print; next }
        /^\[/ { in_section = 0 }
        in_section && /^version *= *"/ { print "version = \"" new "\""; next }
        { print }
    ' Cargo.toml > Cargo.toml.tmp && mv Cargo.toml.tmp Cargo.toml

    # Cargo owns its lockfile: --offline touches nothing but the workspace
    # members' own entries, and never reaches the network.
    if ! cargo update --workspace --offline >/dev/null 2>&1; then
        git checkout -- Cargo.toml 2>/dev/null
        die "cargo could not refresh Cargo.lock (Cargo.toml left at $old)"
    fi

    # npm owns both web files: package.json and the two places the lockfile
    # repeats the version. --no-git-tag-version keeps it out of git entirely.
    if ! (cd web && npm version "$new" --no-git-tag-version --allow-same-version >/dev/null 2>&1); then
        git checkout -- Cargo.toml Cargo.lock 2>/dev/null
        die "npm could not set the Playground version (nothing changed)"
    fi

    echo "$old -> $new"
    show
    echo
    echo "next:"
    echo "  git commit -am 'ignis $new' && git push origin main"
    echo "  git tag -a v$new -m 'ignis $new' && git push origin v$new   # builds and publishes the release"
}

bump() {
    local part="$1" current major minor patch pre
    current="$(cargo_version)"
    [[ "$current" =~ $version_re ]] || die "the workspace version '$current' is not x.y.z -- use V=<version>"
    pre="${current#*-}"
    [ "$pre" = "$current" ] && pre=""
    local core="${current%%-*}"
    core="${core%%+*}"
    IFS=. read -r major minor patch <<<"$core"
    case "$part" in
        # A prerelease bumps to its own release, the way `npm version patch`
        # does: 1.2.3-rc.1 patches to 1.2.3, not 1.2.4.
        patch) [ -n "$pre" ] || patch=$((patch + 1)) ;;
        minor) minor=$((minor + 1)); patch=0 ;;
        major) major=$((major + 1)); minor=0; patch=0 ;;
        *) die "unknown part '$part' (expected patch, minor or major)" ;;
    esac
    write_version "$major.$minor.$patch"
}

action="${1:-show}"
case "$action" in
    show) show ;;
    check) check ;;
    set)
        [ $# -ge 2 ] || die "set needs a version"
        [[ "$2" =~ $version_re ]] ||
            die "'$2' is not a version: expected x.y.z, optionally -prerelease and +build (e.g. 1.2.3, 1.2.3-rc.1, 1.2.3+cuda13)"
        require_clean
        write_version "$2"
        ;;
    bump)
        [ $# -ge 2 ] || die "bump needs patch, minor or major"
        require_clean
        bump "$2"
        ;;
    *) die "unknown action '$action' (expected show, check, set or bump)" ;;
esac
