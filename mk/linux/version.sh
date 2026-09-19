#!/usr/bin/env bash
# The release version, for the Makefile (the Linux VERSION_TOOL hook,
# mk/os/linux.mk). The counterpart of mk/windows/version.ps1 -- same three
# actions, same files, same output.
#
#   show           print the version each file declares
#   check          exit 1 when they disagree: the comparison
#                  .github/workflows/release.yml makes before it builds
#                  anything (Cargo.toml against web/package.json), runnable
#                  before you tag -- plus the two lockfiles, which the
#                  workflow does not look at and the next build would rewrite
#   set <version>  write an explicit version, `x.y.z` with an optional
#                  `-prerelease`
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

# Put back what a failed step already wrote, and say so either way: a rollback
# that quietly failed would leave a half-bumped tree behind a message claiming
# it did not.
restore() {
    if git checkout -- "$@" 2>/dev/null; then
        echo "restored: $*" >&2
    else
        echo "warning: could not restore $* -- check them by hand" >&2
    fi
}

# `x.y.z`, optionally `-prerelease`. Semver's `+build` is deliberately not
# accepted: `npm version` drops build metadata, so it would land in the two
# Rust files and not in the two web ones -- and the release workflow compares
# those two by string, so such a version could never tag at all.
version_re='^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$'

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

# What the lockfile says the workspace members are at: every `ignis-*`
# entry's version, deduplicated. One value when they agree -- which is what
# `check` wants to compare -- and a `/`-joined list when they do not, so a
# single stale member is visible rather than hidden behind the first one.
cargo_lock_version() {
    awk '/^name = "ignis-[a-z-]*"$/ { getline; sub(/^version = "/, ""); sub(/"$/, ""); print }' Cargo.lock |
        sort -u | paste -sd/ -
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

# Say so when those files already carry changes, and carry on. It was a
# refusal at first, and that was wrong twice over: a second bump in the same
# session is ordinary, and repairing a tree a half-done bump left is the case
# this tool exists for. What the refusal was really guarding -- a release
# commit sweeping up unrelated work -- is handled where it belongs, by the
# `git commit` line printed below naming its four files.
warn_if_dirty() {
    local dirty
    dirty="$(git status --porcelain -- Cargo.toml Cargo.lock web/package.json web/package-lock.json 2>/dev/null)"
    [ -z "$dirty" ] || echo "note: the version files already carry changes:
$dirty" >&2
}

write_version() {
    local new="$1" old
    old="$(cargo_version)"
    [ -n "$old" ] || die "no version in Cargo.toml [workspace.package]"
    # Refuse only when every file already says it. Setting the version a
    # disagreeing tree half-carries is the repair this exists for -- that is
    # the state a half-done bump leaves, and what the failed v0.1.1 tag was.
    if [ "$new" = "$old" ] && [ "$new" = "$(cargo_lock_version)" ] &&
        [ "$new" = "$(web_version)" ] && [ "$new" = "$(web_lock_version)" ]; then
        die "every file already declares $new"
    fi

    # Only inside [workspace.package]: `version = "1"` appears under a dozen
    # dependencies in the same file.
    if ! awk -v new="$new" '
        /^\[workspace\.package\]$/ { in_section = 1; print; next }
        /^\[/ { in_section = 0 }
        in_section && /^version *= *"/ { print "version = \"" new "\""; next }
        { print }
    ' Cargo.toml > Cargo.toml.tmp; then
        rm -f Cargo.toml.tmp
        die "could not rewrite Cargo.toml (nothing changed)"
    fi
    mv Cargo.toml.tmp Cargo.toml || die "could not replace Cargo.toml"

    # Cargo owns its lockfile: --offline touches nothing but the workspace
    # members' own entries, and never reaches the network.
    if ! cargo update --workspace --offline >/dev/null 2>&1; then
        restore Cargo.toml
        die "cargo could not refresh Cargo.lock"
    fi

    # npm owns both web files: package.json and the two places the lockfile
    # repeats the version. --no-git-tag-version keeps it out of git entirely.
    if ! (cd web && npm version "$new" --no-git-tag-version --allow-same-version >/dev/null 2>&1); then
        restore Cargo.toml Cargo.lock
        die "npm could not set the Playground version"
    fi

    # Never report a bump the release workflow would refuse: the two tools
    # above each normalize what they are given (npm drops build metadata,
    # for one), so what they wrote is checked rather than assumed.
    for file_version in "$(cargo_version)" "$(cargo_lock_version)" "$(web_version)" "$(web_lock_version)"; do
        if [ "$file_version" != "$new" ]; then
            show
            die "the files did not all take $new -- fix them before tagging"
        fi
    done

    if [ "$old" = "$new" ]; then
        echo "repaired $new"
    else
        echo "$old -> $new"
    fi
    show
    echo
    echo "next:"
    echo "  git commit -m 'ignis $new' -- Cargo.toml Cargo.lock web/package.json web/package-lock.json && git push origin main"
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
            die "'$2' is not a version: expected x.y.z, optionally -prerelease (e.g. 1.2.3, 1.2.3-rc.1). Semver +build metadata is not accepted -- npm drops it, and the release workflow compares Cargo.toml against web/package.json by string"
        warn_if_dirty
        write_version "$2"
        ;;
    bump)
        [ $# -ge 2 ] || die "bump needs patch, minor or major"
        warn_if_dirty
        bump "$2"
        ;;
    *) die "unknown action '$action' (expected show, check, set or bump)" ;;
esac
