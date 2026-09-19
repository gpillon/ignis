#!/usr/bin/env bash
# The release notes, drafted from what the repo already records.
#
#   bash mk/changelog.sh [<from> [<to>]]
#
# `from` defaults to the last v* tag reachable from HEAD, `to` to HEAD, so a
# bare run drafts the notes for the release you are about to cut.
#
# The draft is Markdown, ready for the `body:` of .github/workflows/release.yml
# and for the body of the version-bump commit. It is a draft and nothing more:
# `make version-bump` prints it next to the commands it does not run either,
# because a release note is a sentence somebody writes, not a list a script
# emits (#230 is a title; "the text body limit sat under one max-context
# prompt" is the same fact said better).
#
# Why issues and not `git log`: the commit subjects here are prose, but the
# range mixes what an operator reads a release for ("The Playground is served
# unless you say otherwise") with what only the next contributor cares about
# ("The GPU harnesses catch up with two signatures they never recompiled").
# A closed issue is already the user-facing half, written as one line by the
# convention in docs/agents/issue-tracker.md. GitHub's own
# `generate_release_notes` cannot do this: it lists merged PRs and this repo
# merges locally, so it produces exactly one line, the compare link.
#
# Which issues, in order of what the range can actually prove:
#
#   the range's own commits   every #NNN a commit between the two ends
#                             mentions. This is the list that is IN the
#                             release, because a commit is in it by
#                             construction. An issue closed long before the
#                             window still lands here when the work shipped
#                             now -- #163 was closed in September and the
#                             Playground became the default in v0.1.2.
#   the closing window        issues closed between the two ends that no
#                             commit mentions. Held back under their own
#                             heading rather than listed as changes: this
#                             repo closes issues on branches that are not
#                             merged yet, and such an issue is not in the
#                             release.
#
# ADRs get their own section. A decision amended or recorded in the window is
# the part of a release that outlives its version number. Commits that name
# no issue at all come last, so the per-issue list can be read for what it
# dropped.
#
# Nothing here writes a file or touches git state.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo" || exit 1

die() { echo "error: $*" >&2; exit 1; }

from="${1:-}"
to="${2:-HEAD}"

if [ -z "$from" ]; then
    from="$(git describe --tags --abbrev=0 --match 'v*' "$to" 2>/dev/null)"
    [ -n "$from" ] || die "no v* tag reachable from $to -- name the range: mk/changelog.sh <from> [<to>]"
fi

git rev-parse --verify --quiet "$from^{commit}" >/dev/null || die "not a commit: $from"
git rev-parse --verify --quiet "$to^{commit}" >/dev/null || die "not a commit: $to"

# The window the closed issues are filtered by. Both ends in UTC with a
# trailing Z, which is the shape gh reports `closedAt` in -- ISO 8601 in one
# zone compares as a string, and a %cI carrying a local offset would not.
commit_utc() {
    TZ=UTC0 git log -1 --format=%cd --date=format-local:'%Y-%m-%dT%H:%M:%SZ' "$1"
}

since="$(commit_utc "$from")"
# An issue closed after the last commit but before the tag belongs to the
# release: the bump commit does not exist yet when this runs. So the window
# ends now whenever `to` is HEAD, and at the commit otherwise.
if [ "$(git rev-parse "$to")" = "$(git rev-parse HEAD)" ]; then
    until_="$(TZ=UTC0 date '+%Y-%m-%dT%H:%M:%SZ')"
else
    until_="$(commit_utc "$to")"
fi

# The heading names the release the range ends at. A tag names itself; an
# untagged end is the release being prepared, which is what the workspace
# declares -- `make version-bump` has already written it by the time this runs.
title="$(git describe --exact-match --tags --match 'v*' "$to" 2>/dev/null)"
if [ -z "$title" ]; then
    title="v$(sed -n '/^\[workspace\.package\]/,/^\[/p' Cargo.toml |
        sed -n 's/^version *= *"\(.*\)"/\1/p' | head -1)"
fi

# Is this commit one of the two the range carries that are not changes?
# A merge, or the version bump itself.
is_plumbing() {
    case "$1" in
        Merge*|"ignis "[0-9]*.[0-9]*.[0-9]*) return 0 ;;
        *) return 1 ;;
    esac
}

# Walk the range once: the issue numbers its commits mention, and the commits
# that mention none. `#12` and `GitHub #12` are the two spellings in this
# history and the number is what matters, so the prefix is not part of the
# match. Subject and body both, because most commits carry the reference in a
# closing line.
mentioned=""
loose=""
by_issue=""
while read -r sha; do
    [ -n "$sha" ] || continue
    subject="$(git log -1 --format=%s "$sha")"
    refs="$(git log -1 --format='%s%n%b' "$sha" | grep -oE '#[0-9]+' | tr -d '#' | sort -u)"
    if [ -n "$refs" ]; then
        mentioned="$mentioned$refs"$'\n'
        # Keep what each commit did next to the issue it names: a follow-up
        # is reported by its commit, because the issue's own title describes
        # the release that carried it first.
        is_plumbing "$subject" || while read -r ref; do
            by_issue="$by_issue$ref"$'\t'"$subject"$'\n'
        done <<< "$refs"
    elif ! is_plumbing "$subject"; then
        loose="$loose$sha"$'\t'"$subject"$'\n'
    fi
done < <(git log --format=%h "$from..$to")
mentioned="$(printf '%s' "$mentioned" | sort -un)"

# Every issue the history up to `from` already mentioned. An issue in both
# sets shipped in an earlier release and came back for a follow-up -- #227
# was YaRN in v0.1.1 and two recompiled GPU call sites in v0.1.2, and only
# the first of those is what a reader of v0.1.2 wants under "Changes".
released="$(git log --format='%s%n%b' "$from" | grep -oE '#[0-9]+' | tr -d '#' | sort -u)"

echo "## ignis $title"
echo
echo "<!-- draft: $from..$to, issues closed $since .. $until_ -->"

# One call for every issue the repo has: the range's references need a title
# whatever their state, and a second query per number would be one round trip
# each. gh's --jq is jq, so no jq on PATH is needed.
catalog=""
if command -v gh >/dev/null 2>&1; then
    catalog="$(gh issue list --state all --limit 600 \
        --json number,title,state,stateReason,closedAt \
        --jq '.[] | [(.number | tostring), .state,
                     ((.stateReason // "") | if . == "" then "-" else . end),
                     ((.closedAt // "") | if . == "" then "-" else . end),
                     .title] | join("\t")' 2>/dev/null)"
    if [ -z "$catalog" ]; then
        echo "<!-- gh listed no issue: the issue sections are missing, ADRs and commits are not -->"
    elif [ "$(printf '%s\n' "$catalog" | wc -l)" -ge 600 ]; then
        # gh returns the most recent first, so a truncated catalog loses the
        # oldest issues -- which are exactly the ones a follow-up looks up.
        echo "<!-- the issue catalog hit the 600 limit: an old issue may be missing its title -->"
    fi
else
    echo "<!-- gh is not on PATH: the issue sections are missing, ADRs and commits are not -->"
fi

# A field this catalog does not have is a "-" and never an empty one: bash
# collapses runs of tabs when it splits on them (a tab is IFS whitespace), so
# an issue with no stateReason would shift every field after it by one.
lookup() { printf '%s\n' "$catalog" | awk -F'\t' -v n="$1" '$1 == n { print; exit }'; }

changes=""
again=""
for number in $mentioned; do
    row="$(lookup "$number")"
    # A number with no issue behind it is a PR reference or a typo, and a
    # title this script cannot print is not worth a line that says so.
    [ -n "$row" ] || continue
    IFS=$'\t' read -r _ state reason _ subject <<< "$row"
    case "$state$reason" in
        # A duplicate or a wontfix that a commit happens to mention is not a
        # change -- #229 was closed as a duplicate of #230 and only one of
        # the two is a release note.
        *NOT_PLANNED) continue ;;
    esac
    if printf '%s\n' "$released" | grep -qx "$number"; then
        # What this range did to it, one line per commit -- the issue title
        # belongs to the release that closed it the first time.
        while IFS=$'\t' read -r ref did; do
            [ "$ref" = "$number" ] || continue
            again="$again$did"$'\t'"$number"$'\n'
        done <<< "$by_issue"
    elif [ "$state" = "OPEN" ]; then
        changes="$changes- $subject (#$number, still open)"$'\n'
    else
        changes="$changes- $subject (#$number)"$'\n'
    fi
done

held=""
while IFS=$'\t' read -r number state reason closed subject; do
    [ -n "$number" ] || continue
    [ "$state" = "CLOSED" ] || continue
    [ "$reason" = "NOT_PLANNED" ] && continue
    [[ "$closed" > "$since" && "$closed" < "$until_" ]] || continue
    printf '%s\n' "$mentioned" | grep -qx "$number" && continue
    # Closed late, shipped early: an issue an earlier range already carried
    # needs no second look just because it was closed inside this window.
    printf '%s\n' "$released" | grep -qx "$number" && continue
    held="$held- $subject (#$number)"$'\n'
done <<< "$catalog"

if [ -n "$changes" ]; then
    echo
    echo "### Changes"
    echo
    printf '%s' "$changes"
fi

if [ -n "$again" ]; then
    echo
    echo "### Follow-ups to issues an earlier release already carried"
    echo
    # One line per commit, carrying every issue it named: a single commit
    # often catches up two of them at once.
    printf '%s' "$again" | awk -F'\t' '
        NF == 2 { if (!($1 in seen)) { order[++n] = $1; seen[$1] = "" }
                  if (index(seen[$1], "#" $2 ",") == 0) seen[$1] = seen[$1] "#" $2 "," }
        END { for (i = 1; i <= n; i++) {
                  refs = substr(seen[order[i]], 1, length(seen[order[i]]) - 1)
                  gsub(/,/, ", ", refs)
                  print "- " order[i] " (" refs ")" } }'
fi

if [ -n "$held" ]; then
    echo
    echo "### Closed in the window, but no commit in the range mentions them"
    echo
    printf '%s' "$held"
    echo "(Check each one: this repo closes issues on branches that are not merged yet.)"
fi

# Decisions. A file added in the range is a new ADR, a file modified is one
# amended -- ADR 0026 was amended in v0.1.2 and that is the shape it takes.
adrs="$(git diff --name-status "$from..$to" -- docs/adr/ 2>/dev/null | grep -E '^[AM]')"
if [ -n "$adrs" ]; then
    echo
    echo "### Decisions"
    echo
    while IFS=$'\t' read -r status path; do
        [ -n "$path" ] || continue
        heading="$(git show "$to:$path" 2>/dev/null | sed -n 's/^# *//p' | head -1)"
        [ -n "$heading" ] || heading="$path"
        case "$status" in
            A*) echo "- $heading" ;;
            M*) echo "- $heading (amended)" ;;
        esac
    done <<< "$adrs"
fi

# What the per-issue list cannot have covered: either housekeeping, or a
# change whose issue this draft is missing.
if [ -n "$loose" ]; then
    echo
    echo "### Commits in the range that name no issue"
    echo
    while IFS=$'\t' read -r sha subject; do
        [ -n "$sha" ] || continue
        echo "- $subject ($sha)"
    done <<< "$loose"
fi
