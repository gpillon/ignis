#!/usr/bin/env bash
# The behavioural test of mk/changelog.sh (`make changelog-selftest`), which
# is docs/agents/testing.md's rule applied to a script.
#
# It builds a throwaway repository under a temp directory and runs the real
# script inside it, so it touches neither this repo's files nor its history --
# unlike mk/version-selftest.sh, which has to drive the four real version
# files. Nothing here reaches GitHub either: `gh` is shadowed by a stub on
# PATH that answers from a fixture, so the test is offline, deterministic,
# and also covers the gh-says-nothing path a machine without a token takes.
#
# What that stub cannot cover is the one seam outside this file: the `--jq`
# expression the real gh is asked to print with. The stub answers with a
# fixture already in the shape that expression produces, so a wrong
# expression passes here -- and one did. `.stateReason` is the empty string
# on an open issue rather than null, `//` does not replace an empty string,
# and every open issue came out titleless until the expression was checked by
# hand against a live gh. Change that line and check it the same way; there
# is no standalone jq to run it through, gh embeds its own.
#
# What it pins is the classification, which is the whole of this script's
# judgement: a commit in the range puts its issue under Changes, the same
# issue mentioned before the range makes it a follow-up reported by its
# commit, an issue closed in the window that nothing mentions is held back,
# a NOT_PLANNED issue is not a release note at all, a commit naming no issue
# is listed rather than dropped, and every ADR is named "ADR NNNN — title"
# whether its heading carries the number or is the bare title of the short
# format. Since ADR 0081 it also pins when a modified ADR is a decision: when
# its text changed, not its frontmatter or a link's target, and when its
# status changed, said as "(now ...)"; the generated index never is.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo/mk/changelog.sh"
[ -f "$script" ] || { echo "error: $script is missing" >&2; exit 1; }

failed=0
note() { printf '  %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

work="$(mktemp -d 2>/dev/null)" || { echo "error: no temp directory" >&2; exit 1; }
trap 'rm -rf "$work"' EXIT

# The fixture the gh stub answers with: the four shapes that have to be told
# apart. Tab-separated the way `gh issue list --jq` is asked to print.
mkdir -p "$work/bin"
cat > "$work/issues.tsv" <<'FIXTURE'
10	CLOSED	-	2026-01-02T00:00:00Z	An issue this range closes
20	CLOSED	-	2026-01-02T00:00:00Z	An issue an earlier release carried
30	CLOSED	-	2026-01-02T00:00:00Z	An issue closed with nothing pointing at it
40	CLOSED	NOT_PLANNED	2026-01-02T00:00:00Z	A duplicate that is not a release note
50	OPEN	-	-	An open issue no commit points at
FIXTURE

# `gh issue list --jq '...'` is the only call the script makes. The stub
# ignores the query and prints the fixture, so what is under test below is
# every judgement the script makes about those rows -- not the expression
# that produced them, which is the seam named in the header.
cat > "$work/bin/gh" <<'STUB'
#!/usr/bin/env bash
case "$*" in
    *"issue list"*) cat "$GH_FIXTURE" ;;
    *) exit 1 ;;
esac
STUB
chmod +x "$work/bin/gh"

# A repository with one tag and a range after it, covering every case above.
export GIT_AUTHOR_NAME=selftest GIT_AUTHOR_EMAIL=selftest@example.invalid
export GIT_COMMITTER_NAME=selftest GIT_COMMITTER_EMAIL=selftest@example.invalid
# The window an issue is judged by runs from the commit `from` points at, so
# the dates are fixed here rather than left at "now": the fixture closes its
# issues on 2026-01-02 and that has to fall inside the range either way.
export GIT_AUTHOR_DATE='2026-01-01T00:00:00Z' GIT_COMMITTER_DATE='2026-01-01T00:00:00Z'
# The script resolves its repository from its own location, the way
# mk/version.sh does, so the copy under test has to live in the sandbox.
sandbox="$work/repo"
mkdir -p "$sandbox/docs/adr" "$sandbox/mk"
cp "$script" "$sandbox/mk/changelog.sh"
script="$sandbox/mk/changelog.sh"
cd "$sandbox" || exit 1
git init -q .
printf '[workspace.package]\nversion = "9.9.9"\n' > Cargo.toml

# Every commit has to change something, so each one leaves its own file.
serial=0
commit() {
    serial=$((serial + 1))
    printf '%s\n' "$1" > "file-$serial.txt"
    git add -A .
    git commit -q -m "$1"
}

printf '# ADR 0001 — a decision taken before the range\n' > docs/adr/0001-early.md
# The ADRs whose frontmatter the range changes (ADR 0081): the long form's
# Status section, a link, a short one's frontmatter status.
printf '# ADR 0004 — a long-form decision\n\n## Status\n\nAccepted (2026-01-01).\n' \
    > docs/adr/0004-long.md
printf '# ADR 0005 — a decision that links\n\nSee [the finding](../findings/a.md) and `docs/a.md`.\n' \
    > docs/adr/0005-linked.md
printf '# ADR 0006 — a decision the range retires\n\n## Status\n\nAccepted.\n' \
    > docs/adr/0006-retired.md
printf '%s\n' '---' 'status: proposed' '---' '' '# A proposed decision' > docs/adr/0007-proposed.md
printf '%s\n' '---' 'status: accepted' '---' '' '# A decision deprecated with a reason' \
    > docs/adr/0008-deprecated.md
printf '# ADR 0009 — a decision superseded already\n\n## Status\n\nSuperseded by ADR 0002.\n' \
    > docs/adr/0009-superseded.md
commit "Groundwork, and the earlier half of #20"
git tag -a v9.9.8 -m v9.9.8
export GIT_AUTHOR_DATE='2026-01-03T00:00:00Z' GIT_COMMITTER_DATE='2026-01-03T00:00:00Z'

# In the range: one issue it closes, one it only follows up, one ADR added,
# one amended, one commit naming nothing, and two the script must skip.
commit "The change this release is about

GitHub #10."
commit "A follow-up to what shipped already

GitHub #20."
commit "Housekeeping with no issue behind it"
printf '# ADR 0002 — a decision taken inside the range\n' > docs/adr/0002-inside.md
commit "The decision, recorded (#10)"
# The short format's heading is the title alone; the number is the file's.
printf '# A decision in the short format\n\nIts context, the decision and why.\n' > docs/adr/0003-short.md
commit "A decision in the short format, recorded (#10)"
printf '# ADR 0001 — a decision taken before the range\n\nAmended.\n' > docs/adr/0001-early.md
commit "The earlier decision, amended (#10)"
# Frontmatter only, as ADR 0081 changes an ADR a later one amends or replaces:
# a status the Status section already said, a link's target, a back-link.
printf '%s\n' '---' 'status: accepted' 'amended-by: [ADR-0002]' '---' '' \
    '# ADR 0004 — a long-form decision' '' '## Status' '' 'Accepted (2026-01-01).' \
    > docs/adr/0004-long.md
printf '%s\n' '---' 'status: accepted' '---' '' '# ADR 0005 — a decision that links' '' \
    'See [the finding](../findings/archive/a.md) and `docs/archive/a.md`.' > docs/adr/0005-linked.md
printf '%s\n' '---' 'status: superseded by ADR-0002' '---' '' \
    '# ADR 0006 — a decision the range retires' '' '## Status' '' 'Accepted.' \
    > docs/adr/0006-retired.md
printf '%s\n' '---' 'status: accepted' '---' '' '# A proposed decision' > docs/adr/0007-proposed.md
printf '%s\n' '---' 'status: deprecated' '---' '' '# A decision deprecated with a reason' '' \
    'The reason.' > docs/adr/0008-deprecated.md
printf '%s\n' '---' 'status: superseded by ADR-0002' '---' '' \
    '# ADR 0009 — a decision superseded already' '' '## Status' '' 'Superseded by ADR 0002.' \
    > docs/adr/0009-superseded.md
# The generated index is not a decision.
printf '# Architecture decision records\n\n| ADR |\n' > docs/adr/README.md
commit "The frontmatter, and the index (#10)"
commit "The duplicate, mentioned anyway (#40)"
commit "Merge: a branch"
commit "ignis 9.9.9"

out="$work/draft.md"
PATH="$work/bin:$PATH" GH_FIXTURE="$work/issues.tsv" \
    bash "$script" v9.9.8 HEAD > "$out" 2> "$work/err.txt"
status=$?
[ "$status" -eq 0 ] || fail "the script exited $status -- $(cat "$work/err.txt")"

# Everything below reads the draft rather than trusting the exit status: the
# point is to catch a draft that files a line under the wrong heading.
section() {
    awk -v want="$1" '
        /^### / { inside = ($0 == "### " want); next }
        inside && NF { print }' "$out"
}

changes="$(section "Changes")"
follow="$(section "Follow-ups to issues an earlier release already carried")"
held="$(section "Closed in the window, but no commit in the range mentions them")"
decisions="$(section "Decisions")"
loose="$(section "Commits in the range that name no issue")"

has() { printf '%s\n' "$2" | grep -qF -- "$1"; }

echo "changelog-selftest"

has "An issue this range closes (#10)" "$changes" \
    && note "a commit in the range files its issue under Changes" \
    || fail "#10 is not under Changes: $changes"

has "A follow-up to what shipped already (#20)" "$follow" \
    && note "an issue an earlier range carried is a follow-up, named by its commit" \
    || fail "#20 is not a follow-up reported by its commit: $follow"

has "(#20)" "$changes" && fail "#20 is under Changes as well as Follow-ups"

has "An issue closed with nothing pointing at it (#30)" "$held" \
    && note "an unmentioned issue closed in the window is held back, not listed" \
    || fail "#30 is not held back: $held"

has "#40" "$changes$follow$held" \
    && fail "the NOT_PLANNED issue #40 reached the draft" \
    || note "a NOT_PLANNED issue is not a release note"

has "#50" "$changes$follow$held" \
    && fail "the open issue #50, which no commit mentions, reached the draft" \
    || note "an open issue nothing points at stays out"

has "ADR 0002 — a decision taken inside the range" "$decisions" \
    && note "an ADR added in the range is a decision" \
    || fail "the added ADR is missing: $decisions"

has "ADR 0001 — a decision taken before the range (amended)" "$decisions" \
    && note "an ADR modified in the range is marked amended" \
    || fail "the amended ADR is not marked: $decisions"

has "ADR 0003 — A decision in the short format" "$decisions" \
    && note "an ADR whose heading is a bare title is named by its file's number" \
    || fail "the bare-title ADR is not named by its number: $decisions"

has "ADR 0002 — ADR 0002" "$decisions" \
    && fail "an ADR whose heading carries its number got it twice: $decisions"

has "ADR 0004" "$decisions" \
    && fail "a frontmatter that restates the Status section made an ADR amended: $decisions" \
    || note "an ADR given the frontmatter its Status section already said is not listed"

has "ADR 0005" "$decisions" \
    && fail "a link's new target made an ADR amended: $decisions" \
    || note "a link's moved target is not an amendment"

has "ADR 0006 — a decision the range retires (now superseded by ADR 0002)" "$decisions" \
    && note "a status changed from the Status section's is said, not called amended" \
    || fail "the superseded ADR is not reported by its new status: $decisions"

has "ADR 0007 — A proposed decision (now accepted)" "$decisions" \
    && note "a frontmatter status that changed is said" \
    || fail "the accepted ADR is not reported by its new status: $decisions"

has "ADR 0008 — A decision deprecated with a reason (amended, now deprecated)" "$decisions" \
    && note "a text and a status that both changed are both said" \
    || fail "the deprecated ADR is not both amended and deprecated: $decisions"

has "ADR 0009" "$decisions" \
    && fail "a Status section's Superseded read as a change of status: $decisions" \
    || note "a Status section's Superseded is the frontmatter's superseded by"

has "Architecture decision records" "$decisions" \
    && fail "the generated index reached the Decisions: $decisions" \
    || note "docs/adr/README.md is not a decision"

has "Housekeeping with no issue behind it" "$loose" \
    && note "a commit naming no issue is listed, not dropped" \
    || fail "the unreferenced commit is missing: $loose"

has "Merge: a branch" "$loose" && fail "a merge commit reached the draft"
has "ignis 9.9.9" "$loose" && fail "the version bump commit reached the draft"
grep -q '^## ignis v9.9.9$' "$out" \
    && note "an untagged end takes its heading from the workspace version" \
    || fail "the heading is not the workspace version: $(head -1 "$out")"

# The same range with a tag on its end names the tag, not the workspace.
git tag -a v9.9.9 -m v9.9.9
PATH="$work/bin:$PATH" GH_FIXTURE="$work/issues.tsv" \
    bash "$script" v9.9.8 v9.9.9 > "$work/tagged.md" 2>/dev/null
grep -q '^## ignis v9.9.9$' "$work/tagged.md" \
    && note "a tagged end names the tag" \
    || fail "the tagged heading is wrong: $(head -1 "$work/tagged.md")"

# A gh that answers nothing -- not logged in, no network, rate limited. The
# git-derived half still has to come out: a draft missing its issues is worth
# having, a script that dies on a machine without a token is not.
mkdir -p "$work/bin-mute"
printf '#!/usr/bin/env bash\nexit 1\n' > "$work/bin-mute/gh"
chmod +x "$work/bin-mute/gh"
PATH="$work/bin-mute:$PATH" bash "$script" v9.9.8 v9.9.9 > "$work/nogh.md" 2> "$work/nogh-err.txt"
status=$?
if [ "$status" -ne 0 ]; then
    fail "with a silent gh the script exited $status -- $(cat "$work/nogh-err.txt")"
elif grep -q 'ADR 0002' "$work/nogh.md" && grep -q 'Housekeeping' "$work/nogh.md"; then
    note "with a silent gh the ADRs and the commits still come out"
else
    fail "with a silent gh the git-derived sections are missing"
fi

# A range whose ADRs gained only a back-link has no Decisions section at all,
# rather than a heading over nothing.
printf '%s\n' '---' 'status: accepted' 'amended-by: [ADR-0003]' '---' '' '# A proposed decision' \
    > docs/adr/0007-proposed.md
commit "A back-link (#10)"
PATH="$work/bin:$PATH" GH_FIXTURE="$work/issues.tsv" \
    bash "$script" v9.9.9 HEAD > "$work/backlink.md" 2>/dev/null
grep -q '^### Decisions$' "$work/backlink.md" \
    && fail "a range of frontmatter edits alone has a Decisions section: $(cat "$work/backlink.md")" \
    || note "a range whose ADRs changed only in their frontmatter has no Decisions section"

# A range with a bad end is an error, not an empty draft that looks fine.
PATH="$work/bin:$PATH" bash "$script" v9.9.8 no-such-ref > /dev/null 2>&1
[ $? -ne 0 ] && note "an unknown ref is refused" || fail "an unknown ref was accepted"

echo
if [ "$failed" -eq 0 ]; then
    echo "changelog-selftest: ok"
else
    echo "changelog-selftest: $failed failed" >&2
fi
exit "$failed"
