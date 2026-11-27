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
# format. It also pins when an ADR the range changes is a decision: when its
# text changed, not its frontmatter or a link's target, by a commit that
# carries no mark for it (`[adr-cleanup]`, `[typo ADR-NNNN]`), a merge by its
# own resolution and not by what it brings in, under whatever
# name its file has, when its status changed, said as "(now ...)", and when
# its file left the tree, said as "(removed)" unless a marked commit deleted
# it; the generated index never is.

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
# The ADRs whose frontmatter the range changes: the long form's
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
printf '%s\n' '---' 'status: accepted' '---' '' '# A decision with a typo' '' 'Its reasno.' \
    > docs/adr/0010-typo.md
# The ADRs the range rewrites in place (ADR 0081), each long before it.
adr() { # <file> <title> <text>
    printf '%s\n' '---' 'status: accepted' '---' '' "# $2" '' "$3" > "docs/adr/$1"
}
adr 0011-cleaned.md 'A decision the cleanup shortens' 'Its decision, with its history.'
adr 0012-cleaned-too.md 'A second decision the cleanup shortens' 'Its decision, with its history.'
adr 0013-cleaned-then-changed.md 'A decision cleaned, then changed' 'Its decision, with its history.'
adr 0014-old-title.md 'A decision under its old title' 'Its old decision.'
adr 0015-old-name.md 'A decision the cleanup retitles' 'Its decision, with its history.'
adr 0016-typo-then-changed.md 'A decision with a typo, then changed' 'Its reasno.'
# The ADRs the range deletes.
adr 0017-retired.md 'A decision the range retires and removes' 'Its decision.'
adr 0018-absorbed.md 'A decision the cleanup absorbs into another' 'Its decision.'
# The ADRs a merge brings in, their first and last rules far enough apart that
# two branches change one each and merge with no conflict.
rules() { # <file> <title> <first rule> <last rule>
    printf '%s\n' '---' 'status: accepted' '---' '' "# $2" '' "$3" '' 'A second rule.' '' \
        'A third rule.' '' 'A fourth rule.' '' "$4" > "docs/adr/$1"
}
rules 0019-merged-cleanup.md 'A decision two cleanup commits meet on' \
    'Its first rule, with history.' 'Its last rule, with history.'
rules 0020-merge-resolved.md 'A decision a merge resolves' 'Its first rule.' 'Its last rule.'
rules 0021-merge-resolved-marked.md 'A decision a marked merge resolves' 'Its first rule.' \
    'Its last rule.'
rules 0022-changed-on-a-branch.md 'A decision changed on a merged branch' 'Its first rule.' \
    'Its last rule, with history.'
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
# Frontmatter only, as an ADR changes when a later one amends or replaces it:
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
# A typo fix its commit names is not an amendment (mk/adr-index.py --since).
printf '%s\n' '---' 'status: accepted' '---' '' '# A decision with a typo' '' 'Its reason.' \
    > docs/adr/0010-typo.md
commit "A typo in a decision, fixed [typo ADR-0010] (#10)"
# A cleanup commit changes no decision: its mark keeps every ADR it rewrites
# out of the amended list, and a file it renames with its title too.
adr 0011-cleaned.md 'A decision the cleanup shortens' 'Its decision.'
adr 0012-cleaned-too.md 'A second decision the cleanup shortens' 'Its decision.'
adr 0013-cleaned-then-changed.md 'A decision cleaned, then changed' 'Its decision.'
git rm -q docs/adr/0015-old-name.md
adr 0015-new-name.md 'A decision the cleanup retitled' 'Its decision.'
git rm -q docs/adr/0018-absorbed.md
commit "Four decisions, shortened to what holds today [adr-cleanup] (#10)"
# A decision that changes rewrites its ADR in place, with no mark: amended,
# whether a marked commit touched the ADR before it, and under its new name.
adr 0013-cleaned-then-changed.md 'A decision cleaned, then changed' 'Another decision.'
git rm -q docs/adr/0014-old-title.md
adr 0014-new-title.md 'A decision under its new title' 'A new decision, and why.'
commit "Two decisions, changed (#10)"
adr 0016-typo-then-changed.md 'A decision with a typo, then changed' 'Its reason.'
commit "A typo in a decision, fixed [typo ADR-0016] (#10)"
adr 0016-typo-then-changed.md 'A decision with a typo, then changed' 'Another reason.'
commit "The decision's reason, changed (#10)"
# An ADR that leaves the tree is said, unless a cleanup commit deletes it.
git rm -q docs/adr/0017-retired.md
commit "A decision, retired and removed (#10)"
# A merge is judged by what it changes against both parents, its own
# resolution, which its own mark covers; what it only brings in was judged on
# the branch. Each branch here leaves the trunk, changes an ADR the trunk
# changes too, and comes back.
trunk="$(git rev-parse --abbrev-ref HEAD)"
git switch -q -c cleanup-a
rules 0019-merged-cleanup.md 'A decision two cleanup commits meet on' 'Its first rule.' \
    'Its last rule, with history.'
commit "A branch shortens a decision (#10) [adr-cleanup]"
git switch -q "$trunk"
git switch -q -c change-b
rules 0022-changed-on-a-branch.md 'A decision changed on a merged branch' 'Another first rule.' \
    'Its last rule, with history.'
commit "A branch changes a decision (#10)"
git switch -q "$trunk"
rules 0019-merged-cleanup.md 'A decision two cleanup commits meet on' \
    'Its first rule, with history.' 'Its last rule.'
rules 0022-changed-on-a-branch.md 'A decision changed on a merged branch' 'Its first rule.' \
    'Its last rule.'
commit "The trunk shortens two decisions (#10) [adr-cleanup]"
git merge -q --no-ff -m "Merge the cleanup branch (#10)" cleanup-a >/dev/null
git merge -q --no-ff -m "Merge the branch that changed a decision (#10) [adr-cleanup]" change-b >/dev/null
# Two branches change one rule each way, and the merge's own resolution is a
# third text: a decision changed by the merge unless its message has the mark.
for marked in "" " [adr-cleanup]"; do
    file=0020-merge-resolved.md title='A decision a merge resolves'
    [ -z "$marked" ] || file=0021-merge-resolved-marked.md title='A decision a marked merge resolves'
    git switch -q -c "resolve-${file%%-*}"
    rules "$file" "$title" 'Its first rule, one way.' 'Its last rule.'
    commit "A branch shortens a decision one way (#10) [adr-cleanup]"
    git switch -q "$trunk"
    rules "$file" "$title" 'Its first rule, another way.' 'Its last rule.'
    commit "The trunk shortens it another way (#10) [adr-cleanup]"
    git merge -q --no-ff "resolve-${file%%-*}" >/dev/null 2>&1
    rules "$file" "$title" 'Its first rule, a third way.' 'Its last rule.'
    git add -A .
    git commit -q -m "Merge a branch, resolving a decision (#10)$marked"
done
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

has "ADR 0010" "$decisions" \
    && fail "a typo fix its commit names made an ADR amended: $decisions" \
    || note "a typo fix its commit names is not an amendment"

has "ADR 0011" "$decisions" || has "ADR 0012" "$decisions" \
    && fail "a commit with the cleanup mark made its ADRs amended: $decisions" \
    || note "a commit with the cleanup mark keeps its ADRs out of the amended list"

has "ADR 0015" "$decisions" \
    && fail "an ADR the cleanup renamed with its title reached the Decisions: $decisions" \
    || note "an ADR a cleanup commit renames with its title is not listed"

has "ADR 0013 — A decision cleaned, then changed (amended)" "$decisions" \
    && note "a decision changed without the mark is amended, a cleanup commit before it or not" \
    || fail "the ADR changed after its cleanup is not marked amended: $decisions"

has "ADR 0014 — A decision under its new title (amended)" "$decisions" \
    && note "a decision rewritten under the file name of its new title is amended, not new" \
    || fail "the ADR rewritten under a new file name is not marked amended: $decisions"

has "ADR 0016 — A decision with a typo, then changed (amended)" "$decisions" \
    && note "a typo mark covers its own commit, not a later change of the decision" \
    || fail "the ADR changed after its typo fix is not marked amended: $decisions"

has "ADR 0017 — A decision the range retires and removes (removed)" "$decisions" \
    && note "an ADR deleted in the range is listed as removed" \
    || fail "the deleted ADR is not listed as removed: $decisions"

has "ADR 0018" "$decisions" \
    && fail "an ADR a cleanup commit deleted reached the Decisions: $decisions" \
    || note "an ADR a commit with the cleanup mark deletes is not listed"

has "ADR 0019" "$decisions" \
    && fail "an unmarked merge made amended what marked commits changed on both sides: $decisions" \
    || note "an unmarked merge of a branch whose ADR change carries the cleanup mark lists nothing"

has "ADR 0020 — A decision a merge resolves (amended)" "$decisions" \
    && note "a merge whose own resolution changes an ADR's text is judged by its own message" \
    || fail "the ADR an unmarked merge resolved to a third text is not amended: $decisions"

has "ADR 0021" "$decisions" \
    && fail "the ADR a marked merge resolved reached the Decisions: $decisions" \
    || note "a marked merge's own resolution of an ADR is not an amendment"

has "ADR 0022 — A decision changed on a merged branch (amended)" "$decisions" \
    && note "a decision changed on a merged branch without the mark is amended, a marked merge or not" \
    || fail "the decision changed on a merged branch is not amended: $decisions"

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
