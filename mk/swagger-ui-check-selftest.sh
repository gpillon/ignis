#!/usr/bin/env bash
# The behavioural test of mk/swagger-ui-check.sh (`make
# swagger-ui-check-selftest`, GitHub #709). The check compares two
# directories, so each case is a pair of temp directories standing in for the
# installed pin and the vendored copy: it needs no npm and reads no
# node_modules. The one real directory it reads is the vendored copy's, to
# hold the check's file list to what is vendored. The release workflow's test
# job runs it on both hosts.

set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
check="$here/swagger-ui-check.sh"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0

fail() {
    echo "FAIL: $1" >&2
    failed=1
}

# The list is the vendored directory: every file in it but the vendor note.
listed=$(bash "$check" --files | sort)
vendored=$(cd "$here/../crates/server/assets/swagger-ui" && ls -A | grep -vx IGNIS-VENDOR.md | sort)
[ "$listed" = "$vendored" ] || fail "--files lists $(echo $listed), the directory holds $(echo $vendored)"

# A pin and its copy, every listed file the same bytes in both.
pair() {
    rm -rf "$dir/src" "$dir/dst"
    mkdir -p "$dir/src" "$dir/dst"
    local f
    for f in $listed; do
        printf 'swagger-ui %s\n' "$f" >"$dir/src/$f"
        cp "$dir/src/$f" "$dir/dst/$f"
    done
}
run() {
    IGNIS_SWAGGER_SRC="$dir/src" IGNIS_SWAGGER_DST="$dir/dst" bash "$check" "$@" >"$dir/out" 2>"$dir/err"
}

pair
run || fail "a faithful copy fails: $(cat "$dir/err")"

pair
echo edited >>"$dir/dst/swagger-ui.css"
if run; then
    fail "an edited vendored file passes"
else
    grep -q '(swagger-ui.css)' "$dir/err" || fail "the edited file is not named: $(cat "$dir/err")"
    [ "$(grep -c '^error: ' "$dir/err")" = 1 ] || fail "a file that did not change is named: $(cat "$dir/err")"
fi

pair
rm "$dir/dst/NOTICE"
if run; then
    fail "a missing vendored file passes"
else
    grep -q '(NOTICE)' "$dir/err" || fail "the missing file is not named: $(cat "$dir/err")"
fi

pair
rm -rf "$dir/src"
if run; then
    fail "a pin that is not installed passes"
else
    grep -q 'npm ci' "$dir/err" || fail "an uninstalled pin does not say to install it: $(cat "$dir/err")"
fi

pair
run --sync
[ $? = 2 ] || fail "an unknown argument is not a usage error"

if [ "$failed" = 0 ]; then
    echo "swagger-ui-check-selftest: ok"
fi
exit "$failed"
