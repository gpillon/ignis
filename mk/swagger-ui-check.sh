#!/usr/bin/env bash
# The vendored Swagger UI against the version web/package.json pins (ADR 0036,
# GitHub #709).
#
#   bash mk/swagger-ui-check.sh           compare; exit 1 naming each file that differs
#   bash mk/swagger-ui-check.sh --files   print the vendored files, one per line
#
# The page the API reference serves is vendored under
# crates/server/assets/swagger-ui, but its version lives where every other
# frontend dependency's does: web/package.json, which `npm ci` installs into
# web/node_modules. The files below are the copy `make swagger-ui-sync` makes
# of that pin, and this is the one list of them: the sync reads it from
# --files. A bump that was not synced, or a hand-edited file, differs here.
# The release workflow's test job runs it after its `npm ci`, and `make
# swagger-ui-check` runs it too.
#
# IGNIS_SWAGGER_SRC and IGNIS_SWAGGER_DST name other directories to compare
# (mk/swagger-ui-check-selftest.sh).

set -euo pipefail

files=(swagger-ui-bundle.js swagger-ui.css swagger-ui-bundle.js.LICENSE.txt LICENSE NOTICE)

if [ $# -eq 1 ] && [ "$1" = --files ]; then
    printf '%s\n' "${files[@]}"
    exit 0
fi
if [ $# -ne 0 ]; then
    echo "usage: bash mk/swagger-ui-check.sh [--files]" >&2
    exit 2
fi

cd "$(dirname "$0")/.."
src=${IGNIS_SWAGGER_SRC:-web/node_modules/swagger-ui-dist}
dst=${IGNIS_SWAGGER_DST:-crates/server/assets/swagger-ui}

if [ ! -d "$src" ]; then
    echo "error: $src is not installed, so there is no pin to compare with" >&2
    echo "  fix: npm ci in web/ (make web-install)" >&2
    exit 1
fi
differ=()
for f in "${files[@]}"; do
    cmp -s "$src/$f" "$dst/$f" || differ+=("$f")
done
if [ ${#differ[@]} -gt 0 ]; then
    for f in "${differ[@]}"; do
        echo "error: vendored Swagger UI differs from web/package.json's pin ($f)" >&2
    done
    echo "  fix: make swagger-ui-sync, then commit $dst" >&2
    exit 1
fi
echo "swagger-ui-check: $dst is $src's copy"
