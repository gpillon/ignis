#!/usr/bin/env bash
# One shard of `cargo test --workspace` (GitHub #552).
#
#   bash mk/test-shard.sh <shard> <count> <target>          build and run it
#   bash mk/test-shard.sh <shard> <count> <target> --plan   print it, run nothing
#
# The release workflow's test job runs the CPU-only suite as <count> shards on
# each host, side by side. Almost all of a test leg's time is compiling, not
# running: every workspace crate and every test binary is built again on every
# push (rust-cache keeps only the dependencies), and the integration tests are
# hundreds of binaries, each linked on its own. A shard builds and runs only
# its part, so the shards split the compile as well as the run.
#
# A shard is `cargo test --workspace` with target flags, so cargo resolves
# features for the whole workspace exactly as the unsharded command does
# (`-p <crate>` would resolve them for that crate alone, and its tests could
# build against other features). Its units are what those flags can select:
#
#   lib        every crate's library unit tests     --lib
#   bins       every binary's unit tests            --bins
#   test NAME  the integration tests named NAME     --test NAME
#   doc        every crate's doc tests              --doc, a command of its own
#   examples   every example, built to check it compiles, as `cargo test`
#              builds them: `cargo build --workspace --examples`
#
# `--test NAME` selects that name in every crate (two crates have integration
# tests of the same name), so a name is one unit. The units are read off
# `cargo metadata`, never listed by hand, sorted, and unit i goes to shard
# ((i - 1) mod count) + 1: the shards of one count hold every unit exactly
# once, and a new test file joins a shard by itself. A target whose required
# features the default build lacks is left out, as `cargo test` leaves it
# out. A target the flags above cannot select as `cargo test --workspace`
# does (a library or a binary marked `test = false`, a library marked
# `doctest = false`, an example or a benchmark marked `test = true`, a kind
# this script has never seen) is refused, so a change to the workspace cannot
# drop or add a test without failing here first.
#
# Every command of a shard runs even after one fails, and the shard fails if
# any did.
#
# IGNIS_TEST_SHARD_METADATA names a file to read in place of `cargo metadata`
# (mk/test-shard-selftest.sh).

set -euo pipefail

usage() {
    echo "usage: bash mk/test-shard.sh <shard> <count> <target> [--plan]" >&2
    exit 2
}
[ $# -eq 3 ] || [ $# -eq 4 ] || usage
shard=$1 count=$2 target=$3 plan=0
if [ $# -eq 4 ]; then
    [ "$4" = --plan ] || usage
    plan=1
fi
case "$shard$count" in
    "" | *[!0-9]*) echo "test-shard: <shard> and <count> are whole numbers: '$shard', '$count'" >&2; exit 2 ;;
esac
if [ "$count" -lt 1 ] || [ "$shard" -lt 1 ] || [ "$shard" -gt "$count" ]; then
    echo "test-shard: shard $shard of $count does not exist" >&2
    exit 2
fi
command -v jq >/dev/null 2>&1 || { echo "test-shard: jq is not on PATH" >&2; exit 2; }

if [ -n "${IGNIS_TEST_SHARD_METADATA:-}" ]; then
    metadata=$(cat "$IGNIS_TEST_SHARD_METADATA")
else
    metadata=$(cargo metadata --no-deps --format-version 1 --locked)
fi

# One line per unit, or `refuse <why>` for a target no unit can select.
all=$(jq -r '
    .packages[] | . as $crate | (($crate.features.default) // []) as $default
    | .targets[]
    | select((((."required-features") // []) - $default) | length == 0)
    | .kind as $kind | "\($crate.name): \($kind | join(",")) target \(.name)" as $what
    | if any($kind[]; . == "lib" or . == "rlib" or . == "dylib" or . == "cdylib" or . == "staticlib" or . == "proc-macro") then
          (if .test then "lib" else "refuse \($what) is marked test = false" end),
          (if .doctest then "doc" else "refuse \($what) is marked doctest = false" end)
      elif $kind == ["bin"] then (if .test then "bins" else "refuse \($what) is marked test = false" end)
      elif $kind == ["test"] then (if .test then "test \(.name)" else "refuse \($what) is marked test = false" end)
      elif $kind == ["example"] then (if .test then "refuse \($what) is marked test = true" else "examples" end)
      elif $kind == ["bench"] then (if .test then "refuse \($what) is marked test = true" else empty end)
      elif $kind == ["custom-build"] then empty
      else "refuse \($what) is a kind mk/test-shard.sh does not know"
      end
' <<<"$metadata" | LC_ALL=C sort -u)

refused=$(grep '^refuse ' <<<"$all" || true)
if [ -n "$refused" ]; then
    sed 's/^refuse /test-shard: cannot shard /' <<<"$refused" >&2
    exit 2
fi
[ -n "$all" ] || { echo "test-shard: the workspace has no test targets" >&2; exit 2; }

mine=$(awk -v shard="$shard" -v count="$count" '(NR - 1) % count == shard - 1' <<<"$all")
if [ "$plan" = 1 ]; then
    [ -z "$mine" ] || printf '%s\n' "$mine"
    exit 0
fi
if [ -z "$mine" ]; then
    echo "test-shard: shard $shard of $count holds nothing ($(wc -l <<<"$all") units in all)"
    exit 0
fi
echo "test-shard: shard $shard of $count, $(wc -l <<<"$mine") of $(wc -l <<<"$all") units"

flags=() doc=0 examples=0
while read -r unit name; do
    case "$unit" in
        lib) flags+=(--lib) ;;
        bins) flags+=(--bins) ;;
        test) flags+=(--test "$name") ;;
        doc) doc=1 ;;
        examples) examples=1 ;;
    esac
done <<<"$mine"

failed=()
run() {
    echo "+ $*"
    "$@" || failed+=("$*")
}
if [ ${#flags[@]} -gt 0 ]; then
    run cargo test --workspace --locked --target "$target" "${flags[@]}"
fi
if [ "$doc" = 1 ]; then
    run cargo test --workspace --locked --target "$target" --doc
fi
if [ "$examples" = 1 ]; then
    run cargo build --workspace --locked --target "$target" --examples
fi

if [ ${#failed[@]} -gt 0 ]; then
    echo "test-shard: shard $shard of $count failed in:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
