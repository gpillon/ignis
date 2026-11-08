#!/usr/bin/env bash
# The behavioural test of mk/test-shard.sh (`make test-shard-selftest`),
# docs/agents/testing.md's rule applied to a script. The shards together must
# run what `cargo test --workspace` runs, each part once, so this checks the
# plan against a hand-written `cargo metadata` and the commands against a
# `cargo` stub that only records them: it builds nothing and needs no Rust
# toolchain. The release workflow's test job runs it on both hosts, before the
# shards it plans are trusted.

set -uo pipefail
shard="$(dirname "$0")/test-shard.sh"
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
failed=0
ok() { echo "ok    $1"; }
fail() { echo "FAIL  $1"; failed=1; }

# A target of `cargo metadata`: kind, name, then optional test, doctest and
# required features.
target() {
    printf '{"kind":["%s"],"name":"%s","test":%s,"doctest":%s,"required-features":[%s]}' \
        "$1" "$2" "${3:-true}" "${4:-false}" "${5:-}"
}
# A crate: name, its default features, then its targets.
crate() {
    local name=$1 default=$2; shift 2
    local IFS=,
    printf '{"name":"%s","features":{"default":[%s]},"targets":[%s]}' "$name" "$default" "$*"
}
metadata() {
    local IFS=,
    printf '{"packages":[%s]}' "$*"
}

# Two crates as the workspace has them: libraries with unit and doc tests,
# binaries, integration tests (one name in both crates), examples, a build
# script, a benchmark `cargo test` does not run, and targets whose required
# features the default build has or lacks.
good=$(metadata \
    "$(crate alpha '' \
        "$(target lib alpha_lib true true)" \
        "$(target bin alpha-cli)" \
        "$(target test zeta)" \
        "$(target test shared)" \
        "$(target test needs_cuda true false '"cuda"')" \
        "$(target example dump false)" \
        "$(target custom-build build-script-build false)")" \
    "$(crate beta '"fast"' \
        "$(target lib beta_lib true true)" \
        "$(target test shared)" \
        "$(target test eta)" \
        "$(target test on_by_default true false '"fast"')" \
        "$(target example slow false false '"cuda"')" \
        "$(target bench speed false)")")
printf '%s' "$good" > "$dir/good.json"
plan() { IGNIS_TEST_SHARD_METADATA="$dir/$1.json" bash "$shard" "$2" "$3" x86_64-test-target --plan; }

expected='bins
doc
examples
lib
test eta
test on_by_default
test shared
test zeta'
[ "$(plan good 1 1)" = "$expected" ] \
    && ok "one shard is every unit cargo test --workspace runs, a shared test name once" \
    || fail "one shard is every unit: got $(plan good 1 1 | tr '\n' '|')"

# Every count: the shards together hold each unit exactly once.
for count in 1 2 3 4 5 8 9; do
    union=""
    for s in $(seq 1 "$count"); do union+="$(plan good "$s" "$count")"$'\n'; done
    if [ "$(grep -v '^$' <<<"$union" | LC_ALL=C sort)" = "$expected" ]; then
        ok "$count shards hold every unit exactly once"
    else
        fail "$count shards hold every unit exactly once: $(grep -v '^$' <<<"$union" | LC_ALL=C sort | tr '\n' '|')"
    fi
done
[ -z "$(plan good 9 9)" ] && ok "a shard past the last unit holds nothing" || fail "a shard past the last unit holds nothing"

# What no target flag of `cargo test --workspace` can select as it does is
# refused, naming the target.
refused() {
    local why=$1 json=$2
    printf '%s' "$json" > "$dir/case.json"
    local err
    err=$(IGNIS_TEST_SHARD_METADATA="$dir/case.json" bash "$shard" 1 1 x86_64-test-target --plan 2>&1 >/dev/null)
    local status=$?
    if [ "$status" = 2 ] && grep -q "cannot shard $3" <<<"$err"; then ok "refused: $why"; else fail "refused: $why (exit $status: $err)"; fi
}
refused "a library marked test = false" "$(metadata "$(crate c '' "$(target lib c false true)")")" "c: lib target c is marked test = false"
refused "a library marked doctest = false" "$(metadata "$(crate c '' "$(target lib c true false)")")" "c: lib target c is marked doctest = false"
refused "a binary marked test = false" "$(metadata "$(crate c '' "$(target bin c false)")")" "c: bin target c is marked test = false"
refused "an integration test marked test = false" "$(metadata "$(crate c '' "$(target test t false)")")" "c: test target t is marked test = false"
refused "an example marked test = true" "$(metadata "$(crate c '' "$(target example e true)")")" "c: example target e is marked test = true"
refused "a benchmark marked test = true" "$(metadata "$(crate c '' "$(target bench b true)")")" "c: bench target b is marked test = true"
refused "a kind the script does not know" "$(metadata "$(crate c '' "$(target wasm w)")")" "c: wasm target w is a kind"

# A shard that does not exist, or arguments that are not one.
for args in "0 4" "5 4" "1 0" "x 4" "1 four"; do
    # shellcheck disable=SC2086 # two words on purpose
    IGNIS_TEST_SHARD_METADATA="$dir/good.json" bash "$shard" $args x86_64-test-target --plan > /dev/null 2>&1
    [ $? = 2 ] && ok "shard $args is refused" || fail "shard $args is refused"
done
IGNIS_TEST_SHARD_METADATA="$dir/good.json" bash "$shard" 1 4 > /dev/null 2>&1
[ $? = 2 ] && ok "a missing target is refused" || fail "a missing target is refused"

# The commands, against a cargo that records them and fails `--test zeta`.
mkdir "$dir/bin"
cat > "$dir/bin/cargo" <<'STUB'
#!/usr/bin/env bash
echo "cargo $*" >> "$CARGO_LOG"
case " $* " in *" --test zeta "*) exit 101 ;; esac
exit 0
STUB
chmod +x "$dir/bin/cargo"
commands() {
    : > "$dir/cargo.log"
    PATH="$dir/bin:$PATH" CARGO_LOG="$dir/cargo.log" IGNIS_TEST_SHARD_METADATA="$dir/good.json" \
        bash "$shard" "$1" "$2" x86_64-test-target > "$dir/out.txt" 2>&1
    status=$?
}
commands 1 1
want='cargo test --workspace --no-fail-fast --locked --target x86_64-test-target --bins --lib --test eta --test on_by_default --test shared --test zeta
cargo test --workspace --no-fail-fast --locked --target x86_64-test-target --doc
cargo build --workspace --locked --target x86_64-test-target --examples'
[ "$(cat "$dir/cargo.log")" = "$want" ] \
    && ok "one shard is one cargo test --workspace with its target flags, the doc tests and the examples' build" \
    || fail "one shard's commands: $(tr '\n' '|' < "$dir/cargo.log")"
[ "$status" = 1 ] && grep -q -- '--test zeta' "$dir/out.txt" \
    && ok "a failing command fails the shard, naming it, after the others ran" \
    || fail "a failing command fails the shard (exit $status)"
commands 2 4
want='cargo test --workspace --no-fail-fast --locked --target x86_64-test-target --test on_by_default
cargo test --workspace --no-fail-fast --locked --target x86_64-test-target --doc'
[ "$(cat "$dir/cargo.log")" = "$want" ] && [ "$status" = 0 ] \
    && ok "a shard runs only its own units" \
    || fail "a shard runs only its own units: $(tr '\n' '|' < "$dir/cargo.log") (exit $status)"
commands 9 9
[ ! -s "$dir/cargo.log" ] && [ "$status" = 0 ] \
    && ok "an empty shard runs nothing and passes" \
    || fail "an empty shard runs nothing and passes"

exit "$failed"
