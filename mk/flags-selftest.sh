#!/usr/bin/env bash
# The behavioural test of the Makefile's engine-flag plumbing
# (`make flags-selftest`): it reads the "server flags" line `make config`
# prints for CUDA=1 and asserts the knobs of the speculation section arrive as
# real arguments. It found a literal "\n" ahead of --draft-rows (#307): every
# CUDA=1 run/start passed the server a stray argument. It writes nothing.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo" || exit 1

failed=0
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

flags() {
    make config CUDA=1 "$@" 2>/dev/null | sed -n 's/^server flags    //p'
}

has() { # <label> <flags> <fixed substring>
    case " $2 " in *" $3 "*) ;; *) fail "$1: '$3' missing from: $2" ;; esac
}
lacks() { # <label> <flags> <fixed substring>
    case "$2" in *"$3"*) fail "$1: '$3' present in: $2" ;; esac
}

# No token of the line may be a backslash escape: make never expands them.
for args in "MODEL=flash-next DRAFT_ROWS=6" "MODEL=flash-next SPEC=mtp DRAFT_TOKENS=3 DRAFT_ROWS=6" ""; do
    out="$(flags $args)"
    [ -n "$out" ] || fail "[$args]: no server flags line"
    case "$out" in *'\'*) fail "[$args]: a literal backslash in: $out" ;; esac
done

out="$(flags MODEL=flash-next DRAFT_ROWS=6)"
has "DRAFT_ROWS alone" "$out" "--draft-rows 6"
lacks "DRAFT_ROWS alone" "$out" "--spec"

out="$(flags MODEL=flash-next SPEC=mtp DRAFT_TOKENS=3 DRAFT_ROWS=6)"
has "SPEC=mtp" "$out" "--spec mtp"
has "SPEC=mtp" "$out" "--draft-tokens 3"
has "SPEC=mtp" "$out" "--draft-rows 6"

out="$(flags MODEL=flash-next)"
has "default" "$out" "--decode-lanes 3"
has "default" "$out" "--max-context 262144"
lacks "default" "$out" "--spec"
lacks "default" "$out" "--draft-rows"

out="$(flags MODEL=flash-next)"
lacks "default" "$out" "--decode-share"
out="$(flags MODEL=flash-next DECODE_SHARE=25)"
has "DECODE_SHARE=25" "$out" "--decode-share 25"
out="$(flags DECODE_SHARE=0)"
has "27B DECODE_SHARE=0" "$out" "--decode-share 0"

# `make config` says what a Flash-Next start will get, before it loads.
plan="$(make config CUDA=1 MODEL=flash-next 2>/dev/null | sed -n 's/^PLAN  *//p')"
for fact in "lanes=3" "context/lane=262144" "kv=hq-e8-2b" "prefill_chunk=8192" "decode_share=25" "retained_host=8" "kv_ram_arena=2G"; do
    case "$plan" in *"$fact"*) ;; *) fail "PLAN: '$fact' missing from: $plan" ;; esac
done
plan="$(make config CUDA=1 MODEL=flash-next DECODE_SHARE=25 LANES=2 2>/dev/null | sed -n 's/^PLAN  *//p')"
case "$plan" in *"decode_share=25%"*"lanes=2"*|*"lanes=2"*"decode_share=25%"*) ;; *) fail "PLAN does not follow the knobs: $plan" ;; esac

out="$(flags MODEL=flash-next LANES=1)"
has "LANES=1" "$out" "--decode-lanes 1"

out="$(flags)"
lacks "27B" "$out" "--decode-lanes"

out="$(flags MODEL=flash-next SPEC=off)"
has "SPEC=off" "$out" "--spec off"

# The server refuses a window beside --spec off, so the 27B's default
# DRAFT_TOKENS=7 must not ride along when SPEC=off turns speculation off.
out="$(flags SPEC=off)"
has "27B SPEC=off" "$out" "--spec off"
lacks "27B SPEC=off" "$out" "--draft-tokens"

if [ "$failed" -ne 0 ]; then
    printf 'flags-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "flags-selftest: ok"
