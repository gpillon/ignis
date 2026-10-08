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

# ADR 0045: the default max_tokens, a non-default value on both models, and
# nothing passed when the knob is empty (the server's own 38912).
for model in "MODEL=flash-next" "MODEL=27b"; do
    out="$(flags $model DEFAULT_MAX_TOKENS=8192)"
    has "$model DEFAULT_MAX_TOKENS=8192" "$out" "--default-max-tokens 8192"
    out="$(flags $model)"
    lacks "$model default" "$out" "--default-max-tokens"
done
out="$(flags MODEL=flash-next ALLOW_EXPERT_CACHE_BELOW_FLOOR=1)"
has "ALLOW_EXPERT_CACHE_BELOW_FLOOR=1" "$out" "--allow-expert-cache-below-floor"
lacks "floor default" "$(flags MODEL=flash-next)" "--allow-expert-cache-below-floor"
out="$(flags MODEL=flash-next KV_POOL_BYTES=512Ktok)"
has "KV_POOL_BYTES=512Ktok" "$out" "--kv-pool-bytes 512Ktok"

# The pool policy and its tokens, before the load (ADR 0045, AC 7): offloaded
# at 524,288 on Flash-Next, 4,104 pages at one lane, the floor at 524,288 of
# context; resident on the 27B.
plan_of() { make config CUDA=1 "$@" 2>/dev/null | sed -n 's/^PLAN  *//p'; }
for case in "|kv_pool=offloaded 524288 tokens" "LANES=1|kv_pool=offloaded 262656 tokens" \
            "MAX_CONTEXT=524288|kv_pool=offloaded 524800 tokens" "KV_POOL_BYTES=512Ktok|kv_pool=offloaded 512Ktok (named, 524288 tokens)" "KV_POOL_BYTES=4G|kv_pool=offloaded 4G (named, 1016768 tokens)"; do
    knobs="${case%%|*}"; fact="${case#*|}"
    plan="$(plan_of MODEL=flash-next $knobs)"
    case "$plan" in *"$fact"*) ;; *) fail "PLAN [$knobs]: '$fact' missing from: $plan" ;; esac
done
engine="$(make config CUDA=1 2>/dev/null | sed -n 's/^engine (CUDA=1) //p')"
case "$engine" in *"pool=resident"*) ;; *) fail "27B engine line: 'pool=resident' missing from: $engine" ;; esac
# A named byte count is worth its model's own tokens: 4 GiB under hq-e8-2b on
# the 27B, 7,281 pages (GitHub #139's figure), and under BF16 1,024.
for case in "|pool=resident 4G (named, 465984 tokens)" "KV_FORMAT=bf16|pool=resident 4G (named, 65536 tokens)"; do
    knobs="${case%%|*}"; fact="${case#*|}"
    engine="$(make config CUDA=1 KV_POOL_BYTES=4G $knobs 2>/dev/null | sed -n 's/^engine (CUDA=1) //p')"
    case "$engine" in *"$fact"*) ;; *) fail "27B engine line [$knobs]: '$fact' missing from: $engine" ;; esac
done

out="$(flags)"
lacks "27B" "$out" "--decode-lanes"

out="$(flags MODEL=flash-next SPEC=off)"
has "SPEC=off" "$out" "--spec off"

# The server refuses a window beside --spec off, so the 27B's default
# DRAFT_TOKENS=7 must not ride along when SPEC=off turns speculation off.
out="$(flags SPEC=off)"
has "27B SPEC=off" "$out" "--spec off"
lacks "27B SPEC=off" "$out" "--draft-tokens"

# KV-disk (spec vram-budget/03): the knobs reach the server as named, the
# server's defaults (4G on Flash-Next, off on the 27B) when they are not,
# and `make config` names the budget and the directory a start will use.
out="$(flags MODEL=flash-next KV_DISK_BYTES=8G KV_DISK_PATH=auto)"
has "KV_DISK_*" "$out" "--kv-disk-bytes 8G"
has "KV_DISK_*" "$out" "--kv-disk-path auto"
out="$(flags MODEL=flash-next)"
lacks "default" "$out" "--kv-disk"
disk="$(make config CUDA=1 MODEL=flash-next KV_DISK_PATH=D:/kv 2>/dev/null | sed -n 's/^KV-DISK (CUDA=1) //p')"
case "$disk" in *"budget=4G"*"dir=D:/kv/ignis-kv-disk/"*) ;; *) fail "KV-DISK: $disk" ;; esac
disk="$(make config CUDA=1 2>/dev/null | sed -n 's/^KV-DISK (CUDA=1) //p')"
case "$disk" in off*) ;; *) fail "27B KV-DISK is not off by default: $disk" ;; esac

if [ "$failed" -ne 0 ]; then
    printf 'flags-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "flags-selftest: ok"
