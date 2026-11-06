#!/usr/bin/env bash
# The behavioural test of the Makefile's engine-flag plumbing
# (`make flags-selftest`): it reads the "server flags" line `make config`
# prints for CUDA=1 and asserts the knobs of the speculation section arrive as
# real arguments. It found a literal "\n" ahead of --spec-draft-rows (#307): every
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
has "DRAFT_ROWS alone" "$out" "--spec-draft-rows 6"
lacks "DRAFT_ROWS alone" "$out" "--spec-backend"

out="$(flags MODEL=flash-next SPEC=mtp DRAFT_TOKENS=3 DRAFT_ROWS=6)"
has "SPEC=mtp" "$out" "--spec-backend mtp"
has "SPEC=mtp" "$out" "--spec-draft-tokens 3"
has "SPEC=mtp" "$out" "--spec-draft-rows 6"

out="$(flags MODEL=flash-next)"
# GitHub #336: in Flash-Next's own section, so a switch leaves the 27B its 8.
has "default" "$out" "--qwen38flashnext-spec-decode-lanes 3"
has "default" "$out" "--model-max-context 262144"
lacks "default" "$out" "--spec-backend"
lacks "default" "$out" "--spec-draft-rows"

out="$(flags MODEL=flash-next)"
lacks "default" "$out" "--model-decode-share"
# The VRAM headroom is the server's 1536M on both models (the owner, 2026-10-09):
# make passes none unless asked.
lacks "default" "$out" "--vram-headroom-bytes"
lacks "27B default" "$(flags)" "--vram-headroom-bytes"
has "VRAM_HEADROOM=4G" "$(flags MODEL=flash-next VRAM_HEADROOM=4G)" "--vram-headroom-bytes 4G"
lacks "VRAM_BUDGET" "$(flags MODEL=flash-next VRAM_BUDGET=28G)" "--vram-headroom-bytes"
out="$(flags MODEL=flash-next DECODE_SHARE=25)"
has "DECODE_SHARE=25" "$out" "--model-decode-share 25"
out="$(flags DECODE_SHARE=0)"
has "27B DECODE_SHARE=0" "$out" "--model-decode-share 0"
# GitHub #489: the solo width is the server's `auto` unless named.
lacks "27B default" "$(flags)" "--model-prefill-chunk-solo"
lacks "default" "$(flags MODEL=flash-next)" "--model-prefill-chunk-solo"
has "PREFILL_CHUNK_SOLO=4096" "$(flags PREFILL_CHUNK_SOLO=4096)" "--model-prefill-chunk-solo 4096"

# `make config` says what a Flash-Next start will get, before it loads.
plan="$(make config CUDA=1 MODEL=flash-next 2>/dev/null | sed -n 's/^PLAN  *//p')"
for fact in "lanes=3" "context/lane=262144" "kv=hq-e8-2b" "prefill_chunk=8192" "decode_share=25" "retained_host=8" "kv_ram_arena=2G"; do
    case "$plan" in *"$fact"*) ;; *) fail "PLAN: '$fact' missing from: $plan" ;; esac
done
plan="$(make config CUDA=1 MODEL=flash-next DECODE_SHARE=25 LANES=2 2>/dev/null | sed -n 's/^PLAN  *//p')"
case "$plan" in *"decode_share=25%"*"lanes=2"*|*"lanes=2"*"decode_share=25%"*) ;; *) fail "PLAN does not follow the knobs: $plan" ;; esac

out="$(flags MODEL=flash-next LANES=1)"
has "LANES=1" "$out" "--qwen38flashnext-spec-decode-lanes 1"
# GitHub #336: the 27B takes the lanes too, in its own section.
has "27B LANES=2" "$(flags LANES=2)" "--qwen38-spec-decode-lanes 2"

# ADR 0045: the default max_tokens, a non-default value on both models, and
# nothing passed when the knob is empty (the server's own 38912).
for model in "MODEL=flash-next" "MODEL=27b"; do
    out="$(flags $model DEFAULT_MAX_TOKENS=8192)"
    has "$model DEFAULT_MAX_TOKENS=8192" "$out" "--model-default-max-tokens 8192"
    out="$(flags $model)"
    lacks "$model default" "$out" "--model-default-max-tokens"
done
# ADR 0049: the idle unload reaches the server as named, 0 included (what a
# measurement passes), and nothing when the knob is empty (the server's 300).
has "IDLE_UNLOAD=0" "$(flags IDLE_UNLOAD=0)" "--model-idle-unload 0"
has "IDLE_UNLOAD=600" "$(flags MODEL=flash-next IDLE_UNLOAD=600)" "--model-idle-unload 600"
lacks "idle default" "$(flags)" "--model-idle-unload"
# ADR 0052: the loop guard reaches the server as named (off is what a bench or
# a gate passes, for reference parity), and nothing when the knob is empty
# (the server's rewind).
has "LOOP_GUARD=off" "$(flags LOOP_GUARD=off)" "--model-loop-guard off"
has "LOOP_GUARD=log" "$(flags MODEL=flash-next LOOP_GUARD=log)" "--model-loop-guard log"
lacks "loop guard default" "$(flags)" "--model-loop-guard"
out="$(flags MODEL=flash-next ALLOW_EXPERT_CACHE_BELOW_FLOOR=1)"
has "ALLOW_EXPERT_CACHE_BELOW_FLOOR=1" "$out" "--vram-allow-expert-cache-below-floor"
lacks "floor default" "$(flags MODEL=flash-next)" "--vram-allow-expert-cache-below-floor"
has "PINNED_EXPERT_BYTES=1G" "$(flags MODEL=flash-next PINNED_EXPERT_BYTES=1G)" "--vram-pinned-expert-bytes 1G"
lacks "pin default" "$(flags MODEL=flash-next)" "--vram-pinned-expert-bytes"
out="$(flags MODEL=flash-next KV_POOL_BYTES=512Ktok)"
has "KV_POOL_BYTES=512Ktok" "$out" "--vram-kv-pool-bytes 512Ktok"

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
lacks "27B" "$out" "--spec-decode-lanes"
# GitHub #336 (ADR 0063 §3): the 27B's context is the profile's, which the
# load adapts to the free memory; a named one is explicit and never adapts.
lacks "27B context" "$out" "--model-max-context"
has "27B MAX_CONTEXT=524288" "$(flags MAX_CONTEXT=524288)" "--model-max-context 524288"
engine="$(make config CUDA=1 2>/dev/null | sed -n 's/^engine (CUDA=1) //p')"
case "$engine" in *"context=default"*) ;; *) fail "27B engine line: 'context=default' missing from: $engine" ;; esac

out="$(flags MODEL=flash-next SPEC=off)"
has "SPEC=off" "$out" "--spec-backend off"

# The server refuses a window beside --spec-backend off, so the 27B's default
# DRAFT_TOKENS=7 must not ride along when SPEC=off turns speculation off.
out="$(flags SPEC=off)"
has "27B SPEC=off" "$out" "--spec-backend off"
lacks "27B SPEC=off" "$out" "--spec-draft-tokens"

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

# The abliterated variant is served under its own catalog id (ADR 0047, spec
# model-download/02 AC 15), still from its artifact path; the default 27B
# names no id (the server's default), and a MODEL id still wins.
out="$(flags UNCENSORED=1)"
has "UNCENSORED=1" "$out" "--model-id qwen3.8-27b-abliterated"
has "UNCENSORED=1" "$out" "--model-artifact ./models/qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer"
lacks "27B default" "$(flags)" "--model-id"
has "UNCENSORED=1 MODEL=mine" "$(flags UNCENSORED=1 MODEL=mine)" "--model-id mine"

# The image model (spec image-model/05): MODEL=image selects its own family,
# artifact and served id, and none of the text engine's knobs reach the server
# (it has no context, lanes, chunk, speculation, KV format or vision); the
# settings every model shares still do.
out="$(flags MODEL=image)"
has "MODEL=image" "$out" "--model-artifact F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer"
has "MODEL=image" "$out" "--model-id ignis-image-2.1-turbo"
lacks "MODEL=image" "$out" "--server-request-timeout"
lacks "MODEL=image" "$out" "--server-stream-timeout"
has "MODEL=image REQUEST_TIMEOUT" "$(flags MODEL=image REQUEST_TIMEOUT=900)" "--server-request-timeout 900"
has "MODEL=image STREAM_TIMEOUT" "$(flags MODEL=image STREAM_TIMEOUT=1200)" "--server-stream-timeout 1200"
for text_knob in --model-max-context --model-prefill-chunk --model-decode-share --model-kv-format --model-rope-scaling                  --spec-decode-lanes --spec-backend --spec-draft-tokens --vision-enabled --reuse-kv-host-pool-bytes --kv-disk; do
    lacks "MODEL=image" "$out" "$text_knob"
done
has "MODEL=image ARTIFACT" "$(flags MODEL=image ARTIFACT=D:/x/img.ninfer)" "--model-artifact D:/x/img.ninfer"
has "MODEL=image IDLE_UNLOAD" "$(flags MODEL=image IDLE_UNLOAD=0)" "--model-idle-unload 0"
fact="$(make config CUDA=1 MODEL=image 2>/dev/null | sed -n 's/^MODEL  *//p')"
case "$fact" in "image  "*) ;; *) fail "MODEL line of the image family: $fact" ;; esac
plan="$(plan_of MODEL=image)"
case "$plan" in *"image model"*) ;; *) fail "PLAN of the image family: $plan" ;; esac
# `make smoke` asks the image model for one image, never a chat completion
# (a 409 model_kind_mismatch); a text model keeps its chat.
smoke="$(make -n smoke MODEL=image 2>/dev/null)"
case "$smoke" in *"/v1/images/generations"*'"model":"ignis-image-2.1-turbo"'*) ;; *) fail "smoke MODEL=image: no image request in: $smoke" ;; esac
lacks "smoke MODEL=image" "$smoke" "/v1/chat/completions"
case "$(make -n smoke 2>/dev/null)" in *"/v1/chat/completions"*) ;; *) fail "smoke: the 27B's chat completion is gone" ;; esac

# Flash-Next's KV-RAM arena is its own (ADR 0046), on every start -- the
# 27B's included, so a switch to Flash-Next takes 2G, never the 27B's 8G.
for model in "" "MODEL=flash-next"; do
    out="$(flags $model)"
    has "[$model] arena" "$out" "--reuse-kv-host-pool-bytes 8G"
    has "[$model] arena" "$out" "--qwen38flashnext-reuse-kv-host-pool-bytes 2G"
done
has "FN_KV_HOST_POOL_BYTES=0" "$(flags FN_KV_HOST_POOL_BYTES=0)" "--qwen38flashnext-reuse-kv-host-pool-bytes 0"
has "KV_HOST_POOL_BYTES=16G" "$(flags KV_HOST_POOL_BYTES=16G)" "--qwen38flashnext-reuse-kv-host-pool-bytes 2G"
# The image model's start carries them too, for a switch to Flash-Next: it
# makes no host plan of its own, and the 27B's 8G arena goes in its scoped spelling.
out="$(flags MODEL=image)"
has "MODEL=image arena" "$out" "--qwen38flashnext-reuse-kv-host-pool-bytes 2G"
lacks "MODEL=image arena" "$out" "--reuse-kv-host-pool-bytes 8G"
has "MODEL=image 27B arena" "$out" "--qwen38-reuse-kv-host-pool-bytes 8G"
has "MODEL=image 27B arena off" "$(flags MODEL=image IMAGE_27B_KV_HOST_POOL_BYTES=0)" "--qwen38-reuse-kv-host-pool-bytes 0"
for model in "" "MODEL=flash-next"; do
    lacks "[$model] no scoped 27B arena" "$(flags $model)" "--qwen38-reuse-kv-host-pool-bytes"
done
has "MODEL=image HOST_MARGIN_BYTES=3G" "$(flags MODEL=image HOST_MARGIN_BYTES=3G)" "--model-host-margin-bytes 3G"
# Metrics are on by default in the server itself (the owner, 2026-10-15):
# METRICS=1 passes nothing, so a second server on the host whose default
# listener is taken still starts; METRICS=0 turns them off.
lacks "metrics default" "$(flags)" "--server-metrics"
has "METRICS=0" "$(flags METRICS=0)" "--server-metrics false"
has "METRICS_BIND" "$(flags METRICS_BIND=127.0.0.1:9100)" "--server-metrics-bind 127.0.0.1:9100"
lacks "METRICS=0 METRICS_BIND" "$(flags METRICS=0 METRICS_BIND=127.0.0.1:9100)" "--server-metrics-bind"
case "$(make config CUDA=1 METRICS=0 2>/dev/null | sed -n 's/^METRICS  *//p')" in "0  (off)") ;; *) fail "METRICS=0 config line" ;; esac
# The host plan's margin: the server's 4G unless named.
lacks "margin default" "$(flags MODEL=flash-next)" "--model-host-margin-bytes"
has "HOST_MARGIN_BYTES=3G" "$(flags HOST_MARGIN_BYTES=3G)" "--model-host-margin-bytes 3G"

if [ "$failed" -ne 0 ]; then
    printf 'flags-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "flags-selftest: ok"
