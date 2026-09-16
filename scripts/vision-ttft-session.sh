#!/usr/bin/env bash
# GitHub #181: the live/live multimodal TTFT session (ADR 0015, pooled over two
# launches per engine per ADR 0021). Run from the repository root on a free
# 5090, after `cargo build --release -p ignis-server -p ignis-bench
# --features ignis-server/cuda`:
#
#   bash scripts/vision-ttft-session.sh <out-dir>
#
# Three engines, each launched twice, in order: the reference with --vision
# (the owner's hq-e8-2b-262k preset), ignis with --vision, ignis without it.
# Every launch writes a text record (1024/8192/32768) and, with vision, an
# image record — separate records, since g2 cannot pair a record carrying the
# image cell with one that has none. Each launch measures its prompts once:
# the prompts are fixed, so a second run against the same live engine would
# be served from its caches, and ignis reports no cached tokens to catch it.
# nvidia-smi is sampled beside every launch; distrust one whose peak reaches
# the card's total (docs/findings/2026-09-16-vision-ttft-live-live.md).
#
# Engines are started with file redirects, never a pipe (a daemon holding the
# pipe hangs the caller), waited on at /v1/models, and killed before the next.
set -u
OUT=${1:?usage: vision-ttft-session.sh <out-dir>}
mkdir -p "$OUT"
ARTIFACT=${ARTIFACT:-'F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer'}
CORPUS=${CORPUS:-'F:\ai\q38\ninfer\bench\fixtures\bench_corpus.ids'}
NINFER=${NINFER:-/f/ai/q38/ninfer/build-ninja/apps/ninfer-serve.exe}
BIN=target/x86_64-pc-windows-msvc/release
IMAGE=crates/bench/tests/fixtures/vision_ttft/screenshot.png
SESSION=${SESSION:-vision-ttft-$(date -u +%Y%m%dT%H%M%SZ)}
CELLS=1024,8192,32768
echo "session $SESSION" | tee "$OUT/session.txt"

wait_ready() { # url log
  for _ in $(seq 1 900); do
    curl -sf "$1/v1/models" > /dev/null && return 0
    sleep 1
  done
  echo "engine at $1 never came up; tail of $2:"; tail -20 "$2"; return 1
}
wait_free() {
  for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$used" -lt 8192 ]; then echo "gpu free ($used MiB held)"; return 0; fi
    sleep 1
  done
  echo "gpu still held"; return 1
}
# launch <name> <url> <process-image> <label> <profile> <with-image> <command...>
launch() {
  local name=$1 url=$2 process=$3 label=$4 profile=$5 image=$6; shift 6
  wait_free || exit 1
  "$@" > "$OUT/$name.log" 2>&1 &
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l 1 > "$OUT/$name-smi.csv" &
  local smi=$!
  if wait_ready "$url" "$OUT/$name.log"; then
    "$BIN/ignis-bench.exe" ttft --endpoint "$url" --artifact "$ARTIFACT" --cells "$CELLS" --corpus "$CORPUS" \
      --label "$label" --profile "$profile" --session "$SESSION" --out "$OUT/$name-text.json" > "$OUT/$name-text.out" 2>&1
    grep -E "tokens  (median|FAILED)" "$OUT/$name-text.out" | sed "s/^/$name text/"
    if [ "$image" = 1 ]; then
      "$BIN/ignis-bench.exe" ttft --endpoint "$url" --artifact "$ARTIFACT" --image "$IMAGE" \
        --label "$label" --profile "$profile" --session "$SESSION" --out "$OUT/$name-image.json" > "$OUT/$name-image.out" 2>&1
      grep -E "tokens  (median|FAILED)" "$OUT/$name-image.out" | sed "s/^/$name image/"
    fi
  fi
  taskkill //F //IM "$process" > /dev/null 2>&1
  kill "$smi" 2>/dev/null
  echo "$name peak memory.used $(cut -d, -f2 "$OUT/$name-smi.csv" | tr -d ' MiB' | sort -n | tail -1) MiB"
}

REFERENCE_PROFILE="ninfer-serve --vision --spec mtp --draft-tokens 3 --lm-head-draft, hq-e8-2b KV, 262144 context, 4 concurrency, 450000 KV capacity, graphs"
IGNIS_PROFILE="hq-e8-2b KV, 262144 context, 1024 chunk, dflash2/7, prompt reuse"
IGNIS=("$BIN/ignis-server.exe" --bind 127.0.0.1:8000 --artifact "$ARTIFACT" --kv-format hq-e8-2b --max-context 262144
  --prefill-chunk 1024 --kv-host-pool-bytes 8G --request-timeout 1800 --spec dflash2 --draft-tokens 7)

for n in 1 2; do
  launch "reference-$n" http://127.0.0.1:8080 ninfer-serve.exe reference "$REFERENCE_PROFILE" 1 \
    "$NINFER" "$ARTIFACT" --model-id qwen3.8-27b-nvfp4full-hq-e8-2b-262k --vision --spec mtp --draft-tokens 3 \
    --lm-head-draft --host 127.0.0.1 --port 8080 --preserve-thinking --max-pending-requests 50 \
    --pending-timeout-ms 3000000 --kv-dtype hq-e8-2b --max-context 262144 --max-concurrency 4 --kv-capacity 450000
done
for n in 1 2; do
  launch "ignis-vision-$n" http://127.0.0.1:8000 ignis-server.exe ignis-vision "$IGNIS_PROFILE, --vision" 1 "${IGNIS[@]}" --vision
done
for n in 1 2; do
  launch "ignis-text-$n" http://127.0.0.1:8000 ignis-server.exe ignis-text "$IGNIS_PROFILE" 0 "${IGNIS[@]}"
done
wait_free

for record in reference-text reference-image ignis-vision-text ignis-vision-image ignis-text-text; do
  engine=${record%-*}; kind=${record##*-}
  echo "pooled $record"
  python scripts/ttft-pool.py --out "$OUT/$record.json" "$OUT/$engine-1-$kind.json" "$OUT/$engine-2-$kind.json"
done
# The 1.5 threshold g2 prints is G2's; these ratios are error detectors.
for pair in "ignis-vision-image reference-image" "ignis-vision-text reference-text" "ignis-vision-text ignis-text-text"; do
  set -- $pair
  "$BIN/ignis-bench.exe" g2 --ours "$OUT/$1.json" --ref "$OUT/$2.json" --out "$OUT/g2-$1-vs-$2.json" 2>&1 | grep -E "^ +[0-9]+ |tokens"
done
echo "session done"
