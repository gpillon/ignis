#!/usr/bin/env bash
set -u
OUT=.scratch/vision-fanout/244/ref
mkdir -p "$OUT"
BODY=.scratch/vision-fanout/244/chat_4096.json

echo "=== ignis ==="
IGNIS_LOG_FORMAT=json ./target/x86_64-pc-windows-msvc/release/ignis-server.exe \
  --bind 127.0.0.1:8000 --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
  --vision --spec dflash2 --draft-tokens 7 --no-ui --enable-thinking false \
  > "$OUT/ignis.log" 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 1; done
for r in 1 2 3; do
  t0=$(date +%s%N)
  curl -s -m 900 -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' --data-binary @"$BODY" -o "$OUT/ignis_$r.json"
  t1=$(date +%s%N); echo "  ignis run $r: $(( (t1-t0)/1000000 )) ms"
done
taskkill //F //IM ignis-server.exe > /dev/null 2>&1
sleep 5

echo "=== ninfer ==="
/f/ai/q38/ninfer/build-ninja/apps/ninfer-serve.exe \
  /f/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --host 127.0.0.1 --port 8080 --vision --greedy --no-thinking \
  --kv-dtype hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
  --max-concurrency 1 --spec dflash2 --draft-tokens 7 \
  > "$OUT/ninfer.log" 2>&1 &
for i in $(seq 1 180); do curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1 && break; sleep 1; done
for r in 1 2 3; do
  t0=$(date +%s%N)
  curl -s -m 900 -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H 'content-type: application/json' --data-binary @"$BODY" -o "$OUT/ninfer_$r.json"
  t1=$(date +%s%N); echo "  ninfer run $r: $(( (t1-t0)/1000000 )) ms"
done
taskkill //F //IM ninfer-serve.exe > /dev/null 2>&1
echo ALLDONE
