#!/usr/bin/env bash
set -u
OUT=.scratch/vision-fanout/244/fan
mkdir -p "$OUT"
IGNIS_LOG_FORMAT=json ./target/x86_64-pc-windows-msvc/release/ignis-server.exe \
  --bind 127.0.0.1:8000 --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
  --vision --spec dflash2 --draft-tokens 7 --no-ui \
  > "$OUT/server.log" 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 1; done
echo ready
for Q in 1q 2q 4q; do
  t0=$(date +%s%N)
  curl -s -m 900 -X POST http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' \
    --data-binary @".scratch/decide-live/fan_$Q.json" -o "$OUT/fan_$Q.json"
  t1=$(date +%s%N); echo "$Q $(( (t1-t0)/1000000 )) ms"
done
taskkill //F //IM ignis-server.exe > /dev/null 2>&1
echo ALLDONE
