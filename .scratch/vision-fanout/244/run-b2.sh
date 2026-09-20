#!/usr/bin/env bash
set -u
OUT=.scratch/vision-fanout/244/size
SRV=./target/x86_64-pc-windows-msvc/release/ignis-server.exe

IGNIS_LOG_FORMAT=json "$SRV" --bind 127.0.0.1:8000 \
  --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
  --vision --spec dflash2 --draft-tokens 7 --no-ui \
  > "$OUT/server.log" 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
  sleep 1
done
echo ready

for S in large medium small; do
  for SZ in 768 1024 1536 2048 3072 4096; do
    t0=$(date +%s%N)
    curl -s -m 900 -X POST http://127.0.0.1:8000/v1/decide \
      -H 'content-type: application/json' --data-binary @"$OUT/q_${S}_${SZ}.json" \
      -o "$OUT/a_${S}_${SZ}.json"
    t1=$(date +%s%N)
    echo "$S $SZ $(( (t1-t0)/1000000 )) ms"
  done
done
# the processor's own ceiling: a 5120x5120 image nobody asked to shrink
curl -s -m 900 -X POST http://127.0.0.1:8000/v1/decide \
  -H 'content-type: application/json' --data-binary @"$OUT/q_large_5120.json" \
  -o "$OUT/a_large_5120.json"
echo "oversize done"
taskkill //F //IM ignis-server.exe > /dev/null 2>&1
echo ALLDONE
