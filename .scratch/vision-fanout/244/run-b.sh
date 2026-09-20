#!/usr/bin/env bash
set -u
OUT=.scratch/vision-fanout/244/budget
mkdir -p "$OUT"
SRV=./target/x86_64-pc-windows-msvc/release/ignis-server.exe

for B in 16384 8192 4096 2048 1024 512; do
  echo "=== budget $B ==="
  IGNIS_LOG_FORMAT=json "$SRV" --bind 127.0.0.1:8000 \
    --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
    --kv-format hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
    --vision --vision-max-tokens "$B" --spec dflash2 --draft-tokens 7 --no-ui \
    > "$OUT/server_$B.log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 120); do
    curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break
    sleep 1
  done
  for S in large medium small; do
    t0=$(date +%s%N)
    curl -s -m 600 -X POST http://127.0.0.1:8000/v1/decide \
      -H 'content-type: application/json' \
      --data-binary @".scratch/vision-fanout/244/b_$S.json" \
      -o "$OUT/ans_${B}_$S.json"
    t1=$(date +%s%N)
    echo "  $S $(( (t1-t0)/1000000 )) ms"
    echo "$(( (t1-t0)/1000000 ))" > "$OUT/ms_${B}_$S.txt"
  done
  kill $SRV_PID 2>/dev/null
  for i in $(seq 1 30); do
    tasklist 2>/dev/null | grep -qi ignis-server || break
    sleep 1
  done
  taskkill //F //IM ignis-server.exe > /dev/null 2>&1
  sleep 2
done
echo ALLDONE
