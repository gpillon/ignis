#!/usr/bin/env bash
set -u
OUT=.scratch/vision-fanout/244
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
SRV=./target/x86_64-pc-windows-msvc/release/ignis-server.exe

rm -f "$OUT"/server.log "$OUT"/nsys-vision.nsys-rep "$OUT"/nsys-vision.sqlite
START=$(date +%s)
IGNIS_LOG_FORMAT=json "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
  --delay 40 --duration 100 --force-overwrite true -o "$OUT/nsys-vision" \
  "$SRV" --bind 127.0.0.1:8000 \
  --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 40960 --prefill-chunk 1024 \
  --vision --spec dflash2 --draft-tokens 7 --no-ui \
  > "$OUT/server.log" 2>&1 &
NSYS_PID=$!

# wait for ready
for i in $(seq 1 120); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then break; fi
  sleep 1
done
echo "ready after $(( $(date +%s) - START ))s"

# warmup outside the capture window
curl -s -m 300 -X POST http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' \
  --data-binary @"$OUT/noul_s768.json" -o "$OUT/warm.json"
echo "warmup done at $(( $(date +%s) - START ))s"

# wait until the capture window has opened
while [ $(( $(date +%s) - START )) -lt 43 ]; do sleep 1; done

echo "-- 1536 at $(( $(date +%s) - START ))s"
curl -s -m 300 -X POST http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' \
  --data-binary @"$OUT/noul_s1536.json" -o "$OUT/prof_1536.json"
sleep 3
echo "-- 4096 at $(( $(date +%s) - START ))s"
curl -s -m 300 -X POST http://127.0.0.1:8000/v1/decide -H 'content-type: application/json' \
  --data-binary @"$OUT/noul_s4096.json" -o "$OUT/prof_4096.json"
echo "-- done at $(( $(date +%s) - START ))s"
sleep 3
wait $NSYS_PID
echo "nsys exited at $(( $(date +%s) - START ))s"
