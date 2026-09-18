#!/usr/bin/env bash
# Capture one window of steady-state decode on the device timeline.
#
#   trace_decode.sh <worktree> <label> <server.exe> <concurrency>
#
# The server runs under Nsight Systems with the capture delayed past model
# load; generations are issued back to back from readiness until the capture
# closes, so the window is squarely inside decode.
set -u

WT="$1"
LABEL="$2"
EXE="$3"
CONC="${4:-1}"
SP="C:/Users/Mille/AppData/Local/Temp/claude/F--ai-opencode-inference/8c7e337d-a059-4eda-8288-c3ea813f8fec/scratchpad"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
LOG="$SP/trace-$LABEL.log"
STOP="$SP/stop-$LABEL"

rm -f "$STOP"
cd "$WT" || exit 1

"$NSYS" profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --delay 60 --duration 8 \
  --force-overwrite=true -o "$SP/decode-$LABEL" \
  "$EXE" \
  --bind 127.0.0.1:8000 \
  --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 262144 --prefill-chunk 1024 \
  --kv-host-pool-bytes 8G --request-timeout 1800 \
  --spec dflash2 --draft-tokens 7 \
  --system-message-policy merge --developer-message-policy inplace \
  > "$LOG" 2>&1 &
NPID=$!

for _ in $(seq 1 300); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then break; fi
  sleep 1
done
echo "$LABEL: server up at $(date +%T), driving $CONC lane(s) until the capture closes"

for i in $(seq 1 "$CONC"); do
  (
    while [ ! -f "$STOP" ]; do
      curl -s -m 120 -X POST http://127.0.0.1:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        --data-binary "@$SP/prompt-long-gen.json" -o "$SP/gen-$LABEL-$i.json"
    done
  ) &
done

wait $NPID 2>/dev/null
echo "$LABEL: capture closed at $(date +%T)"
touch "$STOP"
sleep 2
taskkill //IM ignis-server.exe //F > /dev/null 2>&1
wait 2>/dev/null
sleep 2
ls -la "$SP/decode-$LABEL".nsys-rep 2>/dev/null
