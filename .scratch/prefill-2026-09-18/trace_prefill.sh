#!/usr/bin/env bash
# Capture a window of back-to-back cold ~1,024-token prefills.
#
#   trace_prefill.sh <worktree> <label> <server.exe> [nsys extra args]
#
# Forty distinct prompt bodies, cycled, each with its own leading marker, so
# no request can reuse another's prefix and every one is a real prefill.
set -u

WT="$1"
LABEL="$2"
EXE="$3"
EXTRA="${4:-}"
SP="C:/Users/Mille/AppData/Local/Temp/claude/F--ai-opencode-inference/8c7e337d-a059-4eda-8288-c3ea813f8fec/scratchpad"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
LOG="$SP/prefill-$LABEL.log"
STOP="$SP/stop-$LABEL"

rm -f "$STOP"
cd "$WT" || exit 1

"$NSYS" profile \
  --trace=cuda --sample=none --cpuctxsw=none $EXTRA \
  --delay 45 --duration 8 \
  --force-overwrite=true -o "$SP/prefill-$LABEL" \
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
echo "$LABEL: server up at $(date +%T), cycling 40 cold prompts"

(
  i=0
  while [ ! -f "$STOP" ]; do
    p=$(printf "%s/prompts1024/p%02d.json" "$SP" $((i % 40)))
    curl -s -m 120 -X POST http://127.0.0.1:8000/v1/chat/completions \
      -H 'Content-Type: application/json' --data-binary "@$p" \
      -o "$SP/pre-$LABEL.json"
    i=$((i + 1))
  done
  echo "$i" > "$SP/count-$LABEL"
) &

wait $NPID 2>/dev/null
echo "$LABEL: capture closed at $(date +%T)"
touch "$STOP"
sleep 2
taskkill //IM ignis-server.exe //F > /dev/null 2>&1
sleep 2
ls -la "$SP/prefill-$LABEL.nsys-rep" 2>/dev/null
echo "requests sent: $(cat "$SP/count-$LABEL" 2>/dev/null || echo '?')"
