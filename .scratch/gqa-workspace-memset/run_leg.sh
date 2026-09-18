#!/usr/bin/env bash
# One leg of the prefill A/B: start the given server binary with the chunk
# profiler on, send one long cold prompt, stop the server, leave the JSONL.
#
#   run_leg.sh <worktree> <label> <server.exe>
set -u

WT="$1"
LABEL="$2"
EXE="$3"
SP="C:/Users/Mille/AppData/Local/Temp/claude/F--ai-opencode-inference/8c7e337d-a059-4eda-8288-c3ea813f8fec/scratchpad"
JSONL="$SP/chunks-$LABEL.jsonl"
LOG="$SP/server-$LABEL.log"

rm -f "$JSONL"
cd "$WT" || exit 1

IGNIS_CHUNK_PROFILE="$JSONL" \
IGNIS_CHUNK_PROFILE_LAYERS=1 \
  "$EXE" \
  --bind 127.0.0.1:8000 \
  --artifact ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --kv-format hq-e8-2b --max-context 262144 --prefill-chunk 1024 \
  --kv-host-pool-bytes 8G --request-timeout 1800 \
  --spec dflash2 --draft-tokens 7 \
  --system-message-policy merge --developer-message-policy inplace \
  > "$LOG" 2>&1 &
PID=$!

for _ in $(seq 1 240); do
  if curl -s -m 2 http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "$LABEL: server died"; tail -20 "$LOG"; exit 1; fi
  sleep 1
done

START=$(date +%s%3N)
curl -s -m 600 -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary "@$SP/prompt.json" -o "$SP/reply-$LABEL.json"
END=$(date +%s%3N)
echo "$LABEL: request wall $((END - START)) ms, chunks $(wc -l < "$JSONL" 2>/dev/null || echo 0)"

kill "$PID" 2>/dev/null
for _ in $(seq 1 60); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
kill -9 "$PID" 2>/dev/null
sleep 2
