#!/usr/bin/env bash
# GitHub #90 / ADR 0017: the real-GPU A/B for Prometheus metrics.
#
# Legs, interleaved so drift cannot masquerade as a metrics effect:
#   warm  metrics off, g3 only, discarded (a cold machine is not a measurement)
#   off   metrics disabled                       (the baseline)
#   on    --metrics, never scraped
#   s15   --metrics, scraped every 15 s          (the ADR's budgeted workload)
#   s1    --metrics, scraped every second        (stress diagnostic, once)
# Each measured leg is one fresh process launch running the g3 cells
# (C=1 / C=4 throughput, ITL: the model thread's step timings as HTTP sees
# them) and the g4 trace replay (per-class TTFT and delivered tok/s), all
# under one --session. The engine is `make start`'s serving configuration
# with only METRICS toggled.
set -u

WT=F:/ai/opencode/.inference-qwen-worktrees/prometheus-metrics
OUT=${OUT:-$WT/.scratch/metrics-90}
# Extra `make start` knobs for every leg, e.g. MAKE_ARGS="SPEC=" (no speculation).
MAKE_ARGS=${MAKE_ARGS:-}
ART=F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer
CORPUS=F:/ai/q38/ninfer/bench/fixtures/bench_corpus.ids
TRACE=F:/ai/opencode/inference/bench/traces/g4-load-trace.jsonl
BENCH=$WT/target/x86_64-pc-windows-msvc/release/ignis-bench.exe
SESSION=${SESSION:-metrics90-$(date -u +%Y%m%dT%H%M%SZ)}
LEGS=${LEGS:-"warm:0 off:1 on:1 s15:1 off:2 on:2 s15:2 s1:1"}

cd "$WT" || exit 1
mkdir -p "$OUT"
echo "$SESSION" > "$OUT/SESSION.txt"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/ab.log"; }

wait_ready() {
  for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:8000/v1/models)
    [ "$code" = 200 ] && return 0
    sleep 5
  done
  return 1
}

scrape_loop() { # $1 = period seconds, $2 = leg name
  local ok=0 failed=0
  while [ -f "$OUT/.scraping" ]; do
    if curl -fs -o /dev/null --max-time 5 http://127.0.0.1:9464/metrics; then ok=$((ok+1)); else failed=$((failed+1)); fi
    echo "ok=$ok failed=$failed" > "$OUT/$2-scrapes.txt"
    sleep "$1"
  done
}

for leg in $LEGS; do
  kind=${leg%%:*}
  launch=${leg##*:}
  name=$kind-$launch
  metrics=1
  [ "$kind" = off ] || [ "$kind" = warm ] && metrics=0
  log "leg $name: start (METRICS=$metrics)"
  make --no-print-directory start ARTIFACT="$ART" METRICS=$metrics UI=0 $MAKE_ARGS > "$OUT/$name-start.log" 2>&1
  if ! wait_ready; then
    log "leg $name: server never became ready; stopping the A/B"
    make --no-print-directory stop >> "$OUT/$name-start.log" 2>&1
    exit 1
  fi
  cp .scratch/serve/*.log "$OUT/" 2>/dev/null

  nvidia-smi --format=csv,noheader -l 1 \
    --query-gpu=timestamp,utilization.gpu,power.draw,memory.used > "$OUT/$name-gpu.csv" &
  sampler=$!

  period=""
  [ "$kind" = s15 ] && period=15
  [ "$kind" = s1 ] && period=1
  if [ -n "$period" ]; then
    touch "$OUT/.scraping"
    scrape_loop "$period" "$name" &
    scraper=$!
  fi

  profile="make serving config ${MAKE_ARGS}, METRICS=$metrics${period:+, scraped every ${period}s}"
  log "leg $name: g3"
  "$BENCH" g3 --endpoint http://127.0.0.1:8000 --artifact "$ART" --corpus "$CORPUS" \
    --label "ignis-$kind" --profile "$profile" --session "$SESSION" \
    --out "$OUT/$name-g3.json" > "$OUT/$name-g3.log" 2>&1
  log "leg $name: g3 exit $?"
  if [ "$kind" != warm ]; then
    log "leg $name: g4"
    "$BENCH" g4 --endpoint http://127.0.0.1:8000 --artifact "$ART" --corpus "$CORPUS" \
      --trace "$TRACE" --label "ignis-$kind" --profile "$profile" --session "$SESSION" \
      --out "$OUT/$name-g4.json" > "$OUT/$name-g4.log" 2>&1
    log "leg $name: g4 exit $?"
  fi

  if [ -n "$period" ]; then
    rm -f "$OUT/.scraping"
    wait "$scraper" 2>/dev/null
  fi
  [ "$metrics" = 1 ] && curl -fs http://127.0.0.1:9464/metrics > "$OUT/$name-final-scrape.txt"
  kill "$sampler" 2>/dev/null
  taskkill //F //IM nvidia-smi.exe > /dev/null 2>&1
  cp .scratch/serve/*.log "$OUT/" 2>/dev/null
  for f in .scratch/serve/*.log; do cp "$f" "$OUT/$name-$(basename "$f")" 2>/dev/null; done
  make --no-print-directory stop >> "$OUT/$name-start.log" 2>&1
  log "leg $name: stopped"
  sleep 20
done
log "A/B done, session $SESSION"
