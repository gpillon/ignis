#!/usr/bin/env bash
# A/B a server configuration under the agent-swarm load (scripts/agent-swarm.py).
#
#   scripts/swarm-ab.sh "RETAINED_SLOTS=8" "RETAINED_SLOTS=2"
#   SWARM_ARGS="--agents 8 --turns 10" scripts/swarm-ab.sh "" "KV_HOST_POOL_BYTES=16G"
#
# Each argument is one leg: the make knobs that leg starts the server with,
# on top of the defaults below. Per leg: `make start` (the last build of this
# tree, behind make's own GPU guard), a /metrics scrape, the swarm, a second
# scrape, `make stop`. Everything lands in $OUT/<n>-<leg>/, the server's log
# included (serve/ignis-server.log), and the legs are then reported side by
# side.
#
# The server binds 127.0.0.1:8100 and its metrics 127.0.0.1:9564 so it never
# collides with a server someone is running on the usual ports -- but the
# card fits one model: make's GPU guard refuses a start while another
# process holds it, and this script stops at the first leg that fails to
# start. Run from bash, never through a PowerShell pipeline (make start's
# daemon inherits the pipe).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

[ $# -ge 1 ] || { sed -n '2,20p' "$0"; exit 2; }

ARTIFACT=${ARTIFACT:-$( [ -f models/qwen3_8_27b_nvfp4full-v2.ninfer ] && echo ./models/qwen3_8_27b_nvfp4full-v2.ninfer \
  || echo F:/ai/opencode/inference/models/qwen3_8_27b_nvfp4full-v2.ninfer )}
BIND=${BIND:-127.0.0.1:8100}
METRICS_BIND=${METRICS_BIND:-127.0.0.1:9564}
OUT=${OUT:-.scratch/swarm-ab/$(date +%Y%m%d-%H%M%S)}
SWARM_ARGS=${SWARM_ARGS:-}
PYTHON=${PYTHON:-python}

common=(UI=0 METRICS=1 LOG_LEVEL=debug "ARTIFACT=$ARTIFACT" "BIND=$BIND" "METRICS_BIND=$METRICS_BIND")
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit"
echo "swarm-ab -> $OUT"

dirs=()
leg_no=0
for leg in "$@"; do
  leg_no=$((leg_no + 1))
  name=$(printf '%s' "${leg:-defaults}" | sed 's/[ =\/]/_/g')
  dir="$OUT/$leg_no-$name"
  mkdir -p "$dir"
  knobs=("${common[@]}" "RUNTIME_DIR=$dir/serve")
  # shellcheck disable=SC2206 # a leg is whitespace-separated knobs by design
  [ -n "$leg" ] && knobs+=($leg)
  printf '%s\n' "${knobs[@]}" > "$dir/knobs"
  echo "== leg $leg_no: ${leg:-defaults}"

  make --no-print-directory config "${knobs[@]}" > "$dir/config.txt" 2>&1 || true
  if ! make --no-print-directory start "${knobs[@]}" > "$dir/start.out" 2>&1; then
    cat "$dir/start.out"
    echo "leg $leg_no did not start; stopping here" >&2
    exit 1
  fi
  stop_leg() { make --no-print-directory stop "${knobs[@]}" >> "$dir/start.out" 2>&1 || true; }
  trap stop_leg EXIT

  curl -fsS "http://$METRICS_BIND/metrics" > "$dir/metrics-before.prom" || echo "no metrics scrape" >&2
  status=0
  # shellcheck disable=SC2086 # SWARM_ARGS is a flag list
  "$PYTHON" -u scripts/agent-swarm.py run --out "$dir" --endpoint "http://$BIND" $SWARM_ARGS \
    > "$dir/swarm.out" 2>&1 || status=$?
  curl -fsS "http://$METRICS_BIND/metrics" > "$dir/metrics-after.prom" || echo "no metrics scrape" >&2
  tail -1 "$dir/swarm.out"

  stop_leg
  trap - EXIT
  dirs+=("$dir")
  [ "$status" -eq 0 ] || echo "leg $leg_no: the swarm reported failures (see $dir/swarm.out)" >&2
done

"$PYTHON" scripts/agent-swarm.py report "${dirs[@]}" | tee "$OUT/report.txt"
