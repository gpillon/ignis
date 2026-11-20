#!/usr/bin/env bash
# The behavioural test of the GPU lock on Linux (GitHub #707): mk/linux/gpulock.py and the
# hooks around it, mk/linux/gpu.sh (status, guard, run), mk/linux/server.py (start, stop) and
# mk/os/linux.mk's `make run` and `make gpu-run`. It copies them into a temporary tree and runs
# them on stand-ins: sleeping processes as holders, a server that answers /ready, and an
# `nvidia-smi` and a `pgrep` first on PATH. It checks that the lock names its holder (pid,
# worktree, command), that a refusal is at once and a wait ends at its limit, both naming the
# holder, that a dead holder or a reused pid is free, that two takers of one stale record never
# both win, and that a start hands the lock to the server and a stop or a failed start frees
# it. Linux only (/proc). IGNIS_GPU_LOCK_DIR and TMPDIR point under a temporary directory it
# removes, so the machine's lock and the profile's marker are never read or written.

set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo" || exit 1

failed=0
fail() { printf '  FAIL: %s\n' "$*" >&2; failed=$((failed + 1)); }

python="$(command -v python3)" || { echo "gpu-lock-selftest: python3 not found" >&2; exit 1; }
tmp="$(mktemp -d)"
sleepers=()
cleanup() {
    local pid
    for pid in "${sleepers[@]}"; do { kill -KILL "$pid" && wait "$pid"; } 2>/dev/null; done
    [ -f "$tmp/run/ignis-server.pid" ] && "$python" "$tmp/repo/mk/linux/server.py" stop --pid "$tmp/run/ignis-server.pid" \
        --log "$tmp/run/ignis-server.log" --url "http://127.0.0.1:1" > /dev/null 2>&1
    rm -rf "$tmp"
}
trap cleanup EXIT

tree="$tmp/repo"
bin="$tmp/bin"
run="$tmp/run"
mkdir -p "$tree/mk/linux" "$tree/mk/os" "$bin" "$run" "$tmp/tmp" "$tmp/state"
cp mk/linux/gpu.sh mk/linux/gpulock.py mk/linux/server.py "$tree/mk/linux/"
cp mk/os/linux.mk "$tree/mk/os/"
export IGNIS_GPU_LOCK_DIR="$tmp/lock" TMPDIR="$tmp/tmp" STUB_DIR="$tmp/state"
unset IGNIS_GPU_CARD
lock="$IGNIS_GPU_LOCK_DIR/gpu-0.lock"
resolved="$("$python" -c 'import sys; sys.path.insert(0, sys.argv[1]); import gpulock; print(gpulock.lock_path())' "$tree/mk/linux")"
case "$resolved" in
    "$tmp"/*) ;;
    *) echo "gpu-lock-selftest: the lock resolves to $resolved, outside $tmp: not running" >&2; exit 1 ;;
esac

# The stand-ins: STUB_USED is the MiB nvidia-smi reports; pgrep finds no GPU process.
cat > "$bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
    *memory.used*) echo "${STUB_USED:-100}" ;;
    *) echo "stub-gpu" ;;
esac
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "$bin/pgrep"
cat > "$bin/stub-server" <<'EOF'
#!/usr/bin/env python3
# The server's stand-in: records its pid and the lock's record as it starts, then serves
# /ready, or with STUB_SERVER=exit-early exits with 7 before it serves.
import http.server, json, os, sys, time
state = os.environ["STUB_DIR"]
lock = os.path.join(os.environ["IGNIS_GPU_LOCK_DIR"], "gpu-0.lock")
with open(os.path.join(state, "server.json"), "w") as f:
    json.dump({"pid": os.getpid()}, f)
if os.environ.get("STUB_SERVER") == "exit-early":
    time.sleep(0.3)
    sys.exit(7)
host, port = sys.argv[sys.argv.index("--server-bind") + 1].rsplit(":", 1)
class Ready(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/ready" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.HTTPServer((host, int(port)), Ready).serve_forever(poll_interval=0.05)
EOF
chmod +x "$bin/nvidia-smi" "$bin/pgrep" "$bin/stub-server"
export PATH="$bin:$PATH"

lockctl() { "$python" "$tree/mk/linux/gpulock.py" "$@"; }
sleeper() { sleep 300 & sleepers+=("$!"); }
field() { "$python" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
holder_pid() { [ -f "$lock" ] && field "$lock" pid; }
start_of() { local line f; [ -n "$1" ] && IFS= read -r line < "/proc/$1/stat" && read -r -a f <<< "${line##*) }" && echo "${f[19]}"; }
# capture <command ...>: sets $code, $out (stdout and stderr) and $secs (whole seconds).
capture() {
    local began=$SECONDS
    out="$("$@" 2>&1)"
    code=$?
    secs=$((SECONDS - began))
}
expect_code() { [ "$code" = "$2" ] || fail "$1: exit $code, expected $2:
$out"; }
says() { grep -qF -- "$2" <<< "$out" || fail "$1: no '$2' in:
$out"; }
free_port() { "$python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# --- the lock ------------------------------------------------------------------------------

capture lockctl status
expect_code "status, no lock yet" 0
says "status, no lock yet" "GPU lock: free"

sleeper; first=$!
capture lockctl acquire --pid "$first" --wait 0 --command "gpu-profile --test a_test"
expect_code "acquire on a free card" 0
[ "$(holder_pid)" = "$first" ] || fail "acquire: the record names pid $(holder_pid), not $first"
[ "$(field "$lock" worktree)" = "$tree" ] || fail "acquire: the record's worktree is $(field "$lock" worktree)"
[ "$(field "$lock" start_time)" = "$(start_of "$first")" ] || fail "acquire: the record's start time is not the process's"

capture lockctl status
expect_code "status, held" 1
says "status, held" "by pid $first since "
says "status, held" "worktree $tree: gpu-profile --test a_test"
capture lockctl status --except "$first"
expect_code "status, held by the caller" 0

sleeper; second=$!
capture lockctl acquire --pid "$second" --wait 0 --command "make start"
expect_code "a second taker, no wait" 1
says "a second taker, no wait" "the card is held ($lock) by pid $first"
[ "$secs" -lt 5 ] && ! grep -qF "GPU lock: waiting" <<< "$out" || fail "a second taker, no wait: it waited (${secs}s):
$out"
[ "$(holder_pid)" = "$first" ] || fail "a second taker, no wait: the holder changed"

capture lockctl acquire --pid "$second" --wait 2 --command "gpu-run"
expect_code "a wait that runs out" 1
says "a wait that runs out" "GPU lock: waiting up to 2s for the card, held ($lock) by pid $first"
says "a wait that runs out" "the card is held ($lock) by pid $first"
[ "$secs" -ge 2 ] || fail "a wait that runs out: it gave up after ${secs}s"

capture lockctl release --pid "$second"
[ "$(holder_pid)" = "$first" ] || fail "a release by a process that does not hold it freed the lock"
capture lockctl acquire --pid "$first" --wait 0 --command "again"
expect_code "the holder takes it again" 0
[ "$(field "$lock" command)" = "gpu-profile --test a_test" ] || fail "taking it again rewrote the hold"

( sleep 1; lockctl release --pid "$first" ) &
releaser=$!
capture lockctl acquire --pid "$second" --wait 10 --command "gpu-run"
expect_code "a wait that ends with the release" 0
[ "$(holder_pid)" = "$second" ] || fail "a wait that ends with the release: the holder is $(holder_pid)"
[ "$secs" -lt 10 ] || fail "a wait that ends with the release: it waited out its limit"
wait "$releaser"

# A holder killed before it released: the record stays, and is free.
kill -KILL "$second"; wait "$second" 2>/dev/null
capture lockctl status
expect_code "a dead holder" 0
says "a dead holder" "GPU lock: free"
capture lockctl acquire --pid "$first" --wait 0 --command "after the dead one"
expect_code "taking a dead holder's lock" 0
[ "$(holder_pid)" = "$first" ] || fail "taking a dead holder's lock: the holder is $(holder_pid)"

# A pid that names another process now (another start time, another boot) is free.
boot="$(cat /proc/sys/kernel/random/boot_id)"
printf '{"pid": %s, "start_time": "1", "boot_id": "%s", "command": "old"}\n' "$first" "$boot" > "$lock"
capture lockctl status
expect_code "a reused pid" 0
printf '{"pid": %s, "start_time": "%s", "boot_id": "another-boot", "command": "old"}\n' "$first" "$(start_of "$first")" > "$lock"
capture lockctl status
expect_code "a pid of another boot" 0

# Five takers of one stale record at once: exactly one holds it.
printf '{"pid": 999999999, "start_time": "1", "boot_id": "%s"}\n' "$boot" > "$lock"
takers=()
racers=()
for i in 1 2 3 4 5; do sleeper; takers+=("$!"); done
for pid in "${takers[@]}"; do
    ( lockctl acquire --pid "$pid" --wait 0 --command "race" > /dev/null 2>&1; echo $? > "$tmp/race.$pid" ) &
    racers+=("$!")
done
wait "${racers[@]}"
won=0
for pid in "${takers[@]}"; do [ "$(cat "$tmp/race.$pid" 2>/dev/null)" = 0 ] && won=$((won + 1)); done
[ "$won" = 1 ] || fail "five takers of a stale record: $won of them hold it"

# One lock per card: card 1's is its own file.
capture env IGNIS_GPU_CARD=1 "$python" "$tree/mk/linux/gpulock.py" acquire --pid "$first" --wait 0 --command "card 1"
expect_code "another card" 0
[ -f "$IGNIS_GPU_LOCK_DIR/gpu-1.lock" ] || fail "another card: no gpu-1.lock"
rm -f "$IGNIS_GPU_LOCK_DIR/gpu-1.lock"

# --- gpu.sh: status, guard, run ------------------------------------------------------------

rm -f "$lock"
lockctl acquire --pid "$first" --wait 0 --command "gpu-profile" > /dev/null
capture bash "$tree/mk/linux/gpu.sh" status
says "gpu.sh status" "GPU lock: held ($lock) by pid $first"
capture bash "$tree/mk/linux/gpu.sh" guard 8192
expect_code "the guard, the lock held" 1
says "the guard, the lock held" "GPU guard: refused -- GPU lock: held ($lock) by pid $first"
[ "$secs" -lt 5 ] && ! grep -qF "GPU lock: waiting" <<< "$out" || fail "the guard, the lock held: it waited (${secs}s):
$out"
capture bash "$tree/mk/linux/gpu.sh" guard 8192 "$first"
expect_code "the guard, the lock held by its caller" 0
lockctl release --pid "$first"
capture bash "$tree/mk/linux/gpu.sh" guard 8192
expect_code "the guard, the lock free" 0
capture bash "$tree/mk/linux/gpu.sh" status
says "gpu.sh status, free" "GPU lock: free ($lock)"

# run: the command holds the lock under its own pid, its exit code is the run's, and the lock
# is free once it has exited.
capture env IGNIS_MK_GPU_CMD='echo "pid $$" > "$STUB_DIR/cmd"; "$PYTHON_SELFTEST" "$TREE/mk/linux/gpulock.py" status > "$STUB_DIR/cmd.status"; exit 7' \
    PYTHON_SELFTEST="$python" TREE="$tree" bash "$tree/mk/linux/gpu.sh" run 8192 5
expect_code "gpu.sh run" 7
own="$(sed -n 's/^pid //p' "$tmp/state/cmd" 2>/dev/null)"
[ -n "$own" ] && grep -qF "by pid $own since" "$tmp/state/cmd.status" ||
    fail "gpu.sh run: the command did not hold the lock under its pid ($own): $(cat "$tmp/state/cmd.status" 2>/dev/null)"
grep -qF ": gpu-run: echo" "$tmp/state/cmd.status" || fail "gpu.sh run: the hold does not name the command"
capture lockctl status
expect_code "gpu.sh run, after" 0
rm -f "$tmp/state/cmd"
lockctl acquire --pid "$first" --wait 0 --command "gpu-profile" > /dev/null
capture env IGNIS_MK_GPU_CMD='touch "$STUB_DIR/cmd"' bash "$tree/mk/linux/gpu.sh" run 8192 1
expect_code "gpu.sh run, held" 1
says "gpu.sh run, held" "the card is held ($lock) by pid $first"
[ ! -e "$tmp/state/cmd" ] || fail "gpu.sh run, held: the command ran"
lockctl release --pid "$first"
capture env STUB_USED=9000 IGNIS_MK_GPU_CMD='touch "$STUB_DIR/cmd"' bash "$tree/mk/linux/gpu.sh" run 8192 1
expect_code "gpu.sh run, the guard refuses" 1
says "gpu.sh run, the guard refuses" "GPU guard: refused -- 9000 MiB already resident"
[ ! -e "$tmp/state/cmd" ] || fail "gpu.sh run, the guard refuses: the command ran"
capture lockctl status
expect_code "gpu.sh run, after the guard refused" 0

# --- server.py: start hands the lock to the server, stop and a failed start free it ---------

port="$(free_port)"
url="http://127.0.0.1:$port"
server_ctl() { "$python" "$tree/mk/linux/server.py" "$1" --pid "$run/ignis-server.pid" --log "$run/ignis-server.log" \
    --url "$url" --timeout 10 "${@:2}"; }
capture server_ctl start --gpu-lock -- "$bin/stub-server" --server-bind "127.0.0.1:$port"
expect_code "start" 0
spid="$(field "$tmp/state/server.json" pid 2>/dev/null)"
[ -n "$spid" ] && [ "$(holder_pid)" = "$spid" ] || fail "start: the lock names pid $(holder_pid), not the server's ($spid)"
[ "$(field "$lock" start_time)" = "$(start_of "$spid")" ] || fail "start: the record's start time is not the server's"
capture server_ctl stop
expect_code "stop" 0
[ ! -e "$lock" ] || fail "stop: the lock is still there: $(cat "$lock")"

rm -f "$tmp/state/server.json"
lockctl acquire --pid "$first" --wait 0 --command "gpu-profile" > /dev/null
capture server_ctl start --gpu-lock -- "$bin/stub-server" --server-bind "127.0.0.1:$port"
expect_code "start, the lock held" 1
says "start, the lock held" "the card is held ($lock) by pid $first"
[ ! -e "$tmp/state/server.json" ] || fail "start, the lock held: the server was started"
[ "$secs" -lt 5 ] && ! grep -qF "GPU lock: waiting" <<< "$out" || fail "start, the lock held: it waited (${secs}s):
$out"
lockctl release --pid "$first"

capture env STUB_SERVER=exit-early "$python" "$tree/mk/linux/server.py" start --pid "$run/ignis-server.pid" \
    --log "$run/ignis-server.log" --url "$url" --timeout 10 --gpu-lock -- "$bin/stub-server" --server-bind "127.0.0.1:$port"
expect_code "start, the server exits early" 1
[ ! -e "$lock" ] || fail "start, the server exits early: the lock is still there: $(cat "$lock")"

capture server_ctl start -- "$bin/stub-server" --server-bind "127.0.0.1:$port"
expect_code "start without the lock" 0
[ ! -e "$lock" ] || fail "start without --gpu-lock took the lock"
server_ctl stop > /dev/null 2>&1

# --- the make hooks: run and gpu-run --------------------------------------------------------

cat > "$tree/Makefile" <<'EOF'
SERVER_BIN := $(SELFTEST)/bin/stub-server
SERVER_FLAGS = --server-bind 127.0.0.1:$(PORT)
SERVER_ENV = IGNIS_LOG_LEVEL=debug
GPU_THRESHOLD_MIB := 8192
GPU_LOCK_WAIT ?= 5
PYTHON := python3
include mk/os/linux.mk
.RECIPEPREFIX := >
.PHONY: run gpu-run
run:
> @$(SERVER_RUN)
gpu-run: export IGNIS_MK_GPU_CMD = $(value CMD)
gpu-run:
> @$(GPU_RUN)
EOF
make_in() { make -C "$tree" --no-print-directory SELFTEST="$tmp" PORT="$port" "$@"; }

rm -f "$tmp/state/server.json"
make_in run > "$tmp/run.out" 2>&1 &
maker=$!
for _ in $(seq 100); do [ -f "$tmp/state/server.json" ] && break; sleep 0.1; done
spid="$(field "$tmp/state/server.json" pid 2>/dev/null)"
[ -n "$spid" ] && [ "$(holder_pid)" = "$spid" ] || fail "make run: the lock names pid $(holder_pid), not the server's ($spid)"
[ "$(field "$lock" command 2>/dev/null)" = "make run" ] || fail "make run: the hold does not say make run: $(cat "$lock" 2>/dev/null)"
capture make_in run
expect_code "make run, a second one" 2
says "make run, a second one" "the card is held ($lock) by pid $spid"
kill -TERM "$spid" 2>/dev/null; wait "$maker" 2>/dev/null
capture lockctl status
expect_code "make run, after the server exited" 0

capture make_in gpu-run CMD='echo "under the lock" > "$STUB_DIR/gpu-run"; exit 4'
expect_code "make gpu-run" 2
grep -qF "under the lock" "$tmp/state/gpu-run" 2>/dev/null || fail "make gpu-run: the command did not run"
lockctl acquire --pid "$first" --wait 0 --command "gpu-profile" > /dev/null
rm -f "$tmp/state/gpu-run"
capture make_in gpu-run GPU_LOCK_WAIT=1 CMD='touch "$STUB_DIR/gpu-run"'
expect_code "make gpu-run, held" 2
says "make gpu-run, held" "the card is held ($lock) by pid $first"
[ ! -e "$tmp/state/gpu-run" ] || fail "make gpu-run, held: the command ran"
lockctl release --pid "$first"

if [ "$failed" -ne 0 ]; then
    printf 'gpu-lock-selftest: %d failure(s)\n' "$failed" >&2
    exit 1
fi
echo "gpu-lock-selftest: ok"
